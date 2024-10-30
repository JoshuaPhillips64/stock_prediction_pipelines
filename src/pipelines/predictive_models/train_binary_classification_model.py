# %%
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE  # For handling class imbalance
import plotly.graph_objects as go
import logging
import warnings
from database_functions import create_engine_from_url, fetch_dataframe, upsert_df
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, S3_BUCKET_NAME
from aws_functions import s3_upload_file
from datetime import datetime, timedelta, date
import json
import pickle
import os
import tempfile

from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %%

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def lambda_handler(event, context):
    """
    AWS Lambda handler function to initiate XGBoost classification model training and prediction.
    """
    # Call the train_classification_model function
    success, output = train_classification_model(event, context)

    if success:
        return {
            'statusCode': 200,
            'body': json.dumps(output,default=json_serial)
        }
    else:
        return {
            'statusCode': 500,
            'body': json.dumps('Prediction failed.')
        }

def check_stationarity(df, column='log_return_1', alpha=0.05):
    logging.info("Checking stationarity of the time series...")
    result = adfuller(df[column].dropna())
    p_value = result[1]
    logging.info(f"ADF Statistic: {result[0]}, p-value: {p_value}")
    return p_value < alpha

def preprocess_data(df, prediction_horizon):
    logging.info("Starting data preprocessing with stationarity check...")
    df = df.copy()
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['close'], inplace=True)
    df.fillna(method='ffill', inplace=True)

    # Calculate log return over prediction_horizon days
    df['log_return_future'] = np.log(df['close'].shift(-prediction_horizon) / df['close'])
    df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
    # Store future date by shifting the 'date' column
    # Drop rows where future data is not available
    df.dropna(subset=['log_return_future'], inplace=True)

    # Convert to binary target
    df['target'] = (df['log_return_future'] > 0).astype(int)  # 1 if return is positive, else 0

    # Perform differencing if the series is non-stationary
    if not check_stationarity(df, column='log_return_1'):
        logging.info("Time series is non-stationary, applying differencing...")
        df['log_return_1'] = df['log_return_1'].diff()
        df.dropna(subset=['log_return_1'], inplace=True)

    logging.info(f"Data preprocessing completed. Remaining rows after preprocessing: {df.shape[0]}")
    return df

# Feature Engineering
def engineer_additional_features(df):
    logging.info("Starting feature engineering...")

    # Calculate technical indicators
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_100'] = df['close'].rolling(window=100).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # Momentum indicators
    df['momentum_30'] = df['close'].pct_change(periods=30)
    df['momentum_60'] = df['close'].pct_change(periods=60)

    # Calculate MACD
    df = calculate_macd(df)

    # Calculate ADX
    df = calculate_adx(df)

    # Calculate RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / roll_down
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    # Bollinger Bands
    df['middle_band'] = df['close'].rolling(window=20).mean()
    df['std_dev'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)
    df.drop(columns=['middle_band', 'std_dev'], inplace=True)

    # Rolling calculations
    df['rolling_volatility_60'] = df['log_return_1'].rolling(window=60).std()
    df['corr_sp500_60'] = df['log_return_1'].rolling(window=60).corr(df['sp500_return'].shift(1))
    df['corr_nasdaq_60'] = df['log_return_1'].rolling(window=60).corr(df['nasdaq_return'].shift(1))

    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['%K'] = (df['close'] - low_14) / (high_14 - low_14) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()

    # Commodity Channel Index
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad)

    df.dropna(inplace=True)
    logging.info(f"Feature engineering completed. Remaining rows after feature engineering: {df.shape[0]}")
    return df

def calculate_macd(df, price_column='close'):
    logging.info("Calculating MACD and related features...")
    df['ema_12'] = df[price_column].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df[price_column].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df.drop(columns=['ema_12', 'ema_26'], inplace=True)
    logging.info("MACD calculation completed.")
    return df

def calculate_adx(df, high_column='high', low_column='low', close_column='close', period=14):
    logging.info("Calculating ADX...")

    # Calculate True Range (TR)
    df['tr'] = df[[high_column, close_column]].max(axis=1) - df[[low_column, close_column]].min(axis=1)

    # Calculate directional movements
    df['up_move'] = df[high_column] - df[high_column].shift(1)
    df['down_move'] = df[low_column].shift(1) - df[low_column]

    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    # Calculate Smoothed True Range, +DM, -DM
    atr = df['tr'].rolling(window=period).mean()
    plus_di = 100 * (df['plus_dm'].rolling(window=period).mean() / atr)
    minus_di = 100 * (df['minus_dm'].rolling(window=period).mean() / atr)

    # Calculate DX and ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(window=period).mean()

    # Drop intermediate columns
    df.drop(columns=['tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm'], inplace=True)

    logging.info("ADX calculation completed.")
    return df

# Feature Selection
def select_features(df, feature_set='advanced'):
    logging.info("Starting feature selection...")

    # Define feature categories
    basic_features = [
        'volume',
        'open',
        'high',
        'low',
        'close',
        'sp500_return',
        'nasdaq_return',
        'gdp_growth',
        'inflation_rate',
        'unemployment_rate',
        'gdp_growth_lag_1', 'gdp_growth_lag_7', 'gdp_growth_lag_30',
        'inflation_rate_lag_1', 'inflation_rate_lag_7',
        'inflation_rate_lag_30',
        'unemployment_rate_lag_1', 'unemployment_rate_lag_7', 'unemployment_rate_lag_30',
        'macd_hist', 'adx', 'macd', 'rsi', 'upper_band', 'lower_band',
        'macd_signal',
        'rolling_volatility_60',
        'corr_sp500_60', 'corr_nasdaq_60',
        'sma_50', 'sma_100', 'ema_50',
        '%K', '%D', 'cci',
        'momentum_30', 'momentum_60'
    ]

    advanced_features = [
        'sentiment_score',
        'implied_volatility',
        'market_capitalization',
        'pe_ratio',
        'dividend_yield',
        'beta',
        'put_call_ratio',
        'sector_performance',
        'sector_performance_lag_1', 'sector_performance_lag_7', 'sector_performance_lag_30',
        'sp500_return_lag_1', 'sp500_return_lag_7', 'sp500_return_lag_30',
        'nasdaq_return_lag_1', 'nasdaq_return_lag_7', 'nasdaq_return_lag_30',
        'sentiment_score_lag_1'
    ]

    if feature_set == 'basic':
        features = basic_features
    elif feature_set == 'advanced':
        features = basic_features + advanced_features
    else:
        raise ValueError(f"Invalid feature_set: {feature_set}")

    # Only include features that are in df columns
    features = [feature for feature in features if feature in df.columns]
    selected_features = df[features].copy()

    # Keep the target variable separate
    target = df['target']

    # Ensure there are no NaNs
    data = pd.concat([selected_features, target], axis=1).dropna()
    selected_features = data[features]
    target = data['target']

    # Optional: Feature Selection based on correlation
    # Remove features with high correlation to reduce multicollinearity
    corr_matrix = selected_features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    if to_drop:
        logging.info(f"Dropping highly correlated features: {to_drop}")
        selected_features.drop(columns=to_drop, inplace=True)
        features = [f for f in features if f not in to_drop]

    logging.info(f"Feature selection completed. Selected features: {len(features)}, Remaining rows: {data.shape[0]}")
    return selected_features, target, features

# Prepare Data
def prepare_data(X_train, X_test, y):
    # Convert any categorical features
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = pd.Categorical(X_train[col]).codes
            X_test[col] = pd.Categorical(X_test[col]).codes

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    logging.info(f"Data scaling completed. Shape of scaled features: {X_train_scaled.shape}")
    return X_train_scaled, X_test_scaled, y, scaler

# Feature Importance Analysis
def feature_importance_analysis(X_train, y_train):
    logging.info("Starting feature importance analysis with XGBoostClassifier...")
    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': xgb.feature_importances_})
    importance_df = importance_df.sort_values('importance', ascending=False)

    # Plot the feature importance using Plotly
    fig = go.Figure([go.Bar(x=importance_df['feature'], y=importance_df['importance'])])
    fig.update_layout(title='Feature Importance', xaxis_title='Features', yaxis_title='Importance', xaxis_tickangle=-45)
    # Commenting out fig.show() for Lambda compatibility
    #fig.show()

    logging.info("Feature importance analysis completed.")
    return importance_df

# Time Series Cross-Validation and Hyperparameter Tuning for Classification
def time_series_cv_classification(X, y, hyperparameter_tuning):
    logging.info("Starting time series cross-validation and hyperparameter tuning with XGBoostClassifier...")

    if hyperparameter_tuning == 'LOW':
        params = {
            'max_depth': [3],
            'n_estimators': [100],
            'learning_rate': [0.1]
        }
    elif hyperparameter_tuning == 'MEDIUM':
        params = {
            'max_depth': [3, 5],
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.05]
        }
    elif hyperparameter_tuning == 'HIGH':
        params = {
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.1, 0.05, 0.01]
        }
    else:
        raise ValueError(f"Invalid hyperparameter_tuning: {hyperparameter_tuning}")

    tscv = TimeSeriesSplit(n_splits=3)
    best_score = 0
    best_params = {}

    for max_depth in params['max_depth']:
        for n_estimators in params['n_estimators']:
            for learning_rate in params['learning_rate']:
                scores = []
                for train_index, test_index in tscv.split(X):
                    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
                    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

                    # Handle class imbalance with SMOTE
                    smote = SMOTE(random_state=42)
                    X_train_fold_res, y_train_fold_res = smote.fit_resample(X_train_fold, y_train_fold)

                    # Calculate scale_pos_weight to handle class imbalance
                    scale_pos_weight = calculate_class_weight(y_train_fold_res)

                    clf = XGBClassifier(
                        max_depth=max_depth,
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        scale_pos_weight=scale_pos_weight,  # Class weight balancing
                        use_label_encoder=False,
                        eval_metric='logloss',
                        random_state=42
                    )
                    clf.fit(X_train_fold_res, y_train_fold_res)
                    y_pred_fold = clf.predict(X_test_fold)
                    score = f1_score(y_test_fold, y_pred_fold)
                    scores.append(score)

                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {'max_depth': max_depth, 'n_estimators': n_estimators, 'learning_rate': learning_rate}
                    logging.info(f"New best params: {best_params} with F1-Score: {best_score}")

    logging.info("Time series cross-validation and hyperparameter tuning completed.")
    logging.info(f"Best parameters: {best_params}")
    return best_params

def calculate_class_weight(y_train):
    """Calculate scale_pos_weight for XGBoost based on class imbalance in the training data."""
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    if positive_count == 0:
        return 1  # Prevent division by zero
    return negative_count / positive_count  # Ratio of majority to minority class

# Train Final Model
def train_final_model_classifier(X_train, y_train, best_params):
    logging.info("Training final classification model with XGBoostClassifier...")

    # Calculate scale_pos_weight for handling class imbalance
    scale_pos_weight = calculate_class_weight(y_train)

    clf = XGBClassifier(
        max_depth=best_params['max_depth'],
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        scale_pos_weight=scale_pos_weight,  # Class weight balancing
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    clf.fit(X_train, y_train)
    logging.info("Final classification model training completed.")
    return clf

# Evaluation Metrics
def evaluate_model_classifier(model, X_test, y_test):
    logging.info("Evaluating classification model...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    conf_matrix = confusion_matrix(y_test, y_pred)

    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'Precision: {precision:.4f}')
    logging.info(f'Recall: {recall:.4f}')
    logging.info(f'F1-Score: {f1:.4f}')
    logging.info(f'ROC AUC: {roc_auc:.4f}')
    logging.info(f'Confusion Matrix:\n{conf_matrix}')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix.tolist()  # Convert to list for JSON serialization
    }

    logging.info("Classification model evaluation completed.")
    return metrics

# Predict Future Class
def predict_future_class(original_data, final_model, scaler, feature_list, prediction_horizon, feature_set):
    logging.info("Predicting future class...")

    try:
        # Get the last date in the original data
        last_date = original_data['date'].max()

        # Prepare the data for prediction
        processed_df = preprocess_data(original_data.copy(), prediction_horizon)
        engineered_df = engineer_additional_features(processed_df)

        # Select features
        logging.info("Starting feature selection...")
        X_future, _, _ = select_features(engineered_df, feature_set=feature_set)

        # Ensure we have the correct features
        logging.info(f"Features before selection: {X_future.columns.tolist()}")
        X_future = X_future[feature_list]

        # Scale the features
        logging.info("Starting feature scaling...")
        X_future_scaled = pd.DataFrame(scaler.transform(X_future),
                                       index=X_future.index,
                                       columns=X_future.columns)

        # Get the last row for prediction
        X_pred = X_future_scaled.iloc[[-1]]

        logging.info(f"Exogenous variables shape for prediction: {X_pred.shape}")

        # Make the prediction using the classifier
        y_pred_future = final_model.predict(X_pred)
        y_proba_future = final_model.predict_proba(X_pred)[:, 1]

        logging.info(f"Predicted class: {y_pred_future[0]} with probability: {y_proba_future[0]}")

        # Collect the last known price
        last_known_price = original_data['close'].values[-1]

        # Create a dataframe with the result
        future_date = last_date + pd.Timedelta(days=prediction_horizon)
        results_df = pd.DataFrame({
            'date': [future_date],
            'predicted_movement': [int(y_pred_future[0])],
            'actual_price': [last_known_price],
            'prediction_probability': [float(y_proba_future[0])]
        })

        logging.info("Future class prediction completed.")
        return results_df

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return None

# Visualize Original Data, Test Projections, and Future Prediction
def plot_full_stock_prediction_classifier(original_data, y_test, y_pred, future_prediction, test_start_date):
    logging.info("Plotting full stock prediction...")

    # Convert test_start_date and future_prediction['date'] to datetime objects (if necessary)
    test_start_date = pd.to_datetime(test_start_date)
    future_date = pd.to_datetime(future_prediction['date'].iloc[0])

    # Ensure y_pred is a pandas Series with the same index as y_test
    y_pred = pd.Series(y_pred, index=y_test.index)

    # Create the figure
    fig = go.Figure()

    # Plot the actual stock prices
    fig.add_trace(go.Scatter(x=original_data['date'], y=original_data['close'], mode='lines', name='Actual Price'))

    # Plot the predicted movements on the test set
    increases = y_pred == 1
    decreases = y_pred == 0

    fig.add_trace(go.Scatter(
        x=y_test.index[increases],
        y=original_data.set_index('date').loc[y_test.index[increases], 'close'],
        mode='markers',
        marker=dict(color='green', symbol='triangle-up', size=10),
        name='Predicted Increase',
    ))

    fig.add_trace(go.Scatter(
        x=y_test.index[decreases],
        y=original_data.set_index('date').loc[y_test.index[decreases], 'close'],
        mode='markers',
        marker=dict(color='red', symbol='triangle-down', size=10),
        name='Predicted Decrease',
    ))

    # Plot the predicted future price
    fig.add_trace(go.Scatter(
        x=future_prediction['date'],
        y=future_prediction['actual_price'],
        mode='markers+lines',
        name='Actual Price',
        marker=dict(symbol='star', size=12, color='blue')
    ))

    # Manually add vertical lines by plotting as scatter points
    fig.add_trace(go.Scatter(
        x=[test_start_date, test_start_date],
        y=[original_data['close'].min(), original_data['close'].max()],
        mode='lines',
        name='Test Set Start',
        line=dict(color='orange', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=[future_date, future_date],
        y=[original_data['close'].min(), original_data['close'].max()],
        mode='lines',
        name='Future Prediction',
        line=dict(color='green', dash='dash')
    ))

    # Update layout for the plot
    fig.update_layout(
        title='Stock Price Prediction: Actual, Predicted Movements, and Future Price',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Commenting out fig.show() for Lambda compatibility
    #fig.show()
    logging.info("Full stock prediction plot completed.")

# %%
def train_classification_model(event, context):
    # Parse input parameters from event payload
    try:
        body = json.loads(event['body'])
        model_key = body.get('model_key')
        stock_symbol = body.get('stock_symbol')
        input_date = body.get('input_date')
        hyperparameter_tuning = body.get('hyperparameter_tuning', 'MEDIUM')
        feature_set = body.get('feature_set', 'advanced')
        lookback_period = body.get('lookback_period', 720)
        prediction_horizon = body.get('prediction_horizon', 30)
    except Exception as e:
        logging.error(f"Error parsing input parameters: {e}")
        return False

    # Compute start_date and end_date
    try:
        input_date_dt = datetime.strptime(input_date, '%Y-%m-%d')
        prediction_date = input_date_dt + timedelta(days=prediction_horizon)
    except ValueError:
        logging.error("Incorrect date format. Expected YYYY-MM-DD.")
        return False

    start_date_dt = input_date_dt - timedelta(days=lookback_period)
    start_date = start_date_dt.strftime('%Y-%m-%d')
    end_date = input_date

    if feature_set == 'advanced':
        stock_table = 'enriched_stock_data'
    else:
        stock_table = 'basic_stock_data'

    query = f"""
    SELECT * FROM {stock_table}
    WHERE symbol = '{stock_symbol}'
    and date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date
    """

    # Database and stock selection
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine_from_url(db_url)

    # Fetch data
    original_data = fetch_dataframe(engine, query)
    original_data['date'] = pd.to_datetime(original_data['date'])

    if original_data.empty:
        logging.error("No data fetched for the given parameters.")
        return False

    # Data preprocessing and feature engineering
    stock_data = preprocess_data(original_data, prediction_horizon)
    processed_data = engineer_additional_features(stock_data)
    X, y, feature_list = select_features(processed_data, feature_set=feature_set)

    # Train-test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Handle class imbalance with SMOTE on training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Prepare data (scaling)
    X_train_scaled, X_test_scaled, y_train_res, scaler = prepare_data(X_train_res, X_test, y_train_res)

    # Feature analysis overview (now only on training data)
    importance_df = feature_importance_analysis(X_train_scaled, y_train_res)

    # Model training and evaluation
    best_params = time_series_cv_classification(X_train_scaled, y_train_res, hyperparameter_tuning)
    final_model = train_final_model_classifier(X_train_scaled, y_train_res, best_params)

    # Save the model with model_key
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file_name = f'xgb_classifier_model_{model_key}.pkl'  # Updated filename
        temp_model_location = os.path.join(tmpdir, model_file_name)

        s3_bucket_name = "trained-models-stock-prediction"
        s3_folder = "classification"

        # Save the model as a pickle in the temporary directory
        with open(temp_model_location, 'wb') as f:
            pickle.dump({
                'model': final_model,
                'scaler': scaler,
                'feature_list': feature_list,
                'best_params': best_params,
                'hyperparameter_tuning': hyperparameter_tuning,
                'feature_set': feature_set,
                'prediction_horizon': prediction_horizon,
                'input_date': input_date
            }, f)

        logging.info(f"Model training completed and saved temporarily at {temp_model_location}")

        # Upload the pickle file to S3
        s3_upload_file(temp_model_location, s3_bucket_name, model_file_name, folder_name=s3_folder)
        logging.info(f"Model saved to S3 at bucket: {s3_bucket_name}, folder: {s3_folder}")

    # Evaluate model
    metrics = evaluate_model_classifier(final_model, X_test_scaled, y_test)

    # Get raw predictions for visualization
    y_pred_raw = final_model.predict(X_test_scaled)

    # When predicting future class
    future_prediction = predict_future_class(original_data, final_model, scaler, feature_list, prediction_horizon,
                                             feature_set)

    if future_prediction is not None and not future_prediction.empty:
        # Limit y_test and X_test_scaled to dates where date + prediction_horizon <= input_date_dt
        valid_dates = [date for date in y_test.index if date + pd.Timedelta(days=prediction_horizon) <= input_date_dt]
        y_test = y_test.loc[valid_dates]
        X_test_scaled = X_test_scaled.loc[valid_dates]

        # Ensure 'date' is the index and sorted
        original_data.set_index('date', inplace=True)
        original_data.sort_index(inplace=True)

        # Build predictions JSON dictionary
        predictions_json = {}
        for date in y_test.index:
            # Get actual movement
            actual_current_price = original_data.loc[date, 'close']
            future_date = date + pd.Timedelta(days=prediction_horizon)

            # Find the closest trading day on or after future_date
            future_dates = original_data.index[original_data.index >= future_date]
            if not future_dates.empty:
                closest_future_date = future_dates[0]
                actual_future_price = original_data.loc[closest_future_date, 'close']
                actual_movement = 1 if actual_future_price > actual_current_price else 0
            else:
                actual_movement = None  # No future date available

            # Get predicted movement
            predicted_movement = int(final_model.predict(X_test_scaled.loc[[date]])[0])
            actual_price = float(actual_current_price)

            predictions_json[str(date.date())] = {
                'actual_movement': actual_movement,
                'predicted_movement': predicted_movement,
                'actual_price': actual_price
            }

        # Add the future price prediction
        future_date = future_prediction['date'].iloc[0].strftime('%Y-%m-%d')
        predictions_json[future_date] = {
            'actual_movement': None,  # No actual movement for future dates
            'predicted_movement': int(future_prediction['predicted_movement'].iloc[0]),
            'prediction_probability': future_prediction['prediction_probability'].iloc[0]
        }

        # Convert to JSON
        predictions_json_str = json.dumps(predictions_json, default=str)

        # Plotting the actual and predicted movements
        #plot_full_stock_prediction_classifier(
        #    original_data,
        #    y_test,
        #    y_pred_raw,  # Use raw predictions here
        #    future_prediction,
        #    test_start_date=X_test.index.min()
        #)

        # Prepare output data
        output = {
            'model_key': model_key,
            'symbol': stock_symbol,
            'prediction_date': prediction_date,
            'prediction_explanation': 'Binary Clssification Based on XGBoostClassifier with feature engineering and SMOTE',
            'prediction_accuracy': metrics['accuracy'],
            'prediction_precision': metrics['precision'],
            'prediction_recall': metrics['recall'],
            'prediction_f1_score': metrics['f1_score'],
            'prediction_roc_auc': metrics['roc_auc'],
            'confusion_matrix': json.dumps(metrics['confusion_matrix']),
            'feature_importance': json.dumps(importance_df.to_dict()),
            'model_parameters': json.dumps({
                'hyperparameter_tuning': hyperparameter_tuning,
                'feature_set': feature_set,
                'prediction_horizon': prediction_horizon,
                'input_date': input_date,
                'lookback_period': lookback_period
            }),
            'predicted_movement': 'Down' if future_prediction['predicted_movement'].iloc[0] == 0 else 'Up',
            'predicted_price': float(0),
            'prediction_probability': float(future_prediction['prediction_probability'].iloc[0]),
            'last_known_price': float(original_data['close'].iloc[-1]),
            'predictions_json': predictions_json_str,
            'model_location': f's3://{s3_bucket_name}/{s3_folder}/{model_file_name}',
            'date_created': datetime.now()
        }

        # Log data to the database (keep this part)
        log_data = pd.DataFrame({k: [v] for k, v in output.items()})
        upsert_df(log_data, 'trained_models_binary', 'model_key', engine,
                  json_columns=['model_parameters', 'feature_importance', 'confusion_matrix', 'predictions_json'],
                  auto_match_schema='public'
                  )
        logging.info("Model saved and data logged to the database.")

        return True, output
    else:
        logging.error("Failed to generate prediction. No data logged to the database.")
        return False, None

#%% Example payload for the Lambda event
def main():
    #%%
    # Sample event payload with sufficient data coverage
    event = {
        'body': json.dumps({
            'model_key': 'sample_model_002',
            'stock_symbol': 'PG',
            'input_date': '2024-10-01',
            'hyperparameter_tuning': 'LOW',
            'feature_set': 'basic',
            'lookback_period': 2000,
            'prediction_horizon': 30
        })
    }
    #%%
    # Mock context object (can be empty or contain necessary attributes)
    context = {}

    # Invoke the lambda_handler
    response = lambda_handler(event, context)

    # Print the response
    print("Lambda Response:")
    print(json.dumps(response, indent=4))
    #%%

#main()