import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RandomizedSearchCV
import logging
import warnings
import json
import pickle
import os
import tempfile
from datetime import datetime
from database_functions import create_engine_from_url, fetch_dataframe, upsert_df
from aws_functions import s3_upload_file
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
import plotly.graph_objs as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_data(df):
    logging.info("Starting data preprocessing...")
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Calculate future price change (30 days ahead)
    df['future_price'] = df['close'].shift(-30)
    df['price_change'] = df['future_price'] - df['close']

    # Create binary target: 1 if price goes up, 0 if it goes down or stays the same
    df['target'] = (df['price_change'] > 0).astype(int)

    df.dropna(subset=['target'], inplace=True)

    # Handle remaining NaN values
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column].fillna(df[column].median(), inplace=True)
        elif df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)

    logging.info(f"Number of NaN values after preprocessing: {df.isna().sum().sum()}")
    logging.info("Data preprocessing completed.")
    return df


def engineer_additional_features(df):
    logging.info("Starting feature engineering...")

    shift_period = 30
    feature_columns = df.columns.difference(['target'])
    df[feature_columns] = df[feature_columns].shift(shift_period)

    technical_indicators = ['macd', 'macd_signal', 'macd_hist', 'rsi', 'upper_band', 'lower_band', 'adx',
                            'implied_volatility', 'other_indicator']
    df.drop(columns=[col for col in technical_indicators if col in df.columns], inplace=True)

    df['close_shifted'] = df['close'].shift(shift_period)
    df['high_shifted'] = df['high'].shift(shift_period)
    df['low_shifted'] = df['low'].shift(shift_period)

    df['sma_50'] = df['close_shifted'].rolling(window=50).mean()
    df['sma_100'] = df['close_shifted'].rolling(window=100).mean()
    df['ema_50'] = df['close_shifted'].ewm(span=50, adjust=False).mean()

    df['momentum_30'] = df['close_shifted'].shift(30) / df['close_shifted'].shift(60) - 1
    df['momentum_60'] = df['close_shifted'].shift(60) / df['close_shifted'].shift(90) - 1

    df = calculate_macd(df, price_column='close_shifted')
    df = calculate_adx(df, high_column='high_shifted', low_column='low_shifted', close_column='close_shifted')

    delta = df['close_shifted'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / roll_down
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    df['middle_band'] = df['close_shifted'].rolling(window=20).mean()
    df['std_dev'] = df['close_shifted'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)
    df.drop(columns=['middle_band', 'std_dev'], inplace=True)

    low_14 = df['low_shifted'].rolling(window=14).min()
    high_14 = df['high_shifted'].rolling(window=14).max()
    df['%K'] = (df['close_shifted'] - low_14) / (high_14 - low_14) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()

    typical_price = (df['high_shifted'] + df['low_shifted'] + df['close_shifted']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad)

    df['day_of_week'] = df.index.shift(shift_period, freq='D').dayofweek
    df['month'] = df.index.shift(shift_period, freq='D').month

    df.drop(columns=['close_shifted', 'high_shifted', 'low_shifted'], inplace=True)
    df.dropna(inplace=True)
    logging.info("Feature engineering completed.")
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

    df['tr'] = df[[high_column, close_column]].max(axis=1) - df[[low_column, close_column]].min(axis=1)
    df['up_move'] = df[high_column] - df[high_column].shift(1)
    df['down_move'] = df[low_column].shift(1) - df[low_column]

    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    atr = df['tr'].rolling(window=period).mean()
    plus_di = 100 * (df['plus_dm'].rolling(window=period).mean() / atr)
    minus_di = 100 * (df['minus_dm'].rolling(window=period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(window=period).mean()

    df.drop(columns=['tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm'], inplace=True)

    logging.info("ADX calculation completed.")
    return df


def select_features(df):
    logging.info("Selecting features...")

    # Remove non-numeric columns and columns we don't want to use as features
    columns_to_drop = ['symbol', 'target', 'future_price', 'price_change', 'next_earnings_date', 'last_earnings_date']
    feature_columns = df.select_dtypes(include=['int64', 'float64']).columns.drop(columns_to_drop, errors='ignore')

    # Ensure we have features to work with
    if len(feature_columns) == 0:
        raise ValueError("No numeric features available for selection.")

    X = df[feature_columns]
    y = df['target']

    # Use mutual information for feature selection
    mi_scores = mutual_info_classif(X, y)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    # Select top 15 features or all if less than 15
    num_features = min(15, len(mi_scores))
    top_features = mi_scores.head(num_features).index.tolist()

    logging.info(f"Top {num_features} features based on mutual information: {top_features}")

    return df[top_features], y, top_features

def create_model_pipeline(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=1,
        random_state=42
    )

    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        min_child_weight=0.001,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary',
        class_weight='balanced',
        random_state=42,
        verbosity=-1
    )

    model = VotingClassifier(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
        voting='soft'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline

def hyperparameter_tuning(pipeline, X, y):
    param_dist = {
        'model__xgb__max_depth': [3, 4, 5],
        'model__xgb__learning_rate': [0.01, 0.1],
        'model__xgb__n_estimators': [50, 100],
        'model__xgb__min_child_weight': [1, 3],
        'model__xgb__subsample': [0.8, 1.0],
        'model__xgb__colsample_bytree': [0.8, 1.0],
        'model__lgb__num_leaves': [15, 31],
        'model__lgb__learning_rate': [0.01, 0.1],
        'model__lgb__n_estimators': [50, 100],
        'model__lgb__min_child_samples': [10, 20],
        'model__lgb__subsample': [0.8, 1.0],
        'model__lgb__colsample_bytree': [0.8, 1.0],
    }

    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=20,
                                       cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc', n_jobs=-1,
                                       random_state=42, verbose=2)
    random_search.fit(X, y)

    logging.info(f"Best parameters: {random_search.best_params_}")
    logging.info(f"Best ROC AUC score: {random_search.best_score_:.4f}")

    return random_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    plot_roc_curve(y_test, y_prob)
    plot_learning_curve(model, X_test, y_test)

    logging.info("Model evaluation completed.")
    return y_pred, y_prob


def walk_forward_validation(X, y, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = roc_auc_score(y_test, y_pred)
        scores.append(score)

    return np.mean(scores)


def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random classifier', line=dict(dash='dash')))
    fig.update_layout(
        title=f'Receiver Operating Characteristic (ROC) Curve (AUC = {auc:.4f})',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )
    fig.show()

def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=TimeSeriesSplit(n_splits=5),
                                                            scoring='roc_auc', n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean, mode='lines', name='Training score',
                             error_y=dict(type='data', array=train_scores_std, visible=True)))
    fig.add_trace(go.Scatter(x=train_sizes, y=test_scores_mean, mode='lines', name='Cross-validation score',
                             error_y=dict(type='data', array=test_scores_std, visible=True)))
    fig.update_layout(title='Learning Curve', xaxis_title='Training examples', yaxis_title='Score')

    fig.show()


def analyze_feature_importance(model, feature_names):
    importances = []
    for name, estimator in model.named_steps['model'].named_estimators_.items():
        if hasattr(estimator, 'feature_importances_'):
            importances.append(estimator.feature_importances_)
        elif hasattr(estimator, 'coef_'):
            importances.append(np.abs(estimator.coef_[0]))

    if not importances:
        logging.warning("No feature importances found in any of the models.")
        return None, None

    mean_importances = np.mean(importances, axis=0)
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': mean_importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # Create Plotly figure
    fig = go.Figure(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h'
    ))
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Features",
        height=800,
        width=1000
    )

    logging.info("Top 10 most important features:")
    logging.info(feature_importance.head(10))

    fig.show()

    return feature_importance


def predict_future_price_direction(pipeline, last_data_point, feature_list):
    logging.info("Predicting future stock price direction...")

    try:
        # Ensure last_data_point is a DataFrame with the correct feature names
        if isinstance(last_data_point, pd.Series):
            last_data_point = pd.DataFrame([last_data_point])
        elif not isinstance(last_data_point, pd.DataFrame):
            last_data_point = pd.DataFrame([last_data_point], columns=feature_list)

        # Ensure all expected features are present
        for feature in feature_list:
            if feature not in last_data_point.columns:
                last_data_point[feature] = 0  # or another appropriate default value

        # Use the full pipeline to make predictions
        prediction_prob = pipeline.predict_proba(last_data_point)[0, 1]
        prediction = int(prediction_prob > 0.5)

        logging.info(f"Predicted probability of price increase: {prediction_prob:.4f}")
        logging.info(f"Predicted direction: {'Up' if prediction == 1 else 'Down'}")

        return prediction, prediction_prob

    except Exception as e:
        logging.error(f"An error occurred in predict_future_price_direction: {str(e)}")
        raise
def extract_serializable_params(model):
    """
    Extracts serializable parameters from a scikit-learn pipeline or estimator.
    """
    def serialize_step(step):
        if hasattr(step, 'get_params'):
            return {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                    for k, v in step.get_params().items()}
        return str(step)

    if hasattr(model, 'steps'):
        return {name: serialize_step(step) for name, step in model.steps}
    elif hasattr(model, 'estimators_'):
        return {f'estimator_{i}': serialize_step(est) for i, est in enumerate(model.estimators_)}
    else:
        return serialize_step(model)


def plot_model_returns(y_pred, y_test, close_prices):
    daily_investment = 100
    model_returns = []
    buy_and_hold_returns = []
    cumulative_investment = 0

    for pred, actual, price in zip(y_pred, y_test, close_prices):
        if pred == 1:  # Model recommends buying
            cumulative_investment += daily_investment
            model_returns.append(cumulative_investment * (1 + actual * 0.01))  # Assuming 1% return for simplicity
        else:
            model_returns.append(model_returns[-1] if model_returns else 0)

        buy_and_hold_returns.append(cumulative_investment * (price / close_prices[0]))
        cumulative_investment += daily_investment

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=close_prices.index, y=model_returns, name="Model Returns"), secondary_y=False)
    fig.add_trace(go.Scatter(x=close_prices.index, y=buy_and_hold_returns, name="Buy and Hold Returns"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=close_prices.index, y=close_prices, name="Stock Price"), secondary_y=True)

    fig.update_layout(title_text="Model Returns vs Buy and Hold Strategy")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Portfolio Value ($)", secondary_y=False)
    fig.update_yaxes(title_text="Stock Price ($)", secondary_y=True)

    fig.show()

#%%
def train_binary_classification_model(stock_symbol, start_date, end_date, s3_bucket_name, s3_folder=''):
    query = f"""
    SELECT * FROM enriched_stock_data
    WHERE symbol = '{stock_symbol}'
    AND date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date
    """

    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine_from_url(db_url)

    original_data = fetch_dataframe(engine, query)

    if original_data.empty:
        raise ValueError(f"No data found for stock symbol {stock_symbol} between {start_date} and {end_date}")

    stock_data = preprocess_data(original_data)
    processed_data = engineer_additional_features(stock_data)
    X, y, feature_list = select_features(processed_data)

    if X.empty:
        raise ValueError("No features available after selection process")

    # Handle class imbalance
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError(f"Only one class present in the target variable: {class_counts}")

    if class_counts.min() / class_counts.max() < 0.4:
        logging.warning("Class imbalance detected. Applying SMOTE.")
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

    # Create and train the model
    model_pipeline = create_model_pipeline(X)
    best_model = hyperparameter_tuning(model_pipeline, X, y)

    # Evaluate on the most recent 2 months of data
    split_date = X.index[-1] - pd.DateOffset(months=2)
    X_train, X_test = X[X.index <= split_date], X[X.index > split_date]
    y_train, y_test = y[y.index <= split_date], y[y.index > split_date]

    if X_test.empty or y_test.empty:
        raise ValueError("Not enough data for testing. Consider using a longer date range.")

    best_model.fit(X_train, y_train)

    mean_roc_auc = walk_forward_validation(X, y, best_model)
    logging.info(f"Mean ROC AUC across folds: {mean_roc_auc:.4f}")

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    #plot_model_returns(y_pred, y_test, X_test)

    feature_importance = analyze_feature_importance(best_model, feature_list)


    with tempfile.TemporaryDirectory() as tmpdir:
        model_file_name = f'ensemble_stock_direction_model_{stock_symbol}.pkl'
        temp_model_location = os.path.join(tmpdir, model_file_name)

        with open(temp_model_location, 'wb') as f:
            pickle.dump({
                'model': best_model,
                'feature_list': feature_list
            }, f)

        logging.info(f"Model training completed and saved temporarily at {temp_model_location}")

        s3_upload_file(temp_model_location, s3_bucket_name, model_file_name, folder_name=s3_folder)
        logging.info(f"Model saved to S3 at bucket: {s3_bucket_name}, folder: {s3_folder}")

        # Predict future direction
        last_data_point = X.iloc[-1]
        prediction, prediction_prob = predict_future_price_direction(best_model, last_data_point, feature_list)

    if prediction is None or prediction_prob is None:
        logging.error("Failed to make prediction. Skipping database update.")
    else:
        # Extract serializable parameters
        serializable_params = extract_serializable_params(best_model)

        # Convert the feature importance DataFrame to a list of dictionaries
        feature_importance_dict = feature_importance.to_dict(orient='records')

        log_data = pd.DataFrame({
            'symbol': [stock_symbol],
            'prediction_date': [datetime.now().date()],
            'prediction_explanation': [
                'Based on Ensemble (XGBoost + LightGBM) binary classification model with hyperparameter tuning'],
            'prediction_accuracy': [accuracy_score(y_test, y_pred)],
            'prediction_roc_auc': [roc_auc_score(y_test, y_prob)],
            'model_parameters': [json.dumps(serializable_params)],  # Use the serializable parameters
            'predicted_direction': ['Up' if prediction == 1 else 'Down'],
            'prediction_probability': [prediction_prob],
            'feature_importance': [json.dumps(feature_importance_dict)],  # Use the feature importance dictionary
            'last_known_price': [original_data['close'].iloc[-1]],
            'model_location': [f"s3://{s3_bucket_name}/{s3_folder}/{model_file_name}"],
            'date_created': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })

        upsert_df(log_data, 'trained_models_binary', 'symbol', engine)
        logging.info("Model saved and data logged to the database.")

#%%
train_binary_classification_model(
    stock_symbol="PG",
    start_date="2015-01-01",  # Extended start date for more training data
    end_date="2024-10-01",
    s3_bucket_name="trained-models-stock-prediction",
    s3_folder="ensemble_binary"
)