import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import warnings
from database_functions import create_engine_from_url, fetch_dataframe, upsert_df
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
from aws_functions import s3_upload_file
from datetime import datetime, timedelta
import json
import pickle
import os
import tempfile

warnings.filterwarnings("ignore")

# Set up logging
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
    logging.info("Data preprocessing completed.")
    return df


def engineer_features(df):
    logging.info("Starting feature engineering...")

    # Lagged features (to prevent look-ahead bias)
    for lag in [1, 3, 7, 14, 30]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'return_lag_{lag}'] = df['close'].pct_change(lag)

    # Rolling window features
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'std_{window}'] = df['close'].rolling(window=window).std()
        df[f'rsi_{window}'] = calculate_rsi(df['close'], window)
        df[f'volatility_{window}'] = df['return_lag_1'].rolling(window=window).std()

    # MACD
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])

    # Bollinger Bands
    df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['close'])

    # Volume features
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Date-based features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter

    # Market-related features (assuming these are already in the dataset)
    market_features = ['sector_performance', 'sp500_return', 'nasdaq_return',
                       'sentiment_score', 'gdp_growth', 'inflation_rate', 'unemployment_rate']

    for feature in market_features:
        if feature in df.columns:
            df[f'{feature}_ma'] = df[feature].rolling(window=30).mean()

    logging.info("Feature engineering completed.")
    return df


def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal


def calculate_bollinger_bands(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band


def select_features(df):
    logging.info("Selecting features...")
    features = [col for col in df.columns if col not in ['target', 'future_price', 'price_change']]
    X = df[features]
    y = df['target']
    return X, y, features


def create_model_pipeline(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    lgb_model = lgb.LGBMClassifier()

    model = VotingClassifier(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
        voting='soft'
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

    return pipeline


def time_series_cv(X, y, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = roc_auc_score(y_test, y_pred)
        cv_scores.append(score)

    return np.mean(cv_scores), np.std(cv_scores)


def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        logging.warning("Model doesn't have feature_importances_ or coef_ attribute")
        return

    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(20)

    fig = go.Figure([go.Bar(x=feature_importance['feature'], y=feature_importance['importance'])])
    fig.update_layout(title='Top 20 Feature Importances', xaxis_title='Features', yaxis_title='Importance',
                      xaxis_tickangle=-45)
    fig.show()


def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    logging.info("Model evaluation completed.")
    return y_pred, y_prob


def plot_roc_curve(y_test, y_prob):
    logging.info("Plotting ROC curve...")
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
    logging.info("ROC curve plotting completed.")


def predict_future_price_direction(model, last_data_point, feature_list):
    logging.info("Predicting future stock price direction...")

    try:
        X_pred = last_data_point[feature_list].values.reshape(1, -1)
        prediction_prob = model.predict_proba(X_pred)[0, 1]
        prediction = int(prediction_prob > 0.5)

        logging.info(f"Predicted probability of price increase: {prediction_prob:.4f}")
        logging.info(f"Predicted direction: {'Up' if prediction == 1 else 'Down'}")

        return prediction, prediction_prob

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        print(f"Error: {str(e)}")

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

    stock_data = preprocess_data(original_data)
    processed_data = engineer_features(stock_data)
    X, y, feature_list = select_features(processed_data)

    model_pipeline = create_model_pipeline(X)

    # Time series cross-validation
    cv_score, cv_std = time_series_cv(X, y, model_pipeline)
    logging.info(f"Cross-validation ROC AUC: {cv_score:.4f} (+/- {cv_std:.4f})")

    # Train on full dataset
    model_pipeline.fit(X, y)

    # Plot feature importance
    plot_feature_importance(model_pipeline.named_steps['model'], feature_list)

    # Evaluate on the most recent 20% of the data
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    y_pred, y_prob = evaluate_model(model_pipeline, X_test, y_test)
    plot_roc_curve(y_test, y_prob)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_file_name = f'ensemble_stock_direction_model_{stock_symbol}.pkl'
        temp_model_location = os.path.join(tmpdir, model_file_name)

        with open(temp_model_location, 'wb') as f:
            pickle.dump({
                'model': model_pipeline,
                'feature_list': feature_list
            }, f)

        logging.info(f"Model training completed and saved temporarily at {temp_model_location}")

        s3_upload_file(temp_model_location, s3_bucket_name, model_file_name, folder_name=s3_folder)
        logging.info(f"Model saved to S3 at bucket: {s3_bucket_name}, folder: {s3_folder}")

    # Predict future direction
    last_data_point = X.iloc[-1]
    prediction, prediction_prob = predict_future_price_direction(model_pipeline, last_data_point, feature_list)

    log_data = pd.DataFrame({
        'symbol': [stock_symbol],
        'prediction_date': [datetime.now().date()],
        'prediction_explanation': ['Based on Ensemble (XGBoost + LightGBM) binary classification model'],
        'prediction_cv_score': [cv_score],
        'prediction_cv_std': [cv_std],
        'prediction_accuracy': [accuracy_score(y_test, y_pred)],
        'prediction_roc_auc': [roc_auc_score(y_test, y_prob)],
        'model_parameters': [json.dumps(model_pipeline.get_params())],
        'predicted_direction': ['Up' if prediction == 1 else 'Down'],
        'prediction_probability': [prediction_prob],
        'last_known_price': [original_data['close'].iloc[-1]],
        'model_location': [temp_model_location],
        'date_created': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    })

    upsert_df(log_data, 'trained_models', 'symbol', engine)
    logging.info("Model saved and data logged to the database.")


#%% Usage
train_binary_classification_model(
    stock_symbol="JNJ",
    start_date="2021-01-01",
    end_date="2024-10-01",
    s3_bucket_name="trained-models-stock-prediction",
    s3_folder="ensemble_binary"
)