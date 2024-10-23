import json
import pickle
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import tempfile
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# AWS S3 and database functions (assumed to be available in your environment)
from aws_functions import pull_from_s3
from database_functions import create_engine_from_url, fetch_dataframe, upsert_df

# Configuration variables (assumed to be available in your environment)
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, S3_BUCKET_NAME

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def lambda_handler(event, context):
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
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid input parameters')
        }

    # Call the make_SARIMAX_prediction function
    result = make_SARIMAX_prediction(
        model_key=model_key,
        stock_symbol=stock_symbol,
        input_date=input_date,
        hyperparameter_tuning=hyperparameter_tuning,
        feature_set=feature_set,
        lookback_period=lookback_period,
        prediction_horizon=prediction_horizon
    )

    if result:
        return {
            'statusCode': 200,
            'body': json.dumps('Prediction completed and logged successfully.')
        }
    else:
        return {
            'statusCode': 500,
            'body': json.dumps('Prediction failed.')
        }

# Helper functions and updated make_SARIMAX_prediction function

def load_model_from_s3(model_key, s3_bucket_name, s3_folder=''):
    model_file_name = f'sarimax_model_{model_key}.pkl'

    s3_bucket_name = 'trained-models-stock-prediction'
    s3_folder = 'sarimax'

    # If folder name is provided, include it in the path
    s3_file_name = f'{s3_folder}/{model_file_name}' if s3_folder else model_file_name

    try:
        # Create a temporary directory using tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_model_location = os.path.join(tmpdir, model_file_name)

            # Download the model from S3 using the `pull_from_s3` function
            pull_from_s3(s3_bucket_name, s3_file_name, temp_model_location)
            logging.info(f"Model {model_file_name} downloaded from S3 successfully.")

            # Load the pickle file
            with open(temp_model_location, 'rb') as file:
                model_data = pickle.load(file)

            # The temporary directory and its contents will be cleaned up automatically
            logging.info("Model loaded successfully from S3 and temporary file will be cleaned up.")

        return model_data
    except Exception as e:
        logging.error(f"Error loading model from S3: {e}")
        return None

def check_stationarity(df, column='log_return_1', alpha=0.05):
    from statsmodels.tsa.stattools import adfuller
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
    df.dropna(subset=['log_return_future'], inplace=True)

    # Perform differencing if the series is non-stationary
    if not check_stationarity(df, column='log_return_1'):
        logging.info("Time series is non-stationary, applying differencing...")
        df['log_return_1'] = df['log_return_1'].diff().dropna()

    logging.info("Data preprocessing completed.")
    return df

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
    target = df['log_return_future']

    # Ensure there are no NaNs
    data = pd.concat([selected_features, target], axis=1).dropna()
    selected_features = data[features]
    target = data['log_return_future']

    logging.info("Feature selection completed.")
    return selected_features, target, features

def make_SARIMAX_prediction(model_key, stock_symbol, input_date, hyperparameter_tuning, feature_set, lookback_period, prediction_horizon):
    s3_bucket_name = 'trained-models-stock-prediction'
    s3_folder = 'sarimax'
    model_data = load_model_from_s3(model_key, s3_bucket_name, s3_folder)
    if model_data is None:
        return False

    final_model = model_data['model']
    scaler = model_data['scaler']
    feature_list = model_data['feature_list']

    # Retrieve parameters from model_data
    model_prediction_horizon = model_data.get('prediction_horizon', prediction_horizon)
    model_feature_set = model_data.get('feature_set', feature_set)
    model_hyperparameter_tuning = model_data.get('hyperparameter_tuning', hyperparameter_tuning)
    model_lookback_period = model_data.get('lookback_period', lookback_period)

    # Use parameters from model_data to ensure consistency
    prediction_horizon = model_prediction_horizon
    feature_set = model_feature_set
    hyperparameter_tuning = model_hyperparameter_tuning
    lookback_period = model_lookback_period

    # Compute start_date and end_date
    input_date_dt = datetime.strptime(input_date, '%Y-%m-%d')
    start_date_dt = input_date_dt - timedelta(days=lookback_period + prediction_horizon)
    start_date = start_date_dt.strftime('%Y-%m-%d')
    end_date = input_date

    if feature_set == 'advanced':
        stock_table = 'enriched_stock_data'
    else:
        stock_table = 'basic_stock_data'

    # Database connection
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine_from_url(db_url)

    # Fetch the data up to the given date
    query = f"""
    SELECT * FROM {stock_table}
    WHERE symbol = '{stock_symbol}'
    AND date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date
    """
    historical_data = fetch_dataframe(engine, query)
    if historical_data.empty:
        logging.error(f"No data found for stock symbol {stock_symbol} up to {input_date}.")
        return False

    # Ensure 'date' column is datetime
    historical_data['date'] = pd.to_datetime(historical_data['date'])

    # Generate a list of the last (N) business days before the input date
    dates = pd.bdate_range(end=input_date_dt, periods=prediction_horizon).tolist()

    # Add the input_date + prediction_horizon to the list of dates
    final_prediction_date = pd.date_range(start=input_date_dt, periods=prediction_horizon + 1)[-1]

    final_dates = pd.bdate_range(start=input_date_dt, end=final_prediction_date).tolist()

    # Extend the dates list with final_dates, flattening the list
    dates.extend(final_dates)

    # Remove duplicates (if any) and sort the dates to ensure proper sequence
    dates = sorted(list(set(dates)))

    predictions = {}
    for prediction_date in dates:
        current_end_date = prediction_date - timedelta(days=prediction_horizon)
        current_start_date = current_end_date - timedelta(days=lookback_period)

        current_data = historical_data[
            (historical_data['date'] >= current_start_date) &
            (historical_data['date'] <= current_end_date)
        ]

        # Preprocess the data
        preprocessed_data = preprocess_data(current_data, prediction_horizon)

        # Engineer features
        engineered_data = engineer_additional_features(preprocessed_data)

        # Select relevant features
        X_future, _, _ = select_features(engineered_data, feature_set=feature_set)
        X_future = X_future[feature_list]

        # Scale the features
        X_scaled = pd.DataFrame(scaler.transform(X_future), index=X_future.index, columns=X_future.columns)

        # Get the last row for prediction
        X_pred = X_scaled.iloc[[-1]]

        # Make the prediction
        y_pred_future = final_model.forecast(steps=1, exog=X_pred)
        predicted_log_return = y_pred_future.iloc[0]

        # Convert log return to price
        last_known_price = current_data['close'].iloc[-1]
        predicted_price = last_known_price * np.exp(predicted_log_return)

        predictions[prediction_date.strftime('%Y-%m-%d')] = {
            "last_known_price": float(last_known_price),
            "predicted_price": float(predicted_price)
        }

    logging.info(f"Predictions generated for {stock_symbol} from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Prepare the model location string
    model_file_name = f'sarimax_model_{model_key}.pkl'
    model_location = f's3://{s3_bucket_name}/{s3_folder}/{model_file_name}'

    # Log the prediction to the database
    log_data = pd.DataFrame({
        'model_key': [model_key],
        'symbol': [stock_symbol],
        'date': [input_date],
        'model': ['SARIMAX'],
        'prediction_date': [dates[-1].strftime('%Y-%m-%d')],  # Use the last date in the prediction range
        'predicted_price': [predictions[dates[-1].strftime('%Y-%m-%d')]["predicted_price"]],  # Use the last predicted price
        'last_known_price': [predictions[dates[0].strftime('%Y-%m-%d')]["last_known_price"]],  # Use the first last known price
        'model_parameters': [json.dumps({
            'hyperparameter_tuning': hyperparameter_tuning,
            'feature_set': feature_set,
            'lookback_period': lookback_period,
            'prediction_horizon': prediction_horizon
        })],
        'model_location': [model_location],
        'date_created': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'predictions_json': [json.dumps(predictions)]  # New field with all predictions
    })

    # Upsert data to the database
    upsert_df(log_data, 'predictions_log', 'model_key,symbol,date', engine, json_columns=['predictions_json','model_parameters'])
    logging.info("Predictions logged successfully.")

    return True
#%% TEST FUnction

# Assume necessary imports are in place, including the lambda_handler and make_SARIMAX_prediction functions
def run_lambda_sarimax_predictions():
    # Define the model parameters
    model_key = 'SARIMAX_JNJ_advanced_HIGH_720_30'
    stock_symbol = 'JNJ'
    hyperparameter_tuning = 'MEDIUM'
    feature_set = 'advanced'
    lookback_period = 720
    prediction_horizon = 30

    # Define the input date (this will be the end date of our prediction range)
    input_date_str = '2024-10-01'

    # Prepare the event payload for the Lambda function
    event = {
        'body': json.dumps({
            'model_key': model_key,
            'stock_symbol': stock_symbol,
            'input_date': input_date_str,
            'hyperparameter_tuning': hyperparameter_tuning,
            'feature_set': feature_set,
            'lookback_period': lookback_period,
            'prediction_horizon': prediction_horizon
        })
    }

    # Simulate Lambda function invocation
    result = lambda_handler(event, None)

    return result

# Call the function to test
#response = run_lambda_sarimax_predictions()