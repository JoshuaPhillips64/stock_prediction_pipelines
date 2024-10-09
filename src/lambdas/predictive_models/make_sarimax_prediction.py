# Import necessary libraries
import pickle
import pandas as pd
import numpy as np
import logging
from database_functions import create_engine_from_url, fetch_dataframe, upsert_df
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
from aws_functions import pull_from_s3
from datetime import datetime, timedelta
import os
from statsmodels.tsa.stattools import adfuller
import warnings
import tempfile

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained SARIMAX model from S3
def load_model_from_s3(stock_symbol, s3_bucket_name, s3_folder=''):
    model_file_name = f'sarimax_stock_prediction_model_{stock_symbol}.pkl'

    # If folder name is provided, include it in the path
    s3_file_name = f'{s3_folder}/{model_file_name}' if s3_folder else model_file_name

    try:
        # Create a temporary directory using tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_model_location = os.path.join(tmpdir, model_file_name)

            # Download the model from S3 using the `pull_from_s3` function you provided
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
    logging.info("Checking stationarity of the time series...")
    result = adfuller(df[column].dropna())
    p_value = result[1]
    logging.info(f"ADF Statistic: {result[0]}, p-value: {p_value}")
    return p_value < alpha


def preprocess_data(df):
    logging.info("Starting data preprocessing with stationarity check...")
    df = df.copy()
    df.sort_values('date', inplace=True)

    # Ensure 'date' column is datetime
    df['date'] = pd.to_datetime(df['date'])

    df.set_index('date', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['close'], inplace=True)
    df.fillna(method='ffill', inplace=True)

    # Calculate log return
    df['log_return_30'] = np.log(df['close'].shift(-30) / df['close'])
    df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(subset=['log_return_30'], inplace=True)

    # Perform differencing if the series is non-stationary
    if not check_stationarity(df, column='log_return_1'):
        logging.info("Time series is non-stationary, applying differencing...")
        df['log_return_1'] = df['log_return_1'].diff().dropna()

    logging.info("Data preprocessing completed.")
    return df

# Feature Engineering
def engineer_additional_features(df):
    logging.info("Starting feature engineering...")

    # Shift period to align features with the target variable
    shift_period = 30

    # Shift features forward by shift_period to ensure only past data is used
    feature_columns = df.columns.difference(['log_return_30'])
    df[feature_columns] = df[feature_columns].shift(shift_period)

    # Drop original technical indicators to prevent data leakage
    technical_indicators = ['macd', 'macd_signal', 'macd_hist', 'rsi', 'upper_band', 'lower_band', 'adx', 'implied_volatility', 'other_indicator']
    df.drop(columns=[col for col in technical_indicators if col in df.columns], inplace=True)

    # Recalculate technical indicators using shifted data
    # Shifted close, high, low prices
    df['close_shifted'] = df['close'].shift(shift_period)
    df['high_shifted'] = df['high'].shift(shift_period)
    df['low_shifted'] = df['low'].shift(shift_period)
    df['log_return_1_shifted'] = df['log_return_1'].shift(shift_period)

    # Recalculate SMA and EMA using shifted close prices
    df['sma_50'] = df['close_shifted'].rolling(window=50).mean()
    df['sma_100'] = df['close_shifted'].rolling(window=100).mean()
    df['ema_50'] = df['close_shifted'].ewm(span=50, adjust=False).mean()

    # Momentum indicators using shifted close prices
    df['momentum_30'] = df['close_shifted'].shift(30) / df['close_shifted'].shift(60) - 1
    df['momentum_60'] = df['close_shifted'].shift(60) / df['close_shifted'].shift(90) - 1

    # Recalculate MACD
    df = calculate_macd(df, price_column='close_shifted')

    # Recalculate ADX
    df = calculate_adx(df, high_column='high_shifted', low_column='low_shifted', close_column='close_shifted')

    # Recalculate RSI using shifted close prices
    delta = df['close_shifted'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / roll_down
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    # Recalculate Bollinger Bands using shifted close prices
    df['middle_band'] = df['close_shifted'].rolling(window=20).mean()
    df['std_dev'] = df['close_shifted'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)
    df.drop(columns=['middle_band', 'std_dev'], inplace=True)

    # Adjusted rolling calculations
    df['rolling_volatility_60'] = df['log_return_1_shifted'].rolling(window=60).std()
    df['corr_sp500_60'] = df['log_return_1_shifted'].rolling(window=60).corr(df['sp500_return'].shift(1 + shift_period))
    df['corr_nasdaq_60'] = df['log_return_1_shifted'].rolling(window=60).corr(df['nasdaq_return'].shift(1 + shift_period))

    # Stochastic Oscillator with shifted data
    low_14 = df['low_shifted'].rolling(window=14).min()
    high_14 = df['high_shifted'].rolling(window=14).max()
    df['%K'] = (df['close_shifted'] - low_14) / (high_14 - low_14) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()

    # Commodity Channel Index with shifted data
    typical_price = (df['high_shifted'] + df['low_shifted'] + df['close_shifted']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad)

    # Use current date features shifted appropriately
    df['day_of_week'] = df.index.shift(shift_period, freq='D').dayofweek
    df['month'] = df.index.shift(shift_period, freq='D').month

    # Drop shifted columns if not needed
    df.drop(columns=['close_shifted', 'high_shifted', 'low_shifted', 'log_return_1_shifted'], inplace=True)

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

# Feature Selection
def select_features(df):
    logging.info("Starting feature selection...")
    features = [
        'volume',
        'sector_performance',  # Retained original variable
        'sector_performance_lag_1', 'sector_performance_lag_7', 'sector_performance_lag_30',
        'sp500_return',  # Retained original variable
        'sp500_return_lag_1', 'sp500_return_lag_7', 'sp500_return_lag_30',
        'nasdaq_return',  # Retained original variable
        'nasdaq_return_lag_1', 'nasdaq_return_lag_7', 'nasdaq_return_lag_30',
        'sentiment_score',  # Retained original variable
        'sentiment_score_lag_1',
        'gdp_growth',  # Retained original variable
        'gdp_growth_lag_1', 'gdp_growth_lag_7', 'gdp_growth_lag_30',
        'inflation_rate',  # Retained original variable
        'inflation_rate_lag_1', 'inflation_rate_lag_7', 'inflation_rate_lag_30',
        'unemployment_rate',  # Retained original variable
        'unemployment_rate_lag_1', 'unemployment_rate_lag_7', 'unemployment_rate_lag_30',
        'market_capitalization', 'pe_ratio', 'dividend_yield', 'beta', 'put_call_ratio',
        'macd_hist', 'adx', 'implied_volatility', 'macd', 'rsi', 'upper_band', 'lower_band',
        'macd_signal',
        'rolling_volatility_60',
        'corr_sp500_60', 'corr_nasdaq_60',
        'sma_50', 'sma_100', 'ema_50',
        '%K', '%D', 'cci',
        'momentum_30', 'momentum_60'
    ]
    features = [feature for feature in features if feature in df.columns]
    selected_features = df[features].copy()

    # Keep the target variable separate
    target = df['log_return_30']

    # Ensure there are no NaNs
    data = pd.concat([selected_features, target], axis=1).dropna()
    selected_features = data[features]
    target = data['log_return_30']

    logging.info("Feature selection completed.")
    return features


# Select the relevant features for the prediction
def select_features(df, feature_list):
    features = df[feature_list].dropna()
    return features


# Predict the future price using the loaded model
def make_prediction(stock_symbol, input_date, s3_bucket_name,s3_folder):
    model_data = load_model_from_s3(stock_symbol, s3_bucket_name, s3_folder)
    if model_data is None:
        return

    final_model = model_data['model']
    scaler = model_data['scaler']
    feature_list = model_data['feature_list']

    # Connect to the database to get data up to the prediction_date
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine_from_url(db_url)

    query = f"""
    SELECT * FROM enriched_stock_data
    WHERE symbol = '{stock_symbol}' AND date <= '{input_date}'
    ORDER BY date
    """

    # Fetch the data up to the given date
    historical_data = fetch_dataframe(engine, query)
    if historical_data.empty:
        logging.error(f"No data found for stock symbol {stock_symbol} up to {prediction_date}.")
        return

    # Preprocess the data
    preprocessed_data = preprocess_data(historical_data)

    # Engineer features
    engineered_data = engineer_additional_features(preprocessed_data)

    # Select relevant features
    X_future = select_features(engineered_data, feature_list)

    # Scale the features
    X_scaled = pd.DataFrame(scaler.transform(X_future), columns=feature_list)

    # Get the last row for prediction (as of the prediction_date)
    X_pred = X_scaled.iloc[-1:].values.reshape(1, -1)

    # Make the prediction
    y_pred_future = final_model.forecast(steps=1, exog=X_pred)
    predicted_log_return = y_pred_future.iloc[0]

    # Convert log return to price
    last_known_price = historical_data['close'].iloc[-1]
    predicted_price = last_known_price * np.exp(predicted_log_return)

    logging.info(f"Predicted price for {stock_symbol} 30 days out from {input_date}: {predicted_price}")

    # Get the prediction date
    prediction_date = (datetime.strptime(input_date, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')

    # Log the prediction to the database
    log_data = pd.DataFrame({
        'symbol': [stock_symbol],
        'date': [input_date],
        'model': ['SARIMAX'],
        'prediction_date': [prediction_date],
        'predicted_price': [predicted_price],
        'last_known_price': [last_known_price],
        'model_location': [f'trained_models/sarimax_stock_prediction_model_{stock_symbol}.pkl'],
        'date_created': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    })

    upsert_df(log_data, 'predictions_log', 'symbol,date,model', engine)
    logging.info("Prediction logged successfully.")


#%% Example usage:
# Create list of dates to loop through
dates = pd.date_range(start='2024-08-01', end='2024-09-30', freq='B').strftime('%Y-%m-%d')
#convert to list
dates = dates.tolist()

# Loop through the dates and make predictions
for date in dates:
    make_prediction(stock_symbol="JNJ",
                    input_date=date,
                    s3_bucket_name='trained-models-stock-prediction',
                    s3_folder='sarimax')
