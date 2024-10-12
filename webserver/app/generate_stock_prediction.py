import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from database_functions import create_engine_from_url, fetch_dataframe
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_DATABASE, S3_BUCKET_NAME
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection string
db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
engine = create_engine_from_url(db_url)

# Lambda function URLs (replace with your actual endpoints or environment variables)
LOAD_DATA_LAMBDA_URL = os.environ.get('LOAD_DATA_LAMBDA_URL')
TRAIN_SARIMAX_LAMBDA_URL = os.environ.get('TRAIN_SARIMAX_LAMBDA_URL')
TRAIN_BINARY_LAMBDA_URL = os.environ.get('TRAIN_BINARY_LAMBDA_URL')
PREDICT_SARIMAX_LAMBDA_URL = os.environ.get('PREDICT_SARIMAX_LAMBDA_URL')
PREDICT_BINARY_LAMBDA_URL = os.environ.get('PREDICT_BINARY_LAMBDA_URL')
CHATGPT_ANALYZE_LAMBDA_URL = os.environ.get('CHATGPT_ANALYZE_LAMBDA_URL')

def generate_model_key(model_type, stock_symbol, feature_set, hyperparameter_tuning, lookback_period, prediction_horizon):
    """
    Generates a unique model key based on input parameters.
    """
    # Get today's date
    today = datetime.today()

    # Format the date
    formatted_date = today.strftime('%Y-%m-%d')

    model_key = f"{model_type}_{stock_symbol}_{feature_set}_{hyperparameter_tuning}_{lookback_period}_{prediction_horizon}_{formatted_date}"
    return model_key

def generate_stock_prediction(
    model_type: str,
    stock_symbol: str,
    input_date: str,
    hyperparameter_tuning: str = 'MEDIUM',
    feature_set: str = 'advanced',
    lookback_period: int = 720,
    prediction_horizon: int = 30,
):
    """
    Generates a stock prediction based on the given parameters.

    Args:
        model_type (str): The type of model to use. Options: 'SARIMAX', 'BINARY CLASSIFICATION'.
        stock_symbol (str): The stock symbol to predict.
        input_date (str): The date for which to make the prediction (format: 'YYYY-MM-DD').
        hyperparameter_tuning (str, optional): The level of hyperparameter tuning. Options: 'LOW', 'MEDIUM', 'HIGH'. Defaults to 'MEDIUM'.
        feature_set (str, optional): The set of features to use. Options: 'basic', 'advanced'. Defaults to 'advanced'.
        lookback_period (int, optional): The number of days of historical data to use for training. Defaults to 720.
        prediction_horizon (int, optional): The number of days into the future to predict. Defaults to 30.

    Returns:
        dict: The prediction results in JSON format, or an error message if unsuccessful.
    """
    try:
        # Validate inputs
        validate_inputs(model_type, stock_symbol, input_date, hyperparameter_tuning, feature_set)

        # Generate model key
        model_key = generate_model_key(model_type, stock_symbol, feature_set, hyperparameter_tuning, lookback_period, prediction_horizon)

        model_exists = check_model_existence(model_key, model_type, engine)
        ### ADD step to skip model training if model key already exists

        # Check if data exists, fetch via Lambda if necessary
        logger.info(f"Checking data existence for {stock_symbol}.")
        data_exists = check_data_existence(stock_symbol, input_date, engine, lookback_period, feature_set)

        if not data_exists:
            logger.info(f"Data for {stock_symbol} Running update lembda via Lambda.")
            payload = {
                'stock_symbol': stock_symbol,
                'start_date': (datetime.strptime(input_date, '%Y-%m-%d') - timedelta(days=lookback_period)).strftime('%Y-%m-%d'),
                'end_date': input_date,
                'feature_set': feature_set
            }
            response = requests.post(LOAD_DATA_LAMBDA_URL, json=payload)
            if response.status_code != 200:
                raise ValueError(f"Failed to load data via Lambda. Status code: {response.status_code}, Response: {response.text}")
            load_data_output = response.json()
            data_exists_second = check_data_existence(stock_symbol, input_date, engine, lookback_period, feature_set)
            if not data_exists_second:
                raise ValueError(f"Failed to load data for {stock_symbol} via Lambda.")

            logger.info(f"Data for {stock_symbol} successfully fetched and saved to the database via Lambda.")
        else:
            logger.info(f"Data for {stock_symbol} already exists in the database.")

        # Step 1: Train model via Lambda
        logger.info(f"Training {model_type} model for {stock_symbol} via Lambda function.")
        if model_type == 'SARIMAX':
            payload = {
                'model_key': model_key,
                'stock_symbol': stock_symbol,
                'input_date': input_date,
                'hyperparameter_tuning': hyperparameter_tuning,
                'lookback_period': lookback_period,
                'prediction_horizon': prediction_horizon,
                'feature_set': feature_set
            }
            response = requests.post(TRAIN_SARIMAX_LAMBDA_URL, json=payload)
            if response.status_code != 200:
                raise ValueError(f"Failed to train SARIMAX model via Lambda. Status code: {response.status_code}, Response: {response.text}")
            train_model_output = response.json()
        elif model_type == 'BINARY CLASSIFICATION':
            payload = {
                'model_key': model_key,
                'stock_symbol': stock_symbol,
                'input_date': input_date,
                'hyperparameter_tuning': hyperparameter_tuning,
                'lookback_period': lookback_period,
                'feature_set': feature_set
            }
            response = requests.post(TRAIN_BINARY_LAMBDA_URL, json=payload)
            if response.status_code != 200:
                raise ValueError(f"Failed to train binary classification model via Lambda. Status code: {response.status_code}, Response: {response.text}")
            train_model_output = response.json()
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        # Step 2: Make prediction via Lambda
        logger.info(f"Generating prediction using {model_type} model for {stock_symbol} on {input_date} via Lambda.")
        if model_type == 'SARIMAX':
            payload = {
                'model_key': model_key,
                'stock_symbol': stock_symbol,
                'input_date': input_date,
                'prediction_horizon': prediction_horizon
            }
            response = requests.post(PREDICT_SARIMAX_LAMBDA_URL, json=payload)
            if response.status_code != 200:
                raise ValueError(f"Failed to make prediction via SARIMAX Lambda. Status code: {response.status_code}, Response: {response.text}")
            prediction_result = response.json()
        elif model_type == 'BINARY CLASSIFICATION':
            payload = {
                'model_key': model_key,
                'stock_symbol': stock_symbol,
                'input_date': input_date,
                'feature_set': feature_set
            }
            response = requests.post(PREDICT_BINARY_LAMBDA_URL, json=payload)
            if response.status_code != 200:
                raise ValueError(f"Failed to make prediction via binary classification Lambda. Status code: {response.status_code}, Response: {response.text}")
            prediction_result = response.json()
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        # Step 3: Call ChatGPT analyze Lambda
        logger.info(f"Analyzing prediction for {stock_symbol} via ChatGPT Lambda.")
        payload = {
            'model_type': model_type,
            'model_key': model_key,
            'stock_symbol': stock_symbol,
            'input_date': input_date,
            'prediction_result': prediction_result,
            'train_model_output': train_model_output
        }
        response = requests.post(CHATGPT_ANALYZE_LAMBDA_URL, json=payload)
        if response.status_code != 200:
            raise ValueError(f"Failed to analyze prediction via ChatGPT Lambda. Status code: {response.status_code}, Response: {response.text}")
        analyze_result = response.json()

        # Construct final result
        final_result = {
            'model_type': model_type,
            'model_key': model_key,
            'stock_symbol': stock_symbol,
            'input_date': input_date,
            'prediction_result': prediction_result,
            'train_model_output': train_model_output,
            'chatgpt_review': analyze_result
        }

        # Return final result in JSON format
        return final_result

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {'error': str(e)}

def validate_inputs(model_type: str, stock_symbol: str, input_date: str, hyperparameter_tuning: str, feature_set: str):
    """Validates the input parameters."""
    if model_type not in ('SARIMAX', 'BINARY CLASSIFICATION'):
        raise ValueError(f"Invalid model_type: {model_type}")
    if not isinstance(stock_symbol, str):
        raise TypeError("stock_symbol must be a string")
    try:
        datetime.strptime(input_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect date format, should be YYYY-MM-DD")
    if hyperparameter_tuning not in ('LOW', 'MEDIUM', 'HIGH'):
        raise ValueError(f"Invalid hyperparameter_tuning: {hyperparameter_tuning}")
    if feature_set not in ('basic', 'technical', 'advanced', 'custom'):
        raise ValueError(f"Invalid feature_set: {feature_set}")

def check_data_existence(stock_symbol: str, input_date: str, engine, lookback_period: int, feature_set) -> bool:
    """
    Checks if data for the given stock symbol exists for all weekdays between the start and end dates in the date range.

    Args:
        stock_symbol (str): The stock symbol.
        input_date (str): The input date (format: 'YYYY-MM-DD').
        engine: The SQLAlchemy engine.
        lookback_period (int): Number of weekdays to check (excluding weekends).

    Returns:
        bool: True if data exists for the full date range, False otherwise.
    """
    # Determine the appropriate table based on feature set
    if feature_set == 'basic':
        upsert_table = 'basic_stock_data'
    else:
        upsert_table = 'enriched_stock_data'

    # Calculate the start date
    start_date = (datetime.strptime(input_date, '%Y-%m-%d') - timedelta(days=lookback_period)).strftime('%Y-%m-%d')

    # Query to get the min and max dates from the database for the specified symbol
    query = f"""
        SELECT MIN(date) AS min_date, MAX(date) AS max_date
        FROM {upsert_table}
        WHERE symbol = '{stock_symbol}' AND date BETWEEN '{start_date}' AND '{input_date}'
    """

    # Execute the query and fetch the result
    result = fetch_dataframe(engine, query)

    # Check if the min and max dates match the expected range
    if not result.empty:
        min_date = result['min_date'][0]
        max_date = result['max_date'][0]

        # Ensure that the min date equals the start date and the max date equals the input date
        if min_date == start_date and max_date == input_date:
            return True

    return False

def check_model_existence(model_key, model_type, engine):
    """
    Checks if the model with the specified key exists in the database.

    Args:
        model_key (str): The model key.
        engine: The SQLAlchemy engine.

    Returns:
        bool: True if the model exists, False otherwise.
    """

    if model_type == 'SARIMAX':
        model_table = 'trained_models'
    elif model_type == 'BINARY CLASSIFICATION':
        model_table = 'trained_models_binary'
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    query = f"""
        SELECT COUNT(*)
        FROM {model_table}
        WHERE model_key = '{model_key}'
    """
    result = fetch_dataframe(engine, query)
    if result.iloc[0, 0] > 0:
        return True
    return False

# Example of a Flask route that uses the generate_stock_prediction function
@app.route('/predict_stock', methods=['POST'])
def predict_stock():
    data = request.get_json()
    model_type = data.get('model_type')
    stock_symbol = data.get('stock_symbol')
    input_date = data.get('input_date')
    hyperparameter_tuning = data.get('hyperparameter_tuning', 'MEDIUM')
    feature_set = data.get('feature_set', 'advanced')
    lookback_period = data.get('lookback_period', 720)
    prediction_horizon = data.get('prediction_horizon', 30)

    result = generate_stock_prediction(
        model_type=model_type,
        stock_symbol=stock_symbol,
        input_date=input_date,
        hyperparameter_tuning=hyperparameter_tuning,
        feature_set=feature_set,
        lookback_period=lookback_period,
        prediction_horizon=prediction_horizon
    )

    return jsonify(result)
