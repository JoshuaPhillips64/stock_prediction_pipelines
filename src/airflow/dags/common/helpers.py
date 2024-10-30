import random
from airflow.hooks.postgres_hook import PostgresHook
from airflow.hooks.base_hook import BaseHook
import boto3
import json
import logging
from datetime import datetime, timedelta
from .config import LAMBDA_FUNCTION_NAME, TOP_50_TICKERS, POSTGRES_CONN_ID

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def invoke_lambda_function(lambda_name: str, payload: dict):
    """
    Invokes the specified AWS Lambda function asynchronously with the given payload.

    Args:
        lambda_name (str): The name of the Lambda function to invoke.
        payload (dict): The payload to send to the Lambda function.

    Returns:
        dict: A simplified response containing only serializable data.
    """
    # Fetching AWS credentials from Airflow 'aws_default' connection
    connection = BaseHook.get_connection('aws_default')

    # Initialize boto3 client with credentials from Airflow connection
    client = boto3.client(
        'lambda',
        aws_access_key_id=connection.login,
        aws_secret_access_key=connection.password,
        region_name='us-east-1'  # Adjust region as needed
    )

    try:
        # Serialize payload to JSON and encode to bytes
        serialized_payload = json.dumps(payload).encode('utf-8')

        # Invoke the Lambda function asynchronously
        response = client.invoke(
            FunctionName=lambda_name,
            InvocationType='Event',  # Asynchronous invocation
            Payload=serialized_payload
        )

        # Extract only serializable parts of the response
        serializable_response = {
            "StatusCode": response.get('StatusCode'),
            "RequestId": response.get('ResponseMetadata', {}).get('RequestId')
        }

        if serializable_response['StatusCode'] == 202:
            logger.info(f'Lambda "{lambda_name}" invoked asynchronously with payload: {payload}')
        else:
            logger.error(f'Lambda invocation failed for "{lambda_name}" with status code {serializable_response["StatusCode"]}')

        return serializable_response

    except Exception as e:
        logger.error(f"Error invoking Lambda '{lambda_name}': {e}")
        raise

def monitor_lambdas_completion(feature_set, **kwargs):
    ti = kwargs['ti']
    task_ids = [f'invoke_lambda_{stock}_{feature_set}' for stock in TOP_50_TICKERS]
    stock_responses = ti.xcom_pull(task_ids=task_ids)

    all_completed = True
    failed_stocks = []
    for stock, response in zip(TOP_50_TICKERS, stock_responses):
        if not response or response.get('statusCode') != 200:
            logger.error(f"Lambda invocation for {stock} failed or did not return a successful response.")
            all_completed = False
            failed_stocks.append(stock)
        else:
            logger.info(f"Lambda invocation for {stock} completed successfully.")

    if not all_completed:
        raise ValueError(f"Lambda invocations failed for stocks: {', '.join(failed_stocks)}")

    # Optionally, query the database to verify data
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    table_name = 'basic_stock_data' if feature_set == 'basic' else 'enriched_stock_data'
    for stock in TOP_50_TICKERS:
        sql = f"""
        SELECT * FROM {table_name}
        WHERE symbol = %s
        ORDER BY date DESC
        LIMIT 1;
        """
        result = pg_hook.get_first(sql, parameters=(stock,))
        if result:
            logger.info(f"Most recent record for {stock} in {table_name}: {result}")
        else:
            logger.warning(f"No data found for {stock} in {table_name}.")

# New Helper Functions for Random Parameter Selection

def get_random_hyperparameter_tuning():
    return random.choice(['LOW', 'MEDIUM', 'HIGH'])

def get_random_feature_set():
    return random.choice(['basic', 'advanced'])

def get_random_lookback_period():
    return random.randint(500, 730, 1000)

def get_random_prediction_horizon():
    return random.randint(7,14,30,60)

def get_random_parameters(model_type: str):
    """
    Returns a dictionary of randomly selected parameters based on the model type.

    Args:
        model_type (str): Either 'binary_classification' or 'sarimax'.

    Returns:
        dict: A dictionary of parameters.
    """
    if model_type == 'binary_classification':
        return {
            'model_key': 'binary_classification',
            'hyperparameter_tuning': get_random_hyperparameter_tuning(),
            'feature_set': get_random_feature_set(),
            'lookback_period': get_random_lookback_period(),
            'prediction_horizon': get_random_prediction_horizon()
        }
    elif model_type == 'sarimax':
        return {
            'model_key': 'sarimax',
            'hyperparameter_tuning': get_random_hyperparameter_tuning(),
            'feature_set': get_random_feature_set(),
            'lookback_period': get_random_lookback_period(),
            'prediction_horizon': get_random_prediction_horizon()
        }
    else:
        raise ValueError("Invalid model_type. Choose 'binary_classification' or 'sarimax'.")