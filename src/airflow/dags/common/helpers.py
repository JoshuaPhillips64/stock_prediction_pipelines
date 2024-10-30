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


def invoke_lambda_function(payload):
    # Fetching AWS credentials from Airflow 'aws_default' connection
    connection = BaseHook.get_connection('aws_default')

    # Initialize boto3 client with credentials from Airflow connection
    client = boto3.client(
        'lambda',
        aws_access_key_id=connection.login,
        aws_secret_access_key=connection.password,
        region_name='us-east-1'
    )

    payload = payload

    try:
        # Invoke the Lambda function asynchronously
        response = client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType='Event',  # Asynchronous invocation
            Payload=json.dumps(payload)
        )

        # Check if the invocation was accepted
        status_code = response['StatusCode']
        if status_code == 202:
            logger.info(f'Lambda invoked asynchronously for {stock_ticker}. No immediate result returned.')
        else:
            logger.error(f'Lambda invocation failed for {stock_ticker} with status code {status_code}')

    except Exception as e:
        logger.error(f"Error invoking Lambda for {stock_ticker}: {e}")
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