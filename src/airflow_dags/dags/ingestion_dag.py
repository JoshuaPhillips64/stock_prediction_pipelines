from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import BranchPythonOperator
from airflow.providers.amazon.aws.operators.lambda_function import LambdaInvokeFunctionOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
import os

default_args = {
    'owner': 'airflow',
}

def choose_operator(**kwargs):
    environment = os.getenv('ENVIRONMENT', 'production')
    if environment == 'local':
        return 'ingest_alpha_vantage_http'
    else:
        return 'ingest_alpha_vantage_lambda'

def choose_enrich_operator(**kwargs):
    environment = os.getenv('ENVIRONMENT', 'production')
    if environment == 'local':
        return 'enrich_stock_data_http'
    else:
        return 'enrich_stock_data_lambda'

with DAG(
    'data_ingestion_dag',
    default_args=default_args,
    description='DAG for ingesting data from various sources into PostgreSQL',
    schedule_interval='@hourly',
    start_date=days_ago(1),
    tags=['ingestion'],
) as dag:

    # Task to choose the ingestion method
    choose_task = BranchPythonOperator(
        task_id='choose_operator',
        python_callable=choose_operator,
    )

    # Task for ingesting via Lambda
    ingest_alpha_vantage_lambda = LambdaInvokeFunctionOperator(
        task_id='ingest_alpha_vantage_lambda',
        function_name='ingest_alpha_vantage_function_name',  # Replace with your Lambda function name
        invocation_type='RequestResponse',
    )

    # Task for ingesting via HTTP
    ingest_alpha_vantage_http = SimpleHttpOperator(
        task_id='ingest_alpha_vantage_http',
        method='POST',
        http_conn_id='ingest_alpha_vantage_service',
        endpoint='/invoke',
        data='{}',
        headers={"Content-Type": "application/json"},
    )

    # Task to choose between Lambda or HTTP for enriching stock data
    choose_enrich_task = BranchPythonOperator(
        task_id='choose_enrich_operator',
        python_callable=choose_enrich_operator,
    )

    # New task for enriching stock data via Lambda
    enrich_stock_data_lambda = LambdaInvokeFunctionOperator(
        task_id='enrich_stock_data_lambda',
        function_name='enrich_stock_data_function_name',  # Replace with your Lambda function name
        invocation_type='RequestResponse',
    )

    # New task for enriching stock data via HTTP
    enrich_stock_data_http = SimpleHttpOperator(
        task_id='enrich_stock_data_http',
        method='POST',
        http_conn_id='enrich_stock_data_service',
        endpoint='/invoke',
        data='{}',
        headers={"Content-Type": "application/json"},
    )

    # Define task dependencies
    choose_task >> [ingest_alpha_vantage_lambda, ingest_alpha_vantage_http]
    ingest_alpha_vantage_lambda >> choose_enrich_task
    ingest_alpha_vantage_http >> choose_enrich_task
    choose_enrich_task >> [enrich_stock_data_lambda, enrich_stock_data_http]