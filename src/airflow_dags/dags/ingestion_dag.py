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

with DAG(
    'data_ingestion_dag',
    default_args=default_args,
    description='DAG for ingesting data from various sources into PostgreSQL',
    schedule_interval='@hourly',
    start_date=days_ago(1),
    tags=['ingestion'],
) as dag:

    choose_task = BranchPythonOperator(
        task_id='choose_operator',
        python_callable=choose_operator,
    )

    ingest_alpha_vantage_lambda = LambdaInvokeFunctionOperator(
        task_id='ingest_alpha_vantage_lambda',
        function_name='ingest_alpha_vantage_function_name',  # Replace with your Lambda function name
        invocation_type='RequestResponse',
    )

    ingest_alpha_vantage_http = SimpleHttpOperator(
        task_id='ingest_alpha_vantage_http',
        method='POST',
        http_conn_id='ingest_alpha_vantage_service',
        endpoint='/invoke',
        data='{}',
        headers={"Content-Type": "application/json"},
    )

    choose_task >> [ingest_alpha_vantage_lambda, ingest_alpha_vantage_http]