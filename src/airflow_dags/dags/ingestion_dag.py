from airflow import DAG
from airflow.providers.amazon.aws.operators.lambda_function import LambdaInvokeFunctionOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
}

with DAG(
    'data_ingestion_dag',
    default_args=default_args,
    description='DAG for ingesting data from various sources into PostgreSQL',
    schedule_interval='@hourly',
    start_date=days_ago(1),
    tags=['ingestion'],
) as dag:

    ingest_alpha_vantage = LambdaInvokeFunctionOperator(
        task_id='ingest_alpha_vantage',
        function_name='${var.environment}-ingest-alpha-vantage',
        invocation_type='RequestResponse',
    )

    [ingest_alpha_vantage]