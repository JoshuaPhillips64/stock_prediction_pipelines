from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
from common.config import TOP_50_TICKERS
from common.helpers import invoke_lambda_function, monitor_lambdas_completion

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag_name = 'nightly_load_basic_stock_data'
description = 'Nightly load of top 50 stock tickers into basic_stock_data table'
schedule = '0 1 * * *'  # Daily at 1 AM
start_date = days_ago(1)
catchup = False
feature_set = 'basic'

def invoke_lambda_task_wrapper(stock_ticker, feature_set, **kwargs):
    execution_date = kwargs['execution_date']
    date_str = (execution_date - timedelta(days=1)).strftime('%Y-%m-%d')
    return invoke_lambda_function(stock_ticker, date_str, date_str, feature_set)

with DAG(
    dag_name,
    default_args=default_args,
    description=description,
    schedule_interval=schedule,
    start_date=start_date,
    catchup=catchup,
    max_active_runs=1
) as dag:
    with TaskGroup('invoke_lambdas') as invoke_lambdas_group:
        for stock in TOP_50_TICKERS:
            invoke_lambda_task = PythonOperator(
                task_id=f'invoke_lambda_{stock}_{feature_set}',
                python_callable=invoke_lambda_task_wrapper,
                op_kwargs={
                    'stock_ticker': stock,
                    'feature_set': feature_set
                }
            )

    monitor_completion_task = PythonOperator(
        task_id='monitor_lambdas_completion',
        python_callable=monitor_lambdas_completion,
        op_kwargs={'feature_set': feature_set}
    )

    invoke_lambdas_group >> monitor_completion_task