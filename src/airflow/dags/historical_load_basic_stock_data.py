from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
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

dag_name = 'historical_load_basic_stock_data'
description = 'Historical load of top 50 stock tickers into basic_stock_data table'
schedule = None  # One-time run
start_date = datetime(2024, 10, 1)
catchup = False
feature_set = 'basic'

def invoke_lambda_task_wrapper(stock_ticker, feature_set, **kwargs):
    start_date_range = '2020-01-01'
    end_date_range = datetime.now().strftime('%Y-%m-%d')
    return invoke_lambda_function(stock_ticker, start_date_range, end_date_range, feature_set)

with DAG(
    dag_name,
    default_args=default_args,
    description=description,
    schedule_interval=schedule,
    start_date=start_date,
    catchup=catchup,
    max_active_runs=1,
    concurrency=4,
    is_paused_upon_creation=True
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