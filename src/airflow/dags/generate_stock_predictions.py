from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from common.config import TOP_50_TICKERS, POSTGRES_CONN_ID
from common.helpers import (
    invoke_lambda_ingest,
    invoke_lambda_train,
    invoke_lambda_predict,
    invoke_lambda_ai_analysis,
    get_random_parameters
)
import logging
import random

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=120),  # Adjust as needed
}

dag_name = 'generate_stock_prediction_dag'
description = 'Generate stock predictions by cycling through 50 stock tickers with random parameters'
schedule = '0 6 * * *'  # 5 AM CST is 11 AM UTC  # Adjust schedule as needed
start_date = days_ago(1)
catchup = False

# Function to generate and push parameters to XCom
def generate_params(stock, **kwargs):
    choose_model_type = random.choice(['SARIMAX', 'BINARY CLASSIFICATION'])
    params = get_random_parameters(choose_model_type)
    kwargs['ti'].xcom_push(key='params', value=params)

with DAG(
    dag_name,
    default_args=default_args,
    description=description,
    schedule_interval=schedule,
    start_date=start_date,
    catchup=catchup,
    max_active_runs=1,
    concurrency=4,  # Adjust based on your Airflow setup and AWS Lambda concurrency
) as dag:
    with TaskGroup('process_stocks') as process_stocks_group:
        previous_task_group = None
        for stock in TOP_50_TICKERS:
            stock_task_group = TaskGroup(group_id=f'generate_full_pipeline_{stock}')
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

            # Task to generate parameters and save to XCom
            generate_params_task = PythonOperator(
                task_id=f'generate_params_{stock}',
                python_callable=generate_params,
                op_kwargs={'stock': stock}
            )


            # Ingest Data Task
            def ingest_data(**kwargs):
                params = kwargs['ti'].xcom_pull(key='params',
                                                task_ids=f'process_stocks.generate_full_pipeline_{stock}.generate_params_{stock}')
                invoke_lambda_ingest(
                    stock_symbol=stock,
                    start_date=(datetime.now() - timedelta(days=params['lookback_period'])).strftime('%Y-%m-%d'),
                    end_date=yesterday,
                    feature_set=params['feature_set']
                )


            ingest_task = PythonOperator(
                task_id=f'ingest_data_{stock}',
                python_callable=ingest_data
            )

            # Train Model
            def train_model(**kwargs):
                params = kwargs['ti'].xcom_pull(key='params',
                                                task_ids=f'process_stocks.generate_full_pipeline_{stock}.generate_params_{stock}')
                return invoke_lambda_train(
                    model_type=params['model_type'],
                    stock_symbol=stock,
                    input_date=yesterday,
                    hyperparameter_tuning=params['hyperparameter_tuning'],
                    feature_set=params['feature_set'],
                    lookback_period=params['lookback_period'],
                    prediction_horizon=params['prediction_horizon']
                )

            train_task = PythonOperator(
                task_id=f'train_model_{stock}',
                python_callable=train_model
            )

            # Make Prediction
            def make_prediction(**kwargs):
                params = kwargs['ti'].xcom_pull(key='params',
                                                task_ids=f'process_stocks.generate_full_pipeline_{stock}.generate_params_{stock}')
                return invoke_lambda_predict(
                    model_type=params['model_type'],
                    stock_symbol=stock,
                    input_date=yesterday,
                    hyperparameter_tuning=params['hyperparameter_tuning'],
                    feature_set=params['feature_set'],
                    lookback_period=params['lookback_period'],
                    prediction_horizon=params['prediction_horizon']
                )

            predict_task = PythonOperator(
                task_id=f'make_prediction_{stock}',
                python_callable=make_prediction
            )


            # Trigger AI Analysis with Data Preparation
            def trigger_ai_analysis_task_callable(**kwargs):
                params = kwargs['ti'].xcom_pull(key='params',
                                                task_ids=f'process_stocks.generate_full_pipeline_{stock}.generate_params_{stock}')
                return invoke_lambda_ai_analysis(
                    stock_symbol=stock,
                    model_type=params['model_type'],
                    input_date=yesterday,
                    feature_set=params['feature_set'],
                    hyperparameter_tuning=params['hyperparameter_tuning'],
                    lookback_period=params['lookback_period'],
                    prediction_horizon=params['prediction_horizon']
                )


            ai_analysis_task = PythonOperator(
                task_id=f'trigger_ai_analysis_{stock}',
                python_callable=trigger_ai_analysis_task_callable
            )

            # Define task dependencies within the stock's TaskGroup
            generate_params_task >> ingest_task >> train_task >> predict_task >> ai_analysis_task

            if previous_task_group:
                previous_task_group >> stock_task_group
            previous_task_group = stock_task_group

        process_stocks_group