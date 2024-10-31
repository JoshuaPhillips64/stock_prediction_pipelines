from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from common.config import TOP_50_TICKERS
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
    'execution_timeout': timedelta(minutes=120),
}

dag_name = 'generate_stock_prediction_dag'
description = 'Generate stock predictions by cycling through 50 stock tickers with random parameters'
schedule = '0 7 * * *'  # 6 AM UTC (adjust as needed)
start_date = days_ago(1)
catchup = False

# Function to generate and push parameters to XCom
def generate_params(stock, **kwargs):
    choose_model_type = random.choice(['SARIMAX', 'BINARY CLASSIFICATION'])
    params = get_random_parameters(choose_model_type)
    logger.info(f"Generated parameters for {stock}: {params}")
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
        for stock in TOP_50_TICKERS:
            group_id = f'generate_full_pipeline_{stock}'
            with TaskGroup(group_id=group_id) as stock_task_group:
                # Task to generate parameters and save to XCom
                generate_params_task = PythonOperator(
                    task_id='generate_params',
                    python_callable=generate_params,
                    op_kwargs={'stock': stock},
                )

                # Ingest Data Task
                def ingest_data(**kwargs):
                    ti = kwargs['ti']
                    params = ti.xcom_pull(key='params', task_ids=f'process_stocks.{group_id}.generate_params')
                    execution_date = kwargs['ds']  # 'YYYY-MM-DD'
                    lookback_period = params['lookback_period']
                    start_date = (datetime.strptime(execution_date, '%Y-%m-%d') - timedelta(days=lookback_period)).strftime('%Y-%m-%d')
                    end_date = execution_date
                    logger.info(f"Ingesting data for {stock}: {start_date} to {end_date}")
                    invoke_lambda_ingest(
                        stock_symbol=stock,
                        start_date=start_date,
                        end_date=end_date,
                        feature_set=params['feature_set']
                    )

                ingest_task = PythonOperator(
                    task_id='ingest_data',
                    python_callable=ingest_data,
                )

                # Train Model Task
                def train_model(**kwargs):
                    ti = kwargs['ti']
                    params = ti.xcom_pull(key='params', task_ids=f'process_stocks.{group_id}.generate_params')
                    execution_date = kwargs['ds']
                    logger.info(f"Training model for {stock} on {execution_date}")
                    invoke_lambda_train(
                        model_type=params['model_type'],
                        stock_symbol=stock,
                        input_date=execution_date,
                        hyperparameter_tuning=params['hyperparameter_tuning'],
                        feature_set=params['feature_set'],
                        lookback_period=params['lookback_period'],
                        prediction_horizon=params['prediction_horizon']
                    )

                train_task = PythonOperator(
                    task_id='train_model',
                    python_callable=train_model,
                )

                # Prediction Task
                def make_prediction(**kwargs):
                    ti = kwargs['ti']
                    params = ti.xcom_pull(key='params', task_ids=f'process_stocks.{group_id}.generate_params')
                    execution_date = kwargs['ds']
                    logger.info(f"Making prediction for {stock} on {execution_date}")
                    invoke_lambda_predict(
                        model_type=params['model_type'],
                        stock_symbol=stock,
                        input_date=execution_date,
                        hyperparameter_tuning=params['hyperparameter_tuning'],
                        feature_set=params['feature_set'],
                        lookback_period=params['lookback_period'],
                        prediction_horizon=params['prediction_horizon']
                    )

                predict_task = PythonOperator(
                    task_id='make_prediction',
                    python_callable=make_prediction,
                )

                # AI Analysis Task
                def trigger_ai_analysis_task_callable(**kwargs):
                    ti = kwargs['ti']
                    params = ti.xcom_pull(key='params', task_ids=f'process_stocks.{group_id}.generate_params')
                    execution_date = kwargs['ds']
                    logger.info(f"Triggering AI analysis for {stock} on {execution_date}")
                    invoke_lambda_ai_analysis(
                        stock_symbol=stock,
                        model_type=params['model_type'],
                        input_date=execution_date,
                        feature_set=params['feature_set'],
                        hyperparameter_tuning=params['hyperparameter_tuning'],
                        lookback_period=params['lookback_period'],
                        prediction_horizon=params['prediction_horizon']
                    )

                ai_analysis_task = PythonOperator(
                    task_id='trigger_ai_analysis',
                    python_callable=trigger_ai_analysis_task_callable,
                )

                # Define task dependencies within the stock's TaskGroup
                generate_params_task >> ingest_task >> train_task >> predict_task >> ai_analysis_task

    # If you want all stock TaskGroups to run in parallel, you don't need to set dependencies between them.
    # The current implementation sets them to run sequentially, which might not be ideal.
    # To enable parallelism, remove any dependencies between TaskGroups.

    # Example: Remove dependencies for parallel execution
    # process_stocks_group

    # If you have downstream tasks after all stocks are processed, define them here
    # For example:
    # final_task = DummyOperator(task_id='final_task')
    # process_stocks_group >> final_task