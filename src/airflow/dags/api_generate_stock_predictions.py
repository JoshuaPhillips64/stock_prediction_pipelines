from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
import logging

from common.helpers import (
    invoke_lambda_ingest,
    invoke_lambda_train,
    invoke_lambda_predict,
    invoke_lambda_ai_analysis,
)

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
    'trigger_rule': 'all_done'  # Continue running even if some tasks fail
}

# Define task-specific timeout
TASK_TIMEOUT = timedelta(minutes=15)

dag_name = 'dynamic_api_generate_stock_prediction_dag'
description = 'Generate stock predictions based on parameters provided via API call'
schedule = None  # Since it will be triggered externally
start_date = days_ago(1)
catchup = False

with DAG(
    dag_id=dag_name,
    default_args=default_args,
    description=description,
    schedule_interval=schedule,
    start_date=start_date,
    catchup=catchup,
    max_active_runs=1,
    concurrency=1,
) as dag:

    def get_settings_for_job(**context):
        """Retrieve config from dag_run.conf"""
        settings_for_job = context['dag_run'].conf or {}
        if not settings_for_job:
            raise ValueError("No configuration provided in dag_run.conf")
        logger.info(f"Received settings: {settings_for_job}")
        return settings_for_job

    get_settings_for_job_task = PythonOperator(
        task_id='get_settings_for_job',
        python_callable=get_settings_for_job,
        provide_context=True,
    )

    def extract_stocks(**context):
        """Extract stock information from configuration"""
        settings_for_job = context['task_instance'].xcom_pull(task_ids='get_settings_for_job')
        stocks = settings_for_job.get('stocks')
        if not stocks:
            raise ValueError("No stocks provided in the configuration")
        logger.info(f"Extracted stocks: {stocks}")
        return stocks

    extract_stocks_task = PythonOperator(
        task_id='extract_stocks',
        python_callable=extract_stocks,
        provide_context=True,
    )

    # Function to create tasks for each stock
    def create_stock_tasks(**context):
        stocks = context['task_instance'].xcom_pull(task_ids='extract_stocks')
        if not stocks:
            raise ValueError("No stocks to process")

        previous_task_group = None

        for stock_info in stocks:
            stock = stock_info.get('stock')
            params = stock_info.get('params')
            if not stock or not params:
                raise ValueError(f"Stock or parameters missing for stock info: {stock_info}")

            group_id = f'process_{stock}'

            with TaskGroup(group_id=group_id) as stock_task_group:

                # Ingest Data Task
                def ingest_data(**context):
                    execution_date = context['ds']
                    lookback_period = params['lookback_period']
                    start_date = (datetime.strptime(execution_date, '%Y-%m-%d') -
                                  timedelta(days=lookback_period)).strftime('%Y-%m-%d')
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
                    provide_context=True,
                    execution_timeout=TASK_TIMEOUT,
                )

                # Train Model Task
                def train_model(**context):
                    execution_date = context['ds']

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
                    provide_context=True,
                    execution_timeout=TASK_TIMEOUT,
                )

                # Prediction Task
                def make_prediction(**context):
                    execution_date = context['ds']

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
                    provide_context=True,
                    execution_timeout=TASK_TIMEOUT,
                )

                # AI Analysis Task
                def trigger_ai_analysis(**context):
                    execution_date = context['ds']

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
                    python_callable=trigger_ai_analysis,
                    provide_context=True,
                    execution_timeout=TASK_TIMEOUT,
                )

                # Define task dependencies within the stock's TaskGroup
                ingest_task >> train_task >> predict_task >> ai_analysis_task

            # Enforce sequential execution of stock task groups
            if previous_task_group:
                previous_task_group >> stock_task_group
            else:
                # Start after extract_stocks_task
                extract_stocks_task >> stock_task_group

            previous_task_group = stock_task_group

    create_stock_tasks_task = PythonOperator(
        task_id='create_stock_tasks',
        python_callable=create_stock_tasks,
        provide_context=True,
    )

    # Define the overall DAG dependencies
    get_settings_for_job_task >> extract_stocks_task >> create_stock_tasks_task