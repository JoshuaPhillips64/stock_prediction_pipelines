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
    'trigger_rule': 'all_done'  # THIS MEANS IT WILL KEEP RUNNING UNTIL ALL TASKS ARE DONE REGARDLESS OF FAILURE
}

# Define task-specific timeout
TASK_TIMEOUT = timedelta(minutes=15)

dag_name = 'generate_stock_prediction_dag'
description = 'Generate stock predictions by cycling through 50 stock tickers with random parameters'
schedule = '0 7 * * *'
start_date = days_ago(1)
catchup = False


def generate_params(stock, **context):
    """Generate parameters for a stock and push to XCom"""
    choose_model_type = random.choice(['SARIMAX', 'BINARY CLASSIFICATION'])
    params = get_random_parameters(choose_model_type)
    params['stock'] = stock  # Include stock in params for validation
    logger.info(f"Generated parameters for {stock}: {params}")
    context['task_instance'].xcom_push(key=f'params_{stock}', value=params)
    return params


def get_stock_params(stock, task_instance):
    """Helper function to get parameters for a specific stock"""
    params = task_instance.xcom_pull(key=f'params_{stock}')
    if not params or params.get('stock') != stock:
        raise ValueError(f"Parameters not found or mismatch for stock {stock}")
    return params


with DAG(
        dag_name,
        default_args=default_args,
        description=description,
        schedule_interval=schedule,
        start_date=start_date,
        catchup=catchup,
        max_active_runs=1,
        concurrency=1,
) as dag:
    previous_task_group = None

    for stock in TOP_50_TICKERS:
        group_id = f'generate_full_pipeline_{stock}'

        with TaskGroup(group_id=group_id) as stock_task_group:
            # Generate parameters task
            generate_params_task = PythonOperator(
                task_id='generate_params',
                python_callable=generate_params,
                op_kwargs={'stock': stock},
                execution_timeout=TASK_TIMEOUT,
            )

            # Ingest Data Task
            def ingest_data(stock=stock, **context):
                params = get_stock_params(stock, context['task_instance'])
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
                execution_timeout=TASK_TIMEOUT,
            )

            # Train Model Task
            def train_model(stock=stock, **context):
                params = get_stock_params(stock, context['task_instance'])
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
                execution_timeout=TASK_TIMEOUT,
            )

            # Prediction Task
            def make_prediction(stock=stock, **context):
                params = get_stock_params(stock, context['task_instance'])
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
                execution_timeout=TASK_TIMEOUT,
            )


            # AI Analysis Task
            def trigger_ai_analysis(stock=stock, **context):
                params = get_stock_params(stock, context['task_instance'])
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
                execution_timeout=TASK_TIMEOUT,
            )

            # Define task dependencies within the stock's TaskGroup
            generate_params_task >> ingest_task >> train_task >> predict_task >> ai_analysis_task

        # Enforce sequential execution of stock task groups
        if previous_task_group:
            previous_task_group >> stock_task_group
        previous_task_group = stock_task_group