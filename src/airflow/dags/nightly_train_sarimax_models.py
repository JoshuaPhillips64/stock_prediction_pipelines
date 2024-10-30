from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
from common.config import TOP_50_TICKERS
from common.helpers import (
    invoke_lambda_function,
    monitor_lambdas_completion,
    get_random_parameters
)
import json

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag_name = 'daily_train_sarimax_model'
description = 'Daily training of SARIMAX models for top 50 stock tickers'
schedule = '0 11 * * *'  # 5 AM CST is 11 AM UTC
start_date = days_ago(1)
catchup = False


def generate_model_key(model_type, stock_symbol, feature_set, hyperparameter_tuning, lookback_period,
                       prediction_horizon, formatted_date):
    """
    Generates a unique model key based on provided parameters.

    Args:
        model_type (str): Type of the model ('sarimax').
        stock_symbol (str): Stock ticker symbol.
        feature_set (str): Feature set used ('advanced').
        hyperparameter_tuning (str): Level of hyperparameter tuning ('LOW', 'MEDIUM', 'HIGH').
        lookback_period (int): Lookback period in days.
        prediction_horizon (int): Prediction horizon in days.
        formatted_date (str): Date in 'YYYY-MM-DD' format.

    Returns:
        str: Generated model key.
    """
    model_key = f"{model_type}_{stock_symbol}_{feature_set}_{hyperparameter_tuning}_{lookback_period}_{prediction_horizon}_{formatted_date}"
    return model_key


def invoke_sarimax_lambda(stock_ticker: str, input_date: str, **kwargs):
    """
    Invokes the train_sarimax_model lambda with the provided parameters.
    """
    params = get_random_parameters('SARIMAX')

    # Generate model_key using the provided function
    model_key = generate_model_key(
        model_type="SARIMAX",
        stock_symbol=stock_ticker,
        feature_set=params['feature_set'],
        hyperparameter_tuning=params['hyperparameter_tuning'],
        lookback_period=params['lookback_period'],
        prediction_horizon=params['prediction_horizon'],
        formatted_date=input_date
    )

    payload = {
        "model_key": model_key,
        "stock_symbol": stock_ticker,
        "input_date": input_date,
        "hyperparameter_tuning": params['hyperparameter_tuning'],
        "feature_set": params['feature_set'],
        "lookback_period": params['lookback_period'],
        "prediction_horizon": params['prediction_horizon']
    }

    return invoke_lambda_function("train_sarimax_model", payload)


with DAG(
        dag_name,
        default_args=default_args,
        description=description,
        schedule_interval=schedule,
        start_date=start_date,
        catchup=catchup,
        max_active_runs=1,
        concurrency=4,
) as dag:
    prepare_parameters_task = PythonOperator(
        task_id='prepare_parameters',
        python_callable=prepare_parameters,
        op_kwargs={'model_type': 'SARIMAX'}
    )

    with TaskGroup('invoke_lambdas') as invoke_lambdas_group:
        for stock in TOP_50_TICKERS:
            invoke_lambda_task = PythonOperator(
                task_id=f'invoke_sarimax_train_lambda_{stock}',
                python_callable=invoke_sarimax_lambda,
                op_kwargs={
                    'stock_ticker': stock,
                    'input_date': "{{ yesterday_ds }}"
                },
                provide_context=True
            )

    invoke_lambdas_group