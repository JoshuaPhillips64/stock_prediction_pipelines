from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from common.config import TOP_50_TICKERS
from common.helpers import invoke_lambda_function, get_random_parameters, generate_model_key
import logging
import json
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
schedule = '0 11 * * *'  # 5 AM CST is 11 AM UTC  # Adjust schedule as needed
start_date = days_ago(1)
catchup = False


def invoke_lambda_ingest(stock_symbol: str, start_date, end_date, feature_set: str, **kwargs):
    """
    Invokes the ingest_stock_data Lambda function.
    """
    payload = {
        "body": json.dumps({
            "stocks": [stock_symbol],
            "start_date": start_date,  # Adjust based on requirements or randomize
            "end_date": end_date,
            "feature_set": feature_set
        })
    }

    # Log the exact payload
    logger.info(f"Invoking ingest_stock_data Lambda for {stock_symbol} with payload: {json.dumps(payload, indent=2)}")

    response = invoke_lambda_function("ingest_stock_data", payload)
    return response

def invoke_lambda_train(model_type: str, stock_symbol: str, input_date: str, hyperparameter_tuning: str, feature_set: str, lookback_period: int, prediction_horizon: int, **kwargs):
    """
    Invokes the train_* Lambda function based on model type.
    """
    if model_type == 'SARIMAX':
        lambda_name = "train_sarimax_model"
    elif model_type == 'BINARY CLASSIFICATION':
        lambda_name = "train_binary_classification_model"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_key = generate_model_key(
        model_type=model_type,
        stock_symbol=stock_symbol,
        feature_set=feature_set,
        hyperparameter_tuning=hyperparameter_tuning,
        lookback_period=lookback_period,
        prediction_horizon=prediction_horizon,
        formatted_date=input_date
    )

    payload = {
        "body": json.dumps({
            "model_key": model_key,
            "stock_symbol": stock_symbol,
            "input_date": input_date,
            "hyperparameter_tuning": hyperparameter_tuning,
            "feature_set": feature_set,
            "lookback_period": lookback_period,
            "prediction_horizon": prediction_horizon
        })
    }

    # Log the exact payload
    logger.info(f"Invoking {lambda_name} Lambda for {stock_symbol} with payload: {json.dumps(payload, indent=2)}")

    response = invoke_lambda_function(lambda_name, payload)
    return response

def invoke_lambda_predict(model_type: str, stock_symbol: str, input_date: str, hyperparameter_tuning: str, feature_set: str, lookback_period: int, prediction_horizon: int, **kwargs):
    """
    Invokes the make_*_prediction Lambda function based on model type.
    """
    if model_type == 'SARIMAX':
        lambda_name = "make_sarimax_prediction"
    elif model_type == 'BINARY CLASSIFICATION':
        lambda_name = "make_binary_prediction"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_key = generate_model_key(
        model_type=model_type,
        stock_symbol=stock_symbol,
        feature_set=feature_set,
        hyperparameter_tuning=hyperparameter_tuning,
        lookback_period=lookback_period,
        prediction_horizon=prediction_horizon,
        formatted_date=input_date
    )

    payload = {
        "body": json.dumps({
            "model_key": model_key,
            "stock_symbol": stock_symbol,
            "input_date": input_date,
            "hyperparameter_tuning": hyperparameter_tuning,
            "feature_set": feature_set,
            "lookback_period": lookback_period,
            "prediction_horizon": prediction_horizon
        })
    }

    # Log the exact payload
    logger.info(f"Invoking {lambda_name} Lambda for {stock_symbol} with payload: {json.dumps(payload, indent=2)}")

    response = invoke_lambda_function(lambda_name, payload)
    return response

def invoke_lambda_ai_analysis(prediction_data: list, **kwargs):
    """
    Invokes the trigger_ai_analysis Lambda function.
    """
    payload = {
        "body": json.dumps({
            "predictions": prediction_data
        })
    }

    # Log the exact payload
    logger.info(f"Invoking trigger_ai_analysis Lambda with payload: {json.dumps(payload, indent=2)}")

    response = invoke_lambda_function("trigger_ai_analysis", payload)
    return response

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
            stock_task_group = TaskGroup(group_id=f'generate_full_pipeline_{stock}')

            choose_model_type = random.choice(['SARIMAX', 'BINARY CLASSIFICATION'])
            params = get_random_parameters(choose_model_type)

            # Ingest Data
            ingest_task = PythonOperator(
                task_id='ingest_data',
                python_callable=invoke_lambda_ingest,
                op_kwargs={
                    'stock_ticker': stock,
                    'start_date': (datetime.strptime('{{ ds }}', '%Y-%m-%d') - timedelta(
                        days=params['lookback_period'])).strftime('%Y-%m-%d'),
                    'end_date': '{{ yesterday_ds }}',
                    'feature_set': params['feature_set']
                },
                provide_context=True
            )

            # Train Model
            def train_model(**kwargs):
                return invoke_lambda_train(
                    model_type=params['model_type'],
                    stock_symbol=stock,
                    input_date="{{ yesterday_ds }}",
                    hyperparameter_tuning=params['hyperparameter_tuning'],
                    feature_set=params['feature_set'],
                    lookback_period=params['lookback_period'],
                    prediction_horizon=params['prediction_horizon']
                )

            train_task = PythonOperator(
                task_id='train_model',
                python_callable=train_model,
                provide_context=True
            )

            # Make Prediction
            def make_prediction(**kwargs):
                ti = kwargs['ti']
                params = ti.xcom_pull(key='model_parameters', task_ids='prepare_parameters')
                if not params:
                    raise ValueError("No parameters found in XCom.")
                return invoke_lambda_predict(
                    model_type=params['model_type'],
                    stock_symbol=stock,
                    input_date="{{ ds }}",
                    hyperparameter_tuning=params['hyperparameter_tuning'],
                    feature_set=params['feature_set'],
                    lookback_period=params['lookback_period'],
                    prediction_horizon=params['prediction_horizon']
                )

            predict_task = PythonOperator(
                task_id='make_prediction',
                python_callable=make_prediction,
                provide_context=True
            )

            # Trigger AI Analysis
            def trigger_ai(**kwargs):
                # Assuming prediction_data is obtained from previous tasks or another source
                prediction_data = []  # Replace with actual data retrieval if needed
                return invoke_lambda_ai_analysis(prediction_data)

            ai_analysis_task = PythonOperator(
                task_id='trigger_ai_analysis',
                python_callable=trigger_ai,
                provide_context=True
            )

            # Define task dependencies within the stock's TaskGroup
            ingest_task >> train_task >> predict_task >> ai_analysis_task

            # Add the stock's TaskGroup to the main TaskGroup
            process_stocks_group >> stock_task_group

    # Define DAG dependencies
    process_stocks_group