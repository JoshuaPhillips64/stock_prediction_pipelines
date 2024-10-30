from airflow import DAG
from airflow.hooks.postgres_hook import PostgresHook
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from common.config import TOP_50_TICKERS, POSTGRES_CONN_ID
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
schedule = '0 6 * * *'  # 5 AM CST is 11 AM UTC  # Adjust schedule as needed
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

    response = invoke_lambda_function("ingest_stock_data", payload,invocation_type='RequestResponse')
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

    response = invoke_lambda_function(lambda_name, payload,invocation_type='RequestResponse')
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

    response = invoke_lambda_function(lambda_name, payload,invocation_type='RequestResponse')
    return response

def get_model_data(stock_symbol: str, model_type: str, input_date: str, feature_set: str, hyperparameter_tuning: str, lookback_period: int, prediction_horizon: int):
    """
    Retrieves model data from the PostgreSQL database using PostgresHook and constructs prediction_data.
    """
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    prediction_data = []

    try:
        # Generate model_key based on the parameters
        model_key = generate_model_key(
            model_type=model_type,
            stock_symbol=stock_symbol,
            feature_set=feature_set,
            hyperparameter_tuning=hyperparameter_tuning,
            lookback_period=lookback_period,
            prediction_horizon=prediction_horizon,
            formatted_date=input_date
        )

        if model_type == 'SARIMAX':
            table_name = 'trained_models'  # Replace with your actual table name
            sql = f"""
                SELECT 
                    model_key,
                    prediction_explanation,
                    prediction_rmse,
                    prediction_mae,
                    prediction_mape,
                    prediction_confidence_score,
                    feature_importance,
                    model_parameters,
                    last_known_price,
                    predicted_amount,
                    predictions_json,
                    model_location,
                    date_created,
                    prediction_horizon
                FROM {table_name}
                WHERE model_key = %s
                LIMIT 1;
            """

            result = pg_hook.get_first(sql, parameters=(model_key,))
            if not result:
                raise Exception(f"Trained SARIMAX model with key {model_key} not found.")

            (
                db_model_key,
                prediction_explanation,
                prediction_rmse,
                prediction_mae,
                prediction_mape,
                prediction_confidence_score,
                feature_importance,
                model_parameters,
                last_known_price,
                predicted_amount,
                predictions_json,
                model_location,
                date_created,
                prediction_horizon_db
            ) = result

            # Calculate prediction_date
            prediction_date = (datetime.strptime(input_date, '%Y-%m-%d') + timedelta(days=prediction_horizon_db)).strftime('%Y-%m-%d')

            prediction_data.append({
                'model_key': db_model_key,
                'symbol': stock_symbol,
                'prediction_date': prediction_date,
                'prediction_explanation': prediction_explanation or '',
                'prediction_rmse': prediction_rmse,
                'prediction_mae': prediction_mae,
                'prediction_mape': prediction_mape,
                'prediction_confidence_score': prediction_confidence_score,
                'confusion_matrix': json.dumps([]),  # Not applicable for regression
                'feature_importance': json.dumps(feature_importance or {}),
                'model_parameters': json.dumps(model_parameters or {}),
                'predicted_movement': None,  # Not applicable for regression
                'predicted_price': None,  # Not directly applicable
                'prediction_probability': None,  # Not applicable for regression
                'last_known_price': last_known_price,
                'predicted_amount': predicted_amount,
                'predictions_json': json.dumps(predictions_json or {}),
                'model_location': model_location,
                'date_created': date_created.strftime('%Y-%m-%d %H:%M:%S') if date_created else None
            })

        elif model_type == 'BINARY CLASSIFICATION':
            table_name = 'trained_models_binary'  # Replace with your actual table name
            sql = f"""
                SELECT 
                    model_key,
                    prediction_explanation,
                    prediction_accuracy,
                    prediction_precision,
                    prediction_recall,
                    prediction_f1_score,
                    prediction_roc_auc,
                    confusion_matrix,
                    feature_importance,
                    model_parameters,
                    predicted_movement,
                    predicted_price,
                    prediction_probability,
                    last_known_price,
                    predictions_json,
                    model_location,
                    date_created,
                    prediction_horizon
                FROM {table_name}
                WHERE model_key = %s
                LIMIT 1;
            """

            result = pg_hook.get_first(sql, parameters=(model_key,))
            if not result:
                raise Exception(f"Trained Binary Classification model with key {model_key} not found.")

            (
                db_model_key,
                prediction_explanation,
                prediction_accuracy,
                prediction_precision,
                prediction_recall,
                prediction_f1_score,
                prediction_roc_auc,
                confusion_matrix,
                feature_importance,
                model_parameters,
                predicted_movement,
                predicted_price,
                prediction_probability,
                last_known_price,
                predictions_json,
                model_location,
                date_created,
                prediction_horizon_db
            ) = result

            # Calculate prediction_date
            prediction_date = (datetime.strptime(input_date, '%Y-%m-%d') + timedelta(days=prediction_horizon_db)).strftime('%Y-%m-%d')

            prediction_data.append({
                'model_key': db_model_key,
                'symbol': stock_symbol,
                'prediction_date': prediction_date,
                'prediction_explanation': prediction_explanation or '',
                'prediction_accuracy': prediction_accuracy,
                'prediction_precision': prediction_precision,
                'prediction_recall': prediction_recall,
                'prediction_f1_score': prediction_f1_score,
                'prediction_roc_auc': prediction_roc_auc,
                'confusion_matrix': json.dumps(confusion_matrix or []),
                'feature_importance': json.dumps(feature_importance or {}),
                'model_parameters': json.dumps(model_parameters or {}),
                'predicted_movement': predicted_movement,
                'predicted_price': predicted_price,
                'prediction_probability': prediction_probability,
                'last_known_price': last_known_price,
                'predicted_amount': None,  # Not applicable for classification
                'predictions_json': json.dumps(predictions_json or {}),
                'model_location': model_location,
                'date_created': date_created.strftime('%Y-%m-%d %H:%M:%S') if date_created else None
            })

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.info(f"Prediction data for {stock_symbol}: {json.dumps(prediction_data, indent=2)}")
        return prediction_data

    except Exception as e:
        logger.error(f"Error retrieving model data for {stock_symbol}: {str(e)}")
        raise

def invoke_lambda_ai_analysis(stock_symbol: str, model_type: str, input_date: str, feature_set: str, hyperparameter_tuning: str, lookback_period: int, prediction_horizon: int, **kwargs):
    """
    Retrieves prediction data from the database and invokes the AI analysis Lambda function.
    """
    try:
        prediction_data = get_model_data(
            stock_symbol=stock_symbol,
            model_type=model_type,
            input_date=input_date,
            feature_set=feature_set,
            hyperparameter_tuning=hyperparameter_tuning,
            lookback_period=lookback_period,
            prediction_horizon=prediction_horizon
        )
        if not prediction_data:
            raise Exception(f"No prediction data available for stock {stock_symbol}.")

        payload = {
            "body": json.dumps({
                "predictions": prediction_data
            })
        }

        # Log the exact payload
        logger.info(f"Invoking trigger_ai_analysis Lambda for {stock_symbol} with payload: {json.dumps(payload, indent=2)}")

        response = invoke_lambda_function("trigger_ai_analysis", payload, invocation_type='RequestResponse')
        logger.info(f"AI Analysis response for {stock_symbol}: {response}")
        return response

    except Exception as e:
        logger.error(f"Failed to invoke AI analysis for {stock_symbol}: {str(e)}")
        raise

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

            choose_model_type = random.choice(['SARIMAX','BINARY CLASSIFICATION'])
            params = get_random_parameters(choose_model_type)

            # Ingest Data
            ingest_task = PythonOperator(
                task_id=f'ingest_data_{stock}',
                python_callable=invoke_lambda_ingest,
                op_kwargs={
                    'stock_symbol': stock,
                    'start_date': (datetime.now() - timedelta(
                        days=params['lookback_period'])).strftime('%Y-%m-%d'),
                    'end_date': '{{ yesterday_ds }}',
                    'feature_set': params['feature_set']
                }
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
                task_id=f'train_model_{stock}',
                python_callable=train_model
            )

            # Make Prediction
            def make_prediction(**kwargs):
                return invoke_lambda_predict(
                    model_type=params['model_type'],
                    stock_symbol=stock,
                    input_date="{{ yesterday_ds }}",
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
                return invoke_lambda_ai_analysis(
                    stock_symbol=stock,
                    model_type=params['model_type'],
                    input_date="{{ yesterday_ds }}",
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
            ingest_task >> train_task >> predict_task >> ai_analysis_task

            # Set cross-task-group dependencies (ensuring sequential stock execution)
            if previous_task_group:
                previous_task_group >> stock_task_group
            previous_task_group = stock_task_group

    # Define DAG dependencies
    process_stocks_group