import random
from airflow.hooks.postgres_hook import PostgresHook
from airflow.hooks.base_hook import BaseHook
import boto3
import json
import logging
from botocore.config import Config
from datetime import datetime, timedelta
from .config import TOP_50_TICKERS, POSTGRES_CONN_ID

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_model_key(model_type, stock_symbol, feature_set, hyperparameter_tuning, lookback_period,
                       prediction_horizon, formatted_date):
    """
    Generates a unique model key based on provided parameters.

    Args:
        model_type (str): Type of the model ('binary_classification').
        stock_symbol (str): Stock ticker symbol.
        feature_set (str): Feature set used ('basic').
        hyperparameter_tuning (str): Level of hyperparameter tuning ('LOW', 'MEDIUM', 'HIGH').
        lookback_period (int): Lookback period in days.
        prediction_horizon (int): Prediction horizon in days.
        formatted_date (str): Date in 'YYYY-MM-DD' format.

    Returns:
        str: Generated model key.
    """
    model_key = f"{model_type}_{stock_symbol}_{feature_set}_{hyperparameter_tuning}_{lookback_period}_{prediction_horizon}_{formatted_date}"
    return model_key


def invoke_lambda_function(lambda_name: str, payload: dict,invocation_type='Event'):
    """
    Invokes the specified AWS Lambda function asynchronously with the given payload.

    Args:
        lambda_name (str): The name of the Lambda function to invoke.
        payload (dict): The payload to send to the Lambda function.

    Returns:
        dict: A simplified response containing only serializable data.
    """
    # Fetching AWS credentials from Airflow 'aws_default' connection
    connection = BaseHook.get_connection('aws_default')

    # Initialize boto3 client with credentials from Airflow connection
    # Configure timeouts directly in Boto3 client
    client = boto3.client(
        'lambda',
        aws_access_key_id=connection.login,
        aws_secret_access_key=connection.password,
        region_name='us-east-1',  # Adjust region as needed
        config=Config(read_timeout=900, connect_timeout=30)
    )

    try:
        # Serialize payload to JSON and encode to bytes
        serialized_payload = json.dumps(payload).encode('utf-8')

        # Invoke the Lambda function asynchronously
        response = client.invoke(
            FunctionName=lambda_name,
            InvocationType=invocation_type,  # Defaults to Asynchronous invocation
            Payload=serialized_payload
        )

        # Extract only serializable parts of the response
        serializable_response = {
            "StatusCode": response.get('StatusCode'),
            "RequestId": response.get('ResponseMetadata', {}).get('RequestId')
        }

        if serializable_response['StatusCode'] == 202:
            logger.info(f'Lambda "{lambda_name}" invoked asynchronously with payload: {payload}')
        elif serializable_response['StatusCode'] == 200:
            logger.info(f'Lambda "{lambda_name}" succeeded after waiting with payload: {payload}')
        else:
            logger.error(f'Lambda invocation failed for "{lambda_name}" with status code {serializable_response["StatusCode"]}')

        return serializable_response

    except Exception as e:
        logger.error(f"Error invoking Lambda '{lambda_name}': {e}")
        raise

def monitor_lambdas_completion(feature_set, **kwargs):
    ti = kwargs['ti']
    task_ids = [f'invoke_lambda_{stock}_{feature_set}' for stock in TOP_50_TICKERS]
    stock_responses = ti.xcom_pull(task_ids=task_ids)

    all_completed = True
    failed_stocks = []
    for stock, response in zip(TOP_50_TICKERS, stock_responses):
        if not response or response.get('statusCode') != 200:
            logger.error(f"Lambda invocation for {stock} failed or did not return a successful response.")
            all_completed = False
            failed_stocks.append(stock)
        else:
            logger.info(f"Lambda invocation for {stock} completed successfully.")

    if not all_completed:
        raise ValueError(f"Lambda invocations failed for stocks: {', '.join(failed_stocks)}")

    # Optionally, query the database to verify data
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    table_name = 'basic_stock_data' if feature_set == 'basic' else 'enriched_stock_data'
    for stock in TOP_50_TICKERS:
        sql = f"""
        SELECT * FROM {table_name}
        WHERE symbol = %s
        ORDER BY date DESC
        LIMIT 1;
        """
        result = pg_hook.get_first(sql, parameters=(stock,))
        if result:
            logger.info(f"Most recent record for {stock} in {table_name}: {result}")
        else:
            logger.warning(f"No data found for {stock} in {table_name}.")

# New Helper Functions for Random Parameter Selection

def get_random_hyperparameter_tuning():
    return random.choice(['LOW', 'MEDIUM'])

def get_random_feature_set():
    return random.choice(['basic', 'advanced'])

def get_random_lookback_period():
    return random.randint(720,1200)

def get_random_prediction_horizon():
    return random.randint(7,30)

def get_random_parameters(model_type: str):
    """
    Returns a dictionary of randomly selected parameters based on the model type.

    Args:
        model_type (str): Either 'binary_classification' or 'sarimax'.

    Returns:
        dict: A dictionary of parameters.
    """
    if model_type == 'BINARY CLASSIFICATION':
        return {
            'model_type': 'BINARY CLASSIFICATION',
            'hyperparameter_tuning': get_random_hyperparameter_tuning(),
            'feature_set': get_random_feature_set(),
            'lookback_period': get_random_lookback_period(),
            'prediction_horizon': get_random_prediction_horizon()
        }
    elif model_type == 'SARIMAX':
        return {
            'model_type': 'SARIMAX',
            'hyperparameter_tuning': get_random_hyperparameter_tuning(),
            'feature_set': get_random_feature_set(),
            'lookback_period': get_random_lookback_period(),
            'prediction_horizon': get_random_prediction_horizon()
        }
    else:
        raise ValueError("Invalid model_type. Choose 'BINARY CLASSIFICATION' or 'SARIMAX'.")

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
        lambda_name = "make_binary_classification_prediction"
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
                    date_created
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
                date_created
            ) = result

            # Calculate prediction_date
            prediction_date = (datetime.strptime(input_date, '%Y-%m-%d') + timedelta(days=prediction_horizon)).strftime('%Y-%m-%d')

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
                    date_created
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
                date_created
            ) = result

            # Calculate prediction_date
            prediction_date = (datetime.strptime(input_date, '%Y-%m-%d') + timedelta(days=prediction_horizon)).strftime('%Y-%m-%d')

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

        response = invoke_lambda_function("ai_analysis", payload, invocation_type='RequestResponse')
        logger.info(f"AI Analysis response for {stock_symbol}: {response}")
        return response

    except Exception as e:
        logger.error(f"Failed to invoke AI analysis for {stock_symbol}: {str(e)}")
        raise