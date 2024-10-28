import json
import boto3
import time
from typing import List, Dict, Any, Optional
from config import Config
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a Lambda client
lambda_client = boto3.client('lambda')

def call_lambda(function_name: str, payload: Optional[Dict[str, Any]] = None):
    retry_count = 0
    max_retries = 0

    if payload is None:
        payload = {}

    while retry_count <= max_retries:
        try:
            response = lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload),
            )
            response_payload = response['Payload'].read()
            response_payload_decoded = response_payload.decode('utf-8')

            if response.get('StatusCode') == 200:
                try:
                    return json.loads(response_payload_decoded)
                except json.JSONDecodeError:
                    return response_payload_decoded
            else:
                logger.error(f"Lambda call failed with status code {response.get('StatusCode')}: {response_payload_decoded}")
                return None
        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                logger.warning(f"Lambda call exception: {str(e)}. Retrying {retry_count}/{max_retries} after 10 seconds...")
                time.sleep(10)
            else:
                logger.error(f"Max retries reached ({max_retries}). Aborting.")
                return None
def ingest_stock_data(stocks: List[str], start_date: str, end_date: str, feature_set: str = "basic"):
    payload = {
        "stocks": stocks,
        "start_date": start_date,
        "end_date": end_date,
        "feature_set": feature_set
    }
    return call_lambda("IngestStockData", payload)

def train_binary_classification_model(model_key: str, stock_symbol: str, input_date: str,
                                       hyperparameter_tuning: str = "MEDIUM", feature_set: str = "basic",
                                       lookback_period: int = 720, prediction_horizon: int = 30):
    payload = {
        "model_key": model_key,
        "stock_symbol": stock_symbol,
        "input_date": input_date,
        "hyperparameter_tuning": hyperparameter_tuning,
        "feature_set": feature_set,
        "lookback_period": lookback_period,
        "prediction_horizon": prediction_horizon
    }
    return call_lambda("TrainBinaryClassificationModel", payload)

def make_sarimax_prediction(model_key: str, stock_symbol: str, input_date: str, hyperparameter_tuning: str = 'MEDIUM',
                             feature_set: str = 'basic', lookback_period: int = 720, prediction_horizon: int = 30):
    payload = {
        "model_key": model_key,
        "stock_symbol": stock_symbol,
        "input_date": input_date,
        "hyperparameter_tuning": hyperparameter_tuning,
        "feature_set": feature_set,
        "lookback_period": lookback_period,
        "prediction_horizon": prediction_horizon
    }
    return call_lambda("MakeSarimaxPrediction", payload)

def make_binary_prediction(model_key: str, stock_symbol: str, input_date: str, hyperparameter_tuning: str = 'LOW', feature_set: str = 'basic', lookback_period: int = 720, prediction_horizon: int = 30):
    payload = {
        "model_key": model_key,
        "stock_symbol": stock_symbol,
        "input_date": input_date,
        "hyperparameter_tuning": hyperparameter_tuning,
        "feature_set": feature_set,
        "lookback_period": lookback_period,
        "prediction_horizon": prediction_horizon
    }
    return call_lambda("MakeBinaryClassificationPrediction", payload)

def train_sarimax_model(model_key: str, stock_symbol: str, input_date: str, hyperparameter_tuning: str = 'MEDIUM',
                        feature_set: str = 'advanced', lookback_period: int = 720, prediction_horizon: int = 30):
    payload = {
        "model_key": model_key,
        "stock_symbol": stock_symbol,
        "input_date": input_date,
        "hyperparameter_tuning": hyperparameter_tuning,
        "feature_set": feature_set,
        "lookback_period": lookback_period,
        "prediction_horizon": prediction_horizon
    }
    return call_lambda("TrainSarimaxModel", payload)

def trigger_ai_analysis(predictions: List[Dict[str, Any]]):
    payload = {"predictions": predictions}
    return call_lambda("AiAnalysis", payload)