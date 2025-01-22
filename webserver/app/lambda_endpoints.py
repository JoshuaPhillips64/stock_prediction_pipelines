#%%
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
lambda_client = boto3.client('lambda', region_name='us-east-1')

#%%
def call_lambda(function_name: str, payload: Optional[Dict[str, Any]] = None):
    retry_count = 0
    max_retries = 0

    if payload is None:
        payload = {}

    # Wrap the payload in an event dictionary with a body key
    event = {
        'body': json.dumps(payload)
    }

    while retry_count <= max_retries:
        try:
            response = lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(event),
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
    return call_lambda("ingest_stock_data", payload)

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
    return call_lambda("train_binary_classification_model", payload)

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
    return call_lambda("make_sarimax_prediction", payload)

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
    return call_lambda("make_binary_classification_prediction", payload)

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
    return call_lambda("train_sarimax_model", payload)

def trigger_ai_analysis(predictions: List[Dict[str, Any]]):
    payload = {"predictions": predictions}
    return call_lambda("ai_analysis", payload)

#%%

#ingest_response = ingest_stock_data(["PG"], "2024-01-01", "2024-10-10")
#print("Lambda Test Run - Ingest Stock Data Response:", ingest_response)
""""
#%%
train_binary_response = train_binary_classification_model(
    model_key="test_model3",
    stock_symbol="PG",
    input_date="2024-10-01",
    hyperparameter_tuning="LOW",
    feature_set="basic",
    lookback_period=1000,
    prediction_horizon=30
)
print("Train Binary Classification Model Response:", train_binary_response)

make_binary_response = make_binary_prediction(
    model_key="test_model3",
    stock_symbol="PG",
    input_date="2024-10-01",
    hyperparameter_tuning="LOW",
    feature_set="basic",
    lookback_period=1000,
    prediction_horizon=30
)
print("Make Binary Prediction Response:", make_binary_response)

#%%
train_sarimax_response = train_sarimax_model(
    model_key='sample_model_001',
    stock_symbol='PG',
    input_date='2024-10-01',
    hyperparameter_tuning='LOW',
    feature_set='advanced',
    lookback_period=720,
    prediction_horizon=30
)
print("Train Sarimax Model Response:", train_sarimax_response)

make_sarimax_response = make_sarimax_prediction(
    model_key="sample_model_001",
    stock_symbol="PG",
    input_date='2024-10-01',
    hyperparameter_tuning='LOW',
    feature_set='advanced',
    lookback_period=720,
    prediction_horizon=30
)
print("Make Sarimax Prediction Response:", make_sarimax_response)

#%%
# Example prediction data (replace with your actual data)
prediction_data = [
    {
        'model_key': 'model_123',
        'symbol': 'KO',
        'prediction_date': '2024-11-01',
        'prediction_explanation': 'Binary Classification Based on XGBoostClassifier with feature engineering and SMOTE',
        'prediction_accuracy': 0.89,
        'prediction_precision': 0.88,
        'prediction_recall': 0.87,
        'prediction_f1_score': 0.87,
        'prediction_roc_auc': 0.9,
        'confusion_matrix': json.dumps([[50, 10], [5, 35]]),
        'feature_importance': json.dumps({'Volume': 0.3, 'Moving Average': 0.2}),
        'model_parameters': json.dumps({'n_estimators': 100, 'max_depth': 5}),
        'predicted_movement': 'Up',
        'predicted_price': 55.3,
        'prediction_probability': 0.78,
        'last_known_price': 52.1,
        'predicted_amount': None,  # Not applicable for classification
        'predictions_json': '{"predictions":[...]}',
        'model_location': 's3://trained-models-stock-prediction/xgboost_model_123.pkl',
        'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    },
    # Regression Model Example
    {
        'model_key': 'model_456',
        'symbol': 'AAPL',
        'prediction_date': '2024-12-01',
        'prediction_explanation': 'Regression Prediction Based on SARIMAX model with feature engineering',
        'prediction_rmse': 1.2,
        'prediction_mae': 0.9,
        'prediction_mape': 5.0,
        'prediction_confidence_score': 0.45,
        'confusion_matrix': json.dumps([]),  # Not applicable for regression
        'feature_importance': json.dumps({'Price Lag': 0.4, 'Volume': 0.35}),
        'model_parameters': json.dumps({
            'order': [1, 1, 1],
            'seasonal_order': [1, 1, 1, 12],
            'hyperparameter_tuning': 'Grid Search',
            'feature_set': 'Price Lag, Volume',
            'prediction_horizon': '30 days'
        }),
        'predicted_movement': None,  # Not applicable for regression
        'predicted_price': None,  # Not directly applicable
        'prediction_probability': None,  # Not applicable for regression
        'last_known_price': 150.0,
        'predicted_amount': 155.5,
        'predictions_json': '{"predictions":[...]}',
        'model_location': 's3://trained-models-stock-prediction/sarimax_model_456.pkl',
        'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
]
ai_analysis_response = trigger_ai_analysis(prediction_data)
print("AI Analysis Response:", ai_analysis_response)
"""