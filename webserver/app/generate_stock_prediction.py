import requests
from datetime import datetime, timedelta
from config import Config  # This should still work
from .api_endpoints import (
    ingest_stock_data,
    train_sarimax_model,
    train_binary_classification_model,
    make_sarimax_prediction,
    trigger_ai_analysis
)


def generate_model_key(model_type, stock_symbol, feature_set, hyperparameter_tuning, lookback_period, prediction_horizon):
    today = datetime.today()
    formatted_date = today.strftime('%Y%m%d')
    model_key = f"{model_type}_{stock_symbol}_{feature_set}_{hyperparameter_tuning}_{lookback_period}_{prediction_horizon}_{formatted_date}"
    return model_key

def generate_stock_prediction(
    model_type: str,
    stock_symbol: str,
    input_date: str,
    hyperparameter_tuning: str = 'MEDIUM',
    feature_set: str = 'advanced',
    lookback_period: int = 720,
    prediction_horizon: int = 30,
):
    try:
        model_key = generate_model_key(model_type, stock_symbol, feature_set, hyperparameter_tuning, lookback_period, prediction_horizon)

        # Step 1: Ingest data
        start_date = (datetime.strptime(input_date, '%Y-%m-%d') - timedelta(days=lookback_period)).strftime('%Y-%m-%d')
        end_date = input_date
        ingest_response = ingest_stock_data(
            stocks=[stock_symbol],
            start_date=start_date,
            end_date=end_date,
            feature_set=feature_set
        )
        if ingest_response.get('status') != 'success':
            raise Exception(f"Data ingestion failed: {ingest_response.get('message')}")

        # Step 2: Train model
        if model_type == 'SARIMAX':
            train_response = train_sarimax_model(
                model_key=model_key,
                stock_symbol=stock_symbol,
                input_date=input_date,
                hyperparameter_tuning=hyperparameter_tuning,
                feature_set=feature_set,
                lookback_period=lookback_period,
                prediction_horizon=prediction_horizon
            )
        elif model_type == 'BINARY CLASSIFICATION':
            train_response = train_binary_classification_model(
                model_key=model_key,
                stock_symbol=stock_symbol,
                input_date=input_date,
                hyperparameter_tuning=hyperparameter_tuning,
                feature_set=feature_set,
                lookback_period=lookback_period,
                prediction_horizon=prediction_horizon
            )
        else:
            raise Exception("Invalid model type")

        if train_response.get('status') != 'success':
            raise Exception(f"Model training failed: {train_response.get('message')}")

        # Step 3: Make prediction
        if model_type == 'SARIMAX':
            prediction_response = make_sarimax_prediction(
                model_key=model_key,
                stock_symbol=stock_symbol,
                input_date=input_date,
                hyperparameter_tuning=hyperparameter_tuning,
                feature_set=feature_set,
                lookback_period=lookback_period,
                prediction_horizon=prediction_horizon
            )
        elif model_type == 'BINARY CLASSIFICATION':
            prediction_response = make_binary_prediction(
                model_key=model_key,
                stock_symbol=stock_symbol,
                input_date=input_date,
                feature_set=feature_set
            )
        else:
            raise Exception("Invalid model type")

        if prediction_response.get('status') != 'success':
            raise Exception(f"Prediction failed: {prediction_response.get('message')}")

        # Step 4: AI Analysis
        ai_analysis_response = trigger_ai_analysis(prediction_response.get('data', []))
        if ai_analysis_response.get('status') != 'success':
            raise Exception(f"AI Analysis failed: {ai_analysis_response.get('message')}")

        # Compile final result
        final_result = {
            'model_type': model_type,
            'model_key': model_key,
            'stock_symbol': stock_symbol,
            'input_date': input_date,
            'hyperparameter_tuning': hyperparameter_tuning,
            'feature_set': feature_set,
            'lookback_period': lookback_period,
            'prediction_horizon': prediction_horizon,
            'prediction_result': prediction_response.get('data'),
            'ai_analysis': ai_analysis_response.get('data')
        }

        return final_result

    except Exception as e:
        return {'error': str(e)}