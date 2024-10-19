import requests
from datetime import datetime, timedelta
from config import Config
from .api_endpoints import (
    ingest_stock_data,
    train_sarimax_model,
    train_binary_classification_model,
    make_sarimax_prediction,
    make_binary_prediction,
    trigger_ai_analysis
)
from flask import session, current_app as app

def generate_model_key(model_type, stock_symbol, feature_set, hyperparameter_tuning, lookback_period, prediction_horizon):
    today = datetime.today()
    formatted_date = today.strftime('%Y%m%d')
    model_key = f"{model_type}_{stock_symbol}_{feature_set}_{hyperparameter_tuning}_{lookback_period}_{prediction_horizon}_{formatted_date}"
    return model_key

def check_data_exists(stock_symbol, start_date, end_date, feature_set):
    if feature_set == 'basic':
        Model = app.BasicStockData
    elif feature_set == 'advanced':
        Model = app.EnrichedStockData
    else:
        raise Exception("Invalid feature set")

    record_count = Model.query.filter(
        Model.symbol == stock_symbol,
        Model.date >= start_date,
        Model.date <= end_date
    ).count()
    total_days = (end_date - start_date).days + 1
    return record_count >= total_days

def generate_stock_prediction(
    model_type: str,
    stock_symbol: str,
    input_date: str,
    hyperparameter_tuning: str = 'MEDIUM',
    feature_set: str = 'basic',
    lookback_period: int = 720,
    prediction_horizon: int = 30,
    make_prediction_step: bool = True
):
    try:
        model_key = generate_model_key(model_type, stock_symbol, feature_set, hyperparameter_tuning, lookback_period, prediction_horizon)

        # Convert input_date to datetime object
        input_date_dt = datetime.strptime(input_date, '%Y-%m-%d')
        start_date_dt = input_date_dt - timedelta(days=lookback_period)
        end_date_dt = input_date_dt

        # Step 1: Check if data exists in the appropriate table
        data_exists = check_data_exists(stock_symbol, start_date_dt.date(), end_date_dt.date(), feature_set)
        if not data_exists:
            # Step 2: If data does not exist, ingest data
            ingest_response = ingest_stock_data(
                stocks=[stock_symbol],
                start_date=start_date_dt.strftime('%Y-%m-%d'),
                end_date=end_date_dt.strftime('%Y-%m-%d'),
                feature_set=feature_set
            )
            if not ingest_response or ingest_response.get('status') != 'success':
                raise Exception(f"Data ingestion failed: {ingest_response.get('message', 'Unknown error')}")

        # Step 3: Train model and save the full output of the train response
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

        if not train_response or train_response.get('status') != 'success':
            raise Exception(f"Model training failed: {train_response.get('message', 'Unknown error')}")

        # Step 4: Optionally call the make prediction API for each date from input_date +/- prediction_horizon
        if make_prediction_step:
            prediction_dates = [
                (input_date_dt + timedelta(days=offset)).strftime('%Y-%m-%d')
                for offset in range(-prediction_horizon, prediction_horizon + 1)
            ]

            for date_str in prediction_dates:
                if model_type == 'SARIMAX':
                    prediction_response = make_sarimax_prediction(
                        model_key=model_key,
                        stock_symbol=stock_symbol,
                        input_date=date_str,
                        hyperparameter_tuning=hyperparameter_tuning,
                        feature_set=feature_set,
                        lookback_period=lookback_period,
                        prediction_horizon=prediction_horizon
                    )
                elif model_type == 'BINARY CLASSIFICATION':
                    prediction_response = make_binary_prediction(
                        model_key=model_key,
                        stock_symbol=stock_symbol,
                        input_date=date_str,
                        feature_set=feature_set
                    )
                else:
                    raise Exception("Invalid model type")

                if not prediction_response or prediction_response.get('status') != 'success':
                    raise Exception(f"Prediction failed for date {date_str}: {prediction_response.get('message', 'Unknown error')}")

                # The predictions are saved to the database by the API; no need to collect them here

        # Step 5: Trigger AI analysis
        ai_analysis_response = trigger_ai_analysis(model_key)
        if not ai_analysis_response or ai_analysis_response.get('status') != 'success':
            raise Exception(f"AI Analysis failed: {ai_analysis_response.get('message', 'Unknown error')}")

        # Save necessary data to session for further processing
        session['model_key'] = model_key

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
        }

        return final_result

    except Exception as e:
        return {'error': str(e)}