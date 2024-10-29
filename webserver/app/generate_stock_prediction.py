import requests
from datetime import datetime, timedelta
from config import Config
from app.lambda_endpoints import (
    ingest_stock_data,
    train_sarimax_model,
    train_binary_classification_model,
    make_sarimax_prediction,
    make_binary_prediction,
    trigger_ai_analysis
)
from flask import session, current_app as app
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_model_key(model_type, stock_symbol, feature_set, hyperparameter_tuning, lookback_period, prediction_horizon, formatted_date):
    model_key = f"{model_type}_{stock_symbol}_{feature_set}_{hyperparameter_tuning}_{lookback_period}_{prediction_horizon}_{formatted_date}"
    return model_key

def check_model_exists(model_key, model_type):
    if model_type == 'SARIMAX':
        Model = app.TrainedModels
    elif model_type == 'BINARY CLASSIFICATION':
        Model = app.TrainedModelsBinary
    else:
        raise Exception("Invalid model type")

    record_count = Model.query.filter(
        Model.model_key == model_key,
    ).count()

    return record_count > 0

def generate_stock_prediction(
    model_type: str,
    stock_symbol: str,
    input_date: str,
    hyperparameter_tuning: str = 'MEDIUM',
    feature_set: str = 'basic',
    lookback_period: int = 720,
    prediction_horizon: int = 30,
    model_key: str = None
):
    try:
        #Error if model key is not provided
        if model_key is None:
            raise Exception("No model key provided. Please provide a model key.")

        # Convert input_date to datetime object
        input_date_dt = datetime.strptime(input_date, '%Y-%m-%d')
        start_date_dt = input_date_dt - timedelta(days=lookback_period)
        end_date_dt = input_date_dt

        #create prediction_date which is input_date + prediction_horizon days. SHould be a string in the format 'YYYY-MM-DD'
        prediction_date = (input_date_dt + timedelta(days=prediction_horizon)).strftime('%Y-%m-%d')

        # Step 1: Check if model exists in the appropriate table
        model_exists = check_model_exists(model_key, model_type)
        if not model_exists:
            # Step 2: If data does not exist, ingest data
            logger.info("Model does not exist. Ingesting data...")
            ingest_response = ingest_stock_data(
                stocks=[stock_symbol],
                start_date=start_date_dt.strftime('%Y-%m-%d'),
                end_date=end_date_dt.strftime('%Y-%m-%d'),
                feature_set=feature_set
            )
            logger.info(f"Ingest response received.")

            # Check if an error occurred
            if ingest_response is None:
                raise Exception("Data ingestion failed: No response from API.")
            elif isinstance(ingest_response, dict) and 'error' in ingest_response:
                raise Exception(f"Data ingestion failed: {ingest_response['error']}")
            else:
                # Success, proceed
                logger.info("Data ingestion successful.")

            # Step 3: Train model
            logger.info(f"Training {model_type} model...")
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

            logger.info(f"Train response received.")

            # Check if an error occurred during training
            if train_response is None:
                raise Exception("Model training failed: No response from API.")
            elif isinstance(train_response, dict) and 'error' in train_response:
                raise Exception(f"Model training failed: {train_response['error']}")
            else:
                # Success, proceed
                logger.info("Model training successful.")

            # Step 4: Make prediction
            logger.info("Making predictions...")
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
                    hyperparameter_tuning=hyperparameter_tuning,
                    feature_set=feature_set,
                    lookback_period=lookback_period,
                    prediction_horizon=prediction_horizon
                )
            else:
                raise Exception("Invalid model type")

            logger.info("Prediction response received.")

            # Check if an error occurred during prediction
            if prediction_response is None:
                raise Exception("Prediction failed: No response from API.")
            elif isinstance(prediction_response, dict) and 'error' in prediction_response:
                raise Exception(f"Prediction failed: {prediction_response['error']}")
            else:
                # Success, proceed
                logger.info("Prediction successful.")

        # Step 5: Prepare data for AI analysis
        logger.info("Preparing data for AI analysis...")
        prediction_data = []

        # Fetch trained model data from the database
        with app.app_context():
            if model_type == 'SARIMAX':
                TrainedModel = app.TrainedModels
            elif model_type == 'BINARY CLASSIFICATION':
                TrainedModel = app.TrainedModelsBinary

            trained_model = TrainedModel.query.filter_by(model_key=model_key).first()
            if not trained_model:
                raise Exception("Trained model not found in the database.")

            trained_model_data = trained_model.to_dict()

        # Build prediction data dictionary
        if model_type == 'SARIMAX':
            prediction_data.append({
                'model_key': model_key,
                'symbol': stock_symbol,
                'prediction_date': prediction_date,
                'prediction_explanation': trained_model_data.get('prediction_explanation', ''),
                'prediction_rmse': trained_model_data.get('prediction_rmse'),
                'prediction_mae': trained_model_data.get('prediction_mae'),
                'prediction_mape': trained_model_data.get('prediction_mape'),
                'prediction_confidence_score': trained_model_data.get('prediction_confidence_score'),
                'confusion_matrix': json.dumps([]),  # Not applicable for regression
                'feature_importance': json.dumps(trained_model_data.get('feature_importance', {})),
                'model_parameters': json.dumps(trained_model_data.get('model_parameters', {})),
                'predicted_movement': None,  # Not applicable for regression
                'predicted_price': None,  # Not directly applicable
                'prediction_probability': None,  # Not applicable for regression
                'last_known_price': trained_model_data.get('last_known_price'),
                'predicted_amount': trained_model_data.get('predicted_amount'),
                'predictions_json': json.dumps(trained_model_data.get('predictions_json', {})),
                'model_location': trained_model_data.get('model_location'),
                'date_created': trained_model_data.get('date_created')
            })
        elif model_type == 'BINARY CLASSIFICATION':
            prediction_data.append({
                'model_key': model_key,
                'symbol': stock_symbol,
                'prediction_date': prediction_date,
                'prediction_explanation': trained_model_data.get('prediction_explanation', ''),
                'prediction_accuracy': trained_model_data.get('prediction_accuracy'),
                'prediction_precision': trained_model_data.get('prediction_precision'),
                'prediction_recall': trained_model_data.get('prediction_recall'),
                'prediction_f1_score': trained_model_data.get('prediction_f1_score'),
                'prediction_roc_auc': trained_model_data.get('prediction_roc_auc'),
                'confusion_matrix': json.dumps(trained_model_data.get('confusion_matrix', [])),
                'feature_importance': json.dumps(trained_model_data.get('feature_importance', {})),
                'model_parameters': json.dumps(trained_model_data.get('model_parameters', {})),
                'predicted_movement': trained_model_data.get('predicted_movement'),
                'predicted_price': trained_model_data.get('predicted_price'),
                'prediction_probability': trained_model_data.get('prediction_probability'),
                'last_known_price': trained_model_data.get('last_known_price'),
                'predicted_amount': None,  # Not applicable for classification
                'predictions_json': json.dumps(trained_model_data.get('predictions_json', {})),
                'model_location': trained_model_data.get('model_location'),
                'date_created': trained_model_data.get('date_created')
            })

        logger.info("Prediction data prepared for AI analysis.")

        # Step 6: Trigger AI analysis with formatted data
        logger.info("Triggering AI analysis...")
        ai_analysis_response = trigger_ai_analysis(prediction_data)
        logger.info("AI analysis response received.")

        # Check if an error occurred during AI analysis
        if ai_analysis_response is None:
            raise Exception("AI Analysis failed: No response from API.")
        elif isinstance(ai_analysis_response, dict) and 'error' in ai_analysis_response:
            raise Exception(f"AI Analysis failed: {ai_analysis_response['error']}")
        else:
            # Success, proceed
            logger.info("AI Analysis successful.")

        # Check if an error occurred during AI analysis
        if ai_analysis_response is None:
            raise Exception("AI Analysis failed: No response from API.")
        elif isinstance(ai_analysis_response, dict) and 'error' in ai_analysis_response:
            raise Exception(f"AI Analysis failed: {ai_analysis_response['error']}")
        else:
            # Success, proceed
            logger.info("AI Analysis successful.")

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

        logger.info("Prediction generation successful.")
        return final_result

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return {'error': str(e)}