#%%
import json
from datetime import datetime, date
import pandas as pd
from ai_functions import execute_chatgpt_call, parse_response_to_json
from database_functions import upsert_df, create_engine_from_url
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_DATABASE

# Database connection string
db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"

def generate_chatgpt_explanation(input_data):
    """
    Generates a detailed explanation for a stock prediction using ChatGPT.

    Args:
        input_data (dict): The prediction data containing all relevant fields.

    Returns:
        str: The explanation generated by ChatGPT.

    Raises:
        ValueError: If required fields are missing or ChatGPT fails to generate an explanation.
    """
    # Determine model type based on 'prediction_explanation'
    prediction_explanation = input_data.get('prediction_explanation', '').lower()
    if 'classification' in prediction_explanation or 'classifier' in prediction_explanation:
        model_category = 'classification'
    elif 'regression' in prediction_explanation or 'regressor' in prediction_explanation:
        model_category = 'regression'
    else:
        model_category = 'general'

    # Example explanations for different model types
    example_explanations = {
        'classification': """
**Conclusion:**
The model confidently predicts that the stock for {symbol} will **{predicted_movement}** by {prediction_date}.

**Performance Metrics:**
- **Accuracy:** {prediction_accuracy} ({prediction_accuracy_percentage}%) - Indicates the proportion of correct predictions. A higher value signifies better performance.
- **Precision:** {prediction_precision} ({prediction_precision_percentage}%) - Measures the accuracy of positive predictions. Higher precision reduces false positives.
- **Recall:** {prediction_recall} ({prediction_recall_percentage}%) - Assesses the model's ability to identify all relevant instances. Higher recall reduces false negatives.
- **F1 Score:** {prediction_f1_score} ({prediction_f1_score_percentage}%) - Balances precision and recall. A higher F1 score indicates better model performance.
- **ROC AUC:** {prediction_roc_auc} - Reflects the model's ability to distinguish between classes. Values closer to 1 indicate excellent performance.

**Prediction Details:**
- **Predicted Price:** {predicted_price}
- **Confidence Level:** {prediction_probability_percentage}%
- **Last Known Price:** {last_known_price}

**Key Influencing Features:**
- {important_features}

**Confusion Matrix:**
- **True Positives:** {confusion_matrix_tp}
- **True Negatives:** {confusion_matrix_tn}
- **False Positives:** {confusion_matrix_fp}
- **False Negatives:** {confusion_matrix_fn}
""",
        'regression': """
**Conclusion:**
The SARIMAX model forecasts the stock price for {symbol} to be **{predicted_amount}** by {prediction_date}.

**Performance Metrics:**
- **RMSE:** {prediction_rmse} - Measures the model's prediction error. Lower values indicate better performance.
- **MAE:** {prediction_mae} - Represents the average absolute error. Lower values are preferable.
- **MAPE:** {prediction_mape}% - Shows the mean absolute percentage error. Values closer to 0% are better.
- **Confidence Score:** {prediction_confidence_score} - Reflects the model's confidence in its predictions. Higher scores denote higher confidence.

**Prediction Details:**
- **Predicted Amount:** {predicted_amount}
- **Last Known Price:** {last_known_price}

**Key Influencing Features:**
- {important_features}

**Model Parameters:**
- **Order:** {order}
- **Seasonal Order:** {seasonal_order}
- **Hyperparameter Tuning:** {hyperparameter_tuning}
""",
        'general': """
**Conclusion:**
The model predicts that the stock for {symbol} will **{predicted_movement_or_details}** by {prediction_date}.

**Performance Metrics:**
- {metrics}

**Prediction Details:**
- {prediction_details}
- **Last Known Price:** {last_known_price}

**Key Influencing Features:**
- {important_features}

**Conclusion:**
The model's prediction suggests **{conclusion}**.
"""
    }

    # Select appropriate example based on model category
    example_template = example_explanations.get(model_category, example_explanations['general'])

    # Extract and prepare necessary fields
    symbol = input_data.get('symbol', 'N/A')
    prediction_date = input_data.get('prediction_date', 'N/A')
    predicted_movement = input_data.get('predicted_movement') if input_data.get(
        'predicted_movement') is not None else 'N/A'
    predicted_price = input_data.get('predicted_price') if input_data.get('predicted_price') is not None else 'N/A'
    last_known_price = input_data.get('last_known_price') if input_data.get('last_known_price') is not None else 'N/A'
    feature_importance = input_data.get('feature_importance', '{}')
    try:
        feature_importance_dict = json.loads(feature_importance)
        important_features = ', '.join([f"{k} ({v})" for k, v in feature_importance_dict.items()])
    except json.JSONDecodeError:
        important_features = 'N/A'

    # Prepare confusion matrix components for classification models
    confusion_matrix = input_data.get('confusion_matrix', '[]')
    try:
        confusion_matrix_list = json.loads(confusion_matrix)
        confusion_matrix_tp = confusion_matrix_list[0][0] if len(confusion_matrix_list) > 0 else 'N/A'
        confusion_matrix_fn = confusion_matrix_list[1][0] if len(confusion_matrix_list) > 1 else 'N/A'
        confusion_matrix_fp = confusion_matrix_list[0][1] if len(confusion_matrix_list[0]) > 1 else 'N/A'
        confusion_matrix_tn = confusion_matrix_list[1][1] if len(confusion_matrix_list) > 1 and len(
            confusion_matrix_list[1]) > 1 else 'N/A'
    except (json.JSONDecodeError, IndexError):
        confusion_matrix_tp = confusion_matrix_fn = confusion_matrix_fp = confusion_matrix_tn = 'N/A'

    # Metrics handling based on model category
    if model_category == 'classification':
        try:
            prediction_accuracy = float(input_data.get('prediction_accuracy', 0))
            prediction_precision = float(input_data.get('prediction_precision', 0))
            prediction_recall = float(input_data.get('prediction_recall', 0))
            prediction_f1_score = float(input_data.get('prediction_f1_score', 0))
            prediction_roc_auc = float(input_data.get('prediction_roc_auc', 0))

            # Precompute percentages
            prediction_accuracy_percentage = round(prediction_accuracy * 100, 2)
            prediction_precision_percentage = round(prediction_precision * 100, 2)
            prediction_recall_percentage = round(prediction_recall * 100, 2)
            prediction_f1_score_percentage = round(prediction_f1_score * 100, 2)
            prediction_probability = float(input_data.get('prediction_probability', 0))
            prediction_probability_percentage = round(prediction_probability * 100, 2)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid metric values for classification: {e}")

        metrics = f"**Accuracy:** {prediction_accuracy} ({prediction_accuracy_percentage}%) - Indicates the proportion of correct predictions. A higher value signifies better performance.\n" \
                  f"**Precision:** {prediction_precision} ({prediction_precision_percentage}%) - Measures the accuracy of positive predictions. Higher precision reduces false positives.\n" \
                  f"**Recall:** {prediction_recall} ({prediction_recall_percentage}%) - Assesses the model's ability to identify all relevant instances. Higher recall reduces false negatives.\n" \
                  f"**F1 Score:** {prediction_f1_score} ({prediction_f1_score_percentage}%) - Balances precision and recall. A higher F1 score indicates better model performance.\n" \
                  f"**ROC AUC:** {prediction_roc_auc} - Reflects the model's ability to distinguish between classes. Values closer to 1 indicate excellent performance."

    elif model_category == 'regression':
        try:
            prediction_rmse = float(input_data.get('prediction_rmse', 0))
            prediction_mae = float(input_data.get('prediction_mae', 0))
            prediction_mape = float(input_data.get('prediction_mape', 0))
            prediction_confidence_score = float(input_data.get('prediction_confidence_score', 0))
            predicted_amount = float(input_data.get('predicted_amount', 0))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid metric values for regression: {e}")

        metrics = f"**RMSE:** {prediction_rmse} - Measures the model's prediction error. Lower values indicate better performance.\n" \
                  f"**MAE:** {prediction_mae} - Represents the average absolute error. Lower values are preferable.\n" \
                  f"**MAPE:** {prediction_mape}% - Shows the mean absolute percentage error. Values closer to 0% are better.\n" \
                  f"**Confidence Score:** {prediction_confidence_score} - Reflects the model's confidence in its predictions. Higher scores denote higher confidence."

    else:
        metrics = input_data.get('metrics', 'N/A')

    # Model parameters for regression models
    if model_category == 'regression':
        model_parameters = input_data.get('model_parameters', '{}')
        try:
            model_parameters_dict = json.loads(model_parameters)
            order = model_parameters_dict.get('order', 'N/A')
            seasonal_order = model_parameters_dict.get('seasonal_order', 'N/A')
            hyperparameter_tuning = model_parameters_dict.get('hyperparameter_tuning', 'N/A')
        except json.JSONDecodeError:
            order = seasonal_order = hyperparameter_tuning = 'N/A'

    # Format the example explanation with actual data
    if model_category == 'classification':
        formatted_example = example_template.format(
            symbol=symbol,
            prediction_date=prediction_date,
            predicted_movement=predicted_movement,
            predicted_price=predicted_price,
            prediction_accuracy=prediction_accuracy,
            prediction_accuracy_percentage=prediction_accuracy_percentage,
            prediction_precision=prediction_precision,
            prediction_precision_percentage=prediction_precision_percentage,
            prediction_recall=prediction_recall,
            prediction_recall_percentage=prediction_recall_percentage,
            prediction_f1_score=prediction_f1_score,
            prediction_f1_score_percentage=prediction_f1_score_percentage,
            prediction_roc_auc=prediction_roc_auc,
            prediction_probability_percentage=prediction_probability_percentage,
            last_known_price=last_known_price,
            important_features=important_features,
            confusion_matrix_tp=confusion_matrix_tp,
            confusion_matrix_tn=confusion_matrix_tn,
            confusion_matrix_fp=confusion_matrix_fp,
            confusion_matrix_fn=confusion_matrix_fn
        )
    elif model_category == 'regression':
        formatted_example = example_template.format(
            symbol=symbol,
            prediction_date=prediction_date,
            predicted_amount=predicted_amount,
            prediction_rmse=prediction_rmse,
            prediction_mae=prediction_mae,
            prediction_mape=prediction_mape,
            prediction_confidence_score=prediction_confidence_score,
            last_known_price=last_known_price,
            important_features=important_features,
            order=order,
            seasonal_order=seasonal_order,
            hyperparameter_tuning=hyperparameter_tuning
        )
    else:
        predicted_movement_or_details = predicted_movement if predicted_movement != 'N/A' else input_data.get(
            'prediction_details', 'N/A')
        conclusion = input_data.get('conclusion', 'N/A')
        prediction_details = input_data.get('prediction_details', 'N/A')
        formatted_example = example_template.format(
            symbol=symbol,
            prediction_date=prediction_date,
            predicted_movement_or_details=predicted_movement_or_details,
            metrics=metrics,
            prediction_details=prediction_details,
            last_known_price=last_known_price,
            important_features=important_features,
            conclusion=conclusion
        )

    # Construct the prompt with input data and example explanation
    if model_category == 'classification':
        # Already computed
        pass
    user_prompt = f"""
Stock Prediction Details:
- **Model Key:** {input_data.get('model_key', 'N/A')}
- **Symbol:** {symbol}
- **Prediction Date:** {prediction_date}
- **Prediction Explanation:** {prediction_explanation}
- **Accuracy:** {input_data.get('prediction_accuracy', 'N/A')}
- **Precision:** {input_data.get('prediction_precision', 'N/A')}
- **Recall:** {input_data.get('prediction_recall', 'N/A')}
- **F1 Score:** {input_data.get('prediction_f1_score', 'N/A')}
- **ROC AUC:** {input_data.get('prediction_roc_auc', 'N/A')}
- **RMSE:** {input_data.get('prediction_rmse', 'N/A')}
- **MAE:** {input_data.get('prediction_mae', 'N/A')}
- **MAPE:** {input_data.get('prediction_mape', 'N/A')}
- **Confidence Score:** {input_data.get('prediction_confidence_score', 'N/A')}
- **Confusion Matrix:** {input_data.get('confusion_matrix', 'N/A')}
- **Feature Importance:** {important_features}
- **Model Parameters:** {input_data.get('model_parameters', 'N/A')}
- **Predicted Movement:** {predicted_movement}
- **Predicted Price:** {predicted_price}
- **Prediction Probability:** {prediction_probability_percentage if model_category == 'classification' else 'N/A'}
- **Predicted Amount:** {input_data.get('predicted_amount', 'N/A')}
- **Last Known Price:** {last_known_price}
- **Model Location:** {input_data.get('model_location', 'N/A')}

**Example Explanation:**
{formatted_example}

Please provide a detailed explanation based on the stock prediction data above.
"""

    # Call ChatGPT to generate the explanation
    system_prompt = "You are a financial AI model that provides clear and concise stock analysis explanations based on model outputs and metrics."
    gpt_response = execute_chatgpt_call(user_prompt, system_prompt)

    return gpt_response


def process_stock_prediction_from_json(prediction, engine):
    """
    Processes a single stock prediction entry, generates an explanation, and logs it to the database.

    Args:
        prediction (dict): The prediction data.
        engine: SQLAlchemy engine for database connection.

    Returns:
        dict: A dictionary containing 'model_key', 'explanation', and 'created_date'.

    Raises:
        ValueError: If required fields are missing.
    """
    required_fields = ['model_key', 'symbol', 'prediction_date', 'prediction_explanation', 'date_created']
    if not all(field in prediction for field in required_fields):
        raise ValueError("Missing one or more required fields in the prediction data")

    # Generate explanation using ChatGPT
    explanation = generate_chatgpt_explanation(prediction)

    # Construct the response
    response = {
        'model_key': prediction['model_key'],
        'explanation': explanation,
        'created_date': prediction['date_created']
    }

    # Log the explanation to the ai_analysis table
    analysis_data = {
        'model_key': prediction['model_key'],
        'explanation': explanation,
        'created_date': prediction['date_created']
    }
    analysis_df = pd.DataFrame([analysis_data])

    # Perform upsert into ai_analysis table
    upsert_success = upsert_df(
        df=analysis_df,
        table_name='ai_analysis',
        upsert_id='model_key',
        postgres_connection=engine,
        json_columns=[],
        auto_match_schema='public'
    )

    if not upsert_success:
        raise RuntimeError(f"Failed to upsert explanation for model_key: {prediction['model_key']}")

    return response


def lambda_handler(event, context):
    """
    AWS Lambda handler function to process stock predictions, generate explanations, and log them to the database.

    Args:
        event (dict): The input event containing predictions.
        context: The runtime information.

    Returns:
        dict: The response with status code and body.
    """
    # Create a database engine
    engine = create_engine_from_url(db_url)

    try:
        # Handle both API Gateway and local invocations
        if 'body' in event:  # Case when invoked through API Gateway
            # Parse the body if it's a string (API Gateway sends body as string)
            if isinstance(event['body'], str):
                event_body = json.loads(event['body'])
            else:
                event_body = event['body']
        else:  # Case when running locally
            event_body = event

        predictions = event_body.get('predictions', [])
        if not predictions:
            raise ValueError("No predictions provided in the input event")

        results = []
        for prediction in predictions:
            try:
                result = process_stock_prediction_from_json(prediction, engine)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e), 'input': prediction})

        return {
            'statusCode': 200,
            'body': json.dumps(results)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({"error": f"Error processing predictions: {str(e)}"})
        }


#%%
import pandas as pd  # Ensure pandas is imported for local testing

test_event = {
    'predictions': [
        # Classification Model Example
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
}
#%%
#print(lambda_handler(test_event, None))