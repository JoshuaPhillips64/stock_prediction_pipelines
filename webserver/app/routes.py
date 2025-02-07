from flask import Blueprint, render_template, request, redirect, url_for, session, current_app as app, flash, jsonify
from .forms import PredictionForm, ContactForm
from .generate_stock_prediction import generate_stock_prediction, generate_model_key
from datetime import datetime, timedelta, date
import json
import logging
import re
import requests
from app import db
from sqlalchemy import desc
from .chatgpt_utils import handle_chatgpt_response, increment_session_request_count
from flask_cors import cross_origin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

main_bp = Blueprint('main_bp', __name__)

def verify_recaptcha(recaptcha_token, action):
    recaptcha_secret = app.config.get('RECAPTCHA_SECRET_KEY')
    recaptcha_response = requests.post(
        'https://www.google.com/recaptcha/api/siteverify',
        data={
            'secret': recaptcha_secret,
            'response': recaptcha_token
        }
    )
    recaptcha_result = recaptcha_response.json()
    logger.info(f"reCAPTCHA verification result: {recaptcha_result}")

    # Check the success key
    if not recaptcha_result.get('success'):
        logger.warning(f"reCAPTCHA verification failed: {recaptcha_result.get('error-codes')}")
        return False

    # Verify the action matches
    if recaptcha_result.get('action') != action:
        logger.warning(f"reCAPTCHA action mismatch: expected '{action}', got '{recaptcha_result.get('action')}'")
        return False

    # Check the score threshold
    if recaptcha_result.get('score', 0.0) < 0.3:
        logger.warning(f"reCAPTCHA score too low: {recaptcha_result.get('score')}")
        return False

    return True

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    form = PredictionForm()

    # Calculate the date three days ago from today
    today = datetime.today()
    three_days_ago = today - timedelta(days=3)

    # Query top 5 Binary Classification models from the last 3 days, sorted by F1 Factor descending
    top_binary_models = app.TrainedModelsBinary.query.filter(
        app.TrainedModelsBinary.date_created >= three_days_ago
    ).order_by(desc(app.TrainedModelsBinary.prediction_f1_score)).limit(5).all()

    # Query top 5 Regression models from the last 3 days, sorted by MAPE ascending (lower is better)
    top_regression_models = app.TrainedModels.query.filter(
        app.TrainedModels.date_created >= three_days_ago
    ).order_by(app.TrainedModels.prediction_mape).limit(5).all()

    if form.validate_on_submit():
        recaptcha_token = request.form.get('recaptcha_token')
        if not recaptcha_token:
            flash('reCAPTCHA token missing. Please try again.', 'danger')
            return redirect(url_for('main_bp.index'))

        if not verify_recaptcha(recaptcha_token, 'prediction'):
            flash('reCAPTCHA verification failed. Please try again.', 'danger')
            return redirect(url_for('main_bp.index'))

        # Save form data to session
        session['form_data'] = request.form
        return redirect(url_for('main_bp.loading'))

    return render_template(
        'index.html',
        form=form,
        top_binary_models=top_binary_models,
        top_regression_models=top_regression_models
    )


@main_bp.route('/api/chat', methods=['POST'])
@cross_origin()
def chat():
    try:
        increment_session_request_count(max_requests=20)

        data = request.json
        messages = data.get("messages", [])

        # Check if it's the first user message in the session
        if session.get('request_count', 0) == 1:
            # Prepend initial messages to the user's first message
            initial_messages = [
                {
                    "role": "system",
                    "content": """You are an expert AI assistant for a website called smartstockpredictor.com. Your task is to guide users step-by-step through the parameters 
                               of the attached functions and execute them after confirmation that parameters are correct. You should approach each step as a separate message with clear 
                               explanations for the user at a basic level. The functions are provided in the schemas doucumenations"""
                }
            ]
            # Append the user's first message to the initial messages
            messages = initial_messages + messages

        # Validate messages
        if not isinstance(messages, list) or not messages:
            return jsonify({"error": "Invalid messages format"}), 400

        ai_response = handle_chatgpt_response(messages)
        return jsonify({"response": ai_response})

    except Exception as e:
        # Handle exceptions and return appropriate HTTP status codes
        error_message = str(e)
        if "Session limit reached" in error_message:
            return jsonify({"error": error_message}), 403
        elif "Rate limit reached" in error_message:
            return jsonify({"error": error_message}), 429
        else:
            return jsonify({"error": error_message}), 500


@main_bp.route('/loading')
def loading():
    return render_template('loading.html')

@main_bp.route('/results')
def results():
    # ---------------------------
    # BRANCH 1: model_key & model_type provided in URL
    # ---------------------------
    model_key = request.args.get('model_key')
    model_type = request.args.get('model_type')

    if model_key and model_type:
        try:
            # Pick correct model table
            if model_type == 'BINARY CLASSIFICATION':
                TrainedModel = app.TrainedModelsBinary
            elif model_type == 'SARIMAX':
                TrainedModel = app.TrainedModels
            else:
                flash('Invalid model type provided.', 'danger')
                return redirect(url_for('main_bp.index'))

            # Query the model from DB
            trained_model = TrainedModel.query.filter_by(model_key=model_key).first()
            if not trained_model:
                flash('Model not found.', 'danger')
                return redirect(url_for('main_bp.index'))

            trained_model_data = trained_model.to_dict()
            # Here we grab the nested JSONB dict:
            model_parameters = trained_model_data.get('model_parameters', {})

            # Prepare combined predictions
            combined_predictions = _get_combined_predictions(model_type, model_key, trained_model)
            # Prepare chart data
            chart_data = _prepare_chart_data(model_type, combined_predictions)
            # Fetch AI analysis
            ai_analysis = _fetch_ai_analysis(model_key)
            # Extract performance metrics
            performance_metrics = _extract_performance_metrics(model_type, trained_model_data)
            # Fetch feature importance
            feature_importance = _fetch_feature_importance(trained_model_data)

            # Render template using JSONB fields
            return render_template(
                'results.html',
                stock_symbol=trained_model.symbol,
                model_type=model_type,
                hyperparameter_tuning=model_parameters.get('hyperparameter_tuning'),
                feature_set=model_parameters.get('feature_set'),
                lookback_period=model_parameters.get('lookback_period'),
                prediction_horizon=model_parameters.get('prediction_horizon'),
                input_date=model_parameters.get('input_date'),
                chart_data=chart_data,
                ai_analysis=ai_analysis,
                performance_metrics=performance_metrics,
                feature_importance=feature_importance
            )

        except Exception as e:
            logger.error(f"Error fetching model results: {str(e)}")
            flash('An error occurred while fetching the results.', 'danger')
            return redirect(url_for('main_bp.index'))

    # ---------------------------
    # BRANCH 2: No model_key/model_type in the URL, so use session form_data
    # ---------------------------
    form_data = session.get('form_data', {})
    if not form_data:
        logger.error("No form data found in session.")
        return redirect(url_for('main_bp.index'))

    logger.info(f"Form data retrieved from session: {form_data}")

    # Extract form data
    model_type = form_data.get('model_type')
    stock_symbol = form_data.get('stock_symbol', '').upper()
    feature_set = form_data.get('feature_set')
    hyperparameter_tuning = form_data.get('hyperparameter_tuning')
    lookback_period = int(form_data.get('lookback_period', 0))
    prediction_horizon = int(form_data.get('prediction_horizon', 0))
    input_date_str = form_data.get('input_date')

    # Convert input_date_str => 'YYYY-MM-DD'
    input_date_str_updated = datetime.strptime(input_date_str, '%Y-%m-%d').strftime('%Y-%m-%d')

    if not all([model_type, stock_symbol, feature_set, hyperparameter_tuning,
                lookback_period, prediction_horizon, input_date_str]):
        logger.error("Incomplete form data provided.")
        return render_template('error.html', error_message='Incomplete form data provided.')

    # Generate model key
    model_key = generate_model_key(
        model_type=model_type,
        stock_symbol=stock_symbol,
        feature_set=feature_set,
        hyperparameter_tuning=hyperparameter_tuning,
        lookback_period=lookback_period,
        prediction_horizon=prediction_horizon,
        formatted_date=input_date_str_updated
    )
    logger.info(f"Generated model_key: {model_key}")

    # Generate stock prediction
    result = generate_stock_prediction(
        model_type=model_type,
        stock_symbol=stock_symbol,
        input_date=input_date_str,
        hyperparameter_tuning=hyperparameter_tuning,
        feature_set=feature_set,
        lookback_period=lookback_period,
        prediction_horizon=prediction_horizon,
        model_key=model_key
    )

    if not result:
        error_message = 'Prediction generation failed.'
        logger.error(error_message)
        return render_template('error.html', error_message=error_message)

    if 'error' in result:
        error_message = result['error']
        logger.error(f"Error in result: {error_message}")
        return render_template('error.html', error_message=error_message)

    # Determine the appropriate TrainedModel class
    TrainedModel = app.TrainedModelsBinary if model_type == 'BINARY CLASSIFICATION' else app.TrainedModels
    trained_model = TrainedModel.query.filter_by(model_key=model_key).first()

    if not trained_model:
        error_message = 'No trained model found in the database.'
        logger.error(error_message)
        return render_template('error.html', error_message=error_message)

    trained_model_data = trained_model.to_dict()
    logger.info(f"Trained model data retrieved: {trained_model_data}")

    # Extract the JSONB sub-dict from the DB record
    model_parameters = trained_model_data.get('model_parameters', {})

    logger.info(f"[FORM Branch] model_parameters = {model_parameters}")
    logger.info(f"[FORM Branch] hyperparameter_tuning = {model_parameters.get('hyperparameter_tuning')}")
    logger.info(f"[FORM Branch] feature_set = {model_parameters.get('feature_set')}")
    logger.info(f"[FORM Branch] lookback_period = {model_parameters.get('lookback_period')}")
    logger.info(f"[FORM Branch] prediction_horizon = {model_parameters.get('prediction_horizon')}")
    logger.info(f"[FORM Branch] input_date = {model_parameters.get('input_date')}")

    # Prepare combined predictions
    combined_predictions = _get_combined_predictions(model_type, model_key, trained_model)
    # Prepare chart data
    chart_data = _prepare_chart_data(model_type, combined_predictions)
    # Fetch AI analysis
    ai_analysis = _fetch_ai_analysis(model_key)
    # Extract performance metrics
    performance_metrics = _extract_performance_metrics(model_type, trained_model_data)
    # Fetch feature importance
    feature_importance = _fetch_feature_importance(trained_model_data)

    # Render the template
    return render_template(
        'results.html',
        stock_symbol=stock_symbol,
        model_type=model_type,
        # Pull from JSONB field:
        hyperparameter_tuning=model_parameters.get('hyperparameter_tuning'),
        feature_set=model_parameters.get('feature_set'),
        lookback_period=model_parameters.get('lookback_period'),
        prediction_horizon=model_parameters.get('prediction_horizon'),
        input_date=model_parameters.get('input_date'),
        chart_data=chart_data,
        ai_analysis=ai_analysis,
        performance_metrics=performance_metrics,
        feature_importance=feature_importance
    )

def _get_combined_predictions(model_type, model_key, trained_model):
    """Fetch and combine predictions from TrainedModel and PredictionsLog."""
    TrainedModelClass = app.TrainedModelsBinary if model_type == 'BINARY CLASSIFICATION' else app.TrainedModels
    predictions_json = trained_model_to_json(TrainedModelClass, model_key)

    PredictionsLog = app.PredictionsLog
    predictions_log_predictions = predictions_log_to_json(PredictionsLog, model_key)

    combined_predictions = predictions_json.copy()
    for date_str, pred in predictions_log_predictions.items():
        if date_str not in combined_predictions:
            combined_predictions[date_str] = pred
        else:
            combined_predictions[date_str].update(pred)

    logger.info(f"Combined predictions: {combined_predictions}")
    return combined_predictions

def trained_model_to_json(TrainedModelClass, model_key):
    """Convert TrainedModel predictions to JSON."""
    trained_model = TrainedModelClass.query.filter_by(model_key=model_key).first()
    if not trained_model:
        return {}

    predictions_json = trained_model.to_dict().get('predictions_json', {})
    if isinstance(predictions_json, str):
        try:
            predictions_json = json.loads(predictions_json)
        except json.JSONDecodeError:
            logger.error("Failed to decode predictions_json from TrainedModel.")
            predictions_json = {}
    return predictions_json

def predictions_log_to_json(PredictionsLogClass, model_key):
    """Convert PredictionsLog records to JSON."""
    predictions_log_records = PredictionsLogClass.query.filter_by(model_key=model_key).all()
    predictions_log_predictions = {}

    for record in predictions_log_records:
        record_predictions_json = record.to_dict().get('predictions_json', {})
        if isinstance(record_predictions_json, str):
            try:
                record_predictions_json = json.loads(record_predictions_json)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode predictions_json from PredictionsLog record ID {record.id}.")
                record_predictions_json = {}
        predictions_log_predictions.update(record_predictions_json)

    return predictions_log_predictions

def _prepare_chart_data(model_type, combined_predictions):
    """Prepare chart data for visualization."""
    chart_data = {
        'x': [],
        'actual_price': [],
        'predicted_price': [],
        'actual_movement': [],
        'predicted_movement': []
    }

    today = datetime.now().date()
    sorted_dates = sorted(combined_predictions.keys())
    logger.info(f"Processing {len(sorted_dates)} dates from combined predictions")

    for date_str in sorted_dates:
        prediction = combined_predictions[date_str]
        current_date = datetime.strptime(date_str, '%Y-%m-%d').date()

        chart_data['x'].append(date_str)

        if model_type == 'BINARY CLASSIFICATION':
            # Extract actual price if available
            actual_price = prediction.get('actual_price')
            chart_data['actual_price'].append(float(actual_price) if actual_price is not None else None)

            # Predicted movement
            predicted_movement = prediction.get('predicted_movement')
            try:
                predicted_value = int(predicted_movement) if predicted_movement is not None else None
            except (ValueError, TypeError):
                predicted_value = None
            chart_data['predicted_movement'].append(predicted_value)

            # Actual movement
            actual_movement = prediction.get('actual_movement')
            try:
                actual_value = int(actual_movement) if actual_movement is not None else None
            except (ValueError, TypeError):
                actual_value = None
            chart_data['actual_movement'].append(actual_value)
        else:
            # SARIMAX model: handle price predictions
            predicted_price = prediction.get('predicted_price')
            chart_data['predicted_price'].append(float(predicted_price) if predicted_price is not None else None)

            if current_date <= today:
                actual_price = prediction.get('actual_price')
                last_known_price = prediction.get('last_known_price')
                actual_value = actual_price if actual_price is not None else last_known_price
                chart_data['actual_price'].append(float(actual_value) if actual_value is not None else None)
            else:
                chart_data['actual_price'].append(None)

    logger.info(f"Prepared chart data: {chart_data}")
    return chart_data

def _fetch_ai_analysis(model_key):
    """Fetch and format AI analysis from the database."""
    AiAnalysis = app.AIAnalysis
    ai_analysis_record = AiAnalysis.query.filter_by(model_key=model_key).first()

    if not ai_analysis_record:
        logger.info("No AI analysis record found.")
        return {}

    raw_explanation = ai_analysis_record.to_dict().get('explanation', '')
    formatted_analysis = _parse_ai_explanation(raw_explanation)
    logger.info(f"Formatted AI analysis: {formatted_analysis}")
    return formatted_analysis

def _parse_ai_explanation(raw_explanation):
    """Parse raw AI explanation text into structured sections."""
    sections = {
        'conclusion': '',
        'performance_metrics': [],
        'key_features': [],
        'model_params': [],
        'additional_notes': ''
    }

    current_section = None

    for line in raw_explanation.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith("**Conclusion:**"):
            current_section = 'conclusion'
            continue
        elif line.startswith("**Performance Metrics:**"):
            current_section = 'performance_metrics'
            continue
        elif line.startswith("**Key Influencing Features:**"):
            current_section = 'key_features'
            continue
        elif line.startswith("**Model Parameters:**"):
            current_section = 'model_params'
            continue
        elif line.startswith("**Additional Notes:**"):
            current_section = 'additional_notes'
            continue

        if current_section == 'conclusion':
            sections['conclusion'] += line + ' '
        elif current_section == 'additional_notes':
            sections['additional_notes'] += line + ' '
        elif current_section in ['performance_metrics', 'model_params']:
            if line.startswith('-'):
                item = _parse_bullet_point(line)
                if item:
                    sections[current_section].append(item)
        elif current_section == 'key_features':
            # For key_features, process any line to extract features and their importances
            features = _parse_feature_line(line)
            if features:
                sections['key_features'].extend(features)
            else:
                # If no features found, treat it as additional notes
                sections['additional_notes'] += line + ' '

    # Trim trailing spaces
    for key, value in sections.items():
        if isinstance(value, str):
            sections[key] = value.strip()

    return sections

def _parse_feature_line(line):
    """Parse a line containing features and their importances."""
    pattern = r"'([^']+)'\s*\(([\d.]+)\)"
    matches = re.findall(pattern, line)
    if matches:
        features = []
        for feature_name, importance in matches:
            features.append({
                'name': feature_name.strip(),
                'value': importance.strip(),
                'numeric_value': float(importance.strip()),
                'description': ''
            })
        return features
    else:
        return None

def _parse_bullet_point(line):
    """Parse a bullet point line into a dictionary with name, value, and description."""
    try:
        line = line.lstrip('- ').replace('**', '').strip()
        if ':' in line:
            name, rest = line.split(':', 1)
            value_part, *description = rest.split('-', 1)
            value_str = value_part.strip()
            # Extract numeric value using regex (if needed)
            numeric_match = re.search(r'[\d.]+', value_str)
            numeric_value = float(numeric_match.group()) if numeric_match else None
            description = description[0].strip() if description else ''
            return {
                'name': name.strip(),
                'value': value_str,
                'numeric_value': numeric_value,
                'description': description
            }
        else:
            # If there's no colon, treat the entire line as a description
            return {
                'name': '',
                'value': '',
                'numeric_value': None,
                'description': line
            }
    except Exception as e:
        logger.warning(f"Failed to parse bullet point: {line}, Error: {e}")
        return None

def _extract_performance_metrics(model_type, trained_model_data):
    """Extract performance metrics from the trained model data."""
    performance_metrics = trained_model_data.copy()

    if model_type == 'BINARY CLASSIFICATION':
        confusion_matrix = trained_model_data.get('confusion_matrix')
        if confusion_matrix:
            if isinstance(confusion_matrix, str):
                try:
                    confusion_matrix = json.loads(confusion_matrix)
                except json.JSONDecodeError:
                    logger.error("Failed to decode confusion_matrix.")
                    confusion_matrix = [[0, 0], [0, 0]]
        else:
            confusion_matrix = [[0, 0], [0, 0]]
        performance_metrics['confusion_matrix'] = confusion_matrix

    return performance_metrics

def _fetch_feature_importance(trained_model_data):
    """Fetch and process feature importance data."""
    feature_importance = trained_model_data.get('feature_importance')
    if not feature_importance:
        return None

    try:
        features = feature_importance.get('feature', {})
        importances = feature_importance.get('importance', {})
        sorted_keys = sorted(features.keys(), key=lambda x: int(x))
        feature_importance_list = [
            {'feature': features.get(key), 'importance': float(importances.get(key, 0))}
            for key in sorted_keys
            if features.get(key) is not None
        ]

        # Sort by importance descending and take top 6
        feature_importance_sorted = sorted(
            feature_importance_list,
            key=lambda x: x['importance'],
            reverse=True
        )[:6]

        return {
            'feature': [item['feature'] for item in feature_importance_sorted],
            'importance': [item['importance'] for item in feature_importance_sorted]
        }
    except (ValueError, TypeError) as e:
        logger.error(f"Error processing feature importance: {e}")
        return None


# Add these routes for 'about' and 'contact' pages
@main_bp.route('/about')
def about():
    return render_template('about.html')

@main_bp.route('/contact', methods=['GET', 'POST'])
def contact():
    form = ContactForm()
    if form.validate_on_submit():
        recaptcha_token = request.form.get('recaptcha_token')
        if not recaptcha_token:
            flash('reCAPTCHA token missing. Please try again.', 'danger')
            return redirect(url_for('main_bp.contact'))

        if not verify_recaptcha(recaptcha_token, 'contact'):
            flash('reCAPTCHA verification failed. Please try again.', 'danger')
            return redirect(url_for('main_bp.contact'))

        # Save to the database
        contact_message = app.ContactMessage(
            name=form.name.data,
            email=form.email.data,
            message=form.message.data,
            date_submitted=datetime.now()
        )
        db.session.add(contact_message)
        db.session.commit()
        flash('Your message has been sent successfully!', 'success')
        return redirect(url_for('main_bp.contact'))
    return render_template('contact.html', form=form)

@main_bp.route('/api-docs')
def api_docs():
    return render_template('api_docs.html')

@main_bp.route('/meta-analysis')
def meta_analysis():
    return render_template('meta-analysis.html')