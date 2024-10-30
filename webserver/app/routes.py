from flask import Blueprint, render_template, request, redirect, url_for, session, current_app as app, flash
from .forms import PredictionForm, ContactForm
from .generate_stock_prediction import generate_stock_prediction, generate_model_key
from datetime import datetime, timedelta, date
import json
import logging
import re
import requests
from app import db
from sqlalchemy import desc

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


@main_bp.route('/loading')
def loading():
    return render_template('loading.html')

@main_bp.route('/results')
def results():
    # Check if model_key and model_type are provided as query parameters
    model_key = request.args.get('model_key')
    model_type = request.args.get('model_type')

    if model_key and model_type:
        try:
            if model_type == 'BINARY CLASSIFICATION':
                TrainedModel = app.TrainedModelsBinary
            elif model_type == 'SARIMAX':
                TrainedModel = app.TrainedModels
            else:
                flash('Invalid model type provided.', 'danger')
                return redirect(url_for('main_bp.index'))

            # Fetch the trained model from the database
            trained_model = TrainedModel.query.filter_by(model_key=model_key).first()
            if not trained_model:
                flash('Model not found.', 'danger')
                return redirect(url_for('main_bp.index'))

            trained_model_data = trained_model.to_dict()

            # Fetch combined predictions from the trained model data and PredictionsLog
            combined_predictions = _get_combined_predictions(model_type, model_key, trained_model)

            # Prepare chart data
            chart_data = _prepare_chart_data(model_type, combined_predictions)

            # Fetch AI analysis
            ai_analysis = _fetch_ai_analysis(model_key)

            # Extract performance metrics
            performance_metrics = _extract_performance_metrics(model_type, trained_model_data)

            # Fetch feature importance
            feature_importance = _fetch_feature_importance(trained_model_data)

            return render_template(
                'results.html',
                stock_symbol=trained_model.symbol,
                model_type=model_type,
                hyperparameter_tuning=trained_model_data.get('hyperparameter_tuning'),
                feature_set=trained_model_data.get('feature_set'),
                lookback_period=trained_model_data.get('lookback_period'),
                prediction_horizon=trained_model_data.get('prediction_horizon'),
                chart_data=chart_data,
                ai_analysis=ai_analysis,
                performance_metrics=performance_metrics,
                feature_importance=feature_importance
            )

        except Exception as e:
            logger.error(f"Error fetching model results: {str(e)}")
            flash('An error occurred while fetching the results.', 'danger')
            return redirect(url_for('main_bp.index'))

    # Existing /results route logic
    form_data = session.get('form_data', {})
    if not form_data:
        logger.error("No form data found in session.")
        return redirect(url_for('main_bp.index'))

    logger.info(f"Form data retrieved from session: {form_data}")

    # Extract form data with default values and type casting
    model_type = form_data.get('model_type')
    stock_symbol = form_data.get('stock_symbol', '').upper()
    feature_set = form_data.get('feature_set')
    hyperparameter_tuning = form_data.get('hyperparameter_tuning')
    lookback_period = int(form_data.get('lookback_period', 0))
    prediction_horizon = int(form_data.get('prediction_horizon', 0))
    input_date_str = form_data.get('input_date')

    # Change input_date_str string to be in the format of 'YYYY-MM-DD'
    input_date_str_updated = datetime.strptime(input_date_str, '%Y-%m-%d').strftime('%Y-%m-%d')

    if not all([model_type, stock_symbol, feature_set, hyperparameter_tuning, lookback_period, prediction_horizon, input_date_str]):
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

    # Determine the appropriate TrainedModel based on model_type
    TrainedModel = app.TrainedModelsBinary if model_type == 'BINARY CLASSIFICATION' else app.TrainedModels

    trained_model = TrainedModel.query.filter_by(model_key=model_key).first()

    if not trained_model:
        error_message = 'No trained model found in the database.'
        logger.error(error_message)
        return render_template('error.html', error_message=error_message)

    trained_model_data = trained_model.to_dict()
    logger.info(f"Trained model data retrieved: {trained_model_data}")

    # Fetch combined predictions from the trained model data and PredictionsLog
    combined_predictions = _get_combined_predictions(model_type, model_key, trained_model)

    # Prepare chart data
    chart_data = _prepare_chart_data(model_type, combined_predictions)

    # Fetch AI analysis
    ai_analysis = _fetch_ai_analysis(model_key)

    # Extract performance metrics
    performance_metrics = _extract_performance_metrics(model_type, trained_model_data)

    # Fetch feature importance
    feature_importance = _fetch_feature_importance(trained_model_data)

    return render_template(
        'results.html',
        stock_symbol=stock_symbol,
        model_type=model_type,
        hyperparameter_tuning=hyperparameter_tuning,
        feature_set=feature_set,
        lookback_period=lookback_period,
        prediction_horizon=prediction_horizon,
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

@main_bp.route('/test_sarimax_results')
def test_results():
    # Define Test Data
    chart_data = {
        'x': [
            '2024-05-29', '2024-05-30', '2024-05-31', '2024-06-03', '2024-06-04',
            '2024-06-05', '2024-06-06', '2024-06-07', '2024-06-10', '2024-06-11',
            '2024-06-12', '2024-06-13', '2024-06-14', '2024-06-17', '2024-06-18',
            '2024-06-20', '2024-06-21', '2024-06-24', '2024-06-25', '2024-06-26',
            '2024-06-27', '2024-06-28', '2024-07-01', '2024-07-02', '2024-07-03',
            '2024-07-05', '2024-07-08', '2024-07-09', '2024-07-10', '2024-07-11',
            '2024-07-12', '2024-07-15', '2024-07-16', '2024-07-17', '2024-07-18',
            '2024-07-19', '2024-07-22', '2024-07-23', '2024-07-24', '2024-07-25',
            '2024-07-26', '2024-07-29', '2024-07-30', '2024-07-31', '2024-08-01',
            '2024-08-02', '2024-08-05', '2024-08-06', '2024-08-07', '2024-08-08',
            '2024-08-09', '2024-08-12', '2024-08-13', '2024-08-14', '2024-08-15',
            '2024-08-16', '2024-08-19', '2024-08-20', '2024-08-21', '2024-08-22',
            '2024-08-23', '2024-08-26', '2024-08-27', '2024-08-28', '2024-08-29',
            '2024-08-30', '2024-09-03', '2024-09-04', '2024-09-05', '2024-09-06',
            '2024-09-09', '2024-09-10', '2024-09-11', '2024-09-12', '2024-09-13',
            '2024-09-16', '2024-09-17', '2024-09-18', '2024-09-19', '2024-09-20',
            '2024-09-23', '2024-09-24', '2024-09-25', '2024-09-26', '2024-09-27',
            '2024-09-30', '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04',
            '2024-10-07', '2024-10-08', '2024-10-09', '2024-10-10', '2024-10-11',
            '2024-10-14', '2024-10-15', '2024-10-16', '2024-10-17', '2024-10-18',
            '2024-10-21', '2024-10-22', '2024-10-23', '2024-10-24', '2024-10-25',
            '2024-10-28', '2024-10-29', '2024-10-30', '2024-10-31', '2024-11-01',
            '2024-11-04', '2024-11-05', '2024-11-06', '2024-11-07', '2024-11-08',
            '2024-11-11', '2024-11-12', '2024-11-13', '2024-11-14', '2024-11-15',
            '2024-11-18', '2024-11-19', '2024-11-20', '2024-11-21', '2024-11-22',
            '2024-11-23'
        ],
        'actual': [
            166.61, 164.58, 166.95, 169.44000000000003, 168.44, 167.96, 168.25,
            166.62, 168.0, 166.9, 169.11, 169.93, 161.7, 160.76, 165.69,
            170.08000000000004, 168.06, 168.09, 170.02, 170.87, 170.54, 166.81,
            167.29, 168.80000000000004, 167.92, 167.89, 168.42, 170.41, 170.16,
            170.15, 169.16999999999996, 170.35, 169.25, 169.06, 170.03,
            171.53999999999996, 174.52, 175.90000000000003, 175.46999999999997,
            175.59, 176.06, 177.79, 173.92, 173.47, 174.08, 177.24, 175.88,
            173.92, 171.54, 174.22, 173.77000000000004, 173.24, 172.26, 173.21,
            173.55, 173.2, 173.04, 171.91999999999996, 170.12, 168.88, 167.12,
            168.16, 169.27, 168.95, 171.09000000000003, 172.51000000000002,
            173.57, 172.38, 172.28, 171.28, 169.54, 169.7, 169.58, 169.62,
            168.8, 167.89, 167.89, 168.42, 170.41, 170.16, 169.17, 169.17,
            170.35, 169.25, 169.06, 171.54, 171.54, 171.54, 174.52, 175.9,
            175.59, 175.59, 176.06, 177.79, 173.92, 174.08, 174.08, 177.24,
            175.88, 173.92, 174.22, 174.22, 173.77, 173.24, 172.26, None, None,
            None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None
        ],
        'predicted': [
            158.3242691246887, 158.5443892263691, 159.08683150084354, 155.84165572266488,
            157.87724813739888, 158.67303499640232, 160.21120912045492, 160.60618754934472,
            161.21025876176492, 161.33526748636876, 159.94576954451978, 161.04063168656944,
            162.1112618986791, 162.16153888316657, 164.34187144364307, 165.33323247471236,
            166.33853111980625, 166.36708953446418, 166.03923319918712, 165.4057198873094,
            163.8693494240231, 162.44878019553488, 158.78982124255674, 160.0921462336533,
            160.03025220659063, 160.4812111582191, 161.4170926676706, 160.59953718418186,
            161.07854691506466, 162.2138274590717, 159.81672397724915, 160.3903361061715,
            160.7483843883044, 161.97121117718274, 163.99641827120814, 166.49330545695662,
            167.24490259572653, 166.30675798927044, 166.93207338120413, 166.51824090217298,
            166.575413439784, 166.85152541504445, 166.66131565662337, 163.17603164094237,
            170.28149901726113, 171.64025421942924, 170.55380836753727, 168.900172843695,
            171.07380128299843, 171.09659771046356, 171.49872230038932, 169.46804083150067,
            169.8470612387839, 170.7938065867536, 171.58517656931846, 171.5324745243558,
            172.24323974487893, 173.53042838268092, 172.56654180173533, 173.05990853399723,
            173.9176639758986, 174.29663661117206, 172.18066875101792, 169.2665118036924,
            169.34923840181273, 170.48751652951762, 174.07205596834334, 174.3139611243445,
            175.912672823138, 177.48709282728444, 177.73996940652904, 178.60079858467498,
            175.23039270979737, 175.48762939507222, 164.8783040297372, 163.08450038009983,
            163.08450038009983, 163.2588669269152, 165.20443627043883, 164.32329335936404,
            162.2723426166247, 162.2723426166247, 166.0134498078945, 162.9629462682379,
            161.60796027654067, 170.04204246332972, 170.04204246332972,
            170.04204246332972, 173.47744569430242, 175.56931084749627,
            175.18836294335486, 175.18836294335486, 173.4212474170579,
            174.56913180698265, 179.25625164945808, 178.90399679381252,
            178.90399679381252, 178.86593684955403, 178.48984709887958,
            174.75827208857265, 174.4510397038224, 174.4510397038224,
            174.74687978841268, 176.00049939277613, 174.89302804841896,
            177.33806141575565, 177.33806141575565, 176.95767750058292,
            176.96811601355714, 175.06807792814536, 171.768188551206,
            171.768188551206, 171.81013148525219, 172.05589881522042,
            172.20101466991395, 170.40499414501292, 170.40499414501292,
            171.45156491356335, 173.12449421734857, 170.82569981143277,
            173.13052713762292, 173.13052713762292, 171.15775539834718,
            170.47390464442103, 170.85769305646312, 171.59285004895457
        ]
    }

    ai_analysis =  {'conclusion': "The SARIMAX model forecasts the stock price for PG to be **171.59285004895457** by 2024-11-23. The model's prediction suggests a positive outlook for PG's stock price in the short term. However, the high MAPE indicates that while the model is confident in its prediction, there may be significant variability in actual outcomes. Investors should consider this forecast alongside other market analyses and economic indicators. The model's reliance on technical indicators and economic factors highlights the importance of both market trends and broader economic conditions in influencing stock prices. Investors should consider these insights along with other market factors before making decisions.",
                    'performance_metrics': [{'name': 'RMSE', 'value': '0.03917068745089896', 'description': "Measures the model's prediction error. Lower values indicate better performance."}, {'name': 'MAE', 'value': '0.03421618231720002', 'description': 'Represents the average absolute error. Lower values are preferable.'}, {'name': 'MAPE', 'value': '570.6641466861464%', 'description': 'Shows the mean absolute percentage error. Values closer to 0% are better.'}, {'name': 'Confidence Score', 'value': '0.962305819511725', 'description': "Reflects the model's confidence in its predictions. Higher scores denote higher confidence."}, {'name': 'Predicted Amount', 'value': '171.59285004895457', 'description': ''}, {'name': 'Last Known Price', 'value': '169.62', 'description': ''}],
                    'key_features': [{'name': "feature ({'0'", 'value': "'volume', '1': 'open', '2': 'high', '3': 'low', '4': 'close', '5': 'sp500_return', '6': 'nasdaq_return', '7': 'gdp_growth', '8': 'inflation_rate', '9': 'unemployment_rate', '10': 'macd_hist', '11': 'adx', '12': 'macd', '13': 'rsi', '14': 'upper_band', '15': 'lower_band', '16': 'macd_signal', '17': 'rolling_volatility_60', '18': 'corr_sp500_60', '19': 'corr_nasdaq_60', '20': 'sma_50', '21': 'sma_100', '22': 'ema_50', '23': '%K', '24': '%D', '25': 'cci', '26': 'momentum_30', '27': 'momentum_60'}), importance ({'0': 0.01299426799974595, '1': 0.03980111304397407, '2': 0.02798965607110957, '3': 0.028641861236531143, '4': 0.037898004103213166, '5': 0.004827633372332341, '6': 0.005294377004611669, '7': 0.01659194144109923, '8': 0.0, '9': 0.0026658746922778955, '10': 0.03510005155361047, '11': 0.0737534015269129, '12': 0.16575141706175925, '13': 0.011464836897147225, '14': 0.09816425966874083, '15': 0.008273895706457314, '16': 0.03822024830734832, '17': 0.07833191695220809, '18': 0.010676187197336698, '19': 0.05951244893173734, '20': 0.024843393774010226, '21': 0.03254327771736324, '22': 0.033043673681418854, '23': 0.00898072163001447, '24': 0.010576769273065488, '25': 0.010385381552161427, '26': 0.10299136405481263, '27': 0.020682025549000345})", 'description': ''}], 'model_params': [{'name': 'Order', 'value': '[0, 0, 0]', 'description': ''}, {'name': 'Seasonal Order', 'value': '[0, 0, 0, 7]', 'description': ''}, {'name': 'Hyperparameter Tuning', 'value': 'LOW', 'description': ''}], 'additional_notes': ''}


    performance_metrics = {
        'symbol': 'PG',
        'prediction_date': date(2024, 11, 23),
        'prediction_explanation': 'Regression Prediction Based on SARIMAX model with feature engineering',
        'prediction_rmse': '0.03917068745089896',
        'prediction_mae': '0.03421618231720002',
        'prediction_mape': '570.6641466861464',
        'prediction_confidence_score': '0.962305819511725',
        'feature_importance': {
            'feature': {
                '0': 'volume', '1': 'open', '2': 'high', '3': 'low', '4': 'close',
                '5': 'sp500_return', '6': 'nasdaq_return', '7': 'gdp_growth',
                '8': 'inflation_rate', '9': 'unemployment_rate', '10': 'macd_hist',
                '11': 'adx', '12': 'macd', '13': 'rsi', '14': 'upper_band',
                '15': 'lower_band', '16': 'macd_signal', '17': 'rolling_volatility_60',
                '18': 'corr_sp500_60', '19': 'corr_nasdaq_60', '20': 'sma_50',
                '21': 'sma_100', '22': 'ema_50', '23': '%K', '24': '%D',
                '25': 'cci', '26': 'momentum_30', '27': 'momentum_60'
            },
            'importance': {
                '0': 0.01299426799974595, '1': 0.03980111304397407, '2': 0.02798965607110957,
                '3': 0.028641861236531143, '4': 0.037898004103213166, '5': 0.004827633372332341,
                '6': 0.005294377004611669, '7': 0.01659194144109923, '8': 0.0,
                '9': 0.0026658746922778955, '10': 0.03510005155361047, '11': 0.0737534015269129,
                '12': 0.16575141706175925, '13': 0.011464836897147225, '14': 0.09816425966874083,
                '15': 0.008273895706457314, '16': 0.03822024830734832, '17': 0.07833191695220809,
                '18': 0.010676187197336698, '19': 0.05951244893173734, '20': 0.024843393774010226,
                '21': 0.03254327771736324, '22': 0.033043673681418854, '23': 0.00898072163001447,
                '24': 0.010576769273065488, '25': 0.010385381552161427, '26': 0.10299136405481263,
                '27': 0.020682025549000345
            }
        },
        'model_parameters': {
            'order': [0, 0, 0],
            'feature_set': 'basic',
            'seasonal_order': [0, 0, 0, 7],
            'prediction_horizon': 30,
            'hyperparameter_tuning': 'LOW'
        },
        'predicted_amount': 171.59285004895457,
        'last_known_price': 169.62,
        'model_location': 's3://trained-models-stock-prediction/sarimax_model_SARIMAX_PG_basic_LOW_720_30_20241025.pkl',
        'date_created': '2024-10-25 14:14:59',
        'model_key': 'SARIMAX_PG_basic_LOW_720_30_20241025',
        'predictions_json': {
            '2024-05-29': {'actual_price': 166.61, 'predicted_price': 158.3242691246887},
            '2024-05-30': {'actual_price': 164.58, 'predicted_price': 158.5443892263691},
            '2024-05-31': {'actual_price': 166.95, 'predicted_price': 159.08683150084354},
            # ... (Truncated for brevity; include all data as needed)
            '2024-11-23': {'actual_price': None, 'predicted_price': 171.59285004895457}
        }
    }

    feature_importance = {
        'feature': ['macd', 'momentum_30', 'upper_band', 'rolling_volatility_60', 'adx', 'corr_nasdaq_60'],
        'importance': [0.16575141706175925, 0.10299136405481263, 0.09816425966874083,
                       0.07833191695220809, 0.0737534015269129, 0.05951244893173734]
    }

    # Additional Static Data for Rendering
    stock_symbol = 'PG'
    model_type = 'SARIMAX'
    hyperparameter_tuning = 'LOW'
    feature_set = 'basic'
    lookback_period = 720
    prediction_horizon = 30

    return render_template(
        'results.html',
        stock_symbol=stock_symbol,
        model_type=model_type,
        hyperparameter_tuning=hyperparameter_tuning,
        feature_set=feature_set,
        lookback_period=lookback_period,
        prediction_horizon=prediction_horizon,
        chart_data=chart_data,
        ai_analysis=ai_analysis,
        performance_metrics=performance_metrics,
        feature_importance=feature_importance
    )