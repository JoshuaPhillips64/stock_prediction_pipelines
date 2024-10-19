from flask import Blueprint, render_template, request, redirect, url_for, session, current_app as app
from .forms import PredictionForm
from .generate_stock_prediction import generate_stock_prediction, generate_model_key
import datetime
import json

main_bp = Blueprint('main_bp', __name__)

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    form = PredictionForm()
    if form.validate_on_submit():
        # Save form data to session
        session['form_data'] = request.form
        return redirect(url_for('main_bp.loading'))
    return render_template('index.html', form=form)

@main_bp.route('/loading')
def loading():
    return render_template('loading.html')

@main_bp.route('/results')
def results():
    form_data = session.get('form_data', {})
    if not form_data:
        return redirect(url_for('main_bp.index'))

    # Generate model key
    model_key = generate_model_key(
        form_data.get('model_type'),
        form_data.get('stock_symbol'),
        form_data.get('feature_set'),
        form_data.get('hyperparameter_tuning'),
        int(form_data.get('lookback_period')),
        int(form_data.get('prediction_horizon'))
    )

    # Call the generate_stock_prediction function
    result = generate_stock_prediction(
        model_type=form_data.get('model_type'),
        stock_symbol=form_data.get('stock_symbol'),
        input_date=form_data.get('input_date'),
        hyperparameter_tuning=form_data.get('hyperparameter_tuning'),
        feature_set=form_data.get('feature_set'),
        lookback_period=int(form_data.get('lookback_period')),
        prediction_horizon=int(form_data.get('prediction_horizon')),
        make_prediction_step=True
    )

    if not result:
        error_message = 'Prediction generation failed.'
        return render_template('error.html', error_message=error_message)

    if 'error' in result:
        error_message = result['error']
        return render_template('error.html', error_message=error_message)

    stock_symbol = result['stock_symbol']
    model_type = result['model_type']
    hyperparameter_tuning = result['hyperparameter_tuning']
    feature_set = result['feature_set']
    lookback_period = result['lookback_period']
    prediction_horizon = result['prediction_horizon']

    # Fetch trained model data
    if model_type == 'SARIMAX':
        TrainedModel = app.TrainedModels
    elif model_type == 'BINARY CLASSIFICATION':
        TrainedModel = app.TrainedModelsBinary
    else:
        error_message = 'Invalid model type.'
        return render_template('error.html', error_message=error_message)

    trained_model = TrainedModel.query.filter_by(model_key=model_key).first()

    if not trained_model:
        error_message = 'No trained model found in the database.'
        return render_template('error.html', error_message=error_message)

    trained_model_data = trained_model.to_dict()

    # Fetch predictions from PredictionsLog
    PredictionsLog = app.PredictionsLog
    predictions = PredictionsLog.query.filter_by(model_key=model_key).order_by(PredictionsLog.date).all()

    if not predictions:
        error_message = 'No predictions found in the database.'
        return render_template('error.html', error_message=error_message)

    # Extract data for charting
    prediction_dates = []
    actual_prices = []
    predicted_prices = []

    for pred in predictions:
        date_str = pred.date
        prediction_dates.append(date_str)
        actual_price = float(pred.last_known_price) if pred.last_known_price else None
        predicted_price = float(pred.predicted_price) if pred.predicted_price else None
        actual_prices.append(actual_price)
        predicted_prices.append(predicted_price)

    # Fetch AI analysis from ai_analysis table
    AiAnalysis = app.AiAnalysis  # Updated model name
    ai_analysis_record = AiAnalysis.query.filter_by(model_key=model_key).first()
    if ai_analysis_record:
        ai_analysis = ai_analysis_record.to_dict()
    else:
        ai_analysis = {}

    # Extract performance metrics
    performance_metrics = trained_model_data

    return render_template('results.html',
                           stock_symbol=stock_symbol,
                           model_type=model_type,
                           hyperparameter_tuning=hyperparameter_tuning,
                           feature_set=feature_set,
                           lookback_period=lookback_period,
                           prediction_horizon=prediction_horizon,
                           prediction_dates=prediction_dates,
                           actual_prices=actual_prices,
                           predicted_prices=predicted_prices,
                           ai_analysis=ai_analysis,
                           performance_metrics=performance_metrics)

# Add these routes for 'about' and 'contact' pages
@main_bp.route('/about')
def about():
    return render_template('about.html')

@main_bp.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Handle form submission if needed
        # For now, we'll just redirect back to the contact page
        return redirect(url_for('main_bp.contact'))
    return render_template('contact.html')