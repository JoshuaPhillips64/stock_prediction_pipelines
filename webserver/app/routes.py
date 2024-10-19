from flask import Blueprint, render_template, request, redirect, url_for, session
from .forms import PredictionForm
from .models import db, PredictionResult
from .generate_stock_prediction import generate_stock_prediction  # Updated import
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
    # Retrieve form data from session
    # Retrieve form data from session
    form_data = session.get('form_data', {})
    if not form_data:
        return redirect(url_for('main_bp.index'))

    # Debugging statement
    print(f"form_data: {form_data}")

    # Call the generate_stock_prediction function
    result = generate_stock_prediction(
        model_type=form_data.get('model_type'),
        stock_symbol=form_data.get('stock_symbol'),
        input_date=form_data.get('input_date'),
        hyperparameter_tuning=form_data.get('hyperparameter_tuning'),
        feature_set=form_data.get('feature_set'),
        lookback_period=int(form_data.get('lookback_period')),
        prediction_horizon=int(form_data.get('prediction_horizon'))
    )

    # Debugging statement
    print(f"result: {result}")

    if not result:
        error_message = 'Prediction generation failed.'
        return render_template('error.html', error_message=error_message)

    # Check for errors in result
    if 'error' in result:
        error_message = result['error']
        return render_template('error.html', error_message=error_message)

    # Save result to database
    prediction_result = PredictionResult(
        model_key=result['model_key'],
        model_type=result['model_type'],
        stock_symbol=result['stock_symbol'],
        input_date=datetime.datetime.strptime(result['input_date'], '%Y-%m-%d'),
        hyperparameter_tuning=result['hyperparameter_tuning'],
        feature_set=result['feature_set'],
        lookback_period=result['lookback_period'],
        prediction_horizon=result['prediction_horizon'],
        prediction_data=json.dumps(result['prediction_result']),
        date_created=datetime.datetime.utcnow()
    )
    db.session.add(prediction_result)
    db.session.commit()

    # Extract data for charting
    prediction_dates = [item['date'] for item in result['prediction_result']['predictions']]
    actual_prices = [item['actual_price'] for item in result['prediction_result']['predictions']]
    predicted_prices = [item['predicted_price'] for item in result['prediction_result']['predictions']]

    return render_template('results.html',
                           stock_symbol=result['stock_symbol'],
                           model_type=result['model_type'],
                           hyperparameter_tuning=result['hyperparameter_tuning'],
                           feature_set=result['feature_set'],
                           lookback_period=result['lookback_period'],
                           prediction_horizon=result['prediction_horizon'],
                           prediction_dates=prediction_dates,
                           actual_prices=actual_prices,
                           predicted_prices=predicted_prices)

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