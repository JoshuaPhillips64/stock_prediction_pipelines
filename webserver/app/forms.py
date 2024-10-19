# app/forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, DateField, IntegerField, SubmitField
from wtforms.validators import DataRequired

class PredictionForm(FlaskForm):
    model_type = SelectField('Model Type', choices=[('SARIMAX', 'Sarimax'), ('BINARY CLASSIFICATION', 'Binary Classification')], validators=[DataRequired()])
    stock_symbol = StringField('Stock Symbol', validators=[DataRequired()], render_kw={"placeholder": "e.g., AAPL"})
    input_date = DateField('Input Date', validators=[DataRequired()], format='%Y-%m-%d')
    hyperparameter_tuning = SelectField('Hyperparameter Tuning', choices=[('LOW', 'Low'), ('MEDIUM', 'Medium'), ('HIGH', 'High')], default='MEDIUM', validators=[DataRequired()])
    feature_set = SelectField('Feature Set', choices=[('basic', 'Basic'), ('advanced', 'Advanced')], default='basic', validators=[DataRequired()])
    lookback_period = IntegerField('Lookback Period (days)', default=720, validators=[DataRequired()])
    prediction_horizon = IntegerField('Prediction Horizon (days)', default=30, validators=[DataRequired()])
    submit = SubmitField('Generate Prediction')