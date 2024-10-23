# app/forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, DateField, IntegerField, SubmitField
from wtforms.validators import DataRequired, Length, NumberRange

class PredictionForm(FlaskForm):
    model_type = SelectField('Model Type', choices=[('SARIMAX', 'Sarimax'), ('BINARY CLASSIFICATION', 'Binary Classification')], validators=[DataRequired()])
    stock_symbol = StringField('Stock Symbol', validators=[DataRequired(), Length(min=1, max=6, message="Symbol must be between 1 and 6 characters")], render_kw={"placeholder": "e.g., AAPL"})
    input_date = DateField('Input Date', validators=[DataRequired()], format='%Y-%m-%d')
    hyperparameter_tuning = SelectField('Hyperparameter Tuning', choices=[('LOW', 'Low'), ('MEDIUM', 'Medium'), ('HIGH', 'High')], default='MEDIUM', validators=[DataRequired()])
    feature_set = SelectField('Feature Set', choices=[('basic', 'Basic'), ('advanced', 'Advanced')], default='basic', validators=[DataRequired()])
    lookback_period = IntegerField('Lookback Period (days)', default=720, validators=[DataRequired(), NumberRange(min=365, max=200, message="Lookback period must be between 365 and 2000")])
    prediction_horizon = IntegerField('Prediction Horizon (days)', default=30, validators=[DataRequired(), NumberRange(min=7, max=60, message="Prediction horizon must be between 7 and 60")])
    submit = SubmitField('Generate Prediction')