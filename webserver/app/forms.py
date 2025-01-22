# app/forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, DateField, IntegerField, SubmitField, TextAreaField, EmailField
from wtforms.validators import DataRequired, Length, NumberRange, Email
from datetime import datetime, timedelta

class PredictionForm(FlaskForm):
    model_type = SelectField('Model Type', choices=[('SARIMAX', 'Regression'), ('BINARY CLASSIFICATION', 'Binary Classification')], validators=[DataRequired()])
    stock_symbol = StringField('Stock Symbol', validators=[DataRequired(), Length(min=1, max=6, message="Symbol must be between 1 and 6 characters")], render_kw={"placeholder": "e.g., PG"})
    input_date = DateField('Start Date', validators=[DataRequired()], format='%Y-%m-%d', default=datetime.now() - timedelta(days=1))
    ## TAKING OUT THE HIGH OPTION ('HIGH', 'High') add back in if needed. Removed due to high taking longer to run
    hyperparameter_tuning = SelectField('Hyperparameter Tuning', choices=[('LOW', 'Low'), ('MEDIUM', 'Medium')], default='LOW', validators=[DataRequired()])
    #This field is hidden via the html file due to advanced taking longer to run
    feature_set = SelectField('Feature Set', choices=[('basic', 'Basic'), ('advanced', 'Advanced')], default='basic', validators=[DataRequired()])
    lookback_period = IntegerField('Training Data Period (days backward)', default=720, validators=[DataRequired(), NumberRange(min=365, max=2000, message="Lookback period must be between 365 and 2000")])
    prediction_horizon = IntegerField('Prediction Horizon (days forward)', default=30, validators=[DataRequired(), NumberRange(min=7, max=60, message="Prediction horizon must be between 7 and 60")])
    submit_button = SubmitField('Create Model')

class ContactForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=1, max=100)])
    email = EmailField('Email', validators=[DataRequired(), Email(), Length(max=100)])
    message = TextAreaField('Message', validators=[DataRequired(), Length(min=1, max=2000)])
    submit_button = SubmitField('Send Message')