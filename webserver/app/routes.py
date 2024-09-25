from flask import Blueprint, render_template
from app.models import Stock, ModelStatistics
from app import db
from app.live_stock_updates import update_stock_data

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    stocks = Stock.query.all()
    for stock in stocks:
        update_stock_data(stock)
    return render_template('index.html', stocks=stocks)

@bp.route('/about')
def about():
    mse_over_time = ModelStatistics.query.filter_by(metric='mse').all()
    r_squared_scores = ModelStatistics.query.filter_by(metric='r_squared').all()
    feature_importances = [
        {'feature': 'Price History', 'importance': 0.35},
        {'feature': 'Trading Volume', 'importance': 0.25},
        {'feature': 'Market Sentiment', 'importance': 0.20},
        {'feature': 'Moving Averages', 'importance': 0.15},
        {'feature': 'Volatility Index', 'importance': 0.05}
    ]
    predicted_vs_actual = [
        {'date': '2023-11-15', 'predicted': 149.50, 'actual': 150.25},
        {'date': '2023-11-16', 'predicted': 151.00, 'actual': 151.75},
        {'date': '2023-12-15', 'predicted': 155.75, 'actual': 155.50}
    ]
    return render_template(
        'about.html',
        mse_over_time=mse_over_time,
        r_squared_scores=r_squared_scores,
        feature_importances=feature_importances,
        predicted_vs_actual=predicted_vs_actual
    )