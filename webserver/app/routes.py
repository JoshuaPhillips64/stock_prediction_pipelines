from datetime import datetime, timedelta
from flask import Blueprint, render_template, jsonify, current_app
from sqlalchemy import func
from app import db
from app.models import EnrichedStockData, CompanyOverview
from config import Config
import json

bp = Blueprint("main", __name__)

@bp.route("/")
@bp.route("/index")
def index():
    stocks = []
    today = db.session.query(func.max(EnrichedStockData.date)).scalar()
    thirty_days_ago = today - timedelta(days=30)

    for symbol in Config.STOCKS:
        latest_data = (
            EnrichedStockData.query.filter_by(symbol=symbol)
                .order_by(EnrichedStockData.date.desc())
                .first()
        )

        historical_data = (
            EnrichedStockData.query.filter(
                EnrichedStockData.symbol == symbol,
                EnrichedStockData.date >= thirty_days_ago
            )
                .order_by(EnrichedStockData.date)
                .all()
        )

        prediction = (
            current_app.AiStockPredictions.query.filter_by(symbol=symbol)
                .order_by(current_app.AiStockPredictions.prediction_date.desc())
                .first()
        )

        if latest_data and historical_data and prediction:
            stocks.append({
                "symbol": symbol,
                "current_price": latest_data.close,
                "prediction": prediction.predicted_amount,
                "historical_data": [
                    {
                        "date": data.date.isoformat(),
                        "open": data.open,
                        "high": data.high,
                        "low": data.low,
                        "close": data.close
                    } for data in historical_data
                ],
            })

    return render_template("index.html", stocks=stocks)

@bp.route("/ai-prediction")
def ai_prediction():
    stocks_data = []
    today = db.session.query(func.max(EnrichedStockData.date)).scalar()
    ninety_days_ago = today - timedelta(days=90)
    one_twenty_days_ago = today - timedelta(days=120)

    for symbol in Config.STOCKS:
        stock_data = (
            EnrichedStockData.query.filter(
                EnrichedStockData.symbol == symbol,
                EnrichedStockData.date >= ninety_days_ago
            )
            .order_by(EnrichedStockData.date)
            .all()
        )

        chart_data = {
            'x': [data.date.strftime('%Y-%m-%d') for data in stock_data],
            'open': [float(data.open) for data in stock_data],
            'high': [float(data.high) for data in stock_data],
            'low': [float(data.low) for data in stock_data],
            'close': [float(data.close) for data in stock_data],
            'volume': [float(data.volume) for data in stock_data],
        }

        prediction_data = (
            current_app.AiStockPredictions.query.filter(
                current_app.AiStockPredictions.symbol == symbol,
                current_app.AiStockPredictions.prediction_date >= one_twenty_days_ago
            )
            .order_by(current_app.AiStockPredictions.prediction_date)
            .all()
        )

        prediction_chart_data = []
        for prediction in prediction_data:
            if isinstance(prediction.feature_importance, str):
                feature_importance = json.loads(prediction.feature_importance)
            else:
                feature_importance = prediction.feature_importance

            prediction_chart_data.append({
                'date': prediction.prediction_date.strftime('%Y-%m-%d'),
                'predicted_amount': float(prediction.predicted_amount),
                'prediction_confidence_score': float(prediction.prediction_confidence_score),
                'prediction_rmse': float(prediction.prediction_rmse),
                'up_or_down': prediction.up_or_down,
                'prediction_explanation': prediction.prediction_explanation,
                'feature_importance': feature_importance
            })

        company_overview = CompanyOverview.query.filter_by(symbol=symbol).order_by(CompanyOverview.last_updated.desc()).first()

        stocks_data.append({
            'symbol': symbol,
            'chart_data': chart_data,
            'prediction_chart_data': prediction_chart_data,
            'company_overview': company_overview.data if company_overview else None
        })

    return render_template("ai-prediction.html", stocks_data=stocks_data)

@bp.route("/about")
def about():
    return render_template("about.html")