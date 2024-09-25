from datetime import timedelta
from flask import Blueprint, render_template
from app import db
from app.models import EnrichedStockData

bp = Blueprint("main", __name__)

@bp.route("/")
def index():
    # Fetch all unique stock tickers, limit to 6
    unique_tickers = db.session.query(EnrichedStockData.symbol).distinct().limit(6).all()
    unique_tickers = [ticker[0] for ticker in unique_tickers]  # Extracting the symbols

    # Fetch stock data for the last 30 days for these tickers
    today = db.session.query(db.func.max(EnrichedStockData.date)).scalar()
    thirty_days_ago = today - timedelta(days=30)

    historical_data_all = (
        EnrichedStockData.query.filter(
            EnrichedStockData.symbol.in_(unique_tickers),
            EnrichedStockData.date >= thirty_days_ago
        )
        .order_by(EnrichedStockData.date)
        .all()
    )

    # Organize the data by symbol
    historical_data_by_symbol = {}
    for data in historical_data_all:
        if data.symbol not in historical_data_by_symbol:
            historical_data_by_symbol[data.symbol] = []
        historical_data_by_symbol[data.symbol].append({
            "date": data.date.isoformat(),
            "open": data.open,
            "high": data.high,
            "low": data.low,
            "close": data.close
        })

    stocks = []
    for symbol in unique_tickers:
        # Get the most recent stock data for this symbol (latest close price)
        latest_data = (
            EnrichedStockData.query.filter_by(symbol=symbol)
            .order_by(EnrichedStockData.date.desc())
            .first()
        )

        if latest_data:
            # Use the cached historical data
            historical_data = historical_data_by_symbol.get(symbol, [])

            # Add a simple prediction logic (modify as needed)
            prediction = latest_data.close * (1 + (0.2 * (latest_data.high - latest_data.low) / latest_data.close))

            stocks.append({
                "symbol": symbol,
                "current_price": latest_data.close,
                "prediction": prediction,
                "historical_data": historical_data,
            })

    return render_template("index.html", stocks=stocks)


@bp.route("/about")
def about():
    return render_template("about.html")