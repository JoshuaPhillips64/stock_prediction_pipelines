from datetime import date, timedelta
from flask import Blueprint, render_template
from app import db
from app.live_stock_updates import update_stock_data
from app.models import EnrichedStockData
bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    # Assume today's date for simplicity
    today = date.today()

    # Fetch all stock data for today
    stocks = []  # Use 'stocks' instead of 'stocks_data'
    db_stocks = EnrichedStockData.query.filter_by(date=today).all()
    for stock in db_stocks:
        # Update stock data
        update_stock_data(stock.symbol, today)

        # Fetch historical data (last 30 days) for the chart
        thirty_days_ago = today - timedelta(days=30)
        historical_data = (
            EnrichedStockData.query.filter(
                EnrichedStockData.symbol == stock.symbol,
                EnrichedStockData.date >= thirty_days_ago,
                EnrichedStockData.date <= today,
            )
            .order_by(EnrichedStockData.date)
            .all()
        )

        # You'll likely have your prediction logic here.
        # For now, let's simulate a simple prediction.
        prediction = stock.close * (1 + (0.2 * (stock.high - stock.low) / stock.close))

        stocks.append( # Append to 'stocks'
            {
                "symbol": stock.symbol,
                "current_price": stock.close,
                "prediction": prediction,
                "historical_data": [
                    {
                        "date": d.date.isoformat(),
                        "open": d.open,
                        "high": d.high,
                        "low": d.low,
                        "close": d.close,
                    }
                    for d in historical_data
                ],
            }
        )
    return render_template("index.html", stocks=stocks)


@bp.route("/about")
def about():
    return render_template("about.html")