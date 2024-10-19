from sqlalchemy import String, Date, Float, Numeric, MetaData
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime, date

db = SQLAlchemy()

# Define the metadata object
metadata = db.MetaData()

# Define company_overview as a class-based model
class CompanyOverview(db.Model):
    __tablename__ = 'company_overview'

    symbol = db.Column(db.String, primary_key=True)
    last_updated = db.Column(db.Date, primary_key=True)
    data = db.Column(JSON)

    def __repr__(self):
        return f"<CompanyOverview {self.symbol}, Last Updated: {self.last_updated}>"

class StockPrice(db.Model):
    __tablename__ = 'stock_prices'

    symbol = db.Column(db.String, primary_key=True)
    date = db.Column(db.Date, primary_key=True)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.Float)
    put_call_ratio = db.Column(db.Float)
    implied_volatility = db.Column(db.Float)
    rsi = db.Column(db.Float)
    macd = db.Column(db.Float)
    macd_signal = db.Column(db.Float)
    macd_hist = db.Column(db.Float)

    def __repr__(self):
        return f"<StockPrice {self.symbol} {self.date}>"

class EnrichedStockData(db.Model):
    __tablename__ = 'enriched_stock_data'

    symbol = db.Column(db.String, primary_key=True)
    date = db.Column(db.Date, primary_key=True)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.Float)
    sector_performance = db.Column(db.Float)
    sp500_return = db.Column(db.Float)
    nasdaq_return = db.Column(db.Float)
    sentiment_score = db.Column(db.Float)
    gdp_growth = db.Column(db.Float)
    inflation_rate = db.Column(db.Float)
    unemployment_rate = db.Column(db.Float)
    market_capitalization = db.Column(db.Numeric)
    pe_ratio = db.Column(db.Float)
    dividend_yield = db.Column(db.Float)
    beta = db.Column(db.Float)

    # Restoring removed columns
    put_call_ratio = db.Column(db.Float)
    macd_hist = db.Column(db.Float)
    adx = db.Column(db.Float)
    implied_volatility = db.Column(db.Float)
    macd = db.Column(db.Float)
    rsi = db.Column(db.Float)
    upper_band = db.Column(db.Float)
    lower_band = db.Column(db.Float)
    macd_signal = db.Column(db.Float)

    def __repr__(self):
        return f"<EnrichedStockData {self.symbol} {self.date}>"

def create_ai_stock_predictions_model():
    """Creates the AiStockPredictions model dynamically after the app is initialized."""
    class AiStockPredictions(db.Model):  # Use the global db object
        __table__ = db.Table('ai_stock_predictions', metadata, autoload=True, autoload_with=db.engine)

        def to_dict(self):
            return {c.name: getattr(self, c.name) for c in self.__table__.columns}

        def __repr__(self):
            return f"<AiStockPredictions {self.symbol} {self.date}>"
    return AiStockPredictions

class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_key = db.Column(db.String(128), unique=True, nullable=False)
    model_type = db.Column(db.String(64), nullable=False)
    stock_symbol = db.Column(db.String(16), nullable=False)
    input_date = db.Column(db.Date, nullable=False)
    hyperparameter_tuning = db.Column(db.String(16), nullable=False)
    feature_set = db.Column(db.String(64), nullable=False)
    lookback_period = db.Column(db.Integer, nullable=False)
    prediction_horizon = db.Column(db.Integer, nullable=False)
    prediction_data = db.Column(db.Text, nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<PredictionResult {self.model_key}>'