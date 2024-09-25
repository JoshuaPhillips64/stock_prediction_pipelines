from app import db
from sqlalchemy.dialects.postgresql import JSON

class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    current_price = db.Column(db.Float)
    prediction = db.Column(db.Float)
    historical_data = db.Column(JSON)

    def __repr__(self):
        return f'<Stock {self.symbol}>'

class ModelStatistics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    metric = db.Column(db.String(50), nullable=False)
    value = db.Column(db.Float, nullable=False)
    stock = db.Column(db.String(10))

    def __repr__(self):
        return f'<ModelStatistics {self.metric} for {self.stock}>'