import requests
from app import db
from config import Config
from app.models import EnrichedStockData


def get_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={Config.ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    data = response.json()

    if 'Global Quote' in data:
        # Extract relevant fields for the enriched_stock_data
        return {
            'open': float(data['Global Quote']['02. open']),
            'high': float(data['Global Quote']['03. high']),
            'low': float(data['Global Quote']['04. low']),
            'close': float(data['Global Quote']['05. price']),
            'volume': float(data['Global Quote']['06. volume'])
        }
    else:
        print(f"Error fetching data for {symbol}:", data)
        return None


def update_stock_data(symbol, date):
    stock_data = get_stock_data(symbol)
    if stock_data:
        # Fetch or create a record for the specific symbol and date
        stock = EnrichedStockData.query.filter_by(symbol=symbol, date=date).first()

        if not stock:
            stock = EnrichedStockData(symbol=symbol, date=date)

        # Update stock record fields
        stock.open = stock_data['open']
        stock.high = stock_data['high']
        stock.low = stock_data['low']
        stock.close = stock_data['close']
        stock.volume = stock_data['volume']

        # Commit the updated stock data to the database
        db.session.add(stock)
        db.session.commit()