import requests
from app import db
from config import Config

def get_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={Config.ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    data = response.json()

    if 'Global Quote' in data:
        return float(data['Global Quote']['05. price'])
    else:
        print(f"Error fetching data for {symbol}:", data)
        return None

def update_stock_data(stock):
    current_price = get_stock_data(stock.symbol)
    if current_price:
        stock.current_price = current_price
        db.session.commit()