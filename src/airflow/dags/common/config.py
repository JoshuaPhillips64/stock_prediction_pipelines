import os

# Environment Variables
LAMBDA_FUNCTION_NAME = 'ingest_stock_data'
POSTGRES_CONN_ID = 'postgres_default'

# Top 50 stock tickers
TOP_50_TICKERS_NEW = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'UNH',
    'HD', 'PG', 'BAC', 'MA', 'DIS', 'VZ', 'CMCSA', 'INTC', 'PFE', 'CSCO',
    'KO', 'ADBE', 'NFLX', 'PEP', 'CRM', 'ABT', 'JNJ', 'ORCL', 'ABBV', 'MRK',
    'MCD', 'COST', 'WMT', 'TMO', 'ACN', 'NKE', 'DHR', 'QCOM', 'MDT', 'LLY',
    'IBM', 'AMGN', 'TXN', 'NEE', 'AVGO', 'PM', 'UNP', 'HON', 'LIN', 'LOW', 'XOM', 'GIS', 'CL', 'MMM', 'SPY'
]

# Top 50 stock tickers
TOP_50_TICKERS = [
     'MMM'
]