import os

# Environment Variables
LAMBDA_FUNCTION_NAME = 'ingest_stock_data'
POSTGRES_CONN_ID = 'postgres_default'

# Top 50 stock tickers
TOP_50_TICKERS = [
    'PG', 'BAC', 'MA', 'DIS', 'CSCO',
    'KO', 'ADBE', 'NFLX', 'PEP', 'CRM', 'ABT', 'JNJ',
    'MCD', 'COST', 'WMT', 'TMO', 'ACN', 'NKE', 'DHR', 'MDT', 'LLY',
    'IBM', 'AMGN', 'TXN', 'HON', 'LIN', 'LOW', 'XOM', 'GIS', 'CL', 'MMM', 'SPY'
]