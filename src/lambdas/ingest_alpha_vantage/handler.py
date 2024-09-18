import os
import requests
import json
import logging
from common_utils.utils import get_db_connection

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ALPHA_VANTAGE_API_KEY = os.environ['ALPHA_VANTAGE_API_KEY']
STOCK_SYMBOLS = os.environ.get('STOCK_SYMBOLS', 'AAPL,GOOGL,MSFT').split(',')

def lambda_handler(event, context):
    conn = get_db_connection()
    cursor = conn.cursor()

    for symbol in STOCK_SYMBOLS:
        logger.info(f"Fetching data for {symbol}")
        response = requests.get(
            f'https://www.alphavantage.co/query',
            params={
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': ALPHA_VANTAGE_API_KEY
            }
        )
        data = response.json()
        time_series = data.get('Time Series (Daily)', {})

        for date, metrics in time_series.items():
            cursor.execute(
                """
          INSERT INTO
          stock_prices(symbol, date, open, high, low,
                       close, volume)
              VALUES( % s, % s, % s, % s, % s, % s, % s)
            ON
            CONFLICT(symbol, date)
            DO
            NOTHING;
            """,
    (   symbol,
        date,
        metrics['1. open'],
        metrics['2. high'],
        metrics['3. low'],
        metrics['4. close'],
        metrics['5. volume'])
    )
    conn.commit()

    cursor.close()
    conn.close()
    logger.info("Data ingestion complete.")

    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Data ingestion complete'})
    }