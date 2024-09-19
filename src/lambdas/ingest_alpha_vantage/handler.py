import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Tuple
import time
import requests
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, Column, String, Date, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import insert

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
ALPHA_VANTAGE_API_KEY = os.environ['ALPHA_VANTAGE_API_KEY']
DATABASE_URL = os.environ['DATABASE_URL']

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class StockPrice(Base):
    __tablename__ = 'stock_prices'

    symbol = Column(String, primary_key=True)
    date = Column(Date, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    put_call_ratio = Column(Float)
    implied_volatility = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)


Base.metadata.create_all(engine)


def get_symbols() -> List[str]:
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']


def fetch_historical_data(symbols: List[str], days: int) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    all_data = []
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        data = data.reset_index()
        data['symbol'] = symbol
        all_data.append(data)

    return pd.concat(all_data, ignore_index=True)


def get_alpha_vantage_data(symbol: str, function: str) -> dict:
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    for attempt in range(3):  # Retry mechanism for 3 attempts
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if not data or "Error Message" in data:
                logger.warning(f"No data for {symbol} with function {function}")
                return {}
            return data
        elif response.status_code == 429:
            logger.warning("Rate limit hit. Retrying after 15 seconds...")
            time.sleep(15)
        else:
            logger.error(f"Failed to fetch data for {symbol}. HTTP Status: {response.status_code}")
            return {}

    raise Exception(f"Unable to retrieve data for {symbol} after multiple attempts.")


def get_options_data(symbol: str) -> pd.DataFrame:
    data = get_alpha_vantage_data(symbol, "OPTION_CHAIN")
    if not data or 'options' not in data or not data['options']:
        logger.warning(f"Options data not available for {symbol}")
        return pd.DataFrame()

    calls = pd.DataFrame(data['options'][0]['calls'])
    puts = pd.DataFrame(data['options'][0]['puts'])

    total_call_volume = calls['volume'].astype(float).sum()
    total_put_volume = puts['volume'].astype(float).sum()

    put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else None
    implied_volatility = calls['impliedVolatility'].astype(float).mean()

    return pd.DataFrame({
        'date': [datetime.now()],  # Assuming we're using today's date
        'put_call_ratio': [put_call_ratio],
        'implied_volatility': [implied_volatility]
    })


# Optional: Pull put/call ratio for a strike date 30 days out (commented out for future usage)
# def get_options_data_with_strike(symbol: str, strike_days: int = 30) -> pd.DataFrame:
#     data = get_alpha_vantage_data(symbol, "OPTION_CHAIN")
#     if not data or 'options' not in data or not data['options']:
#         logger.warning(f"Options data not available for {symbol}")
#         return pd.DataFrame()

#     # Assuming the data contains expiration dates, you'd filter here based on the strike_days
#     calls = pd.DataFrame(data['options'][0]['calls'])
#     puts = pd.DataFrame(data['options'][0]['puts'])

#     # Filter calls and puts for 30-day expiration (you need premium access for such details)
#     filtered_calls = calls[calls['expiration'] == (datetime.now() + timedelta(days=strike_days))]
#     filtered_puts = puts[puts['expiration'] == (datetime.now() + timedelta(days=strike_days))]

#     total_call_volume = filtered_calls['volume'].astype(float).sum()
#     total_put_volume = filtered_puts['volume'].astype(float).sum()

#     put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else None
#     implied_volatility = filtered_calls['impliedVolatility'].astype(float).mean()

#     return pd.DataFrame({
#         'date': [datetime.now()],
#         'put_call_ratio': [put_call_ratio],
#         'implied_volatility': [implied_volatility]
#     })

def get_technical_indicators(symbol: str, date: str) -> Tuple[float, float, float, float]:
    rsi_data = get_alpha_vantage_data(symbol, "RSI")
    macd_data = get_alpha_vantage_data(symbol, "MACD")

    rsi = None
    macd = macd_signal = macd_hist = None

    if 'Technical Analysis: RSI' in rsi_data and date in rsi_data['Technical Analysis: RSI']:
        rsi = float(rsi_data['Technical Analysis: RSI'][date]['RSI'])

    if 'Technical Analysis: MACD' in macd_data and date in macd_data['Technical Analysis: MACD']:
        last_data = macd_data['Technical Analysis: MACD'][date]
        macd = float(last_data['MACD'])
        macd_signal = float(last_data['MACD_Signal'])
        macd_hist = float(last_data['MACD_Hist'])

    return rsi, macd, macd_signal, macd_hist


def enrich_data_with_alpha_vantage(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    for symbol in symbols:
        for idx, row in df[df['symbol'] == symbol].iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            rsi, macd, macd_signal, macd_hist = get_technical_indicators(symbol, date_str)

            df.at[idx, 'rsi'] = rsi
            df.at[idx, 'macd'] = macd
            df.at[idx, 'macd_signal'] = macd_signal
            df.at[idx, 'macd_hist'] = macd_hist

            # Put/Call ratio and implied volatility can be daily for the latest data
            options_data = get_options_data(symbol)
            df.at[idx, 'put_call_ratio'] = options_data['put_call_ratio'].iloc[0]
            df.at[idx, 'implied_volatility'] = options_data['implied_volatility'].iloc[0]

    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.isnull().values.any():
        logger.warning("Data contains missing values, cleaning...")
        df = df.dropna()

    price_cols = ['open', 'high', 'low', 'close']
    if (df[price_cols] <= 0).any().any():
        logger.warning("Non-positive prices detected, cleaning...")
        df = df[~(df[price_cols] <= 0).any(axis=1)]

    return df


def bulk_upsert(session, df: pd.DataFrame):
    table = StockPrice.__table__

    data = df.to_dict(orient='records')
    stmt = insert(table).values(data)
    stmt = stmt.on_conflict_do_update(
        index_elements=['symbol', 'date'],
        set_={c.key: c for c in stmt.excluded if c.key not in ['symbol', 'date']}
    )
    session.execute(stmt)
    session.commit()


def lambda_handler(event, context):
    symbols = get_symbols()
    days_to_fetch = 30

    logger.info(f"Fetching data for symbols: {symbols}")

    try:
        df = fetch_historical_data(symbols, days_to_fetch)
        df = enrich_data_with_alpha_vantage(df, symbols)
        df = validate_data(df)

        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        with Session() as session:
            bulk_upsert(session, df)

        logger.info("Data ingestion complete for all symbols")
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Data ingestion complete'})
    }