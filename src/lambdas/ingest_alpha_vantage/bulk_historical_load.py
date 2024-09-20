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
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
ALPHA_VANTAGE_API_KEY = os.environ['ALPHA_VANTAGE_API_KEY']
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

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


def fetch_historical_data(symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
    all_data = []
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        data = data.reset_index()
        data['symbol'] = symbol
        # Rename columns to lowercase
        data.columns = data.columns.str.lower()
        all_data.append(data)

    return pd.concat(all_data, ignore_index=True)


def get_alpha_vantage_data(symbol: str, function: str, additional_params: dict = None) -> dict:
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    if additional_params:
        params.update(additional_params)

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


def get_options_data(symbol: str, date: str) -> pd.DataFrame:
    current_date = datetime.now().date()
    requested_date = datetime.strptime(date, '%Y-%m-%d').date()

    if requested_date > current_date:
        logger.warning(f"Requested future date {date} for {symbol}. Skipping options data.")
        return pd.DataFrame()

    additional_params = {'date': date}
    data = get_alpha_vantage_data(symbol, "HISTORICAL_OPTIONS", additional_params)

    if not data or 'data' not in data or not data['data']:
        logger.warning(f"Options data not available for {symbol} on date {date}")
        return pd.DataFrame()

    options_data = data['data']

    # Separate calls and puts
    calls = [option for option in options_data if option['type'] == 'call']
    puts = [option for option in options_data if option['type'] == 'put']

    # Calculate volumes
    call_volume = sum(float(option.get('volume', 0)) for option in calls)
    put_volume = sum(float(option.get('volume', 0)) for option in puts)

    # Handle the division by zero if call_volume is 0
    put_call_ratio = put_volume / call_volume if call_volume > 0 else None

    # Calculate implied volatilities
    implied_volatilities = [
        float(option['implied_volatility']) for option in options_data
        if option.get('implied_volatility') and option['implied_volatility'] != 'None'
    ]
    avg_implied_volatility = sum(implied_volatilities) / len(implied_volatilities) if implied_volatilities else None

    # Return DataFrame with metrics
    return pd.DataFrame({
        'date': [requested_date],
        'put_call_ratio': [put_call_ratio],
        'implied_volatility': [avg_implied_volatility]
    })


def get_technical_indicators(symbol: str, date: str) -> Tuple[float, float, float, float]:
    rsi_params = {
        "function": "RSI",
        "symbol": symbol,
        "interval": "daily",
        "time_period": 14,
        "series_type": "close"
    }
    rsi_data = get_alpha_vantage_data(symbol, "RSI", additional_params=rsi_params)

    macd_params = {
        "function": "MACD",
        "symbol": symbol,
        "interval": "daily",
        "series_type": "close"
    }
    macd_data = get_alpha_vantage_data(symbol, "MACD", additional_params=macd_params)

    rsi = macd = macd_signal = macd_hist = None

    if 'Technical Analysis: RSI' in rsi_data:
        rsi_values = rsi_data['Technical Analysis: RSI']
        if date in rsi_values:
            rsi = float(rsi_values[date]['RSI'])
        elif rsi_values:
            closest_date = max(d for d in rsi_values.keys() if d <= date)
            rsi = float(rsi_values[closest_date]['RSI'])

    if 'Technical Analysis: MACD' in macd_data:
        macd_values = macd_data['Technical Analysis: MACD']
        if date in macd_values:
            macd = float(macd_values[date]['MACD'])
            macd_signal = float(macd_values[date]['MACD_Signal'])
            macd_hist = float(macd_values[date]['MACD_Hist'])
        elif macd_values:
            closest_date = max(d for d in macd_values.keys() if d <= date)
            macd = float(macd_values[closest_date]['MACD'])
            macd_signal = float(macd_values[closest_date]['MACD_Signal'])
            macd_hist = float(macd_values[closest_date]['MACD_Hist'])

    return rsi, macd, macd_signal, macd_hist


def enrich_data_with_alpha_vantage(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    current_date = datetime.now().date()

    for symbol in symbols:
        symbol_data = df[df['symbol'] == symbol]

        # Get the most recent options data (up to 7 days in the past)
        options_data = pd.DataFrame()
        for i in range(7):
            date_to_check = (current_date - timedelta(days=i)).strftime('%Y-%m-%d')
            options_data = get_options_data(symbol, date_to_check)
            if not options_data.empty:
                break

        if options_data.empty:
            logger.warning(f"No recent options data available for {symbol}")

        for idx, row in symbol_data.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')

            rsi, macd, macd_signal, macd_hist = get_technical_indicators(symbol, date_str)

            df.at[idx, 'rsi'] = rsi
            df.at[idx, 'macd'] = macd
            df.at[idx, 'macd_signal'] = macd_signal
            df.at[idx, 'macd_hist'] = macd_hist

            if not options_data.empty:
                df.at[idx, 'put_call_ratio'] = options_data['put_call_ratio'].iloc[0]
                df.at[idx, 'implied_volatility'] = options_data['implied_volatility'].iloc[0]
            else:
                df.at[idx, 'put_call_ratio'] = None
                df.at[idx, 'implied_volatility'] = None

    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    price_cols = ['open', 'high', 'low', 'close']
    existing_price_cols = [col for col in price_cols if col in df.columns]

    if existing_price_cols:
        if (df[existing_price_cols] <= 0).any().any():
            logger.warning("Non-positive prices detected, cleaning...")
            df = df[~(df[existing_price_cols] <= 0).any(axis=1)]
    else:
        logger.warning("No price columns found in the DataFrame")

    return df


def bulk_upsert(session, df: pd.DataFrame):
    # Get the column names from the StockPrice class
    expected_columns = [column.name for column in StockPrice.__table__.columns]

    # Drop any columns that are not part of the expected columns
    df = df[[col for col in df.columns if col in expected_columns]]

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
    total_days = 365 * 3  # 3 years
    batch_size = 90  # days

    end_date = datetime.now()
    start_date = end_date - timedelta(days=total_days)

    current_date = start_date
    while current_date <= end_date:
        batch_end = min(current_date + timedelta(days=batch_size), end_date)
        logger.info(
            f"Fetching data for period: {current_date.strftime('%Y-%m-%d')} - {batch_end.strftime('%Y-%m-%d')}")

        try:
            df = fetch_historical_data(symbols, current_date, batch_end)
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

            logger.info("Data ingestion complete for current batch")
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise  # Re-raise the exception to trigger Lambda retry behavior

        current_date = batch_end + timedelta(days=1)

    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Data ingestion complete for all periods'})
    }


if __name__ == "__main__":
    lambda_handler(None, None)