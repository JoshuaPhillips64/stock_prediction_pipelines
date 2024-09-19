import os
import json
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Table, MetaData, Column, String, Date, Float, Integer, select, insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import yfinance as yf
from fredapi import Fred
from functools import lru_cache
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
ALPHA_VANTAGE_API_KEY = os.environ['ALPHA_VANTAGE_API_KEY']
NEWSAPI_API_KEY = os.environ['NEWSAPI_API_KEY']
FRED_API_KEY = os.environ['FRED_API_KEY']

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
metadata = MetaData()

# Define tables
stock_prices = Table('stock_prices', metadata, autoload_with=engine)
enriched_stock_data = Table('enriched_stock_data', metadata,
                            Column('symbol', String, primary_key=True),
                            Column('date', Date, primary_key=True),
                            Column('open', Float),
                            Column('high', Float),
                            Column('low', Float),
                            Column('close', Float),
                            Column('volume', Float),
                            Column('upper_band', Float),
                            Column('lower_band', Float),
                            Column('adx', Float),
                            Column('sector_performance', Float),
                            Column('sp500_return', Float),
                            Column('nasdaq_return', Float),
                            Column('sentiment_score', Float),
                            Column('gdp_growth', Float),
                            Column('inflation_rate', Float),
                            Column('unemployment_rate', Float),
                            Column('put_call_ratio', Float),
                            Column('implied_volatility', Float),
                            Column('rsi', Float),
                            Column('macd', Float),
                            Column('macd_signal', Float),
                            Column('macd_hist', Float)
                            )

# Create the new table if it doesn't exist
metadata.create_all(engine)

# Setup retry strategy for API calls
retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],
    backoff_factor=1
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)


@lru_cache(maxsize=1)
def get_symbols():
    """Fetch all unique symbols from the stock_prices table."""
    with engine.connect() as conn:
        distinct_symbols = conn.execute(select(stock_prices.c.symbol).distinct()).fetchall()
    return [symbol[0] for symbol in distinct_symbols]


def fetch_stock_data(symbols, start_date, end_date):
    """Fetch stock data for given symbols and date range."""
    query = select(stock_prices).where(
        (stock_prices.c.symbol.in_(symbols)) &
        (stock_prices.c.date >= start_date) &
        (stock_prices.c.date <= end_date)
    )
    with engine.connect() as conn:
        result = conn.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


def calculate_technical_indicators(df):
    """Calculate Bollinger Bands and ADX for the given dataframe."""
    # Bollinger Bands
    df['middle_band'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=20).mean())
    df['std_dev'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=20).std())
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)

    # ADX
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['plus_dm'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
                             df['high'] - df['high'].shift(), 0)
    df['minus_dm'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
                              df['low'].shift() - df['low'], 0)
    df['tr'] = df.groupby('symbol')[true_range].transform(lambda x: x.rolling(window=14).sum())
    df['plus_di'] = 100 * df.groupby('symbol')['plus_dm'].transform(lambda x: x.rolling(window=14).sum()) / df['tr']
    df['minus_di'] = 100 * df.groupby('symbol')['minus_dm'].transform(lambda x: x.rolling(window=14).sum()) / df['tr']
    df['dx'] = 100 * np.abs((df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
    df['adx'] = df.groupby('symbol')['dx'].transform(lambda x: x.rolling(window=14).mean())

    return df


@lru_cache(maxsize=1000)
def get_sector_performance(symbol, date):
    """Get sector performance for a given symbol and date."""
    stock = yf.Ticker(symbol)
    sector = stock.info.get('sector', None)
    if sector:
        sector_etf = {
            'Information Technology': 'XLK',
            'Health Care': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Communication Services': 'XLC',
            'Industrials': 'XLI',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB'
        }.get(sector)

        if sector_etf:
            etf_data = yf.download(sector_etf, start=date, end=date + timedelta(days=1))
            if not etf_data.empty:
                return etf_data['Close'].pct_change().iloc[-1] * 100
    return None


@lru_cache(maxsize=1000)
def get_market_index_performance(date):
    """Get market index performance for a given date."""
    indices = yf.download(['^GSPC', '^IXIC'], start=date, end=date + timedelta(days=1))
    return {
        'sp500_return': indices['Close']['^GSPC'].pct_change().iloc[-1] * 100 if not indices.empty else None,
        'nasdaq_return': indices['Close']['^IXIC'].pct_change().iloc[-1] * 100 if not indices.empty else None
    }


@lru_cache(maxsize=1000)
def get_sentiment_score(symbol, date):
    """Get sentiment score for a given symbol and date."""
    end_date = date.strftime('%Y-%m-%d')
    start_date = (date - timedelta(days=7)).strftime('%Y-%m-%d')  # Look back 7 days for news
    url = f"https://newsapi.org/v2/everything?q={symbol}&from={start_date}&to={end_date}&apiKey={NEWSAPI_API_KEY}&language=en&sortBy=publishedAt&pageSize=100"
    response = http.get(url)
    if response.status_code == 200:
        news = response.json()
        if news['totalResults'] > 0:
            titles = [article['title'] for article in news['articles']]
            return analyze_sentiment(titles)
    return None


def analyze_sentiment(texts):
    """Analyze sentiment of given texts."""
    positive_words = set(['upgrade', 'buy', 'bullish', 'outperform', 'strong', 'positive'])
    negative_words = set(['downgrade', 'sell', 'bearish', 'underperform', 'weak', 'negative'])

    sentiment_score = sum(len(set(text.lower().split()) & positive_words) -
                          len(set(text.lower().split()) & negative_words)
                          for text in texts)

    return sentiment_score / len(texts) if texts else 0


@lru_cache(maxsize=100)
def get_economic_indicators(year, month):
    """Get economic indicators for a given year and month using FRED API."""
    fred = Fred(api_key=FRED_API_KEY)

    date = datetime(year, month, 1)
    end_date = date.replace(day=28) + timedelta(days=4)  # This will get us the last day of the month
    end_date = end_date - timedelta(days=end_date.day)

    # GDP growth (quarterly)
    gdp_growth = fred.get_series('GDP', observation_start=date - timedelta(days=90), observation_end=end_date)
    gdp_growth = gdp_growth.pct_change().iloc[-1] * 100 if not gdp_growth.empty else None

    # Inflation rate (monthly)
    inflation_rate = fred.get_series('CPIAUCSL', observation_start=date, observation_end=end_date)
    inflation_rate = inflation_rate.pct_change().iloc[-1] * 100 if not inflation_rate.empty else None

    # Unemployment rate (monthly)
    unemployment_rate = fred.get_series('UNRATE', observation_start=date, observation_end=end_date)
    unemployment_rate = unemployment_rate.iloc[-1] if not unemployment_rate.empty else None

    return {
        'gdp_growth': gdp_growth,
        'inflation_rate': inflation_rate,
        'unemployment_rate': unemployment_rate
    }


def enrich_stock_data(df):
    """Enrich stock data with additional features."""
    df = calculate_technical_indicators(df)

    unique_dates = df['date'].unique()

    # Fetch market index performance and economic indicators for all unique dates
    with ThreadPoolExecutor(max_workers=10) as executor:
        market_performance_futures = {executor.submit(get_market_index_performance, date): date for date in
                                      unique_dates}
        economic_indicators_futures = {
            executor.submit(get_economic_indicators, date.year, date.month): (date.year, date.month) for date in
            unique_dates}

        market_performance = {date: future.result() for future, date in market_performance_futures.items()}
        economic_indicators = {(year, month): future.result() for future, (year, month) in
                               economic_indicators_futures.items()}

    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]

        # Fetch sector performance and sentiment scores for all dates for this symbol
        with ThreadPoolExecutor(max_workers=10) as executor:
            sector_performance_futures = {executor.submit(get_sector_performance, symbol, date): date for date in
                                          symbol_df['date']}
            sentiment_score_futures = {executor.submit(get_sentiment_score, symbol, date): date for date in
                                       symbol_df['date']}

            sector_performance = {date: future.result() for future, date in sector_performance_futures.items()}
            sentiment_scores = {date: future.result() for future, date in sentiment_score_futures.items()}

        # Update the dataframe with the fetched data
        for index, row in symbol_df.iterrows():
            date = row['date']
            df.at[index, 'sector_performance'] = sector_performance[date]
            df.at[index, 'sp500_return'] = market_performance[date]['sp500_return']
            df.at[index, 'nasdaq_return'] = market_performance[date]['nasdaq_return']
            df.at[index, 'sentiment_score'] = sentiment_scores[date]

            econ_indicators = economic_indicators[(date.year, date.month)]
            df.at[index, 'gdp_growth'] = econ_indicators['gdp_growth']
            df.at[index, 'inflation_rate'] = econ_indicators['inflation_rate']
            df.at[index, 'unemployment_rate'] = econ_indicators['unemployment_rate']

    return df


def update_enriched_data(df):
    """Update the enriched_stock_data table with new data using upsert."""
    insert_stmt = pg_insert(enriched_stock_data).values(df.to_dict('records'))
    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=['symbol', 'date'],
        set_={c.name: c for c in insert_stmt.excluded if c.name not in ['symbol', 'date']}
    )

    with engine.begin() as conn:
        conn.execute(upsert_stmt)


def lambda_handler(event, context):
    """Main Lambda function handler."""
    try:
        symbols = get_symbols()
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=int(event.get('days_to_process', 30)))

        # Process symbols in batches
        batch_size = 50  # Increased batch size
        for i in range(0, len(symbols), batch_size):
            symbol_batch = symbols[i:i + batch_size]
            logger.info(f"Processing symbols: {symbol_batch}")

            df = fetch_stock_data(symbol_batch, start_date, end_date)
            if not df.empty:
                enriched_df = enrich_stock_data(df)
                update_enriched_data(enriched_df)
            else:
                logger.warning(f"No data found for symbols: {symbol_batch}")

        logger.info("Data enrichment complete for all symbols")
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Data enrichment complete'})
        }
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Database error occurred'})
        }
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Error processing data'})
        }


if __name__ == "__main__":
    lambda_handler({}, None)