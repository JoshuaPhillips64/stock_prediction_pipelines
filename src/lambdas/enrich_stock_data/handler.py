import os
import json
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from sqlalchemy import (
    create_engine,
    Table,
    MetaData,
    Column,
    String,
    Date,
    Float,
    Numeric,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert, JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
metadata = MetaData()

# Define tables
stock_prices = Table("stock_prices", metadata, autoload_with=engine)
company_overview = Table(
    "company_overview",
    metadata,
    Column("symbol", String, primary_key=True),
    Column("last_updated", Date, primary_key=True),
    Column("data", JSON),
)
enriched_stock_data = Table(
    "enriched_stock_data",
    metadata,
    Column("symbol", String, primary_key=True),
    Column("date", Date, primary_key=True),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("volume", Float),
    Column("sector_performance", Float),
    Column("sp500_return", Float),
    Column("nasdaq_return", Float),
    Column("sentiment_score", Float),
    Column("gdp_growth", Float),
    Column("inflation_rate", Float),
    Column("unemployment_rate", Float),
    Column("market_capitalization", Numeric),
    Column("pe_ratio", Float),
    Column("dividend_yield", Float),
    Column("beta", Float),
)

# Create the tables if they don't exist
metadata.create_all(engine)

# Alpha Vantage API base URL
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


def fetch_data_from_alpha_vantage(function, symbol=None, **kwargs):
    """
    Fetches data from Alpha Vantage API.

    Args:
        function (str): The API function to call.
        symbol (str, optional): The stock symbol. Required for some functions.
        **kwargs: Additional parameters for the API call.

    Returns:
        dict: The JSON response from the API.
    """
    params = {"function": function, "apikey": ALPHA_VANTAGE_API_KEY, **kwargs}
    if symbol:
        params["symbol"] = symbol
    response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.json()


def get_symbols():
    """Fetch all unique symbols from the stock_prices table."""
    with engine.connect() as conn:
        distinct_symbols = conn.execute(
            select(stock_prices.c.symbol).distinct()
        ).fetchall()
    return [symbol[0] for symbol in distinct_symbols]


def fetch_stock_data(symbols, start_date, end_date):
    """Fetch stock data for given symbols and date range."""
    query = select(stock_prices).where(
        (stock_prices.c.symbol.in_(symbols))
        & (stock_prices.c.date >= start_date)
        & (stock_prices.c.date <= end_date)
    )
    with engine.connect() as conn:
        result = conn.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


def get_company_overview_data(symbol):
    """
    Fetches and caches company overview data from Alpha Vantage API.
    """
    try:
        with Session() as session:
            # Check if data for today already exists in the database
            existing_data = session.execute(
                select(company_overview.c.data).where(
                    (company_overview.c.symbol == symbol)
                    & (company_overview.c.last_updated == datetime.now().date())
                )
            ).fetchone()

            if existing_data:
                logger.info(
                    f"Using cached company overview data for {symbol}"
                )
                return existing_data[0]

            # Fetch data from API if not cached
            logger.info(
                f"Fetching company overview data for {symbol} from API"
            )
            data = fetch_data_from_alpha_vantage("OVERVIEW", symbol)
            # Store the data in the database
            stmt = pg_insert(company_overview).values(
                symbol=symbol, last_updated=datetime.now().date(), data=data
            )
            session.execute(stmt)
            session.commit()
            return data
    except Exception as e:
        logger.error(
            f"Error fetching or caching company overview data for {symbol}: {e}"
        )
        return {}


def get_sector_performance(symbol, date, company_overview_data):
    """Get sector performance for a given symbol and date."""
    sector = company_overview_data.get("Sector")
    if sector:
        sector_etf_mapping = {
            "Information Technology": "XLK",
            "Health Care": "XLV",
            "Financials": "XLF",
            "Consumer Discretionary": "XLY",
            "Communication Services": "XLC",
            "Industrials": "XLI",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Materials": "XLB",
        }
        sector_etf = sector_etf_mapping.get(sector)
        if sector_etf:
            try:
                data = fetch_data_from_alpha_vantage(
                    "TIME_SERIES_DAILY",
                    symbol=sector_etf,
                    outputsize="compact",
                )
                daily_data = data["Time Series (Daily)"]
                date_str = date.strftime("%Y-%m-%d")
                previous_date_str = (date - timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                )
                if (
                    date_str in daily_data
                    and previous_date_str in daily_data
                ):
                    close_price = float(daily_data[date_str]["4. close"])
                    previous_close_price = float(
                        daily_data[previous_date_str]["4. close"]
                    )
                    return (
                        (close_price - previous_close_price)
                        / previous_close_price
                    ) * 100
            except Exception as e:
                logger.error(
                    f"Error fetching sector performance for {symbol} on {date}: {e}"
                )
    return None


def get_market_index_performance(date):
    """Get market index performance for a given date."""
    indices = ["^GSPC", "^IXIC"]  # S&P 500 and Nasdaq
    results = {}
    for index in indices:
        try:
            data = fetch_data_from_alpha_vantage(
                "TIME_SERIES_DAILY", symbol=index, outputsize="compact"
            )
            daily_data = data["Time Series (Daily)"]
            date_str = date.strftime("%Y-%m-%d")
            previous_date_str = (date - timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
            if date_str in daily_data and previous_date_str in daily_data:
                close_price = float(daily_data[date_str]["4. close"])
                previous_close_price = float(
                    daily_data[previous_date_str]["4. close"]
                )
                results[
                    index
                ] = ((close_price - previous_close_price) / previous_close_price) * 100
            else:
                results[index] = None
        except Exception as e:
            logger.error(
                f"Error fetching market index performance for {index} on {date}: {e}"
            )
            results[index] = None

    return {
        "sp500_return": results.get("^GSPC"),
        "nasdaq_return": results.get("^IXIC"),
    }


def get_sentiment_score(symbol, date):
    """
    Get sentiment score from recent news for a given symbol and date.
    """
    end_date = date.strftime("%Y%m%d")
    start_date = (date - timedelta(days=7)).strftime("%Y%m%d")
    try:
        data = fetch_data_from_alpha_vantage(
            "NEWS_SENTIMENT",
            tickers=symbol,
            time_from=start_date,
            time_to=end_date,
            limit=100,
        )
        if data.get("items", 0) > 0:
            sentiment_sum = sum(
                float(item["ticker_sentiment_score"])
                for item in data["feed"]
                if "ticker_sentiment_score" in item
            )
            return sentiment_sum / data["items"]
    except Exception as e:
        logger.error(
            f"Error fetching sentiment score for {symbol} on {date}: {e}"
        )
    return None


def get_economic_indicators(date):
    """Get economic indicators for a given date."""
    results = {}

    try:
        gdp_data = fetch_data_from_alpha_vantage("REAL_GDP", interval="quarterly")
        gdp_date_str = f"{date.year}-Q{(date.month - 1) // 3 + 1}"
        if gdp_date_str in gdp_data["data"]:
            results["gdp_growth"] = float(gdp_data["data"][gdp_date_str]["value"])
        else:
            results["gdp_growth"] = None
    except Exception as e:
        logger.error(f"Error fetching GDP growth: {e}")
        results["gdp_growth"] = None

    try:
        inflation_data = fetch_data_from_alpha_vantage("INFLATION")
        inflation_year_str = str(date.year)
        if inflation_year_str in inflation_data["data"]:
            results["inflation_rate"] = float(inflation_data["data"][inflation_year_str]["value"])
        else:
            results["inflation_rate"] = None
    except Exception as e:
        logger.error(f"Error fetching inflation rate: {e}")
        results["inflation_rate"] = None

    try:
        unemployment_data = fetch_data_from_alpha_vantage("UNEMPLOYMENT")
        unemployment_date_str = date.strftime("%Y-%m")
        if unemployment_date_str in unemployment_data["data"]:
            results["unemployment_rate"] = float(unemployment_data["data"][unemployment_date_str]["value"])
        else:
            results["unemployment_rate"] = None
    except Exception as e:
        logger.error(f"Error fetching unemployment rate: {e}")
        results["unemployment_rate"] = None
    return results


def enrich_stock_data(df):
    """Enrich stock data with additional features."""
    unique_dates = df["date"].unique()

    with ThreadPoolExecutor(max_workers=10) as executor:
        # Fetch market index performance and economic indicators for all unique dates
        market_performance_futures = {
            executor.submit(get_market_index_performance, date): date
            for date in unique_dates
        }
        economic_indicators_futures = {
            executor.submit(get_economic_indicators, date): date
            for date in unique_dates
        }

        market_performance_results = {}
        for future in as_completed(market_performance_futures):
            date = market_performance_futures[future]
            try:
                market_performance_results[date] = future.result()
            except Exception as e:
                logger.error(
                    f"Error fetching market performance for {date}: {e}"
                )
                market_performance_results[
                    date
                ] = {"sp500_return": None, "nasdaq_return": None}

        economic_indicators_results = {}
        for future in as_completed(economic_indicators_futures):
            date = economic_indicators_futures[future]
            try:
                economic_indicators_results[date] = future.result()
            except Exception as e:
                logger.error(
                    f"Error fetching economic indicators for {date}: {e}"
                )
                economic_indicators_results[date] = {
                    "gdp_growth": None,
                    "inflation_rate": None,
                    "unemployment_rate": None,
                }

    for symbol in df["symbol"].unique():
        # Fetch company overview data for current symbol
        company_overview_data = get_company_overview_data(symbol)
        symbol_df = df[df["symbol"] == symbol]
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Fetch sector performance and sentiment scores for all dates for this symbol
            sector_performance_futures = {
                executor.submit(
                    get_sector_performance, symbol, date, company_overview_data
                ): date
                for date in symbol_df["date"]
            }
            sentiment_score_futures = {
                executor.submit(get_sentiment_score, symbol, date): date
                for date in symbol_df["date"]
            }
            sector_performance_results = {}
            for future in as_completed(sector_performance_futures):
                date = sector_performance_futures[future]
                try:
                    sector_performance_results[date] = future.result()
                except Exception as e:
                    logger.error(
                        f"Error fetching sector performance for {symbol} on {date}: {e}"
                    )
                    sector_performance_results[date] = None

            sentiment_scores_results = {}
            for future in as_completed(sentiment_score_futures):
                date = sentiment_score_futures[future]
                try:
                    sentiment_scores_results[date] = future.result()
                except Exception as e:
                    logger.error(
                        f"Error fetching sentiment score for {symbol} on {date}: {e}"
                    )
                    sentiment_scores_results[date] = None

        # Update the dataframe with the fetched data
        for index, row in symbol_df.iterrows():
            date = row["date"]
            df.at[index, "sector_performance"] = (
                sector_performance_results.get(date)
            )
            market_performance = market_performance_results.get(
                date, {"sp500_return": None, "nasdaq_return": None}
            )
            # Check for "None" before converting to float
            df.at[index, "sp500_return"] = (
                float(market_performance["sp500_return"])
                if market_performance["sp500_return"] != "None"
                else None
            )
            df.at[index, "nasdaq_return"] = (
                float(market_performance["nasdaq_return"])
                if market_performance["nasdaq_return"] != "None"
                else None
            )
            df.at[index, "sentiment_score"] = sentiment_scores_results.get(
                date
            )
            econ_indicators = economic_indicators_results.get(
                date,
                {
                    "gdp_growth": None,
                    "inflation_rate": None,
                    "unemployment_rate": None,
                },
            )
            # Check for "None" before converting to float
            df.at[index, "gdp_growth"] = (
                float(econ_indicators["gdp_growth"])
                if econ_indicators["gdp_growth"] != "None"
                else None
            )
            df.at[index, "inflation_rate"] = (
                float(econ_indicators["inflation_rate"])
                if econ_indicators["inflation_rate"] != "None"
                else None
            )
            df.at[index, "unemployment_rate"] = (
                float(econ_indicators["unemployment_rate"])
                if econ_indicators["unemployment_rate"] != "None"
                else None
            )
            # Check for "None" before converting to float
            df.at[index, "market_capitalization"] = (
                float(company_overview_data.get("MarketCapitalization", 0))
                if company_overview_data.get("MarketCapitalization", "None") != "None"
                else None
            )
            df.at[index, "pe_ratio"] = (
                float(company_overview_data.get("PERatio", 0))
                if company_overview_data.get("PERatio", "None") != "None"
                else None
            )
            df.at[index, "dividend_yield"] = (
                float(company_overview_data.get("DividendYield", 0))
                if company_overview_data.get("DividendYield", "None") != "None"
                else None
            )
            df.at[index, "beta"] = (
                float(company_overview_data.get("Beta", 0))
                if company_overview_data.get("Beta", "None") != "None"
                else None
            )

    return df


def update_enriched_data(df):
    """Update the enriched_stock_data table with new data using upsert."""
    # Convert datetime.date to datetime.datetime for SQLAlchemy
    df["date"] = pd.to_datetime(df["date"])
    # Build insert statement
    insert_stmt = pg_insert(enriched_stock_data).values(
        df.to_dict(orient="records")
    )
    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=["symbol", "date"],
        set_={
            c.name: c
            for c in insert_stmt.excluded
            if c.name not in ["symbol", "date"]
        },
    )

    with engine.begin() as conn:
        conn.execute(upsert_stmt)


def lambda_handler(event, context):
    """Main Lambda function handler."""

    symbols = get_symbols()
    end_date = datetime.now().date()
    days_to_process = 5
    start_date = end_date - timedelta(days=days_to_process)

    # Process symbols in batches
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        symbol_batch = symbols[i : i + batch_size]
        logger.info(f"Processing symbols: {symbol_batch}")

        df = fetch_stock_data(symbol_batch, start_date, end_date)
        if not df.empty:
            enriched_df = enrich_stock_data(df)
            update_enriched_data(enriched_df)
        else:
            logger.warning(f"No data found for symbols: {symbol_batch}")

    logger.info("Data enrichment complete for all symbols")
    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Data enrichment complete"}),
    }


if __name__ == "__main__":
    lambda_handler({}, None)