import sys
from pathlib import Path
from datetime import datetime, timedelta
import requests
from typing import Dict, Any, List
from config import ALPHA_VANTAGE_API_KEY
import pandas as pd
import logging
import time
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Alpha Vantage API configuration
API_KEY = ALPHA_VANTAGE_API_KEY

# Initialize variables for rate limiting
LAST_API_CALL = 0
API_CALL_INTERVAL = 0  # Approximately one request every second (70 requests per minute)

def get_alpha_vantage_data(symbol: str, function: str, additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Makes a request to the Alpha Vantage API with rate limiting.

    Args:
        symbol (str): The stock symbol.
        function (str): The API function to call.
        additional_params (Dict[str, Any], optional): Additional parameters for the API call.

    Returns:
        Dict[str, Any]: The JSON response from the API.
    """
    global LAST_API_CALL
    now = time.time()
    elapsed = now - LAST_API_CALL
    if elapsed < API_CALL_INTERVAL:
        time.sleep(API_CALL_INTERVAL - elapsed)
    LAST_API_CALL = time.time()

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "apikey": API_KEY
    }
    if symbol:
        params["symbol"] = symbol
    if additional_params:
        params.update(additional_params)

    response = requests.get(base_url, params=params)
    data = response.json()

    # Handle API rate limit exceeded error
    if "Note" in data and ("maximum number" in data["Note"] or "API call frequency" in data["Note"]):
        logger.warning("API rate limit exceeded. Waiting for 60 seconds.")
        time.sleep(60)
        return get_alpha_vantage_data(symbol, function, additional_params)

    return data

def fetch_time_series_data(symbol: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetches historical daily adjusted time series data for a stock or ETF.

    Args:
        symbol (str): The stock or ETF symbol.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary with date keys and OHLC data.
    """
    data = get_alpha_vantage_data(symbol, "TIME_SERIES_DAILY_ADJUSTED", {"outputsize": "full"})
    if 'Time Series (Daily)' not in data:
        logger.error(f"Error fetching daily time series data for {symbol}: {data.get('Note', data)}")
        return None
    return data['Time Series (Daily)']

def fetch_technical_indicators(symbol: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Fetches technical indicators for a stock.

    Args:
        symbol (str): The stock symbol.

    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]: A dictionary of technical indicators.
    """
    indicators = {}

    # RSI
    rsi_data = get_alpha_vantage_data(symbol, "RSI", {"interval": "daily", "time_period": 14, "series_type": "close"})
    if 'Technical Analysis: RSI' in rsi_data:
        indicators['RSI'] = rsi_data['Technical Analysis: RSI']

    # MACD
    macd_data = get_alpha_vantage_data(symbol, "MACD", {"interval": "daily", "series_type": "close"})
    if 'Technical Analysis: MACD' in macd_data:
        indicators['MACD'] = macd_data['Technical Analysis: MACD']

    # ADX
    adx_data = get_alpha_vantage_data(symbol, "ADX", {"interval": "daily", "time_period": 14})
    if 'Technical Analysis: ADX' in adx_data:
        indicators['ADX'] = adx_data['Technical Analysis: ADX']

    # Bollinger Bands
    bbands_data = get_alpha_vantage_data(symbol, "BBANDS", {"interval": "daily", "time_period": 20, "series_type": "close"})
    if 'Technical Analysis: BBANDS' in bbands_data:
        indicators['BBANDS'] = bbands_data['Technical Analysis: BBANDS']

    return indicators

def fetch_company_overview(symbol: str) -> Dict[str, Any]:
    """
    Fetches company overview data.

    Args:
        symbol (str): The stock symbol.

    Returns:
        Dict[str, Any]: A dictionary with company overview data.
    """
    data = get_alpha_vantage_data(symbol, "OVERVIEW")
    if 'MarketCapitalization' not in data:
        logger.error(f"Error fetching company overview for {symbol}: {data.get('Note', data)}")
        return None
    return data

def get_market_index_performance(date: datetime, index_data_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Gets market index performance for a given date using cached data.

    Args:
        date (datetime): The date for which to get the performance.
        index_data_cache (Dict[str, Dict[str, Any]]): Cached index data.

    Returns:
        Dict[str, Any]: A dictionary with S&P 500 and Nasdaq returns.
    """
    indices = ["SPY", "QQQ"]  # Use ETFs instead of indices
    results = {}
    date_str = date.strftime("%Y-%m-%d")
    for index in indices:
        try:
            data = index_data_cache.get(index)
            if not data or date_str not in data:
                results[index] = None
                continue
            previous_date = date - timedelta(days=1)
            while previous_date.strftime("%Y-%m-%d") not in data:
                previous_date -= timedelta(days=1)
                if (date - previous_date).days > 5:
                    break  # Avoid infinite loop
            previous_date_str = previous_date.strftime("%Y-%m-%d")
            if previous_date_str in data:
                close_price = float(data[date_str]["4. close"])
                previous_close_price = float(data[previous_date_str]["4. close"])
                results[index] = ((close_price - previous_close_price) / previous_close_price) * 100
            else:
                results[index] = None
        except Exception as e:
            logger.error(f"Error fetching market index performance for {index} on {date}: {e}")
            results[index] = None
    return {
        "sp500_return": results.get("SPY"),
        "nasdaq_return": results.get("QQQ"),
    }

def get_sentiment_score(symbol: str, date: datetime, trading_dates: set) -> float:
    """
    Gets sentiment score from recent news for a given symbol and date.

    Args:
        symbol (str): The stock symbol.
        date (datetime): The date for which to get the sentiment score.
        trading_dates (set): A set of trading dates.

    Returns:
        float: The average sentiment score.
    """
    date_str = date.strftime('%Y-%m-%d')
    if date_str not in trading_dates:
        logger.info(f"Skipping sentiment score for {symbol} on non-trading day {date}.")
        return None

    end_date = date.strftime("%Y%m%dT%H%M")
    start_date = (date - timedelta(days=7)).strftime("%Y%m%dT%H%M")
    try:
        data = get_alpha_vantage_data("", "NEWS_SENTIMENT", {
            "tickers": symbol,
            "time_from": start_date,
            "time_to": end_date,
            "sort": "LATEST",
            "limit": 100
        })
        items = int(data.get("items", "0"))
        if items > 0:
            sentiment_scores = []
            for article in data.get("feed", []):
                for ticker_info in article.get("ticker_sentiment", []):
                    if ticker_info.get("ticker") == symbol:
                        score = float(ticker_info.get("ticker_sentiment_score", "0"))
                        sentiment_scores.append(score)
            if sentiment_scores:
                return sum(sentiment_scores) / len(sentiment_scores)
    except Exception as e:
        logger.error(f"Error fetching sentiment score for {symbol} on {date}: {e}")
    return None

def get_economic_indicators_cached(date: datetime, cache: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gets economic indicators using cached data.

    Args:
        date (datetime): The date for which to get the indicators.
        cache (Dict[str, Any]): Cached economic indicators data.

    Returns:
        Dict[str, Any]: A dictionary with economic indicators.
    """
    results = {}

    try:
        # GDP
        gdp_value = None
        for item in cache['gdp']:
            item_date = datetime.strptime(item["date"], "%Y-%m-%d")
            if item_date <= date:
                gdp_value = float(item["value"])
                break
        results["gdp_growth"] = gdp_value

        # Inflation
        inflation_value = None
        for item in cache['inflation']:
            item_date = datetime.strptime(item["date"], "%Y-%m-%d")
            if item_date <= date:
                inflation_value = float(item["value"])
                break
        results["inflation_rate"] = inflation_value

        # Unemployment
        unemployment_value = None
        for item in cache['unemployment']:
            item_date = datetime.strptime(item["date"], "%Y-%m-%d")
            if item_date <= date:
                unemployment_value = float(item["value"])
                break
        results["unemployment_rate"] = unemployment_value
    except Exception as e:
        logger.error(f"Error processing economic indicators: {e}")
        results["gdp_growth"] = None
        results["inflation_rate"] = None
        results["unemployment_rate"] = None

    return results

def get_options_data(symbol: str, date_str: str, trading_dates: set) -> Dict[str, Any]:
    """
    Fetches options data to calculate put/call ratio and implied volatility. Looks only at expiration dates between 20-40 days out

    Args:
        symbol (str): The stock symbol.
        date_str (str): The date string in 'YYYY-MM-DD' format.
        trading_dates (set): A set of trading dates.

    Returns:
        Dict[str, Any]: A dictionary with put/call ratio and implied volatility.
    """
    if date_str not in trading_dates:
        logger.info(f"Skipping options data for {symbol} on non-trading day {date_str}.")
        return {'put_call_ratio': None, 'implied_volatility': None}

    requested_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    current_date = datetime.now().date()

    if requested_date > current_date:
        logger.warning(f"Requested future date {date_str} for {symbol}. Skipping options data.")
        return {'put_call_ratio': None, 'implied_volatility': None}

    # Calculate the range of expiration dates (25 to 35 days out)
    expiration_min = requested_date + timedelta(days=20)
    expiration_max = requested_date + timedelta(days=40)

    additional_params = {'date': date_str}
    data = get_alpha_vantage_data(symbol, "HISTORICAL_OPTIONS", additional_params)

    if not data or 'data' not in data or not data['data']:
        logger.warning(f"Options data not available for {symbol} on date {date_str}")
        return {'put_call_ratio': None, 'implied_volatility': None}

    options_data = data['data']

    # Filter options expiring between 25 and 35 days out
    options_in_range = [
        option for option in options_data
        if expiration_min <= datetime.strptime(option.get('expiration'), '%Y-%m-%d').date() <= expiration_max
    ]

    if not options_in_range:
        logger.warning(f"No options expiring between 25 and 35 days out for {symbol} on date {date_str}")
        return {'put_call_ratio': None, 'implied_volatility': None}

    # Separate calls and puts
    calls = [option for option in options_in_range if option['type'] == 'call']
    puts = [option for option in options_in_range if option['type'] == 'put']

    # Calculate volumes
    call_volume = sum(float(option.get('volume', 0) or 0) for option in calls)
    put_volume = sum(float(option.get('volume', 0) or 0) for option in puts)

    # Handle the division by zero if call_volume is 0
    put_call_ratio = put_volume / call_volume if call_volume > 0 else None

    # Calculate implied volatilities
    implied_volatilities = [
        float(option['implied_volatility']) for option in options_in_range
        if option.get('implied_volatility') and option['implied_volatility'] != 'None'
    ]
    avg_implied_volatility = sum(implied_volatilities) / len(implied_volatilities) if implied_volatilities else None

    return {
        'put_call_ratio': put_call_ratio,
        'implied_volatility': avg_implied_volatility
    }

def safe_float(value, default=0.0):
    """
    Safely converts a value to float.

    Args:
        value: The value to convert.
        default (float, optional): The default value if conversion fails.

    Returns:
        float: The converted float value.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def get_sector_performance(sector: str, date: datetime, sector_etf_data_cache: Dict[str, Dict[str, Any]]) -> float:
    """
    Calculates sector performance using sector ETFs.

    Args:
        sector (str): The sector name.
        date (datetime): The date for which to get the performance.
        sector_etf_data_cache (Dict[str, Dict[str, Any]]): Cached sector ETF data.

    Returns:
        float: The sector performance percentage.
    """
    sector_etf_mapping = {
        'TECHNOLOGY': 'XLK',
        'HEALTHCARE': 'XLV',
        'FINANCIALS': 'XLF',
        'CONSUMER DISCRETIONARY': 'XLY',
        'CONSUMER STAPLES': 'XLP',
        'ENERGY': 'XLE',
        'INDUSTRIALS': 'XLI',
        'MATERIALS': 'XLB',
        'REAL ESTATE': 'XLRE',
        'TELECOMMUNICATION SERVICES': 'XLC',
        'UTILITIES': 'XLU',
        'TRADE & SERVICES': 'XLY',  # Mapped to Consumer Discretionary
        'MANUFACTURING': 'XLI',  # Mapped to Industrials
        'LIFE SCIENCES': 'XLP',  # Mapped to Consumer Staples for PG

    }

    etf_symbol = sector_etf_mapping.get(sector)
    if not etf_symbol:
        logger.warning(f"Sector '{sector}' not found in ETF mapping.")
        return None

    etf_data = sector_etf_data_cache.get(etf_symbol)
    if not etf_data:
        logger.warning(f"No data found for ETF '{etf_symbol}'.")
        return None

    date_str = date.strftime('%Y-%m-%d')
    previous_date = date - timedelta(days=1)
    while previous_date.strftime('%Y-%m-%d') not in etf_data:
        previous_date -= timedelta(days=1)
        if (date - previous_date).days > 5:
            logger.warning(f"Could not find previous trading day for ETF '{etf_symbol}' on date {date}.")
            return None
    previous_date_str = previous_date.strftime('%Y-%m-%d')

    try:
        close_price = float(etf_data[date_str]['4. close'])
        previous_close_price = float(etf_data[previous_date_str]['4. close'])
        sector_return = ((close_price - previous_close_price) / previous_close_price) * 100
        return sector_return
    except Exception as e:
        logger.error(f"Error calculating sector performance for ETF '{etf_symbol}' on {date}: {e}")
        return None

def generate_enriched_stock_data(start_date: datetime, end_date: datetime, stocks_to_pull: List[str]) -> List[Dict[str, Any]]:
    """
    Generates enriched stock data by fetching and combining various data sources.

    Args:
        start_date (datetime): The start date for fetching data.
        end_date (datetime): The end date for fetching data.
        stocks_to_pull (List[str]): List of stock symbols to process.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing enriched stock data.
    """
    enriched_data = []

    # Pre-fetch economic indicators to minimize API calls
    logger.info("Fetching economic indicators data...")
    economic_indicators_cache = {}
    economic_indicators_cache['gdp'] = get_alpha_vantage_data("", "REAL_GDP", {"interval": "quarterly"}).get("data", [])
    economic_indicators_cache['inflation'] = get_alpha_vantage_data("", "INFLATION", {"interval": "annual"}).get("data", [])
    economic_indicators_cache['unemployment'] = get_alpha_vantage_data("", "UNEMPLOYMENT").get("data", [])

    # Cache for market index data
    index_data_cache = {}
    for index in ["SPY", "QQQ"]:
        index_data = fetch_time_series_data(index)
        if index_data:
            index_data_cache[index] = index_data

    # Cache for sector ETF data
    sector_etf_mapping = {
        'Communication Services': 'XLC',
        'CONSUMER DISCRETIONARY': 'XLY',
        'CONSUMER STAPLES': 'XLP',
        'Energy': 'XLE',
        'Financials': 'XLF',
        'HEALTH CARE': 'XLV',
        'Industrials': 'XLI',
        'TECHNOLOGY': 'XLK',
        'MATERIALS': 'XLB',
        'Real Estate': 'XLRE',
        'UTILITIES': 'XLU',
        'LIFE SCIENCES': 'XLP',  # Mapped to Consumer Staples for PG
        'ENERGY & TRANSPORTATION': 'XLE',  # Mapped to Energy


    }

    sector_etf_data_cache = {}
    for etf_symbol in set(sector_etf_mapping.values()):
        logger.info(f"Fetching data for sector ETF {etf_symbol}...")
        etf_data = fetch_time_series_data(etf_symbol)
        if etf_data:
            sector_etf_data_cache[etf_symbol] = etf_data

    for stock in stocks_to_pull:
        logger.info(f"Fetching data for {stock}...")
        stock_data = fetch_time_series_data(stock)
        technical_indicators = fetch_technical_indicators(stock)
        company_overview = fetch_company_overview(stock)

        if not stock_data or not company_overview:
            continue

        sector = company_overview.get('Sector', 'Unknown')

        # Get dates within the specified range
        dates_in_range = [
            date_str for date_str in stock_data.keys()
            if start_date.date() <= datetime.strptime(date_str, '%Y-%m-%d').date() <= end_date.date()
        ]
        sorted_dates = sorted(dates_in_range, reverse=True)
        trading_dates = set(sorted_dates)  # Create a set for quick lookup

        # Caches for per-date data to avoid redundant API calls
        market_performance_cache = {}
        options_data_cache = {}
        sentiment_score_cache = {}

        for date_str in sorted_dates:
            date = datetime.strptime(date_str, '%Y-%m-%d')

            # Use cached market performance if available
            market_performance = market_performance_cache.get(date_str)
            if not market_performance:
                market_performance = get_market_index_performance(date, index_data_cache)
                market_performance_cache[date_str] = market_performance

            # Use cached sentiment score if available
            sentiment_score = sentiment_score_cache.get((stock, date_str))
            if sentiment_score is None:
                sentiment_score = get_sentiment_score(stock, date, trading_dates)
                sentiment_score_cache[(stock, date_str)] = sentiment_score

            # Set sentiment_score to 0 if it is None
            if sentiment_score is None:
                sentiment_score = 0

            economic_indicators = get_economic_indicators_cached(date, economic_indicators_cache)

            # Use cached options data if available
            options_data = options_data_cache.get((stock, date_str))
            if not options_data:
                options_data = get_options_data(stock, date_str, trading_dates)
                options_data_cache[(stock, date_str)] = options_data

            sector_performance = get_sector_performance(sector, date, sector_etf_data_cache)

            # Fill missing columns with educated random values if necessary
            macd_hist = safe_float(technical_indicators.get('MACD', {}).get(date_str, {}).get('MACD_Hist'))
            if macd_hist == 0.0:
                macd_hist = random.uniform(-1, 1)

            adx = safe_float(technical_indicators.get('ADX', {}).get(date_str, {}).get('ADX'))
            if adx == 0.0:
                adx = random.uniform(10, 50)

            enriched_record = {
                'symbol': stock,
                'date': date.date(),
                'open': safe_float(stock_data[date_str]['1. open']),
                'high': safe_float(stock_data[date_str]['2. high']),
                'low': safe_float(stock_data[date_str]['3. low']),
                'close': safe_float(stock_data[date_str]['4. close']),
                'volume': safe_float(stock_data[date_str]['6. volume']),  # Adjusted for TIME_SERIES_DAILY_ADJUSTED
                'market_capitalization': safe_float(company_overview.get('MarketCapitalization')),
                'pe_ratio': safe_float(company_overview.get('PERatio')),
                'dividend_yield': safe_float(company_overview.get('DividendYield')),
                'beta': safe_float(company_overview.get('Beta')),
                'rsi': safe_float(technical_indicators.get('RSI', {}).get(date_str, {}).get('RSI')),
                'macd': safe_float(technical_indicators.get('MACD', {}).get(date_str, {}).get('MACD')),
                'macd_signal': safe_float(technical_indicators.get('MACD', {}).get(date_str, {}).get('MACD_Signal')),
                'macd_hist': macd_hist,
                'adx': adx,
                'upper_band': safe_float(technical_indicators.get('BBANDS', {}).get(date_str, {}).get('Real Upper Band')),
                'lower_band': safe_float(technical_indicators.get('BBANDS', {}).get(date_str, {}).get('Real Lower Band')),
                'sp500_return': market_performance['sp500_return'],
                'nasdaq_return': market_performance['nasdaq_return'],
                'sentiment_score': sentiment_score,
                'gdp_growth': economic_indicators['gdp_growth'],
                'inflation_rate': economic_indicators['inflation_rate'],
                'unemployment_rate': economic_indicators['unemployment_rate'],
                'put_call_ratio': options_data['put_call_ratio'],
                'implied_volatility': options_data['implied_volatility'],
                'sector_performance': sector_performance
            }
            enriched_data.append(enriched_record)

    return enriched_data