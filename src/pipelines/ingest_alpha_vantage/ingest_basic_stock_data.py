#%%
from alpha_vantage_functions import fetch_time_series_data, safe_float

def generate_basic_stock_data(symbol: str) -> List[Dict[str, Any]]:
    """
    Generates basic stock data by fetching time series data for a given stock symbol.

    Args:
        symbol (str): The stock symbol to fetch data for.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing basic stock data.
    """
    logger.info(f"Fetching basic stock data for {symbol}...")
    stock_data = fetch_time_series_data(symbol)

    if not stock_data:
        logger.error(f"No data found for symbol {symbol}")
        return []

    # Convert the stock data into a list of dictionaries with basic fields
    basic_data = []
    for date_str, data in stock_data.items():
        try:
            basic_record = {
                'symbol': symbol,
                'date': datetime.strptime(date_str, '%Y-%m-%d').date(),
                'open': safe_float(data['1. open']),
                'high': safe_float(data['2. high']),
                'low': safe_float(data['3. low']),
                'close': safe_float(data['4. close']),
                'volume': safe_float(data['6. volume'])
            }
            basic_data.append(basic_record)
        except Exception as e:
            logger.error(f"Error processing data for {symbol} on {date_str}: {e}")

    return basic_data

df = pd.DataFrame(generate_basic_stock_data('AAPL'))