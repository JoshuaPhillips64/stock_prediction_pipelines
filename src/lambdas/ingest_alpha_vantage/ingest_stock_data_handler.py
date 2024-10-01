import json
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
from alpha_vantage_functions import generate_enriched_stock_data
from database_functions import create_engine_from_url, upsert_df
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#%%
def lambda_handler(event, context):
    try:
        # Parse input parameters
        body = json.loads(event['body'])
        stocks = body.get('stocks', [])
        start_date = datetime.strptime(body.get('start_date'), '%Y-%m-%d')
        end_date = datetime.strptime(body.get('end_date'), '%Y-%m-%d')

        if not stocks:
            return {
                'statusCode': 400,
                'body': json.dumps('No stocks provided')
            }

        # Generate enriched stock data
        enriched_data = generate_enriched_stock_data(start_date, end_date, stocks)

        if not enriched_data:
            return {
                'statusCode': 404,
                'body': json.dumps('No data found for the specified date range')
            }

        # Convert to DataFrame
        df = pd.DataFrame(enriched_data)

        # Create database connection
        db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine_from_url(db_url)

        upsert_df(
            df=df,
            table_name='enriched_stock_data',
            upsert_id='symbol, date',
            postgres_connection=engine,
            auto_match_schema='public'
        )

        return {
            'statusCode': 200,
            'body': json.dumps(f'Successfully processed {len(enriched_data)} records for {len(stocks)} stocks')
        }

    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }


# Example implementation
if __name__ == "__main__":
    # This section demonstrates how to use the lambda function locally
    example_event = {
        'body': json.dumps({
            'stocks': ['KO', 'JNJ', 'PG', 'PEP', 'WMT', 'XOM', 'GIS', 'MCD', 'CL', 'MMM'],
            'start_date': '2023-01-01',
            'end_date': '2024-01-01'
        })
    }

    example_context = {}

    result = lambda_handler(example_event, example_context)
    print(f"Lambda function result: {result}")