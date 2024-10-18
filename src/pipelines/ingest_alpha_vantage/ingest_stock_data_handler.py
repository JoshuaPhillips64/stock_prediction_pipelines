#%%
import json
import logging
from datetime import datetime, date
from alpha_vantage_functions import generate_enriched_stock_data, generate_basic_stock_data
from database_functions import create_engine_from_url, upsert_df
import pandas as pd
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_DATABASE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def lambda_handler(event, context):
    try:
        # Parse input parameters
        body = json.loads(event['body'])
        logger.info(f"Received body: {body}")

        stocks = body.get('stocks', [])
        start_date_str = body.get('start_date')
        end_date_str = body.get('end_date')
        feature_set = body.get('feature_set', 'basic')

        # Validate start_date and end_date
        if not start_date_str or not end_date_str:
            return {
                'statusCode': 400,
                'body': json.dumps('Start date or end date is missing')
            }

        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

        if not stocks:
            return {
                'statusCode': 400,
                'body': json.dumps('No stocks provided')
            }

        # Generate stock data
        if feature_set == 'basic':
            stock_data = generate_basic_stock_data(start_date, end_date, stocks)
        else:
            stock_data = generate_enriched_stock_data(start_date, end_date, stocks)

        if not stock_data:
            return {
                'statusCode': 404,
                'body': json.dumps('No data found for the specified date range')
            }

        # Convert to DataFrame
        df = pd.DataFrame(stock_data)

        # Create database connection
        db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine_from_url(db_url)

        # Determine upsert table based on feature set
        upsert_table = 'basic_stock_data' if feature_set == 'basic' else 'enriched_stock_data'

        # Upsert data into the database
        upsert_df(
            df=df,
            table_name=upsert_table,
            upsert_id='symbol, date',
            postgres_connection=engine,
            auto_match_platform='postgres'
        )

        return {
            'statusCode': 200,
            'body': json.dumps(stock_data, default=json_serial)
        }

    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }


# Example implementation

#%% This section demonstrates how to use the lambda function locally
example_event = {
    'body': json.dumps({
        'stocks': ['PG'],
        'start_date': '2024-01-01',
        'end_date': '2024-10-10',
        'feature_set': 'basic'
    })
}

example_context = {}

#result = lambda_handler(example_event, example_context)
#print(f"Lambda function result: {result}")