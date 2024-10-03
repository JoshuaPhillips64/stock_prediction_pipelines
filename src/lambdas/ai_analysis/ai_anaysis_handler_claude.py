from datetime import datetime, timedelta
import pandas as pd
from database_functions import fetch_dataframe, upsert_df, create_engine_from_url
from ai_functions import parse_response_to_json, get_regression_system_prompt
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_DATABASE
import json
from ai_functions import cache_static_content_with_anthropic, execute_claude_call_with_cached_content

# Database connection string
db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"

#%%
def process_stock_prediction(stock_symbol, analysis_date, prediction_date, cached_messages, engine):
    # User input based on stock symbol and prediction date
    user_input = f"""
    Please analyze the data above for {stock_symbol} and predict the stock price for {prediction_date}:
    
    You are not to use any data given beyond {analysis_date} for your prediction. This includes any data that may have been released after this date. 
    You must provide your analysis and prediction in JSON format with no other text.
    """

    # Use the cached messages and append user input for this specific stock and date
    gpt_response = execute_claude_call_with_cached_content(user_input, cached_messages)
    parsed_response = parse_response_to_json(gpt_response)

    print(f"Parsed response for {stock_symbol} on {analysis_date}: {parsed_response}")

    if parsed_response is None:
        raise ValueError("Failed to parse GPT response to JSON")

    prediction_data = pd.DataFrame([{
        'symbol': stock_symbol,
        'date': analysis_date,
        'predicted_amount': parsed_response.get('predictedamount'),
        'prediction_date': prediction_date,
        'prediction_explanation': parsed_response.get('predictionexplanation'),
        'prediction_rmse': parsed_response.get('predictionrmse'),
        'prediction_confidence_score': parsed_response.get('predictionconfidencescore'),
        'up_or_down': parsed_response.get('upordown'),
        'date_created': datetime.now().strftime('%Y-%m-%d'),
        'feature_importance': json.dumps(parsed_response.get('featureimportance')),
    }])

    if prediction_data['predicted_amount'].isnull().values.any():
        raise ValueError("Prediction not available in the response")

    # Ensure that date and prediction_date are properly cast as dates in the DataFrame
    prediction_data['date'] = pd.to_datetime(prediction_data['date']).dt.date
    prediction_data['prediction_date'] = pd.to_datetime(prediction_data['prediction_date']).dt.date

    # Load prediction data into database
    success = upsert_df(
        df=prediction_data,
        table_name='ai_stock_predictions',
        upsert_id='symbol, date',
        postgres_connection=engine,
        json_columns=['feature_importance'],
        auto_match_schema='public'
    )

    if not success:
        raise RuntimeError("Failed to upsert prediction data")

    return f"Successfully processed prediction for {stock_symbol} on {prediction_date}"


def lambda_handler(event, context):
    engine = create_engine_from_url(db_url)

    try:
        stock_symbols = event.get('stock_symbols', [])
        analysis_dates = event.get('analysis_dates', [])

        if not stock_symbols or not analysis_dates:
            raise ValueError("Both stock_symbols and analysis_dates must be provided")

        # Get the system prompt
        system_prompt = get_regression_system_prompt()

        results = []

        # Process each stock symbol separately
        for symbol in stock_symbols:
            # Fetch data for the current stock
            query = f"""
                SELECT * FROM enriched_stock_data
                WHERE symbol = '{symbol}'
                ORDER BY date DESC
                LIMIT 400
            """
            df = fetch_dataframe(engine, query)

            if df.empty:
                raise ValueError(f"No data found for the stock symbol: {symbol}")

            # Convert the dataset to JSON format
            data_for_ai = df.to_json(orient='records')

            # Cache the static content (system prompt + data) for this stock
            cached_messages = cache_static_content_with_anthropic(system_prompt, data_for_ai)

            # Process predictions for all analysis dates for this stock
            for analysis_date in analysis_dates:
                try:
                    analysis_date_obj = datetime.strptime(analysis_date, '%Y-%m-%d')
                    prediction_date = (analysis_date_obj + timedelta(days=30)).strftime('%Y-%m-%d')

                    result = process_stock_prediction(symbol, analysis_date, prediction_date, cached_messages, engine)
                    results.append(result)
                except Exception as e:
                    results.append(f"Error processing {symbol} for {analysis_date}: {str(e)}")

        return {
            'statusCode': 200,
            'body': "\n".join(results)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': f"Error processing predictions: {str(e)}"
        }


# For local testing
if __name__ == "__main__":
    test_event = {
        'stock_symbols': ['PG'],
        'analysis_dates': [
            '2024-06-30',
            '2024-07-29',
            '2024-05-28',
        ]
    }
    print(lambda_handler(test_event, None))