import os
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
from database_functions import fetch_dataframe, upsert_df, create_engine_from_url
from ai_functions import execute_chatgpt_call, parse_response_to_json, get_regression_system_prompt
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_DATABASE
import json

# Database connection string
db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"

#%%

def process_stock_prediction(stock_symbol, analysis_date, engine):
    analysis_date_obj = datetime.strptime(analysis_date, '%Y-%m-%d')
    prediction_date = (analysis_date_obj + timedelta(days=30)).strftime('%Y-%m-%d')

    query = f"""
        SELECT * FROM enriched_stock_data
        WHERE symbol = '{stock_symbol}' AND date <= '{analysis_date}'
        ORDER BY date DESC
        LIMIT 200
    """
    df = fetch_dataframe(engine, query)

    if df.empty:
        raise ValueError(f"No data found for stock {stock_symbol} up to {analysis_date}")

    data_for_ai = df.to_json(orient='records')
    system_prompt = get_regression_system_prompt()
    user_input = f"""
    Please analyze the following data for {stock_symbol} and predict the stock price for {prediction_date}:
    {data_for_ai}
    Provide your analysis and prediction in JSON format.
    """

    gpt_response = execute_chatgpt_call(user_input, system_prompt)
    parsed_response = parse_response_to_json(gpt_response)

    print(parsed_response)

    if parsed_response is None:
        raise ValueError("Failed to parse GPT response to JSON")

    prediction_data = pd.DataFrame([{
        'symbol': stock_symbol,
        'date': analysis_date,
        'predicted_amount': parsed_response.get('PredictedPrice30Days'),
        'prediction_date': prediction_date,
        'prediction_explanation': parsed_response.get('PredictionExplanation'),
        'prediction_rmse': parsed_response.get('PredictionRMSE'),
        'prediction_confidence_score': parsed_response.get('ConfidenceScore'),
        'up_or_down': parsed_response.get('UporDown'),
        'date_created': datetime.now().strftime('%Y-%m-%d'),
        'feature_importance': json.dumps(parsed_response.get('FeatureImportance')),
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

        results = []
        for symbol in stock_symbols:
            for date in analysis_dates:
                try:
                    result = process_stock_prediction(symbol, date, engine)
                    results.append(result)
                except Exception as e:
                    results.append(f"Error processing {symbol} for {date}: {str(e)}")

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
        'stock_symbols': ['WMT'],
        'analysis_dates': [
    '2024-08-30', '2024-08-29', '2024-08-28', '2024-08-27',
    '2024-08-26', '2024-08-25', '2024-08-24', '2024-08-23',
    '2024-08-22', '2024-08-21', '2024-08-20', '2024-08-19',
    '2024-08-18', '2024-08-17', '2024-08-16', '2024-08-15',
    '2024-08-14', '2024-08-13', '2024-08-12', '2024-08-11',
    '2024-08-10', '2024-08-09', '2024-08-08', '2024-08-07',
    '2024-08-06', '2024-08-05', '2024-08-04', '2024-08-03'
]
    }
    print(lambda_handler(test_event, None))