import openai
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta

# Initialize OpenAI API and database connection
openai.api_key = "YOUR_API_KEY"
engine = create_engine('postgresql+psycopg2://user:password@host/dbname')


# Define stock data fetching function
def fetch_stock_data(symbol: str, start_date: str):
    query = f"SELECT * FROM enriched_stock_data WHERE symbol = '{symbol}' AND date >= '{start_date}'"
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')


# Define stock prediction function
def predict_stock_price(stock_data: str):
    stock_df = pd.read_json(stock_data)
    if stock_df.empty:
        return {"error": "No stock data available."}

    # Dummy model: Example - increase the price by 5% (Replace with your own model)
    predicted_price = stock_df['close'].iloc[-1] * 1.05
    return {"predicted_price": predicted_price}


# Define comparison function
def compare_prediction_to_actual(symbol: str, predicted_price: float, prediction_date: str):
    query = f"SELECT close FROM enriched_stock_data WHERE symbol = '{symbol}' AND date = '{prediction_date}'"
    actual_price_df = pd.read_sql(query, engine)
    if actual_price_df.empty:
        return {"error": "No actual price found for the prediction date."}

    actual_price = actual_price_df['close'].iloc[0]
    return {
        "predicted_price": predicted_price,
        "actual_price": actual_price,
        "difference": actual_price - predicted_price
    }


# ChatGPT API Call with tools integration
def chat_gpt_stock_assistant(user_input, symbol, analysis_date):
    # Define the functions (tools) ChatGPT can call
    functions = [
        {
            "name": "fetch_stock_data",
            "description": "Fetch historical stock data.",
            "parameters": {
                "symbol": {"type": "string"},
                "start_date": {"type": "string"}
            }
        },
        {
            "name": "predict_stock_price",
            "description": "Predict stock price based on historical data.",
            "parameters": {
                "stock_data": {"type": "string"}
            }
        },
        {
            "name": "compare_prediction_to_actual",
            "description": "Compare the predicted stock price to the actual price.",
            "parameters": {
                "symbol": {"type": "string"},
                "predicted_price": {"type": "float"},
                "prediction_date": {"type": "string"}
            }
        }
    ]

    # Create initial system prompt and messages
    messages = [
        {"role": "system", "content": "You are a stock trading assistant."},
        {"role": "user", "content": user_input}
    ]

    # Create a ChatGPT call with function calling
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=messages,
        functions=functions,
        function_call="auto"
    )

    # Extract response content and function calls
    response_message = response.choices[0].message
    return response_message


# Example conversation flow using the chatbot
if __name__ == "__main__":
    # Inputs from the user
    symbol = "AAPL"
    analysis_date = "2024-09-01"
    user_input = f"Predict the price of {symbol} 30 days from {analysis_date}"

    # Trigger the ChatGPT API
    chat_response = chat_gpt_stock_assistant(user_input, symbol, analysis_date)
    print(chat_response)