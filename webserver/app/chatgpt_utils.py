"""
This is placeholder code for when we launch the interactive AI chat agent that will predict stocks
"""

import os
import json
import re
from datetime import datetime
from flask import session
from openai import OpenAI
from .lambda_endpoints import (
    ingest_stock_data,
    train_binary_classification_model,
    make_sarimax_prediction,
    make_binary_prediction,
    train_sarimax_model,
    trigger_ai_analysis)

from .airflow_request import call_airflow_api

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function schemas
schemas = [
    {
        "name": "train_stock_prediction_model_airflow_api",
        "description": "Trains a model for stock prediction based on specified parameters for any given stock.",
        "parameters": {
            "type": "object",
            "properties": {
                "stock_symbol": {
                    "type": "string",
                    "description": "Stock symbol to train on."
                },
                "model_type": {
                    "type": "string",
                    "description": "Type of model to train.",
                    "enum": ["SARIMAX", "BINARY CLASSIFICATION"]
                },
                "feature_set": {
                    "type": "string",
                    "description": "Feature set to use for model training.",
                    "enum": ["basic", "advanced"],
                    "default": "basic"
                },
                "lookback_period": {
                    "type": "integer",
                    "description": "Number of days to look back for training data.",
                    "default": 720
                },
                "prediction_horizon": {
                    "type": "integer",
                    "description": "Number of days to predict forward.",
                    "default": 30
                },
                "hyperparameter_tuning": {
                    "type": "string",
                    "description": "Whether to apply hyperparameter tuning.",
                    "enum": ["LOW", "MEDIUM", "HIGH"],
                    "default": "LOW"
                }
            },
            "required": ["stock_symbol", "model_type", "lookback_period", "prediction_horizon"]
        }
    }
    # Add other schemas as needed - Will need to add make prediction and then retrieve top models
]

# Map function names to actual functions
function_map = {
    "make_sarimax_prediction": make_sarimax_prediction,
    "make_binary_prediction": make_binary_prediction,
    "train_stock_prediction_model_airflow_api": call_airflow_api
}


def manage_message_history(messages, limit=5):
    """Limit message history to control token usage."""
    return messages[-limit:]


def execute_chatgpt_call(messages):
    """
    Executes a call to the OpenAI ChatGPT API with the provided messages.

    Args:
        messages (list): A list of message dictionaries following the OpenAI Chat API format.

    Returns:
        ChatCompletion or None: The API response object if successful, otherwise None.
    """
    # Set the OpenAI API key
    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        # Make the API call to the OpenAI Chat Completion endpoint
        response = client.chat.completions.create(
            model="gpt-4o",  # Use a valid model name
            messages=messages,
            temperature=0,
            max_tokens=1000,
            functions=schemas,  # Add schemas here for function calling
            function_call="auto",  # Allows ChatGPT to call functions automatically based on message content
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        # Log the token usage information
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}")

        # Return the entire response for further processing
        return response

    except Exception as e:
        # Handle API errors
        print(f"An error occurred during the API call: {e}")
        return None


def handle_chatgpt_response(messages):
    """
    Handles the response from ChatGPT, including function calls if present.
    """
    messages = manage_message_history(messages)
    response = execute_chatgpt_call(messages)

    # Ensure the response contains choices
    if not response or not response.choices:
        return "An error occurred during the API call."

    message = response.choices[0].message

    # Check if 'function_call' is present in the message
    if message.function_call:
        function_name = message.function_call.name
        function_args = json.loads(message.function_call.arguments or '{}')

        if function_name in function_map:
            # Call the corresponding function
            function_to_call = function_map[function_name]
            try:
                function_response = function_to_call(**function_args)
            except Exception as e:
                return f"Error in function '{function_name}': {str(e)}"

            # Append the function's response to the messages
            messages.append({
                "role": "function",
                "name": function_name,
                "content": json.dumps(function_response)
            })

            # Call OpenAI API again with the function's response
            response = execute_chatgpt_call(messages)
            if not response or not response.choices:
                return "An error occurred during the API call."

            ai_message = response.choices[0].message.content or ''
            return ai_message
        else:
            return f"Function '{function_name}' not supported."
    else:
        ai_message = message.content or ''
        return ai_message


def increment_session_request_count(max_requests=20):
    """Increments session request count to enforce a request limit."""
    if 'request_count' not in session:
        session['request_count'] = 0
    if session['request_count'] >= max_requests:
        raise Exception("Session limit reached. Please start a new session.")
    session['request_count'] += 1


def parse_response_to_json(response):
    """Parse response content safely into JSON or a sanitized dictionary."""
    if not response or not isinstance(response, str):
        print(f"Invalid response: {response}")
        return None

    try:
        # Remove any JSON formatting markers like ```json or surrounding whitespace
        cleaned_response = re.sub(r'^```json\s*|\s*```$', '', response.strip())

        # Attempt to parse the response as JSON
        try:
            data = json.loads(cleaned_response)
        except json.JSONDecodeError:
            print(f"Failed to parse response as JSON: {response}")
            return None

        # Create a sanitized dictionary with non-null values and cleaned keys
        sanitized_data = {
            re.sub(r'[^a-zA-Z0-9_]', '', key): value
            for key, value in data.items() if value is not None
        }

        # Check if sanitized_data is non-empty
        if not sanitized_data:
            print("All values were null or removed during sanitization.")
            return None

        return sanitized_data

    except Exception as e:
        print(f"Unexpected error during response parsing: {e}")
        print("Raw response:", response)
        return None