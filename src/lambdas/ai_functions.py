from openai import OpenAI
import re
import os
import json
from config import OPENAI_API_KEY
import ast

def execute_chatgpt_call(user_input, system_prompt):
    # Create an OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    try:

        # Define the messages to be sent to the API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_input}"}
        ]
        print(f'Here are the conversation sent to chatgpt: {messages}')

        # Make the API call to the OpenAI Chat Completion endpoint
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use a valid model name
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        # Log the token usage information
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}")

        # Return the generated content from the response
        return response.choices[0].message.content
    except Exception as e:
        # Handle API errors
        print(f"An error occurred: {e}")
        return None

def parse_response_to_json(response):
    if not response or not isinstance(response, str):
        print(f"Invalid response: {response}")
        return None

    try:
        # Clean the response to remove any wrapping ```json and whitespace
        cleaned_response = re.sub(r'^```json\s*|\s*```$', '', response.strip())

        # Try to parse as JSON first
        try:
            data = json.loads(cleaned_response)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to parse as a Python dictionary
            try:
                # Use ast.literal_eval to safely evaluate the string as a Python literal
                data = ast.literal_eval(cleaned_response)
                # Convert the Python dictionary to a JSON string and then back to a dictionary
                data = json.loads(json.dumps(data))
            except (ValueError, SyntaxError) as e:
                print(f"Invalid JSON and unable to parse as Python dict: {e}")
                print("Raw response:", response)
                return None

        # Create a new dictionary with sanitized keys and non-null values
        sanitized_data = {}
        for key, value in data.items():
            # Remove special characters from the key
            sanitized_key = re.sub(r'[^a-zA-Z0-9 ]', '', key)
            # Only include non-null values
            if value is not None:
                sanitized_data[sanitized_key] = value

        # Check if sanitized_data is empty
        if not sanitized_data:
            print("All values were null or removed during sanitization.")
            return None

        return sanitized_data

    except Exception as e:
        print(f"Unexpected error in parse_response_to_json: {e}")
        print("Raw response:", response)
        return None


def get_regression_system_prompt():
    return """
   You are an advanced AI model trained to predict stock prices using historical and financial data. Analyze a dataset of stock market indicators and make a prediction for the price 30 days from the date given. 

    Focus on relevant features such as consumer sentiment, put_call_ratio, sector performance, GDP growth, inflation rates, and key technical indicators. Evaluate feature importance critically and ensure the most relevant indicators are highlighted. Provide a JSON output with the following fields: 
    
    - Predicted_Price_30_Days
    - Confidence Score (r2 * 100)
    - Prediction RMSE (Root Mean Squared Error)
    - Prediction_Explanation (3-4 concise sentences not restating any other json key that highlight the most significant feature impacts to the model and their importance score.)
    - Further_Considerations (suggest additional data points or factors to improve accuracy)
    - Up_or_Down (binary classification of likely price movement)
    - Nested JSON object with feature importance scores for top 6 indicator used in the model.
    
    Ensure the analysis is thorough and clear for non-experts, and if the initial output is unsatisfactory, refine it based on feedback.
    """

def get_classification_system_prompt():
    return """
    
    """