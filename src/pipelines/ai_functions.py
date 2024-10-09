from openai import OpenAI
import re
import os
import json
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY
import ast
import os
import time
import logging
from anthropic import Anthropic, RateLimitError, APIError

def execute_chatgpt_call(user_input, system_prompt, data=None):
    # Create an OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        # Define the messages to be sent to the API
        messages = []

        # Add the system prompt and user input
        messages.append({"role": "system", "content": system_prompt})
        if data:
            messages.append({"role": "system", "content": data})
        messages.append({"role": "user", "content": user_input})

        # Make the API call to the OpenAI Chat Completion endpoint
        response = client.chat.completions.create(
            model="gpt-4o",  # Use a valid model name
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
        cached_tokens = response.usage.prompt_tokens_details.get("cached_tokens", 0)
        reasoning_tokens = response.usage.completion_tokens_details.get("reasoning_tokens", 0)

        # Determine how many new tokens were processed
        uncached_tokens = prompt_tokens - cached_tokens

        print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}")
        print(f"Cached tokens: {cached_tokens}, Uncached tokens processed: {uncached_tokens}")
        print(f"Reasoning tokens: {reasoning_tokens}")

        # Return the generated content from the response
        return response.choices[0].message.content

    except Exception as e:
        # Handle API errors
        print(f"An error occurred: {e}")
        return None

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

    Focus on all features given, including consumer sentiment, put_call_ratio, sector performance, GDP growth, inflation rates, and key technical indicators. 
    Evaluate feature importance critically and ensure the most relevant indicators are highlighted. 
    Check for anomalies and outliers in the data. Earnings info and dates are in data set as the eps columns.
    
    - Predicted_Price_30_Days
    - Confidence Score (r2 * 100)
    - Prediction RMSE (Root Mean Squared Error)
    - Prediction_Explanation (3-4 concise sentences not restating any other json key that highlight the most significant feature impacts to the model and their importance score.)
    - Further_Considerations (suggest additional data points or factors to improve accuracy)
    - Up_or_Down (binary classification of likely price movement)
    - Nested JSON object with feature importance scores for top 6 indicator used in the model.
    
    You must provide the output in JSON format with the exact keys as below (including underscores), here are the types:
        {
      "symbol": "str",
      "date": "datetime.date",
      "predicted_amount": "float",
      "prediction_date": "datetime.date",
      "prediction_explanation": "str",
      "prediction_rmse": "float",
      "prediction_confidence_score": "float",
      "up_or_down": "str",
      "date_created": "str",
      "feature_importance": "str",
      "further_considerations": "str"
    }
    
    Review the prediction against last known price to ensure it is logical and well-supported by the data.
    Ensure the analysis is thorough and clear for non-experts, and if the initial output is unsatisfactory, refine it based on feedback.
    """

def get_classification_system_prompt():
    return """
    
    """

"""

FOR USING ANTHROPIC API FOR CACHING PROMPTS

"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Anthropic client with the beta header
client = Anthropic(
    api_key=ANTHROPIC_API_KEY,
    default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
)

class CacheManager:
    def __init__(self, refresh_interval=285):  # 285 seconds = 4m 45s
        self.last_refresh_time = None
        self.refresh_interval = refresh_interval

    def needs_refresh(self):
        if self.last_refresh_time is None:
            return True
        return time.time() - self.last_refresh_time >= self.refresh_interval

    def update_refresh_time(self):
        self.last_refresh_time = time.time()

cache_manager = CacheManager()

def retry_with_fixed_delay(func, max_retries=5, delay=30):
    """Retry a function with a fixed delay."""
    retries = 0
    while retries < max_retries:
        try:
            response = func()
            print_response_details(response)
            return response
        except RateLimitError as e:
            if retries == max_retries - 1:
                raise
            logging.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)
            retries += 1
        except APIError as e:
            logging.error(f"API Error: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise

def print_response_details(response):
    """Print details of the Claude API response."""
    logging.info(f"Response received:")
    logging.info(f"Usage: {response.usage}")
    logging.info(f"Model: {response.model}")
    logging.info(f"Content: {response.content}")

def cache_static_content_with_anthropic(system_prompt, data=None):
    """Caches static content using Anthropic's prompt caching."""
    if not cache_manager.needs_refresh():
        logging.info("Cache is still valid. No need to refresh.")
        return None

    logging.info("Refreshing the cache...")

    static_messages = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"}
        }
    ]

    if data:
        static_messages.append({
            "type": "text",
            "text": data,
            "cache_control": {"type": "ephemeral"}
        })

    def api_call():
        return client.messages.create(
            model="claude-3-5-sonnet-20240620",
            system=static_messages,
            messages=[{"role": "user", "content": "Initialize cache"}],
            max_tokens=1
        )

    try:
        response = retry_with_fixed_delay(api_call)
        cache_manager.update_refresh_time()
        return static_messages
    except Exception as e:
        logging.error(f"Failed to cache content after multiple retries: {e}")
        return None

def execute_claude_call_with_cached_content(user_input, cached_messages):
    """Executes the Claude API call using Anthropic's cached content."""
    if not cached_messages:
        logging.warning("No cached messages provided.")
        system_messages = []
    else:
        system_messages = cached_messages

    def api_call():
        return client.messages.create(
            model="claude-3-5-sonnet-20240620",
            system=system_messages,
            messages=[{"role": "user", "content": user_input}],
            max_tokens=1000,
            temperature=0.3
        )

    try:
        response = retry_with_fixed_delay(api_call)
        return response.content[0].text
    except Exception as e:
        logging.error(f"Failed to execute Claude call after multiple retries: {e}")
        return None