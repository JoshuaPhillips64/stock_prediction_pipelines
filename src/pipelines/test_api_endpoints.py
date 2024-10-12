#%%
from config import API_URL
import json
import requests

def invoke_ingest_stock_data():
    # Define the API Gateway URL and endpoint
    api_url = f"{API_URL}/ingest-stock-data"

    # Define the payload (event data)
    example_event = {
        'body': json.dumps({
            'stocks': ['PG'],
            'start_date': '2024-01-01',
            'end_date': '2024-10-10',
            'feature_set': 'basic'
        })
    }

    # Make the POST request
    response = requests.post(api_url, json=example_event)

    # Check if the request was successful
    if response.status_code == 200:
        print("Response received:", response.json())
    else:
        print(f"Failed to invoke API. Status code: {response.status_code}")
        print("Error response:", response.text)

#%%
# Call the function to invoke the API
invoke_ingest_stock_data()