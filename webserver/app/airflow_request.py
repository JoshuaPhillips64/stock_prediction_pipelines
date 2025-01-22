"""
Example of how to call the Airflow API to trigger a DAG run with a specific configuration from the webapp.

airflow code is in src/airflow/dags/dynamic_api_generate_stock_prediction_dag.py
"""

import requests

airflow_ip = "10.10.10.10"
airflow_url = f"http://{airflow_ip}:8080/api/v1/dags/dynamic_api_generate_stock_prediction_dag/dagRuns"
airflow_user = ""
airflow_pass = ""

def call_airflow_api():
    url = airflow_url

    data = {
    "conf": {
        "stocks": [
            {
                "stock": "AAPL",
                "params": {
                    "model_type": "SARIMAX",
                    "feature_set": "basic",
                    "lookback_period": 30,
                    "prediction_horizon": 7,
                    "hyperparameter_tuning": "LOW"
                }
            }
        ]
    }
    }

    headers = {
    "Content-Type": "application/json"
    }


    auth = (airflow_user, airflow_pass)

    try:
        response = requests.post(url, json=data, headers=headers, auth=auth)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        print("Status Code:", response.status_code)
        print("Response Body:", response.json())
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # Python 3.6+
        print("Response Body:", response.text)
    except Exception as err:
        print(f"Other error occurred: {err}")