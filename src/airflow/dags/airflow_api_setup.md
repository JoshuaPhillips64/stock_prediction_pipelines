# Explicit Instructions for Airflow Setup

To make this DAG work in your Airflow environment, follow these detailed steps:

### Configure Authentication for the Airflow REST API

#### Enable the REST API:

Airflow's REST API is enabled by default in Airflow 2.x.

#### Authentication Backend:

Choose an authentication method. For simplicity, you can use basic authentication or token authentication.

#### Update `airflow.cfg`:

```ini
[api]
auth_backend = airflow.api.auth.backend.basic_auth
```

#### Create Users:

Create an Airflow user with the necessary permissions:

```bash
airflow users create \
    --username your_username \
    --firstname FIRSTNAME \
    --lastname LASTNAME \
    --role Admin \
    --email your_email@example.com
```

### 8. Trigger the DAG via the REST API

#### Prepare the Payload:

Create a JSON payload with the configuration you wish to pass. For example:

```json
{
  "conf": {
    "stocks": [
      {
        "stock": "AAPL",
        "params": {
          "model_type": "SARIMAX",
          "feature_set": "basic",
          "lookback_period": 30,
          "prediction_horizon": 7,
          "hyperparameter_tuning": true
        }
      },
      {
        "stock": "GOOGL",
        "params": {
          "model_type": "BINARY CLASSIFICATION",
          "feature_set": "advanced",
          "lookback_period": 60,
          "prediction_horizon": 14,
          "hyperparameter_tuning": false
        }
      }
    ]
  }
}
```

#### Make the API Call:

Use curl or any HTTP client to trigger the DAG:

```bash
curl -X POST "http://your-airflow-url.com/api/v1/dags/generate_stock_prediction_dag/dagRuns" \
     -H "Content-Type: application/json" \
     -u your_username:your_password \
     -d '{
           "conf": {
             "stocks": [
               {
                 "stock": "AAPL",
                 "params": {
                   "model_type": "SARIMAX",
                   "feature_set": "basic",
                   "lookback_period": 30,
                   "prediction_horizon": 7,
                   "hyperparameter_tuning": true
                 }
               },
               {
                 "stock": "GOOGL",
                 "params": {
                   "model_type": "BINARY CLASSIFICATION",
                   "feature_set": "advanced",
                   "lookback_period": 60,
                   "prediction_horizon": 14,
                   "hyperparameter_tuning": false
                 }
               }
             ]
           }
         }'
```
Replace `your-airflow-url.com` with your Airflow server's URL.
Replace `your_username` and `your_password` with your Airflow credentials.

###  Monitor the DAG Execution

#### Airflow UI:

Navigate to the Airflow web UI to monitor the DAG runs.
Check the logs for each task to ensure they are executing correctly.

### Verify AWS Lambda Functions Execution

#### AWS Console:

Check the AWS Lambda console to verify that the functions are being invoked.
Monitor CloudWatch logs for detailed execution logs.

## Important Notes

- **Airflow Version Compatibility**: The code uses Dynamic Task Mapping, which is available in Airflow 2.3 and later. Ensure your Airflow environment is up to date.
- **Parallel Execution**: Tasks for different stocks run in parallel, improving efficiency. If you need to enforce sequential execution, additional dependencies need to be added, but this is not recommended unless necessary.
- **Resource Management**: Ensure your Airflow workers have sufficient resources (CPU, memory) to handle the parallel tasks. Adjust `max_active_runs` and `concurrency` settings as needed.
- **Error Handling**: The DAG includes basic error handling. For more robust error management, consider implementing retries, alerts, and more detailed exception handling.
- **Security Considerations**: Avoid hardcoding sensitive information in the DAG code. Use Airflow's Variables and Connections features to securely store credentials.
- **Testing**: Before deploying to production, test the DAG in a development environment. Use sample configurations to ensure all tasks execute as expected.

## Summary

By following these instructions and using the updated code, you should be able to set up an Airflow DAG that accepts parameters via API calls and dynamically processes stock predictions accordingly. The code adheres to Airflow's best practices and is designed for scalability and maintainability.
