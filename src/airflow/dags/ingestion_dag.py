from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.hooks.postgres_hook import PostgresHook
from datetime import datetime, timedelta
import boto3
from datetime import datetime, timedelta
import time

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Set yesterday's date dynamically
yesterday = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

# Stock tickers and the corresponding default date (yesterday) for Lambda invocations
stock_inputs = [
    {'ticker': 'AAPL', 'date': yesterday},
    {'ticker': 'MSFT', 'date': yesterday},
    {'ticker': 'GOOGL', 'date': yesterday},
    {'ticker': 'AMZN', 'date': yesterday},
    {'ticker': 'TSLA', 'date': yesterday},
    {'ticker': 'META', 'date': yesterday},
    {'ticker': 'JNJ', 'date': yesterday},
    {'ticker': 'PEP', 'date': yesterday},
    {'ticker': 'KO', 'date': yesterday},
    {'ticker': 'WMT', 'date': yesterday}
]


# Function to invoke the Lambda function with stock ticker and date as input
def invoke_lambda_function(stock_ticker, date, **kwargs):
    client = boto3.client('lambda')

    payload = {
        'ticker': stock_ticker,
        'date': date
    }

    response = client.invoke(
        FunctionName='your-lambda-function-name',
        InvocationType='RequestResponse',  # synchronous invocation
        Payload=bytes(str(payload), encoding='utf-8')
    )

    response_payload = response['Payload'].read()
    print(f'Lambda response for {stock_ticker}: {response_payload}')

    # Simulate storing the response for later monitoring
    return response_payload


# Function to monitor the completion of all Lambda invocations and query the PostgreSQL database
def monitor_lambdas_completion(**kwargs):
    # Pull the XCom data for all Lambda invocations
    stock_tickers = kwargs['ti'].xcom_pull(task_ids=[f'invoke_lambda_{stock["ticker"]}' for stock in stock_inputs])

    # Check the responses for each stock
    all_completed = True
    for ticker, ticker_response in zip([stock['ticker'] for stock in stock_inputs], stock_tickers):
        if not ticker_response:
            print(f"Lambda invocation for {ticker} not completed yet.")
            all_completed = False
        else:
            print(f"Lambda invocation for {ticker} completed successfully.")

    # If all Lambda invocations were successful, check the PostgreSQL table for each stock ticker
    if all_completed:
        pg_hook = PostgresHook(postgres_conn_id='your_postgres_conn_id')  # Ensure connection is set in Airflow UI
        for stock in stock_inputs:
            sql = """
            SELECT * FROM enriched_stock_data
            WHERE stock = %s
            ORDER BY date DESC
            LIMIT 1;  # Get the most recent record
            """
            result = pg_hook.get_first(sql, parameters=(stock['ticker'],))
            if result:
                print(f"Most recent record for {stock['ticker']}: {result}")
            else:
                print(f"No data found for {stock['ticker']} in enriched_stock_data.")
    else:
        raise ValueError("Not all Lambda invocations completed successfully.")


# Define the DAG
with DAG(
        'lambda_batch_invoke_dag',
        default_args=default_args,
        description='Invoke Lambda for 10 stock tickers, query PostgreSQL, and monitor completion',
        schedule_interval='0 1 * * *',  # Every day at 1 AM CST
        catchup=False,
) as dag:
    # Task Group to invoke 10 Lambdas in parallel
    with TaskGroup('invoke_lambdas') as invoke_lambdas_group:
        for stock in stock_inputs:
            # Create a PythonOperator for each Lambda invocation
            invoke_lambda_task = PythonOperator(
                task_id=f'invoke_lambda_{stock["ticker"]}',
                python_callable=invoke_lambda_function,
                op_kwargs={'stock_ticker': stock['ticker'], 'date': stock['date']},
                provide_context=True
            )

    # Task to monitor completion of all Lambda invocations and query PostgreSQL
    monitor_completion_task = PythonOperator(
        task_id='monitor_lambdas_completion',
        python_callable=monitor_lambas_completion,
        provide_context=True
    )

    # Set task dependencies
    invoke_lambdas_group >> monitor_completion_task