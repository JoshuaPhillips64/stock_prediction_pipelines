from airflow import DAG
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import logging

from common.helpers import (
    invoke_lambda_ingest,
    invoke_lambda_train,
    invoke_lambda_predict,
    invoke_lambda_ai_analysis,
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define task-specific timeout
TASK_TIMEOUT = timedelta(minutes=15)

dag_name = 'dynamic_api_generate_stock_prediction_dag'
description = 'Called via the airflow api with dynamic parameters to Generate stock predictions'
schedule = None  # Set to None since it will be triggered externally
start_date = days_ago(1)
catchup = False

@dag(
    dag_id=dag_name,
    default_args=default_args,
    description=description,
    schedule_interval=schedule,
    start_date=start_date,
    catchup=catchup,
    max_active_runs=1,
)
def dynamic_api_generate_stock_prediction_dag():

    @task
    def get_conf(**context):
        """Retrieve configuration from dag_run.conf"""
        conf = context['dag_run'].conf or {}
        if not conf:
            raise ValueError("No configuration provided in dag_run.conf")
        return conf

    @task
    def extract_stocks(conf):
        """Extract stock information from configuration"""
        stocks = conf.get('stocks')
        if not stocks:
            raise ValueError("No stocks provided in the configuration")
        return stocks

    @task
    def process_stock(stock_info, **context):
        """Process a single stock with provided parameters"""
        stock = stock_info.get('stock')
        params = stock_info.get('params')

        if not stock or not params:
            raise ValueError(f"Stock or parameters missing for stock info: {stock_info}")

        execution_date_str = context['ds']
        execution_date = datetime.strptime(execution_date_str, '%Y-%m-%d')

        # Ingest Data
        lookback_period = params['lookback_period']
        start_date = (execution_date - timedelta(days=lookback_period)).strftime('%Y-%m-%d')
        end_date = execution_date_str

        logger.info(f"Ingesting data for {stock}: {start_date} to {end_date}")
        invoke_lambda_ingest(
            stock_symbol=stock,
            start_date=start_date,
            end_date=end_date,
            feature_set=params['feature_set']
        )

        # Train Model
        logger.info(f"Training model for {stock} on {execution_date_str}")
        invoke_lambda_train(
            model_type=params['model_type'],
            stock_symbol=stock,
            input_date=execution_date_str,
            hyperparameter_tuning=params['hyperparameter_tuning'],
            feature_set=params['feature_set'],
            lookback_period=params['lookback_period'],
            prediction_horizon=params['prediction_horizon']
        )

        # Make Prediction
        logger.info(f"Making prediction for {stock} on {execution_date_str}")
        invoke_lambda_predict(
            model_type=params['model_type'],
            stock_symbol=stock,
            input_date=execution_date_str,
            hyperparameter_tuning=params['hyperparameter_tuning'],
            feature_set=params['feature_set'],
            lookback_period=params['lookback_period'],
            prediction_horizon=params['prediction_horizon']
        )

        # Trigger AI Analysis
        logger.info(f"Triggering AI analysis for {stock} on {execution_date_str}")
        invoke_lambda_ai_analysis(
            stock_symbol=stock,
            model_type=params['model_type'],
            input_date=execution_date_str,
            feature_set=params['feature_set'],
            hyperparameter_tuning=params['hyperparameter_tuning'],
            lookback_period=params['lookback_period'],
            prediction_horizon=params['prediction_horizon']
        )

    # Retrieve configuration
    conf = get_conf()

    # Extract stocks from configuration
    stocks = extract_stocks(conf)

    # Use Dynamic Task Mapping to process each stock
    process_stock.expand(
        stock_info=stocks,
    )

dag = dynamic_api_generate_stock_prediction_dag()