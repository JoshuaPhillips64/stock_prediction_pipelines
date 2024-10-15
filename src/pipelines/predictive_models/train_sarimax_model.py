# %%
# Import necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
import logging
import warnings
from database_functions import create_engine_from_url, fetch_dataframe, upsert_df
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, S3_BUCKET_NAME
from aws_functions import s3_upload_file
from datetime import datetime, timedelta, date
import json
import pickle
import os
import tempfile
import requests

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# %%

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def lambda_handler(event, context):
    """
    AWS Lambda handler function to initiate SARIMAX model training and prediction.
    """
    try:
        body = json.loads(event['body'])
        model_key = body.get('model_key')
        stock_symbol = body.get('stock_symbol')
        input_date = body.get('input_date')
        hyperparameter_tuning = body.get('hyperparameter_tuning', 'MEDIUM')
        feature_set = body.get('feature_set', 'basic')
        lookback_period = body.get('lookback_period', 720)
        prediction_horizon = body.get('prediction_horizon', 30)
    except Exception as e:
        logging.error(f"Error parsing input parameters: {e}")
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid input parameters')
        }

    try:
        # Call the train_SARIMAX_model function
        result = train_SARIMAX_model(
            model_key=model_key,
            stock_symbol=stock_symbol,
            input_date=input_date,
            hyperparameter_tuning=hyperparameter_tuning,
            feature_set=feature_set,
            lookback_period=lookback_period,
            prediction_horizon=prediction_horizon
        )

        if result:
            return {
                'statusCode': 200,
                'body': json.dumps(result, default=json_serial)
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps('Prediction failed.')
            }

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps('An unexpected error occurred during processing.')}


def check_stationarity(df, column='log_return_1', alpha=0.05):
    logging.info("Checking stationarity of the time series...")
    result = adfuller(df[column].dropna())
    p_value = result[1]
    logging.info(f"ADF Statistic: {result[0]}, p-value: {p_value}")
    return p_value < alpha


def preprocess_data(df, prediction_horizon):
    logging.info("Starting data preprocessing with stationarity check...")
    df = df.copy()
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['close'], inplace=True)
    df.fillna(method='ffill', inplace=True)

    # Calculate log return over prediction_horizon days
    df['log_return_future'] = np.log(df['close'].shift(-prediction_horizon) / df['close'])
    df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(subset=['log_return_future'], inplace=True)

    # Perform differencing if the series is non-stationary
    if not check_stationarity(df, column='log_return_1'):
        logging.info("Time series is non-stationary, applying differencing...")
        df['log_return_1'] = df['log_return_1'].diff().dropna()

    logging.info("Data preprocessing completed.")
    return df


# Feature Engineering
def engineer_additional_features(df):
    logging.info("Starting feature engineering...")

    # Calculate technical indicators
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_100'] = df['close'].rolling(window=100).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # Momentum indicators
    df['momentum_30'] = df['close'].pct_change(periods=30)
    df['momentum_60'] = df['close'].pct_change(periods=60)

    # Calculate MACD
    df = calculate_macd(df)

    # Calculate ADX
    df = calculate_adx(df)

    # Calculate RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / roll_down
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    # Bollinger Bands
    df['middle_band'] = df['close'].rolling(window=20).mean()
    df['std_dev'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)
    df.drop(columns=['middle_band', 'std_dev'], inplace=True)

    # Rolling calculations
    df['rolling_volatility_60'] = df['log_return_1'].rolling(window=60).std()
    df['corr_sp500_60'] = df['log_return_1'].rolling(window=60).corr(df['sp500_return'].shift(1))
    df['corr_nasdaq_60'] = df['log_return_1'].rolling(window=60).corr(df['nasdaq_return'].shift(1))

    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['%K'] = (df['close'] - low_14) / (high_14 - low_14) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()

    # Commodity Channel Index
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad)

    df.dropna(inplace=True)
    logging.info("Feature engineering completed.")
    return df


def calculate_macd(df, price_column='close'):
    logging.info("Calculating MACD and related features...")
    df['ema_12'] = df[price_column].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df[price_column].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df.drop(columns=['ema_12', 'ema_26'], inplace=True)
    logging.info("MACD calculation completed.")
    return df


def calculate_adx(df, high_column='high', low_column='low', close_column='close', period=14):
    logging.info("Calculating ADX...")

    # Calculate True Range (TR)
    df['tr'] = df[[high_column, close_column]].max(axis=1) - df[[low_column, close_column]].min(axis=1)

    # Calculate directional movements
    df['up_move'] = df[high_column] - df[high_column].shift(1)
    df['down_move'] = df[low_column].shift(1) - df[low_column]

    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    # Calculate Smoothed True Range, +DM, -DM
    atr = df['tr'].rolling(window=period).mean()
    plus_di = 100 * (df['plus_dm'].rolling(window=period).mean() / atr)
    minus_di = 100 * (df['minus_dm'].rolling(window=period).mean() / atr)

    # Calculate DX and ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(window=period).mean()

    # Drop intermediate columns
    df.drop(columns=['tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm'], inplace=True)

    logging.info("ADX calculation completed.")
    return df


# Feature Selection
def select_features(df, feature_set='advanced'):
    logging.info("Starting feature selection...")

    # Define feature categories
    basic_features = [
        'volume',
        'open',
        'high',
        'low',
        'close',
        'sp500_return',
        'nasdaq_return',
        'gdp_growth',
        'inflation_rate',
        'unemployment_rate',
        'gdp_growth_lag_1', 'gdp_growth_lag_7', 'gdp_growth_lag_30',
        'inflation_rate_lag_1', 'inflation_rate_lag_7',
        'inflation_rate_lag_30',
        'unemployment_rate_lag_1', 'unemployment_rate_lag_7', 'unemployment_rate_lag_30',
        'macd_hist', 'adx', 'macd', 'rsi', 'upper_band', 'lower_band',
        'macd_signal',
        'rolling_volatility_60',
        'corr_sp500_60', 'corr_nasdaq_60',
        'sma_50', 'sma_100', 'ema_50',
        '%K', '%D', 'cci',
        'momentum_30', 'momentum_60'
    ]

    advanced_features = [
        'sentiment_score',
        'implied_volatility',
        'market_capitalization',
        'pe_ratio',
        'dividend_yield',
        'beta',
        'put_call_ratio',
        'sector_performance',
        'sector_performance_lag_1', 'sector_performance_lag_7', 'sector_performance_lag_30',
        'sp500_return_lag_1', 'sp500_return_lag_7', 'sp500_return_lag_30',
        'nasdaq_return_lag_1', 'nasdaq_return_lag_7', 'nasdaq_return_lag_30',
        'sentiment_score_lag_1'
    ]

    if feature_set == 'basic':
        features = basic_features
    elif feature_set == 'advanced':
        features = basic_features + advanced_features
    else:
        raise ValueError(f"Invalid feature_set: {feature_set}")

    # Only include features that are in df columns
    features = [feature for feature in features if feature in df.columns]
    selected_features = df[features].copy()

    # Keep the target variable separate
    target = df['log_return_future']

    # Ensure there are no NaNs
    data = pd.concat([selected_features, target], axis=1).dropna()
    selected_features = data[features]
    target = data['log_return_future']

    logging.info("Feature selection completed.")
    return selected_features, target, features


# Prepare Data
def prepare_data(X, y):
    # Convert any categorical features
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes

    # Scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    return X_scaled, y, scaler


# Feature Importance Analysis
def feature_importance_analysis(X, y):
    logging.info("Starting feature importance analysis...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance_df = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
    importance_df = importance_df.sort_values('importance', ascending=False)

    # Plot the feature importance using Plotly
    fig = go.Figure([go.Bar(x=importance_df['feature'], y=importance_df['importance'])])
    fig.update_layout(title='Feature Importance', xaxis_title='Features', yaxis_title='Importance', xaxis_tickangle=-45)
    #fig.show()

    logging.info("Feature importance analysis completed.")
    return importance_df


# Time Series Analysis with Seasonality Support
def evaluate_sarimax(X, y, train_index, test_index, order, seasonal_order):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    try:
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, exog=X_train, enforce_stationarity=False,
                        enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        y_pred = model_fit.forecast(steps=len(y_test), exog=X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return order, seasonal_order, rmse
    except Exception as e:
        logging.warning(f"Error in SARIMAX evaluation for order {order} and seasonal_order {seasonal_order}: {str(e)}")
        return order, seasonal_order, float('inf')


def time_series_cv(X, y, hyperparameter_tuning, seasonal_periods):
    logging.info("Starting time series cross-validation and hyperparameter tuning...")

    if hyperparameter_tuning == 'LOW':
        p_values = range(0, 1)
        d_values = range(0, 1)
        q_values = range(0, 1)
    elif hyperparameter_tuning == 'MEDIUM':
        p_values = range(0, 2)
        d_values = range(0, 2)
        q_values = range(0, 2)
    elif hyperparameter_tuning == 'HIGH':
        p_values = range(0, 3)
        d_values = range(0, 3)
        q_values = range(0, 3)
    else:
        raise ValueError(f"Invalid hyperparameter_tuning: {hyperparameter_tuning}")

    tscv = TimeSeriesSplit(n_splits=3)
    best_score = float('inf')
    best_order = None
    best_seasonal_order = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for seasonal_period in seasonal_periods:
                    orders = []
                    scores = []
                    for train_index, test_index in tscv.split(X):
                        seasonal_order = (p, d, q, seasonal_period)
                        order_temp, seasonal_temp, score = evaluate_sarimax(X, y, train_index, test_index, (p, d, q),
                                                                            seasonal_order)
                        orders.append((order_temp, seasonal_temp))
                        scores.append(score)
                    avg_score = np.mean(scores)
                    if avg_score < best_score:
                        best_score = avg_score
                        best_order = (p, d, q)
                        best_seasonal_order = seasonal_temp
                        logging.info(
                            f"New best order: {best_order} with seasonal order: {best_seasonal_order} and RMSE: {best_score}")
    logging.info("Time series cross-validation and hyperparameter tuning completed.")
    logging.info(f"Best SARIMAX order: {best_order} with seasonal order: {best_seasonal_order}")
    return best_order, best_seasonal_order


# Train Final Model
def train_final_model(X_train, y_train, order, seasonal_order):
    logging.info("Training final model...")
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, exog=X_train, enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    logging.info("Final model training completed.")
    return model_fit


# Convert Log Returns Back to Price Predictions
def convert_predictions(y_test, y_pred, original_close_prices):
    logging.info("Converting log returns to price predictions...")
    y_test = y_test.copy()
    y_pred = pd.Series(y_pred, index=y_test.index)

    # Get the close prices at prediction times
    shifted_close = original_close_prices.loc[y_test.index]

    # Compute the predicted and actual close prices using the log returns
    predicted_close = shifted_close * np.exp(y_pred)
    actual_close = shifted_close * np.exp(y_test)

    logging.info("Conversion completed.")
    return actual_close, predicted_close


# Updated evaluate_model to include conversion and confidence intervals
def evaluate_model(model_fit, X_test, y_test, original_close_prices, prediction_horizon):
    logging.info("Evaluating model with prediction intervals...")

    # Forecast using SARIMAX and get prediction intervals
    forecast_object = model_fit.get_forecast(steps=len(y_test), exog=X_test)
    y_pred = forecast_object.predicted_mean
    confidence_intervals = forecast_object.conf_int()

    # Align indices
    y_pred.index = y_test.index
    confidence_intervals.index = y_test.index

    # Convert log returns back to stock prices
    actual_close, predicted_close = convert_predictions(y_test, y_pred, original_close_prices)

    # Calculate metrics based on log returns
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f'Mean Absolute Error: {mae:.6f}')
    print(f'Root Mean Squared Error: {rmse:.6f}')
    print(f'Mean Absolute Percentage Error: {mape:.2f}%')

    logging.info("Model evaluation with prediction intervals and price conversion completed.")
    return actual_close, predicted_close, confidence_intervals, mae, rmse, mape


def predict_future_price(original_data, final_model, scaler, feature_list, prediction_horizon, feature_set):
    logging.info("Predicting future stock price...")

    try:
        # Get the last date in the original data
        last_date = original_data['date'].max()

        # Prepare the data for prediction
        processed_df = preprocess_data(original_data.copy(), prediction_horizon)
        engineered_df = engineer_additional_features(processed_df)

        # Select features
        logging.info("Starting feature selection...")
        X_future, _, _ = select_features(engineered_df, feature_set=feature_set)

        # Ensure we have the correct features
        logging.info(f"Features before selection: {X_future.columns.tolist()}")
        X_future = X_future[feature_list]

        # Scale the features
        logging.info("Starting feature scaling...")
        X_future_scaled = pd.DataFrame(scaler.transform(X_future),
                                       index=X_future.index,
                                       columns=X_future.columns)

        # Get the last row for prediction
        X_pred = X_future_scaled.iloc[[-1]]

        logging.info(f"Exogenous variables shape for prediction: {X_pred.shape}")

        # Make the prediction using SARIMAX
        y_pred_future = final_model.forecast(steps=1, exog=X_pred)

        # Extract the predicted log return (ensure it's a scalar or a proper float)
        predicted_log_return = y_pred_future.iloc[0]

        logging.info(f"Predicted log return: {predicted_log_return}")

        # Convert log return to price
        last_known_price = original_data['close'].values[-1]
        predicted_price = last_known_price * np.exp(predicted_log_return)

        logging.info(f"Predicted future price: {predicted_price}")

        # Create a dataframe with the result
        future_date = last_date + pd.Timedelta(days=prediction_horizon)
        results_df = pd.DataFrame({
            'date': [future_date],
            'predicted_price': [predicted_price]
        })

        logging.info("Future price prediction completed.")
        return results_df

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        print(f"Error: {str(e)}")
        return None


# Visualize Original Data, Test Projections, and Future Price
def plot_full_stock_prediction(actual_close, predicted_close, future_price, test_start_date):
    logging.info("Plotting full stock prediction...")

    # Convert test_start_date and future_price['date'] to datetime objects (if necessary)
    test_start_date = pd.to_datetime(test_start_date)
    future_date = pd.to_datetime(future_price['date'].iloc[0])

    # Create the figure
    fig = go.Figure()

    # Plot the actual stock prices
    fig.add_trace(go.Scatter(x=actual_close.index, y=actual_close, mode='lines', name='Actual Price'))

    # Plot the predicted prices on the test set
    fig.add_trace(
        go.Scatter(x=predicted_close.index, y=predicted_close, mode='lines', name='Predicted Price (Test Set)',
                   line=dict(dash='dash')))

    # Plot the predicted future price
    fig.add_trace(go.Scatter(x=future_price['date'], y=future_price['predicted_price'], mode='markers+lines',
                             name='Predicted Future Price', marker=dict(symbol='star', size=10, color='blue')))

    # Manually add vertical lines by plotting as scatter points
    fig.add_trace(go.Scatter(
        x=[test_start_date, test_start_date],
        y=[min(actual_close.min(), predicted_close.min()), max(actual_close.max(), predicted_close.max())],
        mode='lines',
        name='Test Set Start',
        line=dict(color='orange', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=[future_date, future_date],
        y=[min(actual_close.min(), predicted_close.min()), max(actual_close.max(), predicted_close.max())],
        mode='lines',
        name='Future Prediction',
        line=dict(color='green', dash='dash')
    ))

    # Update layout for the plot
    fig.update_layout(
        title='Stock Price Prediction: Actual, Test Projections, and Future Price',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    #fig.show()
    logging.info("Full stock prediction plot completed.")


# %%
def train_SARIMAX_model(model_key, stock_symbol, input_date, hyperparameter_tuning='MEDIUM', feature_set='advanced',
                        lookback_period=720, prediction_horizon=30):
    """
    Function to train SARIMAX model, evaluate it, make predictions, and log the results.
    Returns True if successful, False otherwise.
    """
    try:
        # Compute start_date and end_date
        input_date_dt = datetime.strptime(input_date, '%Y-%m-%d')
        prediction_date = input_date_dt + timedelta(days=prediction_horizon)
        start_date_dt = input_date_dt - timedelta(days=lookback_period)
        start_date = start_date_dt.strftime('%Y-%m-%d')
        end_date = input_date

        if feature_set == 'advanced':
            stock_table = 'enriched_stock_data'
        else:
            stock_table = 'basic_stock_data'

        query = f"""
        SELECT * FROM {stock_table}
        WHERE symbol = '{stock_symbol}'
        and date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
        """

        # Database and stock selection
        db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine_from_url(db_url)

        # Fetch data
        original_data = fetch_dataframe(engine, query)
        original_data['date'] = pd.to_datetime(original_data['date'])

        # Data preprocessing and feature engineering
        stock_data = preprocess_data(original_data, prediction_horizon)
        processed_data = engineer_additional_features(stock_data)
        X, y, feature_list = select_features(processed_data, feature_set=feature_set)
        X_scaled, y, scaler = prepare_data(X, y)

        # Feature analysis overview
        importance_df = feature_importance_analysis(X_scaled, y)

        # Train-test split
        train_size = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled.iloc[:train_size], X_scaled.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # Model training and evaluation
        seasonal_periods = [7, prediction_horizon]
        best_order, best_seasonal_order = time_series_cv(X_train, y_train, hyperparameter_tuning, seasonal_periods)
        final_model = train_final_model(X_train, y_train, best_order, best_seasonal_order)

        # Save the model with model_key
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file_name = f'sarimax_model_{model_key}.pkl'
            temp_model_location = os.path.join(tmpdir, model_file_name)

            s3_bucket_name = "trained-models-stock-prediction"
            s3_folder = "sarimax"

            # Save the model as a pickle in the temporary directory
            with open(temp_model_location, 'wb') as f:
                pickle.dump({
                    'model': final_model,
                    'scaler': scaler,
                    'feature_list': feature_list,
                    'best_order': best_order,
                    'best_seasonal_order': best_seasonal_order,
                    'hyperparameter_tuning': hyperparameter_tuning,
                    'feature_set': feature_set,
                    'prediction_horizon': prediction_horizon,
                    'input_date': input_date
                }, f)

            logging.info(f"Model training completed and saved temporarily at {temp_model_location}")

            # Upload the pickle file to S3
            s3_upload_file(temp_model_location, s3_bucket_name, model_file_name, folder_name=s3_folder)
            logging.info(f"Model saved to S3 at bucket: {s3_bucket_name}, folder: {s3_folder}")

        # Evaluate model
        original_close_prices = original_data.set_index('date')['close']
        evaluation_results = evaluate_model(
            final_model, X_test, y_test, original_close_prices, prediction_horizon)

        if evaluation_results is None:
            logging.error("Model evaluation failed.")
            return False

        actual_close, predicted_close, confidence_intervals, mae, rmse, mape = evaluation_results

        # Predict future price
        future_price = predict_future_price(original_data, final_model, scaler, feature_list, prediction_horizon,
                                            feature_set)

        if future_price is None:
            logging.error("Future price prediction failed.")
            return False

        # Build predictions JSON dictionary
        predictions_json = {
            str(date.date()): {
                'actual_price': actual_close.get(date, None),  # Handle missing dates safely
                'predicted_price': predicted_close.get(date, None)
            }
            for date in actual_close.index.union(predicted_close.index)  # Ensure all dates are covered
        }

        # Add the future price prediction
        future_date_str = future_price['date'].iloc[0].strftime('%Y-%m-%d')
        predictions_json[future_date_str] = {
            'actual_price': None,  # No actual price for future dates
            'predicted_price': future_price['predicted_price'].iloc[0]
        }

        # Convert to JSON
        predictions_json_str = json.dumps(predictions_json)

        # Plotting the actual and predicted prices
        test_start_date = X_test.index.min()
        plot_full_stock_prediction(actual_close, predicted_close, future_price, test_start_date)

        # Prepare data for logging
        log_data = {
            'model_key': model_key,
            'symbol': stock_symbol,
            'prediction_date': prediction_date,
            'prediction_explanation': 'Regression Prediction Based on SARIMAX model with feature engineering',
            'prediction_rmse': str(rmse),
            'prediction_mae': str(mae),
            'prediction_mape': str(mape),
            'prediction_confidence_score': str(1 / (1 + rmse)),  # Simple confidence score
            'feature_importance': json.dumps(importance_df.to_dict()),
            'model_parameters': json.dumps({
                'order': best_order,
                'seasonal_order': best_seasonal_order,
                'hyperparameter_tuning': hyperparameter_tuning,
                'feature_set': feature_set,
                'prediction_horizon': prediction_horizon
            }),
            'predicted_amount': float(future_price['predicted_price'].iloc[0]),
            'last_known_price': float(original_data['close'].iloc[-1]),
            'predictions_json': predictions_json_str,
            'model_location': f's3://trained-models-stock-prediction/sarimax_model_{model_key}.pkl',
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Convert log_data to DataFrame for database insertion
        log_df = pd.DataFrame([log_data])

        # Upsert data to the database
        upsert_df(log_df, 'trained_models', 'model_key', engine,
                  json_columns=['feature_importance', 'model_parameters', 'predictions_json'])
        logging.info("Model saved and data logged to the database.")

        return log_data

    except Exception as e:
        logging.error(f"An error occurred during model training or prediction: {e}")
        return False

    except Exception as e:
        logging.error(f"An error occurred during model training or prediction: {e}")
        return False
#%% Sample implementation
def main():
    # Sample event payload
    event = {
        'body': json.dumps({
            'model_key': 'sample_model_001',
            'stock_symbol': 'PG',
            'input_date': '2024-10-01',
            'hyperparameter_tuning': 'LOW',
            'feature_set': 'advanced',
            'lookback_period': 720,
            'prediction_horizon': 30
        })
    }

    # Mock context object (can be empty or contain necessary attributes)
    context = {}

    # Invoke the lambda_handler
    response = lambda_handler(event, context)

    # Print the response
    print("Lambda Response:")
    print(json.dumps(response, indent=4))

# Run the main function
#main()