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
from database_functions import create_engine_from_url, fetch_dataframe,  upsert_df
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
from aws_functions import s3_upload_file
from datetime import datetime
import json
import pickle
import os
import tempfile

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %%
def check_stationarity(df, column='log_return_1', alpha=0.05):
    logging.info("Checking stationarity of the time series...")
    result = adfuller(df[column].dropna())
    p_value = result[1]
    logging.info(f"ADF Statistic: {result[0]}, p-value: {p_value}")
    return p_value < alpha

def preprocess_data(df):
    logging.info("Starting data preprocessing with stationarity check...")
    df = df.copy()
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['close'], inplace=True)
    df.fillna(method='ffill', inplace=True)

    # Calculate log return
    df['log_return_30'] = np.log(df['close'].shift(-30) / df['close'])
    df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(subset=['log_return_30'], inplace=True)

    # Perform differencing if the series is non-stationary
    if not check_stationarity(df, column='log_return_1'):
        logging.info("Time series is non-stationary, applying differencing...")
        df['log_return_1'] = df['log_return_1'].diff().dropna()

    logging.info("Data preprocessing completed.")
    return df

# Feature Engineering
def engineer_additional_features(df):
    logging.info("Starting feature engineering...")

    # Shift period to align features with the target variable
    shift_period = 30

    # Shift features forward by shift_period to ensure only past data is used
    feature_columns = df.columns.difference(['log_return_30'])
    df[feature_columns] = df[feature_columns].shift(shift_period)

    # Drop original technical indicators to prevent data leakage
    technical_indicators = ['macd', 'macd_signal', 'macd_hist', 'rsi', 'upper_band', 'lower_band', 'adx', 'implied_volatility', 'other_indicator']
    df.drop(columns=[col for col in technical_indicators if col in df.columns], inplace=True)

    # Recalculate technical indicators using shifted data
    # Shifted close, high, low prices
    df['close_shifted'] = df['close'].shift(shift_period)
    df['high_shifted'] = df['high'].shift(shift_period)
    df['low_shifted'] = df['low'].shift(shift_period)
    df['log_return_1_shifted'] = df['log_return_1'].shift(shift_period)

    # Recalculate SMA and EMA using shifted close prices
    df['sma_50'] = df['close_shifted'].rolling(window=50).mean()
    df['sma_100'] = df['close_shifted'].rolling(window=100).mean()
    df['ema_50'] = df['close_shifted'].ewm(span=50, adjust=False).mean()

    # Momentum indicators using shifted close prices
    df['momentum_30'] = df['close_shifted'].shift(30) / df['close_shifted'].shift(60) - 1
    df['momentum_60'] = df['close_shifted'].shift(60) / df['close_shifted'].shift(90) - 1

    # Recalculate MACD
    df = calculate_macd(df, price_column='close_shifted')

    # Recalculate ADX
    df = calculate_adx(df, high_column='high_shifted', low_column='low_shifted', close_column='close_shifted')

    # Recalculate RSI using shifted close prices
    delta = df['close_shifted'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / roll_down
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    # Recalculate Bollinger Bands using shifted close prices
    df['middle_band'] = df['close_shifted'].rolling(window=20).mean()
    df['std_dev'] = df['close_shifted'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)
    df.drop(columns=['middle_band', 'std_dev'], inplace=True)

    # Adjusted rolling calculations
    df['rolling_volatility_60'] = df['log_return_1_shifted'].rolling(window=60).std()
    df['corr_sp500_60'] = df['log_return_1_shifted'].rolling(window=60).corr(df['sp500_return'].shift(1 + shift_period))
    df['corr_nasdaq_60'] = df['log_return_1_shifted'].rolling(window=60).corr(df['nasdaq_return'].shift(1 + shift_period))

    # Stochastic Oscillator with shifted data
    low_14 = df['low_shifted'].rolling(window=14).min()
    high_14 = df['high_shifted'].rolling(window=14).max()
    df['%K'] = (df['close_shifted'] - low_14) / (high_14 - low_14) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()

    # Commodity Channel Index with shifted data
    typical_price = (df['high_shifted'] + df['low_shifted'] + df['close_shifted']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad)

    # Use current date features shifted appropriately
    df['day_of_week'] = df.index.shift(shift_period, freq='D').dayofweek
    df['month'] = df.index.shift(shift_period, freq='D').month

    # Drop shifted columns if not needed
    df.drop(columns=['close_shifted', 'high_shifted', 'low_shifted', 'log_return_1_shifted'], inplace=True)

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
def select_features(df):
    logging.info("Starting feature selection...")
    features = [
        'volume',
        'sector_performance',  # Retained original variable
        'sector_performance_lag_1', 'sector_performance_lag_7', 'sector_performance_lag_30',
        'sp500_return',  # Retained original variable
        'sp500_return_lag_1', 'sp500_return_lag_7', 'sp500_return_lag_30',
        'nasdaq_return',  # Retained original variable
        'nasdaq_return_lag_1', 'nasdaq_return_lag_7', 'nasdaq_return_lag_30',
        'sentiment_score',  # Retained original variable
        'sentiment_score_lag_1',
        'gdp_growth',  # Retained original variable
        'gdp_growth_lag_1', 'gdp_growth_lag_7', 'gdp_growth_lag_30',
        'inflation_rate',  # Retained original variable
        'inflation_rate_lag_1', 'inflation_rate_lag_7', 'inflation_rate_lag_30',
        'unemployment_rate',  # Retained original variable
        'unemployment_rate_lag_1', 'unemployment_rate_lag_7', 'unemployment_rate_lag_30',
        'market_capitalization', 'pe_ratio', 'dividend_yield', 'beta', 'put_call_ratio',
        'macd_hist', 'adx', 'implied_volatility', 'macd', 'rsi', 'upper_band', 'lower_band',
        'macd_signal',
        'rolling_volatility_60',
        'corr_sp500_60', 'corr_nasdaq_60',
        'sma_50', 'sma_100', 'ema_50',
        '%K', '%D', 'cci',
        'momentum_30', 'momentum_60'
    ]
    features = [feature for feature in features if feature in df.columns]
    selected_features = df[features].copy()

    # Keep the target variable separate
    target = df['log_return_30']

    # Ensure there are no NaNs
    data = pd.concat([selected_features, target], axis=1).dropna()
    selected_features = data[features]
    target = data['log_return_30']

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
    fig.show()

    logging.info("Feature importance analysis completed.")
    return importance_df

# Time Series Analysis with Seasonality Support
def evaluate_sarimax(X, y, train_index, test_index, order, seasonal_order):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    try:
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, exog=X_train, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        y_pred = model_fit.forecast(steps=len(y_test), exog=X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return order, seasonal_order, rmse
    except Exception as e:
        logging.warning(f"Error in SARIMAX evaluation for order {order} and seasonal_order {seasonal_order}: {str(e)}")
        return order, seasonal_order, float('inf')

def time_series_cv(X, y, p_values, d_values, q_values, seasonal_periods):
    logging.info("Starting time series cross-validation and hyperparameter tuning...")
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
                        # Using a standard seasonal order format (p,d,q,seasonal_period)
                        seasonal_order = (p, d, q, seasonal_period)
                        order_temp, seasonal_temp, score = evaluate_sarimax(X, y, train_index, test_index, (p, d, q), seasonal_order)
                        orders.append((order_temp, seasonal_temp))
                        scores.append(score)
                    avg_score = np.mean(scores)
                    if avg_score < best_score:
                        best_score = avg_score
                        best_order = (p, d, q)
                        best_seasonal_order = seasonal_temp
                        logging.info(f"New best order: {best_order} with seasonal order: {best_seasonal_order} and RMSE: {best_score}")
    logging.info("Time series cross-validation and hyperparameter tuning completed.")
    logging.info(f"Best SARIMAX order: {best_order} with seasonal order: {best_seasonal_order}")
    return best_order, best_seasonal_order


# Train Final Model
def train_final_model(X_train, y_train, order):
    logging.info("Training final model...")
    model = SARIMAX(y_train, order=order, exog=X_train, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    logging.info("Final model training completed.")
    return model_fit


# Convert Log Returns Back to Price Predictions
def convert_predictions(y_test, y_pred, original_close_prices):
    logging.info("Converting log returns to price predictions...")
    y_test = y_test.copy()
    y_pred = pd.Series(y_pred, index=y_test.index)

    # Use the close price from 30 days ago (shifted data)
    shifted_close = original_close_prices.shift(30).loc[y_test.index]

    # Compute the predicted and actual close prices using the log returns
    predicted_close = shifted_close * np.exp(y_pred)
    actual_close = shifted_close * np.exp(y_test)

    logging.info("Conversion completed.")
    return actual_close, predicted_close


# Updated evaluate_model to include conversion and confidence intervals
def evaluate_model(model_fit, X_test, y_test, original_close_prices):
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


# Updated plotting function to include intervals
def plot_actual_vs_predicted_with_intervals(actual_close, predicted_close, confidence_intervals):
    logging.info("Plotting actual vs predicted prices with confidence intervals...")
    fig = go.Figure()

    # Plot actual and predicted values
    fig.add_trace(go.Scatter(x=actual_close.index, y=actual_close, mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=predicted_close.index, y=predicted_close, mode='lines', name='Predicted Price'))

    # Plot confidence intervals
    fig.add_trace(go.Scatter(
        x=confidence_intervals.index,
        y=confidence_intervals.iloc[:, 0],
        fill=None, mode='lines', line_color='lightgrey', name='Lower Bound'
    ))
    fig.add_trace(go.Scatter(
        x=confidence_intervals.index,
        y=confidence_intervals.iloc[:, 1],
        fill='tonexty', mode='lines', line_color='lightgrey', name='Upper Bound'
    ))

    fig.update_layout(title='Actual vs. Predicted Prices with Confidence Intervals', xaxis_title='Date',
                      yaxis_title='Price')
    fig.show()

def predict_future_price(original_data, final_model, scaler, feature_list):
    logging.info("Predicting future stock price...")

    try:
        # Get the last date in the original data
        last_date = original_data['date'].max()

        # Create a new dataframe for the next 30 days
        future_date = last_date + pd.Timedelta(days=30)
        future_df = pd.DataFrame({'date': [future_date]})

        # Merge with original data to get the latest available features
        combined_df = pd.concat([original_data, future_df]).reset_index(drop=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df.sort_values('date', inplace=True)

        # Preprocess the data
        logging.info("Starting data preprocessing...")
        processed_df = preprocess_data(combined_df)

        # Engineer features
        logging.info("Starting feature engineering...")
        engineered_df = engineer_additional_features(processed_df)

        # Select features
        logging.info("Starting feature selection...")
        X_future, _, _ = select_features(engineered_df)

        # Ensure we have the correct features
        logging.info(f"Features before selection: {X_future.columns.tolist()}")
        X_future = X_future[feature_list]

        # Scale the features
        logging.info("Starting feature scaling...")
        X_future_scaled = pd.DataFrame(scaler.transform(X_future),
                                       index=X_future.index,
                                       columns=X_future.columns)

        # Get the last row for prediction and ensure it's 2D
        X_pred = X_future_scaled.iloc[-1:].values.reshape(1, -1)  # Ensuring it is 2D

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
        results_df = pd.DataFrame({
            'date': [future_date],
            'predicted_price': [predicted_price]
        })

        logging.info("Future price prediction completed.")
        return results_df

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        print(f"Error: {str(e)}")

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
    fig.add_trace(go.Scatter(x=predicted_close.index, y=predicted_close, mode='lines', name='Predicted Price (Test Set)', line=dict(dash='dash')))

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

    fig.show()
    logging.info("Full stock prediction plot completed.")

# %%
def train_SARIMAX_model(stock_symbol,start_date,end_date,s3_bucket_name, s3_folder=''):

    query = f"""
    SELECT * FROM enriched_stock_data
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
    stock_data = preprocess_data(original_data)
    processed_data = engineer_additional_features(stock_data)
    X, y, feature_list = select_features(processed_data)
    X_scaled, y, scaler = prepare_data(X, y)

    # Feature analysis overview
    importance_df = feature_importance_analysis(X_scaled, y)

    # Train-test split
    train_size = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled.iloc[:train_size], X_scaled.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Model training and evaluation
    p_values, d_values, q_values = range(0, 2), range(0, 1), range(0, 2)
    seasonal_periods = [7,30]
    best_order, best_seasonal_order = time_series_cv(X_train, y_train, p_values, d_values, q_values, seasonal_periods)
    final_model = train_final_model(X_train, y_train, best_order)

    # Use the cross-platform method to handle temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file_name = f'sarimax_stock_prediction_model_{stock_symbol}.pkl'
        temp_model_location = os.path.join(tmpdir, model_file_name)

        # Save the model as a pickle in the temporary directory
        with open(temp_model_location, 'wb') as f:
            pickle.dump({
                'model': final_model,
                'scaler': scaler,
                'feature_list': feature_list,
                'best_order': best_order,
                'best_seasonal_order': best_seasonal_order
            }, f)

        logging.info(f"Model training completed and saved temporarily at {temp_model_location}")

        # Upload the pickle file to S3
        s3_upload_file(temp_model_location, s3_bucket_name, model_file_name, folder_name=s3_folder)
        logging.info(f"Model saved to S3 at bucket: {s3_bucket_name}, folder: {s3_folder}")

    # Evaluate model
    original_close_prices = original_data.set_index('date')['close']
    actual_close, predicted_close, confidence_intervals, mae, rmse, mape = evaluate_model(final_model, X_test, y_test,
                                                                                          original_close_prices)
    # Predict future price
    future_price = predict_future_price(original_data, final_model, scaler, feature_list)

    # Plotting the actual and predicted prices
    test_start_date = X_test.index.min()
    plot_full_stock_prediction(actual_close, predicted_close, future_price, test_start_date)

    if future_price is not None:

        # Prepare data for logging
        log_data = pd.DataFrame({
            'symbol': [stock_symbol],
            'prediction_date': [datetime.now().date()],
            'prediction_explanation': ['Based on SARIMAX model with feature engineering'],
            'prediction_rmse': [str(rmse)],
            'prediction_mae': [str(mae)],
            'prediction_mape': [str(mape)],
            'prediction_confidence_score': [str(1 / (1 + rmse))],  # Simple confidence score
            'feature_importance': [json.dumps(importance_df.to_dict())],
            'model_parameters': [json.dumps({'order': best_order, 'seasonal_order': best_seasonal_order})],
            'predicted_amount': [future_price['predicted_price'].iloc[0]],
            'last_known_price': [original_data['close'].iloc[-1]],
            'model_location': [temp_model_location],
            'date_created': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })

        # Upsert data to the database
        upsert_df(log_data, 'trained_models', 'symbol', engine)
        logging.info("Model saved and data logged to the database.")
    else:
        logging.error("Failed to generate prediction. No data logged to the database.")

#%%
train_SARIMAX_model(stock_symbol="JNJ",
                    start_date="2021-01-01",
                    end_date="2024-10-01",
                    s3_bucket_name="trained-models-stock-prediction",
                    s3_folder="sarimax")