# %%
# Import necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import logging
from concurrent.futures import ProcessPoolExecutor
import warnings
from database_functions import create_engine_from_url, upsert_df, fetch_dataframe
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %%
# Database and stock selection
db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine_from_url(db_url)
stock_symbol = "PG"

query = f"""
SELECT * FROM enriched_stock_data
WHERE symbol = '{stock_symbol}'
and date >= '2023-01-01'
ORDER BY date
"""
stock_data = fetch_dataframe(engine, query)
stock_data['date'] = pd.to_datetime(stock_data['date'])


# %%
# Data Preprocessing
def preprocess_data(df):
    logging.info("Starting data preprocessing...")
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['close'], inplace=True)
    df.fillna(method='ffill', inplace=True)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(subset=['log_return'], inplace=True)
    logging.info("Data preprocessing completed.")
    return df


stock_data = preprocess_data(stock_data)


# %%
# Feature Engineering
def engineer_additional_features(df):
    logging.info("Starting feature engineering...")

    # Existing features
    macro_vars = ['inflation_rate', 'unemployment_rate', 'gdp_growth', 'sector_performance']
    for var in macro_vars:
        for lag in range(1, 4):
            df[f'{var}_lag_{lag}'] = df[var].shift(lag)

    df['rolling_volatility'] = df['log_return'].rolling(window=20).std()
    df['corr_sp500'] = df['log_return'].rolling(window=20).corr(df['sp500_return'])
    df['corr_nasdaq'] = df['log_return'].rolling(window=20).corr(df['nasdaq_return'])
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # New technical indicators
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

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


stock_data = engineer_additional_features(stock_data)


# %%
# Feature Selection
def select_features(df):
    logging.info("Starting feature selection...")
    features = [
        'log_return', 'volume', 'sector_performance', 'sp500_return', 'nasdaq_return',
        'sentiment_score', 'gdp_growth', 'inflation_rate', 'unemployment_rate',
        'market_capitalization', 'pe_ratio', 'dividend_yield', 'beta', 'put_call_ratio',
        'macd_hist', 'adx', 'implied_volatility', 'macd', 'rsi', 'upper_band', 'lower_band',
        'macd_signal', 'reported_eps', 'estimated_eps', 'eps_surprise', 'eps_surprise_percentage',
        'last_earnings_date', 'next_earnings_estimated_eps',
        'inflation_rate_lag_1', 'inflation_rate_lag_2', 'inflation_rate_lag_3',
        'unemployment_rate_lag_1', 'unemployment_rate_lag_2', 'unemployment_rate_lag_3',
        'gdp_growth_lag_1', 'gdp_growth_lag_2', 'gdp_growth_lag_3',
        'sector_performance_lag_1', 'sector_performance_lag_2', 'sector_performance_lag_3',
        'rolling_volatility', 'corr_sp500', 'corr_nasdaq', 'day_of_week', 'month',
        'sma_20', 'sma_50', 'ema_20', '%K', '%D', 'cci',
    ]
    features = [feature for feature in features if feature in df.columns]
    df.dropna(subset=features, inplace=True)
    logging.info("Feature selection completed.")
    return df, features

stock_data, feature_list = select_features(stock_data)


# %%
# Stationarity Check
def check_stationarity(series):
    logging.info("Checking stationarity...")
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[4].items():
        print('Critical Value (%s): %.3f' % (key, value))
    print("Series is " + ("non-stationary" if result[1] > 0.05 else "stationary"))


check_stationarity(stock_data['log_return'])

# %%
from datetime import datetime, date

def is_date(x):
    return isinstance(x, (datetime, date))


def prepare_data(stock_data, feature_list):
    X = stock_data[feature_list].drop(columns=['log_return'])
    y = stock_data['log_return']

    for col in X.columns:
        if X[col].dtype == 'object':
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = pd.to_datetime(X[col]).astype('int64') // 10 ** 9  # Convert to Unix timestamp
            else:
                X[col] = pd.Categorical(X[col]).codes  # Convert categorical to numeric
        elif X[col].dtype == 'datetime64[ns]':
            X[col] = X[col].astype('int64') // 10 ** 9  # Convert datetime64[ns] to Unix timestamp

    # Now all columns should be numeric
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    return X_scaled, y


# Feature Importance Analysis
def feature_importance_analysis(X, y):
    logging.info("Starting feature importance analysis...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance_df = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
    importance_df = importance_df.sort_values('importance', ascending=False)

    fig = px.bar(importance_df, x='feature', y='importance', title='Feature Importance')
    fig.show()

    logging.info("Feature importance analysis completed.")
    return importance_df


# Apply the functions
stock_data = stock_data.sort_index()  # Ensure the index is sorted
X_scaled, y = prepare_data(stock_data, feature_list)
print("Shape of X_scaled:", X_scaled.shape)
print("Columns in X_scaled:", X_scaled.columns)
print("Data types in X_scaled:")
print(X_scaled.dtypes)

importance_df = feature_importance_analysis(X_scaled, y)
print(importance_df)


# %%
# Time Series Analysis
def evaluate_sarimax(X, y, train_index, test_index, order):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    try:
        model = SARIMAX(y_train, order=order, exog=X_train, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        y_pred = model_fit.forecast(steps=len(y_test), exog=X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return order, rmse
    except Exception as e:
        logging.warning(f"Error in SARIMAX evaluation for order {order}: {str(e)}")
        return order, float('inf')

def time_series_cv(X, y, p_values, d_values, q_values):
    logging.info("Starting time series cross-validation and hyperparameter tuning...")
    tscv = TimeSeriesSplit(n_splits=5)
    best_score = float('inf')
    best_order = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                scores = []
                for train_index, test_index in tscv.split(X):
                    order, score = evaluate_sarimax(X, y, train_index, test_index, (p, d, q))
                    scores.append(score)
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_order = (p, d, q)
                    logging.info(f"New best order: {best_order} with RMSE: {best_score}")

    logging.info("Time series cross-validation and hyperparameter tuning completed.")
    return best_order

# Time Series Analysis
p_values = range(0, 3)  # Reduced range for faster execution
d_values = range(0, 2)
q_values = range(0, 3)

best_order = time_series_cv(X_scaled, y, p_values, d_values, q_values)
print(f"Best SARIMAX order: {best_order}")

# %%
# Train Final Model
def train_final_model(X_train, y_train, order):
    logging.info("Training final model...")
    model = SARIMAX(y_train, order=order, exog=X_train, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    logging.info("Final model training completed.")
    return model_fit


train_size = int(len(X_scaled) * 0.8)
X_train = X_scaled.iloc[:train_size]
X_test = X_scaled.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("X_train index:", X_train.index)
print("X_test index:", X_test.index)
print("y_train index:", y_train.index)
print("y_test index:", y_test.index)

final_model = train_final_model(X_train, y_train, best_order)

print("Model summary:")
print(final_model.summary())


# %%
# Model Evaluation
# 1. Check data alignment
print("y_test index:", y_test.index)
print("y_pred index:", y_pred.index)

# 2. Verify forecast steps
print("Length of X_test:", len(X_test))
print("Length of y_pred:", len(y_pred))

# 3. Examine index types
print("y_test index type:", type(y_test.index))
print("y_pred index type:", type(y_pred.index))

# 4. Update evaluate_model function
def evaluate_model(model_fit, X_test, y_test):
    logging.info("Evaluating model...")

    # Generate predictions
    y_pred = model_fit.forecast(steps=len(X_test), exog=X_test)

    # Ensure y_pred has the same index as y_test
    y_pred = pd.Series(y_pred, index=y_test.index)

    # Ensure y_test and y_pred have the same index
    common_index = y_test.index.intersection(y_pred.index)
    y_test = y_test.loc[common_index]
    y_pred = y_pred.loc[common_index]

    if len(y_test) == 0 or len(y_pred) == 0:
        logging.error("No common dates between y_test and y_pred")
        return None

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Directional Accuracy
    direction_actual = np.sign(y_test.diff().dropna())
    direction_pred = np.sign(y_pred.diff().dropna())
    directional_accuracy = np.mean(direction_actual == direction_pred) * 100

    print(f'Mean Absolute Error: {mae:.6f}')
    print(f'Root Mean Squared Error: {rmse:.6f}')
    print(f'Mean Absolute Percentage Error: {mape:.2f}%')
    print(f'Directional Accuracy: {directional_accuracy:.2f}%')

    logging.info("Model evaluation completed.")
    return y_pred

# 5. Re-run the evaluation
y_pred = evaluate_model(final_model, X_test, y_test)

# 6. If the issue persists, check for data leakage
print("X_train last date:", X_train.index[-1])
print("X_test first date:", X_test.index[0])
print("y_train last date:", y_train.index[-1])
print("y_test first date:", y_test.index[0])


# %%
# Residual Analysis
def residual_analysis(model_fit):
    logging.info("Performing residual analysis...")
    residuals = model_fit.resid

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
    'Residuals over Time', 'Histogram of Residuals', 'ACF of Residuals', 'PACF of Residuals'))

    # Residuals over Time
    fig.add_trace(go.Scatter(x=residuals.index, y=residuals, mode='lines', name='Residuals'), row=1, col=1)

    # Histogram of Residuals
    fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name='Residuals'), row=1, col=2)

    # ACF of Residuals
    acf_values = plot_acf(residuals, lags=40, alpha=0.05, display=False)
    fig.add_trace(
        go.Scatter(x=acf_values.lines[0].get_xdata(), y=acf_values.lines[0].get_ydata(), mode='lines', name='ACF'),
        row=2, col=1)

    # PACF of Residuals
    pacf_values = plot_pacf(residuals, lags=40, alpha=0.05, method='ols', display=False)
    fig.add_trace(
        go.Scatter(x=pacf_values.lines[0].get_xdata(), y=pacf_values.lines[0].get_ydata(), mode='lines', name='PACF'),
        row=2, col=2)

    fig.update_layout(height=900, title_text="Residual Analysis")
    fig.show()

    # QQ Plot
    fig = px.scatter(x=np.random.normal(size=len(residuals)), y=np.sort(residuals), title='QQ Plot of Residuals')
    fig.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines', name='y=x'))
    fig.show()

    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print("Ljung-Box test results:")
    print(lb_test)

    logging.info("Residual analysis completed.")

residual_analysis(final_model)
#%%
# %%
# Convert Log Returns Back to Price Predictions
def convert_predictions(y_test, y_pred, original_close_prices):
    logging.info("Converting log returns to price predictions...")
    y_test = y_test.copy()
    y_pred = pd.Series(y_pred, index=y_test.index)
    shifted_close = original_close_prices.shift(1).loc[y_test.index]
    predicted_close = shifted_close * np.exp(y_pred)
    actual_close = shifted_close * np.exp(y_test)
    logging.info("Conversion completed.")
    return actual_close, predicted_close


original_close_prices = stock_data['close']
actual_close, predicted_close = convert_predictions(y_test, y_pred, original_close_prices)


# %%
# Plot Actual vs Predicted Prices
def plot_actual_vs_predicted(actual_close, predicted_close):
    logging.info("Plotting actual vs predicted prices...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_close.index, y=actual_close, mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=predicted_close.index, y=predicted_close, mode='lines', name='Predicted Price'))
    fig.update_layout(title='Actual vs. Predicted Stock Prices', xaxis_title='Date', yaxis_title='Stock Price')
    fig.show()
    logging.info("Plotting completed.")


plot_actual_vs_predicted(actual_close, predicted_close)


# %%
# Forecasting Future Prices
def forecast_future_prices(model_fit, last_row, steps=30):
    logging.info(f"Forecasting future prices for {steps} steps...")
    future_exog = pd.DataFrame(columns=X.columns)
    future_dates = []

    for i in range(steps):
        future_row = last_row.copy()
        future_date = last_row.name + pd.Timedelta(days=1)
        while future_date.weekday() >= 5:
            future_date += pd.Timedelta(days=1)
        future_dates.append(future_date)

        future_row['day_of_week'] = future_date.weekday()
        future_row['month'] = future_date.month

        # Update other time-dependent features (this is a simplified approach)
        for col in future_row.index:
            if col.endswith('_lag_1'):
                base_col = col[:-6]
                future_row[col] = future_row[base_col]
            elif col.endswith('_lag_2'):
                lag_1_col = col[:-1] + '1'
                future_row[col] = future_row[lag_1_col]
            elif col.endswith('_lag_3'):
                lag_2_col = col[:-1] + '2'
                future_row[col] = future_row[lag_2_col]

        future_exog = future_exog.append(future_row, ignore_index=True)
        last_row = future_row.copy()
        last_row.name = future_date

    future_exog_scaled = scaler.transform(future_exog)

    future_predictions = model_fit.forecast(steps=steps, exog=future_exog_scaled)

    last_close_price = original_close_prices.iloc[-1]
    forecasted_prices = [last_close_price * np.exp(future_predictions.iloc[0])]
    for i in range(1, len(future_predictions)):
        next_price = forecasted_prices[-1] * np.exp(future_predictions.iloc[i])
        forecasted_prices.append(next_price)

    forecast_df = pd.DataFrame({'date': future_dates, 'forecasted_close': forecasted_prices}).set_index('date')

    # Calculate confidence intervals
    conf_int = model_fit.get_forecast(steps=steps, exog=future_exog_scaled).conf_int()
    forecast_df['lower_ci'] = last_close_price * np.exp(conf_int.iloc[:, 0])
    forecast_df['upper_ci'] = last_close_price * np.exp(conf_int.iloc[:, 1])

    logging.info("Forecasting completed.")
    return forecast_df


last_exog_row = X.iloc[-1]
forecast_df = forecast_future_prices(final_model, last_exog_row, steps=30)


# %%
# Plot Forecasted Prices
def plot_forecasted_prices(original_close_prices, forecast_df):
    logging.info("Plotting forecasted prices...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=original_close_prices.index[-100:], y=original_close_prices[-100:], mode='lines',
                             name='Historical Prices'))
    fig.add_trace(
        go.Scatter(x=forecast_df.index, y=forecast_df['forecasted_close'], mode='lines', name='Forecasted Prices'))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['lower_ci'], mode='lines', line=dict(dash='dash'),
                             name='Lower CI'))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['upper_ci'], mode='lines', line=dict(dash='dash'),
                             name='Upper CI'))
    fig.update_layout(title='Stock Price Forecast for Next 30 Days', xaxis_title='Date', yaxis_title='Stock Price')
    fig.show()
    logging.info("Plotting completed.")


plot_forecasted_prices(original_close_prices, forecast_df)


# %%
# Risk Metrics
def calculate_risk_metrics(returns, alpha=0.05):
    logging.info("Calculating risk metrics...")
    var = np.percentile(returns, alpha * 100)
    cvar = returns[returns <= var].mean()

    print(f"Value at Risk (VaR) at {alpha * 100}% confidence level: {var:.4f}")
    print(f"Conditional Value at Risk (CVaR) at {alpha * 100}% confidence level: {cvar:.4f}")
    logging.info("Risk metric calculation completed.")


# Calculate daily returns
daily_returns = stock_data['close'].pct_change().dropna()

calculate_risk_metrics(daily_returns)


# %%
# Implement stop-loss and take-profit signals
def generate_trading_signals(actual_prices, predicted_prices, stop_loss_pct=0.02, take_profit_pct=0.05):
    logging.info("Generating trading signals...")
    signals = pd.DataFrame(index=actual_prices.index, columns=['Signal', 'Price'])
    position = 0
    entry_price = 0

    for i in range(1, len(actual_prices)):
        if position == 0:
            if predicted_prices[i] > actual_prices[i]:
                signals.iloc[i] = ['Buy', actual_prices[i]]
                position = 1
                entry_price = actual_prices[i]
        elif position == 1:
            if actual_prices[i] <= entry_price * (1 - stop_loss_pct):
                signals.iloc[i] = ['Sell', actual_prices[i]]
                position = 0
            elif actual_prices[i] >= entry_price * (1 + take_profit_pct):
                signals.iloc[i] = ['Sell', actual_prices[i]]
                position = 0

    signals = signals.dropna()
    logging.info("Trading signal generation completed.")
    return signals


trading_signals = generate_trading_signals(actual_close, predicted_close)


# %%
# Plot trading signals
def plot_trading_signals(actual_prices, signals):
    logging.info("Plotting trading signals...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_prices.index, y=actual_prices, mode='lines', name='Actual Price'))

    buy_signals = signals[signals['Signal'] == 'Buy']
    sell_signals = signals[signals['Signal'] == 'Sell']

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Price'], mode='markers',
                             marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Price'], mode='markers',
                             marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell Signal'))

    fig.update_layout(title='Trading Signals', xaxis_title='Date', yaxis_title='Stock Price')
    fig.show()
    logging.info("Trading signal plotting completed.")


plot_trading_signals(actual_close, trading_signals)


# %%
# Model Retraining
def retrain_model(data, features, target, order, retrain_interval=30):
    logging.info("Starting model retraining process...")

    for i in range(0, len(data) - retrain_interval, retrain_interval):
        train_data = data.iloc[:i + retrain_interval]
        X_train = train_data[features]
        y_train = train_data[target]

        model = SARIMAX(y_train, order=order, exog=X_train, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)

        logging.info(f"Model retrained with data up to index {i + retrain_interval}")

        # Here you could save the model, update predictions, or perform other tasks with the retrained model

    logging.info("Model retraining process completed.")


# Example usage (commented out to avoid long runtime)
# retrain_model(stock_data, feature_list, 'log_return', best_order)

# %%
# Final Summary
logging.info("Generating final summary...")

print("Stock Price Prediction Model Summary")
print("====================================")
print(f"Stock Symbol: {stock_symbol}")
print(f"Data Range: {stock_data.index[0]} to {stock_data.index[-1]}")
print(f"Number of features used: {len(feature_list)}")
print(f"Best SARIMAX order: {best_order}")
print("\nModel Performance Metrics:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MAPE: {np.mean(np.abs((y_test - y_pred) / y_test)) * 100:.2f}%")

print("\nRisk Metrics:")
calculate_risk_metrics(daily_returns)

print("\nTrading Signals Summary:")
print(f"Number of Buy signals: {len(trading_signals[trading_signals['Signal'] == 'Buy'])}")
print(f"Number of Sell signals: {len(trading_signals[trading_signals['Signal'] == 'Sell'])}")

print("\nNext Steps:")
print("1. Continuously monitor model performance and retrain as necessary")
print("2. Incorporate additional data sources (e.g., news sentiment) for potential improvement")
print("3. Consider ensemble methods or advanced models (e.g., LSTM) for comparison")
print("4. Implement real-time data processing for live trading scenarios")
print("5. Conduct thorough backtesting over various market conditions")

logging.info("Final summary generated. Stock price prediction model analysis complete.")