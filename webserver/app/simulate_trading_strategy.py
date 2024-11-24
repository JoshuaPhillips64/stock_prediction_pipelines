import numpy as np
import pandas as pd
from flask import session, current_app as app
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
#FOR USE WHEN MOVING INTO ROUTES
# Fetch trained model data from the database
with app.app_context():
    if model_type == 'SARIMAX':
        TrainedModel = app.TrainedModels
    elif model_type == 'BINARY CLASSIFICATION':
        TrainedModel = app.TrainedModelsBinary

    trained_model = TrainedModel.query.filter_by(model_key=model_key).first()
    if not trained_model:
        raise Exception("Trained model not found in the database.")

    trained_model_data = trained_model.to_dict()
"""

# Full mocked data from your record
trained_model_data = {
    'prediction_json': {
        "2024-05-13": {"actual_price": 149.12, "predicted_price": 158.1947735455583},
        "2024-05-14": {"actual_price": 147.19, "predicted_price": 158.83097242449352},
        "2024-05-15": {"actual_price": 146.82, "predicted_price": 158.70205179662028},
        "2024-05-16": {"actual_price": 145.8, "predicted_price": 157.1237211725658},
        "2024-05-17": {"actual_price": 146.16, "predicted_price": 156.64369844128598},
        "2024-05-20": {"actual_price": 146.44, "predicted_price": 156.04271782902705},
        "2024-05-21": {"actual_price": 146.03, "predicted_price": 159.04882177842333},
        "2024-05-22": {"actual_price": 145.69, "predicted_price": 163.06195771701857},
        "2024-05-23": {"actual_price": 146.48, "predicted_price": 161.7667524666493},
        "2024-05-24": {"actual_price": 145.48, "predicted_price": 157.5527858819905},
        "2024-05-28": {"actual_price": 147.05, "predicted_price": 155.07028762886105},
        "2024-05-29": {"actual_price": 149.43, "predicted_price": 156.35576607760368},
        "2024-05-30": {"actual_price": 149.70, "predicted_price": 154.60785112335932},
        "2024-05-31": {"actual_price": 149.88, "predicted_price": 153.38650921601356},
        "2024-06-03": {"actual_price": 149.24, "predicted_price": 155.30761721789838},
        "2024-06-04": {"actual_price": 151.01, "predicted_price": 156.74382473712947},
        "2024-06-05": {"actual_price": 156.58, "predicted_price": 152.0908281896564},
        "2024-06-06": {"actual_price": 155.42, "predicted_price": 150.52371119999532},
        "2024-06-07": {"actual_price": 154.69, "predicted_price": 153.33778523955837},
        "2024-06-10": {"actual_price": 154.24, "predicted_price": 156.05746648285512},
        "2024-06-11": {"actual_price": 152.35, "predicted_price": 153.13903225628331},
        "2024-06-12": {"actual_price": 156.28, "predicted_price": 158.62874407269805},
        "2024-06-13": {"actual_price": 159.64, "predicted_price": 152.21806438225798},
        "2024-06-14": {"actual_price": 160.64, "predicted_price": 157.16887649942072},
        "2024-06-17": {"actual_price": 158.56, "predicted_price": 160.79697070968112},
        "2024-06-18": {"actual_price": 161.33, "predicted_price": 159.46975329734087},
        "2024-06-20": {"actual_price": 157.85, "predicted_price": 160.18674068475087},
        "2024-06-21": {"actual_price": 160.76, "predicted_price": 163.67305504170466},
        "2024-06-24": {"actual_price": 164.14, "predicted_price": 163.0724050427646},
        "2024-06-25": {"actual_price": 161.25, "predicted_price": 160.24416919167052},
        "2024-06-26": {"actual_price": 158.97, "predicted_price": 160.43110666435646},
        "2024-06-27": {"actual_price": 158.90, "predicted_price": 159.26295911732817},
        "2024-06-28": {"actual_price": 160.22, "predicted_price": 160.5151690988805},
        "2024-07-01": {"actual_price": 160.62, "predicted_price": 159.6906989119759},
        "2024-07-02": {"actual_price": 159.88, "predicted_price": 153.07018650966376},
        "2024-07-03": {"actual_price": 158.39, "predicted_price": 161.20200718936817},
        "2024-07-05": {"actual_price": 158.48, "predicted_price": 158.9193077631146},
        "2024-07-08": {"actual_price": 159.09, "predicted_price": 156.944115139354},
        "2024-07-09": {"actual_price": 159.39, "predicted_price": 160.20590634742393},
        "2024-07-10": {"actual_price": 159.63, "predicted_price": 164.17925890517023},
        "2024-07-11": {"actual_price": 160.16, "predicted_price": 164.10192892776027},
        "2024-07-12": {"actual_price": 161.43, "predicted_price": 160.77367491004637},
        "2024-07-15": {"actual_price": 162.35, "predicted_price": 162.35306336787613},
        "2024-07-16": {"actual_price": 164.13, "predicted_price": 161.32403816875856},
        "2024-07-17": {"actual_price": 164.61, "predicted_price": 160.32625967725053},
        "2024-07-18": {"actual_price": 162.95, "predicted_price": 158.2295627103512},
        "2024-07-19": {"actual_price": 163.92, "predicted_price": 161.6622321013929},
        "2024-07-22": {"actual_price": 164.23, "predicted_price": 162.0578959285549},
        "2024-07-23": {"actual_price": 165.86, "predicted_price": 160.8201161168862},
        "2024-07-24": {"actual_price": 167.16, "predicted_price": 161.99298601070186},
        "2024-07-25": {"actual_price": 167.36, "predicted_price": 156.84502326196477},
        "2024-07-26": {"actual_price": 164.99, "predicted_price": 152.6695221816654},
        "2024-07-29": {"actual_price": 164.38, "predicted_price": 153.72825807413432},
        "2024-07-30": {"actual_price": 166.61, "predicted_price": 160.0993640670051},
        "2024-07-31": {"actual_price": 167.38, "predicted_price": 156.97826108425522},
        "2024-08-01": {"actual_price": 164.82, "predicted_price": 159.5676805608548},
        "2024-08-02": {"actual_price": 164.64, "predicted_price": 157.1985128661514},
        "2024-08-05": {"actual_price": 165.52, "predicted_price": 160.69556178915175},
        "2024-08-06": {"actual_price": 166.99, "predicted_price": 160.93756382784673},
        "2024-08-07": {"actual_price": 167.07, "predicted_price": 163.8688694945725},
        "2024-08-08": {"actual_price": 166.15, "predicted_price": 165.27768821544265},
        "2024-08-09": {"actual_price": 164.82, "predicted_price": 165.35886990756586},
        "2024-08-12": {"actual_price": 164.16, "predicted_price": 164.1941599185979},
        "2024-08-13": {"actual_price": 163.22, "predicted_price": 163.77015918428125},
        "2024-08-14": {"actual_price": 162.78, "predicted_price": 161.03902349843582},
        "2024-08-15": {"actual_price": 160.6, "predicted_price": 163.38269874341705},
        "2024-08-16": {"actual_price": 161.39, "predicted_price": 161.43371325790906},
        "2024-08-19": {"actual_price": 161.4, "predicted_price": 161.6476565571351},
        "2024-08-20": {"actual_price": 162.06, "predicted_price": 161.19489260291672},
        "2024-08-21": {"actual_price": 161.99, "predicted_price": 163.12077527363198},
        "2024-08-22": {"actual_price": 161.17, "predicted_price": 163.05807044554047},
        "2024-08-23": {"actual_price": 160.5, "predicted_price": 166.2535647416917},
        "2024-08-26": {"actual_price": 160.29, "predicted_price": 166.71196594651363},
        "2024-08-27": {"actual_price": 159.53, "predicted_price": 160.80426802118876},
        "2024-08-28": {"actual_price": 159.69, "predicted_price": 160.69298955186397},
        "2024-08-29": {"actual_price": 160.65, "predicted_price": 159.87786420775848},
        "2024-08-30": {"actual_price": 160.51, "predicted_price": 159.00536333010325},
        "2024-09-03": {"actual_price": 161.46, "predicted_price": 164.9668634275016},
        "2024-09-04": {"actual_price": 161.6, "predicted_price": 164.54156695347802},
        "2024-09-05": {"actual_price": 164.1, "predicted_price": 164.8199898136723},
        "2024-09-06": {"actual_price": 164.28, "predicted_price": 161.54385267725678},
        "2024-09-09": {"actual_price": 164.47, "predicted_price": 159.39282806712725},
        "2024-09-10": {"actual_price": 165.12, "predicted_price": 157.82573069447196},
        "2024-09-11": {"actual_price": 162.83, "predicted_price": 159.93896397833097},
        "2024-09-12": {"actual_price": 163.45, "predicted_price": 157.34834323875887},
        "2024-09-13": {"actual_price": 165.86, "predicted_price": 159.18948028997795},
        "2024-09-16": {"actual_price": 163.67, "predicted_price": 158.82056954054357},
        "2024-09-17": {"actual_price": 160.88, "predicted_price": 163.37073946203313},
        "2024-09-18": {"actual_price": 161.6, "predicted_price": 161.91737784259612},
        "2024-09-19": {"actual_price": 160.09, "predicted_price": 160.3909119470087},
        "2024-09-20": {"actual_price": 160.61, "predicted_price": 159.83092850740974},
        "2024-09-23": {"actual_price": 159.86, "predicted_price": 157.94243947892164},
        "2024-09-24": {"actual_price": 160.13, "predicted_price": 157.51753887937593},
        "2024-11-29": {"actual_price": None, "predicted_price": 154.95320985842528}
    },
    'model_parameters': {
        "order": [0, 0, 0],
        "input_date": "2024-11-02",
        "feature_set": "advanced",
        "seasonal_order": [0, 0, 0, 7],
        "lookback_period": 857,
        "prediction_horizon": 28,
        "hyperparameter_tuning": "LOW"
    }
}

#pull the column prediction_json form the trained_model_data
predictions_json = trained_model_data['prediction_json']

#pull the prediction_horizon from the trained_model_data.model_parameters.prediction_horizon
prediction_horizon = trained_model_data['model_parameters']['prediction_horizon']

def regression_simulate_trading_strategy(predictions_json, prediction_horizon, initial_capital=10000, threshold=0.01):
    """
    Simulate a trading strategy based on model predictions and calculate performance metrics,
    including a comparison with a buy-and-hold strategy.
    """
    logging.info("Starting trading strategy simulation...")

    capital = initial_capital
    trade_results = []
    capital_over_time = []
    open_trades = []
    position_size_fraction = 0.1  # Allocate 10% of available capital to each trade

    # Convert predictions_json to DataFrame
    predictions_df = pd.DataFrame.from_dict(predictions_json, orient='index')
    predictions_df.index = pd.to_datetime(predictions_df.index)
    predictions_df.sort_index(inplace=True)

    # Exclude future predictions without actual prices
    predictions_df = predictions_df.dropna(subset=['actual_price'])

    # Shift predicted prices to align predictions made on date D for date D + prediction_horizon
    predictions_df['predicted_price_shifted'] = predictions_df['predicted_price'].shift(-prediction_horizon)

    # Remove rows where shifted predictions are not available
    predictions_df = predictions_df.dropna(subset=['predicted_price_shifted'])

    # Initialize variables for buy-and-hold strategy
    buy_and_hold_initial_price = predictions_df['actual_price'].iloc[0]
    buy_and_hold_prices = predictions_df['actual_price']
    buy_and_hold_capital_over_time = initial_capital * (buy_and_hold_prices / buy_and_hold_initial_price)

    # Initialize capital series
    capital_series = pd.Series(index=predictions_df.index)

    for current_date in predictions_df.index:
        # Update capital over time
        capital_over_time.append(capital)

        # Close trades that are due on the current date
        trades_to_close = [trade for trade in open_trades if trade['sell_date'] == current_date]
        for trade in trades_to_close:
            # Close the trade
            sell_price = predictions_df.loc[current_date, 'actual_price']
            profit_loss = trade['shares'] * (sell_price - trade['buy_price'])
            capital += profit_loss
            trade['sell_price'] = sell_price
            trade['profit_loss'] = profit_loss
            trade['actual_return'] = (sell_price - trade['buy_price']) / trade['buy_price']
            trade_results.append(trade)
            open_trades.remove(trade)

        # Calculate available capital
        invested_capital = sum(trade['invested_capital'] for trade in open_trades)
        available_capital = capital - invested_capital

        # Get the expected return
        actual_price_on_D = predictions_df.loc[current_date, 'actual_price']
        predicted_price_on_D_plus_X = predictions_df.loc[current_date, 'predicted_price_shifted']
        expected_return = (predicted_price_on_D_plus_X - actual_price_on_D) / actual_price_on_D

        # Check if expected return exceeds threshold
        if expected_return > threshold and available_capital > 0:
            # Allocate a fraction of the available capital
            invested_amount = available_capital * position_size_fraction
            shares = invested_amount / actual_price_on_D
            target_date = current_date + pd.Timedelta(days=prediction_horizon)

            # Record the trade
            trade = {
                'buy_date': current_date,
                'sell_date': target_date,
                'buy_price': actual_price_on_D,
                'shares': shares,
                'invested_capital': invested_amount
            }
            open_trades.append(trade)

    # Close any remaining open trades at the end
    last_date = predictions_df.index[-1]
    for trade in open_trades:
        sell_price = predictions_df.loc[last_date, 'actual_price']
        profit_loss = trade['shares'] * (sell_price - trade['buy_price'])
        capital += profit_loss
        trade['sell_date'] = last_date
        trade['sell_price'] = sell_price
        trade['profit_loss'] = profit_loss
        trade['actual_return'] = (sell_price - trade['buy_price']) / trade['buy_price']
        trade_results.append(trade)
    open_trades = []

    # Update capital over time for remaining dates
    capital_over_time.extend([capital] * (len(predictions_df.index) - len(capital_over_time)))

    # Calculate metrics
    total_trades = len(trade_results)
    profitable_trades = sum(1 for trade in trade_results if trade['profit_loss'] > 0)
    hit_rate = profitable_trades / total_trades if total_trades > 0 else 0
    total_profit_loss = capital - initial_capital
    roi = total_profit_loss / initial_capital
    cumulative_return = capital / initial_capital

    # Calculate Sharpe Ratio
    returns = [trade['actual_return'] for trade in trade_results]
    if len(returns) > 1:
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = avg_return / std_return if std_return != 0 else np.nan
    else:
        sharpe_ratio = np.nan

    # Maximum Drawdown calculation for model-based strategy
    capital_series = pd.Series(capital_over_time, index=predictions_df.index)
    roll_max = capital_series.cummax()
    drawdown = (capital_series - roll_max) / roll_max
    max_drawdown = drawdown.min()

    # Buy-and-hold strategy metrics
    buy_and_hold_final_price = buy_and_hold_prices.iloc[-1]
    buy_and_hold_return = (buy_and_hold_final_price - buy_and_hold_initial_price) / buy_and_hold_initial_price
    buy_and_hold_capital = initial_capital * (1 + buy_and_hold_return)
    buy_and_hold_max_drawdown = ((buy_and_hold_capital_over_time - buy_and_hold_capital_over_time.cummax()) / buy_and_hold_capital_over_time.cummax()).min()

    # Prepare metrics dictionary
    metrics = {
        'model_strategy': {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'hit_rate': hit_rate,
            'total_profit_loss': total_profit_loss,
            'roi': roi,
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        },
        'buy_and_hold_strategy': {
            'total_profit_loss': buy_and_hold_capital - initial_capital,
            'roi': buy_and_hold_return,
            'cumulative_return': buy_and_hold_capital / initial_capital,
            'max_drawdown': buy_and_hold_max_drawdown
        }
    }

    logging.info("Trading strategy simulation completed.")

    return metrics, trade_results

# Run the trading strategy simulation
metrics, trade_results = regression_simulate_trading_strategy(predictions_json, prediction_horizon)

import plotly.graph_objs as go
from plotly.subplots import make_subplots

def plot_trading_simulation(predictions_json, trade_results, metrics):
    # Convert predictions_json to DataFrame
    predictions_df = pd.DataFrame.from_dict(predictions_json, orient='index')
    predictions_df.index = pd.to_datetime(predictions_df.index)
    predictions_df.sort_index(inplace=True)
    predictions_df['Date'] = predictions_df.index

    # Prepare trade results DataFrame
    trades_df = pd.DataFrame(trade_results)
    trades_df['buy_date'] = pd.to_datetime(trades_df['buy_date'])
    trades_df['sell_date'] = pd.to_datetime(trades_df['sell_date'])

    # Align predictions for future dates
    predictions_df['predicted_future_price'] = predictions_df['predicted_price'].shift(-prediction_horizon)
    predictions_df['expected_return'] = (predictions_df['predicted_future_price'] - predictions_df['actual_price']) / \
                                        predictions_df['actual_price']

    # Determine trading signals
    threshold = 0.01  # The threshold used in the simulation
    predictions_df['trading_signal'] = np.where(predictions_df['expected_return'] > threshold, 'Buy Signal',
                                                'No Action')

    # Capital over time for model strategy
    initial_capital = 10000
    capital_over_time = pd.Series([initial_capital], index=[predictions_df.index[0]])

    # Calculate capital changes over time
    for date in predictions_df.index[1:]:
        # Find trades closing on this date
        closing_trades = trades_df[trades_df['sell_date'] == date]
        profit_loss = closing_trades['profit_loss'].sum() if not closing_trades.empty else 0
        # Update capital
        new_capital = capital_over_time[-1] + profit_loss
        capital_over_time = pd.concat([capital_over_time, pd.Series([new_capital], index=[date])])
    capital_over_time = capital_over_time.reindex(predictions_df.index, method='ffill')

    # Buy-and-Hold Strategy Capital Over Time
    initial_price = predictions_df['actual_price'].iloc[0]
    buy_and_hold_capital = initial_capital * (predictions_df['actual_price'] / initial_price)

    # Calculate cumulative returns
    model_cumulative_return = (capital_over_time - initial_capital) / initial_capital
    buy_and_hold_cumulative_return = (buy_and_hold_capital - initial_capital) / initial_capital

    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("Trading Strategy Performance",
                                        "Returns over Time vs Buy and Hold"))

    # --- Chart 1: Actual vs. Predicted Prices with Trading Signals ---

    # Plot Actual Price
    fig.add_trace(go.Scatter(
        x=predictions_df['Date'], y=predictions_df['actual_price'],
        mode='lines', name='Current Price',
        line=dict(color='blue', width=2)
    ), row=1, col=1)

    # Plot Buy Signals
    buy_signals = predictions_df[predictions_df['trading_signal'] == 'Buy Signal']
    fig.add_trace(go.Scatter(
        x=buy_signals['Date'], y=buy_signals['actual_price'],
        mode='markers', name='Buy Signal',
        marker=dict(symbol='triangle-up', color='green', size=10),
        hovertemplate=(
            'Predicted Future Price: $%{customdata[0]:.2f}<br>'
            'Expected Return: %{customdata[1]:.2%}<br>'
            'Decision: Buy Signal<extra></extra>'
        ),
        customdata=np.stack((buy_signals['predicted_future_price'], buy_signals['expected_return']), axis=-1)
    ), row=1, col=1)

    # Update Chart 1 Layout
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)

    # --- Chart 2: Cumulative Returns Over Time ---

    # Plot Model Strategy Cumulative Returns
    fig.add_trace(go.Scatter(
        x=capital_over_time.index, y=model_cumulative_return,
        mode='lines', name='Model Strategy Return',
        line=dict(color='green', width=2)
    ), row=2, col=1)

    # Plot Buy-and-Hold Cumulative Returns
    fig.add_trace(go.Scatter(
        x=predictions_df['Date'], y=buy_and_hold_cumulative_return,
        mode='lines', name='Buy-and-Hold Return',
        line=dict(color='purple', width=2, dash='dot')
    ), row=2, col=1)

    # Highlight Maximum Drawdowns
    # Model Strategy Max Drawdown
    max_dd_start = capital_over_time.idxmax()
    max_dd_end = capital_over_time.idxmin()
    fig.add_shape(type="rect",
                  x0=max_dd_start, y0=model_cumulative_return.min(),
                  x1=max_dd_end, y1=model_cumulative_return.max(),
                  fillcolor="red", opacity=0.2, layer="below", line_width=0,
                  row=2, col=1)

    # Update Chart 2 Layout
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Return", tickformat=".1%", row=2, col=1)

    # --- Overall Figure Layout ---

    fig.update_layout(
        height=800, width=1200,
        title={
            'text': "Trading Simulation Results",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hovermode='x unified',
        template='plotly_white'
    )

    # Add footnotes with performance metrics
    fig.add_annotation(
        text=f"Model Strategy ROI: {metrics['model_strategy']['roi']*100:.2f}% | "
             f"Buy-and-Hold ROI: {metrics['buy_and_hold_strategy']['roi']*100:.2f}% | "
             f"Model Sharpe Ratio: {metrics['model_strategy']['sharpe_ratio']:.2f}",
        align='left',
        showarrow=False,
        xref='paper', yref='paper',
        x=0, y=-0.1,
        bordercolor='black',
        borderwidth=1,
        borderpad=4,
        bgcolor='white',
        opacity=0.8
    )

    # Show the figure
    fig.show()

# Call the function
plot_trading_simulation(predictions_json, trade_results, metrics)

# Call the function
plot_trading_simulation(predictions_json, trade_results, metrics)