import numpy as np
import pandas as pd



def regression_simulate_trading_strategy(predictions_json, prediction_horizon, initial_capital=10000, threshold=0.01):
    """
    Simulate a trading strategy based on model predictions and calculate performance metrics,
    including a comparison with a buy-and-hold strategy.
    """
    logging.info("Starting trading strategy simulation...")

    capital = initial_capital
    trade_results = []
    capital_over_time = [initial_capital]

    # Convert predictions_json to DataFrame
    predictions_df = pd.DataFrame.from_dict(predictions_json, orient='index')
    predictions_df.index = pd.to_datetime(predictions_df.index)
    predictions_df.sort_index(inplace=True)

    # Exclude future predictions without actual prices
    predictions_df = predictions_df.dropna(subset=['actual_price'])

    # Initialize variables for buy-and-hold strategy
    buy_and_hold_initial_price = predictions_df['actual_price'].iloc[0]
    buy_and_hold_final_price = predictions_df['actual_price'].iloc[-1]
    buy_and_hold_return = (buy_and_hold_final_price - buy_and_hold_initial_price) / buy_and_hold_initial_price
    buy_and_hold_capital = initial_capital * (1 + buy_and_hold_return)

    # For calculating buy-and-hold capital over time
    buy_and_hold_capital_over_time = initial_capital * (predictions_df['actual_price'] / buy_and_hold_initial_price)

    for prediction_date in predictions_df.index:
        actual_price_on_D = predictions_df.loc[prediction_date, 'actual_price']
        predicted_price_on_D_plus_X = predictions_df.loc[prediction_date, 'predicted_price']

        # Compute the expected return
        expected_return = (predicted_price_on_D_plus_X - actual_price_on_D) / actual_price_on_D

        # Check if expected return exceeds threshold
        if expected_return > threshold:
            # Buy on date D
            shares = capital / actual_price_on_D
            # Sell on date D + X
            target_date = prediction_date + pd.Timedelta(days=prediction_horizon)
            if target_date in predictions_df.index:
                actual_price_on_target_date = predictions_df.loc[target_date, 'actual_price']
                if not pd.isna(actual_price_on_target_date):
                    # Calculate actual return
                    actual_return = (actual_price_on_target_date - actual_price_on_D) / actual_price_on_D
                    profit_loss = shares * (actual_price_on_target_date - actual_price_on_D)
                    capital += profit_loss
                    capital_over_time.append(capital)
                    trade_results.append({
                        'buy_date': prediction_date.strftime('%Y-%m-%d'),
                        'sell_date': target_date.strftime('%Y-%m-%d'),
                        'buy_price': actual_price_on_D,
                        'sell_price': actual_price_on_target_date,
                        'shares': shares,
                        'profit_loss': profit_loss,
                        'actual_return': actual_return
                    })
                else:
                    logging.warning(f"Actual price not available on target date {target_date}. Skipping trade.")
            else:
                logging.warning(f"Target date {target_date} not in predictions. Skipping trade.")

    # Calculate metrics for the model-based strategy
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
    capital_series = pd.Series(capital_over_time)
    roll_max = capital_series.cummax()
    drawdown = (capital_series - roll_max) / roll_max
    max_drawdown = drawdown.min()

    # Maximum Drawdown calculation for buy-and-hold strategy
    buy_and_hold_series = buy_and_hold_capital_over_time
    buy_and_hold_roll_max = buy_and_hold_series.cummax()
    buy_and_hold_drawdown = (buy_and_hold_series - buy_and_hold_roll_max) / buy_and_hold_roll_max
    buy_and_hold_max_drawdown = buy_and_hold_drawdown.min()

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