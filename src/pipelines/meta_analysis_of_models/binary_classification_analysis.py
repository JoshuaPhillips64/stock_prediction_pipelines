#%%
"""
Description:
  This script demonstrates a meta-analysis of stock predictions using a binary
  classification approach (predicted_movement = "Up" or "Down"). It:
    1. Loads data from the 'trained_models' table via a SQL query (which now returns
       predicted_movement instead of a numeric predicted_amount).
    2. Fetches actual closing prices from Yahoo Finance (based on each row’s symbol & end_date).
    3. Computes performance metrics:
       - Direction correctness
       - Strategy returns (long if predicted Up, short if predicted Down)
    4. Performs an overall meta-analysis across all predictions, using Plotly for:
       - A bar chart (Correct vs. Incorrect predictions)
       - A histogram (Distribution of strategy returns)
"""

import pandas as pd
import numpy as np
import plotly.express as px
from typing import Optional
from database_functions import create_engine_from_url, fetch_dataframe
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
import yfinance as yf
from datetime import datetime, timedelta

def load_data() -> pd.DataFrame:
    """
    Load data from the 'trained_models' table. Instead of fetching a numeric predicted_amount,
    we now retrieve a binary classification output: 'predicted_movement' (Up/Down).

    Returns
    -------
    df : pd.DataFrame
        A DataFrame with columns:
          symbol, start_date, end_date, start_price, predicted_movement
    """
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine_from_url(db_url)

    query = """
        Select symbol, model_parameters ->> 'input_date' as start_date, prediction_date as end_date, last_known_price as start_price, predicted_movement
        from trained_models_binary
        where prediction_f1_score > 0.82
        and prediction_date < now()
    """
    df = fetch_dataframe(engine, query)
    return df

def get_close_price(symbol: str, end_date: pd.Timestamp) -> Optional[float]:
    """
    Attempt to fetch the actual close price via Yahoo Finance.

    Logic:
    - If end_date is in the future (relative to today), return None.
    - Otherwise, search up to 7 calendar days from end_date to find a trading day.
    - For the first day that returns data, grab its 'Close'.
    - If none found within 7 days, return None.
    """
    today = datetime.today().date()
    if end_date.date() > today:
        return None

    for i in range(8):
        check_day = end_date + timedelta(days=i)
        start = check_day
        end = check_day + timedelta(days=1)
        hist = yf.Ticker(symbol).history(start=start, end=end)
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    return None

def fetch_actual_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch 'actual_price' for each row using Yahoo Finance, then drop rows where no price is found.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns 'symbol' and 'end_date'.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with an 'actual_price' column (float). Rows with None for actual_price are dropped.
    """
    # Convert date columns to datetime
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    df["actual_price"] = df.apply(
        lambda row: get_close_price(row["symbol"], row["end_date"]),
        axis=1
    )

    # Drop rows without an actual_price
    df.dropna(subset=["actual_price"], inplace=True)

    return df

def compute_performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute key metrics of prediction performance for the binary classification model:
      - predicted_direction = +1 if predicted_movement = "Up", otherwise -1
      - actual_direction = +1 if actual_price >= start_price, otherwise -1
      - correct_direction = 1 if predicted_direction == actual_direction
      - strategy_return:
         * If predicted Up:   (actual_price - start_price) / start_price
         * If predicted Down: (start_price - actual_price) / start_price

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'predicted_movement', 'start_price', 'actual_price'.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added columns:
          predicted_direction, actual_direction, correct_direction, strategy_return.
    """
    # Convert "Up"/"Down" -> +1 / -1
    df["predicted_direction"] = df["predicted_movement"].apply(lambda x: 1 if x == "Up" else -1)

    # Actual direction: +1 if actual_price >= start_price else -1
    df["actual_direction"] = np.where(df["actual_price"] >= df["start_price"], 1, -1)

    # Correctness
    df["correct_direction"] = (df["predicted_direction"] == df["actual_direction"]).astype(int)

    # Strategy return
    df["strategy_return"] = np.where(
        df["predicted_direction"] == 1,
        (df["actual_price"] - df["start_price"]) / df["start_price"],
        (df["start_price"] - df["actual_price"]) / df["start_price"]
    )

    return df

def meta_analysis_overall(df: pd.DataFrame) -> None:
    """
    Perform an overall meta-analysis (across ALL rows) by:
      - Printing overall accuracy.
      - Plotting correct vs. incorrect predictions (bar chart).
      - Plotting distribution of strategy returns (histogram).

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: 'correct_direction' and 'strategy_return'.
    """
    total_predictions = len(df)
    total_correct = df["correct_direction"].sum()
    overall_accuracy = (total_correct / total_predictions) * 100 if total_predictions > 0 else 0.0

    print(f"Overall Correct Predictions: {total_correct} / {total_predictions}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    # Correct vs. Incorrect (bar chart)
    df["correct_label"] = df["correct_direction"].apply(lambda x: "Correct" if x == 1 else "Incorrect")
    correctness_count = df["correct_label"].value_counts().reset_index()
    correctness_count.columns = ["correct_label", "count"]

    fig_bar = px.bar(
        correctness_count,
        x="correct_label",
        y="count",
        title="Overall Correct vs. Incorrect Predictions",
        text="count",
        labels={"correct_label": "", "count": "Number of Predictions"},
        template="plotly_white"
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.show()

    # Distribution of strategy returns (histogram)
    fig_hist = px.histogram(
        df,
        x="strategy_return",
        nbins=20,
        title="Distribution of Strategy Returns (All Models Combined)",
        labels={"strategy_return": "Strategy Return"},
        template="plotly_white"
    )
    fig_hist.show()

def main():
    """
    Main function that:
      1) Loads data from the DB (with 'predicted_movement' = "Up"/"Down").
      2) Fetches actual prices from Yahoo Finance for each row.
      3) Computes performance metrics (correctness, strategy_return).
      4) Performs an overall meta-analysis across all rows.
    """
    # 1. Load data (now includes predicted_movement instead of numeric predicted_amount)
    df = load_data()

    # 2. Fetch actual prices (drop rows with missing actual_price)
    df = fetch_actual_prices(df)

    # 3. Compute performance metrics
    df = compute_performance_metrics(df)

    # 4. Perform overall meta-analysis
    meta_analysis_overall(df)

if __name__ == "__main__":
    main()