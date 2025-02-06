#%%
"""
Description: This script demonstrates a meta-analysis of stock predictions,
             comparing predicted prices vs. actual prices (fetched via Yahoo Finance)
             and computing performance metrics (e.g., directional accuracy, strategy returns).
             The final analysis aggregates across ALL models, rather than by symbol.
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
    Load data from the 'trained_models' table into a pandas DataFrame.
    We include 'predicted_amount' for performance calculations.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame with columns:
            symbol, start_date, end_date, start_price, predicted_amount
    """
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine_from_url(db_url)

    # NOTE: We also pull 'predicted_amount' to compare vs. actual.
    #       Adjust the WHERE clause to suit your needs (e.g., filtering by MAPE, date).
    query = """
        SELECT
            symbol,
            model_parameters ->> 'input_date' AS start_date,
            prediction_date AS end_date,
            last_known_price AS start_price,
            predicted_amount
        FROM trained_models_binary
        WHERE prediction_mape < 1000
          AND prediction_date < now()
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
    Fetch the 'actual_price' for each row using Yahoo Finance.
    Then drop rows that could not retrieve an actual price.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (must have columns 'symbol' and 'end_date').

    Returns
    -------
    df : pd.DataFrame
        DataFrame with an 'actual_price' column. Rows that have None for actual_price are dropped.
    """
    # Ensure date columns are proper datetime
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    # Fetch actual_price from Yahoo Finance
    df["actual_price"] = df.apply(
        lambda row: get_close_price(row["symbol"], row["end_date"]),
        axis=1
    )

    # Drop rows with missing actual_price, since we can't measure performance
    df.dropna(subset=["actual_price"], inplace=True)

    return df

def compute_performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute directional accuracy and strategy returns for each prediction.

    1. predicted_direction = sign(predicted_amount - start_price)
    2. actual_direction = sign(actual_price - start_price)
    3. correct_direction = 1 if predicted_direction == actual_direction, else 0
    4. strategy_return:
         if predicted_direction > 0 (long):  (actual_price - start_price) / start_price
         if predicted_direction < 0 (short): (start_price - actual_price) / start_price
         else: 0

    Returns
    -------
    df : pd.DataFrame
        DataFrame with new columns: 'predicted_direction', 'actual_direction',
        'correct_direction', 'strategy_return'.
    """
    df["predicted_direction"] = np.sign(df["predicted_amount"] - df["start_price"])
    df["actual_direction"] = np.sign(df["actual_price"] - df["start_price"])
    df["correct_direction"] = (df["predicted_direction"] == df["actual_direction"]).astype(int)

    long_return = (df["actual_price"] - df["start_price"]) / df["start_price"]
    short_return = (df["start_price"] - df["actual_price"]) / df["start_price"]

    df["strategy_return"] = np.where(
        df["predicted_direction"] > 0,
        long_return,
        np.where(df["predicted_direction"] < 0, short_return, 0)
    )

    return df

def meta_analysis_overall(df: pd.DataFrame) -> None:
    """
    Perform an overall meta-analysis (across all models) by:
    1) Printing overall accuracy.
    2) Plotting (Plotly) a bar chart showing correct vs. incorrect predictions.
    3) Plotting (Plotly) a histogram of strategy returns distribution.
    4) Optionally highlight top 3 most frequent symbols.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'symbol', 'correct_direction', 'strategy_return'
    """
    total_predictions = len(df)
    total_correct = df["correct_direction"].sum()
    overall_accuracy = (total_correct / total_predictions) * 100 if total_predictions else 0.0

    print(f"Overall Correct Predictions: {total_correct} / {total_predictions}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    # Plot Correct vs. Incorrect (bar chart)
    df["correct_label"] = df["correct_direction"].apply(lambda x: "Correct" if x == 1 else "Incorrect")
    correctness_count = df["correct_label"].value_counts().reset_index()
    correctness_count.columns = ["correct_label", "count"]

    fig_bar = px.bar(
        correctness_count,
        x="correct_label",
        y="count",
        title="Overall Correct vs. Incorrect Predictions",
        text="count",
        labels={"correct_label": "", "count": "Number of Predictions"}
    )
    fig_bar.update_layout(template="plotly_white")
    fig_bar.update_traces(textposition="outside")
    fig_bar.show()

    # Plot distribution of strategy returns (histogram)
    fig_hist = px.histogram(
        df,
        x="strategy_return",
        nbins=20,
        title="Distribution of Strategy Returns (All Models Combined)",
        labels={"strategy_return": "Strategy Return"},
        template="plotly_white"
    )
    fig_hist.show()

    # Highlight top symbols (optional)
    if "symbol" in df.columns:
        top_symbols = df["symbol"].value_counts().head(3)
        print("\nTop 3 symbols (by frequency in predictions):")
        print(top_symbols)

def main():
    """
    Main function that:
    1) Loads data from DB.
    2) Fetches actual prices from Yahoo Finance.
    3) Computes performance metrics.
    4) Performs a meta-analysis across all models/predictions.
    """
    # 1. Load data (includes predicted_amount)
    df = load_data()

    # 2. Fetch actual prices (drop rows with missing actual_price)
    df = fetch_actual_prices(df)

    # 3. Compute performance metrics (correctness, strategy_return)
    df = compute_performance_metrics(df)

    # 4. Perform overall meta-analysis
    meta_analysis_overall(df)

if __name__ == "__main__":
    main()