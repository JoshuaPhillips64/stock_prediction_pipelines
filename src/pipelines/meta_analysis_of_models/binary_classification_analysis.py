# %%
"""
Description:
  This script demonstrates a meta-analysis of stock predictions using a binary
  classification approach (predicted_movement = "Up" or "Down"). It:
    1. Loads data from the 'trained_models_binary' table via a SQL query.
    2. Fetches actual closing prices from Yahoo Finance (based on each row’s symbol & end_date).
    3. Computes performance metrics:
       - Strategy returns (only invest if predicted_movement is "Up")
    4. Performs an overall meta-analysis for investment decisions:
       - Only invests on predictions that say "Up"
       - Calculates total invested amount, the final total value, and the ROI,
         assuming $1,000 is invested per prediction (only on "Up" predictions).
       - Identifies the biggest winners and losers contributing to ROI.
       - Displays an advanced, interactive scatter plot showing each trade’s contribution:
         * Circle size = absolute value of ROI (bigger winners/losers → bigger circles).
         * Negative ROI trades = red circles; positive ROI trades = green circles.
         * A dashed line at $1,000 for break-even.
         * A second y-axis for "Cumulative ROI (%)" with a fixed range from -3% to 10%,
           including a dashed line at 0% ROI.
         * A summary annotation highlighting overall ROI.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
from database_functions import create_engine_from_url, fetch_dataframe
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
import yfinance as yf
from datetime import datetime, timedelta


def load_data() -> pd.DataFrame:
    """
    Load data from the 'trained_models_binary' table.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame with columns:
          symbol, start_date, end_date, start_price, predicted_movement
    """
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine_from_url(db_url)
    query = """
        SELECT symbol, 
               model_parameters ->> 'input_date' AS start_date, 
               prediction_date AS end_date, 
               last_known_price AS start_price, 
               predicted_movement
        FROM trained_models_binary
        WHERE prediction_f1_score > 0.82
          AND prediction_date < NOW()
    """
    df = fetch_dataframe(engine, query)
    return df


def get_close_price(symbol: str, end_date: pd.Timestamp) -> Optional[float]:
    """
    Fetch the actual close price via Yahoo Finance.

    If end_date is in the future, returns None.
    Searches up to 7 days from end_date for a trading day and returns the first available 'Close' price.
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
    Fetch 'actual_price' for each row using Yahoo Finance,
    then drop rows where no price is found.
    """
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["actual_price"] = df.apply(
        lambda row: get_close_price(row["symbol"], row["end_date"]),
        axis=1
    )
    df.dropna(subset=["actual_price"], inplace=True)
    return df


def compute_performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute performance metrics.

    - For predictions labeled "Up", strategy_return = (actual_price - start_price) / start_price.
    - For "Down" predictions (not invested), strategy_return is computed but will be ignored in the investment analysis.
    """
    df["predicted_direction"] = df["predicted_movement"].apply(lambda x: 1 if x == "Up" else -1)
    df["actual_direction"] = np.where(df["actual_price"] >= df["start_price"], 1, -1)
    df["correct_direction"] = (df["predicted_direction"] == df["actual_direction"]).astype(int)
    df["strategy_return"] = np.where(
        df["predicted_direction"] == 1,
        (df["actual_price"] - df["start_price"]) / df["start_price"],
        (df["start_price"] - df["actual_price"]) / df["start_price"]
    )
    return df


def plot_individual_contributions(
        invest_df: pd.DataFrame,
        total_investment: float,
        total_final_money: float,
        roi: float) -> None:
    """
    Creates an advanced, interactive scatter plot showing individual investment contributions.
    Each point represents a $1,000 investment’s final money value.

    Key Features:
      - Negative ROI trades are colored red; positive ROI trades are green.
      - Circle sizes are scaled by the absolute ROI.
      - A dashed horizontal line is placed at $1,000 (break-even).
      - A second y-axis displays the Cumulative ROI (%) with a fixed range from -3 to 10.
      - A dashed line on the right axis marks 0% ROI.
      - The hover popup now includes the start date along with other details.
      - A summary annotation highlights the overall ROI.
    """

    # Copy data and add a sequential trade index
    df_plot = invest_df.copy()
    df_plot["trade_index"] = range(1, len(df_plot) + 1)

    # Create final money per trade = $1,000 * (1 + strategy_return)
    df_plot["final_money"] = 1000 * (1 + df_plot["strategy_return"])
    df_plot["abs_roi"] = df_plot["strategy_return"].abs()

    # Color negative ROI red, positive ROI green
    df_plot["roi_color"] = np.where(df_plot["strategy_return"] < 0, "Negative", "Positive")

    # Create a column for hover: format the start_date as a string (YYYY-MM-DD)
    df_plot["start_date_str"] = df_plot["start_date"].dt.strftime("%Y-%m-%d")
    df_plot["strategy_return_pct"] = df_plot["strategy_return"].apply(lambda x: f"{x:.2%}")

    # -----------------------------
    # Compute Cumulative ROI
    # -----------------------------
    df_plot["cumulative_final"] = df_plot["final_money"].cumsum()
    df_plot["cumulative_invested"] = 1000 * df_plot["trade_index"]  # $1,000 per trade
    df_plot["cumulative_roi"] = ((df_plot["cumulative_final"] - df_plot["cumulative_invested"]) /
                                 df_plot["cumulative_invested"]) * 100

    # -----------------------------
    # Plotly Express Scatter for individual trades
    # -----------------------------
    # Include start_date_str in custom_data for the hover popup
    fig = px.scatter(
        df_plot,
        x="trade_index",
        y="final_money",
        color="roi_color",
        color_discrete_map={"Negative": "red", "Positive": "green"},
        size="abs_roi",
        size_max=30,
        custom_data=["symbol", "start_date_str", "start_price", "actual_price", "strategy_return_pct"],
        title=f"Individual Investment Contributions (Total ROI: {roi:.2f}%)"
    )

    # Update the hover template for the trade markers to include start date
    fig.update_traces(
        hovertemplate=(
            "<b>Trade #: %{x}</b><br>"
            "Final Money: %{y:$,.2f}<br>"
            "Symbol: %{customdata[0]}<br>"
            "Start Date: %{customdata[1]}<br>"
            "Start Price: %{customdata[2]:$,.2f}<br>"
            "Actual Price: %{customdata[3]:$,.2f}<br>"
            "Strategy Return: %{customdata[4]}<br>"
            "<extra></extra>"
        )
    )

    # -----------------------------
    # Add the Cumulative ROI line (second y-axis)
    # -----------------------------
    fig.add_trace(
        go.Scatter(
            x=df_plot["trade_index"],
            y=df_plot["cumulative_roi"],
            mode="lines",
            line=dict(color="orange", width=3),
            name="Cumulative ROI (%)",
            hovertemplate="Trade #: %{x}<br>Cumulative ROI: %{y:.2f}%<extra></extra>",
            yaxis="y2"
        )
    )

    # -----------------------------
    # Add dashed horizontal lines
    # -----------------------------
    # Horizontal line at $1,000 on the left axis (break-even)
    fig.add_hline(
        y=1000,
        line_dash="dash",
        line_color="black",
        annotation_text="Break-even (0% ROI)",
        annotation_position="bottom right",
        opacity=0.7
    )

    # -----------------------------
    # Update layout for axes
    # -----------------------------
    fig.update_layout(
        xaxis=dict(
            title="Investment Trade Number",
            tickfont=dict(size=14, family="Arial, sans-serif", color="#333")
        ),
        yaxis=dict(
            title="Final Money per Trade (USD)",
            tickprefix="$",
            tickfont=dict(size=14, family="Arial, sans-serif", color="#333"),
            showgrid=True
        ),
        yaxis2=dict(
            title="Cumulative ROI (%)",
            overlaying="y",
            side="right",
            range=[-6, 12],  # Fixed range for the right axis from -3% to 10%
            tickfont=dict(size=14, family="Arial, sans-serif", color="orange"),
            showgrid=False
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", color="#333", size=14),
        margin=dict(l=60, r=60, t=80, b=50),
        legend=dict(
            x=0.01,
            y=0.99,
            borderwidth=1
        )
    )

    # -----------------------------
    # Add a summary annotation at the top
    # -----------------------------
    fig.add_annotation(
        text=f"Overall ROI: {roi:.2f}%",
        x=0.5,
        y=1.15,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="#333", family="Arial, sans-serif")
    )

    fig.show()
    ## Using full_html=False produces an embeddable snippet (i.e. without <html>/<body> tags).
    fig.write_html("meta_analysis_chart.html", full_html=False, include_plotlyjs="cdn")
    print("Chart exported to meta_analysis_chart.html")


def meta_analysis_overall(df: pd.DataFrame) -> None:
    """
    Performs overall meta-analysis for investment decisions.

    Only investments where predicted_movement is "Up" are considered.
    Calculates total investment, final value, ROI, and identifies the biggest winners and losers.
    Finally, displays an interactive scatter plot showing individual contributions.
    """
    # Filter only "Up" predictions for investing
    invest_df = df[df["predicted_movement"] == "Up"].copy()
    invest_count = len(invest_df)
    if invest_count == 0:
        print("No 'Up' predictions found. No investments made.")
        return

    # Calculate investment results for each trade
    invest_df["final_money"] = 1000 * (1 + invest_df["strategy_return"])
    total_investment = 1000 * invest_count
    total_final_money = invest_df["final_money"].sum()
    roi = ((total_final_money - total_investment) / total_investment) * 100

    print("\n--- Investment Performance (Investing Only on 'Up' Predictions) ---")
    print(f"Number of Investments: {invest_count}")
    print(f"Total Investment: ${total_investment:,.2f}")
    print(f"Total Final Money: ${total_final_money:,.2f}")
    print(f"ROI: {roi:.2f}%")

    # Identify biggest winners and losers
    winners = invest_df.nlargest(3, "strategy_return")
    losers = invest_df.nsmallest(3, "strategy_return")

    print("\n--- Biggest Winners ---")
    print(winners[["symbol", "predicted_movement", "start_price", "actual_price", "strategy_return"]].to_string(index=False))

    print("\n--- Biggest Losers ---")
    print(losers[["symbol", "predicted_movement", "start_price", "actual_price", "strategy_return"]].to_string(index=False))

    # Create the final scatter + line plot
    plot_individual_contributions(invest_df, total_investment, total_final_money, roi)


def main():
    """
    Main function:
      1) Loads data (with 'predicted_movement' = "Up"/"Down").
      2) Fetches actual prices from Yahoo Finance.
      3) Computes performance metrics.
      4) Performs meta-analysis for investments on "Up" predictions,
         including ROI calculations, identification of outliers,
         and an advanced interactive scatter plot that:
           - Colors negative-ROI trades in red, positive-ROI in green
           - Sizes circles by absolute ROI
           - Shows a second axis for cumulative ROI with fixed range [-3, 10]
           - Places dashed lines at $1,000 (break-even) and 0% ROI on the secondary axis
           - Includes an annotation summarizing overall ROI
    """
    df = load_data()
    df = fetch_actual_prices(df)
    df = compute_performance_metrics(df)
    meta_analysis_overall(df)


if __name__ == "__main__":
    main()