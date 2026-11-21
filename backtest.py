import pandas as pd
import matplotlib.pyplot as plt


def run_backtest(
    data,
    short_window=50,
    long_window=200,
    transaction_cost=0.001,
    slippage=0.0005
):
    """
    Run a moving-average crossover backtest.

    Parameters:
        data: DataFrame containing a 'Close' column
        short_window: Short moving-average period
        long_window: Long moving-average period
        transaction_cost: Cost per trade as a decimal
        slippage: Slippage per trade as a decimal

    Returns:
        DataFrame containing backtest results
    """

    data = data.copy()

    # Moving averages
    data["Short_MA"] = (
        data["Close"]
        .rolling(window=short_window)
        .mean()
    )

    data["Long_MA"] = (
        data["Close"]
        .rolling(window=long_window)
        .mean()
    )

    # Trading signal
    data["Signal"] = 0

    data.loc[
        data["Short_MA"] > data["Long_MA"],
        "Signal"
    ] = 1

    # Daily market return
    data["Daily_Return"] = (
        data["Close"].pct_change()
    )

    # Position from previous day
    data["Position"] = (
        data["Signal"].shift(1)
    )

    # Gross strategy return
    data["Gross_Strategy_Return"] = (
        data["Position"] *
        data["Daily_Return"]
    )

    # Detect trades
    data["Trade"] = (
        data["Signal"]
        .diff()
        .abs()
        .fillna(0)
    )

    # Total trading cost
    trading_cost = (
        transaction_cost + slippage
    )

    data["Trading_Cost"] = (
        data["Trade"] *
        trading_cost
    )

    # Net strategy return
    data["Strategy_Return"] = (
        data["Gross_Strategy_Return"]
        - data["Trading_Cost"]
    )

    # Cumulative market return
    data["Market_Return"] = (
        1 + data["Daily_Return"]
    ).cumprod()

    # Cumulative strategy return
    data["Strategy_Returns"] = (
        1 + data["Strategy_Return"]
    ).cumprod()

    return data


def calculate_metrics(data):
    """
    Calculate performance metrics for a completed backtest.
    """

    returns = data["Strategy_Return"].dropna()

    cumulative_returns = (
        1 + returns
    ).cumprod()

    total_return = (
        cumulative_returns.iloc[-1] - 1
    )

    # Number of years in the dataset
    years = (
        (data.index[-1] - data.index[0]).days
        / 365.25
    )

    annual_return = (
        (1 + total_return) ** (1 / years)
        - 1
    )

    volatility = (
        returns.std() *
        (252 ** 0.5)
    )

    # Assume 0% risk-free rate
    sharpe_ratio = (
        annual_return / volatility
        if volatility != 0
        else 0
    )

    # Maximum drawdown
    rolling_max = (
        cumulative_returns.cummax()
    )

    drawdown = (
        cumulative_returns /
        rolling_max
        - 1
    )

    max_drawdown = drawdown.min()

    # Sortino ratio
    downside_returns = returns[
        returns < 0
    ]

    downside_deviation = (
        downside_returns.std() *
        (252 ** 0.5)
        if len(downside_returns) > 0
        else 0
    )

    sortino_ratio = (
        annual_return / downside_deviation
        if downside_deviation != 0
        else 0
    )

    # Win rate
    winning_days = (
        returns > 0
    ).sum()

    total_days = (
        returns != 0
    ).sum()

    win_rate = (
        winning_days / total_days
        if total_days != 0
        else 0
    )

    return {
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate
    }


def print_metrics(metrics):
    """
    Print performance metrics in a clean format.
    """

    print("\nPerformance Report")
    print("----------------------")

    print(
        f"Total Return: "
        f"{metrics['Total Return']:.2%}"
    )

    print(
        f"Annual Return: "
        f"{metrics['Annual Return']:.2%}"
    )

    print(
        f"Volatility: "
        f"{metrics['Volatility']:.2%}"
    )

    print(
        f"Sharpe Ratio: "
        f"{metrics['Sharpe Ratio']:.2f}"
    )

    print(
        f"Sortino Ratio: "
        f"{metrics['Sortino Ratio']:.2f}"
    )

    print(
        f"Max Drawdown: "
        f"{metrics['Max Drawdown']:.2%}"
    )

    print(
        f"Win Rate: "
        f"{metrics['Win Rate']:.2%}"
    )


def plot_results(data, short_window, long_window):
    """
    Plot strategy performance against buy-and-hold.
    """

    plt.figure(figsize=(12, 6))

    plt.plot(
        data.index,
        data["Market_Return"],
        label="Buy & Hold"
    )

    plt.plot(
        data.index,
        data["Strategy_Returns"],
        label="Moving Average Strategy"
    )

    plt.title(
        f"AAPL Quant Strategy Backtest "
        f"({short_window}/{long_window} MA)"
    )

    plt.xlabel("Date")
    plt.ylabel("Growth of $1")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.show()


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    # Load data
    data = pd.read_csv(
        "AAPL_data.csv",
        index_col="Date",
        parse_dates=True
    )

    # Parameters
    short_window = 50
    long_window = 200

    transaction_cost = 0.001
    slippage = 0.0005

    # Run backtest
    results = run_backtest(
        data,
        short_window=short_window,
        long_window=long_window,
        transaction_cost=transaction_cost,
        slippage=slippage
    )

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Print results
    print_metrics(metrics)

    # Plot results
    plot_results(
        results,
        short_window,
        long_window
    )