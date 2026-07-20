import pandas as pd

from data import download_data
from strat import moving_average_strategy
from portfolio import backtest, compare_to_buy_hold

from performance import (
    total_return,
    maximum_drawdown
)

from stats import sharpe_ratio

from benchmark import benchmark_statistics

from risk_metrics import (
    value_at_risk,
    conditional_var,
    beta,
    alpha,
    tracking_error,
    information_ratio
)

from optimization import optimize_strategy
from walk_forward import walk_forward_analysis


# =====================================
# Stocks to Test
# =====================================

tickers = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "TSLA",
    "META",
    "SPY"
]


start_date = "2015-01-01"
end_date = "2025-01-01"


all_results = []


# =====================================
# Run Quant Framework
# =====================================

for ticker in tickers:

    print("\n================================")
    print(f"Analyzing {ticker}")
    print("================================")


    # Download Data
    prices = download_data(
        ticker=ticker,
        start=start_date,
        end=end_date
    )


    prices = prices["Close"].squeeze()


    # Strategy
    signals = moving_average_strategy(
        prices,
        short_window=100,
        long_window=300
    )


    # Backtest
    portfolio = backtest(
        signals
    )


    # Returns
    strategy_returns = (
        portfolio["Total"]
        .pct_change()
        .dropna()
    )


    # Performance Metrics

    total_ret = total_return(
        portfolio
    )


    sharpe = sharpe_ratio(
        strategy_returns
    )


    drawdown = maximum_drawdown(
        portfolio
    )


    # Benchmark

    benchmark = benchmark_statistics(
        prices
    )


    # Risk Metrics

    var = value_at_risk(
        strategy_returns
    )


    cvar = conditional_var(
        strategy_returns
    )


    # Optimization

    optimization_results = optimize_strategy(
        prices
    )

    best_strategy = optimization_results.iloc[0]


    # Walk Forward

    walk_forward_results = walk_forward_analysis(
        prices
    )


    walk_forward_results.to_csv(
        f"{ticker}_walk_forward.csv",
        index=False
    )


    # Store Results

    all_results.append(
        {

            "Ticker": ticker,

            "Strategy Return":
            total_ret,

            "Strategy Sharpe":
            sharpe,

            "Maximum Drawdown":
            drawdown,


            "Buy Hold Return":
            benchmark["Total Return"],

            "Buy Hold Sharpe":
            benchmark["Sharpe Ratio"],


            "VaR":
            var,

            "CVaR":
            cvar,


            "Best Short MA":
            best_strategy["Short MA"],

            "Best Long MA":
            best_strategy["Long MA"]

        }
    )



# =====================================
# Final Results Table
# =====================================

results = pd.DataFrame(
    all_results
)


print("\n\nFINAL MULTI-ASSET RESULTS")
print(results)


results.to_csv(
    "multi_asset_results.csv",
    index=False
)


print("\nSaved:")
print("multi_asset_results.csv")