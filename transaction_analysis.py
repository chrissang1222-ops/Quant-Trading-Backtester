import pandas as pd

from data import download_data
from strat import moving_average_strategy
from portfolio import backtest
from performance import (
    total_return,
    maximum_drawdown
)
from stats import sharpe_ratio


def run_transaction_analysis():

    # Load data
    prices = download_data(
        ticker="AAPL",
        start="2015-01-01",
        end="2025-01-01"
    )

    prices = prices["Close"].squeeze()


    # Use your optimized strategy
    signals = moving_average_strategy(
        prices,
        short_window=100,
        long_window=300
    )


    results = []


    scenarios = [
        {
            "Scenario": "No Costs",
            "commission": 0,
            "slippage": 0
        },
        {
            "Scenario": "Realistic Costs",
            "commission": 0.001,
            "slippage": 0.0005
        }
    ]


    for scenario in scenarios:

        portfolio = backtest(
            signals,
            commission=scenario["commission"],
            slippage=scenario["slippage"]
        )


        strategy_returns = (
            portfolio["Total"]
            .pct_change()
            .dropna()
        )


        results.append(
            {
                "Scenario": scenario["Scenario"],
                "Total Return": total_return(portfolio),
                "Sharpe Ratio": sharpe_ratio(strategy_returns),
                "Maximum Drawdown": maximum_drawdown(portfolio)
            }
        )


    results = pd.DataFrame(results)

    print("\nTransaction Cost Analysis")
    print(results)


    results.to_csv(
        "transaction_cost_results.csv",
        index=False
    )


if __name__ == "__main__":
    run_transaction_analysis()