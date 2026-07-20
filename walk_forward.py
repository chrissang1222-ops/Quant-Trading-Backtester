import pandas as pd

from optimization import optimize_strategy
from strat import moving_average_strategy
from portfolio import backtest
from performance import (
    total_return,
    maximum_drawdown
)
from stats import sharpe_ratio


def walk_forward_analysis(
    prices,
    train_size=756,
    test_size=252
):

    results = []

    start = 0

    while start + train_size + test_size <= len(prices):

        train_prices = prices.iloc[
            start:start + train_size
        ]

        test_prices = prices.iloc[
            start + train_size:
            start + train_size + test_size
        ]


        # Step 1: Optimize only on training data
        optimization_results = optimize_strategy(
            train_prices
        )


        best_strategy = optimization_results.iloc[0]


        short_window = int(
            best_strategy["Short MA"]
        )

        long_window = int(
            best_strategy["Long MA"]
        )


        # Step 2: Apply best parameters to unseen data
        signals = moving_average_strategy(
            test_prices,
            short_window=short_window,
            long_window=long_window
        )


        # Step 3: Backtest unseen period
        portfolio = backtest(
            signals
        )


        strategy_returns = (
            portfolio["Total"]
            .pct_change()
            .dropna()
        )


        # Step 4: Save results
        results.append(
            {
                "Train Start": train_prices.index[0],
                "Train End": train_prices.index[-1],
                "Test Start": test_prices.index[0],
                "Test End": test_prices.index[-1],
                "Short MA": short_window,
                "Long MA": long_window,
                "Return": total_return(portfolio),
                "Sharpe Ratio": sharpe_ratio(strategy_returns),
                "Maximum Drawdown": maximum_drawdown(portfolio)
            }
        )


        # Move forward one year
        start += test_size


    return pd.DataFrame(results)
