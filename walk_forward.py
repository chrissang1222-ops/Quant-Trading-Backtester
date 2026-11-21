
import pandas as pd

from strat import moving_average_strategy
from portfolio import backtest
from stats import sharpe_ratio
from performance import (
    total_return,
    maximum_drawdown
)


def walk_forward_analysis(
    prices,
    train_size=756,
    test_size=252,
    short_windows=None,
    long_windows=None
):
    """
    Walk-forward analysis for a moving-average strategy.

    1. Optimize moving-average parameters using training data.
    2. Apply the selected parameters to the next unseen test period.
    3. Repeat across successive test periods.
    4. Combine out-of-sample results.

    Returns a DataFrame containing fold-level performance.
    """

    if short_windows is None:
        short_windows = [20, 30, 50, 75, 100]

    if long_windows is None:
        long_windows = [100, 150, 200, 250, 300]

    prices = prices.dropna().copy()

    if len(prices) < train_size + test_size:
        raise ValueError(
            "Not enough historical data for walk-forward analysis."
        )

    results = []

    fold = 1
    start = 0

    while start + train_size + test_size <= len(prices):

        train_prices = prices.iloc[
            start:start + train_size
        ]

        test_prices = prices.iloc[
            start + train_size:
            start + train_size + test_size
        ]

        best_sharpe = float("-inf")
        best_short = None
        best_long = None

        # Optimize using training data only.
        for short in short_windows:

            for long in long_windows:

                if short >= long:
                    continue

                train_signals = moving_average_strategy(
                    train_prices,
                    short_window=short,
                    long_window=long
                )

                train_portfolio = backtest(train_signals)

                train_returns = (
                    train_portfolio["Total"]
                    .pct_change()
                    .dropna()
                )

                score = sharpe_ratio(train_returns)

                if pd.notna(score) and score > best_sharpe:
                    best_sharpe = score
                    best_short = short
                    best_long = long

        if best_short is None:
            raise ValueError(
                "Unable to find valid moving-average parameters."
            )

        # Include training history to calculate moving averages
        # at the beginning of the test period.
        combined_prices = prices.iloc[
            start:start + train_size + test_size
        ]

        combined_signals = moving_average_strategy(
            combined_prices,
            short_window=best_short,
            long_window=best_long
        )

        test_signals = combined_signals.loc[
            test_prices.index
        ].copy()

        # Do not assume an existing position at the test boundary.
        # The strategy enters only when a new buy signal occurs.
        test_portfolio = backtest(test_signals)

        test_returns = (
            test_portfolio["Total"]
            .pct_change()
            .dropna()
        )

        results.append({
            "Fold": fold,
            "Training Start": train_prices.index[0],
            "Training End": train_prices.index[-1],
            "Testing Start": test_prices.index[0],
            "Testing End": test_prices.index[-1],
            "Best Short MA": best_short,
            "Best Long MA": best_long,
            "Training Sharpe": best_sharpe,
            "Test Return": total_return(test_portfolio),
            "Test Sharpe": sharpe_ratio(test_returns),
            "Test Maximum Drawdown": maximum_drawdown(
                test_portfolio
            )
        })

        fold += 1
        start += test_size

    return pd.DataFrame(results)
