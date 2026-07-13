import pandas as pd

from strat import moving_average_strategy
from portfolio import backtest
from stats import sharpe_ratio
from performance import (
    total_return,
    maximum_drawdown
)


def optimize_strategy(
    prices,
    short_windows=None,
    long_windows=None
):
    """
    Tests multiple moving average combinations
    and ranks strategies by Sharpe Ratio.
    """

    if short_windows is None:
        short_windows = [20, 30, 50, 75, 100]

    if long_windows is None:
        long_windows = [100, 150, 200, 250, 300]


    results = []


    for short in short_windows:

        for long in long_windows:

            # Avoid invalid combinations
            if short >= long:
                continue


            # Create strategy
            signals = moving_average_strategy(
                prices,
                short_window=short,
                long_window=long
            )


            # Run backtest
            portfolio = backtest(signals)


            # Calculate returns
            returns = (
                portfolio["Total"]
                .pct_change()
                .dropna()
            )


            results.append(
                {
                    "Short MA": short,
                    "Long MA": long,
                    "Total Return": total_return(portfolio),
                    "Sharpe Ratio": sharpe_ratio(returns),
                    "Maximum Drawdown": maximum_drawdown(portfolio)
                }
            )


    results = pd.DataFrame(results)


    # Rank by Sharpe Ratio
    results = results.sort_values(
        by="Sharpe Ratio",
        ascending=False
    )


    return results