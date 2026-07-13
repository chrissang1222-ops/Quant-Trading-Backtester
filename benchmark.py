import pandas as pd
import numpy as np


def benchmark_return(prices):
    """
    Buy-and-hold return
    """

    return (
        prices.iloc[-1] /
        prices.iloc[0]
    ) - 1



def benchmark_volatility(prices):
    """
    Annualized volatility
    """

    returns = prices.pct_change().dropna()

    return returns.std() * np.sqrt(252)



def benchmark_sharpe(prices):
    """
    Annualized Sharpe Ratio
    """

    returns = prices.pct_change().dropna()

    return (
        returns.mean() /
        returns.std()
    ) * np.sqrt(252)



def benchmark_max_drawdown(prices):
    """
    Maximum drawdown
    """

    cumulative = (
        prices /
        prices.iloc[0]
    )

    running_max = cumulative.cummax()

    drawdown = (
        cumulative -
        running_max
    ) / running_max

    return drawdown.min()



def benchmark_statistics(prices):

    return {
        "Total Return": benchmark_return(prices),
        "Volatility": benchmark_volatility(prices),
        "Sharpe Ratio": benchmark_sharpe(prices),
        "Maximum Drawdown": benchmark_max_drawdown(prices)
    }