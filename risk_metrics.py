import numpy as np
import pandas as pd


def value_at_risk(returns, confidence=0.95):
    """
    Historical Value at Risk (VaR)

    Measures the worst expected loss
    at a given confidence level.
    """

    return np.percentile(
        returns,
        (1 - confidence) * 100
    )



def conditional_var(returns, confidence=0.95):
    """
    Conditional Value at Risk (CVaR)
    Also called Expected Shortfall.

    Measures average loss beyond VaR.
    """

    var = value_at_risk(
        returns,
        confidence
    )

    losses = returns[returns <= var]

    return losses.mean()



def beta(strategy_returns, benchmark_returns):
    """
    Measures sensitivity to benchmark.

    Beta > 1:
    More volatile than market

    Beta < 1:
    Less volatile than market
    """

    covariance = np.cov(
        strategy_returns,
        benchmark_returns
    )[0][1]

    benchmark_variance = np.var(
        benchmark_returns
    )

    return covariance / benchmark_variance



def alpha(strategy_returns, benchmark_returns, risk_free=0):
    """
    Jensen's Alpha

    Measures excess return
    compared with benchmark.
    """

    strategy_return = (
        strategy_returns.mean()
        * 252
    )

    benchmark_return = (
        benchmark_returns.mean()
        * 252
    )

    b = beta(
        strategy_returns,
        benchmark_returns
    )

    return (
        strategy_return
        -
        (
            risk_free
            +
            b * (benchmark_return - risk_free)
        )
    )



def tracking_error(strategy_returns, benchmark_returns):
    """
    Measures difference between
    strategy and benchmark returns.
    """

    difference = (
        strategy_returns
        -
        benchmark_returns
    )

    return difference.std() * np.sqrt(252)



def information_ratio(strategy_returns, benchmark_returns):
    """
    Active return per unit of tracking error.
    """

    active_return = (
        strategy_returns.mean()
        -
        benchmark_returns.mean()
    ) * 252


    te = tracking_error(
        strategy_returns,
        benchmark_returns
    )

    return active_return / te