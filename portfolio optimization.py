
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04


def load_price_data():
    tickers = [
        "AAPL", "MSFT", "NVDA", "AMZN",
        "GOOGL", "TSLA", "META", "SPY"
    ]

    prices = pd.read_csv(
        "portfolio_prices.csv",
        index_col=0,
        parse_dates=True
    )

    return prices[tickers].dropna()


def calculate_portfolio_statistics(prices):
    returns = prices.pct_change().dropna()

    expected_returns = returns.mean() * TRADING_DAYS

    covariance_matrix = returns.cov() * TRADING_DAYS

    return expected_returns, covariance_matrix


def portfolio_performance(weights, expected_returns, covariance):
    portfolio_return = np.dot(weights, expected_returns)

    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(covariance, weights))
    )

    sharpe_ratio = (
        portfolio_return - RISK_FREE_RATE
    ) / portfolio_volatility

    return portfolio_return, portfolio_volatility, sharpe_ratio


def optimize_portfolio(expected_returns, covariance, objective):
    number_assets = len(expected_returns)

    initial_weights = np.ones(number_assets) / number_assets

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    ]

    bounds = tuple((0, 1) for _ in range(number_assets))

    if objective == "max_sharpe":

        def loss(weights):
            return -portfolio_performance(
                weights, expected_returns, covariance
            )[2]

    elif objective == "min_volatility":

        def loss(weights):
            return portfolio_performance(
                weights, expected_returns, covariance
            )[1]

    else:
        raise ValueError("Unknown optimization objective")

    result = minimize(
        loss,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.x


def generate_efficient_frontier(expected_returns, covariance):
    target_returns = np.linspace(
        expected_returns.min(),
        expected_returns.max(),
        50
    )

    frontier_volatility = []

    for target in target_returns:

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {
                "type": "eq",
                "fun": lambda w, target=target:
                    np.dot(w, expected_returns) - target
            }
        ]

        result = minimize(
            lambda w: np.sqrt(
                np.dot(w.T, np.dot(covariance, w))
            ),
            np.ones(len(expected_returns)) / len(expected_returns),
            method="SLSQP",
            bounds=tuple((0, 1) for _ in expected_returns),
            constraints=constraints
        )

        if result.success:
            frontier_volatility.append(result.fun)
        else:
            frontier_volatility.append(np.nan)

    return target_returns, frontier_volatility


def main():

    prices = load_price_data()

    expected_returns, covariance = (
        calculate_portfolio_statistics(prices)
    )

    tickers = prices.columns.tolist()

    max_sharpe_weights = optimize_portfolio(
        expected_returns, covariance, "max_sharpe"
    )

    min_volatility_weights = optimize_portfolio(
        expected_returns, covariance, "min_volatility"
    )

    for name, weights in [
        ("MAXIMUM SHARPE PORTFOLIO", max_sharpe_weights),
        ("MINIMUM VOLATILITY PORTFOLIO", min_volatility_weights)
    ]:

        performance = portfolio_performance(
            weights, expected_returns, covariance
        )

        print("\n" + "=" * 45)
        print(name)
        print("=" * 45)

        print(f"Expected Return: {performance[0]:.2%}")
        print(f"Volatility:      {performance[1]:.2%}")
        print(f"Sharpe Ratio:    {performance[2]:.3f}")

        allocation = pd.Series(
            weights,
            index=tickers
        ).sort_values(ascending=False)

        print("\nASSET ALLOCATION")
        print(allocation.map(lambda x: f"{x:.2%}"))

    frontier_returns, frontier_volatility = (
        generate_efficient_frontier(
            expected_returns, covariance
        )
    )

    plt.figure(figsize=(11, 7))

    plt.plot(
        frontier_volatility,
        frontier_returns,
        label="Efficient Frontier"
    )

    plt.scatter(
        portfolio_performance(
            max_sharpe_weights,
            expected_returns,
            covariance
        )[1],
        portfolio_performance(
            max_sharpe_weights,
            expected_returns,
            covariance
        )[0],
        marker="*",
        s=250,
        label="Maximum Sharpe"
    )

    plt.scatter(
        portfolio_performance(
            min_volatility_weights,
            expected_returns,
            covariance
        )[1],
        portfolio_performance(
            min_volatility_weights,
            expected_returns,
            covariance
        )[0],
        marker="o",
        s=120,
        label="Minimum Volatility"
    )

    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Expected Return")
    plt.title("Portfolio Efficient Frontier")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("results/efficient_frontier.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
