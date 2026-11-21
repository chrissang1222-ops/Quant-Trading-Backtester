


import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIGURATION
# ============================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "TSLA",
    "META",
    "SPY"
]

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04
FRONTIER_POINTS = 50

DATA_FILE = "portfolio_prices.csv"
RESULTS_DIR = "results"


# ============================================================
# 1. LOAD HISTORICAL PRICE DATA
# ============================================================

def load_price_data():

    print("\nLOADING HISTORICAL DATA")
    print("-" * 50)

    # First try the combined historical price dataset.
    if os.path.exists(DATA_FILE):

        prices = pd.read_csv(
            DATA_FILE,
            index_col=0,
            parse_dates=True
        )

        prices = prices[TICKERS]

    else:

        # Otherwise load the individual asset CSV files.
        asset_data = {}

        for ticker in TICKERS:

            filename = f"{ticker}_data.csv"

            if not os.path.exists(filename):
                raise FileNotFoundError(
                    f"Missing data file: {filename}"
                )

            df = pd.read_csv(filename)

            # Identify the date column.
            date_column = next(
                (
                    col for col in df.columns
                    if col.lower() in ["date", "datetime"]
                ),
                None
            )

            if date_column is None:
                raise ValueError(
                    f"No date column found in {filename}"
                )

            df[date_column] = pd.to_datetime(
                df[date_column]
            )

            df = df.set_index(date_column)

            # Prefer adjusted closing prices.
            price_column = next(
                (
                    col for col in df.columns
                    if col.lower() in [
                        "adj close",
                        "adj_close",
                        "adjusted close"
                    ]
                ),
                None
            )

            if price_column is None:
                price_column = next(
                    (
                        col for col in df.columns
                        if col.lower() == "close"
                    ),
                    None
                )

            if price_column is None:
                raise ValueError(
                    f"No closing price column found in {filename}"
                )

            asset_data[ticker] = df[price_column]

        prices = pd.DataFrame(asset_data)

    prices.index = pd.to_datetime(prices.index)

    prices = prices.sort_index()

    prices = prices.apply(
        pd.to_numeric,
        errors="coerce"
    )

    prices = prices.replace(
        [np.inf, -np.inf], np.nan
    )

    prices = prices.dropna()

    if prices.empty:
        raise ValueError("Historical price dataset is empty.")

    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive.")

    if len(prices) < 252:
        raise ValueError(
            "At least one year of historical data is required."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    prices.to_csv(DATA_FILE)

    print("HISTORICAL DATA LOADED")
    print("-" * 50)
    print("Assets:", list(prices.columns))
    print("Observations:", len(prices))
    print("Start:", prices.index.min())
    print("End:", prices.index.max())
    print("\nMissing values:")
    print(prices.isna().sum())

    return prices


# ============================================================
# 2. CALCULATE RETURNS AND COVARIANCE
# ============================================================

def calculate_portfolio_statistics(prices):

    print("\nCALCULATING PORTFOLIO STATISTICS")
    print("-" * 50)

    daily_returns = prices.pct_change().dropna()

    expected_returns = (
        daily_returns.mean() * TRADING_DAYS
    )

    covariance_matrix = (
        daily_returns.cov() * TRADING_DAYS
    )

    correlation_matrix = daily_returns.corr()

    print("\nANNUALIZED EXPECTED RETURNS")
    print(expected_returns.map(lambda x: f"{x:.2%}"))

    return (
        daily_returns,
        expected_returns,
        covariance_matrix,
        correlation_matrix
    )


# ============================================================
# 3. PORTFOLIO PERFORMANCE
# ============================================================

def portfolio_performance(
    weights,
    expected_returns,
    covariance
):

    portfolio_return = np.dot(
        weights,
        expected_returns
    )

    portfolio_variance = np.dot(
        weights.T,
        np.dot(covariance, weights)
    )

    portfolio_volatility = np.sqrt(
        max(portfolio_variance, 0)
    )

    if portfolio_volatility == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = (
            portfolio_return - RISK_FREE_RATE
        ) / portfolio_volatility

    return (
        portfolio_return,
        portfolio_volatility,
        sharpe_ratio
    )


# ============================================================
# 4. PORTFOLIO OPTIMIZATION
# ============================================================

def optimize_portfolio(
    expected_returns,
    covariance,
    objective
):

    number_assets = len(expected_returns)

    initial_weights = np.ones(
        number_assets
    ) / number_assets

    bounds = tuple(
        (0, 1) for _ in range(number_assets)
    )

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1
        }
    ]

    if objective == "max_sharpe":

        def loss(weights):

            return -portfolio_performance(
                weights,
                expected_returns,
                covariance
            )[2]

    elif objective == "min_volatility":

        def loss(weights):

            return portfolio_performance(
                weights,
                expected_returns,
                covariance
            )[1]

    else:

        raise ValueError("Invalid optimization objective.")

    result = minimize(
        loss,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-10
        }
    )

    if not result.success:
        raise RuntimeError(
            f"Optimization failed: {result.message}"
        )

    return result.x


# ============================================================
# 5. EFFICIENT FRONTIER
# ============================================================

def generate_efficient_frontier(
    expected_returns,
    covariance
):

    number_assets = len(expected_returns)

    minimum_return = expected_returns.min()
    maximum_return = expected_returns.max()

    target_returns = np.linspace(
        minimum_return,
        maximum_return,
        FRONTIER_POINTS
    )

    frontier_returns = []
    frontier_volatility = []

    bounds = tuple(
        (0, 1) for _ in range(number_assets)
    )

    initial_weights = np.ones(
        number_assets
    ) / number_assets

    for target in target_returns:

        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1
            },
            {
                "type": "eq",
                "fun": lambda w, target=target:
                    np.dot(w, expected_returns) - target
            }
        ]

        result = minimize(
            lambda w: np.sqrt(
                max(
                    np.dot(w.T, np.dot(covariance, w)),
                    0
                )
            ),
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000}
        )

        if result.success:

            frontier_returns.append(target)

            frontier_volatility.append(
                result.fun
            )

        else:

            frontier_returns.append(np.nan)
            frontier_volatility.append(np.nan)

    return (
        np.array(frontier_returns),
        np.array(frontier_volatility)
    )


# ============================================================
# 6. SAVE PORTFOLIO ALLOCATIONS
# ============================================================

def save_allocation(
    weights,
    tickers,
    portfolio_name
):

    allocation = pd.DataFrame({
        "Ticker": tickers,
        "Weight": weights,
        "Allocation_Percentage": weights * 100
    })

    allocation = allocation.sort_values(
        by="Weight",
        ascending=False
    )

    filename = os.path.join(
        RESULTS_DIR,
        f"{portfolio_name}_allocation.csv"
    )

    allocation.to_csv(
        filename,
        index=False
    )

    print("\nASSET ALLOCATION")
    print("-" * 40)

    for _, row in allocation.iterrows():

        print(
            f"{row['Ticker']:<8} "
            f"{row['Weight']:>10.2%}"
        )

    print("\nSaved:", filename)

    return allocation


# ============================================================
# 7. VISUALIZE EFFICIENT FRONTIER
# ============================================================

def plot_efficient_frontier(
    frontier_returns,
    frontier_volatility,
    max_sharpe,
    min_volatility
):

    plt.figure(figsize=(12, 8))

    plt.plot(
        frontier_volatility,
        frontier_returns,
        linewidth=2,
        label="Efficient Frontier"
    )

    plt.scatter(
        max_sharpe[1],
        max_sharpe[0],
        marker="*",
        s=300,
        label="Maximum Sharpe Portfolio"
    )

    plt.scatter(
        min_volatility[1],
        min_volatility[0],
        marker="o",
        s=150,
        label="Minimum Volatility Portfolio"
    )

    plt.xlabel(
        "Annualized Volatility"
    )

    plt.ylabel(
        "Annualized Expected Return"
    )

    plt.title(
        "Mean-Variance Efficient Frontier"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = os.path.join(
        RESULTS_DIR,
        "efficient_frontier.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()

    print("\nEfficient frontier saved:", filename)


# ============================================================
# 8. CORRELATION MATRIX
# ============================================================

def plot_correlation_matrix(correlation_matrix):

    plt.figure(figsize=(10, 8))

    plt.imshow(
        correlation_matrix,
        interpolation="nearest",
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(
        label="Correlation"
    )

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=45
    )

    plt.yticks(
        range(len(correlation_matrix.index)),
        correlation_matrix.index
    )

    plt.title("Asset Return Correlation Matrix")

    plt.tight_layout()

    filename = os.path.join(
        RESULTS_DIR,
        "correlation_matrix.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()

    print("Correlation matrix saved:", filename)


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 55)
    print("QUANTITATIVE PORTFOLIO OPTIMIZATION ENGINE")
    print("=" * 55)

    # Load data.
    prices = load_price_data()

    # Calculate statistics.
    (
        daily_returns,
        expected_returns,
        covariance_matrix,
        correlation_matrix
    ) = calculate_portfolio_statistics(prices)

    tickers = prices.columns.tolist()

    # Optimize maximum Sharpe portfolio.
    print("\nOPTIMIZING MAXIMUM SHARPE PORTFOLIO...")

    max_sharpe_weights = optimize_portfolio(
        expected_returns,
        covariance_matrix,
        "max_sharpe"
    )

    # Optimize minimum volatility portfolio.
    print("\nOPTIMIZING MINIMUM VOLATILITY PORTFOLIO...")

    min_volatility_weights = optimize_portfolio(
        expected_returns,
        covariance_matrix,
        "min_volatility"
    )

    # Calculate performance.
    max_sharpe_performance = portfolio_performance(
        max_sharpe_weights,
        expected_returns,
        covariance_matrix
    )

    min_volatility_performance = portfolio_performance(
        min_volatility_weights,
        expected_returns,
        covariance_matrix
    )

    # Display results.
    portfolios = [
        (
            "MAXIMUM SHARPE PORTFOLIO",
            max_sharpe_weights,
            max_sharpe_performance,
            "max_sharpe"
        ),
        (
            "MINIMUM VOLATILITY PORTFOLIO",
            min_volatility_weights,
            min_volatility_performance,
            "min_volatility"
        )
    ]

    summary = []

    for name, weights, performance, filename in portfolios:

        print("\n" + "=" * 55)
        print(name)
        print("=" * 55)

        print(
            f"Expected Annual Return: {performance[0]:.2%}"
        )

        print(
            f"Annualized Volatility:  {performance[1]:.2%}"
        )

        print(
            f"Sharpe Ratio:           {performance[2]:.3f}"
        )

        save_allocation(
            weights,
            tickers,
            filename
        )

        summary.append({
            "Portfolio": name,
            "Expected_Return": performance[0],
            "Volatility": performance[1],
            "Sharpe_Ratio": performance[2]
        })

    # Save portfolio comparison.
    summary_df = pd.DataFrame(summary)

    summary_path = os.path.join(
        RESULTS_DIR,
        "portfolio_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\nPORTFOLIO COMPARISON")
    print(summary_df.to_string(index=False))

    # Generate efficient frontier.
    print("\nGENERATING EFFICIENT FRONTIER...")

    frontier_returns, frontier_volatility = (
        generate_efficient_frontier(
            expected_returns,
            covariance_matrix
        )
    )

    plot_efficient_frontier(
        frontier_returns,
        frontier_volatility,
        max_sharpe_performance,
        min_volatility_performance
    )

    # Generate correlation visualization.
    plot_correlation_matrix(
        correlation_matrix
    )

    # Save historical statistics.
    expected_returns.to_csv(
        os.path.join(
            RESULTS_DIR,
            "expected_returns.csv"
        )
    )

    covariance_matrix.to_csv(
        os.path.join(
            RESULTS_DIR,
            "covariance_matrix.csv"
        )
    )

    print("\n" + "=" * 55)
    print("PORTFOLIO OPTIMIZATION COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print("\nResults saved in:", RESULTS_DIR)


# ============================================================
# 10. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
