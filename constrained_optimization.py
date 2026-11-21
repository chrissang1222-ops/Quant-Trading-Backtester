
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize


# ============================================================
# CONSTRAINED PORTFOLIO OPTIMIZATION ENGINE
# ============================================================

print("=" * 65)
print("CONSTRAINED PORTFOLIO OPTIMIZATION ENGINE")
print("=" * 65)


# ============================================================
# CONFIGURATION
# ============================================================

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04
MAX_WEIGHT = 0.30

ASSETS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "TSLA",
    "META",
    "SPY"
]

OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD HISTORICAL PRICE DATA
# ============================================================


def load_price_data():

    from portfolio_optimization import load_price_data as existing_loader

    print("\nLOADING HISTORICAL DATA FROM EXISTING OPTIMIZATION ENGINE")

    data = existing_loader()

    # Ensure all required assets are available
    missing_assets = [
        asset for asset in ASSETS
        if asset not in data.columns
    ]

    if missing_assets:
        raise ValueError(
            f"Missing assets: {missing_assets}"
        )

    prices = data[ASSETS].dropna()

    print("\nHISTORICAL DATA SUCCESSFULLY LOADED")
    print("-" * 65)
    print("Assets:", list(prices.columns))
    print("Observations:", len(prices))
    print("Start:", prices.index.min())
    print("End:", prices.index.max())

    return prices

# ============================================================
# 2. CALCULATE RETURNS AND STATISTICS
# ============================================================

def calculate_statistics(prices):

    returns = prices.pct_change().dropna()

    mean_returns = returns.mean() * TRADING_DAYS

    covariance_matrix = returns.cov() * TRADING_DAYS

    return returns, mean_returns, covariance_matrix


# ============================================================
# 3. PORTFOLIO PERFORMANCE
# ============================================================

def portfolio_performance(weights, mean_returns, covariance_matrix):

    portfolio_return = np.dot(weights, mean_returns)

    portfolio_volatility = np.sqrt(
        np.dot(
            weights.T,
            np.dot(covariance_matrix, weights)
        )
    )

    sharpe_ratio = (
        portfolio_return - RISK_FREE_RATE
    ) / portfolio_volatility

    return portfolio_return, portfolio_volatility, sharpe_ratio


# ============================================================
# 4. OPTIMIZATION OBJECTIVES
# ============================================================

def negative_sharpe(
    weights,
    mean_returns,
    covariance_matrix
):

    return -portfolio_performance(
        weights,
        mean_returns,
        covariance_matrix
    )[2]


def portfolio_volatility(
    weights,
    mean_returns,
    covariance_matrix
):

    return portfolio_performance(
        weights,
        mean_returns,
        covariance_matrix
    )[1]


# ============================================================
# 5. OPTIMIZATION ENGINE
# ============================================================

def optimize_portfolio(
    mean_returns,
    covariance_matrix,
    objective="sharpe",
    max_weight=None
):

    num_assets = len(mean_returns)

    initial_weights = np.array(
        [1 / num_assets] * num_assets
    )

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1
        }
    ]

    if objective == "sharpe":

        objective_function = negative_sharpe

    elif objective == "volatility":

        objective_function = portfolio_volatility

    else:

        raise ValueError("Unknown optimization objective.")

    if max_weight is None:

        bounds = tuple(
            (0, 1) for _ in range(num_assets)
        )

    else:

        if max_weight * num_assets < 1:
            raise ValueError(
                "Maximum weight is too restrictive."
            )

        bounds = tuple(
            (0, max_weight) for _ in range(num_assets)
        )

    result = minimize(
        objective_function,
        initial_weights,
        args=(mean_returns, covariance_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12
        }
    )

    if not result.success:

        raise RuntimeError(
            f"Optimization failed: {result.message}"
        )

    return result.x


# ============================================================
# 6. CALCULATE PORTFOLIO METRICS
# ============================================================

def calculate_metrics(weights, returns):

    portfolio_returns = returns.dot(weights)

    annual_return = portfolio_returns.mean() * TRADING_DAYS

    annual_volatility = (
        portfolio_returns.std() * np.sqrt(TRADING_DAYS)
    )

    sharpe = (
        annual_return - RISK_FREE_RATE
    ) / annual_volatility

    cumulative_returns = (
        1 + portfolio_returns
    ).cumprod()

    running_max = cumulative_returns.cummax()

    drawdowns = (
        cumulative_returns / running_max
    ) - 1

    max_drawdown = drawdowns.min()

    return {
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": max_drawdown
    }


# ============================================================
# 7. CREATE ALL FIVE PORTFOLIOS
# ============================================================

def build_portfolios(
    returns,
    mean_returns,
    covariance_matrix
):

    portfolios = {}

    print("\nOPTIMIZING PORTFOLIOS")
    print("-" * 65)

    # Unconstrained maximum Sharpe

    portfolios["Maximum Sharpe"] = optimize_portfolio(
        mean_returns,
        covariance_matrix,
        objective="sharpe"
    )

    print("Unconstrained maximum Sharpe completed.")

    # Constrained maximum Sharpe

    portfolios["Constrained Maximum Sharpe"] = optimize_portfolio(
        mean_returns,
        covariance_matrix,
        objective="sharpe",
        max_weight=MAX_WEIGHT
    )

    print("30% maximum allocation Sharpe completed.")

    # Constrained minimum volatility

    portfolios["Constrained Minimum Volatility"] = optimize_portfolio(
        mean_returns,
        covariance_matrix,
        objective="volatility",
        max_weight=MAX_WEIGHT
    )

    print("30% maximum allocation minimum volatility completed.")

    # Equal-weight portfolio

    portfolios["Equal Weight"] = np.array(
        [1 / len(ASSETS)] * len(ASSETS)
    )

    print("Equal-weight portfolio completed.")

    # SPY benchmark

    spy_weights = np.zeros(len(ASSETS))
    spy_weights[ASSETS.index("SPY")] = 1.0

    portfolios["SPY Benchmark"] = spy_weights

    print("SPY benchmark completed.")

    return portfolios


# ============================================================
# 8. COMPARE PORTFOLIO PERFORMANCE
# ============================================================

def compare_portfolios(portfolios, returns):

    results = {}

    for name, weights in portfolios.items():

        results[name] = calculate_metrics(
            weights,
            returns
        )

    performance_df = pd.DataFrame(results).T

    performance_df.index.name = "Portfolio"

    performance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "constrained_portfolio_performance.csv"
        )
    )

    print("\nPORTFOLIO PERFORMANCE COMPARISON")
    print("=" * 90)

    print(
        performance_df.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return performance_df


# ============================================================
# 9. PORTFOLIO ALLOCATION TABLE
# ============================================================

def create_allocation_table(portfolios):

    allocation_df = pd.DataFrame(
        portfolios,
        index=ASSETS
    ).T

    allocation_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "constrained_portfolio_allocations.csv"
        )
    )

    print("\nPORTFOLIO ALLOCATIONS")
    print("=" * 90)

    print(
        allocation_df.to_string(
            float_format=lambda x: f"{x:.2%}"
        )
    )

    return allocation_df


# ============================================================
# 10. PERFORMANCE VISUALIZATION
# ============================================================

def plot_performance(portfolios, returns):

    plt.figure(figsize=(14, 8))

    for name, weights in portfolios.items():

        portfolio_returns = returns.dot(weights)

        cumulative_returns = (
            1 + portfolio_returns
        ).cumprod()

        plt.plot(
            cumulative_returns.index,
            cumulative_returns,
            label=name,
            linewidth=2
        )

    plt.title(
        "Portfolio Performance Comparison",
        fontsize=16
    )

    plt.xlabel("Date")
    plt.ylabel("Growth of $1")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "constrained_portfolio_performance.png"
        ),
        dpi=300
    )

    plt.show()


# ============================================================
# 11. ALLOCATION VISUALIZATION
# ============================================================

def plot_allocations(allocation_df):

    allocation_df.plot(
        kind="bar",
        stacked=True,
        figsize=(15, 8),
        width=0.8
    )

    plt.title(
        "Portfolio Asset Allocation Comparison",
        fontsize=16
    )

    plt.xlabel("Portfolio")
    plt.ylabel("Allocation")

    plt.xticks(rotation=20)
    plt.legend(
        title="Assets",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "constrained_portfolio_allocations.png"
        ),
        dpi=300
    )

    plt.show()


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():

    # Load data

    prices = load_price_data()

    print("\nDATA SUMMARY")
    print("-" * 65)

    print("Assets:", list(prices.columns))
    print("Observations:", len(prices))
    print("Start:", prices.index.min())
    print("End:", prices.index.max())

    # Calculate returns

    returns, mean_returns, covariance_matrix = (
        calculate_statistics(prices)
    )

    # Build portfolios

    portfolios = build_portfolios(
        returns,
        mean_returns,
        covariance_matrix
    )

    # Compare performance

    performance_df = compare_portfolios(
        portfolios,
        returns
    )

    # Create allocation table

    allocation_df = create_allocation_table(
        portfolios
    )

    # Visualizations

    plot_performance(
        portfolios,
        returns
    )

    plot_allocations(
        allocation_df
    )

    print("\n" + "=" * 65)
    print("CONSTRAINED PORTFOLIO OPTIMIZATION COMPLETED")
    print("=" * 65)

    print("\nGenerated files:")
    print("- results/constrained_portfolio_performance.csv")
    print("- results/constrained_portfolio_allocations.csv")
    print("- results/constrained_portfolio_performance.png")
    print("- results/constrained_portfolio_allocations.png")


if __name__ == "__main__":

    main()
