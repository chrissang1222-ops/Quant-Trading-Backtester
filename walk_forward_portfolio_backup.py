
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize


# ============================================================
# WALK-FORWARD PORTFOLIO OPTIMIZATION ENGINE
# ============================================================

print("=" * 70)
print("OUT-OF-SAMPLE WALK-FORWARD PORTFOLIO OPTIMIZATION ENGINE")
print("=" * 70)


# ============================================================
# 1. CONFIGURATION
# ============================================================

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

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04

TRAINING_WINDOW = 1260
TESTING_WINDOW = 252

MAX_WEIGHT = 0.30

OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD HISTORICAL DATA
# ============================================================

def load_price_data():

    from portfolio_optimization import (
        load_price_data as existing_loader
    )

    print("\nLOADING HISTORICAL PRICE DATA")

    prices = existing_loader()

    missing_assets = [
        asset for asset in ASSETS
        if asset not in prices.columns
    ]

    if missing_assets:
        raise ValueError(
            f"Missing required assets: {missing_assets}"
        )

    prices = prices[ASSETS].copy()

    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.dropna()

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates are not sorted.")

    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive.")

    print("\nDATA LOADED SUCCESSFULLY")
    print("-" * 60)
    print("Assets:", list(prices.columns))
    print("Observations:", len(prices))
    print("Start:", prices.index.min())
    print("End:", prices.index.max())

    return prices


# ============================================================
# 3. PORTFOLIO OPTIMIZATION OBJECTIVES
# ============================================================

def portfolio_statistics(
    weights,
    mean_returns,
    covariance_matrix
):

    expected_return = np.dot(
        weights,
        mean_returns
    )

    volatility = np.sqrt(
        weights.T @ covariance_matrix @ weights
    )

    sharpe = (
        expected_return - RISK_FREE_RATE
    ) / volatility

    return expected_return, volatility, sharpe


def negative_sharpe(
    weights,
    mean_returns,
    covariance_matrix
):

    return -portfolio_statistics(
        weights,
        mean_returns,
        covariance_matrix
    )[2]


def portfolio_volatility(
    weights,
    mean_returns,
    covariance_matrix
):

    return portfolio_statistics(
        weights,
        mean_returns,
        covariance_matrix
    )[1]


# ============================================================
# 4. OPTIMIZATION FUNCTION
# ============================================================

def optimize_portfolio(
    mean_returns,
    covariance_matrix,
    objective="sharpe",
    max_weight=None
):

    num_assets = len(mean_returns)

    initial_weights = np.full(
        num_assets,
        1 / num_assets
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

        bounds = [(0, 1)] * num_assets

    else:

        if max_weight * num_assets < 1:
            raise ValueError(
                "Maximum allocation makes the problem infeasible."
            )

        bounds = [
            (0, max_weight)
            for _ in range(num_assets)
        ]

    result = minimize(
        objective_function,
        initial_weights,
        args=(mean_returns, covariance_matrix),
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
            f"Portfolio optimization failed: {result.message}"
        )

    weights = result.x

    validate_weights(
        weights,
        max_weight=max_weight
    )

    return weights


# ============================================================
# 5. PORTFOLIO VALIDATION
# ============================================================

def validate_weights(weights, max_weight=None):

    if not np.all(np.isfinite(weights)):
        raise ValueError("Portfolio contains invalid weights.")

    if not np.isclose(np.sum(weights), 1, atol=1e-6):
        raise ValueError("Portfolio weights do not sum to 100%.")

    if np.any(weights < -1e-6):
        raise ValueError("Negative weights are not permitted.")

    if max_weight is not None:

        if np.any(weights > max_weight + 1e-6):
            raise ValueError(
                "Portfolio exceeds the maximum asset allocation."
            )

    return True


# ============================================================
# 6. BUILD PORTFOLIO STRATEGIES
# ============================================================

def build_portfolios(
    training_returns,
    mean_returns,
    covariance_matrix
):

    portfolios = {}

    # Unconstrained maximum Sharpe

    portfolios["Maximum Sharpe"] = optimize_portfolio(
        mean_returns,
        covariance_matrix,
        objective="sharpe"
    )

    # Constrained maximum Sharpe

    portfolios["Constrained Maximum Sharpe"] = optimize_portfolio(
        mean_returns,
        covariance_matrix,
        objective="sharpe",
        max_weight=MAX_WEIGHT
    )

    # Constrained minimum volatility

    portfolios["Constrained Minimum Volatility"] = optimize_portfolio(
        mean_returns,
        covariance_matrix,
        objective="volatility",
        max_weight=MAX_WEIGHT
    )

    # Equal weight

    portfolios["Equal Weight"] = np.full(
        len(ASSETS),
        1 / len(ASSETS)
    )

    # SPY benchmark

    spy_weights = np.zeros(len(ASSETS))

    spy_weights[ASSETS.index("SPY")] = 1.0

    portfolios["SPY Benchmark"] = spy_weights

    return portfolios


# ============================================================
# 7. WALK-FORWARD OPTIMIZATION
# ============================================================


def simulate_buy_and_hold_period(
    testing_returns,
    initial_weights
):

    current_weights = np.array(
        initial_weights,
        dtype=float
    ).copy()

    daily_portfolio_returns = []

    for _, daily_returns in testing_returns.iterrows():

        daily_returns = daily_returns.to_numpy()

        portfolio_return = np.dot(
            current_weights,
            daily_returns
        )

        daily_portfolio_returns.append(
            portfolio_return
        )

        # Allow allocations to drift with asset performance

        current_weights = (
            current_weights * (1 + daily_returns)
        ) / (1 + portfolio_return)

    return pd.Series(
        daily_portfolio_returns,
        index=testing_returns.index
    )

def run_walk_forward(prices):

    returns = prices.pct_change().dropna()

    if len(returns) <= TRAINING_WINDOW:
        raise ValueError(
            "Not enough observations for the training window."
        )

    all_portfolio_returns = {
        name: []
        for name in [
            "Maximum Sharpe",
            "Constrained Maximum Sharpe",
            "Constrained Minimum Volatility",
            "Equal Weight",
            "SPY Benchmark"
        ]
    }

    all_allocations = []
    fold_records = []

    print("\nSTARTING WALK-FORWARD OPTIMIZATION")
    print("=" * 70)

    fold_number = 1

    for test_start in range(
        TRAINING_WINDOW,
        len(returns),
        TESTING_WINDOW
    ):

        train_start = max(
            0,
            test_start - TRAINING_WINDOW
        )

        test_end = min(
            test_start + TESTING_WINDOW,
            len(returns)
        )

        training_returns = returns.iloc[
            train_start:test_start
        ]

        testing_returns = returns.iloc[
            test_start:test_end
        ]

        if len(testing_returns) == 0:
            break

        # Estimate parameters using training data only

        mean_returns = (
            training_returns.mean() * TRADING_DAYS
        )

        covariance_matrix = (
            training_returns.cov() * TRADING_DAYS
        )

        # Optimize using historical training data

        portfolios = build_portfolios(
            training_returns,
            mean_returns,
            covariance_matrix
        )

        print(f"\nFOLD {fold_number}")
        print("-" * 60)

        print(
            "Training:",
            training_returns.index[0],
            "to",
            training_returns.index[-1]
        )

        print(
            "Testing:",
            testing_returns.index[0],
            "to",
            testing_returns.index[-1]
        )

        print(
            "Training observations:",
            len(training_returns)
        )

        print(
            "Testing observations:",
            len(testing_returns)
        )

        # Test the frozen weights on unseen returns

        for name, weights in portfolios.items():

            test_portfolio_returns = simulate_buy_and_hold_period(
    		testing_returns,
    		weights
	)		

            all_portfolio_returns[name].append(
                test_portfolio_returns
            )

            allocation_record = {
                "Fold": fold_number,
                "Training Start": training_returns.index[0],
                "Training End": training_returns.index[-1],
                "Testing Start": testing_returns.index[0],
                "Testing End": testing_returns.index[-1],
                "Portfolio": name
            }

            for asset, weight in zip(ASSETS, weights):
                allocation_record[asset] = weight

            all_allocations.append(
                allocation_record
            )

        fold_records.append({
            "Fold": fold_number,
            "Training Start": training_returns.index[0],
            "Training End": training_returns.index[-1],
            "Testing Start": testing_returns.index[0],
            "Testing End": testing_returns.index[-1],
            "Training Observations": len(training_returns),
            "Testing Observations": len(testing_returns)
        })

        fold_number += 1

    # Combine all out-of-sample periods

    combined_returns = {}

    for name, return_periods in all_portfolio_returns.items():

        combined_returns[name] = pd.concat(
            return_periods
        ).sort_index()

    oos_returns = pd.DataFrame(
        combined_returns
    )

    allocations_df = pd.DataFrame(
        all_allocations
    )

    folds_df = pd.DataFrame(
        fold_records
    )

    return oos_returns, allocations_df, folds_df


# ============================================================
# 8. OUT-OF-SAMPLE PERFORMANCE METRICS
# ============================================================

def calculate_performance(portfolio_returns):

    portfolio_returns = portfolio_returns.dropna()

    wealth = (1 + portfolio_returns).cumprod()

    total_return = wealth.iloc[-1] - 1

    years = len(portfolio_returns) / TRADING_DAYS

    annualized_return = (
        wealth.iloc[-1] ** (1 / years) - 1
    )

    annualized_volatility = (
        portfolio_returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

    sharpe_ratio = (
        portfolio_returns.mean() * TRADING_DAYS
        - RISK_FREE_RATE
    ) / annualized_volatility

    downside_returns = np.minimum(
        portfolio_returns.to_numpy(),
        0
    )

    downside_deviation = (
        np.sqrt(np.mean(downside_returns ** 2))
        * np.sqrt(TRADING_DAYS)
    )

    sortino_ratio = (
        portfolio_returns.mean() * TRADING_DAYS
        - RISK_FREE_RATE
    ) / downside_deviation if downside_deviation > 0 else np.nan

    running_max = wealth.cummax()

    drawdowns = (
        wealth / running_max
    ) - 1

    maximum_drawdown = drawdowns.min()

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Maximum Drawdown": maximum_drawdown
    }


# ============================================================
# 9. PERFORMANCE COMPARISON
# ============================================================

def compare_performance(oos_returns):

    results = {}

    for name in oos_returns.columns:

        results[name] = calculate_performance(
            oos_returns[name]
        )

    performance_df = pd.DataFrame(
        results
    ).T

    performance_df.index.name = "Portfolio"

    performance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "walk_forward_performance.csv"
        )
    )

    print("\nOUT-OF-SAMPLE PERFORMANCE COMPARISON")
    print("=" * 100)

    print(
        performance_df.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return performance_df


# ============================================================
# 10. EQUITY CURVE VISUALIZATION
# ============================================================

def plot_equity_curves(oos_returns):

    plt.figure(figsize=(15, 8))

    for name in oos_returns.columns:

        equity_curve = (
            1 + oos_returns[name]
        ).cumprod()

        plt.plot(
            equity_curve.index,
            equity_curve,
            label=name,
            linewidth=2
        )

    plt.title(
        "Out-of-Sample Walk-Forward Portfolio Performance",
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
            "walk_forward_equity_curves.png"
        ),
        dpi=300
    )

    plt.show()


# ============================================================
# 11. WALK-FORWARD ALLOCATION VISUALIZATION
# ============================================================

def plot_allocation_history(allocations_df):

    portfolios = allocations_df["Portfolio"].unique()

    for portfolio in portfolios:

        portfolio_data = allocations_df[
            allocations_df["Portfolio"] == portfolio
        ]

        weight_data = portfolio_data[
            ASSETS
        ]

        plt.figure(figsize=(13, 6))

        plt.stackplot(
            portfolio_data["Testing Start"],
            *[
                weight_data[asset].to_numpy()
                for asset in ASSETS
            ],
            labels=ASSETS,
            alpha=0.85
        )

        plt.title(
            f"Walk-Forward Allocation History: {portfolio}",
            fontsize=14
        )

        plt.xlabel("Rebalancing Date")
        plt.ylabel("Portfolio Weight")

        plt.ylim(0, 1)
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1)
        )

        plt.grid(alpha=0.2)
        plt.tight_layout()

        safe_name = (
            portfolio.lower()
            .replace(" ", "_")
        )

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"walk_forward_allocations_{safe_name}.png"
            ),
            dpi=300
        )

        plt.show()


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():

    # Load prices

    prices = load_price_data()

    # Run walk-forward optimization

    oos_returns, allocations_df, folds_df = (
        run_walk_forward(prices)
    )

    # Calculate performance

    performance_df = compare_performance(
        oos_returns
    )

    # Export out-of-sample returns

    oos_returns.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "walk_forward_daily_returns.csv"
        )
    )

    # Export allocation history

    allocations_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "walk_forward_allocation_history.csv"
        ),
        index=False
    )

    # Export fold information

    folds_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "walk_forward_folds.csv"
        ),
        index=False
    )

    # Generate equity curves

    plot_equity_curves(
        oos_returns
    )

    # Generate allocation charts

    plot_allocation_history(
        allocations_df
    )

    print("\n" + "=" * 70)
    print("WALK-FORWARD OPTIMIZATION COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")

    print("- results/walk_forward_performance.csv")
    print("- results/walk_forward_daily_returns.csv")
    print("- results/walk_forward_allocation_history.csv")
    print("- results/walk_forward_folds.csv")
    print("- results/walk_forward_equity_curves.png")
    print("- results/walk_forward_allocations_*.png")


if __name__ == "__main__":

    main()
