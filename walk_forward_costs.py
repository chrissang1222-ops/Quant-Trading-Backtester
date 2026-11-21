
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from walk_forward_portfolio import (
    ASSETS,
    TRADING_DAYS,
    RISK_FREE_RATE,
    TRAINING_WINDOW,
    TESTING_WINDOW,
    MAX_WEIGHT,
    load_price_data,
    build_portfolios,
    calculate_performance
)

# ============================================================
# CONFIGURATION
# ============================================================

TRANSACTION_COST_BPS = 5
SLIPPAGE_BPS = 5

TOTAL_COST_RATE = (
    TRANSACTION_COST_BPS + SLIPPAGE_BPS
) / 10000

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PORTFOLIOS = [
    "Maximum Sharpe",
    "Constrained Maximum Sharpe",
    "Constrained Minimum Volatility",
    "Equal Weight",
    "SPY Benchmark"
]


# ============================================================
# PORTFOLIO SIMULATION WITH TRADING COSTS
# ============================================================


def simulate_period(test_returns, target_weights, starting_weights):

    target_weights = np.asarray(target_weights, dtype=float)
    starting_weights = np.asarray(starting_weights, dtype=float)

    # Calculate initial investment or rebalance turnover.
    if np.isclose(starting_weights.sum(), 0.0):
        turnover = target_weights.sum()
    else:
        turnover = np.abs(
            target_weights - starting_weights
        ).sum() / 2

    # Transaction costs plus slippage.
    trading_cost = turnover * TOTAL_COST_RATE

    portfolio_value = 1.0

    gross_returns = []
    net_returns = []

    current_weights = target_weights.copy()

    for daily_returns in test_returns.to_numpy():

        gross_daily_return = np.dot(
            current_weights,
            daily_returns
        )

        # Apply trading costs on the first day.
        if len(net_returns) == 0:
            net_daily_return = (
                (1 - trading_cost)
                * (1 + gross_daily_return)
            ) - 1
        else:
            net_daily_return = gross_daily_return

        gross_returns.append(gross_daily_return)
        net_returns.append(net_daily_return)

        # Update portfolio value using net returns.
        portfolio_value *= (1 + net_daily_return)

        # Update portfolio weights based on market returns.
        updated_weights = current_weights * (
            1 + daily_returns
        )

        if updated_weights.sum() != 0:
            current_weights = (
                updated_weights / updated_weights.sum()
            )

    return {
        "gross_returns": pd.Series(
            gross_returns,
            index=test_returns.index
        ),
        "net_returns": pd.Series(
            net_returns,
            index=test_returns.index
        ),
        "ending_weights": current_weights,
        "turnover": turnover,
        "trading_cost": trading_cost
    }




# ============================================================
# WALK-FORWARD ENGINE
# ============================================================

def run_engine(prices):

    returns = prices.pct_change().dropna()

    previous_weights = {
        name: None for name in PORTFOLIOS
    }

    gross_results = {
        name: [] for name in PORTFOLIOS
    }

    net_results = {
        name: [] for name in PORTFOLIOS
    }

    allocation_records = []
    fold_records = []

    total_turnover = {
        name: 0.0 for name in PORTFOLIOS
    }

    total_costs = {
        name: 0.0 for name in PORTFOLIOS
    }

    print("\nSTARTING COST-AWARE WALK-FORWARD ENGINE")
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

        training = returns.iloc[
            train_start:test_start
        ]

        testing = returns.iloc[
            test_start:test_end
        ]

        mean_returns = (
            training.mean() * TRADING_DAYS
        )

        covariance = (
            training.cov() * TRADING_DAYS
        )

        portfolios = build_portfolios(
            mean_returns,
            covariance
        )

        print(f"\nFOLD {fold_number}")
        print("-" * 60)

        print(
            "Training:",
            training.index[0],
            "to",
            training.index[-1]
        )

        print(
            "Testing:",
            testing.index[0],
            "to",
            testing.index[-1]
        )

        for name, target_weights in portfolios.items():

            if previous_weights[name] is None:
                starting_weights = np.zeros(len(ASSETS))
            else:
                starting_weights = previous_weights[name]

            simulation = simulate_period(
                testing,
                target_weights,
                starting_weights
            )

            gross_results[name].append(
                simulation["gross_returns"]
            )

            net_results[name].append(
                simulation["net_returns"]
            )

            total_turnover[name] += (
                simulation["turnover"]
            )

            total_costs[name] += (
                simulation["trading_cost"]
            )

            previous_weights[name] = (
                simulation["ending_weights"]
            )

            allocation = {
                "Fold": fold_number,
                "Portfolio": name,
                "Turnover": simulation["turnover"],
                "Trading Cost": simulation["trading_cost"]
            }

            for asset, weight in zip(
                ASSETS,
                target_weights
            ):
                allocation[asset] = weight

            allocation_records.append(allocation)

        fold_records.append({
            "Fold": fold_number,
            "Training Start": training.index[0],
            "Training End": training.index[-1],
            "Testing Start": testing.index[0],
            "Testing End": testing.index[-1],
            "Training Observations": len(training),
            "Testing Observations": len(testing)
        })

        fold_number += 1

    gross_df = pd.DataFrame({
        name: pd.concat(periods).sort_index()
        for name, periods in gross_results.items()
    })

    net_df = pd.DataFrame({
        name: pd.concat(periods).sort_index()
        for name, periods in net_results.items()
    })

    return (
        gross_df,
        net_df,
        pd.DataFrame(allocation_records),
        pd.DataFrame(fold_records),
        total_turnover,
        total_costs
    )


# ============================================================
# PERFORMANCE COMPARISON
# ============================================================

def compare_results(
    gross_df,
    net_df,
    total_turnover,
    total_costs
):

    records = []

    for name in PORTFOLIOS:

        gross_metrics = calculate_performance(
            gross_df[name]
        )

        net_metrics = calculate_performance(
            net_df[name]
        )

        records.append({
            "Portfolio": name,
            "Gross Total Return": gross_metrics["Total Return"],
            "Net Total Return": net_metrics["Total Return"],
            "Gross Annual Return": gross_metrics["Annualized Return"],
            "Net Annual Return": net_metrics["Annualized Return"],
            "Net Volatility": net_metrics["Annualized Volatility"],
            "Net Sharpe": net_metrics["Sharpe Ratio"],
            "Net Sortino": net_metrics["Sortino Ratio"],
            "Net Max Drawdown": net_metrics["Maximum Drawdown"],
            "Cumulative Turnover": total_turnover[name],
            "Cumulative Trading Costs": total_costs[name]
        })

    results = pd.DataFrame(records).set_index("Portfolio")

    results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "transaction_cost_performance.csv"
        )
    )

    print("\nGROSS VS NET PERFORMANCE")
    print("=" * 110)

    print(
        results.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return results


# ============================================================
# EQUITY CURVE
# ============================================================

def plot_equity_curves(gross_df, net_df):

    plt.figure(figsize=(15, 8))

    for name in PORTFOLIOS:

        equity = (
            1 + net_df[name]
        ).cumprod()

        plt.plot(
            equity.index,
            equity,
            label=name,
            linewidth=2
        )

    plt.title(
        "Out-of-Sample Portfolio Performance After Trading Costs"
    )

    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "transaction_cost_equity_curves.png"
        ),
        dpi=300
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    prices = load_price_data()

    (
        gross_df,
        net_df,
        allocations_df,
        folds_df,
        total_turnover,
        total_costs
    ) = run_engine(prices)

    compare_results(
        gross_df,
        net_df,
        total_turnover,
        total_costs
    )

    gross_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "transaction_cost_gross_returns.csv"
        )
    )

    net_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "transaction_cost_net_returns.csv"
        )
    )

    allocations_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "transaction_cost_allocations.csv"
        ),
        index=False
    )

    folds_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "transaction_cost_folds.csv"
        ),
        index=False
    )

    plot_equity_curves(gross_df, net_df)

    print("\n" + "=" * 70)
    print("TRANSACTION COST ENGINE COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")
    print("- results/transaction_cost_performance.csv")
    print("- results/transaction_cost_gross_returns.csv")
    print("- results/transaction_cost_net_returns.csv")
    print("- results/transaction_cost_allocations.csv")
    print("- results/transaction_cost_folds.csv")
    print("- results/transaction_cost_equity_curves.png")


if __name__ == "__main__":
    main()
