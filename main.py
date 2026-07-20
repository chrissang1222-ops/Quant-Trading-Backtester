from portfolio import (
    backtest,
    compare_to_buy_hold
)

from strat import moving_average_strategy
from data import download_data
from returns import simple_returns
from stats import sharpe_ratio, volatility
from simulation import gbm
from trade_log import generate_trade_log
from benchmark import benchmark_statistics
from optimization import optimize_strategy
from results_table import create_results_table
from visualizations import equity_comparison_chart
from walk_forward import walk_forward_analysis

from performance import (
    total_return,
    cagr,
    maximum_drawdown,
    sortino_ratio,
    buy_and_hold_return,
    trade_statistics,
    print_trade_statistics
)

from visualizations import (
    historical_price_chart,
    return_distribution_chart,
    monte_carlo_chart,
    equity_curve_chart,
    strategy_chart,
    benchmark_chart,
    full_benchmark_chart
)
from risk_metrics import (
    value_at_risk,
    conditional_var,
    beta,
    alpha,
    tracking_error,
    information_ratio
)


# STEP 1: Download historical stock data
prices = download_data(
    ticker="AAPL",
    start="2015-01-01",
    end="2025-01-01"
)

# Use the closing prices
prices = prices["Close"].squeeze()

spy_prices = download_data(
    ticker="SPY",
    start="2015-01-01",
    end="2025-01-01"
)

spy_prices = spy_prices["Close"].squeeze()

# STEP 2: Compute returns
rets = simple_returns(prices)

# STEP 3: Compute statistics
print(f"Annualized Sharpe Ratio: {sharpe_ratio(rets):.2f}")
print(f"Annualized Volatility: {volatility(rets):.2%}")
signals = moving_average_strategy(
    prices,
    short_window=100,
    long_window=300
)
portfolio = backtest(signals)
trades = generate_trade_log(signals)

print("\nTrade Log")
print(trades)

stats = trade_statistics(trades)

comparison = compare_to_buy_hold(
    portfolio,
    prices
)

print_trade_statistics(stats)

print("\nTrade Statistics")

for key, value in stats.items():
    print(f"{key}: {value}")

print(portfolio.tail())

print("\nPerformance Metrics")

print(f"Total Return: {total_return(portfolio):.2%}")

print(f"CAGR: {cagr(portfolio):.2%}")

print(f"Maximum Drawdown: {maximum_drawdown(portfolio):.2%}")

print(f"Sortino Ratio: {sortino_ratio(rets):.2f}")

strategy_returns = (
    portfolio["Total"]
    .pct_change()
    .dropna()
)

print(
    f"Strategy Sharpe Ratio: "
    f"{sharpe_ratio(strategy_returns):.2f}"
)

print(f"Buy & Hold Return: {buy_and_hold_return(prices):.2%}")

print(signals.tail())

print("\nAAPL Buy & Hold Benchmark")

aapl_benchmark = benchmark_statistics(prices)

for key, value in aapl_benchmark.items():

    if key in ["Total Return", "Maximum Drawdown", "Volatility"]:
        print(f"{key}: {value:.2%}")

    else:
        print(f"{key}: {value:.2f}")
print("\nS&P 500 Benchmark")

spy_benchmark = benchmark_statistics(spy_prices)

for key, value in spy_benchmark.items():

    if key in ["Total Return", "Maximum Drawdown", "Volatility"]:
        print(f"{key}: {value:.2%}")

    else:
        print(f"{key}: {value:.2f}")


print("\nRisk Metrics")


strategy_returns = (
    portfolio["Total"]
    .pct_change()
    .dropna()
)


spy_returns = (
    spy_prices
    .pct_change()
    .dropna()
)


# Align strategy and benchmark dates
strategy_returns, spy_returns = (
    strategy_returns
    .align(
        spy_returns,
        join="inner"
    )
)


print(
    f"Strategy Sharpe Ratio: "
    f"{sharpe_ratio(strategy_returns):.2f}"
)


print(
    f"95% VaR: "
    f"{value_at_risk(strategy_returns):.2%}"
)


print(
    f"95% CVaR: "
    f"{conditional_var(strategy_returns):.2%}"
)


print(
    f"Beta vs SPY: "
    f"{beta(strategy_returns, spy_returns):.2f}"
)


print(
    f"Alpha: "
    f"{alpha(strategy_returns, spy_returns):.2%}"
)


print(
    f"Tracking Error: "
    f"{tracking_error(strategy_returns, spy_returns):.2%}"
)


print(
    f"Information Ratio: "
    f"{information_ratio(strategy_returns, spy_returns):.2f}"
)

print("\nStrategy Optimization")

optimization_results = optimize_strategy(prices)

print(
    optimization_results.head(10)
)

print("\nWalk Forward Analysis")

walk_forward_results = walk_forward_analysis(
    prices
)

print(walk_forward_results)

print("\nStrategy vs Buy & Hold")

strategy_return = (
    comparison["Strategy"].iloc[-1] - 1
) * 100

buyhold_return = (
    comparison["Buy & Hold"].iloc[-1] - 1
) * 100

print(f"Strategy Return: {strategy_return:.2f}%")
print(f"Buy & Hold Return: {buyhold_return:.2f}%")

# STEP 4: Monte Carlo simulation
simulations = gbm()

print(f"Final simulated price: ${simulations[-1][-1]:.2f}")

historical_price_chart(prices)

return_distribution_chart(rets)

monte_carlo_chart(simulations)

equity_curve_chart(portfolio)

strategy_chart(signals)

benchmark_chart(comparison)

equity_comparison_chart(comparison)

full_benchmark_chart(comparison, spy_prices)

print("\nFinal Research Results")

results = create_results_table(

    total_return(portfolio),

    sharpe_ratio(strategy_returns),

    maximum_drawdown(portfolio),

    aapl_benchmark["Total Return"],

    aapl_benchmark["Sharpe Ratio"],

    aapl_benchmark["Maximum Drawdown"],

    spy_benchmark["Total Return"],

    spy_benchmark["Sharpe Ratio"],

    spy_benchmark["Maximum Drawdown"]

)


print(results)


results.to_csv(
    "research_results.csv",
    index=False
)


walk_forward_results.to_csv(
    "walk_forward_results.csv",
    index=False
)
