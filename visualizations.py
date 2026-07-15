import matplotlib.pyplot as plt


def historical_price_chart(prices):

    plt.figure(figsize=(12, 6))

    plt.plot(prices.index, prices)

    plt.title("AAPL Historical Price")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.grid()

    plt.savefig(
        "results/historical_price.png",
        dpi=300
    )

    plt.close()


def return_distribution_chart(returns):

    plt.figure(figsize=(12, 6))

    plt.hist(
        returns,
        bins=50
    )

    plt.title("Return Distribution")
    plt.xlabel("Daily Returns")
    plt.ylabel("Frequency")

    plt.grid()

    plt.savefig(
        "results/return_distribution.png",
        dpi=300
    )

    plt.close()


def monte_carlo_chart(simulations):

    plt.figure(figsize=(12, 6))

    for simulation in simulations:
        plt.plot(simulation, alpha=0.1)

    plt.title("Monte Carlo Price Simulation")
    plt.xlabel("Days")
    plt.ylabel("Price")

    plt.grid()

    plt.savefig(
        "results/monte_carlo_simulation.png",
        dpi=300
    )

    plt.close()


def equity_curve_chart(portfolio):

    plt.figure(figsize=(12, 6))

    plt.plot(
        portfolio.index,
        portfolio["Total"]
    )

    plt.title("Portfolio Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")

    plt.grid()

    plt.savefig(
        "results/equity_curve.png",
        dpi=300
    )

    plt.close()


def strategy_chart(signals):

    plt.figure(figsize=(12, 6))

    plt.plot(
        signals.index,
        signals["Price"],
        label="Price"
    )

    plt.plot(
        signals.index,
        signals["Short_MA"],
        label="Short MA"
    )

    plt.plot(
        signals.index,
        signals["Long_MA"],
        label="Long MA"
    )

    plt.title("Moving Average Strategy Signals")

    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.legend()
    plt.grid()

    plt.savefig(
        "results/strategy_signals.png",
        dpi=300
    )

    plt.close()


def benchmark_chart(comparison):

    plt.figure(figsize=(12, 6))

    plt.plot(
        comparison.index,
        comparison["Strategy"],
        label="Strategy"
    )

    plt.plot(
        comparison.index,
        comparison["Buy & Hold"],
        label="Buy & Hold"
    )

    plt.title("Strategy vs Buy & Hold")

    plt.xlabel("Date")
    plt.ylabel("Growth")

    plt.legend()
    plt.grid()

    plt.savefig(
        "results/benchmark_chart.png",
        dpi=300
    )

    plt.close()


def equity_comparison_chart(comparison):

    plt.figure(figsize=(12, 6))

    plt.plot(
        comparison.index,
        comparison["Strategy"],
        label="Strategy"
    )

    plt.plot(
        comparison.index,
        comparison["Buy & Hold"],
        label="Buy & Hold"
    )

    plt.title("Equity Comparison")

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")

    plt.legend()
    plt.grid()

    plt.savefig(
        "results/equity_comparison.png",
        dpi=300
    )

    plt.close()


def full_benchmark_chart(comparison, spy_prices):

    plt.figure(figsize=(12, 6))

    plt.plot(
        comparison.index,
        comparison["Strategy"],
        label="Strategy"
    )

    plt.plot(
        comparison.index,
        comparison["Buy & Hold"],
        label="AAPL Buy & Hold"
    )

    plt.plot(
        spy_prices.index,
        spy_prices / spy_prices.iloc[0],
        label="SPY"
    )

    plt.title("Strategy vs AAPL vs S&P 500")

    plt.xlabel("Date")
    plt.ylabel("Normalized Value")

    plt.legend()
    plt.grid()

    plt.savefig(
        "results/full_benchmark_comparison.png",
        dpi=300
    )

    plt.close()
