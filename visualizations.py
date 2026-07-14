import matplotlib.pyplot as plt


def historical_price_chart(prices):

    plt.figure(figsize=(10,5))

    plt.plot(prices)

    plt.title("Historical Price Movement")
    plt.xlabel("Trading Days")
    plt.ylabel("Price ($)")

    plt.grid(True)

    plt.savefig("historical_price.png", dpi=300)

    plt.close()



def return_distribution_chart(returns):

    plt.figure(figsize=(10,5))

    plt.hist(returns, bins=50)

    plt.title("Daily Return Distribution")
    plt.xlabel("Daily Returns")
    plt.ylabel("Frequency")

    plt.grid(True)

    plt.savefig("return_distribution.png", dpi=300)

    plt.close()



def monte_carlo_chart(simulations):

    plt.figure(figsize=(10,5))

    for path in simulations[:100]:
        plt.plot(path)

    plt.title("Monte Carlo Simulation")
    plt.xlabel("Trading Days")
    plt.ylabel("Simulated Price ($)")

    plt.grid(True)

    plt.savefig(
        "monte_carlo_simulation.png",
        dpi=300
    )

    plt.close()



def equity_curve_chart(portfolio):

    plt.figure(figsize=(10,5))

    plt.plot(portfolio["Total"])

    plt.title("Portfolio Equity Curve")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")

    plt.grid(True)

    plt.savefig(
        "equity_curve.png",
        dpi=300
    )

    plt.close()


def strategy_chart(signals):

    plt.figure(figsize=(12,6))

    plt.plot(
        signals["Price"],
        label="AAPL Price"
    )

    plt.plot(
        signals["Short_MA"],
        label="50 Day MA"
    )

    plt.plot(
        signals["Long_MA"],
        label="200 Day MA"
    )

    buys = signals[signals["Position"] == 1]
    sells = signals[signals["Position"] == -1]

    plt.scatter(
        buys.index,
        buys["Price"],
        marker="^",
        label="Buy"
    )

    plt.scatter(
        sells.index,
        sells["Price"],
        marker="v",
        label="Sell"
    )

    plt.title("Moving Average Crossover Strategy")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        "strategy_signals.png",
        dpi=300
    )

    plt.close()


def benchmark_chart(comparison):

    plt.figure(figsize=(12,6))

    plt.plot(
        comparison["Strategy"],
        label="Moving Average Strategy"
    )

    plt.plot(
        comparison["Buy & Hold"],
        label="Buy & Hold"
    )

    plt.title(
        "Strategy Performance vs Buy & Hold Benchmark"
    )

    plt.xlabel("Date")
    plt.ylabel("Growth of $1")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        "benchmark_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()