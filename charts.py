import matplotlib.pyplot as plt


def historical_price_chart(prices):

    plt.figure(figsize=(12,6))
    plt.plot(prices.index, prices)

    plt.title("Historical Price")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.grid()
    plt.savefig("results/historical_price.png")
    plt.show()


def equity_comparison_chart(comparison):

    plt.figure(figsize=(12,6))

    plt.plot(comparison.index, comparison)

    plt.title("Equity Curve")
    plt.grid()

    plt.savefig("results/equity_curve.png")
    plt.show()


def full_benchmark_chart(strategy, spy):

    plt.figure(figsize=(12,6))

    plt.plot(strategy.index, strategy, label="Strategy")
    plt.plot(spy.index, spy, label="SPY")

    plt.title("Strategy vs SPY Benchmark")
    plt.legend()
    plt.grid()

    plt.savefig("results/benchmark_comparison.png")
    plt.show()