import pandas as pd
import matplotlib.pyplot as plt


# Load data
data = pd.read_csv(
    "AAPL_data.csv",
    index_col="Date",
    parse_dates=True
)


# Calculate moving averages
data["50_MA"] = data["Close"].rolling(window=50).mean()
data["200_MA"] = data["Close"].rolling(window=200).mean()


# Trading signal
data["Signal"] = 0

data.loc[data["50_MA"] > data["200_MA"], "Signal"] = 1


# Calculate returns
data["Daily_Return"] = data["Close"].pct_change()

data["Strategy_Return"] = (
    data["Signal"].shift(1) *
    data["Daily_Return"]
)


# Calculate cumulative returns
data["Market_Return"] = (
    1 + data["Daily_Return"]
).cumprod()


data["Strategy_Returns"] = (
    1 + data["Strategy_Return"]
).cumprod()


print(data.tail())


# Plot results
plt.figure(figsize=(12,6))

plt.plot(
    data.index,
    data["Market_Return"],
    label="Buy & Hold"
)

plt.plot(
    data.index,
    data["Strategy_Returns"],
    label="Moving Average Strategy"
)

plt.legend()
plt.title("AAPL Quant Strategy Backtest")
plt.show()

# Performance Metrics

total_return = (
    data["Strategy_Returns"].iloc[-1] - 1
)

annual_return = (
    data["Strategy_Returns"].iloc[-1]
    ** (252 / len(data))
    - 1
)

volatility = (
    data["Strategy_Return"]
    .std()
    *
    (252 ** 0.5)
)

sharpe_ratio = (
    annual_return / volatility
)


# Maximum Drawdown

rolling_max = (
    data["Strategy_Returns"]
    .cummax()
)

drawdown = (
    data["Strategy_Returns"]
    /
    rolling_max
    - 1
)

max_drawdown = drawdown.min()


print("\nPerformance Report")
print("----------------------")
print(f"Total Return: {total_return:.2%}")
print(f"Annual Return: {annual_return:.2%}")
print(f"Volatility: {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")