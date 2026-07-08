from data import download_data
from returns import simple_returns
from stats import sharpe_ratio, volatility
from simulation import gbm

# STEP 1: Download historical stock data
prices = download_data(
    ticker="AAPL",
    start="2015-01-01",
    end="2025-01-01"
)

# Use the closing prices
prices = prices["Close"]

# STEP 2: Compute returns
rets = simple_returns(prices)

# STEP 3: Compute statistics
print(f"Annualized Sharpe Ratio: {sharpe_ratio(rets):.2f}")
print(f"Annualized Volatility: {volatility(rets):.2%}")

# STEP 4: Monte Carlo simulation
sim = gbm()

print(f"Final simulated price: ${sim[-1]:.2f}")