from data import generate_price_series
from returns import simple_returns
from stats import sharpe_ratio
from simulation import gbm

# STEP 1: generate prices
prices = generate_price_series()

# STEP 2: compute returns
rets = simple_returns(prices)

# STEP 3: compute Sharpe ratio
sharpe = sharpe_ratio(rets)

print("Sharpe Ratio:", sharpe)

# STEP 4: run simulation
sim = gbm()
print("Final simulated price:", sim[-1])