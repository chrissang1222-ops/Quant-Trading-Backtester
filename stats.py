import numpy as np

def sharpe_ratio(returns, rf=0):
    excess = returns - rf
    return np.sqrt(252) * excess.mean() / excess.std()

def volatility(returns):
    return returns.std() * np.sqrt(252)