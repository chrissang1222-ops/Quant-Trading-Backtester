import numpy as np

def sharpe_ratio(returns, risk_free_rate=0):

    excess = returns - risk_free_rate / 252

    if excess.std() == 0:
        return 0

    return np.sqrt(252) * excess.mean() / excess.std()

def volatility(returns):
    return returns.std() * np.sqrt(252)