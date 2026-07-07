import numpy as np

def portfolio_return(returns):
    return returns.mean()

def portfolio_volatility(returns):
    return returns.std()