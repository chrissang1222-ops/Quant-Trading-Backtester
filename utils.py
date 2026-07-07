import numpy as np
import pandas as pd

def clean_data(df):
    return df.dropna()

def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def cumulative_returns(returns):
    return (1 + returns).cumprod()