import pandas as pd

def moving_average(prices, window=20):
    return prices.rolling(window).mean()

def momentum(prices, window=10):
    return prices / prices.shift(window) - 1