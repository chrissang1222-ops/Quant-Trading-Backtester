import pandas as pd

def moving_average_strategy(prices, short_window=50, long_window=200):
    """
    Moving Average Crossover Strategy

    Returns a DataFrame containing:
    - Short Moving Average
    - Long Moving Average
    - Buy/Sell Signal
    """

    signals = pd.DataFrame(index=prices.index)

    signals["Price"] = prices
    signals["Short_MA"] = prices.rolling(window=short_window).mean()
    signals["Long_MA"] = prices.rolling(window=long_window).mean()

    # Buy = 1, Sell = 0
    signals["Signal"] = 0

    signals.loc[
        signals["Short_MA"] > signals["Long_MA"],
        "Signal"
    ] = 1

    # Detect changes in position
    signals["Position"] = signals["Signal"].diff()

    return signals