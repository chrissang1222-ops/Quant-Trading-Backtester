import pandas as pd


def backtest(
    signals,
    initial_cash=100000,
    commission=1,
    slippage=0.0005
):
    """
    Executes a moving average crossover backtest.

    signals must contain:
    - Price
    - Position
        1  = Buy
       -1  = Sell
        0  = Hold
    """

    portfolio = pd.DataFrame(index=signals.index)

    cash = initial_cash
    position = 0

    holdings = []
    cash_history = []


    for date, row in signals.iterrows():

        price = row["Price"]
        signal = row["Position"]


        # ======================
        # BUY
        # ======================
        if signal == 1 and position == 0:

            execution_price = price * (1 + slippage)

            shares = 100

            cost = (
                shares * execution_price
                + commission
            )

            if cash >= cost:
                cash -= cost
                position = shares



        # ======================
        # SELL
        # ======================
        elif signal == -1 and position > 0:

            execution_price = price * (1 - slippage)

            proceeds = (
                position * execution_price
                - commission
            )

            cash += proceeds
            position = 0



        # Track portfolio
        holdings.append(position)
        cash_history.append(cash)



    # Store results
    portfolio["Holdings"] = holdings

    portfolio["Stock Value"] = (
        portfolio["Holdings"]
        *
        signals["Price"]
    )

    portfolio["Cash"] = cash_history


    portfolio["Total"] = (
        portfolio["Cash"]
        +
        portfolio["Stock Value"]
    )


    return portfolio