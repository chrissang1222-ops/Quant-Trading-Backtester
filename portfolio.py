import pandas as pd


def backtest(
    signals,
    initial_cash=100000,
    commission=1,
    slippage=0.0005
):

    portfolio = pd.DataFrame(index=signals.index)

    cash = initial_cash
    position = 0

    holdings = []
    cash_history = []


    for date, row in signals.iterrows():

        price = row["Price"]
        signal = row["Position"]


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


        elif signal == -1 and position > 0:

            execution_price = price * (1 - slippage)

            proceeds = (
                position * execution_price
                - commission
            )

            cash += proceeds
            position = 0


        holdings.append(position)
        cash_history.append(cash)


    portfolio["Holdings"] = holdings

    portfolio["Stock Value"] = (
        portfolio["Holdings"] * signals["Price"]
    )

    portfolio["Cash"] = cash_history

    portfolio["Total"] = (
        portfolio["Cash"]
        +
        portfolio["Stock Value"]
    )


    return portfolio



def compare_to_buy_hold(portfolio, prices):

    results = pd.DataFrame(index=prices.index)

    results["Market Return"] = prices.pct_change()

    results["Buy & Hold"] = (
        1 + results["Market Return"]
    ).cumprod()

    results["Strategy"] = (
        portfolio["Total"]
        /
        portfolio["Total"].iloc[0]
    )

    return results