import pandas as pd
import numpy as np


# ==========================
# Trade Statistics
# ==========================

def trade_statistics(trades):
    """
    Calculates performance statistics from completed trades.
    """

    if trades.empty:
        return {
            "Total Trades": 0,
            "Winning Trades": 0,
            "Losing Trades": 0,
            "Win Rate": 0,
            "Average Win": 0,
            "Average Loss": 0,
            "Total Profit/Loss": 0,
            "Average Return %": 0,
            "Average Holding Days": 0
        }

    total_trades = len(trades)

    winners = trades[trades["Net Profit/Loss"] > 0]
    losers = trades[trades["Net Profit/Loss"] < 0]

    winning_trades = len(winners)
    losing_trades = len(losers)

    win_rate = winning_trades / total_trades

    average_win = (
        winners["Net Profit/Loss"].mean()
        if winning_trades > 0 else 0
    )

    average_loss = (
        losers["Net Profit/Loss"].mean()
        if losing_trades > 0 else 0
    )

    total_profit_loss = trades["Net Profit/Loss"].sum()

    average_return = trades["Return %"].mean()

    average_holding_days = trades["Holding Days"].mean()

    return {
        "Total Trades": total_trades,
        "Winning Trades": winning_trades,
        "Losing Trades": losing_trades,
        "Win Rate": win_rate,
        "Average Win": average_win,
        "Average Loss": average_loss,
        "Total Profit/Loss": total_profit_loss,
        "Average Return %": average_return,
        "Average Holding Days": average_holding_days
    }



def print_trade_statistics(stats):

    print("\nPerformance Statistics")
    print("----------------------")

    for key, value in stats.items():

        if "Rate" in key:
            print(f"{key}: {value:.2%}")

        elif "%" in key:
            print(f"{key}: {value:.2%}")

        elif isinstance(value, float):
            print(f"{key}: {value:.2f}")

        else:
            print(f"{key}: {value}")



# ==========================
# Portfolio Performance
# ==========================

def total_return(portfolio):
    """
    Calculates total portfolio return.
    """

    total = portfolio["Total"]

    return (total.iloc[-1] / total.iloc[0]) - 1



def cagr(portfolio):
    """
    Calculates Compound Annual Growth Rate.
    """

    total = portfolio["Total"]

    years = len(total) / 252

    return (total.iloc[-1] / total.iloc[0]) ** (1 / years) - 1



def maximum_drawdown(portfolio):
    """
    Calculates maximum drawdown.
    """

    total = portfolio["Total"]

    running_max = total.cummax()

    drawdown = (total - running_max) / running_max

    return drawdown.min()


def sortino_ratio(returns):
    """
    Calculates Sortino Ratio using downside deviation.
    """

    downside_returns = returns[returns < 0]

    downside_std = downside_returns.std()

    if downside_std == 0:
        return 0

    return (
        returns.mean() / downside_std
    ) * np.sqrt(252)



def buy_and_hold_return(prices):
    """
    Calculates buy-and-hold return.
    """

    return (prices.iloc[-1] / prices.iloc[0]) - 1