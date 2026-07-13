import pandas as pd


def generate_trade_log(
    signals,
    commission=1,
    slippage=0.0005
):

    trades = []

    position = False

    entry_date = None
    entry_price = None


    for date, row in signals.iterrows():

        price = row["Price"]


        # BUY
        if row["Position"] == 1 and position == False:

            position = True

            entry_date = date

            # Apply slippage on purchase
            entry_price = price * (1 + slippage)


        # SELL
        elif row["Position"] == -1 and position == True:

            exit_date = date

            # Apply slippage on sale
            exit_price = price * (1 - slippage)


            shares = 100


            gross_profit = (
                exit_price - entry_price
            ) * shares


            total_costs = commission * 2


            net_profit = (
                gross_profit - total_costs
            )


            trade_return = (
                net_profit /
                (entry_price * shares)
            )


            holding_period = (
                exit_date - entry_date
            ).days


            trades.append({

                "Entry Date": entry_date,

                "Exit Date": exit_date,

                "Entry Price": entry_price,

                "Exit Price": exit_price,

                "Shares": shares,

                "Gross Profit/Loss": gross_profit,

                "Costs": total_costs,

                "Net Profit/Loss": net_profit,

                "Return %": trade_return,

                "Holding Days": holding_period

            })


            position = False


    return pd.DataFrame(trades)