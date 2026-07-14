import pandas as pd


def create_results_table(
    strategy_return,
    strategy_sharpe,
    strategy_drawdown,
    aapl_return,
    aapl_sharpe,
    aapl_drawdown,
    spy_return,
    spy_sharpe,
    spy_drawdown
):

    results = pd.DataFrame({

        "Strategy": [
            "MA 100/300",
            "AAPL Buy & Hold",
            "SPY Buy & Hold"
        ],

        "Total Return": [
            strategy_return,
            aapl_return,
            spy_return
        ],

        "Sharpe Ratio": [
            strategy_sharpe,
            aapl_sharpe,
            spy_sharpe
        ],

        "Maximum Drawdown": [
            strategy_drawdown,
            aapl_drawdown,
            spy_drawdown
        ]

    })


    return results
