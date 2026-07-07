import yfinance as yf
import pandas as pd


def download_data(ticker, start, end):

    data = yf.download(
        ticker,
        start=start,
        end=end
    )

    return data


if __name__ == "__main__":

    stock = download_data(
        "AAPL",
        "2015-01-01",
        "2025-01-01"
    )

    stock.columns = stock.columns.droplevel(1)

    print(stock.head())

    stock.to_csv("AAPL_data.csv")

    print("Data saved successfully")