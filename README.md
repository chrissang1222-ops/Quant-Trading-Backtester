# Quant Trading Backtester

## Overview

A Python-based quantitative trading research framework designed to test algorithmic trading strategies using historical market data. The system collects market data, generates trading signals, simulates portfolio performance, evaluates risk-adjusted returns, compares results against benchmarks, and optimizes strategy parameters.

The project was developed to explore quantitative finance concepts including algorithmic trading, portfolio simulation, risk measurement, Monte Carlo modeling, and systematic strategy evaluation.

---

# Project Objective

The objective of this project is to create a systematic trading research pipeline that removes emotional decision-making from investment decisions by testing strategies through historical simulation.

The framework evaluates whether trading strategies can generate risk-adjusted returns compared with passive investing benchmarks.

---

# System Architecture

```
Historical Market Data
↓
Data Processing
↓
Technical Indicators
↓
Trading Strategy
↓
Portfolio Simulation
↓
Performance Analysis
=======
          ↓
Data Processing
          ↓
Technical Indicators
          ↓
Trading Strategy
          ↓
Portfolio Simulation
          ↓
Trade Analysis
          ↓
Risk & Performance Evaluation
          ↓
Strategy Optimization
```
* Historical equity data processing
* Algorithmic trading signal generation
* Portfolio tracking and simulation
* Return calculations
* Performance statistics
* Strategy backtesting framework
=======
# Features


## Data Processing

* Python
* Pandas
* NumPy
* Matplotlib
* Git/GitHub
=======
* Historical equity price data retrieval
* Data cleaning and preprocessing
* Return calculations


## Trading Strategy

* Moving average crossover strategy
* Automated buy and sell signal generation
* Strategy backtesting framework



## Future Improvements

* Add Sharpe ratio calculations
* Add maximum drawdown analysis
* Implement Monte Carlo simulations
* Add machine learning-based trading signals
* Develop portfolio optimization models



\# Visual Results



\## Portfolio Performance



!\[Equity Curve](images/equity\_curve.png)



\## Benchmark Comparison



!\[Benchmark](images/benchmark\_comparison.png)



\## Monte Carlo Simulation



!\[Monte Carlo](images/monte\_carlo\_simulation.png)



\## Trading Signals



!\[Signals](images/strategy\_signals.png)

## 
=======
## Portfolio Simulation

* Position tracking
* Cash management
* Portfolio valuation
* Trade execution simulation


## Performance Analysis

* Total return calculation
* CAGR calculation
* Sharpe ratio
* Sortino ratio
* Maximum drawdown
* Win rate analysis
* Trade statistics

## Benchmark Analysis

* Comparison against AAPL Buy & Hold
* Comparison against S&P 500 benchmark
* Alpha and beta calculations
* Tracking error analysis
* Information ratio

## Risk Analysis

* Value at Risk (VaR)
* Conditional Value at Risk (CVaR)
* Market exposure analysis

## Strategy Optimization

* Moving average parameter testing
* Performance ranking
* Sharpe ratio optimization

## Monte Carlo Simulation

* Geometric Brownian Motion price modeling
* Future price scenario simulation
* Stochastic market modeling

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* yFinance
* Git/GitHub

---

# Project Structure

```
Quant-Trading-Backtester/

├── main.py
├── data.py
├── indicators.py
├── strat.py
├── backtest.py
├── portfolio.py
├── performance.py
├── benchmark.py
├── risk_metrics.py
├── optimization.py
├── trade_log.py
├── simulation.py
├── visualizations.py
└── README.md
```

---

# Results

## Strategy Performance

| Metric           |  Value |
| ---------------- | -----: |
| Total Return     | 14.55% |
| CAGR             |  1.37% |
| Sharpe Ratio     |   0.58 |
| Sortino Ratio    |   1.33 |
| Maximum Drawdown | -4.21% |
| Total Trades     |      4 |
| Win Rate         |    75% |

---

# Benchmark Comparison

## AAPL Buy & Hold

| Metric           |   Value |
| ---------------- | ------: |
| Total Return     | 928.54% |
| Sharpe Ratio     |    0.96 |
| Maximum Drawdown | -38.52% |

## S&P 500 Benchmark

| Metric           |   Value |
| ---------------- | ------: |
| Total Return     | 239.57% |
| Sharpe Ratio     |    0.78 |
| Maximum Drawdown | -33.72% |

The results demonstrate that the trading strategy reduced downside risk but underperformed passive equity exposure during a strong bull market.

---

# Strategy Optimization

The optimization module tested multiple moving average combinations.

Best-performing configuration:

```
Short Moving Average: 100 days
Long Moving Average: 300 days
```

Performance:

```
Return: 21.39%
Sharpe Ratio: 0.80
Maximum Drawdown: -4.41%
```

---

# Visualizations

The project generates:

* Equity curve analysis
* Strategy signals
* Benchmark comparison
* Return distribution
* Monte Carlo simulations
* Historical price charts

---

# Future Improvements

Potential extensions include:

* Multi-asset portfolio testing
* Factor-based models (CAPM / Fama-French)
* Machine learning trading signals
* Portfolio optimization algorithms
* Additional technical indicators
* Transaction cost modeling
* Walk-forward testing

---

# Author

Chris Sang

