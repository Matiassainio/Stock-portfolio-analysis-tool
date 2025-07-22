#  Stock Portfolio Analysis Tool (Work in Progress)

This project is a personal finance tool built in Python that allows you to analyze and optimize a stock portfolio based on risk and return metrics. This project is not intended as financial advice; I also take no responsibility of the reliability of data.

## Purpose

The goal is to build a portfolio analysis bot that:
- Calculates performance metrics (return, volatility, Sharpe ratio, beta, expected return)
- Suggests changes to improve risk-adjusted return
- Visualizes the efficient frontier and Sharpe-maximizing portfolios
- Allows the user to choose their risk-free rate and benchmark index
- Eventually supports Monte Carlo simulations and a simple UI

## Current Features

- [x] Fetch historical price data using `yfinance`
- [x] Calculate daily returns and portfolio-level performance
- [x] Compute annualized return, volatility, and Sharpe ratio


##  Used tools 

- Python 3.13
- yfinance
- pandas
- numpy
- matplotlib / plotly (for charts)


