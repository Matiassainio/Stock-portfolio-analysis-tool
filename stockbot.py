import yfinance as yf
import pandas as pd
import numpy as np


def fetch_data(ticker):
    data = yf.download(ticker, period="1y", auto_adjust=False)
    if data.empty:
        print("⚠️ Ei löydetty dataa tickerille:", ticker)
        return None
    if 'Adj Close' not in data.columns:
        print("⚠️ 'Adj Close' ei löytynyt. Saatavilla olevat sarakkeet:", data.columns)
        return None
    return data['Adj Close']

def calculate_metrics(prices):
    returns = prices.pct_change().dropna()
    avg_return = returns.mean()
    volatility = returns.std()
    sharpe_ratio = avg_return / volatility * np.sqrt(252)  # annualisoitu

    # Muunnetaan kaikki kelluviksi (varmuuden vuoksi)
    return float(avg_return), float(volatility), float(sharpe_ratio)


def analyze_stock(ticker):
    print(f"\nAnalysoidaan osake: {ticker}")
    prices = fetch_data(ticker)
    if prices is None:
        print("Tietoja ei löytynyt. Tarkista ticker.")
        return
    avg, vol, sharpe = calculate_metrics(prices)
    print(f"Keskimääräinen päivätuotto: {avg:.4%}")
    print(f"Volatiliteetti: {vol:.4%}")
    print(f"Sharpen suhde: {sharpe:.2f}")

portfolio = []

while True:
    user_input = input("\nSyötä osakkeen ticker (tai 'exit' lopettaaksesi): ").upper()
    if user_input == "EXIT":
        break
    analyze_stock(user_input)
    portfolio.append(user_input)

print("\nAnalysoidut osakkeet:", portfolio)