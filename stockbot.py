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
import pandas as pd

def parse_portfolio_input(user_input):
    lines = user_input.strip().split("\n")
    portfolio_data = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2 or not parts[1].endswith('%'):
            raise ValueError(f"Rivi ei ole oikeassa muodossa: '{line}' (muodon tulisi olla: 'TICKER 25%')")
        
        ticker = parts[0].upper()
        try:
            weight = float(parts[1].replace('%', '').replace(',', '.')) / 100
        except ValueError:
            raise ValueError(f"Paino ei ole numero: '{parts[1]}'")
        
        if not 0 <= weight <= 1:
            raise ValueError(f"Painon tulee olla välillä 0–100 %. Annettu: {weight*100:.2f}%")
        
        portfolio_data.append({"Ticker": ticker, "Weight": weight})

    portfolio_df = pd.DataFrame(portfolio_data)

    total_weight = portfolio_df["Weight"].sum()
    if not (0.999 <= total_weight <= 1.001):
        raise ValueError(f"Painojen summa on {round(total_weight*100, 2)} %, sen pitäisi olla 100 %.")

    return portfolio_df


portfolio = []

# Pyydetään käyttäjää syöttämään portfolio
print("\nSyötä salkkusi osakkeet ja painot muodossa esim:\nAAPL 30%\nMSFT 50%\nKONE.HE 20%")
print("Kun olet valmis, paina Enter kahdesti.\n")

user_input_lines = []
while True:
    line = input()
    if line.strip() == "":
        break
    user_input_lines.append(line)

user_input_text = "\n".join(user_input_lines)

try:
    portfolio_df = parse_portfolio_input(user_input_text)
    print("\n✅ Portfoliosi luettu onnistuneesti:")
    print(portfolio_df)
except ValueError as e:
    print(f"\n❌ Virhe: {e}")


print("\nAnalysoidut osakkeet:", portfolio)