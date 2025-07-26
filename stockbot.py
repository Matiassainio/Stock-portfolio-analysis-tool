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
print("\nSyötä salkkusi osakkeet (Usa) ja painot muodossa esim:\nAAPL 30%\nMSFT 50%\nKo 20%")
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


def fetch_data_multi(tickers, start="2023-01-01", end="2024-01-01"):
    """
    Hakee osakkeiden Adjusted Close -hinnat listalta tickers.
    Tukee sekä yksittäistä että useampaa osaketta.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if data.empty:
        print("⚠️ Ei dataa ladattu tickereille:", tickers)
        return None

    # Jos data sisältää useamman sarjan, se on monitasoinen DataFrame
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            return data['Adj Close']
        else:
            print("⚠️ 'Adj Close' ei löytynyt monitasoisesta DataFramesta. Sarakkeet:", data.columns.levels[0])
            return None
    else:
        # Yksittäisen osakkeen tapauksessa data on yksinkertainen DataFrame
        if 'Adj Close' in data.columns:
            return data['Adj Close'].to_frame(tickers if isinstance(tickers, str) else tickers[0])
        else:
            print("⚠️ 'Adj Close' ei löytynyt yksinkertaisesta DataFramesta. Sarakkeet:", data.columns)
            return None


import numpy as np  # lisää vain jos tätä ei jo ole ylhäällä
def calculate_portfolio_metrics(prices, weights):
    """
    Laskee portfolion vuosituoton, volatiliteetin ja Sharpen luvun.
    """
    returns = prices.pct_change().dropna()
    weights = pd.Series(weights, index=returns.columns)  # <- tämä on tärkeä korjaus!

    weights = pd.Series(weights)
    weights = weights.loc[returns.columns]  # järjestää painot oikein kolumnien mukaan
 
    portfolio_returns = (returns * weights).sum(axis=1)

    annual_return = portfolio_returns.mean() * 252
    volatility = portfolio_returns.std() * (252**0.5)
    sharpe_ratio = annual_return / volatility

    return annual_return, volatility, sharpe_ratio
    if __name__ == "__main__":
        tickers = ["AAPL", "MSFT", "KO"]
        weights = [0.3, 0.5, 0.2]

    prices = fetch_data_multi(tickers)
    print(prices)
    print(tickers)
    print("DEBUG: PRICES\n", prices)

    if prices is None or prices.empty:
        print("❌ Ei saatu dataa osakkeista.")
    else:
        print("✅ Data saatu onnistuneesti.")
        if prices is not None:
            ret, vol, sharpe = calculate_portfolio_metrics(prices, weights)
            print(f"\n📊 Portfolio metrics for {tickers}")
            print(f"  Annual return: {ret:.2%}")
            print(f"  Volatility:    {vol:.2%}")
            print(f"  Sharpe ratio:  {sharpe:.2f}")

