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
    sharpe_ratio = avg_return / volatility * np.sqrt(252)  

    
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
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)

    if data.empty:
        print("⚠️ Ei dataa ladattu tickereille:", tickers)
        return None

    
    if isinstance(data.columns, pd.MultiIndex):
        try:
            adj_close = data['Adj Close']
        except KeyError:
            print("⚠️ 'Adj Close' ei löytynyt. Saatavilla olevat tasot:", data.columns.levels)
            return None
        return adj_close

    
    elif 'Adj Close' in data.columns:
        return data[['Adj Close']].rename(columns={'Adj Close': tickers if isinstance(tickers, str) else tickers[0]})
    else:
        print("⚠️ 'Adj Close' ei löytynyt. Saatavilla olevat sarakkeet:", data.columns)
        return None


import numpy as np  
def calculate_portfolio_metrics(prices, weights):
    """
    Laskee portfolion vuosituoton, volatiliteetin ja Sharpen luvun.
    """
    returns = prices.pct_change().dropna()
    weights = pd.Series(weights, index=returns.columns)  

    weights = pd.Series(weights)
    weights = weights.loc[returns.columns]  
 
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

if __name__ == "__main__":
    if 'portfolio_df' in locals():
        tickers = portfolio_df['Ticker'].tolist()
        weights = portfolio_df['Weight'].tolist()

        prices = fetch_data_multi(tickers)
        if prices is None or prices.empty:
            print("❌ Ei saatu dataa osakkeista.")
        else:
            print("✅ Data saatu onnistuneesti.")
            ret, vol, sharpe = calculate_portfolio_metrics(prices, weights)
            print(f"\n📊 Portfolio metrics for {tickers}")
            print(f"  Annual return: {ret:.2%}")
            print(f"  Volatility:    {vol:.2%}")
            print(f"  Sharpe ratio:  {sharpe:.2f}")
    else:
        print("⚠️ Portfoliota ei määritelty oikein.")

    import matplotlib.pyplot as plt

    def get_risk_free_rate():
        user_input = input("\nSyötä riskitön korko (esim. 2,5 tai paina Enter oletusarvoon 0 %): ").strip().replace(',', '.')
        if user_input == "":
            print("⚠️ Käytetään oletuksena riskitöntä korkoa: 0,0 %")
            return 0.0
        try:
            rate = float(user_input) / 100
            print(f"✅ Riskitön korko asetettu: {rate*100:.2f} %")
            return rate
        except ValueError:
            print("❌ Syöte ei ollut kelvollinen luku. Käytetään oletuksena 0,0 %")
            return 0.0

    def simulate_efficient_frontier(returns_df, current_weights=None):
        np.random.seed(42)

        risk_free_rate = get_risk_free_rate()

        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        num_assets = len(mean_returns)
        num_portfolios = 10000

        results = {
            "returns": [],
            "volatility": [],
            "sharpe": [],
            "weights": []
        }

        for _ in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)

            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0

            results["returns"].append(portfolio_return)
            results["volatility"].append(portfolio_volatility)
            results["sharpe"].append(sharpe)
            results["weights"].append(weights)

        
        frontier_df = pd.DataFrame({
            "Return": results["returns"],
            "Volatility": results["volatility"],
            "Sharpe": results["sharpe"]
        })

        
        max_sharpe_idx = frontier_df["Sharpe"].idxmax()
        optimal_weights = results["weights"][max_sharpe_idx]

        
        if current_weights is not None:
            current_weights = np.array(current_weights)
            current_return = np.dot(current_weights, mean_returns)
            current_volatility = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights)))
            current_sharpe = (current_return - risk_free_rate) / current_volatility

        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(frontier_df["Volatility"], frontier_df["Return"],
                            c=frontier_df["Sharpe"], cmap="viridis", alpha=0.7)
        plt.colorbar(scatter, label="Sharpe ratio")
        plt.xlabel("Volatiliteetti (Riski)")
        plt.ylabel("Tuotto")
        plt.title("Tehokas rintama (Efficient Frontier)")

        
        plt.scatter(frontier_df.loc[max_sharpe_idx, "Volatility"],
                    frontier_df.loc[max_sharpe_idx, "Return"],
                    color="red", marker="*", s=200, label="Optimaalinen Sharpe")

        
        if current_weights is not None:
            plt.scatter(current_volatility, current_return, color="blue", marker="o", s=100, label="Oma salkku")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return frontier_df, optimal_weights
    if __name__ == "__main__":
        if 'portfolio_df' in locals():
            tickers = portfolio_df['Ticker'].tolist()
            weights = portfolio_df['Weight'].tolist()

            prices = fetch_data_multi(tickers)
            if prices is None or prices.empty:
                print("❌ Ei saatu dataa osakkeista.")
            else:
                print("✅ Data saatu onnistuneesti.")
                returns_df = prices.pct_change().dropna()
                frontier_df, optimal_weights = simulate_efficient_frontier(returns_df, weights)

                print("\n📊 Tehokas rintama (Efficient Frontier) simuloitu.")
                print("Optimaalinen Sharpe-suhde saavutetaan painoilla:")
                for ticker, weight in zip(tickers, optimal_weights):
                    print(f"{ticker}: {weight:.2%}")
        else:
            print("⚠️ Portfoliota ei määritelty oikein.")