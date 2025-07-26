import yfinance as yf

ticker = "aapl"
data = yf.download(ticker, start="2023-01-01", end="2024-01-01", progress=False)
print(data.head())
