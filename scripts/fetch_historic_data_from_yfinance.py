import yfinance as yf

df = yf.download(
    "QQQ",
    start="2000-01-01",
    end="2024-01-01",
    auto_adjust=False
)

df.to_csv("qqq_daily.csv")
print(df.head())