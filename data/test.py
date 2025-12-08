import pandas as pd
df = pd.read_parquet("data/tickers/SBER/M30.parquet")
print(df.tail(2000))