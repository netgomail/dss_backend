import pandas as pd
df = pd.read_parquet("data/tickers/ASTR/D1.parquet")
print(df.tail(2000))