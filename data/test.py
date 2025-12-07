import pandas as pd
df = pd.read_parquet("data/tickers/AFKS/M2.parquet")
print(df.tail(1000))