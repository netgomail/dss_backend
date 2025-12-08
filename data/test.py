import pandas as pd
df = pd.read_parquet("data/tickers/GAZP/M5.parquet")
print(df.tail(2000))