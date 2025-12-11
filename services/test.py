import pandas as pd
df = pd.read_parquet("data/tickers/GAZP/M30.parquet")
print(df.tail(2000))