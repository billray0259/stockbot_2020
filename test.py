import pandas as pd

df = pd.read_csv("temp.csv")

df = df[["underlying", "delta", "strikePrice"]]

print(df)