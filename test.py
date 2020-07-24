import td_api as td
import pandas as pd

acc = td.Account("keys.json")
dfs = []
for ticker in ["TSLA", "AMD", "GDS", "GOOG", "FB", "AMZN", "MOD", "SPY", "VXX", "AAPL", "SWCH"]:
    options = acc.get_options_chain(ticker, strike_count=27)
    dfs.append(options)
    print(options)

df = pd.concat(dfs)

quotes = acc.get_quotes(df.index)

quotes.to_csv("test.csv")