import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pylint: disable=import-error
from td_api import Account
import pandas as pd

account = Account("keys.json")

small_cap = pd.read_csv("small_cap.csv")
small_cap = small_cap.sort_values(by="Volume", ascending=False)[:2]

tickers = small_cap["Ticker"]

for ticker in tickers:
    history_df = account.history(ticker, 1, 365)
    if history_df is False:
        print("No data for", ticker)
        continue
    print(history_df)
    # history_df.to_csv("histories2/%s.csv" % ticker)
