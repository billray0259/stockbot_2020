import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pylint: disable=import-error
from td_api import Account
import pandas as pd
from tqdm import tqdm

DATA_FILE = "/home/bill/stockbot_2020/data/finviz_tech.csv"
SAVE_DIRECTORY = "/home/bill/stockbot_2020/data/tech_histories/"

def collect_csv(finviz_csv, save_directory, keys_json="keys.json"):
    account = Account(keys_json)

    finviz_df = pd.read_csv(finviz_csv)
    tickers = finviz_df["Ticker"]

    for ticker in tqdm(tickers):
        history_df = account.history(ticker, 15, 365)
        if len(history_df) > 0:
            history_df.to_csv(save_directory + "%s.csv" % ticker)

if __name__ == "__main__":
    collect_csv(DATA_FILE, SAVE_DIRECTORY)
