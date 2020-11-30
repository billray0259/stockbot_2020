from data_handler import DataHandler
from delta_bot_V3 import get_profiles
from td_api import Account

import numpy as np
import pandas as pd
import traceback
from datetime import datetime
import time

KEYS_FILE = "/home/bill/rillion/delta_bot/keys.json"
DATA_DIR = "/home/bill/rillion/delta_bot/data/"
TICKERS_FILE = "/home/bill/rillion/delta_bot/tickers.txt"
COVARIANCE_FILE = "/home/bill/rillion/delta_bot/covariance.csv"
RISK_FREE_RATE_FILE = "/home/bill/rillion/delta_bot/risk_free_rate.txt"


def get_returns_and_covariance(tickers, acc):
    # pylint: disable=no-member
    returns = pd.DataFrame()
    last_query_time = 0
    hist_lengths = []
    for ticker in tickers:
        time.sleep(max(0, 0.6 - (time.time() - last_query_time)))
        print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Getting history for", ticker, flush=True)
        history = acc.history(ticker, 1, 365, frequency_type="daily")["close"]
        hist_lengths.append(len(history))
        last_query_time = time.time()
        returns[ticker] = np.log(history / history.shift(1))
    covariance = returns.cov()
    len_history = np.mean(hist_lengths)
    return returns.mean() * len_history, covariance * len_history

try:
    dh = DataHandler("optionable", "v=111&f=cap_smallover,ind_stocksonly,sh_avgvol_o500,sh_curvol_o500,sh_opt_option,sh_price_o2&o=-price", keys_json=KEYS_FILE, data_dir=DATA_DIR)
    dh.save_finviz()

    tickers = pd.read_hdf(dh.finviz_file).index
    acc = Account(KEYS_FILE)
    profiles = get_profiles(tickers, acc)
    tickers = list(profiles.index)[:100] + ["SHV"]
    print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Saving tickers...")
    np.savetxt(TICKERS_FILE, np.array(tickers), delimiter="\n", fmt="%s")

    returns, covariance = get_returns_and_covariance(tickers, acc)
    covariance.index.name = "index"
    print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Saving covariance...")
    covariance.to_csv(COVARIANCE_FILE)

    with open(RISK_FREE_RATE_FILE, "w") as risk_free_rate_file:
        risk_free_rate_file.write("%f" % returns["SHV"])

except Exception as e:
    print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Exception when updating tickers")
    traceback.print_exc()
    print("^^", flush=True)
