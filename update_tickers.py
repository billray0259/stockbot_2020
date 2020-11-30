from data_handler import DataHandler
from delta_bot_V3 import get_profiles
from td_api import Account

import numpy as np
import pandas as pd
import traceback
from datetime import datetime
import time


def get_returns_and_covariance(tickers, acc):
    # pylint: disable=no-member
    returns = pd.DataFrame()
    last_query_time = 0
    for ticker in tickers:
        time.sleep(max(0, 0.6 - (time.time() - last_query_time)))
        print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Getting data for", ticker, flush=True)
        history = acc.history(ticker, 1, 365, frequency_type="daily")["close"]
        last_query_time = time.time()
        returns[ticker] = np.log(history / history.shift(1))
    covariance = returns.cov()
    return returns.mean() * len(history), covariance * len(history)

try:
    dh = DataHandler("optionable", "v=111&f=cap_smallover,sh_curvol_o750,sh_opt_option,sh_price_o2&o=-volume")
    dh.save_finviz()

    tickers = pd.read_hdf(dh.finviz_file).index

    acc = Account("/home/bill/rillion/delta_bot/keys.json")

    profiles = get_profiles(tickers, acc)

    tickers = np.array(profiles.index)[:100]
    np.savetxt("tickers.txt", tickers, delimiter="\n", fmt="%s")

    returns, covariance = get_returns_and_covariance(tickers, acc)

    covariance.to_csv("covariance.csv")

except Exception as e:
    print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Exception when updating tickers")
    traceback.print_exc()
