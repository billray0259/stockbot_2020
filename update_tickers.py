from data_handler import DataHandler
from delta_bot_V2 import get_profiles
from td_api import Account

import numpy as np
import pandas as pd
import traceback
from datetime import datetime

try:
    dh = DataHandler("optionable", "v=111&f=cap_smallover,sh_curvol_o750,sh_opt_option,sh_price_o2&o=-volume")
    # dh.save_finviz()

    tickers = pd.read_hdf(dh.finviz_file).index

    acc = Account("/home/bill/rillion/delta_bot/keys.json")

    profiles = get_profiles(tickers, acc)
    print(profiles)

    tickers = np.array(profiles.index)[:100]
    np.savetxt("tickers.txt", tickers, delimiter="\n", fmt="%s")
except Exception as e:
    print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Exception when updating tickers")
    traceback.print_exc()
