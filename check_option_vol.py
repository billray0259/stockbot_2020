import pandas as pd
import td_api as td
import options_data_collector as op_collect
from datetime import datetime
import time

data = pd.read_hdf('data/optionable/finviz.h5')[:600]
account = td.Account("keys.json")

to_date = min(op_collect.get_fridays(datetime.now()))
from_date = datetime.now()
STRIKE_COUNT = 100
vals = []
last_request_time = time.time()
for ticker in data.index:
    time.sleep(max(0.001, 0.6 - (time.time() - last_request_time)))
    last_request_time = time.time()
    options_chain = account.get_options_chain(ticker, from_date=from_date, to_date=to_date, strike_count=STRIKE_COUNT)
    if not options_chain is None:
        vals.append((sum(options_chain['totalVolume']), ticker))


vals.sort(reverse = True)
print(vals)
print(len(vals))
final = []
for tickers in vals:
    if len(final) == 100:
        break
    final.append(tickers[1])
print(final)
print(len(final))