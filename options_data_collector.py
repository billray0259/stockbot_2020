from datetime import datetime, timedelta
from data_handler import DataHandler
import pandas as pd
import numpy as np
import td_api as td
import time
import os
import sys
import math

NUM_TICKERS = 100
STRIKE_COUNT = 5
NUM_QUOTES = 250

PERIOD_SECONDS = 60*10

TIME_MOD_EST = -2

SAVE_DIR = "options_data"

def get_monthly_expiration_dates(num_dates):
    expiration_days = []

    day = datetime.now()
    day = datetime(year=day.year, month=day.month, day=1)
    delta = timedelta(days=1)
    friday_count = 0
    while len(expiration_days) < 6:
        if day.weekday() == 4:
            friday_count += 1

        if friday_count == 3:
            if day > datetime.now() - timedelta(days=1):
                expiration_days.append(datetime(year=day.year, month=day.month, day=day.day))
            friday_count = 0

            if day.month == 12:
                day = datetime(year=day.year+1, month=1, day=1)
            else:
                day = datetime(year=day.year, month=day.month+1, day=1)
        else:
            day += delta
    
    return expiration_days

account = td.Account("keys.json")

if sys.argv[1] == "list":
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r") as tickers_file:
            tickers = list(map(lambda x: x.strip(), tickers_file.readlines()))
    else:
        dh = DataHandler("vol500k")
        tickers = pd.read_hdf(dh.finviz_file).index

        np.random.seed(0)
        tickers = np.random.choice(tickers, size=NUM_TICKERS)
        with open("tickers.txt", "w") as tickers_file:
            tickers_file.write("\n".join(tickers))

    symbols = []

    expiration_dates = get_monthly_expiration_dates(6)

    last_request_time = time.time()
    for ticker in tickers:
        from_date = datetime.now()
        if from_date.month > 5:
            to_date = datetime(from_date.year+1, (from_date.month+7)%12, from_date.day)
        else:
            to_date = datetime(from_date.year, from_date.month+7, from_date.day)

        time.sleep(max(0, 0.6 - (time.time() - last_request_time)))
        last_request_time = time.time()
        options_chain = account.get_options_chain(ticker, from_date=from_date, to_date=to_date, strike_count=STRIKE_COUNT)
        if options_chain is None:
            continue
        options_chain["expirationDate"] = pd.to_datetime(options_chain["expirationDate"], unit="ms")
        is_monthly = [datetime(date.year, date.month, date.day) in expiration_dates for date in options_chain["expirationDate"]]
        options_chain = options_chain[is_monthly]

        symbols.extend(options_chain.index)
        print("Collected:", len(symbols), end="\r", flush=True)
    print()

    with open("symbols.txt", "w") as symbols_file:
        symbols_file.write("\n".join(symbols))

elif sys.argv[1] == "collect":
    with open("symbols.txt", "r") as symbols_file:
        symbols = list(map(lambda x: x.strip(), symbols_file.readlines()))
    
    start = 0
    wait_time = PERIOD_SECONDS / math.ceil(len(symbols) / NUM_QUOTES)
    last_request_time = time.time()
    fetched = 0
    while datetime.now().hour < 16 + TIME_MOD_EST:
        if start + NUM_QUOTES >= len(symbols):
            symbols_to_fetch = symbols[start:]
            start = (start + NUM_QUOTES) % len(symbols)
            symbols_to_fetch.extend(symbols[:start])
        else:
            symbols_to_fetch = symbols[start:start+NUM_QUOTES]

        time.sleep(max(0, wait_time - (time.time() - last_request_time)))
        last_request_time = time.time()
        quotes = account.get_quotes(symbols_to_fetch)
        fetched += len(quotes)
        print("Fetched:", fetched, end="\r", flush=True)
        path = os.path.join(SAVE_DIR, str(int(time.time()*1000)) + ".h5")
        quotes.to_hdf(path, key="df", mode="w")
        quotes = None
        
