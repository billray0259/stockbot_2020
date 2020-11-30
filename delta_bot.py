import requests
from bs4 import BeautifulSoup
from td_api import Account
from datetime import datetime, timedelta
from price_probability_distobution import get_pdfs_from_deltas, get_pdfs_from_marks
from scipy import stats
from scipy.integrate import quad
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback

CAPITAL = 10000
NUM_STOCKS = 10

STATIC_HOLDINGS = {
    "TSLA": 10
}

ROUND = True
print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Starting...", flush=True)
try:
    page = requests.get("https://research.investors.com/options-center/reports/option-volume", headers={"User-Agent": "Chrome"})
    soup = BeautifulSoup(page.text, features="lxml")

    items = soup.findAll("a", {"class": "stockRoll"})
    tickers = [item.text for item in items]
except Exception as e:
    print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Exception when gettings tickers from investors.com", flush=True)
    print(e, flush=True)
    exit()

try:
    acc = Account("/home/bill/rillion/delta_bot/keys.json")
except Exception as e:
    print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Exception when creating TD account", flush=True)
    # print(e, flush=True)
    

profiles = []
from_date = datetime.now()
to_date = from_date + timedelta(days=40)

last_query_time = 0

profiles = pd.DataFrame(columns=["ticker", "time", "share_price", "mean", "std", "err_mean", "err_std", "sort_key"])

for ticker in tickers:
    try:
        time.sleep(max(0, 0.6 - (time.time() - last_query_time)))
        print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Getting data for", ticker, flush=True)
        data = acc.get_options_chain(ticker, from_date, to_date, strike_count=50)
        if data is None:
            continue
        mark = acc.get_quotes([ticker])["mark"].iloc[0]
        last_query_time = time.time()
        print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Calculating PDF for", ticker, flush=True)
        ticker_pdfs = get_pdfs_from_deltas(data, distribution=stats.logistic)
        profile = {}
        (u, s, errs) = list(ticker_pdfs.values())[-1]
        profile["ticker"] = ticker
        profile["time"] = int(time.time())
        profile["share_price"] = mark
        profile["mean"] = round(u, 3)
        profile["std"] = round(s, 3)
        profile["err_mean"] = round(errs[0], 3)
        profile["err_std"] = round(errs[1], 3)
        s_sign = 1 if u > mark else -1
        # loss_odds_max = quad(lambda x: stats.logistic.pdf(x, u-errs[0], s + errs[1]*s_sign), 0, mark)[0]
        profit_odds = 1 - stats.logistic.cdf(mark, u-errs[0], s + errs[1]*s_sign)
        profile["sort_key"] = (u / mark - errs[0] / u) * profit_odds
        
        if u - errs[0] > mark:
            profiles = profiles.append(profile, ignore_index=True)
    except Exception as e:
        print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Exception when getting data or gettings PDFs for", ticker, flush=True)
        # print(e, flush=True)
        traceback.print_exc()

profiles.sort_values("sort_key", ascending=False, inplace=True)
profiles.drop(columns=["sort_key"], inplace=True)
profiles.set_index("ticker", inplace=True)

profiles = profiles[~profiles.index.duplicated(keep='first')]

for ticker in STATIC_HOLDINGS:
    if ticker in profiles.index:
        profiles.drop(ticker, inplace=True)

tickers = profiles[:10].index
positions = acc.get_positions()

for ticker in STATIC_HOLDINGS:
    if ticker in positions:
        del positions[ticker]

sell_tickers = [t for t in positions.keys() if not t in tickers]
for ticker in sell_tickers:
    print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "SELL %d %s" % (positions[ticker], ticker), flush=True)
    time.sleep(max(0, 0.6 - (time.time() - last_query_time)))
    acc.sell(ticker, positions[ticker])
    last_query_time = time.time()

old_tickers = [t for t in tickers if t in positions.keys()]
for ticker in old_tickers:
    mark = profiles["share_price"][ticker]
    target_shares = CAPITAL / NUM_STOCKS / mark
    change = target_shares - positions[ticker]
    change = round(change) if ROUND else int(change)
    percent_change = abs(change * mark / CAPITAL)
    if percent_change > 0.1:
        if change < 0:
            print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "SELL %d %s" % (-change, ticker), flush=True)
            acc.sell(ticker, -change)
        else:
            print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "BUY %d %s" % (change, ticker), flush=True)
            acc.buy(ticker, change)

new_tickers = [t for t in tickers if not t in positions.keys()]
for ticker in new_tickers:
    mark = profiles["share_price"][ticker]
    target_shares = CAPITAL / NUM_STOCKS / mark
    target_shares = round(target_shares) if ROUND else int(target_shares)
    print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "BUY %d %s" % (target_shares, ticker), flush=True)
    time.sleep(max(0, 0.6 - (time.time() - last_query_time)))
    acc.buy(ticker, target_shares)
    last_query_time = time.time()
