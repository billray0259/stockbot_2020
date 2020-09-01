import requests
from bs4 import BeautifulSoup
from td_api import Account
from datetime import datetime, timedelta
from price_probability_distobution import get_pdfs_from_deltas, get_pdfs_from_marks
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import curve_fit
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

def get_profiles(tickers, acc):

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
            pdfs = get_pdfs_from_deltas(data, distribution=stats.norm)
            if len(pdfs.keys()) == 1:
                print(ticker, " doesn't have weekly expirations")
                continue
            days = []
            means = []
            mean_errs = []
            stds = []
            std_errs = []
            total_err = 0
            for label in pdfs:
                u, s, errs = pdfs[label]
                err = 100 * np.linalg.norm(errs / (u, s))
                total_err += err
                means.append(u)
                mean_errs.append(errs[0])
                stds.append(s)
                std_errs.append(errs[1])

                expiration = datetime.strptime(label, "%Y-%m-%d")
                today = datetime.now()
                today = datetime(today.year, today.month, today.day)
                days.append((expiration - today).days)
            
            if total_err / len(pdfs) > 20:
                print("Too much error in", ticker)
                continue

            def mean_line(x, m, b):
                return m * x + b

            def std_line(x, m, b, c):
                return m * x**c + b

            days = np.array(days)
            means = np.array(means)
            stds = np.array(stds)

            mean_opts, mean_cov = curve_fit(mean_line, days, means, p0=(1, mark), bounds=((-np.inf, 0), (np.inf, np.inf)), sigma=stds)
            mean_m, mean_b = mean_opts

            # if pcov[0][0] == np.nan:
            # import matplotlib.pyplot as plt
            # plt.scatter(days, means)
            # plt.show()
            mean_line_errs = np.sqrt(np.diag(mean_cov))

            std_opts, std_cov = curve_fit(std_line, days, stds, p0=(1, 0, 0.5), bounds=((-np.inf, -np.inf, 0), (np.inf, np.inf, 1)), sigma=std_errs)
            std_m, std_b, std_c = std_opts

            std_line_errs = np.sqrt(np.diag(std_cov))

            u = mean_line(2, mean_m - mean_line_errs[0], mean_b - mean_line_errs[1])
            s = std_line(2, std_m + std_line_errs[0], std_b + std_line_errs[1], std_c + std_line_errs[2])

            profile = {}
            profile["ticker"] = ticker
            profile["time"] = int(time.time())
            profile["share_price"] = mark
            profile["mean"] = round(u, 3)
            profile["std"] = round(s, 3)
            profile["err_mean"] = np.sqrt(np.abs(np.sum(np.diag(mean_cov) / mean_opts)))
            profile["err_std"] = np.sqrt(np.abs(np.sum(np.diag(std_cov) / std_opts)))
            s_sign = 1 if u > mark else -1
            # loss_odds_max = quad(lambda x: stats.logistic.pdf(x, u-errs[0], s + errs[1]*s_sign), 0, mark)[0]
            profit_odds = 1 - stats.logistic.cdf(mark, u, s)
            profile["sort_key"] = (u / mark) * profit_odds

            if u > mark:
                profiles = profiles.append(profile, ignore_index=True)
            else:
                print(ticker, " is shortable")
        except Exception as e:
            print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Exception when getting data or gettings PDFs for", ticker, flush=True)
            traceback.print_exc()

    profiles.sort_values("sort_key", ascending=False, inplace=True)
    profiles.drop(columns=["sort_key"], inplace=True)
    profiles.set_index("ticker", inplace=True)

    profiles = profiles[~profiles.index.duplicated(keep='first')]

    return profiles

if __name__ == "__main__":
    print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Starting...", flush=True)

    try:
        tickers = np.loadtxt("/home/bill/rillion/delta_bot/tickers.txt", delimiter="\n", dtype=str)
    except Exception as e:
        print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Exception when loading tickers.txt", flush=True)
        traceback.print_exc()
        
    try:
        acc = Account("/home/bill/rillion/delta_bot/keys.json")
        # acc = Account("keys.json")
    except Exception as e:
        print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Exception when creating TD account", flush=True)
        # print(e, flush=True)

    profiles = get_profiles(tickers, acc)

    for ticker in STATIC_HOLDINGS:
        if ticker in profiles.index:
            profiles.drop(ticker, inplace=True)

    tickers = profiles[:10].index
    positions = acc.get_positions()

    for ticker in STATIC_HOLDINGS:
        if ticker in positions:
            del positions[ticker]
    
    last_query_time = 0

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
