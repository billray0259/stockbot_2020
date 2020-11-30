import requests
from bs4 import BeautifulSoup
from td_api import Account
from datetime import datetime, timedelta
from price_probability_distobution import get_pdfs_from_deltas
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import curve_fit, minimize
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback

CAPITAL = 5000
MAX_KELLY = 2
BLACKLIST = ["CD", "VNET", "TSLA", "SPY"]
ROUND_SHARES = True


KEYS_FILE = "/home/bill/rillion/delta_bot/keys.json"
TICKERS_FILE = "/home/bill/rillion/delta_bot/tickers.txt"
COVARIANEC_FILE = "/home/bill/rillion/delta_bot/covariance.csv"
RISK_FREE_RATE_FILE = "/home/bill/rillion/delta_bot/risk_free_rate.txt"

with open(RISK_FREE_RATE_FILE, "r") as risk_free_rate_file:
    risk_free_rate = float(risk_free_rate_file.read())


def get_optimal_portfolio_weights(avg_returns, stds, covariance_mat):
    assert (avg_returns.index == stds.index).all() and (avg_returns.index == covariance_mat.index).all(), "avg_returns, stds, and covariance_mat don't have the same index"

    adj_covariance_mat = covariance_mat / np.diag(covariance_mat) * stds**2

    def negative_sharpe(weights):
        avg_return = np.sum(avg_returns * weights) * 252
        volatility = np.sqrt(np.dot(weights.T, np.dot(adj_covariance_mat * 252, weights)))
        return -avg_return / volatility

    constraints = ({
        "type": "eq",
        "fun": lambda x: np.sum(x) - 1
    })

    bounds = tuple((0, 1) for x in range(len(avg_returns)))

    initial_weights = np.ones(len(avg_returns)) / len(avg_returns)

    best_weights = minimize(negative_sharpe, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)["x"]

    avg_return = np.sum(avg_returns * best_weights) * 252
    volatility = np.sqrt(np.dot(best_weights.T, np.dot(adj_covariance_mat * 252, best_weights)))
    print("Avg Return:", avg_return)
    print("Volatility:", volatility)
    print("Sharpe Ratio:", avg_return/volatility)
    kelly = (avg_return - risk_free_rate)/volatility**2
    print("Kelly Criterion:", kelly)
    best_weights = pd.Series(np.round(best_weights, 7), avg_returns.index)

    return best_weights[best_weights > 0], kelly


def get_returns_and_covariance(tickers, acc):
    # pylint: disable=no-member
    returns = pd.DataFrame()
    last_query_time = 0
    hist_lengths = []
    for ticker in tickers:
        time.sleep(max(0, 0.6 - (time.time() - last_query_time)))
        print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Getting data for", ticker, flush=True)
        history = acc.history(ticker, 1, 365, frequency_type="daily")["close"]
        hist_lengths.append(len(history))
        last_query_time = time.time()
        returns[ticker] = np.log(history / history.shift(1))
    covariance = returns.cov()
    len_history = np.mean(hist_lengths)
    return returns.mean() * len_history, covariance * len_history


def get_profiles(tickers, acc):

    positions = acc.get_positions().keys()

    profiles = []
    from_date = datetime.now()
    to_date = from_date + timedelta(days=40)

    last_query_time = 0

    profiles = pd.DataFrame(columns=["ticker", "time", "share_price", "mean", "std", "err_mean", "err_std", "sort_key"])
    for ticker in tickers:
        try:
            time.sleep(max(0, 1.2 - (time.time() - last_query_time)))
            print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Getting data for", ticker, flush=True)
            

            stock_history = acc.history(ticker, 30, 365, frequency_type="minute")
            price_metric = "bidPrice" if ticker in positions else "askPrice"
            ask = acc.get_quotes([ticker])["askPrice"].iloc[0]
            bid = acc.get_quotes([ticker])["bidPrice"].iloc[0]

            mark = bid if ticker in positions else ask

            if ticker == "SHV":
                profile = {}
                profile["ticker"] = ticker
                profile["time"] = int(time.time())
                profile["share_price"] = mark
                profile["mean"] = mark + risk_free_rate/252
                profile["std"] = 1e-5 * mark
                profile["err_mean"] = 0
                profile["err_std"] = 0
                # s_sign = 1 if u > mark else -1
                # loss_odds_max = quad(lambda x: stats.norm.pdf(x, u-errs[0], s + errs[1]*s_sign), 0, mark)[0]
                # profit_odds = 1 - stats.norm.cdf(mark, u, s)
                profile["sort_key"] = 0

                profiles = profiles.append(profile, ignore_index=True)
                continue
            
            options_data = acc.get_options_chain(ticker, from_date, to_date, strike_count=50)
            if options_data is None:
                continue

            last_query_time = time.time()
            print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Calculating PDF for", ticker, flush=True)
            pdfs = get_pdfs_from_deltas(options_data, distribution=stats.norm)
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

            u = mean_line(1, mean_m - mean_line_errs[0], mean_b - mean_line_errs[1])
            s = std_line(1, std_m + std_line_errs[0], std_b + std_line_errs[1], std_c + std_line_errs[2])

            profile = {}
            profile["ticker"] = ticker
            profile["time"] = int(time.time())
            profile["share_price"] = mark
            profile["mean"] = u
            profile["std"] = s
            profile["err_mean"] = np.sqrt(np.abs(np.sum(np.diag(mean_cov) / mean_opts)))
            profile["err_std"] = np.sqrt(np.abs(np.sum(np.diag(std_cov) / std_opts)))
            # s_sign = 1 if u > mark else -1
            # loss_odds_max = quad(lambda x: stats.norm.pdf(x, u-errs[0], s + errs[1]*s_sign), 0, mark)[0]
            # profit_odds = 1 - stats.norm.cdf(mark, u, s)
            profile["sort_key"] = (u / mark) / s

            profiles = profiles.append(profile, ignore_index=True)
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
        covariance = pd.read_csv(COVARIANEC_FILE, index_col="index")
        tickers = list(covariance.index)
    except Exception as e:
        print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Exception when loading tickers.txt", flush=True)
        traceback.print_exc()

    try:
        acc = Account(KEYS_FILE)
        # acc = Account("keys.json")
    except Exception as e:
        print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Exception when creating TD account", flush=True)
        # print(e, flush=True)

    try:
        profiles = get_profiles(tickers, acc)
        profiles.dropna(inplace=True)

        for ticker in BLACKLIST:
            if ticker in profiles.index:
                profiles.drop(ticker, inplace=True)

        tickers = profiles.index

        positions = acc.get_positions()

        for ticker in BLACKLIST:
            if ticker in positions:
                del positions[ticker]
        
        covariance = covariance.loc[tickers, tickers]

        avg_returns = profiles["mean"] / profiles["share_price"] - 1
        normalized_stds = profiles["std"] / profiles["share_price"]
        opt_weights, kelly = get_optimal_portfolio_weights(avg_returns, normalized_stds, covariance)
        opt_weights = opt_weights[opt_weights * CAPITAL > 10]
        opt_weights /= np.sum(opt_weights)
        opt_weights *= min(MAX_KELLY, kelly)

        last_query_time = 0
    except Exception as e:
        print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Exception when getting profiles and calculating opt_weights")
        traceback.print_exc()
        print("", end="", flush=True)
        exit()

    sell_tickers = [t for t in positions.keys() if not t in opt_weights]
    for ticker in sell_tickers:
        print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "SELL %d %s" % (positions[ticker], ticker), flush=True)
        time.sleep(max(0, 0.6 - (time.time() - last_query_time)))
        acc.sell(ticker, positions[ticker])
        last_query_time = time.time()

    old_tickers = [t for t in opt_weights.index if t in positions.keys()]
    for ticker in old_tickers:
        mark = profiles["share_price"][ticker]
        target_shares = CAPITAL * opt_weights[ticker] / mark
        change = target_shares - positions[ticker]
        change = round(change) if ROUND_SHARES else int(change)
        percent_change = abs(change * mark / CAPITAL)
        if percent_change > 0.1:
            if change < 0:
                print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "SELL %d %s" % (-change, ticker), flush=True)
                acc.sell(ticker, -change)
            else:
                print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "BUY %d %s" % (change, ticker), flush=True)
                acc.buy(ticker, change)

    new_tickers = [t for t in opt_weights.index if not t in positions.keys()]
    for ticker in new_tickers:
        mark = profiles["share_price"][ticker]
        target_shares = CAPITAL * opt_weights[ticker] / mark
        target_shares = round(target_shares) if ROUND_SHARES else int(target_shares)
        print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "BUY %d %s" % (target_shares, ticker), flush=True)
        time.sleep(max(0, 0.6 - (time.time() - last_query_time)))
        acc.buy(ticker, target_shares)
        last_query_time = time.time()
