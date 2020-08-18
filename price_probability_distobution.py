from td_api import Account
# from data_handler import DataHandler
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy import stats
import math
# from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

# exp_value_cdf = lambda x, u, s: (np.exp(x/s)*x)/(np.exp(x/s)+np.exp(u/s))-s*np.log(np.exp(x/s)+np.exp(u/s))

# callback_list = []

def _calc(values):
    r, u, s, distribution = values
    dist = lambda x: distribution.pdf(x, u, s)
    return r - quad(lambda x: dist(x) * x, -np.inf, r)[0] - r * quad(dist, r, np.inf)[0]
    # return r - quad(lambda x: dist(x) * x, -10*r, r)[0] - r * (1-distribution.cdf(r, u, s))
    # x1 = np.linspace(-r, r, num=10000)
    # y1 = dist(x1) * x1
    # x2 = np.linspace(r, 3*r, num=10000)
    # y2 = dist(x2)
    # p = trapz(y2, x1)
    # return r - trapz(y1, x1) - r * trapz(y2, x2)

def get_pdfs_from_marks(options_chain, distribution=stats.logistic):
    data = options_chain

    date_groups = data.groupby(["expirationDate"])

    pool = Pool(processes=3)

    pdfs = get_pdfs_from_deltas(data, distribution)
    for group in date_groups.groups:
        indicies = date_groups.groups[group]
        # Get only the contracts for this single expiration date
        group_data = data.loc[indicies, :]
        label = pd.to_datetime(data["expirationDate"][indicies[0]], unit="ms").strftime("%Y-%m-%d")

        u0, s0, _ = pdfs[label]

        calls = group_data[group_data["putCall"] == "CALL"]
        puts = group_data[group_data["putCall"] == "PUT"]

        call_x = -calls["strikePrice"]
        call_y = calls["mark"]

        put_x = puts["strikePrice"]
        put_y = puts["mark"]

        # intergral_tolerance = mark * 1e-2

        def option_price(rs, u, s):
            return list(pool.map(_calc, [(r, u, s, distribution) for r in rs]))

        # print("starting call curve fit")
        call_popt, call_pcov = curve_fit(option_price, call_x, call_y, (-u0, s0))
        call_popt[0] *= -1
        (call_u, call_s) = call_popt
        # print(call_popt)

        call_err = np.sqrt(np.sum(np.diag(call_pcov) / call_popt))

        # print("starting put curve fit")
        put_popt, put_pcov = curve_fit(option_price, put_x, put_y, call_popt)
        (put_u, put_s) = put_popt
        # print(put_popt)

        put_err = np.sqrt(np.sum(np.diag(put_pcov) / put_popt))

        call_weight = 1-(call_err/(call_err + put_err))
        put_weight = 1-(put_err/(call_err + put_err))

        err = (call_weight * call_err + put_weight * put_err)

        u = call_u * call_weight + put_u * put_weight
        s = call_s * call_weight + put_s * put_weight

        call_errs = np.sqrt(np.diag(call_pcov))
        put_errs = np.sqrt(np.diag(put_pcov))

        errs = call_errs * call_weight + put_errs * put_weight

        pdfs[label] = u, s, errs
        # print(label, u, s)

    return pdfs

def get_pdfs_from_deltas(options_chain, distribution=stats.logistic):
    
    data = options_chain
    # Make put deltas positive
    data["delta"] = np.abs(data["delta"])

    date_groups = data.groupby(["expirationDate"])
    
    pdfs = {}
    for group in date_groups.groups:
        indicies = date_groups.groups[group]
        # Get only the contracts for this single expiration date
        group_data = data.loc[indicies, :]
        label = pd.to_datetime(data["expirationDate"][indicies[0]], unit="ms").strftime("%Y-%m-%d")

        calls = group_data[group_data["putCall"] == "CALL"]
        puts = group_data[group_data["putCall"] == "PUT"]

        call_x = -calls["strikePrice"]
        call_y = calls["delta"]

        put_x = puts["strikePrice"]
        put_y = puts["delta"]
        
        curve = lambda x, u, s: distribution.cdf(x, u, s)

        mark = put_x[0]

        call_popt, call_pcov = curve_fit(curve, call_x, call_y, (-mark, mark/10))
        call_popt[0] *= -1
        (call_u, call_s) = call_popt

        call_err = np.sqrt(np.sum(np.diag(call_pcov) / call_popt))

        put_curve = lambda x, u, s: distribution.cdf(x, u, s)
        
        put_popt, put_pcov = curve_fit(curve, put_x, put_y, (mark, mark/10))
        (put_u, put_s) = put_popt

        put_err = np.sqrt(np.sum(np.diag(put_pcov) / put_popt))
        
        call_weight = 1-(call_err/(call_err + put_err))
        put_weight = 1-(put_err/(call_err + put_err))

        err = (call_weight * call_err + put_weight * put_err)

        u = call_u * call_weight + put_u * put_weight
        s = call_s * call_weight + put_s * put_weight

        call_errs = np.sqrt(np.diag(call_pcov))
        put_errs = np.sqrt(np.diag(put_pcov))

        errs = call_errs * call_weight + put_errs * put_weight

        pdfs[label] = u, s, errs
    
    return pdfs

if __name__ == "__main__":
    acc = Account("keys.json")

    symbol = input("Enter Symbol: ")
    days = int(input("Enter Days Out: "))
    # strike_count = int(input("Enter Strike Count: "))
    strike_count = 50

    # symbol="AMD"
    # days=21
    # strike_count=100

    from_date = datetime.now()
    to_date = from_date + timedelta(days=days)
    data = acc.get_options_chain(symbol, from_date, to_date, strike_count=strike_count)
    # data.to_csv("temp.csv")
    # data = pd.read_csv("temp.csv", index_col="symbol")
    mark = acc.get_quotes([symbol])["mark"].iloc[0]

    pdfs = get_pdfs_from_marks(data, distribution=stats.logistic)
    for i, label in enumerate(pdfs):
        u, s, errs = pdfs[label]
        err = 100 * np.linalg.norm(errs / (u, s))
        label += "\nmean: %.2f±%.2f%%\nstd: %.2f±%.2f%%\n" % (u, 100*errs[0]/u, s, 100*errs[1]/s)

        distribution = stats.logistic.pdf

        x = np.linspace(u-5*s, u+5*s, 100)
        y = distribution(x, u, s)

        loss_odds = quad(lambda x: distribution(x, u, s), 0, mark)[0]
        s_sign = 1 if u > mark else -1
        loss_odds_min = quad(lambda x: distribution(x, u+errs[0], s - errs[1]*s_sign), 0, mark)[0]
        loss_odds_max = quad(lambda x: distribution(x, u-errs[0], s + errs[1]*s_sign), 0, mark)[0]

        label += "E[return]: %.2f%%\n" % (100 * (u/mark-1))

        label += "profit: %.2f%%-%.2f%%-%.2f%%\n" % (100*(1-loss_odds_max), 100*(1-loss_odds), 100*(1-loss_odds_min))
        label += "err: %.2f\n" % err
        plt.plot(x, y, label=label)

        # print(quad(lambda x: distribution(x, u, s) * x, 0, u)[0])

        op_value = u/2 - quad(lambda x: distribution(x, u, s) * x, -10*u, u)[0]
        print(op_value)

    plt.vlines(mark, *plt.gca().get_ylim(), label="Mark %.4f" % mark)
    plt.legend()
    plt.grid(True)
    plt.title(symbol + " Probability Distributions given Deltas")
    plt.xlabel("Share price ($)")
    plt.show()
