from td_api import Account
from data_handler import DataHandler
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy import stats
import math


def get_pdfs_from_deltas(options_chain, distrobution=stats.logistic):
    
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
        
        curve = lambda x, u, s: distrobution.cdf(x, u, s)

        call_popt, call_pcov = curve_fit(curve, call_x, call_y, (-mark, mark/10))
        call_popt[0] *= -1
        (call_u, call_s) = call_popt

        call_err = np.sqrt(np.sum(np.diag(call_pcov) / call_popt))

        put_curve = lambda x, u, s: distrobution.cdf(x, u, s)
        
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
    data.to_csv("temp.csv")
    # data = pd.read_csv("temp.csv", index_col="symbol")
    mark = acc.get_quotes([symbol])["mark"].iloc[0]

    pdfs = get_pdfs_from_deltas(data, distrobution=stats.logistic)
    for label in pdfs:
        u, s, errs = pdfs[label]
        err = 100 * np.linalg.norm(errs / (u, s))
        label += "\nmean: %.2f±%.2f%%\nstd: %.2f±%.2f%%\n" % (u, 100*errs[0]/u, s, 100*errs[1]/s)

        distrobution = stats.logistic.pdf

        x = np.linspace(mark-5*s, mark+5*s, 100)
        y = distrobution(x, u, s)

        loss_odds = quad(lambda x: distrobution(x, u, s), 0, mark)[0]
        s_sign = 1 if u > mark else -1
        loss_odds_min = quad(lambda x: distrobution(x, u+errs[0], s - errs[1]*s_sign), 0, mark)[0]
        loss_odds_max = quad(lambda x: distrobution(x, u-errs[0], s + errs[1]*s_sign), 0, mark)[0]

        label += "profit: %.2f%%-%.2f%%-%.2f%%\n" % (100*(1-loss_odds_max), 100*(1-loss_odds), 100*(1-loss_odds_min))
        label += "err: %.2f\n" % err
        plt.plot(x, y, label=label)

    plt.vlines(mark, *plt.gca().get_ylim(), label="Mark %.4f" % mark)
    plt.legend()
    plt.grid(True)
    plt.title(symbol + " Probability Distributions given Deltas")
    plt.xlabel("Share price ($)")
    plt.show()
