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

# def _calc(values):
#     r, u, s, distribution = values
#     dist = lambda x: distribution.pdf(x, u, s)
#     return r - quad(lambda x: dist(x) * x, -np.inf, r)[0] - r * quad(dist, r, np.inf)[0]
    # return r - quad(lambda x: dist(x) * x, -10*r, r)[0] - r * (1-distribution.cdf(r, u, s))
    # x1 = np.linspace(-r, r, num=10000)
    # y1 = dist(x1) * x1
    # x2 = np.linspace(r, 3*r, num=10000)
    # y2 = dist(x2)
    # p = trapz(y2, x1)
    # return r - trapz(y1, x1) - r * trapz(y2, x2)


def get_pdfs_from_marks(options_chain):

    delta_pdfs = get_pdfs_from_deltas(options_chain)

    data = options_chain.loc[:, ["mark", "expirationDate", "putCall", "strikePrice"]]
    data.replace("NaN", np.nan, inplace=True)
    len0 = len(data)
    data.dropna(inplace=True)
    lenf = len(data)
    if lenf < len0:
        print("Warning dropped %d rows containing NaN" % (len0-lenf))

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
        call_y = calls["mark"]

        put_x = puts["strikePrice"]
        put_y = puts["mark"]

        

        def curve(r, u, s):
            norm = stats.norm(u, s)
            cdf = norm.cdf
            pdf = norm.pdf
            profit_below_strike = u * cdf(r) - s**2 * pdf(r)
            profit_above_strike = r * (1-cdf(r))
            return r - profit_below_strike - profit_above_strike
        
        call_popt, call_pcov = curve_fit(curve, call_x, call_y, (-mark, mark/10))
        # call_popt[0] *= -1
        (call_u, call_s) = call_popt

        plt.scatter(call_x, call_y)
        plt.plot(call_x, curve(call_x, call_u, call_s))
        plt.show()

        call_err = np.sqrt(np.sum(np.diag(call_pcov) / call_popt))

        # def put_curve(r, u, s): return distribution.cdf(x, u, s)

        put_popt, put_pcov = curve_fit(curve, put_x, put_y, (mark, mark/10))
        (put_u, put_s) = put_popt

        plt.scatter(put_x, put_y)
        plt.plot(put_x, curve(put_x, put_u, put_s))
        plt.show()

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

def get_pdfs_from_deltas(options_chain, distribution=stats.logistic):
    
    data = options_chain.loc[:, ["delta", "expirationDate", "putCall", "strikePrice"]]
    data.replace("NaN", np.nan, inplace=True)
    len0 = len(data)
    data.dropna(inplace=True)
    lenf = len(data)
    if lenf < len0:
        print("Warning dropped %d rows containing NaN" % (len0-lenf))
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

        mark = np.mean(put_x)

        call_popt, call_pcov = curve_fit(curve, call_x, call_y, (-mark, mark/10))
        call_popt[0] *= -1
        (call_u, call_s) = call_popt

        call_err = np.sqrt(np.sum(np.diag(call_pcov) / call_popt))

        # put_curve = lambda x, u, s: distribution.cdf(x, u, s)
        
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
    # days = int(input("Enter Days Out: "))
    days=32
    # strike_count = int(input("Enter Strike Count: "))
    strike_count = 150

    # symbol="AMD"
    # days=21
    # strike_count=100

    from_date = datetime.now()
    to_date = from_date + timedelta(days=days)
    data = acc.get_options_chain(symbol, from_date, to_date, strike_count=strike_count)
    # data.to_csv("temp.csv")
    # data = pd.read_csv("temp.csv", index_col="symbol")
    mark = acc.get_quotes([symbol])["mark"].iloc[0]

    pdfs = get_pdfs_from_marks(data)

    # days = [0]
    # means = [mark]
    # mean_errs = [0]
    # stds = [0]
    # std_errs = [0]

    days = []
    means = []
    mean_errs = []
    stds = []
    std_errs = []

    if False:
        for label in pdfs:
            u, s, errs = pdfs[label]
            means.append(u)
            mean_errs.append(errs[0])
            stds.append(s)
            std_errs.append(errs[1])

            expiration = datetime.strptime(label, "%Y-%m-%d")
            days.append((expiration - datetime.now()).days)
        
        mean_line = lambda x, m, b: m * x + b
        std_line = lambda x, m, b, c: m * x**c + b

        days = np.array(days)
        means = np.array(means)
        stds = np.array(stds)

        # plt.plot(days, stds)
        # plt.show()
        
        popts, pcov = curve_fit(mean_line, days, means, p0=(1, mark), sigma=mean_errs)
        mean_m, mean_b = popts
        
        popts, pcov = curve_fit(std_line, days, stds, p0=(1, 0, 0.5), bounds=((-np.inf, -np.inf, 0), (np.inf, np.inf, 1)), sigma=std_errs)
        std_m, std_b, std_c = popts

        mean = lambda x: mean_line(x, mean_m, mean_b)
        std = lambda x: std_line(x, std_m, std_b, std_c)

        # days = np.linspace(0, expirations[-1], num=128)
        # means = mean(days)
        # stds = std(days)
        days_p = days + 0.1
        plt.errorbar(days, mean(days), yerr=std(days))
        plt.errorbar(days[1:], means[1:], yerr=stds, fmt="o")
        
        plt.plot([0, days[-1]], [mark, mark], "k")
        

        # prices = np.linspace(mark - stds[-1], mark + stds[-1], num=128)

        # Z = np.zeros((128, 128))
        # for r in range(128):
        #     for c in range(128):
        #         p = stats.logistic.cdf(prices[r], means[c], stds[c])
        #         Z[r][c] = max(p, 1-p)
        # X, Y = np.meshgrid(days, means)
        # cs = plt.contour(X, Y, Z, levels=np.arange(0, 1, 0.1), colors="k")
        # plt.clabel(cs, inline=True)
        # plt.imshow(Z)
        # plt.set_xticklabels(X)
        # plt.set_yticklabels(Y)
        # # plt.legend()
        plt.show()
        
    else:

        for i, label in enumerate(pdfs):
            u, s, errs = pdfs[label]

            err = 100 * np.linalg.norm(errs / (u, s))
            label += "\nmean: %.2f±%.2f%%\nstd: %.2f±%.2f%%\n" % (u, 100*errs[0]/u, s, 100*errs[1]/s)

            distribution = stats.logistic

            x = np.linspace(u-5*s, u+5*s, 100)
            y = distribution.pdf(x, u, s)

            # loss_odds = quad(lambda x: distribution(x, u, s), 0, mark)[0]
            loss_odds = distribution.cdf(mark, u, s)

            s_sign = 1 if u > mark else -1
            # loss_odds_min = quad(lambda x: distribution(x, u+errs[0], s - errs[1]*s_sign), 0, mark)[0]
            loss_odds_min = distribution.cdf(mark, u+errs[0], s - errs[1]*s_sign)
            # loss_odds_max = quad(lambda x: distribution(x, u-errs[0], s + errs[1]*s_sign), 0, mark)[0]
            loss_odds_max = distribution.cdf(mark, u-errs[0], s + errs[1]*s_sign)

            label += "E[return]: %.2f%%\n" % (100 * (u/mark-1))

            label += "profit: %.2f%%-%.2f%%-%.2f%%\n" % (100*(1-loss_odds_max), 100*(1-loss_odds), 100*(1-loss_odds_min))
            label += "err: %.2f\n" % err
            plt.plot(x, y, label=label)

            # print(quad(lambda x: distribution(x, u, s) * x, 0, u)[0])

            # op_value = u/2 - quad(lambda x: distribution(x, u, s) * x, -10*u, u)[0]

        plt.vlines(mark, *plt.gca().get_ylim(), label="Mark %.4f" % mark)
        plt.legend()
        plt.grid(True)
        plt.title(symbol + " Probability Distributions given Deltas")
        plt.xlabel("Share price ($)")
        plt.show()

    
