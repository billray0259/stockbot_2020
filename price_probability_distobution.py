from td_api import Account
from data_handler import DataHandler
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad

curves = {
    "logistic": lambda x, u, s: 0.5 + 0.5*np.tanh((x-u)/(2*s))
}

distrobutions = {
    "logistic": lambda x, u, s: np.exp(-(x-u)/s)/(s*(1+np.exp(-(x-u)/s))**2)
}


def get_pdfs_from_deltas(options_chain, curve_type="logistic"):
    
    data = options_chain
    data["delta"] = np.abs(data["delta"])
    
    date_groups = data.groupby(["expirationDate"])
    
    pdfs = {}
    for group in date_groups.groups:
        indicies = date_groups.groups[group]
        group_data = data.loc[indicies, :]
        label = pd.to_datetime(data["expirationDate"][indicies[0]], unit="ms").strftime("%Y-%m-%d")

        calls = group_data[group_data["putCall"] == "CALL"]
        puts = group_data[group_data["putCall"] == "PUT"]

        call_x = calls["strikePrice"]
        call_y = calls["delta"]

        put_x = puts["strikePrice"]
        put_y = puts["delta"]
        
        curve = curves[curve_type]

        (call_u, call_s), pcov = curve_fit(curve, call_x, call_y, (calls["strikePrice"][0], -1))
        call_s = -call_s

        call_err = np.sqrt(np.sum(np.diag(pcov)))
        
        (put_u, put_s), pcov = curve_fit(curve, put_x, put_y, (puts["strikePrice"][0], 1))

        put_err = np.sqrt(np.sum(np.diag(pcov)))

        call_weight = 1-(call_err/(call_err + put_err))
        put_weight = 1-(put_err/(call_err + put_err))

        u = call_u * call_weight + put_u * put_weight
        s = call_s * call_weight + put_s * put_weight

        # pdfs["call " + label] = (call_u, call_s)
        # pdfs["put " + label] = (put_u, put_s)
        pdfs[label] = u, s
    
    return pdfs


if __name__ == "__main__":
    acc = Account("keys.json")

    # symbol = input("Enter Symbol: ")
    # days = int(input("Enter Days Out: "))
    # strike_count = int(input("Enter Strike Count: "))

    symbol="AMD"
    days=21
    strike_count=100

    # from_date = datetime.now() + timedelta(days=1)
    # to_date = from_date + timedelta(days=days)
    # data = acc.get_options_chain(symbol, from_date, to_date, strike_count=strike_count)
    # data.to_csv("temp.csv")
    data = pd.read_csv("temp.csv", index_col="symbol")
    mark = acc.get_quotes([symbol])["mark"].iloc[0]

    pdfs = get_pdfs_from_deltas(data)
    for label in pdfs:
        u, s = pdfs[label]
        label += "\nmean: %.2f\nstd: %.2f\n" % (u, s)

        distrobution = distrobutions["logistic"]

        x = np.linspace(mark-5*s, mark+5*s, 100)
        y = distrobution(x, u, s)
        loss_odds = quad(lambda x: distrobution(x, u, s), 0, mark)[0]

        label += "profit: %.2f" % (100*(1-loss_odds)) + "%\n"
        plt.plot(x, y, label=label)

    plt.vlines(mark, *plt.gca().get_ylim(), label="Mark %.4f" % mark)
    plt.legend()
    plt.grid(True)
    plt.show()
