from td_api import Account
from data_handler import DataHandler
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import curve_fit
from scipy.misc import derivative

'''

A small script that plots option greeks (or any other attribute returned by TD-Ameritrade's API) in 3D, relative to the percent in the money and the days until expiration.

Useful for visualizing how attributes change over time and strike.

'''

acc = Account("keys.json")

symbol = input("Enter Symbol: ")
days = int(input("Enter Days Out: "))
strike_count = int(input("Enter Strike Count: "))

from_date = datetime.now()
to_date = from_date + timedelta(days=days)
data = acc.get_options_chain(symbol, from_date, to_date, strike_count=strike_count)
mark = acc.get_quotes([symbol])["mark"].iloc[0]

data["time"] = (pd.to_datetime(data["expirationDate"], unit="ms") - datetime.now()).dt.total_seconds() / (60 * 60 * 24)
data["ptheta"] = data["theta"] / data["mark"]
call = data[data["putCall"] == "CALL"]
put = data[data["putCall"] == "PUT"]
# put["delta"] = put["delta"] + 1

# date_groups = call.groupby(["expirationDate"])

pd.set_option('mode.chained_assignment', None)
call["pitm"] = 100*(1-call["strikePrice"]/mark)
put["pitm"] = 100*(put["strikePrice"]/mark-1)
pd.set_option('mode.chained_assignment', 'warn')

greeks = ["delta", "gamma", "theta", "vega"]

while True:
    item = input("Enter Greek: ")
    if item == "exit":
        break
    elif item == "list":
        print(", ".join(data.columns))
        continue
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(put["time"], put["pitm"], put[item], marker="v")
        ax.scatter(call["time"], call["pitm"], call[item], marker="^")        

        ax.set_title("%s: %s vs Time and Strike" % (symbol, item))
        ax.set_xlabel("Days to expiration")
        ax.set_ylabel("% ITM")
        ax.set_zlabel(item)

        plt.show()
    except KeyError:
        print("Unknown Key")
        
