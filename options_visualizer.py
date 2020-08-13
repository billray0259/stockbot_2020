from td_api import Account
from data_handler import DataHandler
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

acc = Account("keys.json")

symbol = input("Enter Symbol: ")
days = int(input("Enter Days Out: "))
strike_count = int(input("Enter Strike Count: "))

from_date = datetime.now()
to_date = from_date + timedelta(days=days)
data = acc.get_options_chain(symbol, from_date, to_date, strike_count=strike_count)
# data.to_csv("temp.csv")
# data = pd.read_csv("temp.csv")
mark = acc.get_quotes([symbol])["mark"].iloc[0]

# data = data[data["putCall"] == "CALL"]
data["time"] = (pd.to_datetime(data["expirationDate"], unit="ms") - datetime.now()).dt.total_seconds() / (60 * 60 * 24)
data["ptheta"] = data["theta"] / data["mark"]

call = data[data["putCall"] == "CALL"]
put = data[data["putCall"] == "PUT"]

call["pitm"] = 100*(1-call["strikePrice"]/mark)
put["pitm"] = 100*(put["strikePrice"]/mark-1)

greeks = ["delta", "gamma", "theta", "vega"]

# fig, axis = plt.subplots(2, 2)

# axis = np.concatenate(axis)
# print(axis)

# for i, ax in enumerate(axis):
while True:
    item = input("Enter Greek: ")
    if item == "quit":
        break
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(put["time"], put["pitm"], put[item], marker="v")
    ax.scatter(call["time"], call["pitm"], call[item], marker="^")
    ax.set_title("%s vs Time and Strike" % item)
    ax.set_xlabel("Days to expiration")
    ax.set_ylabel("% ITM")
    ax.set_zlabel(item)

    plt.show(block=False)