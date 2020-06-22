import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import timedelta

from data_handler import DataHandler


def multi_symbol_binary_classification(data_handler):
    with open(data_handler.groups_file, "rb") as groups_file:
        groups = pickle.load(groups_file)
    
    sorted_keys = sorted(groups.keys(), reverse=True)
    group = groups[sorted_keys[1]]

    finviz = pd.read_hdf(data_handler.finviz_file)
    not_etf = []
    names = []
    for ticker in group:
        if finviz["Industry"][ticker] != "Exchange Traded Fund":
            not_etf.append(ticker)
            names.append(finviz["Company"][ticker])
        
    print(not_etf)
    print(len(not_etf))
            

    filled = pd.read_hdf(data_handler.filled_file)
    closes = filled.filter(like="_close")
    closes.columns = list(map(lambda column: column.split("_")[0], closes.columns))
    group_prices = closes[not_etf]
    group_prices.index = pd.to_datetime(group_prices.index)
    df = group_prices.resample("D").mean().dropna()
    df = df/df.mean()

    for key in df:
        plt.plot(df[key])
    plt.legend(names)
    plt.show()
    
    




if __name__ == "__main__":
    dh = DataHandler("all")
    multi_symbol_binary_classification(dh)
