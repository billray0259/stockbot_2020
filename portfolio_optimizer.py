import numpy as np
from td_api import Account
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
from data_handler import DataHandler
import os
import random

periods = 26*252

def get_optimal_portfolio_weights(avg_returns, covariance_mat):

    def negative_sharpe(weights):
        avg_return = np.sum(avg_returns * weights)
        volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_mat, weights)))
        return -avg_return / volatility

    constraints = ({
        "type": "eq",
        "fun": lambda x: np.sum(x) - 1
    })

    bounds = tuple((0, 1) for x in range(len(avg_returns)))

    initial_weights = np.random.random(len(avg_returns))
    initial_weights /= np.sum(initial_weights)
    
    best_weights = sco.minimize(negative_sharpe, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)["x"]

    return np.round(best_weights, 7)

# tickers = ["AAPL", "MSFT", "SPY", "GLD"]
# acc = Account("keys.json")
# histories = pd.DataFrame(columns=tickers)
# for ticker in tickers:
#     history = acc.history(ticker, 30, 365, frequency_type="minute")
#     histories[ticker] = history["close"]

# histories.to_csv("temp.csv")

dh = DataHandler("vol500k")
histories = pd.DataFrame()
files = os.listdir(dh.filled_histories_dir)
random.seed(1)
random.shuffle(files)
for file in files[:100]:
    ticker = os.path.splitext(file)[0]
    history = pd.read_hdf(os.path.join(dh.filled_histories_dir, file))
    histories[ticker] = history[ticker + "_close"]
histories.to_csv("temp.csv")

histories = pd.read_csv("temp.csv", index_col="datetime")

returns = np.log(histories / histories.shift(1))[1:].replace([np.inf, -np.inf], np.nan)
returns = returns[returns.columns[returns.isnull().sum() == 0]]
covariance_mat = returns.cov() * periods
avg_returns = returns.mean() * periods


# n = 10000

# rets = np.zeros(n)
# vols = np.zeros(n)

# for i in range(n):
#     weights = np.random.rand(len(avg_returns))
#     weights /= np.sum(weights)
#     ret = np.dot(weights, avg_returns) * periods
#     vol = np.sqrt(np.dot(weights.T, np.dot(covariance_mat * periods, weights)))
#     rets[i] = ret
#     vols[i] = vol

print("Getting optimal weights")
for _ in range(10):
    opt_weights = get_optimal_portfolio_weights(avg_returns, covariance_mat)

    ret = np.sum(opt_weights * avg_returns)
    vol = np.sqrt(np.dot(opt_weights.T, np.dot(covariance_mat, opt_weights)))
    weights, labels = [], []
    for weight, label in zip(opt_weights, avg_returns.index):
        if weight > 0:
            weights.append(weight)
            labels.append(label)
    print(dict(zip(labels, weights)))
    print(ret / vol)



plt.pie(weights, labels=labels, autopct="%1.1f%%", startangle=90)

# plt.scatter(vols, rets, c=rets/vols)
plt.show()


