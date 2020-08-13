import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_handler import DataHandler
from smabc import SimpleMovingAverageBinaryClassifier

class Trader:

    def wanted_tickers(self):
        """ Gets a list of tickers whose data should be passed to get_trades

        Returns:
            list<string>: List of tickers whose data should be passed to get_trades
        """
        pass

    def get_holdings(self, data, current_holdings):
        """ Calculates which stocks to buy / sell

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        return {}


class BackTest:

    def __init__(self, name, trader, initial_balance=100000):
        self.data_handler = DataHandler(name)
        self.data = pd.read_hdf(self.data_handler.filled_file)
        self.trader = trader

        self.balance = 10000
        self.holdings = {}

    
    def test(self):
        self.data["value"] = np.zeros(len(self.data))
        for i, row in self.data.iterrows():
            wanted_tickers = self.trader.wanted_tickers()
            columns = []
            for ticker in wanted_tickers:
                columns.extend([ticker + "_open", ticker + "_high", ticker + "_low", ticker + "_close", ticker + "_volume"])
            wanted_df = self.data[columns][:i]

            new_holdings = self.trader.get_holdings(wanted_df, self.holdings)

            for key in self.holdings:
                self.order(key, "sell", self.holdings[key], i)

            self.data["value"][i] = self.balance
            print(str(i) + str(round(self.balance, 2)), end="\r", flush=True)
            
            for key in new_holdings:
                share_price = self.data[ticker + "_open"][i]
                amount = int(new_holdings[key] * self.balance / share_price)
                self.order(key, "buy", amount, i)

    
    def order(self, ticker, side, amount, index):
        share_price = self.data[ticker + "_open"][index]
        total_price = share_price * amount
        if ticker not in self.holdings:
            self.holdings[ticker] = 0
        
        if side.lower() == "buy":
            self.balance -= total_price
            self.holdings[ticker] += amount
        elif side.lower() == "sell":
            self.balance += total_price
            self.holdings[ticker] -= amount


class AAPLTrader(Trader):

    def wanted_tickers(self):
        return ["TSLA"]
    
    def get_holdings(self, data, current_holdings):
        return {"TSLA": 1}
    
if __name__ == "__main__":
    name = "stocks_only"
    # trader = SimpleMovingAverageBinaryClassifier(name)
    trader = AAPLTrader()
    test = BackTest(name, trader)
    test.test()
    test.data["value"].plot()
    plt.show()
