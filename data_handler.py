import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from tqdm import tqdm
from td_api import Account
import numpy as np
import time
from scipy.cluster.vq import kmeans, whiten
import pickle
import math

class DataHandler:

    def __init__(self, name, finviz_filter="", keys_json="keys.json", data_dir="data/"):
        """ Retrieve, store, and process data from TD Ameritrade and Finviz

        Args:
            name (str): A name for the subdirecotry in the data_dir directory.
            finviz_filter (str, optional): The end of a finviz URL. Used in save_finviz() to filter the finviz results. Defaults to "".
            keys_json (str, optional): Path to JSON file used to connect to the TD Ameritrade API. Must contain refresh_token, client_id, and account_id. Defaults to "keys.json".
            data_dir (str, optional): Path to a direcotory to store / read data from. Defaults to "data/".
        """
        self.finviz_filter = finviz_filter
        self.name = name
        self.keys_json = keys_json
        self.data_dir = os.path.join(data_dir, name)

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        # Stores the contents of the finviz table filtered by finviz_filter
        # Generated by self.save_finviz()
        self.finviz_file = os.path.join(self.data_dir, "finviz.h5")

        # Directory that contains the resulting candles from the TD Ameritrade API in the form TICKER.h5
        # Each ticker has its own file
        # Generated by self.collect_histories()
        self.histories_dir = os.path.join(self.data_dir, "histories")
        
        # Similar to histories_dir
        # The files in this directory all have the same length and frequency
        # Missing data from histories_dir is filled by assuming if the candle was missing there was no trading volume
        self.filled_histories_dir = os.path.join(self.data_dir, "filled_histories")

        # Files that contain concatenated DataFrames of the histories_dir and filled_histories_dir
        # These DataFrames have 5 columns per symbol in the format:
        # TICKER_open, TICKER_high, TICKER_low, TICKER_close, TICKER_volume
        # Dataframe index is the datetime object corresponding to the candles
        self.joined_file = os.path.join(self.data_dir, "joined.h5")
        self.filled_file = os.path.join(self.data_dir, "filled.h5")

        # Stores a correlation matrix for every symbol in filled_file
        self.correlation_file = os.path.join(self.data_dir, "correlation.h5")

        # Stores a dictionary with values being lists of correlated tickers and values being floats related to the average
        # correlation of the members in that group
        # Higher key value indicated more correlation
        self.groups_file = os.path.join(self.data_dir, "groups.pkl")

    def save_finviz(self, pages=None):
        """ Saves a finviz table from this DataHandle's finviz_filter

        Args:
            pages (int, optional): How many pages to retrieve from finviz. Each page has 20 rows. Defaults to None, meaning all pages.
        """
        session = requests.Session()
        session.headers.update({"User-Agent": "Chrome 80"})

        url_base = "https://finviz.com/screener.ashx?" + self.finviz_filter + "&r="

        i = 1
        df = None
        while pages is None or i < pages*20:
            url = url_base + str(i)
            response = session.get(url)
            soup = BeautifulSoup(response.text, features="lxml")

            tables = soup.find_all("table")
            data_table = tables[16]
            rows = data_table.find_all("tr")

            if len(rows) == 2:
                pages = 0

            if df is None:
                columns = []
                for cell in rows[0].find_all("td"):
                    columns.append(cell.text)
                df = pd.DataFrame(columns=columns)

            for html_row in rows[1:]:
                row = []
                cells = html_row.find_all("td")
                for cell in cells:
                    row.append(cell.text)
                df = df.append(dict(zip(columns, row)), ignore_index=True)
            print("Collected:", len(df), end="\r", flush=True)
            i += 20
        df = df.drop(columns=["No."]).set_index("Ticker")
        if (df.values[-1] == df.values[-2]).all():
            df = df[:-1]
        df.to_hdf(self.finviz_file, "df", "w")

    def collect_histories(self, frequency=15, frequency_type="minute", days=365):
        """ Uses the TD Ameritrade API to collect candles for all tickers in the finviz file

        Args:
            frequency (int, optional): Number of frequency_type in a candle. 1, 5, 10, 15, or 30 if frequency_type is "minute" otherwise 1. Defaults to 15.
            frequency_type (str, optional): The units of frequency. minute, daily, weekly, or monthly. Defaults to "minute".
            days (int, optional): Number of days in the time frame of interest. Defaults to 365.
        """
        if not os.path.exists(self.histories_dir):
            os.mkdir(self.histories_dir)
        
        account = Account(self.keys_json)

        finviz_df = pd.read_hdf(self.finviz_file)
        tickers = finviz_df.index

        last_request_time = 0
        for ticker in tqdm(tickers):
            ticker_history_save_file = os.path.join(self.histories_dir, "%s.h5" % ticker)
            if os.path.exists(ticker_history_save_file):
                # TODO load what is currently there and add any new data
                continue
            time.sleep(max(0, 0.5 - (time.time() - last_request_time)))
            last_request_time = time.time()
            history_df = account.history(ticker, frequency, days, frequency_type=frequency_type)
            if len(history_df) > 1:
                history_df.to_hdf(ticker_history_save_file, "df", "w")

    def join_candles(self, filled=False):
        """ Joines DataFrames saved to files in the histories or filled_histories directories depending on the value of 'filled'

        Args:
            filled (bool, optional): If filled is True joins DataFrames from filled_histories directory else joins DataFrames from
            histories directory. Defaults to False.
        """
        if not os.path.exists(self.filled_histories_dir):
            os.mkdir(self.filled_histories_dir)
        
        dataframes = []
        if filled:
            histories_dir = self.filled_histories_dir
        else:
            histories_dir = self.histories_dir

        for data_file_name in tqdm(os.listdir(histories_dir), desc="Joining candles from %s" % histories_dir):
            ticker, _ = os.path.splitext(os.path.basename(data_file_name))
            data_df = pd.read_hdf(os.path.join(histories_dir, data_file_name), index_col="datetime")
            if not filled:
                data_df.columns = list(map(lambda column: ticker + "_" + column, data_df.columns))
            dataframes.append(data_df)

        big_df = dataframes[0].join(dataframes[1:])
        if filled:
            big_df.to_hdf(self.filled_file, "df", "w")
            print("Saved joined candles to " + self.filled_file)
        else:
            big_df.to_hdf(self.joined_file, "df", "w")
            print("Saved joined candles to " + self.joined_file)
    
    def zero_volume_fill(self):
        """ Polulates the filled file with a copy of the joined file where the na values have been filled in a way
            that assumes na candles represent candles with zero volume and no price change.
        """
        tickers = pd.read_hdf(self.finviz_file).index
        joined = pd.read_hdf(self.joined_file)

        for ticker in tqdm(tickers):
            try:
                save_file = os.path.join(self.filled_histories_dir, "%s.h5" % ticker)
                if os.path.exists(save_file):
                    continue
                df = joined.filter(regex="^%s_" % ticker)
                open_, high, low, close, volume = df.columns
                new_dict = {open_: [], high: [], low: [], close: [], volume: []}
                price_cols = [open_, high, low, close]

                last_close = 0
                for time, row in df.iterrows():
                    if row.isna().any():
                        for col in price_cols:
                            row[col] = last_close
                        row[volume] = 0
                    else:
                        last_close = row[close]

                    for key in new_dict:
                        new_dict[key].append(row[key])

                df = pd.DataFrame(new_dict, index=df.index)
                df.to_hdf(save_file, "df", "w")
            except Exception as e:
                print("failed on", ticker)
                print(e)
                
    def correlation_matrix(self, method="pearson", rolling_window=26):
        """ Populates the correlation file with a DataFrame where the indecies and columns are all the tickers
            in the filled file and the values are correlation matricies of the SMA

        Args:
            method (str, optional): Method to pass to pandas' corr(). Defaults to "pearson".
            rolling_window (int, optional): Number of candles in SMA. Defaults to 26.
        """
        candles = pd.read_hdf(self.filled_file)
        opens = candles.filter(like="_open")
        closes = candles.filter(like="_close")

        opens.columns = list(map(lambda column: column.split("_")[0], opens.columns))
        closes.columns = list(map(lambda column: column.split("_")[0], closes.columns))

        ratios = (closes / opens) - 1
        ratios.fillna(0, inplace=True)
        ratios = ratios.rolling(rolling_window).mean()

        corr_mat = ratios.corr(method=method)
        corr_mat.to_hdf(self.correlation_file, "df", "w")


    def get_groupings(self):
        """ Uses scypi's implementation of kmeans to group the columns of the correlation matrix.
            Saves these groupings to the groups.pkl file as a dictionary with values being lists of
            correlated tickers and values being floats related to the average
            correlation of the members in that group.
            Higher key value indicated more correlation. 
        """
        corr_mat = pd.read_hdf(self.correlation_file)
        tickers = corr_mat.columns
        abs_ndarray = np.abs(corr_mat.to_numpy())

        print("Running kmeans...\t%d" % time.time())
        whitened = whiten(abs_ndarray)
        codebook, distortion = kmeans(whitened, round(math.sqrt(len(corr_mat))), check_finite=False)

        print("Calculating centroid distances...\t%d" % time.time())
        centroid_dists = []
        for centroid in codebook:
            dist = np.linalg.norm(whitened - centroid, axis=1)
            centroid_dists.append(dist)
        
        centroid_dists = np.array(centroid_dists).transpose()
        groups = [[] for _ in range(len(codebook))]

        print("Grouping...\t%d" % time.time())
        for i, dist in enumerate(centroid_dists):
            groups[dist.argmin()].append(tickers[i])
        
        std = corr_mat.values.std(ddof=1)
        means = {}
        for group in groups:
            group_corr = corr_mat.loc[group, group]
            mean_corr = group_corr.mean().mean()
            means[mean_corr/std] = group
        
        print("Saving to %s....\t%d" % (self.groups_file, time.time()))
        with open(self.groups_file, "wb") as groups_file:
            pickle.dump(means, groups_file)

    def handle_data(self):
        """ Calls all the methods in the correct order
        """
        self.save_finviz()
        self.collect_histories()
        self.join_candles()
        self.zero_volume_fill()
        self.join_candles(filled=True)
        self.correlation_matrix()
        self.get_groupings()

if __name__ == "__main__":

    data_handler = DataHandler("vol500k", "v=111&f=ind_stocksonly,sh_avgvol_o500,sh_price_o2")
    data_handler.save_finviz()
