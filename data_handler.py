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

    def __init__(self, name, finviz_filter="", keys_json="keys.json", data_dir="data/", training_data_dir="training_data"):
        self.finviz_filter = finviz_filter
        self.name = name
        self.keys_json = keys_json
        self.data_dir = os.path.join(data_dir, name)
        self.training_data_dir = training_data_dir

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        self.finviz_file = os.path.join(self.data_dir, "finviz.h5")

        self.histories_dir = os.path.join(self.data_dir, "histories")
        if not os.path.exists(self.histories_dir):
            os.mkdir(self.histories_dir)
        
        self.filled_histories_dir = os.path.join(self.data_dir, "filled_histories")
        if not os.path.exists(self.filled_histories_dir):
            os.mkdir(self.filled_histories_dir)

        self.joined_file = os.path.join(self.data_dir, "joined.h5")
        self.filled_file = os.path.join(self.data_dir, "filled.h5")

        self.correlation_file = os.path.join(self.data_dir, "correlation.h5")

        self.examples_file = os.path.join(self.training_data_dir, self.name, "examples.npy")
        self.labels_file = os.path.join(self.training_data_dir, self.name, "labels.npy")

        self.groups_file = os.path.join(self.data_dir, "groups.pkl")

    def save_finviz(self, pages=None):
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

    def join_candles(self, save_to_file=True, filled=False):
        dataframes = []
        if filled:
            histories_dir = self.filled_histories_dir
        else:
            histories_dir = self.histories_dir

        for data_file_name in tqdm(os.listdir(histories_dir), desc="Joining candles from %s" % histories_dir):
            ticker, _ = os.path.splitext(os.path.basename(data_file_name))
            data_df = pd.read_hdf(os.path.join(histories_dir, data_file_name), index_col="datetime")
            if not filled:
                data_df.columns = list(
                    map(lambda column: ticker + "_" + column, data_df.columns))
            dataframes.append(data_df)

        big_df = dataframes[0].join(dataframes[1:])

        if save_to_file:
            if filled:
                big_df.to_hdf(self.filled_file, "df", "w")
                print("Saved joined candles to " + self.filled_file)
            else:
                big_df.to_hdf(self.joined_file, "df", "w")
                print("Saved joined candles to " + self.joined_file)
        else:
            return big_df
    
    def zero_volume_fill(self):        
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
        

    def preprocess(self, example_length=64, save_to_file=True):
        candles = pd.read_hdf(self.joined_file, index_col="datetime")
        all_examples = []
        all_labels = []
        tickers = set(
            map(lambda column: column.split("_")[0], candles.columns))
        closes = candles.filter(like="close")
        for i in tqdm(range(example_length, len(candles)-1), desc="Preprocessing candles from %s" % self.joined_file):
            examples = candles.iloc[i-example_length:i]
            next_closes = closes.iloc[i+1]

            means = np.mean(examples)
            stds = np.std(examples)
            examples = (examples - means) / stds
            next_closes = (next_closes - means.filter(like="close")) / stds.filter(like="close")

            for ticker in tickers:
                example = examples.filter(regex="^%s_" % ticker)
                close_column = ticker + "_close"
                close = example[close_column].values[-1]
                next_close = next_closes[close_column]
                label = np.zeros(2)
                label[int(close < next_close)] = 1
                all_examples.append(example.to_numpy())
                all_labels.append(label)

        examples, labels = np.array(all_examples), np.array(all_labels)
        if save_to_file:
            np.save(examples, self.examples_file)
            np.save(labels, self.labels_file)
            print("Saved training data to %s and %s" %
                  (self.examples_file, self.labels_file))
        else:
            return examples, labels

    def handle_data(self):
        self.save_finviz()
        self.collect_histories()
        self.join_candles()
        self.zero_volume_fill()
        self.join_candles(filled=True)
        self.correlation_matrix()
        self.get_groupings()

if __name__ == "__main__":

    data_handler = DataHandler("test", "v=111&f=ind_gold,sec_basicmaterials,sh_price_u1&o=-marketcap")
    data_handler.handle_data()
