import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from tqdm import tqdm
from td_api import Account
import numpy as np
from threading import Thread
import concurrent.futures
import time


class TrainingDataCollector:

    def __init__(self, name, finviz_filter, keys_json="keys.json", data_dir="data/", training_data_dir="training_data/"):
        self.finviz_filter = finviz_filter
        self.name = name
        self.keys_json = keys_json
        self.data_dir = data_dir
        self.training_data_dir = training_data_dir

        self.finviz_file = self.data_dir + self.name + "_finviz.csv"
        self.histories_dir = self.data_dir + self.name + "_histories/"

        self.joined_file = self.data_dir + self.name + "_joined.csv"

        self.examples_file = self.training_data_dir + self.name + "_examples.npy"
        self.labels_file = self.training_data_dir + self.name + "_labels.npy"

    def save_finviz_csv(self, pages=None):
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
        df.to_csv(self.finviz_file)

    def collect_histories(self, frequency=15, frequency_type="minute", days=365):
        account = Account(self.keys_json)

        if not os.path.exists(self.histories_dir):
            os.mkdir(self.histories_dir)

        finviz_df = pd.read_csv(self.finviz_file)
        tickers = finviz_df["Ticker"]

        last_request_time = 0
        for ticker in tqdm(tickers):
            ticker_history_save_file = self.histories_dir + "%s.csv" % ticker
            if os.path.exists(ticker_history_save_file):
                # TODO load what is currently there and add any new data
                continue
            time.sleep(max(0, 0.5 - (time.time() - last_request_time)))
            last_request_time = time.time()
            history_df = account.history(ticker, frequency, days, frequency_type=frequency_type)
            if len(history_df) > 0:
                history_df.to_csv(ticker_history_save_file)
            

    def join_candles_csv(self, save_to_file=True, dropna=True):
        dataframes = []
        for data_file_name in tqdm(os.listdir(self.histories_dir), desc="Joining candles from %s" % self.histories_dir):
            ticker = data_file_name[:-len(".csv")]
            data_df = pd.read_csv(
                self.histories_dir + data_file_name, index_col="datetime")
            data_df.columns = list(
                map(lambda column: ticker + "_" + column, data_df.columns))
            dataframes.append(data_df)

        big_df = dataframes[0].join(dataframes[1:])
        if dropna:
            big_df = big_df.dropna(axis=1)

        if save_to_file:
            big_df.to_csv(self.joined_file)
            print("Saved joined candles to " + self.joined_file)
        else:
            return big_df

    def preprocess(self, example_length=64, save_to_file=True):
        candles = pd.read_csv(self.joined_file, index_col="datetime")
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
            next_closes = (next_closes - means.filter(like="close")
                           ) / stds.filter(like="close")

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


if __name__ == "__main__":

    tdc = TrainingDataCollector("all", "v=111&o=-marketcap")
    # tdc.collect_histories(days=365)
    tdc.join_candles_csv(dropna=False)
