import pandas as pd
import numpy as np
import pickle as pkl
import os
from tqdm import tqdm

def join_candles_csv(histories_dir, save_file):
    dataframes = []
    for data_file_name in tqdm(os.listdir(histories_dir), desc="Joining candles from %s\n" % histories_dir):
        ticker = data_file_name[:-len(".csv")]
        data_df = pd.read_csv(
            histories_dir + data_file_name, index_col="datetime")
        data_df.columns = list(
            map(lambda column: ticker + "_" + column, data_df.columns))
        dataframes.append(data_df)

    big_df = dataframes[0].join(dataframes[1:]).dropna(axis=1)

    big_df.to_csv(save_file)
    print("Saved joined candles to " + save_file)
    

def get_examples_from_joined(candles_csv, example_length=64):
    # TODO make sure all candles are the same number of milliseconds apart
    candles = pd.read_csv(candles_csv, index_col="datetime")
    all_examples = []
    all_labels = []
    tickers = set(map(lambda column: column.split("_")[0], candles.columns))
    closes = candles.filter(like="close")
    for i in tqdm(range(example_length, len(candles)-1), desc="Preprocessing candles from %s\n" % candles_csv):
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

    return np.array(all_examples), np.array(all_labels)

if __name__ == "__main__":

    DATA_DIR = "/home/bill/stockbot_2020/data/"
    DATA_FILE = "tech_candles.csv"
    DATA_SAVE_DIR = "/home/bill/stockbot_2020/processed_data/"
    HISTORIES_DIR = "/home/bill/stockbot_2020/data/tech_histories"

    join_candles_csv(HISTORIES_DIR, DATA_DIR + DATA_FILE)

    examples, labels = get_examples_from_joined(DATA_DIR + DATA_FILE)

    np.save(DATA_SAVE_DIR + "x.npy", examples)
    np.save(DATA_SAVE_DIR + "y.npy", labels)
