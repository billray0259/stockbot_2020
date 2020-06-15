import pandas as pd
import numpy as np
import pickle as pkl
import os
from progressbar import ProgressBar, Percentage, AdaptiveETA

DATA_DIR = "/home/bill/stockbot_2020/data/"
DATA_FILE = "small_cap_candles.csv"
DATA_SAVE_DIR = "/home/bill/stockbot_2020/processed_data/"


def get_examples(candles_csv, example_length=64):
    candles = pd.read_csv(candles_csv, index_col="datetime")
    examples = []
    for i in range(example_length, len(candles)-1):
        example = candles.iloc[i-example_length:i]

        next_close = candles.iloc[i+1]["close"]
        label = [0, 0]
        label[example.iloc[-1]["close"] < next_close] = 1

        means = np.mean(example)
        stds = np.std(example)
        example = (example - means) / stds

        examples.append(np.array([example.to_numpy(), np.array(label)]))

    return np.array(examples)

def get_examples_from_joined(candles_csv, example_length=64):
    # TODO make sure all candles are the same number of milliseconds apart
    candles = pd.read_csv(candles_csv, index_col="datetime")
    all_examples = []
    tickers = set(map(lambda column: column.split("_")[0], candles.columns))
    closes = candles.filter(like="close")
    pbar = ProgressBar(widgets=[AdaptiveETA(), "\t", Percentage()])
    for i in pbar(range(example_length, len(candles)-1)):
        examples = candles.iloc[i-example_length:i]
        next_closes = closes.iloc[i+1]

        means = np.mean(examples)
        stds = np.std(examples)
        examples = (examples - means) / stds
        next_closes = (next_closes - means.filter(like="close")) / stds.filter(like="close")

        for ticker in tickers:
            example = examples.filter(like=ticker)
            close_column = ticker + "_close"
            close = example[close_column].values[-1]
            next_close = next_closes[close_column]
            label = np.zeros(2)
            label[int(close < next_close)] = 1
            all_examples.append(np.array([example.to_numpy(), label]))

    return np.array(all_examples)

if __name__ == "__main__":

    examples = get_examples_from_joined(DATA_DIR + DATA_FILE)
    print(examples.shape)
    # all_examples = []
    # for i, file_name in enumerate(os.listdir(DATA_DIR)):
    #     print(round(i/len(os.listdir(DATA_DIR))*100, 2))
        
    #     examples = get_examples(DATA_DIR + file_name)
    #     if len(examples) == 0:
    #         continue

    #     all_examples.append(examples)
    #     print("Preprocessed", file_name)

    np.save(DATA_SAVE_DIR + "small_cap_example.npy", examples)

    # print(np.load(DATA_SAVE_DIR + "small_cap_example.npy", allow_pickle=True).shape)
