import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense
from keras.models import Sequential
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from data_handler import DataHandler

class MultiSymbolBinaryClassifier:

    def __init__(self, name):
        self.data_handler = DataHandler(name)

    def preprocess(self, num_groups=3, candles_per_example=64):
        finviz = pd.read_hdf(self.data_handler.finviz_file)
        filled = pd.read_hdf(self.data_handler.filled_file)

        with open(self.data_handler.groups_file, "rb") as groups_file:
            groups = pickle.load(groups_file)

        sorted_keys = sorted(groups.keys(), reverse=True)
        print("Multi-Symbol Binary Classifier preprocessing from" + self.data_handler.filled_file)
        for highest_corr in tqdm(sorted_keys[:num_groups]):
            tickers = groups[highest_corr]
            biggest_ticker = max(tickers, key=lambda ticker: finviz["Market Cap"][ticker])

            columns = []
            for ticker in tickers:
                columns.extend([ticker + "_open", ticker + "_high", ticker + "_low", ticker + "_close", ticker + "_volume"])

            tickers_df = filled[columns]

            x = []
            y = []
            closes = tickers_df.filter(like="_close")
            for i in tqdm(range(candles_per_example, len(tickers_df)-1)):
                example = tickers_df.iloc[i-candles_per_example:i]
                next_close = closes.iloc[i+1]

                means = np.mean(example)
                stds = np.std(example)
                example = (example - means) / stds
                next_close = (next_close - means.filter(like="_close")) / stds.filter(like="close")
                this_close = example.filter(like="_close").iloc[-1]
                label = this_close < next_close
                label = np.array(list(map(lambda b: [0, 1] if b else [1, 0], label.to_numpy())))
                example = example.to_numpy()

                x.append(example)
                y.append(label)

            x, y = np.array(x), np.array(y)

            dir_path = os.path.join(self.data_handler.data_dir, "msbc_training", biggest_ticker + "_group")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            x_file = os.path.join(dir_path, "x.npy")
            y_file = os.path.join(dir_path, "y.npy")
            np.save(x_file, x)
            np.save(y_file, y)
    
    def train(self, data_file_name=None):
        model = Sequential([
            LSTM(32, activation="relu", input_shape=(64, 5), return_sequences=True),
            LSTM(16, activation="relu", return_sequences=True),
            LSTM(8, activation="relu"),
            Dense(8, activation="relu"),
            Dense(2, activation="softmax")
        ])


        model.compile(loss="categorical_crossentropy", optimizer="adam",
                    metrics=["categorical_accuracy"])

        epochs = 10
        # model.fit(x, y, epochs=epochs, batch_size=256, validation_split=0.2)
        # model.save("model")


if __name__ == "__main__":
    pass
