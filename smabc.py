import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
import math

from data_handler import DataHandler

import matplotlib.pyplot as plt

class SimpleMovingAverageBinaryClassifier:

    def __init__(self, name):
        self.data_handler = DataHandler(name)
        self.dir = os.path.join(self.data_handler.data_dir, "smabc_training")

    def preprocess(self, density=0.8, win1_start=1, win1_end=256, win1_step=8, win2_start=2, win2_end=1024, win2_step=32):
        candles = pd.read_hdf(self.data_handler.filled_file)
        opens = candles.filter(like="_open")
        closes = candles.filter(like="_close")

        opens.columns = list(map(lambda column: column.split("_")[0], opens.columns))
        closes.columns = list(map(lambda column: column.split("_")[0], closes.columns))
        ups = closes > opens

        avg = (opens + closes) / 2
        vectors = []
        for i in tqdm(range(win2_end, len(avg))[:1]):
            win1 = win1_start
            while win1 < win1_end:
                win2 = win2_start
                while win2 < win2_end:
                    if win1 < win2:
                        past = avg[i-win2:i]
                        sma1 = past.rolling(win1).mean().iloc[-1]
                        sma2 = past.rolling(win2).mean().iloc[-1]
                        print(sma1.isna().sum(), sma2.isna().sum())
                        vec = np.where(sma1 > sma2, 1, -1)
                        vectors.append(vec)
                        print(vec)
                        exit()
                    win2 += max(round((1-density) * win2), win1_step)
                win1 += max(round((1-density) * win1), win2_step)
        ups = ups[win2_end: len(avg)]

        vectors = np.array(vectors)
        print(vectors.shape)


        # sorted_keys = sorted(groups.keys(), reverse=True)
        # print("Multi-Symbol Binary Classifier preprocessing from" + self.data_handler.filled_file)
        # for highest_corr in tqdm(sorted_keys[:num_groups]):
        #     tickers = groups[highest_corr]
        #     biggest_ticker = max(tickers, key=lambda ticker: finviz["Market Cap"][ticker])

        #     columns = []
        #     for ticker in tickers:
        #         columns.extend([ticker + "_open", ticker + "_high", ticker + "_low", ticker + "_close", ticker + "_volume"])

        #     tickers_df = filled[columns]

        #     x = []
        #     y = []
        #     closes = tickers_df.filter(like="_close")
        #     for i in tqdm(range(candles_per_example, len(tickers_df)-1)):
        #         example = tickers_df.iloc[i-candles_per_example:i]
        #         next_close = closes.iloc[i+1]

        #         means = np.mean(example)
        #         stds = np.std(example)
        #         example = (example - means) / stds
        #         next_close = (next_close - means.filter(like="_close")) / stds.filter(like="close")
        #         this_close = example.filter(like="_close").iloc[-1]
        #         label = this_close < next_close
        #         label = np.array(list(map(lambda b: [0, 1] if b else [1, 0], label.to_numpy())))
        #         example = example.to_numpy()

        #         x.append(example)
        #         y.append(label)

        #     x, y = np.array(x), np.array(y)

        #     dir_path = os.path.join(self.dir, biggest_ticker + "_group")
        #     if not os.path.exists(dir_path):
        #         os.makedirs(dir_path)

        #     x_file = os.path.join(dir_path, "x.npy")
        #     y_file = os.path.join(dir_path, "y.npy")
        #     np.save(x_file, x)
        #     np.save(y_file, y)

    def train(self):
        pass
        # from keras.callbacks import ModelCheckpoint
        # from keras.layers import LSTM, Dense, Reshape
        # from keras.models import Sequential
        # from keras.activations import softmax
        # import os
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        # group_dirs = self.group_dirs

        # if group_directory is not None:
        #     group_dirs = [group_directory]

        # for group_dir in group_dirs:
        #     x_file = os.path.join(group_dir, "x.npy")
        #     y_file = os.path.join(group_dir, "y.npy")
        #     x = np.load(x_file)
        #     y = np.load(y_file)

        #     num_batches, num_samples, num_featrues = x.shape
        #     _, group_size, _ = y.shape

        #     print(x.shape, y.shape)

        #     model = Sequential([
        #         LSTM(round(num_featrues/2), input_shape=(num_samples, num_featrues), activation="relu", return_sequences=True),
        #         LSTM(round(num_featrues/5), activation="relu"),
        #         # LSTM(round(num_featrues/2), activation="relu"),
        #         Dense(group_size*2, activation="relu"),
        #         Reshape((group_size, 2)),
        #         Dense(2, activation="softmax")
        #     ])

        #     model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        #     model.summary()

        #     filepath = os.path.join(group_dir, "msbc_{epoch:02d}_{val_accuracy:.2f}.hdf5")
        #     checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        #     callbacks = [checkpoint]

        #     epochs = 100
        #     # model.fit(x, y, epochs=epochs, batch_size=round(num_batches/200), validation_split=0.2, callbacks=callbacks)
        #     model.fit(x, y, epochs=epochs, batch_size=round(math.sqrt(num_batches)), validation_split=0.2)


if __name__ == "__main__":
    msbc = SimpleMovingAverageBinaryClassifier("stocks_only")
    msbc.preprocess()
