import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
import math

from data_handler import DataHandler

import matplotlib.pyplot as plt

class SimpleMovingAverageBinaryClassifier:

    def __init__(self, name, sub_dir=None):
        self.data_handler = DataHandler(name)
        self.dir = os.path.join(self.data_handler.data_dir, "smabc_training")
        if sub_dir is not None:
            self.dir = os.path.join(self.dir, sub_dir)
        self.model_name = "model.hdf5"
        self.model_file = os.path.join(self.dir, self.model_name)
        print(self.model_file)

        self.model = None
        if os.path.exists(self.model_file):
            import keras
            self.model = keras.models.load_model(self.model_file)

    def preprocess(self, candles=None, pred_range=208, density=0.8, win1_start=1, win1_end=256, win1_step=8, win2_start=2, win2_end=1024, win2_step=32):
        live = candles is not None
        candles = pd.read_hdf(self.data_handler.filled_file) if not live else candles
        opens = candles.filter(like="_open")
        closes = candles.filter(like="_close")

        opens.columns = list(map(lambda column: column.split("_")[0], opens.columns))
        closes.columns = list(map(lambda column: column.split("_")[0], closes.columns))
        returns = np.log(opens/closes)
        returns.fillna(value=0, inplace=True)  # pylint: disable=no-member

        vectors = []
        labels = []
        if live:
            iterator = range(len(closes)-1, len(closes))
        else:
            iterator = tqdm(range(win2_end, len(closes)-pred_range, 10))
        for i in iterator:
            features = []

            if not live:
                labels.append(returns[i:i+pred_range].sum() > 0)

            win1 = win1_start
            while win1 < win1_end:
                win2 = win2_start
                while win2 < win2_end:
                    if win1 < win2:
                        sma1 = closes[i-win1:i].mean()
                        sma2 = closes[i-win2:i].mean()
                        vec = np.where(sma1 > sma2, 1, -1)
                        features.append(vec)

                    win2 += max(round((1-density) * win2), win1_step)
                win1 += max(round((1-density) * win1), win2_step)

            features = np.array(features)
            vectors.append(np.transpose(features))

        vectors = np.array(vectors)
        x = np.concatenate(vectors)
        if live:
            return x

        for i, label in enumerate(labels):
            labels[i] = list(map(lambda x: [0, 1] if x else [1, 0], label))

        labels = np.array(labels)
        y = np.concatenate(labels)

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        x_file = os.path.join(self.dir, "x.npy")
        y_file = os.path.join(self.dir, "y.npy")
        np.save(x_file, x)
        np.save(y_file, y)

    def train(self, epochs=25):
        from keras.callbacks import ModelCheckpoint
        from keras.layers import Dense, Dropout
        from keras.models import Sequential
        from keras.activations import softmax
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        x_file = os.path.join(self.dir, "x.npy")
        y_file = os.path.join(self.dir, "y.npy")
        x = np.load(x_file)
        y = np.load(y_file)

        num_batches, num_features = x.shape

        model = Sequential([
            Dense(8, activation="relu", input_shape=[num_features]),
            Dropout(0.5),
            Dense(4, activation="relu"),
            Dropout(0.5),
            Dense(2, activation="softmax")
        ])

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()

        filepath = os.path.join(self.dir, "model_{val_accuracy:.4f}.hdf5")
        # filepath = os.path.join(self.dir, "model.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks = [checkpoint]

        model.fit(x, y, epochs=epochs, batch_size=round(math.sqrt(num_batches)), validation_split=0.2, callbacks=callbacks)
    
    def wanted_tickers(self):
        return ["TSLA"]

    def get_holdings(self, data, current_holdings):
        x = self.preprocess(data)
        model = self.model
        pred = model.predict(x)[0]
        return {"TSLA": np.argmax(pred)}


if __name__ == "__main__":
    for i in tqdm(range(1, 27, 1)):
        smabc = SimpleMovingAverageBinaryClassifier("stocks_only", sub_dir=str(i))
        smabc.preprocess(pred_range=i)
        smabc.train()
