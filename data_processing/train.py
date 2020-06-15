import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense
from keras.models import Sequential
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

TRAINING_DATA = "/home/bill/stockbot_2020/processed_data/small_cap_example.npy"
MODEL_DIR = "/home/bill/stockbot_2020/models"

examples = np.load(TRAINING_DATA, allow_pickle=True)
print(examples.shape)
x = examples[:, 0]
y = examples[:, 1]

print(x.shape, y.shape)

model = Sequential([
    LSTM(32, activation="relu", input_shape=(5, 64), return_sequences=True),
    LSTM(16, activation="relu", return_sequences=True),
    LSTM(8, activation="relu"),
    Dense(8, activation="relu"),
    Dense(2, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["categorical_accuracy"])

epochs = 10
model.fit(x, y, epochs=epochs, batch_size=256, validation_split=0.2)
model.save("model")

