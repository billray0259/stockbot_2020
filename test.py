import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

candles = pd.read_hdf("data/stocks_only/filled.h5")
opens = candles.filter(like="_open")
closes = candles.filter(like="_close")

opens.columns = list(map(lambda column: column.split("_")[0], opens.columns))
closes.columns = list(map(lambda column: column.split("_")[0], closes.columns))

avg = (opens + closes) / 2

# df = avg[[random.choice(avg.columns) for _ in range(100)]]
df = avg[["AAPL"]]
count = 0

points = {}
last = {}

win1_start = 1
win1_end = 256
win1_step = 8

win2_start = 2
win2_end = 1024
win2_step = 32


for i in tqdm(range(win2_end, len(df), 100)):
    win1 = win1_start
    while win1 < win1_end:
        if win1 not in points:
            points[win1] = {}
        if win1 not in last:
            last[win1] = {}
        win2 = win2_start
        while win2 < win2_end:
            if win1 < win2:
                past = df[i-win2:i]
                sma1 = past.rolling(win1).mean().iloc[-1].dropna()
                sma2 = past.rolling(win2).mean().iloc[-1].dropna()
                vec = np.where(sma1 > sma2, 1, -1)
                if win2 not in points[win1]:
                    points[win1][win2] = 0
                if win2 not in last[win1]:
                    last[win1][win2] = vec
                else:
                    points[win1][win2] += np.sum(vec != last[win1][win2])
            win2 += max(round(0.2 * win2), win1_step)
        win1 += max(round(0.2 * win1), win2_step)

p = []
for k1 in points:
    for k2 in points[k1]:
        p.append((k1, k2, points[k1][k2]))
    
print(len(p))

points = np.array(p)
x, y, c = points[:, 0], points[:, 1], points[:, 2]

plt.scatter(x, y, c=c, cmap="cool")
plt.show()
