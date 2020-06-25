import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

xs = []
ys = []
path = "data/stocks_only/smabc_training"
for item in os.listdir(path):
    if os.path.isdir(os.path.join(path, item)):
        xs.append(int(item))
        accs = []
        for file in os.listdir(os.path.join(path, item)):
            name, ext = os.path.splitext(file)
            if ext == ".hdf5":
                accs.append(float(name.split("_")[1]))
        ys.append(max(accs))

points = [(xs[i], ys[i]) for i in range(len(xs))]

points.sort(key=lambda p: p[0])

points = np.array(points)
plt.plot(points[:, 0], points[:, 1])
plt.show()