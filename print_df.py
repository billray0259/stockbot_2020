import pandas as pd
import sys
import os

path = "/home/bill/stockbot_2020/data/all_filled.h5"

if len(sys.argv) > 1:
    path = sys.argv[1]

if not os.path.exists(path):
    path = os.path.join("data", path)

df = pd.read_hdf(path)
dfna = df[df.columns[df.isna().sum() > 0]]

if __name__ == "__main__":
    print(df)

    if "info" in sys.argv:
        print(df.info())

    if "describe" in sys.argv:
        print(df.describe())
