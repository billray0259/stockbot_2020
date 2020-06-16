import pandas as pd
import os
from tqdm import tqdm

HISTORIES_DIR = "/home/bill/stockbot_2020/data/histories/"
DATA_SAVE_DIR = "/home/bill/stockbot_2020/data/"


dataframes = []
for data_file_name in tqdm(os.listdir(HISTORIES_DIR), desc="Joining candles"):
    ticker = data_file_name[:-len(".csv")]
    data_df = pd.read_csv(HISTORIES_DIR + data_file_name, index_col="datetime")
    data_df.columns = list(
        map(lambda column: ticker + "_" + column, data_df.columns))
    dataframes.append(data_df)

big_df = dataframes[0].join(dataframes[1:]).dropna(axis=1)

big_df.to_csv(DATA_SAVE_DIR + "small_cap_candles.csv")
