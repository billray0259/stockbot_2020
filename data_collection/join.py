import pandas as pd
import os

data_dir = "histories"

dataframes = []
for data_file_name in os.listdir(data_dir):
    ticker = data_file_name[:-len(".csv")]
    data_df = pd.read_csv(data_dir + "/" + data_file_name, index_col="datetime")
    data_df.columns = list(map(lambda column: ticker + "_" + column, data_df.columns))
    dataframes.append(data_df)

big_df = dataframes[0].join(dataframes[1:]).dropna(axis=1)

big_df.to_csv("small_cap_candles.csv")

