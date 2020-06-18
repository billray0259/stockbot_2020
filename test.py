import pandas as pd
import os

empty = []
with open("empty.txt", "r") as empty_file:
    for line in empty_file.readlines():
        empty.append(line.strip())

joined_files = ["all_joined.h5", "all_filled.h5"]

for file in joined_files:
    df = pd.read_hdf("data/" + file)
    for ticker in empty:
        empty_df = df.filter(regex="^%s_" % ticker)
        print(empty_df.columns)
        df = df.drop(columns=empty_df.columns)
    df.to_hdf("data/" + file, "df", "w")

dirnames = ["all_filled_histories/", "all_histories/"]
for dir_ in dirnames:
    for ticker in empty:
        file = "data/" + dir_ + ticker + ".h5"
        print(file)
        os.remove(file)

finviz = pd.read_hdf("data/all_finviz.h5")
finviz.drop(index=empty, inplace=True)
finviz.to_hdf("data/all_finviz.h5", "df", "w")





# filled_dir = "/home/bill/stockbot_2020/data/all_filled_histories/"

# for file in os.listdir(filled_dir):
#     df = pd.read_hdf(os.path.join(filled_dir, file))

