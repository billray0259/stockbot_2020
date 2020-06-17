import pandas as pd
import os
from tqdm import tqdm

file_name = "/home/bill/stockbot_2020/data/all_joined.h5"

df = pd.read_hdf(file_name)
print(df)
