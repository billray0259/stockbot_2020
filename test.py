import pandas as pd
import os

filled_dir = "/home/bill/stockbot_2020/data/all_filled_histories/"

for file in os.listdir(filled_dir):
    df = pd.read_hdf(os.path.join(filled_dir, file))
    
