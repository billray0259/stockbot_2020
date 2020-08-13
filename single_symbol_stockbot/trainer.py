import finnhub
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ticker = "AMD"

with open("finnhub_key.txt", "r") as key_file:
    config = finnhub.Configuration(
        api_key={
            "token": key_file.readline()
        }
    )

client = finnhub.DefaultApi(finnhub.ApiClient(config))

now = round(time.time())
year = 60*60*24*365

from_ = now - 20*year
to = now

candles = client.stock_candles(ticker, "D", from_, to, _preload_content=False).data
try:
    df = pd.read_json(candles)
except:
    print(candles)
    exit()

df.set_index("t", inplace=True)
df.index = pd.to_datetime(df.index, unit="s")

df["log_returns"] = np.log(df["c"] / df["c"].shift(1))
print(len(df))
df["c"].plot()
plt.show()




