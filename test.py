import pandas as pd
import numpy as np

with open("finnhub_key.txt", "r") as key_file:
    config = finnhub.Configuration(
        api_key={
            "token": key_file.readline()
        }
    )

client = finnhub.DefaultApi(finnhub.ApiClient(config))

now = round(time.time())
year = 60*60*24*365

from_ = now - 5*year
to = now

def process(ticker):
    candles = client.stock_candles(ticker, 'D', from_, to, _preload_content=False).data
    try:
        df = pd.read_json(candles)
    except:
        print(candles)
        exit()
    df.set_index("t", inplace=True)

    df["log_returns"] = np.log(df["c"] / df["c"].shift(1))
    df["returns"] = np.exp(df["log_returns"].cumsum())

    mu = df["log_returns"].mean() * 7
    sig = df["log_returns"].std() * 7**0.5
    risk_free_interest_rate = 0

    kelly = (mu - risk_free_interest_rate) / sig**2

    return df, kelly


df = pd.DataFrame()
for ticker in ["ARKK", "ARKQ", "ARKW", "ARKG", "ARKF"]:
    spy, spy_kelly = process(ticker)
    print("%s Kelly" % ticker, spy_kelly)

    df[ticker] = spy["returns"]

df.index = pd.to_datetime(df.index, unit="s")

df = pd.read_hdf("options_data\\1595832579098.h5")

print(df)
