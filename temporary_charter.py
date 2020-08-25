import os
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

symbols = {}
for csv in os.listdir('options_data'):
    if 'csv' in csv and 'tickers' not in csv:
        data = pd.read_csv('options_data/' + csv)
        if not data.empty:
            for index, item in enumerate(data['symbol'][:100]):
                print(index, item)
                #if (data['askPrice'][index]/data['underlyingPrice'][index]) < 1.03 and (data['askPrice'][index]/data['underlyingPrice'][index]) > 0.97:
                if not item in symbols:
                    symbols[item] = []
                symbols[item].append(data['askPrice'][index])
                #print(item)
print(len(symbols))
subs = []
abovs = []
for item in symbols:
    #log(final/initial)
    # x = [math.log(y/symbols[item][index-1]) for index, y in enumerate(symbols[item])]
    data = pd.Series(symbols[item])
    x = np.log(data/data.shift(1)).cumsum().apply(np.exp)
    for val in x:
        if val <= 0.9:
            subs.append(item)
            break 
        elif val >= 1.1:
            abovs.append(item)
            break
    #print(type(x))
    #if (val[0]/val[1]) < 1.03 and (val[0]/val[1]) > 0.97:
    plt.plot(x, label=item)
    #print(x)
print(len(subs))
print(len(abovs))
plt.legend()
plt.show()