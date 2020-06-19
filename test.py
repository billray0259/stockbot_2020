import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nums = np.random.normal(size=10000)

data = {
    "A": nums[:-1],
    "B": nums[1:]
}

df = pd.DataFrame(data)

window = 26
rolled = df.rolling(window).mean()


print(rolled.corr())


plt.scatter(rolled["A"], rolled["B"])
plt.show()