from scipy import stats
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from td_api import Account
from datetime import datetime, timedelta

def option_price(r, u, s): 
    dist = lambda x: stats.logistic.pdf(x, u, s)
    return r - quad(lambda x: dist(x) * x, -10*r, r)[0] - r * quad(dist, r, 10*r)[0]

x = np.linspace(1000, 2500)
y = [option_price(r, 1854, 100) for r in x]

acc = Account("keys.json")

symbol = "TSLA"
days = 8
strike_count = 50

from_date = datetime.now()
to_date = from_date + timedelta(days=days)
data = acc.get_options_chain(symbol, from_date, to_date, strike_count=strike_count)
mark = acc.get_quotes([symbol])["mark"].iloc[0]

plt.plot(x, y)
plt.scatter(data["strikePrice"], data["mark"])
plt.show()
