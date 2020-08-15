from scipy import stats
from scipy import optimize
import numpy as np

n = 100
x = np.linspace(-4, 4, n)


def f(x, mu, sigma): return stats.norm(mu, sigma).cdf(x)

stats.logistic.cdf(3)


data = f(x, 0.2, 1) + 0.05*np.random.randn(n)

mu, sigma = optimize.curve_fit(f, x, data)[0]


print(mu, sigma)