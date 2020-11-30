from scipy.integrate import quad
from scipy.stats import norm
import numpy as np

R = 100
u = 100
s = 10

pdf = norm(u, s).pdf

ans1 = quad(lambda x: pdf(x) * (x-R), R, np.inf)[0]

print(ans1)

ans2 = R - quad(lambda x: pdf(x) * x, -np.inf, R)[0] - R * quad(lambda x: pdf(x), R, np.inf)[0]

print(ans2)