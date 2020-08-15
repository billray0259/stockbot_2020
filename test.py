import numpy as np
from scipy.integrate import quad

s = 1.175400343900767
u = -0.4293767406319755
f = lambda x: np.exp(-(x-u)/s)/(s*(1+np.exp(-(x-u)/s))**2)

print(quad(f, u-2*s, u+2*s)[0])