import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from real_option.RO_LSMC import LSMC_RO, GBM


# electricity price
A = 40
# Quantity per year
Q = 40000
# efficiency rate of the plant
epsilon = 0.85
# maintenance and operating cost per year
O_M = 40000
# initial investment
I = 1400000
# tax rate
Tc = 0.25
# discount rate (WACC?)
r = 0.06

# initial gas price
S_0 = 20
# drift rate mu of gas price
mu = 0.02
# volatility of the gas price

# life of the power plant(in years)
T_plant = 30
# life of the option(in years)
T = 2
# time periods per year
dt = 180

# number of paths per simulations
paths = 1000

sigma = np.linspace(0,1,10)
thresholdval = []

for s in sigma:
    price_matrix = GBM(T, dt, paths, mu, s, S_0)
    value, thresholdvalue = LSMC_RO(price_matrix, r, mu, paths, T, dt, A, Q, epsilon, O_M, Tc, I)
    thresholdval.append(thresholdvalue)

plt.plot(sigma, thresholdval)
plt.show()

