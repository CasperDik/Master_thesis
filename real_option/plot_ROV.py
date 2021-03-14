from real_option.ROV import RO_compare_stochatic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# inputs
T = 1
dt = 150
paths = 3
N = T * dt

theta = 1.1
sigma = 0.2
Sbar = 100  # long run equilibrium price
S_0 = 100
mu = 0.2

LR_0 = 100  # initial equilibrium price
sigma_g = 0.2
sigma_e = 0.05
theta_e = 5
theta_g = 0.2

# plotting
x = [[], [], []]
range = np.linspace(30,70,5)
for S_0 in range:
    RO = RO_compare_stochatic(T, dt, paths, mu, sigma, sigma_e, theta, theta_e, Sbar, LR_0, S_0)
    x = np.hstack((x, RO))

x = np.transpose(np.array(x))
string_range = [str(int) for int in range]
df = pd.DataFrame(data=x, index=string_range, columns=["GBM", "MR2", "MR3"])
# df.to_excel("output.xlsx")

plt.plot(range, df.MR3, label="MR1")
plt.plot(range, df.MR2, label="MR2")
plt.plot(range, df.GBM, label="GBM")

plt.legend()
plt.show()
