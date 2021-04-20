from LSMC.LSMC_faster import GBM
from real_option.MR import MR2, MR1, MR3
import matplotlib.pyplot as plt
import numpy as np

T = 1
dt = 150
paths = 3
N = T * dt

S_0 = 4.43

#GBM
mu_GBM = 0.058
sigma_GBM = 0.32073

# MR
theta = 0.07
Sbar = 5
sigma_MR = 0.3

MR1 = MR1(T, dt, paths, sigma_MR, S_0, theta, Sbar)
# MR2 = MR2(T, dt, paths, sigma, S_0, theta, Sbar)
# MR3 = MR3(T, dt, paths, sigma_g, sigma_e, S_0, theta_e, theta_g, Sbar, LR_0)
GBM = GBM(T, dt, paths, mu_GBM, sigma_GBM, S_0)

plt.plot(np.linspace(0, N, N+1), MR1, label="MR1", c="r")
# plt.plot(np.linspace(0, N, N+1), MR2, label="MR2", c="b")
plt.plot(np.linspace(0, N+1, N+1), GBM, label="GBM", c="k")
# plt.plot(np.linspace(0, N+1, N+1), MR3, label="MR3", c="y")

plt.title("MR vs GBM")
# plt.legend()
plt.show()
