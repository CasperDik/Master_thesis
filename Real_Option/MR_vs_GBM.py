from LSMC.LSMC_faster import GBM
from Real_Option.MR import MR2, MR1, MR3, MRvsGBM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T = 10
dt = 50
paths = 1
N = T * dt

S_0 = 8.00

#GBM
mu_GBM = 0.00
sigma_GBM = 0.15

# MR
theta = 0.1
Sbar = 10
sigma_MR = 0.15

#MR1 = MR1(T, dt, paths, sigma_MR, S_0, theta, Sbar)
#MR2 = MR2(T, dt, paths, sigma_MR, S_0, theta, Sbar)
#GBM = GBM(T, dt, paths, mu_GBM, sigma_GBM, S_0)
# MR3 = MR3(T, dt, paths, sigma_g, sigma_e, S_0, theta_e, theta_g, Sbar, LR_0)

#plt.plot(np.linspace(0, N, N+1), MR1, label="MR1", c="y", alpha=0.2)
#plt.plot(np.linspace(0, N, N+1), MR2, label="MR2", c="b", alpha=0.2)
#plt.plot(np.linspace(0, N+1, N+1), GBM, label="GBM", c="r", alpha=0.1)
# plt.plot(np.linspace(0, N+1, N+1), MR3, label="MR3", c="y")

sigma = 0.20
mu = 0.00
S_0 = 10
Sbar = 10
theta = 0.2

GBM, MR = MRvsGBM(T, dt, paths, sigma, S_0, theta, Sbar, mu)
df = pd.DataFrame(columns=["GBM", "MR"])

df["GBM"] = GBM.tolist()
df["MR"] = MR.tolist()

plt.plot(np.linspace(0, N+1, N+2), GBM, label="GBM")
plt.plot(np.linspace(0, N+1, N+2), MR, label="MR")
plt.title("MR vs GBM")

plt.legend()
plt.show()
df.to_excel("GBMvsMR.xlsx")