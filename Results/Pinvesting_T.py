import matplotlib.pyplot as plt
import numpy as np
from Real_Option.RO_LSMC import GBM
from Real_Option.MR import MR2
import pandas as pd

# GBM
mu = 0.05743
sigma_gbm = 0.32048

# MR
Sbar = 15.261
theta = 0.254
sigma_mr = 0.22777

# todo: check if correct sheet is at right place with right name
inputs = pd.read_excel("raw_data/time_to_maturity.xlsx", sheet_name="results")

T = inputs["Time to maturity"].to_numpy()
dt = 50
paths = 25000

S_0 = 2.90 / 0.29329722222222
Pinvesting_gbm = []
Pinvesting_mr = []

thresholdvalue_mr = inputs["threshold price GBM"].to_numpy()
thresholdvalue_gbm = inputs["threshold price MR"].to_numpy()

i = 0
for t in T:
    price_matrix_gbm = GBM(t, dt, paths, mu, sigma_gbm, S_0)
    price_matrix_mr = MR2(t, dt, paths, sigma_mr, S_0, theta, Sbar)
    # minimal values of each path
    GBM_min = price_matrix_gbm[1:].min(axis=0)
    MR_min = price_matrix_mr[1:].min(axis=0)
    # sum if min value above threshold

    Pinvesting_mr.append(sum(MR_min < thresholdvalue_gbm[i])/(paths * 2))
    Pinvesting_gbm.append(sum(GBM_min < thresholdvalue_mr[i])/(paths * 2))
    i += 1

plt.plot(T, Pinvesting_gbm, label="GBM")
plt.plot(T, Pinvesting_mr, label="MR")
plt.legend()
plt.show()

df = pd.DataFrame(columns=["T", "Pinvesting GBM", "Pinvesting MR"])
df["T"] = T
df["Pinvesting GBM"] = Pinvesting_gbm
df["Pinvesting MR"] = Pinvesting_mr
df.to_excel("raw_data/Pinvesting_T.xlsx")
