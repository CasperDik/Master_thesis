import matplotlib.pyplot as plt
import numpy as np
from Real_Option.RO_LSMC import GBM
from Real_Option.MR import MR2

# todo: range with different T --> threshold price differs per T

# GBM
mu = 0.05743
sigma_gbm = 0.32048

# MR
Sbar = 15.261
theta = 0.254
sigma_mr = 0.22777

T = 10
dt = 50
paths = 25000

S_0 = np.linspace(1, 15, 30)
Pinvesting_gbm = []
Pinvesting_mr = []

# todo: these differ per T(and dt)
thresholdvalue_mr = 5
thresholdvalue_gbm = 6

for s in S_0:
    price_matrix_gbm = GBM(T, dt, paths, mu, sigma_gbm, s)
    price_matrix_mr = MR2(T, dt, paths, sigma_mr, s, theta, Sbar)
    # minimal values of each path
    GBM_min = price_matrix_gbm[1:].min(axis=0)
    MR_min = price_matrix_mr[1:].min(axis=0)
    # sum if min value above threshold
    Pinvesting_mr.append(sum(MR_min < thresholdvalue_gbm)/(paths * 2))
    Pinvesting_gbm.append(sum(GBM_min < thresholdvalue_mr)/(paths * 2))

plt.plot(S_0, Pinvesting_gbm, label="GBM")
plt.plot(S_0, Pinvesting_mr, label="MR")
plt.legend()
plt.show()

# todo: store to excel

