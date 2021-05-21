import numpy as np
import pandas as pd

from Real_Option.RO_LSMC import LSMC_RO, GBM
from Real_Option.MR import MR2
from Real_Option.threshold_value import NPV1, thresholdvalue, NPV_TP

T = 10
dt = 50
paths = 25000

# inputs:
# real option setting
A = 30.00
Q = 4993200
epsilon = 1 / 0.6
O_M = 25 * 600 * 1000
I = 850 * 1000 * 600
Tc = 0.00
wacc = 0.056
T_plant = 30

# initial gas price
S_0 = np.linspace(0.5, 16, 40)

# GBM
mu = 0.05743
sigma_gbm = 0.32048

# MR
# Sbar = 15.305
Sbar1 = [15.305/1.4, 15.305, 15.305*1.4]
theta1 = [0, 0.2, 0.4]
# theta = 0.006
sigma_mr = 0.17152

thresholds = pd.DataFrame(columns=["theta", "Cbar", "threshold MR"])

for theta in theta1:
    for Sbar in Sbar1:
        MR_v = []
        NPV = []
        for s in S_0:
            # MR
            price_matrix_mr = MR2(T, dt, paths, sigma_mr, s, theta, Sbar)
            MR_v.append(
                LSMC_RO(price_matrix_mr, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, I))
            NPV.append(NPV1(s, A, Q, epsilon, O_M, wacc, I, T_plant))
        threshold_MR = thresholdvalue(MR_v, NPV, S_0)

        print("thresholdprice MR: ", threshold_MR)
        row = [theta, Sbar, threshold_MR]
        thresholds.loc[len(thresholds)] = row

thresholds.loc[len(thresholds)] = ["Threshold NPV", NPV_TP(A, Q, epsilon, O_M, wacc, I, T_plant), "_"]

GBM_v = []
NPV = []
for s in S_0:
    price_matrix_gbm = GBM(T, dt, paths, mu, sigma_gbm, s)
    GBM_v.append(LSMC_RO(price_matrix_gbm, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, I))
    NPV.append(NPV1(s, A, Q, epsilon, O_M, wacc, I, T_plant))

threshold_GBM = thresholdvalue(GBM_v, NPV, S_0)
thresholds.loc[len(thresholds)] = ["Threshold GBM", threshold_GBM, "_"]

inputs = pd.DataFrame({"_": ["A", "Q", "Epsilon", "O&M", "I", "Tc", "wacc", "Tplant", "S0", "mu", "sigmaGBM",
                            "sigmaMR", "dt", "paths", "T"],
                       "Inputs": [A, Q, epsilon, O_M, I, Tc, wacc, T_plant, S_0, mu, sigma_gbm,
                                  sigma_mr, dt, paths, T]})

writer = pd.ExcelWriter("raw_data/Cbar_theta_table.xlsx", engine="xlsxwriter")
inputs.to_excel(writer, sheet_name="inputs")
thresholds.to_excel(writer, sheet_name="results")
writer.save()
