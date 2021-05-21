import numpy as np
import pandas as pd

from Real_Option.RO_LSMC import LSMC_RO, GBM
from Real_Option.MR import MR2
from Real_Option.threshold_value import NPV1, thresholdvalue, NPV_TP

T = 10
dt = 50
paths = 25000
s1 = [0.8, 1, 1.2]
I1 = [650 * 1000 * 600, 850 * 1000 * 600, 1050 * 1000 * 600]

# inputs:
# real option setting
A = 30.00
Q = 4993200
epsilon = 1 / 0.6
O_M = 25 * 600 * 1000
# I = 850 * 1000 * 600
Tc = 0.00
wacc = 0.056
T_plant = 30

# initial gas price
S_0 = np.linspace(0.5, 16, 40)

# GBM
mu = 0.05743

# MR
Sbar = 15.305
theta = 0.006

thresholds = pd.DataFrame(columns=["sigma gbm", "sigma mr", "I", "threshold MR", "threshold GBM", "threshold NPV"])

for I in I1:
    for x in s1:
        sigma_gbm = 0.32048 * x
        sigma_mr = 0.17152 * x
        GBM_v = []
        MR_v = []
        NPV = []
        for s in S_0:
            # GBM
            price_matrix_gbm = GBM(T, dt, paths, mu, sigma_gbm, s)
            GBM_v.append(LSMC_RO(price_matrix_gbm, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, I))

            # MR
            price_matrix_mr = MR2(T, dt, paths, sigma_mr, s, theta, Sbar)
            MR_v.append(
                LSMC_RO(price_matrix_mr, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, I))

            NPV.append(NPV1(s, A, Q, epsilon, O_M, wacc, I, T_plant))
        threshold_GBM = thresholdvalue(GBM_v, NPV, S_0)
        threshold_MR = thresholdvalue(MR_v, NPV, S_0)
        threshold_NPV = NPV_TP(A, Q, epsilon, O_M, wacc, I, T_plant)

        print("thresholdprice GBM: ", threshold_GBM, "thresholdprice MR: ", threshold_MR, "thresholdpirce NPV: ", threshold_NPV)
        row = [sigma_gbm, sigma_mr, I, threshold_MR, threshold_GBM, threshold_NPV]
        thresholds.loc[len(thresholds)] = row

inputs = pd.DataFrame({"_": ["A", "Q", "Epsilon", "O&M", "wacc", "Tplant", "S0", "mu",
                                     "Sbar", "theta", "dt", "paths", "T"],
                               "Inputs": [A, Q, epsilon, O_M, wacc, T_plant, S_0, mu, Sbar, theta,
                                          dt, paths, T]})

writer = pd.ExcelWriter("raw_data/sigma_I_table.xlsx", engine="xlsxwriter")
inputs.to_excel(writer, sheet_name="inputs")
thresholds.to_excel(writer, sheet_name="results")
writer.save()
