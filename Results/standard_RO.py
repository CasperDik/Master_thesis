import numpy as np
import pandas as pd

from Real_Option.RO_LSMC import LSMC_RO, GBM
from Real_Option.MR import MR2
from Real_Option.threshold_value import NPV1, thresholdvalue, NPV_TP

def standard_RO(paths, dt, T, s):
    # todo: check inputs
    # inputs:
    # real option setting
    A = 30.00
    Q = 4993200
    epsilon = 1/0.5
    O_M = 25*600*1000
    I = 850*1000*600
    Tc = 0.21
    wacc = 0.056
    T_plant = 30

    # initial gas price
    S_0 = np.linspace(4, 12, 40)

    # GBM
    mu = 0.05743
    sigma_gbm = 0.32048 * s

    # MR
    Sbar = 9.801
    theta = 0.044
    sigma_mr = 0.15289 * s

    GBM_v = []
    MR_v = []
    NPV = []

    for s in S_0:
        # GBM
        price_matrix_gbm = GBM(T, dt, paths, mu, sigma_gbm, s)
        # GBM_v.append(LSMC_RO(price_matrix_gbm, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I))
        GBM_v.append(LSMC_RO(price_matrix_gbm, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I, mu, theta, Sbar, "GBM"))

        # MR
        price_matrix_mr = MR2(T, dt, paths, sigma_mr, s, theta, Sbar)
        #MR_v.append(LSMC_RO(price_matrix_mr, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I))
        MR_v.append(
            LSMC_RO(price_matrix_mr, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I, mu, theta, Sbar, "MR1"))

        NPV.append(NPV1(s, A, Q, epsilon, O_M, wacc, Tc, I, T_plant))
    threshold_GBM = thresholdvalue(GBM_v, NPV, S_0)
    threshold_MR = thresholdvalue(MR_v, NPV, S_0)
    threshold_NPV = NPV_TP(A, Q, epsilon, O_M, wacc, I, T_plant)

    print("thresholdprice GBM: ", threshold_GBM, "thresholdprice MR: ", threshold_MR, "thresholdpirce NPV: ", threshold_NPV)

    inputs = pd.DataFrame({"_": ["A", "Q", "Epsilon", "O&M", "I", "Tc", "wacc", "Tplant", "S0", "mu", "sigmaGBM",
                                 "Sbar", "theta", "sigmaMR", "dt", "paths", "T"],
                           "Inputs": [A, Q, epsilon, O_M, I, Tc, wacc, T_plant, S_0, mu, sigma_gbm, Sbar, theta,
                                      sigma_mr, dt, paths, T]})

    return threshold_GBM, threshold_MR, threshold_NPV, inputs

