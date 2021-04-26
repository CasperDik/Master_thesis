import numpy as np
import pandas as pd
from Real_Option.RO_LSMC import LSMC_RO, GBM
from Real_Option.MR import MR2

#todo: run all settings 5 times and take average?
if __name__ == "__main__":
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
    S_0 = 8.00

    # GBM
    mu = 0.05743
    sigma_gbm = 0.32048

    # MR
    Sbar = 9.801
    theta = 0.044
    sigma_mr = 0.15289

    # life of the option(in years)
    T = np.linspace(1,30,20)
    # time periods per year
    dt = 10
    # number of paths per simulations
    paths = 100000

    GBM_v = []
    MR_v = []
    GBM_tp = []
    MR_tp = []

    for t in T:
        # GBM
        price_matrix_gbm = GBM(t, dt, paths, mu, sigma_gbm, S_0)
        value_gbm, tp_gbm = LSMC_RO(price_matrix_gbm, wacc, paths, t, T_plant, dt, A, Q, epsilon, O_M, Tc, I, S_0, 1)
        GBM_v.append(value_gbm)
        GBM_tp.append(tp_gbm)

        # MR
        price_matrix_mr = MR2(t, dt, paths, sigma_mr, S_0, theta, Sbar)
        value_mr, tp_mr = LSMC_RO(price_matrix_mr, wacc, paths, t, T_plant, dt, A, Q, epsilon, O_M, Tc, I, S_0, 1)
        MR_v.append(value_mr)
        MR_tp.append(tp_mr)

df = pd.DataFrame(columns=["value GBM", "TP GBM", "value MR", "TP MR"])

df["value GBM"] = GBM_v
df["TP GBM"] = GBM_tp
df["value MR"] = MR_v
df["TP MR"] = MR_tp
df.to_excel("time to maturity data")