import numpy as np
import pandas as pd
from Real_Option.RO_LSMC import LSMC_RO, GBM
from Real_Option.MR import MR2

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
    T = [0.2, 0.5, 0.7, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 9, 11, 14, 18, 22, 25, 28, 30]
    # time periods per year
    dt = 25
    # number of paths per simulations
    paths = 25000

    GBM_v = []
    MR_v = []
    GBM_tp = []
    MR_tp = []
    GBM_pinv = []
    MR_pinv = []

    for t in T:
        print("ran with maturity ", t, "\n")

        # GBM
        val = []
        tp = []
        pinv = []
        for _ in range(5):
            price_matrix_gbm = GBM(t, dt, paths, mu, sigma_gbm, S_0)
            value_gbm, tp_gbm, prob = LSMC_RO(price_matrix_gbm, wacc, paths, t, T_plant, dt, A, Q, epsilon, O_M, Tc, I, S_0, 1)
            val.append(value_gbm)
            tp.append(tp_gbm)
            pinv.append(prob)
        val = np.mean(val)
        tp = np.mean(tp)
        pinv = np.mean(prob)

        GBM_v.append(val)
        GBM_tp.append(tp)
        GBM_pinv.append(pinv)

        # MR
        val = []
        tp = []
        pinv = []
        for _ in range(5):
            price_matrix_mr = MR2(t, dt, paths, sigma_mr, S_0, theta, Sbar)
            value_mr, tp_mr, prob = LSMC_RO(price_matrix_mr, wacc, paths, t, T_plant, dt, A, Q, epsilon, O_M, Tc, I, S_0, 1)
            val.append(value_mr)
            tp.append(tp_mr)
            pinv.append(prob)

        val = np.mean(val)
        tp = np.mean(tp)
        pinv = np.mean(prob)

        MR_v.append(value_mr)
        MR_tp.append(tp_mr)
        MR_pinv.append(pinv)

df = pd.DataFrame(columns=["Time", "value GBM", "TP GBM", "Prob GBM", "value MR", "TP MR", "Prob MR"])
inputs = pd.DataFrame({"_":["A","Q","Epsilon","O&M", "I", "Tc", "wacc", "Tplant", "S0", "mu", "sigmaGBM", "Sbar", "theta", "sigmaMR", "dt", "pahts", "T"],
                 "Inputs": [A, Q, epsilon, O_M, I, Tc, wacc, T_plant, S_0, mu, sigma_gbm, Sbar, theta, sigma_mr, dt, paths, T]})

df["Time"] = T
df["value GBM"] = GBM_v
df["TP GBM"] = GBM_tp
df["Prob GBM"] = GBM_pinv
df["value MR"] = MR_v
df["TP MR"] = MR_tp
df["Prob MR"] = MR_pinv


writer = pd.ExcelWriter("raw_data/time_to_maturity.xlsx", engine="xlsxwriter")
inputs.to_excel(writer, sheet_name="inputs")
df.to_excel(writer, sheet_name="results")
writer.save()

