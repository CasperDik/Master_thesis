import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Real_Option.RO_LSMC import LSMC_RO, GBM
from Real_Option.MR import MR2
from Real_Option.threshold_value import NPV1

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
    S_0 = np.linspace(4, 12, 9)

    # GBM
    mu = 0.05743
    sigma_gbm = 0.32048

    # MR
    Sbar = 9.801
    theta = 0.044
    sigma_mr = 0.15289

    # life of the option(in years)
    T = 5
    # time periods per year
    dt = 100
    # number of paths per simulations
    paths = 25000

    GBM_v = []
    MR_v = []
    GBM_tp = []
    MR_tp = []
    GBM_pinv = []
    MR_pinv = []
    NPV = []

    for s in S_0:
        print("ran with initial gas price ", s, "\n")

        # GBM
        val = []
        tp = []
        pinv = []
        for _ in range(5):
            price_matrix_gbm = GBM(T, dt, paths, mu, sigma_gbm, s)
            value_gbm, tp_gbm, prob = LSMC_RO(price_matrix_gbm, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I, s, 1)
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
            price_matrix_mr = MR2(T, dt, paths, sigma_mr, s, theta, Sbar)
            value_mr, tp_mr, prob = LSMC_RO(price_matrix_mr, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I, s, 1)
            val.append(value_mr)
            tp.append(tp_mr)
            pinv.append(prob)

        val = np.mean(val)
        tp = np.mean(tp)
        pinv = np.mean(prob)

        MR_v.append(value_mr)
        MR_tp.append(tp_mr)
        MR_pinv.append(pinv)

        NPV.append(NPV1(s, A, Q, epsilon, O_M, wacc, Tc, I, T_plant))

df = pd.DataFrame(columns=["S_0", "value GBM", "TP GBM", "Prob GBM", "value MR", "TP MR", "Prob MR", "NPV"])
inputs = pd.DataFrame({"_":["A","Q","Epsilon","O&M", "I", "Tc", "wacc", "Tplant", "S0", "mu", "sigmaGBM", "Sbar", "theta", "sigmaMR", "dt", "pahts", "T"],
                 "Inputs": [A, Q, epsilon, O_M, I, Tc, wacc, T_plant, S_0, mu, sigma_gbm, Sbar, theta, sigma_mr, dt, paths, T]})

df["S_0"] = S_0
df["value GBM"] = GBM_v
df["TP GBM"] = GBM_tp
df["Prob GBM"] = GBM_pinv
df["value MR"] = MR_v
df["TP MR"] = MR_tp
df["Prob MR"] = MR_pinv
df["NPV"] = NPV

writer = pd.ExcelWriter("raw_data/rangeS0.xlsx", engine="xlsxwriter")
inputs.to_excel(writer, sheet_name="inputs")
df.to_excel(writer, sheet_name="results")
writer.save()

plt.plot(S_0, NPV, label="NPV")
plt.plot(S_0, GBM_v, label="GBM")
plt.plot(S_0, MR_v, label="MR")
#todo: add threshold line
plt.show()
