import numpy as np
import pandas as pd
from Real_Option.RO_LSMC import LSMC_RO, GBM
from Real_Option.MR import MR2
from Real_Option.threshold_value import NPV1, thresholdvalue

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

    for t in T:
        print("ran with maturity ", t, "\n")

        # GBM
        price_matrix_gbm = GBM(t, dt, paths, mu, sigma_gbm, S_0)
        GBM_v.append(LSMC_RO(price_matrix_gbm, wacc, paths, t, T_plant, dt, A, Q, epsilon, O_M, Tc, I))

        # MR
        price_matrix_mr = MR2(t, dt, paths, sigma_mr, S_0, theta, Sbar)
        MR_v.append(LSMC_RO(price_matrix_mr, wacc, paths, t, T_plant, dt, A, Q, epsilon, O_M, Tc, I))
    NPV = NPV1(S_0, A, Q, epsilon, O_M, wacc, Tc, I, T_plant)
    threshold_GBM = thresholdvalue(GBM_v, NPV)
    threshold_MR = thresholdvalue(MR_v, NPV)

df = pd.DataFrame(columns=["Time", "value GBM",  "value MR"])
inputs = pd.DataFrame({"_":["A","Q","Epsilon","O&M", "I", "Tc", "wacc", "Tplant", "S0", "mu", "sigmaGBM", "Sbar", "theta", "sigmaMR", "dt", "pahts", "T"],
                 "Inputs": [A, Q, epsilon, O_M, I, Tc, wacc, T_plant, S_0, mu, sigma_gbm, Sbar, theta, sigma_mr, dt, paths, T]})

df["Time"] = T
df["value GBM"] = GBM_v
df["value MR"] = MR_v
# todo: for threshold need a range of S_0 maybe import other
# todo: add thresholds and NPV to dataframe
# todo: plot graphs of thresholds and NPV

writer = pd.ExcelWriter("raw_data/time_to_maturity.xlsx", engine="xlsxwriter")
inputs.to_excel(writer, sheet_name="inputs")
df.to_excel(writer, sheet_name="results")
writer.save()

