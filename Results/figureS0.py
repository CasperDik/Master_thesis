import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Real_Option.RO_LSMC import LSMC_RO, GBM
from Real_Option.MR import MR2
from Real_Option.threshold_value import NPV1, thresholdvalue, NPV_TP

if __name__ == "__main__":
    # inputs:
    # real option setting
    A = 30.00
    Q = 4993200
    epsilon = 1/0.6
    O_M = 25*600*1000
    I = 850*1000*600
    Tc = 0.0
    wacc = 0.056
    T_plant = 30

    # initial gas price
    S_0 = np.linspace(4, 12, 40)
    #S_0 = np.linspace(0.5, 7, 40)

    # GBM
    mu = 0.05743
    sigma_gbm = 0.32048

    # MR
    Sbar = 15.305
    theta = 0.006
    sigma_mr = 0.17152

    # life of the option(in years)
    T = 25
    # time periods per year
    dt = 50
    # number of paths per simulations
    paths = 25000

    GBM_v = []
    MR_v = []
    NPV = []

    for s in S_0:
        print("ran with initial gas price ", s, "\n")

        # GBM
        price_matrix_gbm = GBM(T, dt, paths, mu, sigma_gbm, s)
        #GBM_v.append(LSMC_RO(price_matrix_gbm, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I))
        GBM_v.append(
            LSMC_RO(price_matrix_gbm, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, I))

        # MR
        price_matrix_mr = MR2(T, dt, paths, sigma_mr, s, theta, Sbar)
        #MR_v.append(LSMC_RO(price_matrix_mr, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I))
        MR_v.append(
            LSMC_RO(price_matrix_mr, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, I))

        NPV.append(NPV1(s, A, Q, epsilon, O_M, wacc, I, T_plant))
    threshold_GBM = thresholdvalue(GBM_v, NPV, S_0)
    threshold_MR = thresholdvalue(MR_v, NPV, S_0)
    print("thresholdprice GBM: ", threshold_GBM, "thresholdprice MR: ",  threshold_MR)



    df = pd.DataFrame(columns=["S_0", "value GBM", "value MR", "NPV"])
    inputs = pd.DataFrame({"_":["A","Q","Epsilon","O&M", "I", "Tc", "wacc", "Tplant", "S0", "mu", "sigmaGBM", "Sbar", "theta", "sigmaMR", "dt", "pahts", "T"],
                     "Inputs": [A, Q, epsilon, O_M, I, Tc, wacc, T_plant, S_0, mu, sigma_gbm, Sbar, theta, sigma_mr, dt, paths, T]})

    df["S_0"] = S_0
    df["value GBM"] = GBM_v
    df["value MR"] = MR_v
    df["NPV"] = NPV

    NPV0 = NPV_TP(A, Q, epsilon, O_M, wacc, I, T_plant)
    row = ["threshold prices: ", threshold_GBM, threshold_MR, NPV0]
    df.loc[len(df)] = row

    writer = pd.ExcelWriter("raw_data/rangeS0.xlsx", engine="xlsxwriter")
    inputs.to_excel(writer, sheet_name="inputs")
    df.to_excel(writer, sheet_name="results")
    writer.save()

    plt.plot(S_0, NPV, label="NPV")
    plt.plot(S_0, GBM_v, label="GBM")
    plt.plot(S_0, MR_v, label="MR")
    plt.axvline(threshold_GBM, c="r", linestyle="--", label="critical gas price GBM", alpha=0.5, linewidth=0.3)
    plt.axvline(threshold_MR, c="r", linestyle="--", label="critical gas price MR", alpha=0.5, linewidth=0.3)
    plt.axhline(0, c="k", linestyle="--", linewidth=0.3)

    plt.legend()
    plt.show()

