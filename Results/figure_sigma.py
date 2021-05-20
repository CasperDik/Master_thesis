from Results.standard_RO import standard_RO
import pandas as pd
import matplotlib.pyplot as plt

T = 5
paths = 25000
dt = 50

TPGBM = []
TPMR  = []
TPNPV = []
sigma = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]

for s in sigma:
    print("ran with sigma * ", s, "\n")

    tpGBM, tpMR, tpNPV, inputs = standard_RO(paths, dt, T, s)
    TPGBM.append(tpGBM)
    TPMR.append(tpMR)
    TPNPV.append(tpNPV)

results = pd.DataFrame(columns=["sigma multiplier", "threshold price GBM", "threshold price MR", "threshold price NPV"])
results["sigma multiplier"] = sigma
results["threshold price GBM"] = TPGBM
results["threshold price MR"] = TPMR
results["threshold price NPV"] = TPNPV


writer = pd.ExcelWriter("raw_data/figure_sigma.xlsx", engine="xlsxwriter")
inputs.to_excel(writer, sheet_name="inputs")
results.to_excel(writer, sheet_name="results")
writer.save()

plt.plot(sigma, TPGBM, label="Threshold price GBM")
plt.plot(sigma, TPMR, label="Threshold price MR")
plt.legend()
plt.show()


