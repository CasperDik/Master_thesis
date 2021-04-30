from Results.standard_RO import standard_RO
import pandas as pd
import matplotlib.pyplot as plt

T = 5
paths = 25000
dt = 25

TPGBM = []
TPMR  = []
sigma = [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3]

for s in sigma:
    tpGBM, tpMR, inputs = standard_RO(paths, dt, T, s)
    TPGBM.append(tpGBM)
    TPMR.append(tpMR)

results = pd.DataFrame(columns=["sigma multiplier", "threshold price GBM", "threshold price MR"])
results["sigma multiplier"] = sigma
results["threshold price GBM"] = TPGBM
results["threshold price MR"] = TPMR

writer = pd.ExcelWriter("raw_data/figure_sigma.xlsx", engine="xlsxwriter")
inputs.to_excel(writer, sheet_name="inputs")
results.to_excel(writer, sheet_name="results")
writer.save()

plt.plot(sigma, TPGBM, label="Threshold price GBM")
plt.plot(sigma, TPMR, label="Threshold price MR")
plt.legend()
plt.show()


