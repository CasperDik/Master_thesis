from Results.standard_RO import standard_RO
import pandas as pd
import matplotlib.pyplot as plt

# todo: check if everything works
# todo: code this also for sigma and out of money

# todo: change this range
T = [1, 2, 3]
paths = 25000
dt = 50

TPGBM = []
TPMR  = []
for t in T:
    tpGBM, tpMR, inputs = standard_RO(paths, dt, t, 0)
    TPGBM.append(tpGBM)
    TPMR.append(tpMR)

results = pd.DataFrame(columns=["Time to maturity", "threshold price GBM", "threshold price MR"])
results["Time to maturity"] = T
results["threshold price GBM"] = TPGBM
results["threshold price MR"] = TPMR

writer = pd.ExcelWriter("raw_data/time_to_maturity.xlsx", engine="xlsxwriter")
inputs.to_excel(writer, sheet_name="inputs")
results.to_excel(writer, sheet_name="results")
writer.save()

plt.plot(T, TPGBM, label="Threshold price GBM")
plt.plot(T, TPMR, label="Threshold price MR")
plt.legend()
plt.show()
