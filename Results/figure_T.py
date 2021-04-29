from Results.standard_RO import standard_RO
# todo: get dataframe with inputs
# todo: store threshold values in dataframe
# todo: export results and inputs to excel
# todo: plot results
# todo: code this also for sigma

# todo: change this range
T = [1, 2, 3]
paths = 25000
dt = 50

TPGBM = []
TPMR  = []
for t in T:
    tpGBM, tpMR = standard_RO(paths, dt, t, 0)
    TPGBM.append(tpGBM)
    TPMR.append(tpMR)

