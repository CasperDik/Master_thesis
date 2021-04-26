import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel(r'C:\Users\Casper Dik\OneDrive\Documenten\MSc Finance\Master Thesis\Data\Gas prices\deflated gas price.xlsx', sheet_name="Sheet1")

t = 10

x = df["LRP3"][t:-1]
y = df["LRP3"][t+1:, ]
y = y.reset_index(drop=True)
x = x.reset_index(drop=True)

fd = y - x
varfd = fd.var()

lags = np.linspace(1, 40, 40)
Rk = []
VARKD = []
fd_xk = []


for k in lags:
    k = int(k)
    x = df["LRP3"][t:-k]
    y = df["LRP3"][t+k:, ]
    y = y.reset_index(drop=True)
    x = x.reset_index(drop=True)
    kd = y - x
    varkd = kd.var()
    Rk.append(1/k * varkd / varfd)


rk = pd.DataFrame(data=Rk)
rk.to_excel("data variance ratio test.xlsx")
rk.plot(legend=False, title="Variance Ratio - Natural gas")
plt.show()