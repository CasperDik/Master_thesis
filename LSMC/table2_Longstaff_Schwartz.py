from LSMC.LSMC_faster import LSMC
from LSMC.LSMC_faster import GBM
import pandas as pd

K = 40
dt = 50
paths = 100000
mu = 0.06
r = 0.06
value = []

for S in [36, 44]:
    for T in [1, 2]:
        for sigma in [0.2, 0.4]:
            for i in range(5):
                price_matrix = GBM(T, dt, paths, mu, sigma, S)
                value.append(LSMC(price_matrix, K, r, paths, T, dt, "put"))

df = pd.DataFrame(data=value)
df.to_excel("data_table2.xlsx")
