from LSMC.LSMC_faster import LSMC, GBM
from Appendix.FD_call_american import American_call_grid
from Appendix.testing_LSMC.european_LSMC_vs_analytical import BSM
from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
import pandas as pd

df = pd.read_excel(r'C:\Users\Casper Dik\OneDrive\Documenten\MSc Finance\Master_Thesis_python\option_prices.xlsx')

# set parameters
S_0 = 584.9
r = 0.05
q = 0.008
mu = r - q
T = 2.463

steps = 100
paths = 10000
dt = 50

# create empty lists
ov_BSM = []
ov_FD = []
ov_LSMC = []
imp_vol = []


for i in range(len(df["K"])):
    # implied volatility based on European option since BSM
    imp_vol.append(iv(df["market price"][i], S_0, df["K"][i], T, mu, "c"))

vol_smile = dict(zip(df["K"], imp_vol))

for K in vol_smile:
    strike_price = K
    volatility = vol_smile[K]

    # option values using Black Scholes Merton --> thus European option
    call, put = BSM(S_0, strike_price, r, q, volatility, T)
    ov_BSM.append(call)

    # option values using Finite Differencing --> American option
    stocks, call_prices = American_call_grid(S_0, T, r, volatility, q, steps, K)
    ov_FD.append(call_prices[steps])

    # option values using Least Squares Monte Carlo --> American option
    price_matrix = GBM(T, dt, paths, mu, volatility, S_0)
    ov_LSMC.append(LSMC(price_matrix, K, r, paths, T, dt, "call"))


ov = pd.DataFrame(columns=["actual price", "K", "BSM", "FD", "LSMC"])
ov["actual price"] = df["market price"]
ov["K"] = df["K"]
ov["BSM"] = ov_BSM
ov["FD"] = ov_FD
ov["LSMC"] = ov_LSMC

df.to_excel("ov_compare.xlsx")