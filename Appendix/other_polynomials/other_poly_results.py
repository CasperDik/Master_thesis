from Appendix.FD_call_american import American_call_grid
from LSMC.LSMC_faster import LSMC, GBM
from other_poly import LSMC_Hermite, LSMC_Legendre, LSMC_Chebyshev, LSMC_Laguerre
import pandas as pd

if __name__ == "__main__":
    paths = 200000
    # years
    T = 2
    # execute possibilities per year
    dt = 50

    S_0 = 36
    sigma = 0.2
    r = 0.06
    q = 0.00
    mu = r - q

    df = pd.DataFrame(columns=["power", "lag", "leg", "cheb", "herm", "FD"])
    i = 0
    polydegree = 6
    for K in [32, 36, 40]:
        for _ in range(5):
            i += 1
            val = []
            price_matrix = GBM(T, dt, paths, mu, sigma, S_0)
            power = LSMC(price_matrix, K, r, paths, T, dt, "call")
            lag = LSMC_Laguerre(price_matrix, K, r, paths, T, dt, "call", polydegree)
            leg = LSMC_Legendre(price_matrix, K, r, paths, T, dt, "call", polydegree)
            cheb = LSMC_Chebyshev(price_matrix, K, r, paths, T, dt, "call", polydegree)
            herm = LSMC_Hermite(price_matrix, K, r, paths, T, dt, "call", polydegree)

            stocks, call_prices = American_call_grid(S_0, T, r, sigma, q, dt, K)
            FD_val = call_prices[dt]
            df.loc[i] = [power, lag, leg, cheb, herm, FD_val]
    df.to_excel("other_poly.xlsx")
