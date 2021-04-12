import numpy as np
import time
import warnings
import pandas as pd
from LSMC.LSMC_faster import thresholdprice

def GBM(T, dt, paths, mu, sigma, S_0):
    # start timer
    tic = time.time()
    N = T * dt
    N = int(N)
    dt = 1 / dt

    # create wiener increments, half being antithetic
    wiener = np.random.normal(0, np.sqrt(dt), size=(paths, N)).T
    wiener_antithetic = wiener / -1
    wiener = np.hstack((wiener, wiener_antithetic))

    price_matrix = np.exp((mu - sigma ** 2 / 2) * dt + sigma * wiener)
    price_matrix = np.vstack([np.ones(paths*2), price_matrix])
    price_matrix = S_0 * price_matrix.cumprod(axis=0)

    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of GBM: {:.2f} seconds'.format(elapsed_time))

    return price_matrix

def LSMC_RO(price_matrix, r, mu, paths, T, dt, A, Q, epsilon, O_M, Tc, I):
    # start timer
    tic = time.time()

    # total number of steps
    N = T * dt
    N = int(N)

    # adjust yearly discount factor
    r = (1 + r) ** (1 / dt) - 1

    # cash flow matrix
    cf_matrix = np.zeros((N + 1, paths*2))

    # calculated cf when executed in time T (cfs European option)
    cf_matrix[N] = payoff_executing_RO(price_matrix[N], A, Q, epsilon, O_M, r, Tc, I, T)

    # 1 if in the money, otherwise 0
    execute = np.where(payoff_executing_RO(price_matrix, A, Q, epsilon, O_M, r, Tc, I, T) > 0, 1, 0)
    # execute = np.ones_like(execute)       # use to convert to consider all paths

    # Dataframe to store continuation function
    df = pd.DataFrame({"alpha": [],"B1": [], "B2": [], "threshold_price": []})

    for t in range(1, N+1):
        # discounted cf 1 time period
        discounted_cf = cf_matrix[N - t + 1] * np.exp(-r)

        # slice matrix and make all out of the money paths = 0 by multiplying with matrix "execute"
        X = price_matrix[N - t, :] * execute[N - t, :]

        # +1 here because otherwise will loose an in the money path at T-t,
        # that is out of the money in T-t+1(and thus has payoff=0)
        Y = (discounted_cf+1) * execute[N - t, :]

        # mask all zero values(out of the money paths) and run regression
        X1 = np.ma.masked_less_equal(X, 0)
        Y1 = np.ma.masked_less_equal(Y, 0) - 1

        # meaning all paths are out of the money, never optimal to exercise
        if X1.count() > 0:
            regression = np.ma.polyfit(X1, Y1, 2)
            warnings.simplefilter('ignore', np.RankWarning)

            # calculate continuation value
            cont_value = np.zeros_like(Y1)
            cont_value = np.polyval(regression, X1)

            # calculate threshold price
            """
            makes it slower, so hide when not needed
            B2 = regression[0]
            B1 = regression[1]
            alpha = regression[2]
            cont_func = [alpha, B1, B2, thresholdprice(B1, B2, alpha, K)]
            df.loc[len(df.index)] = cont_func
            """

            # update cash flow matrix
            imm_ex = payoff_executing_RO(X1, A, Q, epsilon, O_M, r, Tc, I, T)
            cf_matrix[N - t] = np.ma.where(imm_ex > cont_value, imm_ex, cf_matrix[N - t + 1] * np.exp(-r))
            cf_matrix[N - t + 1:] = np.ma.where(imm_ex > cont_value, 0, cf_matrix[N - t + 1:])
        else:
            cf_matrix[N - t] = cf_matrix[N - t + 1] * np.exp(-r)

    # obtain option value
    option_value = np.sum(cf_matrix[0]) / paths*2

    # st dev
    st_dev = np.std(cf_matrix[0])/np.sqrt(N)

    # thresholdprice
    thresholdprice = ((((I + option_value) * (r - mu) / (1 - Tc)) + O_M) / Q - A) / -epsilon
    # df.to_excel("cont_func.xlsx")

    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of LSMC: {:.2f} seconds'.format(elapsed_time))
    print("Threshold price of the option is: ", thresholdprice)
    print("Value of this option is:", option_value)
    print("St dev of this", type, "option is:", st_dev)

    return option_value, thresholdprice


def payoff_executing_RO(price, A, Q, epsilon, O_M, r, Tc, I, T):
    # todo: is yearly adjust with dt?
    # todo: check this payoff function, some brackets missing
    # discount factor
    DF = (1-(1+r)**-T)/r
    Payoff = (((A - epsilon * price) * Q - O_M) * (1 - Tc) * DF) - I
    return Payoff.clip(min=0)


if __name__ == "__main__":
    # inputs

    # electricity price
    A = 40
    # Quantity per year
    Q = 40000
    # efficiency rate of the plant
    epsilon = 0.85
    # maintenance and operating cost per year
    O_M = 40000
    # initial investment
    I = 1400000
    # tax rate
    Tc = 0.25
    # discount rate (WACC?)
    r = 0.06

    # initial gas price
    S_0 = 20
    # drift rate mu of gas price
    mu = 0.02
    # volatility of the gas price
    sigma = 0.15

    # life of the power plant(in years)
    T_plant = 30
    # life of the option(in years)
    T = 5
    # time periods per year
    dt = 365

    # number of paths per simulations
    paths = 1000

    price_matrix = GBM(T, dt, paths, mu, sigma, S_0)
    value = LSMC_RO(price_matrix, r, mu, paths, T, dt, A, Q, epsilon, O_M, Tc, I)