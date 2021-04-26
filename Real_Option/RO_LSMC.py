import numpy as np
import time
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from Real_Option.threshold_value import NPV1

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

def LSMC_RO(price_matrix, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I, S_0, n):
    # start timer
    tic = time.time()

    # total number of steps
    N = T * dt
    N = int(N)

    # adjust yearly discount factor
    r = (1 + wacc) ** (1 / dt) - 1

    # cash flow matrix
    cf_matrix = np.zeros((N + 1, paths*2))

    # calculated cf when executed in time T (cfs European option)
    cf_matrix[N] = payoff_executing_RO(price_matrix[N], A, Q, epsilon, O_M, wacc, Tc, I, T_plant, S_0)

    # 1 if in the money, otherwise 0
    execute = np.where(payoff_executing_RO(price_matrix, A, Q, epsilon, O_M, wacc, Tc, I, T_plant, S_0) > 0, 1, 0)
    # execute = np.ones_like(execute)       # use to convert to consider all paths
    # df = pd.DataFrame(columns=["alpha", "B1", "B2", "C*_+", "C*_-"])

    for t in range(1, N):
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
            if t > N-2:
                DF = (1-(1+wacc)**(-T_plant))/wacc
                a = regression[0]
                b = regression[1] + epsilon * Q * (1-Tc) * DF
                c = regression[2] - (A * Q - O_M) * (1 - Tc) * DF + I

                threshold = max(np.roots([a, b, c]))

            # update cash flow matrix
            imm_ex = payoff_executing_RO(X1, A, Q, epsilon, O_M, wacc, Tc, I, T_plant, S_0)
            cf_matrix[N - t] = np.ma.where(imm_ex > cont_value, imm_ex, cf_matrix[N - t + 1] * np.exp(-r))
            cf_matrix[N - t + 1:] = np.ma.where(imm_ex > cont_value, 0, cf_matrix[N - t + 1:])
        else:
            cf_matrix[N - t] = cf_matrix[N - t + 1] * np.exp(-r)

    # obtain option value
    cf_matrix[0] = cf_matrix[1] * np.exp(-r)
    option_value = np.sum(cf_matrix[0]) / (paths*2)

    # st dev
    st_dev = np.std(cf_matrix[0])/np.sqrt(paths)

    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of LSMC: {:.2f} seconds'.format(elapsed_time), "\n")
    print("Value of this option is:", option_value, "with a critical gas price of: ", threshold)
    print("St dev of this option is:", st_dev, "\n")

    """
    if n == 1:
        xra = np.linspace(2, 19, 20)
        npvvv = NPV1(xra, A, Q, epsilon, O_M, wacc, Tc, I, T_plant)
        cont_func = regression[2] + regression[1] * xra + regression[0] * xra ** 2
        plt.plot(xra, npvvv)
        plt.plot(xra, cont_func)
        plt.axvline(thresholdvalue_plus, label="Threshold value", linestyle="--", c="r")
        plt.show()

        price_matrix1 = GBM(T, dt, paths, mu, sigma, thresholdvalue_plus)
        print(thresholdvalue_plus, "threshold",
              LSMC_RO(price_matrix1, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I, thresholdvalue_plus, 0) -
              NPV1(thresholdvalue_plus, A, Q, epsilon, O_M, wacc, Tc, I, T_plant))
        
    """
    return option_value, threshold


def payoff_executing_RO(price, A, Q, epsilon, O_M, wacc, Tc, I, T_plant, S_0):
    # discount factor
    DF = (1-(1+wacc)**(-T_plant))/wacc
    Payoff = (((A - epsilon * price) * Q - O_M) * (1 - Tc) * DF) - I
    return Payoff.clip(min=0)


if __name__ == "__main__":
    # inputs

    # electricity price
    A = 30.38
    # Quantity per year
    Q = 4993200
    # efficiency rate of the plant
    epsilon = 1/0.55
    # maintenance and operating cost per year
    O_M = 13200000
    # initial investment
    I = 487200000
    # tax rate
    Tc = 0.21
    # discount rate (WACC?)
    wacc = 0.056

    # initial gas price
    S_0 = 8.00
    # drift rate mu of gas price
    mu = 0.0
    # volatility of the gas price
    sigma = 0.2

    # life of the power plant(in years)
    T_plant = 30
    # life of the option(in years)
    T = 6
    # time periods per year
    dt = 10

    # number of paths per simulations
    paths = 10000

    price_matrix = GBM(T, dt, paths, mu, sigma, S_0)
    value = LSMC_RO(price_matrix, wacc, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I, S_0, 1)