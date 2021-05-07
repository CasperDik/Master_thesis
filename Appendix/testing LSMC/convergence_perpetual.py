from LSMC.LSMC_faster import GBM, payoff_executing
import numpy as np
import time
import matplotlib.pyplot as plt

# same but stops at year 35 and returns continuation function
def LSMC1(price_matrix, K, r, paths, T, dt, type):
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
    cf_matrix[N] = payoff_executing(K, price_matrix[N], type)

    # 1 if in the money, otherwise 0
    execute = np.where(payoff_executing(K, price_matrix, type) > 0, 1, 0)
    # execute = np.ones_like(execute)       # use to convert to consider all paths

    # end year = 35
    ey = 25
    # total - end
    endloop = N - ey * dt + 1
    for t in range(1, endloop):
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

        if X1.count() > 0:      # meaning all paths are out of the money, thus never optimal to exercise
            regression = np.ma.polyfit(X1, Y1, 2)
            # warnings.simplefilter('ignore', np.RankWarning)

            # calculate continuation value
            cont_value = np.zeros_like(Y1)
            cont_value = np.polyval(regression, X1)

            # update cash flow matrix
            imm_ex = payoff_executing(K, X1, type)
            cf_matrix[N - t] = np.ma.where(imm_ex > cont_value, imm_ex, cf_matrix[N - t + 1] * np.exp(-r))
            cf_matrix[N - t + 1:] = np.ma.where(imm_ex > cont_value, 0, cf_matrix[N - t + 1:])
        else:
            cf_matrix[N - t] = cf_matrix[N - t + 1] * np.exp(-r)

    # option value is average of continuation value in year 35
    option_value = np.sum(cont_value) / (paths*2) * np.exp(-r*dt*ey)

    # st dev
    st_dev = np.std(cont_value)/np.sqrt(paths)

    # threshold value
    threshold_price = option_value + K

    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of LSMC: {:.2f} seconds'.format(elapsed_time))
    print("Ran this with T: ", T, " and dt: ", dt, "\n")

    print("Value of this", type, "option is:", option_value)
    print("St dev of this", type, "option is:", st_dev, "\n")

    print("Threshold price of the option is: ", threshold_price)

    return option_value

