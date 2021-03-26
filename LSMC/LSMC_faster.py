import numpy as np
import time
import matplotlib.pyplot as plt
import warnings


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

    # plt.plot(np.linspace(0, N+1, N+1), price_matrix)
    # plt.show()

    return price_matrix


def payoff_executing(K, price, type):
    if type == "put":
        payoff_put = K - price
        return payoff_put.clip(min=0)
    elif type == "call":
        payoff_call = price - K
        return payoff_call.clip(min=0)
    else:
        print("Error, only put or call is possible")
        raise SystemExit(0)

def thresholdprice(B1,B2, alpha, K, X1):
    if X1.count() > 0:
        print(">", alpha, B1, B2)
        threshold_price_plus = ((1-B1)+np.sqrt((B1-1)**2 - 4 * B2 * (alpha+K)))/(2*B2)
        print(threshold_price_plus, "+")
        threshold_price_minus = ((1 - B1) - np.sqrt((B1 - 1) ** 2 - 4 * B2 * (alpha + K))) / (2 * B2)
        print(threshold_price_minus, "-")
        threshold_price = min(threshold_price_minus, threshold_price_plus)

    else:        # todo: doesnt work because if all out of the money, there is no regression + function is not called
        print("<=")
        threshold_price_plus = (-B1 + np.sqrt(B1**2-4*B2*alpha))/(2*B2)
        print(threshold_price_plus, "+")
        threshold_price_minus = (-B1 - np.sqrt(B1**2-4*B2*alpha))/(2*B2)
        print(threshold_price_minus, "-")
        threshold_price = min(threshold_price_minus, threshold_price_plus)
    print(threshold_price)

    if np.isnan(threshold_price) == True:
        X = np.linspace(K, 2*K, 41)
        imm_ex = payoff_executing(K, X, "call")
        cont = alpha + B1*X + B2 * X**2
        plt.plot(X, imm_ex, label="imm_ex", linewidth=0.2, alpha=1)
        plt.plot(X, cont, label="continuation value", linewidth=0.2, alpha=1)
        plt.legend()
        plt.show()
    # graph
    # think it doesnt intersect

    return threshold_price

def LSMC(price_matrix, K, r, paths, T, dt, type):
    # start timer
    tic = time.time()

    # total number of steps
    N = T * dt
    N = int(N)

    # adjust yearly discount factor
    r = (1 + r) ** (1 / dt) - 1

    # returns -1 if call, 1 for put --> this way the inequality statements can be used for both put and call
    sign = 1
    if type == "call":
        sign = -1

    # cash flow matrix
    cf_matrix = np.zeros((N + 1, paths*2))

    # calculated cf when executed in time T (cfs European option)
    cf_matrix[N] = payoff_executing(K, price_matrix[N], type)

    # 1 if in the money, otherwise 0
    execute = np.where(payoff_executing(K, price_matrix, type) > 0, 1, 0)
    # execute = np.ones_like(execute)       # use to convert to consider all paths

    # threshold price matrix
    threshold_price = np.zeros((N+1, 1))

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

            # calculate threshold price
            B2 = regression[0]
            B1 = regression[1]
            alpha = regression[2]
            # threshold_price[N-t] = thresholdprice(B1, B2, alpha, K, X1)

            # update cash flow matrix
            imm_ex = payoff_executing(K, X1, type)
            cf_matrix[N - t] = np.ma.where(imm_ex > cont_value, imm_ex, cf_matrix[N - t + 1] * np.exp(-r))
            cf_matrix[N - t + 1:] = np.ma.where(imm_ex > cont_value, 0, cf_matrix[N - t + 1:])
        else:
            cf_matrix[N - t] = cf_matrix[N - t + 1] * np.exp(-r)

    # obtain option value
    cf_matrix[0] = cf_matrix[1] * np.exp(-r)
    option_value = np.sum(cf_matrix[0]) / (paths*2)

    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of LSMC: {:.2f} seconds'.format(elapsed_time))

    print("Value of this", type, "option is:", option_value)
    print("Ran this with T: ", T, " and dt: ", dt)

    return option_value


if __name__ == "__main__":
    # inputs
    """
    # example longstaff and schwartz
    price_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88], [1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.84, 1.22], [1.34, 1.54, 1.03, 0.92, 1.52, 0.9, 1.01, 1.34]])
    paths = 8
    T = 3
    K = 1.1
    rf = 0.06
    """

    paths = 15000
    # years
    T = 1
    # execute possibilities per year
    dt = 50

    K = 32
    S_0 = 36
    sigma = 0.2
    r = 0.06
    q = 0.00
    mu = r - q

    price_matrix = GBM(T, dt, paths, mu, sigma, S_0)
    value = LSMC(price_matrix, K, r, paths, T, dt, "put")
