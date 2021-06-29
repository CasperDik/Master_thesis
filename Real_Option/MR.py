import numpy as np
import matplotlib.pyplot as plt
import time

# https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

def MRvsGBM(T, dt, paths, sigma, S_0, theta, Sbar, mu):
    N = T * dt
    N = int(N)
    dt = 1 / dt

    wiener = (sigma * np.random.normal(0, np.sqrt(dt), size=(paths, N + 1))).T

    MR_matrix = np.zeros((N+2, paths))
    MR_matrix[0] = S_0
    for i in range(1, N + 2):
        dx = np.exp(theta * (np.log(Sbar) - np.log(MR_matrix[i - 1])) * dt + wiener[i-1])
        MR_matrix[i] = MR_matrix[i - 1] * dx

    price_matrix = np.exp((mu) * dt + wiener)
    price_matrix = np.vstack([np.ones(paths), price_matrix])
    price_matrix = S_0 * price_matrix.cumprod(axis=0)

    return price_matrix, MR_matrix

def MR1(T, dt, paths, sigma, S_0, theta, Sbar):
    tic = time.time()
    N = T * dt
    N = int(N)
    dt = 1 / dt

    wiener = (sigma * np.random.normal(0, np.sqrt(dt), size=(paths, N + 1))).T
    wiener_antithetic = wiener / -1
    wiener = np.hstack((wiener, wiener_antithetic))

    MR_matrix = np.zeros_like(wiener)
    MR_matrix[0] = S_0
    for i in range(1, N + 1):
        dx = theta * (Sbar - MR_matrix[i - 1]) * dt + wiener[i] * MR_matrix[i - 1]
        MR_matrix[i] = MR_matrix[i - 1] + dx

    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of MR1: {:.2f} seconds'.format(elapsed_time))

    return MR_matrix

def MR2(T, dt, paths, sigma, S_0, theta, Sbar):
    tic = time.time()
    N = T * dt
    N = int(N)
    dt = 1 / dt

    wiener = (sigma * np.random.normal(0, np.sqrt(dt), size=(paths, N + 1))).T
    wiener_antithetic = wiener / -1
    wiener = np.hstack((wiener, wiener_antithetic))

    MR_matrix = np.zeros_like(wiener)
    MR_matrix[0] = S_0
    for i in range(1, N + 1):
        dx = np.exp(theta * (np.log(Sbar) - sigma**2/2*theta - np.log(MR_matrix[i - 1])) * dt + wiener[i])
        MR_matrix[i] = MR_matrix[i - 1] * dx

    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time of MR2: {:.2f} seconds'.format(elapsed_time))

    return MR_matrix


def MR3(T, dt, paths, sigma_g, sigma_e, S_0, theta_e, theta_g, Sbar, LR_0):
    tic = time.time()
    N = T * dt
    N = int(N)
    dt = 1 / dt

    dW_G = (sigma_g * np.random.normal(0, np.sqrt(dt), size=(paths, N + 1))).T
    dW_G_antithetic = dW_G / -1
    dW_G = np.hstack((dW_G, dW_G_antithetic))

    dW_E = (sigma_e * np.random.normal(0, np.sqrt(dt), size=(paths, N + 1))).T
    dW_E_antithetic = dW_E / -1
    dW_E = np.hstack((dW_E, dW_E_antithetic))

    # long run equilibrium level
    LR_eq = np.zeros_like(dW_E)
    LR_eq[0] = LR_0

    # price matrix
    MR_matrix = np.zeros_like(dW_G)
    MR_matrix[0] = S_0

    for i in range(1, N+1):
        drift = (theta_e * (np.log(Sbar) - (sigma_e**2)/(2*theta_e) - np.log(LR_eq[i - 1])))
        LR_eq[i] = LR_eq[i-1] * np.exp(drift * dt + dW_E[i])
        MR_matrix[i] = MR_matrix[i-1] * np.exp((theta_g * (np.log(LR_eq[i]) - sigma_g**2/2*theta_g - np.log(MR_matrix[i-1])))*dt + dW_G[i])

    # print("Average long run equilibrium value: ", np.sum(LR_eq[N]/paths))
    # print("Average value at T=", N, ": ",np.sum(MR_matrix[N] / paths))

    toc = time.time()
    elapsed_time = toc - tic
    # print('Total running time of MR1: {:.2f} seconds'.format(elapsed_time))

    return MR_matrix

if __name__ == "__main__":
    T = 1
    dt = 365
    paths = 10

    theta = 2
    sigma = 0.2
    Sbar = 100 # long run equilibrium price
    S_0 = 100

    LR_0 = 100  # initial equilibrium price
    sigma_g = 0.2
    sigma_e = 0.05
    theta_e = 5
    theta_g = 2

    MR1 = MR1(T, dt, paths, sigma, S_0, theta, Sbar)
    MR2 = MR2(T, dt, paths, sigma, S_0, theta, Sbar)
    #MR3 = MR3(T, dt, paths, sigma_g, sigma_e, S_0, theta_e, theta_g, Sbar, LR_0)

    N = T * dt
    plt.plot(np.linspace(0, N, N + 1), MR1, label="MR1", c="r", alpha=0.3)
    plt.plot(np.linspace(0, N, N + 1), MR2, label="MR2", c="b", alpha=0.3)
    #plt.plot(np.linspace(0, N + 1, N + 1), MR3, label="MR3", c="y", alpha=0.2)

    plt.show()


