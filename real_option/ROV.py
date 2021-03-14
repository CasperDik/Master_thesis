def RO_compare_stochatic(T, dt, paths, mu, sigma, sigma_e, theta, theta_e, Sbar, LR_0, S_0):
    import numpy as np
    from real_option.MR import MR2, MR3
    from real_option.RO_LSMC import GBM, LSMC_RO

    # to better compare the prices, changes this later
    sigma_g = sigma
    theta_g = theta
    LR_0 = Sbar

    # stochastic processes
    GBM = GBM(T, dt, paths, mu, sigma, S_0)
    MR2 = MR2(T, dt, paths, sigma, S_0, theta, Sbar)
    MR3 = MR3(T, dt, paths, sigma_g, sigma_e, S_0, theta_e, theta_g, Sbar, LR_0)

    # real option inputs
    A = 120
    Q = 40000
    epsilon = 0.85
    OPEX = 40000
    I = 1400000
    Tc = 0.25
    r = 0.06

    # generate option values from the different stochastic processes
    x = [[],[],[]]
    i = 0
    for process in [GBM, MR2, MR3]:
        x[i].append(LSMC_RO(process, r, paths, T, dt, A, Q, epsilon, OPEX, Tc, I))
        i += 1
    return np.array(x)

if __name__ == "__main__":
    T = 1
    dt = 150
    paths = 3
    N = T * dt

    theta = 1.1
    sigma = 0.2
    Sbar = 100  # long run equilibrium price
    S_0 = 100
    mu = 0.2

    LR_0 = 100   # initial equilibrium price
    sigma_g = 0.2
    sigma_e = 0.05
    theta_e = 5
    theta_g = 0.2

    RO_compare_stochatic(T, dt, paths, mu, sigma, sigma_e, theta, theta_e, Sbar, LR_0, S_0)
