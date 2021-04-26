import numpy as np
import matplotlib.pyplot as plt
from intersect import intersection

def NPV1(price, A, Q, epsilon, O_M, r, Tc, I, T_plant):
    # discount factor
    DF = (1-(1+r)**(-T_plant))/r
    Payoff = (((A - epsilon * price) * Q - O_M) * (1 - Tc) * DF) - I
    return Payoff

if __name__ == "__main__":
    from Real_Option.RO_LSMC import LSMC_RO, GBM

    # inputs

    # electricity price
    A = 30.38
    # Quantity per year
    Q = 4993200
    # efficiency rate of the plant
    epsilon = 1 / 0.55
    # maintenance and operating cost per year
    O_M = 13200000
    # initial investment
    I = 487200000
    # tax rate
    Tc = 0.21
    # discount rate (WACC?)
    r = 0.056

    # initial gas price
    #S_0 = 8.00
    # drift rate mu of gas price
    mu = 0.0
    # volatility of the gas price
    sigma = 0.4

    # life of the power plant(in years)
    T_plant = 30
    # life of the option(in years)
    T = 1
    # time periods per year
    dt = 10

    # number of paths per simulations
    paths = 10000

    # initial gas price
    S_0 = np.linspace(0.5,15,10)

    NPV = []
    OV = []

    for s in S_0:
        price_matrix = GBM(T, dt, paths, mu, sigma, s)
        value = LSMC_RO(price_matrix, r, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I, s)
        npv = NPV1(s, A, Q, epsilon, O_M, r, Tc, I, T_plant)
        OV.append(value)
        NPV.append(npv)

    x1 = OV
    y1 = S_0

    x2 = NPV
    y2 = S_0

    x, y = intersection(x1, y1, x2, y2)
    plt.plot(y1, x1, c="r")
    plt.plot(y2, x2, c="g")
    plt.plot(y[-1], x[-1], "*k")

    thresholdvalue1 = y[-1]

    plt.show()

    #"""
    insurance_value = (np.array(OV) - np.array(NPV))
    for x in range(len(insurance_value)-1):
        if np.isclose(insurance_value[x], insurance_value[x+1], atol=1e-05) == True:
            insurance_value[x] = 0
            insurance_value[x+1] = 0
    min_val = min(i for i in insurance_value if i > 0)
    insurance_value = list(insurance_value)
    idx = insurance_value.index(min_val)

    # approximation of the threshold value
    thresholdvalue = (S_0[idx-1] + S_0[idx])/2

    #todo: store estimates or sth to make faster
    price_matrix = GBM(T, dt, paths, mu, sigma, thresholdvalue)
    value = LSMC_RO(price_matrix, r, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I, thresholdvalue)
    npv = NPV1(thresholdvalue, A, Q, epsilon, O_M, r, Tc, I, T_plant)
    print(thresholdvalue, value-npv, "initial estimate")
    iteration = 0
    while value - npv > 1500000:
        iteration += 1
        thresholdvalue -= 0.05
        price_matrix = GBM(T, dt, paths, mu, sigma, thresholdvalue)
        value = LSMC_RO(price_matrix, r, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I, thresholdvalue)
        npv = NPV1(thresholdvalue, A, Q, epsilon, O_M, r, Tc, I, T_plant)
        print(thresholdvalue, value - npv, "iteration_step1", iteration)
        if value - npv < 1500000:
            while value - npv > 100000:
                iteration +=1
                thresholdvalue -= 0.01
                price_matrix = GBM(T, dt, paths, mu, sigma, thresholdvalue)
                value = LSMC_RO(price_matrix, r, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I, thresholdvalue)
                npv = NPV1(thresholdvalue, A, Q, epsilon, O_M, r, Tc, I, T_plant)
                print(thresholdvalue, value - npv, "iteration_step2", iteration)
                if value - npv < 10:
                    while value - npv < 10:
                        iteration +=1
                        thresholdvalue += 0.001
                        price_matrix = GBM(T, dt, paths, mu, sigma, thresholdvalue)
                        value = LSMC_RO(price_matrix, r, paths, T, T_plant, dt, A, Q, epsilon, O_M, Tc, I, thresholdvalue)
                        npv = NPV1(thresholdvalue, A, Q, epsilon, O_M, r, Tc, I, T_plant)
                        print(thresholdvalue, value - npv, "iteration_step3", iteration)
    
    plt.axvline(thresholdvalue, label="Threshold value",linestyle="--", c="r")
    plt.plot(S_0, NPV, label="NPV")
    plt.plot(S_0, OV, label="option value")

    plt.legend()
    plt.show()
    print(thresholdvalue1, "<-------")
    #"""
