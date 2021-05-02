import numpy as np


def NPV1(price, A, Q, epsilon, O_M, r, Tc, I, T_plant):
    # discount factor
    DF = (1-(1+r)**(-T_plant))/r
    Payoff = (((A - epsilon * price) * Q - O_M) * (1 - Tc) * DF) - I
    return Payoff


def thresholdvalue(OV, NPV, S_0):
    insurance_value = (np.array(OV) - np.array(NPV))

    for x in range(len(insurance_value) - 1):
        if abs(insurance_value[x]) < 0.001:
            insurance_value[x] = 0

    # smallest value greater than zero
    min_val = min(i for i in insurance_value if i > 0)
    # get index of smallest value
    idx = list(insurance_value).index(min_val)

    # approximation of the threshold value
    thresholdvalue = (S_0[idx - 1] + S_0[idx]) / 2
    return thresholdvalue
