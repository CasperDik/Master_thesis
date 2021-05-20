import numpy as np


def NPV1(price, A, Q, epsilon, O_M, r, I, T_plant):
    Payoff = ((A - epsilon * price) * Q - O_M) / r - (((A - epsilon * price) * Q - O_M) * np.exp(-r * T_plant)) / r - I

    return Payoff

def NPV_TP(A, Q, epsilon, O_M, r, I, T_plant):
    NPV_TP = (A * Q - O_M)/(epsilon * Q) - (I * r)/((1-np.exp(-r*T_plant))*epsilon*Q)

    return NPV_TP

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
