import numpy as np
from Real_Option.RO_LSMC import GBM
from Real_Option.MR import MR2

# todo: range with different T --> threshold price differs per T
# todo: range for different S_0

T = 20
dt = 4
paths = 100000
sigma = 0.2
mu = 0.05
S_0 = 12

thresholdvalue = 8

price_matrix = GBM(T, dt, paths, mu, sigma, S_0)

# minimal values of each path
x = price_matrix.min(axis=0)
# sum if min value above threshold
Pinvesting = sum(x < thresholdvalue)/(paths * 2)

print(Pinvesting)
