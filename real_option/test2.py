import numpy as np
import matplotlib.pyplot as plt

x = np.logspace(1, 1.34, num=10, base=7.5)
x1 = 7.5 - (x - 7.5)
x1 = x1[::-1]

x = np.hstack((x1, x[1:]))
print(x)
