import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.laguerre import lagfit, lagval
import random

random.seed(1)

x = np.linspace(10,100,30)
y = random.sample(range(10, 100), 30)
y = np.ma.masked_less_equal(y, 50)
print(np.linspace(4,12,9))
p = lagfit(x,y,3)
print(p)
c = lagval(x, p)

plt.scatter(x, y, label="actual")
plt.plot(x, c, label="fit")
plt.legend()
plt.show()
