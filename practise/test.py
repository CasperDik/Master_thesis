import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.laguerre import lagfit, lagval


x = np.linspace(10,100,10)
z = np.random.randn(1, 10)

y = lagval(x, [1, 2, 3])

print(lagfit(x, y, 2))
