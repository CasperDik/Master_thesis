from numpy.polynomial.laguerre import lagfit, lagval
from numpy.polynomial.legendre import legval, legfit
import numpy as np
import matplotlib.pyplot as plt

N = 100
X = np.linspace(0, N, N)
Y = np.sin(0.3 * X)
Y = np.sin(0.1 * (X + np.random.normal(0, 0.1, size=100)))

regression = lagfit(X, Y, 4)
regression1 = legfit(X, Y, 4)

fit = lagval(X, regression)
fit1 = legval(X, regression1)
print(fit - fit1)

plt.plot(X, Y, linewidth=1, alpha=1, label="Actual")
plt.plot(X, fit, linewidth=0.5, alpha=1, label="Laguerre")
plt.plot(X, fit1, linewidth=0.5, alpha=1, label="Legendre")
plt.legend()
plt.show()
