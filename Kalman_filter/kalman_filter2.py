from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel(r'C:\Users\Casper Dik\OneDrive\Documenten\MSc Finance\Master Thesis\Data\Gas prices\deflated gas price.xlsx', sheet_name="Sheet1")

Pt = df.drop(columns=["Date", "LRP1", "LRP2"])
Pt = Pt.to_numpy()
measurements = Pt[:50]

kf = KalmanFilter(initial_state_mean=5, n_dim_obs=1)
x = kf.em(measurements).filter(Pt)[0]

plt.plot(Pt)
plt.plot(x)
plt.show()


#kf = KalmanFilter(em_vars=['transition_covariance', 'observation_covariance'])
#x = kf.em(measurements).filter([[1], [1], [1]])[0]
#print(x)

@signal lrp3 = c(1) + c(2)*lrp3(-1) + sv1 + sv2*Time + [var = exp(c(6))]

@state sv1 = c(3)*sv1(-1) + [var = exp(c(5))]
@state sv2 = c(4)*sv2(-1) + [var = exp(c(7))]