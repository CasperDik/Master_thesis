r'''
==================================
Kalman Filter tracking a sine wave
==================================
This example shows how to use the Kalman Filter for state estimation.
In this example, we generate a fake target trajectory using a sine wave.
Instead of observing those positions exactly, we observe the position plus some
random noise.  We then use a Kalman Filter to estimate the velocity of the
system as well.
The figure drawn illustrates the observations, and the position and velocity
estimates predicted by the Kalman Smoother.
'''
import numpy as np
import pylab as pl
import pandas as pd
from pykalman import KalmanFilter

rnd = np.random.RandomState(0)

# generate a noisy sine wave to act as our fake observations
#n_timesteps = 100
#x = np.linspace(0, 3 * np.pi, n_timesteps)
#observations = 20 * (np.sin(x) + 0.5 * rnd.randn(n_timesteps))

df = pd.read_excel(r'C:\Users\Casper Dik\OneDrive\Documenten\MSc Finance\Master Thesis\Data\Gas prices\deflated gas price.xlsx', sheet_name="Sheet1")
x = df["Date"]
df = df["LRP3"]
observations = df.to_numpy()
x = x.to_numpy()

# create a Kalman Filter by hinting at the size of the state and observation
# space.  If you already have good guesses for the initial parameters, put them
# in here.  The Kalman Filter will try to learn the values of all variables.
#kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
#                  transition_covariance=0.01 * np.eye(2))
kf = KalmanFilter(initial_state_mean=1, n_dim_obs=1)

# You can use the Kalman Filter immediately without fitting, but its estimates
# may not be as good as if you fit first.
states_pred = kf.em(observations).smooth(observations)[0]
print('fitted model: {0}'.format(kf))

# Plot lines for the observations without noise, the estimated position of the
# target before fitting, and the estimated position after fitting.
obs_scatter = pl.scatter(x, observations, marker='x', color='b',
                         label='observations', alpha=0.5)
position_line = pl.plot(x, states_pred[:, 0],
                        linestyle='-', marker='o', color='r',
                        label='estimate', alpha=0.5)

pl.legend(loc='lower right')
pl.xlim(xmin=1920, xmax=2025)
pl.xlabel('time')
pl.show()