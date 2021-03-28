import pandas as pd
import numpy as np
from pykalman import KalmanFilter

def your_function_name(x):
    kf = KalmanFilter(transition_matrices = [1],
                                    observation_matrices = [1],
                                    observation_covariance=1,
                                    transition_covariance=.01,
                                    initial_state_mean = 0,
                                    initial_state_covariance = 1)

    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    return state_means


def your_regression_filter(x, y):
    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2)  #random walk wiggle
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    kf = KalmanFilter(n_dim_obs=1,
                      n_dim_state=2,
                      initial_state_mean=[0,0],
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      observation_covariance=2,
                      transition_covariance=trans_cov)

    # Use the observations y to get running estimates and errors for the state parameters
    state_means, state_covs = kf.filter(y.values)
    return state_means