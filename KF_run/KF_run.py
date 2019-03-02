#!/usr/bin/env python

"""
================================================
Kalman filter applied to a linear Gaussian model
================================================
- The matrices contained in the 'KF_data.txt' file describe the state and
  observation properties of the state space model. See the Kalman_filter class
  for a description of how to enter these matrices.

- The script takes one value as an input: the time at which the simulations end.
"""

import sys
<<<<<<< HEAD
#### Add file path for the kalman_filters class. ####
=======
>>>>>>> origin/master
sys.path.insert(0, '/Users/danburrows/Desktop/Programming/Python/PhD/Classes')
import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from kalman_filters import Kalman_filter


########
# DATA #
########
kf = Kalman_filter('KF_data.txt')   
T = int(sys.argv[1])                # Finishing time.


##################
# STORAGE ARRAYS #
##################
state_dim = np.matrix(kf.A).shape[0]
obs_dim = np.matrix(kf.H).shape[0]
<<<<<<< HEAD
states = np.empty((T + 1, state_dim))
filts = np.empty((T + 1, state_dim))
obs = np.empty((T + 1, obs_dim))
=======
if state_dim == 1:
    states = np.empty(T + 1)
    filts = np.empty(T + 1)
else:
    states = np.empty((T + 1, state_dim))
    filts = np.empty((T + 1, state_dim))
if obs_dim == 1:
    obs = np.empty(T + 1)
else:
    obs = np.empty((T + 1, obs_dim))
>>>>>>> origin/master
    

##################
# INITIAL VALUES #
##################
np.random.seed(13101988)
states[0] = norm.rvs(cov=kf.Q)
filts[0] = states[0]
filt_cov = kf.Q


#######################
# AUXILIARY FUNCTIONS #
#######################
def mult(x, y):        
    if isinstance(x, (int, float, complex)) or \
    isinstance(y, (int, float, complex)):
        return x*y
    return x@y


####################
# MODEL SIMULATION #
####################
for t in np.arange(1, T + 1):
    
    #### System evolution. ####
    states[t] = norm.rvs(mean=mult(kf.A, states[t - 1]), cov=kf.Q)
    obs[t] = norm.rvs(mean=mult(kf.H, states[t]), cov=kf.R)
    
    #### Kalman filter estimates. ####
    filts[t], filt_cov = kf.KF(obs[t], filts[t - 1], filt_cov)


#########
# PLOTS #
#########
ax = plt.subplot(111)
ax.plot(np.arange(T + 1), states[:, 0], label='States')
ax.plot(np.arange(T + 1), filts[:, 0], label='Estimates')
ax.set(xlabel='First coordinate values', ylabel='Second coordinate values', \
title='Plot of simulated 2-dimensional trajectory and its Kalman filter \
estimate')
ax.legend()
plt.show()