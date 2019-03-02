import numpy as np
from scipy.stats import multivariate_normal as norm

class Kalman_filter(object):
    
    """
    #### Class methods inputs ####
    data_file = .txt input data file, entered as a string. Should contain the
                row-wise entries of the system matrices A, Q, H and R
                respectively, where:
                A = transition matrix
                Q = system covariance
                H = observation matrix
                R = observation covariance
    obs = Observation of the state at the current timestep.
    prev_filt = Kalman filter estimate to the state at the previous timestep.
    prev_filt_cov = Kalman filter estimate to the covariance of the filtered 
                    posterior distribution of the state at the previous timestep.
    pred_mean = Predicted value of the state at the current timestep, prior to 
                receiving the latest observation.
    pred_cov = Predicted value of the covariance of the filtered posterior 
               distribution of the state at the current timestep, prior to
               receiving the latest observation.
    
    - __init__ method reads the entries of each matrix from the input file. All
      other methods in the class adapt to all combinations of state and 
      observation space dimensions by virtue of the system and measurement 
      matrices entered in the input file. Note that A, Q and R must be floats or
      square matrices and the observation matrix H must have the same number of
      columns as these matrices.
    
    - KF method returns the estimate to the state and the estimate to the 
      fitered posterior's covariance at the current time step. Note that in the
      case when the state space model is linear Gaussian, these quantities are
      exact.
    
    - KF_predict method returns the predictive mean and the predictive 
      covariance to the filtered distribution at the current time step, prior to
      the current timestep observation arriving.
    
    - KF_update method returns the same quantities as the KF method.
    """

    def __init__(self, data_file):

        #### Read flattened array data. ####
        data = open(data_file, 'r')
        self.A = np.array(list(map(float, data.readline().split())))
        self.Q = np.array(list(map(float, data.readline().split())))
        self.H = np.array(list(map(float, data.readline().split())))
        self.R = np.array(list(map(float, data.readline().split())))
        
        #### Reshape arrays according to their dimensions. ####
        state_dim = int(np.sqrt(self.A.size))
        if state_dim == 1:
            self.A = self.A[0]
            self.Q = self.Q[0]
            self.m0 = self.m0[0]
        else:
            self.A = self.A.reshape((state_dim, state_dim))
            self.Q = self.Q.reshape((state_dim, state_dim))
        if self.H.size == 1:
            obs_dim = state_dim
            self.H = self.H[0]
            self.R = self.R[0]
        elif self.H.size == state_dim:
            obs_dim = 1
            self.R = self.R[0]
        else:
            obs_dim = int(self.H.size/state_dim)
            self.H.reshape((obs_dim, state_dim))
            self.R = self.R.reshape((obs_dim, obs_dim))
    
    
    def KF(self, obs, prev_filt, prev_filt_cov):
        pred_mean, pred_cov = self.KF_predict(prev_filt, prev_filt_cov)
        curr_filt, curr_filt_cov = self.KF_update(pred_mean, pred_cov, obs)
        return curr_filt, curr_filt_cov
    
    
    def KF_predict(self, prev_filt, prev_filt_cov):
        if np.matrix(self.A).shape[0] == 1:
            return mult(self.A, prev_filt), \
            mult(self.A, mult(prev_filt_cov, self.A)) + self.Q
        return mult(self.A, prev_filt), \
        mult(self.A, mult(prev_filt, self.A.T)) + self.Q
    
    
    def KF_update(self, pred_mean, pred_cov, obs):
        v = obs - mult(self.H, pred_mean)
        if np.matrix(self.H).shape[0] == 1:
            S = self.R + mult(self.H, mult(pred_cov, self.H))
            K = mult(mult(pred_cov, self.H), invert(S))
            return pred_mean + mult(K, v), \
            mult((np.eye(len(pred_mean)) - np.outer(K, self.H)), pred_cov)
        S = self.R + mult(self.H, mult(pred_cov, self.H.T))
        K = mult(mult(pred_cov, self.H.T), invert(S))
        return pred_mean + mult(K, v), mult((np.eye(x_dim) - mult(K, H)), pred_cov)


#######################
# AUXILIARY FUNCTIONS #
#######################
   
def mult(x, y):

    """
    - Allows for mathematical interpretation of scalar*scalar, scalar*vector,
    scalar*matrix, vector*matrix and matrix*matrix multiplication (all
    commutative) based on the nature of the inputs.
    - Will compute the scalar product of two vectors but will not perform
    vector*vector = matrix, for which np.outer() is instead required.
    """
        
    if isinstance(x, (int, float, complex)) or isinstance(y, (int, float, \
    complex)):
        return x*y
    return x@y


def invert(x):
    
    """
    Allows for the inverse of x to be calculated regardless of whether x is a
    scalar or a matrix.
    """
    
    if np.matrix(x).shape[0] > 1:
        return np.linalg.inv(x)
    return 1/x