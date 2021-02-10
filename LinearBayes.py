import numpy as np
from numpy.random import normal, uniform
from scipy.stats import multivariate_normal as mv_norm
import matplotlib.pyplot as plt

class LinearBayes: 
    """
    A class that holds parameter prior/posterior and handles 
    the hyper-parameter updates with new data

    Note:  variables starting with "v_" indicate Nx1 dimensional 
        column vectors, those starting with "m_" indicate 
        matrices, and those starting with "a_" indicate 
        1xN dimensional arrays.

    Args:
        a_m0 (np.array): prior mean vector of size 1xM
        m_S0 (np.ndarray): prior covariance matrix of size MxM
        beta (float): known real-data noise precision
    """
    def __init__(self, a_m0, m_S0, beta, dataSize, x_start, x_end, x_real):
        print('Initializing Bayesian Regression...')
        #LinearBayes(v_m0, m_S0, beta) 
        #v_m0 = np.array([0., 0.]) 
        #m_S0 = 1/alpha*np.identity(2)
        self.prior = mv_norm(mean=a_m0, cov=m_S0)
        self.v_m0 = a_m0.reshape(a_m0.shape + (1,)) #reshape to column vector
        self.m_S0 = m_S0
        self.beta = beta
        
        self.v_mN = self.v_m0
        self.m_SN = self.m_S0
        self.posterior = self.prior
        self.dataSize = dataSize
        self.x_start = x_start
        self.x_end = x_end
        self.x_real = x_real
           
    def get_phi(self, a_x):
        """
        Returns the design matrix of size (NxM) for a feature vector v_x.
        In this case, this function merely adds the phi_0 dummy basis
        that's equal to 1 for all elements.
        
        Args:
            a_x (np.array): input features of size 1xN
        """
        m_phi = np.ones((len(a_x), 4))
        m_phi[:, 1] = a_x
        m_phi[:, 2] = a_x**2
        m_phi[:, 3] = a_x**3
        return m_phi
        
    def set_posterior(self, a_x, a_t):
        """
        Updates mN and SN given vectors of x-values and t-values
        """
        # Need to convert v_t from an array into a column vector
        # to correctly compute matrix multiplication
        v_t = a_t.reshape(a_t.shape + (1,))

        m_phi = self.get_phi(a_x)
        
        self.m_SN = np.linalg.inv(np.linalg.inv(self.m_S0) + self.beta*m_phi.T.dot(m_phi))
        self.v_mN = self.m_SN.dot(np.linalg.inv(self.m_S0).dot(self.v_m0) + \
                                      self.beta*m_phi.T.dot(v_t))
        
        self.posterior = mv_norm(mean=self.v_mN.flatten(), cov=self.m_SN)

    
    def prediction_limit(self, a_x, stdevs):
        """
        Calculates the limit that's "stdevs" standard deviations
        away from the mean at a given value of x.
        
        Args:
            a_x (np.array): x-axis values of size 1xN
            stdevs (float): Number of standard deviations away from
                the mean to calculate the prediction limit
        
        Returns:
            np.array: the prediction limit "stdevs" standard deviations
                away from the mean corresponding to x-values in "v_x"
        
        """
        N = len(a_x)
        m_x = self.get_phi(a_x).T.reshape((4, 1, N))
        
        predictions = []
        for idx in range(N):
            x = m_x[:,:,idx]
            sig_sq_x = 1/self.beta + x.T.dot(self.m_SN.dot(x))
            mean_x = self.v_mN.T.dot(x)
            predictions.append((mean_x+stdevs*np.sqrt(sig_sq_x)).flatten())
        return np.concatenate(predictions)
    
    def generate_data(self, a_x):
        N = len(a_x)
        m_x = self.get_phi(a_x).T.reshape((4, 1, N))
        
        predictions = []
        for idx in range(N):
            x = m_x[:,:,idx]
            sig_sq_x = 1/self.beta + x.T.dot(self.m_SN.dot(x))
            mean_x = self.v_mN.T.dot(x)
            predictions.append(normal(mean_x.flatten(), np.sqrt(sig_sq_x)))
        return np.array(predictions)
    
    def make_contour(self, a_x, a_y, which_two, real_parms=[], N=0):
        """
        A helper function to generate contour plots of our probability distribution
        """
        pos = np.empty(a_x.shape + (4,))
        pos[:, :, which_two[0]] = a_x
        pos[:, :, which_two[1]] = a_y
        plt.contourf(a_x, a_y, self.posterior.pdf(pos), 23)
        plt.xlabel(f'w{which_two[0]}', fontsize=16)
        plt.ylabel(f'w{which_two[1]}', fontsize=16)
        
        if real_parms:
            plt.scatter(real_parms[which_two[0]], real_parms[which_two[1]], marker='+', c='white', s=100, label = f'{real_parms[which_two[0]]}, {real_parms[which_two[1]]}')
            
        _ = plt.title('Weight Parameters Distribution - %d datapoint(s)' % N, fontsize=10)
    
    def make_scatter(self, a_x, a_t, real_parms, samples=None, stdevs=None):
        """
        A helper function to plot noisey data, the true function, 
        and optionally a set of lines specified by the nested array of
        weights of size NxM where N is number of lines, M is 2 for 
        this simple model
        """
        
        plt.scatter(a_x, a_t, alpha=1, label=f'available input data, total count {len(a_x)}', c='k', s=100, marker='o')
        plt.xlabel('x')
        plt.ylabel('t')

        x_range = np.linspace(self.x_start,self.x_end, self.dataSize)
        plt.plot(self.x_real, self.real_function(real_parms, 0, self.x_real), 'r', label='true curve')

        _ = plt.title('Real Data from Noisey Linear Function')
        
        if samples:
            weights = self.posterior.rvs(samples)
            for weight in weights: 
                plt.plot(x_range, self.real_function([weight[0], weight[1], weight[2], weight[3]], 0, x_range), 'black')
                _ = plt.title(f'Lines (black) Sampled from Posterior Distribution vs Real Line and Data with {len(a_x)} data points')
                
        if stdevs:
            y_upper = self.prediction_limit(x_range, stdevs)
            y_lower = self.prediction_limit(x_range, -stdevs)
            plt.plot(x_range, y_upper, c='green', linewidth=3, label='upper prediction')
            plt.plot(x_range, y_lower, c='blue', linewidth=3, label='lower prediction')
            _ = plt.title(f'Lines (black) Sampled from Posterior Distribution vs Real Line and Data with {len(a_x)} data points')
    
    def real_function(self, real_parms, noise_sigma, x):
        """
        Evaluates the real function
        """
        a_0 = real_parms[0]
        a_1 = real_parms[1]
        a_2 = real_parms[2]
        a_3 = real_parms[3]
        N = len(x)
        if noise_sigma==0:
            # Recovers the true function
            return a_0 + a_1*x + a_2*x**2 + a_3*x**3
        else:
            return a_0 + a_1*x + a_2*x**2 + a_3*x**3 + normal(0, noise_sigma, N)