import numpy as np
import pylab as plt
from scipy.stats import chi2
from matplotlib.patches import Ellipse

def generate_model(z, param_vec, zc=0):
    """
    This function generate our polynomial model
    
    Parameters
     ----------
    z: 1d array
        the redshifts at which the model is constructed
    param_vec: list
        the list of parameters of our model
    zc: integer
        the redshift pivot
    """
    d = 0
    for pow, param in enumerate(param_vec):
        d += param * (z - zc) ** pow
    return d

def generate_covariance(z, sigma0, sigma1, n, corr):
    """
    This function generate the noise covariance matrix
    from a set of parameters
    
    Parameters
     ----------
    z: 1d array
        the redshifts at which the model is constructed
    sigma0: float
        the parameter for the cst part of the noise
    sigma1: float
        the parameter for the noise evolution as a function of z
    n: integer
        the power law we consider for the noise evolution as a function of z
    """
    ones = np.ones(len(z))
    corr = np.diag(ones, 0) + np.diag(ones[:-1] * corr, -1) + np.diag(ones[:-1] * corr, 1)
    var =  sigma0 ** 2 + sigma1 ** 2 * (1 + z) ** n
    cov = corr * np.sqrt(var[:,None] * var[None,:])
    return cov
    
def generate_noise(cholensky_cov):
    """
    Draw a noise realisation from a covariance matrix
    
    Parameters
     ----------
    cholensky_cov: 2d array
        the Cholensky decomposition of the covariance matrix
    """
    n = cholensky_cov.shape[0]
    norm_noise = np.random.randn(n)
    noise = cholensky_cov @ norm_noise
    return noise

def get_ponting_matrix(n_obs, z, n_params, z_c=0):
    """
    Computing the pointing matrix associated to our problem
    
    Parameters
     ----------
    n_obs: integer
        the number of observations
    z: 1d array
        the redshifts at which the model is constructed
    n_params: integer
        the number of parameters in our model
    zc: integer
        the redshift pivot
    """

    P = np.zeros((n_obs, n_params))
    for pow in range(n_params):
        P[:, pow] = (z - z_c) ** pow
    return P
    
def get_analytic_param_cov(P, inv_data_cov):
    """
    Compute the analytic covariance matrix of our parameters
    
    Parameters
     ----------
    P: 2d array of shape (n_obs, n_params)
        the pointing matrix that project the parameters
        into observation
    inv_data_cov: 2d array of shape (n_obs, n_obs)
        the inverse covariance matrix of the noise
    """
    param_cov = np.linalg.inv(P.T @ inv_data_cov @ P)
    return param_cov
    
def get_best_fit_params(data, inv_data_cov, param_cov, P):
    """
    Compute the Maximum Likelihood solution
    
    Parameters
     ----------
    data: 1d array of shape n_obs
        the data
    inv_data_cov: 2d array of shape (n_obs, n_obs)
        the inverse covariance matrix of the noise
    param_cov: 2d array of shape (n_params, n_params)
        the covariance matrix of the parameters
    """

    param_estimated = param_cov @ P.T @ inv_data_cov @ data
    return param_estimated

def get_sigma_and_corr(cov):
    """
    get the stds and correlation associated to a covariance matrix
    
    Parameters
     ----------
    cov: 2d array
        the covariance matrix
    """

    sigma = np.sqrt(cov.diagonal())
    corr = cov / (sigma[:, None] * sigma[None, :])
    return sigma, corr

def get_chi2(data, model, inv_data_cov):
    """
    compute a chi2
    
    Parameters
     ----------
    data: 1d array
        the data
    model: 1d array
        the model
    inv_data_cov: 2d array
        the inverse covariance matrix of the noise
    """

    residual = data - model
    chi2_res = residual.T @ inv_data_cov @ residual
    return chi2_res

def get_p_value(chi2_val, dof):
    """
    get the p value associated to a chi2
    
    Parameters
     ----------
    chi2_val: float
        the chi2
    dof: integer
        the number of degree of freedom
    """
    return 1 - chi2.cdf(chi2_val, dof)

def plot_ellipse(mu_2params, cov_2params, p_list=[0.68, 0.95], axis_name=None, fname=None):
    """
    plot an ellipse associated to a cov mat
    (to be checked, but I think this works)
    
    Parameters
     ----------
    mu_2params: 2d list
        the mean value of the two parameters
    cov_2params: 2d array of shape 2x2
        the covariance matrix of the 2 parameters
    p_list: list
        a list of the contour of probability we want to consider
    axis_name: list of str
        the name of the x and y axis
    """

    w, V = np.linalg.eig(cov_2params)
    angle = 180 / np.pi * np.arctan2(V[1,0], V[0,0])
    fig, ax = plt.subplots()
    for p in p_list:
        y = chi2.ppf(p, 2)
        width = 2 * np.sqrt(w[0] * y)
        height = 2 * np.sqrt(w[1] * y)
        el = Ellipse(xy=mu_2params,  width=width,  height=height,
                    angle=angle, edgecolor='black', lw=1, facecolor='none')
        
        ax.add_artist(el)
        
    sig0 = np.sqrt(cov_2params[0, 0])
    sig1 = np.sqrt(cov_2params[1, 1])

    ax.set_xlim(mu_2params[0] -  5 * sig0, mu_2params[0] + 5 * sig0)
    ax.set_ylim(mu_2params[1] -  5 * sig1, mu_2params[1] + 5 * sig1)
    if axis_name is not None:
        ax.set_xlabel(axis_name[0], fontsize=22)
        ax.set_ylabel(axis_name[1], fontsize=22)

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()
