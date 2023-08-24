import numpy as np
import function_p1 as fp1


def acceptance(logp, logp_new):
    """
    This function decide if the step of the mcmc
    chain is accepted or not
    
    Parameters
     ----------
    logp: float
        current log probability
    logp_new: float
        proposed log probability
    """

    if logp_new > logp:
        return True
    else:
        draw = np.random.uniform(0, 1)
        return (draw < (np.exp(logp_new - logp)))

    
def stepper(last_point, cholensky_cov):
    """
    This function randomly draw a step from
    the proposal covariance matrix
    
    Parameters
     ----------
    last_point: list
        the current position in parameters space
    cholensky_cov: 2d array
        the cholensky decomposition of the proposal covariance matrix
    """
    
    step = fp1.generate_noise(cholensky_cov)
    
    return last_point + step
 

def log_prior(param, pmin_list, pmax_list):
    """
    the logarithm of the prior
    
    Parameters
     ----------
    param: list
        the current position in parameters space
    pmin_list: list
        the minimal value allowed for each parameter
    pmax_list: list
        the maximal value allowed for each parameter
    """

    def flat_logp(p, pmin, pmax):
        if (p < pmin) or (p > pmax):
            return -np.inf
        else:
            return 0
    
    log_prior = 0
    for count, p in enumerate(param):
        log_prior += flat_logp(p, pmin_list[count], pmax_list[count])
    return log_prior
    
def mcmc(z, data, inv_data_cov, z_c, param_init,  proposal_cov, pmin_list, pmax_list, n_steps):
    """
    The mcmc algorithm
    
    Parameters
     ----------
    z: 1d array
        the redshifts at which the model is constructed
    data: 1d array
        the data
    inv_data_cov: 2d array
        the inverse covariance matrix of the noise
    zc: integer
        the redshift pivot
    param_init: list
        a list with the initial value of the parameters
    proposal_cov: 2d array of shape (n_params, n_params)
        the proposal covariance matrix for the algorithm
    pmin_list: list
        the minimal value allowed for each parameter
    pmax_list: list
        the maximal value allowed for each parameter
    n_steps: integer
        the number of steps in the mcmc chains
    """

    
    def log_prob(param):
        """
        Return the log probability associated to the data and the model
        Parameters
          ----------
         param: list
             the current position in parameters space
         """
        model = fp1.generate_model(z, param, z_c)
        res = data - model
        return -0.5 * res @ inv_data_cov @ res


    L = np.linalg.cholesky(proposal_cov)

    accep_count = 0
    current_point = param_init
    chains_params = np.zeros((len(param_init), n_steps))
    chain_like = np.zeros(n_steps)

    for i in range(n_steps):
        
        current_like = log_prob(current_point)
        current_prior = log_prior(current_point, pmin_list, pmax_list)
        
        new_point = stepper(current_point, L)

        new_like = log_prob(new_point)
        new_prior = log_prior(new_point, pmin_list, pmax_list)
    
        if (acceptance(current_like + current_prior, new_like + new_prior)):
        
            current_point = new_point
            chains_params[:, i] = current_point
            chain_like[i] = new_like
            
            accep_count += 1
        else:
            chains_params[:, i] = current_point
            chain_like[i] = current_like
        
    print(f"{accep_count} steps accepted /{n_steps} steps total")
    return chain_like, chains_params

def gaussian(mean, std):
    """
    The 1d gaussian function
    
    Parameters
     ----------
    mean: float
        the mean of the distribution
    std: float
        the std of the distribution
    """

    x = np.linspace(mean - 5 * std, mean + 5 *  std, 1000)
    norm = 1 / np.sqrt(2 * np.pi * std ** 2)
    gauss = norm * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
    
    return x, gauss
