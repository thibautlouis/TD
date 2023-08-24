"""
This script implements our own mcmc for recovering parameters,
This was made for the Euclid summer school tutorial on statistical methods
"""

import numpy as np
import function_p1 as fp1
import function_p3 as fp3
import function_cobaya as fco
import pylab as plt
import os


os.makedirs("mcmc", exist_ok=True)

precise = False
param_max_like = {}

if precise == True:
    z, data, err = np.loadtxt("data/data_example_precise.txt", unpack=True)
    sigma0 = np.sqrt(3) / 5
    sigma1 = 1 / 5
    xtra_string = "_precise"
    param_max_like[0] = [8.828, 0.274]
    param_max_like[1] = [5.999, 0.329]
    param_max_like[2] = [0.355, 0.069]
    param_max_like[3] = [-0.041, 0.003]

else:
    z, data, err = np.loadtxt("data/data_example.txt", unpack=True)
    sigma0 = np.sqrt(3)
    sigma1 = 1
    xtra_string = ""
    param_max_like[0] = [9.719, 1.370]
    param_max_like[1] = [3.938, 1.647]
    param_max_like[2] = [0.779, 0.344]
    param_max_like[3] = [-0.057, 0.016]


n_obs = len(z)
n_params = 4
corr = 0.1
n = 3
z_c = 0
param_name_vec = [r"$a_{0}$", r"$a_{1}$", r"$a_{2}$", r"$a_{3}$"]

data_cov = fp1.generate_covariance(z, sigma0, sigma1, n, corr)
inv_data_cov = np.linalg.inv(data_cov)

pmin_array = [-10, -10, -5, -1]
pmax_array = [100, 10, 5, 1]

param_init = np.zeros(n_params)
proposal_cov = np.load(f"analytic_solution/param_cov{xtra_string}.npy")
proposal_cov *= (2.4) ** 2 / n_params
current_point = param_init
n_steps = 10000

chain_like, chains_params = fp3.mcmc(z, data, inv_data_cov, z_c, param_init,
                                     proposal_cov, pmin_array, pmax_array, n_steps)

for i, name in enumerate(param_name_vec):
    plt.plot(chains_params[i,:], label = name)
    plt.savefig(f"mcmc/{name}{xtra_string}.png", bbox_inches='tight')
    plt.clf()
    plt.close()
    
burn_in = 1000
plt.figure(figsize=(15,8))
for i in range(n_params): #change that ! use param name !
    plt.subplot(1, n_params, i+1)
    plt.hist(chains_params[i, burn_in:], bins = 100, density=True, alpha=0.3)
    mean, std = param_max_like[i]
    x, gauss = fp3.gaussian(mean, std)
    plt.plot(x, gauss, label = "maximisation analytique")
    plt.xlabel(r"$a_{%d}$" % i, fontsize = 22)
    plt.legend()
plt.savefig(f"mcmc/histo{xtra_string}.png", bbox_inches='tight')
plt.clf()
plt.close()
