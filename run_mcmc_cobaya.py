"""
This script show how to use cobaya for recovering parameters,
This was made for the Euclid summer school tutorial on statistical methods
"""


import numpy as np
import function_p1 as fp1
import function_cobaya as fco
import os

precise = False

if precise == True:
    z, data, err = np.loadtxt("data/data_example_precise.txt", unpack=True)
    sigma0 = np.sqrt(3) / 5
    sigma1 = 1 / 5
    xtra_string = "_precise"

else:
    z, data, err = np.loadtxt("data/data_example.txt", unpack=True)
    sigma0 = np.sqrt(3)
    sigma1 = 1
    xtra_string = ""


corr = 0.1
n = 3
z_c = 0
param_name_vec = [r"$a_{0}$", r"$a_{1}$", r"$a_{2}$", r"$a_{3}$"]

data_cov = fp1.generate_covariance(z, sigma0, sigma1, n, corr)
inv_data_cov = np.linalg.inv(data_cov)

pmin_array = [-10, -10, -5, -1]
pmax_array = [100, 10, 5, 1]

chain_name = f"mcm_cobaya/my_chain{xtra_string}"
fco.cobaya_mcmc(z, data, inv_data_cov, z_c,
                pmin_array, pmax_array,chain_name)
                
                            
params = ["a0", "a1", "a2", "a3"]
fco.plot_cobaya_chain(chain_name, params=params)
fco.four_d_plot(chain_name, params)
