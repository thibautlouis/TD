"""
This script implements the montecarlo verification of our estimator,
This was made for the Euclid summer school tutorial on statistical methods
"""

import numpy as np
import pylab as plt
import time
import function_p1 as fp1
import os
from scipy.stats import chi2

np.random.seed(1)
os.makedirs("monte_carlo", exist_ok=True)

n_obs = 100
z_max = 20
sigma0 = np.sqrt(3)
sigma1 = 1
corr = 0.1
n_sims = 30000
n = 3
z_c = 0
param_vec = [9.719, 3.938, 0.779, -0.057]
n_params = len(param_vec)
param_name_vec = [r"$a_{0}$", r"$a_{1}$", r"$a_{2}$", r"$a_{3}$"]

z = np.linspace(0, z_max, n_obs)

model = fp1.generate_model(z, param_vec)
data_cov = fp1.generate_covariance(z, sigma0, sigma1, n, corr)
L = np.linalg.cholesky(data_cov)

data_sigma = np.sqrt(data_cov.diagonal())

inv_data_cov = np.linalg.inv(data_cov)
P = fp1.get_ponting_matrix(n_obs, z, n_params, z_c=0)
param_cov = fp1.get_analytic_param_cov(P, inv_data_cov)
param_sigma, param_corr = fp1.get_sigma_and_corr(param_cov)

dof = n_obs - n_params
params_list, chi2_list, pval_list = [], [], []

t = time.time()
for i in range(n_sims):
    data_err = fp1.generate_noise(L)
    data = model + data_err
    param_estimated = fp1.get_best_fit_params(data, inv_data_cov, param_cov, P)
    model_estimated = fp1.generate_model(z, param_estimated)
    my_chi2 = fp1.get_chi2(data, model_estimated, inv_data_cov)
    p_val = fp1.get_p_value(my_chi2, dof)

    params_list += [param_estimated]
    chi2_list += [my_chi2]
    pval_list += [p_val]

print("************************")
print(f"time for doing n_sims={n_sims}", time.time() - t)
print("************************")


mc_mean = np.mean(params_list, axis=0)
mc_cov = np.cov(params_list, y=params_list,  rowvar=False)[:n_params,:n_params]
mc_std, mc_corr = fp1.get_sigma_and_corr(mc_cov)

plt.hist(chi2_list, bins=80, density=True)

x = np.arange(50, 150, 0.1)
plt.plot(x, chi2.pdf(x, df=dof))

plt.savefig("monte_carlo/chi2_dist.png", bbox_inches='tight')
plt.clf()
plt.close()

plt.hist(pval_list, bins=80)
plt.savefig("monte_carlo/pval_dist.png", bbox_inches='tight')
plt.clf()
plt.close()

for i, (name, param) in enumerate(zip(param_name_vec, param_vec)):
    print("")
    print(name)
    print("true value", param)
    print("montecarlo mean", mc_mean[i])
    print("montecarlo sigma", mc_std[i])
    print("analytic sigma", param_sigma[i])
    print("")

