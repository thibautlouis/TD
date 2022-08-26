"""
This script implements the Analytical solution to the polynomial model fit
This was made for the Euclid summer school tutorial on statistical methods
"""

import numpy as np
import function_p1 as fp1
import pylab as plt
from matplotlib import cm
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

os.makedirs("analytic_solution", exist_ok=True)
n_obs = len(z)
n_params = 4
corr = 0.1
n = 3
z_c = 0
dof = n_obs - n_params
param_name_vec = [r"$a_{0}$", r"$a_{1}$", r"$a_{2}$", r"$a_{3}$"]

### Solution to part1 question 3

P = fp1.get_ponting_matrix(n_obs, z, n_params, z_c)

### Solution to part1 question 4

data_cov = fp1.generate_covariance(z, sigma0, sigma1, n, corr)
data_sigma = np.sqrt(data_cov.diagonal())
print("************************")
print("check cov and data file err agree:", (data_sigma==err).all())
print("************************")
inv_data_cov = np.linalg.inv(data_cov)
param_test = [8, 3, 0.4, -0.02]
model_test= fp1.generate_model(z, param_test, z_c)
chi2_test = fp1.get_chi2(data, model_test, inv_data_cov)
p_val_test = fp1.get_p_value(chi2_test, dof)

print("************************")
print(r"param_test = [8, 3, 0.4, -0.02], $\chi^{2} = %0.4f, p= %0.4f" % (chi2_test, p_val_test))
print("************************")


### Solution to part1 question >= 7

param_cov = fp1.get_analytic_param_cov(P, inv_data_cov)
np.save(f"analytic_solution/param_cov{xtra_string}.npy", param_cov)
param_sigma, param_corr = fp1.get_sigma_and_corr(param_cov)
param_estimated = fp1.get_best_fit_params(data, inv_data_cov, param_cov, P)
model_estimated = fp1.generate_model(z, param_estimated, z_c)

chi2 = fp1.get_chi2(data, model_estimated, inv_data_cov)
p_val = fp1.get_p_value(chi2, dof)

print("************************")
print(r"max likelihood params, $\chi^{2} = %0.4f, p= %0.4f" % (chi2, p_val))
print("************************")

print("************************")
print("param correlation")
print(param_corr)
print("************************")

plt.imshow(param_corr, cmap=cm.seismic)
plt.colorbar()
plt.title("Corr")
plt.savefig(f"analytic_solution/correlation_matrix_{z_c}{xtra_string}.png", bbox_inches='tight')
plt.clf()
plt.close()

print("************************")
for count, (param, sigma, param_name) in enumerate(zip(param_estimated, param_sigma, param_name_vec)):
    print(param_name, "%.3f +/- %.3f" % (param, sigma))
print("************************")

plt.errorbar(z - z_c, data, data_sigma, fmt=".")
plt.errorbar(z - z_c, model_estimated, color="red", label="$\chi^{2}$ = %0.2f/%d" % (chi2, dof))
plt.xlabel(r"z",fontsize=16)
plt.ylabel("data",fontsize=16)
plt.legend(fontsize=16)
plt.savefig(f"analytic_solution/data_and_model_{z_c}{xtra_string}.png" , bbox_inches='tight')
plt.clf()
plt.close()

plt.errorbar(z - z_c, data, data_sigma, fmt=".")
plt.errorbar(z - z_c, model_estimated, color="red", label="$\chi^{2}$ = %0.2f/%d" % (chi2, dof))
color_vec = ["green", "orange", "blue", "gray"]
for pow, (param, param_name, color) in enumerate(zip(param_estimated, param_name_vec, color_vec)):
    plt.errorbar(z - z_c, param * (z - z_c) ** pow, color=color, label="%s= %0.3f" %(param_name, param))
plt.xlabel(r"z-zc",fontsize=16)
plt.ylabel("data",fontsize=16)
plt.legend(fontsize=16)
plt.savefig(f"analytic_solution/data_and_detailled_{z_c}{xtra_string}.png", bbox_inches='tight')
plt.clf()
plt.close()


##### Optionnal part ####
for i, name1 in enumerate(param_name_vec):
    for j, name2 in enumerate(param_name_vec):
        if i>=j: continue
        sub_mean = [param_estimated[i], param_estimated[j]]
        sub_param_cov = np.array([[param_cov[i,i], param_cov[i,j]], [param_cov[i,j], param_cov[j,j]]])
        fp1.plot_ellipse(sub_mean, sub_param_cov, axis_name=[name1, name2],
                         fname=f"analytic_solution/ellipse_{name1}_{name2}.png")


z_c_list = np.linspace(0, 20, 1000)
metric = []
for z_c in z_c_list:
    P = fp1.get_ponting_matrix(n_obs, z, n_params, z_c)
    param_cov = fp1.get_analytic_param_cov(P, inv_data_cov)
    param_sigma, param_corr = fp1.get_sigma_and_corr(param_cov)
    metric += [np.sum(np.abs(param_corr))]
    
plt.plot(z_c_list, metric)
plt.savefig(f"analytic_solution/metrics{xtra_string}.png", bbox_inches='tight')
plt.clf()
plt.close()
