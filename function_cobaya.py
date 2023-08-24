import numpy as np
from scipy.stats import chi2
import pylab as plt
import function_p1 as fp1

def cobaya_mcmc(z, data, inv_data_cov, z_c, pmin_list, pmax_list, chain_name, Rminus1_stop=0.002, Rminus1_cl_stop=0.04):
    
    """
    The cobaya mcmc algorithm
    
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
    pmin_list: list
        the minimal value allowed for each parameter
    pmax_list: list
        the maximal value allowed for each parameter
    chain_name: str
        the name of the chains datafile
    Rminus1_stop: float
    Rminus1_cl_stop: float
    """

    from cobaya.run import run

    def log_prob(a0, a1, a2, a3):
        params = [a0, a1, a2, a3]
        model = fp1.generate_model(z, params, z_c)
        res = data - model
        return -0.5 * res @ inv_data_cov @ res

    info = {
        "likelihood": {"my_like": log_prob},
        "params": {
            "a0": {"prior": {"min": pmin_list[0], "max": pmax_list[0]}, "latex": "a_{0}"},
            "a1": {"prior": {"min": pmin_list[1], "max": pmax_list[1]}, "latex": "a_{1}"},
            "a2": {"prior": {"min": pmin_list[2], "max": pmax_list[2]}, "latex": "a_{2}"},
            "a3": {"prior": {"min": pmin_list[3], "max": pmax_list[3]}, "latex": "a_{3}"},
        },
        "sampler": {
            "mcmc": {
                "max_tries": 10 ** 8,
                "Rminus1_stop": Rminus1_stop,
                "Rminus1_cl_stop": Rminus1_cl_stop,
            }
        },
        "output": f"{chain_name}",
        "force": True,
        "debug": False,
    }
    

    updated_info, sampler = run(info)

def plot_cobaya_chain(chain_name, params):

    """
    Plot the chains produced by cobaya
    Parameters
    ----------
    chain_name: str
        the name of the chains datafile
    params: list
        list of the parameters

    """

    from getdist.mcsamples import loadMCSamples
    import getdist.plots as gdplt

    samples = loadMCSamples( f"{chain_name}", settings = {"ignore_rows": 0.5})
    gdplot = gdplt.get_subplot_plotter()
    gdplot.triangle_plot(samples, params, filled = True, title_limit=1)
    plt.savefig(f"{chain_name}.png", dpi = 300)
    plt.clf()
    plt.close()

def four_d_plot(chain_name, params):

    """
    4d plot of the chains produced by cobaya
    Parameters
    ----------
    chain_name: str
        the name of the chains datafile
    params: list
        list of the parameters

    """

    from getdist.mcsamples import loadMCSamples
    import getdist.plots as gdplt

    samples = loadMCSamples( f"{chain_name}", settings = {"ignore_rows": 0.5})
    gdplot = gdplt.get_single_plotter()
    gdplot.plot_4d(samples, params, cmap='jet',
    alpha=0.4, shadow_alpha=0.05, shadow_color=True,
    max_scatter_points=6000,
    colorbar_args={'shrink': 0.6},
    animate=True,
    mp4_filename = f"{chain_name}.mp4")
