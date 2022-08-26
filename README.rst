**************************
TD Ecole Euclid
**************************


First let's compute the analytical solution to the polynomial fitting problem

.. code:: shell

    python analytic_solution.py

Then let's check that our estimator is indeed unbiased and that the analytic covariance is correct

.. code:: shell

    python montecarlo.py

Let's code up our own mcmc and let's plot the histogram of the chains

.. code:: shell

    python run_mcmc.py


Finally, let's use cobaya mcmc to solve our problem (and getdist to plot the chains)

.. code:: shell

    python run_mcmc_cobaya.py

For part4 the idea is to adapt these functions for our new model and the new data set data_example_precise.txt
