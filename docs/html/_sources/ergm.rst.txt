Exponential random graph models
===============================

.. automodule:: graphoskop.ergms
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv", "nodes.tsv")                  # node attributes age (numeric), city (qualitative)
   gsk.ergm_statistics(net, "edges + triangle + nodematch('city')")  # g(y), as R's summary(net ~ ...)

   fit = gsk.ergm(net, "edges + nodecov('age') + nodefactor('city') + nodematch('city')")   # dyad independent: exact MLE
   fit.summary                                                       # estimates, standard errors, z, p
   fit.aic

   fit = gsk.ergm(net, "edges + nodematch('city') + gwesp(0.5, fixed = TRUE)", seed = 1)   # MCMC MLE
   fit.coef, fit.stderr, fit.converged
   fit.samples.mean() - fit.observed                                 # sampled statistics around the observed ones

Fitting
-------

.. autofunction:: graphoskop.ergm

.. autoclass:: graphoskop.ERGMFit
   :members: simulate, gof, pseudo_r2

Simulation and goodness of fit
------------------------------

.. code-block:: python

   sims = fit.simulate(100, seed = 2)                                  # networks from the fitted model, with the nodes of net
   gof = fit.gof(100, seed = 3)                                        # as R's gof(): model, degree, espartners, distance
   gof["degree"]                                                       # obs, min, mean, max, MC p-value per degree
   r2, parts = fit.pseudo_r2()                                         # McFadden pseudo-R2 and its PMVD / LMG decomposition
   gsk.randomization_test(net, gsk.average_clustering, fit, n = 100)   # clustering of net against the model

Statistics
----------

.. autofunction:: graphoskop.ergm_statistics
