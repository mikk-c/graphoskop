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
   :no-members:

Statistics
----------

.. autofunction:: graphoskop.ergm_statistics
