Statistical significance
========================

*Atlas:* Chapter 19 (Evaluating Statistical Significance).

Network shuffling
-----------------

*Atlas:* Section 19.1 (Network Shuffling).

Edge swaps randomize an existing network keeping the degree of every node, fully or partially. Like
the generators of :doc:`generators` and fitted exponential random graph models, they give null
models, against which a statistic of the network is tested.

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv")
   rnd = gsk.edge_swap(net)                                        # degrees of net exactly, 10 swaps per edge
   part = gsk.edge_swap(net, swaps = 100)                          # only a few edges moved

.. autofunction:: graphoskop.edge_swap

Testing against null models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any generator (or a fitted ERGM) is a null model for a statistic: the statistic of the network is
compared with its distribution over randomizations.

.. code-block:: python

   test = gsk.randomization_test(net, gsk.average_clustering, gsk.configuration_model, n = 200, seed = 1)
   test["observed"], test["mean"], test["z"], test["above"], test["below"], test["p_value"]
   test["distribution"]                                                # rounded values and their counts
   gsk.randomization_test(net, gsk.average_clustering, gsk.edge_swap, n = 50, swaps = 1000)

.. autofunction:: graphoskop.randomization_test

Exponential random graphs
-------------------------

.. automodule:: graphoskop.ergms
   :no-members:

*Atlas:* Section 19.2 (Exponential Random Graphs).

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
~~~~~~~

.. autofunction:: graphoskop.ergm

.. autoclass:: graphoskop.ERGMFit
   :members: simulate, gof, pseudo_r2

Simulation and goodness of fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   sims = fit.simulate(100, seed = 2)                                  # networks from the fitted model, with the nodes of net
   gof = fit.gof(100, seed = 3)                                        # as R's gof(): model, degree, espartners, distance
   gof["degree"]                                                       # obs, min, mean, max, MC p-value per degree
   r2, parts = fit.pseudo_r2()                                         # McFadden pseudo-R2 and its PMVD / LMG decomposition
   gsk.randomization_test(net, gsk.average_clustering, fit, n = 100)   # clustering of net against the model

Statistics
~~~~~~~~~~

.. autofunction:: graphoskop.ergm_statistics
