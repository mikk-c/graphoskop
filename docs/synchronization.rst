Synchronization
===============

.. automodule:: graphoskop.synchronization
   :no-members:

.. code-block:: python

   import graphoskop as gsk
   import numpy as np

   net = gsk.load_network("edges.tsv")
   curve = gsk.kuramoto_model(net, np.linspace(0, 1, 21), runs = 10, seed = 0)   # all at once
   curve.groupby("sigma")["order"].mean()                                         # the transition

   sc = gsk.simplicial_configuration_model(degrees = [6] * 200, dimensions = {2: 400}, seed = 1)
   up = np.linspace(0.05, 4, 16)
   edges = gsk.simplicial_kuramoto_model(sc, up, k = 1, omega = (2., 1.), seed = 0)   # oscillators on the edges
   loop = gsk.node_link_kuramoto_model(sc, np.r_[up, up[::-1]], model = "nlt", omega = (2., 1.), link_omega = (2., 1.), adiabatic = True, seed = 1)
   forward, backward = loop["order_nodes"][:16], loop["order_nodes"][16:]         # explosive: a hysteresis loop

Higher-order Kuramoto models
----------------------------

.. autofunction:: graphoskop.node_link_kuramoto_model

.. autofunction:: graphoskop.simplicial_kuramoto_model

Kuramoto model
--------------

.. autofunction:: graphoskop.kuramoto_model
