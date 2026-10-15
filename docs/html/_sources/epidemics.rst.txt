Epidemics
=========

.. automodule:: graphoskop.epidemics
   :no-members:

Simple contagion
----------------

.. automodule:: graphoskop.epidemics.simple_contagion
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("contacts.tsv")
   si = gsk.si_model(net, beta = 0.1, seeds = 5, runs = 1000, seed = 1)   # random seeds, component 0
   si.loc[10]                                                       # S, I and their intervals at step 10
   sis = gsk.sis_model(net, beta = 0.1, mu = 0.05, seeds = 5)       # stops at the endemic plateau
   sir = gsk.sir_model(net, beta = 0.2, mu = 0.1, seeds = 5)
   sir[["S", "I", "R"]].plot()

   k = gsk.degree(net, distribution = None, component = 0)
   sir = gsk.sir_model(net, 0.2, 0.1, 5, ranking = k)               # the 5 hubs
   sir = gsk.sir_model(net, 0.2, 0.1, 5, ranking = k, selection = "weighted")   # drawn proportionally
   sir = gsk.sir_model(net, 0.2, 0.1, 0.01, ranking = "age")        # 1% of the nodes: the oldest

   # one step at a time: the states are stored in the qualitative node attribute "state"
   gsk.sir_model(net, 0.2, 0.1, 5, runs = 1, max_steps = 0)         # seeds infected, step 0
   net.qualitative_node_attributes                                  # [..., "state"]
   for step in range(20):
      gsk.sir_model(net, 0.2, 0.1, runs = 1, max_steps = 1, resume = True)
      states = net.get_node_attribute("state")                     # "S", "I" or "R" for each node

   # immunization: 5% of the nodes, never infected ("V" in the stored states)
   sir = gsk.sir_model(net, 0.2, 0.1, 5, immunize = 0.05)                                 # at random
   sir = gsk.sir_model(net, 0.2, 0.1, 5, immunize = 0.05, strategy = "friend_of_random")  # acquaintances
   sir = gsk.sir_model(net, 0.2, 0.1, 5, immunize = 0.05, strategy = k)                   # the hubs
   sir = gsk.sir_model(net, 0.2, 0.1, 5, immunize = ["alice", "bob"])                     # these nodes
   # vaccinating in the middle of an epidemic, one step at a time
   gsk.sir_model(net, 0.2, 0.1, runs = 1, max_steps = 1, resume = True, immunize = 10, strategy = "friend_of_random")

   net = gsk.load_network("contacts.tsv", device = "cuda")
   sir = gsk.sir_model(net, 0.2, 0.1, 5)                           # runs batched on the GPU

Performance
~~~~~~~~~~~

With ``engine = "numba"``, each CPU thread simulates whole runs and only touches the nodes that
can change and the edges of the nodes that changed, and runs scale linearly with the number of
threads. With ``engine = "torch"``, the runs of a batch are simulated together on the device of
the network; it is the default on a GPU, where it is the faster engine. Times for 100 runs on
random graphs with ten times as many edges as nodes, ``beta = 0.1`` (SIR: ``mu = 0.05``, SIS:
``mu = 0.2``), on two CPU threads:

=================  =====  =================  ==========  ==========
Network            Model  Steps              Numba       Torch, CPU
=================  =====  =================  ==========  ==========
100,000 nodes      SI     171                1.5 s       17 s
100,000 nodes      SIR    378                4.2 s
100,000 nodes      SIS    228 (plateau)      32 s
100,000 nodes      SIS    300 (no plateau)   50 s
1,000,000 nodes    SI     192                26 s
1,000,000 nodes    SIR    379                93 s
=================  =====  =================  ==========  ==========

On a workstation with a GPU, with 1,000,000 nodes and 5,000,000 edges, 100 runs took:

=====================================  ===================  ===================
Model                                  Numba (CPU threads)  Torch (GPU)
=====================================  ===================  ===================
SI, ``beta = 0.1``                     3.4 s                2.0 s
SIR, ``beta = mu = 0.01``              16.9 s               4.4 s
=====================================  ===================  ===================

Endemic SIS is the most expensive case: a large fraction of the nodes is infected at every step,
and each of them draws its recovery at every step. Stopping the runs at the plateau (the default)
avoids simulating the endemic state longer than needed to estimate its level: on 20,000 nodes,
it gave the same endemic level as 3,000 steps without the test, 9 to 15 times faster.

SI model
~~~~~~~~

.. autofunction:: graphoskop.si_model

SIS model
~~~~~~~~~

.. autofunction:: graphoskop.sis_model

SIR model
~~~~~~~~~

.. autofunction:: graphoskop.sir_model

Complex contagion
-----------------

.. automodule:: graphoskop.epidemics.complex_contagion
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("contacts.tsv")
   si = gsk.complex_si_model(net, 0.02, "classical", seeds = 5)       # one coin flip per infected neighbor
   si = gsk.complex_si_model(net, 1, "threshold", 2, seeds = 0.05)    # Granovetter: at least 2 infected neighbors
   si = gsk.complex_si_model(net, 1, "cascade", 0.18, seeds = 1)      # Watts: at least 18% of the neighbors
   sir = gsk.complex_sir_model(net, 0.5, 0.1, "threshold", 2, seeds = 0.05)
   sis = gsk.complex_sis_model(net, 0.05, 0.2, "classical", seeds = 10)

   # heterogeneous thresholds, one per node, from a numeric node attribute
   import numpy as np
   net.set_node_attribute("phi", np.random.default_rng(1).uniform(0.1, 0.3, net.number_of_nodes(0)))
   si = gsk.complex_si_model(net, 1, "cascade", "phi", seeds = 5)

   # independent and decreasing cascades (Kempe, Kleinberg and Tardos): one attempt per edge
   si = gsk.complex_si_model(net, 0.1, "independent", seeds = 5)      # the same probability on every edge
   net.set_edge_attribute("p", np.random.default_rng(2).uniform(0, 0.3, net.components[0].edge_index.shape[1]))
   si = gsk.complex_si_model(net, "p", "independent", seeds = 5)      # one probability per edge (and direction)
   P = np.random.default_rng(3).uniform(0, 0.3, (net.number_of_nodes(0),) * 2)
   si = gsk.complex_si_model(net, P, "decreasing", decay = 0.5, seeds = 5)   # table: origins by targets

   # who infected whom in the last run: a forest, one tree per seed
   si, forest = gsk.complex_si_model(net, 0.2, "independent", seeds = 3, forest = True)
   edges, nodes = forest.to_pandas_dataframes()                     # source infected target at step; node attributes
   trees = gsk.connected_components_nodes(forest)                   # the nodes reached from each seed

On 100,000 nodes and 500,000 edges, 100 runs on two CPU threads with the Numba engine take
about as long as simple contagion: 3.6 s for the classical SI model with ``beta = 0.02``
(486 steps), 1.7 s for the threshold model with ``k = 2`` and 1% of seeds, 1.2 s for the cascade
model with ``phi = 0.1``, 3.0 s for the classical SIR model with ``beta = 0.05``, ``mu = 0.1``,
1.8 s for the independent cascade with ``p = 0.2`` (2.2 s with probabilities from an edge
attribute), 1.3 s for the decreasing cascade with ``p = 0.3`` and ``decay = 0.5`` (10 seeds).

Complex SI model
~~~~~~~~~~~~~~~~

.. autofunction:: graphoskop.complex_si_model

Complex SIS model
~~~~~~~~~~~~~~~~~

.. autofunction:: graphoskop.complex_sis_model

Complex SIR model
~~~~~~~~~~~~~~~~~

.. autofunction:: graphoskop.complex_sir_model
