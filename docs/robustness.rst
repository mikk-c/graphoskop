Robustness
==========

.. automodule:: graphoskop.robustness
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv")
   failures = gsk.node_removal(net, runs = 10, seed = 1)                 # random failures
   k = gsk.degree(net, distribution = None)
   attack = gsk.node_removal(net, strategy = k)                           # hubs first
   soft = gsk.node_removal(net, strategy = k, selection = "weighted")     # hubs more likely first
   friends = gsk.node_removal(net, strategy = "friend_of_random")         # acquaintance attack
   chosen = gsk.node_removal(net, strategy = ["alice", "bob"])            # these nodes, in order
   apl = gsk.node_removal(net, strategy = k, quality = "apl", points = 20, samples = 500)
   degree = lambda g: gsk.degree(g, distribution = None)
   adaptive = gsk.node_removal(net, strategy = degree, recompute = True)  # largest current degree first

   cuts = gsk.edge_removal(net)                                           # random edge failures
   eb = gsk.edge_betweenness_centrality(net)
   cuts = gsk.edge_removal(net, strategy = eb, component = 0)            # highest betweenness first

   ax = failures.plot(x = "fraction", y = "lcc", label = "random")
   attack.plot(x = "fraction", y = "lcc", label = "hubs", ax = ax)

   # cascades: a failed node passes its load to its neighbors, overloaded neighbors fail next
   c = gsk.cascade_failures(net, failures = 5, margin = 0.2, runs = 10)   # random loads and capacities
   c = gsk.cascade_failures(net, k, failures = 1, strategy = k, margin = 0.5)   # load = degree, the hub fails
   c[["failed", "lcc"]].plot()                                            # by step
   ml = gsk.load_network("layers.tsv", multilayer = True, coupling = "all")
   c = gsk.cascade_failures(ml, failures = 5)                             # all copies of an entity fail together

Cascade failures
----------------

.. autofunction:: graphoskop.cascade_failures

Edge removal
------------

.. autofunction:: graphoskop.edge_removal

Node removal
------------

.. autofunction:: graphoskop.node_removal

Performance
-----------

On two CPU threads, 10 runs of random node failures with the largest component as quality take
0.7 s on a random network with 100,000 nodes and 500,000 edges, and 10 s with 1,000,000 nodes and
5,000,000 edges (every removal recorded internally). The average path length needs a full search
at every recorded point: 20 points on 5,000 nodes and 25,000 edges take 8 s exactly, 0.5 s with
``samples = 200`` (3.956 instead of 3.950 before any removal).

With ``recompute``, the measure is computed again on a rebuilt copy of the remaining network, so
the cost grows with the number of computations: an adaptive degree attack on 10,000 nodes and
50,000 edges takes 13 s with ``recompute = 100`` and 1.4 s with ``recompute = 1000``.

:func:`~graphoskop.cascade_failures` runs the runs in parallel and only touches the items that
receive load: 10 runs from 5 random failures take 0.9 s on 100,000 nodes and 500,000 edges, and
14 s on 1,000,000 nodes and 5,000,000 edges.
