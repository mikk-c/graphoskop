Transformations
===============

.. automodule:: graphoskop.transformations
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("trust.tsv", directed = True)          # columns weight, sign
   und = gsk.to_undirected(net)                                    # weights summed: A + A^T
   und = gsk.to_undirected(net, combine = "max", disagree = "positive")
   core = gsk.remove_isolates(net)                                 # nodes without edges dropped
   clean = gsk.remove_isolates(gsk.remove_self_loops(net))         # no self loops, then no isolates
   simple = gsk.remove_parallel_edges(net)                         # one edge per pair (per direction)
   ego = gsk.subgraph(net, ["alice", "bob", "carol"])              # induced subgraph
   scc = gsk.to_strongly_connected(net)                            # one component per strongly connected component
   rev = gsk.flip_edge_direction(net)                              # all edges reversed: A^T
   rev = gsk.flip_edge_direction(net, ("alice", "bob"))            # only alice -> bob, now bob -> alice

Nodes and edges are added or removed one at a time, from any iterable, or from a
:class:`pandas.DataFrame` table. Missing attributes are NaN, except edge weights, which default to 1.

.. code-block:: python

   import pandas as pd

   net = gsk.add_nodes(net, "dave")                                # node attributes NaN
   net = gsk.add_nodes(net, pd.DataFrame({"id": ["erin"], "age": [41]}))
   net = gsk.add_edges(net, [("alice", "dave"), ("dave", "frank")])   # weight 1, frank added
   net = gsk.add_edges(net, pd.DataFrame({"source": ["bob"], "target": ["erin"], "weight": [2.5]}))
   net = gsk.remove_edges(net, ("alice", "bob"))                   # all edges between them
   net = gsk.remove_nodes(net, ["carol", "dave"])                  # with their edges

See also :func:`~graphoskop.layer_network` and :func:`~graphoskop.aggregate_network` for multilayer
networks, and :func:`~graphoskop.signed_network`.

.. autofunction:: graphoskop.flip_edge_direction

.. autofunction:: graphoskop.remove_isolates

.. autofunction:: graphoskop.remove_parallel_edges

.. autofunction:: graphoskop.remove_self_loops

.. autofunction:: graphoskop.subgraph

.. autofunction:: graphoskop.to_strongly_connected

.. autofunction:: graphoskop.to_undirected

Adding and removing nodes and edges
-----------------------------------

.. autofunction:: graphoskop.add_edges

.. autofunction:: graphoskop.add_nodes

.. autofunction:: graphoskop.remove_edges

.. autofunction:: graphoskop.remove_nodes

Collapsing nodes
----------------

:func:`quotient_network` collapses sets of nodes into single nodes, merging the edges between them;
:func:`network_condensation` collapses the strongly connected components of a directed network,
giving a directed acyclic graph.

.. code-block:: python

   blocks = gsk.quotient_network(net, [{"alice", "bob"}, {"carol", "dave"}])   # nodes 0 and 1
   blocks = gsk.quotient_network(net, [{"alice", "bob"}], self_loops = False)
   dag = gsk.network_condensation(net)   # node i: i-th strongly connected component

.. autofunction:: graphoskop.network_condensation

.. autofunction:: graphoskop.quotient_network

Combining networks
------------------

Set operations on the edges of two networks with the same node ids: two edges are the same if they
join the same two nodes. Attributes come from the first network, completed by the second.

.. code-block:: python

   both = gsk.network_union(week1, week2)            # edges of either
   gone = gsk.network_difference(week1, week2)       # edges of week1 not in week2
   kept = gsk.network_intersection(week1, week2)     # edges of week1 also in week2

.. autofunction:: graphoskop.network_difference

.. autofunction:: graphoskop.network_intersection

.. autofunction:: graphoskop.network_union
