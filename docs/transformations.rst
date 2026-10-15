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

.. autofunction:: graphoskop.to_undirected

.. autofunction:: graphoskop.remove_isolates

.. autofunction:: graphoskop.remove_self_loops

.. autofunction:: graphoskop.remove_parallel_edges

.. autofunction:: graphoskop.subgraph

.. autofunction:: graphoskop.to_strongly_connected

Adding and removing nodes and edges
-----------------------------------

.. autofunction:: graphoskop.add_nodes

.. autofunction:: graphoskop.add_edges

.. autofunction:: graphoskop.remove_nodes

.. autofunction:: graphoskop.remove_edges
