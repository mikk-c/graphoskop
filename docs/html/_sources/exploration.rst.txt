Exploration
===========

.. automodule:: graphoskop.exploration
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv")
   gsk.bfs(net, "alice")                                   # node, depth, parent, in visit order
   gsk.bfs(net, "alice", max_depth = 2)                    # neighborhood up to two edges away
   gsk.bfs(net, ["alice", "bob"])                          # one search per source
   gsk.dfs(net, "alice")
   gsk.random_walk(net, "alice", 20, walks = 5)            # walk, step, node
   gsk.metropolis_hastings_random_walk(net, "alice", 20)   # not biased towards hubs

.. autofunction:: graphoskop.bfs

.. autofunction:: graphoskop.dfs

.. autofunction:: graphoskop.metropolis_hastings_random_walk

.. autofunction:: graphoskop.random_walk
