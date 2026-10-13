Exploration
===========

.. automodule:: graphoskop.exploration
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv")
   gsk.bfs(net, "alice")                     # node, depth, parent, in visit order
   gsk.bfs(net, "alice", max_depth = 2)      # neighborhood up to two edges away
   gsk.dfs(net, "alice")

.. autofunction:: graphoskop.bfs

.. autofunction:: graphoskop.dfs
