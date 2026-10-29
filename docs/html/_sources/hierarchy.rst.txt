Hierarchy
=========

.. automodule:: graphoskop.hierarchy
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv", directed = True)
   gsk.global_reach_centrality(net)
   gsk.flow_hierarchy(net)                    # fraction of edges in no cycle
   gsk.flow_hierarchy(net, by = "nodes")      # fraction of nodes in no cycle
   gsk.arborescence_hierarchy(net)            # fraction of edges left in the arborescence
   score, forest = gsk.arborescence_hierarchy(net, arborescence = True)
   h, levels = gsk.agony_hierarchy(net, ranks = True)   # 1 - agony / number of edges, levels

.. autofunction:: graphoskop.agony_hierarchy

.. autofunction:: graphoskop.arborescence_hierarchy

.. autofunction:: graphoskop.flow_hierarchy

.. autofunction:: graphoskop.global_reach_centrality
