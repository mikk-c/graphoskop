Hierarchy
=========

.. automodule:: graphoskop.hierarchy
   :no-members:

*Atlas:* Chapter 33 (Hierarchies); drawing hierarchies, Section 33.6 (Drawing Hierarchies), with
:func:`~graphoskop.hierarchical_layout` (:doc:`layouts`).

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv", directed = True)
   gsk.global_reach_centrality(net)
   gsk.flow_hierarchy(net)                    # fraction of edges in no cycle
   gsk.flow_hierarchy(net, by = "nodes")      # fraction of nodes in no cycle
   gsk.arborescence_hierarchy(net)            # fraction of edges left in the arborescence
   score, forest = gsk.arborescence_hierarchy(net, arborescence = True)
   h, levels = gsk.agony_hierarchy(net, ranks = True)   # 1 - agony / number of edges, levels

   # hierarchy of a directed network: an edge i -> j pulls i one level above j
   d = gsk.load_network("dominance.tsv", directed = True)
   s = gsk.springrank(d, weight = "weight")

SpringRank
----------

*Atlas:* Section 33.1 (Types of Hierarchies).

.. autofunction:: graphoskop.springrank

Flow hierarchy
--------------

*Atlas:* Section 33.2 (Cycles).

.. autofunction:: graphoskop.flow_hierarchy

Global reach centrality
-----------------------

*Atlas:* Section 33.3 (Global Reach Centrality).

.. autofunction:: graphoskop.global_reach_centrality

Arborescences
-------------

*Atlas:* Section 33.4 (Arborescences).

.. autofunction:: graphoskop.arborescence_hierarchy

Agony
-----

*Atlas:* Section 33.5 (Agony).

.. autofunction:: graphoskop.agony_hierarchy
