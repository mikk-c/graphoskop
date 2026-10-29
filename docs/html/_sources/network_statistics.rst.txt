Network statistics
==================

.. automodule:: graphoskop.network_statistics

Assortativity
-------------

.. code-block:: python

   gsk.assortativity(net)              # degree assortativity
   gsk.assortativity(net, "age")       # numeric attribute: Pearson correlation over the edges
   gsk.assortativity(net, "city")      # qualitative attribute: Newman's categorical coefficient
   hg = gsk.load_hypergraph("meetings.tsv")
   gsk.assortativity(hg, kind = "top-2")   # Chodrow: degrees of the two best connected members of every hyperedge

.. autofunction:: graphoskop.assortativity

Average shortest path length
----------------------------

.. autofunction:: graphoskop.average_shortest_path_length

Clustering
----------

On hypergraphs and simplicial complexes, the clustering of the projection
(:func:`~graphoskop.clique_expansion`) is high by construction: every group of three nodes or more
is a clique. Two coefficients measure what the groups add: the triangle closure, the share of the
triangles of the projection whose three nodes are together in one hyperedge (in a complex, the
share of the triangles of the 1-skeleton that are filled), and the two-node clustering, how much
the hyperedges of two neighbors overlap.

.. code-block:: python

   hg = gsk.load_hypergraph("meetings.tsv")
   gsk.global_triangle_closure(hg)                  # share of closed triangles
   c = gsk.local_triangle_closure(hg)               # per node
   o = gsk.two_node_clustering(hg, kind = "union")  # overlap of the meetings of neighbors, per node
   test = gsk.randomization_test(hg, gsk.global_triangle_closure, gsk.hypergraph_configuration_model, n = 100)

.. autofunction:: graphoskop.average_clustering

.. autofunction:: graphoskop.global_clustering

.. autofunction:: graphoskop.global_triangle_closure

.. autofunction:: graphoskop.local_clustering

.. autofunction:: graphoskop.local_triangle_closure

.. autofunction:: graphoskop.two_node_clustering

Connected components
--------------------

.. autofunction:: graphoskop.connected_components_count

.. autofunction:: graphoskop.connected_components_nodes

Degree
------

.. autofunction:: graphoskop.average_degree

.. autofunction:: graphoskop.degree

.. autofunction:: graphoskop.degree_centralization

.. autofunction:: graphoskop.degree_power_law

.. autofunction:: graphoskop.hyperedge_size

Density
-------

.. autofunction:: graphoskop.density

Reciprocity
-----------

.. autofunction:: graphoskop.reciprocity
