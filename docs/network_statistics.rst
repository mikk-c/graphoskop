Simple properties
=================

.. automodule:: graphoskop.network_statistics

*Atlas:* Part III (Chapters 9 to 12).

Degree
------

*Atlas:* Chapter 9 (Degree).

Degree variants
~~~~~~~~~~~~~~~

*Atlas:* Section 9.1 (Degree Variants); hyperedge sizes, Section 7.3 (Many-to-Many Relationships).

.. autofunction:: graphoskop.degree

.. autofunction:: graphoskop.average_degree

.. autofunction:: graphoskop.hyperedge_size

Degree distributions and power laws
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Atlas:* Sections 9.2 (Degree Distributions) to 9.4 (Testing Power Laws).

.. autofunction:: graphoskop.degree_power_law

The degree centralization, :func:`~graphoskop.degree_centralization`, is with the other
centralizations in :doc:`centrality`.

Paths and walks
---------------

*Atlas:* Chapter 10 (Paths & Walks).

Reciprocity
~~~~~~~~~~~

*Atlas:* Section 10.3 (Reciprocity).

.. autofunction:: graphoskop.reciprocity

Connected components
~~~~~~~~~~~~~~~~~~~~

*Atlas:* Section 10.4 (Connected Components).

To keep the largest component of a network, or to split it into its strongly connected components,
use :func:`~graphoskop.largest_component` and :func:`~graphoskop.to_strongly_connected`
(:doc:`transformations`).

.. autofunction:: graphoskop.connected_components_count

.. autofunction:: graphoskop.connected_components_nodes

.. _random-walks:

Random walks
------------

*Atlas:* Chapter 11 (Random Walks).

Simulating random walks
~~~~~~~~~~~~~~~~~~~~~~~

*Atlas:* random walks to sample networks, Section 29.3 (Random Walk).

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv")
   gsk.random_walk(net, "alice", 20, walks = 5)            # walk, step, node
   gsk.metropolis_hastings_random_walk(net, "alice", 20)   # not biased towards hubs

.. autofunction:: graphoskop.random_walk

.. autofunction:: graphoskop.metropolis_hastings_random_walk

Stationary distribution
~~~~~~~~~~~~~~~~~~~~~~~

*Atlas:* Section 11.1 (Stationary Distribution).

.. autofunction:: graphoskop.stationary_distribution

Hitting time
~~~~~~~~~~~~

*Atlas:* Section 11.3 (Hitting Time).

.. autofunction:: graphoskop.hitting_time

.. autofunction:: graphoskop.fundamental_matrix

Effective resistance
~~~~~~~~~~~~~~~~~~~~

*Atlas:* Section 11.4 (Effective Resistance).

.. autofunction:: graphoskop.effective_resistance

Density
-------

*Atlas:* Chapter 12 (Density).

Network density
~~~~~~~~~~~~~~~

*Atlas:* Section 12.1 (Density & Real Networks).

.. autofunction:: graphoskop.density

Clustering coefficient
~~~~~~~~~~~~~~~~~~~~~~

*Atlas:* Section 12.2 (Clustering Coefficient).

The signed clustering coefficient is :func:`~graphoskop.signed_clustering` (:doc:`signed`).

.. autofunction:: graphoskop.average_clustering

.. autofunction:: graphoskop.global_clustering

.. autofunction:: graphoskop.local_clustering

Clustering in hypergraphs and simplicial complexes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Atlas:* Section 12.2 (Clustering Coefficient); hypergraphs and simplicial complexes, Section 7.3
(Many-to-Many Relationships).

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

.. autofunction:: graphoskop.global_triangle_closure

.. autofunction:: graphoskop.local_triangle_closure

.. autofunction:: graphoskop.two_node_clustering

Cliques
~~~~~~~

*Atlas:* Section 12.3 (Cliques).

.. autofunction:: graphoskop.maximal_cliques

.. _independent-sets:

Independent sets and graph coloring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Atlas:* Section 12.4 (Independent Sets).

.. automodule:: graphoskop.coloring
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv")
   colors = gsk.graph_coloring(net)              # one array of node ids per color, fewest colors
   mis = gsk.maximum_independent_set(net)        # node ids of a largest independent set

.. autofunction:: graphoskop.maximum_independent_set

.. autofunction:: graphoskop.graph_coloring
