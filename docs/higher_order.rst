Higher-order networks
=====================

.. automodule:: graphoskop.higher_order
   :no-members:

*Atlas:* Section 7.3 (Many-to-Many Relationships); their analysis, Section 34.1 (High Order with
Simplicial Complexes), is in :doc:`high_order_dynamics`.

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_hypergraph([["a", "b", "c"], ["b", "d"]])
   net.number_of_nodes(), net.number_of_hyperedges()       # 4, 2
   net.set_hyperedge_attribute("hours", [2., 1.])
   g = gsk.clique_expansion(net)                           # a plain network of the nodes
   gsk.degree(g, "b")                                      # 3: a, c and d
   gsk.degree(net, "b")                                    # 2: the hyperedges of b
   gsk.laplacian(net)                                      # Zhou's hypergraph Laplacian
   gsk.local_clustering(net)                               # ValueError: not supported (yet)

   sc = gsk.load_simplicial_complex([["a", "b", "c"], ["a", "b"], ["c", "d"]])
   sc.number_of_facets()                                   # 2: {a, b} is a face of {a, b, c}

Loading
-------

.. autofunction:: graphoskop.load_hypergraph

.. autofunction:: graphoskop.load_simplicial_complex

Ids
---

.. autoclass:: graphoskop.Hyperedge

.. autoclass:: graphoskop.Simplex

Attributes and counts
---------------------

These methods of :class:`~graphoskop.Network` are for hypergraphs and simplicial complexes;
:meth:`~graphoskop.Network.get_node_attribute`, :meth:`~graphoskop.Network.set_node_attribute`
and :meth:`~graphoskop.Network.number_of_nodes` cover their nodes only. The simplices of one
dimension of a simplicial complex (its edges, its triangles, ...) are listed by
:meth:`~graphoskop.Network.simplices`, enumerated from the facets when first asked for and cached.

A simplicial complex has three kinds of attributes: node attributes, facet attributes (a value of
a facet as a whole, such as the date of a meeting) and **simplex attributes**, with a value on
every simplex of one dimension, faces included (a flow on every edge, the area of every
triangle). Simplex attributes are stored in one table per dimension, aligned with
:meth:`~graphoskop.Network.simplices`, and only for the dimensions that have them. Oriented
values, such as flows, refer to the reference orientation of each simplex (its nodes in
increasing internal index); a table of values given for other orientations is converted with
``oriented = True``.

.. code-block:: python

   import graphoskop as gsk
   import pandas as pd

   net = gsk.load_simplicial_complex([["a", "b", "c"], ["c", "d"]])
   flows = pd.DataFrame({"from": ["b", "d"], "to": ["a", "c"], "flow": [2., 5.]})
   net.set_simplex_attribute(flows, dimension = 1, oriented = True)
   net.get_simplex_attribute("flow")       # [-2, nan, nan, -5] over the edges ab, ac, bc, cd
   net.simplex_attributes                  # {"flow": 1}

.. automethod:: graphoskop.Network.get_facet_attribute

.. automethod:: graphoskop.Network.get_hyperedge_attribute

.. automethod:: graphoskop.Network.get_simplex_attribute

.. automethod:: graphoskop.Network.number_of_facets

.. automethod:: graphoskop.Network.number_of_hyperedges

.. automethod:: graphoskop.Network.number_of_simplices

.. automethod:: graphoskop.Network.remove_simplex_attribute

.. automethod:: graphoskop.Network.set_facet_attribute

.. automethod:: graphoskop.Network.set_hyperedge_attribute

.. automethod:: graphoskop.Network.set_simplex_attribute

.. autoproperty:: graphoskop.Network.simplex_attributes

.. automethod:: graphoskop.Network.simplices

Conversions
-----------

:func:`~graphoskop.clique_promotion` and :func:`~graphoskop.clique_expansion`, which make
higher-order networks from the cliques of a network and back, are in :doc:`transformations`.
Synthetic simplicial complexes come from :func:`~graphoskop.growing_manifold` and
:func:`~graphoskop.simplicial_configuration_model` (see :doc:`generators`).

.. autofunction:: graphoskop.incidence_network

.. _hypergraph-walks:

Random walks and Laplacians
---------------------------

*Atlas:* Laplacians, Section 8.4 (Laplacian); random walks, Chapter 11 (Random Walks).

On a hypergraph, :func:`~graphoskop.random_walk`, the Laplacian functions
(:doc:`laplacians`, :doc:`network_statistics`) and the node vector functions (:doc:`node_vector_analysis`) use the random
walk of Zhou, Huang and Schölkopf (2006): from a node, a hyperedge containing it with probability
proportional to its weight (``weight`` names a hyperedge attribute), then a node of that
hyperedge, uniformly. They run on the network of that walk, :func:`hypergraph_walk_network`,
whose Laplacian is the hypergraph Laplacian of Zhou et al.; :func:`~graphoskop.random_walk` also
reports the hyperedge of every step. The walks of Carletti et al. (2020) and of Chitra and Raphael
(2019, edge-dependent vertex weights) are other networks of :func:`hypergraph_walk_network`, to
pass to any function. On a simplicial complex, they use its 1-skeleton (see also
:ref:`hodge-laplacians`). Directed hypergraphs have random walks (from tails to heads) but no
Laplacians yet.

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_hypergraph({"m1": ["a", "b", "c"], "m2": ["b", "d"]})
   gsk.laplacian(net)                                       # Zhou's hypergraph Laplacian
   gsk.effective_resistance(net)
   gsk.random_walk(net, "a", 5, seed = 0)                   # walk, step, node, hyperedge
   g = gsk.hypergraph_walk_network(net, "carletti")
   gsk.laplacian(g, weight = "weight")                      # Carletti et al.'s walk

With large hyperedges the walk network is large: a hyperedge of :math:`s` nodes links
:math:`s (s - 1) / 2` pairs. Laplacian solvers, stationary distributions and sparse Fourier bases
then work on the incidence matrix instead, with the same results in memory linear in the
memberships; :func:`set_hypergraph_route` sets when.

.. autofunction:: graphoskop.hypergraph_walk_network

.. autofunction:: graphoskop.set_hypergraph_route

Supported functions
-------------------

Besides loading, the methods above and :func:`~graphoskop.incidence_network`, hypergraphs and
simplicial complexes are accepted by:

* :func:`~graphoskop.assortativity` (degree assortativity of Chodrow), on the hyperedges or
  facets;
* :func:`~graphoskop.centralization`, against a sunflower, with :func:`~graphoskop.degree` or
  :func:`~graphoskop.node_edge_centrality` (nonlinear node and hyperedge eigenvector centrality)
  (see :doc:`centrality`);
* :func:`~graphoskop.local_triangle_closure`, :func:`~graphoskop.global_triangle_closure` and
  :func:`~graphoskop.two_node_clustering` (see :doc:`network_statistics`);
* :func:`~graphoskop.average_degree`, :func:`~graphoskop.degree` and
  :func:`~graphoskop.degree_power_law`, on the degrees of the nodes (see
  :ref:`generalized-degrees`); :func:`~graphoskop.degree_centralization` is not, as its random
  baseline is a random graph (for a simplicial complex, apply it to
  :func:`~graphoskop.clique_expansion`, whose degrees are the same);
* :func:`~graphoskop.clique_expansion`;
* :func:`~graphoskop.connected_components_count` and :func:`~graphoskop.connected_components_nodes`
  (nodes only; with ``dimension``, the :math:`q`-connected components of simplicial complexes);
* :func:`~graphoskop.generalized_degree` and :func:`~graphoskop.incidence_number`, for simplicial
  complexes;
* :func:`~graphoskop.betti_numbers` and :func:`~graphoskop.euler_characteristic`, for simplicial
  complexes (see :ref:`homology`);
* :func:`~graphoskop.boundary_matrix`, :func:`~graphoskop.hodge_laplacian` and
  :func:`~graphoskop.simplicial_random_walk`, for simplicial complexes;
* :func:`~graphoskop.hodge_decomposition`, :func:`~graphoskop.simplicial_fourier_basis`,
  :func:`~graphoskop.simplicial_fourier_transform` and
  :func:`~graphoskop.inverse_simplicial_fourier_transform`, for simplicial complexes (see
  :ref:`topological-signals`);
* :func:`~graphoskop.kuramoto_model` (on the 1-skeleton), :func:`~graphoskop.simplicial_kuramoto_model`
  and :func:`~graphoskop.node_link_kuramoto_model`, for simplicial complexes (see
  :doc:`synchronization`);
* :func:`~graphoskop.growing_manifold` and :func:`~graphoskop.simplicial_configuration_model`,
  generators of simplicial complexes, and :func:`~graphoskop.hypergraph_configuration_model`, of
  hypergraphs, which also randomize them, and :func:`~graphoskop.randomization_test` with them
  (see :doc:`generators`);
* :func:`~graphoskop.higher_order_si_model`, :func:`~graphoskop.higher_order_sis_model` and
  :func:`~graphoskop.higher_order_sir_model`, the simplicial contagion model on complexes and
  undirected hypergraphs (see :ref:`higher-order-contagion`); the pairwise epidemic models do not take them
  (pass :func:`~graphoskop.clique_expansion` for those);
* :func:`~graphoskop.hgnn` (hypergraph neural network, node attributes) and
  :func:`~graphoskop.scnn` (simplicial neural network, simplex attributes; simplicial complexes),
  see :doc:`gnn`;
* :func:`~graphoskop.hyperedge_size`;
* :func:`~graphoskop.hypergraph_walk_network` and :func:`~graphoskop.set_hypergraph_route`;
* :func:`~graphoskop.is_pseudomanifold`, :func:`~graphoskop.is_pure` and
  :func:`~graphoskop.is_uniform` (see :ref:`higher-order-tests`);
* the Laplacian functions (:doc:`laplacians`, :doc:`network_statistics`), :func:`~graphoskop.random_walk`,
  :func:`~graphoskop.metropolis_hastings_random_walk` and the node vector functions
  (:doc:`node_vector_analysis`), on the network of the random walk of a hypergraph, or the
  1-skeleton of a complex (see :ref:`hypergraph-walks`); directed hypergraphs only in
  :func:`~graphoskop.random_walk`;
* :func:`~graphoskop.load_pickled_network`;
* :func:`~graphoskop.save_matrix`, which gives their incidence matrix: one row per node, one column
  per hyperedge or facet;
* :func:`~graphoskop.save_network`, which writes a membership table, a node table and a hyperedge
  (simplex) table, as given by :meth:`~graphoskop.Network.to_pandas_dataframes`.

All the other functions raise :class:`ValueError` for now; each says so in its documentation.
Analyses written for hypergraphs and simplicial complexes will be added function by function.
