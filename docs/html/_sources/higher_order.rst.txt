Higher-order networks
=====================

.. automodule:: graphoskop.higher_order
   :no-members:

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

.. _generalized-degrees:

Generalized degrees
-------------------

The degree of a node (:func:`~graphoskop.degree`) is its number of hyperedges in a hypergraph and
its number of edges in a simplicial complex. Simplicial complexes generalize it to every face:
the generalized degree :math:`k_{d,m}` of an :math:`m`-simplex counts the :math:`d`-simplices
containing it (:math:`k_{1,0}` is the degree), and the incidence number
:math:`k_{d,d-1} - 1` counts the :math:`d`-simplices sharing a :math:`(d - 1)`-face beyond the first.
Their values are over the :math:`m`-simplices, in the order of
:meth:`~graphoskop.Network.simplices`.

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_simplicial_complex([["a", "b", "c", "d"], ["c", "d", "e"]])
   gsk.degree(net, "c")                              # 4: the edges at c
   gsk.generalized_degree(net, 2, 0, "c")            # 4: the triangles at c
   gsk.generalized_degree(net, 2, 1, ["c", "d"])     # 3: the triangles on the edge {c, d}
   gsk.incidence_number(net, 2, ["c", "d"])          # 2

.. autofunction:: graphoskop.generalized_degree

.. autofunction:: graphoskop.incidence_number

.. _hodge-laplacians:

Hodge Laplacians
----------------

The boundary matrix :math:`B_k` maps every :math:`k`-simplex to its faces, with signs from the
order of their nodes; the Hodge Laplacian :math:`L_k = B_k^T B_k + B_{k+1} B_{k+1}^T` acts on
functions of the :math:`k`-simplices (flows on edges for :math:`k = 1`), and the dimension of its
kernel counts the :math:`k`-dimensional holes of the complex (its Betti number). The random walk
of Schaub et al. (2020) moves on oriented edges; its average dynamics is the normalized Hodge
1-Laplacian. Rows and columns follow :meth:`~graphoskop.Network.simplices`.

.. code-block:: python

   import graphoskop as gsk
   import torch

   hollow = gsk.load_simplicial_complex([["a", "b"], ["b", "c"], ["a", "c"]])
   L1 = gsk.hodge_laplacian(hollow, 1)
   int((torch.linalg.eigvalsh(L1).abs() < 1e-6).sum())      # 1: one hole
   gsk.simplicial_random_walk(hollow, ("a", "b"), 10, seed = 0)

.. autofunction:: graphoskop.boundary_matrix

.. autofunction:: graphoskop.hodge_laplacian

.. autofunction:: graphoskop.simplicial_random_walk

.. _homology:

Homology
--------

The Betti numbers count the holes of a simplicial complex in each dimension: :math:`\beta_0` its
connected components, :math:`\beta_1` its loops that bound no surface, :math:`\beta_2` its
voids. They come from exact ranks of the boundary matrices, computed by sparse column reduction in
compiled code; the Euler characteristic is their alternating sum, and the alternating sum of the
numbers of simplices.

.. code-block:: python

   import graphoskop as gsk
   import itertools

   sphere = gsk.load_simplicial_complex(itertools.combinations("abcd", 3))   # a hollow tetrahedron
   gsk.betti_numbers(sphere)               # [1, 0, 1]: one component, no loop, one void
   gsk.euler_characteristic(sphere)        # 2

.. autofunction:: graphoskop.betti_numbers

.. autofunction:: graphoskop.euler_characteristic

Ids
---

.. autoclass:: graphoskop.Hyperedge

.. autoclass:: graphoskop.Simplex

Loading
-------

.. autofunction:: graphoskop.load_hypergraph

.. autofunction:: graphoskop.load_simplicial_complex

.. _paths-with-memory:

Paths with memory
-----------------

.. automodule:: graphoskop.higher_order_dependencies
   :no-members:

Hypergraphs and simplicial complexes describe interactions among several nodes at once. Paths
carry another kind of higher order: memory, where the next step depends on the last few. Both
models here keep it in an ordinary directed network of states (``c|b``: at ``c``, coming from
``b``), whose transition probabilities are those of the paths, so that a random walk on it
reproduces their memory, and every walk-based analysis (PageRank, stationary distributions,
Infomap, Laplacians) follows the paths more closely than on the first-order network.

* :func:`memory_network` (Rosvall et al., 2014): a Markov chain of fixed order, every past of
  that length a state. With the teleportation of the paper, PageRank gives the flow through every
  place; Infomap with ``physical = "first_order_node"`` lets the states of a place share its
  codeword in a module, and places belong to several modules, as the flows through them do.
* :func:`higher_order_dependency_network` (Xu, Wickramarathne and Chawla, 2016): pasts of
  varying length, each added only where the data show that it changes the next step.

.. code-block:: python

   import graphoskop as gsk

   mem = gsk.memory_network("trajectories.txt")           # second order: states "k|j"
   flow = gsk.first_order_values(mem, gsk.pagerank(mem, weight = "weight", personalization = "count"))
   gsk.infomap_communities(mem, weight = "weight", physical = "first_order_node")
   gsk.first_order_values(mem, "community", how = set)    # overlapping modules of the places

   hon = gsk.higher_order_dependency_network("trajectories.txt", max_order = 5)
   # files in the authors' format, "id node node ...": add labeled = True
   hon.get_node_attribute("order")                       # 1 for the places, 2+ for their pasts
   pr = gsk.first_order_values(hon, gsk.pagerank(hon, weight = "weight"))   # PageRank of the places

   # a check against chance: walks without memory, about as many steps as the data
   # (set walks and their length to match); higher-order nodes found here are spurious
   fon = gsk.higher_order_dependency_network("trajectories.txt", max_order = 1)
   walks = gsk.random_walk(fon, list(fon.node_ids[0]), 100, walks = 50, weight = "weight")
   null = gsk.higher_order_dependency_network(walks, max_order = 5)

.. autofunction:: graphoskop.first_order_values

.. autofunction:: graphoskop.higher_order_dependency_network

.. autofunction:: graphoskop.memory_network

.. _hypergraph-walks:

Random walks and Laplacians
---------------------------

On a hypergraph, :func:`~graphoskop.random_walk`, the Laplacian functions
(:doc:`laplacians`) and the node vector functions (:doc:`node_vector_analysis`) use the random
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
  undirected hypergraphs (see :doc:`epidemics`); the pairwise epidemic models do not take them
  (pass :func:`~graphoskop.clique_expansion` for those);
* :func:`~graphoskop.hgnn` (hypergraph neural network, node attributes) and
  :func:`~graphoskop.scnn` (simplicial neural network, simplex attributes; simplicial complexes),
  see :doc:`gnn`;
* :func:`~graphoskop.hyperedge_size`;
* :func:`~graphoskop.hypergraph_walk_network` and :func:`~graphoskop.set_hypergraph_route`;
* :func:`~graphoskop.is_pseudomanifold`, :func:`~graphoskop.is_pure` and
  :func:`~graphoskop.is_uniform` (see :ref:`higher-order-tests`);
* the Laplacian functions (:doc:`laplacians`), :func:`~graphoskop.random_walk`,
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

.. _higher-order-tests:

Tests
-----

A hypergraph is uniform when all its hyperedges have the same size; a simplicial complex is pure
when all its facets have the same dimension, and a pseudomanifold when, moreover, every face of
dimension :math:`d - 1` is shared by at most two :math:`d`-simplices (incidence number 0 or 1) and
the :math:`d`-simplices are connected through these faces. Pseudomanifolds are the discrete
manifolds of network geometry with flavor; being a manifold in the topological sense needs more
(see :func:`~graphoskop.is_pseudomanifold`).

.. code-block:: python

   import graphoskop as gsk

   gsk.is_uniform(gsk.load_hypergraph([["a", "b", "c"], ["b", "c", "d"]]))      # True: 3-uniform
   disk = gsk.load_simplicial_complex([["a", "b", "c"], ["b", "c", "d"]])
   gsk.is_pure(disk), gsk.is_pseudomanifold(disk)                               # True, True
   gsk.is_pseudomanifold(disk, boundary = False)                                # False: it has a boundary

.. autofunction:: graphoskop.is_pseudomanifold

.. autofunction:: graphoskop.is_pure

.. autofunction:: graphoskop.is_uniform

.. _topological-signals:

Topological signals
-------------------

Signals on the :math:`k`-simplices of a complex (flows on its edges, values on its triangles) are
simplex attributes, or vectors aligned with :meth:`~graphoskop.Network.simplices`. The Hodge
decomposition splits such a signal into a gradient part (differences of a potential on the
:math:`(k - 1)`-simplices), a curl part (circulations around the :math:`(k + 1)`-simplices) and a
harmonic part (around the holes of the complex), and the simplicial Fourier transform
(Barbarossa and Sardellitti, 2020) expands it on the eigenvectors of the Hodge Laplacian, each
of one of these three kinds. Signals on the nodes are analyzed in :doc:`node_vector_analysis`;
oscillators on simplices, in :doc:`synchronization`.

.. code-block:: python

   import graphoskop as gsk
   import pandas as pd

   net = gsk.load_simplicial_complex([["a", "b", "c"], ["c", "d"], ["d", "e"], ["c", "e"]])
   flows = pd.DataFrame({"u": ["a", "b", "a", "c", "d", "e"], "v": ["b", "c", "c", "d", "e", "c"], "flow": [1., 1., 0., 2., 2., 2.]})
   net.set_simplex_attribute(flows, dimension = 1, oriented = True)
   gradient, curl, harmonic = gsk.hodge_decomposition(net, "flow")
   harmonic                                 # [0, 0, 0, 2, -2, 2]: the circulation c -> d -> e -> c
   freq, U, kinds = basis = gsk.simplicial_fourier_basis(net, 1)
   c = gsk.simplicial_fourier_transform(net, "flow", basis = basis)
   smooth = gsk.inverse_simplicial_fourier_transform(net, c * (freq < 2), basis = basis)

.. autofunction:: graphoskop.hodge_decomposition

.. autofunction:: graphoskop.inverse_simplicial_fourier_transform

.. autofunction:: graphoskop.simplicial_fourier_basis

.. autofunction:: graphoskop.simplicial_fourier_transform
