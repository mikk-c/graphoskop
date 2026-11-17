High-order dynamics
===================

*Atlas:* Chapter 34 (High-Order Dynamics).

Hypergraphs and simplicial complexes themselves (loading, attributes, conversions, and the functions
that accept them) are in :doc:`higher_order`.

High order with simplicial complexes
------------------------------------

*Atlas:* Section 34.1 (High Order with Simplicial Complexes).

.. _generalized-degrees:

Generalized degrees
~~~~~~~~~~~~~~~~~~~

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

.. _higher-order-tests:

Tests
~~~~~

*Atlas:* manifolds, Section 34.1 (High Order with Simplicial Complexes).

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

.. _hodge-laplacians:

Hodge Laplacians
~~~~~~~~~~~~~~~~

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
~~~~~~~~

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

.. _topological-signals:

Topological signals
~~~~~~~~~~~~~~~~~~~

*Atlas:* signals on the nodes and their graph Fourier transform, Section 47.4 (Graph Fourier
Transform).

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

.. _higher-order-contagion:

Higher-order contagion
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphoskop.epidemics.higher_order_contagion
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_simplicial_complex("groups.tsv")                     # e.g. cliques of contacts
   sis = gsk.higher_order_sis_model(net, [0.01, 0.1], 0.05, seeds = 0.2)   # beta_1 (edges), beta_2 (triangles)

   # the rescaled infectivities of Iacopini et al.: lambda = beta_1 <k> / mu, lambda_D = beta_2 <k_D> / mu
   mu, k = 0.05, gsk.average_degree(net)
   kd = 3 * net.number_of_simplices(2) / net.number_of_nodes()          # mean triangles per node
   beta = [0.8 * mu / k, 2.5 * mu / kd]                                 # lambda = 0.8, lambda_D = 2.5: bistable
   gsk.higher_order_sis_model(net, beta, mu, seeds = 0.01, window = 0)["I"].iloc[-1]   # dies out
   gsk.higher_order_sis_model(net, beta, mu, seeds = 0.4)["I"].iloc[-1]    # endemic

   hg = gsk.load_hypergraph("meetings.tsv")                             # hyperedges of any size
   sir = gsk.higher_order_sir_model(hg, 0.1, 0.1, threshold = 0.5, seeds = 5)   # half of a group infects

On a random simplicial complex with the parameters of Iacopini et al. (2,000 nodes, mean degree
20, 6 triangles per node, :math:`\lambda_\Delta = 2.5`, ``mu = 0.05``), 20 runs of the SIS model
take 2.4 s for four settings on two CPU threads, and reproduce the bistable region of the mean
field: at :math:`\lambda = 0.8`, the epidemic dies out from 1% of seeds and is endemic from 40%
(46.5% of infected nodes, against 52.9% in the homogeneous mean field, which neglects the
correlations between neighbors and the spread of degrees); at :math:`\lambda = 0.5`, below the critical point :math:`2 \sqrt{\lambda_\Delta} -
\lambda_\Delta = 0.66`, it dies out from 40%. Pairs are counted as in pairwise contagion: with
edges only, the model takes about as long as :func:`~graphoskop.complex_sis_model` with
``"classical"`` reinforcement (9.4 s against 8.0 s, 20 runs on 100,000 nodes and 900,000 edges).
With 100,000 nodes, 900,000 edges and 200,000 triangles, 100 runs of an endemic SIS model (87% of
the nodes infected, 214 steps to the plateau) take 72 s.

Higher-order SI model
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: graphoskop.higher_order_si_model

Higher-order SIS model
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: graphoskop.higher_order_sis_model

Higher-order SIR model
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: graphoskop.higher_order_sir_model

Synchronization
~~~~~~~~~~~~~~~

Kuramoto oscillators on the nodes, links and simplices of simplicial complexes are in
:doc:`synchronization`.

.. _paths-with-memory:

Paths with memory
-----------------

.. automodule:: graphoskop.higher_order_dependencies
   :no-members:

*Atlas:* Section 34.2 (Embedding Memory into the Structure).

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
