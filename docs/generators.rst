Generators
==========

.. automodule:: graphoskop.generators
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv", "nodes.tsv")                # column weight; age, city
   rnd = gsk.gnm_random_graph(net = net, seed = 1)                 # same nodes, attributes and m
   rnd.get_edge_attribute("weight")                                # the weights of net, shuffled

   ml = gsk.load_network("multiplex.tsv", multilayer = True, coupling = "all")
   rnd = gsk.gnm_random_graph(net = ml, m = {"work": 100})         # each layer on its own; couplings kept

Broad degree graphs
-------------------

Growing networks whose new nodes link more to nodes with more links, so that a few hubs emerge
and degrees follow a power law: by knowing all degrees (preferential attachment), or locally,
through random links (link selection) or the links of random nodes (copying); with triangles
closed by new nodes, degrees stay broad and clustering stays high (Holme-Kim).

.. code-block:: python

   ba = gsk.barabasi_albert_graph(10000, 6, seed = 1)              # average degree 6: 3 edges per new node
   ls = gsk.link_selection_graph(10000, 6, seed = 1)               # each new node picks 3 random links
   cp = gsk.copying_graph(10000, 6, seed = 1)                      # each new node copies 3 links of random nodes
   hk = gsk.holme_kim_graph(10000, 6, 0.8, seed = 1)               # power law with clustering
   rnd = gsk.holme_kim_graph(net = net)                            # n, m and average clustering of net
   gsk.degree_power_law(ls)                                        # exponent about 3
   rnd = gsk.barabasi_albert_graph(net = net)                      # n and m of net

.. autofunction:: graphoskop.barabasi_albert_graph

.. autofunction:: graphoskop.copying_graph

.. autofunction:: graphoskop.holme_kim_graph

.. autofunction:: graphoskop.link_selection_graph

Clustered graphs
----------------

Many triangles: cliques (caves), or neighborhoods on a ring with a few random shortcuts (small
world).

.. code-block:: python

   c = gsk.caveman_graph(10, 5, connected = True)                  # 10 cliques of 5 nodes in a ring
   gsk.modularity(c, "cave")                                       # planted partition, node attribute
   rnd = gsk.caveman_graph(net = net, seed = 1)                    # nodes of net in random caves

   w = gsk.watts_strogatz_graph(1000, 6, 0.1, connected = True)    # small world, one component
   w = gsk.watts_strogatz_graph(1000, 6, 0.1, rewire = "add")     # Newman-Watts shortcuts
   rnd = gsk.watts_strogatz_graph(net = net, p = 0.1)              # n and m of net, p always given

.. autofunction:: graphoskop.caveman_graph

.. autofunction:: graphoskop.watts_strogatz_graph

Community benchmarks
--------------------

Graphs with planted communities, the ground truth against which community discovery algorithms
are tested: equal communities and degrees (Girvan-Newman), or broad degrees and community sizes,
with directed, weighted and overlapping variants (LFR). The mixing parameter ``mu``, the fraction
of the edges of a node leaving its communities, sets how hard the communities are to find.

.. code-block:: python

   gn = gsk.girvan_newman_graph(0.3, seed = 1)                         # 4 x 32 nodes, degree 16, mu 0.3
   rnd = gsk.girvan_newman_graph(net = net)                            # random equal communities, mu and degree of net

   lfr = gsk.lfr_benchmark_graph(1000, 20, 50, 0.3, minc = 20, maxc = 100, seed = 1)
   gsk.modularity(lfr, "community")                                    # quality of the planted communities
   lfr = gsk.lfr_benchmark_graph(1000, 20, 50, 0.3, directed = True, muw = 0.2, on = 100, om = 2)
   rnd = gsk.lfr_benchmark_graph(net = net, mu = 0.5)                  # degrees and communities of net, more mixed

.. autofunction:: graphoskop.girvan_newman_graph

.. autofunction:: graphoskop.lfr_benchmark_graph

Higher-order networks
---------------------

Simplicial complexes and hypergraphs (see :doc:`higher_order`): discrete manifolds grown simplex
by simplex (network geometry with flavor, with the closing move of Wu et al.), complexes with given
facets per node and facet dimensions (the simplicial configuration model of Young et al.), and
hypergraphs with given hyperedges per node and hyperedge sizes (the hypergraph configuration model
of Chodrow): null models for higher-order networks, which :func:`~graphoskop.randomization_test`
takes. Plain networks become complexes with :func:`~graphoskop.clique_promotion`.

.. code-block:: python

   disk = gsk.growing_manifold(1000, seed = 1)                         # triangles glued on free links
   net = gsk.growing_manifold(1000, d = 3, p = 0.3, seed = 1)          # tetrahedra, with closing moves
   sc = gsk.simplicial_configuration_model([2, 2, 2, 2, 1], {1: 3, 2: 1}, seed = 1)
   null = gsk.simplicial_configuration_model(net = net, seed = 2)      # facets per node and facet dimensions of net
   hg = gsk.hypergraph_configuration_model([2, 2, 1, 1, 1, 1], [3, 3, 2], seed = 1)
   meetings = gsk.load_hypergraph("meetings.tsv")
   null = gsk.hypergraph_configuration_model(net = meetings, seed = 2)  # same degrees and sizes, labels and attributes kept
   test = gsk.randomization_test(net, lambda g: gsk.betti_numbers(g)[1], gsk.simplicial_configuration_model, n = 100)   # holes against the null

.. autofunction:: graphoskop.growing_manifold

.. autofunction:: graphoskop.hypergraph_configuration_model

.. autofunction:: graphoskop.simplicial_configuration_model

Neural generators
-----------------

.. automodule:: graphoskop.neural_generators
   :no-members:

.. code-block:: python

   molecules = gsk.load_network("bonds.tsv")                  # one component per molecule
   fit = gsk.graphrnn(molecules, seed = 1)                    # train (long: see the performance below)
   fake = fit.simulate(seed = 2)[0]                           # as many generated molecules
   test = gsk.randomization_test(molecules, gsk.average_clustering, fit, n = 20)

GraphRNN needs long training: the paper trains for 96,000 steps of 32 graphs, which takes hours
even on a GPU. On two CPU threads, with a smaller model (``hidden = 64, layers = 2``), compared with
the training graphs and with random graphs of the same numbers of nodes and edges:

* 100 grids of 4 to 7 by 4 to 7 nodes: after 1,000 steps (111 s) the generated graphs are too
  large (38.1 nodes, against 31.9) and their degrees as far from the data as random graphs'
  (distance of the degree distributions 0.42); after 4,000 steps (417 s), 30.8 nodes, mean degree
  3.28 (data 3.29), degree distance 0.12, clustering 0.010 (data 0, random graphs 0.099);
* 100 graphs of two dense communities (8 to 15 nodes each, density 0.5, one edge between them),
  3,000 steps (270 s): 23.4 nodes (data 23.6), clustering 0.466 (data 0.479, random 0.247),
  modularity 0.400 (data 0.443, random 0.246); GraphRNN-S (110 s): 16.2 nodes, modularity 0.287;
* one network of 8 cliques of 10 nodes in a ring (single network mode), 3,000 steps (634 s): mean
  degree 8.98 (9.00), clustering 0.874 (0.936), 65 nodes (80).

With the paper's model (4 layers of 128) and only 1,500 steps, on grids of up to 12 by 12, nothing
was learned yet (graphs of 2 nodes): give it many more steps, preferably on a GPU. On paths (to
check the model), 1,000 steps give graphs of 19.5 nodes (data 20), 99% of them paths.

Memory does not grow with the size of the network (see :func:`graphoskop.graphrnn`): on single
networks of 6,000 nodes and 30,000 edges, a random geometric graph (``M`` = 338) and a preferential
attachment network (``M`` = 4,507), training takes about 3 GB, and on two CPU threads 13 and 22 seconds
per step (a GPU is much faster). Generating a graph that large takes about ``n * M`` small steps. There, the loss of a step is on 6% (``M``
= 338) or less of the nodes, which needs more steps: on the network of 8 cliques above, with the
loss forced on 9% of the nodes, 3,000 steps give mean degree 7.38, clustering 0.631 and 37.4 nodes
(against 8.98, 0.874 and 65 with the full loss; data 9.00, 0.936, 80).

.. autofunction:: graphoskop.graphrnn

.. autoclass:: graphoskop.GraphRNNFit
   :members: simulate

Random graphs
-------------

Edges placed uniformly at random, given their number or density (Erdős–Rényi), or the degree of
every node (configuration model, or edge swaps, which only randomize an existing network, fully
or partially); or with probabilities given by Kronecker powers of a small initiator matrix
(Kronecker graphs); or between nodes close in space (geometric graphs).

.. code-block:: python

   g = gsk.gnm_random_graph(1000, 5000, seed = 1)                  # nodes 0 to 999, 5000 random edges
   b = gsk.gnm_random_graph((100, 300), 1000, directed = True)     # bipartite, edges in either direction
   g = gsk.gnp_random_graph(1000, 0.01, seed = 1)                  # each pair with probability 0.01
   rnd = gsk.gnp_random_graph(net = net)                           # p = density of net
   cm = gsk.configuration_model([3, 3, 2, 2, 1, 1], seed = 1)      # given degrees
   rnd = gsk.configuration_model(net = net)                        # degrees of net, random wiring
   rnd = gsk.edge_swap(net)                                        # degrees of net exactly, 10 swaps per edge
   part = gsk.edge_swap(net, swaps = 100)                          # only a few edges moved
   kg = gsk.kronecker_graph([[0.9, 0.5], [0.5, 0.1]], n = 10000)   # stochastic Kronecker graph
   rnd = gsk.kronecker_graph(net = net)                            # 2 x 2 initiator fitted to net
   geo = gsk.geometric_graph(1000, r = 0.05)                       # pairs closer than 0.05 in the unit square
   geo = gsk.geometric_graph(1000, m = 5000, d = 3, torus = True)  # the 5000 closest pairs in a 3D torus

.. autofunction:: graphoskop.configuration_model

.. autofunction:: graphoskop.edge_swap

.. autofunction:: graphoskop.geometric_graph

.. autofunction:: graphoskop.gnm_random_graph

.. autofunction:: graphoskop.gnp_random_graph

.. autofunction:: graphoskop.kronecker_graph

Simple graphs
-------------

Simple fixed topologies, with nodes ``0`` to ``n - 1``: a chain (path), undirected or with its
edges in one direction, at random, towards or away from its middle node.

.. code-block:: python

   g = gsk.chain_graph(10)                                        # 0 - 1 - ... - 9
   g = gsk.chain_graph(10, directed = True)                       # sequential: 0 -> 1 -> ... -> 9
   g = gsk.chain_graph(10, directed = "random", seed = 1)         # each edge in a random direction
   g = gsk.chain_graph(10, directed = "converging")               # 0 -> ... -> 5 <- ... <- 9
   g = gsk.chain_graph(10, directed = "diverging")                # 0 <- ... <- 5 -> ... -> 9

.. autofunction:: graphoskop.chain_graph

Stochastic blockmodels
----------------------

Nodes in communities, with edge probabilities depending on the communities of the two nodes;
in mixed membership blockmodels, nodes can be in several communities (see
:ref:`overlapping-communities`); in the nested blockmodel, communities are themselves in groups,
at several levels (see :doc:`error_correction`).

.. code-block:: python

   pp = gsk.planted_partition_graph([100, 100, 50], 0.2, 0.01, seed = 1)   # p_in and p_out
   gsk.modularity(pp, "community")                                     # planted partition, node attribute
   rnd = gsk.planted_partition_graph(net = net)                        # communities of net ("community")
   rnd = gsk.planted_partition_graph(net = net, attribute = "party")   # densities within and between parties

   g = gsk.stochastic_block_model([50, 450], [[0.5, 0.1], [0.1, 0.01]])   # core-periphery
   rnd = gsk.stochastic_block_model(net = net)                         # density between every pair of communities

   g = gsk.degree_corrected_block_model([5, 4], [10, 1, 1, 1, 1, 6, 2, 2, 2], [[6, 1], [1, 5]])   # hubs in blocks
   rnd = gsk.degree_corrected_block_model(net = net)                   # degrees and block edges of net, on average

   B = [[0.3, 0.01], [0.01, 0.3]]
   g = gsk.mixed_membership_block_model(B, n = 200, alpha = 0.1)      # memberships from a Dirichlet
   g = gsk.mixed_membership_block_model(B, memberships = [[1, 0], [0.5, 0.5], [0, 1]])
   gsk.modularity(g, "community")                                      # overlapping modularity
   rnd = gsk.mixed_membership_block_model(net = net)                   # "community" can list several per node

   g = gsk.nested_block_model(1000, 5000, [32, 8, 2], seed = 1)        # community, community_1, community_2
   rnd = gsk.nested_block_model(net = net)                             # fitted hierarchy of net

.. autofunction:: graphoskop.degree_corrected_block_model

.. autofunction:: graphoskop.mixed_membership_block_model

.. autofunction:: graphoskop.nested_block_model

.. autofunction:: graphoskop.planted_partition_graph

.. autofunction:: graphoskop.stochastic_block_model

Testing against null models
---------------------------

Any generator (or a fitted ERGM) is a null model for a statistic: the statistic of the network is
compared with its distribution over randomizations.

.. code-block:: python

   test = gsk.randomization_test(net, gsk.average_clustering, gsk.configuration_model, n = 200, seed = 1)
   test["observed"], test["mean"], test["z"], test["above"], test["below"], test["p_value"]
   test["distribution"]                                                # rounded values and their counts
   gsk.randomization_test(net, gsk.average_clustering, gsk.edge_swap, n = 50, swaps = 1000)

.. autofunction:: graphoskop.randomization_test
