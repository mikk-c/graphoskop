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

.. autofunction:: graphoskop.gnm_random_graph

.. autofunction:: graphoskop.gnp_random_graph

.. autofunction:: graphoskop.configuration_model

.. autofunction:: graphoskop.edge_swap

.. autofunction:: graphoskop.kronecker_graph

.. autofunction:: graphoskop.geometric_graph

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

.. autofunction:: graphoskop.link_selection_graph

.. autofunction:: graphoskop.copying_graph

.. autofunction:: graphoskop.holme_kim_graph

Stochastic blockmodels
----------------------

Nodes in communities, with edge probabilities depending on the communities of the two nodes;
in mixed membership blockmodels, nodes can be in several communities (see
:ref:`overlapping-communities`).

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

.. autofunction:: graphoskop.planted_partition_graph

.. autofunction:: graphoskop.stochastic_block_model

.. autofunction:: graphoskop.degree_corrected_block_model

.. autofunction:: graphoskop.mixed_membership_block_model

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
