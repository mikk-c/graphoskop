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
every node (configuration model).

.. code-block:: python

   g = gsk.gnm_random_graph(1000, 5000, seed = 1)                  # nodes 0 to 999, 5000 random edges
   b = gsk.gnm_random_graph((100, 300), 1000, directed = True)     # bipartite, edges in either direction
   g = gsk.gnp_random_graph(1000, 0.01, seed = 1)                  # each pair with probability 0.01
   rnd = gsk.gnp_random_graph(net = net)                           # p = density of net
   cm = gsk.configuration_model([3, 3, 2, 2, 1, 1], seed = 1)      # given degrees
   rnd = gsk.configuration_model(net = net)                        # degrees of net, random wiring

.. autofunction:: graphoskop.gnm_random_graph

.. autofunction:: graphoskop.gnp_random_graph

.. autofunction:: graphoskop.configuration_model

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

Nodes in communities, with edge probabilities depending on the communities of the two nodes.

.. code-block:: python

   pp = gsk.planted_partition_graph([100, 100, 50], 0.2, 0.01, seed = 1)   # p_in and p_out
   gsk.modularity(pp, "community")                                     # planted partition, node attribute
   rnd = gsk.planted_partition_graph(net = net)                        # communities of net ("community")
   rnd = gsk.planted_partition_graph(net = net, attribute = "party")   # densities within and between parties

   g = gsk.stochastic_block_model([50, 450], [[0.5, 0.1], [0.1, 0.01]])   # core-periphery
   rnd = gsk.stochastic_block_model(net = net)                         # density between every pair of communities

.. autofunction:: graphoskop.planted_partition_graph

.. autofunction:: graphoskop.stochastic_block_model
