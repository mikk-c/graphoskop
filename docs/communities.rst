Communities
===========

.. automodule:: graphoskop.communities
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv", "nodes.tsv")         # nodes.tsv has a "party" column
   gsk.modularity(net, "party")                             # partition from a node attribute
   gsk.modularity(net, {"alice": 0, "bob": 0, "carol": 1})  # or from labels by node id
   cliques = gsk.maximal_cliques(net)                       # node sets of the cliques of 3+ nodes
   gsk.bigclam_communities(net, attribute = "cover")        # overlapping communities (BigCLAM)
   gsk.demon_communities(net, attribute = "demon")          # overlapping communities from ego networks
   gsk.oslom_communities(net, attribute = "oslom")          # statistically significant (overlapping) communities
   gsk.block_model_communities(net, seed = 0)               # nested blockmodel: community, community_1, ...
   gsk.hierarchical_random_graph_communities(net, attribute = "hrg")   # consensus dendrogram: hrg, hrg_1, ...
   gsk.infomap_communities(net, attribute = "module")       # Infomap: module, module_1, ...
   gsk.label_propagation_communities(net, attribute = "lpa") # label propagation
   gsk.leiden_communities(net, attribute = "leiden")         # Leiden: maximum modularity
   gsk.hierarchical_link_communities(net)                   # link communities: edge and node attribute link_community
   gsk.conductance(net, "leiden")                           # quality of the communities found
   gsk.normalized_mutual_information(net, "party", "leiden")  # similarity of two partitions

Cliques
-------

.. autofunction:: graphoskop.maximal_cliques

Discovery
---------

BigCLAM
~~~~~~~

BigCLAM fits a community affiliation model, a nonnegative factorization in which every
community a pair of nodes shares raises the probability of an edge between them: communities
overlap, and their overlaps are denser than the rest, as in real networks.

.. code-block:: python

   res = gsk.bigclam_communities(net, seed = 0)                   # number of communities chosen by held-out pairs
   res["groups"], res["overlapping"], res["memberships"]          # affiliation of every node to every community
   gsk.bigclam_communities(net, groups = 50, attribute = "cover") # given number: much faster

.. autofunction:: graphoskop.bigclam_communities

DEMON
~~~~~

DEMON finds communities in the ego network of every node (its neighbors, without it), by label
propagation, and merges those similar enough: since every node is in the local communities of all
its neighbors, communities overlap.

.. code-block:: python

   res = gsk.demon_communities(net, seed = 0)                     # merge if sharing 25% of the smaller
   res["communities"], res["local"], res["overlapping"]
   gsk.demon_communities(net, epsilon = 0.75, min_size = 5)       # fewer merges, larger communities only

.. autofunction:: graphoskop.demon_communities

Hierarchical random graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~

Hierarchical random graphs, also used to predict links (:doc:`link_prediction`), give a hierarchy
of groups: the consensus of the dendrograms sampled in proportion to their likelihood, from the
groups of nodes found in most of them. The groups can be communities or disassortative sets.

.. code-block:: python

   res = gsk.hierarchical_random_graph_communities(net, seed = 0)   # community, community_1, ...: top level first
   res["groups"], res["communities"], res["log_likelihood"]
   gsk.hierarchical_random_graph_communities(net, burnin = 50000, consensus = 0.9)   # longer chains, surer groups

.. autofunction:: graphoskop.hierarchical_random_graph_communities

Infomap
~~~~~~~

Infomap finds the modules in which a random walker stays the longest, by the map equation:
hierarchically, following edge directions, and across the layers of multilayer networks; or
overlapping modules, with nodes coded in several of them.

.. code-block:: python

   res = gsk.infomap_communities(net, seed = 0)                   # community, community_1, ...: top level first
   res["codelength"], res["codelength_one_module"], res["modules"]
   res = gsk.infomap_communities(net, hierarchical = False, trials = 10, seed = 0)   # two levels, best of 10
   res = gsk.infomap_communities(citations, teleportation = 0.15)  # walker following edge directions
   res = gsk.infomap_communities(net, max_communities = 2, attribute = "cover")   # overlapping modules
   gsk.codelength(net, "cover")                                   # overlapping map equation

.. autofunction:: graphoskop.infomap_communities

Label propagation
~~~~~~~~~~~~~~~~~

Nodes repeatedly take the label most common among their neighbors, until labels settle; updating
the nodes by colors (or one at a time) keeps the two sides of bipartite structures from swapping
labels forever. With several labels per node (COPRA), communities overlap.

.. code-block:: python

   gsk.label_propagation_communities(net, seed = 0)                               # by colors, on the device
   gsk.label_propagation_communities(net, strategy = "asynchronous", weight = "weight", seed = 0)
   gsk.label_propagation_communities(net, max_communities = 3, attribute = "cover")   # overlapping

.. autofunction:: graphoskop.label_propagation_communities

Leiden
~~~~~~

The Leiden algorithm maximizes modularity (in the variant of the network: directed, bipartite,
multilayer) or the constant Potts model, refining the communities at every level so that they are
always connected.

.. code-block:: python

   res = gsk.leiden_communities(net, seed = 0)                    # maximum modularity
   res["quality"], res["communities"]
   gsk.leiden_communities(net, resolution = 2, attribute = "small")              # smaller communities
   gsk.leiden_communities(net, quality = "cpm", resolution = 0.05)               # internal density >= 0.05
   gsk.leiden_communities(net, initial = "community", attribute = "improved")    # from a partition

.. autofunction:: graphoskop.leiden_communities

Link communities
~~~~~~~~~~~~~~~~

Link communities group the links of the network rather than its nodes; a node is then in the
communities of all its links, so communities overlap. They are found by hierarchical clustering of
the links, cut at the largest partition density, or by partitioning the line graph (whose nodes
are the links) by modularity. Both write the community of every edge as an edge attribute and the
communities of every node as a node attribute.

.. code-block:: python

   res = gsk.hierarchical_link_communities(net)                   # Ahn, Bagrow and Lehmann
   res["communities"], res["partition_density"], res["threshold"]
   gsk.hierarchical_link_communities(net, weight = "weight", threshold = 0.3)   # cut at a similarity
   res = gsk.line_graph_communities(net, seed = 0)                # Evans and Lambiotte, random walk line graph
   net.get_edge_attribute("link_community", None)                 # community of every edge

.. autofunction:: graphoskop.hierarchical_link_communities

.. autofunction:: graphoskop.line_graph_communities

OSLOM
~~~~~

OSLOM keeps only the clusters that are statistically significant, compared with random networks
with the same degrees, through the order statistics of how strongly external nodes are linked to
them: clusters can overlap, and nodes significantly linked to no cluster are homeless.

.. code-block:: python

   res = gsk.oslom_communities(net, seed = 0)
   res["communities"], res["homeless"], res["overlapping"]
   gsk.oslom_communities(net, tolerance = 0.5, iterations = 2, attribute = "oslom")   # looser, faster

.. autofunction:: graphoskop.oslom_communities

Stochastic blockmodels
~~~~~~~~~~~~~~~~~~~~~~

Every stochastic blockmodel of the generators (see :doc:`generators`) finds communities when fitted
to a network: the groups under which the network is most likely. Models with more structure
describe more kinds of groups, at a higher cost in description length.

.. code-block:: python

   fit = gsk.block_model_communities(net, seed = 0)                                  # nested, degree-corrected
   fit = gsk.block_model_communities(net, gsk.degree_corrected_block_model, seed = 0)
   fit = gsk.block_model_communities(net, "stochastic_block_model", groups = 4, seed = 0)
   fit = gsk.block_model_communities(net, "planted_partition_graph", seed = 0)
   fit = gsk.block_model_communities(net, "mixed_membership_block_model", groups = 3, attribute = "cover", seed = 0)
   gsk.modularity(net, "cover")                                                      # overlapping modularity

.. autofunction:: graphoskop.block_model_communities

Quality of a partition
----------------------

Modularity compares the edges within communities with those expected at random; the codelength of
the map equation measures how well communities trap a random walker, in bits per step (normalized,
relative to no communities, to compare networks of different sizes). Conductance, internal density
and out-degree fraction score every community by its edges (inside it, or leaving it), either one
community or all of them (their mean, or a :class:`pandas.Series` by community); they accept
overlapping communities.

.. code-block:: python

   gsk.modularity(net, "community")
   gsk.codelength(net, "community")                         # bits per step of the walker
   gsk.codelength(net, "community", normalized = True)      # below 1: the communities compress the walk
   gsk.codelength(net, ["module", "module_1"])              # a hierarchy, top level first (as returned by Infomap)
   gsk.conductance(net, "community")                        # mean over the communities
   gsk.conductance(net, "community", community = 3)         # one community
   gsk.internal_density(net, "community", aggregate = "size")       # mean weighted by community size
   gsk.out_degree_fraction(net, "community", kind = "flake", aggregate = None)   # Series by community

.. autofunction:: graphoskop.codelength

.. autofunction:: graphoskop.conductance

.. autofunction:: graphoskop.internal_density

.. autofunction:: graphoskop.modularity

.. autofunction:: graphoskop.out_degree_fraction

Similarity of partitions
------------------------

Two partitions of the same nodes (for instance, communities found and planted ones, or the results
of two methods) are compared by information theory (normalized and adjusted mutual information) or
by pairs of nodes (adjusted Rand index). Normalized mutual information and the adjusted Rand index
also compare covers, with overlapping communities: by the overlapping normalized mutual information
(McDaid et al., or Lancichinetti et al.) and by the Omega index.

.. code-block:: python

   gsk.normalized_mutual_information(net, "community", "found")
   gsk.normalized_mutual_information(net, "community", "found", average = "max")
   gsk.adjusted_mutual_information(net, "community", "found")    # corrected for chance
   gsk.adjusted_rand_index(net, "community", "found")
   gsk.normalized_mutual_information(net, "cover", "found", overlap = "lfk")   # a node in several communities
   gsk.adjusted_rand_index(net, "cover", "found")                             # Omega index

.. autofunction:: graphoskop.adjusted_mutual_information

.. autofunction:: graphoskop.adjusted_rand_index

.. autofunction:: graphoskop.normalized_mutual_information
