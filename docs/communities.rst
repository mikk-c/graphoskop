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
   gsk.block_model_communities(net, seed = 0)               # nested blockmodel: community, community_1, ...

Cliques
-------

.. autofunction:: graphoskop.maximal_cliques

Discovery
---------

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

.. autofunction:: graphoskop.modularity
