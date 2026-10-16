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

Cliques
-------

.. autofunction:: graphoskop.maximal_cliques

Quality of a partition
----------------------

.. autofunction:: graphoskop.modularity
