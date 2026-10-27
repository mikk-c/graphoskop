Centrality
==========

.. automodule:: graphoskop.centrality
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv")
   P = gsk.all_pairs_shortest_path_lengths(net)             # computed once ...
   c = gsk.closeness_centrality(net, lengths = P)           # ... used by all
   h = gsk.harmonic_centrality(net, lengths = P)
   b = gsk.betweenness_centrality(net, lengths = P)
   eb = gsk.edge_betweenness_centrality(net, lengths = P)   # one value per stored edge
   gsk.average_shortest_path_length(net, lengths = P)

   # all components at once, comparable across components
   c = gsk.closeness_centrality(net, component = None, lin = True)
   h = gsk.harmonic_centrality(net, component = None)

   # hierarchy of a directed network: an edge i -> j pulls i one level above j
   d = gsk.load_network("dominance.tsv", directed = True)
   s = gsk.springrank(d, weight = "weight")

.. autofunction:: graphoskop.alpha_centrality

.. autofunction:: graphoskop.betweenness_centrality

.. autofunction:: graphoskop.centralization

.. autofunction:: graphoskop.closeness_centrality

.. autofunction:: graphoskop.edge_betweenness_centrality

.. autofunction:: graphoskop.eigenvector_centrality

.. autofunction:: graphoskop.flow_centrality

.. autofunction:: graphoskop.harmonic_centrality

.. autofunction:: graphoskop.hits

.. autofunction:: graphoskop.k_core_centrality

.. autofunction:: graphoskop.katz_centrality

.. autofunction:: graphoskop.node_edge_centrality

.. autofunction:: graphoskop.pagerank

.. autofunction:: graphoskop.reach_centrality

.. autofunction:: graphoskop.springrank
