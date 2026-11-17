Centrality
==========

.. automodule:: graphoskop.centrality
   :no-members:

*Atlas:* Chapter 14 (Node Ranking); degree centrality, Chapter 9 (Degree), is
:func:`~graphoskop.degree` (:doc:`network_statistics`), and SpringRank, Section 33.1 (Types of
Hierarchies), is in :doc:`hierarchy`.

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

Closeness
---------

*Atlas:* Section 14.1 (Closeness).

.. autofunction:: graphoskop.closeness_centrality

Betweenness
-----------

*Atlas:* Section 14.2 (Betweenness), with flow centrality at its end.

.. autofunction:: graphoskop.betweenness_centrality

.. autofunction:: graphoskop.edge_betweenness_centrality

.. autofunction:: graphoskop.flow_centrality

Reach
-----

*Atlas:* Section 14.3 (Reach).

.. autofunction:: graphoskop.reach_centrality

Eigenvector
-----------

*Atlas:* Section 14.4 (Eigenvector); the eigenvector centrality of nodes and hyperedges extends it
to hypergraphs (:doc:`higher_order`).

.. autofunction:: graphoskop.eigenvector_centrality

.. autofunction:: graphoskop.katz_centrality

.. autofunction:: graphoskop.alpha_centrality

.. autofunction:: graphoskop.pagerank

.. autofunction:: graphoskop.node_edge_centrality

HITS
----

*Atlas:* Section 14.5 (HITS).

.. autofunction:: graphoskop.hits

Harmonic
--------

*Atlas:* Section 14.6 (Harmonic).

.. autofunction:: graphoskop.harmonic_centrality

k-core
------

*Atlas:* Section 14.7 (k-Core).

.. autofunction:: graphoskop.k_core_centrality

Centralization
--------------

*Atlas:* Section 14.8 (Centralization).

.. autofunction:: graphoskop.centralization

.. autofunction:: graphoskop.degree_centralization
