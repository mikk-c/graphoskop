Frequent subgraph mining
========================

.. automodule:: graphoskop.frequent_subgraphs
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   db = gsk.load_network("molecules.tsv", "atoms.tsv")         # one molecule per component
   patterns, support = gsk.transactional_graph_mining(db, 100, node_label = "element", edge_label = "bond")
   support.sort_values("support", ascending = False).head()   # most frequent patterns
   list(patterns.edges(3, labels = True, data = "bond"))       # edges of pattern 3
   top, support = gsk.transactional_graph_mining(db, 0.1, node_label = "element", maximal = True)

   ml = gsk.load_network("layers.tsv", multilayer = True)      # layers as edge labels
   patterns, support = gsk.transactional_graph_mining(ml, 5, edge_label = gsk.LAYER, max_edges = 4)

   net = gsk.load_network("citations.tsv", "papers.tsv", directed = True)   # a single network
   patterns, support = gsk.single_graph_mining(net, 500, node_label = "field", max_edges = 4)

   rules = gsk.graph_association_rules(net, patterns, support, transactional = False)
   rules[rules["confidence"] > 0.5].sort_values("lift", ascending = False)

.. autofunction:: graphoskop.graph_association_rules

.. autofunction:: graphoskop.single_graph_mining

.. autofunction:: graphoskop.transactional_graph_mining

Performance
-----------

On two CPU threads, with databases of random molecule-like graphs (15 to 30 nodes, a tree plus one
to three rings, four node labels and two edge labels): 300 graphs, support 20%, 251 frequent
patterns in 0.7 s (0.9 s for the 112 maximal ones); support 10%, 910 patterns in 1.5 s; 1000
graphs, support 10%, 831 patterns in 3.8 s (maximal: 5.3 s). The pure Python gSpan of the
``gspan-mining`` package finds the same patterns (it does not report single nodes) in 3.9 s, 7.8 s
and 22 s.

In a single network (:func:`~graphoskop.single_graph_mining`), with Barabási-Albert networks whose
nodes have one of three random labels: 10,000 nodes and 30,000 edges, MNI threshold 500, 169
frequent patterns (up to 5 edges) in 2.1 s, the 94 maximal ones in 4.6 s; 100,000 nodes and
300,000 edges, threshold 5,000, 75 patterns up to 3 edges in 3.6 s and 156 up to 4 edges in 10.5 s.

:func:`~graphoskop.graph_association_rules` matches patterns by their canonical codes: the 2,088
rules between the 831 patterns of the 1,000 molecule-like graphs take 0.25 s.
