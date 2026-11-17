Shortest paths
==============

.. automodule:: graphoskop.shortest_paths

*Atlas:* Chapter 13 (Shortest Paths).

.. _graph-exploration:

Graph exploration
-----------------

*Atlas:* Section 13.1 (Graph Exploration); breadth first search to sample networks, Section 29.2
(Topological Breadth First Search Variants).

.. automodule:: graphoskop.exploration
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv")
   gsk.bfs(net, "alice")                                   # node, depth, parent, in visit order
   gsk.bfs(net, "alice", max_depth = 2)                    # neighborhood up to two edges away
   gsk.bfs(net, ["alice", "bob"])                          # one search per source
   gsk.dfs(net, "alice")

.. autofunction:: graphoskop.bfs

.. autofunction:: graphoskop.dfs

Finding shortest paths
----------------------

*Atlas:* Section 13.2 (Finding Shortest Paths).

.. autofunction:: graphoskop.shortest_path

.. autofunction:: graphoskop.shortest_path_length

.. autofunction:: graphoskop.single_source_shortest_paths

.. autofunction:: graphoskop.single_source_shortest_path_lengths

.. autofunction:: graphoskop.all_pairs_shortest_paths

.. autofunction:: graphoskop.all_pairs_shortest_path_lengths

Path length distribution
------------------------

*Atlas:* Section 13.3 (Path Length Distribution).

.. autofunction:: graphoskop.shortest_path_length_distribution

.. autofunction:: graphoskop.average_shortest_path_length

.. autofunction:: graphoskop.diameter

Centralities based on shortest paths are in :doc:`centrality`; they and
:func:`~graphoskop.average_shortest_path_length` accept the lengths computed by
:func:`~graphoskop.all_pairs_shortest_path_lengths`.

Spanning trees and filtered graphs
----------------------------------

*Atlas:* Section 13.4 (Spanning Trees & Other Filtered Graphs).

Spanning trees and maximally filtered graphs (:func:`~graphoskop.spanning_tree`,
:func:`~graphoskop.maximally_filtered_graph`, :func:`~graphoskop.triangulated_filtered_graph`) are
in :doc:`backboning`, since the Atlas also counts them among the backbones (Chapter 27).
