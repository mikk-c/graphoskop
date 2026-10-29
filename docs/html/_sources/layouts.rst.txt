Layouts
=======

.. automodule:: graphoskop.layouts

Six families of layouts are available: circular, grid and matrix layouts, driven by node
attributes (groups and orders) and by the edges between groups; force-directed layouts, driven
by the edges; hierarchical layouts, in layers given by the direction of the edges, by a node
attribute or by the distance from a root; and UMAP layouts, driven by node vectors (node
attributes, embeddings, or the adjacency matrix). Any layout can then be mirrored, rotated and
zoomed, and its overlaps removed, and its edges can be bundled.

Circular layouts
----------------

Circular layouts put the nodes of every component on a circle. Hierarchical groups become sectors
of sectors, arranged so that groups sharing many edges are close, and a numeric attribute, or
the edges to the neighboring sectors, orders the nodes inside each sector.

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv", "nodes.tsv")
   gsk.circular_layout(net, ["region", "city"], order = "age")
   gsk.circular_layout(net, "community")      # inside sectors: by edges to the adjacent sectors

.. autofunction:: graphoskop.circular_layout

Edge bundling
-------------

Edge bundling draws the edges as curves that follow a hierarchy of groups of the nodes
(hierarchical edge bundling, as in graph-tool): edges between the same groups run together, and
the drawing shows how groups connect rather than a tangle of lines. The hierarchy can be given by
node attributes (from the coarsest level), such as the levels of a nested blockmodel, or found in
the layout itself. The curves are returned as chains of cubic Bézier curves (as graph-tool
does), which plotting libraries draw exactly.

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv")
   gsk.block_model_communities(net)                        # community, community_1, ...
   levels = ["community_2", "community_1", "community"]
   gsk.circular_layout(net, groups = levels)
   bends = gsk.bundle_edges(net, groups = levels)          # graph-tool's look
   gsk.force_directed_layout(net)
   bends = gsk.bundle_edges(net, beta = 0.6)               # hierarchy of the layout's space

.. autofunction:: graphoskop.bundle_edges

Force-directed layouts
----------------------

Force-directed layouts treat edges as springs pulling their nodes together and nodes as
particles pushing each other apart, and let the nodes move until the forces balance. The layout
of graphoskop aims at the look of organic layouts: groups of nodes that share many edges stay
together, the drawing takes the shape of the network rather than of a disk, and nodes with few
edges stay next to their neighbors instead of being thrown to the border.

.. code-block:: python

   import graphoskop as gsk
   import matplotlib.pyplot as plt
   from matplotlib.collections import LineCollection

   net = gsk.planted_partition_graph([40] * 6, 0.25, 0.005, seed = 1)
   pos = gsk.force_directed_layout(net, seed = 0)          # node attributes pos_x, pos_y
   xy = {node: p for node, p in zip(pos.index, pos.values)}
   edges = [(xy[a], xy[b]) for a, b in net.edges(None, labels = True)]
   community = net.to_pandas_dataframes()[1].set_index("node").loc[pos.index, "community"]
   fig, ax = plt.subplots()
   ax.add_collection(LineCollection(edges, colors = "gray", linewidths = 0.4, zorder = 1))
   ax.scatter(pos.pos_x, pos.pos_y, s = 8, c = community, cmap = "tab10", zorder = 2)
   ax.set_aspect("equal")
   ax.axis("off")

``clustering`` trades uniform edge lengths (0) for groups further apart (1), joined by longer
edges; the default, 0.25, leans towards uniform edges. Positions in three dimensions are stored in ``pos_x``, ``pos_y`` and
``pos_z``.

.. autofunction:: graphoskop.force_directed_layout

Grid layouts
------------

Grid layouts put the nodes on the cells of a regular grid. They work as circular layouts do, in
two dimensions: hierarchical groups become blocks of blocks, groups sharing many edges are side
by side, and inside each block nodes linked to other blocks sit on the border facing them, the
others in the middle (or a numeric attribute fills the block in reading order).

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv", "nodes.tsv")
   gsk.grid_layout(net, ["region", "city"])
   gsk.grid_layout(net, "community", order = "age")

.. autofunction:: graphoskop.grid_layout

Hierarchical layouts
--------------------

Hierarchical (layered) layouts draw directed acyclic networks with all edges pointing down, trees
from their root, and any network in layers given by a node attribute, with few edge crossings
and parents above the middle of their children. Radial layouts put the layers on concentric
circles instead, the root at the center, each subtree in its own wedge.

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("citations.tsv", directed = True)
   gsk.hierarchical_layout(net)                                   # edges point down
   gsk.hierarchical_layout(net, levels = "year", direction = "right")
   tree = gsk.load_network("tree.tsv")
   gsk.hierarchical_layout(tree, root = "ceo", direction = "radial")   # inside out

.. autofunction:: graphoskop.hierarchical_layout

Matrix layouts
--------------

Matrix layouts order the rows and columns of the adjacency matrix, to draw it as a matrix (heat
map): one order for square matrices, independent orders for the rows and columns of bipartite
networks. Groups become blocks, and the network's structure places similar rows and columns next
to each other; sorting by degree draws nested networks as triangles.

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv", "nodes.tsv")
   gsk.matrix_layout(net, "community")                  # communities as blocks on the diagonal
   bip = gsk.load_network("plants_pollinators.tsv", "species.tsv", bipartite = True)
   gsk.matrix_layout(bip, order = "degree", column_order = "degree", descending = True)

.. autofunction:: graphoskop.matrix_layout

Transforming layouts
--------------------

Positions computed by any layout can be mirrored, rotated and zoomed (along both axes or one of
them) around the center of the drawing, either in the node attributes of the network or in the
table a layout function returned. Nodes drawn as disks of a given size can be moved apart until
none overlap, keeping the arrangement of the layout.

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv")
   gsk.hierarchical_layout(net)              # positions in pos_x, pos_y
   gsk.mirror_layout(net, "y")               # root at the bottom
   gsk.rotate_layout(net, -90)               # clockwise: root on the left
   gsk.scale_layout(net, 0.5, axis = "y")    # half as tall
   gsk.remove_overlaps(net, 0.3)             # disks of diameter 0.3 do not overlap

.. autofunction:: graphoskop.mirror_layout

.. autofunction:: graphoskop.remove_overlaps

.. autofunction:: graphoskop.rotate_layout

.. autofunction:: graphoskop.scale_layout

UMAP
----

UMAP lays out nodes so that those with similar vectors end up close. The vectors can be numeric
node attributes, node embeddings computed on the fly, or the rows of the adjacency matrix, in
which case the edges of the network are the neighbor graph that UMAP keeps, and the result is a
force-directed layout with UMAP's forces. graphoskop has its own implementation, in PyTorch, so
it runs on the device of the network and adds no dependency; its neighbor graph matches the one of
the ``umap-learn`` package to float precision, and its layouts score as well on trustworthiness
and on nearest neighbor classification of known groups (handwritten digits, Gaussian blobs).

.. code-block:: python

   import graphoskop as gsk

   net = gsk.lfr_benchmark_graph(2000, 10, 50, 0.2, minc = 20, maxc = 100, seed = 4)
   gsk.umap_layout(net, seed = 0)                               # adjacency matrix
   gsk.node2vec(net, 64, seed = 0, attribute = "emb")
   gsk.umap_layout(net, "emb", metric = "cosine", attribute = "pos_n2v")

To follow a network over time, pass the layout of the previous snapshot as ``initial``: nodes
that were already there start at their old positions and are held near them (more firmly with a
larger ``alignment``), new nodes start next to their neighbors, and the layout changes only as much
as the new edges and nodes require (aligned UMAP).

.. code-block:: python

   pos_2024 = gsk.umap_layout(gsk.load_network("edges_2024.tsv"), seed = 0)
   net_2025 = gsk.load_network("edges_2025.tsv")
   pos_2025 = gsk.umap_layout(net_2025, initial = pos_2024, seed = 0)

.. autofunction:: graphoskop.umap_layout
