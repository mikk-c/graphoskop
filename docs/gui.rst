Graphical interface
===================

Graphoskop has a graphical interface, in the ``gui`` folder of the source tree: a
`Godot <https://godotengine.org/>`_ 4.7 application that calls the library and draws its results.
The library does not depend on it, and works the same without it.

The interface never computes anything about a network: a separate Python process, the worker,
runs the library, and the interface draws a result only once the function producing it has
finished. Since the worker is another process, the interface cannot slow a computation down,
and the drawing of a network is done by the GPU, a few draw calls for any number of nodes and
edges. A network is drawn again only when it changes (a new layout, new colors, a pan, a zoom, a
selection): an idle window draws nothing, so it takes no processor or GPU time from the worker.
While a long function runs, the progress bar of the status bar is updated once per second.

Analysis
--------

**Analysis › Centrality** computes a centrality of every node with the library (see
:doc:`centrality`): alpha, betweenness, closeness, degree, eigenvector, flow, harmonic, HITS
(hub and authority scores), k-core (coreness), Katz, PageRank, reach and SpringRank, and the
betweenness of every edge. Each opens a dialog with its parameters (the dialog also names the
attribute written), then stores the values as a node attribute (an edge attribute for edge
betweenness), for all components, e.g.

.. code-block:: python

   net.set_node_attribute("betweenness", gsk.betweenness_centrality(net, component = None), None)

The values can then be used for colors, sizes or opacities in the `Visual elements`_ panel; no
analysis changes the look of the drawing by itself. HITS writes two attributes,
``hits_hub`` and ``hits_authority``. Errors of the library (e.g. reach centrality and
SpringRank need directed networks) are shown in the dialog.

**Analysis › Clustering** has the local clustering coefficient (:func:`~graphoskop.local_clustering`,
attribute ``clustering``) and, for undirected signed networks, the signed clustering coefficient
of Costantini and Perugini (:func:`~graphoskop.signed_clustering`, attribute
``signed_clustering``, from -1 to 1: triangles count with their sign), with its ``kind``
(``watts_strogatz``: signs only; ``onnela`` and ``zhang``: with the magnitudes of ``weight``),
and, for hypergraphs and simplicial complexes, the local triangle closure and the two-node
clustering (written on the incidence network, as the higher-order centralities above), e.g.

.. code-block:: python

   net.set_node_attribute("signed_clustering", gsk.signed_clustering(net, component = None, kind = "zhang", weight = "correlation"), None)

Hypergraphs and simplicial complexes are drawn as their incidence networks (the nodes and, as
nodes of the other type, the hyperedges or facets), and the centralities above run on that
incidence network (``inc``). The last section of the menu has the measures of the library made for
them (see :doc:`higher_order`), computed on the hypergraph or complex itself and written on the
incidence network with :func:`gsk_worker.incidence_values` (a function of the interface: the values
of the nodes, of the hyperedges, or both, NaN for the others): **Hyperedge size…**
(:func:`~graphoskop.hyperedge_size`, on the hyperedges), **Local triangle closure…**
(:func:`~graphoskop.local_triangle_closure`), **Two-node clustering…**
(:func:`~graphoskop.two_node_clustering`; both under Analysis › Clustering), **Generalized degree (complexes)…**
(:func:`~graphoskop.generalized_degree` of the nodes) and **Node and hyperedge centrality…**
(:func:`~graphoskop.node_edge_centrality`, the nonlinear eigenvector centrality of Tudisco and
Higham, on both), e.g.

.. code-block:: python

   scores = gsk.node_edge_centrality(net, component = None, hyperedges = True)   # per component: node scores, hyperedge scores
   inc.set_node_attribute("node_edge", incidence_values(inc, [x for x, _ in scores], [y for _, y in scores]), None)

They refuse networks that are neither (:data:`gsk_worker.HIGHER_KINDS`).

**Degree** has one more option, **plot**: the complementary cumulative degree distribution (the
fraction of nodes with degree :math:`k` or more, on log-log axes) with the best fit of
:func:`~graphoskop.degree_power_law` (Clauset, Shalizi and Newman: the smallest degree of the
power-law regime, the exponent, and the tests against the exponential, the lognormal and the
power law with exponential cutoff), in a window that says whether the degree distribution is a
power law (with its exponent and the share of nodes in the power-law regime), with the warnings
of the tests (e.g. when a lognormal fits as well). The plot has no title; the colors of its
elements (data, fit, smallest degree of the fit) have color buttons under it, and the plot is
drawn again with them (:data:`gsk_worker.PLOT_ELEMENTS`); each data series (here the data and the
fit) is drawn as lines, points, lines with points, or bars (:data:`gsk_worker.PLOT_STYLES`).
Under them, each axis can be linear or logarithmic (log-log by default for the degree
distribution, linear for epidemics), has its label, which can be any text, and its range (from,
to: empty for automatic); heatmaps also have a row for their color scale (linear or logarithmic, its label and its range,
:data:`gsk_worker.PLOT_OPTIONS`); **Gradient** colors every bar, point and line segment of the
data series by its x value (**along x**) or its y value (**along y**) along a sequential or
diverging ColorBrewer palette (a menu showing its colors; **Reverse** reverses it), instead of the
color of its series: e.g. the bars of a histogram by the value of their bin, or a line by its
height (:func:`gsk_worker.gradient_colors`: from the smallest to the largest value, on a
logarithmic scale when the axis is, diverging palettes centered on 0 when the values have both
signs; not for heatmaps); **Line width** and **Point size** (times the usual ones) and **Bar
width** (a fraction of the gap between bars, 0.8 by default) size the data series, each shown
when a series is drawn that way (options ``line_width``, ``point_size``, ``bar_width``); the font
size can be set (14 points by default). Tick marks are inside
the axes, on all four sides. Colors, styles, axes, gradient and font size show only with **Advanced**, at
the bottom left (see `Generators`_); the plot, its text, its menu and the save buttons always. The font is the
default font of gnuplot, Helvetica, or the first font with the same metrics that is installed
(Arial, Nimbus Sans, Liberation Sans…; :data:`gsk_worker.PLOT_FONTS`). **Save PNG…** and **Save PDF…** write
the plot, with its colors, to a file (:func:`gsk_worker.degree_plot`, PNG at 200 dpi), and add
to the code panel the code that writes it again (:func:`gsk_worker.plot_code`): a call of
:func:`gsk_worker.plot_figure`, a function of the interface, with the data of the plot taken
from the variables of the tab and the colors, styles and options shown (the order of a heatmap
computed again), e.g.

.. code-block:: python

   # plot_figure: plots of the graphical interface, as saved from the plot window
   plot_figure("degree", {"degrees": node_values(net, "degree"), "fit": fit}, colors = {"data": "#377eb8", "fit": "#e41a1c", "k_min": "#999999"}, styles = {"data": "points", "fit": "lines"}, options = {}, path = "/data/degree.png")

(**Save table…** of heatmaps likewise writes ``pd.DataFrame(...).to_csv(...)``). The same
plot window shows the plots of the epidemic models (see `Simulation`_), the distributions of
attributes (see `Data panel`_), the randomization tests of fitted generators (see
`Generators`_) and the nestedness of bipartite networks (below). The fit is part of the code:

.. code-block:: python

   fit = gsk.degree_power_law(net, ccdf = True)

The plot needs matplotlib in the Python environment of the worker (``pip install
graphoskop[gui]``); weighted degrees cannot be fitted.

**Analysis › Core-periphery** has the core-periphery models of the library (see
:doc:`core_periphery`): **Discrete (Borgatti-Everett)** (:func:`~graphoskop.discrete_core_periphery`),
**Continuous (Borgatti-Everett)** (:func:`~graphoskop.continuous_core_periphery`) and **Rombach**
(:func:`~graphoskop.rombach_core_periphery`). As the centralities, each opens its parameter dialog
and writes the core value of every node as a node attribute (``core_discrete``: 1 in the core, 0
in the periphery; ``core_continuous``; ``core_rombach``; the dialog can change the name), for the
whole network (nodes of different components are pairs without edges), keeping the fit in the
code, e.g.

.. code-block:: python

   core_continuous, rho = gsk.continuous_core_periphery(net, component = None)
   net.set_node_attribute("core_continuous", core_continuous, None)

The status bar gives the fit: the correlation :math:`\rho` of the Borgatti-Everett models, the best
core quality of Rombach's (``quality``, the table of every pair of parameters, stays in the code).

**Nestedness**, for bipartite networks (and hypergraphs and simplicial complexes, as their
incidence networks; on other networks a message says so), computes the NODF nestedness, compared
with a null model if one is chosen (equiprobable, proportional, fixed degrees; with the number of
samples), and the isocline (by the fill of the matrix, or fitted to it), e.g.

.. code-block:: python

   nested = gsk.nestedness(net, null_model = "fixed")
   isocline = gsk.nestedness_isocline(net, method = "fill")

(:func:`gsk_worker.nestedness_code`), and opens the plot window with the biadjacency matrix, rows
(nodes of the first type) and columns sorted by decreasing degree, as a heatmap: matrices larger
than 400 rows or columns are compressed into 400 blocks per side, each colored by its share of ones
(:func:`gsk_worker.nestedness_matrix`, with a color bar), and the isocline drawn on top: in a
nested matrix the ones are above and left of it (:func:`gsk_worker.nestedness_plot`). Under the
plot, the NODF (of rows and columns), the comparison with the null model (mean, standard deviation,
z-score and p-value) and the parameter of the isocline (:func:`gsk_worker.nestedness_text`). The
colors of the matrix and of the isocline, the labels and ranges and the font can be changed, and
the plot saved.

**Analysis › Community discovery** lists every community discovery algorithm of the library (see
:doc:`communities`): BigCLAM, DEMON, hierarchical random graphs, Infomap, label propagation,
Leiden, link communities (hierarchical, and from the line graph), OSLOM and stochastic
blockmodels. Each opens a dialog with its parameters, with the same controls as for the
generators (see `Generators`_), then runs on the network of the current tab and adds its line to
the code panel, e.g.

.. code-block:: python

   gsk.leiden_communities(net, resolution = 2)

The communities are written as an attribute (``community`` by default, ``link_community`` for
link communities; the dialog can change the name), shown in the data panel, which can then be
used for colors in the `Visual elements`_ panel. The look of the drawing does not change, and
the view keeps its zoom and center. Hypergraphs and
simplicial complexes are analyzed as the incidence networks that are drawn.

The hierarchical methods (hierarchical random graphs, Infomap and the nested stochastic
blockmodel, :data:`gsk_worker.HIERARCHICAL_COMMUNITIES`) write one attribute per level
(``community``, ``community_1``, ``community_2``, …), and their dialog has **tree**: after the
run, the tree of the levels (:func:`~graphoskop.community_tree`: a root, the groups of every
level, the nodes) opens as a network in a new tab, laid out radially with the root at the
center, its nodes colored by ``level`` (worker command ``community_tree``), e.g.

.. code-block:: python

   net = gsk.community_tree(net, ["community", "community_1"])
   gsk.hierarchical_layout(net, direction = "radial")   # the root at the center

As any network, it can be laid out again, colored by the attributes of the groups (``size``,
``depth``) or of the nodes, analyzed and saved. **Tree of a hierarchy…** opens it later, for the
levels of the last hierarchical run or any node attributes that make levels (``levels``; with
``leaves`` off, the groups of the finest level are the leaves, for large networks). **Dendrogram
of a hierarchy…**, the last item of the menu, draws the same hierarchy as a dendrogram in the plot
window (:func:`gsk_worker.dendrogram_plot`), for the levels of the last hierarchical run or any
node attributes that make levels (``levels``, by default those of the last hierarchical run, in any
order: they are sorted from the one with most groups). Every group hangs from the group of the
next level holding most of its nodes (the only one, when the levels are nested), the coarsest from
a root; the leaves are the nodes (up to :data:`gsk_worker.DENDROGRAM_NODES`, named by their id or
by ``label``) or the groups of the finest level, with their sizes (``leaves``), in an order without
crossing branches (:func:`gsk_worker.dendrogram_tree`, a function of the interface, also used by
the code of the saved plot). Link communities (a single cut of a hierarchy of the edges) have no
levels.

**Analysis › Node vectors** has the analyses of vectors on the nodes of the library (see
:doc:`node_vector_analysis`), on the numeric node attributes: **Annihilation distance**,
**Earth mover's distance**, **Generalized Euclidean distance**, **Network correlation** (two
attributes, ``a`` and ``b``; the generalized Euclidean distance of ``a`` alone is its norm),
**Network entropy**, **Network variance** (one attribute) and **Graph Fourier transform** (one
attribute). Each opens its parameter dialog, where the attributes are chosen from menus of the
numeric node attributes, with the other parameters of the function; for directed networks, the
Laplacian (``kind``: Chung, the default of the library, magnetic or out) with its charge ``q``
and teleportation ``alpha`` (not shown for undirected networks, :func:`gsk_worker.vector_hidden`). The analysis runs on the
largest component (the default of the library) and keeps its result in the code
(:func:`gsk_worker.vector_code`), e.g.

.. code-block:: python

   correlation = gsk.network_correlation(net, "age", "income", k = 2)
   net.network_attributes["network_correlation(age, income)"] = correlation   # a network attribute

The dialog of the generalized Euclidean distance starts with ``sparse`` on (the Laplacian solver
of :func:`~graphoskop.laplacian_solver`): on a directed network of 11,000 nodes, a dense
pseudoinverse takes 8 seconds on a CPU with the Chung Laplacian (5 minutes with the out
Laplacian, a singular value decomposition), the solver a fraction of a second, with the same
result. The other analyses of one or two attributes need the dense pseudoinverse (the effective
resistances of every pair of nodes). The Chung Laplacian sums to 1 over the nodes, so its
resistances are about ``n`` times larger than those of the other Laplacians: choose the scale
``k`` of the network correlation and entropy about ``n`` times smaller (see
:func:`~graphoskop.network_correlation`).

A value is shown in a message and in the status bar (saying when it is of the largest of several
components). **Generalized Euclidean distances, pairwise** and **Network correlations, pairwise**
compare a set of attributes, chosen with check boxes (none: all the numeric attributes;
``attributes`` of :func:`~graphoskop.pairwise_generalized_euclidean` and
:func:`~graphoskop.pairwise_network_correlation`), and open the plot window with their matrix as
a heatmap (:func:`gsk_worker.matrix_plot`: distances from white to one color, correlations from
-1 to 1 through white, missing values grey; the names of the attributes along the axes up to
:data:`gsk_worker.MATRIX_NAMES` attributes, the values in the cells up to
:data:`gsk_worker.MATRIX_VALUES`), e.g.

.. code-block:: python

   correlations = gsk.pairwise_network_correlation(net, attributes = ["age", "income", "height"])

Under the heatmap, the **colors** row sets the scale of the colors, linear or logarithmic
(symmetric logarithmic when the range includes zero or negative
values, e.g. correlations), the label of the color bar and the range of the colors (from, to:
empty for automatic, -1 to 1 for correlations and 0 to the largest value for distances; values
beyond get the color of the end), as for the nestedness heatmap. The **Order** menu orders the
rows and columns: **as computed**, **random**, or **block diagonal (clusters)**, which puts
similar rows next to each other so that groups of similar attributes become blocks on the
diagonal (:func:`gsk_worker.block_order`: the leaves of an average-linkage hierarchical clustering
with the optimal leaf ordering; distances as they are, one minus the correlation for
correlations; :func:`gsk_worker.matrix_order`). **Save
table…** writes the matrix as a tab separated table (in the order computed). **Graph Fourier transform**
computes the Fourier basis with the parameters of the dialog, then the coefficients of the
attribute (``basis = gsk.graph_fourier_basis(net)``, ``coefficients =
gsk.graph_fourier_transform(net, "age", basis = basis)``), and plots the squared coefficients
against the frequencies (:func:`gsk_worker.spectrum_plot`), with the energy, the total variation
and how many of the lowest frequencies hold half and 90% of the energy under the plot
(:func:`gsk_worker.spectrum_text`).

**Analysis › Hierarchy** measures how hierarchical a directed network is (see :doc:`hierarchy`),
as one score shown in a message and in the status bar, of the largest component or, with
**all_components**, of the whole network (:func:`gsk_worker.hierarchy_code`): **Global reaching
centrality…** (:func:`~graphoskop.global_reach_centrality`), **Flow hierarchy…**
(:func:`~graphoskop.flow_hierarchy`, by edges or nodes, weighted or not), **Arborescence
hierarchicalness…** (:func:`~graphoskop.arborescence_hierarchy`) and **Agony…**
(:func:`~graphoskop.agony_hierarchy`; with **levels**, the level of every node, 0 at the top,
becomes a node attribute). The score is kept in ``hierarchy`` and as a network attribute
(Network tab of the data panel) named as the function (``_all_components`` added for the whole
network), e.g.

.. code-block:: python

   hierarchy, levels = gsk.agony_hierarchy(net, ranks = True)
   net.network_attributes["agony_hierarchy"] = hierarchy   # a network attribute
   net.set_node_attribute("level", levels)

Undirected networks are refused by the library, with its message.

**Analysis › Network statistics** computes global values of the network (see
:doc:`network_statistics`), each kept as a network attribute and shown in a message
(:func:`gsk_worker.statistic_code`): **Density…**, **Average degree…**, **Reciprocity…**
(directed networks), **Global clustering coefficient…**, **Average clustering coefficient…**,
**Global triangle closure…** (hypergraphs and simplicial complexes), **Assortativity…** (by
degree, or by a node attribute), **Average shortest path length…**, **Diameter…**
(:func:`~graphoskop.diameter`: the longest shortest path, of the largest component by default),
**Degree centralization…** (its centralization, relative centralization and entropies,
``degree_…``), **Connected components…**, for hypergraphs **Uniform hypergraph…**
(:func:`~graphoskop.is_uniform`), for simplicial complexes **Pure complex…**
(:func:`~graphoskop.is_pure`), **Pseudomanifold…** (:func:`~graphoskop.is_pseudomanifold`),
**Betti numbers…** (:func:`~graphoskop.betti_numbers`) and **Euler characteristic…**
(:func:`~graphoskop.euler_characteristic`), and **Modularity of a node attribute…**
(:func:`~graphoskop.modularity` of the partition of the nodes by the values of an attribute,
:func:`gsk_worker.modularity_code`): qualitative values are the communities as they are, and so
are numeric ones with at most :data:`gsk_worker.MODULARITY_LABELS` distinct integers (e.g.
community numbers); other numeric attributes are cut into **classes** (4 by default) with a
**method** (quantiles: classes of about as many nodes; evenly spaced; natural breaks, as the color
stops, :func:`gsk_worker.breaks`), or at the **stops** typed (e.g. ``30, 40``: below 30, from 30
to 40, from 40 on), written in the code; missing values make a class of their own; the network
attribute is ``modularity(age)`` unless named. Each opens its parameter dialog, whose first parameter is the name of the network
attribute (the function, by default). **All of them** computes, with their defaults, all those
that apply to the network (:func:`gsk_worker.all_statistics`; the average shortest path length
from :data:`gsk_worker.EXACT_PATHS` sampled sources in larger networks, the diameter of directed
networks only up to that many nodes, not the modularity; for hypergraphs and simplicial complexes
only those the library computes for them, with their own), and names those the
library refuses; the message lists one value per line. E.g.

.. code-block:: python

   net.network_attributes["global_clustering"] = gsk.global_clustering(net)   # a network attribute
   net.network_attributes.update({f"degree_{k}": v for k, v in gsk.degree_centralization(net).items()})   # network attributes
   import numpy as np
   x = np.concatenate([v.cpu().numpy() for v in net.get_node_attribute("age", None)])   # values of age
   classes = np.where(np.isnan(x), -1, np.digitize(x, [30.0, 40.0]))   # classes of age: below 30, from 30 to 40, from 40 on; missing values apart
   net.network_attributes["modularity(age)"] = gsk.modularity(net, classes)   # a network attribute

**Analysis › Shortest paths › All pairs lengths…** computes the lengths of the shortest paths
between all pairs of nodes of every component (:func:`~graphoskop.all_pairs_shortest_path_lengths`
with ``component = None``; the dialog asks the edge lengths, ``weight`` and ``inverse``) and keeps
them in ``paths``, a list with one matrix per component, largest first
(:func:`gsk_worker.paths_code`, worker command ``paths``):

* the analyses that take precomputed lengths reuse them while the network does not change and
  their ``weight`` and ``inverse`` are the same (:func:`gsk_worker.paths_argument`): betweenness,
  closeness and harmonic centrality (``lengths = paths``), the diameter and the average shortest
  path length (``lengths = paths[0]`` for the largest component; not when it is sampled);
* the lengths of the largest component, whatever its size, are the table **Shortest path
  lengths** of the Tables tab of the data panel, nodes as rows and columns, sharing the memory of
  the matrix (``copy = False``; the tab shows its first :data:`gsk_worker.TABLE_ATTRIBUTES`
  columns);
* **average_distance** (empty for none) is the node attribute of the average length of the shortest
  paths from every node to the others of its component, the inverse of its closeness centrality;
* with **heatmap** (on by default), the plot window shows the lengths of the largest component as
  a heatmap, whose **Order** menu orders the rows and columns: as computed (the order of the
  nodes), random, block diagonal (clusters of nodes at similar distances,
  :func:`gsk_worker.block_order`, up to :data:`gsk_worker.BLOCK_ORDER_MAX` nodes: it is quadratic
  in memory), block diagonal as the matrix view of the network (the order of
  :func:`~graphoskop.matrix_layout`: densely connected nodes next to each other), or by a node
  attribute (grouped by a qualitative one, sorted by a numeric one, also with the matrix layout;
  up to :data:`gsk_worker.MAX_ORDER_ATTRIBUTES` attributes; :func:`gsk_worker.matrix_orders`). Any
  size is drawn: beyond :data:`gsk_worker.MATRIX_SHOWN` nodes, one node in k of the order is
  drawn (the text under the plot says so), the data staying whole.

**Analysis › Node embeddings** computes shallow embeddings, unsupervised vectors of the nodes
learned from the structure alone (see :doc:`gnn`): **node2vec** (random walks; ``p = q = 1``
is DeepWalk), **struc2vec** (structural roles), **metapath2vec** (walks along a chain of node
types, ``metapath``, of the qualitative node attribute ``node_type``), **Laplacian eigenmaps**
and **Locally linear embedding** (eigenvectors: no training). Unlike the graph neural networks,
they predict nothing: each writes its vectors as numeric node attributes ``<attribute>_0``,
``<attribute>_1``, … (``attribute``, by default ``node2vec``, ``struc2vec``, ``metapath2vec``,
``eigenmaps`` or ``lle``), of :data:`gsk_worker.EMBEDDING_DIMENSIONS` dimensions unless set
(the library's default for the trained ones, 128, would make as many attributes); the status bar
names them (:func:`gsk_worker.embedding_code`), e.g.

.. code-block:: python

   gsk.node2vec(net, dimensions = 16, seed = 1, attribute = "node2vec")

They can then color, size or position the nodes, and be the ``features`` of the graph neural
networks (Analysis › Graph neural networks…): in the menu of ``features``, **Pattern…** checks
``node2vec_*`` at once, e.g. ``features = [a for a in net.node_attributes if
a.startswith("node2vec_")]``; with a target, a GCN on the embedding and an MLP on it compare
what the walks and the graph convolution each add. Trained embeddings show a progress bar.

Nothing is asked or announced: the status bar says where the lengths are, and they are reused
without a word. The matrices take 4 bytes per pair of nodes (400 MB for 10,000 nodes, 576 MB
for 12,000, computed in about 20 seconds).

E.g.

.. code-block:: python

   paths = gsk.all_pairs_shortest_path_lengths(net, component = None)   # one matrix per component, largest first
   tables = {}
   tables["Shortest path lengths"] = pd.DataFrame(paths[0].cpu().numpy(), index = net.node_ids[0], columns = net.node_ids[0])   # the largest component
   net.set_node_attribute("average_distance", [1 / c for c in gsk.closeness_centrality(net, component = None, lengths = paths)], None)
   net.set_node_attribute("betweenness", gsk.betweenness_centrality(net, component = None, lengths = paths), None)   # later, reusing them
   net.network_attributes["diameter"] = gsk.diameter(net, lengths = paths[0])

**Analysis › Frequent subgraphs** finds the frequent labeled patterns of the network (see
:doc:`frequent_subgraphs`): **In the whole network…** (:func:`~graphoskop.single_graph_mining`,
support: the minimum image based support) or **Across components…**
(:func:`~graphoskop.transactional_graph_mining`, support: the number of components containing the
pattern). The parameter dialog asks the least support, the node labels (a node attribute; none:
all nodes alike), the edge labels (an edge attribute), the largest number of edges of a pattern,
whether to keep the maximal patterns only, and **rules**: also the association rules between
patterns that differ by one edge (:func:`~graphoskop.graph_association_rules`). The patterns open
in a new tab, as a network with one component per pattern (laid out by the force-directed
layout): each node keeps its label and gets the node attribute ``pattern``, the id of its pattern
(color the nodes by it, or show the labels, see `Visual elements`_). The support of every pattern
(table **Pattern support**: pattern, support) and the rules (**Association rules**: antecedent and
consequent patterns, support, confidence, lift) are in the Tables tab of the new tab
(:func:`gsk_worker.mining_code`), e.g.

.. code-block:: python

   patterns, support = gsk.single_graph_mining(net, 30, node_label = "city", max_edges = 3)
   assert patterns is not None, "no pattern reaches the support: lower min_support"
   rules = gsk.graph_association_rules(net, patterns, support, transactional = False)
   net = patterns
   net.set_node_attribute("pattern", [[k] * c.num_nodes for k, c in enumerate(net.components)], None)   # the id of the pattern of each node
   tables = {"Pattern support": support, "Association rules": rules}

**Analysis › Link prediction…** opens the dialog of the link predictions of the network
(``link_dialog.gd``; see :doc:`link_prediction`). All its runs score the same validation set: a
fraction of the edges, hidden from the training network (the positives), and as many (or a
multiple of) unconnected pairs (the negatives), made by :func:`~graphoskop.link_prediction_split`
at the first run, with 10% of the edges and ``seed = 1`` so that the code gives the same pairs
when it runs again; **Validation set…** makes another one (fraction, negatives, seed), dropping the
runs so far. **Add method** opens the parameter dialog of a method: common neighbors, Jaccard,
Adamic-Adar, resource allocation, CAR index, mutual information, preferential attachment, local
random walk, SimRank, vertex similarity, hierarchical random graph, block model, embedding (the
embedding of the training network, node2vec, Laplacian eigenmaps, locally linear embedding or
struc2vec, with its dimensions, then :func:`~graphoskop.embedding_lp`), association rules (the
patterns of the training network, mined with :func:`~graphoskop.single_graph_mining` with the
least support and largest size of the dialog, and their rules, then
:func:`~graphoskop.association_rules_lp`) and ensemble (the rankings of the runs checked in its
dialog, all by default, aggregated by :func:`~graphoskop.ensemble_link_prediction`, Borda or
Kendall). Every run is named (e.g. "Adamic-Adar 1"): its scores are a column of the validation
set, and its evaluation (:func:`~graphoskop.link_prediction_evaluation`, with the curves) is kept
in the dictionary ``link_predictions`` (:func:`gsk_worker.link_prediction_code`), e.g.

.. code-block:: python

   train, validation = gsk.link_prediction_split(net, seed = 1)
   link_predictions = {}
   validation = gsk.adamic_adar_lp(train, validation, name = "Adamic-Adar 1")
   link_predictions["Adamic-Adar 1"] = gsk.link_prediction_evaluation(validation, score = "Adamic-Adar 1", curves = True)

The dialog lists the runs with their area under the ROC curve (AUC) and average precision (AP),
check boxes and **Copy…** (the parameter dialog of the same method, with the values of the run).
**Plot** draws the checked runs together, as ROC curves or precision-recall curves (the menu next
to it; the **Curve** menu of the plot window switches too), each with its area in the legend, and
a random score dashed (the diagonal, or the fraction of positives) (:func:`gsk_worker.link_plot`;
under the plot, the runs from the best, :func:`gsk_worker.link_text`). In signed networks,
**Signs** tells how well social balance and social status predict the signs of the hidden edges
(:func:`~graphoskop.link_prediction_sign_evaluation`). As the epidemics dialog, it stays open, with
the parameter dialog and the plot above it.

**Analysis › Graph neural networks…** opens the dialog of the graph neural networks of the network
(``gnn_dialog.gd``; see :doc:`gnn`). **Add model** opens the parameter dialog of **GCN**,
**GraphSAGE**, **GAT**, **GATv2** (predicting a node attribute), **DiffPool** (an attribute of
the components, each a graph; without a target it only embeds them), **MLP**
(:func:`~graphoskop.mlp`: the same model as GCN without the network, plain linear layers on the
attributes of each node, without weights: plotted with the other runs, it shows what the network
structure adds to the prediction) or **HGNN** (:func:`~graphoskop.hgnn`, the hypergraph
convolution, for hypergraphs and simplicial complexes, which the other models refuse; the
simplicial convolutional network of the library, :func:`~graphoskop.scnn`, predicts attributes of
the edges or triangles of a complex, simplex attributes that the interface does not load, show or
draw, so it is not offered): the target (a menu of the
node attributes), its transformation for regressions (**log** or **log1p**, ``log(1 + x)``: the
model is trained on a new node attribute, e.g. ``log_income``, made first with
:func:`gsk_worker.derived` as in Edit › New node attribute; nodes whose logarithm is not defined
have no target and are left out, and the features are all the numeric attributes but the target
and its transformation, unless chosen), the hidden layers (``32, 16`` by default: the output layer is added, one
channel per class of the target for classification, qualitative targets or ``task`` classification,
one for regression), the features (check boxes; none: all but the target), the split of the nodes
into training and validation sets, the training (epochs, optimizer, learning rate, weight decay,
dropout, loss) and the seed (1 by default, so that runs compare on the same split and repeat when
the code runs again). Every run is named (e.g. "GCN 1"), and its result is kept in the dictionary
``gnn_runs`` (:func:`gsk_worker.gnn_code`), e.g.

.. code-block:: python

   gnn_runs = {}
   gnn_runs["GCN 1"] = gsk.gcn(net, "city", [32, 16, 4], seed = 1, qualitative = True)

The dialog lists the runs with their scores on the validation set (classification: accuracy,
macro F1 and the areas under the ROC and precision-recall curves; regression: :math:`R^2` and the
mean squared error; :func:`gsk_worker.gnn_evaluation`), check boxes and **Copy…**. **Plot** draws
the checked runs together, chosen in the menu next to it (and in the **Plot** menu of the plot
window) (:func:`gsk_worker.gnn_plot`): **ROC** or **Precision-recall** curves of the
classification runs (one-vs-rest and micro-averaged: every node and class is a pair, scored by
the probability of the class, positive if it is the class of the node; a random score dashed),
**Predicted against observed** values of the regression runs (a scatter of the nodes of the
validation set, with the diagonal of perfect predictions), or the **Learning curves** of any run
(training loss solid, validation loss dashed, at each epoch). Under the plot, the scores of the
runs drawn, from the best (:func:`gsk_worker.gnn_text`).

Architecture
------------

.. code-block:: text

   Godot interface (gui/*.gd, *.tscn)          Python worker (gui/python/gsk_worker.py)
   ----------------------------------          -----------------------------------------
   menus, dialogs, tabs, panels       --->      runs graphoskop, one namespace per tab
   draws nodes and edges, tooltips    <---      code, drawing buffers, colors, node ids,
   tables, progress bar                         pages of tables, progress
                       local TCP connection, frames: lengths, JSON header, binary payload

The interface listens on a free local port and starts the worker with the Python interpreter of
the user's environment, passing it the port. Requests and replies are frames (see
:mod:`gsk_worker`). Results come back as the code that produced them and, for drawings, as
float32 buffers that go unchanged into two Godot ``MultiMesh`` resources, one for the nodes and
one for the edges: one draw call each, with no loop over nodes or edges in the interface. Colors
are computed by the worker too, and so is a grid of the node positions, with which the interface
finds the node under the cursor by looking at a few nodes only.

Every tab has its own namespace in the worker, where the code shown in its code panel runs: the
code panel always shows exactly the code that produced what is drawn, and can be copied to a
script to reproduce it. A few lines call functions of the interface, defined in the worker
(:data:`gsk_worker.INTERFACE_FUNCTIONS`: e.g. :func:`gsk_worker.derived` for new node attributes,
:func:`gsk_worker.move_nodes` for dragged nodes, :func:`gsk_worker.plot_figure` for saved plots); before the first of them the code imports them
from the worker, its folder added to the path (:func:`gsk_worker.interface_import`), so that the
code runs as it is in any Python with the library, e.g.

.. code-block:: python

   import sys
   sys.path.append("/opt/graphoskop/gui/python")   # the folder of gsk_worker.py, the worker of the graphical interface
   from gsk_worker import derived, groups, matching, move_nodes, node_sizes, node_with, set_edge_values   # functions of the graphical interface

The scripts of the interface:

* ``main.gd`` (``main.tscn``): the main window, its menus, tabs, panels, status bar and the Fit
  button of the visualization panel.
* ``worker.gd``: starts the worker, sends requests and dispatches the replies to callbacks.
* ``environments.gd``: finds the Python environments the worker can run in.
* ``app_settings.gd``: the memory of the application between sessions (see `Settings`_).
* ``load_dialog.gd`` (``load_dialog.tscn``): the load dialog.
* ``params_dialog.gd`` (``params_dialog.tscn``): the parameters of a generator, a layout or an
  analysis.
* ``task_dialog.gd`` (``task_dialog.tscn``): the dialog of a long task, with Abort.
* ``plot_window.gd`` (``plot_window.tscn``): the plots (degree distribution, epidemic models,
  matrices of node vector analyses…).
* ``player.gd`` (``player.tscn``): the player of the spread of an epidemic.
* ``ring.gdshader``: the rings of the selected nodes.
* ``select_dialog.gd`` (``select_dialog.tscn``): Select › Select by attribute.
* ``epidemics_dialog.gd`` (``epidemics_dialog.tscn``): Simulation › Epidemics.
* ``robustness_dialog.gd`` (``robustness_dialog.tscn``): Simulation › Robustness and Simulation ›
  Synchronization (two instances, with their own functions).
* ``link_dialog.gd`` (``link_dialog.tscn``): Analysis › Link prediction.
* ``gnn_dialog.gd`` (``gnn_dialog.tscn``): Analysis › Graph neural networks.
* ``derive_dialog.gd`` (``derive_dialog.tscn``): Edit › New node attribute.
* ``network_view.gd`` (``network_view.tscn``): the visualization panel of one network, with the
  shaders ``node.gdshader`` (discs of at least 3 pixels), ``edge.gdshader`` (lines of a sixth
  of the node diameter, at least 1 pixel), ``curve.gdshader`` (bundled edges: cubic Bézier
  curves computed by the GPU), ``loop.gdshader`` (self loops: circles above their node) and
  ``arrow.gdshader`` (arrowheads of directed edges).
* ``visual_panel.gd`` (``visual_panel.tscn``): the visual elements panel, with its dialogs
  ``color_dialog.gd`` (``color_dialog.tscn``), ``size_dialog.gd`` (``size_dialog.tscn``) and
  ``position_dialog.gd`` (``position_dialog.tscn``).
* ``backbone_dialog.gd`` (``backbone_dialog.tscn``): the backbone dialog (Edit › Backbone).
* ``swatch_button.gd`` and ``swatch_picker.gd`` (``swatch_picker.tscn``): the color buttons and
  their color picker; ``colorbrewer.json``: the ColorBrewer palettes.
* ``data_panel.gd`` (``data_panel.tscn``) and ``table_view.gd`` (``table_view.tscn``): the data
  panel.
* ``python_highlighter.tres``: Python syntax colors of the code panel; ``theme.tres``: the fonts
  of the operating system.

They are documented with Godot's documentation comments (``##``). Dialogs are as large as their
contents (long lists scroll).

Data panel
----------

The data panel, at the bottom left, has four tabs:

* **Nodes**: the attributes of all nodes, one row per node (positions included, as ``pos_x`` and
  ``pos_y``); when one node is selected, its attributes and their values; when several are, their
  rows;
* **Edges**: all edges (source, target and attributes); the selected edges, if any; otherwise
  the edges of the selected nodes;
* **Network**: the properties of the network (kind, direction, numbers of nodes, edges and
  components, device, attributes), then its network attributes (``net.network_attributes``, see
  :doc:`network`): the global values of the analyses (network statistics, hierarchy, node vector
  values, the fit of an error correction), one row each;
* **Tables**: the tables of the network, one at a time, chosen in the menu above the table: the
  runs of epidemic models (see `Simulation`_: for each step, the mean fraction of the nodes in
  each state, with its confidence interval), and the tables of the dictionary ``tables`` of the
  code, such as the support of frequent patterns and their association rules (see `Analysis`_;
  :func:`gsk_worker.tab_tables`).

The Nodes and Edges tables have a **Columns** menu, at the top right: **Show all**, **Hide all**,
and a check box per attribute (checked: shown), with patterns of names as in the parameter dialogs
(e.g. **label_* (9819)** shows or hides all the attributes starting with ``label_``; see
`Generators`_; beyond 200 names, those of a pattern are not listed one by one). Hidden attributes
are not columns (nor rows of the table of one node), and the title says how many are hidden
(:func:`gsk_worker.shown_attributes`). The choice is kept for each network until it is closed;
it changes only the tables, not the network or the code.

Tables show 200 rows at a time, with buttons for the previous and next pages; only the visible
page is asked to the worker. Only the first 100 attributes are columns of the Nodes and Edges
tables (:data:`gsk_worker.TABLE_ATTRIBUTES`): a long node table can have thousands of attributes
(e.g. one per label), and a column for each would stall the interface. The title then says so
(e.g. "All nodes · first 100 of 10,323 attributes"); the table of one selected node lists all of
its attributes (:func:`gsk_worker.node_row`), and the Network tab names the first 20 of each kind
with their number (:data:`gsk_worker.SUMMARY_ATTRIBUTES`). Clicking the title of a column sorts the whole table by it (not only
the rows shown), ascending, then descending at the next click (an arrow marks it), and the first
rows of the sorted table are shown: numbers numerically, texts alphabetically, missing values last
(:func:`gsk_worker.sort_rows`). Numbers are aligned to the right, texts to the left, with their
column titles.

Double clicking a value of the Nodes or Edges table (or of the attributes of a single node) edits
it in place, in a field over the cell: Enter sets it (worker command ``set_value``), Escape, a
click elsewhere or scrolling the table leaves it as it was. Numeric
attributes take a number (empty: missing), qualitative ones any text (a text equal to an existing
value is that value; empty: missing). Ids, sources, targets and the node type of bipartite
networks cannot be edited, nor the attributes of hypergraphs and simplicial complexes. Each edit
is a line of code (:func:`gsk_worker.value_code`), which Edit › Undo takes back, and the drawing
follows (e.g. colors by the attribute, or ``pos_x`` moving the node), e.g.

.. code-block:: python

   net.set_node_attribute(pd.DataFrame({"node": ["b"], "age": [55.0]}), node_column = "node")
   # set_edge_values: values of edge attributes edited in the graphical interface
   set_edge_values(net, "weight", {0: 9.0})   # a -> b

(:func:`gsk_worker.set_edge_values`, a function of the interface defined in the namespace of every
tab: the library sets edge attributes as whole vectors). Hypergraphs and simplicial complexes show their incidence networks:
hyperedges and facets are rows of the node table, memberships of the edge table.

A right click on the title of a column of the Nodes or Edges table (an attribute; not in the
attributes of a single selected node) opens its menu:

* **Plot the distribution of …** opens the plot window (see `Analysis`_) with the distribution
  of the values of the attribute (:func:`gsk_worker.distribution_plot`), and its number of
  values, mean, median and range (numbers) or number of categories, and its missing values. The
  **Plot** menu of the window chooses what is plotted: **Count** (the number of nodes or edges
  with each value: one bar per value for at most 100 distinct integers, else 50 bins of the same
  width, logarithmic when the x axis is and the values are positive), **PDF** (the same as
  fractions: the probability density for bins), **CDF** (the fraction of values at most as large)
  or **CCDF** (at least as large) (:func:`gsk_worker.distribution`). Count and PDF are drawn as
  bars, CDF and CCDF as lines, until the style is changed; qualitative attributes have Count and
  PDF, with their 40 most frequent categories along the x axis. Empty axis labels are automatic
  (the name of the attribute; what the kind plots). As for every plot, the axes can be linear or
  logarithmic, with their labels and ranges, and the plot saved as PNG or PDF. The plot runs no
  code: the code panel does not change.
* **Delete …** asks to confirm, then removes the attribute, e.g.
  ``net.remove_node_attribute("score")`` (``remove_edge_attribute`` for edges); it cannot be
  undone. The positions of the layout (``pos_x``, ``pos_y``) and the attributes the drawing uses
  (for colors, sizes, opacities or positions) are not removed: a message says why (the style never
  changes by itself, so it has to be changed first).
* **New node attribute from …** (Nodes table) opens Edit › New node attribute with the attribute
  as formula (see `Editing`_);
* **Hide …** hides the column (see the Columns menu above).

Editing
-------

The **Edit** menu starts with the operations that make a new network from the network of the
current tab, each opening in a new tab: **Bipartite projection**, **Backbone** and **Transform**
(see below), then **New node attribute…**, a node attribute computed from others, and **Error
correction**, the true network behind a noisy one (in a new tab). Then it changes the network of the current tab using the selected nodes and edges
(see `Visualization panel`_ and `Selecting`_):

* **Undo** (Ctrl+Z, ⌘Z on macOS) and **Redo** (Ctrl+Shift+Z): back and forth over the moves of
  nodes (dragged in the visualization panel), the removals of nodes and edges, the nodes and
  edges added (right click in the visualization panel, see `Visualization panel`_) and the values
  edited in the `Data panel`_, with their
  lines of code (worker command ``undo``: the network, its positions and its code as they were).
  They go back to the last other change of the network (a layout, an analysis, a simulation…),
  which cannot be undone; up to :data:`gsk_worker.MAX_UNDO` steps;
* **Remove selected nodes**, **Remove selected edges**, **Remove selected nodes and edges**
  (also the Delete key): :func:`~graphoskop.remove_nodes`, :func:`~graphoskop.remove_edges`; the
  selection is cleared;
* **Induced subgraph of the selected nodes**: :func:`~graphoskop.subgraph`, the selected nodes
  and all the edges between them;
* **Ego network of the selected nodes**: :func:`~graphoskop.ego_network` of the selected node,
  its neighbors and the edges between them (of several selected nodes: the subgraph induced by
  them and their neighbors); **…, without them** leaves the egos out (``include_ego = False``).

Removals change the network of the tab; the subgraph and the ego networks open in a new tab,
whose code is the code of the tab followed by the new line, e.g.

.. code-block:: python

   net = gsk.subgraph(net, ["n12", "n40", "n41"])

The nodes keep their positions, and the drawing its colors and sizes (:func:`gsk_worker.edit_code`).

**Edit › Transform** has the transformations of the library (see :doc:`transformations` and
:doc:`higher_order`), each making a new network in a new tab, e.g.

.. code-block:: python

   net = gsk.to_undirected(net, combine = "mean")

* **Flip edge direction** (:func:`~graphoskop.flip_edge_direction`, all the edges),
  **Undirected…** (:func:`~graphoskop.to_undirected`), **Strongly connected components**
  (:func:`~graphoskop.to_strongly_connected`) and **Condensation…**
  (:func:`~graphoskop.network_condensation`), for directed networks;
* **Largest connected component** (:func:`~graphoskop.largest_component`): the largest
  component only, weakly connected in directed networks (for the largest strongly connected
  one, Strongly connected components first, then this);
* **Remove isolated nodes** (:func:`~graphoskop.remove_isolates`), **Remove parallel edges…**
  (:func:`~graphoskop.remove_parallel_edges`) and **Remove self loops**
  (:func:`~graphoskop.remove_self_loops`);
* **Quotient by a node attribute…** (:func:`~graphoskop.quotient_network`): the nodes with the
  same values of the chosen node attributes (one, or several: each combination of their values is
  a set) collapse into one node (nodes without a value of any of them stay as they are); the sets
  are made by :func:`gsk_worker.groups`, a function of the interface defined in the namespace of
  every tab, as the comment of the code says. Only the chosen qualitative attributes are kept
  (``qualitative``: the others would mix the values of the collapsed nodes), and nodes and edges
  get the numeric attribute ``count`` (``count_2``, … if taken), the number of nodes collapsed
  into each node and of edges merged into each edge. Edges within a set become self loops, by
  default:

  .. code-block:: python

     # groups: sets of nodes with the same value of a node attribute, a function of the graphical interface
     net = gsk.quotient_network(net, sets = groups(net, ["region", "party"]), counts = "count", qualitative = ["region", "party"])

* **Complement…** (:func:`~graphoskop.complement`): the pairs of nodes not joined become the
  edges, and the edges go (``self_loops``: every node also gets a self loop; ``max_edges``, by
  default 100 million, refuses larger complements: the complement of a sparse network is dense);
* **Union with another tab…**, **Intersection with another tab…** and **Difference with another
  tab…** (:func:`~graphoskop.network_union`, :func:`~graphoskop.network_intersection`,
  :func:`~graphoskop.network_difference`): a dialog asks for the other network, one of the other
  open tabs. So that the code runs on its own, the code of the other tab becomes a function
  (:func:`gsk_worker.other_code`; the interface uses the network the tab already has instead of
  running it again):

  .. code-block:: python

     def other_network():   # the network of the tab "friends", made again by its code
        import graphoskop as gsk
        net = gsk.load_network("friends.tsv")
        return net
     other = other_network()
     net = gsk.network_union(net, other)

* **Line graph** (:func:`~graphoskop.line_graph`): the edges become the nodes, joined when they
  share a node (in directed networks, when one ends where the other starts), with the edge
  attributes and the ``source`` and ``target`` of each as node attributes; for hypergraphs and
  simplicial complexes, the line graph of the hyperedges or facets, joined when they share nodes
  (``shared`` of them);
* **Clique expansion…** (:func:`~graphoskop.clique_expansion`), for hypergraphs and simplicial
  complexes, **Clique promotion…** (:func:`~graphoskop.clique_promotion`), for networks (its
  hypergraph or simplicial complex is laid out as its incidence network), and **Hypergraph walk
  network…** (:func:`~graphoskop.hypergraph_walk_network`: the network of the steps of a random
  walk on a hypergraph, by the model of Zhou, Carletti or with edge-dependent vertex weights);
* **One layer…** (:func:`~graphoskop.layer_network`, a menu of the layers) and **Aggregate of the
  layers…** (:func:`~graphoskop.aggregate_network`), for multilayer networks.

Items with parameters (how parallel edges combine, the attribute of the quotient, …) open their
parameter dialog first (:data:`gsk_worker.TRANSFORMATIONS`); the others run at once. Errors of the
library (e.g. a transformation of directed networks on an undirected one) are shown in the dialog,
or in a message. New networks keep the positions of their nodes when they have them (the quotient
places its nodes at the mean position of theirs), or get the quick grid layout.

**Edit › New node attribute…** (also **New node attribute from …** in the menu of a column of the
Nodes table, see `Data panel`_) opens a dialog (``derive_dialog.gd``) that computes a node
attribute, node by node, from a formula of other node attributes
(:func:`gsk_worker.formula_values`):

* node attribute names (in backquotes if they are not plain words, e.g. ```age in 2020```),
  numbers and texts in quotes;
* ``+``, ``-``, ``*``, ``/``, ``**`` (power) and ``%`` (remainder); ``+`` also joins two texts;
* comparisons ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=`` and ``and``, ``or``, ``not``, giving 1
  or 0;
* the functions of :data:`gsk_worker.FORMULA_FUNCTIONS`: ``log``, ``log2``, ``log10``,
  ``log1p``, ``exp``, ``sqrt``, ``abs``, ``round``, ``floor``, ``ceil``; ``min``, ``max``,
  ``mean`` and ``sum`` of one attribute (over all nodes, e.g. ``x / max(x)``), and ``min`` and
  ``max`` of two (node by node); ``where(condition, a, b)``; ``missing(x)`` and
  ``fill(x, value)``; ``number`` and ``text`` (from texts to numbers and back); and, for texts,
  ``substr(x, start, length)`` (positions from 1), ``upper``, ``lower``, ``length``,
  ``contains`` and ``replace``.

**Attribute** and **Function** insert a name or a function (each with its description) where the
cursor is, and the buttons insert the operators. While the formula is typed, the dialog describes
the values it gives (numbers and their range, or texts and how many distinct, the missing values
and the first values; :func:`gsk_worker.formula_summary`), or says what is wrong (an unknown name
or function, a text where a number is needed, …). Missing values stay missing, and so do the
logarithms of numbers that are not positive. **Create** runs one line of code, with
:func:`gsk_worker.derived`, a function of the interface (:data:`gsk_worker.DERIVED_NOTE`), e.g.

.. code-block:: python

   # derived: values of a node attribute computed from others by a formula, a function of the graphical interface
   net.set_node_attribute("income_per_year", derived(net, "income / (age - 18)"), None)
   net.set_node_attribute("region", derived(net, "upper(substr(city, 1, 3))"), None)

Numeric results make a numeric attribute, texts a qualitative one; an attribute of the same name
and kind is replaced (one of the other kind is refused). In long node tables, missing numbers
read as 0 (as in the library).

**Edit › Error correction** (right below Backbone) infers the true network behind a noisy one (see
:doc:`error_correction`): **From measurements…** (:func:`~graphoskop.measured_network_correction`:
the edges as observed, with their numbers of measurements and positive outcomes if known, and
error rates known or unknown) or **From edge probabilities…**
(:func:`~graphoskop.uncertain_network_correction`: an edge attribute with the probability of each
pair). The parameter dialog asks the model of the true network (a menu: nested blockmodel, the
default, flat blockmodels, configuration model, Erdős-Rényi), the edge attributes of the
measurements or probabilities, and, with Advanced, the sampling (samples, burn-in, interval,
chains) and the seed (1 by default). The first correction compiles the kernels of the library
(about two minutes, once). The corrected network opens in a new tab: every node pair sampled as an
edge, with its posterior probability of being one (edge attribute ``probability``), and the
values of the fit (model, groups, edges, description length, error rates, …) as network
attributes (:func:`gsk_worker.correction_code`). The edge attribute ``observed`` tells the pairs
that are edges of the network (1) from those only sampled (0). The correction works both ways: an
observed edge can get a low probability (likely spurious) and a pair that was not observed a high
one (likely missing). The backbone dialog then opens on that probability, at 0.5 (the pairs more
likely than not: the most likely network): its plot shows, at every threshold, the percentage of
the observed edges kept and the pairs added (not observed, above the threshold, as a percentage of
the observed edges), with the nodes not isolated; the text under the slider counts the observed
edges kept and removed and the pairs added. **Extract network** opens the pairs above the
threshold in another tab, as for backbones (see below), e.g.

.. code-block:: python

   corrected, correction = gsk.measured_network_correction(net, seed = 1)
   net = corrected
   net.network_attributes.update({k: v for k, v in correction.items() if isinstance(v, (int, float, str))})   # network attributes
   bb_test = gsk.test_bb_thresholds(net, quantiles = 100, score = "probability")
   bb_found = gsk.find_bb_thresholds(net, score = "probability")

Error correction is for undirected networks (the library says so otherwise).

**Edit › Bipartite projection** has every projection of bipartite networks of the library (see
:doc:`bipartite_projections`): **Simple count**, **Histogram intersection**, **Hyperbolic**,
**Resource allocation**, **Random walk**, **Cosine similarity**, **Pearson correlation** and
**Inverse Euclidean**. Each opens its parameter dialog: the type of the nodes to keep (a menu of
the two node types of the network), the weights (an edge attribute, or *unweighted*), and the
parameters of the method (e.g. ``lambd`` of resource allocation, ``directed`` of random walks);
the batches of paths (memory, not results) are left to the library. **Run** opens the projection
in a new tab, e.g.

.. code-block:: python

   net = gsk.simple_count_projection(net, node_type = "user", weight = None)

The projection keeps the nodes of the chosen type, with their attributes and positions; the
drawing keeps its colors and sizes, except those given by attributes the projection does not have
(the node types), which go back to their defaults (:func:`gsk_worker.prune_style`). The incidence
networks of hypergraphs and simplicial complexes are bipartite too: their projection is a network
(``net = gsk.….projection(inc, …)``, then ``del inc``). On networks that are not bipartite, a
message says so.

**Edit › Backbone** has every backboning function of the library (see :doc:`backboning`), each
opening its parameter dialog first (the weights: an edge attribute, or *unweighted*):

* the methods that score the edges, **Noise corrected**, **Disparity filter**, **Pólya filter**,
  **Doubly stochastic**, **High salience skeleton** and **Convex skeleton**, write their scores as
  edge attributes, then test the thresholds from keeping every edge to keeping none, in 100 steps
  that each drop about 1% of the edges, and find the remarkable ones, e.g.

  .. code-block:: python

     gsk.noise_corrected(net)
     bb_test = gsk.test_bb_thresholds(net, quantiles = 100, score = "nc_score")
     bb_found = gsk.find_bb_thresholds(net, score = "nc_score")

  and open the backbone dialog, whose plot has the threshold on the x axis, and on the y axis the
  percentage of the edges kept and of the nodes that are not isolated. A handful of edges far
  below (or above) all the others would stretch the x axis: when the first (last) step of the
  thresholds is longer than a quarter of all the other steps together, it is a tail, and the axis
  starts after it (ends before it; :func:`gsk_worker.backbone_range`). **Extend to the left** and
  **Extend to the right** show the tails (they are off when there is none); choosing or typing a
  threshold in a tail turns them on. The axis is on a symmetric logarithmic scale when the
  thresholds it shows have a long tail, as the z-scores of the noise corrected backbone often do
  (:func:`gsk_worker.backbone_scale`); **Components** adds the number of components
  with more than one node, on a second y axis (:func:`gsk_worker.backbone_plot`). The thresholds of
  :func:`~graphoskop.find_bb_thresholds` are dashed vertical lines: the first isolate, the first
  component break, and the max separation (the largest share of nodes not isolated minus share of
  edges kept: as many nodes and as few edges as possible). The **Threshold** slider is continuous
  (the value can also be typed) and draws the chosen threshold as a line on the plot, with the size
  of its backbone under it; the buttons of the three thresholds snap to them. **Extract with**
  sets the options of :func:`~graphoskop.extract_backbone` (:data:`gsk_worker.BACKBONE_OPTIONS`;
  they stay as set for the next backbones): **Isolated nodes kept** (``keep_isolates``: nodes left
  without edges stay, instead of being dropped), **Components reconnected**
  (``reconnect_components``: the fewest most significant dropped edges that keep every original
  component connected are added back; this reconnects the isolated nodes too, so that box is
  checked and locked) and **Isolated nodes reconnected** (``reconnect_isolates``: every node left
  without edges gets back its most significant dropped edge). The sizes under the slider are those
  of the backbone without these options. With **Advanced** on, rows under the plot change it as
  in the plot window: the color of every line, the scale of the threshold axis (automatic,
  linear or logarithmic), the labels of the axes, the ranges of the threshold axis and of the y
  axis, and the font size (:data:`gsk_worker.BACKBONE_OPTIONS_PLOT`). A range of the threshold
  axis (from, to) tests the thresholds again in it, 100 of them evenly spaced
  (:func:`~graphoskop.test_bb_thresholds` with a step; worker command ``backbone_plot``; the code
  does not change), so that the backbones of a narrow range can be told apart, and the slider
  spans that range; a range that cannot be drawn is explained under the rows. **Extract
  backbone** opens the backbone at the chosen threshold in a new tab:

  .. code-block:: python

     net = gsk.extract_backbone(net, 2.31, score = "nc_score", reconnect_isolates = True)

* the methods that filter the edges directly, **Maximum spanning tree**, **Maximally filtered
  graph** and **Triangulated filtered graph**, open their backbone in a new tab at once (e.g.
  ``net = gsk.spanning_tree(net)``).

Generators
----------

The **Generators** menu creates a synthetic network in a new tab, with every generator of the
library (see :doc:`generators`), by family: chain; random graphs (G(n, m), G(n, p), configuration
model, Kronecker graph, random geometric graph); clustered graphs (caveman, Watts-Strogatz);
broad degree graphs (Barabási-Albert, link selection, copying model, Holme-Kim); stochastic block
models (planted partition, general, degree-corrected, mixed membership, nested); community
benchmarks (Girvan-Newman, LFR); higher-order networks (growing manifold, simplicial and
hypergraph configuration models). :func:`~graphoskop.graphrnn` is not in the menu, since it
learns from a network.

Each item opens a dialog with the parameters of the function (their descriptions from the
documentation, as tooltips), set to preset values that make networks of about a thousand nodes,
or to the defaults. Every parameter gets the control that needs the least typing (see
:func:`gsk_worker.widget`):

* a switch for true / false values (e.g. ``directed``), and a **GPU** switch for the device
  (offered when PyTorch finds CUDA);
* a menu for values from a list (e.g. ``rewire`` of the Watts-Strogatz model, ``direction`` of
  hierarchical layouts) and for one attribute name (layouts, analyses);
* a menu of check boxes for several attribute names (e.g. the ``groups`` of a circular layout),
  kept in the order they are checked, with patterns of names (see below);
* a slider with a number for values between two bounds (probabilities, mixing parameters,
  angles);
* a number with arrows for integers and other numbers (e.g. ``n``);
* a field for names (e.g. the ``attribute`` written) and node ids, typed without quotes;
* a field for numbers that can be left empty (e.g. ``seed``);
* a field for lists: numbers separated by commas (``250, 250, 250, 250``), rows of a matrix
  separated by semicolons (``0.05, 0.002; 0.002, 0.05``), pairs ``key: value`` for dictionaries
  (``1: 75, 2: 50``); Python expressions such as ``[3] * 1000`` work too.

**Advanced.** Parameter dialogs first show only the parameters that matter most: those without a
default (required, or preset by the interface, such as the sizes of a generator) and a few
others (:data:`gsk_worker.BASIC`: weights, direction, groups and orders of layouts, the
attributes of analyses, the resolution of community discovery, the seeds and runs of epidemics,
the target, features, layers and epochs of graph neural networks, …;
:func:`gsk_worker.mark_advanced`). The **Advanced** switch, at the bottom left, shows all of them;
hidden parameters keep their values (defaults or presets), and when all are hidden the dialog
says so. The switch is off at first, and the choice is remembered, also for the other dialogs
that have it: the load dialog (see `Loading a network`_) and the plot window (see `Analysis`_).

**Patterns of attribute names.** Networks can have thousands of attributes, e.g. a long node
table with one attribute per label (``label_0``, ``label_1``, …). Menus of check boxes of node or
edge attributes then offer patterns at the top, each with the number of names it checks: every
prefix shared by three names or more, up to its last ``_``, ``-``, ``.``, ``:`` or space (or
before final digits), followed by ``*``, e.g. **label_* (9819)**. Checking a pattern checks all
the attributes matching it; **Pattern…** asks for any other (``*`` stands for any text, ``?``
for one character, e.g. ``gene_1??``; a pattern matching nothing is refused). Beyond 200 names,
the names a pattern covers are not listed one by one, nor the others beyond 200 (the menu says
how many are left out: Pattern… reaches them). With a pattern checked, the code calls
:func:`gsk_worker.matching`, a function of the interface (:data:`gsk_worker.MATCHING_NOTE`),
rather than listing the names, e.g.

.. code-block:: python

   # matching: names of the attributes matching patterns (* any text), a function of the graphical interface
   correlations = gsk.pairwise_network_correlation(net, attributes = matching(net, "gene_*", "age", numeric = True))

Menus of one attribute list all of them (they do not size themselves to their items: thousands
of items would take seconds).

A default ``None`` is shown as the value it stands for (e.g. ``False`` for ``directed``, ``4``
communities in the Girvan-Newman benchmark); when it stands for no value, the field is empty,
with a hint (e.g. ``random`` for ``seed``). Values are written into the code as Python
expressions, e.g.

.. code-block:: python

   net = gsk.planted_partition_graph(sizes = [250, 250, 250, 250], p_in = 0.05, p_out = 0.002)

with only the values that differ from the defaults. The dialog stays open while the function
runs: if a value is not valid (e.g. a seed that is not a number) or the library raises an error
(e.g. matrices of the wrong size), the error is shown in the dialog, which keeps the values
entered, to be corrected; it closes once the function has run. The network is then laid out and
drawn as a loaded one.

**Generators › Fitted to the current network** has the same generators (all but the chain), fitted
to the network of the current tab (their ``net`` parameter: e.g. the configuration model keeps its
degrees, the stochastic block models fit their blocks). Their parameter dialog leaves empty, with
the hint *from the network*, every parameter the network gives (only those typed override it;
yes / no settings are menus whose first item is *from the network*), and has two more parameters
first (:func:`gsk_worker.signature` with ``fit``):

* **test**: a measure of the network (:data:`gsk_worker.MEASURES`: average and global clustering,
  degree assortativity, average shortest path length, diameter, number of components; reciprocity
  and global reach centrality for directed networks; global triangle closure for hypergraphs and
  simplicial complexes), or *none*;
* **runs**: the number of null networks of the test (100).

**Generate** opens one null network in a new tab, after the code of the network, e.g.

.. code-block:: python

   net = gsk.configuration_model(net = net)

and, with a measure, also runs the randomization test in the tab of the network
(:func:`gsk_worker.fit_code`), e.g.

.. code-block:: python

   null_test = gsk.randomization_test(net, gsk.average_clustering, gsk.configuration_model, n = 100)

Its plot opens in the plot window (see `Analysis`_; :func:`gsk_worker.randomization_plot`): on the
x axis the value of the measure, on the y axis the number of null networks with that value
(rounded as by the library, about 20 bins), and the value of the network as a dashed vertical
line. Under it, the observed value, the mean and standard deviation of the null networks, and the
pseudo p-value, whose tail is chosen in the **Pseudo p-value** menu: right-tailed (the share of
null networks scoring at least the observed value), left-tailed (at most), or two-sided (twice the
smaller one, at most 1) (:func:`gsk_worker.randomization_text`). A pseudo p-value of 0 means that
no null network is as extreme: it is below one over the number of null networks, which the text
says. As every plot, its colors, styles, axes and font can be changed, and it can be saved.

Layouts
-------

The **Layout** menu lays out the network of the current tab with a layout of node-link diagrams
of the library (see :doc:`layouts`):

* **Circular**: :func:`~graphoskop.circular_layout`;
* **Force-directed**: :func:`~graphoskop.force_directed_layout`;
* **Grid**: :func:`~graphoskop.grid_layout`;
* **Grid, quick**: :func:`~graphoskop.grid_layout` with ``fast = True``, at once: the nodes of
  every component in rows, in their order, without reading the edges; the cheapest layout, and
  the first layout of every new network (laid out in a fraction of a second, even with 100,000
  nodes), to be replaced by a better one;
* **Hierarchical**: :func:`~graphoskop.hierarchical_layout`;
* **Hierarchical, radial**: :func:`~graphoskop.hierarchical_layout` with ``direction = "radial"``;
* **Matrix**: :func:`~graphoskop.matrix_layout`, drawn as the adjacency matrix (see below);
* **UMAP**: :func:`~graphoskop.umap_layout`.

Each opens a dialog with the parameters of the layout, with the same controls as for the
generators (see `Generators`_), set to the default values, descriptions as tooltips.
Parameters that take one attribute name, ``order`` and ``levels`` (node attributes) and
``weight`` (edge attribute), are menus of the attributes of the network, whose first item is the
default (e.g. ``by the edges``); ``groups`` and ``attributes`` (UMAP), which take several names
in order, are menus of check boxes, which keep the names in the order they are checked (the
button shows them). For example, a circular layout with sectors by ``country`` and, inside, by
``city``, clockwise by ``age``:

.. code-block:: python

   gsk.circular_layout(net, groups = ["country", "city"], order = "age")

The number of dimensions and the prefix of the position attributes are not offered: the
interface draws the two dimensions of ``pos``.

**Matrix** orders the rows and columns of the adjacency matrix with
:func:`~graphoskop.matrix_layout` (groups and orders of rows, and of columns in bipartite
networks; the order menus also offer ``degree``), and draws the network as the matrix: every
edge is a square cell, at the row of its source and the column of its target (both cells for
undirected edges; in bipartite networks, and in the incidence networks of hypergraphs and
simplicial complexes, rows are the nodes of the first type and columns the others), in the
colors and opacities of the edges. The nodes are drawn left of their rows (columns of bipartite
networks: above them), with their colors and sizes, tooltips and selection. The matrix stays
until another layout is run (see :func:`gsk_worker.drawing`).

Further items transform the current layout, at once: **Mirror left to right** and **Mirror top to
bottom** (:func:`~graphoskop.mirror_layout`), and **Remove overlaps**
(:func:`~graphoskop.remove_overlaps`), with ``size`` the diameters of the drawn nodes: a number, or, when node sizes are given
by an attribute, the default diameter times the relative sizes of the nodes, e.g.

.. code-block:: python

   # node_sizes: relative sizes of the drawn nodes, a function of the graphical interface
   gsk.remove_overlaps(net, size = 0.7 * node_sizes(net, "degree", smallest = 0.5, largest = 4.0))

so that no two nodes overlap, whatever their sizes. ``node_sizes`` (:func:`gsk_worker.node_sizes`)
is a plotting utility of the interface, not a function of the library: it is defined in the
namespace of every tab, and the comment says so in the code panel.

The last section curves the edges. **Bundle edges…** bundles the edges of the current layout with
:func:`~graphoskop.bundle_edges` (hierarchical edge bundling, as in graph-tool), with a dialog
for the hierarchy (``groups``: attributes from the coarsest level, or the space of the layout)
and the strength ``beta`` (a slider). The curves, chains of cubic Bézier curves, are stored in
``bends``:

.. code-block:: python

   bends = gsk.bundle_edges(net, groups = ["region", "city"], beta = 0.9)

and the edges are drawn along them, as curves computed by the GPU (``curve.gdshader``: every
Bézier piece is a strip whose vertices the shader puts on the curve, smooth at any zoom), until
another item of the menu changes the layout (run **Bundle edges…** again then); changing colors,
sizes and opacities, and community discovery, keep them. For very large networks, pieces are
drawn with fewer segments, so that there are at most :data:`gsk_worker.MAX_SEGMENTS`.

**Route edges around nodes** (and **Route edges around nodes…**, with a dialog) is an edge
router, like the organic edge router of Cytoscape: a straight edge that passes over a node it does
not connect looks like an edge to that node (a "fake edge"), so
:func:`~graphoskop.route_edges` bends it around the node instead. Edges that cross no node stay
straight; the others are cut into points pushed out of the nodes they cross (with an empty
``margin`` around every node) and smoothed, and drawn as curves through them, stored in
``bends`` as the bundled ones and drawn the same way. The nodes are as large as they are drawn,
fixed or mapped from an attribute (``size``, from :func:`gsk_worker.size_code`, as for **Remove
overlaps**; the dialog leaves it empty for that, and has ``margin``, by default a quarter of the
median diameter, and, among the advanced parameters, the rounds and the most points of an edge):

.. code-block:: python

   bends = gsk.route_edges(net, size = 0.353)

Nodes touching one of the ends of an edge, and nodes packed so tightly that there is no room
between them, may still be crossed. The positions do not change, so run it again after changing
the node sizes; like bundling, it lasts until another layout.

Every item adds its line to the code panel, e.g. ``gsk.force_directed_layout(net)``. While a
layout runs, the status bar says so, with a progress bar for long layouts.

The default diameter of the nodes is 70% of the median distance between a node and its nearest
neighbor (see :func:`gsk_worker.node_size`), recomputed after each layout and kept by the
transformations and by edge bundling; on screen, nodes are never smaller than 3 pixels. The default width of the
edges is a sixth of the default node diameter, and at least 1 pixel. Sizes and widths can be
changed in the `Visual elements`_ panel.

Loading a network
-----------------

**File › Load network…** opens the load dialog, which decides how
:func:`~graphoskop.load_network` (or :func:`~graphoskop.load_hypergraph`,
:func:`~graphoskop.load_simplicial_complex`, :func:`~graphoskop.load_matrix`) is called:

* the edge table (required) and the node table (optional), with the separator of both files and
  whether they have a header row. The first ten rows of each are shown, as the library reads
  them; wide tables scroll sideways, and long column names are cut in the tables and in the
  menus of columns, so the dialog never grows wider than the screen (e.g. a comma-separated file
  read with the tab separator, whose header is one very long column);
* the role of the columns, chosen on the row of column names over each preview, which stays in
  place when the rows scroll (and follows the columns when they scroll sideways): clicking a name
  (left or right button) opens the menu of the roles that apply. A column with a
  role has its symbol next to its name, its name in bold and its values in the color of the role
  (ColorBrewer Set1): a filled circle with an arrow out of it, red, for the source; an arrow into
  a filled circle, blue, for the target; a filled circle, black, for the node id; a tag, purple,
  for the attribute name and an equals sign, green, for the attribute value; stacked bars,
  orange, for layers; plus and minus, brown, for signs; two arrows, pink, for the direction in
  hypergraphs; a box, blue, for hyperedges and simplices. For the edge table: the source and target nodes (the first two
  columns by default); in multilayer networks, the layer; in signed networks, the signs; in
  directed hypergraphs, the column with ``"tail"`` or ``"head"``. For the node table: the node id
  (entity and layer in multilayer networks; the first columns by default), then, in long format,
  the attribute name and value. A role given to a column that had another one swaps them;
* the format of the node table, wide or long (guessed: long if it has three or more columns and
  a node id repeats in the first one);
* whether the network is directed, bipartite, signed, multilayer, a hypergraph or a simplicial
  complex (Signed, Multilayer, Hypergraph, Simplicial, Adjacency matrix and the device show only
  with **Advanced**, at the bottom left, or when chosen; see `Generators`_). In hypergraphs and simplicial complexes each row of the edge table is a node in a
  hyperedge or simplex (columns node and hyperedge or simplex), and the node table has the node
  id in its first column. Signed networks are loaded and then signed with
  :func:`~graphoskop.signed_network` (``weight`` = the sign column: 0 or more is positive);
* **Adjacency matrix**: the first file is an adjacency matrix, read with
  :func:`~graphoskop.load_matrix`: one row and one column per node (in bipartite networks, rows
  for the nodes of one type and columns for the others), every non-zero entry an edge, with its
  value as the edge attribute ``weight`` unless all of them are 1. With the header, the first row
  and the first column have the node ids (``header = True``; otherwise nodes are numbered from 0);
  the separator *whitespace* (``\s+``) is the default of matrices. Unchecked, **Directed** and
  **Bipartite** are found from the matrix: directed if it is not symmetric, bipartite if it is not
  square. The node table, if any, has the node id in its first column, e.g.

  .. code-block:: python

     net = gsk.load_matrix("/data/adjacency.tsv", sep = "\t", header = True)

* the device, CPU or GPU (offered when PyTorch finds CUDA).

**Comments start with** (next to the separator, ``#`` by default): lines of the tables starting with
it, after any spaces, are skipped, in the previews and when loading (``comment`` of
:func:`~graphoskop.load_network`, written in the code when it is not ``#``); empty for none. A
``#`` inside a value (e.g. a color ``#ff0000``) stays.

Without a file, the first field lists the formats it reads (edge table, Pajek, UCINET, adjacency
list, DOT, GML, GraphML, GEXF, Cytoscape JSON). **Network files of other programs**: when the first file has the extension of a format of
:func:`~graphoskop.read_network` (``.net`` and ``.paj`` Pajek, ``.dl`` UCINET, ``.adj`` and
``.adjlist`` adjacency lists, ``.dot`` and ``.gv`` Graphviz, ``.gml``, ``.graphml``, ``.gexf``
Gephi, ``.cyjs`` and ``.json`` Cytoscape; ``.##h`` and ``.##d`` UCINET binary data, whose
relations are the layers of a multilayer network), there is nothing to choose: the file holds the nodes,
the edges, their attributes and whether the network is directed. The dialog shows only the file,
a note, **Directed** for adjacency lists (each line lists the targets of its first node; unchecked,
an edge on the lines of both its nodes is one edge) and the device, and the code is e.g.

.. code-block:: python

   net = gsk.read_network("/data/karate.gml")

The visual attributes of the file (:data:`~graphoskop.formats.VISUAL_ATTRIBUTES`) become the style of the
drawing (:func:`gsk_worker.file_style`), as if chosen in `Visual elements`_: node and edge colors
and node border colors (each value its own color: hexadecimal colors, or CSS color names; other
names, e.g. the Pajek colors that are not CSS colors, and nodes without one, get the default
color), node sizes (diameters proportional to the values, the median value at the default
diameter; nodes without one at the default), edge widths (1 is the default width), border widths
(relative to the node sizes when the file has them), opacities, node shapes and edge line styles
(:data:`gsk_worker.SHAPE_NAMES`, :data:`gsk_worker.LINE_NAMES`: e.g. ``box`` is a square,
``dashed`` a dash; others are circles and solid lines), labels (shown) and the most common label
color. Visual attributes the interface does not draw (e.g. fonts, curved edges, arrowheads) are
ignored, though they stay node and edge attributes. When every node has a position, the network is
drawn with them, without laying it out; otherwise it is laid out as other networks.

On **Import**, the network opens in a new tab: the code panel shows the code that loaded it, for
example

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("/data/edges.tsv", nodes = "/data/nodes.tsv", directed = True)
   gsk.grid_layout(net, fast = True)

with only the arguments that differ from their defaults. The network is then laid out with the
quick :func:`~graphoskop.grid_layout` (hypergraphs and simplicial complexes as their
:func:`~graphoskop.incidence_network`, with hyperedges or facets drawn as nodes of another color),
and drawn. The layout line is part of the code, since it stores the positions as the node
attributes ``pos_x`` and ``pos_y``. Errors of the library (e.g. a bipartite network without node
types) are shown in the load dialog, which stays open with its choices until the network is
loaded.

Networks with more than 10 million nodes and edges together (:data:`gsk_worker.MAX_ELEMENTS`;
generated ones too) can probably not be drawn: they are not laid out nor drawn, and a dialog says
so. **Analyze without drawing** (also closing the dialog) has nothing left to do: the network
stays without a layout and without a drawing. The visualization panel shows its size and **Draw
anyway**, and the visual elements panel, the search box, the Select menu and Export › Image… are off,
since nothing can be seen or selected. Everything else works as usual, with its code: layouts
(which store the positions without drawing them), analyses, simulations, backbones,
projections, the data panel. New tabs made from such a network (backbones, projections,
subgraphs) are drawn if they are small enough, and laid out with the quick grid first if they
have no positions. **Draw anyway** (in the dialog, or later in the visualization panel) draws
the network, after laying it out with the quick grid if it has no layout yet.

Main window
-----------

While the worker starts (when the interface opens, and when it restarts in another environment),
the window is greyed out and its menus are off, except the status bar: nothing can be asked of the
worker before it is ready (e.g. the load dialog would not know yet whether a GPU is available).

When a function of the library (loading, generating, a layout, an analysis) runs for more than
0.8 seconds, a dialog opens (above the dialog that started it): what runs, its progress bar for
functions with one, the time elapsed and left, and **Abort**. Abort interrupts the function as
soon as Python code runs again (at once for most functions; a long call into compiled code, e.g.
a Numba kernel, finishes first): it then fails with the error ``Aborted``, shown in the dialog
that started it, and nothing changes in the network or in the code. If the function has not
stopped 5 seconds later, the button becomes **Force stop**: the worker is stopped and started
again, and the open networks are rebuilt by running their code. Closing the dialog leaves the
function running. The dialog updates once per second, so it takes no time from the function.

The worker answers one request at a time. Opening a dialog asks it first (the parameters of a
function, the runs of the epidemics, a plot): if it is still finishing earlier work (e.g. a table
or an analysis of a network of millions of nodes), the same dialog opens after 0.8 seconds and
says what the interface is waiting for, instead of nothing happening.


* The menu bar: File, in four groups: **New network…** (see below); **Open session…**,
  **Recent sessions** (the last ten sessions saved or opened, by file name, the folder in the
  tooltip: choosing one opens it), **Save session…** and **Close session** (closes every tab,
  after asking; see below); **Load network…** (see `Loading a network`_) and **Export**, with
  **Network…** (the network of the current tab, in the formats of the load dialog, chosen by the
  file type of the file dialog: tab-separated tables, as
  :meth:`~graphoskop.Network.to_pandas_dataframes` gives them: ``name_edges.tsv``,
  ``name_nodes.tsv`` if there are node attributes, and ``name_interlayer.tsv`` for multilayer
  networks; or a Pajek, UCINET, adjacency list, DOT, GML, GraphML, GEXF or Cytoscape JSON file,
  written by :func:`~graphoskop.write_network` with the drawing as its visual attributes,
  :func:`gsk_worker.drawn_visuals`: positions, colors, opacities, sizes, shapes, borders, labels
  and label colors of the nodes, colors, opacities, relative widths and line styles of the edges,
  as on screen; File › Load network reads them all back; the code does not change; worker
  command ``export``) and **Image…** (the visualization panel of the current network as on screen,
  without the selection marks: a dialog asks its width in pixels, twice the width on screen by
  default, the height following the shape of the panel, and its format: a **PNG image**, with
  smooth edges, drawn three times larger still then reduced; a **PDF document, vector**, drawn by
  the worker from the drawing and the view (:func:`gsk_worker.vector_image`, command
  ``image_vector``): nodes as circles, edges as lines or Bézier curves ending at their
  arrowheads, matrix cells as squares, loops as rings, arrowheads as triangles and labels as
  text, with the colors, sizes, opacities and smallest sizes on screen of the panel, sharp at any
  zoom and editable in a vector editor, the page as large as the width at 300 dots per inch (tens
  of seconds for 100,000 nodes); or a **PDF document, image**, holding the PNG image at 300 dots
  per inch, command ``image_pdf``); **Settings…** (light or dark mode and the colors of the
  interface, see `Settings`_). Edit (see `Editing`_), Select (see
  `Selecting`_), Layout (see `Layouts`_), Analysis (see `Analysis`_),
  Simulation (see `Simulation`_), Generators (see `Generators`_), Help (**Online
  documentation**, and this chapter of it, opened in the web browser, and **Report a bug or
  request a feature**, the issue tracker of graphoskop on GitHub,
  https://github.com/mikk-c/graphoskop/issues, and **About graphoskop…**: a dialog with the icon
  of the program (``res://icon.svg``, if the project has it), the version of graphoskop and its
  folder, Python (version, environment, interpreter), the operating system, Godot, the GPUs PyTorch
  sees (CUDA version and device names, or none), and the versions of PyTorch, PyTorch Geometric,
  NumPy, SciPy, pandas, Numba, tqdm, Matplotlib, networkx, igraph and graph-tool (or "not
  installed"); its text can be selected, or copied whole with **Copy**, e.g. for a bug report;
  worker command ``about``, :func:`gsk_worker.about`). On macOS it is the global
  menu bar of the system.
* The tab bar: one tab per open network; the close button of a tab forgets its network. On its
  right, the search box finds nodes of the current network by their id, or by the node attribute
  chosen in the menu before it, ignoring case (Enter searches; :func:`gsk_worker.find_nodes`): a
  single node with exactly that text is selected, and the view centers on it (zooming in if nodes
  are too small to be seen); otherwise a menu lists the matching nodes, exact matches first, then
  those containing the text (at most 200), to pick one, with an item selecting all the exact
  matches when there are several (e.g. every node of a group).
* Above, the `Visual elements`_ panel on the left and the `Visualization panel`_ on the right.
* Below, over the whole width, side by side, the `Data panel`_ and the code panel (the code of
  the current tab).
* At the bottom, the status bar: on the left, the size of the current network (and how many of
  its nodes and edges are selected) or what is running; then, while a function of the library with a progress bar runs (e.g. a
  force-directed layout), the bar, with the time left; on the right, the Python environment of
  the worker, with the Python and library versions and whether a GPU is available. Click it to
  choose another environment (see `Python environments`_). The progress bars of the library
  (see :doc:`progress`) are sent by the worker (:class:`gsk_worker.ProgressBar`) at most once
  per second, and go to the status bar instead of the terminal.

Python environments
-------------------

The worker runs in a Python environment where graphoskop is installed. Clicking the environment
in the status bar lists the environments found:

* the interpreters used before, most recent first;
* the conda environments, as listed by conda in ``~/.conda/environments.txt`` (``base`` for its
  base environment);
* the active virtual environment (``$VIRTUAL_ENV``);
* the interpreters on the ``PATH``;

and **Other interpreter…** to choose one with a file dialog. Choosing one restarts the worker
there, and the networks of the open tabs are rebuilt by running their code again (a tab whose
code fails in the new environment, e.g. on a GPU it does not have, is closed with a message).
Layouts with random starts (force-directed, UMAP) can then come out different.

Sessions
~~~~~~~~

**File › Save session…** writes everything open into one file (``.gsk``), and **File › Open
session…** puts it back exactly as it was, also in another run of the interface: the tabs, in
order, with their names, their networks (all the variables of their code: networks, tables,
results of analyses and models, …, as they are, not computed again, so random results are the
same), their code (the code panel), their undo and redo histories, their styles (colors, sizes,
opacities, positions, labels, background), their views (center, zoom, rotation, scale), their
selected nodes and edges and their hidden attributes; the current tab, and the tab and table
shown in the data panel. Opening a session replaces the open networks (the interface asks first).
The file is a zip archive with ``session.json`` (the state of the interface, readable) and
``spaces.pickle`` (the variables of the tabs, pickled together, :func:`gsk_worker.session_space`;
caches are left out and made again, :data:`gsk_worker.SESSION_SKIPPED`); worker commands
``save_session`` and ``open_session``. A session saved with another version of the library opens
with a warning in the status bar. Opening a session unpickles it, which can run code: open only
sessions you trust. Networks on the GPU need a GPU to be opened. **File › Recent sessions** lists
the sessions saved or opened lately (those still on disk), and **File › Close session** closes all
the tabs.

New network
~~~~~~~~~~~

**File › New network…** opens a new tab with an empty network, directed or not (a check box), to
draw by hand: the visualization panel is an empty canvas, a right click on it adds a node there
(the first free integer id), and a right click on a node adds an edge from it to the node clicked
next (see `Visualization panel`_). The code is the empty network (worker command ``empty``,
:func:`gsk_worker.empty_code`), then the nodes and edges added, e.g.

.. code-block:: python

   import pandas as pd
   net = gsk.load_network(pd.DataFrame({"source": [], "target": []}), directed = True)
   # add_nodes / add_edges: nodes and edges added in the graphical interface
   net = gsk.add_nodes(net, 0)
   # move_nodes: positions of nodes dragged in the graphical interface
   move_nodes(net, {0: (-0.5, -0.2)})

Nodes on the empty canvas have the diameter :data:`gsk_worker.EMPTY_NODE_SIZE`, kept as nodes are
added (layouts then size them as usual); attributes are added as for any network (e.g. Edit ›
New node attribute…, or by editing values in the data panel).

Python worker
-------------

.. automodule:: gsk_worker

.. autodata:: gsk_worker.ADD_NOTE

.. autodata:: gsk_worker.ARROW_EDGE_STRIDE

.. autodata:: gsk_worker.BACKBONE_COLORS
   :no-value:

.. autodata:: gsk_worker.BACKBONE_HIDDEN
   :no-value:

.. autodata:: gsk_worker.BACKBONE_LABELS
   :no-value:

.. autodata:: gsk_worker.BACKBONE_OPTIONS
   :no-value:

.. autodata:: gsk_worker.BACKBONE_OPTIONS_PLOT
   :no-value:

.. autodata:: gsk_worker.BACKBONE_STEPS

.. autodata:: gsk_worker.BACKBONE_TAIL

.. autodata:: gsk_worker.BACKBONES
   :no-value:

.. autodata:: gsk_worker.BASIC
   :no-value:

.. autodata:: gsk_worker.BENDS

.. autodata:: gsk_worker.BLOCK_ORDER_MAX

.. autodata:: gsk_worker.BORDER_COLOR

.. autodata:: gsk_worker.BORDER_FRACTION

.. autodata:: gsk_worker.CENTRALITIES
   :no-value:

.. autodata:: gsk_worker.CENTRALITY_HIDDEN
   :no-value:

.. autodata:: gsk_worker.CHART_PALETTE

.. autodata:: gsk_worker.CHART_WIDTH

.. autodata:: gsk_worker.CHOICES
   :no-value:

.. autodata:: gsk_worker.COMMANDS
   :no-value:

.. autodata:: gsk_worker.COMMUNITIES
   :no-value:

.. autodata:: gsk_worker.CORRECTIONS
   :no-value:

.. autodata:: gsk_worker.CORRECTION_MODELS
   :no-value:

.. autodata:: gsk_worker.CURVE_STRIDE

.. autodata:: gsk_worker.CURVES

.. autodata:: gsk_worker.DASHES

.. autodata:: gsk_worker.DENDROGRAM_LABELS

.. autodata:: gsk_worker.DENDROGRAM_NODES

.. autodata:: gsk_worker.DERIVED_NOTE

.. autodata:: gsk_worker.DIRECTED_ONLY
   :no-value:

.. autodata:: gsk_worker.DISTRIBUTION_BINS

.. autodata:: gsk_worker.DISTRIBUTION_CATEGORIES

.. autodata:: gsk_worker.DISTRIBUTION_DISCRETE

.. autodata:: gsk_worker.DISTRIBUTIONS
   :no-value:

.. autodata:: gsk_worker.DIVERGING_PALETTES
   :no-value:

.. autodata:: gsk_worker.DOCS
   :no-value:

.. autodata:: gsk_worker.EDGE_COLOR

.. autodata:: gsk_worker.EDGE_STRIDE

.. autodata:: gsk_worker.ABOUT_PACKAGES

.. autodata:: gsk_worker.EDGE_STYLES

.. autodata:: gsk_worker.INTERLAYER

.. autodata:: gsk_worker.LINE_NAMES

.. autodata:: gsk_worker.SHAPE_NAMES


.. autodata:: gsk_worker.EDGE_VALUES_NOTE

.. autodata:: gsk_worker.EMBEDDING_DIMENSIONS

.. autodata:: gsk_worker.EMBEDDINGS

.. autodata:: gsk_worker.EMPTY

.. autodata:: gsk_worker.EMPTY_BOX

.. autodata:: gsk_worker.EMPTY_NODE_SIZE

.. autodata:: gsk_worker.EPIDEMIC_HIDDEN
   :no-value:

.. autodata:: gsk_worker.EPIDEMICS
   :no-value:

.. autodata:: gsk_worker.EXACT_PATHS

.. autodata:: gsk_worker.EXPLORATIONS
   :no-value:

.. autodata:: gsk_worker.EXTRA_CHOICES
   :no-value:

.. autodata:: gsk_worker.FILTERS
   :no-value:

.. autodata:: gsk_worker.FIT_REQUIRED
   :no-value:

.. autodata:: gsk_worker.FORMULA_FUNCTIONS
   :no-value:

.. autodata:: gsk_worker.GENERATORS
   :no-value:

.. autodata:: gsk_worker.HIERARCHICAL_COMMUNITIES

.. autodata:: gsk_worker.HIGHER_KINDS

.. autodata:: gsk_worker.HYPERGRAPH_STATISTICS

.. autodata:: gsk_worker.INTERFACE_FUNCTIONS
   :no-value:

.. autodata:: gsk_worker.LABEL_FRACTION

.. autodata:: gsk_worker.LAYOUTS
   :no-value:

.. autodata:: gsk_worker.LINK_CURVES
   :no-value:

.. autodata:: gsk_worker.LINK_DIMENSIONS

.. autodata:: gsk_worker.LINK_EMBEDDINGS
   :no-value:

.. autodata:: gsk_worker.LINK_HIDDEN
   :no-value:

.. autodata:: gsk_worker.LINK_PATTERN_EDGES

.. autodata:: gsk_worker.LINK_PREDICTIONS
   :no-value:

.. autodata:: gsk_worker.LINK_SUPPORT

.. autodata:: gsk_worker.LISTS
   :no-value:

.. autodata:: gsk_worker.GNN_CURVES
   :no-value:

.. autodata:: gsk_worker.GNN_HIDDEN
   :no-value:

.. autodata:: gsk_worker.GNN_LAYERS

.. autodata:: gsk_worker.GNNS
   :no-value:

.. autodata:: gsk_worker.GROUPS_NOTE

.. autodata:: gsk_worker.HIDDEN
   :no-value:

.. autodata:: gsk_worker.HIERARCHIES
   :no-value:

.. autodata:: gsk_worker.HIERARCHY_HIDDEN
   :no-value:

.. autodata:: gsk_worker.HINTS
   :no-value:

.. autodata:: gsk_worker.INTEGERS
   :no-value:

.. autodata:: gsk_worker.LOGIC

.. autodata:: gsk_worker.LOOP_SIZE

.. autodata:: gsk_worker.MATRIX_SHOWN

.. autodata:: gsk_worker.MAX_BREAK_VALUES

.. autodata:: gsk_worker.MAX_CELLS

.. autodata:: gsk_worker.MAX_ELEMENTS

.. autodata:: gsk_worker.MAX_FOUND

.. autodata:: gsk_worker.MAX_ORDER_ATTRIBUTES

.. autodata:: gsk_worker.MAX_SLICES

.. autodata:: gsk_worker.MEASURES
   :no-value:

.. autodata:: gsk_worker.MAX_SEGMENTS

.. autodata:: gsk_worker.MATCHING_NOTE

.. autodata:: gsk_worker.MATRIX_NAMES

.. autodata:: gsk_worker.MATRIX_VALUES

.. autodata:: gsk_worker.MAX_SUGGESTIONS

.. autodata:: gsk_worker.MAX_UNDO

.. autodata:: gsk_worker.MAX_VALUES

.. autodata:: gsk_worker.MINERS
   :no-value:

.. autodata:: gsk_worker.MISSING_COLOR

.. autodata:: gsk_worker.MODULARITY_LABELS

.. autodata:: gsk_worker.MODULARITY_METHODS
   :no-value:

.. autodata:: gsk_worker.MOVE_NOTE

.. autodata:: gsk_worker.NESTEDNESS_CELLS

.. autodata:: gsk_worker.NODE_COLOR

.. autodata:: gsk_worker.NODE_SHAPES

.. autodata:: gsk_worker.NODE_SIZES_NOTE

.. autodata:: gsk_worker.NODE_STRIDE

.. autodata:: gsk_worker.NODE_WITH_NOTE

.. autodata:: gsk_worker.NODES
   :no-value:

.. autodata:: gsk_worker.NUMBERS
   :no-value:

.. autodata:: gsk_worker.OPTIONS
   :no-value:

.. autodata:: gsk_worker.PATHS_TABLE

.. autodata:: gsk_worker.PLOT_NOTE

.. autodata:: gsk_worker.PROGRESS_INTERVAL

.. autodata:: gsk_worker.PLOT_COLORS
   :no-value:

.. autodata:: gsk_worker.PLOT_ELEMENTS
   :no-value:

.. autodata:: gsk_worker.PLOT_FONTS

.. autodata:: gsk_worker.PLOT_OPTIONS
   :no-value:

.. autodata:: gsk_worker.PLOT_STYLES

.. autodata:: gsk_worker.PROJECTION_HIDDEN
   :no-value:

.. autodata:: gsk_worker.PROJECTIONS
   :no-value:

.. autodata:: gsk_worker.RANGES
   :no-value:

.. autodata:: gsk_worker.ROBUSTNESS
   :no-value:

.. autodata:: gsk_worker.ROBUSTNESS_HIDDEN
   :no-value:

.. autodata:: gsk_worker.ROBUSTNESS_QUALITIES
   :no-value:

.. autodata:: gsk_worker.SESSION_FORMAT

.. autodata:: gsk_worker.SESSION_SKIPPED
   :no-value:

.. autodata:: gsk_worker.SET_OPERATIONS

.. autodata:: gsk_worker.SHAPE_ALIASES

.. autodata:: gsk_worker.SHOWN_DEFAULTS
   :no-value:

.. autodata:: gsk_worker.SIMPLICIAL_STATISTICS

.. autodata:: gsk_worker.STATE_COLORS

.. autodata:: gsk_worker.STATISTICS
   :no-value:

.. autodata:: gsk_worker.STATISTIC_HIDDEN
   :no-value:

.. autodata:: gsk_worker.STRINGS
   :no-value:

.. autodata:: gsk_worker.STYLE_LABELS
   :no-value:

.. autodata:: gsk_worker.SUMMARY_ATTRIBUTES

.. autodata:: gsk_worker.SYNCHRONIZATION
   :no-value:

.. autodata:: gsk_worker.SYNCHRONIZATION_HIDDEN
   :no-value:

.. autodata:: gsk_worker.TABLE_ATTRIBUTES

.. autodata:: gsk_worker.TAILS
   :no-value:

.. autodata:: gsk_worker.TRANSFORMATION_HIDDEN
   :no-value:

.. autodata:: gsk_worker.TRANSFORMATIONS
   :no-value:

.. autodata:: gsk_worker.TRANSFORMS
   :no-value:

.. autodata:: gsk_worker.TYPE1_COLOR

.. autodata:: gsk_worker.VECTOR_HIDDEN
   :no-value:

.. autodata:: gsk_worker.VECTORS
   :no-value:

.. autodata:: gsk_worker.VIEW_SIZES
   :no-value:

.. autofunction:: gsk_worker.add_code

.. autofunction:: gsk_worker.adding

.. autofunction:: gsk_worker.all_statistics

.. autofunction:: gsk_worker.animation

.. autofunction:: gsk_worker.arrows

.. autofunction:: gsk_worker.attribute_values

.. autofunction:: gsk_worker.about

.. autofunction:: gsk_worker.attributes

.. autofunction:: gsk_worker.backbone_code

.. autofunction:: gsk_worker.backbone_plot

.. autofunction:: gsk_worker.backbone_range

.. autofunction:: gsk_worker.backbone_scale


.. autofunction:: gsk_worker.block_order

.. autofunction:: gsk_worker.category_codes

.. autofunction:: gsk_worker.centrality_code

.. autofunction:: gsk_worker.breaks

.. autofunction:: gsk_worker.chart_slices

.. autofunction:: gsk_worker.colors

.. autofunction:: gsk_worker.correction_code

.. autofunction:: gsk_worker.correction_signature

.. autofunction:: gsk_worker.default_style

.. autofunction:: gsk_worker.drawn_visuals

.. autofunction:: gsk_worker.complete_positions

.. autofunction:: gsk_worker.file_style

.. autofunction:: gsk_worker.label_inks

.. autofunction:: gsk_worker.layer_values


.. autofunction:: gsk_worker.degree_plot

.. autofunction:: gsk_worker.dendrogram_plot

.. autofunction:: gsk_worker.dendrogram_signature

.. autofunction:: gsk_worker.dendrogram_tree

.. autofunction:: gsk_worker.derived

.. autofunction:: gsk_worker.distinct

.. autofunction:: gsk_worker.distribution

.. autofunction:: gsk_worker.distribution_plot

.. autofunction:: gsk_worker.drawing

.. autofunction:: gsk_worker.edge_values

.. autofunction:: gsk_worker.edit_code

.. autofunction:: gsk_worker.embedding_code

.. autofunction:: gsk_worker.empty_code

.. autofunction:: gsk_worker.environment

.. autofunction:: gsk_worker.epidemic_plot

.. autofunction:: gsk_worker.exploration_code

.. autofunction:: gsk_worker.exploration_signature

.. autofunction:: gsk_worker.find_nodes

.. autofunction:: gsk_worker.fit_code

.. autofunction:: gsk_worker.formula_name

.. autofunction:: gsk_worker.formula_summary

.. autofunction:: gsk_worker.formula_values

.. autofunction:: gsk_worker.free_id

.. autofunction:: gsk_worker.generator_code

.. autofunction:: gsk_worker.gnn_code

.. autofunction:: gsk_worker.gnn_evaluation

.. autofunction:: gsk_worker.gnn_plot

.. autofunction:: gsk_worker.gnn_signature

.. autofunction:: gsk_worker.gnn_text

.. autofunction:: gsk_worker.gradient_colors

.. autofunction:: gsk_worker.incidence_values

.. autofunction:: gsk_worker.interface_import

.. autofunction:: gsk_worker.label_sizes

.. autofunction:: gsk_worker.last_layout

.. autofunction:: gsk_worker.groups

.. autofunction:: gsk_worker.hierarchy_code

.. autofunction:: gsk_worker.hierarchy_signature

.. autofunction:: gsk_worker.histories

.. autofunction:: gsk_worker.layout_code

.. autofunction:: gsk_worker.level_attributes

.. autofunction:: gsk_worker.link_plot

.. autofunction:: gsk_worker.link_prediction_code

.. autofunction:: gsk_worker.link_signature

.. autofunction:: gsk_worker.link_split_code

.. autofunction:: gsk_worker.link_text

.. autofunction:: gsk_worker.literal

.. autofunction:: gsk_worker.matches

.. autofunction:: gsk_worker.matching

.. autofunction:: gsk_worker.mark_advanced

.. autofunction:: gsk_worker.matrix_order

.. autofunction:: gsk_worker.matrix_orders

.. autofunction:: gsk_worker.matrix_plot

.. autofunction:: gsk_worker.load_code

.. autofunction:: gsk_worker.main

.. autofunction:: gsk_worker.measures

.. autofunction:: gsk_worker.mining_code

.. autofunction:: gsk_worker.mining_signature

.. autofunction:: gsk_worker.modularity_code

.. autofunction:: gsk_worker.move_nodes

.. autofunction:: gsk_worker.natural_breaks

.. autofunction:: gsk_worker.nestedness_code

.. autofunction:: gsk_worker.nestedness_matrix

.. autofunction:: gsk_worker.nestedness_plot

.. autofunction:: gsk_worker.nestedness_text

.. autofunction:: gsk_worker.node_borders

.. autofunction:: gsk_worker.node_labels

.. autofunction:: gsk_worker.node_matrix

.. autofunction:: gsk_worker.node_row

.. autofunction:: gsk_worker.node_size

.. autofunction:: gsk_worker.node_sizes

.. autofunction:: gsk_worker.node_values

.. autofunction:: gsk_worker.node_with

.. autofunction:: gsk_worker.numbers

.. autofunction:: gsk_worker.opacities

.. autofunction:: gsk_worker.order_source

.. autofunction:: gsk_worker.other_code

.. autofunction:: gsk_worker.paths_argument

.. autofunction:: gsk_worker.paths_code

.. autofunction:: gsk_worker.paths_signature

.. autofunction:: gsk_worker.pick_grid

.. autofunction:: gsk_worker.plot_code

.. autofunction:: gsk_worker.plot_figure

.. autofunction:: gsk_worker.positions

.. autofunction:: gsk_worker.preview

.. autofunction:: gsk_worker.prune_style

.. autofunction:: gsk_worker.randomization_plot

.. autofunction:: gsk_worker.randomization_text

.. autoclass:: gsk_worker.ProgressBar

.. autofunction:: gsk_worker.robustness_code

.. autofunction:: gsk_worker.robustness_plot

.. autofunction:: gsk_worker.robustness_runs

.. autofunction:: gsk_worker.robustness_signature

.. autofunction:: gsk_worker.session_space

.. autofunction:: gsk_worker.set_edge_values

.. autofunction:: gsk_worker.shape_polygon

.. autofunction:: gsk_worker.shown_attributes

.. autofunction:: gsk_worker.shown_edges

.. autofunction:: gsk_worker.signature

.. autofunction:: gsk_worker.simulation_code

.. autofunction:: gsk_worker.size_code

.. autofunction:: gsk_worker.sort_rows

.. autofunction:: gsk_worker.spread

.. autofunction:: gsk_worker.sizes

.. autofunction:: gsk_worker.spectrum_plot

.. autofunction:: gsk_worker.spectrum_text

.. autofunction:: gsk_worker.statistic_code

.. autofunction:: gsk_worker.statistic_signature

.. autofunction:: gsk_worker.styled_lines

.. autofunction:: gsk_worker.subdivisions

.. autofunction:: gsk_worker.summary

.. autofunction:: gsk_worker.synchronization_code

.. autofunction:: gsk_worker.synchronization_curves

.. autofunction:: gsk_worker.synchronization_runs

.. autofunction:: gsk_worker.synchronization_signature

.. autofunction:: gsk_worker.tab_tables

.. autofunction:: gsk_worker.text

.. autofunction:: gsk_worker.too_large

.. autofunction:: gsk_worker.transform

.. autofunction:: gsk_worker.tree_signature

.. autofunction:: gsk_worker.value_code

.. autofunction:: gsk_worker.vector_code

.. autofunction:: gsk_worker.vector_hidden

.. autofunction:: gsk_worker.vector_image

.. autofunction:: gsk_worker.verdict_text

.. autofunction:: gsk_worker.widget

Running the interface
---------------------

The interface needs Godot 4.7 and a Python environment with the dependencies of graphoskop
(PyTorch, PyTorch Geometric, NumPy, SciPy, pandas, Numba, tqdm), and matplotlib for plots
(``pip install graphoskop[gui]`` installs it). graphoskop itself does not
need to be installed: the worker imports the library from the source tree that holds the
``gui`` folder, so that the interface and the library always match, even when another version of
graphoskop is installed in the environment. Without the source tree (e.g. an exported
interface), it imports the installed one. Open ``gui/project.godot`` in the Godot editor
and run it, or run it directly:

.. code-block:: text

   godot --path gui

The worker is started with the interpreter chosen before (see `Python environments`_), or else
the one in the environment variable ``GRAPHOSKOP_PYTHON``, or else ``python3`` (``python`` on
Windows). If it cannot start or cannot import graphoskop, a message offers to choose the
interpreter, e.g. ``~/miniconda3/envs/gsk/bin/python``.

While the worker starts, a splash screen shows: the window opens small (905 by 372 pixels) and
without borders, with the boot splash of the project (``application/boot_splash/image``,
``res://icon.png``, 825 by 292 pixels, at its own size on white: no stretching
(``boot_splash/stretch_mode`` 0) nor filtering (``boot_splash/use_filter`` off; the interface
draws it with the nearest filter), centered with 40 pixels around it, so that it shows pixel
for pixel and whole; another image needs a window larger by an even number of pixels in each
direction, for whole-pixel centering), and the interface keeps it there, with "Starting
Python…" under it, until the worker has imported the library; then the window takes its borders
and, a few frames later (once the window manager has applied them), its size and state of the last
session, and comes in front of the other windows with the focus (as it does when it opens, and
before the messages and the file dialog to choose Python, which would otherwise open behind other
windows when the interface is behind them): maximized if it was, else its last size; on the first start, 1440 by 900 pixels
(``display/window/size/viewport_width`` and ``viewport_height``), centered, maximized on smaller
screens. If Python cannot start, its message shows on top of the splash; once it is dismissed
without another interpreter starting, the splash goes, so that one can be chosen from the status
bar. The size of the splash is ``display/window/size/window_width_override`` and
``window_height_override`` in ``project.godot``.

Runs from the Godot editor (F5) skip the small window: the ``.editor`` overrides of these settings
in ``project.godot`` open it at 1440 by 900 pixels with borders, with the splash drawn inside it,
and the interface leaves its size to the editor (which restores it when it gets the focus back) and
does not store it for the exported interface.

**Exporting the interface** (Project › Export… in the Godot editor): Godot exports only the
resources it knows, so the plugin ``addons/gsk_export`` (enabled in ``project.godot``) adds the
worker: it puts ``python/gsk_worker.py`` inside the package and next to the exported executable
(``python/gsk_worker.py``): in a folder beside it, or, with Export Project… to a ``.zip`` file,
inside the archive (no filter of the export preset is needed). The plugin is loaded when the
project is opened: after an update, reopen the project in the editor (or check that "Graphoskop
export" is on in Project › Project Settings › Plugins). The exported interface runs the copy
next to it; without one (e.g. a macOS application bundle), it copies the worker out of the package
to ``user://python/`` (the user data folder, see `Settings`_), since Python cannot read inside the
package. Python and its environment are not part of the export: they are chosen as above.

The title of the window is ``WINDOW_TITLE`` of ``main.gd`` ("Graphoskop 6054"; the name of the
project, ``application/config/name``, stays "Graphoskop", since it names the user data folder).
Debug builds add " (DEBUG)" to it: the runs from the editor, and the exports made with **Export
With Debug** checked in the Export Project… file dialog (the default); uncheck it for the release
build, which is also faster.

The interface uses Godot's Compatibility renderer (OpenGL 3.3, OpenGL ES 3.0 or WebGL 2), which
runs on any GPU of the last fifteen years; its file dialogs are those of the operating system.

Selecting
---------

Nodes and edges are selected in the visualization panel (click, Shift-click, Shift-drag a
rectangle; see `Visualization panel`_), and with the **Select** menu:

* **Select all nodes**, **Clear selection**;
* **Neighbors of the selection** (Ctrl+6, Command+6 on macOS): the selected nodes, the endpoints
  of the selected edges and all their neighbors (along edges in either direction) become the
  selection; pressed again, it grows one more step, until the whole components are selected (the
  status bar says how many nodes were added; worker command ``neighbors``). Selected edges stay
  selected;
* **Select by attribute…**: a dialog selecting the nodes, or the edges, that satisfy a chain of
  conditions. A condition compares an attribute with a value: equal to (=), different from (≠),
  smaller than (<, ≤) or larger than (>, ≥), or contains a text (a part of the text, or one of the
  values of a node with several). Numeric attributes take a number; qualitative ones a value from
  the menu of their values (or a text, when they have too many). **Add condition** adds a row,
  which starts with how it chains to the rows above: **AND** (both), **OR** (either) or **XOR**
  (one but not both); rows are applied from top to bottom, so ``k > 6``, AND ``half = 1``, OR
  ``age < 30`` selects (k > 6 and half = 1) or age < 30. **×** removes a row (the last one stays).
  The selection replaces the current one, or, with **Add to the current selection**, is added to
  it. If nothing matches, the dialog says so and stays open (worker command ``select``, with
  :func:`gsk_worker.matches` for each condition and :data:`gsk_worker.LOGIC` for the chaining).

Self loops are drawn as circles above their node, a little more than half as wide as the node, as
thick as the edges and in their colors, which follow the node when it is dragged (in matrix views
they are the cells of the diagonal).

In directed networks, edges end with an arrowhead in the color of their line, its tip on the
border of the node it points to (2.5 times as long as the line is wide, at least 5 pixels;
``arrow.gdshader``, from a buffer of the worker, :func:`gsk_worker.arrows`), and straight lines
stop at the base of their arrowheads (``edge.gdshader``, from the diameters of their nodes in
the edge buffer, :data:`gsk_worker.ARROW_EDGE_STRIDE`; also while nodes are dragged or the
layout is scaled). Reciprocal edges
(and parallel ones) are drawn as one line, the first of the edges between the two nodes, with an
arrowhead at each end that has an edge pointing to it (:func:`gsk_worker.shown_edges`); the other
edges get no width, but can still be selected with the rectangle. Bundled
edges have their arrowheads along the end of their curve (the curve still runs to the node,
under the arrowhead); matrix views have none (rows are
sources, columns targets). Arrowheads follow dragged nodes.

**Adding nodes and edges.** Right clicking a node opens a menu with **Add an edge from …**: a line
then follows the mouse from the node, and clicking another node (or the same one, for a self
loop) adds the edge; clicking elsewhere, right clicking or Escape cancels it. Right clicking the
background opens **Add a node here**, and the node is added where the mouse was. Nothing is asked:
the new node gets the first integer from 0 up that is not an id yet (:func:`gsk_worker.free_id`;
a number if the ids are numbers, else a text), the attributes of new nodes and edges are missing
(edge weights are 1), and they can be filled in the `Data panel`_. In multilayer networks the menu
has one item per layer, in bipartite networks one per node type (:func:`gsk_worker.adding`).
Nodes are added only to node-link drawings positioned by a layout (not to matrix views, nor when
the positions come from node attributes), and neither nodes nor edges to hypergraphs and
simplicial complexes. Both are lines of code, which Edit › Undo takes back
(:func:`gsk_worker.add_code`), e.g.

.. code-block:: python

   # add_nodes / add_edges: nodes and edges added in the graphical interface
   net = gsk.add_nodes(net, 0)
   # move_nodes: positions of nodes dragged in the graphical interface
   move_nodes(net, {0: (1.5, -0.2)})
   net = gsk.add_edges(net, ("a", 0))

**Smooth edges in saved images.** On screen, edges and arrowheads are drawn without antialiasing
(nodes and self loops are smooth, by their shaders): the Compatibility renderer of Godot, which
the interface uses so that it runs on any graphics card, has no multisample antialiasing for 2D
drawings, and smoothing every frame would cost time on large networks. Saved images and the frames
of exported animations are smooth: they are drawn three times larger (``SMOOTH`` of
``network_view.gd``, with the smallest sizes on screen three times larger too, so that they look
the same), then reduced with a Lanczos filter. Selected nodes are circled (all of them, by the GPU: ``ring.gdshader``) and selected edges drawn
over in black (lines, curves of bundled edges,
outlines of the cells of matrix views), at any zoom; the data panel shows their rows. The selection is what the `Editing`_ operations work on.

Settings
--------

The interface remembers, between sessions, in the file ``settings.cfg`` of its user data folder
(``~/.local/share/godot/app_userdata/Graphoskop`` on Linux, ``~/Library/Application
Support/Godot/app_userdata/Graphoskop`` on macOS, ``%APPDATA%\Godot\app_userdata\Graphoskop``
on Windows):

* the Python interpreter of the worker, and the ten used last;
* the folder of the last edge table and of the last node table. The edge table dialog opens in
  the folder of the last edge table; the node table dialog in the folder of the edge table chosen
  in the load dialog (the most likely to hold its node table), or else in that of the last node
  table;
* the separator, header and device chosen last in the load dialog;
* the size of the window, and whether it was maximized;
* the look of the interface, chosen in **File › Settings…** (last item of the File menu): the
  mode, **Light** (the default) or **Dark**, and the **Colors**, one of the ColorBrewer sequential
  palettes Blues (the default), Greens, Greys, Purples, Reds or Oranges (each shown with its nine
  colors). Choices apply at once to the whole interface, its dialogs and menus (the theme is built
  in code by ``ui_theme.gd`` on the fonts of ``theme.tres``: in light mode, the lightest colors of
  the palette for backgrounds, panels and buttons and its darkest for the text; in dark mode,
  darkened and less saturated shades of its eighth color, with a light one for the text; check
  boxes drawn in the colors of the palette and the icons of arrows and close buttons recolored as
  the text), and to the colors of the code (those of Visual Studio Code's Light+ and Dark+
  themes). The drawing of a network keeps its own background (Visual elements › Network), white
  by default; the file dialogs are those of the operating system.

Deleting the file resets them.

The interface spends little CPU when nothing happens: it redraws only what changes (low
processor mode), its main loop checks for input and for replies of the worker 60 times a second
(``application/run/low_processor_mode_sleep_usec`` in ``project.godot``; Godot's default, made
for editors, is 145), 10 times a second when no window of the interface has the focus
(``BACKGROUND_SLEEP_USEC`` of ``main.gd``), and it opens no audio device (the dummy audio
driver: the interface has no sound, and an open audio stream wakes the CPU all the time). The
size of the network makes no difference while nothing moves (the drawing is rendered once, then
kept).

Simulation
----------

**Simulation › Epidemics…** opens the dialog of the epidemic models of the current network.
**Add model** lists every epidemic model of the library (see :doc:`epidemics`): SI, SIS, SIR,
their complex contagion versions (and the complex nonprogressive model) and their higher-order
versions (for hypergraphs and simplicial complexes). Each opens its parameter dialog, whose first
parameter is the name of the run (``SIR 1``, ``SIR 2``, …, proposed); **Run** runs the model and
stores its table of states in the dictionary ``epidemics``, network-wide (shown in the Tables tab
of the `Data panel`_), e.g.

.. code-block:: python

   epidemics = {}
   epidemics["SIR 1"] = gsk.sir_model(net, beta = 0.2, mu = 0.05)
   epidemics["SIR 2"] = gsk.sir_model(net, beta = 0.05, mu = 0.05)

(:func:`gsk_worker.simulation_code`). The states of the last run are also a node attribute. The
**onsets** switch, on by default, adds the step at which every node was first infected in the
last run, as the numeric node attribute ``state_first_I``, and in SIR models the step at which it
first recovered, ``state_first_R`` (missing for nodes that never were; ``onsets = True``), e.g.
to color the nodes by when the epidemic reached them. Any
number of runs, of any models, can be added. While a model runs, the progress bar of the task
dialog follows its runs (the library shows a bar for the epidemic models, see :doc:`progress`),
and a line under the runs says what is going on, until the worker is done with the run (the
model, then the attributes of the network and the history of the spread). The runs are listed
with check boxes, and with **Copy…**, which opens the parameter dialog of the same model with
the values of that run (and the next free name), to change some of them and compare. **Plot**
draws the checked runs in one plot, for the checked **States** (e.g. ``I``, the infected, or
``S``, ``I`` and ``R`` together): the fraction of the nodes in each state at each step, one
series per run and state with its confidence interval as a band, to compare how fast each model
spreads on the same network, and how high its peak is (the plot window gives the peak of every
series; their colors, styles, the axes and the font can be changed as for the degree
distribution, see `Analysis`_; :func:`gsk_worker.epidemic_plot`). The dialog stays open while the parameter dialogs and the plot
open above it (they are its child windows; the plot window stays open when the dialog closes).

The **history** switch of the parameter dialog, on by default, also stores the state of every
node at every step of the last run of the model (``history = True``): the qualitative node
attributes ``state_0``, ``state_1``, …, replacing those of an earlier run. Then a player appears
at the bottom of the visualization panel, to see the epidemic spread through the network:

* the name of the run;
* buttons as on a record player: back to step 0, one step back, play / pause, one step forward,
  to the last step;
* a timeline slider, with the step shown and the last one;
* the speed of play, in steps per second;
* the color of every state (S blue, I red, R green, V purple by default), with color buttons;
* **States**: on, the nodes have the color of their state at the step shown; off, the colors of
  the style;
* **Export…**: saves the whole spread, from step 0 to the last step, as an animation at the
  speed of play: a GIF (``.gif``), or an MP4 video (``.mp4``, with ffmpeg on the ``PATH`` or the
  ``imageio-ffmpeg`` package). Every step is drawn as on screen (with smooth edges, as saved
  images, see `Visualization panel`_), with its number in the top left corner
  (:func:`gsk_worker.animation`).

Using any button or the timeline turns States on; the colors of the style come back when it is
turned off. The player only changes the colors on screen: the style, the network and the code do
not change. The worker sends the states at step 0 and the changes of every step
(:func:`gsk_worker.spread`), and the interface colors only the nodes that change, so playing is
fast on large networks. Nodes without a state (other components, hyperedges and facets of
incidence networks) keep the colors of the style.

**Simulation › Exploration** explores the network from one node (see :doc:`exploration`):
**Breadth-first search…** (:func:`~graphoskop.bfs`), **Depth-first search…**
(:func:`~graphoskop.dfs`), **Random walk…** (:func:`~graphoskop.random_walk`) and
**Metropolis-Hastings random walk…** (:func:`~graphoskop.metropolis_hastings_random_walk`). Each
opens its parameter dialog: the source node (the selected node, if one is selected; with **by**,
the value of a node attribute instead of the id, e.g. a name: exactly one node must have it,
:func:`gsk_worker.node_with`, a function of the interface, :data:`gsk_worker.NODE_WITH_NOTE`,
written as ``gsk.bfs(net, node_with(net, "name", "alice"))``; **Find**, or Enter in the field,
lists the values of that attribute, or the ids, that contain the text typed, exact ones first, as
the search box does: choosing one writes it in the field, and values that several nodes have are
listed but cannot be chosen; worker command ``suggest``, at most
:data:`gsk_worker.MAX_SUGGESTIONS` values), and for
searches the largest depth, for walks their length (steps), their number (all from the source)
and the weights; the direction in directed networks; the seed of the walks is 1 by default. The
result is one node attribute (:func:`gsk_worker.exploration_code`): for the searches, the order
of the visit (``bfs_order``, ``dfs_order``: 0 for the source, 1 for the next node visited, …);
for the walks, the step of the first visit of each node, averaged over the walks that reach it
(``rw_step``, ``mhrw_step``; in Metropolis-Hastings walks the steps count the refused moves too).
Nodes never reached get no value, and the status bar says how many were reached, e.g.

.. code-block:: python

   exploration = gsk.random_walk(net, "alice", 100, walks = 10, seed = 1)
   first = exploration.groupby(["walk", "node"]).step.min()   # first visit of each node in each walk
   net.set_node_attribute(first.groupby(level = "node").mean().rename("rw_step").reset_index(), node_column = "node")

As every analysis, the exploration does not change the look of the drawing: color or size the
nodes by the attribute to see it. Explorations are for plain networks (not multilayer networks,
hypergraphs or simplicial complexes).

**Simulation › Robustness…** measures how the network falls apart (see :doc:`robustness`), in a
dialog of runs like the epidemics dialog (``robustness_dialog.gd``): **Add run** offers **Node
removal…** (:func:`~graphoskop.node_removal`), **Edge removal…**
(:func:`~graphoskop.edge_removal`) and **Cascading failures…**
(:func:`~graphoskop.cascade_failures`). The parameter dialog asks the name of the run (by default
the function and its strategy, e.g. ``node_removal(degree)``, numbered if taken), the strategy (a menu: random
failures, the friend of a random node, ``degree`` for the largest degrees first, or a numeric
attribute, largest values first; for cascades also ``overload``), the quality (the fraction of
the nodes in the largest component, or the average shortest path length) and the number of runs;
cascades also the loads and capacities (random, or a numeric attribute), the number of initial
failures and whether nodes or edges fail; with Advanced, the rest (selection, points, samples,
seed, …). The table of the run is stored in the dictionary ``robustness`` by name, and its global
value becomes a network attribute of the same name (:func:`gsk_worker.robustness_code`): for
removals the mean quality over all the numbers of removals (with the largest component, the
robustness index R, the area under the curve), for cascades the final fraction of failed items,
e.g.

.. code-block:: python

   robustness = {}   # runs of Simulation › Robustness, by name
   robustness["node_removal(degree)"] = gsk.node_removal(net, strategy = gsk.degree(net, distribution = None))
   net.network_attributes["node_removal(degree)"] = float(robustness["node_removal(degree)"]["lcc"].mean())   # a network attribute: mean quality over the removals

The dialog lists the runs of the tab (:func:`gsk_worker.robustness_runs`, worker command
``robustness_runs``), each with a check box and **Copy…** (the parameter dialog of the same
function with the values of that run, to change some); the line under them shows the global value
of the last run. **Plot** opens the plot window with the checked runs (worker command
``robustness_plot``): for removals, of nodes or edges, the quality against the fraction removed;
for cascades, the fraction of items failed at every step; one line per run with its confidence
interval as a band (:func:`gsk_worker.robustness_plot`), and their global values under the plot.
Removals and cascades, or runs of two qualities, do not share axes: the dialog asks to plot them
apart. Like every plot, its colors, axes and font can be changed and it can be saved. The dialog
is not exclusive: it stays open, with the parameter dialog and the plot above it.

**Simulation › Synchronization…** runs the coupled oscillators of the library (see
:doc:`synchronization`) in a dialog of runs like the robustness dialog: **Add run** offers
**Kuramoto…** (:func:`~graphoskop.kuramoto_model`, phases on the nodes), **Simplicial Kuramoto…**
(:func:`~graphoskop.simplicial_kuramoto_model`, phases on the ``k``-simplices, with the explosive
variant) and **Node-link Kuramoto…** (:func:`~graphoskop.node_link_kuramoto_model`, phases on
nodes and edges at once). The parameter dialog asks the name of the run (``Kuramoto 1``, …,
proposed), the coupling strengths as a range (**sigma_from**, **sigma_to**, **steps**, evenly
spaced), **backward** (the range up and then down again, adiabatically, each strength starting
from the phases of the previous one: the two branches show the hysteresis of an explosive
transition), the number of runs and the parameters of the model; with Advanced, the natural
frequencies (mean, standard deviation), times and seed (1 by default)
(:func:`gsk_worker.synchronization_signature`). The table of the run (the order parameters by
strength and run) is stored in the dictionary ``synchronization`` by name
(:func:`gsk_worker.synchronization_code`), e.g.

.. code-block:: python

   synchronization = {}   # runs of Simulation › Synchronization, by name
   up = np.linspace(0.0, 1.0, 21)   # the strengths up, then down again
   synchronization["Kuramoto 1"] = gsk.kuramoto_model(net, np.r_[up, up[::-1]], runs = 10, seed = 1, adiabatic = True)

**Plot** draws the transition curves of the checked runs: the order parameter against the
coupling strength, the mean over the runs with a band of one standard deviation, one curve per
order parameter of the model (nodes, edges, …) and per branch (``down``)
(:func:`gsk_worker.synchronization_curves`).

Visual elements
---------------

The visual elements panel, left of the visualization panel, sets the look of the drawing of the
current network, in three tabs. In the Nodes and Edges tabs the elements come in groups (nodes:
**Fill**: color, opacity, chart; **Shape**: shape, size; **Border**: color, width; **Label**:
text, size, color; **Placement**: position; edges: **Line**: color, thickness, opacity, style),
each a row with its name and its value: a click anywhere on the row edits it.

In multilayer networks, the attributes offered for nodes and edges (colors, shapes, line styles,
…) include ``layer`` (unless the network has an attribute of that name): the layer of each node,
and of each edge, ``interlayer`` for the edges between layers (:func:`gsk_worker.layer_values`).
It is drawing only: the code does not change.

* **Nodes**: their color, size, opacity, position and label. By default all nodes are ``#e41a1c``; in bipartite networks
  (and incidence networks of hypergraphs and complexes) the second type of nodes is ``#377eb8``,
  the next color of ColorBrewer's Set1 palette. **Position** places the nodes by the layout (the
  default), or takes their x and y coordinates from numeric node attributes (e.g. longitude and
  latitude), each axis on its own (:func:`gsk_worker.positions`). The attributes override the
  layout, the one computed and any later one, until the layout is chosen again; the view is fitted
  to the new positions. Nodes without a value of the attribute are put at its smallest value.
  Like the other visual elements, this changes the drawing only, not the network or the code; while
  positions come from attributes, nodes cannot be dragged, and bundled edges are drawn straight
  (their curves follow the layout). **Label** (none by default) writes the id of every node
  (**node id**), or the value of a node attribute, chosen in the menu that opens under its row,
  centered on every node, upright, in black or
  white (by the background) with an outline (:func:`gsk_worker.node_labels`). **Label size** sets
  their height, like node sizes (see `Sizes`_): one size for all, or sizes given by a node
  attribute, from the smallest to the largest; 1 × is half the default node diameter
  (:func:`gsk_worker.label_sizes`). Labels are sized in the drawing, as nodes are, so they grow
  and shrink with the zoom; labels smaller than 4 pixels on screen are left out, and only the
  nodes in view are labeled, when there are at most 2,000 of them: zoom in to see the labels of a
  large network. Saved images have them too. **Chart** (none by default) turns the nodes into pie
  charts, from the top, clockwise: **Pie of numeric attributes** (check boxes: a slice per
  attribute, as large as its share of their sum on the node; missing and negative values count
  0, nodes whose values sum to 0 keep their color), or **Pie of the values of a qualitative
  attribute** (equal slices of the values of each node: several for nodes with many, such as the
  overlapping communities of DEMON or BigCLAM, the whole disc for nodes with one), in the colors
  of a qualitative ColorBrewer **Palette** (Set1 by default; the k-th attribute, or the k-th value,
  has the k-th color), and each slice can take another color: the dialog lists the attributes, or
  the values (those of nodes with several split), each with a color button (values beyond those
  listed keep the palette). Charts keep the size and opacity of the nodes, at most
  :data:`gsk_worker.MAX_SLICES` slices each (:func:`gsk_worker.chart_slices`); the node shader
  draws them from a texture of the slices, so they cost nothing more to move and zoom. Epidemic
  states, when shown, color the nodes instead. Vector PDF images draw them as wedges.
  **Shape** (circles by default): one shape for all, or one per value of a qualitative node
  attribute (or of a numeric one with at most 20 values, such as communities), among circle,
  triangle, square, diamond, pentagon, hexagon, heptagon, octagon and a five-pointed star
  (:data:`gsk_worker.NODE_SHAPES`), in a menu per value (values beyond those the interface lists,
  and missing ones, take **Others**; :func:`gsk_worker.category_codes`). Shapes are inscribed in
  the disc of the node (their vertices on its circle), and turn with the view; charts take the
  shape of their node. **Border color** (black by default) and **Border width** (0: no border)
  draw a border inside the shape of every node, so that its size does not change: one color or
  colors from a node attribute (as node colors), one width or widths from a node attribute (as
  node sizes; 1 × is a tenth of the radius; :func:`gsk_worker.node_borders`). **Label color** is
  edited as the node colors (the same dialog): one color, or colors from a node attribute
  (qualitative or quantitative scales, palettes, colors of single values; sent with the label
  heights, :func:`gsk_worker.label_inks`). Until it is edited it is automatic, black or white by
  the background; the halo around the letters of each label is white or black, whichever stands
  out from its color. **Opacity** (above) sets the opacity of the
  nodes, one for all or from a node attribute. The node shader draws shapes and borders, so they
  cost nothing to pan and zoom; vector images draw them as polygons (charts stay round there).
* **Edges**: their color (by default ``#999999``), thickness and opacity, and **Line style**
  (solid by default): one for all, or one per value of a qualitative edge attribute (or of a
  numeric one with few values), among solid, dash, dash-dot, dots, parallel lines, sine wave and
  zigzag (:data:`gsk_worker.EDGE_STYLES`; the dot of dash-dot in the middle of its gap), drawn in units of the width of each line by the edge
  shaders (straight and bundled or routed edges; self loops and matrix cells stay solid). Vector
  images draw dashes and dots as dash patterns, the others as polylines
  (:func:`gsk_worker.styled_lines`).
* **Network**: the background color; **Rotation**, a slider turning the view (-180° to 180°,
  counterclockwise) without changing the positions or the code; **Scale**, a logarithmic slider
  (and a number) from a tenth to a hundred times, which blows up the layout: the distances
  between the nodes on screen grow by the factor, while nodes and edges keep their size on screen,
  so that crowded nodes come apart (the positions and the code do not change). **Reset** puts the
  distances of the layout back (1 ×). The mouse wheel zooms the view, nodes included, and does
  not change the scale. Each tab has its own; new tabs derived from a network take its rotation
  and scale. **Copy › From tab…** lists the other tabs: choosing one gives the current tab all
  its visual elements (colors, sizes, opacities, positions and labels of nodes and edges, the
  background, the rotation and the scale); those mapped to attributes the current network does not
  have go back to their defaults (:func:`gsk_worker.prune_style`).

Each row sums up the current choice (a swatch or the colors of the scale); a click on the row
opens a dialog. Changes in the dialogs apply to the drawing as they are made; **Cancel** restores
what was there when the dialog opened. Colors and sizes do not change the network or the code:
they are part of the drawing only, computed by the worker (:func:`gsk_worker.colors`,
:func:`gsk_worker.sizes`, :func:`gsk_worker.opacities`), and kept by new layouts, by analyses
and when networks are rebuilt in another environment. They change only when the user changes
them: no function of the interface recolors or resizes anything by itself.

Colors
~~~~~~

Nodes and edges can have **one color**, or colors given by one of their attributes (any
attribute, on either scale), with a ColorBrewer palette (sequential and diverging palettes for
quantitative scales, qualitative ones for qualitative scales; the menu shows their colors):

* **Quantitative scale**: a continuous gradient through color stops, as many as the chosen
  number of stops (3 to 9, or 11 for diverging palettes), with the colors of that version of the
  palette: a value between two stops gets the color between theirs, in proportion (linear
  interpolation), so that every value has its own color, not the color of the nearest stop. The
  dialog and the panel show the gradient, each stop at the place of its value. **Placement** puts the stops at **quantiles** of the values (the default: stop ``i`` of
  ``k`` at the quantile ``i / (k - 1)``, so that about as many nodes fall between any two stops),
  **evenly spaced** from the smallest to the largest value, or at **natural breaks** (the class
  boundaries of :func:`gsk_worker.natural_breaks`). The values and colors of the stops can be
  edited (placement: custom). With **Logarithmic scale**, values are placed and interpolated on
  the logarithm of the values (see :func:`gsk_worker.transform`; values at or below 0 are
  shifted above 0 by the gap between the two smallest values). Quantiles are the same values on
  both scales (the logarithm keeps the order of the values), so turning the logarithmic scale
  on with quantiles places the stops evenly spaced on the logarithm instead; natural breaks and
  evenly spaced stops are placed again on the logarithmic scale. Values beyond the first or last
  stop get its color. Qualitative attributes count as the positions of their values in order
  (numbers numerically, then texts alphabetically; see :func:`gsk_worker.distinct`).
* **Reverse colors** (both scales) takes the colors of the palette in reverse order: the stops
  keep their values and swap their colors end to end, and later changes of palette or number of
  stops keep the reverse order.
* **Qualitative scale**: every distinct value gets the next color of the palette, in the order of
  the values; colors repeat only when there are more values than colors in the palette. Any value
  can get another color; **Automatic colors** gives back the colors of the palette, and **Clear**
  makes all values ``#999999``, to give them colors one by one. The first 1000
  values are listed; the others get the colors of the palette too.

Missing values get the **Missing values** color. Colors are picked from a color picker in three
tabs: **ColorBrewer** (the colors of all its palettes, as swatches), **Swatches** (the color space
as a grid of swatches: hues across, from light tints to dark shades down, and grays) and
**Custom** (the continuous color space). The picker opens below the color button, above the
dialog of the button.

The palettes are those of `ColorBrewer 2.0 <https://colorbrewer2.org>`_, by Cynthia A. Brewer
(Pennsylvania State University), under the Apache License 2.0.

Opacity
~~~~~~~

Nodes and edges can be made transparent, with sliders: one opacity for all (from 0,
transparent, to 1, opaque, the default), or opacities given by one of their attributes, from the **Smallest**
opacity for the smallest value to the **Largest** for the largest, on a linear or logarithmic
scale (missing values get the smallest). Lower edge opacities show the bundles of
:func:`~graphoskop.bundle_edges` where many edges overlap. Saved images keep the opacities.

Sizes
~~~~~

Node sizes (computed with :func:`gsk_worker.node_sizes`) and edge thicknesses are relative to
the default ones (1 ×): one size for all, or
sizes given by one of their attributes (any attribute; qualitative values count by their order),
from the **Smallest** size for the smallest value to the **Largest** for the largest, on a linear
or logarithmic scale. Node *areas*, not diameters, grow linearly with the values, so that a node
with twice the value of another (above the smallest) has twice its extra area; edge widths grow
linearly. Missing values get the smallest size.

Visualization panel
-------------------

Drag the background to pan, and scroll to zoom around the cursor (the sliders of the Network tab
of the `Visual elements`_ panel turn the view and scale the layout). The **Fit** button, at the
top right, zooms and centers the view so that the whole network fills it, nodes included with
their sizes, a few pixels from the borders (as when a network is first drawn or laid out again).
When nodes (of the default size) are at least 6 pixels wide on screen:

* hovering a node shows its id in a tooltip;
* clicking a node selects it: it is circled, and the data panel shows its attributes and its
  edges; Shift-click adds a node to the selection, or takes it out; clicking the background
  clears the selection;
* clicking an edge (within 4 pixels; in matrix views, its cell) selects it, and Shift-click adds
  it to the selection or takes it out; the data panel shows the selected edges;
* Shift-dragging on the background draws a rectangle, which adds the nodes in it, and the edges
  that cross it, to the selection (at any zoom, unlike the other actions of this list);
* dragging a node moves it, with all the selected nodes if it is one of them: nodes and their
  edges follow the mouse, and when the mouse is released the new positions are stored, with a
  line of code (:func:`gsk_worker.move_nodes`, a function of the interface defined in the
  namespace of every tab, as the comment says), e.g.

  .. code-block:: python

     # move_nodes: positions of nodes dragged in the graphical interface
     move_nodes(net, {"n12": (3.5, -2.0), "n40": (4.1, -2.6)})

  Nodes of matrix views, and nodes placed by attributes (see `Visual elements`_), cannot be dragged.

Where nodes overlap, the node drawn on top is the one picked. When the network is zoomed out so
far that nodes are smaller, they cannot be told apart, and tooltips, clicks, rectangles and drags
are off (the selection stays, and the Select menu still works).
