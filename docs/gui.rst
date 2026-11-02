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
to: empty for automatic); the font size can be set (14 points by default). Tick marks are inside
the axes, on all four sides. The font is the
default font of gnuplot, Helvetica, or the first font with the same metrics that is installed
(Arial, Nimbus Sans, Liberation Sans…; :data:`gsk_worker.PLOT_FONTS`). **Save PNG…** and **Save PDF…** write
the plot, with its colors, to a file (:func:`gsk_worker.degree_plot`, PNG at 200 dpi). The same
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
script to reproduce it.

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
* ``plot_window.gd`` (``plot_window.tscn``): the plots (degree distribution, epidemic models).
* ``player.gd`` (``player.tscn``): the player of the spread of an epidemic.
* ``ring.gdshader``: the rings of the selected nodes.
* ``select_dialog.gd`` (``select_dialog.tscn``): Select › Select by attribute.
* ``epidemics_dialog.gd`` (``epidemics_dialog.tscn``): Simulation › Epidemics.
* ``network_view.gd`` (``network_view.tscn``): the visualization panel of one network, with the
  shaders ``node.gdshader`` (discs of at least 3 pixels), ``edge.gdshader`` (lines of a sixth
  of the node diameter, at least 1 pixel), ``curve.gdshader`` (bundled edges: cubic Bézier
  curves computed by the GPU) and ``loop.gdshader`` (self loops: circles above their node).
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
  components, device, attributes);
* **Tables**: the tables of the runs of epidemic models of the network (see `Simulation`_), one
  at a time, chosen in the menu above the table: for each step, the mean fraction of the nodes in
  each state, with its confidence interval.

Tables show 200 rows at a time, with buttons for the previous and next pages; only the visible
page is asked to the worker. Clicking the title of a column sorts the whole table by it (not only
the rows shown), ascending, then descending at the next click (an arrow marks it), and the first
rows of the sorted table are shown: numbers numerically, texts alphabetically, missing values last
(:func:`gsk_worker.sort_rows`). Numbers are aligned to the right, texts to the left, with their
column titles. Hypergraphs and simplicial complexes show their incidence networks:
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

Editing
-------

The **Edit** menu starts with the operations that make a new network from the network of the
current tab, each opening in a new tab: **Bipartite projection**, **Backbone** and **Transform**
(see below). Then it changes the network of the current tab using the selected nodes and edges
(see `Visualization panel`_ and `Selecting`_):

* **Undo** (Ctrl+Z, ⌘Z on macOS) and **Redo** (Ctrl+Shift+Z): back and forth over the moves of
  nodes (dragged in the visualization panel) and the removals of nodes and edges, with their
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

* **Clique expansion…** (:func:`~graphoskop.clique_expansion`), for hypergraphs and simplicial
  complexes, and **Clique promotion…** (:func:`~graphoskop.clique_promotion`), for networks (its
  hypergraph or simplicial complex is laid out as its incidence network).

Items with parameters (how parallel edges combine, the attribute of the quotient, …) open their
parameter dialog first (:data:`gsk_worker.TRANSFORMATIONS`); the others run at once. Errors of the
library (e.g. a transformation of directed networks on an undirected one) are shown in the dialog,
or in a message. New networks keep the positions of their nodes when they have them (the quotient
places its nodes at the mean position of theirs), or get the quick grid layout.

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
  of the backbone without these options. **Extract backbone** opens the backbone at the chosen
  threshold in a new tab:

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
  kept in the order they are checked;
* a slider with a number for values between two bounds (probabilities, mixing parameters,
  angles);
* a number with arrows for integers and other numbers (e.g. ``n``);
* a field for names (e.g. the ``attribute`` written) and node ids, typed without quotes;
* a field for numbers that can be left empty (e.g. ``seed``);
* a field for lists: numbers separated by commas (``250, 250, 250, 250``), rows of a matrix
  separated by semicolons (``0.05, 0.002; 0.002, 0.05``), pairs ``key: value`` for dictionaries
  (``1: 75, 2: 50``); Python expressions such as ``[3] * 1000`` work too.

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

The last item, **Bundle edges…**, bundles the edges of the current layout with
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
* the columns of the source and target nodes; in multilayer networks, the layer column; in
  signed networks, the column of the signs; in directed hypergraphs, the column with ``"tail"``
  or ``"head"``;
* the format of the node table, wide or long (guessed: long if it has three or more columns and
  a node id repeats in the first one), and its key columns: node id (entity and layer in
  multilayer networks), then attribute name and value in long format;
* whether the network is directed, bipartite, signed, multilayer, a hypergraph or a simplicial
  complex. In hypergraphs and simplicial complexes each row of the edge table is a node in a
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
anyway**, and the visual elements panel, the search box, the Select menu and Save image are off,
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


* The menu bar: File (**Load network…**, **Save image…**: the visualization panel of the current
  network as a PNG image, twice as large as on screen, without the selection marks), Edit (see
  `Editing`_), Select (see `Selecting`_), Layout (see `Layouts`_), Analysis (see `Analysis`_),
  Simulation (see `Simulation`_), Generators (see `Generators`_), Help. On macOS it is the global menu bar of
  the system.
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

Python worker
-------------

.. automodule:: gsk_worker

.. autodata:: gsk_worker.BACKBONE_COLORS
   :no-value:

.. autodata:: gsk_worker.BACKBONE_HIDDEN
   :no-value:

.. autodata:: gsk_worker.BACKBONE_OPTIONS
   :no-value:

.. autodata:: gsk_worker.BACKBONE_STEPS

.. autodata:: gsk_worker.BACKBONE_TAIL

.. autodata:: gsk_worker.BACKBONES
   :no-value:

.. autodata:: gsk_worker.BENDS

.. autodata:: gsk_worker.CENTRALITIES
   :no-value:

.. autodata:: gsk_worker.CENTRALITY_HIDDEN
   :no-value:

.. autodata:: gsk_worker.CHOICES
   :no-value:

.. autodata:: gsk_worker.COMMANDS
   :no-value:

.. autodata:: gsk_worker.COMMUNITIES
   :no-value:

.. autodata:: gsk_worker.CURVE_STRIDE

.. autodata:: gsk_worker.DISTRIBUTION_BINS

.. autodata:: gsk_worker.DISTRIBUTION_CATEGORIES

.. autodata:: gsk_worker.DISTRIBUTION_DISCRETE

.. autodata:: gsk_worker.DISTRIBUTIONS
   :no-value:

.. autodata:: gsk_worker.DOCS
   :no-value:

.. autodata:: gsk_worker.EDGE_COLOR

.. autodata:: gsk_worker.EDGE_STRIDE

.. autodata:: gsk_worker.EPIDEMIC_HIDDEN
   :no-value:

.. autodata:: gsk_worker.EPIDEMICS
   :no-value:

.. autodata:: gsk_worker.EXTRA_CHOICES
   :no-value:

.. autodata:: gsk_worker.FILTERS
   :no-value:

.. autodata:: gsk_worker.FIT_REQUIRED
   :no-value:

.. autodata:: gsk_worker.GENERATORS
   :no-value:

.. autodata:: gsk_worker.LAYOUTS
   :no-value:

.. autodata:: gsk_worker.LISTS
   :no-value:

.. autodata:: gsk_worker.GROUPS_NOTE

.. autodata:: gsk_worker.HIDDEN
   :no-value:

.. autodata:: gsk_worker.HINTS
   :no-value:

.. autodata:: gsk_worker.INTEGERS
   :no-value:

.. autodata:: gsk_worker.LOOP_SIZE

.. autodata:: gsk_worker.MAX_BREAK_VALUES

.. autodata:: gsk_worker.MAX_CELLS

.. autodata:: gsk_worker.MAX_ELEMENTS

.. autodata:: gsk_worker.MAX_FOUND

.. autodata:: gsk_worker.MEASURES
   :no-value:

.. autodata:: gsk_worker.MAX_SEGMENTS

.. autodata:: gsk_worker.MAX_UNDO

.. autodata:: gsk_worker.MAX_VALUES

.. autodata:: gsk_worker.MISSING_COLOR

.. autodata:: gsk_worker.MOVE_NOTE

.. autodata:: gsk_worker.NESTEDNESS_CELLS

.. autodata:: gsk_worker.NODE_COLOR

.. autodata:: gsk_worker.NODE_SIZES_NOTE

.. autodata:: gsk_worker.NODE_STRIDE

.. autodata:: gsk_worker.NODES
   :no-value:

.. autodata:: gsk_worker.NUMBERS
   :no-value:

.. autodata:: gsk_worker.OPTIONS
   :no-value:

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

.. autodata:: gsk_worker.SHOWN_DEFAULTS
   :no-value:

.. autodata:: gsk_worker.STATE_COLORS

.. autodata:: gsk_worker.STRINGS
   :no-value:

.. autodata:: gsk_worker.STYLE_LABELS
   :no-value:

.. autodata:: gsk_worker.TAILS
   :no-value:

.. autodata:: gsk_worker.TRANSFORMATION_HIDDEN
   :no-value:

.. autodata:: gsk_worker.TRANSFORMATIONS
   :no-value:

.. autodata:: gsk_worker.TRANSFORMS
   :no-value:

.. autodata:: gsk_worker.TYPE1_COLOR

.. autofunction:: gsk_worker.animation

.. autofunction:: gsk_worker.attributes

.. autofunction:: gsk_worker.backbone_code

.. autofunction:: gsk_worker.backbone_plot

.. autofunction:: gsk_worker.backbone_range

.. autofunction:: gsk_worker.backbone_scale


.. autofunction:: gsk_worker.centrality_code

.. autofunction:: gsk_worker.breaks

.. autofunction:: gsk_worker.colors

.. autofunction:: gsk_worker.default_style

.. autofunction:: gsk_worker.degree_plot

.. autofunction:: gsk_worker.distinct

.. autofunction:: gsk_worker.distribution

.. autofunction:: gsk_worker.distribution_plot

.. autofunction:: gsk_worker.drawing

.. autofunction:: gsk_worker.edge_values

.. autofunction:: gsk_worker.edit_code

.. autofunction:: gsk_worker.environment

.. autofunction:: gsk_worker.epidemic_plot

.. autofunction:: gsk_worker.find_nodes

.. autofunction:: gsk_worker.fit_code

.. autofunction:: gsk_worker.generator_code

.. autofunction:: gsk_worker.last_layout

.. autofunction:: gsk_worker.groups

.. autofunction:: gsk_worker.histories

.. autofunction:: gsk_worker.layout_code

.. autofunction:: gsk_worker.literal

.. autofunction:: gsk_worker.load_code

.. autofunction:: gsk_worker.main

.. autofunction:: gsk_worker.measures

.. autofunction:: gsk_worker.move_nodes

.. autofunction:: gsk_worker.natural_breaks

.. autofunction:: gsk_worker.nestedness_code

.. autofunction:: gsk_worker.nestedness_matrix

.. autofunction:: gsk_worker.nestedness_plot

.. autofunction:: gsk_worker.nestedness_text

.. autofunction:: gsk_worker.node_size

.. autofunction:: gsk_worker.node_sizes

.. autofunction:: gsk_worker.node_values

.. autofunction:: gsk_worker.numbers

.. autofunction:: gsk_worker.opacities

.. autofunction:: gsk_worker.pick_grid

.. autofunction:: gsk_worker.positions

.. autofunction:: gsk_worker.preview

.. autofunction:: gsk_worker.prune_style

.. autofunction:: gsk_worker.randomization_plot

.. autofunction:: gsk_worker.randomization_text

.. autoclass:: gsk_worker.ProgressBar

.. autofunction:: gsk_worker.signature

.. autofunction:: gsk_worker.simulation_code

.. autofunction:: gsk_worker.size_code

.. autofunction:: gsk_worker.sort_rows

.. autofunction:: gsk_worker.spread

.. autofunction:: gsk_worker.sizes

.. autofunction:: gsk_worker.subdivisions

.. autofunction:: gsk_worker.summary

.. autofunction:: gsk_worker.text

.. autofunction:: gsk_worker.too_large

.. autofunction:: gsk_worker.transform

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

The interface uses Godot's Compatibility renderer (OpenGL 3.3, OpenGL ES 3.0 or WebGL 2), which
runs on any GPU of the last fifteen years; its file dialogs are those of the operating system.

Selecting
---------

Nodes and edges are selected in the visualization panel (click, Shift-click, Shift-drag a
rectangle; see `Visualization panel`_), and with the **Select** menu:

* **Select all nodes**, **Clear selection**;
* **Select by attribute…**: a dialog selecting the nodes, or the edges, whose attribute is equal
  to (=), different from (≠), smaller than (<, ≤) or larger than (>, ≥) a value, or contains a
  text (a part of the text, or one of the values of a node with several). Numeric attributes take
  a number; qualitative ones a value from the menu of their values (or a text, when they have
  too many). The selection replaces the current one, or, with **Add to the current selection**,
  is added to it. If nothing matches, the dialog says so and stays open.

Self loops are drawn as circles above their node, a little more than half as wide as the node, as
thick as the edges and in their colors, which follow the node when it is dragged (in matrix views
they are the cells of the diagonal). Selected nodes are circled (all of them, by the GPU: ``ring.gdshader``) and selected edges drawn
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
* the size of the window, and whether it was maximized.

Deleting the file resets them.

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
  ``imageio-ffmpeg`` package). Every step is drawn as on screen, with its number in the top left
  corner (:func:`gsk_worker.animation`).

Using any button or the timeline turns States on; the colors of the style come back when it is
turned off. The player only changes the colors on screen: the style, the network and the code do
not change. The worker sends the states at step 0 and the changes of every step
(:func:`gsk_worker.spread`), and the interface colors only the nodes that change, so playing is
fast on large networks. Nodes without a state (other components, hyperedges and facets of
incidence networks) keep the colors of the style.

Visual elements
---------------

The visual elements panel, left of the visualization panel, sets the look of the drawing of the
current network, in three tabs:

* **Nodes**: their color, size, opacity and position. By default all nodes are ``#e41a1c``; in bipartite networks
  (and incidence networks of hypergraphs and complexes) the second type of nodes is ``#377eb8``,
  the next color of ColorBrewer's Set1 palette. **Position** places the nodes by the layout (the
  default), or takes their x and y coordinates from numeric node attributes (e.g. longitude and
  latitude), each axis on its own (:func:`gsk_worker.positions`). The attributes override the
  layout, the one computed and any later one, until the layout is chosen again; the view is fitted
  to the new positions. Nodes without a value of the attribute are put at its smallest value.
  Like the other visual elements, this changes the drawing only, not the network or the code; while
  positions come from attributes, nodes cannot be dragged, and bundled edges are drawn straight
  (their curves follow the layout).
* **Edges**: their color (by default ``#999999``), thickness and opacity.
* **Network**: the background color; **Rotation**, a slider turning the view (-180° to 180°,
  counterclockwise) without changing the positions or the code; **Scale**, a logarithmic slider
  (and a number) from a tenth to a hundred times, which blows up the layout: the distances
  between the nodes on screen grow by the factor, while nodes and edges keep their size on screen,
  so that crowded nodes come apart (the positions and the code do not change). **Reset** puts the
  distances of the layout back (1 ×). The mouse wheel zooms the view, nodes included, and does
  not change the scale. Each tab has its own; new tabs derived from a network take its rotation
  and scale.

Each row sums up the current choice (a swatch or the colors of the scale); its **Edit…** button
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
