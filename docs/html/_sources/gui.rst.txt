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

**Analysis › Community discovery** lists every community discovery algorithm of the library (see
:doc:`communities`): BigCLAM, DEMON, hierarchical random graphs, Infomap, label propagation,
Leiden, link communities (hierarchical, and from the line graph), OSLOM and stochastic
blockmodels. Each opens a dialog with its parameters, with the same controls as for the
generators (see `Generators`_), then runs on the network of the current tab and adds its line to
the code panel, e.g.

.. code-block:: python

   gsk.leiden_communities(net, resolution = 2)

The communities are written as an attribute (``community`` by default, ``link_community`` for
link communities; the dialog can change the name), shown in the data panel, and the nodes are
colored by it on a qualitative scale (the edges, for link communities), which can then be
changed in the `Visual elements`_ panel. The view keeps its zoom and center. Hypergraphs and
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
* ``params_dialog.gd`` (``params_dialog.tscn``): the parameters of a generator, a layout or a
  community discovery algorithm.
* ``network_view.gd`` (``network_view.tscn``): the visualization panel of one network, with the
  shaders ``node.gdshader`` (discs of at least 3 pixels), ``edge.gdshader`` (lines of a sixth
  of the node diameter, at least 1 pixel) and ``curve.gdshader`` (bundled edges: cubic Bézier
  curves computed by the GPU).
* ``visual_panel.gd`` (``visual_panel.tscn``): the visual elements panel, with its dialogs
  ``color_dialog.gd`` (``color_dialog.tscn``) and ``size_dialog.gd`` (``size_dialog.tscn``).
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

The data panel, at the bottom left, has three tabs:

* **Nodes**: the attributes of all nodes, one row per node (positions included, as ``pos_x`` and
  ``pos_y``), or, when a node is selected, its attributes and their values;
* **Edges**: all edges (source, target and attributes), or the edges of the selected node;
* **Network**: the properties of the network (kind, direction, numbers of nodes, edges and
  components, device, attributes).

Tables show 200 rows at a time, with buttons for the previous and next pages; only the visible
page is asked to the worker. Numbers are aligned to the right, texts to the left, with their
column titles. Hypergraphs and simplicial complexes show their incidence networks:
hyperedges and facets are rows of the node table, memberships of the edge table.

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
drawn as a loaded one. Fitting a generator to an open network (the ``net`` parameter) is not offered yet.

Layouts
-------

The **Layout** menu lays out the network of the current tab with a layout of node-link diagrams
of the library (see :doc:`layouts`):

* **Circular**: :func:`~graphoskop.circular_layout`;
* **Force-directed**: :func:`~graphoskop.force_directed_layout`;
* **Grid**: :func:`~graphoskop.grid_layout`;
* **Hierarchical**: :func:`~graphoskop.hierarchical_layout`;
* **Hierarchical, radial**: :func:`~graphoskop.hierarchical_layout` with ``direction = "radial"``;
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

Further items transform the current layout: **Mirror left to right** and **Mirror top to
bottom** (:func:`~graphoskop.mirror_layout`), **Rotate 90° counterclockwise** and **Rotate 90°
clockwise** (:func:`~graphoskop.rotate_layout`), which run at once, **Rotate…**, which opens a
dialog with a slider for the angle, and **Remove overlaps** (:func:`~graphoskop.remove_overlaps`,
at once), with ``size`` the diameters of the drawn nodes: a number, or, when node sizes are given
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
:func:`~graphoskop.load_simplicial_complex`) is called:

* the edge table (required) and the node table (optional), with the separator of both files and
  whether they have a header row. The first ten rows of each are shown, as the library reads
  them;
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
* the device, CPU or GPU (offered when PyTorch finds CUDA).

On **Import**, the network opens in a new tab: the code panel shows the code that loaded it, for
example

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("/data/edges.tsv", nodes = "/data/nodes.tsv", directed = True)
   gsk.grid_layout(net)

with only the arguments that differ from their defaults. The network is then laid out with
:func:`~graphoskop.grid_layout` (hypergraphs and simplicial complexes as their
:func:`~graphoskop.incidence_network`, with hyperedges or facets drawn as nodes of another color),
and drawn. The layout line is part of the code, since it stores the positions as the node
attributes ``pos_x`` and ``pos_y``. Errors of the library (e.g. a bipartite network without node
types) are shown in the load dialog, which stays open with its choices until the network is
loaded.

Main window
-----------

* The menu bar: File (**Load network…**, **Save image…**: the visualization panel of the current
  network as a PNG image, twice as large as on screen, without the selection ring), Edit,
  Select, Layout (see `Layouts`_), Analysis (see `Analysis`_), Simulation, Generators (see
  `Generators`_), Help. On macOS it is the global menu bar of
  the system.
* The tab bar: one tab per open network; the close button of a tab forgets its network.
* Above, the `Visual elements`_ panel on the left and the `Visualization panel`_ on the right.
* Below, over the whole width, side by side, the `Data panel`_ and the code panel (the code of
  the current tab).
* At the bottom, the status bar: on the left, the size of the current network or what is
  running; then, while a function of the library with a progress bar runs (e.g. a
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

.. autodata:: gsk_worker.BENDS

.. autodata:: gsk_worker.CHOICES
   :no-value:

.. autodata:: gsk_worker.COMMANDS
   :no-value:

.. autodata:: gsk_worker.COMMUNITIES
   :no-value:

.. autodata:: gsk_worker.CURVE_STRIDE

.. autodata:: gsk_worker.EDGE_COLOR

.. autodata:: gsk_worker.EDGE_STRIDE

.. autodata:: gsk_worker.GENERATORS
   :no-value:

.. autodata:: gsk_worker.LAYOUTS
   :no-value:

.. autodata:: gsk_worker.LISTS
   :no-value:

.. autodata:: gsk_worker.HIDDEN
   :no-value:

.. autodata:: gsk_worker.HINTS
   :no-value:

.. autodata:: gsk_worker.INTEGERS
   :no-value:

.. autodata:: gsk_worker.MAX_BREAK_VALUES

.. autodata:: gsk_worker.MAX_CELLS

.. autodata:: gsk_worker.MAX_SEGMENTS

.. autodata:: gsk_worker.MAX_VALUES

.. autodata:: gsk_worker.MISSING_COLOR

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

.. autodata:: gsk_worker.QUALITATIVE_PALETTE
   :no-value:

.. autodata:: gsk_worker.RANGES
   :no-value:

.. autodata:: gsk_worker.SHOWN_DEFAULTS
   :no-value:

.. autodata:: gsk_worker.STRINGS
   :no-value:

.. autodata:: gsk_worker.TRANSFORMS
   :no-value:

.. autodata:: gsk_worker.TYPE1_COLOR

.. autofunction:: gsk_worker.attributes

.. autofunction:: gsk_worker.bent

.. autofunction:: gsk_worker.breaks

.. autofunction:: gsk_worker.colors

.. autofunction:: gsk_worker.default_style

.. autofunction:: gsk_worker.distinct

.. autofunction:: gsk_worker.drawing

.. autofunction:: gsk_worker.edge_values

.. autofunction:: gsk_worker.environment

.. autofunction:: gsk_worker.generator_code

.. autofunction:: gsk_worker.layout_code

.. autofunction:: gsk_worker.literal

.. autofunction:: gsk_worker.load_code

.. autofunction:: gsk_worker.main

.. autofunction:: gsk_worker.natural_breaks

.. autofunction:: gsk_worker.node_size

.. autofunction:: gsk_worker.node_sizes

.. autofunction:: gsk_worker.node_values

.. autofunction:: gsk_worker.numbers

.. autofunction:: gsk_worker.opacities

.. autofunction:: gsk_worker.pick_grid

.. autofunction:: gsk_worker.preview

.. autoclass:: gsk_worker.ProgressBar

.. autofunction:: gsk_worker.signature

.. autofunction:: gsk_worker.size_code

.. autofunction:: gsk_worker.sizes

.. autofunction:: gsk_worker.subdivisions

.. autofunction:: gsk_worker.summary

.. autofunction:: gsk_worker.text

.. autofunction:: gsk_worker.transform

.. autofunction:: gsk_worker.widget

Running the interface
---------------------

The interface needs Godot 4.7 and a Python environment with the dependencies of graphoskop
(PyTorch, PyTorch Geometric, NumPy, SciPy, pandas, Numba, tqdm). graphoskop itself does not
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

Visual elements
---------------

The visual elements panel, left of the visualization panel, sets the look of the drawing of the
current network, in three tabs:

* **Nodes**: their color, size and opacity. By default all nodes are ``#e41a1c``; in bipartite networks
  (and incidence networks of hypergraphs and complexes) the second type of nodes is ``#377eb8``,
  the next color of ColorBrewer's Set1 palette.
* **Edges**: their color (by default ``#999999``), thickness and opacity.
* **Network**: the background color.

Each row sums up the current choice (a swatch or the colors of the scale); its **Edit…** button
opens a dialog. Changes in the dialogs apply to the drawing as they are made; **Cancel** restores
what was there when the dialog opened. Colors and sizes do not change the network or the code:
they are part of the drawing only, computed by the worker (:func:`gsk_worker.colors`,
:func:`gsk_worker.sizes`, :func:`gsk_worker.opacities`), and kept by new layouts and when networks are rebuilt in another
environment.

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
  the logarithm of the values (see :func:`gsk_worker.transform`). Values beyond the first or last
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

Nodes and edges can be made transparent: one opacity for all (from 0, transparent, to 1,
opaque, the default), or opacities given by one of their attributes, from the **Smallest**
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

Drag to pan, and scroll to zoom around the cursor. The **Fit** button, at the top right, zooms
and centers the view so that the whole network fills it, nodes included with their sizes, a few
pixels from the borders (as when a network is first drawn or laid out again). When nodes (of the
default size) are at least 6 pixels wide on screen, hovering a node shows its id in a tooltip, and clicking it selects it:
it is circled, and the data panel shows its attributes and its edges. Where nodes overlap, the
node drawn on top is the one picked. Clicking elsewhere clears the selection. When
the network is zoomed out so far that nodes are smaller, they cannot be told apart, and tooltips
and selection are off.
