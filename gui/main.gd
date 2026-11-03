extends Control
## Main window: menu bar, one tab per open network, the visual elements panel on the left, the
## visualization panel on the right with the data and code panels below it, and a status bar
## (progress of long computations, and the Python environment of the worker: click it to change
## environment).

const NetworkView := preload("res://network_view.tscn")
## Ids of the items of the File menu.
const FILE_LOAD := 0
const FILE_IMAGE := 1
## Scale of saved images, relative to the size of the visualization panel on screen.
const IMAGE_SCALE := 2
## Items of the Layout menu: label, function of the library, preset values of the parameters
## (Python source code), and whether a parameter dialog opens (layouts, edge bundling) or the
## function runs at once (other transformations); an empty label is a separator. The view is
## rotated with the slider of the visual elements panel (Network tab), which does not change the
## positions.
const LAYOUTS := [
   ["Circular…", "circular_layout", {}, true],
   ["Force-directed…", "force_directed_layout", {}, true],
   ["Grid…", "grid_layout", {}, true],
   ["Grid, quick", "grid_layout", {"fast": "True"}, false],
   ["Hierarchical…", "hierarchical_layout", {}, true],
   ["Hierarchical, radial…", "hierarchical_layout", {"direction": "\"radial\""}, true],
   ["Matrix…", "matrix_layout", {}, true],
   ["UMAP…", "umap_layout", {}, true],
   ["", "", {}, false],
   ["Mirror left to right", "mirror_layout", {}, false],
   ["Mirror top to bottom", "mirror_layout", {"axis": "\"y\""}, false],
   ["Remove overlaps", "remove_overlaps", {}, false],
   ["", "", {}, false],
   ["Bundle edges…", "bundle_edges", {}, true],
]
## Items of the Edit menu: label, operation of the worker command [code]edit[/code] (see
## [code]edit_code[/code] in the worker), and whether ego networks keep the egos. Removals change
## the network of the tab; the induced subgraph and ego networks open in a new tab. Undo and Redo
## go back and forth over the moves, removals and additions (worker command [code]undo[/code]).
const EDITS := [
   ["Undo", "undo", true],
   ["Redo", "redo", true],
   ["", "", true],
   ["Remove selected nodes", "remove_nodes", true],
   ["Remove selected edges", "remove_edges", true],
   ["Remove selected nodes and edges", "remove", true],
   ["", "", true],
   ["Induced subgraph of the selected nodes", "subgraph", true],
   ["Ego network of the selected nodes", "ego", true],
   ["Ego network of the selected nodes, without them", "ego", false],
]
## Shortcuts of the items of the Edit menu, by operation (Ctrl is Command on macOS): Ctrl+Z,
## Ctrl+Shift+Z, Delete.
const EDIT_KEYS := {"undo": KEY_Z, "redo": KEY_MASK_SHIFT | KEY_Z, "remove": KEY_DELETE}
## Items of Edit › Backbone: label, function, and whether it filters the edges directly (its
## backbone opens in a new tab) rather than scoring them (the backbone dialog opens); an empty
## label is a separator.
const BACKBONES := [
   ["Noise corrected…", "noise_corrected", false],
   ["Disparity filter…", "disparity_filter", false],
   ["Pólya filter…", "polya_filter", false],
   ["Doubly stochastic…", "doubly_stochastic", false],
   ["High salience skeleton…", "high_salience_skeleton", false],
   ["Convex skeleton…", "convex_skeleton", false],
   ["", "", false],
   ["Maximum spanning tree…", "spanning_tree", true],
   ["Maximally filtered graph…", "maximally_filtered_graph", true],
   ["Triangulated filtered graph…", "triangulated_filtered_graph", true],
]
## Items of Edit › Bipartite projection: label, projection function of the library, and
## [code]true[/code] (the projection opens in a new tab, see [method _run_backbone]).
const PROJECTIONS := [
   ["Simple count…", "simple_count_projection", true],
   ["Histogram intersection…", "hist_intersection_projection", true],
   ["Hyperbolic…", "hyperbolic_projection", true],
   ["Resource allocation…", "resource_allocation_projection", true],
   ["Random walk…", "random_walk_projection", true],
   ["Cosine similarity…", "cosine_similarity_projection", true],
   ["Pearson correlation…", "pearson_correlation_projection", true],
   ["Inverse Euclidean…", "inverse_euclidean_projection", true],
]
## Items of Edit › Transform: label, transformation of the library, and whether it has parameters
## (a parameter dialog opens first; the others run at once); the result opens in a new tab (see
## [method _transform]).
const TRANSFORMS := [
   ["Flip edge direction", "flip_edge_direction", false],
   ["Undirected…", "to_undirected", true],
   ["Strongly connected components", "to_strongly_connected", false],
   ["Largest connected component", "largest_component", false],
   ["Condensation…", "network_condensation", true],
   ["", "", false],
   ["Remove isolated nodes", "remove_isolates", false],
   ["Remove parallel edges…", "remove_parallel_edges", true],
   ["Remove self loops", "remove_self_loops", false],
   ["", "", false],
   ["Quotient by a node attribute…", "quotient_network", true],
   ["", "", false],
   ["Clique expansion…", "clique_expansion", true],
   ["Clique promotion…", "clique_promotion", true],
]
## Items of the Help menu: label and web page (opened in the browser).
const HELP := [
   ["Online documentation", "https://graphoskop.readthedocs.io/en/latest/"],
   ["Graphical interface (online documentation)", "https://graphoskop.readthedocs.io/en/latest/gui.html"],
]
## Ids of the items of the menu of the search box that are not nodes.
const FOUND_ALL := -2
const FOUND_MORE := -3
## Items of the Select menu.
const SELECTS := [["Select all nodes"], ["Clear selection"], [""], ["Select by attribute…"]]
## Items of Analysis › Centrality: label and function (degree also plots the degree distribution).
const CENTRALITIES := [
   ["Alpha…", "alpha_centrality"],
   ["Betweenness…", "betweenness_centrality"],
   ["Closeness…", "closeness_centrality"],
   ["Degree…", "degree"],
   ["Edge betweenness…", "edge_betweenness_centrality"],
   ["Eigenvector…", "eigenvector_centrality"],
   ["Flow…", "flow_centrality"],
   ["Harmonic…", "harmonic_centrality"],
   ["HITS (hubs and authorities)…", "hits"],
   ["k-core (coreness)…", "k_core_centrality"],
   ["Katz…", "katz_centrality"],
   ["PageRank…", "pagerank"],
   ["Reach…", "reach_centrality"],
   ["SpringRank…", "springrank"],
]
## Items of Analysis › Core-periphery: label and function. The three models write the core value
## of every node as a node attribute (worker command [code]centrality[/code]); nestedness, for
## bipartite networks, opens the plot of the biadjacency matrix with its isocline (worker command
## [code]nestedness[/code]).
const CORE_PERIPHERY := [
   ["Discrete (Borgatti-Everett)…", "discrete_core_periphery"],
   ["Continuous (Borgatti-Everett)…", "continuous_core_periphery"],
   ["Rombach…", "rombach_core_periphery"],
   ["", ""],
   ["Nestedness…", "nestedness"],
]
## Items of Analysis › Node vectors: label, function and preset values of the parameters (worker
## command [code]vectors[/code]). The generalized Euclidean distance uses the Laplacian solver
## ([code]sparse = True[/code]): a dense pseudoinverse costs minutes on directed networks of
## 10,000 nodes (a singular value decomposition), the solver a fraction of a second.
## One or two node attributes give a value, shown in a message; the pairwise analyses compare a set
## of attributes (check boxes; none for all) and open the heatmap of their matrix in the plot
## window; the graph Fourier transform opens the plot of its spectrum.
const VECTORS := [
   ["Annihilation distance…", "annihilation_distance"],
   ["Earth mover's distance…", "earth_movers_distance"],
   ["Generalized Euclidean distance…", "generalized_euclidean", {"sparse": "True"}],
   ["Generalized Euclidean distances, pairwise…", "pairwise_generalized_euclidean"],
   ["Graph Fourier transform…", "graph_fourier_transform"],
   ["Network correlation…", "network_correlation"],
   ["Network correlations, pairwise…", "pairwise_network_correlation"],
   ["Network entropy…", "network_entropy"],
   ["Network variance…", "network_variance"],
]
## Items of Analysis › Community discovery, as in the documentation of the library: label and
## function.
const COMMUNITIES := [
   ["BigCLAM…", "bigclam_communities"],
   ["DEMON…", "demon_communities"],
   ["Hierarchical random graphs…", "hierarchical_random_graph_communities"],
   ["Infomap…", "infomap_communities"],
   ["Label propagation…", "label_propagation_communities"],
   ["Leiden…", "leiden_communities"],
   ["Link communities, hierarchical…", "hierarchical_link_communities"],
   ["Link communities, line graph…", "line_graph_communities"],
   ["OSLOM…", "oslom_communities"],
   ["Stochastic blockmodels…", "block_model_communities"],
]
## Items of the Generators menu, by family as in the documentation of the library: label, function,
## preset values of the parameters (Python source code) that make a network of a useful size; an
## empty label is a separator.
const GENERATORS := [
   ["Chain", "chain_graph", {"n": "100"}],
   ["", "", {}],
   ["Random graph G(n, m)", "gnm_random_graph", {"n": "1000", "m": "3000"}],
   ["Random graph G(n, p)", "gnp_random_graph", {"n": "1000", "p": "0.006"}],
   ["Configuration model", "configuration_model", {"degrees": "[3] * 1000"}],
   ["Kronecker graph", "kronecker_graph", {"initiator": "0.9, 0.6; 0.6, 0.1", "k": "10"}],
   ["Random geometric graph", "geometric_graph", {"n": "1000", "r": "0.05"}],
   ["", "", {}],
   ["Caveman graph", "caveman_graph", {"l": "20", "k": "10", "connected": "True"}],
   ["Watts-Strogatz", "watts_strogatz_graph", {"n": "1000", "k": "4", "p": "0.1"}],
   ["", "", {}],
   ["Barabási-Albert", "barabasi_albert_graph", {"n": "1000", "k": "3"}],
   ["Link selection", "link_selection_graph", {"n": "1000", "k": "3"}],
   ["Copying model", "copying_graph", {"n": "1000", "k": "3"}],
   ["Holme-Kim", "holme_kim_graph", {"n": "1000", "k": "3", "p": "0.5"}],
   ["", "", {}],
   ["Planted partition", "planted_partition_graph", {"sizes": "250, 250, 250, 250", "p_in": "0.05", "p_out": "0.002"}],
   ["Stochastic block model", "stochastic_block_model", {"sizes": "300, 200, 100", "probs": "0.05, 0.002, 0.002; 0.002, 0.05, 0.002; 0.002, 0.002, 0.05"}],
   ["Degree-corrected block model", "degree_corrected_block_model", {"sizes": "500, 500", "degrees": "[10] * 1000", "edges": "4500, 500; 500, 4500"}],
   ["Mixed membership block model", "mixed_membership_block_model", {"probs": "0.1, 0.005; 0.005, 0.1", "n": "500", "alpha": "0.1"}],
   ["Nested block model", "nested_block_model", {"n": "1000", "m": "5000", "levels": "16, 4"}],
   ["", "", {}],
   ["Girvan-Newman benchmark", "girvan_newman_graph", {"mu": "0.2"}],
   ["LFR benchmark", "lfr_benchmark_graph", {"n": "1000", "k": "20", "maxk": "50", "mu": "0.3", "minc": "20", "maxc": "100"}],
   ["", "", {}],
   ["Growing manifold", "growing_manifold", {"n": "200", "d": "2"}],
   ["Simplicial configuration model", "simplicial_configuration_model", {"degrees": "[3] * 100", "dimensions": "1: 75, 2: 50"}],
   ["Hypergraph configuration model", "hypergraph_configuration_model", {"degrees": "[2] * 300", "sizes": "[3] * 200"}],
]

## One entry per tab: {"id": tab id in the worker, "name", "view": its NetworkView, "code",
## "summary", "style": style of the drawing, "attributes": node and edge attributes for the
## visual elements panel}.
var _tabs := []
## The tab of the epidemics dialog.
var _epidemics_tab := {}
## The tab of the link prediction dialog, and its runs (from the worker).
var _links_tab := {}
var _link_info := {}
## The tab of the graph neural network dialog.
var _gnns_tab := {}
## Ids of Analysis › Link prediction… and Graph neural networks… (apart from the submenus).
const ANALYSIS_LINKS := 2000
const ANALYSIS_GNNS := 2001
## Text of the visualization panel without networks.
const HINT := "File › Load network… or Generators to open a network"
## The tab of the backbone dialog.
var _backbone_tab := {}
## Menu of a column of the Nodes and Edges tables (right click on its title), the column it is for
## ([kind, attribute]), and the confirmation of the deletion of an attribute.
var _column_menu := PopupMenu.new()
var _column := ["", ""]
var _confirm_delete := ConfirmationDialog.new()
## Question asked when a new network is too large to draw (worker [code]too_large[/code]), and the
## tab it is about.
var _large := AcceptDialog.new()
var _large_tab := {}
## Menu of a right click in the visualization panel (add an edge from the node, or a node there),
## and what it is about: the tab, the node (or -1) and the point (drawing coordinates).
var _context_menu := PopupMenu.new()
var _context := {}
## Nodes listed by the menu of the search box, and the exact matches among them.
var _found_nodes := PackedInt32Array()
var _found_exact := PackedInt32Array()
var _next_tab := 0
## Callbacks of the tasks running in the worker (see [method _task]), by task id.
var _pending := {}
var _next_task := 0
## The task dialog (kept here: it moves into the dialog that started a task, outside the unique
## names of this scene).
@onready var _task_dialog: AcceptDialog = $TaskDialog
## The parameter dialog and the plot window (kept here: they move into the epidemics dialog while
## they are opened from it, see [method _above]).
@onready var _params: ConfirmationDialog = $ParamsDialog
@onready var _plot: AcceptDialog = $PlotWindow
var _choose_python: Button
var _found := PackedStringArray()


func _ready() -> void:
   var size: Vector2i = AppSettings.value("window", "size", Vector2i.ZERO)
   if size != Vector2i.ZERO:
      get_window().size = size
   if AppSettings.value("window", "maximized", false):
      get_window().mode = Window.MODE_MAXIMIZED
   %File.id_pressed.connect(_on_file)
   %ImageDialog.file_selected.connect(_save_image)
   _menu(%Layout, LAYOUTS)
   %Layout.id_pressed.connect(_run_layout)
   var centralities := PopupMenu.new()
   _menu(centralities, CENTRALITIES)
   centralities.id_pressed.connect(func(id): _run_analysis("centrality", "Centrality", CENTRALITIES[id]))
   %Analysis.add_submenu_node_item("Centrality", centralities)
   var cores := PopupMenu.new()
   _menu(cores, CORE_PERIPHERY)
   cores.id_pressed.connect(func(id): _run_analysis("nestedness" if CORE_PERIPHERY[id][1] == "nestedness" else "centrality", "Core-periphery", CORE_PERIPHERY[id]))
   %Analysis.add_submenu_node_item("Core-periphery", cores)
   var communities := PopupMenu.new()
   _menu(communities, COMMUNITIES)
   communities.id_pressed.connect(func(id): _run_analysis("communities", "Community discovery", COMMUNITIES[id]))
   %Analysis.add_submenu_node_item("Community discovery", communities)
   var vectors := PopupMenu.new()
   _menu(vectors, VECTORS)
   vectors.id_pressed.connect(func(id): _run_analysis("vectors", "Node vectors", VECTORS[id]))
   %Analysis.add_submenu_node_item("Node vectors", vectors)
   %Analysis.add_separator("", 100000)
   %Analysis.add_item("Link prediction…", ANALYSIS_LINKS)
   %Analysis.add_item("Graph neural networks…", ANALYSIS_GNNS)
   %Analysis.id_pressed.connect(func(id):
      if id == ANALYSIS_LINKS:
         _open_links()
      elif id == ANALYSIS_GNNS:
         _open_gnns())
   %Gnns.add_requested.connect(_add_gnn)
   %Gnns.plot_requested.connect(_plot_gnns)
   %Gnns.visibility_changed.connect(func():
      if not %Gnns.visible:
         _detach.call_deferred(%Gnns))
   %Links.add_requested.connect(_add_link)
   %Links.split_requested.connect(_link_split)
   %Links.plot_requested.connect(_plot_links)
   %Links.signs_requested.connect(_link_signs)
   %Links.visibility_changed.connect(func():
      if not %Links.visible:
         _detach.call_deferred(%Links))
   var projections := PopupMenu.new()
   _menu(projections, PROJECTIONS)
   projections.id_pressed.connect(func(id): _run_backbone(PROJECTIONS[id], "Bipartite projection"))
   var backbones := PopupMenu.new()
   _menu(backbones, BACKBONES)
   backbones.id_pressed.connect(func(id): _run_backbone(BACKBONES[id]))
   var transforms := PopupMenu.new()
   _menu(transforms, TRANSFORMS)
   transforms.id_pressed.connect(func(id): _transform(TRANSFORMS[id]))
   %Edit.add_submenu_node_item("Bipartite projection", projections, 1001)   # ids apart from those of EDITS
   %Edit.add_submenu_node_item("Backbone", backbones, 1002)
   %Edit.add_submenu_node_item("Transform", transforms, 1003)
   %Edit.add_separator()
   _menu(%Edit, EDITS)
   var command := KEY_MASK_META if OS.get_name() == "macOS" else KEY_MASK_CTRL
   for i in EDITS.size():
      if EDIT_KEYS.has(EDITS[i][1]):
         %Edit.set_item_accelerator(%Edit.get_item_index(i), EDIT_KEYS[EDITS[i][1]] | (command if EDITS[i][1] != "remove" else 0))
   %Edit.id_pressed.connect(func(id): if id < EDITS.size(): _edit(EDITS[id]))
   %Backbone.plot_changed.connect(_backbone_replot)
   %Backbone.threshold_changed.connect(_backbone_stats)
   %Backbone.extract_requested.connect(_extract_backbone)
   %Find.text_submitted.connect(_find)
   %Found.id_pressed.connect(func(id):
      if _current().is_empty():
         return
      if id == FOUND_ALL:
         _focus(_current(), _found_exact)
      elif id >= 0 and id < _found_nodes.size():
         _focus(_current(), PackedInt32Array([_found_nodes[id]])))
   _menu(%Select, SELECTS)
   %Select.id_pressed.connect(_on_select)
   %Simulation.add_item("Epidemics…", 0)
   %Simulation.id_pressed.connect(func(_id): _open_epidemics())
   %Epidemics.add_requested.connect(_add_epidemic)
   %Epidemics.plot_requested.connect(_plot_epidemics)
   %Epidemics.visibility_changed.connect(func():
      if not %Epidemics.visible:
         _detach.call_deferred(%Epidemics))
   %Player.step_changed.connect(func(t): if not _current().is_empty(): _go(_current(), t))
   %Player.showing_changed.connect(_show_states)
   %Player.export_requested.connect(_export_animation)
   %Player.colors_changed.connect(func(c): if not _current().is_empty(): _current().palette = c; _show_states(_current().showing))
   $Layout/StatusBar.resized.connect(func(): %Blocker.offset_bottom = -$Layout/StatusBar.size.y)
   %Worker.progress.connect(_task_dialog.progress)
   _task_dialog.abort_requested.connect(%Worker.abort)
   _task_dialog.force_stop_requested.connect(_force_stop)
   _task_dialog.host = _task_host
   _plot.save_requested.connect(_save_plot)
   _plot.changed.connect(_replot)
   var fits := PopupMenu.new()   # the generators that take a network: all but the chain
   _menu(fits, GENERATORS)
   fits.remove_item(0)
   fits.remove_item(0)
   fits.id_pressed.connect(func(id): _fit_generator(GENERATORS[id]))
   %Generators.add_submenu_node_item("Fitted to the current network", fits, 1000)
   %Generators.set_item_disabled(0, true)   # until a network is open
   %Generators.add_separator()
   _menu(%Generators, GENERATORS)
   %Generators.id_pressed.connect(_open_generator)
   _menu(%Help, HELP)
   %Help.id_pressed.connect(func(id): OS.shell_open(HELP[id][1]))
   %Tabs.tab_changed.connect(_show_tab)
   %FitButton.pressed.connect(_fit_view)
   %Tabs.tab_close_pressed.connect(_close_tab)
   %LoadDialog.worker = %Worker
   %LoadDialog.import_requested.connect(func(spec, done): _new_tab(spec.edges.get_file(), {"cmd": "load", "spec": spec}, done))
   %Data.worker = %Worker
   %Data.attribute_menu_requested.connect(_attribute_menu)
   %Data.value_edit_requested.connect(_edit_value)
   add_child(_column_menu)
   _column_menu.id_pressed.connect(func(id): _plot_distribution() if id == 0 else _ask_delete())
   add_child(_context_menu)
   _context_menu.id_pressed.connect(_on_context)
   add_child(_confirm_delete)
   _confirm_delete.ok_button_text = "Delete"
   _confirm_delete.confirmed.connect(_delete_attribute)
   add_child(_large)
   _large.title = "Large network"
   _large.ok_button_text = "Analyze without drawing"
   _large.add_button("Draw anyway", true, "draw")
   _large.custom_action.connect(func(_a): _large.hide(); _set_drawn(_large_tab, true))
   %DrawAnyway.pressed.connect(func(): if not _current().is_empty(): _set_drawn(_current(), true))
   %Visual.style_changed.connect(_restyle)
   %Visual.breaks = _breaks
   %Visual.background_changed.connect(_on_background)
   %Visual.turn_changed.connect(func(d): if not _current().is_empty(): _current().view.set_turn(d))
   %Visual.scale_changed.connect(func(f): if not _current().is_empty(): _current().view.set_spread(f))
   %Worker.started.connect(_on_started)
   %Worker.failed.connect(_on_failed)
   %Worker.progress.connect(_on_progress)
   %EnvButton.pressed.connect(_show_environments)
   %EnvMenu.id_pressed.connect(_on_environment)
   _choose_python = %Message.add_button("Choose Python…", true, "python")
   %Message.custom_action.connect(func(_a): %Message.hide(); _browse_python())
   %PythonDialog.file_selected.connect(_switch_python)
   _show_tab(-1)
   _start_worker()


## Fit button of the visualization panel: the whole network of the current tab fills the panel.
func _fit_view() -> void:
   if not _current().is_empty():
      _current().view.fit()


## Fills a menu with [param items] (label, ...), with their positions as ids.
func _menu(menu: PopupMenu, items: Array) -> void:
   for i in items.size():
      if items[i][0].is_empty():
         menu.add_separator("", 100000 + i)   # an id of its own: no clash with the ids of the items
      else:
         menu.add_item(items[i][0], i)


func _notification(what: int) -> void:
   if what == NOTIFICATION_WM_CLOSE_REQUEST:
      var maximized := get_window().mode == Window.MODE_MAXIMIZED
      AppSettings.store("window", "maximized", maximized)
      if not maximized:
         AppSettings.store("window", "size", get_window().size)


## Interpreter of the worker: the one chosen before, or $GRAPHOSKOP_PYTHON, or python3 / python.
func _python() -> String:
   var chosen: String = AppSettings.value("python", "path", "")
   if not chosen.is_empty():
      return chosen
   if OS.has_environment("GRAPHOSKOP_PYTHON"):
      return OS.get_environment("GRAPHOSKOP_PYTHON")
   return "python" if OS.get_name() == "Windows" else "python3"


func _start_worker() -> void:
   _block(true)
   %Status.text = "Starting Python…"
   %EnvButton.text = "%s (starting)" % Environments.name_of(_python())
   %EnvButton.tooltip_text = _python()
   %Worker.start(_python())


func _on_started(info: Dictionary) -> void:
   _block(false)
   %Status.text = "Ready"
   %EnvButton.text = "%s · Python %s · graphoskop %s · %s" % [info.get("environment", "?"), info.get("python", "?"), info.get("version", "?"), "GPU" if info.get("cuda") else "CPU only"]
   %EnvButton.tooltip_text = "%s\nClick to choose another Python environment." % info.get("executable", "")


func _on_failed(message: String) -> void:
   %Status.text = "Python worker not running"
   %EnvButton.text = "%s (not running)" % Environments.name_of(_python())
   _progress_done()
   _message("Python", message + "\n\nChoose the Python interpreter of an environment with the dependencies of graphoskop.", true)


## Progress bar of the status bar, while a function of the library with a bar runs.
func _on_progress(header: Dictionary) -> void:
   if header.get("done", false):
      _progress_done()
      return
   %Progress.visible = true
   %Remaining.visible = true
   %Progress.value = header.fraction
   %Progress.tooltip_text = header.name
   var left = header.get("remaining")
   %Remaining.text = "%s: %s" % [header.name, "%d:%02d left" % [int(left) / 60, int(left) % 60] if left != null else "starting"]


func _progress_done() -> void:
   %Progress.visible = false
   %Remaining.visible = false


## Greys out the window (all but the status bar) while the worker is not ready (starting, or
## failed to start), so that nothing can be asked of it before (e.g. the load dialog would not
## know yet whether a GPU is available).
func _block(on: bool) -> void:
   %Blocker.visible = on
   %Blocker.offset_bottom = -$Layout/StatusBar.size.y
   for i in %Menus.get_menu_count():
      %Menus.set_menu_disabled(i, on)


## Menu of the environments found (the current one checked), above the environment button.
func _show_environments() -> void:
   _found = Environments.find()
   %EnvMenu.clear()
   for i in _found.size():
      var name := Environments.name_of(_found[i])
      %EnvMenu.add_radio_check_item(_found[i] if name == _found[i] else "%s   %s" % [name, _found[i]], i)
      %EnvMenu.set_item_checked(i, _found[i] == _python())
   %EnvMenu.add_separator()
   %EnvMenu.add_item("Other interpreter…", _found.size())
   %EnvMenu.reset_size()
   var r: Rect2 = %EnvButton.get_global_rect()
   var pos := Vector2i(r.end.x - %EnvMenu.size.x, r.position.y - %EnvMenu.size.y) + get_window().position
   %EnvMenu.popup(Rect2i(pos, %EnvMenu.size))


func _on_environment(id: int) -> void:
   if id < _found.size():
      _switch_python(_found[id])
   else:
      _browse_python()


func _browse_python() -> void:
   %PythonDialog.current_dir = _python().get_base_dir() if _python().is_absolute_path() else AppSettings.folder("python")
   %PythonDialog.popup_centered_ratio(0.6)


## Restarts the worker with [param python], and rebuilds the networks of the open tabs there by
## running their code again, with the same style.
func _switch_python(python: String) -> void:
   if python == _python() and not %Worker.info.is_empty():
      return
   Environments.choose(python)
   _start_worker()
   _rebuild_tabs()


## Rebuilds the networks of the open tabs in the worker by running their code again, with the same
## style (after a change of environment or a restart).
func _rebuild_tabs() -> void:
   for tab in _tabs:
      _task("Rebuilding %s…" % tab.name, {"cmd": "replay", "tab": tab.id, "code": tab.code, "style": tab.style, "undrawn": tab.get("undrawn", false)}, func(reply, payload):
         if not _tabs.has(tab):
            return
         if not reply.ok:
            _message("Python environment", "%s could not be rebuilt, and was closed:\n%s" % [tab.name, reply.error])
            _close_tab(_tabs.find(tab))
            return
         _drawn(tab, reply, payload)
         %Status.text = "Ready")


## Sends a quick [param request] whose reply opens a dialog (the parameters of a function, the
## epidemics dialog, a plot), as a task (see [method _task]): the worker answers one request at a
## time, so while it finishes earlier work (e.g. on a network of millions of nodes) the task dialog
## opens after a moment and says what the interface waits for, instead of nothing happening.
func _ask(request: Dictionary, callback: Callable) -> void:
   var what: String = "the parameters of gsk.%s" % request.function if request.has("function") else {"simulations": "the epidemics", "distribution": "the plot"}.get(request.cmd, "the dialog")
   _task("Opening %s (the Python worker is finishing earlier work)…" % what, request, func(reply, payload):
      if _current().is_empty():
         %Status.text = "Ready"
      else:
         _idle_status(_current())
      callback.call(reply, payload))


## Sends a request that runs the library (it may take long): the status bar says [param text], the
## task dialog opens if it lasts (with Abort), and [param callback] gets the reply.
func _task(text: String, request: Dictionary, callback: Callable) -> void:
   %Status.text = text
   _next_task += 1
   var id := _next_task
   _pending[id] = callback
   _task_dialog.begin(text)
   %Worker.request(request, func(reply, payload):
      _pending.erase(id)
      if _pending.is_empty():
         _task_dialog.end()
      _progress_done()
      callback.call(reply, payload))


## The window the task dialog opens in: the dialog that started the task (which stays open, to
## show an error), so that the task dialog is drawn above it, or else the main window.
func _task_host() -> Window:
   for dialog: Window in [_params, %LoadDialog, %SelectDialog, _plot, %Epidemics, %Links, %Gnns, %Backbone]:
      if dialog.visible:
         return dialog
   return get_window()


## Force stop of the task dialog: the worker is stopped and started again, the tasks fail (their
## dialogs show it), and the open networks are rebuilt from their code.
func _force_stop() -> void:
   var callbacks := _pending.values()
   _pending.clear()
   _task_dialog.end()
   _progress_done()
   _start_worker()
   for callback in callbacks:
      callback.call({"ok": false, "error": "Stopped: the Python worker was restarted."}, PackedByteArray())
   _rebuild_tabs()


func _message(title: String, text: String, python := false) -> void:
   %Message.title = title
   %Message.dialog_text = text
   _choose_python.visible = python
   %Message.popup_centered()


func _on_file(id: int) -> void:
   if id == FILE_LOAD:
      %LoadDialog.open()
   elif id == FILE_IMAGE and not _current().is_empty():
      %ImageDialog.current_dir = AppSettings.folder("images", AppSettings.folder("edges"))
      %ImageDialog.current_file = _current().name.get_basename() + ".png"
      %ImageDialog.popup_centered_ratio(0.6)


## File › Save image: the visualization panel of the current tab as a PNG, at [constant
## IMAGE_SCALE] times its size on screen.
func _save_image(path: String) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   if path.get_extension().to_lower() != "png":
      path += ".png"
   AppSettings.store_folder("images", path)
   var image: Image = await tab.view.capture(IMAGE_SCALE)
   var err := image.save_png(path)
   %Status.text = "Saved %s (%d × %d)" % [path.get_file(), image.get_width(), image.get_height()] if err == OK else "Could not save %s (error %d)" % [path, err]


## Generators menu: asks the worker for the parameters of the generator, then opens its dialog.
func _open_generator(id: int) -> void:
   var item: Array = GENERATORS[id]
   _ask({"cmd": "signature", "function": item[1]}, func(reply, _payload):
      if reply.ok:
         _params.open("%s (gsk.%s)" % [item[0], item[1]], reply, item[2], "Generate", {}, func(arguments, done): _new_tab(item[0], {"cmd": "generate", "function": item[1], "arguments": arguments}, done))
      else:
         _message("Generators", str(reply.error)))


## Generators › Fitted to the current network ([param item] of [constant GENERATORS]): the
## parameter dialog of the generator for the network of the current tab (worker command
## [code]signature[/code] with [code]fit[/code]: parameters empty are taken from the network; a
## measure to test, and the number of null networks). Generate opens one null network in a new
## tab (worker command [code]fit[/code]); with a measure, the randomization test also runs in the
## tab of the network, and its plot opens in the plot window (with the pseudo p-value, right- or
## left-tailed or two-sided).
func _fit_generator(item: Array) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   _ask({"cmd": "signature", "function": item[1], "tab": tab.id, "fit": true}, func(reply, _p):
      if not reply.ok:
         _message("Generators", str(reply.error))
         return
      _params.open("%s fitted to %s (gsk.%s)" % [item[0], tab.name, item[1]], reply, {}, "Generate", tab.attributes, func(arguments, done):
         _next_tab += 1
         var id := _next_tab
         var testing: bool = arguments.has("test")
         _task("%s: %s…" % [tab.name, "testing against %s" % item[1] if testing else "fitting %s" % item[1]], {"cmd": "fit", "tab": tab.id, "new_tab": id, "function": item[1], "arguments": arguments}, func(r, payload):
            if not _tabs.has(tab):
               return
            _idle_status(tab)
            if not r.ok:
               _failed("Generators", str(r.error), done)
               return
            done.call("")
            _open_derived(tab, "%s fitted to %s" % [item[0], tab.name], id, r, payload)
            if r.has("test"):
               var t: Dictionary = r.test
               tab.code = t.code   # the test ran in the tab of the network
               var png: PackedByteArray = payload.slice(payload.size() - int(r.parts.back()))
               if _plot.get_parent() != self:
                  _plot.hide()
                  _plot.reparent(self, false)
               _plot.show_plot(tab.id, "%s of %s against %s" % [t.measure, tab.name, item[0]], png, t.elements, t.options, t.styles, t.text, t.warnings, "randomization_test", t.kinds, t.kind_label))))


## Creates the network of a new tab with [param request] ("load" or "generate"), then lays it out;
## [param done] (from a parameter dialog) gets the error, or an empty text when the network is
## made (without it, errors show in a message).
func _new_tab(name: String, request: Dictionary, done := Callable()) -> void:
   _next_tab += 1
   var id := _next_tab
   request.tab = id
   _task("Creating %s…" % name, request, func(reply, _payload):
      if not reply.ok:
         %Status.text = "Could not create %s" % name
         _failed(name, str(reply.error), done)
         return
      if done.is_valid():
         done.call("")
      var tab := {"id": id, "name": name, "view": NetworkView.instantiate(), "code": reply.code, "summary": "%s: %s" % [name, reply.summary], "style": {}, "attributes": {}, "undrawn": reply.get("too_large", false)}
      _add_tab(tab)
      if tab.undrawn:   # not drawn by the worker: asks whether to draw it anyway
         _drawn(tab, {"code": reply.code, "nodes": 0, "undrawn": true}, PackedByteArray())   # its attributes
         _large_tab = tab
         _large.dialog_text = "%s has %s.\nNetworks with more than %s nodes and edges together can probably not be drawn.\nIt can be laid out, analyzed, simulated and edited without drawing it." % [name, reply.summary, String.num_int64(int(reply.get("max_elements", 10000000)))]
         _large.popup_centered()
      else:
         _layout(tab, "grid_layout", {"fast": "True"}))


## Draws the network of [param tab], or not ([param on] false: worker command
## [code]drawing[/code]; the tab keeps its code, analyses and edits, and the visualization panel
## says it is not drawn). Drawn, a network without a layout yet gets the quick grid layout. (A
## network too large to draw is not drawn from the start: Analyze without drawing has nothing to
## do.)
func _set_drawn(tab: Dictionary, on: bool) -> void:
   if not _tabs.has(tab):
      return
   _task("%s: %s…" % [tab.name, "drawing" if on else "skipping the drawing"], {"cmd": "drawing", "tab": tab.id, "on": on}, func(reply, payload):
      if not _tabs.has(tab):
         return
      _idle_status(tab)
      if not reply.ok:
         _message("Drawing", str(reply.error))
         return
      _drawn(tab, reply, payload)
      if on and int(reply.nodes) == 0:
         _layout(tab, "grid_layout", {"fast": "True"}))


## Adds the tab [param tab] (see [member _tabs]) and makes it current: its view shows the
## selection in the data panel and sends dragged nodes to the worker.
func _add_tab(tab: Dictionary) -> void:
   var v = tab.view
   %Views.add_child(v)
   %Views.move_child(%FitButton, -1)
   v.selection_changed.connect(func(): if _current() == tab: _show_data(tab))
   v.nodes_moved.connect(func(nodes, positions): _move(tab, nodes, positions))
   v.context_requested.connect(func(node, at, _screen): _open_context(tab, node, at))
   v.link_chosen.connect(func(source, target): _link_chosen(tab, source, target))
   %Views.move_child(%Player, -1)
   tab.merge({"spread": {}, "codes": PackedByteArray(), "step": 0, "showing": false, "palette": [], "undo": false, "redo": false})
   _tabs.append(tab)
   %Tabs.add_tab(tab.name)
   %Tabs.current_tab = _tabs.size() - 1
   _show_tab(_tabs.size() - 1)


## The data panel shows the network of [param tab] with its selected nodes and edges.
func _show_data(tab: Dictionary) -> void:
   var v = tab.view
   var nodes: PackedInt32Array = v.selected_nodes()
   %Data.show_network(tab.id, nodes, v.selected_edges, v.ids[nodes[0]] if nodes.size() == 1 else "")
   _idle_status(tab)


## The status bar at rest: the size of the network of [param tab] (if it is the current tab), and
## how many nodes and edges are selected.
func _idle_status(tab: Dictionary) -> void:
   if _current() != tab:
      return
   var v = tab.view
   var chosen := []
   for pair in [[v.selected_nodes().size(), "node"], [v.selected_edges.size(), "edge"]]:
      if pair[0] > 0:
         chosen.append("%d %s%s" % [pair[0], pair[1], "" if pair[0] == 1 else "s"])
   %Status.text = tab.summary + ("" if chosen.is_empty() else "  ·  selected: " + ", ".join(chosen))


## Nodes dragged in the view of [param tab] (positions in the drawing order, new positions in
## drawing coordinates): the worker moves them (code [code]move_nodes[/code]) and draws again.
func _move(tab: Dictionary, nodes: PackedInt32Array, positions: PackedVector2Array) -> void:
   var places := []
   for p in positions:
      places.append([p.x, p.y])
   %Worker.request({"cmd": "move", "tab": tab.id, "nodes": Array(nodes), "positions": places}, func(reply, payload):
      if not _tabs.has(tab):
         return
      if reply.ok:
         _drawn(tab, reply, payload, true)
      else:
         _message("Move nodes", str(reply.error))
         _restyle(tab.style, tab))


## Edit menu ([param item] of [constant EDITS]) on the selection of the current tab: removals
## change its network (the selection is cleared); the induced subgraph and ego networks open in a
## new tab, with the code of the tab plus the new lines, the same style and positions.
func _edit(item: Array) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   var v = tab.view
   if item[1] in ["undo", "redo"]:
      _task("%s: %s…" % [tab.name, item[1]], {"cmd": "undo", "tab": tab.id, "redo": item[1] == "redo"}, func(reply, payload):
         if not _tabs.has(tab):
            return
         if not reply.ok:
            _message("Edit", str(reply.error))
            return
         tab.summary = "%s: %s" % [tab.name, reply.summary]
         _idle_status(tab)
         v.set_selection(PackedInt32Array())
         _drawn(tab, reply, payload, true))
      return
   if item[1].begins_with("remove") and v.selected_nodes().is_empty() and v.selected_edges.is_empty():
      %Status.text = "Nothing selected"
      return
   var request := {"cmd": "edit", "tab": tab.id, "op": item[1], "nodes": Array(v.selected_nodes()), "edges": Array(v.selected_edges), "include_ego": item[2]}
   var name := ""
   if item[1] in ["subgraph", "ego"]:
      _next_tab += 1
      request.new_tab = _next_tab
      name = ("Subgraph of %s" if item[1] == "subgraph" else "Ego network of %s") % tab.name
   _task("%s: %s…" % [tab.name, item[0].to_lower()], request, func(reply, payload):
      if not _tabs.has(tab):
         return
      if not reply.ok:
         _idle_status(tab)
         _message("Edit", str(reply.error))
         return
      if name.is_empty():
         tab.summary = "%s: %s" % [tab.name, reply.summary]
         _idle_status(tab)
         v.set_selection(PackedInt32Array())
         _drawn(tab, reply, payload, true)
         return
      _open_derived(tab, name, request.new_tab, reply, payload))


## Right click in the view of [param tab]: on the [param node] (a position in the drawing order),
## a menu to add an edge from it; elsewhere, to add a node at [param at] (drawing coordinates; only
## in node-link drawings with the positions of a layout), one item per layer of multilayer networks
## and per node type of bipartite ones (from the drawing, [code]adding[/code] of the worker).
## Nothing is asked: the new node gets the first free integer id, attributes are missing.
func _open_context(tab: Dictionary, node: int, at: Vector2) -> void:
   var adding = tab.view.adding
   _context = {"tab": tab, "node": node, "at": at, "choices": []}
   _context_menu.clear()
   if node >= 0:
      _context_menu.add_item("Add an edge from %s" % tab.view.ids[node], 0)
   else:
      var choices := [{}]
      if adding is Dictionary:
         for key in ["layers", "types"]:
            if adding.get(key) != null:
               var more := []
               for c in choices:
                  for v in adding[key]:
                     more.append(c.merged({"layer" if key == "layers" else "type": v}))
               choices = more
      _context.choices = choices
      for k in choices.size():
         var c: Dictionary = choices[k]
         var parts := []
         if c.has("type"):
            parts.append("of type %s" % c.type)
         if c.has("layer"):
            parts.append("in layer %s" % c.layer)
         _context_menu.add_item(" ".join(["Add a node here"] + parts), 1 + k)
         if not tab.view.can_add_nodes() or not adding is Dictionary:
            _context_menu.set_item_disabled(k, true)
            _context_menu.set_item_tooltip(k, "Nodes are added where the positions come from a layout (not in matrix views, nor with positions from node attributes), and not to hypergraphs or simplicial complexes.")
   if node >= 0 and not adding is Dictionary:
      _context_menu.set_item_disabled(0, true)
      _context_menu.set_item_tooltip(0, "Edges cannot be added to hypergraphs or simplicial complexes.")
   _context_menu.reset_size()
   _context_menu.popup(Rect2i(get_window().position + Vector2i(get_window().get_mouse_position()), Vector2i.ZERO))


func _on_context(id: int) -> void:
   var tab: Dictionary = _context.get("tab", {})
   if not _tabs.has(tab):
      return
   if id == 0:
      tab.view.start_link(_context.node)
      %Status.text = "Click the node the new edge from %s goes to (Escape or right click cancels)" % tab.view.ids[_context.node]
   else:
      var request := {"cmd": "add", "tab": tab.id, "kind": "node", "position": [_context.at.x, _context.at.y]}
      request.merge(_context.choices[id - 1])
      _added(tab, request)


## The target of a new edge of [param tab] was clicked (-1: canceled): the edge is added, its
## attributes missing (weight 1).
func _link_chosen(tab: Dictionary, source: int, target: int) -> void:
   _idle_status(tab)
   if target >= 0 and _tabs.has(tab):
      _added(tab, {"cmd": "add", "tab": tab.id, "kind": "edge", "source": source, "target": target})


## Adds the node or edge of [param request] to the network of [param tab] (worker command
## [code]add[/code]; Edit › Undo takes it back); errors go to a message.
func _added(tab: Dictionary, request: Dictionary) -> void:
   _task("%s: adding %s…" % [tab.name, "a node" if request.kind == "node" else "an edge"], request, func(reply, payload):
      if not _tabs.has(tab):
         return
      if not reply.ok:
         _idle_status(tab)
         _message("Add", str(reply.error))
         return
      tab.summary = "%s: %s" % [tab.name, reply.summary]
      _drawn(tab, reply, payload, true))


## Opens a new tab [param name] (worker tab [param id]) with a network derived from the one of
## [param tab] (subgraph, ego network, backbone), from the [param reply] of the worker: same
## background and rotation of the view.
func _open_derived(tab: Dictionary, name: String, id: int, reply: Dictionary, payload: PackedByteArray) -> void:
   var new := {"id": id, "name": name, "view": NetworkView.instantiate(), "code": reply.code, "summary": "%s: %s" % [name, reply.summary], "style": {"background": tab.style.get("background", "#ffffff")}, "attributes": {}}
   new.view.turn = tab.view.turn
   new.view.spread = tab.view.spread
   _add_tab(new)
   _drawn(new, reply, payload)
   if int(reply.nodes) == 0 and not reply.get("undrawn", false):   # no layout yet (the network had none)
      _layout(new, "grid_layout", {"fast": "True"})


## Edit › Backbone ([param item] of [constant BACKBONES]) and Edit › Bipartite projection (of
## [constant PROJECTIONS], with [param what] [code]"Bipartite projection"[/code]): the parameter
## dialog of the function; a scoring function scores the edges of the current tab (worker command
## [code]backbone[/code]) and opens the backbone dialog (backbone_dialog.gd); a filter (spanning
## tree, planar filtered graphs) and a projection open their network in a new tab (worker command
## [code]extract[/code]).
func _run_backbone(item: Array, what := "Backbone") -> void:
   var tab := _current()
   if tab.is_empty():
      return
   _ask({"cmd": "signature", "function": item[1], "tab": tab.id}, func(reply, _p):
      if not reply.ok:
         _message(what, str(reply.error))
         return
      _params.open("%s (gsk.%s)" % [item[0].trim_suffix("…"), item[1]], reply, {}, "Run", tab.attributes, func(arguments, done):
         if item[2]:   # a filter or a projection: in a new tab
            _next_tab += 1
            var id := _next_tab
            _task("%s: running %s…" % [tab.name, item[1]], {"cmd": "extract", "tab": tab.id, "new_tab": id, "function": item[1], "arguments": arguments}, func(r, payload):
               if not _tabs.has(tab):
                  return
               _idle_status(tab)
               if not r.ok:
                  _failed(what, str(r.error), done)
                  return
               done.call("")
               var kind := " projection" if what != "Backbone" else ""
               _open_derived(tab, "%s%s of %s" % [item[0].trim_suffix("…"), kind, tab.name], id, r, payload))
            return
         _task("%s: running %s…" % [tab.name, item[1]], {"cmd": "backbone", "tab": tab.id, "function": item[1], "arguments": arguments}, func(r, payload):
            if not _tabs.has(tab):
               return
            _idle_status(tab)
            if not r.ok:
               _failed("Backbone", str(r.error), done)
               return
            done.call("")
            var parts: Array = r.parts
            var png: PackedByteArray = payload.slice(payload.size() - int(parts.back()))
            _drawn(tab, r, payload, true)
            _backbone_tab = tab
            var b: Dictionary = r.backbone
            var start = b.found[0][1] if b.found[0][1] != null else b.box[4]
            %Backbone.open("%s backbone of %s" % [item[0].trim_suffix("…"), tab.name], png, b.box, b.scale, b.tails, b.range, b.found, start))))


## Edit › Transform ([param item] of [constant TRANSFORMS]): the transformation of the network of
## the current tab, after its parameter dialog if it has parameters, in a new tab (worker command
## [code]extract[/code], e.g. [code]net = gsk.to_undirected(net)[/code]); errors of the library
## (e.g. a transformation of directed networks on an undirected one) show in the dialog, or in a
## message.
func _transform(item: Array) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   var run := func(arguments: Dictionary, done: Callable) -> void:
      _next_tab += 1
      var id := _next_tab
      _task("%s: running %s…" % [tab.name, item[1]], {"cmd": "extract", "tab": tab.id, "new_tab": id, "function": item[1], "arguments": arguments}, func(r, payload):
         if not _tabs.has(tab):
            return
         _idle_status(tab)
         if not r.ok:
            _failed("Transform", str(r.error), done)
            return
         if done.is_valid():
            done.call("")
         _open_derived(tab, "%s (%s)" % [tab.name, item[0].trim_suffix("…").to_lower()], id, r, payload))
   if not item[2]:
      run.call({}, Callable())
      return
   _ask({"cmd": "signature", "function": item[1], "tab": tab.id}, func(reply, _p):
      if not reply.ok:
         _message("Transform", str(reply.error))
         return
      _params.open("%s (gsk.%s)" % [item[0].trim_suffix("…"), item[1]], reply, {}, "Run", tab.attributes, run))


## Backbone dialog: the plot again, with or without the [param components] and the [param left]
## and [param right] tails.
func _backbone_replot(components: bool, left: bool, right: bool) -> void:
   var tab := _backbone_tab
   if _tabs.has(tab):
      %Worker.request({"cmd": "backbone_plot", "tab": tab.id, "components": components, "left": left, "right": right}, func(reply, payload):
         if reply.ok:
            %Backbone.set_plot(payload, reply.box, reply.scale))


## Backbone dialog: the size of the backbone at the chosen [param threshold].
func _backbone_stats(threshold: float) -> void:
   var tab := _backbone_tab
   if _tabs.has(tab):
      %Worker.request({"cmd": "backbone_stats", "tab": tab.id, "threshold": threshold}, func(reply, _p):
         if reply.ok and is_equal_approx(%Backbone.threshold(), threshold):
            %Backbone.show_stats(reply))


## Backbone dialog › Extract backbone: the backbone at [param threshold] in a new tab; the dialog
## closes once it is open.
func _extract_backbone(threshold: float, options: Dictionary) -> void:
   var tab := _backbone_tab
   if not _tabs.has(tab):
      return
   _next_tab += 1
   var id := _next_tab
   _task("%s: extracting the backbone…" % tab.name, {"cmd": "extract", "tab": tab.id, "new_tab": id, "threshold": threshold, "options": options}, func(r, payload):
      if not _tabs.has(tab):
         return
      if not r.ok:
         _message("Backbone", str(r.error))
         return
      %Backbone.hide()
      _open_derived(tab, "Backbone of %s" % tab.name, id, r, payload))


## Search box of the tab bar: the nodes of the current tab whose id (or the attribute chosen in
## the menu) matches [param text] (worker command [code]find[/code]). A single exact match is
## selected and shown; otherwise a menu lists the matches (exact ones first) to pick one, with an
## item selecting all the exact matches if there are several.
func _find(text: String) -> void:
   var tab := _current()
   if tab.is_empty() or text.strip_edges().is_empty():
      return
   var by = null if %FindBy.selected <= 0 else %FindBy.get_item_text(%FindBy.selected)
   %Worker.request({"cmd": "find", "tab": tab.id, "attribute": by, "text": text}, func(reply, _p):
      if not reply.ok or not _tabs.has(tab):
         if not reply.ok:
            _message("Find", str(reply.error))
         return
      var exact: Array = reply.exact
      var found: Array = exact + reply.partial
      if reply.exact_count == 1:
         _focus(tab, PackedInt32Array([exact[0]]))
         return
      if found.is_empty():
         %Status.text = "No node matches \"%s\"" % text
         return
      _found_nodes = PackedInt32Array(found)
      _found_exact = PackedInt32Array(exact)
      %Found.clear()
      if reply.exact_count > 1:
         %Found.add_item("Select the %d nodes with \"%s\"%s" % [reply.exact_count, text, "" if reply.exact_count <= exact.size() else " (the first %d)" % exact.size()], FOUND_ALL)
         %Found.add_separator()
      for k in found.size():
         %Found.add_item(reply.labels[k], k)
      var more: int = reply.exact_count + reply.partial_count - found.size()
      if more > 0:
         %Found.add_item("… %d more: refine the search" % more, FOUND_MORE)
         %Found.set_item_disabled(%Found.item_count - 1, true)
      %Found.reset_size()
      var r: Rect2 = %Find.get_global_rect()
      %Found.popup(Rect2i(Vector2i(r.position + Vector2(0, r.size.y)) + get_window().position, Vector2i(int(r.size.x), 0))))


## Selects the nodes [param nodes] of [param tab] and centers the view on the first, zooming in
## if the nodes are too small to be seen.
func _focus(tab: Dictionary, nodes: PackedInt32Array) -> void:
   tab.view.set_selection(nodes)
   tab.view.focus(nodes[0])


## Fills the menu of the search box: the ids, then the node attributes of the current tab (the
## choice is kept while it exists).
func _find_choices(tab: Dictionary) -> void:
   var keep: String = %FindBy.get_item_text(%FindBy.selected) if %FindBy.selected >= 0 else "id"
   %FindBy.clear()
   %FindBy.add_item("id")
   for a in tab.get("attributes", {}).get("nodes", []):
      %FindBy.add_item(a.name)
   %FindBy.select(0)
   for i in %FindBy.item_count:
      if %FindBy.get_item_text(i) == keep:
         %FindBy.select(i)


## Select menu: all nodes, none, or by attribute (select_dialog.gd).
func _on_select(id: int) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   var v = tab.view
   if id == 0:
      v.set_selection(PackedInt32Array(range(v.ids.size())))
   elif id == 1:
      v.set_selection(PackedInt32Array())
   else:
      %SelectDialog.open(tab.attributes, func(kind, attribute, op, value, add, done):
         %Worker.request({"cmd": "select", "tab": tab.id, "kind": kind, "attribute": attribute, "op": op, "value": value}, func(reply, _p):
            if not _tabs.has(tab):
               return
            if not reply.ok:
               done.call(str(reply.error))
               return
            var found := PackedInt32Array(reply.indices)
            if found.is_empty():
               done.call("No %s match." % kind)
               return
            var nodes: PackedInt32Array = v.selected_nodes() if add else PackedInt32Array()
            var edges: PackedInt32Array = v.selected_edges if add else PackedInt32Array()
            if kind == "nodes":
               nodes.append_array(found)
            else:
               edges.append_array(found)
            v.set_selection(nodes, edges)
            done.call("")))


## Simulation › Epidemics: opens the dialog of the runs of the current tab (epidemics_dialog.gd).
func _open_epidemics() -> void:
   var tab := _current()
   if tab.is_empty():
      return
   _ask({"cmd": "simulations", "tab": tab.id}, func(reply, _p):
      if reply.ok and _tabs.has(tab):
         _epidemics_tab = tab
         %Epidemics.open("Epidemics of %s" % tab.name, reply))


## Add model or Copy… of the epidemics dialog: the parameter dialog of the model [param function],
## whose run is named [param name] (it can be changed), with the [param values] of the run copied;
## the run is added to the dialog. The dialog says what is going on until the worker is done with
## the run (the model, then the attributes and the history of the network).
func _add_epidemic(function: String, name: String, values: Dictionary) -> void:
   var tab := _epidemics_tab
   if not _tabs.has(tab):
      return
   _ask({"cmd": "signature", "function": function, "tab": tab.id}, func(reply, _p):
      if not reply.ok:
         %Epidemics.show_error(str(reply.error))
         return
      _above(_params, %Epidemics)
      var presets := {"history": "True", "onsets": "True"}
      presets.merge(values, true)
      presets.name = JSON.stringify(name)
      _params.open("%s (gsk.%s)" % [name.rsplit(" ", true, 1)[0], function], reply, presets, "Run", tab.attributes, func(arguments, done):
         var run: String = JSON.parse_string(arguments.get("name", "\"\"")) if arguments.has("name") else ""
         %Epidemics.show_status("Running %s…" % run)
         _task("%s: running %s…" % [tab.name, function], {"cmd": "simulate", "tab": tab.id, "function": function, "arguments": arguments}, func(r, payload):
            if not _tabs.has(tab):
               return
            _idle_status(tab)
            if not r.ok:
               %Epidemics.show_status("")
               _failed("Epidemics", str(r.error), done)
               return
            done.call("")
            %Epidemics.show_runs(r)
            %Epidemics.show_status("%s done; updating the attributes of the network and the history of the spread…" % run)
            _drawn(tab, r, payload, true)
            %Worker.request({"cmd": "simulations", "tab": tab.id}, func(s, _q):   # after the requests of the drawing: all done
               if _epidemics_tab == tab:
                  %Epidemics.show_status("")
                  if s.ok:
                     %Epidemics.show_runs(s)))))


## Plot of the epidemics dialog: the runs [param names] in the [param states], in one plot.
func _plot_epidemics(names: Array, states: Array) -> void:
   var tab := _epidemics_tab
   if not _tabs.has(tab):
      return
   %Worker.request({"cmd": "epidemic_plot", "tab": tab.id, "names": names, "states": states}, func(reply, payload):
      if not reply.ok:
         %Epidemics.show_error(str(reply.error))
         return
      %Epidemics.show_error("")
      _above(_plot, %Epidemics)
      _plot.show_plot(tab.id, "Epidemics of %s" % tab.name, payload, reply.elements, reply.options, reply.styles, reply.text, reply.warnings, "epidemics"))


## Analysis › Link prediction: opens the dialog of the link predictions of the current tab
## (link_dialog.gd; worker command [code]link_predictions[/code]).
func _open_links() -> void:
   var tab := _current()
   if tab.is_empty():
      return
   _ask({"cmd": "link_predictions", "tab": tab.id}, func(reply, _p):
      if reply.ok and _tabs.has(tab):
         _links_tab = tab
         _link_info = reply
         %Links.open("Link prediction in %s" % tab.name, reply))


## The runs of the link prediction dialog changed ([param reply] of the worker, with the code and
## the drawing).
func _links_done(tab: Dictionary, reply: Dictionary, payload: PackedByteArray) -> void:
   _link_info = reply
   %Links.show_status("")
   %Links.show_runs(reply)
   _drawn(tab, reply, payload, true)


## Add method or Copy… of the link prediction dialog: the parameter dialog of [param function],
## whose run is named [param name] (it can be changed), with the [param values] of the run copied
## (worker command [code]link_predict[/code]); ensembles choose among the runs so far.
func _add_link(function: String, name: String, values: Dictionary) -> void:
   var tab := _links_tab
   if not _tabs.has(tab):
      return
   _ask({"cmd": "signature", "function": function, "tab": tab.id}, func(reply, _p):
      if not reply.ok:
         %Links.show_error(str(reply.error))
         return
      _above(_params, %Links)
      var presets := values.duplicate()
      presets.name = JSON.stringify(name)
      var choices: Dictionary = tab.attributes.duplicate()
      choices.runs = _link_info.get("runs", []).map(func(r): return {"name": r.name, "numeric": true})
      _params.open("%s (gsk.%s)" % [name.rsplit(" ", true, 1)[0], function], reply, presets, "Run", choices, func(arguments, done):
         var run: String = JSON.parse_string(arguments.get("name", "\"\"")) if arguments.has("name") else ""
         %Links.show_status("Running %s…" % run)
         _task("%s: running %s…" % [tab.name, function], {"cmd": "link_predict", "tab": tab.id, "function": function, "arguments": arguments}, func(r, payload):
            if not _tabs.has(tab):
               return
            _idle_status(tab)
            if not r.ok:
               %Links.show_status("")
               _failed("Link prediction", str(r.error), done)
               return
            done.call("")
            _links_done(tab, r, payload))))


## Validation set… of the link prediction dialog: the parameter dialog of
## gsk.link_prediction_split; a new validation set drops the runs (worker command
## [code]link_split[/code]).
func _link_split() -> void:
   var tab := _links_tab
   if not _tabs.has(tab):
      return
   _ask({"cmd": "signature", "function": "link_prediction_split", "tab": tab.id}, func(reply, _p):
      if not reply.ok:
         %Links.show_error(str(reply.error))
         return
      _above(_params, %Links)
      _params.open("Validation set (gsk.link_prediction_split)", reply, _link_info.get("split", {}) if _link_info.get("split") != null else {}, "Make", tab.attributes, func(arguments, done):
         _task("%s: making the validation set…" % tab.name, {"cmd": "link_split", "tab": tab.id, "arguments": arguments}, func(r, payload):
            if not _tabs.has(tab):
               return
            _idle_status(tab)
            if not r.ok:
               _failed("Link prediction", str(r.error), done)
               return
            done.call("")
            _links_done(tab, r, payload))))


## Plot of the link prediction dialog: the ROC or precision-recall curves ([param kind]) of the
## runs [param names] (worker command [code]link_plot[/code]; the plot window can switch curve).
func _plot_links(names: Array, kind: String) -> void:
   var tab := _links_tab
   if not _tabs.has(tab):
      return
   %Worker.request({"cmd": "link_plot", "tab": tab.id, "names": names, "kind": kind}, func(reply, payload):
      if not reply.ok:
         %Links.show_error(str(reply.error))
         return
      %Links.show_error("")
      _above(_plot, %Links)
      _plot.show_plot(tab.id, "Link prediction in %s" % tab.name, payload, reply.elements, reply.options, reply.styles, reply.text, reply.warnings, "link_prediction", reply.kinds, reply.kind_label))


## Signs of the link prediction dialog (signed networks): how well social balance and status
## predict the signs of the hidden edges (worker command [code]link_signs[/code]), under the runs.
func _link_signs() -> void:
   var tab := _links_tab
   if not _tabs.has(tab):
      return
   %Links.show_status("Predicting the signs of the hidden edges…")
   _task("%s: predicting signs…" % tab.name, {"cmd": "link_signs", "tab": tab.id}, func(r, payload):
      if not _tabs.has(tab):
         return
      _idle_status(tab)
      if not r.ok:
         %Links.show_status("")
         %Links.show_error(str(r.error))
         return
      _links_done(tab, r, payload)
      %Links.show_status(r.text))


## Analysis › Graph neural networks: opens the dialog of the graph neural networks of the current
## tab (gnn_dialog.gd; worker command [code]gnns[/code]).
func _open_gnns() -> void:
   var tab := _current()
   if tab.is_empty():
      return
   _ask({"cmd": "gnns", "tab": tab.id}, func(reply, _p):
      if reply.ok and _tabs.has(tab):
         _gnns_tab = tab
         %Gnns.open("Graph neural networks of %s" % tab.name, reply))


## Add model or Copy… of the graph neural network dialog: the parameter dialog of [param function],
## whose run is named [param name] (it can be changed), with the [param values] of the run copied
## (worker command [code]gnn[/code]).
func _add_gnn(function: String, name: String, values: Dictionary) -> void:
   var tab := _gnns_tab
   if not _tabs.has(tab):
      return
   _ask({"cmd": "signature", "function": function, "tab": tab.id}, func(reply, _p):
      if not reply.ok:
         %Gnns.show_error(str(reply.error))
         return
      _above(_params, %Gnns)
      var presets := values.duplicate()
      presets.name = JSON.stringify(name)
      _params.open("%s (gsk.%s)" % [name.rsplit(" ", true, 1)[0], function], reply, presets, "Train", tab.attributes, func(arguments, done):
         var run: String = JSON.parse_string(arguments.get("name", "\"\"")) if arguments.has("name") else ""
         %Gnns.show_status("Training %s…" % run)
         _task("%s: training %s…" % [tab.name, function], {"cmd": "gnn", "tab": tab.id, "function": function, "arguments": arguments}, func(r, payload):
            if not _tabs.has(tab):
               return
            _idle_status(tab)
            if not r.ok:
               %Gnns.show_status("")
               _failed("Graph neural networks", str(r.error), done)
               return
            done.call("")
            %Gnns.show_runs(r)
            %Gnns.show_status(r.get("note", ""))
            _drawn(tab, r, payload, true))))


## Plot of the graph neural network dialog: the runs [param names] as ROC or precision-recall
## curves, predicted against observed values or learning curves ([param kind]; worker command
## [code]gnn_plot[/code]; the plot window can switch).
func _plot_gnns(names: Array, kind: String) -> void:
   var tab := _gnns_tab
   if not _tabs.has(tab):
      return
   %Worker.request({"cmd": "gnn_plot", "tab": tab.id, "names": names, "kind": kind}, func(reply, payload):
      if not reply.ok:
         %Gnns.show_error(str(reply.error))
         return
      %Gnns.show_error("")
      _above(_plot, %Gnns)
      _plot.show_plot(tab.id, "Graph neural networks of %s" % tab.name, payload, reply.elements, reply.options, reply.styles, reply.text, reply.warnings, "graph_neural_networks", reply.kinds, reply.kind_label))


## Makes [param dialog] a child window of [param host], so that it is drawn above it (see
## task_dialog.gd); it goes back to the main window when the host closes ([method _detach]).
func _above(dialog: Window, host: Window) -> void:
   if dialog.get_parent() != host:
      dialog.reparent(host, false)


## The epidemics or link prediction dialog ([param host]) closed: the parameter dialog (closed)
## and the plot window (still open, if it was) go back to the main window.
func _detach(host: Window) -> void:
   for dialog: Window in [_params, _plot]:
      if dialog.get_parent() == host:
         var open := dialog.visible and dialog == _plot
         dialog.hide()
         dialog.reparent(self, false)
         if open:
            dialog.popup_centered()


## Player: the network of [param tab] shows the states of step [param t] of its history (the
## changes from the step shown are applied, and only the nodes they touch are colored again).
func _go(tab: Dictionary, t: int) -> void:
   var sp: Dictionary = tab.spread
   if sp.is_empty():
      return
   var codes: PackedByteArray = tab.codes
   var touched := PackedInt32Array()
   var step: int = tab.step
   while step < t:
      step += 1
      for j in range(sp.starts[step - 1], sp.starts[step]):
         codes[sp.nodes[j]] = sp.new[j]
         touched.append(sp.nodes[j])
   while step > t:
      for j in range(sp.starts[step - 1], sp.starts[step]):
         codes[sp.nodes[j]] = sp.old[j]
         touched.append(sp.nodes[j])
      step -= 1
   tab.codes = codes
   tab.step = step
   var all: bool = not tab.showing or touched.size() > codes.size() / 4
   tab.showing = true
   tab.view.show_states(codes, tab.palette, touched, all)


## Player: States on (the nodes of the current tab show their states) or off (the colors of the
## style).
func _show_states(on: bool) -> void:
   var tab := _current()
   if tab.is_empty() or tab.spread.is_empty():
      return
   tab.showing = on
   if on:
      tab.view.show_states(tab.codes, tab.palette)
   else:
      tab.view.hide_states()


## Asks the worker for the history of the epidemic of [param tab] (command [code]spread[/code]),
## shown by the player from step 0, the nodes keeping their colors until the player is used.
func _fetch_spread(tab: Dictionary, key: String) -> void:
   tab.spread_key = key
   tab.spread = {}
   if tab.showing:
      tab.view.hide_states()
   tab.showing = false
   if key.is_empty():
      _show_player()
      return
   %Worker.request({"cmd": "spread", "tab": tab.id}, func(reply, payload):
      if not _tabs.has(tab) or tab.spread_key != key or not reply.ok or int(reply.steps) < 0:
         return
      var parts := []
      var at := 0
      for n in reply.parts:
         parts.append(payload.slice(at, at + int(n)))
         at += int(n)
      tab.spread = {"name": reply.name, "steps": int(reply.steps), "states": reply.states, "run": reply.run, "initial": parts[0], "starts": parts[1].to_int32_array(), "nodes": parts[2].to_int32_array(), "new": parts[3], "old": parts[4]}
      tab.codes = parts[0].duplicate()
      tab.step = 0
      tab.palette = reply.colors.map(func(c): return Color(c))
      _show_player())


## Player › Export…: every step of the history of the current tab is drawn and captured (as on
## screen, with the states colored), then the worker makes the animation (command
## [code]animation[/code]: a GIF, or an MP4 video with ffmpeg) at [param fps] frames per second,
## with the step written on every frame; the player then shows the step it showed before.
func _export_animation(path: String, fps: float) -> void:
   var tab := _current()
   if tab.is_empty() or tab.spread.is_empty():
      return
   var sp: Dictionary = tab.spread
   var back: int = tab.step
   var showing: bool = tab.showing
   var folder := OS.get_user_data_dir().path_join("frames")
   DirAccess.make_dir_recursive_absolute(folder)
   var frames := []
   var labels := []
   %Player.set_busy(true)
   for t in sp.steps + 1:
      %Status.text = "Exporting the animation: step %d of %d…" % [t, sp.steps]
      _go(tab, t)
      var image: Image = await tab.view.capture(1)
      if not _tabs.has(tab):
         return
      var file := folder.path_join("frame_%06d.png" % t)
      image.save_png(file)
      frames.append(file)
      labels.append("step %d / %d" % [t, sp.steps])
   _go(tab, back)
   if not showing:
      _show_states(false)
   %Player.set_busy(false)
   _task("Exporting %s…" % path.get_file(), {"cmd": "animation", "frames": frames, "path": path, "fps": fps, "labels": labels}, func(reply, _p):
      if reply.ok:
         %Status.text = "Saved %s (%d frames)" % [path.get_file(), frames.size()]
      else:
         _message("Export the animation", str(reply.error)))


## Shows the player under the network of the current tab if it has a history.
func _show_player() -> void:
   var tab := _current()
   var sp: Dictionary = tab.get("spread", {})
   %Player.visible = not sp.is_empty()
   if not sp.is_empty():
      %Player.show_history(sp.run if not sp.run.is_empty() else sp.name, sp.steps, sp.states, tab.palette.map(func(c): return "#" + c.to_html(false)), tab.step, tab.showing)


## The current tab, or an empty dictionary.
func _current() -> Dictionary:
   var i: int = %Tabs.current_tab
   return _tabs[i] if i >= 0 and i < _tabs.size() else {}


## Layout menu: opens the parameter dialog of the chosen layout for the network of the current tab
## (its attributes are offered for the parameters that take attribute names), or runs a
## transformation at once.
func _run_layout(id: int) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   var item: Array = LAYOUTS[id]
   if not item[3]:
      _layout(tab, item[1], item[2])
      return
   _ask({"cmd": "signature", "function": item[1], "tab": tab.id}, func(reply, _payload):
      if reply.ok:
         _params.open("%s (gsk.%s)" % [item[0].trim_suffix("…"), item[1]], reply, item[2], "Run", tab.attributes, func(arguments, done): _layout(tab, item[1], arguments, done))
      else:
         _message("Layout", str(reply.error)))


## Analysis menu: opens the parameter dialog of an analysis ([param item]: label and function) for
## the current network, then runs it with the worker command [param cmd] ("communities",
## "centrality", "nestedness" or "vectors"): the results become attributes; the look of the
## drawing does not change. Degree centrality can also plot the degree distribution, and
## nestedness opens its plot; the core-periphery models show their fit in the status bar; node
## vector analyses show their value in a message (and in the status bar), or open their plot.
func _run_analysis(cmd: String, heading: String, item: Array) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   _ask({"cmd": "signature", "function": item[1], "tab": tab.id}, func(reply, _payload):
      if not reply.ok:
         _message(heading, str(reply.error))
         return
      _params.open("%s (gsk.%s)" % [item[0].trim_suffix("…"), item[1]], reply, item[2] if item.size() > 2 else {}, "Run", tab.attributes, func(arguments, done):
         _task("%s: running %s…" % [tab.name, item[1]], {"cmd": cmd, "tab": tab.id, "function": item[1], "arguments": arguments, "name": tab.name}, func(r, payload):
            if not _tabs.has(tab):
               return
            _idle_status(tab)
            if not r.ok:
               _failed(heading, str(r.error), done)
               return
            done.call("")
            _drawn(tab, r, payload, true)
            if r.has("note") and _current() == tab:   # e.g. the correlation of a core-periphery model
               %Status.text = "%s  ·  %s" % [tab.summary, r.note]
            if r.has("result"):   # the value of a node vector analysis
               if _current() == tab:
                  %Status.text = "%s  ·  %s" % [tab.summary, r.result]
               _message(heading, r.result)
            if r.has("plot"):
               var parts: Array = r.parts
               var png: PackedByteArray = payload.slice(payload.size() - int(parts.back()))
               if _plot.get_parent() != self:   # out of the epidemics dialog
                  _plot.hide()
                  _plot.reparent(self, false)
               var file: String = item[1] if cmd != "centrality" else "degree_distribution"
               _plot.show_plot(tab.id, r.plot.get("title", "Degree distribution of %s" % tab.name), png, r.plot.elements, r.plot.options, r.plot.styles, r.plot.text, r.plot.warnings, file, [], "Plot", r.plot.get("table", false)))))


## Draws the plot of the plot window again with new [param colors] (element key to color),
## [param styles] (data series key to style) and [param options] (axes, font size, labels,
## ranges).
func _replot(tab_id: int, colors: Dictionary, styles: Dictionary, options: Dictionary) -> void:
   %Worker.request({"cmd": "replot", "tab": tab_id, "colors": colors, "styles": styles, "options": options}, func(reply, payload):
      if reply.ok:
         _plot.set_image(payload)
         if reply.has("text"):   # e.g. the pseudo p-value of another tail
            _plot.show_text(reply.text)
      else:
         _message("Plot", str(reply.error)))


## Right click on the title of the column of [param attribute] in the Nodes or Edges table
## ([param kind] "nodes" or "edges"): a menu to plot its distribution or to delete it.
func _attribute_menu(kind: String, attribute: String) -> void:
   if _current().is_empty():
      return
   _column = [kind, attribute]
   _column_menu.clear()
   _column_menu.add_item("Plot the distribution of %s…" % attribute, 0)
   _column_menu.add_item("Delete %s…" % attribute, 1)
   _column_menu.popup(Rect2i(get_window().position + Vector2i(get_window().get_mouse_position()), Vector2i.ZERO))


## A value edited in the data panel (worker command [code]set_value[/code]: one line of code,
## which Edit › Undo takes back); the drawing follows (the style may use the attribute), and the
## table shows the value again if the worker refuses it.
func _edit_value(kind: String, position: int, attribute: String, typed: String) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   %Worker.request({"cmd": "set_value", "tab": tab.id, "kind": kind, "position": position, "attribute": attribute, "text": typed}, func(reply, payload):
      if not _tabs.has(tab):
         return
      if not reply.ok:
         _message("Edit value", str(reply.error))
         if _current() == tab:
            _show_data(tab)
         return
      _drawn(tab, reply, payload, true))


## Column menu › Plot: the distribution of the attribute in the plot window (worker command
## [code]distribution[/code]; count, PDF, CDF or CCDF, chosen there).
func _plot_distribution() -> void:
   var tab := _current()
   if tab.is_empty():
      return
   var kind: String = _column[0]
   var attribute: String = _column[1]
   _ask({"cmd": "distribution", "tab": tab.id, "kind": kind, "attribute": attribute}, func(reply, payload):
      if not reply.ok:
         _message("Plot", str(reply.error))
         return
      if _plot.get_parent() != self:   # out of the epidemics dialog
         _plot.hide()
         _plot.reparent(self, false)
      _plot.show_plot(tab.id, "Distribution of the %s attribute %s of %s" % [kind.trim_suffix("s"), attribute, tab.name], payload, reply.elements, reply.options, reply.styles, reply.text, reply.warnings, attribute, reply.kinds))


## Column menu › Delete: asks to confirm (an attribute cannot be put back).
func _ask_delete() -> void:
   _confirm_delete.title = "Delete %s" % _column[1]
   _confirm_delete.dialog_text = "Delete the %s attribute %s? This cannot be undone." % [_column[0].trim_suffix("s"), _column[1]]
   _confirm_delete.popup_centered()


## Deletes the attribute of the column menu (worker command [code]remove_attribute[/code]); the
## worker refuses ids, positions and the attributes the drawing uses, and says why.
func _delete_attribute() -> void:
   var tab := _current()
   if tab.is_empty():
      return
   %Worker.request({"cmd": "remove_attribute", "tab": tab.id, "kind": _column[0], "attribute": _column[1]}, func(reply, payload):
      if not _tabs.has(tab):
         return
      if not reply.ok:
         _message("Delete attribute", str(reply.error))
         return
      _drawn(tab, reply, payload, true))


## Saves the plot of the plot window (PNG or PDF, by the extension of [param path]).
func _save_plot(tab_id: int, path: String) -> void:
   %Worker.request({"cmd": "save_plot", "tab": tab_id, "path": path}, func(reply, _p):
      if reply.ok:
         %Status.text = "Saved %s" % path.get_file()
      else:
         _message("Plot", str(reply.error)))


## Runs a layout function of the library on the network of a tab, then draws it; [param done]
## (from a parameter dialog) gets the error, or an empty text when the layout ran.
func _layout(tab: Dictionary, function: String, arguments: Dictionary, done := Callable()) -> void:
   _task("%s: running %s…" % [tab.name, function], {"cmd": "layout", "tab": tab.id, "function": function, "arguments": arguments}, func(reply, payload):
      if not _tabs.has(tab):
         return
      _idle_status(tab)
      if not reply.ok:
         _failed("Layout", str(reply.error), done)
         return
      if done.is_valid():
         done.call("")
      _drawn(tab, reply, payload))


## An error of the worker: shown in the parameter dialog that asked for the run ([param done]),
## which stays open with its values, or else in a message.
func _failed(heading: String, error: String, done: Callable) -> void:
   if done.is_valid():
      done.call(error)
   else:
      _message(heading, error)


## Shows the code and the drawing of a reply in a tab, and refreshes its attributes (the code may
## have added some, e.g. positions) and its data panel; the view keeps its zoom and center with
## [param keep_view] (the positions did not change).
func _drawn(tab: Dictionary, reply: Dictionary, payload: PackedByteArray, keep_view := false) -> void:
   tab.code = reply.code
   tab.undrawn = reply.get("undrawn", false)
   if int(reply.nodes) > 0:
      var background: String = tab.style.get("background", "#ffffff")
      tab.style = reply.style
      tab.style.background = background
      tab.view.show_network(reply, payload, keep_view)
      tab.view.set_background(Color(background))
      if tab.showing and reply.get("history", "") == tab.get("spread_key", ""):
         tab.view.show_states(tab.codes, tab.palette)   # the new drawing has the colors of the style
   tab.undo = reply.get("undo", false)
   tab.redo = reply.get("redo", false)
   if reply.get("history", "") != tab.get("spread_key", ""):
      _fetch_spread(tab, reply.get("history", ""))
   if _current() == tab:
      %Code.text = tab.code
      _show_data(tab)
      _edit_items(tab)
      _show_undrawn(tab)
   %Worker.request({"cmd": "attributes", "tab": tab.id}, func(r, _p):
      if r.ok and _tabs.has(tab):
         tab.attributes = r
         if _current() == tab:
            %Visual.show_style(tab.style, tab.attributes)
            _find_choices(tab))


## Visual elements panel: a new style for the current tab (or [param tab]), drawn by the worker;
## the view is fitted again when the positions change (see position_dialog.gd).
func _restyle(style: Dictionary, tab := {}) -> void:
   if tab.is_empty():
      tab = _current()
   if tab.is_empty():
      return
   var old: Dictionary = tab.style
   var moved: bool = old.get("position", {}) != style.get("position", {})   # new positions: fit the view
   tab.style = style
   %Worker.request({"cmd": "style", "tab": tab.id, "style": style}, func(reply, payload):
      if reply.ok and _tabs.has(tab):
         tab.view.show_network(reply, payload, not moved)
         if tab.showing:
            tab.view.show_states(tab.codes, tab.palette)
      elif not reply.ok:
         tab.style = old   # the worker kept it too
         if _current() == tab:
            %Visual.show_style(tab.style, tab.attributes)
         _message("Visual elements", str(reply.error)))


## Asks the worker for the stop values of a quantitative color scale of the current tab, and calls
## [param done] with them.
func _breaks(attribute: String, kind: String, method: String, k: int, log: bool, done: Callable) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   %Worker.request({"cmd": "breaks", "tab": tab.id, "kind": kind, "attribute": attribute, "method": method, "k": k, "log": log}, func(reply, _p):
      if reply.ok:
         done.call(reply.stops)
      else:
         _message("Colors", str(reply.error)))


func _on_background(color: Color) -> void:
   var tab := _current()
   if not tab.is_empty():
      tab.style.background = "#" + color.to_html(false)
      tab.view.set_background(color)


func _show_tab(index: int) -> void:
   for i in _tabs.size():
      _tabs[i].view.visible = i == index
   var tab := _current()
   %Code.text = tab.get("code", "")
   %Hint.visible = _tabs.is_empty()
   %FitButton.visible = not _tabs.is_empty()
   %Visual.visible = not tab.is_empty()
   for menu in [%Layout, %Analysis, %Edit, %Select, %Simulation]:
      for i in menu.item_count:
         menu.set_item_disabled(i, tab.is_empty())
   %File.set_item_disabled(FILE_IMAGE, tab.is_empty())
   %Generators.set_item_disabled(0, tab.is_empty())
   %Find.editable = not tab.is_empty()
   %FindBy.disabled = tab.is_empty()
   _show_player()
   if tab.is_empty():
      _show_undrawn(tab)
      %Data.show_network(-1)
      return
   _edit_items(tab)
   _idle_status(tab)
   _show_undrawn(tab)
   %Visual.show_style(tab.style, tab.attributes)
   var v = tab.view
   %Visual.show_turn(v.turn)
   %Visual.show_scale(v.spread)
   _show_data(tab)
   if %Epidemics.visible and _epidemics_tab != tab:
      %Epidemics.hide()
   if %Links.visible and _links_tab != tab:
      %Links.hide()
   if %Gnns.visible and _gnns_tab != tab:
      %Gnns.hide()
   if %Backbone.visible and _backbone_tab != tab:
      %Backbone.hide()
   _find_choices(tab)


## What depends on whether the network of the current tab [param tab] is drawn: a network that is
## not drawn has, in the visualization panel, a note and Draw anyway instead of the view, and no
## visual elements, search, Select menu, Fit or image (nothing can be selected or seen).
func _show_undrawn(tab: Dictionary) -> void:
   var off: bool = tab.get("undrawn", false)
   %Hint.visible = tab.is_empty() or off
   %Hint.text = HINT if tab.is_empty() else "%s\n\nNot drawn: too many nodes and edges.\nLayouts, analyses, simulations and edits still work, with their code." % tab.summary
   %DrawAnyway.visible = off
   if not tab.is_empty():
      tab.view.visible = not off
   %FitButton.visible = not tab.is_empty() and not off
   %Visual.visible = not tab.is_empty() and not off
   %Find.editable = not tab.is_empty() and not off
   %FindBy.disabled = tab.is_empty() or off
   %File.set_item_disabled(FILE_IMAGE, tab.is_empty() or off)
   for i in %Select.item_count:
      %Select.set_item_disabled(i, tab.is_empty() or off)


## Undo and Redo of the Edit menu, enabled when the current tab has something to undo or redo.
func _edit_items(tab: Dictionary) -> void:
   %Edit.set_item_disabled(%Edit.get_item_index(0), not tab.get("undo", false))
   %Edit.set_item_disabled(%Edit.get_item_index(1), not tab.get("redo", false))


func _close_tab(index: int) -> void:
   var tab: Dictionary = _tabs[index]
   %Worker.request({"cmd": "close", "tab": tab.id}, func(_r, _p): pass)
   tab.view.queue_free()
   _tabs.remove_at(index)
   %Tabs.remove_tab(index)
   _show_tab(%Tabs.current_tab)
