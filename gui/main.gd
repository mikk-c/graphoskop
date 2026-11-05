extends Control
## Main window: menu bar, one tab per open network, the visual elements panel on the left, the
## visualization panel on the right with the data and code panels below it, and a status bar
## (progress of long computations, and the Python environment of the worker: click it to change
## environment).

const NetworkView := preload("res://network_view.tscn")
## Ids of the items of the File menu.
const FILE_LOAD := 0
const FILE_IMAGE := 1
## Items File › Open session… and Save session… (one file with all the tabs, see
## [method _save_session]).
const FILE_OPEN_SESSION := 2
const FILE_SAVE_SESSION := 3
## Item File › Export › Network tables…: the edge and node tables of the current network (worker
## command [code]export[/code]); File › Export › Image… is [constant FILE_IMAGE].
const FILE_EXPORT := 4
## Items File › New network… (an empty network, drawn by hand), File › Close session (closes all
## the tabs), and the submenus File › Recent sessions and File › Export.
const FILE_NEW := 5
const FILE_CLOSE_SESSION := 6
const FILE_RECENT := 7
const FILE_EXPORTS := 8
## Most sessions listed by File › Recent sessions.
const MAX_RECENT := 10
## Default width of saved images, relative to the width of the visualization panel on screen.
const IMAGE_SCALE := 2
## Largest width or height of a saved image, in pixels.
const MAX_IMAGE_PIXELS := 16384
## Formats of File › Export › Image…: label, extension. Vector PDF documents are drawn by the
## worker from the drawing and the view (command [code]image_vector[/code]: circles, lines, curves,
## text), PDF images hold the PNG image at [constant PDF_DPI] dots per inch (command
## [code]image_pdf[/code]); the width is in pixels at that resolution.
const IMAGE_FORMATS := [["PNG image", "png"], ["PDF document, vector", "pdf"], ["PDF document, image", "pdf"]]
const PDF_DPI := 300
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
   ["Route edges around nodes", "route_edges", {}, false],
   ["Route edges around nodes…", "route_edges", {}, true],
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
## [method _transform]). Union, intersection and difference ask for the other network: another
## tab (worker [code]SET_OPERATIONS[/code]).
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
   ["Line graph", "line_graph", false],
   ["Complement…", "complement", true],
   ["", "", false],
   ["Union with another tab…", "network_union", false],
   ["Intersection with another tab…", "network_intersection", false],
   ["Difference with another tab…", "network_difference", false],
   ["", "", false],
   ["Clique expansion…", "clique_expansion", true],
   ["Clique promotion…", "clique_promotion", true],
   ["Hypergraph walk network…", "hypergraph_walk_network", true],
   ["", "", false],
   ["One layer…", "layer_network", true],
   ["Aggregate of the layers…", "aggregate_network", true],
]
## Items of the Help menu: label and web page (opened in the browser).
const HELP := [
   ["Online documentation", "https://graphoskop.readthedocs.io/en/latest/"],
   ["Graphical interface (online documentation)", "https://graphoskop.readthedocs.io/en/latest/gui.html"],
   ["Report a bug or request a feature", "https://github.com/mikk-c/graphoskop/issues"],
]
## Ids of the items of the menu of the search box that are not nodes.
const FOUND_ALL := -2
const FOUND_MORE := -3
## Items of the Select menu.
const SELECTS := [["Select all nodes"], ["Clear selection"], [""], ["Select by attribute…"], ["Neighbors of the selection"]]
## Item of Select › Neighbors of the selection (Ctrl+6, Command+6 on macOS).
const SELECT_NEIGHBORS := 4
## Items of Analysis › Centrality: label and function (degree also plots the degree distribution).
## The last ones are for hypergraphs and simplicial complexes: computed on the network, their
## values are drawn on the incidence network (nodes, hyperedges or both).
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
   ["", ""],
   ["Hyperedge size…", "hyperedge_size"],
   ["Local triangle closure…", "local_triangle_closure"],
   ["Two-node clustering…", "two_node_clustering"],
   ["Generalized degree (complexes)…", "generalized_degree"],
   ["Node and hyperedge centrality…", "node_edge_centrality"],
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
## function. The last two open the tree of the levels of a hierarchy of communities in a new tab,
## laid out radially (worker command [code]community_tree[/code]; the hierarchical methods offer
## it in their dialog), and its dendrogram in the plot window (command [code]dendrogram[/code]).
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
   ["", ""],
   ["Tree of a hierarchy…", "community_tree"],
   ["Dendrogram of a hierarchy…", "dendrogram"],
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
## The tab of the robustness dialog.
var _robustness_tab := {}
## The tab of the synchronization dialog.
var _synchronization_tab := {}
## The tab of the link prediction dialog, and its runs (from the worker).
var _links_tab := {}
var _link_info := {}
## The tab of the graph neural network dialog.
var _gnns_tab := {}
## Ids of Analysis › Link prediction… and Graph neural networks… (apart from the submenus).
const ANALYSIS_LINKS := 2000
const ANALYSIS_GNNS := 2001
## Id of the submenu Analysis › Frequent subgraphs, and its items: label and function (worker
## command [code]mine[/code]: the patterns open in a new tab).
const ANALYSIS_MINES := 2002
## Id of the submenu Analysis › Hierarchy, and its items: label and function (worker command
## [code]hierarchy[/code]: a score of directed networks, shown in a message).
const ANALYSIS_HIERARCHY := 2003
## Id of the submenu Analysis › Network statistics, and its items: label and function (worker
## command [code]statistics[/code]: the values become network attributes); the last computes all
## those that apply.
const ANALYSIS_STATISTICS := 2004
## Id of the submenu Analysis › Shortest paths, and its items: label and function (worker command
## [code]paths[/code]: the lengths kept for later analyses and as a table, the average distance of
## every node, and the heatmap of the lengths in the plot window).
const ANALYSIS_PATHS := 2005
const PATHS := [
   ["All pairs lengths…", "all_pairs_shortest_path_lengths"],
]
const STATISTICS := [
   ["Density…", "density"],
   ["Average degree…", "average_degree"],
   ["Reciprocity…", "reciprocity"],
   ["Global clustering coefficient…", "global_clustering"],
   ["Average clustering coefficient…", "average_clustering"],
   ["Global triangle closure…", "global_triangle_closure"],
   ["Assortativity…", "assortativity"],
   ["Average shortest path length…", "average_shortest_path_length"],
   ["Diameter…", "diameter"],
   ["Degree centralization…", "degree_centralization"],
   ["Connected components…", "connected_components_count"],
   ["Modularity of a node attribute…", "modularity"],
   ["", ""],
   ["Uniform hypergraph…", "is_uniform"],
   ["Pure complex…", "is_pure"],
   ["Pseudomanifold…", "is_pseudomanifold"],
   ["Betti numbers…", "betti_numbers"],
   ["Euler characteristic…", "euler_characteristic"],
   ["", ""],
   ["All of them", "all"],
]
## Id of the submenu Edit › Error correction, and its items: label and function (worker command
## [code]correct[/code]: the corrected network in a new tab, and the backbone dialog on the
## probabilities of its edges).
const EDIT_CORRECTION := 1005
const CORRECTIONS := [
   ["From measurements…", "measured_network_correction"],
   ["From edge probabilities…", "uncertain_network_correction"],
]
const HIERARCHIES := [
   ["Global reaching centrality…", "global_reach_centrality"],
   ["Flow hierarchy…", "flow_hierarchy"],
   ["Arborescence hierarchicalness…", "arborescence_hierarchy"],
   ["Agony…", "agony_hierarchy"],
]
const MINES := [
   ["In the whole network…", "single_graph_mining"],
   ["Across components…", "transactional_graph_mining"],
]
## Id of Edit › New node attribute… (apart from those of [constant EDITS]).
const EDIT_DERIVE := 1004
## Items of Simulation › Exploration: label and function (worker command [code]explore[/code]).
const EXPLORES := [
   ["Breadth-first search…", "bfs"],
   ["Depth-first search…", "dfs"],
   ["Random walk…", "random_walk"],
   ["Metropolis-Hastings random walk…", "metropolis_hastings_random_walk"],
]
## Items of Simulation › Synchronization (the dialog of its runs, robustness_dialog.gd): label and
## function (worker commands [code]synchronize[/code], [code]synchronization_runs[/code] and
## [code]synchronization_plot[/code]).
const SYNCHRONIZATION := [
   ["Kuramoto", "kuramoto_model"],
   ["Simplicial Kuramoto", "simplicial_kuramoto_model"],
   ["Node-link Kuramoto", "node_link_kuramoto_model"],
]
## Text of the visualization panel without networks.
const HINT := "File › Load network…, File › New network… or Generators to open a network"
## The tab of the backbone dialog.
var _backbone_tab := {}
## Menu of a column of the Nodes and Edges tables (right click on its title), the column it is for
## ([kind, attribute]), and the confirmation of the deletion of an attribute.
var _column_menu := PopupMenu.new()
var _column := ["", ""]
var _confirm_delete := ConfirmationDialog.new()
## Question asked before a session replaces the open networks, and the file it is about.
var _confirm_open := ConfirmationDialog.new()
var _session_path := ""
## Question of File › Close session.
var _confirm_close := ConfirmationDialog.new()
## Dialog of File › New network… (directed or not).
var _new_dialog := ConfirmationDialog.new()
var _new_directed := CheckBox.new()
## Dialog of File › Export › Image… (width in pixels, format), before the file dialog.
var _image_dialog := ConfirmationDialog.new()
var _image_width := SpinBox.new()
var _image_height := Label.new()
var _image_format := OptionButton.new()
## Submenus File › Recent sessions and File › Export.
var _recent_menu := PopupMenu.new()
var _export_menu := PopupMenu.new()
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
   communities.id_pressed.connect(func(id):
      if COMMUNITIES[id][1] == "community_tree":
         _ask_tree()
      else:
         _run_analysis("dendrogram" if COMMUNITIES[id][1] == "dendrogram" else "communities", "Community discovery", COMMUNITIES[id]))
   %Analysis.add_submenu_node_item("Community discovery", communities)
   var vectors := PopupMenu.new()
   _menu(vectors, VECTORS)
   vectors.id_pressed.connect(func(id): _run_analysis("vectors", "Node vectors", VECTORS[id]))
   %Analysis.add_submenu_node_item("Node vectors", vectors)
   var mines := PopupMenu.new()
   _menu(mines, MINES)
   mines.id_pressed.connect(func(id): _mine(MINES[id]))
   %Analysis.add_submenu_node_item("Frequent subgraphs", mines, ANALYSIS_MINES)
   var hierarchies := PopupMenu.new()
   _menu(hierarchies, HIERARCHIES)
   hierarchies.id_pressed.connect(func(id): _run_analysis("hierarchy", "Hierarchy", HIERARCHIES[id]))
   %Analysis.add_submenu_node_item("Hierarchy", hierarchies, ANALYSIS_HIERARCHY)
   var statistics := PopupMenu.new()
   _menu(statistics, STATISTICS)
   statistics.id_pressed.connect(func(id): _run_statistics(STATISTICS[id]))
   %Analysis.add_submenu_node_item("Network statistics", statistics, ANALYSIS_STATISTICS)
   var paths := PopupMenu.new()
   _menu(paths, PATHS)
   paths.id_pressed.connect(func(id): _run_analysis("paths", "Shortest paths", PATHS[id]))
   %Analysis.add_submenu_node_item("Shortest paths", paths, ANALYSIS_PATHS)
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
   var corrections := PopupMenu.new()
   _menu(corrections, CORRECTIONS)
   corrections.id_pressed.connect(func(id): _correct(CORRECTIONS[id]))
   %Edit.add_submenu_node_item("Bipartite projection", projections, 1001)   # ids apart from those of EDITS
   %Edit.add_submenu_node_item("Backbone", backbones, 1002)
   %Edit.add_submenu_node_item("Error correction", corrections, EDIT_CORRECTION)
   %Edit.add_submenu_node_item("Transform", transforms, 1003)
   %Edit.add_item("New node attribute…", EDIT_DERIVE)
   %Edit.add_separator()
   _menu(%Edit, EDITS)
   var command := KEY_MASK_META if OS.get_name() == "macOS" else KEY_MASK_CTRL
   for i in EDITS.size():
      if EDIT_KEYS.has(EDITS[i][1]):
         %Edit.set_item_accelerator(%Edit.get_item_index(i), EDIT_KEYS[EDITS[i][1]] | (command if EDITS[i][1] != "remove" else 0))
   %Edit.id_pressed.connect(func(id):
      if id < EDITS.size():
         _edit(EDITS[id])
      elif id == EDIT_DERIVE:
         _open_derive())
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
   %Select.set_item_accelerator(%Select.get_item_index(SELECT_NEIGHBORS), KEY_6 | command)
   %Select.id_pressed.connect(_on_select)
   %Simulation.add_item("Epidemics…", 0)
   var explorations := PopupMenu.new()
   _menu(explorations, EXPLORES)
   explorations.id_pressed.connect(_explore)
   %Simulation.add_submenu_node_item("Exploration", explorations, 1)
   %Simulation.add_item("Robustness…", 2)
   %Simulation.add_item("Synchronization…", 3)
   %Synchronization.set_functions(SYNCHRONIZATION, "Runs a Kuramoto model of coupled oscillators on the network, over a range of coupling strengths")
   %Synchronization.add_requested.connect(_add_synchronization)
   %Synchronization.plot_requested.connect(_plot_synchronization)
   %Synchronization.visibility_changed.connect(func():
      if not %Synchronization.visible:
         _detach.call_deferred(%Synchronization))
   %Simulation.id_pressed.connect(func(id):
      if id == 0:
         _open_epidemics()
      elif id == 2:
         _open_robustness()
      elif id == 3:
         _open_synchronization())
   %Robustness.add_requested.connect(_add_robustness)
   %Robustness.plot_requested.connect(_plot_robustness)
   %Robustness.visibility_changed.connect(func():
      if not %Robustness.visible:
         _detach.call_deferred(%Robustness))
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
   _params.finder = func(attribute, typed, done):
      if _current().is_empty():
         return
      %Worker.request({"cmd": "suggest", "tab": _current().id, "attribute": attribute, "text": typed}, func(r, _p):
         if r.ok:
            done.call(r.values, int(r.more)))
   _file_menu()
   %ExportDialog.file_selected.connect(_export)
   %SessionOpenDialog.file_selected.connect(_ask_open_session)
   %SessionSaveDialog.file_selected.connect(_save_session)
   add_child(_confirm_open)
   _confirm_open.ok_button_text = "Open"
   _confirm_open.confirmed.connect(func(): _open_session(_session_path))
   add_child(_confirm_close)
   _confirm_close.ok_button_text = "Close"
   _confirm_close.confirmed.connect(_close_session)
   %Data.attributes_of = func(): return _current().get("attributes", {})
   %Data.attribute_menu_requested.connect(_attribute_menu)
   %Data.value_edit_requested.connect(_edit_value)
   add_child(_column_menu)
   _column_menu.id_pressed.connect(func(id):
      if id == 2:
         _open_derive(_column[1])
      elif id == 3:
         %Data.hide_attribute(_column[0], _column[1])
      else:
         _plot_distribution() if id == 0 else _ask_delete())
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
   %Visual.tab_names = func(): return _tabs.filter(func(t): return t != _current()).map(func(t): return t.name)
   %Visual.copy_requested.connect(func(i): _copy_style(_tabs.filter(func(t): return t != _current())[i]))
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


## Pause of the main loop between frames when nothing changes on screen (low processor mode), in
## microseconds, while the application is in the background: 10 checks a second for replies of the
## worker, instead of [code]application/run/low_processor_mode_sleep_usec[/code] (60 a second).
const BACKGROUND_SLEEP_USEC := 100000


## Saves the window on closing; in the background (no window of the application focused), the
## main loop runs less often ([constant BACKGROUND_SLEEP_USEC]), to save the CPU.
func _notification(what: int) -> void:
   if what == NOTIFICATION_APPLICATION_FOCUS_OUT:
      OS.low_processor_usage_mode_sleep_usec = BACKGROUND_SLEEP_USEC
   elif what == NOTIFICATION_APPLICATION_FOCUS_IN:
      OS.low_processor_usage_mode_sleep_usec = ProjectSettings.get_setting("application/run/low_processor_mode_sleep_usec", 16667)
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
   for dialog: Window in [_params, %LoadDialog, %SelectDialog, %Derive, _plot, %Epidemics, %Robustness, %Synchronization, %Links, %Gnns, %Backbone]:
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
   elif id == FILE_OPEN_SESSION:
      %SessionOpenDialog.current_dir = AppSettings.folder("sessions", AppSettings.folder("edges"))
      %SessionOpenDialog.popup_centered_ratio(0.6)
   elif id == FILE_SAVE_SESSION and not _tabs.is_empty():
      %SessionSaveDialog.current_dir = AppSettings.folder("sessions", AppSettings.folder("edges"))
      %SessionSaveDialog.current_file = _current().name.get_basename() + ".gsk"
      %SessionSaveDialog.popup_centered_ratio(0.6)
   elif id == FILE_EXPORT and not _current().is_empty():
      %ExportDialog.current_dir = AppSettings.folder("exports", AppSettings.folder("edges"))
      %ExportDialog.current_file = _current().name.get_basename() + ".tsv"
      %ExportDialog.popup_centered_ratio(0.6)
   elif id == FILE_IMAGE and not _current().is_empty():
      var w: float = _current().view.size.x
      _image_width.max_value = MAX_IMAGE_PIXELS
      _image_width.value = roundi(w * IMAGE_SCALE)
      _image_height_text()
      _image_dialog.popup_centered()
   elif id == FILE_NEW:
      _new_dialog.popup_centered()
   elif id == FILE_CLOSE_SESSION and not _tabs.is_empty():
      _confirm_close.dialog_text = "Close the %d open networks? What is not saved in a session is lost." % _tabs.size()
      _confirm_close.popup_centered()


## The state of the interface saved in a session (worker command [code]save_session[/code], which
## adds the namespaces of the tabs: networks, tables, code, styles, undo histories): the tabs in
## order, each with its worker id, name, summary, style (with the background), whether it is drawn,
## its view (center, zoom, rotation, scale), its selected nodes and edges and its hidden
## attributes; the current tab, and the tab and table of the data panel.
func _session_state() -> Dictionary:
   var tabs := []
   for tab in _tabs:
      var v = tab.view
      tabs.append({"tab": tab.id, "name": tab.name, "summary": tab.summary, "style": tab.style, "undrawn": tab.get("undrawn", false), "view": {"center": [v.center.x, v.center.y], "zoom": v.zoom, "turn": v.turn, "spread": v.spread}, "nodes": Array(v.selected_nodes()), "edges": Array(v.selected_edges), "hidden": %Data.hidden_state(tab.id)})
   return {"tabs": tabs, "current": %Tabs.current_tab, "data": %Data.panel_state()}


## File › Save session…: all the tabs in one file ([param path], ending in .gsk), which File ›
## Open session… puts back as they are.
func _save_session(path: String) -> void:
   if _tabs.is_empty():
      return
   if path.get_extension().to_lower() != "gsk":
      path += ".gsk"
   AppSettings.store_folder("sessions", path)
   _task("Saving the session…", {"cmd": "save_session", "path": path, "gui": _session_state()}, func(r, _p):
      _idle_status(_current())
      if not r.ok:
         _message("Save session", str(r.error))
         return
      _remember_session(path)
      %Status.text = "Saved the session in %s (%d tabs, %s)" % [path.get_file(), r.tabs, String.humanize_size(int(r.size))])


## File › Open session…: asks first if networks are open (the session replaces them).
func _ask_open_session(path: String) -> void:
   _session_path = path
   AppSettings.store_folder("sessions", path)
   if _tabs.is_empty():
      _open_session(path)
      return
   _confirm_open.dialog_text = "Opening %s closes the %d open networks (save them first with File › Save session… to keep them)." % [path.get_file(), _tabs.size()]
   _confirm_open.popup_centered()


## Opens the session of [param path] (worker command [code]open_session[/code]): the open tabs
## are closed, then every tab of the session comes back with its namespace, code, drawing, style,
## view, selection and hidden attributes, and the current tab and data panel as they were.
func _open_session(path: String) -> void:
   while not _tabs.is_empty():
      _close_tab(0)
   _task("Opening the session %s…" % path.get_file(), {"cmd": "open_session", "path": path, "first": _next_tab + 1}, func(r, _p):
      if not r.ok:
         _message("Open session", str(r.error))
         return
      _remember_session(path)
      var states: Array = r.gui.tabs
      _next_tab += states.size()
      for t in states:
         var tab := {"id": int(t.tab), "name": t.name, "view": NetworkView.instantiate(), "code": "", "summary": t.summary, "style": t.style, "attributes": {}, "undrawn": t.get("undrawn", false)}
         _add_tab(tab)
         %Data.set_hidden_state(tab.id, t.get("hidden", {}))
         var view: Dictionary = t.view
         tab.view.restore_view(Vector2(view.center[0], view.center[1]), view.zoom, view.turn, view.spread)
         %Worker.request({"cmd": "redraw", "tab": tab.id}, func(reply, payload):
            if not reply.ok or not _tabs.has(tab):
               return
            _drawn(tab, reply, payload, true)
            tab.view.set_selection(PackedInt32Array(t.nodes), PackedInt32Array(t.edges)))
      var current := clampi(int(r.gui.get("current", 0)), 0, _tabs.size() - 1)
      if current >= 0:
         %Tabs.current_tab = current
         _show_tab(current)
      %Data.set_panel_state(r.gui.get("data", {}))
      %Status.text = "Opened the session %s (%d tabs)%s" % [path.get_file(), states.size(), " · " + r.warning if r.has("warning") else ""])


## File › Export › Network tables…: the network of the current tab as tab-separated tables next to
## [param path] (worker command [code]export[/code]: name_edges.tsv, name_nodes.tsv, and
## name_interlayer.tsv for multilayer networks), as [code]to_pandas_dataframes[/code] gives them;
## the code does not change.
func _export(path: String) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   AppSettings.store_folder("exports", path)
   _task("%s: exporting…" % tab.name, {"cmd": "export", "tab": tab.id, "path": path}, func(r, _p):
      _idle_status(tab)
      if not r.ok:
         _message("Export network", str(r.error))
         return
      %Status.text = "Exported %s: %s" % [tab.name, ", ".join(r.files.map(func(f): return f.get_file()))])


## Analysis › Network statistics ([param item] of [constant STATISTICS]): one statistic with its
## parameter dialog, or all those that apply at once (worker command [code]statistics[/code]); the
## values become network attributes (Network tab of the data panel) and are shown in a message.
func _run_statistics(item: Array) -> void:
   if item[1] != "all":
      _run_analysis("statistics", "Network statistics", item)
      return
   var tab := _current()
   if tab.is_empty():
      return
   _task("%s: network statistics…" % tab.name, {"cmd": "statistics", "tab": tab.id, "function": "all", "name": tab.name}, func(r, payload):
      if not _tabs.has(tab):
         return
      _idle_status(tab)
      if not r.ok:
         _message("Network statistics", str(r.error))
         return
      _drawn(tab, r, payload, true)
      _message("Network statistics", r.result))


## Edit › Error correction ([param item] of [constant CORRECTIONS]): the parameter dialog, then the
## corrected network in a new tab (worker command [code]correct[/code]: every pair with its
## probability of being an edge, summary values as network attributes), and the backbone dialog on
## that probability, at 0.5 (the most likely network): Extract opens the edges above the
## threshold in another tab.
func _correct(item: Array) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   _ask({"cmd": "signature", "function": item[1], "tab": tab.id}, func(reply, _p):
      if not reply.ok:
         _message("Error correction", str(reply.error))
         return
      _params.open("Error correction, %s (gsk.%s)" % [item[0].trim_suffix("…").to_lower(), item[1]], reply, {}, "Correct", tab.attributes, func(arguments, done):
         _next_tab += 1
         var id := _next_tab
         _task("%s: correcting the network (the first time, the kernels compile: about two minutes)…" % tab.name, {"cmd": "correct", "tab": tab.id, "new_tab": id, "function": item[1], "arguments": arguments}, func(r, payload):
            if not _tabs.has(tab):
               return
            _idle_status(tab)
            if not r.ok:
               _failed("Error correction", str(r.error), done)
               return
            done.call("")
            var parts: Array = r.parts
            var png: PackedByteArray = payload.slice(payload.size() - int(parts.back()))
            _open_derived(tab, "%s (corrected)" % tab.name, id, r, payload)   # the plot, last, is not read as a part of the drawing
            var new: Dictionary = _tabs.back()
            %Status.text = "%s  ·  %s" % [new.summary, r.note]
            _backbone_tab = new
            var b: Dictionary = r.backbone
            %Backbone.open("Edge probabilities of %s (corrected)" % tab.name, png, b.box, b.scale, b.tails, b.range, b.found, clampf(0.5, b.range[0], b.range[1]), true))))


## File › Export › Image…: the visualization panel of the current tab as a PNG image, a vector PDF
## document (drawn by the worker as the view shows it) or a PDF document holding the image at
## [constant PDF_DPI] dots per inch, as wide as chosen in the dialog (by default
## [constant IMAGE_SCALE] times its width on screen).
func _save_image(path: String) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   var ext: String = IMAGE_FORMATS[_image_format.selected][1]
   if path.get_extension().to_lower() != ext:
      path += "." + ext
   AppSettings.store_folder("images", path)
   if _image_format.selected == 1:   # vector graphics, drawn by the worker
      var v = tab.view
      var view := {"center": [v.center.x, v.center.y], "zoom": v.zoom, "turn": v.turn, "spread": v.spread, "size": [v.size.x, v.size.y]}
      _task("Saving %s…" % path.get_file(), {"cmd": "image_vector", "tab": tab.id, "path": path, "width": _image_width.value, "dpi": PDF_DPI, "view": view, "background": tab.style.get("background", "#ffffff")}, func(r, _p):
         _idle_status(tab)
         if r.ok:
            %Status.text = "Saved %s (vector, %s)" % [path.get_file(), String.humanize_size(int(r.size))]
         else:
            _message("Export image", "Could not save %s: %s" % [path, r.error]))
      return
   var image: Image = await tab.view.capture(_image_width.value / tab.view.size.x)
   if ext == "png":
      var err := image.save_png(path)
      %Status.text = "Saved %s (%d × %d)" % [path.get_file(), image.get_width(), image.get_height()] if err == OK else "Could not save %s (error %d)" % [path, err]
      return
   var png := OS.get_user_data_dir().path_join("image_for_pdf.png")   # the worker turns it into the PDF document
   image.save_png(png)
   %Worker.request({"cmd": "image_pdf", "png": png, "path": path, "dpi": PDF_DPI}, func(r, _p):
      DirAccess.remove_absolute(png)
      %Status.text = "Saved %s (%d × %d)" % [path.get_file(), image.get_width(), image.get_height()] if r.ok else "Could not save %s: %s" % [path, r.error])


## Builds the File menu: New network…; Open session…, Recent sessions, Save session…, Close
## session; Load network…, Export (Network tables…, Image…). Also the dialogs of New network…
## and Export › Image….
func _file_menu() -> void:
   %File.clear()
   %File.add_item("New network…", FILE_NEW)
   %File.add_separator("", 100)
   %File.add_item("Open session…", FILE_OPEN_SESSION)
   %File.add_submenu_node_item("Recent sessions", _recent_menu, FILE_RECENT)
   %File.add_item("Save session…", FILE_SAVE_SESSION)
   %File.add_item("Close session", FILE_CLOSE_SESSION)
   %File.add_separator("", 101)
   %File.add_item("Load network…", FILE_LOAD)
   _export_menu.add_item("Network tables…", FILE_EXPORT)
   _export_menu.add_item("Image…", FILE_IMAGE)
   %File.add_submenu_node_item("Export", _export_menu, FILE_EXPORTS)
   %File.id_pressed.connect(_on_file)
   _export_menu.id_pressed.connect(_on_file)
   %File.about_to_popup.connect(_fill_recent)
   _recent_menu.index_pressed.connect(func(i): _ask_open_session(_recent()[i]))
   add_child(_new_dialog)
   _new_dialog.title = "New network"
   _new_dialog.ok_button_text = "Create"
   var box := VBoxContainer.new()
   var hint := Label.new()
   hint.text = "An empty network: right click in the view to add nodes,\nand on a node to add an edge from it."
   box.add_child(hint)
   _new_directed.text = "Directed edges"
   box.add_child(_new_directed)
   _new_dialog.add_child(box)
   _new_dialog.confirmed.connect(_new_network)
   add_child(_image_dialog)
   _image_dialog.title = "Export image"
   _image_dialog.ok_button_text = "Save…"
   var grid := GridContainer.new()
   grid.columns = 2
   for pair in [["Width", _image_width], ["Height", _image_height], ["Format", _image_format]]:
      var label := Label.new()
      label.text = pair[0]
      grid.add_child(label)
      grid.add_child(pair[1])
   _image_width.min_value = 16
   _image_width.suffix = "pixels"
   _image_width.tooltip_text = "Width of the image; the height follows the shape of the visualization panel"
   _image_width.value_changed.connect(func(_v): _image_height_text())
   for f in IMAGE_FORMATS:
      _image_format.add_item(f[0])
   _image_format.select(0)
   _image_format.tooltip_text = "PDF vector documents: nodes, edges and labels as shapes and text (any zoom); PDF images: the picture at %d dots per inch" % PDF_DPI
   _image_dialog.add_child(grid)
   _image_dialog.confirmed.connect(func():
      var ext: String = IMAGE_FORMATS[_image_format.selected][1]
      %ImageDialog.filters = PackedStringArray(["*.%s ; %s" % [ext, IMAGE_FORMATS[_image_format.selected][0]]])
      %ImageDialog.current_dir = AppSettings.folder("images", AppSettings.folder("edges"))
      %ImageDialog.current_file = _current().name.get_basename() + "." + ext
      %ImageDialog.popup_centered_ratio(0.6))


func _image_height_text() -> void:
   var tab := _current()
   if tab.is_empty() or tab.view.size.x <= 0:
      return
   _image_height.text = "%d pixels" % roundi(_image_width.value * tab.view.size.y / tab.view.size.x)


## Sessions saved or opened recently, the latest first (files that still exist).
func _recent() -> Array:
   return Array(AppSettings.value("sessions", "recent", [])).filter(func(p): return FileAccess.file_exists(p))


## Puts [param path] first among the recent sessions.
func _remember_session(path: String) -> void:
   var recent := _recent().filter(func(p): return p != path)
   recent.push_front(path)
   AppSettings.store("sessions", "recent", recent.slice(0, MAX_RECENT))


## File › Recent sessions: the sessions saved or opened recently (click one to open it).
func _fill_recent() -> void:
   _recent_menu.clear()
   var recent := _recent()
   for p: String in recent:
      _recent_menu.add_item(p.get_file())
      _recent_menu.set_item_tooltip(_recent_menu.item_count - 1, p)
   %File.set_item_disabled(%File.get_item_index(FILE_RECENT), recent.is_empty())


## File › New network…: a new tab with an empty network (worker command [code]empty[/code]),
## drawn as an empty canvas: nodes are added with a right click in the view, edges with a right
## click on their first node (see [method _open_context]).
func _new_network() -> void:
   var taken := _tabs.map(func(t): return t.name)
   var name := "New network"
   var k := 1
   while taken.has(name):
      k += 1
      name = "New network %d" % k
   _new_tab(name, {"cmd": "empty", "directed": _new_directed.button_pressed})


## File › Close session: closes all the tabs.
func _close_session() -> void:
   while not _tabs.is_empty():
      _close_tab(0)


## Network › Copy › From tab… of the visual elements panel: the style of [param source] (colors,
## sizes, opacities, positions, labels of nodes and edges, background) and its rotation and scale
## for the current tab. Choices mapped to attributes the current network does not have go back to
## their defaults (the worker prunes them, see [code]prune_style[/code]).
func _copy_style(source: Dictionary) -> void:
   var tab := _current()
   if tab.is_empty() or not _tabs.has(source) or source == tab:
      return
   var style: Dictionary = source.style.duplicate(true)
   var background: String = style.get("background", "#ffffff")
   _restyle(style, tab)
   _on_background(Color(background))
   tab.view.set_turn(source.view.turn)
   tab.view.set_spread(source.view.spread)
   %Visual.show_style(tab.style, tab.attributes)
   %Visual.show_turn(source.view.turn)
   %Visual.show_scale(source.view.spread)


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
      if request.cmd == "empty":   # File › New network: an empty canvas
         _set_drawn(tab, true)
      elif tab.undrawn:   # not drawn by the worker: asks whether to draw it anyway
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
      if on and int(reply.nodes) == 0 and not reply.get("empty", false):
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


## Analysis › Frequent subgraphs ([param item] of [constant MINES]): the parameter dialog of the
## mining function, then the patterns in a new tab (worker command [code]mine[/code]: one
## component per pattern, node attribute [code]pattern[/code], support and association rules in
## the Tables tab), laid out by the force-directed layout (small components).
func _mine(item: Array) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   _ask({"cmd": "signature", "function": item[1], "tab": tab.id}, func(reply, _p):
      if not reply.ok:
         _message("Frequent subgraphs", str(reply.error))
         return
      _params.open("Frequent subgraphs, %s (gsk.%s)" % [item[0].trim_suffix("…").to_lower(), item[1]], reply, {}, "Mine", tab.attributes, func(arguments, done):
         _next_tab += 1
         var id := _next_tab
         _task("%s: mining frequent subgraphs…" % tab.name, {"cmd": "mine", "tab": tab.id, "new_tab": id, "function": item[1], "arguments": arguments}, func(r, payload):
            if not _tabs.has(tab):
               return
            _idle_status(tab)
            if not r.ok:
               _failed("Frequent subgraphs", str(r.error), done)
               return
            done.call("")
            var name := "%s (patterns)" % tab.name
            var new := {"id": id, "name": name, "view": NetworkView.instantiate(), "code": r.code, "summary": "%s: %s" % [name, r.summary], "style": {"background": tab.style.get("background", "#ffffff")}, "attributes": {}}
            _add_tab(new)
            _drawn(new, r, payload)
            if int(r.nodes) == 0 and not r.get("undrawn", false):
               _layout(new, "force_directed_layout", {})
            %Status.text = "%s  ·  %s" % [new.summary, r.note])))


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
func _transform(item: Array, other := {}) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   if item[1] in ["network_union", "network_intersection", "network_difference"] and other.is_empty():
      _choose_other(tab, item)
      return
   var run := func(arguments: Dictionary, done: Callable) -> void:
      _next_tab += 1
      var id := _next_tab
      var request := {"cmd": "extract", "tab": tab.id, "new_tab": id, "function": item[1], "arguments": arguments}
      if not other.is_empty():
         request.merge({"other": other.id, "other_name": other.name})
      _task("%s: running %s…" % [tab.name, item[1]], request, func(r, payload):
         if not _tabs.has(tab):
            return
         _idle_status(tab)
         if not r.ok:
            _failed("Transform", str(r.error), done)
            return
         if done.is_valid():
            done.call("")
         _open_derived(tab, "%s (%s)" % [tab.name, item[0].trim_suffix("…").replace(" with another tab", " with " + str(other.get("name", ""))).to_lower()], id, r, payload))
   if not item[2]:
      run.call({}, Callable())
      return
   _ask({"cmd": "signature", "function": item[1], "tab": tab.id}, func(reply, _p):
      if not reply.ok:
         _message("Transform", str(reply.error))
         return
      _params.open("%s (gsk.%s)" % [item[0].trim_suffix("…"), item[1]], reply, {}, "Run", tab.attributes, run))


## Community discovery › Tree of a hierarchy…: its parameter dialog (the levels, by default those
## of the last hierarchical run), then [method _community_tree].
func _ask_tree() -> void:
   var tab := _current()
   if tab.is_empty():
      return
   _ask({"cmd": "signature", "function": "community_tree", "tab": tab.id}, func(reply, _p):
      if not reply.ok:
         _message("Community discovery", str(reply.error))
         return
      _params.open("Tree of a hierarchy (gsk.community_tree)", reply, {}, "Open", tab.attributes, func(arguments, done): _community_tree(tab, arguments, done)))


## The tree of the levels of communities of [param tab] in a new tab, laid out radially (worker
## command [code]community_tree[/code], with the [param arguments] of its dialog).
func _community_tree(tab: Dictionary, arguments: Dictionary, done := Callable()) -> void:
   _next_tab += 1
   var id := _next_tab
   _task("%s: tree of the hierarchy…" % tab.name, {"cmd": "community_tree", "tab": tab.id, "new_tab": id, "arguments": arguments}, func(r, payload):
      if not _tabs.has(tab):
         return
      _idle_status(tab)
      if not r.ok:
         _failed("Community discovery", str(r.error), done)
         return
      if done.is_valid():
         done.call("")
      _open_derived(tab, "%s (tree of the hierarchy)" % tab.name, id, r, payload))


## The other network of a union, intersection or difference ([param item] of [constant TRANSFORMS]
## on [param tab]): a dialog with a menu of the other tabs.
func _choose_other(tab: Dictionary, item: Array) -> void:
   var others := _tabs.filter(func(t): return t != tab)
   if others.is_empty():
      _message("Transform", "Open the other network in a tab first: %s combines the network with the network of another tab." % item[0].trim_suffix("…").to_lower())
      return
   var dialog := ConfirmationDialog.new()
   dialog.title = item[0].trim_suffix("…")
   dialog.ok_button_text = "Run"
   var box := VBoxContainer.new()
   var label := Label.new()
   label.text = "%s of %s and the network of the tab:" % [item[0].split(" ")[0], tab.name]
   var menu := OptionButton.new()
   for t in others:
      menu.add_item(t.name)
   box.add_child(label)
   box.add_child(menu)
   dialog.add_child(box)
   dialog.confirmed.connect(func(): _transform(item, others[menu.selected]))
   dialog.visibility_changed.connect(func(): if not dialog.visible: dialog.queue_free())
   add_child(dialog)
   dialog.popup_centered()


## Backbone dialog: the plot again, with or without the [param components] and the [param left]
## and [param right] tails, with the [param colors] and [param options] of the dialog (a range of
## the threshold axis tests the thresholds again in it); what goes wrong is shown in the dialog.
func _backbone_replot(components: bool, left: bool, right: bool, colors: Dictionary, options: Dictionary) -> void:
   var tab := _backbone_tab
   if _tabs.has(tab):
      %Worker.request({"cmd": "backbone_plot", "tab": tab.id, "components": components, "left": left, "right": right, "colors": colors, "options": options}, func(reply, payload):
         %Backbone.show_note("" if reply.ok else str(reply.error))
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
   elif id == SELECT_NEIGHBORS:
      _grow_selection(tab)
   else:
      %SelectDialog.open(tab.attributes, func(kind, conditions, add, done):
         %Worker.request({"cmd": "select", "tab": tab.id, "kind": kind, "conditions": conditions}, func(reply, _p):
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


## Simulation › Robustness: opens the dialog of the robustness runs of the current tab
## (robustness_dialog.gd; worker command [code]robustness_runs[/code]).
func _open_robustness() -> void:
   var tab := _current()
   if tab.is_empty():
      return
   _ask({"cmd": "robustness_runs", "tab": tab.id}, func(reply, _p):
      if reply.ok and _tabs.has(tab):
         _robustness_tab = tab
         %Robustness.open("Robustness of %s" % tab.name, reply))


## Add run or Copy… of the robustness dialog: the parameter dialog of [param function], with the
## [param values] of the run copied (its name left empty: the function and the strategy, see
## [code]robustness_code[/code] in the worker); the run is added to the dialog (worker command
## [code]robustness[/code]), its global value in the status bar.
func _add_robustness(function: String, values: Dictionary) -> void:
   var tab := _robustness_tab
   if not _tabs.has(tab):
      return
   _ask({"cmd": "signature", "function": function, "tab": tab.id}, func(reply, _p):
      if not reply.ok:
         %Robustness.show_error(str(reply.error))
         return
      _above(_params, %Robustness)
      _params.open("%s (gsk.%s)" % [function.capitalize(), function], reply, values, "Run", tab.attributes, func(arguments, done):
         %Robustness.show_status("Running %s…" % function)
         _task("%s: running %s…" % [tab.name, function], {"cmd": "robustness", "tab": tab.id, "function": function, "arguments": arguments, "name": tab.name}, func(r, payload):
            if not _tabs.has(tab):
               return
            _idle_status(tab)
            %Robustness.show_status("")
            if not r.ok:
               _failed("Robustness", str(r.error), done)
               return
            done.call("")
            _drawn(tab, r, payload, true)
            if _robustness_tab == tab:
               %Robustness.show_runs(r)
               %Robustness.show_status(r.result)
            if _current() == tab:
               %Status.text = "%s  ·  %s" % [tab.summary, r.result])))


## Plot of the robustness dialog: the checked runs [param names] in the plot window (worker command
## [code]robustness_plot[/code]); runs that cannot share axes are refused in the dialog.
func _plot_robustness(names: Array) -> void:
   var tab := _robustness_tab
   if not _tabs.has(tab):
      return
   %Worker.request({"cmd": "robustness_plot", "tab": tab.id, "names": names}, func(reply, payload):
      if not reply.ok:
         %Robustness.show_error(str(reply.error))
         return
      %Robustness.show_error("")
      _above(_plot, %Robustness)
      _plot.show_plot(tab.id, "Robustness of %s" % tab.name, payload, reply.elements, reply.options, reply.styles, reply.text, reply.warnings, "robustness"))


## Simulation › Synchronization: opens the dialog of the synchronization runs of the current tab
## (robustness_dialog.gd with the functions of [constant SYNCHRONIZATION]; worker command
## [code]synchronization_runs[/code]).
func _open_synchronization() -> void:
   var tab := _current()
   if tab.is_empty():
      return
   _ask({"cmd": "synchronization_runs", "tab": tab.id}, func(reply, _p):
      if reply.ok and _tabs.has(tab):
         _synchronization_tab = tab
         %Synchronization.open("Synchronization of %s" % tab.name, reply))


## Add run or Copy… of the synchronization dialog: the parameter dialog of the model
## [param function], with the [param values] of the run copied and a free name ("Kuramoto 2");
## the run is added to the dialog (worker command [code]synchronize[/code]).
func _add_synchronization(function: String, values: Dictionary) -> void:
   var tab := _synchronization_tab
   if not _tabs.has(tab):
      return
   var label: String = SYNCHRONIZATION.filter(func(x): return x[1] == function)[0][0]
   _ask({"cmd": "signature", "function": function, "tab": tab.id}, func(reply, _p):
      if not reply.ok:
         %Synchronization.show_error(str(reply.error))
         return
      _above(_params, %Synchronization)
      var presets := values.duplicate()
      presets.name = JSON.stringify(%Synchronization.free_name(label))
      _params.open("%s (gsk.%s)" % [label, function], reply, presets, "Run", tab.attributes, func(arguments, done):
         %Synchronization.show_status("Running %s…" % function)
         _task("%s: running %s…" % [tab.name, function], {"cmd": "synchronize", "tab": tab.id, "function": function, "arguments": arguments}, func(r, payload):
            if not _tabs.has(tab):
               return
            _idle_status(tab)
            %Synchronization.show_status("")
            if not r.ok:
               _failed("Synchronization", str(r.error), done)
               return
            done.call("")
            _drawn(tab, r, payload, true)
            if _synchronization_tab == tab:
               %Synchronization.show_runs(r)
               %Synchronization.show_status(r.result))))


## Plot of the synchronization dialog: the transition curves of the checked runs [param names] in
## the plot window (worker command [code]synchronization_plot[/code]).
func _plot_synchronization(names: Array) -> void:
   var tab := _synchronization_tab
   if not _tabs.has(tab):
      return
   %Worker.request({"cmd": "synchronization_plot", "tab": tab.id, "names": names}, func(reply, payload):
      if not reply.ok:
         %Synchronization.show_error(str(reply.error))
         return
      %Synchronization.show_error("")
      _above(_plot, %Synchronization)
      _plot.show_plot(tab.id, "Synchronization of %s" % tab.name, payload, reply.elements, reply.options, reply.styles, reply.text, reply.warnings, "synchronization"))


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
            if r.has("tree"):   # the tree of the levels found, in a new tab
               _community_tree(tab, {"levels": JSON.stringify(r.tree)})
            if r.has("note") and _current() == tab:   # e.g. the correlation of a core-periphery model
               %Status.text = "%s  ·  %s" % [tab.summary, r.note]
            if r.has("result"):   # the value of a node vector analysis
               if _current() == tab:
                  %Status.text = "%s  ·  %s" % [tab.summary, r.result]
               if not r.has("plot"):   # with a plot, the text is under the plot
                  _message(heading, r.result)
            if r.has("plot"):
               var parts: Array = r.parts
               var png: PackedByteArray = payload.slice(payload.size() - int(parts.back()))
               if _plot.get_parent() != self:   # out of the epidemics dialog
                  _plot.hide()
                  _plot.reparent(self, false)
               var file: String = item[1] if cmd != "centrality" else "degree_distribution"
               _plot.show_plot(tab.id, r.plot.get("title", "Degree distribution of %s" % tab.name), png, r.plot.elements, r.plot.options, r.plot.styles, r.plot.text, r.plot.warnings, file, r.plot.get("kinds", []), r.plot.get("kind_label", "Plot"), r.plot.get("table", false)))))


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
   if kind == "nodes":
      _column_menu.add_item("New node attribute from %s…" % attribute, 2)
   _column_menu.add_item("Hide %s" % attribute, 3)
   _column_menu.popup(Rect2i(get_window().position + Vector2i(get_window().get_mouse_position()), Vector2i.ZERO))


## Select › Neighbors of the selection (Ctrl+6): the selected nodes, the endpoints of the selected
## edges and all their neighbors become the selection (worker command [code]neighbors[/code]);
## pressed again, it grows one more step. Edges stay selected.
func _grow_selection(tab: Dictionary) -> void:
   var v = tab.view
   if v.selected_nodes().is_empty() and v.selected_edges.is_empty():
      %Status.text = "%s  ·  select a node or an edge first: Ctrl+6 adds its neighbors" % tab.summary
      return
   %Worker.request({"cmd": "neighbors", "tab": tab.id, "nodes": Array(v.selected_nodes()), "edges": Array(v.selected_edges)}, func(reply, _p):
      if not _tabs.has(tab) or not reply.ok:
         return
      v.set_selection(PackedInt32Array(reply.nodes), v.selected_edges)
      if _current() == tab:
         %Status.text = "%s  ·  %d neighbors added" % [tab.summary, reply.added] if reply.added > 0 else "%s  ·  no neighbors to add: the whole component is selected" % tab.summary)


## Edit › New node attribute… (and the column menu, from [param from]): the dialog of a node
## attribute computed from a formula (derive_dialog.gd; worker commands [code]formulas[/code] and
## [code]derive[/code], with [code]preview[/code] while typing). The drawing follows (the style
## may use the attribute).
func _open_derive(from := "") -> void:
   var tab := _current()
   if tab.is_empty():
      return
   _ask({"cmd": "formulas", "tab": tab.id}, func(reply, _p):
      if not reply.ok:
         _message("New node attribute", str(reply.error))
         return
      var start := ""
      for a in reply.attributes:
         if a.name == from:
            start = a.formula
      %Derive.open(reply.attributes, reply.functions, "", start, func(name, formula, done):
         %Worker.request({"cmd": "derive", "tab": tab.id, "name": name, "formula": formula, "preview": true}, func(r, _q):
            done.call(r.get("text", ""), "" if r.ok else str(r.error))),
         func(name, formula, done):
            _task("%s: new node attribute %s…" % [tab.name, name], {"cmd": "derive", "tab": tab.id, "name": name, "formula": formula}, func(r, payload):
               if not _tabs.has(tab):
                  return
               _idle_status(tab)
               if not r.ok:
                  _failed("New node attribute", str(r.error), done)
                  return
               done.call("")
               _drawn(tab, r, payload, true)
               if _current() == tab:
                  %Status.text = "%s  ·  %s" % [tab.summary, r.note])))


## Simulation › Exploration ([param id] in [constant EXPLORES]): the parameter dialog of the
## search or walk, starting from the selected node if there is one, then the worker command
## [code]explore[/code] (a node attribute with the order or mean step of the visits; its note in
## the status bar).
func _explore(id: int) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   var presets := {}
   var v = tab.view
   if v.selected_nodes().size() == 1:
      presets["source"] = JSON.stringify(v.ids[v.selected_nodes()[0]])
   _run_analysis("explore", "Exploration", [EXPLORES[id][0], EXPLORES[id][1], presets])


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


## Saves the plot of the plot window (PNG or PDF, by the extension of [param path]; a table for
## .tsv); the code that writes it is added to the code of its tab.
func _save_plot(tab_id: int, path: String) -> void:
   %Worker.request({"cmd": "save_plot", "tab": tab_id, "path": path}, func(reply, _p):
      if reply.ok:
         %Status.text = "Saved %s" % path.get_file()
         for tab in _tabs:   # the code that writes the plot
            if tab.id == tab_id and reply.has("code"):
               tab.code = reply.code
               if _current() == tab:
                  %Code.text = tab.code
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
   if int(reply.nodes) > 0 or reply.get("empty", false) and reply.parts.size() >= 11:   # an empty network is an empty canvas
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
   %File.set_item_disabled(%File.get_item_index(FILE_SAVE_SESSION), _tabs.is_empty())
   %File.set_item_disabled(%File.get_item_index(FILE_CLOSE_SESSION), _tabs.is_empty())
   %File.set_item_disabled(%File.get_item_index(FILE_EXPORTS), tab.is_empty())
   _export_menu.set_item_disabled(_export_menu.get_item_index(FILE_IMAGE), tab.is_empty() or tab.get("undrawn", false))
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
   if %Robustness.visible and _robustness_tab != tab:
      %Robustness.hide()
   if %Synchronization.visible and _synchronization_tab != tab:
      %Synchronization.hide()
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
   _export_menu.set_item_disabled(_export_menu.get_item_index(FILE_IMAGE), tab.is_empty() or off)
   for i in %Select.item_count:
      %Select.set_item_disabled(i, tab.is_empty() or off)


## Undo and Redo of the Edit menu, enabled when the current tab has something to undo or redo.
func _edit_items(tab: Dictionary) -> void:
   %Edit.set_item_disabled(%Edit.get_item_index(0), not tab.get("undo", false))
   %Edit.set_item_disabled(%Edit.get_item_index(1), not tab.get("redo", false))


func _close_tab(index: int) -> void:
   var tab: Dictionary = _tabs[index]
   %Worker.request({"cmd": "close", "tab": tab.id}, func(_r, _p): pass)
   %Data.forget(tab.id)
   tab.view.queue_free()
   _tabs.remove_at(index)
   %Tabs.remove_tab(index)
   _show_tab(%Tabs.current_tab)
