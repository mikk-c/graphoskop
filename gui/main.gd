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
## (Python source code), and whether a parameter dialog opens (layouts, rotation by any angle,
## edge bundling) or the function runs at once (other transformations); an empty label is a
## separator.
const LAYOUTS := [
   ["Circular…", "circular_layout", {}, true],
   ["Force-directed…", "force_directed_layout", {}, true],
   ["Grid…", "grid_layout", {}, true],
   ["Hierarchical…", "hierarchical_layout", {}, true],
   ["Hierarchical, radial…", "hierarchical_layout", {"direction": "\"radial\""}, true],
   ["UMAP…", "umap_layout", {}, true],
   ["", "", {}, false],
   ["Mirror left to right", "mirror_layout", {}, false],
   ["Mirror top to bottom", "mirror_layout", {"axis": "\"y\""}, false],
   ["Rotate 90° counterclockwise", "rotate_layout", {"degrees": "90"}, false],
   ["Rotate 90° clockwise", "rotate_layout", {"degrees": "-90"}, false],
   ["Rotate…", "rotate_layout", {}, true],
   ["Remove overlaps", "remove_overlaps", {}, false],
   ["", "", {}, false],
   ["Bundle edges…", "bundle_edges", {}, true],
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
var _next_tab := 0
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
   var communities := PopupMenu.new()
   _menu(communities, COMMUNITIES)
   communities.id_pressed.connect(_run_communities)
   %Analysis.add_submenu_node_item("Community discovery", communities)
   _menu(%Generators, GENERATORS)
   %Generators.id_pressed.connect(_open_generator)
   %Tabs.tab_changed.connect(_show_tab)
   %FitButton.pressed.connect(_fit_view)
   %Tabs.tab_close_pressed.connect(_close_tab)
   %LoadDialog.worker = %Worker
   %LoadDialog.import_requested.connect(func(spec, done): _new_tab(spec.edges.get_file(), {"cmd": "load", "spec": spec}, done))
   %Data.worker = %Worker
   %Visual.style_changed.connect(_restyle)
   %Visual.breaks = _breaks
   %Visual.background_changed.connect(_on_background)
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
         menu.add_separator()
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
   %Status.text = "Starting Python…"
   %EnvButton.text = "%s (starting)" % Environments.name_of(_python())
   %EnvButton.tooltip_text = _python()
   %Worker.start(_python())


func _on_started(info: Dictionary) -> void:
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
   for tab in _tabs:
      %Status.text = "Rebuilding the open networks…"
      %Worker.request({"cmd": "replay", "tab": tab.id, "code": tab.code, "style": tab.style}, func(reply, payload):
         if not _tabs.has(tab):
            return
         if not reply.ok:
            _message("Python environment", "%s could not be rebuilt in this environment, and was closed:\n%s" % [tab.name, reply.error])
            _close_tab(_tabs.find(tab))
            return
         _drawn(tab, reply, payload)
         %Status.text = "Ready")


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
   %Worker.request({"cmd": "signature", "function": item[1]}, func(reply, _payload):
      if reply.ok:
         %ParamsDialog.open("%s (gsk.%s)" % [item[0], item[1]], reply, item[2], "Generate", {}, func(arguments, done): _new_tab(item[0], {"cmd": "generate", "function": item[1], "arguments": arguments}, done))
      else:
         _message("Generators", str(reply.error)))


## Creates the network of a new tab with [param request] ("load" or "generate"), then lays it out;
## [param done] (from a parameter dialog) gets the error, or an empty text when the network is
## made (without it, errors show in a message).
func _new_tab(name: String, request: Dictionary, done := Callable()) -> void:
   _next_tab += 1
   var id := _next_tab
   request.tab = id
   %Status.text = "Creating %s…" % name
   %Worker.request(request, func(reply, _payload):
      _progress_done()
      if not reply.ok:
         %Status.text = "Could not create %s" % name
         _failed(name, str(reply.error), done)
         return
      if done.is_valid():
         done.call("")
      var tab := {"id": id, "name": name, "view": NetworkView.instantiate(), "code": reply.code, "summary": "%s: %s" % [name, reply.summary], "style": {}, "attributes": {}}
      %Views.add_child(tab.view)
      %Views.move_child(%FitButton, -1)
      tab.view.node_selected.connect(func(index, label): if _current() == tab: %Data.show_network(tab.id, index, label))
      _tabs.append(tab)
      %Tabs.add_tab(name)
      %Tabs.current_tab = _tabs.size() - 1
      _show_tab(_tabs.size() - 1)
      _layout(tab, "grid_layout", {}))


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
   %Worker.request({"cmd": "signature", "function": item[1], "tab": tab.id}, func(reply, _payload):
      if reply.ok:
         %ParamsDialog.open("%s (gsk.%s)" % [item[0].trim_suffix("…"), item[1]], reply, item[2], "Run", tab.attributes, func(arguments, done): _layout(tab, item[1], arguments, done))
      else:
         _message("Layout", str(reply.error)))


## Analysis › Community discovery: opens the parameter dialog of a community discovery function,
## then runs it on the current network; the worker colors the nodes (the edges, for link
## communities) by the communities found.
func _run_communities(id: int) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   var item: Array = COMMUNITIES[id]
   %Worker.request({"cmd": "signature", "function": item[1], "tab": tab.id}, func(reply, _payload):
      if not reply.ok:
         _message("Community discovery", str(reply.error))
         return
      %ParamsDialog.open("%s (gsk.%s)" % [item[0].trim_suffix("…"), item[1]], reply, {}, "Run", tab.attributes, func(arguments, done):
         %Status.text = "%s: running %s…" % [tab.name, item[1]]
         %Worker.request({"cmd": "communities", "tab": tab.id, "function": item[1], "arguments": arguments}, func(r, payload):
            _progress_done()
            if not _tabs.has(tab):
               return
            %Status.text = tab.summary
            if not r.ok:
               _failed("Community discovery", str(r.error), done)
               return
            done.call("")
            _drawn(tab, r, payload, true))))


## Runs a layout function of the library on the network of a tab, then draws it; [param done]
## (from a parameter dialog) gets the error, or an empty text when the layout ran.
func _layout(tab: Dictionary, function: String, arguments: Dictionary, done := Callable()) -> void:
   %Status.text = "%s: running %s…" % [tab.name, function]
   %Worker.request({"cmd": "layout", "tab": tab.id, "function": function, "arguments": arguments}, func(reply, payload):
      _progress_done()
      if not _tabs.has(tab):
         return
      %Status.text = tab.summary
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
   if int(reply.nodes) > 0:
      var background: String = tab.style.get("background", "#ffffff")
      tab.style = reply.style
      tab.style.background = background
      tab.view.show_network(reply, payload, keep_view)
      tab.view.set_background(Color(background))
   if _current() == tab:
      %Code.text = tab.code
      %Data.show_network(tab.id, tab.view.selected, tab.view.ids[tab.view.selected] if tab.view.selected >= 0 else "")
   %Worker.request({"cmd": "attributes", "tab": tab.id}, func(r, _p):
      if r.ok and _tabs.has(tab):
         tab.attributes = r
         if _current() == tab:
            %Visual.show_style(tab.style, tab.attributes))


## Visual elements panel: new colors for the current tab, drawn by the worker.
func _restyle(style: Dictionary) -> void:
   var tab := _current()
   if tab.is_empty():
      return
   tab.style = style
   %Worker.request({"cmd": "style", "tab": tab.id, "style": style}, func(reply, payload):
      if reply.ok and _tabs.has(tab):
         tab.view.show_network(reply, payload, true)
      elif not reply.ok:
         _message("Colors", str(reply.error)))


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
   for menu in [%Layout, %Analysis]:
      for i in menu.item_count:
         menu.set_item_disabled(i, tab.is_empty())
   %File.set_item_disabled(FILE_IMAGE, tab.is_empty())
   if tab.is_empty():
      %Data.show_network(-1)
      return
   %Status.text = tab.summary
   %Visual.show_style(tab.style, tab.attributes)
   var v = tab.view
   %Data.show_network(tab.id, v.selected, v.ids[v.selected] if v.selected >= 0 else "")


func _close_tab(index: int) -> void:
   var tab: Dictionary = _tabs[index]
   %Worker.request({"cmd": "close", "tab": tab.id}, func(_r, _p): pass)
   tab.view.queue_free()
   _tabs.remove_at(index)
   %Tabs.remove_tab(index)
   _show_tab(%Tabs.current_tab)
