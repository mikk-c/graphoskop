extends ConfirmationDialog
## Load dialog: decides how [code]gsk.load_network[/code] (or [code]load_hypergraph[/code],
## [code]load_simplicial_complex[/code], or [code]load_matrix[/code] for an adjacency matrix) is
## called. The previews are read by the Python worker, as the library will read the files. With
## Adjacency matrix, the first file is the matrix: Header means ids in its first row and column,
## Directed and Bipartite unchecked are found from the matrix, and the node table has the node id
## in its first column.

## Emitted on Import, with the description of the call read by [code]load_code[/code] of the
## worker, and a function to call with the error, or an empty text once the network is loaded:
## the dialog stays open (with its choices) until then, and shows the error if there is one.
signal import_requested(spec: Dictionary, done: Callable)

## Separators of the "Separator" menu, in its order.
const SEPARATORS := ["\t", ",", ";", "|", "\\s+"]

## The worker (set by the main scene).
var worker: Node
var _edge_columns := PackedStringArray()
var _node_columns := PackedStringArray()
var _run := 0


## Width of the menus of columns, in pixels (long column names are cut).
const COLUMN_MENU_WIDTH := 150
## Smallest width of a column of the previews, in pixels (longer texts are cut; the table scrolls).
const PREVIEW_COLUMN_WIDTH := 80


func _ready() -> void:
   canceled.connect(func(): _run += 1; get_ok_button().disabled = false)
   %EdgeBrowse.pressed.connect(_browse_edges)
   %NodeBrowse.pressed.connect(_browse_nodes)
   %EdgeFile.file_selected.connect(func(path): AppSettings.store_folder("edges", path); %EdgePath.text = path; _preview_edges())
   %NodeFile.file_selected.connect(func(path): AppSettings.store_folder("nodes", path); %NodePath.text = path; _preview_nodes())
   %Sep.select(AppSettings.value("load", "separator", 0))
   %Header.button_pressed = AppSettings.value("load", "header", true)
   %EdgePath.text_submitted.connect(func(_t): _preview_edges())
   %NodePath.text_submitted.connect(func(_t): _preview_nodes())
   %NodeClear.pressed.connect(func(): %NodePath.text = ""; _preview_nodes())
   %Sep.item_selected.connect(func(_i): _preview_edges(); _preview_nodes())
   %Header.toggled.connect(func(_b): _preview_edges(); _preview_nodes())
   %Multilayer.toggled.connect(func(_b): _node_defaults(); _update())
   for box in [%Matrix, %Directed, %Bipartite, %Signed, %Hypergraph, %Simplicial, %Long]:
      box.toggled.connect(func(_b): _update())
   confirmed.connect(_on_import)
   for ob in $Box/EdgeColumns.get_children() + %NodeColumns.get_children():   # long column names must not widen the dialog
      if ob is OptionButton:
         ob.fit_to_longest_item = false
         ob.clip_text = true
         ob.custom_minimum_size.x = COLUMN_MENU_WIDTH
   _update()


## Opens the dialog, with the device chosen last time; the GPU is offered only if the worker found
## CUDA.
func open() -> void:
   _run += 1
   get_ok_button().disabled = false
   %Error.visible = false
   %Device.set_item_disabled(1, not worker.info.get("cuda", false))
   %Device.select(AppSettings.value("load", "device", 0))
   if %Device.is_item_disabled(%Device.selected):
      %Device.select(0)
   size = Vector2i.ZERO
   reset_size()
   popup_centered()


## Edge table: opens in the last folder of an edge table.
func _browse_edges() -> void:
   %EdgeFile.current_dir = AppSettings.folder("edges")
   %EdgeFile.popup_centered_ratio(0.6)


## Node table: opens in the folder of the chosen edge table (most likely to hold it), or else in
## the last folder of a node table.
func _browse_nodes() -> void:
   var edges: String = %EdgePath.text.strip_edges()
   var here := edges.get_base_dir() if not edges.is_empty() and DirAccess.dir_exists_absolute(edges.get_base_dir()) else ""
   %NodeFile.current_dir = here if not here.is_empty() else AppSettings.folder("nodes", AppSettings.folder("edges"))
   %NodeFile.popup_centered_ratio(0.6)


func _preview_edges() -> void:
   _edge_columns = PackedStringArray()
   _preview(%EdgePath.text, %EdgePreview, func(columns, long_guess):
      _edge_columns = columns
      for ob in [%Source, %Target, %Layer, %Direction, %Sign]:
         _options(ob, columns)
      %Target.select(mini(1, columns.size() - 1))
      _select_named(%Layer, "layer", 2)
      _select_named(%Direction, "direction", 2)
      _select_named(%Sign, "sign", 2))


func _preview_nodes() -> void:
   _node_columns = PackedStringArray()
   _preview(%NodePath.text, %NodePreview, func(columns, long_guess):
      _node_columns = columns
      for ob in [%NodeId, %NodeLayer, %AttrName, %AttrValue]:
         _options(ob, columns)
      %Long.set_pressed_no_signal(long_guess)
      _node_defaults())


## Asks the worker for the first rows of [param path], shows them in [param tree], then calls
## [param done] with the column names and the format guess.
func _preview(path: String, tree: Tree, done: Callable) -> void:
   tree.clear()
   tree.columns = 1
   _update()
   if path.strip_edges().is_empty():
      return
   var request := {"cmd": "preview", "path": path, "sep": SEPARATORS[%Sep.selected], "header": %Header.button_pressed}
   worker.request(request, func(reply, _payload):
      tree.clear()
      if not reply.ok:
         tree.columns = 1
         tree.set_column_title(0, "Cannot read the file")
         tree.create_item().set_text(0, str(reply.error))
         _update()
         return
      var columns := PackedStringArray(reply.columns)
      tree.columns = columns.size()
      tree.add_theme_constant_override("item_margin", 0)
      for j in columns.size():
         tree.set_column_title(j, columns[j] if %Header.button_pressed else "column %d" % j)
         tree.set_column_title_alignment(j, HORIZONTAL_ALIGNMENT_LEFT)
         tree.set_column_expand(j, true)
         tree.set_column_clip_content(j, true)
         tree.set_column_custom_minimum_width(j, PREVIEW_COLUMN_WIDTH)
      var root := tree.create_item()
      for row in reply.rows:
         var item := tree.create_item(root)
         for j in columns.size():
            item.set_text(j, str(row[j]))
      done.call(columns, reply.long)
      _update())


func _options(ob: OptionButton, columns: PackedStringArray) -> void:
   ob.clear()
   for j in columns.size():
      ob.add_item(columns[j] if %Header.button_pressed else "column %d" % j)


## Selects the column called [param name], or else the one at [param fallback] (if any).
func _select_named(ob: OptionButton, name: String, fallback: int) -> void:
   var j := _edge_columns.find(name) if %Header.button_pressed else -1
   if j < 0:
      j = mini(fallback, ob.item_count - 1)
   ob.select(j)


## Default key columns of the node table: id (or entity and layer), then attribute name and value.
func _node_defaults() -> void:
   var k := 2 if %Multilayer.button_pressed else 1
   var last := _node_columns.size() - 1
   %NodeId.select(mini(0, last))
   %NodeLayer.select(mini(1, last))
   %AttrName.select(mini(k, last))
   %AttrValue.select(mini(k + 1, last))


## Enables, disables and shows the controls that apply to the current choices.
func _update() -> void:
   var hyper: bool = %Hypergraph.button_pressed
   var simplicial: bool = %Simplicial.button_pressed
   var higher := hyper or simplicial
   var matrix: bool = %Matrix.button_pressed
   _allow(%Matrix, not higher and not %Multilayer.button_pressed and not %Signed.button_pressed)
   _allow(%Simplicial, not hyper and not matrix)
   _allow(%Hypergraph, not simplicial and not matrix)
   _allow(%Directed, not simplicial)
   _allow(%Bipartite, not higher)
   _allow(%Signed, not higher and not %Multilayer.button_pressed and not matrix)
   _allow(%Multilayer, not higher and not %Signed.button_pressed and not matrix)
   _allow(%Long, not matrix)
   %EdgeLabel.text = "Matrix" if matrix else "Edge table"
   %EdgeTitle.text = "%s: first 10 rows" % %EdgeLabel.text
   %EdgeColumns.visible = not matrix
   %SourceLabel.text = "Node" if higher else "Source"
   %TargetLabel.text = "Hyperedge" if hyper else ("Simplex" if simplicial else "Target")
   var edges := not _edge_columns.is_empty()
   %Layer.visible = edges and %Multilayer.button_pressed
   %LayerLabel.visible = %Layer.visible
   %Direction.visible = edges and hyper and %Directed.button_pressed
   %DirectionLabel.visible = %Direction.visible
   %Sign.visible = edges and %Signed.button_pressed
   %SignLabel.visible = %Sign.visible
   var nodes := not _node_columns.is_empty()
   var long: bool = %Long.button_pressed
   %NodeColumns.visible = nodes and not matrix
   %NodeId.visible = nodes and not higher
   %NodeIdLabel.visible = %NodeId.visible
   %NodeIdLabel.text = "Entity" if %Multilayer.button_pressed else "Node id"
   %NodeLayer.visible = nodes and %Multilayer.button_pressed
   %NodeLayerLabel.visible = %NodeLayer.visible
   %AttrName.visible = nodes and long and not higher
   %NameLabel.visible = %AttrName.visible
   %AttrValue.visible = %AttrName.visible
   %ValueLabel.visible = %AttrName.visible
   var notes := PackedStringArray()
   if matrix:
      notes.append("One row and one column per node (rows of one type and columns of the other for bipartite networks); non-zero entries are edges, their values the weights unless all are 1. Unchecked, Directed and Bipartite are found from the matrix (not symmetric: directed; not square: bipartite). With Header, the ids are in the first row and column. The node table has the node id in its first column.")
   elif %Bipartite.button_pressed:
      notes.append("Bipartite networks need a node table giving every node its type in the column \"bipartite_type\".")
   if higher:
      notes.append("Each row of the edge table is a node in a %s. The node table has the node id in its first column (then the attribute name and value, in long format)." % ("hyperedge" if hyper else "simplex"))
   if %Signed.button_pressed:
      notes.append("Edges with a sign of 0 or more are positive, the others negative.")
   %Note.text = "\n".join(notes)
   %Note.visible = not notes.is_empty()
   get_ok_button().disabled = not edges


func _allow(box: CheckBox, allowed: bool) -> void:
   box.disabled = not allowed
   if not allowed:
      box.set_pressed_no_signal(false)


## Value of the column chosen in [param ob]: its name with a header, its position without.
func _column(ob: OptionButton, columns: PackedStringArray) -> Variant:
   return columns[ob.selected] if %Header.button_pressed else ob.selected


func _on_import() -> void:
   var hyper: bool = %Hypergraph.button_pressed
   var kind := "matrix" if %Matrix.button_pressed else ("hypergraph" if hyper else ("simplicial" if %Simplicial.button_pressed else "network"))
   var spec := {
      "edges": %EdgePath.text,
      "nodes": null if %NodePath.text.strip_edges().is_empty() else %NodePath.text,
      "sep": SEPARATORS[%Sep.selected],
      "header": %Header.button_pressed,
      "kind": kind,
      "source": null if %Source.selected == 0 else _column(%Source, _edge_columns),
      "target": null if %Target.selected == 1 else _column(%Target, _edge_columns),
      "directed": %Directed.button_pressed,
      "bipartite": %Bipartite.button_pressed,
      "multilayer": %Multilayer.button_pressed,
      "device": "cuda" if %Device.selected == 1 else "cpu",
      "node_format": "long" if %Long.button_pressed else "wide",
   }
   if %Multilayer.button_pressed:
      spec.layer = _column(%Layer, _edge_columns)
   if hyper and %Directed.button_pressed:
      spec.direction = _column(%Direction, _edge_columns)
   if %Signed.button_pressed:
      spec.sign = _column(%Sign, _edge_columns)
   if not _node_columns.is_empty() and kind == "network":
      var keys := [%NodeId]
      if %Multilayer.button_pressed:
         keys.append(%NodeLayer)
      if %Long.button_pressed:
         keys.append_array([%AttrName, %AttrValue])
      var chosen := keys.map(func(ob): return ob.selected)
      if chosen != range(keys.size()):
         spec.node_columns = keys.map(func(ob): return _column(ob, _node_columns))
   AppSettings.store("load", "separator", %Sep.selected)
   AppSettings.store("load", "header", %Header.button_pressed)
   AppSettings.store("load", "device", %Device.selected)
   AppSettings.store_folder("edges", %EdgePath.text)
   if spec.nodes != null:
      AppSettings.store_folder("nodes", spec.nodes)
   get_ok_button().disabled = true
   %Error.visible = false
   _run += 1
   var run := _run
   import_requested.emit(spec, func(error: String):
      if run != _run:   # closed or imported again meanwhile
         return
      get_ok_button().disabled = false
      if error.is_empty():
         hide()
      else:
         %Error.text = error
         %Error.visible = true
         reset_size())
