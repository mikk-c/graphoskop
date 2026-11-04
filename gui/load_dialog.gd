extends ConfirmationDialog
## Load dialog: decides how [code]gsk.load_network[/code] (or [code]load_hypergraph[/code],
## [code]load_simplicial_complex[/code], or [code]load_matrix[/code] for an adjacency matrix) is
## called. The previews are read by the Python worker, as the library will read the files. With
## Adjacency matrix, the first file is the matrix: Header means ids in its first row and column,
## Directed and Bipartite unchecked are found from the matrix, and the node table has the node id
## in its first column.
##
## The roles of the columns (source, target, layer, sign, … of the edge table; node id, layer,
## attribute name and value of the node table) are chosen on the row of column names over each
## preview (drawn by the interface, [method _draw_head]: it stays in place when the rows scroll,
## and follows the columns when they scroll sideways): clicking a name (left or right) opens the
## menu of the roles that apply.
## A column with a role has its symbol next to its name ([constant ROLE_ICONS]: ●→ source, →●
## target, ● node id, …), its name in bold and in the color of the role ([constant ROLE_COLORS],
## ColorBrewer Set1: red source, blue target, black node id, purple and green attribute name and
## value), and its values in that color. A role given to a column that had another one swaps them.
## (The menus of columns are kept, hidden, as the state of the choices.)
##
## Without [b]Advanced[/b] (bottom left; off by default, see [method AppSettings.advanced]) the
## dialog shows only the files, their format and columns, Directed and Bipartite: Adjacency
## matrix, Signed, Multilayer, Hypergraph, Simplicial and the device show only when on (or when
## chosen).

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
var _advanced := CheckButton.new()
## Menu of the roles of a column of a preview, and the column and tree it is open for.
var _role_menu := PopupMenu.new()
var _role_at := [null, -1]


## Colors of the roles of the columns (ColorBrewer Set1; black for node ids).
const ROLE_COLORS := {"source": "#e41a1c", "target": "#377eb8", "node": "#000000", "hyperedge": "#377eb8", "layer": "#ff7f00", "direction": "#f781bf", "sign": "#a65628", "attr_name": "#984ea3", "attr_value": "#4daf4a"}
## Symbols of the roles (SVG, drawn in the color of the role, see [method _icon]): a filled circle
## with an arrow out of it (source), an arrow into it (target), alone (node id), a rounded box
## (hyperedge or simplex), stacked bars (layer), two arrows (direction), plus and minus (sign), a
## tag (attribute name) and an equals sign (attribute value).
const ROLE_ICONS := {
   "source": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="16"><circle cx="6" cy="8" r="5" fill="C"/><path d="M12 8h8M17 4l4 4-4 4" stroke="C" stroke-width="2" fill="none"/></svg>',
   "target": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="16"><path d="M2 8h8M7 4l4 4-4 4" stroke="C" stroke-width="2" fill="none"/><circle cx="18" cy="8" r="5" fill="C"/></svg>',
   "node": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><circle cx="8" cy="8" r="5" fill="C"/></svg>',
   "hyperedge": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect x="1.5" y="3" width="13" height="10" rx="4" fill="none" stroke="C" stroke-width="2"/><circle cx="5.5" cy="8" r="1.8" fill="C"/><circle cx="10.5" cy="8" r="1.8" fill="C"/></svg>',
   "layer": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect x="2" y="2" width="12" height="3" fill="C"/><rect x="2" y="6.5" width="12" height="3" fill="C"/><rect x="2" y="11" width="12" height="3" fill="C"/></svg>',
   "direction": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><path d="M2 5h11M10 2l3 3-3 3M14 11H3M6 8l-3 3 3 3" stroke="C" stroke-width="1.6" fill="none"/></svg>',
   "sign": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><path d="M8 1v8M4 5h8M4 13h8" stroke="C" stroke-width="2"/></svg>',
   "attr_name": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><path d="M1 3h9l5 5-5 5H1z" fill="C"/><circle cx="4.5" cy="8" r="1.5" fill="white"/></svg>',
   "attr_value": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><path d="M2 5h12M2 11h12" stroke="C" stroke-width="2.5"/></svg>',
}
## Background of the row of column names of the previews.
const HEAD_COLOR := Color("#e6e6e6")
var _icons := {}
var _bold := FontVariation.new()
## Row of column names over each preview (tree to its Control), and what each shows: for every
## column, its name and the key of its role ("" for none).
var _heads := {}
var _names := {}
## Height of the row of column names, in pixels.
const HEAD_HEIGHT := 26

## Width of the menus of columns, in pixels (long column names are cut).
const COLUMN_MENU_WIDTH := 150
## Smallest width of a column of the previews, in pixels (longer texts are cut; the table scrolls).
const PREVIEW_COLUMN_WIDTH := 80


func _ready() -> void:
   _advanced.text = "Advanced"
   _advanced.tooltip_text = "Show all the options: adjacency matrices, signed, multilayer and higher-order networks, the device"
   var bar := get_ok_button().get_parent()
   bar.add_child(_advanced)
   bar.move_child(_advanced, 0)
   _advanced.toggled.connect(func(on):
      AppSettings.set_advanced(on)
      _update()
      size = Vector2i.ZERO
      reset_size())
   %Device.item_selected.connect(func(_i): _update())
   add_child(_role_menu)
   _role_menu.index_pressed.connect(_on_role)
   _bold.base_font = ThemeDB.fallback_font
   _bold.variation_embolden = 0.9
   for tree in [%EdgePreview, %NodePreview]:
      tree.column_titles_visible = false   # the names are drawn over it, with the symbols of the roles
      var head := Control.new()
      head.custom_minimum_size.y = HEAD_HEIGHT
      head.mouse_filter = Control.MOUSE_FILTER_STOP
      head.clip_contents = true
      tree.get_parent().add_child(head)
      tree.get_parent().move_child(head, tree.get_index())
      _heads[tree] = head
      _names[tree] = []
      head.draw.connect(func(): _draw_head(tree, head))
      head.gui_input.connect(func(event):
         var j := _column_at(tree, event.position.x) if event is InputEventMouse else -1
         if event is InputEventMouseMotion:
            head.tooltip_text = _head_tip(tree, j)
         elif event is InputEventMouseButton and event.pressed and event.button_index in [MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT] and j >= 0:
            _ask_role(tree, j)
            head.accept_event())
      tree.resized.connect(head.queue_redraw)
      tree.draw.connect(head.queue_redraw)   # its columns are placed when it is drawn
      for scroll in tree.get_children(true):   # its own scroll bars: the names follow the columns sideways
         if scroll is ScrollBar:
            scroll.value_changed.connect(func(_v): head.queue_redraw())
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
   _advanced.set_pressed_no_signal(AppSettings.advanced())
   _update()
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
         var root0 := tree.create_item()
         tree.create_item(root0).set_text(0, "Cannot read the file")
         tree.create_item(root0).set_text(0, str(reply.error))
         _names[tree] = []
         _heads[tree].queue_redraw()
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
   for box in [%Matrix, %Signed, %Multilayer, %Hypergraph, %Simplicial]:   # advanced options, unless chosen
      box.visible = _advanced.button_pressed or box.button_pressed
   %Device.visible = _advanced.button_pressed or %Device.selected != 0
   $Box/Kind/DeviceLabel.visible = %Device.visible
   for c in $Box/EdgeColumns.get_children() + %NodeColumns.get_children():   # the roles are chosen on the column titles
      if c != %Long:
         c.visible = false
   %EdgeColumns.visible = false
   %EdgeTitle.text += " (click a column name to choose its role)" if edges and not matrix else ""
   %NodeTitle.text = "Node table: first 10 rows" + (" (click a column name to choose its role)" if nodes and not matrix else "")
   _title_columns(%EdgePreview, _edge_columns)
   _title_columns(%NodePreview, _node_columns)
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


## The roles that apply to the columns of [param tree] (edge or node preview): label, the (hidden)
## menu holding the column chosen, and the key of its symbol and color.
func _roles(tree: Tree) -> Array:
   var higher: bool = %Hypergraph.button_pressed or %Simplicial.button_pressed
   var multilayer: bool = %Multilayer.button_pressed
   if %Matrix.button_pressed:
      return []
   if tree == %EdgePreview:
      var roles := [[%SourceLabel.text, %Source, "node" if higher else "source"], [%TargetLabel.text, %Target, "hyperedge" if higher else "target"]]
      if multilayer:
         roles.append(["Layer", %Layer, "layer"])
      if %Hypergraph.button_pressed and %Directed.button_pressed:
         roles.append(["Direction", %Direction, "direction"])
      if %Signed.button_pressed:
         roles.append(["Sign", %Sign, "sign"])
      return roles
   var roles := [] if higher else [[%NodeIdLabel.text, %NodeId, "node"]]
   if multilayer:
      roles.append(["Layer", %NodeLayer, "layer"])
   if %Long.button_pressed and not higher:
      roles.append_array([["Attribute name", %AttrName, "attr_name"], ["Attribute value", %AttrValue, "attr_value"]])
   return roles


## Symbol of a role in its color (made once).
func _icon(key: String) -> Texture2D:
   if not _icons.has(key):
      var image := Image.new()
      image.load_svg_from_string(ROLE_ICONS[key].replace("\"C\"", "\"%s\"" % ROLE_COLORS[key]), 1.0)
      _icons[key] = ImageTexture.create_from_image(image)
   return _icons[key]


## Shows the role of each column of [param tree] on its row of names (symbol, bold name in the
## color of the role: the first role, if a column has two) and its values in that color (but
## black, which would not show on the dark rows).
func _title_columns(tree: Tree, columns: PackedStringArray) -> void:
   var root := tree.get_root()
   if columns.is_empty() or tree.columns != columns.size() or root == null:
      _names[tree] = []
      _heads[tree].queue_redraw()
      return
   var roles := _roles(tree)
   var shown := []
   for j in columns.size():
      var mine := roles.filter(func(r): return r[1].selected == j)
      var key: String = mine[0][2] if not mine.is_empty() else ""
      var color := Color(ROLE_COLORS[key]) if key != "" else Color.WHITE
      shown.append({"name": columns[j] if %Header.button_pressed else "column %d" % j, "key": key, "roles": mine.map(func(r): return r[0])})
      var item := root.get_first_child()
      while item != null:
         if key != "" and key != "node":
            item.set_custom_color(j, color)
         else:
            item.clear_custom_color(j)
         item = item.get_next()
   _names[tree] = shown
   _heads[tree].queue_redraw()


## Horizontal span (x, width) of column [param j] of [param tree], in the coordinates of the tree
## and of its row of names (they share their left border), as it is drawn (scrolled sideways).
func _column_span(tree: Tree, j: int) -> Vector2:
   var first := tree.get_root().get_first_child() if tree.get_root() != null else null
   if first == null:
      return Vector2(-1, 0)
   var r: Rect2 = tree.get_item_area_rect(first, j)
   return Vector2(r.position.x, r.size.x)


## The column of [param tree] under [param x] (row of names), or -1.
func _column_at(tree: Tree, x: float) -> int:
   for j in (_names[tree] as Array).size():
      var span := _column_span(tree, j)
      if x >= span.x and x < span.x + span.y:
         return j
   return -1


func _head_tip(tree: Tree, j: int) -> String:
   if j < 0:
      return ""
   var roles: Array = _names[tree][j].roles
   return ("%s · " % " and ".join(roles) if not roles.is_empty() else "") + "click to choose the role of this column"


## Draws the row of column names over [param tree]: on a light band, every name over its column,
## after the symbol of its role, bold and in the color of the role.
func _draw_head(tree: Tree, head: Control) -> void:
   head.draw_rect(Rect2(Vector2.ZERO, head.size), HEAD_COLOR)
   var size := head.get_theme_default_font_size()
   for j in (_names[tree] as Array).size():
      var span := _column_span(tree, j)
      if span.x < 0 or span.x > head.size.x:
         continue
      var shown: Dictionary = _names[tree][j]
      var x := span.x + 4
      if shown.key != "":
         var icon := _icon(shown.key)
         head.draw_texture(icon, Vector2(x, (HEAD_HEIGHT - icon.get_height()) / 2.0))
         x += icon.get_width() + 4
      var font: Font = _bold if shown.key != "" else ThemeDB.fallback_font
      var color := Color(ROLE_COLORS[shown.key]) if shown.key != "" else Color("#333333")
      head.draw_string(font, Vector2(x, (HEAD_HEIGHT + size * 0.7) / 2.0), shown.name, HORIZONTAL_ALIGNMENT_LEFT, maxf(span.x + span.y - x - 4, 1), size, color)
      head.draw_line(Vector2(span.x + span.y - 1, 4), Vector2(span.x + span.y - 1, HEAD_HEIGHT - 4), Color(0, 0, 0, 0.15))


## Opens the menu of the roles of column [param j] of [param tree] at the mouse.
func _ask_role(tree: Tree, j: int) -> void:
   var roles := _roles(tree)
   if roles.is_empty():
      return
   _role_at = [tree, j]
   _role_menu.clear()
   for r in roles:
      _role_menu.add_icon_radio_check_item(_icon(r[2]), r[0])
      _role_menu.set_item_checked(_role_menu.item_count - 1, r[1].selected == j)
   _role_menu.popup(Rect2i(DisplayServer.mouse_get_position(), Vector2i.ZERO))


## A role chosen in the menu of a column: the column gets it; the role the column had (if any)
## goes to the column that had this one.
func _on_role(i: int) -> void:
   var roles := _roles(_role_at[0])
   if i >= roles.size():
      return
   var ob: OptionButton = roles[i][1]
   var j: int = _role_at[1]
   for r in roles:
      if r[1] != ob and r[1].selected == j:
         r[1].select(ob.selected)
   ob.select(j)
   _update()


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
