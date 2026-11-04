extends TabContainer
## Data panel, with four tabs: Nodes (the attributes of all nodes, of the selected node, or the rows
## of the selected nodes), Edges (all edges, those of the selected nodes, or the selected edges),
## Network (its properties) and Tables (the tables of the simulations of the network, one at a
## time, chosen in a menu). Only the visible tab is asked to the worker, one page at a time; each
## table is sorted as a whole by the column whose title was clicked (table_view.gd). A right click
## on the title of a column of the Nodes or Edges table (not of the attributes of a single node)
## asks for the menu of its attribute. Double clicking a value of the Nodes or Edges table (the
## value column of a single node) edits it in place ([signal value_edit_requested]); ids, sources
## and targets cannot be edited. Networks with many attributes (more than the worker's
## [code]TABLE_ATTRIBUTES[/code]) show only the first ones as columns, and say so in the title;
## the table of one selected node lists them all.
##
## The Nodes and Edges tables have a [b]Columns[/b] menu: [b]Show all[/b], [b]Hide all[/b] and a
## check box per attribute (checked: shown), with patterns of names as in the parameter dialog
## (e.g. [code]label_*[/code] for all the attributes starting with label_; beyond its
## 200 names, those of a pattern are not listed one by one).
## Hidden attributes are left out of the table (also of the table of one node); the choice is
## kept for each network and kind, until the network is closed ([method hide_attribute] hides one,
## from the menu of its column).

## Emitted when the user right-clicks the title of the column of [param attribute] in the Nodes
## ([param kind] [code]"nodes"[/code]) or Edges ([code]"edges"[/code]) table.
signal attribute_menu_requested(kind: String, attribute: String)
## Emitted when a value was edited in the Nodes ([param kind] [code]"nodes"[/code]) or Edges table:
## the [param position] of the node or edge in the drawing order, the [param attribute] and the
## [param text] typed.
signal value_edit_requested(kind: String, position: int, attribute: String, text: String)

## The worker (set by the main scene).
var worker: Node
var _tab := -1
var _nodes := PackedInt32Array()
var _edges := PackedInt32Array()
var _label := ""
var _offsets := {}
var _serial := 0
var _table := ""
## Returns the attributes of the current network (from the worker: nodes and edges; set by the
## main scene), for the Columns menus.
var attributes_of := func() -> Dictionary: return {}
## Hidden attributes, by network (worker tab) and kind: whether all are hidden ([code]all[/code]),
## and the names that are not as the others ([code]except[/code], a set).
var _hide := {}
const Params := preload("res://params_dialog.gd")


func _ready() -> void:
   tab_changed.connect(func(_i): refresh())
   for view in get_children():
      view.page_requested.connect(func(offset): _offsets[view.name] = offset; refresh())
      view.sort_requested.connect(func(_c, _a): _offsets.erase(view.name); refresh())
   for view in [$Nodes, $Edges]:
      var kind := String(view.name).to_lower()
      view.column_menu_requested.connect(func(column):
         if _tab >= 0 and not (kind == "nodes" and _nodes.size() == 1):
            attribute_menu_requested.emit(kind, column))
   $Nodes.editable = func(column): return column == "value" if _nodes.size() == 1 else column != "node"
   $Edges.editable = func(column): return not column in ["source", "target"]
   $Nodes.value_edited.connect(func(index, column, typed):
      if _nodes.size() == 1:   # the attributes of one node: the row names the attribute
         var row: TreeItem = $Nodes.get_node("%Table").get_root().get_child(index)
         value_edit_requested.emit("nodes", _nodes[0], row.get_text(0), typed)
      else:
         value_edit_requested.emit("nodes", index, column, typed))
   $Edges.value_edited.connect(func(index, column, typed): value_edit_requested.emit("edges", index, column, typed))
   $Tables.choice_changed.connect(func(name): _table = name; _offsets.erase("Tables"); refresh())
   for view in [$Nodes, $Edges]:
      var kind := String(view.name).to_lower()
      var menu: MenuButton = view.get_node("%Columns")
      menu.visible = true
      menu.get_popup().hide_on_checkable_item_selection = false
      menu.about_to_popup.connect(func(): _fill_columns(kind, menu.get_popup()))
      menu.get_popup().index_pressed.connect(func(i): _on_column(kind, menu.get_popup(), i))


## Shows the network of the worker tab [param tab] (-1 for none), with the selected [param nodes]
## and [param edges] (positions in the drawing order); [param label] is the id of the node when
## one is selected.
func show_network(tab: int, nodes := PackedInt32Array(), edges := PackedInt32Array(), label := "") -> void:
   _tab = tab
   _nodes = nodes
   _edges = edges
   _label = label
   _offsets = {}
   refresh()


## Asks the worker for the visible page of the current tab.
func refresh() -> void:
   var view := get_current_tab_control()
   var kind := String(view.name).to_lower()
   var offset: int = _offsets.get(view.name, 0)
   _serial += 1
   var serial := _serial
   if _tab < 0:
      for v in get_children():
         v.show_rows({}, 0, "")
      $Tables.set_choices([], "")
      return
   if kind == "tables":
      worker.request({"cmd": "simulations", "tab": _tab}, func(reply, _p):
         if serial != _serial or not reply.ok:
            return
         var names: Array = reply.tables
         if not names.has(_table):
            _table = names[0] if not names.is_empty() else ""
         view.set_choices(names, _table)
         if _table.is_empty():
            view.show_rows({}, 0, "No tables yet (Simulation › Epidemics…, Analysis › Frequent subgraphs)")
            return
         worker.request({"cmd": "table", "tab": _tab, "kind": "tables", "name": _table, "offset": offset, "count": view.PAGE, "sort": view.sort_column, "ascending": view.ascending}, func(r, _q):
            if serial == _serial and r.ok:
               view.show_rows(r, offset, _table)))
      return
   var title := {"nodes": "All nodes", "edges": "All edges", "network": "Network"}[kind] as String
   var request := {"cmd": "table", "tab": _tab, "kind": kind, "offset": offset, "count": view.PAGE, "sort": view.sort_column, "ascending": view.ascending}
   var hide := _hidden(kind) if kind in ["nodes", "edges"] else {"all": false, "except": {}}
   if hide.all or not hide.except.is_empty():
      request.hide = {"all": hide.all, "except": hide.except.keys()}
   if kind == "nodes" and _nodes.size() == 1:
      request.node = _nodes[0]
      title = "Node " + _label
   elif kind == "nodes" and _nodes.size() > 1:
      request.nodes = Array(_nodes)
      title = "%d selected nodes" % _nodes.size()
   elif kind == "edges" and not _edges.is_empty():
      request.edges = Array(_edges)
      title = "%d selected edges" % _edges.size()
   elif kind == "edges" and _nodes.size() == 1:
      request.node = _nodes[0]
      title = "Edges of " + _label
   elif kind == "edges" and _nodes.size() > 1:
      request.nodes = Array(_nodes)
      title = "Edges of the %d selected nodes" % _nodes.size()
   worker.request(request, func(reply, _payload):
      if serial == _serial and reply.ok:
         view.show_rows(reply, offset, title + _left_out(reply, kind)))


## Note for the title of a table whose attributes are not all columns (see the worker's
## [code]table[/code] command), or nothing.
static func _left_out(reply: Dictionary, kind: String) -> String:
   var shown: int = reply.get("columns", []).size() - (1 if kind == "nodes" else 2)
   var hidden: int = int(reply.get("hidden", 0))
   var total: int = int(reply.get("attributes", 0)) - hidden
   var note := " · %s hidden (Columns)" % (_thousands(hidden) + (" attribute" if hidden == 1 else " attributes")) if hidden > 0 else ""
   if total <= shown:
      return note
   var hint := " (select one node to see them all)" if kind == "nodes" else ""
   return " · first %d of %s attributes%s%s" % [shown, _thousands(total), hint, note]


## [param n] with thousands separated by commas.
static func _thousands(n: int) -> String:
   var t := str(n)
   var out := ""
   while t.length() > 3:
      out = "," + t.right(3) + out
      t = t.left(t.length() - 3)
   return t + out


## The hidden attributes of [param kind] ("nodes" or "edges") of the current network.
func _hidden(kind: String) -> Dictionary:
   if not _hide.has(_tab):
      _hide[_tab] = {"nodes": {"all": false, "except": {}}, "edges": {"all": false, "except": {}}}
   return _hide[_tab][kind]


## Whether the attribute [param name] of [param kind] is shown.
func _shown(kind: String, name: String) -> bool:
   var hide := _hidden(kind)
   return hide.except.has(name) == hide.all


func _set_shown(kind: String, name: String, on: bool) -> void:
   var hide := _hidden(kind)
   if on == hide.all:
      hide.except[name] = true
   else:
      hide.except.erase(name)


## Hides the attribute [param name] of [param kind] (from the menu of its column).
func hide_attribute(kind: String, name: String) -> void:
   _set_shown(kind, name, false)
   refresh()


## Forgets the hidden attributes of the network of worker tab [param tab] (closed).
func forget(tab: int) -> void:
   _hide.erase(tab)


## Fills the Columns menu of [param kind]: Show all, Hide all, then the patterns and the names
## (checked: shown); the metadata of each item is what it toggles.
func _fill_columns(kind: String, popup: PopupMenu) -> void:
   popup.clear()
   var names: Array = attributes_of.call().get(kind, []).map(func(a): return a.name)
   popup.add_item("Show all")
   popup.set_item_metadata(0, ["all", false])
   popup.add_item("Hide all")
   popup.set_item_metadata(1, ["all", true])
   if names.is_empty():
      return
   popup.add_separator()
   var patterns: Array = Params._patterns(names)
   for pt in patterns:
      var members := names.filter(func(n): return n.match(pt))
      popup.add_check_item("%s (%d)" % [pt, members.size()])
      popup.set_item_checked(popup.item_count - 1, members.all(func(n): return _shown(kind, n)))
      popup.set_item_metadata(popup.item_count - 1, ["names", members])
   if not patterns.is_empty():
      popup.add_separator()
   var listed := names if names.size() <= Params.MAX_CHECKS else names.filter(func(n): return not patterns.any(func(pt): return n.match(pt)))
   for n in listed.slice(0, Params.MAX_CHECKS):
      popup.add_check_item(n)
      popup.set_item_checked(popup.item_count - 1, _shown(kind, n))
      popup.set_item_metadata(popup.item_count - 1, ["names", [n]])
   if listed.size() > Params.MAX_CHECKS:
      popup.add_item("… %d more: Show all and Hide all reach them" % (listed.size() - Params.MAX_CHECKS))
      popup.set_item_disabled(popup.item_count - 1, true)


func _on_column(kind: String, popup: PopupMenu, i: int) -> void:
   var what = popup.get_item_metadata(i)
   if what == null:
      return
   if what[0] == "all":
      _hidden(kind).all = what[1]
      _hidden(kind).except = {}
   else:
      var on := not popup.is_item_checked(i)
      for n in what[1]:
         _set_shown(kind, n, on)
   _fill_columns(kind, popup)
   _offsets.erase("Nodes" if kind == "nodes" else "Edges")
   refresh()


## The hidden attributes of the network of worker tab [param tab], for a session: for nodes and
## edges, whether all are hidden and the names that are not as the others.
func hidden_state(tab: int) -> Dictionary:
   var state := {}
   for kind in (_hide.get(tab, {}) as Dictionary):
      state[kind] = {"all": _hide[tab][kind].all, "except": _hide[tab][kind].except.keys()}
   return state


## Puts back the hidden attributes of the network of worker tab [param tab] (see [method hidden_state]).
func set_hidden_state(tab: int, state: Dictionary) -> void:
   if state.is_empty():
      return
   _hide[tab] = {}
   for kind in ["nodes", "edges"]:
      var k: Dictionary = state.get(kind, {"all": false, "except": []})
      var names := {}
      for n in k.get("except", []):
         names[n] = true
      _hide[tab][kind] = {"all": bool(k.get("all", false)), "except": names}


## The tab shown and the table chosen in the Tables tab, for a session.
func panel_state() -> Dictionary:
   return {"tab": current_tab, "table": _table}


## Puts back the tab shown and the table of the Tables tab (see [method panel_state]).
func set_panel_state(state: Dictionary) -> void:
   _table = str(state.get("table", ""))
   current_tab = clampi(int(state.get("tab", 0)), 0, get_tab_count() - 1)
   refresh()
