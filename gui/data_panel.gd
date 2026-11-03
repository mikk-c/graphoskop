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
            view.show_rows({}, 0, "No simulations yet (Simulation › Epidemics…)")
            return
         worker.request({"cmd": "table", "tab": _tab, "kind": "tables", "name": _table, "offset": offset, "count": view.PAGE, "sort": view.sort_column, "ascending": view.ascending}, func(r, _q):
            if serial == _serial and r.ok:
               view.show_rows(r, offset, "Run " + _table)))
      return
   var title := {"nodes": "All nodes", "edges": "All edges", "network": "Network"}[kind] as String
   var request := {"cmd": "table", "tab": _tab, "kind": kind, "offset": offset, "count": view.PAGE, "sort": view.sort_column, "ascending": view.ascending}
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
   var total: int = int(reply.get("attributes", 0))
   if total <= shown:
      return ""
   var hint := " (select one node to see them all)" if kind == "nodes" else ""
   return " · first %d of %s attributes%s" % [shown, _thousands(total), hint]


## [param n] with thousands separated by commas.
static func _thousands(n: int) -> String:
   var t := str(n)
   var out := ""
   while t.length() > 3:
      out = "," + t.right(3) + out
      t = t.left(t.length() - 3)
   return t + out
