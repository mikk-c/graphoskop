extends TabContainer
## Data panel, with three tabs: Nodes (the attributes of all nodes, or of the selected node),
## Edges (all edges, or those of the selected node) and Network (its properties). Only the visible
## tab is asked to the worker, one page at a time.

## The worker (set by the main scene).
var worker: Node
var _tab := -1
var _node := -1
var _label := ""
var _offsets := {}
var _serial := 0


func _ready() -> void:
   tab_changed.connect(func(_i): refresh())
   for view in get_children():
      view.page_requested.connect(func(offset): _offsets[view.name] = offset; refresh())


## Shows the network of the worker tab [param tab] (-1 for none), with the node at position
## [param node] of the drawing selected (-1 for none), whose id is [param label].
func show_network(tab: int, node := -1, label := "") -> void:
   _tab = tab
   _node = node
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
      return
   var title: String = {"nodes": "All nodes" if _node < 0 else "Node " + _label, "edges": "All edges" if _node < 0 else "Edges of " + _label, "network": "Network"}[kind]
   var request := {"cmd": "table", "tab": _tab, "kind": kind, "offset": offset, "count": view.PAGE}
   if _node >= 0 and kind != "network":
      request.node = _node
   worker.request(request, func(reply, _payload):
      if serial == _serial and reply.ok:
         view.show_rows(reply, offset, title))
