extends ConfirmationDialog
## Select › Select by attribute: selects the nodes (or edges) whose attribute satisfies a
## condition: equal, different, smaller, smaller or equal, larger, larger or equal (numbers are
## compared as numbers), or contains (a part of the text, or one of the values of a node with
## several). The value is a number for numeric attributes, a menu of the values for qualitative
## attributes with few values, or a text. The selection replaces the current one, or is added to
## it. The dialog stays open if nothing matches or the condition is not valid.

## Conditions of numeric attributes and of qualitative ones.
const NUMERIC := ["=", "≠", "<", "≤", ">", "≥"]
const QUALITATIVE := ["=", "≠", "contains"]

var _attributes := {}
var _on_accept := Callable()


func _ready() -> void:
   confirmed.connect(_on_confirmed)
   %Kind.item_selected.connect(func(_i): _show_attributes())
   %Attribute.item_selected.connect(func(_i): _show_value())


## Opens the dialog with the [param attributes] of the network (from the worker: nodes and edges);
## [param on_accept] is called with the kind ("nodes" or "edges"), the attribute, the condition,
## the value (a text), whether to add to the selection, and a function to call with an error (or
## an empty text when the selection is made).
func open(attributes: Dictionary, on_accept: Callable) -> void:
   _attributes = attributes
   _on_accept = on_accept
   show_error("")
   _show_attributes()
   popup_centered()


func _kind() -> String:
   return "nodes" if %Kind.selected == 0 else "edges"


func _current() -> Dictionary:
   var list: Array = _attributes.get(_kind(), [])
   return list[%Attribute.selected] if %Attribute.selected >= 0 and %Attribute.selected < list.size() else {}


func _show_attributes() -> void:
   %Attribute.clear()
   for a in _attributes.get(_kind(), []):
      %Attribute.add_item(a.name)
   _show_value()


## The conditions and the value control of the chosen attribute.
func _show_value() -> void:
   var a := _current()
   var numeric: bool = a.get("numeric", false)
   var op: String = %Op.get_item_text(%Op.selected) if %Op.selected >= 0 else "="
   %Op.clear()
   for o in (NUMERIC if numeric else QUALITATIVE):
      %Op.add_item(o)
   %Op.select(maxi((NUMERIC if numeric else QUALITATIVE).find(op), 0))
   var listed: bool = not numeric and not a.is_empty() and a.count <= a.values.size()
   %Number.visible = numeric
   %Values.visible = listed
   %Text.visible = not numeric and not listed
   if numeric:
      %Number.min_value = -1e300
      %Number.max_value = 1e300
      %Number.value = (a.min + a.max) / 2
      %Number.tooltip_text = "Values from %s to %s" % [a.min, a.max]
   elif listed:
      %Values.clear()
      for v in a.values:
         %Values.add_item(v)
   get_ok_button().disabled = a.is_empty()
   reset_size()


## Shows an error under the controls, or hides it.
func show_error(text: String) -> void:
   %Error.text = text
   %Error.visible = not text.is_empty()
   reset_size()


func _on_confirmed() -> void:
   var a := _current()
   if a.is_empty():
      return
   var value: String
   if %Number.visible:
      value = str(%Number.value)
   elif %Values.visible:
      value = %Values.get_item_text(%Values.selected)
   else:
      value = %Text.text.strip_edges()
   _on_accept.call(_kind(), a.name, %Op.get_item_text(%Op.selected), value, %Add.button_pressed, func(error: String):
      if error.is_empty():
         hide()
      else:
         show_error(error))
