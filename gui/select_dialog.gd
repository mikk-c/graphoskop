extends ConfirmationDialog
## Select › Select by attribute: selects the nodes (or edges) that satisfy a chain of conditions.
## Each condition compares an attribute with a value: equal, different, smaller, smaller or
## equal, larger, larger or equal (numbers are compared as numbers), or contains (a part of the
## text, or one of the values of a node with several). The value is a number for numeric
## attributes, a menu of the values for qualitative attributes with few values, or a text.
## "Add condition" adds a row; from the second on, each row starts with how it chains to the rows
## above it: AND (both), OR (either), XOR (one but not both), applied from top to bottom (the
## first two rows, then the third with their result, and so on); × removes a row. The selection
## replaces the current one, or is added to it. The dialog stays open if nothing matches or a
## condition is not valid.

## Conditions of numeric attributes and of qualitative ones.
const NUMERIC := ["=", "≠", "<", "≤", ">", "≥"]
const QUALITATIVE := ["=", "≠", "contains"]
## Ways of chaining a condition to the ones above it: label and value for the worker.
const LOGIC := [["AND", "and"], ["OR", "or"], ["XOR", "xor"]]

var _attributes := {}
var _on_accept := Callable()


func _ready() -> void:
   confirmed.connect(_on_confirmed)
   %Kind.item_selected.connect(func(_i): _reset())
   %AddCondition.pressed.connect(_add_row)


## Opens the dialog with the [param attributes] of the network (from the worker: nodes and edges);
## [param on_accept] is called with the kind ("nodes" or "edges"), the conditions (dictionaries
## with attribute, op, value as a text, and logic from the second on), whether to add to the
## selection, and a function to call with an error (or an empty text when the selection is made).
func open(attributes: Dictionary, on_accept: Callable) -> void:
   _attributes = attributes
   _on_accept = on_accept
   show_error("")
   _reset()
   popup_centered()


func _kind() -> String:
   return "nodes" if %Kind.selected == 0 else "edges"


## A single row of conditions on the attributes of the chosen kind.
func _reset() -> void:
   for row in %Conditions.get_children():
      %Conditions.remove_child(row)
      row.queue_free()
   _add_row()


## Adds a row: how it chains (or "whose" on the first row), attribute, condition, value, ×.
func _add_row() -> void:
   var row := HBoxContainer.new()
   var first := %Conditions.get_child_count() == 0
   if first:
      var whose := Label.new()
      whose.text = "whose"
      whose.custom_minimum_size.x = 70
      row.add_child(whose)
   else:
      var logic := OptionButton.new()
      for l in LOGIC:
         logic.add_item(l[0])
      logic.custom_minimum_size.x = 70
      logic.tooltip_text = "AND: both; OR: either; XOR: one but not both (with the rows above)"
      row.add_child(logic)
   var attribute := OptionButton.new()
   attribute.custom_minimum_size.x = 170
   attribute.fit_to_longest_item = false
   for a in _attributes.get(_kind(), []):
      attribute.add_item(a.name)
   var op := OptionButton.new()
   op.custom_minimum_size.x = 90
   op.tooltip_text = "Condition: numbers are compared as numbers; contains matches a part of the text, or one of several values"
   var number := SpinBox.new()
   number.step = 0.
   number.allow_greater = true
   number.allow_lesser = true
   number.custom_arrow_step = 1.
   var values := OptionButton.new()
   values.fit_to_longest_item = false
   var text := LineEdit.new()
   text.placeholder_text = "value"
   for c in [number, values, text]:
      c.custom_minimum_size.x = 150
   var remove := Button.new()
   remove.text = "×"
   remove.tooltip_text = "Remove this condition"
   remove.pressed.connect(func(): _remove_row(row))
   for c in [attribute, op, number, values, text, remove]:
      row.add_child(c)
   %Conditions.add_child(row)
   attribute.item_selected.connect(func(_i): _show_value(row))
   _show_value(row)
   _update_removes()


func _remove_row(row: HBoxContainer) -> void:
   if %Conditions.get_child_count() < 2:
      return
   var was_first := row.get_index() == 0
   %Conditions.remove_child(row)
   row.queue_free()
   if was_first:   # the new first row says "whose"
      var head: HBoxContainer = %Conditions.get_child(0)
      var logic := head.get_child(0)
      var whose := Label.new()
      whose.text = "whose"
      whose.custom_minimum_size.x = 70
      head.remove_child(logic)
      logic.queue_free()
      head.add_child(whose)
      head.move_child(whose, 0)
   _update_removes()


func _update_removes() -> void:
   for row in %Conditions.get_children():
      row.get_child(6).disabled = %Conditions.get_child_count() < 2
   reset_size()


func _current(row: HBoxContainer) -> Dictionary:
   var list: Array = _attributes.get(_kind(), [])
   var i: int = row.get_child(1).selected
   return list[i] if i >= 0 and i < list.size() else {}


## The conditions and the value control of the attribute of a row.
func _show_value(row: HBoxContainer) -> void:
   var a := _current(row)
   var numeric: bool = a.get("numeric", false)
   var op: OptionButton = row.get_child(2)
   var number: SpinBox = row.get_child(3)
   var values: OptionButton = row.get_child(4)
   var kept: String = op.get_item_text(op.selected) if op.selected >= 0 else "="
   op.clear()
   for o in (NUMERIC if numeric else QUALITATIVE):
      op.add_item(o)
   op.select(maxi((NUMERIC if numeric else QUALITATIVE).find(kept), 0))
   var listed: bool = not numeric and not a.is_empty() and a.count <= a.values.size()
   number.visible = numeric
   values.visible = listed
   row.get_child(5).visible = not numeric and not listed
   if numeric:
      number.min_value = -1e300
      number.max_value = 1e300
      number.value = (a.min + a.max) / 2
      number.tooltip_text = "Values from %s to %s" % [a.min, a.max]
   elif listed:
      values.clear()
      for v in a.values:
         values.add_item(v)
   get_ok_button().disabled = a.is_empty()
   reset_size()


## Shows an error under the controls, or hides it.
func show_error(text: String) -> void:
   %Error.text = text
   %Error.visible = not text.is_empty()
   reset_size()


func _on_confirmed() -> void:
   var conditions := []
   for row in %Conditions.get_children():
      var a := _current(row)
      if a.is_empty():
         return
      var value: String
      if row.get_child(3).visible:
         value = str(row.get_child(3).value)
      elif row.get_child(4).visible:
         value = row.get_child(4).get_item_text(row.get_child(4).selected)
      else:
         value = row.get_child(5).text.strip_edges()
      var c := {"attribute": a.name, "op": row.get_child(2).get_item_text(row.get_child(2).selected), "value": value}
      if row.get_index() > 0:
         c.logic = LOGIC[row.get_child(0).selected][1]
      conditions.append(c)
   _on_accept.call(_kind(), conditions, %Add.button_pressed, func(error: String):
      if error.is_empty():
         hide()
      else:
         show_error(error))
