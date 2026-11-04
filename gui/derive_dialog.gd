extends ConfirmationDialog
## Dialog of Edit › New node attribute…: a node attribute computed, node by node, from a formula
## of other node attributes (see the worker's [code]formula_values[/code]). The formula is typed,
## with help: [b]Attribute[/b] inserts the name of an attribute (in backquotes if it is not a
## plain word), [b]Function[/b] a function (with its description), and the buttons the operators.
## While typing, the values the formula gives are described under it (their kind, range, missing
## values and first values), or what is wrong with it. [b]Create[/b] runs the code; the dialog
## stays open on errors.

## Operators of the buttons: text inserted, and tooltip.
const OPERATORS := [["+", "add numbers, or join texts"], ["-", "subtract"], ["*", "multiply"], ["/", "divide"], ["**", "power"], ["(", "open a parenthesis"], [")", "close a parenthesis"], [">", "greater than (1 or 0)"], ["<", "less than (1 or 0)"], ["==", "equal to (1 or 0)"], ["\"\"", "a text, between quotes"]]

var _names := []
var _functions := []
var _preview := Callable()
var _create := Callable()
var _serial := 0


func _ready() -> void:
   confirmed.connect(_on_confirmed)
   register_text_enter(%Formula)
   for line in [%Name, %Formula]:
      line.text_changed.connect(func(_t): %Debounce.start())
   %Debounce.timeout.connect(_ask_preview)
   %Attributes.get_popup().index_pressed.connect(func(i): _insert(_names[i]))
   %Functions.get_popup().index_pressed.connect(func(i): _insert(_functions[i] + "()", 1))
   for op in OPERATORS:
      var b := Button.new()
      b.text = op[0]
      b.tooltip_text = op[1]
      b.custom_minimum_size.x = 30
      var binary: bool = not op[0] in ["(", ")", "\"\""]   # with spaces around
      b.pressed.connect(func(): _insert(" %s " % op[0] if binary else op[0], 1 if op[0] == "\"\"" else 0))
      %Tools.add_child(b)


## Opens the dialog with the node [param attributes] (from the worker's [code]formulas[/code]:
## name, formula, numeric) and [param functions] (name, doc), starting with [param name] and
## [param formula]. [param preview] is called with the name, the formula and a function taking
## the description of the values and an error; [param create] with the name, the formula and a
## function taking an error (empty: created).
func open(attributes: Array, functions: Array, name: String, formula: String, preview: Callable, create: Callable) -> void:
   _preview = preview
   _create = create
   _names = attributes.map(func(a): return a.formula)
   _functions = functions.map(func(f): return f.name)
   var pop: PopupMenu = %Attributes.get_popup()
   pop.clear()
   for a in attributes:
      pop.add_item(a.name)
      pop.set_item_tooltip(pop.item_count - 1, "numeric" if a.numeric else "qualitative (texts)")
   pop = %Functions.get_popup()
   pop.clear()
   for f in functions:
      pop.add_item(f.doc)
   %Name.text = name
   %Formula.text = formula
   %Error.visible = false
   get_ok_button().disabled = false
   reset_size()
   popup_centered()
   %Formula.grab_focus()
   %Formula.caret_column = formula.length()
   _ask_preview()


## Inserts [param text] in the formula at the caret, then moves the caret [param back] characters
## back (inside the parentheses of a function, or the quotes of a text).
func _insert(text: String, back := 0) -> void:
   %Formula.insert_text_at_caret(text)
   %Formula.caret_column -= back
   %Formula.grab_focus()
   %Debounce.start()


func _ask_preview() -> void:
   var formula: String = %Formula.text.strip_edges()
   _serial += 1
   var serial := _serial
   if formula.is_empty():
      %Preview.text = "Write the formula of the values: attributes, numbers, operators and functions."
      _show_error("")
      return
   _preview.call(%Name.text.strip_edges(), formula, func(text: String, error: String):
      if serial != _serial:
         return
      %Preview.text = "Values: " + text if error.is_empty() else ""
      _show_error(error))


func _show_error(text: String) -> void:
   %Error.text = text
   %Error.visible = not text.is_empty()
   reset_size()


func _on_confirmed() -> void:
   get_ok_button().disabled = true
   _create.call(%Name.text.strip_edges(), %Formula.text.strip_edges(), func(error: String):
      get_ok_button().disabled = false
      if error.is_empty():
         hide()
      else:
         _show_error(error))
