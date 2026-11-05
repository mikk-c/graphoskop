extends AcceptDialog
## A dialog of runs, as the epidemics dialog: Simulation › Robustness (the default functions) and
## Simulation › Synchronization (the functions given by [method set_functions]). For robustness:
## runs the robustness functions of the library on the network of the current tab and compares
## them, as the epidemics dialog does for epidemic models. Add run opens
## the parameter dialog of node removal, edge removal or cascading failures (the name of the run,
## by default the function and its strategy, e.g. node_removal(degree)); every run stores its
## table in the dictionary [code]robustness[/code] and its global value as a network attribute.
## The runs are listed with check boxes, and with Copy…, which opens the parameter dialog of the
## same function with the values of that run, to change some of them. Plot draws the checked runs
## in one plot: removals as the quality against the fraction removed, cascades as the fraction of
## items failed at each step (removals and cascades, or two qualities, are plotted apart). A line
## under the runs says what is going on. The dialog is not exclusive: it stays open to add more
## runs, with the parameter dialog and the plot above it.

## Emitted by Add run or Copy…, with the function and the values of the parameters (Python source
## code, by name; empty for Add run).
signal add_requested(function: String, values: Dictionary)
## Emitted by Plot, with the names of the checked runs.
signal plot_requested(names: Array)

## Functions: label and function of the library.
const FUNCTIONS := [
   ["Node removal", "node_removal"],
   ["Edge removal", "edge_removal"],
   ["Cascading failures", "cascade_failures"],
]

var _names := []
var _unchecked := {}


## Functions of the Add run menu (label and function of the library).
var functions := FUNCTIONS


func _ready() -> void:
   set_functions(FUNCTIONS)
   %Add.get_popup().id_pressed.connect(func(i): add_requested.emit(functions[i][1], {}))
   %Plot.pressed.connect(func(): plot_requested.emit(_names.filter(func(n): return not _unchecked.has(n))))


## The functions of the Add run menu ([label, function]), with the [param tooltip] of the menu
## (empty: as it is).
func set_functions(items: Array, tooltip := "") -> void:
   functions = items
   var popup: PopupMenu = %Add.get_popup()
   popup.clear()
   for i in items.size():
      popup.add_item(items[i][0] + "…", i)
   if not tooltip.is_empty():
      %Add.tooltip_text = tooltip


## The first name "[param prefix] k" not used by a run.
func free_name(prefix: String) -> String:
   var k := 1
   while _names.has("%s %d" % [prefix, k]):
      k += 1
   return "%s %d" % [prefix, k]


## Opens the dialog titled [param heading] with the runs of a network (see [method show_runs]).
func open(heading: String, runs: Dictionary) -> void:
   title = heading
   _unchecked = {}
   show_runs(runs)
   show_status("")
   popup_centered()


## Lists the runs of a network (from the worker: [code]tables[/code], their names;
## [code]models[/code], run name to its function, arguments and quality): new runs are checked.
func show_runs(runs: Dictionary) -> void:
   _names = runs.get("tables", [])
   var models: Dictionary = runs.get("models", {})
   show_error("")
   for child in %Runs.get_children():
      %Runs.remove_child(child)
      child.queue_free()
   for n in _names:
      var row := HBoxContainer.new()
      var box := CheckBox.new()
      box.text = n
      box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
      box.clip_text = true
      box.custom_minimum_size = Vector2(300, 0)
      box.button_pressed = not _unchecked.has(n)
      box.toggled.connect(func(on): if on: _unchecked.erase(n) else: _unchecked[n] = true)
      row.add_child(box)
      if models.has(n):
         var copy := Button.new()
         copy.text = "Copy…"
         copy.tooltip_text = "Runs %s again with the values of %s, to change some" % [models[n].function, n]
         copy.pressed.connect(func(): add_requested.emit(models[n].function, models[n].arguments))
         row.add_child(copy)
      %Runs.add_child(row)
   %None.visible = _names.is_empty()
   %Plot.disabled = _names.is_empty()
   _fit()


## Shows what is going on (e.g. a run), or hides the line with an empty text.
func show_status(text: String) -> void:
   %Status.text = text
   %Status.visible = not text.is_empty()
   _fit()


## Shows an error under the runs, or hides it.
func show_error(text: String) -> void:
   %Error.text = text
   %Error.visible = not text.is_empty()
   _fit()


## Shrinks the dialog to its controls.
func _fit() -> void:
   size = Vector2i.ZERO
   reset_size()
