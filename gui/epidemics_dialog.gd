extends AcceptDialog
## Simulation › Epidemics: runs epidemic models on the network of the current tab and compares
## them. Add model opens the parameter dialog of a model (its run gets a name, e.g. "SIR 1"); every
## run stores its table of states in the network-wide dictionary [code]epidemics[/code] (shown in
## the Tables tab of the data panel). The runs are listed with check boxes, and with Copy…, which
## opens the parameter dialog of the same model with the values of that run, to change some of
## them. Plot draws the checked runs in one plot, for the checked states (e.g. I, the infected, or
## S, I and R), to compare how fast and how far the models spread on the same network. A line
## under the runs says what is going on (a model running, the runs being updated). The dialog is
## not exclusive: it stays open to add more runs, with the parameter dialog and the plot above it.

## Emitted by Add model or Copy…, with the function of the model, the name proposed for its run,
## and the values of the parameters (Python source code, by name; empty for Add model).
signal add_requested(function: String, name: String, values: Dictionary)
## Emitted by Plot, with the names of the checked runs and the checked states.
signal plot_requested(names: Array, states: Array)

## Models: label (also the start of the names of their runs) and function of the library.
const MODELS := [
   ["SI", "si_model"],
   ["SIS", "sis_model"],
   ["SIR", "sir_model"],
   ["Complex SI", "complex_si_model"],
   ["Complex SIS", "complex_sis_model"],
   ["Complex SIR", "complex_sir_model"],
   ["Complex nonprogressive", "complex_nonprogressive_model"],
   ["Higher-order SI", "higher_order_si_model"],
   ["Higher-order SIS", "higher_order_sis_model"],
   ["Higher-order SIR", "higher_order_sir_model"],
]

var _names := []
var _unchecked := {}
var _chosen := {"I": true}


func _ready() -> void:
   var popup: PopupMenu = %Add.get_popup()
   for i in MODELS.size():
      popup.add_item(MODELS[i][0] + "…", i)
   popup.id_pressed.connect(func(i): add_requested.emit(MODELS[i][1], _free_name(MODELS[i][0]), {}))
   %Plot.pressed.connect(func():
      var states := []
      for box in %States.get_children():
         if box.button_pressed:
            states.append(box.text)
      plot_requested.emit(_names.filter(func(n): return not _unchecked.has(n)), states))


## Opens the dialog titled [param heading] with the runs of a network (see [method show_runs]).
func open(heading: String, runs: Dictionary) -> void:
   title = heading
   _unchecked = {}
   show_runs(runs)
   show_status("")
   popup_centered()


## Lists the runs of a network (from the worker: [code]tables[/code], their names; [code]states
## [/code], run name to its states; [code]models[/code], run name to its function and arguments):
## new runs are checked, and runs made here get Copy….
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
      box.button_pressed = not _unchecked.has(n)
      box.toggled.connect(func(on): if on: _unchecked.erase(n) else: _unchecked[n] = true)
      row.add_child(box)
      if models.has(n):
         var copy := Button.new()
         copy.text = "Copy…"
         copy.tooltip_text = "Runs %s again with the values of %s, to change some" % [models[n].function, n]
         var label := _label(models[n].function)
         copy.pressed.connect(func(): add_requested.emit(models[n].function, _free_name(label), models[n].arguments))
         row.add_child(copy)
      %Runs.add_child(row)
   %None.visible = _names.is_empty()
   var all := []
   for n in _names:
      for st in runs.get("states", {}).get(n, []):
         if not all.has(st):
            all.append(st)
   for child in %States.get_children():
      _chosen[child.text] = child.button_pressed
      %States.remove_child(child)
      child.queue_free()
   for st in all:
      var box := CheckBox.new()
      box.text = st
      box.button_pressed = _chosen.get(st, false)
      %States.add_child(box)
   %Plot.disabled = _names.is_empty()
   _fit()


## Shows what is going on (e.g. a model running), or hides the line with an empty text.
func show_status(text: String) -> void:
   %Status.text = text
   %Status.visible = not text.is_empty()
   _fit()


## Shows an error under the runs, or hides it.
func show_error(text: String) -> void:
   %Error.text = text
   %Error.visible = not text.is_empty()
   _fit()


## The label of the model [param function] (see [constant MODELS]).
func _label(function: String) -> String:
   for m in MODELS:
      if m[1] == function:
         return m[0]
   return function


## The first name "[param prefix] k" not used by a run.
func _free_name(prefix: String) -> String:
   var k := 1
   while _names.has("%s %d" % [prefix, k]):
      k += 1
   return "%s %d" % [prefix, k]


## Shrinks the dialog to its controls.
func _fit() -> void:
   size = Vector2i.ZERO
   reset_size()
