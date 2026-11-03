extends AcceptDialog
## Analysis › Graph neural networks: trains graph neural networks on the network of the current tab
## and compares them on their validation sets. Add model opens the parameter dialog of a model
## (its run gets a name, e.g. "GCN 1"; the target, the hidden layers, the features…); the runs are
## listed with their validation scores (classification: accuracy, macro F1 and the areas under the
## ROC and precision-recall curves; regression: R² and mean squared error), check boxes and Copy…,
## which opens the parameter dialog of the same model with the values of that run. Plot draws the
## checked runs in one plot (Plot menu; the plot window can switch too): ROC or precision-recall
## curves of the classification runs, predicted against observed values of the regression runs,
## or the learning curves of any run. The dialog is not exclusive: it stays open to add more runs,
## with the parameter dialog and the plot above it.

## Emitted by Add model or Copy…, with the function of the model, the name proposed for its run,
## and the values of the parameters (Python source code, by name; empty for Add model).
signal add_requested(function: String, name: String, values: Dictionary)
## Emitted by Plot, with the names of the checked runs and the plot ("roc", "pr", "scatter" or
## "loss").
signal plot_requested(names: Array, kind: String)

## Models: label (also the start of the names of their runs) and function of the library.
const MODELS := [
   ["GCN", "gcn"],
   ["GraphSAGE", "graphsage"],
   ["GAT", "gat"],
   ["GATv2", "gatv2"],
   ["DiffPool", "diffpool"],
]
## Plots of the Plot menu, in its order.
const KINDS := ["roc", "pr", "scatter", "loss"]

var _names := []
var _unchecked := {}


func _ready() -> void:
   var popup: PopupMenu = %Add.get_popup()
   for i in MODELS.size():
      popup.add_item(MODELS[i][0] + "…", i)
   popup.id_pressed.connect(func(i): add_requested.emit(MODELS[i][1], _free_name(MODELS[i][0]), {}))
   %Plot.pressed.connect(func(): plot_requested.emit(_names.filter(func(n): return not _unchecked.has(n)), KINDS[%Curve.selected]))


## Opens the dialog titled [param heading] with the runs of a network (see [method show_runs]).
func open(heading: String, runs: Dictionary) -> void:
   title = heading
   _unchecked = {}
   show_runs(runs)
   show_status("")
   popup_centered()


## Lists the runs of a network (from the worker: [code]runs[/code], each with name, task,
## validation scores, auc and average_precision for classification and, for runs made here,
## function and arguments): new runs are checked.
func show_runs(info: Dictionary) -> void:
   var runs: Array = info.get("runs", [])
   _names = runs.map(func(r): return r.name)
   show_error("")
   for child in %Runs.get_children():
      %Runs.remove_child(child)
      child.queue_free()
   for r in runs:
      var n: String = r.name
      var row := HBoxContainer.new()
      var box := CheckBox.new()
      box.text = n
      box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
      box.clip_text = true
      box.custom_minimum_size.x = 140
      box.button_pressed = not _unchecked.has(n)
      box.toggled.connect(func(on): if on: _unchecked.erase(n) else: _unchecked[n] = true)
      row.add_child(box)
      var scores := Label.new()
      var s: Dictionary = r.scores
      if r.task == "classification":
         scores.text = "accuracy %.3f · F1 %.3f · AUC %.3f · AP %.3f" % [s.get("accuracy", NAN), s.get("f1_macro", NAN), r.get("auc", NAN), r.get("average_precision", NAN)]
         scores.tooltip_text = "Validation set: accuracy, macro F1, and the areas under the ROC and precision-recall curves (one-vs-rest, micro-averaged)"
      else:
         scores.text = "R² %.3f · MSE %s" % [s.get("r2", NAN), String.num(s.get("mse", NAN), 4)]
         scores.tooltip_text = "Validation set: coefficient of determination and mean squared error"
      scores.mouse_filter = Control.MOUSE_FILTER_PASS
      row.add_child(scores)
      if r.has("function"):
         var copy := Button.new()
         copy.text = "Copy…"
         copy.tooltip_text = "Trains %s again with the values of %s, to change some" % [r.function, n]
         var label := _label(r.function)
         copy.pressed.connect(func(): add_requested.emit(r.function, _free_name(label), r.arguments))
         row.add_child(copy)
      %Runs.add_child(row)
   %None.visible = _names.is_empty()
   %Plot.disabled = _names.is_empty()
   _fit()


## Shows what is going on (e.g. a model training), or hides the line with an empty text.
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
