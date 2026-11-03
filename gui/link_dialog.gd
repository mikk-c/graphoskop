extends AcceptDialog
## Analysis › Link prediction: runs link prediction methods on the network of the current tab and
## compares them. All the runs score the pairs of one validation set (hidden edges, the positives,
## and unconnected pairs, the negatives; gsk.link_prediction_split, made with the defaults at the
## first run, or with Validation set…, which drops the runs so far) from the training network
## without the hidden edges, and are evaluated (area under the ROC curve, AUC, and average
## precision, AP). Add method opens the parameter dialog of a method (its run gets a name, e.g.
## "Adamic-Adar 1"); the runs are listed with their AUC and AP, check boxes and Copy…, which opens
## the parameter dialog of the same method with the values of that run. Plot draws the ROC or the
## precision-recall curves (Curve menu; the plot window can switch too) of the checked runs in one
## plot. Signed networks also have Signs: how well social balance and social status predict the
## signs of the hidden edges. The dialog is not exclusive: it stays open to add more runs, with the
## parameter dialog and the plot above it.

## Emitted by Add method or Copy…, with the function of the method, the name proposed for its run,
## and the values of the parameters (Python source code, by name; empty for Add method).
signal add_requested(function: String, name: String, values: Dictionary)
## Emitted by Validation set….
signal split_requested
## Emitted by Plot, with the names of the checked runs and the curve ("roc" or "pr").
signal plot_requested(names: Array, kind: String)
## Emitted by Signs.
signal signs_requested

## Methods: label (also the start of the names of their runs) and function of the library; an
## empty label is a separator.
const METHODS := [
   ["Common neighbors", "common_neighbors_lp"],
   ["Jaccard", "common_neighbors_jaccard_lp"],
   ["Adamic-Adar", "adamic_adar_lp"],
   ["Resource allocation", "resource_allocation_lp"],
   ["CAR index", "car_index_lp"],
   ["Mutual information", "mutual_information_index_lp"],
   ["Preferential attachment", "preferential_attachment_lp"],
   ["", ""],
   ["Local random walk", "local_random_walk_lp"],
   ["SimRank", "simrank_lp"],
   ["Vertex similarity", "vertex_similarity_lp"],
   ["", ""],
   ["Hierarchical random graph", "hierarchical_random_graph_lp"],
   ["Block model", "block_model_reliability_lp"],
   ["", ""],
   ["Embedding", "embedding_lp"],
   ["Association rules", "association_rules_lp"],
   ["", ""],
   ["Ensemble", "ensemble_link_prediction"],
]

var _names := []
var _unchecked := {}


func _ready() -> void:
   var popup: PopupMenu = %Add.get_popup()
   for i in METHODS.size():
      if METHODS[i][0].is_empty():
         popup.add_separator("", 100000 + i)
      else:
         popup.add_item(METHODS[i][0] + "…", i)
   popup.id_pressed.connect(func(i): add_requested.emit(METHODS[i][1], _free_name(METHODS[i][0]), {}))
   %ChangeSplit.pressed.connect(func(): split_requested.emit())
   %Signs.pressed.connect(func(): signs_requested.emit())
   %Plot.pressed.connect(func(): plot_requested.emit(_names.filter(func(n): return not _unchecked.has(n)), ["roc", "pr"][%Curve.selected]))


## Opens the dialog titled [param heading] with the runs of a network (see [method show_runs]).
func open(heading: String, runs: Dictionary) -> void:
   title = heading
   _unchecked = {}
   show_runs(runs)
   show_status("")
   popup_centered()


## Shows the validation set and lists the runs of a network (from the worker: [code]runs[/code],
## each with name, auc, average_precision and, for runs made here, function and arguments;
## [code]split[/code], [code]positives[/code], [code]negatives[/code], [code]signed[/code]): new
## runs are checked.
func show_runs(info: Dictionary) -> void:
   var runs: Array = info.get("runs", [])
   _names = runs.map(func(r): return r.name)
   show_error("")
   var split = info.get("split")
   if split == null:
      %Split.text = "Validation set: made at the first run (10% of the edges hidden, as many unconnected pairs, seed 1)."
   else:
      %Split.text = "Validation set: %d hidden edges and %d unconnected pairs%s." % [info.get("positives", 0), info.get("negatives", 0), "" if split.is_empty() else " (%s)" % ", ".join(split.keys().map(func(k): return "%s %s" % [k, split[k]]))]
   %Signs.visible = info.get("signed", false)
   %Signs.disabled = split == null
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
      box.custom_minimum_size.x = 160
      box.button_pressed = not _unchecked.has(n)
      box.toggled.connect(func(on): if on: _unchecked.erase(n) else: _unchecked[n] = true)
      row.add_child(box)
      var scores := Label.new()
      scores.text = "AUC %.3f · AP %.3f" % [r.auc, r.average_precision]
      scores.tooltip_text = "Area under the ROC curve, and average precision (area under the precision-recall curve)"
      scores.mouse_filter = Control.MOUSE_FILTER_PASS
      row.add_child(scores)
      if r.has("function"):
         var copy := Button.new()
         copy.text = "Copy…"
         copy.tooltip_text = "Runs %s again with the values of %s, to change some" % [r.function, n]
         var label := _label(r.function)
         copy.pressed.connect(func(): add_requested.emit(r.function, _free_name(label), r.arguments))
         row.add_child(copy)
      %Runs.add_child(row)
   %None.visible = _names.is_empty()
   %Plot.disabled = _names.is_empty()
   _fit()


## Shows what is going on (e.g. a method running), or hides the line with an empty text.
func show_status(text: String) -> void:
   %Status.text = text
   %Status.visible = not text.is_empty()
   _fit()


## Shows an error under the runs, or hides it.
func show_error(text: String) -> void:
   %Error.text = text
   %Error.visible = not text.is_empty()
   _fit()


## The label of the method [param function] (see [constant METHODS]).
func _label(function: String) -> String:
   for m in METHODS:
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
