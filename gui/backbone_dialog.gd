extends AcceptDialog
## Backbone of a network (Edit › Backbone), after its edges are scored: a plot of the backbones
## from keeping every edge to keeping none (the percentage of edges kept and of nodes that are not
## isolated, and, with Components, the number of components on a second axis), with the
## remarkable thresholds as dashed lines (first isolate, first component break, max separation).
## The threshold is chosen with a continuous slider (or typed), drawn as a line on the plot, with
## the size of its backbone under it; the buttons of the remarkable thresholds snap to them.
## A handful of edges far below (above) all the others would stretch the threshold axis: such tails
## are left out of the plot, unless Extend to the left (right) is on; choosing a threshold in a tail
## turns it on. Extract backbone opens the backbone at the chosen threshold in a new tab, with the
## options checked under the plot (as [code]extract_backbone[/code] has them): isolated nodes
## kept, components reconnected (which reconnects the isolated nodes too: that box is then checked
## and locked), isolated nodes reconnected. The options stay as set for the next backbones. The
## sizes under the slider are those of the backbone without these options.
##
## With [b]Advanced[/b] on (one switch for all the dialogs, see [method AppSettings.advanced]),
## rows under the plot change it as in the plot window: the color of every element, the scale
## (automatic, linear or logarithmic), label and range of the threshold axis, the labels and range
## of the y axes, and the font size. A range of the threshold axis tests the thresholds again in
## it, evenly spaced (worker command [code]backbone_plot[/code]), and the slider spans it: the
## backbones in a narrow range can be told apart.
##
## The dialog also shows the probabilities of the edges of a corrected network (Edit › Error
## correction): every pair observed in the network or sampled as an edge, with the probability
## that it is one. A threshold keeps the pairs above it, so it both removes observed edges (likely
## spurious) and adds pairs that were not observed (likely missing): the plot shows the
## percentage of the observed edges kept and the pairs added (as a percentage of the observed
## edges), the text under the slider counts both, and Extract network opens the network at that
## threshold.

## Emitted when Components, Extend to the left or Extend to the right is turned on or off, or (after
## a short pause) when an option of the plot changes: the plot is to be drawn again with the
## [param components], the [param left] and [param right] tails, the [param colors] of its
## elements (key to "#rrggbb") and its [param options] (font_size, xlabel, ylabel, y2label, xscale,
## xmin, xmax, ymin, ymax; ranges null when automatic).
signal plot_changed(components: bool, left: bool, right: bool, colors: Dictionary, options: Dictionary)
## Emitted (after a short pause) when the threshold changes.
signal threshold_changed(threshold: float)
## Emitted by Extract backbone, with the threshold and the [param options] of
## [code]extract_backbone[/code] (keep_isolates, reconnect_components, reconnect_isolates: bool).
signal extract_requested(threshold: float, options: Dictionary)

## Labels of the remarkable thresholds of [code]find_bb_thresholds[/code], and their colors in the
## plot (as in the worker, [code]BACKBONE_COLORS[/code]).
const FOUND := {"first_isolate": "First isolate", "first_component_break": "First component break", "max_separation": "Max separation"}
## Color of the chosen threshold.
const LINE_COLOR := Color(0, 0, 0)
## Elements of the plot whose colors can be changed: key, label and default color (as in the
## worker, [code]BACKBONE_COLORS[/code]).
const ELEMENTS := [["edges_pct", "% edges kept", "#377eb8"], ["nodes_pct", "% nodes not isolated", "#e41a1c"], ["components", "Components", "#999999"], ["first_isolate", "First isolate", "#4daf4a"], ["first_component_break", "First component break", "#984ea3"], ["max_separation", "Max separation", "#ff7f00"], ["added", "Pairs added", "#a65628"]]
## Scales of the threshold axis, in the order of its menu.
const SCALES := ["automatic", "linear", "logarithmic"]

var _box := [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
var _range := [0.0, 1.0]
var _scale := {"kind": "linear"}
var _threshold := 0.0
var _busy := false
var _correction := false
var _colors := {}
var _advanced := CheckButton.new()
var _rows := VBoxContainer.new()
var _swatches := HFlowContainer.new()
var _xscale := OptionButton.new()
var _edits := {}
var _font := SpinBox.new()
var _note := Label.new()
var _replot := Timer.new()


func _ready() -> void:
   %Overlay.draw.connect(_draw_line)
   %Slider.value_changed.connect(func(u): if not _busy: _choose(_inverse(_f(_box[4]) + u * (_f(_box[5]) - _f(_box[4])))))
   %Value.value_changed.connect(func(v): if not _busy: _choose(v))
   for button: BaseButton in [%Components, %Left, %Right]:
      button.toggled.connect(func(_on): _emit_plot())
   _plot_rows()
   %Debounce.timeout.connect(func(): threshold_changed.emit(_threshold))
   confirmed.connect(func(): extract_requested.emit(_threshold, options()))
   %ReconnectComponents.toggled.connect(func(on):
      %ReconnectIsolates.disabled = on
      if on:
         %ReconnectIsolates.button_pressed = true)


## Builds the rows of the options of the plot (shown with Advanced).
func _plot_rows() -> void:
   _advanced.text = "Advanced"
   _advanced.tooltip_text = "Show the colors, axes and font of the plot"
   $Box/Options.add_child(_advanced)
   _advanced.toggled.connect(func(on):
      AppSettings.set_advanced(on)
      _show_rows())
   $Box.add_child(_rows)
   $Box.move_child(_rows, $Box/Options.get_index() + 1)
   _rows.add_child(_swatches)
   for row in [["x axis", ["xlabel", "xmin", "xmax"]], ["y axis", ["ylabel", "ymin", "ymax"]], ["Right axis", ["y2label"]]]:
      var box := HBoxContainer.new()
      var label := Label.new()
      label.text = row[0]
      label.custom_minimum_size = Vector2(70, 0)
      box.add_child(label)
      if row[0] == "x axis":
         for s in SCALES:
            _xscale.add_item(s)
         _xscale.tooltip_text = "Scale of the threshold axis (automatic: logarithmic when the thresholds have a long tail)"
         _xscale.item_selected.connect(func(_i): _replot.start())
         box.add_child(_xscale)
      for key in row[1]:
         var edit := LineEdit.new()
         if key.ends_with("label"):
            edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
            edit.tooltip_text = "Label of the axis"
         else:
            var from := Label.new()
            from.text = "from" if key.ends_with("min") else "to"
            box.add_child(from)
            edit.custom_minimum_size = Vector2(90, 0)
            edit.placeholder_text = "automatic"
            edit.tooltip_text = "Smallest value shown (empty: automatic)" if key.ends_with("min") else "Largest value shown (empty: automatic)"
            if key.begins_with("x"):
               edit.tooltip_text += "; the thresholds in the range are tested again"
         edit.text_changed.connect(func(_t): _replot.start())
         box.add_child(edit)
         _edits[key] = edit
      _rows.add_child(box)
   var font_row := HBoxContainer.new()
   var font_label := Label.new()
   font_label.text = "Font size"
   font_label.custom_minimum_size = Vector2(70, 0)
   font_row.add_child(font_label)
   _font.min_value = 4
   _font.max_value = 72
   _font.value_changed.connect(func(_v): _replot.start())
   font_row.add_child(_font)
   _rows.add_child(font_row)
   _note.modulate = Color(1, 1, 1, 0.7)
   _note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
   _note.custom_minimum_size = Vector2(760, 0)
   _rows.add_child(_note)
   _replot.one_shot = true
   _replot.wait_time = 0.4
   _replot.timeout.connect(_emit_plot)
   add_child(_replot)


## Default colors, axes and font of a new plot.
func _reset_plot() -> void:
   for child in _swatches.get_children():
      _swatches.remove_child(child)
      child.queue_free()
   _colors = {}
   for e in ELEMENTS:
      if e[0] == "added" and not _correction:   # only for error corrections
         continue
      var box := HBoxContainer.new()
      var label := Label.new()
      label.text = e[1] if not (_correction and e[0] == "edges_pct") else "% observed edges kept"
      box.add_child(label)
      var swatch := SwatchButton.new()
      swatch.color = Color(e[2])
      var key: String = e[0]
      swatch.color_changed.connect(func(c): _colors[key] = "#" + c.to_html(false); _replot.start())
      box.add_child(swatch)
      _swatches.add_child(box)
   _xscale.select(0)
   for key in _edits:
      _edits[key].text = ""
   _edits.xlabel.text = "threshold" if not _correction else "probability"
   _edits.ylabel.text = "% of the network"
   _edits.y2label.text = "components"
   _font.set_value_no_signal(12)
   _note.text = ""
   _replot.stop()


## Shows the rows of the plot options only with [b]Advanced[/b] on, and sizes the dialog.
func _show_rows() -> void:
   _rows.visible = _advanced.button_pressed
   size = Vector2i.ZERO
   reset_size()


func _emit_plot() -> void:
   var options := {"font_size": _font.value, "xscale": SCALES[_xscale.selected]}
   for key in _edits:
      var t: String = _edits[key].text.strip_edges()
      if key.ends_with("label"):
         options[key] = t
      elif t.is_empty() or t.is_valid_float():   # half-typed numbers wait
         options[key] = float(t) if not t.is_empty() else null
      else:
         return
   plot_changed.emit(%Components.button_pressed, %Left.button_pressed, %Right.button_pressed, _colors.duplicate(), options)


## Shows why the plot could not be drawn with the options (empty: it was).
func show_note(text: String) -> void:
   _note.text = text


## The options of the extraction, as checked.
func options() -> Dictionary:
   return {"keep_isolates": %KeepIsolates.button_pressed, "reconnect_components": %ReconnectComponents.button_pressed, "reconnect_isolates": %ReconnectIsolates.button_pressed}


## Opens the dialog titled [param heading] with the plot [param png], the [param box] of its axes,
## the [param scale] of its threshold axis and whether it has [param tails] on the left and on the
## right (from the worker), the [param whole] range of the thresholds, the [param found]
## thresholds ([criterion, threshold or null]) and the threshold to start from; with
## [param correction], the probabilities of the edges of a corrected network (Extract network).
func open(heading: String, png: PackedByteArray, box: Array, scale: Dictionary, tails: Array, whole: Array, found: Array, start: float, correction := false) -> void:
   title = heading
   _correction = correction
   ok_button_text = "Extract network" if correction else "Extract backbone"
   %Slider.tooltip_text = "Threshold of the probability: the pairs more likely to be edges are kept, observed or not" if correction else "Threshold of the backbone: the edges whose significance is above it are kept"
   _reset_plot()
   _advanced.set_pressed_no_signal(AppSettings.advanced())
   _rows.visible = _advanced.button_pressed
   _range = whole
   for button: BaseButton in [%Components, %Left, %Right]:
      button.set_pressed_no_signal(false)
   %Left.disabled = not tails[0]
   %Right.disabled = not tails[1]
   set_plot(png, box, scale)
   _busy = true
   %Value.min_value = whole[0]
   %Value.max_value = whole[1]
   _busy = false
   for child in %Snaps.get_children():
      %Snaps.remove_child(child)
      child.queue_free()
   for f in found:
      if not FOUND.has(f[0]):
         continue
      var b := Button.new()
      b.text = "%s: %s" % [FOUND[f[0]], "never" if f[1] == null else String.num(f[1], 4)]
      b.disabled = f[1] == null
      b.tooltip_text = "Sets the threshold to the %s" % FOUND[f[0]].to_lower()
      var t = f[1]
      b.pressed.connect(func(): _choose(t))
      %Snaps.add_child(b)
   %Stats.text = ""
   _choose(start)
   reset_size()
   popup_centered()


## Shows a new [param png] of the plot, with the [param box] of its axes and the [param scale] of
## its threshold axis.
func set_plot(png: PackedByteArray, box: Array, scale: Dictionary) -> void:
   var image := Image.new()
   image.load_png_from_buffer(png)
   %Plot.texture = ImageTexture.create_from_image(image)
   %Plot.custom_minimum_size = Vector2(image.get_size())
   _box = box
   _scale = scale
   _place()


## Shows the size of the backbone at the chosen threshold (from the worker).
func show_stats(stats: Dictionary) -> void:
   %Stats.text = "%.2f%% of the edges (%d), %.2f%% of the nodes not isolated (%d), %d components with more than one node" % [stats.edges_pct, stats.edges, stats.nodes_pct, stats.nodes, stats.components]
   if stats.has("observed"):   # error correction: what the threshold removes and adds
      var removed := int(stats.observed) - int(stats.observed_kept)
      %Stats.text = "%d of the %d observed edges kept (%d removed as likely spurious), %d pairs not observed added (likely missing): %d edges.\n%.2f%% of the nodes not isolated (%d), %d components with more than one node" % [int(stats.observed_kept), int(stats.observed), removed, int(stats.added), int(stats.edges), stats.nodes_pct, stats.nodes, stats.components]


## The chosen threshold.
func threshold() -> float:
   return _threshold


func _choose(t: float) -> void:
   _threshold = clampf(t, _range[0], _range[1])
   _place()
   %Stats.text = "…"
   %Debounce.start()
   if _threshold < _box[4] and not %Left.disabled and not %Left.button_pressed:   # in a tail: shown
      %Left.button_pressed = true
   elif _threshold > _box[5] and not %Right.disabled and not %Right.button_pressed:
      %Right.button_pressed = true


## The chosen threshold in the text box, on the slider and on the plot.
func _place() -> void:
   _busy = true
   %Value.value = _threshold
   var span := _f(_box[5]) - _f(_box[4])
   %Slider.value = clampf((_f(_threshold) - _f(_box[4])) / span, 0.0, 1.0) if span != 0 else 0.0
   _busy = false
   %Overlay.queue_redraw()


## Position of a threshold on the axis, as matplotlib's scales place it: linear, or symmetric
## logarithmic (linear within linthresh of 0).
func _f(x: float) -> float:
   if _scale.get("kind") != "symlog":
      return x
   var c: float = _scale.linthresh
   var l := 1.0 / (1.0 - 0.1)
   return x * l if absf(x) <= c else signf(x) * c * (l + log(absf(x) / c) / log(10.0))


func _inverse(y: float) -> float:
   if _scale.get("kind") != "symlog":
      return y
   var c: float = _scale.linthresh
   var l := 1.0 / (1.0 - 0.1)
   return y / l if absf(y) <= c * l else signf(y) * c * pow(10.0, absf(y) / c - l)


## The chosen threshold as a vertical line across the axes of the plot.
func _draw_line() -> void:
   var span := _f(_box[5]) - _f(_box[4])
   if span == 0 or _threshold < _box[4] or _threshold > _box[5]:
      return
   var x: float = _box[0] + (_f(_threshold) - _f(_box[4])) / span * (_box[2] - _box[0])
   %Overlay.draw_line(Vector2(x, _box[1]), Vector2(x, _box[3]), LINE_COLOR, 2.0)
