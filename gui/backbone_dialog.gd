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

## Emitted when Components, Extend to the left or Extend to the right is turned on or off: the plot
## is to be drawn again with the [param components] and the [param left] and [param right] tails.
signal plot_changed(components: bool, left: bool, right: bool)
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

var _box := [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
var _range := [0.0, 1.0]
var _scale := {"kind": "linear"}
var _threshold := 0.0
var _busy := false


func _ready() -> void:
   %Overlay.draw.connect(_draw_line)
   %Slider.value_changed.connect(func(u): if not _busy: _choose(_inverse(_f(_box[4]) + u * (_f(_box[5]) - _f(_box[4])))))
   %Value.value_changed.connect(func(v): if not _busy: _choose(v))
   for button: BaseButton in [%Components, %Left, %Right]:
      button.toggled.connect(func(_on): plot_changed.emit(%Components.button_pressed, %Left.button_pressed, %Right.button_pressed))
   %Debounce.timeout.connect(func(): threshold_changed.emit(_threshold))
   confirmed.connect(func(): extract_requested.emit(_threshold, options()))
   %ReconnectComponents.toggled.connect(func(on):
      %ReconnectIsolates.disabled = on
      if on:
         %ReconnectIsolates.button_pressed = true)


## The options of the extraction, as checked.
func options() -> Dictionary:
   return {"keep_isolates": %KeepIsolates.button_pressed, "reconnect_components": %ReconnectComponents.button_pressed, "reconnect_isolates": %ReconnectIsolates.button_pressed}


## Opens the dialog titled [param heading] with the plot [param png], the [param box] of its axes,
## the [param scale] of its threshold axis and whether it has [param tails] on the left and on the
## right (from the worker), the [param whole] range of the thresholds, the [param found]
## thresholds ([criterion, threshold or null]) and the threshold to start from.
func open(heading: String, png: PackedByteArray, box: Array, scale: Dictionary, tails: Array, whole: Array, found: Array, start: float) -> void:
   title = heading
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
