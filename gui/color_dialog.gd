extends AcceptDialog
## Colors of the nodes, or of the edges: one color for all, or a ColorBrewer scale on an
## attribute (any attribute on any scale). Quantitative scales interpolate between color stops
## (a continuous gradient, shown above the stops: values between two stops get the colors in
## between), placed at quantiles (default), evenly spaced or at natural breaks of the values
## (computed by the worker), on a linear or logarithmic scale; their values and colors can be
## edited. Qualitative scales give every distinct value the next color of the palette (colors
## repeat only when there are more values than colors), and any value can get another color; Clear makes all values
## [constant CLEARED], to color them one by one. Changes apply as they are made; Cancel restores
## the colors the dialog opened with. The dialog edits a color specification (see
## [code]colors[/code] in the worker).

## Emitted (after a short pause, so that dragging a color sends one change) with the new
## specification, and with the original one on Cancel.
signal changed(spec: Dictionary)

## Default palettes of quantitative and qualitative scales.
const DEFAULT_PALETTES := {"quantitative": "YlOrRd", "qualitative": "Set1"}
## Placements of the stops, in the order of the Placement menu ("custom": edited by hand).
const METHODS := ["quantiles", "equal", "natural", "custom"]
## Color of all the values of a qualitative scale after Clear.
const CLEARED := "#999999"
## Most rows shown in the scroll area before it scrolls, in pixels of height.
const ROWS_HEIGHT := 300

## ColorBrewer palettes: kind (sequential, diverging, qualitative), name, number of classes,
## colors.
static var palettes := {}
static var _icons := {}

var _spec := {}
var _original := {}
var _attributes := []
var _default := Color.WHITE
var _breaks := Callable()
var _building := false


func _ready() -> void:
   if palettes.is_empty():
      palettes = JSON.parse_string(FileAccess.get_file_as_string("res://colorbrewer.json"))
   add_cancel_button("Cancel")
   canceled.connect(func(): %Debounce.stop(); changed.emit(_original.duplicate(true)))
   %By.item_selected.connect(func(_i): _on_attribute())
   %Scale.item_selected.connect(func(i): _spec.scale = ["quantitative", "qualitative"][i]; _scale_defaults(); _emit())
   %Palette.item_selected.connect(func(_i): _on_palette())
   %Classes.value_changed.connect(func(_v): if not _building: _recolor(); _place())
   %Method.item_selected.connect(func(i): _spec.method = METHODS[i]; _place())
   %Log.toggled.connect(_on_log)
   %Auto.pressed.connect(_automatic)
   %Clear.pressed.connect(func(): _spec.palette_name = ""; _spec.palette = [CLEARED]; _spec.overrides = []; _show(); _emit())
   %Single.color_changed.connect(func(c): _spec.color = "#" + c.to_html(false); _emit())
   %Missing.color_changed.connect(func(c): _spec.missing = "#" + c.to_html(false); _emit())
   %Debounce.timeout.connect(func(): changed.emit(_spec.duplicate(true)))


## Opens the dialog titled [param heading] on a specification, with the [param attributes] that
## can color (from the worker), the default single color, and [param breaks], a function
## (attribute, method, k, log, done) that asks the worker for the stop values and calls done with
## them.
func open(heading: String, spec: Dictionary, attributes: Array, default_color: Color, breaks: Callable) -> void:
   title = heading
   _spec = spec.duplicate(true)
   _original = spec.duplicate(true)
   _attributes = attributes
   _default = default_color
   _breaks = breaks
   _show()
   reset_size()
   popup_centered()


## Description of the chosen attribute, or an empty dictionary.
func _info() -> Dictionary:
   for a in _attributes:
      if a.name == _spec.get("attribute"):
         return a
   return {}


func _kind_of(palette: String) -> String:
   for kind in ["sequential", "diverging", "qualitative"]:
      if palettes[kind].has(palette):
         return kind
   return ""


## Class counts of a palette, sorted.
func _sizes(palette: String) -> Array:
   var keys: Array = (palettes[_kind_of(palette)][palette] as Dictionary).keys().map(func(s): return int(s))
   keys.sort()
   return keys


## Colors of a palette with [param k] classes (the largest available if [param k] is 0).
func _colors(palette: String, k := 0) -> Array:
   var sizes := _sizes(palette)
   var n: int = sizes.back() if k == 0 else clampi(k, sizes.front(), sizes.back())
   return palettes[_kind_of(palette)][palette][str(n)]


## Strip of the colors of a palette, for its menu item.
func _icon(palette: String) -> Texture2D:
   if not _icons.has(palette):
      var colors := _colors(palette)
      var image := Image.create(8 * colors.size(), 12, false, Image.FORMAT_RGB8)
      for i in colors.size():
         image.fill_rect(Rect2i(8 * i, 0, 8, 12), Color(colors[i]))
      _icons[palette] = ImageTexture.create_from_image(image)
   return _icons[palette]


func _on_attribute() -> void:
   if %By.selected == 0:
      _spec = {"attribute": null, "color": _spec.get("color", "#" + _default.to_html(false))}
   else:
      var a: Dictionary = _attributes[%By.selected - 1]
      _spec = {"attribute": a.name, "scale": "quantitative" if a.numeric else "qualitative", "missing": _spec.get("missing", "#cccccc")}
      _scale_defaults()
      return
   _show()
   _emit()


## Default palette of the scale; stops at quantiles of the values (asked to the worker), or the
## palette in order for qualitative scales.
func _scale_defaults() -> void:
   _spec.palette_name = DEFAULT_PALETTES[_spec.scale]
   if _spec.scale == "quantitative":
      _spec.erase("overrides")
      _spec.method = "quantiles"
      _spec.log = false
      _spec.stops = []
      var a := _info()
      for c in _colors(_spec.palette_name, 5):
         _spec.stops.append([a.min, c])
      _place()
   else:
      _spec.erase("stops")
      _spec.palette = _colors(_spec.palette_name)
      _spec.overrides = []
      _show()
      _emit()


## Qualitative colors from the palette again (Set1 after Clear), without the colors set by hand.
func _automatic() -> void:
   if _spec.get("palette_name", "").is_empty():
      _spec.palette_name = DEFAULT_PALETTES.qualitative
   _spec.palette = _colors(_spec.palette_name)
   _spec.overrides = []
   _show()
   _emit()


func _on_palette() -> void:
   _spec.palette_name = %Palette.get_item_text(%Palette.selected)
   if _spec.scale == "quantitative":
      _recolor()
   else:
      _spec.palette = _colors(_spec.palette_name)
   _show()
   _emit()


## Colors of the stops from the palette, with the number of stops chosen.
func _recolor() -> void:
   var colors := _colors(_spec.palette_name, int(%Classes.value))
   var stops := []
   for i in colors.size():
      stops.append([_spec.stops[mini(i, _spec.stops.size() - 1)][0], colors[i]])
   _spec.stops = stops


## Places the stops with the chosen method (asked to the worker), then shows and emits them.
func _place() -> void:
   if _spec.get("method", "quantiles") == "custom":
      _spec.method = "quantiles"
   var attribute = _spec.attribute
   _breaks.call(attribute, _spec.method, _spec.stops.size(), _spec.get("log", false), func(values):
      if _spec.get("attribute") != attribute or values.size() != _spec.stops.size():
         return
      for i in values.size():
         _spec.stops[i][0] = values[i]
      _show()
      _emit())
   _show()


## Logarithmic scale on or off. Quantiles are the same values on both scales (the logarithm keeps
## the order), so turning it on with quantiles places the stops evenly spaced on the logarithm of
## the values instead; natural breaks and evenly spaced stops are placed again on the new scale.
func _on_log(on: bool) -> void:
   if _building:
      return
   _spec.log = on
   if on and _spec.get("method", "quantiles") == "quantiles":
      _spec.method = "equal"
   if _spec.method == "custom":
      _emit()
   else:
      _place()


## Rebuilds the controls from the specification.
func _show() -> void:
   _building = true
   %By.clear()
   %By.add_item("Single color")
   for a in _attributes:
      %By.add_item(a.name)
   var a := _info()
   %By.select(0 if a.is_empty() else _attributes.find(a) + 1)
   %Single.color = Color(_spec.get("color", "#" + _default.to_html(false)))
   for n in [%SingleLabel, %Single]:
      n.visible = a.is_empty()
   var quantitative: bool = not a.is_empty() and _spec.scale == "quantitative"
   for n in [%ScaleLabel, %Scale, %PaletteLabel, %Palette, %Scroll, %AutoRow]:
      n.visible = not a.is_empty()
   for n in [%ClassesLabel, %Classes, %MethodLabel, %Method, %LogLabel, %Log]:
      n.visible = quantitative
   %Auto.visible = not a.is_empty() and not quantitative
   %Clear.visible = %Auto.visible
   for child in %Rows.get_children():
      child.queue_free()
   %Note.text = ""
   if not a.is_empty():
      %Scale.select(0 if quantitative else 1)
      %Palette.clear()
      for kind in (["sequential", "diverging"] if quantitative else ["qualitative"]):
         for name in palettes[kind]:
            %Palette.add_icon_item(_icon(name), name)
            if name == _spec.palette_name:
               %Palette.select(%Palette.item_count - 1)
      if _spec.get("palette_name", "").is_empty():
         %Palette.select(-1)   # cleared: no palette
      %Missing.color = Color(_spec.get("missing", "#cccccc"))
      var notes := PackedStringArray()
      if quantitative:
         var sizes := _sizes(_spec.palette_name)
         %Classes.min_value = sizes.front()
         %Classes.max_value = sizes.back()
         %Classes.value = _spec.stops.size()
         %Method.select(METHODS.find(_spec.get("method", "quantiles")))
         %Log.button_pressed = _spec.get("log", false)
         %Rows.add_child(load("res://visual_panel.gd").gradient_rect(_spec.stops, Vector2(320, 14)))
         for i in _spec.stops.size():
            _stop_row(i)
         if not a.numeric:
            notes.append("Values are numbered in their order, from 0 (%s) to %d (%s)." % [a["values"][0] if a["values"].size() > 0 else "", a.count - 1, a["values"][a.count - 1] if a.count <= a["values"].size() else "…"])
         if _spec.get("log", false) and not a.positive:
            notes.append("Not all values are positive: they are shifted above 0 by the gap between the two smallest values before the logarithm.")
      else:
         var palette: Array = _spec.palette
         var overrides := {}
         for pair in _spec.get("overrides", []):
            overrides[pair[0]] = pair[1]
         for i in a["values"].size():
            var value: String = a["values"][i]
            _value_row(value, overrides.get(value, palette[i % palette.size()]))
         if a.count > palette.size() and not _spec.get("palette_name", "").is_empty():
            notes.append("%d values and %d colors in the palette: colors repeat." % [a.count, palette.size()])
         if a.count > a["values"].size():
            notes.append("The first %d values are listed; the others get the colors of the palette too." % a["values"].size())
      %Note.text = "\n".join(notes)
   %Note.visible = not %Note.text.is_empty()
   _building = false
   await get_tree().process_frame
   %Scroll.custom_minimum_size.y = mini(int(%Rows.get_combined_minimum_size().y), ROWS_HEIGHT)
   reset_size()


## Row of stop [param i]: its value (editable), the value it stands for (qualitative attributes),
## and its color.
func _stop_row(i: int) -> void:
   var row := HBoxContainer.new()
   var edit := LineEdit.new()
   edit.text = String.num(float(_spec.stops[i][0]), 6)
   edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
   edit.text_submitted.connect(func(_t): _edited())
   edit.focus_exited.connect(_edited)
   row.add_child(edit)
   var a := _info()
   if not a.numeric:
      var label := Label.new()
      var k := roundi(float(_spec.stops[i][0]))
      label.text = a["values"][k] if k >= 0 and k < a["values"].size() else ""
      label.custom_minimum_size = Vector2(100, 0)
      label.clip_text = true
      row.add_child(label)
   var swatch := SwatchButton.new()
   swatch.color = Color(_spec.stops[i][1])
   swatch.color_changed.connect(func(c): _spec.stops[i][1] = "#" + c.to_html(false); _emit())
   row.add_child(swatch)
   %Rows.add_child(row)


## Row of a qualitative value and its color.
func _value_row(value: String, color: String) -> void:
   var row := HBoxContainer.new()
   var label := Label.new()
   label.text = value
   label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
   label.clip_text = true
   row.add_child(label)
   var swatch := SwatchButton.new()
   swatch.color = Color(color)
   swatch.color_changed.connect(func(c): _override(value, "#" + c.to_html(false)))
   row.add_child(swatch)
   %Rows.add_child(row)


func _override(value: String, color: String) -> void:
   var overrides: Array = _spec.get("overrides", []).filter(func(p): return p[0] != value)
   overrides.append([value, color])
   _spec.overrides = overrides
   _emit()


## Reads the stop values typed by the user (placement: custom).
func _edited() -> void:
   if _building:
      return
   var values := []
   for row in %Rows.get_children():
      if row.is_queued_for_deletion():
         continue
      var t: String = (row.get_child(0) as LineEdit).text
      if not t.is_valid_float():
         return
      values.append(t.to_float())
   var changed_any := false
   for i in mini(values.size(), _spec.stops.size()):
      if not is_equal_approx(values[i], float(_spec.stops[i][0])):
         _spec.stops[i][0] = values[i]
         changed_any = true
   if changed_any:
      _spec.method = "custom"
      %Method.select(METHODS.find("custom"))
      _emit()


func _emit() -> void:
   if not _building:
      %Debounce.start()
