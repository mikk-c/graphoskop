extends TabContainer
## Visual elements panel, with three tabs: Nodes (color, size and opacity), Edges (color,
## thickness and opacity) and Network (background). Each row sums up the current choice; its Edit button opens the dialog
## that changes it (color_dialog.gd, size_dialog.gd). The panel edits the style of the drawing of
## the current network: [code]{"nodes": color, "edges": color, "node_size": size, "edge_width":
## size, "node_opacity": opacity, "edge_opacity": opacity, "background": "#rrggbb"}[/code] (see
## [code]default_style[/code] in the worker).

## Emitted when the colors or sizes of nodes or edges change, with the whole style.
signal style_changed(style: Dictionary)
## Emitted when the background color changes.
signal background_changed(color: Color)

## Function (attribute, kind, method, k, log, done) asking the worker for stop values (set by the
## main scene).
var breaks := Callable()
var _style := {}
var _attributes := {}
var _editing := ""


func _ready() -> void:
   %EditNodeColor.pressed.connect(func(): _edit_color("nodes", "Node colors", Color("#e41a1c")))
   %EditEdgeColor.pressed.connect(func(): _edit_color("edges", "Edge colors", Color("#999999")))
   %EditNodeSize.pressed.connect(func(): _edit_size("node_size", "Node size", "nodes", "area"))
   %EditEdgeWidth.pressed.connect(func(): _edit_size("edge_width", "Edge thickness", "edges", "width"))
   %EditNodeOpacity.pressed.connect(func(): _edit_size("node_opacity", "Node opacity", "nodes", "opacity"))
   %EditEdgeOpacity.pressed.connect(func(): _edit_size("edge_opacity", "Edge opacity", "edges", "opacity"))
   %ColorDialog.changed.connect(_changed)
   %SizeDialog.changed.connect(_changed)
   %Background.color_changed.connect(func(c): _style.background = "#" + c.to_html(false); background_changed.emit(c))


func _edit_color(key: String, heading: String, default_color: Color) -> void:
   _editing = key
   var kind := key
   %ColorDialog.open(heading, _style.get(key, {}), _attributes.get(key, []), default_color, func(attribute, method, k, log, done): breaks.call(attribute, kind, method, k, log, done))


func _edit_size(key: String, heading: String, kind: String, values: String) -> void:
   _editing = key
   %SizeDialog.open(heading, _style.get(key, {}), _attributes.get(kind, []), values)


func _changed(spec: Dictionary) -> void:
   _style[_editing] = spec
   _summarize()
   style_changed.emit(_style.duplicate(true))


## Shows the style of a network, with its node and edge [param attributes] (from the worker).
func show_style(style: Dictionary, attributes: Dictionary) -> void:
   _style = style.duplicate(true)
   _attributes = attributes
   %Background.color = Color(style.get("background", "#ffffff"))
   _summarize()


func _summarize() -> void:
   _color_summary(%NodeColor, _style.get("nodes", {}))
   _color_summary(%EdgeColor, _style.get("edges", {}))
   %NodeSize.text = _size_summary(_style.get("node_size", {}))
   %EdgeWidth.text = _size_summary(_style.get("edge_width", {}))
   %NodeOpacity.text = _size_summary(_style.get("node_opacity", {}), "opacity")
   %EdgeOpacity.text = _size_summary(_style.get("edge_opacity", {}), "opacity")


## Swatch or color ramp of a color specification, and what it colors by.
func _color_summary(box: HBoxContainer, spec: Dictionary) -> void:
   for child in box.get_children():
      child.queue_free()
   var colors := []
   var text := "single color"
   if spec.get("attribute") == null:
      colors = [spec.get("color", "#000000")]
   elif spec.get("scale") == "quantitative":
      colors = spec.get("stops", []).map(func(s): return s[1])
      text = "%s%s" % [spec.attribute, ", log" if spec.get("log", false) else ""]
   else:
      colors = spec.get("palette", []).duplicate()
      text = "%s, qualitative" % spec.attribute
   if spec.get("attribute") != null and spec.get("scale") == "quantitative":
      box.add_child(gradient_rect(spec.get("stops", []), Vector2(48, 16)))
   else:
      var ramp := HBoxContainer.new()
      ramp.add_theme_constant_override("separation", 0)
      for c in colors.slice(0, 12):
         var r := ColorRect.new()
         r.color = Color(c)
         r.custom_minimum_size = Vector2(maxf(48.0 / maxf(colors.size(), 1), 4), 16)
         r.size_flags_vertical = Control.SIZE_SHRINK_CENTER
         ramp.add_child(r)
      box.add_child(ramp)
   var label := Label.new()
   label.text = text
   label.clip_text = true
   label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
   box.add_child(label)


## Text of a size (or, with [param key] "opacity", opacity) specification.
func _size_summary(spec: Dictionary, key := "size") -> String:
   var unit := "" if key == "opacity" else " ×"
   if spec.get("attribute") == null:
      return "%s%s" % [String.num(spec.get(key, 1.0), 2), unit]
   return "%s: %s–%s%s%s" % [spec.attribute, String.num(spec.get("min", 0.5), 2), String.num(spec.get("max", 2.0), 2), unit, ", log" if spec.get("log", false) else ""]


## Continuous color ramp of quantitative [param stops] ([value, color], sorted by value), each
## color at the place of its value, interpolated in between as nodes and edges are colored.
static func gradient_rect(stops: Array, min_size: Vector2) -> TextureRect:
   var g := Gradient.new()
   var sorted := stops.duplicate()
   sorted.sort_custom(func(a, b): return float(a[0]) < float(b[0]))
   var lo: float = float(sorted[0][0]) if sorted.size() > 0 else 0.0
   var hi: float = float(sorted[-1][0]) if sorted.size() > 0 else 1.0
   var offsets := PackedFloat32Array()
   var colors := PackedColorArray()
   for i in sorted.size():
      offsets.append((float(sorted[i][0]) - lo) / (hi - lo) if hi > lo else float(i) / maxf(sorted.size() - 1, 1))
      colors.append(Color(sorted[i][1]))
   g.offsets = offsets
   g.colors = colors
   var tex := GradientTexture1D.new()
   tex.gradient = g
   var r := TextureRect.new()
   r.texture = tex
   r.stretch_mode = TextureRect.STRETCH_SCALE
   r.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
   r.custom_minimum_size = min_size
   r.size_flags_vertical = Control.SIZE_SHRINK_CENTER
   return r
