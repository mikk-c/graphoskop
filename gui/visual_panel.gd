extends TabContainer
## Visual elements panel, with three tabs: Nodes (color, size, opacity, position: the layout,
## or numeric node attributes for x and y, position_dialog.gd, label: none, the default, or
## the node ids or the values of a node attribute, chosen in the menu of its Edit button, and label size: one
## size, or sizes from a node attribute, size_dialog.gd), Edges (color, thickness and
## opacity) and Network (background, rotation of the view, scale of the layout, and Copy › From
## tab…: all the visual elements of another tab). The scale
## multiplies the distances between the nodes on screen (a tenth to a hundred times those of the
## layout, on a logarithmic slider) without changing their size on screen: it blows up the layout,
## so that crowded nodes come apart; Reset puts it back to 1. (The mouse wheel zooms the view,
## nodes included, and does not touch the scale.) Each row sums up the current choice; its Edit button opens the dialog
## that changes it (color_dialog.gd, size_dialog.gd). The panel edits the style of the drawing of
## the current network: [code]{"nodes": color, "edges": color, "node_size": size, "edge_width":
## size, "node_opacity": opacity, "edge_opacity": opacity, "position": {"x": name, "y": name},
## "labels": {"attribute": name, "ids": bool}, "label_size": size, "node_chart": chart,
## "background": "#rrggbb"}[/code] (see [code]default_style[/code] in the worker). Nodes › Chart
## turns the nodes into pie charts (its dialog, made here): none, the shares of numeric node
## attributes (check boxes, a slice each), or the values of a qualitative attribute (equal slices
## of the values of each node, e.g. overlapping communities), in the colors of a qualitative
## ColorBrewer palette, then a color button per slice (each attribute, or each value, those of nodes
## with several split): [code]{"attributes": [names], "colors": {name: color}, "palette": [colors],
## "palette_name": name}[/code] or [code]{"attribute": name, "colors": {value: color}, ...}[/code]
## ([code]chart_slices[/code] in the worker; values not listed take the palette). Nodes › Shape
## and Edges › Line style (dialogs made here) give one shape or style for all
## ([code]{"value": name}[/code]) or one per value of a qualitative attribute
## ([code]{"attribute": name, "map": {value: name}, "other": name}[/code]; [code]category_codes[/code]
## in the worker); Border color and Border width edit [code]node_border[/code] (a color) and
## [code]node_border_width[/code] (1 × is a tenth of the radius, 0 none); Label color edits
## [code]label_color[/code] as the node colors (one color, or colors from a node attribute;
## automatic, black or white by the background, until edited: "").

## Emitted when the colors or sizes of nodes or edges change, with the whole style.
signal style_changed(style: Dictionary)
## Emitted when the background color changes.
signal background_changed(color: Color)
## Emitted when the rotation of the view changes (degrees, counterclockwise).
signal turn_changed(degrees: float)
## Emitted when the scale of the layout changes (1: the distances of the layout).
signal scale_changed(factor: float)
## Emitted when Network › Copy › From tab… is chosen, with the position of the other tab among the
## names given by [member tab_names].
signal copy_requested(index: int)

## Function (attribute, kind, method, k, log, done) asking the worker for stop values (set by the
## main scene).
var breaks := Callable()
## Function giving the names of the other tabs, whose visual elements can be copied (set by the
## main scene).
var tab_names := Callable()
var _style := {}
var _attributes := {}
var _editing := ""
## Node shapes and edge line styles (NODE_SHAPES and EDGE_STYLES of the worker, in code order).
const NODE_SHAPES := ["circle", "triangle", "square", "diamond", "pentagon", "hexagon", "heptagon", "octagon", "star"]
const EDGE_STYLES := ["solid", "dash", "dash-dot", "dots", "parallel", "sine", "zigzag"]
## Most distinct values of a numeric attribute offered for shapes and line styles.
const MAX_CATEGORIES := 20
## Groups of the rows of the Nodes and Edges tabs: heading, then the rows (name, value node,
## edit control, as unique names of the scene). Each row is a button: a click edits it.
const GROUPS := {
   "Nodes": [
      ["Fill", [["Color", "NodeColor", "EditNodeColor"], ["Opacity", "NodeOpacity", "EditNodeOpacity"], ["Chart", "NodeChart", "EditNodeChart"]]],
      ["Shape", [["Shape", "NodeShape", "EditNodeShape"], ["Size", "NodeSize", "EditNodeSize"]]],
      ["Border", [["Color", "NodeBorder", "EditNodeBorder"], ["Width", "NodeBorderWidth", "EditNodeBorderWidth"]]],
      ["Label", [["Text", "NodeLabel", "EditNodeLabel"], ["Size", "LabelSize", "EditLabelSize"], ["Color", "LabelColor", "EditLabelColor"]]],
      ["Placement", [["Position", "NodePosition", "EditNodePosition"]]],
   ],
   "Edges": [
      ["Line", [["Color", "EdgeColor", "EditEdgeColor"], ["Thickness", "EdgeWidth", "EditEdgeWidth"], ["Opacity", "EdgeOpacity", "EditEdgeOpacity"], ["Style", "EdgeStyle", "EditEdgeStyle"]]],
   ],
}
var _chart := ConfirmationDialog.new()
var _chart_kind := OptionButton.new()
var _chart_numeric := VBoxContainer.new()
var _chart_qualitative := OptionButton.new()
var _chart_palette := OptionButton.new()
var _chart_rows_box := GridContainer.new()
var _chart_colors := {}


func _ready() -> void:
   %EditNodeColor.pressed.connect(func(): _edit_color("nodes", "Node colors", Color("#e41a1c")))
   %EditEdgeColor.pressed.connect(func(): _edit_color("edges", "Edge colors", Color("#999999")))
   %EditNodeSize.pressed.connect(func(): _edit_size("node_size", "Node size", "nodes", "area"))
   %EditEdgeWidth.pressed.connect(func(): _edit_size("edge_width", "Edge thickness", "edges", "width"))
   %EditNodeOpacity.pressed.connect(func(): _edit_size("node_opacity", "Node opacity", "nodes", "opacity"))
   %EditEdgeOpacity.pressed.connect(func(): _edit_size("edge_opacity", "Edge opacity", "edges", "opacity"))
   %EditLabelSize.pressed.connect(func(): _edit_size("label_size", "Label size", "nodes", "label"))
   _build_chart_dialog()
   %EditNodeChart.pressed.connect(_edit_chart)
   %EditNodeShape.pressed.connect(func(): _edit_category("node_shape", "Node shapes", NODE_SHAPES, "nodes"))
   %EditEdgeStyle.pressed.connect(func(): _edit_category("edge_style", "Edge line styles", EDGE_STYLES, "edges"))
   %EditNodeBorder.pressed.connect(func(): _edit_color("node_border", "Node border colors", Color("#000000"), "nodes"))
   %EditNodeBorderWidth.pressed.connect(func(): _edit_size("node_border_width", "Node border width", "nodes", "border"))
   %EditLabelColor.pressed.connect(func(): _edit_color("label_color", "Label colors", Color.BLACK, "nodes"))
   %ColorDialog.changed.connect(_changed)
   %SizeDialog.changed.connect(_changed)
   %EditNodePosition.pressed.connect(func(): _editing = "position"; %PositionDialog.open(_style.get("position", {}), _attributes.get("nodes", [])))
   %PositionDialog.changed.connect(_changed)
   var labels: PopupMenu = %EditNodeLabel.get_popup()
   %EditNodeLabel.about_to_popup.connect(func():
      labels.clear()
      labels.add_item("none")
      labels.add_item("node id")
      labels.set_item_tooltip(1, "The id of every node")
      labels.add_separator()
      for a in _attributes.get("nodes", []):
         labels.add_item(a.name))
   labels.index_pressed.connect(func(i):
      _editing = "labels"
      if i <= 1:
         _changed({"attribute": null, "ids": i == 1})
      else:
         _changed({"attribute": labels.get_item_text(i)}))
   %Background.color_changed.connect(func(c): _style.background = "#" + c.to_html(false); background_changed.emit(c))
   %TurnSlider.share(%Turn)
   %Turn.value_changed.connect(func(v): turn_changed.emit(v))
   %ScaleSlider.value_changed.connect(func(v): _scale(pow(10.0, v)))
   %Scale.value_changed.connect(_scale)
   %ResetScale.pressed.connect(func(): _scale(1.0))
   var copies: PopupMenu = %CopyFrom.get_popup()
   %CopyFrom.about_to_popup.connect(func():
      copies.clear()
      var names: Array = tab_names.call() if tab_names.is_valid() else []
      for i in names.size():
         copies.add_item(names[i], i)
      if names.is_empty():
         copies.add_item("No other tab", 100000)
         copies.set_item_disabled(0, true))
   copies.id_pressed.connect(func(id): if id < 100000: copy_requested.emit(id))
   _group_rows()   # last: the rows move into their groups


## The dialog of Nodes › Chart: the kind of chart, the numeric attributes (check boxes) or the
## qualitative one (menu), and the palette.
func _build_chart_dialog() -> void:
   _chart.title = "Node charts"
   _chart.ok_button_text = "Apply"
   var box := VBoxContainer.new()
   for k in ["None", "Pie of numeric attributes", "Pie of the values of a qualitative attribute"]:
      _chart_kind.add_item(k)
   _chart_kind.item_selected.connect(func(_i): _chart_show())
   _chart_qualitative.item_selected.connect(func(_i): _chart_rows())
   var palettes: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://colorbrewer.json")).qualitative
   for name in palettes:
      _chart_palette.add_item(name)
   _chart_palette.select(maxi(Array(palettes.keys()).find("Set1"), 0))
   _chart_palette.item_selected.connect(func(_i): _chart_colors.clear(); _chart_rows())   # the colors of the palette
   var scroll := ScrollContainer.new()
   scroll.custom_minimum_size = Vector2(320, 160)
   scroll.add_child(_chart_numeric)
   _chart_numeric.set_meta("scroll", scroll)
   var row := HBoxContainer.new()
   var label := Label.new()
   label.text = "Palette"
   row.add_child(label)
   row.add_child(_chart_palette)
   var colors := ScrollContainer.new()
   colors.custom_minimum_size = Vector2(320, 160)
   _chart_rows_box.columns = 2
   colors.add_child(_chart_rows_box)
   var heading := Label.new()
   heading.text = "Colors of the slices"
   for c in [_chart_kind, scroll, _chart_qualitative, row, heading, colors]:
      box.add_child(c)
   _chart.add_child(box)
   _chart.confirmed.connect(_apply_chart)
   add_child(_chart)


func _edit_chart() -> void:
   var spec: Dictionary = _style.get("node_chart", {})
   for c in _chart_numeric.get_children():
      c.queue_free()
   _chart_qualitative.clear()
   var chosen: Array = spec.get("attributes", [])
   for a in _attributes.get("nodes", []):
      if a.get("numeric", false):
         var check := CheckBox.new()
         check.text = a.name
         check.button_pressed = a.name in chosen
         check.toggled.connect(func(_on): _chart_rows())
         _chart_numeric.add_child(check)
      else:
         _chart_qualitative.add_item(a.name)
         if a.name == spec.get("attribute"):
            _chart_qualitative.select(_chart_qualitative.item_count - 1)
   _chart_kind.select(1 if not chosen.is_empty() else (2 if spec.get("attribute") != null else 0))
   if spec.has("palette_name"):
      for i in _chart_palette.item_count:
         if _chart_palette.get_item_text(i) == spec.palette_name:
            _chart_palette.select(i)
   _chart_colors = spec.get("colors", {}).duplicate()
   _chart_show()
   _chart.popup_centered()


func _chart_show() -> void:
   _chart_numeric.get_meta("scroll").visible = _chart_kind.selected == 1
   _chart_qualitative.visible = _chart_kind.selected == 2
   _chart_rows()


## The colors of the palette (its largest set).
func _palette_colors() -> Array:
   var palette: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://colorbrewer.json")).qualitative[_chart_palette.get_item_text(_chart_palette.selected)]
   return palette[str(Array(palette.keys()).map(func(k): return int(k)).max())]


## What the slices are: the checked numeric attributes, or the values of the qualitative one
## (those of nodes with several values split, numbers in numeric order).
func _chart_keys() -> Array:
   if _chart_kind.selected == 1:
      return _chart_numeric.get_children().filter(func(c): return c.button_pressed and not c.is_queued_for_deletion()).map(func(c): return c.text)
   if _chart_kind.selected == 2 and _chart_qualitative.selected >= 0:
      var name := _chart_qualitative.get_item_text(_chart_qualitative.selected)
      for a in _attributes.get("nodes", []):
         if a.name == name:
            var keys := []
            for v in a.get("values", []):
               for part in str(v).split(", "):
                  if not part in keys:
                     keys.append(part)
            keys.sort_custom(func(x, y): return float(x) < float(y) if x.is_valid_float() and y.is_valid_float() else x < y)
            return keys
   return []


## A color button per slice ([method _chart_keys]), in its color or the next of the palette.
func _chart_rows() -> void:
   for c in _chart_rows_box.get_children():
      _chart_rows_box.remove_child(c)
      c.queue_free()
   var palette := _palette_colors()
   var keys := _chart_keys()
   for i in keys.size():
      var label := Label.new()
      label.text = keys[i]
      var pick := ColorPickerButton.new()
      pick.custom_minimum_size = Vector2(60, 24)
      pick.edit_alpha = false
      pick.color = Color(_chart_colors.get(keys[i], palette[i % palette.size()]))
      var key: String = keys[i]
      pick.color_changed.connect(func(c): _chart_colors[key] = "#" + c.to_html(false))
      _chart_rows_box.add_child(label)
      _chart_rows_box.add_child(pick)
   _chart.reset_size()


func _apply_chart() -> void:
   var name := _chart_palette.get_item_text(_chart_palette.selected)
   var colors := {}
   var children := _chart_rows_box.get_children()
   for i in range(0, children.size(), 2):
      colors[children[i].text] = "#" + children[i + 1].color.to_html(false)
   var spec := {}
   if _chart_kind.selected == 1:
      var chosen := _chart_keys()
      if not chosen.is_empty():
         spec = {"attributes": chosen, "colors": colors, "palette": _palette_colors(), "palette_name": name}
   elif _chart_kind.selected == 2 and _chart_qualitative.selected >= 0:
      spec = {"attribute": _chart_qualitative.get_item_text(_chart_qualitative.selected), "colors": colors, "palette": _palette_colors(), "palette_name": name}
   _editing = "node_chart"
   _changed(spec)


func _scale(factor: float) -> void:
   show_scale(factor)
   scale_changed.emit(factor)


## Shows the scale of the layout [param factor] (without changing the view).
func show_scale(factor: float) -> void:
   %ScaleSlider.set_value_no_signal(clampf(log(factor) / log(10.0), -1.0, 2.0))
   %Scale.set_value_no_signal(factor)


## Shows the rotation of the view, [param degrees], without changing it.
func show_turn(degrees: float) -> void:
   %Turn.set_value_no_signal(degrees)
   %TurnSlider.set_value_no_signal(degrees)


func _edit_color(key: String, heading: String, default_color: Color, kind := "") -> void:
   _editing = key
   kind = key if kind.is_empty() else kind
   var spec = _style.get(key)
   if spec is String and not spec.is_empty():   # label colors of former versions: one color
      spec = {"attribute": null, "color": spec}
   if not spec is Dictionary or spec.is_empty():
      spec = {"attribute": null, "color": default_color.to_html(false)}
   %ColorDialog.open(heading, spec, _attributes.get(kind, []), default_color, func(attribute, method, k, log, done): breaks.call(attribute, kind, method, k, log, done))


## Rebuilds the Nodes and Edges tabs as the groups of [constant GROUPS]: a heading per group, then a
## button per row (its name and its value, the summary of the style); the edit controls of the scene
## stay, hidden, and a click on a row presses its control (or opens its menu, under the row).
func _group_rows() -> void:
   for tab in GROUPS:
      var grid: GridContainer = get_node(tab)
      var keep := {}
      for group in GROUPS[tab]:
         for r in group[1]:
            keep[r[1]] = get_node("%" + r[1])
            keep[r[2]] = get_node("%" + r[2])
      var hidden := Control.new()
      hidden.visible = false
      for c in grid.get_children():
         if not keep.values().has(c):
            grid.remove_child(c)
            c.queue_free()
      grid.columns = 1
      grid.add_child(hidden)
      for group in GROUPS[tab]:
         var heading := Label.new()
         heading.text = group[0]
         heading.theme_type_variation = "HeadingLabel"   # the heading color of the theme
         heading.size_flags_horizontal = Control.SIZE_EXPAND_FILL
         grid.add_child(heading)
         for r in group[1]:
            var value: Control = keep[r[1]]
            var edit: Control = keep[r[2]]
            var row := Button.new()
            row.custom_minimum_size = Vector2(0, 28)
            row.size_flags_horizontal = Control.SIZE_EXPAND_FILL
            row.tooltip_text = edit.tooltip_text
            row.focus_mode = Control.FOCUS_NONE
            var box := HBoxContainer.new()
            box.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
            box.offset_left = 8
            box.offset_right = -8
            box.mouse_filter = Control.MOUSE_FILTER_IGNORE
            var name := Label.new()
            name.text = r[0]
            name.custom_minimum_size.x = 80
            box.add_child(name)
            row.add_child(box)
            grid.add_child(row)
            value.reparent(box, false)
            value.owner = self   # its unique name again
            value.size_flags_horizontal = Control.SIZE_EXPAND_FILL
            value.size_flags_vertical = Control.SIZE_SHRINK_CENTER
            edit.reparent(hidden, false)
            edit.owner = self
            row.pressed.connect(func(): _press(edit, row))
   _ignore_mouse(self)


## Presses the edit control of a row, or opens its menu under the row (filled first, as a press
## of the menu button would: [signal MenuButton.about_to_popup]).
func _press(edit: Control, row: Control) -> void:
   if edit is MenuButton:
      edit.about_to_popup.emit()
      var popup: PopupMenu = edit.get_popup()
      popup.reset_size()
      popup.position = Vector2i(row.get_screen_position() + Vector2(0, row.size.y))
      popup.popup()
   else:
      edit.pressed.emit()


## Lets the clicks on the values of the rows reach the row buttons: every control inside a row
## (or, with [param inside], inside [param node]) ignores the mouse.
func _ignore_mouse(node: Node, inside := false) -> void:
   for c in node.get_children():
      if inside and c is Control:
         c.mouse_filter = Control.MOUSE_FILTER_IGNORE
      _ignore_mouse(c, inside or (c is Button and c.get_parent() is GridContainer))


## Node shapes or edge line styles ([param key] of the style, a choice among [param names]): one
## for all, or one per value of a qualitative attribute, or of a numeric one with few values
## (menus of the values the worker lists;
## the other values, and missing ones, take [b]Others[/b]). A dialog made here, each time.
func _edit_category(key: String, heading: String, names: Array, kind: String) -> void:
   var spec: Dictionary = _style.get(key, {})
   var dialog := ConfirmationDialog.new()
   dialog.title = heading
   dialog.ok_button_text = "Apply"
   var box := VBoxContainer.new()
   var by := OptionButton.new()
   by.add_item("One for all")
   var qualitative: Array = _attributes.get(kind, []).filter(func(a): return not a.get("numeric", false) or a.get("count", 0) <= mini(MAX_CATEGORIES, a.get("values", []).size()))   # numeric ones with few values too (e.g. communities)
   for a in qualitative:
      by.add_item("By " + a.name)
   var rows := VBoxContainer.new()
   var menu := func(chosen: String) -> OptionButton:
      var m := OptionButton.new()
      for n in names:
         m.add_item(n)
      m.select(maxi(names.find(chosen), 0))
      return m
   var fill := func():
      for c in rows.get_children():
         c.queue_free()
      var grid := GridContainer.new()
      grid.columns = 2
      rows.add_child(grid)
      if by.selected <= 0:
         var label := Label.new()
         label.text = "All"
         grid.add_child(label)
         grid.add_child(menu.call(spec.get("value", names[0])))
         return
      var a: Dictionary = qualitative[by.selected - 1]
      var mapping: Dictionary = spec.get("map", {}) if spec.get("attribute") == a.name else {}
      var values: Array = a.get("values", [])
      for i in values.size():
         var label := Label.new()
         label.text = str(values[i])
         grid.add_child(label)
         grid.add_child(menu.call(mapping.get(str(values[i]), names[i % names.size()])))
      var label := Label.new()
      label.text = "Others" if a.get("count", 0) > values.size() else "Missing"
      label.tooltip_text = "Values not listed, and missing values"
      grid.add_child(label)
      grid.add_child(menu.call(spec.get("other", names[0]) if spec.get("attribute") == a.name else names[0]))
   by.select(0)
   for i in qualitative.size():
      if qualitative[i].name == spec.get("attribute"):
         by.select(i + 1)
   by.item_selected.connect(func(_i): fill.call(); dialog.reset_size())
   fill.call()
   var scroll := ScrollContainer.new()
   scroll.custom_minimum_size = Vector2(300, 260)
   scroll.add_child(rows)
   box.add_child(by)
   box.add_child(scroll)
   dialog.add_child(box)
   dialog.confirmed.connect(func():
      var grid: GridContainer = rows.get_child(rows.get_child_count() - 1)
      var menus := grid.get_children().filter(func(c): return c is OptionButton)
      var texts := grid.get_children().filter(func(c): return c is Label)
      var out := {}
      if by.selected <= 0:
         out = {"value": names[menus[0].selected]} if menus[0].selected > 0 else {}
      else:
         var mapping := {}
         for i in menus.size() - 1:
            mapping[texts[i].text] = names[menus[i].selected]
         out = {"attribute": qualitative[by.selected - 1].name, "map": mapping, "other": names[menus[-1].selected]}
      _editing = key
      _changed(out))
   dialog.visibility_changed.connect(func(): if not dialog.visible: dialog.queue_free())
   add_child(dialog)
   dialog.popup_centered()


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
   var p: Dictionary = _style.get("position", {})
   var axes := []
   for axis in ["x", "y"]:
      if p.get(axis) != null:
         axes.append("%s: %s" % [axis, p[axis]])
   %NodePosition.text = "the layout" if axes.is_empty() else ", ".join(axes)
   %NodePosition.tooltip_text = %NodePosition.text
   var label = _style.get("labels", {}).get("attribute")
   %NodeLabel.text = str(label) if label != null else ("node id" if _style.get("labels", {}).get("ids", false) else "none")
   %LabelSize.text = _size_summary(_style.get("label_size", {}))
   var chart: Dictionary = _style.get("node_chart", {})
   %NodeChart.text = ("pie: " + ", ".join(chart.attributes)) if chart.has("attributes") else ("pie: " + str(chart.attribute) if chart.has("attribute") else "none")
   %NodeChart.tooltip_text = %NodeChart.text
   %NodeShape.text = _category_summary(_style.get("node_shape", {}), "circle")
   %EdgeStyle.text = _category_summary(_style.get("edge_style", {}), "solid")
   _color_summary(%NodeBorder, _style.get("node_border", {"attribute": null, "color": "#000000"}))
   %NodeBorderWidth.text = _size_summary(_style.get("node_border_width", {"attribute": null, "size": 0.0}))
   var ink = _style.get("label_color", "")
   if ink is String and not ink.is_empty():   # former versions: one color
      ink = {"attribute": null, "color": ink}
   if ink is Dictionary and not ink.is_empty():
      _color_summary(%LabelColor, ink)
   else:   # automatic: no swatch
      for child in %LabelColor.get_children():
         child.queue_free()
      var auto := Label.new()
      auto.text = "automatic (black or white)"
      auto.mouse_filter = Control.MOUSE_FILTER_IGNORE
      %LabelColor.add_child(auto)


## Text of a node shape or edge line style specification.
func _category_summary(spec: Dictionary, default: String) -> String:
   if spec.get("attribute") != null:
      return "by %s" % spec.attribute
   return str(spec.get("value", default))


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
   _ignore_mouse(box, true)


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
