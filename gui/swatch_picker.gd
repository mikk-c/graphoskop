extends PopupPanel
## Color picker shared by all the color buttons ([code]SwatchButton[/code]), in three tabs:
## ColorBrewer (the colors of all its palettes, sequential, diverging and qualitative, as
## swatches), Swatches (the color space as a grid of swatches: hues across, from light tints to
## dark shades down, then grays) and Custom (the continuous color space). Clicking a swatch picks
## its color and closes the picker; the custom picker applies its color as it changes.

## Side of a swatch, in pixels.
const SWATCH := 20
## Hues of the swatch grid, in degrees.
const HUES := 18
## Saturation and value of the rows of the swatch grid: tints, pure colors, shades.
const ROWS := [[0.12, 1.0], [0.25, 1.0], [0.4, 1.0], [0.6, 1.0], [0.8, 1.0], [1.0, 1.0], [1.0, 0.8], [1.0, 0.62], [1.0, 0.45], [1.0, 0.3]]

static var _shared: Node
var _on_pick := Callable()


## Opens the picker under [param button] (a SwatchButton), which receives the picked color. The
## picker becomes a child of the window of the button (e.g. a dialog), so that it shows above it.
static func open(button: Control) -> void:
   if _shared == null:
      _shared = load("res://swatch_picker.tscn").instantiate()
      button.get_window().add_child(_shared)
   elif _shared.get_parent() != button.get_window():
      _shared.reparent(button.get_window())
   _shared.show_for(button)


func _ready() -> void:
   var palettes: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://colorbrewer.json"))
   for kind in ["sequential", "diverging", "qualitative"]:
      var title := Label.new()
      title.text = kind.capitalize()
      %Brewer.add_child(title)
      for name in palettes[kind]:
         var sizes: Dictionary = palettes[kind][name]
         var biggest: String = str(sizes.keys().map(func(k): return int(k)).max())
         var row := HBoxContainer.new()
         row.add_theme_constant_override("separation", 2)
         var label := Label.new()
         label.text = name
         label.custom_minimum_size = Vector2(70, 0)
         row.add_child(label)
         for c in sizes[biggest]:
            row.add_child(_swatch(Color(c)))
         %Brewer.add_child(row)
   for r in ROWS:
      for h in HUES:
         %Space.add_child(_swatch(Color.from_hsv(float(h) / HUES, r[0], r[1])))
   for g in HUES:
      %Space.add_child(_swatch(Color.from_hsv(0, 0, 1.0 - float(g) / (HUES - 1))))
   %Custom.color_changed.connect(func(c): if _on_pick.is_valid(): _on_pick.call(c))


## One swatch: a square of [param color], with its code as tooltip, that picks it when clicked.
func _swatch(color: Color) -> Control:
   var s := ColorRect.new()
   s.color = color
   s.custom_minimum_size = Vector2(SWATCH, SWATCH)
   s.tooltip_text = "#" + color.to_html(false)
   s.mouse_filter = Control.MOUSE_FILTER_STOP
   s.gui_input.connect(func(e): if e is InputEventMouseButton and e.pressed and e.button_index == MOUSE_BUTTON_LEFT: _pick(color))
   return s


func _pick(color: Color) -> void:
   if _on_pick.is_valid():
      _on_pick.call(color)
   hide()


## Shows the picker below [param button], with its color in the custom picker.
func show_for(button: Control) -> void:
   _on_pick = button.set_picked
   %Custom.color = button.color
   reset_size()
   var at := Vector2i(button.get_screen_position()) + Vector2i(0, int(button.size.y))
   popup(Rect2i(at, size))
