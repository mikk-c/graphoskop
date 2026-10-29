class_name SwatchButton
extends Button
## Button showing a color; clicking it opens the color picker (swatch_picker.gd).

## Emitted when the user picks a color.
signal color_changed(color: Color)

## The color shown.
@export var color := Color.WHITE:
   set(c):
      color = c
      queue_redraw()


func _init() -> void:
   custom_minimum_size = Vector2(44, 24)
   pressed.connect(func(): load("res://swatch_picker.gd").open(self))


func _draw() -> void:
   draw_rect(Rect2(Vector2(4, 4), size - Vector2(8, 8)), color)


## Called by the picker with the picked color.
func set_picked(c: Color) -> void:
   color = c
   color_changed.emit(c)
