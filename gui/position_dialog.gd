extends AcceptDialog
## Position of the nodes (visual elements panel, Nodes tab): the x and the y coordinates of every
## node can come from a numeric node attribute (e.g. longitude and latitude), instead of the
## layout, each axis on its own; the attributes override any layout, then and later, until the
## layout is chosen again. Changes apply as they are made; Cancel restores the positions the dialog
## opened with. The dialog edits the [code]position[/code] of the style:
## [code]{"x": name or null, "y": name or null}[/code] (see [code]positions[/code] in the worker).

## Emitted with the new position specification, and with the original one on Cancel.
signal changed(spec: Dictionary)

## Text of the first item of the menus.
const LAYOUT := "the layout"

var _original := {}
var _names := []


func _ready() -> void:
   add_cancel_button("Cancel")
   canceled.connect(func(): changed.emit(_original.duplicate()))
   for menu in [%X, %Y]:
      menu.item_selected.connect(func(_i): changed.emit(_spec()))


## Opens the dialog on a position [param spec], with the node [param attributes] of the network
## (from the worker; the numeric ones are offered).
func open(spec: Dictionary, attributes: Array) -> void:
   _original = {"x": spec.get("x"), "y": spec.get("y")}
   _names = attributes.filter(func(a): return a.numeric).map(func(a): return a.name)
   for pair in [[%X, "x"], [%Y, "y"]]:
      var menu: OptionButton = pair[0]
      menu.clear()
      menu.add_item(LAYOUT)
      for n in _names:
         menu.add_item(n)
      menu.select(_names.find(_original[pair[1]]) + 1)
   reset_size()
   popup_centered()


func _spec() -> Dictionary:
   return {"x": _names[%X.selected - 1] if %X.selected > 0 else null, "y": _names[%Y.selected - 1] if %Y.selected > 0 else null}
