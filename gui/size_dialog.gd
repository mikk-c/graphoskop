extends AcceptDialog
## Size of the nodes, thickness of the edges, or opacity of either: one value for all, or values
## given by an attribute (any attribute: qualitative values count by their order), from the
## smallest value for the smallest attribute value to the largest for the largest, on a linear or
## logarithmic scale. Sizes are relative to the default ones (1 ×); node areas, not diameters,
## grow linearly with the values, edge widths and opacities grow linearly. Opacities go from 0
## (transparent) to 1 (opaque), set with sliders. Changes apply as they are made; Cancel restores the values the
## dialog opened with. The dialog edits a size or opacity specification (see [code]sizes[/code]
## and [code]opacities[/code] in the worker).

## Emitted (after a short pause) with the new specification, and with the original one on Cancel.
signal changed(spec: Dictionary)

## Settings of the kinds of values: key of the single value, range and step of the numbers,
## defaults of the smallest and largest values, suffix, text of the first item of the menu.
const KINDS := {
   "area": {"key": "size", "max": 50.0, "step": 0.05, "low": 0.5, "high": 2.0, "suffix": "×", "single": "Single size"},
   "width": {"key": "size", "max": 50.0, "step": 0.05, "low": 0.5, "high": 3.0, "suffix": "×", "single": "Single size"},
   "opacity": {"key": "opacity", "max": 1.0, "step": 0.01, "low": 0.2, "high": 1.0, "suffix": "", "single": "Single opacity"},
}

var _spec := {}
var _original := {}
var _attributes := []
var _kind := "area"
var _building := false


func _ready() -> void:
   add_cancel_button("Cancel")
   canceled.connect(func(): %Debounce.stop(); changed.emit(_original.duplicate(true)))
   %By.item_selected.connect(func(_i): _on_attribute())
   %Single.value_changed.connect(func(v): if not _building: _spec[KINDS[_kind].key] = v; _emit())
   %Min.value_changed.connect(func(v): if not _building: _spec.min = v; _emit())
   %Max.value_changed.connect(func(v): if not _building: _spec.max = v; _emit())
   %Log.toggled.connect(func(b): if not _building: _spec.log = b; _emit())
   %Debounce.timeout.connect(func(): changed.emit(_spec.duplicate(true)))


## Opens the dialog titled [param heading] on a specification, with the [param attributes] that
## can give the values (from the worker); [param kind] is "area" (node sizes: areas proportional
## to the values interpolated), "width" (edge thickness) or "opacity".
func open(heading: String, spec: Dictionary, attributes: Array, kind: String) -> void:
   title = heading
   _spec = spec.duplicate(true)
   _original = spec.duplicate(true)
   _attributes = attributes
   _kind = kind
   _building = true   # setting the ranges changes the values: not an edit
   var k: Dictionary = KINDS[kind]
   for pair in [[%Single, %SingleSlider], [%Min, %MinSlider], [%Max, %MaxSlider]]:
      var spin: SpinBox = pair[0]
      spin.min_value = 0.0 if kind == "opacity" else 0.05
      spin.max_value = k.max
      spin.step = k.step
      spin.suffix = k.suffix
      pair[1].visible = kind == "opacity"
      if kind == "opacity":
         pair[1].share(spin)
   %SingleLabel.text = "Opacity" if kind == "opacity" else "Size"
   %ByLabel.text = "Opacity by" if kind == "opacity" else "Size by"
   _show()
   reset_size()
   popup_centered()


func _on_attribute() -> void:
   var k: Dictionary = KINDS[_kind]
   if %By.selected == 0:
      _spec = {"attribute": null, k.key: 1.0}
   else:
      _spec = {"attribute": _attributes[%By.selected - 1].name, "min": k.low, "max": k.high, "log": false}
   _show()
   _emit()


func _show() -> void:
   _building = true
   var k: Dictionary = KINDS[_kind]
   %By.clear()
   %By.add_item(k.single)
   var selected := 0
   for i in _attributes.size():
      %By.add_item(_attributes[i].name)
      if _attributes[i].name == _spec.get("attribute"):
         selected = i + 1
   %By.select(selected)
   var single := selected == 0
   for n in [%SingleLabel, %SingleRow]:
      n.visible = single
   for n in [%MinLabel, %MinRow, %MaxLabel, %MaxRow, %LogLabel, %Log]:
      n.visible = not single
   %Single.value = _spec.get(k.key, 1.0)
   %Min.value = _spec.get("min", k.low)
   %Max.value = _spec.get("max", k.high)
   %Log.button_pressed = _spec.get("log", false)
   var what: String = {"area": "Node areas", "width": "Edge widths", "opacity": "Opacities"}[_kind]
   if _kind == "opacity":
      %Note.text = "0 is transparent, 1 opaque." if single else "%s grow linearly from the smallest value to the largest; qualitative values count by their order, missing values get the smallest opacity." % what
   else:
      %Note.text = "Sizes are relative to the default size (1 ×)." if single else "%s grow linearly from the smallest value (smallest size) to the largest (largest size); qualitative values count by their order, missing values get the smallest size." % what
   _building = false


func _emit() -> void:
   if not _building:
      %Debounce.start()
