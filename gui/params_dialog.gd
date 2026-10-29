extends ConfirmationDialog
## Parameters of a generator (Generators menu), a layout (Layout menu) or an analysis (Analysis
## menu), set with as little typing as possible: every parameter gets the control its widget asks
## for (from the worker): a switch for true / false values and for the GPU, a menu for values
## from a list and for one attribute name, a menu of check boxes for several attribute names (in
## the order they are checked), a slider with a number for values between two bounds, a number
## with arrows for other numbers, and fields for texts (names, written without quotes), node ids,
## numbers that can be left empty, and lists (numbers separated by commas, rows of matrices by
## semicolons, pairs as [code]key: value[/code]). Controls start at the preset or default value
## (fields without one are empty, with a hint); the description of a parameter is its tooltip.
## Only the values that differ from the defaults are passed, as Python source code.
##
## The dialog stays open while the function runs: if a value is not valid, or the function fails,
## the error is shown under the controls, which keep what was entered; it closes when the
## function has run. It is as large as its controls, scrolling only beyond [constant MAX_HEIGHT]
## of the screen height.

## Largest height of the controls, as a fraction of the screen height.
const MAX_HEIGHT := 0.6
## Width of fields and menus, in pixels.
const FIELD_WIDTH := 260
## Width of the slider of a value between bounds, in pixels.
const SLIDER_WIDTH := 160
## Largest value of numbers without an upper bound.
const LARGEST := 1e12

var _params := []
var _values: Array[Callable] = []
var _on_accept := Callable()
var _run := 0


func _ready() -> void:
   confirmed.connect(_on_confirmed)
   canceled.connect(func(): _run += 1; get_ok_button().disabled = false)


## Opens the dialog titled [param heading], with the signature [param sig] of a function (from the
## worker: params with name, default, hint, doc and widget), [param presets] (parameter name to
## Python source), the text of the OK button, the [param attributes] of the current network (from
## the worker: nodes and edges, or empty) and [param on_accept], called with the arguments
## (parameter name to Python source code) and a function to call when the run is over, with the
## error message, or an empty one if the function ran.
func open(heading: String, sig: Dictionary, presets: Dictionary, ok_text: String, attributes: Dictionary, on_accept: Callable) -> void:
   title = heading
   ok_button_text = ok_text
   get_ok_button().disabled = false
   _params = sig.params
   _values = []
   _on_accept = on_accept
   _run += 1
   %Error.visible = false
   for child in %Grid.get_children():
      %Grid.remove_child(child)
      child.queue_free()
   for p in _params:
      var name := Label.new()
      name.text = p.name
      name.tooltip_text = p.doc
      name.mouse_filter = Control.MOUSE_FILTER_PASS
      %Grid.add_child(name)
      var start: String = presets.get(p.name, p.default)
      var w: Dictionary = p.get("widget", {"kind": "text"})
      var control: Control
      match w.kind:
         "bool":
            control = _switch("", start == "True", ["False", "True"])
         "device":
            var gpu := _switch("GPU", start.contains("cuda"), ["\"cpu\"", "\"cuda\""])
            gpu.disabled = not w.get("gpu", false) and not start.contains("cuda")
            control = gpu
         "choice":
            control = _menu(w.options, w.options.map(func(o): return p.hint if o == "" else o.trim_prefix("\"").trim_suffix("\"")), start)
         "attribute":
            var names: Array = attributes.get(w.choices, []).map(func(a): return a.name)
            if w.multiple:
               control = _checks(names, start, p.hint)
            else:
               var options := [""]
               options.append_array(names.map(func(a): return JSON.stringify(a)))
               var labels := [p.hint]
               labels.append_array(names)
               control = _menu(options, labels, start)
         "range":
            control = _range(w, start)
         "int", "number":
            control = _number(w, start)
         "string":
            control = _field(_unquoted(start), p.hint, func(t): return JSON.stringify(t) if not t.is_empty() else "")
         "node":
            control = _field(_unquoted(start), p.hint, func(t): return t if w.get("numeric", false) and t.is_valid_int() else (JSON.stringify(t) if not t.is_empty() else ""))
         "optional":
            control = _field(start, p.hint, func(t): return t if t.is_empty() or t.is_valid_float() else "!" + t)
         "list":
            control = _field(start, p.hint, _list_source)
         _:
            control = _field(start, p.hint, func(t): return t)
      control.tooltip_text = p.doc
      %Grid.add_child(control)
      %Grid.add_child(Control.new())
   var content: Vector2 = %Grid.get_combined_minimum_size()
   var screen := DisplayServer.screen_get_usable_rect(DisplayServer.window_get_current_screen()).size
   %Scroll.custom_minimum_size = Vector2(content.x, minf(content.y, MAX_HEIGHT * screen.y))
   %Error.custom_minimum_size.x = content.x
   _fit()
   popup_centered()


func _fit() -> void:
   size = Vector2i.ZERO
   reset_size()


## A switch, on if [param on], whose value is [param sources] (off, on).
func _switch(text: String, on: bool, sources: Array) -> CheckButton:
   var b := CheckButton.new()
   b.text = text
   b.button_pressed = on
   _values.append(func(): return sources[1] if b.button_pressed else sources[0])
   return b


## A menu of [param labels], whose value is the matching item of [param options] (Python source),
## showing [param start] (or the first item).
func _menu(options: Array, labels: Array, start: String) -> OptionButton:
   var b := OptionButton.new()
   b.custom_minimum_size.x = FIELD_WIDTH
   for label in labels:
      b.add_item(label)
   b.select(maxi(options.find(start), 0))
   _values.append(func(): return options[b.selected])
   return b


## Menu of check boxes for several attribute [param names], kept in the order they are checked;
## the button shows them (or [param hint] for none). Its value is the list of names, or empty.
func _checks(names: Array, start: String, hint: String) -> MenuButton:
   var menu := MenuButton.new()
   menu.flat = false
   menu.custom_minimum_size.x = FIELD_WIDTH
   menu.alignment = HORIZONTAL_ALIGNMENT_LEFT
   menu.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
   var chosen: Array = []
   var parsed = JSON.parse_string(start) if not start.is_empty() else null
   for n in (parsed if parsed is Array else ([parsed] if parsed is String else [])):
      if names.has(n):
         chosen.append(n)
   var popup := menu.get_popup()
   popup.hide_on_checkable_item_selection = false
   for n in names:
      popup.add_check_item(n)
      popup.set_item_checked(popup.item_count - 1, chosen.has(n))
   var label := func(): menu.text = ", ".join(chosen) if not chosen.is_empty() else hint
   label.call()
   popup.index_pressed.connect(func(i):
      var n: String = names[i]
      if chosen.has(n):
         chosen.erase(n)
      else:
         chosen.append(n)
      popup.set_item_checked(i, chosen.has(n))
      label.call())
   _values.append(func(): return "" if chosen.is_empty() else "[%s]" % ", ".join(chosen.map(func(n): return JSON.stringify(n))))
   return menu


## A field showing [param start] or the [param hint] when empty; [param source] turns its text
## into Python source code (a text starting with "!" is not valid).
func _field(start: String, hint: String, source: Callable) -> LineEdit:
   var edit := LineEdit.new()
   edit.custom_minimum_size.x = FIELD_WIDTH
   edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
   edit.text = start
   edit.placeholder_text = hint
   edit.clear_button_enabled = true
   _values.append(func(): return source.call(edit.text.strip_edges()))
   return edit


## A number with arrows for widgets "int" (min) and "number" (min, step), without an upper bound.
func _number(w: Dictionary, start: String) -> SpinBox:
   var spin := SpinBox.new()
   spin.min_value = w.min
   spin.max_value = LARGEST
   spin.step = 1.0 if w.kind == "int" else w.step
   spin.allow_greater = true
   spin.custom_minimum_size.x = FIELD_WIDTH
   spin.value = float(start) if start.is_valid_float() else w.min
   _values.append(func(): return _number_text(spin.value, w.kind == "int"))
   return spin


## A slider and a number, linked, for widgets "range" (min, max, step).
func _range(w: Dictionary, start: String) -> HBoxContainer:
   var box := HBoxContainer.new()
   var slider := HSlider.new()
   slider.custom_minimum_size.x = SLIDER_WIDTH
   slider.size_flags_vertical = Control.SIZE_SHRINK_CENTER
   var spin := SpinBox.new()
   for r: Range in [slider, spin]:
      r.min_value = w.min
      r.max_value = w.max
      r.step = w.step
   spin.custom_minimum_size.x = FIELD_WIDTH - SLIDER_WIDTH - 4
   slider.share(spin)
   spin.value = float(start) if start.is_valid_float() else clampf(0.0, w.min, w.max)
   box.add_child(slider)
   box.add_child(spin)
   _values.append(func(): return _number_text(spin.value, w.step >= 1.0))
   return box


## Python source of a number: an integer (also for whole values), or a decimal number without
## trailing zeros.
static func _number_text(v: float, integer: bool) -> String:
   if integer or is_equal_approx(v, roundf(v)):
      return str(roundi(v))
   return String.num(v, 6)


## A text without the quotes of a Python string.
static func _unquoted(t: String) -> String:
   if t.length() >= 2 and t.begins_with("\"") and t.ends_with("\""):
      var parsed = JSON.parse_string(t)
      return parsed if parsed is String else t
   return t


## Python source of a list as typed: a number, or Python source starting with a bracket or
## containing [code]*[/code] or [code]([/code], is kept; otherwise pairs
## [code]key: value, ...[/code] become a dictionary, rows separated by semicolons a matrix and
## values separated by commas (or spaces) a list.
static func _list_source(t: String) -> String:
   if t.is_empty() or t.is_valid_float() or t[0] in ["[", "{", "("] or t.contains("*") or t.contains("("):
      return t
   if t.contains(":"):
      return "{%s}" % t
   var rows := Array(t.split(";", false)).map(func(r): return "[%s]" % ", ".join(Array(r.replace(",", " ").split(" ", false))))
   return rows[0] if rows.size() == 1 else "[%s]" % ", ".join(rows)


## Whether two Python sources are the same value (numbers compared as numbers).
static func _same(a: String, b: String) -> bool:
   if a.is_valid_float() and b.is_valid_float():
      return is_equal_approx(float(a), float(b))
   return a == b


## Shows an error under the controls (keeping them as they are), or hides it.
func _show_error(text: String) -> void:
   %Error.text = text
   %Error.visible = not text.is_empty()
   get_ok_button().disabled = false
   _fit()


func _on_confirmed() -> void:
   var arguments := {}
   for i in _params.size():
      var t: String = _values[i].call()
      if t.begins_with("!"):
         _show_error("%s: not a number: %s" % [_params[i].name, t.substr(1)])
         return
      if not t.is_empty() and not _same(t, _params[i].default):
         arguments[_params[i].name] = t
   _run += 1
   var run := _run
   get_ok_button().disabled = true
   %Error.visible = false
   _on_accept.call(arguments, func(error: String):
      if run != _run:   # closed or run again meanwhile
         return
      if error.is_empty():
         hide()
      else:
         _show_error(error))
