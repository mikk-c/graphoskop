extends ConfirmationDialog
## Parameters of a generator (Generators menu), a layout (Layout menu) or an analysis (Analysis
## menu), set with as little typing as possible: every parameter gets the control its widget asks
## for (from the worker): a switch for true / false values and for the GPU, a menu for values
## from a list and for one attribute name, a menu of check boxes for several attribute names (in
## the order they are checked; only numeric attributes where the function needs numbers; with
## patterns of names, see [method _checks]), a slider with a number for values between two bounds, a number
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
## Most attribute names listed one by one in a menu of check boxes: beyond, the names that belong
## to a pattern are left out (the pattern checks them all), and the others beyond this many too.
const MAX_CHECKS := 200
## Fewest attributes sharing a prefix that get a pattern of their own in a menu of check boxes.
const PATTERN_MIN := 3
## Characters that end the prefix of a pattern (the last one in a name).
const SEPARATORS := "_-.: "

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
            var names: Array = attributes.get(w.choices, []).filter(func(a): return a.numeric or not w.get("numeric", false)).map(func(a): return a.name)
            if w.multiple:
               var args := (", edges = True" if w.choices == "edges" else "") + (", numeric = True" if w.get("numeric", false) else "")
               control = _checks(names, start, p.hint, args if w.choices in ["nodes", "edges"] else "")
            else:
               var extra: Array = w.get("extra", [])
               var options := [w.get("empty", "")]   # the hint
               options.append_array(extra)
               options.append_array(names.map(func(a): return JSON.stringify(a)))
               var labels := [p.hint]
               labels.append_array(extra.map(func(o): return o.trim_prefix("\"").trim_suffix("\"")))
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
   b.fit_to_longest_item = false   # its width is not from its items (also: seconds to fill with thousands of them)
   b.clip_text = true
   for label in labels:
      b.add_item(label)
   b.select(maxi(options.find(start), 0))
   _values.append(func(): return options[b.selected])
   return b


## Menu of check boxes for several attribute [param names], kept in the order they are checked;
## the button shows them (or [param hint] for none). Its value is the list of names, or empty.
##
## With [param matching] (the keyword arguments of the worker's [code]matching[/code] after the
## patterns, possibly empty; [code]null[/code] for lists that are not attribute names), the menu
## also offers patterns of names: [code]*[/code] stands for any text and [code]?[/code] for one
## character. Every prefix shared by at least [constant PATTERN_MIN] names (up to the last of
## [constant SEPARATORS], or the letters before final digits) is a pattern at the top of the menu,
## e.g. [code]label_*[/code] with the number of names it checks; [b]Pattern…[/b] (with such
## patterns, or beyond [constant MAX_CHECKS] names) asks for any other. Beyond
## [constant MAX_CHECKS] names, those covered by a pattern are not listed one by one (ten
## thousand check boxes would stall the dialog), nor the others beyond that many. With a
## pattern checked, the value is a call of the worker's [code]matching[/code], e.g.
## [code]matching(net, "label_*", "age")[/code], so that the code does not list thousands of names.
func _checks(names: Array, start: String, hint: String, matching = null) -> MenuButton:
   var menu := MenuButton.new()
   menu.flat = false
   menu.custom_minimum_size.x = FIELD_WIDTH
   menu.alignment = HORIZONTAL_ALIGNMENT_LEFT
   menu.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
   var patterns: Array = _patterns(names) if matching != null else []
   var offer := matching != null and (not patterns.is_empty() or names.size() > MAX_CHECKS)   # Pattern…
   var chosen: Array = []
   for n in _start_names(start):
      if names.has(n) or (matching != null and _is_pattern(n)):
         chosen.append(n)
         if _is_pattern(n) and not patterns.has(n):
            patterns.append(n)
            offer = true
   var popup := menu.get_popup()
   popup.hide_on_checkable_item_selection = false
   var label := func(): menu.text = ", ".join(chosen) if not chosen.is_empty() else hint
   var entries: Array = []   # what each item checks: a name, a pattern, or "" (no check box)
   var fill := func():
      popup.clear()
      entries.clear()
      if offer:
         popup.add_item("Pattern…")
         popup.set_item_tooltip(0, "Check all the names matching a pattern: * stands for any text, ? for one character (e.g. label_* or gene_1??)")
         entries.append("")
         for pt in patterns:
            popup.add_check_item("%s (%d)" % [pt, _matches(names, pt)])
            popup.set_item_checked(popup.item_count - 1, chosen.has(pt))
            entries.append(pt)
         popup.add_separator()
         entries.append("")
      var listed := names if names.size() <= MAX_CHECKS else names.filter(func(n): return not patterns.any(func(pt): return n.match(pt)))
      for n in listed.slice(0, MAX_CHECKS):
         popup.add_check_item(n)
         popup.set_item_checked(popup.item_count - 1, chosen.has(n))
         entries.append(n)
      if listed.size() > MAX_CHECKS:
         popup.add_item("… %d more names: use Pattern…" % (listed.size() - MAX_CHECKS))
         popup.set_item_disabled(popup.item_count - 1, true)
         entries.append("")
   fill.call()
   label.call()
   popup.index_pressed.connect(func(i):
      var n: String = entries[i]
      if n.is_empty():
         if i == 0 and offer:
            _ask_pattern(names, func(pt):
               if not patterns.has(pt):
                  patterns.append(pt)
               if not chosen.has(pt):
                  chosen.append(pt)
               fill.call()
               label.call())
         return
      if chosen.has(n):
         chosen.erase(n)
      else:
         chosen.append(n)
      popup.set_item_checked(i, chosen.has(n))
      label.call())
   _values.append(func():
      if chosen.is_empty():
         return ""
      var quoted := chosen.map(func(n): return JSON.stringify(n if _is_pattern(n) else _escaped(n)) if chosen.any(_is_pattern) else JSON.stringify(n))
      return "matching(net, %s%s)" % [", ".join(quoted), matching] if chosen.any(_is_pattern) else "[%s]" % ", ".join(quoted))
   return menu


## The patterns of [param names] (see [method _checks]), in the order of their first name.
static func _patterns(names: Array) -> Array:
   var counts := {}
   for n: String in names:
      var pt := _prefix(n)
      if not pt.is_empty():
         counts[pt] = counts.get(pt, 0) + 1
   return counts.keys().filter(func(pt): return counts[pt] >= PATTERN_MIN)


## Pattern of a name: its text up to the last separator (or before its final digits), then
## [code]*[/code]; empty if it has none.
static func _prefix(n: String) -> String:
   for i in range(n.length() - 2, 0, -1):
      if SEPARATORS.contains(n[i]):
         return n.left(i + 1) + "*"
   var k := n.length()
   while k > 0 and n[k - 1] >= "0" and n[k - 1] <= "9":
      k -= 1
   return n.left(k) + "*" if k > 0 and k < n.length() else ""


## Whether a checked entry is a pattern.
static func _is_pattern(n: String) -> bool:
   return n.contains("*") or n.contains("?")


## How many of [param names] match the pattern [param pt].
static func _matches(names: Array, pt: String) -> int:
   return names.reduce(func(c, n): return c + int(n.match(pt)), 0)


## A name written so that [code]fnmatch[/code] matches it alone ([code]*[/code], [code]?[/code]
## and [code][[/code] between brackets).
static func _escaped(n: String) -> String:
   return n.replace("[", "[[]").replace("*", "[*]").replace("?", "[?]")


## Names and patterns of a preset: a list of names, one name, or a call of
## [code]matching[/code].
static func _start_names(start: String) -> Array:
   var t := start.strip_edges()
   if t.begins_with("matching("):
      var inner := t.trim_prefix("matching(").trim_suffix(")")
      var found := []
      var rx := RegEx.create_from_string("\"((?:[^\"\\\\]|\\\\.)*)\"")
      for m in rx.search_all(inner):
         found.append(JSON.parse_string(m.get_string(0)))
      return found
   var parsed = JSON.parse_string(t) if not t.is_empty() else null
   return parsed if parsed is Array else ([parsed] if parsed is String else [])


## Asks for a pattern of [param names] in a small dialog: [param done] gets it if it matches some
## name; otherwise the dialog says so and stays open.
func _ask_pattern(names: Array, done: Callable) -> void:
   var ask := ConfirmationDialog.new()
   ask.title = "Pattern of names"
   var box := VBoxContainer.new()
   var note := Label.new()
   note.text = "* stands for any text, ? for one character."
   var edit := LineEdit.new()
   edit.custom_minimum_size.x = FIELD_WIDTH
   edit.placeholder_text = "e.g. label_*"
   box.add_child(note)
   box.add_child(edit)
   ask.add_child(box)
   ask.register_text_enter(edit)
   ask.confirmed.connect(func():
      var pt := edit.text.strip_edges()
      var k := _matches(names, pt)
      if k == 0:
         note.text = "No name matches “%s”." % pt
         ask.popup_centered.call_deferred()
         return
      done.call(pt)
      ask.queue_free())
   ask.canceled.connect(ask.queue_free)
   add_child(ask)
   ask.popup_centered()
   edit.grab_focus()


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
