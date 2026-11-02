extends PanelContainer
## Player of the spread of an epidemic, at the bottom of the visualization panel, shown when the
## network of the current tab has the history of a run (the states of its nodes at every step,
## [code]history = True[/code] of the epidemic models): buttons to go back to step 0, one step
## back, play or pause, one step forward and to the last step, a timeline slider, the speed of
## play (steps per second), the colors of the states, and Export…, which saves the whole spread as
## an animation (GIF, or MP4 video), at the speed of play. Using any of them colors the nodes by
## their state at the step shown; unchecking States gives them back the colors of the style.

## Emitted when the user goes to step [param step].
signal step_changed(step: int)
## Emitted when the States check box is turned on or off.
signal showing_changed(on: bool)
## Emitted when the color of a state changes, with the colors of all the states.
signal colors_changed(colors: Array)
## Emitted with the file chosen to export the animation, and the speed (frames per second).
signal export_requested(path: String, fps: float)

## Icons of the buttons, as SVG shapes in a 16 by 16 box.
const ICONS := {
   "Start": "<rect x='1' y='3' width='2' height='10'/><path d='M8 3v10L3 8zM14 3v10L9 8z'/>",
   "Back": "<path d='M10 3v10L4 8z'/><rect x='11' y='3' width='2' height='10'/>",
   "Play": "<path d='M4 2v12l10-6z'/>",
   "Pause": "<rect x='3' y='3' width='3.5' height='10'/><rect x='9.5' y='3' width='3.5' height='10'/>",
   "Forward": "<rect x='3' y='3' width='2' height='10'/><path d='M6 3v10l6-5z'/>",
   "End": "<path d='M2 3v10l5-5zM8 3v10l5-5z'/><rect x='13' y='3' width='2' height='10'/>",
}

## Last step of the history.
var steps := 0
## Step shown.
var step := 0
var _icons := {}
var _colors := []


func _ready() -> void:
   var ink := get_theme_color("font_color", "Button").to_html(false)
   for name in ICONS:
      var image := Image.new()
      image.load_svg_from_string("<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='#%s'>%s</svg>" % [ink, ICONS[name]], 1.0)
      _icons[name] = ImageTexture.create_from_image(image)
   for name in ["Start", "Back", "Play", "Forward", "End"]:
      get_node("%" + name).icon = _icons[name]
   %Start.pressed.connect(func(): go(0))
   %Back.pressed.connect(func(): go(step - 1))
   %Play.pressed.connect(func(): pause() if not %Timer.is_stopped() else _play())
   %Forward.pressed.connect(func(): go(step + 1))
   %End.pressed.connect(func(): go(steps))
   %Timeline.value_changed.connect(func(v): go(int(v)))
   %Timer.timeout.connect(func(): pause() if step >= steps else go(step + 1))
   %Speed.value_changed.connect(func(v): %Timer.wait_time = 1.0 / v)
   %Export.pressed.connect(func():
      pause()
      %ExportDialog.current_dir = AppSettings.folder("animations", AppSettings.folder("images"))
      %ExportDialog.current_file = "spread.gif"
      %ExportDialog.popup_centered_ratio(0.6))
   %ExportDialog.file_selected.connect(func(path):
      if not path.get_extension().to_lower() in ["gif", "mp4"]:
         path += ".gif"
      AppSettings.store_folder("animations", path)
      export_requested.emit(path, %Speed.value))
   %Show.toggled.connect(func(on):
      if not on:
         pause()
      showing_changed.emit(on))


## Shows the history of the run [param run] (last step [param last]), with its [param states] and
## their [param colors] (texts "#rrggbb"), at step [param at], states shown or not ([param on]).
func show_history(run: String, last: int, states: Array, colors: Array, at: int, on: bool) -> void:
   pause()
   steps = last
   step = at
   %Run.text = run
   %Timeline.max_value = last
   %Timeline.set_value_no_signal(at)
   %Show.set_pressed_no_signal(on)
   _colors = colors.map(func(c): return Color(c))
   for child in %States.get_children():
      %States.remove_child(child)
      child.queue_free()
   for i in states.size():
      var label := Label.new()
      label.text = states[i]
      %States.add_child(label)
      var swatch := SwatchButton.new()
      swatch.color = _colors[i]
      swatch.tooltip_text = "Color of the nodes in state %s" % states[i]
      swatch.color_changed.connect(func(c): _colors[i] = c; colors_changed.emit(_colors.duplicate()))
      %States.add_child(swatch)
   _label()


## The colors of the states.
func colors() -> Array:
   return _colors.duplicate()


## Goes to step [param t] (clamped to the history), showing the states.
func go(t: int) -> void:
   step = clampi(t, 0, steps)
   %Timeline.set_value_no_signal(step)
   %Show.set_pressed_no_signal(true)
   _label()
   step_changed.emit(step)


## Turns the controls off (while an animation is exported) or on.
func set_busy(on: bool) -> void:
   for c in $Row.get_children():
      if c is BaseButton:
         c.disabled = on
      elif c is Slider or c is SpinBox:
         c.editable = not on


## Stops playing.
func pause() -> void:
   %Timer.stop()
   %Play.icon = _icons.get("Play")


func _play() -> void:
   if step >= steps:
      go(0)
   %Timer.wait_time = 1.0 / %Speed.value
   %Timer.start()
   %Play.icon = _icons["Pause"]


func _label() -> void:
   %Step.text = "step %d / %d" % [step, steps]
