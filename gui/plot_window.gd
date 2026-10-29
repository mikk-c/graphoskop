extends AcceptDialog
## Window of a plot drawn by the worker with matplotlib (for now, the degree distribution:
## complementary cumulative distribution on log-log axes, with the best power-law fit), without a
## title, with a text (e.g. the verdict of the fit: whether the distribution is a power law) and
## notes (its warnings). The colors of the elements of the plot can be changed (each with a color
## button: the worker draws the plot again). Save PNG… and Save PDF… write the plot, with its
## colors, to a file (PNG at 200 dpi).

## Emitted with the path chosen to save the plot of the network of tab [param tab].
signal save_requested(tab: int, path: String)
## Emitted (after a short pause) with the colors of the elements of the plot (key to "#rrggbb").
signal colors_changed(tab: int, colors: Dictionary)

var _tab := -1
var _colors := {}


func _ready() -> void:
   %SavePNG.pressed.connect(func(): _ask("png"))
   %SavePDF.pressed.connect(func(): _ask("pdf"))
   %SaveDialog.file_selected.connect(func(path):
      AppSettings.store_folder("plots", path)
      if not %Debounce.is_stopped():   # colors changed just before: they go first
         %Debounce.stop()
         colors_changed.emit(_tab, _colors.duplicate())
      save_requested.emit(_tab, path))
   %Debounce.timeout.connect(func(): colors_changed.emit(_tab, _colors.duplicate()))


## Shows the plot [param png] (image bytes) of the network of tab [param tab] in a window titled
## [param heading], with color buttons for its [param elements] ([key, label, color]), the
## [param verdict] and the [param warnings] of the fit.
func show_plot(tab: int, heading: String, png: PackedByteArray, elements: Array, verdict: String, warnings: Array) -> void:
   _tab = tab
   title = heading
   set_image(png)
   _colors = {}
   for child in %Colors.get_children():
      child.queue_free()
   for e in elements:
      var label := Label.new()
      label.text = e[1]
      %Colors.add_child(label)
      var button := SwatchButton.new()
      button.color = Color(e[2])
      var key: String = e[0]
      button.color_changed.connect(func(c): _colors[key] = "#" + c.to_html(false); %Debounce.start())
      %Colors.add_child(button)
      _colors[key] = e[2]
   %Verdict.text = verdict
   %Warnings.text = "\n".join(warnings)
   %Warnings.visible = not warnings.is_empty()
   reset_size()
   popup_centered()


## Shows the image [param png] (PNG bytes) of the plot.
func set_image(png: PackedByteArray) -> void:
   var image := Image.new()
   image.load_png_from_buffer(png)
   %Image.texture = ImageTexture.create_from_image(image)


func _ask(format: String) -> void:
   %SaveDialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
   %SaveDialog.filters = PackedStringArray(["*.%s ; %s" % [format, format.to_upper()]])
   %SaveDialog.current_dir = AppSettings.folder("plots")
   %SaveDialog.current_file = "degree_distribution.%s" % format
   %SaveDialog.popup_centered_ratio(0.6)
