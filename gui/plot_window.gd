extends AcceptDialog
## Window of a plot drawn by the worker with matplotlib (the degree distribution: complementary
## cumulative distribution on log-log axes, with the best power-law fit; the comparison of runs
## of epidemic models; or the distribution of an attribute), without a title, with a text (e.g.
## the verdict of the fit: whether the distribution is a power law) and notes (its warnings). Tick
## marks are inside the axes.
##
## The distribution of an attribute is plotted as a count, PDF, CDF or CCDF (Plot menu; count and
## PDF only for qualitative attributes); choosing one draws its series in its usual way (bars for
## count and PDF, lines for CDF and CCDF), and empty axis labels are automatic. The plot of a
## randomization test (the null networks and the observed value) has the same menu for the tail of
## its pseudo p-value (right, left, two-sided), which the text under the plot gives.
##
## Under the plot: the color of every element (each with a color button) and, for data series,
## how they are drawn (lines, points, lines with points, or bars); for each axis, linear or
## logarithmic scale, its label (any text) and its range (from, to: empty for automatic); the
## font size (the font is the default font of gnuplot, Helvetica, or one with the same metrics).
## Heatmaps (matrices of node vector analyses, nestedness) also have a row for their color scale:
## linear or logarithmic, its label and its range (from, to: empty for automatic).
## The worker draws the plot again after every change (after a short pause). Save PNG… and Save
## PDF… write the plot as shown to a file (PNG at 200 dpi); plots of a matrix (pairwise node
## vector analyses) also have Save table…, which writes its values (tab separated).
##
## With [b]Gradient[/b] (along x or along y), every bar, point and line segment of the data series
## gets the color of its x or y value along a ColorBrewer palette (sequential or diverging, in
## reverse with Reverse), instead of the color of its series (not for heatmaps).
##
## The colors, styles, axes, gradient and font size show only with [b]Advanced[/b] on (off by default, see
## [method AppSettings.advanced]); the plot, its text, its Plot menu and the Save buttons always.

## Emitted with the path chosen to save the plot of the network of tab [param tab].
signal save_requested(tab: int, path: String)
## Emitted (after a short pause) with the colors of the elements of the plot (key to "#rrggbb"),
## the styles of its data series (key to "lines", "points", "linespoints" or "bars") and its
## options (xlog, ylog, font_size, xlabel, ylabel, xmin, xmax, ymin, ymax, gradient (axis, palette,
## reversed, or null), kind for distributions and orders, and zlog, zlabel, zmin, zmax for
## heatmaps).
signal changed(tab: int, colors: Dictionary, styles: Dictionary, options: Dictionary)

## Labels of the styles of data series, as in the worker ([code]PLOT_STYLES[/code]).
const STYLE_LABELS := {"lines": "lines", "points": "points", "linespoints": "lines and points", "bars": "bars"}

var _tab := -1
var _colors := {}
var _styles := {}
var _menus := {}
var _style_names := []
var _kinds := []
var _file := "plot"
## Whether the plot is a heatmap (it has a color scale row).
var _heatmap := false
var _advanced := CheckButton.new()
## Row of the gradient: none, along x or along y; the palette; reversed.
var _gradient_row := HBoxContainer.new()
var _gradient := OptionButton.new()
var _palette := OptionButton.new()
var _reverse := CheckBox.new()


func _ready() -> void:
   _advanced.text = "Advanced"
   _advanced.tooltip_text = "Show the colors, styles, axes and font of the plot"
   $Box/Buttons.add_child(_advanced)
   $Box/Buttons.move_child(_advanced, 0)
   _advanced.toggled.connect(func(on):
      AppSettings.set_advanced(on)
      _show_rows())
   %SavePNG.pressed.connect(func(): _ask("png"))
   %SavePDF.pressed.connect(func(): _ask("pdf"))
   %SaveTable.pressed.connect(func(): _ask("tsv"))
   %SaveDialog.file_selected.connect(func(path):
      AppSettings.store_folder("plots", path)
      if not %Debounce.is_stopped():   # changes made just before go first
         %Debounce.stop()
         _emit()
      save_requested.emit(_tab, path))
   %Debounce.timeout.connect(_emit)
   for b in [%XLog, %YLog, %ZLog]:
      b.item_selected.connect(func(_i): %Debounce.start())
   for e in [%XLabel, %YLabel, %XMin, %XMax, %YMin, %YMax, %ZLabel, %ZMin, %ZMax]:
      e.text_changed.connect(func(_t): %Debounce.start())
   %FontSize.value_changed.connect(func(_v): %Debounce.start())
   _gradient_menus()
   %Kind.item_selected.connect(func(i):
      for key in _menus:   # the usual style of the kind
         _styles[key] = _kinds[i][2]
         _menus[key].select(maxi(_style_names.find(_kinds[i][2]), 0))
      %Debounce.start())


func _emit() -> void:
   var options := {"xlog": %XLog.selected == 1, "ylog": %YLog.selected == 1, "font_size": %FontSize.value, "xlabel": %XLabel.text, "ylabel": %YLabel.text}
   var ranges := [["xmin", %XMin], ["xmax", %XMax], ["ymin", %YMin], ["ymax", %YMax]]
   if _heatmap:
      options.zlabel = %ZLabel.text
      options.zlog = %ZLog.selected == 1
      ranges.append_array([["zmin", %ZMin], ["zmax", %ZMax]])
   for pair in ranges:
      var t: String = pair[1].text.strip_edges()
      if t.is_empty() or t.is_valid_float():   # half-typed numbers wait
         options[pair[0]] = float(t) if not t.is_empty() else null
   if not _kinds.is_empty():
      options.kind = _kinds[%Kind.selected][0]
   if not _heatmap:
      options.gradient = null if _gradient.selected == 0 else {"axis": "x" if _gradient.selected == 1 else "y", "palette": _palette.get_item_text(_palette.selected), "reversed": _reverse.button_pressed}
   changed.emit(_tab, _colors.duplicate(), _styles.duplicate(), options)


## Shows the plot [param png] (image bytes) of the network of tab [param tab] in a window titled
## [param heading], with a color button for each of its [param elements] ([key, label, color,
## style], style empty for elements that are not data series), a menu of [param styles] for each
## data series, its [param options], and the [param verdict] and [param warnings] of the fit (all
## from the worker); [param file] is the name proposed to save it; [param kinds] are the kinds of
## a distribution or the tails of a randomization test ([key, label, usual style]; empty for
## other plots), chosen in a menu labeled [param kind_label]; with [param table], the values of the
## plot can also be saved as a table.
func show_plot(tab: int, heading: String, png: PackedByteArray, elements: Array, options: Dictionary, styles: Array, verdict: String, warnings: Array, file := "degree_distribution", kinds := [], kind_label := "Plot", table := false) -> void:
   _tab = tab
   %SaveTable.visible = table
   _file = file
   title = heading
   set_image(png)
   _colors = {}
   _styles = {}
   _menus = {}
   _style_names = styles
   _kinds = kinds
   %KindRow.visible = not kinds.is_empty()
   %KindLabel.text = kind_label
   %Kind.clear()
   for k in kinds:
      %Kind.add_item(k[1])
   %Kind.select(maxi(kinds.map(func(k): return k[0]).find(options.get("kind", "")), 0) if not kinds.is_empty() else -1)
   for pair in [[%XLabel, "xlabel"], [%YLabel, "ylabel"]]:   # plots whose empty labels are automatic
      pair[0].placeholder_text = "automatic" if not kinds.is_empty() or str(options.get(pair[1], "")).is_empty() else ""
   for child in %Colors.get_children():
      %Colors.remove_child(child)
      child.queue_free()
   for e in elements:
      var key: String = e[0]
      var box := HBoxContainer.new()
      var label := Label.new()
      label.text = e[1]
      box.add_child(label)
      var button := SwatchButton.new()
      button.color = Color(e[2])
      button.color_changed.connect(func(c): _colors[key] = "#" + c.to_html(false); %Debounce.start())
      box.add_child(button)
      _colors[key] = e[2]
      if e.size() > 3 and not str(e[3]).is_empty():
         var menu := OptionButton.new()
         for st in styles:
            menu.add_item(STYLE_LABELS.get(st, st))
         menu.select(maxi(styles.find(e[3]), 0))
         menu.tooltip_text = "How %s is drawn" % e[1]
         menu.item_selected.connect(func(i): _styles[key] = styles[i]; %Debounce.start())
         box.add_child(menu)
         _styles[key] = e[3]
         _menus[key] = menu
      %Colors.add_child(box)
   %XLog.select(1 if options.get("xlog", false) else 0)
   %YLog.select(1 if options.get("ylog", false) else 0)
   %XLabel.text = str(options.get("xlabel", ""))
   %YLabel.text = str(options.get("ylabel", ""))
   _heatmap = options.has("zmin")   # heatmaps: the range of the color scale
   %ZLabel.text = str(options.get("zlabel", ""))
   %ZLog.select(1 if options.get("zlog", false) else 0)
   for pair in [["xmin", %XMin], ["xmax", %XMax], ["ymin", %YMin], ["ymax", %YMax], ["zmin", %ZMin], ["zmax", %ZMax]]:
      var v = options.get(pair[0])
      pair[1].text = "" if v == null else str(v)
   %FontSize.set_value_no_signal(options.get("font_size", 14))
   var g = options.get("gradient")
   _gradient.select(0 if g == null else (1 if g.axis == "x" else 2))
   if g != null:
      for i in _palette.item_count:
         if _palette.get_item_text(i) == g.palette:
            _palette.select(i)
      _reverse.set_pressed_no_signal(g.get("reversed", false))
   %Debounce.stop()
   %Verdict.text = verdict
   %Warnings.text = "\n".join(warnings)
   %Warnings.visible = not warnings.is_empty()
   _advanced.set_pressed_no_signal(AppSettings.advanced())
   _show_rows()
   popup_centered()


## Shows the rows of the colors, axes and font only with [b]Advanced[/b] on, and sizes the window.
func _show_rows() -> void:
   var on := _advanced.button_pressed
   for row in [%Colors, $Box/XAxis, $Box/YAxis, $Box/Buttons/FontLabel, %FontSize]:
      row.visible = on
   %ZAxis.visible = on and _heatmap
   _gradient_row.visible = on and not _heatmap
   _palette.disabled = _gradient.selected == 0
   _reverse.disabled = _gradient.selected == 0
   size = Vector2i.ZERO
   reset_size()


## Shows the text under the plot (e.g. the pseudo p-value of a randomization test).
func show_text(text: String) -> void:
   %Verdict.text = text


## Shows the image [param png] (PNG bytes) of the plot.
func set_image(png: PackedByteArray) -> void:
   var image := Image.new()
   image.load_png_from_buffer(png)
   %Image.texture = ImageTexture.create_from_image(image)


func _ask(format: String) -> void:
   %SaveDialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
   %SaveDialog.filters = PackedStringArray(["*.%s ; %s" % [format, format.to_upper()]])
   %SaveDialog.current_dir = AppSettings.folder("plots")
   %SaveDialog.current_file = "%s.%s" % [_file, format]
   %SaveDialog.popup_centered_ratio(0.6)


## Builds the row of the gradient, under the axes: Gradient (none, along x, along y), the
## sequential and diverging ColorBrewer palettes (with their colors), Reverse.
func _gradient_menus() -> void:
   var label := Label.new()
   label.text = "Gradient"
   label.custom_minimum_size = Vector2(50, 0)
   _gradient_row.add_child(label)
   for t in ["none", "along x", "along y"]:
      _gradient.add_item(t)
   _gradient.select(0)
   _gradient.tooltip_text = "Colors every bar, point and line segment by its x or y value, along the palette"
   _gradient_row.add_child(_gradient)
   var palettes: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://colorbrewer.json"))
   for kind in ["sequential", "diverging"]:
      for name in palettes[kind]:
         var sizes: Array = (palettes[kind][name] as Dictionary).keys().map(func(k): return int(k))
         sizes.sort()
         var colors: Array = palettes[kind][name][str(sizes.back())]
         var image := Image.create(8 * colors.size(), 12, false, Image.FORMAT_RGB8)
         for i in colors.size():
            image.fill_rect(Rect2i(8 * i, 0, 8, 12), Color(colors[i]))
         _palette.add_icon_item(ImageTexture.create_from_image(image), name)
   _palette.fit_to_longest_item = false
   _palette.select(maxi(range(_palette.item_count).map(func(i): return _palette.get_item_text(i)).find("YlOrRd"), 0))
   _gradient_row.add_child(_palette)
   _reverse.text = "Reverse"
   _gradient_row.add_child(_reverse)
   $Box.add_child(_gradient_row)
   $Box.move_child(_gradient_row, %ZAxis.get_index() + 1)
   _gradient.item_selected.connect(func(i):
      _palette.disabled = i == 0
      _reverse.disabled = i == 0
      %Debounce.start())
   _palette.item_selected.connect(func(_i): %Debounce.start())
   _reverse.toggled.connect(func(_on): %Debounce.start())
