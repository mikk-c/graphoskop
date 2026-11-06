extends RefCounted
## Theme of the interface (File › Settings…): built from one of the ColorBrewer sequential
## palettes of [constant PALETTES] (Blues by default), in light or dark mode, on the fonts of
## [code]theme.tres[/code]. [method apply] sets it on the main window (the dialogs and menus,
## children of it, follow), with the background behind the panels and the colors of the code
## highlighter. [member colors] holds the colors in use, for what the interface draws itself (e.g.
## the column names of the load dialog).

## ColorBrewer sequential palettes, nine classes each, lightest first.
const PALETTES := {
   "Blues": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"],
   "Greens": ["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#006d2c", "#00441b"],
   "Greys": ["#ffffff", "#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252", "#252525", "#000000"],
   "Purples": ["#fcfbfd", "#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#54278f", "#3f007d"],
   "Reds": ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"],
   "Oranges": ["#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#a63603", "#7f2704"],
}
## Palette and mode without settings.
const DEFAULT_PALETTE := "Blues"

## Colors of the theme in use (see [method palette_colors]).
static var colors := palette_colors(DEFAULT_PALETTE, false)


## The roles of the colors of a theme, from the nine classes of [param name]: light mode on the
## lightest classes with the darkest one as text; dark mode on darkened, less saturated shades of
## the eighth class with a light one as text. Keys: window (background), panel, field (text fields, lists), button,
## hover, pressed, accent (focus, checked boxes), border, text, dim (disabled text, placeholders),
## selection, heading (group titles), tab (unselected tabs).
static func palette_colors(name: String, dark: bool) -> Dictionary:
   var c: Array = (PALETTES.get(name, PALETTES[DEFAULT_PALETTE]) as Array).map(func(h): return Color(h))
   if not dark:
      return {"window": c[0], "panel": c[1], "field": Color.WHITE, "button": c[2], "hover": c[3], "pressed": c[4], "accent": c[6], "border": c[3], "text": c[8], "dim": c[5], "selection": c[3], "heading": c[6], "tab": c[2]}
   var shade := func(k: float) -> Color:   # the eighth class darkened, less saturated
      var d: Color = (c[7] as Color).darkened(k)
      d.s *= 0.65
      return d
   return {"window": shade.call(0.7), "panel": shade.call(0.55), "field": shade.call(0.8), "button": shade.call(0.3), "hover": c[7], "pressed": c[6], "accent": c[4], "border": shade.call(0.15), "text": c[1], "dim": c[4], "selection": c[6], "heading": c[3], "tab": shade.call(0.7)}


## Builds the theme of [param name] in light or [param dark] mode on [param base] (its fonts),
## sets it on [param main] and the background behind it, and colors [param highlighter].
static func apply(main: Control, base: Theme, name: String, dark: bool, highlighter: CodeHighlighter) -> void:
   colors = palette_colors(name, dark)
   main.theme = build(base, colors)
   RenderingServer.set_default_clear_color(colors.window)
   highlight(highlighter, dark)


## A flat box of [param bg], with a [param border] of [param width] pixels.
static func _box(bg: Color, border := Color(0, 0, 0, 0), width := 0, radius := 4, margin := 6) -> StyleBoxFlat:
   var b := StyleBoxFlat.new()
   b.bg_color = bg
   b.border_color = border
   b.set_border_width_all(width)
   b.set_corner_radius_all(radius)
   b.content_margin_left = margin
   b.content_margin_right = margin
   b.content_margin_top = 3
   b.content_margin_bottom = 3
   return b


## A check box (or, [param round], a radio button) of 16 pixels: a frame on the field color, or
## filled with the accent and a mark when [param on].
static func _check(p: Dictionary, on: bool, round := false) -> ImageTexture:
   var img := Image.create(16, 16, false, Image.FORMAT_RGBA8)
   img.fill(Color(0, 0, 0, 0))
   for x in range(1, 15):
      for y in range(1, 15):
         var r := Vector2(x - 7.5, y - 7.5).length()
         if round and r > 7.0:
            continue
         var edge := (r > 5.8) if round else (x == 1 or x == 14 or y == 1 or y == 14)
         img.set_pixel(x, y, p.accent if on else (p.border if edge else p.field))
   if on and round:
      for x in range(5, 11):
         for y in range(5, 11):
            if Vector2(x - 7.5, y - 7.5).length() < 2.8:
               img.set_pixel(x, y, Color.WHITE)
   elif on:
      for k in range(-1, 2):
         for i in 4:
            img.set_pixel(4 + i, 7 + i + k, Color.WHITE)
         for i in 6:
            img.set_pixel(7 + i, 10 - i + k, Color.WHITE)
   return ImageTexture.create_from_image(img)


## The default icon [param icon] of [param type] recolored: dark pixels [param low], light ones
## [param high], in between by their luminance, with their transparency.
static func _tint(icon: String, type: String, low: Color, high: Color) -> Texture2D:
   var source := ThemeDB.get_default_theme().get_icon(icon, type)
   if source == null:
      return null
   var img := source.get_image()
   if img == null:
      return source
   img.decompress()
   img.convert(Image.FORMAT_RGBA8)
   for x in img.get_width():
      for y in img.get_height():
         var c := img.get_pixel(x, y)
         var t := low.lerp(high, c.get_luminance())
         t.a = c.a
         img.set_pixel(x, y, t)
   return ImageTexture.create_from_image(img)


## Icons recolored for the theme: glyphs (arrows, close buttons, …) in the text color; switches
## of CheckButton on the accent; check boxes and radio buttons drawn by [method _check].
const GLYPHS := {"TabBar": ["close", "increment", "decrement", "increment_highlight", "decrement_highlight"], "TabContainer": ["increment", "decrement", "increment_highlight", "decrement_highlight", "menu", "menu_highlight"], "OptionButton": ["arrow"], "Tree": ["arrow", "arrow_collapsed", "arrow_collapsed_mirrored", "select_arrow", "updown"], "PopupMenu": ["submenu", "submenu_mirrored"], "SpinBox": ["updown", "up", "down", "up_hover", "down_hover", "up_pressed", "down_pressed"], "LineEdit": ["clear"], "MenuButton": [], "FileDialog": ["parent_folder", "reload", "toggle_hidden", "folder", "file"], "HSlider": ["grabber", "grabber_highlight", "tick"], "VSlider": ["grabber", "grabber_highlight", "tick"], "Window": ["close", "close_pressed"]}


## The theme of the colors [param p] on the fonts of [param base].
static func build(base: Theme, p: Dictionary) -> Theme:
   var t: Theme = base.duplicate()
   for type in ["Label", "Button", "MenuButton", "OptionButton", "CheckBox", "CheckButton", "LineEdit", "TextEdit", "CodeEdit", "Tree", "ItemList", "TabBar", "TabContainer", "PopupMenu", "MenuBar", "LinkButton", "SpinBox", "TooltipLabel", "ProgressBar", "AcceptDialog", "Window", "RichTextLabel", "ColorPickerButton"]:
      for item in ["font_color", "font_hover_color", "font_hovered_color", "font_hovered_selected_color", "font_hovered_dimmed_color", "font_pressed_color", "font_focus_color", "font_hover_pressed_color", "font_selected_color", "default_color", "title_color"]:
         t.set_color(item, type, p.text)
      for item in ["font_disabled_color", "font_placeholder_color", "font_uneditable_color", "font_readonly_color"]:
         t.set_color(item, type, p.dim)
      t.set_color("font_outline_color", type, p.window)
      t.set_color("icon_normal_color", type, p.text)
      t.set_color("icon_hover_color", type, p.text)
      t.set_color("icon_pressed_color", type, p.text)
      t.set_color("icon_focus_color", type, p.text)
      t.set_color("icon_disabled_color", type, p.dim)
   t.set_type_variation("HeadingLabel", "Label")
   t.set_color("font_color", "HeadingLabel", p.heading)
   for type in ["Button", "MenuButton", "OptionButton", "ColorPickerButton"]:
      t.set_stylebox("normal", type, _box(p.button))
      t.set_stylebox("hover", type, _box(p.hover))
      t.set_stylebox("pressed", type, _box(p.pressed))
      t.set_stylebox("hover_pressed", type, _box(p.pressed))
      t.set_stylebox("disabled", type, _box(p.panel))
      t.set_stylebox("focus", type, _box(Color(0, 0, 0, 0), p.accent, 1))
   for type in ["CheckBox", "CheckButton"]:
      for s in ["normal", "pressed", "hover", "hover_pressed", "disabled"]:
         t.set_stylebox(s, type, _box(Color(0, 0, 0, 0)))
      t.set_stylebox("focus", type, _box(Color(0, 0, 0, 0), p.accent, 1))
   for type in ["CheckBox", "PopupMenu"]:
      t.set_icon("unchecked", type, _check(p, false))
      t.set_icon("checked", type, _check(p, true))
      t.set_icon("unchecked_disabled", type, _check(p, false))
      t.set_icon("checked_disabled", type, _check(p, true))
      t.set_icon("radio_unchecked", type, _check(p, false, true))
      t.set_icon("radio_checked", type, _check(p, true, true))
      t.set_icon("radio_unchecked_disabled", type, _check(p, false, true))
      t.set_icon("radio_checked_disabled", type, _check(p, true, true))
   for icon in ["checked", "checked_disabled", "checked_mirrored", "checked_disabled_mirrored"]:
      var tinted := _tint(icon, "CheckButton", p.accent, p.field.lerp(Color.WHITE, 0.5))
      if tinted != null:
         t.set_icon(icon, "CheckButton", tinted)
   for icon in ["unchecked", "unchecked_disabled", "unchecked_mirrored", "unchecked_disabled_mirrored"]:
      var tinted := _tint(icon, "CheckButton", p.border, p.text)
      if tinted != null:
         t.set_icon(icon, "CheckButton", tinted)
   for type in GLYPHS:
      for icon in GLYPHS[type]:
         var tinted := _tint(icon, type, p.text, p.text)
         if tinted != null:
            t.set_icon(icon, type, tinted)
   for type in ["LineEdit", "TextEdit", "CodeEdit"]:
      t.set_stylebox("normal", type, _box(p.field, p.border, 1))
      t.set_stylebox("focus", type, _box(Color(0, 0, 0, 0), p.accent, 1))
      t.set_stylebox("read_only", type, _box(p.panel, p.border, 1))
      t.set_color("caret_color", type, p.text)
      t.set_color("selection_color", type, p.selection)
      t.set_color("clear_button_color", type, p.text)
      t.set_color("clear_button_color_pressed", type, p.accent)
   t.set_color("background_color", "CodeEdit", p.field)
   t.set_color("current_line_color", "CodeEdit", p.panel)
   t.set_color("line_number_color", "CodeEdit", p.dim)
   t.set_color("word_highlighted_color", "CodeEdit", p.panel)
   for type in ["Tree", "ItemList"]:
      t.set_stylebox("panel", type, _box(p.field, p.border, 1, 2, 2))
      t.set_stylebox("focus", type, _box(Color(0, 0, 0, 0), p.accent, 1, 2, 2))
      t.set_stylebox("selected", type, _box(p.selection, Color(0, 0, 0, 0), 0, 0, 2))
      t.set_stylebox("selected_focus", type, _box(p.selection, Color(0, 0, 0, 0), 0, 0, 2))
      t.set_stylebox("cursor", type, _box(Color(0, 0, 0, 0), p.accent, 1, 0, 2))
      t.set_stylebox("cursor_unfocused", type, _box(Color(0, 0, 0, 0), p.border, 1, 0, 2))
      t.set_stylebox("hovered", type, _box(p.panel, Color(0, 0, 0, 0), 0, 0, 2))
      t.set_color("guide_color", type, p.border)
      t.set_color("relationship_line_color", type, p.border)
      t.set_color("children_hl_line_color", type, p.border)
      t.set_color("parent_hl_line_color", type, p.border)
   t.set_stylebox("title_button_normal", "Tree", _box(p.button, Color(0, 0, 0, 0), 0, 0))
   t.set_stylebox("title_button_hover", "Tree", _box(p.hover, Color(0, 0, 0, 0), 0, 0))
   t.set_stylebox("title_button_pressed", "Tree", _box(p.pressed, Color(0, 0, 0, 0), 0, 0))
   t.set_color("title_button_color", "Tree", p.text)
   t.set_stylebox("panel", "TabContainer", _box(p.panel, Color(0, 0, 0, 0), 0, 0))
   t.set_stylebox("tabbar_background", "TabContainer", _box(p.window, Color(0, 0, 0, 0), 0, 0))
   for type in ["TabContainer", "TabBar"]:
      t.set_stylebox("tab_selected", type, _box(p.panel, Color(0, 0, 0, 0), 0, 4, 10))
      t.set_stylebox("tab_unselected", type, _box(p.tab, Color(0, 0, 0, 0), 0, 4, 10))
      t.set_stylebox("tab_hovered", type, _box(p.hover, Color(0, 0, 0, 0), 0, 4, 10))
      t.set_stylebox("tab_disabled", type, _box(p.tab, Color(0, 0, 0, 0), 0, 4, 10))
      t.set_stylebox("tab_focus", type, _box(Color(0, 0, 0, 0), p.accent, 1))
      t.set_stylebox("button_highlight", type, _box(p.hover))
      t.set_color("font_selected_color", type, p.accent if p.text.get_luminance() < 0.5 else p.text)
      t.set_color("font_unselected_color", type, p.dim)
      t.set_color("font_hovered_color", type, p.text)
      t.set_color("drop_mark_color", type, p.accent)
   for type in ["Panel", "PanelContainer"]:
      t.set_stylebox("panel", type, _box(p.panel, Color(0, 0, 0, 0), 0, 0, 0))
   var line := StyleBoxLine.new()
   line.color = p.border
   line.thickness = 1
   t.set_stylebox("panel", "PopupMenu", _box(p.field, p.border, 1, 4, 4))
   t.set_stylebox("hover", "PopupMenu", _box(p.selection, Color(0, 0, 0, 0), 0, 3))
   t.set_stylebox("separator", "PopupMenu", line)
   t.set_stylebox("labeled_separator_left", "PopupMenu", line)
   t.set_stylebox("labeled_separator_right", "PopupMenu", line)
   t.set_color("font_separator_color", "PopupMenu", p.dim)
   t.set_color("font_accelerator_color", "PopupMenu", p.dim)
   t.set_stylebox("panel", "PopupPanel", _box(p.field, p.border, 1, 4, 4))
   t.set_stylebox("panel", "AcceptDialog", _box(p.window, Color(0, 0, 0, 0), 0, 0, 8))
   t.set_stylebox("panel", "TooltipPanel", _box(p.field, p.border, 1, 4, 6))
   t.set_color("font_color", "TooltipLabel", p.text)
   t.set_stylebox("embedded_border", "Window", _box(p.window, p.border, 1, 4, 0))
   for s in ["normal", "hover", "pressed", "disabled", "hover_pressed"]:
      t.set_stylebox(s, "MenuBar", _box(p.window if s == "normal" or s == "disabled" else p.hover, Color(0, 0, 0, 0), 0, 3))
   for type in ["HScrollBar", "VScrollBar"]:
      t.set_stylebox("scroll", type, _box(p.panel, Color(0, 0, 0, 0), 0, 4, 0))
      t.set_stylebox("scroll_focus", type, _box(p.panel, Color(0, 0, 0, 0), 0, 4, 0))
      t.set_stylebox("grabber", type, _box(p.hover, Color(0, 0, 0, 0), 0, 4, 0))
      t.set_stylebox("grabber_highlight", type, _box(p.pressed, Color(0, 0, 0, 0), 0, 4, 0))
      t.set_stylebox("grabber_pressed", type, _box(p.accent, Color(0, 0, 0, 0), 0, 4, 0))
   for type in ["HSlider", "VSlider"]:
      t.set_stylebox("slider", type, _box(p.button, Color(0, 0, 0, 0), 0, 2, 0))
      t.set_stylebox("grabber_area", type, _box(p.pressed, Color(0, 0, 0, 0), 0, 2, 0))
      t.set_stylebox("grabber_area_highlight", type, _box(p.accent, Color(0, 0, 0, 0), 0, 2, 0))
   for type in ["HSeparator", "VSeparator"]:
      t.set_stylebox("separator", type, line)
   for type in ["HSplitContainer", "VSplitContainer", "SplitContainer"]:
      t.set_color("touch_dragger_color", type, p.border)
   t.set_stylebox("background", "ProgressBar", _box(p.panel, p.border, 1))
   t.set_stylebox("fill", "ProgressBar", _box(p.pressed))
   return t


## Colors of the Python code highlighter for a light or a [param dark] background (those of
## Visual Studio Code's Light+ and Dark+ themes): keywords, constants and definitions, numbers,
## functions, members, symbols, strings and comments.
static func highlight(h: CodeHighlighter, dark: bool) -> void:
   if h == null:
      return
   var flow := Color("#c586c0") if dark else Color("#af00db")
   var defining := Color("#569cd6") if dark else Color("#0000ff")
   h.number_color = Color("#b5cea8") if dark else Color("#098658")
   h.function_color = Color("#dcdcaa") if dark else Color("#795e26")
   h.member_variable_color = Color("#9cdcfe") if dark else Color("#001080")
   h.symbol_color = Color("#d4d4d4") if dark else Color("#383a42")
   var keywords := h.keyword_colors
   for k in keywords:
      keywords[k] = defining if k in ["False", "None", "True", "class", "def", "lambda"] else flow
   h.keyword_colors = keywords
   var string := Color("#ce9178") if dark else Color("#a31515")
   h.color_regions = {"\" \"": string, "' '": string, "#": Color("#6a9955") if dark else Color("#008000")}
   h.clear_highlighting_cache()
