extends VBoxContainer
## One table of the data panel: a page of rows (see the worker's [code]table[/code] command), with
## buttons for the previous and next pages. Clicking the title of a column sorts the whole table by
## it (the worker sorts all the rows, then sends the page), ascending, then descending; a right
## click asks for the menu of the column (see data_panel.gd). Column
## titles are aligned as their values: numbers to the right, texts to the left, with no
## indentation of the rows. Double clicking a cell of an editable column ([member editable])
## edits its value in a field over the cell: Enter sends it ([signal value_edited]), Escape or a
## click elsewhere leaves it as it was.

## Emitted when the user asks for the page starting at row [param offset].
signal page_requested(offset: int)
## Emitted when the user chooses another table in the menu above the table (see
## [method set_choices]).
signal choice_changed(name: String)
## Emitted when the user clicks the title of a column: the whole table is to be sorted by it,
## ascending (a second click: descending).
signal sort_requested(column: String, ascending: bool)
## Emitted when the user right-clicks the title of a column.
signal column_menu_requested(column: String)
## Emitted when a value was edited in place: the position of its row in the whole table (from the
## worker, [code]index[/code]; the row of the page if there is none), its column and the text typed.
signal value_edited(index: int, column: String, text: String)

## Whether a column can be edited (by name; the table checks nothing else): set by the data panel.
var editable := func(_column: String) -> bool: return false
var _index := []
## The field editing a cell, or null.
var _field: LineEdit

## Column the table is sorted by (empty: not sorted), and its order.
var sort_column := ""
var ascending := true
var _columns := []

## Rows per page.
const PAGE := 200

var _offset := 0
var _total := 0


func _ready() -> void:
   %Table.add_theme_constant_override("item_margin", 0)
   %Prev.pressed.connect(func(): page_requested.emit(maxi(_offset - PAGE, 0)))
   %Next.pressed.connect(func(): page_requested.emit(_offset + PAGE))
   %Choice.item_selected.connect(func(i): choice_changed.emit(%Choice.get_item_text(i)))
   %Table.column_title_clicked.connect(func(j, button):
      if button == MOUSE_BUTTON_RIGHT and j < _columns.size():
         column_menu_requested.emit(_columns[j])
      if button != MOUSE_BUTTON_LEFT or j >= _columns.size():
         return
      ascending = not ascending if sort_column == _columns[j] else true
      sort_column = _columns[j]
      sort_requested.emit(sort_column, ascending))
   %Table.gui_input.connect(_on_table_input)
   for bar in %Table.get_children(true):   # its own scroll bars: the field closes when the table scrolls
      if bar is ScrollBar:
         bar.value_changed.connect(func(_v): _end_edit())
   show_rows({}, 0, "")


## Double click on a cell: edits it in place if its column is editable, in a field drawn over the
## cell (a child of the table, so it stays where the cell is; the editor of Tree opens in a
## window of its own, misplaced with native windows). Enter sends the value, Escape or a click
## elsewhere leaves it as it was.
func _on_table_input(event: InputEvent) -> void:
   if not (event is InputEventMouseButton and event.double_click and event.button_index == MOUSE_BUTTON_LEFT):
      return
   var item: TreeItem = %Table.get_item_at_position(event.position)
   var j: int = %Table.get_column_at_position(event.position)
   if item == null or j < 0 or j >= _columns.size() or not editable.call(_columns[j]):
      return
   _end_edit()
   var field := LineEdit.new()
   field.text = item.get_text(j)
   field.alignment = item.get_text_alignment(j)
   var area: Rect2 = %Table.get_item_area_rect(item, j)
   field.position = area.position
   field.size = area.size
   field.custom_minimum_size = Vector2(0, area.size.y)
   field.select_all_on_focus = true
   var row := item.get_index()
   var before := field.text
   field.text_submitted.connect(func(typed):
      _end_edit()
      if typed != before:
         value_edited.emit(int(_index[row]) if row < _index.size() else row, _columns[j], typed))
   field.focus_exited.connect(_end_edit.call_deferred)
   field.gui_input.connect(func(e):
      if e is InputEventKey and e.pressed and e.keycode == KEY_ESCAPE:
         _end_edit()
         field.accept_event())
   %Table.add_child(field)
   _field = field
   field.grab_focus()
   %Table.accept_event()


## Closes the field editing a cell, if any (the value is not sent).
func _end_edit() -> void:
   if _field != null and is_instance_valid(_field):
      var f := _field
      _field = null
      f.queue_free()


## Shows a menu of tables [param names] above the table (hidden if empty), with [param current]
## chosen.
func set_choices(names: Array, current: String) -> void:
   %Choice.clear()
   for n in names:
      %Choice.add_item(n)
   %Choice.select(maxi(names.find(current), 0) if not names.is_empty() else -1)
   %Choice.visible = not names.is_empty()


## Shows a reply of the worker (columns, rows, total), whose first row is row [param offset], under
## the title [param title]; the column the table is sorted by has an arrow (▲ ascending, ▼
## descending).
func show_rows(reply: Dictionary, offset: int, title: String) -> void:
   %Title.text = title
   _end_edit()
   %Table.clear()
   var columns: Array = reply.get("columns", [])
   var numeric: Array = reply.get("numeric", [])
   _offset = offset
   _total = int(reply.get("total", 0))
   _columns = columns
   _index = reply.get("index", [])
   %Table.columns = maxi(columns.size(), 1)
   for j in columns.size():
      %Table.set_column_title(j, columns[j] + ((" ▲" if ascending else " ▼") if columns[j] == sort_column else ""))
      %Table.set_column_expand(j, true)
      %Table.set_column_clip_content(j, true)
      %Table.set_column_title_alignment(j, HORIZONTAL_ALIGNMENT_RIGHT if j < numeric.size() and numeric[j] else HORIZONTAL_ALIGNMENT_LEFT)
   var root: TreeItem = %Table.create_item()
   for row in reply.get("rows", []):
      var item: TreeItem = %Table.create_item(root)
      for j in row.size():
         item.set_text(j, row[j])
         if j < numeric.size() and numeric[j]:
            item.set_text_alignment(j, HORIZONTAL_ALIGNMENT_RIGHT)
   var rows: int = reply.get("rows", []).size()
   %Pager.visible = _total > PAGE
   %Range.text = "%d–%d of %d" % [offset + 1, offset + rows, _total] if rows > 0 else ""
   %Prev.disabled = offset == 0
   %Next.disabled = offset + rows >= _total
