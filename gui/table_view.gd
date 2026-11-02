extends VBoxContainer
## One table of the data panel: a page of rows (see the worker's [code]table[/code] command), with
## buttons for the previous and next pages. Clicking the title of a column sorts the whole table by
## it (the worker sorts all the rows, then sends the page), ascending, then descending; a right
## click asks for the menu of the column (see data_panel.gd). Column
## titles are aligned as their values: numbers to the right, texts to the left, with no
## indentation of the rows.

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
   show_rows({}, 0, "")


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
   %Table.clear()
   var columns: Array = reply.get("columns", [])
   var numeric: Array = reply.get("numeric", [])
   _offset = offset
   _total = int(reply.get("total", 0))
   _columns = columns
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
