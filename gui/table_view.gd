extends VBoxContainer
## One table of the data panel: a page of rows (see the worker's [code]table[/code] command), with
## buttons for the previous and next pages. Column titles are aligned as their values: numbers to
## the right, texts to the left, with no indentation of the rows.

## Emitted when the user asks for the page starting at row [param offset].
signal page_requested(offset: int)

## Rows per page.
const PAGE := 200

var _offset := 0
var _total := 0


func _ready() -> void:
   %Table.add_theme_constant_override("item_margin", 0)
   %Prev.pressed.connect(func(): page_requested.emit(maxi(_offset - PAGE, 0)))
   %Next.pressed.connect(func(): page_requested.emit(_offset + PAGE))
   show_rows({}, 0, "")


## Shows a reply of the worker (columns, rows, total), whose first row is row [param offset], under
## the title [param title].
func show_rows(reply: Dictionary, offset: int, title: String) -> void:
   %Title.text = title
   %Table.clear()
   var columns: Array = reply.get("columns", [])
   var numeric: Array = reply.get("numeric", [])
   _offset = offset
   _total = int(reply.get("total", 0))
   %Table.columns = maxi(columns.size(), 1)
   for j in columns.size():
      %Table.set_column_title(j, columns[j])
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
