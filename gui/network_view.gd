extends SubViewportContainer
## Visualization panel of one network: its nodes and edges, as two MultiMesh instances (one draw
## call each), filled with the buffers computed by the Python worker. Drag to pan, scroll to zoom.
## When nodes are large enough on screen ([constant SELECT_PIXELS]), hovering a node shows its id
## and clicking it selects it (clicking elsewhere clears the selection).
##
## The viewport is drawn only after a change (new drawing, pan, zoom, resize, selection), so an
## idle view costs nothing and does not compete with the worker for the processor or the GPU. (A
## Camera2D would make Godot redraw at every frame: the view is moved with the canvas transform
## instead.)

## Emitted when the selection changes: the position of the node in the drawing and its id, or -1
## and "" when the selection is cleared.
signal node_selected(index: int, id: String)

## Minimum diameter of a node on screen, in pixels.
const NODE_PIXELS := 3.0
## Width of the edges, as a fraction of the diameter of the nodes.
const EDGE_FRACTION := 1.0 / 6.0
## Minimum width of an edge on screen, in pixels.
const EDGE_PIXELS := 1.0
## Floats per node, per edge and per Bézier piece of bundled edges in the buffers of the worker.
const NODE_STRIDE := 16
const EDGE_STRIDE := 12
const CURVE_STRIDE := 16
## Shader of bundled edges (cubic Bézier pieces computed by the GPU).
const CurveShader := preload("res://curve.gdshader")
## Smallest default diameter of the nodes on screen, in pixels, at which they can be hovered and
## selected.
const SELECT_PIXELS := 6.0
## Largest move of the mouse, in pixels, between press and release that counts as a click.
const CLICK_PIXELS := 4.0
## Zoom factor of one wheel step.
const ZOOM_STEP := 1.15
## Empty space around the network when it is fitted to the view, in pixels.
const FIT_MARGIN := 8.0

## World point at the center of the view.
var center := Vector2.ZERO
## Pixels per world unit.
var zoom := 1.0
## Default diameter of the nodes in world units (each node has its own, in the buffer).
var node_size := 0.5
## Input ids of the nodes, in the drawing order.
var ids := PackedStringArray()
## Position of the selected node in the drawing, or -1.
var selected := -1
var _box := Rect2(0, 0, 1, 1)
var _extent := Rect2(0, 0, 1, 1)
var _fitted := false
var _buffer := PackedFloat32Array()
var _grid := {}
var _order := PackedInt32Array()
var _starts := PackedInt32Array()
var _press := Vector2.ZERO
var _straight: Material
var _curved := ShaderMaterial.new()
var _strips := {}
var _pressed := false
@onready var _viewport: SubViewport = $Viewport
@onready var _nodes: MultiMeshInstance2D = %Nodes
@onready var _edges: MultiMeshInstance2D = %Edges
@onready var _highlight: Node2D = %Highlight


func _ready() -> void:
   for item in [_nodes, _edges]:
      var mm := MultiMesh.new()
      mm.transform_format = MultiMesh.TRANSFORM_2D
      mm.use_colors = true
      mm.use_custom_data = item == _nodes
      mm.mesh = QuadMesh.new()
      item.multimesh = mm
      item.material = item.material.duplicate()
   _straight = _edges.material
   _curved.shader = CurveShader
   _highlight.draw.connect(_draw_highlight)
   resized.connect(func(): fit() if not _fitted else _redraw())


## Shows a network: [param header] is a reply of the worker with a drawing, [param payload] its
## parts (node buffer, edge buffer, ids, grid order, grid starts). The view is fitted to the
## network unless [param keep_view] (e.g. after a change of colors).
func show_network(header: Dictionary, payload: PackedByteArray, keep_view := false) -> void:
   var parts := []
   var at := 0
   for n in header.parts:
      parts.append(payload.slice(at, at + int(n)))
      at += int(n)
   _buffer = parts[0].to_float32_array()
   _fill(_nodes.multimesh, int(header.nodes), _buffer)
   _edge_kind(header.get("curved", false), int(header.get("subdivisions", 16)))
   _fill(_edges.multimesh, int(header.edges), parts[1].to_float32_array())
   ids = parts[2].get_string_from_utf8().split(char(31)) if int(header.nodes) > 0 else PackedStringArray()
   _order = parts[3].to_int32_array()
   _starts = parts[4].to_int32_array()
   _grid = header.grid
   var b: Array = header.box
   node_size = header.node_size
   _box = Rect2(b[0], b[1], b[2] - b[0], b[3] - b[1]).grow(header.get("biggest", node_size))
   _extent = Rect2(b[0], b[1], b[2] - b[0], b[3] - b[1]).grow(header.get("biggest", node_size) / 2)
   var aabb := AABB(Vector3(_box.position.x, _box.position.y, -1), Vector3(_box.size.x, _box.size.y, 2))
   _nodes.multimesh.custom_aabb = aabb
   _edges.multimesh.custom_aabb = aabb
   if selected >= ids.size():
      select(-1)
   _highlight.queue_redraw()
   if keep_view:
      _set_zoom(zoom)
   else:
      fit()


## Edges as straight quads, or as Bézier pieces drawn with [param subdivisions] segments each
## (the format of the multimesh can change only while it is empty).
func _edge_kind(curved: bool, subdivisions: int) -> void:
   var mm := _edges.multimesh
   var mesh: Mesh = _strip(subdivisions) if curved else QuadMesh.new()
   if mm.use_custom_data != curved or (curved and mm.mesh != mesh):
      mm.instance_count = 0
      mm.use_custom_data = curved
      mm.mesh = mesh
   _edges.material = _curved if curved else _straight
   _uniforms(zoom)


## Strip of [param n] segments, x from 0 to 1 along it and y from -0.5 to 0.5 across (made once
## per [param n]).
func _strip(n: int) -> ArrayMesh:
   if not _strips.has(n):
      var v := PackedVector2Array()
      var idx := PackedInt32Array()
      for i in n + 1:
         v.append(Vector2(float(i) / n, -0.5))
         v.append(Vector2(float(i) / n, 0.5))
      for i in n:
         idx.append_array([2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2])
      var arrays := []
      arrays.resize(Mesh.ARRAY_MAX)
      arrays[Mesh.ARRAY_VERTEX] = v
      arrays[Mesh.ARRAY_INDEX] = idx
      var mesh := ArrayMesh.new()
      mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
      _strips[n] = mesh
   return _strips[n]


func _fill(mm: MultiMesh, count: int, buffer: PackedFloat32Array) -> void:
   mm.instance_count = count
   if count > 0:
      mm.buffer = buffer


## Sets the background color.
func set_background(color: Color) -> void:
   %Background.color = color
   _redraw()


## Zooms and centers the view so that the whole network (nodes included, with their sizes) fills
## it, [constant FIT_MARGIN] pixels from its borders.
func fit() -> void:
   if size.x < 1 or size.y < 1:
      return
   _fitted = true
   center = _extent.get_center()
   var room := (size - Vector2(2, 2) * FIT_MARGIN).max(Vector2.ONE)
   _set_zoom(minf(room.x / maxf(_extent.size.x, 1e-6), room.y / maxf(_extent.size.y, 1e-6)))


func _set_zoom(z: float) -> void:
   zoom = z
   _uniforms(z)
   _highlight.queue_redraw()
   _redraw()


## Sets the shader parameters for the zoom [param z] (pixels per world unit).
func _uniforms(z: float) -> void:
   _nodes.material.set_shader_parameter("zoom", z)
   _nodes.material.set_shader_parameter("min_pixels", NODE_PIXELS)
   _edges.material.set_shader_parameter("width", maxf(node_size * EDGE_FRACTION, EDGE_PIXELS / z))


## Applies [member center] and [member zoom], and draws the viewport once, at the next frame.
func _redraw() -> void:
   _viewport.canvas_transform = Transform2D(0., Vector2(zoom, zoom), 0., Vector2(_viewport.size) / 2 - center * zoom)
   _viewport.render_target_update_mode = SubViewport.UPDATE_ONCE


## Image of the view as on screen, [param scale] times larger (without the selection ring): drawn
## once by an off-screen viewport sharing the 2D world of the view, over the background color. The
## view itself is not touched.
func capture(scale := 2) -> Image:
   var z := zoom
   var vp := SubViewport.new()
   vp.size = Vector2i(size * scale)
   vp.world_2d = _viewport.world_2d
   var back := CanvasLayer.new()   # the background, behind the shared world (transparent nodes blend with it)
   back.layer = -1
   var rect := ColorRect.new()
   rect.color = %Background.color
   rect.size = Vector2(vp.size)
   back.add_child(rect)
   vp.add_child(back)
   vp.canvas_transform = Transform2D(0., Vector2(z * scale, z * scale), 0., Vector2(vp.size) / 2 - center * z * scale)
   vp.render_target_update_mode = SubViewport.UPDATE_ONCE
   _highlight.visible = false
   _uniforms(z * scale)
   get_tree().root.add_child(vp)
   await RenderingServer.frame_post_draw
   var drawn := vp.get_texture().get_image()
   vp.queue_free()
   _highlight.visible = true
   _set_zoom(z)
   drawn.convert(Image.FORMAT_RGBA8)
   return drawn


## Whether nodes are large enough on screen to be hovered and selected.
func selectable() -> bool:
   return node_size * zoom >= SELECT_PIXELS


## Position in the drawing of the node under [param at] (local coordinates), or -1: among the
## nodes whose disc contains the point, the one drawn on top (the last in the drawing order), as
## seen on screen; only the nodes of the 9 cells of the grid around the point are checked (cells
## are as large as the largest node).
func pick(at: Vector2) -> int:
   if not selectable() or _grid.is_empty():
      return -1
   var w := center + (at - size / 2) / zoom
   var cell: float = _grid.cell
   var gx: int = _grid.gx
   var gy: int = _grid.gy
   var cx := floori((w.x - _grid.x0) / cell)
   var cy := floori((w.y - _grid.y0) / cell)
   var best := -1
   for y in range(maxi(cy - 1, 0), mini(cy + 2, gy)):
      for x in range(maxi(cx - 1, 0), mini(cx + 2, gx)):
         var c := y * gx + x
         for k in range(_starts[c], _starts[c + 1]):
            var i := _order[k]
            var d := w.distance_to(Vector2(_buffer[i * NODE_STRIDE + 3], _buffer[i * NODE_STRIDE + 7]))
            if d <= _buffer[i * NODE_STRIDE] / 2 and i > best:
               best = i
   return best


## Selects the node at position [param index] of the drawing (-1 clears the selection).
func select(index: int) -> void:
   if index == selected:
      return
   selected = index
   _highlight.queue_redraw()
   _redraw()
   node_selected.emit(index, ids[index] if index >= 0 else "")


## Ring around the selected node, 2 pixels wide.
func _draw_highlight() -> void:
   if selected < 0 or selected * NODE_STRIDE + 7 >= _buffer.size():
      return
   var i := selected * NODE_STRIDE
   var p := Vector2(_buffer[i + 3], _buffer[i + 7])
   var r := maxf(_buffer[i], NODE_PIXELS / zoom) / 2 + 2.0 / zoom
   _highlight.draw_arc(p, r, 0, TAU, 48, Color(0.1, 0.1, 0.1), 2.0 / zoom, true)


## Tooltip: the id of the node under the cursor.
func _get_tooltip(at: Vector2) -> String:
   var i := pick(at)
   return ids[i] if i >= 0 else ""


func _gui_input(event: InputEvent) -> void:
   if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
      if event.pressed:
         _press = event.position
         _pressed = true
      elif _pressed:
         _pressed = false
         if event.position.distance_to(_press) <= CLICK_PIXELS:
            select(pick(event.position))
      accept_event()
   elif event is InputEventMouseMotion and event.button_mask & (MOUSE_BUTTON_MASK_LEFT | MOUSE_BUTTON_MASK_MIDDLE):
      center -= event.relative / zoom
      _redraw()
      accept_event()
   elif event is InputEventMouseButton and event.pressed and event.button_index in [MOUSE_BUTTON_WHEEL_UP, MOUSE_BUTTON_WHEEL_DOWN]:
      var factor := ZOOM_STEP if event.button_index == MOUSE_BUTTON_WHEEL_UP else 1.0 / ZOOM_STEP
      var offset: Vector2 = (event.position - size / 2) / zoom
      center += offset - offset / factor   # keeps the point under the cursor still
      _set_zoom(zoom * factor)
      accept_event()
