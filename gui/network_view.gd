extends SubViewportContainer
## Visualization panel of one network: its nodes and edges, as two MultiMesh instances (one draw
## call each), filled with the buffers computed by the Python worker. Drag the background to pan,
## scroll to zoom; the view can be rotated ([member turn]) and the layout scaled ([member spread]:
## the distances between nodes grow, not their size on screen). When nodes are large enough on screen
## ([constant SELECT_PIXELS]):
##
## - hovering a node shows its id; clicking it selects it (clicking elsewhere clears the
##   selection), and Shift-clicking adds it to the selection or takes it out;
## - clicking an edge (within [constant EDGE_PICK_PIXELS] pixels; in matrix views, its cell)
##   selects it, Shift-clicking adds it to the selection or takes it out;
## - Shift-dragging on the background (at any zoom) draws a rectangle, which adds the nodes in it
##   and the edges crossing it to the selection;
## - dragging a node moves it, with the other selected nodes if it is one of them (not in matrix
##   views, nor when the positions come from node attributes): the nodes and their straight edges
##   follow the mouse, and the new positions go to the worker when the mouse is released
##   ([signal nodes_moved]).
##
## Right clicking asks for the menu of the view ([signal context_requested]: on a node, to add an
## edge from it, elsewhere to add a node there). While an edge is added ([method start_link]), a
## line follows the mouse from its source; clicking a node ends it there ([signal link_chosen]),
## clicking elsewhere, right clicking or Escape cancels it.
##
## Self loops are circles above their node (loop.gdshader), drawn from a buffer of the worker
## (one instance per loop), which follow the node when it is dragged.
##
## In directed networks the edges have arrowheads (arrow.gdshader, in the color of their line, at
## the border of the node they point to), from a buffer of the worker; the edges between the same
## two nodes share one line, with an arrowhead per direction, and straight lines stop at the base
## of their arrowheads (edge.gdshader, from the diameters of their nodes in the edge buffer).
## Arrowheads follow dragged nodes.
##
## Selected nodes get rings drawn by the GPU (ring.gdshader: any number of them); selected edges
## are drawn over, as lines, curves or cell outlines (the first [constant MAX_HIGHLIGHTS]). Nodes can also be colored by
## the states of an epidemic ([method show_states]), over the colors of the style.
##
## Node charts (Visual elements › Nodes › Chart): pie charts drawn by the node shader, from the
## slices sent by the worker (a texture, [method _show_charts]); epidemic states hide them.
##
## Node labels (Visual elements › Nodes › Label, off by default): the texts sent by the worker,
## centered on their nodes, upright, in black or white with an outline (by the background), at
## the heights sent by the worker (Label size: drawing units, like node diameters, so that labels
## grow and shrink with the zoom). Labels smaller than [constant LABEL_MIN_PIXELS] pixels on
## screen are left out, and none are drawn when more than [constant MAX_LABELS] nodes are in view:
## zoom in to see the labels of a large network. Saved images have them too.
##
## The viewport is drawn only after a change (new drawing, pan, zoom, resize, selection, drag), so
## an idle view costs nothing and does not compete with the worker for the processor or the GPU.
## (A Camera2D would make Godot redraw at every frame: the view is moved with the canvas transform
## instead.)

## Emitted when the selected nodes or edges change.
signal selection_changed
## Emitted when dragged nodes are dropped: their positions in the drawing order, and their new
## positions (drawing coordinates, y down).
signal nodes_moved(nodes: PackedInt32Array, positions: PackedVector2Array)
## Emitted on a right click: the node under the mouse (its position in the drawing order, or -1),
## the point in drawing coordinates (y down) and on the screen.
signal context_requested(node: int, at: Vector2, screen: Vector2)
## Emitted when the target of a new edge is clicked ([method start_link]), or with -1 when the
## new edge is canceled.
signal link_chosen(source: int, target: int)

## Minimum diameter of a node on screen, in pixels.
const NODE_PIXELS := 3.0
## Width of the edges, as a fraction of the diameter of the nodes.
const EDGE_FRACTION := 1.0 / 6.0
## Minimum width of an edge on screen, in pixels.
const EDGE_PIXELS := 1.0
## Floats per node, per edge and per Bézier piece of bundled edges in the buffers of the worker
## (edges of directed networks have 16, with the diameters of their nodes and their arrowheads).
const NODE_STRIDE := 16
## Texels per row of the texture of the slices of the node charts (CHART_WIDTH of the worker).
const CHART_WIDTH := 4096
## Colors of the node labels from the style ([code]label_color[/code], one per node, sent with
## their heights), or empty: black or white by the background.
var _label_inks := PackedColorArray()
const EDGE_STRIDE := 12
const CURVE_STRIDE := 16
## Shader of bundled edges (cubic Bézier pieces computed by the GPU).
const CurveShader := preload("res://curve.gdshader")
## Smallest default diameter of the nodes on screen, in pixels, at which they can be hovered,
## selected and dragged.
const SELECT_PIXELS := 6.0
## Largest distance, in pixels, from the cursor to an edge that is picked.
const EDGE_PICK_PIXELS := 4.0
## Points per Bézier piece of bundled edges in picking and highlighting.
const CURVE_POINTS := 9
## Largest move of the mouse, in pixels, between press and release that counts as a click.
const CLICK_PIXELS := 4.0
## Zoom factor of one wheel step.
const ZOOM_STEP := 1.15
## Empty space around the network when it is fitted to the view, in pixels.
const FIT_MARGIN := 8.0
## Most selected edges highlighted (beyond, the first ones only).
const MAX_HIGHLIGHTS := 20000
## Smallest length of an arrowhead on screen, in pixels, and its length and width in widths of its
## line (arrow.gdshader; edge.gdshader stops the lines at their base).
const ARROW_PIXELS := 5.0
const ARROW_LENGTH := 2.5
const ARROW_BREADTH := 2.2
## Supersampling of saved images and animation frames ([method capture]): drawn this many times
## larger, then reduced, which smooths the edges.
const SMOOTH := 3
## Largest side of the off-screen drawing of a saved image, in pixels (a limit of the GPU).
const MAX_CAPTURE := 16384
## Color of the selection marks.
const MARK_COLOR := Color(0.1, 0.1, 0.1)
## Smallest font size of the node labels drawn, in pixels, and the most labels drawn (beyond,
## none: zoom in).
const LABEL_MIN_PIXELS := 4
const MAX_LABELS := 2000

## World point at the center of the view.
var center := Vector2.ZERO
## Pixels per world unit.
var zoom := 1.0
## Rotation of the view, in degrees, counterclockwise (a visual setting: positions do not change).
var turn := 0.0
## Scale of the layout (a visual setting: positions do not change): the view is zoomed by it, and
## nodes and edges are drawn this many times thinner in world units, so that they keep their size
## on screen while the distances between them grow.
var spread := 1.0
## Default diameter of the nodes in world units (each node has its own, in the buffer).
var node_size := 0.5
## Input ids of the nodes, in the drawing order.
var ids := PackedStringArray()
## Labels of the nodes, in the drawing order (empty: no labels).
var labels := PackedStringArray()
## Heights of the labels, in world units (as node diameters: divided by the scale of the layout).
var _label_heights := PackedFloat32Array()
var _labels := Node2D.new()
## What nodes can be added (from the worker: false, or the layers and node types the menu offers).
var adding = false
## Positions (in the drawing order of the worker) of the selected edges.
var selected_edges := PackedInt32Array()
var _chosen := {}
var _box := Rect2(0, 0, 1, 1)
## Box of the node positions, and the largest node diameter (world units).
var _extent := Rect2(0, 0, 1, 1)
var _biggest := 0.5
var _fitted := false
var _buffer := PackedFloat32Array()
var _edge_data := PackedFloat32Array()
var _src := PackedInt32Array()
var _dst := PackedInt32Array()
var _kind := "lines"
var _grid := {}
var _order := PackedInt32Array()
var _starts := PackedInt32Array()
var _press := Vector2.ZERO
var _pressed := false
var _mode := ""
var _press_node := -1
var _press_edge := -1
var _movable := true
var _style_buffer := PackedFloat32Array()
var _pieces := {}
var _rect_end := Vector2.ZERO
var _drag_edges := PackedInt32Array()
var _straight: Material
var _curved := ShaderMaterial.new()
var _strips := {}
@onready var _viewport: SubViewport = $Viewport
@onready var _nodes: MultiMeshInstance2D = %Nodes
@onready var _edges: MultiMeshInstance2D = %Edges
@onready var _highlight: Node2D = %Highlight
@onready var _rings: MultiMeshInstance2D = %Rings
@onready var _loops: MultiMeshInstance2D = %Loops
@onready var _arrows: MultiMeshInstance2D = %Arrows
var _loop_data := PackedFloat32Array()
var _arrow_data := PackedFloat32Array()
var _drag_arrows := PackedInt32Array()
## Floats per edge in the edge buffer of straight lines and cells.
var _edge_stride := EDGE_STRIDE
var _rings_dirty := true
## Source of the edge being added, or -1, and the mouse.
var _link := -1
var _link_end := Vector2.ZERO


func _ready() -> void:
   for item in [_nodes, _edges, _rings, _loops, _arrows]:
      var mm := MultiMesh.new()
      mm.transform_format = MultiMesh.TRANSFORM_2D
      mm.use_colors = true
      mm.use_custom_data = item != _edges
      mm.mesh = _triangle() if item == _arrows else QuadMesh.new()
      item.multimesh = mm
      item.material = item.material.duplicate()
   _straight = _edges.material
   _curved.shader = CurveShader
   _highlight.draw.connect(_draw_highlight)
   _viewport.add_child(_labels)   # over the nodes and the selection
   _labels.draw.connect(_draw_labels)
   resized.connect(func(): fit() if not _fitted else _redraw())


## Shows a network: [param header] is a reply of the worker with a drawing, [param payload] its
## parts (node buffer, edge buffer, ids, grid order, grid starts, sources and targets of the
## edges, self loop buffer, arrowhead buffer). The view is fitted to the network unless [param keep_view] (e.g. after a change of
## colors). The selection is kept while its nodes and edges exist.
func show_network(header: Dictionary, payload: PackedByteArray, keep_view := false) -> void:
   var parts := []
   var at := 0
   for n in header.parts:
      parts.append(payload.slice(at, at + int(n)))
      at += int(n)
   _buffer = parts[0].to_float32_array()
   _fill(_nodes.multimesh, int(header.nodes), _buffer)
   _style_buffer = _buffer.duplicate()
   _pieces = {}
   _kind = "cells" if header.get("cells", false) else ("curves" if header.get("curved", false) else "lines")
   adding = header.get("adding", false)
   _movable = header.get("movable", _kind != "cells")   # not matrix views, nor positions from attributes
   _edge_stride = int(header.get("stride", EDGE_STRIDE))
   _edge_kind(_kind, int(header.get("subdivisions", 16)), _kind == "lines" and _edge_stride > EDGE_STRIDE)
   _edge_data = parts[1].to_float32_array()
   _fill(_edges.multimesh, int(header.edges), _edge_data)
   ids = parts[2].get_string_from_utf8().split(char(31)) if int(header.nodes) > 0 else PackedStringArray()
   _order = parts[3].to_int32_array()
   _starts = parts[4].to_int32_array()
   _src = parts[5].to_int32_array() if parts.size() > 6 else PackedInt32Array()
   _dst = parts[6].to_int32_array() if parts.size() > 6 else PackedInt32Array()
   _loop_data = parts[7].to_float32_array() if int(header.get("loops", 0)) > 0 else PackedFloat32Array()
   _fill(_loops.multimesh, int(header.get("loops", 0)), _loop_data)
   _arrow_data = parts[8].to_float32_array() if int(header.get("arrows", 0)) > 0 else PackedFloat32Array()
   labels = parts[9].get_string_from_utf8().split(char(31)) if header.get("labels", false) and parts.size() > 10 else PackedStringArray()
   _label_heights = parts[10].to_float32_array() if not labels.is_empty() else PackedFloat32Array()
   _label_inks = PackedColorArray()
   if header.get("label_inks", false) and not labels.is_empty():   # five floats per label: height, color
      var both := _label_heights
      _label_heights = PackedFloat32Array()
      _label_heights.resize(both.size() / 5)
      _label_inks.resize(both.size() / 5)
      for i in _label_heights.size():
         _label_heights[i] = both[5 * i]
         _label_inks[i] = Color(both[5 * i + 1], both[5 * i + 2], both[5 * i + 3], both[5 * i + 4])
   _show_charts(parts[11] if parts.size() > 11 else PackedByteArray())
   _show_texture(_nodes.material, "borders", "bordered", parts[12] if parts.size() > 12 else PackedByteArray(), Image.FORMAT_RGBAF, 16)
   _show_texture(_edges.material, "styles", "styled", parts[13] if parts.size() > 13 else PackedByteArray(), Image.FORMAT_RF, 4)
   _labels.queue_redraw()
   _fill(_arrows.multimesh, int(header.get("arrows", 0)), _arrow_data)
   _grid = header.grid
   var b: Array = header.box
   node_size = header.node_size
   _box = Rect2(b[0], b[1], b[2] - b[0], b[3] - b[1]).grow(header.get("biggest", node_size))
   _extent = Rect2(b[0], b[1], b[2] - b[0], b[3] - b[1])
   _biggest = header.get("biggest", node_size)
   var aabb := AABB(Vector3(_box.position.x, _box.position.y, -1), Vector3(_box.size.x, _box.size.y, 2))
   _nodes.multimesh.custom_aabb = aabb
   _edges.multimesh.custom_aabb = aabb
   _loops.multimesh.custom_aabb = aabb.grow(header.get("biggest", node_size) * 2)
   _arrows.multimesh.custom_aabb = aabb
   var n := ids.size()
   var kept := {}
   for i in _chosen:
      if i < n:
         kept[i] = true
   var edges := PackedInt32Array(Array(selected_edges).filter(func(e): return e < _src.size()))
   if kept.size() != _chosen.size() or edges.size() != selected_edges.size():
      _chosen = kept
      selected_edges = edges
      selection_changed.emit()
   _rings_dirty = true
   _highlight.queue_redraw()
   if keep_view:
      _set_zoom(zoom)
   else:
      fit()


## The slices of the node charts ([param table]: float32 texels, end, red, green, blue, in rows of
## [constant CHART_WIDTH]) as the texture of the node shader; the nodes say where their slices are.
func _show_charts(table: PackedByteArray) -> void:
   var texels := table.size() / 16
   if texels == 0:
      _nodes.material.set_shader_parameter("slices", null)
      return
   var image := Image.create_from_data(CHART_WIDTH, texels / CHART_WIDTH, false, Image.FORMAT_RGBAF, table)
   _nodes.material.set_shader_parameter("slices", ImageTexture.create_from_image(image))
   _nodes.material.set_shader_parameter("chart_width", CHART_WIDTH)


## A texture of one texel per node or edge (rows of [constant CHART_WIDTH], [param bytes] per texel)
## as the uniform [param uniform] of [param material], and [param flag] whether there is one: the
## node borders and the edge line styles.
func _show_texture(material: ShaderMaterial, uniform: String, flag: String, table: PackedByteArray, format: int, bytes: int) -> void:
   if material == null:   # matrix cells
      return
   var texels := table.size() / bytes
   material.set_shader_parameter(flag, texels > 0)
   material.set_shader_parameter(uniform, ImageTexture.create_from_image(Image.create_from_data(CHART_WIDTH, texels / CHART_WIDTH, false, format, table)) if texels > 0 else null)
   material.set_shader_parameter("chart_width", CHART_WIDTH)


## Edges as straight quads ([param kind] "lines", the edge shader setting their width), as Bézier
## pieces drawn with [param subdivisions] segments each ("curves"), or as the unit squares of the
## cells of a matrix ("cells", without shader); lines of directed networks have custom data
## ([param ends]: their arrowheads). The format of the multimesh can change only while it is
## empty.
func _edge_kind(kind: String, subdivisions: int, ends := false) -> void:
   var mm := _edges.multimesh
   var curved := kind == "curves"
   var custom := curved or ends
   var mesh: Mesh = _strip(subdivisions) if curved else QuadMesh.new()
   if mm.use_custom_data != custom or (curved and mm.mesh != mesh) or (not curved and mm.mesh is ArrayMesh):
      mm.instance_count = 0
      mm.use_custom_data = custom
      mm.mesh = mesh
   _edges.material = {"lines": _straight, "curves": _curved, "cells": null}[kind]
   _uniforms(zoom)


## Triangle of the arrowheads: tip at (0, 0), base from (-1, -0.5) to (-1, 0.5).
func _triangle() -> ArrayMesh:
   var arrays := []
   arrays.resize(Mesh.ARRAY_MAX)
   arrays[Mesh.ARRAY_VERTEX] = PackedVector2Array([Vector2.ZERO, Vector2(-1, -0.5), Vector2(-1, 0.5)])
   var mesh := ArrayMesh.new()
   mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
   return mesh


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


## Zooms and centers the view so that the whole network (nodes included, with their sizes as
## drawn at the current [member spread], and self loops, as turned) fills it, [constant FIT_MARGIN] pixels from its
## borders.
func fit() -> void:
   if size.x < 1 or size.y < 1:
      return
   _fitted = true
   var extent := _extent.grow(_biggest * (1.0 if _loops.multimesh.instance_count > 0 else 0.5) / spread)   # loops rise about a diameter above their node
   center = extent.get_center()
   var half := extent.size / 2
   var r := deg_to_rad(turn)
   var span := Vector2(absf(half.x * cos(r)) + absf(half.y * sin(r)), absf(half.x * sin(r)) + absf(half.y * cos(r))) * 2
   var room := (size - Vector2(2, 2) * FIT_MARGIN).max(Vector2.ONE)
   _set_zoom(minf(room.x / maxf(span.x, 1e-6), room.y / maxf(span.y, 1e-6)))


## Puts back a view as it was ([param c] center, [param z] zoom, [param t] rotation, [param s]
## scale of the layout; e.g. from a session): the next drawing keeps it (shown with keep_view)
## instead of fitting the network.
func restore_view(c: Vector2, z: float, t: float, s: float) -> void:
   center = c
   zoom = z
   turn = t
   spread = s
   _fitted = true


## Sets the scale of the layout to [param factor] (1: the distances of the layout), keeping the
## center of the view and the size of the nodes on screen.
func set_spread(factor: float) -> void:
   var z := zoom * factor / spread
   spread = factor
   _set_zoom(z)


## Turns the view by [param degrees] (counterclockwise), around its center.
func set_turn(degrees: float) -> void:
   turn = degrees
   _nodes.material.set_shader_parameter("turn", deg_to_rad(turn))   # node charts start at the top of the screen
   _redraw()


func _set_zoom(z: float) -> void:
   zoom = z
   _uniforms(z)
   _highlight.queue_redraw()
   _redraw()


## Sets the shader parameters for the zoom [param z] (pixels per world unit), with smallest sizes
## on screen [param pixels] times larger (images drawn larger to be reduced, see [method capture]).
func _uniforms(z: float, pixels := 1.0) -> void:
   var width := maxf(node_size * EDGE_FRACTION / spread, EDGE_PIXELS * pixels / z)
   for m: ShaderMaterial in [_straight, _arrows.material]:
      m.set_shader_parameter("arrow_min", ARROW_PIXELS * pixels)
      m.set_shader_parameter("arrow_length", ARROW_LENGTH)
      m.set_shader_parameter("zoom", z)
      m.set_shader_parameter("min_pixels", NODE_PIXELS * pixels)
      m.set_shader_parameter("spread", spread)
   _arrows.material.set_shader_parameter("arrow_breadth", ARROW_BREADTH)
   for item in [_nodes, _rings, _arrows]:
      item.material.set_shader_parameter("zoom", z)
      item.material.set_shader_parameter("min_pixels", NODE_PIXELS * pixels)
      item.material.set_shader_parameter("spread", spread)
   if _edges.material is ShaderMaterial:
      _edges.material.set_shader_parameter("width", width)
   _loops.material.set_shader_parameter("width", width)
   _arrows.material.set_shader_parameter("width", width)
   _loops.material.set_shader_parameter("spread", spread)


## The canvas transform of a view of [param view_size] pixels at zoom [param z]: the world turned
## by [member turn] (counterclockwise on screen, y down) and scaled around [member center].
func _view_transform(view_size: Vector2, z: float) -> Transform2D:
   var r := -deg_to_rad(turn)
   return Transform2D(r, Vector2(z, z), 0., view_size / 2 - (center * z).rotated(r))


## World point under the point [param q] of the view.
func _to_world(q: Vector2) -> Vector2:
   return center + ((q - size / 2) / zoom).rotated(deg_to_rad(turn))


## Point of the view of the world point [param p].
func _to_screen(p: Vector2) -> Vector2:
   return ((p - center) * zoom).rotated(-deg_to_rad(turn)) + size / 2


## Applies [member center], [member zoom] and [member turn], and draws the viewport once, at the
## next frame.
func _redraw() -> void:
   _viewport.canvas_transform = _view_transform(Vector2(_viewport.size), zoom)
   if not labels.is_empty():
      _labels.queue_redraw()
   _viewport.render_target_update_mode = SubViewport.UPDATE_ONCE


## Image of the view as on screen, [param scale] times larger (without the selection marks):
## drawn once by an off-screen viewport sharing the 2D world of the view, over the background
## color. The view itself is not touched. The image is antialiased by supersampling: drawn
## [param smooth] times larger still (smallest sizes included; less if the drawing would be
## larger than [constant MAX_CAPTURE] pixels), then reduced (Lanczos filter); 1 draws it as on
## screen, with jagged edges.
func capture(scale: float = 2.0, smooth := SMOOTH) -> Image:
   var z := zoom
   var final := Vector2i(size * scale)
   smooth = clampi(MAX_CAPTURE / maxi(maxi(final.x, final.y), 1), 1, smooth)
   scale *= smooth
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
   vp.render_target_update_mode = SubViewport.UPDATE_ONCE
   _highlight.visible = false
   _rings.visible = false
   _uniforms(z * scale, smooth)
   get_tree().root.add_child(vp)
   vp.canvas_transform = _view_transform(Vector2(vp.size), z * scale)   # once in the tree, with its canvas
   await RenderingServer.frame_post_draw
   var drawn := vp.get_texture().get_image()
   vp.queue_free()
   _highlight.visible = true
   _rings.visible = true
   _set_zoom(z)
   drawn.convert(Image.FORMAT_RGBA8)
   if smooth > 1:
      drawn.resize(final.x, final.y, Image.INTERPOLATE_LANCZOS)
   return drawn


## Whether nodes can be added: node-link drawings with the positions of a layout (not matrix views,
## nor positions from node attributes).
func can_add_nodes() -> bool:
   return _movable and _kind != "cells"


## Whether nodes are large enough on screen to be hovered, selected and dragged.
func selectable() -> bool:
   return node_size * zoom / spread >= SELECT_PIXELS


## Position in the drawing of the node under [param at] (local coordinates), or -1: among the
## nodes whose disc contains the point, the one drawn on top (the last in the drawing order), as
## seen on screen; only the nodes of the 9 cells of the grid around the point are checked (cells
## are as large as the largest node).
func pick(at: Vector2) -> int:
   if not selectable() or _grid.is_empty():
      return -1
   var w := _to_world(at)
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
            if d <= _buffer[i * NODE_STRIDE] / 2 / spread and i > best:
               best = i
   return best


## Centers the view on the node [param i] (a position in the drawing order), zooming in if the
## nodes are too small there to be told apart and selected.
func focus(i: int) -> void:
   if i < 0 or i >= ids.size():
      return
   center = _position(i)
   if not selectable():
      _set_zoom(2.0 * SELECT_PIXELS * spread / maxf(node_size, 1e-9))
   _redraw()


## Positions in the drawing of the selected nodes, in increasing order.
func selected_nodes() -> PackedInt32Array:
   var nodes := PackedInt32Array(_chosen.keys())
   nodes.sort()
   return nodes


## Selects the nodes [param nodes] and the edges [param edges] (positions in the drawing order).
func set_selection(nodes: PackedInt32Array, edges := PackedInt32Array()) -> void:
   _chosen = {}
   for i in nodes:
      if i >= 0 and i < ids.size():
         _chosen[i] = true
   selected_edges = edges
   _rings_dirty = true
   _highlight.queue_redraw()
   _redraw()
   selection_changed.emit()


## Selects the node at position [param index] of the drawing alone (-1 clears the selection).
func select(index: int) -> void:
   set_selection(PackedInt32Array([index]) if index >= 0 else PackedInt32Array())


## The rings of the selected nodes: one instance each, with the transform and the diameter of the
## node (from the node buffer) and the color of the marks.
func _fill_rings() -> void:
   var data := PackedFloat32Array()
   data.resize(_chosen.size() * NODE_STRIDE)
   var j := 0
   for i in _chosen:
      if i * NODE_STRIDE + NODE_STRIDE > _buffer.size():
         continue
      for f in NODE_STRIDE:
         data[j * NODE_STRIDE + f] = _buffer[i * NODE_STRIDE + f]
      data[j * NODE_STRIDE + 8] = MARK_COLOR.r
      data[j * NODE_STRIDE + 9] = MARK_COLOR.g
      data[j * NODE_STRIDE + 10] = MARK_COLOR.b
      data[j * NODE_STRIDE + 11] = 1.0
      j += 1
   _fill(_rings.multimesh, j, data.slice(0, j * NODE_STRIDE))
   _rings.multimesh.custom_aabb = _nodes.multimesh.custom_aabb


## Lines on the selected edges (2 pixels wide), and the rectangle being drawn; the rings of the
## selected nodes are filled again if the selection or the positions changed (not for a zoom: the
## ring shader follows it).
func _draw_highlight() -> void:
   var w := 2.0 / zoom
   if _rings_dirty:
      _rings_dirty = false
      _fill_rings()
   for e in selected_edges.slice(0, MAX_HIGHLIGHTS):
      if e < _src.size():
         for line in _edge_lines(e):
            _highlight.draw_polyline(line, MARK_COLOR, w)
   if _link >= 0:
      _highlight.draw_line(_position(_link), _to_world(_link_end), MARK_COLOR, w)
   if _mode == "rect":
      var a := _to_world(_press)
      var c := _to_world(_rect_end)
      var b := _to_world(Vector2(_rect_end.x, _press.y))
      var d := _to_world(Vector2(_press.x, _rect_end.y))
      _highlight.draw_polyline(PackedVector2Array([a, b, c, d, a]), MARK_COLOR, w / 2)


## Draws the labels of the nodes in view (none if more than [constant MAX_LABELS]), each upright,
## centered on its node, at its height (none below [constant LABEL_MIN_PIXELS] pixels).
func _draw_labels() -> void:
   if labels.is_empty() or labels.size() != ids.size() or _label_heights.size() != ids.size():
      return
   var xf := _view_transform(size, zoom)
   var room := Rect2(Vector2.ZERO, size).grow(40)
   var shown := PackedInt32Array()
   for i in labels.size():
      if not labels[i].is_empty() and room.has_point(xf * _position(i)):
         shown.append(i)
         if shown.size() > MAX_LABELS:
            return
   var font := ThemeDB.fallback_font
   var dark: bool = %Background.color.get_luminance() > 0.5
   var colored := _label_inks.size() == labels.size()
   for i in shown:
      var ink := (Color.BLACK if dark else Color.WHITE) if not colored else _label_inks[i]
      var light := dark if not colored else ink.get_luminance() < 0.5   # a halo that stands out from the color
      var halo := Color(1, 1, 1, 0.8) if light else Color(0, 0, 0, 0.8)
      var px := roundi(_label_heights[i] * zoom / spread)
      if px < LABEL_MIN_PIXELS:
         continue
      _labels.draw_set_transform(_position(i), deg_to_rad(turn), Vector2.ONE / zoom)
      var width := font.get_string_size(labels[i], HORIZONTAL_ALIGNMENT_LEFT, -1, px).x
      var at := Vector2(-width / 2, (font.get_ascent(px) - font.get_descent(px)) / 2)   # centered on the node
      _labels.draw_string_outline(font, at, labels[i], HORIZONTAL_ALIGNMENT_LEFT, -1, px, maxi(2, px / 5), halo)
      _labels.draw_string(font, at, labels[i], HORIZONTAL_ALIGNMENT_LEFT, -1, px, ink)
   _labels.draw_set_transform(Vector2.ZERO)


## Position in the drawing of the edge under [param at] (local coordinates), or -1: the nearest
## edge within [constant EDGE_PICK_PIXELS] pixels (in matrix views, the edge of the cell under the
## point), when nodes are large enough on screen to be selected ([method selectable]).
func pick_edge(at: Vector2) -> int:
   if not selectable() or _src.is_empty():
      return -1
   var w := _to_world(at)
   var m := _src.size()
   var best := -1
   var reach := EDGE_PICK_PIXELS / zoom
   if _kind == "cells":
      for c in _edge_data.size() / EDGE_STRIDE:
         var k := c * EDGE_STRIDE
         if absf(w.x - _edge_data[k + 3]) <= 0.5 and absf(w.y - _edge_data[k + 7]) <= 0.5:
            return c % m
   elif _kind == "curves":
      for p in _edge_data.size() / CURVE_STRIDE:
         var line := _piece(p)
         for j in line.size() - 1:
            var d := Geometry2D.get_closest_point_to_segment(w, line[j], line[j + 1]).distance_to(w)
            if d < reach:
               reach = d
               best = int(_edge_data[p * CURVE_STRIDE + 15])
   else:
      for e in m:
         var d := Geometry2D.get_closest_point_to_segment(w, _position(_src[e]), _position(_dst[e])).distance_to(w)
         if d < reach:
            reach = d
            best = e
   return best


## Positions in the drawing of the edges that cross the rectangle [param area] (local
## coordinates): edges with a point in it, or crossing one of its sides; in matrix views, edges
## with a cell overlapping it.
func edges_in(area: Rect2) -> PackedInt32Array:
   var found := {}
   var m := _src.size()
   var corners := [area.position, Vector2(area.end.x, area.position.y), area.end, Vector2(area.position.x, area.end.y)]
   var crosses := func(a: Vector2, b: Vector2) -> bool:
      if area.has_point(a) or area.has_point(b):
         return true
      for j in 4:
         if Geometry2D.segment_intersects_segment(a, b, corners[j], corners[(j + 1) % 4]) != null:
            return true
      return false
   if _kind == "cells":
      for c in _edge_data.size() / EDGE_STRIDE:
         var k := c * EDGE_STRIDE
         var o := Vector2(_edge_data[k + 3], _edge_data[k + 7])
         var cell := Rect2(_to_screen(o), Vector2.ZERO)
         for corner in [Vector2(-0.5, -0.5), Vector2(0.5, -0.5), Vector2(0.5, 0.5), Vector2(-0.5, 0.5)]:
            cell = cell.expand(_to_screen(o + corner))
         if cell.intersects(area):
            found[c % m] = true
   elif _kind == "curves":
      for p in _edge_data.size() / CURVE_STRIDE:
         var line := _piece(p)
         for j in line.size() - 1:
            if crosses.call(_to_screen(line[j]), _to_screen(line[j + 1])):
               found[int(_edge_data[p * CURVE_STRIDE + 15])] = true
               break
   else:
      for e in m:
         if crosses.call(_to_screen(_position(_src[e])), _to_screen(_position(_dst[e]))):
            found[e] = true
   var edges := PackedInt32Array(found.keys())
   edges.sort()
   return edges


## Points of the Bézier piece [param p] of bundled edges (world coordinates).
func _piece(p: int) -> PackedVector2Array:
   var k := p * CURVE_STRIDE
   var p0 := Vector2(_edge_data[k + 3], _edge_data[k + 7])
   var p1 := Vector2(_edge_data[k], _edge_data[k + 4])
   var p2 := Vector2(_edge_data[k + 1], _edge_data[k + 5])
   var p3 := Vector2(_edge_data[k + 12], _edge_data[k + 13])
   var line := PackedVector2Array()
   for j in CURVE_POINTS:
      var t := float(j) / (CURVE_POINTS - 1)
      var u := 1.0 - t
      line.append(u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3)
   return line


## Lines drawing the edge [param e] (world coordinates): a segment, the pieces of its curve, or the
## outlines of its cells.
func _edge_lines(e: int) -> Array:
   if _kind == "cells":
      var lines := []
      var m := _src.size()
      for c in [e, e + m]:
         if c * EDGE_STRIDE < _edge_data.size():
            var o := Vector2(_edge_data[c * EDGE_STRIDE + 3], _edge_data[c * EDGE_STRIDE + 7])
            lines.append(PackedVector2Array([o + Vector2(-0.5, -0.5), o + Vector2(0.5, -0.5), o + Vector2(0.5, 0.5), o + Vector2(-0.5, 0.5), o + Vector2(-0.5, -0.5)]))
      return lines
   if _kind == "curves":
      if _pieces.is_empty():
         for p in _edge_data.size() / CURVE_STRIDE:
            var f := int(_edge_data[p * CURVE_STRIDE + 15])
            if not _pieces.has(f):
               _pieces[f] = []
            _pieces[f].append(p)
      return _pieces.get(e, []).map(func(p): return _piece(p))
   return [PackedVector2Array([_position(_src[e]), _position(_dst[e])])]


## Colors the nodes by state: node [code]i[/code] gets [code]palette[codes[i]][/code] (codes beyond
## the palette, e.g. 255 for nodes without a state, keep the color of the style), with the opacity
## of the style; only the [param nodes] given, or all of them with [param all].
func show_states(codes: PackedByteArray, palette: Array, nodes := PackedInt32Array(), all := true) -> void:
   var count := ids.size() if all else nodes.size()
   for j in count:
      var i: int = j if all else nodes[j]
      var k := i * NODE_STRIDE
      var c: int = codes[i] if i < codes.size() else 255
      if c < palette.size():
         var color: Color = palette[c]
         _buffer[k + 8] = color.r
         _buffer[k + 9] = color.g
         _buffer[k + 10] = color.b
         _buffer[k + 14] = 0.0   # the state, not the chart
      else:
         _buffer[k + 8] = _style_buffer[k + 8]
         _buffer[k + 9] = _style_buffer[k + 9]
         _buffer[k + 10] = _style_buffer[k + 10]
         _buffer[k + 14] = _style_buffer[k + 14]
   _nodes.multimesh.buffer = _buffer
   _redraw()


## Gives the nodes back the colors of the style (after [method show_states]).
func hide_states() -> void:
   for i in ids.size():
      var k := i * NODE_STRIDE
      _buffer[k + 8] = _style_buffer[k + 8]
      _buffer[k + 9] = _style_buffer[k + 9]
      _buffer[k + 10] = _style_buffer[k + 10]
      _buffer[k + 14] = _style_buffer[k + 14]
   _nodes.multimesh.buffer = _buffer
   _redraw()


func _position(i: int) -> Vector2:
   return Vector2(_buffer[i * NODE_STRIDE + 3], _buffer[i * NODE_STRIDE + 7])


## Tooltip: the id of the node under the cursor.
func _get_tooltip(at: Vector2) -> String:
   var i := pick(at)
   return ids[i] if i >= 0 else ""


## Starts adding an edge from the node [param source] (a position in the drawing order): a line
## follows the mouse until a node is clicked.
func start_link(source: int) -> void:
   _link = source
   _link_end = get_local_mouse_position()
   _highlight.queue_redraw()
   _redraw()


func _end_link(target: int) -> void:
   var source := _link
   _link = -1
   _highlight.queue_redraw()
   _redraw()
   link_chosen.emit(source, target)


func _input(event: InputEvent) -> void:
   if _link >= 0 and event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
      _end_link(-1)
      get_viewport().set_input_as_handled()


func _gui_input(event: InputEvent) -> void:
   if _link >= 0:   # adding an edge
      if event is InputEventMouseMotion:
         _link_end = event.position
         _highlight.queue_redraw()
         _redraw()
      elif event is InputEventMouseButton and event.pressed and event.button_index in [MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT]:
         _end_link(pick(event.position) if event.button_index == MOUSE_BUTTON_LEFT else -1)
         accept_event()
         return
      if not (event is InputEventMouseButton and event.button_index in [MOUSE_BUTTON_WHEEL_UP, MOUSE_BUTTON_WHEEL_DOWN]):
         return
   if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_RIGHT and event.pressed:
      context_requested.emit(pick(event.position), _to_world(event.position), get_screen_position() + event.position)
      accept_event()
   elif event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
      if event.pressed:
         _press = event.position
         _pressed = true
         _press_node = pick(event.position)
         _press_edge = pick_edge(event.position) if _press_node < 0 else -1
         _mode = "node" if _press_node >= 0 else ("edge" if _press_edge >= 0 else ("rect" if event.shift_pressed else "pan"))
         _rect_end = _press
      elif _pressed:
         _pressed = false
         _release(event)
      accept_event()
   elif event is InputEventMouseMotion and event.button_mask & MOUSE_BUTTON_MASK_LEFT and _pressed:
      _drag(event)
      accept_event()
   elif event is InputEventMouseMotion and event.button_mask & MOUSE_BUTTON_MASK_MIDDLE:
      _pan(event.relative)
      accept_event()
   elif event is InputEventMouseButton and event.pressed and event.button_index in [MOUSE_BUTTON_WHEEL_UP, MOUSE_BUTTON_WHEEL_DOWN]:
      var factor := ZOOM_STEP if event.button_index == MOUSE_BUTTON_WHEEL_UP else 1.0 / ZOOM_STEP
      var offset: Vector2 = ((event.position - size / 2) / zoom).rotated(deg_to_rad(turn))
      center += offset - offset / factor   # keeps the point under the cursor still
      _set_zoom(zoom * factor)
      accept_event()


func _pan(relative: Vector2) -> void:
   center -= (relative / zoom).rotated(deg_to_rad(turn))
   _redraw()


## Mouse moved with the left button down: pans, drags nodes or draws the rectangle.
func _drag(event: InputEventMouseMotion) -> void:
   if _mode == "edge" and event.position.distance_to(_press) > CLICK_PIXELS:   # not a click on the edge
      _mode = "rect" if event.shift_pressed else "pan"
      if _mode == "pan":
         _pan(event.position - _press - event.relative)
   if _mode == "pan":
      _pan(event.relative)
   elif _mode == "rect":
      _rect_end = event.position
      _highlight.queue_redraw()
      _redraw()
   elif _mode == "node" and event.position.distance_to(_press) > CLICK_PIXELS and _movable:
      if not _chosen.has(_press_node):
         if not event.shift_pressed:
            _chosen = {}
         _chosen[_press_node] = true
         selection_changed.emit()
      _start_drag()
      _mode = "drag"
      _move_by(((event.position - _press) / zoom).rotated(deg_to_rad(turn)))
   elif _mode == "drag":
      _move_by((event.relative / zoom).rotated(deg_to_rad(turn)))


## The edges of the selected nodes, which follow them while they are dragged (straight edges), with
## their arrowheads.
func _start_drag() -> void:
   _drag_edges = PackedInt32Array()
   _drag_arrows = PackedInt32Array()
   if _kind != "lines":
      return
   for e in _src.size():
      if _chosen.has(_src[e]) or _chosen.has(_dst[e]):
         _drag_edges.append(e)
   for j in _arrow_data.size() / NODE_STRIDE:   # custom data z: the edge
      var e := int(_arrow_data[j * NODE_STRIDE + 14])
      if _chosen.has(_src[e]) or _chosen.has(_dst[e]):
         _drag_arrows.append(j)


## Moves the selected nodes by [param delta] (world units), with their edges.
func _move_by(delta: Vector2) -> void:
   var nm := _nodes.multimesh
   for i in _chosen:
      var k: int = i * NODE_STRIDE
      _buffer[k + 3] += delta.x
      _buffer[k + 7] += delta.y
      nm.set_instance_transform_2d(i, Transform2D(Vector2(_buffer[k], 0), Vector2(0, _buffer[k + 5]), Vector2(_buffer[k + 3], _buffer[k + 7])))
   var em := _edges.multimesh
   for e in _drag_edges:
      var a := _position(_src[e])
      var b := _position(_dst[e])
      var k := e * _edge_stride
      var w := Vector2(_edge_data[k + 1], _edge_data[k + 5]).length()
      var along := b - a
      var normal := Vector2(-along.y, along.x).normalized() * w
      _edge_data[k] = along.x
      _edge_data[k + 4] = along.y
      _edge_data[k + 1] = normal.x
      _edge_data[k + 5] = normal.y
      em.set_instance_transform_2d(e, Transform2D(along, normal, (a + b) / 2))
   var am := _arrows.multimesh
   for j in _drag_arrows:   # custom data w: whether the arrow is at the target of the edge
      var k := j * NODE_STRIDE
      var e := int(_arrow_data[k + 14])
      var at_target := _arrow_data[k + 15] > 0.5
      var tip := _position(_dst[e] if at_target else _src[e])
      var along := (tip - _position(_src[e] if at_target else _dst[e])).normalized()
      if along == Vector2.ZERO:
         continue
      _arrow_data[k] = along.x
      _arrow_data[k + 4] = along.y
      _arrow_data[k + 1] = -along.y
      _arrow_data[k + 5] = along.x
      _arrow_data[k + 3] = tip.x
      _arrow_data[k + 7] = tip.y
      am.set_instance_transform_2d(j, Transform2D(along, Vector2(-along.y, along.x), tip))
   var lm := _loops.multimesh
   for j in lm.instance_count:   # the loops of the moved nodes (custom data w: the node)
      var k := j * NODE_STRIDE
      var i := int(_loop_data[k + 15])
      if _chosen.has(i):
         var p := _position(i)
         _loop_data[k + 3] = p.x
         _loop_data[k + 7] = p.y
         lm.set_instance_transform_2d(j, Transform2D(Vector2(_loop_data[k], 0), Vector2(0, _loop_data[k + 5]), p))
   _rings_dirty = true
   _highlight.queue_redraw()
   _redraw()


## Mouse released: drops dragged nodes, selects the nodes in the rectangle, or (a click) selects
## the node under the cursor (Shift: adds it or takes it out), or clears the selection.
func _release(event: InputEventMouseButton) -> void:
   var mode := _mode
   _mode = ""
   if mode == "drag":
      var nodes := selected_nodes()
      var positions := PackedVector2Array()
      for i in nodes:
         positions.append(_position(i))
      nodes_moved.emit(nodes, positions)
      return
   if mode == "rect":
      var area := Rect2(_press, Vector2.ZERO).expand(event.position)
      for i in ids.size():
         if area.has_point(_to_screen(_position(i))):
            _chosen[i] = true
      var edges := selected_edges.duplicate()
      for e in edges_in(area):
         if not edges.has(e):
            edges.append(e)
      selected_edges = edges
      _rings_dirty = true
      _highlight.queue_redraw()
      _redraw()
      selection_changed.emit()
      return
   if event.position.distance_to(_press) > CLICK_PIXELS:
      return
   if _press_node >= 0 and event.shift_pressed:
      if _chosen.has(_press_node):
         _chosen.erase(_press_node)
      else:
         _chosen[_press_node] = true
      _rings_dirty = true
      _highlight.queue_redraw()
      _redraw()
      selection_changed.emit()
   elif _press_node >= 0:
      select(_press_node)
   elif _press_edge >= 0 and event.shift_pressed:
      var edges := selected_edges.duplicate()
      var at := edges.find(_press_edge)
      if at >= 0:
         edges.remove_at(at)
      else:
         edges.append(_press_edge)
      set_selection(selected_nodes(), edges)
   elif _press_edge >= 0:
      set_selection(PackedInt32Array(), PackedInt32Array([_press_edge]))
   elif not event.shift_pressed and (not _chosen.is_empty() or not selected_edges.is_empty()):
      set_selection(PackedInt32Array())
