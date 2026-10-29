// Edges: quads whose width is the default width (in world units, set from the zoom so that it is
// at least one pixel) times the relative width of each edge (the length of its second axis, set
// by the worker).
shader_type canvas_item;

// Default width in world units.
uniform float width = 0.05;

void vertex() {
   VERTEX.y *= width;
}
