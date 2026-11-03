// Arrowheads of the edges of directed networks: one triangle per instance (tip at x = 0, base at
// x = -1, from y = -0.5 to 0.5). The transform puts the origin on the node the arrow points to and
// the x axis along the direction of the edge there (unit length); custom data x is the diameter of
// that node (world units, divided by the scale of the layout as in node.gdshader, never smaller
// than min_pixels on screen), so the tip touches its border, and y the relative width of the edge.
// The arrowhead is arrow_length times as long as the line is wide, and arrow_breadth times as
// wide, but at least arrow_min pixels long (the edge shader stops the line at its base).
shader_type canvas_item;

// Pixels per world unit.
uniform float zoom = 1.0;
// Smallest diameter of a node on screen, in pixels.
uniform float min_pixels = 3.0;
// Scale of the layout.
uniform float spread = 1.0;
// Default width of the edges in world units.
uniform float width = 0.05;
// Length and width of an arrowhead, in widths of its line.
uniform float arrow_length = 2.5;
uniform float arrow_breadth = 2.2;
// Smallest length of an arrowhead on screen, in pixels.
uniform float arrow_min = 5.0;

void vertex() {
   float r = max(INSTANCE_CUSTOM.x / spread, min_pixels / zoom) / 2.0;
   float l = max(arrow_length * width * INSTANCE_CUSTOM.y, arrow_min / zoom);
   VERTEX = vec2(VERTEX.x * l - r, VERTEX.y * l * arrow_breadth / arrow_length);
}
