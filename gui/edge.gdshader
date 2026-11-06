// Edges: quads whose width is the default width (in world units, set from the zoom and the scale
// of the layout so that it follows the size of the nodes on screen and is at least one pixel) times
// the relative width of each edge (the length of its second axis, set by the worker). In directed
// networks the custom data holds the diameters of the source and of the target and whether each end
// has an arrowhead (1 or 0): the line then stops at the base of the arrowhead (arrow.gdshader: the
// node radius, never smaller than min_pixels / 2 on screen, plus the length of the arrowhead).
// With styles, the texel of each edge (by instance) is its line style (EDGE_STYLES of the worker),
// drawn in units of the width of the line (see line_style.gdshaderinc).
shader_type canvas_item;

#include "res://line_style.gdshaderinc"

// Default width in world units.
uniform float width = 0.05;
// Pixels per world unit, smallest diameter of a node on screen and scale of the layout.
uniform float zoom = 1.0;
uniform float min_pixels = 3.0;
uniform float spread = 1.0;
// Length of an arrowhead in widths of its line, and its smallest length on screen in pixels.
uniform float arrow_length = 2.5;
uniform float arrow_min = 5.0;

void vertex() {
   float along = length(MODEL_MATRIX[0].xy);
   float l = max(arrow_length * width * length(MODEL_MATRIX[1].xy), arrow_min / zoom);
   float a = INSTANCE_CUSTOM.z * (max(INSTANCE_CUSTOM.x / spread, min_pixels / zoom) / 2.0 + l) / max(along, 1e-12);
   float b = INSTANCE_CUSTOM.w * (max(INSTANCE_CUSTOM.y / spread, min_pixels / zoom) / 2.0 + l) / max(along, 1e-12);
   float cut = max(a + b, 1.0);   // nodes too close: nothing left of the line
   VERTEX.x = mix(-0.5 + a / cut, 0.5 - b / cut, VERTEX.x + 0.5);
   float w = max(width * length(MODEL_MATRIX[1].xy), 1e-12);   // width of the line, world units
   style = styled ? texelFetch(styles, ivec2(INSTANCE_ID % chart_width, INSTANCE_ID / chart_width), 0).r : 0.0;
   float grow = style_breadth(style);
   along_w = (VERTEX.x + 0.5) * along / w;
   across_w = VERTEX.y * grow;
   VERTEX.y *= width * grow;
}

void fragment() {
   if (!line_kept(style, along_w, across_w)) {
      discard;
   }
}
