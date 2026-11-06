// Bundled edges: one instance per cubic Bézier piece, a strip of straight segments whose vertices
// the GPU puts on the curve. The worker packs the four points of the piece in the instance: the
// first as the origin of the transform, the second and third as its axes, the fourth in the
// custom data (x, y), with the relative width (z). The mesh has x from 0 to 1 along the strip
// (the curve parameter) and y -0.5 or 0.5 across it. With styles, the texel of the edge of each
// piece (custom data w, the position of the edge) is its line style (line_style.gdshaderinc),
// along a length estimated from the control points (dashes restart at every piece).
shader_type canvas_item;
render_mode skip_vertex_transform;

#include "res://line_style.gdshaderinc"

// Default width in world units.
uniform float width = 0.05;

void vertex() {
   vec2 p0 = MODEL_MATRIX[3].xy;
   vec2 p1 = MODEL_MATRIX[0].xy;
   vec2 p2 = MODEL_MATRIX[1].xy;
   vec2 p3 = INSTANCE_CUSTOM.xy;
   float t = VERTEX.x;
   float s = 1.0 - t;
   vec2 p = s * s * s * p0 + 3.0 * s * s * t * p1 + 3.0 * s * t * t * p2 + t * t * t * p3;
   vec2 d = 3.0 * s * s * (p1 - p0) + 6.0 * s * t * (p2 - p1) + 3.0 * t * t * (p3 - p2);
   if (dot(d, d) < 1e-20) {
      d = p3 - p0 + vec2(1e-9, 0.0);
   }
   vec2 n = normalize(vec2(-d.y, d.x));
   int e = int(INSTANCE_CUSTOM.w + 0.5);
   style = styled ? texelFetch(styles, ivec2(e % chart_width, e / chart_width), 0).r : 0.0;
   float grow = style_breadth(style);
   float w = max(width * INSTANCE_CUSTOM.z, 1e-12);
   float l = (length(p1 - p0) + length(p2 - p1) + length(p3 - p2) + length(p3 - p0)) / 2.0;   // about the length of the piece
   along_w = t * l / w;
   across_w = VERTEX.y * grow;
   VERTEX = p + n * VERTEX.y * width * INSTANCE_CUSTOM.z * grow;
}

void fragment() {
   if (!line_kept(style, along_w, across_w)) {
      discard;
   }
}
