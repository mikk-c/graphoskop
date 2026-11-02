// Selection marks: antialiased rings around the selected nodes, drawn by the GPU (one instance per
// selected node, with the transform and the diameter of the node, as in node.gdshader), from 1 to
// 3 pixels outside the node on screen, whatever the zoom.
shader_type canvas_item;

// Pixels per world unit.
uniform float zoom = 1.0;
// Smallest diameter of a node on screen, in pixels.
uniform float min_pixels = 3.0;
// Scale of the layout (nodes are this many times smaller in world units, see node.gdshader).
uniform float spread = 1.0;

varying float inner;
varying float outer;

void vertex() {
   float node = max(INSTANCE_CUSTOM.x * zoom / spread, min_pixels);   // diameter of the node on screen
   float quad = node + 8.0;                                  // room for the ring and its antialiasing
   VERTEX *= quad / max(INSTANCE_CUSTOM.x * zoom, 1e-6);
   inner = (node / 2.0 + 1.0) / (quad / 2.0);
   outer = (node / 2.0 + 3.0) / (quad / 2.0);
}

void fragment() {
   float d = length(UV * 2.0 - 1.0);
   float aa = fwidth(d);
   COLOR.a *= smoothstep(inner - aa, inner, d) * (1.0 - smoothstep(outer, outer + aa, d));
}
