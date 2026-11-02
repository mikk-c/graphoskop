// Self loops: antialiased circles on top of their node, one instance per loop. The transform holds
// the position of the node, scaled by the diameter of the loop (custom data x, in world units);
// custom data y is the diameter of the node, z the relative width of the edge. Both diameters are
// divided by the scale of the layout (spread), as for the nodes (node.gdshader); the width of the
// line is that of the edges (width, from the zoom so that it is at least one pixel) times the
// relative width. The loop is above the node on screen, its lower part over the node.
shader_type canvas_item;

// Scale of the layout.
uniform float spread = 1.0;
// Default width of the edges in world units.
uniform float width = 0.05;

varying float inner;

void vertex() {
   float D = max(INSTANCE_CUSTOM.x, 1e-12);
   float r = D / 2.0 / spread;                                // radius of the loop
   float w = width * INSTANCE_CUSTOM.z;
   float R = r + w / 2.0;                                     // outer radius
   float h = INSTANCE_CUSTOM.y / 2.0 / spread + r * 0.5;      // center above the center of the node
   VERTEX = VERTEX * (2.0 * R / D) + vec2(0.0, -h / D);
   inner = max(r - w / 2.0, 0.0) / R;
}

void fragment() {
   float d = length(UV * 2.0 - 1.0);
   float aa = fwidth(d);
   COLOR.a *= smoothstep(inner - aa, inner, d) * (1.0 - smoothstep(1.0 - aa, 1.0, d));
}
