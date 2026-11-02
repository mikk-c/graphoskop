// Nodes: antialiased discs of the diameter given by the worker (custom data x, in world units),
// divided by the scale of the layout (spread: the distances grow, the nodes keep their size on
// screen), never smaller than min_pixels on screen.
shader_type canvas_item;

// Pixels per world unit.
uniform float zoom = 1.0;
// Smallest diameter on screen, in pixels.
uniform float min_pixels = 3.0;
// Scale of the layout.
uniform float spread = 1.0;

void vertex() {
   VERTEX *= max(1.0, min_pixels / max(INSTANCE_CUSTOM.x * zoom / spread, 1e-6)) / spread;
}

void fragment() {
   float d = length(UV * 2.0 - 1.0);
   float aa = fwidth(d);
   COLOR.a *= 1.0 - smoothstep(1.0 - aa, 1.0, d);
}
