// Nodes: antialiased discs of the diameter given by the worker (custom data x, in world units),
// divided by the scale of the layout (spread: the distances grow, the nodes keep their size on
// screen), never smaller than min_pixels on screen. Nodes with a chart (custom data y: its first
// slice in the texture of the slices, z: its number of slices) are pie charts: from the top of
// the screen, clockwise, each slice in its color up to its end (red channel: the fraction of the
// turn), with the opacity of the node.
shader_type canvas_item;

// Pixels per world unit.
uniform float zoom = 1.0;
// Smallest diameter on screen, in pixels.
uniform float min_pixels = 3.0;
// Scale of the layout.
uniform float spread = 1.0;
// Slices of the node charts: one texel each (end, red, green, blue), in rows of chart_width.
uniform sampler2D slices : filter_nearest;
uniform int chart_width = 4096;
// Rotation of the view, in radians (counterclockwise): the charts start at the top of the screen.
uniform float turn = 0.0;

varying flat vec2 chart;

void vertex() {
   VERTEX *= max(1.0, min_pixels / max(INSTANCE_CUSTOM.x * zoom / spread, 1e-6)) / spread;
   chart = INSTANCE_CUSTOM.yz;
}

void fragment() {
   vec2 v = UV * 2.0 - 1.0;
   float d = length(v);
   float aa = fwidth(d);
   COLOR.a *= 1.0 - smoothstep(1.0 - aa, 1.0, d);
   int count = int(chart.y + 0.5);
   if (count > 0) {
      float t = fract((atan(v.x, -v.y) - turn) / 6.28318530718);
      int first = int(chart.x + 0.5);
      for (int k = 0; k < 64; k++) {
         if (k >= count) {
            break;
         }
         int at = first + k;
         vec4 slice = texelFetch(slices, ivec2(at % chart_width, at / chart_width), 0);
         if (t <= slice.r || k == count - 1) {
            COLOR.rgb = slice.gba;
            break;
         }
      }
   }
}
