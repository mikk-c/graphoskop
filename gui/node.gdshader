// Nodes: antialiased shapes inscribed in a disc of the diameter given by the worker (custom data
// x, in world units), divided by the scale of the layout (spread: the distances grow, the nodes
// keep their size on screen), never smaller than min_pixels on screen. The shape is custom data w
// (NODE_SHAPES of the worker: circle, triangle, square, diamond, pentagon, hexagon, heptagon,
// octagon, star). Nodes with a chart (custom data y: its first slice in the texture of the slices, z:
// its number of slices) are pie charts: from the top of the screen, clockwise, each slice in its
// color up to its end (red channel: the fraction of the turn), with the opacity of the node. With
// borders, the texel of each node in the texture of the borders (by instance) gives the color
// and the width of its border, as a fraction of the radius, inside the shape.
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
// Borders: one texel per node (red, green, blue, width), when bordered.
uniform sampler2D borders : filter_nearest;
uniform bool bordered = false;

varying flat vec2 chart;
varying flat float shape;
varying flat vec4 border;

void vertex() {
   VERTEX *= max(1.0, min_pixels / max(INSTANCE_CUSTOM.x * zoom / spread, 1e-6)) / spread;
   chart = INSTANCE_CUSTOM.yz;
   shape = INSTANCE_CUSTOM.w;
   border = bordered ? texelFetch(borders, ivec2(INSTANCE_ID % chart_width, INSTANCE_ID / chart_width), 0) : vec4(0.0);
}

// Signed distance (radius 1: negative inside) to a regular polygon of n sides with a vertex at
// the angle offset from the top.
float polygon(vec2 p, float n, float offset) {
   float an = 3.14159265359 / n;
   float a = atan(p.x, p.y) - offset;
   return length(p) * cos(mod(a, 2.0 * an) - an) - cos(an);
}

// Signed distance to a five-pointed star of outer radius 1 (inner radius 0.382), a point on top.
float star(vec2 p) {
   const vec2 k1 = vec2(0.809016994375, -0.587785252292);
   const vec2 k2 = vec2(-0.809016994375, -0.587785252292);
   p.x = abs(p.x);
   p -= 2.0 * max(dot(k1, p), 0.0) * k1;
   p -= 2.0 * max(dot(k2, p), 0.0) * k2;
   p.x = abs(p.x);
   p.y -= 1.0;
   vec2 ba = 0.382 * vec2(-k1.y, k1.x) - vec2(0.0, 1.0);
   float h = clamp(dot(p, ba) / dot(ba, ba), 0.0, 1.0);
   return length(p - ba * h) * sign(p.y * ba.x - p.x * ba.y);
}

void fragment() {
   vec2 p = UV * 2.0 - 1.0;   // y up: the quad mesh has UV.y 0 at its bottom on screen
   int k = int(shape + 0.5);
   float d = length(p) - 1.0;
   if (k == 1) { d = polygon(p, 3.0, 0.0); }
   else if (k == 2) { d = polygon(p, 4.0, 0.785398163397); }
   else if (k == 3) { d = polygon(p, 4.0, 0.0); }
   else if (k == 4) { d = polygon(p, 5.0, 0.0); }
   else if (k == 5) { d = polygon(p, 6.0, 0.0); }
   else if (k == 6) { d = polygon(p, 7.0, 0.0); }
   else if (k == 7) { d = polygon(p, 8.0, 0.392699081699); }
   else if (k == 8) { d = star(p); }
   float aa = fwidth(d);
   COLOR.a *= 1.0 - smoothstep(-aa, 0.0, d);
   int count = int(chart.y + 0.5);
   if (count > 0) {
      float t = fract((atan(p.x, p.y) - turn) / 6.28318530718);
      int first = int(chart.x + 0.5);
      for (int s = 0; s < 64; s++) {
         if (s >= count) {
            break;
         }
         int at = first + s;
         vec4 slice = texelFetch(slices, ivec2(at % chart_width, at / chart_width), 0);
         if (t <= slice.r || s == count - 1) {
            COLOR.rgb = slice.gba;
            break;
         }
      }
   }
   if (border.a > 0.0) {   // inside the shape, within the width of the border
      COLOR.rgb = mix(COLOR.rgb, border.rgb, smoothstep(-border.a - aa, -border.a, d));
   }
}
