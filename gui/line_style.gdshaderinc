// Line styles of edges (EDGE_STYLES of the worker): 0 solid, 1 dash, 2 dash-dot, 3 dots,
// 4 parallel lines, 5 sine wave, 6 zigzag, in units of the width w of the line: dashes of 3.6w
// every 6w, a dash of 3.6w and a dot of 1w every 8w (the dot in the middle of the gap), round dots of 1w every 2.5w, two lines of 0.75w 1.5w apart,
// waves of amplitude 1.5w and period 6w. The quads of parallel lines are 3w wide, of waves 4w.

// Line styles: one texel per edge, when styled.
uniform sampler2D styles : filter_nearest;
uniform bool styled = false;
uniform int chart_width = 4096;

varying flat float style;
// Position along the line and across it, in widths of the line.
varying float along_w;
varying float across_w;

// Breadth of the quad of a style, in widths of the line.
float style_breadth(float s) {
   int k = int(s + 0.5);
   return k == 4 ? 3.0 : (k == 5 || k == 6 ? 4.0 : 1.0);
}

// Whether the point (x along, y across, in widths) is on the line of style s.
bool line_kept(float s, float x, float y) {
   int k = int(s + 0.5);
   if (k == 1) {
      return fract(x / 6.0) < 0.6;
   } else if (k == 2) {
      float f = fract(x / 8.0) * 8.0;
      return f < 3.6 || (f > 5.3 && f < 6.3);
   } else if (k == 3) {
      float dx = (fract(x / 2.5) - 0.5) * 2.5;
      return dx * dx + y * y < 0.25;
   } else if (k == 4) {
      return abs(y) > 0.75;
   } else if (k == 5 || k == 6) {
      float phase = x / 6.0;
      float c, slope;
      if (k == 5) {
         c = 1.5 * sin(6.28318530718 * phase);
         slope = 1.5 * 6.28318530718 / 6.0 * cos(6.28318530718 * phase);
      } else {
         float f = fract(phase);
         c = 1.5 * (4.0 * abs(f - 0.5) - 1.0);
         slope = 1.5 * 4.0 / 6.0;
      }
      return abs(y - c) < 0.5 * sqrt(1.0 + slope * slope);
   }
   return true;
}
