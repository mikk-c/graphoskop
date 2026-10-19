"""
Benchmark harness: times one network statistic with Graphoskop (CPU and GPU), NetworkX,
python-igraph and graph-tool on LFR benchmark graphs of growing size, and plots the runtimes.

Every library gets the same graph: the largest connected component of an undirected
:func:`graphoskop.lfr_benchmark_graph` (average degree 20, largest degree 50, mixing 0.3,
communities of 20 to 100 nodes), with nodes numbered 0 to n - 1. Only the computation of the
statistic is timed, not the construction of the graph objects. Each library runs with its default
settings (Graphoskop's shortest paths and graph-tool use all CPU cores, NetworkX and igraph one).
Graphoskop is timed ``--repeats`` times (3 by default) and the plot shows the median, which
ignores one unusually slow run without choosing which runs are outliers; the other libraries,
whose runtimes are steady and long, run once. Untimed first calls warm every library up (CUDA
context, kernel compilation) on a small graph, and Graphoskop also on the smallest graph of the
test, which starts the worker processes of its parallel searches. Libraries that are not installed, or no
GPU, are skipped. Results are compared across libraries through a number (the statistic, or a sum
of the entries of a matrix), computed outside the timing.

Every run is appended to ``<name>.csv`` as it comes, and plotted in ``<name>.png``, in the folder
of the test script. Usage, from a test script::

   python average_shortest_path_length.py                  # run and plot
   python average_shortest_path_length.py --plot-only      # plot the csv again
   python average_shortest_path_length.py --sizes 500 1000 --skip networkx --repeats 1
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

import graphoskop as gsk

LIBRARIES = ["graphoskop (GPU)", "graphoskop (CPU)", "graph-tool", "igraph", "networkx"]
COLORS = ["#cb181d", "#fcbba1", "#984ea3", "#377eb8", "#4daf4a"]
MARKERS = ["o", "s", "^", "D", "v"]
LFR = dict(k = 20, maxk = 50, mu = 0.3, minc = 20, maxc = 100)


def lfr_edges(n, seed = 0):
   """
   Edges of the largest connected component of an LFR benchmark graph.

   :param n: Number of nodes of the LFR graph.
   :param seed: Seed of the generator.
   :return: ``(s, t, n_lcc)``: int64 arrays with every edge once (``s < t``), nodes ``0`` to
      ``n_lcc - 1``.
   """
   device = "cuda" if torch.cuda.is_available() else "cpu"
   net = gsk.lfr_benchmark_graph(n = n, seed = seed, device = device, **LFR)
   s, t = net.components[0].edge_index.cpu().numpy()
   keep = s < t
   return s[keep], t[keep], net.components[0].num_nodes


def build(library, s, t, n):
   """
   The graph object of a library.

   :param library: One of :data:`LIBRARIES`.
   :param s: Sources.
   :param t: Targets.
   :param n: Number of nodes.
   :return: The graph, or ``None`` if the library (or a GPU) is not available.
   """
   try:
      if library.startswith("graphoskop"):
         device = "cuda" if "GPU" in library else "cpu"
         if device == "cuda" and not torch.cuda.is_available():
            return None
         return gsk.load_network(pd.DataFrame({"source": s, "target": t}), device = device)
      if library == "networkx":
         import networkx as nx
         g = nx.Graph()
         g.add_nodes_from(range(n))
         g.add_edges_from(zip(s.tolist(), t.tolist()))
         return g
      if library == "igraph":
         import igraph as ig
         return ig.Graph(n = n, edges = np.column_stack([s, t]).tolist())
      if library == "graph-tool":
         import graph_tool.all as gt
         g = gt.Graph(directed = False)
         g.add_vertex(n)
         g.add_edge_list(np.column_stack([s, t]))
         return g
   except ImportError:
      return None


def timed(f, g, library, summary):
   """
   Runtime of one call, and the summary of its result (not timed).

   :param f: Function of the graph.
   :param g: Graph.
   :param library: Library name (GPU calls are synchronized).
   :param summary: Function of the result returning a number.
   :return: ``(seconds, value)``.
   """
   if "GPU" in library:
      torch.cuda.synchronize()
   start = time.perf_counter()
   result = f(g)
   if "GPU" in library:
      torch.cuda.synchronize()
   seconds = time.perf_counter() - start
   value = float(summary(result))
   del result
   return seconds, value


def plot(name, title):
   """
   Log-log plot of the runtimes in ``<name>.csv`` (median of the runs), saved as ``<name>.png``.

   :param name: Base name of the files.
   :param title: Title of the plot.
   """
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from matplotlib.ticker import FixedLocator, NullLocator
   df = pd.read_csv(f"{name}.csv")
   fig, ax = plt.subplots(figsize = (7, 4.5), dpi = 150)
   for library, color, marker in zip(LIBRARIES, COLORS, MARKERS):
      d = df[df.library == library].groupby("nodes").seconds.median()
      if not len(d):
         continue
      style = "-" if library.startswith("graphoskop") else "--"
      ax.plot(d.index, d.values, style, color = color, marker = marker, markersize = 6, linewidth = 2, label = library)
   ax.set_xscale("log")
   ax.set_yscale("log")
   sizes = np.sort(df.nodes.unique())
   ax.xaxis.set_major_locator(FixedLocator(sizes))
   ax.xaxis.set_minor_locator(NullLocator())
   ax.set_xticklabels([f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}k" for x in sizes])
   ax.set_xlabel("Nodes (average degree 20)")
   ax.set_ylabel("Runtime (seconds)")
   ax.set_title(title, loc = "left")
   ax.grid(True, which = "major", color = "#e4e3df", linewidth = 0.8)
   ax.set_axisbelow(True)
   for side in ("top", "right"):
      ax.spines[side].set_visible(False)
   ax.legend(frameon = False)
   fig.tight_layout()
   fig.savefig(f"{name}.png")
   print(f"saved {name}.png")


def run(name, title, functions, sizes, summary = float):
   """
   Command line entry point of a test: times ``functions`` on LFR graphs of the given sizes,
   appends every run to ``<name>.csv`` and plots them.

   A library is not run again after a run longer than ``--limit`` seconds. Values that differ
   between libraries by more than one part in a million are reported.

   :param name: Base name of the result files.
   :param title: Title of the plot.
   :param functions: Dictionary from library name (see :data:`LIBRARIES`) to a function of its
      graph returning the result.
   :param sizes: Default numbers of nodes of the LFR graphs, increasing.
   :param summary: Function of a result returning the number compared across libraries, or a
      dictionary from library name to such functions.
   """
   parser = argparse.ArgumentParser(description = title)
   parser.add_argument("--sizes", type = int, nargs = "+", default = sizes, help = "numbers of nodes")
   parser.add_argument("--skip", nargs = "+", default = [], help = "libraries to skip")
   parser.add_argument("--repeats", type = int, default = 3, help = "runs of graphoskop per size")
   parser.add_argument("--limit", type = float, default = 1800, help = "seconds after which a library stops")
   parser.add_argument("--plot-only", action = "store_true", help = "only plot the csv")
   args = parser.parse_args()
   name = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), name)   # next to the script
   reduce = summary if isinstance(summary, dict) else {l: summary for l in functions}
   if not args.plot_only:
      if os.path.exists(f"{name}.csv"):
         os.remove(f"{name}.csv")
      active = [l for l in functions if l not in args.skip]
      for size, warm in ((500, active), (min(args.sizes), [l for l in active if l.startswith("graphoskop")])):
         s, t, n = lfr_edges(size)
         for library in warm:
            g = build(library, s, t, n)
            if g is not None:
               functions[library](g)
      for size in sorted(args.sizes):
         s, t, n = lfr_edges(size)
         values = {}
         for library in list(active):
            g = build(library, s, t, n)
            if g is None:
               print(f"{library}: not available, skipped")
               active.remove(library)
               continue
            for r in range(args.repeats if library.startswith("graphoskop") else 1):
               seconds, values[library] = timed(functions[library], g, library, reduce[library])
               gc.collect()
               print(f"n = {n:>9}  m = {len(s):>10}  {library:<18} run {r}  {seconds:10.3f} s   {values[library]:.6f}", flush = True)
               row = pd.DataFrame([dict(library = library, nodes = n, edges = len(s), run = r, seconds = seconds, value = values[library])])
               row.to_csv(f"{name}.csv", mode = "a", header = not os.path.exists(f"{name}.csv"), index = False)
               if seconds > args.limit:
                  active.remove(library)
                  break
            del g
            gc.collect()
            if torch.cuda.is_available():
               torch.cuda.empty_cache()
         v = np.array(list(values.values()))
         if len(v) and np.ptp(v) > 1e-6 * np.abs(v).max():
            print(f"warning: values differ at n = {n}: {values}")
   plot(name, title)
