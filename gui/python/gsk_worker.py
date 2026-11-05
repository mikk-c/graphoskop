"""
Python worker of the Graphoskop graphical interface.

The interface (a Godot application, in the ``gui`` folder) starts this script with the Python
interpreter of the user's environment, as ``python gsk_worker.py <port>``, and talks to it over a
local TCP connection on ``port``. The worker runs the library: the interface never computes
anything about a network, it only draws what the worker sends back once a function has finished.
The worker is a separate process, so the interface cannot slow down a computation; while one
runs, the worker answers nothing else.

Every message, in both directions, is a frame:

* 8 bytes: the length of the header and the length of the payload, as two little-endian unsigned
  32-bit integers;
* the header: a JSON object, UTF-8;
* the payload: raw bytes (possibly none).

Requests have a ``cmd`` (see :data:`COMMANDS`) and an ``id``, which the reply repeats. Replies have
``ok`` (``True`` or ``False``) and, if ``False``, ``error`` (the message) and ``traceback``. When
it connects, the worker sends a first frame with ``cmd = "hello"``, then imports the library and
sends a second one with ``cmd = "ready"`` (its ``ok`` tells whether the import worked). While a
function with a progress bar runs, the worker also sends frames with ``cmd = "progress"`` (see
:class:`ProgressBar`). A frame with ``cmd = "abort"`` interrupts the command running (see
:func:`main`).

Every tab of the interface has its own namespace (a dictionary), where the code shown in the code
panel of that tab runs: the code panel always shows exactly the code that produced what is drawn.
The namespace also keeps the style of the drawing (``__style__``, see :func:`default_style`), the
default diameter of the drawn nodes (``__size__``) and the tables of the data panel
(``__tables__``).
"""

import _thread
import ast
import inspect
import json
import keyword
import os
import queue
import re
import socket
import struct
import sys
import threading
import time
import traceback

import numpy as np

#: Floats per node in the Godot ``MultiMesh`` buffer of the nodes: a 2D transform (8 floats), a
#: color (4) and custom data (4: the diameter of the node, then zeros).
NODE_STRIDE = 16
#: Floats per edge in the buffer of the edges: a 2D transform (8 floats) and a color (4); in
#: directed networks also custom data (4, :data:`ARROW_EDGE_STRIDE`).
EDGE_STRIDE = 12
#: Half the width of the canvas of an empty network (File › New network), in drawing units, and the
#: diameter of the nodes drawn on it: the nodes added by hand keep it.
EMPTY_BOX = 1.0
EMPTY_NODE_SIZE = 0.06
#: Height of the node labels of size 1 ×, relative to the default node diameter (labels are
#: centered on their nodes).
LABEL_FRACTION = 0.5
#: Floats per edge of directed networks: as :data:`EDGE_STRIDE`, then the diameters of the source
#: and of the target, and whether an arrowhead is at the source and at the target (1 or 0): the
#: edge shader of the interface ends the line at the base of its arrowheads.
ARROW_EDGE_STRIDE = 16
#: Floats per piece of the buffer of bundled edges (cubic Bézier curves): the first three points
#: in a 2D transform (8 floats: second point as x axis, third as y axis, first as origin), a color
#: (4) and custom data (4: the last point, the relative width, 0).
CURVE_STRIDE = 16

#: Default color of the nodes.
NODE_COLOR = "#e41a1c"
#: Default color of the edges.
EDGE_COLOR = "#999999"
#: Color of the second type of nodes of bipartite networks (hyperedges and facets of incidence
#: networks), by default: the second color of ColorBrewer's Set1, whose first one is
#: :data:`NODE_COLOR`.
TYPE1_COLOR = "#377eb8"
#: Default color of missing values.
MISSING_COLOR = "#cccccc"
#: Most distinct values of an attribute sent to the interface (all values get colors anyway).
MAX_VALUES = 1000
#: Most node (or edge) attributes shown as columns of the tables of the data panel (the first
#: ones; all of them in the table of one node).
TABLE_ATTRIBUTES = 100
#: Most attribute names listed in the "Network" tab of the data panel, for each kind.
SUMMARY_ATTRIBUTES = 20
#: Most values used to find natural breaks (a sample of quantiles beyond).
MAX_BREAK_VALUES = 1000
#: Most cells of the grid used by the interface to find the node under the cursor.
MAX_CELLS = 1 << 22
#: Most segments drawn for bundled edges: every Bézier piece is drawn with up to 16 straight
#: segments (computed by the GPU), fewer beyond this total (at least 2).
MAX_SEGMENTS = 1 << 24
#: Smallest interval between two progress frames, in seconds.
PROGRESS_INTERVAL = 1.

#: Values shown in the parameter dialog of the generators for parameters whose default is
#: ``None``: their actual value without ``net`` (by name, or by function and name; ``""`` for none:
#: Infomap's ``directed`` follows the network).
SHOWN_DEFAULTS = {"directed": "False", "self_loops": "False", "device": "\"cpu\"", ("girvan_newman_graph", "l"): "4", ("girvan_newman_graph", "k"): "32", ("girvan_newman_graph", "degree"): "16", ("growing_manifold", "d"): "2", ("lfr_benchmark_graph", "beta"): "1.5", ("infomap_communities", "directed"): ""}
#: Hints shown in empty fields of the parameter dialog, for parameters whose default ``None``
#: has no value of its own.
HINTS = {"seed": "random", ("geometric_graph", "m"): "unused (r)", ("geometric_graph", "positions"): "random", ("kronecker_graph", "n"): "from k", ("lfr_benchmark_graph", "muw"): "unweighted", ("lfr_benchmark_graph", "minc"): "smallest degree", ("lfr_benchmark_graph", "maxc"): "maxk", ("mixed_membership_block_model", "threshold"): "1 / (k + 1)", ("mixed_membership_block_model", "memberships"): "from n and alpha", ("nested_block_model", "degrees"): "uniform", "groups": "none", "order": "by the edges", "weight": "unweighted", "levels": "from the edges", "root": "most neighbors", "attributes": "adjacency matrix", "embedding": "none", "epochs": "automatic", "initial": "none", ("bundle_edges", "groups"): "from the layout", ("route_edges", "size"): "drawn nodes", ("route_edges", "margin"): "a quarter of the median size", ("bigclam_communities", "groups"): "automatic", ("block_model_communities", "groups"): "automatic", ("block_model_communities", "threshold"): "1 / (groups + 1)", ("hierarchical_link_communities", "threshold"): "best partition density", ("hierarchical_random_graph_communities", "chains"): "one per thread", ("infomap_communities", "directed"): "as the network", ("infomap_communities", "physical"): "none", ("leiden_communities", "initial"): "singletons", "column_groups": "none", "column_order": "by the edges"}
#: Parameters of layouts and community discovery that take attribute names (by name, or by
#: function and name): the kind of attribute (``"nodes"`` or ``"edges"``) and whether several
#: names can be given (a list); the dialog offers the attributes of the network. (``groups`` of
#: community discovery is a number.)
CHOICES = {"groups": ("nodes", True), "order": ("nodes", False), "levels": ("nodes", False), "attributes": ("nodes", True), "weight": ("edges", False), ("leiden_communities", "initial"): ("nodes", False), ("infomap_communities", "physical"): ("nodes", False), "column_groups": ("nodes", True), "column_order": ("nodes", False)}
#: Values offered before the attribute names in the menus of some attribute parameters (by
#: function and name): the matrix layout also sorts by ``"degree"``.
EXTRA_CHOICES = {("matrix_layout", "order"): ["\"degree\""], ("matrix_layout", "column_order"): ["\"degree\""]}

#: Parameters of layouts not shown in the parameter dialog: the network (layouts work on the
#: network of the tab), the prefix of the positions (the interface draws ``pos``), the number of
#: dimensions (it draws two) and the progress bar (it shows it). Generators hide only ``net``
#: (fitted generators are not offered) and ``progress``.
HIDDEN = {"net", "layout", "attribute", "dimensions", "progress"}
#: Parameters chosen from a list of values (Python source code; ``""`` is ``None``, shown with its
#: hint), by name or by function and name: the dialog shows a menu.
OPTIONS = {"direction": ["\"down\"", "\"up\"", "\"left\"", "\"right\"", "\"radial\""], "metric": ["\"euclidean\"", "\"cosine\""], "rewire": ["\"target\"", "\"both\"", "\"add\""], ("mirror_layout", "axis"): ["\"x\"", "\"y\"", "\"both\""], ("rotate_layout", "axis"): ["\"z\"", "\"x\"", "\"y\""], "embedding": ["", "\"node2vec\"", "\"laplacian_eigenmaps\"", "\"locally_linear_embedding\"", "\"struc2vec\""], "model": ["\"nested_block_model\"", "\"stochastic_block_model\"", "\"degree_corrected_block_model\"", "\"planted_partition_graph\"", "\"mixed_membership_block_model\""], "quality": ["\"modularity\"", "\"cpm\""], "strategy": ["\"semisynchronous\"", "\"asynchronous\""], "line_graph": ["\"random_walk\"", "\"weighted\"", "\"plain\""], ("infomap_communities", "directed"): ["", "True", "False"], ("degree", "direction"): ["\"out\"", "\"in\"", "\"total\""], ("degree", "by"): ["\"layer\"", "\"entity\""], ("closeness_centrality", "direction"): ["\"out\"", "\"in\""], ("harmonic_centrality", "direction"): ["\"out\"", "\"in\""], ("reach_centrality", "direction"): ["\"out\"", "\"in\""], ("eigenvector_centrality", "direction"): ["\"in\"", "\"out\""], ("katz_centrality", "direction"): ["\"in\"", "\"out\""], ("alpha_centrality", "direction"): ["\"in\"", "\"out\""], "mode": ["\"all\"", "\"in\"", "\"out\""]}
#: Numeric parameters between two bounds: smallest and largest value and step; the dialog shows a
#: slider next to the number.
RANGES = {"p": (0., 1., 0.0001), "mu": (0., 1., 0.01), "p_in": (0., 1., 0.0001), "p_out": (0., 1., 0.0001), "clustering": (0., 1., 0.01), "alignment": (0., 1., 0.01), "min_dist": (0., 1., 0.01), ("growing_manifold", "p"): (0., 0.99, 0.01), ("rotate_layout", "degrees"): (-180., 180., 1.), ("bundle_edges", "beta"): (0., 1., 0.01), "epsilon": (0.01, 1., 0.01), "damping": (0., 1., 0.01), "consensus": (0.5, 0.99, 0.01), "tolerance": (0., 1., 0.01), "union": (0., 1., 0.01), "teleportation": (0., 1., 0.01)}
#: Parameters taking a text (a name), written without quotes in the dialog (by name, or by
#: function and name).
STRINGS = {"attribute", "position", "initial", ("lfr_benchmark_graph", "weight")}
#: Parameters taking a node id, written without quotes in the dialog.
NODES = {"root"}
#: Parameters taking lists, matrices or dictionaries (or a number), written in the dialog as
#: numbers separated by commas, rows separated by semicolons, or ``key: value`` pairs.
LISTS = {"sizes", "degrees", "probs", "edges", "initiator", "dimensions", "memberships", "positions", "alpha", ("nested_block_model", "levels")}
#: Integer parameters: their smallest value; the dialog shows a number with arrows.
INTEGERS = {"n": 1, "m": 0, "l": 1, "d": 1, "degree": 1, "iterations": 0, "n_neighbors": 2, "rounds": 0, "tries": 1, "max_iter": 1, "max_blocks": 1, "mixing": 0, "repair": 0, "maxk": 1, "on": 0, "om": 1, "points": 2, "max_groups": 2, "min_size": 1, "samples": 1, "burnin": 0, "interval": 1, "trials": 1, "sweeps": 1, "max_communities": 1, "iterations": 1, "runs": 1, ("barabasi_albert_graph", "k"): 1, ("caveman_graph", "k"): 1, ("watts_strogatz_graph", "k"): 2, ("kronecker_graph", "k"): 1, ("girvan_newman_graph", "k"): 1}
#: Other numeric parameters: their smallest value and step; the dialog shows a number with arrows.
NUMBERS = {("resource_allocation_projection", "lambd"): (0., 0.1), "spacing": (0., 0.1), "gap": (0., 0.5), "edge_length": (0., 0.1), "layer_spacing": (0., 0.1), "size": (0., 0.1), "margin": (0., 0.01), "t1": (1., 0.1), "t2": (1., 0.1), "r": (0., 0.01), ("geometric_graph", "p"): (1., 0.5), ("lfr_benchmark_graph", "beta"): (0., 0.1), ("lfr_benchmark_graph", "k"): (1., 1.), ("holme_kim_graph", "k"): (1., 1.), ("copying_graph", "k"): (1., 1.), ("link_selection_graph", "k"): (1., 1.), "resolution": (0., 0.1), "randomness": (0., 0.001), ("katz_centrality", "alpha"): (0., 0.01), ("katz_centrality", "beta"): (0., 0.1), ("alpha_centrality", "alpha"): (0., 0.1), ("alpha_centrality", "exo"): (0., 0.1), ("springrank", "alpha"): (0., 0.1)}

_EPI = ["si_model", "sis_model", "sir_model", "complex_si_model", "complex_sis_model", "complex_sir_model", "complex_nonprogressive_model"]
_HIGHER = ["higher_order_si_model", "higher_order_sis_model", "higher_order_sir_model"]
OPTIONS.update({"selection": ["\"top\"", "\"weighted\""], "engine": ["", "\"numba\"", "\"torch\""], "reinforcement": ["\"classical\"", "\"threshold\"", "\"cascade\"", "\"independent\"", "\"decreasing\""]})
RANGES.update({(f, "beta"): (0., 1., 0.001) for f in _EPI})
RANGES.update({(f, "mu"): (0., 1., 0.001) for f in _EPI + _HIGHER})
RANGES.update({"decay": (0., 1., 0.01)})
LISTS.update({(f, "beta") for f in _HIGHER})
INTEGERS.update({"max_steps": 1, "window": 1})
NUMBERS.update({"seeds": (0., 0.01)})
NUMBERS.update({(f, "tolerance"): (0., 0.001) for f in ["sis_model", "complex_sis_model", "higher_order_sis_model"]})
CHOICES.update({(f, "ranking"): ("nodes", False) for f in _EPI + _HIGHER})
CHOICES.update({(f, "strategy"): ("nodes", False) for f in _EPI + _HIGHER})
EXTRA_CHOICES.update({(f, "strategy"): ["\"friend_of_random\""] for f in _EPI + _HIGHER})
HINTS.update({"ranking": "random seeds", "engine": "automatic", "batch": "automatic", "immunize": "none", "strategy": "random", ("complex_si_model", "threshold"): "none", ("complex_sis_model", "threshold"): "none", ("complex_sir_model", "threshold"): "none", ("complex_nonprogressive_model", "threshold"): "none", ("higher_order_si_model", "threshold"): "all the others", ("higher_order_sis_model", "threshold"): "all the others", ("higher_order_sir_model", "threshold"): "all the others"})

OPTIONS.update({"policy": ["\"max\"", "\"min\"", "\"out\"", "\"in\""]})
OPTIONS.update({(f, "kind"): ["\"maximum\"", "\"minimum\""] for f in ["spanning_tree", "maximally_filtered_graph", "triangulated_filtered_graph"]})
INTEGERS.update({"exact_max": 0, "n_jobs": 1})
HINTS.update({("polya_filter", "a"): "fitted", ("convex_skeleton", "weight"): "unweighted"})
OPTIONS.update({(f, "combine"): ["\"sum\"", "\"mean\"", "\"max\""] for f in ["remove_parallel_edges", "to_undirected", "network_condensation", "quotient_network"]})
OPTIONS.update({(f, "node_combine"): ["\"mean\"", "\"sum\"", "\"max\""] for f in ["network_condensation", "quotient_network"]})
OPTIONS.update({(f, "disagree"): ["\"zero\"", "\"positive\"", "\"negative\""] for f in ["remove_parallel_edges", "to_undirected"]})
OPTIONS.update({("discrete_core_periphery", "core_periphery_ties"): ["\"present\"", "\"absent\"", "\"missing\""]})
INTEGERS.update({"restarts": 0})
INTEGERS.update({("route_edges", "max_points"): 4})
#: Parameters shown empty in the dialog, whatever their default: ``size`` of ``route_edges`` is
#: then the diameter of the drawn nodes (:func:`size_code`).
EMPTY = {("route_edges", "size")}
LISTS.update({("rombach_core_periphery", "alpha"), ("rombach_core_periphery", "beta")})
HINTS.update({("rombach_core_periphery", "alpha"): "0.05, 0.1, …, 1", ("rombach_core_periphery", "beta"): "0.05, 0.1, …, 1", ("rombach_core_periphery", "p"): "product of core values"})
OPTIONS.update({("nestedness", "null_model"): ["", "\"equiprobable\"", "\"proportional\"", "\"fixed\""]})
HINTS.update({("nestedness", "null_model"): "none (no comparison)"})
OPTIONS.update({("clique_promotion", "kind"): ["\"simplicial\"", "\"hypergraph\""]})
HINTS.update({("quotient_network", "sets"): "required", ("clique_expansion", "weight"): "unweighted"})

gsk = None
_sock = None
_spaces = {}
_busy = False


def _send(sock, header, payload = b""):
   """
   Sends one frame.

   :param sock: Connected socket.
   :param header: JSON-serializable dictionary.
   :param payload: Bytes after the header.
   """
   h = json.dumps(header).encode("utf-8")
   sock.sendall(struct.pack("<II", len(h), len(payload)) + h + payload)   # one call: an abort cannot cut a frame


def _receive(sock, n):
   """
   Reads exactly ``n`` bytes, or returns ``None`` if the connection closes first.
   """
   chunks = []
   while n > 0:
      b = sock.recv(min(n, 1 << 20))
      if not b:
         return None
      chunks.append(b)
      n -= len(b)
   return b"".join(chunks)


class ProgressBar:
   """
   Progress bar of the library shown in the status bar of the interface: it replaces the tqdm bar
   of :mod:`graphoskop.progress` (same calls), so that the library needs nothing for the
   interface. It sends a ``progress`` frame when it opens, then at most every
   :data:`PROGRESS_INTERVAL` seconds, with ``name``, ``fraction`` (of the work done),
   ``elapsed`` and ``remaining`` (seconds, estimated as the bars of the library do), and a last
   frame with ``done = True`` when it closes.
   """

   def __init__(self, total = None, desc = "", **kwargs):
      self.total, self.desc, self.n = total or 0, desc, 0
      self.start = self.last = time.time()
      self._frame()

   def update(self, n = 1):
      self.n += n
      now = time.time()
      if now - self.last >= PROGRESS_INTERVAL:
         self.last = now
         self._frame()

   def close(self):
      if _sock is not None:
         _send(_sock, {"cmd": "progress", "done": True})

   def _frame(self):
      if _sock is None:
         return
      elapsed = time.time() - self.start
      fraction = self.n / self.total if self.total else 0.
      remaining = elapsed / self.n * (self.total - self.n) if self.n else None
      _send(_sock, {"cmd": "progress", "name": self.desc, "fraction": fraction, "elapsed": elapsed, "remaining": remaining})


def literal(x):
   """
   Python literal of a value, as written in the code panel: strings in double quotes.

   :param x: String, number, boolean, ``None``, or a list of them.
   :return: Source code of the value.
   """
   if isinstance(x, list):
      return "[" + ", ".join(literal(v) for v in x) + "]"
   return json.dumps(x, ensure_ascii = False) if isinstance(x, str) else repr(x)


def text(v):
   """
   Text of a value in the data panel and in qualitative colors: integral numbers without decimals,
   others with 6 significant digits, several values (of a qualitative attribute) joined by commas;
   ``None`` for missing values.

   :param v: A value.
   :return: Its text, or ``None``.
   """
   if v is None:
      return None
   if isinstance(v, (tuple, list)):
      return ", ".join(t for t in (text(x) for x in v) if t is not None)
   if isinstance(v, (float, np.floating)):
      if np.isnan(v):
         return None
      return str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else f"{v:.6g}"
   if isinstance(v, (int, np.integer, bool, np.bool_)):
      return str(v)
   try:
      import pandas as pd
      if pd.isna(v):
         return None
   except (TypeError, ValueError):
      pass
   return str(v)


def _call(function, positional, keywords):
   """
   Source code of a call: ``function(positional, name = value, ...)``.
   """
   args = [literal(positional)] + [f"{k} = {literal(v)}" for k, v in keywords]
   return f"{function}({', '.join(args)})"


def load_code(spec):
   """
   Code loading a network as described by the load dialog of the interface: a call to
   :func:`~graphoskop.load_network`, :func:`~graphoskop.load_hypergraph`,
   :func:`~graphoskop.load_simplicial_complex` or :func:`~graphoskop.load_matrix` (an adjacency
   matrix: ``directed`` and ``bipartite`` only when checked, otherwise found from the matrix;
   ``header``: ids in the first row and column; the separator ``\\s+`` is any whitespace, the
   default), with only the arguments that differ from their defaults, followed by
   :func:`~graphoskop.signed_network` for signed networks.

   :param spec: Dictionary with ``edges`` (path), ``nodes`` (path or ``None``), ``sep``,
      ``header``, ``kind`` (``"network"``, ``"hypergraph"``, ``"simplicial"`` or ``"matrix"``), ``source`` and
      ``target`` (column names, or positions without a header; node and hyperedge or simplex
      column in higher-order networks), ``directed``, ``bipartite``, ``multilayer``, ``layer``,
      ``direction`` (directed hypergraphs), ``node_format`` (``"wide"`` or ``"long"``),
      ``node_columns`` (list, or ``None`` for the first columns), ``device`` and ``sign`` (edge
      attribute of the signs, or ``None``).
   :return: List of lines of code, defining ``net``.
   """
   kind = spec.get("kind", "network")
   kw = []
   def add(name, value, default):
      if value != default and value is not None:
         kw.append((name, value))
   if kind == "matrix":
      add("nodes", spec.get("nodes"), None)
      add("sep", spec.get("sep", "\\s+"), "\\s+")
      add("header", spec.get("header", False), False)
      add("directed", spec.get("directed", False), False)
      add("bipartite", spec.get("bipartite", False), False)
      add("device", spec.get("device", "cpu"), "cpu")
      return ["net = gsk." + _call("load_matrix", spec["edges"], kw)]
   add("nodes", spec.get("nodes"), None)
   add("sep", spec.get("sep", "\t"), "\t")
   add("header", spec.get("header", True), True)
   if kind == "network":
      add("source", spec.get("source", 0), 0)
      add("target", spec.get("target", 1), 1)
      add("directed", spec.get("directed", False), False)
      add("node_format", spec.get("node_format", "wide"), "wide")
      add("bipartite", spec.get("bipartite", False), False)
      add("device", spec.get("device", "cpu"), "cpu")
      if spec.get("multilayer"):
         kw.append(("multilayer", True))
         add("layer", spec.get("layer"), "layer")
      if spec.get("node_columns") is not None and spec.get("nodes") is not None:
         kw.append(("node_columns", list(spec["node_columns"])))
      lines = ["net = gsk." + _call("load_network", spec["edges"], kw)]
      if spec.get("sign") is not None:
         lines.append(f"net = gsk.signed_network(net, weight = {literal(spec['sign'])})")
      return lines
   name, column = ("load_hypergraph", "hyperedge") if kind == "hypergraph" else ("load_simplicial_complex", "simplex")
   add("node", spec.get("source", 0), 0)
   add(column, spec.get("target", 1), 1)
   if kind == "hypergraph" and spec.get("directed"):
      kw.append(("directed", True))
      add("direction", spec.get("direction"), "direction")
   add("node_format", spec.get("node_format", "wide"), "wide")
   add("device", spec.get("device", "cpu"), "cpu")
   return ["net = gsk." + _call(name, spec["edges"], kw)]


def generator_code(function, arguments):
   """
   Code generating a synthetic network with a function of :data:`GENERATORS`.

   :param function: Name of the function.
   :param arguments: Dictionary from parameter name to its value as Python source code (e.g.
      ``"[250] * 4"``), written as given.
   :return: List of lines of code, defining ``net``.
   :raises ValueError: If the function is not a generator, or a value is not a Python expression.
   """
   if function not in GENERATORS:
      raise ValueError(f"unknown generator: {function}")
   for k, v in arguments.items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   kw = [f"{k} = {v.strip()}" for k, v in arguments.items()]
   return [f"net = gsk.{function}({', '.join(kw)})"]


def signature(function, net = None, fit = False):
   """
   Parameters of a generator or of a layout, for the parameter dialog of the interface, leaving
   out those of :data:`HIDDEN` for layouts, ``net`` and ``progress`` for generators: for each, its
   ``name``, its ``default`` as shown in the dialog (Python source code: the default value, or
   for ``None`` its actual value from :data:`SHOWN_DEFAULTS`; an empty string if there is none),
   a ``hint`` for empty fields (from :data:`HINTS`, or ``"required"``),
   its description ``doc`` (from the ``:param:`` lines of the docstring) and the ``widget`` that
   sets it (:func:`widget`).

   :param function: Name of a function of :data:`GENERATORS`, :data:`LAYOUTS`,
      :data:`COMMUNITIES`, :data:`VECTORS` (their parameters of directed networks only for directed
      networks, :func:`vector_hidden`) or :data:`CENTRALITIES` (these get two more parameters: ``attribute``,
      where the values go, and, for ``degree``, ``plot``, see :func:`centrality_code`).
   :param net: The network the function runs on, or ``None``.
   :param fit: For a generator fitted to ``net`` (see :func:`fit_code`): parameters whose default
      ``None`` stands for a value taken from the network are empty, with the hint ``from the
      network`` (yes / no settings get a menu with that as first item), and two parameters come
      first: ``test``, the measure of the randomization test (a menu of :data:`MEASURES` for
      ``net``, first ``none``: one null network only), and ``runs``, its number of null networks.
   :return: Dictionary with ``params``.
   """
   f = getattr(gsk, function)
   doc = inspect.getdoc(f) or ""
   descriptions = {m.group(1): " ".join(m.group(2).split()) for m in re.finditer(r":param (\w+): (.*?)(?=\n:|\Z)", doc, re.S)}
   hidden = vector_hidden(function, net) if function in VECTORS else HIDDEN if function in LAYOUTS else (CENTRALITY_HIDDEN if function in CENTRALITIES else (EPIDEMIC_HIDDEN if function in EPIDEMICS else (BACKBONE_HIDDEN if function in set(BACKBONES) | FILTERS else (PROJECTION_HIDDEN if function in PROJECTIONS else (TRANSFORMATION_HIDDEN if function in TRANSFORMATIONS else {"net", "progress"})))))   # community discovery shows ``attribute``
   params = []
   for p in inspect.signature(f).parameters.values():
      if p.name in hidden or function in VECTORS and p.name == "sparse" and p.default is None:   # None: chosen by the library
         continue
      if p.default is inspect.Parameter.empty or p.default is None:
         shown = SHOWN_DEFAULTS.get((function, p.name), SHOWN_DEFAULTS.get(p.name, ""))
      else:
         shown = literal(p.default)
      if (function, p.name) in EMPTY:
         shown = ""
      hint = HINTS.get((function, p.name), HINTS.get(p.name, "required"))
      yes_no = fit and p.default is None and shown in ("True", "False")
      if fit and p.default is None:   # from the network (the device: that of the network)
         shown, hint = "", HINTS.get(p.name) if p.name == "seed" else ("required" if (function, p.name) in FIT_REQUIRED else "from the network")
         if p.name == "device" and net is not None and net.components:
            shown = literal(net.components[0].edge_index.device.type)
      param = {"name": p.name, "default": shown, "hint": hint, "doc": DOCS.get((function, p.name), descriptions.get(p.name, ""))}
      param["widget"] = {"kind": "choice", "options": ["", "True", "False"]} if yes_no else widget(function, p.name, shown, hint, net)
      if function == "layer_network" and p.name == "layer" and net is not None and net.layer_labels is not None:   # a menu of the layers
         param.update(default = literal(net.layer_labels[0]), widget = {"kind": "choice", "options": [literal(l) for l in net.layer_labels]})
      params.append(param)
   if fit:
      params[:0] = [{"name": "test", "default": "", "hint": "none (one null network)", "doc": "Measure compared with its values on null networks of the model (gsk.randomization_test), or none for one null network only", "widget": {"kind": "choice", "options": [""] + [literal(m) for m in measures(net)]}},
                    {"name": "runs", "default": "100", "hint": "", "doc": "Number of null networks of the randomization test", "widget": {"kind": "int", "min": 1}}]
   if function in HIERARCHICAL_COMMUNITIES:   # the plot of the dendrogram of the levels
      params.append({"name": "tree", "default": "False", "hint": "", "doc": "Open the tree of the levels found in a new tab, laid out radially (also later: Community discovery › Tree of a hierarchy…)", "widget": {"kind": "bool"}})
   if function == "nestedness":   # the method of the isocline
      params.append({"name": "isocline", "default": "\"fill\"", "hint": "", "doc": "How the isocline is drawn (gsk.nestedness_isocline): with the area below it equal to the fill of the matrix (fill), or closest to the boundary of the ones if the matrix were perfectly nested (fit)", "widget": {"kind": "choice", "options": ["\"fill\"", "\"fit\""]}})
   if function in EPIDEMICS:   # the name of the run
      params.insert(0, {"name": "name", "default": "", "hint": "required", "doc": "Name of the run: its table of states is stored as epidemics[name]", "widget": {"kind": "string"}})
   if function in CENTRALITIES:   # where the values go, and the plot of the degree distribution
      what = "edge" if CENTRALITIES[function][1] == "edges" else "node"
      params.insert(0, {"name": "attribute", "default": literal(CENTRALITIES[function][0]), "hint": "", "doc": f"Name of the {what} attribute written" + (" (with _hub and _authority)" if function == "hits" else ""), "widget": {"kind": "string"}})
      if function == "degree":
         params.append({"name": "plot", "default": "False", "hint": "", "doc": "Plot the complementary cumulative degree distribution with the best power-law fit and the tests of :func:`~graphoskop.degree_power_law` (unweighted degrees)", "widget": {"kind": "bool"}})
   return {"params": params}


def widget(function, name, shown, hint, net = None):
   """
   How the parameter dialog lets the user set a parameter, with as little typing as possible:

   * ``{"kind": "device", "gpu": bool}`` for ``device`` (a GPU switch, off without a GPU);
   * ``{"kind": "bool"}`` for defaults ``True`` and ``False`` (a check box);
   * ``{"kind": "choice", "options": [...]}`` for :data:`OPTIONS` (a menu), and for the
     ``node_type`` of the bipartite projections (:data:`PROJECTIONS`), the two node types of the
     network;
   * ``{"kind": "attribute", "choices": "nodes" or "edges", "multiple": bool}`` for the
     attribute names of :data:`CHOICES`, in layouts and analyses (a menu of the attributes of
     the network, after the values of :data:`EXTRA_CHOICES`; check boxes for several); with
     ``"empty"``, the value of the first item of the menu (its hint), e.g. ``None`` for the
     weights of the backbones, whose default is ``"weight"``; with ``"numeric"``, only the
     numeric attributes (the vectors of the node vector analyses, :data:`VECTORS`, and their
     weights);
   * ``{"kind": "string"}`` for :data:`STRINGS` and ``{"kind": "node", "numeric": bool}`` for
     :data:`NODES` (fields without quotes; node ids are numbers if the ids of the network are);
   * ``{"kind": "list"}`` for :data:`LISTS` (a field of numbers separated by commas);
   * ``{"kind": "optional"}`` for numbers whose default ``None`` has no value of its own (an
     empty field with a hint);
   * ``{"kind": "range", "min", "max", "step"}`` for :data:`RANGES` (a slider and a number);
   * ``{"kind": "int", "min"}`` for :data:`INTEGERS` and ``{"kind": "number", "min", "step"}``
     for :data:`NUMBERS` (numbers with arrows);
   * ``{"kind": "text"}`` for any other parameter (Python source code).

   :param function: Name of the function.
   :param name: Name of the parameter.
   :param shown: Default shown in the dialog (see :func:`signature`).
   :param hint: Hint of an empty field.
   :param net: The network the function runs on (for node ids), or ``None``.
   :return: Dictionary with ``kind`` and its settings.
   """
   tables = [RANGES, INTEGERS, NUMBERS]
   own = [t for t in tables if (function, name) in t]   # settings of the function first
   def get(table):
      if own:
         return table.get((function, name)) if table is own[0] else None
      return table.get(name)
   if name == "device":
      import torch
      return {"kind": "device", "gpu": torch.cuda.is_available()}
   if function in VECTORS and name in ("a", "b", "attributes", "weight"):   # numeric attributes only
      return {"kind": "attribute", "choices": "edges" if name == "weight" else "nodes", "multiple": name == "attributes", "extra": [], "numeric": True}
   if (function, name) in OPTIONS and shown in ("True", "False"):   # more than yes and no
      return {"kind": "choice", "options": OPTIONS[(function, name)]}
   if shown in ("True", "False"):
      return {"kind": "bool"}
   if function == "quotient_network" and name == "sets":   # sets of nodes: the combinations of values of node attributes
      return {"kind": "attribute", "choices": "nodes", "multiple": True, "extra": []}
   if function in PROJECTIONS and name == "node_type":
      return {"kind": "choice", "options": [literal(v) for v in (net.bipartite_labels if net is not None and net.bipartite_labels is not None else [])]}
   options = OPTIONS.get((function, name), OPTIONS.get(name) if (function, name) not in CHOICES else None)
   if options is not None:
      return {"kind": "choice", "options": options}
   choice = CHOICES.get((function, name), CHOICES.get(name) if function in LAYOUTS or (function in COMMUNITIES | set(CENTRALITIES) | set(BACKBONES) | FILTERS | set(PROJECTIONS) | set(TRANSFORMATIONS) | set(LINK_PREDICTIONS) | set(GNNS) | set(EXPLORATIONS) | set(HIERARCHIES) | set(STATISTICS) and name == "weight") else None)
   if choice is not None:
      w = {"kind": "attribute", "choices": choice[0], "multiple": choice[1], "extra": EXTRA_CHOICES.get((function, name), [])}
      if name == "weight" and function in set(BACKBONES) | FILTERS | set(PROJECTIONS) and shown not in ("", "None"):
         w["empty"] = "None"   # the hint (unweighted) is weight = None, not the default "weight"
      return w
   if (function, name) in STRINGS or name in STRINGS:
      return {"kind": "string"}
   if name in NODES:
      ids = _ids(net) if net is not None else []
      return {"kind": "node", "numeric": len(ids) > 0 and all(isinstance(i, (int, np.integer)) for i in ids[:100])}
   if (function, name) in LISTS or (name in LISTS and not own):
      return {"kind": "list"}
   if shown == "" and hint != "required":
      return {"kind": "optional"}
   if get(RANGES) is not None:
      lo, hi, step = get(RANGES)
      return {"kind": "range", "min": lo, "max": hi, "step": step}
   if get(INTEGERS) is not None:
      return {"kind": "int", "min": get(INTEGERS)}
   if get(NUMBERS) is not None:
      lo, step = get(NUMBERS)
      return {"kind": "number", "min": lo, "step": step}
   return {"kind": "text"}


def layout_code(ns, function = "grid_layout", arguments = None):
   """
   Code of a layout (or a transformation of a layout, edge bundling, or community discovery) of
   the network of a tab.
   Hypergraphs and simplicial complexes are laid out (and drawn) as their incidence networks,
   defined as ``inc`` the first time. The curves of :func:`~graphoskop.bundle_edges` and
   :func:`~graphoskop.route_edges` (:data:`CURVES`) are stored in ``bends`` (:data:`BENDS`).

   :param ns: Namespace of the tab, with the network ``net``.
   :param function: One of :data:`LAYOUTS` or :data:`COMMUNITIES`; by default
      :func:`~graphoskop.grid_layout`, the first layout of every network (the cheapest of the
      library).
   :param arguments: Dictionary from parameter name to its value as Python source code (e.g.
      ``'["region", "city"]'``), written as given, or ``None``.
   :return: List of lines of code.
   :raises ValueError: If ``function`` is not in :data:`LAYOUTS` or :data:`COMMUNITIES`, or a
      value is not a Python expression.
   """
   if function not in LAYOUTS | COMMUNITIES:
      raise ValueError(f"unknown layout function: {function}")
   for k, v in (arguments or {}).items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   higher = ns["net"].higher_order is not None
   lines = ["inc = gsk.incidence_network(net)"] if higher and "inc" not in ns else []
   kw = [f"{k} = {v.strip()}" for k, v in (arguments or {}).items()]
   call = f"gsk.{function}({', '.join(['inc' if higher else 'net'] + kw)})"
   return lines + ["bends = " + call if function in CURVES else call]


def node_size(x, y):
   """
   Default diameter of the drawn nodes for a layout: 70% of the median distance from a node to
   the nearest other node, so that nodes in typical places do not overlap, whatever the scale of
   the layout (grid spacing, edge length, UMAP units).

   :param x: Horizontal positions (NumPy array).
   :param y: Vertical positions.
   :return: Diameter, in layout units.
   """
   from scipy.spatial import cKDTree
   X = np.column_stack([x, y])
   if len(X) < 2:
      return 0.5
   d = cKDTree(X).query(X, k = 2)[0][:, 1]
   d = d[d > 0]
   if len(d) == 0:
      return 0.5
   return float(0.7 * np.median(d))


def _ids(net):
   """
   Input ids of the nodes, in the network wide order (object array).
   """
   return np.concatenate([np.asarray(ids, dtype = object) for ids in net.node_ids]) if net.node_ids else np.zeros(0, object)


def _edges(net):
   """
   Sources and targets of the input edges, as positions in the network wide node order.
   """
   src, dst = [], []
   for c in range(len(net.components)):
      ei = net._input_edges(c)[0].cpu().numpy()
      src.append(ei[0] + net.node_offsets[c])
      dst.append(ei[1] + net.node_offsets[c])
   return (np.concatenate(src), np.concatenate(dst)) if src else (np.zeros(0, int), np.zeros(0, int))


def node_values(net, name):
   """
   Values of a node attribute, in the network wide order: float64 array for numeric attributes,
   object array for qualitative ones.
   """
   import torch
   if name in net.node_attributes:
      return torch.cat(net.get_node_attribute(name, None)).double().cpu().numpy()
   return np.concatenate([np.asarray(v, dtype = object) for v in net.get_node_attribute(name, None)])


def _column(net, name):
   """
   Values of a node attribute for the data panel and the attribute summaries: the categorical
   column itself for qualitative attributes of wide node tables (no Python object per value), else
   :func:`node_values`.
   """
   if name in net.qualitative_node_attributes and net.node_format == "wide" and net.higher_order is None:
      import pandas as pd
      return pd.Categorical(net.qualitative_node_attr[name])
   return node_values(net, name)


def node_matrix(net):
   """
   All the numeric node attributes of a network at once (columns in the order of
   ``net.node_attributes``, rows in the network wide order): a :class:`scipy.sparse.csr_matrix`
   for long node tables (missing values are 0, as for the library), a float64 array otherwise.
   Reading the attributes one by one copies the whole sparse table each time (seconds for
   thousands of attributes).
   """
   import scipy.sparse as sp, torch
   blocks = []
   for c, data in enumerate(net.components):
      X = data.node_attr
      rows = net.bipartite_split[c] if net.higher_order is not None else data.num_nodes
      if X.layout == torch.sparse_csr:
         X = sp.csr_matrix((X.values().double().cpu().numpy(), X.col_indices().cpu().numpy(), X.crow_indices().cpu().numpy()), shape = tuple(X.shape))
      else:
         X = X.double().cpu().numpy()
      blocks.append(X[:rows])
   if any(sp.issparse(b) for b in blocks):
      return sp.vstack(blocks, format = "csr")
   return np.vstack(blocks) if blocks else np.zeros((0, len(net.node_attributes)))


def _sparse_nodes(net):
   """
   Whether the numeric node attributes of a network are a sparse table (long node tables).
   """
   import torch
   return any(c.node_attr is not None and c.node_attr.layout == torch.sparse_csr for c in net.components)


def _numeric_columns(net, chosen):
   """
   Values of the numeric node attributes ``chosen`` (float64 arrays, as :func:`node_values`):
   from :func:`node_matrix` for sparse tables, one by one otherwise (no copy of the whole table).
   """
   if not _sparse_nodes(net):
      return {a: node_values(net, a) for a in chosen}
   j = {a: i for i, a in enumerate(net.node_attributes)}
   M = node_matrix(net)[:, [j[a] for a in chosen]].toarray()
   return {a: M[:, i] for i, a in enumerate(chosen)}


def _distinct_numbers(net):
   """
   Sorted distinct numbers (NaN left out) of every numeric node attribute; the implicit zeros of
   sparse tables count (:func:`node_matrix`).
   """
   if not _sparse_nodes(net):
      return [np.unique(v[~np.isnan(v)]) for v in map(lambda a: node_values(net, a), net.node_attributes)]
   X = node_matrix(net).tocsc()
   n, found = X.shape[0], []
   for j in range(X.shape[1]):
      v = X.data[X.indptr[j]:X.indptr[j + 1]]
      v = v[~np.isnan(v)]
      found.append(np.unique(np.append(v, 0.) if X.indptr[j + 1] - X.indptr[j] < n else v))
   return found


def _qualitative_uniques(net):
   """
   Distinct values of every qualitative node attribute (object arrays; NaN where some node has no
   value), reading long node tables once.
   """
   names = list(net.qualitative_node_attributes)
   if net.node_format == "wide" or net.higher_order is not None or not names:
      return {a: _values_once(_column(net, a)) for a in names}
   Q, n = net.qualitative_node_attr, net.number_of_nodes()
   found = {}
   for a, g in Q.groupby("attribute", observed = True, sort = False):
      v = _values_once(np.asarray(g["value"], dtype = object))
      found[str(a)] = np.append(v, np.nan) if g["node"].nunique() < n else v
   return {a: found.get(a, np.array([np.nan], dtype = object)) for a in names}


def _values_once(values):
   """
   Distinct values of an object array, missing ones included (all the values if some cannot be
   hashed, e.g. lists), for :func:`attributes`.
   """
   import pandas as pd
   if isinstance(values, pd.Categorical):   # its categories, without a Python object per value
      v = np.asarray(values.categories, dtype = object)
      return np.append(v, np.nan) if (values.codes < 0).any() else v
   try:
      return pd.unique(pd.Series(values, dtype = object)).astype(object)
   except TypeError:
      return values


def node_row(net, position):
   """
   Values of all the node attributes of one node (position in the network wide order), for the
   table of one node of the data panel: dictionary from name to value (NaN if missing).
   """
   if _sparse_nodes(net):
      row = node_matrix(net)[position].toarray()[0]
   elif net.higher_order is None and net.node_attributes:   # the row of its component
      c = int(np.searchsorted(net.node_offsets, position, side = "right")) - 1
      row = net.components[c].node_attr[position - int(net.node_offsets[c])].double().cpu().numpy()
   else:
      row = [node_values(net, a)[position] for a in net.node_attributes]
   values = dict(zip(net.node_attributes, row))
   Q = net.qualitative_node_attr
   if net.node_format == "wide" or net.higher_order is not None:
      values.update({a: _column(net, a)[position] for a in net.qualitative_node_attributes})
   else:
      values.update({a: np.nan for a in net.qualitative_node_attributes})
      r = Q[(Q["node"] == position).to_numpy()]
      values.update({str(a): v for a, v in zip(r["attribute"], r["value"])})
   return values


def matching(net, *patterns, edges = False, numeric = False):
   """
   Names of the node (or edge) attributes of a network matching patterns, in the order of the
   patterns (each in the order of the attributes of the network, each name once): ``*`` stands
   for any text, ``?`` for one character, ``[...]`` for one of the characters inside (see
   :mod:`fnmatch`; a plain name matches itself). The parameter dialog of the interface writes it
   for the check boxes of attribute patterns (``"label_*"`` checks all the attributes starting
   with ``label_``), so that the code does not list thousands of names. A function of the
   interface, defined in the namespace of every tab (:data:`MATCHING_NOTE`).

   :param net: A :class:`~graphoskop.Network`.
   :param patterns: Patterns (or names).
   :param edges: Edge attributes (else node attributes).
   :param numeric: Only numeric attributes.
   :return: List of names.
   """
   import fnmatch
   names = list(net.edge_attributes if edges else net.node_attributes)
   if not numeric:
      names += list(net.qualitative_edge_attributes if edges else net.qualitative_node_attributes)
   found = {}
   for p in patterns:
      found.update(dict.fromkeys(fnmatch.filter(names, p) if any(c in p for c in "*?[") else [p] if p in names else []))
   return list(found)


#: Comment of the code before the first :func:`matching`, a function of the interface.
MATCHING_NOTE = "# matching: names of the attributes matching patterns (* any text), a function of the graphical interface"


def _source_value(ns, source):
   """
   Value of the Python source of an argument written by the parameter dialog: a literal, or a
   call of :func:`matching` (patterns of attribute names) evaluated in the namespace ``ns``.
   """
   return ast.literal_eval(source) if not source.lstrip().startswith("matching(") else eval(source, dict(ns, matching = matching))


#: Functions of the interface that the code of a tab can call (defined in this worker): the code
#: imports them from it the first time it calls one (:func:`interface_import`).
INTERFACE_FUNCTIONS = ["attribute_values", "block_order", "derived", "edge_values", "gnn_evaluation", "dendrogram_tree", "groups", "incidence_values", "matching", "move_nodes", "nestedness_matrix", "node_sizes", "node_values", "node_with", "plot_figure", "set_edge_values", "synchronization_curves"]


def interface_import():
   """
   Lines of code importing the functions of the interface (:data:`INTERFACE_FUNCTIONS`) from this
   worker, its folder added to the path of Python, so that the code of a tab runs as it is in
   any Python with the library, e.g.

   .. code-block:: python

      import sys
      sys.path.append("/opt/graphoskop/gui/python")   # the folder of gsk_worker.py, the worker of the graphical interface
      from gsk_worker import derived, groups, matching, move_nodes, node_sizes, node_with, set_edge_values   # functions of the graphical interface
   """
   folder = os.path.dirname(os.path.abspath(__file__))
   return ["import sys", f"sys.path.append({literal(folder)})   # the folder of gsk_worker.py, the worker of the graphical interface",
           f"from gsk_worker import {', '.join(INTERFACE_FUNCTIONS)}   # functions of the graphical interface"]


def _matching_note(ns, code):
   """
   The comments :data:`MATCHING_NOTE`, :data:`DERIVED_NOTE` and :data:`NODE_WITH_NOTE` before
   ``code`` (lines) if it calls :func:`matching`, :func:`derived` or :func:`node_with` for the
   first time in the tab; before them, :func:`interface_import` if it calls a function of the
   interface and the code of the tab does not import them yet.
   """
   notes = [note for call, note in (("matching(", MATCHING_NOTE), ("derived(", DERIVED_NOTE), ("node_with(", NODE_WITH_NOTE)) if any(call in c for c in code) and note not in ns["__code__"]]
   calls = re.compile(r"(?<![\w.])(" + "|".join(INTERFACE_FUNCTIONS) + r")\(")
   imported = any(c.startswith("from gsk_worker import") for c in ns["__code__"])
   needed = not imported and any(calls.search(c.split("#")[0]) for c in code)
   return (interface_import() if needed else []) + notes + code


def edge_values(net, name):
   """
   Values of an edge attribute, for the input edges in the network wide order (as :func:`_edges`):
   float64 array for numeric attributes, object array for qualitative ones.
   """
   if name in net.edge_attributes:
      j = list(net.edge_attributes).index(name)
      return np.concatenate([net._input_edges(c)[1][:, j].double().cpu().numpy() for c in range(len(net.components))])
   return net.qualitative_edge_attr[name].to_numpy(dtype = object)


def _rgba(color):
   """
   RGBA floats of a ``#rrggbb`` (or ``#rrggbbaa``) color.
   """
   c = color.lstrip("#")
   v = [int(c[i:i + 2], 16) / 255. for i in range(0, len(c), 2)]
   return v + [1.] * (4 - len(v))


def distinct(values):
   """
   Distinct values of an attribute, as texts (:func:`text`) in their order (numbers, and texts
   that are numbers, numerically first, then other texts alphabetically), with the position of the text of each value in that order (-1
   for missing values): the order of qualitative colors, and the numbers given to qualitative
   values on quantitative scales and sizes.

   :param values: Values (NumPy array).
   :return: List of texts and int64 array of positions.
   """
   import pandas as pd
   try:   # each distinct value once (values of qualitative attributes repeat a lot), then their texts
      raw, found = (values.codes, values.categories) if isinstance(values, pd.Categorical) else pd.factorize(pd.Series(values, dtype = object), use_na_sentinel = True)
      merged, uniques = pd.factorize(pd.Series([text(u) for u in found], dtype = object), use_na_sentinel = True)   # e.g. 1 and 1.0
      codes = np.append(merged, -1)[raw]
   except TypeError:   # unhashable values (lists): the text of every value
      codes, uniques = pd.factorize(pd.Series(values, dtype = object).map(text), use_na_sentinel = True)
   uniques = list(uniques)
   def key(i):
      try:
         return (0, float(uniques[i]), "")
      except ValueError:
         return (1, 0., uniques[i])
   order = sorted(range(len(uniques)), key = key)
   rank = np.empty(len(uniques) + 1, dtype = np.int64)
   rank[order] = np.arange(len(uniques))
   rank[-1] = -1
   return [uniques[i] for i in order], rank[codes]


def numbers(values):
   """
   Numbers of the values of an attribute for quantitative scales and sizes: the values of numeric
   attributes, the positions in :func:`distinct` of qualitative ones; NaN for missing values.
   """
   if values.dtype.kind == "f":
      return values.astype(float)
   pos = distinct(values)[1].astype(float)
   pos[pos < 0] = np.nan
   return pos


def transform(x, log):
   """
   The numbers of a scale, logarithmic if ``log``: :math:`\\log_{10} x` if all values are positive,
   else :math:`\\log_{10} (x - m + d)` with :math:`m` the smallest value and :math:`d` the
   smallest positive difference from it (the distance between the two smallest values): values
   at or below 0 (e.g. betweenness 0) are moved above 0 by as little as the data allow, so that the
   scale spreads small values as the logarithm of positive values does, whatever their magnitude.

   :param x: Numbers (NaN for missing values).
   :param log: Whether to take logarithms.
   :return: Function from numbers to scale positions (applied to stops as well).
   """
   if not log:
      return lambda v: np.asarray(v, dtype = float)
   shift = _shift(x)
   return lambda v: np.log10(np.asarray(v, dtype = float) + shift)


def _shift(x):
   """
   What :func:`transform` adds to the values before their logarithm: 0 if all are positive, else
   :math:`d - m` (see there; 1 - m if all values are equal).
   """
   ok = x[~np.isnan(x)]
   m = ok.min() if len(ok) else 1.
   if m > 0:
      return 0.
   above = ok[ok > m] - m
   return (above.min() if len(above) else 1.) - m


def _inverse(x, log):
   """
   Inverse of :func:`transform` (the scale positions back to numbers).
   """
   if not log:
      return lambda t: np.asarray(t, dtype = float)
   shift = _shift(x)
   return lambda t: 10. ** np.asarray(t, dtype = float) - shift


def natural_breaks(x, k):
   """
   Natural breaks (Jenks; Fisher's exact optimization): the ``k`` classes of consecutive sorted
   values with the smallest total squared deviation from their means, on at most
   :data:`MAX_BREAK_VALUES` values (quantiles of the values beyond).

   :param x: Values (no NaN).
   :param k: Number of classes.
   :return: The ``k + 1`` class boundaries: smallest value, first value of every class but the
      first, largest value.
   """
   x = np.sort(np.asarray(x, dtype = float))
   if len(x) > MAX_BREAK_VALUES:
      x = np.quantile(x, np.linspace(0, 1, MAX_BREAK_VALUES))
   n = len(x)
   k = max(1, min(k, n))
   s1 = np.concatenate([[0.], np.cumsum(x)])
   s2 = np.concatenate([[0.], np.cumsum(x * x)])
   D = np.full((k + 1, n + 1), np.inf)
   D[0, 0] = 0.
   B = np.zeros((k + 1, n + 1), dtype = np.int64)
   for c in range(1, k + 1):
      for j in range(c, n + 1):
         i = np.arange(c - 1, j)
         s = s1[j] - s1[i]
         cost = D[c - 1, i] + (s2[j] - s2[i]) - s * s / (j - i)
         m = int(np.argmin(cost))
         D[c, j], B[c, j] = cost[m], i[m]
   starts, j = [], n
   for c in range(k, 0, -1):
      j = B[c, j]
      starts.append(j)
   starts = sorted(starts)[1:]
   return [float(x[0])] + [float(x[i]) for i in starts] + [float(x[-1])]


def breaks(values, method, k, log = False):
   """
   Values of the ``k`` color stops of a quantitative scale: ``"quantiles"`` (stop ``i`` at the
   quantile ``i / (k - 1)``), ``"equal"`` (evenly spaced from the smallest to the largest value)
   or ``"natural"`` (the boundaries of ``k - 1`` classes of :func:`natural_breaks`). With ``log``,
   stops are placed on the logarithmic scale (:func:`transform`) and given back as values.

   :param values: Values of the attribute (qualitative ones as :func:`numbers`).
   :param method: ``"quantiles"``, ``"equal"`` or ``"natural"``.
   :param k: Number of stops, at least 2.
   :param log: Whether the scale is logarithmic.
   :return: List of ``k`` stop values.
   """
   x = numbers(values)
   t = transform(x, log)(x)
   t = t[~np.isnan(t)]
   if len(t) == 0:
      return [0.] * k
   if method == "quantiles":
      s = np.quantile(t, np.linspace(0, 1, k))
   elif method == "natural":
      s = np.array(natural_breaks(t, k - 1))
      s = np.interp(np.linspace(0, len(s) - 1, k), np.arange(len(s)), s) if len(s) != k else s
   else:
      s = np.linspace(t.min(), t.max(), k)
   return [float(v) for v in _inverse(x, log)(s)]


def colors(values, spec, default):
   """
   RGBA colors (float32, one row per value) of nodes or edges, from a color specification:

   * ``{"attribute": None, "color": "#rrggbb"}``: one color for all;
   * ``{"attribute": name, "scale": "quantitative", "stops": [[value, "#rrggbb"], ...], "log":
     bool, "missing": "#rrggbb"}``: colors interpolated linearly between the stops (sorted by
     value), on the logarithmic scale of :func:`transform` with ``log``, the first and last color
     beyond them. Qualitative attributes count as the positions of their values (:func:`numbers`);
   * ``{"attribute": name, "scale": "qualitative", "palette": [colors], "overrides": [[text,
     "#rrggbb"], ...], "missing": "#rrggbb"}``: every distinct value (:func:`distinct`) gets the
     next color of the palette, again from the first after the last one, unless it has a color in
     ``overrides``.

   Missing values get the ``missing`` color.

   :param values: Values of the attribute (ignored without an attribute), or their number.
   :param spec: Color specification, or ``None`` for ``default``.
   :param default: Color without a specification.
   :return: Array of shape ``(n, 4)``.
   """
   n = values if isinstance(values, int) else len(values)
   spec = spec or {"attribute": None, "color": default}
   if spec.get("attribute") is None:
      return np.tile(np.array(_rgba(spec.get("color", default)), dtype = np.float32), (n, 1))
   missing = np.array(_rgba(spec.get("missing", MISSING_COLOR)), dtype = np.float32)
   out = np.tile(missing, (n, 1))
   if spec.get("scale") == "quantitative":
      x = numbers(values)
      f = transform(x, spec.get("log", False))
      stops = sorted(spec["stops"], key = lambda s: float(s[0]))
      xs = f([float(s[0]) for s in stops])
      cs = np.array([_rgba(s[1]) for s in stops])
      t = f(x)
      ok = ~np.isnan(t)
      for c in range(4):
         out[ok, c] = np.interp(t[ok], xs, cs[:, c])
      return out
   texts, pos = distinct(values)
   palette = [_rgba(c) for c in spec.get("palette") or [default]]
   table = np.array([palette[i % len(palette)] for i in range(len(texts))] + [list(missing)], dtype = np.float32)
   overrides = {str(t): _rgba(c) for t, c in spec.get("overrides", [])}
   for i, t in enumerate(texts):
      if t in overrides:
         table[i] = overrides[t]
   return table[pos]   # position -1: the last row, missing


def sizes(values, spec):
   """
   Relative sizes of edges (1 is the default width) and nodes (:func:`node_sizes`, with
   ``area``), from a size specification:
   ``{"attribute": None, "size": s}`` (all ``s``), or ``{"attribute": name, "min": a, "max": b,
   "log": bool, "area": bool}``: the smallest value (on the logarithmic scale of :func:`transform`
   with ``log``) gets ``a``, the largest ``b``, and the others are interpolated linearly in between,
   in the squares of the sizes with ``area`` (node diameters, so that areas are proportional), in
   the sizes otherwise (edge widths). Qualitative attributes count as the positions of their
   values (:func:`numbers`); missing values get ``a``.

   :param values: Values of the attribute (ignored without an attribute), or their number.
   :param spec: Size specification, or ``None`` for 1.
   :return: float64 array.
   """
   n = values if isinstance(values, int) else len(values)
   if not spec or spec.get("attribute") is None:
      return np.full(n, float((spec or {}).get("size", 1.)))
   x = numbers(values)
   t = transform(x, spec.get("log", False))(x)
   ok = ~np.isnan(t)
   lo, hi = (t[ok].min(), t[ok].max()) if ok.any() else (0., 1.)
   u = np.zeros(n)
   u[ok] = (t[ok] - lo) / (hi - lo) if hi > lo else 0.5
   a, b = float(spec.get("min", 0.5)), float(spec.get("max", 2.))
   if spec.get("area", False):
      return np.sqrt(a * a + u * (b * b - a * a))
   return a + u * (b - a)


def opacities(values, spec):
   """
   Opacities of nodes or edges (0 transparent, 1 opaque), from a specification:
   ``{"attribute": None, "opacity": a}`` (all ``a``), or ``{"attribute": name, "min": a, "max":
   b, "log": bool}``: the smallest value gets ``a``, the largest ``b``, the others in between
   linearly (:func:`sizes` without ``area``; missing values get ``a``).

   :param values: Values of the attribute (ignored without an attribute), or their number.
   :param spec: Opacity specification, or ``None`` for 1.
   :return: float64 array, between 0 and 1.
   """
   n = values if isinstance(values, int) else len(values)
   if not spec or spec.get("attribute") is None:
      return np.full(n, float((spec or {}).get("opacity", 1.)))
   return np.clip(sizes(values, dict(spec, area = False)), 0., 1.)


def _helpers():
   """
   Functions of the interface defined in the namespace of every tab, for the code they write:
   :func:`node_sizes`, :func:`move_nodes`, :func:`groups`, :func:`set_edge_values`,
   :func:`matching`, :func:`derived`, :func:`node_with`, and for the code of saved plots
   :func:`plot_figure` with what their data need (:func:`attribute_values`, :func:`node_values`,
   :func:`edge_values`, :func:`nestedness_matrix`, :func:`gnn_evaluation`, :func:`block_order`,
   :func:`synchronization_curves`, :func:`dendrogram_tree`), and :func:`incidence_values` for the values of hypergraphs
   and simplicial complexes drawn on their incidence networks.
   """
   return {"node_sizes": node_sizes, "move_nodes": move_nodes, "groups": groups, "set_edge_values": set_edge_values, "matching": matching, "derived": derived, "node_with": node_with, "plot_figure": plot_figure, "attribute_values": attribute_values, "node_values": node_values, "edge_values": edge_values, "nestedness_matrix": nestedness_matrix, "gnn_evaluation": gnn_evaluation, "block_order": block_order, "synchronization_curves": synchronization_curves, "incidence_values": incidence_values, "dendrogram_tree": dendrogram_tree}


def incidence_values(inc, nodes = None, hyperedges = None):
   """
   Values of the nodes and of the hyperedges (or facets) of a hypergraph or a simplicial complex as
   a node attribute of its incidence network (:func:`~graphoskop.incidence_network`, the network
   the interface draws): in every component, the nodes come first, then the hyperedges, so the
   values of a component are those of its nodes followed by those of its hyperedges; missing ones
   are NaN. A function of the interface, not of the library.

   :param inc: The incidence network.
   :param nodes: Values of the nodes, one vector per component (as the library gives them with
      ``component = None``), or ``None``.
   :param hyperedges: Values of the hyperedges, one vector per component, or ``None``.
   :return: List of float64 vectors, one per component, for
      :meth:`~graphoskop.Network.set_node_attribute`.
   """
   import torch
   out = []
   for c in range(len(inc.components)):
      k, n = int(inc.bipartite_split[c]), len(inc.node_ids[c])
      v = torch.full((n,), float("nan"), dtype = torch.float64, device = inc.components[c].edge_index.device)
      if nodes is not None:
         v[:k] = torch.as_tensor(nodes[c], dtype = torch.float64).to(v.device)
      if hyperedges is not None:
         v[k:] = torch.as_tensor(hyperedges[c], dtype = torch.float64).to(v.device)
      out.append(v)
   return out


def move_nodes(net, positions, attribute = "pos"):
   """
   Moves nodes, as dragged in the visualization panel: sets the position attributes
   ``<attribute>_x`` and ``<attribute>_y`` of the given nodes. A function of the interface, not of
   the library: it is defined in the namespace of every tab, for the code of the moves
   (:data:`MOVE_NOTE`).

   :param net: A :class:`~graphoskop.Network` with positions.
   :param positions: Dictionary from node id to its new position ``(x, y)``.
   :param attribute: Prefix of the position attributes.
   """
   import torch
   index = {v: k for k, v in enumerate(_ids(net).tolist())}
   rows = torch.tensor([index[v] for v in positions], dtype = torch.long)
   offs = net.node_offsets
   for j, axis in enumerate("xy"):
      if f"{attribute}_{axis}" not in net.node_attributes:   # no positions yet (a new network): 0 for the others
         values = torch.zeros(net.number_of_nodes())
      else:
         values = torch.cat(net.get_node_attribute(f"{attribute}_{axis}", None)).clone()
      values[rows.to(values.device)] = torch.tensor([float(xy[j]) for xy in positions.values()], dtype = values.dtype, device = values.device)
      net.set_node_attribute(f"{attribute}_{axis}", [values[o:o2] for o, o2 in zip(offs[:-1], offs[1:])], None)


def _id_literal(v):
   """
   Python literal of a node id (numbers, texts, or tuples of them in multilayer networks).
   """
   if isinstance(v, tuple):
      return "(" + ", ".join(_id_literal(x) for x in v) + ("," if len(v) == 1 else "") + ")"
   if isinstance(v, np.generic):
      v = v.item()
   return literal(v) if isinstance(v, (str, int, float, bool)) else repr(v)


def node_sizes(net, attribute = None, smallest = 0.5, largest = 2., log = False):
   """
   Relative diameters of the drawn nodes (1 is the default diameter), from a node attribute: the
   node with the smallest value gets ``smallest``, the one with the largest value ``largest``, and
   the areas of the others grow linearly with their values in between (:func:`sizes` with
   ``area``; qualitative values count as their positions in :func:`distinct`, missing values get
   ``smallest``; ``log`` places the values on a logarithmic scale). A plotting utility of the
   interface, not of the library: it is also defined in the namespace of every tab, for the code
   of :func:`size_code`.

   :param net: A :class:`~graphoskop.Network`.
   :param attribute: Name of a node attribute, or ``None`` for 1 for all nodes.
   :param smallest: Size of the node with the smallest value.
   :param largest: Size of the node with the largest value.
   :param log: Whether the values are placed on a logarithmic scale.
   :return: float64 array, one value per node in the network wide order.
   """
   if attribute is None:
      return np.ones(len(_ids(net)))
   return sizes(node_values(net, attribute), {"attribute": attribute, "min": smallest, "max": largest, "log": log, "area": True})


def positions(net, style, axis, attribute = "pos"):
   """
   Coordinates of the nodes on one axis (``"x"`` or ``"y"``, y pointing up): the numeric node
   attribute chosen for that axis in the ``position`` of the style (``{"x": name, "y": name}``,
   ``None`` for the layout), which overrides the layout; else the layout, ``<attribute>_x`` or
   ``<attribute>_y``. Nodes without a value of the chosen attribute are put at its smallest value.

   :raises ValueError: If the chosen attribute is not a numeric node attribute.
   """
   import torch
   name = (style.get("position") or {}).get(axis)
   if name is None:
      return torch.cat(net.get_node_attribute(f"{attribute}_{axis}", None)).double().cpu().numpy()
   if name not in net.node_attributes:
      raise ValueError(f"position: {name!r} is not a numeric node attribute")
   v = node_values(net, name)
   ok = ~np.isnan(v)
   return np.where(ok, v, v[ok].min() if ok.any() else 0.)


def _positioned(ns):
   """
   Whether the position of the style of a tab comes from node attributes (see :func:`positions`).
   """
   return any((ns.get("__style__", {}).get("position") or {}).values())


def default_style(net):
   """
   Style of a new drawing: ``nodes`` and ``edges`` (colors, see :func:`colors`), ``node_size`` and
   ``edge_width`` (see :func:`sizes`), ``node_opacity`` and ``edge_opacity`` (see
   :func:`opacities`), ``position`` (``{"x": None, "y": None}``: the layout, see
   :func:`positions`), ``labels`` (``{"attribute": None}``: no labels; ``"ids": True`` for the
   node ids, see :func:`node_labels`), ``label_size`` (``{"attribute": None, "size": 1.}``, see
   :func:`label_sizes`), ``node_chart`` (``{}``: none; see :func:`chart_slices`). Nodes are
   :data:`NODE_COLOR` and edges :data:`EDGE_COLOR`, all of the
   default size and opaque; in bipartite networks (and incidence networks of hypergraphs and
   complexes), nodes are colored by type, the second type (hyperedges, facets)
   :data:`TYPE1_COLOR`.
   """
   nodes = {"attribute": None, "color": NODE_COLOR}
   if net.bipartite_labels is not None:
      a, b = (text(v) for v in net.bipartite_labels)
      if b == "node":
         a, b = b, a
      nodes = {"attribute": gsk.BIPARTITE, "scale": "qualitative", "palette_name": "Set1", "palette": [NODE_COLOR, TYPE1_COLOR], "overrides": [[a, NODE_COLOR], [b, TYPE1_COLOR]], "missing": MISSING_COLOR}
   return {"nodes": nodes, "edges": {"attribute": None, "color": EDGE_COLOR}, "node_size": {"attribute": None, "size": 1.}, "edge_width": {"attribute": None, "size": 1.}, "node_opacity": {"attribute": None, "opacity": 1.}, "edge_opacity": {"attribute": None, "opacity": 1.}, "position": {"x": None, "y": None}, "labels": {"attribute": None}, "label_size": {"attribute": None, "size": 1.}, "node_chart": {}}


def prune_style(style, net):
   """
   A style without the attributes the network does not have (any more): the visual attributes
   mapped to a missing node or edge attribute go back to their default (:func:`default_style`),
   the positions to the layout. Happens to derived networks (bipartite projections drop the node
   types) and after an attribute is removed.

   :param style: Style (see :func:`default_style`), changed in place.
   :param net: The drawn network.
   :return: ``style``.
   """
   nodes = set(net.node_attributes) | set(net.qualitative_node_attributes) | ({gsk.BIPARTITE} if net.bipartite_labels is not None else set())
   edges = set(net.edge_attributes) | set(net.qualitative_edge_attributes)
   default = None
   for key, names in [("nodes", nodes), ("node_size", nodes), ("node_opacity", nodes), ("edges", edges), ("edge_width", edges), ("edge_opacity", edges), ("labels", nodes), ("label_size", nodes)]:
      a = (style.get(key) or {}).get("attribute")
      if a is not None and a not in names:
         default = default or default_style(net)
         style[key] = default[key]
   position = style.get("position") or {}
   style["position"] = {k: v if v in nodes else None for k, v in position.items()}
   chart = style.get("node_chart") or {}
   if chart.get("attributes"):
      chart = dict(chart, attributes = [a for a in chart["attributes"] if a in net.node_attributes])
   style["node_chart"] = chart if chart.get("attributes") or chart.get("attribute") in nodes else {}
   return style


def pick_grid(x, y, size):
   """
   Grid of cells of side ``size`` (the largest node diameter; doubled until there are at most
   :data:`MAX_CELLS`), with which the interface finds the node under the cursor by looking at the
   9 cells around it.

   :return: Dictionary with ``x0``, ``y0``, ``cell``, ``gx``, ``gy``; the node positions sorted by
      cell, and the start of every cell in them (followed by the number of nodes), as int32 arrays.
   """
   if len(x) == 0:
      return {"x0": 0., "y0": 0., "cell": 1., "gx": 1, "gy": 1}, np.zeros(0, np.int32), np.zeros(2, np.int32)
   cell = max(size, 1e-9)
   x0, y0 = float(x.min()), float(y.min())
   while True:
      gx, gy = int((x.max() - x0) / cell) + 1, int((y.max() - y0) / cell) + 1
      if gx * gy <= MAX_CELLS:
         break
      cell *= 2
   cid = ((y - y0) / cell).astype(np.int64) * gx + ((x - x0) / cell).astype(np.int64)
   order = np.argsort(cid, kind = "stable")
   starts = np.searchsorted(cid[order], np.arange(gx * gy + 1)).astype(np.int32)
   return {"x0": x0, "y0": y0, "cell": cell, "gx": gx, "gy": gy}, order.astype(np.int32), starts


def drawing(net, size = None, style = None, attribute = "pos", bends = None, matrix = None):
   """
   Drawing of a laid out network, as buffers of Godot ``MultiMesh`` resources: one instance per
   node (a disc, its diameter in the transform and in the custom data) and one per edge (a quad
   from the source to the target, whose second axis is the relative width: the edge shader of the
   interface multiplies it by the default width), or, for bundled edges, one per cubic Bézier
   piece of their curves (:data:`CURVE_STRIDE`: the curve shader of the interface computes the
   curve, with :func:`subdivisions` segments; the last float of a piece is the position of its
   edge, for picking). Each instance is a 2D transform, row by row
   (``x.x, y.x, 0, origin.x, x.y, y.y, 0, origin.y``), and a color (:func:`colors`, its alpha
   the opacity, :func:`opacities`); nodes also have their diameter as custom data. The y axis is
   flipped, since it points down on screen.

   With ``matrix`` (the node attribute written by :func:`~graphoskop.matrix_layout`), the network
   is drawn as its adjacency matrix: every edge is a unit square (a cell, in the edge colors and
   opacities) at the column of its target and the row of its source (both cells of undirected
   edges in square matrices; in bipartite networks, the row of the node of the first type and the
   column of the other), and the nodes are drawn next to their row, on the left (nodes of the
   second type of bipartite networks above their column), keeping their colors, sizes, tooltips
   and selection.

   Self loops (not in matrices, where they are cells of the diagonal) also get a buffer of their
   own (``loops``): one instance per loop, as a node (:data:`NODE_STRIDE` floats) whose transform
   holds the position of its node scaled by the diameter of the loop (:data:`LOOP_SIZE` times the
   diameter of the node), with the edge color, and as custom data the diameter of the loop and of
   the node, the relative width of the edge and the position of the node; the loop shader of the
   interface draws it as a circle above the node.

   In directed networks (not in matrices), the edges between the same two nodes share one line
   (:func:`shown_edges`: the others get no width), and every direction has an arrowhead, in the
   color of the line, at the border of the node it points to: buffer ``arrows`` (:func:`arrows`,
   drawn by the arrow shader of the interface). Straight lines then have
   :data:`ARROW_EDGE_STRIDE` floats, with the diameters of their nodes and where their
   arrowheads are, so that the edge shader ends them at the base of the arrowheads.

   :param net: A :class:`~graphoskop.Network` with the numeric node attributes
      ``<attribute>_x`` and ``<attribute>_y``.
   :param size: Default diameter of the nodes, or ``None`` for :func:`node_size`.
   :param style: Style (see :func:`default_style`), or ``None`` for the default one.
   :param attribute: Prefix of the position attributes.
   :param bends: Curves of the edges, as returned by :func:`~graphoskop.bundle_edges`, or
      ``None`` for straight edges.
   :param matrix: Name of the node attribute of a matrix layout, or ``None`` for a node-link
      drawing.
   :return: Dictionary with the node, edge, loop and arrow buffers (float32 arrays), whether edges are curves
      (``curved``) or matrix cells (``cells``), the node ids joined by ``"\\x1f"`` (UTF-8), the picking grid
      (:func:`pick_grid`), the bounding box ``[x0, y0, x1, y1]``, the default diameter of the
      nodes and the largest one.
   """
   import torch
   if matrix is not None:
      place = torch.cat(net.get_node_attribute(matrix, None)).double().cpu().numpy()
      sizes_ = np.diff(net.node_offsets)
      local = np.arange(len(place)) - np.repeat(net.node_offsets[:-1], sizes_)
      column = local >= np.repeat(net.bipartite_split, sizes_) if net.bipartite_split is not None else np.zeros(len(place), bool)
      x, y = np.where(column, place, -1.), np.where(column, -1., place)   # rows on the left, columns on top
   else:
      style = style or default_style(net)
      x, y = (positions(net, style, axis, attribute) for axis in "xy")
      y = -y
   style = style or default_style(net)
   n = len(x)
   size = node_size(x, y) if size is None else size
   ns, nz = style.get("nodes"), style.get("node_size") or {}
   if nz.get("attribute") is None:
      d = np.full(n, size * float(nz.get("size", 1.)))
   else:
      d = size * node_sizes(net, nz["attribute"], float(nz.get("min", 0.5)), float(nz.get("max", 2.)), bool(nz.get("log", False)))
   N = np.zeros((n, NODE_STRIDE), dtype = np.float32)
   N[:, 0], N[:, 5], N[:, 3], N[:, 7], N[:, 12] = d, d, x, y, d
   N[:, 8:12] = colors(node_values(net, ns["attribute"]) if ns and ns.get("attribute") else n, ns, NODE_COLOR)
   no = style.get("node_opacity")
   N[:, 11] *= opacities(node_values(net, no["attribute"]) if no and no.get("attribute") else n, no)
   s, t = _edges(net)
   m = len(s)
   es, ew = style.get("edges"), style.get("edge_width")
   w = sizes(edge_values(net, ew["attribute"]) if ew and ew.get("attribute") else m, ew)
   c = colors(edge_values(net, es["attribute"]) if es and es.get("attribute") else m, es, EDGE_COLOR)
   eo = style.get("edge_opacity")
   c[:, 3] *= opacities(edge_values(net, eo["attribute"]) if eo and eo.get("attribute") else m, eo)
   curved = matrix is None and bends is not None and m and bends["edge"].nunique() == m
   if matrix is not None:   # one unit square per cell
      if net.bipartite_split is not None:
         rows = np.where(column[s], t, s)
         cols = np.where(column[s], s, t)
         ce = c
      elif net.directed:
         rows, cols, ce = s, t, c
      else:
         rows, cols, ce = np.concatenate([s, t]), np.concatenate([t, s]), np.concatenate([c, c])
      E = np.zeros((len(rows), EDGE_STRIDE), dtype = np.float32)
      E[:, 0], E[:, 5] = 1., 1.
      E[:, 3], E[:, 7] = place[cols], place[rows]
      E[:, 8:] = ce
   elif curved:   # one instance per Bezier piece
      P = bends[["x", "y"]].to_numpy(dtype = float) * [1., -1.]
      e, at = bends["edge"].to_numpy(), bends["point"].to_numpy()
      first = np.flatnonzero((at % 3 == 0) & np.append(e[1:] == e[:-1], False))
      E = np.zeros((len(first), CURVE_STRIDE), dtype = np.float32)
      E[:, 3], E[:, 7] = P[first, 0], P[first, 1]                # origin: first point
      E[:, 0], E[:, 4] = P[first + 1, 0], P[first + 1, 1]        # x axis: second point
      E[:, 1], E[:, 5] = P[first + 2, 0], P[first + 2, 1]        # y axis: third point
      E[:, 12], E[:, 13], E[:, 14], E[:, 15] = P[first + 3, 0], P[first + 3, 1], w[e[first]], e[first]   # the edge of the piece, for picking
      E[:, 8:12] = c[e[first]]
   else:
      dx, dy = x[t] - x[s], y[t] - y[s]
      length = np.maximum(np.hypot(dx, dy), 1e-12)
      E = np.zeros((m, ARROW_EDGE_STRIDE if net.directed else EDGE_STRIDE), dtype = np.float32)
      E[:, 0], E[:, 4] = dx, dy                                  # quad's x axis: along the edge
      E[:, 1], E[:, 5] = -dy / length * w, dx / length * w       # y axis: across it, the relative width
      E[:, 3], E[:, 7] = (x[s] + x[t]) / 2, (y[s] + y[t]) / 2
      E[:, 8:12] = c
   A = np.zeros((0, NODE_STRIDE), dtype = np.float32)
   if matrix is None and net.directed and m:   # one line per pair of nodes, an arrowhead per direction
      shown, back = shown_edges(s, t, n)
      ahead, behind = np.flatnonzero(shown & (s != t)), np.flatnonzero(back)   # arrowheads at targets, at sources
      edges = np.concatenate([ahead, behind])
      tips = np.concatenate([t[ahead], s[behind]])
      if curved:   # the rows of a curve are consecutive: its first and last pieces
         E[~shown[e[first]], 14] = 0.   # pieces of hidden edges: no width
         head, tail = np.full(m, len(P)), np.full(m, -1)
         np.minimum.at(head, e[first], first)
         np.maximum.at(tail, e[first], first)
         v = np.concatenate([P[tail[ahead] + 3] - P[tail[ahead] + 2], P[head[behind]] - P[head[behind] + 1]])   # tangents at the ends, towards them
      else:
         E[~shown, 1] = E[~shown, 5] = 0.
         E[:, 12], E[:, 13], E[:, 14], E[:, 15] = d[s], d[t], back, shown & (s != t)   # lines end at their arrowheads
         others = np.concatenate([s[ahead], t[behind]])
         v = np.column_stack([x[tips] - x[others], y[tips] - y[others]])
      A = arrows(edges, tips, len(ahead), v, x, y, d, w, c)
   loops = np.flatnonzero(s == t) if matrix is None else np.zeros(0, dtype = np.int64)   # self loops: circles over their node
   L = np.zeros((len(loops), NODE_STRIDE), dtype = np.float32)
   if len(loops):
      node = s[loops]
      L[:, 0] = L[:, 5] = L[:, 12] = LOOP_SIZE * d[node]
      L[:, 3], L[:, 7], L[:, 13], L[:, 14], L[:, 15] = x[node], y[node], d[node], w[loops], node
      L[:, 8:12] = c[loops]
   biggest = float(d.max()) if n else size
   grid, order, starts = pick_grid(x, y, biggest)
   ids = "\x1f".join(str(i) for i in _ids(net)).encode("utf-8")
   box = [float(x.min()), float(y.min()), float(x.max()), float(y.max())] if n else [0., 0., 1., 1.]
   if matrix is not None and len(E):
      box = [min(box[0], float(E[:, 3].min()) - 0.5), min(box[1], float(E[:, 7].min()) - 0.5), max(box[2], float(E[:, 3].max()) + 0.5), max(box[3], float(E[:, 7].max()) + 0.5)]
   return {"nodes": N, "edges": E, "loops": L, "arrows": A, "curved": bool(curved), "cells": matrix is not None, "ids": ids, "grid": grid, "order": order, "starts": starts, "box": box, "size": size, "biggest": biggest}


#: Diameter of the circle of a self loop, relative to the diameter of its node.
LOOP_SIZE = 0.6


def shown_edges(s, t, n):
   """
   Which edges of a directed network are drawn: one line per pair of nodes, the first edge
   between them (reciprocal and parallel edges share it), with an arrowhead for every direction
   the pair has.

   :param s: Sources of the edges (positions of the nodes).
   :param t: Targets of the edges.
   :param n: Number of nodes.
   :return: Two boolean arrays, one entry per edge: whether the edge is drawn, and whether it is
      drawn and the opposite edge exists too (its line gets a second arrowhead, at its source).
   """
   s, t = s.astype(np.int64), t.astype(np.int64)
   key = np.minimum(s, t) * n + np.maximum(s, t)
   order = np.argsort(key, kind = "stable")
   k = key[order]
   shown = np.zeros(len(s), bool)
   shown[order[np.append(True, k[1:] != k[:-1])]] = True
   codes = np.sort(s * n + t)
   wanted = t * n + s
   at = np.minimum(np.searchsorted(codes, wanted), max(len(codes) - 1, 0))
   return shown, shown & (s != t) & (codes[at] == wanted)


def arrows(edges, tips, ahead, v, x, y, d, w, c):
   """
   Buffer of the arrowheads of a directed drawing, one instance each (:data:`NODE_STRIDE` floats),
   drawn by the arrow shader of the interface: the transform has the position of the node the
   arrow points to as origin and the direction of the arrow (unit length) as x axis; the color is
   the color of the line; the custom data are the diameter of the node (the arrow stops at its
   border), the relative width of the edge (the size of the arrow follows it), the position of the
   edge and whether the arrow is at its target (1) or at its source (0).

   :param edges: Positions of the edges with an arrowhead.
   :param tips: For each, the node it points to.
   :param ahead: How many of them come first with the arrowhead at their target (the others at
      their source).
   :param v: Direction of each arrowhead (x, y rows, any length).
   :param x: x coordinates of the nodes.
   :param y: y coordinates of the nodes (down).
   :param d: Diameters of the nodes.
   :param w: Relative widths of the edges.
   :param c: Colors of the edges (RGBA rows).
   :return: float32 array, one row per arrowhead.
   """
   A = np.zeros((len(edges), NODE_STRIDE), dtype = np.float32)
   v = v.astype(np.float32)
   length = np.maximum(np.hypot(v[:, 0], v[:, 1]), np.float32(1e-30))
   A[:, 0] = A[:, 5] = v[:, 0] / length
   A[:, 4] = v[:, 1] / length
   A[:, 1] = -A[:, 4]
   A[:, 3], A[:, 7], A[:, 12] = x[tips], y[tips], d[tips]
   A[:, 8:12] = c[edges]
   A[:, 13], A[:, 14], A[:, 15] = w[edges] if np.ndim(w) else w, edges, 0.
   A[:ahead, 15] = 1.
   return A


def subdivisions(pieces):
   """
   Straight segments drawn per Bézier piece of bundled edges: 16, fewer when there are more than
   :data:`MAX_SEGMENTS` segments in all (at least 2).

   :param pieces: Number of Bézier pieces.
   """
   return int(np.clip(MAX_SEGMENTS // max(pieces, 1), 2, 16))


def preview(path, sep = "\t", header = True, rows = 10):
   """
   First rows of a table, read as :func:`~graphoskop.load_network` reads it (all values as text),
   and a guess of its node table format: long if it has three or more columns and a node id
   repeats in the first column among the first thousand rows.

   :param path: Path of the file.
   :param sep: Column separator.
   :param header: Whether the file has a header row.
   :param rows: Number of rows returned.
   :return: Dictionary with ``columns`` (names; positions as text without a header), ``rows``
      (lists of text) and ``long`` (the format guess).
   """
   import pandas as pd
   T = pd.read_csv(path, sep = sep, header = 0 if header else None, nrows = 1000, dtype = str, keep_default_na = False, engine = "python" if len(sep) > 1 else "c")
   long = T.shape[1] >= 3 and T.iloc[:, 0].duplicated().any()
   return {"columns": [str(c) for c in T.columns], "rows": T.head(rows).values.tolist(), "long": bool(long)}


def attributes(net):
   """
   Node and edge attributes of a network, for the visual elements panel: for each, its ``name``,
   whether it is ``numeric``, its smallest and largest number (``min``, ``max``: values, or
   positions of the values of qualitative attributes, see :func:`numbers`), whether all its numbers
   are positive (``positive``), its number of distinct values (``count``) and the first
   :data:`MAX_VALUES` of them in order (``values``, texts, see :func:`distinct`). Numeric
   attributes with more than :data:`MAX_VALUES` distinct values are not turned into texts (on
   100,000 nodes, a quarter of a second per attribute): ``count`` is then their number of distinct
   numbers. Node attributes are read all at once (:func:`node_matrix`, and the
   long table of the qualitative ones once): ten thousand attributes of a long node table take a fraction of
   a second, not seconds each.

   :return: Dictionary with lists ``nodes`` and ``edges``.
   """
   def numeric(name, u):   # u: the sorted distinct numbers
      texts = list(dict.fromkeys(text(x) for x in u[:MAX_VALUES]))
      lo, hi = (float(u[0]), float(u[-1])) if len(u) else (0., 1.)
      return {"name": name, "numeric": True, "min": lo, "max": hi, "positive": lo > 0, "count": len(u) if len(u) > MAX_VALUES else len(texts), "values": texts}
   def qualitative(name, v):   # v: the distinct values
      texts = distinct(np.asarray(v, dtype = object))[0]
      lo, hi = (0., float(len(texts) - 1)) if texts else (0., 1.)
      return {"name": name, "numeric": False, "min": lo, "max": hi, "positive": False, "count": len(texts), "values": texts[:MAX_VALUES]}
   def edge(name, v):
      return numeric(name, np.unique(v[~np.isnan(v)])) if v.dtype.kind == "f" else qualitative(name, _values_once(v))
   nodes = [numeric(a, u) for a, u in zip(net.node_attributes, _distinct_numbers(net))]
   nodes += [qualitative(a, v) for a, v in _qualitative_uniques(net).items()]
   edges = [edge(a, edge_values(net, a)) for a in list(net.edge_attributes) + list(net.qualitative_edge_attributes)]
   return {"nodes": nodes, "edges": edges}


def _drawn(ns):
   """
   The network drawn in a tab: the incidence network of hypergraphs and complexes (made here if no
   layout made it yet: it is not stored, so the code is unchanged), the network otherwise.
   """
   if "inc" in ns:
      return ns["inc"]
   net = ns["net"]
   return gsk.incidence_network(net) if net.higher_order is not None else net


def shown_attributes(names, hide):
   """
   The attributes ``names`` shown in a table of the data panel: with ``hide`` (from the
   interface: ``all``, whether all are hidden, and ``except``, the names that are not as the
   others), those not hidden; all without.
   """
   if not hide:
      return list(names)
   flip = set(hide.get("except") or [])
   return [a for a in names if (a in flip) == bool(hide.get("all"))]


def _tables(ns, kind = "nodes", hide = None):
   """
   Tables of the data panel of a tab, each made when first asked for (``kind``) and cached until
   the code of the tab runs again: one row per node of the drawn network (``nodes``: in the
   drawing order, with its id and attributes), and one per edge (``edges``: source and target, as
   positions of nodes, then the attributes; with the positions ``s`` and ``t`` and the node
   ``ids``, which name the sources and targets of the rows shown only, see :func:`_table`). Only
   the first :data:`TABLE_ATTRIBUTES` attributes are columns (a table with a column per attribute
   of ten thousand would stall the interface); ``attributes_nodes`` and ``attributes_edges`` count
   all of them. Attributes hidden in the data panel (``hide``, see :func:`shown_attributes`) are
   left out; a table is made again when they change.
   """
   import pandas as pd
   T = ns.get("__tables__")
   if T is None:
      T = ns["__tables__"] = {}
   key = (bool((hide or {}).get("all")), tuple(sorted((hide or {}).get("except") or [])))
   if kind not in T or hide is not None and T.get(kind + "_hide") != key:
      T[kind + "_hide"] = key
      net = _drawn(ns)
      if kind == "nodes":
         names = list(net.node_attributes) + list(net.qualitative_node_attributes)
         shown = shown_attributes(names, hide)[:TABLE_ATTRIBUTES]
         numeric = [a for a in shown if a in net.node_attributes]
         columns = _numeric_columns(net, numeric)
         T["nodes"] = pd.DataFrame({"node": _ids(net), **{a: columns[a] if a in columns else _column(net, a) for a in shown}})   # one table at once
         T["attributes_nodes"] = len(names)
      else:
         s, t = _edges(net)
         names = list(net.edge_attributes) + list(net.qualitative_edge_attributes)
         T.update(edges = pd.DataFrame({"source": s, "target": t, **{a: edge_values(net, a) for a in shown_attributes(names, hide)[:TABLE_ATTRIBUTES]}}), s = s, t = t, ids = _ids(net), attributes_edges = len(names))
   return T


def summary(net):
   """
   Properties of a network for the "Network" tab of the data panel, as pairs of texts, followed
   by its network attributes (``net.network_attributes``: global values of analyses, e.g.
   ``flow_hierarchy``).
   """
   kind = {"hypergraph": "hypergraph", "simplicial": "simplicial complex"}.get(net.higher_order, "network")
   rows = [["kind", kind], ["directed", str(net.directed)], ["bipartite", str(net.bipartite_labels is not None and net.higher_order is None)]]
   if net.layer_labels is not None:
      rows.append(["layers", ", ".join(text(v) for v in net.layer_labels)])
   rows += [["nodes", str(net.number_of_nodes())], ["edges", str(net.number_of_edges())], ["components", str(net.number_of_components())]]
   if net.higher_order == "hypergraph":
      rows.append(["hyperedges", str(net.number_of_hyperedges())])
   elif net.higher_order == "simplicial":
      rows.append(["facets", str(net.number_of_facets())])
   rows.append(["device", str(net.components[0].edge_index.device) if net.components else "cpu"])
   def listed(names):   # the first SUMMARY_ATTRIBUTES names, and how many there are in all
      names = list(names)
      return ", ".join(names[:SUMMARY_ATTRIBUTES]) + (f", … ({len(names):,} in all)" if len(names) > SUMMARY_ATTRIBUTES else "")
   rows += [["numeric node attributes", listed(net.node_attributes)], ["qualitative node attributes", listed(net.qualitative_node_attributes)]]
   rows += [["numeric edge attributes", listed(net.edge_attributes)], ["qualitative edge attributes", listed(net.qualitative_edge_attributes)]]
   rows += [[str(k), text(v) if np.ndim(v) == 0 else str(v)] for k, v in net.network_attributes.items()]   # network attributes, after the properties
   return rows


def _table(request):
   """
   Command ``table``: a page of a table of the data panel. ``kind`` is ``"nodes"`` (all nodes, the
   attributes of node ``node``, a position in the drawing order, as pairs attribute and value, or
   the rows of the nodes ``nodes``, a list of positions), ``"edges"`` (all edges, those of node
   ``node`` or of the nodes ``nodes``, or the edges ``edges``, positions in the order of the
   drawing), ``"network"`` (:func:`summary`) or ``"tables"`` (the table ``name`` of the
   simulations, :func:`_simulate`, or of ``tables``; its first :data:`TABLE_ATTRIBUTES` columns);
   rows ``offset`` to ``offset + count`` (texts, ``""`` for missing values) of the table sorted
   as a whole by the column ``sort`` (if given; ``ascending`` or not, see :func:`sort_rows`), with
   the ``columns``, whether each is ``numeric`` (the interface aligns them to the right), the
   ``total`` number of rows and, for nodes and edges, the ``index`` of the rows shown (positions
   of the nodes or edges in the drawing order, for the values edited in the data panel,
   :func:`_set_value`) and the number of ``attributes`` (more than the columns beyond
   :data:`TABLE_ATTRIBUTES`; the table of one node has them all, :func:`node_row`). Attributes
   hidden in the data panel (``hide``, see :func:`shown_attributes`) are left out (also of the
   table of one node), and ``hidden`` counts them.
   """
   import pandas as pd
   ns = _spaces[request["tab"]]
   kind, node = request["kind"], request.get("node")
   offset, count = int(request.get("offset", 0)), int(request.get("count", 200))
   if kind == "network":
      df = pd.DataFrame(summary(ns["net"]), columns = ["property", "value"])
   elif kind == "tables":
      df = tab_tables(ns).get(request.get("name"))
      if df is None:
         return {"columns": [], "numeric": [], "rows": [], "total": 0}, b""
      df = df.iloc[:, :TABLE_ATTRIBUTES]   # wide tables (e.g. the shortest path lengths): their first columns
      df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
   elif kind == "nodes" and node is not None:   # all the attributes, also those that are not columns of the table (but the hidden ones)
      net = _drawn(ns)
      row = node_row(net, int(node))
      values = {"node": _ids(net)[int(node)], **{a: row[a] for a in shown_attributes(list(row), request.get("hide"))}}
      df = pd.DataFrame({"attribute": [str(c) for c in values], "value": pd.Series([text(v) or "" for v in values.values()], dtype = object)})
   else:
      df = _rows(_tables(ns, kind, request.get("hide") or {}), request)
   ids = _tables(ns, "edges")["ids"] if kind == "edges" else None
   if ids is not None and request.get("sort") in ("source", "target"):   # sorted by the ids
      df = df.assign(**{request["sort"]: ids[df[request["sort"]].to_numpy()]})
   if request.get("sort") in df.columns:
      df = sort_rows(df, request["sort"], bool(request.get("ascending", True)))
   page = df.iloc[offset:offset + count]
   if ids is not None:   # the ids of the rows shown
      page = page.assign(**{c: ids[page[c].to_numpy()] for c in ("source", "target") if page[c].dtype.kind in "iu"})
   rows = [[text(v) or "" for v in r] for r in page.itertuples(index = False, name = None)]
   numeric = [(ids.dtype.kind in "fiu" if ids is not None and c in ("source", "target") else df[c].dtype.kind in "fiu") for c in df.columns]
   reply = {"columns": [str(c) for c in df.columns], "numeric": numeric, "rows": rows, "total": len(df)}
   if kind == "edges" or kind == "nodes" and node is None:   # attributes left out of the columns, and hidden
      reply["attributes"] = _tables(ns, kind)["attributes_" + kind]
   if kind in ("nodes", "edges") and request.get("hide"):
      net = _drawn(ns)
      names = list(net.node_attributes) + list(net.qualitative_node_attributes) if kind == "nodes" else list(net.edge_attributes) + list(net.qualitative_edge_attributes)
      reply["hidden"] = len(names) - len(shown_attributes(names, request["hide"]))
   if kind == "edges" or kind == "nodes" and node is None:   # positions of the rows in the drawing order, for edits
      reply["index"] = [int(i) for i in page.index]
   return reply, b""


def sort_rows(df, column, ascending = True):
   """
   A table of the data panel sorted as a whole by one of its columns (stable; missing values last
   either way): numbers numerically, texts alphabetically; columns mixing numbers and texts by
   their texts, numbers first.

   :param df: :class:`pandas.DataFrame`.
   :param column: Name of the column.
   :param ascending: Ascending (or descending) order.
   :return: The sorted table.
   """
   import pandas as pd
   col = df[column].reset_index(drop = True)
   if isinstance(col.dtype, pd.CategoricalDtype):
      col = col.astype(object)
   try:
      order = col.sort_values(ascending = ascending, na_position = "last", kind = "stable").index
   except TypeError:   # mixed types: by text, numbers first
      texts = pd.Series(distinct(col.to_numpy(dtype = object))[1], dtype = float).replace(-1., np.nan)   # positions in the order of distinct
      order = texts.sort_values(ascending = ascending, na_position = "last", kind = "stable").index
   return df.iloc[np.asarray(order)]


def _rows(T, request):
   """
   The rows of the node or edge table of the data panel shown for a selection (see
   :func:`_table`).
   """
   kind, node = request["kind"], request.get("node")
   df = T[kind]
   chosen = np.asarray(request.get("nodes") or [], dtype = np.int64)
   if kind == "edges" and node is not None:
      df = df.iloc[np.flatnonzero((T["s"] == node) | (T["t"] == node))]
   elif kind == "edges" and request.get("edges"):
      df = df.iloc[np.asarray(request["edges"], dtype = np.int64)]
   elif kind == "edges" and len(chosen):
      df = df.iloc[np.flatnonzero(np.isin(T["s"], chosen) | np.isin(T["t"], chosen))]
   elif kind == "nodes" and len(chosen):
      df = df.iloc[chosen]
   return df



def _run(tab, code, undoable = False, attribute = None):
   """
   Runs lines of code in the namespace of a tab and adds them to its code. Unless ``undoable``
   (moves, removals, additions and edited values, see :func:`_snapshot`; ``attribute``: the
   ``(kind, name)`` of the attribute an edit changes), the code cannot be undone, and neither can
   what came before it: the undo and redo lists of the tab are emptied.

   :return: The whole code of the tab.
   """
   ns = _spaces.setdefault(tab, dict(gsk = gsk, __code__ = ["import graphoskop as gsk", ""], **_helpers()))
   ns["__tables__"] = None
   code = _matching_note(ns, code)
   if undoable:
      before = _snapshot(ns, attribute)
   exec("\n".join(code), ns)
   ns["__code__"] += code
   if undoable:
      ns["__undo__"] = (ns.get("__undo__") or [])[-MAX_UNDO + 1:] + [before]
   else:
      ns["__undo__"] = []
   ns["__redo__"] = []
   return "\n".join(ns["__code__"])


#: Largest number of moves, removals and additions that can be undone in a tab.
MAX_UNDO = 100


def _snapshot(ns, attribute = None):
   """
   What undo and redo restore in the namespace of a tab: its code, its drawn network (removals and
   additions make a new network object, the old one is kept as it was), the positions of its
   nodes (moves change them in place), and the values of the ``attribute`` (``(kind, name)``, kind
   ``"nodes"`` or ``"edges"``) edited in the data panel, if any (edits change them in place).
   """
   t = _target(ns)
   net = ns[t]
   pos = {a: [x.clone() for x in net.get_node_attribute(a, None)] for a in ("pos_x", "pos_y") if a in net.node_attributes}
   snap = {"code": list(ns["__code__"]), "target": t, "net": net, "pos": pos}
   if attribute is not None:
      kind, name = attribute
      values = (net.get_node_attribute if kind == "nodes" else net.get_edge_attribute)(name, None)
      snap["attribute"] = (kind, name, [v.clone() if hasattr(v, "clone") else np.array(v, dtype = object) for v in values])
   return snap


def _restore(ns, snap):
   """
   Puts back a :func:`_snapshot` in the namespace of a tab.
   """
   ns["__code__"] = list(snap["code"])
   ns[snap["target"]] = snap["net"]
   for a, values in snap["pos"].items():
      snap["net"].set_node_attribute(a, [x.clone() for x in values], None)
   if "attribute" in snap:
      kind, name, values = snap["attribute"]
      (snap["net"].set_node_attribute if kind == "nodes" else snap["net"].set_edge_attribute)(name, [v.clone() if hasattr(v, "clone") else v.copy() for v in values], None)
   ns["__tables__"] = None


def _undo(request):
   """
   Command ``undo`` (or ``redo`` with ``redo = True``): goes back over the last move, removal or
   addition of nodes or edges of a tab (or forward again), with its code, and replies with the
   code, a ``summary`` and the drawing. Only the moves, removals and additions since the last
   other change of the network can be undone.
   """
   ns = _spaces[request["tab"]]
   back, ahead = ("__redo__", "__undo__") if request.get("redo") else ("__undo__", "__redo__")
   if not ns.get(back):
      raise ValueError("nothing to " + ("redo" if request.get("redo") else "undo"))
   snap = ns[back].pop()
   ns[ahead] = (ns.get(ahead) or []) + [_snapshot(ns, snap["attribute"][:2] if "attribute" in snap else None)]
   _restore(ns, snap)
   header, payload = _draw(ns, "\n".join(ns["__code__"]), True)
   drawn = _drawn(ns)
   header["summary"] = f"{drawn.number_of_nodes()} nodes, {drawn.number_of_edges()} edges"
   return header, payload


#: Largest number of elements (nodes plus edges) of a network that the interface draws without
#: asking: beyond, it offers to analyze the network without drawing it (:func:`too_large`).
MAX_ELEMENTS = 10_000_000


def too_large(net):
   """
   Whether a network has more than :data:`MAX_ELEMENTS` nodes and edges together, too many to
   draw.
   """
   return net.number_of_nodes() + net.number_of_edges() > MAX_ELEMENTS


def _new(request, code):
   """
   Runs the code creating the network of a new tab, and replies with the code, a summary, and
   whether it is :func:`too_large` to draw (with the limit, ``max_elements``): such a network is
   not drawn (see :func:`_drawing`) until the interface asks for it.
   """
   _spaces.pop(request["tab"], None)
   try:
      full = _run(request["tab"], code)
   except BaseException:   # also an abort
      _spaces.pop(request["tab"], None)
      raise
   net = _spaces[request["tab"]]["net"]
   if too_large(net):
      _spaces[request["tab"]]["__undrawn__"] = True
   return {"code": full, "summary": f"{net.number_of_nodes()} nodes, {net.number_of_edges()} edges", "too_large": too_large(net), "max_elements": MAX_ELEMENTS}, b""


def _drawing(request):
   """
   Command ``drawing``: whether the network of a tab is drawn (``on``). A network that is not
   drawn (e.g. one too large to draw, :func:`too_large`) can still be laid out, analyzed and
   edited: every reply has its code, without drawing (:func:`_draw`). Replies with the drawing
   (``nodes = 0`` if the network has no layout yet, or is not drawn).
   """
   ns = _spaces[request["tab"]]
   if request.get("on", True):
      ns.pop("__undrawn__", None)
   else:
      ns["__undrawn__"] = True
   return _draw(ns, "\n".join(ns["__code__"]))


def _load(request):
   """
   Command ``load``: runs the code of :func:`load_code` in a new namespace for the tab, and replies
   with the code and a summary of the network.
   """
   return _new(request, load_code(request["spec"]))


def _generate(request):
   """
   Command ``generate``: runs the code of :func:`generator_code` (``function``, ``arguments``) in
   a new namespace for the tab, and replies with the code and a summary of the network.
   """
   return _new(request, generator_code(request["function"], request.get("arguments") or {}))


def empty_code(directed = False):
   """
   Code of File › New network: an empty network (no nodes, no edges), to draw by hand with the
   menu of the visualization panel (add a node, add an edge), e.g. ``net =
   gsk.load_network(pd.DataFrame({"source": [], "target": []}), directed = True)``.

   :param directed: Whether its edges are directed.
   :return: List of lines of code, defining ``net``.
   """
   return ["import pandas as pd", "net = gsk.load_network(pd.DataFrame({\"source\": [], \"target\": []})" + (", directed = True" if directed else "") + ")"]


def _empty(request):
   """
   Command ``empty``: a new tab with the empty network of :func:`empty_code` (``directed``).
   """
   return _new(request, empty_code(bool(request.get("directed"))))


#: Most slices of a node chart (the node shader looks through them): the smallest ones beyond are
#: left out.
MAX_SLICES = 64
#: Width, in texels, of the texture of the slices of the node charts (one texel per slice: the end
#: of the slice as a fraction of the turn, and its color).
CHART_WIDTH = 4096
#: Colors of the slices of node charts without a palette (ColorBrewer Set1), cycled.
CHART_PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"]


def chart_slices(net, spec):
   """
   Slices of the node charts (Visual elements › Nodes › Chart): every node a pie chart over its
   disc, from the top, clockwise.

   * ``{"attributes": [names]}``: the values of numeric node attributes, a slice each, as large
     as its share of their sum (missing and negative values count 0; nodes whose sum is 0 have
     no chart); slice ``k`` has the ``k``-th color of the palette;
   * ``{"attribute": name}``: the values of a qualitative attribute, equal slices of the values of
     each node (several for nodes with many, e.g. overlapping communities; one, the whole disc,
     for the others; none for missing values); every distinct value (in the order of
     :func:`distinct`) has the next color of the palette.

   The ``palette`` (list of ``#rrggbb``, cycled) is :data:`CHART_PALETTE` unless given. A node has
   at most :data:`MAX_SLICES` slices, its largest.

   :param net: The drawn network.
   :param spec: The chart specification, or ``None`` / ``{}`` for no charts.
   :return: ``(offsets, counts, table)``: for every node in the drawing order, the position of its
      first slice in the table and its number of slices (0: no chart), and the float32 table of
      the slices, a row each: end of the slice (fraction of the turn), red, green, blue; padded
      to rows of :data:`CHART_WIDTH`.
   """
   n = int(net.number_of_nodes())
   spec = spec or {}
   palette = np.array([_rgba(c)[:3] for c in (spec.get("palette") or CHART_PALETTE)], dtype = np.float32)
   rows, cols, share = [], [], []   # node, color, size of every slice
   if spec.get("attributes"):
      for k, a in enumerate(spec["attributes"]):
         v = np.nan_to_num(numbers(node_values(net, a)), nan = 0.).astype(float)
         keep = np.flatnonzero(v > 0)
         rows.append(keep), cols.append(np.full(len(keep), k)), share.append(v[keep])
   elif spec.get("attribute") is not None:
      values = node_values(net, spec["attribute"])
      items = [[x for x in (v if isinstance(v, (tuple, list)) else [v]) if text(x) is not None] for v in values]
      order = {t: k for k, t in enumerate(distinct(np.array([text(x) for it in items for x in it], dtype = object))[0])}
      sizes = np.array([len(it) for it in items], dtype = int)
      rows.append(np.repeat(np.arange(n), sizes)), cols.append(np.array([order[text(x)] for it in items for x in it], dtype = int)), share.append(np.ones(int(sizes.sum())))
   offsets, counts = np.zeros(n, dtype = np.float32), np.zeros(n, dtype = np.float32)
   if not rows:
      return offsets, counts, np.zeros((0, 4), dtype = np.float32)
   r, c, w = np.concatenate(rows).astype(int), np.concatenate(cols).astype(int), np.concatenate(share).astype(float)
   o = np.lexsort((-w, r))   # by node, largest first
   r, c, w = r[o], c[o], w[o]
   rank = np.arange(len(r)) - np.searchsorted(r, r)
   keep = rank < MAX_SLICES
   r, c, w = r[keep], c[keep], w[keep]
   total = np.bincount(r, w, minlength = n)
   o = np.lexsort((c, r))   # by node, in the order of the colors
   r, c, w = r[o], c[o], w[o]
   start = np.searchsorted(r, np.arange(n))
   ends = (np.cumsum(w) - np.repeat(np.cumsum(total) - total, np.bincount(r, minlength = n))) / total[r]
   counts[:] = np.bincount(r, minlength = n)
   offsets[:] = np.where(counts > 0, start, 0)
   table = np.zeros((-(-len(r) // CHART_WIDTH) * CHART_WIDTH, 4), dtype = np.float32)
   table[:len(r), 0] = ends
   table[:len(r), 1:] = palette[c % len(palette)]
   return offsets, counts, table


def node_labels(net, style):
   """
   Labels of the nodes of a drawing (Visual elements › Nodes › Label): the texts (:func:`text`;
   empty for missing values) of the node attribute ``style["labels"]["attribute"]``, or the node
   ids with ``style["labels"]["ids"]``, joined by ``"\x1f"`` (UTF-8) in the drawing order, or
   empty bytes without labels (the default).
   """
   labels = style.get("labels") or {}
   a = labels.get("attribute")
   if a is None:
      return "\x1f".join(str(i) for i in _ids(net)).encode("utf-8") if labels.get("ids") and net.number_of_nodes() else b""
   v = _column(net, a)
   texts, pos = distinct(v)
   lookup = np.array(texts + [""], dtype = object)
   return "\x1f".join(lookup[pos]).encode("utf-8")


def label_sizes(net, style, size):
   """
   Heights of the node labels of a drawing, in drawing units (as the node diameters, so that they
   grow and shrink with the zoom): :data:`LABEL_FRACTION` of the default node diameter ``size``,
   times the relative sizes of ``style["label_size"]`` (Visual elements › Nodes › Label size, see
   :func:`sizes`: one size for all, or sizes from a node attribute), as float32 bytes in the
   drawing order; empty bytes without labels.
   """
   labels = style.get("labels") or {}
   if labels.get("attribute") is None and not labels.get("ids"):
      return b""
   spec = style.get("label_size") or {}
   values = node_values(net, spec["attribute"]) if spec.get("attribute") is not None else net.number_of_nodes()
   return (LABEL_FRACTION * size * sizes(values, spec)).astype(np.float32).tobytes()


def _draw(ns, code, keep = False):
   """
   Reply with the code of a tab and, if its network has positions, its drawing (:func:`drawing`):
   the header has the numbers of nodes, edges, self loops and arrowheads, the floats per edge
   (``stride``: :data:`EDGE_STRIDE`, :data:`ARROW_EDGE_STRIDE` or :data:`CURVE_STRIDE`), the byte sizes of the
   parts of the payload (node buffer, edge buffer, ids, grid order, grid starts, sources and
   targets of the edges as int32 positions of the nodes, loop buffer, arrow buffer, labels
   (:func:`node_labels`, empty without), label heights (:func:`label_sizes`), the slices of the
   node charts (:func:`chart_slices`: rows of :data:`CHART_WIDTH` texels, empty without; the
   nodes have the position of their first slice and their number in the custom data ``y`` and
   ``z`` of their buffer), in this order), the grid, the box, the
   default and largest node diameters and the style. The default node diameter of the tab is
   recomputed for the layout, unless ``keep`` (transformations of a layout and changes of style
   keep it). Edges are drawn as the curves of ``bends`` when no layout came after them, and the
   network as its adjacency matrix when the last layout is :func:`~graphoskop.matrix_layout`
   (:func:`last_layout`). A tab whose network is not drawn (``__undrawn__``, see :func:`_drawing`)
   gets only its code, ``nodes = 0`` and ``undrawn``.
   """
   if ns.get("__undrawn__"):
      return {"code": code, "nodes": 0, "undrawn": True, "undo": bool(ns.get("__undo__")), "redo": bool(ns.get("__redo__"))}, b""
   draw = ns.get("inc", ns["net"])
   last = last_layout(ns["__code__"])
   matrix = _matrix_attribute(ns, last) if last == "matrix_layout" else None
   if not draw.components and draw.higher_order is None:   # an empty network (File › New network): an empty canvas to add nodes to
      ns["__style__"] = prune_style(dict(default_style(draw), **ns.get("__style__", {})), draw)
      ns["__size__"] = ns.get("__size__") or EMPTY_NODE_SIZE
      b = EMPTY_BOX
      header = {"labels": False, "code": code, "nodes": 0, "empty": True, "edges": 0, "loops": 0, "arrows": 0, "stride": EDGE_STRIDE, "curved": False, "cells": False, "subdivisions": 1, "box": [-b, -b, b, b], "node_size": ns["__size__"], "biggest": ns["__size__"], "grid": {}, "style": ns["__style__"], "parts": [0] * 12}
      header.update(undo = bool(ns.get("__undo__")), redo = bool(ns.get("__redo__")), history = "", movable = True, adding = adding(ns["net"]))
      return header, b""
   if "pos_x" not in draw.node_attributes and matrix is None or draw.higher_order is not None:   # no layout yet (a new hypergraph, e.g. from clique promotion: laid out as its incidence network)
      return {"code": code, "nodes": 0}, b""
   ns["__style__"] = prune_style(dict(default_style(draw), **ns.get("__style__", {})), draw)
   bends = ns.get("bends") if last in CURVES and not _positioned(ns) else None   # the curves follow the layout
   d = drawing(draw, ns.get("__size__") if keep else None, ns["__style__"], bends = bends, matrix = matrix)
   ns["__size__"] = d["size"]
   s, t = _edges(draw)
   labels = node_labels(draw, ns["__style__"]) if matrix is None else b""
   heights = label_sizes(draw, ns["__style__"], d["size"]) if labels else b""
   offsets, counts, table = chart_slices(draw, ns["__style__"].get("node_chart")) if matrix is None else (None, None, np.zeros((0, 4), dtype = np.float32))
   if len(table):   # node charts: their slices, a texel each
      d["nodes"][:, 13], d["nodes"][:, 14] = offsets, counts
   parts = [d["nodes"].tobytes(), d["edges"].tobytes(), d["ids"], d["order"].tobytes(), d["starts"].tobytes(), s.astype(np.int32).tobytes(), t.astype(np.int32).tobytes(), d["loops"].tobytes(), d["arrows"].tobytes(), labels, heights, table.tobytes()]
   header = {"labels": bool(labels), "code": code, "nodes": len(d["nodes"]), "edges": len(d["edges"]), "loops": len(d["loops"]), "arrows": len(d["arrows"]), "stride": int(d["edges"].shape[1]), "curved": d["curved"], "cells": d["cells"], "subdivisions": subdivisions(len(d["edges"])), "box": d["box"], "node_size": d["size"], "biggest": d["biggest"], "grid": d["grid"], "style": ns["__style__"], "parts": [len(p) for p in parts]}
   header.update(undo = bool(ns.get("__undo__")), redo = bool(ns.get("__redo__")), history = _history_key(ns, draw), movable = matrix is None and not _positioned(ns), adding = adding(ns["net"]))
   return header, b"".join(parts)


def centrality_code(ns, function, arguments):
   """
   Code computing a centrality of :data:`CENTRALITIES` for all the nodes (or edges) of the drawn
   network of a tab, and storing it as an attribute (:meth:`~graphoskop.Network.set_node_attribute`
   or :meth:`~graphoskop.Network.set_edge_attribute`), e.g.
   ``net.set_node_attribute("betweenness", gsk.betweenness_centrality(net, component = None),
   None)``. The core-periphery models also keep their fit, ``rho`` (the correlation of the
   discrete and continuous models) or ``quality`` (Rombach), e.g. ``core_discrete, rho =
   gsk.discrete_core_periphery(net, component = None)`` (the whole network: nodes of different
   components are pairs without edges). With ``plot`` (degree only), also the fit of a power law to the degree distribution,
   ``fit = gsk.degree_power_law(net, ccdf = True)``.

   :param ns: Namespace of the tab.
   :param function: Name of the function.
   :param arguments: Parameter name to Python source code, with ``attribute`` (the name, by
      default the one of :data:`CENTRALITIES`) and ``plot`` (``"True"`` or ``"False"``).
   :return: List of lines of code, name of the attribute and whether to plot.
   :raises ValueError: If a value is not a Python expression, or the plot is asked for weighted
      degrees.
   """
   arguments = dict(arguments)
   name = ast.literal_eval(arguments.pop("attribute")) if "attribute" in arguments else CENTRALITIES[function][0]
   plot = arguments.pop("plot", "False") == "True"
   if plot and arguments.get("weight", "None") != "None":
      raise ValueError("the power-law fit needs integer degrees: leave weight empty to plot the degree distribution")
   for k, v in arguments.items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   higher = ns["net"].higher_order is not None
   kind = CENTRALITIES[function][1]
   if kind in HIGHER_KINDS and not higher:
      raise ValueError(f"{function} is for hypergraphs and simplicial complexes")
   lines = ["inc = gsk.incidence_network(net)"] if higher and "inc" not in ns else []
   target = "inc" if higher else "net"
   if kind in HIGHER_KINDS:   # computed on the network, written on the drawn incidence network
      call = f"gsk.{function}(net, component = None" + "".join(f", {k} = {v.strip()}" for k, v in arguments.items()) + (", distribution = None" if function in ("hyperedge_size", "generalized_degree") else "") + (", hyperedges = True" if kind == "node_edge" else "") + ")"
      if kind == "node_edge":
         lines += [f"scores = {call}   # per component: node scores, hyperedge scores", f"inc.set_node_attribute({literal(name)}, incidence_values(inc, [x for x, _ in scores], [y for _, y in scores]), None)"]
      else:
         part = "hyperedges" if kind == "hyperedges" else "nodes"
         lines.append(f"inc.set_node_attribute({literal(name)}, incidence_values(inc, {part} = {call}), None)   # values of the {'hyperedges' if part == 'hyperedges' else 'nodes'}, NaN for the {'nodes' if part == 'hyperedges' else 'hyperedges'}")
      return lines, name, False
   extra = "".join(f", {k} = {v.strip()}" for k, v in arguments.items()) + ("" if higher else paths_argument(ns, function, dict(arguments, component = "None")))
   call = f"gsk.degree({target}, distribution = None, component = None{extra})" if function == "degree" else f"gsk.{function}({target}, component = None{extra})"
   if kind == "hits":
      lines += [f"hubs, authorities = {call}", f"{target}.set_node_attribute({literal(name + '_hub')}, hubs, None)", f"{target}.set_node_attribute({literal(name + '_authority')}, authorities, None)"]
      name += "_hub"
   elif kind == "core":   # core values and the fit: rho, or the core quality of each (alpha, beta)
      values = name if name.isidentifier() else "core"
      lines += [f"{values}, {'quality' if function == 'rombach_core_periphery' else 'rho'} = {call}", f"{target}.set_node_attribute({literal(name)}, {values}, None)"]
   else:
      lines.append(f"{target}.set_{'edge' if kind == 'edges' else 'node'}_attribute({literal(name)}, {call}, None)")
   if plot:
      direction = f", direction = {arguments['direction'].strip()}" if "direction" in arguments else ""
      lines.append(f"fit = gsk.degree_power_law({target}{direction}, ccdf = True)")
   return lines, name, plot


def paths_argument(ns, function, arguments):
   """
   The precomputed shortest path lengths of a tab (Analysis › Shortest paths, ``paths``, see
   :func:`paths_code`) as an argument of a function that takes them (``lengths``): ``", lengths =
   paths"`` for all the components (``component = None``), ``", lengths = paths[c]"`` for component
   ``c``, when the network has not changed since and ``weight`` and ``inverse`` are the same;
   otherwise ``""`` (the function computes them). Sampled average path lengths do not use them.
   """
   meta = ns.get("__paths__")
   if not meta or meta.get("net") is not ns.get("net") or not isinstance(ns.get("paths"), list) or "lengths" not in inspect.signature(getattr(gsk, function)).parameters:
      return ""
   if arguments.get("samples", "None").strip() != "None" or arguments.get("weight", "None").strip() != meta["weight"] or arguments.get("inverse", "False").strip() != meta["inverse"]:
      return ""
   component = arguments.get("component", str(inspect.signature(getattr(gsk, function)).parameters["component"].default)).strip()
   if component == "None":
      return ", lengths = paths"
   try:
      c = int(component)
   except ValueError:
      return ""
   return f", lengths = paths[{c}]" if 0 <= c < len(ns["paths"]) else ""


#: Name of the table of the shortest path lengths of the largest component (Tables tab).
PATHS_TABLE = "Shortest path lengths"
#: Most rows (and columns) of a heatmap drawn: larger matrices are drawn with every k-th row and
#: column of their order (:func:`matrix_plot`), the data kept whole.
MATRIX_SHOWN = 2000
#: Largest matrix ordered in blocks by hierarchical clustering (:func:`block_order`, quadratic in
#: memory); larger ones have the matrix layout order.
BLOCK_ORDER_MAX = 5000


def paths_signature(net):
   """
   Parameters of Analysis › Shortest paths › All pairs lengths…: the edge lengths (``weight``,
   ``inverse``), the node attribute of the average distance of every node (empty for none) and
   whether to draw the heatmap.
   """
   params = [p for p in signature("all_pairs_shortest_path_lengths", net)["params"] if p["name"] in ("weight", "inverse")]
   params.append({"name": "average_distance", "default": literal("average_distance"), "hint": "none", "doc": "Node attribute written with the average length of the shortest paths from every node to the others of its component (the inverse of the closeness centrality); empty for none", "widget": {"kind": "string"}})
   params.append({"name": "heatmap", "default": "True", "hint": "", "doc": "Draw the lengths of the largest component as a heatmap (rows and columns can then be ordered, e.g. in blocks)", "widget": {"kind": "bool"}})
   return {"params": params}


def paths_code(ns, arguments):
   """
   Code of Analysis › Shortest paths › All pairs lengths…: the lengths of the shortest paths
   between all pairs of nodes of every component, kept in ``paths`` (a list of matrices, one per
   component) for the analyses that can use them (:func:`paths_argument`: betweenness, closeness,
   harmonic centrality, diameter, average shortest path length), those of the largest component as
   the table :data:`PATHS_TABLE` of the Tables tab (sharing the memory of the matrix), and,
   with ``average_distance``, the average length from every node to the others of its component
   as a node attribute (the inverse of :func:`~graphoskop.closeness_centrality`), e.g.

   .. code-block:: python

      paths = gsk.all_pairs_shortest_path_lengths(net, component = None)   # one matrix per component
      tables = {}
      tables["Shortest path lengths"] = pd.DataFrame(paths[0].cpu().numpy(), index = net.node_ids[0], columns = net.node_ids[0], copy = False)   # the largest component
      net.set_node_attribute("average_distance", [1 / c for c in gsk.closeness_centrality(net, component = None, lengths = paths)], None)

   :return: Lines of code.
   """
   arguments = {k: v.strip() for k, v in dict(arguments or {}).items() if str(v).strip()}
   for k, v in arguments.items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   attribute = ast.literal_eval(arguments.pop("average_distance", literal("average_distance")) or "\"\"")
   arguments.pop("heatmap", None)
   extra = "".join(f", {k} = {v}" for k, v in arguments.items())
   net = _drawn(ns)
   if net.higher_order is not None:
      raise ValueError("shortest paths of hypergraphs and simplicial complexes are not offered here (see the library)")
   lines = [f"paths = gsk.all_pairs_shortest_path_lengths(net, component = None{extra})   # one matrix per component, largest first"]
   if net.components:
      lines += _pandas(ns) + ([] if any(l.startswith("tables = ") for l in ns["__code__"]) and isinstance(ns.get("tables"), dict) else ["tables = {}"])
      lines.append(f"tables[{literal(PATHS_TABLE)}] = pd.DataFrame(paths[0].cpu().numpy(), index = net.node_ids[0], columns = net.node_ids[0], copy = False)   # the largest component")
   if attribute:
      lines.append(f"net.set_node_attribute({literal(attribute)}, [1 / c for c in gsk.closeness_centrality(net, component = None{extra}, lengths = paths)], None)   # average steps to the others of the component")
   return lines


def _paths(request):
   """
   Command ``paths``: runs :func:`paths_code` in a tab, remembers what the lengths are for
   (``__paths__``: the network, ``weight``, ``inverse``) and replies with the code, the drawing
   and a ``note`` (for the status bar);
   with ``heatmap``, also ``plot``, the heatmap of the lengths of the largest component
   (:func:`matrix_plot`, its PNG the last part of the payload), whose rows and columns can be
   ordered in the plot window (:func:`matrix_order`).
   """
   ns = _spaces[request["tab"]]
   arguments = request.get("arguments") or {}
   code = _run(request["tab"], paths_code(ns, arguments))
   ns["__paths__"] = {"net": ns["net"], "weight": arguments.get("weight", "None").strip() or "None", "inverse": arguments.get("inverse", "False").strip() or "False"}
   header, payload = _draw(ns, code, True)
   P = ns["paths"]
   n = P[0].shape[0] if P else 0
   header["note"] = f"shortest path lengths of {len(P)} components (largest: {n:,} nodes) in paths"
   if arguments.get("heatmap", "True").strip() == "True" and n:
      M = P[0].cpu().numpy()   # no copy
      ns["__plot__"] = _plot("matrix", {"matrix": M, "names": [str(text(i)) for i in _drawn(ns).node_ids[0]], "what": "shortest path length", "diverging": False, "orders": {}, "rows": "paths"}, dict(PLOT_ELEMENTS["matrix"][1]), {}, source = {"matrix": "paths[0].cpu().numpy()", "names": "[str(i) for i in net.node_ids[0]]", "what": literal("shortest path length"), "diverging": "False", "rows": literal("paths")})
      png = _render(ns["__plot__"], "png")
      shown = f" (one node in {-(-n // MATRIX_SHOWN)} drawn)" if n > MATRIX_SHOWN else ""
      header["plot"] = dict(_plot_header(ns["__plot__"], ns), text = f"Shortest path lengths of the largest component, {n:,} nodes{shown}.", warnings = [], title = f"Shortest path lengths of {request.get('name', 'the network')}")
      header["parts"] = header.get("parts", []) + [len(png)] if header.get("nodes") else [len(png)]
      return header, payload + png
   return header, payload


def _centrality(request):
   """
   Command ``centrality``: runs the code of :func:`centrality_code` in the namespace of a tab, and
   replies with the code and the drawing (unchanged in its look: the values become an attribute,
   which the user can choose for colors or sizes);
   with ``plot``, also with the plot of the degree distribution (:func:`degree_plot`): its PNG
   image as the last part of the payload, and ``plot`` in the header (:func:`_plot_header`, with
   the ``verdict``, ``text`` and ``warnings`` of the fit). The core-periphery models reply also a
   ``note`` with their fit (``rho``, or the best core quality).
   """
   ns = _spaces[request["tab"]]
   function, arguments = request["function"], dict(request.get("arguments") or {})
   if function not in CENTRALITIES:
      raise ValueError(f"unknown centrality: {function}")
   lines, name, plot = centrality_code(ns, function, arguments)
   if plot:
      import matplotlib   # before running anything: the plot needs it
   import warnings
   with warnings.catch_warnings(record = True) as caught:
      warnings.simplefilter("always")
      code = _run(request["tab"], lines)
   draw = _drawn(ns)
   header, payload = _draw(ns, code, True)
   if CENTRALITIES[function][1] == "core":   # the fit, for the status bar
      header["note"] = f"{function.replace('_core_periphery', ' core-periphery')}: " + (f"rho = {ns['rho']:.4g}" if function != "rombach_core_periphery" else f"best core quality R = {ns['quality']['R'].max():.4g}")
   if plot:
      fit = ns["fit"]
      degrees = node_values(draw, name)
      ns["__plot__"] = _plot("degree", {"degrees": degrees, "fit": fit}, dict(PLOT_ELEMENTS["degree"][1]), {"data": "points", "fit": "lines"}, source = {"degrees": f"node_values({_target(ns)}, {literal(name)})", "fit": "fit"})
      png = _render(ns["__plot__"], "png")
      notes = [str(w.message) for w in caught if any(t in str(w.message).lower() for t in ("power", "lognormal", "exponential")) and not str(w.message).startswith("the degree distribution is not a power law")]   # the verdict says it
      header["plot"] = dict(_plot_header(ns["__plot__"]), verdict = fit["verdict"], text = verdict_text(fit), warnings = notes)
      header["parts"].append(len(png))
      payload += png
   return header, payload


def verdict_text(fit):
   """
   One sentence on a fit of :func:`~graphoskop.degree_power_law`: whether the degree distribution
   is a power law, with its exponent and the range of degrees fitted.
   """
   if fit["verdict"] == "not a power law":
      return f"The degree distribution is not a power law: an exponential fits better (rate {fit['rate']:.3g}, degrees from {fit['k_min']})."
   cutoff = f", with an exponential cutoff (λ = {fit['cutoff']:.3g})" if fit["verdict"] == "power law with exponential cutoff" else ""
   return f"The degree distribution is a power law{cutoff}: exponent α = {fit['exponent']:.3g} for degrees from {fit['k_min']} ({100 * fit['tail_fraction']:.1f}% of the nodes)."


#: Fonts of the plots: the first one installed is used. Helvetica is the default font of gnuplot
#: (of its PostScript and PDF output); the others have the same metrics, with DejaVu Sans as the
#: last resort.
PLOT_FONTS = ["Helvetica", "Arial", "Nimbus Sans", "Nimbus Sans L", "Liberation Sans", "TeX Gyre Heros", "DejaVu Sans"]
#: Default options of the plots, which the user can change in the plot window: logarithmic axes,
#: font size (points), axis labels (``None``: the label of the plot; empty: automatic, for
#: distributions) and ranges (``None``: automatic); distributions also have their ``kind`` (see
#: :data:`DISTRIBUTIONS`), heatmaps the label, range and scale of their color scale (``zlabel``,
#: ``zmin``, ``zmax``, ``zlog``).
PLOT_OPTIONS = {"gnn": {"kind": "roc", "xlog": False, "ylog": False, "font_size": 14, "xlabel": "", "ylabel": "", "xmin": None, "xmax": None, "ymin": None, "ymax": None}, "links": {"kind": "roc", "xlog": False, "ylog": False, "font_size": 14, "xlabel": "", "ylabel": "", "xmin": None, "xmax": None, "ymin": None, "ymax": None}, "matrix": {"kind": "computed", "xlog": False, "ylog": False, "font_size": 14, "xlabel": "", "ylabel": "", "xmin": None, "xmax": None, "ymin": None, "ymax": None, "zlabel": "", "zlog": False, "zmin": None, "zmax": None}, "spectrum": {"xlog": False, "ylog": False, "font_size": 14, "xlabel": "frequency (eigenvalue of the Laplacian)", "ylabel": "", "xmin": None, "xmax": None, "ymin": None, "ymax": None}, "nestedness": {"xlog": False, "ylog": False, "font_size": 14, "xlabel": "", "ylabel": "", "xmin": None, "xmax": None, "ymin": None, "ymax": None, "zlabel": "", "zlog": False, "zmin": None, "zmax": None}, "randomization": {"kind": "right", "xlog": False, "ylog": False, "font_size": 14, "xlabel": "", "ylabel": "", "xmin": None, "xmax": None, "ymin": None, "ymax": None}, "distribution": {"kind": "count", "xlog": False, "ylog": False, "font_size": 14, "xlabel": "", "ylabel": "", "xmin": None, "xmax": None, "ymin": None, "ymax": None}, "degree": {"xlog": True, "ylog": True, "font_size": 14, "xlabel": "degree k", "ylabel": "fraction of nodes with degree ≥ k", "xmin": None, "xmax": None, "ymin": None, "ymax": None}, "epidemics": {"xlog": False, "ylog": False, "font_size": 14, "xlabel": "step", "ylabel": None, "xmin": None, "xmax": None, "ymin": None, "ymax": None}}
#: How a data series can be drawn: lines, points, lines with points, or bars.
PLOT_STYLES = ["lines", "points", "linespoints", "bars"]


#: Gradient of the data series of the plot being drawn (the option ``gradient`` of the plot, set by
#: ``_figure`` while it draws; ``None`` for one color per series).
_GRADIENT = None
#: Diverging palettes of ColorBrewer (centered on 0 when the values have both signs); the
#: sequential ones start at a quarter of their range, so that the smallest values stay visible on
#: white.
DIVERGING_PALETTES = {"BrBG", "PiYG", "PRGn", "PuOr", "RdBu", "RdGy", "RdYlBu", "RdYlGn", "Spectral"}


def gradient_colors(values, gradient, log = False):
   """
   Colors of values along a ColorBrewer gradient (the option ``gradient`` of the plots: ``axis``,
   ``"x"`` or ``"y"``, ``palette``, a sequential or diverging ColorBrewer palette, as named by
   matplotlib, and ``reversed``), from the smallest to the largest finite value (on a logarithmic
   scale with ``log``, for values above 0); diverging palettes are centered on 0 when the values
   have both signs.

   :return: RGBA rows, one per value (grey for missing values).
   """
   import matplotlib
   from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
   name = str(gradient.get("palette", "YlOrRd"))
   if name not in matplotlib.colormaps:
      raise ValueError(f"unknown palette: {name}")
   cmap = matplotlib.colormaps[name]
   if gradient.get("reversed"):
      cmap = cmap.reversed()
   v = np.asarray(values, dtype = float)
   ok = np.isfinite(v) & ((v > 0) if log else True)
   if not ok.any():
      return np.tile([0.6, 0.6, 0.6, 1.], (len(v), 1))
   lo, hi = float(v[ok].min()), float(v[ok].max())
   if hi <= lo:
      t = np.full(len(v), 0.5)
   elif log:
      t = LogNorm(lo, hi)(np.where(ok, v, lo))
   elif name in DIVERGING_PALETTES and lo < 0 < hi:
      t = TwoSlopeNorm(0., lo, hi)(v)
   else:
      t = Normalize(lo, hi)(v)
   t = np.clip(np.asarray(t, dtype = float), 0., 1.)
   if name not in DIVERGING_PALETTES:
      t = 0.25 + 0.75 * t
   out = cmap(t)
   out[~ok] = [0.6, 0.6, 0.6, 1.]
   return out


def _series(ax, x, y, style, color, label):
   """
   Draws a data series in one of the :data:`PLOT_STYLES`. Bars are 0.8 times the smallest gap
   between the x values wide (on a logarithmic x axis, as wide relatively). With a gradient
   (``_GRADIENT``), every bar, point and segment of line gets the color of its x or y value
   (:func:`gradient_colors`; a segment, the value at its middle).
   """
   x, y = np.asarray(x, dtype = float), np.asarray(y, dtype = float)
   g = _GRADIENT
   if g:
      along = x if g.get("axis") == "x" else y
      log = (ax.get_xscale() if g.get("axis") == "x" else ax.get_yscale()) == "log"
      color = gradient_colors(along, g, log)
   if style == "bars":
      gaps = np.diff(np.unique(x))
      if ax.get_xscale() == "log" and len(gaps) and (x > 0).all():
         r = np.diff(np.log(np.unique(x))).min()
         ax.bar(x * np.exp(-0.4 * r), y, width = x * (np.exp(0.4 * r) - np.exp(-0.4 * r)), align = "edge", color = color, alpha = 0.8, label = label)
      else:
         ax.bar(x, y, width = 0.8 * (gaps.min() if len(gaps) else 1.), color = color, alpha = 0.8, label = label)
   elif not g:
      ax.plot(x, y, {"lines": "-", "points": "o", "linespoints": "-o"}[style], color = color, lw = 2 if style == "lines" else 1.5, ms = 4, label = label)
   else:
      from matplotlib.collections import LineCollection
      if style in ("lines", "linespoints") and len(x) > 1:
         middle = (along[:-1] + along[1:]) / 2
         segments = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis = 1)
         ax.add_collection(LineCollection(segments, colors = gradient_colors(middle, g, log), linewidths = 2 if style == "lines" else 1.5, label = label))
         ax.update_datalim(np.column_stack([x, y])[np.isfinite(x) & np.isfinite(y)])
         ax.autoscale_view()
         label = None
      if style in ("points", "linespoints"):
         ax.scatter(x, y, c = color, s = 16, label = label, zorder = 3)


def _plot_font():
   """
   The first font of :data:`PLOT_FONTS` that matplotlib finds.
   """
   from matplotlib import font_manager
   have = {f.name for f in font_manager.fontManager.ttflist}
   return next((f for f in PLOT_FONTS if f in have), "DejaVu Sans")


def _figure(draw, options, format, path, dpi):
   """
   A plot drawn with matplotlib (needed for plots only), 8 by 5.5 inches, without a title:
   the ``options`` (see :data:`PLOT_OPTIONS`) set the scales of the axes, ``draw(ax)`` draws the
   data, then the options set the labels and ranges of the axes and the size of the font
   (:func:`_plot_font`); tick marks are inside the axes, on all four sides, and a legend is added
   (if a series has a label). The option ``gradient`` colors the data series by their x or y
   values (:func:`gradient_colors`, see ``_series``).

   :param format: ``"png"`` or ``"pdf"``.
   :param path: File to write, or ``None`` to return the bytes.
   :param dpi: Resolution of PNG images.
   :return: The bytes of the image, or ``None`` with ``path``.
   """
   import io
   import matplotlib
   matplotlib.use("Agg")
   from matplotlib.figure import Figure
   global _GRADIENT
   with matplotlib.rc_context({"font.family": _plot_font(), "font.size": float(options["font_size"]), "pdf.fonttype": 42}):
      fig = Figure(figsize = (8, 5.5))
      ax = fig.add_subplot()
      ax.set_xscale("log" if options["xlog"] else "linear")
      ax.set_yscale("log" if options["ylog"] else "linear")
      _GRADIENT = options.get("gradient") or None
      try:
         draw(ax)
      finally:
         _GRADIENT = None
      ax.set_xlabel(options["xlabel"])
      ax.set_ylabel(options["ylabel"])
      ax.set_xlim(options.get("xmin"), options.get("xmax"))
      ax.set_ylim(options.get("ymin"), options.get("ymax"))
      ax.tick_params(which = "both", direction = "in", top = True, right = True)   # tick marks inside, on all sides
      if ax.get_legend_handles_labels()[0]:
         ax.legend(frameon = False)
      fig.tight_layout()
      out = path if path is not None else io.BytesIO()
      fig.savefig(out, format = format, dpi = dpi)
   return None if path is not None else out.getvalue()


def degree_plot(degrees, fit, colors = None, options = None, styles = None, format = "png", path = None, dpi = 100):
   """
   Plot of the complementary cumulative degree distribution (the fraction of nodes with degree
   :math:`k` or more; log-log axes by default), with the best fit of
   :func:`~graphoskop.degree_power_law` from its smallest degree (see ``_figure``).

   :param degrees: Degrees of the nodes.
   :param fit: Result of :func:`~graphoskop.degree_power_law` with ``ccdf = True``.
   :param colors: Colors of the elements of the plot (``data``, ``fit``, ``k_min``, see
      :data:`PLOT_ELEMENTS`), or ``None`` for the defaults.
   :param options: Options of the plot (see :data:`PLOT_OPTIONS`), or ``None`` for the defaults.
   :param styles: How the data and the fit are drawn (:data:`PLOT_STYLES`; by default points and
      lines).
   :param format: ``"png"`` or ``"pdf"``.
   :param path: File to write, or ``None`` to return the bytes.
   :param dpi: Resolution of PNG images.
   :return: The bytes of the image, or ``None`` with ``path``.
   """
   colors = dict(PLOT_ELEMENTS["degree"][1], **(colors or {}))
   styles = dict({"data": "points", "fit": "lines"}, **(styles or {}))
   k = np.asarray(degrees, dtype = float)
   k = k[~np.isnan(k)]
   values, counts = np.unique(k[k > 0], return_counts = True)
   ccdf = (np.cumsum(counts[::-1])[::-1]) / max(len(k), 1)
   table = fit["ccdf"]
   label = {"power law": f"power law, α = {fit['exponent']:.3g}", "power law with exponential cutoff": f"power law with cutoff, α = {fit['exponent']:.3g}, λ = {fit['cutoff']:.3g}", "exponential": f"exponential, λ = {fit.get('rate', float('nan')):.3g}"}[fit["model"]]
   def draw(ax):
      _series(ax, values, ccdf, styles["data"], colors["data"], "data")
      _series(ax, table["degree"], table["fit"], styles["fit"], colors["fit"], f"{label} (k ≥ {fit['k_min']})")
      ax.axvline(fit["k_min"], color = colors["k_min"], ls = "--", lw = 1)
   return _figure(draw, dict(PLOT_OPTIONS["degree"], **(options or {})), format, path, dpi)


#: Plots of the plot window: their function and the colors of their elements, which the user can
#: change (key: default color), with the label of each in the window (plots with elements that
#: depend on the data, as the runs of :func:`epidemic_plot`, keep theirs with the plot).
PLOT_ELEMENTS = {"nestedness": (lambda **kw: nestedness_plot(**kw), {"matrix": "#377eb8", "isocline": "#e41a1c"}, {"matrix": "Matrix", "isocline": "Isocline"}), "randomization": (lambda **kw: randomization_plot(**kw), {"null": "#377eb8", "observed": "#e41a1c"}, {"null": "Null networks", "observed": "Observed value"}), "distribution": (lambda **kw: distribution_plot(**kw), {"data": "#377eb8"}, {"data": "Data"}), "degree": (degree_plot, {"data": "#377eb8", "fit": "#e41a1c", "k_min": "#999999"}, {"data": "Data", "fit": "Fit", "k_min": "Smallest degree of the fit"}), "epidemics": (lambda tables, states = ("I",), **kw: epidemic_plot(tables, states, **kw), {}, {}), "links": (lambda **kw: link_plot(**kw), {}, {}), "gnn": (lambda **kw: gnn_plot(**kw), {}, {}), "matrix": (lambda **kw: matrix_plot(**kw), {"matrix": "#377eb8"}, {"matrix": "Matrix"}), "spectrum": (lambda **kw: spectrum_plot(**kw), {"data": "#377eb8"}, {"data": "Coefficients"})}


def _render(plot, format, path = None, dpi = 100):
   """
   Draws a plot kept in a namespace (``kind``, ``data``, ``colors`` and ``options``) with its
   function of :data:`PLOT_ELEMENTS`.
   """
   return PLOT_ELEMENTS[plot["kind"]][0](**plot["data"], colors = plot["colors"], options = plot["options"], styles = plot["styles"], format = format, path = path, dpi = dpi)


def _plot(kind, data, colors, styles, labels = None, source = None, **options):
   """
   A plot to keep in a namespace (``ns["__plot__"]``): its ``kind`` (of :data:`PLOT_ELEMENTS`),
   the arguments of its function, the colors of its elements, how its data series are drawn
   (:data:`PLOT_STYLES`, by key), the labels of the elements (``None``: those of
   :data:`PLOT_ELEMENTS`), its options (:data:`PLOT_OPTIONS`, updated with ``options``) and the
   ``source`` of its data: for every argument, the Python expression giving it in the namespace of
   the tab (the code written when the plot is saved, :func:`plot_code`).
   """
   return {"kind": kind, "data": data, "colors": colors, "styles": styles, "labels": labels, "source": source or {}, "options": dict(PLOT_OPTIONS[kind], **options)}


def plot_figure(kind, data, colors = None, styles = None, options = None, labels = None, path = None, dpi = 200):
   """
   Draws a plot of the plot window of the interface and writes it to ``path`` (PNG at ``dpi``
   dots per inch, or PDF, by the extension): the code written when a plot is saved
   (:func:`plot_code`). A function of the interface, not of the library: it is defined in the
   namespace of every tab, and imported from the worker by the code (:data:`PLOT_NOTE`).

   :param kind: Kind of plot (:data:`PLOT_ELEMENTS`).
   :param data: Arguments of its plotting function (e.g. ``degrees`` and ``fit`` of
      :func:`degree_plot`).
   :param colors: Colors of its elements (key to ``#rrggbb``), ``None`` for the defaults.
   :param styles: How its data series are drawn (:data:`PLOT_STYLES`), ``None`` for the defaults.
   :param options: Options of the plot (:data:`PLOT_OPTIONS`), ``None`` for the defaults.
   :param labels: Labels of the elements, ``None`` for those of :data:`PLOT_ELEMENTS`.
   :param path: File to write.
   """
   plot = _plot(kind, data, dict(PLOT_ELEMENTS[kind][1], **(colors or {})), dict(styles or {}), labels)
   plot["options"].update(options or {})
   _render(plot, "pdf" if str(path).lower().endswith(".pdf") else "png", path, dpi)


def attribute_values(net, kind, name):
   """
   Values of a node (``kind`` ``"nodes"``) or edge attribute as the distribution plots take them
   (qualitative node values as categories). A function of the interface (see :func:`plot_figure`).
   """
   return _column(net, name) if kind == "nodes" else edge_values(net, name)


#: Comment written before the first code saving a plot (:func:`plot_code`).
PLOT_NOTE = "# plot_figure: plots of the graphical interface, as saved from the plot window"


def plot_code(ns, path):
   """
   Code saving the last plot of a tab, written in the code panel when the plot is saved from the
   plot window: the data of the plot from the variables of the tab (the ``source`` of the plot,
   :func:`_plot`), with its colors, styles and options, e.g.
   ``plot_figure("degree", {"degrees": node_values(net, "degree"), "fit": fit}, colors =
   {"data": "#377eb8", …}, styles = {…}, options = {…}, path = "/data/degree.png")``
   (:func:`plot_figure`); a table of a heatmap (``.tsv``) as a :class:`pandas.DataFrame`
   written with ``to_csv``.

   :return: Lines of code, or ``[]`` for plots whose data have no source.
   """
   plot = ns["__plot__"]
   source = dict(plot.get("source") or {})
   if not source:
      return []
   if plot["kind"] == "matrix":   # the order shown, computed again
      key = plot["options"].get("kind", "computed")
      source["orders"] = "{}" if key == "computed" else "{" + literal(key) + ": " + order_source(key, source["matrix"], bool(plot["data"].get("diverging")), len(plot["data"]["matrix"])) + "}"
   lines = [] if PLOT_NOTE in ns["__code__"] else [PLOT_NOTE]
   if any("np." in v for v in source.values()) and not any(l.startswith("import numpy as np") for l in ns["__code__"]):
      lines.append("import numpy as np")
   if str(path).lower().endswith(".tsv"):
      return lines + _pandas(ns) + [f"pd.DataFrame({source['matrix']}, index = {source['names']}, columns = {source['names']}).to_csv({literal(path)}, sep = \"\\t\")"]
   data = "{" + ", ".join(f"{literal(k)}: {v}" for k, v in source.items()) + "}"
   extra = "" if plot.get("labels") is None else f", labels = {plot['labels']!r}"
   options = {k: v for k, v in plot["options"].items() if v != PLOT_OPTIONS[plot["kind"]].get(k, None) or k == "kind"}
   return lines + [f"plot_figure({literal(plot['kind'])}, {data}, colors = {plot['colors']!r}, styles = {plot['styles']!r}, options = {options!r}{extra}, path = {literal(path)})"]


def order_source(key, matrix, diverging, k):
   """
   Expression of the order ``key`` of a heatmap (:func:`matrix_order`) in the code of a saved plot.
   """
   if key == "random":
      return f"np.random.default_rng(0).permutation({k}).tolist()"
   if key == "blocks":
      return f"block_order({matrix}, {diverging})"
   if key == "matrix":
      call = "gsk.matrix_layout(net, attribute = None)"
   else:
      a = key.split(":", 1)[1]
      call = f"gsk.matrix_layout(net, groups = {literal(a)}, attribute = None)"
   return f"np.argsort({call}.iloc[:{k}, 0].to_numpy(), kind = \"stable\").tolist()"


def _plot_header(plot, ns = None):
   """
   What the plot window needs to let the user change the plot: ``elements``, a list of [key,
   label, color, style] (style: one of :data:`PLOT_STYLES` for data series, ``""`` for other
   elements), ``options`` (see :data:`PLOT_OPTIONS`) and ``styles`` (:data:`PLOT_STYLES`).
   """
   labels = plot.get("labels") or PLOT_ELEMENTS[plot["kind"]][2]
   header = {"elements": [[k, labels[k], c, plot["styles"].get(k, "")] for k, c in plot["colors"].items()], "options": plot["options"], "styles": PLOT_STYLES}
   if plot["kind"] == "distribution":   # the kinds of distribution offered: [key, label, default style]
      header["kinds"] = [[k, v[0], v[1]] for k, v in DISTRIBUTIONS.items() if k in ("count", "pdf") or not plot["data"]["qualitative"]]
   elif plot["kind"] == "randomization":   # the tails of the pseudo p-value, and its text
      header.update(kinds = [[k, v, "bars"] for k, v in TAILS.items()], kind_label = "Pseudo p-value", text = randomization_text(plot))
   elif plot["kind"] == "gnn":   # the plots of the tasks of the runs, and their scores
      tasks = {plot["data"]["evaluations"][n]["task"] for n in plot["data"]["names"]}
      header.update(kinds = [[k, v[0], "points" if k == "scatter" else "lines"] for k, v in GNN_CURVES.items() if v[1] is None or v[1] in tasks], kind_label = "Plot", text = gnn_text(plot["data"]["evaluations"], plot["data"]["names"], plot["options"]["kind"]))
   elif plot["kind"] == "matrix":   # orders of the rows and columns
      header.update(kinds = [[k, label, "lines"] for k, label in matrix_orders(plot, ns)], kind_label = "Order")
   elif plot["kind"] == "links":   # ROC or precision-recall curves, and the areas under them
      header.update(kinds = [[k, v[0], "lines"] for k, v in LINK_CURVES.items()], kind_label = "Curve", text = link_text(plot["data"]["results"], plot["data"]["names"], plot["options"]["kind"]))
   return header


def _replot(request):
   """
   Command ``replot``: draws the last plot of a tab again with new ``colors`` (element key to
   ``#rrggbb``), ``styles`` (series key to one of :data:`PLOT_STYLES`) and ``options``
   (logarithmic axes ``xlog`` and ``ylog``, ``font_size`` from 4 to 72 points, axis labels
   ``xlabel`` and ``ylabel``, ranges ``xmin``, ``xmax``, ``ymin``, ``ymax``: numbers or ``None``;
   for heatmaps the label ``zlabel``, range ``zmin``, ``zmax`` and logarithmic scale ``zlog`` of
   the color scale; ``gradient``, the colors of the data series along x or y, see
   :func:`gradient_colors`, or ``None``),
   and replies with the PNG image as payload.
   """
   plot = _spaces[request["tab"]]["__plot__"]
   plot["colors"].update({k: v for k, v in (request.get("colors") or {}).items() if k in plot["colors"]})
   plot["styles"].update({k: v for k, v in (request.get("styles") or {}).items() if k in plot["styles"] and v in PLOT_STYLES})
   o = request.get("options") or {}
   plot["options"].update({k: (None if o[k] is None else float(o[k])) for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax") if k in o and k in plot["options"]})
   if "zlabel" in o and "zlabel" in plot["options"]:
      plot["options"]["zlabel"] = str(o["zlabel"])
   plot["options"].update({k: bool(o[k]) for k in ("xlog", "ylog", "zlog") if k in o and k in plot["options"]})
   plot["options"].update({k: str(o[k]) for k in ("xlabel", "ylabel") if k in o})
   if "font_size" in o:
      plot["options"]["font_size"] = min(max(float(o["font_size"]), 4.), 72.)
   if "gradient" in o:   # colors along x or y, or none
      g = o["gradient"]
      plot["options"]["gradient"] = None if not g else {"axis": "x" if g.get("axis") == "x" else "y", "palette": str(g.get("palette", "YlOrRd")), "reversed": bool(g.get("reversed", False))}
   ns = _spaces[request["tab"]]
   if plot["kind"] == "matrix" and o.get("kind") in dict(matrix_orders(plot, ns)):
      matrix_order(ns, plot, o["kind"])
      plot["options"]["kind"] = o["kind"]
   if plot["kind"] == "distribution" and o.get("kind") in DISTRIBUTIONS or plot["kind"] == "randomization" and o.get("kind") in TAILS or plot["kind"] == "links" and o.get("kind") in LINK_CURVES or plot["kind"] == "gnn" and o.get("kind") in GNN_CURVES:
      plot["options"]["kind"] = o["kind"]
   return _plot_header(plot, ns), _render(plot, "png")


#: Tails of the pseudo p-value of randomization tests (:func:`randomization_text`), with their
#: labels.
TAILS = {"right": "right-tailed", "left": "left-tailed", "two": "two-sided"}


def randomization_text(plot):
   """
   Text of the plot of a randomization test: the observed value, the mean and standard deviation
   of the null networks, and the pseudo p-value of the tail chosen (``options["kind"]``, see
   :data:`TAILS`): the share of null networks scoring at least the observed value (right-tailed),
   at most it (left-tailed), or twice the smaller of the two, at most 1 (two-sided, the
   ``p_value`` of :func:`~graphoskop.randomization_test`); a pseudo p-value of 0 is below one over
   the number of null networks, which the text says.
   """
   d = plot["data"]
   right, left = d["above"] + d["equal"], d["below"] + d["equal"]
   tail = plot["options"]["kind"]
   p = {"right": right, "left": left}.get(tail, min(1., 2 * min(right, left)))
   n = int(d["distribution"]["count"].sum())
   shown = f"{p:.4g}" if p > 0 else f"0 (below 1 / {n} = {1 / max(n, 1):.3g}: no null network is as extreme)"
   return f"{d['measure']}: observed {d['observed']:.4g}; {n} null networks: mean {d['mean']:.4g}, standard deviation {d['std']:.4g}. Pseudo p-value ({TAILS.get(tail, tail)}): {shown}."


def randomization_plot(distribution, observed, measure, mean = None, std = None, above = None, below = None, equal = None, colors = None, options = None, styles = None, format = "png", path = None, dpi = 100):
   """
   Plot of a randomization test (:func:`~graphoskop.randomization_test`): the number of null
   networks scoring each value of the measure (its rounded values, ``distribution``), and the
   ``observed`` value as a dashed vertical line. Empty axis labels are automatic (the measure, and
   the number of null networks; see ``_figure``).

   :param distribution: :class:`pandas.DataFrame` with columns ``value`` and ``count``.
   :param observed: Value of the measure on the network.
   :param measure: Label of the measure.
   :param colors: Colors of the ``null`` networks and the ``observed`` value, or ``None``.
   :param options: Options of the plot (see :data:`PLOT_OPTIONS`), or ``None`` for the defaults.
   :param styles: How the ``null`` networks are drawn (:data:`PLOT_STYLES`; bars by default).
   :return: The bytes of the image, or ``None`` with ``path``.
   """
   options = dict(PLOT_OPTIONS["randomization"], **(options or {}))
   options["xlabel"] = options["xlabel"] or measure
   options["ylabel"] = options["ylabel"] or "number of null networks"
   colors = dict(PLOT_ELEMENTS["randomization"][1], **(colors or {}))
   def draw(ax):
      _series(ax, distribution["value"], distribution["count"], (styles or {}).get("null", "bars"), colors["null"], "null networks")
      ax.axvline(observed, color = colors["observed"], ls = "--", lw = 2, label = "observed")
   return _figure(draw, options, format, path, dpi)


#: Largest number of rows and of columns of the matrix of the nestedness plot: larger biadjacency
#: matrices are compressed into this many bins of rows and of columns (:func:`nestedness_matrix`).
NESTEDNESS_CELLS = 400


def nestedness_code(ns, arguments):
   """
   Code of Analysis › Core-periphery › Nestedness for the drawn network of a tab (bipartite: the
   incidence network of a hypergraph or simplicial complex too): the NODF nestedness, with the
   null model of the ``arguments`` if any, and the isocline (``isocline``: its ``method``), e.g.
   ``nested = gsk.nestedness(net, null_model = "fixed")`` and ``isocline =
   gsk.nestedness_isocline(net, method = "fill")``.

   :param ns: Namespace of the tab.
   :param arguments: Parameter name to Python source code.
   :return: List of lines of code.
   :raises ValueError: If a value is not a Python expression.
   """
   arguments = dict(arguments)
   method = arguments.pop("isocline", "\"fill\"").strip()
   for k, v in list(arguments.items()) + [("isocline", method)]:
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   higher = ns["net"].higher_order is not None
   t = "inc" if higher else "net"
   extra = "".join(f", {k} = {v.strip()}" for k, v in arguments.items())
   return (["inc = gsk.incidence_network(net)"] if higher and "inc" not in ns else []) + [f"nested = gsk.nestedness({t}{extra})", f"isocline = gsk.nestedness_isocline({t}, method = {method})"]


def nestedness_matrix(net, rows, columns, cells = NESTEDNESS_CELLS):
   """
   The biadjacency matrix of a bipartite network, rows (nodes of type 0) and columns (type 1) in the
   given orders (those of :func:`~graphoskop.nestedness_isocline`: by decreasing degree),
   compressed into at most ``cells`` bins of consecutive rows and of consecutive columns: the share
   of ones in each block (parallel edges count once).

   :param net: Bipartite network.
   :param rows: Ids of the nodes of type 0, in order.
   :param columns: Ids of the nodes of type 1, in order.
   :param cells: Largest number of bins of each side.
   :return: NumPy array, bins of rows by bins of columns, values from 0 to 1.
   """
   import pandas as pd
   index = pd.Index(_ids(net))
   R, C = len(rows), len(columns)
   n = len(index)
   rank_r, rank_c = np.full(n, -1, dtype = np.int64), np.full(n, -1, dtype = np.int64)
   rank_r[index.get_indexer(list(rows))] = np.arange(R)
   rank_c[index.get_indexer(list(columns))] = np.arange(C)
   s, t = _edges(net)
   first = rank_r[s] >= 0
   r = np.where(first, rank_r[s], rank_r[t])
   c = np.where(first, rank_c[t], rank_c[s])
   ok = (r >= 0) & (c >= 0)
   key = np.unique(r[ok] * C + c[ok])
   r, c = key // max(C, 1), key % max(C, 1)
   br, bc = min(R, cells), min(C, cells)
   ir, ic = r * br // max(R, 1), c * bc // max(C, 1)
   counts = np.bincount(ir * bc + ic, minlength = br * bc).reshape(br, bc).astype(float)
   sr = np.bincount(np.arange(R) * br // max(R, 1), minlength = br)   # rows and columns per bin
   sc = np.bincount(np.arange(C) * bc // max(C, 1), minlength = bc)
   return counts / np.maximum(np.outer(sr, sc), 1)


def nestedness_text(nested, isocline, method):
   """
   Text under the nestedness plot: the NODF (and of rows and columns), the comparison with the null
   model if any, and the parameter of the isocline.
   """
   text_ = f"NODF {nested['nodf']:.4g} (rows {nested['nodf_rows']:.4g}, columns {nested['nodf_columns']:.4g})"
   if "null_mean" in nested:
      text_ += f"; null networks: mean {nested['null_mean']:.4g}, standard deviation {nested['null_std']:.4g}, z = {nested['z']:.3g}, p-value {nested['p_value']:.3g}"
   return text_ + f". Isocline ({method}): p = {isocline['p']:.3g}, fill {isocline['fill']:.3g}."


def nestedness_plot(matrix, line, rows, columns, p = None, colors = None, options = None, styles = None, format = "png", path = None, dpi = 100):
   """
   Plot of the nestedness of a bipartite network: its biadjacency matrix, rows and columns by
   decreasing degree, as a heatmap from white to the color of the ``matrix`` (the share of ones of
   every block of :func:`nestedness_matrix`, with a color bar), and the isocline of
   :func:`~graphoskop.nestedness_isocline` drawn on top (the ``line``, from the top left corner:
   the ones of a nested matrix are above and left of it). Axes are the fraction of the rows (top
   to bottom) and of the columns; empty axis labels are automatic (see ``_figure``).

   :param matrix: Compressed matrix (NumPy array).
   :param line: :class:`pandas.DataFrame` with columns ``x`` and ``y`` (fractions of the columns
      and of the rows, from the top).
   :param rows: Name of the node type of the rows.
   :param columns: Name of the node type of the columns.
   :param p: Parameter of the isocline, for its label.
   :param colors: Colors of the ``matrix`` and of the ``isocline``, or ``None``.
   :return: The bytes of the image, or ``None`` with ``path``.
   """
   from matplotlib.colors import LinearSegmentedColormap
   options = dict(PLOT_OPTIONS["nestedness"], **(options or {}))
   options.update(xlog = False, ylog = False)
   options["xlabel"] = options["xlabel"] or f"{columns} (columns), by decreasing degree"
   options["ylabel"] = options["ylabel"] or f"{rows} (rows), by decreasing degree"
   colors = dict(PLOT_ELEMENTS["nestedness"][1], **(colors or {}))
   def draw(ax):
      cmap = LinearSegmentedColormap.from_list("ones", ["#ffffff", colors["matrix"]])
      norm = _color_norm(0., max(float(matrix.max()), 1e-12) if matrix.size else 1., options, matrix)
      image = ax.imshow(matrix, extent = (0, 1, 1, 0), aspect = "auto", interpolation = "nearest", cmap = cmap, norm = norm)
      ax.figure.colorbar(image, ax = ax, label = options.get("zlabel") or "share of ones")
      label = "isocline" + (f" (p = {p:.3g})" if p is not None and np.isfinite(p) else "")
      ax.plot(line["x"], line["y"], color = colors["isocline"], lw = 2, label = label)
      ax.set_xlim(0, 1)
      ax.set_ylim(1, 0)
   return _figure(draw, options, format, path, dpi)


def _nestedness(request):
   """
   Command ``nestedness``: runs the code of :func:`nestedness_code` in the namespace of a tab, and
   replies with the code, the drawing (unchanged) and ``plot`` (:func:`_plot_header` of the
   nestedness plot, :func:`nestedness_plot`, with its ``text``, :func:`nestedness_text`, and
   ``title``); its PNG image is the last part of the payload.
   """
   ns = _spaces[request["tab"]]
   arguments = dict(request.get("arguments") or {})
   method = ast.literal_eval(arguments.get("isocline", "\"fill\""))
   code = _run(request["tab"], nestedness_code(ns, arguments))
   draw = _drawn(ns)
   iso, nested = ns["isocline"], ns["nested"]
   a, b = (str(text(v)) for v in draw.bipartite_labels)
   matrix = nestedness_matrix(draw, iso["rows"], iso["columns"])
   ns["__plot__"] = _plot("nestedness", {"matrix": matrix, "line": iso["line"], "rows": a, "columns": b, "p": iso["p"]}, dict(PLOT_ELEMENTS["nestedness"][1]), {}, source = {"matrix": f"nestedness_matrix({_target(ns)}, isocline[\"rows\"], isocline[\"columns\"])", "line": "isocline[\"line\"]", "rows": literal(a), "columns": literal(b), "p": "isocline[\"p\"]"})
   png = _render(ns["__plot__"], "png")
   header, payload = _draw(ns, code, True)
   header["plot"] = dict(_plot_header(ns["__plot__"]), text = nestedness_text(nested, iso, method), warnings = [], title = f"Nestedness of {request.get('name', 'the network')}")
   header["parts"] = header.get("parts", []) + [len(png)] if header["nodes"] else [len(png)]
   return header, payload + png


#: Node vector analyses of Analysis › Node vectors (:func:`vector_code`): what they take (one
#: node attribute, two, one with an optional second, or a set of them, ``"pairwise"``), the
#: variable of their result in the code and what the result is called.
VECTORS = {"annihilation_distance": ("two", "distance", "Annihilation distance"), "earth_movers_distance": ("two", "distance", "Earth mover's distance"), "generalized_euclidean": ("optional", "distance", "Generalized Euclidean distance"), "pairwise_generalized_euclidean": ("pairwise", "distances", "Generalized Euclidean distances"), "graph_fourier_transform": ("one", "coefficients", "Graph Fourier transform"), "network_correlation": ("two", "correlation", "Network correlation"), "pairwise_network_correlation": ("pairwise", "correlations", "Network correlations"), "network_entropy": ("one", "entropy", "Network entropy"), "network_variance": ("one", "variance", "Network variance")}

#: Parameters of the node vector analyses not in their dialog: cached matrices, the component (the
#: analyses run on the largest one), devices and types of the results, and tolerances.
VECTOR_HIDDEN = {"net", "component", "lpinv", "W", "R", "F", "basis", "output_device", "output_dtype", "flow", "tol"}

#: Parameters that only matter for directed networks (the Laplacian of directed networks), hidden
#: for undirected ones.
DIRECTED_ONLY = {"kind", "q", "alpha"}

OPTIONS.update({(f, "normalize"): ["True", "\"min\"", "\"max\"", "False"] for f in ("earth_movers_distance", "annihilation_distance")})
OPTIONS.update({(f, "kind"): ["\"chung\"", "\"magnetic\"", "\"out\""] for f in VECTORS if f != "graph_fourier_transform"})
OPTIONS[("graph_fourier_transform", "kind")] = ["\"chung\"", "\"magnetic\""]
RANGES.update({(f, "alpha"): (0., 1., 0.01) for f in VECTORS})
RANGES.update({(f, "q"): (0., 0.5, 0.01) for f in VECTORS})
NUMBERS.update({(f, "k"): (0., 0.1) for f in ("network_correlation", "pairwise_network_correlation", "network_entropy")})
HINTS.update({("generalized_euclidean", "b"): "none (the norm of a)", ("graph_fourier_transform", "k"): "all modes", ("pairwise_generalized_euclidean", "attributes"): "all numeric attributes", ("pairwise_network_correlation", "attributes"): "all numeric attributes"})


def vector_hidden(function, net):
   """
   Parameters of a node vector analysis left out of its dialog: :data:`VECTOR_HIDDEN`, and for
   undirected networks :data:`DIRECTED_ONLY` (but the teleportation ``alpha`` of the annihilation
   distance).
   """
   if net is not None and net.directed:
      return VECTOR_HIDDEN
   return VECTOR_HIDDEN | (DIRECTED_ONLY - ({"alpha"} if function == "annihilation_distance" else set()))


def vector_code(ns, function, arguments):
   """
   Code of a node vector analysis of :data:`VECTORS` on the network of a tab (its largest
   component, the default of the library), keeping its result in a variable, e.g.
   ``correlation = gsk.network_correlation(net, "age", "income", k = 2)``, also kept as a network
   attribute, e.g. ``net.network_attributes["network_correlation(age, income)"] = correlation``
   (not the pairwise matrices). The graph Fourier
   transform first computes the Fourier basis with the parameters of the transform
   (``basis = gsk.graph_fourier_basis(net, ...)``, then ``coefficients =
   gsk.graph_fourier_transform(net, "a", basis = basis)``), for the frequencies of the plot.

   :param ns: Namespace of the tab.
   :param function: Name of the function.
   :param arguments: Parameter name to Python source code: the attributes ``a`` and ``b`` (or
      ``attributes`` for the pairwise functions) and the other parameters.
   :return: List of lines of code.
   :raises ValueError: If a required attribute is missing or a value is not a Python expression.
   """
   arguments = dict(arguments)
   kind, result = VECTORS[function][:2]
   for k, v in arguments.items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   vectors = [arguments.pop(k, "").strip() for k in ("a", "b")]
   need = {"one": 1, "two": 2, "optional": 1, "pairwise": 0}[kind]
   for name, v in zip("ab", vectors[:need]):
      if not v:
         raise ValueError(f"{name}: choose a numeric node attribute")
   given = [v for v in vectors if v]
   extra = "".join(f", {k} = {v.strip()}" for k, v in arguments.items())
   if function == "graph_fourier_transform":
      return [f"basis = gsk.graph_fourier_basis(net{extra})", f"coefficients = gsk.graph_fourier_transform(net, {given[0]}, basis = basis)"]
   line = f"{result} = gsk.{function}(net{''.join(', ' + v for v in given)}{extra})"
   if kind == "pairwise":
      return [line]
   key = f"{function}({', '.join(str(ast.literal_eval(v)) for v in given)})"
   return [line, f"net.network_attributes[{literal(key)}] = {result}   # a network attribute"]


#: Largest number of attributes whose names are written along the axes of the matrix of a pairwise
#: analysis, and whose values are written in its cells (:func:`matrix_plot`).
MATRIX_NAMES = 40
MATRIX_VALUES = 12


def matrix_plot(matrix, names, what, diverging = False, orders = None, rows = None, colors = None, options = None, styles = None, format = "png", path = None, dpi = 100):
   """
   Plot of the result of a pairwise node vector analysis (:data:`VECTORS`): the matrix of the
   attributes as a heatmap with a color bar, from white to the color of the ``matrix`` (distances),
   or, with ``diverging``, from the color of ``negative`` values (-1) through white (0) to the
   color of ``positive`` ones (1) (correlations); missing values (NaN) are grey. The names of the
   attributes are written along the axes (up to :data:`MATRIX_NAMES` of them), and the values in
   the cells (up to :data:`MATRIX_VALUES`). The ``options`` ``zmin`` and ``zmax`` set the range
   of the colors (correlations: -1 to 1 by default, distances: 0 to the largest), ``zlog`` makes
   their scale logarithmic (see :func:`_color_norm`), ``zlabel`` is the label of the color bar.

   :param matrix: Square NumPy array.
   :param names: Names of the attributes, in the order of the rows.
   :param what: Label of the color bar.
   :param diverging: Whether the values are correlations.
   :param orders: Orders of the rows and columns, by key of the option ``kind`` (permutations,
      :func:`matrix_order`); the matrix as computed for other keys. Matrices with more than
      :data:`MATRIX_SHOWN` rows are drawn with every k-th row and column of the order.
   :param rows: What the rows are: ``"paths"`` for the nodes of the shortest path lengths, ``None``
      for attributes (it decides the orders offered, :func:`matrix_orders`).
   :return: The bytes of the image, or ``None`` with ``path``.
   """
   from matplotlib.colors import LinearSegmentedColormap
   options = dict(PLOT_OPTIONS["matrix"], **(options or {}))
   order = (orders or {}).get(options.get("kind"))
   order = np.arange(len(matrix)) if order is None else np.asarray(order)
   if len(order) > MATRIX_SHOWN:   # every k-th row and column of the order
      order = order[::-(-len(order) // MATRIX_SHOWN)]
   if order is not None:   # rows and columns in that order
      matrix = np.asarray(matrix[np.ix_(order, order)], dtype = float)
      names = [names[i] for i in order]
   options.update(xlog = False, ylog = False)
   colors = dict({"negative": "#377eb8", "positive": "#e41a1c"} if diverging else PLOT_ELEMENTS["matrix"][1], **(colors or {}))
   k = len(names)
   def draw(ax):
      if diverging:
         cmap, lo, hi = LinearSegmentedColormap.from_list("values", [colors["negative"], "#ffffff", colors["positive"]]), -1., 1.
      else:
         finite = matrix[np.isfinite(matrix)]
         cmap, lo, hi = LinearSegmentedColormap.from_list("values", ["#ffffff", colors["matrix"]]), 0., max(float(finite.max()) if finite.size else 1., 1e-12)
      norm = _color_norm(lo, hi, options, matrix)
      cmap.set_bad("#cccccc")
      image = ax.imshow(np.ma.masked_invalid(matrix), cmap = cmap, norm = norm, interpolation = "nearest", aspect = "equal")
      ax.figure.colorbar(image, ax = ax, label = options.get("zlabel") or what)
      if k <= MATRIX_NAMES:
         ax.set_xticks(range(k), names, rotation = 90)
         ax.set_yticks(range(k), names)
      else:
         ax.set_xticks([])
         ax.set_yticks([])
      if k <= MATRIX_VALUES:
         for i in range(k):
            for j in range(k):
               if np.isfinite(matrix[i, j]):
                  ax.text(j, i, f"{matrix[i, j]:.3g}", ha = "center", va = "center", fontsize = options["font_size"] * 0.7)
      ax.set_xlim(-0.5, k - 0.5)
      ax.set_ylim(k - 0.5, -0.5)
   return _figure(draw, options, format, path, dpi)


#: Most node attributes offered as orders of the heatmap of the shortest path lengths.
MAX_ORDER_ATTRIBUTES = 100


def matrix_orders(plot, ns = None):
   """
   Orders of the rows and columns of a heatmap (the Order menu of the plot window): as computed,
   random, block diagonal (clusters of similar rows next to each other, :func:`block_order`; up to
   :data:`BLOCK_ORDER_MAX` rows);
   for the shortest path lengths also the order of the matrix layout of the network (blocks of
   densely connected nodes) and by every node attribute (up to :data:`MAX_ORDER_ATTRIBUTES`).

   :return: List of ``(key, label)``.
   """
   found = [("computed", "as computed"), ("random", "random")] + ([("blocks", "block diagonal (clusters)")] if len(plot["data"]["matrix"]) <= BLOCK_ORDER_MAX else [])
   if plot["data"].get("rows") == "paths" and ns is not None:
      net = _drawn(ns)
      found.append(("matrix", "block diagonal (matrix layout)"))
      names = [a for a in list(net.node_attributes) + list(net.qualitative_node_attributes) if a not in ("pos_x", "pos_y")]
      found += [(f"attribute:{a}", f"by {a}") for a in names[:MAX_ORDER_ATTRIBUTES]]
   return found


def block_order(matrix, diverging = False):
   """
   Order of the rows (and columns) of a square matrix that puts similar rows next to each other,
   so that groups become blocks on the diagonal: the leaves of an average-linkage hierarchical
   clustering of the rows, with the optimal leaf ordering (Bar-Joseph et al., 2001) up to 2,000
   rows. Distances are the values themselves (made symmetric; missing and infinite ones count as
   twice the largest), or, for correlations (``diverging``), one minus the correlation.

   :return: List of row positions.
   """
   from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
   from scipy.spatial.distance import squareform
   M = np.asarray(matrix, dtype = float)
   k = len(M)
   if k < 3:
      return list(range(k))
   D = 1. - M if diverging else M.copy()
   finite = np.isfinite(D)
   D[~finite] = 2. * (np.abs(D[finite]).max() if finite.any() else 1.)
   D = np.maximum((D + D.T) / 2, 0.)
   np.fill_diagonal(D, 0.)
   y = squareform(D, checks = False)
   Z = linkage(y, "average")
   if k <= 2000:
      Z = optimal_leaf_ordering(Z, y)
   return [int(i) for i in leaves_list(Z)]


def matrix_order(ns, plot, key):
   """
   Computes (once) the order ``key`` of :func:`matrix_orders` of the heatmap ``plot`` of a tab,
   into ``plot["data"]["orders"]``: a random permutation (seed 0), :func:`block_order`, or, for
   the shortest path lengths of the largest component, the positions of its nodes in
   :func:`~graphoskop.matrix_layout` of the network (without attribute, or grouped by a
   qualitative attribute, or sorted by a numeric one; nothing is written to the network).
   """
   data = plot["data"]
   orders = data.setdefault("orders", {})
   if key == "computed" or key in orders:
      return
   M = data["matrix"]
   k = len(M)
   if key == "random":
      orders[key] = [int(i) for i in np.random.default_rng(0).permutation(k)]
   elif key == "blocks":
      orders[key] = block_order(M, data.get("diverging", False))
   else:
      net = _drawn(ns)
      if key == "matrix":
         table = gsk.matrix_layout(net, attribute = None)
      else:
         a = key.split(":", 1)[1]
         table = gsk.matrix_layout(net, groups = a, attribute = None) if a not in net.node_attributes else gsk.matrix_layout(net, order = a, attribute = None)
      orders[key] = [int(i) for i in np.argsort(table.iloc[:k, 0].to_numpy(), kind = "stable")]


def _color_norm(lo, hi, options, values):
   """
   Color scale of a heatmap (a matplotlib normalization): from ``zmin`` to ``zmax`` of the plot
   ``options`` where given, else from ``lo`` to ``hi`` (swapped if given the wrong way round,
   widened a little if empty); linear, or with ``zlog`` logarithmic: on a logarithmic scale if
   the range is positive, else symmetric logarithmic (logarithmic on both sides of zero, linear
   within the smallest nonzero magnitude of the ``values``, so that zeros and negative values,
   e.g. correlations, keep a color).
   """
   from matplotlib.colors import LogNorm, Normalize, SymLogNorm
   lo = lo if options.get("zmin") is None else float(options["zmin"])
   hi = hi if options.get("zmax") is None else float(options["zmax"])
   lo, hi = min(lo, hi), max(lo, hi)
   if hi <= lo:
      lo, hi = lo - 0.5, hi + 0.5
   if not options.get("zlog"):
      return Normalize(lo, hi)
   if lo > 0:
      return LogNorm(lo, hi)
   v = np.abs(np.asarray(values, dtype = float))
   v = v[np.isfinite(v) & (v > 0)]
   return SymLogNorm(float(v.min()) if v.size else max(abs(lo), abs(hi)) * 1e-3, vmin = lo, vmax = hi, base = 10)


def spectrum_plot(frequencies, energy, name, colors = None, options = None, styles = None, format = "png", path = None, dpi = 100):
   """
   Plot of the graph Fourier transform of a node attribute: the squared modulus of its coefficient
   on every Fourier mode (its energy there) against the frequency of the mode
   (:func:`~graphoskop.graph_fourier_transform`). Smooth signals have their energy at low
   frequencies.

   :param frequencies: Frequencies of the modes.
   :param energy: Squared moduli of the coefficients.
   :param name: Name of the attribute.
   :return: The bytes of the image, or ``None`` with ``path``.
   """
   colors = dict(PLOT_ELEMENTS["spectrum"][1], **(colors or {}))
   styles = dict({"data": "points"}, **(styles or {}))
   options = dict(PLOT_OPTIONS["spectrum"], **(options or {}))
   options["ylabel"] = options["ylabel"] or f"squared coefficient of {name}"
   return _figure(lambda ax: _series(ax, frequencies, energy, styles["data"], colors["data"], None), options, format, path, dpi)


def spectrum_text(frequencies, energy):
   """
   Text under the plot of a graph Fourier transform: the energy of the signal (the sum of its
   squared coefficients), its total variation :math:`\\sum_l \\lambda_l |\\hat{a}_l|^2`, and how
   many of the lowest frequencies hold half and 90% of its energy.
   """
   total = float(energy.sum())
   cum = np.cumsum(energy) / total if total > 0 else np.ones(len(energy))
   half, most = (int(np.searchsorted(cum, f)) + 1 for f in (0.5, 0.9))
   return f"Energy {total:.4g}, total variation {float((frequencies * energy).sum()):.4g}; half of the energy in the {half} lowest of {len(energy)} frequencies, 90% in the {most} lowest."


def _number(v):
   """
   A result of the library as a Python number (tensors, NumPy numbers; complex numbers with no
   imaginary part as real).
   """
   if hasattr(v, "item"):
      v = v.item()
   if isinstance(v, complex) and abs(v.imag) < 1e-12 * max(abs(v.real), 1.):
      v = v.real
   return v


def _vectors(request):
   """
   Command ``vectors``: runs the code of :func:`vector_code` in the namespace of a tab, and replies
   with the code and the drawing (unchanged), and with ``result``, a text with the value (one or
   two attributes), or ``plot`` (:func:`_plot_header` with its ``text`` and ``title``, the PNG
   image as the last part of the payload): the matrix of a pairwise analysis (:func:`matrix_plot`;
   ``table``: it can also be saved as a table) or the spectrum of a graph Fourier transform
   (:func:`spectrum_plot`). When the network has several components, the analysis is of the
   largest one, and the text says so.
   """
   ns = _spaces[request["tab"]]
   function, arguments = request["function"], dict(request.get("arguments") or {})
   if function not in VECTORS:
      raise ValueError(f"unknown node vector analysis: {function}")
   kind, result, label = VECTORS[function]
   if kind in ("pairwise", "one") and function != "network_variance" and function != "network_entropy":
      import matplotlib   # before running anything: the plot needs it
   names = [ast.literal_eval(arguments[k]) for k in ("a", "b") if arguments.get(k, "").strip()]
   code = _run(request["tab"], vector_code(ns, function, arguments))
   net = ns["net"]
   sizes_ = [int(c.node_attr.shape[0]) if c.node_attr is not None else 0 for c in net.components] if net.higher_order is None else []
   where = f" (largest component: {sizes_[0]} of {sum(sizes_)} nodes)" if len(sizes_) > 1 else ""
   header, payload = _draw(ns, code, True)
   name = request.get("name", "the network")
   if kind == "pairwise":
      chosen = _source_value(ns, arguments["attributes"]) if arguments.get("attributes", "").strip() else list(net.node_attributes)
      M = ns[result].double().cpu().numpy()
      diverging = function == "pairwise_network_correlation"
      what = "network correlation" if diverging else "generalized Euclidean distance"
      ns["__plot__"] = _plot("matrix", {"matrix": M, "names": [str(a) for a in chosen], "what": what, "diverging": diverging, "orders": {}}, {"negative": "#377eb8", "positive": "#e41a1c"} if diverging else dict(PLOT_ELEMENTS["matrix"][1]), {}, {"negative": "Negative values", "positive": "Positive values"} if diverging else None, source = {"matrix": f"{result}.double().cpu().numpy()", "names": repr([str(a) for a in chosen]), "what": literal(what), "diverging": repr(diverging)})
      ns["__plot__"]["table"] = _matrix_table(M, chosen)
      text_ = f"{label} between {len(chosen)} node attributes{where}."
   elif function == "graph_fourier_transform":
      import torch
      freq = ns["basis"][0].double().cpu().numpy()
      c = ns[result]
      energy = (c.abs() ** 2).double().cpu().numpy() if torch.is_complex(c) else (c.double() ** 2).cpu().numpy()
      if np.isnan(energy).any():
         raise ValueError(f"{names[0]} has missing values: its Fourier transform is not defined")
      ns["__plot__"] = _plot("spectrum", {"frequencies": freq, "energy": energy, "name": names[0]}, dict(PLOT_ELEMENTS["spectrum"][1]), {"data": "points"}, source = {"frequencies": "basis[0].double().cpu().numpy()", "energy": f"({result}.abs() ** 2).double().cpu().numpy()", "name": literal(names[0])})
      text_ = spectrum_text(freq, energy) + (f" {where.strip(' ()').capitalize()}." if where else "")
   else:
      value = _number(ns[result])
      of = " and ".join(names)
      shown = f"{value:.6g}" if isinstance(value, (int, float)) else str(value)
      header["result"] = f"{label} of {of}{where}: {shown}"
      return header, payload
   png = _render(ns["__plot__"], "png")
   header["plot"] = dict(_plot_header(ns["__plot__"], ns), text = text_, warnings = [], title = f"{label} of {name}", table = kind == "pairwise")
   header["parts"] = header.get("parts", []) + [len(png)] if header["nodes"] else [len(png)]
   return header, payload + png


def _matrix_table(matrix, names):
   """
   Table of a pairwise matrix, rows and columns named by the attributes (saved by Save table… of
   the plot window, see :func:`_save_plot`).
   """
   import pandas as pd
   return pd.DataFrame(matrix, index = [str(a) for a in names], columns = [str(a) for a in names])


#: Kinds of distribution plots (:func:`distribution_plot`): label in the plot window, default
#: style of the data, and label of the y axis (with the plural of what has the values).
DISTRIBUTIONS = {"count": ("Count", "bars", "number of {what}"), "pdf": ("PDF", "bars", "fraction of {what}"), "cdf": ("CDF", "lines", "fraction of {what} with value ≤ x"), "ccdf": ("CCDF", "lines", "fraction of {what} with value ≥ x")}
#: Number of bins of the counts and PDFs of numeric attributes with many distinct values.
DISTRIBUTION_BINS = 50
#: Numeric attributes with at most this many distinct values, all integers, have one bar per value.
DISTRIBUTION_DISCRETE = 100
#: Most categories shown in the distribution of a qualitative attribute (the most frequent ones).
DISTRIBUTION_CATEGORIES = 40


def distribution(values, kind = "count", log = False):
   """
   Distribution of the values of an attribute, for :func:`distribution_plot`.

   * Qualitative values (anything but a float or integer array, or a :class:`pandas.Categorical`):
     the number (``"count"``) or the fraction (``"pdf"``) of the values in each category, most
     frequent first, the first :data:`DISTRIBUTION_CATEGORIES` only.
   * Numbers, missing ones left out: with ``"count"`` and ``"pdf"``, one bar per value if they are
     at most :data:`DISTRIBUTION_DISCRETE` integers (the fraction of the values for ``"pdf"``),
     else :data:`DISTRIBUTION_BINS` bins of the same width (logarithmic with ``log`` if all values
     are positive), at their centers (the probability density for ``"pdf"``); with ``"cdf"``
     (``"ccdf"``), at every distinct value the fraction of the values at most (at least) as large.

   :param values: Array of the values, or :class:`pandas.Categorical`.
   :param kind: One of :data:`DISTRIBUTIONS`.
   :param log: Whether the x axis is logarithmic (for the bins).
   :return: ``(x, y, labels, n)``: positions and heights (NumPy arrays), the category of every
      position (``None`` for numbers), and the number of values (not missing).
   """
   import pandas as pd
   v = values if isinstance(values, pd.Categorical) else np.asarray(values)
   if isinstance(v, pd.Categorical) or v.dtype.kind not in "fiu":
      counts = pd.Series(v).value_counts(dropna = True)
      counts = counts[counts > 0]
      n = int(counts.sum())
      top = counts.iloc[:DISTRIBUTION_CATEGORIES]
      y = top.to_numpy(dtype = float) / (n if kind == "pdf" else 1)
      return np.arange(len(top), dtype = float), y, [str(text(c)) for c in top.index], n
   v = v.astype(float)
   v = v[np.isfinite(v)]
   n = len(v)
   if n == 0:
      return np.zeros(0), np.zeros(0), None, 0
   if kind in ("cdf", "ccdf"):
      x, c = np.unique(v, return_counts = True)
      y = np.cumsum(c) / n if kind == "cdf" else np.cumsum(c[::-1])[::-1] / n
      return x, y, None, n
   x, c = np.unique(v, return_counts = True)
   if len(x) <= DISTRIBUTION_DISCRETE and np.array_equal(x, np.round(x)):
      return x, c / (n if kind == "pdf" else 1), None, n
   positive = log and x[0] > 0
   edges = np.geomspace(x[0], x[-1], DISTRIBUTION_BINS + 1) if positive else np.linspace(x[0], x[-1], DISTRIBUTION_BINS + 1)
   c, edges = np.histogram(v, edges)
   x = np.sqrt(edges[:-1] * edges[1:]) if positive else (edges[:-1] + edges[1:]) / 2
   return x, c / (n * np.diff(edges)) if kind == "pdf" else c.astype(float), None, n


def distribution_plot(values, name, what = "nodes", qualitative = False, colors = None, options = None, styles = None, format = "png", path = None, dpi = 100):
   """
   Plot of the distribution of the values of an attribute (:func:`distribution`): count, PDF, CDF
   or CCDF (``options["kind"]``, see :data:`DISTRIBUTIONS`), the categories of qualitative
   attributes (count and PDF only) along the x axis. Empty axis labels are automatic: the name of
   the attribute, and the label of the kind (see ``_figure``).

   :param values: Values of the attribute (array, or :class:`pandas.Categorical`).
   :param name: Name of the attribute.
   :param what: What has the values, plural (``"nodes"``, ``"edges"``).
   :param qualitative: Whether the attribute is qualitative.
   :param colors: Color of the ``data``, or ``None`` for the default.
   :param options: Options of the plot (see :data:`PLOT_OPTIONS`), or ``None`` for the defaults.
   :param styles: How the ``data`` is drawn (:data:`PLOT_STYLES`; by default as in
      :data:`DISTRIBUTIONS`).
   :param format: ``"png"`` or ``"pdf"``.
   :param path: File to write, or ``None`` to return the bytes.
   :param dpi: Resolution of PNG images.
   :return: The bytes of the image, or ``None`` with ``path``.
   """
   options = dict(PLOT_OPTIONS["distribution"], **(options or {}))
   kind = options["kind"] if options["kind"] in DISTRIBUTIONS and not (qualitative and options["kind"] in ("cdf", "ccdf")) else "count"
   if qualitative:
      options["xlog"] = False
   x, y, labels, n = distribution(values, kind, options["xlog"])
   discrete = labels is not None or (kind in ("count", "pdf") and len(x) and np.array_equal(x, np.round(x)) and len(x) <= DISTRIBUTION_DISCRETE)
   options["xlabel"] = options["xlabel"] or name
   options["ylabel"] = options["ylabel"] or ("probability density" if kind == "pdf" and not discrete else DISTRIBUTIONS[kind][2].format(what = what))
   color = (colors or {}).get("data", PLOT_ELEMENTS["distribution"][1]["data"])
   style = (styles or {}).get("data", DISTRIBUTIONS[kind][1])
   def draw(ax):
      _series(ax, x, y, style, color, None)
      if labels is not None:
         ax.set_xticks(x, labels, rotation = 45 if len(labels) > 6 else 0, ha = "right" if len(labels) > 6 else "center")
   return _figure(draw, options, format, path, dpi)


def _distribution(request):
   """
   Command ``distribution``: plots the distribution of the node (``kind = "nodes"``) or edge
   attribute ``attribute`` of the drawn network of a tab (:func:`distribution_plot`, as a count
   at first), and replies with the PNG image as payload, ``elements``, ``options``, ``styles``,
   ``kinds`` (see :func:`_plot_header`) and ``text``: the number of values, their mean, median
   and range (numbers) or their number of categories, and the number of missing values.
   """
   import pandas as pd
   ns = _spaces[request["tab"]]
   draw = _drawn(ns)
   kind, name = request["kind"], request["attribute"]
   names = (list(draw.node_attributes) + list(draw.qualitative_node_attributes) + ([gsk.BIPARTITE] if draw.bipartite_labels is not None else [])) if kind == "nodes" else list(draw.edge_attributes) + list(draw.qualitative_edge_attributes)
   if name not in names:
      raise ValueError(f"{name} is not {'a node' if kind == 'nodes' else 'an edge'} attribute: only attributes have a distribution")
   values = _column(draw, name) if kind == "nodes" else edge_values(draw, name)
   qualitative = isinstance(values, pd.Categorical) or np.asarray(values).dtype.kind not in "fiu"
   ns["__plot__"] = _plot("distribution", {"values": values, "name": name, "what": kind, "qualitative": qualitative}, dict(PLOT_ELEMENTS["distribution"][1]), {"data": DISTRIBUTIONS["count"][1]}, source = {"values": f"attribute_values({_target(ns)}, {literal(kind)}, {literal(name)})", "name": literal(name), "what": literal(kind), "qualitative": repr(bool(qualitative))})
   total = len(values)
   if qualitative:
      counts = pd.Series(values).value_counts(dropna = True)
      n, k = int(counts.sum()), int((counts > 0).sum())
      text_ = f"{n} {kind}, {k} categories" + (f" (the {DISTRIBUTION_CATEGORIES} most frequent shown)" if k > DISTRIBUTION_CATEGORIES else "")
   else:
      v = np.asarray(values, dtype = float)
      v = v[np.isfinite(v)]
      n = len(v)
      text_ = f"{n} {kind}" + (f": mean {v.mean():.4g}, median {np.median(v):.4g}, from {v.min():.4g} to {v.max():.4g}" if n else "")
   text_ += f"; {total - n} missing." if total > n else "."
   header = dict(_plot_header(ns["__plot__"]), text = text_, warnings = [])
   return header, _render(ns["__plot__"], "png")


#: Names of the visual attributes of the style (see :func:`default_style`), for messages.
STYLE_LABELS = {"nodes": "node color", "node_size": "node size", "node_opacity": "node opacity", "edges": "edge color", "edge_width": "edge width", "edge_opacity": "edge opacity"}


def _remove_attribute(request):
   """
   Command ``remove_attribute``: removes the node (``kind = "nodes"``) or edge attribute
   ``attribute`` of the drawn network of a tab, with the code ``net.remove_node_attribute("x")``
   (``remove_edge_attribute``), and replies with the code and the drawing. Ids, the positions of
   the layout (``pos_x``, ``pos_y``) and the attributes the drawing uses (colors, sizes,
   opacities, positions: the visual attributes never change by themselves) are not removed: the
   error says why.
   """
   ns = _spaces[request["tab"]]
   draw = _drawn(ns)
   kind, name = request["kind"], request["attribute"]
   names = list(draw.node_attributes) + list(draw.qualitative_node_attributes) if kind == "nodes" else list(draw.edge_attributes) + list(draw.qualitative_edge_attributes)
   if name not in names:
      raise ValueError(f"{name} is not {'a node' if kind == 'nodes' else 'an edge'} attribute: it cannot be removed")
   if kind == "nodes" and name in ("pos_x", "pos_y"):
      raise ValueError(f"{name} holds the positions of the layout: it cannot be removed")
   style = ns.get("__style__", {})
   keys = ["nodes", "node_size", "node_opacity"] if kind == "nodes" else ["edges", "edge_width", "edge_opacity"]
   used = [STYLE_LABELS[k] for k in keys if (style.get(k) or {}).get("attribute") == name]
   used += ["node position"] if kind == "nodes" and name in (style.get("position") or {}).values() else []
   if used:
      raise ValueError(f"{name} is used by the drawing ({', '.join(used)}): map {'it' if len(used) == 1 else 'them'} to something else first")
   code = _run(request["tab"], [f"{_target(ns)}.remove_{kind[:-1]}_attribute({literal(name)})"])
   return _draw(ns, code, True)


def _save_plot(request):
   """
   Command ``save_plot``: writes the last plot of a tab (with its current colors) to ``path``, as
   PNG (200 dpi) or PDF, by the extension of the path; or, for a path ending in ``.tsv``, the
   table of the plot (the matrix of a heatmap), tab separated. The code that writes it is added
   to the code of the tab (:func:`plot_code`), and replied as ``code``.
   """
   ns = _spaces[request["tab"]]
   if "__plot__" not in ns:
      raise ValueError("no plot to save")
   path = request["path"]
   if path.lower().endswith(".tsv") and "table" not in ns["__plot__"]:
      raise ValueError("this plot has no table")
   lines = plot_code(ns, path)
   if lines:   # the code writes the file
      code = _run(request["tab"], lines)
      return {"path": path, "code": code}, b""
   if path.lower().endswith(".tsv"):
      ns["__plot__"]["table"].to_csv(path, sep = "\t")
   else:
      _render(ns["__plot__"], "pdf" if path.lower().endswith(".pdf") else "png", path, 200)
   return {"path": path}, b""


def _target(ns):
   """
   Name of the drawn network in the namespace of a tab: ``inc`` (incidence network of a hypergraph
   or complex) or ``net``.
   """
   return "inc" if "inc" in ns else "net"


def _move(request):
   """
   Command ``move``: nodes dragged in the visualization panel (``nodes``, positions in the drawing
   order; ``positions``, their new places on screen axes, y down) get new positions, with the code
   ``move_nodes(net, {id: (x, y), ...})`` (:func:`move_nodes`, after :data:`MOVE_NOTE`); replies
   with the code and the drawing.
   """
   ns = _spaces[request["tab"]]
   ids = _ids(_drawn(ns))
   moved = ", ".join(f"{_id_literal(ids[int(i)])}: ({float(f'{x:.6g}')!r}, {float(f'{-y:.6g}')!r})" for i, (x, y) in zip(request["nodes"], request["positions"]))
   lines = ([] if MOVE_NOTE in ns["__code__"] else [MOVE_NOTE]) + [f"move_nodes({_target(ns)}, {{{moved}}})"]
   code = _run(request["tab"], lines, True)
   return _draw(ns, code, True)


#: Comment written before the code of the first edge or node added in the visualization panel.
ADD_NOTE = "# add_nodes / add_edges: nodes and edges added in the graphical interface"


def adding(net):
   """
   What the menu of the visualization panel needs to add a node to a network without asking
   anything: whether nodes and edges can be added (``False`` for hypergraphs and simplicial
   complexes), else the ``layers`` of a multilayer network and the node ``types`` of a bipartite
   one (texts; ``None`` if it has none), one of which the new node takes (menu items).
   """
   if net.higher_order is not None:
      return False
   return {"layers": [str(text(l)) for l in net.layer_labels] if net.layer_labels is not None else None, "types": [str(text(v)) for v in net.bipartite_labels] if net.bipartite_labels is not None else None}


def free_id(net):
   """
   The id of a new node: the smallest integer from 0 up that is not the id of a node (of an entity,
   in multilayer networks), as a number if the ids of the network are numbers, else as a text.
   """
   ids = _ids(net)
   entities = [i[0] for i in ids] if net.layer_labels is not None else list(ids)
   numeric = all(isinstance(i, (int, np.integer)) for i in entities)
   used = set(int(i) for i in entities) if numeric else set(str(i) for i in entities)
   k = 0
   while (k if numeric else str(k)) in used:
      k += 1
   return k if numeric else str(k)


def _value(net, kind, name, typed):
   """
   Python source of a value typed in the data panel: a number for numeric attributes
   (:class:`ValueError` if it is not one, ``float("nan")`` if empty), and for qualitative ones the
   existing value with that text (so that ``3`` stays a number if the values are numbers), the text
   otherwise, or ``None`` if empty (missing).
   """
   numeric = name in (net.node_attributes if kind == "nodes" else net.edge_attributes)
   if not typed:
      return "float(\"nan\")" if numeric else "None"
   if numeric:
      try:
         return repr(float(typed))
      except ValueError:
         raise ValueError(f"{name}: not a number: {typed}") from None
   values = (node_values if kind == "nodes" else edge_values)(net, name)
   for v in _unique(values):
      if text(v) == typed:
         return _id_literal(v)
   return literal(typed)


def _pandas(ns):
   """
   ``import pandas as pd`` if the code of a tab does not import it yet (the code is checked, not
   the namespace: a line that failed may have imported it without being kept).
   """
   return [] if "import pandas as pd" in ns["__code__"] else ["import pandas as pd"]


def _unique(values):
   """
   Distinct values of an object array (missing values left out), in order of appearance.
   """
   import pandas as pd
   return [v for v in pd.unique(pd.Series(values, dtype = object)) if text(v) is not None]


def add_code(ns, request):
   """
   Code adding a node or an edge to the network of a tab, from the menu of the visualization
   panel (:func:`_add`), without asking anything: attributes of new nodes and edges are missing
   (edge weights 1, see :func:`~graphoskop.add_edges`).

   A node (``kind`` ``"node"``) gets the id :func:`free_id`, the ``layer`` of the menu item in
   multilayer networks and its ``type`` in bipartite ones, and is put at ``position`` (drawing
   coordinates, y down), e.g. ``net = gsk.add_nodes(net, 0)`` and ``move_nodes(net, {0: (1.5,
   -0.2)})``; in bipartite networks with a table, ``net = gsk.add_nodes(net,
   pd.DataFrame({"node": [0], "bipartite_type": ["u"]}))``. An edge (``kind`` ``"edge"``) goes
   from ``source`` to ``target`` (positions of the nodes in the drawing order), e.g. ``net =
   gsk.add_edges(net, ("a", "d"))``.

   :param ns: Namespace of the tab.
   :param request: The request.
   :return: List of lines of code.
   :raises ValueError: For hypergraphs and simplicial complexes, or an unknown layer or type.
   """
   net = ns["net"]
   if net.higher_order is not None:
      raise ValueError("nodes and edges can be added only to networks, not to hypergraphs or simplicial complexes")
   lines = [] if ADD_NOTE in ns["__code__"] else [ADD_NOTE]
   if request.get("kind") != "node":
      ids = _ids(_drawn(ns))
      u, v = ids[int(request["source"])], ids[int(request["target"])]
      return lines + [f"net = gsk.add_edges(net, ({_id_literal(u)}, {_id_literal(v)}))"]
   node = free_id(net)
   if net.layer_labels is not None:
      layer = next((l for l in net.layer_labels if str(text(l)) == str(request.get("layer"))), None)
      if layer is None:
         raise ValueError(f"no layer {request.get('layer')!r}")
      node = (node, layer)
   if net.bipartite_labels is not None:
      kind = next((t for t in net.bipartite_labels if str(text(t)) == str(request.get("type"))), None)
      if kind is None:
         raise ValueError(f"no node type {request.get('type')!r}")
      key = [("node", _id_literal(node[0] if isinstance(node, tuple) else node))] + ([("layer", _id_literal(node[1]))] if isinstance(node, tuple) else [])
      columns = key + ([("attribute", literal(gsk.BIPARTITE)), ("value", _id_literal(kind))] if net.node_format == "long" else [(gsk.BIPARTITE, _id_literal(kind))])
      lines += _pandas(ns) + ["net = gsk.add_nodes(net, pd.DataFrame({" + ", ".join(f"{literal(k)}: [{v}]" for k, v in columns) + "}))"]
   else:
      lines.append(f"net = gsk.add_nodes(net, {_id_literal(node)})")
   x, y = request.get("position") or (0., 0.)
   return lines + ([] if MOVE_NOTE in ns["__code__"] else [MOVE_NOTE]) + [f"move_nodes(net, {{{_id_literal(node)}: ({float(f'{x:.6g}')!r}, {float(f'{-y:.6g}')!r})}})"]


def _add(request):
   """
   Command ``add``: adds a node or an edge to the network of a tab, with the code of
   :func:`add_code`; it can be undone (Edit › Undo). Replies with the code, a ``summary`` and the
   drawing.
   """
   ns = _spaces[request["tab"]]
   code = _run(request["tab"], add_code(ns, request), True)
   header, payload = _draw(ns, code, True)
   drawn = _drawn(ns)
   header["summary"] = f"{drawn.number_of_nodes()} nodes, {drawn.number_of_edges()} edges"
   return header, payload


#: Comment written before the code of the first edge attribute value edited in the data panel.
EDGE_VALUES_NOTE = "# set_edge_values: values of edge attributes edited in the graphical interface"


def set_edge_values(net, name, values):
   """
   Sets values of an edge attribute, as edited in the data panel. A function of the interface, not
   of the library: it is defined in the namespace of every tab, for the code of the edits
   (:data:`EDGE_VALUES_NOTE`); the library sets whole vectors
   (:meth:`~graphoskop.Network.set_edge_attribute`).

   :param net: A :class:`~graphoskop.Network`.
   :param name: Name of the edge attribute.
   :param values: Dictionary from the position of an edge in the input order of the network (the
      rows of the Edges table) to its new value (NaN or ``None`` for missing).
   """
   import torch
   positions = np.fromiter(values.keys(), dtype = np.int64, count = len(values))
   found = {int(p): v for p, v in values.items()}
   numeric = name in net.edge_attributes
   columns = []
   for c in range(len(net.components)):
      old = net.get_edge_attribute(name, c)
      rows = net._arc_rows(c)   # input edge of every stored edge
      hit = np.flatnonzero(np.isin(rows, positions))
      if numeric:
         old = old.clone()
         if len(hit):
            old[torch.as_tensor(hit, device = old.device)] = torch.tensor([float("nan") if found[int(rows[h])] is None else float(found[int(rows[h])]) for h in hit], dtype = old.dtype, device = old.device)
      else:
         old = np.array(old, dtype = object)
         for h in hit:
            old[h] = found[int(rows[h])]
      columns.append(old)
   net.set_edge_attribute(name, columns, None)


def value_code(ns, request):
   """
   Code setting one value of an attribute, edited in the data panel (``kind`` ``"nodes"`` or
   ``"edges"``, the ``position`` of the node or edge in the drawing order, the ``attribute`` and
   the ``text`` typed, :func:`_value`): for a node, with the table form of
   :meth:`~graphoskop.Network.set_node_attribute`, e.g. ``net.set_node_attribute(pd.DataFrame(
   {"node": ["a"], "age": [41.0]}), node_column = "node")`` (multilayer: with its layer,
   ``layer_column``); for an
   edge, with :func:`set_edge_values`, e.g. ``set_edge_values(net, "weight", {3: 2.5})   # a ->
   b``.

   :raises ValueError: For hypergraphs and simplicial complexes, ids, the node type of bipartite
      networks, unknown attributes, and texts that are not numbers for numeric attributes.
   """
   net = ns["net"]
   if net.higher_order is not None:
      raise ValueError("the attributes of hypergraphs and simplicial complexes cannot be edited here")
   kind, name, i = request["kind"], request["attribute"], int(request["position"])
   if kind == "nodes":
      if name not in list(net.node_attributes) + list(net.qualitative_node_attributes):
         raise ValueError(f"{name}: ids cannot be edited" if name == "node" else f"no node attribute {name!r}")
      if net.bipartite_labels is not None and name == gsk.BIPARTITE:
         raise ValueError("the node type of a bipartite network cannot be edited")
      node = _ids(net)[i]
      value = _value(net, kind, name, str(request.get("text", "")).strip())
      key = [("node", _id_literal(node[0])), ("layer", _id_literal(node[1]))] if net.layer_labels is not None else [("node", _id_literal(node))]
      frame = "pd.DataFrame({" + ", ".join(f"{literal(k)}: [{v}]" for k, v in key + [(name, value)]) + "})"
      return _pandas(ns) + [f"net.set_node_attribute({frame}, node_column = \"node\"{', layer_column = ' + literal('layer') if net.layer_labels is not None else ''})"]
   if name not in list(net.edge_attributes) + list(net.qualitative_edge_attributes):
      raise ValueError(f"{name}: the endpoints of an edge cannot be edited" if name in ("source", "target") else f"no edge attribute {name!r}")
   ids = _ids(net)
   s, t = _edges(net)
   value = _value(net, kind, name, str(request.get("text", "")).strip())
   arrow = "->" if net.directed else "--"
   return ([] if EDGE_VALUES_NOTE in ns["__code__"] else [EDGE_VALUES_NOTE]) + [f"set_edge_values(net, {literal(name)}, {{{i}: {value}}})   # {text(ids[s[i]])} {arrow} {text(ids[t[i]])}"]


def _set_value(request):
   """
   Command ``set_value``: one value of a node or edge attribute edited in the data panel (code of
   :func:`value_code`); it can be undone (Edit › Undo: the attribute gets back its values, see
   :func:`_snapshot`). Replies with the code and the drawing (the style may use the attribute).
   """
   ns = _spaces[request["tab"]]
   code = _run(request["tab"], value_code(ns, request), True, (request["kind"], request["attribute"]))
   return _draw(ns, code, True)


def _neighbors(net, nodes, both):
   """
   Positions of the neighbors of ``nodes`` in the drawn network (successors only in directed
   networks, unless ``both``).
   """
   s, t = _edges(net)
   chosen = np.isin(s, nodes)
   found = set(t[chosen].tolist())
   if both or not net.directed:
      found |= set(s[np.isin(t, nodes)].tolist())
   return sorted(found)


def edit_code(ns, op, nodes, edges, include_ego = True):
   """
   Code of an operation of the Edit menu on the drawn network of a tab: ``remove_nodes``,
   ``remove_edges``, ``remove`` (both), ``subgraph`` (induced by the nodes) or ``ego`` (the ego
   networks of the nodes: them, without them if not ``include_ego``, and their neighbors).

   :param ns: Namespace of the tab.
   :param op: The operation.
   :param nodes: Positions of the selected nodes in the drawing order.
   :param edges: Positions of the selected edges in the drawing order.
   :param include_ego: Whether ego networks keep the egos.
   :return: List of lines of code.
   :raises ValueError: If nothing is selected for the operation.
   """
   t = _target(ns)
   net = _drawn(ns)
   ids = _ids(net)
   s, d = _edges(net)
   node_list = "[" + ", ".join(_id_literal(ids[i]) for i in nodes) + "]"
   edge_list = "[" + ", ".join(f"({_id_literal(ids[s[e]])}, {_id_literal(ids[d[e]])})" for e in edges) + "]"
   lines = []
   if op in ("remove_edges", "remove") and len(edges):
      lines.append(f"{t} = gsk.remove_edges({t}, {edge_list})")
   if op in ("remove_nodes", "remove") and len(nodes):
      lines.append(f"{t} = gsk.remove_nodes({t}, {node_list})")
   if op == "subgraph" and len(nodes):
      lines.append(f"{t} = gsk.subgraph({t}, {node_list})")
   if op == "ego" and len(nodes) == 1:
      lines.append(f"{t} = gsk.ego_network({t}, {_id_literal(ids[nodes[0]])}{'' if include_ego else ', include_ego = False'})")
   elif op == "ego" and len(nodes):
      keep = set(_neighbors(net, nodes, False)) | (set(nodes) if include_ego else set())
      keep -= set() if include_ego else set(nodes)
      lines += [f"# ego networks of {node_list}" + ("" if include_ego else ", without the egos"), f"{t} = gsk.subgraph({t}, [" + ", ".join(_id_literal(ids[i]) for i in sorted(keep)) + "])"]
   if not lines:
      raise ValueError("select the nodes (or edges) first")
   return lines


def _edit(request):
   """
   Command ``edit``: an operation of :func:`edit_code` (``op``, ``nodes``, ``edges``,
   ``include_ego``). Removals change the network of the tab; ``subgraph`` and ``ego`` make the
   network of a new tab ``new_tab``, whose code is the code of the tab plus the new lines. Nodes
   keep their positions and the style is kept. Replies with the code, a ``summary`` and the
   drawing.
   """
   ns = _spaces[request["tab"]]
   lines = edit_code(ns, request["op"], [int(i) for i in request.get("nodes") or []], [int(e) for e in request.get("edges") or []], bool(request.get("include_ego", True)))
   if request["op"] in ("subgraph", "ego"):
      return _derived(ns, request["new_tab"], lines)
   code = _run(request["tab"], lines, True)   # removals can be undone
   header, payload = _draw(ns, code, True)
   drawn = _drawn(ns)
   header["summary"] = f"{drawn.number_of_nodes()} nodes, {drawn.number_of_edges()} edges"
   return header, payload


def _derived(ns, tab, lines, preset = None, unrun = ()):
   """
   A new tab ``tab`` whose network is derived from the network of the namespace ``ns`` (induced
   subgraph, ego network, backbone): its namespace copies the variables, the code, the style and
   the node size of ``ns``, then runs ``lines``. Replies with the code, a ``summary`` and the
   drawing; the new namespace is dropped if the lines fail. If ``ns`` is not drawn, the new
   network is not drawn either while it is :func:`too_large`. ``preset`` variables are set in the
   new namespace first, and the ``unrun`` lines are written in the code but not run (they make
   those variables when the code runs on its own: the network of another tab, see
   :func:`other_code`).
   """
   import copy
   _spaces[tab] = {k: v for k, v in ns.items() if not k.startswith("__")}
   _spaces[tab].update(__code__ = list(ns["__code__"]), __style__ = copy.deepcopy(ns.get("__style__", {})), __size__ = ns.get("__size__"))
   _spaces[tab].update(preset or {})
   run = [l for l in lines if l not in unrun]
   try:
      code = _run(tab, run)
   except BaseException:
      _spaces.pop(tab, None)
      raise
   new = _spaces[tab]
   if len(run) < len(lines):   # the code as written
      new["__code__"] = new["__code__"][:len(new["__code__"]) - len(run)] + list(lines)
      code = "\n".join(new["__code__"])
   if ns.get("__undrawn__") and too_large(_drawn(new)):
      new["__undrawn__"] = True
   header, payload = _draw(new, code, True)
   drawn = _drawn(new)
   header["summary"] = f"{drawn.number_of_nodes()} nodes, {drawn.number_of_edges()} edges"
   return header, payload


#: Most nodes listed by a search (:func:`find_nodes`).
MAX_FOUND = 200


def find_nodes(net, text_, attribute = None):
   """
   Nodes of the drawn network whose id (or ``attribute``) matches a text, ignoring case: those
   whose text (:func:`text`) is the text, and those whose text contains it.

   :param net: Network.
   :param text_: Text searched.
   :param attribute: Node attribute, or ``None`` for the ids.
   :return: ``(exact, partial)``: positions in the drawing order (int64 arrays); ``partial``
      leaves out the exact matches.
   """
   import pandas as pd
   values = _ids(net) if attribute is None else _column(net, attribute)
   texts, pos = distinct(values)
   if not texts:
      return np.zeros(0, np.int64), np.zeros(0, np.int64)
   low = pd.Series(texts, dtype = object).str.lower()
   q = str(text_).strip().lower()
   same, inside = (low == q).to_numpy(), low.str.contains(q, regex = False).to_numpy()
   ok = pos >= 0
   exact = np.flatnonzero(ok & np.append(same, False)[pos])
   partial = np.flatnonzero(ok & np.append(inside & ~same, False)[pos])
   return exact, partial


def _find(request):
   """
   Command ``find``: :func:`find_nodes` of ``text`` in the ``attribute`` of the drawn network of a
   tab (``None``: the ids). Replies with the counts and the first :data:`MAX_FOUND` positions of
   the ``exact`` and ``partial`` matches, and ``labels``: the id of each of them, with the
   matching value when searching an attribute.
   """
   net = _drawn(_spaces[request["tab"]])
   attribute = request.get("attribute")
   exact, partial = find_nodes(net, request.get("text", ""), attribute)
   ids = _ids(net)
   values = None if attribute is None else _column(net, attribute)
   label = lambda i: str(text(ids[i])) + ("" if values is None else f" ({text(values[i])})")
   shown = np.concatenate([exact[:MAX_FOUND], partial[:max(0, MAX_FOUND - len(exact))]])
   return {"exact": exact[:MAX_FOUND].tolist(), "exact_count": len(exact), "partial": partial[:max(0, MAX_FOUND - len(exact))].tolist(), "partial_count": len(partial), "labels": [label(int(i)) for i in shown]}, b""


def matches(values, op, value):
   """
   Whether every value of an attribute satisfies a condition: ``op`` (``"="``, ``"≠"``, ``"<"``,
   ``"≤"``, ``">"``, ``"≥"`` or ``"contains"``) with ``value`` (a text: a number for numeric
   comparisons; texts of qualitative values are compared, see :func:`text`; with ``contains``, a
   part of the text, or one of the values of a node with several). Missing values never match.

   :param values: Values of the attribute (:func:`node_values` or :func:`edge_values`).
   :return: Boolean array.
   :raises ValueError: If a numeric comparison gets a value that is not a number.
   """
   value = str(value).strip()
   if values.dtype.kind == "f" and op != "contains":
      x = float(value)
      with np.errstate(invalid = "ignore"):
         return {"=": values == x, "≠": (values != x) & ~np.isnan(values), "<": values < x, "≤": values <= x, ">": values > x, "≥": values >= x}[op]
   def match(t):
      if t is None:
         return False
      if op == "contains":
         return value in t or value in [p.strip() for p in t.split(",")]
      try:
         a, b = float(t), float(value)
      except ValueError:
         a, b = t, value
      return {"=": a == b, "≠": a != b, "<": a < b, "≤": a <= b, ">": a > b, "≥": a >= b}[op] if type(a) is type(b) else op == "≠"
   return np.array([match(text(v)) for v in values], dtype = bool)


#: How Select › Select by attribute chains a condition to the ones before it.
LOGIC = {"and": np.logical_and, "or": np.logical_or, "xor": np.logical_xor}


def _select(request):
   """
   Command ``select``: positions (in the drawing order) of the nodes (``kind = "nodes"``) or edges
   (``"edges"``) of the drawn network of a tab that satisfy the ``conditions``: a list of
   dictionaries with ``attribute``, ``op``, ``value`` (see :func:`matches`) and, from the second
   on, ``logic`` (``"and"``, ``"or"`` or ``"xor"``, :data:`LOGIC`), which chains the condition to
   the result of the ones before it, from left to right (``a or b and c`` is ``(a or b) and c``).
   A request with ``attribute``, ``op`` and ``value`` instead is a single condition.
   """
   net = _drawn(_spaces[request["tab"]])
   kind = request["kind"]
   conditions = request.get("conditions") or [{"attribute": request["attribute"], "op": request["op"], "value": request["value"]}]
   hit = None
   for c in conditions:
      values = node_values(net, c["attribute"]) if kind == "nodes" else edge_values(net, c["attribute"])
      h = matches(values, c["op"], c["value"])
      hit = h if hit is None else LOGIC[c.get("logic", "and")](hit, h)
   return {"indices": np.flatnonzero(hit).tolist()}, b""


#: Functions of Edit › Backbone that score the edges: the prefix of the attributes they write
#: (``prefix + "_score"``, and ``prefix + "_sdev"`` for the noise corrected backbone).
BACKBONES = {"noise_corrected": "nc", "disparity_filter": "df", "polya_filter": "pf", "doubly_stochastic": "ds", "high_salience_skeleton": "hss", "convex_skeleton": "cs"}
#: Functions of Edit › Backbone that filter the edges directly: their backbone opens in a new tab.
FILTERS = {"spanning_tree", "maximally_filtered_graph", "triangulated_filtered_graph"}
#: Parameters of the backbone functions not shown in the parameter dialog.
BACKBONE_HIDDEN = {"net", "prefix", "progress"}
#: Steps of the backbone plot, from keeping every edge to dropping them all (quantiles of the
#: significances, see :func:`~graphoskop.test_bb_thresholds`).
BACKBONE_STEPS = 100
#: Colors of the backbone plot: the series, and the thresholds of
#: :func:`~graphoskop.find_bb_thresholds`.
BACKBONE_COLORS = {"edges_pct": "#377eb8", "nodes_pct": "#e41a1c", "components": "#999999", "first_isolate": "#4daf4a", "first_component_break": "#984ea3", "max_separation": "#ff7f00", "added": "#a65628"}


def backbone_code(ns, function, arguments):
   """
   Code scoring the edges of the drawn network of a tab with a backbone function of
   :data:`BACKBONES` (the scores become edge attributes, see :mod:`graphoskop.backboning`), e.g.
   ``gsk.noise_corrected(net)``.

   :return: List of lines, and the name of the score attribute.
   """
   extra = "".join(f", {k} = {v.strip()}" for k, v in arguments.items())
   return [f"gsk.{function}({_target(ns)}{extra})"], BACKBONES[function] + "_score"


def backbone_scale(thresholds):
   """
   Scale of the threshold axis of the backbone plot: linear, unless the thresholds have a long tail
   (their range is more than 20 times their interquartile range, as z-scores of the noise
   corrected backbone often are), then symmetric logarithmic (matplotlib's ``symlog``: linear
   within ``linthresh`` of 0, logarithmic beyond, so that negative thresholds work too), with
   ``linthresh`` the median absolute threshold.

   :return: ``{"kind": "linear"}`` or ``{"kind": "symlog", "linthresh": c}``.
   """
   x = np.asarray(thresholds, dtype = float)
   q1, q3 = np.quantile(x, [0.25, 0.75])
   if x.max() - x.min() <= 20 * max(q3 - q1, 1e-300):
      return {"kind": "linear"}
   a = np.abs(x[x != 0])
   return {"kind": "symlog", "linthresh": float(np.median(a)) if len(a) else 1.}


#: Share of the central range of the thresholds of the backbone plot (from the first to the last
#: percentile of the edges) beyond which the first (last) percentile is a tail left out of the plot
#: (:func:`backbone_range`).
BACKBONE_TAIL = 0.25


def backbone_range(thresholds, left = False, right = False):
   """
   Range of the threshold axis of the backbone plot. The thresholds (:func:`~graphoskop.test_bb_thresholds`
   with ``quantiles = 100``) are about the percentiles of the edge scores, so a handful of edges far
   below (above) every other one makes the first (last) step much longer than the others: if it is
   more than :data:`BACKBONE_TAIL` times the range of the other steps, it is a tail and the axis
   starts (ends) 2% of that range before the second (after the last but one) threshold, unless
   ``left`` (``right``) extends it to the first (last) threshold.

   :param thresholds: Increasing thresholds.
   :param left: Whether to show the left tail.
   :param right: Whether to show the right tail.
   :return: ``(xmin, xmax, tails)``, with ``tails`` whether each side has a tail ``[left, right]``.
   """
   x = np.asarray(thresholds, dtype = float)
   c = x[-2] - x[1] if len(x) > 3 else 0.
   tails = [bool(c > 0 and x[1] - x[0] > BACKBONE_TAIL * c), bool(c > 0 and x[-1] - x[-2] > BACKBONE_TAIL * c)]
   lo = x[0] if left or not tails[0] else x[1] - 0.02 * c
   hi = x[-1] if right or not tails[1] else x[-2] + 0.02 * c
   return float(lo), float(hi), tails


#: Options of the backbone plot that the backbone dialog can change (Advanced), with their
#: defaults: font size, axis labels (``y2label``: the components axis), scale of the threshold axis
#: (``xscale``: ``"automatic"``, see :func:`backbone_scale`, ``"linear"`` or ``"logarithmic"``, a
#: symmetric logarithm), ranges (``None``: automatic; a range of the threshold axis tests the
#: thresholds again in it, see :func:`_backbone_plot`).
BACKBONE_OPTIONS_PLOT = {"font_size": 12., "xlabel": "threshold", "ylabel": "% of the network", "y2label": "components", "xscale": "automatic", "xmin": None, "xmax": None, "ymin": None, "ymax": None}
#: Labels of the elements of the backbone plot whose colors can be changed (:data:`BACKBONE_COLORS`).
BACKBONE_LABELS = {"edges_pct": "% edges kept", "nodes_pct": "% nodes not isolated", "components": "Components", "first_isolate": "First isolate", "first_component_break": "First component break", "max_separation": "Max separation", "added": "pairs added (% of the observed edges)"}


def backbone_plot(test, found, components = False, left = False, right = False, dpi = 100, colors = None, options = None, observed = None):
   """
   Plot of the backbone dialog: for each threshold of ``test`` (:func:`~graphoskop.test_bb_thresholds`),
   the percentage of edges kept and of nodes that are not isolated, and with ``components`` the
   number of components with more than one node, on a second y axis; the thresholds of ``found``
   (:func:`~graphoskop.find_bb_thresholds`: first isolate, first component break, max separation)
   as dashed vertical lines. The threshold axis covers the range of :func:`backbone_range` (tails
   left out unless ``left``, ``right``), or the range ``xmin``, ``xmax`` of the ``options``, with
   the scale of :func:`backbone_scale` of the thresholds in it (or the one of ``xscale``). Fonts
   and ticks as in ``_figure``.

   :param colors: Colors of the elements (keys of :data:`BACKBONE_COLORS`), ``None`` for the
      defaults.
   :param options: Options (:data:`BACKBONE_OPTIONS_PLOT`), ``None`` for the defaults.
   :param observed: For the probabilities of an error correction, ``(probability, seen)``: the
      probability of every pair and whether it is an edge of the network (edge attribute
      ``observed``); the percentage of edges kept is then that of the observed edges, with the
      pairs added (not observed, above the threshold) as a percentage of the observed edges.
   :return: PNG bytes; the box of the axes in the image ``[left, top, right, bottom]`` (pixels) with
      the limits of the x axis ``[xmin, xmax]``, and the scale of the x axis, for the interface to
      draw the chosen threshold; and the ``tails`` of :func:`backbone_range`.
   """
   import io
   import matplotlib
   matplotlib.use("Agg")
   from matplotlib.figure import Figure
   colors = dict(BACKBONE_COLORS, **(colors or {}))
   o = dict(BACKBONE_OPTIONS_PLOT, **(options or {}))
   with matplotlib.rc_context({"font.family": _plot_font(), "font.size": float(o["font_size"])}):
      fig = Figure(figsize = (7.6, 4.2))
      ax = fig.add_subplot()
      x = test["threshold"].to_numpy()
      lo, hi, tails = backbone_range(x, left, right)
      lo = lo if o["xmin"] is None else float(o["xmin"])
      hi = hi if o["xmax"] is None else float(o["xmax"])
      scale = backbone_scale(x[(x >= lo) & (x <= hi)]) if o["xscale"] == "automatic" else {"kind": "linear"}
      if o["xscale"] == "logarithmic":
         a = np.abs(x[x != 0])
         scale = {"kind": "symlog", "linthresh": float(np.median(a)) if len(a) else 1.}
      if scale["kind"] == "symlog":
         ax.set_xscale("symlog", linthresh = scale["linthresh"])
      if observed is None:
         ax.plot(x, test["edges_pct"], color = colors["edges_pct"], lw = 2, label = BACKBONE_LABELS["edges_pct"])
      else:   # error correction: observed edges kept, pairs added
         p, seen = observed
         n = max(int(seen.sum()), 1)
         ax.plot(x, [100 * np.count_nonzero(p[seen] > np.float32(t)) / n for t in x], color = colors["edges_pct"], lw = 2, label = "% observed edges kept")
         ax.plot(x, [100 * np.count_nonzero(p[~seen] > np.float32(t)) / n for t in x], color = colors["added"], lw = 2, label = BACKBONE_LABELS["added"])
      ax.plot(x, test["nodes_pct"], color = colors["nodes_pct"], lw = 2, label = BACKBONE_LABELS["nodes_pct"])
      for name, thr in zip(found["criterion"], found["threshold"]):
         if name in colors and np.isfinite(thr) and lo <= thr <= hi:
            ax.axvline(thr, color = colors[name], ls = "--", lw = 1.5, label = name.replace("_", " "))
      ax.set_xlim(lo, hi)
      ax.set_ylim(0 if o["ymin"] is None else float(o["ymin"]), 102 if o["ymax"] is None else float(o["ymax"]))
      ax.set_xlabel(o["xlabel"])
      ax.set_ylabel(o["ylabel"])
      ax.tick_params(which = "both", direction = "in", top = True, right = not components)
      handles, labels = ax.get_legend_handles_labels()
      if components:
         ax2 = ax.twinx()
         line, = ax2.plot(x, test["components"], color = colors["components"], lw = 2, label = "components")
         ax2.set_ylabel(o["y2label"])
         ax2.set_ylim(bottom = 0)
         ax2.tick_params(which = "both", direction = "in")
         handles.append(line)
         labels.append("components")
      ax.legend(handles, labels, frameon = False, fontsize = 0.85 * float(o["font_size"]), loc = "best")
      fig.tight_layout()
      out = io.BytesIO()
      fig.savefig(out, format = "png", dpi = dpi)
      b = ax.get_window_extent()
      h = fig.get_figheight() * dpi
   return out.getvalue(), [float(b.x0), float(h - b.y1), float(b.x1), float(h - b.y0), lo, hi], scale, tails


def _backbone(request):
   """
   Command ``backbone``: scores the edges of a tab with a function of :data:`BACKBONES`
   (``function``, ``arguments``; :func:`backbone_code`), then tests the thresholds from keeping
   every edge to keeping none, in :data:`BACKBONE_STEPS` steps
   (:func:`~graphoskop.test_bb_thresholds` with ``quantiles``) and finds the remarkable ones
   (:func:`~graphoskop.find_bb_thresholds`), with the code ``bb_test = ...`` and ``bb_found = ...``.
   Replies with the code, the drawing, and ``backbone``: the score, the ``range`` of the
   thresholds, the ``found`` thresholds (criterion, threshold), the ``box`` of the axes of the plot,
   the ``scale`` of its threshold axis and its ``tails`` (:func:`backbone_plot`, tails left out; its
   PNG is the last part of the payload).
   """
   ns = _spaces[request["tab"]]
   function = request["function"]
   if function not in BACKBONES:
      raise ValueError(f"unknown backbone function: {function}")
   lines, score = backbone_code(ns, function, request.get("arguments") or {})
   t = _target(ns)
   code = _run(request["tab"], lines + [f"bb_test = gsk.test_bb_thresholds({t}, quantiles = {BACKBONE_STEPS}, score = {literal(score)})", f"bb_found = gsk.find_bb_thresholds({t}, score = {literal(score)})"])
   x = ns["bb_test"]["threshold"].to_numpy()
   ns["__backbone__"] = {"score": score, "test": ns["bb_test"], "found": ns["bb_found"]}
   header, payload = _draw(ns, code, True)
   png, box, scale, tails = backbone_plot(ns["bb_test"], ns["bb_found"])
   found = [[name, float(thr) if np.isfinite(thr) else None] for name, thr in zip(ns["bb_found"]["criterion"], ns["bb_found"]["threshold"])]
   header["backbone"] = {"score": score, "range": [float(x[0]), float(x[-1])], "found": found, "box": box, "scale": scale, "tails": tails}
   if header["nodes"]:
      header["parts"].append(len(png))
   else:
      header["parts"] = [len(png)]
   return header, payload + png


def _backbone_plot(request):
   """
   Command ``backbone_plot``: the plot of the last backbone of a tab again, with or without the
   ``components`` and the ``left`` and ``right`` tails, with the ``colors`` and ``options`` of the
   dialog (:func:`backbone_plot`). A range of the threshold axis (``xmin``, ``xmax``) tests
   :data:`BACKBONE_STEPS` evenly spaced thresholds in it
   (:func:`~graphoskop.test_bb_thresholds` with a step, kept for the next changes; the code does
   not change), so that the backbones in it are shown in detail. Replies with the ``box`` of its
   axes, the ``scale`` of its x axis, and its PNG.
   """
   ns = _spaces[request["tab"]]
   b = ns["__backbone__"]
   colors = {k: str(v) for k, v in (request.get("colors") or {}).items() if k in BACKBONE_COLORS}
   o = request.get("options") or {}
   options = {k: o[k] for k in BACKBONE_OPTIONS_PLOT if k in o}
   for k in ("xmin", "xmax", "ymin", "ymax"):
      if k in options and options[k] is not None:
         options[k] = float(options[k])
   if "font_size" in options:
      options["font_size"] = min(max(float(options["font_size"]), 4.), 72.)
   if options.get("xscale") not in (None, "automatic", "linear", "logarithmic"):
      raise ValueError(f"unknown scale of the threshold axis: {options['xscale']}")
   test = b["test"]
   if options.get("xmin") is not None or options.get("xmax") is not None:   # a range of its own: its thresholds tested again, evenly spaced
      x = b["test"]["threshold"].to_numpy()
      lo = options["xmin"] if options.get("xmin") is not None else float(x[0])
      hi = options["xmax"] if options.get("xmax") is not None else float(x[-1])
      if not hi > lo:
         raise ValueError("the threshold axis must go from a smaller value to a larger one")
      key = (lo, hi)
      if b.get("zoom", (None, None))[0] != key:
         b["zoom"] = (key, gsk.test_bb_thresholds(_drawn(ns), lo, hi, (hi - lo) / BACKBONE_STEPS, score = b["score"]))
      test = b["zoom"][1]
   png, box, scale, _ = backbone_plot(test, b["found"], bool(request.get("components")), bool(request.get("left")), bool(request.get("right")), colors = colors, options = options, observed = _observed(ns))
   return {"box": box, "scale": scale}, png


def _observed(ns):
   """
   For the backbone dialog on the probabilities of an error correction (score ``probability`` and
   edge attribute ``observed``), the probabilities of the pairs (float32) and whether each is an
   observed edge; ``None`` otherwise.
   """
   net = _drawn(ns)
   if ns.get("__backbone__", {}).get("score") != "probability" or "observed" not in net.edge_attributes:
      return None
   return np.asarray(edge_values(net, "probability"), dtype = np.float32), np.asarray(edge_values(net, "observed")) == 1


def _backbone_stats(request):
   """
   Command ``backbone_stats``: the backbone of a tab at ``threshold`` (one row of
   :func:`~graphoskop.test_bb_thresholds`): ``nodes``, ``nodes_pct``, ``edges``, ``edges_pct``,
   ``components``; for the probabilities of an error correction (edge attribute ``observed``),
   also the number of ``observed`` edges, how many of them are kept (``observed_kept``) and how
   many new edges are ``added``.
   """
   ns = _spaces[request["tab"]]
   b = ns["__backbone__"]
   thr = float(request["threshold"])
   net = _drawn(ns)
   row = gsk.test_bb_thresholds(net, thr, thr, 1., score = b["score"]).iloc[0]
   reply = {k: float(row[k]) for k in ("nodes", "nodes_pct", "edges", "edges_pct", "components")}
   observed = _observed(ns)
   if observed is not None:   # error correction: observed edges kept or removed, new ones added
      kept = observed[0] > np.float32(thr)   # as the library compares them
      seen = observed[1]
      reply.update(observed = int(seen.sum()), observed_kept = int((kept & seen).sum()), added = int((kept & ~seen).sum()))
   return reply, b""


def _extract(request):
   """
   Command ``extract``: the backbone or the bipartite projection of a tab in a new tab ``new_tab``
   (:func:`_derived`): with ``threshold``, :func:`~graphoskop.extract_backbone` at that threshold
   of the last backbone scores of the tab, with the switches of :data:`BACKBONE_OPTIONS` that are
   on in ``options`` (e.g. ``net = gsk.extract_backbone(net, 1.64, score = "nc_score",
   reconnect_components = True)``); with ``function`` (one of :data:`FILTERS` or :data:`PROJECTIONS`) and
   ``arguments``, that function (e.g. ``net = gsk.spanning_tree(net)``,
   ``net = gsk.simple_count_projection(net, node_type = "user", weight = None)``). The projection
   of the incidence network of a hypergraph or complex is a network: ``net``, without ``inc``.
   The quotient (:func:`~graphoskop.quotient_network`) collapses the nodes of every combination of
   values of the chosen node attributes (:func:`groups`), counts the nodes and edges merged
   (``counts = "count"``, or the first free name ``count_2``, ...) and keeps only the chosen
   qualitative attributes (``qualitative``).
   """
   ns = _spaces[request["tab"]]
   t = _target(ns)
   if "threshold" in request:
      thr = float(f"{float(request['threshold']):.6g}")
      options = request.get("options") or {}
      extra = "".join(f", {k} = True" for k in BACKBONE_OPTIONS if options.get(k))
      lines = [f"{t} = gsk.extract_backbone({t}, {thr!r}, score = {literal(ns['__backbone__']['score'])}{extra})"]
   else:
      function = request.get("function")
      if function not in FILTERS | set(PROJECTIONS) | set(TRANSFORMATIONS):
         raise ValueError(f"unknown backbone filter, projection or transformation: {function}")
      arguments = request.get("arguments") or {}
      for k, v in arguments.items():
         try:
            compile(v, k, "eval")
         except SyntaxError:
            raise ValueError(f"{k}: not a Python value: {v}") from None
      if function == "quotient_network":   # the sets: the groups of node attributes; their counts; only their qualitative attributes
         if not arguments.get("sets", "").strip():
            raise ValueError("sets: choose the node attributes whose combinations of values are the sets")
         chosen = _source_value(ns, arguments["sets"])
         chosen = [chosen] if isinstance(chosen, str) else list(chosen)
         if not chosen:
            raise ValueError("sets: no node attribute matches the patterns")
         net = ns["net"]
         names = set(net.node_attributes) | set(net.qualitative_node_attributes) | set(net.edge_attributes) | set(net.qualitative_edge_attributes)
         counts = next(c for c in ["count"] + [f"count_{k}" for k in range(2, 1000)] if c not in names)
         keep = [a for a in chosen if a in net.qualitative_node_attributes]
         sets = arguments["sets"] if arguments["sets"].lstrip().startswith("matching(") else literal(chosen[0]) if len(chosen) == 1 else literal(chosen)
         arguments = dict(arguments, sets = f"groups(net, {sets})", counts = literal(counts), qualitative = literal(keep))
      if function in SET_OPERATIONS:   # with the network of another tab, made again by its code
         if request.get("other") not in _spaces:
            raise ValueError("choose the other network: an open tab")
         define, make = other_code(_spaces[request["other"]], request.get("other_name", "the other tab"))
         lines = define + [make, f"net = gsk.{function}(net, other)"]
         return _derived(ns, request["new_tab"], lines, preset = {"other": _spaces[request["other"]]["net"]}, unrun = (make,))
      if function == "layer_network" and not arguments.get("layer", "").strip() and ns["net"].layer_labels:   # the default of the dialog: the first layer
         arguments = dict(arguments, layer = literal(ns["net"].layer_labels[0]))
      extra = "".join(f", {k} = {v.strip()}" for k, v in arguments.items())
      if function in TRANSFORMATIONS:   # of the network itself (not of the incidence network drawn)
         note = [GROUPS_NOTE] if function == "quotient_network" and GROUPS_NOTE not in ns["__code__"] else []
         lines = note + [f"net = gsk.{function}(net{extra})"] + (["del inc"] if t == "inc" else [])
      elif function in PROJECTIONS:
         lines = [f"net = gsk.{function}({t}{extra})"] + (["del inc"] if t == "inc" else [])
      else:
         lines = [f"{t} = gsk.{function}({t}{extra})"]
   return _derived(ns, request["new_tab"], lines)


#: Functions of Edit › Transform combining the network with the network of another tab
#: (:func:`other_code`).
SET_OPERATIONS = {"network_union", "network_intersection", "network_difference"}


def other_code(other, name):
   """
   Code making the network of another tab, so that the code of a tab combining the two runs on
   its own: the code of the other tab as a function, and the line calling it, e.g.

   .. code-block:: python

      def other_network():   # the network of the tab "friends", made again by its code
         import graphoskop as gsk
         net = gsk.load_network("friends.tsv")
         return net
      other = other_network()

   :param other: Namespace of the other tab.
   :param name: Name of the other tab.
   :return: The lines of the function, and the line calling it.
   """
   body = [("   " + l) if l.strip() else "" for l in other["__code__"]]
   return [f"def other_network():   # the network of the tab {literal(name)}, made again by its code"] + body + ["   return net"], "other = other_network()"


#: Switches of :func:`~graphoskop.extract_backbone` offered by the backbone dialog: keep the nodes
#: left without edges, reconnect the components split by the threshold (which reconnects the
#: isolated nodes too), reconnect the isolated nodes.
BACKBONE_OPTIONS = ["keep_isolates", "reconnect_components", "reconnect_isolates"]


#: Functions of Edit › Transform: transformations of the network of the library, whose result opens
#: in a new tab (:func:`_extract`), e.g. ``net = gsk.to_undirected(net)``.
TRANSFORMATIONS = ["flip_edge_direction", "largest_component", "remove_isolates", "remove_parallel_edges", "remove_self_loops", "to_strongly_connected", "to_undirected", "network_condensation", "quotient_network", "line_graph", "complement", "network_union", "network_intersection", "network_difference", "clique_expansion", "clique_promotion", "hypergraph_walk_network", "layer_network", "aggregate_network"]
OPTIONS.update({("hypergraph_walk_network", "model"): ["\"zhou\"", "\"carletti\"", "\"edvw\""], ("node_edge_centrality", "model"): ["\"linear\"", "\"log-exp\"", "\"max\""], ("two_node_clustering", "kind"): ["\"union\"", "\"min\"", "\"max\""], ("hyperedge_size", "part"): ["\"all\"", "\"tail\"", "\"head\""]})
CHOICES.update({("node_edge_centrality", "weight"): ("nodes", False), ("node_edge_centrality", "node_weight"): ("nodes", False), ("hypergraph_walk_network", "weight"): ("nodes", False), ("hypergraph_walk_network", "vertex_weight"): ("edges", False), ("hyperedge_size", "weight"): ("edges", False)})
HINTS.update({("node_edge_centrality", "weight"): "unweighted", ("node_edge_centrality", "node_weight"): "unweighted", ("hypergraph_walk_network", "vertex_weight"): "none", ("hyperedge_size", "weight"): "unweighted"})
#: Parameters of the transformations not shown in the parameter dialog: the network, the edges to
#: flip (all), the sign attribute (``"sign"``), the size of the batches (memory, not results), and
#: the counts and qualitative attributes of the quotient (set by :func:`_extract`), and the other
#: network of the set operations (another tab, :data:`SET_OPERATIONS`).
TRANSFORMATION_HIDDEN = {"net", "edges", "sign", "batch_size", "counts", "qualitative", "other"}
#: Descriptions of parameters that the interface sets differently from the library (by function
#: and name), shown as tooltips instead of those of the documentation.
DOCS = {("quotient_network", "sets"): "Node attributes: the nodes with the same combination of their values collapse into one node (nodes without a value stay as they are); only these qualitative attributes are kept, and nodes and edges get the count of what they merge"}
#: Comment of the code before the first :func:`groups`, a function of the interface.
GROUPS_NOTE = "# groups: sets of nodes with the same value of a node attribute, a function of the graphical interface"


def groups(net, attributes):
   """
   Sets of nodes with the same values of node attributes, for
   :func:`~graphoskop.quotient_network` (Edit › Transform › Quotient): one list of node ids per
   combination of values (of one attribute, or of a list of them), nodes without a value of any
   of them left out (they stay as they are). A function of the interface, defined in the namespace
   of every tab (:data:`GROUPS_NOTE`).

   :param net: A :class:`~graphoskop.Network`.
   :param attributes: Name of a node attribute, or list of names.
   :return: List of lists of node ids.
   """
   import pandas as pd
   names = [attributes] if isinstance(attributes, str) else list(attributes)
   table = pd.DataFrame({k: pd.Series(node_values(net, a)).astype(object) for k, a in enumerate(names)})
   keep = table.notna().all(axis = 1).to_numpy()
   ids = _ids(net)[keep]
   order = table[keep].groupby(list(table.columns), sort = False).ngroup().to_numpy() if keep.any() else np.zeros(0, dtype = np.int64)
   split = np.argsort(order, kind = "stable")
   bounds = np.searchsorted(order[split], np.arange(1, order.max() + 1 if len(order) else 0))
   return [x.tolist() for x in np.split(ids[split], bounds)] if len(order) else []


#: Functions of Edit › Bipartite projection: the projections of the library (their projection
#: opens in a new tab).
PROJECTIONS = ["simple_count_projection", "hist_intersection_projection", "hyperbolic_projection", "resource_allocation_projection", "random_walk_projection", "cosine_similarity_projection", "pearson_correlation_projection", "inverse_euclidean_projection"]
#: Parameters of the projections not shown in the parameter dialog: the network, and the size of
#: the batches (memory, not results).
PROJECTION_HIDDEN = {"net", "batch_size"}


#: Functions of Simulation › Epidemics: the epidemic models of the library.
EPIDEMICS = ["si_model", "sis_model", "sir_model", "complex_si_model", "complex_sis_model", "complex_sir_model", "complex_nonprogressive_model", "higher_order_si_model", "higher_order_sis_model", "higher_order_sir_model"]
#: Parameters of the epidemic models not shown in the parameter dialog: the network, the component
#: (the largest, the default of the library) and the contagion forest (a second result).
EPIDEMIC_HIDDEN = {"net", "component", "forest", "progress"}


def simulation_code(ns, function, arguments):
   """
   Code running an epidemic model of :data:`EPIDEMICS` on the network of a tab and storing its
   table of states (mean fraction of the nodes in each state at each step, with confidence
   intervals) in the dictionary ``epidemics``, by run name: e.g.
   ``epidemics["SIR 1"] = gsk.sir_model(net, beta = 0.1, mu = 0.05)``.

   :param ns: Namespace of the tab.
   :param function: Name of the model.
   :param arguments: Parameter name to Python source code, with ``name`` (the name of the run).
   :return: List of lines of code, and the name of the run.
   :raises ValueError: If the run has no name, or a value is not a Python expression.
   """
   arguments = dict(arguments)
   name = ast.literal_eval(arguments.pop("name", '""'))
   if not name:
      raise ValueError("name: the run needs a name")
   for k, v in arguments.items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   extra = "".join(f", {k} = {v.strip()}" for k, v in arguments.items())
   lines = [] if "epidemics" in ns else ["epidemics = {}"]
   return lines + [f"epidemics[{literal(name)}] = gsk.{function}(net{extra})"], name


def _simulate(request):
   """
   Command ``simulate``: runs the code of :func:`simulation_code` in the namespace of a tab, and
   replies with the code, the runs (:func:`_runs`: ``tables``, ``states``, ``models``) and the
   drawing (the states become a node attribute; the look of the drawing does not change).
   """
   ns = _spaces[request["tab"]]
   if request["function"] not in EPIDEMICS:
      raise ValueError(f"unknown epidemic model: {request['function']}")
   lines, name = simulation_code(ns, request["function"], request.get("arguments") or {})
   code = _run(request["tab"], lines)
   ns["__runs__"] = ns.get("__runs__", 0) + 1
   ns.setdefault("__models__", {})[name] = {"function": request["function"], "arguments": dict(request.get("arguments") or {})}
   if (request.get("arguments") or {}).get("history") == "True":
      ns["__history_run__"] = name
   header, payload = _draw(ns, code, True)
   header.update(_runs(ns))
   return header, payload


def animation(frames, path, fps = 5., labels = None):
   """
   An animation from images of the visualization panel (e.g. the steps of the spread of an
   epidemic, from the player of the interface): a GIF (with Pillow) or an MP4 video (H.264, with
   ffmpeg: on the ``PATH``, or from the ``imageio-ffmpeg`` package), by the extension of ``path``;
   with ``labels``, each frame gets its text in the top left corner (e.g. ``"step 3 / 40"``).

   :param frames: Paths of the PNG images, in order.
   :param path: File to write (``.gif`` or ``.mp4``).
   :param fps: Frames per second.
   :param labels: Text of every frame, or ``None``.
   :raises ValueError: If the format is not known, or MP4 is asked for without ffmpeg.
   """
   import shutil
   import subprocess
   import tempfile
   from PIL import Image, ImageDraw, ImageFont
   kind = os.path.splitext(path)[1].lower()
   if kind not in (".gif", ".mp4"):
      raise ValueError(f"animations are .gif or .mp4 files, not {kind or 'files without extension'}")
   def image(k):
      im = Image.open(frames[k]).convert("RGB")
      if labels:
         draw = ImageDraw.Draw(im)
         try:
            font = ImageFont.load_default(size = max(12, im.height // 30))
         except TypeError:   # Pillow before 10.1
            font = ImageFont.load_default()
         draw.text((10, 8), labels[k], fill = (0, 0, 0), font = font, stroke_width = 2, stroke_fill = (255, 255, 255))
      return im
   if kind == ".gif":
      first = image(0)
      first.save(path, save_all = True, append_images = (image(k) for k in range(1, len(frames))), duration = int(round(1000 / fps)), loop = 0, optimize = False)
      return
   ffmpeg = shutil.which("ffmpeg")
   if ffmpeg is None:
      try:
         import imageio_ffmpeg
         ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
      except ImportError:
         raise ValueError("MP4 videos need ffmpeg (install it, or pip install imageio-ffmpeg); GIF animations need nothing more") from None
   with tempfile.TemporaryDirectory() as folder:
      for k in range(len(frames)):
         image(k).save(os.path.join(folder, f"frame_{k:06d}.png"))
      done = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(folder, "frame_%06d.png"), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", path], capture_output = True, text = True)
   if done.returncode:
      raise ValueError(f"ffmpeg failed: {done.stderr.strip()[-500:]}")


def _animation(request):
   """
   Command ``animation``: :func:`animation` of the PNG ``frames`` written by the interface (deleted
   afterwards), to ``path``, at ``fps``, with the ``labels`` of the frames.
   """
   try:
      animation(request["frames"], request["path"], float(request.get("fps", 5)), request.get("labels"))
   finally:
      for f in request["frames"]:
         try:
            os.remove(f)
         except OSError:
            pass
   return {"path": request["path"]}, b""


def _simulations(request):
   """
   Command ``simulations``: the names of the tables of the Tables tab of a tab (``tables``: the
   runs of epidemic models, then the tables of ``tables``, e.g. of frequent subgraphs, see
   :func:`tab_tables`) and the states of the runs (``states``: the columns without confidence
   intervals, e.g. ``["S", "I", "R"]``).
   """
   ns = _spaces[request["tab"]]
   reply = _runs(ns)
   reply["tables"] = list(tab_tables(ns))
   return reply, b""


def tab_tables(ns):
   """
   The tables of the Tables tab of a tab: the runs of epidemic models (``epidemics``), then the
   :class:`pandas.DataFrame` values of the dictionary ``tables`` of its code (e.g. the support of
   frequent patterns, :func:`mining_code`).
   """
   import pandas as pd
   found = dict(ns.get("epidemics") or {})
   other = ns.get("tables")
   if isinstance(other, dict):
      found.update({str(k): v for k, v in other.items() if isinstance(v, pd.DataFrame) and str(k) not in found})
   return found


def _runs(ns):
   """
   The runs of the epidemic models of a tab: their names (``tables``), the states of each
   (``states``: the columns without confidence intervals, e.g. ``["S", "I", "R"]``) and, for the
   runs made in the interface, their model and arguments (``models``: name to ``{"function",
   "arguments"}``, to run them again with other values).
   """
   runs = ns.get("epidemics", {})
   models = ns.get("__models__", {})
   return {"tables": list(runs), "states": {k: [c for c in df.columns if not c.endswith(("_low", "_high"))] for k, df in runs.items()}, "models": {k: models[k] for k in runs if k in models}}


def epidemic_plot(tables, states = ("I",), colors = None, options = None, styles = None, format = "png", path = None, dpi = 100):
   """
   Plot comparing runs of epidemic models: the mean fraction of the nodes in each of ``states`` at
   each step, one series per run and state (key ``"run, state"``), with its confidence interval as
   a band (linear axes by default; see ``_figure``).

   :param tables: Dictionary from run name to its table (see :func:`simulation_code`).
   :param states: The states plotted (``"I"`` for infected, ``"S"``, ``"R"``).
   :param colors: Series key to color, or ``None`` (palette of the plot window).
   :param options: Options of the plot (see :data:`PLOT_OPTIONS`), or ``None`` for the defaults.
   :param styles: Series key to one of :data:`PLOT_STYLES` (by default lines).
   :param format: ``"png"`` or ``"pdf"``.
   :param path: File to write, or ``None`` to return the bytes.
   :param dpi: Resolution of PNG images.
   :return: The bytes of the image, or ``None`` with ``path``.
   """
   if isinstance(states, str):
      states = [states]
   def draw(ax):
      k = 0
      for name, df in tables.items():
         for state in states:
            if state not in df:
               continue
            key = f"{name}, {state}"
            c = (colors or {}).get(key, PLOT_COLORS[k % len(PLOT_COLORS)])
            k += 1
            _series(ax, df.index, df[state], (styles or {}).get(key, "lines"), c, key if len(states) > 1 else name)
            if f"{state}_low" in df and f"{state}_high" in df:
               ax.fill_between(df.index, df[f"{state}_low"], df[f"{state}_high"], color = c, alpha = 0.2, lw = 0)
   options = dict(PLOT_OPTIONS["epidemics"], **(options or {}))
   if options["ylabel"] is None:
      options["ylabel"] = f"fraction of nodes in state {states[0]}" if len(states) == 1 else "fraction of nodes"
   return _figure(draw, options, format, path, dpi)


#: Default colors of the lines of plots comparing several things (ColorBrewer's Set1).
PLOT_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]


def _epidemic_plot(request):
   """
   Command ``epidemic_plot``: plots the runs ``names`` of a tab in the ``states`` (a list; or one
   ``state``) (:func:`epidemic_plot`), and replies with the PNG image as payload, ``elements`` (one
   color and style per run and state) and ``text``: the peak of every series (largest fraction
   and its step).
   """
   ns = _spaces[request["tab"]]
   runs = ns.get("epidemics", {})
   names = [n for n in request.get("names") or [] if n in runs]
   if not names:
      raise ValueError("choose the runs to plot")
   states = request.get("states") or [request.get("state", "I")]
   tables = {n: runs[n] for n in names}
   keys = [f"{n}, {x}" for n, df in tables.items() for x in states if x in df]
   if not keys:
      raise ValueError("choose a state of the runs")
   ylabel = f"fraction of nodes in state {states[0]}" if len(states) == 1 else "fraction of nodes"
   ns["__plot__"] = _plot("epidemics", {"tables": tables, "states": list(states)}, {k: PLOT_COLORS[i % len(PLOT_COLORS)] for i, k in enumerate(keys)}, {k: "lines" for k in keys}, {k: k for k in keys}, source = {"tables": "{" + ", ".join(f"{literal(n)}: epidemics[{literal(n)}]" for n in names) + "}", "states": repr(list(states))}, ylabel = ylabel)
   peaks = [f"{n}, {x}: peak {df[x].max():.3g} at step {df[x].idxmax()}" for n, df in tables.items() for x in states if x in df]
   header = dict(_plot_header(ns["__plot__"]), text = f"Fraction of nodes in state {', '.join(states)}. " + "; ".join(peaks) + ".", warnings = [])
   return header, _render(ns["__plot__"], "png")


#: Link predictions of the link prediction dialog (Analysis › Link prediction…): function and the
#: label of its runs. Every score is computed for the pairs of one validation set
#: (:func:`link_split_code`) from its training network, then evaluated
#: (:func:`link_prediction_code`).
LINK_PREDICTIONS = {"preferential_attachment_lp": "Preferential attachment", "common_neighbors_lp": "Common neighbors", "common_neighbors_jaccard_lp": "Jaccard", "adamic_adar_lp": "Adamic-Adar", "resource_allocation_lp": "Resource allocation", "car_index_lp": "CAR index", "mutual_information_index_lp": "Mutual information", "local_random_walk_lp": "Local random walk", "simrank_lp": "SimRank", "vertex_similarity_lp": "Vertex similarity", "hierarchical_random_graph_lp": "Hierarchical random graph", "block_model_reliability_lp": "Block model", "embedding_lp": "Embedding", "association_rules_lp": "Association rules", "ensemble_link_prediction": "Ensemble"}

#: Parameters of the link predictions not in their dialog: the networks and pairs (the training
#: network and the validation set of the dialog), the name of the score column (the name of the
#: run), what the interface computes itself (embeddings, mined patterns and rules, the runs an
#: ensemble aggregates) and tolerances.
LINK_HIDDEN = {"net", "pairs", "name", "attributes", "patterns", "rules", "methods", "columns", "new_nodes", "tol", "max_iter"}

#: Embeddings of :func:`~graphoskop.embedding_lp`, computed on the training network (the
#: ``embedding`` of its dialog), with their number of ``dimensions``.
LINK_EMBEDDINGS = ["node2vec", "laplacian_eigenmaps", "locally_linear_embedding", "struc2vec"]

#: Default number of dimensions of the embeddings of :func:`~graphoskop.embedding_lp`, least
#: support (MNI) and largest number of edges of the patterns mined for
#: :func:`~graphoskop.association_rules_lp`.
LINK_DIMENSIONS = 16
LINK_SUPPORT = 10
LINK_PATTERN_EDGES = 3

OPTIONS.update({("embedding_lp", "embedding"): [literal(e) for e in LINK_EMBEDDINGS], ("embedding_lp", "operator"): ["\"hadamard\"", "\"average\"", "\"weighted_l1\"", "\"weighted_l2\""], ("ensemble_link_prediction", "aggregation"): ["\"borda\"", "\"kendall\""], ("association_rules_lp", "measure"): ["\"confidence\"", "\"lift\"", "\"support\""], ("association_rules_lp", "combine"): ["\"max\"", "\"sum\""]})
CHOICES.update({("association_rules_lp", "node_label"): ("nodes", False), ("association_rules_lp", "edge_label"): ("edges", False)})
RANGES.update({("link_prediction_split", "fraction"): (0.01, 0.99, 0.01), ("simrank_lp", "decay"): (0., 1., 0.01), ("vertex_similarity_lp", "phi"): (0., 1., 0.01)})
INTEGERS.update({("link_prediction_split", "negatives"): 1, ("local_random_walk_lp", "steps"): 1, ("simrank_lp", "walks"): 1, ("simrank_lp", "length"): 1, ("embedding_lp", "dimensions"): 1, ("embedding_lp", "negatives"): 1, ("embedding_lp", "samples"): 1, ("association_rules_lp", "min_support"): 1, ("association_rules_lp", "max_edges"): 1})
NUMBERS.update({("embedding_lp", "ridge"): (0., 0.1)})
HINTS.update({("hierarchical_random_graph_lp", "chains"): "one per thread", ("block_model_reliability_lp", "chains"): "one per thread", ("block_model_reliability_lp", "groups"): "automatic", ("association_rules_lp", "node_label"): "no labels", ("association_rules_lp", "edge_label"): "no labels"})


def link_signature(function, net, runs = None):
   """
   Parameters of the dialog of a link prediction (:data:`LINK_PREDICTIONS`), as :func:`signature`
   (without :data:`LINK_HIDDEN`), after the ``name`` of its run; the embedding of
   :func:`~graphoskop.embedding_lp` (``embedding``, ``dimensions``), the mining of
   :func:`~graphoskop.association_rules_lp` (``min_support``, ``max_edges``), and the ``runs``
   aggregated by :func:`~graphoskop.ensemble_link_prediction` (check boxes; none: all). For
   ``"link_prediction_split"``, the parameters of the validation set.
   """
   sig = signature(function, net)
   params = [p for p in sig["params"] if p["name"] not in LINK_HIDDEN | {"sign"}]
   if function == "link_prediction_split":
      for p in params:
         if p["name"] == "seed":
            p["default"] = "1"
      return {"params": params}
   if function == "embedding_lp":
      params[:0] = [{"name": "embedding", "default": literal(LINK_EMBEDDINGS[0]), "hint": "", "doc": "Embedding of the nodes, computed on the training network: its vectors are the node features of the regression", "widget": {"kind": "choice", "options": OPTIONS[("embedding_lp", "embedding")]}},
                    {"name": "dimensions", "default": str(LINK_DIMENSIONS), "hint": "", "doc": "Number of dimensions of the embedding", "widget": {"kind": "int", "min": 1}}]
   if function == "association_rules_lp":
      params[:0] = [{"name": "min_support", "default": str(LINK_SUPPORT), "hint": "", "doc": "Least support (MNI) of the patterns mined in the training network (gsk.single_graph_mining)", "widget": {"kind": "int", "min": 1}},
                    {"name": "max_edges", "default": str(LINK_PATTERN_EDGES), "hint": "", "doc": "Largest number of edges of the patterns mined", "widget": {"kind": "int", "min": 1}}]
   if function == "ensemble_link_prediction":
      params.insert(0, {"name": "runs", "default": "", "hint": "all the runs", "doc": "The runs whose rankings are aggregated", "widget": {"kind": "attribute", "choices": "runs", "multiple": True, "extra": []}})
   params.insert(0, {"name": "name", "default": "", "hint": "required", "doc": "Name of the run: the column of its scores in the validation set, and its key in link_predictions", "widget": {"kind": "string"}})
   return {"params": params}


def link_split_code(arguments):
   """
   Code making the training network and the validation set of the link prediction dialog of a
   tab (:func:`~graphoskop.link_prediction_split`, by default with ``seed = 1``, so that running
   the code again gives the same pairs), and emptying its runs, e.g. ``train, validation =
   gsk.link_prediction_split(net, fraction = 0.2, seed = 1)`` and ``link_predictions = {}``.
   """
   arguments = dict({"seed": "1"}, **{k: v for k, v in (arguments or {}).items() if str(v).strip()})
   for k, v in arguments.items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   extra = "".join(f", {k} = {v.strip()}" for k, v in arguments.items())
   return [f"train, validation = gsk.link_prediction_split(net{extra})", "link_predictions = {}"]


def link_prediction_code(ns, function, arguments):
   """
   Code of a run of the link prediction dialog: the score ``function`` of :data:`LINK_PREDICTIONS`
   computed for the validation set from the training network, as a column named after the run,
   then its evaluation with the curves (:func:`~graphoskop.link_prediction_evaluation`), kept in
   the dictionary ``link_predictions``, e.g. ``validation = gsk.adamic_adar_lp(train, validation,
   name = "Adamic-Adar 1")`` and ``link_predictions["Adamic-Adar 1"] =
   gsk.link_prediction_evaluation(validation, score = "Adamic-Adar 1", curves = True)``. Embedding
   scores first compute the embedding on the training network (e.g. ``gsk.node2vec(train,
   dimensions = 16, attribute = "lp_node2vec")``); association rules first mine it
   (``patterns, support = gsk.single_graph_mining(train, 10, max_edges = 3)``, ``rules =
   gsk.graph_association_rules(train, patterns, support, transactional = False)``); an ensemble
   aggregates the scores of the runs chosen (all by default) from the validation set. Without a
   validation set yet, one is made first with the defaults (:func:`link_split_code`).

   :param ns: Namespace of the tab.
   :param function: Name of the function.
   :param arguments: Parameter name to Python source code, with ``name`` (of the run).
   :return: List of lines of code, and the name of the run.
   :raises ValueError: If the run has no name, an ensemble has fewer than two runs, or a value is
      not a Python expression.
   """
   if function not in LINK_PREDICTIONS:
      raise ValueError(f"unknown link prediction: {function}")
   arguments = {k: v for k, v in dict(arguments).items() if str(v).strip()}
   name = ast.literal_eval(arguments.pop("name", '""'))
   if not name:
      raise ValueError("name: the run needs a name")
   for k, v in arguments.items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   lines = [] if "validation" in ns and "link_predictions" in ns else link_split_code({})
   column = literal(name)
   if function == "embedding_lp":
      embedding = ast.literal_eval(arguments.pop("embedding", literal(LINK_EMBEDDINGS[0])))
      dimensions = int(arguments.pop("dimensions", str(LINK_DIMENSIONS)))
      prefix = f"lp_{embedding}"
      seed = f", seed = {arguments['seed']}" if "seed" in arguments and embedding in ("node2vec", "struc2vec") else ""
      lines.append(f"gsk.{embedding}(train, dimensions = {dimensions}{seed}, attribute = {literal(prefix)})")
      arguments = dict({"attributes": f"[f\"{prefix}_{{i}}\" for i in range({dimensions})]"}, **arguments)
   if function == "association_rules_lp":
      support = arguments.pop("min_support", str(LINK_SUPPORT))
      edges = arguments.pop("max_edges", str(LINK_PATTERN_EDGES))
      labels = "".join(f", {k} = {arguments[k]}" for k in ("node_label", "edge_label") if k in arguments)
      lines += [f"patterns, support = gsk.single_graph_mining(train, {support}, max_edges = {edges}{labels})", "rules = gsk.graph_association_rules(train, patterns, support, transactional = False)"]
      arguments = dict({"patterns": "patterns", "rules": "rules"}, **arguments)
   if function == "ensemble_link_prediction":
      chosen = ast.literal_eval(arguments.pop("runs", "[]")) or [n for n in ns.get("link_predictions", {})]
      if len(chosen) < 2:
         raise ValueError("an ensemble aggregates at least two runs: run other link predictions first")
      call = f"gsk.ensemble_link_prediction(None, validation, columns = {literal(chosen)}" + "".join(f", {k} = {v}" for k, v in arguments.items()) + f", name = {column})"
   else:
      call = f"gsk.{function}(train, validation" + "".join(f", {k} = {v}" for k, v in arguments.items()) + f", name = {column})"
   lines += [f"validation = {call}", f"link_predictions[{column}] = gsk.link_prediction_evaluation(validation, score = {column}, curves = True)"]
   return lines, name


def _link_runs(ns):
   """
   The runs of the link prediction dialog of a tab: ``runs``, one per evaluated score (``name``,
   ``auc``, ``average_precision``, and for the runs made in the interface their ``function`` and
   ``arguments``, for Copy…); the ``split`` (the arguments of the validation set, or ``None``
   before the first run), its numbers of ``positives`` and ``negatives``, and whether the
   network is ``signed`` (sign prediction).
   """
   results = ns.get("link_predictions", {})
   models = ns.get("__link_models__", {})
   runs = [dict({"name": n, "auc": float(r["auc"]), "average_precision": float(r["average_precision"])}, **models.get(n, {})) for n, r in results.items()]
   v = ns.get("validation")
   info = {"runs": runs, "split": ns.get("__link_split__") if v is not None else None, "signed": "sign" in ns["net"].edge_attributes}
   if v is not None and hasattr(v, "columns") and "label" in v:
      info.update(positives = int((v["label"] == 1).sum()), negatives = int((v["label"] == 0).sum()))
   return info


def _link_predictions(request):
   """
   Command ``link_predictions``: the runs of the link prediction dialog of a tab (:func:`_link_runs`).
   """
   return _link_runs(_spaces[request["tab"]]), b""


def _link_split(request):
   """
   Command ``link_split``: a new validation set for the link prediction dialog of a tab
   (:func:`link_split_code`, ``arguments``): the runs so far are dropped. Replies with the code,
   the drawing (unchanged) and the runs.
   """
   ns = _spaces[request["tab"]]
   code = _run(request["tab"], link_split_code(request.get("arguments") or {}))
   ns["__link_split__"] = dict({"seed": "1"}, **{k: v for k, v in (request.get("arguments") or {}).items() if str(v).strip()})
   ns["__link_models__"] = {}
   header, payload = _draw(ns, code, True)
   header.update(_link_runs(ns))
   return header, payload


def _link_predict(request):
   """
   Command ``link_predict``: a run of the link prediction dialog of a tab
   (:func:`link_prediction_code`, ``function``, ``arguments``). Replies with the code, the drawing
   (unchanged) and the runs.
   """
   ns = _spaces[request["tab"]]
   lines, name = link_prediction_code(ns, request["function"], request.get("arguments") or {})
   fresh = "validation" not in ns or "link_predictions" not in ns
   code = _run(request["tab"], lines)
   if fresh:
      ns["__link_split__"] = {"seed": "1"}
      ns["__link_models__"] = {}
   ns.setdefault("__link_models__", {})[name] = {"function": request["function"], "arguments": dict(request.get("arguments") or {})}
   header, payload = _draw(ns, code, True)
   header.update(_link_runs(ns))
   return header, payload


def _link_signs(request):
   """
   Command ``link_signs``: for signed networks, how well social balance and social status predict
   the signs of the hidden edges of the validation set (:func:`~graphoskop.link_prediction_sign_evaluation`,
   kept in ``sign_prediction``); replies with the code, the drawing and ``text``.
   """
   ns = _spaces[request["tab"]]
   if "validation" not in ns or "link_predictions" not in ns:
      raise ValueError("make the validation set first (Validation set… or a first run)")
   code = _run(request["tab"], ["sign_prediction = gsk.link_prediction_sign_evaluation(train, validation)"])
   r = ns["sign_prediction"]
   pct = lambda x: "none" if x != x else f"{100 * x:.1f}%"
   header, payload = _draw(ns, code, True)
   header.update(_link_runs(ns), text = f"Signs of {r['edges']} hidden edges: social balance {pct(r['balance_accuracy'])} correct ({pct(r['balance_coverage'])} predicted), social status {pct(r['status_accuracy'])} correct ({pct(r['status_coverage'])} predicted); always the most common sign: {pct(r['majority_accuracy'])} correct.")
   return header, payload


#: Curves of :func:`link_plot`: label, columns (x, y) of the tables of
#: :func:`~graphoskop.link_prediction_evaluation`, axis labels and the measure in the legend.
LINK_CURVES = {"roc": ("ROC", "roc_curve", "fpr", "tpr", "false positive rate", "true positive rate", "auc", "AUC"), "pr": ("Precision-recall", "precision_recall_curve", "recall", "precision", "recall", "precision", "average_precision", "AP")}


def link_plot(results, names, base_rate = None, colors = None, options = None, styles = None, format = "png", path = None, dpi = 100):
   """
   Plot comparing link predictions: the ROC curves (with the diagonal of a random score) or the
   precision-recall curves (with the fraction of positives, the precision of a random score) of
   the runs ``names`` (option ``kind``: ``"roc"`` or ``"pr"``), each labeled with its area (AUC or
   average precision).

   :param results: Dictionary from run name to its evaluation (``link_predictions``).
   :param names: The runs plotted.
   :param base_rate: Fraction of positives in the validation set.
   :return: The bytes of the image, or ``None`` with ``path``.
   """
   options = dict(PLOT_OPTIONS["links"], **(options or {}))
   label, table, x, y, xlabel, ylabel, measure, short = LINK_CURVES[options.get("kind", "roc")]
   options["xlabel"] = options["xlabel"] or xlabel
   options["ylabel"] = options["ylabel"] or ylabel
   def draw(ax):
      for k, n in enumerate(names):
         df = results[n][table]
         xs, ys = df[x].to_numpy(), df[y].to_numpy()
         if table == "roc_curve" and len(xs) and xs[0] == 0 and ys[0] > 0:   # the curve starts at the origin
            xs, ys = np.r_[0., xs], np.r_[0., ys]
         _series(ax, xs, ys, (styles or {}).get(n, "lines"), (colors or {}).get(n, PLOT_COLORS[k % len(PLOT_COLORS)]), f"{n} ({short} {results[n][measure]:.3f})")
      if options.get("kind", "roc") == "roc":
         ax.plot([0, 1], [0, 1], color = "#999999", ls = "--", lw = 1)
      elif base_rate is not None:
         ax.axhline(base_rate, color = "#999999", ls = "--", lw = 1)
   return _figure(draw, options, format, path, dpi)


def link_text(results, names, kind):
   """
   Text under the plot of link predictions: the area under the curve of every run, best first.
   """
   label, _, _, _, _, _, measure, short = LINK_CURVES[kind]
   ranked = sorted(names, key = lambda n: -results[n][measure])
   return f"{label} curves, dashed: a random score. " + "; ".join(f"{n}: {short} {results[n][measure]:.4g}" for n in ranked) + "."


def _link_plot(request):
   """
   Command ``link_plot``: plots the runs ``names`` of the link prediction dialog of a tab
   (:func:`link_plot`, ``kind`` ``"roc"`` or ``"pr"``, which the plot window can switch), and
   replies with the PNG image as payload, ``elements`` (one color and style per run) and ``text``
   (:func:`link_text`).
   """
   ns = _spaces[request["tab"]]
   results = ns.get("link_predictions", {})
   names = [n for n in request.get("names") or [] if n in results]
   if not names:
      raise ValueError("choose the runs to plot")
   kind = request.get("kind", "roc") if request.get("kind") in LINK_CURVES else "roc"
   v = ns["validation"]
   ns["__plot__"] = _plot("links", {"results": results, "names": names, "base_rate": float((v["label"] == 1).mean())}, {n: PLOT_COLORS[i % len(PLOT_COLORS)] for i, n in enumerate(names)}, {n: "lines" for n in names}, {n: n for n in names}, source = {"results": "link_predictions", "names": repr(names), "base_rate": "float((validation[\"label\"] == 1).mean())"})
   ns["__plot__"]["options"]["kind"] = kind
   header = dict(_plot_header(ns["__plot__"]), warnings = [])
   return header, _render(ns["__plot__"], "png")


#: Graph neural networks of the dialog of Analysis › Graph neural networks…: function and the
#: label of its runs. Node models predict a node attribute; DiffPool a graph attribute (one value
#: per component); MLP, the same model without the network, is the baseline (its ``weight`` is not
#: asked).
GNNS = {"gcn": "GCN", "graphsage": "GraphSAGE", "gat": "GAT", "gatv2": "GATv2", "diffpool": "DiffPool", "mlp": "MLP", "hgnn": "HGNN (hypergraphs and complexes)"}

#: Parameters of the graph neural networks not in their dialog: the network, the layers (the
#: hidden layers are asked, ``hidden``, and the output layer follows from the target), their
#: activations (the defaults) and the progress bar.
GNN_HIDDEN = {"net", "channels", "activations", "progress"}

#: Default hidden layers of the node models (output channels of each).
GNN_LAYERS = "32, 16"

OPTIONS.update({(f, "task"): ["", "\"classification\"", "\"regression\""] for f in GNNS})
OPTIONS.update({(f, "optimizer"): ["\"adam\"", "\"adamw\"", "\"sgd\"", "\"rmsprop\""] for f in GNNS})
OPTIONS.update({(f, "loss"): ["", "\"cross_entropy\"", "\"nll\"", "\"mse\"", "\"l1\"", "\"huber\""] for f in GNNS})
CHOICES.update({(f, "target"): ("nodes", False) for f in GNNS})
CHOICES.update({(f, "features"): ("nodes", True) for f in GNNS})
RANGES.update({(f, "dropout"): (0., 0.95, 0.01) for f in GNNS})
NUMBERS.update({(f, "learning_rate"): (0., 0.001) for f in GNNS})
NUMBERS.update({(f, "weight_decay"): (0., 0.0001) for f in GNNS})
INTEGERS.update({(f, "epochs"): 1 for f in GNNS})
INTEGERS.update({(f, "heads"): 1 for f in GNNS})
INTEGERS.update({("diffpool", "channels"): 1})
LISTS.update({(f, "split") for f in GNNS} | {("diffpool", "clusters")})
HINTS.update({(f, "task"): "from the target" for f in GNNS})
HINTS.update({(f, "loss"): "by the task" for f in GNNS})
HINTS.update({(f, "features"): "all but the target" for f in GNNS})
HINTS.update({(f, "target"): "required" for f in GNNS})
HINTS[("diffpool", "target")] = "none (only embeddings)"


def gnn_signature(function, net):
   """
   Parameters of the dialog of a graph neural network (:data:`GNNS`), as :func:`signature`
   (without :data:`GNN_HIDDEN`), after the ``name`` of its run; the node models ask their
   ``hidden`` layers instead of all the channels, and a ``transform`` of regression targets
   (:func:`gnn_code`). The ``seed`` is 1 by default, so that runs compare on the same split of
   the nodes and repeat when the code runs again.
   """
   params = [p for p in signature(function, net)["params"] if p["name"] not in GNN_HIDDEN and not (function == "mlp" and p["name"] == "weight")]
   for p in params:
      if p["name"] == "seed":
         p["default"] = "1"
   if function != "diffpool":
      at = next(i for i, p in enumerate(params) if p["name"] == "target") + 1
      params.insert(at, {"name": "transform", "default": "", "hint": "none", "doc": "Transformation of a numeric target of a regression, trained on and scored as a new node attribute (e.g. log_income): log (nodes with values ≤ 0 are left out) or log1p, log(1 + x) (for values ≥ 0, e.g. counts)", "widget": {"kind": "choice", "options": ["", "\"log\"", "\"log1p\""]}})
      params.insert(1, {"name": "hidden", "default": GNN_LAYERS, "hint": "", "doc": "Output channels of the hidden layers, separated by commas; the output layer is added: one channel per class for classification, 1 for regression", "widget": {"kind": "list"}})
   params.insert(0, {"name": "name", "default": "", "hint": "required", "doc": "Name of the run: its result is gnn_runs[name]", "widget": {"kind": "string"}})
   return {"params": params}


def gnn_code(ns, function, arguments):
   """
   Code of a run of the graph neural network dialog: the model ``function`` of :data:`GNNS` on the
   network of a tab, its result (scores, predictions, learning curves, model) kept in the dictionary
   ``gnn_runs`` by run name, e.g. ``gnn_runs["GCN 1"] = gsk.gcn(net, "city", [32, 16, 4], seed =
   1)`` (seed 1 and the hidden layers :data:`GNN_LAYERS` unless the dialog says otherwise): the
   node models get the hidden layers of the dialog and an output layer with one channel
   per class of the target (classification: qualitative targets, or ``task = "classification"``)
   or one channel (regression).

   With ``transform`` (``"log"`` or ``"log1p"``, numeric targets of regressions), the model is
   trained on a new node attribute, e.g. ``log_income``, made first with :func:`derived`
   (``net.set_node_attribute("log_income", derived(net, "log(income)"), None)``; nodes whose
   logarithm is not defined have no target, and are left out); unless the dialog chooses the
   features, they are all the numeric attributes but the target and its transformation.

   :param ns: Namespace of the tab.
   :param function: Name of the model.
   :param arguments: Parameter name to Python source code, with ``name`` (of the run), ``target``
      and ``hidden``.
   :return: List of lines of code, and the name of the run.
   :raises ValueError: If the run has no name or no target, the target has fewer than two classes,
      or a value is not a Python expression.
   """
   if function not in GNNS:
      raise ValueError(f"unknown graph neural network: {function}")
   arguments = dict({"seed": "1"}, **{k: v for k, v in dict(arguments).items() if str(v).strip()})   # the dialog sends only what differs from its defaults
   name = ast.literal_eval(arguments.pop("name", '""'))
   if not name:
      raise ValueError("name: the run needs a name")
   for k, v in arguments.items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   net = ns["net"]
   target = arguments.pop("target", None)
   lines = [] if "gnn_runs" in ns else ["gnn_runs = {}"]
   if function == "diffpool":
      call = f"gsk.diffpool(net{', target = ' + target if target else ''}" + "".join(f", {k} = {v}" for k, v in arguments.items()) + ")"
      return lines + [f"gnn_runs[{literal(name)}] = {call}"], name
   if not target:
      raise ValueError("target: choose the node attribute to predict")
   attribute = ast.literal_eval(target)
   hidden = ast.literal_eval("[" + arguments.pop("hidden", GNN_LAYERS).strip("[]()") + "]")
   hidden = [int(hidden)] if isinstance(hidden, (int, float)) else [int(h) for h in hidden]
   task = ast.literal_eval(arguments.get("task", "None"))
   transform = ast.literal_eval(arguments.pop("transform", "None"))
   values = node_values(net, attribute)
   if transform:
      if values.dtype.kind != "f" or task == "classification" or ast.literal_eval(arguments.get("qualitative", "False")):
         raise ValueError(f"transform: only numeric targets of a regression are transformed ({attribute} is not one here)")
      new = f"{transform}_{attribute}"
      lines.append(f"net.set_node_attribute({literal(new)}, derived(net, {literal(f'{transform}({formula_name(attribute)})')}), None)   # the target, transformed")
      if "features" not in arguments:
         arguments["features"] = f"[a for a in net.node_attributes if a not in {literal([attribute, new])}]"
      target = literal(new)
   if task == "classification" or task is None and values.dtype.kind != "f":
      classes = len(_unique(values)) if values.dtype.kind != "f" else len(np.unique(values[~np.isnan(values)]))
      if classes < 2:
         raise ValueError(f"target: {attribute} has {classes} class: at least two are needed")
      out = classes
   else:
      out = 1
   call = f"gsk.{function}(net, {target}, {literal(hidden + [out])}" + "".join(f", {k} = {v}" for k, v in arguments.items()) + ")"
   return lines + [f"gnn_runs[{literal(name)}] = {call}"], name


def gnn_evaluation(result):
   """
   What the dialog and the plots show of a run (:func:`gnn_code`), on the validation set: its
   ``task``, its ``scores`` (accuracy and macro F1, or MSE, MAE and :math:`R^2`), and for
   classification the ROC and precision-recall curves with their areas
   (:func:`~graphoskop.link_prediction_evaluation`), one-vs-rest and micro-averaged: every node
   and class is a pair, scored by the probability of the class, positive if it is the class of
   the node; for regression the targets and predictions of the nodes (graphs, for DiffPool) of
   the validation set. ``history`` is the loss at each epoch.
   """
   import pandas as pd
   p = result.get("predictions")
   if p is None:
      raise ValueError("this run has no target: it only computed embeddings")
   v = p[p["set"] == "validation"]
   ev = {"task": result["task"], "scores": {k: float(x) for k, x in result["scores"].loc["validation"].items()}, "history": result["history"]}
   if result["task"] == "classification":
      columns = [c for c in v.columns if c.startswith("p_")]
      targets = v["target"].astype(str).to_numpy()   # the columns are p_<class>, as str(class)
      pairs = pd.DataFrame({"score": np.concatenate([v[c].to_numpy(dtype = float) for c in columns]), "label": np.concatenate([targets == c[2:] for c in columns]).astype(int)})
      ev.update(gsk.link_prediction_evaluation(pairs, curves = True))
   else:
      ev.update(target = v["target"].to_numpy(dtype = float), prediction = v["prediction"].to_numpy(dtype = float))
   return ev


def _gnn_runs(ns):
   """
   The runs of the graph neural network dialog of a tab: ``runs``, one per result of ``gnn_runs``
   with a target (``name``, ``task``, validation ``scores``, ``auc`` and ``average_precision``
   for classification, and for the runs made in the interface their ``function`` and
   ``arguments``, for Copy…). Evaluations are computed once per run (:func:`gnn_evaluation`).
   """
   results = ns.get("gnn_runs", {})
   cache = ns.setdefault("__gnn_eval__", {})
   models = ns.get("__gnn_models__", {})
   runs = []
   for n, r in results.items():
      if r.get("predictions") is None:
         continue
      if cache.get(n, (None,))[0] is not r:
         cache[n] = (r, gnn_evaluation(r))
      ev = cache[n][1]
      run = {"name": n, "task": ev["task"], "scores": ev["scores"]}
      if ev["task"] == "classification":
         run.update(auc = float(ev["auc"]), average_precision = float(ev["average_precision"]))
      runs.append(dict(run, **models.get(n, {})))
   return {"runs": runs}


def _gnns(request):
   """
   Command ``gnns``: the runs of the graph neural network dialog of a tab (:func:`_gnn_runs`).
   """
   return _gnn_runs(_spaces[request["tab"]]), b""


def _gnn(request):
   """
   Command ``gnn``: a run of the graph neural network dialog of a tab (:func:`gnn_code`,
   ``function``, ``arguments``). Replies with the code, the drawing (unchanged) and the runs.
   """
   ns = _spaces[request["tab"]]
   lines, name = gnn_code(ns, request["function"], request.get("arguments") or {})
   code = _run(request["tab"], lines)
   ns.setdefault("__gnn_models__", {})[name] = {"function": request["function"], "arguments": dict(request.get("arguments") or {})}
   header, payload = _draw(ns, code, True)
   header.update(_gnn_runs(ns))
   if ns["gnn_runs"][name].get("predictions") is None:
      header["note"] = f"{name}: no target, only the embeddings of the graphs (gnn_runs[{literal(name)}][\"embeddings\"])"
   transformed = re.search(r'set_node_attribute\(("[^"]*")', lines[-2]) if len(lines) > 1 and "# the target, transformed" in lines[-2] else None
   if transformed:
      new = ast.literal_eval(transformed.group(1))
      left = int(np.isnan(node_values(ns["net"], new)).sum())
      header["note"] = f"{name}: trained on {new}, the transformed target" + (f"; {left:,} nodes without a value are left out" if left else "")
   return header, payload


#: Plots of :func:`gnn_plot`: label, the task of the runs drawn (``None``: any), and axis labels.
GNN_CURVES = {"roc": ("ROC", "classification", "false positive rate", "true positive rate"), "pr": ("Precision-recall", "classification", "recall", "precision"), "scatter": ("Predicted against observed", "regression", "observed value", "predicted value"), "loss": ("Learning curves", None, "epoch", "loss")}


def gnn_plot(evaluations, names, colors = None, options = None, styles = None, format = "png", path = None, dpi = 100):
   """
   Plot comparing runs of graph neural networks on their validation sets (option ``kind``): the
   ROC (``"roc"``, with the diagonal of a random score) or precision-recall curves (``"pr"``, with
   the fraction of positive pairs) of classification runs, each with its area in the legend; the
   predicted against the observed values of regression runs (``"scatter"``, points, with the
   diagonal of perfect predictions and :math:`R^2` in the legend); or the learning curves of any
   run (``"loss"``: training loss, solid, and validation loss, dashed, at each epoch). Runs of
   another task are left out.

   :param evaluations: Dictionary from run name to its :func:`gnn_evaluation`.
   :param names: The runs plotted.
   :return: The bytes of the image, or ``None`` with ``path``.
   """
   options = dict(PLOT_OPTIONS["gnn"], **(options or {}))
   kind = options.get("kind", "roc")
   label, task, xlabel, ylabel = GNN_CURVES[kind]
   options["xlabel"] = options["xlabel"] or xlabel
   options["ylabel"] = options["ylabel"] or ylabel
   shown = [n for n in names if task is None or evaluations[n]["task"] == task]
   def draw(ax):
      lo, hi = np.inf, -np.inf
      for k, n in enumerate(shown):
         ev, c = evaluations[n], (colors or {}).get(n, PLOT_COLORS[k % len(PLOT_COLORS)])
         style = (styles or {}).get(n, "points" if kind == "scatter" else "lines")
         if kind in ("roc", "pr"):
            df = ev["roc_curve" if kind == "roc" else "precision_recall_curve"]
            xs, ys = (df["fpr"], df["tpr"]) if kind == "roc" else (df["recall"], df["precision"])
            xs, ys = xs.to_numpy(), ys.to_numpy()
            if kind == "roc" and len(xs) and xs[0] == 0 and ys[0] > 0:
               xs, ys = np.r_[0., xs], np.r_[0., ys]
            _series(ax, xs, ys, style, c, f"{n} ({'AUC' if kind == 'roc' else 'AP'} {ev['auc' if kind == 'roc' else 'average_precision']:.3f})")
         elif kind == "scatter":
            _series(ax, ev["target"], ev["prediction"], style, c, f"{n} (R² {ev['scores'].get('r2', float('nan')):.3f})")
            both = np.r_[ev["target"], ev["prediction"]]
            both = both[np.isfinite(both)]
            if both.size:
               lo, hi = min(lo, both.min()), max(hi, both.max())
         else:
            h = ev["history"]
            _series(ax, h["epoch"], h["training_loss"], style, c, f"{n}, training")
            ax.plot(h["epoch"], h["validation_loss"], color = c, ls = "--", lw = 1.5, label = f"{n}, validation")
      if kind == "roc":
         ax.plot([0, 1], [0, 1], color = "#999999", ls = "--", lw = 1)
      elif kind == "pr" and shown:
         ev = evaluations[shown[0]]
         rate = ev["precision_recall_curve"]["precision"].iloc[-1]   # at recall 1: all the pairs, the fraction of positives
         ax.axhline(rate, color = "#999999", ls = "--", lw = 1)
      elif kind == "scatter" and np.isfinite(lo):
         ax.plot([lo, hi], [lo, hi], color = "#999999", ls = "--", lw = 1)
      if not shown:
         ax.text(0.5, 0.5, f"no {task} runs checked", ha = "center", va = "center", transform = ax.transAxes)
   return _figure(draw, options, format, path, dpi)


def gnn_text(evaluations, names, kind):
   """
   Text under the plot of graph neural networks: the validation scores of the runs drawn, best
   first.
   """
   label, task, _, _ = GNN_CURVES[kind]
   shown = [n for n in names if task is None or evaluations[n]["task"] == task]
   def describe(n):
      ev = evaluations[n]
      if ev["task"] == "classification":
         return f"{n}: AUC {ev['auc']:.4g}, AP {ev['average_precision']:.4g}, accuracy {ev['scores'].get('accuracy', float('nan')):.4g}, macro F1 {ev['scores'].get('f1_macro', float('nan')):.4g}"
      return f"{n}: R² {ev['scores'].get('r2', float('nan')):.4g}, MSE {ev['scores'].get('mse', float('nan')):.4g}, MAE {ev['scores'].get('mae', float('nan')):.4g}"
   key = (lambda n: -evaluations[n].get("auc" if kind == "roc" else "average_precision", 0.)) if task == "classification" else ((lambda n: -evaluations[n]["scores"].get("r2", 0.)) if task == "regression" else (lambda n: 0))
   left = len(names) - len(shown)
   return f"{label}, validation sets" + ("; classification curves are one-vs-rest, micro-averaged over the classes" if task == "classification" else "") + ". " + "; ".join(describe(n) for n in sorted(shown, key = key)) + "." + (f" {left} runs of the other task left out." if left else "")


def _gnn_plot(request):
   """
   Command ``gnn_plot``: plots the runs ``names`` of the graph neural network dialog of a tab
   (:func:`gnn_plot`, ``kind`` of :data:`GNN_CURVES`, which the plot window can switch), and
   replies with the PNG image as payload, ``elements`` (one color and style per run) and ``text``
   (:func:`gnn_text`).
   """
   ns = _spaces[request["tab"]]
   info = _gnn_runs(ns)
   names = [n for n in request.get("names") or [] if n in {r["name"] for r in info["runs"]}]
   if not names:
      raise ValueError("choose the runs to plot")
   kind = request.get("kind") if request.get("kind") in GNN_CURVES else "roc"
   evaluations = {n: ns["__gnn_eval__"][n][1] for n in names}
   ns["__plot__"] = _plot("gnn", {"evaluations": evaluations, "names": names}, {n: PLOT_COLORS[i % len(PLOT_COLORS)] for i, n in enumerate(names)}, {n: "points" if kind == "scatter" else "lines" for n in names}, {n: n for n in names}, source = {"evaluations": "{" + ", ".join(f"{literal(n)}: gnn_evaluation(gnn_runs[{literal(n)}])" for n in names) + "}", "names": repr(names)})
   ns["__plot__"]["options"]["kind"] = kind
   return dict(_plot_header(ns["__plot__"]), warnings = []), _render(ns["__plot__"], "png")


def last_layout(code):
   """
   The function of the last line of code that changes the drawing: the function of :data:`CURVES`
   of a line computing curves of the edges (:data:`BENDS`), else the function of :data:`LAYOUTS`
   of the last line calling one, or of a line moving nodes (:func:`move_nodes`) or changing the
   network (``net = gsk.…``); ``None`` if none. Edges are drawn as curves after a function of
   :data:`CURVES`, the network as a matrix after ``matrix_layout``; anything else draws nodes and
   straight edges again.

   :param code: Lines of code of a tab.
   """
   for line in reversed(code):
      if line.startswith(BENDS):
         return line[len(BENDS):].split("(")[0]
      if line.startswith(("move_nodes(", "net = gsk.", "inc = gsk.")):   # moved nodes, a changed network
         return line.split("(")[0].split(".")[-1]
      for f in LAYOUTS:
         if line.startswith(f"gsk.{f}("):
            return f
   return None


def _matrix_attribute(ns, last):
   """
   Name of the node attribute of the last matrix layout of a tab (``attribute`` of its call,
   ``"matrix"`` by default).
   """
   line = next(l for l in reversed(ns["__code__"]) if l.startswith("gsk.matrix_layout("))
   m = re.search(r"attribute\s*=\s*(\"[^\"]*\"|'[^']*')", line)
   return ast.literal_eval(m.group(1)) if m else "matrix"


def _communities(request):
   """
   Command ``communities``: runs a function of :data:`COMMUNITIES` (``function``, ``arguments``) on
   the drawn network of a tab (:func:`layout_code`), and replies with the code and the drawing. The
   communities become an attribute; the look of the drawing does not change (colors and sizes are
   always the user's choice, in the visual elements panel). The hierarchical ones
   (:data:`HIERARCHICAL_COMMUNITIES`) remember their levels (``__levels__``, :func:`level_attributes`)
   and, with ``tree`` (``"True"``), reply them as ``tree``: the interface then opens the tree of
   the hierarchy in a new tab (command ``community_tree``, :func:`_community_tree`).
   """
   ns = _spaces[request["tab"]]
   function, arguments = request["function"], dict(request.get("arguments") or {})
   tree = arguments.pop("tree", "False").strip() == "True"
   code = _run(request["tab"], layout_code(ns, function, arguments))
   if function in HIERARCHICAL_COMMUNITIES:
      attribute = ast.literal_eval(arguments.get("attribute", literal(inspect.signature(getattr(gsk, function)).parameters["attribute"].default)))
      ns["__levels__"] = level_attributes(_drawn(ns), attribute)
   header, payload = _draw(ns, code, True)
   if tree:
      header["tree"] = ns["__levels__"]
   return header, payload


def _community_tree(request):
   """
   Command ``community_tree`` (Community discovery › Tree of a hierarchy…, and the ``tree`` option
   of the hierarchical methods): the tree of the levels of communities of a tab
   (:func:`~graphoskop.community_tree`, ``arguments``: ``levels``, by default those of the last
   hierarchical run, and ``leaves``) in a new tab ``new_tab``, laid out radially, the root at the
   center (:func:`~graphoskop.hierarchical_layout`), its nodes colored by ``level``, e.g.

   .. code-block:: python

      net = gsk.community_tree(net, ["community", "community_1"])
      gsk.hierarchical_layout(net, direction = "radial")
   """
   ns = _spaces[request["tab"]]
   arguments = {k: v.strip() for k, v in dict(request.get("arguments") or {}).items() if str(v).strip()}
   levels = _source_value(ns, arguments.pop("levels")) if arguments.get("levels") else ns.get("__levels__")   # the dialog sends only what differs from its defaults
   if not levels:
      raise ValueError("levels: choose the node attributes of the levels of the hierarchy")
   t = _target(ns)
   extra = "".join(f", {k} = {v}" for k, v in arguments.items())
   lines = [f"net = gsk.community_tree({t}, {literal(levels)}{extra})", "gsk.hierarchical_layout(net, direction = \"radial\")   # the root at the center"] + (["del inc"] if t == "inc" else [])
   style = dict(ns.get("__style__") or {}, nodes = {"attribute": "level", "scale": "qualitative", "palette": list(CHART_PALETTE), "overrides": [], "missing": MISSING_COLOR}, node_chart = {})   # nodes colored by level
   return _derived(ns, request["new_tab"], lines, preset = {"__style__": style})


def tree_signature(ns):
   """
   Parameters of Community discovery › Tree of a hierarchy…: the ``levels`` (node attributes; by
   default those of the last hierarchical community discovery of the tab) and ``leaves``.
   """
   params = dendrogram_signature(ns)["params"][:1]
   params.append({"name": "leaves", "default": "True", "hint": "", "doc": "The nodes are the leaves of the tree (else the groups of the finest level are)", "widget": {"kind": "bool"}})
   return {"params": params}


#: Community discovery functions that find a hierarchy: levels of communities written as
#: ``attribute``, ``attribute_1``, ``attribute_2``, ... (:func:`level_attributes`); their dialog
#: offers the tree of the hierarchy in a new tab.
HIERARCHICAL_COMMUNITIES = {"block_model_communities", "hierarchical_random_graph_communities", "infomap_communities"}
#: Largest number of nodes drawn as the leaves of a dendrogram (:func:`dendrogram_tree`): above,
#: the leaves are the groups of the finest level.
DENDROGRAM_NODES = 150
#: Largest number of leaves of a dendrogram whose labels are written (:func:`dendrogram_plot`).
DENDROGRAM_LABELS = 80


def level_attributes(net, attribute):
   """
   Node attributes of the levels of a hierarchy of communities: ``attribute`` and those named
   ``attribute_1``, ``attribute_2``, ... that the network has (stopping at the first missing).
   """
   names = set(net.node_attributes) | set(net.qualitative_node_attributes)
   levels = [attribute] if attribute in names else []
   k = 1
   while f"{attribute}_{k}" in names:
      levels.append(f"{attribute}_{k}")
      k += 1
   return levels


def dendrogram_tree(net, levels, leaves = None, label = None):
   """
   The tree of a hierarchy of communities, for its dendrogram (:func:`dendrogram_plot`): the levels
   (node attributes, in any order) are sorted from the finest (most groups) to the coarsest, every
   group of a level hangs from the group of the next level that holds most of its nodes (exactly
   one if the levels are nested), the groups of the coarsest level from a root. The leaves are the
   nodes (``leaves = "nodes"``, by default up to :data:`DENDROGRAM_NODES` nodes) or the groups of
   the finest level (``"groups"``); they are ordered so that no branches cross. Missing values make
   a group of their own. A function of the interface, not of the library.

   :param net: The network (the drawn one).
   :param levels: Names of the node attributes of the levels.
   :param leaves: ``"nodes"``, ``"groups"`` or ``None`` (by the number of nodes).
   :param label: Node attribute naming the leaf nodes, or ``None`` for their ids.
   :return: Dictionary with ``x``, ``y`` and ``parent`` (-1 for the root) of every vertex of the
      tree, the ``labels`` of the leaves (the vertices with ``y`` 0, in the order of ``x``), what
      the ``leaves`` are (``"nodes"`` or the finest level) and the names of the ``levels`` of the
      heights 1, 2, ... (finest first; the root above them).
   :raises ValueError: If there are no levels.
   """
   import pandas as pd
   if not levels:
      raise ValueError("levels: choose the node attributes of the levels of the hierarchy")
   n = int(net.number_of_nodes())
   codes = [pd.factorize(pd.Series([text(v) for v in node_values(net, a)], dtype = object).fillna("(none)"))[0] for a in levels]
   order = sorted(range(len(levels)), key = lambda i: -len(np.unique(codes[i])))
   codes, names = [codes[i] for i in order], [levels[i] for i in order]
   if leaves is None:
      leaves = "nodes" if n <= DENDROGRAM_NODES else "groups"
   ids = [text(v) for v in (_ids(net) if label is None else node_values(net, label))]
   if leaves == "nodes":
      codes, names = [np.arange(n)] + codes, ["nodes"] + names
   codes.append(np.zeros(n, dtype = np.int64))   # the root
   sort = np.lexsort(codes[::-1])                  # coarse to fine: no crossings
   vx, vy, vp, labels, prev = [], [], [], [], None
   for depth, c in enumerate(codes):
      groups, first = np.unique(c[sort], return_index = True)
      groups = groups[np.argsort(first)]           # in the order of the leaves
      if depth == 0:
         xs = np.arange(len(groups), dtype = float)
         sizes = np.bincount(c, minlength = c.max() + 1)
         members = {g: np.flatnonzero(c == g)[0] for g in groups}
         labels = [ids[members[g]] if leaves == "nodes" else f"{text(node_values(net, names[0])[members[g]])} ({sizes[g]})" for g in groups]
         at_leaf = dict(zip(groups.tolist(), xs))
         leaf_x = np.array([at_leaf[g] for g in c])   # x of the leaf of every node
      else:   # over the children, or over the nodes of a group no finer group chose
         xs = np.array([np.mean([vx[at] for at in children[g]]) if g in children else leaf_x[c == g].mean() for g in groups])
      base = len(vx)
      index = {g: base + k for k, g in enumerate(groups)}
      if prev is not None:   # the parents of the previous level: the group holding most of their nodes
         for g, at in prev.items():
            vp[at] = index[np.bincount(c[codes[depth - 1] == g]).argmax()]
      vx += list(xs)
      vy += [float(depth)] * len(groups)
      vp += [-1] * len(groups)
      if depth + 1 < len(codes):
         nxt = codes[depth + 1]
         children = {}
         for g, at in index.items():
            children.setdefault(np.bincount(nxt[c == g]).argmax(), []).append(at)
      prev = index
   return {"x": vx, "y": vy, "parent": vp, "labels": labels, "leaves": names[0], "levels": names[1:]}


def dendrogram_plot(tree, colors = None, options = None, styles = None, format = "png", path = None, dpi = 100):
   """
   Plot of the dendrogram of a hierarchy of communities (:func:`dendrogram_tree`): every group a
   horizontal bar over its children at the height of its level, the leaves at the bottom (their
   labels written up to :data:`DENDROGRAM_LABELS` leaves), the names of the levels on the y axis
   (see ``_figure``).

   :param tree: Result of :func:`dendrogram_tree`.
   :return: The bytes of the image, or ``None`` with ``path``.
   """
   from matplotlib.collections import LineCollection
   colors = dict(PLOT_ELEMENTS["dendrogram"][1], **(colors or {}))
   options = dict(PLOT_OPTIONS["dendrogram"], **(options or {}))
   x, y, parent = np.asarray(tree["x"]), np.asarray(tree["y"]), np.asarray(tree["parent"])
   def draw(ax):
      child = np.flatnonzero(parent >= 0)
      up = [[(x[i], y[i]), (x[i], y[parent[i]])] for i in child]
      across = [[(x[child[parent[child] == p]].min(), y[p]), (x[child[parent[child] == p]].max(), y[p])] for p in np.unique(parent[child])]
      ax.add_collection(LineCollection(up + across, colors = colors["tree"], linewidths = 1.2))
      leaves = np.flatnonzero(y == 0)
      ax.scatter(x[leaves], y[leaves], s = 12, color = colors["leaves"], zorder = 3)
      ax.set_xlim(-0.5, max(len(leaves) - 0.5, 0.5))
      ax.set_ylim(-0.3, y.max() + 0.3)
      ticks = np.arange(int(y.max()) + 1)
      ax.set_yticks(ticks, [(tree["leaves"] if t == 0 else (tree["levels"][t - 1] if t - 1 < len(tree["levels"]) else "root")) for t in ticks])
      if len(leaves) <= DENDROGRAM_LABELS:
         ax.set_xticks(x[leaves], tree["labels"], rotation = 90, fontsize = max(6., float(options["font_size"]) * 0.6))
      else:
         ax.set_xticks([])
   return _figure(draw, dict(options, xmin = options.get("xmin"), xmax = options.get("xmax")), format, path, dpi)


def dendrogram_signature(ns):
   """
   Parameters of Analysis › Community discovery › Dendrogram of a hierarchy…: the node attributes
   of the ``levels`` (by default those of the last hierarchical community discovery of the tab,
   ``__levels__``), the ``leaves`` (nodes or the groups of the finest level) and the attribute
   naming the leaf nodes (``label``).
   """
   levels = ns.get("__levels__") if ns is not None else None
   return {"params": [{"name": "levels", "default": literal(levels) if levels else "", "hint": "required", "doc": "Node attributes of the levels of the hierarchy, in any order (sorted from the one with most groups), e.g. community, community_1, community_2 of a hierarchical community discovery", "widget": {"kind": "attribute", "choices": "nodes", "multiple": True, "extra": []}},
                      {"name": "leaves", "default": "", "hint": f"nodes up to {DENDROGRAM_NODES}", "doc": "The leaves of the tree: the nodes, or the groups of the finest level (with their sizes)", "widget": {"kind": "choice", "options": ["", "\"nodes\"", "\"groups\""]}},
                      {"name": "label", "default": "", "hint": "node id", "doc": "Node attribute naming the leaf nodes", "widget": {"kind": "attribute", "choices": "nodes", "multiple": False, "extra": []}}]}


def _dendrogram(request, code = None):
   """
   Command ``dendrogram``: the plot of the dendrogram of a hierarchy of communities of a tab
   (``arguments``: ``levels``, ``leaves``, ``label``, see :func:`dendrogram_signature`), in the
   plot window (``plot`` in the header, its PNG the last part of the payload), with the drawing
   (unchanged). The data of the plot are :func:`dendrogram_tree` of the drawn network.
   """
   ns = _spaces[request["tab"]]
   arguments = {k: v.strip() for k, v in dict(request.get("arguments") or {}).items() if str(v).strip()}
   levels = _source_value(ns, arguments["levels"]) if arguments.get("levels") else ns.get("__levels__") or []   # the dialog sends only what differs from its defaults
   levels = [levels] if isinstance(levels, str) else list(levels)
   extra = "".join(f", {k} = {arguments[k]}" for k in ("leaves", "label") if k in arguments)
   tree = dendrogram_tree(_drawn(ns), levels, **{k: ast.literal_eval(arguments[k]) for k in ("leaves", "label") if k in arguments})
   header, payload = _draw(ns, code if code is not None else "\n".join(ns["__code__"]), True)
   ns["__plot__"] = _plot("dendrogram", {"tree": tree}, dict(PLOT_ELEMENTS["dendrogram"][1]), {}, source = {"tree": f"dendrogram_tree({_target(ns)}, {literal(levels)}{extra})"})
   png = _render(ns["__plot__"], "png")
   leaves = sum(1 for v in tree["y"] if v == 0)
   shown = ([] if tree["leaves"] == "nodes" else [tree["leaves"]]) + tree["levels"]
   header["plot"] = dict(_plot_header(ns["__plot__"], ns), text = f"Dendrogram of {len(shown)} levels ({', '.join(shown)}), {leaves:,} leaves ({'nodes' if tree['leaves'] == 'nodes' else 'groups of ' + tree['leaves']}).", warnings = [], title = f"Dendrogram of {request.get('name') or 'the network'}")
   header["parts"] = header.get("parts", []) + [len(png)] if header.get("nodes") else [len(png)]
   return header, payload + png


def _layout(request):
   """
   Command ``layout``: runs the code of :func:`layout_code` (``function`` and ``arguments`` of the
   request) in the namespace of the tab, and replies with the code and the drawing.
   ``remove_overlaps`` and ``route_edges`` get as ``size`` the diameters of the drawn nodes
   (:func:`size_code`). The drawing keeps the default node diameter after transformations of the
   layout and curves of the edges (:data:`TRANSFORMS`).
   """
   ns = _spaces[request["tab"]]
   function, arguments = request.get("function", "grid_layout"), dict(request.get("arguments") or {})
   if function in ("remove_overlaps", "route_edges") and "size" not in arguments:
      arguments["size"] = size_code(ns)
   lines = layout_code(ns, function, arguments)
   if "node_sizes(" in arguments.get("size", ""):
      lines.insert(-1, NODE_SIZES_NOTE)
   code = _run(request["tab"], lines)
   return _draw(ns, code, function in TRANSFORMS)


def size_code(ns):
   """
   Source code of the diameters of the drawn nodes of a tab, for ``remove_overlaps`` and
   ``route_edges``: the default
   diameter (3 significant digits) times the relative size of the nodes, a number, or
   :func:`node_sizes` of the attribute of the node sizes (defined in the namespace of the tab, not
   in the library: the code panel says so in a comment, :data:`NODE_SIZES_NOTE`).
   """
   d = float(f"{ns['__size__']:.3g}")
   nz = ns.get("__style__", {}).get("node_size") or {}
   if nz.get("attribute") is None:
      return repr(float(f"{d * float(nz.get('size', 1.)):.3g}"))
   target = "inc" if "inc" in ns else "net"
   log = ", log = True" if nz.get("log") else ""
   return f"{d!r} * node_sizes({target}, {literal(nz['attribute'])}, smallest = {float(nz.get('min', 0.5))!r}, largest = {float(nz.get('max', 2.))!r}{log})"


def _style(request):
   """
   Command ``style``: sets the ``style`` of the drawing of a tab (see :func:`default_style`), and
   replies with the drawing; a new ``position`` changes the scale of the drawing, so the default
   diameter of the nodes is computed again. An invalid style is not kept.
   """
   ns = _spaces[request["tab"]]
   old = ns.get("__style__", {})
   ns["__style__"] = request["style"]
   try:
      return _draw(ns, "\n".join(ns["__code__"]), (old.get("position") or {}) == (request["style"].get("position") or {}))
   except Exception:
      ns["__style__"] = old
      raise


def _breaks(request):
   """
   Command ``breaks``: :func:`breaks` of the node (``kind = "nodes"``) or edge attribute
   ``attribute`` of the drawn network of a tab, with ``method``, ``k`` and ``log``.
   """
   ns = _spaces[request["tab"]]
   draw = _drawn(ns)
   v = node_values(draw, request["attribute"]) if request["kind"] == "nodes" else edge_values(draw, request["attribute"])
   return {"stops": breaks(v, request.get("method", "quantiles"), int(request.get("k", 5)), bool(request.get("log", False)))}, b""


def _attributes(request):
   """
   Command ``attributes``: :func:`attributes` of the drawn network of a tab.
   """
   ns = _spaces[request["tab"]]
   return attributes(_drawn(ns)), b""


def histories(net):
   """
   The histories of epidemics stored in a network (``history = True`` of the epidemic models):
   dictionary from the name of the states attribute to the last step, for every qualitative node
   attribute ``name_0``, ``name_1``, ... with consecutive steps from 0.
   """
   steps = {}
   for a in net.qualitative_node_attributes:
      name, _, t = a.rpartition("_")
      if name and t.isdigit():
         steps.setdefault(name, set()).add(int(t))
   found = {}
   for name, ts in steps.items():
      if 0 in ts:
         last = 0
         while last + 1 in ts:
            last += 1
         found[name] = last
   return found


def _history_name(net):
   """
   The history shown by the player of the interface: ``state`` if there is one, else the first one
   (see :func:`histories`), or ``None``.
   """
   found = histories(net)
   return ("state" if "state" in found else next(iter(found))) if found else None


def _history_key(ns, net):
   """
   A text that changes when the history shown by the player of a tab changes (another network
   object, after a removal or an undo; another simulation), or an empty text without history.
   """
   name = _history_name(net)
   return f"{name}:{histories(net)[name]}:{id(net)}:{ns.get('__runs__', 0)}" if name else ""


#: Colors of the states in the player of the interface (other states: :data:`PLOT_COLORS`).
STATE_COLORS = {"S": "#377eb8", "I": "#e41a1c", "R": "#4daf4a", "V": "#984ea3"}


def spread(net, name):
   """
   The history ``name`` of an epidemic stored in a network (see :func:`histories`), for the
   player of the interface: the states, the state code of every node at step 0, and, for every
   following step, the nodes whose state changed, with their new and old codes (missing states,
   e.g. nodes of other components, are 255).

   :return: ``(states, initial, starts, nodes, new, old)``: ``starts[t - 1]`` to ``starts[t]``
      are the changes of step ``t`` in the arrays ``nodes`` (int32), ``new`` and ``old`` (uint8).
   """
   import pandas as pd
   last = histories(net)[name]
   if net.node_format == "wide" and net.higher_order is None:   # the categorical columns: codes and their categories
      steps = [pd.Categorical(net.qualitative_node_attr[f"{name}_{t}"]) for t in range(last + 1)]
      values = lambda x: (x.codes, x.categories.astype(str))
   else:
      steps = [pd.Series(node_values(net, f"{name}_{t}"), dtype = object) for t in range(last + 1)]
      values = lambda x: pd.factorize(x.map(lambda v: v if pd.isna(v) else str(v)))
   pairs = [values(x) for x in steps]
   found = sorted({str(v) for _, c in pairs for v in c}, key = lambda v: ("SIRV".find(v) if v in STATE_COLORS else 9, v))
   index = pd.Index(found, dtype = object)
   C = np.full((last + 1, len(pairs[0][0])), 255, dtype = np.uint8)
   for t, (codes, categories) in enumerate(pairs):
      lut = np.append(index.get_indexer(pd.Index(categories, dtype = object)), 255)   # code -1 (missing): 255
      C[t] = lut[np.asarray(codes)]
   nodes, new, old, starts = [], [], [], [0]
   for t in range(1, last + 1):
      i = np.flatnonzero(C[t] != C[t - 1])
      nodes.append(i)
      new.append(C[t, i])
      old.append(C[t - 1, i])
      starts.append(starts[-1] + len(i))
   cat = lambda x, dtype: np.concatenate(x).astype(dtype) if x else np.zeros(0, dtype)
   return found, C[0], np.array(starts, dtype = np.int32), cat(nodes, np.int32), cat(new, np.uint8), cat(old, np.uint8)


def _spread(request):
   """
   Command ``spread``: the history of the epidemic of a tab shown by the player (:func:`spread` of
   :func:`_history_name`), with its ``name``, last step (``steps``), ``states``, their colors
   (:data:`STATE_COLORS`), the ``run`` it comes from, and the parts of the payload (initial codes,
   starts, nodes, new and old codes); ``steps`` is -1 without history.
   """
   ns = _spaces[request["tab"]]
   net = _drawn(ns)
   name = _history_name(net)
   if name is None:
      return {"steps": -1}, b""
   states, initial, starts, nodes, new, old = spread(net, name)
   parts = [initial.tobytes(), starts.tobytes(), nodes.tobytes(), new.tobytes(), old.tobytes()]
   colors_ = [STATE_COLORS.get(v, PLOT_COLORS[k % len(PLOT_COLORS)]) for k, v in enumerate(states)]
   return {"name": name, "steps": len(starts) - 1, "states": states, "colors": colors_, "run": ns.get("__history_run__", ""), "parts": [len(p) for p in parts]}, b"".join(parts)


#: Measures of the randomization tests of fitted generators (:func:`fit_code`): their label in the
#: interface, their code (a function of a network returning a number), and the networks they apply
#: to (``"all"``, ``"directed"``, ``"networks"`` (not hypergraphs nor simplicial complexes),
#: ``"higher"`` (hypergraphs and simplicial complexes)).
MEASURES = {"Average clustering": ("gsk.average_clustering", "networks"), "Global clustering": ("gsk.global_clustering", "networks"), "Degree assortativity": ("gsk.assortativity", "networks"), "Average shortest path length": ("gsk.average_shortest_path_length", "all"), "Diameter": ("gsk.diameter", "all"), "Number of components": ("lambda g: len(g.components)", "all"), "Reciprocity": ("gsk.reciprocity", "directed"), "Global reach centrality": ("gsk.global_reach_centrality", "directed"), "Global triangle closure": ("gsk.global_triangle_closure", "higher")}


#: Parameters of generators that are not taken from the network they are fitted to.
FIT_REQUIRED = {("watts_strogatz_graph", "p")}


def measures(net):
   """
   Labels of the :data:`MEASURES` that apply to a network (all of them for ``None``).
   """
   if net is None:
      return list(MEASURES)
   kind = {"all"} | ({"higher"} if net.higher_order is not None else {"networks"} | ({"directed"} if net.directed else set()))
   return [m for m, (_, k) in MEASURES.items() if k in kind]


def fit_code(ns, function, arguments):
   """
   Code fitting a generator of :data:`GENERATORS` to the network of a tab: one null network, in a
   new tab (``net = gsk.configuration_model(net = net)``, then ``del inc`` for hypergraphs and
   simplicial complexes drawn as their incidence networks), and, if ``arguments`` has a ``test``
   (a label of :data:`MEASURES`), the randomization test of that measure with ``runs`` null networks
   (default 100), in the tab of the network, e.g. ``null_test = gsk.randomization_test(net,
   gsk.average_clustering, gsk.configuration_model, n = 100)``. The other arguments go to the
   generator in both.

   :param ns: Namespace of the tab.
   :param function: Name of the generator.
   :param arguments: Parameter name to Python source code.
   :return: The lines of the new tab, and the lines of the test (empty without test).
   :raises ValueError: If the function does not take a network, the measure is unknown, or a value
      is not a Python expression.
   """
   if function not in GENERATORS or "net" not in inspect.signature(getattr(gsk, function)).parameters:
      raise ValueError(f"{function} cannot be fitted to a network")
   arguments = dict(arguments)
   test, runs = arguments.pop("test", ""), arguments.pop("runs", "100")
   for k, v in list(arguments.items()) + [("runs", runs)]:
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   extra = "".join(f", {k} = {v.strip()}" for k, v in arguments.items())
   lines = [f"net = gsk.{function}(net = net{extra})"] + (["del inc"] if "inc" in ns else [])
   if not test.strip():
      return lines, []
   label = ast.literal_eval(test)
   if label not in MEASURES:
      raise ValueError(f"unknown measure: {label}")
   return lines, [f"null_test = gsk.randomization_test(net, {MEASURES[label][0]}, gsk.{function}, n = {runs.strip()}{extra})"]


def _fit(request):
   """
   Command ``fit``: a generator fitted to the network of a tab (:func:`fit_code`): one null network
   in the new tab ``new_tab`` (:func:`_derived`), and, with a ``test``, the randomization test in
   the tab, kept as the plot of the tab (:func:`randomization_plot`). Replies with the drawing of
   the new tab and, with a test, ``test``: the code of the tab, and what the plot window needs
   (:func:`_plot_header`, with the ``text`` of :func:`randomization_text`); the PNG of the plot is
   the last part of the payload.
   """
   ns = _spaces[request["tab"]]
   arguments = request.get("arguments") or {}
   lines, test = fit_code(ns, request["function"], arguments)
   header, payload = _derived(ns, request["new_tab"], lines)
   if not test:
      return header, payload
   code = _run(request["tab"], test)
   r = ns["null_test"]
   label = ast.literal_eval(arguments["test"])
   ns["__plot__"] = _plot("randomization", {"distribution": r["distribution"], "observed": r["observed"], "measure": label, "mean": r["mean"], "std": r["std"], "above": r["above"], "below": r["below"], "equal": r["equal"]}, dict(PLOT_ELEMENTS["randomization"][1]), {"null": "bars"}, source = dict({k: f"null_test[{literal(k)}]" for k in ("distribution", "observed", "mean", "std", "above", "below", "equal")}, measure = literal(label)))
   png = _render(ns["__plot__"], "png")
   header["test"] = dict(_plot_header(ns["__plot__"]), code = code, measure = label, warnings = [])
   header["parts"] = header.get("parts", []) + [len(png)] if header["nodes"] else [len(png)]
   return header, payload + png


#: Parameters shown in the parameter dialog also without **Advanced** (by name, or by function
#: and name), besides those without a default (:func:`mark_advanced`).
BASIC = {"name", "tree", ("community_tree", "leaves"), ("complement", "self_loops"), ("dendrogram", "leaves"), ("dendrogram", "label"), ("layer_network", "layer"), ("route_edges", "size"), ("route_edges", "margin"), ("modularity", "partition"), ("modularity", "classes"), ("modularity", "method"), ("all_pairs_shortest_path_lengths", "average_distance"), ("all_pairs_shortest_path_lengths", "heatmap"), ("node_removal", "strategy"), ("edge_removal", "strategy"), ("cascade_failures", "strategy"), ("node_removal", "quality"), ("edge_removal", "quality"), ("cascade_failures", "load"), ("cascade_failures", "capacity"), ("cascade_failures", "failures"), ("cascade_failures", "elements"), "target", "by", "all_components", "levels", "model", "measurements", "positives", "probability", ("assortativity", "attribute"), "node_label", "edge_label", "max_edges", "rules", "transform", "task", "features", "hidden", "epochs", "weight", "directed", "direction", "groups", "order", "levels", "root", "attributes", "a", "b", "seeds", "runs", "node_type", "sets", "plot", "resolution", "source", "length", "walks", "max_depth", "test", "embedding", "fraction",
         ("matrix_layout", "column_groups"), ("matrix_layout", "column_order"), ("mirror_layout", "axis"), ("spanning_tree", "kind"), ("maximally_filtered_graph", "kind"), ("triangulated_filtered_graph", "kind")}


def mark_advanced(function, reply):
   """
   Marks the parameters of a signature reply that the parameter dialog shows only with
   **Advanced** on (``"advanced": True``): all but those without a default (required, or preset
   by the interface) and those of :data:`BASIC`, so that a first look shows only the parameters
   that matter most.
   """
   for p in reply["params"]:
      p["advanced"] = not (p["default"] == "" and p["hint"].startswith("required") or p["name"] in BASIC or (function, p["name"]) in BASIC)
   return reply


def _signature(request):
   """
   Command ``signature``: :func:`signature` of a generator, a layout or a community discovery
   function, for the drawn network of the tab ``tab`` (if given); with ``fit``, of a generator
   fitted to the network of the tab. Parameters are marked by :func:`mark_advanced`.
   """
   reply, payload = _signature_of(request)
   return mark_advanced(request["function"], reply), payload


def _signature_of(request):
   """
   The reply of :func:`_signature`, before :func:`mark_advanced`.
   """
   if request["function"] not in GENERATORS | LAYOUTS | COMMUNITIES | set(CENTRALITIES) | set(EPIDEMICS) | set(BACKBONES) | FILTERS | set(PROJECTIONS) | set(TRANSFORMATIONS) | set(VECTORS) | set(LINK_PREDICTIONS) | set(GNNS) | set(EXPLORATIONS) | set(MINERS) | set(HIERARCHIES) | set(STATISTICS) | set(CORRECTIONS) | set(ROBUSTNESS) | set(SYNCHRONIZATION) | {"all_pairs_shortest_path_lengths", "nestedness", "link_prediction_split", "dendrogram", "community_tree"}:
      raise ValueError(f"unknown function: {request['function']}")
   tab = request.get("tab")
   net = _drawn(_spaces[tab]) if tab in _spaces else None
   if request.get("fit"):
      if tab not in _spaces:
         raise ValueError("open a network to fit the generator to")
      net = _spaces[tab]["net"]
      if "net" not in inspect.signature(getattr(gsk, request["function"])).parameters:
         raise ValueError(f"{request['function']} cannot be fitted to a network")
      return signature(request["function"], net, True), b""
   if request["function"] == "nestedness" and (net is None or net.bipartite_labels is None):
      raise ValueError("nestedness is for bipartite networks (Bipartite in the load dialog), and hypergraphs and simplicial complexes")
   if request["function"] in PROJECTIONS and (net is None or net.bipartite_labels is None):
      raise ValueError("bipartite projections need a bipartite network (Bipartite in the load dialog)")
   if request["function"] in EXPLORATIONS:
      return exploration_signature(request["function"], net), b""
   if request["function"] in MINERS:
      return mining_signature(request["function"], net), b""
   if request["function"] in HIERARCHIES:
      return hierarchy_signature(request["function"], net), b""
   if request["function"] in STATISTICS:
      return statistic_signature(request["function"], net), b""
   if request["function"] in CORRECTIONS:
      return correction_signature(request["function"], net), b""
   if request["function"] in ROBUSTNESS:
      return robustness_signature(request["function"], net), b""
   if request["function"] == "all_pairs_shortest_path_lengths":
      return paths_signature(net), b""
   if request["function"] == "dendrogram":
      return dendrogram_signature(_spaces.get(tab)), b""
   if request["function"] == "community_tree":
      return tree_signature(_spaces.get(tab)), b""
   if request["function"] in SYNCHRONIZATION:
      return synchronization_signature(request["function"], _spaces[tab]["net"] if tab in _spaces else None), b""
   if request["function"] in GNNS:
      return gnn_signature(request["function"], _spaces[tab]["net"] if tab in _spaces else None), b""
   if request["function"] in set(LINK_PREDICTIONS) | {"link_prediction_split"}:
      return link_signature(request["function"], _spaces[tab]["net"] if tab in _spaces else None, None), b""
   return signature(request["function"], net), b""


def _replay(request):
   """
   Command ``replay``: runs the whole ``code`` of a tab in a new namespace (e.g. after switching
   to another Python environment), with the ``style`` of its drawing, and replies with the code
   and the drawing (none with ``undrawn``).
   """
   ns = dict(__code__ = request["code"].split("\n"), **_helpers())
   if request.get("style"):
      ns["__style__"] = request["style"]
   if request.get("undrawn"):
      ns["__undrawn__"] = True
   exec(request["code"], ns)
   _spaces[request["tab"]] = ns
   return _draw(ns, request["code"])


def _preview(request):
   """
   Command ``preview``: :func:`preview` of ``path``.
   """
   return preview(request["path"], request.get("sep", "\t"), request.get("header", True)), b""


def _close(request):
   """
   Command ``close``: forgets the namespace of a tab.
   """
   _spaces.pop(request["tab"], None)
   return {}, b""


#: Functions of the formulas of derived node attributes (:func:`formula_values`): name to its
#: description (shown in the menu of the dialog).
FORMULA_FUNCTIONS = {
   "log": "log(x): natural logarithm (missing where x ≤ 0)",
   "log2": "log2(x): base 2 logarithm",
   "log10": "log10(x): base 10 logarithm",
   "log1p": "log1p(x): log(1 + x), also defined at 0",
   "exp": "exp(x): exponential",
   "sqrt": "sqrt(x): square root (missing where x < 0)",
   "abs": "abs(x): absolute value",
   "round": "round(x) or round(x, digits)",
   "floor": "floor(x): largest integer not above x",
   "ceil": "ceil(x): smallest integer not below x",
   "min": "min(x): smallest value over all nodes; min(x, y): the smaller of the two, node by node",
   "max": "max(x): largest value over all nodes; max(x, y): the larger of the two, node by node",
   "mean": "mean(x): mean over all nodes (missing values left out)",
   "sum": "sum(x): sum over all nodes",
   "where": "where(condition, a, b): a where the condition holds (e.g. age > 30), b elsewhere",
   "missing": "missing(x): 1 where x is missing, 0 elsewhere",
   "fill": "fill(x, value): x, with value where it is missing",
   "number": "number(x): a text as a number (missing if it is not one)",
   "text": "text(x): a value as text",
   "substr": "substr(x, start, length): the characters of a text from position start (1 is the first), length of them (all the rest without length)",
   "upper": "upper(x): text in capitals",
   "lower": "lower(x): text in small letters",
   "length": "length(x): number of characters of a text",
   "contains": "contains(x, \"abc\"): 1 where the text contains abc, 0 elsewhere",
   "replace": "replace(x, \"old\", \"new\"): the text with every old replaced by new",
}
#: Comment of the code before the first :func:`derived`, a function of the interface.
DERIVED_NOTE = "# derived: values of a node attribute computed from others by a formula, a function of the graphical interface"


def formula_values(net, formula):
   """
   Values of a formula of node attributes, node by node (in the network wide order), as written
   in the dialog of Edit › New node attribute: node attribute names (in backquotes if they are
   not plain words, e.g. ```age in 2020```), numbers, texts in quotes, ``+ - * / ** %`` (``+``
   also joins texts), comparisons (``< <= > >= == !=``, 1 or 0), ``and``, ``or``, ``not``,
   parentheses and the functions of :data:`FORMULA_FUNCTIONS`, e.g. ``log(income) - log(mean(income))``
   or ``substr(city, 1, 3) + "-" + text(age)``. Missing values stay missing (so do the results of
   ``log`` of numbers ≤ 0, and infinities).

   :param net: A :class:`~graphoskop.Network`.
   :param formula: The formula.
   :return: :class:`pandas.Series` of floats (numeric results) or of objects (texts, missing as
      NaN).
   :raises ValueError: If the formula is not valid, names no node attribute, or mixes texts and
      numbers in an operation.
   """
   import pandas as pd
   names = {}
   def quoted(m):
      names[f"__a{len(names)}__"] = m.group(1)
      return f"__a{len(names) - 1}__"
   source = re.sub(r"`([^`]*)`", quoted, formula.strip())
   try:
      tree = ast.parse(source, mode = "eval")
   except SyntaxError as e:
      raise ValueError(f"formula: not complete, or not valid: {formula}") from None
   n = int(sum(net.bipartite_split) if net.higher_order is not None else sum(d.num_nodes for d in net.components))
   known = set(net.node_attributes) | set(net.qualitative_node_attributes)
   def series(x):   # a value as a Series over the nodes
      return x if isinstance(x, pd.Series) else pd.Series([x] * n, dtype = float if isinstance(x, (int, float)) else object)
   def numeric(x, what):
      if isinstance(x, pd.Series) and x.dtype.kind not in "fiub":
         raise ValueError(f"formula: {what} is text: use number(...) to read numbers in it")
      if isinstance(x, str):
         raise ValueError(f"formula: {what} is a text, not a number")
      return x.astype(float) if isinstance(x, pd.Series) else float(x)
   def texts(x):
      if isinstance(x, pd.Series):
         return x.map(lambda v: v if isinstance(v, str) else text(v), na_action = "ignore").astype(object)
      return x if isinstance(x, str) else text(x)
   def clean(x):   # infinities are missing
      return x.replace([np.inf, -np.inf], np.nan) if isinstance(x, pd.Series) and x.dtype.kind == "f" else x
   def is_text(x):
      return isinstance(x, str) or isinstance(x, pd.Series) and x.dtype.kind not in "fiub"
   def flag(x, missing):   # booleans as 1 / 0, missing where an operand is
      x = series(x).astype(float)
      return x.mask(missing) if missing is not None else x
   def isna(*xs):
      m = None
      for x in xs:
         if isinstance(x, pd.Series):
            m = x.isna() if m is None else m | x.isna()
      return m
   def call(f, args):
      if f not in FORMULA_FUNCTIONS:
         raise ValueError(f"formula: unknown function {f}: the functions are {', '.join(FORMULA_FUNCTIONS)}")
      a = args[0] if args else None
      with np.errstate(all = "ignore"):
         if f in ("log", "log2", "log10", "log1p", "exp", "sqrt", "abs", "floor", "ceil"):
            x = numeric(a, f"the value of {f}")
            if f in ("log", "log2", "log10"):
               x = x.where(x > 0) if isinstance(x, pd.Series) else (x if x > 0 else np.nan)
            return clean(getattr(np, {"abs": "abs", "floor": "floor", "ceil": "ceil"}.get(f, f))(x))
         if f == "round":
            return np.round(numeric(a, "the value of round"), int(args[1]) if len(args) > 1 else 0)
         if f in ("min", "max", "mean", "sum") and len(args) == 1:
            x = numeric(a, f"the value of {f}")
            return float(getattr(series(x), f)())
         if f in ("min", "max"):
            return (np.minimum if f == "min" else np.maximum)(series(numeric(a, f)), series(numeric(args[1], f)))
         if f == "where":
            c = series(numeric(a, "the condition"))
            res = series(args[1]).where(c.astype(bool), series(args[2]))
            return res.mask(c.isna())
         if f == "missing":
            return series(a).isna().astype(float)
         if f == "fill":
            return series(a).fillna(args[1])
         if f == "number":
            return pd.to_numeric(series(a), errors = "coerce").astype(float)
         if f == "text":
            return texts(series(a))
         t = texts(series(a))
         if f == "substr":
            start = int(args[1]) - 1 if len(args) > 1 else 0
            return t.str.slice(max(start, 0), max(start, 0) + int(args[2]) if len(args) > 2 else None)
         if f in ("upper", "lower"):
            return getattr(t.str, f)()
         if f == "length":
            return t.str.len().astype(float)
         if f == "contains":
            return flag(t.str.contains(texts(args[1]), regex = False).fillna(False), t.isna())
         if f == "replace":
            return t.str.replace(texts(args[1]), texts(args[2]), regex = False)
   def ev(node):
      if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)) and not isinstance(node.value, bool):
         return node.value
      if isinstance(node, ast.Name):
         a = names.get(node.id, node.id)
         if a not in known:
            raise ValueError(f"formula: no node attribute named {a}" + (" (names that are not plain words go in backquotes, e.g. `my name`)" if node.id in names else ""))
         v = node_values(net, a)
         return pd.Series(v, dtype = float) if v.dtype.kind == "f" else pd.Series(v, dtype = object)
      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
         return call(node.func.id, [ev(x) for x in node.args])
      if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
         x = ev(node.operand)
         if isinstance(node.op, ast.Not):
            return flag(series(numeric(x, "the value of not")) == 0, isna(x))
         x = numeric(x, "the value")
         return -x if isinstance(node.op, ast.USub) else x
      if isinstance(node, ast.BinOp):
         a, b = ev(node.left), ev(node.right)
         if isinstance(node.op, ast.Add) and (is_text(a) or is_text(b)):
            if not (is_text(a) and is_text(b)):
               raise ValueError("formula: + joins two texts, or adds two numbers: use text(...) or number(...)")
            return series(texts(a)) + series(texts(b))
         ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.true_divide, ast.Pow: np.power, ast.Mod: np.mod}
         if type(node.op) not in ops:
            raise ValueError("formula: the operations are + - * / ** %")
         with np.errstate(all = "ignore"):
            return clean(ops[type(node.op)](numeric(a, "a value"), numeric(b, "a value")))
      if isinstance(node, ast.Compare) and len(node.ops) == 1:
         a, b = ev(node.left), ev(node.comparators[0])
         if is_text(a) != is_text(b):
            raise ValueError("formula: a comparison of a text with a number: use text(...) or number(...)")
         ops = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge", ast.Eq: "eq", ast.NotEq: "ne"}
         if type(node.ops[0]) not in ops:
            raise ValueError("formula: the comparisons are < <= > >= == !=")
         return flag(getattr(series(a), ops[type(node.ops[0])])(b if not isinstance(b, pd.Series) else b), isna(a, b))
      if isinstance(node, ast.BoolOp):
         xs = [series(numeric(ev(v), "a condition")) for v in node.values]
         res = xs[0] != 0
         for x in xs[1:]:
            res = (res & (x != 0)) if isinstance(node.op, ast.And) else (res | (x != 0))
         return flag(res, isna(*xs))
      raise ValueError(f"formula: not understood: {ast.unparse(node)}")
   result = series(ev(tree.body))
   return result.astype(float) if result.dtype.kind in "fiub" else result


def derived(net, formula):
   """
   Values of a new node attribute computed from others by a formula (:func:`formula_values`),
   one vector per component, for :meth:`~graphoskop.Network.set_node_attribute` with component
   ``None``, e.g. ``net.set_node_attribute("ratio", derived(net, "income / age"), None)``
   (Edit › New node attribute). A function of the interface, defined in the namespace of every
   tab (:data:`DERIVED_NOTE`).

   :param net: A :class:`~graphoskop.Network`.
   :param formula: The formula.
   :return: List of float arrays (numeric values) or object arrays (texts).
   """
   v = formula_values(net, formula)
   v = v.to_numpy(dtype = float) if v.dtype.kind == "f" else v.to_numpy(dtype = object)
   sizes = list(net.bipartite_split) if net.higher_order is not None else [d.num_nodes for d in net.components]
   return np.split(v, np.cumsum(sizes)[:-1])


def formula_name(name):
   """
   A node attribute name as written in a formula: as it is if it is a plain word (not a function
   of :data:`FORMULA_FUNCTIONS`), in backquotes otherwise.
   """
   return name if name.isidentifier() and not keyword.iskeyword(name) and name not in FORMULA_FUNCTIONS else f"`{name}`"


def formula_summary(values):
   """
   What the dialog of Edit › New node attribute shows of the values of a formula before they are
   created: their kind, range or distinct values, missing values, and the first ones.
   """
   missing = int(values.isna().sum())
   known = values.dropna()
   first = ", ".join(text(v) for v in known.head(5))
   if values.dtype.kind == "f":
      what = f"numbers from {text(known.min())} to {text(known.max())}" if len(known) else "numbers, all missing"
   else:
      what = f"texts, {known.astype(str).nunique():,} distinct" if len(known) else "texts, all missing"
   return f"{what}; {missing:,} of {len(values):,} missing" + (f" (e.g. {first})" if first else "")


def _derive(request):
   """
   Command ``derive``: a new node attribute ``name`` of the network of a tab from a ``formula``
   (:func:`derived`), written as ``net.set_node_attribute(name, derived(net, formula), None)``; an
   existing attribute of that name is replaced. With ``preview``, only checks the formula and
   replies its :func:`formula_summary` as ``text``.
   """
   ns = _spaces[request["tab"]]
   net = ns["net"]
   name, formula = str(request.get("name", "")).strip(), str(request.get("formula", "")).strip()
   if not formula:
      raise ValueError("formula: write the formula of the values, e.g. log(age)")
   values = formula_values(net, formula)
   exists = name in set(net.node_attributes) | set(net.qualitative_node_attributes)
   if exists and (name in net.node_attributes) != (values.dtype.kind == "f"):   # the library keeps the kind of an attribute
      raise ValueError(f"name: {name} is {'numeric' if name in net.node_attributes else 'qualitative'}, and the formula gives {'numbers' if values.dtype.kind == 'f' else 'texts'}: choose another name, or delete {name} first")
   if request.get("preview"):
      return {"text": formula_summary(values) + (f"; replaces the attribute {name}" if exists else "")}, b""
   if not name:
      raise ValueError("name: the new attribute needs a name")
   code = _run(request["tab"], [f"net.set_node_attribute({literal(name)}, derived(net, {literal(formula)}), None)"])
   header, payload = _draw(ns, code, True)
   header["note"] = f"node attribute {name}: {formula_summary(values)}"
   return header, payload


def _formulas(request):
   """
   Command ``formulas``: what the dialog of Edit › New node attribute offers: the node attribute
   names as written in formulas (:func:`formula_name`) and the functions
   (:data:`FORMULA_FUNCTIONS`).
   """
   net = _spaces[request["tab"]]["net"]
   names = list(net.node_attributes) + list(net.qualitative_node_attributes)
   return {"attributes": [{"name": a, "formula": formula_name(a), "numeric": a in net.node_attributes} for a in names], "functions": [{"name": f, "doc": d} for f, d in FORMULA_FUNCTIONS.items()]}, b""


#: Comment of the code before the first :func:`node_with`, a function of the interface.
NODE_WITH_NOTE = "# node_with: the id of the node with a value of a node attribute, a function of the graphical interface"


def node_with(net, attribute, value):
   """
   The id of the one node whose node attribute ``attribute`` has the value ``value`` (compared as
   texts, :func:`text`: ``3`` and ``"3"`` are the same), e.g. the source of an exploration given
   by name: ``gsk.bfs(net, node_with(net, "name", "alice"))``. A function of the interface,
   defined in the namespace of every tab (:data:`NODE_WITH_NOTE`).

   :raises ValueError: If no node, or more than one, has that value.
   """
   import pandas as pd
   wanted = text(value)
   found = np.flatnonzero(pd.Series(node_values(net, attribute)).map(text, na_action = "ignore").to_numpy(dtype = object) == wanted)
   if len(found) != 1:
      raise ValueError(f"no node has {attribute} = {wanted}" if len(found) == 0 else f"{len(found)} nodes have {attribute} = {wanted} (e.g. {', '.join(text(i) for i in _ids(net)[found[:3]])}): choose an attribute with one node per value")
   return _ids(net)[found[0]].item() if hasattr(_ids(net)[found[0]], "item") else _ids(net)[found[0]]


#: Explorations of Simulation › Exploration: function to its label and the default name of the
#: node attribute written.
EXPLORATIONS = {"bfs": ("Breadth-first search", "bfs_order"), "dfs": ("Depth-first search", "dfs_order"), "random_walk": ("Random walk", "rw_step"), "metropolis_hastings_random_walk": ("Metropolis-Hastings random walk", "mhrw_step")}


for _f in ("bfs", "dfs", "random_walk", "metropolis_hastings_random_walk"):
   OPTIONS[(_f, "direction")] = ["\"out\"", "\"in\"", "\"both\""]
   HINTS[(_f, "source")] = "required (a node id)"
   INTEGERS.update({(_f, "length"): 0, (_f, "walks"): 1})
NODES.add("source")
HINTS[("bfs", "max_depth")] = "all"


def exploration_signature(function, net):
   """
   Parameters of the dialog of an exploration (:data:`EXPLORATIONS`), as :func:`signature`, after
   ``attribute`` (the node attribute written), with ``by`` after ``source`` (a node attribute
   whose value is the source, :func:`node_with`); ``direction`` only for directed networks; the
   ``seed`` of the walks is 1 by default, so that the code gives the same result when it runs
   again.
   """
   params = [p for p in signature(function, net)["params"] if p["name"] != "direction" or net is not None and net.directed]
   for p in params:
      if p["name"] == "seed":
         p["default"] = "1"
   what = "the order of the visit (0 for the source)" if function in ("bfs", "dfs") else "the step of the first visit, averaged over the walks reaching the node"
   params.insert(0, {"name": "attribute", "default": literal(EXPLORATIONS[function][1]), "hint": "", "doc": f"Name of the node attribute written: {what}; nodes not reached get no value", "widget": {"kind": "string"}})
   at = next(i for i, p in enumerate(params) if p["name"] == "source") + 1
   params[at - 1]["widget"]["lookup"] = "by"   # the field suggests the values of the attribute of by (or the ids), see _suggest
   params.insert(at, {"name": "by", "default": "", "hint": "the node id", "doc": "Node attribute whose value is given as source (e.g. a name), instead of the node id: exactly one node must have that value (node_with)", "widget": {"kind": "attribute", "choices": "nodes", "multiple": False, "extra": []}})
   return {"params": params}


def exploration_code(ns, function, arguments):
   """
   Code of an exploration of Simulation › Exploration (:data:`EXPLORATIONS`) from the node
   ``source``, writing one node attribute (``attribute``, by default that of
   :data:`EXPLORATIONS`): for the searches (:func:`~graphoskop.bfs`, :func:`~graphoskop.dfs`),
   the order of the visit (0 for the source); for the walks (:func:`~graphoskop.random_walk`,
   :func:`~graphoskop.metropolis_hastings_random_walk`, ``walks`` of ``length`` steps, all from
   the source; seed 1 unless the dialog says otherwise), the step of the first visit of each node,
   averaged over the walks that reach it (for Metropolis-Hastings walks, steps count the refused
   moves too: the sum of the ``duration`` of the visits before). Nodes never reached get no value.
   For example:

   .. code-block:: python

      exploration = gsk.random_walk(net, "alice", 100, walks = 10, seed = 1)
      first = exploration.groupby(["walk", "node"]).step.min()   # first visit of each node in each walk
      net.set_node_attribute(first.groupby(level = "node").mean().rename("rw_step").reset_index(), node_column = "node")

   :return: Lines of code, and the name of the attribute.
   :raises ValueError: If there is no source, or the network is multilayer or higher-order.
   """
   if function not in EXPLORATIONS:
      raise ValueError(f"unknown exploration: {function}")
   net = ns["net"]
   if net.layer_labels is not None or net.higher_order is not None:
      raise ValueError("explorations here are for plain networks (not multilayer networks, hypergraphs or simplicial complexes): see gsk.random_walk for those")
   arguments = {k: v for k, v in dict(arguments).items() if str(v).strip()}
   if function not in ("bfs", "dfs"):
      arguments = dict({"seed": "1"}, **arguments)
   name = ast.literal_eval(arguments.pop("attribute", literal(EXPLORATIONS[function][1])))
   source = arguments.pop("source", "")
   if not source:
      raise ValueError("source: choose the node the exploration starts from")
   by = arguments.pop("by", "")
   if by:   # the source given by the value of a node attribute (checked here: one node has it)
      node_with(net, ast.literal_eval(by), ast.literal_eval(source))
      source = f"node_with(net, {by}, {source})"
   args = [source] + ([arguments.pop("length")] if "length" in arguments else [])
   if function not in ("bfs", "dfs") and len(args) < 2:
      raise ValueError("length: give the number of steps of each walk")
   lines = _pandas(ns) + [f"exploration = gsk.{function}(net, {', '.join(args)}" + "".join(f", {k} = {v}" for k, v in arguments.items()) + ")"]
   if function in ("bfs", "dfs"):
      lines.append(f"net.set_node_attribute(pd.DataFrame({{\"node\": exploration.node, {literal(name)}: range(len(exploration))}}), node_column = \"node\")   # order of the visit, 0 for the source")
   else:
      when = "step"
      if function == "metropolis_hastings_random_walk":
         lines.append("exploration[\"time\"] = exploration.groupby(\"walk\").duration.cumsum() - exploration.duration   # steps before each arrival, refused moves included")
         when = "time"
      lines += [f"first = exploration.groupby([\"walk\", \"node\"]).{when}.min()   # first visit of each node in each walk",
                f"net.set_node_attribute(first.groupby(level = \"node\").mean().rename({literal(name)}).reset_index(), node_column = \"node\")   # mean over the walks reaching the node"]
   return lines, name


def _explore(request):
   """
   Command ``explore``: runs :func:`exploration_code` in the namespace of a tab and replies with
   the code, the drawing (unchanged in its look) and a ``note`` on what was reached.
   """
   ns = _spaces[request["tab"]]
   function = request["function"]
   lines, name = exploration_code(ns, function, request.get("arguments") or {})
   code = _run(request["tab"], lines)
   header, payload = _draw(ns, code, True)
   ex, net = ns["exploration"], ns["net"]
   reached = ex.node.nunique()
   what = "order of the visit, 0 for the source" if function in ("bfs", "dfs") else f"mean step of the first visit over {int(ex.walk.max()) + 1} walks"
   header["note"] = f"{EXPLORATIONS[function][0]}: {reached:,} of {net.number_of_nodes():,} nodes reached; node attribute {name} ({what})"
   return header, payload


def _neighbors(request):
   """
   Command ``neighbors`` (Select › Neighbors of the selection, Ctrl+6): the selected ``nodes``
   (positions in the drawing order), the endpoints of the selected ``edges`` and all their
   neighbors (along edges in either direction), as ``nodes``; pressed again, the selection grows
   one more step. ``added`` counts the new ones.
   """
   draw = _drawn(_spaces[request["tab"]])
   s, t = _edges(draw)
   n = len(_ids(draw))
   mask = np.zeros(n, dtype = bool)
   mask[np.asarray(request.get("nodes") or [], dtype = np.int64)] = True
   e = np.asarray(request.get("edges") or [], dtype = np.int64)
   e = e[e < len(s)]
   mask[s[e]] = True
   mask[t[e]] = True
   grown = mask.copy()
   grown[t[mask[s]]] = True
   grown[s[mask[t]]] = True
   return {"nodes": np.flatnonzero(grown).tolist(), "added": int(grown.sum() - mask.sum())}, b""


#: Functions of Analysis › Frequent subgraphs: function to its label (:func:`mining_code`).
MINERS = {"single_graph_mining": "In the whole network…", "transactional_graph_mining": "Across components…"}
for _f in MINERS:
   CHOICES.update({(_f, "node_label"): ("nodes", False), (_f, "edge_label"): ("edges", False)})
   HINTS.update({(_f, "node_label"): "none (all alike)", (_f, "edge_label"): "none (all alike)", (_f, "max_edges"): "no limit"})
INTEGERS[("single_graph_mining", "min_support")] = 1


def mining_signature(function, net):
   """
   Parameters of the dialog of Analysis › Frequent subgraphs (:data:`MINERS`), as
   :func:`signature`, with ``rules`` last (also the association rules).
   """
   params = signature(function, net)["params"]
   params.append({"name": "rules", "default": "False", "hint": "", "doc": "Also the association rules between patterns that differ by one edge (gsk.graph_association_rules), as a table", "widget": {"kind": "bool"}})
   return {"params": params}


def mining_code(ns, function, arguments):
   """
   Code of Analysis › Frequent subgraphs, run in a new tab (:func:`_mine`): the frequent patterns
   of the network (:func:`~graphoskop.single_graph_mining` or
   :func:`~graphoskop.transactional_graph_mining`) become the network of the tab, one component
   per pattern, its nodes with their label and the node attribute ``pattern`` (the id of their
   pattern, the component); the support of every pattern, and with ``rules`` the association rules
   (:func:`~graphoskop.graph_association_rules`), are tables of the Tables tab (``tables``), e.g.

   .. code-block:: python

      patterns, support = gsk.single_graph_mining(net, 20, node_label = "city", max_edges = 3)
      assert patterns is not None, "no pattern reaches the support: lower min_support"
      rules = gsk.graph_association_rules(net, patterns, support, transactional = False)
      net = patterns
      net.set_node_attribute("pattern", [[k] * c.num_nodes for k, c in enumerate(net.components)], None)
      tables = {"Pattern support": support, "Association rules": rules}

   :return: Lines of code.
   :raises ValueError: If ``min_support`` is missing or a value is not a Python expression.
   """
   if function not in MINERS:
      raise ValueError(f"unknown mining function: {function}")
   arguments = {k: v.strip() for k, v in dict(arguments).items() if str(v).strip()}
   for k, v in arguments.items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   rules = ast.literal_eval(arguments.pop("rules", "False"))
   support = arguments.pop("min_support", "")
   if not support:
      raise ValueError("min_support: give the least support of a pattern")
   lines = [f"patterns, support = gsk.{function}(net, {support}" + "".join(f", {k} = {v}" for k, v in arguments.items()) + ")",
            "assert patterns is not None, \"no pattern reaches the support: lower min_support\""]
   tables = "{\"Pattern support\": support" + (", \"Association rules\": rules}" if rules else "}")
   if rules:
      lines.append(f"rules = gsk.graph_association_rules(net, patterns, support, transactional = {function == 'transactional_graph_mining'})")
   lines += ["net = patterns", "net.set_node_attribute(\"pattern\", [[k] * c.num_nodes for k, c in enumerate(net.components)], None)   # the id of the pattern of each node", f"tables = {tables}"]
   return lines + (["del inc"] if "inc" in ns else [])


def _mine(request):
   """
   Command ``mine``: :func:`mining_code` of a tab, in a new tab ``new_tab`` (:func:`_derived`),
   with a ``note`` (how many patterns).
   """
   ns = _spaces[request["tab"]]
   header, payload = _derived(ns, request["new_tab"], mining_code(ns, request["function"], request.get("arguments") or {}))
   new = _spaces[request["new_tab"]]
   header["note"] = f"{len(new['support']):,} frequent patterns, one component each (node attribute pattern); their support" + (" and the association rules" if "rules" in new and "Association rules" in new["tables"] else "") + " in the Tables tab"
   return header, payload


#: Functions of Analysis › Network statistics: global values of the network, each kept as a
#: network attribute (:func:`statistic_code`): function to its label.
STATISTICS = {"density": "Density", "average_degree": "Average degree", "reciprocity": "Reciprocity", "global_clustering": "Global clustering coefficient", "average_clustering": "Average clustering coefficient", "global_triangle_closure": "Global triangle closure", "assortativity": "Assortativity", "average_shortest_path_length": "Average shortest path length", "diameter": "Diameter", "degree_centralization": "Degree centralization", "connected_components_count": "Connected components", "modularity": "Modularity of a node attribute", "is_uniform": "Uniform hypergraph", "is_pure": "Pure complex", "is_pseudomanifold": "Pseudomanifold", "betti_numbers": "Betti numbers", "euler_characteristic": "Euler characteristic"}
#: Network statistics of :data:`STATISTICS` for hypergraphs only.
HYPERGRAPH_STATISTICS = {"is_uniform"}
#: Network statistics of :data:`STATISTICS` for simplicial complexes only.
SIMPLICIAL_STATISTICS = {"is_pure", "is_pseudomanifold", "betti_numbers", "euler_characteristic"}
#: Ways of making classes of a numeric node attribute for its modularity (:func:`statistic_code`),
#: as the placements of color stops (:func:`breaks`).
MODULARITY_METHODS = ["\"quantiles\"", "\"equal\"", "\"natural\""]
#: Numeric node attributes with at most this many distinct values, all integers (e.g. community
#: numbers), are communities as they are, unless stops are given.
MODULARITY_LABELS = 50
#: Parameters of the network statistics not shown in the dialog (memory and processes, not
#: results; precomputed lengths).
STATISTIC_HIDDEN = {"net", "batch_size", "n_jobs", "lengths"}
#: Largest number of nodes for which "All of them" computes the exact average shortest path
#: length: beyond, it samples this many sources.
EXACT_PATHS = 10_000
for _f in STATISTICS:
   OPTIONS.setdefault((_f, "by"), ["\"layer\"", "\"entity\""])
OPTIONS[("average_degree", "direction")] = ["\"out\"", "\"in\"", "\"total\""]
OPTIONS[("degree_centralization", "mode")] = ["\"all\"", "\"in\"", "\"out\""]
OPTIONS[("connected_components_count", "type")] = ["\"weak\"", "\"strong\""]
OPTIONS[("assortativity", "kind")] = ["\"uniform\"", "\"weighted\""]
CHOICES[("assortativity", "attribute")] = ("nodes", False)
HINTS.update({("assortativity", "attribute"): "degree", ("average_shortest_path_length", "samples"): "all nodes", ("density", "component"): "all", ("average_shortest_path_length", "component"): "largest"})
HINTS.update({("diameter", "component"): "largest", ("modularity", "component"): "all", ("modularity", "stops"): "from classes and method"})
CHOICES[("modularity", "partition")] = ("nodes", False)
OPTIONS[("modularity", "overlap")] = ["\"shen\"", "\"nicosia\""]


def statistic_signature(function, net):
   """
   Parameters of the dialog of a network statistic (:data:`STATISTICS`), as :func:`signature`
   without :data:`STATISTIC_HIDDEN`, after ``network_attribute``: the name of the network
   attribute written (the function; for the degree centralization, the prefix ``degree`` of its
   values).
   """
   params = [p for p in signature(function, net)["params"] if p["name"] not in STATISTIC_HIDDEN]
   name = "degree" if function == "degree_centralization" else function
   if function == "modularity":   # the attribute, and classes of numeric values
      for p in params:
         if p["name"] == "partition":
            p.update(default = "", hint = "required", doc = "Node attribute whose values are the communities: qualitative values as they are, numeric ones cut into classes")
      at = 1 + next(i for i, p in enumerate(params) if p["name"] == "partition")
      params[at:at] = [{"name": "classes", "default": "4", "hint": "", "doc": "Number of classes of a numeric attribute", "widget": {"kind": "int", "min": 2}},
                       {"name": "method", "default": MODULARITY_METHODS[0], "hint": "", "doc": "How the classes of a numeric attribute are cut: at its quantiles (classes of about as many nodes), evenly spaced from the smallest to the largest value, or at its natural breaks", "widget": {"kind": "choice", "options": MODULARITY_METHODS}},
                       {"name": "stops", "default": "", "hint": "from classes and method", "doc": "Values separating the classes of a numeric attribute, separated by commas (e.g. 30, 40: below 30, from 30 to 40, from 40 on), instead of classes and method", "widget": {"kind": "list"}}]
   params.insert(0, {"name": "network_attribute", "default": literal(name) if function != "modularity" else "", "hint": "modularity(attribute)" if function == "modularity" else "", "doc": "Name of the network attribute written" + (" (prefix of its values: degree_centralization, degree_entropy, …)" if function == "degree_centralization" else ""), "widget": {"kind": "string"}})
   return {"params": params}


def statistic_code(function, arguments = None, ns = None):
   """
   Code of a network statistic (:data:`STATISTICS`), its value kept as a network attribute (shown
   in the Network tab of the data panel), e.g.
   ``net.network_attributes["global_clustering"] = gsk.global_clustering(net)``; the degree
   centralization keeps each of its values, e.g. ``degree_centralization`` and ``degree_entropy``.
   Statistics that take precomputed shortest path lengths get those of Analysis › Shortest paths
   when they match (:func:`paths_argument`), e.g. ``gsk.diameter(net, lengths = paths[0])``.

   The modularity of a node attribute (``partition``) takes qualitative values as communities
   (``gsk.modularity(net, "city")``), as well as numeric ones with at most
   :data:`MODULARITY_LABELS` distinct integers (e.g. community numbers), and cuts other numeric
   ones into classes: at ``stops``, or at the
   values that make ``classes`` classes with ``method`` (:func:`breaks`), written in the code;
   missing values make a class of their own, e.g.

   .. code-block:: python

      x = np.concatenate([v.cpu().numpy() for v in net.get_node_attribute("age", None)])   # values of age
      classes = np.where(np.isnan(x), -1, np.digitize(x, [30.0, 40.0]))   # classes of age: below 30, from 30 to 40, from 40 on; missing values apart
      net.network_attributes["modularity(age)"] = gsk.modularity(net, classes)   # a network attribute

   :param ns: Namespace of the tab (for the lengths, the numeric values and the imports), or
      ``None``.
   :return: Lines of code, and the names of the network attributes.
   """
   if function not in STATISTICS:
      raise ValueError(f"unknown network statistic: {function}")
   arguments = {k: v.strip() for k, v in dict(arguments or {}).items() if str(v).strip()}
   for k, v in arguments.items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   if function == "modularity":
      return modularity_code(arguments, ns)
   name = ast.literal_eval(arguments.pop("network_attribute", literal("degree" if function == "degree_centralization" else function)))
   call = f"gsk.{function}(net" + "".join(f", {k} = {v}" for k, v in arguments.items()) + (paths_argument(ns, function, arguments) if ns is not None else "") + ")"
   if function == "degree_centralization":
      return [f"net.network_attributes.update({{f\"{name}_{{k}}\": v for k, v in {call}.items()}})   # network attributes"], None
   return [f"net.network_attributes[{literal(name)}] = {call}   # a network attribute"], name


def modularity_code(arguments, ns):
   """
   Code of the modularity of a node attribute (see :func:`statistic_code`).
   """
   arguments = dict(arguments)
   attribute = ast.literal_eval(arguments.pop("partition", "\"\"") or "\"\"")
   if not attribute:
      raise ValueError("partition: choose the node attribute whose values are the communities")
   k = int(arguments.pop("classes", "4"))
   method = ast.literal_eval(arguments.pop("method", MODULARITY_METHODS[0]))
   stops = arguments.pop("stops", "").strip()
   name = ast.literal_eval(arguments.pop("network_attribute", "\"\"") or "\"\"") or f"modularity({attribute})"
   extra = "".join(f", {a} = {v}" for a, v in arguments.items())
   net = _drawn(ns) if ns is not None else None
   values = node_values(net, attribute) if net is not None and attribute in net.node_attributes else None
   distinct = np.unique(values[~np.isnan(values)]) if values is not None else None
   if values is None or not stops and len(distinct) <= MODULARITY_LABELS and np.all(distinct == np.round(distinct)):   # qualitative, or a few integers (community labels): its values
      return [f"net.network_attributes[{literal(name)}] = gsk.modularity(net, {literal(attribute)}{extra})   # a network attribute"], name
   if stops:
      cuts = sorted(float(v) for v in ast.literal_eval(f"[{stops.strip('()[]')}]"))
   else:
      if k < 2:
         raise ValueError("classes: at least 2")
      cuts = breaks(values, method, k + 1)[1:-1]
   cuts = list(dict.fromkeys(float(f"{c:.6g}") for c in cuts))
   if not cuts:
      raise ValueError("stops: give at least one value separating the classes")
   parts = [f"below {text(cuts[0])}"] + [f"from {text(a)} to {text(b)}" for a, b in zip(cuts[:-1], cuts[1:])] + [f"from {text(cuts[-1])} on"]
   lines = [] if any(l.startswith("import numpy as np") for l in ns["__code__"]) else ["import numpy as np"]
   return lines + [f"x = np.concatenate([v.cpu().numpy() for v in net.get_node_attribute({literal(attribute)}, None)])   # values of {attribute}",
                   f"classes = np.where(np.isnan(x), -1, np.digitize(x, {cuts!r}))   # classes of {attribute}: {', '.join(parts)}; missing values apart",
                   f"net.network_attributes[{literal(name)}] = gsk.modularity(net, classes{extra})   # a network attribute"], name


def all_statistics(net):
   """
   The network statistics of Analysis › Network statistics › All of them, with their defaults,
   that apply to ``net``: reciprocity for directed networks, triangle closure for hypergraphs and
   simplicial complexes, uniformity for hypergraphs (:data:`HYPERGRAPH_STATISTICS`), purity,
   pseudomanifold, Betti numbers and Euler characteristic for complexes
   (:data:`SIMPLICIAL_STATISTICS`), and for those only the statistics the library computes for them
   (the others refuse them), the average shortest path length sampled from :data:`EXACT_PATHS`
   sources in larger networks, the diameter of directed networks only up to that many nodes
   (undirected ones need a handful of searches), not the modularity (it needs an attribute).

   :return: List of ``(function, arguments)``.
   """
   chosen = []
   for f in STATISTICS:
      if f == "reciprocity" and not net.directed or f == "global_triangle_closure" and net.higher_order is None or f == "modularity" or f == "diameter" and net.directed and net.number_of_nodes() > EXACT_PATHS or f in HYPERGRAPH_STATISTICS and net.higher_order != "hypergraph" or f in SIMPLICIAL_STATISTICS and net.higher_order != "simplicial" or net.higher_order is not None and not getattr(gsk, f)._higher_order[net.higher_order]:
         continue
      args = {"samples": str(EXACT_PATHS)} if f == "average_shortest_path_length" and net.number_of_nodes() > EXACT_PATHS else {}
      chosen.append((f, args))
   return chosen


def _statistics(request):
   """
   Command ``statistics``: a network statistic of a tab (``function``, ``arguments``;
   :func:`statistic_code`), or all of them (``function`` ``"all"``, :func:`all_statistics`: each
   runs on its own, those the library refuses are skipped and named). Replies with the code, the
   drawing (unchanged) and ``result``, the values.
   """
   ns = _spaces[request["tab"]]
   net = ns["net"]
   jobs = all_statistics(net) if request["function"] == "all" else [(request["function"], request.get("arguments") or {})]
   code, done, skipped = None, [], []
   for f, args in jobs:
      lines, _ = statistic_code(f, args, ns)
      before = dict(ns["net"].network_attributes)
      try:
         code = _run(request["tab"], lines)
      except Exception as e:
         if request["function"] != "all":
            raise
         skipped.append(f"{STATISTICS[f].lower()} ({str(e).splitlines()[0][:80]})")
         continue
      net = ns["net"]
      done += [k for k, v in net.network_attributes.items() if k not in before or before[k] is not v]   # written by this run
   header, payload = _draw(ns, code if code is not None else "\n".join(ns["__code__"]), True)
   values = net.network_attributes
   shown = list(dict.fromkeys(done))
   lines = [f"{k} = {text(values[k])}" for k in shown if k in values] or ["none"]
   header["result"] = "Network attributes of %s:\n%s" % (request.get("name", "the network"), "\n".join(lines)) + ("" if not skipped else "\n\nSkipped:\n" + "\n".join(skipped))   # one value per line
   return header, payload


def _image_pdf(request):
   """
   Command ``image_pdf`` (File › Export › Image…, as a PDF document): the PNG image ``png`` (saved
   by the interface) written as the one page of the PDF document ``path``, at ``dpi`` dots per
   inch (the page is as large as the image at that resolution). Replies with the ``size`` of the
   file.
   """
   from PIL import Image
   with Image.open(request["png"]) as image:
      image.convert("RGB").save(request["path"], "PDF", resolution = float(request.get("dpi", 300)))
   return {"size": os.path.getsize(request["path"])}, b""


#: Smallest sizes on screen, in pixels, as in the visualization panel (network_view.gd): node
#: diameters, default edge width, arrowhead length; the default edge width relative to the
#: default node diameter, the shape of the arrowheads (length and breadth relative to the edge
#: width), the smallest label drawn and the most labels.
VIEW_SIZES = {"node_pixels": 3., "edge_pixels": 1., "arrow_pixels": 5., "edge_fraction": 1. / 6., "arrow_length": 2.5, "arrow_breadth": 2.2, "label_pixels": 4., "max_labels": 2000}


def vector_image(ns, view, path, width, dpi = 300, background = "#ffffff"):
   """
   The drawing of a tab as the visualization panel shows it, as vector graphics (a PDF document
   by default, or any format matplotlib writes from the extension of ``path``): nodes as circles,
   straight edges as lines ending at their arrowheads, bundled edges as Bézier curves, matrix
   cells as squares, self loops as rings, arrowheads as triangles and labels as text, in the
   colors, sizes and opacities of the style, with the smallest sizes on screen of
   :data:`VIEW_SIZES` (as in the panel, so that the document looks as the screen), edges under
   loops and arrowheads, under nodes, under labels. Only what is in view is visible; the selection
   marks are left out.

   :param ns: Namespace of the tab.
   :param view: The view: ``center`` (world point at its center), ``zoom`` (pixels per world unit,
      the scale of the layout included), ``turn`` (degrees, counterclockwise), ``spread`` (scale
      of the layout), ``size`` (width and height in pixels).
   :param path: File to write.
   :param width: Width of the page in pixels at ``dpi`` dots per inch (the height follows).
   :param background: Background color.
   """
   import matplotlib
   matplotlib.use("Agg")
   from matplotlib.figure import Figure
   from matplotlib.collections import EllipseCollection, LineCollection, PolyCollection, PathCollection, PatchCollection
   from matplotlib.patches import Wedge
   from matplotlib.path import Path
   from matplotlib import patheffects
   from matplotlib.colors import to_rgb
   draw = ns.get("inc", ns["net"])
   last = last_layout(ns["__code__"])
   matrix = _matrix_attribute(ns, last) if last == "matrix_layout" else None
   if "pos_x" not in draw.node_attributes and matrix is None or draw.higher_order is not None:
      raise ValueError("the network has no drawing yet")
   style = ns["__style__"]
   bends = ns.get("bends") if last in CURVES and not _positioned(ns) else None
   d = drawing(draw, ns.get("__size__"), style, bends = bends, matrix = matrix)
   V = VIEW_SIZES
   W, H = (float(v) for v in view["size"])
   zoom, spread = float(view["zoom"]), float(view.get("spread", 1.))
   cx, cy = (float(v) for v in view["center"])
   phi = -np.deg2rad(float(view.get("turn", 0.)))
   cos, sin = np.cos(phi), np.sin(phi)
   def screen(x, y):   # world points (y down) to pixels of the view
      u, v = (np.asarray(x, dtype = float) - cx) * zoom, (np.asarray(y, dtype = float) - cy) * zoom
      return u * cos - v * sin + W / 2, u * sin + v * cos + H / 2
   def rotate(x, y):   # world directions to directions on screen
      return x * cos - y * sin, x * sin + y * cos
   inch = float(width) / W / float(dpi)   # inches per pixel of the view
   pt = 72. * inch                         # points per pixel
   fig = Figure(figsize = (W * inch, H * inch))
   fig.patch.set_facecolor(background)
   ax = fig.add_axes([0, 0, 1, 1])
   ax.set_xlim(0, W)
   ax.set_ylim(H, 0)
   ax.set_aspect("equal")
   ax.set_axis_off()
   N, E, L, A = d["nodes"], d["edges"], d["loops"], d["arrows"]
   size = d["size"]
   line = max(size * V["edge_fraction"] / spread, V["edge_pixels"] / zoom)   # default edge width (world)
   radius = np.maximum(N[:, 12] / spread, V["node_pixels"] / zoom) / 2 if len(N) else np.zeros(0)
   if d["cells"]:   # matrix cells: unit squares
      x0, y0 = E[:, 3] - 0.5, E[:, 7] - 0.5
      corners = [screen(x0 + a, y0 + b) for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))]
      polys = np.stack([np.column_stack(c) for c in corners], axis = 1)
      ax.add_collection(PolyCollection(polys, facecolors = E[:, 8:12], edgecolors = "none", zorder = 1))
   elif d["curved"]:   # Bézier pieces
      keep = E[:, 14] > 0
      pts = [screen(E[keep, i], E[keep, j]) for i, j in ((3, 7), (0, 4), (1, 5), (12, 13))]
      codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
      paths = [Path(np.array([[p[0][k], p[1][k]] for p in pts]), codes) for k in range(int(keep.sum()))]
      ax.add_collection(PathCollection(paths, facecolors = "none", edgecolors = E[keep, 8:12], linewidths = line * zoom * E[keep, 14] * pt, transform = ax.transData, capstyle = "butt", zorder = 1))
   elif len(E):   # straight lines, ending at their arrowheads
      w = np.hypot(E[:, 1], E[:, 5])
      dx, dy = E[:, 0], E[:, 4]
      length = np.maximum(np.hypot(dx, dy), 1e-12)
      sx, sy, tx, ty = E[:, 3] - dx / 2, E[:, 7] - dy / 2, E[:, 3] + dx / 2, E[:, 7] + dy / 2
      a = b = np.zeros(len(E))
      if E.shape[1] > EDGE_STRIDE:   # directed: cut at the arrowheads
         head = np.maximum(V["arrow_length"] * line * w, V["arrow_pixels"] / zoom)
         a = E[:, 14] * (np.maximum(E[:, 12] / spread, V["node_pixels"] / zoom) / 2 + head)
         b = E[:, 15] * (np.maximum(E[:, 13] / spread, V["node_pixels"] / zoom) / 2 + head)
      keep = (w > 0) & (a + b < length)
      ux, uy = dx / length, dy / length
      p, q = screen(sx + a * ux, sy + a * uy), screen(tx - b * ux, ty - b * uy)
      segments = np.stack([np.column_stack(p), np.column_stack(q)], axis = 1)[keep]
      ax.add_collection(LineCollection(segments, colors = E[keep, 8:12], linewidths = line * zoom * w[keep] * pt, capstyle = "butt", zorder = 1))
   if len(L):   # self loops: rings above their nodes
      r = L[:, 12] / 2 / spread
      h = L[:, 13] / 2 / spread + r * 0.5
      ox, oy = rotate(np.zeros(len(L)), -h)
      x, y = screen(L[:, 3], L[:, 7])
      ax.add_collection(EllipseCollection(2 * r * zoom, 2 * r * zoom, np.zeros(len(L)), units = "xy", offsets = np.column_stack([x + ox * zoom, y + oy * zoom]), offset_transform = ax.transData, facecolors = "none", edgecolors = L[:, 8:12], linewidths = line * zoom * L[:, 14] * pt, zorder = 2))
   if len(A):   # arrowheads: tip at the border of the node
      r = np.maximum(A[:, 12] / spread, V["node_pixels"] / zoom) / 2
      head = np.maximum(V["arrow_length"] * line * A[:, 13], V["arrow_pixels"] / zoom)
      ux, uy = A[:, 0], A[:, 4]
      nx, ny = -uy, ux
      half = head * V["arrow_breadth"] / V["arrow_length"] / 2
      tip = (A[:, 3] - r * ux, A[:, 7] - r * uy)
      base = (A[:, 3] - (r + head) * ux, A[:, 7] - (r + head) * uy)
      corners = [screen(*tip), screen(base[0] + half * nx, base[1] + half * ny), screen(base[0] - half * nx, base[1] - half * ny)]
      polys = np.stack([np.column_stack(c) for c in corners], axis = 1)
      ax.add_collection(PolyCollection(polys, facecolors = A[:, 8:12], edgecolors = "none", zorder = 2))
   if len(N):   # nodes
      x, y = screen(N[:, 3], N[:, 7])
      ax.add_collection(EllipseCollection(2 * radius * zoom, 2 * radius * zoom, np.zeros(len(N)), units = "xy", offsets = np.column_stack([x, y]), offset_transform = ax.transData, facecolors = N[:, 8:12], edgecolors = "none", zorder = 3))
      offsets, counts, table = chart_slices(draw, style.get("node_chart")) if matrix is None else (None, None, [])
      if len(table):   # node charts: wedges over the discs, from the top, clockwise on the page (y down)
         wedges, faces = [], []
         for i in np.flatnonzero(counts):
            a = 0.
            for k in range(int(offsets[i]), int(offsets[i] + counts[i])):
               b = float(table[k, 0])
               wedges.append(Wedge((x[i], y[i]), radius[i] * zoom, -90. + 360. * a, -90. + 360. * b))
               faces.append((*table[k, 1:], N[i, 11]))
               a = b
         ax.add_collection(PatchCollection(wedges, facecolors = faces, edgecolors = "none", zorder = 3))
      texts = node_labels(draw, style).decode("utf-8").split("\x1f") if matrix is None else []
      if len(texts) == len(N):   # labels, centered, as in the panel
         heights = np.frombuffer(label_sizes(draw, style, size), dtype = np.float32) * zoom / spread
         inside = (x > -40) & (x < W + 40) & (y > -40) & (y < H + 40) & (np.array([bool(t) for t in texts]))
         if inside.sum() <= V["max_labels"]:
            dark = float(np.dot(to_rgb(background), [0.2126, 0.7152, 0.0722])) > 0.5   # a light background: dark text
            ink, halo = ("black", (1, 1, 1, 0.8)) if dark else ("white", (0, 0, 0, 0.8))
            for i in np.flatnonzero(inside & (heights >= V["label_pixels"])):
               px = float(heights[i])
               ax.text(x[i], y[i], texts[i], ha = "center", va = "center_baseline", fontsize = px * pt, color = ink, clip_on = True, zorder = 4, path_effects = [patheffects.withStroke(linewidth = max(2., px / 5) * pt * 0.6, foreground = halo)])
   fig.savefig(path, facecolor = background)


def _image_vector(request):
   """
   Command ``image_vector`` (File › Export › Image…, as a vector PDF document): the drawing of a
   tab as the ``view`` of the interface shows it (:func:`vector_image`), ``width`` pixels wide at
   ``dpi`` dots per inch, on the ``background`` color, written to ``path``. Replies with the
   ``size`` of the file.
   """
   vector_image(_spaces[request["tab"]], request["view"], request["path"], float(request["width"]), float(request.get("dpi", 300)), request.get("background", "#ffffff"))
   return {"size": os.path.getsize(request["path"])}, b""


def _export(request):
   """
   Command ``export`` (File › Export network…): the network of a tab as the tables of
   :meth:`~graphoskop.Network.to_pandas_dataframes`, tab separated, next to ``path``: its name
   without extension, then ``_edges.tsv``, ``_nodes.tsv`` (if there are node attributes) and, for
   multilayer networks, ``_interlayer.tsv``; they load back with File › Load network (multilayer
   networks with their interlayer edges, from code). Replies with the ``files`` written. The code
   does not change.
   """
   import pandas as pd
   net = _spaces[request["tab"]]["net"]
   tables = net.to_pandas_dataframes()
   names = ["edges", "interlayer", "nodes"] if net.layer_labels is not None else ["edges", "nodes"]
   tables = tables if isinstance(tables, tuple) else (tables,)
   base = os.path.splitext(request["path"])[0]
   files = []
   for name, table in zip(names, tables):
      if isinstance(table, pd.DataFrame):
         path = f"{base}_{name}.tsv"
         table.to_csv(path, sep = "\t", index = False)
         files.append(path)
   return {"files": files}, b""


#: Functions of Edit › Error correction: function to its label (:func:`correction_code`).
CORRECTIONS = {"measured_network_correction": "From measurements…", "uncertain_network_correction": "From edge probabilities…"}
#: Models of the true network offered by the error correction dialog.
CORRECTION_MODELS = ["nested_block_model", "degree_corrected_block_model", "stochastic_block_model", "planted_partition_graph", "configuration_model", "gnm_random_graph"]
for _f in CORRECTIONS:
   OPTIONS[(_f, "model")] = [literal(m) for m in CORRECTION_MODELS]
   CHOICES[(_f, "groups")] = ("nodes", False)
   HINTS.update({(_f, "groups"): "fitted", (_f, "seed"): "random"})
CHOICES.update({("measured_network_correction", "measurements"): ("edges", False), ("measured_network_correction", "positives"): ("edges", False), ("uncertain_network_correction", "probability"): ("edges", False)})
HINTS.update({("measured_network_correction", "measurements"): "1 per edge", ("measured_network_correction", "positives"): "1 per edge", ("measured_network_correction", "priors"): "uniform", ("measured_network_correction", "false_negative_rate"): "unknown", ("measured_network_correction", "false_positive_rate"): "unknown"})


def correction_signature(function, net):
   """
   Parameters of the dialog of Edit › Error correction (:data:`CORRECTIONS`), as :func:`signature`
   (the model as a menu of :data:`CORRECTION_MODELS`), with the ``seed`` 1 by default so that the
   code gives the same network when it runs again.
   """
   params = signature(function, net)["params"]
   for p in params:
      if p["name"] == "seed":
         p["default"] = "1"
   return {"params": params}


def correction_code(ns, function, arguments):
   """
   Code of Edit › Error correction, run in a new tab (:func:`_correct`): the network corrected by
   :func:`~graphoskop.measured_network_correction` or
   :func:`~graphoskop.uncertain_network_correction` (seed 1 unless the dialog says otherwise)
   becomes the network of the tab, every node pair with a posterior probability of being an edge
   (edge attribute ``probability``); the scalar values of the summary (model, groups, error rates,
   description length, …) become network attributes; then the thresholds on the probability are
   tested as for a backbone (``bb_test``, ``bb_found``), for the plot of the dialog, e.g.

   .. code-block:: python

      corrected, correction = gsk.measured_network_correction(net, seed = 1)
      net = corrected
      net.network_attributes.update({k: v for k, v in correction.items() if isinstance(v, (int, float, str))})   # network attributes
      bb_test = gsk.test_bb_thresholds(net, quantiles = 100, score = "probability")
      bb_found = gsk.find_bb_thresholds(net, score = "probability")

   :return: Lines of code.
   """
   if function not in CORRECTIONS:
      raise ValueError(f"unknown error correction: {function}")
   arguments = dict({"seed": "1"}, **{k: v.strip() for k, v in dict(arguments).items() if str(v).strip()})
   for k, v in arguments.items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   call = f"gsk.{function}(net" + "".join(f", {k} = {v}" for k, v in arguments.items()) + ")"
   return [f"corrected, correction = {call}", "net = corrected",
           "net.network_attributes.update({k: v for k, v in correction.items() if isinstance(v, (int, float, str))})   # network attributes",
           f"bb_test = gsk.test_bb_thresholds(net, quantiles = {BACKBONE_STEPS}, score = \"probability\")",
           "bb_found = gsk.find_bb_thresholds(net, score = \"probability\")"] + (["del inc"] if "inc" in ns else [])


def _correct(request):
   """
   Command ``correct``: :func:`correction_code` of a tab in a new tab ``new_tab`` (:func:`_derived`),
   replying as ``backbone`` does (:func:`_backbone`: the plot of the thresholds of the edge
   probability is the last part of the payload), so that the backbone dialog extracts the edges
   whose probability reaches a threshold (0.5 by default: the most likely network).
   """
   ns = _spaces[request["tab"]]
   header, payload = _derived(ns, request["new_tab"], correction_code(ns, request["function"], request.get("arguments") or {}))
   new = _spaces[request["new_tab"]]
   new["__backbone__"] = {"score": "probability", "test": new["bb_test"], "found": new["bb_found"]}
   png, box, scale, tails = backbone_plot(new["bb_test"], new["bb_found"], options = {"xlabel": "probability"}, observed = _observed(new))
   x = new["bb_test"]["threshold"].to_numpy()
   found = [[name, float(thr) if np.isfinite(thr) else None] for name, thr in zip(new["bb_found"]["criterion"], new["bb_found"]["threshold"])]
   header["backbone"] = {"score": "probability", "range": [float(x[0]), float(x[-1])], "found": found, "box": box, "scale": scale, "tails": tails}
   values = new["net"].network_attributes
   header["note"] = "corrected network: every pair observed or sampled as an edge, with its probability of being one (edge attribute probability; observed: 1 for the edges of the network, 0 for the pairs it misses); " + ", ".join(f"{k} = {text(v)}" for k, v in values.items())
   header["parts"] = header.get("parts", []) + [len(png)] if header.get("nodes") else [len(png)]
   return header, payload + png


#: Functions of Analysis › Hierarchy (directed networks): function to its label and what its
#: score is (:func:`hierarchy_code`).
HIERARCHIES = {"global_reach_centrality": ("Global reaching centrality", "how much the most reaching node exceeds the others (1: a star out of its center)"),
               "flow_hierarchy": ("Flow hierarchy", "the fraction of the edges in no directed cycle"),
               "arborescence_hierarchy": ("Arborescence hierarchicalness", "the fraction of the edges surviving the reduction to an arborescence"),
               "agony_hierarchy": ("Agony hierarchy", "1 minus the agony of the best levels over the total weight")}
#: Parameters of the hierarchy functions not shown in the dialog (set by :func:`hierarchy_code`).
HIERARCHY_HIDDEN = {"net", "component", "lengths", "n_jobs", "ranks", "arborescence"}
OPTIONS[("global_reach_centrality", "direction")] = ["\"out\"", "\"in\""]
OPTIONS[("flow_hierarchy", "by")] = ["\"edges\"", "\"nodes\""]


def hierarchy_signature(function, net):
   """
   Parameters of the dialog of Analysis › Hierarchy (:data:`HIERARCHIES`), as :func:`signature`
   without :data:`HIDDEN`, with ``all_components`` (the whole network, instead of its largest
   component) and, for the agony, ``levels`` (the name of a node attribute for the levels of the
   nodes, none by default).
   """
   params = [p for p in signature(function, net)["params"] if p["name"] not in HIERARCHY_HIDDEN]
   params.append({"name": "all_components", "default": "False", "hint": "", "doc": "Measure the whole network (component = None) instead of its largest component", "widget": {"kind": "bool"}})
   if function == "agony_hierarchy":
      params.append({"name": "levels", "default": "", "hint": "none", "doc": "Name of a node attribute written with the level of every node (0 at the top; ranks = True)", "widget": {"kind": "string"}})
   return {"params": params}


def hierarchy_code(ns, function, arguments):
   """
   Code of Analysis › Hierarchy: the score of a function of :data:`HIERARCHIES` (directed
   networks), of the largest component or, with ``all_components``, of the whole network, kept in
   ``hierarchy`` and as a network attribute named as the function (``_all_components`` added for
   the whole network), e.g. ``hierarchy = gsk.flow_hierarchy(net, weight = "weight")`` and
   ``net.network_attributes["flow_hierarchy"] = hierarchy``; the agony with
   ``levels`` also writes the levels of the nodes as a node attribute:

   .. code-block:: python

      hierarchy, levels = gsk.agony_hierarchy(net, ranks = True)
      net.set_node_attribute("level", levels)

   :return: Lines of code.
   """
   if function not in HIERARCHIES:
      raise ValueError(f"unknown hierarchy function: {function}")
   arguments = {k: v.strip() for k, v in dict(arguments).items() if str(v).strip()}
   for k, v in arguments.items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   whole = ast.literal_eval(arguments.pop("all_components", "False"))
   levels = ast.literal_eval(arguments.pop("levels", '""'))
   if whole:
      arguments["component"] = "None"
   call = f"gsk.{function}(net" + "".join(f", {k} = {v}" for k, v in arguments.items())
   key = literal(function + ("_all_components" if whole else ""))
   keep = f"net.network_attributes[{key}] = hierarchy   # a network attribute"
   if function == "agony_hierarchy" and levels:
      return [f"hierarchy, levels = {call}, ranks = True)", keep, f"net.set_node_attribute({literal(levels)}, levels{', None' if whole else ''})"]
   return [f"hierarchy = {call})", keep]


def _hierarchy(request):
   """
   Command ``hierarchy``: runs :func:`hierarchy_code` in the namespace of a tab and replies with the
   code, the drawing (unchanged in its look) and the score as ``result`` (the dialog shows it).
   """
   ns = _spaces[request["tab"]]
   function = request["function"]
   arguments = request.get("arguments") or {}
   code = _run(request["tab"], hierarchy_code(ns, function, arguments))
   header, payload = _draw(ns, code, True)
   label, what = HIERARCHIES[function]
   part = "the whole network" if "all_components" in arguments and "True" in arguments["all_components"] else "the largest component"
   levels = ast.literal_eval(arguments.get("levels", '""') or '""')
   header["result"] = f"{label} of {request.get('name', 'the network')} ({part}): {ns['hierarchy']:.4g}, {what}" + (f"; levels in the node attribute {levels}" if levels else "")
   return header, payload


#: Functions of Simulation › Robustness: function to its label and what it removes.
ROBUSTNESS = {"node_removal": ("Node removal", "nodes"), "edge_removal": ("Edge removal", "edges"), "cascade_failures": ("Cascading failures", "nodes")}
#: Parameters of the robustness functions not shown in the parameter dialog.
ROBUSTNESS_HIDDEN = {"net", "component", "progress", "recompute"}
OPTIONS.update({(f, "quality"): ["\"lcc\"", "\"apl\""] for f in ROBUSTNESS})
OPTIONS.update({("cascade_failures", "elements"): ["\"nodes\"", "\"edges\""], ("cascade_failures", "distribution"): ["\"equal\"", "\"random\"", "\"load\"", "\"capacity\""]})
HINTS.update({(f, "samples"): "all (exact)" for f in ROBUSTNESS})


def robustness_signature(function, net):
   """
   Parameters of the dialog of Simulation › Robustness (:data:`ROBUSTNESS`), as :func:`signature`
   without :data:`ROBUSTNESS_HIDDEN`, after ``name`` (the name of the run, by default the function
   and its strategy). The strategy is a menu: ``random``, ``friend_of_random`` (nodes), ``overload``
   (cascades), ``degree`` (largest degrees first, computed if the network has no node attribute
   of that name) and the numeric attributes of the nodes (edges for the edge removal); the load
   and capacity of the cascades are ``random`` or a numeric attribute.
   """
   params = [p for p in signature(function, net)["params"] if p["name"] not in ROBUSTNESS_HIDDEN]
   nodes = [literal(a) for a in (net.node_attributes if net is not None else []) if not a.startswith("pos_")]
   edges = [literal(a) for a in (net.edge_attributes if net is not None else [])]
   fixed = {"node_removal": ["\"random\"", "\"friend_of_random\"", "\"degree\""], "edge_removal": ["\"random\""], "cascade_failures": ["\"random\"", "\"friend_of_random\"", "\"overload\"", "\"degree\""]}[function]
   for p in params:
      if p["name"] == "strategy":
         p["widget"] = {"kind": "choice", "options": list(dict.fromkeys(fixed + (edges if function == "edge_removal" else nodes)))}
         p["doc"] = "Order of the removals: random (failures), friend_of_random (a random neighbor of a random node), degree (largest degrees first), or a numeric attribute (largest values first, or drawn by them with selection weighted)" + (", overload (the items already above their capacity)" if function == "cascade_failures" else "")
      elif p["name"] in ("load", "capacity"):
         p["widget"] = {"kind": "choice", "options": list(dict.fromkeys(["\"random\""] + nodes + edges))}
   params.insert(0, {"name": "name", "default": "", "hint": "function(strategy)", "doc": "Name of the run: its table is robustness[name] and its global value the network attribute name", "widget": {"kind": "string"}})
   return {"params": params}


def robustness_code(ns, function, arguments):
   """
   Code of Simulation › Robustness: a run of a function of :data:`ROBUSTNESS` kept in the dictionary
   ``robustness`` by name (by default the function and the strategy, e.g.
   ``node_removal(degree)``), and its global value as a network attribute of the same name: for
   removals the mean quality over all the numbers of removals (with ``lcc``, the robustness index
   R of Schneider et al., the area under the curve), for cascades the final fraction of failed
   items, e.g. ``robustness["node_removal(degree)"] = gsk.node_removal(net, strategy =
   gsk.degree(net, distribution = None))`` and ``net.network_attributes["node_removal(degree)"] =
   float(robustness["node_removal(degree)"]["lcc"].mean())``. The strategy ``degree`` is the
   degrees of the nodes unless the network has a numeric node attribute of that name.

   :return: Lines of code, the name of the run and its quality.
   """
   if function not in ROBUSTNESS:
      raise ValueError(f"unknown robustness function: {function}")
   arguments = {k: v.strip() for k, v in dict(arguments or {}).items() if str(v).strip()}
   for k, v in arguments.items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   name = ast.literal_eval(arguments.pop("name", "\"\""))
   if not name:   # the function and the strategy, numbered if taken
      base = name = f"{function}({ast.literal_eval(arguments.get('strategy', literal('random')))})"
      k = 1
      while name in (ns.get("robustness") or {}):
         k += 1
         name = f"{base} {k}"
   name = str(name)
   quality = ast.literal_eval(arguments.get("quality", "\"lcc\""))
   t = _target(ns)
   if arguments.get("strategy") == "\"degree\"" and "degree" not in _drawn(ns).node_attributes:
      arguments["strategy"] = f"gsk.degree({t}, distribution = None)"
   lines = [] if any(l.startswith("robustness = {}") for l in ns["__code__"]) and isinstance(ns.get("robustness"), dict) else ["robustness = {}   # runs of Simulation › Robustness, by name"]
   lines.append(f"robustness[{literal(name)}] = gsk.{function}({t}" + "".join(f", {k} = {v}" for k, v in arguments.items()) + ")")
   value = f"robustness[{literal(name)}][{literal(quality)}].mean()" if function != "cascade_failures" else f"robustness[{literal(name)}][\"failed\"].iloc[-1]"
   what = "mean quality over the removals" if function != "cascade_failures" else "fraction of the items failed in the end"
   lines.append(f"net.network_attributes[{literal(name)}] = float({value})   # a network attribute: {what}")
   return lines, name, quality


def robustness_plot(tables, x, y, colors = None, options = None, styles = None, format = "png", path = None, dpi = 100):
   """
   Plot comparing runs of a robustness function: the column ``y`` of every table (the quality, or
   the fraction failed of cascades) against ``x`` (the fraction removed, or the step of cascades;
   ``None``: the index), one line per run with its confidence interval as a band (see
   ``_figure``).

   :param tables: Dictionary from run name to its table.
   """
   def draw(ax):
      for k, (name, df) in enumerate(tables.items()):
         c = (colors or {}).get(name, PLOT_COLORS[k % len(PLOT_COLORS)])
         xs = df[x] if x is not None else df.index
         _series(ax, xs, df[y], (styles or {}).get(name, "lines"), c, name)
         if f"{y}_low" in df and f"{y}_high" in df and df[f"{y}_low"].notna().any():
            ax.fill_between(np.asarray(xs, dtype = float), df[f"{y}_low"], df[f"{y}_high"], color = c, alpha = 0.2, lw = 0)
   return _figure(draw, dict(PLOT_OPTIONS["robustness"], **(options or {})), format, path, dpi)


PLOT_ELEMENTS["robustness"] = (lambda **kw: robustness_plot(**kw), {}, {})
PLOT_ELEMENTS["dendrogram"] = (lambda **kw: dendrogram_plot(**kw), {"tree": "#377eb8", "leaves": "#e41a1c"}, {"tree": "Tree", "leaves": "Leaves"})
PLOT_OPTIONS["dendrogram"] = {"xlog": False, "ylog": False, "font_size": 14, "xlabel": "", "ylabel": "", "xmin": None, "xmax": None, "ymin": None, "ymax": None}
PLOT_OPTIONS["robustness"] = {"xlog": False, "ylog": False, "font_size": 14, "xlabel": "", "ylabel": "", "xmin": None, "xmax": None, "ymin": None, "ymax": None}
#: Labels of the qualities of the robustness functions, for the axes of their plots.
ROBUSTNESS_QUALITIES = {"lcc": "fraction of the nodes in the largest component", "apl": "average shortest path length"}


def robustness_runs(ns):
   """
   The runs of Simulation › Robustness of a tab (the dialog lists them): ``tables``, their names in
   order; ``models``, run name to its ``function`` and ``arguments`` (as typed, for Copy…; the name
   left out) and ``quality``; ``kinds``, run name to ``"removal"`` or ``"cascade"``.
   """
   meta = ns.setdefault("__robustness__", {})
   for n, m in list(meta.items()):   # sessions of 0.13.40: (function, quality)
      if isinstance(m, tuple):
         meta[n] = {"function": m[0], "arguments": {}, "quality": m[1]}
   runs = ns.get("robustness") if isinstance(ns.get("robustness"), dict) else {}
   names = [n for n in runs if n in meta]
   return {"tables": names, "models": {n: {"function": meta[n]["function"], "arguments": meta[n]["arguments"], "quality": meta[n]["quality"]} for n in names}, "kinds": {n: "cascade" if meta[n]["function"] == "cascade_failures" else "removal" for n in names}}


def _robustness(request):
   """
   Command ``robustness``: runs :func:`robustness_code` in the namespace of a tab (a run of the
   robustness dialog), and replies with the code, the drawing (unchanged in its look), the global
   value as ``result`` and the runs (:func:`robustness_runs`).
   """
   ns = _spaces[request["tab"]]
   function = request["function"]
   arguments = dict(request.get("arguments") or {})
   lines, name, quality = robustness_code(ns, function, arguments)
   code = _run(request["tab"], lines)
   arguments.pop("name", None)
   ns.setdefault("__robustness__", {})[name] = {"function": function, "arguments": arguments, "quality": quality}
   header, payload = _draw(ns, code, True)
   value = ns["net"].network_attributes.get(name)
   header["result"] = f"{name} = {text(value)}" + (f" (final {ROBUSTNESS_QUALITIES[quality]}: {text(float(ns['robustness'][name][quality].iloc[-1]))})" if function == "cascade_failures" else "")
   header.update(robustness_runs(ns))
   return header, payload


def _robustness_runs(request):
   """
   Command ``robustness_runs``: the runs of the robustness dialog of a tab (:func:`robustness_runs`).
   """
   return robustness_runs(_spaces[request["tab"]]), b""


def _robustness_plot(request):
   """
   Command ``robustness_plot``: the plot comparing the runs ``names`` of a tab
   (:func:`robustness_plot`): the quality against the fraction removed for removals of nodes or
   edges, the fraction of items failed at every step for cascades. Runs of removals and cascades,
   or of two qualities, do not share axes and are refused. Replies with the PNG image as payload,
   the ``elements`` of the plot window and ``text`` (the global value of every run).
   """
   ns = _spaces[request["tab"]]
   info = robustness_runs(ns)
   names = [n for n in request.get("names") or [] if n in info["models"]]
   if not names:
      raise ValueError("check the runs to plot")
   kinds = {info["kinds"][n] for n in names}
   qualities = {info["models"][n]["quality"] for n in names if info["kinds"][n] == "removal"}
   if len(kinds) > 1:
      raise ValueError("removals and cascading failures do not share axes: plot them apart")
   if len(qualities) > 1:
      raise ValueError("runs measuring the largest component and the path lengths do not share an axis: plot them apart")
   runs = {n: ns["robustness"][n] for n in names}
   if kinds == {"cascade"}:
      x, y, xlabel, ylabel = None, "failed", "step", "fraction of the items failed"
   else:
      quality = qualities.pop()
      what = {ROBUSTNESS[info["models"][n]["function"]][1] for n in names}
      x, y, xlabel, ylabel = "fraction", quality, f"fraction of the {what.pop()} removed" if len(what) == 1 else "fraction removed", ROBUSTNESS_QUALITIES[quality]
   ns["__plot__"] = _plot("robustness", {"tables": runs, "x": x, "y": y}, {n: PLOT_COLORS[i % len(PLOT_COLORS)] for i, n in enumerate(runs)}, {n: "lines" for n in runs}, {n: n for n in runs}, source = {"tables": "{" + ", ".join(f"{literal(n)}: robustness[{literal(n)}]" for n in runs) + "}", "x": repr(x), "y": repr(y)}, xlabel = xlabel, ylabel = ylabel)
   values = ns["net"].network_attributes
   text_ = "; ".join(f"{n} = {text(values[n])}" for n in names if n in values)
   return dict(_plot_header(ns["__plot__"]), text = text_, warnings = []), _render(ns["__plot__"], "png")


#: Functions of Simulation › Synchronization: function to its label (also the start of the names
#: of its runs).
SYNCHRONIZATION = {"kuramoto_model": "Kuramoto", "simplicial_kuramoto_model": "Simplicial Kuramoto", "node_link_kuramoto_model": "Node-link Kuramoto"}
#: Parameters of the synchronization models not in their dialog: the coupling strengths are asked
#: as a range (:func:`synchronization_signature`); initial phases, final phases and progress are
#: left to the library.
SYNCHRONIZATION_HIDDEN = {"net", "sigma", "theta", "phi", "phases", "progress"}
LISTS.update({(f, "omega") for f in SYNCHRONIZATION} | {("node_link_kuramoto_model", "link_omega")})
OPTIONS[("node_link_kuramoto_model", "model")] = ["\"nl\"", "\"ln\"", "\"lnl\""]
INTEGERS.update({(f, "runs"): 1 for f in SYNCHRONIZATION})
BASIC.update({(f, n) for f in SYNCHRONIZATION for n in ("sigma_from", "sigma_to", "steps", "backward", "runs")} | {("simplicial_kuramoto_model", "k"), ("simplicial_kuramoto_model", "explosive"), ("node_link_kuramoto_model", "model")})


def synchronization_signature(function, net):
   """
   Parameters of the dialog of a synchronization model (:data:`SYNCHRONIZATION`), as
   :func:`signature` without :data:`SYNCHRONIZATION_HIDDEN`: first the name of the run, then the
   coupling strengths ``sigma`` as a range (``sigma_from``, ``sigma_to``, ``steps``, evenly
   spaced) and ``backward`` (the range followed up then down, adiabatically: the hysteresis of
   an explosive transition); the seed is 1 by default.
   """
   params = [p for p in signature(function, net)["params"] if p["name"] not in SYNCHRONIZATION_HIDDEN]
   for p in params:
      if p["name"] == "seed":
         p.update(default = "1")
   params[0:0] = [{"name": "name", "default": "", "hint": "required", "doc": "Name of the run: its table is synchronization[name]", "widget": {"kind": "string"}},
                  {"name": "sigma_from", "default": "0.0", "hint": "", "doc": "Smallest coupling strength", "widget": {"kind": "number", "min": 0., "step": 0.05}},
                  {"name": "sigma_to", "default": "1.0", "hint": "", "doc": "Largest coupling strength", "widget": {"kind": "number", "min": 0., "step": 0.05}},
                  {"name": "steps", "default": "21", "hint": "", "doc": "Number of coupling strengths, evenly spaced from the smallest to the largest", "widget": {"kind": "int", "min": 2}},
                  {"name": "backward", "default": "False", "hint": "", "doc": "Follow the strengths up then down again, each starting from the phases of the previous one (adiabatic): the two branches of the curve show the hysteresis of an explosive transition", "widget": {"kind": "bool"}}]
   return {"params": params}


def synchronization_code(ns, function, arguments):
   """
   Code of a run of Simulation › Synchronization: a synchronization model of the library with
   coupling strengths evenly spaced in a range (and back again with ``backward``, adiabatically),
   its table (order parameters by strength and run) kept in the dictionary ``synchronization`` by
   name, e.g. ``synchronization["Kuramoto 1"] = gsk.kuramoto_model(net, np.linspace(0.0, 1.0,
   21), runs = 10, seed = 1)``.

   :return: Lines of code, and the name of the run.
   """
   if function not in SYNCHRONIZATION:
      raise ValueError(f"unknown synchronization model: {function}")
   arguments = {k: str(v).strip() for k, v in dict(arguments or {}).items() if str(v).strip()}
   for k in ("omega", "link_omega"):   # (mean, sd): typed as two numbers
      if k in arguments and not arguments[k].startswith(("(", "[")):
         arguments[k] = f"({arguments[k]})"
   for k, v in arguments.items():
      try:
         compile(v, k, "eval")
      except SyntaxError:
         raise ValueError(f"{k}: not a Python value: {v}") from None
   name = str(ast.literal_eval(arguments.pop("name", "\"\"")))
   if not name:
      raise ValueError("name: give the run a name")
   lo, hi = float(arguments.pop("sigma_from", "0.0")), float(arguments.pop("sigma_to", "1.0"))
   steps = int(arguments.pop("steps", "21"))
   backward = arguments.pop("backward", "False") == "True"
   arguments.setdefault("seed", "1")
   if backward:
      arguments["adiabatic"] = "True"
   sigma = f"np.linspace({lo!r}, {hi!r}, {steps})"
   lines = [] if any(l.startswith("import numpy as np") for l in ns["__code__"]) else ["import numpy as np"]
   if not (any(l.startswith("synchronization = {}") for l in ns["__code__"]) and isinstance(ns.get("synchronization"), dict)):
      lines.append("synchronization = {}   # runs of Simulation › Synchronization, by name")
   if backward:
      lines.append(f"up = {sigma}   # the strengths up, then down again")
      sigma = "np.r_[up, up[::-1]]"
   lines.append(f"synchronization[{literal(name)}] = gsk.{function}({_target(ns) if ns['net'].higher_order is None else 'net'}, {sigma}" + "".join(f", {k} = {v}" for k, v in arguments.items()) + ")")
   return lines, name


def synchronization_curves(results, names):
   """
   Curves of the synchronization dialog: for every run (a table of a synchronization model) and
   order parameter (``order``, or ``order_nodes``, ``order_down``, ``order_up``; those without
   values left out), the mean over the runs of the model at every coupling strength, with the
   mean plus and minus one standard deviation as a band; the strengths followed down again
   (``backward``) make a second branch. A function of the interface (see :func:`plot_figure`).

   :param results: Dictionary from run name to its table.
   :param names: The runs plotted.
   :return: Dictionary from curve name (the run, the order parameter, ``down``) to a table with
      ``sigma``, ``order``, ``order_low`` and ``order_high``.
   """
   import pandas as pd
   curves = {}
   for name in names:
      df = results[name]
      runs = int(df["run"].nunique())
      step = np.arange(len(df)) // max(runs, 1)   # rows: strength by strength, the runs of each together
      sig = df["sigma"].to_numpy()[::max(runs, 1)]
      turn = np.flatnonzero(np.diff(sig) <= 0)   # the strengths go down again (from the top one, repeated)
      branch = (step > turn[0]).astype(int) if len(turn) else np.zeros(len(df), dtype = int)
      columns = [c for c in df.columns if c.startswith("order") and df[c].notna().any()]
      for c in columns:
         for b in np.unique(branch):
            part = df[branch == b].groupby(step[branch == b], sort = True)
            mean, sd = part[c].mean(), part[c].std().fillna(0.)
            key = name + ("" if len(columns) == 1 else f", {c}") + (", down" if b else "")
            curves[key] = pd.DataFrame({"sigma": part["sigma"].first().to_numpy(), "order": mean.to_numpy(), "order_low": (mean - sd).to_numpy(), "order_high": (mean + sd).to_numpy()})
   return curves


def synchronization_runs(ns):
   """
   The runs of Simulation › Synchronization of a tab (the dialog lists them): ``tables``, their
   names in order; ``models``, run name to its ``function`` and ``arguments`` (for Copy…).
   """
   meta = ns.setdefault("__synchronization__", {})
   runs = ns.get("synchronization") if isinstance(ns.get("synchronization"), dict) else {}
   names = [n for n in runs if n in meta]
   return {"tables": names, "models": {n: meta[n] for n in names}}


def _synchronize(request):
   """
   Command ``synchronize``: runs :func:`synchronization_code` in a tab and replies with the code,
   the drawing (unchanged), the runs (:func:`synchronization_runs`) and ``result``: the order
   parameter at the smallest and the largest coupling strength.
   """
   ns = _spaces[request["tab"]]
   function = request["function"]
   arguments = dict(request.get("arguments") or {})
   lines, name = synchronization_code(ns, function, arguments)
   code = _run(request["tab"], lines)
   arguments.pop("name", None)
   ns.setdefault("__synchronization__", {})[name] = {"function": function, "arguments": arguments}
   header, payload = _draw(ns, code, True)
   df = ns["synchronization"][name]
   c = next(c for c in df.columns if c.startswith("order"))
   mean = df.groupby("sigma", sort = False)[c].mean()
   header["result"] = f"{name}: order parameter {mean.iloc[0]:.3g} at sigma = {mean.index[0]:.3g}, {mean.max():.3g} at most"
   header.update(synchronization_runs(ns))
   return header, payload


def _synchronization_runs(request):
   """
   Command ``synchronization_runs``: the runs of the synchronization dialog of a tab
   (:func:`synchronization_runs`).
   """
   return synchronization_runs(_spaces[request["tab"]]), b""


def _synchronization_plot(request):
   """
   Command ``synchronization_plot``: the transition curves of the runs ``names`` of a tab (order
   parameter against coupling strength, :func:`synchronization_curves`, drawn as
   :func:`robustness_plot` draws its curves), with the PNG image as payload.
   """
   ns = _spaces[request["tab"]]
   info = synchronization_runs(ns)
   names = [n for n in request.get("names") or [] if n in info["models"]]
   if not names:
      raise ValueError("check the runs to plot")
   curves = synchronization_curves(ns["synchronization"], names)
   ns["__plot__"] = _plot("synchronization", {"tables": curves, "x": "sigma", "y": "order"}, {n: PLOT_COLORS[i % len(PLOT_COLORS)] for i, n in enumerate(curves)}, {n: "lines" for n in curves}, {n: n for n in curves}, source = {"tables": f"synchronization_curves(synchronization, {names!r})", "x": "\"sigma\"", "y": "\"order\""})
   return dict(_plot_header(ns["__plot__"]), text = "Order parameter R against the coupling strength sigma: mean over the runs, the band one standard deviation around it.", warnings = []), _render(ns["__plot__"], "png")


PLOT_ELEMENTS["synchronization"] = (lambda **kw: robustness_plot(**kw), {}, {})
PLOT_OPTIONS["synchronization"] = {"xlog": False, "ylog": False, "font_size": 14, "xlabel": "coupling strength sigma", "ylabel": "order parameter R", "xmin": None, "xmax": None, "ymin": None, "ymax": None}


#: Most suggestions of :func:`_suggest`.
MAX_SUGGESTIONS = 30


def _suggest(request):
   """
   Command ``suggest``: the values of the node ``attribute`` of the drawn network of a tab (``None``:
   the ids) that contain ``text`` (ignoring case), the exact ones first, with the number of nodes
   having each (``values``: text and count, at most :data:`MAX_SUGGESTIONS`; ``more``: how many
   were left out), as the search box does (:func:`find_nodes`), e.g. for the source of an
   exploration given by a node attribute.
   """
   import pandas as pd
   net = _drawn(_spaces[request["tab"]])
   attribute = request.get("attribute")
   exact, partial = find_nodes(net, request.get("text", ""), attribute)
   values = _ids(net) if attribute is None else _column(net, attribute)
   found = pd.Series([text(values[i]) for i in np.concatenate([exact, partial])], dtype = object)
   counts = pd.Series(np.asarray(values, dtype = object)).map(text, na_action = "ignore").value_counts() if attribute is not None else None
   distinct_ = list(dict.fromkeys(found))
   shown = [{"text": v, "count": int(counts[v]) if counts is not None else 1} for v in distinct_[:MAX_SUGGESTIONS]]
   return {"values": shown, "more": max(0, len(distinct_) - MAX_SUGGESTIONS)}, b""


#: Format of the session files written by :func:`_save_session`.
SESSION_FORMAT = 1
#: Variables of the namespaces of the tabs not saved in sessions: caches made again when needed.
SESSION_SKIPPED = {"__builtins__", "__tables__", "__plot__", "__gnn_eval__"}


def session_space(ns):
   """
   What a session keeps of the namespace of a tab: every variable that can be pickled (networks,
   tables, results, the code ``__code__``, the style, the undo and redo histories, the models of
   the runs, …), but modules, functions and classes (the functions of the interface are defined
   again when the session opens) and the caches of :data:`SESSION_SKIPPED`.
   """
   import pickle, types
   kept = {}
   for k, v in ns.items():
      if k in SESSION_SKIPPED or isinstance(v, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)):
         continue
      try:
         pickle.dumps(v)
      except Exception:
         continue
      kept[k] = v
   return kept


def _save_session(request):
   """
   Command ``save_session``: writes the session to ``path``, one file (a zip archive) with
   ``session.json`` (the format, the version of the library and ``gui``, the state of the
   interface sent by it: its tabs, in order, with their names, views, selections, …) and
   ``spaces.pickle``, the namespaces of the tabs listed in ``gui["tabs"]`` (:func:`session_space`),
   pickled together so that the objects tabs share stay shared. Replies with the ``size`` of the
   file.
   """
   import pickle, zipfile, time
   gui = request["gui"]
   spaces = {int(t["tab"]): session_space(_spaces[int(t["tab"])]) for t in gui.get("tabs", []) if int(t["tab"]) in _spaces}
   meta = {"format": SESSION_FORMAT, "graphoskop": _version(), "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "gui": gui}
   path = request["path"]
   with zipfile.ZipFile(path + ".part", "w", zipfile.ZIP_DEFLATED) as z:
      z.writestr("session.json", json.dumps(meta, indent = 1))
      z.writestr("spaces.pickle", pickle.dumps(spaces, protocol = pickle.HIGHEST_PROTOCOL))
   os.replace(path + ".part", path)   # never half a file under the name
   return {"size": os.path.getsize(path), "tabs": len(spaces)}, b""


def _open_session(request):
   """
   Command ``open_session``: reads a session file (:func:`_save_session`) and puts the namespaces
   of its tabs back, the tab of id ``i`` of the file (in the order of the tabs) as tab
   ``first + i``, with the functions of the interface (:func:`_helpers`); replies with the state of
   the interface (``gui``, its tabs with their new ids in ``tab``), the ``version`` of the library
   that saved it and a ``warning`` if it differs from the running one. The file is unpickled: open
   only sessions from trusted sources.
   """
   import pickle, zipfile
   with zipfile.ZipFile(request["path"]) as z:
      meta = json.loads(z.read("session.json"))
      spaces = pickle.loads(z.read("spaces.pickle"))
   if meta.get("format") != SESSION_FORMAT:
      raise ValueError(f"not a session file of this version of the interface (format {meta.get('format')})")
   gui = meta["gui"]
   first = int(request["first"])
   for i, t in enumerate(gui.get("tabs", [])):
      ns = spaces[int(t["tab"])]
      ns.update(_helpers(), gsk = gsk)
      t["tab"] = first + i
      _spaces[first + i] = ns
   reply = {"gui": gui, "version": meta.get("graphoskop")}
   if meta.get("graphoskop") != _version():
      reply["warning"] = f"saved with graphoskop {meta.get('graphoskop')}, running {_version()}"
   return reply, b""


def _redraw(request):
   """
   Command ``redraw``: the code and the drawing of a tab as they are (e.g. after a session was
   opened), keeping its node size.
   """
   ns = _spaces[request["tab"]]
   return _draw(ns, "\n".join(ns["__code__"]), True)


#: Commands understood by the worker.
COMMANDS = {"preview": _preview, "load": _load, "generate": _generate, "signature": _signature, "layout": _layout, "communities": _communities, "style": _style, "breaks": _breaks, "attributes": _attributes, "table": _table, "replay": _replay, "close": _close, "centrality": _centrality, "save_plot": _save_plot, "replot": _replot, "move": _move, "edit": _edit, "select": _select, "simulate": _simulate, "simulations": _simulations, "epidemic_plot": _epidemic_plot, "undo": _undo, "spread": _spread, "animation": _animation, "find": _find, "backbone": _backbone, "backbone_plot": _backbone_plot, "backbone_stats": _backbone_stats, "extract": _extract, "distribution": _distribution, "remove_attribute": _remove_attribute, "drawing": _drawing, "fit": _fit, "nestedness": _nestedness, "vectors": _vectors, "add": _add, "set_value": _set_value, "link_predictions": _link_predictions, "link_split": _link_split, "link_predict": _link_predict, "link_signs": _link_signs, "link_plot": _link_plot, "gnns": _gnns, "gnn": _gnn, "gnn_plot": _gnn_plot, "derive": _derive, "formulas": _formulas, "explore": _explore, "neighbors": _neighbors, "mine": _mine, "hierarchy": _hierarchy, "suggest": _suggest, "save_session": _save_session, "open_session": _open_session, "redraw": _redraw, "statistics": _statistics, "export": _export, "correct": _correct, "empty": _empty, "image_pdf": _image_pdf, "image_vector": _image_vector, "robustness": _robustness, "robustness_runs": _robustness_runs, "robustness_plot": _robustness_plot, "paths": _paths, "synchronize": _synchronize, "dendrogram": _dendrogram, "community_tree": _community_tree, "synchronization_runs": _synchronization_runs, "synchronization_plot": _synchronization_plot}

#: Functions of the Layout menu: layouts of node-link diagrams, the matrix layout (drawn as the
#: adjacency matrix), transformations of layouts, edge bundling and edge routing.
LAYOUTS = {"circular_layout", "force_directed_layout", "grid_layout", "hierarchical_layout", "matrix_layout", "umap_layout", "mirror_layout", "rotate_layout", "remove_overlaps", "bundle_edges", "route_edges"}

#: Functions of :data:`LAYOUTS` transforming the current layout (or curving its edges): the drawn
#: nodes keep their size, and ``remove_overlaps`` and ``route_edges`` get it as ``size`` (unless
#: given), to remove the overlaps on screen and to route the edges around the drawn nodes.
TRANSFORMS = {"mirror_layout", "rotate_layout", "remove_overlaps", "bundle_edges", "route_edges"}
#: Functions of :data:`LAYOUTS` computing curves of the edges, drawn instead of straight edges.
CURVES = {"bundle_edges", "route_edges"}
#: Start of the line of code computing curves of the edges: the curves are stored in ``bends``.
BENDS = "bends = gsk."
#: Comment before code using :func:`node_sizes`, which is defined by the interface.
NODE_SIZES_NOTE = "# node_sizes: relative sizes of the drawn nodes, a function of the graphical interface"
#: Comment before the first code moving nodes with :func:`move_nodes`, defined by the interface.
MOVE_NOTE = "# move_nodes: positions of nodes dragged in the graphical interface"

#: Functions of Analysis › Community discovery: the community discovery functions of the library.
COMMUNITIES = {"bigclam_communities", "block_model_communities", "demon_communities", "hierarchical_link_communities", "hierarchical_random_graph_communities", "infomap_communities", "label_propagation_communities", "leiden_communities", "line_graph_communities", "oslom_communities"}
#: Functions of Analysis › Centrality: name of the attribute they write by default, and what
#: they give: ``"nodes"`` (a value per node), ``"edges"`` (a value per edge) or ``"hits"`` (hub
#: and authority scores, written as ``<attribute>_hub`` and ``<attribute>_authority``).
CENTRALITIES = {"alpha_centrality": ("alpha", "nodes"), "betweenness_centrality": ("betweenness", "nodes"), "closeness_centrality": ("closeness", "nodes"), "degree": ("degree", "nodes"), "edge_betweenness_centrality": ("edge_betweenness", "edges"), "eigenvector_centrality": ("eigenvector", "nodes"), "flow_centrality": ("flow", "nodes"), "harmonic_centrality": ("harmonic", "nodes"), "hits": ("hits", "hits"), "katz_centrality": ("katz", "nodes"), "k_core_centrality": ("coreness", "nodes"), "pagerank": ("pagerank", "nodes"), "reach_centrality": ("reach", "nodes"), "springrank": ("springrank", "nodes"), "discrete_core_periphery": ("core_discrete", "core"), "continuous_core_periphery": ("core_continuous", "core"), "rombach_core_periphery": ("core_rombach", "core"), "hyperedge_size": ("size", "hyperedges"), "local_triangle_closure": ("triangle_closure", "higher"), "two_node_clustering": ("two_node_clustering", "higher"), "generalized_degree": ("generalized_degree", "higher"), "node_edge_centrality": ("node_edge", "node_edge")}
#: Kinds of :data:`CENTRALITIES` of hypergraphs and simplicial complexes only, computed on the
#: network and written on the incidence network the interface draws (:func:`incidence_values`):
#: values of the nodes, of the hyperedges, or both (the node and edge eigenvector centrality).
HIGHER_KINDS = {"higher", "hyperedges", "node_edge"}
#: Parameters of the centralities not shown in the parameter dialog: the interface computes them
#: for all components, with the default numerical settings.
CENTRALITY_HIDDEN = {"net", "component", "node", "distribution", "lengths", "batch_size", "n_jobs", "tol", "max_iter", "personalization", "progress", "hyperedge", "hyperedges", "simplex", "m"}
#: Functions of the Generators menu: the generators of synthetic networks of the library (not
#: :func:`~graphoskop.graphrnn`, which learns from a network).
GENERATORS = {"chain_graph", "gnm_random_graph", "gnp_random_graph", "configuration_model", "kronecker_graph", "geometric_graph", "caveman_graph", "watts_strogatz_graph", "barabasi_albert_graph", "link_selection_graph", "copying_graph", "holme_kim_graph", "planted_partition_graph", "stochastic_block_model", "degree_corrected_block_model", "mixed_membership_block_model", "nested_block_model", "girvan_newman_graph", "lfr_benchmark_graph", "growing_manifold", "simplicial_configuration_model", "hypergraph_configuration_model"}


def environment():
   """
   Name of the Python environment of the worker: the name of a virtual or conda environment
   (``"base"`` for conda's base environment), or ``"system"``.
   """
   prefix = sys.prefix
   if prefix != sys.base_prefix or os.path.isdir(os.path.join(prefix, "conda-meta")):
      return "base" if os.path.isdir(os.path.join(prefix, "envs")) else os.path.basename(prefix)
   return "system"


def _import():
   """
   Imports the library from the source tree the interface belongs to (the parent of the ``gui``
   folder), so that the worker and the library always match, or, without it, the installed one;
   and shows its progress bars in the interface (:class:`ProgressBar`).

   :return: Dictionary for the ``ready`` frame.
   """
   global gsk
   tree = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
   if os.path.isfile(os.path.join(tree, "graphoskop", "__init__.py")):
      sys.path.insert(0, tree)
   import graphoskop
   import torch
   gsk = graphoskop
   sys.modules["graphoskop.progress"]._tqdm = lambda: ProgressBar
   return {"version": _version(), "cuda": torch.cuda.is_available()}


def _version():
   """
   Version of the library: its ``__version__`` (when known), else the version in the ``pyproject.toml`` of its
   source tree, else the version of the installed package (``None`` if none is found).
   """
   version = getattr(gsk, "__version__", None)
   version = None if version == "unknown" else version   # the library's own fallback outside an installation
   pyproject = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(gsk.__file__))), "pyproject.toml")
   if version is None and os.path.isfile(pyproject):
      with open(pyproject) as f:
         m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
      version = m.group(1) if m else None
   if version is None:
      try:
         from importlib.metadata import version as installed
         version = installed("graphoskop")
      except Exception:
         pass
   return version


def _reader(sock, requests):
   """
   Reads the frames of the interface (in a thread of its own) into ``requests``, and ``None`` when
   the connection closes. An ``abort`` frame does not wait in the queue: it interrupts the command
   running, if any, which then fails with :class:`KeyboardInterrupt` (raised as soon as Python
   code runs again: a long call into compiled code finishes first).
   """
   while True:
      lengths = _receive(sock, 8)
      if lengths is None:
         requests.put(None)
         return
      h, p = struct.unpack("<II", lengths)
      request = json.loads(_receive(sock, h).decode("utf-8"))
      _receive(sock, p) if p else None
      if request.get("cmd") == "abort":
         if _busy:
            _thread.interrupt_main()
      else:
         requests.put(request)


def main(port):
   """
   Connects to the interface on ``port`` and serves its requests, one at a time and in order,
   until the connection closes. A command interrupted by an ``abort`` frame (see :func:`_reader`)
   replies ``ok = False`` with the error ``"Aborted"``.

   :param port: TCP port where the interface listens, on 127.0.0.1.
   """
   global _busy
   global _sock
   sock = socket.create_connection(("127.0.0.1", port))
   _send(sock, {"cmd": "hello", "ok": True, "python": sys.version.split()[0], "executable": sys.executable, "environment": environment()})
   try:
      _send(sock, dict(cmd = "ready", ok = True, **_import()))
   except Exception as e:
      _send(sock, {"cmd": "ready", "ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
      return
   _sock = sock
   requests = queue.Queue()
   threading.Thread(target = _reader, args = (sock, requests), daemon = True).start()
   while True:
      try:
         request = requests.get()
      except KeyboardInterrupt:   # an abort that came as its command ended
         continue
      if request is None:
         return
      reply, payload = {"cmd": request.get("cmd"), "id": request.get("id"), "ok": True}, b""
      try:
         _busy = True
         extra, payload = COMMANDS[request["cmd"]](request)
         _busy = False
         reply.update(extra)
      except KeyboardInterrupt:
         _busy = False
         reply.update(ok = False, error = "Aborted")
         payload = b""
      except Exception as e:
         _busy = False
         reply.update(ok = False, error = f"{type(e).__name__}: {e}", traceback = traceback.format_exc())
         payload = b""
         traceback.print_exc()
      _send(sock, reply, payload)


if __name__ == "__main__":
   main(int(sys.argv[1]))
