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
#: Floats per edge in the buffer of the edges: a 2D transform (8 floats) and a color (4).
EDGE_STRIDE = 12
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
HINTS = {"seed": "random", ("geometric_graph", "m"): "unused (r)", ("geometric_graph", "positions"): "random", ("kronecker_graph", "n"): "from k", ("lfr_benchmark_graph", "muw"): "unweighted", ("lfr_benchmark_graph", "minc"): "smallest degree", ("lfr_benchmark_graph", "maxc"): "maxk", ("mixed_membership_block_model", "threshold"): "1 / (k + 1)", ("mixed_membership_block_model", "memberships"): "from n and alpha", ("nested_block_model", "degrees"): "uniform", "groups": "none", "order": "by the edges", "weight": "unweighted", "levels": "from the edges", "root": "most neighbors", "attributes": "adjacency matrix", "embedding": "none", "epochs": "automatic", "initial": "none", ("bundle_edges", "groups"): "from the layout", ("bigclam_communities", "groups"): "automatic", ("block_model_communities", "groups"): "automatic", ("block_model_communities", "threshold"): "1 / (groups + 1)", ("hierarchical_link_communities", "threshold"): "best partition density", ("hierarchical_random_graph_communities", "chains"): "one per thread", ("infomap_communities", "directed"): "as the network", ("infomap_communities", "physical"): "none", ("leiden_communities", "initial"): "singletons", "column_groups": "none", "column_order": "by the edges"}
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
      :data:`COMMUNITIES` or :data:`CENTRALITIES` (these get two more parameters: ``attribute``,
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
   hidden = HIDDEN if function in LAYOUTS else (CENTRALITY_HIDDEN if function in CENTRALITIES else (EPIDEMIC_HIDDEN if function in EPIDEMICS else (BACKBONE_HIDDEN if function in set(BACKBONES) | FILTERS else (PROJECTION_HIDDEN if function in PROJECTIONS else (TRANSFORMATION_HIDDEN if function in TRANSFORMATIONS else {"net", "progress"})))))   # community discovery shows ``attribute``
   params = []
   for p in inspect.signature(f).parameters.values():
      if p.name in hidden:
         continue
      if p.default is inspect.Parameter.empty or p.default is None:
         shown = SHOWN_DEFAULTS.get((function, p.name), SHOWN_DEFAULTS.get(p.name, ""))
      else:
         shown = literal(p.default)
      hint = HINTS.get((function, p.name), HINTS.get(p.name, "required"))
      yes_no = fit and p.default is None and shown in ("True", "False")
      if fit and p.default is None:   # from the network (the device: that of the network)
         shown, hint = "", HINTS.get(p.name) if p.name == "seed" else ("required" if (function, p.name) in FIT_REQUIRED else "from the network")
         if p.name == "device" and net is not None and net.components:
            shown = literal(net.components[0].edge_index.device.type)
      param = {"name": p.name, "default": shown, "hint": hint, "doc": DOCS.get((function, p.name), descriptions.get(p.name, ""))}
      param["widget"] = {"kind": "choice", "options": ["", "True", "False"]} if yes_no else widget(function, p.name, shown, hint, net)
      params.append(param)
   if fit:
      params[:0] = [{"name": "test", "default": "", "hint": "none (one null network)", "doc": "Measure compared with its values on null networks of the model (gsk.randomization_test), or none for one null network only", "widget": {"kind": "choice", "options": [""] + [literal(m) for m in measures(net)]}},
                    {"name": "runs", "default": "100", "hint": "", "doc": "Number of null networks of the randomization test", "widget": {"kind": "int", "min": 1}}]
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
     weights of the backbones, whose default is ``"weight"``;
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
   if shown in ("True", "False"):
      return {"kind": "bool"}
   if function == "quotient_network" and name == "sets":   # sets of nodes: the combinations of values of node attributes
      return {"kind": "attribute", "choices": "nodes", "multiple": True, "extra": []}
   if function in PROJECTIONS and name == "node_type":
      return {"kind": "choice", "options": [literal(v) for v in (net.bipartite_labels if net is not None and net.bipartite_labels is not None else [])]}
   options = OPTIONS.get((function, name), OPTIONS.get(name) if (function, name) not in CHOICES else None)
   if options is not None:
      return {"kind": "choice", "options": options}
   choice = CHOICES.get((function, name), CHOICES.get(name) if function in LAYOUTS or (function in COMMUNITIES | set(CENTRALITIES) | set(BACKBONES) | FILTERS | set(PROJECTIONS) | set(TRANSFORMATIONS) and name == "weight") else None)
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
   defined as ``inc`` the first time. The curves of :func:`~graphoskop.bundle_edges` are stored
   in ``bends`` (:data:`BENDS`).

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
   return lines + [BENDS + call[len("gsk.bundle_edges"):] if function == "bundle_edges" else call]


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
   :func:`node_sizes`, :func:`move_nodes` and :func:`groups`.
   """
   return {"node_sizes": node_sizes, "move_nodes": move_nodes, "groups": groups}


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
   :func:`positions`). Nodes are :data:`NODE_COLOR` and edges :data:`EDGE_COLOR`, all of the
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
   return {"nodes": nodes, "edges": {"attribute": None, "color": EDGE_COLOR}, "node_size": {"attribute": None, "size": 1.}, "edge_width": {"attribute": None, "size": 1.}, "node_opacity": {"attribute": None, "opacity": 1.}, "edge_opacity": {"attribute": None, "opacity": 1.}, "position": {"x": None, "y": None}}


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
   for key, names in [("nodes", nodes), ("node_size", nodes), ("node_opacity", nodes), ("edges", edges), ("edge_width", edges), ("edge_opacity", edges)]:
      a = (style.get(key) or {}).get("attribute")
      if a is not None and a not in names:
         default = default or default_style(net)
         style[key] = default[key]
   position = style.get("position") or {}
   style["position"] = {k: v if v in nodes else None for k, v in position.items()}
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

   :param net: A :class:`~graphoskop.Network` with the numeric node attributes
      ``<attribute>_x`` and ``<attribute>_y``.
   :param size: Default diameter of the nodes, or ``None`` for :func:`node_size`.
   :param style: Style (see :func:`default_style`), or ``None`` for the default one.
   :param attribute: Prefix of the position attributes.
   :param bends: Curves of the edges, as returned by :func:`~graphoskop.bundle_edges`, or
      ``None`` for straight edges.
   :param matrix: Name of the node attribute of a matrix layout, or ``None`` for a node-link
      drawing.
   :return: Dictionary with the node, edge and loop buffers (float32 arrays), whether edges are curves
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
   elif bends is not None and m and bends["edge"].nunique() == m:   # one instance per Bezier piece
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
      E = np.zeros((m, EDGE_STRIDE), dtype = np.float32)
      E[:, 0], E[:, 4] = dx, dy                                  # quad's x axis: along the edge
      E[:, 1], E[:, 5] = -dy / length * w, dx / length * w       # y axis: across it, the relative width
      E[:, 3], E[:, 7] = (x[s] + x[t]) / 2, (y[s] + y[t]) / 2
      E[:, 8:] = c
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
   return {"nodes": N, "edges": E, "loops": L, "curved": E.shape[1] == CURVE_STRIDE, "cells": matrix is not None, "ids": ids, "grid": grid, "order": order, "starts": starts, "box": box, "size": size, "biggest": biggest}


#: Diameter of the circle of a self loop, relative to the diameter of its node.
LOOP_SIZE = 0.6


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
   :data:`MAX_VALUES` of them in order (``values``, texts, see :func:`distinct`).

   :return: Dictionary with lists ``nodes`` and ``edges``.
   """
   def describe(name, v):
      texts, pos = distinct(v) if v.dtype.kind != "f" else (distinct(v)[0], None)
      x = v.astype(float) if pos is None else np.where(pos < 0, np.nan, pos.astype(float))
      ok = x[~np.isnan(x)]
      lo, hi = (float(ok.min()), float(ok.max())) if len(ok) else (0., 1.)
      return {"name": name, "numeric": v.dtype.kind == "f", "min": lo, "max": hi, "positive": lo > 0, "count": len(texts), "values": texts[:MAX_VALUES]}
   nodes = [describe(a, _column(net, a)) for a in list(net.node_attributes) + list(net.qualitative_node_attributes)]
   edges = [describe(a, edge_values(net, a)) for a in list(net.edge_attributes) + list(net.qualitative_edge_attributes)]
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


def _tables(ns, kind = "nodes"):
   """
   Tables of the data panel of a tab, each made when first asked for (``kind``) and cached until
   the code of the tab runs again: one row per node of the drawn network (``nodes``: in the
   drawing order, with its id and attributes), and one per edge (``edges``: source and target, as
   positions of nodes, then the attributes; with the positions ``s`` and ``t`` and the node
   ``ids``, which name the sources and targets of the rows shown only, see :func:`_table`).
   """
   import pandas as pd
   T = ns.get("__tables__")
   if T is None:
      T = ns["__tables__"] = {}
   if kind not in T:
      net = _drawn(ns)
      if kind == "nodes":
         T["nodes"] = pd.DataFrame({"node": _ids(net), **{a: _column(net, a) for a in list(net.node_attributes) + list(net.qualitative_node_attributes)}})   # one table at once
      else:
         s, t = _edges(net)
         T.update(edges = pd.DataFrame({"source": s, "target": t, **{a: edge_values(net, a) for a in list(net.edge_attributes) + list(net.qualitative_edge_attributes)}}), s = s, t = t, ids = _ids(net))
   return T


def summary(net):
   """
   Properties of a network for the "Network" tab of the data panel, as pairs of texts.
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
   rows += [["numeric node attributes", ", ".join(net.node_attributes)], ["qualitative node attributes", ", ".join(net.qualitative_node_attributes)]]
   rows += [["numeric edge attributes", ", ".join(net.edge_attributes)], ["qualitative edge attributes", ", ".join(net.qualitative_edge_attributes)]]
   return rows


def _table(request):
   """
   Command ``table``: a page of a table of the data panel. ``kind`` is ``"nodes"`` (all nodes, the
   attributes of node ``node``, a position in the drawing order, as pairs attribute and value, or
   the rows of the nodes ``nodes``, a list of positions), ``"edges"`` (all edges, those of node
   ``node`` or of the nodes ``nodes``, or the edges ``edges``, positions in the order of the
   drawing), ``"network"`` (:func:`summary`) or ``"tables"`` (the table ``name`` of the
   simulations, :func:`_simulate`);
   rows ``offset`` to ``offset + count`` (texts, ``""`` for missing values) of the table sorted
   as a whole by the column ``sort`` (if given; ``ascending`` or not, see :func:`sort_rows`), with
   the ``columns``, whether each is ``numeric`` (the interface aligns them to the right) and the
   ``total`` number of rows.
   """
   import pandas as pd
   ns = _spaces[request["tab"]]
   kind, node = request["kind"], request.get("node")
   offset, count = int(request.get("offset", 0)), int(request.get("count", 200))
   if kind == "network":
      df = pd.DataFrame(summary(ns["net"]), columns = ["property", "value"])
   elif kind == "tables":
      df = ns.get("epidemics", {}).get(request.get("name"))
      if df is None:
         return {"columns": [], "numeric": [], "rows": [], "total": 0}, b""
      df = df.reset_index()
   elif kind == "nodes" and node is not None:
      T = _tables(ns, "nodes")
      row = T["nodes"].iloc[int(node)]
      df = pd.DataFrame({"attribute": [str(c) for c in T["nodes"].columns], "value": pd.Series([text(row[c]) or "" for c in T["nodes"].columns], dtype = object)})
   else:
      df = _rows(_tables(ns, kind), request)
   ids = _tables(ns, "edges")["ids"] if kind == "edges" else None
   if ids is not None and request.get("sort") in ("source", "target"):   # sorted by the ids
      df = df.assign(**{request["sort"]: ids[df[request["sort"]].to_numpy()]})
   if request.get("sort") in df.columns:
      df = sort_rows(df, request["sort"], bool(request.get("ascending", True)))
   page = df.iloc[offset:offset + count]
   if ids is not None:   # the ids of the rows shown
      page = page.assign(**{c: ids[page[c].to_numpy()] for c in ("source", "target") if page[c].dtype.kind in "iu"})
   rows = [[text(v) or "" for v in r] for r in page.itertuples(index = False, name = None)]
   numeric = [kind == "tables" or (ids.dtype.kind in "fiu" if ids is not None and c in ("source", "target") else df[c].dtype.kind in "fiu") for c in df.columns]
   return {"columns": [str(c) for c in df.columns], "numeric": numeric, "rows": rows, "total": len(df)}, b""


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



def _run(tab, code, undoable = False):
   """
   Runs lines of code in the namespace of a tab and adds them to its code. Unless ``undoable``
   (moves and removals, see :func:`_snapshot`), the code cannot be undone, and neither can what
   came before it: the undo and redo lists of the tab are emptied.

   :return: The whole code of the tab.
   """
   ns = _spaces.setdefault(tab, dict(gsk = gsk, __code__ = ["import graphoskop as gsk", ""], **_helpers()))
   ns["__tables__"] = None
   if undoable:
      before = _snapshot(ns)
   exec("\n".join(code), ns)
   ns["__code__"] += code
   if undoable:
      ns["__undo__"] = (ns.get("__undo__") or [])[-MAX_UNDO + 1:] + [before]
   else:
      ns["__undo__"] = []
   ns["__redo__"] = []
   return "\n".join(ns["__code__"])


#: Largest number of moves and removals that can be undone in a tab.
MAX_UNDO = 100


def _snapshot(ns):
   """
   What undo and redo restore in the namespace of a tab: its code, its drawn network (removals
   make a new network object, the old one is kept as it was), and the positions of its nodes
   (moves change them in place).
   """
   t = _target(ns)
   net = ns[t]
   pos = {a: [x.clone() for x in net.get_node_attribute(a, None)] for a in ("pos_x", "pos_y") if a in net.node_attributes}
   return {"code": list(ns["__code__"]), "target": t, "net": net, "pos": pos}


def _restore(ns, snap):
   """
   Puts back a :func:`_snapshot` in the namespace of a tab.
   """
   ns["__code__"] = list(snap["code"])
   ns[snap["target"]] = snap["net"]
   for a, values in snap["pos"].items():
      snap["net"].set_node_attribute(a, [x.clone() for x in values], None)
   ns["__tables__"] = None


def _undo(request):
   """
   Command ``undo`` (or ``redo`` with ``redo = True``): goes back over the last move or removal of
   nodes or edges of a tab (or forward again), with its code, and replies with the code, a
   ``summary`` and the drawing. Only the moves and removals since the last other change of the
   network can be undone.
   """
   ns = _spaces[request["tab"]]
   back, ahead = ("__redo__", "__undo__") if request.get("redo") else ("__undo__", "__redo__")
   if not ns.get(back):
      raise ValueError("nothing to " + ("redo" if request.get("redo") else "undo"))
   ns[ahead] = (ns.get(ahead) or []) + [_snapshot(ns)]
   _restore(ns, ns[back].pop())
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


def _draw(ns, code, keep = False):
   """
   Reply with the code of a tab and, if its network has positions, its drawing (:func:`drawing`):
   the header has the numbers of nodes, edges and self loops, the byte sizes of the parts of the
   payload (node buffer, edge buffer, ids, grid order, grid starts, sources and targets of the
   edges as int32 positions of the nodes, loop buffer, in this order), the grid, the box, the
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
   if "pos_x" not in draw.node_attributes and matrix is None or draw.higher_order is not None:   # no layout yet (a new hypergraph, e.g. from clique promotion: laid out as its incidence network)
      return {"code": code, "nodes": 0}, b""
   ns["__style__"] = prune_style(dict(default_style(draw), **ns.get("__style__", {})), draw)
   bends = ns.get("bends") if last == "bundle_edges" and not _positioned(ns) else None   # the curves follow the layout
   d = drawing(draw, ns.get("__size__") if keep else None, ns["__style__"], bends = bends, matrix = matrix)
   ns["__size__"] = d["size"]
   s, t = _edges(draw)
   parts = [d["nodes"].tobytes(), d["edges"].tobytes(), d["ids"], d["order"].tobytes(), d["starts"].tobytes(), s.astype(np.int32).tobytes(), t.astype(np.int32).tobytes(), d["loops"].tobytes()]
   header = {"code": code, "nodes": len(d["nodes"]), "edges": len(d["edges"]), "loops": len(d["loops"]), "curved": d["curved"], "cells": d["cells"], "subdivisions": subdivisions(len(d["edges"])), "box": d["box"], "node_size": d["size"], "biggest": d["biggest"], "grid": d["grid"], "style": ns["__style__"], "parts": [len(p) for p in parts]}
   header.update(undo = bool(ns.get("__undo__")), redo = bool(ns.get("__redo__")), history = _history_key(ns, draw), movable = matrix is None and not _positioned(ns))
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
   lines = ["inc = gsk.incidence_network(net)"] if higher and "inc" not in ns else []
   target = "inc" if higher else "net"
   extra = "".join(f", {k} = {v.strip()}" for k, v in arguments.items())
   call = f"gsk.degree({target}, distribution = None, component = None{extra})" if function == "degree" else f"gsk.{function}({target}, component = None{extra})"
   kind = CENTRALITIES[function][1]
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
      ns["__plot__"] = _plot("degree", {"degrees": degrees, "fit": fit}, dict(PLOT_ELEMENTS["degree"][1]), {"data": "points", "fit": "lines"})
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
#: :data:`DISTRIBUTIONS`).
PLOT_OPTIONS = {"nestedness": {"xlog": False, "ylog": False, "font_size": 14, "xlabel": "", "ylabel": "", "xmin": None, "xmax": None, "ymin": None, "ymax": None}, "randomization": {"kind": "right", "xlog": False, "ylog": False, "font_size": 14, "xlabel": "", "ylabel": "", "xmin": None, "xmax": None, "ymin": None, "ymax": None}, "distribution": {"kind": "count", "xlog": False, "ylog": False, "font_size": 14, "xlabel": "", "ylabel": "", "xmin": None, "xmax": None, "ymin": None, "ymax": None}, "degree": {"xlog": True, "ylog": True, "font_size": 14, "xlabel": "degree k", "ylabel": "fraction of nodes with degree ≥ k", "xmin": None, "xmax": None, "ymin": None, "ymax": None}, "epidemics": {"xlog": False, "ylog": False, "font_size": 14, "xlabel": "step", "ylabel": None, "xmin": None, "xmax": None, "ymin": None, "ymax": None}}
#: How a data series can be drawn: lines, points, lines with points, or bars.
PLOT_STYLES = ["lines", "points", "linespoints", "bars"]


def _series(ax, x, y, style, color, label):
   """
   Draws a data series in one of the :data:`PLOT_STYLES`. Bars are 0.8 times the smallest gap
   between the x values wide (on a logarithmic x axis, as wide relatively).
   """
   x, y = np.asarray(x, dtype = float), np.asarray(y, dtype = float)
   if style == "bars":
      gaps = np.diff(np.unique(x))
      if ax.get_xscale() == "log" and len(gaps) and (x > 0).all():
         r = np.diff(np.log(np.unique(x))).min()
         ax.bar(x * np.exp(-0.4 * r), y, width = x * (np.exp(0.4 * r) - np.exp(-0.4 * r)), align = "edge", color = color, alpha = 0.8, label = label)
      else:
         ax.bar(x, y, width = 0.8 * (gaps.min() if len(gaps) else 1.), color = color, alpha = 0.8, label = label)
   else:
      ax.plot(x, y, {"lines": "-", "points": "o", "linespoints": "-o"}[style], color = color, lw = 2 if style == "lines" else 1.5, ms = 4, label = label)


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
   (if a series has a label).

   :param format: ``"png"`` or ``"pdf"``.
   :param path: File to write, or ``None`` to return the bytes.
   :param dpi: Resolution of PNG images.
   :return: The bytes of the image, or ``None`` with ``path``.
   """
   import io
   import matplotlib
   matplotlib.use("Agg")
   from matplotlib.figure import Figure
   with matplotlib.rc_context({"font.family": _plot_font(), "font.size": float(options["font_size"]), "pdf.fonttype": 42}):
      fig = Figure(figsize = (8, 5.5))
      ax = fig.add_subplot()
      ax.set_xscale("log" if options["xlog"] else "linear")
      ax.set_yscale("log" if options["ylog"] else "linear")
      draw(ax)
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
PLOT_ELEMENTS = {"nestedness": (lambda **kw: nestedness_plot(**kw), {"matrix": "#377eb8", "isocline": "#e41a1c"}, {"matrix": "Matrix", "isocline": "Isocline"}), "randomization": (lambda **kw: randomization_plot(**kw), {"null": "#377eb8", "observed": "#e41a1c"}, {"null": "Null networks", "observed": "Observed value"}), "distribution": (lambda **kw: distribution_plot(**kw), {"data": "#377eb8"}, {"data": "Data"}), "degree": (degree_plot, {"data": "#377eb8", "fit": "#e41a1c", "k_min": "#999999"}, {"data": "Data", "fit": "Fit", "k_min": "Smallest degree of the fit"}), "epidemics": (lambda tables, states = ("I",), **kw: epidemic_plot(tables, states, **kw), {}, {})}


def _render(plot, format, path = None, dpi = 100):
   """
   Draws a plot kept in a namespace (``kind``, ``data``, ``colors`` and ``options``) with its
   function of :data:`PLOT_ELEMENTS`.
   """
   return PLOT_ELEMENTS[plot["kind"]][0](**plot["data"], colors = plot["colors"], options = plot["options"], styles = plot["styles"], format = format, path = path, dpi = dpi)


def _plot(kind, data, colors, styles, labels = None, **options):
   """
   A plot to keep in a namespace (``ns["__plot__"]``): its ``kind`` (of :data:`PLOT_ELEMENTS`),
   the arguments of its function, the colors of its elements, how its data series are drawn
   (:data:`PLOT_STYLES`, by key), the labels of the elements (``None``: those of
   :data:`PLOT_ELEMENTS`), and its options (:data:`PLOT_OPTIONS`, updated with ``options``).
   """
   return {"kind": kind, "data": data, "colors": colors, "styles": styles, "labels": labels, "options": dict(PLOT_OPTIONS[kind], **options)}


def _plot_header(plot):
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
   return header


def _replot(request):
   """
   Command ``replot``: draws the last plot of a tab again with new ``colors`` (element key to
   ``#rrggbb``), ``styles`` (series key to one of :data:`PLOT_STYLES`) and ``options``
   (logarithmic axes ``xlog`` and ``ylog``, ``font_size`` from 4 to 72 points, axis labels
   ``xlabel`` and ``ylabel``, ranges ``xmin``, ``xmax``, ``ymin``, ``ymax``: numbers or ``None``),
   and replies with the PNG image as payload.
   """
   plot = _spaces[request["tab"]]["__plot__"]
   plot["colors"].update({k: v for k, v in (request.get("colors") or {}).items() if k in plot["colors"]})
   plot["styles"].update({k: v for k, v in (request.get("styles") or {}).items() if k in plot["styles"] and v in PLOT_STYLES})
   o = request.get("options") or {}
   plot["options"].update({k: (None if o[k] is None else float(o[k])) for k in ("xmin", "xmax", "ymin", "ymax") if k in o})
   plot["options"].update({k: bool(o[k]) for k in ("xlog", "ylog") if k in o})
   plot["options"].update({k: str(o[k]) for k in ("xlabel", "ylabel") if k in o})
   if "font_size" in o:
      plot["options"]["font_size"] = min(max(float(o["font_size"]), 4.), 72.)
   if plot["kind"] == "distribution" and o.get("kind") in DISTRIBUTIONS or plot["kind"] == "randomization" and o.get("kind") in TAILS:
      plot["options"]["kind"] = o["kind"]
   return _plot_header(plot), _render(plot, "png")


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
      image = ax.imshow(matrix, extent = (0, 1, 1, 0), aspect = "auto", interpolation = "nearest", cmap = cmap, vmin = 0, vmax = max(float(matrix.max()), 1e-12) if matrix.size else 1)
      ax.figure.colorbar(image, ax = ax, label = "share of ones")
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
   ns["__plot__"] = _plot("nestedness", {"matrix": matrix, "line": iso["line"], "rows": a, "columns": b, "p": iso["p"]}, dict(PLOT_ELEMENTS["nestedness"][1]), {})
   png = _render(ns["__plot__"], "png")
   header, payload = _draw(ns, code, True)
   header["plot"] = dict(_plot_header(ns["__plot__"]), text = nestedness_text(nested, iso, method), warnings = [], title = f"Nestedness of {request.get('name', 'the network')}")
   header["parts"] = header.get("parts", []) + [len(png)] if header["nodes"] else [len(png)]
   return header, payload + png


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
   ns["__plot__"] = _plot("distribution", {"values": values, "name": name, "what": kind, "qualitative": qualitative}, dict(PLOT_ELEMENTS["distribution"][1]), {"data": DISTRIBUTIONS["count"][1]})
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
   PNG (200 dpi) or PDF, by the extension of the path.
   """
   ns = _spaces[request["tab"]]
   if "__plot__" not in ns:
      raise ValueError("no plot to save")
   path = request["path"]
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


def _derived(ns, tab, lines):
   """
   A new tab ``tab`` whose network is derived from the network of the namespace ``ns`` (induced
   subgraph, ego network, backbone): its namespace copies the variables, the code, the style and
   the node size of ``ns``, then runs ``lines``. Replies with the code, a ``summary`` and the
   drawing; the new namespace is dropped if the lines fail. If ``ns`` is not drawn, the new
   network is not drawn either while it is :func:`too_large`.
   """
   import copy
   _spaces[tab] = {k: v for k, v in ns.items() if not k.startswith("__")}
   _spaces[tab].update(__code__ = list(ns["__code__"]), __style__ = copy.deepcopy(ns.get("__style__", {})), __size__ = ns.get("__size__"))
   try:
      code = _run(tab, lines)
   except BaseException:
      _spaces.pop(tab, None)
      raise
   new = _spaces[tab]
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


def _select(request):
   """
   Command ``select``: positions (in the drawing order) of the nodes (``kind = "nodes"``) or edges
   (``"edges"``) of the drawn network of a tab whose ``attribute`` satisfies ``op`` (``"="``,
   ``"≠"``, ``"<"``, ``"≤"``, ``">"``, ``"≥"`` or ``"contains"``) with ``value`` (a text: a number
   for numeric comparisons; texts of qualitative values are compared, see :func:`text`; with
   ``contains``, a part of the text, or one of the values of a node with several). Missing values
   never match.
   """
   net = _drawn(_spaces[request["tab"]])
   kind, op, value = request["kind"], request["op"], str(request["value"]).strip()
   values = node_values(net, request["attribute"]) if kind == "nodes" else edge_values(net, request["attribute"])
   if values.dtype.kind == "f" and op != "contains":
      x = float(value)
      with np.errstate(invalid = "ignore"):
         hit = {"=": values == x, "≠": (values != x) & ~np.isnan(values), "<": values < x, "≤": values <= x, ">": values > x, "≥": values >= x}[op]
   else:
      texts = [text(v) for v in values]
      def match(t, v):
         if t is None:
            return False
         if op == "contains":
            return value in t or value in [p.strip() for p in t.split(",")]
         try:
            a, b = float(t), float(value)
         except ValueError:
            a, b = t, value
         return {"=": a == b, "≠": a != b, "<": a < b, "≤": a <= b, ">": a > b, "≥": a >= b}[op] if type(a) is type(b) else op == "≠"
      hit = np.array([match(t, v) for t, v in zip(texts, values)], dtype = bool)
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
BACKBONE_COLORS = {"edges_pct": "#377eb8", "nodes_pct": "#e41a1c", "components": "#999999", "first_isolate": "#4daf4a", "first_component_break": "#984ea3", "max_separation": "#ff7f00"}


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


def backbone_plot(test, found, components = False, left = False, right = False, dpi = 100):
   """
   Plot of the backbone dialog: for each threshold of ``test`` (:func:`~graphoskop.test_bb_thresholds`),
   the percentage of edges kept and of nodes that are not isolated, and with ``components`` the
   number of components with more than one node, on a second y axis; the thresholds of ``found``
   (:func:`~graphoskop.find_bb_thresholds`: first isolate, first component break, max separation)
   as dashed vertical lines. The threshold axis covers the range of :func:`backbone_range` (tails
   left out unless ``left``, ``right``), with the scale of :func:`backbone_scale` of the thresholds in
   it. Fonts and ticks as in ``_figure``.

   :return: PNG bytes; the box of the axes in the image ``[left, top, right, bottom]`` (pixels) with
      the limits of the x axis ``[xmin, xmax]``, and the scale of the x axis, for the interface to
      draw the chosen threshold; and the ``tails`` of :func:`backbone_range`.
   """
   import io
   import matplotlib
   matplotlib.use("Agg")
   from matplotlib.figure import Figure
   with matplotlib.rc_context({"font.family": _plot_font(), "font.size": 12.}):
      fig = Figure(figsize = (7.6, 4.2))
      ax = fig.add_subplot()
      x = test["threshold"].to_numpy()
      lo, hi, tails = backbone_range(x, left, right)
      scale = backbone_scale(x[(x >= lo) & (x <= hi)])
      if scale["kind"] == "symlog":
         ax.set_xscale("symlog", linthresh = scale["linthresh"])
      ax.plot(x, test["edges_pct"], color = BACKBONE_COLORS["edges_pct"], lw = 2, label = "% edges kept")
      ax.plot(x, test["nodes_pct"], color = BACKBONE_COLORS["nodes_pct"], lw = 2, label = "% nodes not isolated")
      for name, thr in zip(found["criterion"], found["threshold"]):
         if name in BACKBONE_COLORS and np.isfinite(thr) and lo <= thr <= hi:
            ax.axvline(thr, color = BACKBONE_COLORS[name], ls = "--", lw = 1.5, label = name.replace("_", " "))
      ax.set_xlim(lo, hi)
      ax.set_ylim(0, 102)
      ax.set_xlabel("threshold")
      ax.set_ylabel("% of the network")
      ax.tick_params(which = "both", direction = "in", top = True, right = not components)
      handles, labels = ax.get_legend_handles_labels()
      if components:
         ax2 = ax.twinx()
         line, = ax2.plot(x, test["components"], color = BACKBONE_COLORS["components"], lw = 2, label = "components")
         ax2.set_ylabel("components")
         ax2.set_ylim(bottom = 0)
         ax2.tick_params(which = "both", direction = "in")
         handles.append(line)
         labels.append("components")
      ax.legend(handles, labels, frameon = False, fontsize = 10, loc = "center right")
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
   ``components`` and the ``left`` and ``right`` tails (:func:`backbone_plot`); replies with the
   ``box`` of its axes, the ``scale`` of its x axis, and its PNG.
   """
   b = _spaces[request["tab"]]["__backbone__"]
   png, box, scale, _ = backbone_plot(b["test"], b["found"], bool(request.get("components")), bool(request.get("left")), bool(request.get("right")))
   return {"box": box, "scale": scale}, png


def _backbone_stats(request):
   """
   Command ``backbone_stats``: the backbone of a tab at ``threshold`` (one row of
   :func:`~graphoskop.test_bb_thresholds`): ``nodes``, ``nodes_pct``, ``edges``, ``edges_pct``,
   ``components``.
   """
   ns = _spaces[request["tab"]]
   b = ns["__backbone__"]
   thr = float(request["threshold"])
   row = gsk.test_bb_thresholds(_drawn(ns), thr, thr, 1., score = b["score"]).iloc[0]
   return {k: float(row[k]) for k in ("nodes", "nodes_pct", "edges", "edges_pct", "components")}, b""


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
         chosen = ast.literal_eval(arguments["sets"])
         chosen = [chosen] if isinstance(chosen, str) else list(chosen)
         net = ns["net"]
         names = set(net.node_attributes) | set(net.qualitative_node_attributes) | set(net.edge_attributes) | set(net.qualitative_edge_attributes)
         counts = next(c for c in ["count"] + [f"count_{k}" for k in range(2, 1000)] if c not in names)
         keep = [a for a in chosen if a in net.qualitative_node_attributes]
         arguments = dict(arguments, sets = f"groups(net, {literal(chosen[0]) if len(chosen) == 1 else literal(chosen)})", counts = literal(counts), qualitative = literal(keep))
      extra = "".join(f", {k} = {v.strip()}" for k, v in arguments.items())
      if function in TRANSFORMATIONS:   # of the network itself (not of the incidence network drawn)
         note = [GROUPS_NOTE] if function == "quotient_network" and GROUPS_NOTE not in ns["__code__"] else []
         lines = note + [f"net = gsk.{function}(net{extra})"] + (["del inc"] if t == "inc" else [])
      elif function in PROJECTIONS:
         lines = [f"net = gsk.{function}({t}{extra})"] + (["del inc"] if t == "inc" else [])
      else:
         lines = [f"{t} = gsk.{function}({t}{extra})"]
   return _derived(ns, request["new_tab"], lines)


#: Switches of :func:`~graphoskop.extract_backbone` offered by the backbone dialog: keep the nodes
#: left without edges, reconnect the components split by the threshold (which reconnects the
#: isolated nodes too), reconnect the isolated nodes.
BACKBONE_OPTIONS = ["keep_isolates", "reconnect_components", "reconnect_isolates"]


#: Functions of Edit › Transform: transformations of the network of the library, whose result opens
#: in a new tab (:func:`_extract`), e.g. ``net = gsk.to_undirected(net)``.
TRANSFORMATIONS = ["flip_edge_direction", "remove_isolates", "remove_parallel_edges", "remove_self_loops", "to_strongly_connected", "to_undirected", "network_condensation", "quotient_network", "clique_expansion", "clique_promotion"]
#: Parameters of the transformations not shown in the parameter dialog: the network, the edges to
#: flip (all), the sign attribute (``"sign"``), the size of the batches (memory, not results), and
#: the counts and qualitative attributes of the quotient (set by :func:`_extract`).
TRANSFORMATION_HIDDEN = {"net", "edges", "sign", "batch_size", "counts", "qualitative"}
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
   Command ``simulations``: the names of the runs of a tab (``tables``) and the states of each
   (``states``: the columns without confidence intervals, e.g. ``["S", "I", "R"]``).
   """
   return _runs(_spaces[request["tab"]]), b""


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
   ns["__plot__"] = _plot("epidemics", {"tables": tables, "states": list(states)}, {k: PLOT_COLORS[i % len(PLOT_COLORS)] for i, k in enumerate(keys)}, {k: "lines" for k in keys}, {k: k for k in keys}, ylabel = ylabel)
   peaks = [f"{n}, {x}: peak {df[x].max():.3g} at step {df[x].idxmax()}" for n, df in tables.items() for x in states if x in df]
   header = dict(_plot_header(ns["__plot__"]), text = f"Fraction of nodes in state {', '.join(states)}. " + "; ".join(peaks) + ".", warnings = [])
   return header, _render(ns["__plot__"], "png")


def last_layout(code):
   """
   The function of the last line of code that changes the drawing: ``"bundle_edges"`` for the
   line bundling the edges (:data:`BENDS`), else the function of :data:`LAYOUTS` of the last line
   calling one, or of a line moving nodes (:func:`move_nodes`) or changing the network (``net =
   gsk.…``); ``None`` if none. Edges are drawn bundled after ``bundle_edges``, the network as a
   matrix after ``matrix_layout``; anything else draws nodes and straight edges again.

   :param code: Lines of code of a tab.
   """
   for line in reversed(code):
      if line.startswith(BENDS):
         return "bundle_edges"
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
   always the user's choice, in the visual elements panel).
   """
   ns = _spaces[request["tab"]]
   function, arguments = request["function"], dict(request.get("arguments") or {})
   code = _run(request["tab"], layout_code(ns, function, arguments))
   return _draw(ns, code, True)


def _layout(request):
   """
   Command ``layout``: runs the code of :func:`layout_code` (``function`` and ``arguments`` of the
   request) in the namespace of the tab, and replies with the code and the drawing.
   ``remove_overlaps`` gets as ``size`` the diameters of the drawn nodes (:func:`size_code`).
   The drawing keeps the default node diameter after transformations of the layout and edge
   bundling (:data:`TRANSFORMS`).
   """
   ns = _spaces[request["tab"]]
   function, arguments = request.get("function", "grid_layout"), dict(request.get("arguments") or {})
   if function == "remove_overlaps" and "size" not in arguments:
      arguments["size"] = size_code(ns)
   lines = layout_code(ns, function, arguments)
   if "node_sizes(" in arguments.get("size", ""):
      lines.insert(-1, NODE_SIZES_NOTE)
   code = _run(request["tab"], lines)
   return _draw(ns, code, function in TRANSFORMS)


def size_code(ns):
   """
   Source code of the diameters of the drawn nodes of a tab, for ``remove_overlaps``: the default
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
   ns["__plot__"] = _plot("randomization", {"distribution": r["distribution"], "observed": r["observed"], "measure": label, "mean": r["mean"], "std": r["std"], "above": r["above"], "below": r["below"], "equal": r["equal"]}, dict(PLOT_ELEMENTS["randomization"][1]), {"null": "bars"})
   png = _render(ns["__plot__"], "png")
   header["test"] = dict(_plot_header(ns["__plot__"]), code = code, measure = label, warnings = [])
   header["parts"] = header.get("parts", []) + [len(png)] if header["nodes"] else [len(png)]
   return header, payload + png


def _signature(request):
   """
   Command ``signature``: :func:`signature` of a generator, a layout or a community discovery
   function, for the drawn network of the tab ``tab`` (if given); with ``fit``, of a generator
   fitted to the network of the tab.
   """
   if request["function"] not in GENERATORS | LAYOUTS | COMMUNITIES | set(CENTRALITIES) | set(EPIDEMICS) | set(BACKBONES) | FILTERS | set(PROJECTIONS) | set(TRANSFORMATIONS) | {"nestedness"}:
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


#: Commands understood by the worker.
COMMANDS = {"preview": _preview, "load": _load, "generate": _generate, "signature": _signature, "layout": _layout, "communities": _communities, "style": _style, "breaks": _breaks, "attributes": _attributes, "table": _table, "replay": _replay, "close": _close, "centrality": _centrality, "save_plot": _save_plot, "replot": _replot, "move": _move, "edit": _edit, "select": _select, "simulate": _simulate, "simulations": _simulations, "epidemic_plot": _epidemic_plot, "undo": _undo, "spread": _spread, "animation": _animation, "find": _find, "backbone": _backbone, "backbone_plot": _backbone_plot, "backbone_stats": _backbone_stats, "extract": _extract, "distribution": _distribution, "remove_attribute": _remove_attribute, "drawing": _drawing, "fit": _fit, "nestedness": _nestedness}

#: Functions of the Layout menu: layouts of node-link diagrams, the matrix layout (drawn as the
#: adjacency matrix), transformations of layouts, and edge bundling.
LAYOUTS = {"circular_layout", "force_directed_layout", "grid_layout", "hierarchical_layout", "matrix_layout", "umap_layout", "mirror_layout", "rotate_layout", "remove_overlaps", "bundle_edges"}

#: Functions of :data:`LAYOUTS` transforming the current layout (or bundling its edges): the drawn
#: nodes keep their size, and ``remove_overlaps`` gets it as ``size`` (unless given), to remove
#: the overlaps on screen.
TRANSFORMS = {"mirror_layout", "rotate_layout", "remove_overlaps", "bundle_edges"}
#: Start of the line of code bundling the edges: the curves are stored in ``bends``.
BENDS = "bends = gsk.bundle_edges"
#: Comment before code using :func:`node_sizes`, which is defined by the interface.
NODE_SIZES_NOTE = "# node_sizes: relative sizes of the drawn nodes, a function of the graphical interface"
#: Comment before the first code moving nodes with :func:`move_nodes`, defined by the interface.
MOVE_NOTE = "# move_nodes: positions of nodes dragged in the graphical interface"

#: Functions of Analysis › Community discovery: the community discovery functions of the library.
COMMUNITIES = {"bigclam_communities", "block_model_communities", "demon_communities", "hierarchical_link_communities", "hierarchical_random_graph_communities", "infomap_communities", "label_propagation_communities", "leiden_communities", "line_graph_communities", "oslom_communities"}
#: Functions of Analysis › Centrality: name of the attribute they write by default, and what
#: they give: ``"nodes"`` (a value per node), ``"edges"`` (a value per edge) or ``"hits"`` (hub
#: and authority scores, written as ``<attribute>_hub`` and ``<attribute>_authority``).
CENTRALITIES = {"alpha_centrality": ("alpha", "nodes"), "betweenness_centrality": ("betweenness", "nodes"), "closeness_centrality": ("closeness", "nodes"), "degree": ("degree", "nodes"), "edge_betweenness_centrality": ("edge_betweenness", "edges"), "eigenvector_centrality": ("eigenvector", "nodes"), "flow_centrality": ("flow", "nodes"), "harmonic_centrality": ("harmonic", "nodes"), "hits": ("hits", "hits"), "katz_centrality": ("katz", "nodes"), "k_core_centrality": ("coreness", "nodes"), "pagerank": ("pagerank", "nodes"), "reach_centrality": ("reach", "nodes"), "springrank": ("springrank", "nodes"), "discrete_core_periphery": ("core_discrete", "core"), "continuous_core_periphery": ("core_continuous", "core"), "rombach_core_periphery": ("core_rombach", "core")}
#: Parameters of the centralities not shown in the parameter dialog: the interface computes them
#: for all components, with the default numerical settings.
CENTRALITY_HIDDEN = {"net", "component", "node", "distribution", "lengths", "batch_size", "n_jobs", "tol", "max_iter", "personalization", "progress"}
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
