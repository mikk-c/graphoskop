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
NUMBERS = {"spacing": (0., 0.1), "gap": (0., 0.5), "edge_length": (0., 0.1), "layer_spacing": (0., 0.1), "size": (0., 0.1), "margin": (0., 0.01), "t1": (1., 0.1), "t2": (1., 0.1), "r": (0., 0.01), ("geometric_graph", "p"): (1., 0.5), ("lfr_benchmark_graph", "beta"): (0., 0.1), ("lfr_benchmark_graph", "k"): (1., 1.), ("holme_kim_graph", "k"): (1., 1.), ("copying_graph", "k"): (1., 1.), ("link_selection_graph", "k"): (1., 1.), "resolution": (0., 0.1), "randomness": (0., 0.001), ("katz_centrality", "alpha"): (0., 0.01), ("katz_centrality", "beta"): (0., 0.1), ("alpha_centrality", "alpha"): (0., 0.1), ("alpha_centrality", "exo"): (0., 0.1), ("springrank", "alpha"): (0., 0.1)}

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
   :func:`~graphoskop.load_network`, :func:`~graphoskop.load_hypergraph` or
   :func:`~graphoskop.load_simplicial_complex`, with only the arguments that differ from their
   defaults, followed by :func:`~graphoskop.signed_network` for signed networks.

   :param spec: Dictionary with ``edges`` (path), ``nodes`` (path or ``None``), ``sep``,
      ``header``, ``kind`` (``"network"``, ``"hypergraph"`` or ``"simplicial"``), ``source`` and
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


def signature(function, net = None):
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
   :return: Dictionary with ``params``.
   """
   f = getattr(gsk, function)
   doc = inspect.getdoc(f) or ""
   descriptions = {m.group(1): " ".join(m.group(2).split()) for m in re.finditer(r":param (\w+): (.*?)(?=\n:|\Z)", doc, re.S)}
   hidden = HIDDEN if function in LAYOUTS else (CENTRALITY_HIDDEN if function in CENTRALITIES else {"net", "progress"})   # community discovery shows ``attribute``
   params = []
   for p in inspect.signature(f).parameters.values():
      if p.name in hidden:
         continue
      if p.default is inspect.Parameter.empty or p.default is None:
         shown = SHOWN_DEFAULTS.get((function, p.name), SHOWN_DEFAULTS.get(p.name, ""))
      else:
         shown = literal(p.default)
      hint = HINTS.get((function, p.name), HINTS.get(p.name, "required"))
      param = {"name": p.name, "default": shown, "hint": hint, "doc": descriptions.get(p.name, "")}
      param["widget"] = widget(function, p.name, shown, hint, net)
      params.append(param)
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
   * ``{"kind": "choice", "options": [...]}`` for :data:`OPTIONS` (a menu);
   * ``{"kind": "attribute", "choices": "nodes" or "edges", "multiple": bool}`` for the
     attribute names of :data:`CHOICES`, in layouts and analyses (a menu of the attributes of
     the network, after the values of :data:`EXTRA_CHOICES`; check boxes for several);
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
   options = OPTIONS.get((function, name), OPTIONS.get(name))
   if options is not None:
      return {"kind": "choice", "options": options}
   choice = CHOICES.get((function, name), CHOICES.get(name) if function in LAYOUTS or (function in COMMUNITIES | set(CENTRALITIES) and name == "weight") else None)
   if choice is not None:
      return {"kind": "attribute", "choices": choice[0], "multiple": choice[1], "extra": EXTRA_CHOICES.get((function, name), [])}
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


def default_style(net):
   """
   Style of a new drawing: ``nodes`` and ``edges`` (colors, see :func:`colors`), ``node_size`` and
   ``edge_width`` (see :func:`sizes`), ``node_opacity`` and ``edge_opacity`` (see
   :func:`opacities`). Nodes are :data:`NODE_COLOR` and edges :data:`EDGE_COLOR`, all of the
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
   return {"nodes": nodes, "edges": {"attribute": None, "color": EDGE_COLOR}, "node_size": {"attribute": None, "size": 1.}, "edge_width": {"attribute": None, "size": 1.}, "node_opacity": {"attribute": None, "opacity": 1.}, "edge_opacity": {"attribute": None, "opacity": 1.}}


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
   curve, with :func:`subdivisions` segments). Each instance is a 2D transform, row by row
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

   :param net: A :class:`~graphoskop.Network` with the numeric node attributes
      ``<attribute>_x`` and ``<attribute>_y``.
   :param size: Default diameter of the nodes, or ``None`` for :func:`node_size`.
   :param style: Style (see :func:`default_style`), or ``None`` for the default one.
   :param attribute: Prefix of the position attributes.
   :param bends: Curves of the edges, as returned by :func:`~graphoskop.bundle_edges`, or
      ``None`` for straight edges.
   :param matrix: Name of the node attribute of a matrix layout, or ``None`` for a node-link
      drawing.
   :return: Dictionary with the node and edge buffers (float32 arrays), whether edges are curves
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
      x = torch.cat(net.get_node_attribute(f"{attribute}_x", None)).double().cpu().numpy()
      y = -torch.cat(net.get_node_attribute(f"{attribute}_y", None)).double().cpu().numpy()
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
      E[:, 12], E[:, 13], E[:, 14] = P[first + 3, 0], P[first + 3, 1], w[e[first]]
      E[:, 8:12] = c[e[first]]
   else:
      dx, dy = x[t] - x[s], y[t] - y[s]
      length = np.maximum(np.hypot(dx, dy), 1e-12)
      E = np.zeros((m, EDGE_STRIDE), dtype = np.float32)
      E[:, 0], E[:, 4] = dx, dy                                  # quad's x axis: along the edge
      E[:, 1], E[:, 5] = -dy / length * w, dx / length * w       # y axis: across it, the relative width
      E[:, 3], E[:, 7] = (x[s] + x[t]) / 2, (y[s] + y[t]) / 2
      E[:, 8:] = c
   biggest = float(d.max()) if n else size
   grid, order, starts = pick_grid(x, y, biggest)
   ids = "\x1f".join(str(i) for i in _ids(net)).encode("utf-8")
   box = [float(x.min()), float(y.min()), float(x.max()), float(y.max())] if n else [0., 0., 1., 1.]
   if matrix is not None and len(E):
      box = [min(box[0], float(E[:, 3].min()) - 0.5), min(box[1], float(E[:, 7].min()) - 0.5), max(box[2], float(E[:, 3].max()) + 0.5), max(box[3], float(E[:, 7].max()) + 0.5)]
   return {"nodes": N, "edges": E, "curved": E.shape[1] == CURVE_STRIDE, "cells": matrix is not None, "ids": ids, "grid": grid, "order": order, "starts": starts, "box": box, "size": size, "biggest": biggest}


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
      texts, _ = distinct(v)
      x = numbers(v)
      ok = x[~np.isnan(x)]
      lo, hi = (float(ok.min()), float(ok.max())) if len(ok) else (0., 1.)
      return {"name": name, "numeric": v.dtype.kind == "f", "min": lo, "max": hi, "positive": lo > 0, "count": len(texts), "values": texts[:MAX_VALUES]}
   nodes = [describe(a, node_values(net, a)) for a in list(net.node_attributes) + list(net.qualitative_node_attributes)]
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


def _tables(ns):
   """
   Tables of the data panel of a tab (cached until its code runs again): one row per node of the
   drawn network (in the drawing order, with its id and attributes), and one per edge (source and
   target ids, then the attributes), with the positions of the endpoints of the edges.
   """
   if ns.get("__tables__") is None:
      import pandas as pd
      net = _drawn(ns)
      ids = _ids(net)
      nodes = pd.DataFrame({"node": ids})
      for a in list(net.node_attributes) + list(net.qualitative_node_attributes):
         nodes[a] = node_values(net, a)
      s, t = _edges(net)
      edges = pd.DataFrame({"source": ids[s], "target": ids[t]})
      for a in list(net.edge_attributes) + list(net.qualitative_edge_attributes):
         edges[a] = edge_values(net, a)
      ns["__tables__"] = {"nodes": nodes, "edges": edges, "s": s, "t": t}
   return ns["__tables__"]


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
   Command ``table``: a page of a table of the data panel. ``kind`` is ``"nodes"`` (all nodes, or
   the attributes of node ``node``, a position in the drawing order, as pairs attribute and
   value), ``"edges"`` (all edges, or those of node ``node``) or ``"network"`` (:func:`summary`);
   rows ``offset`` to ``offset + count`` (texts, ``""`` for missing values), with the
   ``columns``, whether each is ``numeric`` (the interface aligns them to the right) and the
   ``total`` number of rows.
   """
   ns = _spaces[request["tab"]]
   kind, node = request["kind"], request.get("node")
   offset, count = int(request.get("offset", 0)), int(request.get("count", 200))
   if kind == "network":
      rows = summary(ns["net"])
      return {"columns": ["property", "value"], "numeric": [False, False], "rows": rows, "total": len(rows)}, b""
   T = _tables(ns)
   if kind == "nodes" and node is not None:
      row = T["nodes"].iloc[int(node)]
      rows = [[str(c), text(row[c]) or ""] for c in T["nodes"].columns]
      return {"columns": ["attribute", "value"], "numeric": [False, False], "rows": rows, "total": len(rows)}, b""
   df = T[kind]
   if kind == "edges" and node is not None:
      df = df.iloc[np.flatnonzero((T["s"] == node) | (T["t"] == node))]
   page = df.iloc[offset:offset + count]
   rows = [[text(v) or "" for v in r] for r in page.itertuples(index = False, name = None)]
   numeric = [df[c].dtype.kind in "fiu" for c in df.columns]
   return {"columns": [str(c) for c in df.columns], "numeric": numeric, "rows": rows, "total": len(df)}, b""


def _run(tab, code):
   """
   Runs lines of code in the namespace of a tab and adds them to its code.

   :return: The whole code of the tab.
   """
   ns = _spaces.setdefault(tab, {"gsk": gsk, "node_sizes": node_sizes, "__code__": ["import graphoskop as gsk", ""]})
   ns["__tables__"] = None
   exec("\n".join(code), ns)
   ns["__code__"] += code
   return "\n".join(ns["__code__"])


def _new(request, code):
   """
   Runs the code creating the network of a new tab, and replies with the code and a summary.
   """
   _spaces.pop(request["tab"], None)
   try:
      full = _run(request["tab"], code)
   except BaseException:   # also an abort
      _spaces.pop(request["tab"], None)
      raise
   net = _spaces[request["tab"]]["net"]
   return {"code": full, "summary": f"{net.number_of_nodes()} nodes, {net.number_of_edges()} edges"}, b""


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
   the header has the numbers of nodes and edges, the byte sizes of the parts of the payload (node
   buffer, edge buffer, ids, grid order, grid starts, in this order), the grid, the box, the
   default and largest node diameters and the style. The default node diameter of the tab is
   recomputed for the layout, unless ``keep`` (transformations of a layout and changes of style
   keep it). Edges are drawn as the curves of ``bends`` when no layout came after them, and the
   network as its adjacency matrix when the last layout is :func:`~graphoskop.matrix_layout`
   (:func:`last_layout`).
   """
   draw = ns.get("inc", ns["net"])
   last = last_layout(ns["__code__"])
   matrix = _matrix_attribute(ns, last) if last == "matrix_layout" else None
   if "pos_x" not in draw.node_attributes and matrix is None:
      return {"code": code, "nodes": 0}, b""
   ns["__style__"] = dict(default_style(draw), **ns.get("__style__", {}))
   bends = ns.get("bends") if last == "bundle_edges" else None
   d = drawing(draw, ns.get("__size__") if keep else None, ns["__style__"], bends = bends, matrix = matrix)
   ns["__size__"] = d["size"]
   parts = [d["nodes"].tobytes(), d["edges"].tobytes(), d["ids"], d["order"].tobytes(), d["starts"].tobytes()]
   header = {"code": code, "nodes": len(d["nodes"]), "edges": len(d["edges"]), "curved": d["curved"], "cells": d["cells"], "subdivisions": subdivisions(len(d["edges"])), "box": d["box"], "node_size": d["size"], "biggest": d["biggest"], "grid": d["grid"], "style": ns["__style__"], "parts": [len(p) for p in parts]}
   return header, b"".join(parts)


def centrality_code(ns, function, arguments):
   """
   Code computing a centrality of :data:`CENTRALITIES` for all the nodes (or edges) of the drawn
   network of a tab, and storing it as an attribute (:meth:`~graphoskop.Network.set_node_attribute`
   or :meth:`~graphoskop.Network.set_edge_attribute`), e.g.
   ``net.set_node_attribute("betweenness", gsk.betweenness_centrality(net, component = None),
   None)``. With ``plot`` (degree only), also the fit of a power law to the degree distribution,
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
   the ``verdict``, ``text`` and ``warnings`` of the fit).
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
   if plot:
      fit = ns["fit"]
      degrees = node_values(draw, name)
      ns["__plot__"] = {"kind": "degree", "data": {"degrees": degrees, "fit": fit}, "colors": dict(PLOT_ELEMENTS["degree"][1])}
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


def degree_plot(degrees, fit, colors = None, format = "png", path = None, dpi = 100):
   """
   Plot of the complementary cumulative degree distribution (the fraction of nodes with degree
   :math:`k` or more, log-log axes), with the best fit of :func:`~graphoskop.degree_power_law`
   from its smallest degree, drawn with matplotlib (needed for plots only), without a title.

   :param degrees: Degrees of the nodes.
   :param fit: Result of :func:`~graphoskop.degree_power_law` with ``ccdf = True``.
   :param colors: Colors of the elements of the plot (``data``, ``fit``, ``k_min``, see
      :data:`PLOT_ELEMENTS`), or ``None`` for the defaults.
   :param format: ``"png"`` or ``"pdf"``.
   :param path: File to write, or ``None`` to return the bytes.
   :param dpi: Resolution of PNG images (the figure is 8 by 5.5 inches).
   :return: The bytes of the image, or ``None`` with ``path``.
   """
   import io
   import matplotlib
   matplotlib.use("Agg")
   from matplotlib.figure import Figure
   colors = dict(PLOT_ELEMENTS["degree"][1], **(colors or {}))
   k = np.asarray(degrees, dtype = float)
   k = k[~np.isnan(k)]
   values, counts = np.unique(k[k > 0], return_counts = True)
   ccdf = (np.cumsum(counts[::-1])[::-1]) / max(len(k), 1)
   fig = Figure(figsize = (8, 5.5))
   ax = fig.add_subplot()
   ax.loglog(values, ccdf, "o", ms = 4, color = colors["data"], alpha = 0.8, label = "data")
   table = fit["ccdf"]
   label = {"power law": f"power law, α = {fit['exponent']:.3g}", "power law with exponential cutoff": f"power law with cutoff, α = {fit['exponent']:.3g}, λ = {fit['cutoff']:.3g}", "exponential": f"exponential, λ = {fit.get('rate', float('nan')):.3g}"}[fit["model"]]
   ax.loglog(table["degree"], table["fit"], "-", lw = 2, color = colors["fit"], label = f"{label} (k ≥ {fit['k_min']})")
   ax.axvline(fit["k_min"], color = colors["k_min"], ls = "--", lw = 1)
   ax.set_xlabel("degree k")
   ax.set_ylabel("fraction of nodes with degree ≥ k")
   ax.legend(frameon = False)
   fig.tight_layout()
   out = path if path is not None else io.BytesIO()
   fig.savefig(out, format = format, dpi = dpi)
   return None if path is not None else out.getvalue()


#: Plots of the plot window: their function and the colors of their elements, which the user can
#: change (key: default color), with the label of each in the window.
PLOT_ELEMENTS = {"degree": (degree_plot, {"data": "#377eb8", "fit": "#e41a1c", "k_min": "#999999"}, {"data": "Data", "fit": "Fit", "k_min": "Smallest degree of the fit"})}


def _render(plot, format, path = None, dpi = 100):
   """
   Draws a plot kept in a namespace (``kind``, ``data`` and ``colors``) with its function of
   :data:`PLOT_ELEMENTS`.
   """
   return PLOT_ELEMENTS[plot["kind"]][0](**plot["data"], colors = plot["colors"], format = format, path = path, dpi = dpi)


def _plot_header(plot):
   """
   What the plot window needs to let the user change colors: ``elements``, a list of [key, label,
   color].
   """
   labels = PLOT_ELEMENTS[plot["kind"]][2]
   return {"elements": [[k, labels[k], c] for k, c in plot["colors"].items()]}


def _replot(request):
   """
   Command ``replot``: draws the last plot of a tab again with new ``colors`` (element key to
   ``#rrggbb``), and replies with the PNG image as payload.
   """
   plot = _spaces[request["tab"]]["__plot__"]
   plot["colors"].update({k: v for k, v in (request.get("colors") or {}).items() if k in plot["colors"]})
   return _plot_header(plot), _render(plot, "png")


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


def last_layout(code):
   """
   The function of the last line of code that changes the drawing: ``"bundle_edges"`` for the
   line bundling the edges (:data:`BENDS`), else the function of :data:`LAYOUTS` of the last line
   calling one (``None`` if none). Edges are drawn bundled after ``bundle_edges``, the network as
   a matrix after ``matrix_layout``; any other layout draws nodes and straight edges again.

   :param code: Lines of code of a tab.
   """
   for line in reversed(code):
      if line.startswith(BENDS):
         return "bundle_edges"
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
   replies with the drawing.
   """
   ns = _spaces[request["tab"]]
   ns["__style__"] = request["style"]
   return _draw(ns, "\n".join(ns["__code__"]), True)


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


def _signature(request):
   """
   Command ``signature``: :func:`signature` of a generator, a layout or a community discovery
   function, for the drawn network of the tab ``tab`` (if given).
   """
   if request["function"] not in GENERATORS | LAYOUTS | COMMUNITIES | set(CENTRALITIES):
      raise ValueError(f"unknown function: {request['function']}")
   tab = request.get("tab")
   return signature(request["function"], _drawn(_spaces[tab]) if tab in _spaces else None), b""


def _replay(request):
   """
   Command ``replay``: runs the whole ``code`` of a tab in a new namespace (e.g. after switching
   to another Python environment), with the ``style`` of its drawing, and replies with the code
   and the drawing.
   """
   ns = {"node_sizes": node_sizes, "__code__": request["code"].split("\n")}
   if request.get("style"):
      ns["__style__"] = request["style"]
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
COMMANDS = {"preview": _preview, "load": _load, "generate": _generate, "signature": _signature, "layout": _layout, "communities": _communities, "style": _style, "breaks": _breaks, "attributes": _attributes, "table": _table, "replay": _replay, "close": _close, "centrality": _centrality, "save_plot": _save_plot, "replot": _replot}

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

#: Functions of Analysis › Community discovery: the community discovery functions of the library.
COMMUNITIES = {"bigclam_communities", "block_model_communities", "demon_communities", "hierarchical_link_communities", "hierarchical_random_graph_communities", "infomap_communities", "label_propagation_communities", "leiden_communities", "line_graph_communities", "oslom_communities"}
#: Functions of Analysis › Centrality: name of the attribute they write by default, and what
#: they give: ``"nodes"`` (a value per node), ``"edges"`` (a value per edge) or ``"hits"`` (hub
#: and authority scores, written as ``<attribute>_hub`` and ``<attribute>_authority``).
CENTRALITIES = {"alpha_centrality": ("alpha", "nodes"), "betweenness_centrality": ("betweenness", "nodes"), "closeness_centrality": ("closeness", "nodes"), "degree": ("degree", "nodes"), "edge_betweenness_centrality": ("edge_betweenness", "edges"), "eigenvector_centrality": ("eigenvector", "nodes"), "flow_centrality": ("flow", "nodes"), "harmonic_centrality": ("harmonic", "nodes"), "hits": ("hits", "hits"), "katz_centrality": ("katz", "nodes"), "k_core_centrality": ("coreness", "nodes"), "pagerank": ("pagerank", "nodes"), "reach_centrality": ("reach", "nodes"), "springrank": ("springrank", "nodes")}
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
