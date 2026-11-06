"""
Networks from and to the files of other programs (Pajek, UCINET, adjacency lists, DOT, GML,
GraphML, GEXF, Cytoscape JSON) and the objects of other libraries (networkx, igraph, graph-tool).

Nodes and edges are read and written with their attributes; the visual attributes the formats
define (e.g. node colors and positions) are node and edge attributes with the names of
:data:`VISUAL_ATTRIBUTES`, which the graphical interface draws.
"""

import colorsys
import json
import os
import re
import shlex
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .network import BIPARTITE, LAYER, load_network


#: Formats of :func:`read_network`, by file extension.
FORMATS = {".net": "pajek", ".paj": "pajek", ".dl": "ucinet", ".##h": "ucinet_binary", ".##d": "ucinet_binary", ".adj": "adjlist", ".adjlist": "adjlist", ".dot": "dot", ".gv": "dot", ".gml": "gml", ".graphml": "graphml", ".gexf": "gexf", ".cyjs": "cytoscape", ".json": "cytoscape"}

#: Names of the attributes holding the visual attributes of the formats, nodes then edges:
#: positions ``pos_x`` and ``pos_y`` (y pointing up, as the layouts of graphoskop; read at
#: ``pos_z`` too when given), ``color`` (``"#rrggbb"``, or a color name as written in the file),
#: ``size`` (diameter, in the units of the file), ``shape`` (the name of the format, lower case,
#: e.g. ``"ellipse"``, ``"box"``), ``opacity`` (0 to 1), ``border_color``, ``border_width``,
#: ``label``, ``label_color``; edges: ``color``, ``width``, ``style`` (e.g. ``"dashed"``),
#: ``opacity``, ``label``.
VISUAL_ATTRIBUTES = (("pos_x", "pos_y", "color", "size", "shape", "opacity", "border_color", "border_width", "label", "label_color"), ("color", "width", "style", "opacity", "label"))


def read_network(path, format = None, directed = None, device = "cpu"):
   """
   Load a network from a file of another program:

   * ``"pajek"`` (``.net``, ``.paj``): ``*Vertices`` (with labels, coordinates, ``ic`` color,
     ``bc`` border color, ``bw`` border width, ``x_fact`` size, shapes ``ellipse``, ``box``,
     ``diamond``, ``triangle``), ``*Arcs`` and ``*Edges`` (weight, ``c`` color, ``w`` width,
     ``p`` pattern, ``l`` label), ``*Arcslist``, ``*Edgeslist`` and ``*Matrix``. Two-mode
     networks (``*Vertices n n1``) are bipartite: the first ``n1`` vertices are of type 0;
   * ``"ucinet"`` (``.dl``): formats ``fullmatrix``, ``edgelist1``, ``nodelist1`` and, for
     two-mode networks (``nr``, ``nc``: bipartite, rows of type 0), ``fullmatrix``,
     ``edgelist2`` and ``nodelist2``; labels given or embedded. Several relations (``nm``, full
     matrices, with ``matrix labels``) make a multilayer network, one layer per relation (every
     node in every layer, no interlayer edges);
   * ``"ucinet_binary"`` (``.##h`` and ``.##d``, either of the two files, the other next to it):
     the binary data of UCINET 6, its matrices and their labels (row, column and relation),
     as above (two-mode when the row and column labels differ);
   * ``"adjlist"`` (``.adj``, ``.adjlist``): one line per node, its id then the ids of its
     neighbors, separated by white space; lines starting with ``#`` are comments. Undirected unless
     ``directed``: an edge listed on the lines of both its nodes is read once;
   * ``"dot"`` (``.dot``, ``.gv``, Graphviz): nodes and edges with their attributes (also the
     defaults of ``node [...]`` and ``edge [...]``, and those of subgraphs), ``pos``, ``fillcolor``
     (else ``color``), ``color`` (the border, with ``fillcolor``), ``width`` (inches), ``shape``,
     ``penwidth``, ``label``, ``fontcolor``, ``style`` of edges; only the first graph of the file;
   * ``"gml"`` (``.gml``): ``node`` and ``edge`` with their keys, and their ``graphics`` (``x``,
     ``y``, ``w``, ``type``, ``fill``, ``outline``, ``outline_width``; ``fill``, ``width``,
     ``style`` of edges);
   * ``"graphml"`` (``.graphml``): ``data`` of nodes and edges, typed by their ``key`` (with
     defaults), nested graphs flattened; the graphics of yEd (``y:Geometry``, ``y:Fill``,
     ``y:BorderStyle``, ``y:Shape``, ``y:NodeLabel``, ``y:LineStyle``) and the attributes of Gephi
     (``x``, ``y``, ``r``, ``g``, ``b``, ``size``);
   * ``"gexf"`` (``.gexf``, Gephi): attribute values (with defaults) and ``viz`` elements
     (``color``, ``position``, ``size``, ``shape``; ``color``, ``thickness``, ``shape`` of
     edges), ``weight`` of edges;
   * ``"cytoscape"`` (``.cyjs``, ``.json``, Cytoscape and Cytoscape.js): ``data`` and ``position``
     of the elements; undirected unless the file says ``"directed": true``.

   Visual attributes become the attributes of :data:`VISUAL_ATTRIBUTES`; others keep their names.
   Values are numeric when all the values of an attribute are numbers. Node ids are the labels of
   the nodes (Pajek, GML, GEXF; ``name`` in Cytoscape) when every node has a different one, else
   the ids of the file. Positions with y pointing down (Pajek, GML, yEd, Cytoscape) are flipped.

   :param path: Path of the file.
   :param format: One of the formats above, or ``None`` for the one of the extension
      (:data:`FORMATS`).
   :param directed: Whether the edges are directed, or ``None`` as the file says (UCINET: a
      symmetric matrix or edge list is undirected).
   :param device: Device where the tensors are created.
   :return: The loaded :class:`~graphoskop.Network`.
   :raises ValueError: If the format is unknown or the file cannot be read in it.

   Examples:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.read_network("karate.gml")
      net = gsk.read_network("graph.txt", format = "adjlist", directed = True)
   """
   format = format or FORMATS.get(os.path.splitext(path)[1].lower())
   if format not in _READERS:
      raise ValueError(f"unknown network format of {path!r}: one of {sorted(_READERS)}")
   text = None
   if format != "ucinet_binary":
      with open(path, encoding = "utf-8", errors = "replace") as f:
         text = f.read()
   nodes, edges, found, bipartite, *layers = _READERS[format](path if format == "ucinet_binary" else text, directed)
   return _network(nodes, edges, found if directed is None else directed, device, bipartite, layers[0] if layers else None)


def from_networkx(graph, device = "cpu"):
   """
   Convert a networkx graph (also directed and multigraphs) into a network, with the node and edge
   attributes of its data dictionaries (attributes holding two or three numbers, e.g. ``pos``,
   become ``pos_x``, ``pos_y``, ``pos_z``). Node ids that are neither integers nor texts (e.g.
   tuples) become texts.

   :param graph: A ``networkx.Graph``, ``DiGraph``, ``MultiGraph`` or ``MultiDiGraph``.
   :param device: Device where the tensors are created.
   :return: The :class:`~graphoskop.Network`.
   """
   nodes = {v: dict(d) for v, d in graph.nodes(data = True)}
   edges = [(u, v, dict(d)) for u, v, d in graph.edges(data = True)]
   return _network(nodes, edges, graph.is_directed(), device)


def from_igraph(graph, device = "cpu"):
   """
   Convert an igraph graph into a network, with its vertex and edge attributes. The vertex
   attribute ``name``, if any, gives the node ids, else they are the vertex indices.

   :param graph: An ``igraph.Graph``.
   :param device: Device where the tensors are created.
   :return: The :class:`~graphoskop.Network`.
   """
   names = graph.vs.attributes()
   ids = graph.vs["name"] if "name" in names else list(range(graph.vcount()))
   columns = {a: graph.vs[a] for a in names if a != "name"}
   nodes = {v: {a: c[i] for a, c in columns.items()} for i, v in enumerate(ids)}
   columns = {a: graph.es[a] for a in graph.es.attributes()}
   edges = [(ids[s], ids[t], {a: c[k] for a, c in columns.items()}) for k, (s, t) in enumerate(graph.get_edgelist())]
   return _network(nodes, edges, graph.is_directed(), device)


def from_graph_tool(graph, device = "cpu"):
   """
   Convert a graph-tool graph into a network, with its vertex and edge property maps (vector
   properties of two or three numbers, e.g. the ``pos`` of its layouts, become ``pos_x``,
   ``pos_y``, ``pos_z``). Node ids are the vertex indices.

   :param graph: A ``graph_tool.Graph``.
   :param device: Device where the tensors are created.
   :return: The :class:`~graphoskop.Network`.
   """
   ids = [int(v) for v in graph.vertices()]
   nodes = {v: {} for v in ids}
   for name, p in graph.vertex_properties.items():
      for v, x in zip(ids, _property(p, graph.vertices())):
         nodes[v][name] = x
   order = list(graph.edges())
   edges = [(int(e.source()), int(e.target()), {}) for e in order]
   for name, p in graph.edge_properties.items():
      for e, x in zip(edges, _property(p, order)):
         e[2][name] = x
   return _network(nodes, edges, graph.is_directed(), device)


def _property(p, items):
   """
   Values of a graph-tool property map, in the order of ``items`` (vertices or edges).
   """
   return [list(x) if hasattr(x, "__len__") and not isinstance(x, str) else x for x in (p[i] for i in items)]


def _network(nodes, edges, directed, device = "cpu", bipartite = False, layers = None):
   """
   Network of nodes (dictionary from id to its attributes) and edges (source, target,
   attributes), through :func:`~graphoskop.load_network`. With ``layers`` (names), a multilayer
   network: every node (entity) has a copy in every layer, and every edge its layer in the
   attribute :data:`~graphoskop.LAYER`.
   """
   for s, t, _ in edges:
      nodes.setdefault(s, {})
      nodes.setdefault(t, {})
   types = {str(a.get(BIPARTITE)) for a in nodes.values()}
   bipartite = bipartite or (all(_present(a.get(BIPARTITE)) for a in nodes.values()) and 0 < len(types) <= 2)   # e.g. written by write_network
   whole = all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in nodes)
   key = (lambda v: int(v)) if whole else str
   N = pd.concat([pd.DataFrame({"node id": [key(v) for v in nodes]}), _table(list(nodes.values()), True)], axis = 1)
   E = pd.concat([pd.DataFrame({"source node": [key(s) for s, _, _ in edges], "target node": [key(t) for _, t, _ in edges]}), _table([a for _, _, a in edges], False)], axis = 1)
   if layers:
      E[LAYER] = [str(a[LAYER]) for _, _, a in edges]
      N = pd.concat([N.assign(**{" layer": str(l)}) for l in layers], ignore_index = True)
      N = N[["node id", " layer"] + [c for c in N.columns if c not in ("node id", " layer")]]
      return load_network(E, N, directed = bool(directed), bipartite = bipartite, device = device, multilayer = True, layers = [str(l) for l in layers])
   try:
      return load_network(E, N, directed = bool(directed), bipartite = bipartite, device = device)
   except ValueError:   # node types that do not make it bipartite: an attribute like the others
      if not bipartite:
         raise
      return load_network(E, N, directed = bool(directed), device = device)


def _number(v):
   """
   A value as a float if it is a number (or a text of one), else ``None``.
   """
   if isinstance(v, (bool, np.bool_, int, float, np.integer, np.floating)):
      return float(v)
   try:
      return float(v) if isinstance(v, str) and v.strip() else None
   except ValueError:
      return None


def _table(rows, many):
   """
   Columns of attributes, one per name in the rows (dictionaries): numeric if all the values are
   numbers; sequences of two or three numbers split into ``<name>_x``, ``_y``, ``_z``; other
   sequences as tuples (several values) if ``many``, else as texts; other values as texts. Missing
   values are NaN.
   """
   columns = {}
   for name in dict.fromkeys(k for r in rows for k in r):
      values = [r.get(name) for r in rows]
      present = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
      seq = [v for v in present if isinstance(v, (list, tuple, np.ndarray))]
      if present and len(seq) == len(present) and all(len(v) in (2, 3) and all(_number(x) is not None for x in v) for v in seq):
         for j in range(max(len(v) for v in seq)):
            columns[f"{name}_{'xyz'[j]}"] = [_number(v[j]) if v is not None and j < len(v) else np.nan for v in values]
      elif present and all(_number(v) is not None for v in present):
         columns[name] = [_number(v) if v is not None else np.nan for v in values]
      else:
         columns[name] = [np.nan if v is None else (tuple if many else ", ".join)(str(x) for x in v) if isinstance(v, (list, tuple, np.ndarray)) else str(v) for v in values]
   return pd.DataFrame(columns, index = range(len(rows)))


def _relabel(nodes, edges, name = "label"):
   """
   Nodes named by their attribute ``name`` when every node has a different one (the attribute is
   then dropped); edges follow.
   """
   labels = [a.get(name) for a in nodes.values()]
   if not labels or None in labels or len(set(map(str, labels))) < len(labels):
      return nodes, edges
   new = {v: str(a.pop(name)) for v, a in nodes.items()}
   return {new[v]: a for v, a in nodes.items()}, [(new[s], new[t], a) for s, t, a in edges]


def _hex(r, g, b, scale = 255.):
   """
   ``"#rrggbb"`` of red, green and blue from 0 to ``scale``.
   """
   return "#" + "".join(f"{int(round(min(max(float(c) / scale, 0.), 1.) * 255)):02x}" for c in (r, g, b))


def _color(text):
   """
   A color of a file: ``"#rrggbb"`` for hexadecimal colors (alpha dropped) and Graphviz HSV
   triples, else the name in lower case without spaces; ``None`` for nothing.
   """
   if text is None:
      return None
   text = str(text).strip().split(":")[0].split(";")[0]   # Graphviz lists: the first color
   if re.fullmatch(r"#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?", text):
      return text[:7].lower()
   if re.fullmatch(r"#[0-9a-fA-F]{3}", text):
      return "#" + "".join(c * 2 for c in text[1:]).lower()
   hsv = re.split(r"[,\s]+", text)
   if len(hsv) == 3 and all(_number(x) is not None for x in hsv):
      return _hex(*colorsys.hsv_to_rgb(*map(float, hsv)), scale = 1.)
   text = text.split("/")[-1].replace(" ", "").lower()   # Graphviz schemes: /scheme/name
   return text or None


def _put(attrs, name, value):
   """
   Sets an attribute when there is a value.
   """
   if value is not None and value != "":
      attrs[name] = value


# Pajek

_PAJEK_SHAPES = {"ellipse", "box", "diamond", "triangle", "cross", "empty"}


def _pajek(text, directed):
   nodes, edges, mode, arcs, two = {}, [], None, False, None
   for raw in text.splitlines():
      line = raw.strip()
      if not line or line.startswith("%"):
         continue
      if line.startswith("*"):
         head = line.split()
         mode = head[0].lower()
         if mode == "*vertices":
            for k in range(1, int(head[1]) + 1):
               nodes.setdefault(str(k), {})
            two = int(head[2]) if len(head) > 2 else None
         arcs = arcs or mode in ("*arcs", "*arcslist", "*matrix")
         row = 0
         continue
      tokens = shlex.split(line, posix = True) if '"' in line else line.split()
      if mode == "*vertices":
         a = nodes.setdefault(tokens[0], {})
         i = 1
         if i < len(tokens) and _number(tokens[i]) is None and tokens[i].lower() not in _PAJEK_SHAPES | {"ic", "bc", "bw", "lc", "x_fact", "y_fact", "s_size"}:
            a["label"] = tokens[i]
            i += 1
         xyz = []
         while i < len(tokens) and len(xyz) < 3 and _number(tokens[i]) is not None:
            xyz.append(float(tokens[i]))
            i += 1
         if len(xyz) >= 2:
            a["pos_x"], a["pos_y"] = xyz[0], -xyz[1]   # y points down in Pajek
         while i < len(tokens):
            t = tokens[i].lower()
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if t in _PAJEK_SHAPES:
               a["shape"] = t
            elif t in ("ic", "bc", "lc") and nxt is not None:
               a[{"ic": "color", "bc": "border_color", "lc": "label_color"}[t]] = _color(nxt)
               i += 1
            elif t in ("bw", "x_fact", "s_size") and _number(nxt) is not None:
               a["border_width" if t == "bw" else "size"] = float(nxt)
               i += 1
            i += 1
      elif mode in ("*arcs", "*edges"):
         a = {}
         i = 2
         if i < len(tokens) and _number(tokens[i]) is not None:
            a["weight"] = float(tokens[i])
            i += 1
         while i + 1 < len(tokens):
            t = tokens[i].lower()
            if t in ("c", "w", "p", "l"):
               a[{"c": "color", "w": "width", "p": "style", "l": "label"}[t]] = _color(tokens[i + 1]) if t == "c" else (float(tokens[i + 1]) if t == "w" and _number(tokens[i + 1]) is not None else tokens[i + 1].lower() if t == "p" else tokens[i + 1])
            i += 2
         edges.append((tokens[0], tokens[1], a, mode == "*arcs"))
      elif mode in ("*arcslist", "*edgeslist"):
         edges.extend((tokens[0], t, {}, mode == "*arcslist") for t in tokens[1:])
      elif mode == "*matrix":
         row += 1
         if two:   # rows are the first mode, columns the second
            edges.extend((str(row), str(two + j + 1), {"weight": float(w)}, False) for j, w in enumerate(tokens) if float(w) != 0)
         else:
            edges.extend((str(row), str(j + 1), {"weight": float(w)}, True) for j, w in enumerate(tokens) if float(w) != 0)
   is_directed = arcs and not two if directed is None else directed
   out = []
   for s, t, a, arc in edges:
      out.append((s, t, a))
      if is_directed and not arc and s != t:   # edges in a directed network: both directions
         out.append((t, s, dict(a)))
   if two:
      for k, a in enumerate(nodes.values()):
         a[BIPARTITE] = int(k >= two)
   nodes, out = _relabel(nodes, out)
   return nodes, out, is_directed, bool(two)


# UCINET DL

def _dl_labels(text):
   return [t.strip('"') for t in re.findall(r'"[^"]*"|[^,\s]+', text)]


def _ucinet(text, directed):
   at = re.search(r"\bdata\s*:", text, re.I)
   if at is None:
      raise ValueError("UCINET file without 'data:'")
   head, data = text[:at.start()], text[at.end():]
   lower = head.lower()
   get = lambda key: int(m.group(1)) if (m := re.search(rf"\b{key}\s*=\s*(\d+)", lower)) else None
   n, nr, nc = get("n"), get("nr"), get("nc")
   nm = get("nm") or 1
   m = re.search(r"format\s*=\s*(\w+)", lower)
   form = m.group(1) if m else "fullmatrix"
   embedded = "embedded" in lower
   labels = {}
   for m in re.finditer(r"(row labels|column labels|col labels|matrix labels|labels)\s*:(.*?)(?=\b(?:row labels|column labels|col labels|matrix labels|labels|format|n|nr|nc|nm|diagonal)\b\s*[:=]|$)", head, re.I | re.S):
      labels[m.group(1).lower()[:3].replace("lab", "all")] = _dl_labels(m.group(2))
   if nm > 1 and (form != "fullmatrix" or embedded):
      raise ValueError("UCINET files with several relations (nm > 1) are supported as full matrices without embedded labels")
   matrices = labels.get("mat") or [str(k + 1) for k in range(nm)]
   two = nr is not None and nc is not None
   free = embedded and form != "fullmatrix"   # the edge list names the nodes
   rows = labels.get("row") or labels.get("all") or ([] if free else [f"r{k + 1}" if two else str(k + 1) for k in range(nr if two else n)])
   cols = (labels.get("col") or ([] if free else [f"c{k + 1}" for k in range(nc)])) if two else rows
   lines = [l for l in data.splitlines() if l.strip()]
   W = {}
   Ws = [W]
   if form == "fullmatrix":
      if embedded:
         cols = _dl_labels(lines[0])
         if not two:
            rows = cols
         lines = lines[1:]
      tokens = [t for l in lines for t in l.split()]
      if embedded:   # every row starts with its label
         width = len(cols) + 1
         rows = [tokens[i] for i in range(0, len(tokens), width)] if two else rows
         tokens = [t for i, t in enumerate(tokens) if i % width]
      size = len(rows) * len(cols)
      Ws = [{} for _ in range(nm)]
      for k, w in enumerate(tokens[:size * nm]):
         if float(w) != 0:
            Ws[k // size][(rows[k % size // len(cols)], cols[k % len(cols)])] = float(w)
   else:
      name = lambda t, side: t if embedded else (side[int(t) - 1] if _number(t) is not None and 0 < int(float(t)) <= len(side) else t)
      for l in lines:
         t = _dl_labels(l)
         if form.startswith("edgelist"):
            W[(name(t[0], rows), name(t[1], cols))] = float(t[2]) if len(t) > 2 else 1.
         else:
            for x in t[1:]:
               W[(name(t[0], rows), name(x, cols))] = 1.
   return _relations(rows, cols, Ws, matrices, two, directed)


def _relations(rows, cols, Ws, names, two, directed):
   """
   Network of UCINET relations: ``Ws`` holds, per relation, the weights of the pairs (row label,
   column label). Two-mode data (``two``) are bipartite, rows of type 0; one-mode data are
   undirected when every relation is symmetric (unless ``directed`` says otherwise), with each pair
   once. Several relations make a multilayer network, one layer per relation (named by ``names``).
   """
   nodes = {v: {} for v in dict.fromkeys(rows + ([c for c in cols if c not in rows] if two else cols))}
   for W in Ws:
      for s, t in W:
         nodes.setdefault(s, {})
         nodes.setdefault(t, {})
   layers = list(names) if len(Ws) > 1 else None
   edges = []
   if two:
      side = {v: 0 for v in rows} | {v: 1 for v in cols}
      for W in Ws:
         for s, t in W:
            side.setdefault(s, 0)
            side.setdefault(t, 1)
      for v in nodes:
         nodes[v][BIPARTITE] = side[v]
      is_directed = False
   else:
      symmetric = all(W.get((t, s)) == w for W in Ws for (s, t), w in W.items())
      is_directed = not symmetric if directed is None else directed
   for k, W in enumerate(Ws):
      edges.extend((s, t, {"weight": w} | ({LAYER: layers[k]} if layers else {})) for (s, t), w in W.items() if two or is_directed or not symmetric or s <= t)
   return (nodes, edges, is_directed, two, layers) if layers else (nodes, edges, is_directed, two)


_UCINET_TYPES = {1: "u1", 2: "u1", 3: "i1", 4: "<u2", 5: "<i2", 6: "<i4", 7: "<f4", 9: "<f8"}


def _ucinet_binary(path, directed):
   """
   UCINET 6 binary data: the header file ``name.##h`` (version, date, data type, dimensions,
   title, labels) and the data file ``name.##d`` (the matrices, one after the other, row by row).
   """
   base = os.path.splitext(path)[0]
   with open(base + ".##h", "rb") as f:
      h = f.read()
   at = 1 + h[0]
   version = h[1:at].decode("latin-1")
   at += 8   # year, month, day, day of the week
   kind = int.from_bytes(h[at:at + 2], "little")   # labels: 1 = fixed records of 20 bytes
   code = h[at + 2]
   ndim = int.from_bytes(h[at + 3:at + 5], "little")
   at += 5
   size = 4 if version == "V6404" else 2
   dims = [int.from_bytes(h[at + size * i:at + size * (i + 1)], "little") for i in range(ndim)]
   at += size * ndim
   at += 1 + h[at]   # title
   has = list(h[at:at + ndim])
   at += ndim
   labels = []
   for d, labeled in zip(dims, has):
      names = []
      for i in range(d if labeled else 0):
         if kind == 1:
            names.append(h[at + 1:at + 1 + h[at]].decode("latin-1"))
            at += 20
         else:
            length = int.from_bytes(h[at:at + 2], "little")
            names.append(h[at + 2:at + 2 + length].decode("latin-1"))
            at += 2 + length
      labels.append(names or [str(i + 1) for i in range(d)])
   if code not in _UCINET_TYPES:
      raise ValueError(f"UCINET data type {code} is not supported")
   r, c = dims[0], dims[1]
   k = dims[2] if ndim > 2 else 1
   X = np.fromfile(base + ".##d", dtype = _UCINET_TYPES[code], count = r * c * k).reshape(k, r, c).astype(float)
   two = labels[0] != labels[1]
   rows, cols = labels[0], labels[1]
   if two and set(rows) & set(cols):   # the same labels on both sides: told apart
      rows, cols = [f"{v} (row)" for v in rows], [f"{v} (column)" for v in cols]
   Ws = [{(rows[i], cols[j]): float(M[i, j]) for i, j in zip(*np.nonzero(M))} for M in X]
   return _relations(rows, cols, Ws, labels[2] if ndim > 2 else ["1"], two, directed)


# adjacency lists

def _adjlist(text, directed):
   nodes, edges, seen = {}, [], set()
   for line in text.splitlines():
      tokens = line.split("#")[0].split()
      if not tokens:
         continue
      nodes.setdefault(tokens[0], {})
      for t in tokens[1:]:
         pair = (tokens[0], t) if directed else frozenset((tokens[0], t))
         if directed or pair not in seen:
            seen.add(pair)
            edges.append((tokens[0], t, {}))
   return nodes, edges, bool(directed), False


# DOT

_DOT_TOKEN = re.compile(r'''\s+|//[^\n]*|/\*.*?\*/|^[ \t]*\#[^\n]*|"(?:\\.|[^"\\])*"|->|--|[{}\[\];=,:+]|[A-Za-z_\x80-\U0010ffff][\w\x80-\U0010ffff]*|-?(?:\.\d+|\d+(?:\.\d*)?)''', re.S | re.M)


def _dot_tokens(text):
   out, i = [], 0
   while i < len(text):
      if text[i] == "<":   # HTML string: up to its matching >
         depth, j = 0, i
         while j < len(text):
            depth += {"<": 1, ">": -1}.get(text[j], 0)
            j += 1
            if depth == 0:
               break
         out.append(("id", text[i + 1:j - 1]))
         i = j
         continue
      m = _DOT_TOKEN.match(text, i)
      if m is None:
         raise ValueError(f"DOT: unexpected {text[i:i + 20]!r}")
      t = m.group()
      i = m.end()
      if not t.strip() or t.startswith(("//", "/*")) or t.lstrip().startswith("#"):
         continue
      if t.startswith('"'):
         s = t[1:-1].replace('\\"', '"').replace("\\\n", "")
         if len(out) > 1 and out[-1] == ("op", "+") and out[-2][0] == "id":   # "a" + "b"
            out.pop()
            out[-1] = ("id", out[-1][1] + s)
         else:
            out.append(("id", s))
      elif t in ("{", "}", "[", "]", ";", "=", ",", ":", "->", "--", "+"):
         out.append(("op", t))
      else:
         out.append(("id", t))
   return out


def _dot(text, directed):
   tokens = _dot_tokens(text)
   pos = [0]
   peek = lambda k = 0: tokens[pos[0] + k] if pos[0] + k < len(tokens) else ("end", None)
   def take():
      pos[0] += 1
      return tokens[pos[0] - 1]
   word = lambda t: t[1].lower() if t[0] == "id" else None
   nodes, edges = {}, []
   strict = word(peek()) == "strict"
   if strict:
      take()
   kind = word(take())
   if kind not in ("graph", "digraph"):
      raise ValueError("DOT: the file does not start with graph or digraph")
   if peek()[0] == "id":
      take()
   def attrs():
      out = {}
      while peek() == ("op", "["):
         take()
         while peek() != ("op", "]"):
            k = take()[1]
            v = "true"
            if peek() == ("op", "="):
               take()
               v = take()[1]
            out[k] = v
            if peek()[1] in (",", ";"):
               take()
         take()
      return out
   def node(name, defaults):
      if name not in nodes:
         nodes[name] = dict(defaults)
      return name
   def block(nd, ed):
      take()   # {
      members = []
      while peek() != ("op", "}") and peek()[0] != "end":
         t = peek()
         if word(t) in ("graph", "node", "edge") and peek(1) == ("op", "["):
            take()
            a = attrs()
            (nd if word(t) == "node" else ed if word(t) == "edge" else {}).update(a)
         elif t[0] == "id" and peek(1) == ("op", "="):
            take(); take(); take()
         else:
            ends = [operand(nd, ed)]
            while peek() in (("op", "->"), ("op", "--")):
               take()
               ends.append(operand(nd, ed))
            a = attrs()
            if len(ends) == 1:
               for v in ends[0]:
                  nodes[v].update(a)
            for x, y in zip(ends, ends[1:]):
               edges.extend((s, t2, dict(ed, **a)) for s in x for t2 in y)
            for e in ends:
               members.extend(e)
         if peek() in (("op", ";"), ("op", ",")):
            take()
      take()   # }
      return members
   def operand(nd, ed):
      if word(peek()) == "subgraph" or peek() == ("op", "{"):
         if word(peek()) == "subgraph":
            take()
            if peek()[0] == "id":
               take()
         return block(dict(nd), dict(ed))
      name = take()[1]
      while peek() == ("op", ":"):   # ports
         take(); take()
      return [node(name, nd)]
   block({}, {})
   if strict:   # no parallel edges: the first one, with the attributes of all
      unique = {}
      for s, t, a in edges:
         unique.setdefault((s, t) if kind == "digraph" else frozenset((s, t)), (s, t, {}))[2].update(a)
      edges = list(unique.values())
   return {v: _dot_visual(a, True) for v, a in nodes.items()}, [(s, t, _dot_visual(a, False)) for s, t, a in edges], kind == "digraph", False


def _dot_visual(a, is_node):
   """
   Attributes of a DOT node or edge, the visual ones renamed (:data:`VISUAL_ATTRIBUTES`).
   """
   a = dict(a)
   out = {}
   if is_node:
      if "pos" in a:
         xy = a.pop("pos").rstrip("!").split(",")
         if len(xy) >= 2 and all(_number(x) is not None for x in xy[:2]):
            out["pos_x"], out["pos_y"] = float(xy[0]), float(xy[1])
      fill, line = a.pop("fillcolor", None), a.pop("color", None)
      _put(out, "color", _color(fill or line))
      if fill:
         _put(out, "border_color", _color(line))
      if _number(a.get("width")) is not None:
         out["size"] = 72 * float(a.pop("width"))   # inches to points, the unit of the positions
      _put(out, "shape", a.pop("shape", "").lower())
      _put(out, "border_width", _number(a.pop("penwidth", None)))
      _put(out, "label_color", _color(a.pop("fontcolor", None)))
   else:
      _put(out, "color", _color(a.pop("color", None)))
      _put(out, "width", _number(a.pop("penwidth", None)))
      _put(out, "style", a.pop("style", "").lower())
   label = a.pop("label", None)
   if label not in (None, "\\N", "\\E"):
      out["label"] = label
   out.update(a)
   return out


# GML

def _gml_parse(tokens, i):
   out = []
   while i < len(tokens) and tokens[i] != "]":
      key, value = tokens[i], tokens[i + 1]
      i += 2
      if value == "[":
         value, i = _gml_parse(tokens, i)
         i += 1
      elif value.startswith('"'):
         value = value[1:-1]
      else:
         value = float(value) if _number(value) is not None else value
      out.append((key, value))
   return out, i


def _gml(text, directed):
   tokens = re.findall(r'"[^"]*"|\[|\]|[^\s\[\]"]+', "\n".join(l for l in text.splitlines() if not l.strip().startswith("#")))
   top, _ = _gml_parse(tokens, 0)
   graph = next((v for k, v in top if k == "graph"), None)
   if graph is None:
      raise ValueError("GML file without graph")
   nodes, edges = {}, []
   for k, v in graph:
      if k not in ("node", "edge"):
         continue
      a, out = {}, {}
      for key, value in v:
         if key == "graphics":
            g = dict(value)
            if k == "node":
               if "x" in g and "y" in g:
                  out["pos_x"], out["pos_y"] = g["x"], -g["y"]   # y points down
               _put(out, "size", g.get("w"))
               _put(out, "shape", str(g.get("type", "")).lower())
               _put(out, "color", _color(g.get("fill")))
               _put(out, "border_color", _color(g.get("outline")))
               _put(out, "border_width", g.get("outline_width", g.get("width")))
            else:
               _put(out, "color", _color(g.get("fill")))
               _put(out, "width", g.get("width"))
               _put(out, "style", str(g.get("style", "")).lower())
         elif not isinstance(value, list):
            a[key] = value
      a.update(out)
      if k == "node":
         nodes[_gml_id(a.pop("id"))] = a
      else:
         edges.append((_gml_id(a.pop("source")), _gml_id(a.pop("target")), a))
   nodes, edges = _relabel(nodes, edges)
   return nodes, edges, dict(graph).get("directed", 0) == 1, False


def _gml_id(v):
   return str(int(v)) if isinstance(v, float) and v == int(v) else str(v)


# GraphML and GEXF

def _strip(root):
   """
   The XML tree without namespaces in the names of elements and attributes.
   """
   for e in root.iter():
      e.tag = e.tag.split("}")[-1] if isinstance(e.tag, str) else e.tag
      for k in list(e.attrib):
         if "}" in k:
            e.attrib[k.split("}")[-1]] = e.attrib.pop(k)
   return root


def _typed(value, kind):
   if kind in ("int", "long", "float", "double", "integer"):
      return _number(value)
   if kind == "boolean":
      return value.strip().lower() in ("true", "1")
   return value


def _graphml(text, directed):
   root = _strip(ET.fromstring(text.encode("utf-8")))
   keys = {k.get("id"): (k.get("attr.name", k.get("id")), k.get("attr.type", "string"), k.get("for", "all"), k.find("default")) for k in root.iter("key")}
   graph = root.find("graph")
   if graph is None:
      raise ValueError("GraphML file without graph")
   def data(element, kind):
      a = {name: _typed(d.text, t) for i, (name, t, f, d) in keys.items() if f in (kind, "all") and d is not None and d.text is not None}
      for d in element.findall("data"):
         name, t, _, _ = keys.get(d.get("key"), (d.get("key"), "string", kind, None))
         if len(d):
            a.update(_yed(d, kind))
         elif d.text is not None:
            a[name] = _typed(d.text, t)
      if kind == "node" and all(_number(a.get(c)) is not None for c in "rgb"):   # Gephi
         a["color"] = _hex(a.pop("r"), a.pop("g"), a.pop("b"))
      if kind == "node" and _number(a.get("x")) is not None and _number(a.get("y")) is not None:
         a["pos_x"], a["pos_y"] = _number(a.pop("x")), _number(a.pop("y"))
      return a
   nodes = {n.get("id"): data(n, "node") for n in graph.iter("node")}
   edges = [(e.get("source"), e.get("target"), data(e, "edge")) for e in graph.iter("edge")]
   return nodes, edges, graph.get("edgedefault", "undirected") == "directed", False


def _yed(d, kind):
   """
   Visual attributes of the yEd graphics of a GraphML node or edge.
   """
   out = {}
   find = lambda tag: next(d.iter(tag), None)
   if kind == "node":
      g, fill, border, shape, label = find("Geometry"), find("Fill"), find("BorderStyle"), find("Shape"), find("NodeLabel")
      if g is not None:
         x, y, w, h = (float(g.get(k, 0)) for k in ("x", "y", "width", "height"))
         out.update(pos_x = x + w / 2, pos_y = -(y + h / 2), size = w)   # the corner, y pointing down
      if fill is not None:
         _put(out, "color", _color(fill.get("color")))
      if border is not None:
         _put(out, "border_color", _color(border.get("color")))
         _put(out, "border_width", _number(border.get("width")))
      if shape is not None:
         _put(out, "shape", shape.get("type", "").lower())
      if label is not None:
         _put(out, "label", (label.text or "").strip())
         _put(out, "label_color", _color(label.get("textColor")))
   else:
      line = find("LineStyle")
      if line is not None:
         _put(out, "color", _color(line.get("color")))
         _put(out, "width", _number(line.get("width")))
         _put(out, "style", line.get("type", "").lower())
   return out


def _gexf(text, directed):
   root = _strip(ET.fromstring(text.encode("utf-8")))
   graph = root.find("graph")
   if graph is None:
      raise ValueError("GEXF file without graph")
   keys = {}
   for block in graph.findall("attributes"):
      for a in block.findall("attribute"):
         d = a.find("default")
         keys[(block.get("class"), a.get("id"))] = (a.get("title", a.get("id")), a.get("type", "string"), d.text if d is not None else None)
   def data(element, kind):
      a = {name: _typed(d, t) for (c, _), (name, t, d) in keys.items() if c == kind and d is not None}
      for v in element.iter("attvalue"):
         name, t, _ = keys.get((kind, v.get("for", v.get("id"))), (v.get("for"), "string", None))
         a[name] = _typed(v.get("value"), t)
      color, size, shape, position, thickness = (element.find(k) for k in ("color", "size", "shape", "position", "thickness"))
      if color is not None:
         a["color"] = _color(color.get("hex")) if color.get("hex") else _hex(color.get("r", 0), color.get("g", 0), color.get("b", 0))
         _put(a, "opacity", _number(color.get("a")))
      if position is not None:
         a["pos_x"], a["pos_y"] = float(position.get("x", 0)), float(position.get("y", 0))
      if size is not None:
         a["size" if kind == "node" else "width"] = _number(size.get("value"))
      if thickness is not None:
         a["width"] = _number(thickness.get("value"))
      if shape is not None:
         a["shape" if kind == "node" else "style"] = shape.get("value", "").lower()
      _put(a, "label", element.get("label"))
      return a
   nodes = {n.get("id"): data(n, "node") for n in graph.iter("node")}
   edges = []
   kinds = set()
   for e in graph.iter("edge"):
      a = data(e, "edge")
      _put(a, "weight", _number(e.get("weight")))
      kinds.add(e.get("type", graph.get("defaultedgetype", "undirected")))
      edges.append((e.get("source"), e.get("target"), a))
   nodes, edges = _relabel(nodes, edges)
   return nodes, edges, "directed" in kinds, False


# Cytoscape

def _cytoscape(text, directed):
   doc = json.loads(text)
   elements = doc.get("elements", doc) if isinstance(doc, dict) else doc
   if isinstance(elements, dict):
      items = [("nodes", e) for e in elements.get("nodes", [])] + [("edges", e) for e in elements.get("edges", [])]
   else:
      items = [(e.get("group", "edges" if "source" in e.get("data", {}) else "nodes"), e) for e in elements]
   nodes, edges = {}, []
   for group, e in items:
      a = dict(e.get("data", {}))
      if group == "nodes":
         p = e.get("position")
         if p and "x" in p and "y" in p:
            a["pos_x"], a["pos_y"] = float(p["x"]), -float(p["y"])   # y points down
         nodes[str(a.pop("id"))] = a
      else:
         a.pop("id", None)
         edges.append((str(a.pop("source")), str(a.pop("target")), a))
   nodes, edges = _relabel(nodes, edges, "name")
   return nodes, edges, bool(doc.get("directed", False)) if isinstance(doc, dict) else False, False


_READERS = {"pajek": _pajek, "ucinet": _ucinet, "ucinet_binary": _ucinet_binary, "adjlist": _adjlist, "dot": _dot, "gml": _gml, "graphml": _graphml, "gexf": _gexf, "cytoscape": _cytoscape}


# writing

def write_network(net, path, format = None, node_visuals = None, edge_visuals = None):
   """
   Save a network in a file of another program, in one of the formats of :func:`read_network`.
   Node and edge attributes are written as the format allows (Pajek: the visual attributes and the
   edge ``weight`` only; UCINET: the edge ``weight`` only; adjacency lists: no attributes), and the
   attributes named as :data:`VISUAL_ATTRIBUTES` (also those given in ``node_visuals`` and
   ``edge_visuals``) as the visual attributes of the format: ``pos_x`` and ``pos_y`` (Pajek:
   scaled to the unit square; y pointing down where the format has it so), ``color``
   (``"#rrggbb"`` or a color name), ``opacity``, ``size``, ``shape`` (the names of the format,
   e.g. ``"box"``, or ``circle``, ``square``, ``triangle``, ``diamond``, ``pentagon``,
   ``hexagon``, ``heptagon``, ``octagon``, ``star``), ``border_color``, ``border_width``,
   ``label``, ``label_color``; edges: ``color``, ``width``, ``style`` (e.g. ``"dashed"``, or
   ``dash``, ``dots``), ``label``. A GEXF node is labeled by its ``label``, else by its id. They
   load back with :func:`read_network` (UCINET: undirected networks as symmetric edge lists;
   GraphML: positions, colors and sizes as Gephi writes them, ``x``, ``y``, ``r``, ``g``, ``b``,
   ``size``; node ids with white space do not survive adjacency lists).

   :param net: A :class:`~graphoskop.Network` (multilayer ones with ``(entity, layer)`` ids
      written as texts; not hypergraphs nor simplicial complexes).
   :param path: Path of the file.
   :param format: A format of :func:`read_network`, or ``None`` for the one of the extension
      (:data:`FORMATS`).
   :param node_visuals: Dictionary from a name of :data:`VISUAL_ATTRIBUTES` to its values, one per
      node in the network wide order (missing values ``None`` or NaN), written instead of the node
      attribute of that name; or ``None``.
   :param edge_visuals: The same for the edges, one value per edge in the order of
      :meth:`~graphoskop.Network.to_pandas_dataframes`; or ``None``.
   :raises ValueError: If the format is unknown, or the network is a hypergraph or a simplicial
      complex.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.read_network("karate.gml")
      gsk.force_directed_layout(net)
      gsk.write_network(net, "karate.gexf", node_visuals = {"color": ["#e41a1c"] * net.number_of_nodes()})
   """
   format = format or FORMATS.get(os.path.splitext(path)[1].lower())
   if format not in _WRITERS:
      raise ValueError(f"unknown network format of {path!r}: one of {sorted(_WRITERS)}")
   N, E = _frames(net)
   for table, extra in ((N, node_visuals), (E, edge_visuals)):
      for k, v in (extra or {}).items():
         table[k] = [np.nan if x is None else x for x in v]
   text = _WRITERS[format](N, E, net.directed, net.bipartite_labels is not None)
   with open(path, "w", encoding = "utf-8") as f:
      f.write(text)


def _frames(net):
   """
   Node table (``node``: ids as texts, then the attributes, in the network wide order) and edge
   table (``source``, ``target`` as texts, then the attributes) of a network.
   """
   from .network import _wide_rows
   if net.higher_order is not None:
      raise ValueError("hypergraphs and simplicial complexes cannot be written in these formats")
   ids = [str(v) for c in net.node_ids for v in c]
   N = _wide_rows(net, np.arange(len(ids))).drop(columns = ["node", "layer"], errors = "ignore")
   N.insert(0, "node", ids)
   tables = net.to_pandas_dataframes()
   if net.layer_labels is not None:
      edges, inter = tables[0], tables[1]
      edges = edges.assign(source = [str((s, l)) for s, l in zip(edges["source"], edges["layer"])], target = [str((t, l)) for t, l in zip(edges["target"], edges["layer"])]).drop(columns = "layer")
      inter = inter.assign(source = [str((s, l)) for s, l in zip(inter["source"], inter["source_layer"])], target = [str((t, l)) for t, l in zip(inter["target"], inter["target_layer"])]).drop(columns = ["source_layer", "target_layer"])
      E = pd.concat([edges, inter], ignore_index = True)
   else:
      E = (tables[0] if isinstance(tables, tuple) else tables).copy()
      E["source"], E["target"] = E["source"].map(str), E["target"].map(str)
   return N.reset_index(drop = True), E.reset_index(drop = True)


def _present(v):
   return v is not None and not (isinstance(v, float) and np.isnan(v)) and v is not pd.NA


def _rows(table, skip = ()):
   """
   Rows of a table as dictionaries of their present values (``skip``: columns left out).
   """
   names = [c for c in table.columns if c not in skip]
   return [{k: (v.item() if isinstance(v, np.generic) else v) for k, v in zip(names, row) if _present(v)} for row in table[names].itertuples(index = False, name = None)]


def _quote(text):
   return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


_GUI_SHAPES = {"pajek": {"circle": "ellipse", "square": "box", "diamond": "diamond", "triangle": "triangle"}, "gexf": {"circle": "disc", "ellipse": "disc", "square": "square", "box": "square", "triangle": "triangle", "diamond": "diamond"}, "dot": {"heptagon": "septagon"}}
_GUI_STYLES = {"dash": "dashed", "dots": "dotted", "dash-dot": "dashdot", "parallel": "double"}


def _rgb(color):
   """
   Red, green and blue (0 to 255) of a ``#rrggbb`` color, or ``None``.
   """
   c = _color(color)
   return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) if c and c.startswith("#") else None


def _write_pajek(N, E, directed, bipartite):
   order = list(N.index)
   two = None
   if bipartite:   # nodes of the first type first
      first = [i for i in order if N.at[i, BIPARTITE] in (0, 0., "0") or str(N.at[i, BIPARTITE]) == str(sorted(N[BIPARTITE].dropna().map(str).unique())[0])]
      two = len(first)
      order = first + [i for i in order if i not in set(first)]
   number = {N.at[i, "node"]: k + 1 for k, i in enumerate(order)}
   lines = [f"*Vertices {len(order)}" + (f" {two}" if two else "")]
   xy = None
   if {"pos_x", "pos_y"} <= set(N.columns) and N[["pos_x", "pos_y"]].notna().all().all():
      x, y = N["pos_x"].astype(float), N["pos_y"].astype(float)
      span = max(x.max() - x.min(), y.max() - y.min()) or 1.
      xy = ((x - x.min()) / span, 1 - (y - y.min()) / span)   # the unit square, y pointing down
   size = N["size"].astype(float).median() if "size" in N and N["size"].notna().any() and _number(N["size"].dropna().iloc[0]) is not None else None
   for k, i in enumerate(order):
      a = _rows(N.loc[[i]])[0]
      line = [str(k + 1), _quote(a["node"])]
      if xy is not None:
         line += [f"{xy[0][i]:.6f}", f"{xy[1][i]:.6f}", "0.5"]
      if a.get("shape") is not None:
         line.append(_GUI_SHAPES["pajek"].get(str(a["shape"]), str(a["shape"])))
      for key, name in (("color", "ic"), ("border_color", "bc"), ("label_color", "lc")):
         if key in a:
            line += [name, str(a[key])]
      if _number(a.get("border_width")) is not None:
         line += ["bw", str(a["border_width"])]
      if size and _number(a.get("size")) is not None:
         line += ["x_fact", f"{float(a['size']) / size:.6g}"]
      lines.append(" ".join(line))
   lines.append("*Arcs" if directed else "*Edges")
   for a in _rows(E):
      line = [str(number[a["source"]]), str(number[a["target"]]), str(_number(a.get("weight")) if _number(a.get("weight")) is not None else 1)]
      for key, name in (("color", "c"), ("width", "w"), ("style", "p"), ("label", "l")):
         if key in a:
            line += [name, _quote(a[key]) if key == "label" else _GUI_STYLES.get(str(a[key]), str(a[key])) if key == "style" else str(a[key])]
      lines.append(" ".join(line))
   return "\n".join(lines) + "\n"


def _dl_label(text):
   text = str(text)
   return _quote(text) if re.search(r'[\s,"]', text) else text


def _write_ucinet(N, E, directed, bipartite):
   weight = "weight" in E and E["weight"].map(lambda v: _number(v) is not None or not _present(v)).all()
   pairs = [(s, t, float(w) if weight and _present(w) else 1.) for s, t, w in zip(E["source"], E["target"], E["weight"] if weight else [1.] * len(E))]
   if bipartite:
      kind = dict(zip(N["node"], N[BIPARTITE].map(str)))
      first = sorted(set(kind.values()))[0]
      pairs = [(s, t, w) if kind[s] == first else (t, s, w) for s, t, w in pairs]
      rows, cols = [v for v in N["node"] if kind[v] == first], [v for v in N["node"] if kind[v] != first]
      head = f"dl nr={len(rows)}, nc={len(cols)} format=edgelist2\nlabels embedded\nrow labels:\n{', '.join(map(_dl_label, rows))}\ncolumn labels:\n{', '.join(map(_dl_label, cols))}\n"
   else:
      if not directed:   # symmetric: read back as undirected
         pairs += [(t, s, w) for s, t, w in pairs if s != t]
      head = f"dl n={len(N)} format=edgelist1\nlabels embedded\nlabels:\n{', '.join(map(_dl_label, N['node']))}\n"
   return head + "data:\n" + "".join(f"{_dl_label(s)} {_dl_label(t)} {w:g}\n" for s, t, w in pairs)


def _write_adjlist(N, E, directed, bipartite):
   out = {v: [] for v in N["node"]}
   for s, t in zip(E["source"], E["target"]):
      out[s].append(t)
   return "".join(" ".join([v] + out[v]) + "\n" for v in out)


def _write_dot(N, E, directed, bipartite):
   def attrs(a, node):
      out = {}
      if node:
         if "pos_x" in a and "pos_y" in a:
            out["pos"] = f"{a.pop('pos_x'):g},{a.pop('pos_y'):g}!"
         if "color" in a:
            color, opacity = str(a.pop("color")), _number(a.get("opacity"))
            out["fillcolor"] = color + (f"{int(round(255 * min(max(opacity, 0.), 1.))):02x}" if opacity is not None and _rgb(color) and color.startswith("#") else "")
            out["style"] = "filled"
         for key, name in (("border_color", "color"), ("border_width", "penwidth"), ("label_color", "fontcolor")):
            if key in a:
               out[name] = a.pop(key)
         if _number(a.get("size")) is not None:
            out["width"] = f"{float(a.pop('size')) / 72:g}"
         if "shape" in a:
            shape = str(a.pop("shape"))
            out["shape"] = _GUI_SHAPES["dot"].get(shape, shape)
      else:
         for key, name in (("color", "color"), ("width", "penwidth")):
            if key in a:
               out[name] = a.pop(key)
         if "style" in a:
            style = str(a.pop("style"))
            out["style"] = _GUI_STYLES.get(style, style)
      a.pop("opacity", None)
      out.update(a)
      return f" [{', '.join(f'{_quote(k)}={_quote(v)}' for k, v in out.items())}]" if out else ""
   arrow = " -> " if directed else " -- "
   lines = [("digraph" if directed else "graph") + " G {"]
   for a in _rows(N):
      lines.append(f"   {_quote(a.pop('node'))}{attrs(a, True)};")
   for a in _rows(E):
      lines.append(f"   {_quote(a.pop('source'))}{arrow}{_quote(a.pop('target'))}{attrs(a, False)};")
   return "\n".join(lines) + "\n}\n"


def _gml_key(name):
   key = re.sub(r"[^A-Za-z0-9]", "_", str(name))
   return key if key[:1].isalpha() else "a" + key


def _gml_value(v):
   return f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else '"' + str(v).replace('"', "&quot;") + '"'


def _write_gml(N, E, directed, bipartite):
   lines = ["graph [", f"  directed {int(directed)}"]
   number = {v: k for k, v in enumerate(N["node"])}
   for k, a in enumerate(_rows(N)):
      lines += ["  node [", f"    id {k}", f"    label {_gml_value(a.pop('node'))}"]
      g = {}
      if "pos_x" in a and "pos_y" in a:
         g.update(x = a["pos_x"], y = -a["pos_y"])   # y pointing down
      if _number(a.get("size")) is not None:
         g.update(w = a["size"], h = a["size"])
      for key, name in (("shape", "type"), ("color", "fill"), ("border_color", "outline"), ("border_width", "outline_width")):
         if key in a:
            g[name] = a[key]
      lines += [f"    {_gml_key(n)} {_gml_value(v)}" for n, v in a.items()]
      if g:
         lines += ["    graphics [", *[f"      {n} {_gml_value(v)}" for n, v in g.items()], "    ]"]
      lines.append("  ]")
   for a in _rows(E):
      lines += ["  edge [", f"    source {number[a.pop('source')]}", f"    target {number[a.pop('target')]}"]
      g = {name: a[key] for key, name in (("color", "fill"), ("width", "width"), ("style", "style")) if key in a}
      lines += [f"    {_gml_key(n)} {_gml_value(v)}" for n, v in a.items()]
      if g:
         lines += ["    graphics [", *[f"      {n} {_gml_value(v)}" for n, v in g.items()], "    ]"]
      lines.append("  ]")
   return "\n".join(lines) + "\n]\n"


def _xml(text):
   return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _kind(table, name):
   values = table[name].dropna()
   return "double" if len(values) and all(_number(v) is not None and not isinstance(v, str) for v in values) else "string"


def _write_graphml(N, E, directed, bipartite):
   N = N.copy()
   if "color" in N:   # as Gephi writes them
      rgb = N["color"].map(lambda c: _rgb(c) if _present(c) else None)
      if rgb.notna().any():
         for j, c in enumerate("rgb"):
            N[c] = rgb.map(lambda v: float(v[j]) if v else np.nan)
         N = N.drop(columns = "color")
   N = N.rename(columns = {"pos_x": "x", "pos_y": "y"})
   keys, lines = {}, ['<?xml version="1.0" encoding="UTF-8"?>', '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">']
   for kind, table, skip in (("node", N, ("node",)), ("edge", E, ("source", "target"))):
      for name in table.columns:
         if name not in skip:
            keys[(kind, name)] = f"{kind[0]}{len(keys)}"
            lines.append(f'  <key id="{keys[(kind, name)]}" for="{kind}" attr.name="{_xml(name)}" attr.type="{_kind(table, name)}"/>')
   lines.append(f'  <graph edgedefault="{"directed" if directed else "undirected"}">')
   def data(kind, a):
      return "".join(f'<data key="{keys[(kind, k)]}">{_xml(v)}</data>' for k, v in a.items())
   for a in _rows(N):
      lines.append(f'    <node id="{_xml(a.pop("node"))}">{data("node", a)}</node>')
   for a in _rows(E):
      lines.append(f'    <edge source="{_xml(a.pop("source"))}" target="{_xml(a.pop("target"))}">{data("edge", a)}</edge>')
   return "\n".join(lines + ["  </graph>", "</graphml>"]) + "\n"


def _write_gexf(N, E, directed, bipartite):
   visual = ({"pos_x", "pos_y", "opacity", "size", "shape", "label"}, {"opacity", "width", "style", "weight"})   # colors also as attributes: names have no viz:color
   lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<gexf xmlns="http://gexf.net/1.3" xmlns:viz="http://gexf.net/1.3/viz" version="1.3">', f'  <graph defaultedgetype="{"directed" if directed else "undirected"}">']
   ids = {}
   for kind, table, skip, seen in (("node", N, {"node"}, visual[0]), ("edge", E, {"source", "target"}, visual[1])):
      names = [c for c in table.columns if c not in skip | seen]
      if names:
         lines.append(f'    <attributes class="{kind}">')
         for name in names:
            ids[(kind, name)] = str(len(ids))
            lines.append(f'      <attribute id="{ids[(kind, name)]}" title="{_xml(name)}" type="{_kind(table, name)}"/>')
         lines.append("    </attributes>")
   def body(kind, a):
      out = []
      values = [(ids[(kind, k)], v) for k, v in a.items() if (kind, k) in ids]
      if values:
         out.append("<attvalues>" + "".join(f'<attvalue for="{i}" value="{_xml(v)}"/>' for i, v in values) + "</attvalues>")
      rgb = _rgb(a["color"]) if "color" in a else None
      if rgb:
         out.append(f'<viz:color r="{rgb[0]}" g="{rgb[1]}" b="{rgb[2]}"' + (f' a="{float(a["opacity"]):g}"' if _number(a.get("opacity")) is not None else "") + "/>")
      if kind == "node":
         if "pos_x" in a and "pos_y" in a:
            out.append(f'<viz:position x="{a["pos_x"]:g}" y="{a["pos_y"]:g}" z="0"/>')
         if _number(a.get("size")) is not None:
            out.append(f'<viz:size value="{float(a["size"]):g}"/>')
         if str(a.get("shape")) in _GUI_SHAPES["gexf"]:
            out.append(f'<viz:shape value="{_GUI_SHAPES["gexf"][str(a["shape"])]}"/>')
      else:
         if _number(a.get("width")) is not None:
            out.append(f'<viz:thickness value="{float(a["width"]):g}"/>')
         style = _GUI_STYLES.get(str(a.get("style")), str(a.get("style")))
         if style in ("solid", "dotted", "dashed", "double"):
            out.append(f'<viz:shape value="{style}"/>')
      return "".join(out)
   lines.append("    <nodes>")
   for a in _rows(N):
      v = a.pop("node")
      lines.append(f'      <node id="{_xml(v)}" label="{_xml(a.get("label", v))}">{body("node", a)}</node>')
   lines += ["    </nodes>", "    <edges>"]
   for k, a in enumerate(_rows(E)):
      weight = f' weight="{float(a["weight"]):g}"' if _number(a.get("weight")) is not None and not isinstance(a.get("weight"), str) else ""
      lines.append(f'      <edge id="{k}" source="{_xml(a.pop("source"))}" target="{_xml(a.pop("target"))}"{weight}>{body("edge", a)}</edge>')
   return "\n".join(lines + ["    </edges>", "  </graph>", "</gexf>"]) + "\n"


def _write_cytoscape(N, E, directed, bipartite):
   nodes, edges = [], []
   for a in _rows(N):
      v = a.pop("node")
      e = {"data": {"id": v, "name": v, **{k: x for k, x in a.items() if k not in ("pos_x", "pos_y")}}}
      if "pos_x" in a and "pos_y" in a:
         e["position"] = {"x": a["pos_x"], "y": -a["pos_y"]}   # y pointing down
      nodes.append(e)
   for k, a in enumerate(_rows(E)):
      edges.append({"data": {"id": f"e{k}", "source": a.pop("source"), "target": a.pop("target"), **a}})
   return json.dumps({"data": {}, "directed": bool(directed), "elements": {"nodes": nodes, "edges": edges}}, default = str) + "\n"


_WRITERS = {"pajek": _write_pajek, "ucinet": _write_ucinet, "adjlist": _write_adjlist, "dot": _write_dot, "gml": _write_gml, "graphml": _write_graphml, "gexf": _write_gexf, "cytoscape": _write_cytoscape}


# conversions to other libraries

def _lists(net):
   """
   Node ids (wide order), node attributes (name to values, ``None`` for missing), edges as pairs
   of positions, edge attributes, of a network (not higher-order).
   """
   if net.higher_order is not None:
      raise ValueError("hypergraphs and simplicial complexes cannot be converted")
   from .network import _wide_rows
   ids = [v for c in net.node_ids for v in c]
   N = _wide_rows(net, np.arange(len(ids))).drop(columns = ["node", "layer"], errors = "ignore")
   tables = net.to_pandas_dataframes()
   E = tables[0] if isinstance(tables, tuple) else tables
   position = {v: k for k, v in enumerate(ids)}
   if net.layer_labels is not None:
      inter = tables[1]
      pairs = [(position[(s, l)], position[(t, l)]) for s, t, l in zip(E["source"], E["target"], E["layer"])] + [(position[(s, a)], position[(t, b)]) for s, a, t, b in zip(inter["source"], inter["source_layer"], inter["target"], inter["target_layer"])]
      E = pd.concat([E.drop(columns = "layer"), inter.drop(columns = ["source_layer", "target_layer"])], ignore_index = True)
   else:
      pairs = [(position[s], position[t]) for s, t in zip(E["source"], E["target"])]
   clean = lambda col: [(v.item() if isinstance(v, np.generic) else v) if _present(v) else None for v in col]
   return ids, {k: clean(N[k]) for k in N.columns}, pairs, {k: clean(E[k]) for k in E.columns if k not in ("source", "target")}


def to_networkx(net):
   """
   Convert a network into a networkx graph: a ``Graph`` or ``DiGraph``, or a ``MultiGraph`` or
   ``MultiDiGraph`` when there are parallel edges, with the node ids (``(entity, layer)`` in
   multilayer networks) and the node and edge attributes (missing values left out).

   :param net: A :class:`~graphoskop.Network` (not a hypergraph nor a simplicial complex).
   :return: The networkx graph.
   """
   import networkx as nx
   ids, nodes, pairs, edges = _lists(net)
   keys = [(s, t) if net.directed else (min(s, t), max(s, t)) for s, t in pairs]
   multi = len(set(keys)) < len(keys)
   graph = (nx.MultiDiGraph if multi else nx.DiGraph)() if net.directed else (nx.MultiGraph if multi else nx.Graph)()
   graph.add_nodes_from((v, {k: c[i] for k, c in nodes.items() if c[i] is not None}) for i, v in enumerate(ids))
   graph.add_edges_from((ids[s], ids[t], {k: c[j] for k, c in edges.items() if c[j] is not None}) for j, (s, t) in enumerate(pairs))
   return graph


def to_igraph(net):
   """
   Convert a network into an igraph graph, with the node ids as the vertex attribute ``name``
   (texts) and the node and edge attributes (``None`` for missing values).

   :param net: A :class:`~graphoskop.Network` (not a hypergraph nor a simplicial complex).
   :return: The ``igraph.Graph``.
   """
   import igraph as ig
   ids, nodes, pairs, edges = _lists(net)
   graph = ig.Graph(n = len(ids), edges = pairs, directed = net.directed)
   graph.vs["name"] = [str(v) for v in ids]
   for k, c in nodes.items():
      graph.vs[k] = c
   for k, c in edges.items():
      graph.es[k] = c
   return graph


def to_graph_tool(net):
   """
   Convert a network into a graph-tool graph, with the node ids as the vertex property ``name``
   (texts) and the node and edge attributes as property maps: numeric ones as ``double`` (NaN for
   missing values), the others as ``string`` (empty for missing values).

   :param net: A :class:`~graphoskop.Network` (not a hypergraph nor a simplicial complex).
   :return: The ``graph_tool.Graph``.
   """
   import graph_tool as gt
   ids, nodes, pairs, edges = _lists(net)
   graph = gt.Graph(directed = net.directed)
   graph.add_vertex(len(ids))
   graph.add_edge_list(np.asarray(pairs, dtype = np.int64).reshape(-1, 2))
   vertices = list(graph.vertices())
   for name, values in [("name", [str(v) for v in ids])] + list(nodes.items()):
      graph.vp[name] = _property_map(graph.new_vertex_property, vertices, values)
   order = sorted(graph.edges(), key = lambda e: int(graph.edge_index[e]))   # the order they were added in
   for name, values in edges.items():
      graph.ep[name] = _property_map(graph.new_edge_property, order, values)
   return graph


def _property_map(new, items, values):
   """
   A graph-tool property map of ``values`` (in the order of ``items``): ``double`` if all are
   numbers (NaN for missing ones), else ``string``.
   """
   numeric = all(v is None or (_number(v) is not None and not isinstance(v, str)) for v in values)
   p = new("double" if numeric else "string")
   for item, v in zip(items, values):
      p[item] = (np.nan if v is None else float(v)) if numeric else ("" if v is None else str(v))
   return p
