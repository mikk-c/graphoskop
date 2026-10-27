"""
Hypergraphs and simplicial complexes: interactions among any number of nodes.

A **hypergraph** has nodes and hyperedges, each hyperedge a set of nodes of any size (a meeting
and its participants, a paper and its authors). Two hyperedges can have the same nodes, and a
hyperedge can have one node or none. A **simplicial complex** is a hypergraph closed under
subsets: if a simplex (a set of nodes) is in it, so are all its faces (its non-empty subsets).
A simplex of :math:`d + 1` nodes has dimension :math:`d`: a filled triangle (dimension 2)
implies its three edges and three nodes, but three edges do not imply the filled triangle. A
complex is then given by its **facets**, the simplices that are not a face of another one.

Both are stored as bipartite networks, their incidence graphs: nodes (type 0, first in every
component) linked to the hyperedges or facets they belong to (type 1). This takes memory
proportional to the sum of the sizes of the hyperedges (the clique expansion grows with their
squares), keeps everything (unlike any projection), and is the layout of
:class:`torch_geometric.nn.HypergraphConv`. The connected components of the incidence graph are
those of the hypergraph. A simplicial complex stores its facets only, never its faces.

* **Ids.** Nodes keep their ids. Hyperedges get ids ``Hyperedge(label)``
  (:class:`~graphoskop.Hyperedge`), with the labels given in input or, without labels, their
  positions from 0. Facets get ids ``Simplex(nodes)`` (:class:`~graphoskop.Simplex`): a simplex is
  its set of nodes. These ids never equal
  a node id, so a node ``3`` and ``Hyperedge(3)`` can coexist. Both appear in
  :attr:`~graphoskop.Network.node_map` and :attr:`~graphoskop.Network.node_ids`.
* **Attributes.** Nodes, hyperedges and facets have numeric and qualitative attributes, stored
  as node attributes (one table, shared names, values missing where not given): read and set
  them with :meth:`~graphoskop.Network.get_node_attribute` (nodes only),
  :meth:`~graphoskop.Network.get_hyperedge_attribute` and
  :meth:`~graphoskop.Network.get_facet_attribute`, and their ``set_`` counterparts. A membership
  (a node in a hyperedge) can carry attributes too, such as the position of an author or a
  weight of the node in the hyperedge: they are the edge attributes.
* **Directed hypergraphs.** A directed hyperedge goes from its tail nodes to its head nodes: a
  tail membership is an edge from the node to the hyperedge, a head membership one from the
  hyperedge to the node.
* **Counts.** :meth:`~graphoskop.Network.number_of_nodes` counts the nodes only;
  :meth:`~graphoskop.Network.number_of_hyperedges` and
  :meth:`~graphoskop.Network.number_of_facets` the rest. :meth:`~graphoskop.Network.edges` and
  :meth:`~graphoskop.Network.number_of_edges` give the memberships.

**Analyses.** Run on the incidence graph, most functions of the library would answer the wrong
question, without any warning: degrees and centralities would count hyperedges as nodes, path
lengths would double, clustering would always be zero. So every function says in its
documentation whether it supports hypergraphs and simplicial complexes, and raises
:class:`ValueError` when given one it does not support. To analyze the incidence graph anyway,
pass :func:`incidence_network` (a plain bipartite network); for the network of nodes sharing a
hyperedge, pass :func:`clique_expansion`.
"""

import math

import numpy as np
import pandas as pd
import torch
from numba import njit, prange
from numba.typed import List
from torch_geometric.data import Data

from ._policy import _higher_order
from .isomorphism import _row_codes
from .network import BIPARTITE, Hyperedge, Network, Simplex, _HigherOrderId, _load, _read, _read_nodes, _release_memory, _same_ids


def _collections(items, directed, what):
   """
   Memberships of hyperedges or simplices given as collections of nodes.

   :param items: Dictionary from label to collection, or iterable of collections (labels are
      then their positions). With ``directed``, each collection is a pair ``(tail, head)`` of
      collections.
   :param directed: Whether hyperedges are directed.
   :param what: ``"hyperedge"`` or ``"simplex"``, for error messages.
   :return: Membership table (columns ``node``, ``group`` and, if directed, ``head``) and the
      labels of all the items, empty ones included.
   :raises ValueError: If a collection is a string, or a directed item is not a pair.
   """
   pairs = items.items() if isinstance(items, dict) else enumerate(items)
   nodes, groups, heads, labels = [], [], [], []
   for label, members in pairs:
      labels.append(label)
      if directed:
         if isinstance(members, str) or len(members) != 2:
            raise ValueError(f"a directed {what} is a pair (tail, head) of collections of nodes, got {members!r}")
         parts = ((members[0], False), (members[1], True))
      else:
         parts = ((members, False),)
      for group, head in parts:
         if isinstance(group, (str, _HigherOrderId)) or not hasattr(group, "__iter__"):
            raise ValueError(f"the nodes of a {what} are given as a collection (list, tuple, set), got {group!r}")
         for v in group:
            nodes.append(v)
            groups.append(label)
            heads.append(head)
   table = pd.DataFrame({"node": pd.Series(nodes, dtype = None if nodes else object), "group": pd.Series(groups, dtype = None if groups else object)})   # integer ids stay integers
   if directed:
      table["head"] = np.array(heads, dtype = bool)
   return table, labels


def _attribute_rows(table, node_format, kind):
   """
   Check a user attribute table (node, hyperedge or simplex table) of a higher-order network.

   :param table: Table read by :func:`~graphoskop.network._read_nodes`, id column first.
   :param node_format: ``"wide"`` or ``"long"``.
   :param kind: Name of the table, for error messages.
   :raises ValueError: If the table sets :data:`~graphoskop.BIPARTITE`.
   """
   names = table.columns[1:] if node_format == "wide" else pd.unique(table.iloc[:, 1].dropna())
   if BIPARTITE in list(names):
      raise ValueError(f"{BIPARTITE!r} cannot be an attribute of the {kind} table: it holds the node types of the incidence graph")


def _node_table(tables, n, labels, node_format):
   """
   Node table of the incidence graph, with integer ids: every id gets its type in
   :data:`~graphoskop.BIPARTITE`, and the attributes of the user tables.

   :param tables: List of ``(table, codes)``: user attribute tables (or ``None``) and the integer
      id of each of their rows.
   :param n: Number of nodes (integer ids ``0`` to ``n - 1``; the others are hyperedges or facets).
   :param labels: Total number of ids, and the values of :data:`~graphoskop.BIPARTITE` of the two
      types: ``(total, (node, other))``.
   :param node_format: ``"wide"`` or ``"long"``.
   :return: :class:`pandas.DataFrame` for :func:`~graphoskop.network._load`.
   """
   total, (a, b) = labels
   ids = np.arange(total)
   kind = np.where(ids < n, a, b).astype(object)
   given = [(t, c) for t, c in tables if t is not None]
   if node_format == "wide":
      base = pd.DataFrame({"id": ids, BIPARTITE: kind})
      if not given:
         return base
      attrs = pd.concat([t.iloc[:, 1:].set_axis(range(len(t))).assign(id = c) for t, c in given], ignore_index = True)
      attrs = attrs.drop_duplicates("id", keep = "last")   # as load_network: the last row of a node wins
      return base.merge(attrs, on = "id", how = "left")
   rows = [pd.DataFrame({"id": c, "attribute": t.iloc[:, 1].to_numpy(), "value": t.iloc[:, 2].to_numpy()}) for t, c in given]
   return pd.concat(rows + [pd.DataFrame({"id": ids, "attribute": BIPARTITE, "value": kind})], ignore_index = True)


def _codes(values, index, what):
   """
   Integer ids of some values in an index of labels.

   :param values: Values (:class:`pandas.Series` or array).
   :param index: :class:`pandas.Index` of the labels.
   :param what: Description of the values, for error messages.
   :return: int64 :class:`numpy.ndarray`.
   :raises ValueError: If a value is missing.
   """
   c = index.get_indexer(pd.Series(values, dtype = object) if not isinstance(values, pd.Series) else values)
   if (c < 0).any():
      raise ValueError(f"{what} {list(values)[int(np.argmax(c < 0))]!r} is missing or unknown")
   return c


def _relabel(net, ids, kind):
   """
   Give the nodes of a network loaded with integer ids their real ids, and mark it as a
   higher-order network.

   :param net: Network whose node ids are integers ``0`` to ``len(ids) - 1``.
   :param ids: Real id of each integer id (:class:`numpy.ndarray` of objects).
   :param kind: ``"hypergraph"`` or ``"simplicial"``.
   :return: ``net``, changed in place.
   """
   net.node_ids = [ids[np.asarray(x, dtype = np.int64)] for x in net.node_ids]
   comp = np.repeat(np.arange(len(net.node_ids)), [len(x) for x in net.node_ids])
   local = np.concatenate([np.arange(len(x)) for x in net.node_ids])
   net.node_map = dict(zip(np.concatenate(net.node_ids).tolist(), zip(comp.tolist(), local.tolist())))
   net.higher_order = kind
   return net


def _object_array(values):
   """
   1-D object array of values, whatever they are (tuples stay elements).

   :param values: Sequence.
   :return: :class:`numpy.ndarray`.
   """
   out = np.empty(len(values), dtype = object)
   out[:] = list(values)
   return out


@_higher_order()
def load_hypergraph(hyperedges, nodes = None, hyperedge_table = None, sep = "\t", header = True, node = 0, hyperedge = 1, membership_attributes = True, directed = False, direction = "direction", node_format = "wide", device = "cpu"):
   """
   Load a hypergraph: its hyperedges, and optionally attribute tables of its nodes and of its
   hyperedges (see the module description).

   ``hyperedges`` is given in one of three ways:

   * a **membership table** (file or :class:`pandas.DataFrame`): one row per node in a hyperedge,
     with the node in column ``node`` and the hyperedge label in column ``hyperedge`` (names or
     positions); the other columns are attributes of the memberships (edge attributes), unless
     ``membership_attributes = False``. Directed hypergraphs need a ``direction`` column, with
     ``"tail"`` or ``"head"`` on every row;
   * a **dictionary** from hyperedge label to the collection of its nodes (list, tuple, set);
   * an **iterable of collections** of nodes, labeled by their positions from 0 (to read a file
     with one hyperedge per line: ``[line.split() for line in open(path)]``).

   With dictionaries and iterables, a directed hyperedge is a pair ``(tail, head)`` of
   collections. A node can be in the tail and in the head of the same hyperedge, but cannot be
   twice in the tail (or head, or undirected hyperedge) of one: this raises :class:`ValueError`.

   ``nodes`` is a node table as in :func:`~graphoskop.load_network` (node id first, then the
   attributes in wide format, or attribute name and value in long format); nodes only there are
   isolated (in no hyperedge). ``hyperedge_table`` is the same for hyperedges, with their labels
   in the first column (positions, for an iterable); hyperedges only there are empty. Both tables
   use ``node_format``. Numeric and qualitative attributes, several values of a qualitative one,
   and ids read as text from files behave as in :func:`~graphoskop.load_network`. Neither table
   can set :data:`~graphoskop.BIPARTITE`, which holds ``"node"`` or ``"hyperedge"``.

   :param hyperedges: Membership table (file path or DataFrame), dictionary or iterable.
   :param nodes: Node attribute table (file path or DataFrame), or ``None``.
   :param hyperedge_table: Hyperedge attribute table (file path or DataFrame), or ``None``.
   :param sep: Column separator of the files.
   :param header: Whether the files have a header row.
   :param node: Name or position of the node column of the membership table.
   :param hyperedge: Name or position of the hyperedge column of the membership table.
   :param membership_attributes: Whether to load the other columns of the membership table as
      attributes of the memberships.
   :param directed: Whether hyperedges are directed (from their tails to their heads).
   :param direction: Name or position of the column of the membership table with ``"tail"`` or
      ``"head"``, in directed hypergraphs.
   :param node_format: ``"wide"`` or ``"long"``, the format of both attribute tables.
   :param device: Device of the tensors, e.g. ``"cuda"``.
   :return: The hypergraph, a :class:`~graphoskop.Network` with ``higher_order = "hypergraph"``.
   :raises ValueError: If a node or label is missing, a membership is repeated, a direction is not
      ``"tail"`` or ``"head"``, a collection is a string, or a table sets
      :data:`~graphoskop.BIPARTITE`.

   Example:

   .. code-block:: python

      import graphoskop as gsk
      import pandas as pd

      net = gsk.load_hypergraph([["a", "b", "c"], ["b", "d"], ["d"]])
      net.number_of_nodes(), net.number_of_hyperedges()        # 4, 3
      net.node_map[gsk.Hyperedge(0)]                           # (component, index) of {a, b, c}

      # papers and their authors, with the position of each author
      authors = pd.DataFrame({"author": ["ann", "bo", "bo", "cy"], "paper": ["p1", "p1", "p2", "p2"], "position": [1, 2, 1, 2]})
      papers = pd.DataFrame({"paper": ["p1", "p2"], "year": [2020, 2024]})
      net = gsk.load_hypergraph(authors, hyperedge_table = papers)
      net.get_hyperedge_attribute("year")                      # over the papers of component 0
      net.get_edge_attribute("position")                       # memberships as stored: both directions

      # directed: reactions from reactants to products
      net = gsk.load_hypergraph({"r1": (["a", "b"], ["c"])}, directed = True)
   """
   hdr = 0 if header else None
   extra = []
   if isinstance(hyperedges, pd.DataFrame) or isinstance(hyperedges, (str, bytes)) or hasattr(hyperedges, "__fspath__"):
      table, cols = _read(hyperedges, sep, hdr, [node, hyperedge] + ([direction] if directed else []))
      if directed:
         d = table[cols[2]]
         if not d.isin(["tail", "head"]).all():
            raise ValueError(f"every membership of a directed hypergraph needs 'tail' or 'head' in column {cols[2]!r}, got {d[~d.isin(['tail', 'head'])].iloc[0]!r}")
         head = (d == "head").to_numpy()
      attrs = table.drop(columns = cols) if membership_attributes else table[[]]
      memb = pd.DataFrame({"node": table[cols[0]], "group": table[cols[1]]})
   else:
      memb, extra = _collections(hyperedges, directed, "hyperedge")
      head = memb.pop("head").to_numpy() if directed else None
      attrs = pd.DataFrame(index = range(len(memb)))
   ntab = None if nodes is None else _read_nodes(nodes, sep, hdr, None, node_format, False)
   htab = None if hyperedge_table is None else _read_nodes(hyperedge_table, sep, hdr, None, node_format, False)
   for t, kind in ((ntab, "node"), (htab, "hyperedge")):
      if t is not None:
         _attribute_rows(t, node_format, kind)
   memb, ntab = _same_ids([(memb, ["node"]), (ntab, [] if ntab is None else [ntab.columns[0]])])
   labels = pd.Series(extra, dtype = None if extra else object)
   memb, htab, labels = _same_ids([(memb, ["group"]), (htab, [] if htab is None else [htab.columns[0]]), (labels.to_frame("l") if extra else None, ["l"])])   # no empty object column: it would force the slow path
   labels = [] if labels is None else [labels["l"]]

   # integer ids: nodes 0 .. n - 1, then hyperedges
   nindex = pd.Index(pd.unique(pd.concat([memb["node"]] + ([] if ntab is None else [ntab.iloc[:, 0]]), ignore_index = True).dropna()))
   hindex = pd.Index(pd.unique(pd.concat(labels + [memb["group"]] + ([] if htab is None else [htab.iloc[:, 0]]), ignore_index = True).dropna()))
   n, h = len(nindex), len(hindex)
   s = _codes(memb["node"], nindex, "node")
   t = _codes(memb["group"], hindex, "hyperedge label") + n
   key = t.astype(np.int64) * (n + 1) + s
   if directed:
      key = key * 2 + head
   key.sort()
   if (key[1:] == key[:-1]).any():
      raise ValueError("a node appears twice in a hyperedge (or twice in its tail, or in its head)")
   if directed:
      s, t = np.where(head, t, s), np.where(head, s, t)
   edges = pd.concat([pd.DataFrame({"__source__": s, "__target__": t}), attrs.reset_index(drop = True)], axis = 1)
   tables = [(ntab, None if ntab is None else _codes(ntab.iloc[:, 0], nindex, "node")), (htab, None if htab is None else _codes(htab.iloc[:, 0], hindex, "hyperedge label") + n)]
   table = _node_table(tables, n, (n + h, ("node", "hyperedge")), node_format)
   ids = np.concatenate([_object_array(nindex), _object_array([Hyperedge(x) for x in hindex])])
   net = _load(edges, table, directed = directed, node_format = node_format, bipartite = True, device = device, order = ("node", "hyperedge"))
   del edges, table
   _release_memory()
   return _relabel(net, ids, "hypergraph")


@njit(parallel = True, cache = True)
def _containment(ptr, members, nptr, groups):
   """
   For each group of nodes (a simplex), whether another group strictly contains it, and the
   first group with the same nodes.

   For each group, only the groups of its node with the fewest groups are candidates; containment
   is checked by binary search of each node of the group in the sorted nodes of the candidate.

   :param ptr: CSR pointers of the groups: group ``g`` has nodes ``members[ptr[g]:ptr[g + 1]]``,
      sorted and distinct.
   :param members: Node of each membership.
   :param nptr: CSR pointers of the nodes: node ``v`` is in groups ``groups[nptr[v]:nptr[v + 1]]``.
   :param groups: Group of each membership, by node.
   :return: Boolean array (strictly contained), int64 array (first equal group, itself if none).
   """
   m = len(ptr) - 1
   inside = np.zeros(m, dtype = np.bool_)
   first = np.arange(m)
   for g in prange(m):
      a, b = ptr[g], ptr[g + 1]
      size = b - a
      best = members[a]
      for p in range(a + 1, b):   # node of g in the fewest groups
         v = members[p]
         if nptr[v + 1] - nptr[v] < nptr[best + 1] - nptr[best]:
            best = v
      for q in range(nptr[best], nptr[best + 1]):
         h = groups[q]
         other = ptr[h + 1] - ptr[h]
         if h == g or other < size or (other == size and h > first[g]):
            continue
         i, j, end = a, ptr[h], ptr[h + 1]   # are the nodes of g all in h? binary search of each in the rest of h
         while i < b:
            j += np.searchsorted(members[j:end], members[i])
            if j == end or members[j] != members[i]:
               break
            i += 1
            j += 1
         if i == b:
            if other > size:
               inside[g] = True
               break
            first[g] = min(first[g], h)
   return inside, first


def _facets(group, node, m, n):
   """
   Facets among groups of nodes: the groups with at least two nodes that are not strictly
   contained in another group, one per distinct set of nodes.

   :param group: Group of each membership (integer ids ``0`` to ``m - 1``).
   :param node: Node of each membership (integer ids ``0`` to ``n - 1``).
   :param m: Number of groups.
   :param n: Number of nodes.
   :return: ``(ptr, members, inside, first)``: CSR of the distinct sorted nodes of every group,
      and the results of :func:`_containment`.
   """
   key = np.sort(group.astype(np.int64) * n + node)   # by group, then node (sort: np.unique is slow on large arrays)
   key = key[np.r_[True, key[1:] != key[:-1]]] if len(key) else key   # repeated nodes of a simplex dropped
   g, v = key // n, key % n
   ptr = np.zeros(m + 1, dtype = np.int64)
   np.cumsum(np.bincount(g, minlength = m), out = ptr[1:])
   order = np.argsort(v, kind = "stable")
   nptr = np.zeros(n + 1, dtype = np.int64)
   np.cumsum(np.bincount(v, minlength = n), out = nptr[1:])
   inside, first = _containment(ptr, v, nptr, g[order])
   return ptr, v, inside, first


def _merge_wide(table, key):
   """
   Rows of a wide attribute table merged by key: each attribute keeps the distinct values of the
   merged rows (one value, or a tuple of them).

   :param table: Wide table (id column first).
   :param key: Key of each row.
   :return: ``(table, keys)`` with one row per key.
   """
   from .network import _distinct
   table = table.iloc[:, 1:].reset_index(drop = True).assign(__key__ = key)
   if not table["__key__"].duplicated().any():
      return table.drop(columns = "__key__"), key
   merged = table.groupby("__key__", sort = False).agg(lambda x: _distinct(x.tolist()))
   return merged.reset_index(drop = True), merged.index.to_numpy()


@_higher_order()
def load_simplicial_complex(simplices, nodes = None, simplex_table = None, sep = "\t", header = True, node = 0, simplex = 1, node_format = "wide", device = "cpu"):
   """
   Load a simplicial complex from simplices, and optionally attribute tables of its nodes and of
   its facets (see the module description).

   Only the facets are stored: a simplex given in input that is a face of another one (or a
   single node) is implied by it, and is dropped; a simplex given twice is kept once. Nodes in
   no simplex of two or more nodes are isolated nodes (vertices of the complex).

   ``simplices`` is given in one of three ways:

   * a **membership table** (file or :class:`pandas.DataFrame`): one row per node in a simplex,
     with the node in column ``node`` and a label of the simplex in column ``simplex`` (names or
     positions), which only groups the rows; other columns are ignored;
   * a **dictionary** from simplex label to the collection of its nodes (list, tuple, set);
   * an **iterable of collections** of nodes.

   ``nodes`` is a node table as in :func:`~graphoskop.load_network`. ``simplex_table`` gives
   attributes of facets: in its first column, the simplex as labeled in ``simplices`` (membership
   table or dictionary) or, for an iterable, the collection of its nodes (list, tuple, set or
   :class:`~graphoskop.Simplex`; in long format, one such cell per row). A simplex given twice,
   with attributes on both, gets the distinct values of each attribute (one value, or a tuple of
   them). Both tables use ``node_format``. Neither can set :data:`~graphoskop.BIPARTITE`, which
   holds ``"node"`` or ``"simplex"``.

   ``simplex_table`` is for facets only: an attribute row for a simplex that is not a facet raises
   :class:`ValueError`. Values on all the simplices of a dimension (an edge flow) are simplex
   attributes, set after loading with :meth:`~graphoskop.Network.set_simplex_attribute`.

   Finding the facets checks each simplex against the simplices of its node that is in the
   fewest, in parallel, in compiled code.

   :param simplices: Membership table (file path or DataFrame), dictionary or iterable.
   :param nodes: Node attribute table (file path or DataFrame), or ``None``.
   :param simplex_table: Facet attribute table (file path or DataFrame), or ``None``.
   :param sep: Column separator of the files.
   :param header: Whether the files have a header row.
   :param node: Name or position of the node column of the membership table.
   :param simplex: Name or position of the simplex column of the membership table.
   :param node_format: ``"wide"`` or ``"long"``, the format of both attribute tables.
   :param device: Device of the tensors, e.g. ``"cuda"``.
   :return: The complex, a :class:`~graphoskop.Network` with ``higher_order = "simplicial"``.
   :raises ValueError: If a node or label is missing or a simplex is empty, if the simplex table
      has a row for a simplex that is not a facet or not in ``simplices``, if a collection is a
      string, or if a table sets :data:`~graphoskop.BIPARTITE`.

   Example:

   .. code-block:: python

      import graphoskop as gsk
      import pandas as pd

      # a filled triangle with a tail, and an edge given twice and as a face
      net = gsk.load_simplicial_complex([["a", "b", "c"], ["c", "d"], ["a", "b"], ["d", "c"]])
      net.number_of_facets()                                   # 2: {a, b, c} and {c, d}
      net.node_map[gsk.Simplex(["a", "b", "c"])]

      table = pd.DataFrame({"simplex": [("a", "b", "c")], "area": [0.5]})
      net = gsk.load_simplicial_complex([["a", "b", "c"], ["c", "d"]], simplex_table = table)
      net.get_facet_attribute("area")
   """
   hdr = 0 if header else None
   listed = False
   if isinstance(simplices, pd.DataFrame) or isinstance(simplices, (str, bytes)) or hasattr(simplices, "__fspath__"):
      table, cols = _read(simplices, sep, hdr, [node, simplex])
      memb, extra = pd.DataFrame({"node": table[cols[0]], "group": table[cols[1]]}), []
      del table
   else:
      listed = not isinstance(simplices, dict)
      memb, extra = _collections(simplices, False, "simplex")
   ntab = None if nodes is None else _read_nodes(nodes, sep, hdr, None, node_format, False)
   stab = None if simplex_table is None else _read_nodes(simplex_table, sep, hdr, None, node_format, False)
   for t, kind in ((ntab, "node"), (stab, "simplex")):
      if t is not None:
         _attribute_rows(t, node_format, kind)
   memb, ntab = _same_ids([(memb, ["node"]), (ntab, [] if ntab is None else [ntab.columns[0]])])
   labels = [pd.Series(extra)] if extra else []
   if not listed:
      memb, stab = _same_ids([(memb, ["group"]), (stab, [] if stab is None else [stab.columns[0]])])

   # integer ids of the nodes and of the given simplices, then the facets
   nindex = pd.Index(pd.unique(pd.concat([memb["node"]] + ([] if ntab is None else [ntab.iloc[:, 0]]), ignore_index = True).dropna()))
   gindex = pd.Index(pd.unique(pd.concat(labels + [memb["group"]], ignore_index = True).dropna()))
   n, m = len(nindex), len(gindex)
   v = _codes(memb["node"], nindex, "node")
   g = _codes(memb["group"], gindex, "simplex label")
   ptr, members, inside, first = _facets(g, v, m, n)
   size = np.diff(ptr)
   if (size == 0).any():
      raise ValueError(f"simplex {gindex[int(np.argmax(size == 0))]!r} has no node")
   facet = ~inside & (first == np.arange(m)) & (size >= 2)
   fid = np.full(m, -1, dtype = np.int64)
   fid[facet] = np.arange(int(facet.sum()))
   owner = np.where(inside, -1, fid[first])   # facet of each given simplex, -1 for faces
   f = int(facet.sum())
   keep = np.repeat(facet, size)
   s, t = members[keep], np.repeat(fid[facet], size[facet]) + n
   names = _object_array(nindex)
   sids = _object_array([Simplex._of(frozenset(names[members[ptr[x]:ptr[x + 1]]].tolist())) for x in np.flatnonzero(facet)])

   # facet attributes: rows of the simplex table moved to their facets
   tables = [(ntab, None if ntab is None else _codes(ntab.iloc[:, 0], nindex, "node"))]
   if stab is not None:
      if listed:
         lookup = pd.Index(sids)
         given = [Simplex(x) for x in stab.iloc[:, 0]]
         row = lookup.get_indexer(given)
         if (row < 0).any():
            raise ValueError(f"simplex {given[int(np.argmax(row < 0))]!r} of the simplex table is not a facet: values on faces are simplex attributes (net.set_simplex_attribute)")
      else:
         row = owner[_codes(stab.iloc[:, 0], gindex, "simplex label")]
         if (row < 0).any():
            raise ValueError(f"simplex {stab.iloc[int(np.argmax(row < 0)), 0]!r} of the simplex table is not a facet: values on faces are simplex attributes (net.set_simplex_attribute)")
      if node_format == "wide":
         merged, row = _merge_wide(stab, row)
         stab = merged.reset_index(drop = True)
         stab.insert(0, "__id__", row)
      tables.append((stab, row + n))
   table = _node_table(tables, n, (n + f, ("node", "simplex")), node_format)
   edges = pd.DataFrame({"__source__": s, "__target__": t})
   net = _load(edges, table, node_format = node_format, bipartite = True, device = device, order = ("node", "simplex"))
   del edges, table
   _release_memory()
   return _relabel(net, np.concatenate([names, sids]), "simplicial")


@_higher_order(hypergraph = True, simplicial = True)
def incidence_network(net):
   """
   The incidence graph of a hypergraph or simplicial complex, as a plain bipartite network: the
   same nodes, hyperedges or facets (type 1, :data:`~graphoskop.BIPARTITE` ``"hyperedge"`` or
   ``"simplex"``), memberships and attributes, with ``higher_order = None``, so that every function
   of the library accepts it and treats hyperedges and facets as nodes. Ids stay
   :class:`~graphoskop.Hyperedge` and :class:`~graphoskop.Simplex`, and nodes stay first in every
   component, so results index the same way.

   Edges are shared with ``net``, which is not copied (nothing in the library changes edges in
   place); node and edge attribute tensors and tables are copied, so setting attributes of one
   network leaves the other unchanged.

   :param net: A :class:`~graphoskop.Network`, hypergraph or simplicial complex (other networks
      are returned as a copy).
   :return: New :class:`~graphoskop.Network`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_hypergraph([["a", "b", "c"], ["b", "d"]])
      inc = gsk.incidence_network(net)
      gsk.degree(inc, distribution = None)      # nodes: their hyperedges; hyperedges: their sizes
   """
   comps = []
   for d in net.components:
      rest = {k: v.clone() if k != "edge_index" and hasattr(v, "clone") else v for k, v in d}
      comps.append(Data(**rest))
   Q = net.qualitative_edge_attr
   out = Network(comps, dict(net.node_map), list(net.node_ids), list(net.node_attributes), list(net.edge_attributes), net.directed, net.qualitative_node_attr.copy(), list(net.qualitative_node_attributes), net.node_format, net.bipartite_labels, None if net.bipartite_split is None else list(net.bipartite_split), net.layer_labels, net.entity_ids, net.qualitative_edge_attributes, None if Q is None else Q.copy())
   return out


@_higher_order(hypergraph = True, simplicial = True)
def clique_expansion(net, weight = None, batch_size = 2 ** 24):
   """
   Clique expansion of a hypergraph or simplicial complex: the plain network of its nodes, two
   nodes connected if they share a hyperedge (facet), with edge attribute ``"weight"`` the number
   of hyperedges they share (with ``weight``, the sum over them of the products of the weights of
   their two memberships). In a simplicial complex this is the 1-skeleton: its edges are the edges
   of the complex, with the number of facets containing them as weight. In a directed hypergraph,
   the edges go from the tail nodes to the head nodes of every hyperedge, and the expansion is
   directed.

   This is :func:`~graphoskop.simple_count_projection` of :func:`incidence_network` on the nodes;
   the other bipartite projections apply the same way, e.g. ``hyperbolic_projection(
   incidence_network(net), "node")`` weighs every hyperedge by one over its size, as the
   hypergraph Laplacian of Zhou et al. (2006). Node attributes are kept; those of hyperedges and
   facets are dropped. Nodes sharing no hyperedge stay isolated. The number of edges grows with
   the squares of the hyperedge sizes: a hyperedge of 10,000 nodes gives 50 million edges.

   :param net: A :class:`~graphoskop.Network`, hypergraph or simplicial complex.
   :param weight: Membership attribute used as weight, or ``None`` for unit weights.
   :param batch_size: As in :func:`~graphoskop.simple_count_projection`.
   :return: New :class:`~graphoskop.Network`, with edge attribute ``"weight"``.
   :raises ValueError: If ``net`` is not a hypergraph or a simplicial complex.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_hypergraph([["a", "b", "c"], ["b", "c", "d"]])
      g = gsk.clique_expansion(net)
      g.get_edge_attribute("weight", edge = ("b", "c"))     # 2: they share two hyperedges
   """
   from .bipartite_projections import simple_count_projection
   if net.higher_order is None:
      raise ValueError("clique_expansion is for hypergraphs and simplicial complexes: use the bipartite projections for bipartite networks")
   return simple_count_projection(incidence_network(net), "node", weight, batch_size)


@njit(parallel = True, cache = True)
def _subsets(ptr, members, facets, offsets, r, out):
   """
   All the subsets of ``r`` nodes of some facets, each as a row of increasing node indices, in
   lexicographic order within each facet.

   :param ptr: CSR pointers: facet ``g`` has the sorted nodes ``members[ptr[g]:ptr[g + 1]]``.
   :param members: Nodes of the facets.
   :param facets: Facets to enumerate, each with at least ``r`` nodes.
   :param offsets: First output row of each of them.
   :param r: Size of the subsets.
   :param out: Output, one row per subset (``offsets[-1]`` rows, ``r`` columns), filled in place.
   """
   for f in prange(len(facets)):
      a = ptr[facets[f]]
      z = ptr[facets[f] + 1] - a
      idx = np.arange(r)
      row = offsets[f]
      while True:
         for j in range(r):
            out[row, j] = members[a + idx[j]]
         row += 1
         j = r - 1   # next combination of positions
         while j >= 0 and idx[j] == z - r + j:
            j -= 1
         if j < 0:
            break
         idx[j] += 1
         for l in range(j + 1, r):
            idx[l] = idx[l - 1] + 1


def _distinct_rows(rows):
   """
   Distinct rows of an int64 matrix, in lexicographic order.

   :param rows: Tensor ``[n, r]``.
   :return: Tensor ``[distinct, r]`` on the same device.
   """
   if not len(rows):
      return rows
   code = _row_codes([rows[:, j] for j in range(rows.shape[1])])
   out = rows.new_empty((int(code.max()) + 1, rows.shape[1]))
   out[code] = rows   # repeated rows write the same values
   return out


def _facet_lists(net, component):
   """
   Nodes of the facets of a component of a simplicial complex, in CSR form.

   :param net: Simplicial complex.
   :param component: Component index.
   :return: ``(ptr, members)``: NumPy int64 arrays, facet ``g`` (in internal order) having the
      sorted node indices ``members[ptr[g]:ptr[g + 1]]``.
   """
   k = int(net.bipartite_split[component])
   f = net.components[component].num_nodes - k
   s, t = net._input_edges(component)[0].cpu().numpy()
   node, facet = np.minimum(s, t), np.maximum(s, t) - k
   order = np.argsort(facet * max(k, 1) + node, kind = "stable")
   ptr = np.zeros(f + 1, dtype = np.int64)
   np.cumsum(np.bincount(facet, minlength = f), out = ptr[1:])
   return ptr, node[order].astype(np.int64)


def _simplices(net, component, dimension, batch_size = 2 ** 24):
   """
   All the simplices of one dimension of a component of a simplicial complex (faces of the facets
   included), cached on the network.

   The subsets of ``dimension + 1`` nodes of every facet are enumerated in compiled code, facets
   grouped in batches of about ``batch_size`` subsets, and deduplicated on the device of the
   network (sorts of single columns).

   :param net: Simplicial complex.
   :param component: Component index.
   :param dimension: Dimension of the simplices (0 for the nodes).
   :param batch_size: Target number of enumerated subsets per batch.
   :return: int64 tensor ``[n, dimension + 1]`` of node indices of the component, each row
      increasing, rows in lexicographic order, on the device of the network.
   """
   cache = net.__dict__.setdefault("_skeletons", {})
   if (component, dimension) in cache:
      return cache[(component, dimension)]
   device = net.components[component].edge_index.device
   k, r = int(net.bipartite_split[component]), dimension + 1
   if dimension == 0:
      out = torch.arange(k, device = device).reshape(-1, 1)
   else:
      ptr, members = _facet_lists(net, component)
      size = np.diff(ptr)
      facets = np.flatnonzero(size >= r)
      counts = np.array([math.comb(int(z), r) for z in size[facets]], dtype = np.int64)
      found, start = [], 0
      while start < len(facets):   # batches of whole facets
         stop = start + max(1, int(np.searchsorted(np.cumsum(counts[start:]), batch_size, "right")))
         offsets = np.r_[0, np.cumsum(counts[start:stop])]
         rows = np.empty((int(offsets[-1]), r), dtype = np.int64)
         _subsets(ptr, members, facets[start:stop], offsets, r, rows)
         found.append(_distinct_rows(torch.from_numpy(rows).to(device)))
         start = stop
      out = _distinct_rows(torch.cat(found)) if found else torch.zeros((0, r), dtype = torch.long, device = device)
   cache[(component, dimension)] = out
   return out


def _simplicial(net, what):
   """
   Check that a network is a simplicial complex.

   :param net: Network.
   :param what: Name of the function, for the error message.
   :raises ValueError: If it is not.
   """
   if net.higher_order != "simplicial":
      raise ValueError(f"{what} is defined for simplicial complexes (load them with gsk.load_simplicial_complex)")


def _generalized(net, component, d, m, batch_size = 2 ** 24):
   """
   Generalized degree of every m-face of a component.

   :param net: Simplicial complex.
   :param component: Component index.
   :param d: Dimension of the counted simplices.
   :param m: Dimension of the faces, below ``d``.
   :param batch_size: Target number of (d-simplex, m-face) pairs per batch.
   :return: int64 vector over the rows of ``_simplices(net, component, m)``.
   """
   M = _simplices(net, component, m)
   D = _simplices(net, component, d)
   count = torch.zeros(len(M), dtype = torch.long, device = M.device)
   if not len(D):
      return count
   T = torch.combinations(torch.arange(d + 1, device = M.device), m + 1)   # positions of the m-faces in a d-simplex
   step = max(1, batch_size // len(T))
   for a in range(0, len(D), step):
      S = D[a:a + step][:, T].reshape(-1, m + 1)   # rows stay increasing: positions are increasing
      code = _row_codes([torch.cat([M[:, j], S[:, j]]) for j in range(m + 1)])   # M holds every face: its rows get codes 0 .. len(M) - 1
      count += torch.bincount(code[len(M):], minlength = len(M))
   return count


def _face_rows(net, items, m):
   """
   Component and row in ``_simplices(net, c, m)`` of some m-faces given by the user.

   :param net: Simplicial complex.
   :param items: List of faces: node ids if ``m = 0``, otherwise collections of node ids or
      :class:`~graphoskop.Simplex`.
   :param m: Dimension of the faces.
   :return: List of ``(component, row)``.
   :raises KeyError: If a node does not exist or a face is not in the complex.
   :raises ValueError: If a face does not have ``m + 1`` distinct nodes, or is a facet id where a
      node is expected.
   """
   out = []
   for x in items:
      nodes = [x] if m == 0 else list(Simplex(x).members)
      if len(nodes) != m + 1:
         raise ValueError(f"{x!r} has {len(nodes)} nodes: a face of dimension {m} has {m + 1}")
      ids = [net._nodes(v) for v in nodes]
      if any(len(i) != 1 or i[0][1] >= net.bipartite_split[i[0][0]] for i in ids):
         raise ValueError(f"{x!r} is not made of nodes of the complex (facet ids are not nodes)")
      comps = {i[0][0] for i in ids}
      if len(comps) > 1:
         raise KeyError(f"{x!r} is not a simplex of the complex: its nodes are in different components")
      c = comps.pop()
      row = torch.tensor(sorted(i[0][1] for i in ids), device = net.components[c].edge_index.device)
      hit = (_simplices(net, c, m) == row).all(1).nonzero()
      if not len(hit):
         raise KeyError(f"{x!r} is not a simplex of the complex")
      out.append((c, int(hit[0])))
   return out


def _face_statistic(net, values, simplex, distribution, component, m, name):
   """
   Output of :func:`generalized_degree` and :func:`incidence_number` from per-face values.

   :param values: Function of a component index giving the vector over its m-faces.
   :param name: Name of the value column.
   """
   if simplex is not None:
      many = isinstance(simplex, (list, np.ndarray, pd.Series, pd.Index)) and not (m > 0 and len(simplex) and not isinstance(simplex[0], (list, tuple, set, frozenset, Simplex, np.ndarray)))
      items = list(simplex) if many else [simplex]
      rows = _face_rows(net, items, m)
      vec = {c: values(c).cpu().numpy() for c in {c for c, _ in rows}}
      got = [int(vec[c][i]) for c, i in rows]
      return pd.DataFrame({"simplex": items, name: got}) if many else got[0]
   comps = range(len(net.components)) if component is None else [component]
   if distribution is None:
      out = [values(c) for c in comps]
      return out if component is None else out[0]
   if distribution not in ("count", "pdf", "ccdf"):
      raise ValueError(f"unknown distribution: {distribution}")
   from .network_statistics import _table
   return _table(np.concatenate([values(c).cpu().numpy() for c in comps]), distribution).rename(columns = {"degree": name})


@_higher_order(simplicial = True)
def generalized_degree(net, d, m = 0, simplex = None, distribution = "count", component = None, batch_size = 2 ** 24):
   """
   Generalized degree of the faces of a simplicial complex (Bianconi and Rahmede, 2016): the
   generalized degree :math:`k_{d,m}(\\alpha)` of an :math:`m`-dimensional simplex :math:`\\alpha`
   is the number of :math:`d`-dimensional simplices of the complex that contain it, for
   :math:`m < d`. Faces count: a tetrahedron facet brings its four triangles and six edges. So
   :math:`k_{1,0}` is the degree of the nodes (:func:`~graphoskop.degree`), :math:`k_{2,0}` the
   number of triangles at a node, :math:`k_{2,1}` the number of triangles on an edge. Faces in no
   :math:`d`-simplex have 0.

   * ``simplex`` given: its generalized degree, as an ``int``. A node id when ``m = 0``; otherwise
     the collection of its nodes (list, tuple, set or :class:`~graphoskop.Simplex`).
   * ``simplex`` a list of them: a :class:`pandas.DataFrame` with columns ``simplex`` (as given)
     and ``generalized_degree``.
   * ``simplex = None``: the distribution over all the :math:`m`-simplices (columns
     ``generalized_degree`` and ``count``, ``pdf`` or ``ccdf``, as :func:`~graphoskop.degree`).
   * ``simplex = None`` and ``distribution = None``: an int64 vector over the :math:`m`-simplices
     of the component, aligned with the rows of :meth:`~graphoskop.Network.simplices` (a list of
     vectors if ``component`` is ``None``), on the device of the network.

   The :math:`d`-simplices and the :math:`m`-simplices are enumerated from the facets once and
   cached (see :meth:`~graphoskop.Network.simplices`); each :math:`d`-simplex then gives its
   :math:`\\binom{d + 1}{m + 1}` faces, matched to the :math:`m`-simplices on the device, in
   batches of about ``batch_size``. The number of simplices grows quickly with the size of the
   facets: a facet of :math:`s` nodes has :math:`\\binom{s}{d + 1}` simplices of dimension
   :math:`d`.

   :param net: A simplicial complex (:func:`~graphoskop.load_simplicial_complex`).
   :param d: Dimension of the counted simplices, at least 1.
   :param m: Dimension of the faces whose generalized degree is computed, from 0 to ``d - 1``.
   :param simplex: Face, list of faces, or ``None``.
   :param distribution: ``"count"``, ``"pdf"``, ``"ccdf"``, or ``None`` for the value of every
      face; used when ``simplex`` is ``None``.
   :param component: For the distribution or the vectors, index of the component (0 is the
      largest), or ``None`` for the whole complex.
   :param batch_size: Target number of (simplex, face) pairs matched at once.
   :return: Generalized degree, :class:`pandas.DataFrame`, or vector (list of vectors).
   :raises ValueError: If ``net`` is not a simplicial complex, if not ``0 <= m < d``, if
      ``distribution`` is not valid, or if a face does not have ``m + 1`` nodes.
   :raises KeyError: If a face is not a simplex of the complex.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_simplicial_complex([["a", "b", "c", "d"], ["c", "d", "e"]])
      gsk.generalized_degree(net, 1, 0, "c")             # 4 edges at c: its degree
      gsk.generalized_degree(net, 2, 1, ["c", "d"])      # 3 triangles on the edge {c, d}
      gsk.generalized_degree(net, 3, 0, distribution = "pdf")
      k = gsk.generalized_degree(net, 2, 1, distribution = None, component = 0)
      edges = net.simplices(1, labels = True)           # the edges, aligned with k
   """
   _simplicial(net, "generalized_degree")
   if not 0 <= m < d:
      raise ValueError(f"the generalized degree needs 0 <= m < d, got d = {d}, m = {m}")
   return _face_statistic(net, lambda c: _generalized(net, c, d, m, batch_size), simplex, distribution, component, m, "generalized_degree")


@_higher_order(simplicial = True)
def incidence_number(net, d, simplex = None, distribution = "count", component = None, batch_size = 2 ** 24):
   """
   Incidence number of the :math:`(d - 1)`-dimensional faces of a simplicial complex (Bianconi
   and Rahmede, 2016; the name is from Cinardi, Rapisarda and Bianconi, 2019): :math:`n_\\alpha = k_{d,d-1}(\\alpha) - 1`, the number
   of :math:`d`-simplices sharing the face :math:`\\alpha` beyond the first (see
   :func:`generalized_degree`). In a manifold every :math:`(d - 1)`-face is shared by at most two
   :math:`d`-simplices, so :math:`n_\\alpha \\le 1`; network geometry with flavor builds complexes by
   gluing simplices on faces according to this number. A face in no :math:`d`-simplex has
   :math:`n_\\alpha = -1` (the definition assumes pure complexes, made of :math:`d`-simplices and
   their faces).

   Output and parameters as :func:`generalized_degree` with ``m = d - 1``, faces given as
   collections of :math:`d` nodes (node ids for ``d = 1``), values in a column
   ``incidence_number``.

   :param net: A simplicial complex (:func:`~graphoskop.load_simplicial_complex`).
   :param d: Dimension of the simplices sharing the faces, at least 1.
   :param simplex: Face, list of faces, or ``None``.
   :param distribution: ``"count"``, ``"pdf"``, ``"ccdf"``, or ``None`` for the value of every
      face; used when ``simplex`` is ``None``.
   :param component: For the distribution or the vectors, index of the component (0 is the
      largest), or ``None`` for the whole complex.
   :param batch_size: As in :func:`generalized_degree`.
   :return: Incidence number, :class:`pandas.DataFrame`, or vector (list of vectors).
   :raises ValueError: As :func:`generalized_degree`.
   :raises KeyError: If a face is not a simplex of the complex.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_simplicial_complex([["a", "b", "c"], ["b", "c", "d"], ["b", "c", "e"]])
      gsk.incidence_number(net, 2, ["b", "c"])           # 2: three triangles share {b, c}
      gsk.incidence_number(net, 2)                       # distribution over the edges
   """
   _simplicial(net, "incidence_number")
   if d < 1:
      raise ValueError(f"the incidence number needs d >= 1, got {d}")
   return _face_statistic(net, lambda c: _generalized(net, c, d, d - 1, batch_size) - 1, simplex, distribution, component, d - 1, "incidence_number")


def _facet_sizes(net):
   """
   Sizes of all the hyperedges or facets, and number of isolated nodes (nodes in none).

   :param net: Hypergraph or simplicial complex.
   :return: int64 NumPy array of sizes, and the number of isolated nodes.
   """
   sizes, isolated = [], 0
   for c, d in enumerate(net.components):
      k = int(net.bipartite_split[c])
      s, t = net._input_edges(c)[0].cpu().numpy()
      other = np.maximum(s, t) - k
      sizes.append(np.bincount(other, minlength = d.num_nodes - k))
      isolated += int(k == d.num_nodes)   # a component without hyperedges is one isolated node
   return np.concatenate(sizes) if sizes else np.zeros(0, dtype = np.int64), isolated


@_higher_order(hypergraph = True)
def is_uniform(net):
   """
   Whether a hypergraph is uniform: all its hyperedges have the same number of nodes (a
   :math:`k`-uniform hypergraph; a 2-uniform hypergraph is a network). Empty hyperedges count, with
   size 0; isolated nodes do not matter. A hypergraph without hyperedges is uniform. The size is
   given by :func:`~graphoskop.hyperedge_size`.

   :param net: A hypergraph.
   :return: ``True`` or ``False``.
   :raises ValueError: If ``net`` is not a hypergraph.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      gsk.is_uniform(gsk.load_hypergraph([["a", "b", "c"], ["b", "c", "d"]]))   # True
      gsk.is_uniform(gsk.load_hypergraph([["a", "b", "c"], ["c", "d"]]))        # False
   """
   if net.higher_order != "hypergraph":
      raise ValueError("is_uniform is for hypergraphs (for simplicial complexes, see is_pure)")
   sizes = _facet_sizes(net)[0]
   return bool(len(sizes) == 0 or (sizes == sizes[0]).all())


@_higher_order(simplicial = True)
def is_pure(net):
   """
   Whether a simplicial complex is pure: all its facets have the same dimension :math:`d`, so that
   every simplex is a face of a :math:`d`-simplex. Isolated nodes are facets of dimension 0: a
   complex of triangles with an isolated node is not pure. A complex made only of isolated nodes is
   pure (of dimension 0).

   :param net: A simplicial complex.
   :return: ``True`` or ``False``.
   :raises ValueError: If ``net`` is not a simplicial complex.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      gsk.is_pure(gsk.load_simplicial_complex([["a", "b", "c"], ["c", "d", "e"]]))   # True
      gsk.is_pure(gsk.load_simplicial_complex([["a", "b", "c"], ["c", "d"]]))        # False
      gsk.is_pure(gsk.load_simplicial_complex([["a", "b", "c"], ["e"]]))             # False
   """
   _simplicial(net, "is_pure")
   sizes, isolated = _facet_sizes(net)
   return bool(len(sizes) == 0 or (isolated == 0 and (sizes == sizes[0]).all()))


@_higher_order(simplicial = True)
def is_pseudomanifold(net, boundary = True, batch_size = 2 ** 24):
   """
   Whether a simplicial complex is a :math:`d`-dimensional pseudomanifold:

   1. it is pure of dimension :math:`d \\ge 1` (:func:`is_pure`);
   2. every :math:`(d - 1)`-face is shared by at most two :math:`d`-simplices: its incidence number
      (:func:`incidence_number`) is 0 or 1; with ``boundary = False``, exactly two (incidence
      number 1 everywhere: a closed pseudomanifold, without boundary);
   3. it is strongly connected: any two :math:`d`-simplices are joined by a chain of
      :math:`d`-simplices, each sharing a :math:`(d - 1)`-face with the next, i.e. it has a single
      :math:`(d - 1)`-connected component (:func:`~graphoskop.connected_components_count` with
      ``dimension = d - 1``).

   These are the discrete manifolds of network geometry with flavor (incidence numbers 0 or 1,
   flavor :math:`s = -1`; Bianconi and Rahmede, 2016). Every triangulated manifold is a
   pseudomanifold, but not conversely: a manifold also needs the link of every node (the faces
   opposite it in its simplices) to be a sphere, or a ball on the boundary. For :math:`d = 1` the
   two coincide (paths and cycles); from :math:`d = 2` they do not: a triangulated sphere with two
   of its nodes merged is a pseudomanifold whose merged node has two circles as link.

   :param net: A simplicial complex.
   :param boundary: Whether a boundary is allowed (faces in a single :math:`d`-simplex).
   :param batch_size: As in :func:`generalized_degree`.
   :return: ``True`` or ``False``.
   :raises ValueError: If ``net`` is not a simplicial complex.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      disk = gsk.load_simplicial_complex([["a", "b", "c"], ["b", "c", "d"]])         # two triangles
      gsk.is_pseudomanifold(disk)                                                  # True
      gsk.is_pseudomanifold(disk, boundary = False)                                # False
      sphere = gsk.load_simplicial_complex([["a", "b", "c"], ["a", "b", "d"], ["a", "c", "d"], ["b", "c", "d"]])
      gsk.is_pseudomanifold(sphere, boundary = False)                              # True
      book = gsk.load_simplicial_complex([["a", "b", "c"], ["a", "b", "d"], ["a", "b", "e"]])
      gsk.is_pseudomanifold(book)                                                  # False: three triangles on {a, b}
   """
   from .network_statistics import connected_components_count
   _simplicial(net, "is_pseudomanifold")
   sizes, isolated = _facet_sizes(net)
   if not len(sizes) or not is_pure(net):
      return False
   d = int(sizes[0]) - 1
   for c in range(len(net.components)):
      k = _generalized(net, c, d, d - 1, batch_size)
      if bool((k > 2).any()) or (not boundary and bool((k < 2).any())):
         return False
   return connected_components_count(net, dimension = d - 1, batch_size = batch_size) == 1


def _node_rows(net):
   """
   Network wide positions of the nodes of a hypergraph or simplicial complex (hyperedges and
   facets left out), and the component and index of each node.

   :param net: Hypergraph or simplicial complex.
   :return: ``(rows, comp, local)`` NumPy int64 arrays.
   """
   k = np.asarray(net.bipartite_split, dtype = np.int64)
   comp = np.repeat(np.arange(len(k)), k)
   local = np.arange(int(k.sum())) - np.repeat(np.cumsum(k) - k, k)
   return net.node_offsets[comp] + local, comp, local


def _node_network(net, edges, directed, weighted):
   """
   Plain network over the nodes of a hypergraph or simplicial complex, with given edges and the
   node attributes that have values on some node (those of hyperedges or facets only are dropped,
   and so is :data:`~graphoskop.BIPARTITE`). Components and node indices are those of ``net``, so
   results index the same way (a component of empty hyperedges only becomes a component without
   nodes).

   :param net: Hypergraph or simplicial complex.
   :param edges: One ``(edge_index, weights or None)`` per component, stored as in a network
      (undirected: input edges, then the reverses of those that are not self loops).
   :param directed: Whether the edges are directed.
   :param weighted: Whether the edges have the attribute ``"weight"``.
   :return: :class:`~graphoskop.Network`.
   """
   from .network import _categorize, _qualitative_rows, _quiet_sparse
   blocks = []
   for c in range(len(edges)):
      k = int(net.bipartite_split[c])
      X = net.components[c].node_attr
      if X.layout == torch.sparse_csr:   # first k rows of a CSR matrix, as COO
         p = X.crow_indices()[:k + 1]
         r = torch.repeat_interleave(torch.arange(k, device = X.device), p[1:] - p[:-1])
         blocks.append((r, X.col_indices()[:int(p[-1])], X.values()[:int(p[-1])], k))
      else:
         blocks.append(X[:k])
   sparse = net.node_format == "long"
   if sparse:   # numeric attributes with values on some node (those of the hyperedges or facets only are dropped)
      used = torch.unique(torch.cat([b[1] for b in blocks])) if blocks else torch.zeros(0, dtype = torch.long)
   else:
      used = torch.nonzero(torch.stack([(~torch.isnan(b)).any(0) for b in blocks]).any(0)).reshape(-1) if blocks else torch.zeros(0, dtype = torch.long)
   used = used.cpu()
   remap = torch.full((len(net.node_attributes),), -1, dtype = torch.long)
   remap[used] = torch.arange(len(used))
   comps = []
   for c, (ei, w) in enumerate(edges):
      d = net.components[c]
      if sparse:
         r, col, v, k = blocks[c]
         col = remap.to(col.device)[col]
         with _quiet_sparse():
            X = torch.sparse_coo_tensor(torch.stack([r, col]), v, (k, len(used))).coalesce().to_sparse_csr()
      else:
         X = blocks[c][:, used.to(blocks[c].device)].clone()
      ea = (w.reshape(-1, 1) if weighted else torch.zeros((ei.shape[1], 0), device = ei.device)).to(d.edge_attr.dtype)
      comps.append(Data(edge_index = ei, edge_attr = ea, node_attr = X, num_nodes = X.shape[0]))
   rows, comp, local = _node_rows(net)
   Q = _qualitative_rows(net, rows)
   if net.node_format == "wide":
      quals = [a for a in net.qualitative_node_attributes if a != BIPARTITE and Q[a].notna().any()]
      Q = Q[quals]
   else:
      quals = [a for a in net.qualitative_node_attributes if a != BIPARTITE and (Q["attribute"] == a).any()]
      Q = _categorize(Q[(Q["attribute"] != BIPARTITE).to_numpy()].reset_index(drop = True))
   ids = [x[:k] for x, k in zip(net.node_ids, net.bipartite_split)]
   node_map = dict(zip(np.concatenate(ids).tolist() if ids else [], zip(comp.tolist(), local.tolist())))
   return Network(comps, node_map, ids, [net.node_attributes[j] for j in used.tolist()], ["weight"] if weighted else [], directed, Q, quals, net.node_format)


def _stored(pairs, values, loops, loop_values, n_comp, comp_of, local_of):
   """
   Undirected edges per component, stored as in a network: pairs and self loops (input edges),
   then the reverses of the pairs.

   :param pairs: ``(i, j)`` network wide node positions among the nodes (tensors).
   :param values: Weight of each pair.
   :param loops: Node of each self loop, or ``None``.
   :param loop_values: Weight of each self loop.
   :param n_comp: Number of components.
   :param comp_of: Component of each node position (tensor).
   :param local_of: Index in its component of each node position (tensor).
   :return: List of ``(edge_index, weights)``.
   """
   i, j = pairs
   if loops is not None:
      i, j, values = torch.cat([i, loops]), torch.cat([j, loops]), torch.cat([values, loop_values])
   c = comp_of[i]
   out = []
   for x in range(n_comp):
      m = c == x
      a, b, v = local_of[i[m]], local_of[j[m]], values[m]
      r = a != b
      out.append((torch.stack([torch.cat([a, b[r]]), torch.cat([b, a[r]])]), torch.cat([v, v[r]])))
   return out


def _check_weights(w, what):
   """
   :raises ValueError: If some weights are not positive and finite.
   """
   if len(w) and not bool(torch.isfinite(w).all() & (w > 0).all()):
      raise ValueError(f"{what} must be positive and finite")


@_higher_order(hypergraph = True)
def hypergraph_walk_network(net, model = "zhou", weight = None, vertex_weight = None, batch_size = 2 ** 24):
   """
   The network of the nodes of a hypergraph on which a random walk model moves: an edge from
   :math:`u` to :math:`v` weighs the probability of a step from :math:`u` to :math:`v`, up to the
   factor that normalizes the steps from :math:`u` (their sum :math:`d_u`, the strength of
   :math:`u`). Every function of the library then applies to it; the Laplacian, random walk and
   node vector functions use ``model = "zhou"`` directly on a hypergraph (see :doc:`higher_order`).
   :math:`w(e)` is the hyperedge attribute ``weight`` (1 without it), :math:`|e|` the size of
   :math:`e`:

   * ``"zhou"`` (Zhou, Huang and Schölkopf, 2006): from :math:`u`, a hyperedge :math:`e` containing
     it with probability proportional to :math:`w(e)`, then a node of :math:`e` uniformly,
     :math:`u` included. Undirected, with weights
     :math:`A_{uv} = \\sum_{e \\ni u, v} w(e) / |e|` (self loops included, :math:`d_u =
     \\sum_{e \\ni u} w(e)`): the Laplacian :math:`D - A` is the hypergraph Laplacian of Zhou et
     al., and the walk is reversible with stationary distribution proportional to :math:`d_u`.
   * ``"carletti"`` (Carletti, Battiston, Cencetti and Fanelli, 2020): steps to another node of a
     shared hyperedge, with :math:`A_{uv} = \\sum_{e \\ni u, v} w(e) (|e| - 1)` for
     :math:`u \\ne v` and no self loops: larger hyperedges weigh more. Undirected and reversible;
     the paper has :math:`w(e) = 1`.
   * ``"edvw"`` (edge-dependent vertex weights, Chitra and Raphael, 2019): a hyperedge with
     probability proportional to :math:`w(e)`, then a node :math:`v` of it with probability
     :math:`\\gamma_e(v) / \\delta(e)`, :math:`\\gamma_e(v)` the membership attribute
     ``vertex_weight`` and :math:`\\delta(e) = \\sum_{v \\in e} \\gamma_e(v)`. Directed (the walk is
     not reversible in general), with :math:`A_{uv} = \\sum_{e \\ni u, v} w(e) \\gamma_e(v) /
     \\delta(e)`, self loops included; its Laplacian in Chitra and Raphael is the Chung Laplacian
     (``kind = "chung"`` with ``alpha = 0`` in :func:`~graphoskop.laplacian`).

   The pairs of nodes sharing hyperedges are generated as in the bipartite projections,
   ``batch_size`` at a time: the number of edges grows with the squares of the hyperedge sizes.
   Nodes are those of ``net``, in the same components and order, with the attributes that have
   values on some node (hyperedge attributes are dropped); the edge attribute ``"weight"`` holds
   :math:`A_{uv}`.

   :param net: An undirected hypergraph.
   :param model: ``"zhou"``, ``"carletti"`` or ``"edvw"``.
   :param weight: Hyperedge attribute :math:`w(e)`, or ``None`` for 1.
   :param vertex_weight: For ``"edvw"``, the membership attribute :math:`\\gamma_e(v)`.
   :param batch_size: Number of pairs of nodes sharing a hyperedge generated at once.
   :return: New :class:`~graphoskop.Network`, undirected (directed for ``"edvw"``).
   :raises ValueError: If ``net`` is not an undirected hypergraph, ``model`` is not valid,
      ``vertex_weight`` is missing for ``"edvw"`` (or given for another model), or weights are not
      positive and finite.
   :raises KeyError: If an attribute does not exist.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_hypergraph([["a", "b", "c"], ["b", "d"]])
      g = gsk.hypergraph_walk_network(net)                       # Zhou: a - b weighs 1/3
      g = gsk.hypergraph_walk_network(net, "carletti")            # a - b weighs 2
      gsk.laplacian(net)                                         # the Laplacian of the Zhou network
   """
   from .bipartite_projections import _biadjacency, _paths
   if net.higher_order != "hypergraph" or net.directed:
      raise ValueError("hypergraph_walk_network is for undirected hypergraphs")
   if model not in ("zhou", "carletti", "edvw"):
      raise ValueError(f"unknown random walk model: {model}")
   if (model == "edvw") != (vertex_weight is not None):
      raise ValueError("vertex_weight is needed by model 'edvw', and only by it")
   inc = incidence_network(net)
   dev = net.components[0].edge_index.device
   om = torch.cat([inc._node_vector(weight, c)[k:] for c, k in enumerate(net.bipartite_split)]).double() if weight is not None else None
   rows, B, _ = _biadjacency(inc, "node", vertex_weight)   # nodes x hyperedges, gamma or 1
   (a, e), g = B.indices(), B.values().double()
   h = sum(d.num_nodes - k for d, k in zip(net.components, net.bipartite_split))
   om = torch.ones(h, dtype = torch.float64, device = dev) if om is None else om.to(dev)
   _check_weights(om, "hyperedge weights")
   _check_weights(g, "vertex weights")
   delta = torch.bincount(e, g, minlength = h)
   _, comp, local = _node_rows(net)
   comp, local = torch.as_tensor(comp, device = dev), torch.as_tensor(local, device = dev)
   n_comp = len(net.components)
   n = len(comp)
   loops = lambda x: torch.bincount(a, x, minlength = n)   # one self loop per node: the sum over its hyperedges
   at = torch.unique(a)
   if model == "zhou":
      _, _, i, j, v = _paths(inc, "node", None, lambda wa, wb, c, s: om[c] / s[c], batch_size)
      edges = _stored((i, j), v, at, loops(om[e] / delta[e])[at], n_comp, comp, local)
   elif model == "carletti":
      _, _, i, j, v = _paths(inc, "node", None, lambda wa, wb, c, s: om[c] * (s[c] - 1), batch_size)
      edges = _stored((i, j), v, None, None, n_comp, comp, local)
   else:
      _, _, i, j, v_ij = _paths(inc, "node", vertex_weight, lambda wa, wb, c, s: om[c] * wb.double() / s[c], batch_size)
      _, _, i2, j2, v_ji = _paths(inc, "node", vertex_weight, lambda wa, wb, c, s: om[c] * wa.double() / s[c], batch_size)
      src, dst = torch.cat([i, j2, at]), torch.cat([j, i2, at])   # both directions of every pair, then the loops
      val = torch.cat([v_ij, v_ji, loops(om[e] * g / delta[e])[at]])
      cs = comp[src]
      edges = [(torch.stack([local[src[cs == x]], local[dst[cs == x]]]), val[cs == x]) for x in range(n_comp)]
   return _node_network(net, edges, model == "edvw", True)


def _skeleton_network(net):
   """
   The 1-skeleton of a simplicial complex (its nodes and its edges, unweighted), as a plain
   network with the components and node order of ``net``.
   """
   edges = []
   for c in range(len(net.components)):
      E = _simplices(net, c, 1)
      edges.append((torch.cat([E.t(), E.t().flip(0)], 1), None))
   return _node_network(net, edges, False, False)


def _walk_view(net, weight):
   """
   Network and weight on which Laplacian, random walk and node vector functions run: ``net``
   itself, the Zhou network of a hypergraph (:func:`hypergraph_walk_network`, ``weight`` a
   hyperedge attribute), or the 1-skeleton of a simplicial complex.

   :param net: Network.
   :param weight: The ``weight`` given to the function.
   :return: ``(network, weight)``.
   :raises ValueError: For directed hypergraphs, or a weight on a simplicial complex.
   """
   if net.higher_order is None:
      return net, weight
   if net.higher_order == "simplicial":
      if weight is not None:
         raise ValueError("the walks and Laplacians of a simplicial complex are those of its 1-skeleton, which is unweighted: use weight = None")
      return _skeleton_network(net), None
   if net.directed:
      raise ValueError("Laplacians and walks of directed hypergraphs are not supported yet (random_walk is)")
   return hypergraph_walk_network(net, "zhou", weight), "weight"


def _boundary(net, component, k):
   """
   Boundary matrix :math:`B_k` of a component as COO entries.

   :return: ``(rows, cols, signs, shape)``: row of each entry in ``_simplices(net, c, k - 1)``,
      column in ``_simplices(net, c, k)``, sign :math:`\\pm 1` (float), and the shape.
   """
   F, S = _simplices(net, component, k - 1), _simplices(net, component, k)
   cols = torch.arange(len(S), device = S.device).repeat_interleave(k + 1)
   faces = torch.stack([torch.cat([S[:, :i], S[:, i + 1:]], 1) for i in range(k + 1)], 1).reshape(-1, k)   # face i drops node i
   signs = torch.tensor([(-1.) ** i for i in range(k + 1)], device = S.device).repeat(len(S))
   if not len(S):
      return cols, cols, signs, (len(F), 0)
   code = _row_codes([torch.cat([F[:, j], faces[:, j]]) for j in range(k)])   # F holds every face: its rows get codes 0 .. len(F) - 1
   return code[len(F):], cols, signs, (len(F), len(S))


def _matrix(rows, cols, values, shape, sparse):
   """
   Dense or sparse (coalesced COO) float32 matrix from entries; duplicates are summed.
   """
   from .network import _quiet_sparse
   if not sparse:
      return torch.zeros(shape, device = rows.device).index_put_((rows, cols), values.float(), accumulate = True)
   key, inv = torch.unique(rows * shape[1] + cols, return_inverse = True)   # faster than coalesce
   v = torch.zeros(len(key), device = rows.device).index_add_(0, inv, values.float())
   with _quiet_sparse():
      return torch.sparse_coo_tensor(torch.stack([key // shape[1], key % shape[1]]), v, shape, is_coalesced = True)


@_higher_order(simplicial = True)
def boundary_matrix(net, k, component = 0, sparse = False):
   """
   Boundary matrix :math:`B_k` of a simplicial complex: one row per :math:`(k - 1)`-simplex, one
   column per :math:`k`-simplex, in the order of :meth:`~graphoskop.Network.simplices`, with
   :math:`(B_k)_{\\tau \\sigma} = (-1)^i` if :math:`\\tau` is :math:`\\sigma` without its
   :math:`i`-th node, 0 otherwise. Every simplex has the reference orientation given by the order
   of its nodes (internal indices, increasing): :math:`B_1` gives each edge :math:`[u, v]`,
   :math:`u < v`, the entries :math:`-1` at :math:`u` and :math:`+1` at :math:`v`, and
   :math:`B_k B_{k + 1} = 0`.

   :param net: A simplicial complex.
   :param k: Dimension of the columns, at least 1.
   :param component: Index of the component (0 is the largest).
   :param sparse: Whether to return a sparse (COO) tensor instead of a dense one.
   :return: float32 tensor on the device of the network.
   :raises ValueError: If ``net`` is not a simplicial complex or ``k`` is below 1.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_simplicial_complex([["a", "b", "c"]])
      gsk.boundary_matrix(net, 1)        # nodes x edges [a, b], [a, c], [b, c]
      gsk.boundary_matrix(net, 2)        # tensor([[1.], [-1.], [1.]]): [b, c] - [a, c] + [a, b]
   """
   _simplicial(net, "boundary_matrix")
   if k < 1:
      raise ValueError(f"boundary matrices start at k = 1, got {k}")
   r, c, v, shape = _boundary(net, component, k)
   return _matrix(r, c, v, shape, sparse)


@_higher_order(simplicial = True)
def hodge_laplacian(net, k, component = 0, normalized = False, sparse = False):
   """
   Hodge Laplacian of the :math:`k`-simplices of a simplicial complex,
   :math:`L_k = B_k^T B_k + B_{k + 1} B_{k + 1}^T` (:func:`boundary_matrix`; the first term is
   absent for :math:`k = 0`, where :math:`L_0` is the Laplacian of the 1-skeleton). Rows and
   columns follow :meth:`~graphoskop.Network.simplices`. Its kernel has the dimension of the
   :math:`k`-th homology of the complex (its :math:`k`-dimensional holes), and it splits flows on
   :math:`k`-simplices into gradient, curl and harmonic parts (Hodge decomposition).

   With ``normalized = True`` and :math:`k = 1`, the normalized Hodge 1-Laplacian of Schaub,
   Benson, Horn, Lippner and Jadbabaie (2020),
   :math:`\\mathcal{L}_1 = D_2 B_1^T D_1^{-1} B_1 + B_2 D_3 B_2^T D_2^{-1}`, with
   :math:`D_2 = \\max(\\mathrm{diag}(|B_2| 1), I)`, :math:`D_1 = 2 \\, \\mathrm{diag}(|B_1| D_2 1)`
   and :math:`D_3 = I / 3`: it is not symmetric, and :math:`-\\mathcal{L}_1 / 2` is the
   projection of the random walk of :func:`simplicial_random_walk` on oriented edges.

   :param net: A simplicial complex.
   :param k: Dimension of the simplices, at least 0.
   :param component: Index of the component (0 is the largest).
   :param normalized: Whether to return the normalized Hodge 1-Laplacian (``k = 1`` only).
   :param sparse: Whether to return a sparse (COO) tensor instead of a dense one.
   :return: float32 tensor, :math:`n_k \\times n_k`, on the device of the network.
   :raises ValueError: If ``net`` is not a simplicial complex, ``k`` is negative, or
      ``normalized`` is asked for ``k != 1``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      hollow = gsk.load_simplicial_complex([["a", "b"], ["b", "c"], ["a", "c"]])
      filled = gsk.load_simplicial_complex([["a", "b", "c"]])
      import torch
      torch.linalg.eigvalsh(gsk.hodge_laplacian(hollow, 1))   # one zero: the hole
      torch.linalg.eigvalsh(gsk.hodge_laplacian(filled, 1))   # no zero: the hole is filled
      gsk.hodge_laplacian(filled, 1, normalized = True)
   """
   _simplicial(net, "hodge_laplacian")
   if k < 0:
      raise ValueError(f"the dimension must be at least 0, got {k}")
   if normalized and k != 1:
      raise ValueError("the normalized Hodge Laplacian (Schaub et al.) is defined for k = 1")
   n = len(_simplices(net, component, k))
   rows, cols, vals = [], [], []

   def product(r1, c1, v1, r2, c2, v2, scale_left = None, scale_mid = None, scale_right = None):
      """Entries of M1^T diag(mid) M2 scaled on both sides, with M1, M2 sharing their rows."""
      key = torch.argsort(r2)
      start = torch.searchsorted(r2[key], r1, right = False)
      count = torch.searchsorted(r2[key], r1, right = True) - start
      a = torch.repeat_interleave(torch.arange(len(r1), device = r1.device), count)
      b = key[torch.repeat_interleave(start, count) + torch.arange(int(count.sum()), device = r1.device) - torch.repeat_interleave(torch.cumsum(count, 0) - count, count)]
      v = v1[a] * v2[b]
      if scale_mid is not None:
         v = v * scale_mid[r1[a]]
      if scale_left is not None:
         v = v * scale_left[c1[a]]
      if scale_right is not None:
         v = v * scale_right[c2[b]]
      return c1[a], c2[b], v

   d2 = d1 = None
   if normalized:
      r2, c2, v2, _ = _boundary(net, component, 2)
      d2 = torch.bincount(r2, minlength = n).clamp(min = 1).float()   # max(number of triangles, 1)
      r1, c1, v1, _ = _boundary(net, component, 1)
      d1 = 2 * torch.zeros(len(_simplices(net, component, 0)), device = r1.device).index_add_(0, r1, d2[c1])
   if k > 0:   # down part: B_k^T B_k (rows of B_k: the (k - 1)-faces)
      r, c, v, _ = _boundary(net, component, k)
      parts = product(r, c, v, r, c, v, d2, None if d1 is None else 1 / d1, None)
      rows.append(parts[0]); cols.append(parts[1]); vals.append(parts[2])
   r, c, v, _ = _boundary(net, component, k + 1)   # up part: B_{k+1} B_{k+1}^T (columns: the (k + 1)-simplices)
   parts = product(c, r, v, c, r, v, None, None if d2 is None else torch.full((len(_simplices(net, component, k + 1)),), 1 / 3, device = r.device), None if d2 is None else 1 / d2)
   rows.append(parts[0]); cols.append(parts[1]); vals.append(parts[2])
   return _matrix(torch.cat(rows), torch.cat(cols), torch.cat(vals), (n, n), sparse)


def _edge_walk_tables(net, component):
   """
   Tables of the random walk of :func:`simplicial_random_walk` on the oriented edges of a
   component: state ``s`` is edge ``s % m`` in its reference orientation if ``s < m``, reversed
   otherwise.

   :return: Dictionary with the edges ``E``, ``m``, the tail and head of every state, the states
      leaving every node sorted by tail with their cumulative weights :math:`\\max(\\deg, 1)`, and
      the triangles of every edge (CSR) with the three edges and signs of each triangle.
   """
   E = _simplices(net, component, 1)
   m, dev = len(E), E.device
   tail, head = torch.cat([E[:, 0], E[:, 1]]), torch.cat([E[:, 1], E[:, 0]])
   r, c, v, _ = _boundary(net, component, 2)   # edge r is in triangle c with sign v
   deg = torch.bincount(r, minlength = m)
   w = deg.clamp(min = 1).double().repeat(2)   # weight of a target state: its upper degree, at least 1
   order = torch.argsort(tail, stable = True)
   n = len(_simplices(net, component, 0))
   out = torch.bincount(tail, minlength = n)
   end = torch.cumsum(out, 0)
   cw = torch.cumsum(w[order], 0)
   before = torch.where(end - out > 0, cw[(end - out - 1).clamp(min = 0)], torch.zeros_like(cw[:1]).expand(n) if len(cw) else torch.zeros(n, dtype = torch.float64, device = dev))
   key = torch.argsort(r, stable = True)   # triangles of each edge
   tstart = torch.cumsum(deg, 0) - deg
   tri_e = torch.empty((int(c.max()) + 1 if len(c) else 0, 3), dtype = torch.long, device = dev)
   tri_s = torch.empty_like(tri_e, dtype = torch.float64)
   pos = torch.arange(len(c), device = dev) % 3   # entries come triangle by triangle, faces in order
   tri_e[c, pos], tri_s[c, pos] = r, v.double()
   return dict(E = E, m = m, tail = tail, head = head, order = order, out = out, end = end, cw = cw, before = before, strength = torch.zeros(n, dtype = torch.float64, device = dev).index_add_(0, tail, w), deg = deg, tstart = tstart, tlist = c[key], tri_e = tri_e, tri_s = tri_s)


def _edge_walk(T, start, length, gen):
   """
   Walks of :func:`simplicial_random_walk` from some states, all advancing together.

   :return: int64 tensor of states, one row per walk, ``length + 1`` columns.
   """
   m, dev = T["m"], start.device
   path = torch.empty((len(start), length + 1), dtype = torch.long, device = dev)
   path[:, 0] = s = start
   u = lambda: torch.rand(len(s), dtype = torch.float64, device = dev, generator = gen)
   for step in range(1, length + 1):
      e, positive = s % m, s < m
      lower, forward = u() < 0.5, u() < 0.5
      # lower: through the head (forward) or the tail (backward), to a state leaving that node, then reversed if backward
      x = torch.where(forward, T["head"][s], T["tail"][s])
      j = torch.searchsorted(T["cw"], T["before"][x] + u() * T["strength"][x], right = True)
      j = torch.minimum(j.clamp(min = 0), T["end"][x] - 1)
      f = T["order"][j]
      low = torch.where(forward, f, (f + m) % (2 * m))
      # upper: a triangle of the edge, then one of its three edges with the opposite relative orientation (the edge itself: reversed)
      deg = T["deg"][e]
      t = T["tlist"][(T["tstart"][e] + (u() * deg).long().clamp(max = (deg - 1).clamp(min = 0))).clamp(max = max(len(T["tlist"]) - 1, 0))] if len(T["tlist"]) else e
      pick = (u() * 3).long().clamp(max = 2)
      if len(T["tlist"]):
         te, ts = T["tri_e"][t], T["tri_s"][t]
         own = (ts * (te == e[:, None])).sum(1)   # sign of the edge in the triangle
         g, sg = te.gather(1, pick[:, None])[:, 0], ts.gather(1, pick[:, None])[:, 0]
         o = torch.where(positive, 1., -1.).double()
         up = torch.where(-o * own * sg > 0, g, g + m)
      else:
         up = s
      flip = u() < 0.5   # an edge in no triangle stays or reverses
      up = torch.where(deg > 0, up, torch.where(flip, (s + m) % (2 * m), s))
      path[:, step] = s = torch.where(lower, low, up)
   return path


@_higher_order(simplicial = True)
def simplicial_random_walk(net, source, length, walks = 1, seed = None):
   """
   Random walks on the oriented edges of a simplicial complex (Schaub, Benson, Horn, Lippner and
   Jadbabaie, 2020), whose average dynamics is the normalized Hodge 1-Laplacian
   (:func:`hodge_laplacian` with ``normalized = True``): it diffuses edge flows, and so sees the
   holes of the complex. At each step, with probability 1/2 each:

   * **lower move**: through the head of the current oriented edge (forward) or its tail
     (backward), with probability 1/2 each, to an oriented edge leaving (forward) or entering
     (backward) that node, the current edge reversed included, with probability proportional to
     :math:`\\max(\\deg, 1)`, :math:`\\deg` its number of triangles;
   * **upper move**: if the edge is in triangles, one of them uniformly, then, with probability 1/3
     each, the edge reversed or one of the two other edges of the triangle, oriented opposite to
     the current one with respect to the triangle; if the edge is in no triangle, it stays or is
     reversed, with probability 1/2 each.

   All the walks of a component advance together, one step at a time, on the device of the
   network.

   :param net: A simplicial complex.
   :param source: Starting edge: a tuple ``(u, v)`` (the oriented edge from ``u`` to ``v``), or a
      :class:`~graphoskop.Simplex` or set (the edge in its reference orientation, from the node with
      the lower internal index); or a list of them, each with its own walks.
   :param length: Number of steps of each walk.
   :param walks: Number of walks from each source.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :return: :class:`pandas.DataFrame` with one row per step: columns ``walk``, ``step`` (0 for the
      source), ``from_node`` and ``to_node`` (the node ids of the tail and the head of the oriented
      edge); first ``source`` (as given) with a list.
   :raises ValueError: If ``net`` is not a simplicial complex, ``length`` is negative or
      ``walks`` below 1, or a source is not an edge of the complex.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_simplicial_complex([["a", "b", "c"], ["c", "d"]])
      w = gsk.simplicial_random_walk(net, ("a", "b"), 10, seed = 1)     # walk, step, from_node, to_node
      w = gsk.simplicial_random_walk(net, [("a", "b"), {"c", "d"}], 10, walks = 5)
   """
   _simplicial(net, "simplicial_random_walk")
   if length < 0 or walks < 1:
      raise ValueError("random walks: length must be at least 0 and walks at least 1")
   many = isinstance(source, list)
   sources = source if many else [source]
   rng = np.random.default_rng(seed)
   starts = []
   for x in sources:
      nodes = list(x) if isinstance(x, tuple) else list(Simplex(x).members)
      ids = [net._nodes(v) for v in nodes]
      if len(nodes) != 2 or any(len(i) != 1 for i in ids) or ids[0][0][0] != ids[1][0][0]:
         raise ValueError(f"{x!r} is not an edge of the complex")
      (c, a), (_, b) = ids[0][0], ids[1][0]
      E = _simplices(net, c, 1)
      hit = ((E[:, 0] == min(a, b)) & (E[:, 1] == max(a, b))).nonzero()
      if not len(hit) or max(a, b) >= net.bipartite_split[c]:
         raise ValueError(f"{x!r} is not an edge of the complex")
      starts.append((c, int(hit[0]) + (len(E) if isinstance(x, tuple) and a > b else 0)))
   comp = np.repeat([c for c, _ in starts], walks)
   first = np.repeat([s for _, s in starts], walks)
   tails = np.empty((len(comp), length + 1), dtype = object)
   heads = np.empty_like(tails)
   for c in np.unique(comp):
      rows = np.flatnonzero(comp == c)
      T = _edge_walk_tables(net, int(c))
      dev = T["E"].device
      gen = torch.Generator(device = dev)
      gen.manual_seed(int(rng.integers(2 ** 62)))
      P = _edge_walk(T, torch.as_tensor(first[rows], device = dev), length, gen)
      ids = net.node_ids[c]
      tails[rows], heads[rows] = ids[T["tail"][P].cpu().numpy()], ids[T["head"][P].cpu().numpy()]
   out = pd.DataFrame({"walk": np.tile(np.repeat(np.arange(walks), length + 1), len(sources)), "step": np.tile(np.arange(length + 1), len(comp)), "from_node": tails.reshape(-1), "to_node": heads.reshape(-1)})
   if many:
      out.insert(0, "source", pd.Series(np.repeat(np.array(sources + [None], dtype = object)[:-1], walks * (length + 1)), dtype = object))
   return out


#: Prime modulo which boundary matrices are reduced (see :func:`betti_numbers`).
_PRIME = 2147483647


@njit(cache = True)
def _inverse(a, p):
   """Inverse of ``a`` modulo the prime ``p``."""
   r, e, a = 1, p - 2, a % p
   while e:
      if e & 1:
         r = r * a % p
      a = a * a % p
      e >>= 1
   return r


@njit(cache = True)
def _axpy(r1, v1, r2, v2, f, p):
   """
   Sparse column ``v1 - f v2`` modulo ``p``, columns given by sorted rows and values.
   """
   out_r = np.empty(len(r1) + len(r2), np.int64)
   out_v = np.empty(len(r1) + len(r2), np.int64)
   i = j = k = 0
   while i < len(r1) or j < len(r2):
      if j == len(r2) or (i < len(r1) and r1[i] < r2[j]):
         row, val = r1[i], v1[i]
         i += 1
      elif i == len(r1) or r2[j] < r1[i]:
         row, val = r2[j], (p - f * v2[j] % p) % p
         j += 1
      else:
         row, val = r1[i], (v1[i] - f * v2[j] % p) % p
         i += 1
         j += 1
      if val:
         out_r[k], out_v[k] = row, val
         k += 1
   return out_r[:k], out_v[:k]


@njit(cache = True)
def _reduce(ptr, rows, vals, skip, n_rows, p):
   """
   Column reduction of a sparse matrix modulo the prime ``p`` (left to right, pivot = last
   nonzero row), skipping the columns known to reduce to zero.

   :param ptr: CSC pointers: column ``j`` has the entries ``ptr[j]`` to ``ptr[j + 1] - 1``.
   :param rows: Row of each entry.
   :param vals: Value of each entry, in ``[0, p)``.
   :param skip: Whether to skip each column.
   :param n_rows: Number of rows.
   :param p: Prime.
   :return: Pivot rows of the nonzero reduced columns (their number is the rank).
   """
   owner = np.full(n_rows, -1, np.int64)
   R, V = List(), List()
   pivots = np.empty(len(ptr) - 1, np.int64)
   rank = 0
   for j in range(len(ptr) - 1):
      if skip[j]:
         continue
      o = np.argsort(rows[ptr[j]:ptr[j + 1]])
      r, v = rows[ptr[j]:ptr[j + 1]][o], vals[ptr[j]:ptr[j + 1]][o]
      while len(r):
         q = owner[r[-1]]
         if q < 0:
            break
         r, v = _axpy(r, v, R[q], V[q], v[-1] * _inverse(V[q][-1], p) % p, p)
      if len(r):
         owner[r[-1]] = rank
         R.append(r)
         V.append(v)
         pivots[rank] = r[-1]
         rank += 1
   return pivots[:rank]


def _ranks(net, component, lo, hi):
   """
   Exact ranks of the boundary matrices :math:`B_k` of a component, :math:`lo \\le k \\le hi`
   (:math:`lo \\ge 1`), modulo a large prime, reduced from :math:`hi` down: a :math:`k`-simplex
   that is the pivot of a reduced column of :math:`B_{k + 1}` has a column of :math:`B_k` that
   reduces to zero, and is skipped (clearing).

   :return: Dictionary from ``k`` to the rank of :math:`B_k`.
   """
   ranks, cleared = {}, np.zeros(0, dtype = np.int64)
   for k in range(hi, lo - 1, -1):
      r, c, v, shape = _boundary(net, component, k)
      skip = np.zeros(shape[1], dtype = np.bool_)
      skip[cleared] = True
      ptr = np.arange(shape[1] + 1, dtype = np.int64) * (k + 1)
      cleared = _reduce(ptr, r.cpu().numpy(), np.where(v.cpu().numpy() > 0, 1, _PRIME - 1).astype(np.int64), skip, shape[0], _PRIME)
      ranks[k] = len(cleared)
   return ranks


def _top(net, component):
   """Largest dimension of a facet of a component (0 without facets)."""
   k = int(net.bipartite_split[component])
   s, t = net._input_edges(component)[0].cpu().numpy()
   return max(int(np.bincount(np.maximum(s, t) - k).max()) - 1, 0) if len(s) else 0


@_higher_order(simplicial = True)
def betti_numbers(net, dimension = None, component = None):
   """
   Betti numbers of a simplicial complex: :math:`\\beta_k` counts its :math:`k`-dimensional holes,
   :math:`\\beta_0` its connected components, :math:`\\beta_1` its loops that do not bound a
   surface (a hollow triangle has one, a filled one none), :math:`\\beta_2` its voids (a hollow
   tetrahedron, a sphere). They are the dimensions of the homology groups with real coefficients,
   :math:`\\beta_k = n_k - \\mathrm{rank} B_k - \\mathrm{rank} B_{k + 1}`, with :math:`n_k` the
   number of :math:`k`-simplices and :math:`B_k` the boundary matrices
   (:func:`boundary_matrix`), and equal the dimension of the kernel of the Hodge Laplacian
   :math:`L_k` (:func:`hodge_laplacian`): the harmonic signals of :func:`hodge_decomposition`.

   The ranks are exact, not numerical: the sparse boundary matrices are reduced by column
   operations in compiled code, modulo the prime :math:`2^{31} - 1` (equal to the rank over the
   reals unless the homology has torsion of that order, which no complex of a sane size has), from
   the highest dimension down, skipping the columns that a reduction of the dimension above shows
   to vanish (clearing; Chen and Kerber, 2011). Every simplex up to dimension
   ``dimension + 1`` is enumerated (see :meth:`~graphoskop.Network.simplices`), which limits this
   to complexes whose facets are not too large: a facet of :math:`m` nodes has :math:`2^m - 1`
   faces.

   :param net: A simplicial complex.
   :param dimension: Largest dimension :math:`k` of the Betti numbers, or ``None`` for the
      dimension of the complex (of its largest facet).
   :param component: Index of the component (0 is the largest), or ``None`` for the whole complex.
   :return: List of the Betti numbers :math:`\\beta_0, \\beta_1, \\ldots`.
   :raises ValueError: If ``net`` is not a simplicial complex or ``dimension`` is negative.

   Example:

   .. code-block:: python

      import graphoskop as gsk
      import itertools

      gsk.betti_numbers(gsk.load_simplicial_complex([["a", "b"], ["b", "c"], ["a", "c"]]))   # [1, 1]
      gsk.betti_numbers(gsk.load_simplicial_complex([["a", "b", "c"]]))                     # [1, 0, 0]
      sphere = gsk.load_simplicial_complex(itertools.combinations("abcd", 3))
      gsk.betti_numbers(sphere)                                                              # [1, 0, 1]
   """
   _simplicial(net, "betti_numbers")
   if dimension is not None and dimension < 0:
      raise ValueError(f"the dimension must be at least 0, got {dimension}")
   comps = range(len(net.components)) if component is None else [component]
   tops = {c: _top(net, c) for c in comps}
   D = max(tops.values()) if dimension is None else int(dimension)
   out = np.zeros(D + 1, dtype = np.int64)
   for c in comps:
      d = min(D, tops[c])
      rank = {0: 0, **_ranks(net, c, 1, d + 1)}
      out[:d + 1] += [len(_simplices(net, c, k)) - rank[k] - rank[k + 1] for k in range(d + 1)]
   return out.tolist()


@_higher_order(simplicial = True)
def euler_characteristic(net, component = None):
   """
   Euler characteristic of a simplicial complex,
   :math:`\\chi = \\sum_k (-1)^k n_k = \\sum_k (-1)^k \\beta_k`, with :math:`n_k` its number of
   :math:`k`-simplices and :math:`\\beta_k` its Betti numbers (:func:`betti_numbers`): 1 for a
   simplex or a tree, 2 for a sphere, 0 for a torus or a circle. It counts every simplex, so every
   face is enumerated (see :meth:`~graphoskop.Network.simplices`).

   :param net: A simplicial complex.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole complex.
   :return: The Euler characteristic.
   :raises ValueError: If ``net`` is not a simplicial complex.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      gsk.euler_characteristic(gsk.load_simplicial_complex([["a", "b", "c"], ["c", "d"]]))   # 1
   """
   _simplicial(net, "euler_characteristic")
   comps = range(len(net.components)) if component is None else [component]
   return int(sum((-1) ** k * len(_simplices(net, c, k)) for c in comps for k in range(_top(net, c) + 1)))


def _signal(net, x, dimension, component):
   """
   Signals on the simplices of one dimension of a component, in double precision.

   :param x: Simplex attribute name (or node attribute name, for dimension 0), list of names, or
      vector / matrix (one signal per column) over :meth:`~graphoskop.Network.simplices`.
   :param dimension: Dimension of the simplices, or ``None`` to take it from the attributes.
   :return: ``(signals, dimension)``.
   :raises ValueError: If the dimension is missing or does not match, or values are missing.
   """
   names = [x] if isinstance(x, str) else x if isinstance(x, (list, tuple)) and len(x) and isinstance(x[0], str) else None
   if names is None:
      if dimension is None:
         raise ValueError("signals given as vectors need the dimension of their simplices")
      X = torch.as_tensor(x, device = net.components[component].edge_index.device).double()
   else:
      dims = {net.simplex_attributes.get(a, 0) for a in names}
      if len(dims) > 1 or (dimension is not None and dimension not in dims):
         raise ValueError(f"the signals {names} are not all on simplices of dimension {dimension if dimension is not None else min(dims)}")
      dimension = dims.pop()
      get = net.get_simplex_attribute if dimension else net.get_node_attribute
      X = torch.stack([get(a, component) for a in names], 1).double()
      X = X[:, 0] if isinstance(x, str) else X
   n = len(_simplices(net, component, dimension))
   if X.shape[0] != n:
      raise ValueError(f"the signal has {X.shape[0]} values, the component {n} simplices of dimension {dimension}")
   if X.isnan().any():
      raise ValueError("the signal has missing values (NaN)")
   return X, int(dimension)


def _operator(net, component, k):
   """
   Sparse boundary matrix :math:`B_k` and its transpose, CSR, double precision; ``None`` if
   :math:`k < 1` or either side has no simplex.
   """
   if k < 1:
      return None
   r, c, v, shape = _boundary(net, component, k)
   if not min(shape):
      return None
   from .network import _quiet_sparse
   B = _matrix(r, c, v, shape, True).double()
   with _quiet_sparse():
      return B.to_sparse_csr(), B.t().coalesce().to_sparse_csr()


def _psd_solve(A, At, b, tol, max_iter):
   """
   Orthogonal projections of the columns of ``b`` on the image of :math:`M^T`: :math:`M^T y`, with
   :math:`y` solving the consistent semidefinite system :math:`M M^T y = M b` by conjugate
   gradients (all columns at once).

   :param A: Sparse :math:`M`.
   :param At: Sparse :math:`M^T`.
   :param b: Right-hand sides, one column each (double).
   :param tol: Relative residual at which a column stops.
   :param max_iter: Largest number of iterations.
   :return: The projections, same shape as ``b``.
   :raises ValueError: If a column does not converge.
   """
   rhs = A @ b
   X = torch.zeros_like(rhs)
   R = rhs.clone()
   P = R.clone()
   rr = (R * R).sum(0)
   stop = tol ** 2 * rr
   for _ in range(max_iter):
      if bool((rr <= stop).all()):
         return At @ X
      AP = A @ (At @ P)
      a = rr / (P * AP).sum(0).clamp(min = 1e-300)
      X += a * P
      R -= a * AP
      new = (R * R).sum(0)
      P = R + (new / rr.clamp(min = 1e-300)) * P
      rr = new
   raise ValueError(f"hodge_decomposition: conjugate gradients did not converge in {max_iter} iterations")


@_higher_order(simplicial = True)
def hodge_decomposition(net, signal, dimension = None, component = 0, tol = 1e-10, max_iter = None):
   """
   Hodge decomposition of a signal on the :math:`k`-simplices of a simplicial complex (a flow on
   its edges for :math:`k = 1`) into three orthogonal parts (Lim, 2020; Barbarossa and Sardellitti,
   2020), :math:`x = B_k^T a + B_{k + 1} b + h`:

   * **gradient** :math:`B_k^T a`: differences of a potential :math:`a` on the
     :math:`(k - 1)`-simplices (for edges, a flow from high to low node potential), free of curl;
   * **curl** :math:`B_{k + 1} b`: circulations around the :math:`(k + 1)`-simplices (for edges,
     flows around filled triangles), free of divergence;
   * **harmonic** :math:`h`: the rest, free of both (:math:`B_k h = 0`, :math:`B_{k + 1}^T h = 0`,
     :math:`L_k h = 0`), flowing around the holes of the complex; it is zero if the complex has no
     :math:`k`-dimensional hole (:func:`betti_numbers`).

   Signals are in the reference orientation of the simplices (see
   :meth:`~graphoskop.Network.set_simplex_attribute`), as the boundary matrices
   (:func:`boundary_matrix`). For :math:`k = 0` (a node signal), the gradient part is zero and the
   harmonic part is the mean over the component. The two projections solve
   :math:`B_k B_k^T a = B_k x` and :math:`B_{k + 1}^T B_{k + 1} b = B_{k + 1}^T x` by conjugate
   gradients on the sparse boundary matrices, in double precision, on the device of the network.

   :param net: A simplicial complex.
   :param signal: Simplex attribute (name, see :meth:`~graphoskop.Network.set_simplex_attribute`;
      or a node attribute for :math:`k = 0`), list of them, or a vector (one value per
      :math:`k`-simplex, in the order of :meth:`~graphoskop.Network.simplices`) or matrix (one
      signal per column).
   :param dimension: Dimension :math:`k` of the simplices; ``None`` to take it from the attributes.
   :param component: Index of the component (0 is the largest).
   :param tol: Relative residual at which conjugate gradients stop.
   :param max_iter: Largest number of conjugate gradient iterations, or ``None`` for 10 times the
      number of simplices of the system.
   :return: ``(gradient, curl, harmonic)``: three tensors of the shape of the signal, summing to it.
   :raises ValueError: If ``net`` is not a simplicial complex, the dimension is missing, the signal
      has the wrong length or missing values, or conjugate gradients do not converge.

   Example:

   .. code-block:: python

      import graphoskop as gsk
      import pandas as pd

      net = gsk.load_simplicial_complex([["a", "b", "c"], ["c", "d"], ["d", "e"], ["c", "e"]])
      flows = pd.DataFrame({"u": ["a", "b", "a", "c", "d", "e"], "v": ["b", "c", "c", "d", "e", "c"], "flow": [1., 1., 0., 2., 2., 2.]})
      net.set_simplex_attribute(flows, dimension = 1, oriented = True)
      gradient, curl, harmonic = gsk.hodge_decomposition(net, "flow")
      (harmonic ** 2).sum() / net.get_simplex_attribute("flow").double().pow(2).sum()   # share around the hole c, d, e
   """
   _simplicial(net, "hodge_decomposition")
   X, k = _signal(net, signal, dimension, component)
   x = X.reshape(len(X), -1)
   parts = []
   for M in (_operator(net, component, k), _operator(net, component, k + 1)):
      if M is None:
         parts.append(torch.zeros_like(x))
         continue
      A, At = M if len(parts) == 0 else (M[1], M[0])   # gradient: B_k; curl: B_{k + 1}^T
      parts.append(_psd_solve(A, At, x, tol, 10 * A.shape[0] if max_iter is None else max_iter))
   parts.append(x - parts[0] - parts[1])
   return tuple(p.reshape(X.shape) for p in parts)


@_higher_order(simplicial = True)
def simplicial_fourier_basis(net, dimension, component = 0):
   """
   Fourier basis of the signals on the :math:`k`-simplices of a simplicial complex (Barbarossa and
   Sardellitti, 2020): the eigenvalues (frequencies) and eigenvectors (modes) of the Hodge
   Laplacian :math:`L_k = B_k^T B_k + B_{k + 1} B_{k + 1}^T` (:func:`hodge_laplacian`), each mode
   of one of the three kinds of :func:`hodge_decomposition`: ``"harmonic"`` (frequency 0, one per
   :math:`k`-dimensional hole), ``"gradient"`` (eigenvectors of :math:`B_k^T B_k`) or ``"curl"``
   (eigenvectors of :math:`B_{k + 1} B_{k + 1}^T`). Gradient and curl modes are sorted by frequency
   within their kind, and all modes by frequency, harmonic first. For :math:`k = 0` this is the
   graph Fourier basis of the 1-skeleton (:func:`~graphoskop.graph_fourier_basis`), with curl
   modes for the nonzero frequencies. Used by :func:`simplicial_fourier_transform` and
   :func:`inverse_simplicial_fourier_transform`; filtering the coefficients of one kind filters
   that part of a signal.

   The number of modes of each kind is exact (the ranks of :func:`betti_numbers`), so no
   numerical threshold splits them. The phase of every mode is fixed (its largest entry, in
   modulus, is positive). Dense eigendecompositions, in double precision on the device of the
   network: :math:`O(n_k^3)` time and :math:`n_k^2` memory.

   :param net: A simplicial complex.
   :param dimension: Dimension :math:`k` of the simplices.
   :param component: Index of the component (0 is the largest).
   :return: ``(frequencies, modes, kinds)``: vector of length :math:`n_k`, :math:`n_k` by
      :math:`n_k` matrix (one mode per column), and NumPy array of the kind of each mode.
   :raises ValueError: If ``net`` is not a simplicial complex or ``dimension`` is negative.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_simplicial_complex([["a", "b", "c"], ["c", "d"], ["d", "e"], ["c", "e"]])
      freq, U, kinds = gsk.simplicial_fourier_basis(net, 1)
      kinds              # 1 harmonic mode (the hole c, d, e), 4 gradient modes, 1 curl mode
   """
   _simplicial(net, "simplicial_fourier_basis")
   if dimension < 0:
      raise ValueError(f"the dimension must be at least 0, got {dimension}")
   k = int(dimension)
   n = len(_simplices(net, component, k))
   rank = {0: 0, **_ranks(net, component, max(k, 1), k + 1)}
   dev = net.components[component].edge_index.device
   L = []
   for j, up in ((k, False), (k + 1, True)):
      M = _operator(net, component, j)
      B = torch.zeros((0, n), dtype = torch.float64, device = dev) if M is None else (M[1] if up else M[0]).to_dense()
      L.append(B.T @ B)   # B_k^T B_k, or B_{k + 1} B_{k + 1}^T
   vals, vecs, kinds = [], [], []
   for kind, A, m in (("harmonic", L[0] + L[1], n - rank[k] - rank[k + 1]), ("gradient", L[0], rank[k]), ("curl", L[1], rank[k + 1])):
      if m:
         w, U = torch.linalg.eigh(A)
         w, U = (torch.zeros_like(w[:m]), U[:, :m]) if kind == "harmonic" else (w[-m:], U[:, -m:])
         vals.append(w); vecs.append(U); kinds += [kind] * m
   val, vec = torch.cat(vals), torch.cat(vecs, 1)
   order = torch.argsort(val, stable = True)
   val, vec = val[order], vec[:, order]
   big = vec.abs().argmax(0)
   vec = vec * torch.sign(vec[big, torch.arange(n, device = dev)])
   return val, vec, np.array(kinds, dtype = object)[order.cpu().numpy()]


@_higher_order(simplicial = True)
def simplicial_fourier_transform(net, signal, dimension = None, component = 0, basis = None):
   """
   Fourier transform of a signal on the :math:`k`-simplices of a simplicial complex,
   :math:`\\hat{x} = U^T x`: its coefficients on the modes of :func:`simplicial_fourier_basis`,
   from the lowest frequency to the highest. The energy of the coefficients of each kind
   (``basis[2]``) is the energy of the matching part of :func:`hodge_decomposition`; the transform
   keeps the norm (Parseval) and is inverted by :func:`inverse_simplicial_fourier_transform`.

   :param net: A simplicial complex.
   :param signal: Simplex attribute name, list of them, vector or matrix (one signal per column),
      as in :func:`hodge_decomposition`.
   :param dimension: Dimension :math:`k` of the simplices; ``None`` to take it from the attributes.
   :param component: Index of the component (0 is the largest).
   :param basis: ``(frequencies, modes, kinds)`` from :func:`simplicial_fourier_basis`, or ``None``
      to compute it; pass it to transform many signals.
   :return: Coefficients: vector of length :math:`n_k` (or matrix, one column per signal), double
      precision, on the device of the network.
   :raises ValueError: As :func:`hodge_decomposition`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      freq, U, kinds = basis = gsk.simplicial_fourier_basis(net, 1)
      c = gsk.simplicial_fourier_transform(net, "flow", basis = basis)
      (c[kinds == "curl"] ** 2).sum()            # energy of the curl part
   """
   _simplicial(net, "simplicial_fourier_transform")
   X, k = _signal(net, signal, dimension, component)
   U = (basis if basis is not None else simplicial_fourier_basis(net, k, component))[1]
   if U.shape[0] != X.shape[0]:
      raise ValueError(f"simplicial_fourier_transform: the signal has {X.shape[0]} values, the basis {U.shape[0]}")
   return U.T @ X


@_higher_order(simplicial = True)
def inverse_simplicial_fourier_transform(net, coefficients, dimension = None, component = 0, basis = None):
   """
   Inverse Fourier transform on the :math:`k`-simplices of a simplicial complex,
   :math:`x = U \\hat{x}`: the signal with the given coefficients on the modes of
   :func:`simplicial_fourier_basis`. Changing the coefficients in between filters the signal, e.g.
   keeping those of one kind, or ``h(freq) * c`` for a filter ``h``.

   :param net: A simplicial complex.
   :param coefficients: Vector of :math:`n_k` coefficients, or matrix (one column per signal).
   :param dimension: Dimension :math:`k` of the simplices, needed if ``basis`` is ``None``.
   :param component: Index of the component (0 is the largest).
   :param basis: ``(frequencies, modes, kinds)`` from :func:`simplicial_fourier_basis`, or ``None``
      to compute it.
   :return: Signal (vector or matrix) over :meth:`~graphoskop.Network.simplices`, double precision;
      store it with :meth:`~graphoskop.Network.set_simplex_attribute`.
   :raises ValueError: If ``net`` is not a simplicial complex, neither ``basis`` nor ``dimension``
      is given, or the number of coefficients is not the number of modes.

   Example:

   .. code-block:: python

      import graphoskop as gsk
      import torch

      freq, U, kinds = basis = gsk.simplicial_fourier_basis(net, 1)
      c = gsk.simplicial_fourier_transform(net, "flow", basis = basis)
      no_curl = gsk.inverse_simplicial_fourier_transform(net, c * torch.as_tensor(kinds != "curl"), basis = basis)
   """
   _simplicial(net, "inverse_simplicial_fourier_transform")
   if basis is None:
      if dimension is None:
         raise ValueError("inverse_simplicial_fourier_transform needs the basis or the dimension")
      basis = simplicial_fourier_basis(net, dimension, component)
   U = basis[1]
   c = torch.as_tensor(coefficients, device = U.device).double()
   if c.shape[0] != U.shape[1]:
      raise ValueError(f"inverse_simplicial_fourier_transform: {c.shape[0]} coefficients for {U.shape[1]} modes")
   return U @ c


def _incidence_operator(net, component, weight):
   """
   The Laplacian of the Zhou random walk of a component of an undirected hypergraph (that of
   :func:`hypergraph_walk_network`), :math:`L = D - H W B^{-1} H^T`, and its walk, as products
   with the sparse incidence matrix :math:`H`: memory linear in the memberships, whatever the
   sizes of the hyperedges.

   :param net: Undirected hypergraph.
   :param component: Component index.
   :param weight: Name of the hyperedge attribute weighting the hyperedges, or ``None``.
   :return: ``(n, mv, diagonal, step, degree)``: number of nodes, product with :math:`L` (of an
      ``n`` by ``s`` matrix), diagonal of :math:`L`, product of a distribution with the transition
      matrix :math:`P^T \\phi`, :math:`P = D^{-1} H W B^{-1} H^T` (vector), and the weighted
      degrees :math:`D` (float64, on the device of the network).
   :raises ValueError: If the weights are not positive and finite.
   """
   from .network import _quiet_sparse
   k = int(net.bipartite_split[component])
   E = net.components[component].num_nodes - k
   s, t = net._input_edges(component)[0]
   v, e = torch.minimum(s, t), torch.maximum(s, t) - k
   dev = v.device
   w = torch.ones(E, dtype = torch.float64, device = dev) if weight is None else net.get_hyperedge_attribute(weight, component).double().to(dev)
   _check_weights(w, "hyperedge weights")
   c = w / torch.bincount(e, minlength = E).double().clamp(min = 1)   # w(e) / |e|
   with _quiet_sparse():
      H = torch.sparse_coo_tensor(torch.stack([v, e]), torch.ones(len(v), dtype = torch.float64, device = dev), (k, E)).coalesce()
      Ht, H = H.t().coalesce().to_sparse_csr(), H.to_sparse_csr()
   d = H @ w
   diag = d - H @ c
   inv = torch.where(d > 0, 1 / d.clamp(min = 1e-300), 0.)
   mv = lambda x: d[:, None] * x - H @ (c[:, None] * (Ht @ x))
   step = lambda phi: H @ (c * (Ht @ (phi * inv)))
   return k, mv, diag, step, d


@_higher_order(hypergraph = True, simplicial = True)
def set_hypergraph_route(route = "auto", ratio = 1.):
   """
   How the Laplacian solvers, stationary distributions and sparse Fourier bases of hypergraphs are
   computed. They are those of the random walk of Zhou et al. (see :ref:`hypergraph-walks`), whose
   network links every pair of nodes sharing a hyperedge: a hyperedge of :math:`s` nodes gives
   :math:`s (s - 1) / 2` pairs, so large hyperedges make it large (one of 10,000 nodes, 50 million
   pairs). The same products can be computed with the incidence matrix instead,
   :math:`L x = D x - H W B^{-1} H^T x`, in memory linear in the memberships, without building the
   network:

   * ``"auto"`` (default): with the incidence matrix when the walk network would have more than
     ``ratio`` times as many pairs as there are memberships (with the default ratio of 1, from a
     mean hyperedge size of about 3), with the walk network otherwise;
   * ``"incidence"``: always with the incidence matrix;
   * ``"walk_network"``: always with the walk network.

   Results are the same either way, up to the tolerance of the iterative methods. The incidence
   matrix is used by :func:`~graphoskop.laplacian_solver`,
   :func:`~graphoskop.stationary_distribution`, :func:`~graphoskop.generalized_euclidean` with a
   solver (``sparse = True``) and :func:`~graphoskop.graph_fourier_basis` with a few modes
   (``k``) on large components (and so the Fourier transforms computing that basis). The other
   functions always build the walk network: dense ones (pseudoinverse, resistances, ...) take
   memory quadratic in the nodes anyway, and some need its edges (e.g.
   :func:`~graphoskop.earth_movers_distance`). Directed hypergraphs have no Laplacian.

   :param route: ``"auto"``, ``"incidence"`` or ``"walk_network"``.
   :param ratio: With ``"auto"``, the number of pairs per membership above which the incidence
      matrix is used.
   :raises ValueError: If ``route`` is unknown or ``ratio`` negative.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_hypergraph("meetings.tsv")             # some meetings of thousands of people
      solve = gsk.laplacian_solver(net)                     # on the incidence matrix (auto)
      gsk.set_hypergraph_route("walk_network")              # build the Zhou network instead
      gsk.set_hypergraph_route()                            # back to the default
   """
   from ._views import _ROUTE
   if route not in ("auto", "incidence", "walk_network"):
      raise ValueError(f"unknown route {route!r}: use 'auto', 'incidence' or 'walk_network'")
   if ratio < 0:
      raise ValueError(f"ratio must be non negative, got {ratio}")
   _ROUTE["route"], _ROUTE["ratio"] = route, float(ratio)
