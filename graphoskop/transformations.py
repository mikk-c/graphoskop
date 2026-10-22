"""
Transformations of a network into another network.

:func:`to_undirected` symmetrizes a directed network: all the edges between two nodes, in either
direction, become one undirected edge, whose attributes combine theirs. :func:`remove_isolates`
drops the nodes without edges, :func:`remove_self_loops` the edges from a node to itself,
:func:`remove_parallel_edges` merges the edges between the same two nodes. :func:`subgraph` keeps a
set of nodes and the edges between them, :func:`ego_network` the neighbors of a node and the edges
between them, :func:`to_strongly_connected` the edges within strongly connected components. :func:`add_nodes`, :func:`add_edges`, :func:`remove_nodes` and
:func:`remove_edges` add or remove nodes and edges, one or many at once.
:func:`flip_edge_direction` reverses some or all of the edges of a directed network.

:func:`network_union`, :func:`network_difference` and :func:`network_intersection` combine the
edges of two networks with the same node ids. :func:`quotient_network` collapses sets of nodes into
single nodes, and :func:`network_condensation` the strongly connected components of a directed
network.
"""

import itertools

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .network import BIPARTITE, LAYER, Network, _distinct, _edge_frame, _load, _release_memory, _take_edges, _to_long, _wide_rows
from .multilayer import _flat_tables
from .network_statistics import _strong_labels, connected_components_nodes
from .signed import SIGN


def _merge(edges, src, tgt, combine, sign, disagree):
   """
   Merge the edges of a table that join the same two nodes, in either direction.

   :param edges: Edge table: endpoint columns, then attributes (qualitative ones keep their
      distinct values, see :func:`~graphoskop.network._distinct`).
   :param src: Columns identifying the first endpoint (e.g. ``["source"]``, or the entity and the
      layer of a copy).
   :param tgt: Columns identifying the second endpoint, matching ``src``.
   :param combine: ``"sum"``, ``"mean"`` or ``"max"``.
   :param sign: Name of the sign attribute, handled separately if in the table.
   :param disagree: Sign of an edge whose merged edges have both signs: 1, -1 or 0.
   :return: Merged table, one row per pair of nodes in order of first appearance, endpoints
      ordered as in their first edge (up to the swap that puts the same node first in all rows).
   """
   m = len(edges)
   ids = pd.concat([edges[src], edges[tgt].set_axis(src, axis = 1)], ignore_index = True)
   codes = ids.groupby(src, sort = False, dropna = False).ngroup().to_numpy()
   a, b = codes[:m], codes[m:]
   swap = a > b
   edges = edges.copy()
   for cs, ct in zip(src, tgt):
      if cs != ct:
         vs, vt = edges[cs].to_numpy(), edges[ct].to_numpy()
         edges[cs], edges[ct] = np.where(swap, vt, vs), np.where(swap, vs, vt)
   keys = list(dict.fromkeys(src + tgt))
   attrs = [c for c in edges.columns if c not in keys]
   g = edges.groupby([np.minimum(a, b), np.maximum(a, b)], sort = False)
   qual = [a for a in attrs if a not in edges[attrs].select_dtypes(["number", "bool"]).columns]
   num = [a for a in attrs if a not in qual]
   out = g[num].sum(min_count = 1) if combine == "sum" else g[num].agg(combine)
   for q in qual:
      out[q] = g[q].agg(_distinct)
   if sign in attrs:
      x = edges[sign].to_numpy()
      both = pd.DataFrame({"p": x > 0, "n": x < 0}).groupby([np.minimum(a, b), np.maximum(a, b)], sort = False).max()
      p, n = both["p"].to_numpy(), both["n"].to_numpy()
      out[sign] = np.where(p & n, disagree, np.where(p, 1, np.where(n, -1, out[sign].to_numpy())))
   ends = g[keys].first()
   return pd.concat([ends, out], axis = 1)[edges.columns].reset_index(drop = True)


def _with_edges(net, edges, qualitative = None):
   """
   Copy of a network with new edges in each component (same nodes, so the same components).

   :param net: Network.
   :param edges: One ``(edge_index, edge_attr)`` per component, stored as in the network
      (undirected: input edges, then the reverses of those that are not self loops).
   :param qualitative: Qualitative edge table of the new input edges, or ``None`` if the input
      edges are those of ``net``, in the same order (its table is then kept).
   :return: New :class:`~graphoskop.Network`; node tensors are copied.
   """
   comps = []
   for d, (ei, ea) in zip(net.components, edges):
      rest = {k: v.clone() if torch.is_tensor(v) else v for k, v in d if k not in ("edge_index", "edge_attr")}
      comps.append(Data(edge_index = ei, edge_attr = ea, **rest))
   return Network(comps, dict(net.node_map), list(net.node_ids), list(net.node_attributes), list(net.edge_attributes), net.directed, net.qualitative_node_attr.copy(), list(net.qualitative_node_attributes), net.node_format, net.bipartite_labels, None if net.bipartite_split is None else list(net.bipartite_split), net.layer_labels, net.entity_ids, net.qualitative_edge_attributes, net.qualitative_edge_attr if qualitative is None else qualitative)


def to_undirected(net, combine = "sum", sign = SIGN, disagree = "zero"):
   """
   Undirected version of a directed network.

   All the edges between the same two nodes, in either direction (parallel edges included), are
   merged into one undirected edge; self loops of a node are merged into one self loop. Each
   numeric edge attribute of the merged edge combines the values of its edges, ignoring NaN:

   * ``"sum"``: their sum, so that weights give :math:`A + A^T`;
   * ``"mean"``: their mean (an edge in one direction only keeps its value);
   * ``"max"``: their maximum.

   Each qualitative edge attribute keeps the distinct values of the merged edges: the value if they
   all have the same (missing values aside), a tuple of the distinct values otherwise.

   The sign attribute (if the network has one, see :doc:`signed`) is not combined: if all the
   merged edges have the same sign, the edge keeps it; if some are positive and some negative,
   ``disagree`` decides: ``"positive"`` (+1), ``"negative"`` (-1) or ``"zero"`` (0, which signed
   analyses treat as no edge). Edges with sign 0 or NaN do not count as disagreeing.

   Nodes and node attributes are kept, isolated nodes included, and so are node types (bipartite
   networks) and the device. In multilayer networks intra-layer edges are merged within each
   layer, and interlayer edges (couplings included) between the same two copies are merged too.

   :param net: A directed :class:`~graphoskop.Network`.
   :param combine: ``"sum"``, ``"mean"`` or ``"max"``.
   :param sign: Name of the sign edge attribute (ignored if the network has no such attribute).
   :param disagree: ``"positive"``, ``"negative"`` or ``"zero"``.
   :return: The undirected network, a new :class:`~graphoskop.Network`.
   :raises ValueError: If ``net`` is undirected, or ``combine`` or ``disagree`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("trust.tsv", directed = True)           # columns weight, sign
      und = gsk.to_undirected(net, combine = "mean", disagree = "negative")
      und.get_edge_attribute("weight", edge = ("alice", "bob"))       # mean of both directions
   """
   if not net.directed:
      raise ValueError("the network is already undirected")
   if combine not in ("sum", "mean", "max"):
      raise ValueError(f"unknown combine: {combine}")
   if disagree not in ("positive", "negative", "zero"):
      raise ValueError(f"unknown disagree: {disagree}")
   d = {"positive": 1, "negative": -1, "zero": 0}[disagree]
   ml = net.layer_labels is not None
   inter = None
   if ml:
      edges, inter, nodes = net.to_pandas_dataframes()
      edges = _merge(edges, ["source", LAYER], ["target", LAYER], combine, sign, d)
      inter = _merge(inter, ["source", "source_layer"], ["target", "target_layer"], combine, sign, d)
   else:
      edges, _, nodes = _tables(net)
      edges = _merge(edges, ["source"], ["target"], combine, sign, d)
   return _reload(net, edges, inter, nodes, directed = False)


def remove_isolates(net):
   """
   Network without its isolated nodes.

   An isolated node has no edges at all; a node whose only edge is a self loop is not isolated.
   Since every isolated node is a component on its own, the other components are kept unchanged,
   in the same order, with the same node and edge attributes, node types and device; only their
   positions in the list of components can shift (isolated nodes are among the smallest
   components, so the largest ones keep their index). Their tensors are copied, so changing one
   network does not change the other.

   In multilayer networks a copy is isolated if it has no edges, couplings and other interlayer
   edges included; a copy coupled to the other copies of its entity is not isolated.

   :param net: A :class:`~graphoskop.Network`.
   :return: The network without isolated nodes, a new :class:`~graphoskop.Network` (with the same
      nodes, copied, if there are none).

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")   # nodes.tsv lists nodes without edges too
      core = gsk.remove_isolates(net)
      core.number_of_nodes() <= net.number_of_nodes()     # True
   """
   keep = [c for c, d in enumerate(net.components) if d.num_nodes > 1 or d.edge_index.shape[1] > 0]
   pos = np.concatenate([np.arange(net.node_offsets[c], net.node_offsets[c + 1]) for c in keep] + [np.zeros(0, dtype = np.int64)])
   Q = net.qualitative_node_attr
   if net.node_format == "wide":
      Q = Q.iloc[pos].reset_index(drop = True)
   else:   # long: positions in the network wide node order, renumbered
      new = np.full(int(net.node_offsets[-1]), -1)
      new[pos] = np.arange(len(pos))
      Q = Q[(new[Q["node"].to_numpy()] >= 0)].reset_index(drop = True)
      Q["node"] = new[Q["node"].to_numpy()]
   node_ids = [net.node_ids[c] for c in keep]
   node_map = {x: (k, i) for k, ids in enumerate(node_ids) for i, x in enumerate(ids.tolist())}
   split = None if net.bipartite_split is None else [net.bipartite_split[c] for c in keep]
   QE = None
   if net.qualitative_edge_attributes:
      eo = net._edge_offsets()
      QE = _take_edges(net.qualitative_edge_attr, np.concatenate([np.arange(eo[c], eo[c + 1]) for c in keep] + [np.zeros(0, dtype = np.int64)]))
   return Network([net.components[c].clone() for c in keep], node_map, node_ids, list(net.node_attributes), list(net.edge_attributes), net.directed, Q, list(net.qualitative_node_attributes), net.node_format, net.bipartite_labels, split, net.layer_labels, net.entity_ids, net.qualitative_edge_attributes, QE)


def remove_self_loops(net):
   """
   Network without its self loops (edges from a node to itself).

   All nodes are kept, so a node whose only edges were self loops becomes isolated (follow with
   :func:`remove_isolates` to drop it as well). Removing self loops never splits a component, so
   components, internal ids ``(component, index)``, node attributes, node types and device stay
   the same, and so do the attributes of the other edges. Tensors are copied, so changing one
   network does not change the other. In multilayer networks self loops are intra-layer edges
   from a copy to itself.

   :param net: A :class:`~graphoskop.Network`.
   :return: The network without self loops, a new :class:`~graphoskop.Network`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("replies.tsv", directed = True)   # people replying to themselves
      clean = gsk.remove_isolates(gsk.remove_self_loops(net))
   """
   edges, rows = [], []
   eo = net._edge_offsets()
   for c, d in enumerate(net.components):
      keep = d.edge_index[0] != d.edge_index[1]   # undirected: reverse copies are never self loops, so the input edges stay first
      edges.append((d.edge_index[:, keep], d.edge_attr[keep]))
      rows.append(eo[c] + np.flatnonzero(keep[:eo[c + 1] - eo[c]].cpu().numpy()))
   QE = _take_edges(net.qualitative_edge_attr, np.concatenate(rows)) if net.qualitative_edge_attributes else None
   return _with_edges(net, edges, QE)


def remove_parallel_edges(net, combine = "sum", sign = SIGN, disagree = "zero"):
   """
   Network without parallel edges: all the edges between the same two nodes become one edge,
   whose attributes combine theirs.

   In directed networks, parallel edges have the same source and the same target: the edges
   :math:`u \\to v` and :math:`v \\to u` are not parallel and both stay (use :func:`to_undirected`
   to merge them). In undirected networks, :math:`u - v` and :math:`v - u` are the same edge. Self
   loops of a node are parallel to each other. In multilayer networks the endpoints are copies
   ``(entity, layer)``, so only edges within the same layer are merged (and interlayer edges
   between the same two copies), never edges of different layers.

   Each numeric edge attribute of a merged edge combines the values of its edges, ignoring NaN, as
   in :func:`to_undirected`: ``"sum"`` (weights then give the same adjacency matrix and Laplacians),
   ``"mean"`` or ``"max"``; each qualitative one keeps their distinct values (one value, or a tuple
   of them). The sign attribute (see :doc:`signed`) keeps the common sign of the
   edges, and ``disagree`` decides when they have both signs: ``"positive"``, ``"negative"`` or
   ``"zero"``. A merged edge keeps the position and orientation of the first of its edges.

   Merging parallel edges never changes which nodes are connected, so nodes, components, internal
   ids, node attributes, node types and device are kept, and all the work runs on the device of
   the network. Tensors are copied, so changing one network does not change the other.

   :param net: A :class:`~graphoskop.Network`.
   :param combine: ``"sum"``, ``"mean"`` or ``"max"``.
   :param sign: Name of the sign edge attribute (ignored if the network has no such attribute).
   :param disagree: ``"positive"``, ``"negative"`` or ``"zero"``.
   :return: The network without parallel edges, a new :class:`~graphoskop.Network`.
   :raises ValueError: If ``combine`` or ``disagree`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("emails.tsv", directed = True)    # one row per email
      simple = gsk.remove_parallel_edges(net)                   # one edge per sender and receiver, weights summed
      simple = gsk.remove_parallel_edges(net, combine = "max")
   """
   if combine not in ("sum", "mean", "max"):
      raise ValueError(f"unknown combine: {combine}")
   if disagree not in ("positive", "negative", "zero"):
      raise ValueError(f"unknown disagree: {disagree}")
   d_sign = {"positive": 1., "negative": -1., "zero": 0.}[disagree]
   j = net.edge_attributes.index(sign) if sign in net.edge_attributes else None
   eo = net._edge_offsets()
   qual = {a: [] for a in net.qualitative_edge_attributes}
   edges, merged = [], 0
   for c, d in enumerate(net.components):
      ei, X = net._input_edges(c)   # undirected: one copy of each edge
      s, t = ei
      n, dev = d.num_nodes, s.device
      key = s * n + t if net.directed else torch.minimum(s, t) * n + torch.maximum(s, t)
      _, g = torch.unique(key, return_inverse = True)
      k = int(g.max()) + 1 if g.shape[0] else 0
      pos = torch.arange(g.shape[0], device = dev)
      first = torch.full((k,), g.shape[0], device = dev).scatter_reduce(0, g, pos, "amin")
      order = torch.argsort(first)   # merged edges in order of their first edge
      rank = torch.empty_like(order)
      rank[order] = torch.arange(k, device = dev)
      g, first = rank[g], first[order]
      x = X.double()
      ok = ~torch.isnan(x)
      cnt = torch.zeros((k, x.shape[1]), dtype = torch.float64, device = dev).index_add_(0, g, ok.double())
      if combine == "max":
         y = torch.full((k, x.shape[1]), -torch.inf, dtype = torch.float64, device = dev)
         y = y.scatter_reduce(0, g[:, None].expand_as(x), torch.where(ok, x, -torch.inf), "amax")
      else:
         y = torch.zeros((k, x.shape[1]), dtype = torch.float64, device = dev).index_add_(0, g, torch.where(ok, x, 0))
         if combine == "mean":
            y = y / cnt.clamp(min = 1)
      y = torch.where(cnt > 0, y, torch.nan)   # all NaN stays NaN
      if j is not None:
         v = x[:, j]
         p = torch.zeros(k, dtype = torch.bool, device = dev).index_put_((g,), v > 0, accumulate = True)
         q = torch.zeros(k, dtype = torch.bool, device = dev).index_put_((g,), v < 0, accumulate = True)
         y[:, j] = torch.where(p & q, d_sign, torch.where(p, 1., torch.where(q, -1., y[:, j])))
      y = y.to(X.dtype)
      gn = g.cpu().numpy()
      merged += k
      for a in qual:   # qualitative: the distinct values of the merged edges
         v = pd.Series(np.asarray(net.qualitative_edge_attr[a].iloc[eo[c]:eo[c + 1]], dtype = object))
         qual[a].append(v.groupby(gn, sort = True).agg(_distinct).to_numpy())
      s, t = s[first], t[first]
      if not net.directed:   # stored again in both directions
         loop = s == t
         s, t, y = torch.cat([s, t[~loop]]), torch.cat([t, s[~loop]]), torch.cat([y, y[~loop]])
      edges.append((torch.stack([s, t]), y))
   QE = _edge_frame({a: np.concatenate(v + [np.zeros(0, dtype = object)]) for a, v in qual.items()}, merged) if qual else None
   return _with_edges(net, edges, QE)


def _tables(net):
   """
   Edge and node tables of a network, which :func:`_reload` turns back into a network. The node
   table declares every node, so that nodes without edges and without values are kept.

   :param net: Network.
   :return: ``(edges, interlayer, nodes)``, with ``interlayer`` ``None`` if the network is not
      multilayer.
   """
   if net.layer_labels is not None:
      return net.to_pandas_dataframes()
   E, N = _flat_tables(net)
   if net.node_format == "long":   # nodes without values are not in the long table
      N = pd.concat([N, pd.DataFrame({"node": np.concatenate(net.node_ids), "attribute": np.nan, "value": np.nan})], ignore_index = True)
   return E, None, N


def _reload(net, edges, inter, nodes, directed = None):
   """
   Network from tables as given by :func:`_tables`, with the settings of another network. Layers
   not in the layer order of ``net`` are appended to it, in order of appearance.

   :param net: Network whose settings are kept.
   :param edges: Edge table.
   :param inter: Interlayer edge table, or ``None``.
   :param nodes: Node table.
   :param directed: Whether the new network is directed, ``None`` for the same as ``net``.
   :return: New :class:`~graphoskop.Network`.
   """
   ml = net.layer_labels is not None
   layers = None
   if ml:
      seen = pd.concat([edges[LAYER], inter["source_layer"], inter["target_layer"], nodes[LAYER]], ignore_index = True)
      layers = list(net.layer_labels) + [x for x in pd.unique(seen.dropna()) if x not in net.layer_labels]
   new = _load(edges, nodes, source = "source", target = "target", directed = net.directed if directed is None else directed, node_format = net.node_format, bipartite = net.bipartite_labels is not None, device = net.components[0].edge_index.device, multilayer = ml, interlayer_edges = inter, layers = layers)
   _release_memory()
   return new


def _restrict(net, nodes, edges):
   """
   Network with some nodes and edges of another, attributes kept, components recomputed.

   :param net: Network.
   :param nodes: Per component, boolean NumPy array of the nodes to keep.
   :param edges: Per component, boolean NumPy array of the stored edges to keep (only edges between
      kept nodes should be kept).
   :return: New :class:`~graphoskop.Network`.
   """
   ml = net.layer_labels is not None
   keep, keep_inter = [], []
   for c in range(len(net.components)):
      m = net._input_edges(c)[0].shape[1]   # tables list each input edge once
      k = edges[c][:m]
      if ml:
         inter = net.interlayer_mask(c)[:m].cpu().numpy()
         keep.append(k[~inter])
         keep_inter.append(k[inter])
      else:
         keep.append(k)
   E, inter, N = _tables(net)
   if ml:
      inter = inter[np.concatenate(keep_inter)]
   E = E[np.concatenate(keep)]
   if net.node_format == "wide":   # one row per node, in network wide order
      N = N[np.concatenate(nodes)]
   else:
      ids = np.concatenate([net.node_ids[c][nodes[c]] for c in range(len(net.components))])
      key = list(zip(N["node"], N[LAYER])) if ml else N["node"]
      N = N[pd.Series(key).isin(set(ids.tolist()) if ml else ids).to_numpy()]
   return _reload(net, E, inter, N)


def subgraph(net, nodes):
   """
   Subgraph induced by a set of nodes: those nodes, with their attributes, and the edges between
   them, with theirs.

   Components are recomputed, so they are those of the subgraph; node types (bipartite networks),
   node table format, directedness and device are kept. In multilayer networks nodes are copies
   ``(entity, layer)``, and an entity stands for all its copies; interlayer edges (couplings
   included) between kept copies are kept.

   :param net: A :class:`~graphoskop.Network`.
   :param nodes: Iterable of nodes (input ids, internal ids ``(component, index)``, or entities in
      multilayer networks), for example one of the arrays of :func:`connected_components_nodes`.
   :return: The subgraph, a new :class:`~graphoskop.Network`.
   :raises KeyError: If a node does not exist.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      ego = gsk.subgraph(net, ["alice", "bob", "carol"])
   """
   keep = [np.zeros(d.num_nodes, dtype = bool) for d in net.components]
   for x in nodes:
      for c, i in net._nodes(x):
         keep[c][i] = True
   edges = []
   for c, d in enumerate(net.components):
      s, t = d.edge_index.cpu().numpy()
      edges.append(keep[c][s] & keep[c][t])
   return _restrict(net, keep, edges)


def ego_network(net, ego, include_ego = True, direction = "out"):
   """
   Ego network: the subgraph induced by a node (the ego) and its neighbors, with all the edges
   among them; with ``include_ego = False``, only the neighbors and the edges among them, the ego
   and its edges being removed.

   Attributes, node types, node table format, directedness and device are kept, and components
   are recomputed, as in :func:`subgraph` (without the ego, the neighbors can fall apart in several
   components). In multilayer networks the neighbors are those in the supra-graph (couplings
   included), and an entity given as ego stands for all its copies, which are all egos.

   :param net: A :class:`~graphoskop.Network`.
   :param ego: The ego (input id, internal id ``(component, index)``, or entity in multilayer
      networks).
   :param include_ego: Whether to keep the ego and its edges.
   :param direction: In directed networks, the neighbors: ``"out"`` (nodes the ego points to),
      ``"in"`` (nodes pointing to the ego) or ``"both"``; ignored otherwise.
   :return: The ego network, a new :class:`~graphoskop.Network`.
   :raises KeyError: If there is no such node.
   :raises ValueError: If ``direction`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      ego = gsk.ego_network(net, "alice")                         # alice, her friends, their edges
      alters = gsk.ego_network(net, "alice", include_ego = False) # how her friends know each other
   """
   if direction not in ("out", "in", "both"):
      raise ValueError(f"unknown direction: {direction}")
   egos = net._nodes(ego)
   keep = [np.zeros(d.num_nodes, dtype = bool) for d in net.components]
   for c, i in egos:
      s, t = net.components[c].edge_index.cpu().numpy()
      if not net.directed or direction in ("out", "both"):
         keep[c][t[s == i]] = True
      if net.directed and direction in ("in", "both"):
         keep[c][s[t == i]] = True
   for c, i in egos:
      keep[c][i] = include_ego
   edges = []
   for c, d in enumerate(net.components):
      s, t = d.edge_index.cpu().numpy()
      edges.append(keep[c][s] & keep[c][t])
   return _restrict(net, keep, edges)


def to_strongly_connected(net):
   """
   Directed network split into its strongly connected components: the edges between different
   strongly connected components are dropped, so that each of them becomes a component of its own
   (see :func:`~graphoskop.connected_components_count`).

   All nodes and their attributes are kept (a node in no directed cycle becomes isolated), and so
   are the attributes of the remaining edges, node types, node table format and device. In
   multilayer networks the strongly connected components are those of the supra-graph.

   :param net: A directed :class:`~graphoskop.Network`.
   :return: The network split into strongly connected components, a new
      :class:`~graphoskop.Network`; its components are sorted by size as usual.
   :raises ValueError: If ``net`` is undirected (its components are already strongly connected).

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("follows.tsv", directed = True)
      scc = gsk.to_strongly_connected(net)
      scc.number_of_components() == gsk.connected_components_count(net, type = "strong")   # True
   """
   if not net.directed:
      raise ValueError("the network is undirected: its components are already strongly connected")
   nodes, edges = [], []
   for c, d in enumerate(net.components):
      labels = _strong_labels(net, c)[1]
      s, t = d.edge_index.cpu().numpy()
      nodes.append(np.ones(d.num_nodes, dtype = bool))
      edges.append(labels[s] == labels[t])
   return _restrict(net, nodes, edges)


def _items(x):
   """
   Items given one at a time or as a collection. A string or a tuple is always one item (a node
   id, a copy ``(entity, layer)`` or an edge); lists, sets, arrays, :class:`pandas.Series` and
   other iterables are collections.

   :param x: One item, or an iterable of items.
   :return: List of items.
   """
   if isinstance(x, (str, bytes, tuple)) or not np.iterable(x):
      return [x.item() if isinstance(x, np.generic) else x]
   x = x.tolist() if isinstance(x, (np.ndarray, pd.Series, pd.Index)) else x   # NumPy values to Python values
   return [tuple(e) if isinstance(e, list) else e for e in x]   # rows of a 2D array: edges


def _id(net, x):
   """
   Node id as stored in a network: ids read from files are strings, so an id that is not in the
   network is replaced by its string if that is (the entity only, for copies ``(entity, layer)``).

   :param net: Network.
   :param x: Node id, or copy ``(entity, layer)`` in multilayer networks.
   :return: The id.
   """
   if net.layer_labels is None:
      return str(x) if x not in net.node_map and str(x) in net.node_map else x
   e, l = x
   return (str(e) if e not in net.entity_map and str(e) in net.entity_map else e, l)


def _endpoints(net, edges):
   """
   Endpoints and attributes of edges given one at a time, as an iterable or as a table.

   :param net: Network.
   :param edges: An edge ``(u, v)``, an iterable of them, or a :class:`pandas.DataFrame` (see
      :func:`add_edges`).
   :return: List of the endpoint pairs ``(u, v)``, and table of the edge attributes, one row per
      edge (no columns if there are none).
   :raises ValueError: If an edge is not a pair, or if a multilayer table has neither layer
      columns nor two endpoint columns.
   """
   if not isinstance(edges, pd.DataFrame):
      pairs = [tuple(e) for e in _items(edges)]
      if any(len(e) != 2 for e in pairs):
         raise ValueError("an edge is a pair (u, v)")
      return pairs, pd.DataFrame(index = pd.RangeIndex(len(pairs)))
   cols = list(edges.columns)
   col = lambda c: edges[c].tolist()
   if net.layer_labels is None:
      ids = cols[:2]
      pairs = list(zip(col(cols[0]), col(cols[1])))
   elif "source_layer" in cols and "target_layer" in cols:   # interlayer table: source, source_layer, target, target_layer
      ids = cols[:4]
      pairs = list(zip(zip(col(cols[0]), col(cols[1])), zip(col(cols[2]), col(cols[3]))))
   elif LAYER in cols:   # edge table: source, target, layer
      ids = cols[:2] + [LAYER]
      pairs = list(zip(zip(col(cols[0]), col(LAYER)), zip(col(cols[1]), col(LAYER))))
   else:   # entities
      ids = cols[:2]
      pairs = list(zip(col(cols[0]), col(cols[1])))
   return pairs, edges.drop(columns = ids).reset_index(drop = True)


def add_nodes(net, nodes):
   """
   Network with new nodes, without edges (use :func:`add_edges` to connect them).

   ``nodes`` is one node id, an iterable of node ids, or a :class:`pandas.DataFrame` node table in
   the format of the network's (``net.node_format``), as taken by :func:`load_network`: wide (node
   id, then one column per attribute) or long (node id, attribute name, value). A string or a
   tuple is always one node. New nodes get NaN (missing) for every attribute they are not given;
   in long (sparse) node tables, a numeric attribute without a value reads as 0. Columns that are
   not attributes of the network become new attributes, NaN for the existing nodes.

   In multilayer networks every new node is a copy ``(entity, layer)``: tuples, or a table with
   the entity in the first column and the layer in the second. Layers not in
   :attr:`~graphoskop.Network.layer_labels` are appended to the layer order. In bipartite networks
   new nodes must have a value of :data:`~graphoskop.BIPARTITE`, so they are given as a table.

   Every new node is a component of its own; the existing components, node and edge attributes,
   node types, node table format and device are kept.

   :param net: A :class:`~graphoskop.Network`.
   :param nodes: A node id, an iterable of node ids, or a :class:`pandas.DataFrame`.
   :return: The network with the new nodes, a new :class:`~graphoskop.Network`.
   :raises ValueError: If a node is already in the network, if a node of a multilayer network is
      not a copy ``(entity, layer)``, or if a new node of a bipartite network has no node type.

   Example:

   .. code-block:: python

      import graphoskop as gsk
      import pandas as pd

      net = gsk.load_network("edges.tsv", "nodes.tsv")            # attributes age, city
      net = gsk.add_nodes(net, "dave")                              # age NaN, city NaN
      net = gsk.add_nodes(net, ["erin", "frank"])
      net = gsk.add_nodes(net, pd.DataFrame({"id": ["gina"], "age": [41]}))   # city NaN
   """
   ml = net.layer_labels is not None
   k = 2 if ml else 1
   ids_names = ["node", LAYER][:k]
   if isinstance(nodes, pd.DataFrame):
      new = nodes.reset_index(drop = True)
      names = ids_names + (["attribute", "value"] if net.node_format == "long" else list(new.columns[k:]))
      new = new.set_axis(names, axis = 1)
      new = new.astype({c: float for c in new.columns[k:] if new[c].dtype == bool})   # stays numeric next to NaN
      keys = list(zip(new["node"].tolist(), new[LAYER].tolist())) if ml else new["node"].tolist()
   else:
      keys = _items(nodes)
      new = None
   if ml and any(not isinstance(x, tuple) or len(x) != 2 or pd.isna(x[1]) for x in keys):
      raise ValueError("the nodes of a multilayer network are copies (entity, layer)")
   keys = [_id(net, x) for x in keys]
   old = [x for x in keys if x in net.node_map]
   if old:
      raise ValueError(f"node {old[0]!r} is already in the network")
   cols = {"node": [x[0] for x in keys], LAYER: [x[1] for x in keys]} if ml else {"node": keys}
   if new is None:
      new = pd.DataFrame(cols)
      if net.node_format == "long":
         new["attribute"], new["value"] = np.nan, np.nan
   else:
      for c, v in cols.items():
         new[c] = v
   E, inter, N = _tables(net)
   return _reload(net, E, inter, pd.concat([N, new], ignore_index = True))


def add_edges(net, edges, weight = "weight"):
   """
   Network with new edges.

   ``edges`` is one edge ``(u, v)``, an iterable of them (e.g. a list of pairs, or a two column
   array), or a :class:`pandas.DataFrame` edge table: source and target in the first two columns,
   then one column per edge attribute (non-numeric columns are ignored, as in
   :func:`load_network`). A tuple is always one edge. New edges get NaN for every attribute they
   are not given, except ``weight``, which defaults to 1: if the network or the new edges have an
   attribute named ``weight``, new edges without a value get 1, and so do the existing edges if
   the attribute is new (they were unweighted). Columns that are not attributes of the network
   become new attributes, NaN for the existing edges.

   Endpoints not in the network are added as new nodes, with NaN (missing) for every node
   attribute, as in :func:`add_nodes`. Adding an edge that already exists adds a parallel edge
   (see :func:`remove_parallel_edges`). In undirected networks ``(u, v)`` and ``(v, u)`` are the
   same edge.

   In multilayer networks the endpoints are copies ``(entity, layer)``: an edge between copies in
   the same layer is an intra-layer edge, otherwise an interlayer edge (a coupling if they are
   copies of the same entity). A table is given as an edge table (source, target, a ``layer``
   column, then attributes) or as an interlayer table (source, ``source_layer``, target,
   ``target_layer``, then attributes), as written by
   :meth:`~graphoskop.Network.to_pandas_dataframes`. Layers not in
   :attr:`~graphoskop.Network.layer_labels` are appended to the layer order. In bipartite networks
   new edges must join nodes of different types, and new endpoints need a type, so they must be
   added first with :func:`add_nodes`.

   Components are recomputed, since new edges can join them; node and edge attributes, node types,
   node table format and device are kept.

   :param net: A :class:`~graphoskop.Network`.
   :param edges: An edge ``(u, v)``, an iterable of edges, or a :class:`pandas.DataFrame`.
   :param weight: Name of the edge attribute holding the weights, or ``None`` to give no default
      weight.
   :return: The network with the new edges, a new :class:`~graphoskop.Network`.
   :raises ValueError: If an edge is not a pair, if an endpoint of a multilayer network is not a
      copy ``(entity, layer)``, or, in bipartite networks, if an edge joins two nodes of the same
      type or a new endpoint has no node type.

   Example:

   .. code-block:: python

      import graphoskop as gsk
      import pandas as pd

      net = gsk.load_network("edges.tsv")                           # attribute weight
      net = gsk.add_edges(net, ("alice", "dave"))                    # weight 1, dave is a new node
      net = gsk.add_edges(net, [("bob", "carol"), ("carol", "dave")])
      net = gsk.add_edges(net, pd.DataFrame({"source": ["alice"], "target": ["erin"], "weight": [2.5]}))
   """
   ml = net.layer_labels is not None
   pairs, new = _endpoints(net, edges)
   if ml and any(not isinstance(x, tuple) or len(x) != 2 for e in pairs for x in e):
      raise ValueError("the edges of a multilayer network join copies (entity, layer)")
   u = [_id(net, x) for x, _ in pairs]
   v = [_id(net, y) for _, y in pairs]
   E, inter, N = _tables(net)
   if weight is not None and (weight in net.edge_attributes or weight in new.columns):
      new[weight] = pd.to_numeric(new[weight]).fillna(1.) if weight in new.columns else 1.
      if weight not in net.edge_attributes:
         for T in (E, inter):
            if T is not None:
               T[weight] = 1.
   if not ml:
      new.insert(0, "target", v)
      new.insert(0, "source", u)
      return _reload(net, pd.concat([E, new], ignore_index = True), None, N)
   same = np.array([a[1] == b[1] for a, b in zip(u, v)], dtype = bool)
   intra, cross = new[same].copy(), new[~same].copy()
   for T, m in ((intra, same), (cross, ~same)):
      a = [x for x, k in zip(u, m) if k]
      b = [y for y, k in zip(v, m) if k]
      T.insert(0, "source", [x[0] for x in a])
      if T is intra:
         T.insert(1, "target", [y[0] for y in b])
         T.insert(2, LAYER, [x[1] for x in a])
      else:
         T.insert(1, "source_layer", [x[1] for x in a])
         T.insert(2, "target", [y[0] for y in b])
         T.insert(3, "target_layer", [y[1] for y in b])
   return _reload(net, pd.concat([E, intra], ignore_index = True), pd.concat([inter, cross], ignore_index = True), N)


def remove_nodes(net, nodes):
   """
   Network without some nodes and their edges: the subgraph induced by all the other nodes (see
   :func:`subgraph`).

   ``nodes`` is one node, an iterable of nodes, or a :class:`pandas.DataFrame` whose first column
   gives the nodes (e.g. a node table); a string or a tuple is always one node. Nodes are input
   ids or internal ids ``(component, index)``; in multilayer networks, copies ``(entity, layer)``
   or entities, which stand for all their copies (in a table, the entity is in the first column
   and the layer in the second, missing for all the copies).

   Components are recomputed, since removing a node can split one; the attributes of the other
   nodes and edges, node types, node table format and device are kept.

   :param net: A :class:`~graphoskop.Network`.
   :param nodes: A node, an iterable of nodes, or a :class:`pandas.DataFrame`.
   :return: The network without the nodes, a new :class:`~graphoskop.Network`.
   :raises KeyError: If a node is not in the network.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      net = gsk.remove_nodes(net, "alice")
      net = gsk.remove_nodes(net, ["bob", "carol"])
   """
   if isinstance(nodes, pd.DataFrame):
      ids = nodes.iloc[:, 0].tolist()
      if net.layer_labels is not None and nodes.shape[1] > 1:
         ids = [e if pd.isna(l) else (e, l) for e, l in zip(ids, nodes.iloc[:, 1].tolist())]
      nodes = ids
   keep = [np.ones(d.num_nodes, dtype = bool) for d in net.components]
   for x in _items(nodes):
      for c, i in net._nodes(x):
         keep[c][i] = False
   edges = []
   for c, d in enumerate(net.components):
      s, t = d.edge_index.cpu().numpy()
      edges.append(keep[c][s] & keep[c][t])
   return _restrict(net, keep, edges)


def remove_edges(net, edges):
   """
   Network without some edges.

   ``edges`` is one edge ``(u, v)``, an iterable of them, or a :class:`pandas.DataFrame` with the
   endpoints in the first two columns (e.g. an edge table; other columns are ignored). A tuple is
   always one edge. All the edges from ``u`` to ``v`` are removed, parallel edges included; in
   undirected networks, also those from ``v`` to ``u``. Endpoints are given as in
   :func:`remove_nodes`. In multilayer networks, an entity stands for all its copies, so
   ``(u, v)`` between two entities removes the edges between their copies in every layer and
   across layers; a table with a ``layer`` column, or with ``source_layer`` and ``target_layer``
   columns, gives copies, as written by :meth:`~graphoskop.Network.to_pandas_dataframes`.

   All nodes are kept, so a node can become isolated (follow with :func:`remove_isolates` to drop
   it). Components are recomputed, since removing an edge can split one; the attributes of the
   nodes and of the other edges, node types, node table format and device are kept.

   :param net: A :class:`~graphoskop.Network`.
   :param edges: An edge ``(u, v)``, an iterable of edges, or a :class:`pandas.DataFrame`.
   :return: The network without the edges, a new :class:`~graphoskop.Network`.
   :raises KeyError: If a node or an edge is not in the network.
   :raises ValueError: If an edge is not a pair.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      net = gsk.remove_edges(net, ("alice", "bob"))
      net = gsk.remove_edges(net, [("bob", "carol"), ("carol", "dave")])
   """
   pairs = _endpoints(net, edges)[0]
   req = [(r, c, i, j) for r, (u, v) in enumerate(pairs) for c, i in net._nodes(u) for c2, j in net._nodes(v) if c == c2]
   req = np.array(req, dtype = np.int64).reshape(-1, 4)
   found = np.zeros(len(pairs), dtype = bool)
   nodes, keep = [], []
   for c, d in enumerate(net.components):
      s, t = d.edge_index.cpu().numpy()
      n = d.num_nodes
      q = req[req[:, 1] == c]
      key = q[:, 2] * n + q[:, 3]
      drop = np.isin(s * n + t, key)
      if not net.directed:   # both directions are stored
         drop |= np.isin(t * n + s, key)
      found[q[np.isin(key, s * n + t), 0]] = True
      nodes.append(np.ones(n, dtype = bool))
      keep.append(~drop)
   if not found.all():
      raise KeyError(f"no edge {pairs[np.argmin(found)]!r}")
   return _restrict(net, nodes, keep)


def flip_edge_direction(net, edges = None):
   """
   Directed network with some or all of its edges reversed: an edge from ``a`` to ``b`` becomes
   an edge from ``b`` to ``a``, with the same attributes.

   Without ``edges``, all the edges are reversed (the transpose of the adjacency matrix). Otherwise
   ``edges`` is one edge ``(u, v)``, an iterable of them, or a :class:`pandas.DataFrame` with the
   endpoints in the first two columns, as in :func:`remove_edges`: all the edges from ``u`` to
   ``v`` are reversed, parallel edges included, and the edges from ``v`` to ``u``, if any, are not.
   In multilayer networks, an entity stands for all its copies, and a table with layer columns
   gives copies, as in :func:`remove_edges`.

   Reversing edges never changes the weakly connected components, so components, internal ids
   ``(component, index)``, node attributes, node types and device stay the same, and so does the
   order of the edges. Tensors are copied, so changing one network does not change the other.

   :param net: A directed :class:`~graphoskop.Network`.
   :param edges: ``None`` for all the edges, or an edge ``(u, v)``, an iterable of edges, or a
      :class:`pandas.DataFrame`.
   :return: The network with the edges reversed, a new :class:`~graphoskop.Network`.
   :raises ValueError: If the network is undirected, or an edge is not a pair.
   :raises KeyError: If a node or an edge is not in the network.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("follows.tsv", directed = True)   # a -> b: a follows b
      followed = gsk.flip_edge_direction(net)                 # b -> a: b is followed by a
      net = gsk.flip_edge_direction(net, ("alice", "bob"))    # now bob -> alice
      net = gsk.flip_edge_direction(net, [("bob", "carol"), ("carol", "dave")])
   """
   if not net.directed:
      raise ValueError("flip_edge_direction needs a directed network: undirected edges have no direction")
   if edges is not None:
      pairs = _endpoints(net, edges)[0]
      req = [(r, c, i, j) for r, (u, v) in enumerate(pairs) for c, i in net._nodes(u) for c2, j in net._nodes(v) if c == c2]
      req = np.array(req, dtype = np.int64).reshape(-1, 4)
      found = np.zeros(len(pairs), dtype = bool)
   out = []
   for c, d in enumerate(net.components):
      ei = d.edge_index
      if edges is None:
         flip = torch.ones(ei.shape[1], dtype = torch.bool, device = ei.device)
      else:
         s, t = ei.cpu().numpy()
         n = d.num_nodes
         q = req[req[:, 1] == c]
         key = q[:, 2] * n + q[:, 3]
         flip = np.isin(s * n + t, key)
         found[q[np.isin(key, s * n + t), 0]] = True
         flip = torch.as_tensor(flip, device = ei.device)
      out.append((torch.where(flip, ei.flip(0), ei), d.edge_attr.clone()))
   if edges is not None and not found.all():
      raise KeyError(f"no edge {pairs[np.argmin(found)]!r}")
   return _with_edges(net, out)


def _check_pair(net, other, bipartite = True):
   """
   Check that two networks can be combined edge by edge.

   :param net: First network.
   :param other: Second network.
   :param bipartite: Also require both or neither to be bipartite.
   :raises ValueError: If one is directed and the other not, one multilayer and the other not, or
      (with ``bipartite``) one bipartite and the other not.
   """
   if net.directed != other.directed:
      raise ValueError("one network is directed and the other is not")
   if (net.layer_labels is None) != (other.layer_labels is None):
      raise ValueError("one network is multilayer and the other is not")
   if bipartite and (net.bipartite_labels is None) != (other.bipartite_labels is None):
      raise ValueError("one network is bipartite and the other is not")


def _edge_keys(net, tables):
   """
   Integer keys of the edges of several tables, equal for edges joining the same two nodes (in
   either direction if the networks are undirected).

   :param net: Network, for its settings.
   :param tables: List of ``(edges, interlayer)`` table pairs, as given by :func:`_tables`.
   :return: List of int64 arrays: for each pair, the keys of its edges, then (multilayer networks)
      of its interlayer edges.
   """
   ends = []
   for E, I in tables:
      if net.layer_labels is None:
         ends.append((E["source"].tolist(), E["target"].tolist()))
      else:
         ends.append((list(zip(E["source"], E[LAYER])), list(zip(E["target"], E[LAYER]))))
         ends.append((list(zip(I["source"], I["source_layer"])), list(zip(I["target"], I["target_layer"]))))
   codes = pd.factorize(pd.Series([x for u, v in ends for x in u + v], dtype = object))[0]
   M, keys, at = max(int(codes.max()) + 1, 1) if len(codes) else 1, [], 0
   for u, v in ends:
      a, b = codes[at:at + len(u)], codes[at + len(u):at + 2 * len(u)]
      at += 2 * len(u)
      if not net.directed:
         a, b = np.minimum(a, b), np.maximum(a, b)
      keys.append(a.astype(np.int64) * M + b)
   return keys


def _node_table_as(net, other):
   """
   Node table of a network in the format of another, as given by :func:`_tables`.

   :param net: Network whose node table format is used.
   :param other: Network whose node table is returned.
   :return: :class:`pandas.DataFrame`.
   """
   N = _tables(other)[2]
   if other.node_format == net.node_format:
      return N
   k = 1 if net.layer_labels is None else 2
   W = _wide_rows(other, np.arange(other.node_offsets[-1]))
   if net.node_format == "wide":
      return W
   return pd.concat([_to_long(W, k), W.iloc[:, :k].assign(attribute = np.nan, value = np.nan)], ignore_index = True)


def _merge_nodes(net, N1, other, both):
   """
   Node table of the nodes of two networks, values of the first one first: the attributes a node
   has no value of in the first network come from the second.

   :param net: First network.
   :param N1: Its node table, as given by :func:`_tables`.
   :param other: Second network.
   :param both: Keep only the nodes of both networks, instead of the nodes of either.
   :return: :class:`pandas.DataFrame` in the format of the node table of ``net``.
   """
   N2 = _node_table_as(net, other)
   keys = ["node"] if net.layer_labels is None else ["node", LAYER]
   if both:
      common = pd.MultiIndex.from_frame(N1[keys]).intersection(pd.MultiIndex.from_frame(N2[keys]))
      N1, N2 = (N[pd.MultiIndex.from_frame(N[keys]).isin(common)] for N in (N1, N2))
   if net.node_format == "wide":
      A, B = N1.set_index(keys), N2.set_index(keys)
      A, B = (X.astype({c: object for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)}) for X in (A, B))
      cols = list(A.columns) + [c for c in B.columns if c not in A.columns]
      return A.combine_first(B)[cols].reset_index()
   pk = keys + ["attribute"]
   seen = N1[pk][N1["attribute"].notna().to_numpy()].drop_duplicates()
   new = N2[pk].merge(seen, how = "left", indicator = True)["_merge"].to_numpy() == "left_only"
   return pd.concat([N1, N2[new]], ignore_index = True)


def _edge_sets(net, other):
   """
   Tables of two networks and the keys of their edges (see :func:`_edge_keys`).

   :param net: First network.
   :param other: Second network.
   :return: ``(E1, I1, N1), (E2, I2), keys1, keys2``: tables of ``net``, edge tables of
      ``other``, and the keys of the edges of each, a list of arrays (edges, then interlayer
      edges in multilayer networks).
   """
   E1, I1, N1 = _tables(net)
   E2, I2, _ = _tables(other)
   keys = _edge_keys(net, [(E1, I1), (E2, I2)])
   h = len(keys) // 2
   return (E1, I1, N1), (E2, I2), keys[:h], keys[h:]


def network_union(net, other):
   """
   Union of two networks: the nodes and the edges of either.

   The two networks are assumed to use the same node ids for the same nodes. Two edges are the
   same if they join the same two nodes (in either direction, in undirected networks; in
   multilayer networks, the same two copies ``(entity, layer)``). The union has all the edges of
   ``net``, as they are (parallel edges included), and the edges of ``other`` joining two nodes
   that ``net`` does not join. Each edge keeps the attributes of its network; attributes of the
   other network are NaN, except ``"weight"``, which is 1 for the edges of a network without it,
   as in :func:`add_edges`. To add up the weights of the edges in both, use :func:`add_edges` with
   the edge table of ``other``, then :func:`remove_parallel_edges`.

   Nodes keep their attributes in ``net``; the values they do not have there come from ``other``,
   and nodes only in ``other`` keep theirs. In multilayer networks, layers only in ``other`` are
   appended to the layer order. Components are recomputed; node table format, node types and
   device are those of ``net``.

   :param net: A :class:`~graphoskop.Network`.
   :param other: A :class:`~graphoskop.Network`, directed if ``net`` is, multilayer if ``net`` is,
      and bipartite if ``net`` is.
   :return: The union, a new :class:`~graphoskop.Network`.
   :raises ValueError: If the networks differ in directedness, or one is multilayer or bipartite
      and the other is not.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      week1 = gsk.load_network("calls_week1.tsv")
      week2 = gsk.load_network("calls_week2.tsv")
      both_weeks = gsk.network_union(week1, week2)
   """
   _check_pair(net, other)
   (E1, I1, N1), (E2, I2), k1, k2 = _edge_sets(net, other)
   for A, B in ((E1, E2), (I1, I2), (E2, E1), (I2, I1)):   # weights default to 1, as in add_edges
      if A is not None and "weight" in B.columns and "weight" not in A.columns:
         A["weight"] = 1.
   seen = np.concatenate(k1)
   E = pd.concat([E1, E2[~np.isin(k2[0], seen)]], ignore_index = True)
   I = None if I1 is None else pd.concat([I1, I2[~np.isin(k2[1], seen)]], ignore_index = True)
   return _reload(net, E, I, _merge_nodes(net, N1, other, False))


def network_difference(net, other):
   """
   Difference of two networks: ``net`` without the edges of ``other``.

   The two networks are assumed to use the same node ids for the same nodes. An edge of ``net`` is
   removed, with its parallel edges, if ``other`` has an edge joining the same two nodes (in
   either direction, in undirected networks; in multilayer networks, the same two copies
   ``(entity, layer)``). All the nodes of ``net`` are kept, with their attributes, and so are the
   attributes of the remaining edges; ``other`` only gives the edges to remove (its nodes not in
   ``net`` are ignored). Components are recomputed.

   :param net: A :class:`~graphoskop.Network`.
   :param other: A :class:`~graphoskop.Network`, directed if ``net`` is, multilayer if ``net`` is.
   :return: The difference, a new :class:`~graphoskop.Network`.
   :raises ValueError: If the networks differ in directedness, or one is multilayer and the other
      is not.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      week1 = gsk.load_network("calls_week1.tsv")
      week2 = gsk.load_network("calls_week2.tsv")
      dropped = gsk.network_difference(week1, week2)   # calls of week 1 not made in week 2
   """
   _check_pair(net, other, bipartite = False)
   (E1, I1, N1), _, k1, k2 = _edge_sets(net, other)
   seen = np.concatenate(k2)
   return _reload(net, E1[~np.isin(k1[0], seen)], None if I1 is None else I1[~np.isin(k1[1], seen)], N1)


def network_intersection(net, other):
   """
   Intersection of two networks: the nodes of both, and the edges of ``net`` joining two nodes
   that ``other`` also joins.

   The two networks are assumed to use the same node ids for the same nodes. Two edges are the
   same if they join the same two nodes (in either direction, in undirected networks; in
   multilayer networks, the same two copies ``(entity, layer)``). The edges kept are those of
   ``net``, with their attributes (parallel edges included). Nodes keep their attributes in
   ``net``; the values they do not have there come from ``other``. Components are recomputed;
   node table format, node types and device are those of ``net``.

   :param net: A :class:`~graphoskop.Network`.
   :param other: A :class:`~graphoskop.Network`, directed if ``net`` is, multilayer if ``net`` is,
      and bipartite if ``net`` is.
   :return: The intersection, a new :class:`~graphoskop.Network`.
   :raises ValueError: If the networks differ in directedness, or one is multilayer or bipartite
      and the other is not.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      week1 = gsk.load_network("calls_week1.tsv")
      week2 = gsk.load_network("calls_week2.tsv")
      stable = gsk.network_intersection(week1, week2)   # pairs calling in both weeks
   """
   _check_pair(net, other)
   (E1, I1, N1), _, k1, k2 = _edge_sets(net, other)
   seen = np.concatenate(k2)
   return _reload(net, E1[np.isin(k1[0], seen)], None if I1 is None else I1[np.isin(k1[1], seen)], _merge_nodes(net, N1, other, True))


def _collapse_nodes(net, N, node_combine):
   """
   Node table with the rows of equal ids (merged nodes) combined.

   :param net: Network, for its attributes and node table format.
   :param N: Node table, as given by :func:`_tables`, with ids already mapped.
   :param node_combine: ``"sum"``, ``"mean"`` or ``"max"``, for numeric attributes.
   :return: :class:`pandas.DataFrame`.
   :raises ValueError: If a merged node of a bipartite network has both node types.
   """
   keys = ["node"] if net.layer_labels is None else ["node", LAYER]
   agg = lambda g: g.sum(min_count = 1) if node_combine == "sum" else g.agg(node_combine)
   if net.node_format == "wide":
      dup = N.duplicated(keys, keep = False).to_numpy()
      M = N[dup]
      M = M.astype({c: object for c in M.columns if isinstance(M[c].dtype, pd.CategoricalDtype)})
      g = M.groupby(keys, sort = False)
      R = agg(g[[c for c in N.columns if c in net.node_attributes]])
      for c in [c for c in N.columns if c in net.qualitative_node_attributes]:
         R[c] = g[c].agg(_distinct)
      N = pd.concat([N[~dup], R.reset_index()[list(N.columns)]], ignore_index = True)
      multi = BIPARTITE in N.columns and N[BIPARTITE].map(lambda x: isinstance(x, tuple)).any()
   else:
      decl = N[N["attribute"].isna().to_numpy()].drop_duplicates(keys)
      vals = N[N["attribute"].notna().to_numpy()]
      num = vals["attribute"].isin(net.node_attributes).to_numpy()
      V = vals[num].assign(value = pd.to_numeric(vals["value"][num]))
      R = agg(V.groupby(keys + ["attribute"], sort = False)["value"]).reset_index()
      Q = vals[~num].drop_duplicates(keys + ["attribute", "value"])
      N = pd.concat([R, Q, decl], ignore_index = True)
      multi = Q[(Q["attribute"] == BIPARTITE).to_numpy()].duplicated(keys).any()
   if net.bipartite_labels is not None and multi:
      raise ValueError("a set has nodes of both types of the bipartite network")
   return N


def quotient_network(net, sets, combine = "sum", node_combine = "mean", self_loops = True):
   """
   Quotient network: the nodes of each set collapse into a single node.

   ``sets`` is a list of sets (or lists, tuples, arrays) of nodes; nodes in no set stay as they
   are. The nodes collapsing the sets get integer ids, in the order of the list, counting from 0
   and skipping the integers that are ids of nodes in no set: set ``i`` gets id ``i`` unless some
   of these integers are taken (with ids ``[0, 2]`` left out of the sets, three sets get ``1``,
   ``3`` and ``4``). Each edge joins the nodes its endpoints
   collapse into; the edges then joining the same two nodes are merged into one, combining their
   attributes with ``combine``, as in :func:`remove_parallel_edges` (also merging the parallel
   edges of ``net``). Edges within a set become self loops of its node, or are dropped with
   ``self_loops = False``. With ``combine = "sum"`` and weights, the adjacency matrix of the
   quotient is :math:`P^T A P`, with :math:`P` the node-to-set membership matrix.

   The numeric node attributes of a collapsed node combine those of its nodes with
   ``node_combine``, ignoring missing values (and, in long sparse node tables, nodes without a
   value); its qualitative attributes have all the distinct values of its nodes, several values
   being stored as for overlapping communities (see :ref:`overlapping-communities`). In bipartite
   networks, a set must have nodes of a single type.

   In multilayer networks the sets hold entities: the copies of the entities of a set collapse,
   layer by layer, into the copies of a new entity, and interlayer edges are merged like the
   others (the couplings of the entities of a set become the couplings of the new entity).
   Components are recomputed; node table format, node types and device are kept.

   :param net: A :class:`~graphoskop.Network`.
   :param sets: List of iterables of nodes (input ids; entities in multilayer networks), disjoint.
   :param combine: ``"sum"``, ``"mean"`` or ``"max"``, for the edge attributes.
   :param node_combine: ``"sum"``, ``"mean"`` or ``"max"``, for the numeric node attributes.
   :param self_loops: Keep the edges within a set, as self loops of its node.
   :return: The quotient network, a new :class:`~graphoskop.Network`.
   :raises KeyError: If a node is not in the network.
   :raises ValueError: If a node is in two sets, if ``combine`` or ``node_combine`` is not valid,
      or if a set has nodes of both types of a bipartite network.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      teams = gsk.quotient_network(net, [{"alice", "bob"}, {"carol", "dave", "erin"}])   # nodes 0 and 1
      blocks = gsk.quotient_network(net, communities, self_loops = False)
   """
   for x in (combine, node_combine):
      if x not in ("sum", "mean", "max"):
         raise ValueError(f"unknown way to combine values: {x}")
   ml = net.layer_labels is not None
   new = {}
   for b, S in enumerate(sets):
      for x in [S] if isinstance(S, str) else list(S):   # a tuple is a set of nodes here
         x = x.item() if isinstance(x, np.generic) else x
         if ml:
            x = str(x) if x not in net.entity_map and str(x) in net.entity_map else x
            if x not in net.entity_map:
               raise KeyError(f"no entity {x!r}")
         else:
            x = _id(net, x)
            if x not in net.node_map:
               raise KeyError(f"no node {x!r}")
         if x in new:
            raise ValueError(f"node {x!r} is in two sets")
         new[x] = b
   ids = list(net.entity_ids) if ml else np.concatenate(net.node_ids).tolist()
   taken = {int(x) for x in ids if x not in new and isinstance(x, (int, float, np.number)) and not isinstance(x, bool) and float(x).is_integer()}
   free = (i for i in itertools.count() if i not in taken)   # ids of the collapsed nodes
   label = [next(free) for _ in range(len(sets))]
   new = {x: label[b] for x, b in new.items()}
   E, I, N = _tables(net)

   def relabel(col):
      out = col.astype(object).copy()
      m = col.isin(new.keys()).to_numpy()
      out[m] = [new[x] for x in col[m].tolist()]
      return out

   for T, cols in ((E, ("source", "target")), (I, ("source", "target")), (N, ("node",))):
      if T is not None:
         for c in cols:
            T[c] = relabel(T[c])
   if not self_loops:
      E = E[(E["source"] != E["target"]).to_numpy()]
   q = _reload(net, E, I, _collapse_nodes(net, N, node_combine))
   return remove_parallel_edges(q, combine)


def network_condensation(net, combine = "sum", node_combine = "mean"):
   """
   Condensation of a directed network: each strongly connected component collapses into a single
   node, giving a directed acyclic graph (see :func:`quotient_network`).

   The node collapsing the strongly connected component at position ``i`` of
   :func:`~graphoskop.connected_components_nodes` (``type = "strong"``, largest first) has id
   ``i``. The edges between two strongly connected components are merged into one, combining their
   attributes with ``combine``; the edges within them are dropped. Numeric node attributes combine
   with ``node_combine`` and qualitative ones keep all their distinct values, as in
   :func:`quotient_network`.

   :param net: A directed :class:`~graphoskop.Network`, not multilayer.
   :param combine: ``"sum"``, ``"mean"`` or ``"max"``, for the edge attributes.
   :param node_combine: ``"sum"``, ``"mean"`` or ``"max"``, for the numeric node attributes.
   :return: The condensation, a new directed :class:`~graphoskop.Network`.
   :raises ValueError: If ``net`` is undirected or multilayer (strongly connected components of
      the supra-graph mix copies of different layers), if ``combine`` or ``node_combine`` is not
      valid, or if a strongly connected component of a bipartite network has nodes of both types.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("citations.tsv", directed = True)
      dag = gsk.network_condensation(net)
      sccs = gsk.connected_components_nodes(net, type = "strong")   # sccs[i] collapsed into node i
   """
   if not net.directed:
      raise ValueError("the condensation is defined for directed networks")
   if net.layer_labels is not None:
      raise ValueError("the condensation is not defined for multilayer networks: strongly connected components mix copies of different layers")
   return quotient_network(net, connected_components_nodes(net, type = "strong"), combine, node_combine, self_loops = False)
