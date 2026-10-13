"""
Transformations of a network into another network.

:func:`to_undirected` symmetrizes a directed network: all the edges between two nodes, in either
direction, become one undirected edge, whose attributes combine theirs. :func:`remove_isolates`
drops the nodes without edges, :func:`remove_self_loops` the edges from a node to itself,
:func:`remove_parallel_edges` merges the edges between the same two nodes. :func:`subgraph` keeps a
set of nodes and the edges between them, :func:`to_strongly_connected` the edges within strongly
connected components.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .network import LAYER, Network, _load, _release_memory
from .multilayer import _flat_tables
from .network_statistics import _strong_labels
from .signed import SIGN


def _merge(edges, src, tgt, combine, sign, disagree):
   """
   Merge the edges of a table that join the same two nodes, in either direction.

   :param edges: Edge table: endpoint columns, then numeric attributes.
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
   out = g[attrs].sum(min_count = 1) if combine == "sum" else g[attrs].agg(combine)
   if sign in attrs:
      x = edges[sign].to_numpy()
      both = pd.DataFrame({"p": x > 0, "n": x < 0}).groupby([np.minimum(a, b), np.maximum(a, b)], sort = False).max()
      p, n = both["p"].to_numpy(), both["n"].to_numpy()
      out[sign] = np.where(p & n, disagree, np.where(p, 1, np.where(n, -1, out[sign].to_numpy())))
   ends = g[keys].first()
   return pd.concat([ends, out], axis = 1)[edges.columns].reset_index(drop = True)


def _with_edges(net, edges):
   """
   Copy of a network with new edges in each component (same nodes, so the same components).

   :param net: Network.
   :param edges: One ``(edge_index, edge_attr)`` per component, stored as in the network
      (undirected: input edges, then the reverses of those that are not self loops).
   :return: New :class:`~graphoskop.Network`; node tensors are copied.
   """
   comps = []
   for d, (ei, ea) in zip(net.components, edges):
      rest = {k: v.clone() if torch.is_tensor(v) else v for k, v in d if k not in ("edge_index", "edge_attr")}
      comps.append(Data(edge_index = ei, edge_attr = ea, **rest))
   return Network(comps, dict(net.node_map), list(net.node_ids), list(net.node_attributes), list(net.edge_attributes), net.directed, net.qualitative_node_attr.copy(), list(net.qualitative_node_attributes), net.node_format, net.bipartite_labels, None if net.bipartite_split is None else list(net.bipartite_split), net.layer_labels, net.entity_ids)


def to_undirected(net, combine = "sum", sign = SIGN, disagree = "zero"):
   """
   Undirected version of a directed network.

   All the edges between the same two nodes, in either direction (parallel edges included), are
   merged into one undirected edge; self loops of a node are merged into one self loop. Each
   numeric edge attribute of the merged edge combines the values of its edges, ignoring NaN:

   * ``"sum"``: their sum, so that weights give :math:`A + A^T`;
   * ``"mean"``: their mean (an edge in one direction only keeps its value);
   * ``"max"``: their maximum.

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
      edges, nodes = _flat_tables(net)
      edges = _merge(edges, ["source"], ["target"], combine, sign, d)
   new = _load(edges, nodes, source = "source", target = "target", directed = False, node_format = net.node_format, bipartite = net.bipartite_labels is not None, device = net.components[0].edge_index.device, multilayer = ml, interlayer_edges = inter, layers = list(net.layer_labels) if ml else None)
   _release_memory()
   return new


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
   return Network([net.components[c].clone() for c in keep], node_map, node_ids, list(net.node_attributes), list(net.edge_attributes), net.directed, Q, list(net.qualitative_node_attributes), net.node_format, net.bipartite_labels, split, net.layer_labels, net.entity_ids)


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
   edges = []
   for d in net.components:
      keep = d.edge_index[0] != d.edge_index[1]   # undirected: reverse copies are never self loops, so the input edges stay first
      edges.append((d.edge_index[:, keep], d.edge_attr[keep]))
   return _with_edges(net, edges)


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
   ``"mean"`` or ``"max"``. The sign attribute (see :doc:`signed`) keeps the common sign of the
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
   edges = []
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
      s, t = s[first], t[first]
      if not net.directed:   # stored again in both directions
         loop = s == t
         s, t, y = torch.cat([s, t[~loop]]), torch.cat([t, s[~loop]]), torch.cat([y, y[~loop]])
      edges.append((torch.stack([s, t]), y))
   return _with_edges(net, edges)


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
   inter = None
   if ml:
      E, inter, N = net.to_pandas_dataframes()
      inter = inter[np.concatenate(keep_inter)]
   else:
      E, N = _flat_tables(net)
   E = E[np.concatenate(keep)]
   if net.node_format == "wide":   # one row per node, in network wide order
      N = N[np.concatenate(nodes)]
   else:
      ids = np.concatenate([net.node_ids[c][nodes[c]] for c in range(len(net.components))])
      key = list(zip(N["node"], N[LAYER])) if ml else N["node"]
      N = N[pd.Series(key).isin(set(ids.tolist()) if ml else ids).to_numpy()]
   new = _load(E, N, source = "source", target = "target", directed = net.directed, node_format = net.node_format, bipartite = net.bipartite_labels is not None, device = net.components[0].edge_index.device, multilayer = ml, interlayer_edges = inter, layers = list(net.layer_labels) if ml else None)
   _release_memory()
   return new


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
