"""
Projections of bipartite networks onto one of their node types.

A projection keeps the nodes of one type and connects two of them according to the nodes of the
other type they share. The result is a new, unipartite :class:`~graphoskop.Network`, with the
edge weights in the edge attribute ``"weight"``, the node attributes that have values on the kept
nodes (attributes only defined on the other type are dropped), and components recomputed.

Let :math:`B` be the biadjacency matrix, rows for the kept nodes and columns for the other ones,
with parallel edges summed. In an undirected network, kept nodes :math:`a` and :math:`b` are
connected through every other node :math:`c` with :math:`B_{ac} > 0` and :math:`B_{bc} > 0`. In a
directed network, through every directed path :math:`a \\to c \\to b` of length two, with
:math:`B_{ac}` the weight of :math:`a \\to c` and :math:`B_{bc}` the weight of :math:`c \\to b`;
the projection is then directed, and its components can split even if the bipartite ones do
not. Each function combines the two weights of every path differently.

The similarity projections (:func:`cosine_similarity_projection`,
:func:`pearson_correlation_projection`, :func:`inverse_euclidean_projection`) compare instead the
whole rows of :math:`B` of two kept nodes, as vectors over all the nodes of the other type. They
take the dot products from the paths and the rest from the sums and sums of squares of each row,
so they too only connect (and only compute) the pairs of kept nodes sharing a neighbor.

All functions generate the paths of length two explicitly, :math:`\\sum_c d_c (d_c - 1) / 2` of
them (:math:`\\sum_c d^{in}_c d^{out}_c` if directed), in batches of about ``batch_size`` paths,
each reduced to the sums of its distinct node pairs on the device of the network. Batches cover
disjoint sets of kept nodes, so no pair is split between two batches and nothing needs merging:
peak memory is about 60 bytes per path of a batch (1 GB with the default ``batch_size = 2 ** 24``)
plus the result. Smaller batches use less memory and take a little longer. This is faster and
leaner than a sparse matrix product, even for projections that are one.

All functions take a ``weight`` parameter, defaulting to the edge attribute ``"weight"``; if the
network has none, a warning is issued and all weights are 1. Any other missing name raises
:class:`KeyError`. Degrees become strengths (sums of edge weights) with weights.

Multilayer bipartite networks are projected layer by layer: paths of length two only use
intra-layer edges, which stay within a layer, and couplings are ignored, so strengths and totals
are those of each layer. The projection is a multilayer network with the copies of the kept type,
re-coupled with the interlayer edges of the input between them (couplings and the other
interlayer edges, which join nodes of the same type), whose ``"weight"`` is their value of the
``weight`` attribute (1 with unit weights).
"""

import warnings

import numpy as np
import scipy.sparse as sp
import torch

from ._policy import _higher_order
from .network import BIPARTITE, _build, _node_codes, _node_table, _node_types, _positions, _qualitative_rows, _categorize, _quiet_sparse, _release_memory


def _biadjacency(net, node_type, weight):
   """
   Biadjacency matrices of all components at once, rows for the kept nodes and columns for the
   other ones, in the network wide node order.

   :param net: Bipartite network.
   :param node_type: Value of :data:`~graphoskop.BIPARTITE` of the kept nodes.
   :param weight: Edge attribute name used as weight, or ``None`` for unit weights.
   :return: Positions of the kept nodes in the network wide node order, and the coalesced sparse
      COO matrices :math:`B_{out}` (kept to other) and :math:`B_{in}` (other to kept, transposed:
      rows for the other nodes). In undirected networks both are :math:`B` and :math:`B^T`.
   :raises ValueError: If the network is not bipartite or has no node of type ``node_type``.
   """
   if net.bipartite_labels is None:
      raise ValueError("the network is not bipartite: load it with bipartite = True")
   if node_type not in net.bipartite_labels:
      raise ValueError(f"no node type {node_type!r}; node types: {net.bipartite_labels}")
   keep = net.bipartite_labels.index(node_type)
   types = _node_types(net)
   rows = np.flatnonzero(types == keep)
   if len(rows) == 0:
      raise ValueError(f"no node of type {node_type!r}")
   dev = net.components[0].edge_index.device
   idx = np.cumsum(types == keep) - 1
   idx[types != keep] = (np.cumsum(types != keep) - 1)[types != keep]
   idx, kept = torch.as_tensor(idx, device = dev), torch.as_tensor(types == keep, device = dev)
   s, t, w = [], [], []
   for c in range(len(net.components)):
      ei, X = net._input_edges(c)
      s.append(ei[0] + int(net.node_offsets[c]))
      t.append(ei[1] + int(net.node_offsets[c]))
      w.append(torch.ones(ei.shape[1], dtype = X.dtype, device = dev) if weight is None else X[:, net._column(net.edge_attributes, weight, "edge")])
   s, t, w = torch.cat(s), torch.cat(t), torch.cat(w)
   cross = kept[s] != kept[t]   # leaves out the interlayer edges of multilayer networks
   if not bool(cross.all()):
      s, t, w = s[cross], t[cross], w[cross]
   out = kept[s]   # edge from a kept node to another one
   k, m = len(rows), len(types) - len(rows)

   def coo(i, j, v, shape):
      with _quiet_sparse():
         return torch.sparse_coo_tensor(torch.stack([i, j]), v, shape).coalesce()

   if not net.directed:
      u, x = torch.where(out, s, t), torch.where(out, t, s)
      B = coo(idx[u], idx[x], w, (k, m))
      return rows, B, B.t().coalesce()
   return rows, coo(idx[s[out]], idx[t[out]], w[out], (k, m)), coo(idx[s[~out]], idx[t[~out]], w[~out], (m, k))


def _projection(net, rows, i, j, v, directed = None, weight = None):
   """
   Unipartite network from the entries of a projection matrix, with the node attributes having
   values on the kept nodes; in multilayer networks, with the interlayer edges between kept nodes.

   :param net: Bipartite network.
   :param rows: Positions of the kept nodes in the network wide node order.
   :param i: Row of each entry (index among the kept nodes).
   :param j: Column of each entry.
   :param v: Weight of each entry. Entries on the diagonal and, in undirected projections, below
      it are ignored.
   :param directed: Whether the projection is directed, ``None`` for the same as ``net``.
   :param weight: Edge attribute giving the weight of the interlayer edges, or ``None`` for 1.
   :return: The new :class:`~graphoskop.Network`.
   """
   directed = net.directed if directed is None else directed
   e = i < j if not directed else i != j
   if not bool(e.all()):   # otherwise no copy (on the CPU)
      i, j, v = i[e], j[e], v[e]
   del e
   s, t = i.cpu().numpy(), j.cpu().numpy()
   ea = v.cpu().numpy().astype(np.float32, copy = False)[:, None]
   ml = None
   if net.layer_labels is not None:   # interlayer edges between kept nodes
      lay, ent = _node_codes(net)
      ml = (lay[rows], ent[rows], net.layer_labels, net.entity_ids)
      pos = np.full(int(net.node_offsets[-1]), -1, dtype = np.int64)
      pos[rows] = np.arange(len(rows))
      si, ti, wi = [], [], []
      for c in range(len(net.components)):
         ei, X = net._input_edges(c)
         m = net.interlayer_mask(c)[:ei.shape[1]]
         a, b = (pos[x + int(net.node_offsets[c])] for x in ei[:, m].cpu().numpy())
         ok = (a >= 0) & (b >= 0)
         si.append(a[ok])
         ti.append(b[ok])
         wi.append(np.ones(int(ok.sum()), dtype = np.float32) if weight is None else X[m, net._column(net.edge_attributes, weight, "edge")].cpu().numpy()[ok])
      si, ti, wi = np.concatenate(si), np.concatenate(ti), np.concatenate(wi)
      if directed and not net.directed:   # both directions of the undirected interlayer edges
         si, ti, wi = np.concatenate([si, ti]), np.concatenate([ti, si]), np.concatenate([wi, wi])
      s, t, ea = np.concatenate([s, si]), np.concatenate([t, ti]), np.concatenate([ea, wi[:, None]])

   # node attributes with at least one value on the kept nodes
   x = _node_table(net, rows)
   cols = np.flatnonzero(np.asarray((x != 0).sum(0)).ravel() > 0 if sp.issparse(x) else np.isfinite(x).any(0))
   x = x[:, cols]
   if sp.issparse(x):
      x.sort_indices()
   Q = _qualitative_rows(net, rows)
   if net.node_format == "wide":
      quals = [a for a in net.qualitative_node_attributes if a != BIPARTITE and Q[a].notna().any()]
      Q = Q[quals]
      qtable = lambda perm: Q.iloc[perm].reset_index(drop = True)
   else:
      Q = Q[(Q["attribute"] != BIPARTITE).to_numpy()]
      quals = [a for a in net.qualitative_node_attributes if a in set(Q["attribute"])]

      def qtable(perm):
         R = Q.copy()
         R["node"] = _positions(perm)[R["node"].to_numpy()]
         return _categorize(R)

   names = np.concatenate(net.node_ids)[rows]
   device = net.components[0].edge_index.device
   return _build(s, t, names, ea, ["weight"], lambda perm: x[perm], [net.node_attributes[c] for c in cols], directed, device, qtable, quals, net.node_format, ml = ml)


def _pair_batches(first, partner, cnt, base, batch_size):
   """
   Paths of length two through the other nodes, generated in batches with disjoint first
   endpoints, so that no pair of kept nodes appears in two batches.

   Entry ``e`` of ``first`` (sorted by row) is paired with the ``cnt[e]`` consecutive entries of
   ``partner`` starting at ``base[e]``. Batches hold about ``batch_size`` pairs, and are split
   between rows (more pairs only if a single row has more).

   :param first: ``(rows, weights, other nodes)`` of the first endpoint of each entry, sorted by
      row.
   :param partner: ``(rows, weights)`` of the partner entries.
   :param cnt: Number of partners of each entry of ``first``.
   :param base: Position of the first partner of each entry.
   :param batch_size: Target number of pairs per batch.
   :return: Generator of ``(a, b, wa, wb, c)``: endpoints, weights of the two edges and other node
      of each path.
   """
   cum = torch.cumsum(cnt, 0).cpu().numpy()
   rows = first[0].cpu().numpy()
   e0 = 0
   while e0 < len(cum):
      done = cum[e0 - 1] if e0 else 0
      e1 = max(int(np.searchsorted(cum, done + batch_size, side = "right")), e0 + 1)
      e1 = int(np.searchsorted(rows, rows[e1 - 1], side = "right"))   # whole rows only
      c = cnt[e0:e1]
      tot = int(cum[e1 - 1] - done)
      if tot:
         e = torch.repeat_interleave(torch.arange(e0, e1, device = c.device), c, output_size = tot)
         q = base[e] + torch.arange(tot, device = c.device) - torch.repeat_interleave(torch.cumsum(c, 0) - c, c, output_size = tot)
         yield first[0][e], partner[0][q], first[1][e], partner[1][q], first[2][e]
      e0 = e1


def _reduce(keys, values):
   """
   Sum the values with equal keys.

   :param keys: int64 keys.
   :param values: Values.
   :return: Sorted unique keys and their sums.
   """
   u, inv = torch.unique(keys, return_inverse = True)
   return u, torch.zeros(len(u), dtype = values.dtype, device = values.device).index_add_(0, inv, values)


def _ratio(a, b):
   """
   Elementwise ``a / b``, with 0 where ``b`` is 0 (nodes whose edges all have weight 0).
   """
   return torch.where(b == 0, 0, a / torch.where(b == 0, 1, b))


def _weight(net, weight):
   """
   Edge attribute to use as weight: the default ``"weight"`` falls back to unit weights, with a
   warning, if the network has no such attribute.

   :param net: Network.
   :param weight: Edge attribute name, or ``None``.
   :return: ``weight``, or ``None`` for unit weights.
   """
   if weight == "weight" and weight not in net.edge_attributes:
      warnings.warn("no edge attribute named 'weight': using unit weights")
      return None
   return weight


def _paths(net, node_type, weight, value, batch_size):
   """
   Sums of a value over the paths of length two between kept nodes, generated explicitly in
   batches (see the module description for time and memory).

   :param net: Bipartite network.
   :param node_type: Value of :data:`~graphoskop.BIPARTITE` of the kept nodes.
   :param weight: Edge attribute name used as weight, or ``None`` for unit weights.
   :param value: Function of ``(wa, wb, c, strength)``, the weights of the two edges of each path,
      its other node and the strength of every other node (sum of the weights of its edges, in
      both directions), returning the value of each path.
   :param batch_size: Target number of paths per batch.
   :return: Positions of the kept nodes in the network wide node order, the strength of each kept
      node (sum of the weights of its edges, in both directions), and the projection entries
      ``(i, j, v)``: one per pair of kept nodes joined by a path, ``i < j`` in undirected networks.
   """
   rows, B_out, B_in = _biadjacency(net, node_type, weight)
   k = len(rows)
   T = B_out.t().coalesce()   # entries grouped by the other node c
   (c1, a), w1 = T.indices(), T.values()
   (c2, b), w2 = B_in.indices(), B_in.values()
   del T, B_out, B_in
   m = int(max(c1.max() if len(c1) else -1, c2.max() if len(c2) else -1)) + 1
   strength, kstrength = torch.bincount(c1, w1, minlength = m).to(w1.dtype), torch.bincount(a, w1, minlength = k).to(w1.dtype)   # integers if empty
   if not net.directed:   # each pair of entries of the same c once
      pos = torch.arange(len(c1), device = c1.device)
      cnt = torch.cumsum(torch.bincount(c1, minlength = m), 0)[c1] - 1 - pos
      base = pos + 1
   else:                  # every entry of c in B_out with every entry of c in B_in
      indeg = torch.bincount(c2, minlength = m)
      cnt, base = indeg[c1], (torch.cumsum(indeg, 0) - indeg)[c1]
      strength += torch.bincount(c2, w2, minlength = m).to(w2.dtype)
      kstrength += torch.bincount(b, w2, minlength = k).to(w2.dtype)
   order = torch.argsort(a, stable = True)   # first endpoints by row: batches never share a pair
   i, j, vals = [], [], []
   for x, y, wx, wy, c in _pair_batches((a[order], w1[order], c1[order]), (b, w2) if net.directed else (a, w1), cnt[order], base[order], batch_size):
      keep = x != y   # undirected: x < y already, as entries of each c are sorted by row
      key, v = _reduce(x[keep] * k + y[keep], value(wx[keep], wy[keep], c[keep], strength))
      del x, y, wx, wy, c, keep
      i.append(key // k)
      j.append(key - i[-1] * k)
      vals.append(v)
      del key
   empty = torch.zeros(0, dtype = torch.long, device = a.device)
   i, j, v = (torch.cat(x) if x else e for x, e in ((i, empty), (j, empty), (vals, w1[:0])))
   del vals
   _release_memory()   # batch buffers, before building the network
   return rows, kstrength, i, j, v


@_higher_order()
def simple_count_projection(net, node_type, weight = "weight", batch_size = 2 ** 24):
   """
   Projection counting the shared neighbors, weighted: two kept nodes :math:`a` and :math:`b` are
   connected with weight :math:`\\sum_c B_{ac} B_{bc}`, i.e. :math:`B B^T`
   (:math:`B_{out} B_{in}` in directed networks), summing over the nodes :math:`c` of the other
   type they share (paths of length two in directed networks, see the module description). With
   unit weights (``weight = None``) this is the number of shared neighbors, parallel edges
   counting as many times as they appear. Kept nodes sharing no neighbor stay as isolated nodes.

   :param net: A bipartite :class:`~graphoskop.Network`.
   :param node_type: Value of :data:`~graphoskop.BIPARTITE` of the nodes to keep, one of
      ``net.bipartite_labels``.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
      The default ``"weight"`` falls back to unit weights, with a warning, if there is no such
      attribute.
   :param batch_size: Target number of paths of length two per batch (see the module
      description).
   :return: The projection, a new unipartite :class:`~graphoskop.Network`, directed if ``net`` is,
      with the edge attribute ``"weight"``.
   :raises ValueError: If ``net`` is not bipartite, or has no node of type ``node_type``.
   :raises KeyError: If ``weight`` is another name and there is no edge attribute with it.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("purchases.tsv", "nodes.tsv", bipartite = True)
      users = gsk.simple_count_projection(net, "user", weight = None)   # items bought in common
      items = gsk.simple_count_projection(net, "item", weight = "quantity")
      users.get_edge_attribute("weight")
   """
   weight = _weight(net, weight)
   rows, _, i, j, v = _paths(net, node_type, weight, lambda wa, wb, c, strength: wa * wb, batch_size)
   return _projection(net, rows, i, j, v, weight = weight)


@_higher_order()
def hist_intersection_projection(net, node_type, weight = "weight", batch_size = 2 ** 24):
   """
   Histogram intersection projection: two kept nodes :math:`a` and :math:`b` are connected with
   weight :math:`\\sum_c \\min(B_{ac}, B_{bc})`, summing over the nodes :math:`c` of the other type
   they share (over the paths :math:`a \\to c \\to b` in directed networks, see the module
   description). Parallel edges are first summed. With unit weights (``weight = None``) this is
   the number of shared neighbors, as in :func:`simple_count_projection`. Kept nodes sharing no
   neighbor stay as isolated nodes.

   A sum of minima is not a matrix product, so the paths of length two are generated explicitly,
   :math:`\\sum_c d_c (d_c - 1) / 2` of them (:math:`\\sum_c d^{in}_c d^{out}_c` if directed), in
   batches of about ``batch_size`` paths, each reduced to the sums of its distinct node pairs on
   the device of the network. Batches cover disjoint sets of kept nodes, so no pair is split
   between two batches and nothing needs merging: peak memory is about 60 bytes per path of a
   batch (1 GB with the default) plus the result. Smaller batches use less memory and take a
   little longer.

   :param net: A bipartite :class:`~graphoskop.Network`.
   :param node_type: Value of :data:`~graphoskop.BIPARTITE` of the nodes to keep, one of
      ``net.bipartite_labels``.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param batch_size: Target number of paths of length two per batch.
   :return: The projection, a new unipartite :class:`~graphoskop.Network`, directed if ``net`` is,
      with the edge attribute ``"weight"``.
   :raises ValueError: If ``net`` is not bipartite, or has no node of type ``node_type``.
   :raises KeyError: If there is no edge attribute named ``weight``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("purchases.tsv", "nodes.tsv", bipartite = True, device = "cuda")
      users = gsk.hist_intersection_projection(net, "user")                          # edge attribute "weight"
      users = gsk.hist_intersection_projection(net, "user", weight = "quantity")
      users = gsk.hist_intersection_projection(net, "user", batch_size = 2 ** 22)    # less memory
   """
   weight = _weight(net, weight)
   rows, _, i, j, v = _paths(net, node_type, weight, lambda wa, wb, c, strength: torch.minimum(wa, wb), batch_size)
   return _projection(net, rows, i, j, v, weight = weight)


@_higher_order()
def hyperbolic_projection(net, node_type, weight = "weight", batch_size = 2 ** 24):
   """
   Newman's hyperbolic projection: every node :math:`c` of the other type shared by two kept nodes
   :math:`a` and :math:`b` counts :math:`1 / k_c`, with :math:`k_c` the degree of :math:`c`, so
   that shared neighbors with many connections count less. With weights, :math:`k_c` is the
   strength :math:`s_c = \\sum_x B_{xc}` (sum of the weights of the edges of :math:`c`) and each
   path counts its two weights, so the weight is
   :math:`\\sum_c B_{ac} B_{bc} / s_c`, i.e. :math:`B S^{-1} B^T` with :math:`S` the diagonal
   matrix of the strengths of the other nodes; with unit weights, :math:`\\sum_c 1 / k_c`.
   Parallel edges are summed, so with unit weights they count as many times as they appear. In
   directed networks the sum runs over the paths :math:`a \\to c \\to b` (see the module
   description), and :math:`s_c` sums the edges of :math:`c` in both directions. Kept nodes
   sharing no neighbor stay as isolated nodes.

   :param net: A bipartite :class:`~graphoskop.Network`.
   :param node_type: Value of :data:`~graphoskop.BIPARTITE` of the nodes to keep, one of
      ``net.bipartite_labels``.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
      The default ``"weight"`` falls back to unit weights, with a warning, if there is no such
      attribute.
   :param batch_size: Target number of paths of length two per batch (see the module
      description).
   :return: The projection, a new unipartite :class:`~graphoskop.Network`, directed if ``net`` is,
      with the edge attribute ``"weight"``.
   :raises ValueError: If ``net`` is not bipartite, or has no node of type ``node_type``.
   :raises KeyError: If ``weight`` is another name and there is no edge attribute with it.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("papers.tsv", "nodes.tsv", bipartite = True)
      authors = gsk.hyperbolic_projection(net, "author", weight = None)   # co-authorship, large papers count less
   """
   weight = _weight(net, weight)
   rows, _, i, j, v = _paths(net, node_type, weight, lambda wa, wb, c, strength: _ratio(wa * wb, strength[c]), batch_size)
   return _projection(net, rows, i, j, v, weight = weight)


@_higher_order()
def resource_allocation_projection(net, node_type, lambd = 1, weight = "weight", batch_size = 2 ** 24):
   """
   Resource allocation projections (Zhou et al.): ProbS (mass diffusion), HeatS (heat conduction)
   and their hybrid. They start from the hyperbolic sum
   :math:`H_{ab} = \\sum_c B_{ac} B_{bc} / s_c` of :func:`hyperbolic_projection`, and normalize it by
   the strengths :math:`s_a` and :math:`s_b` of the kept nodes (sums of the weights of their
   edges; their degrees with unit weights). Each projected edge :math:`a \\to b` carries what flows
   from :math:`a` to :math:`b`:

   .. math::

      w_{a \\to b} = \\frac{H_{ab}}{s_a^{\\lambda} s_b^{1 - \\lambda}}

   * ``lambd = 1``, ProbS: normalized by the sender. In undirected networks the weights leaving a
     node, its self loop included (which is dropped), sum to 1: the fraction of its resource
     reaching each other node.
   * ``lambd = 0``, HeatS: normalized by the receiver. In undirected networks the weights entering
     a node, its self loop included, sum to 1: the share of its value averaged from each other
     node.
   * ``0 < lambd < 1``: the hybrid. With ``lambd = 0.5``, :math:`H_{ab} / \\sqrt{s_a s_b}` is
     symmetric.

   The projection is undirected only if ``lambd = 0.5`` and ``net`` is undirected; otherwise it is
   directed, with both directions of every pair if ``net`` is undirected. Parallel edges are
   summed. In directed networks :math:`H` sums over the paths :math:`a \\to c \\to b` (see the
   module description), and strengths sum the edges in both directions. Kept nodes sharing no
   neighbor stay as isolated nodes.

   :param net: A bipartite :class:`~graphoskop.Network`.
   :param node_type: Value of :data:`~graphoskop.BIPARTITE` of the nodes to keep, one of
      ``net.bipartite_labels``.
   :param lambd: Hybrid parameter :math:`\\lambda`, from 0 (HeatS) to 1 (ProbS).
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
      The default ``"weight"`` falls back to unit weights, with a warning, if there is no such
      attribute.
   :param batch_size: Target number of paths of length two per batch (see the module
      description).
   :return: The projection, a new unipartite :class:`~graphoskop.Network` with the edge attribute
      ``"weight"``.
   :raises ValueError: If ``lambd`` is not between 0 and 1, if ``net`` is not bipartite, or if it
      has no node of type ``node_type``.
   :raises KeyError: If ``weight`` is another name and there is no edge attribute with it.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("purchases.tsv", "nodes.tsv", bipartite = True)
      items = gsk.resource_allocation_projection(net, "item")                 # ProbS, directed
      items = gsk.resource_allocation_projection(net, "item", lambd = 0)      # HeatS, directed
      items = gsk.resource_allocation_projection(net, "item", lambd = 0.5)    # hybrid, undirected
   """
   if not 0 <= lambd <= 1:
      raise ValueError(f"lambd must be between 0 and 1, got {lambd}")
   weight = _weight(net, weight)
   rows, ks, i, j, v = _paths(net, node_type, weight, lambda wa, wb, c, strength: _ratio(wa * wb, strength[c]), batch_size)
   directed = net.directed or lambd != 0.5
   if directed and not net.directed:   # H is symmetric: both directions of every pair
      i, j, v = torch.cat([i, j]), torch.cat([j, i]), torch.cat([v, v])
   ks = ks.to(v.dtype)
   v = _ratio(v, ks[i] ** lambd * ks[j] ** (1 - lambd))
   return _projection(net, rows, i, j, v, directed, weight)


@_higher_order()
def random_walk_projection(net, node_type, directed = False, weight = "weight", batch_size = 2 ** 24):
   """
   Random walk projection (YCN, Yildirim and Coscia, 2014): the weight from :math:`a` to :math:`b`
   is the probability that a random walker on the bipartite network moves from :math:`a` to
   :math:`b` in two steps (through a node of the other type), times the probability of finding it
   on :math:`b` after infinitely many steps:

   .. math::

      U_{a \\to b} = T_{ab} \\pi_b, \\qquad T_{ab} = \\sum_c \\frac{B_{ac}}{s_a} \\frac{B_{bc}}{s_c}

   with :math:`B` the biadjacency matrix, :math:`s` the strengths (sums of edge weights; degrees
   with unit weights), :math:`T` the two-step transition matrix (ProbS, see
   :func:`resource_allocation_projection`) and :math:`\\pi` its stationary distribution. The walk
   is reversible, so :math:`\\pi_b = s_b / S`, with :math:`S` the total edge weight, and no
   eigenvector is computed: :math:`U_{a \\to b} = H_{ab} s_b / (s_a S)`, with :math:`H` the
   hyperbolic sum of :func:`hyperbolic_projection`. With ``directed = False`` the two directions
   are averaged, :math:`(U_{a \\to b} + U_{b \\to a}) / 2`. Parallel edges are summed. Kept nodes
   sharing no neighbor stay as isolated nodes.

   This is the stationary distribution of the whole network, so on networks with several
   components, weights are comparable across components (the reference implementation takes an
   arbitrary eigenvector of eigenvalue 1 instead, which is unique only on connected networks). In
   multilayer networks, :math:`S` is the total edge weight of the layer.

   :param net: An undirected bipartite :class:`~graphoskop.Network`.
   :param node_type: Value of :data:`~graphoskop.BIPARTITE` of the nodes to keep, one of
      ``net.bipartite_labels``.
   :param directed: Return the directed projection :math:`U`, with both directions of every
      pair, instead of the average of the two directions.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
      The default ``"weight"`` falls back to unit weights, with a warning, if there is no such
      attribute.
   :param batch_size: Target number of paths of length two per batch (see the module
      description).
   :return: The projection, a new unipartite :class:`~graphoskop.Network` with the edge attribute
      ``"weight"``.
   :raises ValueError: If ``net`` is directed (the walk is then not reversible), is not
      bipartite, or has no node of type ``node_type``.
   :raises KeyError: If ``weight`` is another name and there is no edge attribute with it.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("occupations.tsv", "nodes.tsv", bipartite = True)
      occupations = gsk.random_walk_projection(net, "occupation")
      flows = gsk.random_walk_projection(net, "occupation", directed = True)
   """
   if net.directed:
      raise ValueError("the random walk projection is defined on undirected bipartite networks")
   weight = _weight(net, weight)
   rows, ks, i, j, v = _paths(net, node_type, weight, lambda wa, wb, c, strength: _ratio(wa * wb, strength[c]), batch_size)
   ks = ks.to(v.dtype)
   if net.layer_labels is None:
      v = v / ks.sum()
   else:   # total edge weight of the layer of each pair
      lay = torch.as_tensor(_node_codes(net)[0][rows], device = v.device)
      v = v / torch.zeros(len(net.layer_labels), dtype = v.dtype, device = v.device).index_add_(0, lay, ks)[lay[i]]
   if directed:   # U_ab = H_ab s_b / (s_a S), both directions of every pair
      i, j, v = torch.cat([i, j]), torch.cat([j, i]), torch.cat([v * _ratio(ks[j], ks[i]), v * _ratio(ks[i], ks[j])])
   else:
      v = v * (_ratio(ks[j], ks[i]) + _ratio(ks[i], ks[j])) / 2
   return _projection(net, rows, i, j, v, directed, weight)


def _similarity_projection(net, node_type, weight, batch_size, similarity):
   """
   Projection weighting every pair of kept nodes sharing a neighbor by a similarity of their rows
   of the biadjacency matrix (out-vector of the first node and in-vector of the second in directed
   networks), computed from their dot product and the moments of the two vectors.

   :param net: Bipartite network.
   :param node_type: Value of :data:`~graphoskop.BIPARTITE` of the kept nodes.
   :param weight: Edge attribute name, or ``"weight"`` with fallback to unit weights.
   :param batch_size: Target number of paths per batch.
   :param similarity: Function of ``(dot, sa, qa, sb, qb, m)``: dot product, sum and sum of squares
      of the two vectors and their length, as float64 tensors, one value per pair.
   :return: The projection.
   """
   weight = _weight(net, weight)
   rows, _, i, j, dot = _paths(net, node_type, weight, lambda wa, wb, c, strength: wa.double() * wb.double(), batch_size)
   _, B_out, B_in = _biadjacency(net, node_type, weight)
   k = len(rows)
   a, wa = B_out.indices()[0], B_out.values().double()
   b, wb = B_in.indices()[1], B_in.values().double()
   del B_out, B_in
   sa, qa = torch.bincount(a, wa, minlength = k), torch.bincount(a, wa ** 2, minlength = k)
   sb, qb = torch.bincount(b, wb, minlength = k), torch.bincount(b, wb ** 2, minlength = k)
   # length of the vectors: number of nodes of the other type (in the layer of each kept node)
   other = _node_types(net) != net.bipartite_labels.index(node_type)
   if net.layer_labels is None:
      m = np.full(k, other.sum())
   else:
      lay = _node_codes(net)[0]
      m = np.bincount(lay[other], minlength = len(net.layer_labels))[lay[rows]]
   m = torch.as_tensor(m, dtype = torch.float64, device = dot.device)
   v = similarity(dot.double(), sa[i], qa[i], sb[j], qb[j], m[i])
   return _projection(net, rows, i, j, v, weight = weight)


@_higher_order()
def cosine_similarity_projection(net, node_type, weight = "weight", batch_size = 2 ** 24):
   """
   Cosine similarity projection: two kept nodes :math:`a` and :math:`b` sharing at least one
   neighbor are connected with the cosine of the angle between their rows :math:`B_a` and
   :math:`B_b` of the biadjacency matrix:

   .. math::

      w_{ab} = \\frac{\\sum_c B_{ac} B_{bc}}{\\sqrt{\\sum_c B_{ac}^2} \\sqrt{\\sum_c B_{bc}^2}}

   The numerator is :func:`simple_count_projection`. With unit weights (``weight = None``) this is
   the number of shared neighbors over :math:`\\sqrt{k_a k_b}` (Salton index). Weights are between
   0 and 1 for non-negative edge weights. Kept nodes sharing no neighbor (similarity 0) stay as
   isolated nodes.

   Vectors use the edge weights with parallel edges summed. In directed networks the weight of
   :math:`a \\to b` compares the out-vector of :math:`a` (weights of :math:`a \\to c`) with the
   in-vector of :math:`b` (weights of :math:`c \\to b`), and the projection is directed. The dot
   products are sums over the paths of length two (see the module description), accumulated in
   float64.

   :param net: A bipartite :class:`~graphoskop.Network`.
   :param node_type: Value of :data:`~graphoskop.BIPARTITE` of the nodes to keep, one of
      ``net.bipartite_labels``.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
      The default ``"weight"`` falls back to unit weights, with a warning, if there is no such
      attribute.
   :param batch_size: Target number of paths of length two per batch (see the module
      description).
   :return: The projection, a new unipartite :class:`~graphoskop.Network`, directed if ``net`` is,
      with the edge attribute ``"weight"``.
   :raises ValueError: If ``net`` is not bipartite, or has no node of type ``node_type``.
   :raises KeyError: If ``weight`` is another name and there is no edge attribute with it.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("purchases.tsv", "nodes.tsv", bipartite = True)
      users = gsk.cosine_similarity_projection(net, "user", weight = "quantity")
   """
   return _similarity_projection(net, node_type, weight, batch_size, lambda dot, sa, qa, sb, qb, m: _ratio(dot, torch.sqrt(qa * qb)))


@_higher_order()
def pearson_correlation_projection(net, node_type, weight = "weight", batch_size = 2 ** 24):
   """
   Pearson correlation projection: two kept nodes :math:`a` and :math:`b` sharing at least one
   neighbor are connected with the Pearson correlation of their rows :math:`B_a` and :math:`B_b`
   of the biadjacency matrix, over all the :math:`m` nodes of the other type (zeros included):

   .. math::

      w_{ab} = \\frac{\\sum_c B_{ac} B_{bc} - m \\bar{B}_a \\bar{B}_b}
      {\\sqrt{\\sum_c B_{ac}^2 - m \\bar{B}_a^2} \\sqrt{\\sum_c B_{bc}^2 - m \\bar{B}_b^2}}

   with :math:`\\bar{B}_a = \\sum_c B_{ac} / m`. Weights are between -1 and 1: two nodes sharing
   a neighbor can still be negatively correlated, e.g. if each has many neighbors the other lacks.
   A node whose vector is constant (connected to every other node with the same weight) has zero
   variance, and its correlations are 0. Kept nodes sharing no neighbor stay as isolated nodes,
   even though their correlation is not 0 (it is negative): only pairs joined by a path of length
   two are computed.

   Vectors use the edge weights with parallel edges summed. :math:`m` counts all the nodes of the
   other type in the network, isolated ones and other components included; in multilayer
   networks, those of the layer. In directed networks the weight of :math:`a \\to b` compares the
   out-vector of :math:`a` with the in-vector of :math:`b`, and the projection is directed. The
   dot products are sums over the paths of length two (see the module description), accumulated
   in float64 to limit cancellation.

   :param net: A bipartite :class:`~graphoskop.Network`.
   :param node_type: Value of :data:`~graphoskop.BIPARTITE` of the nodes to keep, one of
      ``net.bipartite_labels``.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
      The default ``"weight"`` falls back to unit weights, with a warning, if there is no such
      attribute.
   :param batch_size: Target number of paths of length two per batch (see the module
      description).
   :return: The projection, a new unipartite :class:`~graphoskop.Network`, directed if ``net`` is,
      with the edge attribute ``"weight"``.
   :raises ValueError: If ``net`` is not bipartite, or has no node of type ``node_type``.
   :raises KeyError: If ``weight`` is another name and there is no edge attribute with it.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("purchases.tsv", "nodes.tsv", bipartite = True)
      users = gsk.pearson_correlation_projection(net, "user", weight = None)
   """
   def pearson(dot, sa, qa, sb, qb, m):
      va, vb = torch.clamp(qa - sa ** 2 / m, min = 0), torch.clamp(qb - sb ** 2 / m, min = 0)
      return _ratio(dot - sa * sb / m, torch.sqrt(va * vb))

   return _similarity_projection(net, node_type, weight, batch_size, pearson)


@_higher_order()
def inverse_euclidean_projection(net, node_type, weight = "weight", batch_size = 2 ** 24):
   """
   Inverse Euclidean distance projection: two kept nodes :math:`a` and :math:`b` sharing at least
   one neighbor are connected with

   .. math::

      w_{ab} = \\frac{1}{1 + d_{ab}}, \\qquad
      d_{ab} = \\sqrt{\\sum_c (B_{ac} - B_{bc})^2}
      = \\sqrt{\\sum_c B_{ac}^2 + \\sum_c B_{bc}^2 - 2 \\sum_c B_{ac} B_{bc}}

   the Euclidean distance between their rows of the biadjacency matrix. Adding 1 keeps the weight
   finite, between 0 and 1, with 1 for identical vectors. With unit weights (``weight = None``),
   :math:`d_{ab}^2` is the number of neighbors of exactly one of the two nodes. Kept nodes sharing
   no neighbor stay as isolated nodes, even though their similarity is not 0: only pairs joined
   by a path of length two are computed.

   Vectors use the edge weights with parallel edges summed. In directed networks the weight of
   :math:`a \\to b` compares the out-vector of :math:`a` with the in-vector of :math:`b`, and the
   projection is directed. The dot products are sums over the paths of length two (see the module
   description), accumulated in float64 to limit cancellation.

   :param net: A bipartite :class:`~graphoskop.Network`.
   :param node_type: Value of :data:`~graphoskop.BIPARTITE` of the nodes to keep, one of
      ``net.bipartite_labels``.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
      The default ``"weight"`` falls back to unit weights, with a warning, if there is no such
      attribute.
   :param batch_size: Target number of paths of length two per batch (see the module
      description).
   :return: The projection, a new unipartite :class:`~graphoskop.Network`, directed if ``net`` is,
      with the edge attribute ``"weight"``.
   :raises ValueError: If ``net`` is not bipartite, or has no node of type ``node_type``.
   :raises KeyError: If ``weight`` is another name and there is no edge attribute with it.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("purchases.tsv", "nodes.tsv", bipartite = True)
      users = gsk.inverse_euclidean_projection(net, "user", weight = "quantity")
   """
   return _similarity_projection(net, node_type, weight, batch_size, lambda dot, sa, qa, sb, qb, m: 1 / (1 + torch.sqrt(torch.clamp(qa + qb - 2 * dot, min = 0))))
