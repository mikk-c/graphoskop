"""
Isomorphism: whether two networks are the same up to a relabeling of their nodes, and whether a
network contains another one.

Two networks are isomorphic when a one-to-one map of the nodes of one onto the nodes of the other
maps edges onto edges and non edges onto non edges. A network contains another one, the pattern,
when the pattern is isomorphic to a subgraph of it: to an induced subgraph (some nodes and all the
edges among them), or, more loosely, to any subgraph (some nodes and some of the edges among
them: a monomorphism).

Both questions have an exact and an approximate answer.

* The approximate answer is one sided: ``False`` is always right, and ``True`` means that the
  networks passed a necessary condition. For isomorphism, the condition is color refinement, the
  1-dimensional Weisfeiler-Leman test: every node gets a color from its attributes, then the color
  of every node is repeatedly combined with the multiset of the colors of its neighbors, on both
  networks at once, until no color class splits; isomorphic networks end with the same number of
  nodes of every color. Regular graphs of the same size and degree are the classic case that the
  test cannot tell apart. For subgraphs, the condition is a filter of the candidate images of
  every pattern node (same attributes, at least the same degree), keeping a candidate only while
  every neighbor of the pattern node has a candidate among the neighbors of the candidate (the
  refinement of Ullmann, 1976, as matrix products), followed by a matching that assigns distinct
  candidates to all pattern nodes. Both are vectorized tensor operations, on the device of the
  networks, except color refinement on the CPU, which runs in compiled code with Hopcroft's
  rule, in :math:`O(m \\log n)` whatever the diameter of the networks (vectorized rounds are
  faster only on a GPU, where they run up to 32 rounds before the compiled code takes over).
* The exact answer to isomorphism comes from individualization-refinement, the strategy of
  nauty, bliss and Traces (McKay and Piperno, 2014): after color refinement, a node of ``net2``
  in a color class with more than one node per network is given a color of its own together with
  each node of ``net1`` of its class, in turn, and the colors are refined again, until every class
  holds one node of each network, a mapping that is then checked. A class whose individualization
  splits nothing else is a class of twins (nodes with the same neighbors) and is matched at once,
  without trying alternatives. The search is compiled (Numba) and the candidates of the first
  node are tried in parallel.
* The exact answer to subgraph questions searches a mapping depth first, in the style of VF2++
  and VF3 (Jüttner and Madarasi, 2018; Carletti et al., 2017): pattern nodes are matched in an
  order that keeps every new node connected to the matched ones (most matched neighbors, then
  fewest candidates, then highest degree), the candidates of a node are the neighbors of the
  image of a matched neighbor that passed the filter above, and every candidate is checked
  against the matched nodes. The search is compiled and runs in parallel over the candidate
  images of the first node (as the parallel VF3 of Carletti et al., 2019).

Node attributes and edge attributes listed by name must be equal on matched nodes and edges.
Edge directions always count; self loops are attributes of their node; parallel edges are one
edge whose attributes are the number of copies and the attribute values of each copy (pairs of
parallel edges must match exactly, also in monomorphisms). In multilayer networks the layer of
every node copy must match as well.
"""

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from ._policy import _higher_order
from . import _isomorphism as _k
from .network import _quiet_sparse

# splitmix64 constants, as signed 64-bit integers
_GOLDEN = 0x9E3779B97F4A7C15 - 2 ** 64
_MIX1 = 0xBF58476D1CE4E5B9 - 2 ** 64
_MIX2 = 0x94D049BB133111EB - 2 ** 64

#: Largest number of (pattern node, target node) pairs of the dense candidate filter.
_FILTER_SIZE = 2 ** 28

#: Most rounds of color refinement on GPU tensors, before the compiled refinement takes over.
_ROUNDS = 32


def _srl(x, k):
   """
   Logical right shift of int64 tensors.
   """
   return (x >> k) & ((1 << (64 - k)) - 1)


def _mix(x):
   """
   splitmix64 hash of an int64 tensor, element by element (deterministic, so colors are
   comparable across calls).
   """
   x = x + _GOLDEN
   x = (x ^ _srl(x, 30)) * _MIX1
   x = (x ^ _srl(x, 27)) * _MIX2
   return x ^ _srl(x, 31)


def _canonical(x):
   """
   Float tensor as int64 keys, equal for equal values (NaN equal to NaN, -0.0 to 0.0).
   """
   x = x.double()
   x = torch.where(x == 0, torch.zeros_like(x), x)
   x = torch.where(torch.isnan(x), torch.full_like(x, float("nan")), x)
   return x.contiguous().view(torch.int64)


def _row_codes(cols):
   """
   Codes ``0, 1, ...`` of the distinct rows of some int64 columns (1-dimensional sorts only,
   much faster than ``torch.unique(dim = 0)``).
   """
   code = None
   for x in cols:
      x = torch.unique(x, return_inverse = True)[1].reshape(-1)
      if code is not None and len(x):
         x = torch.unique(code * (int(x.max()) + 1) + x, return_inverse = True)[1].reshape(-1)
      code = x
   return code


def _arcs(net, edge_attributes):
   """
   All the stored arcs of a network (both directions of undirected edges), with node ids in the
   network wide order, and the selected edge attributes.

   :return: ``(sources, targets, attributes)``: sources and targets on the device of the network,
      and one vector per attribute over the arcs (float64 tensor, or NumPy object array for a
      qualitative attribute).
   """
   s, t = [], []
   for c, d in enumerate(net.components):
      o = int(net.node_offsets[c])
      s.append(d.edge_index[0] + o)
      t.append(d.edge_index[1] + o)
   s, t = torch.cat(s), torch.cat(t)
   X = []
   for a in edge_attributes:
      if not len(s):   # an edgeless network may lack the attribute
         X.append(torch.zeros(0, dtype = torch.float64, device = s.device))
      elif a in net.qualitative_edge_attributes:
         X.append(np.concatenate(net.get_edge_attribute(a, None)))
      else:
         X.append(torch.cat(net.get_edge_attribute(a, None)).double())
   return s, t, X


def _node_values(net, key):
   """
   Values of a node attribute over all the stored nodes, in the network wide order, as a NumPy
   object array; in long format, the set of the values of every node (multi-valued attributes).
   """
   if key in net.qualitative_node_attributes and net.node_format == "long":
      Q = net.qualitative_node_attr
      Q = Q[(Q["attribute"] == key).to_numpy()]
      v = np.empty(int(net.node_offsets[-1]), dtype = object)
      v[:] = [frozenset()] * len(v)
      for node, vals in Q.groupby("node", observed = True)["value"]:
         v[node] = frozenset(vals.tolist())
      return v
   vals = [net._node_vector(key, c) for c in range(len(net.components))]   # all stored rows (hyperedges and facets of higher-order networks included)
   if key in net.qualitative_node_attributes:
      return np.concatenate(vals) if vals else np.empty(0, dtype = object)
   return np.concatenate([x.double().cpu().numpy() for x in vals]).astype(object)


def _node_keys(net1, net2, node_attributes):
   """
   One integer per node of the union of the two networks (``net1`` first), equal for nodes with
   equal attributes (and layers, in multilayer networks).
   """
   n1, n2 = net1.number_of_nodes(), net2.number_of_nodes()
   cols = [np.zeros(n1 + n2, dtype = np.int64)]
   for a in node_attributes:
      v = np.concatenate([_node_values(net1, a), _node_values(net2, a)])
      cols.append(pd.factorize(v, use_na_sentinel = False)[0].astype(np.int64))
   if net1.layer_labels is not None:
      lay = [np.asarray(net.layer_labels, dtype = object)[np.concatenate([d.layer.cpu().numpy() for d in net.components]).astype(np.int64)] for net in (net1, net2)]
      cols.append(pd.factorize(np.concatenate(lay), use_na_sentinel = False)[0].astype(np.int64))
   return _row_codes([torch.as_tensor(c) for c in cols]).numpy()


def _pair_graph(net1, net2, node_attributes, edge_attributes):
   """
   The union of the two networks (nodes of ``net1`` first) as a pair graph: one arc per ordered
   pair of distinct adjacent nodes, both ways, labeled ``forward * K + backward`` by the codes of
   the arcs between them (number of parallel copies and their attribute values; 0 if absent),
   plus the code of the self loops of every node and the node keys.

   :return: Dictionary with ``n1``, ``n2``, ``s``, ``t``, ``lab`` (sorted by source, then target,
      on the device of ``net1``), ``K``, ``loop`` and ``key`` (per node).
   """
   if net1.directed != net2.directed:
      raise ValueError("isomorphism: one network is directed and the other is not")
   if (net1.layer_labels is None) != (net2.layer_labels is None):
      raise ValueError("isomorphism: one network is multilayer and the other is not")
   node_attributes = [node_attributes] if isinstance(node_attributes, str) else list(node_attributes or [])
   edge_attributes = [edge_attributes] if isinstance(edge_attributes, str) else list(edge_attributes or [])
   dev = net1.components[0].edge_index.device if net1.components else torch.device("cpu")
   n1, n2 = net1.number_of_nodes(), net2.number_of_nodes()
   n = n1 + n2
   s1, t1, X1 = _arcs(net1, edge_attributes)
   s2, t2, X2 = _arcs(net2, edge_attributes)
   s = torch.cat([s1.to(dev), s2.to(dev) + n1])
   t = torch.cat([t1.to(dev), t2.to(dev) + n1])
   X = []   # one int64 key per attribute and arc, equal for equal values in both networks
   for x1, x2 in zip(X1, X2):
      if torch.is_tensor(x1) and torch.is_tensor(x2):
         X.append(_canonical(torch.cat([x1.to(dev), x2.to(dev)])))
      else:
         v = pd.Series([*(x1.cpu().tolist() if torch.is_tensor(x1) else x1), *(x2.cpu().tolist() if torch.is_tensor(x2) else x2)], dtype = object)
         X.append(torch.as_tensor(pd.factorize(v, use_na_sentinel = False)[0].astype(np.int64), device = dev))
   pair = s * n + t
   if not X:   # no attributes: the code of a pair is its number of copies
      pair, code = torch.unique_consecutive(torch.sort(pair).values, return_counts = True)
      s, t = pair // n, pair % n
   else:   # sorted by pair, then attributes
      r = _row_codes(X)
      idx = torch.argsort(r, stable = True)
      idx = idx[torch.argsort(pair[idx], stable = True)]
      s, t, r, pair = s[idx], t[idx], r[idx], pair[idx]
      new = torch.ones(len(s), dtype = torch.bool, device = dev)
      new[1:] = (pair[1:] != pair[:-1]) | (r[1:] != r[:-1])
      cnt = torch.bincount(torch.cumsum(new.long(), 0) - 1)   # copies in runs of equal (pair, attributes)
      first = torch.nonzero(new).reshape(-1)
      ps, pt, pr, pp = s[first], t[first], r[first], pair[first]
      gnew = torch.ones(len(pp), dtype = torch.bool, device = dev)
      gnew[1:] = pp[1:] != pp[:-1]
      if bool(gnew.all()):   # one kind of copy per pair: code of (attributes, copies)
         code = _row_codes([pr, cnt]) + 1
      else:   # parallel copies with different attributes: code of the multiset, in Python
         group = torch.cumsum(gnew.long(), 0) - 1
         key = {}
         code, items, g0 = [], [], 0
         for g, a, b in list(zip(group.tolist(), pr.tolist(), cnt.tolist())) + [(-1, 0, 0)]:
            if g != g0 and items:
               code.append(key.setdefault(tuple(items), len(key) + 1))
               items = []
            items.append((a, b))
            g0 = g
         code = torch.tensor(code, dtype = torch.long, device = dev)
      first = torch.nonzero(gnew).reshape(-1)
      s, t = ps[first], pt[first]
   K = int(code.max()) + 1 if len(code) else 1
   loop = torch.zeros(n, dtype = torch.long, device = dev)
   lp = s == t
   if bool(lp.any()):
      loop[s[lp]] = code[lp]
      s, t, code = s[~lp], t[~lp], code[~lp]
   if net1.directed:   # both directions of every pair
      fw = torch.cat([code, torch.zeros_like(code)])
      a, b = torch.cat([s, t]), torch.cat([t, s])
      o = torch.argsort(a * n + b)
      a, b, fw = a[o], b[o], fw[o]
      bw = torch.cat([torch.zeros_like(code), code])[o]
      g = torch.ones(len(a), dtype = torch.bool, device = dev)
      g[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
      gi = torch.cumsum(g.long(), 0) - 1
      k = int(g.sum())
      lab = torch.zeros(k, dtype = torch.long, device = dev).index_add_(0, gi, fw) * K + torch.zeros(k, dtype = torch.long, device = dev).index_add_(0, gi, bw)
      s, t = a[g], b[g]
   else:
      lab = code * K
   key = torch.as_tensor(_node_keys(net1, net2, node_attributes), device = dev)
   return dict(n1 = n1, n2 = n2, s = s, t = t, lab = lab, K = K, loop = loop, key = key, directed = net1.directed)


def _refine(G, rounds = _ROUNDS):
   """
   Color refinement (1-dimensional Weisfeiler-Leman) on the union of the two networks, on their
   device: the new color of a node hashes its color with the sum of the hashes of the colors of
   its neighbors, each with the label of the pair (so with the direction of the arcs). Hashes are
   deterministic 64-bit (splitmix64): collisions are vanishingly unlikely.

   :param rounds: Most rounds (``None``: until stable).
   :return: ``(colors, same, stable)``: the colors, whether the two networks have the same number
      of nodes of every color (the refinement stops as soon as they do not), and whether the
      colors are stable.
   """
   n1, s, t = G["n1"], G["s"], G["t"]
   col = _mix(_mix(G["key"]) + G["loop"])
   lab = _mix(G["lab"] + 1)
   k = len(torch.unique(col))
   r = 0
   while True:
      if not torch.equal(torch.sort(col[:n1]).values, torch.sort(col[n1:]).values):
         return col, False, False
      if r == rounds:
         return col, True, False
      agg = torch.zeros_like(col).index_add_(0, s, _mix(col[t] ^ lab))
      new = _mix(col * 31 + agg)
      k2 = len(torch.unique(new))
      if k2 == k:
         return col, True, True
      col, k = new, k2
      r += 1


def _union_csr(G):
   """
   CSR arrays (NumPy) of the pair graph of the union of the two networks, with the hash of the
   label of every arc as seen from its target.

   :return: ``(ptr, nbr, lab, harc)``.
   """
   N, K = G["n1"] + G["n2"], G["K"]
   nbr, lab = G["t"].cpu().numpy().astype(np.int64), G["lab"].cpu().numpy().astype(np.int64)
   ptr = np.zeros(N + 1, dtype = np.int64)
   np.cumsum(np.bincount(G["s"].cpu().numpy(), minlength = N), out = ptr[1:])
   return ptr, nbr, lab, _mix(torch.as_tensor((lab % K) * K + lab // K + 1)).numpy()


def _partition(G, col):
   """
   Stable partition of the union of the two networks (compiled color refinement with Hopcroft's
   rule, :math:`O(m \\log n)` whatever the diameter), from colors.

   :return: ``(union CSR, partition)``, or ``None`` if some color class is unbalanced (the
      networks are not isomorphic).
   """
   n1 = G["n1"]
   N = 2 * n1
   csr = _union_csr(G)
   part = _k._new_partition(n1, N, _row_codes([col, G["key"], G["loop"]]).cpu().numpy())   # keys exact
   elem, pos, cell, cstart, csize, c1, parent, st, qbuf, inq = part
   if not _k._balanced(csize, c1, st):
      return None
   if not _k._refine_cells(n1, *csr[:2], csr[3], *part, np.zeros(N, np.int64), np.zeros(N, np.bool_), np.empty(N + 1, np.int64), np.zeros(N + 1, np.int64), np.empty(N + 1, np.int64), np.empty(N + 1, np.int64)):
      return None
   return csr, part


def _filter(G, mode):
   """
   Candidates of every pattern node (nodes of ``net2``) among the target nodes (``net1``): same
   key, compatible self loops, at least the same number of neighbors (successors and
   predecessors in directed networks); then, repeatedly, a candidate ``v`` of ``u`` is dropped if
   some neighbor of ``u`` has no candidate among the neighbors of ``v`` (sparse-dense matrix
   products, on the device of the network).

   :return: Boolean ``n2`` by ``n1`` tensor, or ``None`` if larger than :data:`_FILTER_SIZE`.
   """
   n1, n2, s, t, K = G["n1"], G["n2"], G["s"], G["t"], G["K"]
   if n1 * n2 > _FILTER_SIZE:
      return None
   dev = s.device
   n = n1 + n2
   dirs = [G["lab"] // K != 0, G["lab"] % K != 0] if G["directed"] else [torch.ones(len(s), dtype = torch.bool, device = dev)]
   key, loop = G["key"], G["loop"]
   C = key[n1:, None] == key[None, :n1]
   C &= (loop[n1:, None] == loop[None, :n1]) if mode == 1 else (loop[n1:, None] == 0) | (loop[n1:, None] == loop[None, :n1])
   mats = []
   for m in dirs:
      deg = torch.bincount(s[m], minlength = n)
      C &= deg[None, :n1] >= deg[n1:, None]
      a, b = s[m], t[m]
      tm, pm = a < n1, a >= n1
      with _quiet_sparse():
         T = torch.sparse_coo_tensor(torch.stack([a[tm], b[tm]]), torch.ones(int(tm.sum()), device = dev), (n1, n1)).coalesce()
         P = torch.sparse_coo_tensor(torch.stack([a[pm] - n1, b[pm] - n1]), torch.ones(int(pm.sum()), device = dev), (n2, n2)).coalesce()
      mats.append((T, P, deg[n1:].float()))
   rows, cols = max(1, 2 ** 24 // max(n1, 1)), max(1, 2 ** 24 // n2)   # float temporaries of 64 MB
   while True:
      before = int(C.sum())
      for T, P, deg in mats:
         M = torch.empty_like(C)
         for i in range(0, n2, rows):   # M[w, v]: some neighbor of v is a candidate of w
            M[i:i + rows] = torch.sparse.mm(T, C[i:i + rows].T.float()).T > 0
         for j in range(0, n1, cols):   # all neighbors of u supported at v
            C[:, j:j + cols] &= torch.sparse.mm(P, M[:, j:j + cols].float()) == deg[:, None]
      if int(C.sum()) == before:
         return C


def _csr(G, side):
   """
   CSR arrays (NumPy, on the CPU) of one of the two networks of the union: 0 for ``net1``, 1 for
   ``net2``.
   """
   n1, n2 = G["n1"], G["n2"]
   s, t, lab = G["s"], G["t"], G["lab"]
   m = s < n1 if side == 0 else s >= n1
   o, n = (0, n1) if side == 0 else (n1, n2)
   s, t, lab = (s[m] - o).cpu().numpy(), (t[m] - o).cpu().numpy(), lab[m].cpu().numpy()
   ptr = np.zeros(n + 1, dtype = np.int64)
   np.cumsum(np.bincount(s, minlength = n), out = ptr[1:])
   return ptr, t.astype(np.int64), lab.astype(np.int64)


def _setup(net1, net2, subgraph, induced, exact, node_attributes, edge_attributes):
   """
   Common part of :func:`is_isomorphic` and :func:`isomorphisms`: the pair graph, the cheap
   necessary conditions and the approximate test.

   :return: ``(answer, G, mode, partition or candidates)``; ``answer`` is ``False`` when a
      necessary condition fails, ``True`` when the approximate test passes and ``exact`` is
      false, ``None`` when the search must decide.
   """
   G = _pair_graph(net1, net2, node_attributes, edge_attributes)
   n1, n2, s = G["n1"], G["n2"], G["s"]
   m1 = int((s < n1).sum())
   m2 = len(s) - m1
   if not subgraph:
      if n1 != n2 or m1 != m2:
         return False, G, 0, None
      if n1 == 0:
         return True, G, 0, None
      col, same, stable = _refine(G, _ROUNDS if s.is_cuda else 0)   # on the CPU, compiled refinement only
      if not same:
         return False, G, 0, None
      if not exact and stable:
         return True, G, 0, None
      P = _partition(G, col)
      if P is None or not exact:
         return P is not None, G, 0, None
      return None, G, 0, P
   mode = 1 if induced else 2
   if n2 > n1 or m2 > m1:
      return False, G, mode, None
   if n2 == 0:
      return True, G, mode, None
   C = _filter(G, mode)
   if C is not None:
      if not bool(C.any(1).all()):
         return False, G, mode, C
      if not exact:
         r, c = torch.nonzero(C, as_tuple = True)
         B = csr_matrix((np.ones(len(r), dtype = np.int8), (r.cpu().numpy(), c.cpu().numpy())), shape = (n2, n1))
         return bool((maximum_bipartite_matching(B, perm_type = "column") >= 0).all()), G, mode, C
   elif not exact:   # too large for the filter: keys and degrees only
      key = G["key"].cpu().numpy()
      a = pd.Series(key[:n1]).value_counts()
      b = pd.Series(key[n1:]).value_counts()
      return bool((b <= a.reindex(b.index).fillna(0)).all()), G, mode, None
   return None, G, mode, C


def _run_iso(G, P, first, limit):
   """
   Exact isomorphism search from the stable partition of the union of the two networks:
   individualization-refinement (existence), or individualization of the first node of the
   matching order and depth first search (enumeration).

   :param P: Union CSR and partition, from :func:`_partition`.
   :param first: Stop at the first isomorphism.
   :param limit: Most isomorphisms to return (``None``: all).
   :return: As :func:`_run`.
   """
   import numba
   n1 = G["n1"]
   (ptr, nbr, lab, harc), (elem, pos, cell, cstart, csize, c1, parent, st, qbuf, inq) = P
   part = (elem, pos, cell, cstart, csize, c1, parent, st)
   if first:
      stop = np.zeros(1, dtype = np.int64)
      sizes = csize[:st[0]]
      if not (sizes > 2).any():
         return int(_k._ir(n1, ptr, nbr, lab, harc, *part, qbuf, inq, -1, np.zeros(0, np.int64), stop) == 1), None
      c = np.flatnonzero(sizes > 2)[np.argmin(sizes[sizes > 2])]
      members = elem[cstart[c]:cstart[c] + csize[c]]
      x0, cands = members[members >= n1][0], np.sort(members[members < n1])
      r = _k._ir(n1, ptr, nbr, lab, harc, *part, qbuf, inq, x0, cands[:1], stop)
      if r == 1:
         return 1, None
      if r == 2 or len(cands) == 1:   # twins: one candidate is enough
         return 0, None
      chunks = min(len(cands) - 1, 4 * numba.get_num_threads())
      return int(_k._ir_parallel(n1, ptr, nbr, lab, harc, *part, x0, cands[1:], chunks)), None
   ptr1, nbr1, lab1 = _csr(G, 0)
   ptr2, nbr2, lab2 = _csr(G, 1)
   loop = G["loop"].cpu().numpy()
   order = _k._order(n1, ptr2, nbr2, csize[cell[n1:]] // 2, np.diff(ptr2))
   x0 = n1 + order[0]
   members = elem[cstart[cell[x0]]:cstart[cell[x0]] + csize[cell[x0]]]
   cands = np.sort(members[members < n1])
   args = (n1, ptr, nbr, lab, harc, *part, qbuf, inq, x0, cands, order, ptr1, nbr1, lab1, ptr2, nbr2, lab2, loop[:n1], loop[n1:], np.diff(ptr1), np.diff(ptr2))
   if limit is None:
      out = np.empty((_k._iso_enum(*args, np.zeros((0, n1), np.int64), -1), n1), dtype = np.int64)
      _k._iso_enum(*args, out, -1)
      return len(out), out
   out = np.empty((limit, n1), dtype = np.int64)
   k = _k._iso_enum(*args, out, limit)
   return k, out[:k]


def _run(G, mode, extra, first, limit):
   """
   Exact search of the mappings of ``net2`` into ``net1``.

   :param extra: Union CSR and partition (isomorphism, from :func:`_partition`) or candidate
      filter (subgraphs, may be ``None``).
   :param first: Stop at the first mapping.
   :param limit: Most mappings to return (``None``: all).
   :return: Number of mappings found, and the mappings (rows: images of the nodes of ``net2``,
      in its network wide order) if not ``first``.
   """
   if mode == 0:
      return _run_iso(G, extra, first, limit)
   n1, n2 = G["n1"], G["n2"]
   ptr1, nbr1, lab1 = _csr(G, 0)
   ptr2, nbr2, lab2 = _csr(G, 1)
   key = G["key"].cpu().numpy()
   if extra is None:
      F = np.zeros((0, 0), dtype = np.uint64)
   else:
      w = (n1 + 63) // 64
      F = np.zeros((n2, w * 8), dtype = np.uint8)
      F[:, :(n1 + 7) // 8] = np.packbits(extra.cpu().numpy(), axis = 1, bitorder = "little")
      F = F.view("<u8").astype(np.uint64)
   key1, key2 = key[:n1], key[n1:]
   loop = G["loop"].cpu().numpy()
   loop1, loop2 = loop[:n1], loop[n1:]
   deg1, deg2 = np.diff(ptr1), np.diff(ptr2)
   km = np.argsort(key1, kind = "stable")
   kptr = np.zeros(int(max(key.max(initial = 0), 0)) + 2, dtype = np.int64)
   np.cumsum(np.bincount(key1, minlength = len(kptr) - 1), out = kptr[1:])
   kmem = km.astype(np.int64)
   if extra is None:
      cand = np.bincount(key1, minlength = len(kptr) - 1)[key2].astype(np.int64)
   else:
      cand = extra.sum(1).cpu().numpy().astype(np.int64)
   order = _k._order(n2, ptr2, nbr2, cand, deg2)
   u = order[0]
   roots = kmem[kptr[key2[u]]:kptr[key2[u] + 1]]
   if extra is not None:
      roots = np.flatnonzero(extra[u].cpu().numpy()).astype(np.int64)
   args = (mode, G["K"], order, ptr1, nbr1, lab1, ptr2, nbr2, lab2, key1, key2, loop1, loop2, deg1, deg2, F, kptr, kmem)
   import numba
   chunks = max(1, min(len(roots), 4 * numba.get_num_threads()))
   empty = np.zeros((0, n2), dtype = np.int64)
   none = np.zeros(len(roots), dtype = np.int64)
   if first:
      return int(_k._search(roots, chunks, True, *args, none, empty).sum()), None
   if limit is not None:
      out = np.empty((limit, n2), dtype = np.int64)
      k = _k._search_limited(roots, limit, *args, out)
      return k, out[:k]
   cnt = _k._search(roots, chunks, False, *args, none, empty)
   offsets = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
   out = np.empty((int(cnt.sum()), n2), dtype = np.int64)
   _k._search(roots, chunks, False, *args, offsets, out)
   return len(out), out


@_higher_order()
def is_isomorphic(net1, net2, subgraph = False, induced = True, exact = True, node_attributes = None, edge_attributes = None):
   """
   Whether two networks are isomorphic or, with ``subgraph = True``, whether ``net1`` contains
   ``net2``: whether some subgraph of ``net1`` is isomorphic to ``net2``.

   With ``exact = True`` (default), the answer comes from a compiled, parallel search of a
   mapping of the nodes of ``net2`` onto nodes of ``net1``: individualization-refinement for
   isomorphism, a depth first search in the style of VF2++ / VF3 for subgraphs, with the
   candidates pruned by the approximate tests first (see the module description). The worst case
   is exponential, as for every known algorithm, but common networks are matched quickly.

   With ``exact = False``, the approximate test only: ``False`` is always right, and ``True``
   means that the networks passed a necessary condition, so that they are likely, not surely,
   isomorphic (``net1`` likely contains ``net2``). For isomorphism, the condition is that color
   refinement (1-dimensional Weisfeiler-Leman) gives both networks the same number of nodes of
   every color; for subgraphs, that every pattern node keeps candidates under neighborhood
   filtering and that distinct candidates can be assigned to all pattern nodes. Both are tensor
   operations, on the GPU if the networks are there. For subgraphs, the filter needs a dense
   matrix of ``n1 * n2`` booleans and is skipped above :math:`2^{28}` pairs, leaving a
   comparison of node attributes only (and, with ``exact = True``, a search without pruning).

   The networks must be both directed or both undirected, and both multilayer or not. Node
   attributes listed in ``node_attributes`` (numeric or qualitative, multi-valued in long format)
   must be equal on matched nodes, NaN matching NaN; edge attributes listed in
   ``edge_attributes`` (numeric or qualitative) must be equal on matched edges. Self loops must match (with
   ``induced = False``, a self loop of ``net1`` may be left out of the subgraph). Parallel edges
   are matched as a whole: same number of copies with the same attribute values. In multilayer
   networks every node copy must match a copy in the same layer (compared by layer label).
   Bipartite networks are matched as plain networks: pass :data:`~graphoskop.BIPARTITE` in
   ``node_attributes`` to match the node types too.

   :param net1: A :class:`~graphoskop.Network` (the larger network, with ``subgraph = True``).
   :param net2: A :class:`~graphoskop.Network` (the pattern, with ``subgraph = True``).
   :param subgraph: Whether to test whether ``net1`` contains ``net2`` instead of isomorphism.
   :param induced: With ``subgraph = True``, whether the subgraph of ``net1`` must be induced
      (all the edges of ``net1`` among the matched nodes are in ``net2``); otherwise ``net1`` may
      have more edges among them (monomorphism).
   :param exact: Whether to search a mapping (exact answer) or to run the approximate test only.
   :param node_attributes: Name or list of names of node attributes that must match.
   :param edge_attributes: Name or list of names of edge attributes that must match.
   :return: ``True`` or ``False``.
   :raises ValueError: If one network is directed (multilayer) and the other is not.
   :raises KeyError: If an attribute is missing.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      ring = gsk.chain_graph(6)
      ring = gsk.add_edges(ring, (5, 0))                        # a 6-cycle
      path = gsk.chain_graph(4)
      gsk.is_isomorphic(ring, ring)                             # True
      gsk.is_isomorphic(ring, path, subgraph = True)            # True: the 6-cycle contains P4
      gsk.is_isomorphic(path, ring, subgraph = True)            # False
      gsk.is_isomorphic(ring, path, exact = False)              # False (different sizes)
   """
   answer, G, mode, extra = _setup(net1, net2, subgraph, induced, exact, node_attributes, edge_attributes)
   if answer is not None:
      return answer
   return _run(G, mode, extra, True, None)[0] > 0


@_higher_order()
def isomorphisms(net1, net2, subgraph = False, induced = True, node_attributes = None, edge_attributes = None, max_matches = None):
   """
   The mappings of the nodes of ``net2`` onto nodes of ``net1`` found by the exact test of
   :func:`is_isomorphic`: isomorphisms or, with ``subgraph = True``, embeddings of ``net2`` (the
   pattern) into ``net1``. Every mapping is listed, so a subgraph of ``net1`` appears once for
   every automorphism of the pattern (a triangle six times), and highly symmetric networks have a
   huge number of isomorphisms: ``max_matches`` stops the search.

   All the mappings are found in parallel; with ``max_matches``, the first ones are found by a
   single thread, in a fixed order.

   :param net1: A :class:`~graphoskop.Network`.
   :param net2: A :class:`~graphoskop.Network`.
   :param subgraph: Whether to find embeddings of ``net2`` into ``net1`` instead of isomorphisms.
   :param induced: With ``subgraph = True``, whether the subgraphs must be induced.
   :param node_attributes: Name or list of names of node attributes that must match.
   :param edge_attributes: Name or list of names of edge attributes that must match.
   :param max_matches: Most mappings to return (``None``: all).
   :return: :class:`pandas.DataFrame` with columns ``match`` (0, 1, ...), ``net2`` and ``net1``:
      in mapping ``match``, node ``net2`` of ``net2`` goes to node ``net1`` of ``net1`` (input
      ids); one row per node of ``net2`` and mapping.
   :raises ValueError: If one network is directed (multilayer) and the other is not.

   Example:

   .. code-block:: python

      import pandas as pd
      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      triangle = gsk.load_network(pd.DataFrame({"source": ["a", "b", "c"], "target": ["b", "c", "a"]}))
      m = gsk.isomorphisms(net, triangle, subgraph = True)
      m["match"].nunique() // 6                                 # number of triangles of net
   """
   answer, G, mode, extra = _setup(net1, net2, subgraph, induced, True, node_attributes, edge_attributes)
   n2 = G["n2"]
   if answer is False:
      out = np.zeros((0, n2), dtype = np.int64)
   elif answer is True:   # empty pattern
      out = np.zeros((1, 0), dtype = np.int64)
   else:
      out = _run(G, mode, extra, False, max_matches)[1]
   ids1 = np.concatenate(net1.node_ids) if net1.node_ids else np.empty(0, dtype = object)
   ids2 = np.concatenate(net2.node_ids) if net2.node_ids else np.empty(0, dtype = object)
   k = len(out)
   return pd.DataFrame({"match": np.repeat(np.arange(k), n2), "net2": np.tile(ids2, k), "net1": ids1[out.reshape(-1)]})
