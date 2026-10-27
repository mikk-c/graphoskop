"""
Frequent subgraph mining: the labeled subgraphs (patterns) that occur in many graphs of a
database.

In transactional mining (:func:`transactional_graph_mining`) the database is a network and each
of its connected components is a graph (a transaction); the support of a pattern is the number of
components that contain it (as a subgraph, not necessarily induced), and a pattern is frequent if
its support reaches a threshold. In single graph mining (:func:`single_graph_mining`) the network
is analyzed as a whole, and the support of a pattern is its minimum image based support (MNI): the
least number of distinct network nodes onto which one of its nodes is mapped. Nodes and edges may
carry labels, taken from a node attribute and an edge attribute (or the layer, in multilayer
networks), and patterns must match them.

The search is gSpan (Yan and Han, 2002). A pattern is written as a DFS code, the list of its edges
in the order of a depth first visit, each with the labels of its endpoints; among the codes of a
pattern, the lexicographically smallest one, the minimum DFS code, is its canonical form. Patterns
grow one edge at a time, only from the end of their code (rightmost extensions: a new node attached
to the rightmost path of the visit, or an edge back from its last node), so that every pattern has
a single parent, the pattern of its code minus the last edge, and the search space is a tree:

* the search starts from single labeled nodes and single edges; node labels and edge types (node
  label, edge label, node label) that are not frequent are removed from the database first, so no
  pattern containing them is ever built;
* in transactional mining, every pattern keeps the list of its embeddings (where it occurs, node
  by node) in the database; its extensions and their supports are all counted in one vectorized
  pass over these embeddings, without any subgraph isomorphism test (single graph mining keeps
  the valid images of each pattern node instead, see :func:`single_graph_mining`);
* an extension that is not frequent is dropped there and never extended: support is
  anti-monotone, so none of its super-patterns can be frequent. A super-pattern reached from
  another, frequent parent is counted in that parent's pass and dropped in the same way;
* an extension whose code is not the minimum DFS code of its pattern is a duplicate of a pattern
  found (or to be found) through its canonical parent, and is dropped with its whole subtree.

A frequent pattern is maximal if no frequent pattern contains it: if no extension by one edge,
anywhere in the pattern, is frequent. Frequent patterns that differ by one edge give association
rules (:func:`graph_association_rules`), which can score node pairs for link prediction
(:func:`~graphoskop.association_rules_lp`).
"""

import math

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ._policy import _higher_order
from . import _mining as _k
from .isomorphism import _node_values
from .network import LAYER, load_network


def _factorize(values):
   """
   Integer codes of values, NaN being a value of its own.

   :return: ``(codes, distinct values)``.
   """
   codes, uniques = pd.factorize(pd.Series(values, dtype = object) if values.dtype == object else values, use_na_sentinel = False)
   return codes.astype(np.int64), np.asarray(uniques, dtype = object)


def _database(net, node_label, edge_label):
   """
   The graph database of a network: one graph per component, or, for a multilayer network with
   the layer as edge label, one graph per component of the network of entities and intra-layer
   edges (each entity a node, each layer an edge label: a multigraph). Self loops are left out;
   parallel edges with the same label are one edge.

   :return: Dictionary with ``gid`` (graph of every node), ``lab`` (node label codes),
      ``nvalues`` (node label values), ``evalues`` (edge label values), the edges ``a``, ``b``,
      ``l`` (label codes; ``a -> b`` if directed), ``directed`` and ``ngraphs``.
   """
   directed = net.directed
   offsets = net.node_offsets
   s = np.concatenate([d.edge_index[0].cpu().numpy() + offsets[c] for c, d in enumerate(net.components)]) if net.components else np.zeros(0, np.int64)
   t = np.concatenate([d.edge_index[1].cpu().numpy() + offsets[c] for c, d in enumerate(net.components)]) if net.components else np.zeros(0, np.int64)
   n = net.number_of_nodes()
   values = _node_values(net, node_label) if node_label is not None else np.zeros(n, dtype = object)
   if net.layer_labels is not None and edge_label == LAYER:
      ent = np.concatenate([d.entity.cpu().numpy() for d in net.components]).astype(np.int64)
      lay = np.concatenate([d.layer.cpu().numpy() for d in net.components]).astype(np.int64)
      keep = lay[s] == lay[t]
      s, t, l = ent[s[keep]], ent[t[keep]], lay[s[keep]]
      ne = len(net.entity_ids)
      codes, nvalues = _factorize(values)
      lab = np.full(ne, -1, np.int64)
      lab[ent] = codes
      if (lab[ent] != codes).any():
         e = ent[np.flatnonzero(lab[ent] != codes)[0]]
         raise ValueError(f"entity {net.entity_ids[e]!r} has different values of {node_label!r} in different layers")
      evalues = np.arange(len(net.layer_labels), dtype = np.int64).astype(object)
      gid = connected_components(csr_matrix((np.ones(len(s)), (s, t)), shape = (ne, ne)), directed = False)[1].astype(np.int64)
   else:
      lab, nvalues = _factorize(values)
      if edge_label is None:
         l, evalues = np.zeros(len(s), np.int64), np.array([None], dtype = object)
      else:
         x = net.get_edge_attribute(edge_label, None)
         x = np.concatenate(x) if edge_label in net.qualitative_edge_attributes else np.concatenate([v.double().cpu().numpy() for v in x])
         l, evalues = _factorize(x)
      gid = np.repeat(np.arange(len(net.components)), np.diff(offsets)).astype(np.int64)
   E = pd.DataFrame({"a": s, "b": t, "l": l})
   E = E[E["a"] != E["b"]]
   if not directed:
      E = pd.DataFrame({"a": np.minimum(E["a"], E["b"]), "b": np.maximum(E["a"], E["b"]), "l": E["l"]})
   E = E.drop_duplicates()
   return dict(gid = gid, lab = lab, nvalues = nvalues, evalues = evalues, a = E["a"].to_numpy(np.int64), b = E["b"].to_numpy(np.int64), l = E["l"].to_numpy(np.int64), directed = directed, ngraphs = int(gid.max()) + 1 if len(gid) else 0)


def _rightmost_path(code):
   """
   Rightmost path of a DFS code: the vertices from the root to the last discovered one.
   """
   parent = {}
   for i, j, _, _, _ in code:
      if j > i:
         parent[j] = i
   v = max(parent) if parent else 0
   path = [v]
   while v != 0:
      v = parent[v]
      path.append(v)
   return path[::-1]


def _is_min(code, directed):
   """
   Whether a DFS code is the minimum DFS code of its pattern (see :func:`_canonical`).

   :param code: List of ``(i, j, label of i, oriented edge label, label of j)``; the oriented edge
      label is ``2 * label + d``, with ``d = 1`` if the edge goes from ``j`` to ``i`` (directed).
   :param directed: Whether edges are directed.
   """
   k = max(max(e[0], e[1]) for e in code) + 1
   lab = [0] * k
   edges = []
   for i, j, li, el, lj in code:
      lab[i], lab[j] = li, lj
      edges.append((i, j, el))
   return _canonical(lab, edges, directed, code) is not None


def _canonical(lab, edges, directed, target = None):
   """
   Minimum DFS code of a connected pattern, built edge by edge (backward edges before forward
   ones, backward edges to earlier vertices first, forward edges from deeper vertices first, then
   by labels). Parallel edges with different labels are different edges.

   :param lab: Label of every node.
   :param edges: Edges ``(a, b, oriented label from a)``.
   :param directed: Whether edges are directed.
   :param target: A DFS code to check: the construction stops at its first difference.
   :return: The minimum code (a list), or ``None`` if it differs from ``target``.
   """
   k = len(lab)
   adj = [[] for _ in range(k)]
   for eid, (a, b, el) in enumerate(edges):
      adj[a].append((b, el, eid))
      adj[b].append((a, el ^ 1 if directed else el, eid))
   first = min((lab[a], el, lab[b]) for a in range(k) for b, el, _ in adj[a])
   if target is not None and first != tuple(target[0][2:]):
      return None
   code = [(0, 1) + first]
   embs = [([a, b], {eid}) for a in range(k) for b, el, eid in adj[a] if (lab[a], el, lab[b]) == first]
   for step in range(1, len(edges)):
      R = _rightmost_path(code)
      rm, nv = R[-1], max(max(e[0], e[1]) for e in code) + 1
      best, ext = None, []
      for m, used in embs:   # backward edges from the rightmost vertex
         for b, el, eid in adj[m[rm]]:
            if eid not in used and b in m:
               j = m.index(b)
               if j in R[:-1]:
                  key = (j, el)
                  if best is None or key < best:
                     best, ext = key, []
                  if key == best:
                     ext.append((m, used, eid, None))
      if best is not None:
         m0 = ext[0][0]
         elem = (rm, best[0], lab[m0[rm]], best[1], lab[m0[best[0]]])
      else:
         for v in reversed(R):   # forward edges, deepest source first
            for m, used in embs:
               for b, el, eid in adj[m[v]]:
                  if b not in m:
                     key = (el, lab[b])
                     if best is None or key < best:
                        best, ext = key, []
                     if key == best:
                        ext.append((m, used, eid, b))
            if best is not None:
               elem = (v, nv, lab[ext[0][0][v]], best[0], best[1])
               break
      if target is not None and elem != tuple(target[step]):
         return None
      code.append(elem)
      embs = [(m + [b] if b is not None else m, used | {eid}) for m, used, eid, b in ext]
   return code


class _Miner:
   """
   gSpan over a graph database (see :func:`_database`), with embeddings kept as arrays: ``G``
   (graph of each embedding) and ``M`` (database node of each pattern vertex, one row per
   embedding).
   """

   def __init__(self, db, min_support, max_edges, maximal):
      self.db, self.min_support, self.max_edges, self.maximal = db, min_support, max_edges, maximal
      self.directed = db["directed"]
      self.patterns = []   # (code, support, maximal)
      lab, gid = db["lab"], db["gid"]
      a, b, l = db["a"], db["b"], db["l"]
      # oriented entries: x -> y with label 2 l + d (d = 1: the edge goes from y to x)
      x = np.concatenate([a, b])
      y = np.concatenate([b, a])
      el = np.concatenate([2 * l, 2 * l + (1 if self.directed else 0)])
      self.nodesup = self._support(lab, gid)
      ok = self.nodesup[lab] >= min_support
      keep = ok[x] & ok[y]
      x, y, el = x[keep], y[keep], el[keep]
      tkey = (lab[x] * (int(el.max(initial = 0)) + 1) + el) * (int(lab.max(initial = 0)) + 1) + lab[y]   # edge type
      tcode = np.unique(tkey, return_inverse = True)[1].reshape(-1) if len(tkey) else tkey
      keep = self._support(tcode, gid[x])[tcode] >= min_support if len(tcode) else np.zeros(0, bool)
      x, y, el = x[keep], y[keep], el[keep]
      o = np.lexsort((y, el, x))
      self.nbr, self.el = y[o], el[o]
      self.ptr = np.zeros(len(lab) + 1, np.int64)
      np.cumsum(np.bincount(x, minlength = len(lab)), out = self.ptr[1:])
      self.x = x[o]

   @staticmethod
   def _support(key, g):
      """
      Number of distinct graphs per key value (keys ``0`` to ``max``).
      """
      if len(key) == 0:
         return np.zeros(0, np.int64)
      pairs = np.unique(np.stack([key, g]), axis = 1)
      return np.bincount(pairs[0], minlength = int(key.max()) + 1)

   def run(self):
      lab, gid = self.db["lab"], self.db["gid"]
      frequent = np.flatnonzero(self.nodesup >= self.min_support)
      edge_labels = set(self.el.tolist())
      for c in frequent:   # single nodes; maximal unless some frequent edge touches the label
         touched = bool((lab[self.x] == c).any())
         self.patterns.append(([], int(self.nodesup[c]), not touched, int(c)))
      if self.max_edges == 0 or len(self.x) == 0:
         return self.patterns
      M = np.stack([self.x, self.nbr], 1)
      cols = (lab[self.x], self.el, lab[self.nbr])
      for (li, el, lj), e, sup in self._groups(cols, gid[self.x]):
         if sup >= self.min_support:
            code = [(0, 1, int(li), int(el), int(lj))]
            if _is_min(code, self.directed):
               self._grow(code, gid[self.x][e], M[e])
      return self.patterns

   def _groups(self, cols, g):
      """
      Groups of equal rows of ``cols`` (sorted lexicographically), with the support of each.

      :return: List of ``(row, indices, support)``.
      """
      if len(g) == 0:
         return []
      o = np.lexsort((g,) + tuple(reversed(cols)))
      C = np.stack([c[o] for c in cols], 1)
      new = np.ones(len(o), bool)
      new[1:] = (C[1:] != C[:-1]).any(1)
      newg = new.copy()
      newg[1:] |= g[o][1:] != g[o][:-1]
      starts = np.flatnonzero(new)
      sup = np.add.reduceat(newg.astype(np.int64), starts)
      ends = np.append(starts[1:], len(o))
      return [(tuple(C[s]), o[s:e], int(k)) for s, e, k in zip(starts, ends, sup)]

   def _extensions(self, code, G, M, rightmost):
      """
      Extensions of a pattern by one edge, with their supports, from its embeddings.

      With ``rightmost``, only rightmost extensions, in DFS code order; otherwise all of them
      (any vertex, for maximality).

      :return: List of ``(code element, embedding indices, new database nodes or None, support)``.
      """
      lab = self.db["lab"]
      k = M.shape[1]
      R = _rightmost_path(code)
      rm = R[-1]
      plab = [0] * k
      for i, j, li, el, lj in code:
         plab[i], plab[j] = li, lj
      L = 2 * (int(self.el.max(initial = 0)) // 2 + 1)
      has = set()   # edges of the pattern: (source, target, label), sources < targets if undirected
      for i, j, li, el, lj in code:
         a, b = (i, j) if el % 2 == 0 else (j, i)
         if not self.directed:
            a, b = min(a, b), max(a, b)
         has.add((a * k + b) * L + el // 2)
      last = code[-1]
      rows = []
      for i in (R if rightmost else range(k)):
         x = M[:, i]
         deg = self.ptr[x + 1] - self.ptr[x]
         e = np.repeat(np.arange(len(x)), deg)
         if len(e) == 0:
            continue
         ent = np.repeat(self.ptr[x] - np.concatenate([[0], np.cumsum(deg)[:-1]]), deg) + np.arange(len(e))
         w, el = self.nbr[ent], self.el[ent]
         eq = M[e] == w[:, None]
         inm = eq.any(1)
         j = eq.argmax(1)
         back = inm.copy()
         if rightmost:
            back &= (i == rm) & np.isin(j, R[:-1])
            if last[0] == rm and last[1] < rm:   # after a backward edge, only later ones
               back &= (j > last[1]) | ((j == last[1]) & (el > last[3]))
         else:
            back &= j != i
         if back.any():   # edges of the pattern are not new
            src = np.where(el % 2 == 0, i, j)
            dst = np.where(el % 2 == 0, j, i)
            if not self.directed:
               src, dst = np.minimum(src, dst), np.maximum(src, dst)
            back &= ~np.isin((src * k + dst) * L + el // 2, list(has))
         fwd = ~inm
         for mask, kind in ((back, 0), (fwd, 1)):
            if mask.any():
               c = int(mask.sum())
               rows.append((np.full(c, kind), np.full(c, i), j[mask] if kind == 0 else np.zeros(c, np.int64), el[mask], np.zeros(c, np.int64) if kind == 0 else lab[w[mask]], e[mask], w[mask]))
      if not rows:
         return []
      kind, src, j, el, lj, e, w = (np.concatenate(c) for c in zip(*rows))
      # gSpan order: backward (by target, label), then forward (deepest source first, by labels)
      cols = (kind, np.where(kind == 0, j, k - src), np.where(kind == 0, src, 0), el, lj)
      out = []
      for (kd, a, i, b, c), idx, sup in self._groups(cols, G[e]):
         if kd == 0:
            out.append(((int(i), int(a), plab[i], int(b), plab[a]), e[idx], None, sup))
         else:
            i = k - int(a)
            out.append(((i, k, plab[i], int(b), int(c)), e[idx], w[idx], sup))
      return out

   def _grow(self, code, G, M):
      """
      Records a frequent pattern with a minimum code and grows it, depth first.
      """
      sup = len(np.unique(G))
      ext = self._extensions(code, G, M, True)
      frequent = [x for x in ext if x[3] >= self.min_support]
      maximal = None
      if self.maximal:
         maximal = not frequent and not any(x[3] >= self.min_support for x in self._extensions(code, G, M, False))
      self.patterns.append((code, sup, maximal, None))
      if self.max_edges is not None and len(code) >= self.max_edges:
         return
      for elem, e, w, _ in frequent:
         new = code + [elem]
         if _is_min(new, self.directed):
            self._grow(new, G[e], M[e] if w is None else np.column_stack([M[e], w]))


class _SingleMiner:
   """
   gSpan-style search over a single network, with minimum image based support (MNI): the support
   of a pattern is the least number of distinct network nodes onto which one of its nodes is
   mapped by an embedding. The valid images of every node of a pattern are kept (``V``, ``k`` by
   ``n``); they are the candidate images of the same nodes in its extensions, whose MNI is then
   computed by searching an embedding for every remaining candidate (see :mod:`._mining`).
   """

   def __init__(self, db, min_support, max_edges, maximal):
      import numba
      self.db, self.tau, self.max_edges, self.maximal = db, min_support, max_edges, maximal
      self.directed = db["directed"]
      self.chunks = 4 * numba.get_num_threads()
      self.patterns = []
      lab = db["lab"]
      n = len(lab)
      a, b, l = db["a"], db["b"], db["l"]
      x = np.concatenate([a, b])
      y = np.concatenate([b, a])
      el = np.concatenate([2 * l, 2 * l + (1 if self.directed else 0)])
      self.count = np.bincount(lab, minlength = int(lab.max(initial = -1)) + 1)
      ok = self.count[lab] >= min_support
      keep = ok[x] & ok[y]
      x, y, el = x[keep], y[keep], el[keep]
      # edge types (label, oriented label, label) with their exact MNI: distinct sources, targets
      L = int(el.max(initial = 0)) + 1
      T = (lab[x] * L + el) * (int(lab.max(initial = 0)) + 1) + lab[y]
      sup = np.zeros(0, np.int64)
      if len(T):
         types, tcode = np.unique(T, return_inverse = True)
         tcode = tcode.reshape(-1)
         nx = np.bincount(np.unique(tcode * n + x) // n, minlength = len(types))
         ny = np.bincount(np.unique(tcode * n + y) // n, minlength = len(types))
         keep = np.minimum(nx, ny)[tcode] >= min_support
         x, y, el = x[keep], y[keep], el[keep]
      self.L = L
      key = y * L + el
      o = np.lexsort((key, x))
      self.x, self.nbr, self.el, self.key = x[o], y[o], el[o], key[o]
      self.ptr = np.zeros(n + 1, np.int64)
      np.cumsum(np.bincount(x, minlength = n), out = self.ptr[1:])
      self.lab, self.n = lab, n

   def run(self):
      lab = self.lab
      touched = set(lab[self.x].tolist())
      for c in np.flatnonzero(self.count >= self.tau):   # single nodes: MNI = number of nodes with the label
         self.patterns.append(([], int(self.count[c]), int(c) not in touched, int(c)))
      if self.max_edges == 0 or len(self.x) == 0:
         return self.patterns
      cols = (lab[self.x], self.el, lab[self.nbr])
      o = np.lexsort(tuple(reversed(cols)))
      seen = set()
      for r in o:
         li, el, lj = int(cols[0][r]), int(cols[1][r]), int(cols[2][r])
         if (li, el, lj) in seen:
            continue
         seen.add((li, el, lj))
         code = [(0, 1, li, el, lj)]
         if not _is_min(code, self.directed):
            continue
         D = np.stack([lab == li, lab == lj])
         sup, V = self._mni(code, D, True)
         if sup >= self.tau:
            self._grow(code, sup, V)
      return self.patterns

   def _pattern(self, code):
      """
      Pattern CSR of oriented entries, and its number of nodes.
      """
      k = max(max(e[0], e[1]) for e in code) + 1
      a, b, e = [], [], []
      for i, j, _, el, _ in code:
         a += [i, j]
         b += [j, i]
         e += [el, el ^ 1 if self.directed else el]
      a, b, e = np.array(a), np.array(b), np.array(e)
      o = np.argsort(a, kind = "stable")
      pptr = np.zeros(k + 1, np.int64)
      np.cumsum(np.bincount(a, minlength = k), out = pptr[1:])
      return k, pptr, b[o].astype(np.int64), e[o].astype(np.int64)

   def _mni(self, code, D, exact):
      """
      MNI of a pattern, from candidate domains (modified).

      :param exact: Whether to compute the exact support; otherwise only whether it reaches the
         threshold (the search of a node stops at the threshold).
      :return: ``(support, valid images)``; the support is 0 (and the images ``None``) once it is
         known to be below the threshold, and only reaches the threshold if not ``exact``.
      """
      k, pptr, pnbr, pel = self._pattern(code)
      _k._arc_consistency(D, pptr, pnbr, pel, self.ptr, self.nbr, self.el)
      size = D.sum(1)
      if (size < self.tau).any():
         return 0, None
      V = np.zeros_like(D)
      for u in np.argsort(size, kind = "stable"):
         order, par, pare = [u], [-1], [-1]   # breadth first from u
         for w in order:
            for f in range(pptr[w], pptr[w + 1]):
               if pnbr[f] not in order:
                  order.append(int(pnbr[f]))
                  par.append(int(w))
                  pare.append(int(pel[f]))
         order, par, pare = np.array(order, np.int64), np.array(par, np.int64), np.array(pare, np.int64)
         batch = max(256, self.chunks)
         while True:
            cand = np.flatnonzero(D[u] & ~V[u])
            got = int(V[u].sum())
            if not exact and got >= self.tau:
               break
            if got + len(cand) < self.tau:
               return 0, None
            if not len(cand):
               break
            cand = cand[:batch]
            _k._resolve(cand, order, par, pare, pptr, pnbr, pel, D, V, self.ptr, self.nbr, self.el, self.key, self.L, min(self.chunks, len(cand)))
            batch *= 2
      return int(V.sum(1).min()), V

   def _candidates(self, code, V, rightmost):
      """
      One-edge extensions of a pattern that can reach the threshold: their edge leaves (and
      reaches) enough distinct valid images. With ``rightmost``, gSpan's rightmost extensions in
      DFS code order; otherwise all of them.

      :return: List of ``(code element, initial domains)``.
      """
      lab, tau = self.lab, self.tau
      k = V.shape[0]
      R = _rightmost_path(code)
      rm = R[-1]
      plab = [0] * k
      for i, j, li, el, lj in code:
         plab[i], plab[j] = li, lj
      has = set()
      for i, j, _, el, _ in code:
         a, b = (i, j) if el % 2 == 0 else (j, i)
         has.add((a, b, el // 2) if self.directed else (min(a, b), max(a, b), el // 2))
      last = code[-1]
      back, fwd = [], []
      for i in (R if rightmost else range(k)):
         xs = np.flatnonzero(V[i])
         deg = self.ptr[xs + 1] - self.ptr[xs]
         ent = np.repeat(self.ptr[xs] - np.concatenate([[0], np.cumsum(deg)[:-1]]), deg) + np.arange(int(deg.sum()))
         x, y, el = self.x[ent], self.nbr[ent], self.el[ent]
         for j in (([j for j in R[:-1]] if i == rm else []) if rightmost else [j for j in range(k) if j != i]):
            m = V[j][y]
            for e in np.unique(el[m]):
               a, b = (i, j) if e % 2 == 0 else (j, i)
               if ((a, b, e // 2) if self.directed else (min(a, b), max(a, b), e // 2)) in has:
                  continue
               if rightmost and last[0] == rm and last[1] < rm and (j, e) <= (last[1], last[3]):
                  continue
               mm = m & (el == e)
               if len(np.unique(x[mm])) >= tau and len(np.unique(y[mm])) >= tau:
                  back.append(((i, j, plab[i], int(e), plab[j]), (j, e)))
         ly = lab[y]
         for e, l2 in sorted(set(zip(el.tolist(), ly.tolist()))):
            mm = (el == e) & (ly == l2)
            if len(np.unique(x[mm])) >= tau and len(np.unique(y[mm])) >= tau:
               fwd.append(((i, k, plab[i], int(e), int(l2)), (k - i, e, l2)))
      out = []
      for elem, _ in sorted(back, key = lambda z: z[1]) + sorted(fwd, key = lambda z: z[1]):
         D = V.copy() if elem[1] < k else np.vstack([V, (lab == elem[4])[None, :]])
         out.append((elem, D))
      return out

   def _grow(self, code, sup, V):
      """
      Records a frequent pattern with a minimum code, with its valid images, and grows it.
      """
      rec, at = [code, sup, None, None], len(self.patterns)
      self.patterns.append(rec)
      grown = False
      if self.max_edges is None or len(code) < self.max_edges:
         for elem, D in self._candidates(code, V, True):
            new = code + [elem]
            if not _is_min(new, self.directed):
               continue
            s, W = self._mni(new, D, True)
            if s >= self.tau:
               grown = True
               self._grow(new, s, W)
      if self.maximal:
         rec[2] = not grown and not any(self._mni(code + [elem], D, False)[0] >= self.tau for elem, D in self._candidates(code, V, False))
      self.patterns[at] = tuple(rec)


@_higher_order()
def single_graph_mining(net, min_support, node_label = None, edge_label = None, maximal = False, max_edges = None):
   """
   Frequent subgraphs of a single network: the connected labeled patterns whose minimum image
   based support (MNI; Bringmann and Nijssen, 2008) reaches ``min_support``.

   The network is mined as a whole. Counting the occurrences of a pattern is not anti-monotone (a
   star with three leaves occurs more often than the edge it contains, around a hub), so the
   support of a pattern is instead, over its nodes, the least number of distinct network nodes
   onto which the node is mapped by some embedding (an occurrence as a subgraph, not necessarily
   induced, with the same labels and directions). MNI never grows when a pattern grows, so the
   search can stop at infrequent patterns; it is never larger than the number of occurrences up to
   automorphisms, and is close to it when occurrences overlap little.

   Patterns grow as in :func:`transactional_graph_mining` (gSpan's minimum DFS codes and rightmost
   extensions, single labeled nodes first), with the same labels, multilayer collapse, options
   and output. Instead of storing every embedding, which explodes around hubs in a large network,
   every pattern keeps the set of valid images of each of its nodes, as in GraMi (Elseidy et al.,
   2014): an extension can map these nodes only onto them (its candidates), is skipped if its new
   edge leaves or reaches fewer distinct valid images than the threshold, has its candidates
   filtered by arc consistency, and then, for each node, each remaining candidate is checked by
   searching an embedding that maps the node onto it (a compiled search, candidates in parallel):
   every embedding found validates all its nodes at once, and the check stops as soon as the
   threshold is out of reach.

   :param net: A :class:`~graphoskop.Network`.
   :param min_support: Least MNI of a pattern, a positive integer.
   :param node_label: Name of the node attribute giving the node labels, or ``None``.
   :param edge_label: Name of the edge attribute giving the edge labels,
      :data:`~graphoskop.LAYER` in multilayer networks, or ``None``.
   :param maximal: Whether to report only maximal patterns: those contained in no other frequent
      pattern (whatever ``max_edges``).
   :param max_edges: Largest number of edges of a pattern (``None``: no limit).
   :return: ``(patterns, support)``, as :func:`transactional_graph_mining`, with ``support`` the
      MNI of each pattern.
   :raises ValueError: If ``min_support`` is not a positive integer, or if an entity has different
      node labels in different layers.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("citations.tsv", "papers.tsv", directed = True)
      patterns, support = gsk.single_graph_mining(net, 500, node_label = "field", max_edges = 4)
      support.sort_values("support", ascending = False).head()
   """
   if min_support < 1 or int(min_support) != min_support:
      raise ValueError(f"min_support must be a positive integer, got {min_support}")
   db = _database(net, node_label, edge_label)
   found = _SingleMiner(db, int(min_support), max_edges, maximal).run()
   return _output(db, found, maximal, node_label, edge_label)


@_higher_order()
def transactional_graph_mining(net, min_support, node_label = None, edge_label = None, maximal = False, max_edges = None):
   """
   Frequent subgraphs of a database of graphs, the connected components of a network: the
   connected labeled patterns that occur in at least ``min_support`` components, found with gSpan
   (see the module description).

   A pattern occurs in a component if some of its nodes and edges form a copy of the pattern (a
   subgraph, not necessarily induced), with the same node labels (the values of the node
   attribute ``node_label``), the same edge labels (the values of the edge attribute
   ``edge_label``) and, in directed networks, the same edge directions. Both attributes can be
   numeric or qualitative. Without ``node_label``
   (``edge_label``), all nodes (edges) have the same label; NaN is a label like any other.
   Patterns start from single labeled nodes and grow one edge at a time; every frequent pattern is
   reported once, with its support.

   In a multilayer network, ``edge_label =`` :data:`~graphoskop.LAYER` makes the layer the edge
   label: couplings and other interlayer edges are ignored, the copies of each entity collapse
   into one node, and an edge present in several layers becomes parallel edges with different
   labels (a labeled multigraph); the graphs of the database are then the connected components of
   this network of entities. ``node_label`` must then have the same value in all the copies of an
   entity. With another edge label, a multilayer network is mined as its supra-graph.

   Self loops are ignored, and parallel edges with the same label count as one edge. The number of
   frequent patterns can grow exponentially as ``min_support`` decreases; ``max_edges`` bounds
   the size of the patterns.

   :param net: A :class:`~graphoskop.Network`.
   :param min_support: Least number of components containing a pattern; a number between 0 and 1
      (excluded) is a fraction of the number of components.
   :param node_label: Name of the node attribute giving the node labels, or ``None``.
   :param edge_label: Name of the edge attribute giving the edge labels, :data:`~graphoskop.LAYER`
      in multilayer networks, or ``None``.
   :param maximal: Whether to report only maximal patterns: those contained in no other frequent
      pattern (whatever ``max_edges``).
   :param max_edges: Largest number of edges of a pattern (``None``: no limit).
   :return: ``(patterns, support)``: a :class:`~graphoskop.Network` with one component per
      pattern, directed if ``net`` is, with node ids ``0, 1, ...`` in component order and only the
      labels as attributes (node attribute ``node_label``, edge attribute ``edge_label``; with
      :data:`~graphoskop.LAYER`, the position of the layer in ``net.layer_labels``); and a
      :class:`pandas.DataFrame` with columns ``pattern`` (index of the component of ``patterns``)
      and ``support`` (number of components of ``net`` containing it). ``patterns`` is ``None``
      if no pattern is frequent.
   :raises ValueError: If ``min_support`` is not positive, or if an entity has different node
      labels in different layers.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      db = gsk.load_network("molecules.tsv", "atoms.tsv")        # one molecule per component
      patterns, support = gsk.transactional_graph_mining(db, 100, node_label = "element", edge_label = "bond")
      support.sort_values("support", ascending = False).head()
      patterns.number_of_nodes(3)                              # size of pattern 3
      big, support = gsk.transactional_graph_mining(db, 0.1, node_label = "element", maximal = True)
   """
   if min_support <= 0:
      raise ValueError(f"min_support must be positive, got {min_support}")
   db = _database(net, node_label, edge_label)
   if min_support < 1:
      min_support = math.ceil(min_support * db["ngraphs"])
   found = _Miner(db, int(math.ceil(min_support)), max_edges, maximal).run()
   return _output(db, found, maximal, node_label, edge_label)


def _output(db, found, maximal, node_label, edge_label):
   """
   Patterns found, as a network with one component per pattern and a table of supports.

   :param db: Graph database (see :func:`_database`).
   :param found: Patterns, as ``(code, support, maximal, label of a single node)``.
   :param maximal: Whether to keep the maximal patterns only.
   :param node_label: Name of the node label attribute, or ``None``.
   :param edge_label: Name of the edge label attribute, or ``None``.
   :return: ``(patterns, support)``, as returned by :func:`transactional_graph_mining`.
   """
   if maximal:
      found = [p for p in found if p[2]]
   if not found:
      return None, pd.DataFrame({"pattern": np.zeros(0, np.int64), "support": np.zeros(0, np.int64)})
   nname = node_label if node_label is not None else None
   ename = (LAYER if edge_label == LAYER else edge_label) if edge_label is not None else None

   def build(order):
      ids, src, tgt, elab, nlab, first = [], [], [], [], [], []
      nid = 0
      for p in order:
         code, _, _, single = found[p]
         labels = [single] if not code else [None] * (max(max(e[0], e[1]) for e in code) + 1)
         for i, j, li, el, lj in code:
            labels[i], labels[j] = li, lj
            a, b = (i, j) if el % 2 == 0 else (j, i)
            src.append(nid + a)
            tgt.append(nid + b)
            elab.append(db["evalues"][el // 2])
         first.append(nid)
         ids.extend(range(nid, nid + len(labels)))
         nlab.extend(db["nvalues"][c] for c in labels)
         nid += len(labels)
      E = pd.DataFrame({"source": np.array(src, np.int64), "target": np.array(tgt, np.int64)})
      if ename is not None:
         E[ename] = pd.Series(elab, dtype = object).infer_objects()   # numeric or qualitative, as in net
      N = pd.DataFrame({"id": np.array(ids, np.int64)})
      if nname is not None:
         N[nname] = pd.Series(nlab, dtype = object).infer_objects()
      return load_network(E, N, directed = db["directed"]), first

   pnet, first = build(range(len(found)))
   comp = [pnet.node_map[f][0] for f in first]
   order = list(np.argsort(comp, kind = "stable"))
   pnet, first = build(order)
   support = pd.DataFrame({"pattern": [pnet.node_map[f][0] for f in first], "support": [found[p][1] for p in order]})
   return pnet, support


def _pattern_keys(patterns):
   """
   Canonical key of every pattern of a pattern network: its minimum DFS code, or the label of its
   single node; labels are integer codes of the tuples of the values of all node (edge)
   attributes, the labels of the patterns.

   :return: ``(keys, structures, directed, ncode, ecode)``: one key and one ``(node labels,
      edges)`` per component, edges as ``(a, b, oriented label from a)``, and the dictionaries
      from label tuples to node and edge label codes.
   """
   directed = patterns.directed
   nnames = patterns.node_attributes + patterns.qualitative_node_attributes
   enames = patterns.edge_attributes + patterns.qualitative_edge_attributes
   clean = lambda v: None if isinstance(v, float) and v != v else v   # NaN as a value of its own
   ncode, ecode = {}, {}
   keys, structures = [], []
   for c in range(patterns.number_of_components()):
      cols = [patterns.get_node_attribute(a, c) for a in nnames]
      cols = [x.cpu().tolist() if hasattr(x, "cpu") else list(x) for x in cols]
      lab = [ncode.setdefault(tuple(clean(x[i]) for x in cols), len(ncode)) for i in range(patterns.number_of_nodes(c))]
      edges = []
      for row in patterns.edges(c, data = True):
         d = row[2]
         el = ecode.setdefault(tuple(clean(d[a]) for a in enames), len(ecode))
         edges.append((row[0], row[1], 2 * el))
      structures.append((lab, edges))
      keys.append(tuple(_canonical(lab, edges, directed)) if edges else ("node", lab[0]))
   return keys, structures, directed, ncode, ecode


def _one_edge_less(lab, edges, directed):
   """
   The connected patterns made by removing one edge from a pattern (and its endpoint, if left
   isolated).

   :return: Iterator of ``(keys, e, keep, sub)``: the canonical keys of the smaller pattern (two
      single nodes for a one-edge pattern), the position of the removed edge, the nodes kept and
      the remaining edges, renumbered over them.
   """
   for e, (a, b, el) in enumerate(edges):
      rest = edges[:e] + edges[e + 1:]
      if not rest:
         yield [("node", lab[a]), ("node", lab[b])], e, None, None
         continue
      deg = np.bincount(np.array([x for y in rest for x in y[:2]]), minlength = len(lab))
      keep = [v for v in range(len(lab)) if deg[v] > 0]
      new = {v: i for i, v in enumerate(keep)}
      sub = [(new[x], new[y], z) for x, y, z in rest]
      if len(keep) < len(lab) - 1 or not _connected(len(keep), sub):
         continue
      yield [tuple(_canonical([lab[v] for v in keep], sub, directed))], e, keep, sub


def _connected(k, edges):
   """
   Whether the nodes ``0`` to ``k - 1`` and the edges form a connected graph.
   """
   adj = [[] for _ in range(k)]
   for a, b, _ in edges:
      adj[a].append(b)
      adj[b].append(a)
   seen, stack = {0}, [0]
   while stack:
      for w in adj[stack.pop()]:
         if w not in seen:
            seen.add(w)
            stack.append(w)
   return len(seen) == k


@_higher_order()
def graph_association_rules(net, patterns, support, transactional = True):
   """
   Association rules between frequent patterns that differ by one edge: the smaller pattern ``P``
   (the antecedent) and the larger one ``Q`` (the consequent), ``Q`` being ``P`` with one more edge
   (between two of its nodes, or to a new node).

   Since ``Q`` contains ``P``, the support of the rule is the support of ``Q``, and its confidence
   :math:`\\sup(Q) / \\sup(P)` the share of the support of ``P`` that extends to ``Q`` (at most 1,
   support being anti-monotone). The lift compares this confidence with the relative support of
   the added edge alone, ``E``, the one-edge pattern with the labels (and direction) of the new
   edge and its endpoints: :math:`\\text{lift} = \\text{confidence} / (\\sup(E) / N)`, above 1 when
   ``P`` makes the new edge more likely than in general (when ``P`` is a single node, ``Q`` is
   ``E`` itself and the lift reduces to :math:`N / \\sup(P)`). ``N`` is the number of graphs of the
   database (components; components of the network of entities for a multilayer network mined
   with the layer as edge label) with ``transactional = True``, or the number of nodes (entities)
   for MNI, with ``transactional = False``.

   Patterns are matched by their minimum DFS codes, with the labels stored in ``patterns``. All
   the frequent patterns are needed: with the maximal ones only (``maximal = True``) almost no
   pair is left. If two different edges of ``Q`` give the same ``P`` with different edge labels,
   the rule uses the first one for the lift.

   :param net: The network that was mined.
   :param patterns: Pattern network returned by :func:`transactional_graph_mining` or
      :func:`single_graph_mining`.
   :param support: Support table returned with ``patterns``.
   :param transactional: Whether the patterns come from :func:`transactional_graph_mining` (or
      from :func:`single_graph_mining`).
   :return: :class:`pandas.DataFrame` with columns ``antecedent`` and ``consequent`` (pattern ids,
      the components of ``patterns``), ``support``, ``confidence`` and ``lift``, one row per pair.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      db = gsk.load_network("molecules.tsv", "atoms.tsv")
      patterns, support = gsk.transactional_graph_mining(db, 100, node_label = "element")
      rules = gsk.graph_association_rules(db, patterns, support)
      rules[rules["confidence"] > 0.9].sort_values("lift", ascending = False)
   """
   cols = ["antecedent", "consequent", "support", "confidence", "lift"]
   if patterns is None:
      return pd.DataFrame({c: [] for c in cols})
   enames = patterns.edge_attributes + patterns.qualitative_edge_attributes
   if net.layer_labels is not None and LAYER in enames:
      db = _database(net, None, LAYER)
      N = db["ngraphs"] if transactional else len(db["lab"])
   else:
      N = net.number_of_components() if transactional else net.number_of_nodes()
   keys, structures, directed = _pattern_keys(patterns)[:3]
   index = {k: p for p, k in enumerate(keys)}
   sup = dict(zip(support["pattern"].tolist(), support["support"].tolist()))
   rows, seen = [], set()
   for q, (lab, edges) in enumerate(structures):
      for ants, e, _, _ in _one_edge_less(lab, edges, directed):
         a, b, el = edges[e]
         ek = tuple(_canonical([lab[a], lab[b]], [(0, 1, el)], directed))
         for ak in ants:
            p = index.get(ak)
            if p is None or (p, q) in seen:
               continue
            seen.add((p, q))
            conf = sup[q] / sup[p]
            ep = index.get(ek)
            rows.append((p, q, sup[q], conf, conf / (sup[ep] / N) if ep is not None else np.nan))
   out = pd.DataFrame(rows, columns = cols)
   return out.sort_values(["antecedent", "consequent"], ignore_index = True)
