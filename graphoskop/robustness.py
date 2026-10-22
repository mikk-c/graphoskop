"""
Robustness: how much a network breaks down when its nodes or edges fail, one after the other.
Like epidemics, failures are a percolation process: removing nodes (site percolation) or edges
(bond percolation) eventually breaks the network into small pieces.

:func:`node_removal` and :func:`edge_removal` remove the nodes or the edges of a network one by
one, in an order given by a strategy, and record a quality of the network after each removal:

* ``"lcc"``: the fraction of all the nodes in the largest connected component (weakly connected,
  in directed networks), the size of the giant component in percolation theory;
* ``"apl"``: the average shortest path length between the remaining nodes, as
  :func:`~graphoskop.average_shortest_path_length` (unit lengths, ordered pairs connected by a
  path, directions followed in directed networks; NaN when no pair is connected).

Strategies:

* ``"random"``: random failures, a uniformly random order, anew in every run;
* ``"friend_of_random"`` (nodes only): an acquaintance attack, the order of the friends named by
  random nodes (as in the immunization of :mod:`~graphoskop.epidemics.simple_contagion`): a random
  node names a random neighbor (out-neighbor in directed networks), which is removed, drawn again
  if already removed. This is the same as drawing the nodes without replacement with probability
  proportional to :math:`w_v = \\sum_{u \\to v} 1 / k_u`, the chance that a random node names
  :math:`v`, which is how the order is drawn; nodes nobody names come last, in random order;
* node or edge ids: removed in the given order, and only those;
* a measure, i.e. values of the nodes (vector, dictionary or :class:`pandas.Series` by node id, or
  the name of a numeric node attribute, e.g. a centrality or :func:`~graphoskop.degree` with
  ``distribution = None``) or of the edges (vector aligned with the edges, e.g.
  :func:`~graphoskop.edge_betweenness_centrality`, or the name of a numeric edge attribute): a
  targeted attack, from the largest value down (``selection = "top"``, NaN last), or drawn
  without replacement with probability proportional to the values (``selection = "weighted"``).

A measure can also be a function of the network returning such values (e.g.
``lambda g: gsk.degree(g, distribution = None)``). By default the measure is computed once, on
the intact network. With ``recompute = True`` and a function, it is computed again on the
remaining network after every removal (an adaptive attack: e.g. the node with the largest current
degree, then the largest degree among the nodes left, and so on), or after every ``recompute``
removals if it is a number, which is much faster on large networks: each computation rebuilds the
remaining network and calls the function. Random strategies are repeated ``runs`` times, and the table gives the mean and its 95% Student's
t confidence interval; the other strategies give the same order every time, and are run once.

The largest component is found for every number of removals at once, by adding the removed
nodes or edges back in reverse order with a union-find (Newman and Ziff, 2000), so a whole curve
costs about as much as finding the components once; runs go in parallel on the CPU threads. The
average path length needs a breadth-first search from every remaining node (or from ``samples``
random ones) at every recorded point, in parallel over the sources: much slower, so record few
``points``.

:func:`cascade_failures` simulates cascades of failures instead: nodes or edges carry a load and
have a capacity, a failed item passes its load to its neighbors, and overloaded items fail in turn.
"""

import numba
import numpy as np
import pandas as pd
import torch
from numba import njit, prange
from scipy.stats import t as student

from .ergms import _random


@njit(cache = True)
def _find(parent, x):
   """Root of ``x`` in a union-find, halving the path."""
   while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
   return x


@njit(cache = True, parallel = True)
def _lcc_curves(n, src, dst, orders, nodes, out):
   """
   Size of the largest connected component after removing the first ``i`` items of every order,
   for every ``i``, by adding the items back in reverse order (Newman and Ziff, 2000), in
   parallel over the orders.

   :param n: Number of nodes.
   :param src: Source of every edge (each undirected edge once).
   :param dst: Target of every edge.
   :param orders: Removal orders, runs by ``k``: nodes, or positions in ``src`` / ``dst``.
   :param nodes: Whether the items are nodes (an edge exists only between present nodes) or edges.
   :param out: Sizes, runs by ``k + 1``, written in place.
   """
   m, k = src.shape[0], orders.shape[1]
   # adjacency through edge positions, for adding nodes back
   deg = np.zeros(n + 1, dtype = np.int64)
   for e in range(m):
      deg[src[e] + 1] += 1
      deg[dst[e] + 1] += 1
   ptr = np.cumsum(deg)
   inc = np.empty(2 * m, dtype = np.int64)
   fill = ptr[:-1].copy()
   for e in range(m):
      inc[fill[src[e]]] = e
      fill[src[e]] += 1
      inc[fill[dst[e]]] = e
      fill[dst[e]] += 1
   for r in prange(orders.shape[0]):
      parent = np.arange(n)
      size = np.ones(n, dtype = np.int64)
      gone = np.zeros(n if nodes else m, dtype = np.bool_)
      for i in range(k):
         gone[orders[r, i]] = True
      best = 0 if nodes else 1
      if nodes:
         for v in range(n):
            if not gone[v]:
               best = 1
               break
      for e in range(m):   # the network after all the removals
         a, b = src[e], dst[e]
         if (gone[e] if not nodes else gone[a] or gone[b]):
            continue
         a, b = _find(parent, a), _find(parent, b)
         if a != b:
            if size[a] < size[b]:
               a, b = b, a
            parent[b] = a
            size[a] += size[b]
            best = max(best, size[a])
      out[r, k] = best
      for i in range(k - 1, -1, -1):   # back in reverse order
         x = orders[r, i]
         gone[x] = False
         if nodes:
            best = max(best, 1)
            for j in range(ptr[x], ptr[x + 1]):
               e = inc[j]
               a, b = src[e], dst[e]
               if gone[a] or gone[b]:
                  continue
               a, b = _find(parent, a), _find(parent, b)
               if a != b:
                  if size[a] < size[b]:
                     a, b = b, a
                  parent[b] = a
                  size[a] += size[b]
                  best = max(best, size[a])
         else:
            a, b = _find(parent, src[x]), _find(parent, dst[x])
            if a != b:
               if size[a] < size[b]:
                  a, b = b, a
               parent[b] = a
               size[a] += size[b]
               best = max(best, size[a])
         out[r, i] = best


@njit(cache = True, parallel = True)
def _path_sums(ptr, nbr, eid, node_ok, edge_ok, sources):
   """
   Sum of the shortest path lengths from ``sources`` to the nodes they reach, and number of such
   pairs, by breadth-first search in parallel over the sources.

   :param ptr: CSR row pointers of the out edges.
   :param nbr: CSR targets of the out edges.
   :param eid: Edge position of every CSR entry (in the arrays of the removed edges).
   :param node_ok: Whether every node is present.
   :param edge_ok: Whether every edge is present.
   :param sources: Source nodes (present).
   :return: Total length and number of ordered pairs.
   """
   n = ptr.shape[0] - 1
   tot = np.zeros(sources.shape[0])
   cnt = np.zeros(sources.shape[0])
   for i in prange(sources.shape[0]):
      dist = np.full(n, -1, dtype = np.int64)
      queue = np.empty(n, dtype = np.int64)
      s = sources[i]
      dist[s], queue[0], head, tail = 0, s, 0, 1
      while head < tail:
         u = queue[head]
         head += 1
         for j in range(ptr[u], ptr[u + 1]):
            v = nbr[j]
            if dist[v] < 0 and node_ok[v] and edge_ok[eid[j]]:
               dist[v] = dist[u] + 1
               queue[tail] = v
               tail += 1
               tot[i] += dist[v]
               cnt[i] += 1
   return tot.sum(), cnt.sum()


def _measure_order(x, selection, runs, rs):
   """
   Removal orders from the values of the items: largest first (NaN last), or drawn without
   replacement with probability proportional to the values (Efraimidis and Spirakis, 2006; items
   with value 0 last, in random order).

   :param x: Values (float64 :class:`numpy.ndarray`).
   :param selection: ``"top"`` or ``"weighted"``.
   :param runs: Number of runs (for ``"weighted"``).
   :param rs: NumPy random generator.
   :return: Orders, one row per run (one row for ``"top"``).
   :raises ValueError: If ``selection`` is not valid, or weights are negative.
   """
   if selection == "top":
      return np.argsort(-np.nan_to_num(x, nan = -np.inf), kind = "stable")[None]
   if selection != "weighted":
      raise ValueError(f"unknown selection: {selection}")
   x = np.nan_to_num(x, nan = 0.)
   if (x < 0).any():
      raise ValueError("weighted selection needs non negative values")
   pos = x > 0
   out = np.empty((runs, x.shape[0]), dtype = np.int64)
   for r in range(runs):
      u = rs.random((2, x.shape[0]))
      key = np.where(pos, np.log(u[0] + 1e-300) / np.where(pos, x, 1), u[1])   # descending keys among positive values
      out[r] = np.lexsort((-key, ~pos))
   return out


def _setup(net, component):
   """
   Nodes and edges of the network (or of a component), for the robustness curves.

   :return: ``(comps, off, n, src, dst, ptr, nbr, eid)``: the components, the position of their
      first nodes, the number of nodes, every edge once (input edges; network wide positions), and
      the CSR of the stored edges (both directions if undirected) with the position of the edge
      of every entry.
   """
   comps = list(range(len(net.components))) if component is None else [component]
   off = np.cumsum([0] + [net.components[c].num_nodes for c in comps])
   src, dst, s_all, t_all, e_all = [], [], [], [], []
   m0 = 0
   for c, o in zip(comps, off):
      ei = net.components[c].edge_index.cpu().numpy()
      m = net._input_edges(c)[0].shape[1]
      src.append(ei[0, :m] + o)
      dst.append(ei[1, :m] + o)
      s_all.append(ei[0] + o)
      t_all.append(ei[1] + o)
      e_all.append(m0 + (np.arange(m) if net.directed else np.r_[np.arange(m), _reverse_of(ei, m)]))   # edge of every stored entry
      m0 += m
   n = int(off[-1])
   cat = lambda x: np.concatenate(x) if x else np.zeros(0, dtype = np.int64)
   src, dst, s_all, t_all, e_all = map(cat, (src, dst, s_all, t_all, e_all))
   order = np.argsort(s_all, kind = "stable")
   ptr = np.zeros(n + 1, dtype = np.int64)
   ptr[1:] = np.cumsum(np.bincount(s_all, minlength = n))
   return comps, off, n, src.astype(np.int64), dst.astype(np.int64), ptr, t_all[order].astype(np.int64), e_all[order].astype(np.int64)


def _reverse_of(ei, m):
   """
   Input edge of every reverse copy stored by an undirected component (after the ``m`` input
   edges, the reverses of those that are not self loops, in order).

   :param ei: Edge index of the component (NumPy).
   :param m: Number of input edges.
   :return: Positions among the input edges.
   """
   return np.flatnonzero(ei[0, :m] != ei[1, :m])


def _orders(net, comps, off, nodes, strategy, selection, runs, rs, ptr, nbr, src, dst, recompute = False):
   """
   Removal orders of the nodes or edges of a strategy (see the module description).

   :param net: Network.
   :param comps: Components considered.
   :param off: Position of the first node of each of ``comps``, then the number of nodes.
   :param nodes: Whether the items are nodes or edges.
   :param strategy: ``"random"``, ``"friend_of_random"``, ids, values, or a function of the
      network returning values.
   :param selection: ``"top"`` or ``"weighted"``.
   :param runs: Number of runs of the random strategies.
   :param rs: NumPy random generator.
   :param ptr: CSR row pointers of the stored edges.
   :param nbr: CSR targets of the stored edges.
   :param src: Source of every edge (each once).
   :param dst: Target of every edge.
   :param recompute: With a function, ``True`` or a number of removals after which it is computed
      again on the remaining network; ``False`` to compute it once.
   :return: Orders, one row per run (one row for deterministic strategies).
   :raises ValueError: If the strategy is not valid for the items, or ``recompute`` is asked
      without a function.
   """
   n = int(off[-1])
   total = n if nodes else src.shape[0]
   if recompute is not False and recompute is not None and not callable(strategy):
      raise ValueError("recompute needs the measure as a function of the network, e.g. lambda g: gsk.degree(g, distribution = None)")
   if callable(strategy) and recompute not in (False, None):
      step = 1 if recompute is True else int(recompute)
      if step < 1:
         raise ValueError(f"recompute must be True or a positive number of removals, got {recompute}")
      return _adaptive(net, comps, off, nodes, strategy, selection, runs if selection == "weighted" else 1, step, rs, src)
   if callable(strategy):
      strategy = strategy(net)
   if isinstance(strategy, str) and strategy == "random":
      return np.stack([rs.permutation(total) for _ in range(runs)])
   if isinstance(strategy, str) and strategy == "friend_of_random":
      if not nodes:
         raise ValueError("friend_of_random removes nodes: it is not available for edges")
      out_deg = np.diff(ptr)
      s_all = np.repeat(np.arange(n), out_deg)
      w = np.bincount(nbr, weights = 1. / out_deg[s_all], minlength = n)   # chance that a random node names v, times n
      return _measure_order(w, "weighted", runs, rs)
   if _is_values(net, strategy, nodes):
      return _measure_order(_values(net, comps, strategy, nodes), selection, runs, rs)
   return _ids(net, comps, off, strategy, nodes, src, dst)[None]


#: Name of the edge attribute holding the position of every edge while recomputing a measure.
_POSITION = "__graphoskop_position"


def _adaptive(net, comps, off, nodes, f, selection, runs, step, rs, src):
   """
   Removal orders of an adaptive attack: the measure ``f`` is computed on the remaining network,
   the ``step`` items with the largest values (or drawn proportionally to them) are removed, and
   so on until nothing is left.

   :param f: Function of a :class:`~graphoskop.Network` returning node values (a vector for
      component 0, a list of vectors, one per component, or a dictionary or
      :class:`pandas.Series` by node id) or edge values (a vector for component 0, or a list of
      them, with one value per edge or per stored edge).
   :return: Orders, one row per run.
   """
   from .transformations import _restrict, _with_edges, subgraph
   n = int(off[-1])
   total = n if nodes else src.shape[0]
   ids = np.empty(n, dtype = object)
   ids[:] = [x for c in comps for x in net.node_ids[c].tolist()]
   pos_of = {x: i for i, x in enumerate(ids.tolist())}
   if not nodes:   # a copy of the network with the position of every edge, to map values back
      if total >= 2 ** 24:
         raise ValueError("recompute on edges is limited to 16,777,216 edges")
      cols, m0 = [], 0
      for c, d in enumerate(net.components):
         col = torch.full((d.edge_index.shape[1], 1), -1., dtype = d.edge_attr.dtype, device = d.edge_attr.device)
         if c in comps:
            m = net._input_edges(c)[0].shape[1]
            ei = d.edge_index.cpu().numpy()
            col[:, 0] = torch.as_tensor(m0 + (np.arange(m) if net.directed else np.r_[np.arange(m), _reverse_of(ei, m)]), dtype = col.dtype)
            m0 += m
         cols.append((d.edge_index, torch.cat([d.edge_attr, col], 1)))
      work = _with_edges(net, cols)
      work.edge_attributes.append(_POSITION)
   out = np.empty((runs, total), dtype = np.int64)
   for r in range(runs):
      alive = np.ones(total, dtype = np.bool_)
      done = 0
      while done < total:
         if nodes:
            sub = subgraph(net, ids[alive].tolist())
         else:
            keep = []
            for c, d in enumerate(work.components):
               p = work.get_edge_attribute(_POSITION, c).cpu().numpy().round().astype(np.int64)
               keep.append((p >= 0) & alive[np.maximum(p, 0)])
            sub = _restrict(work, [np.ones(d.num_nodes, dtype = np.bool_) for d in work.components], keep)
         x = _mapped(sub, f(sub), nodes, pos_of, total)
         cand = np.flatnonzero(alive)
         take = cand[_measure_order(x[cand], selection, 1, rs)[0][:step]]
         out[r, done:done + take.shape[0]] = take
         alive[take] = False
         done += take.shape[0]
   return out


def _mapped(sub, v, nodes, pos_of, total):
   """
   Values computed on a remaining network, at the positions of the items in the original one
   (NaN for items without a value).
   """
   x = np.full(total, np.nan)
   if isinstance(v, str):
      v = (sub.get_node_attribute if nodes else sub.get_edge_attribute)(v, None)
   if nodes and isinstance(v, (dict, pd.Series)):
      ids = [i for c in range(len(sub.components)) for i in sub.node_ids[c].tolist()]
      x[[pos_of[i] for i in ids]] = pd.Series(ids, dtype = object).map(v).to_numpy(dtype = np.float64)
      return x
   vecs = list(v) if isinstance(v, (list, tuple)) and len(v) and (torch.is_tensor(v[0]) or isinstance(v[0], np.ndarray)) else [v]
   for c, vec in enumerate(vecs):
      vec = np.asarray(vec.cpu().numpy() if torch.is_tensor(vec) else vec, dtype = np.float64)
      if nodes:
         x[[pos_of[i] for i in sub.node_ids[c].tolist()]] = vec
      else:
         p = sub.get_edge_attribute(_POSITION, c).cpu().numpy().round().astype(np.int64)
         m = sub._input_edges(c)[0].shape[1]
         p = p if vec.shape[0] == p.shape[0] else p[:m]
         ok = p >= 0
         x[p[ok]] = vec[ok]
   return x


def _curves(net, nodes, strategy, selection, quality, runs, points, component, samples, seed, recompute = False):
   """
   Body of :func:`node_removal` and :func:`edge_removal`.
   """
   if quality not in ("lcc", "apl"):
      raise ValueError(f"unknown quality: {quality}")
   if runs < 1:
      raise ValueError("runs must be positive")
   comps, off, n, src, dst, ptr, nbr, eid = _setup(net, component)
   total = n if nodes else src.shape[0]
   rs = np.random.default_rng(seed)

   orders = _orders(net, comps, off, nodes, strategy, selection, runs, rs, ptr, nbr, src, dst, recompute)
   orders = np.ascontiguousarray(orders, dtype = np.int64)
   k = orders.shape[1]

   # quality after 0 ... k removals, at the recorded points
   at = np.arange(k + 1) if points is None else np.unique(np.round(np.linspace(0, k, max(2, int(points)))).astype(np.int64))
   if quality == "lcc":
      sizes = np.zeros((orders.shape[0], k + 1), dtype = np.int64)
      _lcc_curves(n, src, dst, orders, nodes, sizes)
      q = sizes[:, at] / n
   else:
      q = np.zeros((orders.shape[0], at.shape[0]))
      for r in range(orders.shape[0]):
         for j, i in enumerate(at):
            node_ok, edge_ok = np.ones(n, dtype = np.bool_), np.ones(src.shape[0], dtype = np.bool_)
            (node_ok if nodes else edge_ok)[orders[r, :i]] = False
            alive = np.flatnonzero(node_ok)
            srcs = alive if samples is None or samples >= alive.shape[0] else np.sort(rs.choice(alive, samples, replace = False))
            tot, cnt = _path_sums(ptr, nbr, eid, node_ok, edge_ok, srcs) if srcs.shape[0] else (0., 0.)
            q[r, j] = tot / cnt if cnt > 0 else np.nan
   name = quality
   mean = np.nanmean(q, 0) if quality == "apl" else q.mean(0)
   R = q.shape[0]
   if R > 1:
      ok = (~np.isnan(q)).sum(0)
      sd = np.nanstd(q, 0, ddof = 1) if quality == "apl" else q.std(0, ddof = 1)
      with np.errstate(invalid = "ignore", divide = "ignore"):
         h = student.ppf(0.975, np.maximum(ok - 1, 1)) * sd / np.sqrt(ok)
   else:
      h = np.full(mean.shape, np.nan)
   return pd.DataFrame({"fraction": at / total, name: mean, name + "_low": mean - h, name + "_high": mean + h}, index = pd.Index(at, name = "removed"))


def _is_values(net, strategy, nodes):
   """
   Whether a strategy is a measure (values of the nodes or of the edges), rather than ids.
   """
   if isinstance(strategy, str):
      return strategy in (net.node_attributes if nodes else net.edge_attributes)
   if isinstance(strategy, (dict, pd.Series)):
      return nodes
   if torch.is_tensor(strategy):
      return True
   if isinstance(strategy, (list, tuple)) and len(strategy) and (torch.is_tensor(strategy[0]) or isinstance(strategy[0], np.ndarray)):
      return True   # one vector per component
   x = np.asarray(strategy) if not isinstance(strategy, (list, tuple)) or not len(strategy) or not isinstance(strategy[0], tuple) else None
   return x is not None and x.ndim == 1 and x.dtype.kind in "fc"


def _values(net, comps, strategy, nodes):
   """
   Values of the nodes, or of the edges (each input edge once), of the components considered.

   :raises ValueError: If a vector does not have one value per node or per edge.
   """
   if nodes:
      from .epidemics.simple_contagion import _node_values
      return _node_values(net, comps, strategy)
   if isinstance(strategy, str):
      vec = [net._numeric_edge(strategy, c) for c in comps]
   elif isinstance(strategy, (list, tuple)) and len(strategy) == len(net.components):
      vec = [strategy[c] for c in comps]
   elif isinstance(strategy, (list, tuple)) and len(strategy) and (torch.is_tensor(strategy[0]) or isinstance(strategy[0], np.ndarray)):
      vec = list(strategy)
   else:
      vec = [strategy]
   if len(vec) != len(comps):
      raise ValueError(f"expected one vector of edge values per component ({len(comps)}), got {len(vec)}")
   out = []
   for c, v in zip(comps, vec):
      v = np.asarray(v.cpu().numpy() if torch.is_tensor(v) else v, dtype = np.float64)
      m, stored = net._input_edges(c)[0].shape[1], net.components[c].edge_index.shape[1]
      if v.shape[0] not in (m, stored):
         raise ValueError(f"expected one value per edge ({m}, or {stored} stored edges), got {v.shape[0]}")
      out.append(v[:m])   # stored edges: the input edges come first
   return np.concatenate(out)


def _ids(net, comps, off, strategy, nodes, src, dst):
   """
   Removal order given by node ids, or by edges ``(u, v)`` (all the edges from ``u`` to ``v``, in
   either direction if undirected).

   :raises KeyError: If a node or an edge does not exist.
   :raises ValueError: If a node is not among the nodes considered.
   """
   if isinstance(strategy, pd.DataFrame):
      strategy = strategy.iloc[:, 0].tolist() if nodes else list(zip(strategy.iloc[:, 0].tolist(), strategy.iloc[:, 1].tolist()))
   items = [strategy] if isinstance(strategy, (str, tuple)) else list(strategy)

   def position(x):
      out = []
      for c, i in net._nodes(x):
         if c not in comps:
            raise ValueError(f"node {x!r} is not among the nodes considered (component {c})")
         out.append(int(off[comps.index(c)] + i))
      return out

   order = []
   if nodes:
      for x in items:
         order += position(x)
   else:
      n = int(off[-1])
      key = pd.Series(np.arange(src.shape[0])).groupby(src * n + dst if net.directed else np.minimum(src, dst) * n + np.maximum(src, dst)).apply(list).to_dict()
      for u, v in items:
         found = False
         for a in position(u):
            for b in position(v):
               for e in key.get(a * n + b if net.directed else min(a, b) * n + max(a, b), []):
                  order.append(e)
                  found = True
         if not found:
            raise KeyError(f"no edge {(u, v)!r}")
   return np.asarray(list(dict.fromkeys(order)), dtype = np.int64)


def node_removal(net, strategy = "random", selection = "top", quality = "lcc", runs = 10, points = 100, component = None, samples = None, seed = None, recompute = False):
   """
   Robustness of a network to node failures or attacks: remove its nodes one by one, with their
   edges, and record the quality of the network after each removal (see the module description).

   :param net: A :class:`~graphoskop.Network`.
   :param strategy: ``"random"`` (failures), ``"friend_of_random"`` (acquaintance attack), node
      ids (the nodes to remove, in order), or node values (a measure: vector, list of vectors,
      dictionary or :class:`pandas.Series` by node id, or the name of a numeric node attribute),
      or a function of the network returning node values.
   :param selection: With a measure: ``"top"`` (largest values first) or ``"weighted"`` (drawn
      with probability proportional to the values).
   :param quality: ``"lcc"`` (fraction of the nodes in the largest connected component) or
      ``"apl"`` (average shortest path length).
   :param runs: Number of runs of the random strategies (``"random"``, ``"friend_of_random"``,
      ``"weighted"``); the others are run once.
   :param points: Number of numbers of removals recorded, evenly spaced from none to all, or
      ``None`` for every removal.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param samples: With ``"apl"``, the number of random source nodes of the searches at each
      point, or ``None`` for all the remaining nodes (exact).
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param recompute: With a measure given as a function of the network, ``True`` to compute it
      again on the remaining network after every removal (adaptive attack), a number of removals
      between computations, or ``False`` to compute it once.
   :return: :class:`pandas.DataFrame` indexed by the number of removed nodes (``removed``), with
      columns ``fraction`` (of the nodes removed), the mean quality (``lcc`` or ``apl``) and its
      confidence interval (``lcc_low``, ``lcc_high``, ...; NaN with a single run).
   :raises ValueError: If a parameter is not valid, a vector does not have one value per node, or
      a node is outside the nodes considered.
   :raises KeyError: If a node does not exist.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      failures = gsk.node_removal(net)                                      # random, 10 runs
      k = gsk.degree(net, distribution = None)                              # one vector per component
      attack = gsk.node_removal(net, strategy = k)                          # hubs first
      soft = gsk.node_removal(net, strategy = k, selection = "weighted")    # hubs more likely first
      friends = gsk.node_removal(net, strategy = "friend_of_random")
      apl = gsk.node_removal(net, strategy = k, quality = "apl", points = 20)
      failures.plot(x = "fraction", y = "lcc")
      degree = lambda g: gsk.degree(g, distribution = None)
      adaptive = gsk.node_removal(net, strategy = degree, recompute = True)   # largest current degree first
      faster = gsk.node_removal(net, strategy = degree, recompute = 100)      # recomputed every 100 removals
   """
   return _curves(net, True, strategy, selection, quality, runs, points, component, samples, seed, recompute)


def edge_removal(net, strategy = "random", selection = "top", quality = "lcc", runs = 10, points = 100, component = None, samples = None, seed = None, recompute = False):
   """
   Robustness of a network to edge failures or attacks: remove its edges one by one (all nodes
   stay), and record the quality of the network after each removal (see the module description).
   Every edge counts once (in undirected networks, both stored directions go together); parallel
   edges are removed one at a time.

   :param net: A :class:`~graphoskop.Network`.
   :param strategy: ``"random"`` (failures), edges ``(u, v)`` (removed in order; all the edges
      between ``u`` and ``v``, in either direction if undirected), or edge values (a measure: a
      vector with one value per edge or per stored edge, e.g.
      :func:`~graphoskop.edge_betweenness_centrality`, a list of them, one per component, or the
      name of a numeric edge attribute), or a function of the network returning edge values.
   :param selection: With a measure: ``"top"`` (largest values first) or ``"weighted"`` (drawn
      with probability proportional to the values).
   :param quality: ``"lcc"`` (fraction of the nodes in the largest connected component) or
      ``"apl"`` (average shortest path length).
   :param runs: Number of runs of the random strategies; the others are run once.
   :param points: Number of numbers of removals recorded, evenly spaced from none to all, or
      ``None`` for every removal.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param samples: With ``"apl"``, the number of random source nodes of the searches at each
      point, or ``None`` for all the nodes (exact).
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param recompute: With a measure given as a function of the network, ``True`` to compute it
      again on the remaining network after every removal (adaptive attack), a number of removals
      between computations, or ``False`` to compute it once.
   :return: :class:`pandas.DataFrame` indexed by the number of removed edges (``removed``), with
      columns ``fraction`` (of the edges removed), the mean quality (``lcc`` or ``apl``) and its
      confidence interval (NaN with a single run).
   :raises ValueError: If a parameter is not valid, or a vector does not have one value per edge.
   :raises KeyError: If a node or an edge does not exist.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      failures = gsk.edge_removal(net)                                      # random, 10 runs
      eb = gsk.edge_betweenness_centrality(net)                             # largest component
      attack = gsk.edge_removal(net, strategy = eb, component = 0)          # highest betweenness first
      cut = gsk.edge_removal(net, strategy = [("alice", "bob"), ("bob", "carol")])
   """
   return _curves(net, False, strategy, selection, quality, runs, points, component, samples, seed, recompute)


@njit(cache = True)
def _with_mates(fail, nf, when, t, gptr, gmem, gid):
   """
   Add to the failing items the other copies of their entities (interdependent layers), failing
   at the same step.

   :param fail: Failing items, the first ``nf`` set (extended in place).
   :param when: Failure step of every item in this run, -1 if not failed (updated in place).
   :param t: Current step.
   :param gptr: Row pointers of the copies of every entity (empty: no interdependence).
   :param gmem: Copies of every entity.
   :param gid: Entity of every item.
   :return: New number of failing items.
   """
   if gptr.shape[0] == 0:
      return nf
   for i in range(nf):
      g = gid[fail[i]]
      for j in range(gptr[g], gptr[g + 1]):
         y = gmem[j]
         if when[y] < 0:
            when[y] = t
            fail[nf] = y
            nf += 1
   return nf


@njit(cache = True, parallel = True)
def _cascade_runs(ptr, nbr, src, dst, iptr, inc, nodes, directed, load, cap, init, dist, rng, max_steps, when, gptr, gmem, gid):
   """
   Cascades of failures with load redistribution, in parallel over the runs. At every step the
   failing items are removed together, and each one splits its load among its remaining
   neighbors; the items whose load exceeds their capacity fail at the next step.

   :param ptr: CSR row pointers of the stored edges (neighbors of nodes).
   :param nbr: CSR targets of the stored edges.
   :param src: Source of every edge (each once).
   :param dst: Target of every edge.
   :param iptr: Row pointers of the edges incident to every node (out-incident if directed).
   :param inc: Incident edges.
   :param nodes: Whether the items are nodes or edges.
   :param directed: Whether the network is directed (edges pass their load downstream).
   :param load: Load of every item at the start, runs by items (modified in place).
   :param cap: Capacity of every item, runs by items.
   :param init: Items failing at step 0, runs by count, padded with -1.
   :param dist: Split of the load: 0 equal, 1 random, 2 proportional to the load of the
      receivers, 3 proportional to their capacity.
   :param rng: xorshift state of every run.
   :param max_steps: Largest number of steps.
   :param when: Step at which every item fails, -1 if never, written in place.
   :param gptr: Row pointers of the copies of every entity, to fail together (empty: none).
   :param gmem: Copies of every entity.
   :param gid: Entity of every item.
   """
   N = load.shape[1]
   for r in prange(load.shape[0]):
      st = rng[r:r + 1]
      ld = load[r]
      alive = np.ones(N, dtype = np.bool_)
      delta = np.zeros(N)
      fail = np.empty(N, dtype = np.int64)
      nxt = np.empty(N, dtype = np.int64)
      recv = np.empty(N if nodes else 2 * src.shape[0] + 1, dtype = np.int64)
      wts = np.empty(recv.shape[0])
      touched = np.empty(N, dtype = np.int64)
      seen = np.zeros(N, dtype = np.bool_)
      nf = 0
      for x in init[r]:
         if x >= 0 and when[r, x] < 0:
            when[r, x] = 0
            fail[nf] = x
            nf += 1
      nf = _with_mates(fail, nf, when[r], 0, gptr, gmem, gid)
      t = 0
      while nf > 0 and t < max_steps:
         for i in range(nf):   # removed together
            alive[fail[i]] = False
         nt = 0
         for i in range(nf):
            x = fail[i]
            nr = 0
            if nodes:
               for j in range(ptr[x], ptr[x + 1]):
                  if alive[nbr[j]]:
                     recv[nr] = nbr[j]
                     nr += 1
            else:
               for end in range(2):
                  if directed and end == 0:
                     continue   # downstream: the edges leaving the target
                  v = dst[x] if end == 1 else src[x]
                  for j in range(iptr[v], iptr[v + 1]):
                     e = inc[j]
                     if alive[e] and nr < recv.shape[0]:
                        recv[nr] = e
                        nr += 1
            if nr == 0:   # nobody left to take the load: lost
               continue
            tot = 0.
            for j in range(nr):
               y = recv[j]
               w = 1. if dist == 0 else _random(st) if dist == 1 else ld[y] if dist == 2 else cap[r, y]
               wts[j] = w
               tot += w
            for j in range(nr):
               y = recv[j]
               delta[y] += ld[x] * (wts[j] / tot if tot > 0 else 1. / nr)
               if not seen[y]:
                  seen[y] = True
                  touched[nt] = y
                  nt += 1
            ld[x] = 0.
         nn = 0
         for i in range(nt):   # the new loads, and the overloaded items
            y = touched[i]
            ld[y] += delta[y]
            delta[y], seen[y] = 0., False
            if alive[y] and ld[y] > cap[r, y]:
               nxt[nn] = y
               nn += 1
         t += 1
         for i in range(nn):
            when[r, nxt[i]] = t
            fail[i] = nxt[i]
         nf = _with_mates(fail, nn, when[r], t, gptr, gmem, gid)


def _load_values(net, comps, x, nodes, total, runs, rs, name):
   """
   Loads or capacities: one row per run of random values, or the same given values in every run.

   :param x: ``"random"`` (uniform in [0, 1), anew in every run), the name of a numeric node (or
      edge) attribute, or values as for a measure.
   :return: Array, runs by items.
   :raises ValueError: If the values are not finite and non negative.
   """
   if isinstance(x, str) and x == "random":
      return rs.random((runs, total))
   v = _values(net, comps, x, nodes)
   if not (np.isfinite(v) & (v >= 0)).all():
      raise ValueError(f"{name} must be finite and non negative")
   return np.repeat(v[None], runs, 0)


def cascade_failures(net, load = "random", capacity = "random", failures = 1, strategy = "random", selection = "top", elements = "nodes", distribution = "equal", margin = 0.5, quality = "lcc", runs = 10, max_steps = 1000, component = None, samples = None, seed = None, interdependent = True):
   """
   Cascading failures with load redistribution, as in the overload models of power grids and
   fiber bundles (Motter and Lai, 2002): every node (or edge) carries a load and has a capacity;
   when it fails it is removed from the network and its load goes to its neighbors, which fail in
   turn at the next step if their load exceeds their capacity, and so on until no item is
   overloaded.

   Loads and capacities are node (or edge) attributes given by name, values (as for a measure,
   see the module description), or ``"random"``: random loads are uniform in [0, 1), and random
   capacities are the load plus a spare capacity uniform in [0, ``margin``), so that nothing is
   overloaded at the start; random values are drawn anew in every run.

   At step 0, the initial failures happen: ``failures`` items (a fraction of them if below 1)
   chosen by ``strategy``, as in :func:`node_removal` (``"random"``, ``"friend_of_random"`` for
   nodes, a measure with ``selection``, or ids, which give the failing items themselves and make
   ``failures`` unused), or ``"overload"``: all the items whose load already exceeds their
   capacity. At every step, the failing items are removed together, and each one splits its
   current load among its remaining neighbors (``distribution``):

   * ``"equal"``: in equal parts;
   * ``"random"``: in random parts (weights uniform in [0, 1), normalized);
   * ``"load"``: proportionally to the current load of the receivers;
   * ``"capacity"``: proportionally to the capacity of the receivers.

   The neighbors of a node are its neighbors (out-neighbors in directed networks, where load
   flows along the edges); the neighbors of an edge are the edges that share an endpoint with it
   (in directed networks, the edges leaving its target). Parallel edges count as several
   neighbors. When ``"load"`` or ``"capacity"`` weights are all zero the split is equal, and the
   load of an item without remaining neighbors is lost. The items whose load exceeds their
   capacity after all the transfers of a step fail at the next step (only the items that received
   load are checked: given loads above capacity at the start fail only through ``"overload"``).

   In multilayer networks the items are the copies of the entities (nodes) or the edges of the
   supra-graph, and couplings are edges like the others. With ``interdependent = True`` (and node
   items), the layers depend on each other as in interdependent networks (Buldyrev et al., 2010):
   when a copy fails, all the copies of its entity fail at the same step, and pass their own loads
   to their own neighbors, so failures percolate from layer to layer (the failing copies do not
   pass load to each other through their couplings). With ``interdependent = False`` the copies
   fail independently, and load flows through the couplings as through any edge.

   :param net: A :class:`~graphoskop.Network`.
   :param load: Load of every item: ``"random"``, the name of a numeric attribute, or values.
   :param capacity: Capacity of every item: ``"random"``, the name of a numeric attribute, or
      values.
   :param failures: Number of initial failures, or a fraction of the items if below 1.
   :param strategy: How the initial failures are chosen: ``"random"``, ``"friend_of_random"``
      (nodes), ``"overload"``, a measure, or ids (the failing items).
   :param selection: With a measure: ``"top"`` or ``"weighted"``.
   :param elements: ``"nodes"`` or ``"edges"``, the items that carry load and fail.
   :param distribution: ``"equal"``, ``"random"``, ``"load"`` or ``"capacity"``.
   :param margin: Largest spare capacity of random capacities.
   :param quality: ``"lcc"`` (fraction of all the nodes in the largest connected component) or
      ``"apl"`` (average shortest path length), after the failures of each step.
   :param runs: Number of runs (a single run if nothing is random).
   :param max_steps: Largest number of steps.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param samples: With ``"apl"``, the number of random source nodes of the searches, or ``None``.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param interdependent: In multilayer networks with node items, whether all the copies of an
      entity fail together.
   :return: :class:`pandas.DataFrame` indexed by step (0: the initial failures), with columns
      ``failed`` (mean fraction of the items failed so far), the mean quality (``lcc`` or ``apl``)
      and their confidence intervals (``failed_low``, ``failed_high``, ``lcc_low``, ...; NaN with a
      single run). Runs that stop earlier keep their final values.
   :raises ValueError: If a parameter is not valid, or loads or capacities are not finite and non
      negative.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("grid.tsv", "stations.tsv")                     # node attributes load, capacity
      c = gsk.cascade_failures(net, "load", "capacity", strategy = "overload", runs = 1)
      c = gsk.cascade_failures(net, failures = 5, margin = 0.2)              # random loads and capacities
      k = gsk.degree(net, distribution = None)
      c = gsk.cascade_failures(net, k, failures = 1, strategy = k, distribution = "capacity")   # the hub fails
      c = gsk.cascade_failures(net, elements = "edges", failures = 0.01)
      c["failed"].iloc[-1]                                                     # final size of the cascade
      ml = gsk.load_network("grid_layers.tsv", multilayer = True, coupling = "all")
      c = gsk.cascade_failures(ml, failures = 5)                             # interdependent layers
      c = gsk.cascade_failures(ml, failures = 5, interdependent = False)     # copies fail independently
   """
   if elements not in ("nodes", "edges"):
      raise ValueError(f"unknown elements: {elements}")
   if distribution not in ("equal", "random", "load", "capacity"):
      raise ValueError(f"unknown distribution: {distribution}")
   if quality not in ("lcc", "apl"):
      raise ValueError(f"unknown quality: {quality}")
   if runs < 1 or max_steps < 0 or margin < 0:
      raise ValueError("runs must be positive, max_steps and margin non negative")
   nodes = elements == "nodes"
   comps, off, n, src, dst, ptr, nbr, eid = _setup(net, component)
   total = n if nodes else src.shape[0]
   rs = np.random.default_rng(seed)
   rand_strategy = isinstance(strategy, str) and strategy in ("random", "friend_of_random") or selection == "weighted"
   if not (load == "random" if isinstance(load, str) else False) and not (capacity == "random" if isinstance(capacity, str) else False) and distribution != "random" and not rand_strategy:
      runs = 1   # nothing random
   ld = _load_values(net, comps, load, nodes, total, runs, rs, "load")
   if isinstance(capacity, str) and capacity == "random":
      cp = ld + margin * rs.random((runs, total))
   else:
      cp = _load_values(net, comps, capacity, nodes, total, runs, rs, "capacity")

   # initial failures
   if isinstance(strategy, str) and strategy == "overload":
      over = [np.flatnonzero(ld[r] > cp[r]) for r in range(runs)]
      init = np.full((runs, max([len(o) for o in over] + [1])), -1, dtype = np.int64)
      for r, o in enumerate(over):
         init[r, :len(o)] = o
   else:
      orders = _orders(net, comps, off, nodes, strategy, selection, runs, rs, ptr, nbr, src, dst)
      given = not (isinstance(strategy, str) or _is_values(net, strategy, nodes) or callable(strategy))
      k = orders.shape[1] if given else (max(1, int(round(failures * total))) if failures < 1 else int(failures))
      if k > total:
         raise ValueError(f"{k} initial failures, but only {total} {elements}")
      init = np.ascontiguousarray(np.stack([orders[r % orders.shape[0], :k] for r in range(runs)]), dtype = np.int64)

   # incidence of the edges on the nodes (out-incidence if directed), for edge items
   ends = src if net.directed else np.concatenate([src, dst])
   which = np.arange(src.shape[0]) if net.directed else np.r_[np.arange(src.shape[0]), np.arange(src.shape[0])]
   o = np.argsort(ends, kind = "stable")
   iptr = np.zeros(n + 1, dtype = np.int64)
   iptr[1:] = np.cumsum(np.bincount(ends, minlength = n))
   when = np.full((runs, total), -1, dtype = np.int64)
   gptr, gmem, gid = (np.zeros(0, dtype = np.int64),) * 3
   if nodes and interdependent and net.layer_labels is not None:
      gid = np.concatenate([net.components[c].entity.cpu().numpy() for c in comps]).astype(np.int64)
      gmem = np.argsort(gid, kind = "stable").astype(np.int64)
      gptr = np.r_[0, np.cumsum(np.bincount(gid, minlength = len(net.entity_ids)))].astype(np.int64)
   _cascade_runs(ptr, nbr, src, dst, iptr, which[o].astype(np.int64), nodes, net.directed, np.ascontiguousarray(ld, dtype = np.float64), np.ascontiguousarray(cp, dtype = np.float64), init,
                 ("equal", "random", "load", "capacity").index(distribution), rs.integers(1, 2 ** 63, size = runs, dtype = np.uint64), int(max_steps), when,
                 gptr, gmem, gid)

   # failed fraction and quality at every step
   T = int(max(when.max(), 0))
   steps = np.arange(T + 1)
   failed = np.stack([np.cumsum(np.bincount(w[w >= 0], minlength = T + 1)) for w in when])   # failed so far, runs by steps
   order = np.argsort(np.where(when >= 0, when, T + 1), axis = 1, kind = "stable").astype(np.int64)   # failed items first, by step
   if quality == "lcc":
      sizes = np.zeros((runs, total + 1), dtype = np.int64)
      _lcc_curves(n, src, dst, np.ascontiguousarray(order), nodes, sizes)
      q = np.take_along_axis(sizes, failed, 1) / n
   else:
      q = np.zeros((runs, T + 1))
      for r in range(runs):
         for t in steps:
            node_ok, edge_ok = np.ones(n, dtype = np.bool_), np.ones(src.shape[0], dtype = np.bool_)
            (node_ok if nodes else edge_ok)[order[r, :failed[r, t]]] = False
            alive = np.flatnonzero(node_ok)
            srcs = alive if samples is None or samples >= alive.shape[0] else np.sort(rs.choice(alive, samples, replace = False))
            tot, cnt = _path_sums(ptr, nbr, eid, node_ok, edge_ok, srcs) if srcs.shape[0] else (0., 0.)
            q[r, t] = tot / cnt if cnt > 0 else np.nan
   cols = {}
   for name, x in (("failed", failed / total), (quality, q)):
      mean = np.nanmean(x, 0)
      if runs > 1:
         ok = (~np.isnan(x)).sum(0)
         with np.errstate(invalid = "ignore", divide = "ignore"):
            h = student.ppf(0.975, np.maximum(ok - 1, 1)) * np.nanstd(x, 0, ddof = 1) / np.sqrt(ok)
      else:
         h = np.full(mean.shape, np.nan)
      cols[name], cols[name + "_low"], cols[name + "_high"] = mean, mean - h, mean + h
   return pd.DataFrame(cols, index = pd.Index(steps, name = "step"))
