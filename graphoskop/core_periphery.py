"""
Core-periphery structure: a densely connected core of nodes, and a periphery of nodes connected
to the core but hardly to each other.

The models of Borgatti and Everett (2000) compare the adjacency matrix :math:`A` with an ideal
pattern matrix :math:`\\Delta`, built from how much each node belongs to the core, by their
Pearson correlation :math:`\\rho` over all pairs of distinct nodes (ordered pairs in directed
networks; self loops are ignored). The best core is the one maximizing :math:`\\rho`.

* Discrete model: each node is in the core (:math:`c_i = 1`) or in the periphery
  (:math:`c_i = 0`). Two core nodes should be connected (:math:`\\delta_{ij} = 1`), two peripheral
  nodes should not (:math:`\\delta_{ij} = 0`). Pairs of a core and a peripheral node should be
  connected (``"present"``, :math:`\\delta_{ij} = \\max(c_i, c_j)`), should not
  (``"absent"``, :math:`\\delta_{ij} = c_i c_j`), or are left out of the correlation
  (``"missing"``), as chosen by ``core_periphery_ties``.
* Continuous model: each node has a coreness :math:`c_i \\ge 0`, and the pattern is
  :math:`\\delta_{ij} = c_i c_j`, so that pairs of more central nodes should be more connected.

The model of Rombach, Porter, Fowler and Mucha (2014), :func:`rombach_core_periphery`, generalizes
the continuous one: core values follow a transition function with two parameters, the size of the
core and the sharpness of its boundary, the edges of more central pairs are rewarded (by the
product of the core values of their nodes, or by their :math:`p`-norm), and the core scores
aggregate the best assignments over a range of the two parameters.

The correlation is computed from sums over the edges and over the nodes, never from :math:`n^2`
pairs: for instance :math:`\\sum_{i \\ne j} c_i c_j = (\\sum_i c_i)^2 - \\sum_i c_i^2`. Without
``weight``, :math:`a_{ij}` is 1 for connected pairs (parallel edges count once); with it, the sum of
the weights of the edges from :math:`i` to :math:`j`. All the work runs on the device of the
network, in double precision.

The discrete model is fit by local search: starting from a partition, nodes are moved between core
and periphery while :math:`\\rho` increases. In each round the change of :math:`\\rho` for moving
every node alone is computed at once, from how strongly each node is connected to the core; the
nodes whose move improves :math:`\\rho` are sorted, and the number of them moved together is the
best, computed exactly (accounting for the edges among them) in one pass, so that a round can move
many nodes, and at least the best one. The search stops when no single move improves
:math:`\\rho`. Starting partitions are the best prefixes (the :math:`k` first nodes in the core,
the best :math:`k` found in one pass over the edges) of the nodes sorted by strength, and of
``restarts`` orders by strength times a random log-normal factor (of standard deviation 1 on the
log scale); the best result is kept. The paper used a genetic algorithm instead: both find local
optima, not certainly the global one.

The continuous model is fit by L-BFGS, on :math:`c_i = x_i^2` (so that coreness stays non
negative), starting from the strengths, until ten iterations improve :math:`\\rho` by less than a
fraction ``tol``. Coreness is scaled to a largest value of 1, since :math:`\\rho` does not depend
on the scale.

Nestedness, the core-periphery structure of bipartite networks, is measured by :func:`nestedness`
(NODF, with a comparison with null models), and its isocline is found by
:func:`nestedness_isocline` (see :ref:`nestedness`).

In multilayer networks the supra-graph is used (couplings are edges like the others).
Sparse tensors are built with invariant checks explicitly turned off (they are coalesced by
construction), which also silences the warning some versions of PyTorch give. With
``component = None`` the whole network is one matrix: nodes of different components are pairs
without edges.
"""

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln

from .generators import _choose, _swap_edges
from .network import _quiet_sparse

_EXACT_SWAPS = 64   # up to this many nodes, all swaps are tried, with exact gains


def _pairs(net, component, weight):
   """
   Distinct connected ordered pairs and their values, self loops left out.

   :param net: Network.
   :param component: Component index, or ``None`` for the whole network.
   :param weight: Edge attribute name used as weight, or ``None``.
   :return: ``(n, u, v, a, sizes)``: number of nodes, int64 sources and targets, float64 values
      (on the device of the network), and the number of nodes of each component taken, in order.
   """
   comps = range(len(net.components)) if component is None else [component]
   us, vs, ws, sizes, offset = [], [], [], [], 0
   for c in comps:
      data = net.components[c]
      s, t = data.edge_index
      w = torch.ones(s.shape[0], dtype = torch.float64, device = s.device) if weight is None else net._numeric_edge(weight, c).double()
      us.append(s + offset)
      vs.append(t + offset)
      ws.append(w)
      sizes.append(data.num_nodes)
      offset += data.num_nodes
   u, v, w = torch.cat(us), torch.cat(vs), torch.cat(ws)
   keep = u != v
   with _quiet_sparse():
      A = torch.sparse_coo_tensor(torch.stack([u[keep], v[keep]]), w[keep], (offset, offset), check_invariants = False).coalesce()
   u, v = A.indices()
   a = A.values() if weight is not None else torch.ones_like(A.values())
   return offset, u, v, a, sizes


def _split(x, sizes, component):
   """
   A vector over the nodes taken, as one vector (one component) or a list of them (``None``).
   """
   return x if component is not None else list(torch.split(x, sizes))


def _rho(N, sa, qa, sad, sd, sd2):
   """
   Pearson correlation of :math:`a` and :math:`\\delta` over ``N`` pairs, from their sums.

   :param N: Number of pairs.
   :param sa: Sum of :math:`a`.
   :param qa: Sum of :math:`a^2`.
   :param sad: Sum of :math:`a \\delta`.
   :param sd: Sum of :math:`\\delta`.
   :param sd2: Sum of :math:`\\delta^2`.
   :return: The correlation (NaN where a variance is 0); tensors broadcast.
   """
   den = (N * qa - sa ** 2) * (N * sd2 - sd ** 2)
   return torch.where(den > 0, (N * sad - sa * sd) / den.clamp(min = 1e-300).sqrt(), torch.nan)


def _discrete_rho(n, k, wcc, wpp, qcc, qpp, sa, qa, ties):
   """
   Correlation of the discrete model for cores of ``k`` nodes, from the sums of :math:`a` and
   :math:`a^2` within the core (``wcc``, ``qcc``) and within the periphery (``wpp``, ``qpp``).
   """
   cc, pp = k * (k - 1), (n - k) * (n - k - 1)
   if ties == "present":
      sd = n * (n - 1) - pp
      return _rho(n * (n - 1), sa, qa, sa - wpp, sd, sd)
   if ties == "absent":
      return _rho(n * (n - 1), sa, qa, wcc, cc, cc)
   return _rho(cc + pp, wcc + wpp, qcc + qpp, wcc, cc, cc)


def _prefix(order, n, u, v, a, sa, qa, ties):
   """
   Best core made of the first nodes of an order.

   :param order: Nodes, most central first.
   :return: ``(x, rho)``: float64 core indicator and its correlation.
   """
   rank = torch.empty_like(order)
   rank[order] = torch.arange(n, device = order.device)
   hi, lo = torch.maximum(rank[u], rank[v]), torch.minimum(rank[u], rank[v])
   zero = torch.zeros(1, dtype = torch.float64, device = order.device)
   wcc = torch.cat([zero, torch.cumsum(torch.bincount(hi, a, minlength = n), 0)])   # pairs within the first k
   qcc = torch.cat([zero, torch.cumsum(torch.bincount(hi, a * a, minlength = n), 0)])
   wpp = sa - torch.cat([zero, torch.cumsum(torch.bincount(lo, a, minlength = n), 0)])   # pairs within the rest
   qpp = qa - torch.cat([zero, torch.cumsum(torch.bincount(lo, a * a, minlength = n), 0)])
   k = torch.arange(n + 1, dtype = torch.float64, device = order.device)
   r = torch.nan_to_num(_discrete_rho(n, k, wcc, wpp, qcc, qpp, sa, qa, ties), nan = -torch.inf)
   best = int(torch.argmax(r))
   x = torch.zeros(n, dtype = torch.float64, device = order.device)
   x[order[:best]] = 1
   return x, r[best]


def _local_search(x, n, bu, bv, b, b2, strength, strength2, sa, qa, ties):
   """
   Local search of the discrete model from a core indicator.

   Each round computes the correlation reached by moving each node alone, sorts the nodes whose
   move improves it, and moves the best prefix of them: the correlation of moving the :math:`t`
   first is exact, the pairs of moved nodes being accounted for, as in :func:`_prefix`. Moving the
   first one alone is a candidate, so every round improves the correlation.

   :param x: float64 core indicator (modified).
   :return: ``(x, rho)``.
   """
   k = x.sum()
   while True:   # rho increases at every round, so this ends
      S = torch.zeros_like(x).index_add_(0, bu, b * x[bv])   # strength towards the core
      S2 = torch.zeros_like(x).index_add_(0, bu, b2 * x[bv])
      T, T2 = strength - S, strength2 - S2
      wcc, qcc, wpp, qpp = (S * x).sum() / 2, (S2 * x).sum() / 2, (T * (1 - x)).sum() / 2, (T2 * (1 - x)).sum() / 2
      rho = torch.nan_to_num(_discrete_rho(n, k, wcc, wpp, qcc, qpp, sa, qa, ties), nan = -torch.inf)
      d = 1 - 2 * x   # +1: periphery to core
      r = torch.nan_to_num(_discrete_rho(n, k + d, wcc + d * S, wpp - d * T, qcc + d * S2, qpp - d * T2, sa, qa, ties), nan = -torch.inf)
      cand = torch.nonzero(r > rho + 1e-12 * rho.abs().clamp(min = 1)).flatten()
      if not len(cand):
         break
      cand = cand[torch.argsort(r[cand], descending = True)]
      dc = d[cand]
      rank = torch.full((n,), -1, dtype = torch.long, device = x.device)
      rank[cand] = torch.arange(len(cand), device = x.device)
      inner = (rank[bu] >= 0) & (rank[bv] >= 0)   # pairs of candidates, both directions
      hi = torch.maximum(rank[bu], rank[bv])[inner]
      dd = (d[bu] * d[bv])[inner]
      pair = torch.cumsum(torch.bincount(hi, dd * b[inner], minlength = len(cand)), 0) / 2
      pair2 = torch.cumsum(torch.bincount(hi, dd * b2[inner], minlength = len(cand)), 0) / 2
      rt = _discrete_rho(n, k + torch.cumsum(dc, 0), wcc + torch.cumsum(dc * S[cand], 0) + pair, wpp - torch.cumsum(dc * T[cand], 0) + pair, qcc + torch.cumsum(dc * S2[cand], 0) + pair2, qpp - torch.cumsum(dc * T2[cand], 0) + pair2, sa, qa, ties)
      t = int(torch.argmax(torch.nan_to_num(rt, nan = -torch.inf))) + 1
      x[cand[:t]] += dc[:t]
      k = k + dc[:t].sum()
   return x, float(rho)


def discrete_core_periphery(net, component = 0, core_periphery_ties = "present", weight = None, restarts = 10, seed = None):
   """
   Discrete core-periphery model (Borgatti and Everett, 2000): the division of the nodes into a
   core and a periphery that best correlates with the network (see the module description).

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param core_periphery_ties: Ideal for the pairs of a core and a peripheral node: ``"present"``
      (connected), ``"absent"`` (not connected) or ``"missing"`` (left out of the correlation).
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param restarts: Number of randomized starting partitions, besides the one from the strengths.
   :param seed: Seed of the randomized starts, or ``None`` for a random seed.
   :return: ``(core, rho)``: a boolean vector over the nodes of the component, on its device,
      ordered by internal index (as node attributes), ``True`` for the core; or a list of them,
      one per component, with ``component = None``. ``rho`` is the correlation reached (NaN if no
      division has a correlation, e.g. without edges).
   :raises ValueError: If ``core_periphery_ties`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      core, rho = gsk.discrete_core_periphery(net)
      net.set_node_attribute("core", core)
      core, rho = gsk.discrete_core_periphery(net, core_periphery_ties = "missing", weight = "weight")
   """
   if core_periphery_ties not in ("present", "absent", "missing"):
      raise ValueError(f"unknown core_periphery_ties: {core_periphery_ties}")
   n, u, v, a, sizes = _pairs(net, component, weight)
   dev = u.device
   sa, qa = a.sum(), (a * a).sum()
   with _quiet_sparse():
      B = torch.sparse_coo_tensor(torch.stack([torch.cat([u, v]), torch.cat([v, u])]), torch.stack([torch.cat([a, a]), torch.cat([a * a, a * a])], 1), (n, n, 2), check_invariants = False).coalesce()
   bu, bv = B.indices()   # B = A + A^T, and the same with a^2
   b, b2 = B.values().T
   strength, strength2 = torch.bincount(bu, b, minlength = n), torch.bincount(bu, b2, minlength = n)
   gen = torch.Generator(device = dev)
   gen.manual_seed(int(np.random.default_rng(seed).integers(2 ** 62)))
   factors = [torch.ones_like(strength)] + [torch.exp(torch.randn(n, dtype = torch.float64, device = dev, generator = gen)) for _ in range(restarts)]
   best_x, best_rho = None, -torch.inf
   for z in factors:   # log-normal factors, sigma 1
      x, _ = _prefix(torch.argsort(strength * z, descending = True, stable = True), n, u, v, a, sa, qa, core_periphery_ties)
      x, rho = _local_search(x, n, bu, bv, b, b2, strength, strength2, sa, qa, core_periphery_ties)
      if best_x is None or rho > best_rho:
         best_x, best_rho = x, rho
   return _split(best_x > 0.5, sizes, component), best_rho if np.isfinite(best_rho) else float("nan")


def continuous_core_periphery(net, component = 0, weight = None, tol = 1e-5, max_iter = 1000):
   """
   Continuous core-periphery model (Borgatti and Everett, 2000): the coreness of every node, non
   negative, such that the products :math:`c_i c_j` best correlate with the network (see the module
   description).

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param tol: The search stops when ten iterations improve :math:`\\rho` by less than this
      fraction.
   :param max_iter: Largest number of L-BFGS iterations.
   :return: ``(coreness, rho)``: a float64 vector over the nodes of the component, on its device,
      ordered by internal index (as node attributes), with largest value 1; or a list of them, one
      per component, with ``component = None``. ``rho`` is the correlation reached (NaN without
      edges).

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      coreness, rho = gsk.continuous_core_periphery(net)
      net.set_node_attribute("coreness", coreness)
   """
   n, u, v, a, sizes = _pairs(net, component, weight)
   dev = u.device
   N, sa, qa = n * (n - 1), a.sum(), (a * a).sum()
   strength = torch.bincount(u, a, minlength = n) + torch.bincount(v, a, minlength = n)
   if not len(a) or strength.max() <= 0:
      return _split(torch.zeros(n, dtype = torch.float64, device = dev), sizes, component), float("nan")

   def rho(x):
      c = x * x
      s1, s2 = c.sum(), (c * c).sum()
      sd, sd2 = s1 ** 2 - s2, s2 ** 2 - (c ** 4).sum()
      return _rho(N, sa, qa, (a * c[u] * c[v]).sum(), sd, sd2)

   x = (strength.clamp(min = 0) / strength.max()).sqrt().requires_grad_()
   opt = torch.optim.LBFGS([x], max_iter = 10, max_eval = 250, history_size = 20, tolerance_grad = 0, tolerance_change = 0, line_search_fn = "strong_wolfe")

   def closure():
      opt.zero_grad()
      loss = -rho(x)
      loss.backward()
      return loss

   last = -float(opt.step(closure).detach())   # rho before
   for _ in range(max(max_iter // 10 - 1, 0)):   # ten iterations per step, the history is kept
      now = -float(opt.step(closure).detach())
      if not now > last + tol * abs(last):
         break
      last = now
   with torch.no_grad():
      c = x * x
      r = rho(x)
      c = c / c.max()
   return _split(c.detach(), sizes, component), float(r)


def _transition(n, alpha, beta, dev):
   """
   Values of the transition function of Rombach et al., largest first, summing to 1.

   :param n: Number of nodes.
   :param alpha: Sharpness of the boundary between core and periphery, in [0, 1].
   :param beta: Fraction of nodes in the periphery, in [0, 1].
   :param dev: Device.
   :return: float64 vector of length ``n``.
   """
   b = int(np.floor(beta * n))
   i = torch.arange(1, n + 1, dtype = torch.float64, device = dev)
   x = torch.where(i <= b, i * (1 - alpha) / (2 * max(b, 1)), (i - b) * (1 - alpha) / (2 * max(n - b, 1)) + (1 + alpha) / 2)
   return (x / x.sum()).flip(0)


def _f(x, y, p):
   """
   Core matrix entry of two core values: their product, or their ``p``-norm.
   """
   return x * y if p is None else (x ** p + y ** p) ** (1 / p)


def _quality(X, bu, bv, b, p):
   """
   Core quality of each column of ``X``: :math:`\\sum_{i < j} B_{ij} f(x_i, x_j)`.
   """
   return (b[:, None] * _f(X[bu], X[bv], p)).sum(0) / 2



def _neighbor_sums(nodes, cols, y, X, ptr, bv, b, p):
   """
   :math:`F_i(y) = \\sum_k B_{ik} f(y, x_k)` for each entry, with the core values of column
   ``cols`` of ``X``.

   :param nodes: Node :math:`i` of each entry.
   :param cols: Column of each entry.
   :param y: Value of each entry.
   :return: float64 vector, one value per entry.
   """
   cnt = ptr[nodes + 1] - ptr[nodes]
   e = torch.repeat_interleave(ptr[nodes] - torch.cumsum(cnt, 0) + cnt, cnt) + torch.arange(int(cnt.sum()), device = X.device)
   entry = torch.repeat_interleave(torch.arange(len(nodes), device = X.device), cnt)
   vals = b[e] * _f(y[entry], X[bv[e], cols[entry]], p)
   return torch.zeros(len(nodes), dtype = X.dtype, device = X.device).index_add_(0, entry, vals)


def _swap_round(X, R, ptr, bu, bv, b, keys, n, p, offsets):
   """
   One round of swaps of core values between pairs of nodes, in every column of ``X``.

   Each node is paired with the nodes ``offsets`` ranks above and below it (by core value), and
   with the node holding the value it would get if the nodes were ranked by the gradient of the
   quality; the exact gain of each swap is computed (for the :math:`p`-norm, only for the partner
   with the best first order gain, :math:`(x_j - x_i) (g_i - g_j)` with :math:`g` the gradient). Then the pairs that are the best for both of
   their nodes are swapped together, if that improves the quality more than the best pair alone;
   otherwise only the best pair is swapped.

   :return: ``(X, R, improved)``, ``improved`` a boolean per column.
   """
   dev = X.device
   P = X.shape[1]
   ar, cols = torch.arange(n, device = dev), torch.arange(P, device = dev)
   at = torch.argsort(X, 0, descending = True)   # node at each rank
   rank = torch.empty_like(at).scatter_(0, at, ar[:, None].expand(n, P))
   if p is None:
      g = torch.zeros_like(X).index_add_(0, bu, b[:, None] * X[bv])
   else:
      xu, xv = X[bu], X[bv]
      g = torch.zeros_like(X).index_add_(0, bu, b[:, None] * xu ** (p - 1) * (xu ** p + xv ** p).clamp(min = 1e-300) ** (1 / p - 1))
   grank = torch.empty_like(at).scatter_(0, torch.argsort(g, 0, descending = True), ar[:, None].expand(n, P))
   J = torch.stack([torch.gather(at, 0, grank)] + [torch.gather(at, 0, (rank + o).clamp(0, n - 1)) for o in offsets], 2)
   I, C = ar[:, None, None].expand_as(J), cols[None, :, None].expand_as(J)
   xi, xj = X[I, C], X[J, C]
   gain = torch.where(I == J, 0., (xj - xi) * (g[I, C] - g[J, C]))   # exact for the product, but for pairs of neighbors
   if p is not None and n > _EXACT_SWAPS:   # the best partner by this first order gain, then its exact gain
      k = gain.argmax(2, keepdim = True)
      J, I, C, xi, xj = [torch.gather(t, 2, k) for t in (J, I, C, xi, xj)]
   key = I * n + J
   pos = torch.searchsorted(keys, key.flatten()).clamp(max = len(keys) - 1).view_as(key)
   bij = torch.where((keys[pos] == key) & (I != J), b[pos], 0.)
   if p is None:
      gain = gain - bij * (xi - xj) ** 2
   else:
      F = lambda nodes, y: _neighbor_sums(nodes.flatten(), C.flatten(), y.flatten(), X, ptr, bv, b, p).view_as(J)
      own = torch.zeros_like(X).index_add_(0, bu, b[:, None] * _f(X[bu], X[bv], p))   # F_i(x_i)
      gain = F(I, xj) - own[I, C] + F(J, xi) - own[J, C] - bij * (_f(xi, xi, p) + _f(xj, xj, p) - 2 * _f(xi, xj, p))
      gain = torch.where(I == J, 0., gain)
   best, k = gain.max(2)   # best partner of each node
   partner = torch.gather(J, 2, k[:, :, None])[:, :, 0]
   mutual = (torch.gather(partner, 0, partner) == ar[:, None]) & (best > 0) & (ar[:, None] < partner)
   Y = X.clone()
   i, c = torch.nonzero(mutual, as_tuple = True)
   j = partner[i, c]
   Y[i, c], Y[j, c] = X[j, c], X[i, c]
   RY = _quality(Y, bu, bv, b, p)
   top, ti = best.max(0)
   improved = top > 1e-12 * R.abs().clamp(min = 1e-300)
   alone = improved & ~(RY > R + top)
   Z = X.clone()
   tj = partner[ti, cols]
   Z[ti, cols], Z[tj, cols] = X[tj, cols], X[ti, cols]
   Xn = torch.where(alone[None, :], Z, torch.where(improved[None, :], Y, X))
   Rn = torch.where(alone, R + top, torch.where(improved, RY, R))
   return Xn, Rn, improved


def _rombach_fit(V, init, ptr, bu, bv, b, keys, n, p, max_iter):
   """
   Best assignment of the core values of each column of ``V`` to the nodes.

   Ascent by ranking (each node gets the value of its rank by the gradient of the quality, while
   this improves it), then rounds of swaps (:func:`_swap_round`) until a round improves the quality
   by less than one part in a million.

   :param V: Core values, largest first, one column per pair of parameters.
   :param init: Starting order of the nodes in each column (the first gets the largest value).
   :return: ``(X, R)``: core values of the nodes and qualities, one column per pair.
   """
   P = V.shape[1]
   X = torch.zeros_like(V).scatter_(0, init, V)
   R = _quality(X, bu, bv, b, p)
   if not len(b):   # no edges: nothing to improve
      return X, R
   for _ in range(max_iter):
      if p is None:
         g = torch.zeros_like(X).index_add_(0, bu, b[:, None] * X[bv])
      else:
         xu, xv = X[bu], X[bv]
         g = torch.zeros_like(X).index_add_(0, bu, b[:, None] * xu ** (p - 1) * (xu ** p + xv ** p).clamp(min = 1e-300) ** (1 / p - 1))
      Y = torch.zeros_like(V).scatter_(0, torch.argsort(g, 0, descending = True), V)
      RY = _quality(Y, bu, bv, b, p)
      better = RY > R + 1e-12 * R.abs()
      if not bool(better.any()):
         break
      X, R = torch.where(better[None, :], Y, X), torch.where(better, RY, R)
   offsets = [s * o for o in (range(1, n) if n <= _EXACT_SWAPS else 2 ** np.arange(int(np.log2(n)) + 1)) for s in (1, -1)]   # all ranks, or powers of 2
   act = torch.ones(P, dtype = torch.bool, device = V.device)
   for _ in range(max_iter):
      idx = torch.nonzero(act).flatten()
      if not len(idx):
         break
      old = R[idx]
      X[:, idx], R[idx], imp = _swap_round(X[:, idx], old, ptr, bu, bv, b, keys, n, p, offsets)
      act[idx] = imp & (R[idx] > old + 1e-6 * old.abs())
   return X, R


def rombach_core_periphery(net, component = 0, alpha = None, beta = None, p = None, weight = None, restarts = 0, seed = None, max_iter = 1000):
   """
   Core-periphery structure of Rombach, Porter, Fowler and Mucha (2014): a core score for every
   node, aggregated over a range of core shapes, from sharp to smooth and from large to small
   cores.

   For a pair of parameters :math:`(\\alpha, \\beta)`, the core values are those of the transition
   function, on the :math:`k`-th node from the most peripheral

   .. math::

      C^*_k = \\begin{cases} \\frac{k (1 - \\alpha)}{2 \\lfloor \\beta n \\rfloor} & k \\le \\lfloor \\beta n \\rfloor \\\\
      \\frac{(k - \\lfloor \\beta n \\rfloor) (1 - \\alpha)}{2 (n - \\lfloor \\beta n \\rfloor)} + \\frac{1 + \\alpha}{2} & k > \\lfloor \\beta n \\rfloor \\end{cases}

   (normalized to sum 1): :math:`\\beta` is the fraction of nodes in the periphery, and
   :math:`\\alpha` the jump of the core values between periphery and core, 1 for a sharp boundary
   (core values 0 or 1), 0 for a smooth one. The values are assigned to the nodes so as to
   maximize the core quality :math:`R = \\sum_{i, j} A_{ij} C_{ij}`, with
   :math:`C_{ij} = C_i C_j`, or, with ``p``, the :math:`p`-norm
   :math:`C_{ij} = (C_i^p + C_j^p)^{1 / p}`, under which a pair is as central as its most central
   node (as ``p`` grows). The core score of node :math:`i` is
   :math:`CS(i) = Z \\sum_{(\\alpha, \\beta)} C_i(\\alpha, \\beta) R(\\alpha, \\beta)`, with
   :math:`Z` such that the largest score is 1.

   The paper assigns values by simulated annealing on a grid of 100 by 100 pairs. Here, for all
   pairs at once on the device of the network, nodes start ranked by strength; then each node gets
   the value of its rank by the gradient of :math:`R` (how much :math:`R` grows with its core
   value), as long as this improves :math:`R`; then pairs of nodes swap values, in rounds, until a
   round improves :math:`R` by less than one part in a million. Every node tries the nodes 1, 2, 4,
   8, ... ranks above and below it (all ranks in networks of at most 64 nodes) and the node whose
   value it would get by its gradient; the pairs that are each other's best swap are swapped
   together. This finds local optima: ``restarts`` from random orders of the nodes may find
   better ones. Self loops are
   ignored; without ``weight``, :math:`A_{ij}` is 1 for connected pairs, with it, the sum of the
   weights; directed edges count in both directions, as the paper is for undirected networks.
   In multilayer networks the supra-graph is used. With the :math:`p`-norm, the exact gain of a
   swap needs a pass over the edges of its two nodes, so every node tries only the partner with
   the best first order gain, and the search is slower than with the product.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network
      (nodes of different components are pairs without edges).
   :param alpha: Value or iterable of values of :math:`\\alpha` in [0, 1]; ``None`` for 0.05, 0.1,
      ..., 1.
   :param beta: Value or iterable of values of :math:`\\beta` in [0, 1]; ``None`` for 0.05, 0.1,
      ..., 1.
   :param p: Exponent of the :math:`p`-norm of core values (at least 1), or ``None`` for their
      product.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param restarts: Number of random starting orders of the nodes, besides the one by strength;
      for each pair of parameters, the best assignment is kept.
   :param seed: Seed of the random starts, or ``None`` for a random seed.
   :param max_iter: Largest number of rounds of each phase of the search.
   :return: ``(score, quality)``: the core scores, a float64 vector over the nodes of the component,
      on its device, ordered by internal index (as node attributes), with largest value 1 (or a
      list of them, one per component, with ``component = None``); and a
      :class:`pandas.DataFrame` with columns ``alpha``, ``beta`` and ``R``, the core quality
      reached for each pair.
   :raises ValueError: If :math:`\\alpha` or :math:`\\beta` are not in [0, 1], or ``p`` is below 1.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      score, quality = gsk.rombach_core_periphery(net)
      net.set_node_attribute("core_score", score)
      score, quality = gsk.rombach_core_periphery(net, alpha = 1, beta = 0.8)   # sharp core of 20%
      score, quality = gsk.rombach_core_periphery(net, p = 2)
   """
   grid = np.arange(1, 21) / 20
   alpha = grid if alpha is None else np.atleast_1d(np.asarray(alpha, dtype = float))
   beta = grid if beta is None else np.atleast_1d(np.asarray(beta, dtype = float))
   if ((alpha < 0) | (alpha > 1)).any() or ((beta < 0) | (beta > 1)).any():
      raise ValueError("alpha and beta must be between 0 and 1")
   if p is not None and p < 1:
      raise ValueError("p must be at least 1")
   n, u, v, a, sizes = _pairs(net, component, weight)
   dev = u.device
   with _quiet_sparse():
      B = torch.sparse_coo_tensor(torch.stack([torch.cat([u, v]), torch.cat([v, u])]), torch.cat([a, a]), (n, n), check_invariants = False).coalesce()
   bu, bv = B.indices()   # B = A + A^T
   b = B.values()
   keys = bu * n + bv
   ptr = torch.cat([torch.zeros(1, dtype = torch.long, device = dev), torch.cumsum(torch.bincount(bu, minlength = n), 0)])
   gen = torch.Generator(device = dev)
   gen.manual_seed(int(np.random.default_rng(seed).integers(2 ** 62)))
   inits = [torch.argsort(torch.bincount(bu, b, minlength = n), descending = True, stable = True)] + [torch.randperm(n, device = dev, generator = gen) for _ in range(restarts)]
   pairs = [(x, y) for x in alpha for y in beta]
   K = 2 * (n if n <= _EXACT_SWAPS else int(np.log2(n)) + 1) + 1
   step = max(1, 2 ** 24 // max(n * K, 2 * len(b), 1) // len(inits))
   score = torch.zeros(n, dtype = torch.float64, device = dev)
   quality = []
   for s in range(0, len(pairs), step):
      V = torch.stack([_transition(n, x, y, dev) for x, y in pairs[s:s + step]], 1).nan_to_num(0)
      P = V.shape[1]
      init = torch.cat([o[:, None].expand(n, P) for o in inits], 1)   # every pair from every start
      X, R = _rombach_fit(V.repeat(1, len(inits)), init, ptr, bu, bv, b, keys, n, p, max_iter)
      X, R = X.view(n, len(inits), P), R.view(len(inits), P)
      best = R.argmax(0)
      X, R = X[:, best, torch.arange(P, device = dev)], R[best, torch.arange(P, device = dev)]
      score += X @ R
      quality += R.tolist()
   if score.max() > 0:
      score = score / score.max()
   return _split(score, sizes, component), pd.DataFrame({"alpha": [x for x, _ in pairs], "beta": [y for _, y in pairs], "R": quality})




def _biadjacency(net):
   """
   Edges of the biadjacency matrix of a bipartite network.

   :param net: Network.
   :return: ``(r, c, R, C, rows, columns)``: int64 row and column of each distinct edge (on the
      device of the network), numbers of rows and of columns, and the input ids of rows and
      columns, in matrix order.
   :raises ValueError: If the network is not bipartite, or is multilayer.
   """
   if net.bipartite_labels is None:
      raise ValueError("nestedness needs a bipartite network")
   if net.layer_labels is not None:
      raise ValueError("nestedness is not defined for multilayer networks")
   rs, cs, rows, columns, R, C = [], [], [], [], 0, 0
   for c, data in enumerate(net.components):
      k = int(net.bipartite_split[c])
      s, t = net._input_edges(c)[0]
      a, b = torch.minimum(s, t), torch.maximum(s, t)   # type 0 first in every component
      rs.append(a + R)
      cs.append(b - k + C)
      rows.append(net.node_ids[c][:k])
      columns.append(net.node_ids[c][k:])
      R, C = R + k, C + data.num_nodes - k
   r, c = torch.cat(rs), torch.cat(cs)
   key = torch.unique(r * C + c) if len(r) else r
   return key // max(C, 1), key % max(C, 1), R, C, np.concatenate(rows), np.concatenate(columns)


def _pair_sums(r, c, kr, groups, n):
   """
   For each group of matrices, the sum over pairs of rows sharing a column, with different
   degrees, of 1 over the smaller degree.

   :param r: Row of each edge (rows of different matrices are different).
   :param c: Column of each edge (columns of different matrices are different).
   :param kr: Degree of the row of each edge.
   :param groups: Matrix of each edge.
   :param n: Number of matrices.
   :return: float64 sums, one per matrix.
   """
   K = int(kr.max()) + 1 if len(kr) else 1
   key = c * K + kr
   keys = torch.sort(key).values
   above = torch.searchsorted(keys, (c + 1) * K) - torch.searchsorted(keys, key, right = True)   # rows of the column with more neighbors
   return torch.zeros(n, dtype = torch.float64, device = r.device).index_add_(0, groups, above.double() / kr.double())


def _nodf(r, c, R, C, groups, n):
   """
   NODF of ``n`` matrices of ``R`` rows and ``C`` columns, whose edges are given together.

   :param r: Row of each edge, from 0 to ``R - 1``.
   :param c: Column of each edge.
   :param groups: Matrix of each edge, from 0 to ``n - 1``.
   :return: ``(nodf, rows, columns)``, float64 tensors of length ``n``.
   """
   rg, cg = r + groups * R, c + groups * C   # every matrix has its own rows and columns
   kr = torch.bincount(rg, minlength = n * R)[rg]
   kc = torch.bincount(cg, minlength = n * C)[cg]
   sr, sc = _pair_sums(rg, cg, kr, groups, n), _pair_sums(cg, rg, kc, groups, n)
   pr, pc = R * (R - 1) / 2, C * (C - 1) / 2
   rows = 100 * sr / pr if pr else torch.full_like(sr, torch.nan)
   cols = 100 * sc / pc if pc else torch.full_like(sc, torch.nan)
   both = 100 * (sr + sc) / (pr + pc) if pr + pc else torch.full_like(sr, torch.nan)
   return both, rows, cols


def _null(r, c, R, C, model, gen):
   """
   Edges of one random matrix of a null model.

   :return: ``(r, c)``.
   """
   dev, m = r.device, len(r)
   if model == "equiprobable":
      key = _choose(R * C, m, gen, dev)
      return key // C, key % C
   if model == "proportional":
      pr = torch.bincount(r, minlength = R).double() / C
      pc = torch.bincount(c, minlength = C).double() / R
      out_r, out_c = [], []
      step = max(1, 2 ** 24 // max(C, 1))
      for a in range(0, R, step):   # blocks of rows
         P = (pr[a:a + step, None] + pc[None, :]) / 2
         i, j = torch.nonzero(torch.rand(P.shape, dtype = torch.float64, device = dev, generator = gen) < P, as_tuple = True)
         out_r.append(i + a)
         out_c.append(j)
      return torch.cat(out_r), torch.cat(out_c)
   a, b = _swap_edges(r, c + R, 10 * m, True, gen)
   return a, b - R


def nestedness(net, null_model = None, samples = 100, seed = None):
   """
   NODF nestedness of a bipartite network, optionally compared with a null model (see the module
   description).

   :param net: A bipartite :class:`~graphoskop.Network`.
   :param null_model: ``"equiprobable"``, ``"proportional"``, ``"fixed"``, or ``None`` for no
      comparison.
   :param samples: Number of random matrices of the null model.
   :param seed: Seed of the random matrices, or ``None`` for a random seed.
   :return: A dictionary with ``nodf`` (between 0 and 100), ``nodf_rows`` and ``nodf_columns`` (the
      averages over pairs of rows, nodes of type 0, and over pairs of columns, nodes of type 1;
      NaN with fewer than two of them); with a null model also ``null_mean``, ``null_std``,
      ``z`` (the z-score of ``nodf``; NaN if all random matrices have the same value, e.g. when the
      degrees allow only one matrix), ``p_value`` (the fraction of random matrices at least as
      nested, counting the network itself among them: :math:`(1 + b) / (1 + \\text{samples})`) and
      ``null`` (the NODF of every random matrix, a NumPy array).
   :raises ValueError: If the network is not bipartite or is multilayer, or ``null_model`` is not
      valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("pollination.tsv", "species.tsv", bipartite = True)
      gsk.nestedness(net)["nodf"]
      res = gsk.nestedness(net, null_model = "fixed", samples = 1000, seed = 1)
      res["z"], res["p_value"]
   """
   if null_model not in (None, "equiprobable", "proportional", "fixed"):
      raise ValueError(f"unknown null_model: {null_model}")
   r, c, R, C, _, _ = _biadjacency(net)
   zero = torch.zeros(len(r), dtype = torch.long, device = r.device)
   nodf, rows, cols = (float(x[0]) for x in _nodf(r, c, R, C, zero, 1))
   out = {"nodf": nodf, "nodf_rows": rows, "nodf_columns": cols}
   if null_model is None:
      return out
   gen = torch.Generator(device = r.device)
   gen.manual_seed(int(np.random.default_rng(seed).integers(2 ** 62)))
   parts = [_null(r, c, R, C, null_model, gen) for _ in range(samples)]
   groups = torch.cat([torch.full((len(a),), i, dtype = torch.long, device = r.device) for i, (a, _) in enumerate(parts)])
   null = _nodf(torch.cat([a for a, _ in parts]), torch.cat([b for _, b in parts]), R, C, groups, samples)[0].cpu().numpy()
   std = null.std(ddof = 1) if samples > 1 else float("nan")
   out.update(null_mean = float(null.mean()), null_std = float(std), z = float((nodf - null.mean()) / std) if std > 1e-9 * max(1, abs(null.mean())) else float("nan"), p_value = float((1 + (null >= nodf - 1e-9).sum()) / (1 + samples)), null = null)
   return out


def _area(p):
   """
   Area below the fill line of parameter ``p``: :math:`1 - \\Gamma(1 + 1 / p)^2 / \\Gamma(1 + 2 / p)`.
   """
   return 1 - np.exp(2 * gammaln(1 + 1 / p) - gammaln(1 + 2 / p))


def nestedness_isocline(net, method = "fill", points = 101):
   """
   Isocline of the nested biadjacency matrix (see the module description): the line
   :math:`(1 - x)^p + (1 - y)^p = 1` separating the expected ones, in the top-left corner of the
   matrix with rows and columns sorted by decreasing degree, from the expected zeros.

   With ``method = "fill"``, :math:`p` makes the area below the line equal to the fill of the
   matrix (the fraction of ones), as in the nestedness temperature. With ``method = "fit"``,
   :math:`p` makes the line closest (least squares) to the boundary of the ones of the sorted
   matrix if it were perfectly nested: column :math:`j` (at :math:`x = (j - 0.5) / C`) filled down
   to :math:`y = k_j / R`, row :math:`i` (at :math:`y = (i - 0.5) / R`) filled up to
   :math:`x = k_i / C`; its distance from the fill line shows how the ones are spread.

   :param net: A bipartite :class:`~graphoskop.Network`.
   :param method: ``"fill"`` or ``"fit"``.
   :param points: Number of points of the line returned.
   :return: A dictionary with ``p``; ``fill``; ``rows`` and ``columns``, the input ids of the
      nodes of type 0 and of type 1 by decreasing degree (ties in network order), the order of the
      matrix; and ``line``, a :class:`pandas.DataFrame` of ``points`` points ``x``, ``y`` of the
      line, for plotting. ``p`` is NaN for an empty or full matrix.
   :raises ValueError: If the network is not bipartite or is multilayer, or ``method`` is not
      valid.

   Example:

   .. code-block:: python

      iso = gsk.nestedness_isocline(net)
      iso["p"]
      f = lambda x: 1 - (1 - (1 - x) ** iso["p"]) ** (1 / iso["p"])   # the line, as a function
      iso["line"].plot(x = "x", y = "y")
   """
   if method not in ("fill", "fit"):
      raise ValueError(f"unknown method: {method}")
   r, c, R, C, rows, columns = _biadjacency(net)
   kr = torch.bincount(r, minlength = R).cpu().numpy()
   kc = torch.bincount(c, minlength = C).cpu().numpy()
   fill = len(r) / (R * C) if R * C else float("nan")
   orow, ocol = np.argsort(-kr, kind = "stable"), np.argsort(-kc, kind = "stable")
   f = lambda x, p: 1 - np.clip(1 - (1 - x) ** p, 0, 1) ** (1 / p)
   p = float("nan")
   if 0 < fill < 1:
      if method == "fill":
         p = float(np.exp(brentq(lambda q: _area(np.exp(q)) - fill, -30, 30)))
      else:
         x = np.r_[(np.arange(C) + 0.5) / C, kr[orow] / C]   # columns, then rows (the line is symmetric)
         y = np.r_[kc[ocol] / R, (np.arange(R) + 0.5) / R]
         p = float(np.exp(minimize_scalar(lambda q: ((f(x, np.exp(q)) - y) ** 2).sum(), bounds = (-10, 10), method = "bounded").x))
   xs = np.linspace(0, 1, points)
   line = pd.DataFrame({"x": xs, "y": f(xs, p) if np.isfinite(p) else np.full(points, np.nan)})
   return {"p": p, "fill": fill, "rows": rows[orow], "columns": columns[ocol], "line": line}
