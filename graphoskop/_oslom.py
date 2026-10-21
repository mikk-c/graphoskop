"""
OSLOM kernels (see :func:`~graphoskop.oslom_communities`): the statistical significance of a
cluster from the order statistics of the r-scores of the nodes (Lancichinetti, Radicchi, Ramasco
and Fortunato, 2011), and the single cluster analysis, compiled (Numba).
"""

import math
import numpy as np
from numba import njit

_QMAX = 200
_TABLE = {}


def _phi_table():
   """
   Cumulative distribution :math:`\\phi(x, N)` of :math:`c_m = \\min_q \\Omega_q(r_q)`, the
   smallest probability of the order statistics of :math:`N` uniform scores (:math:`q` up to
   200), by simulation: 2000 samples for 30 values of :math:`N` from 1 to :math:`10^5` (evenly
   spaced on a log scale; larger :math:`N` use the last, the distribution converging), computed
   once and cached.

   :return: ``(grid, samples)``: the values of :math:`N` and, for each, the sorted samples.
   """
   if not _TABLE:
      from scipy.special import betainc
      rng = np.random.default_rng(12345)
      grid = np.unique(np.round(np.geomspace(1, 1e5, 30)).astype(np.int64))
      rows = []
      for N in grid:
         Q = int(min(N, _QMAX))
         E = rng.exponential(size = (2000, Q))
         tot = E.sum(1) + rng.gamma(N + 1 - Q, size = 2000)   # the remaining spacings
         r = np.cumsum(E, 1) / tot[:, None]
         q = np.arange(1, Q + 1)
         rows.append(np.sort(betainc(q[None, :], N - q[None, :] + 1, r).min(1)))
      _TABLE["grid"], _TABLE["samples"] = grid, np.array(rows)
   return _TABLE["grid"], _TABLE["samples"]


@njit(cache = True, inline = "always")
def _lchoose(lf, a, b):
   """:math:`\\log \\binom{a}{b}` from the table ``lf`` of :math:`\\log k!`."""
   return lf[a] - lf[b] - lf[a - b]


@njit(cache = True)
def _tail(x, k, K, Np, lf):
   """
   :math:`P(X \\geq x)` and :math:`P(X = x)`, :math:`X` hypergeometric (``k`` draws, ``K``
   successes among ``Np``), summing the terms by their ratios, with the table ``lf`` of
   :math:`\\log k!`.
   """
   hi = min(k, K)
   lo = max(k - (Np - K), 0)
   if x > hi:
      return 0., 0.
   if x < lo:
      return 1., 0.
   term = math.exp(_lchoose(lf, K, x) + _lchoose(lf, Np - K, k - x) - _lchoose(lf, Np, k))
   first, sf = term, 0.
   for j in range(x, hi + 1):
      sf += term
      term *= (K - j) * (k - j) / ((j + 1.) * (Np - K - k + j + 1.))
   return min(sf, 1.), first


@njit(cache = True)
def _rscore(kin, k, out, eint, M2, lf):
   """
   Randomized r-score of a node of degree ``k`` with ``kin`` links to a cluster with ``out``
   stubs pointing out and ``eint`` internal links (``M2`` stubs in the network): uniform between
   :math:`P(X > k_{in})` and :math:`P(X \\geq k_{in})`, :math:`X` the links to the cluster in the
   configuration model (hypergeometric approximation: ``k`` draws among the
   :math:`M_2 - 2 e_{int} - k` stubs not inside the cluster, of which ``out`` leave it).
   """
   Np = max(M2 - 2 * eint - k, k, out)
   if kin <= 0:
      sf1, _ = _tail(1, k, out, Np, lf)
      return sf1 + np.random.random() * (1. - sf1)
   sf, first = _tail(kin, k, out, Np, lf)
   return sf - first + np.random.random() * first


@njit(cache = True)
def _omega(q, r, N):
   """:math:`\\Omega_q(r) = P(\\mathrm{Bin}(N, r) \\geq q)`, the distribution of the q-th score."""
   if r <= 0.:
      return 0.
   if r >= 1.:
      return 1.
   term = math.exp(N * math.log1p(-r))
   ratio = r / (1. - r)
   s = 0.
   for j in range(q):   # successive terms by their ratio
      s += term
      term *= (N - j) / (j + 1.) * ratio
   return max(1. - s, 0.)


@njit(cache = True)
def _phi(x, N, grid, samples):
   """:math:`\\phi(x, N)` from the simulated table (nearest :math:`N` on a log scale)."""
   i = np.searchsorted(grid, N)
   if i >= len(grid):
      i = len(grid) - 1
   elif i > 0 and math.log(N) - math.log(grid[i - 1]) < math.log(grid[i]) - math.log(N):
      i -= 1
   return np.searchsorted(samples[i], x, side = "right") / samples.shape[1]


@njit(cache = True)
def _clean(ptr, nbr, deg, M2, lf, init, grow, P, runs, grid, samples, seed, inC, kin, flag):
   """
   Single cluster analysis of OSLOM from the nodes ``init`` (after adding the ``grow`` best
   neighbors, for seeds), repeated ``runs`` times (the scores are random): alternately, the
   external nodes are ranked by r-score and the best :math:`q` are added, for the first
   :math:`q` with :math:`\\phi(\\Omega_q(r_q)) < P`, and the worst internal node is removed while not
   significant (:math:`\\phi(\\Omega_1(r)) \\geq P`, as an external node of the rest), once no more
   nodes can be added; until nothing changes. ``lf`` is the table of :math:`\\log k!`; ``inC``, ``kin`` and ``flag`` are work arrays of one entry per node, all
   zero (left so).

   :return: The cluster: the nodes in more than half the non-empty outcomes (empty if at most half
      of the outcomes are non-empty).
   """
   np.random.seed(seed)
   n = len(deg)
   freq = np.zeros(n, dtype = np.int64)
   seen = np.empty(n, dtype = np.int64)
   nseen = 0
   good = 0
   members = np.empty(n, dtype = np.int64)
   touched = np.empty(n, dtype = np.int64)
   cand = np.empty(n, dtype = np.int64)
   rs = np.empty(n)
   for run in range(runs):
      size, nt, K, E = 0, 0, 0, 0
      for u in init:   # start
         inC[u] = True
         members[size] = u
         size += 1
         K += deg[u]
         E += kin[u]
         for e in range(ptr[u], ptr[u + 1]):
            x = nbr[e]
            kin[x] += 1
            if not flag[x]:
               flag[x] = True
               touched[nt] = x
               nt += 1
      for step in range(-1 if grow > 0 else 0, 1000):
         changed = False
         # add: rank external neighbors
         nc = 0
         for t in range(nt):
            x = touched[t]
            if kin[x] > 0 and not inC[x]:
               cand[nc] = x
               rs[nc] = _rscore(kin[x], deg[x], K - 2 * E, E, M2, lf)
               nc += 1
         if nc > 0:
            order = np.argsort(rs[:nc])
            Next = n - size
            if step < 0:   # seed: grow by the best neighbors
               qbest = min(grow, nc)
               ok = True
            else:   # the first q (in increasing order) whose q best are significant
               qbest, ok = 0, False
               for q in range(1, min(nc, 200) + 1):
                  if _phi(_omega(q, rs[order[q - 1]], Next), Next, grid, samples) < P:
                     qbest, ok = q, True
                     break
            if ok:
               for q in range(qbest):
                  u = cand[order[q]]
                  inC[u] = True
                  members[size] = u
                  size += 1
                  K += deg[u]
                  E += kin[u]
                  for e in range(ptr[u], ptr[u + 1]):
                     x = nbr[e]
                     kin[x] += 1
                     if not flag[x]:
                        flag[x] = True
                        touched[nt] = x
                        nt += 1
               changed = True
               continue   # add while significant nodes are found, then prune
         # prune: remove the worst internal node while not significant
         while size > 0:
            worst, wr = -1, -1.
            for p in range(size):
               u = members[p]
               r = _rscore(kin[u], deg[u], (K - deg[u]) - 2 * (E - kin[u]), E - kin[u], M2, lf)
               if r > wr:
                  worst, wr = p, r
            if _phi(_omega(1, wr, n - size + 1), n - size + 1, grid, samples) < P:
               break
            u = members[worst]
            members[worst] = members[size - 1]
            size -= 1
            inC[u] = False
            K -= deg[u]
            E -= kin[u]
            for e in range(ptr[u], ptr[u + 1]):
               kin[nbr[e]] -= 1
            changed = True
         if not changed or size == 0:
            break
      if size > 0:
         good += 1
         for p in range(size):
            u = members[p]
            if freq[u] == 0:
               seen[nseen] = u
               nseen += 1
            freq[u] += 1
      for p in range(size):   # reset the work arrays
         inC[members[p]] = False
      for t in range(nt):
         kin[touched[t]] = 0
         flag[touched[t]] = False
   out = np.empty(nseen, dtype = np.int64)
   no = 0
   if 2 * good > runs:
      for t in range(nseen):
         if 2 * freq[seen[t]] > good:
            out[no] = seen[t]
            no += 1
   return out[:no]
