"""
Link community kernels (see :func:`~graphoskop.hierarchical_link_communities`): single linkage
clustering of links with the partition density of Ahn, Bagrow and Lehmann (2010), compiled (Numba).
"""

import numpy as np
from numba import njit, types
from numba.typed import Dict


@njit(cache = True, inline = "always")
def _find(parent, x):
   while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
   return x


@njit(cache = True, inline = "always")
def _density(m, n):
   """Partition density term of a community with ``m`` links and ``n`` nodes (0 if ``n <= 2``)."""
   return m * (m - n + 1.) / ((n - 2.) * (n - 1.)) if n > 2 else 0.


@njit(cache = True)
def _single_linkage(N, a, b, p, q, sim):
   """
   Single linkage of the links ``(a, b)``: the pairs of links ``(p, q)`` (sharing a node) merge
   their clusters by decreasing similarity ``sim`` (sorted), equal similarities together. Every
   cluster keeps its number of links and of nodes (its nodes in a dictionary keyed by
   ``cluster * N + node``, merged smaller into larger), so the partition density
   :math:`D = \\frac{2}{M} \\sum_c m_c \\frac{m_c - n_c + 1}{(n_c - 2)(n_c - 1)}` is updated at
   every merge.

   :return: ``(best, threshold)``: the largest partition density and the similarity at which it is
      reached (``inf`` if never above 0: every link alone).
   """
   M = len(a)
   parent = np.arange(M)
   m = np.ones(M)
   n = np.full(M, 2.)
   head = np.empty(M, dtype = np.int64)   # node entries of each cluster, as linked lists
   nxt = np.full(2 * M, -1, dtype = np.int64)
   tail = np.empty(M, dtype = np.int64)
   node = np.empty(2 * M, dtype = np.int64)
   where = Dict.empty(key_type = types.int64, value_type = types.int64)
   for e in range(M):
      node[2 * e], node[2 * e + 1] = a[e], b[e]
      nxt[2 * e] = 2 * e + 1
      head[e], tail[e] = 2 * e, 2 * e + 1
      where[e * N + a[e]] = 2 * e
      where[e * N + b[e]] = 2 * e + 1
   D, best, threshold = 0., 0., np.inf
   i = 0
   while i < len(p):
      j = i
      while j < len(p) and sim[j] == sim[i]:
         x, y = _find(parent, p[j]), _find(parent, q[j])
         if x != y:
            if n[x] > n[y]:
               x, y = y, x   # x smaller, merged into y
            D -= _density(m[x], n[x]) + _density(m[y], n[y])
            shared = 0
            k = head[x]
            while k >= 0:
               nk = nxt[k]
               v = node[k]
               del where[x * N + v]
               if y * N + v in where:
                  shared += 1
               else:
                  where[y * N + v] = k
                  nxt[k] = -1
                  nxt[tail[y]] = k
                  tail[y] = k
               k = nk
            parent[x] = y
            m[y] += m[x]
            n[y] += n[x] - shared
            D += _density(m[y], n[y])
         j += 1
      if 2 * D / M > best + 1e-12:
         best, threshold = 2 * D / M, sim[i]
      i = j
   return best, threshold


@njit(cache = True)
def _cut(M, p, q, sim, threshold):
   """Clusters of links joined by the pairs with similarity at least ``threshold``."""
   parent = np.arange(M)
   for k in range(len(p)):
      if sim[k] >= threshold:
         x, y = _find(parent, p[k]), _find(parent, q[k])
         if x != y:
            parent[x] = y
   for e in range(M):
      parent[e] = _find(parent, e)
   return parent
