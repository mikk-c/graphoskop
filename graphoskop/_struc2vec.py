"""
Kernels of :func:`~graphoskop.struc2vec`, compiled (Numba): degree sequences of the rings of every
node, structural distances by dynamic time warping, and random walks on the multilayer graph.

The network is an undirected simple graph in CSR form (``ptr``, ``nbr``), with the degree of every
node. The ring of node ``u`` at layer ``k`` is the set of nodes at distance exactly ``k``; its
degree sequence is stored compressed, as sorted distinct degrees with their counts, for every node
and layer, at positions ``sptr[u * L + k]`` to ``sptr[u * L + k + 1]`` of ``sdeg`` and ``scnt``
(``L`` layers).
"""

import numpy as np
from numba import njit, prange


@njit(cache = True)
def _ring(ptr, nbr, deg, u, L, dist, touched, front, nxt, buf):
   """
   Breadth first search from ``u`` up to distance ``L - 1``: the degrees of the nodes of every
   ring, ring ``k`` in ``buf[starts[k]:starts[k + 1]]`` (``dist``, ``touched``, ``front``, ``nxt``
   are work arrays, ``dist`` all -1 before and after).

   :return: ``(sizes, starts)``: ring sizes, and the start of every ring in ``buf``.
   """
   sizes = np.zeros(L, np.int64)
   starts = np.zeros(L + 1, np.int64)
   nt = 0
   dist[u] = 0
   touched[nt] = u
   nt += 1
   nf = 1
   front[0] = u
   pos = 0
   for k in range(L):
      starts[k] = pos
      for i in range(nf):
         buf[pos] = deg[front[i]]
         pos += 1
      sizes[k] = nf
      if k == L - 1:
         break
      nn = 0
      for i in range(nf):
         x = front[i]
         for e in range(ptr[x], ptr[x + 1]):
            y = nbr[e]
            if dist[y] < 0:
               dist[y] = k + 1
               touched[nt] = y
               nt += 1
               nxt[nn] = y
               nn += 1
      for i in range(nn):
         front[i] = nxt[i]
      nf = nn
   starts[L] = pos
   for i in range(nt):
      dist[touched[i]] = -1
   return sizes, starts


@njit(cache = True, parallel = True)
def _ring_lengths(ptr, nbr, deg, nodes, L, out):
   """
   Length of the compressed degree sequence (number of distinct degrees) of every ring of every
   node of ``nodes``, written to ``out[i, k]``, in parallel over threads.
   """
   n = ptr.shape[0] - 1
   B = min(nodes.shape[0], 64)
   for b in prange(B):
      dist = np.full(n, -1, np.int64)
      touched, front, nxt, buf = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64)
      for i in range(b, nodes.shape[0], B):
         sizes, starts = _ring(ptr, nbr, deg, nodes[i], L, dist, touched, front, nxt, buf)
         for k in range(L):
            a, z = starts[k], starts[k + 1]
            seg = np.sort(buf[a:z])
            c = 0
            for j in range(seg.shape[0]):
               if j == 0 or seg[j] != seg[j - 1]:
                  c += 1
            out[i, k] = c


@njit(cache = True, parallel = True)
def _ring_fill(ptr, nbr, deg, nodes, L, sptr, sdeg, scnt):
   """
   Compressed degree sequences of every ring of every node of ``nodes`` (node ``nodes[i]``,
   layer ``k`` at ``sptr[i * L + k]``), in parallel over threads.
   """
   n = ptr.shape[0] - 1
   B = min(nodes.shape[0], 64)
   for b in prange(B):
      dist = np.full(n, -1, np.int64)
      touched, front, nxt, buf = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64)
      for i in range(b, nodes.shape[0], B):
         sizes, starts = _ring(ptr, nbr, deg, nodes[i], L, dist, touched, front, nxt, buf)
         for k in range(L):
            a, z = starts[k], starts[k + 1]
            seg = np.sort(buf[a:z])
            p = sptr[i * L + k] - 1
            for j in range(seg.shape[0]):
               if j == 0 or seg[j] != seg[j - 1]:
                  p += 1
                  sdeg[p] = seg[j]
                  scnt[p] = 0
               scnt[p] += 1


@njit(cache = True, inline = "always")
def _cost(a, b, ca, cb):
   """Cost of matching degrees ``a`` and ``b`` (with counts): :math:`(max + 0.5) / (min + 0.5) - 1`, times the larger count."""
   return ((max(a, b) + 0.5) / (min(a, b) + 0.5) - 1.) * max(ca, cb)


@njit(cache = True)
def _dtw(da, ca, db, cb):
   """
   Dynamic time warping distance of two compressed degree sequences (two rows of the table at a
   time).
   """
   la, lb = da.shape[0], db.shape[0]
   prev = np.full(lb + 1, np.inf)
   cur = np.full(lb + 1, np.inf)
   prev[0] = 0.
   for i in range(1, la + 1):
      cur[0] = np.inf
      for j in range(1, lb + 1):
         best = min(prev[j], cur[j - 1], prev[j - 1])
         cur[j] = _cost(da[i - 1], db[j - 1], ca[i - 1], cb[j - 1]) + best
      for j in range(lb + 1):
         prev[j] = cur[j]
   return prev[lb]


@njit(cache = True, parallel = True)
def _distances(us, vs, L, sptr, sdeg, scnt, out):
   """
   Structural distances of the pairs ``(us[p], vs[p])`` (rows of the sequence table) at every
   layer: :math:`f_k = f_{k - 1} + DTW(s(R_k(u)), s(R_k(v)))`, infinite from the first layer where
   a ring of either node is empty. Written to ``out[p, k]``, in parallel over pairs.
   """
   for p in prange(us.shape[0]):
      u, v = us[p], vs[p]
      f = 0.
      for k in range(L):
         a0, a1 = sptr[u * L + k], sptr[u * L + k + 1]
         b0, b1 = sptr[v * L + k], sptr[v * L + k + 1]
         if a1 == a0 or b1 == b0:
            f = np.inf
         if f < np.inf:
            f += _dtw(sdeg[a0:a1], scnt[a0:a1], sdeg[b0:b1], scnt[b0:b1])
         out[p, k] = f


@njit(cache = True, inline = "always")
def _next(state):
   """Uniform number in [0, 1) from a xorshift64* state (one per walk)."""
   x = state[0]
   x ^= x >> np.uint64(12)
   x ^= x << np.uint64(25)
   x ^= x >> np.uint64(27)
   state[0] = x
   return float((x * np.uint64(2685821657736338717)) >> np.uint64(11)) * (1. / 9007199254740992.)


@njit(cache = True, parallel = True)
def _walks(starts, length, L, lptr, lnbr, lcw, up, stay, seeds, out):
   """
   Random walks on the multilayer graph of struc2vec, all starting at layer 0, in parallel over
   walks. At every step, with probability ``stay`` the walk moves within its layer, to a neighbor
   drawn by edge weight (and the node is added to the walk), otherwise it changes layer: up with
   probability :math:`w_{up} / (w_{up} + 1)`, ``up[k, u]`` being :math:`w_{up}` (0 if the node
   has no edge in the layer above), down otherwise (at layer 0 it stays). A node without
   neighbors in its layer changes layer; a walk stops at layer 0 if its node has no edge there and
   none above (it then ends with -1). Layer ``k`` of node ``u`` has its edges at ``lptr[k, u]`` to ``lptr[k, u + 1]`` of
   ``lnbr``, with cumulative weights ``lcw`` restarting at every node.

   :param seeds: Seed of every walk (uint64).
   :param out: Walks, one row per start, ``length`` nodes, written in place.
   """
   for w in prange(starts.shape[0]):
      st = np.empty(1, np.uint64)
      st[0] = seeds[w] | np.uint64(1)
      u, k = starts[w], 0
      out[w, 0] = u
      i = 1
      while i < length:
         a, z = lptr[k, u], lptr[k, u + 1]
         if z > a and _next(st) < stay:   # within the layer, by edge weight
            r = _next(st) * lcw[z - 1]
            lo, hi = a, z - 1
            while lo < hi:   # first cumulative weight above r
               mid = (lo + hi) // 2
               if lcw[mid] > r:
                  hi = mid
               else:
                  lo = mid + 1
            u = lnbr[lo]
            out[w, i] = u
            i += 1
         else:   # change layer
            wu = up[k, u]
            if wu > 0 and _next(st) < wu / (wu + 1.):
               k += 1
            elif k > 0:
               k -= 1
            elif z == a:   # layer 0, no edge and no layer above: the walk stops
               break
      for j in range(i, length):
         out[w, j] = -1
