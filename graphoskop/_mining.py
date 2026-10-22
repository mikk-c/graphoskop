"""
Kernels of single graph mining (see :func:`~graphoskop.single_graph_mining`), compiled (Numba):
domain filtering and the search of an embedding of a pattern with one of its nodes mapped onto a
given node of the network, in parallel over candidate images.

The network is a CSR of oriented entries ``x -> y`` with label ``el`` (``2 * label + d``, ``d = 1``
when the edge goes from ``y`` to ``x``), sorted by ``key = y * L + el`` within each row. A pattern
with ``k`` nodes is a CSR ``(pptr, pnbr, pel)`` of its oriented entries in the same encoding.
Domains ``D`` and valid images ``V`` are ``k`` by ``n`` boolean matrices.
"""

import numpy as np
from numba import njit, prange


@njit(cache = True, inline = "always")
def _has(key, a, b, x):
   """
   Whether ``x`` is in the sorted ``key[a:b]``.
   """
   j = a + np.searchsorted(key[a:b], x)
   return j < b and key[j] == x


@njit(cache = True)
def _arc_consistency(D, pptr, pnbr, pel, ptr, nbr, el):
   """
   Removes from the domain of every pattern node the network nodes without a matching edge
   (same oriented label) towards the domain of each of its pattern neighbors, until nothing
   changes.
   """
   k, n = D.shape
   changed = True
   while changed:
      changed = False
      for a in range(k):
         for f in range(pptr[a], pptr[a + 1]):
            b, e = pnbr[f], pel[f]
            for x in range(n):
               if not D[a, x]:
                  continue
               ok = False
               for j in range(ptr[x], ptr[x + 1]):
                  if el[j] == e and D[b, nbr[j]]:
                     ok = True
                     break
               if not ok:
                  D[a, x] = False
                  changed = True


@njit(cache = True)
def _embed(v, order, par, pare, pptr, pnbr, pel, D, ptr, nbr, el, key, L, img, pos, end, fixw, fixv):
   """
   Depth first search of an embedding (injective, edges with their labels, not necessarily
   induced) with pattern node ``order[0]`` mapped onto ``v``, pattern node ``fixw`` (if not -1)
   onto ``fixv``, and every node in its domain.

   :param order: Pattern nodes, each after a neighbor (``par``, joined by oriented label ``pare``).
   :param img: Output: image of every pattern node.
   :return: Whether an embedding exists.
   """
   k = len(order)
   for i in range(k):
      img[i] = -1
   img[order[0]] = v
   if k == 1:
      return True
   d = 1
   p = img[par[1]]
   pos[1], end[1] = ptr[p], ptr[p + 1]
   while True:
      w = order[d]
      found = False
      while pos[d] < end[d]:
         e = pos[d]
         pos[d] += 1
         if el[e] != pare[d]:
            continue
         y = nbr[e]
         if not D[w, y] or (w == fixw and y != fixv):
            continue
         ok = True
         for i in range(k):
            if img[i] == y:
               ok = False
               break
         if not ok:
            continue
         for f in range(pptr[w], pptr[w + 1]):   # every edge to a mapped node, parallel ones included
            ix = img[pnbr[f]]
            if ix >= 0 and not _has(key, ptr[y], ptr[y + 1], ix * L + pel[f]):
               ok = False
               break
         if ok:
            img[w] = y
            found = True
            break
      if found:
         if d == k - 1:
            return True
         d += 1
         p = img[par[d]]
         pos[d], end[d] = ptr[p], ptr[p + 1]
      else:
         d -= 1
         if d == 0:
            return False
         img[order[d]] = -1


@njit(cache = True, parallel = True)
def _resolve(cands, order, par, pare, pptr, pnbr, pel, D, V, ptr, nbr, el, key, L, chunks):
   """
   For every candidate image ``v`` of pattern node ``order[0]``, not yet known to be valid:
   searches an embedding; if one exists, every node of it becomes a valid image of its pattern
   node, otherwise ``v`` leaves the domain. Candidates are spread over parallel chunks.
   """
   k = len(order)
   u = order[0]
   for ch in prange(chunks):
      img = np.empty(k, np.int64)
      pos = np.empty(k, np.int64)
      end = np.empty(k, np.int64)
      for i in range(ch, len(cands), chunks):
         v = cands[i]
         if V[u, v]:
            continue
         if _embed(v, order, par, pare, pptr, pnbr, pel, D, ptr, nbr, el, key, L, img, pos, end, -1, -1):
            for x in range(k):
               V[x, img[x]] = True
         else:
            D[u, v] = False


@njit(cache = True, parallel = True)
def _pair_embeds(us, vs, fixw, order, par, pare, pptr, pnbr, pel, D, ptr, nbr, el, key, L, chunks):
   """
   For every pair ``(us[i], vs[i])``, whether an embedding maps pattern node ``order[0]`` onto
   ``us[i]`` and pattern node ``fixw`` onto ``vs[i]``, pairs spread over parallel chunks.

   :return: Boolean array.
   """
   k = len(order)
   out = np.zeros(len(us), np.bool_)
   for ch in prange(chunks):
      img = np.empty(k, np.int64)
      pos = np.empty(k, np.int64)
      end = np.empty(k, np.int64)
      for i in range(ch, len(us), chunks):
         if us[i] != vs[i] and D[order[0], us[i]] and D[fixw, vs[i]]:
            out[i] = _embed(us[i], order, par, pare, pptr, pnbr, pel, D, ptr, nbr, el, key, L, img, pos, end, fixw, vs[i])
   return out
