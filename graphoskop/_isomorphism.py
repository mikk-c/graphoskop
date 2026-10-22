"""
Search kernels of the isomorphism tests (see :func:`~graphoskop.is_isomorphic`), compiled
(Numba): the matching order of the nodes of the pattern and a depth first search of the mappings
of the pattern into the target, in the style of VF2++ / VF3, run in parallel over the candidate
images of the first node of the order; partition refinement and individualization-refinement for
isomorphism.

Graphs are given as sorted CSR arrays ``(ptr, nbr, lab)`` of their pair graph: one entry per
ordered pair of distinct adjacent nodes, ``lab = forward * K + backward``, where ``forward`` and
``backward`` are the codes of the arcs ``u -> v`` and ``v -> u`` (0 if absent; undirected edges
only have ``forward``). ``mode`` is 0 for isomorphism, 1 for induced subgraph isomorphism and 2
for monomorphism (subgraphs, not necessarily induced).
"""

import numpy as np
from numba import njit, prange


@njit(cache = True, inline = "always")
def _before(a1, a2, a3, b1, b2, b3):
   return a1 < b1 or (a1 == b1 and (a2 < b2 or (a2 == b2 and a3 < b3)))


@njit(cache = True)
def _order(n, ptr, nbr, cand, deg):
   """
   Matching order of the pattern: repeatedly the unordered node with most neighbors already in
   the order, then fewest candidates, then highest degree (a lazy binary heap); a new connected
   component starts from its node with fewest candidates.

   :param n: Number of nodes of the pattern.
   :param ptr: CSR pointers of the pattern.
   :param nbr: CSR neighbors of the pattern.
   :param cand: Number of candidate images of every node.
   :param deg: Degree of every node.
   :return: The nodes, in matching order.
   """
   order = np.empty(n, dtype = np.int64)
   done = np.zeros(n, dtype = np.bool_)
   conn = np.zeros(n, dtype = np.int64)
   roots = np.argsort(cand * (n + 1) + (n - deg), kind = "mergesort")
   size = n + len(nbr) + 1
   h1, h2, h3 = np.empty(size, np.int64), np.empty(size, np.int64), np.empty(size, np.int64)
   hv = np.empty(size, np.int64)
   hn, ri, k = 0, 0, 0
   while k < n:
      if hn == 0:
         while done[roots[ri]]:
            ri += 1
         v = roots[ri]
         h1[0], h2[0], h3[0], hv[0] = 0, cand[v], -deg[v], v
         hn = 1
      a1, a2, a3, v = h1[0], h2[0], h3[0], hv[0]   # pop the minimum
      hn -= 1
      h1[0], h2[0], h3[0], hv[0] = h1[hn], h2[hn], h3[hn], hv[hn]
      i = 0
      while True:
         l, r, m = 2 * i + 1, 2 * i + 2, i
         if l < hn and _before(h1[l], h2[l], h3[l], h1[m], h2[m], h3[m]):
            m = l
         if r < hn and _before(h1[r], h2[r], h3[r], h1[m], h2[m], h3[m]):
            m = r
         if m == i:
            break
         h1[i], h1[m] = h1[m], h1[i]
         h2[i], h2[m] = h2[m], h2[i]
         h3[i], h3[m] = h3[m], h3[i]
         hv[i], hv[m] = hv[m], hv[i]
         i = m
      if done[v] or -a1 != conn[v]:   # stale entry
         continue
      order[k] = v
      done[v] = True
      k += 1
      for e in range(ptr[v], ptr[v + 1]):
         w = nbr[e]
         if done[w]:
            continue
         conn[w] += 1
         i = hn   # push
         h1[i], h2[i], h3[i], hv[i] = -conn[w], cand[w], -deg[w], w
         hn += 1
         while i > 0:
            p = (i - 1) // 2
            if not _before(h1[i], h2[i], h3[i], h1[p], h2[p], h3[p]):
               break
            h1[i], h1[p] = h1[p], h1[i]
            h2[i], h2[p] = h2[p], h2[i]
            h3[i], h3[p] = h3[p], h3[i]
            hv[i], hv[p] = hv[p], hv[i]
            i = p
   return order


@njit(cache = True, inline = "always")
def _find(nbr, a, b, x):
   """
   Position of ``x`` in the sorted ``nbr[a:b]``, or -1.
   """
   j = a + np.searchsorted(nbr[a:b], x)
   return j if j < b and nbr[j] == x else -1


@njit(cache = True)
def _feasible(u, v, d, mode, K, order, map2, inv1, ptr1, nbr1, lab1, ptr2, nbr2, lab2, key1, key2, loop1, loop2, deg1, deg2, F):
   """
   Whether pattern node ``u`` can be mapped onto target node ``v``, given the mapping of the first
   ``d`` nodes of the order: same key, a free ``v``, the pairs of ``u`` with mapped nodes mapped
   onto pairs of ``v`` with the same labels (contained labels in monomorphisms), and, except in
   monomorphisms, no further pair of ``v`` with a mapped node.
   """
   if inv1[v] >= 0 or key1[v] != key2[u]:
      return False
   if mode != 0:
      if deg1[v] < deg2[u]:
         return False
      if mode == 1 and loop1[v] != loop2[u]:
         return False
      if mode == 2 and loop2[u] != 0 and loop1[v] != loop2[u]:
         return False
      if F.shape[0] > 0 and (F[u, v >> 6] >> np.uint64(v & 63)) & np.uint64(1) == 0:
         return False
   a, b = ptr1[v], ptr1[v + 1]
   c = 0
   for e in range(ptr2[u], ptr2[u + 1]):
      x = map2[nbr2[e]]
      if x < 0:
         continue
      j = _find(nbr1, a, b, x)
      if j < 0:
         return False
      p, t = lab2[e], lab1[j]
      if mode == 2:
         pf, pb = p // K, p % K
         if (pf != 0 and pf != t // K) or (pb != 0 and pb != t % K):
            return False
      elif p != t:
         return False
      c += 1
   if mode != 2:   # no pair of v with a mapped node beyond those of u
      if b - a <= 4 * d:
         m = 0
         for j in range(a, b):
            if inv1[nbr1[j]] >= 0:
               m += 1
      else:
         m = 0
         for i in range(d):
            if _find(nbr1, a, b, map2[order[i]]) >= 0:
               m += 1
      if m != c:
         return False
   return True


@njit(cache = True)
def _dfs(r, mode, K, order, ptr1, nbr1, lab1, ptr2, nbr2, lab2, key1, key2, loop1, loop2, deg1, deg2, F, kptr, kmem, map2, inv1, pos, hi, src, out, start, limit, stop):
   """
   All mappings of the pattern with the first node of the order mapped onto target node ``r``.

   :param map2: Image of every pattern node, -1 (work array, left as found).
   :param inv1: Preimage of every target node, -1 (work array, left as found).
   :param pos: Work array: next candidate position at every depth.
   :param hi: Work array: end of the candidates at every depth.
   :param src: Work array: target node whose neighbors are the candidates, or -1 for the target
      nodes with the key of the pattern node (``kmem[kptr[key]:kptr[key + 1]]``).
   :param out: Mappings found are written from row ``start`` (no row: only counted).
   :param limit: Stop after this many mappings (-1: no limit).
   :param stop: ``stop[0] = 1`` stops the search (checked at every step).
   :return: Number of mappings found.
   """
   n2 = len(order)
   u = order[0]
   if not _feasible(u, r, 0, mode, K, order, map2, inv1, ptr1, nbr1, lab1, ptr2, nbr2, lab2, key1, key2, loop1, loop2, deg1, deg2, F):
      return 0
   map2[u] = r
   inv1[r] = u
   count = 0
   d = 1
   if n2 == 1:
      if out.shape[0] > 0:
         out[start, :] = map2
      count = 1
      d = 0
   else:
      _enter(1, order, map2, ptr1, ptr2, nbr2, key2, deg1, kptr, pos, hi, src)
   while d >= 1:
      if stop[0] != 0:
         break
      u = order[d]
      found = -1
      while pos[d] < hi[d]:
         v = nbr1[pos[d]] if src[d] >= 0 else kmem[pos[d]]
         pos[d] += 1
         if _feasible(u, v, d, mode, K, order, map2, inv1, ptr1, nbr1, lab1, ptr2, nbr2, lab2, key1, key2, loop1, loop2, deg1, deg2, F):
            found = v
            break
      if found < 0:   # backtrack
         d -= 1
         if d >= 1:
            w = order[d]
            inv1[map2[w]] = -1
            map2[w] = -1
         continue
      map2[u] = found
      inv1[found] = u
      if d == n2 - 1:
         if out.shape[0] > 0:
            out[start + count, :] = map2
         count += 1
         inv1[found] = -1
         map2[u] = -1
         if count == limit:
            break
         continue
      d += 1
      _enter(d, order, map2, ptr1, ptr2, nbr2, key2, deg1, kptr, pos, hi, src)
   for i in range(d + 1):   # undo what is left of the mapping
      w = order[i]
      if map2[w] >= 0:
         inv1[map2[w]] = -1
         map2[w] = -1
   map2[order[0]] = -1
   inv1[r] = -1
   return count


@njit(cache = True)
def _enter(d, order, map2, ptr1, ptr2, nbr2, key2, deg1, kptr, pos, hi, src):
   """
   Candidates of the node at depth ``d``: the neighbors of the image of lowest degree among its
   mapped neighbors, or all the target nodes with its key if none is mapped.
   """
   u = order[d]
   best = -1
   for e in range(ptr2[u], ptr2[u + 1]):
      x = map2[nbr2[e]]
      if x >= 0 and (best < 0 or deg1[x] < deg1[best]):
         best = x
   src[d] = best
   if best >= 0:
      pos[d], hi[d] = ptr1[best], ptr1[best + 1]
   else:
      pos[d], hi[d] = kptr[key2[u]], kptr[key2[u] + 1]


@njit(cache = True, parallel = True)
def _search(roots, chunks, first, mode, K, order, ptr1, nbr1, lab1, ptr2, nbr2, lab2, key1, key2, loop1, loop2, deg1, deg2, F, kptr, kmem, offsets, out):
   """
   Search from every root (candidate image of the first node of the order), roots interleaved
   over ``chunks`` parallel chunks.

   :param first: Stop everything at the first mapping found.
   :param offsets: Row of ``out`` where the mappings of each root start (when ``out`` has rows).
   :param out: Output rows, or an empty array to count only.
   :return: Number of mappings from every root.
   """
   n1, n2 = len(ptr1) - 1, len(order)
   cnt = np.zeros(len(roots), dtype = np.int64)
   stop = np.zeros(1, dtype = np.int64)
   for ch in prange(chunks):
      map2 = np.full(n2, -1, dtype = np.int64)
      inv1 = np.full(n1, -1, dtype = np.int32)
      pos, hi, src = np.zeros(n2, np.int64), np.zeros(n2, np.int64), np.zeros(n2, np.int64)
      for i in range(ch, len(roots), chunks):
         if stop[0] != 0:
            break
         start = offsets[i] if out.shape[0] > 0 else 0
         cnt[i] = _dfs(roots[i], mode, K, order, ptr1, nbr1, lab1, ptr2, nbr2, lab2, key1, key2, loop1, loop2, deg1, deg2, F, kptr, kmem, map2, inv1, pos, hi, src, out, start, 1 if first else -1, stop)
         if first and cnt[i] > 0:
            stop[0] = 1
   return cnt


@njit(cache = True)
def _search_limited(roots, limit, mode, K, order, ptr1, nbr1, lab1, ptr2, nbr2, lab2, key1, key2, loop1, loop2, deg1, deg2, F, kptr, kmem, out):
   """
   Serial search of the first ``limit`` mappings, written to ``out``.

   :return: Number of mappings found.
   """
   n1, n2 = len(ptr1) - 1, len(order)
   map2 = np.full(n2, -1, dtype = np.int64)
   inv1 = np.full(n1, -1, dtype = np.int32)
   pos, hi, src = np.zeros(n2, np.int64), np.zeros(n2, np.int64), np.zeros(n2, np.int64)
   stop = np.zeros(1, dtype = np.int64)
   total = 0
   for r in roots:
      if total == limit:
         break
      total += _dfs(r, mode, K, order, ptr1, nbr1, lab1, ptr2, nbr2, lab2, key1, key2, loop1, loop2, deg1, deg2, F, kptr, kmem, map2, inv1, pos, hi, src, out, total, limit - total, stop)
   return total


# Partition refinement and individualization-refinement (isomorphism) -----------------------------
#
# The union of the two graphs (nodes 0 to n1 - 1, then n1 to N - 1) is partitioned into cells,
# contiguous ranges of ``elem``: cell c holds ``elem[cstart[c]:cstart[c] + csize[c]]``, and
# ``cell[v]``, ``pos[v]`` locate node v. ``st[0]`` is the number of cells, ``st[1]`` the length of
# the queue of splitter cells ``qbuf`` (``inq`` marks queued cells). A split keeps the old id for
# its last part and gives new ids, with ``parent`` the old id, to the parts before it, so undoing
# the splits in reverse order of creation merges adjacent ranges. Cells are balanced when they
# hold as many nodes of each graph: isomorphic graphs keep every cell balanced.


@njit(cache = True)
def _new_partition(n1, N, cls):
   """
   Partition from node classes, every cell queued; ``c1`` counts the nodes of the first graph in
   every cell.

   :return: ``(elem, pos, cell, cstart, csize, c1, parent, st, qbuf, inq)``.
   """
   elem = np.argsort(cls, kind = "mergesort").astype(np.int64)
   pos = np.empty(N, np.int64)
   cell = np.empty(N, np.int64)
   cstart = np.zeros(N + 1, np.int64)
   csize = np.zeros(N + 1, np.int64)
   c1 = np.zeros(N + 1, np.int64)
   parent = np.full(N + 1, -1, np.int64)
   qbuf = np.empty(N + 1, np.int64)
   inq = np.zeros(N + 1, np.bool_)
   nc = 0
   for i in range(N):
      v = elem[i]
      pos[v] = i
      if i == 0 or cls[v] != cls[elem[i - 1]]:
         cstart[nc] = i
         qbuf[nc] = nc
         inq[nc] = True
         nc += 1
      cell[v] = nc - 1
      csize[nc - 1] += 1
      if v < n1:
         c1[nc - 1] += 1
   st = np.array([nc, nc], np.int64)
   return elem, pos, cell, cstart, csize, c1, parent, st, qbuf, inq


@njit(cache = True)
def _balanced(csize, c1, st):
   """
   Whether every cell holds as many nodes of each graph.
   """
   for c in range(st[0]):
      if 2 * c1[c] != csize[c]:
         return False
   return True


@njit(cache = True)
def _refine_cells(n1, ptr, nbr, harc, elem, pos, cell, cstart, csize, c1, parent, st, qbuf, inq, key, touched, tl, ccount, cfill, tcells):
   """
   Refines the partition until no queued splitter splits a cell: the nodes of a cell are split by
   the sum of the hashes of the labels of their arcs into the splitter (the multiset of labels,
   with counts), and new cells are queued, all but the largest part of a cell not in the queue
   (Hopcroft's rule), for :math:`O(m \\log n)` work in all.

   Work arrays (``N + 1`` long): ``key``, ``touched`` (all false), ``tl``, ``ccount`` (all 0, and
   left so), ``cfill``, ``tcells``.

   :return: ``False`` (queue emptied) as soon as a part is unbalanced.
   """
   while st[1] > 0:
      st[1] -= 1
      s = qbuf[st[1]]
      inq[s] = False
      nt = 0
      for i in range(cstart[s], cstart[s] + csize[s]):
         w = elem[i]
         for e in range(ptr[w], ptr[w + 1]):
            v = nbr[e]
            if not touched[v]:
               touched[v] = True
               tl[nt] = v
               nt += 1
               key[v] = 0
            key[v] += harc[e]
      # touched nodes grouped by cell (counting sort over the touched cells)
      ntc = 0
      for i in range(nt):
         v = tl[i]
         touched[v] = False
         c = cell[v]
         if ccount[c] == 0:
            tcells[ntc] = c
            ntc += 1
         ccount[c] += 1
      o = 0
      for i in range(ntc):
         c = tcells[i]
         cfill[c] = o
         o += ccount[c]
      tc = np.empty(nt, np.int64)
      for i in range(nt):
         v = tl[i]
         c = cell[v]
         tc[cfill[c]] = v
         cfill[c] += 1
      a = 0
      for i in range(ntc):
         c = tcells[i]
         cnt = ccount[c]
         ccount[c] = 0
         b = a + cnt
         g = tc[a:b]
         start, size = cstart[c], csize[c]
         same = True
         for k in range(1, cnt):
            if key[g[k]] != key[g[0]]:
               same = False
               break
         if cnt == size and same:
            a = b
            continue
         if not same:
            g = g[np.argsort(key[g], kind = "mergesort")]
         for k in range(cnt):   # touched nodes to the front, in key order
            v = g[k]
            j, p = start + k, pos[v]
            w = elem[j]
            elem[j], pos[v] = v, j
            elem[p], pos[w] = w, p
         # parts: groups of equal key, then the untouched rest; the last part keeps id c
         nparts = 0
         k = 0
         while k < cnt:
            l = k
            while l < cnt and key[g[l]] == key[g[k]]:
               l += 1
            tl[nparts] = l   # part ends (tl is free now)
            nparts += 1
            k = l
         if cnt < size:
            tl[nparts] = size
            nparts += 1
         big, bigsize, lo, rest = 0, -1, 0, c1[c]
         for q in range(nparts):
            hi = tl[q]
            if hi - lo > bigsize:
               big, bigsize = q, hi - lo
            if lo >= cnt:   # the untouched rest: by difference, in O(1)
               m = rest
            else:
               m = 0
               for z in range(start + lo, start + hi):
                  if elem[z] < n1:
                     m += 1
               rest -= m
            cfill[q] = m   # nodes of the first graph in the part (cfill is free now)
            if 2 * m != hi - lo:
               for z in range(st[1]):
                  inq[qbuf[z]] = False
               st[1] = 0
               for z in range(i + 1, ntc):
                  ccount[tcells[z]] = 0
               return False
            lo = hi
         was = inq[c]
         lo = 0
         for q in range(nparts):
            hi = tl[q]
            if q < nparts - 1:
               d = st[0]
               st[0] += 1
               cstart[d], csize[d], c1[d], parent[d] = start + lo, hi - lo, cfill[q], c
               for z in range(start + lo, start + hi):
                  cell[elem[z]] = d
            else:
               d = c
               cstart[c], csize[c], c1[c] = start + lo, hi - lo, cfill[q]
            if (was or q != big) and not inq[d]:
               qbuf[st[1]] = d
               st[1] += 1
               inq[d] = True
            lo = hi
         a = b
   return True


@njit(cache = True)
def _undo(nc0, elem, cell, cstart, csize, c1, parent, st):
   """
   Undoes the splits back to ``nc0`` cells.
   """
   while st[0] > nc0:
      st[0] -= 1
      d = st[0]
      p = parent[d]
      for i in range(cstart[d], cstart[d] + csize[d]):
         cell[elem[i]] = p
      cstart[p] = cstart[d]
      csize[p] += csize[d]
      c1[p] += c1[d]


@njit(cache = True)
def _individualize(x, y, elem, pos, cell, cstart, csize, c1, parent, st, qbuf, inq, queue):
   """
   Moves ``x`` and ``y`` (same cell) into a new cell of their own, queued if ``queue``.
   """
   c = cell[x]
   start = cstart[c]
   for k in range(2):
      v = x if k == 0 else y
      j, p = start + k, pos[v]
      w = elem[j]
      elem[j], pos[v] = v, j
      elem[p], pos[w] = w, p
   d = st[0]
   st[0] += 1
   cstart[d], csize[d], c1[d], parent[d] = start, 2, 1, c
   cell[x] = cell[y] = d
   cstart[c] += 2
   csize[c] -= 2
   c1[c] -= 1
   if queue:
      qbuf[st[1]] = d
      st[1] += 1
      inq[d] = True


@njit(cache = True)
def _verify(n1, ptr, nbr, lab, elem, cstart, st, phi):
   """
   With every cell a pair (one node of each graph), whether the map it gives is an isomorphism:
   every arc of the second graph goes to an arc of the first with the same label (the graphs have
   as many arcs, and node keys are in the initial cells).
   """
   for c in range(st[0]):
      a, b = elem[cstart[c]], elem[cstart[c] + 1]
      if a >= n1:
         a, b = b, a
      phi[b - n1] = a
   N = len(ptr) - 1
   for u in range(n1, N):
      x = phi[u - n1]
      lo, hi = ptr[x], ptr[x + 1]
      for e in range(ptr[u], ptr[u + 1]):
         j = _find(nbr, lo, hi, phi[nbr[e] - n1])
         if j < 0 or lab[j] != lab[e]:
            return False
   return True


@njit(cache = True)
def _ir(n1, ptr, nbr, lab, harc, elem, pos, cell, cstart, csize, c1, parent, st, qbuf, inq, x0, cands0, stop):
   """
   Individualization-refinement search of an isomorphism, from a refined balanced partition: a
   node of the second graph in the first cell with more than two nodes is individualized with
   each node of the first graph in that cell, in turn, and the partition refined, until every
   cell is a pair, whose map is checked. When individualizing a pair splits nothing else, the
   cell is a set of twins (same neighbors, with the same labels) in both graphs: one candidate is
   enough, and the rest of the cell is paired arbitrarily.

   :param x0: Node to individualize first, with the candidates ``cands0`` (if ``x0 >= 0``).
   :param stop: ``stop[0] = 1`` stops the search.
   :return: 1 if an isomorphism is found (the partition is left as found), 2 if not and the first
      cell is a set of twins, 0 if not.
   """
   N = len(ptr) - 1
   key = np.zeros(N, np.int64)
   touched = np.zeros(N, np.bool_)
   tl = np.empty(N + 1, np.int64)
   ccount = np.zeros(N + 1, np.int64)
   cfill, tcells = np.empty(N + 1, np.int64), np.empty(N + 1, np.int64)
   phi = np.empty(N - n1, np.int64)
   cbuf = np.empty(max(16, N), np.int64)
   L = max(16, N)
   lx, lnc, lpos, lend, ltwin = np.empty(L, np.int64), np.empty(L, np.int64), np.empty(L, np.int64), np.empty(L, np.int64), np.zeros(L, np.bool_)
   ltried = np.zeros(L, np.bool_)
   lscan = np.zeros(L, np.int64)
   top, used = 0, 0
   choose = True
   twin0 = False
   while True:
      if stop[0] != 0:
         return 0
      if choose:
         choose = False
         if top == 0 and x0 >= 0:
            c, x = cell[x0], x0
         else:   # first cell with more than a pair: cells before the parent's choice are pairs
            c = -1
            for q in range(lscan[top - 1] if top > 0 else 0, st[0]):
               if csize[q] > 2:
                  c = q
                  break
            x = -1
            if c >= 0:
               for i in range(cstart[c], cstart[c] + csize[c]):
                  if elem[i] >= n1:
                     x = elem[i]
                     break
         if c < 0:
            if _verify(n1, ptr, nbr, lab, elem, cstart, st, phi):
               return 1
         else:
            if top == 0 and x0 >= 0:
               k = len(cands0)
            else:
               k = csize[c] // 2
            if used + k > len(cbuf):
               nb = np.empty(2 * (used + k), np.int64)
               nb[:used] = cbuf[:used]
               cbuf = nb
            if top == 0 and x0 >= 0:
               cbuf[used:used + k] = cands0
            else:
               j = used
               for i in range(cstart[c], cstart[c] + csize[c]):
                  if elem[i] < n1:
                     cbuf[j] = elem[i]
                     j += 1
            lx[top], lnc[top], lpos[top], lend[top], lscan[top] = x, st[0], used, used + k, c if not (top == 0 and x0 >= 0) else 0
            ltwin[top], ltried[top] = False, False
            used += k
            top += 1
      if top == 0:
         return 2 if twin0 else 0
      t = top - 1
      _undo(lnc[t], elem, cell, cstart, csize, c1, parent, st)
      if (ltwin[t] and ltried[t]) or lpos[t] == lend[t]:
         if t == 0:
            twin0 = ltwin[0]
         top -= 1
         used = lend[t - 1] if t > 0 else 0   # levels hold consecutive ranges of cbuf
         continue
      y = cbuf[lpos[t]]
      lpos[t] += 1
      first = not ltried[t]
      ltried[t] = True
      x = lx[t]
      _individualize(x, y, elem, pos, cell, cstart, csize, c1, parent, st, qbuf, inq, True)
      if not _refine_cells(n1, ptr, nbr, harc, elem, pos, cell, cstart, csize, c1, parent, st, qbuf, inq, key, touched, tl, ccount, cfill, tcells):
         continue
      if first and st[0] == lnc[t] + 1:   # nothing split: twins; pair the rest of the cell
         ltwin[t] = True
         c = parent[st[0] - 1]   # the rest of the cell of x
         a, b = 0, 0
         for i in range(cstart[c], cstart[c] + csize[c]):
            v = elem[i]
            if v < n1:
               tl[a] = v
               a += 1
            else:
               key[b] = v
               b += 1
         for i in range(a - 1):
            _individualize(key[i], tl[i], elem, pos, cell, cstart, csize, c1, parent, st, qbuf, inq, False)
         key[:b] = 0
      choose = True


@njit(cache = True, parallel = True)
def _ir_parallel(n1, ptr, nbr, lab, harc, elem, pos, cell, cstart, csize, c1, parent, st, x0, cands, chunks):
   """
   :func:`_ir` from the same partition, with the candidates of ``x0`` spread over parallel
   chunks, each on its own copy of the partition.

   :return: Whether an isomorphism was found.
   """
   stop = np.zeros(1, np.int64)
   N = len(ptr) - 1
   for ch in prange(chunks):
      mine = cands[ch::chunks].copy()
      if len(mine) == 0:
         continue
      e2, p2, c2, s2, z2, o2, pa2, st2 = elem.copy(), pos.copy(), cell.copy(), cstart.copy(), csize.copy(), c1.copy(), parent.copy(), st.copy()
      q2, i2 = np.empty(N + 1, np.int64), np.zeros(N + 1, np.bool_)
      if _ir(n1, ptr, nbr, lab, harc, e2, p2, c2, s2, z2, o2, pa2, st2, q2, i2, x0, mine, stop) == 1:
         stop[0] = 1
   return stop[0] == 1


@njit(cache = True)
def _iso_enum(n1, ptr, nbr, lab, harc, elem, pos, cell, cstart, csize, c1, parent, st, qbuf, inq, x0, cands, order, ptr1, nbr1, lab1, ptr2, nbr2, lab2, loop1, loop2, deg1, deg2, out, limit):
   """
   All isomorphisms (up to ``limit``, -1 for all), written to ``out`` (no row: only counted): for
   every candidate image ``y`` of ``x0`` (the first node of ``order``), ``x0`` and ``y`` are
   individualized, the partition refined, and the depth first search of :func:`_dfs` runs with
   the cells as keys.

   :return: Number of isomorphisms.
   """
   N = len(ptr) - 1
   n2 = N - n1
   key = np.zeros(N, np.int64)
   touched = np.zeros(N, np.bool_)
   tl = np.empty(N + 1, np.int64)
   ccount = np.zeros(N + 1, np.int64)
   cfill, tcells = np.empty(N + 1, np.int64), np.empty(N + 1, np.int64)
   map2 = np.full(n2, -1, np.int64)
   inv1 = np.full(n1, -1, np.int32)
   p, h, sr = np.zeros(n2, np.int64), np.zeros(n2, np.int64), np.zeros(n2, np.int64)
   F = np.zeros((0, 0), np.uint64)
   stop = np.zeros(1, np.int64)
   nc0 = st[0]
   total = 0
   for y in cands:
      if total == limit:
         break
      _individualize(x0, y, elem, pos, cell, cstart, csize, c1, parent, st, qbuf, inq, True)
      if _refine_cells(n1, ptr, nbr, harc, elem, pos, cell, cstart, csize, c1, parent, st, qbuf, inq, key, touched, tl, ccount, cfill, tcells):
         key1, key2 = cell[:n1].copy(), cell[n1:].copy()
         kptr = np.zeros(st[0] + 1, np.int64)
         for v in range(n1):
            kptr[key1[v] + 1] += 1
         for c in range(st[0]):
            kptr[c + 1] += kptr[c]
         kmem = np.empty(n1, np.int64)
         fill = kptr[:-1].copy()
         for v in range(n1):
            kmem[fill[key1[v]]] = v
            fill[key1[v]] += 1
         total += _dfs(y, 0, 1, order, ptr1, nbr1, lab1, ptr2, nbr2, lab2, key1, key2, loop1, loop2, deg1, deg2, F, kptr, kmem, map2, inv1, p, h, sr, out, total, limit - total if limit >= 0 else -1, stop)
      _undo(nc0, elem, cell, cstart, csize, c1, parent, st)
   return total
