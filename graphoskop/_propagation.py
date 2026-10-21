"""
Label propagation kernels (see :func:`~graphoskop.label_propagation_communities`): the
asynchronous sweeps and the greedy coloring of the semi-synchronous updates, compiled (Numba).
"""

import numpy as np
from numba import njit


@njit(cache = True)
def _async_sweeps(n, ptr, nbr, w, labels, max_iter, seed):
   """
   Asynchronous label propagation: in every sweep, the nodes in random order take the label of
   largest total weight among their neighbors (keeping their own if it is one of them, otherwise
   one of them at random), until a sweep changes nothing.

   :return: ``(labels, sweeps, converged)``.
   """
   np.random.seed(seed)
   labels = labels.copy()
   score = np.zeros(n)
   seen = np.zeros(n, dtype = np.bool_)
   touched = np.empty(n, dtype = np.int64)
   best = np.empty(n, dtype = np.int64)
   order = np.arange(n)
   for it in range(max_iter):
      np.random.shuffle(order)
      changed = 0
      for x in order:
         if ptr[x] == ptr[x + 1]:
            continue
         nt = 0
         for a in range(ptr[x], ptr[x + 1]):
            l = labels[nbr[a]]
            score[l] += w[a]
            if not seen[l]:
               seen[l] = True
               touched[nt] = l
               nt += 1
         top = 0.
         for t in range(nt):
            top = max(top, score[touched[t]])
         nb, keep = 0, False
         for t in range(nt):
            l = touched[t]
            if score[l] >= top * (1 - 1e-12):
               best[nb] = l
               nb += 1
               if l == labels[x]:
                  keep = True
         if not keep:
            labels[x] = best[np.random.randint(nb)]
            changed += 1
         for t in range(nt):
            score[touched[t]], seen[touched[t]] = 0., False
      if changed == 0:
         return labels, it + 1, True
   return labels, max_iter, False


@njit(cache = True)
def _greedy_coloring(n, ptr, nbr):
   """
   Greedy coloring, nodes by decreasing degree, each with the smallest color absent among its
   neighbors.

   :return: The color of every node.
   """
   color = np.full(n, -1, dtype = np.int64)
   used = np.full(n + 1, -1, dtype = np.int64)
   order = np.argsort(-(ptr[1:] - ptr[:-1]), kind = "mergesort")
   for x in order:
      for a in range(ptr[x], ptr[x + 1]):
         c = color[nbr[a]]
         if c >= 0:
            used[c] = x
      c = 0
      while used[c] == x:
         c += 1
      color[x] = c
   return color


@njit(cache = True)
def _copra(n, ptr, nbr, w, v, cptr, cls, shuffle, max_iter, seed):
   """
   Overlapping label propagation (COPRA, Gregory 2010): every node holds up to ``v`` labels with
   belonging coefficients summing to 1. A node takes the coefficients of its neighbors' labels,
   averaged with the edge weights, drops those below ``1 / v`` (keeping the largest, ties at
   random, if all are below) and normalizes the others. The nodes of a class (``cls`` between
   ``cptr[k]`` and ``cptr[k + 1]``) update together; with ``shuffle`` (classes of one node), in a
   new random order at every sweep. Stops when a sweep changes no set of labels or, as in COPRA,
   when the labels in use are those of the previous sweep and no label has fewer nodes than its
   fewest since they became so (labels can keep oscillating on a few nodes).

   :return: ``(labels, coefficients, counts, sweeps, converged)``, labels and coefficients as
      ``(n, v)`` arrays with ``counts`` valid entries per node.
   """
   np.random.seed(seed)
   lab = np.full((n, v), -1, dtype = np.int64)
   coef = np.zeros((n, v))
   cnt = np.ones(n, dtype = np.int64)
   lab[:, 0] = np.arange(n)
   coef[:, 0] = 1.
   nlab, ncoef, ncnt = lab.copy(), coef.copy(), cnt.copy()
   score = np.zeros(n)
   seen = np.zeros(n, dtype = np.bool_)
   touched = np.empty(n, dtype = np.int64)
   strength = np.zeros(n)
   for x in range(n):
      for a in range(ptr[x], ptr[x + 1]):
         strength[x] += w[a]
   ncls = len(cptr) - 1
   count, low = np.zeros(n, dtype = np.int64), np.zeros(n, dtype = np.int64)
   for it in range(max_iter):
      if shuffle:
         np.random.shuffle(cls)
      changed = False
      for k in range(ncls):
         for q in range(cptr[k], cptr[k + 1]):
            x = cls[q]
            if strength[x] == 0:
               continue
            nt = 0
            for a in range(ptr[x], ptr[x + 1]):
               y = nbr[a]
               for b in range(cnt[y]):
                  l = lab[y, b]
                  score[l] += w[a] * coef[y, b] / strength[x]
                  if not seen[l]:
                     seen[l] = True
                     touched[nt] = l
                     nt += 1
            m, top, total = 0, 0., 0.
            for t in range(nt):
               top = max(top, score[touched[t]])
            for t in range(nt):
               l = touched[t]
               if score[l] >= 1. / v - 1e-12 and m < v:
                  nlab[x, m], ncoef[x, m] = l, score[l]
                  total += score[l]
                  m += 1
            if m == 0:   # all below the threshold: the largest, ties at random
               nb = 0
               for t in range(nt):
                  if score[touched[t]] >= top * (1 - 1e-12):
                     nb += 1
               pick = np.random.randint(nb)
               for t in range(nt):
                  if score[touched[t]] >= top * (1 - 1e-12):
                     if pick == 0:
                        nlab[x, 0], ncoef[x, 0] = touched[t], 1.
                     pick -= 1
               m, total = 1, 1.
            order = np.argsort(nlab[x, :m])
            nlab[x, :m], ncoef[x, :m] = nlab[x, :m][order], ncoef[x, :m][order] / total
            ncnt[x] = m
            for t in range(nt):
               score[touched[t]], seen[touched[t]] = 0., False
         for q in range(cptr[k], cptr[k + 1]):
            x = cls[q]
            if ncnt[x] != cnt[x]:
               changed = True
            else:
               for b in range(cnt[x]):
                  if nlab[x, b] != lab[x, b]:
                     changed = True
            lab[x], coef[x], cnt[x] = nlab[x], ncoef[x], ncnt[x]
      if not changed:
         return lab, coef, cnt, it + 1, True
      count[:] = 0
      for x in range(n):
         for b in range(cnt[x]):
            count[lab[x, b]] += 1
      same, lower = True, False
      for l in range(n):
         if (count[l] > 0) != (low[l] > 0):
            same = False
         elif count[l] < low[l]:
            lower = True
      if same and not lower and it > 0:
         return lab, coef, cnt, it + 1, True
      for l in range(n):
         low[l] = count[l] if not same else min(low[l], count[l])
   return lab, coef, cnt, max_iter, False
