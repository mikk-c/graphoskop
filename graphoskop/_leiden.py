"""
Leiden kernels (see :func:`~graphoskop.leiden_communities`): the fast local moving of nodes, the
refinement of communities and the aggregation of the network, compiled (Numba).

The quality is :math:`\\sum_c [E_c - \\gamma \\sum_l \\mu_l A_{cl} B_{cl}]`, with :math:`E_c` the
weight inside community :math:`c`, :math:`A_{cl}` and :math:`B_{cl}` the sums of the node weights
``a`` and ``b`` of its nodes in slot :math:`l` (a layer, or the whole network) and :math:`\\mu_l`
the slot scale ``inv_m``: modularity (degrees) and the constant Potts model (node counts) are
both of this form. The gain of moving node :math:`i` to community :math:`c` is
:math:`w_{ic} - \\gamma \\sum_l \\mu_l (a_{il} B_{cl} + b_{il} A_{cl})`, with :math:`w_{ic}` the
weight of the edges between them in both directions (``w`` is symmetric, without self loops).
"""

import numpy as np
from numba import njit


@njit(cache = True)
def _null(x, y, X, Y, inv_m):
   """
   Null model term between node (or cluster) weights ``x``, ``y`` and set weights ``X``, ``Y``.
   """
   r = 0.
   for l in range(len(inv_m)):
      r += inv_m[l] * (x[l] * Y[l] + y[l] * X[l])
   return r


@njit(cache = True)
def _local_moving(n, ptr, nbr, w, a, b, inv_m, gamma, comm, seed):
   """
   Fast local moving: nodes in a queue, initially all in random order, each moved to the
   community of largest gain (an empty one if no gain is positive); the neighbors of a moved node
   outside its new community join the queue.

   :return: ``(comm, moves)``, the communities (labels below ``n``) and the number of moves.
   """
   np.random.seed(seed)
   comm = comm.copy()
   L = a.shape[1]
   A, B = np.zeros((n, L)), np.zeros((n, L))
   size = np.zeros(n, dtype = np.int64)
   for i in range(n):
      A[comm[i]] += a[i]
      B[comm[i]] += b[i]
      size[comm[i]] += 1
   free = np.empty(n, dtype = np.int64)
   nf = 0
   for c in range(n):
      if size[c] == 0:
         free[nf] = c
         nf += 1
   queue = np.random.permutation(n)
   inq = np.ones(n, dtype = np.bool_)
   head, count, moves = 0, n, 0
   score = np.zeros(n)
   seen = np.zeros(n, dtype = np.bool_)
   touched = np.empty(n, dtype = np.int64)
   while count > 0:
      i = queue[head]
      head = (head + 1) % n
      count -= 1
      inq[i] = False
      c0 = comm[i]
      A[c0] -= a[i]
      B[c0] -= b[i]
      size[c0] -= 1
      touched[0], seen[c0] = c0, True
      nt = 1
      for e in range(ptr[i], ptr[i + 1]):
         c = comm[nbr[e]]
         score[c] += w[e]
         if not seen[c]:
            seen[c] = True
            touched[nt] = c
            nt += 1
      best = c0
      top = score[c0] - gamma * _null(a[i], b[i], A[c0], B[c0], inv_m)
      for k in range(1, nt):
         c = touched[k]
         g = score[c] - gamma * _null(a[i], b[i], A[c], B[c], inv_m)
         if g > top + 1e-12 * (abs(g) + abs(top)):
            best, top = c, g
      if top <= 0 and size[c0] > 0:   # an empty community (no gain staying: alone)
         best = free[nf - 1]
         nf -= 1
      for k in range(nt):
         score[touched[k]], seen[touched[k]] = 0., False
      A[best] += a[i]
      B[best] += b[i]
      size[best] += 1
      comm[i] = best
      if best != c0:
         moves += 1
         if size[c0] == 0:
            free[nf] = c0
            nf += 1
         for e in range(ptr[i], ptr[i + 1]):
            j = nbr[e]
            if not inq[j] and comm[j] != best:
               inq[j] = True
               queue[(head + count) % n] = j
               count += 1
   return comm, moves


@njit(cache = True)
def _refine(n, ptr, nbr, w, a, b, inv_m, gamma, comm, theta, scale, seed):
   """
   Refinement: within every community, starting from singletons, well-connected nodes still alone
   join (in random order) a well-connected cluster of their community, chosen at random with
   probability proportional to :math:`\\exp(\\Delta H / \\theta)` among those with
   :math:`\\Delta H \\geq 0` (staying alone included); ``scale`` converts gains to quality units.

   :return: The refined cluster of every node (labels below ``n``).
   """
   np.random.seed(seed)
   L = a.shape[1]
   CA, CB = np.zeros((n, L)), np.zeros((n, L))
   for i in range(n):
      CA[comm[i]] += a[i]
      CB[comm[i]] += b[i]
   ref = np.arange(n)
   size = np.ones(n, dtype = np.int64)
   RA, RB = a.copy(), b.copy()
   ext = np.zeros(n)   # weight between a cluster and the rest of its community
   for i in range(n):
      for e in range(ptr[i], ptr[i + 1]):
         if comm[nbr[e]] == comm[i]:
            ext[i] += w[e]
   score = np.zeros(n)
   seen = np.zeros(n, dtype = np.bool_)
   touched = np.empty(n, dtype = np.int64)
   gains = np.empty(n)
   for v in np.random.permutation(n):
      if size[ref[v]] > 1:
         continue
      c = comm[v]
      if ext[v] < gamma * _null(a[v], b[v], CA[c] - a[v], CB[c] - b[v], inv_m):
         continue
      nt = 0
      for e in range(ptr[v], ptr[v + 1]):
         j = nbr[e]
         if comm[j] == c:
            r = ref[j]
            score[r] += w[e]
            if not seen[r]:
               seen[r] = True
               touched[nt] = r
               nt += 1
      top = 0.
      for k in range(nt):
         r = touched[k]
         gains[k] = -1.
         if ext[r] >= gamma * _null(RA[r], RB[r], CA[c] - RA[r], CB[c] - RB[r], inv_m):
            g = score[r] - gamma * _null(a[v], b[v], RA[r], RB[r], inv_m)
            if g >= 0:
               gains[k] = g * scale
               top = max(top, gains[k])
      total = np.exp(-top / theta)   # staying alone: gain 0
      for k in range(nt):
         if gains[k] >= 0:
            total += np.exp((gains[k] - top) / theta)
      x = np.random.random() * total - np.exp(-top / theta)
      pick = -1
      for k in range(nt):
         if gains[k] >= 0 and x >= 0:
            x -= np.exp((gains[k] - top) / theta)
            if x < 0:
               pick = touched[k]
      if pick >= 0:
         ref[v] = pick
         size[pick] += 1
         size[v] -= 1
         RA[pick] += a[v]
         RB[pick] += b[v]
         ext[pick] += ext[v] - 2 * score[pick]
      for k in range(nt):
         score[touched[k]], seen[touched[k]] = 0., False
   return ref


def _aggregate(labels, s, t, w, a, b):
   """
   Network of the groups of ``labels`` (0, ..., k - 1): edges between groups summed (edges within
   a group dropped), node weights summed.

   :return: ``(k, ptr, nbr, w, a, b, src)``, the CSR of the aggregated network (``src`` the source
      of every edge), and its node weights.
   """
   k = int(labels.max()) + 1
   x, y = labels[s], labels[t]
   keep = x != y
   key, inv = np.unique(x[keep] * k + y[keep], return_inverse = True)
   w = np.bincount(inv, w[keep])
   s = key // k
   ptr = np.r_[0, np.cumsum(np.bincount(s, minlength = k))]
   A = np.stack([np.bincount(labels, a[:, l], minlength = k) for l in range(a.shape[1])], 1)
   B = np.stack([np.bincount(labels, b[:, l], minlength = k) for l in range(b.shape[1])], 1)
   return k, ptr, key % k, w, A, B, s
