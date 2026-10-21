"""
DEMON kernel (see :func:`~graphoskop.demon_communities`): label propagation on the ego network of
every node, minus the node, all egos in parallel, compiled (Numba).
"""

import numpy as np
from numba import njit, prange


@njit(cache = True, parallel = True)
def _ego_labels(n, ptr, nbr, w, max_iter, seed):
   """
   Asynchronous label propagation (as :func:`~graphoskop.label_propagation_communities`) on the
   network induced by the neighbors of every node, without the node: chunks of egos in parallel,
   each with its own random numbers.

   :return: The local label of every entry of the CSR ``(ptr, nbr)`` (the community of neighbor
      ``nbr[e]`` in the ego network of the node owning entry ``e``), as a local index.
   """
   out = np.empty(len(nbr), dtype = np.int64)
   chunks = 64
   for ch in prange(chunks):
      np.random.seed(seed + ch)
      mark = np.full(n, -1, dtype = np.int64)
      for v in range(ch, n, chunks):
         d = ptr[v + 1] - ptr[v]
         if d == 0:
            continue
         for i in range(d):
            mark[nbr[ptr[v] + i]] = i
         lptr = np.zeros(d + 1, dtype = np.int64)   # local CSR
         for i in range(d):
            u = nbr[ptr[v] + i]
            c = 0
            for e in range(ptr[u], ptr[u + 1]):
               x = nbr[e]
               if x != v and mark[x] >= 0:
                  c += 1
            lptr[i + 1] = lptr[i] + c
         lnbr = np.empty(lptr[d], dtype = np.int64)
         lw = np.empty(lptr[d])
         for i in range(d):
            u = nbr[ptr[v] + i]
            k = lptr[i]
            for e in range(ptr[u], ptr[u + 1]):
               x = nbr[e]
               if x != v and mark[x] >= 0:
                  lnbr[k], lw[k] = mark[x], w[e]
                  k += 1
         lab = np.arange(d)
         score = np.zeros(d)
         seen = np.zeros(d, dtype = np.bool_)
         touched = np.empty(d, dtype = np.int64)
         best = np.empty(d, dtype = np.int64)
         order = np.arange(d)
         for it in range(max_iter):
            np.random.shuffle(order)
            changed = 0
            for x in order:
               if lptr[x] == lptr[x + 1]:
                  continue
               nt = 0
               for a in range(lptr[x], lptr[x + 1]):
                  l = lab[lnbr[a]]
                  score[l] += lw[a]
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
                     if l == lab[x]:
                        keep = True
               if not keep:
                  lab[x] = best[np.random.randint(nb)]
                  changed += 1
               for t in range(nt):
                  score[touched[t]], seen[touched[t]] = 0., False
            if changed == 0:
               break
         for i in range(d):
            out[ptr[v] + i] = lab[i]
            mark[nbr[ptr[v] + i]] = -1
   return out


@njit(cache = True)
def _merge(N, cptr, cnodes, epsilon):
   """
   Merge local communities (CSR ``cptr``, ``cnodes``), in order, into global ones: each joins the
   global community sharing the largest fraction of the smaller of the two, if at least
   ``epsilon``, or becomes a new one. Global communities are kept as an inverted index (the
   communities of every node, as linked lists), so the overlaps of a local community are counted
   by visiting the communities of its nodes only.

   :return: ``(node, community)`` of every membership, and the number of global communities.
   """
   K = len(cptr) - 1
   cap = len(cnodes)
   ent_comm = np.empty(cap, dtype = np.int64)
   ent_next = np.empty(cap, dtype = np.int64)
   ent_node = np.empty(cap, dtype = np.int64)
   head = np.full(N, -1, dtype = np.int64)
   size = np.zeros(K, dtype = np.int64)
   cnt = np.zeros(K, dtype = np.int64)
   touched = np.empty(K, dtype = np.int64)
   ne, ng = 0, 0
   for k in range(K):
      a, b = cptr[k], cptr[k + 1]
      m = b - a
      nt = 0
      for p in range(a, b):
         e = head[cnodes[p]]
         while e >= 0:
            g = ent_comm[e]
            if cnt[g] == 0:
               touched[nt] = g
               nt += 1
            cnt[g] += 1
            e = ent_next[e]
      best, ratio = -1, 0.
      for i in range(nt):
         g = touched[i]
         r = cnt[g] / min(m, size[g])
         if r >= epsilon - 1e-12 and r > ratio:
            best, ratio = g, r
         cnt[g] = 0
      if best < 0:
         best = ng
         ng += 1
      for p in range(a, b):
         v = cnodes[p]
         e = head[v]
         inside = False
         while e >= 0:
            if ent_comm[e] == best:
               inside = True
               break
            e = ent_next[e]
         if not inside:
            ent_comm[ne], ent_node[ne], ent_next[ne] = best, v, head[v]
            head[v] = ne
            ne += 1
            size[best] += 1
   return ent_node[:ne], ent_comm[:ne], ng
