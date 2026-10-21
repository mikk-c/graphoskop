"""
Overlapping Infomap kernels (see :func:`~graphoskop.infomap_communities`), after Esquivel and
Rosvall (2011): a node in several modules has a state (and a codeword) in each; the walker stays in
its module when it steps to a node of the module, otherwise it exits and enters one of the modules
of the node, chosen uniformly. Memberships are ``(n, v)`` arrays ``mem`` with ``nm`` valid entries
per node, state visit rates ``p`` of the same shape.
"""

import numpy as np
from numba import njit

from ._infomap import _plogp


@njit(cache = True, inline = "always")
def _slot(mem, nm, x, k):
   """Position of module ``k`` among the modules of node ``x``, or -1."""
   for q in range(nm[x]):
      if mem[x, q] == k:
         return q
   return -1


@njit(cache = True)
def _teleports(n, p, mem, nm, tau, dang):
   """Teleportation flow leaving every module (indexed by module) and in total."""
   T = np.zeros(n)
   for x in range(n):
      r = 1. if dang[x] else tau
      for q in range(nm[x]):
         T[mem[x, q]] += r * p[x, q]
   return T, T.sum()


@njit(cache = True)
def _state_flows(n, optr, onbr, oq, tau, v, dang, mem, nm, p, tol, max_iter):
   """
   Stationary visit rates of the states, in place (from the given ``p``), by power iteration of
   the lazy walk (same stationary state, no oscillation): steps along links with probability
   :math:`1 - \\tau` (to the start of a random link otherwise, and always from dangling nodes).
   """
   new = np.zeros_like(p)
   for it in range(max_iter):
      new[:] = 0.
      T, R = _teleports(n, p, mem, nm, tau, dang)
      for x in range(n):
         if not dang[x]:
            for q in range(nm[x]):
               j, f = mem[x, q], p[x, q] * (1 - tau)
               for e in range(optr[x], optr[x + 1]):
                  y = onbr[e]
                  s = _slot(mem, nm, y, j)
                  if s >= 0:
                     new[y, s] += f * oq[e]
                  else:
                     for r in range(nm[y]):
                        new[y, r] += f * oq[e] / nm[y]
      if R > 0:
         for y in range(n):
            if v[y] > 0:
               rest = R
               for q in range(nm[y]):
                  new[y, q] += v[y] * T[mem[y, q]]
                  rest -= T[mem[y, q]]
               for q in range(nm[y]):
                  new[y, q] += v[y] * rest / nm[y]
      diff = 0.
      for x in range(n):
         for q in range(nm[x]):
            u = 0.5 * (p[x, q] + new[x, q])
            diff += abs(u - p[x, q])
            p[x, q] = u
      if diff < tol:
         return it + 1
   return max_iter


@njit(cache = True)
def _aggregates(n, optr, onbr, oq, tau, dang, mem, nm, p):
   """
   Coded flows (unrecorded teleportation: only steps along links): the visit rate of every state
   (its incoming link flow) and, by module, the entry and exit rates, the visit rates summed and
   :math:`\\sum p \\log_2 p` of the visit rates.

   :return: ``(vis, enter, exit, P, H, F)``, ``F`` the total link flow (the normalization).
   """
   F = 0.
   for x in range(n):
      if not dang[x]:
         for q in range(nm[x]):
            F += p[x, q] * (1 - tau)
   vis = np.zeros_like(p)
   enter, exit_, P, H = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
   if F == 0:
      return vis, enter, exit_, P, H, F
   for x in range(n):
      if not dang[x]:
         for q in range(nm[x]):
            j, f = mem[x, q], p[x, q] * (1 - tau) / F
            for e in range(optr[x], optr[x + 1]):
               y, c = onbr[e], f * oq[e]
               s = _slot(mem, nm, y, j)
               if s >= 0:
                  vis[y, s] += c
               else:
                  exit_[j] += c
                  for r in range(nm[y]):
                     vis[y, r] += c / nm[y]
                     enter[mem[y, r]] += c / nm[y]
   for x in range(n):
      for q in range(nm[x]):
         P[mem[x, q]] += vis[x, q]
         H[mem[x, q]] += _plogp(vis[x, q])
   return vis, enter, exit_, P, H, F


@njit(cache = True)
def _codelength(enter, exit_, P, H):
   """Two-level map equation from the module aggregates of :func:`_aggregates`."""
   L = _plogp(enter.sum())
   for k in range(len(enter)):
      L += -_plogp(enter[k]) + _plogp(exit_[k] + P[k]) - _plogp(exit_[k]) - H[k]
   return L


@njit(cache = True)
def _deltas(n, optr, onbr, oq, iptr, isrc, iq, tau, v, dang, mem, nm, p, vmax):
   """
   Change of the codelength when one node joins one more module of a neighbor, for every node with
   fewer than ``vmax`` modules and every module of its neighbors (Esquivel and Rosvall's first
   step), with the visit rates of the other states fixed: the states of the node are updated by
   one step of the walk, and only the flows on its links change.

   :return: ``(node, module, delta)`` arrays.
   """
   vis, enter, exit_, P, H, F = _aggregates(n, optr, onbr, oq, tau, dang, mem, nm, p)
   T, R = _teleports(n, p, mem, nm, tau, dang)
   Qtot = enter.sum()
   cap = 0
   for x in range(n):
      cap += (optr[x + 1] - optr[x] + iptr[x + 1] - iptr[x]) * mem.shape[1]
   out_x, out_k, out_d = np.empty(cap, dtype = np.int64), np.empty(cap, dtype = np.int64), np.empty(cap)
   no = 0
   dE, dX, dP, dH = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
   seen = np.zeros(n, dtype = np.bool_)
   touched = np.empty(n, dtype = np.int64)
   cand = np.zeros(n, dtype = np.bool_)
   clist = np.empty(n, dtype = np.int64)
   newmem = np.empty(vmax, dtype = np.int64)
   pnew, vnew = np.zeros(vmax), np.zeros(vmax)
   if F == 0:
      return out_x[:0], out_k[:0], out_d[:0]
   for a in range(n):
      if nm[a] >= vmax:
         continue
      nc = 0   # candidate modules: those of the neighbors, not of a
      for e in range(optr[a], optr[a + 1]):
         y = onbr[e]
         for r in range(nm[y]):
            k = mem[y, r]
            if not cand[k] and _slot(mem, nm, a, k) < 0:
               cand[k] = True
               clist[nc] = k
               nc += 1
      for e in range(iptr[a], iptr[a + 1]):
         y = isrc[e]
         for r in range(nm[y]):
            k = mem[y, r]
            if not cand[k] and _slot(mem, nm, a, k) < 0:
               cand[k] = True
               clist[nc] = k
               nc += 1
      for ci in range(nc):
         k = clist[ci]
         m2 = nm[a] + 1
         for q in range(nm[a]):
            newmem[q] = mem[a, q]
         newmem[nm[a]] = k
         pnew[:m2], vnew[:m2] = 0., 0.
         nt = 0
         # links into a: from every state of every in-neighbor
         for e in range(iptr[a], iptr[a + 1]):
            y = isrc[e]
            if dang[y]:
               continue
            for r in range(nm[y]):
               j = mem[y, r]
               f = p[y, r] * (1 - tau) * iq[e]
               c = f / F
               so = _slot(mem, nm, a, j)
               sn = so if so >= 0 else (nm[a] if j == k else -1)
               if sn >= 0:
                  pnew[sn] += f
                  vnew[sn] += c
               else:
                  for q in range(m2):
                     pnew[q] += f / m2
                     vnew[q] += c / m2
               # entry and exit changes: old routing minus new routing
               if so < 0:
                  if not seen[j]:
                     seen[j] = True
                     touched[nt] = j
                     nt += 1
                  dX[j] -= c
                  for q in range(nm[a]):
                     t = mem[a, q]
                     if not seen[t]:
                        seen[t] = True
                        touched[nt] = t
                        nt += 1
                     dE[t] -= c / nm[a]
               if sn < 0:
                  if not seen[j]:
                     seen[j] = True
                     touched[nt] = j
                     nt += 1
                  dX[j] += c
                  for q in range(m2):
                     t = newmem[q]
                     if not seen[t]:
                        seen[t] = True
                        touched[nt] = t
                        nt += 1
                     dE[t] += c / m2
         if v[a] > 0 and R > 0:   # teleportation into a (uncoded, but part of its states' rates)
            rest = R
            for q in range(m2):
               pnew[q] += v[a] * T[newmem[q]]
               rest -= T[newmem[q]]
            for q in range(m2):
               pnew[q] += v[a] * rest / m2
         # visit rates of the states of a
         for q in range(nm[a]):
            t = mem[a, q]
            if not seen[t]:
               seen[t] = True
               touched[nt] = t
               nt += 1
            dP[t] -= vis[a, q]
            dH[t] -= _plogp(vis[a, q])
         if not seen[k]:
            seen[k] = True
            touched[nt] = k
            nt += 1
         for q in range(m2):
            dP[newmem[q]] += vnew[q]
            dH[newmem[q]] += _plogp(vnew[q])
         # links out of a: old states and rates, then new ones
         if not dang[a]:
            for e in range(optr[a], optr[a + 1]):
               y = onbr[e]
               if y == a:
                  continue
               for side in range(2):
                  sign = -1. if side == 0 else 1.
                  for q in range(nm[a] if side == 0 else m2):
                     i = mem[a, q] if side == 0 else newmem[q]
                     c = sign * (p[a, q] if side == 0 else pnew[q]) * (1 - tau) * oq[e] / F
                     s = _slot(mem, nm, y, i)
                     if s >= 0:
                        if not seen[i]:
                           seen[i] = True
                           touched[nt] = i
                           nt += 1
                        dP[i] += c
                        # plogp of the state of y: changed below from its total change
                        vis[y, s] += c
                     else:
                        if not seen[i]:
                           seen[i] = True
                           touched[nt] = i
                           nt += 1
                        dX[i] += c
                        for r in range(nm[y]):
                           g = mem[y, r]
                           if not seen[g]:
                              seen[g] = True
                              touched[nt] = g
                              nt += 1
                           dE[g] += c / nm[y]
                           dP[g] += c / nm[y]
                           vis[y, r] += c / nm[y]
            for e in range(optr[a], optr[a + 1]):   # the states of the out-neighbors: H, then restore
               y = onbr[e]
               if y == a:
                  continue
               for r in range(nm[y]):
                  g = mem[y, r]
                  old = 0.
                  # undo the change on a copy: recompute the old value from the coded flows
                  old_v = vis[y, r]
                  for side in range(2):
                     sign = -1. if side == 0 else 1.
                     for q in range(nm[a] if side == 0 else m2):
                        i = mem[a, q] if side == 0 else newmem[q]
                        c = sign * (p[a, q] if side == 0 else pnew[q]) * (1 - tau) * oq[e] / F
                        if mem[y, r] == i:
                           old += c
                        elif _slot(mem, nm, y, i) < 0:
                           old += c / nm[y]
                  dH[g] += _plogp(old_v) - _plogp(old_v - old)
                  vis[y, r] = old_v - old
         dQ = 0.
         d = 0.
         for u in range(nt):
            t = touched[u]
            dQ += dE[t]
            d += -_plogp(enter[t] + dE[t]) + _plogp(enter[t])
            d += _plogp(exit_[t] + dX[t] + P[t] + dP[t]) - _plogp(exit_[t] + P[t])
            d += -_plogp(exit_[t] + dX[t]) + _plogp(exit_[t])
            d += -(dH[t])
            dE[t], dX[t], dP[t], dH[t], seen[t] = 0., 0., 0., 0., False
         d += _plogp(Qtot + dQ) - _plogp(Qtot)
         out_x[no], out_k[no], out_d[no] = a, k, d
         no += 1
      for ci in range(nc):
         cand[clist[ci]] = False
   return out_x[:no], out_k[:no], out_d[:no]


def _fuzzy_search(n, s, t, w, directed, tau, assign, vmax, rounds = 50):
   """
   Esquivel and Rosvall's search from a hard partition ``assign``: the changes of codelength of all
   single extra memberships (:func:`_deltas`, the best per node) are sorted, and the best ``k`` of
   them are applied together, ``k`` chosen by computing the exact codelength (stationary state
   rates by :func:`_state_flows`) on a doubling grid refined around its best point; repeated while
   the codelength decreases.

   :return: ``(mem, nm, codelength, rounds)``.
   """
   out = np.bincount(s, w, minlength = n)
   q = w / out[s]
   optr = np.r_[0, np.cumsum(np.bincount(s, minlength = n))]
   io = np.argsort(t, kind = "stable")
   iptr = np.r_[0, np.cumsum(np.bincount(t, minlength = n))]
   isrc, iq = s[io], q[io]
   dang = out == 0
   v = out / out.sum() if directed and out.sum() > 0 else np.zeros(n)
   tau = tau if directed else 0.
   mem = np.full((n, vmax), -1, dtype = np.int64)
   mem[:, 0] = assign
   nm = np.ones(n, dtype = np.int64)
   p = np.zeros((n, vmax))
   p[:, 0] = 1 / n

   def exact(mem, nm, p):
      _state_flows(n, optr, t, q, tau, v, dang, mem, nm, p, 1e-15, 20000)
      return _codelength(*_aggregates(n, optr, t, q, tau, dang, mem, nm, p)[1:5])

   L = exact(mem, nm, p)
   for r in range(rounds):
      x, k, d = _deltas(n, optr, t, q, iptr, isrc, iq, tau, v, dang, mem, nm, p, vmax)
      order = np.argsort(d, kind = "stable")
      x, k, d = x[order], k[order], d[order]
      first = np.unique(x, return_index = True)[1]   # the best per node
      sel = np.sort(first[d[first] < -1e-12])
      if not len(sel):
         break
      x, k = x[sel], k[sel]
      tried = {}

      def trial(m):
         if m not in tried:
            M, C, Q = mem.copy(), nm.copy(), p.copy()
            for a, b in zip(x[:m], k[:m]):
               M[a, C[a]] = b
               C[a] += 1
               Q[a, :C[a]] = Q[a, :C[a] - 1].sum() / C[a]
            tried[m] = (exact(M, C, Q), M, C, Q)
         return tried[m][0]

      grid = sorted(set([2 ** i for i in range(int(np.log2(len(x))) + 1)] + [len(x)]))
      best = min(grid, key = trial)
      i = grid.index(best)
      lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
      for _ in range(8):   # refine between the neighbors on the grid
         cands = [m for m in ((lo + best) // 2, (best + hi) // 2) if m not in tried and lo < m < hi]
         if not cands:
            break
         for m in cands:
            trial(m)
         new = min((m for m in tried if lo <= m <= hi), key = lambda m: tried[m][0])
         if new == best:
            lo, hi = (lo + best) // 2, (best + hi) // 2
         else:
            lo, hi = (lo, best) if new < best else (best, hi)
            best = new
      Lb, M, C, Q = tried[best]
      if not Lb < L - 1e-10:
         break
      L, mem, nm, p = Lb, M, C, Q
   return mem, nm, L, r + 1 if rounds else 0
