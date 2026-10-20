"""
Network error correction: inferring the true network behind noisy observations.

Observed networks have errors: spurious edges that are not there and missing edges that are.
Network data can be cleaned in two separate steps, filtering out the weakest edges with a
backbone (see :doc:`backboning`) and adding the most likely missing ones with link prediction
(see :doc:`link_prediction`). The methods of this chapter do both at once, in a single and
consistent framework (Peixoto, 2018): a generative model of the network, by default the nested
degree-corrected stochastic blockmodel, is fitted jointly with a model of the measurement errors,
and the posterior probability of every node pair being an edge is estimated by sampling. Edges
with a low probability are likely spurious, non-edges with a high probability likely missing.
Simpler generative models can be used instead, given as the generators of their families
(:func:`~graphoskop.degree_corrected_block_model`, :func:`~graphoskop.stochastic_block_model`,
:func:`~graphoskop.planted_partition_graph`, :func:`~graphoskop.configuration_model`,
:func:`~graphoskop.gnm_random_graph`): the more structure a model captures, the better it tells
errors apart.

:func:`fit_nested_block_model` finds the hierarchy of groups that best compresses a network
(the minimum description length of the nested blockmodel). :func:`measured_network_correction`
reconstructs a network from repeated measurements of its node pairs (or from a single
observation), with error rates known or unknown, the same for every pair or heterogeneous.
:func:`uncertain_network_correction` reconstructs a network whose node pairs come with
probabilities of being edges, estimated by some other measurement. The generator
:func:`~graphoskop.nested_block_model` samples networks from the model, or from its fit to a
network.

All the inference runs in compiled kernels (Numba) on the CPU, as in graph-tool: Markov chain
moves are sequential and branchy, which suits a CPU core better than a GPU. The candidate merges
of the fit are evaluated in parallel, and the posterior is sampled by several independent chains
in parallel threads. The kernels are compiled the first time they run (about two minutes, once:
they are cached). For scale, on two CPU cores: the fit takes 0.3
seconds for 800 nodes and 4,400 edges, 4.5 seconds for 9,600 nodes and 56,000 edges, and 40
seconds for 50,000 nodes and 350,000 edges; a sweep of the sampler, about one microsecond per
move, takes 11 milliseconds for 960 nodes and 5,800 edges, so reconstructing it with the default
1,200 sweeps per chain takes about 14 seconds with two chains (one per core).
"""

import math
import threading

import numpy as np
import pandas as pd
import torch
from numba import njit, prange
from numba.typed import List

_LN2 = math.log(2.)
_Q_EXACT = 1000   # partition counts q(m, n) computed exactly up to m = _Q_EXACT
_U_MAX = 50.      # beyond u = n / sqrt(m) = 50, q(m, n) is the number of partitions of m
_CACHE = {}
_DNEW = 0.01      # probability of proposing a new group (Peixoto 2018, Eq. B4)


# ------------------------------------------------------------------------------------------------
# Description length: elementary terms
# ------------------------------------------------------------------------------------------------

def _li2(y):
   """
   Dilogarithm :math:`\\mathrm{Li}_2(y)` for :math:`y \\in [0, 1/2]` (power series).
   """
   s, t, k = 0., y, 1
   while t > 0:
      term = t / (k * k)
      s += term
      if term < 1e-17 * s:
         break
      k += 1
      t *= y
   return s


def _li2c(w):
   """
   :math:`\\mathrm{Li}_2(1 - w)` for :math:`w \\in (0, 1]`, accurate for small ``w``.
   """
   if w < 0.5:
      return math.pi ** 2 / 6 - math.log1p(-w) * math.log(w) - _li2(w)
   return _li2(1 - w)


def _q_tables():
   """
   Tables for :func:`_log_q`, computed once: exact :math:`\\ln q(m, n)` for :math:`m \\leq 1000`
   (recurrence :math:`q(m, n) = q(m, n - 1) + q(m - n, n)`), and the solution :math:`v(u)` of
   :math:`v = u \\sqrt{\\mathrm{Li}_2(1 - e^{-v})}` of Szekeres' asymptotic formula on a grid of
   :math:`\\ln u`.

   :return: ``(qtab, lu, v)``.
   """
   if "q" not in _CACHE:
      M = _Q_EXACT
      q = np.zeros((M + 1, M + 1))
      q[0, :] = 1.
      for m in range(1, M + 1):   # q(m, n) = sum over j <= n of q(m - j, j): parts of size j at least once
         j = np.arange(1, m + 1)
         q[m, 1:m + 1] = np.cumsum(q[m - j, j])
         q[m, m + 1:] = q[m, m]
      with np.errstate(divide = "ignore"):
         qtab = np.log(q)
      lu = np.linspace(math.log(1e-6), math.log(_U_MAX), 4001)
      v = np.empty(len(lu))
      for i, x in enumerate(lu):
         u = math.exp(x)
         lo, hi = 0., 10. + 2. * u
         for _ in range(200):   # bisection: v - u sqrt(Li2(1 - e^-v)) increases with v
            mid = (lo + hi) / 2
            if mid - u * math.sqrt(_li2c(math.exp(-mid))) > 0:
               hi = mid
            else:
               lo = mid
         v[i] = (lo + hi) / 2
      _CACHE["q"] = (qtab, lu, v)
   return _CACHE["q"]


@njit(cache = True, inline = "always")
def _lg(x):
   """:math:`\\ln x!`."""
   return math.lgamma(x + 1.)


@njit(cache = True, inline = "always")
def _dlg(x, d):
   """:math:`\\ln (x + d)! - \\ln x!`, with logarithms instead of log-gamma for :math:`|d| \\leq 2`."""
   if d == 0:
      return 0.
   if d == 1:
      return math.log(x + 1.)
   if d == -1:
      return -math.log(x)
   if d == 2:
      return math.log((x + 1.) * (x + 2.))
   if d == -2:
      return -math.log(x * (x - 1.))
   return _lg(x + d) - _lg(x)


@njit(cache = True, inline = "always")
def _dms(n, x, dm):
   """Change of :func:`_lms` ``(n, x)`` when ``x`` changes by ``dm`` (:math:`\\pm 1`)."""
   if dm == 1:
      return math.log(n + x) - math.log(x + 1.)
   return math.log(x) - math.log(n + x - 1.)


@njit(cache = True, inline = "always")
def _lms(n, k):
   """
   Logarithm of the multiset coefficient :math:`\\left(\\!\\binom{n}{k}\\!\\right) =
   \\binom{n + k - 1}{k}`, the number of ways of placing ``k`` edges in ``n`` pairs.
   """
   if k <= 0:
      return 0.
   return math.lgamma(n + k) - math.lgamma(k + 1.) - math.lgamma(n)


@njit(cache = True, inline = "always")
def _lbinom(n, k):
   """:math:`\\ln \\binom{n}{k}`."""
   return math.lgamma(n + 1.) - math.lgamma(k + 1.) - math.lgamma(n - k + 1.)


@njit(cache = True, inline = "always")
def _log_q(m, n, qtab, lu, vt):
   """
   Logarithm of :math:`q(m, n)`, the number of partitions of the integer ``m`` into at most ``n``
   parts (the number of degree sequences of ``n`` nodes with ``m`` half-edges, up to order).
   Exact for :math:`m \\leq 1000` and :math:`n \\leq 3`; otherwise Szekeres' asymptotic formula
   :math:`q(m, n) \\approx f(u) e^{\\sqrt{m} g(u)} / m`, :math:`u = n / \\sqrt{m}` (error below 0.1).
   """
   if m <= 0:
      return 0.
   if n <= 0:
      return -np.inf
   if n > m:
      n = m
   if m <= _Q_EXACT:
      return qtab[m, n]
   if n == 1:
      return 0.
   if n == 2:
      return math.log(m // 2 + 1)
   if n == 3:
      return math.log(round((m + 3) ** 2 / 12))
   u = min(n / math.sqrt(m), _U_MAX)
   v = np.interp(math.log(u), lu, vt)
   f = v / (2 ** 1.5 * math.pi * u) / math.sqrt(1 - (1 + u * u / 2) * math.exp(-v))
   g = 2 * v / u - u * math.log1p(-math.exp(-v))
   return math.log(f / m) + math.sqrt(m) * g


@njit(cache = True, inline = "always")
def _adj(x, diag):
   """
   Term of a pair of level-0 groups in the degree-corrected likelihood: :math:`-\\ln e_{rs}!`, or
   :math:`-\\ln e_{rr}!!` for the ``x`` edges within a group (:math:`e_{rr} = 2x`).
   """
   return -(x * _LN2 + _lg(x)) if diag else -_lg(x)


@njit(cache = True, inline = "always")
def _ms(nt, nu, x, diag):
   """
   Term of a pair of groups of an upper level: the logarithm of the number of ways of placing
   ``x`` edges among the pairs of their members.
   """
   return _lms(nt * (nt + 1) / 2. if diag else nt * float(nu), x)


# ------------------------------------------------------------------------------------------------
# Hash tables (open addressing) and growable lists
# ------------------------------------------------------------------------------------------------

@njit(cache = True, inline = "always")
def _hfind(keys, key):
   """
   Slot of ``key`` in an open addressing table (-1 empty, -2 deleted), or the first free slot.
   """
   mask = len(keys) - 1
   h = ((key * -7046029254386353131) >> 29) & mask   # Fibonacci hashing: high bits of the product
   free = -1
   while True:
      k = keys[h]
      if k == key:
         return h
      if k == -1:
         return h if free < 0 else free
      if k == -2 and free < 0:
         free = h
      h = (h + 1) & mask


@njit(cache = True, inline = "always")
def _hget(keys, vals, key):
   """Value of ``key``, 0 if absent."""
   h = _hfind(keys, key)
   return vals[h] if keys[h] == key else 0


@njit(cache = True)
def _hrehash(keys, vals, size):
   """
   Table of at least ``2 size`` slots with the entries of ``keys``, ``vals``.
   """
   cap = 16
   while cap < 2 * size:
      cap *= 2
   nk, nv = np.full(cap, -1, dtype = np.int64), np.zeros(cap, dtype = np.int64)
   for h in range(len(keys)):
      if keys[h] >= 0:
         p = _hfind(nk, keys[h])
         nk[p], nv[p] = keys[h], vals[h]
   return nk, nv


@njit(cache = True, inline = "always")
def _hslot(keys, vals, used, key):
   """
   Slot of ``key``, inserting it (value 0) if absent; the table grows when half full.

   :param used: One-element array with the number of used slots (deleted ones included).
   :return: ``(keys, vals, slot)``, the table possibly reallocated.
   """
   h = _hfind(keys, key)
   if keys[h] != key:
      if keys[h] == -1:
         used[0] += 1
      keys[h], vals[h] = key, 0
      if 2 * used[0] > len(keys):
         keys, vals = _hrehash(keys, vals, used[0])
         used[0] = np.sum(keys >= 0)
         h = _hfind(keys, key)
   return keys, vals, h


@njit(cache = True, inline = "always")
def _push(lst, size, i, val):
   """
   Append ``val`` to the growable list ``i`` (arrays ``lst[i]`` with ``size[i]`` items).

   :return: Its position.
   """
   a = lst[i]
   if size[i] == len(a):
      b = np.empty(max(4, 2 * len(a)), dtype = np.int64)
      b[:len(a)] = a
      lst[i] = b
      a = b
   a[size[i]] = val
   size[i] += 1
   return size[i] - 1


# ------------------------------------------------------------------------------------------------
# Blockmodels: state and description length
# ------------------------------------------------------------------------------------------------
# Model kinds: 0, degree-corrected (nested; flat with a single level; the configuration model with
# a single group); 2, without degree correction (Erdos-Renyi with a single group); 3, planted
# partition. All are microcanonical multigraph models (Peixoto 2017): the sampled network is a
# multigraph G, with self loops, and the network seen by the measurements is its collapse A
# (A_ij = 1 if G_ij > 0, i != j; Peixoto 2018, Appendix D).
#
# Levels (flat arrays: typed lists of arrays are slow to index in Numba). Ps[l, x] is the group
# at level l of the element x of level l - 1 (Ps[0, i]: the level 0 group of node i), Ns[l, t]
# the number of elements of the group t of level l, and Ms = (Mf, mo, Bl) the edge counts between
# groups: level l is the Bl[l] x Bl[l] matrix stored from Mf[mo[l]] (diagonal: edges within, self
# loops included). Level 0 also has the degrees k (a self loop counting 2), group degree sums e,
# degree histograms (hash of k * B0 + r: degrees are unbounded in a multigraph), and aux = (edges within groups, sum of the squared
# group sizes). Above the last level, a single group with all the E edges.
#
# Graph: one slot per edge of G, ed[slot] = (u, v, position in the list of edges, position in the
# adjacency of u, of v, position of the half-edge of u, of v in the half-edge list of its group,
# next slot of the same pair, multiplicity of the pair (in its first slot)), u <= v; free slots;
# pair hash (u * N + v) to the first slot of the pair; adjacency lists of slots (a self loop twice);
# half-edge lists per level-0 group (half-edge 2 slot + side); member lists per level-0 group.

@njit(cache = True, inline = "always")
def _mget(Ms, l, a, b):
   """Edges between the groups ``a`` and ``b`` of level ``l``."""
   Mf, mo, Bl = Ms
   return Mf[mo[l] + a * Bl[l] + b]


@njit(cache = True, inline = "always")
def _madd(Ms, l, a, b, x):
   """Add ``x`` edges between the groups ``a`` and ``b`` of level ``l``."""
   Mf, mo, Bl = Ms
   Mf[mo[l] + a * Bl[l] + b] += x
   if a != b:
      Mf[mo[l] + b * Bl[l] + a] += x


@njit(cache = True)
def _levels(N, eu, ev, b0, par, Bs):
   """
   Level arrays ``(Ps, Ns, Ms)`` of a nested partition, the degrees ``k`` of the edge list and
   ``aux`` (``par``: the groups of the upper levels, concatenated).
   """
   L = len(Bs)
   Ps = np.full((L, max(N, Bs[0])), -1, dtype = np.int64)
   Ns = np.zeros((L, Bs[0]), dtype = np.int64)
   mo = np.zeros(L + 1, dtype = np.int64)
   Ps[0, :N] = b0
   at = 0
   for l in range(L):
      if l > 0:
         Ps[l, :Bs[l - 1]] = par[at:at + Bs[l - 1]]
         at += Bs[l - 1]
      for x in range(N if l == 0 else Bs[l - 1]):
         if l == 0 or Ns[l - 1, x] > 0:   # empty groups below have no group
            Ns[l, Ps[l, x]] += 1
      mo[l + 1] = mo[l] + Bs[l] * Bs[l]
   Ms = (np.zeros(mo[L], dtype = np.int64), mo, Bs.copy())
   k = np.zeros(N, dtype = np.int64)
   aux = np.zeros(2, dtype = np.int64)
   for t in range(len(eu)):
      k[eu[t]] += 1
      k[ev[t]] += 1
      a, b = b0[eu[t]], b0[ev[t]]
      aux[0] += a == b
      for l in range(L):
         if l > 0:
            a, b = Ps[l, a], Ps[l, b]
         _madd(Ms, l, a, b, 1)
   for r in range(Bs[0]):
      aux[1] += Ns[0, r] * Ns[0, r]
   return Ps, Ns, Ms, k, aux


@njit(cache = True, inline = "always")
def _pp_lik(ein, eout, sq, N):
   """
   Minus the log-likelihood of the planted partition model, without the multiplicity terms:
   ``ein`` edges placed uniformly among the pairs within groups (:math:`\\sum_r n_r^2 / 2`, ``sq``
   the sum of the squared group sizes), ``eout`` among the pairs between groups.
   """
   return -_lg(ein) - _lg(eout) + _xlog(ein, sq / 2.) + _xlog(eout, (N * float(N) - sq) / 2.)


@njit(cache = True, inline = "always")
def _ptop(kind, Bocc):
   """
   Number of classes of pairs of groups among which the edges are spread at the top: pairs of
   groups of the last level (pairs within and between groups for the planted partition), from
   the numbers of occupied groups ``Bocc`` of every level.
   """
   D = Bocc[len(Bocc) - 1]
   if kind == 3:
      return 2. if Bocc[0] > 1 else 1.
   return D * (D + 1) / 2.


@njit(cache = True, inline = "always")
def _edges_prior(P, E, lam):
   """
   Minus the log-probability of ``E`` edges, negative binomial: the number of edges of each of the
   ``P`` classes of pairs geometric with mean ``lam`` (Peixoto 2018, Eq. C9), divided by the
   uniform multiset already counted; 0 for ``lam = 0`` (``E`` fixed).
   """
   if lam <= 0:
      return 0.
   return -_lms(P, E) - E * math.log(lam / (1 + lam)) + P * math.log1p(lam)


@njit(cache = True)
def _bglobal(N, E, Bocc, lx, dx, Cs, kind, lam, vary):
   """
   Terms of the description length that depend only on the numbers of occupied groups
   ``Bocc`` (that of level ``lx`` changed by ``dx``): the partition priors but for the group
   sizes, the prior of the edge counts at the top and of the number of edges, and, if ``vary``,
   the labels: with ``Cs[l]`` labels for the groups of level ``l``, empty ones included, every
   partition into ``B`` groups has :math:`C! / (C - B)!` labellings instead of :math:`B!`.
   """
   L = len(Bocc)
   S = 0.
   prev = N
   for l in range(L):
      B = Bocc[l] + (dx if l == lx else 0)
      if kind == 0 or l == 0:
         S += _lbinom(prev - 1, B - 1) + _lg(prev) + math.log(prev)
      if vary:
         S -= _lg(B) + _lg(Cs[l] - B)
      prev = B
   B0 = Bocc[0] + (dx if lx == 0 else 0)
   if kind == 0:
      S += math.log(prev)
      P = prev * (prev + 1) / 2.
   elif kind == 2:
      P = B0 * (B0 + 1) / 2.
   else:
      P = 2. if B0 > 1 else 1.
   return S + _lms(P, E) + _edges_prior(P, E, lam)


@njit(cache = True)
def _size_delta(l, T, dn, Ns, Ms):
   """
   Change of the terms of level ``l >= 1`` when the size of its group ``T`` changes by ``dn``
   (an element below created or emptied), the edge counts unchanged.
   """
   n = Ns[l, T]
   d = -_dlg(n, dn)
   for u in range(Ms[2][l]):
      if u == T:
         m = _mget(Ms, l, T, T)
         d += _ms(n + dn, n + dn, m, True) - _ms(n, n, m, True)
      elif Ns[l, u] > 0:
         m = _mget(Ms, l, T, u)
         d += _ms(n + dn, Ns[l, u], m, False) - _ms(n, Ns[l, u], m, False)
   return d


@njit(cache = True)
def _occupied(Ns, Ms):
   """Number of occupied groups of every level."""
   Bl = Ms[2]
   Bocc = np.zeros(len(Bl), dtype = np.int64)
   for l in range(len(Bl)):
      for t in range(Bl[l]):
         Bocc[l] += Ns[l, t] > 0
   return Bocc


@njit(cache = True)
def _description_length(N, E, k, Ps, Ns, Ms, e, hk, hv, kind, aux, mterm, lam, vary, qtab, lu, vt):
   """
   Description length (negative log-probability) of a multigraph and its partition (nested for
   kind 0): the microcanonical likelihood, the degree prior (kind 0), the priors of the edge
   counts between groups, of the number of edges (if ``lam > 0``, see :func:`_edges_prior`) and
   the partition priors (Peixoto 2014, 2017), over the occupied groups (with the label term of
   :func:`_bglobal` if ``vary``). ``mterm`` is
   :math:`\\sum_{i < j} \\ln G_{ij}! + \\sum_i \\ln G_{ii}!!`, 0 for a simple graph.
   """
   Bl = Ms[2]
   L = len(Bl)
   C0 = Bl[0]
   S = mterm + _bglobal(N, E, _occupied(Ns, Ms), -1, 0, Bl, kind, lam, vary)
   if kind != 0:   # flat partition, with its group sizes
      for r in range(C0):
         S -= _lg(Ns[0, r])
      if kind == 3:
         return S + _pp_lik(aux[0], E - aux[0], aux[1], N)
      for r in range(C0):
         for s in range(r, C0):
            S += _adj(_mget(Ms, 0, r, s), r == s)
         S += _xlog(e[r], Ns[0, r])
      return S
   for r in range(C0):
      for s in range(r, C0):
         S += _adj(_mget(Ms, 0, r, s), r == s)
      S += _lg(e[r]) + _log_q(e[r], Ns[0, r], qtab, lu, vt)
   for i in range(N):
      S -= _lg(k[i])
   for h in range(len(hk)):
      if hk[h] >= 0:
         S -= _lg(hv[h])
   for l in range(1, L):   # level 0 group sizes cancel with the degree prior
      for t in range(Bl[l]):
         if Ns[l, t] == 0:
            continue
         S -= _lg(Ns[l, t])
         for u in range(t, Bl[l]):
            if Ns[l, u] > 0:
               S += _ms(Ns[l, t], Ns[l, u], _mget(Ms, l, t, u), t == u)
   return S


@njit(cache = True)
def _build(N, eu, ev, b0, par, Bs, cap):
   """
   Blockmodel state of a simple undirected graph (the starting multigraph).

   :param N: Number of nodes.
   :param eu: First endpoints of the edges (``eu < ev``, no repeated pairs).
   :param ev: Second endpoints.
   :param b0: Level 0 group of every node, from 0 to ``Bs[0] - 1``, all groups non-empty.
   :param par: Groups of the upper levels, concatenated: for each level ``l >= 1``, the group of
      each group of level ``l - 1``.
   :param Bs: Number of groups of each level.
   :param cap: Initial capacity of the edge arrays.
   :return: The state (see the module comments).
   """
   Ps, Ns, Ms, k, aux = _levels(N, eu, ev, b0, par, Bs)
   m = len(eu)
   cap = max(cap, 2 * m + 16)
   ed = np.full((cap, 9), -1, dtype = np.int64)
   alive = np.empty(cap, dtype = np.int64)
   free = np.empty(cap, dtype = np.int64)   # stack of free slots, room for all
   free[:cap - m] = np.arange(cap - 1, m - 1, -1)
   nfree = cap - m
   ek, evl = _hrehash(np.full(16, -1, dtype = np.int64), np.zeros(16, dtype = np.int64), 2 * m + 16)
   adj, nadj = List(), np.zeros(N, dtype = np.int64)
   memb, nmemb, mpos = List(), np.zeros(Bs[0], dtype = np.int64), np.empty(N, dtype = np.int64)
   eg, neg = List(), np.zeros(Bs[0], dtype = np.int64)
   for i in range(N):
      adj.append(np.empty(4, dtype = np.int64))
   for r in range(Bs[0]):
      memb.append(np.empty(4, dtype = np.int64))
      eg.append(np.empty(4, dtype = np.int64))
   for i in range(N):
      mpos[i] = _push(memb, nmemb, b0[i], i)
   for sl in range(m):
      u, v = eu[sl], ev[sl]
      ed[sl, 0], ed[sl, 1], ed[sl, 2], ed[sl, 8] = u, v, sl, 1
      alive[sl] = sl
      ed[sl, 3] = _push(adj, nadj, u, sl)
      ed[sl, 4] = _push(adj, nadj, v, sl)
      ed[sl, 5] = _push(eg, neg, b0[u], 2 * sl)
      ed[sl, 6] = _push(eg, neg, b0[v], 2 * sl + 1)
      h = _hfind(ek, u * N + v)
      ek[h], evl[h] = u * N + v, sl
   e = np.zeros(Bs[0], dtype = np.int64)
   for i in range(N):
      e[b0[i]] += k[i]
   hk, hv = _hrehash(np.full(16, -1, dtype = np.int64), np.zeros(16, dtype = np.int64), N + 16)
   for i in range(N):
      h = _hfind(hk, k[i] * Bs[0] + b0[i])
      hk[h] = k[i] * Bs[0] + b0[i]
      hv[h] += 1
   hn, en = np.array([np.sum(hk >= 0)]), np.array([m])
   return Ps, Ns, Ms, k, e, hk, hv, hn, ed, alive, free, nfree, ek, evl, en, adj, nadj, memb, nmemb, mpos, eg, neg, aux


@njit(cache = True, inline = "always")
def _mult(ek, evl, ed, key):
   """Multiplicity of the pair ``key`` in the multigraph."""
   h = _hfind(ek, key)
   return ed[evl[h], 8] if ek[h] == key else 0


@njit(cache = True)
def _mterm(ek, evl, ed, N):
   """:math:`\\sum_{i < j} \\ln G_{ij}! + \\sum_i \\ln G_{ii}!!` of the multigraph."""
   S = 0.
   for h in range(len(ek)):
      if ek[h] >= 0:
         g = ed[evl[h], 8]
         S += (g * _LN2 + _lg(g)) if ek[h] // N == ek[h] % N else _lg(g)
   return S


@njit(cache = True)
def _upper_delta(Ps, Ns, Ms, l0, a, b, cnt, tl, ntl, inside, apply, scr):
   """
   Change of the edge count terms of the levels above ``l0`` when edges move from group ``a`` to
   group ``b`` of level ``l0``: ``cnt[t]`` edges joining the moved element to group ``t`` of level
   ``l0`` (``t`` in ``tl[:ntl]``), plus ``inside`` edges with both ends in the moved element.
   With ``apply``, the counts are updated too.

   :param scr: Scratch arrays ``(dA, dB, cur, vl)`` of length at least the largest number of
      groups plus 2, ``dA`` and ``dB`` zero.
   :return: Change of the description length.
   """
   dA, dB, cur, vl = scr
   L = len(Ms[2])
   d = 0.
   for j in range(ntl):
      cur[j] = tl[j]
   for l in range(l0 + 1, L):
      a, b = Ps[l, a], Ps[l, b]
      for j in range(ntl):
         cur[j] = Ps[l, cur[j]]
      if a == b:
         break
      nv = 0   # groups of this level touched, each listed once
      for j in range(ntl):
         v = cur[j]
         if dA[v] == 0 and dB[v] == 0:
            vl[nv] = v
            nv += 1
         c = cnt[tl[j]]
         dA[v] -= c
         dB[v] += c
      for v in (a, b):
         found = False
         for x in range(nv):
            if vl[x] == v:
               found = True
         if not found:
            vl[nv] = v
            nv += 1
      dA[a] -= inside
      dB[b] += inside
      for x in range(nv):
         v = vl[x]
         for y, dx in ((a, dA[v]), (b, dB[v])):
            if dx == 0 or (y == a and v == b) or (y == b and v == a):
               continue
            m = _mget(Ms, l, y, v)
            d += _ms(Ns[l, y], Ns[l, v], m + dx, y == v) - _ms(Ns[l, y], Ns[l, v], m, y == v)
            if apply:
               _madd(Ms, l, y, v, dx)
      dab = dA[b] + dB[a]   # the pair (a, b) collects from both sides
      if dab != 0:
         m = _mget(Ms, l, a, b)
         d += _ms(Ns[l, a], Ns[l, b], m + dab, False) - _ms(Ns[l, a], Ns[l, b], m, False)
         if apply:
            _madd(Ms, l, a, b, dab)
      for x in range(nv):
         dA[vl[x]] = 0
         dB[vl[x]] = 0
   return d


@njit(cache = True, inline = "always")
def _node_counts(i, Ps, adj, nadj, ed, cnt, tl):
   """
   Number of edges from node ``i`` to every level 0 group, self loops apart: ``cnt[t]`` for ``t``
   in ``tl[:n]`` (``cnt`` must be zero for all groups before, and should be cleared after).

   :return: ``(n, loops)``.
   """
   n, loops = 0, 0
   a = adj[i]
   for p in range(nadj[i]):
      sl = a[p]
      if ed[sl, 0] == ed[sl, 1]:
         loops += p == ed[sl, 3]   # listed twice, counted once
         continue
      j = ed[sl, 1] if ed[sl, 0] == i else ed[sl, 0]
      t = Ps[0, j]
      if cnt[t] == 0:
         tl[n] = t
         n += 1
      cnt[t] += 1
   return n, loops


@njit(cache = True)
def _node_delta(i, s, N, E, k, Ps, Ns, Ms, e, hk, hv, cnt, tl, ntl, loops, kind, aux, scr, qtab, lu, vt):
   """
   Change of the description length when node ``i`` moves to the level 0 group ``s`` (``cnt``,
   ``tl``, ``ntl``, ``loops`` from :func:`_node_counts`).
   """
   r = Ps[0, i]
   ki = k[i]
   cr, cs = cnt[r], cnt[s]
   nr, ns = Ns[0, r], Ns[0, s]
   if kind == 3:   # planted partition: edges within and between groups, and group sizes
      ein, sq = aux[0], aux[1]
      ein2, sq2 = ein + cs - cr, sq - 2 * nr + 2 * ns + 2
      return _pp_lik(ein2, E - ein2, sq2, N) - _pp_lik(ein, E - ein, sq, N) + math.log(nr) - math.log(ns + 1.)
   d = 0.
   for j in range(ntl):
      t = tl[j]
      if t == r or t == s:
         continue
      c = cnt[t]
      d += -_dlg(_mget(Ms, 0, r, t), -c) - _dlg(_mget(Ms, 0, s, t), c)
   d += _adj(_mget(Ms, 0, r, s) - cs + cr, False) - _adj(_mget(Ms, 0, r, s), False)
   mrr, mss = _mget(Ms, 0, r, r), _mget(Ms, 0, s, s)
   d += _adj(mrr - cr - loops, True) - _adj(mrr, True) + _adj(mss + cs + loops, True) - _adj(mss, True)
   if kind == 2:
      return d + _xlog(e[r] - ki, nr - 1) - _xlog(e[r], nr) + _xlog(e[s] + ki, ns + 1) - _xlog(e[s], ns) + math.log(nr) - math.log(ns + 1.)
   d += _dlg(e[r], -ki) + _dlg(e[s], ki)
   d += _log_q(e[r] - ki, nr - 1, qtab, lu, vt) - _log_q(e[r], nr, qtab, lu, vt)
   d += _log_q(e[s] + ki, ns + 1, qtab, lu, vt) - _log_q(e[s], ns, qtab, lu, vt)
   B0 = Ms[2][0]
   d -= _dlg(_hget(hk, hv, ki * B0 + r), -1) + _dlg(_hget(hk, hv, ki * B0 + s), 1)
   return d + _upper_delta(Ps, Ns, Ms, 0, r, s, cnt, tl, ntl, loops, False, scr)


@njit(cache = True)
def _node_apply(i, s, N, k, Ps, Ns, Ms, e, hk, hv, hn, cnt, tl, ntl, loops, aux, scr, ed, adj, nadj, memb, nmemb, mpos, eg, neg):
   """
   Move node ``i`` to the level 0 group ``s``, updating all counts and lists.

   :return: The (possibly reallocated) degree histogram table ``(hk, hv)``.
   """
   r = Ps[0, i]
   for j in range(ntl):
      t = tl[j]
      c = cnt[t]
      if t == r:
         _madd(Ms, 0, r, r, -c)
         _madd(Ms, 0, s, r, c)
      elif t == s:
         _madd(Ms, 0, r, s, -c)
         _madd(Ms, 0, s, s, c)
      else:
         _madd(Ms, 0, r, t, -c)
         _madd(Ms, 0, s, t, c)
   _madd(Ms, 0, r, r, -loops)
   _madd(Ms, 0, s, s, loops)
   _upper_delta(Ps, Ns, Ms, 0, r, s, cnt, tl, ntl, loops, True, scr)
   aux[0] += cnt[s] - cnt[r]
   aux[1] += 2 * (Ns[0, s] - Ns[0, r]) + 2
   ki = k[i]
   e[r] -= ki
   e[s] += ki
   Ns[0, r] -= 1
   Ns[0, s] += 1
   B0 = Ms[2][0]
   h = _hfind(hk, ki * B0 + r)
   hv[h] -= 1
   hk, hv, h = _hslot(hk, hv, hn, ki * B0 + s)
   hv[h] += 1
   Ps[0, i] = s
   # member and half-edge lists
   m = memb[r]
   last = m[nmemb[r] - 1]
   m[mpos[i]] = last
   mpos[last] = mpos[i]
   nmemb[r] -= 1
   mpos[i] = _push(memb, nmemb, s, i)
   a = adj[i]
   g = eg[r]
   for p in range(nadj[i]):
      sl = a[p]
      if ed[sl, 0] == ed[sl, 1]:   # self loop: both half-edges, at its first listing
         if p != ed[sl, 3]:
            continue
         sides = 2
      else:
         sides = 1
      for q in range(sides):
         side = q if sides == 2 else (0 if ed[sl, 0] == i else 1)
         last = g[neg[r] - 1]
         g[ed[sl, 5 + side]] = last
         ed[last // 2, 5 + last % 2] = ed[sl, 5 + side]
         neg[r] -= 1
         ed[sl, 5 + side] = _push(eg, neg, s, 2 * sl + side)
   return hk, hv


@njit(cache = True)
def _elem_rows(l, g, Ps, Ms, cnt, tl):
   """
   Edges from the group ``g`` of level ``l - 1`` to every group of level ``l`` (``cnt``, ``tl`` as
   in :func:`_node_counts`), and the edges within ``g``.

   :return: ``(n, inside)``.
   """
   n = 0
   for h in range(Ms[2][l - 1]):
      w = _mget(Ms, l - 1, g, h)
      if h == g or w == 0:
         continue
      t = Ps[l, h]
      if cnt[t] == 0:
         tl[n] = t
         n += 1
      cnt[t] += w
   return n, _mget(Ms, l - 1, g, g)


@njit(cache = True)
def _elem_delta(l, g, S, Ps, Ns, Ms, cnt, tl, ntl, inside, scr):
   """
   Change of the description length when the group ``g`` of level ``l - 1`` moves to the group
   ``S`` of level ``l`` (``cnt``, ``tl``, ``ntl``, ``inside`` from :func:`_elem_rows`).
   """
   T = Ps[l, g]
   B = Ms[2][l]
   nT, nS = Ns[l, T], Ns[l, S]
   d = -_dlg(nT, -1) - _dlg(nS, 1)
   cT, cS = cnt[T], cnt[S]
   for U in range(B):
      if U == T or U == S:
         continue
      c, nU = cnt[U], Ns[l, U]
      mT, mS = _mget(Ms, l, T, U), _mget(Ms, l, S, U)
      d += _ms(nT - 1, nU, mT - c, False) - _ms(nT, nU, mT, False)
      d += _ms(nS + 1, nU, mS + c, False) - _ms(nS, nU, mS, False)
   mTT, mSS, mTS = _mget(Ms, l, T, T), _mget(Ms, l, S, S), _mget(Ms, l, T, S)
   d += _ms(nT - 1, nT - 1, mTT - cT - inside, True) - _ms(nT, nT, mTT, True)
   d += _ms(nS + 1, nS + 1, mSS + cS + inside, True) - _ms(nS, nS, mSS, True)
   d += _ms(nT - 1, nS + 1, mTS - cS + cT, False) - _ms(nT, nS, mTS, False)
   return d + _upper_delta(Ps, Ns, Ms, l, T, S, cnt, tl, ntl, inside, False, scr)


@njit(cache = True)
def _elem_apply(l, g, S, Ps, Ns, Ms, cnt, tl, ntl, inside, scr):
   """Move the group ``g`` of level ``l - 1`` to the group ``S`` of level ``l``."""
   T = Ps[l, g]
   for j in range(ntl):
      U = tl[j]
      c = cnt[U]
      if U == T:
         _madd(Ms, l, T, T, -c)
         _madd(Ms, l, S, T, c)
      elif U == S:
         _madd(Ms, l, T, S, -c)
         _madd(Ms, l, S, S, c)
      else:
         _madd(Ms, l, T, U, -c)
         _madd(Ms, l, S, U, c)
   _madd(Ms, l, T, T, -inside)
   _madd(Ms, l, S, S, inside)
   _upper_delta(Ps, Ns, Ms, l, T, S, cnt, tl, ntl, inside, True, scr)
   Ns[l, T] -= 1
   Ns[l, S] += 1
   Ps[l, g] = S


@njit(cache = True, inline = "always")
def _edge_delta(i, j, dm, N, E, k, Ps, Ns, Ms, e, hk, hv, kind, aux, lam, Bocc, qtab, lu, vt):
   """
   Change of the blockmodel description length when an edge ``(i, j)`` (``i <= j``; a self loop
   if equal) is added to the multigraph (``dm = 1``) or removed (``dm = -1``), without the
   multiplicity terms (see :func:`_mult_delta`). ``Bocc``: the numbers of occupied groups.
   """
   if lam > 0:   # the uniform multiset at the top cancels with the prior of the number of edges
      d = -dm * math.log(lam / (1 + lam))
   else:
      d = _dms(_ptop(kind, Bocc), E, dm)
   return d + _edge_delta_counts(i, j, dm, N, E, k, Ps, Ns, Ms, e, hk, hv, kind, aux, qtab, lu, vt)


@njit(cache = True, inline = "always")
def _edge_delta_counts(i, j, dm, N, E, k, Ps, Ns, Ms, e, hk, hv, kind, aux, qtab, lu, vt):
   """:func:`_edge_delta` without the terms of the number of edges at the top."""
   r, s = Ps[0, i], Ps[0, j]
   B0 = Ms[2][0]
   if kind == 3:
      ein = aux[0] + (dm if r == s else 0)
      return _pp_lik(ein, E + dm - ein, aux[1], N) - _pp_lik(aux[0], E - aux[0], aux[1], N)
   d = -_dlg(_mget(Ms, 0, r, s), dm) - (dm * _LN2 if r == s else 0.)
   if kind == 2:
      return d + dm * (math.log(Ns[0, r]) + math.log(Ns[0, s]))
   if i == j:   # self loop: degree changes by 2
      d -= _dlg(k[i], 2 * dm)
      d += _dlg(e[r], 2 * dm) + _log_q(e[r] + 2 * dm, Ns[0, r], qtab, lu, vt) - _log_q(e[r], Ns[0, r], qtab, lu, vt)
      d -= _dlg(_hget(hk, hv, k[i] * B0 + r), -1) + _dlg(_hget(hk, hv, (k[i] + 2 * dm) * B0 + r), 1)
   else:
      d -= _dlg(k[i], dm) + _dlg(k[j], dm)
      if r == s:
         d += _dlg(e[r], 2 * dm) + _log_q(e[r] + 2 * dm, Ns[0, r], qtab, lu, vt) - _log_q(e[r], Ns[0, r], qtab, lu, vt)
      else:
         d += _dlg(e[r], dm) + _dlg(e[s], dm)
         d += _log_q(e[r] + dm, Ns[0, r], qtab, lu, vt) - _log_q(e[r], Ns[0, r], qtab, lu, vt)
         d += _log_q(e[s] + dm, Ns[0, s], qtab, lu, vt) - _log_q(e[s], Ns[0, s], qtab, lu, vt)
      # degree histograms: four entries at most, possibly coinciding
      keys = (k[i] * B0 + r, (k[i] + dm) * B0 + r, k[j] * B0 + s, (k[j] + dm) * B0 + s)
      chg = (-1, 1, -1, 1)
      for x in range(4):
         seen = False
         for y in range(x):
            if keys[y] == keys[x]:
               seen = True
         if seen:
            continue
         tot = 0
         for y in range(x, 4):
            if keys[y] == keys[x]:
               tot += chg[y]
         if tot != 0:
            d -= _dlg(_hget(hk, hv, keys[x]), tot)
   Bl = Ms[2]
   L = len(Bl)
   a, b = r, s
   for l in range(1, L):
      a, b = Ps[l, a], Ps[l, b]
      na, nb = Ns[l, a], Ns[l, b]
      d += _dms(na * (na + 1) / 2. if a == b else na * float(nb), _mget(Ms, l, a, b), dm)
   return d


@njit(cache = True, inline = "always")
def _mult_delta(loop, g, dm):
   """Change of the multiplicity terms when a pair of multiplicity ``g`` changes by ``dm``."""
   if dm == 1:
      return _LN2 + math.log(g + 1.) if loop else math.log(g + 1.)
   return -_LN2 - math.log(g) if loop else -math.log(g)


@njit(cache = True, inline = "always")
def _edge_counts(i, j, dm, N, k, Ps, Ms, e, hk, hv, hn, aux):
   """
   Update the counts for an edge ``(i, j)`` added (``dm = 1``) or removed (``dm = -1``).

   :return: The (possibly reallocated) degree histogram table ``(hk, hv)``.
   """
   r, s = Ps[0, i], Ps[0, j]
   a, b = r, s
   for l in range(len(Ms[2])):
      if l > 0:
         a, b = Ps[l, a], Ps[l, b]
      _madd(Ms, l, a, b, dm)
   aux[0] += dm if r == s else 0
   B0 = Ms[2][0]
   for x, bx in ((i, r), (j, s)):
      h = _hfind(hk, k[x] * B0 + bx)
      hv[h] -= 1
      k[x] += dm
      e[bx] += dm
      hk, hv, h = _hslot(hk, hv, hn, k[x] * B0 + bx)
      hv[h] += 1
   return hk, hv


@njit(cache = True)
def _adj_del(adj, nadj, ed, x, p):
   """Remove the entry at position ``p`` of the adjacency list of ``x``."""
   a = adj[x]
   last = nadj[x] - 1
   ls = a[last]
   a[p] = ls
   for side in range(2):   # the side of ls listed last (both sides are x's for a self loop)
      if ed[ls, side] == x and ed[ls, 3 + side] == last:
         ed[ls, 3 + side] = p
         break
   nadj[x] -= 1


@njit(cache = True)
def _graph_add(u, v, N, Ps, ed, alive, nE, free, nfree, ek, evl, en, adj, nadj, eg, neg):
   """
   Add an edge ``(u, v)``, ``u <= v``, to the multigraph lists (not to the counts).

   :return: ``(ed, alive, free, nfree, ek, evl)``, possibly reallocated.
   """
   if nfree == 0:   # grow the edge arrays
      cap = len(ed)
      ned = np.full((2 * cap, 9), -1, dtype = np.int64)
      ned[:cap] = ed
      nal = np.empty(2 * cap, dtype = np.int64)
      nal[:cap] = alive
      ed, alive = ned, nal
      free = np.empty(2 * cap, dtype = np.int64)
      free[:cap] = np.arange(2 * cap - 1, cap - 1, -1)
      nfree = cap
   nfree -= 1
   sl = free[nfree]
   ed[sl, 0], ed[sl, 1], ed[sl, 2] = u, v, nE
   alive[nE] = sl
   ed[sl, 3] = _push(adj, nadj, u, sl)
   ed[sl, 4] = _push(adj, nadj, v, sl)
   ed[sl, 5] = _push(eg, neg, Ps[0, u], 2 * sl)
   ed[sl, 6] = _push(eg, neg, Ps[0, v], 2 * sl + 1)
   h = _hfind(ek, u * N + v)
   prev = evl[h] if ek[h] == u * N + v else -1
   ek, evl, h = _hslot(ek, evl, en, u * N + v)
   ed[sl, 7] = prev
   ed[sl, 8] = 1 if prev < 0 else ed[prev, 8] + 1
   evl[h] = sl
   return ed, alive, free, nfree, ek, evl


@njit(cache = True)
def _graph_remove(key, N, Ps, ed, alive, nE, free, nfree, ek, evl, adj, nadj, eg, neg):
   """
   Remove an edge of the pair ``key`` from the multigraph lists (not from the counts).

   :return: ``nfree``.
   """
   h = _hfind(ek, key)
   sl = evl[h]
   nxt = ed[sl, 7]
   if nxt < 0:
      ek[h] = -2
   else:
      evl[h] = nxt
      ed[nxt, 8] = ed[sl, 8] - 1
   last = alive[nE - 1]   # list of edges
   alive[ed[sl, 2]] = last
   ed[last, 2] = ed[sl, 2]
   for side in range(2):
      x = ed[sl, side]
      _adj_del(adj, nadj, ed, x, ed[sl, 3 + side])
      r = Ps[0, x]   # half-edges of the group of x
      g = eg[r]
      p = ed[sl, 5 + side]
      lh = g[neg[r] - 1]
      g[p] = lh
      ed[lh // 2, 5 + lh % 2] = p
      neg[r] -= 1
   ed[sl, :] = -1
   free[nfree] = sl
   return nfree + 1


@njit(cache = True)
def _dl_light(N, eu, ev, b0, par, Bs, kind, qtab, lu, vt):
   """
   Description length of a partition of a simple graph from its edge list only (no graph lists),
   as :func:`_description_length`.
   """
   Ps, Ns, Ms, k, aux = _levels(N, eu, ev, b0, par, Bs)
   e = np.zeros(Bs[0], dtype = np.int64)
   for i in range(N):
      e[b0[i]] += k[i]
   key = np.empty(N, dtype = np.int64)
   for i in range(N):
      key[i] = k[i] * Bs[0] + b0[i]
   key.sort()
   hk, hv = np.zeros(0, dtype = np.int64), np.zeros(0, dtype = np.int64)
   S = _description_length(N, len(eu), k, Ps, Ns, Ms, e, hk, hv, kind, aux, 0., 0., False, qtab, lu, vt)
   if kind != 0:
      return S
   run = 1
   for t in range(1, N + 1):   # degree histograms, from the sorted keys
      if t < N and key[t] == key[t - 1]:
         run += 1
      else:
         S -= _lg(run)
         run = 1
   return S


# ------------------------------------------------------------------------------------------------
# Measurement models
# ------------------------------------------------------------------------------------------------
# model 0: none (the network is known); 1: repeated measurements with homogeneous errors;
# 2: heterogeneous errors; 3: extrinsic probabilities. Listed pairs (keys u * N + v, u < v) carry
# n (measurements), x (positive outcomes) or q (probability); the other pairs the defaults.

@njit(cache = True, inline = "always")
def _lbeta(a, b):
   """:math:`\\ln B(a, b)`."""
   return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


@njit(cache = True, inline = "always")
def _xlog(x, p):
   """:math:`x \\ln p`, 0 for :math:`x = 0`."""
   return 0. if x == 0 else x * math.log(p)


@njit(cache = True, inline = "always")
def _measured_S(Em, Tm, Mt, Xt, h):
   """
   Minus the log-likelihood of homogeneous measurement errors (Peixoto 2018, Eqs. 15 and 19,
   without the binomial coefficients): each error rate either known (``h[4]`` for false
   negatives, ``h[5]`` for false positives) or, if NaN, integrated over its Beta prior.

   :param Em: Measurements of the pairs that are edges.
   :param Tm: Positive outcomes of the pairs that are edges.
   :param Mt: Measurements of all pairs.
   :param Xt: Positive outcomes of all pairs.
   :param h: ``(alpha, beta, mu, nu, p, q)``.
   """
   a, b, mu, nu, p, q = h[0], h[1], h[2], h[3], h[4], h[5]
   if np.isnan(p):
      S = -(_lbeta(Em - Tm + a, Tm + b) - _lbeta(a, b))
   else:
      S = -(_xlog(Em - Tm, p) + _xlog(Tm, 1 - p))
   if np.isnan(q):
      S -= _lbeta(Xt - Tm + mu, Mt - Xt - Em + Tm + nu) - _lbeta(mu, nu)
   else:
      S -= _xlog(Xt - Tm, q) + _xlog(Mt - Xt - Em + Tm, 1 - q)
   return S


@njit(cache = True, inline = "always")
def _mixed_terms(n, x, h):
   """
   Minus the log-likelihood of one pair with heterogeneous errors (Eq. 58): ``(edge, non-edge)``.
   """
   a, b, mu, nu = h[0], h[1], h[2], h[3]
   return -(_lbeta(n - x + a, x + b) - _lbeta(a, b)), -(_lbeta(x + mu, n - x + nu) - _lbeta(mu, nu))


@njit(cache = True)
def _mixed_S(cn, cx, ctot, cA, h):
   """
   Minus the log-likelihood of all pairs with heterogeneous errors, over classes of pairs with
   the same measurements: ``ctot`` pairs of each class, ``cA`` of them edges.
   """
   S = 0.
   for c in range(len(cn)):
      t1, t0 = _mixed_terms(cn[c], cx[c], h)
      S += cA[c] * t1 + (ctot[c] - cA[c]) * t0
   return S


@njit(cache = True, inline = "always")
def _pair_meas(model, idx, ln, lx, lq, lc, dn, dx, dq, dc):
   """
   Measurements of a pair: ``(n, x, q, class)``; ``idx`` is its position among the listed pairs,
   or -1 for the defaults.
   """
   if idx < 0:
      return dn, dx, dq, dc
   return ln[idx], lx[idx], lq[idx], lc[idx]


@njit(cache = True, inline = "always")
def _meas_delta(model, dm, n, x, q, c, Em, Tm, Mt, Xt, cn, cx, h):
   """
   Change of minus the log-likelihood of the measurements when a pair becomes an edge
   (``dm = 1``) or stops being one (``dm = -1``).
   """
   if model == 1:
      return _measured_S(Em + dm * n, Tm + dm * x, Mt, Xt, h) - _measured_S(Em, Tm, Mt, Xt, h)
   if model == 2:
      t1, t0 = _mixed_terms(cn[c], cx[c], h)
      return dm * (t1 - t0)
   if model == 3:
      if q <= 0. and dm == 1 or q >= 1. and dm == -1:
         return np.inf
      if q <= 0. or q >= 1.:
         return -np.inf
      return dm * (math.log1p(-q) - math.log(q))
   return 0.


# ------------------------------------------------------------------------------------------------
# Markov chain over partitions (and latent edges)
# ------------------------------------------------------------------------------------------------

@njit(cache = True, inline = "always")
def _hh(Ms, r, s):
   """Half-edges of the level 0 group ``r`` whose other end is in group ``s``."""
   return 2 * _mget(Ms, 0, r, s) if r == s else _mget(Ms, 0, r, s)


@njit(cache = True, inline = "always")
def _node_prop(i, r, s, k, Ms, e, cnt, tl, ntl, loops, eps, B0):
   """
   Probability of proposing the move of node ``i`` from group ``r`` to group ``s``: a random
   neighbor is chosen (a self loop counting twice, its group then ``r``), then, with probability
   ``eps``, a uniform group, otherwise the group at the other end of a random half-edge of the
   neighbor's group.
   """
   if k[i] == 0:
      return 1. / (B0 - 1)
   p = 0.
   for j in range(ntl):
      t = tl[j]
      p += cnt[t] / k[i] * (eps / (B0 - 1) + (1 - eps) * _hh(Ms, t, s) / e[t])
   if loops > 0:
      p += 2 * loops / k[i] * (eps / (B0 - 1) + (1 - eps) * _hh(Ms, r, s) / e[r])
   return p


@njit(cache = True, inline = "always")
def _node_prop_back(i, r, s, k, Ms, e, cnt, tl, ntl, loops, eps, B0):
   """
   Probability of proposing the move of node ``i`` back from group ``s`` to group ``r`` once moved
   to ``s``, computed from the current counts (:func:`_node_prop` after the move).
   """
   if k[i] == 0:
      return 1. / (B0 - 1)
   p = 0.
   for j in range(ntl):
      t = tl[j]
      if t == r:
         et, hh = e[r] - k[i], 2 * (_mget(Ms, 0, r, r) - cnt[r] - loops)
      elif t == s:
         et, hh = e[s] + k[i], _mget(Ms, 0, s, r) + cnt[r] - cnt[s]
      else:
         et, hh = e[t], _mget(Ms, 0, t, r) - cnt[t]
      p += cnt[t] / k[i] * (eps / (B0 - 1) + (1 - eps) * hh / et)
   if loops > 0:
      hh = _mget(Ms, 0, s, r) + cnt[r] - cnt[s]
      p += 2 * loops / k[i] * (eps / (B0 - 1) + (1 - eps) * hh / (e[s] + k[i]))
   return p


@njit(cache = True, inline = "always")
def _pair_prop(i, j, g, E, N, nlist, listed, Ps, Ns, Ms, e, w):
   """
   Probability of proposing the pair ``(i, j)``, ``i <= j``, of multiplicity ``g``, for an edge
   move, with weights ``w``: a random edge of the multigraph (``E`` edges), a uniform pair (two
   uniform nodes), a random listed pair, or a block-based pair (a random node, the group at the
   other end of a random half-edge of its group, a random member of it).
   """
   p = w[1] * (1. if i == j else 2.) / (N * float(N))
   if g > 0:
      p += w[0] * g / E
   if listed and nlist > 0:
      p += w[2] / nlist
   r, s = Ps[0, i], Ps[0, j]
   pb = 0.
   if i == j:
      if e[r] > 0:
         pb = _hh(Ms, r, r) / e[r] / Ns[0, r]
   else:
      if e[r] > 0:
         pb += _hh(Ms, r, s) / e[r] / Ns[0, s]
      if e[s] > 0:
         pb += _hh(Ms, s, r) / e[s] / Ns[0, r]
   return p + w[3] * pb / N


@njit(cache = True, inline = "always")
def _softplus(x):
   """:math:`\\ln(1 + e^x)`, without overflow."""
   return max(x, 0.) + math.log1p(math.exp(-abs(x)))


@njit(cache = True)
def _ndelta(i, s, N, E, k, Ps, Ns, Ms, e, hk, hv, cnt, tl, kind, aux, scr, ed, adj, nadj, qtab, lu, vt):
   """Change of the description length (but for the group-count terms) if node ``i`` moves to ``s``."""
   ntl, loops = _node_counts(i, Ps, adj, nadj, ed, cnt, tl)
   d = _node_delta(i, s, N, E, k, Ps, Ns, Ms, e, hk, hv, cnt, tl, ntl, loops, kind, aux, scr, qtab, lu, vt)
   cnt[tl[:ntl]] = 0
   return d


@njit(cache = True)
def _nmove(i, s, N, k, Ps, Ns, Ms, e, hk, hv, hn, cnt, tl, aux, scr, ed, adj, nadj, memb, nmemb, mpos, eg, neg, Bocc, L):
   """Move node ``i`` to group ``s``, keeping the occupancy of level 0 and the sizes above."""
   r = Ps[0, i]
   ntl, loops = _node_counts(i, Ps, adj, nadj, ed, cnt, tl)
   hk, hv = _node_apply(i, s, N, k, Ps, Ns, Ms, e, hk, hv, hn, cnt, tl, ntl, loops, aux, scr, ed, adj, nadj, memb, nmemb, mpos, eg, neg)
   cnt[tl[:ntl]] = 0
   if Ns[0, r] == 0:
      Bocc[0] -= 1
      if L > 1:
         Ns[1, Ps[1, r]] -= 1
   if Ns[0, s] == 1:
      Bocc[0] += 1
      if L > 1:
         Ns[1, Ps[1, s]] += 1
   return hk, hv


@njit(cache = True)
def _size_part(l, T, Ns, Ms):
   """Terms of level ``l >= 1`` that involve the size of its group ``T``."""
   n = Ns[l, T]
   S = -_lg(n) + _ms(n, n, _mget(Ms, l, T, T), True)
   for u in range(Ms[2][l]):
      if u != T and Ns[l, u] > 0:
         S += _ms(n, Ns[l, u], _mget(Ms, l, T, u), False)
   return S


@njit(cache = True)
def _merge_split(N, E, k, Ps, Ns, Ms, e, hk, hv, hn, cnt, tl, aux, scr, ed, adj, nadj, memb, nmemb, mpos, eg, neg, Bocc, Bs, kind, lam,
                 nodes, orig, sweeps, qtab, lu, vt):
   """
   Split-merge move of the groups of level 0 (Jain and Neal 2004; Peixoto 2020): with probability
   1/2, a random group is split into itself and a new group, its nodes assigned at random and
   then reassigned by ``sweeps`` restricted Gibbs sweeps between the two, plus a last sweep whose
   probability is the proposal probability; otherwise two random groups (under the same group
   above) are merged, the probability of the reverse split computed by the same procedure from
   the merged state, its last sweep forced to the original groups. Accepted with the
   Metropolis-Hastings criterion.

   :return: ``(hk, hv, dS)``: the (possibly reallocated) degree histograms and the change of the
      description length (0 if rejected).
   """
   L, C0, B = len(Bs), Bs[0], Bocc[0]
   if np.random.random() < 0.5:   # split s, the new group r
      if B == C0:
         return hk, hv, 0.
      s = np.random.randint(C0)
      while Ns[0, s] == 0:
         s = np.random.randint(C0)
      if Ns[0, s] < 2:
         return hk, hv, 0.
      r = np.random.randint(C0)
      while Ns[0, r] > 0:
         r = np.random.randint(C0)
      if L > 1:
         Ps[1, r] = Ps[1, s]
      n = nmemb[s]
      nodes[:n] = memb[s][:n]
      nodes[:n].sort()
      P = Ps[1, s] if L > 1 else 0
      U0 = _bglobal(N, E, Bocc, -1, 0, Bs, kind, lam, True) + (_size_part(1, P, Ns, Ms) if L > 1 else 0.)
      merged = 0
   else:   # merge r into s
      if B < 2:
         return hk, hv, 0.
      r = np.random.randint(C0)
      while Ns[0, r] == 0:
         r = np.random.randint(C0)
      s = np.random.randint(C0)
      while Ns[0, s] == 0 or s == r:
         s = np.random.randint(C0)
      if L > 1 and Ps[1, r] != Ps[1, s]:
         return hk, hv, 0.
      n = 0
      for g in (r, s):
         for x in range(nmemb[g]):
            nodes[n] = memb[g][x]
            n += 1
      nodes[:n].sort()
      for x in range(n):
         orig[x] = Ps[0, nodes[x]]
      P = Ps[1, s] if L > 1 else 0
      U0 = _bglobal(N, E, Bocc, -1, 0, Bs, kind, lam, True) + (_size_part(1, P, Ns, Ms) if L > 1 else 0.)
      d1 = 0.
      for x in range(n):
         if orig[x] == r:
            d1 += _ndelta(nodes[x], s, N, E, k, Ps, Ns, Ms, e, hk, hv, cnt, tl, kind, aux, scr, ed, adj, nadj, qtab, lu, vt)
            hk, hv = _nmove(nodes[x], s, N, k, Ps, Ns, Ms, e, hk, hv, hn, cnt, tl, aux, scr, ed, adj, nadj, memb, nmemb, mpos, eg, neg, Bocc, L)
      dS = d1 + _bglobal(N, E, Bocc, -1, 0, Bs, kind, lam, True) + (_size_part(1, P, Ns, Ms) if L > 1 else 0.) - U0
      merged = 1
   # launch: random assignment and restricted Gibbs sweeps between r and s, then the last sweep
   dsum, lq = 0., 0.
   for t in range(sweeps + 2):
      for x in range(n):
         i = nodes[x]
         cur = Ps[0, i]
         oth = s if cur == r else r
         if t == 0:   # random assignment
            go = np.random.random() < 0.5
         else:
            d = _ndelta(i, oth, N, E, k, Ps, Ns, Ms, e, hk, hv, cnt, tl, kind, aux, scr, ed, adj, nadj, qtab, lu, vt)
            lgo, lstay = -_softplus(d), -_softplus(-d)   # log-probabilities of moving and staying
            if t == sweeps + 1 and merged == 1:   # last sweep forced to the original groups
               go = orig[x] != cur
            else:
               go = np.random.random() < math.exp(lgo)
            if t == sweeps + 1:
               lq += lgo if go else lstay
         if go:
            if t > 0:
               dsum += d
            else:
               dsum += _ndelta(i, oth, N, E, k, Ps, Ns, Ms, e, hk, hv, cnt, tl, kind, aux, scr, ed, adj, nadj, qtab, lu, vt)
            hk, hv = _nmove(i, oth, N, k, Ps, Ns, Ms, e, hk, hv, hn, cnt, tl, aux, scr, ed, adj, nadj, memb, nmemb, mpos, eg, neg, Bocc, L)
   if merged == 1:   # back at the original groups: accept the merge?
      lr = -math.log(B - 1.) - math.log(C0 - B + 1.) + lq + math.log(B) + math.log(B - 1.)
      if np.random.random() < math.exp(min(0., -dS + lr)):
         for x in range(n):
            if orig[x] == r:
               hk, hv = _nmove(nodes[x], s, N, k, Ps, Ns, Ms, e, hk, hv, hn, cnt, tl, aux, scr, ed, adj, nadj, memb, nmemb, mpos, eg, neg, Bocc, L)
         return hk, hv, dS
      return hk, hv, 0.
   ok = Ns[0, r] > 0 and Ns[0, s] > 0
   if ok:
      dS = dsum + _bglobal(N, E, Bocc, -1, 0, Bs, kind, lam, True) + (_size_part(1, P, Ns, Ms) if L > 1 else 0.) - U0
      lr = -math.log(B + 1.) - math.log(B) + math.log(B) + math.log(C0 - B) - lq
      ok = np.random.random() < math.exp(min(0., -dS + lr))
   if ok:
      return hk, hv, dS
   for x in range(n):   # rejected: back to s
      if Ps[0, nodes[x]] != s:
         hk, hv = _nmove(nodes[x], s, N, k, Ps, Ns, Ms, e, hk, hv, hn, cnt, tl, aux, scr, ed, adj, nadj, memb, nmemb, mpos, eg, neg, Bocc, L)
   return hk, hv, 0.


@njit(cache = True, nogil = True)
def _chain(N, eu, ev, b0, par, Bs, kind, pmoves, dnew, lam, model, lkeys, ln, lx, lq, lc, dn, dx, dq, dc, cn, cx, ctot, h0, hfix,
           beta, sweeps, burnin, interval, eps, wprop, seed, qtab, lu, vt):
   """
   Markov chain Monte Carlo over the partition (if ``pmoves``) and, with a measurement model, over
   the latent multigraph, at inverse temperature ``beta`` (infinite: greedy descent).

   Each sweep tries ``N`` node moves, as many moves of the groups of every upper level as it has
   groups, and, with a measurement model, ``max(E, N)`` moves of the multigraph (``E`` edges of
   the collapsed network), plus hyperparameter moves for model 2. A multigraph move picks a pair,
   then adds an edge to it or, if it has some, adds or removes one with probability 1/2 each
   (Peixoto 2018, Eq. D8). With ``dnew > 0``, the numbers of groups change too: ``Bs`` are then
   the numbers of labels of every level, empty groups included, a move goes to a new group with
   probability ``dnew`` (Eq. B4; the new group takes the parent of the group left), and a group can
   only be emptied into a group with the same parent, so that every move can be reversed. After
   ``burnin`` sweeps, the edges of the collapsed network are counted every ``interval`` sweeps.

   :return: ``(b0, par, keys, counts, samples, trace, eu, ev, S)``: final partitions, the pairs
      ever counted as edges and how many times, the number of samples, per sample the description
      length, the number of edges of the collapsed network, the model statistics (model 1:
      measurements and positive outcomes of the edges; model 2: the four hyperparameters) and the
      number of groups of level 0 (column 6), the final collapsed network and its description
      length (with the label term of :func:`_bglobal` if ``dnew > 0``).
   """
   np.random.seed(seed)
   st = _build(N, eu, ev, b0, par, Bs, 2 * len(eu) + 1024)
   Ps, Ns, Ms, k, e, hk, hv, hn, ed, alive, free, nfree, ek, evl, en, adj, nadj, memb, nmemb, mpos, eg, neg, aux = st
   L = len(Bs)
   EG = len(eu)   # edges of the multigraph
   EA = EG        # edges of the collapsed network
   greedy = not np.isfinite(beta)
   # listed pairs
   nlist = len(lkeys)
   lk, lv = _hrehash(np.full(16, -1, dtype = np.int64), np.zeros(16, dtype = np.int64), nlist + 16)
   for t in range(nlist):
      hh = _hfind(lk, lkeys[t])
      lk[hh], lv[hh] = lkeys[t], t
   h = h0.copy()
   Em, Tm, Mt, Xt = 0, 0, 0, 0
   cA = np.zeros(len(cn), dtype = np.int64)
   if model == 1 or model == 2:
      Mt = dn * (N * (N - 1) // 2 - nlist) + np.sum(ln)
      Xt = dx * (N * (N - 1) // 2 - nlist) + np.sum(lx)
   for t in range(EG):
      hh = _hfind(lk, eu[t] * N + ev[t])
      idx = lv[hh] if lk[hh] == eu[t] * N + ev[t] else -1
      n, x, q, c = _pair_meas(model, idx, ln, lx, lq, lc, dn, dx, dq, dc)
      Em += n
      Tm += x
      if model == 2:
         cA[c] += 1
   vary = dnew > 0
   msn, mso = np.empty(N, dtype = np.int64), np.empty(N, dtype = np.int64)
   MS_TRIES = 3
   Bocc = _occupied(Ns, Ms)
   S = _description_length(N, EG, k, Ps, Ns, Ms, e, hk, hv, kind, aux, 0., lam, vary, qtab, lu, vt)
   if model == 1:
      S += _measured_S(Em, Tm, Mt, Xt, h)
   elif model == 2:
      S += _mixed_S(cn, cx, ctot, cA, h)
   elif model == 3:   # minus the log-probability of every pair being an edge or not (0 for certain ones)
      ed_listed = 0
      for t in range(EG):
         hh = _hfind(lk, eu[t] * N + ev[t])
         q = lq[lv[hh]] if lk[hh] == eu[t] * N + ev[t] else dq
         ed_listed += lk[hh] == eu[t] * N + ev[t]
         S -= math.log(q)
      for t in range(nlist):
         if ek[_hfind(ek, lkeys[t])] != lkeys[t]:
            S -= math.log1p(-lq[t])
      nd = N * (N - 1) // 2 - nlist - (EG - ed_listed)   # default pairs that are not edges
      if nd > 0 and dq > 0:
         S -= nd * math.log1p(-dq)
   Bmax = Bs[0]
   cnt, tl = np.zeros(Bmax, dtype = np.int64), np.zeros(Bmax, dtype = np.int64)
   scr = (np.zeros(Bmax + 2, dtype = np.int64), np.zeros(Bmax + 2, dtype = np.int64), np.zeros(Bmax + 2, dtype = np.int64), np.zeros(Bmax + 2, dtype = np.int64))
   mk, mv = np.full(1024, -1, dtype = np.int64), np.zeros(1024, dtype = np.int64)
   mn = np.zeros(1, dtype = np.int64)
   nsamp = (sweeps - burnin) // interval if sweeps > burnin else 0
   trace = np.zeros((max(nsamp, 1), 7))
   samples = 0
   C0 = Bs[0]
   for sweep in range(sweeps):
      # node moves; with vary, to a new group with probability dnew (Peixoto 2018, Eq. B4), the
      # new group taking the parent of the node's group, and a group emptied only into a group
      # with the same parent (so that every move can be reversed)
      if pmoves and (Bocc[0] > 1 or vary):
         for _ in range(N):
            i = np.random.randint(N)
            r = Ps[0, i]
            nr = Ns[0, r]
            new = vary and np.random.random() < dnew
            if new:
               if nr == 1 or Bocc[0] == C0:
                  continue
               s = np.random.randint(C0)
               while Ns[0, s] > 0:
                  s = np.random.randint(C0)
               if L > 1:
                  Ps[1, s] = Ps[1, r]
            else:
               if (nr == 1 and not vary) or Bocc[0] == 1:
                  continue
               if k[i] == 0 or np.random.random() < eps:
                  s = np.random.randint(C0)
                  while Ns[0, s] == 0 or s == r:
                     s = np.random.randint(C0)
               else:
                  j = adj[i][np.random.randint(nadj[i])]
                  t = Ps[0, ed[j, 1] if ed[j, 0] == i else ed[j, 0]]
                  he = eg[t][np.random.randint(neg[t])]
                  s = Ps[0, ed[he // 2, 1 - he % 2]]
               if s == r or (nr == 1 and L > 1 and Ps[1, s] != Ps[1, r]):
                  continue
            dB = 1 if new else (-1 if nr == 1 else 0)
            ntl, loops = _node_counts(i, Ps, adj, nadj, ed, cnt, tl)
            d = _node_delta(i, s, N, EG, k, Ps, Ns, Ms, e, hk, hv, cnt, tl, ntl, loops, kind, aux, scr, qtab, lu, vt)
            if dB != 0:
               if L > 1:
                  d += _size_delta(1, Ps[1, r], dB, Ns, Ms)
               d += _bglobal(N, EG, Bocc, 0, dB, Bs, kind, lam, vary) - _bglobal(N, EG, Bocc, -1, 0, Bs, kind, lam, vary)
            if greedy:
               ok = d < -1e-9
            else:
               B = Bocc[0]
               if dB == 1:
                  lhr = math.log((1 - dnew) * _node_prop_back(i, r, s, k, Ms, e, cnt, tl, ntl, loops, eps, B + 1) * (C0 - B) / dnew)
               elif dB == -1:
                  lhr = math.log(dnew / (C0 - B + 1) / ((1 - dnew) * _node_prop(i, r, s, k, Ms, e, cnt, tl, ntl, loops, eps, B)))
               else:
                  lhr = math.log(_node_prop_back(i, r, s, k, Ms, e, cnt, tl, ntl, loops, eps, B) / _node_prop(i, r, s, k, Ms, e, cnt, tl, ntl, loops, eps, B))
               ok = np.random.random() < math.exp(min(0., -beta * d + lhr))
            if ok:
               hk, hv = _node_apply(i, s, N, k, Ps, Ns, Ms, e, hk, hv, hn, cnt, tl, ntl, loops, aux, scr, ed, adj, nadj, memb, nmemb, mpos, eg, neg)
               if dB != 0:
                  if L > 1:
                     Ns[1, Ps[1, r]] += dB
                  Bocc[0] += dB
               S += d
            cnt[tl[:ntl]] = 0
      # split-merge moves of the groups of level 0, for mixing over the number of groups
      if pmoves and vary and not greedy:
         for _ in range(MS_TRIES):
            hk, hv, dS = _merge_split(N, EG, k, Ps, Ns, Ms, e, hk, hv, hn, cnt, tl, aux, scr, ed, adj, nadj, memb, nmemb, mpos, eg, neg, Bocc, Bs, kind, lam,
                                      msn, mso, 4, qtab, lu, vt)
            S += dS
      # moves of the groups of the upper levels, likewise
      for l in range(1, L if pmoves else 1):
         Cl, Cp = Bs[l], Bs[l - 1]
         for _ in range(Cp):
            g = np.random.randint(Cp)
            if Ns[l - 1, g] == 0:
               continue
            T = Ps[l, g]
            nT = Ns[l, T]
            new = vary and np.random.random() < dnew
            if new:
               if nT == 1 or Bocc[l] == Cl:
                  continue
               Sg = np.random.randint(Cl)
               while Ns[l, Sg] > 0:
                  Sg = np.random.randint(Cl)
               if l + 1 < L:
                  Ps[l + 1, Sg] = Ps[l + 1, T]
            else:
               if (nT == 1 and not vary) or Bocc[l] < 2:
                  continue
               Sg = np.random.randint(Cl)
               while Ns[l, Sg] == 0 or Sg == T:
                  Sg = np.random.randint(Cl)
               if nT == 1 and l + 1 < L and Ps[l + 1, Sg] != Ps[l + 1, T]:
                  continue
            dB = 1 if new else (-1 if nT == 1 else 0)
            ntl, inside = _elem_rows(l, g, Ps, Ms, cnt, tl)
            d = _elem_delta(l, g, Sg, Ps, Ns, Ms, cnt, tl, ntl, inside, scr)
            lhr = 0.
            if dB != 0:
               if l + 1 < L:
                  d += _size_delta(l + 1, Ps[l + 1, T], dB, Ns, Ms)
               d += _bglobal(N, EG, Bocc, l, dB, Bs, kind, lam, vary) - _bglobal(N, EG, Bocc, -1, 0, Bs, kind, lam, vary)
               B = Bocc[l]
               lhr = math.log((1 - dnew) / B * (Cl - B) / dnew) if dB == 1 else math.log(dnew / (Cl - B + 1) * (B - 1) / (1 - dnew))
            if (d < -1e-9) if greedy else (np.random.random() < math.exp(min(0., -beta * d + lhr))):
               _elem_apply(l, g, Sg, Ps, Ns, Ms, cnt, tl, ntl, inside, scr)
               if dB != 0:
                  if l + 1 < L:
                     Ns[l + 1, Ps[l + 1, T]] += dB
                  Bocc[l] += dB
               S += d
            cnt[tl[:ntl]] = 0
      # moves of the multigraph
      if model > 0:
         for _ in range(max(EA, N)):
            u = np.random.random()
            if u < wprop[0]:
               if EG == 0:
                  continue
               sl = alive[np.random.randint(EG)]
               i, j = ed[sl, 0], ed[sl, 1]
            elif u < wprop[0] + wprop[1]:
               i, j = np.random.randint(N), np.random.randint(N)
            elif u < wprop[0] + wprop[1] + wprop[2]:
               if nlist == 0:
                  continue
               key = lkeys[np.random.randint(nlist)]
               i, j = key // N, key % N
            else:
               i = np.random.randint(N)
               r = Ps[0, i]
               if neg[r] == 0:
                  continue
               he = eg[r][np.random.randint(neg[r])]
               s = Ps[0, ed[he // 2, 1 - he % 2]]
               j = memb[s][np.random.randint(nmemb[s])]
            if i > j:
               i, j = j, i
            key = i * N + j
            g = _mult(ek, evl, ed, key)
            dm = 1 if g == 0 or np.random.random() < 0.5 else -1
            dA = dm if i != j and g + (dm == 1) == 1 else 0   # the collapsed network changes
            idx = -1
            if i != j:
               hl = _hfind(lk, key)
               idx = lv[hl] if lk[hl] == key else -1
            n, x, q, c = _pair_meas(model, idx, ln, lx, lq, lc, dn, dx, dq, dc)
            dmeas = _meas_delta(model, dA, n, x, q, c, Em, Tm, Mt, Xt, cn, cx, h) if dA != 0 else 0.
            if dmeas == np.inf:
               continue
            d = _edge_delta(i, j, dm, N, EG, k, Ps, Ns, Ms, e, hk, hv, kind, aux, lam, Bocc, qtab, lu, vt) + _mult_delta(i == j, g, dm) + dmeas
            if greedy:
               if not d < -1e-9:
                  continue
            else:
               pf = _pair_prop(i, j, g, EG, N, nlist, idx >= 0, Ps, Ns, Ms, e, wprop) * (0.5 if g > 0 else 1.)
               # reverse proposal, from the counts after the move
               r, s = Ps[0, i], Ps[0, j]
               _madd(Ms, 0, r, s, dm)
               e[r] += dm
               e[s] += dm
               pr = _pair_prop(i, j, g + dm, EG + dm, N, nlist, idx >= 0, Ps, Ns, Ms, e, wprop) * (0.5 if g + dm > 0 else 1.)
               _madd(Ms, 0, r, s, -dm)
               e[r] -= dm
               e[s] -= dm
               if not np.random.random() < math.exp(min(0., -beta * d + math.log(pr / pf))):
                  continue
            hk, hv = _edge_counts(i, j, dm, N, k, Ps, Ms, e, hk, hv, hn, aux)
            if dm == 1:
               ed, alive, free, nfree, ek, evl = _graph_add(i, j, N, Ps, ed, alive, EG, free, nfree, ek, evl, en, adj, nadj, eg, neg)
            else:
               nfree = _graph_remove(key, N, Ps, ed, alive, EG, free, nfree, ek, evl, adj, nadj, eg, neg)
            EG += dm
            EA += dA
            S += d
            Em += dA * n
            Tm += dA * x
            if model == 2:
               cA[c] += dA
         # hyperparameters of the heterogeneous model: log-scale random walk, flat prior on [0.01, 1000]
         if model == 2 and not hfix:
            for _ in range(10):
               for p in range(4):
                  old = h[p]
                  new = old * math.exp(0.2 * np.random.standard_normal())
                  if new < 0.01 or new > 1000.:
                     continue
                  s0 = _mixed_S(cn, cx, ctot, cA, h)
                  h[p] = new
                  d = _mixed_S(cn, cx, ctot, cA, h) - s0
                  if (d < -1e-9) if greedy else (np.random.random() < math.exp(min(0., -beta * d + math.log(new / old)))):
                     S += d
                  else:
                     h[p] = old
      # samples: the edges of the collapsed network
      if sweep >= burnin and (sweep - burnin) % interval == interval - 1 and samples < nsamp:
         for t in range(len(ek)):
            if ek[t] >= 0 and ek[t] // N != ek[t] % N:
               mk, mv, hh = _hslot(mk, mv, mn, ek[t])
               mv[hh] += 1
         lab = 0.   # the label term, not part of the description length
         for l in range(L):
            lab -= (_lg(Bocc[l]) + _lg(Bs[l] - Bocc[l])) if vary else 0.
         trace[samples, 0], trace[samples, 1], trace[samples, 6] = S - lab, EA, Bocc[0]
         if model == 1:
            trace[samples, 2], trace[samples, 3] = Em, Tm
         elif model == 2:
            trace[samples, 2:6] = h[:4]
         samples += 1
   out_par = np.empty(np.sum(Bs[:L - 1]), dtype = np.int64)
   at = 0
   for l in range(1, L):
      out_par[at:at + Bs[l - 1]] = Ps[l, :Bs[l - 1]]
      at += Bs[l - 1]
   sel = mk >= 0
   eu2 = np.empty(EA, dtype = np.int64)
   ev2 = np.empty(EA, dtype = np.int64)
   at = 0
   for t in range(len(ek)):
      if ek[t] >= 0 and ek[t] // N != ek[t] % N:
         eu2[at], ev2[at] = ek[t] // N, ek[t] % N
         at += 1
   return Ps[0, :N].copy(), out_par, mk[sel], mv[sel], samples, trace[:samples], eu2, ev2, S


# ------------------------------------------------------------------------------------------------
# Fit: agglomerative heuristic at level 0 (sparse block tables, parallel merge evaluation)
# ------------------------------------------------------------------------------------------------

@njit(cache = True)
def _l0_tables(N, eu, ev, b, B, k):
   """
   Static tables of a level 0 partition with ``B`` groups: for every group, the groups at the
   other end of its half-edges (``hptr``, ``ht``, sorted), its distinct neighbor groups with the
   number of half-edges to each (``rptr``, ``rcol``, ``rval``; within a group, twice the edges),
   its distinct degrees with their counts (``dptr``, ``dk``, ``dc``), sizes ``n`` and degree sums
   ``e``.
   """
   m = len(eu)
   key = np.empty(2 * m, dtype = np.int64)
   for t in range(m):
      key[2 * t] = b[eu[t]] * B + b[ev[t]]
      key[2 * t + 1] = b[ev[t]] * B + b[eu[t]]
   key.sort()
   hptr = np.zeros(B + 1, dtype = np.int64)
   ht = np.empty(2 * m, dtype = np.int64)
   for t in range(2 * m):
      hptr[key[t] // B + 1] += 1
      ht[t] = key[t] % B
   hptr = np.cumsum(hptr)
   nu = 0
   for t in range(2 * m):
      if t == 0 or key[t] != key[t - 1]:
         nu += 1
   rptr = np.zeros(B + 1, dtype = np.int64)
   rcol, rval = np.empty(nu, dtype = np.int64), np.zeros(nu, dtype = np.int64)
   p = -1
   for t in range(2 * m):
      if t == 0 or key[t] != key[t - 1]:
         p += 1
         rcol[p] = key[t] % B
         rptr[key[t] // B + 1] += 1
      rval[p] += 1
   rptr = np.cumsum(rptr)
   kmax = np.max(k) + 1 if N > 0 else 1
   dkey = np.empty(N, dtype = np.int64)
   for i in range(N):
      dkey[i] = b[i] * kmax + k[i]
   dkey.sort()
   nd = 0
   for t in range(N):
      if t == 0 or dkey[t] != dkey[t - 1]:
         nd += 1
   dptr = np.zeros(B + 1, dtype = np.int64)
   dk, dc = np.empty(nd, dtype = np.int64), np.zeros(nd, dtype = np.int64)
   p = -1
   for t in range(N):
      if t == 0 or dkey[t] != dkey[t - 1]:
         p += 1
         dk[p] = dkey[t] % kmax
         dptr[dkey[t] // kmax + 1] += 1
      dc[p] += 1
   dptr = np.cumsum(dptr)
   n = np.zeros(B, dtype = np.int64)
   e = np.zeros(B, dtype = np.int64)
   for i in range(N):
      n[b[i]] += 1
      e[b[i]] += k[i]
   return hptr, ht, rptr, rcol, rval, dptr, dk, dc, n, e


@njit(cache = True)
def _row_get(ptr, col, val, r, t):
   """Value of column ``t`` in the sorted sparse row ``r`` (0 if absent)."""
   lo, hi = ptr[r], ptr[r + 1]
   p = lo + np.searchsorted(col[lo:hi], t)
   return val[p] if p < hi and col[p] == t else 0


@njit(cache = True)
def _l0_merge_delta(r, s, B, N, E, tabs, qtab, lu, vt):
   """
   Change of the flat description length (level 0 with a single group above) when the level 0
   group ``r`` merges into ``s``.
   """
   hptr, ht, rptr, rcol, rval, dptr, dk, dc, n, e = tabs
   d = 0.
   for p in range(rptr[r], rptr[r + 1]):
      t = rcol[p]
      if t == r or t == s:
         continue
      c = rval[p]
      mst = _row_get(rptr, rcol, rval, s, t)
      d += _adj(mst + c, False) - _adj(mst, False) - _adj(c, False)
   mrs = _row_get(rptr, rcol, rval, r, s)
   mrr = _row_get(rptr, rcol, rval, r, r) // 2
   mss = _row_get(rptr, rcol, rval, s, s) // 2
   d += _adj(mss + mrr + mrs, True) - _adj(mss, True) - _adj(mrr, True) - _adj(mrs, False)
   d += _lg(e[r] + e[s]) - _lg(e[r]) - _lg(e[s])
   d += _log_q(e[r] + e[s], n[r] + n[s], qtab, lu, vt) - _log_q(e[r], n[r], qtab, lu, vt) - _log_q(e[s], n[s], qtab, lu, vt)
   for p in range(dptr[r], dptr[r + 1]):
      cs = _row_get(dptr, dk, dc, s, dk[p])
      d += -_lg(cs + dc[p]) + _lg(cs) + _lg(dc[p])
   d += _lbinom(N - 1, B - 2) - _lbinom(N - 1, B - 1)
   d += _ms(B - 1, B - 1, E, True) - _ms(B, B, E, True) + math.log(B - 1) - math.log(B)
   return d


@njit(cache = True, parallel = True)
def _l0_merge_eval(B, N, E, tabs, rnd, eps, qtab, lu, vt):
   """
   Best of ``rnd.shape[1]`` proposed merges of every level 0 group, evaluated in parallel: a
   uniform group with probability ``eps``, otherwise the group at the other end of a random
   half-edge of the group at the other end of a random half-edge of ``r``.

   :param rnd: Uniform random numbers, shape ``(B, candidates, 3)``.
   :return: ``(target, delta)`` per group (target -1 if none).
   """
   hptr, ht = tabs[0], tabs[1]
   e = tabs[9]
   best = np.full(B, -1, dtype = np.int64)
   bd = np.full(B, np.inf)
   for r in prange(B):
      for c in range(rnd.shape[1]):
         if e[r] == 0 or rnd[r, c, 0] < eps:
            s = min(int(rnd[r, c, 1] * B), B - 1)
         else:
            t = ht[hptr[r] + min(int(rnd[r, c, 1] * e[r]), e[r] - 1)]
            s = ht[hptr[t] + min(int(rnd[r, c, 2] * e[t]), e[t] - 1)]
         if s == r:
            continue
         d = _l0_merge_delta(r, s, B, N, E, tabs, qtab, lu, vt)
         if d < bd[r]:
            bd[r], best[r] = d, s
   return best, bd


@njit(cache = True)
def _flat_sweeps(N, ptr, nbr, b, B, k, sweeps, tries, eps, seed, qtab, lu, vt):
   """
   Greedy node moves at level 0 with a single group above, on sparse tables (any number of
   groups): each node tries ``tries`` groups (of a random neighbor, of a random neighbor of a
   random neighbor, or uniform with probability ``eps``) and moves to the best one that lowers
   the description length. Groups are not emptied. Stops early when no node moves.

   :param ptr: CSR adjacency of the graph (``nbr`` sorted per node).
   :return: The new partition.
   """
   np.random.seed(seed)
   b = b.copy()
   n = np.zeros(B, dtype = np.int64)
   e = np.zeros(B, dtype = np.int64)
   for i in range(N):
      n[b[i]] += 1
      e[b[i]] += k[i]
   m = ptr[N] // 2
   mk, mv = _hrehash(np.full(16, -1, dtype = np.int64), np.zeros(16, dtype = np.int64), 2 * m + 16)
   mu = np.zeros(1, dtype = np.int64)
   for i in range(N):
      for p in range(ptr[i], ptr[i + 1]):
         j = nbr[p]
         if i < j:
            r, t = min(b[i], b[j]), max(b[i], b[j])
            mk, mv, h = _hslot(mk, mv, mu, r * B + t)
            mv[h] += 1
   kmax = np.max(k) + 1
   hk, hv = _hrehash(np.full(16, -1, dtype = np.int64), np.zeros(16, dtype = np.int64), N + 16)
   hu = np.zeros(1, dtype = np.int64)
   for i in range(N):
      hk, hv, h = _hslot(hk, hv, hu, b[i] * kmax + k[i])
      hv[h] += 1
   cnt, tl = np.zeros(B, dtype = np.int64), np.zeros(B, dtype = np.int64)
   for sweep in range(sweeps):
      moves = 0
      for _ in range(N):
         i = np.random.randint(N)
         r = b[i]
         if n[r] == 1 or k[i] == 0:
            continue
         ntl = 0
         for p in range(ptr[i], ptr[i + 1]):
            t = b[nbr[p]]
            if cnt[t] == 0:
               tl[ntl] = t
               ntl += 1
            cnt[t] += 1
         ki = k[i]
         bs, bd = -1, -1e-9
         for c in range(tries):
            if np.random.random() < eps:
               s = np.random.randint(B)
            else:
               j = nbr[ptr[i] + np.random.randint(ki)]
               if c % 2 == 1 and k[j] > 0:
                  j = nbr[ptr[j] + np.random.randint(k[j])]
               s = b[j]
            if s == r:
               continue
            cr, cs = cnt[r], cnt[s]
            d = 0.
            for q in range(ntl):
               t = tl[q]
               if t == r or t == s:
                  continue
               ct = cnt[t]
               mrt = _hget(mk, mv, min(r, t) * B + max(r, t))
               mst = _hget(mk, mv, min(s, t) * B + max(s, t))
               d += _adj(mrt - ct, False) - _adj(mrt, False) + _adj(mst + ct, False) - _adj(mst, False)
            mrs = _hget(mk, mv, min(r, s) * B + max(r, s))
            mrr, mss = _hget(mk, mv, r * B + r), _hget(mk, mv, s * B + s)
            d += _adj(mrs - cs + cr, False) - _adj(mrs, False)
            d += _adj(mrr - cr, True) - _adj(mrr, True) + _adj(mss + cs, True) - _adj(mss, True)
            d += _lg(e[r] - ki) - _lg(e[r]) + _lg(e[s] + ki) - _lg(e[s])
            d += _log_q(e[r] - ki, n[r] - 1, qtab, lu, vt) - _log_q(e[r], n[r], qtab, lu, vt)
            d += _log_q(e[s] + ki, n[s] + 1, qtab, lu, vt) - _log_q(e[s], n[s], qtab, lu, vt)
            hr, hs = _hget(hk, hv, r * kmax + ki), _hget(hk, hv, s * kmax + ki)
            d += -_lg(hr - 1) + _lg(hr) - _lg(hs + 1) + _lg(hs)
            if d < bd:
               bd, bs = d, s
         if bs >= 0:
            s = bs
            for q in range(ntl):   # edges to t move from the pair (r, t) to (s, t), t = r and t = s included
               t = tl[q]
               ct = cnt[t]
               for x, y, dx in ((r, t, -ct), (s, t, ct)):
                  mk, mv, h = _hslot(mk, mv, mu, min(x, y) * B + max(x, y))
                  mv[h] += dx
            e[r] -= ki
            e[s] += ki
            n[r] -= 1
            n[s] += 1
            hk, hv, h = _hslot(hk, hv, hu, r * kmax + ki)
            hv[h] -= 1
            hk, hv, h = _hslot(hk, hv, hu, s * kmax + ki)
            hv[h] += 1
            b[i] = s
            moves += 1
         for q in range(ntl):
            cnt[tl[q]] = 0
      if moves <= N // 1000:
         break
   return b


@njit(cache = True)
def _apply_merges(b, B, best, bd, target):
   """
   Merge groups in order of increasing description length change until ``target`` groups are
   left (a group merges at most once per round; chains follow the merged groups).

   :return: ``(b, B)`` with labels 0 .. B - 1.
   """
   parent = np.arange(B)
   order = np.argsort(bd)
   merged = 0
   moved = np.zeros(B, dtype = np.bool_)
   for o in order:
      if merged >= B - target or best[o] < 0 or not np.isfinite(bd[o]):
         break
      if moved[o]:
         continue
      s = best[o]
      while parent[s] != s:
         s = parent[s]
      if s == o:
         continue
      parent[o] = s
      moved[o] = True
      merged += 1
   for x in range(B):   # compress
      y = x
      while parent[y] != y:
         y = parent[y]
      parent[x] = y
   lab = np.full(B, -1, dtype = np.int64)
   nb = 0
   for x in range(B):
      if parent[x] == x:
         lab[x] = nb
         nb += 1
   out = np.empty(len(b), dtype = np.int64)
   for i in range(len(b)):
      out[i] = lab[parent[b[i]]]
   return out, nb


# ------------------------------------------------------------------------------------------------
# Fit: upper levels (dense), built level by level
# ------------------------------------------------------------------------------------------------

@njit(cache = True)
def _aggregate(W, g, B):
   """Edges between groups (``g`` maps elements to ``B`` groups) of a dense count matrix."""
   D = W.shape[0]
   M = np.zeros((B, B), dtype = np.int64)
   for x in range(D):
      for y in range(x, D):
         if W[x, y]:
            a, c = g[x], g[y]
            M[a, c] += W[x, y]
            if a != c:
               M[c, a] += W[x, y]
   return M


@njit(cache = True)
def _up_objective(M, n, D, E):
   """
   Description length of a level grouping ``D`` elements into groups of sizes ``n`` with edge
   counts ``M``, with a single group above: edge count terms, partition prior and top terms.
   """
   B = M.shape[0]
   S = _lbinom(D - 1, B - 1) + _lg(D) + math.log(D)
   for t in range(B):
      S -= _lg(n[t])
      for u in range(t, B):
         S += _ms(n[t], n[u], M[t, u], t == u)
   return S + math.log(B) + _ms(B, B, E, True)


@njit(cache = True)
def _up_merge_delta(T, U, M, n, D, E):
   """Change of :func:`_up_objective` when group ``T`` merges into ``U``."""
   B = M.shape[0]
   nn = n[T] + n[U]
   d = 0.
   for V in range(B):
      if V == T or V == U:
         continue
      d += _ms(nn, n[V], M[U, V] + M[T, V], False) - _ms(n[U], n[V], M[U, V], False) - _ms(n[T], n[V], M[T, V], False)
   d += _ms(nn, nn, M[U, U] + M[T, T] + M[T, U], True) - _ms(n[U], n[U], M[U, U], True) - _ms(n[T], n[T], M[T, T], True) - _ms(n[T], n[U], M[T, U], False)
   d += _lbinom(D - 1, B - 2) - _lbinom(D - 1, B - 1) - _lg(nn) + _lg(n[T]) + _lg(n[U])
   return d + _ms(B - 1, B - 1, E, True) - _ms(B, B, E, True) + math.log(B - 1) - math.log(B)


@njit(cache = True, parallel = True)
def _up_merge_eval(M, n, D, E, rnd, eps):
   """
   Best of ``rnd.shape[1]`` proposed merges of every group, evaluated in parallel: a uniform group
   with probability ``eps``, otherwise a group drawn in proportion to the edges to it.
   """
   B = M.shape[0]
   best = np.full(B, -1, dtype = np.int64)
   bd = np.full(B, np.inf)
   for T in prange(B):
      row = M[T].astype(np.float64)
      row[T] = 0.
      tot = row.sum()
      cum = np.cumsum(row)
      for c in range(rnd.shape[1]):
         if tot == 0 or rnd[T, c, 0] < eps:
            U = min(int(rnd[T, c, 1] * B), B - 1)
         else:
            U = min(np.searchsorted(cum, rnd[T, c, 1] * tot, side = "right"), B - 1)
         if U == T:
            continue
         d = _up_merge_delta(T, U, M, n, D, E)
         if d < bd[T]:
            bd[T], best[T] = d, U
   return best, bd


@njit(cache = True)
def _up_sweeps(W, g, B, sweeps, eps, seed):
   """
   Greedy moves of the elements between the ``B`` groups of a level (no group emptied), each to
   the best of a few groups proposed through its edges or uniformly.

   :return: The new grouping.
   """
   np.random.seed(seed)
   D = W.shape[0]
   Ps = np.zeros((2, D), dtype = np.int64)   # level 0: the D elements, level 1: their groups
   Ps[1] = g
   Ns = np.zeros((2, D), dtype = np.int64)
   Ns[0] = 1
   for x in range(D):
      Ns[1, g[x]] += 1
   Mf = np.concatenate((W.ravel(), _aggregate(W, g, B).ravel()))
   Ms = (Mf, np.array([0, D * D, D * D + B * B]), np.array([D, B]))
   cnt, tl = np.zeros(B, dtype = np.int64), np.zeros(B, dtype = np.int64)
   scr = (np.zeros(B + 2, dtype = np.int64), np.zeros(B + 2, dtype = np.int64), np.zeros(B + 2, dtype = np.int64), np.zeros(B + 2, dtype = np.int64))
   wsum = W.sum(axis = 1) - np.diag(W)
   for sweep in range(sweeps):
      moves = 0
      for _ in range(D):
         x = np.random.randint(D)
         T = Ps[1, x]
         if Ns[1, T] == 1:
            continue
         ntl, inside = _elem_rows(1, x, Ps, Ms, cnt, tl)
         bestS, bestd = -1, -1e-9
         for c in range(4):
            if wsum[x] == 0 or np.random.random() < eps:
               S = np.random.randint(B)
            else:
               r = np.random.random() * wsum[x]
               y = 0
               acc = 0.
               for z in range(D):
                  if z != x:
                     acc += W[x, z]
                     if acc > r:
                        y = z
                        break
               S = Ps[1, y]
            if S == T:
               continue
            d = _elem_delta(1, x, S, Ps, Ns, Ms, cnt, tl, ntl, inside, scr)
            if d < bestd:
               bestd, bestS = d, S
         if bestS >= 0:
            _elem_apply(1, x, bestS, Ps, Ns, Ms, cnt, tl, ntl, inside, scr)
            moves += 1
         cnt[tl[:ntl]] = 0
      if moves == 0:
         break
   return Ps[1].copy()


_NONE = (np.zeros(0, dtype = np.int64), np.zeros(0, dtype = np.int64), np.zeros(0, dtype = np.int64), np.zeros(0), np.zeros(0, dtype = np.int64))


def _sweep_chain(N, eu, ev, b0, par, Bs, sweeps, seed, beta = np.inf, eps = 0.1, kind = 0):
   """Partition moves only (no measurement model), greedy by default."""
   qtab, lu, vt = _q_tables()
   zi = np.zeros(0, dtype = np.int64)
   return _chain(N, eu, ev, b0, par, np.asarray(Bs, dtype = np.int64), kind, True, 0., 0., 0, *_NONE, 1, 0, 0., 0, zi, zi, zi, np.ones(6), True,
                 beta, sweeps, sweeps, 1, eps, np.full(4, 0.25), seed, qtab, lu, vt)


def _hierarchy(W, E, rng, merges, sweeps, eps):
   """
   Upper levels above a level 0 partition with edge counts ``W``, built level by level: at each
   level, groups are merged in halving rounds, with greedy sweeps, and the grouping with the
   smallest description length (a single group above) is kept.

   :return: ``(pars, Bs)``: the grouping of each level and its number of groups, the last level
      having more than one group (the single group on top is implicit).
   """
   pars, Bs = [], []
   while W.shape[0] > 1:
      D = W.shape[0]
      g, B = np.arange(D, dtype = np.int64), D
      M, n = W.copy(), np.ones(D, dtype = np.int64)
      best = None
      while B > 1:
         tgt, bd = _up_merge_eval(M, n, D, E, rng.random((B, merges, 2)), eps)
         g, B = _apply_merges(g, B, tgt, bd, max(1, B // 2))
         if B > 1:
            g = _up_sweeps(W, g, B, sweeps, eps, int(rng.integers(2 ** 31)))
         M, n = _aggregate(W, g, B), np.bincount(g, minlength = B).astype(np.int64)
         obj = _up_objective(M, n, D, E)
         if best is None or obj < best[0]:
            best = (obj, g.copy(), B)
      _, g, B = best
      if B == 1:
         break
      pars.append(g)
      Bs.append(B)
      W = _aggregate(W, g, B)
   return pars, Bs


def _nested_dl(N, eu, ev, b0, par, Bs, kind = 0):
   """Description length of a (nested, for kind 0) partition of a simple graph."""
   qtab, lu, vt = _q_tables()
   return _dl_light(N, eu, ev, b0, par, np.asarray(Bs, dtype = np.int64), kind, qtab, lu, vt)


def _fit(N, eu, ev, max_blocks = 4096, sweeps = 10, merges = 10, eps = 0.1, seed = None, kind = 0, flat = False, fixed = None):
   """
   Nested partition minimizing the description length of a simple undirected graph (or flat
   partition, for ``flat`` or kinds other than 0).

   Level 0: agglomerative heuristic (Peixoto 2014), groups merged in halving rounds from one group
   per node, the best of ``merges`` proposed merges of every group evaluated in parallel, followed
   by greedy node sweeps once at most ``max_blocks`` groups are left (merges and sweeps with the
   degree-corrected description length). Each partition of these rounds gets the upper levels of
   :func:`_hierarchy` or, flat, greedy sweeps with the description length of its kind, and the
   one with the smallest description length is refined by greedy sweeps at all levels. With
   ``fixed``, the level 0 partition has that many groups (or fewer, if the final sweeps empty
   some), instead of the number minimizing the description length.

   :return: ``(b0, par, Bs, S)``: level 0 groups, upper groupings concatenated, numbers of groups
      per level, description length.
   """
   qtab, lu, vt = _q_tables()
   rng = np.random.default_rng(seed)
   E = len(eu)
   b, B = np.arange(N, dtype = np.int64), N
   if E == 0 or N < 2:
      b = np.zeros(N, dtype = np.int64)
      return b, np.zeros(0, dtype = np.int64), [1], _nested_dl(N, eu, ev, b, np.zeros(0, dtype = np.int64), [1], kind)
   k = np.bincount(np.concatenate([eu, ev]), minlength = N).astype(np.int64)
   order = np.lexsort((np.concatenate([ev, eu]), np.concatenate([eu, ev])))
   nbr = np.concatenate([ev, eu])[order].astype(np.int64)
   ptr = np.concatenate([[0], np.cumsum(k)]).astype(np.int64)
   states = {}   # number of level 0 groups -> partition

   def reduce(b, B, target):   # merge rounds (halving, not below target), each followed by greedy sweeps
      while B > target:
         tabs = _l0_tables(N, eu, ev, b, B, k)
         tg, bd = _l0_merge_eval(B, N, E, tabs, rng.random((B, merges, 3)), eps, qtab, lu, vt)
         b, B = _apply_merges(b, B, tg, bd, max(target, B // 2))
         if 1 < B < N:
            b = _flat_sweeps(N, ptr, nbr, b, B, k, sweeps, 3, eps, int(rng.integers(2 ** 31)), qtab, lu, vt)
      return b, B

   results = {}   # number of level 0 groups -> (description length, b0, par, Bs)

   def evaluate(B0):
      if B0 not in results:
         src = min(x for x in states if x >= B0)
         b0, B0 = reduce(states[src].copy(), src, B0)
         states[B0] = b0
         if flat or kind != 0:
            if kind != 0 and B0 > 1:
               b0 = _sweep_chain(N, eu, ev, b0, np.zeros(0, dtype = np.int64), [B0], sweeps, int(rng.integers(2 ** 31)), kind = kind)[0]
            results[B0] = (_nested_dl(N, eu, ev, b0, np.zeros(0, dtype = np.int64), [B0], kind), b0, np.zeros(0, dtype = np.int64), [B0])
            return results[B0][0]
         W = np.zeros((B0, B0), dtype = np.int64)
         np.add.at(W, (b0[eu], b0[ev]), 1)
         W = W + W.T - np.diag(np.diag(W))
         pars, Bs = _hierarchy(W, E, rng, merges, sweeps, eps)
         par = np.concatenate(pars).astype(np.int64) if pars else np.zeros(0, dtype = np.int64)
         results[B0] = (_nested_dl(N, eu, ev, b0, par, [B0] + Bs), b0, par, [B0] + Bs)
      return results[B0][0]

   grid = [B]   # halving rounds down to one group, keeping the partitions of at most max_blocks groups
   while B > 1:
      b, B = reduce(b, B, max(1, B // 2))
      grid.append(B)
      if B <= max_blocks:
         states[B] = b.copy()
   grid = sorted(x for x in states)
   if fixed is not None:
      grid = [int(fixed)]
   # nested description length from few groups upwards, until it rises twice in a row
   rises, prev = 0, np.inf
   for B0 in grid:
      S = evaluate(B0)
      rises = rises + 1 if S > prev else 0
      prev = min(prev, S)
      if rises == 2:
         break
   done = sorted(results)
   i = int(np.argmin([results[x][0] for x in done]))
   lo, hi = done[max(i - 1, 0)], done[min(i + 1, len(done) - 1)]
   if fixed is not None:
      lo = hi = int(fixed)
   # golden section search between the neighbors of the best candidate
   phi = (math.sqrt(5) - 1) / 2
   while hi - lo > 2:
      m1 = int(round(hi - phi * (hi - lo)))
      m2 = int(round(lo + phi * (hi - lo)))
      m1, m2 = max(lo + 1, min(m1, hi - 1)), max(lo + 1, min(m2, hi - 1))
      if m1 == m2:
         m2 = m1 - 1 if m1 - 1 > lo else m1 + 1
      a1, a2 = max(m1, m2), min(m1, m2)
      if evaluate(a1) <= evaluate(a2):
         lo = a2
      else:
         hi = a1
   for x in range(lo, hi + 1):
      evaluate(x)
   best = min(results.values(), key = lambda r: r[0])
   _, b0, par, Bs = best
   best = (b0, par, Bs, 0.)
   b0, par, Bs, _ = best
   out = _sweep_chain(N, eu, ev, b0, par, Bs, sweeps, int(rng.integers(2 ** 31)), kind = kind)
   b0, par = out[0], out[1]
   return b0, par, Bs, _nested_dl(N, eu, ev, b0, par, Bs, kind)


# ------------------------------------------------------------------------------------------------
# Public functions
# ------------------------------------------------------------------------------------------------

def _pairs(net, name):
   """
   Node pairs of the edges of an undirected network, by network wide node position, without self
   loops, with the values of edge attributes.

   :param net: Network.
   :param name: Function name, for the error messages.
   :return: ``(N, s, t, attrs)``: ``s < t`` for every edge (input order, repeated pairs kept),
      and a dict from numeric edge attribute name to its values.
   :raises ValueError: If ``net`` is directed or multilayer.
   """
   if net.directed:
      raise ValueError(f"{name} works on undirected networks (see to_undirected)")
   if net.layer_labels is not None:
      raise ValueError(f"{name} does not work on multilayer networks (see layer_network or aggregate_network)")
   S, T, X = [], [], []
   for c in range(len(net.components)):
      ei, x = net._input_edges(c)
      S.append(ei[0].cpu().numpy() + int(net.node_offsets[c]))
      T.append(ei[1].cpu().numpy() + int(net.node_offsets[c]))
      X.append(x.cpu().numpy())
   s, t, X = np.concatenate(S), np.concatenate(T), np.concatenate(X)
   keep = s != t
   s, t, X = s[keep], t[keep], X[keep]
   return int(net.node_offsets[-1]), np.minimum(s, t).astype(np.int64), np.maximum(s, t).astype(np.int64), {a: X[:, j].astype(np.float64) for j, a in enumerate(net.edge_attributes)}


def _write_levels(net, b0, par, Bs, attribute):
   """
   Write the groups of every level as numeric node attributes: ``attribute`` for level 0,
   ``attribute + "_1"``, ``attribute + "_2"``, ... for the levels above.
   """
   offs = net.node_offsets
   g = b0.copy()
   at = 0
   for l in range(len(Bs)):
      if l > 0:
         g = par[at:at + Bs[l - 1]][g]
         at += Bs[l - 1]
      key = attribute if l == 0 else f"{attribute}_{l}"
      net.set_node_attribute(key, [g[offs[c]:offs[c + 1]].astype(np.float64) for c in range(len(net.components))], None)


def fit_nested_block_model(net, attribute = "community", max_blocks = 4096, sweeps = 10, seed = None):
   """
   Fit the nested degree-corrected stochastic blockmodel (Peixoto, 2014, 2017): the hierarchy of
   groups that minimizes the description length of the network. The same fit, and those of the
   other blockmodels, are also available for community discovery as
   :func:`~graphoskop.block_model_communities`.

   The model generates a network in levels. At the bottom, nodes are in groups, and edges are
   placed between groups according to the numbers of edges between every pair of groups, at
   random among their nodes in proportion to the node degrees (degree correction). These numbers
   are themselves a network between the groups, generated in the same way by a coarser grouping
   of the groups, and so on up to a single group. The description length is minus the logarithm
   of the probability of the network and of all its parameters (groups, numbers of edges between
   groups, degrees), the microcanonical formulation without free parameters:

   .. math::

      \\Sigma = -\\ln P(A \\mid k, e, b) - \\ln P(k \\mid e, b) - \\sum_l \\ln P(e_l \\mid e_{l + 1}, b_{l + 1}) - \\sum_l \\ln P(b_l)

   with the degree-corrected likelihood
   :math:`P(A \\mid k, e, b) = \\prod_{r < s} e_{rs}! \\prod_r e_{rr}!! \\prod_i k_i! / \\prod_r e_r!`,
   the degree prior :math:`P(k \\mid e, b) = \\prod_r \\prod_k \\eta^r_k! / (n_r!\\, q(e_r, n_r))`
   (degree histograms :math:`\\eta^r_k` uniform given the number of partitions :math:`q(e_r, n_r)`
   of :math:`e_r` into at most :math:`n_r` parts), the edge counts of each level uniform among the
   multigraphs between the groups of the level above,
   :math:`P(e_l \\mid e_{l + 1}, b_{l + 1}) = \\prod_{r < s} \\left(\\!\\binom{n_r n_s}{e_{rs}}\\!\\right)^{-1}
   \\prod_r \\left(\\!\\binom{n_r (n_r + 1) / 2}{e_{rr} / 2}\\!\\right)^{-1}`, and each partition of
   :math:`N` elements into :math:`B` groups uniform given its group sizes,
   :math:`P(b) = \\prod_r n_r! / N! \\binom{N - 1}{B - 1}^{-1} N^{-1}`. More groups, or more
   levels, are only used if they compress the network more than they cost, so the model selects
   its own numbers of groups and levels. :math:`q(m, n)` is exact for :math:`m \\leq 1000` and given
   by Szekeres' asymptotic formula otherwise (within 0.1 of its logarithm).

   The minimization follows Peixoto's agglomerative heuristic: starting from one group per node,
   groups are merged in rounds that halve their number, each group merging into the best of
   ``10`` proposed groups (proposals through its edges, all groups evaluated in parallel), each
   round followed by up to ``sweeps`` greedy sweeps of node moves. The upper levels are built in
   the same way on the network between groups. The number of groups of the bottom level is chosen
   by the nested description length, first along the halving rounds and then by golden section
   search; the result is refined by greedy sweeps at all levels. It is a heuristic: different
   seeds can give slightly different results.

   The groups are written as numeric node attributes: ``attribute`` for the bottom level,
   ``attribute + "_1"``, ``attribute + "_2"``, ... for the levels above (the group of every node).
   Parallel edges count once and self loops are ignored. The bottom level has at most
   ``max_blocks`` groups (the levels above are dense tables of that size).

   :param net: An undirected :class:`~graphoskop.Network`, not multilayer, modified in place.
   :param attribute: Name of the node attribute of the bottom level.
   :param max_blocks: Maximum number of groups of the bottom level.
   :param sweeps: Maximum number of greedy sweeps after each merge round.
   :param seed: Seed of the random numbers, or ``None``.
   :return: Dict with the ``description_length`` and the number of ``groups`` of every level,
      bottom first (without the single group on top).
   :raises ValueError: If ``net`` is directed or multilayer.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      fit = gsk.fit_nested_block_model(net, seed = 0)
      fit["groups"]                             # e.g. [64, 8, 2]
      net.get_node_attribute("community_1")      # groups of the second level, component 0
   """
   N, s, t, _ = _pairs(net, "fit_nested_block_model")
   key = np.unique(s * N + t)
   eu, ev = (key // N).astype(np.int64), (key % N).astype(np.int64)
   b0, par, Bs, S = _fit(N, eu, ev, max_blocks = max_blocks, sweeps = sweeps, seed = seed)
   _write_levels(net, b0, par, Bs, attribute)
   return {"description_length": float(S), "groups": [int(x) for x in Bs]}


# models of the true network: generator name -> (kind, shape of the partition)
_MODELS = {"nested_block_model": (0, "nested"), "degree_corrected_block_model": (0, "flat"), "stochastic_block_model": (2, "flat"),
           "planted_partition_graph": (3, "flat"), "configuration_model": (0, "one"), "gnm_random_graph": (2, "one"), "gnp_random_graph": (2, "one")}


def _model_state(net, model, groups, N, eu, ev, max_blocks, seed):
   """
   Model of the true network and its starting partition, fitted to the edges ``eu``, ``ev``.

   :return: ``(name, kind, b0, par, Bs, Cs, pmoves, lam)``: the generator name, the model kind,
      the partition with its numbers of groups ``Bs``, the numbers of labels of every level
      ``Cs`` (room for new groups when the partition is sampled: twice the groups of the fit, at
      least 10 more, at most the elements below), whether the partition is sampled, and the mean
      number of edges per class of pairs of the prior of the number of edges (see
      :func:`_edges_prior`).
   :raises ValueError: For an unknown model, or ``groups`` with a model without a flat partition.
   """
   name = getattr(model, "__name__", model)
   if name not in _MODELS:
      raise ValueError(f"unknown model {model!r}: give one of the generators {', '.join(sorted(_MODELS))}")
   kind, shape = _MODELS[name]
   none = np.zeros(0, dtype = np.int64)
   if groups is not None:
      if shape != "flat":
         raise ValueError("groups can be fixed only for degree_corrected_block_model, stochastic_block_model and planted_partition_graph")
      from .generators import _communities
      b0 = _communities(net, groups, None, None)[0].cpu().numpy().astype(np.int64)
      b0, par, Bs, pmoves = b0, none, [int(b0.max()) + 1], False
   elif shape == "one":
      b0, par, Bs, pmoves = np.zeros(N, dtype = np.int64), none, [1], False
   else:
      b0, par, Bs, _ = _fit(N, eu, ev, max_blocks = max_blocks, seed = seed, kind = kind, flat = shape == "flat")
      pmoves = True
   D = Bs[-1]
   P = (2. if Bs[0] > 1 else 1.) if kind == 3 else D * (D + 1) / 2.
   Cs, prev = list(Bs), N
   if pmoves:
      parts, at = [], 0
      for l, B in enumerate(Bs):
         Cs[l] = prev = min(prev, max(2 * B, B + 10))
         if l > 0:   # parents of the labels of the level below, empty ones included
            x = np.zeros(Cs[l - 1], dtype = np.int64)
            x[:Bs[l - 1]] = par[at:at + Bs[l - 1]]
            at += Bs[l - 1]
            parts.append(x)
      par = np.concatenate(parts) if parts else none
   return name, kind, b0, par, Bs, Cs, pmoves, max(len(eu), 1) / P


def _run_chains(N, eu, ev, b0, par, Bs, kind, pmoves, lam, model, meas, h, hfix, samples, burnin, interval, chains, seed):
   """
   Run independent chains in parallel threads, from the same state (``Bs``: the numbers of
   labels of every level; with ``pmoves``, the numbers of groups are sampled too).

   :return: Merged ``(keys, counts, total samples, traces)``.
   """
   qtab, lu, vt = _q_tables()
   rng = np.random.default_rng(seed)
   seeds = rng.integers(0, 2 ** 31, chains)
   out = [None] * chains
   sweeps = burnin + samples * interval

   def work(c):
      out[c] = _chain(N, eu, ev, b0, par, np.asarray(Bs, dtype = np.int64), kind, pmoves, _DNEW if pmoves else 0., float(lam), model, *meas, np.asarray(h, dtype = np.float64), hfix,
                      1., sweeps, burnin, interval, 0.1, np.full(4, 0.25), int(seeds[c]), qtab, lu, vt)

   threads = [threading.Thread(target = work, args = (c,)) for c in range(chains)]
   for th in threads:
      th.start()
   for th in threads:
      th.join()
   keys = np.concatenate([o[2] for o in out])
   counts = np.concatenate([o[3] for o in out])
   uk, inv = np.unique(keys, return_inverse = True)
   return uk, np.bincount(inv, counts, minlength = len(uk)), sum(o[4] for o in out), np.concatenate([o[5] for o in out])


def _corrected(net, N, uk, prob, s, t, attrs):
   """
   Corrected network: the nodes of ``net`` with their attributes, an edge for every pair with a
   positive posterior probability or observed in ``net``, with the edge attribute
   ``"probability"`` and the numeric edge attributes of the observed edges (NaN for new ones).
   """
   from .network import _build, _node_table, _qualitative_rows, _node_types
   obs = s * N + t
   ok, first = np.unique(obs, return_index = True)
   allk = np.union1d(uk, ok)
   p = np.zeros(len(allk))
   p[np.searchsorted(allk, uk)] = prob
   names = ["probability"] + [a for a in attrs if a != "probability"]
   ea = np.full((len(allk), len(names)), np.nan, dtype = np.float32)
   ea[:, 0] = p
   pos = np.searchsorted(allk, ok)
   for j, a in enumerate(names[1:], 1):
      ea[pos, j] = attrs[a][first]
   rows = np.arange(N)
   ids = np.concatenate(net.node_ids)
   types = _node_types(net)
   device = net.components[0].edge_index.device
   return _build((allk // N).astype(np.int64), (allk % N).astype(np.int64), ids, ea, names, lambda perm: _node_table(net, rows[perm]), net.node_attributes, False, device,
                 lambda perm: _qualitative_rows(net, rows[perm]), net.qualitative_node_attributes, net.node_format, types, net.bipartite_labels)


def measured_network_correction(net, measurements = None, positives = None, default_measurements = 1, default_positives = 0, heterogeneous = False, priors = None,
                                false_negative_rate = None, false_positive_rate = None, model = "nested_block_model", groups = None,
                                samples = 200, burnin = 200, interval = 5, chains = 4, max_blocks = 4096, seed = None):
   """
   Reconstruct a network from noisy measurements of its node pairs, with known or unknown error
   rates (Peixoto, 2018), using a generative model of the true network: by default the nested
   degree-corrected stochastic blockmodel (see :func:`fit_nested_block_model`), or a simpler one.

   Every node pair :math:`(i, j)` was measured :math:`n_{ij}` times, and an edge was observed in
   :math:`x_{ij}` of them. The pairs listed as edges of ``net`` carry their numbers in the edge
   attributes ``measurements`` and ``positives`` (by default 1 and 1: each edge of ``net`` was
   observed once); all other pairs were measured ``default_measurements`` times with
   ``default_positives`` positive outcomes (by default, once, without seeing an edge). A single
   observed network is thus the default case. Measured non-edges are listed as edges of ``net``
   with ``positives`` 0.

   An existing edge is missed in a measurement with probability :math:`p` (false negative), a
   non-edge is seen as an edge with probability :math:`q` (false positive), so that

   .. math::

      P(x \\mid n, A, p, q) = \\prod_{i < j} \\binom{n_{ij}}{x_{ij}}
      \\left[(1 - p)^{x_{ij}} p^{n_{ij} - x_{ij}}\\right]^{A_{ij}}
      \\left[q^{x_{ij}} (1 - q)^{n_{ij} - x_{ij}}\\right]^{1 - A_{ij}}

   * **Homogeneous errors** (``heterogeneous = False``): :math:`p` and :math:`q` are the same for
     all pairs, unknown, with priors :math:`p \\sim \\mathrm{Beta}(\\alpha, \\beta)`,
     :math:`q \\sim \\mathrm{Beta}(\\mu, \\nu)` (``priors = (alpha, beta, mu, nu)``, default
     ``(1, 1, 1, 1)``, uniform), integrated out (Eq. 19 of the paper). Either rate can instead be
     given as known (``false_negative_rate``, ``false_positive_rate``), and its prior is then
     ignored.
   * **Heterogeneous errors** (``heterogeneous = True``): every pair has its own error rates, drawn
     from these Beta distributions, whose hyperparameters are themselves unknown, with a flat prior
     on :math:`[0.01, 1000]` and sampled (``priors`` are their initial values, default
     ``(1, 10, 1, 10)``); the rates are integrated out pair by pair (Eq. 58).

   The posterior distribution of the true network :math:`A` and of the groups is sampled by Markov
   chain Monte Carlo, and the posterior probability :math:`\\pi_{ij}` of every pair being an edge
   is the fraction of samples in which it is one. The structure inferred from the network (pairs
   of groups densely or sparsely connected) is what tells real edges from errors: an edge between
   two groups that are otherwise unconnected is likely spurious, a missing edge within a dense
   group likely missing. Edges with :math:`\\pi_{ij} > 1/2` form the maximum marginal posterior
   estimate of the network:

   .. code-block:: python

      corrected, summary = gsk.measured_network_correction(net)
      clean = gsk.extract_backbone(corrected, 0.5, score = "probability", keep_isolates = True)

   **Models of the true network.** ``model`` is one of the generators below (the function itself,
   e.g. ``gsk.configuration_model``, or its name); the reconstruction uses Peixoto's nonparametric
   microcanonical version of its family, with all its parameters integrated out under
   noninformative priors (not the parametrization of the generator):

   * :func:`~graphoskop.nested_block_model` (default): the nested degree-corrected stochastic
     blockmodel, a hierarchy of groups (see :func:`fit_nested_block_model`);
   * :func:`~graphoskop.degree_corrected_block_model`: degree-corrected blockmodel, one level of
     groups;
   * :func:`~graphoskop.stochastic_block_model`: blockmodel without degree correction (all nodes
     of a group alike);
   * :func:`~graphoskop.planted_partition_graph`: one density of edges within groups, one between;
   * :func:`~graphoskop.configuration_model`: the degrees only (degree-corrected, one group);
   * :func:`~graphoskop.gnm_random_graph` or :func:`~graphoskop.gnp_random_graph`: Erdős-Rényi,
     no structure (one group without degree correction).

   The more structure a model captures, the better it tells errors apart: Erdős-Rényi gives every
   unobserved pair the same chance, the configuration model favors pairs of high degree nodes,
   blockmodels pairs of densely connected groups (as in Fig. 3 of the paper). The groups of the
   flat blockmodels are fitted by minimizing their description length, as
   :func:`fit_nested_block_model` does for the nested one (the number of groups chosen along the
   same merges, refined by greedy sweeps under each model), then sampled; or fixed, read from the
   node attribute ``groups`` (numeric or qualitative).

   **Latent multigraph.** These models generate multigraphs, with parallel edges and self loops,
   while measurements see simple networks: as in Appendix D of the paper, the chain samples a
   multigraph :math:`G`, and the true network is its collapse, :math:`A_{ij} = 1` if
   :math:`G_{ij} > 0`, :math:`i \\neq j`. Its number of edges has the negative binomial prior of
   Eq. C9 (geometric edge counts per pair of groups, their mean set by the observed edges), so
   that the posterior is proper.

   **Single observations.** With one measurement per pair, a missing edge looks like every other
   unobserved pair between the same groups, so it cannot be singled out: the posterior spreads
   its probability evenly over all of them (low, unless the groups are dense), and only spurious
   edges, observed between sparsely connected groups, are told apart. With both rates unknown, the
   data cannot tell a structure with errors from the same structure without them either, and the
   estimated rates go to their minimum; known rates (``false_negative_rate``,
   ``false_positive_rate``), even approximate, make the probabilities and the expected number of
   edges calibrated. Repeated measurements (``measurements``, ``default_measurements`` above 1)
   are what makes individual missing edges recoverable.

   **Sampling.** The chain starts from the edges observed in most of their measurements, with the
   groups fitted to them. Each sweep tries a move of every node between the groups of the bottom
   level, of every group between the groups of the level above (for all levels; no group moves if
   the groups are fixed, or for the configuration model and Erdős-Rényi), and
   :math:`\\max(E, N)` moves of the multigraph:
   a pair is chosen at random among the edges of the multigraph, among all pairs, among the listed
   pairs, or through the blockmodel (a node, a group connected to its group, a member of it), each
   with probability 1/4; an edge is added to it or, if it has some, one is added or removed with
   probability 1/2 each (Eq. D8), and the move is accepted with the Metropolis-Hastings
   criterion.

   The numbers of groups are sampled too, as in graph-tool: a node or group moves to a new group
   with probability 0.01 (Eq. B4), the new group joining the group of the level above of the one
   it left, and groups that lose their last member disappear (into a group under the same group
   above, so that every move can be undone). Since single moves change the number of groups
   slowly, each sweep also tries 3 split-merge moves of the bottom groups (Jain and Neal, 2004;
   Peixoto, 2020): a group is split in two, its nodes assigned at random and then by 4 restricted
   Gibbs sweeps between the two halves, or two groups are merged, the probability of the reverse
   split computed in the same way. The numbers of groups are thus averaged over in the posterior,
   which matters when many partitions with more groups, each less likely than the best one,
   together outweigh it. Every level has room for twice the groups of the fit, at least 10 more
   (at most its number of elements); the hierarchy keeps the depth of the fit. On Zachary's karate club
   measured twice, with one inconsistent pair (Fig. 3 of the paper), Erdős-Rényi gives that pair a
   probability of 0.50, the configuration model 0.87 and the nested blockmodel 0.93 (0.5, 0.87
   and 0.93 in the paper; 0.92 to 0.96 with the default number of sweeps), averaging over about 3
   groups, although a single group describes the club best. The number of groups changes slowly
   (over hundreds of sweeps): its posterior mean (``mean_groups``) needs longer chains than the
   probabilities of the edges.

   ``chains`` independent chains run in parallel threads, each discarding ``burnin`` sweeps and
   then taking ``samples`` samples every ``interval`` sweeps. The chains run in a compiled kernel
   (Numba) on the CPU.

   :param net: An undirected :class:`~graphoskop.Network`, not multilayer.
   :param measurements: Edge attribute with the number of measurements of each listed pair, or
      ``None`` for 1.
   :param positives: Edge attribute with the number of positive measurements, or ``None`` for all
      of them.
   :param default_measurements: Number of measurements of the pairs not in ``net``.
   :param default_positives: Number of positive measurements of the pairs not in ``net``.
   :param heterogeneous: Pair by pair error rates instead of one rate for all pairs.
   :param priors: ``(alpha, beta, mu, nu)``, see above.
   :param false_negative_rate: Known :math:`p`, or ``None`` (unknown); homogeneous errors only.
   :param false_positive_rate: Known :math:`q`, or ``None`` (unknown); homogeneous errors only.
   :param model: Model of the true network: a generator, or its name (see above).
   :param groups: Node attribute with fixed groups (flat blockmodels only), or ``None`` to fit and
      sample them.
   :param samples: Samples per chain.
   :param burnin: Sweeps discarded at the start of every chain.
   :param interval: Sweeps between samples.
   :param chains: Number of chains (threads).
   :param max_blocks: Maximum number of groups of the bottom level of the fit.
   :param seed: Seed of the random numbers, or ``None``.
   :return: ``(corrected, summary)``: the corrected network, a new :class:`~graphoskop.Network`
      with the nodes of ``net`` and an edge for every pair observed or ever sampled as an edge,
      with the edge attribute ``"probability"`` (:math:`\\pi_{ij}`) and the numeric edge attributes
      of ``net`` (NaN for new pairs); and a dict with the ``model``, the numbers of ``groups`` per
      level of its fit, the posterior means of the number of groups of the bottom level
      (``mean_groups``), of the number of ``edges``, of the ``description_length``, of
      the ``false_negative_rate`` :math:`p` and the ``false_positive_rate`` :math:`q` (with
      heterogeneous errors, the means of their Beta distributions, and the
      ``hyperparameters``), and the number of ``samples``.
   :raises ValueError: If ``net`` is directed or multilayer, if measurements are negative or
      fewer than the positive outcomes, if known rates are not in :math:`(0, 1)` or are given
      with heterogeneous errors, if ``model`` is not one of the generators above, or if
      ``groups`` is given to another model or names a missing attribute.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("interactions.tsv")      # one observation of the network
      corrected, summary = gsk.measured_network_correction(net, false_negative_rate = 0.1, false_positive_rate = 0.001, seed = 0)
      spurious = gsk.extract_backbone(corrected, 0.5, score = "probability")   # edges kept

      net = gsk.load_network("screens.tsv")           # columns trials, hits: repeated measurements
      corrected, summary = gsk.measured_network_correction(net, "trials", "hits", default_measurements = 3, heterogeneous = True)
      cm, _ = gsk.measured_network_correction(net, "trials", "hits", default_measurements = 3, model = gsk.configuration_model)
      sbm, _ = gsk.measured_network_correction(net, "trials", "hits", default_measurements = 3, model = "stochastic_block_model", groups = "lab")
   """
   N, s, t, attrs = _pairs(net, "measured_network_correction")
   n = np.ones(len(s), dtype = np.int64) if measurements is None else np.rint(attrs[measurements]).astype(np.int64)
   x = n.copy() if positives is None else np.rint(attrs[positives]).astype(np.int64)
   if (n < 0).any() or (x < 0).any() or (x > n).any() or default_measurements < 0 or not 0 <= default_positives <= default_measurements:
      raise ValueError("measurements must be non-negative and at least the positive outcomes")
   key = s * N + t
   lkeys, inv = np.unique(key, return_inverse = True)   # repeated pairs: measurements added up
   ln, lx = np.bincount(inv, n, minlength = len(lkeys)).astype(np.int64), np.bincount(inv, x, minlength = len(lkeys)).astype(np.int64)
   unl = N * (N - 1) // 2 - len(lkeys)
   cls, lc = np.unique(np.stack([np.concatenate([ln, [default_measurements]]), np.concatenate([lx, [default_positives]])], 1), axis = 0, return_inverse = True)
   lc = lc.ravel().astype(np.int64)
   ctot = np.bincount(lc[:-1], minlength = len(cls)).astype(np.int64)
   ctot[lc[-1]] += unl
   meas = (lkeys.astype(np.int64), ln, lx, np.zeros(len(lkeys)), lc[:-1], int(default_measurements), int(default_positives), 0., int(lc[-1]), cls[:, 0].astype(np.int64), cls[:, 1].astype(np.int64), ctot)
   rates = [np.nan if r is None else float(r) for r in (false_negative_rate, false_positive_rate)]
   if any(not (np.isnan(r) or 0 < r < 1) for r in rates) or (heterogeneous and not np.isnan(rates).all()):
      raise ValueError("known error rates must be in (0, 1), with homogeneous errors only")
   h = np.array(list(priors if priors is not None else ((1., 10., 1., 10.) if heterogeneous else (1., 1., 1., 1.))) + rates, dtype = np.float64)
   init = lkeys[2 * lx > ln]
   eu, ev = (init // N).astype(np.int64), (init % N).astype(np.int64)
   name, kind, b0, par, Bs, Cs, pmoves, lam = _model_state(net, model, groups, N, eu, ev, max_blocks, seed)
   uk, cnt, ns, trace = _run_chains(N, eu, ev, b0, par, Cs, kind, pmoves, lam, 2 if heterogeneous else 1, meas, h, False, samples, burnin, interval, chains, seed)
   summary = {"model": name, "groups": [int(B) for B in Bs], "mean_groups": float(trace[:, 6].mean()), "edges": float(trace[:, 1].mean()),
              "description_length": float(trace[:, 0].mean()), "samples": int(ns)}
   if heterogeneous:
      hm = trace[:, 2:6].mean(axis = 0)
      summary["hyperparameters"] = tuple(float(v) for v in hm)
      summary["false_negative_rate"] = float(np.mean(trace[:, 2] / (trace[:, 2] + trace[:, 3])))
      summary["false_positive_rate"] = float(np.mean(trace[:, 4] / (trace[:, 4] + trace[:, 5])))
   else:
      Mt = default_measurements * unl + ln.sum()
      Xt = default_positives * unl + lx.sum()
      Em, Tm = trace[:, 2], trace[:, 3]
      summary["false_negative_rate"] = float(np.mean((Em - Tm + h[0]) / (Em + h[0] + h[1]))) if np.isnan(h[4]) else h[4]
      summary["false_positive_rate"] = float(np.mean((Xt - Tm + h[2]) / (Mt - Em + h[2] + h[3]))) if np.isnan(h[5]) else h[5]
   return _corrected(net, N, uk, cnt / max(ns, 1), s, t, attrs), summary


def uncertain_network_correction(net, probability = "probability", default_probability = 0., model = "nested_block_model", groups = None,
                                 samples = 200, burnin = 200, interval = 5, chains = 4, max_blocks = 4096, seed = None):
   """
   Reconstruct a network whose node pairs come with probabilities of being edges, estimated by
   some other measurement (extrinsic uncertainty, Section III of Peixoto, 2018), using a
   generative model of the true network: by default the nested degree-corrected stochastic
   blockmodel, or a simpler one (see :func:`measured_network_correction`).

   Every pair listed as an edge of ``net`` has the probability :math:`Q_{ij}` in the edge attribute
   ``probability``; the other pairs have ``default_probability`` (by default 0: they cannot be
   edges, so only the listed pairs are uncertain). :math:`Q_{ij}` is read as the probability of the
   edge given its measurement alone, as if with a uniform prior, so it enters the posterior as the
   likelihood ratio :math:`Q_{ij} / (1 - Q_{ij})`:

   .. math::

      P(A \\mid Q) \\propto P(A) \\prod_{i < j} Q_{ij}^{A_{ij}} (1 - Q_{ij})^{1 - A_{ij}}

   with :math:`P(A)` the model, which pulls the probabilities of the pairs towards what the
   structure of the rest of the network makes likely. Models, sampling, outputs and parameters
   are as in :func:`measured_network_correction`; the chain starts from the pairs with
   :math:`Q_{ij} > 1/2`. Pairs with probability 0 never become edges, pairs with probability 1
   are always edges.

   :param net: An undirected :class:`~graphoskop.Network`, not multilayer.
   :param probability: Edge attribute with the probability of each listed pair.
   :param default_probability: Probability of the pairs not in ``net``.
   :param model: Model of the true network: a generator, or its name (see
      :func:`measured_network_correction`).
   :param groups: Node attribute with fixed groups (flat blockmodels only), or ``None`` to fit and
      sample them.
   :param samples: Samples per chain.
   :param burnin: Sweeps discarded at the start of every chain.
   :param interval: Sweeps between samples.
   :param chains: Number of chains (threads).
   :param max_blocks: Maximum number of groups of the bottom level of the fit.
   :param seed: Seed of the random numbers, or ``None``.
   :return: ``(corrected, summary)``, as :func:`measured_network_correction` (the summary has
      ``model``, ``groups``, ``mean_groups``, ``edges``, ``description_length`` and ``samples``). The edge
      attribute ``"probability"`` of the corrected network replaces the input one.
   :raises ValueError: If ``net`` is directed or multilayer, if a probability is outside
      :math:`[0, 1]`, if a pair is listed twice, or for ``model`` and ``groups`` as in
      :func:`measured_network_correction`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("predicted_interactions.tsv")   # column probability
      corrected, summary = gsk.uncertain_network_correction(net, seed = 0)
      likely = gsk.extract_backbone(corrected, 0.5, score = "probability")
      cm, _ = gsk.uncertain_network_correction(net, model = gsk.configuration_model, seed = 0)
   """
   N, s, t, attrs = _pairs(net, "uncertain_network_correction")
   q = attrs[probability]
   if (q < 0).any() or (q > 1).any() or not 0 <= default_probability <= 1:
      raise ValueError("probabilities must be between 0 and 1")
   key = s * N + t
   lkeys = np.unique(key)
   if len(lkeys) < len(key):
      raise ValueError("every pair can be listed only once")
   order = np.argsort(key)
   lq = q[order].astype(np.float64)
   z = np.zeros(len(lkeys), dtype = np.int64)
   zi = np.zeros(0, dtype = np.int64)
   meas = (lkeys.astype(np.int64), z, z, lq, z, 0, 0, float(default_probability), 0, zi, zi, zi)
   init = lkeys[lq > 0.5]
   eu, ev = (init // N).astype(np.int64), (init % N).astype(np.int64)
   name, kind, b0, par, Bs, Cs, pmoves, lam = _model_state(net, model, groups, N, eu, ev, max_blocks, seed)
   uk, cnt, ns, trace = _run_chains(N, eu, ev, b0, par, Cs, kind, pmoves, lam, 3, meas, np.ones(6), True, samples, burnin, interval, chains, seed)
   summary = {"model": name, "groups": [int(B) for B in Bs], "mean_groups": float(trace[:, 6].mean()), "edges": float(trace[:, 1].mean()),
              "description_length": float(trace[:, 0].mean()), "samples": int(ns)}
   rest = {a: v for a, v in attrs.items() if a != probability}
   return _corrected(net, N, uk, cnt / max(ns, 1), s, t, rest), summary
