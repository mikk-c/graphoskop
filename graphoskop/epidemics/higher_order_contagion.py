"""
Higher-order contagion: epidemics on simplicial complexes and hypergraphs, where groups of nodes,
not only pairs, carry the infection (Iacopini, Petri, Barrat and Latora, 2019, the simplicial
contagion model).

Every simplex (on a simplicial complex: every face of every facet, edges included) or hyperedge
(on a hypergraph) of order :math:`\\omega` (dimension :math:`\\omega`, :math:`\\omega + 1` nodes) is
an infection channel with probability :math:`\\beta_\\omega`: at each step, a susceptible node of
the group is infected through it with probability :math:`\\beta_\\omega` if all the other
:math:`\\omega` nodes of the group are infected. Edges (order 1) give the pairwise contagion of
:mod:`~graphoskop.epidemics.simple_contagion` with one chance per infected neighbor (the
``"classical"`` reinforcement of :mod:`~graphoskop.epidemics.complex_contagion`); a filled
triangle (order 2) infects its third node only when two of them are infected, a social
reinforcement that the edges of a triangle alone do not give. Channels are independent: a
susceptible node is infected with probability :math:`1 - \\prod_g (1 - \\beta_g)` over its groups
:math:`g` that meet the condition. With ``threshold``, a group infects as soon as a share of its
other nodes is infected (at least one): a critical mass, which suits hypergraphs with large
hyperedges, where waiting for all the other nodes rarely happens.

In a simplicial complex, a facet of :math:`m` nodes holds all its faces as channels: a filled
triangle is three edges and one triangle. The orders with a probability are enumerated (see
:meth:`~graphoskop.Network.simplices`); higher ones are not. A hypergraph has only its hyperedges:
a hyperedge of three nodes is a triangle channel, without its pairs.

Time, states, recoveries, seeds, immunization, stored states and the output are those of
:mod:`~graphoskop.epidemics.simple_contagion`: discrete, synchronous steps; SI, SIS (the model of
Iacopini et al.) and SIR. In the mean field, with :math:`\\lambda = \\beta_1 \\langle k \\rangle /
\\mu` and :math:`\\lambda_\\Delta = \\beta_2 \\langle k_\\Delta \\rangle / \\mu`
(:math:`\\langle k_\\Delta \\rangle` the mean number of triangles of a node, from
:func:`~graphoskop.generalized_degree`), the SIS model has a discontinuous transition with a
bistable region when :math:`\\lambda_\\Delta > 1`: whether the epidemic survives depends on the
share of initial seeds.

Runs are simulated by compiled code on CPU threads, each thread running whole runs. Every node
keeps its number of infected neighbors through edges, as in pairwise contagion, and the escape
probability of its larger groups that can infect it, updated only when a group starts or stops
infecting, so a step costs time proportional to the exposed nodes and to the edges and groups of
the nodes that changed. Networks on a GPU are copied to the CPU for the simulation.
"""

import numba
import numpy as np
import scipy.sparse as sp
from numba import njit, prange

from .._policy import _higher_order
from ..ergms import _random
from .simple_contagion import _initial, _size, _sums, _table


@njit(cache = True)
def _toggle(g, sign, gptr, gmem, lb, acc, nact, sure, state, listed, front, nf):
   """
   A group starts (``sign = 1``) or stops (``-1``) infecting: its nodes gain or lose its channel;
   susceptible ones that gain it join the list of exposed nodes. Returns the new length of the
   list.
   """
   for e in range(gptr[g], gptr[g + 1]):
      v = gmem[e]
      nact[v] += sign
      if np.isinf(lb[g]):
         sure[v] += sign
      elif nact[v] == 0:
         acc[v] = 0.   # no rounding residue
      else:
         acc[v] += sign * lb[g]
      if sign > 0 and state[v] == 0 and not listed[v]:
         listed[v] = True
         front[nf] = v
         nf += 1
   return nf


@njit(cache = True, parallel = True)
def _run_groups(eptr, enbr, lb1, nptr, ngrp, gptr, gmem, lb, need, seeds, removed, immune, mu, sis, max_steps, window, tolerance, rng, total, square, ends, finals, last):
   """
   Runs of the SI, SIS or SIR model with group channels, in parallel: thread ``b`` simulates runs
   ``b``, ``b + B``, ..., with the bookkeeping of
   :func:`~graphoskop.epidemics.simple_contagion._run_numba`.

   Pairs (groups of two nodes) are edges: every node counts its infected neighbors, as in the
   pairwise engine. Every larger group keeps its number of infected nodes, and every node the sum
   of :math:`\\log(1 - \\beta_g)` over its larger groups that infect (those with enough infected
   nodes), updated only when a group starts or stops infecting. A step costs time proportional to
   the exposed susceptible nodes, to the infected nodes (recoveries), and to the edges and groups
   of the nodes that changed.

   :param eptr: CSR pointers of the neighbors of each node through pairs.
   :param enbr: Neighbors through pairs (repeated for repeated pairs).
   :param lb1: :math:`\\log(1 - \\beta_1)` of the pairs (``-inf`` for probability 1).
   :param nptr: CSR pointers of the larger groups of each node.
   :param ngrp: Larger groups of each node.
   :param gptr: CSR pointers of the nodes of each larger group.
   :param gmem: Nodes of each larger group.
   :param lb: :math:`\\log(1 - \\beta_g)` of each larger group (``-inf`` for probability 1).
   :param need: Infected nodes a group needs to infect its susceptible nodes.
   :param seeds: Infected nodes at step 0: run ``r`` uses row ``r % len(seeds)``.
   :param removed: Removed nodes at step 0 (every run).
   :param immune: Immunized nodes: run ``r`` uses row ``r % len(immune)``.
   :param mu: Recovery probability (0 for SI).
   :param sis: Whether recovered nodes become susceptible (SIS) or removed (SIR).
   :param max_steps: Largest number of steps.
   :param window: Steps of the plateau test, or 0 for none.
   :param tolerance: Relative tolerance of the plateau test.
   :param rng: xorshift state of every run.
   :param total: Threads by ``max_steps + 1`` by 3 (S, I, R), written in place.
   :param square: Same shape as ``total``, written in place.
   :param ends: Last step of every run, written in place.
   :param finals: Number of nodes in each state at the end of every run, written in place.
   :param last: State of every node at the end of the last run, written in place.
   """
   R, B, n, G = rng.shape[0], total.shape[0], nptr.shape[0] - 1, gptr.shape[0] - 1
   for b in prange(B):
      state = np.zeros(n, dtype = np.uint8)   # 0 S, 1 I, 2 R, 4 immunized
      cnt = np.zeros(G, dtype = np.int32)     # infected nodes of each group
      acc = np.zeros(n)                       # log of the probability of escaping the infecting groups
      nact = np.zeros(n, dtype = np.int32)    # infecting groups of each node
      sure = np.zeros(n, dtype = np.int32)    # infecting groups of probability 1
      ecnt = np.zeros(n, dtype = np.int32)    # infected neighbors through pairs
      listed = np.zeros(n, dtype = np.bool_)
      front = np.empty(n, dtype = np.int64)   # exposed susceptible nodes (and stale entries)
      inf = np.empty(n, dtype = np.int64)
      new = np.empty(n, dtype = np.int64)
      rec = np.empty(n, dtype = np.int64)
      cum = np.zeros(max_steps + 2, dtype = np.int64)
      for r in range(b, R, B):
         st = rng[r:r + 1]
         state[:] = 0
         cnt[:] = 0
         acc[:] = 0.
         nact[:] = 0
         sure[:] = 0
         ecnt[:] = 0
         listed[:] = False
         for v in removed:
            state[v] = 2
         nV = immune.shape[1]
         for v in immune[r % immune.shape[0]]:
            state[v] = 4
         nn, nr, ni, nf = 0, 0, 0, 0
         for v in seeds[r % seeds.shape[0]]:
            state[v] = 1
            new[nn] = v
            nn += 1
         nR, t, m2 = removed.shape[0], 0, 0.
         while True:
            for i in range(nn):   # newly infected: counts of their neighbors and groups
               u = new[i]
               inf[ni] = u
               ni += 1
               for e in range(eptr[u], eptr[u + 1]):
                  v = enbr[e]
                  ecnt[v] += 1
                  if state[v] == 0 and not listed[v]:
                     listed[v] = True
                     front[nf] = v
                     nf += 1
               for e in range(nptr[u], nptr[u + 1]):
                  g = ngrp[e]
                  cnt[g] += 1
                  if cnt[g] == need[g]:
                     nf = _toggle(g, 1, gptr, gmem, lb, acc, nact, sure, state, listed, front, nf)
            for i in range(nr):   # newly recovered
               u = rec[i]
               for e in range(eptr[u], eptr[u + 1]):
                  ecnt[enbr[e]] -= 1
               for e in range(nptr[u], nptr[u + 1]):
                  g = ngrp[e]
                  cnt[g] -= 1
                  if cnt[g] == need[g] - 1:
                     nf = _toggle(g, -1, gptr, gmem, lb, acc, nact, sure, state, listed, front, nf)
            for i in range(nr):   # recovered nodes still exposed
               u = rec[i]
               if sis and (nact[u] > 0 or ecnt[u] > 0) and not listed[u]:
                  listed[u] = True
                  front[nf] = u
                  nf += 1
            total[b, t, 0] += n - ni - nR - nV
            total[b, t, 1] += ni
            total[b, t, 2] += nR
            square[b, t, 0] += (n - ni - nR - nV) * (n - ni - nR - nV)
            square[b, t, 1] += ni * ni
            square[b, t, 2] += nR * nR
            cum[t + 1] = cum[t] + ni
            plateau = False
            if window > 0 and t + 1 >= 2 * window:
               m2 = (cum[t + 1] - cum[t + 1 - window]) / window
               m1 = (cum[t + 1 - window] - cum[t + 1 - 2 * window]) / window
               plateau = m2 > 0 and abs(m2 - m1) <= tolerance * m2
            keep = 0
            for i in range(nf):   # drop the nodes that are no longer exposed
               v = front[i]
               if state[v] == 0 and (nact[v] > 0 or ecnt[v] > 0):
                  front[keep] = v
                  keep += 1
               else:
                  listed[v] = False
            nf = keep
            if plateau or (nf == 0 and (mu == 0 or ni == 0)) or t == max_steps:
               break
            t += 1
            nn, keep = 0, 0
            for i in range(nf):   # infections, from the states at the start of the step
               v = front[i]
               if sure[v] > 0 or (ecnt[v] > 0 and lb1 == -np.inf) or _random(st) < 1 - np.exp(acc[v] + (ecnt[v] * lb1 if ecnt[v] > 0 else 0.)):
                  new[nn] = v
                  nn += 1
                  listed[v] = False
               else:
                  front[keep] = v
                  keep += 1
            nf, nr = keep, 0
            if mu > 0:   # recoveries of the nodes infected at the start of the step
               keep = 0
               for i in range(ni):
                  u = inf[i]
                  if _random(st) < mu:
                     rec[nr] = u
                     nr += 1
                  else:
                     inf[keep] = u
                     keep += 1
               ni = keep
            for i in range(nn):
               state[new[i]] = 1
            for i in range(nr):
               state[rec[i]] = 0 if sis else 2
            if not sis:
               nR += nr
         ends[r] = t
         finals[r, 0], finals[r, 1], finals[r, 2] = n - ni - nR - nV, ni, nR
         if plateau:
            finals[r, 0], finals[r, 1] = n - nR - nV - m2, m2
         if r == R - 1:
            last[:] = state


def _orders(beta):
   """
   Infection probability of every order: dictionary from order to probability, or a single one
   for all orders (``None`` key).

   :raises ValueError: If a probability is not in [0, 1] or an order is below 1.
   """
   if isinstance(beta, (int, float, np.number)):
      p = {None: float(beta)}
   elif isinstance(beta, dict):
      p = {int(k): float(v) for k, v in beta.items()}
   else:
      p = {k + 1: float(v) for k, v in enumerate(beta)}
   if any(k is not None and k < 1 for k in p) or not all(0 <= v <= 1 for v in p.values()) or not any(v > 0 for v in p.values()):
      raise ValueError(f"beta must give probabilities in [0, 1] (not all 0) to orders of at least 1 (edges), got {beta}")
   return p


def _groups(net, comps, off, beta, threshold):
   """
   Infection channels of the nodes of some components: the simplices of the orders with a
   probability (simplicial complex) or the hyperedges of two nodes or more (hypergraph).

   :return: ``(eptr, enbr, lb1, nptr, ngrp, gptr, gmem, lb, need, everyone)``, NumPy arrays: the
      neighbors of each node through pairs (CSR) and :math:`\\log(1 - \\beta_1)`; the larger
      groups of each node (CSR), the nodes of each larger group (CSR), its
      :math:`\\log(1 - \\beta)` and its need; and all the groups of each node (CSR pointers and
      groups).
   """
   from ..higher_order import _simplices, _top
   p = _orders(beta)
   prob = lambda w: p.get(w, p.get(None, 0.))
   sizes, members = [], []
   for c, o in zip(comps, off):
      if net.higher_order == "simplicial":
         top = _top(net, c) if None in p else min(max(p), _top(net, c))
         for w in range(1, top + 1):
            if prob(w) > 0:
               S = _simplices(net, c, w).cpu().numpy()
               sizes.append(np.full(len(S), w + 1))
               members.append(S.reshape(-1) + o)
      else:
         k = int(net.bipartite_split[c])
         s, t = net._input_edges(c)[0].cpu().numpy()
         node, h = np.minimum(s, t), np.maximum(s, t) - k
         order = np.argsort(h, kind = "stable")
         z = np.bincount(h, minlength = net.components[c].num_nodes - k)
         keep = (z >= 2) & (np.array([prob(w - 1) for w in range(int(z.max(initial = 0)) + 1)])[z] > 0)
         sizes.append(z[keep])
         members.append(node[order][np.repeat(keep, z)] + o)
   size = np.concatenate(sizes).astype(np.int64) if sizes else np.zeros(0, dtype = np.int64)
   gmem = np.concatenate(members).astype(np.int64) if members else np.zeros(0, dtype = np.int64)
   n = int(off[-1])
   csr = lambda key, val: (np.r_[0, np.cumsum(np.bincount(key, minlength = n))].astype(np.int64), val[np.argsort(key, kind = "stable")].astype(np.int64))
   everyone = csr(gmem, np.repeat(np.arange(len(size)), size))
   pair = np.repeat(size == 2, size)   # pairs: edges
   u, v = gmem[pair][0::2], gmem[pair][1::2]
   eptr, enbr = csr(np.r_[u, v], np.r_[v, u])
   size, gmem = size[size > 2], gmem[~pair]
   gptr = np.r_[0, np.cumsum(size)].astype(np.int64)
   q = np.array([prob(w - 1) for w in range(int(size.max(initial = 0)) + 1)], dtype = np.float64)[size]
   with np.errstate(divide = "ignore"):
      lb, lb1 = np.log1p(-q), float(np.log1p(-prob(1)))
   need = size - 1 if threshold is None else np.maximum(np.ceil(threshold * (size - 1) - 1e-9), 1)
   nptr, ngrp = csr(gmem, np.repeat(np.arange(len(size)), size))
   return eptr, enbr, lb1, nptr, ngrp, gptr, gmem, lb, need.astype(np.int32), everyone


def _co_members(nptr, ngrp, n):
   """Neighbors of every node (nodes sharing a group with it), in CSR form, for acquaintance immunization."""
   H = sp.csr_matrix((np.ones(len(ngrp)), ngrp, nptr), shape = (n, int(ngrp.max(initial = -1)) + 1))
   A = (H @ H.T).tocsr()
   A.setdiag(0)
   A.eliminate_zeros()
   return A.indptr.astype(np.int64), A.indices.astype(np.int64)


def _simulate_groups(net, model, beta, mu, threshold, seeds, ranking, selection, runs, max_steps, component, seed, attribute, resume, window, tolerance, immunize, strategy):
   """
   Body of the higher-order contagion models (see the module description).

   :param model: ``"SI"``, ``"SIS"`` or ``"SIR"``.
   :return: The table of the states over time.
   """
   if net.higher_order is None:
      raise ValueError("higher-order contagion runs on simplicial complexes and hypergraphs (gsk.load_simplicial_complex, gsk.load_hypergraph, gsk.clique_promotion); on a network, use the simple and complex contagion models")
   if net.directed:
      raise ValueError("higher-order contagion on directed hypergraphs is not supported yet")
   if model != "SI" and not 0 < mu <= 1:
      raise ValueError(f"mu must be in (0, 1], got {mu}")
   if threshold is not None and not 0 < threshold <= 1:
      raise ValueError(f"threshold must be a share of the other nodes of a group in (0, 1], got {threshold}")
   if selection not in ("top", "weighted"):
      raise ValueError(f"unknown selection: {selection}")
   if runs < 1 or max_steps < 0:
      raise ValueError("runs must be positive and max_steps non negative")
   window = int(window or 0)
   if window < 0 or tolerance < 0:
      raise ValueError("window and tolerance must be non negative")
   comps = list(range(len(net.components))) if component is None else [component]
   off = np.cumsum([0] + [_size(net, c) for c in comps])
   n = int(off[-1])
   rs = np.random.default_rng(seed)
   eptr, enbr, lb1, nptr, ngrp, gptr, gmem, lb, need, everyone = _groups(net, comps, off, beta, threshold)
   ptr, nbr = _co_members(*everyone, n) if isinstance(strategy, str) and strategy == "friend_of_random" and immunize is not None else (np.zeros(n + 1, dtype = np.int64), np.zeros(0, dtype = np.int64))
   out, removed, imm = _initial(net, comps, off, model, seeds, ranking, selection, runs, attribute, resume, immunize, strategy, ptr, nbr, rs)
   B = min(runs, numba.get_num_threads())
   total, square = np.zeros((B, max_steps + 1, 3)), np.zeros((B, max_steps + 1, 3))
   ends, finals, last = np.zeros(runs, dtype = np.int64), np.zeros((runs, 3)), np.zeros(n, dtype = np.uint8)
   states = rs.integers(1, 2 ** 63, size = runs, dtype = np.uint64)
   _run_groups(eptr, enbr, lb1, nptr, ngrp, gptr, gmem, lb, need, out, removed, imm, 0. if model == "SI" else float(mu), model == "SIS", int(max_steps), window, float(tolerance), states, total, square, ends, finals, last)
   total, square = _sums(total, square, ends, finals, n)
   return _table(net, comps, off, component, attribute, last, total, square, runs, model)


@_higher_order(hypergraph = True, simplicial = True)
def higher_order_si_model(net, beta, threshold = None, seeds = 1, ranking = None, selection = "top", runs = 100, max_steps = 1000, component = 0, seed = None, attribute = "state", resume = False, immunize = None, strategy = "random"):
   """
   SI model of higher-order contagion on a simplicial complex or a hypergraph: a susceptible node
   is infected through each group of order :math:`\\omega` (a simplex, or a hyperedge of
   :math:`\\omega + 1` nodes) whose other nodes are all infected, with probability
   :math:`\\beta_\\omega` (see the module description), and stays infected. Seeds, immunization,
   stored states and the table are as in :func:`~graphoskop.si_model`.

   :param net: A simplicial complex or a hypergraph (undirected).
   :param beta: Infection probabilities by order: a sequence :math:`(\\beta_1, \\beta_2, \\ldots)`
      (edges, triangles, ...), a dictionary from order to probability, or one probability for all
      orders. Orders without a probability (or with 0) do not infect; on a simplicial complex,
      simplices of higher orders are not enumerated.
   :param threshold: ``None`` for groups that infect when all their other nodes are infected
      (Iacopini et al.), or the share of the other nodes that must be infected, in (0, 1] (at
      least one).
   :param seeds: Number of infected nodes at step 0, or their share if below 1.
   :param ranking: Node values to choose the seeds from, or ``None`` for random seeds (see
      :func:`~graphoskop.si_model`).
   :param selection: ``"top"`` or ``"weighted"`` (see :func:`~graphoskop.si_model`).
   :param runs: Number of independent runs.
   :param max_steps: Largest number of steps of a run.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param attribute: Qualitative node attribute storing the states at the end of the last run, or
      ``None``.
   :param resume: Whether to start every run from the states stored in ``attribute``.
   :param immunize: Nodes to immunize at step 0 (see :func:`~graphoskop.si_model`); acquaintance
      immunization picks a node sharing a simplex or hyperedge with a random node.
   :param strategy: How to choose the immunized nodes (see :func:`~graphoskop.si_model`).
   :return: :class:`pandas.DataFrame` indexed by ``step``, with columns ``S``, ``S_low``,
      ``S_high``, ``I``, ``I_low`` and ``I_high``: the mean fraction of nodes in each state and
      its 95% confidence interval.
   :raises ValueError: If ``net`` is a plain network or a directed hypergraph, or a parameter is
      not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_simplicial_complex("groups.tsv")
      si = gsk.higher_order_si_model(net, [0.05, 0.3], seeds = 5)       # edges, then triangles
   """
   return _simulate_groups(net, "SI", beta, 0., threshold, seeds, ranking, selection, runs, max_steps, component, seed, attribute, resume, 0, 0., immunize, strategy)


@_higher_order(hypergraph = True, simplicial = True)
def higher_order_sis_model(net, beta, mu, threshold = None, seeds = 1, ranking = None, selection = "top", runs = 100, max_steps = 1000, component = 0, seed = None, attribute = "state", resume = False, window = 100, tolerance = 0.01, immunize = None, strategy = "random"):
   """
   SIS model of higher-order contagion on a simplicial complex or a hypergraph: the simplicial
   contagion model of Iacopini, Petri, Barrat and Latora (2019). A susceptible node is infected
   through each group of order :math:`\\omega` whose other nodes are all infected, with probability
   :math:`\\beta_\\omega` (see the module description); infected nodes recover with probability
   ``mu`` per step and become susceptible again. Runs stop at the endemic plateau as in
   :func:`~graphoskop.sis_model`, and the other parameters are those of
   :func:`higher_order_si_model`.

   The rescaled infectivities of the paper are :math:`\\lambda = \\beta_1 \\langle k \\rangle / \\mu`
   and :math:`\\lambda_\\Delta = \\beta_2 \\langle k_\\Delta \\rangle / \\mu`, with
   :math:`\\langle k \\rangle` the mean degree (:func:`~graphoskop.average_degree`) and
   :math:`\\langle k_\\Delta \\rangle` the mean number of triangles of a node
   (:func:`~graphoskop.generalized_degree` with ``d = 2``). For :math:`\\lambda_\\Delta > 1`, the
   endemic state appears abruptly, below :math:`\\lambda = 1`, and only from enough initial seeds.

   :param net: A simplicial complex or a hypergraph (undirected).
   :param beta: Infection probabilities by order (see :func:`higher_order_si_model`).
   :param mu: Recovery probability per step.
   :param threshold: ``None`` (all the other nodes), or the share of the other nodes of a group
      that must be infected (see :func:`higher_order_si_model`).
   :param seeds: Number of infected nodes at step 0, or their share if below 1.
   :param ranking: Node values to choose the seeds from, or ``None``.
   :param selection: ``"top"`` or ``"weighted"``.
   :param runs: Number of independent runs.
   :param max_steps: Largest number of steps of a run.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param attribute: Qualitative node attribute storing the final states of the last run, or
      ``None``.
   :param resume: Whether to start every run from the states stored in ``attribute``.
   :param window: Steps of the plateau test (see :func:`~graphoskop.sis_model`), or 0 for none.
      An epidemic dying out slowly with few infected nodes (as in the bistable region, from few
      seeds) can pass the test at a noisy low level: use 0 to follow it to the end.
   :param tolerance: Relative tolerance of the plateau test.
   :param immunize: Nodes to immunize at step 0.
   :param strategy: How to choose them.
   :return: :class:`pandas.DataFrame` indexed by ``step``, with columns ``S``, ``S_low``,
      ``S_high``, ``I``, ``I_low`` and ``I_high``.
   :raises ValueError: If ``net`` is a plain network or a directed hypergraph, or a parameter is
      not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_simplicial_complex("groups.tsv")
      mu, k = 0.05, gsk.average_degree(net)
      kt = 3 * net.number_of_simplices(2) / net.number_of_nodes()        # mean triangles per node
      lam, lam_t = 0.8, 2.5                                              # bistable region
      beta = [lam * mu / k, lam_t * mu / kt]
      low = gsk.higher_order_sis_model(net, beta, mu, seeds = 0.01, window = 0)     # dies out
      high = gsk.higher_order_sis_model(net, beta, mu, seeds = 0.4)     # endemic
   """
   return _simulate_groups(net, "SIS", beta, mu, threshold, seeds, ranking, selection, runs, max_steps, component, seed, attribute, resume, window, tolerance, immunize, strategy)


@_higher_order(hypergraph = True, simplicial = True)
def higher_order_sir_model(net, beta, mu, threshold = None, seeds = 1, ranking = None, selection = "top", runs = 100, max_steps = 1000, component = 0, seed = None, attribute = "state", resume = False, immunize = None, strategy = "random"):
   """
   SIR model of higher-order contagion on a simplicial complex or a hypergraph: infection through
   groups as in :func:`higher_order_si_model`; infected nodes recover with probability ``mu`` per
   step and become removed (R), as in :func:`~graphoskop.sir_model`.

   :param net: A simplicial complex or a hypergraph (undirected).
   :param beta: Infection probabilities by order (see :func:`higher_order_si_model`).
   :param mu: Recovery probability per step.
   :param threshold: ``None`` (all the other nodes), or the share of the other nodes of a group
      that must be infected.
   :param seeds: Number of infected nodes at step 0, or their share if below 1.
   :param ranking: Node values to choose the seeds from, or ``None``.
   :param selection: ``"top"`` or ``"weighted"``.
   :param runs: Number of independent runs.
   :param max_steps: Largest number of steps of a run.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param attribute: Qualitative node attribute storing the final states of the last run, or
      ``None``.
   :param resume: Whether to start every run from the states stored in ``attribute``.
   :param immunize: Nodes to immunize at step 0.
   :param strategy: How to choose them.
   :return: :class:`pandas.DataFrame` indexed by ``step``, with columns ``S``, ``I`` and ``R``
      and their 95% confidence intervals (``_low``, ``_high``).
   :raises ValueError: If ``net`` is a plain network or a directed hypergraph, or a parameter is
      not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_hypergraph("meetings.tsv")
      sir = gsk.higher_order_sir_model(net, {1: 0.2, 2: 0.2, 3: 0.4}, 0.1, threshold = 0.5, seeds = 3)
   """
   return _simulate_groups(net, "SIR", beta, mu, threshold, seeds, ranking, selection, runs, max_steps, component, seed, attribute, resume, 0, 0., immunize, strategy)
