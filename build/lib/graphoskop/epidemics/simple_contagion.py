"""
Simple contagion: compartmental models of epidemics where one infected neighbor is enough to
infect a node (as opposed to complex contagion, which needs reinforcement from several
neighbors). Every node is in a compartment, its state:

* :func:`si_model`: Susceptible (S) or Infected (I); infected nodes stay infected forever;
* :func:`sis_model`: infected nodes recover with probability ``mu`` per step and become
  susceptible again;
* :func:`sir_model`: infected nodes recover with probability ``mu`` per step and become Removed
  (R): they cannot be infected again, nor infect.

Simulations run in discrete, synchronous time steps. At each step, every susceptible node with
at least one infected neighbor becomes infected with probability ``beta``, with one draw per node
per step, whatever the number of its infected neighbors; at the same time every node infected at
the start of the step recovers with probability ``mu``. A node infected in a step can recover
from the next one. In directed networks the infection follows the direction of the edges, from
source to target; undirected edges carry it both ways. Edge weights, parallel edges and self
loops do not matter. In multilayer networks the models run on the supra-graph: copies are nodes,
and the infection also spreads along couplings and other interlayer edges.

The infection starts from seed nodes, chosen at random or from a ranking of the nodes (e.g. a
centrality), and many independent runs are simulated in parallel and averaged. After the
simulation, the state of every node at the end of the last run is stored as a qualitative node
attribute (``"S"``, ``"I"`` or ``"R"``), from which a later simulation can start, e.g. to follow
an epidemic one step at a time.

All the models can immunize nodes at step 0 (``immunize``): immunized nodes are never infected,
so they never infect either. They are given as node ids, or as a number of nodes (a fraction of
them if below 1) chosen by ``strategy``:

* ``"random"``: nodes drawn uniformly at random, anew in every run;
* ``"friend_of_random"``: acquaintance immunization (Cohen, Havlin and ben-Avraham, 2003): a node
  drawn uniformly at random names one of its neighbors (out-neighbors in directed networks),
  drawn uniformly, which is immunized; drawn again until enough distinct nodes are immunized,
  anew in every run. By the friendship paradox, neighbors of random nodes have more connections
  than random nodes, so this immunizes hubs without knowing the network;
* node values, such as a centrality or :func:`~graphoskop.degree` with ``distribution = None``
  (vector, dictionary or :class:`pandas.Series` by node id, or the name of a numeric node
  attribute): the nodes with the largest values (targeted immunization).

Seeds are drawn among the nodes that are not immunized. Immunized nodes are in none of the states
S, I and R of the table: S + I + R is 1 minus the share of immunized nodes. In the stored states
they are ``"V"`` (vaccinated), which every model reads back when resuming. When resuming, only
susceptible nodes can be immunized (infected and removed nodes given as ids are skipped), so a
vaccination campaign can start in the middle of an epidemic.
"""

import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from numba import njit, prange
from scipy.stats import t as student

from ..ergms import _random

#: Labels of the states, by code.
STATES = ("S", "I", "R")

#: Label of every state code of the engines: 3 is a susceptible node in the exposed list, 4 an
#: immunized node.
LABELS = np.array(["S", "I", "R", "S", "V"], dtype = object)


@njit(cache = True, parallel = True)
def _draw_seeds(n, w, order, k, excl, rng, out):
   """
   Seed nodes of every run, without repetition and outside the excluded nodes of the run, in
   parallel over the runs: the first ``k`` nodes of ``order`` if it is not empty, otherwise
   uniformly at random if ``w`` is empty, otherwise proportionally to the weights ``w``
   (Efraimidis and Spirakis, 2006: the ``k`` nodes with the largest keys :math:`\\log(u) / w`,
   :math:`u` uniform).

   :param n: Number of nodes.
   :param w: Non negative weight of every node, or an empty array.
   :param order: Nodes in order of preference, or an empty array.
   :param k: Number of seeds per run.
   :param excl: Excluded nodes (e.g. immunized): run ``r`` uses row ``r % len(excl)``; no rows
      for none.
   :param rng: xorshift state of every run.
   :param out: Seeds, runs by ``k``, written in place.
   """
   for r in prange(out.shape[0]):
      st = rng[r:r + 1]
      taken = np.zeros(n, dtype = np.bool_)
      if excl.shape[0] > 0:
         for v in excl[r % excl.shape[0]]:
            taken[v] = True
      c = 0
      if order.shape[0] > 0:
         for v in order:
            if c < k and not taken[v]:
               out[r, c] = v
               c += 1
      elif w.shape[0] == 0:   # rejection of repeated and excluded nodes
         while c < k:
            v = min(int(_random(st) * n), n - 1)
            if not taken[v]:
               taken[v] = True
               out[r, c] = v
               c += 1
      else:
         key = np.empty(n)
         for i in range(n):
            key[i] = np.log(_random(st) + 1e-300) / w[i] if w[i] > 0 and not taken[i] else -np.inf
         thr = np.partition(key, n - k)[n - k]
         for i in range(n):
            if key[i] >= thr and c < k:
               out[r, c] = i
               c += 1


@njit(cache = True, parallel = True)
def _draw_immune(ptr, nbr, friend, k, excl, rng, out):
   """
   Immunized nodes of every run, without repetition and outside the excluded nodes, in parallel
   over the runs: uniformly at random, or (``friend``) a random out-neighbor of a node drawn
   uniformly at random, drawn again until ``k`` distinct nodes are found (acquaintance
   immunization).

   :param ptr: CSR row pointers of the out edges.
   :param nbr: CSR targets of the out edges.
   :param friend: Whether to immunize friends of random nodes.
   :param k: Number of immunized nodes per run.
   :param excl: Whether each node is excluded (e.g. already infected).
   :param rng: xorshift state of every run.
   :param out: Immunized nodes, runs by ``k``, written in place.
   """
   n = ptr.shape[0] - 1
   for r in prange(out.shape[0]):
      st = rng[r:r + 1]
      taken = excl.copy()
      c = 0
      while c < k:
         v = min(int(_random(st) * n), n - 1)
         if friend:
            d = ptr[v + 1] - ptr[v]
            if d == 0:
               continue
            v = nbr[ptr[v] + min(int(_random(st) * d), d - 1)]
         if not taken[v]:
            taken[v] = True
            out[r, c] = v
            c += 1


@njit(cache = True, parallel = True)
def _run_numba(ptr, nbr, seeds, removed, immune, beta, mu, sis, kind, need, pe, decay, max_steps, window, tolerance, rng, total, square, ends, finals, last, ev, nev):
   """
   Runs of the SI, SIS or SIR model, in parallel: thread ``b`` simulates runs ``b``, ``b + B``,
   ... and adds the number of nodes in each state at each step to ``total[b]`` (and its square to
   ``square[b]``), until nothing can change any more, until the number of infected nodes reaches
   a plateau (see :func:`sis_model`), or ``max_steps`` steps.

   Every node keeps the number of its infected in-neighbors, updated only when a node is
   infected or recovers (scanning its out edges), and the susceptible nodes with infected
   neighbors are kept in a list (state 3, cleaned lazily when they lose them), so a step costs
   time proportional to the nodes that can change and to the edges of the nodes that changed. In
   the simple SI model nobody recovers, and the counts are not kept. In complex contagion
   (:mod:`~graphoskop.epidemics.complex_contagion`) a susceptible node is listed only while it has
   at least ``need`` infected neighbors. In independent and decreasing cascades, a newly infected
   node attempts once to infect each susceptible out-neighbor, when its edges are scanned, and the
   list holds the nodes whose attempts succeeded, infected at the next step.

   :param ptr: CSR row pointers of the out edges.
   :param nbr: CSR targets of the out edges.
   :param seeds: Infected nodes at step 0: run ``r`` uses row ``r % len(seeds)``.
   :param removed: Removed nodes at step 0 (every run).
   :param immune: Immunized nodes (state 4: never infected): run ``r`` uses row
      ``r % len(immune)``; every row has the same length.
   :param beta: Infection probability.
   :param mu: Recovery probability (0 for SI).
   :param sis: Whether recovered nodes become susceptible (SIS) or removed (SIR).
   :param kind: Reinforcement: 0 simple, 1 classical, 2 threshold or cascade (``need``), 3
      independent cascade, 4 decreasing cascade.
   :param need: Number of infected neighbors needed by each node to be infected (int32; kinds 1
      and 2), or an empty array.
   :param pe: Infection probability of every out edge, in CSR order (kinds 3 and 4), or an empty
      array to use ``beta`` for all edges.
   :param decay: Factor of the probability for every earlier failed attempt on a node (kind 4).
   :param max_steps: Largest number of steps.
   :param window: Steps of the plateau test, or 0 for none.
   :param tolerance: Relative tolerance of the plateau test.
   :param rng: xorshift state of every run.
   :param total: Threads by ``max_steps + 1`` by 3 (S, I, R), written in place.
   :param square: Same shape as ``total``, written in place.
   :param ends: Last step of every run, written in place.
   :param finals: Number of nodes in each state at the end of every run (mean over the last
      ``window`` steps if the run stopped at a plateau), written in place.
   :param last: State of every node at the end of the last run, written in place.
   :param ev: Infections of the last run (cascades), as rows ``(origin, target, step)``, written
      in place up to its length (no rows: not recorded).
   :param nev: Number of infections of the last run (one value), written in place, even beyond
      the length of ``ev``.
   """
   R, B, n = rng.shape[0], total.shape[0], ptr.shape[0] - 1
   complex_, ic = kind > 0, kind >= 3
   need_ = complex_ and not ic    # listed only with enough infected neighbors
   counts = (mu > 0 or complex_) and not ic   # simple SI: exposed nodes stay exposed, no counts needed
   marks = mu > 0 or complex_     # nodes can be touched several times
   for b in prange(B):
      state = np.zeros(n, dtype = np.uint8)   # 0 S, 1 I, 2 R, 3 S in the exposed list
      cnt = np.zeros(n, dtype = np.int32)     # infected in-neighbors
      mark = np.zeros(n, dtype = np.bool_)    # in the touched list (SIS, SIR, complex), reset after the run
      fail = np.zeros(n, dtype = np.int32)    # failed attempts on each node (decreasing cascade)
      front = np.empty(n, dtype = np.int64)   # exposed: susceptible with infected neighbors (and stale entries)
      inf = np.empty(n, dtype = np.int64)     # infected
      new = np.empty(n, dtype = np.int64)     # infected in the last step
      rec = np.empty(n, dtype = np.int64)     # recovered in the last step
      touched = np.empty(n, dtype = np.int64)
      cum = np.zeros(max_steps + 2, dtype = np.int64)   # cumulative number of infected nodes, for the plateau test
      for r in range(b, R, B):
         st = rng[r:r + 1]
         nt, nf, ni, nn, nr = 0, 0, 0, 0, 0
         for v in removed:
            state[v] = 2
            mark[v] = True
            touched[nt] = v
            nt += 1
         nV = immune.shape[1]
         for v in immune[r % immune.shape[0]]:
            state[v] = 4
            mark[v] = True
            touched[nt] = v
            nt += 1
         for v in seeds[r % seeds.shape[0]]:
            state[v] = 1
            new[nn] = v
            nn += 1
            if not mark[v]:
               mark[v] = True
               touched[nt] = v
               nt += 1
         nR, t, m2 = removed.shape[0], 0, 0.
         while True:
            for i in range(nn):   # newly infected: counts of their neighbors, new exposed nodes
               u = new[i]
               inf[ni] = u
               ni += 1
               for e in range(ptr[u], ptr[u + 1]):
                  v = nbr[e]
                  if counts:
                     cnt[v] += 1
                  hit = state[v] == 0 and (not need_ or cnt[v] >= need[v])
                  if hit and ic:   # one attempt of u on v
                     q = pe[e] if pe.shape[0] > 0 else beta
                     hit = _random(st) < (q * decay ** fail[v] if kind == 4 else q)
                     if not hit and kind == 4:
                        fail[v] += 1
                        if not mark[v]:
                           mark[v] = True
                           touched[nt] = v
                           nt += 1
                  if hit:
                     if ic and r == R - 1 and ev.shape[0] > 0:   # who infected whom, in the last run
                        if nev[0] < ev.shape[0]:
                           ev[nev[0], 0], ev[nev[0], 1], ev[nev[0], 2] = u, v, t + 1
                        nev[0] += 1
                     state[v] = 3
                     front[nf] = v
                     nf += 1
                     if not (marks and mark[v]):   # simple SI: nodes leave S once, no marks needed
                        if marks:
                           mark[v] = True
                        touched[nt] = v
                        nt += 1
            for i in range(nr):   # newly recovered
               u = rec[i]
               for e in range(ptr[u], ptr[u + 1]):
                  cnt[nbr[e]] -= 1
            if sis and not ic:
               for i in range(nr):
                  u = rec[i]
                  if cnt[u] >= (need[u] if need_ else 1) and state[u] == 0:
                     state[u] = 3
                     front[nf] = u
                     nf += 1
            total[b, t, 0] += n - ni - nR - nV   # integers: exact sums, whatever the order of the runs
            total[b, t, 1] += ni
            total[b, t, 2] += nR
            square[b, t, 0] += (n - ni - nR - nV) * (n - ni - nR - nV)
            square[b, t, 1] += ni * ni
            square[b, t, 2] += nR * nR
            cum[t + 1] = cum[t] + ni
            plateau = False
            if window > 0 and t + 1 >= 2 * window:   # mean infected in the last window against the window before
               m2 = (cum[t + 1] - cum[t + 1 - window]) / window
               m1 = (cum[t + 1 - window] - cum[t + 1 - 2 * window]) / window
               plateau = m2 > 0 and abs(m2 - m1) <= tolerance * m2
            if plateau or (nf == 0 and (mu == 0 or ni == 0)) or t == max_steps:
               break
            t += 1
            nn, keep = 0, 0
            for i in range(nf):   # infections, from the states at the start of the step
               v = front[i]
               if counts and cnt[v] < (need[v] if need_ else 1):   # lost its infected neighbors
                  state[v] = 0
               elif ic or _random(st) < (1 - (1 - beta) ** cnt[v] if kind == 1 else beta):   # cascades: decided
                  new[nn] = v
                  nn += 1
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
               fail[new[i]] = 0
            for i in range(nr):
               state[rec[i]] = 0 if sis else 2
            if not sis:
               nR += nr
         ends[r] = t
         finals[r, 0], finals[r, 1], finals[r, 2] = n - ni - nR - nV, ni, nR
         if plateau:
            finals[r, 0], finals[r, 1] = n - nR - nV - m2, m2
         if r == R - 1:
            for v in range(n):
               last[v] = 0 if state[v] == 3 else state[v]
         for i in range(nt):
            v = touched[i]
            state[v], cnt[v], mark[v], fail[v] = 0, 0, False, 0


def _run_torch(ptr, nbr, seeds, removed, immune, beta, mu, sis, kind, need, pe, decay, max_steps, window, tolerance, runs, seed, batch, record = False):
   """
   Runs of the SI, SIS or SIR model on the device of ``ptr``, all the runs of a batch at once:
   node ``v`` of run ``r`` is entry ``r * n + v`` of the state vectors, and every step draws all
   the exposed and infected (run, node) pairs together (same model and bookkeeping as
   :func:`_run_numba`). A run that reaches a plateau is frozen: its entries leave the lists, and
   it counts with its mean over the last ``window`` steps.

   :param ptr: CSR row pointers of the out edges (tensor).
   :param nbr: CSR targets of the out edges (tensor).
   :param seeds: Infected nodes at step 0, run ``r`` uses row ``r % len(seeds)`` (tensor).
   :param removed: Removed nodes at step 0 (tensor).
   :param immune: Immunized nodes, run ``r`` uses row ``r % len(immune)`` (tensor).
   :param beta: Infection probability.
   :param mu: Recovery probability (0 for SI).
   :param sis: Whether recovered nodes become susceptible (SIS) or removed (SIR).
   :param kind: Reinforcement code, as in :func:`_run_numba`.
   :param need: Number of infected neighbors needed by each node (tensor), or an empty tensor.
   :param pe: Infection probability of every out edge, in CSR order (tensor), or an empty tensor.
   :param decay: Factor of the probability for every earlier failed attempt (decreasing cascade).
   :param max_steps: Largest number of steps.
   :param window: Steps of the plateau test, or 0 for none.
   :param tolerance: Relative tolerance of the plateau test.
   :param runs: Number of runs.
   :param seed: Seed of the random numbers.
   :param batch: Number of runs simulated at once.
   :param record: Whether to record who infected whom in the last run (cascades).
   :return: Sums over the runs of the fraction of nodes in each state at each step, and of their
      squares (steps by 3, finished runs count with their final value), as NumPy arrays, the state
      of every node at the end of the last run, and its infections as rows ``(origin, target,
      step)`` (NumPy array, empty if not recorded).
   """
   dev, n = ptr.device, ptr.shape[0] - 1
   gen = torch.Generator(device = dev)
   gen.manual_seed(seed)
   empty = torch.zeros(0, dtype = torch.long, device = dev)
   ic = kind >= 3
   counts = (mu > 0 or kind > 0) and not ic   # simple SI: exposed nodes stay exposed, no counts needed
   need = need if need.shape[0] else torch.ones(n, dtype = torch.int32, device = dev)

   def neighbors(x):   # out-neighbors of (run, node) entries, as entries, the CSR positions of the edges, and their origins
      u = x % n
      deg = ptr[u + 1] - ptr[u]
      rep = torch.repeat_interleave(torch.arange(x.shape[0], device = dev), deg)
      pos = ptr[u][rep] + torch.arange(rep.shape[0], device = dev) - (torch.cumsum(deg, 0) - deg)[rep]
      return nbr[pos] + (x // n)[rep] * n, pos, x[rep]

   total, square, events = [], [], []
   for a in range(0, runs, batch):
      R = min(batch, runs - a)
      base = torch.arange(R, device = dev) * n
      state = torch.zeros(R * n, dtype = torch.uint8, device = dev)
      cnt = torch.zeros(R * n, dtype = torch.int32, device = dev)
      inl = torch.zeros(R * n, dtype = torch.bool, device = dev)
      fail = torch.zeros(R * n if kind == 4 else 0, dtype = torch.int32, device = dev)
      state[(removed[None] + base[:, None]).reshape(-1)] = 2
      state[(immune[torch.arange(a, a + R, device = dev) % immune.shape[0]] + base[:, None]).reshape(-1)] = 4
      nV = float(immune.shape[1])
      new = (seeds[torch.arange(a, a + R, device = dev) % seeds.shape[0]] + base[:, None]).reshape(-1)
      state[new] = 1
      inf, rec, front = empty, empty, empty
      nI = torch.zeros(R, dtype = torch.float64, device = dev)
      nR = torch.full((R,), float(removed.shape[0]), dtype = torch.float64, device = dev)
      count = lambda x: torch.bincount(x // n, minlength = R)   # per run
      cum = torch.zeros((max_steps + 2, R), dtype = torch.float64, device = dev)   # for the plateau test
      done = torch.zeros(R, dtype = torch.bool, device = dev)
      frozen = torch.zeros((3, R), dtype = torch.float64, device = dev)
      xs, t, last_end = [], 0, None
      while True:
         nI += count(new) - count(rec)
         w, pos, org = neighbors(new)
         cand = torch.cat([w, rec]) if sis and not ic else w
         if mu > 0:
            inf = torch.cat([inf, new])
         if counts:
            cnt.index_add_(0, w, torch.ones_like(w, dtype = torch.int32))
            wr = neighbors(rec)[0]
            cnt.index_add_(0, wr, torch.full_like(wr, -1, dtype = torch.int32))
            cand = cand[cnt[cand] >= need[cand % n]]
         if ic:   # one attempt per edge of a newly infected node, on susceptible neighbors
            m = (state[w] == 0) & ~inl[w]
            cand, org = w[m], org[m]
            q = pe[pos[m]] if pe.shape[0] else torch.full(cand.shape, beta, dtype = torch.float64, device = dev)
            u = torch.rand(cand.shape[0], generator = gen, device = dev)
            if kind == 4:   # simultaneous attempts on a node in sequence: attempt j after f + j failures
               order = torch.argsort(cand, stable = True)
               cand, q, u, org = cand[order], q[order], u[order], org[order]
               _, inv, size = torch.unique_consecutive(cand, return_inverse = True, return_counts = True)
               j = torch.arange(cand.shape[0], device = dev) - (torch.cumsum(size, 0) - size)[inv]
               q = q * decay ** (fail[cand] + j).double()
            hit = u < q
            if kind == 4:
               fail.index_add_(0, cand[~hit], torch.ones_like(cand[~hit], dtype = torch.int32))
            if record and a + R == runs:   # who infected whom in the last run: the first successful attempt on each node
               hv, ho = cand[hit], org[hit]
               mine = hv // n == R - 1
               hv, ho = hv[mine], ho[mine]
               o = torch.argsort(hv, stable = True)
               hv, ho = hv[o], ho[o]
               first = torch.ones_like(hv, dtype = torch.bool)
               first[1:] = hv[1:] != hv[:-1]
               events.append(torch.stack([ho[first] % n, hv[first] % n, torch.full_like(hv[first], t + 1)], 1))
            cand = cand[hit]
         cand = torch.unique(cand[(state[cand] == 0) & ~inl[cand]])
         inl[cand] = True
         front = torch.cat([front, cand])
         xs.append(torch.where(done, frozen, torch.stack([n - nI - nR - nV, nI, nR])) / n)
         cum[t + 1] = cum[t] + nI
         if window > 0 and t + 1 >= 2 * window:   # mean infected in the last window against the window before
            m2 = (cum[t + 1] - cum[t + 1 - window]) / window
            m1 = (cum[t + 1 - window] - cum[t + 1 - 2 * window]) / window
            stop = ~done & (m2 > 0) & ((m2 - m1).abs() <= tolerance * m2)
            if bool(stop.any()):
               frozen = torch.where(stop, torch.stack([n - nR - nV - m2, m2, nR]), frozen)
               if bool(stop[R - 1]):
                  last_end = t
               done |= stop
               inf, front = inf[~done[inf // n]], front[~done[front // n]]
         if (front.shape[0] == 0 and (mu == 0 or inf.shape[0] == 0)) or t == max_steps:
            break
         t += 1
         ok = state[front] == 0
         if counts:
            ok &= cnt[front] >= need[front % n]
         inl[front[~ok]] = False
         front = front[ok]
         p = 1 - (1 - beta) ** cnt[front].double() if kind == 1 else beta
         hit = torch.ones(front.shape[0], dtype = torch.bool, device = dev) if ic else torch.rand(front.shape[0], generator = gen, device = dev) < p   # cascades: decided
         new, front = front[hit], front[~hit]
         inl[new] = False
         rec = empty
         if mu > 0:
            hit = torch.rand(inf.shape[0], generator = gen, device = dev) < mu
            rec, inf = inf[hit], inf[~hit]
         state[new] = 1
         if kind == 4:
            fail[new] = 0
         state[rec] = 0 if sis else 2
         if not sis:
            nR += count(rec)
      X = torch.stack(xs)   # steps by 3 by runs
      total.append(X.sum(2).cpu().numpy())
      square.append((X * X).sum(2).cpu().numpy())
   T = max(len(x) for x in total)
   pad = lambda x: np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)])   # finished batches stay at their final value
   ev = torch.cat(events).cpu().numpy() if events else np.zeros((0, 3), dtype = np.int64)
   ev = ev[ev[:, 2] <= (t if last_end is None else last_end)]   # attempts decided at the last step were never applied
   return sum(pad(x) for x in total), sum(pad(x) for x in square), state[(R - 1) * n:].cpu().numpy(), ev


def _node_values(net, comps, values):
   """
   Values of the nodes of some components, in the network wide node order.

   :param net: Network.
   :param comps: Component indices.
   :param values: Name of a numeric node attribute; dictionary or :class:`pandas.Series` from
      input node id to value (missing nodes get NaN); vector over the nodes of the component
      (:class:`torch.Tensor`, array or list); or a list of vectors, one per component of the
      network (as returned with ``component = None``), of which those of ``comps`` are used.
   :return: float64 :class:`numpy.ndarray`.
   :raises ValueError: If a vector does not have one value per node.
   """
   n = sum(net.components[c].num_nodes for c in comps)
   if isinstance(values, str):
      x = [net.get_node_attribute(values, c) for c in comps]
   elif isinstance(values, (dict, pd.Series)):
      ids = np.concatenate([net.node_ids[c] for c in comps])
      x = [pd.Series(ids).map(values).to_numpy(dtype = np.float64)]
   elif isinstance(values, (list, tuple)) and len(values) and (torch.is_tensor(values[0]) or np.ndim(values[0]) == 1):
      x = [values[c] for c in comps] if len(values) == len(net.components) else values
   else:
      x = [values]
   x = np.concatenate([v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v, dtype = np.float64) for v in x]).astype(np.float64)
   if x.shape != (n,):
      raise ValueError(f"expected one value per node ({n}), got {x.shape[0]}")
   return x


def _immunization(net, comps, off, immunize, strategy, ptr, nbr, excl, runs, rs):
   """
   Immunized nodes of every run.

   :param net: Network.
   :param comps: Component indices of the simulation.
   :param off: Position of the first node of each of ``comps``, then the number of nodes.
   :param immunize: ``None``; a number of nodes (a fraction of them if below 1); or node ids (an
      iterable, a single id, or a :class:`pandas.DataFrame` whose first column gives them).
   :param strategy: With a number: ``"random"``, ``"friend_of_random"``, or node values (as
      ``ranking`` in :func:`si_model`) whose largest ones are immunized.
   :param ptr: CSR row pointers of the out edges (NumPy).
   :param nbr: CSR targets of the out edges (NumPy).
   :param excl: Whether each node cannot be immunized (not susceptible when resuming).
   :param runs: Number of runs.
   :param rs: NumPy random generator.
   :return: Immunized nodes, one row per run (random strategies) or one row for all runs.
   :raises ValueError: If there are not enough nodes to immunize, if ``strategy`` is not valid, or
      if a node is not among the nodes simulated.
   """
   n = int(off[-1])
   none = np.zeros((1, 0), dtype = np.int64)
   if immunize is None:
      return none
   if isinstance(immunize, (int, float, np.number)) and not isinstance(immunize, bool):
      k = max(1, int(round(immunize * n))) if 0 < immunize < 1 else int(immunize)
      free = ~excl
      if k < 0 or k > free.sum():
         raise ValueError(f"cannot immunize {k} nodes: {int(free.sum())} nodes can be immunized")
      if k == 0:
         return none
      if isinstance(strategy, str) and strategy in ("random", "friend_of_random"):
         friend = strategy == "friend_of_random"
         if friend:   # nodes that are someone's friend
            free = free & (np.bincount(nbr, minlength = n) > 0)
            if k > free.sum():
               raise ValueError(f"cannot immunize {k} friends of random nodes: only {int(free.sum())} nodes are someone's neighbor")
         out = np.zeros((runs, k), dtype = np.int64)
         _draw_immune(ptr, nbr, friend, k, excl, rs.integers(1, 2 ** 63, size = runs, dtype = np.uint64), out)
         return out
      if isinstance(strategy, str) and strategy not in net.node_attributes:
         raise ValueError(f"unknown strategy: {strategy!r} (\"random\", \"friend_of_random\", or node values)")
      x = _node_values(net, comps, strategy)   # the largest values
      cand = np.flatnonzero(free)
      return cand[np.argsort(-np.nan_to_num(x[cand], nan = -np.inf), kind = "stable")[:k]][None]
   if isinstance(immunize, pd.DataFrame):
      immunize = immunize.iloc[:, 0]
   items = [immunize] if isinstance(immunize, (str, tuple)) else list(immunize)
   pos = []
   for x in items:
      for c, i in net._nodes(x):
         if c not in comps:
            raise ValueError(f"node {x!r} is not among the nodes simulated (component {c})")
         pos.append(off[comps.index(c)] + i)
   pos = np.unique(np.asarray(pos, dtype = np.int64))
   return pos[~excl[pos]][None]   # when resuming, infected and removed nodes are not immunized


def _simulate(net, model, beta, mu, seeds, ranking, selection, runs, max_steps, component, seed, engine, batch, attribute, resume, window = None, tolerance = 0., reinforcement = None, threshold = None, decay = 1., forest = False, immunize = None, strategy = "random"):
   """
   Body of the simple and complex contagion models.

   :param model: ``"SI"``, ``"SIS"`` or ``"SIR"``.
   :param reinforcement: ``None`` (simple contagion), ``"classical"``, ``"threshold"``,
      ``"cascade"``, ``"independent"`` or ``"decreasing"`` (see
      :mod:`~graphoskop.epidemics.complex_contagion`).
   :param threshold: With ``"threshold"`` and ``"cascade"``, the threshold of every node, or one
      for all.
   :param decay: With ``"decreasing"``, the factor of the probability for every earlier failed
      attempt on a node.
   :param forest: Whether to return also the forest of who infected whom in the last run
      (cascades only).
   :param immunize: Immunized nodes (see :func:`_immunization`).
   :param strategy: How to choose them.
   :return: The table of the states over time, and the forest if asked.
   """
   pairwise = not isinstance(beta, (int, float, np.number))   # a table of probabilities per pair
   if pairwise and reinforcement not in ("independent", "decreasing"):
      raise ValueError("a table of probabilities per pair of nodes needs reinforcement 'independent' or 'decreasing'")
   if not pairwise and not 0 < beta <= 1:
      raise ValueError(f"beta must be in (0, 1], got {beta}")
   if forest and reinforcement not in ("independent", "decreasing"):
      raise ValueError("forest needs reinforcement 'independent' or 'decreasing', where each infection has one origin")
   if reinforcement == "decreasing" and not 0 < decay <= 1:
      raise ValueError(f"decay must be in (0, 1], got {decay}")
   if model != "SI" and not 0 < mu <= 1:
      raise ValueError(f"mu must be in (0, 1], got {mu}")
   if selection not in ("top", "weighted"):
      raise ValueError(f"unknown selection: {selection}")
   if reinforcement not in (None, "classical", "threshold", "cascade", "independent", "decreasing"):
      raise ValueError(f"unknown reinforcement: {reinforcement}")
   if reinforcement in ("threshold", "cascade") and threshold is None:
      raise ValueError(f"reinforcement {reinforcement!r} needs a threshold")
   if runs < 1 or max_steps < 0:
      raise ValueError("runs must be positive and max_steps non negative")
   window = int(window or 0)
   if window < 0 or tolerance < 0:
      raise ValueError("window and tolerance must be non negative")
   comps = list(range(len(net.components))) if component is None else [component]
   sizes = [net.components[c].num_nodes for c in comps]
   off = np.cumsum([0] + sizes)
   n = int(off[-1])
   cuda = net.components[comps[0]].edge_index.device.type == "cuda"
   engine = engine or ("torch" if cuda else "numba")
   if engine not in ("numba", "torch"):
      raise ValueError(f"unknown engine: {engine}")
   rs = np.random.default_rng(seed)

   # out edges in CSR form
   ei = torch.cat([net.components[c].edge_index + int(o) for c, o in zip(comps, off)], 1)
   dev = ei.device if engine == "torch" else torch.device("cpu")
   s, t = ei.to(dev)
   order = torch.argsort(s, stable = True)
   nbr = t[order]
   ptr = torch.zeros(n + 1, dtype = torch.long, device = dev)
   ptr[1:] = torch.cumsum(torch.bincount(s, minlength = n), 0)

   # states at step 0: stored states (resume), immunized nodes, seeds
   removed, vres, excl = np.zeros(0, dtype = np.int64), np.zeros(0, dtype = np.int64), np.zeros(n, dtype = np.bool_)
   if resume:
      if attribute is None or attribute not in net.qualitative_node_attributes:
         raise ValueError(f"resume needs the states in a qualitative node attribute, but there is no {attribute!r} in net.qualitative_node_attributes: run a model on this network object first (without resume), or set the states with net.set_node_attribute")
      x = pd.Series(np.concatenate([net.get_node_attribute(attribute, c) for c in comps]), dtype = object)
      code = x.map({**{s: i for i, s in enumerate(STATES[:3 if model == "SIR" else 2])}, "V": 4})
      bad = x.notna() & code.isna()
      if bad.any():
         raise ValueError(f"{attribute!r} has value {x[bad].iloc[0]!r}, not a state of the {model} model")
      code = code.fillna(0).to_numpy(dtype = np.int64)   # missing: susceptible
      removed, vres, excl = np.flatnonzero(code == 2), np.flatnonzero(code == 4), code != 0
   imm = _immunization(net, comps, off, immunize, strategy, ptr.cpu().numpy(), nbr.cpu().numpy(), excl, runs, rs)
   imm = np.concatenate([np.broadcast_to(vres, (imm.shape[0], vres.shape[0])), imm], 1)
   if resume:
      out = np.flatnonzero(code == 1)[None]
   else:
      ki = imm.shape[1]
      k = max(1, int(round(seeds * n))) if seeds < 1 else int(seeds)
      if k > n - ki:
         raise ValueError(f"{k} seeds, but only {n - ki} nodes that are not immunized")
      out = np.zeros((runs, k), dtype = np.int64)
      x, order_ = np.zeros(0), np.zeros(0, dtype = np.int64)
      if ranking is not None:
         x = _node_values(net, comps, ranking)
         if selection == "top":
            order_, x = np.argsort(-np.nan_to_num(x, nan = -np.inf), kind = "stable"), np.zeros(0)
         else:
            x = np.nan_to_num(x, nan = 0.)
            if (x < 0).any():
               raise ValueError("weighted selection needs non negative ranking values")
            if (x > 0).sum() - ki < k:
               raise ValueError(f"{k} seeds, but only {int((x > 0).sum())} nodes with a positive ranking value, and {ki} immunized nodes")
      states = rs.integers(1, 2 ** 63, size = runs, dtype = np.uint64) if order_.shape[0] == 0 else np.ones(runs, dtype = np.uint64)
      _draw_seeds(n, x, order_, k, imm if ki else np.zeros((0, 0), dtype = np.int64), states, out)

   # infection probability of every edge, in CSR order, from a table per pair of nodes
   pe = np.zeros(0)
   if pairwise:
      a, b = s.cpu().numpy().copy(), t.cpu().numpy().copy()
      if isinstance(beta, str):   # edge attribute
         x = np.concatenate([net.get_edge_attribute(beta, c).double().cpu().numpy() for c in comps])
      elif isinstance(beta, pd.DataFrame):   # source, target, probability, by node id; missing pairs: 0
         ids = pd.Index(np.concatenate([net.node_ids[c] for c in comps]), dtype = object, tupleize_cols = False)
         i, j = ids.get_indexer(beta.iloc[:, 0].tolist()), ids.get_indexer(beta.iloc[:, 1].tolist())
         ok = (i >= 0) & (j >= 0)
         table = pd.Series(beta.iloc[:, 2].to_numpy(dtype = np.float64)[ok], index = i[ok] * n + j[ok])
         x = table[~table.index.duplicated(keep = "last")].reindex(a * n + b).fillna(0.).to_numpy()
      elif sp.issparse(beta):   # matrix, rows: origin, columns: target
         x = np.asarray(beta.tocsr()[a, b], dtype = np.float64).ravel()
      else:
         x = np.asarray(beta.cpu() if torch.is_tensor(beta) else beta, dtype = np.float64)[a, b]
      if not (np.isfinite(x) & (x >= 0) & (x <= 1)).all():
         raise ValueError("probabilities per pair of nodes must be in [0, 1]")
      pe = x[order.cpu().numpy()]

   # infected neighbors needed by each node: 1 (simple, classical), a count or a share of the in-degree
   need = np.zeros(0, dtype = np.int32)
   if reinforcement in ("classical", "threshold", "cascade"):
      if reinforcement == "classical":
         x = np.ones(n)
      else:
         x = np.full(n, float(threshold)) if isinstance(threshold, (int, float, np.number)) else _node_values(net, comps, threshold)
      if reinforcement == "threshold" and not (np.isfinite(x) & (x >= 1)).all():
         raise ValueError("threshold: thresholds must be counts of at least 1")
      elif reinforcement == "cascade":
         if not (np.isfinite(x) & (x > 0) & (x <= 1)).all():
            raise ValueError("cascade: thresholds must be shares of the neighbors in (0, 1]")
         x = np.maximum(x * torch.bincount(t, minlength = n).cpu().numpy() - 1e-9, 1)   # share of the in-degree, as a count
      need = np.minimum(np.ceil(x), 2 ** 31 - 1).astype(np.int32)

   # simulation
   mu, sis = (0., False) if model == "SI" else (float(mu), model == "SIS")
   kind, beta = {None: 0, "classical": 1, "threshold": 2, "cascade": 2, "independent": 3, "decreasing": 4}[reinforcement], 0. if pairwise else float(beta)
   if engine == "numba":
      B = min(runs, numba.get_num_threads())
      total, square = np.zeros((B, max_steps + 1, 3)), np.zeros((B, max_steps + 1, 3))
      ends, finals, last = np.zeros(runs, dtype = np.int64), np.zeros((runs, 3)), np.zeros(n, dtype = np.uint8)
      states = rs.integers(1, 2 ** 63, size = runs, dtype = np.uint64)
      ev, nev = np.zeros((n if forest else 0, 3), dtype = np.int64), np.zeros(1, dtype = np.int64)
      args = (ptr.numpy(), nbr.numpy())
      rest = (beta, mu, sis, kind, need, pe, float(decay), int(max_steps), window, float(tolerance))
      last_state = states[-1:].copy()   # the kernel advances the states
      _run_numba(*args, out, removed, imm, *rest, states, total, square, ends, finals, last, ev, nev)
      if nev[0] > ev.shape[0]:   # SIS: more infections than nodes; the last run again, alone, with room for all
         ev, one = np.zeros((nev[0], 3), dtype = np.int64), np.zeros(1, dtype = np.int64)
         _run_numba(*args, out[(runs - 1) % out.shape[0]][None], removed, imm[(runs - 1) % imm.shape[0]][None], *rest, last_state, np.zeros((1, max_steps + 1, 3)), np.zeros((1, max_steps + 1, 3)), np.zeros(1, dtype = np.int64), np.zeros((1, 3)), np.zeros(n, dtype = np.uint8), ev, one)
      ev = ev[:nev[0]]
      ev = ev[ev[:, 2] <= ends[-1]]   # attempts decided at the last step were never applied
      T = int(ends.max())
      pad = np.zeros((2, T + 2, 3))   # finished runs keep their final value until step T
      np.add.at(pad[0], ends + 1, finals)
      np.add.at(pad[1], ends + 1, finals * finals)
      pad = np.cumsum(pad, 1)[:, :T + 1]
      total, square = (total.sum(0)[:T + 1] + pad[0]) / n, (square.sum(0)[:T + 1] + pad[1]) / n ** 2
   else:
      tensor = lambda a: torch.as_tensor(a, device = dev)
      total, square, last, ev = _run_torch(ptr, nbr, tensor(out), tensor(removed), tensor(imm), beta, mu, sis, kind, tensor(need), tensor(pe), float(decay), int(max_steps), window, float(tolerance), runs, int(rs.integers(2 ** 62)), batch or max(1, 2 ** 27 // n), forest)

   # final states of the last run, and the table
   if attribute is not None:
      labels = np.split(LABELS[last], off[1:-1])
      net.set_node_attribute(attribute, labels if component is None else labels[0], component)
   mean = total / runs
   if runs > 1:
      sd = np.sqrt(np.maximum(square - runs * mean * mean, 0) / (runs - 1))
      h = student.ppf(0.975, runs - 1) * sd / np.sqrt(runs)
   else:
      h = np.full(mean.shape, np.nan)
   cols = {}
   for j, x in enumerate(STATES[:3 if model == "SIR" else 2]):
      cols[x], cols[x + "_low"], cols[x + "_high"] = mean[:, j], mean[:, j] - h[:, j], mean[:, j] + h[:, j]
   table = pd.DataFrame(cols, index = pd.RangeIndex(len(mean), name = "step"))
   if not forest:
      return table
   return table, _forest(net, comps, out[(runs - 1) % out.shape[0]], ev)


def _forest(net, comps, roots, ev):
   """
   Network of who infected whom: an edge from the origin to the target of every infection, with
   the step of the infection, and the infected nodes with their attributes.

   :param net: Network.
   :param comps: Component indices of the simulation.
   :param roots: Infected nodes at step 0 (positions in the nodes of ``comps``).
   :param ev: Infections, rows ``(origin, target, step)`` (positions in the nodes of ``comps``).
   :return: Directed :class:`~graphoskop.Network`, on the device of ``net``.
   """
   from ..network import load_network
   from ..transformations import _tables
   ids = np.empty(sum(len(net.node_ids[c]) for c in comps), dtype = object)
   ids[:] = [x for c in comps for x in net.node_ids[c].tolist()]
   edges = pd.DataFrame({"source": ids[ev[:, 0]], "target": ids[ev[:, 1]], "step": ev[:, 2]})
   keep = ids[np.union1d(roots, ev[:, 1]).astype(np.int64)]
   if net.layer_labels is None:   # the node table of the infected nodes, with their attributes
      nodes = _tables(net)[2]
      nodes = nodes[nodes["node"].isin(set(keep.tolist())).to_numpy()]
   else:   # copies (entity, layer) as node ids
      nodes = pd.DataFrame({"node": keep}) if net.node_format == "wide" else pd.DataFrame({"node": keep, "attribute": np.nan, "value": np.nan})
   return load_network(edges, nodes, source = "source", target = "target", directed = True, node_format = net.node_format, device = net.components[0].edge_index.device)


def si_model(net, beta, seeds = 1, ranking = None, selection = "top", runs = 100, max_steps = 1000, component = 0, seed = None, engine = None, batch = None, attribute = "state", resume = False, immunize = None, strategy = "random"):
   """
   Simulate the SI (Susceptible-Infected) model, and return the fraction of nodes in each state
   at every time step, averaged over independent runs.

   At step 0 the seed nodes are infected, all the other nodes are susceptible. At every following
   step, each susceptible node with at least one infected neighbor becomes infected with
   probability ``beta`` (one draw per node and step, whatever the number of its infected
   neighbors; see the module description).

   A run stops when nothing can change any more, or after ``max_steps`` steps. In the SI model,
   that is when no susceptible node has an infected neighbor, so that the infection has reached
   every node it can reach (all the nodes of an undirected component; in directed networks, the
   nodes reachable from the seeds). A finished run keeps its final value until the last run
   stops, so the table has one row per step up to the longest run.

   Seeds are drawn anew for every run, among the nodes considered:

   * ``ranking = None``: ``seeds`` nodes uniformly at random;
   * ``selection = "top"``: the ``seeds`` nodes with the largest ``ranking`` values, the same in
     every run (ties are broken by node order; NaN counts as the smallest value);
   * ``selection = "weighted"``: ``seeds`` nodes drawn without replacement with probability
     proportional to ``ranking`` (which must be non negative; NaN counts as 0).

   ``ranking`` is any vector of node values, e.g. the output of a centrality or of
   :func:`~graphoskop.degree` with ``distribution = None``, on the same component, or the list of
   vectors of all components (as returned with ``component = None``); a dictionary or
   :class:`pandas.Series` from node id to value; or the name of a numeric node attribute.

   After the simulation, the state of every node at the end of the last run (``"S"`` or ``"I"``)
   is stored in the qualitative node attribute ``attribute``, on the nodes considered (other
   components get NaN if the attribute is new). Like every qualitative attribute, it is listed in
   :attr:`~graphoskop.Network.qualitative_node_attributes` (not in
   :attr:`~graphoskop.Network.node_attributes`, which lists numeric attributes, stored in the
   ``node_attr`` tensors), and is read with ``net.get_node_attribute(attribute)``. With
   ``resume = True`` the seeds are not drawn: every run starts from the states in ``attribute``
   (missing values count as susceptible). One step at a time, an epidemic is followed by running
   ``runs = 1, max_steps = 1, resume = True`` again and again.

   The runs are independent, and are simulated in parallel. With ``engine = "numba"`` each CPU
   thread simulates whole runs, touching only the nodes that can change and the edges of the
   nodes that changed. With ``engine = "torch"`` the runs of a batch are simulated together, as
   vector operations on the device of the network; ``batch`` limits the number of runs at once
   (by default, so that ``batch`` times the number of nodes is at most :math:`2^{27}`). By
   default, the torch engine is used for networks on a GPU, the Numba engine otherwise. Random
   numbers come from ``seed``: with the Numba engine results are the same whatever the number of
   threads; the two engines give different draws of the same model.

   The confidence interval of the mean at each step is the 95% Student's t interval, from the
   standard deviation across runs (NaN with a single run).

   :param net: A :class:`~graphoskop.Network`.
   :param beta: Infection probability per step of a susceptible node with infected neighbors, in
      (0, 1].
   :param seeds: Number of seed nodes, or a fraction of the nodes if below 1 (at least one node).
   :param ranking: Node values used to choose the seeds, or ``None`` for random seeds.
   :param selection: ``"top"`` or ``"weighted"``, how seeds are chosen from ``ranking``.
   :param runs: Number of independent runs.
   :param max_steps: Largest number of steps of a run.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network
      (seeds anywhere; the infection cannot leave the components of the seeds).
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param engine: ``"numba"`` (CPU threads), ``"torch"`` (device of the network), or ``None``.
   :param batch: With the torch engine, the number of runs simulated at once, or ``None``.
   :param attribute: Name of the node attribute holding the states, or ``None`` not to store them.
   :param resume: Whether to start from the states in ``attribute`` instead of seeds.
   :param immunize: Nodes immunized at step 0, never infected (see
      :mod:`~graphoskop.epidemics.simple_contagion`): a number of nodes (a fraction of them if
      below 1), chosen by ``strategy``, or the node ids themselves; ``None`` for none.
   :param strategy: With a number of nodes to immunize: ``"random"``, ``"friend_of_random"``, or
      node values (vector, dictionary or :class:`pandas.Series`, name of a numeric node
      attribute) whose largest ones are immunized. Ignored with node ids.
   :return: :class:`pandas.DataFrame` indexed by step (0 is the start), with columns ``S`` and
      ``I`` (mean fraction of nodes in each state), ``S_low``, ``S_high``, ``I_low`` and
      ``I_high`` (confidence intervals).
   :raises ValueError: If a parameter is not valid, if ``ranking`` does not give one value per
      node, if it has negative values with ``selection = "weighted"``, if fewer nodes than
      ``seeds`` have a positive value, or, with ``resume``, if ``attribute`` is not a qualitative
      node attribute or has a value that is not a state of the model.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("contacts.tsv")
      si = gsk.si_model(net, beta = 0.1, seeds = 5, runs = 1000, seed = 1)
      si.plot(y = "I")                                                # epidemic curve
      net.get_node_attribute("state")                                 # "S" or "I", end of the last run
      hubs = gsk.si_model(net, 0.1, 5, ranking = gsk.degree(net, distribution = None))
      pr = gsk.pagerank(net, component = None)
      si = gsk.si_model(net, 0.1, 5, ranking = pr, selection = "weighted", component = None)

      gsk.si_model(net, 0.1, 5, runs = 1, max_steps = 0)              # seeds only, step 0
      for step in range(10):                                          # then one step at a time
         gsk.si_model(net, 0.1, runs = 1, max_steps = 1, resume = True)
   """
   return _simulate(net, "SI", beta, 0., seeds, ranking, selection, runs, max_steps, component, seed, engine, batch, attribute, resume, immunize = immunize, strategy = strategy)


def sis_model(net, beta, mu, seeds = 1, ranking = None, selection = "top", runs = 100, max_steps = 1000, component = 0, seed = None, engine = None, batch = None, attribute = "state", resume = False, window = 100, tolerance = 0.01, immunize = None, strategy = "random"):
   """
   Simulate the SIS (Susceptible-Infected-Susceptible) model, and return the fraction of nodes in
   each state at every time step, averaged over independent runs.

   As :func:`si_model` (seeds, engines, stored states and resuming work the same way), but at
   every step each node infected at the start of the step recovers with probability ``mu`` and
   becomes susceptible again, at the same time as the infections.

   A run stops when no node is infected, after ``max_steps`` steps, or when the infection has
   become endemic: the number of infected nodes fluctuates around a plateau. A run is at the
   plateau when the mean number of infected nodes over the last ``window`` steps differs by at
   most ``tolerance`` (relative) from the mean over the ``window`` steps before. From then on, the
   run counts with its mean over the last ``window`` steps, the estimate of its endemic level,
   and the stored states are those of the step where it stopped. In small networks the
   fluctuations can keep the test from passing (the runs then last ``max_steps`` steps): a larger
   ``window`` or ``tolerance`` helps. Near the epidemic threshold the number of infected nodes
   drifts slowly, and a small ``tolerance`` avoids stopping too early. ``window = None`` turns the
   test off.

   :param net: A :class:`~graphoskop.Network`.
   :param beta: Infection probability per step, in (0, 1].
   :param mu: Recovery probability per step, in (0, 1].
   :param seeds: Number of seed nodes, or a fraction of the nodes if below 1.
   :param ranking: Node values used to choose the seeds, or ``None`` for random seeds.
   :param selection: ``"top"`` or ``"weighted"``.
   :param runs: Number of independent runs.
   :param max_steps: Largest number of steps of a run.
   :param component: Index of the component, or ``None`` for the whole network.
   :param seed: Seed of the random numbers, or ``None``.
   :param engine: ``"numba"``, ``"torch"`` or ``None``.
   :param batch: With the torch engine, the number of runs simulated at once, or ``None``.
   :param attribute: Name of the node attribute holding the states (``"S"`` or ``"I"``), or
      ``None``.
   :param resume: Whether to start from the states in ``attribute`` instead of seeds.
   :param window: Number of steps of each of the two windows compared by the plateau test, or
      ``None`` not to stop at the plateau.
   :param tolerance: Largest relative difference between the means of the two windows at the
      plateau.
   :param immunize: Nodes immunized at step 0, never infected (see
      :mod:`~graphoskop.epidemics.simple_contagion`): a number of nodes (a fraction of them if
      below 1), chosen by ``strategy``, or the node ids themselves; ``None`` for none.
   :param strategy: With a number of nodes to immunize: ``"random"``, ``"friend_of_random"``, or
      node values (vector, dictionary or :class:`pandas.Series`, name of a numeric node
      attribute) whose largest ones are immunized. Ignored with node ids.
   :return: :class:`pandas.DataFrame` indexed by step, with columns ``S``, ``S_low``,
      ``S_high``, ``I``, ``I_low`` and ``I_high``.
   :raises ValueError: As :func:`si_model`, or if ``mu``, ``window`` or ``tolerance`` is not
      valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("contacts.tsv")
      sis = gsk.sis_model(net, beta = 0.1, mu = 0.05, seeds = 5)     # runs stop at the plateau
      sis["I"].iloc[-1]                                               # endemic level
      sis = gsk.sis_model(net, 0.1, 0.05, 5, window = None, max_steps = 2000)   # no plateau test
   """
   return _simulate(net, "SIS", beta, mu, seeds, ranking, selection, runs, max_steps, component, seed, engine, batch, attribute, resume, window, tolerance, immunize = immunize, strategy = strategy)


def sir_model(net, beta, mu, seeds = 1, ranking = None, selection = "top", runs = 100, max_steps = 1000, component = 0, seed = None, engine = None, batch = None, attribute = "state", resume = False, immunize = None, strategy = "random"):
   """
   Simulate the SIR (Susceptible-Infected-Removed) model, and return the fraction of nodes in
   each state at every time step, averaged over independent runs.

   As :func:`si_model` (seeds, engines, stored states and resuming work the same way), but at
   every step each node infected at the start of the step recovers with probability ``mu`` and is
   removed (it can no longer infect nor be infected), at the same time as the infections. A run
   stops when no node is infected, or after ``max_steps`` steps.

   :param net: A :class:`~graphoskop.Network`.
   :param beta: Infection probability per step, in (0, 1].
   :param mu: Recovery probability per step, in (0, 1].
   :param seeds: Number of seed nodes, or a fraction of the nodes if below 1.
   :param ranking: Node values used to choose the seeds, or ``None`` for random seeds.
   :param selection: ``"top"`` or ``"weighted"``.
   :param runs: Number of independent runs.
   :param max_steps: Largest number of steps of a run.
   :param component: Index of the component, or ``None`` for the whole network.
   :param seed: Seed of the random numbers, or ``None``.
   :param engine: ``"numba"``, ``"torch"`` or ``None``.
   :param batch: With the torch engine, the number of runs simulated at once, or ``None``.
   :param attribute: Name of the node attribute holding the states (``"S"``, ``"I"`` or ``"R"``),
      or ``None``.
   :param resume: Whether to start from the states in ``attribute`` instead of seeds.
   :param immunize: Nodes immunized at step 0, never infected (see
      :mod:`~graphoskop.epidemics.simple_contagion`): a number of nodes (a fraction of them if
      below 1), chosen by ``strategy``, or the node ids themselves; ``None`` for none.
   :param strategy: With a number of nodes to immunize: ``"random"``, ``"friend_of_random"``, or
      node values (vector, dictionary or :class:`pandas.Series`, name of a numeric node
      attribute) whose largest ones are immunized. Ignored with node ids.
   :return: :class:`pandas.DataFrame` indexed by step, with columns ``S``, ``I`` and ``R`` and
      their confidence intervals (``S_low``, ``S_high``, ...).
   :raises ValueError: As :func:`si_model`, or if ``mu`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("contacts.tsv")
      sir = gsk.sir_model(net, beta = 0.2, mu = 0.1, seeds = 5, runs = 1000)
      sir["R"].iloc[-1]                                               # final size of the epidemic
      sir["I"].idxmax()                                               # step of the peak
   """
   return _simulate(net, "SIR", beta, mu, seeds, ranking, selection, runs, max_steps, component, seed, engine, batch, attribute, resume, immunize = immunize, strategy = strategy)
