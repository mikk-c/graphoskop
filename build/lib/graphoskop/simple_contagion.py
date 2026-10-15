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
"""

import numba
import numpy as np
import pandas as pd
import torch
from numba import njit, prange
from scipy.stats import t as student

from .ergms import _random

#: Labels of the states, by code.
STATES = ("S", "I", "R")


@njit(cache = True, parallel = True)
def _draw_seeds(n, w, k, rng, out):
   """
   Seed nodes of every run, without repetition, in parallel over the runs: uniformly at random if
   ``w`` is empty, otherwise proportionally to the weights ``w`` (Efraimidis and Spirakis, 2006:
   the ``k`` nodes with the largest keys :math:`\\log(u) / w`, :math:`u` uniform).

   :param n: Number of nodes.
   :param w: Non negative weight of every node, or an empty array for uniform draws.
   :param k: Number of seeds per run.
   :param rng: xorshift state of every run.
   :param out: Seeds, runs by ``k``, written in place.
   """
   for r in prange(out.shape[0]):
      st = rng[r:r + 1]
      if w.shape[0] == 0:   # rejection of repeated nodes
         taken = np.zeros(n, dtype = np.bool_)
         c = 0
         while c < k:
            v = min(int(_random(st) * n), n - 1)
            if not taken[v]:
               taken[v] = True
               out[r, c] = v
               c += 1
      else:
         key = np.empty(n)
         for i in range(n):
            key[i] = np.log(_random(st) + 1e-300) / w[i] if w[i] > 0 else -np.inf
         thr = np.partition(key, n - k)[n - k]
         c = 0
         for i in range(n):
            if key[i] >= thr and c < k:
               out[r, c] = i
               c += 1


@njit(cache = True, parallel = True)
def _run_numba(ptr, nbr, seeds, removed, beta, mu, sis, max_steps, rng, total, square, ends, finals, last):
   """
   Runs of the SI, SIS or SIR model, in parallel: thread ``b`` simulates runs ``b``, ``b + B``,
   ... and adds the number of nodes in each state at each step to ``total[b]`` (and its square to
   ``square[b]``), until nothing can change any more or ``max_steps`` steps.

   Every node keeps the number of its infected in-neighbors, updated only when a node is
   infected or recovers (scanning its out edges), and the susceptible nodes with infected
   neighbors are kept in a list (state 3, cleaned lazily when they lose them), so a step costs
   time proportional to the nodes that can change and to the edges of the nodes that changed. In
   the SI model nobody recovers, and the counts are not kept.

   :param ptr: CSR row pointers of the out edges.
   :param nbr: CSR targets of the out edges.
   :param seeds: Infected nodes at step 0: run ``r`` uses row ``r % len(seeds)``.
   :param removed: Removed nodes at step 0 (every run).
   :param beta: Infection probability.
   :param mu: Recovery probability (0 for SI).
   :param sis: Whether recovered nodes become susceptible (SIS) or removed (SIR).
   :param max_steps: Largest number of steps.
   :param rng: xorshift state of every run.
   :param total: Threads by ``max_steps + 1`` by 3 (S, I, R), written in place.
   :param square: Same shape as ``total``, written in place.
   :param ends: Last step of every run, written in place.
   :param finals: Number of nodes in each state at the end of every run, written in place.
   :param last: State of every node at the end of the last run, written in place.
   """
   R, B, n = rng.shape[0], total.shape[0], ptr.shape[0] - 1
   for b in prange(B):
      state = np.zeros(n, dtype = np.uint8)   # 0 S, 1 I, 2 R, 3 S in the exposed list
      cnt = np.zeros(n, dtype = np.int32)     # infected in-neighbors
      mark = np.zeros(n, dtype = np.bool_)    # in the touched list (SIS, SIR), reset after the run
      front = np.empty(n, dtype = np.int64)   # exposed: susceptible with infected neighbors (and stale entries)
      inf = np.empty(n, dtype = np.int64)     # infected
      new = np.empty(n, dtype = np.int64)     # infected in the last step
      rec = np.empty(n, dtype = np.int64)     # recovered in the last step
      touched = np.empty(n, dtype = np.int64)
      for r in range(b, R, B):
         st = rng[r:r + 1]
         nt, nf, ni, nn, nr = 0, 0, 0, 0, 0
         for v in removed:
            state[v] = 2
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
         nR, t = removed.shape[0], 0
         while True:
            for i in range(nn):   # newly infected: counts of their neighbors, new exposed nodes
               u = new[i]
               inf[ni] = u
               ni += 1
               for e in range(ptr[u], ptr[u + 1]):
                  v = nbr[e]
                  if mu > 0:   # SI: nobody recovers, so exposed nodes stay exposed, no counts needed
                     cnt[v] += 1
                  if state[v] == 0:
                     state[v] = 3
                     front[nf] = v
                     nf += 1
                     if not (mu > 0 and mark[v]):   # SI: nodes leave S once, no marks needed
                        if mu > 0:
                           mark[v] = True
                        touched[nt] = v
                        nt += 1
            for i in range(nr):   # newly recovered
               u = rec[i]
               for e in range(ptr[u], ptr[u + 1]):
                  cnt[nbr[e]] -= 1
            if sis:
               for i in range(nr):
                  u = rec[i]
                  if cnt[u] > 0 and state[u] == 0:
                     state[u] = 3
                     front[nf] = u
                     nf += 1
            total[b, t, 0] += n - ni - nR   # integers: exact sums, whatever the order of the runs
            total[b, t, 1] += ni
            total[b, t, 2] += nR
            square[b, t, 0] += (n - ni - nR) * (n - ni - nR)
            square[b, t, 1] += ni * ni
            square[b, t, 2] += nR * nR
            if (nf == 0 and (mu == 0 or ni == 0)) or t == max_steps:
               break
            t += 1
            nn, keep = 0, 0
            for i in range(nf):   # infections, from the states at the start of the step
               v = front[i]
               if mu > 0 and cnt[v] == 0:   # lost its infected neighbors
                  state[v] = 0
               elif _random(st) < beta:
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
            for i in range(nr):
               state[rec[i]] = 0 if sis else 2
            if not sis:
               nR += nr
         ends[r] = t
         finals[r, 0], finals[r, 1], finals[r, 2] = n - ni - nR, ni, nR
         if r == R - 1:
            for v in range(n):
               last[v] = 0 if state[v] == 3 else state[v]
         for i in range(nt):
            v = touched[i]
            state[v], cnt[v], mark[v] = 0, 0, False


def _run_torch(ptr, nbr, seeds, removed, beta, mu, sis, max_steps, runs, seed, batch):
   """
   Runs of the SI, SIS or SIR model on the device of ``ptr``, all the runs of a batch at once:
   node ``v`` of run ``r`` is entry ``r * n + v`` of the state vectors, and every step draws all
   the exposed and infected (run, node) pairs together (same model and bookkeeping as
   :func:`_run_numba`).

   :param ptr: CSR row pointers of the out edges (tensor).
   :param nbr: CSR targets of the out edges (tensor).
   :param seeds: Infected nodes at step 0, run ``r`` uses row ``r % len(seeds)`` (tensor).
   :param removed: Removed nodes at step 0 (tensor).
   :param beta: Infection probability.
   :param mu: Recovery probability (0 for SI).
   :param sis: Whether recovered nodes become susceptible (SIS) or removed (SIR).
   :param max_steps: Largest number of steps.
   :param runs: Number of runs.
   :param seed: Seed of the random numbers.
   :param batch: Number of runs simulated at once.
   :return: Sums over the runs of the fraction of nodes in each state at each step, and of their
      squares (steps by 3, finished runs count with their final value), as NumPy arrays, and the
      state of every node at the end of the last run.
   """
   dev, n = ptr.device, ptr.shape[0] - 1
   gen = torch.Generator(device = dev)
   gen.manual_seed(seed)
   empty = torch.zeros(0, dtype = torch.long, device = dev)

   def neighbors(x):   # out-neighbors of (run, node) entries, as entries
      u = x % n
      deg = ptr[u + 1] - ptr[u]
      rep = torch.repeat_interleave(torch.arange(x.shape[0], device = dev), deg)
      pos = ptr[u][rep] + torch.arange(rep.shape[0], device = dev) - (torch.cumsum(deg, 0) - deg)[rep]
      return nbr[pos] + (x // n)[rep] * n

   total, square = [], []
   for a in range(0, runs, batch):
      R = min(batch, runs - a)
      base = torch.arange(R, device = dev) * n
      state = torch.zeros(R * n, dtype = torch.uint8, device = dev)
      cnt = torch.zeros(R * n, dtype = torch.int32, device = dev)
      inl = torch.zeros(R * n, dtype = torch.bool, device = dev)
      state[(removed[None] + base[:, None]).reshape(-1)] = 2
      new = (seeds[torch.arange(a, a + R, device = dev) % seeds.shape[0]] + base[:, None]).reshape(-1)
      state[new] = 1
      inf, rec, front = empty, empty, empty
      nI = torch.zeros(R, dtype = torch.float64, device = dev)
      nR = torch.full((R,), float(removed.shape[0]), dtype = torch.float64, device = dev)
      count = lambda x: torch.bincount(x // n, minlength = R)   # per run
      xs, t = [], 0
      while True:
         nI += count(new) - count(rec)
         w = neighbors(new)
         cand = torch.cat([w, rec]) if sis else w
         if mu > 0:   # SI: nobody recovers, so exposed nodes stay exposed, no counts needed
            inf = torch.cat([inf, new])
            cnt.index_add_(0, w, torch.ones_like(w, dtype = torch.int32))
            wr = neighbors(rec)
            cnt.index_add_(0, wr, torch.full_like(wr, -1, dtype = torch.int32))
            cand = cand[cnt[cand] > 0]
         cand = torch.unique(cand[(state[cand] == 0) & ~inl[cand]])
         inl[cand] = True
         front = torch.cat([front, cand])
         xs.append(torch.stack([n - nI - nR, nI, nR]) / n)
         if (front.shape[0] == 0 and (mu == 0 or nI.sum() == 0)) or t == max_steps:
            break
         t += 1
         ok = state[front] == 0
         if mu > 0:
            ok &= cnt[front] > 0
         inl[front[~ok]] = False
         front = front[ok]
         hit = torch.rand(front.shape[0], generator = gen, device = dev) < beta
         new, front = front[hit], front[~hit]
         inl[new] = False
         rec = empty
         if mu > 0:
            hit = torch.rand(inf.shape[0], generator = gen, device = dev) < mu
            rec, inf = inf[hit], inf[~hit]
         state[new] = 1
         state[rec] = 0 if sis else 2
         if not sis:
            nR += count(rec)
      X = torch.stack(xs)   # steps by 3 by runs
      total.append(X.sum(2).cpu().numpy())
      square.append((X * X).sum(2).cpu().numpy())
   T = max(len(x) for x in total)
   pad = lambda x: np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)])   # finished batches stay at their final value
   return sum(pad(x) for x in total), sum(pad(x) for x in square), state[(R - 1) * n:].cpu().numpy()


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


def _simulate(net, model, beta, mu, seeds, ranking, selection, runs, max_steps, component, seed, engine, batch, attribute, resume):
   """
   Body of :func:`si_model`, :func:`sis_model` and :func:`sir_model`.

   :param model: ``"SI"``, ``"SIS"`` or ``"SIR"``.
   :return: The table of the states over time.
   """
   if not 0 < beta <= 1:
      raise ValueError(f"beta must be in (0, 1], got {beta}")
   if model != "SI" and not 0 < mu <= 1:
      raise ValueError(f"mu must be in (0, 1], got {mu}")
   if selection not in ("top", "weighted"):
      raise ValueError(f"unknown selection: {selection}")
   if runs < 1 or max_steps < 0:
      raise ValueError("runs must be positive and max_steps non negative")
   comps = list(range(len(net.components))) if component is None else [component]
   sizes = [net.components[c].num_nodes for c in comps]
   off = np.cumsum([0] + sizes)
   n = int(off[-1])
   cuda = net.components[comps[0]].edge_index.device.type == "cuda"
   engine = engine or ("torch" if cuda else "numba")
   if engine not in ("numba", "torch"):
      raise ValueError(f"unknown engine: {engine}")
   rs = np.random.default_rng(seed)

   # states at step 0
   removed = np.zeros(0, dtype = np.int64)
   if resume:
      if attribute is None or attribute not in net.qualitative_node_attributes:
         raise ValueError(f"resume needs the states in a qualitative node attribute, not {attribute!r}")
      x = pd.Series(np.concatenate([net.get_node_attribute(attribute, c) for c in comps]), dtype = object)
      code = x.map({s: i for i, s in enumerate(STATES[:3 if model == "SIR" else 2])})
      bad = x.notna() & code.isna()
      if bad.any():
         raise ValueError(f"{attribute!r} has value {x[bad].iloc[0]!r}, not a state of the {model} model")
      code = code.fillna(0).to_numpy(dtype = np.int64)   # missing: susceptible
      out, removed = np.flatnonzero(code == 1)[None], np.flatnonzero(code == 2)
   else:
      k = max(1, int(round(seeds * n))) if seeds < 1 else int(seeds)
      if k > n:
         raise ValueError(f"{k} seeds, but only {n} nodes")
      out = np.zeros((runs, k), dtype = np.int64)
      if ranking is None:
         _draw_seeds(n, np.zeros(0), k, rs.integers(1, 2 ** 63, size = runs, dtype = np.uint64), out)
      else:
         x = _node_values(net, comps, ranking)
         if selection == "top":
            out[:] = np.argsort(-np.nan_to_num(x, nan = -np.inf), kind = "stable")[:k]
         else:
            x = np.nan_to_num(x, nan = 0.)
            if (x < 0).any():
               raise ValueError("weighted selection needs non negative ranking values")
            if (x > 0).sum() < k:
               raise ValueError(f"{k} seeds, but only {int((x > 0).sum())} nodes with a positive ranking value")
            _draw_seeds(n, x, k, rs.integers(1, 2 ** 63, size = runs, dtype = np.uint64), out)

   # out edges in CSR form
   ei = torch.cat([net.components[c].edge_index + int(o) for c, o in zip(comps, off)], 1)
   dev = ei.device if engine == "torch" else torch.device("cpu")
   s, t = ei.to(dev)
   nbr = t[torch.argsort(s, stable = True)]
   ptr = torch.zeros(n + 1, dtype = torch.long, device = dev)
   ptr[1:] = torch.cumsum(torch.bincount(s, minlength = n), 0)

   # simulation
   mu, sis = (0., False) if model == "SI" else (float(mu), model == "SIS")
   if engine == "numba":
      B = min(runs, numba.get_num_threads())
      total, square = np.zeros((B, max_steps + 1, 3)), np.zeros((B, max_steps + 1, 3))
      ends, finals, last = np.zeros(runs, dtype = np.int64), np.zeros((runs, 3)), np.zeros(n, dtype = np.uint8)
      _run_numba(ptr.numpy(), nbr.numpy(), out, removed, float(beta), mu, sis, int(max_steps), rs.integers(1, 2 ** 63, size = runs, dtype = np.uint64), total, square, ends, finals, last)
      T = int(ends.max())
      pad = np.zeros((2, T + 2, 3))   # finished runs keep their final value until step T
      np.add.at(pad[0], ends + 1, finals)
      np.add.at(pad[1], ends + 1, finals * finals)
      pad = np.cumsum(pad, 1)[:, :T + 1]
      total, square = (total.sum(0)[:T + 1] + pad[0]) / n, (square.sum(0)[:T + 1] + pad[1]) / n ** 2
   else:
      tensor = lambda a: torch.as_tensor(a, device = dev)
      total, square, last = _run_torch(ptr, nbr, tensor(out), tensor(removed), float(beta), mu, sis, int(max_steps), runs, int(rs.integers(2 ** 62)), batch or max(1, 2 ** 27 // n))

   # final states of the last run, and the table
   if attribute is not None:
      labels = np.split(np.array(STATES, dtype = object)[last], off[1:-1])
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
   return pd.DataFrame(cols, index = pd.RangeIndex(len(mean), name = "step"))


def si_model(net, beta, seeds = 1, ranking = None, selection = "top", runs = 100, max_steps = 1000, component = 0, seed = None, engine = None, batch = None, attribute = "state", resume = False):
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
   is stored in the qualitative node attribute ``attribute``, on the nodes considered. With
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
   return _simulate(net, "SI", beta, 0., seeds, ranking, selection, runs, max_steps, component, seed, engine, batch, attribute, resume)


def sis_model(net, beta, mu, seeds = 1, ranking = None, selection = "top", runs = 100, max_steps = 1000, component = 0, seed = None, engine = None, batch = None, attribute = "state", resume = False):
   """
   Simulate the SIS (Susceptible-Infected-Susceptible) model, and return the fraction of nodes in
   each state at every time step, averaged over independent runs.

   As :func:`si_model` (seeds, engines, stored states and resuming work the same way), but at
   every step each node infected at the start of the step recovers with probability ``mu`` and
   becomes susceptible again, at the same time as the infections. A run stops when no node is
   infected, or after ``max_steps`` steps: above the epidemic threshold the infection becomes
   endemic, and the runs last ``max_steps`` steps.

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
   :return: :class:`pandas.DataFrame` indexed by step, with columns ``S``, ``S_low``,
      ``S_high``, ``I``, ``I_low`` and ``I_high``.
   :raises ValueError: As :func:`si_model`, or if ``mu`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("contacts.tsv")
      sis = gsk.sis_model(net, beta = 0.1, mu = 0.05, seeds = 5, max_steps = 500)
      sis["I"].iloc[-100:].mean()                                     # endemic level
   """
   return _simulate(net, "SIS", beta, mu, seeds, ranking, selection, runs, max_steps, component, seed, engine, batch, attribute, resume)


def sir_model(net, beta, mu, seeds = 1, ranking = None, selection = "top", runs = 100, max_steps = 1000, component = 0, seed = None, engine = None, batch = None, attribute = "state", resume = False):
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
   return _simulate(net, "SIR", beta, mu, seeds, ranking, selection, runs, max_steps, component, seed, engine, batch, attribute, resume)
