"""
Compartmental models of epidemics on networks: every node is in a compartment (a state), and
moves between compartments by contagion through the edges or spontaneously.

:func:`si_model` simulates the SI model: nodes are Susceptible (S) or Infected (I), and an
infected node stays infected forever. Simulations run in discrete, synchronous time steps: at
each step, every susceptible node with at least one infected neighbor becomes infected with
probability ``beta``, with one draw per node per step, whatever the number of its infected
neighbors. The infection starts from a set of seed nodes, chosen at random or from a ranking of
the nodes (e.g. a centrality), and many independent runs are simulated in parallel and averaged.
"""

import numba
import numpy as np
import pandas as pd
import torch
from numba import njit, prange
from scipy.stats import t as student

from .ergms import _random


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
def _si_numba(ptr, nbr, seeds, beta, max_steps, rng, total, square, ends, finals):
   """
   SI runs, in parallel: thread ``b`` simulates runs ``b``, ``b + B``, ... and adds the number of
   infected nodes at each step to ``total[b]`` (and its square to ``square[b]``), until no
   susceptible node has an infected neighbor or ``max_steps`` steps. Only the nodes with an
   infected neighbor are drawn, and only the edges of newly infected nodes are scanned, so a run
   costs time proportional to the nodes and edges it reaches.

   :param ptr: CSR row pointers of the out edges.
   :param nbr: CSR targets of the out edges.
   :param seeds: Seed nodes, runs by number of seeds.
   :param beta: Infection probability.
   :param max_steps: Largest number of steps.
   :param rng: xorshift state of every run.
   :param total: Threads by ``max_steps + 1``, written in place.
   :param square: Same shape as ``total``, written in place.
   :param ends: Last step of every run, written in place.
   :param finals: Final number of infected nodes of every run, written in place.
   """
   R, B, n = seeds.shape[0], total.shape[0], ptr.shape[0] - 1
   for b in prange(B):
      state = np.zeros(n, dtype = np.uint8)   # 0 susceptible, 1 susceptible with an infected neighbor, 2 infected
      front = np.empty(n, dtype = np.int64)   # nodes in state 1
      new = np.empty(n, dtype = np.int64)     # nodes infected in the last step
      touched = np.empty(n, dtype = np.int64) # nodes not in state 0, reset after the run
      for r in range(b, R, B):
         st = rng[r:r + 1]
         nt, nf, nn = 0, 0, 0
         for s in seeds[r]:
            state[s] = 2
            new[nn] = s
            touched[nt] = s
            nn += 1
            nt += 1
         count, t = nn, 0
         while True:
            for i in range(nn):   # neighbors of the newly infected nodes can now be infected
               u = new[i]
               for e in range(ptr[u], ptr[u + 1]):
                  v = nbr[e]
                  if state[v] == 0:
                     state[v] = 1
                     front[nf] = v
                     touched[nt] = v
                     nf += 1
                     nt += 1
            total[b, t] += count   # integers: exact sums, whatever the order of the runs
            square[b, t] += count * count
            if nf == 0 or t == max_steps:
               break
            t += 1
            nn, keep = 0, 0
            for i in range(nf):   # one draw per exposed node
               v = front[i]
               if _random(st) < beta:
                  state[v] = 2
                  new[nn] = v
                  nn += 1
               else:
                  front[keep] = v
                  keep += 1
            nf = keep
            count += nn
         ends[r], finals[r] = t, count
         for i in range(nt):
            state[touched[i]] = 0


def _si_torch(ptr, nbr, seeds, beta, max_steps, seed, batch):
   """
   SI runs on the device of ``ptr``, all the runs of a batch at once: the state of node ``v`` in
   run ``r`` is entry ``r * n + v`` of one vector, and every step draws the exposed (run, node)
   pairs together and scans the edges of the newly infected ones (same model as
   :func:`_si_numba`).

   :param ptr: CSR row pointers of the out edges (tensor).
   :param nbr: CSR targets of the out edges (tensor).
   :param seeds: Seed nodes, runs by number of seeds (tensor).
   :param beta: Infection probability.
   :param max_steps: Largest number of steps.
   :param seed: Seed of the random numbers, or ``None``.
   :param batch: Number of runs simulated at once.
   :return: Sum over the runs of the fraction of infected nodes at each step, and of its square
      (finished runs count with their final value), as float64 NumPy arrays.
   """
   dev, n = ptr.device, ptr.shape[0] - 1
   gen = torch.Generator(device = dev)
   gen.manual_seed(int(np.random.default_rng(seed).integers(2 ** 62)))
   total, square = [], []
   for a in range(0, seeds.shape[0], batch):
      S = seeds[a:a + batch]
      R = S.shape[0]
      state = torch.zeros(R * n, dtype = torch.uint8, device = dev)

      def expose(new):
         u = new % n
         deg = ptr[u + 1] - ptr[u]
         rep = torch.repeat_interleave(torch.arange(new.shape[0], device = dev), deg)
         pos = ptr[u][rep] + torch.arange(rep.shape[0], device = dev) - (torch.cumsum(deg, 0) - deg)[rep]
         v = nbr[pos] + (new // n)[rep] * n
         v = torch.unique(v[state[v] == 0])
         state[v] = 1
         return v

      new = (S + torch.arange(R, device = dev)[:, None] * n).reshape(-1)
      state[new] = 2
      count = torch.full((R,), S.shape[1], dtype = torch.float64, device = dev)
      front = expose(new)
      xs = [count / n]
      while front.shape[0] and len(xs) <= max_steps:
         hit = torch.rand(front.shape[0], generator = gen, device = dev) < beta
         new, front = front[hit], front[~hit]
         state[new] = 2
         count += torch.bincount(new // n, minlength = R)
         front = torch.cat([front, expose(new)])
         xs.append(count / n)
      X = torch.stack(xs)
      total.append(X.sum(1).cpu().numpy())
      square.append((X * X).sum(1).cpu().numpy())
   T = max(len(x) for x in total)
   pad = lambda x: np.concatenate([x, np.full(T - len(x), x[-1])])   # finished batches stay at their final value
   return sum(pad(x) for x in total), sum(pad(x) for x in square)


def _node_values(net, comps, values):
   """
   Values of the nodes of some components, in the network wide node order.

   :param net: Network.
   :param comps: Component indices.
   :param values: Name of a numeric node attribute; dictionary or :class:`pandas.Series` from
      input node id to value (missing nodes get NaN); or vector over the nodes of the component
      (:class:`torch.Tensor`, array or list), or a list of them, one per component, when there are
      several (as returned by the centralities with ``component = None``).
   :return: float64 :class:`numpy.ndarray`.
   :raises ValueError: If a vector does not have one value per node.
   """
   n = sum(net.components[c].num_nodes for c in comps)
   if isinstance(values, str):
      x = [net.get_node_attribute(values, c) for c in comps]
      x = [v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v, dtype = np.float64) for v in x]
   elif isinstance(values, (dict, pd.Series)):
      ids = np.concatenate([net.node_ids[c] for c in comps])
      x = [pd.Series(ids).map(values).to_numpy(dtype = np.float64)]
   elif isinstance(values, (list, tuple)) and len(values) and (torch.is_tensor(values[0]) or np.ndim(values[0]) == 1):
      x = [v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v) for v in values]
   else:
      x = [values.cpu().numpy() if torch.is_tensor(values) else np.asarray(values)]
   x = np.concatenate(x).astype(np.float64)
   if x.shape != (n,):
      raise ValueError(f"expected one value per node ({n}), got {x.shape[0]}")
   return x


def si_model(net, beta, seeds = 1, ranking = None, selection = "top", runs = 100, max_steps = 1000, component = 0, seed = None, engine = "numba", batch = None):
   """
   Simulate the SI (Susceptible-Infected) model, and return the fraction of infected nodes at
   every time step, averaged over independent runs.

   At step 0 the seed nodes are infected, all the other nodes are susceptible. At every following
   step, each susceptible node with at least one infected neighbor becomes infected with
   probability ``beta``: one draw per node and step, whatever the number of its infected
   neighbors. All nodes are updated at the same time (synchronous updates). In directed networks
   the infection follows the direction of the edges, from source to target; undirected edges
   carry it both ways. Edge weights, parallel edges and self loops do not matter. In multilayer
   networks the model runs on the supra-graph: copies are nodes, and the infection also spreads
   along couplings and other interlayer edges.

   A run stops when no susceptible node has an infected neighbor, so that the infection has
   reached every node it can reach (all the nodes of an undirected component; in directed
   networks, the nodes reachable from the seeds), or after ``max_steps`` steps. A finished run
   keeps its final value until the last run stops, so the timeline has one value per step up to
   the longest run.

   Seeds are drawn anew for every run, among the nodes considered:

   * ``ranking = None``: ``seeds`` nodes uniformly at random;
   * ``selection = "top"``: the ``seeds`` nodes with the largest ``ranking`` values, the same in
     every run (ties are broken by node order; NaN counts as the smallest value);
   * ``selection = "weighted"``: ``seeds`` nodes drawn without replacement with probability
     proportional to ``ranking`` (which must be non negative; NaN counts as 0).

   ``ranking`` is any vector of node values, e.g. the output of a centrality on the same
   component (or, with ``component = None``, the list of vectors of all components), a dictionary
   or :class:`pandas.Series` from node id to value, or the name of a numeric node attribute.

   The runs are independent, and are simulated in parallel. With ``engine = "numba"`` (the
   default) each CPU thread simulates whole runs, scanning only the nodes that have an infected
   neighbor and the edges of the newly infected nodes, so a run costs time proportional to the
   size of the network it reaches. With ``engine = "torch"`` the runs of a batch are simulated
   together, as vector operations on the device of the network (e.g. a GPU); ``batch`` limits the
   number of runs at once (by default, so that ``batch`` times the number of nodes is at most
   :math:`2^{27}`). Random numbers come from ``seed``: with ``engine = "numba"`` results are the
   same whatever the number of threads; the two engines give different draws of the same model.

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
   :param engine: ``"numba"`` (CPU threads) or ``"torch"`` (device of the network).
   :param batch: With ``engine = "torch"``, the number of runs simulated at once, or ``None``.
   :return: :class:`pandas.DataFrame` indexed by step (0 is the seeds), with columns
      ``infected`` (mean fraction of infected nodes), ``ci_low`` and ``ci_high``.
   :raises ValueError: If a parameter is not valid, if ``ranking`` does not give one value per
      node, if it has negative values with ``selection = "weighted"``, or if fewer nodes than
      ``seeds`` have a positive value.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("contacts.tsv")
      si = gsk.si_model(net, beta = 0.1, seeds = 5, runs = 1000, seed = 1)
      si.plot(y = "infected")                                         # epidemic curve
      close = gsk.closeness_centrality(net)                           # component 0
      top = gsk.si_model(net, 0.1, 5, ranking = close)                # the 5 most central nodes
      pr = gsk.pagerank(net, component = None)                        # whole network
      si = gsk.si_model(net, 0.1, 5, ranking = pr, selection = "weighted", component = None)
   """
   if not 0 < beta <= 1:
      raise ValueError(f"beta must be in (0, 1], got {beta}")
   if selection not in ("top", "weighted"):
      raise ValueError(f"unknown selection: {selection}")
   if engine not in ("numba", "torch"):
      raise ValueError(f"unknown engine: {engine}")
   if runs < 1 or max_steps < 0:
      raise ValueError("runs must be positive and max_steps non negative")
   comps = list(range(len(net.components))) if component is None else [component]
   sizes = [net.components[c].num_nodes for c in comps]
   off = np.cumsum([0] + sizes)
   n = int(off[-1])
   k = max(1, int(round(seeds * n))) if seeds < 1 else int(seeds)
   if k > n:
      raise ValueError(f"{k} seeds, but only {n} nodes")

   # seeds of every run
   rs = np.random.default_rng(seed)
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
   order = torch.argsort(s, stable = True)
   nbr = t[order]
   ptr = torch.zeros(n + 1, dtype = torch.long, device = dev)
   ptr[1:] = torch.cumsum(torch.bincount(s, minlength = n), 0)

   # simulation
   if engine == "numba":
      B = min(runs, numba.get_num_threads())
      total, square = np.zeros((B, max_steps + 1)), np.zeros((B, max_steps + 1))
      ends, finals = np.zeros(runs, dtype = np.int64), np.zeros(runs)
      _si_numba(ptr.numpy(), nbr.numpy(), out, float(beta), int(max_steps), rs.integers(1, 2 ** 63, size = runs, dtype = np.uint64), total, square, ends, finals)
      T = int(ends.max())
      pad = np.zeros((2, T + 2))   # finished runs keep their final value until step T
      np.add.at(pad[0], ends + 1, finals)
      np.add.at(pad[1], ends + 1, finals * finals)
      pad = np.cumsum(pad, 1)[:, :T + 1]
      total, square = (total.sum(0)[:T + 1] + pad[0]) / n, (square.sum(0)[:T + 1] + pad[1]) / n ** 2
   else:
      batch = batch or max(1, 2 ** 27 // n)
      total, square = _si_torch(ptr, nbr, torch.as_tensor(out, device = dev), float(beta), int(max_steps), int(rs.integers(2 ** 62)), batch)

   mean = total / runs
   if runs > 1:
      sd = np.sqrt(np.maximum(square - runs * mean * mean, 0) / (runs - 1))
      h = student.ppf(0.975, runs - 1) * sd / np.sqrt(runs)
   else:
      h = np.full(len(mean), np.nan)
   return pd.DataFrame({"infected": mean, "ci_low": mean - h, "ci_high": mean + h}, index = pd.RangeIndex(len(mean), name = "step"))
