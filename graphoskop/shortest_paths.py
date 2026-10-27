"""
Shortest paths between the nodes of a network.

Paths are computed on the CPU with Dijkstra's algorithm (breadth first search for unit lengths)
from :mod:`scipy.sparse.csgraph`, whatever the device of the network: on sparse networks this needs
:math:`O(m + n \\log n)` work per source, while GPU alternatives either need :math:`O(n^3)` work
(Floyd-Warshall) or synchronize at every step of the search. Searches from many sources are split
across processes, since scipy's implementation does not release the Python global interpreter lock
and so cannot run in parallel threads. The processes are started by the first parallel search and
kept, idle, for the next ones until Python exits (or until a search asks for a different
``n_jobs``), so their start, which takes seconds where processes are spawned and import the
library again, is paid once per session and not at every call. Matrices of lengths are returned
on the device of the component. On platforms where new processes do not fork (the default on Windows and macOS, and on
Linux from Python 3.14), scripts that compute all pairs on large components must protect their
entry point with ``if __name__ == "__main__":``, and ``n_jobs = 1`` turns the processes off.

Edge lengths are the values of an edge attribute (``weight``), or 1. For edge attributes that
measure affinity, as the weights used by the Laplacians do, ``inverse = True`` uses their
reciprocal as length. Parallel edges keep the shortest length; lengths must be non negative.
Directed networks are searched along edge directions. Nodes are given by their input ids;
nodes in different components are not connected.

In multilayer networks, paths run on the supra-graph: couplings and other interlayer edges are
edges like the others, so changing layer is a step of the path, with the length of the coupling
(1 with unit lengths). Couplings store weights read as affinities by the Laplacians (the larger,
the easier to change layer), so use ``inverse = True`` for lengths consistent with them; a
coupling of length 0 makes changing layer free. Nodes are copies ``(entity, layer)``, and paths
are lists of copies, showing where they change layer. An entity can be given instead of a copy:
then all its copies are origins (or destinations), and the shortest path from (or to) any of them
is used.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import dijkstra
from ._policy import _higher_order

_PARALLEL_WORK = 1e8   # sources times edges above which searches are split across processes
_POOL = None           # (number of processes, executor), kept between parallel searches


def _map(worker, args, n_jobs):
   """
   A top level function mapped over arguments in the worker processes, which are started at the
   first call and reused by the next ones with the same ``n_jobs``. If a process dies (e.g. out
   of memory), the pool is dropped and the next call starts a new one.

   :param worker: Top level function.
   :param args: List of arguments, one per call of ``worker``.
   :param n_jobs: Number of processes.
   :return: List of the results, in the order of ``args``.
   :raises concurrent.futures.process.BrokenProcessPool: If a process died.
   """
   global _POOL
   if _POOL is None or _POOL[0] != n_jobs:
      if _POOL is not None:
         _POOL[1].shutdown()
      _POOL = (n_jobs, ProcessPoolExecutor(n_jobs))
   try:
      return list(_POOL[1].map(worker, args))
   except BrokenProcessPool:
      _POOL = None
      raise


def _locate(net, node):
   """
   Internal ids of a node, by component.

   :param net: Network.
   :param node: Input node id (also accepted as a number, if the ids are numeric strings), or
      entity in multilayer networks.
   :return: Dictionary from component to the array of indices of the node (all copies of an
      entity).
   """
   out = {}
   for c, i in net._nodes(node):
      out.setdefault(c, []).append(i)
   return {c: np.array(i) for c, i in out.items()}


def _from(net, c, sources, weight, inverse, predecessors):
   """
   Shortest paths from the nearest of some sources.

   :param net: Network.
   :param c: Component index.
   :param sources: Source indices.
   :param weight: Edge attribute name used as length, or ``None``.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :param predecessors: Whether to also return the predecessors.
   :return: Lengths, and predecessors if requested (negative for sources and unreachable nodes).
   """
   out = dijkstra(_graph(net, c, weight, inverse), indices = sources, unweighted = weight is None, return_predecessors = predecessors, min_only = True)
   return out[:2] if predecessors else out


def _lengths(net, component, weight, inverse):
   """
   Length of every stored edge of a component.

   :param net: Network.
   :param component: Component index.
   :param weight: Edge attribute name used as length, or ``None`` for unit lengths.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :return: float64 array aligned with ``edge_index``.
   :raises ValueError: If some lengths are negative or NaN.
   """
   data = net.components[component]
   if weight is None:
      return np.ones(data.edge_index.shape[1])
   w = data.edge_attr[:, net.edge_attributes.index(weight)].double().cpu().numpy()
   if inverse:
      with np.errstate(divide = "ignore"):
         w = 1 / w   # zero affinity: infinite length, i.e. no connection
   if np.isnan(w).any() or (w < 0).any():
      raise ValueError("shortest paths: edge lengths must be non negative numbers")
   return w


def _graph(net, component, weight, inverse):
   """
   Edge lengths of a component as a scipy CSR matrix, keeping the shortest of parallel edges.

   :param net: Network.
   :param component: Component index.
   :param weight: Edge attribute name used as length, or ``None`` for unit lengths.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :return: CSR matrix.
   :raises ValueError: If some lengths are negative or NaN.
   """
   data = net.components[component]
   u, v = data.edge_index.cpu().numpy()
   w = _lengths(net, component, weight, inverse)
   order = np.lexsort((w, v, u))
   u, v, w = u[order], v[order], w[order]
   first = np.r_[True, (u[1:] != u[:-1]) | (v[1:] != v[:-1])][:len(u)]   # no edges: empty
   return sp.csr_matrix((w[first], (u[first], v[first])), shape = (data.num_nodes,) * 2)


def _search(args):
   """
   Dijkstra from a set of sources (top level, so that worker processes can run it).

   :param args: ``(graph, sources, unweighted, predecessors)``.
   :return: Lengths, or lengths and predecessors.
   """
   g, sources, unweighted, predecessors = args
   return dijkstra(g, indices = sources, unweighted = unweighted, return_predecessors = predecessors)


def _dijkstra(g, sources, unweighted, predecessors, n_jobs):
   """
   Dijkstra from many sources, split across processes when the work is large.

   :param g: CSR matrix of edge lengths.
   :param sources: Source indices.
   :param unweighted: Whether all lengths are 1 (breadth first search).
   :param predecessors: Whether to also return the predecessor matrix.
   :param n_jobs: Number of processes, ``None`` for all CPUs.
   :return: Lengths (sources by nodes), and predecessors if requested.
   """
   n_jobs = os.cpu_count() if n_jobs is None else n_jobs
   if n_jobs <= 1 or len(sources) < 2 * n_jobs or len(sources) * g.nnz < _PARALLEL_WORK:
      return _search((g, sources, unweighted, predecessors))
   parts = _map(_search, [(g, c, unweighted, predecessors) for c in np.array_split(sources, 4 * n_jobs)], n_jobs)
   if not predecessors:
      return np.vstack(parts)
   return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])


def _paths(pred, source, ids):
   """
   All paths from a source (or from the nearest of several sources), from its row of
   predecessors.

   :param pred: Predecessors of each node on its shortest path from ``source`` (negative if none).
   :param source: Source index, or array of source indices.
   :param ids: Input node ids by index.
   :return: Dictionary from the input id of each reachable node to its path (list of input ids).
   """
   memo = {s: [ids[s]] for s in np.atleast_1d(source).tolist()}
   for t in range(len(pred)):
      chain = []
      while t not in memo and pred[t] >= 0:
         chain.append(t)
         t = pred[t]
      if t not in memo:
         continue
      path = memo[t]
      for x in reversed(chain):
         path = path + [ids[x]]
         memo[x] = path
   return {ids[t]: p for t, p in memo.items()}


@_higher_order()
def shortest_path(net, source, target, weight = None, inverse = False):
   """
   Shortest path between two nodes.

   :param net: A :class:`~graphoskop.Network`.
   :param source: Input id of the origin (or entity, in multilayer networks).
   :param target: Input id of the destination (or entity, in multilayer networks).
   :param weight: Name of the edge attribute used as edge length, or ``None`` for unit lengths.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :return: List of input node ids from ``source`` to ``target``, empty if there is no path. With
      entities, the shortest of the paths between any of their copies (ties: the first component,
      then the first copy in layer order).

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.shortest_path(net, "alice", "carol")                 # ['alice', 'bob', 'carol']

      net = gsk.load_network("multiplex.tsv", multilayer = True, coupling = "all")
      gsk.shortest_path(net, "alice", "carol")                 # [('alice', 'work'), ('bob', 'work'), ('bob', 'home'), ('carol', 'home')]
   """
   S, T = _locate(net, source), _locate(net, target)
   best, path = np.inf, []
   for c in sorted(S.keys() & T.keys()):
      dist, pred = _from(net, c, S[c], weight, inverse, True)
      j = T[c][np.argmin(dist[T[c]])]
      if dist[j] < best:
         best, path = dist[j], [j]
         while pred[path[-1]] >= 0:
            path.append(pred[path[-1]])
         path = [net.node_ids[c][x] for x in reversed(path)]
   return path


@_higher_order()
def shortest_path_length(net, source, target, weight = None, inverse = False):
   """
   Length of the shortest path between two nodes.

   :param net: A :class:`~graphoskop.Network`.
   :param source: Input id of the origin (or entity, in multilayer networks).
   :param target: Input id of the destination (or entity, in multilayer networks).
   :param weight: Name of the edge attribute used as edge length, or ``None`` for unit lengths.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :return: Length, ``inf`` if there is no path. With entities, the shortest between any of their
      copies.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.shortest_path_length(net, "alice", "carol", weight = "weight")
   """
   S, T = _locate(net, source), _locate(net, target)
   return float(min((_from(net, c, S[c], weight, inverse, False)[T[c]].min() for c in S.keys() & T.keys()), default = np.inf))


@_higher_order()
def single_source_shortest_paths(net, source, weight = None, inverse = False):
   """
   Shortest paths from one node to every node it reaches.

   :param net: A :class:`~graphoskop.Network`.
   :param source: Input id of the origin (or entity, in multilayer networks: paths from the nearest
      of its copies, each copy reaching itself with a path of one node).
   :param weight: Name of the edge attribute used as edge length, or ``None`` for unit lengths.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :return: Dictionary from the input id of each reachable node to its path (list of input ids).

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      paths = gsk.single_source_shortest_paths(net, "alice")
      paths["carol"]                                           # ['alice', 'bob', 'carol']
   """
   out = {}
   for c, i in _locate(net, source).items():
      out.update(_paths(_from(net, c, i, weight, inverse, True)[1], i, net.node_ids[c]))
   return out


@_higher_order()
def single_source_shortest_path_lengths(net, source, weight = None, inverse = False):
   """
   Lengths of the shortest paths from one node to every node it reaches.

   :param net: A :class:`~graphoskop.Network`.
   :param source: Input id of the origin (or entity, in multilayer networks: lengths from the
      nearest of its copies, 0 for the copies themselves).
   :param weight: Name of the edge attribute used as edge length, or ``None`` for unit lengths.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :return: Dictionary from the input id of each reachable node to its length.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      lengths = gsk.single_source_shortest_path_lengths(net, "alice")
   """
   out = {}
   for c, i in _locate(net, source).items():
      dist, ids = _from(net, c, i, weight, inverse, False), net.node_ids[c]
      out.update({ids[t]: float(dist[t]) for t in np.flatnonzero(np.isfinite(dist))})
   return out


@_higher_order()
def all_pairs_shortest_paths(net, component = 0, weight = None, inverse = False, n_jobs = None):
   """
   Shortest paths between all pairs of nodes of a component.

   The number of paths grows with the square of the number of nodes, so this is meant for small
   components; for large ones, use :func:`all_pairs_shortest_path_lengths` or
   :func:`single_source_shortest_paths` on the sources of interest.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param weight: Name of the edge attribute used as edge length, or ``None`` for unit lengths.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :param n_jobs: Number of processes for the searches, ``None`` for all CPUs.
   :return: Dictionary from each source id to the dictionary of its paths (as in
      :func:`single_source_shortest_paths`), or a list of them if ``component`` is ``None``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      paths = gsk.all_pairs_shortest_paths(net)
      paths["alice"]["carol"]                                  # ['alice', 'bob', 'carol']
   """
   if component is None:
      return [all_pairs_shortest_paths(net, c, weight, inverse, n_jobs) for c in range(len(net.components))]
   g = _graph(net, component, weight, inverse)
   ids = net.node_ids[component]
   pred = _dijkstra(g, np.arange(g.shape[0]), weight is None, True, n_jobs)[1]
   return {ids[s]: _paths(pred[s], s, ids) for s in range(g.shape[0])}


@_higher_order()
def all_pairs_shortest_path_lengths(net, component = 0, weight = None, inverse = False, matrix = True, n_jobs = None):
   """
   Lengths of the shortest paths between all pairs of nodes of a component.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param weight: Name of the edge attribute used as edge length, or ``None`` for unit lengths.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :param matrix: Return an ``n`` by ``n`` tensor, in the same format as
      :func:`~graphoskop.effective_resistance` (rows and columns ordered by internal node
      index, dtype of the edge attributes, device of the component; ``inf`` where there is no
      path). If ``False``, return a dictionary from each source id to the dictionary of its
      lengths (as in :func:`single_source_shortest_path_lengths`).
   :param n_jobs: Number of processes for the searches, ``None`` for all CPUs.
   :return: Lengths, or a list of them if ``component`` is ``None``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      P = gsk.all_pairs_shortest_path_lengths(net, weight = "weight", inverse = True)
      c, i = net.node_map["alice"]
      c, j = net.node_map["carol"]
      P[i, j]                                                  # if both are in component 0
   """
   if component is None:
      return [all_pairs_shortest_path_lengths(net, c, weight, inverse, matrix, n_jobs) for c in range(len(net.components))]
   data = net.components[component]
   g = _graph(net, component, weight, inverse)
   dist = _dijkstra(g, np.arange(g.shape[0]), weight is None, False, n_jobs)
   if matrix:
      return torch.tensor(dist, dtype = data.edge_attr.dtype, device = data.edge_index.device)
   ids = net.node_ids[component]
   return {ids[s]: {ids[t]: float(row[t]) for t in np.flatnonzero(np.isfinite(row))} for s, row in enumerate(dist)}


def _reduce_rows(d, sources, xp = torch):
   """
   Per-source sums over a block of shortest path lengths.

   :param d: Lengths from the sources (one row each) to all nodes, :class:`torch.Tensor` (or
      :class:`numpy.ndarray` with ``xp = np``).
   :param sources: Index of each source (its own column is left out).
   :param xp: ``torch`` or ``numpy`` (worker processes use NumPy only: torch can deadlock in
      forked processes).
   :return: Float64 arrays: number of reachable other nodes, sum of their lengths, sum of the
      reciprocals of the positive lengths.
   """
   d = d.double().clone() if xp is torch else np.array(d, dtype = np.float64)
   d[xp.arange(d.shape[0]), sources] = np.inf
   ok = xp.isfinite(d)
   with np.errstate(divide = "ignore"):
      inv = xp.where(ok & (d > 0), 1 / d, 0)
   return 1. * ok.sum(1), xp.where(ok, d, 0).sum(1), inv.sum(1)


def _stats_worker(args):
   """
   Dijkstra from some sources, reduced with :func:`_reduce_rows` in NumPy (top level, for worker
   processes).

   :param args: ``(graph, sources, unweighted)``.
   :return: Tuple of NumPy arrays.
   """
   g, sources, unweighted = args
   return _reduce_rows(dijkstra(g, indices = sources, unweighted = unweighted), sources, np)


def _check_lengths(net, component, lengths):
   """
   Precomputed lengths of a component, checked.

   :param net: Network.
   :param component: Component index.
   :param lengths: ``n`` by ``n`` tensor, as given by :func:`all_pairs_shortest_path_lengths`.
   :return: The tensor.
   :raises ValueError: If its shape does not match the component.
   """
   n = net.components[component].num_nodes
   if not torch.is_tensor(lengths) or tuple(lengths.shape) != (n, n):
      raise ValueError(f"lengths of component {component} must be a {n} by {n} tensor, as given by all_pairs_shortest_path_lengths")
   return lengths


def _source_stats(net, component, sources, weight, inverse, n_jobs, transpose = False, lengths = None):
   """
   For each source: number of nodes it reaches, sum of the lengths to them and of their
   reciprocals; from precomputed lengths (on their device) or by Dijkstra (CPU, in processes for
   large work), in blocks of about :math:`2^{24}` lengths.

   :param net: Network.
   :param component: Component index.
   :param sources: Source indices.
   :param weight: Edge attribute used as length, or ``None``.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :param n_jobs: Number of processes, ``None`` for all CPUs.
   :param transpose: Lengths to the sources instead of from them.
   :param lengths: Precomputed lengths of the component, or ``None``.
   :return: Three float64 NumPy arrays aligned with ``sources``.
   """
   n = net.components[component].num_nodes
   if lengths is not None:
      D = _check_lengths(net, component, lengths)
      D = D.T if transpose else D
      idx = torch.as_tensor(sources, device = D.device)
      step = max(1, 2 ** 24 // n)
      parts = [_reduce_rows(D[idx[i:i + step]], idx[i:i + step]) for i in range(0, len(sources), step)]
      return tuple(torch.cat([p[k] for p in parts]).cpu().numpy() if parts else np.zeros(0) for k in range(3))
   g = _graph(net, component, weight, inverse)
   if transpose:
      g = g.T.tocsr()
   parts = _blocks(g, sources, weight is None, n_jobs, _stats_worker)
   return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def _blocks(g, sources, unweighted, n_jobs, worker):
   """
   A worker run on blocks of sources (about :math:`2^{24}` lengths each), in parallel processes
   when the work is large.

   :param g: CSR matrix of edge lengths.
   :param sources: Source indices.
   :param unweighted: Whether all lengths are 1.
   :param n_jobs: Number of processes, ``None`` for all CPUs.
   :param worker: Top level function of ``(graph, sources, unweighted)``.
   :return: List of the worker results, in block order.
   """
   n_jobs = os.cpu_count() if n_jobs is None else n_jobs
   parallel = n_jobs > 1 and len(sources) * g.nnz >= _PARALLEL_WORK and len(sources) >= 2 * n_jobs
   chunks = max(int(np.ceil(len(sources) * g.shape[0] / 2 ** 24)), 4 * n_jobs if parallel else 1)
   args = [(g, x, unweighted) for x in np.array_split(np.asarray(sources), min(chunks, max(len(sources), 1)))]
   return _map(worker, args, n_jobs) if parallel else [worker(a) for a in args]


def _hist_worker(args):
   """
   Breadth first searches from some sources, reduced to the counts of the finite lengths (number
   of edges) to the other nodes (top level, for worker processes; NumPy only).

   :param args: ``(graph, sources, unweighted)``.
   :return: Counts by length.
   """
   g, sources, _ = args
   d = dijkstra(g, indices = sources, unweighted = True)
   d[np.arange(len(sources)), sources] = np.inf
   return np.bincount(d[np.isfinite(d)].astype(np.int64))


@_higher_order()
def shortest_path_length_distribution(net, component = None, samples = None, seed = 0, n_jobs = None, lengths = None):
   """
   Distribution of the lengths of shortest paths, in number of edges: how many pairs of nodes
   are joined by a shortest path of each length.

   In undirected networks pairs are unordered, so the count for length 1 is the number of edges
   (parallel edges once, self loops left out); in directed networks pairs are ordered (a path from
   the source to the target), so the count for length 1 is the number of distinct directed edges.
   Pairs without a path (in different components, or not reachable in directed networks) are not
   counted. Lengths are in number of edges, whatever the edge attributes. In multilayer networks
   paths run on the supra-graph, where changing layer along a coupling is one edge.

   Lengths from each block of sources are counted as soon as they are found, so memory stays at a
   few rows of lengths, with searches split across processes for large work (see the module
   description for ``n_jobs``). ``samples`` uses only that many random source nodes (drawn
   uniformly from the nodes considered, ``seed`` for reproducibility) and scales the counts up to
   all sources, as an estimate. With ``lengths`` (computed by
   :func:`all_pairs_shortest_path_lengths` with unit lengths, ``weight = None``, for the same
   ``component``) nothing is searched again.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param samples: Number of random source nodes, or ``None`` for all nodes (exact).
   :param seed: Seed of the random sources.
   :param n_jobs: Number of processes for the searches, ``None`` for all CPUs.
   :param lengths: Precomputed lengths: a tensor for one component, or the list of tensors of
      ``all_pairs_shortest_path_lengths(net, None)`` for ``component = None``.
   :return: :class:`pandas.DataFrame` with columns ``length`` (1, 2, ...) and ``count`` (integer,
      or float when estimated with ``samples``), one row per length that occurs.
   :raises ValueError: If ``lengths`` does not match.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      dist = gsk.shortest_path_length_distribution(net)
      dist.loc[dist.length == 1, "count"]          # the number of edges
      gsk.shortest_path_length_distribution(net, samples = 1000)   # estimate on large networks
   """
   comps = list(range(len(net.components)) if component is None else [component])
   sizes = np.array([net.components[c].num_nodes for c in comps])
   chosen, scale = [np.arange(k) for k in sizes], 1.
   if samples is not None and samples < sizes.sum():
      pick = np.sort(np.random.default_rng(seed).choice(int(sizes.sum()), samples, replace = False))
      bounds = np.concatenate([[0], np.cumsum(sizes)])
      chosen = [pick[(pick >= a) & (pick < b)] - a for a, b in zip(bounds[:-1], bounds[1:])]
      scale = sizes.sum() / samples
   if lengths is not None and component is not None:
      lengths = [lengths]
   counts = np.zeros(1, dtype = np.int64)
   for k, (c, src) in enumerate(zip(comps, chosen)):
      if net.components[c].num_nodes < 2 or not len(src):
         continue
      if lengths is not None:
         D = _check_lengths(net, c, lengths[k])
         idx = torch.as_tensor(src, device = D.device)
         step = max(1, 2 ** 24 // D.shape[0])
         parts = []
         for i in range(0, len(src), step):
            d = D[idx[i:i + step]].double().clone()
            d[torch.arange(d.shape[0], device = d.device), idx[i:i + step]] = torch.inf
            parts.append(torch.bincount(d[torch.isfinite(d)].round().long()).cpu().numpy())
      else:
         parts = _blocks(_graph(net, c, None, False), src, True, n_jobs, _hist_worker)
      for p in parts:
         if len(p) > len(counts):
            counts = np.pad(counts, (0, len(p) - len(counts)))
         counts[:len(p)] += p
   counts = counts / (1 if net.directed else 2) * scale
   length = np.flatnonzero(counts[1:] > 0) + 1
   values = counts[length]
   return pd.DataFrame({"length": length, "count": values.astype(np.int64) if scale == 1 else values})


def _max_worker(args):
   """
   Largest finite shortest path length from some sources (top level, for worker processes).

   :param args: ``(graph, sources, unweighted)``.
   :return: The largest length (0 if none).
   """
   g, sources, unweighted = args
   d = dijkstra(g, indices = sources, unweighted = unweighted)
   d = d[np.isfinite(d)]
   return float(d.max()) if len(d) else 0.


def _bounding_diameter(g, unweighted):
   """
   Diameter of a connected undirected graph by BoundingDiameters (Takes and Kosters, 2011).

   A search from :math:`v` gives its eccentricity :math:`e(v)`, and for every node :math:`w`,
   :math:`\\max(e(v) - d(v, w), d(v, w)) \\le e(w) \\le e(v) + d(v, w)`. Searches start from the
   remaining nodes with the largest upper and the smallest lower bound in turn (ties: highest
   degree); nodes whose bounds cannot change the result are dropped, until the lower and upper
   bounds of the diameter meet.

   :param g: Symmetric CSR matrix of edge lengths.
   :param unweighted: Whether all lengths are 1.
   :return: ``(diameter, number of searches)``.
   """
   n = g.shape[0]
   lo, hi = np.zeros(n), np.full(n, np.inf)
   deg = np.diff(g.indptr)
   alive = np.ones(n, dtype = bool)
   dlo, dhi, k = 0., np.inf, 0
   while dlo < dhi and alive.any():
      idx = np.flatnonzero(alive)
      key = (hi[idx], deg[idx]) if k % 2 == 0 else (-lo[idx], deg[idx])
      v = idx[np.lexsort(key[::-1])[-1]]
      d = dijkstra(g, indices = v, unweighted = unweighted)
      e = d.max()
      k += 1
      lo = np.maximum(lo, np.maximum(e - d, d))
      hi = np.minimum(hi, e + d)
      dlo, dhi = max(dlo, lo.max()), min(dhi, 2 * e, hi.max())
      alive &= ~(((hi <= dlo) & (lo >= dhi / 2)) | (lo == hi))
   return dlo, k


@_higher_order()
def diameter(net, component = 0, weight = None, inverse = False, n_jobs = None, lengths = None):
   """
   Diameter: the length of the longest shortest path, :math:`\\max_{s, t} d(s, t)` over the pairs
   connected by a path.

   Paths are those of the rest of the module (``weight`` and ``inverse``, directions followed in
   directed networks, the supra-graph in multilayer networks); with unit lengths the diameter is
   in number of edges. Pairs without a path are left out: with ``component = None`` the result is
   the largest diameter over the components, and in directed networks the longest shortest path
   between pairs where one reaches the other (NetworkX instead raises an error unless the network is
   strongly connected). A single node has diameter 0.

   In undirected networks the diameter is found exactly with BoundingDiameters (Takes and
   Kosters, 2011): each search from a node bounds the eccentricities (largest lengths) of all the
   others, and searches stop when the lower and upper bounds of the diameter meet, usually after a
   handful of searches instead of one per node. Directed networks search from every node, in blocks
   and processes as :func:`all_pairs_shortest_path_lengths` (``n_jobs``). With ``lengths`` (from
   :func:`all_pairs_shortest_path_lengths` with the same ``component``, ``weight`` and
   ``inverse``) nothing is searched.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param weight: Name of the edge attribute used as edge length, or ``None`` for unit lengths.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :param n_jobs: Number of processes for the searches of directed networks, ``None`` for all CPUs.
   :param lengths: Precomputed lengths: a tensor for one component, or the list of tensors of
      ``all_pairs_shortest_path_lengths(net, None, ...)`` for ``component = None``.
   :return: The diameter: an ``int`` with unit lengths, a ``float`` otherwise.
   :raises ValueError: If some lengths are negative or NaN, or ``lengths`` does not match.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.diameter(net)                                        # largest component, in edges
      gsk.diameter(net, weight = "distance")                   # weighted
   """
   comps = list(range(len(net.components)) if component is None else [component])
   if lengths is not None and component is not None:
      lengths = [lengths]
   best = 0.
   for k, c in enumerate(comps):
      if net.components[c].num_nodes < 2:
         continue
      if lengths is not None:
         D = _check_lengths(net, c, lengths[k])
         best = max(best, float(D[torch.isfinite(D)].max()))
      elif not net.directed:
         best = max(best, _bounding_diameter(_graph(net, c, weight, inverse), weight is None)[0])
      else:
         g = _graph(net, c, weight, inverse)
         best = max([best] + _blocks(g, np.arange(g.shape[0]), weight is None, n_jobs, _max_worker))
   return int(best) if weight is None else best
