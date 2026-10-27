"""
Graph exploration: breadth first and depth first search, random walks and Metropolis-Hastings
random walks from a node.

Searches return the nodes reached from the source, each once, in the order they are visited, with
the node they were reached from (their parent in the search tree) and their depth in that tree; in
breadth first search the depth is the number of edges of a shortest path from the source. Searches
run on the CPU with :mod:`scipy.sparse.csgraph` (compiled code, :math:`O(n + m)`), whatever the
device of the network, as in :mod:`~graphoskop.shortest_paths`. Neighbors are visited in order of
internal index; parallel edges and self loops do not matter.

Random walks instead move for a given number of steps, each to a random neighbor of the current
node: they need not reach every node, and can visit a node many times. A vanilla random walk
visits nodes in proportion to their degree, in the long run (undirected networks); a
Metropolis-Hastings random walk corrects this bias by refusing some moves: from :math:`u` it
proposes a random neighbor :math:`v` and moves there with probability
:math:`\\min(1, k_u / k_v)`, otherwise it stays at :math:`u` for that step, so that in the long run
the walk spends the same number of steps on every node (as used in network sampling). Refused moves
add nothing to the output, so a node is listed once for each time the walk arrives on it, with
the number of steps it stayed there (its ``duration``): weighting visits by duration gives the
uniform distribution, counting visits alone still favors hubs, though less than a vanilla walk. With ``weight``, neighbors are
chosen with probability proportional to the weight of the edge, and :math:`k` is the strength (sum
of the weights) instead of the degree. Parallel edges count as many times as they appear and self
loops are edges like the others (the walk can stay on its node). In directed networks a walk stops
at a node without edges to follow; there, the correction of the Metropolis-Hastings walk is exact
only with ``direction = "both"``. All the walks of a component advance together, one step at a
time, on the device of the network.

Only the component of the source is explored. A source can be given as an iterable of nodes (a
list, set, range, NumPy array or pandas Series or Index; a tuple is always one node): each is then
explored on its own, and the result gains a first column ``source``, with the node as given.

In hypergraphs (see :doc:`higher_order`), a step of a random walk follows Zhou et al. (2006): a
hyperedge of the current node, drawn in proportion to its weight :math:`w(e)` (the hyperedge
attribute ``weight``, or uniformly), then a node of that hyperedge, uniformly, the current one
included. In a directed hypergraph the step goes from a tail to a head (``direction = "out"``).
Walks run as walks of twice the length on the incidence graph, and also report the hyperedge of
every step. Metropolis-Hastings walks run on :func:`~graphoskop.hypergraph_walk_network`, and
explorations of simplicial complexes on their 1-skeleton (their nodes and edges); other walk models
are walks on :func:`~graphoskop.hypergraph_walk_network`. Searches do not support them.

In directed networks ``direction`` chooses the edges followed: ``"out"`` (along their direction),
``"in"`` (against it) or ``"both"``. In multilayer networks the exploration runs on the supra-graph
(couplings are edges like the others). An entity given as source stands for all its copies: a
search starts from all of them at once (as one source linked to each of them), and each random
walk starts from one of them, drawn uniformly at random.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import breadth_first_order, depth_first_order
from ._policy import _higher_order
from ._views import _walk_view

_ITERABLE = (list, set, frozenset, range, np.ndarray, pd.Series, pd.Index)


def _each(source, explore):
   """
   An exploration from one source, or from each source of an iterable.

   :param source: A node, or an iterable of nodes (see :data:`_ITERABLE`).
   :param explore: Function of one source returning a :class:`pandas.DataFrame`.
   :return: The result for the one source, or the results stacked with a first column ``source``.
   """
   if not isinstance(source, _ITERABLE):
      return explore(source)
   parts = [explore(x) for x in source]
   for x, part in zip(source, parts):
      part.insert(0, "source", pd.Series([x] * len(part), dtype = object))
   return pd.concat(parts, ignore_index = True) if parts else pd.DataFrame(columns = ["source"])


def _search(net, source, direction, depth_first):
   """
   Search from the copies of a source, component by component.

   :return: :class:`pandas.DataFrame` with columns ``node``, ``depth`` and ``parent``.
   :raises ValueError: If ``direction`` is not valid.
   :raises KeyError: If there is no such node.
   """
   if direction not in ("out", "in", "both"):
      raise ValueError(f"unknown direction: {direction}")
   starts = {}
   for c, i in net._nodes(source):
      starts.setdefault(c, []).append(i)
   parts = []
   for c, src in sorted(starts.items()):
      data = net.components[c]
      n = data.num_nodes
      s, t = data.edge_index.cpu().numpy()
      if net.directed and direction == "in":
         s, t = t, s
      elif net.directed and direction == "both":
         s, t = np.concatenate([s, t]), np.concatenate([t, s])
      src = np.array(sorted(src))   # a virtual node n, linked to every source
      g = sp.csr_matrix((np.ones(len(s) + len(src)), (np.concatenate([s, np.full(len(src), n)]), np.concatenate([t, src]))), shape = (n + 1, n + 1))
      order, pred = (depth_first_order if depth_first else breadth_first_order)(g, n, directed = True, return_predecessors = True)
      order = order[1:]
      pred = np.where(pred == n, -1, pred)
      depth = np.full(n + 1, -1)
      depth[src] = 0
      for v in order:   # parents come first in the order
         if pred[v] >= 0:
            depth[v] = depth[pred[v]] + 1
      ids = net.node_ids[c]
      parent = np.full(len(order), None, dtype = object)
      has = pred[order] >= 0
      parent[has] = ids[pred[order][has]]
      parts.append(pd.DataFrame({"node": ids[order], "depth": depth[order], "parent": parent}))
   return pd.concat(parts, ignore_index = True)


@_higher_order()
def bfs(net, source, direction = "out", max_depth = None):
   """
   Breadth first search: the nodes reachable from ``source``, by increasing number of edges from
   it (see the module description).

   :param net: A :class:`~graphoskop.Network`.
   :param source: Starting node (input id, internal id ``(component, index)``, or entity in
      multilayer networks), or an iterable of them, searched one by one.
   :param direction: In directed networks, ``"out"``, ``"in"`` or ``"both"``; ignored otherwise.
   :param max_depth: Only nodes at most this many edges away, or ``None`` for all.
   :return: :class:`pandas.DataFrame` in visit order, with columns ``node`` (input id),
      ``depth`` (number of edges from the source, 0 for the source) and ``parent`` (input id of
      the node it was reached from, missing for the source); first ``source`` with an iterable.
   :raises ValueError: If ``direction`` is not valid.
   :raises KeyError: If there is no such node.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      tree = gsk.bfs(net, "alice")
      friends_of_friends = tree.node[tree.depth == 2]
      gsk.bfs(net, "alice", max_depth = 1)          # alice and her neighbors
      gsk.bfs(net, ["alice", "bob"], max_depth = 1) # both, with a column source
   """
   def one(x):
      out = _search(net, x, direction, False)
      return out if max_depth is None else out[out.depth <= max_depth].reset_index(drop = True)
   return _each(source, one)


@_higher_order()
def dfs(net, source, direction = "out"):
   """
   Depth first search: the nodes reachable from ``source``, each followed by all the nodes reached
   through it before backtracking (preorder; see the module description).

   :param net: A :class:`~graphoskop.Network`.
   :param source: Starting node (input id, internal id ``(component, index)``, or entity in
      multilayer networks), or an iterable of them, searched one by one.
   :param direction: In directed networks, ``"out"``, ``"in"`` or ``"both"``; ignored otherwise.
   :return: :class:`pandas.DataFrame` in visit order, with columns ``node`` (input id),
      ``depth`` (depth in the search tree, 0 for the source) and ``parent`` (input id of the node
      it was reached from, missing for the source); first ``source`` with an iterable.
   :raises ValueError: If ``direction`` is not valid.
   :raises KeyError: If there is no such node.

   Example:

   .. code-block:: python

      tree = gsk.dfs(net, "alice")
      tree.node.tolist()                            # visit order
   """
   return _each(source, lambda x: _search(net, x, direction, True))


def _walk_component(net, c, start, length, weight, direction, metropolis, gen):
   """
   Random walks in one component, all advancing together on its device.

   :param net: Network.
   :param c: Component index.
   :param start: Starting node of each walk (int64 tensor on the device of the component).
   :param length: Number of steps.
   :param weight: Edge attribute used as weight, or ``None``.
   :param direction: ``"out"``, ``"in"`` or ``"both"`` (directed networks).
   :param metropolis: Whether moves are accepted with probability :math:`\\min(1, k_u / k_v)`.
   :param gen: Random generator on the device.
   :return: ``(path, moved)``: int64 tensor, one row per walk and ``length + 1`` columns (the
      start, then the node after each step), -1 after a walk stopped at a node without edges to
      follow; and a boolean tensor of the same shape, ``False`` where a move was refused.
   :raises ValueError: If some weights are negative or NaN.
   """
   data = net.components[c]
   n, dev = data.num_nodes, data.edge_index.device
   s, t = data.edge_index
   w = None if weight is None else net._numeric_edge(weight, c).double()
   if w is not None and (torch.isnan(w).any() or (w < 0).any()):
      raise ValueError("random walks: weights must be non negative numbers")
   if net.directed and direction == "in":
      s, t = t, s
   elif net.directed and direction == "both":
      s, t = torch.cat([s, t]), torch.cat([t, s])
      w = None if w is None else torch.cat([w, w])
   order = torch.argsort(s, stable = True)
   t = t[order]
   deg = torch.bincount(s, minlength = n)
   end = torch.cumsum(deg, 0)   # edges of node u: end[u] - deg[u] to end[u] - 1
   if w is not None:
      cw = torch.cumsum(w[order], 0)
      k = torch.zeros(n, dtype = torch.float64, device = dev).index_add_(0, s, w)
      before = cw[(end - deg - 1).clamp(min = 0)] * (deg < end)   # weight of the edges of earlier nodes
   else:
      k = deg.double()
   path = torch.full((len(start), length + 1), -1, dtype = torch.long, device = dev)
   path[:, 0] = cur = start
   moved = torch.ones_like(path, dtype = torch.bool)
   for step in range(1, length + 1):
      alive = cur >= 0
      u = cur.clamp(min = 0)
      r = torch.rand(len(cur), dtype = torch.float64, device = dev, generator = gen)
      if w is None:
         e = end[u] - deg[u] + (r * deg[u]).long()
      else:
         e = torch.searchsorted(cw, before[u] + r * k[u], right = True)
      e = torch.minimum(e.clamp(min = end[u] - deg[u]), end[u] - 1).clamp(min = 0)
      nxt = torch.where((k[u] > 0) & alive, t[e.clamp(max = max(len(t) - 1, 0))] if len(t) else u, -1)
      if metropolis:
         keep = torch.rand(len(cur), dtype = torch.float64, device = dev, generator = gen) * k[nxt.clamp(min = 0)] >= k[u]
         moved[:, step] = ~keep | (nxt < 0)
         nxt = torch.where(keep & (nxt >= 0), u, nxt)
      path[:, step] = cur = nxt
   return path, moved


def _walks(net, source, length, walks, weight, direction, metropolis, seed):
   """
   Random walks from each source (see the module description).

   :return: :class:`pandas.DataFrame` with columns ``walk``, ``step`` and ``node`` (and
      ``duration`` for Metropolis-Hastings walks, whose refused moves are left out), and first
      ``source`` with an iterable of sources.
   :raises ValueError: If ``direction``, ``length`` or ``walks`` are not valid.
   :raises KeyError: If there is no such node.
   """
   if direction not in ("out", "in", "both"):
      raise ValueError(f"unknown direction: {direction}")
   if length < 0 or walks < 1:
      raise ValueError("random walks: length must be at least 0 and walks at least 1")
   many = isinstance(source, _ITERABLE)
   sources = list(source) if many else [source]
   rng = np.random.default_rng(seed)
   copies = [net._nodes(x) for x in sources]
   picks = [[cp[j] for j in rng.integers(len(cp), size = walks)] for cp in copies]   # a copy per walk
   comp = np.array([c for p in picks for c, _ in p], dtype = np.int64)   # walk a * walks + b
   first = np.array([i for p in picks for _, i in p], dtype = np.int64)
   nodes = np.empty((len(comp), length + 1), dtype = object)
   valid = np.zeros((len(comp), length + 1), dtype = bool)
   moved = np.zeros((len(comp), length + 1), dtype = bool)
   for c in np.unique(comp):
      rows = np.flatnonzero(comp == c)
      dev = net.components[c].edge_index.device
      gen = torch.Generator(device = dev)
      gen.manual_seed(int(rng.integers(2 ** 62)))
      P, M = _walk_component(net, int(c), torch.as_tensor(first[rows], device = dev), length, weight, direction, metropolis, gen)
      P = P.cpu().numpy()
      valid[rows], moved[rows] = P >= 0, M.cpu().numpy()
      nodes[rows] = net.node_ids[c][P.clip(min = 0)]
   row, col = np.nonzero(valid & moved)   # by walk, then step
   new = np.r_[True, row[1:] != row[:-1]]
   start = np.flatnonzero(new)
   step = np.arange(len(row)) - np.repeat(start, np.diff(np.r_[start, len(row)]))
   out = pd.DataFrame({"walk": row % walks, "step": step, "node": nodes[row, col]})
   if metropolis:   # steps until the next arrival, or the end of the walk
      until = np.where(np.r_[~new[1:], False], np.r_[col[1:], 0], valid.sum(1)[row])
      out["duration"] = until - col
   if many:
      out.insert(0, "source", pd.Series(np.array(sources + [None], dtype = object)[:-1][row // walks], dtype = object))
   return out


@_higher_order(hypergraph = True, simplicial = True)
def random_walk(net, source, length, walks = 1, weight = None, direction = "out", seed = None):
   """
   Random walks: from ``source``, ``length`` steps, each to a neighbor of the current node drawn
   uniformly at random (or in proportion to the edge weights), possibly repeated (see the module
   description). Nodes can be visited many times, and not every node is reached.

   :param net: A :class:`~graphoskop.Network`.
   :param source: Starting node (input id, internal id ``(component, index)``, or entity in
      multilayer networks), or an iterable of them, each with its own walks.
   :param length: Number of steps of each walk.
   :param walks: Number of walks from each source.
   :param weight: Name of the edge attribute used as edge weight (neighbors are drawn in proportion
      to it), or ``None`` for uniform choices; in hypergraphs, a hyperedge attribute :math:`w(e)`.
   :param direction: In directed networks, ``"out"``, ``"in"`` or ``"both"``; ignored otherwise.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :return: :class:`pandas.DataFrame` with one row per visit, by walk and step: columns ``walk``
      (0 to ``walks - 1``), ``step`` (0 for the source) and ``node`` (input id); first ``source``
      with an iterable; for hypergraphs, last ``hyperedge`` (the :class:`~graphoskop.Hyperedge`
      taken to reach the node, ``None`` at step 0). In directed networks a walk ends early at a
      node without edges to follow.
   :raises ValueError: If ``direction`` is not valid, ``length`` is negative, ``walks`` is below
      1, or some weights are negative or NaN.
   :raises KeyError: If there is no such node or edge attribute.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      w = gsk.random_walk(net, "alice", 10)                 # walk, step, node
      w = gsk.random_walk(net, "alice", 10, walks = 100)    # 100 walks from alice
      visits = w.node.value_counts()
      gsk.random_walk(net, ["alice", "bob"], 10, weight = "weight", seed = 1)

      net = gsk.load_hypergraph({"m1": ["a", "b", "c"], "m2": ["b", "d"]})
      w = gsk.random_walk(net, "a", 10, seed = 1)           # walk, step, node, hyperedge
   """
   if net.higher_order == "hypergraph":
      return _hypergraph_walks(net, source, length, walks, weight, direction, seed)
   net, weight = _walk_view(net, weight)
   return _walks(net, source, length, walks, weight, direction, False, seed)


def _hypergraph_walks(net, source, length, walks, weight, direction, seed):
   """
   Random walks of Zhou et al. on a hypergraph (see :func:`random_walk`), as walks of twice the
   length on its incidence graph: node, hyperedge, node, ...

   :return: As :func:`random_walk`, with a column ``hyperedge``.
   """
   from .higher_order import incidence_network
   sources = list(source) if isinstance(source, _ITERABLE) else [source]
   for x in sources:
      if any(i >= net.bipartite_split[c] for c, i in net._nodes(x)):
         raise ValueError(f"random walks start from nodes, not from {x!r}")
   inc = incidence_network(net)
   if weight is not None:   # every membership of a hyperedge weighs w(e): hyperedges drawn in proportion to w(e), nodes uniformly
      vectors = []
      for c, k in enumerate(net.bipartite_split):
         w = inc._node_vector(weight, c)[k:]
         s, t = inc.components[c].edge_index
         vectors.append(w[torch.maximum(s, t) - k])
      weight = "__hyperedge_weight__"
      inc.set_edge_attribute(weight, vectors, None)
   out = _walks(inc, source, 2 * length, walks, weight, direction, False, seed)
   step = out["step"].to_numpy()
   ids = out["node"].to_numpy()
   prev = np.r_[np.empty(1, dtype = object), ids[:-1]]   # rows are by walk, then step: the hyperedge just before each node
   out["hyperedge"] = pd.Series(np.where(step > 0, prev, None), dtype = object)
   out = out[step % 2 == 0].reset_index(drop = True)   # nodes only; a walk ending on a hyperedge stops at its last node
   out["step"] //= 2
   return out


@_higher_order(hypergraph = True, simplicial = True)
def metropolis_hastings_random_walk(net, source, length, walks = 1, weight = None, direction = "out", seed = None):
   """
   Metropolis-Hastings random walks: as :func:`random_walk`, but a move from :math:`u` to the
   drawn neighbor :math:`v` happens only with probability :math:`\\min(1, k_u / k_v)` (degrees, or
   strengths with ``weight``); otherwise the walk stays at :math:`u` for that step, and nothing is
   added to the output. Moves towards hubs are refused more often, which corrects the bias of
   random walks towards them: in the long run the walk spends the same number of steps on every
   node (undirected networks, or ``direction = "both"``; see the module description). The
   ``duration`` of each visit (1 plus the moves refused there) keeps that count.

   :param net: A :class:`~graphoskop.Network`.
   :param source: Starting node (input id, internal id ``(component, index)``, or entity in
      multilayer networks), or an iterable of them, each with its own walks.
   :param length: Number of steps of each walk, refused moves included (the walk lists at most
      ``length + 1`` nodes).
   :param walks: Number of walks from each source.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param direction: In directed networks, ``"out"``, ``"in"`` or ``"both"``; ignored otherwise.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :return: :class:`pandas.DataFrame` with one row per arrival on a node, by walk: columns
      ``walk``, ``step`` (0 for the source, then 1, 2, ... over accepted moves only), ``node``
      (input id) and ``duration`` (number of steps spent on the node: 1 plus the moves refused
      there; durations of a walk add up to ``length + 1``, unless it stopped early); first
      ``source`` with an iterable.
   :raises ValueError: If ``direction`` is not valid, ``length`` is negative, ``walks`` is below
      1, or some weights are negative or NaN.
   :raises KeyError: If there is no such node or edge attribute.

   Example:

   .. code-block:: python

      w = gsk.metropolis_hastings_random_walk(net, "alice", 1000, walks = 10, seed = 0)
      share = w.groupby("node").duration.sum() / w.duration.sum()   # about uniform
   """
   net, weight = _walk_view(net, weight)
   return _walks(net, source, length, walks, weight, direction, True, seed)
