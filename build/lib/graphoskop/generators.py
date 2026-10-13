"""
Synthetic network generators, in three families:

* random graphs, where every pair of nodes is equally likely to be linked:
  :func:`gnm_random_graph`, :func:`gnp_random_graph`;
* clustered graphs, with many triangles: :func:`caveman_graph`, :func:`watts_strogatz_graph`;
* broad degree graphs, with hubs: :func:`barabasi_albert_graph`, :func:`link_selection_graph`,
  :func:`copying_graph`, :func:`holme_kim_graph` (with clustering).

Every generator takes its natural parameters (e.g. the number of nodes and edges) or, with
``net``, an existing network, from which it infers them: the result is then a randomization of
``net``. A randomization keeps the nodes of ``net``, with their ids, numeric and qualitative
attributes, node types (bipartite networks) and layers (multilayer networks), and the device; only
the edges are generated anew. Parameters given explicitly together with ``net`` override the
inferred ones, when they can change without changing the nodes.

New edges take their attributes from the edges they replace: each gets the attribute row of one
original edge, drawn at random without replacement; if there are more new edges than original
ones, every original row is used once and the extra edges get rows drawn with replacement.
Weights and signs (see :doc:`signed`) are therefore shuffled over the new edges, keeping their
distribution.

All random numbers are drawn with :mod:`torch` on the device of the network, so results depend
on ``seed`` and on the device.
"""

import math
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

from .network import BIPARTITE, _build, _node_codes, _node_types, _subset


# ------------------------------------------------------------------------------------------------
# Shared machinery: random numbers, pair sampling, randomization of a network
# ------------------------------------------------------------------------------------------------


def _generator(seed, device):
   """
   Random generator on a device.

   :param seed: Seed, or ``None`` for a random one.
   :param device: Device.
   :return: :class:`torch.Generator`.
   """
   gen = torch.Generator(device = device)
   if seed is None:
      gen.seed()
   else:
      gen.manual_seed(seed)
   return gen


def _choose(N, m, gen, device, exclude = None):
   """
   Uniformly random subset of ``m`` distinct integers in ``[0, N)``, in increasing order.

   Dense cases use a random permutation of the allowed integers; sparse ones draw integers until
   ``m`` are distinct (and allowed), then keep a random ``m`` of them, which keeps the subset
   uniform.

   :param N: Size of the range.
   :param m: Size of the subset.
   :param gen: Random generator.
   :param device: Device.
   :param exclude: int64 tensor of integers that cannot be chosen, or ``None``.
   :return: int64 tensor.
   :raises ValueError: If fewer than ``m`` integers are allowed.
   """
   free = N if exclude is None else N - torch.unique(exclude).shape[0]
   if m > free:
      raise ValueError(f"cannot place {m:,} edges: there are only {free:,} free pairs")
   if free <= 4 * m:
      if exclude is None:
         return torch.randperm(N, generator = gen, device = device)[:m].sort().values
      k = torch.arange(N, device = device)
      k = k[~torch.isin(k, exclude)]
      return k[torch.randperm(k.shape[0], generator = gen, device = device)[:m]].sort().values
   k = torch.zeros(0, dtype = torch.long, device = device)
   while k.shape[0] < m:
      draw = torch.randint(N, (int(1.1 * (m - k.shape[0])) + 16,), generator = gen, device = device)
      k = torch.unique(torch.cat([k, draw if exclude is None else draw[~torch.isin(draw, exclude)]]))
   return k[torch.randperm(k.shape[0], generator = gen, device = device)[:m]].sort().values


def _pair_key(a, b):
   """
   Position of undirected pairs in the strictly lower triangle, row by row (inverse of
   :func:`_triangle`).

   :param a: int64 tensor, one endpoint.
   :param b: int64 tensor, other endpoint (different).
   :return: int64 tensor of positions.
   """
   u, v = torch.maximum(a, b), torch.minimum(a, b)
   return u * (u - 1) // 2 + v


def _triangle(k):
   """
   Pairs ``u > v`` from their positions in the strictly lower triangle, row by row.

   :param k: int64 tensor of positions, ``k = u (u - 1) / 2 + v``.
   :return: ``(u, v)``.
   """
   u = ((1 + torch.sqrt(1 + 8 * k.double())) / 2).floor().long()
   u = u - (u * (u - 1) // 2 > k).long()   # floating point corrections
   u = u + ((u + 1) * u // 2 <= k).long()
   return u, k - u * (u - 1) // 2


def _pair_count(n, directed, self_loops, n0 = None):
   """
   Number of possible edges of a simple graph.

   :param n: Number of nodes.
   :param directed: Whether ``(u, v)`` and ``(v, u)`` are different pairs.
   :param self_loops: Whether pairs ``(u, u)`` are allowed (ignored if bipartite).
   :param n0: Bipartite: number of nodes of type 0; ``None`` otherwise.
   :return: Number of pairs.
   """
   if n0 is not None:
      return n0 * (n - n0) * (2 if directed else 1)
   if directed:
      return n * n if self_loops else n * (n - 1)
   return n * (n + 1) // 2 if self_loops else n * (n - 1) // 2


def _binomial(N, p, gen, device):
   """
   Number of edges of a Gnp random graph: a binomial draw over ``N`` pairs.

   :param N: Number of possible pairs.
   :param p: Probability of each pair.
   :param gen: Random generator.
   :param device: Device.
   :return: Number of edges.
   :raises ValueError: If ``p`` is not in ``[0, 1]``.
   """
   if not 0 <= p <= 1:
      raise ValueError(f"p must be between 0 and 1, got {p}")
   x = torch.full((1,), float(N), dtype = torch.float64, device = device)
   return int(torch.binomial(x, torch.full_like(x, p), generator = gen).item())


def _gnm_edges(n, m, directed, self_loops, gen, device, n0 = None):
   """
   Edges of a Gnm random graph: ``m`` distinct node pairs drawn uniformly at random.

   :param n: Number of nodes, local ids ``0`` to ``n - 1``.
   :param m: Number of edges.
   :param directed: Whether ``(u, v)`` and ``(v, u)`` are different pairs.
   :param self_loops: Whether pairs ``(u, u)`` are allowed (ignored if bipartite).
   :param gen: Random generator.
   :param device: Device.
   :param n0: Bipartite: number of nodes of type 0 (ids ``0`` to ``n0 - 1``), which only connect
      to nodes of type 1; ``None`` otherwise.
   :return: Source and target local ids, int64 tensors.
   :raises ValueError: If there are fewer than ``m`` possible pairs.
   """
   N = _pair_count(n, directed, self_loops, n0)
   if m > N:
      raise ValueError(f"cannot place {m:,} edges: there are only {N:,} possible ones")
   k = _choose(N, m, gen, device)
   if m == 0:
      return k, k.clone()
   if n0 is not None:
      n1 = n - n0
      flip = k >= n0 * n1   # directed: second half of the pairs go from type 1 to type 0
      k = k % (n0 * n1)
      a, b = k // n1, n0 + k % n1
      return torch.where(flip, b, a), torch.where(flip, a, b)
   if directed and self_loops:
      return k // n, k % n
   if directed:
      u, r = k // (n - 1), k % (n - 1)
      return u, r + (r >= u).long()
   if self_loops:   # pairs u > v among n + 1 nodes are the pairs u - 1 >= v among n nodes
      u, v = _triangle(k)
      return u - 1, v
   return _triangle(k)


def _attribute_rows(rows, m, gen, device):
   """
   Original edges giving their attributes to new edges.

   :param rows: int64 tensor, positions of the original edges.
   :param m: Number of new edges.
   :param gen: Random generator.
   :param device: Device.
   :return: Positions, one per new edge, or ``None`` if there are no original edges.
   """
   m0 = rows.shape[0]
   if m0 == 0:
      return None
   if m <= m0:
      return rows[torch.randperm(m0, generator = gen, device = device)[:m]]
   pick = torch.cat([torch.arange(m0, device = device), torch.randint(m0, (m - m0,), generator = gen, device = device)])   # each once, then random extras
   return rows[pick[torch.randperm(m, generator = gen, device = device)]]


def _layer_value(x, default, net, l, name, scalar):
   """
   Value of a parameter in one layer (the only one, for networks that are not multilayer).

   :param x: Value, dictionary from layer to value, or ``None``.
   :param default: Default value in this layer.
   :param net: Network.
   :param l: Position of the layer in ``net.layer_labels`` (0 if not multilayer).
   :param name: Parameter name, for error messages.
   :param scalar: Whether a single value applies to all the layers of a multilayer network.
   :return: The value.
   :raises ValueError: If ``x`` is a dictionary for a network that is not multilayer, names
      layers not in the network, or is a single value when ``scalar`` is ``False`` in a
      multilayer network.
   """
   if x is None:
      return default
   if net.layer_labels is None:
      if isinstance(x, dict):
         raise ValueError(f"{name} is a number for networks that are not multilayer")
      return x
   if not isinstance(x, dict):
      if not scalar:
         raise ValueError(f"in multilayer networks {name} is a dictionary from layer to value")
      return x
   unknown = set(x) - set(net.layer_labels)
   if unknown:
      raise ValueError(f"layers not in the network: {sorted(map(str, unknown))}")
   return x.get(net.layer_labels[l], default)


def _randomize(net, seed, make):
   """
   Randomization of a network: new intra-layer edges in each layer (in all the network, if not
   multilayer), interlayer edges kept, attributes of the new edges drawn from the old edges of
   the same layer, nodes kept (see the module description).

   :param net: Network.
   :param seed: Seed of the random numbers, or ``None``.
   :param make: Function ``make(l, nodes, n0, m, loops, gen)`` giving the new edges of layer
      ``l`` as two tensors of local ids (positions in ``nodes``), from: ``nodes``, network wide
      positions of the nodes of the layer, type 0 first; ``n0``, number of nodes of type 0
      (``None`` if not bipartite); ``m``, number of intra-layer edges of the layer; ``loops``,
      whether the network has self loops; ``gen``, the random generator.
   :return: The new :class:`~graphoskop.Network`.
   """
   device = net.components[0].edge_index.device
   gen = _generator(seed, device)
   N = int(net.node_offsets[-1])
   types = _node_types(net)
   codes = _node_codes(net)
   layer = torch.zeros(N, dtype = torch.long, device = device) if codes is None else torch.as_tensor(codes[0], device = device)

   # input edges, by network wide node position
   S, T, X = [], [], []
   for c in range(len(net.components)):
      ei, x = net._input_edges(c)
      S.append(ei[0] + int(net.node_offsets[c]))
      T.append(ei[1] + int(net.node_offsets[c]))
      X.append(x)
   S, T, X = torch.cat(S), torch.cat(T), torch.cat(X)
   intra = layer[S] == layer[T]
   loops = bool((S == T).any())

   # new edges of each layer, interlayer edges kept
   L = 1 if codes is None else len(net.layer_labels)
   count = torch.bincount(layer[S[intra]], minlength = L).tolist()
   kind = torch.as_tensor(types if types is not None else np.zeros(N, dtype = np.int8), device = device)
   s, t, rows = [S[~intra]], [T[~intra]], [torch.nonzero(~intra).squeeze(1)]
   for l in range(L):
      nodes = torch.nonzero(layer == l).squeeze(1)
      nodes = nodes[torch.argsort(kind[nodes], stable = True)]   # type 0 first
      n0 = None if types is None else int((kind[nodes] == 0).sum())
      a, b = make(l, nodes, n0, count[l], loops, gen)
      s.append(nodes[a])
      t.append(nodes[b])
      r = _attribute_rows(torch.nonzero(intra & (layer[S] == l)).squeeze(1), a.shape[0], gen, device)
      rows.append(torch.full((a.shape[0],), -1, dtype = torch.long, device = device) if r is None else r)
   s, t, rows = torch.cat(s), torch.cat(t), torch.cat(rows)
   ea = torch.where((rows >= 0)[:, None], X[rows.clamp(min = 0)], torch.nan) if X.shape[0] else X.new_full((rows.shape[0], X.shape[1]), torch.nan)
   return _subset(net, s.cpu().numpy(), t.cpu().numpy(), ea.cpu().numpy().astype(np.float32), np.arange(N))


# ------------------------------------------------------------------------------------------------
# Random graphs: Gnm, Gnp
# ------------------------------------------------------------------------------------------------


def _erdos_renyi(net, m, self_loops, seed, gnp = False, p = None):
   """
   Gnm or Gnp randomization of a network: see :func:`gnm_random_graph` and
   :func:`gnp_random_graph`.
   """
   if self_loops and net.bipartite_labels is not None:
      raise ValueError("bipartite networks have no self loops")

   def make(l, nodes, n0, count, loops, gen):
      sl = (loops and n0 is None) if self_loops is None else self_loops
      k = _layer_value(m, count, net, l, "m", False)
      if gnp:   # binomial number of edges, then Gnm
         pairs = _pair_count(nodes.shape[0], net.directed, sl, n0)
         q = _layer_value(p, None, net, l, "p", True)
         k = _binomial(pairs, (min(count / pairs, 1) if pairs else 0) if q is None else q, gen, nodes.device)
      return _gnm_edges(nodes.shape[0], k, net.directed, sl, gen, nodes.device, n0)

   return _randomize(net, seed, make)


def _check_net(net, n, directed, device):
   """
   Check that the parameters given with a network to randomize are those of the network.

   :raises ValueError: If ``n``, ``directed`` or ``device`` are given and differ from ``net``.
   """
   if n is not None and n != (net.number_of_nodes() if net.bipartite_labels is None else (sum(net.bipartite_split), net.number_of_nodes() - sum(net.bipartite_split))):
      raise ValueError("n is the number of nodes of net: leave it out")
   if directed is not None and directed != net.directed:
      raise ValueError("directed is the direction of net: leave it out")
   if device is not None and torch.device(device) != net.components[0].edge_index.device:
      raise ValueError("device is the device of net: leave it out, or move net there first")


def _random_graph(n, m, directed, self_loops, seed, device, p = None):
   """
   Gnm random graph on new nodes, or Gnp with ``p``: see :func:`gnm_random_graph` and
   :func:`gnp_random_graph`.
   """
   directed, self_loops, device = bool(directed), bool(self_loops), torch.device("cpu" if device is None else device)
   bipartite = isinstance(n, (tuple, list))
   if bipartite and self_loops:
      raise ValueError("bipartite networks have no self loops")
   n0, n = (int(n[0]), int(n[0]) + int(n[1])) if bipartite else (None, int(n))
   gen = _generator(seed, device)
   m = int(m) if p is None else _binomial(_pair_count(n, directed, self_loops, n0), p, gen, device)
   s, t = _gnm_edges(n, m, directed, self_loops, gen, device, n0)
   names = np.arange(n)
   types = (names >= n0).astype(np.int8) if bipartite else None
   qtable = (lambda perm: pd.DataFrame({BIPARTITE: pd.Categorical(types[perm])})) if bipartite else None
   return _build(s.cpu().numpy(), t.cpu().numpy(), names, np.zeros((m, 0), dtype = np.float32), [], None, [], directed, device, qtable, [BIPARTITE] if bipartite else [], "wide", types, (0, 1) if bipartite else None)


def gnm_random_graph(n = None, m = None, net = None, directed = None, self_loops = None, seed = None, device = None):
   """
   Gnm random graph (Erdős and Rényi, 1959): ``m`` edges placed uniformly at random among all
   possible pairs of ``n`` nodes, without parallel edges.

   The pairs are distinct node pairs (ordered if directed), with or without self loops. A
   bipartite random graph has ``n = (n0, n1)`` nodes of two types and its edges only join nodes
   of different types (if directed, in either direction). Sparse graphs draw random pairs until
   ``m`` are distinct, dense ones permute all pairs, so the time is about linear in ``m``, on the
   device.

   Without ``net``, nodes are the integers ``0`` to ``n - 1`` without attributes; bipartite nodes
   ``0`` to ``n0 - 1`` have :data:`~graphoskop.BIPARTITE` 0, the others 1. With ``net``, the
   result is a randomization of ``net`` (see the module description) and:

   * ``n`` and ``directed`` are those of ``net`` (giving others raises an error), and so are the
     node types of bipartite networks;
   * ``m`` is the number of edges of ``net`` (parallel edges and self loops count, as in
     :meth:`~graphoskop.Network.number_of_edges`), unless given;
   * ``self_loops`` is whether ``net`` has any self loop, unless given;
   * in multilayer networks each layer is randomized on its own, with its own copies and number
     of intra-layer edges (``m`` can be a dictionary from layer to number of edges, for the
     layers to change), drawing attributes from the edges of that layer; interlayer edges
     (couplings included) are kept as they are.

   The network is split into weakly connected components as usual, so the internal ids
   ``(component, index)`` of the nodes of a randomization differ from those in ``net``; their
   input ids do not.

   :param n: Number of nodes, or ``(n0, n1)`` for a bipartite graph; ``None`` with ``net``.
   :param m: Number of edges (in multilayer networks, a dictionary from layer to number of
      edges), or ``None`` to take it from ``net``.
   :param net: A :class:`~graphoskop.Network` to randomize, or ``None``.
   :param directed: Whether the edges are directed; ``None`` for undirected, or for the
      direction of ``net``.
   :param self_loops: Whether edges from a node to itself are allowed; ``None`` for no, or for
      whether ``net`` has them. Bipartite graphs have none.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The random graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If neither ``n`` and ``m`` nor ``net`` are given, if ``n``, ``directed``
      or ``device`` differ from those of ``net``, if ``m`` exceeds the number of possible pairs,
      if a multilayer ``m`` is not a dictionary (or names unknown layers), or if self loops are
      asked for a bipartite graph.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.gnm_random_graph(1000, 5000, seed = 1)                    # undirected, nodes 0 to 999
      g = gsk.gnm_random_graph(1000, 5000, directed = True, device = "cuda")
      b = gsk.gnm_random_graph((100, 300), 1000)                        # bipartite: 100 + 300 nodes

      net = gsk.load_network("edges.tsv", "nodes.tsv")                  # column weight; age, city
      rnd = gsk.gnm_random_graph(net = net, seed = 1)                   # same nodes and attributes, random edges
      gsk.assortativity(net, "city"), gsk.assortativity(rnd, "city")    # observed vs random
      rnd = gsk.gnm_random_graph(m = 2 * net.number_of_edges(), net = net)   # twice as dense
   """
   if net is not None:
      _check_net(net, n, directed, device)
      return _erdos_renyi(net, m, self_loops, seed)
   if n is None or m is None:
      raise ValueError("give n and m, or net")
   return _random_graph(n, m, directed, self_loops, seed, device)


def gnp_random_graph(n = None, p = None, net = None, directed = None, self_loops = None, seed = None, device = None):
   """
   Gnp random graph (Gilbert, 1959; the Erdős–Rényi model): each possible pair of ``n`` nodes is
   an edge with probability ``p``, independently of the others.

   Pairs, bipartite graphs, self loops and node ids are as in :func:`gnm_random_graph`. The
   number of edges is drawn from its binomial distribution over all possible pairs, then that
   many pairs are placed uniformly at random, which is the same distribution as testing every
   pair (a Gnp graph with ``m`` edges is a Gnm graph), in time about linear in the number of
   edges instead of ``n ** 2``.

   With ``net``, the result is a randomization of ``net`` (see the module description), as in
   :func:`gnm_random_graph`, except that the number of edges is random:

   * ``p`` is the density of ``net``, the number of its edges (parallel edges and self loops
     count) over the number of possible pairs (with self loops if ``net`` has any, unless
     ``self_loops`` is given), so that ``net`` has as many edges as expected; or as given;
   * in multilayer networks each layer gets the density of its own intra-layer edges; ``p`` can
     be one probability for all layers, or a dictionary from layer to probability, for the
     layers to change. Interlayer edges (couplings included) are kept as they are.

   As the number of edges varies, new edges take the attributes of the original edges of their
   layer as described in the module description: shuffled, with some left out if fewer edges are
   drawn, and some repeated if more are.

   :param n: Number of nodes, or ``(n0, n1)`` for a bipartite graph; ``None`` with ``net``.
   :param p: Probability of each edge (in multilayer networks, also a dictionary from layer to
      probability), or ``None`` to take it from ``net``.
   :param net: A :class:`~graphoskop.Network` to randomize, or ``None``.
   :param directed: Whether the edges are directed; ``None`` for undirected, or for the
      direction of ``net``.
   :param self_loops: Whether edges from a node to itself are allowed; ``None`` for no, or for
      whether ``net`` has them. Bipartite graphs have none.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The random graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If neither ``n`` and ``p`` nor ``net`` are given, if ``p`` is not between
      0 and 1, if ``n``, ``directed`` or ``device`` differ from those of ``net``, if a
      dictionary ``p`` is given for a network that is not multilayer (or names unknown layers),
      or if self loops are asked for a bipartite graph.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.gnp_random_graph(1000, 0.01, seed = 1)                    # about 4995 edges
      b = gsk.gnp_random_graph((100, 300), 0.05, directed = True)       # bipartite

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      rnd = gsk.gnp_random_graph(net = net, seed = 1)                   # same nodes, density of net
      ml = gsk.load_network("multiplex.tsv", multilayer = True, coupling = "all")
      rnd = gsk.gnp_random_graph(net = ml, p = {"work": 0.1})           # other layers: their own density
   """
   if net is not None:
      _check_net(net, n, directed, device)
      return _erdos_renyi(net, None, self_loops, seed, True, p)
   if n is None or p is None:
      raise ValueError("give n and p, or net")
   return _random_graph(n, None, directed, self_loops, seed, device, p)


# ------------------------------------------------------------------------------------------------
# Clustered graphs: caveman, small world
# ------------------------------------------------------------------------------------------------


def _cave_edges(sizes, connected, directed):
   """
   Edges of a caveman graph: a clique per cave, caves of consecutive nodes.

   :param sizes: int64 tensor, number of nodes of each cave.
   :param connected: Whether to rewire one edge per cave into a ring of caves (Watts, 1999).
   :param directed: Whether to give every edge in both directions.
   :return: Source and target ids, int64 tensors.
   """
   L = sizes.shape[0]
   starts = torch.cumsum(sizes, 0) - sizes
   P = sizes * (sizes - 1) // 2
   cave = torch.repeat_interleave(torch.arange(L, device = sizes.device), P)
   k = torch.arange(int(P.sum()), device = sizes.device) - torch.repeat_interleave(torch.cumsum(P, 0) - P, P)
   u, v = _triangle(k)
   s, t = starts[cave] + u, starts[cave] + v
   if connected and L > 1:   # edge between the first two nodes of a cave -> first node to the last of the previous cave
      keep = k != 0
      a, b = starts, torch.roll(starts + sizes - 1, 1)
      ring = torch.unique(torch.stack([torch.maximum(a, b), torch.minimum(a, b)]), dim = 1)   # two caves of one node: one edge
      s, t = torch.cat([s[keep], ring[0]]), torch.cat([t[keep], ring[1]])
   return (torch.cat([s, t]), torch.cat([t, s])) if directed else (s, t)


def _cave_sizes(n, pairs, l = None, k = None):
   """
   Sizes of the caves of ``n`` nodes, as equal as possible (they differ by at most one).

   The number of caves is ``l``; or ``n / k`` rounded; or else the one whose caves have the
   number of node pairs closest to ``pairs`` (the fewest caves, on ties).

   :param n: Number of nodes.
   :param pairs: Target number of connected node pairs.
   :param l: Number of caves, or ``None``.
   :param k: Target size of the caves, or ``None``.
   :return: Sizes, as a list (empty if ``n`` is 0).
   :raises ValueError: If ``l`` is not between 1 and ``n``.
   """
   if n == 0:
      return []
   if l is None and k is not None:
      l = min(max(1, round(n / k)), n)
   if l is None:
      c = np.arange(1, n + 1)
      q, r = n // c, n % c
      l = int(c[np.argmin(np.abs((c - r) * q * (q - 1) / 2 + r * (q + 1) * q / 2 - pairs))])
   if not 1 <= l <= n:
      raise ValueError(f"cannot split {n:,} nodes into {l:,} caves")
   q, r = divmod(n, l)
   return [q + 1] * r + [q] * (l - r)


def caveman_graph(l = None, k = None, net = None, connected = False, directed = None, attribute = "cave", seed = None, device = None):
   """
   Caveman graph (Watts, 1999): ``l`` caves, each a clique of ``k`` nodes.

   Caves are disconnected, or, with ``connected = True``, joined in a ring (the connected caveman
   graph): in every cave the edge between its first two nodes is removed, and its first node is
   linked to the last node of the previous cave instead, so that every node keeps degree
   ``k - 1`` (a cave of one node only gains its ring edge). Directed caveman graphs have every
   edge in both directions. The cave of every node is stored as the numeric node attribute
   ``attribute``, a planted partition for e.g. :func:`~graphoskop.modularity`.

   Without ``net``, nodes are the integers ``0`` to ``n - 1``, cave by cave; ``k`` can also be a
   list of cave sizes, for caves of different sizes. With ``net``, the result is a randomization
   of ``net`` (see the module description): its nodes are assigned at random to caves, whose
   sizes differ by at most one. As every node of a caveman graph has degree ``k - 1``, the size
   of the caves follows from the average degree of ``net``: the number of caves is the one that
   gives the number of edges closest to that of ``net`` (pairs of nodes in either direction, if
   directed), unless ``l`` (number of caves) or ``k`` (cave size, from which ``l`` is ``n / k``
   rounded) is given. In multilayer networks each layer is randomized on its own, with its own
   caves (numbered across layers, so that no two layers share one); ``l`` can be a dictionary
   from layer to number of caves, ``k`` one size for all layers or a dictionary. Interlayer
   edges are kept. The new edges take the attributes of the old ones of their layer.

   :param l: Number of caves; ``None`` with ``net``, or if ``k`` is a list.
   :param k: Number of nodes of each cave, or list of cave sizes; ``None`` with ``net``.
   :param net: A :class:`~graphoskop.Network` to randomize, or ``None``.
   :param connected: Whether to join the caves in a ring.
   :param directed: Whether the edges are directed (both directions); ``None`` for undirected,
      or for the direction of ``net``.
   :param attribute: Name of the node attribute holding the cave of each node (overwritten if
      ``net`` has it), or ``None`` for no attribute.
   :param seed: Seed of the random numbers (with ``net``), or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The caveman graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If neither ``l`` and ``k`` nor ``net`` are given, if both ``l`` and ``k``
      are given with ``net``, if ``net`` is bipartite (it has no cliques), if ``directed`` or
      ``device`` differ from those of ``net``, if a layer cannot be split into ``l`` caves, or for
      dictionaries as in :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.caveman_graph(10, 5)                                   # 10 disconnected 5-cliques
      g = gsk.caveman_graph(10, 5, connected = True)                 # in a ring: one component
      g = gsk.caveman_graph(k = [3, 4, 5])                           # caves of different sizes
      gsk.modularity(g, "cave")

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      rnd = gsk.caveman_graph(net = net, seed = 1)                   # nodes in random caves, same average degree
      gsk.assortativity(rnd, "age")                                  # attributes of net, on caves
   """
   if net is None:
      if k is None or (l is None and np.ndim(k) == 0):
         raise ValueError("give l and k, a list of sizes k, or net")
      sizes = list(map(int, k)) if np.ndim(k) else [int(k)] * int(l)
      if np.ndim(k) and l is not None and l != len(sizes):
         raise ValueError(f"l is {l}, but k gives {len(sizes)} cave sizes")
      directed, device = bool(directed), torch.device("cpu" if device is None else device)
      s, t = _cave_edges(torch.tensor(sizes, dtype = torch.long, device = device), connected, directed)
      cave = np.repeat(np.arange(len(sizes)), sizes).astype(np.float32)
      names = np.arange(len(cave))
      table = None if attribute is None else lambda perm: cave[perm][:, None]
      return _build(s.cpu().numpy(), t.cpu().numpy(), names, np.zeros((s.shape[0], 0), dtype = np.float32), [], table, [] if attribute is None else [attribute], directed, device)
   if net.bipartite_labels is not None:
      raise ValueError("bipartite networks have no cliques, so they cannot be caveman graphs")
   if l is not None and k is not None:
      raise ValueError("with net, give l or k, not both: the number of nodes fixes the other")
   _check_net(net, None, directed, device)
   caves = []

   def make(j, nodes, n0, count, loops, gen):
      n = nodes.shape[0]
      sizes = _cave_sizes(n, count / 2 if net.directed else count, _layer_value(l, None, net, j, "l", False), _layer_value(k, None, net, j, "k", True))
      perm = torch.randperm(n, generator = gen, device = nodes.device)   # random caves
      off = max((int(c[-1]) + 1 for _, c in caves if len(c)), default = 0)   # caves numbered across layers
      caves.append((nodes[perm], np.repeat(np.arange(off, off + len(sizes)), sizes)))
      s, t = _cave_edges(torch.tensor(sizes, dtype = torch.long, device = nodes.device), connected, net.directed)
      return perm[s], perm[t]

   new = _randomize(net, seed, make)
   if attribute is not None:
      cave = np.empty(int(net.node_offsets[-1]), dtype = np.float32)
      for pos, c in caves:
         cave[pos.cpu().numpy()] = c
      where = {x: i for i, x in enumerate(np.concatenate(net.node_ids).tolist())}
      new.set_node_attribute(attribute, [cave[[where[x] for x in ids.tolist()]] for ids in new.node_ids], None)
   return new


def _ring_lattice(n, m, device):
   """
   Ring lattice with ``m`` edges: every node linked to its nearest neighbors on the ring, one
   distance at a time. The edges of the last, partial distance are chosen evenly spaced along the
   ring and sharing as few nodes as possible (a maximum matching first), so that degrees differ
   by at most one whenever possible.

   :param n: Number of nodes, in ring order.
   :param m: Number of edges, at most ``n (n - 1) / 2``.
   :param device: Device.
   :return: Source and target ids, int64 tensors.
   """
   s, t, d = [], [], 1
   while m > 0:
      c = n if 2 * d < n else n // 2   # ring of distance n / 2: each pair once
      take = min(c, m)
      i = torch.arange(c, device = device)
      if take < c:   # partial: along each cycle i, i + d, i + 2 d, ... alternate edges first (a matching)
         g = math.gcd(n, d)
         L = n // g
         pos = (i // g) * pow(d // g, -1, L) % L   # position of edge (i, i + d) on its cycle
         group = torch.where(pos % 2 == 1, 2, torch.where((pos == L - 1) & (L % 2 == 1), 1, 0))   # 1: reaches the node left out
         pick, left = [], take
         for o in range(3):   # evenly spaced within each group
            e = i[group == o]
            j = min(left, e.shape[0])
            pick.append(e[torch.arange(j, device = device) * e.shape[0] // max(j, 1)])
            left -= j
         i = torch.cat(pick)
      s.append(i)
      t.append((i + d) % n)
      m, d = m - take, d + 1
   z = torch.zeros(0, dtype = torch.long, device = device)
   return torch.cat([z] + s), torch.cat([z] + t)


def _connected(n, s, t):
   """
   Whether a graph is connected.

   :param n: Number of nodes.
   :param s: Source ids, int64 tensor.
   :param t: Target ids, int64 tensor.
   :return: ``True`` if it has one component.
   """
   adj = sp.coo_matrix((np.ones(s.shape[0]), (s.cpu().numpy(), t.cpu().numpy())), shape = (n, n))
   return n <= 1 or connected_components(adj, directed = False)[0] == 1


def _retarget(n, u, v, kept, gen, rounds = 100):
   """
   Rewire the target of edges: ``(u, v)`` becomes ``(u, w)``, ``w`` uniformly random among the
   nodes not ``u`` and not already linked to ``u``, the edges being rewired in parallel.

   Targets that clash (with a kept edge, or another rewired edge) are drawn again, for up to
   ``rounds`` rounds. Edges still clashing (only in very dense graphs) go to uniformly random
   free pairs, so that no edge is lost.

   :param n: Number of nodes.
   :param u: Fixed endpoints, int64 tensor.
   :param v: Old targets (unused: kept for the record of which edges move).
   :param kept: Keys (see :func:`_pair_key`) of the edges not rewired.
   :param gen: Random generator.
   :param rounds: Maximum number of draws.
   :return: New sources and targets.
   """
   dev = u.device
   done_u, done_w = torch.zeros(0, dtype = torch.long, device = dev), torch.zeros(0, dtype = torch.long, device = dev)
   for _ in range(rounds):
      if u.shape[0] == 0:
         break
      w = torch.randint(n - 1, u.shape, generator = gen, device = dev)
      w = w + (w >= u).long()   # never u itself
      key = _pair_key(u, w)
      ok = ~torch.isin(key, torch.cat([kept, _pair_key(done_u, done_w)]))
      _, g = torch.unique(key, return_inverse = True)
      first = torch.full((int(g.max()) + 1,), key.shape[0], device = dev).scatter_reduce(0, g, torch.arange(key.shape[0], device = dev), "amin")
      ok &= torch.zeros_like(ok).index_fill_(0, first, True)   # one of equal new edges
      done_u, done_w = torch.cat([done_u, u[ok]]), torch.cat([done_w, w[ok]])
      u = u[~ok]
   a, b = _triangle(_choose(n * (n - 1) // 2, u.shape[0], gen, dev, torch.cat([kept, _pair_key(done_u, done_w)])))
   return torch.cat([done_u, a]), torch.cat([done_w, b])


def _small_world(n, m, p, rewire, connected, tries, gen, device):
   """
   Edges of a Watts-Strogatz graph: see :func:`watts_strogatz_graph`.

   :return: Source and target ids, int64 tensors.
   :raises ValueError: If no connected graph is found in ``tries`` attempts.
   """
   if m > n * (n - 1) // 2:
      raise ValueError(f"cannot place {m:,} edges: there are only {n * (n - 1) // 2:,} possible ones")
   if connected and m < n - (n == 2):
      raise ValueError("a connected ring lattice needs at least n edges (average degree 2)")
   s, t = _ring_lattice(n, m, device)
   for _ in range(tries if connected else 1):
      pick = torch.rand(m, generator = gen, device = device) < p
      if rewire == "target":
         a, b = _retarget(n, s[pick], t[pick], _pair_key(s[~pick], t[~pick]), gen)
         a, b = torch.cat([s[~pick], a]), torch.cat([t[~pick], b])
      else:   # new edges among all free pairs
         keep = ~pick if rewire == "both" else torch.ones_like(pick)
         r = int(pick.sum()) if rewire == "both" else min(int(pick.sum()), n * (n - 1) // 2 - m)   # "add": up to a complete graph
         u, w = _triangle(_choose(n * (n - 1) // 2, r, gen, device, _pair_key(s[keep], t[keep])))
         a, b = torch.cat([s[keep], u]), torch.cat([t[keep], w])
      if not connected or _connected(n, a, b):
         return a, b
   raise ValueError(f"no connected graph in {tries} attempts: lower p, or raise k or tries")


def watts_strogatz_graph(n = None, k = None, p = None, net = None, rewire = "target", connected = False, tries = 100, seed = None, device = None):
   """
   Watts-Strogatz small world graph (Watts and Strogatz, 1998): a ring lattice whose edges are
   rewired at random with probability ``p``.

   The ``n`` nodes sit on a ring, each linked to its nearest neighbors: with average degree ``k``,
   the lattice has ``round(n k / 2)`` edges, linking every node to the ``k / 2`` nearest nodes on
   each side for even ``k`` (the lattice of NetworkX). Any ``k`` is allowed: for odd (or
   fractional) ``k`` the last distance links only some pairs, evenly spaced, so that the lattice
   has exactly the number of edges asked and degrees differ by at most one whenever possible.
   Then each edge, independently with probability ``p``, is rewired according to ``rewire``:

   * ``"target"``: one endpoint moves, as in the original model: ``(u, v)`` becomes ``(u, w)``,
     ``w`` uniformly random among the nodes not linked to ``u``. All picked edges are rewired at
     once (on the device) instead of one after the other; in very dense graphs, an edge whose
     node is already linked to all the others moves to a random free pair instead;
   * ``"both"``: both endpoints move: the edge becomes a uniformly random pair of unlinked nodes;
   * ``"add"``: the edge stays, and a new edge joins a uniformly random pair of unlinked nodes
     (Newman and Watts, 1999), so there are about ``(1 + p) n k / 2`` edges (at most all pairs)
     and the lattice is never broken.

   The graph has no self loops nor parallel edges, and is undirected. With ``connected = True``,
   rewiring is drawn again, up to ``tries`` times, until the graph is connected (the lattice must
   be, so ``k`` at least 2).

   Without ``net``, nodes are the integers ``0`` to ``n - 1`` in ring order. With ``net``, the
   result is a randomization of ``net`` (see the module description): its nodes are placed on the
   ring in random order, and ``n`` and the number of edges (so ``k``, the average degree) are
   those of ``net``, unless ``k`` is given; ``p`` is always needed. In multilayer networks each
   layer is randomized on its own, with its own ring and number of edges (``k`` can be one value
   for all layers or a dictionary from layer to average degree); interlayer edges are kept.

   :param n: Number of nodes; ``None`` with ``net``.
   :param k: Average degree; ``None`` with ``net``, for its own.
   :param p: Probability of rewiring each edge.
   :param net: An undirected, not bipartite :class:`~graphoskop.Network` to randomize, or
      ``None``.
   :param rewire: ``"target"``, ``"both"`` or ``"add"``.
   :param connected: Whether the graph must be connected (each layer, in multilayer networks).
   :param tries: Maximum number of attempts to get a connected graph.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The small world graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If ``p`` is missing or not between 0 and 1, if neither ``n`` and ``k`` nor
      ``net`` are given, if ``rewire`` is not valid, if ``net`` is directed or bipartite, if
      ``n`` or ``device`` differ from those of ``net``, if there are more edges than node pairs,
      if ``connected`` is asked with ``k`` below 2 or no connected graph is found in ``tries``
      attempts, or for dictionaries as in :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.watts_strogatz_graph(1000, 6, 0.1, seed = 1)                  # rewire one endpoint
      g = gsk.watts_strogatz_graph(1000, 6, 0.1, rewire = "both", connected = True)
      g = gsk.watts_strogatz_graph(1000, 6, 0.1, rewire = "add")            # Newman-Watts shortcuts

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      rnd = gsk.watts_strogatz_graph(net = net, p = 0.05, seed = 1)          # same n and m, nodes on a ring
      gsk.average_clustering(net), gsk.average_clustering(rnd)
   """
   if p is None or not 0 <= p <= 1:
      raise ValueError(f"p must be given, between 0 and 1, got {p}")
   if rewire not in ("target", "both", "add"):
      raise ValueError(f"unknown rewire: {rewire}")
   if net is None:
      if n is None or k is None:
         raise ValueError("give n and k, or net")
      device = torch.device("cpu" if device is None else device)
      s, t = _small_world(int(n), round(n * k / 2), p, rewire, connected, tries, _generator(seed, device), device)
      return _build(s.cpu().numpy(), t.cpu().numpy(), np.arange(int(n)), np.zeros((s.shape[0], 0), dtype = np.float32), [], None, [], False, device)
   if net.directed or net.bipartite_labels is not None:
      raise ValueError("Watts-Strogatz graphs are undirected and not bipartite (see to_undirected)")
   _check_net(net, n, None, device)

   def make(l, nodes, n0, count, loops, gen):
      size = nodes.shape[0]
      kl = _layer_value(k, None, net, l, "k", True)
      perm = torch.randperm(size, generator = gen, device = nodes.device)   # random places on the ring
      s, t = _small_world(size, count if kl is None else round(size * kl / 2), p, rewire, connected, tries, gen, nodes.device)
      return perm[s], perm[t]

   return _randomize(net, seed, make)


# ------------------------------------------------------------------------------------------------
# Broad degree graphs: preferential attachment, link selection, copying, Holme-Kim
# ------------------------------------------------------------------------------------------------


def _attachment_counts(n, m, device):
   """
   Number of edges each node brings when it arrives, as even as possible: node ``i`` links to
   ``min(i, L)`` earlier nodes, and evenly spaced nodes to one more, for ``m`` edges in all.

   :param n: Number of nodes.
   :param m: Number of edges, at most ``n (n - 1) / 2``.
   :param device: Device.
   :return: int64 tensor of length ``n`` (0 for node 0).
   """
   L = np.arange(n)
   S = L * (L + 1) // 2 + L * (n - 1 - L)   # edges with level L
   L = int(np.flatnonzero(S <= m)[-1]) if n > 1 else 0
   i = torch.arange(n, device = device)
   e = torch.minimum(i, torch.tensor(L, device = device))
   r = m - int(S[L]) if n > 1 else 0
   late = i[L + 1:]
   e[late[torch.arange(r, device = device) * late.shape[0] // max(r, 1)]] += 1
   return e


def _expand(starts, counts):
   """
   Concatenated ranges ``starts[i]`` to ``starts[i] + counts[i] - 1``.

   :param starts: int64 tensor.
   :param counts: int64 tensor.
   :return: int64 tensor.
   """
   off = torch.cumsum(counts, 0) - counts
   return torch.repeat_interleave(starts - off, counts) + torch.arange(int(counts.sum()), device = starts.device)


def _preferential(n, m, gen, device, links = False):
   """
   Edges of a Barabási-Albert graph: see :func:`barabasi_albert_graph`.

   Edges are stored in arrival order, endpoints in ``V`` (``V[2 j]`` the new node of edge ``j``,
   ``V[2 j + 1]`` its target). A target copies the endpoint at a uniformly random position among
   those of the edges that arrived before its node (Batagelj and Brandes, 2005), which picks a
   node with probability proportional to its degree. Each edge has a stream of such candidate
   positions, and takes the first one whose endpoint is not a target of an earlier edge of its
   node: exactly the sequential model, where a node draws until its targets are distinct.

   All copies are first resolved at once by pointer doubling, with the first candidate of every
   edge. Nodes with repeated targets are then recomputed from the start of their streams (drawing
   more candidates as needed), and so are, in turn, the nodes with a candidate copying an
   endpoint that changed, until nothing changes. Copies point backwards, so this converges to the
   sequential result.

   With ``links``, this is the link selection model: a candidate is a random earlier edge and one
   of its two endpoints, which is the same as a random endpoint position, and is also rejected if
   its edge was already used by an earlier edge of its node: a node never uses a link twice.

   :param links: Whether a node uses each link at most once (link selection).
   :return: Source and target ids, int64 tensors.
   :raises ValueError: If there are more edges than node pairs.
   """
   if m > n * (n - 1) // 2:
      raise ValueError(f"cannot place {m:,} edges: there are only {n * (n - 1) // 2:,} possible ones")
   e = _attachment_counts(n, m, device)
   if m == 0:
      return e[:0], e[:0]
   src = torch.repeat_interleave(torch.arange(n, device = device), e)
   start = torch.cumsum(e, 0) - e   # first edge of each node
   first = start[src]
   j = torch.arange(m, device = device)
   rank = j - first
   V = torch.stack([src, torch.zeros_like(src)], 1).view(-1)
   T = V[1::2]   # targets (a view)
   fixed = e[src] == src   # nodes linking to all earlier nodes
   T[fixed] = rank[fixed]
   pa = j[~fixed]
   lone = first[pa] == 0   # no earlier edge (sparse graphs only): uniformly random earlier node
   T[pa[lone]] = (torch.rand(int(lone.sum()), generator = gen, device = device, dtype = torch.float64) * src[pa[lone]]).long()
   pa = pa[~lone]
   draw = lambda s: (torch.rand(s.shape[0], generator = gen, device = device, dtype = torch.float64) * 2 * first[s]).long()   # float64: exact positions beyond 2 ** 24

   # first candidates, all copies resolved at once
   ptr = torch.arange(2 * m, device = device)   # known endpoints point to themselves
   ptr[2 * pa + 1] = draw(pa)
   root = ptr
   while True:   # pointer doubling: the known endpoint at the end of each chain of copies
      nxt = torch.index_select(root, 0, root)
      if torch.equal(nxt, root):
         break
      root = nxt
   V.copy_(torch.index_select(V, 0, root))
   P = ptr[1::2].clone()   # position copied by each edge: its link is P // 2
   copied, order = torch.sort(ptr[2 * pa + 1])   # who copies whom, first candidates
   copier = pa[order]
   xe, xp = j[:0], j[:0]   # further candidates: edge and position, in order of drawing

   def clash(E, v, q):   # whether target v (or link q // 2) is used by an earlier edge of the node of each edge E
      bad = torch.zeros_like(E, dtype = torch.bool)
      for d in range(1, int(rank[E].max()) + 1 if E.shape[0] else 1):
         f = (E - d).clamp(min = 0)
         bad |= (rank[E] >= d) & ((v == T[f]) | (links & (q // 2 == P[f] // 2)))
      return bad

   dirty = j[:0]
   for d in range(1, int(e.max())):
      same = (T[d:] == T[:-d]) | (links & (P[d:] // 2 == P[:-d] // 2))
      dirty = torch.cat([dirty, src[d:][same & (src[d:] == src[:-d]) & ~fixed[d:]]])
   dirty = torch.unique(dirty)
   while dirty.shape[0]:
      E = _expand(start[dirty], e[dirty])
      old = T[E].clone()
      for a in range(int(e.max())):   # rank by rank: each edge sees the earlier edges of its node
         R = E[rank[E] == a]
         q = ptr[2 * R + 1]
         v = V[q]
         ok = ~clash(R, v, q)
         T[R[ok]], P[R[ok]] = v[ok], q[ok]
         R = R[~ok]
         while R.shape[0]:   # further candidates, first acceptable one
            i = torch.nonzero(torch.isin(xe, R)).squeeze(1)
            i = i[~clash(xe[i], V[xp[i]], xp[i])]
            got, inv = torch.unique(xe[i], return_inverse = True)
            best = torch.full((got.shape[0],), xe.shape[0], device = device).scatter_reduce(0, inv, i, "amin")
            T[got], P[got] = V[xp[best]], xp[best]
            R = R[~torch.isin(R, got)]
            xe, xp = torch.cat([xe, R]), torch.cat([xp, draw(R)])
      S = 2 * E[T[E] != old] + 1   # changed endpoints: their copiers are recomputed
      lo, hi = torch.searchsorted(copied, S), torch.searchsorted(copied, S, right = True)
      dirty = torch.unique(src[torch.cat([copier[_expand(lo, hi - lo)], xe[torch.isin(xp, S)]])])
   return V[0::2], T


def _uniform(gen):
   """
   Uniform random numbers one at a time, for the sequential growth models, drawn in batches with
   a CPU generator seeded from ``gen``.

   :param gen: Random generator, on any device.
   :return: A function returning a float in ``[0, 1)``, and the CPU generator.
   """
   cpu = torch.Generator().manual_seed(int(torch.randint(2 ** 62, (1,), generator = gen, device = gen.device)))
   more = iter(())

   def draw():
      nonlocal more
      x = next(more, None)
      if x is None:
         more = iter(torch.rand(4096, generator = cpu, dtype = torch.float64).tolist())
         x = next(more)
      return x

   return draw, cpu

def _copying(n, m, gen, device):
   """
   Edges of a copying graph: see :func:`copying_graph`.

   Each target depends on the whole neighborhood of a random node when its node arrives, so
   arrivals are simulated one after the other, in Python on the CPU; the random numbers come from
   ``gen``. Neighbors are kept in order of linking, and a node's own new edges count only after
   it has chosen all its targets.

   :return: Source and target ids, int64 tensors on ``device``.
   :raises ValueError: If there are more edges than node pairs.
   """
   if m > n * (n - 1) // 2:
      raise ValueError(f"cannot place {m:,} edges: there are only {n * (n - 1) // 2:,} possible ones")
   e = _attachment_counts(n, m, "cpu").tolist()
   draw, cpu = _uniform(gen)
   U, R = (torch.rand(m, generator = cpu, dtype = torch.float64).tolist() for _ in range(2))

   adj = [[] for _ in range(n)]
   T, j = [], 0
   for i in range(1, n):
      k = e[i]
      if k == i:   # links to all earlier nodes
         tg = list(range(i))
      elif j == 0:   # no edge to copy yet (sparse graphs only): a uniformly random earlier node
         tg = [int(draw() * i)] if k else []
      else:
         tg = []
         for a in range(k):
            u, r = U[j + a], R[j + a]
            while True:   # a random earlier node, and one of its neighbors not chosen yet
               nb = adj[int(u * i)]
               if nb:
                  x = nb[int(r * len(nb))]
                  if x not in tg:
                     break
               u, r = draw(), draw()
            tg.append(x)
      for x in tg:
         adj[x].append(i)
         adj[i].append(x)
      T += tg
      j += k
   return torch.repeat_interleave(torch.arange(n), torch.tensor(e)).to(device), torch.tensor(T, dtype = torch.long).to(device)


def _triads(n, m, p, gen, device):
   """
   Edges of a Holme-Kim graph: see :func:`holme_kim_graph`.

   Arrivals are simulated one after the other, in Python on the CPU, as in :func:`_copying`: a
   triad formation step needs the neighbors of a node when the new node arrives. Preferential
   attachment picks a uniformly random endpoint among those of the earlier edges.

   :param p: Probability of a triad formation step.
   :return: Source and target ids, int64 tensors on ``device``.
   :raises ValueError: If there are more edges than node pairs.
   """
   if m > n * (n - 1) // 2:
      raise ValueError(f"cannot place {m:,} edges: there are only {n * (n - 1) // 2:,} possible ones")
   e = _attachment_counts(n, m, "cpu").tolist()
   draw = _uniform(gen)[0]
   adj = [[] for _ in range(n)]
   ends, T = [], []   # endpoints of the earlier edges: a random one is a node chosen in proportion to its degree
   for i in range(1, n):
      k = e[i]
      if k == i:   # links to all earlier nodes
         tg = list(range(i))
      elif not ends:   # no edge yet (sparse graphs only): a uniformly random earlier node
         tg = [int(draw() * i)] if k else []
      else:
         tg, w = [], None
         for a in range(k):
            if a and draw() < p:   # triad formation: a neighbor of the last node chosen by attachment
               free = [x for x in adj[w] if x not in tg]
               if free:
                  tg.append(free[int(draw() * len(free))])
                  continue
            while True:   # preferential attachment, to a node not chosen yet
               x = ends[int(draw() * len(ends))]
               if x not in tg:
                  break
            tg.append(x)
            w = x
      for x in tg:
         adj[x].append(i)
         adj[i].append(x)
         ends += [i, x]
      T += tg
   return torch.repeat_interleave(torch.arange(n), torch.tensor(e)).to(device), torch.tensor(T, dtype = torch.long).to(device)

def _calibrate_triads(n, m, target, gen, cap = 100000):
   """
   Probability of triad formation giving a Holme-Kim graph the average clustering ``target``.

   Average clustering grows smoothly and almost linearly with ``p``, and depends little on the
   number of nodes above a few thousand (but much on the average degree), so it is matched on a
   simulated graph of at most ``cap`` nodes with the same average degree, by regula falsi
   (Illinois variant) on ``p``, every simulation with the same random numbers.

   :param n: Number of nodes.
   :param m: Number of edges.
   :param target: Average clustering to match.
   :param gen: Random generator.
   :param cap: Maximum number of nodes of the simulated graphs.
   :return: ``p``; 0 or 1 if ``target`` is out of reach, with a warning in the latter case.
   """
   from .network_statistics import average_clustering
   size = min(n, cap)
   edges = round(m * size / n) if n else 0
   seed = int(torch.randint(2 ** 62, (1,), generator = gen, device = gen.device))

   def clustering(p):
      s, t = _triads(size, edges, p, torch.Generator().manual_seed(seed), "cpu")
      return average_clustering(_build(s.numpy(), t.numpy(), np.arange(size), np.zeros((edges, 0), dtype = np.float32), [], None, [], False, "cpu"))

   if size < 3 or edges < 2:
      return 0.
   lo, hi = 0., 1.
   c_lo, c_hi = clustering(lo), clustering(hi)
   if target <= c_lo:
      return lo
   if target >= c_hi:
      warnings.warn(f"average clustering {target:.3f} is above that of any Holme-Kim graph of this size and density ({c_hi:.3f}): p = 1")
      return hi
   side = 0
   for _ in range(12):
      p = lo + (target - c_lo) * (hi - lo) / (c_hi - c_lo)
      c = clustering(p)
      if abs(c - target) < 0.002:
         break
      if c < target:
         lo, c_lo = p, c
         c_hi, side = (c_hi + target) / 2 if side == -1 else c_hi, -1   # Illinois: halve the stuck end
      else:
         hi, c_hi = p, c
         c_lo, side = (c_lo + target) / 2 if side == 1 else c_lo, 1
   return p

def _growth(n, k, net, seed, device, sample):
   """
   Graph grown by arrivals of nodes: see :func:`barabasi_albert_graph`,
   :func:`link_selection_graph`, :func:`copying_graph` and :func:`holme_kim_graph`.

   :param sample: Function ``sample(n, m, gen, device, l)`` giving the sources and targets of the
      ``m`` edges among ``n`` nodes, in order of arrival, for layer ``l`` (0 if not multilayer).
   """
   if net is None:
      if n is None or k is None:
         raise ValueError("give n and k, or net")
      device = torch.device("cpu" if device is None else device)
      s, t = sample(int(n), round(n * k / 2), _generator(seed, device), device, 0)
      return _build(s.cpu().numpy(), t.cpu().numpy(), np.arange(int(n)), np.zeros((s.shape[0], 0), dtype = np.float32), [], None, [], False, device)
   if net.directed or net.bipartite_labels is not None:
      raise ValueError("growth models make undirected graphs, not bipartite (see to_undirected)")
   _check_net(net, n, None, device)

   def make(l, nodes, n0, count, loops, gen):
      size = nodes.shape[0]
      kl = _layer_value(k, None, net, l, "k", True)
      perm = torch.randperm(size, generator = gen, device = nodes.device)   # random order of arrival
      s, t = sample(size, count if kl is None else round(size * kl / 2), gen, nodes.device, l)
      return perm[s], perm[t]

   return _randomize(net, seed, make)


def barabasi_albert_graph(n = None, k = None, net = None, seed = None, device = None):
   """
   Barabási-Albert graph (Barabási and Albert, 1999): nodes arrive one at a time and link to
   earlier nodes chosen with probability proportional to their degree (preferential attachment),
   giving a power law degree distribution with exponent about 3.

   ``k`` is the average degree: each node brings ``k / 2`` edges, to distinct earlier nodes, so
   ``k = 2 m`` is the classic model with ``m`` edges per node. The graph starts from the first
   ``k / 2 + 1`` nodes, each linked to all the earlier ones (a clique). Any ``k`` is allowed: for
   odd (or fractional) ``k``, evenly spaced nodes bring one edge more, so that the graph has
   exactly ``round(n k / 2)`` edges. Probabilities are those before the node arrives, as in the
   original model and in NetworkX (whose seed is a star instead of a clique). The graph is
   undirected, without self loops nor parallel edges, and connected if ``k`` is at least 2.

   Arrivals are simulated in parallel on the device: each new edge copies an endpoint at a
   random position in the list of the earlier edges (Batagelj and Brandes, 2005), copies of
   copies are resolved together, and targets a node already chose are drawn again.

   Without ``net``, nodes are the integers ``0`` to ``n - 1`` in order of arrival. With ``net``,
   the result is a randomization of ``net`` (see the module description): its nodes arrive in
   random order, and ``n`` and the number of edges (so ``k``) are those of ``net``, unless ``k``
   is given. In multilayer networks each layer is randomized on its own (``k`` can be one value
   for all layers or a dictionary from layer to average degree); interlayer edges are kept.

   :param n: Number of nodes; ``None`` with ``net``.
   :param k: Average degree; ``None`` with ``net``, for its own.
   :param net: An undirected, not bipartite :class:`~graphoskop.Network` to randomize, or
      ``None``.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The Barabási-Albert graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If neither ``n`` and ``k`` nor ``net`` are given, if ``net`` is directed
      or bipartite, if ``n`` or ``device`` differ from those of ``net``, if there are more edges
      than node pairs, or for dictionaries as in :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.barabasi_albert_graph(10000, 6, seed = 1)          # m = 3 edges per new node
      gsk.degree_power_law(g)                                    # exponent about 3

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      rnd = gsk.barabasi_albert_graph(net = net, seed = 1)       # same n and m, nodes arrive at random
   """
   return _growth(n, k, net, seed, device, lambda n, m, gen, device, l: _preferential(n, m, gen, device))


def link_selection_graph(n = None, k = None, net = None, seed = None, device = None):
   """
   Link selection graph: nodes arrive one at a time, and each picks random links of the network
   and connects to one of the two nodes of each, chosen uniformly at random.

   This is a local process: a node needs to know a few random links, not everyone's degree. Yet
   a node with degree :math:`k_i` is an endpoint of :math:`k_i` of the :math:`2 m` endpoint slots,
   so it is reached with probability :math:`k_i / 2 m`: exactly linear preferential attachment,
   and the same broad degree distribution as :func:`barabasi_albert_graph` (power law with
   exponent about 3). The only difference is that a node never uses the same link twice: once
   linked to a node through a link, it cannot reach the other node of that link through it.

   As in :func:`barabasi_albert_graph`, ``k`` is the average degree (each node brings ``k / 2``
   links, to distinct nodes, so ``k = 2 m`` picks ``m`` links), any ``k`` gives exactly
   ``round(n k / 2)`` edges, the first ``k / 2 + 1`` nodes form a clique, a link or a node already
   used is drawn again, and links are picked among those existing before the node arrives. The
   graph is undirected, without self loops nor parallel edges, connected if ``k`` is at least 2,
   and generated in parallel on the device, with the same result as the sequential process.

   Without ``net``, nodes are the integers ``0`` to ``n - 1`` in order of arrival. With ``net``,
   the result is a randomization of ``net`` (see the module description): its nodes arrive in
   random order, and ``n`` and the number of edges (so ``k``) are those of ``net``, unless ``k``
   is given. In multilayer networks each layer is randomized on its own (``k`` can be one value
   for all layers or a dictionary from layer to average degree); interlayer edges are kept.

   :param n: Number of nodes; ``None`` with ``net``.
   :param k: Average degree; ``None`` with ``net``, for its own.
   :param net: An undirected, not bipartite :class:`~graphoskop.Network` to randomize, or
      ``None``.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The link selection graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If neither ``n`` and ``k`` nor ``net`` are given, if ``net`` is directed
      or bipartite, if ``n`` or ``device`` differ from those of ``net``, if there are more edges
      than node pairs, or for dictionaries as in :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.link_selection_graph(10000, 6, seed = 1)            # each new node picks 3 links
      gsk.degree_power_law(g)                                     # exponent about 3, as preferential attachment

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      rnd = gsk.link_selection_graph(net = net, seed = 1)         # same n and m, nodes arrive at random
   """
   return _growth(n, k, net, seed, device, lambda n, m, gen, device, l: _preferential(n, m, gen, device, links = True))


def copying_graph(n = None, k = None, net = None, seed = None, device = None):
   """
   Copying graph: nodes arrive one at a time, and each picks an earlier node uniformly at random
   and copies one of its connections, chosen uniformly at random: it links to a random neighbor
   of a random node.

   Like :func:`link_selection_graph`, this is a local process: the new node knows nothing about
   the network, and picks a node instead of a link. A node is reached through each of its
   neighbors :math:`u`, with probability :math:`1 / (N k_u)` for :math:`N` earlier nodes, so the
   more neighbors a node has, the more likely it is to be copied. Since the neighbors of hubs
   tend to have few neighbors themselves, hubs are copied even more than by preferential
   attachment: degrees are broader than in :func:`barabasi_albert_graph`, with a power law
   exponent close to 2 instead of 3 (as for pure copying in Kumar et al., 2000).

   As in :func:`barabasi_albert_graph`, ``k`` is the average degree (each node makes ``k / 2``
   copies, of distinct neighbors, so ``k = 2 m`` makes ``m`` copies), any ``k`` gives exactly
   ``round(n k / 2)`` edges, the first ``k / 2 + 1`` nodes form a clique, and a copy is drawn again
   (another random node, another random neighbor) if it gives a node already chosen. Nodes and
   neighborhoods are those before the node arrives. The graph is undirected, without self loops
   nor parallel edges, and connected if ``k`` is at least 2.

   Each copy depends on the whole neighborhood of a node, so unlike the other generators arrivals
   are simulated one after the other, in Python on the CPU (about 12 seconds per million nodes
   with ``k = 10``), and the result is then moved to ``device``.

   Without ``net``, nodes are the integers ``0`` to ``n - 1`` in order of arrival. With ``net``,
   the result is a randomization of ``net`` (see the module description): its nodes arrive in
   random order, and ``n`` and the number of edges (so ``k``) are those of ``net``, unless ``k``
   is given. In multilayer networks each layer is randomized on its own (``k`` can be one value
   for all layers or a dictionary from layer to average degree); interlayer edges are kept.

   :param n: Number of nodes; ``None`` with ``net``.
   :param k: Average degree; ``None`` with ``net``, for its own.
   :param net: An undirected, not bipartite :class:`~graphoskop.Network` to randomize, or
      ``None``.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The copying graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If neither ``n`` and ``k`` nor ``net`` are given, if ``net`` is directed
      or bipartite, if ``n`` or ``device`` differ from those of ``net``, if there are more edges
      than node pairs, or for dictionaries as in :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.copying_graph(10000, 6, seed = 1)                   # each new node makes 3 copies
      gsk.degree_power_law(g)

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      rnd = gsk.copying_graph(net = net, seed = 1)                # same n and m, nodes arrive at random
   """
   return _growth(n, k, net, seed, device, lambda n, m, gen, device, l: _copying(n, m, gen, device))

def holme_kim_graph(n = None, k = None, p = None, net = None, seed = None, device = None):
   """
   Holme-Kim graph (Holme and Kim, 2002): a power law with clustering. Nodes arrive one at a time
   as in :func:`barabasi_albert_graph`, but after linking to a node by preferential attachment,
   each further edge of the new node is, with probability ``p``, a triad formation step: it links
   to a random neighbor of the node just chosen by preferential attachment, closing a triangle.

   Degrees follow a power law with exponent about 3, as in preferential attachment, while ``p``
   tunes the clustering, from that of :func:`barabasi_albert_graph` (``p = 0``) to a high value
   that does not vanish as the graph grows (``p = 1``).

   As in :func:`barabasi_albert_graph`, ``k`` is the average degree (each node brings ``k / 2``
   edges, to distinct nodes, so ``k = 2 m`` is the model with ``m`` edges per node), any ``k``
   gives exactly ``round(n k / 2)`` edges, the first ``k / 2 + 1`` nodes form a clique, a node
   already chosen is drawn again, and degrees and neighbors are those before the node arrives.
   The first edge of a node is always by preferential attachment, and a triad formation step
   whose node has no neighbor left falls back to it. The graph is undirected, without self loops
   nor parallel edges, and connected if ``k`` is at least 2.

   Triad formation needs the neighbors of a node when the new node arrives, so, as for
   :func:`copying_graph`, arrivals are simulated one after the other, in Python on the CPU, and
   the result is then moved to ``device``.

   Without ``net``, nodes are the integers ``0`` to ``n - 1`` in order of arrival. With ``net``,
   the result is a randomization of ``net`` (see the module description): its nodes arrive in
   random order, and ``n`` and the number of edges (so ``k``) are those of ``net``, unless ``k``
   is given. ``p`` too is inferred, unless given: it is the probability whose graphs have the
   average clustering (:func:`~graphoskop.average_clustering`) of ``net``, found by simulation
   (regula falsi on ``p``, on graphs of at most 100,000 nodes with the average degree of the
   result, all with the same random numbers; clustering varies little with size beyond a few
   thousand nodes), which costs a few extra generations. A network more clustered than any
   Holme-Kim graph of its density gets ``p = 1``, with a warning. In multilayer networks each
   layer is randomized on its own, with its own ``p`` (``k`` and ``p`` can be one value for all
   layers or a dictionary from layer to value); interlayer edges are kept.

   :param n: Number of nodes; ``None`` with ``net``.
   :param k: Average degree; ``None`` with ``net``, for its own.
   :param p: Probability of a triad formation step, after the first edge of each node; ``None``
      with ``net``, to match its average clustering.
   :param net: An undirected, not bipartite :class:`~graphoskop.Network` to randomize, or
      ``None``.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The Holme-Kim graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If ``p`` is not between 0 and 1, or missing without ``net``, if neither
      ``n`` and ``k`` nor ``net`` are given, if ``net`` is directed or bipartite, if ``n`` or ``device`` differ from
      those of ``net``, if there are more edges than node pairs, or for dictionaries as in
      :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.holme_kim_graph(10000, 6, 0.8, seed = 1)            # 3 edges per node, mostly closing triangles
      gsk.average_clustering(g), gsk.degree_power_law(g)

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      rnd = gsk.holme_kim_graph(net = net, seed = 1)              # same n, m and average clustering
      rnd = gsk.holme_kim_graph(net = net, p = 0.5, seed = 1)     # same n and m, p given
   """
   if p is None and net is None:
      raise ValueError("give p, or net to infer it from")
   for q in (p.values() if isinstance(p, dict) else [p]):
      if q is not None and not 0 <= q <= 1:
         raise ValueError(f"p must be between 0 and 1, got {q}")
   target = None
   if net is not None:
      from .network_statistics import average_clustering
      c = average_clustering(net)
      target = [c] if net.layer_labels is None else c.set_index("layer")["clustering"].reindex(list(net.layer_labels)).tolist()

   def sample(n, m, gen, device, l):
      q = p if net is None else _layer_value(p, None, net, l, "p", True)
      return _triads(n, m, _calibrate_triads(n, m, target[l], gen) if q is None else q, gen, device)

   return _growth(n, k, net, seed, device, sample)
