"""
Synthetic network generators, in six families:

* simple graphs, with simple fixed topologies: :func:`chain_graph`;
* random graphs, wired at random given a few numbers (edges, density, the degree of every node,
  a small initiator matrix, or positions in space): :func:`gnm_random_graph`,
  :func:`gnp_random_graph`, :func:`configuration_model`, :func:`edge_swap` (only randomizes a
  network), :func:`kronecker_graph`, :func:`geometric_graph`;
* clustered graphs, with many triangles: :func:`caveman_graph`, :func:`watts_strogatz_graph`;
* broad degree graphs, with hubs: :func:`barabasi_albert_graph`, :func:`link_selection_graph`,
  :func:`copying_graph`, :func:`holme_kim_graph` (with clustering);
* stochastic blockmodels, with communities: :func:`planted_partition_graph`,
  :func:`stochastic_block_model`, :func:`degree_corrected_block_model`,
  :func:`mixed_membership_block_model` (overlapping communities), :func:`nested_block_model`
  (hierarchy of groups);
* community benchmarks, with planted communities to test community discovery:
  :func:`girvan_newman_graph`, :func:`lfr_benchmark_graph` (broad degrees and community sizes;
  directed, weighted, overlapping).

Every random generator takes its natural parameters (e.g. the number of nodes and edges) or, with
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

:func:`randomization_test` compares a statistic of a network with its distribution over the
randomizations of any of these generators (or networks drawn from a fitted ERGM).
"""

import itertools
import math
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

from .network import BIPARTITE, _build, _explode, _membership, _memberships, _node_codes, _node_types, _subset, _take_edges


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
      whether the network has self loops; ``gen``, the random generator. ``make`` can also return
      a third value: a dictionary from edge attribute name to the values of the new edges, which
      replace the attributes drawn from the old edges; or a tensor with, for every new edge, the
      position of the old edge of the layer (in input order) whose attributes it takes.
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
   given, start = [], int((~intra).sum())   # attribute values made by the model, and where
   for l in range(L):
      nodes = torch.nonzero(layer == l).squeeze(1)
      nodes = nodes[torch.argsort(kind[nodes], stable = True)]   # type 0 first
      n0 = None if types is None else int((kind[nodes] == 0).sum())
      out = make(l, nodes, n0, count[l], loops, gen)
      a, b = out[0], out[1]
      own = torch.nonzero(intra & (layer[S] == l)).squeeze(1)   # old edges of the layer, in input order
      if len(out) > 2 and torch.is_tensor(out[2]):   # positions of the old edges giving their attributes
         r = own[out[2]]
      else:
         if len(out) > 2:
            given.append((start, out[2]))
         r = _attribute_rows(own, a.shape[0], gen, device)
      start += a.shape[0]
      s.append(nodes[a])
      t.append(nodes[b])
      rows.append(torch.full((a.shape[0],), -1, dtype = torch.long, device = device) if r is None else r)
   s, t, rows = torch.cat(s), torch.cat(t), torch.cat(rows)
   ea = torch.where((rows >= 0)[:, None], X[rows.clamp(min = 0)], torch.nan) if X.shape[0] else X.new_full((rows.shape[0], X.shape[1]), torch.nan)
   for start, values in given:
      for name, v in values.items():
         ea[start:start + v.shape[0], net.edge_attributes.index(name)] = v.to(ea.dtype)
   qe = _take_edges(net.qualitative_edge_attr, rows.cpu().numpy()) if net.qualitative_edge_attributes else None   # -1: new edge, missing values
   return _subset(net, s.cpu().numpy(), t.cpu().numpy(), ea.cpu().numpy().astype(np.float32), np.arange(N), qe)


# ------------------------------------------------------------------------------------------------
# Simple graphs
# ------------------------------------------------------------------------------------------------


def chain_graph(n, directed = None, seed = None, device = None):
   """
   Chain graph (path): ``n`` nodes, the integers ``0`` to ``n - 1``, each linked to the next, so
   that there are ``n - 1`` edges and the two ends have degree 1.

   Directed chains orient each edge ``(i, i + 1)`` as given by ``directed``:

   * ``True`` or ``"sequential"``: from ``i`` to ``i + 1``, all in one direction, from node 0 to
     node ``n - 1``;
   * ``"random"``: in a direction drawn at random, with probability 1/2 each, independently for
     every edge;
   * ``"converging"``: towards the middle node ``n // 2``, which only receives edges;
   * ``"diverging"``: away from the middle node ``n // 2``, which only sends edges.

   :param n: Number of nodes, at least 1.
   :param directed: ``None`` or ``False`` for an undirected chain, or the orientation of the
      edges: ``True``, ``"sequential"``, ``"random"``, ``"converging"`` or ``"diverging"``.
   :param seed: Seed of the random numbers (``"random"`` only), or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU.
   :return: The chain graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If ``n`` is not positive or ``directed`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.chain_graph(10)                                        # 0 - 1 - ... - 9
      g = gsk.chain_graph(10, directed = True)                       # 0 -> 1 -> ... -> 9
      g = gsk.chain_graph(10, directed = "converging")               # 0 -> ... -> 5 <- ... <- 9
      g = gsk.chain_graph(10, directed = "random", seed = 1)
   """
   if directed is True:
      directed = "sequential"
   if directed not in (None, False, "sequential", "random", "converging", "diverging"):
      raise ValueError(f"unknown direction: {directed!r}")
   if int(n) < 1:
      raise ValueError(f"a chain needs at least one node, got {n}")
   n, device = int(n), torch.device("cpu" if device is None else device)
   s = torch.arange(n - 1, device = device)
   t = s + 1
   flip = None   # edges pointing from i + 1 to i
   if directed == "random":
      flip = torch.rand(n - 1, generator = _generator(seed, device), device = device) < 0.5
   elif directed == "converging":
      flip = s >= n // 2
   elif directed == "diverging":
      flip = s < n // 2
   if flip is not None:
      s, t = torch.where(flip, t, s), torch.where(flip, s, t)
   return _build(s.cpu().numpy(), t.cpu().numpy(), np.arange(n), np.zeros((n - 1, 0), dtype = np.float32), [], None, [], bool(directed), device)


# ------------------------------------------------------------------------------------------------
# Random graphs: Gnm, Gnp, configuration model, edge swaps, Kronecker, geometric
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


def _stub_match(dout, din, oriented, rounds, drop_self_loops, drop_parallel, gen, offset = 0, group = None, forbid = None):
   """
   Edges of a configuration model: stubs matched at random, then a few rounds of repairs.

   Undirected (``oriented`` false): the ``dout`` stubs of all nodes are shuffled and paired.
   Oriented (directed, or bipartite with sources of type 0): the ``din`` stubs are shuffled and
   matched with the ``dout`` stubs, targets shifted by ``offset``. Then, in each round, every self
   loop and every repeated edge (all but one copy) trades endpoints with a random edge, in
   parallel: ``(a, b), (c, d)`` become ``(a, c), (b, d)`` or ``(a, d), (b, c)`` (only the latter
   when oriented), which keeps all degrees; a trade is kept only if it creates neither a self
   loop nor an edge already there, and touches no edge of another trade; partners are edges
   whose pair appears once, so that no pair loses its last copy. Trades never make new spurious
   edges, so the edges are sorted once, and each round only looks at the spurious edges left.
   Leftovers are dropped as asked.

   With ``group``, stubs are only matched within groups (e.g. communities), and repairs trade with
   edges of the same group. With ``forbid``, edges for which it is true are spurious too, are not
   made by repairs, and are always dropped if left.

   :param dout: int64 tensor, degree (out-degree, or degree of type 0) of each node.
   :param din: int64 tensor, in-degree (degree of type 1) of each node; ignored if not oriented.
   :param oriented: Whether sources and targets are matched separately.
   :param rounds: Maximum number of repair rounds.
   :param drop_self_loops: Whether to drop the self loops left.
   :param drop_parallel: Whether to drop the repeated edges left (one copy stays).
   :param gen: Random generator.
   :param offset: Added to the targets (bipartite: number of nodes of type 0).
   :param group: int64 tensor, group of each node, nondecreasing with the node id, with stub
      counts that can be matched within every group (even, or equal on the two sides); or
      ``None``.
   :param forbid: Function of two id tensors giving a boolean tensor, or ``None``.
   :return: Source and target ids, int64 tensors.
   :raises ValueError: If the stubs cannot all be matched.
   """
   dev = dout.device
   ids = lambda d: torch.repeat_interleave(torch.arange(d.shape[0], device = dev), d)
   if group is None:
      shuffle = lambda x: x[torch.randperm(x.shape[0], generator = gen, device = dev)]
   else:   # within groups (stubs come sorted by group)
      shuffle = lambda x: x[torch.argsort(group[x].double() + torch.rand(x.shape[0], generator = gen, device = dev, dtype = torch.float64))]
   no = lambda a, b: torch.zeros_like(a, dtype = torch.bool) if forbid is None else forbid(a, b)
   if oriented:
      s, t = ids(dout), shuffle(ids(din)) + offset
      if s.shape[0] != t.shape[0]:
         raise ValueError(f"the degrees on the two sides must have the same sum, got {s.shape[0]:,} and {t.shape[0]:,}")
   else:
      x = shuffle(ids(dout))
      if x.shape[0] % 2:
         raise ValueError("the sum of the degrees must be even")
      s, t = x[0::2], x[1::2]
   m = s.shape[0]
   if m == 0 or (rounds == 0 and not drop_self_loops and not drop_parallel and forbid is None):   # plain matching
      return s, t
   n = int(max(s.max(), t.max())) + 1
   key = lambda a, b: a * n + b if oriented else torch.minimum(a, b) * n + torch.maximum(a, b)
   k = key(s, t)
   sk, cnt = torch.unique(k, return_counts = True)   # sorted once: repairs never make new bad edges
   cnt = cnt[torch.searchsorted(sk, k)]
   first = torch.full((sk.shape[0],), m, device = dev).scatter_reduce(0, torch.searchsorted(sk, k), torch.arange(m, device = dev), "amin")
   loop, banned = s == t, no(s, t)
   bad = loop | banned | torch.ones(m, dtype = torch.bool, device = dev).index_fill_(0, first, False)   # self loops, copies but the first
   clean = (cnt == 1) & ~loop & ~banned   # possible partners: edges whose pair appears once
   if group is not None:   # partners in the same group: edges are sorted by group
      eg = group[s]
      gstart = torch.searchsorted(eg, eg)
      gsize = torch.searchsorted(eg, eg, right = True) - gstart
   sets = [(sk, torch.bincount(torch.searchsorted(sk, k), minlength = sk.shape[0]))]   # sorted keys and copies of each

   def taken(x):   # whether keys are edges now
      hit = torch.zeros_like(x, dtype = torch.bool)
      for keys, alive in sets:
         p = torch.searchsorted(keys, x).clamp(max = keys.shape[0] - 1)
         hit |= (keys[p] == x) & (alive[p] > 0)
      return hit

   def drop(x):   # one copy less of each key
      for keys, alive in sets:
         p = torch.searchsorted(keys, x).clamp(max = keys.shape[0] - 1)
         h = (keys[p] == x) & (alive[p] > 0)
         alive.index_add_(0, p[h], torch.full_like(p[h], -1))
         x = x[~h]

   for _ in range(rounds):
      B = torch.nonzero(bad).squeeze(1)
      if B.shape[0] == 0:
         break
      if group is None:
         P = torch.randint(m, B.shape, generator = gen, device = dev)
      else:
         P = gstart[B] + (torch.rand(B.shape, generator = gen, device = dev, dtype = torch.float64) * gsize[B]).long()
      a, b, c, d = s[B], t[B], s[P], t[P]
      flip = torch.zeros_like(B, dtype = torch.bool) if oriented else torch.rand(B.shape, generator = gen, device = dev) < 0.5
      s1, t1, s2, t2 = a, torch.where(flip, c, d), torch.where(flip, b, c), torch.where(flip, d, b)
      k1, k2 = key(s1, t1), key(s2, t2)
      ok = clean[P] & (s1 != t1) & (s2 != t2) & (k1 != k2) & ~taken(k1) & ~taken(k2) & ~no(s1, t1) & ~no(s2, t2)
      _, inv, used = torch.unique(P, return_inverse = True, return_counts = True)
      ok &= used[inv] == 1   # one trade per partner
      both = torch.cat([k1[ok], k2[ok]])
      _, inv, dup = torch.unique(both, return_inverse = True, return_counts = True)
      h = int(ok.sum())
      ok[ok.clone()] &= (dup[inv[:h]] == 1) & (dup[inv[h:]] == 1)   # no two trades make the same edge
      B, P = B[ok], P[ok]
      if B.shape[0] == 0:
         continue
      drop(torch.cat([k[B], k[P]]))
      s[B], t[B], s[P], t[P] = s1[ok], t1[ok], s2[ok], t2[ok]
      k[B], k[P] = k1[ok], k2[ok]
      bad[B] = False
      clean[B] = True
      sets.append((torch.sort(torch.cat([k1[ok], k2[ok]])).values, torch.ones(2 * B.shape[0], dtype = torch.long, device = dev)))
      if len(sets) > 6:   # merge the keys made by repairs, so that lookups stay few
         keys, alive = torch.cat([x[0] for x in sets[1:]]), torch.cat([x[1] for x in sets[1:]])
         o = torch.argsort(keys[alive > 0])
         sets[1:] = [(keys[alive > 0][o], alive[alive > 0][o])]
   loop = s == t
   keep = ~(bad & ((loop & drop_self_loops) | (~loop & drop_parallel) | no(s, t)))
   return s[keep], t[keep]


def _degrees(net):
   """
   Degrees of the nodes from their intra-layer edges (all edges if not multilayer), self loops
   counting twice in undirected networks.

   :param net: Network.
   :return: Out-degree and in-degree (both the degree if undirected) of each node, int64 tensors in
      the network wide node order, on the device of the network.
   """
   N = int(net.node_offsets[-1])
   codes = _node_codes(net)
   S, T = [], []
   for c in range(len(net.components)):
      ei = net._input_edges(c)[0]
      S.append(ei[0] + int(net.node_offsets[c]))
      T.append(ei[1] + int(net.node_offsets[c]))
   S, T = torch.cat(S), torch.cat(T)
   if codes is not None:
      layer = torch.as_tensor(codes[0], device = S.device)
      intra = layer[S] == layer[T]
      S, T = S[intra], T[intra]
   out, inn = torch.bincount(S, minlength = N), torch.bincount(T, minlength = N)
   return (out, inn) if net.directed else (out + inn, out + inn)


def configuration_model(degrees = None, net = None, directed = None, rounds = 10, drop_self_loops = True, drop_parallel = True, seed = None, device = None):
   """
   Configuration model (Molloy and Reed, 1995): a random graph with a given degree sequence.

   Every node gets as many stubs (half edges) as its degree, and the stubs are matched uniformly
   at random, by one random permutation on the device. Random matching makes some self loops and
   parallel edges, rare unless some degrees are large compared with the square root of the
   number of edges. To avoid them without slowing generation down, a few rounds (``rounds``) of
   repairs follow, all in parallel: each self loop and each repeated edge trades endpoints with a
   random edge, keeping all degrees, when the trade creates neither a self loop nor an edge
   already there. The spurious edges still left (when the degrees leave no room, or after the
   last round) are then dropped, so that the degrees of their nodes are a little lower than asked,
   unless ``drop_self_loops`` or ``drop_parallel`` is ``False``, in which case they are kept and
   degrees are exact. Most degree sequences need one or two rounds; for very heavy tails (degrees
   comparable with the number of nodes) more rounds leave fewer edges to drop, at a cost: a power
   law with exponent 2.2 on a million nodes drops 1.6% of its edges with 10 rounds (5 s on a CPU),
   0.3% with 30 (10 s), 0.05% with 100 (27 s). With ``rounds = 0`` and both kept, the graph is the classic configuration
   model multigraph, every matching of stubs equally likely; repairs make simple graphs slightly
   non-uniform.

   ``degrees`` is a sequence with the degree of each node, nodes ``0`` to ``n - 1``; for directed
   graphs, a pair ``(out_degrees, in_degrees)``, aligned, with the same sum; for bipartite graphs
   (``directed`` false), a pair ``(degrees_0, degrees_1)`` with the degrees of the nodes of type 0
   (``0`` to ``n0 - 1``) and of type 1 (the next ones), with the same sum, whose edges always join
   nodes of different types. Undirected degrees must have an even sum.

   With ``net``, the result is a randomization of ``net`` (see the module description) with the
   degrees of its nodes (out- and in-degrees if directed; self loops count twice if undirected;
   parallel edges count), its node types if bipartite (edges only between types) and, in
   multilayer networks, each layer randomized on its own with the degrees of its intra-layer
   edges; interlayer edges are kept. Directed bipartite networks are not supported.

   :param degrees: Degree sequence, or a pair of them (see above); ``None`` with ``net``.
   :param net: A :class:`~graphoskop.Network` to randomize, or ``None``.
   :param directed: Whether the edges are directed; ``None`` for undirected, or for the direction
      of ``net``.
   :param rounds: Maximum number of repair rounds (0 for none).
   :param drop_self_loops: Whether to drop the self loops left after the repairs.
   :param drop_parallel: Whether to drop the parallel edges left after the repairs (one edge of
      each group stays).
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The random graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If neither ``degrees`` nor ``net`` are given, or both, if the degree sums
      do not allow a matching (odd, or different on the two sides), if ``directed`` or ``device``
      differ from those of ``net``, or if ``net`` is directed and bipartite.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.configuration_model([3, 3, 2, 2, 1, 1], seed = 1)
      d = gsk.configuration_model(([2, 1, 0], [0, 1, 2]), directed = True)   # out- and in-degrees
      b = gsk.configuration_model(([2, 2], [1, 1, 2]))                       # bipartite: 2 + 3 nodes

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      rnd = gsk.configuration_model(net = net, seed = 1)                     # degrees of net, random wiring
      gsk.assortativity(net), gsk.assortativity(rnd)                         # observed vs random, same degrees
      multi = gsk.configuration_model(net = net, rounds = 0, drop_self_loops = False, drop_parallel = False)
   """
   if (degrees is None) == (net is None):
      raise ValueError("give degrees or net, not both")
   if net is None:
      directed, device = bool(directed), torch.device("cpu" if device is None else device)
      gen = _generator(seed, device)
      pair = np.ndim(degrees[0]) == 1 if len(degrees) else False
      if pair:
         d0, d1 = (torch.as_tensor(list(x), dtype = torch.long, device = device) for x in degrees)
      else:
         d0 = torch.as_tensor(list(degrees), dtype = torch.long, device = device)
      bipartite = pair and not directed
      if directed and not pair:
         raise ValueError("directed degrees are a pair (out_degrees, in_degrees)")
      if directed and d0.shape != d1.shape:
         raise ValueError("out- and in-degrees must have one value per node")
      if bipartite:
         s, t = _stub_match(d0, d1, True, rounds, drop_self_loops, drop_parallel, gen, d0.shape[0])
      else:
         s, t = _stub_match(d0, d1 if directed else d0, directed, rounds, drop_self_loops, drop_parallel, gen)
      n0 = d0.shape[0]
      n = n0 + d1.shape[0] if bipartite else n0
      names = np.arange(n)
      types = (names >= n0).astype(np.int8) if bipartite else None
      qtable = (lambda perm: pd.DataFrame({BIPARTITE: pd.Categorical(types[perm])})) if bipartite else None
      return _build(s.cpu().numpy(), t.cpu().numpy(), names, np.zeros((s.shape[0], 0), dtype = np.float32), [], None, [], directed, device, qtable, [BIPARTITE] if bipartite else [], "wide", types, (0, 1) if bipartite else None)
   if net.directed and net.bipartite_labels is not None:
      raise ValueError("directed bipartite networks are not supported")
   _check_net(net, None, directed, device)
   out, inn = _degrees(net)

   def make(l, nodes, n0, count, loops, gen):
      if n0 is not None:   # bipartite: type 0 first in nodes
         return _stub_match(out[nodes[:n0]], out[nodes[n0:]], True, rounds, drop_self_loops, drop_parallel, gen, n0)
      return _stub_match(out[nodes], inn[nodes], net.directed, rounds, drop_self_loops, drop_parallel, gen)

   return _randomize(net, seed, make)


def _swap_edges(s, t, swaps, oriented, gen):
   """
   Double edge swaps: see :func:`edge_swap`.

   Rounds of proposals: the edges are paired at random, and each pair ``(a, b), (c, d)``
   proposes ``(a, c), (b, d)`` or ``(a, d), (b, c)`` (only ``(a, d), (c, b)`` if oriented), valid
   if it makes no self loop and no edge already there. Applying the proposals one by one, in
   order, is a Markov chain whose steps are all symmetric, so the uniform distribution over the
   graphs with these degrees is stationary. Proposals are decided all at once when none of their
   four node pairs appears in another proposal of the round, since then no other proposal can
   change their outcome; the others, rare, one by one in order, so the result is the same as
   that of the sequential chain. The last round stops at the proposal that completes ``swaps``.
   Proposals count whether they succeed or not: stopping after a number of successful swaps
   would favor the graphs that allow more swaps.

   :param s: int64 tensor, sources.
   :param t: int64 tensor, targets.
   :param swaps: Number of swaps tried.
   :param oriented: Whether edges keep their sources as sources (directed, or bipartite with
      sources of type 0).
   :param gen: Random generator.
   :return: New sources and targets, edge by edge (an edge keeps its position).
   """
   dev, m = s.device, s.shape[0]
   s, t = s.clone(), t.clone()
   if m < 2 or swaps <= 0:
      return s, t
   n = int(max(s.max(), t.max())) + 1
   key = lambda a, b: a * n + b if oriented else torch.minimum(a, b) * n + torch.maximum(a, b)
   done, h = 0, m // 2
   while done < swaps:
      k = key(s, t)
      cur, where = torch.sort(k)   # current edges
      dup = torch.zeros(m, dtype = torch.bool, device = dev)   # edges with a parallel copy
      same = cur[1:] == cur[:-1]
      dup[where[1:][same]] = True
      dup[where[:-1][same]] = True
      perm = torch.randperm(m, generator = gen, device = dev)
      A, B = perm[0:2 * h:2], perm[1:2 * h:2]
      a, b, c, d = s[A], t[A], s[B], t[B]
      flip = torch.zeros(h, dtype = torch.bool, device = dev) if oriented else torch.rand(h, generator = gen, device = dev) < 0.5
      s1, t1 = a, torch.where(flip, c, d)
      s2, t2 = torch.where(flip, b, c), torch.where(flip, d, b)
      new = torch.cat([key(s1, t1), key(s2, t2)])
      shape = (s1 != t1) & (s2 != t2) & (new[:h] != new[h:])
      new = torch.where(torch.cat([shape, shape]), new, -1 - torch.arange(2 * h, device = dev))   # invalid: distinct, no edge
      ns, order = torch.sort(new)   # sorted queries: faster lookups, and repeated pairs side by side
      pos = torch.empty_like(order)
      pos[order] = torch.searchsorted(cur, ns).clamp(max = m - 1)
      exists = cur[pos] == new
      ok = shape & ~exists[:h] & ~exists[h:]
      # proposals another one can affect: new pairs that are edges (which an earlier swap may
      # remove), the swaps moving those edges, parallel edges, and new pairs made twice
      moving = torch.zeros(m, dtype = torch.bool, device = dev)
      moving[where[pos[exists]]] = True
      moving |= dup
      twice = torch.zeros(2 * h, dtype = torch.bool, device = dev)
      same = ns[1:] == ns[:-1]
      twice[order[1:][same]] = True
      twice[order[:-1][same]] = True
      twice = twice.view(2, h).any(0)
      tangled = torch.nonzero((shape & (exists[:h] | exists[h:] | twice)) | moving[A] | moving[B]).squeeze(1)
      if tangled.shape[0]:   # one by one, in order, against the edges as the earlier ones left them
         keys = torch.stack([k[A[tangled]], k[B[tangled]], new[tangled], new[h + tangled]], 1)
         flat = keys.flatten()
         count = dict(zip(flat.tolist(), (torch.searchsorted(cur, flat, right = True) - torch.searchsorted(cur, flat)).tolist()))
         for j, (k1, k2, k3, k4), v in zip(tangled.tolist(), keys.tolist(), shape[tangled].tolist()):
            take = v and count[k3] == 0 and count[k4] == 0
            if take:
               count[k1] -= 1
               count[k2] -= 1
               count[k3], count[k4] = 1, 1
            ok[j] = take
      idx = torch.nonzero(ok[:swaps - done]).squeeze(1)   # stop at the attempt completing the swaps
      s[A[idx]], t[A[idx]], s[B[idx]], t[B[idx]] = s1[idx], t1[idx], s2[idx], t2[idx]
      done += min(h, swaps - done)
   return s, t


def edge_swap(net, swaps = None, seed = None):
   """
   Edge swap randomization (Maslov and Sneppen, 2002): a randomization of ``net`` that keeps the
   degree of every node exactly (out- and in-degrees if directed, and node types if bipartite),
   by double edge swaps: two edges ``a - b`` and ``c - d`` become ``a - c`` and ``b - d`` (or
   ``a - d`` and ``b - c``; ``a -> d`` and ``c -> b`` if directed), when this makes no self loop
   and no edge already there.

   With many swaps the result is a uniformly random simple graph with the degrees of ``net``,
   the null model of :func:`configuration_model` without its approximations (no dropped
   edges, no bias from repairs; for directed graphs swaps alone cannot reach every graph, e.g.
   the reversal of a directed triangle, so the result is uniform among the graphs they reach).
   With few swaps, it is a partial randomization: at most ``2 swaps`` edges change (fewer, as
   some swaps are not valid), the others stay, so the number of swaps controls how much of the structure of ``net`` is destroyed (e.g.
   to follow how clustering or assortativity decay towards the null model). Every edge keeps its
   attributes (weight, sign) as it moves. Self loops and parallel edges of ``net`` are kept or
   swapped away, never created.

   Swaps run in rounds, all at once on the device: every round pairs the edges at random and
   applies all the valid swaps that touch no node pair of another swap of the round, and the
   rare others one by one, so that the result is the same as that of the sequential Markov chain.
   ``swaps`` counts the swaps tried, valid or not: the uniform distribution needs a fixed number
   of steps of the chain (stopping after a number of successful swaps, as some libraries do,
   favors the graphs that allow more swaps). In multilayer networks each layer is
   swapped on its own (``swaps`` can be a dictionary from layer to number); interlayer edges are
   kept.

   :param net: A :class:`~graphoskop.Network`.
   :param swaps: Number of swaps tried; ``None`` for 10 per edge, enough to randomize most
      networks (Milo et al., 2003). In multilayer networks, a number per layer or a dictionary.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :return: The randomized network, a new :class:`~graphoskop.Network`.
   :raises ValueError: For dictionaries as in :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      rnd = gsk.edge_swap(net, seed = 1)                      # degrees kept, everything else random
      part = gsk.edge_swap(net, swaps = net.number_of_edges() // 4)   # at most half of the edges moved
      gsk.average_clustering(net), gsk.average_clustering(part), gsk.average_clustering(rnd)
   """
   S, T = [], []
   for c in range(len(net.components)):
      ei = net._input_edges(c)[0]
      S.append(ei[0] + int(net.node_offsets[c]))
      T.append(ei[1] + int(net.node_offsets[c]))
   S, T = torch.cat(S), torch.cat(T)
   oriented = net.directed or net.bipartite_labels is not None

   def make(l, nodes, n0, count, loops, gen):
      pos = torch.full((int(net.node_offsets[-1]),), -1, dtype = torch.long, device = nodes.device)
      pos[nodes] = torch.arange(nodes.shape[0], device = nodes.device)
      mine = torch.nonzero(torch.isin(S, nodes) & torch.isin(T, nodes)).squeeze(1)   # this layer's edges, in input order
      a, b = pos[S[mine]], pos[T[mine]]
      if n0 is not None and not net.directed:   # bipartite: sources of type 0 (first in nodes)
         a, b = torch.minimum(a, b), torch.maximum(a, b)
      q = _layer_value(swaps, 10 * count, net, l, "swaps", True)
      a, b = _swap_edges(a, b, int(q), oriented, gen)
      return a, b, torch.arange(a.shape[0], device = a.device)

   return _randomize(net, seed, make)

def _partitions(items):
   """
   All set partitions of a list.

   :param items: List.
   :return: Generator of partitions, as lists of blocks (lists).
   """
   if not items:
      yield []
      return
   first, rest = items[0], items[1:]
   for p in _partitions(rest):
      for i in range(len(p)):
         yield p[:i] + [[first] + p[i]] + p[i + 1:]
      yield [[first]] + p


# patterns whose expected counts fit an initiator: positions, edges, automorphisms
_KRONECKER_PATTERNS = {
   False: [(2, [(0, 1)], 2), (3, [(0, 1), (0, 2)], 2), (4, [(0, 1), (0, 2), (0, 3)], 6), (3, [(0, 1), (1, 2), (2, 0)], 6)],
   True: [(2, [(0, 1)], 1), (3, [(0, 1), (0, 2)], 2), (3, [(1, 0), (2, 0)], 2), (3, [(0, 1), (1, 2)], 1), (2, [(0, 1), (1, 0)], 2)],
}


def _kronecker_moments(theta, k, n, directed):
   """
   Expected counts of small subgraphs in a stochastic Kronecker graph (Gleich and Owen, 2012):
   undirected edges, wedges, 3-stars and triangles; directed edges, out- and in-wedges, 2-paths
   and mutual pairs. A sum over all tuples of nodes factorizes over the ``k`` digits of their
   ids, so the sum over tuples of distinct nodes follows by Möbius inversion over the set
   partitions of the positions.

   :param theta: Initiator, ``N1`` by ``N1`` :class:`numpy.ndarray`.
   :param k: Kronecker power.
   :param n: Number of nodes kept, at random, among the ``N1 ** k``.
   :param directed: Whether the edges are directed.
   :return: :class:`numpy.ndarray` of expected counts.
   """
   N1 = theta.shape[0]
   N = N1 ** k
   out = []
   for m, edges, aut in _KRONECKER_PATTERNS[directed]:
      total = 0.
      for p in _partitions(list(range(m))):
         block = {x: b for b, B in enumerate(p) for x in B}
         s = 0.
         for idx in itertools.product(range(N1), repeat = len(p)):   # one digit per block
            s += math.prod(theta[idx[block[a]], idx[block[b]]] for a, b in edges)
         total += math.prod((-1) ** (len(B) - 1) * math.factorial(len(B) - 1) for B in p) * s ** k
      kept = math.prod((n - i) / (N - i) for i in range(m))   # nodes kept at random
      out.append(max(total, 0.) * kept / aut)
   return np.array(out)


def _graph_moments(s, t, n, directed):
   """
   Counts of the subgraphs of :func:`_kronecker_moments` in a graph (self loops and parallel
   edges left out).

   :param s: int64 tensor, sources.
   :param t: int64 tensor, targets.
   :param n: Number of nodes.
   :param directed: Whether the edges are directed.
   :return: :class:`numpy.ndarray` of counts.
   """
   from .network_statistics import _paths, _skeleton
   keep = s != t
   s, t = s[keep], t[keep]
   if directed:
      key = torch.unique(s * n + t)
      s, t = key // n, key % n
      dout, din = torch.bincount(s, minlength = n).double(), torch.bincount(t, minlength = n).double()
      mutual = int(torch.isin(t * n + s, key).sum()) // 2
      return np.array([s.shape[0], float((dout * (dout - 1) / 2).sum()), float((din * (din - 1) / 2).sum()), float((din * dout).sum()) - 2 * mutual, mutual])
   src, dst = _skeleton(s, t, n)[:2]
   d = torch.bincount(torch.cat([src, dst]), minlength = n).double()
   triangles = sum(x[0].shape[0] for x in _paths(src, dst, n, 2 ** 22))
   return np.array([src.shape[0], float((d * (d - 1) / 2).sum()), float((d * (d - 1) * (d - 2) / 6).sum()), triangles])


def _fit_initiator(s, t, n, directed, gen):
   """
   2 by 2 initiator of a stochastic Kronecker graph whose expected counts of small subgraphs
   (:func:`_kronecker_moments`) are closest to those of a graph, in relative terms (squared
   differences of logarithms), by Nelder-Mead from several starting points (Gleich and Owen, 2012).

   :param s: int64 tensor, sources.
   :param t: int64 tensor, targets.
   :param n: Number of nodes.
   :param directed: Whether the edges are directed.
   :param gen: Random generator (for the starting points).
   :return: Initiator, 2 by 2 :class:`numpy.ndarray` (symmetric if undirected), with the larger
      diagonal value first (swapping the two digits gives the same model).
   """
   from scipy.optimize import minimize
   k = max(1, math.ceil(math.log2(max(n, 2))))
   obs = np.log1p(_graph_moments(s, t, n, directed))
   make = (lambda x: x.reshape(2, 2)) if directed else (lambda x: np.array([[x[0], x[1]], [x[1], x[2]]]))
   loss = lambda z: float(((np.log1p(_kronecker_moments(make(1 / (1 + np.exp(-z))), k, n, directed)) - obs) ** 2).sum())
   starts = torch.rand(10, 4 if directed else 3, generator = gen, dtype = torch.float64, device = s.device).cpu().numpy()
   best = min((minimize(loss, np.log(x / (1 - x)), method = "Nelder-Mead", options = {"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000}) for x in 0.05 + 0.9 * starts), key = lambda r: r.fun)
   theta = make(1 / (1 + np.exp(-best.x)))
   return theta if theta[0, 0] >= theta[1, 1] else theta[::-1, ::-1].copy()   # same model with the digits swapped


def _kronecker(theta, k, n, directed, self_loops, gen, device):
   """
   Edges of a Kronecker graph: see :func:`kronecker_graph`.

   A 0/1 initiator gives a deterministic graph, whose edges are all the sequences of ``k``
   nonzero cells. Otherwise pairs are drawn exactly with their probabilities by ball dropping
   with thinning: a Poisson number of balls, with mean ``c`` times the expected number of edges,
   each landing on a pair by choosing one cell per digit with probability proportional to its
   value, so that every pair gets a Poisson number of balls with mean :math:`c\\, p_{uv}`
   (half of that in each order if undirected); a pair with balls is kept with probability
   :math:`p_{uv} / (1 - e^{-c p_{uv}})`, with :math:`c = -\\ln(1 - p_{max}) / p_{max}` making
   that at most 1. Then ``n`` of the ``N1 ** k`` nodes are kept at random.

   :param theta: float64 tensor, ``N1`` by ``N1`` initiator.
   :param k: Kronecker power.
   :param n: Number of nodes kept.
   :return: Source and target ids (0 to ``n - 1``), int64 tensors.
   """
   N1 = theta.shape[0]
   N = N1 ** k
   cells = theta.flatten()
   if bool(((cells == 0) | (cells == 1)).all()):   # deterministic: every sequence of nonzero cells
      nz = torch.nonzero(cells).squeeze(1)
      u = torch.zeros(1, dtype = torch.long, device = device)
      v = u.clone()
      for _ in range(k):
         u = (u[:, None] * N1 + nz // N1).flatten()
         v = (v[:, None] * N1 + nz % N1).flatten()
      if not directed:
         u, v = u[u <= v], v[u <= v]
   else:
      pmax = min(float(cells.max()) ** k, 1 - 1e-9)
      c = -math.log1p(-pmax) / pmax
      f = 1 if directed or self_loops else 0.5
      balls = int(torch.poisson(torch.tensor([f * c * float(cells.sum()) ** k], dtype = torch.float64, device = device), generator = gen).item())
      cum = torch.cumsum(cells / cells.sum(), 0)
      u = torch.zeros(balls, dtype = torch.long, device = device)
      v = u.clone()
      for _ in range(k):   # one cell per digit
         cell = torch.searchsorted(cum, torch.rand(balls, generator = gen, device = device, dtype = torch.float64), right = True).clamp(max = cells.shape[0] - 1)
         u, v = u * N1 + cell // N1, v * N1 + cell % N1
      if not directed:
         u, v = torch.minimum(u, v), torch.maximum(u, v)
      key = torch.unique(u * N + v)
      u, v = key // N, key % N
      p = torch.ones(u.shape[0], dtype = torch.float64, device = device)
      a, b = u.clone(), v.clone()
      for _ in range(k):
         p *= theta[a % N1, b % N1]
         a, b = a // N1, b // N1
      rate = f * c * p * torch.where((u != v) & (not directed), 2., 1.)
      keep = torch.rand(p.shape[0], generator = gen, device = device, dtype = torch.float64) * -torch.expm1(-rate) < p
      u, v = u[keep], v[keep]
   if not self_loops:
      u, v = u[u != v], v[u != v]
   if n < N:   # n nodes kept at random, renumbered in order
      kept = torch.randperm(N, generator = gen, device = device)[:n].sort().values
      pos = lambda x: torch.searchsorted(kept, x).clamp(max = n - 1)
      ok = (kept[pos(u)] == u) & (kept[pos(v)] == v)
      u, v = pos(u[ok]), pos(v[ok])
   return u, v


def kronecker_graph(initiator = None, n = None, k = None, net = None, directed = None, self_loops = False, seed = None, device = None):
   """
   Kronecker graph (Leskovec, Chakrabarti, Kleinberg, Faloutsos and Ghahramani, 2010): the
   adjacency matrix is a Kronecker power of a small initiator matrix :math:`\\Theta`, of
   :math:`N_1` nodes. Repeating the product makes graphs that are self-similar at all scales,
   with heavy-tailed degrees, small diameter and community-within-community structure, from a
   handful of parameters.

   With a 0/1 initiator, the graph is the deterministic Kronecker graph. With probabilities, it
   is a stochastic Kronecker graph: writing nodes ``u`` and ``v`` in base :math:`N_1` with
   ``k`` digits, the edge ``(u, v)`` is there independently with probability
   :math:`p_{uv} = \\prod_l \\Theta_{u_l v_l}`. Pairs are drawn exactly with these probabilities by
   ball dropping (one initiator cell per digit) with a thinning correction, in time about linear in
   the number of edges, not of pairs; initiator values equal to 1 mixed with others make
   generation slower (up to 20 candidate pairs per edge). The Kronecker power is ``k``, or the
   smallest one with at least ``n`` nodes, of which ``n`` are then kept at random. Self loops
   (pairs ``u = v``, which have a probability too) are dropped unless ``self_loops``; undirected
   graphs need a symmetric initiator.

   Without ``net``, nodes are the integers ``0`` to ``n - 1`` (in the order of their Kronecker
   ids). With ``net``, the result is a randomization of ``net`` (see the module description) with
   ``n`` its number of nodes and, unless given, a 2 by 2 initiator fitted to it by the method of
   moments of Gleich and Owen (2012): the one whose expected numbers of edges, wedges, 3-stars and
   triangles (if undirected; edges, out- and in-wedges, 2-paths and mutual pairs if directed) are
   closest to those of ``net``, in relative terms (with three or four parameters for four or five
   counts, each is matched within a few percent, e.g. edges 5% fewer on average on a 3,000 node
   Kronecker graph). This is much faster than the maximum
   likelihood fit (KronFit) of Leskovec et al., but its nodes are not matched to those of ``net``:
   the nodes of ``net`` take random positions. In multilayer networks each layer is randomized on
   its own, with its own initiator (``initiator`` can be a dictionary from layer to matrix);
   interlayer edges are kept.

   :param initiator: ``N1`` by ``N1`` matrix (nested lists, array or tensor) of probabilities, or
      of 0 and 1; ``None`` with ``net``, for one fitted to it.
   :param n: Number of nodes; ``None`` with ``net`` or ``k``.
   :param k: Kronecker power, for ``N1 ** k`` nodes; ``None`` with ``n`` or ``net``.
   :param net: A :class:`~graphoskop.Network` to randomize, not bipartite, or ``None``.
   :param directed: Whether the edges are directed; ``None`` for undirected, or for the direction
      of ``net``.
   :param self_loops: Whether to keep self loops.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The Kronecker graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If neither ``initiator`` with ``n`` or ``k`` nor ``net`` are given, if
      ``n`` or ``k`` are given with ``net``, if the initiator is not a square matrix of
      probabilities, or not symmetric for an undirected graph, if ``n`` is not positive, if
      ``net`` is bipartite, if ``directed`` or ``device`` differ from those of ``net``, or for
      dictionaries as in :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.kronecker_graph([[0.9, 0.5], [0.5, 0.1]], n = 10000, seed = 1)   # stochastic, 2^14 nodes thinned to 10,000
      g = gsk.kronecker_graph([[1, 1, 0], [1, 1, 1], [0, 1, 1]], k = 6)       # deterministic, 729 nodes
      g = gsk.kronecker_graph([[0.9, 0.6], [0.3, 0.2]], k = 12, directed = True)

      net = gsk.load_network("edges.tsv")
      rnd = gsk.kronecker_graph(net = net, seed = 1)                          # initiator fitted to net
   """

   def matrix(x):
      T = torch.as_tensor(np.asarray(x, dtype = np.float64))
      if T.ndim != 2 or T.shape[0] != T.shape[1] or T.shape[0] < 1 or not bool(((T >= 0) & (T <= 1)).all()):
         raise ValueError(f"the initiator must be a square matrix of probabilities, got shape {tuple(T.shape)}")
      if not directed and not torch.equal(T, T.T):
         raise ValueError("the initiator must be symmetric for an undirected graph")
      return T

   if net is None:
      if initiator is None or (n is None) == (k is None):
         raise ValueError("give the initiator with n or k, or net")
      directed, device = bool(directed), torch.device("cpu" if device is None else device)
      T = matrix(initiator).to(device)
      if k is None:
         if n < 1:
            raise ValueError(f"n must be positive, got {n}")
         k = max(1, math.ceil(math.log(n) / math.log(T.shape[0]) - 1e-12)) if T.shape[0] > 1 else 1
         n = int(n) if T.shape[0] > 1 else 1
      else:
         n = T.shape[0] ** int(k)
      s, t = _kronecker(T, int(k), n, directed, self_loops, _generator(seed, device), device)
      return _build(s.cpu().numpy(), t.cpu().numpy(), np.arange(n), np.zeros((s.shape[0], 0), dtype = np.float32), [], None, [], directed, device)
   if n is not None or k is not None:
      raise ValueError("with net, the number of nodes is that of net: leave out n and k")
   if net.bipartite_labels is not None:
      raise ValueError("bipartite networks are not supported")
   _check_net(net, None, directed, device)
   directed = net.directed
   S, T = [], []
   for c in range(len(net.components)):
      ei = net._input_edges(c)[0]
      S.append(ei[0] + int(net.node_offsets[c]))
      T.append(ei[1] + int(net.node_offsets[c]))
   S, T = torch.cat(S), torch.cat(T)

   def make(l, nodes, n0, count, loops, gen):
      N = nodes.shape[0]
      given = _layer_value(initiator, None, net, l, "initiator", True)
      if given is None:
         pos = torch.full((int(net.node_offsets[-1]),), -1, dtype = torch.long, device = nodes.device)
         pos[nodes] = torch.arange(N, device = nodes.device)
         mine = torch.isin(S, nodes) & torch.isin(T, nodes)   # intra-layer edges of this layer
         given = _fit_initiator(pos[S[mine]], pos[T[mine]], N, directed, gen)
      Th = matrix(given).to(nodes.device)
      kk = max(1, math.ceil(math.log(max(N, 2)) / math.log(Th.shape[0]) - 1e-12)) if Th.shape[0] > 1 else 1
      s, t = _kronecker(Th, kk, N, directed, self_loops, gen, nodes.device)
      perm = torch.randperm(N, generator = gen, device = nodes.device)   # nodes of net at random positions
      return perm[s], perm[t]

   return _randomize(net, seed, make)


def _close_pairs(pos, r, p, torus, chunk = 1 << 24):
   """
   Pairs of points within distance ``r``, by a grid of cells of side at least ``r``: points are
   sorted by cell, and each is compared only with the points of its cell and of the adjacent
   ones, all at once, in chunks.

   :param pos: float64 tensor, ``n`` by ``d`` positions (in the unit cube if ``torus``).
   :param r: Radius.
   :param p: Minkowski exponent of the distance (``math.inf`` for the largest coordinate
      difference).
   :param torus: Whether the unit cube wraps around (periodic boundaries).
   :param chunk: Pairs compared at once.
   :return: Endpoints ``i < j`` and distances of the pairs.
   """
   n, d = pos.shape
   dev = pos.device
   lo = torch.zeros(d, dtype = torch.float64, device = dev) if torus else pos.min(0).values
   extent = torch.ones(d, dtype = torch.float64, device = dev) if torus else (pos.max(0).values - lo).clamp(min = 1e-300)
   g = (extent / max(r, 1e-300)).floor().long().clamp(min = 1, max = max(1, int(n ** (1 / d)) + 1))   # cells per dimension
   if torus:
      g = torch.where(g < 3, 1, g)   # with fewer than 3 cells, neighbors would repeat
   if 3 ** d > 729:   # many dimensions: all pairs
      g = torch.ones_like(g)
   cell = torch.minimum(((pos - lo) / extent * g).floor().long(), g - 1)
   weight = torch.cumprod(torch.cat([g.new_ones(1), g[:-1]]), 0)
   cid = (cell * weight).sum(1)
   order = torch.argsort(cid)
   sid = cid[order]
   offsets = torch.cartesian_prod(*[torch.arange(-1, 2, device = dev) if int(x) > 1 else torch.zeros(1, dtype = torch.long, device = dev) for x in g]).reshape(-1, d)

   def dist(a, b):
      x = pos[b] - pos[a]
      if torus:
         x = x - torch.round(x)
      x = x.abs()
      return x.max(1).values if math.isinf(p) else (x ** p).sum(1) ** (1 / p)

   I, J, D = [], [], []
   for off in offsets:
      c = cell + off
      inside = ((c >= 0) & (c < g)).all(1)
      c = c % g if torus else c
      nb = (c * weight).sum(1)
      start = torch.searchsorted(sid, nb)
      count = torch.where(inside | torus, torch.searchsorted(sid, nb, right = True) - start, 0)
      end = torch.cumsum(count, 0)
      lo_i = 0
      while lo_i < n:   # points in chunks of about `chunk` comparisons
         base = int(end[lo_i] - count[lo_i])
         hi_i = max(int(torch.searchsorted(end, base + chunk, right = True)), lo_i + 1)
         a = torch.repeat_interleave(torch.arange(lo_i, hi_i, device = dev), count[lo_i:hi_i])
         b = order[_expand(start[lo_i:hi_i], count[lo_i:hi_i])]
         keep = a < b
         a, b = a[keep], b[keep]
         x = dist(a, b)
         keep = x <= r
         I.append(a[keep])
         J.append(b[keep])
         D.append(x[keep])
         lo_i = hi_i
   return torch.cat(I), torch.cat(J), torch.cat(D)


def _geometric(pos, r, m, p, torus):
   """
   Edges of a random geometric graph: see :func:`geometric_graph`.

   :param pos: float64 tensor, ``n`` by ``d`` positions.
   :param r: Radius, or ``None`` for the ``m`` closest pairs.
   :param m: Number of edges, with ``r`` ``None``.
   :param p: Minkowski exponent.
   :param torus: Whether the unit cube wraps around.
   :return: Source and target ids, int64 tensors.
   """
   n, d = pos.shape
   if r is not None:
      return _close_pairs(pos, r, p, torus)[:2]
   m = min(int(m), n * (n - 1) // 2)
   if m == 0:
      e = torch.zeros(0, dtype = torch.long, device = pos.device)
      return e, e
   volume = (2 * math.gamma(1 / p + 1)) ** d / math.gamma(d / p + 1) if not math.isinf(p) else 2. ** d   # unit ball
   space = 1. if torus else float((pos.max(0).values - pos.min(0).values).prod())
   r = (1.2 * m * space / (n * (n - 1) / 2 * volume)) ** (1 / d)   # expected m pairs, and some more
   while True:
      I, J, D = _close_pairs(pos, r, p, torus)
      if I.shape[0] >= m:
         break
      r *= 1.1 * (m / max(I.shape[0], 1)) ** (1 / d)   # boundaries left fewer pairs: grow the radius
   keep = torch.argsort(D, stable = True)[:m]   # the m closest pairs
   return I[keep], J[keep]


def geometric_graph(n = None, r = None, m = None, d = 2, positions = None, net = None, p = 2, torus = False, position = "position", seed = None, device = None):
   """
   Random geometric graph (Gilbert, 1961; Penrose, 2003): ``n`` nodes at random positions,
   uniformly in the unit cube of ``d`` dimensions, joined when they are close, within distance
   ``r``; or, with ``m`` instead, joined by the ``m`` closest pairs, as if the radius grew until
   ``m`` pairs were within it. Geometric graphs model networks embedded in space (wireless and
   sensor networks, contacts between nearby individuals, road junctions): they have high
   clustering and long paths, and are the classic example of networks whose structure comes from
   a hidden metric space.

   Close pairs are found on a grid of cells with side at least ``r``: points are sorted by cell
   and each is compared with the points of its cell and of the adjacent ones only, all in parallel
   on the device, in time about linear in the number of nodes and edges. For ``m`` edges, the
   radius is first estimated from the expected density of pairs, grown if the boundaries left
   fewer than ``m`` pairs within it, and the ``m`` closest pairs are kept (ties broken
   arbitrarily). In more than 6 dimensions all pairs are compared. Distances are Minkowski
   distances of exponent ``p`` (2 for Euclidean, 1 for Manhattan, ``math.inf`` for the largest
   coordinate difference); with ``torus = True`` the unit cube wraps around, so that nodes near
   opposite sides are close (no boundary effects).

   Without ``net``, nodes are the integers ``0`` to ``n - 1``, at random positions or at
   ``positions`` (an ``n`` by ``d`` array, in the unit cube if ``torus``), which are stored as the
   numeric node attributes ``position_0``, ``position_1``, ... (``position`` is the prefix;
   ``None`` for no attributes). With ``net``, the result is a randomization of ``net`` (see the
   module description) with its number of nodes and, unless ``r`` or ``m`` is given, its number
   of edges (pairs of nodes, self loops left out, at most all pairs): if ``net`` has the node
   attributes ``position_0``, ``position_1``, ..., they are the positions (e.g. coordinates of
   real places, whose graph of closest pairs is then built), otherwise nodes get random
   positions, which are stored. In multilayer networks each layer is randomized on its own, with
   the positions of its copies; ``r`` and ``m`` can be dictionaries from layer to value.
   Interlayer edges are kept.

   :param n: Number of nodes; ``None`` with ``positions`` or ``net``.
   :param r: Radius: pairs at distance at most ``r`` are edges.
   :param m: Number of edges, the closest pairs, instead of ``r``.
   :param d: Number of dimensions of random positions.
   :param positions: ``n`` by ``d`` positions (array, nested lists or tensor), or ``None`` for
      random ones.
   :param net: A :class:`~graphoskop.Network` to randomize, undirected and not bipartite, or
      ``None``.
   :param p: Minkowski exponent of the distance.
   :param torus: Whether the unit cube wraps around (periodic boundaries).
   :param position: Prefix of the names of the node attributes with the coordinates, or ``None``.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The geometric graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If neither ``r`` nor ``m`` is given without ``net``, or both are given, if
      neither ``n`` nor ``positions`` is given without ``net``, or ``n`` differs from the number of
      positions, if ``r``, ``m``, ``d`` or ``p`` are not positive, if positions are outside the
      unit cube with ``torus``, if ``net`` is directed or bipartite, if ``device`` differs from that
      of ``net``, or for dictionaries as in :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.geometric_graph(1000, r = 0.05, seed = 1)                # pairs closer than 0.05 in the unit square
      g = gsk.geometric_graph(1000, m = 5000, d = 3, torus = True)     # the 5000 closest pairs in a 3D torus
      g.get_node_attribute("position_0")                               # first coordinate of every node

      cities = [[12.57, 55.68], [10.20, 56.16], [10.40, 55.40]]        # longitude, latitude
      g = gsk.geometric_graph(positions = cities, m = 2)               # the two closest pairs of cities

      net = gsk.load_network("edges.tsv")
      rnd = gsk.geometric_graph(net = net, seed = 1)                   # same n and m, random positions
   """
   if r is not None and m is not None:
      raise ValueError("give r or m, not both")
   if not p > 0:
      raise ValueError(f"p must be positive, got {p}")
   for x in [y for v in (r, m) for y in (v.values() if isinstance(v, dict) else [v])]:
      if x is not None and not x > 0:
         raise ValueError(f"r and m must be positive, got {x}")

   def place(count, dims, given, gen, dev):
      if given is None:
         if not int(dims) >= 1:
            raise ValueError(f"d must be a positive integer, got {dims}")
         return torch.rand(count, int(dims), generator = gen, device = dev, dtype = torch.float64)
      x = torch.as_tensor(np.asarray(given, dtype = np.float64), device = dev)
      if x.ndim == 1:
         x = x[:, None]
      if torus and not bool(((x >= 0) & (x <= 1)).all()):
         raise ValueError("with torus, positions must be in the unit cube")
      return x

   if net is None:
      if r is None and m is None:
         raise ValueError("give r or m, or net")
      if n is None and positions is None:
         raise ValueError("give n or positions, or net")
      device = torch.device("cpu" if device is None else device)
      pos = place(n, d, positions, _generator(seed, device), device)
      if n is not None and pos.shape[0] != n:
         raise ValueError(f"n is {n}, but there are {pos.shape[0]} positions")
      s, t = _geometric(pos, r, m, p, torus)
      x = pos.cpu().numpy().astype(np.float32)
      names = [] if position is None else [f"{position}_{i}" for i in range(x.shape[1])]
      return _build(s.cpu().numpy(), t.cpu().numpy(), np.arange(x.shape[0]), np.zeros((s.shape[0], 0), dtype = np.float32), [], (lambda perm: x[perm]) if names else None, names, False, device)
   if n is not None or positions is not None:
      raise ValueError("with net, nodes and positions come from net: leave out n and positions")
   if net.directed or net.bipartite_labels is not None:
      raise ValueError("geometric graphs of directed or bipartite networks are not supported")
   _check_net(net, None, None, device)
   cols = []
   while position is not None and f"{position}_{len(cols)}" in net.node_attributes:
      cols.append(f"{position}_{len(cols)}")
   P = torch.cat([torch.stack([net.get_node_attribute(c, j).double() for c in cols], 1) if cols else torch.zeros(net.number_of_nodes(j), 0, dtype = torch.float64, device = net.components[0].edge_index.device) for j in range(len(net.components))]) if cols else None
   S, T = [], []
   for c in range(len(net.components)):
      ei = net._input_edges(c)[0]
      S.append(ei[0] + int(net.node_offsets[c]))
      T.append(ei[1] + int(net.node_offsets[c]))
   S, T = torch.cat(S), torch.cat(T)
   placed = []

   def make(l, nodes, n0, count, loops, gen):
      N = nodes.shape[0]
      x = P[nodes] if P is not None else place(N, d, None, gen, nodes.device)
      if P is None:
         placed.append((nodes, x.cpu().numpy()))
      rr, mm = _layer_value(r, None, net, l, "r", True), _layer_value(m, None, net, l, "m", True)
      if rr is None and mm is None:
         mine = torch.isin(S, nodes) & torch.isin(T, nodes) & (S != T)   # pairs of this layer
         mm = torch.unique(torch.minimum(S[mine], T[mine]) * int(net.node_offsets[-1]) + torch.maximum(S[mine], T[mine])).shape[0]
      return _geometric(x, rr, mm, p, torus)

   new = _randomize(net, seed, make)
   for i in range(d if P is None and position is not None and placed else 0):
      new = _with_groups(net, new, f"{position}_{i}", [(nodes, x[:, i]) for nodes, x in placed])
   return new


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

   return _with_groups(net, _randomize(net, seed, make), attribute, caves)


def _with_groups(net, new, attribute, groups):
   """
   Randomization of ``net`` with the groups (caves, communities) its nodes were assigned to.

   :param net: Original network.
   :param new: Its randomization.
   :param attribute: Name of the numeric node attribute to write the groups to, or ``None``.
   :param groups: List of ``(positions, group)`` pairs: network wide positions of nodes of
      ``net`` (tensor) and their group (array).
   :return: ``new``, with the attribute set.
   """
   if attribute is None:
      return new
   g = np.empty(int(net.node_offsets[-1]), dtype = np.float32)
   for pos, c in groups:
      g[pos.cpu().numpy()] = c
   where = {x: i for i, x in enumerate(np.concatenate(net.node_ids).tolist())}
   new.set_node_attribute(attribute, [g[[where[x] for x in ids.tolist()]] for ids in new.node_ids], None)
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


# ------------------------------------------------------------------------------------------------
# Stochastic blockmodels: planted partition, general, degree-corrected, mixed membership, nested
# ------------------------------------------------------------------------------------------------


def _decode(k, n, directed):
   """
   Node pairs from their positions among all the pairs of ``n`` nodes, without self loops.

   :param k: int64 tensor of positions.
   :param n: Number of nodes (int, or int64 tensor aligned with ``k``).
   :param directed: Whether pairs are ordered.
   :return: ``(u, v)``; ``u > v`` if undirected.
   """
   if directed:
      u, r = k // (n - 1), k % (n - 1)
      return u, r + (r >= u).long()
   return _triangle(k)


def _planted(sizes, p_in, p_out, directed, gen, device):
   """
   Edges of a planted partition graph: see :func:`planted_partition_graph`.

   Edges within blocks and between blocks are two Gnp graphs: a binomial number of edges, then a
   uniformly random set of that many pairs. Pairs within blocks are numbered block by block;
   pairs between blocks are drawn among all pairs, keeping those between blocks (or, when they
   are few and dense, listed and permuted).

   :param sizes: int64 tensor, number of nodes of each block; block ``b`` holds the consecutive
      nodes after those of the earlier blocks.
   :return: Source and target ids, int64 tensors.
   """
   n = int(sizes.sum())
   start = torch.cumsum(sizes, 0) - sizes
   block = torch.repeat_interleave(torch.arange(sizes.shape[0], device = device), sizes)
   c = sizes * (sizes - 1) // (1 if directed else 2)   # pairs within each block
   C = torch.cumsum(c, 0)
   n_in = int(C[-1]) if sizes.shape[0] else 0
   n_all = n * (n - 1) // (1 if directed else 2)
   k = _choose(n_in, _binomial(n_in, p_in, gen, device), gen, device)
   b = torch.searchsorted(C, k, right = True)
   u, v = _decode(k - (C[b] - c[b]), sizes[b], directed)
   s, t = [start[b] + u], [start[b] + v]
   m = _binomial(n_all - n_in, p_out, gen, device)
   if m:
      if 4 * m >= n_all - n_in and n_all <= 2 ** 26:   # dense: list all pairs between blocks
         a, z = _decode(torch.arange(n_all, device = device), n, directed)
         a, z = a[block[a] != block[z]], z[block[a] != block[z]]
         pick = torch.randperm(a.shape[0], generator = gen, device = device)[:m]
         s.append(a[pick])
         t.append(z[pick])
      else:   # sparse: random pairs among all, kept if between blocks, until m distinct
         got = k[:0]
         while got.shape[0] < m:
            draw = torch.randint(n_all, (int(1.2 * (m - got.shape[0]) * n_all / (n_all - n_in)) + 16,), generator = gen, device = device)
            a, z = _decode(draw, n, directed)
            got = torch.unique(torch.cat([got, draw[block[a] != block[z]]]))
         got = got[torch.randperm(got.shape[0], generator = gen, device = device)[:m]]
         a, z = _decode(got, n, directed)
         s.append(a)
         t.append(z)
   return torch.cat(s), torch.cat(t)


def planted_partition_graph(sizes = None, p_in = None, p_out = None, net = None, attribute = "community", directed = None, seed = None, device = None):
   """
   Planted partition graph, the classic stochastic blockmodel (Holland, Laskey and Leinhardt,
   1983; Condon and Karp, 2001): nodes are split into communities of the given sizes, and every
   pair of nodes is an edge independently, with probability ``p_in`` if both are in the same
   community and ``p_out`` otherwise. With ``p_in > p_out`` communities are denser inside than
   between them (assortative), as in community detection benchmarks; with ``p_in < p_out`` the
   structure is disassortative.

   Edges are drawn as two Gnp graphs (see :func:`gnp_random_graph`), within and between
   communities: a binomial number of edges, then a uniformly random set of that many pairs, on the
   device, in time about linear in the number of edges. The graph has no self loops nor parallel
   edges.

   Without ``net``, nodes are the integers ``0`` to ``n - 1``, community by community, and the
   community of each node (its position in ``sizes``) is stored as the numeric node attribute
   ``attribute`` (unless ``None``), a planted partition for e.g. :func:`~graphoskop.modularity`.
   With ``net``, the result is a randomization of ``net`` (see the module description): its
   nodes keep their communities, read from the node attribute ``attribute`` (numeric or
   qualitative; every node needs a value), which give ``sizes``; ``p_in`` and ``p_out`` are the
   densities of the edges of ``net`` within and between communities (self loops left out,
   parallel edges counting), unless given. In multilayer networks each layer is randomized on its
   own, with the communities of its copies and its own densities (``p_in`` and ``p_out`` can be
   one value for all layers or a dictionary from layer to value); interlayer edges are kept.

   :param sizes: Number of nodes of each community; ``None`` with ``net``.
   :param p_in: Probability of an edge between two nodes of the same community; ``None`` with
      ``net``, for its own.
   :param p_out: Probability of an edge between two nodes of different communities; ``None`` with
      ``net``, for its own.
   :param net: A :class:`~graphoskop.Network` to randomize, not bipartite, or ``None``.
   :param attribute: Name of the node attribute holding the community of each node: read from
      ``net``, or written to the result without ``net`` (``None`` for no attribute).
   :param directed: Whether the edges are directed (ordered pairs); ``None`` for undirected, or
      for the direction of ``net``.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The planted partition graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If neither ``sizes``, ``p_in`` and ``p_out`` nor ``net`` are given, if
      ``sizes`` is given with ``net``, if a probability is not between 0 and 1, if ``net`` is
      bipartite, has no node attribute ``attribute`` or a node without a value, if ``directed``
      or ``device`` differ from those of ``net``, or for dictionaries as in
      :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.planted_partition_graph([100, 100, 50], 0.2, 0.01, seed = 1)   # 3 communities
      gsk.modularity(g, "community")

      net = gsk.load_network("edges.tsv", "nodes.tsv")                       # node attribute "community"
      rnd = gsk.planted_partition_graph(net = net, seed = 1)                 # same communities and densities
      rnd = gsk.planted_partition_graph(net = net, attribute = "party", p_out = 0)   # no edge between parties
   """
   for q in [x for v in (p_in, p_out) for x in (v.values() if isinstance(v, dict) else [v])]:
      if q is not None and not 0 <= q <= 1:
         raise ValueError(f"probabilities must be between 0 and 1, got {q}")
   if net is None:
      if sizes is None or p_in is None or p_out is None:
         raise ValueError("give sizes, p_in and p_out, or net")
      directed, device = bool(directed), torch.device("cpu" if device is None else device)
      sz = torch.as_tensor(list(sizes), dtype = torch.long, device = device)
      s, t = _planted(sz, p_in, p_out, directed, _generator(seed, device), device)
      comm = np.repeat(np.arange(len(sz)), sz.cpu().numpy()).astype(np.float32)
      table = None if attribute is None else lambda perm: comm[perm][:, None]
      return _build(s.cpu().numpy(), t.cpu().numpy(), np.arange(comm.shape[0]), np.zeros((s.shape[0], 0), dtype = np.float32), [], table, [] if attribute is None else [attribute], directed, device)
   if sizes is not None:
      raise ValueError("with net, sizes come from its communities: leave them out")
   code, _, S, T = _communities(net, attribute, directed, device)

   def make(l, nodes, n0, count, loops, gen):
      order = torch.argsort(code[nodes], stable = True)   # community by community
      _, sizes = torch.unique(code[nodes], return_counts = True)
      mine = torch.isin(S, nodes) & torch.isin(T, nodes)   # intra-layer edges of this layer
      inside = int((code[S[mine]] == code[T[mine]]).sum())
      pairs = lambda x: int((x * (x - 1)).sum()) // (1 if net.directed else 2)
      n_in, n_all = pairs(sizes), pairs(torch.tensor([nodes.shape[0]]))
      q_in = _layer_value(p_in, None, net, l, "p_in", True)
      q_out = _layer_value(p_out, None, net, l, "p_out", True)
      if q_in is None:
         q_in = min(inside / n_in, 1) if n_in else 0
      if q_out is None:
         q_out = min((int(mine.sum()) - inside) / (n_all - n_in), 1) if n_all > n_in else 0
      s, t = _planted(sizes, q_in, q_out, net.directed, gen, nodes.device)
      return order[s], order[t]

   return _randomize(net, seed, make)


def _communities(net, attribute, directed, device, overlap = False):
   """
   Communities of the nodes of a network, for the blockmodels, with its edges.

   :param net: Network, not bipartite.
   :param attribute: Name of the node attribute with the communities.
   :param directed: ``directed`` as given to the generator, checked against ``net``.
   :param device: ``device`` as given to the generator, checked against ``net``.
   :param overlap: Whether nodes can be in several communities (a tuple of values).
   :return: Community code of every node (int64 tensor in the network wide node order; codes
      follow the sorted values, or their order of appearance if they cannot be sorted), the
      community values by code, and the sources and targets of the edges that are not self
      loops, by network wide position. With ``overlap``, the code is replaced by two int64
      tensors, node position and community code of every membership, sorted by node then code.
   :raises ValueError: If ``net`` is bipartite, has no such attribute or a node without a value,
      if a node has several communities without ``overlap``, or if ``directed`` or ``device``
      differ from those of ``net``.
   """
   if net.bipartite_labels is not None:
      raise ValueError("bipartite networks are not supported")
   if attribute not in net.node_attributes and attribute not in net.qualitative_node_attributes:
      raise ValueError(f"net has no node attribute {attribute!r} with the communities")
   _check_net(net, None, directed, device)
   values = _memberships(np.concatenate([np.asarray(v.cpu().numpy() if torch.is_tensor(v) else v, dtype = object) for v in net.get_node_attribute(attribute, None)]))
   cols, flat = _explode({"node": np.arange(len(values))}, values)
   if len(flat) > len(values) and not overlap:
      raise ValueError(f"some nodes have several {attribute!r} values (overlapping communities): use mixed_membership_block_model")
   flat = pd.Series(flat, dtype = object)
   try:
      code, labels = pd.factorize(flat, sort = True)
   except TypeError:
      code, labels = pd.factorize(flat)
   if (code < 0).any():
      raise ValueError(f"node {np.concatenate(net.node_ids)[cols['node'][np.argmax(code < 0)]]!r} has no {attribute!r}")
   S, T = [], []
   for c in range(len(net.components)):
      ei = net._input_edges(c)[0]
      S.append(ei[0] + int(net.node_offsets[c]))
      T.append(ei[1] + int(net.node_offsets[c]))
   S, T = torch.cat(S), torch.cat(T)
   labels = [x.item() if isinstance(x, np.generic) else x for x in labels]
   if overlap:
      order = np.lexsort((code, cols["node"]))
      node = torch.as_tensor(cols["node"][order].astype(np.int64), device = S.device)
      return (node, torch.as_tensor(code[order].astype(np.int64), device = S.device)), labels, S[S != T], T[S != T]
   return torch.as_tensor(code, device = S.device), labels, S[S != T], T[S != T]


def _blocks(sizes, P, directed, gen, device):
   """
   Edges of a stochastic blockmodel: see :func:`stochastic_block_model`.

   Every pair of blocks (each block with itself, and each unordered pair, or ordered if directed)
   is a Gnp graph: a binomial number of edges, all drawn at once, then that many distinct pairs
   uniformly at random in each, all block pairs together: by random ranks among all their pairs
   when dense, by drawing positions until enough are distinct when sparse.

   :param sizes: int64 tensor, number of nodes of each block (consecutive nodes).
   :param P: float64 tensor, ``k`` by ``k`` probabilities (symmetric if undirected).
   :return: Source and target ids, int64 tensors.
   """
   k = sizes.shape[0]
   start = torch.cumsum(sizes, 0) - sizes
   A, B = torch.triu_indices(k, k, device = device) if not directed else torch.cartesian_prod(torch.arange(k, device = device), torch.arange(k, device = device)).T
   same = A == B
   N = torch.where(same, sizes[A] * (sizes[A] - 1) // (1 if directed else 2), sizes[A] * sizes[B])
   m = torch.binomial(N.double(), P[A, B].double().clamp(0, 1), generator = gen).long()
   live = m > 0
   A, B, N, m, same = A[live], B[live], N[live], m[live], same[live]
   grp, pos = [m[:0]], [m[:0]]
   dense = 4 * m >= N
   for d in (True, False):
      g = torch.nonzero(dense == d).squeeze(1)
      if g.shape[0] == 0:
         continue
      if d:   # all pairs, random ranks
         gi = torch.repeat_interleave(g, N[g])
         pi = _expand(torch.zeros_like(g), N[g])
      else:   # positions drawn until enough are distinct in every block pair
         off = torch.cumsum(N[g], 0) - N[g]
         keys, have = m[:0], torch.zeros_like(g)
         while (have < m[g]).any():
            c = torch.where(have < m[g], (1.2 * (m[g] - have)).long() + 4, 0)
            j = torch.repeat_interleave(torch.arange(g.shape[0], device = device), c)
            draw = off[j] + (torch.rand(j.shape[0], generator = gen, device = device, dtype = torch.float64) * N[g][j]).long()
            keys = torch.unique(torch.cat([keys, draw]))
            have = torch.bincount(torch.searchsorted(off, keys, right = True) - 1, minlength = g.shape[0])
         j = torch.searchsorted(off, keys, right = True) - 1
         gi, pi = g[j], keys - off[j]
      r = torch.rand(gi.shape[0], generator = gen, device = device, dtype = torch.float64)
      o = torch.argsort(gi * 2.0 + r)   # by block pair, then at random (r < 1)
      gi, pi = gi[o], pi[o]
      first = torch.cumsum(torch.bincount(gi, minlength = N.shape[0]), 0) - torch.bincount(gi, minlength = N.shape[0])
      keep = torch.arange(gi.shape[0], device = device) - first[gi] < m[gi]
      grp.append(gi[keep])
      pos.append(pi[keep])
   g, p = torch.cat(grp), torch.cat(pos)
   a, b = A[g], B[g]
   u, v = _decode(p, sizes[a].clamp(min = 2), directed)   # within a block (size 2 or more there)
   s = torch.where(same[g], start[a] + u, start[a] + p // sizes[b].clamp(min = 1))
   t = torch.where(same[g], start[a] + v, start[b] + p % sizes[b].clamp(min = 1))
   return s, t


def stochastic_block_model(sizes = None, probs = None, net = None, attribute = "community", directed = None, seed = None, device = None):
   """
   Stochastic blockmodel (Holland, Laskey and Leinhardt, 1983): nodes are split into blocks
   (communities) of the given sizes, and every pair of nodes is an edge independently, with a
   probability that depends on their two blocks: ``probs[a][b]`` for a node of block ``a`` and
   one of block ``b``. Any mixing pattern can be planted: assortative communities (large
   diagonal), disassortative or bipartite-like structures, core-periphery, hierarchies.
   :func:`planted_partition_graph` is the special case with one probability inside blocks and
   one between them.

   Every pair of blocks is a Gnp graph (see :func:`gnp_random_graph`): a binomial number of
   edges, then a uniformly random set of that many pairs, all pairs of blocks at once on the
   device, in time about linear in the number of edges and in the number of pairs of blocks. The
   graph has no self loops nor parallel edges. In undirected graphs ``probs`` must be symmetric;
   in directed graphs ``probs[a][b]`` is the probability of an edge from block ``a`` to block
   ``b``.

   Without ``net``, nodes are the integers ``0`` to ``n - 1``, block by block, and the block of
   each node (its position in ``sizes``) is stored as the numeric node attribute ``attribute``
   (unless ``None``). With ``net``, the result is a randomization of ``net`` (see the module
   description): its nodes keep their communities, read from the node attribute ``attribute``
   (numeric or qualitative; every node needs a value), which give ``sizes``; ``probs`` is the
   density of the edges of ``net`` between each pair of communities (self loops left out,
   parallel edges counting), unless given, as a matrix over the communities in sorted order (or
   order of appearance, if their values cannot be sorted). In multilayer networks each layer is
   randomized on its own, with the communities of its copies and its own densities (a ``probs``
   given applies to the communities present in each layer, or can be a dictionary from layer to
   matrix); interlayer edges are kept.

   :param sizes: Number of nodes of each block; ``None`` with ``net``.
   :param probs: ``k`` by ``k`` matrix of probabilities (nested lists, array or tensor); ``None``
      with ``net``, for its own.
   :param net: A :class:`~graphoskop.Network` to randomize, not bipartite, or ``None``.
   :param attribute: Name of the node attribute holding the block of each node: read from
      ``net``, or written to the result without ``net`` (``None`` for no attribute).
   :param directed: Whether the edges are directed; ``None`` for undirected, or for the direction
      of ``net``.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The stochastic blockmodel graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If neither ``sizes`` and ``probs`` nor ``net`` are given, if ``sizes`` is
      given with ``net``, if ``probs`` is not a ``k`` by ``k`` matrix of probabilities between 0
      and 1, or not symmetric for an undirected graph, if ``net`` is bipartite, has no node
      attribute ``attribute`` or a node without a value, if ``directed`` or ``device`` differ
      from those of ``net``, or for dictionaries as in :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      core_periphery = [[0.5, 0.1], [0.1, 0.01]]
      g = gsk.stochastic_block_model([50, 450], core_periphery, seed = 1)
      g = gsk.stochastic_block_model([100, 100], [[0.0, 0.1], [0.02, 0.0]], directed = True)   # from block 0 to 1

      net = gsk.load_network("edges.tsv", "nodes.tsv")                 # node attribute "community"
      rnd = gsk.stochastic_block_model(net = net, seed = 1)            # same densities between every pair of communities
   """

   def matrix(x, k):
      P = torch.as_tensor(np.asarray(x, dtype = np.float64))
      if P.shape != (k, k):
         raise ValueError(f"probs must be a {k} by {k} matrix, got shape {tuple(P.shape)}")
      if not bool(((P >= 0) & (P <= 1)).all()):
         raise ValueError("probabilities must be between 0 and 1")
      if not directed and not torch.equal(P, P.T):
         raise ValueError("probs must be symmetric for an undirected graph")
      return P

   if net is None:
      if sizes is None or probs is None:
         raise ValueError("give sizes and probs, or net")
      directed, device = bool(directed), torch.device("cpu" if device is None else device)
      sz = torch.as_tensor(list(sizes), dtype = torch.long, device = device)
      P = matrix(probs, sz.shape[0]).to(device)
      s, t = _blocks(sz, P, directed, _generator(seed, device), device)
      comm = np.repeat(np.arange(len(sz)), sz.cpu().numpy()).astype(np.float32)
      table = None if attribute is None else lambda perm: comm[perm][:, None]
      return _build(s.cpu().numpy(), t.cpu().numpy(), np.arange(comm.shape[0]), np.zeros((s.shape[0], 0), dtype = np.float32), [], table, [] if attribute is None else [attribute], directed, device)
   if sizes is not None:
      raise ValueError("with net, sizes come from its communities: leave them out")
   code, labels, S, T = _communities(net, attribute, directed, device)
   directed = net.directed

   def make(l, nodes, n0, count, loops, gen):
      order = torch.argsort(code[nodes], stable = True)   # community by community
      present, sizes = torch.unique(code[nodes], return_counts = True)
      given = _layer_value(probs, None, net, l, "probs", True)
      if given is not None:
         P = matrix(given, len(labels)).to(nodes.device)[present][:, present]
      else:
         local = torch.full((len(labels),), -1, dtype = torch.long, device = nodes.device)
         local[present] = torch.arange(present.shape[0], device = nodes.device)
         mine = torch.isin(S, nodes) & torch.isin(T, nodes)   # intra-layer edges of this layer
         a, b = local[code[S[mine]]], local[code[T[mine]]]
         if not directed:
            a, b = torch.minimum(a, b), torch.maximum(a, b)
         k = present.shape[0]
         E = torch.bincount(a * k + b, minlength = k * k).view(k, k).double()
         N = torch.outer(sizes, sizes).double() - torch.diag(sizes.double())
         if not directed:
            E, N = torch.triu(E) + torch.triu(E, 1).T, N / (1 + torch.eye(k, device = nodes.device, dtype = torch.float64))
         P = torch.where(N > 0, E / N.clamp(min = 1), 0).clamp(max = 1)
      s, t = _blocks(sizes, P, directed, gen, nodes.device)
      return order[s], order[t]

   return _randomize(net, seed, make)


def _dc_blocks(sizes, w_out, w_in, omega, directed, exact, rounds, drop_self_loops, drop_parallel, gen, device):
   """
   Edges of a degree-corrected blockmodel: see :func:`degree_corrected_block_model`.

   The number of edges of each pair of blocks is Poisson with mean ``omega[r][s]`` (or its rounded
   value, if ``exact``), all pairs at once; every edge then picks its endpoints in its two blocks,
   each node with probability proportional to its weight. Weights are normalized within each block,
   so that block ``r`` spans ``[r, r + 1)`` of their cumulative sum, and an endpoint is one
   ``searchsorted``. Self loops and repeated edges are drawn again within their pair of blocks, for
   up to ``rounds`` rounds, every redrawn edge kept only if new; the keys of the edges are sorted
   once (redrawn edges never take a pair already there). Leftovers are dropped as asked.

   :param sizes: int64 tensor, number of nodes of each block (consecutive nodes).
   :param w_out: float64 tensor, weight of each node as a source (as an endpoint if undirected).
   :param w_in: float64 tensor, weight of each node as a target (ignored if undirected).
   :param omega: float64 tensor, ``k`` by ``k`` expected numbers of edges (from ``r`` to ``s`` if
      directed; symmetric otherwise, the diagonal counting the edges within each block).
   :return: Source and target ids, int64 tensors.
   """
   k = sizes.shape[0]
   block = torch.repeat_interleave(torch.arange(k, device = device), sizes)

   def cumulative(w):   # normalized within each block; a block of zero weights is uniform
      tot = torch.zeros(k, dtype = torch.float64, device = device).index_add_(0, block, w)
      w = torch.where(tot[block] > 0, w / tot[block].clamp(min = 1e-300), 1 / sizes[block].double())
      return torch.cumsum(w, 0)

   c_out = cumulative(w_out)
   c_in = c_out if not directed else cumulative(w_in)
   n = int(sizes.sum())
   start = torch.cumsum(sizes, 0) - sizes
   pick = lambda c, r: torch.minimum(torch.maximum(torch.searchsorted(c, r.double() + torch.rand(r.shape[0], generator = gen, device = device, dtype = torch.float64), right = True), start[r]), start[r] + sizes[r] - 1)   # inside block r despite rounding
   A, B = torch.triu_indices(k, k, device = device) if not directed else torch.cartesian_prod(torch.arange(k, device = device), torch.arange(k, device = device)).T
   lam = omega[A, B].clamp(min = 0)
   M = torch.round(lam).long() if exact else torch.poisson(lam, generator = gen).long()
   g = torch.repeat_interleave(torch.arange(A.shape[0], device = device), M)
   a, b = A[g], B[g]
   s, t = pick(c_out, a), pick(c_in, b)
   if s.shape[0] == 0 or (rounds == 0 and not drop_self_loops and not drop_parallel):
      return s, t
   key = lambda x, y: x * n + y if directed else torch.minimum(x, y) * n + torch.maximum(x, y)
   kk = key(s, t)
   sk, inv = torch.unique(kk, return_inverse = True)
   first = torch.full((sk.shape[0],), kk.shape[0], device = device).scatter_reduce(0, inv, torch.arange(kk.shape[0], device = device), "amin")
   bad = (s == t) | torch.ones_like(kk, dtype = torch.bool).index_fill_(0, first, False)
   new = kk[:0]
   for _ in range(rounds):
      R = torch.nonzero(bad).squeeze(1)
      if R.shape[0] == 0:
         break
      s1, t1 = pick(c_out, a[R]), pick(c_in, b[R])
      k1 = key(s1, t1)
      ok = (s1 != t1) & ~(sk[torch.searchsorted(sk, k1).clamp(max = sk.shape[0] - 1)] == k1) & ~torch.isin(k1, new)
      _, iv, cnt = torch.unique(k1, return_inverse = True, return_counts = True)
      ok &= cnt[iv] == 1   # two redraws make the same pair: neither is kept this round
      R = R[ok]
      s[R], t[R] = s1[ok], t1[ok]
      bad[R] = False
      new = torch.cat([new, k1[ok]])
   loop = s == t
   keep = ~(bad & ((loop & drop_self_loops) | (~loop & drop_parallel)))
   return s[keep], t[keep]


def degree_corrected_block_model(sizes = None, degrees = None, edges = None, net = None, attribute = "community", directed = None, exact = False, rounds = 10, drop_self_loops = True, drop_parallel = True, seed = None, device = None):
   """
   Degree-corrected stochastic blockmodel (Karrer and Newman, 2011): a stochastic blockmodel in
   which nodes of the same block can have very different degrees, as in real communities with
   hubs and peripheral members.

   ``edges[r][s]`` is the expected number of edges between blocks ``r`` and ``s`` (from ``r`` to
   ``s`` if directed; the diagonal counts the edges within each block), and ``degrees`` gives the
   expected degree of every node relative to the others of its block: each edge between ``r`` and
   ``s`` picks its endpoint in ``r`` with probability proportional to its degree within ``r``,
   :math:`\\theta_i = d_i / \\sum_{j \\in r} d_j`, and likewise in ``s``. A node then expects
   :math:`\\theta_i \\kappa_r` edges, :math:`\\kappa_r` being the edges of its block (twice those
   within), so degrees that add up to :math:`\\kappa_r` in each block are kept on average. The number
   of edges of each pair of blocks is Poisson, as in the original model, or exactly ``edges[r][s]``
   (rounded) with ``exact = True``. A block whose degrees are all 0 picks its nodes uniformly.

   Drawing endpoints independently makes some self loops and parallel edges, mostly among hubs.
   As in :func:`configuration_model`, without slowing generation down, each of them is drawn again
   within its pair of blocks for up to ``rounds`` rounds, in parallel, and those left are dropped,
   unless ``drop_self_loops`` or ``drop_parallel`` is ``False``; with ``rounds = 0`` and both kept,
   the graph is the original multigraph model. Redrawing and dropping lower the degrees of hubs a
   little when they are large compared with the square root of the number of edges of their blocks.
   Everything runs on the device, in time about linear in the number of edges and of pairs of
   blocks.

   Without ``net``, nodes are the integers ``0`` to ``n - 1``, block by block (``sizes``), with
   ``degrees`` in that order (a pair ``(out_degrees, in_degrees)`` if directed), and the block of
   each node is stored as the numeric node attribute ``attribute`` (unless ``None``). With ``net``,
   the result is a randomization of ``net`` (see the module description): its nodes keep their
   communities, read from the node attribute ``attribute`` (numeric or qualitative; every node needs
   a value), which give ``sizes``; ``degrees`` are their degrees (out and in if directed) and
   ``edges`` the numbers of edges between communities, self loops left out, parallel edges
   counting, so that the result keeps both on average (exactly the numbers of edges between
   communities with ``exact = True``, before drops). ``edges`` can be given instead, as a matrix over
   the communities in sorted order (or order of appearance, if their values cannot be sorted), or a
   dictionary from layer to matrix. In multilayer networks each layer is randomized on its own, with
   the degrees and edges of its intra-layer edges; interlayer edges are kept.

   :param sizes: Number of nodes of each block; ``None`` with ``net``.
   :param degrees: Expected degree of each node, block by block (a pair of out- and in-degrees if
      directed); ``None`` with ``net``.
   :param edges: ``k`` by ``k`` matrix of expected numbers of edges between blocks; ``None`` with
      ``net``, for its own.
   :param net: A :class:`~graphoskop.Network` to randomize, not bipartite, or ``None``.
   :param attribute: Name of the node attribute holding the block of each node: read from ``net``,
      or written to the result without ``net`` (``None`` for no attribute).
   :param directed: Whether the edges are directed; ``None`` for undirected, or for the direction
      of ``net``.
   :param exact: Whether each pair of blocks gets exactly ``edges[r][s]`` edges (rounded) before
      drops, instead of a Poisson number.
   :param rounds: Maximum number of rounds of redrawing self loops and parallel edges (0 for none).
   :param drop_self_loops: Whether to drop the self loops left.
   :param drop_parallel: Whether to drop the parallel edges left (one edge of each group stays).
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The degree-corrected blockmodel graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If neither ``sizes``, ``degrees`` and ``edges`` nor ``net`` are given, if
      ``sizes`` or ``degrees`` are given with ``net``, if ``degrees`` does not have one non-negative
      value per node, if ``edges`` is not a ``k`` by ``k`` non-negative matrix, symmetric for an
      undirected graph, if ``net`` is bipartite, has no node attribute ``attribute`` or a node
      without a value, if ``directed`` or ``device`` differ from those of ``net``, or for
      dictionaries as in :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      degrees = [10, 1, 1, 1, 1, 6, 2, 2, 2]                  # a hub in each block
      g = gsk.degree_corrected_block_model([5, 4], degrees, [[6, 1], [1, 5]], seed = 1)

      net = gsk.load_network("edges.tsv", "nodes.tsv")         # node attribute "community"
      rnd = gsk.degree_corrected_block_model(net = net, seed = 1)   # degrees and block edges of net, on average
      rnd = gsk.degree_corrected_block_model(net = net, exact = True)
   """

   def matrix(x, k):
      E = torch.as_tensor(np.asarray(x, dtype = np.float64))
      if E.shape != (k, k) or not bool((E >= 0).all()):
         raise ValueError(f"edges must be a {k} by {k} matrix of non-negative numbers, got shape {tuple(E.shape)}")
      if not directed and not torch.equal(E, E.T):
         raise ValueError("edges must be symmetric for an undirected graph")
      return E

   if net is None:
      if sizes is None or degrees is None or edges is None:
         raise ValueError("give sizes, degrees and edges, or net")
      directed, device = bool(directed), torch.device("cpu" if device is None else device)
      sz = torch.as_tensor(list(sizes), dtype = torch.long, device = device)
      d = [torch.as_tensor(list(x), dtype = torch.float64, device = device) for x in (degrees if directed else [degrees])]
      if any(x.shape != (int(sz.sum()),) or bool((x < 0).any()) for x in d):
         raise ValueError(f"degrees must have one non-negative value per node ({int(sz.sum())})")
      E = matrix(edges, sz.shape[0]).to(device)
      s, t = _dc_blocks(sz, d[0], d[-1], E, directed, exact, rounds, drop_self_loops, drop_parallel, _generator(seed, device), device)
      comm = np.repeat(np.arange(len(sz)), sz.cpu().numpy()).astype(np.float32)
      table = None if attribute is None else lambda perm: comm[perm][:, None]
      return _build(s.cpu().numpy(), t.cpu().numpy(), np.arange(comm.shape[0]), np.zeros((s.shape[0], 0), dtype = np.float32), [], table, [] if attribute is None else [attribute], directed, device)
   if sizes is not None or degrees is not None:
      raise ValueError("with net, sizes and degrees come from its communities and edges: leave them out")
   code, labels, S, T = _communities(net, attribute, directed, device)
   directed = net.directed

   def make(l, nodes, n0, count, loops, gen):
      order = torch.argsort(code[nodes], stable = True)   # community by community
      present, sizes = torch.unique(code[nodes], return_counts = True)
      sorted_nodes = nodes[order]
      mine = torch.isin(S, nodes) & torch.isin(T, nodes)   # intra-layer edges of this layer, no self loops
      pos = torch.full((int(code.shape[0]),), -1, dtype = torch.long, device = nodes.device)
      pos[sorted_nodes] = torch.arange(nodes.shape[0], device = nodes.device)
      u, v = pos[S[mine]], pos[T[mine]]
      w_out = torch.bincount(u, minlength = nodes.shape[0]).double()
      w_in = torch.bincount(v, minlength = nodes.shape[0]).double()
      if not directed:
         w_out = w_in = w_out + w_in
      given = _layer_value(edges, None, net, l, "edges", True)
      if given is not None:
         E = matrix(given, len(labels)).to(nodes.device)[present][:, present]
      else:
         local = torch.full((len(labels),), -1, dtype = torch.long, device = nodes.device)
         local[present] = torch.arange(present.shape[0], device = nodes.device)
         a, b = local[code[S[mine]]], local[code[T[mine]]]
         k = present.shape[0]
         E = torch.bincount(a * k + b, minlength = k * k).view(k, k).double()
         if not directed:
            E = E + E.T - torch.diag(torch.diag(E))
      s, t = _dc_blocks(sizes, w_out, w_in, E, directed, exact, rounds, drop_self_loops, drop_parallel, gen, nodes.device)
      return order[s], order[t]

   return _randomize(net, seed, make)


def _membership_pairs(first, count, s, t, chunk = 1 << 24):
   """
   All pairs of memberships of pairs of nodes, in chunks of about ``chunk`` pairs.

   :param first: int64 tensor, position of the first membership of every node (memberships
      sorted by node).
   :param count: int64 tensor, number of memberships of every node.
   :param s: int64 tensor, one node of every pair.
   :param t: int64 tensor, other node of every pair.
   :param chunk: Number of pairs of memberships per chunk.
   :return: Generator of ``(e, i, j)``: pair position, membership of ``s[e]`` and membership of
      ``t[e]``, int64 tensors.
   """
   size = count[s] * count[t]
   end = torch.cumsum(size, 0)
   lo = 0
   while lo < s.shape[0]:
      base = int(end[lo] - size[lo])
      hi = max(int(torch.searchsorted(end, base + chunk, right = True)), lo + 1)
      e = torch.repeat_interleave(torch.arange(lo, hi, device = s.device), size[lo:hi])
      r = torch.arange(e.shape[0], device = s.device) - (end[e] - size[e] - base)
      c = count[t[e]]
      yield e, first[s[e]] + r // c, first[t[e]] + r % c
      lo = hi


def _mixed_blocks(node, block, weight, n, P, directed, gen, device):
   """
   Edges of a mixed membership blockmodel: see :func:`mixed_membership_block_model`.

   Candidate pairs come from a Poisson process with rate :math:`c\\, p_{ij}` on every pair, which
   dominates the edge probabilities :math:`p_{ij} = \\pi_i^T B \\pi_j`: for every pair of blocks
   ``(a, b)``, a Poisson number of candidates with mean :math:`c B_{ab} S_a S_b` (half of it if
   undirected, as both orders are drawn), :math:`S_a = \\sum_i \\pi_{ia}`, each joining a node drawn
   with probability :math:`\\pi_{ia} / S_a` to one drawn with probability :math:`\\pi_{jb} / S_b`.
   A pair is a candidate with probability :math:`1 - e^{-c p_{ij}}`, and a candidate is kept with
   probability :math:`p_{ij} / (1 - e^{-c p_{ij}})`, so pairs are edges independently with
   probability exactly :math:`p_{ij}`, :math:`c = -\\ln(1 - p_{max}) / p_{max}` making that at most
   1. The work is about linear in the number of edges and of pairs of blocks.

   :param node: int64 tensor, node of every membership, sorted.
   :param block: int64 tensor, block of every membership.
   :param weight: float64 tensor, weight of every membership (adding up to 1 for every node).
   :param n: Number of nodes.
   :param P: float64 tensor, ``k`` by ``k`` probabilities (symmetric if undirected).
   :return: Source and target ids, int64 tensors.
   """
   k = P.shape[0]
   pmax = min(float(P.max()) if k else 0., 1 - 1e-9)
   if pmax <= 0:
      return node[:0], node[:0]
   c = -math.log1p(-pmax) / pmax
   count = torch.bincount(node, minlength = n)
   first = torch.cumsum(count, 0) - count

   # memberships block by block, with cumulative weights to draw nodes within a block
   o = torch.argsort(block, stable = True)
   bnode, W = node[o], torch.cumsum(weight[o], 0)
   size = torch.bincount(block, minlength = k)
   end = torch.cumsum(size, 0)
   start = end - size
   S = torch.zeros(k, dtype = torch.float64, device = device).index_add_(0, block, weight)
   before = torch.cat([W.new_zeros(1), W])[start]

   def draw(x):
      i = torch.searchsorted(W, before[x] + torch.rand(x.shape[0], generator = gen, device = device, dtype = torch.float64) * S[x], right = True)
      return bnode[torch.minimum(torch.maximum(i, start[x]), end[x] - 1)]   # within the block, despite rounding

   rate = (1 if directed else 0.5) * c * P * torch.outer(S, S)
   N = torch.poisson(rate.flatten(), generator = gen).long()
   pair = torch.repeat_interleave(torch.arange(k * k, device = device), N)
   s, t = draw(pair // k), draw(pair % k)
   s, t = s[s != t], t[s != t]
   if not directed:
      s, t = torch.minimum(s, t), torch.maximum(s, t)
   key = torch.unique(s * n + t)
   s, t = key // n, key % n

   # probability of every candidate pair, then thinning
   p = torch.zeros(s.shape[0], dtype = torch.float64, device = device)
   if 4 * node.shape[0] >= n * k:   # dense memberships: pi_i^T B once per node, then k terms per pair
      pi = torch.zeros(n, k, dtype = torch.float64, device = device)
      pi[node, block] = weight
      piB, step = pi @ P, max(1, (1 << 24) // k)
      for lo in range(0, s.shape[0], step):
         p[lo:lo + step] = (piB[s[lo:lo + step]] * pi[t[lo:lo + step]]).sum(1)
   else:
      for e, i, j in _membership_pairs(first, count, s, t):
         p.index_add_(0, e, weight[i] * P[block[i], block[j]] * weight[j])
   keep = torch.rand(p.shape[0], generator = gen, device = device, dtype = torch.float64) * -torch.expm1(-c * p) < p
   return s[keep], t[keep]


def _mixed_probs(node, block, weight, n, k, s, t, directed):
   """
   Probabilities between blocks of a mixed membership blockmodel that give, in expectation, the
   observed weight of the edges between every pair of blocks: see
   :func:`mixed_membership_block_model`.

   :param node: int64 tensor, node of every membership, sorted.
   :param block: int64 tensor, block of every membership.
   :param weight: float64 tensor, weight of every membership.
   :param n: Number of nodes.
   :param k: Number of blocks.
   :param s: int64 tensor, sources of the edges (no self loops).
   :param t: int64 tensor, targets of the edges.
   :param directed: Whether the edges are directed.
   :return: ``k`` by ``k`` float64 tensor of probabilities.
   """
   device = node.device
   count = torch.bincount(node, minlength = n)
   first = torch.cumsum(count, 0) - count
   E = torch.zeros(k * k, dtype = torch.float64, device = device)
   for e, i, j in _membership_pairs(first, count, s, t):
      E.index_add_(0, block[i] * k + block[j], weight[i] * weight[j])
   E = E.view(k, k)
   if not directed:
      E = E + E.T   # both orders, as the pairs below
   v = torch.arange(n, device = device)
   e, i, j = (torch.cat(x) for x in zip(*_membership_pairs(first, count, v, v)))
   ww, bij = weight[i] * weight[j], block[i] * k + block[j]
   M = torch.zeros(k * k, dtype = torch.float64, device = device).index_add_(0, bij, ww).view(k, k)
   Mp = torch.linalg.pinv(M)
   B = Mp @ E @ Mp
   for _ in range(200):   # E = M B M - (pairs of a node with itself)
      q = torch.zeros(n, dtype = torch.float64, device = device).index_add_(0, e, ww * B.flatten()[bij])
      D = torch.zeros(k * k, dtype = torch.float64, device = device).index_add_(0, bij, ww * q[e]).view(k, k)
      new = Mp @ (E + D) @ Mp
      done = torch.allclose(new, B, rtol = 1e-12, atol = 1e-15)
      B = new
      if done:
         break
   if not directed:
      B = (B + B.T) / 2
   return B.clamp(0, 1)


def mixed_membership_block_model(probs = None, memberships = None, n = None, alpha = None, net = None, attribute = "community", threshold = None, directed = None, seed = None, device = None):
   """
   Mixed membership stochastic blockmodel (Airoldi, Blei, Fienberg and Xing, 2008): every node
   belongs to several blocks (overlapping communities) with a vector of weights
   :math:`\\pi_i` adding up to 1, and every pair of nodes is an edge independently with probability
   :math:`p_{ij} = \\pi_i^T B \\pi_j = \\sum_{a, b} \\pi_{ia} B_{ab} \\pi_{jb}`, ``B`` being the matrix
   ``probs`` of probabilities between blocks (:func:`stochastic_block_model` is the special case of
   nodes in one block each). This is the version of the model with fixed memberships, as in its
   inference; in the original generative story every pair of nodes draws a block for each node
   anew, which gives the same probabilities.

   Pairs are drawn exactly with these probabilities, without going through all pairs of nodes: a
   Poisson process proposes candidate pairs block pair by block pair, with a rate a bit larger than
   the probability of every pair, and each candidate is kept with the probability that corrects the
   difference. Everything runs on the device, in time about linear in the number of edges, of
   memberships and of pairs of blocks. The graph has no self loops nor parallel edges. In
   undirected graphs ``probs`` must be symmetric; in directed graphs ``probs[a][b]`` is for edges
   from block ``a`` to block ``b``.

   Without ``net``, nodes are the integers ``0`` to ``n - 1``, with ``memberships``, an ``n`` by
   ``k`` matrix of non-negative weights (normalized to add up to 1 in every row), or with weights
   drawn from a Dirichlet distribution of parameter ``alpha`` (a number for all blocks, or one per
   block; small values give nodes mostly in one block, large values nodes spread over all of them).
   The blocks of each node, those with weight at least ``threshold`` (by default ``1 / (k + 1)``,
   so that a node spread evenly over any number of blocks is in all of them), are stored as the qualitative node attribute ``attribute`` (unless ``None``): a tuple of blocks, or
   just the block for a node in one.

   With ``net``, the result is a randomization of ``net`` (see the module description): its nodes
   keep their communities, read from the node attribute ``attribute``, in which a node in several
   communities has a list, tuple or set of them (see :ref:`overlapping-communities`); each node
   weighs its communities equally. ``probs``, unless given (as a matrix over the communities in
   sorted order, or order of appearance if their values cannot be sorted), is the method of
   moments estimate: the matrix with which the expected weight of the edges between every pair of
   communities, every edge counting :math:`\\pi_{ia} \\pi_{jb}` for communities ``a`` and ``b``,
   equals the one of ``net`` (self loops left out, parallel edges counting). For communities that
   do not overlap this is the density of the edges between them, as in
   :func:`stochastic_block_model`; values outside 0 and 1, possible when overlaps are not explained
   by the model, are clipped. In multilayer networks each layer is randomized on its own, with the
   communities of its copies and its own probabilities (or a dictionary from layer to matrix);
   interlayer edges are kept.

   :param probs: ``k`` by ``k`` matrix of probabilities between blocks (nested lists, array or
      tensor); ``None`` with ``net``, for its own.
   :param memberships: ``n`` by ``k`` matrix of membership weights, or ``None`` for ``n`` and
      ``alpha``.
   :param n: Number of nodes, with ``alpha``.
   :param alpha: Parameter of the Dirichlet distribution of the memberships: a positive number, or
      one per block.
   :param net: A :class:`~graphoskop.Network` to randomize, not bipartite, or ``None``.
   :param attribute: Name of the node attribute holding the communities of each node: read from
      ``net``, or written to the result without ``net`` (``None`` for no attribute).
   :param threshold: Without ``net``, smallest weight of a block for a node to be listed in it;
      ``None`` for ``1 / (k + 1)``.
   :param directed: Whether the edges are directed; ``None`` for undirected, or for the direction
      of ``net``.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The mixed membership blockmodel graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If neither ``probs`` with ``memberships`` or ``n`` and ``alpha`` nor ``net``
      are given, if ``memberships``, ``n`` or ``alpha`` are given with ``net``, if ``probs`` is not
      a ``k`` by ``k`` matrix of probabilities between 0 and 1, or not symmetric for an undirected
      graph, if ``memberships`` has negative values or a row of zeros, if ``alpha`` is not positive,
      if ``net`` is bipartite, has no node attribute ``attribute`` or a node without a value, if
      ``directed`` or ``device`` differ from those of ``net``, or for dictionaries as in
      :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      B = [[0.3, 0.01], [0.01, 0.3]]
      g = gsk.mixed_membership_block_model(B, n = 200, alpha = 0.1, seed = 1)    # mostly in one block
      g = gsk.mixed_membership_block_model(B, memberships = [[1, 0], [0.5, 0.5], [0, 1]])   # "community": 0, (0, 1), 1
      gsk.modularity(g, "community")                                             # overlapping modularity

      net = gsk.load_network("edges.tsv", "nodes.tsv")     # "community": e.g. [0, 1] for a node in both
      rnd = gsk.mixed_membership_block_model(net = net, seed = 1)
   """

   def matrix(x, k):
      P = torch.as_tensor(np.asarray(x, dtype = np.float64))
      if P.shape != (k, k):
         raise ValueError(f"probs must be a {k} by {k} matrix, got shape {tuple(P.shape)}")
      if not bool(((P >= 0) & (P <= 1)).all()):
         raise ValueError("probabilities must be between 0 and 1")
      if not directed and not torch.equal(P, P.T):
         raise ValueError("probs must be symmetric for an undirected graph")
      return P

   if net is None:
      if probs is None or (memberships is None and (n is None or alpha is None)):
         raise ValueError("give probs with memberships, or with n and alpha, or net")
      directed, device = bool(directed), torch.device("cpu" if device is None else device)
      gen = _generator(seed, device)
      k = np.asarray(probs).shape[0]
      if memberships is None:
         a = np.broadcast_to(np.asarray(alpha, dtype = np.float64), (k,))
         if not (a > 0).all():
            raise ValueError("alpha must be positive")
         pi = np.random.default_rng(int(torch.randint(2 ** 62, (1,), generator = gen, device = device))).dirichlet(a, int(n))
      else:
         pi = np.asarray(memberships, dtype = np.float64)
         if pi.ndim != 2 or pi.shape[1] != k or (pi < 0).any() or not (pi.sum(1) > 0).all():
            raise ValueError(f"memberships must be an n by {k} matrix of non-negative weights, with a positive one in every row")
         pi = pi / pi.sum(1, keepdims = True)
      P = matrix(probs, k).to(device)
      node, block = np.nonzero(pi)
      s, t = _mixed_blocks(torch.as_tensor(node, device = device), torch.as_tensor(block, device = device), torch.as_tensor(pi[node, block], device = device), pi.shape[0], P, directed, gen, device)
      table = None
      if attribute is not None:
         listed = pi >= (1 / (k + 1) if threshold is None else threshold)
         pattern, which = np.unique(listed, axis = 0, return_inverse = True)   # few distinct sets of blocks
         comm = np.empty(len(pattern), dtype = object)
         comm[:] = [_membership(tuple(np.flatnonzero(r).tolist())) for r in pattern]
         comm = comm[which.ravel()]
         table = lambda perm: pd.DataFrame({attribute: pd.Categorical(comm[perm])})
      return _build(s.cpu().numpy(), t.cpu().numpy(), np.arange(pi.shape[0]), np.zeros((s.shape[0], 0), dtype = np.float32), [], None, [], directed, device, table, [] if attribute is None else [attribute])
   if memberships is not None or n is not None or alpha is not None:
      raise ValueError("with net, memberships come from its communities: leave out memberships, n and alpha")
   (mnode, mcode), labels, S, T = _communities(net, attribute, directed, device, overlap = True)
   directed = net.directed
   O = torch.bincount(mnode, minlength = int(net.node_offsets[-1]))

   def make(l, nodes, n0, count, loops, gen):
      sel = torch.isin(mnode, nodes)
      pos = torch.full((O.shape[0],), -1, dtype = torch.long, device = nodes.device)
      pos[nodes] = torch.arange(nodes.shape[0], device = nodes.device)
      present, block = torch.unique(mcode[sel], return_inverse = True)
      node, weight = pos[mnode[sel]], 1 / O[mnode[sel]].double()
      o = torch.argsort(node, stable = True)
      node, block, weight = node[o], block[o], weight[o]
      given = _layer_value(probs, None, net, l, "probs", True)
      if given is not None:
         P = matrix(given, len(labels)).to(nodes.device)[present][:, present]
      else:
         mine = torch.isin(S, nodes) & torch.isin(T, nodes)   # intra-layer edges of this layer
         P = _mixed_probs(node, block, weight, nodes.shape[0], present.shape[0], pos[S[mine]], pos[T[mine]], directed)
      return _mixed_blocks(node, block, weight, nodes.shape[0], P, directed, gen, nodes.device)

   return _randomize(net, seed, make)


def _composition(D, B, gen, device):
   """Sizes of ``B`` non-empty groups of ``D`` elements, uniform among all compositions."""
   cuts = torch.randperm(D - 1, generator = gen, device = device)[:B - 1] + 1
   edges = torch.cat([torch.zeros(1, dtype = torch.long, device = device), torch.sort(cuts).values, torch.full((1,), D, dtype = torch.long, device = device)])
   return torch.diff(edges)


def _drop(edges, P, D, directed, gen, device):
   """
   Move edges one level down the hierarchy: every edge between the groups ``a``, ``b`` of a level
   picks one of the pairs of their elements (groups of the level below, ``P``: their groups), with
   probability proportional to Exp(1) weights drawn per pair, so that the counts between the
   pairs are a uniform multiset (Dirichlet-multinomial with parameters 1).

   :param edges: Pairs of groups of the edges, ``(a, b)`` (``a <= b`` if undirected).
   :param P: Group of each of the ``D`` elements.
   :return: The pairs of elements of the edges.
   """
   t, u = torch.cartesian_prod(torch.arange(D, device = device), torch.arange(D, device = device)).T if directed else torch.triu_indices(D, D, device = device)
   a, b = P[t], P[u]
   if not directed:
      a, b = torch.minimum(a, b), torch.maximum(a, b)
   B = int(P.max()) + 1
   key = a * B + b
   order = torch.argsort(key, stable = True)
   t, u, key = t[order], u[order], key[order]
   w = torch.cumsum(-torch.log1p(-torch.rand(t.shape[0], generator = gen, device = device, dtype = torch.float64)), 0)   # cumulative Exp(1) weights
   start = torch.searchsorted(key, torch.arange(B * B, device = device))
   end = torch.searchsorted(key, torch.arange(B * B, device = device), right = True)
   k = edges[0] * B + edges[1]
   lo = torch.where(start[k] > 0, w[(start[k] - 1).clamp(min = 0)], torch.zeros_like(w[:1]))
   hi = w[end[k] - 1]
   x = torch.searchsorted(w, lo + (hi - lo) * torch.rand(k.shape[0], generator = gen, device = device, dtype = torch.float64), right = True)
   x = torch.minimum(torch.maximum(x, start[k]), end[k] - 1)   # inside the segment despite rounding
   return t[x], u[x]


def nested_block_model(n = None, m = None, levels = None, degrees = None, net = None, attribute = "community", directed = None, rounds = 10, drop_self_loops = True, drop_parallel = True, max_blocks = 4096, seed = None, device = None):
   """
   Nested degree-corrected stochastic blockmodel (Peixoto, 2014, 2017): a degree-corrected
   blockmodel whose numbers of edges between groups are themselves generated by a blockmodel of
   the groups, and so on up a hierarchy of groups of groups, ending in a single group. It is the
   most general of the blockmodels here: communities, core-periphery or bipartite-like structures
   at several scales, and heterogeneous degrees, all generated from the numbers of groups of every
   level alone. It is also the model that :func:`fit_nested_block_model` fits, and that the
   functions of :doc:`error_correction` use to reconstruct networks.

   Without ``net``, the network is drawn top-down from the model's priors:

   * the partition of each level (the ``n`` nodes into ``levels[0]`` groups, those into
     ``levels[1]`` groups, ...) is uniform given its group sizes, which are uniform among all
     compositions (every group non-empty);
   * the ``m`` edges are spread over the pairs of groups of the top level, and those of every pair
     of groups over the pairs of their subgroups one level down, down to the pairs of bottom
     groups, each time uniformly among all multisets (a Dirichlet-multinomial with all
     parameters 1, drawn by giving each edge a pair with probability proportional to Exp(1)
     weights of the pairs);
   * the edges of each bottom group are spread over its nodes, each half-edge picking a node in
     proportion to ``degrees`` (relative degrees within the group) or, by default, to Exp(1)
     weights, which makes the degree sequence of each group uniform among all those with its
     number of half-edges (the "uniform" degree prior of the microcanonical model; the description
     length of :func:`fit_nested_block_model` uses the more permissive prior of degree
     histograms).

   The edges are then wired as in :func:`degree_corrected_block_model` with ``exact = True``:
   the numbers of edges between bottom groups are exact, self loops and parallel edges redrawn
   for up to ``rounds`` rounds and those left dropped (unless asked otherwise). The model is a
   multigraph, and its priors are heavy-tailed: they often pile the edges of the network on a few
   pairs of groups, beyond what their pairs of nodes can hold, so that dropping can remove a good
   share of the ``m`` edges (a quarter, for 2,000 nodes, 10,000 edges and levels ``[32, 8, 2]``);
   ``drop_parallel = False`` keeps the multigraph, with exactly ``m`` edges if self loops are kept
   too. Directed networks spread the edges over ordered pairs of groups, with out- and
   in-degrees. Everything runs on the device, in time and memory linear in the number of edges and
   of pairs of groups of every level. Nodes are the integers ``0`` to ``n - 1``, bottom group by bottom group, and the
   group of every node at every level is stored as numeric node attributes: ``attribute`` for the
   bottom level, ``attribute + "_1"``, ``attribute + "_2"``, ... for the levels above.

   With ``net``, the model is first fitted to ``net`` by :func:`fit_nested_block_model`
   (undirected networks only, not multilayer nor bipartite; parallel edges count once, self loops
   are ignored), and the result is a randomization of ``net`` with the fitted hierarchy: it keeps
   the nodes, the exact numbers of edges between bottom groups and, on average, the degrees, and
   gets the groups of all levels as node attributes. The fit runs on the CPU (Numba), the wiring on
   the device.

   :param n: Number of nodes; ``None`` with ``net``.
   :param m: Number of edges; ``None`` with ``net``.
   :param levels: Number of groups of every level, bottom first, non-increasing, at most ``n``
      (e.g. ``[16, 4]``: 16 groups of nodes, in 4 groups); ``None`` with ``net``.
   :param degrees: Relative degree of each node, bottom group by bottom group (a pair of out- and
      in-degrees if directed), or ``None`` for the uniform prior; ``None`` with ``net``.
   :param net: A :class:`~graphoskop.Network` to fit and randomize, or ``None``.
   :param attribute: Name of the node attribute of the bottom groups (``None`` for no
      attributes).
   :param directed: Whether the edges are directed; ``None`` for undirected. Must be undirected
      with ``net``.
   :param rounds: Maximum number of rounds of redrawing self loops and parallel edges (0 for none).
   :param drop_self_loops: Whether to drop the self loops left.
   :param drop_parallel: Whether to drop the parallel edges left (one edge of each group stays).
   :param max_blocks: Maximum number of bottom groups of the fit (with ``net``).
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The nested blockmodel graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If neither ``n``, ``m`` and ``levels`` nor ``net`` are given, if they are
      given together, if ``levels`` is not a non-increasing list of positive numbers at most ``n``,
      if ``degrees`` does not have one non-negative value per node, or if ``net`` is directed,
      multilayer or bipartite.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.nested_block_model(1000, 5000, [32, 8, 2], seed = 1)   # three levels of groups
      g.get_node_attribute("community_1")                           # groups of the middle level

      net = gsk.load_network("edges.tsv")
      rnd = gsk.nested_block_model(net = net, seed = 1)              # fitted hierarchy of net
   """
   from .error_correction import _fit, _pairs

   if net is not None:
      if n is not None or m is not None or levels is not None or degrees is not None:
         raise ValueError("with net, n, m, levels and degrees come from the fit: leave them out")
      if net.bipartite_labels is not None:
         raise ValueError("nested_block_model does not randomize bipartite networks")
      N, s0, t0, _ = _pairs(net, "nested_block_model")
      key = np.unique(s0 * N + t0)
      eu, ev = (key // N).astype(np.int64), (key % N).astype(np.int64)
      b0, par, Bs, _ = _fit(N, eu, ev, max_blocks = max_blocks, seed = seed)

      def make(l, nodes, n0, count, loops, gen):
         dev = nodes.device
         code = torch.as_tensor(b0, device = dev)[nodes]
         order = torch.argsort(code, stable = True)
         sizes = torch.bincount(code, minlength = int(Bs[0]))
         pos = torch.empty(N, dtype = torch.long, device = dev)
         pos[nodes[order]] = torch.arange(N, device = dev)
         u, v = pos[torch.as_tensor(eu, device = dev)], pos[torch.as_tensor(ev, device = dev)]
         w = torch.bincount(torch.cat([u, v]), minlength = N).double()
         B = int(Bs[0])
         bu, bv = code[order][u], code[order][v]
         E = torch.bincount(bu * B + bv, minlength = B * B).view(B, B).double()
         E = E + E.T - torch.diag(torch.diag(E))
         s, t = _dc_blocks(sizes, w, w, E, False, True, rounds, drop_self_loops, drop_parallel, gen, dev)
         return order[s], order[t]

      new = _randomize(net, seed, make)
      if attribute is not None:
         pos = torch.arange(N)
         g, at = b0.copy(), 0
         for l in range(len(Bs)):
            if l > 0:
               g = par[at:at + Bs[l - 1]][g]
               at += Bs[l - 1]
            _with_groups(net, new, attribute if l == 0 else f"{attribute}_{l}", [(pos, g)])
      return new

   if n is None or m is None or levels is None:
      raise ValueError("give n, m and levels, or net")
   directed, device = bool(directed), torch.device("cpu" if device is None else device)
   n, m, Bs = int(n), int(m), [int(x) for x in levels]
   if n < 1 or m < 0 or not Bs or Bs[0] > n or any(x < 1 for x in Bs) or any(Bs[l] > Bs[l - 1] for l in range(1, len(Bs))):
      raise ValueError(f"levels must be a non-increasing list of positive numbers of groups, at most n = {n}, got {levels}")
   gen = _generator(seed, device)
   sizes = _composition(n, Bs[0], gen, device)
   P = [torch.repeat_interleave(torch.arange(Bs[0], device = device), sizes)]   # bottom groups, node by node
   for l in range(1, len(Bs)):
      sz = _composition(Bs[l - 1], Bs[l], gen, device)
      P.append(torch.repeat_interleave(torch.arange(Bs[l], device = device), sz)[torch.randperm(Bs[l - 1], generator = gen, device = device)])
   e = torch.zeros((2, m), dtype = torch.long, device = device)   # all edges in the single top group
   for l in range(len(Bs), 0, -1):
      Pl = P[l] if l < len(Bs) else torch.zeros(Bs[-1], dtype = torch.long, device = device)
      e = torch.stack(_drop(e, Pl, Bs[l - 1], directed, gen, device))
   B = Bs[0]
   omega = torch.bincount(e[0] * B + e[1], minlength = B * B).view(B, B).double()
   if not directed:
      omega = omega + omega.T - torch.diag(torch.diag(omega))
   if degrees is None:
      d = [-torch.log1p(-torch.rand(n, generator = gen, device = device, dtype = torch.float64)) for _ in range(2 if directed else 1)]
   else:
      d = [torch.as_tensor(list(x), dtype = torch.float64, device = device) for x in (degrees if directed else [degrees])]
      if any(x.shape != (n,) or bool((x < 0).any()) for x in d):
         raise ValueError(f"degrees must have one non-negative value per node ({n})")
   s, t = _dc_blocks(sizes, d[0], d[-1], omega, directed, True, rounds, drop_self_loops, drop_parallel, gen, device)
   names, cols, g = [], [], P[0]
   for l in range(len(Bs)):
      if l > 0:
         g = P[l][g]
      names.append(attribute if l == 0 else f"{attribute}_{l}")
      cols.append(g.cpu().numpy().astype(np.float32))
   table = None if attribute is None else (lambda perm: np.stack(cols, 1)[perm])
   return _build(s.cpu().numpy(), t.cpu().numpy(), np.arange(n), np.zeros((s.shape[0], 0), dtype = np.float32), [], table, [] if attribute is None else names, directed, device)



# ------------------------------------------------------------------------------------------------
# Community benchmarks: Girvan-Newman, LFR
# ------------------------------------------------------------------------------------------------


def _gn_probs(sizes, d, mu):
   """
   Probabilities of a Girvan-Newman benchmark: see :func:`girvan_newman_graph`.

   :param sizes: Community sizes.
   :param d: Average degree (out-degree if directed).
   :param mu: Mixing parameter.
   :return: ``(p_in, p_out)``.
   :raises ValueError: If ``mu`` is not between 0 and 1, or a probability would exceed 1.
   """
   if not 0 <= mu <= 1:
      raise ValueError(f"mu must be between 0 and 1, got {mu}")
   sizes = np.asarray(sizes, dtype = np.float64)
   n = sizes.sum()
   ratio = lambda a, b: a / b if b else (math.inf if a else 0.)
   p_in = ratio(n * d * (1 - mu), (sizes * (sizes - 1)).sum())
   p_out = ratio(n * d * mu, n * n - (sizes ** 2).sum())
   if p_in > 1 or p_out > 1:
      raise ValueError(f"average degree {d:g} with mu = {mu:g} needs p_in = {p_in:.3g}, p_out = {p_out:.3g}: communities of {sizes.min():g}{'' if sizes.min() == sizes.max() else f' to {sizes.max():g}'} nodes cannot hold it")
   return p_in, p_out


def girvan_newman_graph(mu = None, l = None, k = None, degree = None, net = None, directed = None, attribute = "community", seed = None, device = None):
   """
   Girvan-Newman benchmark (Girvan and Newman, 2002) for community discovery: ``l`` communities
   of ``k`` nodes, and every node expects ``degree`` edges, a fraction ``mu`` of them (the mixing
   parameter) to nodes of other communities. Pairs of nodes are edges independently, with
   probability :math:`p_{in}` within communities and :math:`p_{out}` between them, so this is a
   :func:`planted_partition_graph` with equal communities, parametrized by how hard they are to
   find rather than by densities; it is also a caveman graph (see :func:`caveman_graph`) whose
   caves are thinned and mixed. The classic benchmark is the default: 128 nodes in 4 communities of
   32, average degree 16 (expected external degree :math:`z_{out} = 16 \\mu`). Communities are
   found easily for small ``mu``, and become undetectable as ``mu`` approaches
   :math:`1 - 1 / l`, where nodes have as many edges to each other community as to their own.

   For ``n`` nodes in communities of sizes :math:`s_c` and average degree :math:`d` (out-degree if
   directed), :math:`p_{in} = n d (1 - \\mu) / \\sum_c s_c (s_c - 1)` and
   :math:`p_{out} = n d \\mu / (n^2 - \\sum_c s_c^2)`, so that the average degree is ``degree`` and
   the expected fraction of edges between communities is ``mu``. The graph has no self loops nor
   parallel edges, and is generated as :func:`planted_partition_graph`, in time about linear in
   the number of edges.

   Without ``net``, nodes are the integers ``0`` to ``n - 1``, community by community. With
   ``net``, the result is a randomization of ``net`` (see the module description): its nodes are
   assigned at random to ``l`` communities, whose sizes differ by at most one (``l`` given, or
   ``n / k`` rounded if ``k`` is given), with the average degree of ``net``. Without ``l`` and
   ``k``, the number of communities is that of the node attribute ``attribute`` of ``net``, and
   without ``mu`` the mixing parameter is the fraction of the edges of ``net`` between its
   communities (self loops left out); for communities of ``net`` that are kept as they are, use
   :func:`planted_partition_graph`. In multilayer networks each layer is randomized on its own,
   with its own communities (numbered across layers, so that no two layers share one), average
   degree and parameters (``mu``, ``l``, ``k`` and ``degree`` can be dictionaries from layer to
   value); interlayer edges are kept.

   In both cases the community of every node is stored as the numeric node attribute
   ``attribute`` (overwriting that of ``net``), unless ``None``: the planted partition, the
   ground truth with which to compare the communities found.

   :param mu: Mixing parameter, between 0 and 1; ``None`` with ``net``, for its own.
   :param l: Number of communities; ``None`` for 4 without ``net``.
   :param k: Number of nodes of each community; ``None`` for 32 without ``net``.
   :param degree: Average degree (out-degree if directed); ``None`` for 16 without ``net``, or
      for the average degree of ``net``.
   :param net: A :class:`~graphoskop.Network` to randomize, not bipartite, or ``None``.
   :param directed: Whether the edges are directed (ordered pairs); ``None`` for undirected, or
      for the direction of ``net``.
   :param attribute: Name of the node attribute holding the community of each node: written to
      the result, and read from ``net`` when ``l``, ``k`` or ``mu`` are missing (``None`` for no
      attribute).
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The benchmark graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If ``mu`` is missing without ``net``, or with a ``net`` without node
      attribute ``attribute``, likewise for the number of communities, if ``mu`` is not between 0
      and 1, if the degree is too high for the communities (a probability above 1), if both ``l``
      and ``k`` are given with ``net``, if ``net`` is bipartite or a node has no community, if
      ``directed`` or ``device`` differ from those of ``net``, or for dictionaries as in
      :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.girvan_newman_graph(0.2, seed = 1)                          # 4 x 32 nodes, z_out = 3.2
      gsk.modularity(g, "community")                                      # quality of the planted partition
      g = gsk.girvan_newman_graph(0.4, l = 10, k = 100, degree = 20)      # larger benchmark

      net = gsk.load_network("edges.tsv", "nodes.tsv")                    # node attribute "community"
      rnd = gsk.girvan_newman_graph(net = net, seed = 1)                  # as many communities, same mu and degree
      rnd = gsk.girvan_newman_graph(0.5, net = net, l = 8)                # harder, with 8 communities
   """
   if net is None:
      if mu is None:
         raise ValueError("give mu, or net")
      l, k, degree = 4 if l is None else int(l), 32 if k is None else int(k), 16 if degree is None else degree
      directed, device = bool(directed), torch.device("cpu" if device is None else device)
      p_in, p_out = _gn_probs([k] * l, degree, mu)
      sz = torch.full((l,), k, dtype = torch.long, device = device)
      s, t = _planted(sz, p_in, p_out, directed, _generator(seed, device), device)
      comm = np.repeat(np.arange(l), k).astype(np.float32)
      table = None if attribute is None else lambda perm: comm[perm][:, None]
      return _build(s.cpu().numpy(), t.cpu().numpy(), np.arange(l * k), np.zeros((s.shape[0], 0), dtype = np.float32), [], table, [] if attribute is None else [attribute], directed, device)
   if l is not None and k is not None:
      raise ValueError("with net, give l or k, not both: the number of nodes fixes the other")
   _check_net(net, None, directed, device)
   known = attribute is not None and (attribute in net.node_attributes or attribute in net.qualitative_node_attributes)
   code, S, T = None, None, None
   if known and (mu is None or (l is None and k is None)):
      code, _, S, T = _communities(net, attribute, directed, device)
   elif net.bipartite_labels is not None:
      raise ValueError("bipartite networks are not supported")
   groups = []

   def make(j, nodes, n0, count, loops, gen):
      n = nodes.shape[0]
      if code is not None:
         mine = torch.isin(S, nodes) & torch.isin(T, nodes)   # intra-layer edges of this layer
      q = _layer_value(mu, None, net, j, "mu", True)
      if q is None:
         if code is None:
            raise ValueError(f"give mu, or a net with node attribute {attribute!r}")
         q = float((code[S[mine]] != code[T[mine]]).double().mean()) if bool(mine.any()) else 0.
      c, size = _layer_value(l, None, net, j, "l", False), _layer_value(k, None, net, j, "k", True)
      if c is None and size is None:
         if code is None:
            raise ValueError(f"give l or k, or a net with node attribute {attribute!r}")
         c = int(torch.unique(code[nodes]).shape[0])
      sizes = _cave_sizes(n, 0, c, size)
      d = _layer_value(degree, count * (1 if net.directed else 2) / max(n, 1), net, j, "degree", True)
      p_in, p_out = _gn_probs(sizes, d, q)
      perm = torch.randperm(n, generator = gen, device = nodes.device)   # random communities
      off = max((int(g[-1]) + 1 for _, g in groups if len(g)), default = 0)   # numbered across layers
      groups.append((nodes[perm], np.repeat(np.arange(off, off + len(sizes)), sizes)))
      s, t = _planted(torch.tensor(sizes, dtype = torch.long, device = nodes.device), p_in, p_out, net.directed, gen, nodes.device)
      return perm[s], perm[t]

   return _with_groups(net, _randomize(net, seed, make), attribute, groups)



def _power_law(count, t, lo, hi, gen, device):
   """
   Integers from a power law: values of a continuous power law of exponent ``t`` on
   ``[lo, hi + 1)``, rounded down (so between ``floor(lo)`` and ``hi``).

   :param count: Number of values.
   :param t: Exponent (:math:`p(x) \\propto x^{-t}`).
   :param lo: Lower bound, a positive real number.
   :param hi: Largest value.
   :param gen: Random generator.
   :param device: Device.
   :return: int64 tensor.
   """
   u = torch.rand(count, generator = gen, device = device, dtype = torch.float64)
   a, b = float(lo), float(hi) + 1
   if abs(t - 1) < 1e-12:
      x = a * (b / a) ** u
   else:
      e = 1 - t
      x = (a ** e + u * (b ** e - a ** e)) ** (1 / e)
   return x.floor().long().clamp(max = int(hi))


def _power_law_bound(mean, t, hi, name):
   """
   Lower bound of the power law of :func:`_power_law` whose values have a given mean.

   :param mean: Mean of the values.
   :param t: Exponent.
   :param hi: Largest value.
   :param name: Name of the mean, for error messages.
   :return: The lower bound.
   :raises ValueError: If no lower bound of at least 1 gives that mean.
   """
   F = np.log if abs(t - 1) < 1e-12 else (lambda x: x ** (1 - t))

   def avg(a):
      j = np.arange(math.floor(a), hi + 1, dtype = np.float64)
      w = F(j + 1) - F(np.maximum(j, a))   # mass of each value, up to a common factor
      return (j * w).sum() / w.sum()

   if not avg(1) <= mean <= hi:
      raise ValueError(f"{name} must be between {avg(1):.3g} (exponent {t:g}, values from 1) and {hi}, got {mean}")
   a, b = 1., float(hi)
   for _ in range(60):   # bisection: the mean grows with the lower bound
      a, b = ((a + b) / 2, b) if avg((a + b) / 2) < mean else (a, (a + b) / 2)
   return a


def _decrement(d, group, excess, gen):
   """
   Lower by one, in place, the values of ``excess[g]`` random elements of each group ``g`` (an
   element can be lowered several times), among positive values.

   :param d: int64 tensor of values.
   :param group: int64 tensor, group of every element.
   :param excess: int64 tensor, how much to remove from every group.
   :param gen: Random generator.
   """
   excess = excess.clone()
   while bool((excess > 0).any()):
      idx = torch.nonzero((d > 0) & (excess[group] > 0)).squeeze(1)
      if idx.shape[0] == 0:
         return
      idx = idx[torch.argsort(group[idx].double() + torch.rand(idx.shape[0], generator = gen, device = d.device, dtype = torch.float64))]
      g = group[idx]
      dec = idx[torch.arange(idx.shape[0], device = d.device) - torch.searchsorted(g, g) < excess[g]]
      d[dec] -= 1
      excess -= torch.bincount(group[dec], minlength = excess.shape[0])


def _sharing(node, comm, n, C):
   """
   Test of whether pairs of nodes share a community.

   :param node: int64 tensor, node of every membership.
   :param comm: int64 tensor, community of every membership.
   :param n: Number of nodes.
   :param C: Number of communities.
   :return: Function of two node tensors giving a boolean tensor.
   """
   keys = torch.sort(node * C + comm).values   # by node, then community
   count = torch.bincount(node, minlength = n)
   first = torch.cumsum(count, 0) - count

   def share(a, b):
      e = torch.repeat_interleave(torch.arange(a.shape[0], device = a.device), count[a])
      q = b[e] * C + keys[_expand(first[a], count[a])] % C
      hit = keys[torch.searchsorted(keys, q).clamp(max = keys.shape[0] - 1)] == q
      return torch.zeros(a.shape[0], dtype = torch.bool, device = a.device).index_fill_(0, e[hit], True)

   return share


def _lfr_assign(node, need, sizes, gen, rounds = 200):
   """
   Communities of the memberships of an LFR benchmark: a random matching of memberships and
   places in communities (a bipartite configuration model), then rounds of swaps of the
   memberships that do not fit, i.e. that need more neighbors than their community has, or put a
   node twice in a community, with random memberships of communities large enough for them.

   :param node: int64 tensor, node of every membership.
   :param need: int64 tensor, internal degree of every membership.
   :param sizes: int64 tensor, size of every community, in increasing order, adding up to the
      number of memberships.
   :param gen: Random generator.
   :param rounds: Maximum number of rounds.
   :return: int64 tensor, community of every membership (some may still not fit).
   """
   dev, M, C = node.device, node.shape[0], sizes.shape[0]
   comm = torch.repeat_interleave(torch.arange(C, device = dev), sizes)[torch.randperm(M, generator = gen, device = dev)]
   idle = 0
   for _ in range(rounds):
      key = node * C + comm
      sk, inv = torch.unique(key, return_inverse = True)
      first = torch.full((sk.shape[0],), M, device = dev).scatter_reduce(0, inv, torch.arange(M, device = dev), "amin")
      wrong = (need > sizes[comm] - 1) | torch.ones(M, dtype = torch.bool, device = dev).index_fill_(0, first, False)
      V = torch.nonzero(wrong).squeeze(1)
      if V.shape[0] == 0 or idle >= 10:
         break
      o = torch.argsort(comm)   # memberships by community, so by community size
      lo = torch.searchsorted(sizes[comm[o]], need[V] + 1)   # first one in a large enough community
      V, lo = V[lo < M], lo[lo < M]
      P = o[lo + (torch.rand(V.shape[0], generator = gen, device = dev, dtype = torch.float64) * (M - lo)).long().clamp(max = M - 1)]
      a, b = comm[V], comm[P]
      k1, k2 = node[V] * C + b, node[P] * C + a
      ok = (a != b) & ~wrong[P] & (need[P] <= sizes[a] - 1) & ~torch.isin(k1, sk) & ~torch.isin(k2, sk)
      _, i, used = torch.unique(P, return_inverse = True, return_counts = True)
      ok &= used[i] == 1
      both = torch.cat([k1, k2])
      _, i, dup = torch.unique(both, return_inverse = True, return_counts = True)
      ok &= (dup[i[:V.shape[0]]] == 1) & (dup[i[V.shape[0]:]] == 1)
      comm[V[ok]], comm[P[ok]] = b[ok], a[ok]
      idle = 0 if bool(ok.any()) else idle + 1
   return comm


def _lfr_edges(kin, kout, node, comm, C, mu, directed, gen):
   """
   Edges of an LFR benchmark with given degrees and communities: each node keeps a fraction
   ``1 - mu`` of its degree (in- and out-degree if directed) inside its communities, split evenly
   among them and at most the size of the community minus one, wired by a configuration model
   within every community; the rest is wired by a configuration model between nodes that share no
   community. Stubs are moved from inside to outside where needed for the counts to match (even in
   every community), or between inside and outside (as many in- as out-stubs in every community,
   if directed: half of the gap by moving stubs of the larger side outside, the rest by moving
   stubs of the other side inside, or outside if there is no room). Self loops, parallel edges
   and edges within a community left by the repairs are dropped, as are edges made in two
   communities by nodes that share both.

   :param kin: int64 tensor, in-degree of every node (the degree if undirected).
   :param kout: int64 tensor, out-degree of every node (the degree if undirected).
   :param node: int64 tensor, node of every membership.
   :param comm: int64 tensor, community of every membership.
   :param C: Number of communities.
   :param mu: Mixing parameter.
   :param directed: Whether the edges are directed.
   :param gen: Random generator.
   :return: Sources, targets, and whether each edge is internal (within a community).
   """
   dev, n = kin.device, kin.shape[0]
   o = torch.argsort(comm * n + node)   # memberships by community, then node
   node, comm = node[o], comm[o]
   M = node.shape[0]
   size, O = torch.bincount(comm, minlength = C), torch.bincount(node, minlength = n)
   by_node = torch.argsort(node, stable = True)
   rank = torch.empty(M, dtype = torch.long, device = dev)
   rank[by_node] = torch.arange(M, device = dev) - (torch.cumsum(O, 0) - O)[node[by_node]]

   def split(k):   # internal degree of every membership
      kint = torch.round((1 - mu) * k.double()).long()
      d = kint[node] // O[node] + (rank < kint[node] % O[node]).long()
      return torch.minimum(d, size[comm] - 1)

   total = lambda d: torch.zeros(C, dtype = torch.long, device = dev).index_add_(0, comm, d)
   if directed:   # as many in- as out-stubs in every community: half of the gap closed on each side
      di, do = split(kin), split(kout)
      for x, y, ky in ((do, di, kin), (di, do, kout)):
         gap = (total(x) - total(y)).clamp(min = 0)
         cap = torch.maximum(y, torch.minimum(size[comm] - 1, ky[node] // O[node]))
         room = cap - y
         _decrement(room, comm, gap - gap // 2, gen)   # y up, where it has external stubs left
         y.copy_(cap - room)
         _decrement(x, comm, (total(x) - total(y)).clamp(min = 0), gen)   # x down for the rest
   else:
      di = do = split(kin)
      _decrement(di, comm, torch.zeros(C, dtype = torch.long, device = dev).index_add_(0, comm, di) % 2, gen)
   s, t = _stub_match(do, di, directed, 100, True, True, gen, group = comm)   # within communities
   s, t = node[s], node[t]
   key = torch.unique(s * n + t if directed else torch.minimum(s, t) * n + torch.maximum(s, t))   # one edge per pair
   s, t = key // n, key % n
   eo = kout - torch.zeros(n, dtype = torch.long, device = dev).index_add_(0, node, do)
   ei = kin - torch.zeros(n, dtype = torch.long, device = dev).index_add_(0, node, di)
   x, y = _stub_match(eo, ei if directed else eo, directed, 100, True, True, gen, forbid = _sharing(node, comm, n, C))   # between
   internal = torch.cat([torch.ones_like(s, dtype = torch.bool), torch.zeros_like(x, dtype = torch.bool)])
   return torch.cat([s, x]), torch.cat([t, y]), internal


def _lfr_weights(s, t, internal, target_in, target_out, tol = 1e-4, iterations = 2000):
   """
   Weights of an LFR benchmark: ``w_ij = x_i x_j`` separately on internal and external edges,
   with the factors ``x`` that give every node its target internal and external strength (sum of
   the weights of its edges, in either direction), found by iterating
   :math:`x_i \\leftarrow \\sqrt{x_i T_i / \\sum_j x_j}`; when the targets cannot all be met, the
   iteration settles close to them.

   :param s: int64 tensor, sources.
   :param t: int64 tensor, targets.
   :param internal: Boolean tensor, whether each edge is internal.
   :param target_in: float64 tensor, target internal strength of every node.
   :param target_out: float64 tensor, target external strength of every node.
   :param tol: Relative change of the factors at which to stop.
   :param iterations: Maximum number of iterations.
   :return: float64 tensor of weights.
   """
   w = torch.zeros(s.shape[0], dtype = torch.float64, device = s.device)
   for mask, target in ((internal, target_in), (~internal, target_out)):
      a, b = s[mask], t[mask]
      x = torch.ones_like(target)
      for _ in range(iterations):
         tot = torch.zeros_like(x).index_add_(0, a, x[b]).index_add_(0, b, x[a])
         new = torch.where(tot > 0, torch.sqrt(x * target / tot.clamp(min = 1e-300)), 0)
         done = bool(((new - x).abs() <= tol * new.abs().clamp(min = 1e-12)).all())
         x = new
         if done:
            break
      w[mask] = x[a] * x[b]
   return w


def lfr_benchmark_graph(n = None, k = None, maxk = None, mu = None, t1 = 2, t2 = 1, minc = None, maxc = None, on = 0, om = 2, muw = None, beta = None, net = None, attribute = "community", weight = "weight", directed = None, seed = None, device = None):
   """
   LFR benchmark (Lancichinetti, Fortunato and Radicchi, 2008) for community discovery, with the
   variants of Lancichinetti and Fortunato (2009): directed, weighted, and with overlapping
   communities. Unlike :func:`girvan_newman_graph`, degrees and community sizes are broad, as in
   real networks: degrees follow a power law of exponent ``t1`` with average ``k`` and maximum
   ``maxk``, community sizes a power law of exponent ``t2`` between ``minc`` and ``maxc``, and every
   node has a fraction ``mu`` (the mixing parameter) of its edges to nodes outside its communities.

   The graph is built as in the original: nodes are placed in communities by a random matching,
   then swaps until every node fits, i.e. its internal degree :math:`(1 - \\mu) k_i` is smaller
   than the size of its community; each community is wired as a configuration model of the
   internal degrees, and the external degrees as a configuration model between nodes that share no
   community. A few hundred rounds of parallel repairs (see :func:`configuration_model`) remove
   self loops, parallel edges and external edges within communities; the few left are dropped, so
   some degrees can be a little lower than drawn, and, when small communities are almost cliques,
   ``mu`` a little higher (e.g. 0.303 for 0.3 with communities of 20 nodes and average degree
   20). Internal degrees larger than the community
   allows (a hub that fits in no community) are cut to fit, the rest going outside. Everything
   runs on the device, in time about linear in the number of edges.

   Variants:

   * directed (``directed = True``): in-degrees follow the power law and out-degrees are as equal
     as possible, as in the tests of Lancichinetti and Fortunato (2009); ``mu`` applies to both,
     and ``k`` and ``maxk`` are the average and largest in-degree;
   * weighted (``muw`` given): node ``i`` has strength :math:`s_i = k_i^\\beta` (total degree if
     directed; ``beta`` 1.5 by default), a fraction ``muw`` of it on external edges, and weights
     are set to give every node its internal and external strengths as closely as possible (by
     iterative scaling, instead of the greedy minimization of the original). They are stored as the
     edge attribute ``weight``;
   * overlapping communities (``on`` above 0): ``on`` random nodes belong to ``om`` communities each,
     their internal degree split evenly among them; sizes add up to ``n + on (om - 1)``
     memberships. Communities are then a qualitative node attribute of tuples (see
     :ref:`overlapping-communities`) and :func:`~graphoskop.modularity` is the overlapping one.

   Without ``net``, nodes are the integers ``0`` to ``n - 1``, and the community of each node is
   stored as the node attribute ``attribute`` (unless ``None``): numeric, communities numbered
   from the smallest, or tuples of them with overlaps. ``minc`` and ``maxc`` default to the smallest
   and largest degree, as in the original program.

   With ``net``, the result is a randomization of ``net`` (see the module description) with its
   degrees (in and out if directed) and its communities, read from the node attribute
   ``attribute`` (a list, tuple or set of communities for nodes in several), which it keeps:
   instead of drawing them from power laws, the benchmark uses those of ``net``, and ``mu`` is
   the fraction of its edges (self loops left out) between nodes that share no community, unless
   given. If ``net`` has the edge attribute ``weight``, the result is weighted: strengths are
   those of ``net`` (or :math:`k_i^\\beta` if ``beta`` is given) and ``muw`` the fraction of its
   weight on external edges, unless given; otherwise edge attributes are drawn from the old edges,
   as in the other generators. In multilayer networks each layer is randomized on its own (``mu``,
   ``muw`` and ``beta`` can be dictionaries from layer to value); interlayer edges are kept.

   :param n: Number of nodes; ``None`` with ``net``.
   :param k: Average degree (in-degree if directed); ``None`` with ``net``.
   :param maxk: Largest degree (in-degree if directed); ``None`` with ``net``.
   :param mu: Mixing parameter, between 0 and 1; ``None`` with ``net``, for its own.
   :param t1: Exponent of the degree power law.
   :param t2: Exponent of the community size power law.
   :param minc: Smallest community size; ``None`` for the smallest degree.
   :param maxc: Largest community size; ``None`` for ``maxk``.
   :param on: Number of nodes in several communities.
   :param om: Number of communities of each of them.
   :param muw: Mixing parameter of the weights, between 0 and 1, for a weighted graph; ``None``
      for an unweighted one, or, with ``net``, for its own.
   :param beta: Exponent of the strengths, :math:`s_i = k_i^\\beta`; ``None`` for 1.5, or, with
      ``net``, for its strengths.
   :param net: A :class:`~graphoskop.Network` to randomize, not bipartite, or ``None``.
   :param attribute: Name of the node attribute holding the communities: written to the result
      without ``net`` (``None`` for none), read from ``net``.
   :param weight: Name of the edge attribute of the weights: written to the result if weighted,
      and read from ``net`` (``None``: unweighted with ``net``).
   :param directed: Whether the edges are directed; ``None`` for undirected, or for the direction
      of ``net``.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param device: Device of the result, e.g. ``"cuda"``; ``None`` for the CPU, or for the device
      of ``net``.
   :return: The benchmark graph, a new :class:`~graphoskop.Network`.
   :raises ValueError: If ``n``, ``k``, ``maxk`` and ``mu`` are missing without ``net``, or the
      first three are given with it, if ``k`` cannot be the average of the power law up to
      ``maxk``, if the community sizes cannot hold the memberships (``minc`` above ``maxc`` or above
      the number of memberships, ``om`` above the number of communities, ``on`` above ``n``), if
      ``mu`` or ``muw`` are not between 0 and 1, if ``net`` is bipartite, has no node attribute
      ``attribute`` or a node without a value, if ``directed`` or ``device`` differ from those of
      ``net``, or for dictionaries as in :func:`gnm_random_graph`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      g = gsk.lfr_benchmark_graph(1000, 20, 50, 0.3, minc = 20, maxc = 100, seed = 1)
      gsk.modularity(g, "community")                                   # quality of the planted communities
      g = gsk.lfr_benchmark_graph(1000, 20, 50, 0.3, directed = True)   # power law in-degrees
      g = gsk.lfr_benchmark_graph(1000, 20, 50, 0.3, muw = 0.2)         # weighted, beta = 1.5
      g = gsk.lfr_benchmark_graph(1000, 20, 50, 0.3, on = 100, om = 3)  # 100 nodes in 3 communities each

      net = gsk.load_network("edges.tsv", "nodes.tsv")                 # node attribute "community"
      rnd = gsk.lfr_benchmark_graph(net = net, seed = 1)               # degrees, communities and mu of net
      rnd = gsk.lfr_benchmark_graph(net = net, mu = 0.6)               # same, with mixed communities
   """
   for q in [x for v in (mu, muw) for x in (v.values() if isinstance(v, dict) else [v])]:
      if q is not None and not 0 <= q <= 1:
         raise ValueError(f"mixing parameters must be between 0 and 1, got {q}")
   if net is None:
      if n is None or k is None or maxk is None or mu is None:
         raise ValueError("give n, k, maxk and mu, or net")
      directed, device = bool(directed), torch.device("cpu" if device is None else device)
      gen = _generator(seed, device)
      n, maxk = int(n), int(maxk)
      kin = _power_law(n, t1, _power_law_bound(k, t1, maxk, "k"), maxk, gen, device)
      if directed:   # out-degrees as equal as possible
         m = int(kin.sum())
         kout = (m // n + (torch.arange(n, device = device) < m % n).long())[torch.randperm(n, generator = gen, device = device)]
      else:
         if int(kin.sum()) % 2:   # even sum: one more edge end at a node below the maximum
            kin[torch.nonzero(kin < maxk).squeeze(1)[0]] += 1
         kout = kin
      minc = int(kin.min()) if minc is None else int(minc)
      maxc = maxk if maxc is None else int(maxc)
      on = int(on)
      if not 0 <= on <= n:
         raise ValueError(f"on must be between 0 and n = {n}, got {on}")
      O = torch.ones(n, dtype = torch.long, device = device)
      O[torch.randperm(n, generator = gen, device = device)[:on]] = om
      M = int(O.sum())
      if not 1 <= minc <= min(maxc, M):
         raise ValueError(f"community sizes must be at least 1, minc at most maxc and at most the {M} memberships: got minc = {minc}, maxc = {maxc}")
      sizes = kin[:0]
      while int(sizes.sum()) < M:
         sizes = torch.cat([sizes, _power_law(max(16, 2 * M // minc), t2, minc, maxc, gen, device)])
      sizes = sizes[:int(torch.searchsorted(torch.cumsum(sizes, 0), M)) + 1]
      d = sizes - minc   # remove the excess from random communities, keeping them at least minc
      _decrement(d, torch.zeros_like(d), torch.tensor([int(sizes.sum()) - M], device = device), gen)
      sizes = d + minc
      _decrement(sizes, torch.zeros_like(d), torch.tensor([int(sizes.sum()) - M], device = device), gen)   # if all are at minc
      sizes = torch.sort(sizes).values
      sizes = sizes[sizes > 0]
      if om > sizes.shape[0] and on:
         raise ValueError(f"om = {om} is more than the {sizes.shape[0]} communities")
      node = torch.repeat_interleave(torch.arange(n, device = device), O)
      need = torch.round((1 - mu) * torch.maximum(kin, kout).double()).long()
      need = need[node] // O[node] + (need[node] % O[node] > 0).long()
      comm = _lfr_assign(node, need, sizes, gen)
      s, t, internal = _lfr_edges(kin, kout, node, comm, sizes.shape[0], mu, directed, gen)
      ea, enames = np.zeros((s.shape[0], 0), dtype = np.float32), []
      if muw is not None:
         st = (kin + kout if directed else kin).double() ** (1.5 if beta is None else beta)
         w = _lfr_weights(s, t, internal, (1 - muw) * st, muw * st)
         ea, enames = w.cpu().numpy().astype(np.float32)[:, None], [weight]
      table, qtable = None, None
      if attribute is not None:
         o = torch.argsort(node * sizes.shape[0] + comm)
         if on == 0:
            cm = comm[o].cpu().numpy().astype(np.float32)
            table = lambda perm: cm[perm][:, None]
         else:
            cs = np.split(comm[o].cpu().numpy(), np.cumsum(O.cpu().numpy())[:-1])
            cm = np.empty(n, dtype = object)
            cm[:] = [_membership(tuple(x.tolist())) for x in cs]
            qtable = lambda perm: pd.DataFrame({attribute: pd.Categorical(cm[perm])})
      return _build(s.cpu().numpy(), t.cpu().numpy(), np.arange(n), ea, enames, table, [] if attribute is None or on else [attribute], directed, device, qtable, [attribute] if qtable else [])
   if n is not None or k is not None or maxk is not None:
      raise ValueError("with net, n, k and maxk come from its degrees: leave them out")
   (mnode, mcode), labels, S, T = _communities(net, attribute, directed, device, overlap = True)
   directed = net.directed
   out, inn = _degrees(net)
   wcol = weight if weight is not None and weight in net.edge_attributes else None
   if wcol is not None:   # weights of the loop-free input edges, aligned with S and T
      j = net.edge_attributes.index(wcol)
      W = []
      for c in range(len(net.components)):
         ei, x = net._input_edges(c)
         W.append(x[ei[0] != ei[1], j].double())
      W = torch.nan_to_num(torch.cat(W))

   def make(l, nodes, n0, count, loops, gen):
      N = nodes.shape[0]
      pos = torch.full((out.shape[0],), -1, dtype = torch.long, device = nodes.device)
      pos[nodes] = torch.arange(N, device = nodes.device)
      sel = torch.isin(mnode, nodes)
      present, comm = torch.unique(mcode[sel], return_inverse = True)
      node = pos[mnode[sel]]
      mine = torch.isin(S, nodes) & torch.isin(T, nodes)   # intra-layer edges of this layer
      a, b = pos[S[mine]], pos[T[mine]]
      ext = ~_sharing(node, comm, N, present.shape[0])(a, b)
      q = _layer_value(mu, None, net, l, "mu", True)
      q = (float(ext.double().mean()) if a.shape[0] else 0.) if q is None else q
      kin, kout = (inn[nodes], out[nodes]) if directed else (out[nodes], out[nodes])
      s, t, internal = _lfr_edges(kin, kout, node, comm, present.shape[0], q, directed, gen)
      if wcol is None:
         return s, t
      w = W[mine]
      e = _layer_value(beta, None, net, l, "beta", True)
      st = torch.zeros(N, dtype = torch.float64, device = w.device).index_add_(0, a, w).index_add_(0, b, w) if e is None else (kin + kout if directed else kin).double() ** e
      qw = _layer_value(muw, None, net, l, "muw", True)
      qw = (float(w[ext].sum() / w.sum()) if float(w.sum()) > 0 else 0.) if qw is None else qw
      return s, t, {wcol: _lfr_weights(s, t, internal, (1 - qw) * st, qw * st)}

   return _randomize(net, seed, make)


# ------------------------------------------------------------------------------------------------
# Testing statistics against null models
# ------------------------------------------------------------------------------------------------


def _bins(values, target = 20):
   """
   Values rounded so that their distribution has about ``target`` distinct values: integers are
   kept when they are few, otherwise values are rounded to the power of ten that splits their
   range into about ``target`` bins.

   :param values: :class:`numpy.ndarray` of floats.
   :return: Rounded values.
   """
   finite = values[np.isfinite(values)]
   if len(finite) == 0 or (np.all(finite == np.round(finite)) and len(np.unique(finite)) <= 3 * target):
      return values
   spread = finite.max() - finite.min()
   if spread == 0:
      return values
   step = 10. ** math.floor(math.log10(spread / target))
   for s in (step, 2 * step, 5 * step, 10 * step):   # widest of 1, 2, 5 steps with enough bins
      if spread / s <= 2 * target:
         step = s
         break
   digits = max(0, -math.floor(math.log10(step)))   # removes float noise of the rounding
   return np.round(np.round(values / step) * step, digits)


def randomization_test(net, statistic, model, n = 100, seed = None, **parameters):
   """
   Test a statistic of a network against a null model: the statistic is computed on ``net`` and
   on ``n`` randomizations of it made by ``model``, and the observed value is compared with their
   distribution.

   ``model`` is any generator of this module that takes ``net`` (e.g.
   :func:`configuration_model`, :func:`edge_swap`, :func:`stochastic_block_model`), called as
   ``model(net = net, seed = ..., **parameters)``, or a fitted ERGM (:class:`~graphoskop.ERGMFit`),
   whose networks are drawn with :meth:`~graphoskop.ERGMFit.simulate`. ``statistic`` is any function
   of a network returning a number, such as :func:`~graphoskop.average_clustering` or
   ``lambda g: gsk.assortativity(g, "age")``.

   The pseudo p-values are the shares of randomizations scoring above, below or equal to the
   observed value; ``p_value`` is the two sided one,
   :math:`\\min(1, 2 \\min(P(x \\geq obs), P(x \\leq obs)))`. The distribution is tabulated on
   values rounded to a scale that gives about 20 bins (integers are kept as they are when they are
   few).

   :param net: A :class:`~graphoskop.Network`.
   :param statistic: Function of a network returning a number.
   :param model: Generator function, or fitted :class:`~graphoskop.ERGMFit`.
   :param n: Number of randomizations.
   :param seed: Seed of the random numbers (randomization ``k`` gets ``seed + k``), or ``None``.
   :param parameters: Other arguments of the generator (e.g. ``swaps``, ``p``), or of
      :meth:`~graphoskop.ERGMFit.simulate` (``burnin``, ``interval``).
   :return: Dictionary with ``observed`` (the value on ``net``), ``samples``
      (:class:`pandas.DataFrame` with the value of every randomization), ``distribution``
      (:class:`pandas.DataFrame` with columns ``value`` and ``count``, the rounded values),
      ``mean`` and ``std`` of the randomizations, ``z`` (standardized observed value), ``above``,
      ``below`` and ``equal`` (shares of randomizations) and ``p_value``.
   :raises ValueError: If ``statistic`` does not return a number.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      test = gsk.randomization_test(net, gsk.average_clustering, gsk.configuration_model, n = 200, seed = 1)
      test["observed"], test["mean"], test["above"], test["p_value"]
      test["distribution"]                                         # rounded values and their counts

      fit = gsk.ergm(net, "edges + gwesp(0.5, fixed = TRUE)", seed = 1)
      test = gsk.randomization_test(net, gsk.average_clustering, fit, n = 200)   # networks from the ERGM
   """

   def value(g):
      x = statistic(g)
      try:
         return float(x.item() if hasattr(x, "item") else x)
      except (TypeError, ValueError):
         raise ValueError(f"the statistic must return a number, got {type(x).__name__}") from None

   observed = value(net)
   if hasattr(model, "simulate"):
      nets = model.simulate(n, seed = seed, **parameters)
   else:
      nets = (model(net = net, seed = None if seed is None else seed + k, **parameters) for k in range(n))
   x = np.array([value(g) for g in nets])
   rounded = _bins(x)
   values, counts = np.unique(rounded, return_counts = True)
   ok = np.isfinite(x)
   std = float(x[ok].std(ddof = 1)) if ok.sum() > 1 else math.nan
   mean = float(x[ok].mean()) if ok.any() else math.nan
   above, below, equal = float((x > observed).mean()), float((x < observed).mean()), float((x == observed).mean())
   return {"observed": observed, "samples": pd.DataFrame({"value": x}), "distribution": pd.DataFrame({"value": values, "count": counts}),
           "mean": mean, "std": std, "z": (observed - mean) / std if std > 0 else math.nan,
           "above": above, "below": below, "equal": equal, "p_value": min(1., 2 * min(above + equal, below + equal))}
