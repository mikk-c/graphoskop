"""
Link prediction: scoring the node pairs of a network by how likely they are to be connected.

Scoring every pair is unfeasible on large networks, since their number grows quadratically with
the number of nodes. A link prediction is therefore tested on a sample of pairs:
:func:`link_prediction_split` hides some edges of a network, the positives, and draws as many pairs
of unconnected nodes, the negatives (or a multiple of them). It returns the training network,
without the hidden edges, and the validation set, a table of the positive and negative pairs. A
score, such as :func:`preferential_attachment_lp`, is then computed only for the pairs of the
validation set, from the training network, and :func:`link_prediction_evaluation` tells how well
it separates the positives from the negatives (area under the ROC and precision-recall curves,
and the curves themselves).

Scores:

* :func:`preferential_attachment_lp`: the product of the degrees of the two nodes;
* common neighbors: the more neighbors the two nodes share, the higher the score, counted as

  * :func:`common_neighbors_lp`: the number of neighbors the two nodes share;
  * :func:`common_neighbors_jaccard_lp`: the common neighbors as a fraction of the neighbors of either
    node;
  * :func:`adamic_adar_lp`: each common neighbor weighted by one over the log of its degree;
  * :func:`resource_allocation_lp`: each common neighbor weighted by one over its degree;
  * :func:`car_index_lp`: the common neighbors times the links among them;
  * :func:`mutual_information_index_lp`: the information the common neighbors carry about an edge;

* random walks: :func:`local_random_walk_lp`, the probability that a short random walk goes from one
  node to the other, and :func:`simrank_lp`, the chance that two walks started at the two nodes
  meet soon;
* :func:`vertex_similarity_lp`: the paths of all lengths between the two nodes, relative to the
  number expected from their degrees;
* models fitted to the network by Markov chain Monte Carlo, which learn any pattern of
  connection, assortative or not: :func:`hierarchical_random_graph_lp`, the edge probability of the
  pair in hierarchical random graphs, and :func:`block_model_reliability_lp`, in stochastic block
  models.

:func:`ensemble_link_prediction` aggregates the rankings of several scores into one (Borda count
or Kendall tau). In signed networks, :func:`link_prediction_sign_evaluation` tests how well social
balance and social status predict the signs of the hidden edges.

All take the network to compute the scores in, the pairs, and the name of the score column; the
scores that have a weighted version take an optional edge attribute as weight, and the others
their own parameters. Degree and common neighbor scores are computed on the device of the
network, as is vertex similarity; local random walks use SciPy, and SimRank and the fitted
models run on the CPU threads, compiled with Numba (Markov chains are sequential). All scores are float64, so products and sums
of degrees are exact up to :math:`2^{53}`, and never overflow; scores only matter through their
order, so they are not normalized.

Pairs are always given by node ids, never by internal ids, since removing edges can split a
component and change the internal ids. Node attributes can therefore be joined to the validation
set by id, e.g. with :meth:`pandas.DataFrame.merge`.

Self loops are neither hidden nor drawn as negatives. In undirected networks a pair is unordered,
in directed networks a pair ``(u, v)`` stands for an edge from ``u`` to ``v``. In bipartite
networks, only pairs of nodes of different types are drawn, and in undirected bipartite networks
the source is the node of type 0. Multilayer networks are not supported yet.

Example:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv", "nodes.tsv")
   train, validation = gsk.link_prediction_split(net, fraction = 0.1, negatives = 1, seed = 1)
   validation = gsk.preferential_attachment_lp(train, validation)
   gsk.link_prediction_evaluation(validation)              # {"auc": ..., "average_precision": ...}
"""

import math

import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from numba import njit, prange
from scipy.sparse.linalg import eigsh
from scipy.stats import rankdata

from .ergms import _random
from .network import _quiet_sparse
from .signed import SIGN
from .network_statistics import _degrees
from .transformations import _items, _restrict

#: Above this number of candidate pairs, negatives are drawn by rejection instead of enumerating
#: all the candidates.
_ENUMERATE = 2 * 10 ** 7


def _unique(x):
   """
   Sorted distinct values, by sorting (faster than :func:`numpy.unique` on large int64 arrays).

   :param x: NumPy array.
   :return: Sorted distinct values.
   """
   x = np.sort(x)
   return x[np.r_[True, x[1:] != x[:-1]]] if len(x) else x


def _check(net):
   """
   Raise an error for networks that link prediction does not support yet.

   :param net: Network.
   :raises ValueError: If the network is multilayer.
   """
   if net.layer_labels is not None:
      raise ValueError("link prediction does not support multilayer networks yet")


def _types(net):
   """
   Node type of every node in the network wide order, in bipartite networks.

   :param net: Network.
   :return: Boolean NumPy array (``True`` for type 1), or ``None`` if the network is not bipartite.
   """
   if net.bipartite_labels is None:
      return None
   return np.concatenate([np.arange(d.num_nodes) >= net.bipartite_split[c] for c, d in enumerate(net.components)])


def _keys(net, s, t, N):
   """
   Keys of node pairs in the network wide order: ``s * N + t``, with ``s < t`` in undirected
   networks, so that both directions of an edge have the same key.

   :param net: Network.
   :param s: Sources (NumPy array of positions).
   :param t: Targets (NumPy array of positions).
   :param N: Number of nodes.
   :return: Keys (int64 NumPy array).
   """
   if not net.directed:
      s, t = np.minimum(s, t), np.maximum(s, t)
   return s.astype(np.int64) * N + t


def _stored_keys(net, N):
   """
   Keys of all the stored edges of every component, both directions in undirected networks.

   :param net: Network.
   :param N: Number of nodes.
   :return: List of key arrays, one per component, aligned with the columns of ``edge_index``.
   """
   out = []
   for c, d in enumerate(net.components):
      s, t = d.edge_index.cpu().numpy() + net.node_offsets[c]
      out.append(_keys(net, s, t, N))
   return out


def _candidates(net, N, types, size, rng):
   """
   Random pairs of distinct nodes (of different types in bipartite networks), uniform over the
   pairs a network can have.

   :param net: Network.
   :param N: Number of nodes.
   :param types: Node types (see :func:`_types`), or ``None``.
   :param size: Number of pairs.
   :param rng: :class:`numpy.random.Generator`.
   :return: Keys of the pairs, possibly repeated.
   """
   if types is None:
      s = rng.integers(0, N, size)
      t = rng.integers(0, N - 1, size)
      t += t >= s   # uniform over t != s
   else:
      zero, one = np.flatnonzero(~types), np.flatnonzero(types)
      s, t = zero[rng.integers(0, len(zero), size)], one[rng.integers(0, len(one), size)]
      if net.directed:
         flip = rng.random(size) < 0.5
         s, t = np.where(flip, t, s), np.where(flip, s, t)
   return _keys(net, s, t, N)


def _all_candidates(net, N, types):
   """
   Keys of all the pairs a network can have (see :func:`_candidates`).

   :param net: Network.
   :param N: Number of nodes.
   :param types: Node types (see :func:`_types`), or ``None``.
   :return: Keys, sorted and unique.
   """
   if types is None:
      s, t = np.triu_indices(N, 1)
      if net.directed:
         s, t = np.concatenate([s, t]), np.concatenate([t, s])
   else:
      zero, one = np.flatnonzero(~types), np.flatnonzero(types)
      s, t = np.repeat(zero, len(one)), np.tile(one, len(zero))
      if net.directed:
         s, t = np.concatenate([s, t]), np.concatenate([t, s])
   return _unique(_keys(net, s, t, N))


def _negatives(net, N, types, edges, count, rng):
   """
   Distinct pairs of unconnected nodes, drawn uniformly.

   :param net: Network.
   :param N: Number of nodes.
   :param types: Node types (see :func:`_types`), or ``None``.
   :param edges: Keys of the connected pairs, sorted and unique.
   :param count: Number of pairs to draw.
   :param rng: :class:`numpy.random.Generator`.
   :return: Keys of the pairs, in random order.
   :raises ValueError: If there are fewer than ``count`` unconnected pairs.
   """
   if types is None:
      space = N * (N - 1) // (1 if net.directed else 2)
   else:
      space = int(types.sum()) * int((~types).sum()) * (2 if net.directed else 1)
   free = space - len(edges)
   if count > free:
      raise ValueError(f"{count} negatives requested, but there are only {free} pairs of unconnected nodes")
   if space <= _ENUMERATE:
      pool = np.setdiff1d(_all_candidates(net, N, types), edges, assume_unique = True)
      return rng.choice(pool, count, replace = False)
   found = np.empty(0, dtype = np.int64)
   while len(found) < count:   # rejection: drop connected pairs and repetitions
      need = count - len(found)
      draw = _candidates(net, N, types, int(need * space / free * 1.1) + 100, rng)
      at = np.minimum(np.searchsorted(edges, draw), len(edges) - 1)
      draw = draw[edges[at] != draw] if len(edges) else draw
      found = _unique(np.concatenate([found, draw]))
   return rng.choice(found, count, replace = False)   # a uniform subset of the distinct pairs drawn


def _pairs(net, keys, N, types):
   """
   Table of node pairs given by their keys.

   :param net: Network.
   :param keys: Keys of the pairs.
   :param N: Number of nodes.
   :param types: Node types (see :func:`_types`), or ``None``.
   :return: :class:`pandas.DataFrame` with columns ``source`` and ``target`` (node ids).
   """
   s, t = keys // N, keys % N
   if types is not None and not net.directed:   # source of type 0
      flip = types[s]
      s, t = np.where(flip, t, s), np.where(flip, s, t)
   ids = np.concatenate(net.node_ids)
   return pd.DataFrame({"source": ids[s], "target": ids[t]})


def link_prediction_split(net, fraction = 0.1, negatives = 1, seed = None, sign = SIGN):
   """
   Training network and validation set for link prediction.

   A ``fraction`` of the connected node pairs of the network (self loops excluded) are drawn
   uniformly at random as positives, and all their edges are removed from the training network
   (parallel edges included and, in undirected networks, both directions). ``negatives`` times as
   many pairs of nodes that are not connected in ``net`` are drawn uniformly as negatives, without
   repetitions: pairs of distinct nodes anywhere in the network, also in different components, of
   different types in bipartite networks.

   The training network keeps all the nodes, isolated ones included, with their attributes, and
   the attributes of the other edges; components are recomputed, since removing edges can split
   one.

   In signed networks (with the numeric edge attribute ``sign``, see :doc:`signed`), the
   validation set also has a column named as ``sign``, with the sign of each positive (+1 or -1;
   with parallel edges, the sign of the sum of their signs; NaN if it is 0 or for the negatives),
   for :func:`link_prediction_sign_evaluation`.

   :param net: A :class:`~graphoskop.Network`, not multilayer.
   :param fraction: Fraction of the connected pairs hidden as positives, between 0 and 1.
   :param negatives: Number of negatives per positive (rounded to the nearest integer in total).
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param sign: Name of the edge attribute holding the signs, if the network is signed.
   :return: ``(train, validation)``: the training network, a new :class:`~graphoskop.Network`, and
      a :class:`pandas.DataFrame` with columns ``source``, ``target`` (node ids) and ``label`` (1
      for the positives, which come first, 0 for the negatives), and ``sign`` in signed networks.
   :raises ValueError: If the network is multilayer, if ``fraction`` is not between 0 and 1 or
      ``negatives`` is negative, or if there are not enough unconnected pairs.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      train, validation = gsk.link_prediction_split(net, fraction = 0.2, negatives = 5, seed = 1)
      edges, nodes = net.to_pandas_dataframes()
      validation = validation.merge(nodes, left_on = "source", right_on = "node")   # source attributes
   """
   _check(net)
   if not 0 <= fraction <= 1:
      raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
   if negatives < 0:
      raise ValueError(f"negatives must be non-negative, got {negatives}")
   rng = np.random.default_rng(seed)
   N = net.number_of_nodes()
   types = _types(net)
   stored = _stored_keys(net, N)
   loops = [d.edge_index[0].cpu().numpy() == d.edge_index[1].cpu().numpy() for d in net.components]
   edges = _unique(np.concatenate([k[~l] for k, l in zip(stored, loops)]))
   positives = rng.choice(edges, int(round(fraction * len(edges))), replace = False)
   negative = _negatives(net, N, types, edges, int(round(negatives * len(positives))), rng)
   hidden = np.sort(positives)
   keys, loop = np.concatenate(stored), np.concatenate(loops)
   at = np.minimum(np.searchsorted(hidden, keys), max(len(hidden) - 1, 0))
   drop = (hidden[at] == keys) & ~loop if len(hidden) else np.zeros(len(keys), dtype = bool)
   keep = np.split(~drop, np.cumsum([len(k) for k in stored])[:-1])   # per component
   train = _restrict(net, [np.ones(d.num_nodes, dtype = bool) for d in net.components], keep)
   validation = _pairs(net, np.concatenate([positives, negative]), N, types)
   validation["label"] = np.repeat(np.array([1, 0], dtype = np.int64), [len(positives), len(negative)])
   if sign in net.edge_attributes:   # sign of each hidden pair: of the sum over its stored edges
      values = np.concatenate([net.get_edge_attribute(sign, c).cpu().numpy() for c in range(len(net.components))])
      at = np.searchsorted(hidden, keys[drop])
      total = np.bincount(at, np.nan_to_num(values[drop]), minlength = len(hidden))
      signs = np.sign(total[np.searchsorted(hidden, positives)]) if len(hidden) else np.zeros(0)
      validation[sign] = np.r_[np.where(signs != 0, signs, np.nan), np.full(len(negative), np.nan)]
   return train, validation


def _positions(net, nodes):
   """
   Positions of nodes in the network wide order, given by their ids. Ids read from files are
   strings, so ids that are not found are tried again as strings, then as internal ids
   ``(component, index)``.

   :param net: Network.
   :param nodes: List of node ids.
   :return: Positions (int64 NumPy array).
   :raises KeyError: If a node is not in the network.
   """
   index = pd.Index(np.concatenate(net.node_ids))
   pos = index.get_indexer(pd.Index(nodes, tupleize_cols = False))
   miss = np.flatnonzero(pos < 0)
   pos[miss] = index.get_indexer(pd.Index([str(nodes[r]) for r in miss], dtype = object))
   for r in np.flatnonzero(pos < 0):
      c, i = net._nodes(nodes[r])[0]
      pos[r] = net.node_offsets[c] + i
   return pos


def _endpoints(pairs):
   """
   Sources and targets of node pairs given as one pair, an iterable of pairs or a table.

   :param pairs: A pair ``(u, v)``, an iterable of pairs, or a :class:`pandas.DataFrame` with the
      endpoints in its first two columns.
   :return: Lists of sources and targets.
   :raises ValueError: If a pair does not have two nodes.
   """
   if isinstance(pairs, pd.DataFrame):
      return pairs.iloc[:, 0].tolist(), pairs.iloc[:, 1].tolist()
   items = [tuple(p) for p in _items(pairs)]
   if any(len(p) != 2 for p in items):
      raise ValueError("a pair is (u, v)")
   return [p[0] for p in items], [p[1] for p in items]


def _scored(pairs, score, name):
   """
   Scores returned in the shape of the pairs they were computed for.

   :param pairs: The pairs, as given to the score function.
   :param score: Scores (NumPy array).
   :param name: Name of the score column.
   :return: A float for one pair; otherwise a :class:`pandas.DataFrame`, a copy of ``pairs`` with
      the column ``name`` added if it is a table, with columns ``source``, ``target`` and ``name``
      if not.
   """
   if isinstance(pairs, pd.DataFrame):
      out = pairs.copy()
      out[name] = score
      return out
   if isinstance(pairs, tuple):
      return float(score[0])
   s, t = _endpoints(pairs)
   return pd.DataFrame({"source": s, "target": t, name: score})


def preferential_attachment_lp(net, pairs, weight = None, name = "score"):
   """
   Preferential attachment score of node pairs: the product of the degrees of the two nodes,
   :math:`k_u k_v`, since in growth models such as the Barabási-Albert model the chance of a new
   edge grows with the degree of both its endpoints. In directed networks, the score of
   ``(u, v)`` is the out-degree of ``u`` times the in-degree of ``v``, :math:`k_u^{out} k_v^{in}`.
   With ``weight``, degrees are strengths (sums of the edge attribute, see
   :func:`~graphoskop.degree`).

   Degrees are computed in ``net``: to test the score, give the training network and the validation
   set of :func:`link_prediction_split`. Degrees are computed on the device of the network.

   :param net: A :class:`~graphoskop.Network`, not multilayer.
   :param pairs: A pair ``(u, v)`` of node ids, an iterable of pairs, or a
      :class:`pandas.DataFrame` with the endpoints in its first two columns (e.g. the validation
      set).
   :param weight: Name of a numeric edge attribute for the weighted degree, or ``None``.
   :param name: Name of the score column.
   :return: The score, a float, for one pair; otherwise a :class:`pandas.DataFrame`: a copy of the
      table with the column ``name`` added, or columns ``source``, ``target`` and ``name``.
   :raises KeyError: If a node or the edge attribute does not exist.
   :raises ValueError: If the network is multilayer, or a pair does not have two nodes.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      train, validation = gsk.link_prediction_split(net, seed = 1)
      validation = gsk.preferential_attachment_lp(train, validation)
      gsk.preferential_attachment_lp(net, ("alice", "bob"))          # one pair
      gsk.preferential_attachment_lp(net, [("alice", "bob"), ("bob", "carol")], weight = "weight")
   """
   _check(net)
   s, t = _endpoints(pairs)
   comps = range(len(net.components))
   k_out = torch.cat([_degrees(net, c, "out", weight) for c in comps]).double()
   k_in = k_out if not net.directed else torch.cat([_degrees(net, c, "in", weight) for c in comps]).double()
   device = k_out.device
   s = torch.as_tensor(_positions(net, s), device = device)
   t = torch.as_tensor(_positions(net, t), device = device)
   return _scored(pairs, (k_out[s] * k_in[t]).cpu().numpy(), name)


#: Largest number of (pair, candidate common neighbor) entries processed at once by the
#: neighborhood scores, to bound memory.
_CHUNK = 2 ** 23


def _arcs(net, weight):
   """
   Distinct arcs of a network in the network wide order, self loops left out and the weights of
   parallel edges summed, sorted by source then target (undirected edges give both arcs).

   :param net: Network.
   :param weight: Name of a numeric edge attribute, or ``None`` for weight 1 (also for parallel
      edges).
   :return: Dictionary with ``N`` (number of nodes), ``key`` (sorted ``s * N + t``), ``src``,
      ``dst``, ``w`` (float64 weights), ``out_ptr`` (CSR pointers of the out-arcs), ``in_order``
      (arcs sorted by target then source) and ``in_ptr`` (CSR pointers of the in-arcs), all on the
      device of the network.
   :raises KeyError: If there is no such edge attribute.
   """
   N = net.number_of_nodes()
   shift = lambda c, d: d.edge_index + int(net.node_offsets[c])
   s, t = torch.cat([shift(c, d) for c, d in enumerate(net.components)], 1)
   if weight is None:
      w = torch.ones(s.shape[0], dtype = torch.float64, device = s.device)
   else:
      w = torch.cat([net.get_edge_attribute(weight, c).double() for c in range(len(net.components))])
   ok = s != t
   arcs = _arc_table(N, s[ok], t[ok], w[ok, None])
   if weight is None:   # neighborhoods are sets
      arcs["w"] = torch.ones_like(arcs["w"])
   return arcs


def _arc_table(N, s, t, values):
   """
   Distinct arcs from a list of arcs, with their values summed over repeated arcs.

   :param N: Number of nodes.
   :param s: Sources (int64 tensor).
   :param t: Targets (int64 tensor).
   :param values: Values of each arc (float64 tensor, one row per arc, one column per value).
   :return: Dictionary as in :func:`_arcs`, with ``w`` the first column of the summed values
      and ``values`` all of them.
   """
   key, inv = torch.unique(s * N + t, return_inverse = True)
   W = torch.zeros((key.shape[0], values.shape[1]), dtype = torch.float64, device = s.device).index_add_(0, inv, values)
   src, dst = key // N, key % N
   ptr = lambda x: torch.cat([x.new_zeros(1), torch.cumsum(torch.bincount(x, minlength = N), 0)])
   return {"N": N, "key": key, "src": src, "dst": dst, "w": W[:, 0], "values": W, "out_ptr": ptr(src), "in_order": torch.argsort(dst * N + src), "in_ptr": ptr(dst)}


def _expand(length, first):
   """
   Positions of consecutive runs, in chunks: run ``i`` covers ``first[i]`` to
   ``first[i] + length[i] - 1``, and the runs are cut into chunks of at most :data:`_CHUNK`
   positions (a longer run is a chunk of its own).

   :param length: Length of each run (int64 tensor).
   :param first: First position of each run (int64 tensor).
   :return: Iterator of ``(i, at)``: the run of each position and the position.
   """
   ends = torch.cumsum(length, 0)
   a, R = 0, length.shape[0]
   while a < R:
      done = 0 if a == 0 else int(ends[a - 1])
      b = max(int(torch.searchsorted(ends, torch.tensor(done + _CHUNK, device = length.device), right = True)), a + 1)
      L = length[a:b]
      i = torch.repeat_interleave(torch.arange(a, b, device = length.device), L)
      yield i, torch.arange(i.shape[0], device = length.device) - torch.repeat_interleave(torch.cumsum(L, 0) - L, L) + torch.repeat_interleave(first[a:b], L)
      a = b


def _has_arc(arcs, q):
   """
   Which arcs, given by key ``s * N + t``, exist, and where.

   :param arcs: Arcs, from :func:`_arcs`.
   :param q: Keys (int64 tensor).
   :return: Boolean tensor, and the position of each key in the sorted arcs (meaningful where
      it exists).
   """
   key = arcs["key"]
   found = torch.searchsorted(key, q).clamp(max = key.shape[0] - 1)
   return key[found] == q, found


def _hits(arcs, u, v):
   """
   Common neighbors ``z`` of each pair (``u -> z -> v`` in directed networks), in chunks. For each
   pair, the shorter list between the out-neighbors of ``u`` and the in-neighbors of ``v`` is
   scanned, and each of its nodes is looked up in the sorted arcs; pairs are processed in chunks
   of at most :data:`_CHUNK` lookups.

   :param arcs: Arcs, from :func:`_arcs`.
   :param u: Sources (positions, int64 tensor).
   :param v: Targets (positions, int64 tensor).
   :return: Iterator of ``(p, z, w_uz, w_zv, i_uz, i_zv)``: pair, common neighbor, the weights of
      its two arcs and their positions in the sorted arcs, one entry per common neighbor of a
      pair.
   """
   if arcs["key"].shape[0] == 0 or u.shape[0] == 0:
      return
   N, w = arcs["N"], arcs["w"]
   optr, iptr = arcs["out_ptr"], arcs["in_ptr"]
   k_out, k_in = optr[1:] - optr[:-1], iptr[1:] - iptr[:-1]
   fwd = k_out[u] <= k_in[v]   # scan the out-neighbors of u, else the in-neighbors of v
   for p, at in _expand(torch.where(fwd, k_out[u], k_in[v]), torch.where(fwd, optr[u], iptr[v])):
      f = fwd[p]
      arc = torch.where(f, at, arcs["in_order"][at])   # position of the scanned arc in the sorted arcs
      z = torch.where(f, arcs["dst"][arc], arcs["src"][arc])
      hit, found = _has_arc(arcs, torch.where(f, z * N + v[p], u[p] * N + z))   # the other arc of u -> z -> v
      w_uz = torch.where(f, w[arc], w[found])
      w_zv = torch.where(f, w[found], w[arc])
      i_uz = torch.where(f, arc, found)
      i_zv = torch.where(f, found, arc)
      yield p[hit], z[hit], w_uz[hit], w_zv[hit], i_uz[hit], i_zv[hit]


def _common(arcs, u, v, combine, factor):
   """
   Sum over the common neighbors ``z`` of each pair (``u -> z -> v`` in directed networks) of
   ``combine(w_uz, w_zv) * factor[z]`` (see :func:`_hits`).

   :param arcs: Arcs, from :func:`_arcs`.
   :param u: Sources (positions, int64 tensor).
   :param v: Targets (positions, int64 tensor).
   :param combine: Function of the two edge weights (float64 tensors).
   :param factor: Weight of each node as a common neighbor (float64 tensor over the nodes).
   :return: Sums (float64 tensor over the pairs).
   """
   total = torch.zeros(u.shape[0], dtype = torch.float64, device = u.device)
   for p, z, w_uz, w_zv, _, _ in _hits(arcs, u, v):
      total.index_add_(0, p, combine(w_uz, w_zv) * factor[z])
   return total


def _local_links(arcs, u, v):
   """
   Number of links among the common neighbors of each pair (arcs between two common neighbors in
   directed networks): for each common neighbor ``z``, its out-neighbors that are common neighbors
   too are counted.

   :param arcs: Arcs, from :func:`_arcs`.
   :param u: Sources (positions, int64 tensor).
   :param v: Targets (positions, int64 tensor).
   :return: Counts (float64 tensor over the pairs; each undirected link is counted twice).
   """
   N, optr = arcs["N"], arcs["out_ptr"]
   total = torch.zeros(u.shape[0], dtype = torch.float64, device = u.device)
   for p, z, _, _, _, _ in _hits(arcs, u, v):
      for h, at in _expand(optr[z + 1] - optr[z], optr[z]):
         x = arcs["dst"][at]   # a neighbor of the common neighbor z
         q = p[h]
         ok = (x != u[q]) & (x != v[q]) & _has_arc(arcs, u[q] * N + x)[0] & _has_arc(arcs, x * N + v[q])[0]
         total.index_add_(0, q[ok], torch.ones_like(q[ok], dtype = torch.float64))
   return total


def _neighborhood_score(net, pairs, weight, name, kind):
   """
   Common neighbors, Jaccard, Adamic-Adar or resource allocation scores of node pairs.

   :param net: Network, not multilayer.
   :param pairs: Pairs, as in :func:`preferential_attachment_lp`.
   :param weight: Name of a numeric edge attribute, or ``None``.
   :param name: Name of the score column.
   :param kind: ``"common_neighbors_lp"``, ``"jaccard"``, ``"adamic_adar_lp"``,
      ``"resource_allocation_lp"`` or ``"car"``.
   :return: Scores, shaped as in :func:`preferential_attachment_lp`.
   """
   _check(net)
   s, t = _endpoints(pairs)
   arcs = _arcs(net, weight)
   device = arcs["key"].device
   u = torch.as_tensor(_positions(net, s), dtype = torch.int64, device = device)
   v = torch.as_tensor(_positions(net, t), dtype = torch.int64, device = device)
   N, w = arcs["N"], arcs["w"]
   s_out = torch.zeros(N, dtype = torch.float64, device = device).index_add_(0, arcs["src"], w)
   s_in = torch.zeros(N, dtype = torch.float64, device = device).index_add_(0, arcs["dst"], w)
   mean = lambda a, b: (a + b) / 2
   ones = torch.ones(N, dtype = torch.float64, device = device)
   k = s_out if not net.directed else s_out + s_in   # strength (degree if unweighted) of a common neighbor
   if kind == "common_neighbors_lp":
      score = _common(arcs, u, v, mean, ones)
   elif kind == "car":
      score = _common(arcs, u, v, mean, ones) * _local_links(arcs, u, v) / (1 if net.directed else 2)
   elif kind == "jaccard":
      inter = _common(arcs, u, v, torch.minimum, ones)
      union = s_out[u] + s_in[v] - inter
      score = torch.where(union > 0, inter / union.clamp(min = 1e-300), torch.zeros_like(inter))
   elif kind == "adamic_adar_lp":
      score = _common(arcs, u, v, mean, 1 / torch.log(k if weight is None else 1 + k))
   else:
      score = _common(arcs, u, v, mean, 1 / k)
   return _scored(pairs, score.cpu().numpy(), name)


def adamic_adar_lp(net, pairs, weight = None, name = "score"):
   """
   Adamic-Adar score of node pairs: the common neighbors, each counting
   :math:`1 / \\log k_z`, so that common neighbors with few neighbors count more,
   :math:`\\sum_{z \\in \\Gamma(u) \\cap \\Gamma(v)} 1 / \\log k_z` (Adamic and Adar, 2003). A common
   neighbor has at least the two neighbors ``u`` and ``v``, so :math:`\\log k_z > 0`.

   Neighborhoods are as in :func:`common_neighbors_lp`; :math:`k_z` is the number of neighbors of
   :math:`z`, in directed networks its in-neighbors plus its out-neighbors. With ``weight``, each
   common neighbor counts :math:`(w_{uz} + w_{zv}) / 2` divided by :math:`\\log(1 + s_z)`, with
   :math:`s_z` its strength (Lü and Zhou, 2009, up to the factor 1/2); unlike the other scores,
   this does not reduce to the unweighted score when all weights are 1.

   :param net: A :class:`~graphoskop.Network`, not multilayer.
   :param pairs: Pairs, as in :func:`preferential_attachment_lp`.
   :param weight: Name of a numeric edge attribute (positive values), or ``None``.
   :param name: Name of the score column.
   :return: Scores, as in :func:`preferential_attachment_lp`.
   :raises KeyError: If a node or the edge attribute does not exist.
   :raises ValueError: If the network is multilayer, or a pair does not have two nodes.

   Example:

   .. code-block:: python

      validation = gsk.adamic_adar_lp(train, validation)
   """
   return _neighborhood_score(net, pairs, weight, name, "adamic_adar_lp")


def car_index_lp(net, pairs, name = "score"):
   """
   CAR index of node pairs (Cannistraci, Alanis-Lobato and Ravasi, 2013): the number of common
   neighbors times the number of links among them, :math:`CN(u, v) \\cdot LCL(u, v)`. Under the
   local community paradigm, two nodes are likely to connect if their common neighbors form a
   local community, i.e. are connected to one another, rather than being unrelated nodes that
   happen to be adjacent to both. Pairs whose common neighbors share no link score 0.

   Neighborhoods are as in :func:`common_neighbors_lp`. In directed networks, the common neighbors
   of ``(u, v)`` are the nodes :math:`z` of the paths :math:`u \\to z \\to v`, and the local
   community links are the arcs between two of them. Each pair costs the degrees of its common
   neighbors.

   :param net: A :class:`~graphoskop.Network`, not multilayer.
   :param pairs: Pairs, as in :func:`preferential_attachment_lp`.
   :param name: Name of the score column.
   :return: Scores, as in :func:`preferential_attachment_lp`.
   :raises KeyError: If a node does not exist.
   :raises ValueError: If the network is multilayer, or a pair does not have two nodes.

   Example:

   .. code-block:: python

      validation = gsk.car_index_lp(train, validation)
   """
   return _neighborhood_score(net, pairs, None, name, "car")


def common_neighbors_lp(net, pairs, weight = None, name = "score"):
   """
   Common neighbors score of node pairs: the number of neighbors that the two nodes share,
   :math:`|\\Gamma(u) \\cap \\Gamma(v)|`, since two nodes with many friends in common are likely
   to become friends (triadic closure).

   Neighborhoods are sets: self loops and parallel edges are ignored. In directed networks,
   :math:`\\Gamma(u)` holds the out-neighbors of ``u`` and :math:`\\Gamma(v)` the in-neighbors of
   ``v``, so the score of ``(u, v)`` counts the paths :math:`u \\to z \\to v`. With ``weight``, each
   common neighbor counts the mean weight of its two edges, :math:`(w_{uz} + w_{zv}) / 2` (the
   weights of parallel edges summed), as in Lü and Zhou (2009) up to the factor 1/2, so that the
   score is the unweighted one when all weights are 1.

   Each pair costs the number of neighbors of its endpoint with fewer neighbors, and the pairs
   are processed in parallel, on the device of the network.

   :param net: A :class:`~graphoskop.Network`, not multilayer.
   :param pairs: Pairs, as in :func:`preferential_attachment_lp`.
   :param weight: Name of a numeric edge attribute, or ``None``.
   :param name: Name of the score column.
   :return: Scores, as in :func:`preferential_attachment_lp`.
   :raises KeyError: If a node or the edge attribute does not exist.
   :raises ValueError: If the network is multilayer, or a pair does not have two nodes.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      train, validation = gsk.link_prediction_split(net, seed = 1)
      validation = gsk.common_neighbors_lp(train, validation)
      gsk.common_neighbors_lp(net, ("alice", "bob"), weight = "weight")
   """
   return _neighborhood_score(net, pairs, weight, name, "common_neighbors_lp")


def common_neighbors_jaccard_lp(net, pairs, weight = None, name = "score"):
   """
   Jaccard score of node pairs: the common neighbors as a fraction of all the neighbors of the two
   nodes, :math:`|\\Gamma(u) \\cap \\Gamma(v)| / |\\Gamma(u) \\cup \\Gamma(v)|`, 0 if both have no
   neighbors. Unlike :func:`common_neighbors_lp`, it does not favor pairs of hubs.

   Neighborhoods are as in :func:`common_neighbors_lp`. With ``weight``, the weighted Jaccard
   (Ruzicka) similarity of the two vectors of edge weights,
   :math:`\\sum_z \\min(w_{uz}, w_{zv}) / \\sum_z \\max(w_{uz}, w_{zv})`, the sums running over all
   nodes (weights of parallel edges summed, 0 for missing edges, positive values expected).

   :param net: A :class:`~graphoskop.Network`, not multilayer.
   :param pairs: Pairs, as in :func:`preferential_attachment_lp`.
   :param weight: Name of a numeric edge attribute (positive values), or ``None``.
   :param name: Name of the score column.
   :return: Scores, as in :func:`preferential_attachment_lp`.
   :raises KeyError: If a node or the edge attribute does not exist.
   :raises ValueError: If the network is multilayer, or a pair does not have two nodes.

   Example:

   .. code-block:: python

      validation = gsk.common_neighbors_jaccard_lp(train, validation)
   """
   return _neighborhood_score(net, pairs, weight, name, "jaccard")


def resource_allocation_lp(net, pairs, weight = None, name = "score"):
   """
   Resource allocation score of node pairs: the amount of a resource that ``u`` sends to ``v``
   through their common neighbors, each of which splits what it receives equally among its
   neighbors, :math:`\\sum_{z \\in \\Gamma(u) \\cap \\Gamma(v)} 1 / k_z` (Zhou, Lü and Zhang, 2009).
   It punishes common neighbors with many neighbors more than :func:`adamic_adar_lp`.

   Neighborhoods and :math:`k_z` are as in :func:`adamic_adar_lp`. With ``weight``, each common
   neighbor counts :math:`(w_{uz} + w_{zv}) / 2` divided by its strength :math:`s_z` (Lü and Zhou,
   2009, up to the factor 1/2), the unweighted score when all weights are 1.

   :param net: A :class:`~graphoskop.Network`, not multilayer.
   :param pairs: Pairs, as in :func:`preferential_attachment_lp`.
   :param weight: Name of a numeric edge attribute (positive values), or ``None``.
   :param name: Name of the score column.
   :return: Scores, as in :func:`preferential_attachment_lp`.
   :raises KeyError: If a node or the edge attribute does not exist.
   :raises ValueError: If the network is multilayer, or a pair does not have two nodes.

   Example:

   .. code-block:: python

      validation = gsk.resource_allocation_lp(train, validation)
   """
   return _neighborhood_score(net, pairs, weight, name, "resource_allocation_lp")


@njit(cache = True)
def _mi_self_information(M, a, b):
   """
   Self-information, in bits, of the existence of an edge between two nodes of degrees ``a`` and
   ``b`` in a network of ``M`` edges: :math:`-\\log_2 p`, with
   :math:`p = 1 - \\binom{M - b}{a} / \\binom{M}{a}`.
   """
   if M - b < a:
      return 0.
   r = math.exp(math.lgamma(M - b + 1) - math.lgamma(M - b - a + 1) - math.lgamma(M + 1) + math.lgamma(M - a + 1))
   return -math.log1p(-r) / math.log(2) if r < 1 else np.inf


@njit(cache = True)
def _mi_pairs(M, a, b):
   """Self-information of an edge for each pair of degrees ``a[i]``, ``b[i]`` (see :func:`_mi_self_information`)."""
   out = np.empty(a.shape[0])
   for i in range(a.shape[0]):
      out[i] = _mi_self_information(M, a[i], b[i])
   return out


@njit(cache = True)
def _mi_node_information(ptr, nbr, deg, M, clustering, out):
   """
   Mutual information :math:`I(L; z)` of every node with the existence of an edge between two of
   its neighbors: the mean over the pairs of its neighbors of their self-information, plus
   :math:`\\log_2 C_z`. Pairs are grouped by the degrees of their two nodes.
   """
   for z in range(ptr.shape[0] - 1):
      k = ptr[z + 1] - ptr[z]
      if k < 2:
         out[z] = np.nan
         continue
      d = np.sort(deg[nbr[ptr[z]:ptr[z + 1]]])
      values, counts = np.empty(k, dtype = np.int64), np.empty(k, dtype = np.int64)
      g = 0
      for i in range(k):
         if i == 0 or d[i] != d[i - 1]:
            values[g], counts[g] = d[i], 0
            g += 1
         counts[g - 1] += 1
      total = 0.
      for i in range(g):
         ci = counts[i]
         total += ci * (ci - 1) / 2 * _mi_self_information(M, values[i], values[i])
         for j in range(i + 1, g):
            total += ci * counts[j] * _mi_self_information(M, values[i], values[j])
      out[z] = total / (k * (k - 1) / 2) + (math.log2(clustering[z]) if clustering[z] > 0 else -np.inf)


def mutual_information_index_lp(net, pairs, name = "score"):
   """
   Mutual information score of node pairs (Tan, Xia and Zhu, 2014): how much knowing their common
   neighbors reduces the uncertainty about an edge between the two nodes,
   :math:`s_{uv} = \\sum_{z \\in \\Gamma(u) \\cap \\Gamma(v)} I(L; z) - I(L_{uv})`.

   * :math:`I(L_{uv}) = -\\log_2 p_{uv}` is the self-information of an edge between :math:`u` and
     :math:`v` given only their degrees, with
     :math:`p_{uv} = 1 - \\binom{M - k_v}{k_u} / \\binom{M}{k_u}` the probability that two nodes
     of these degrees are connected when :math:`M` edges are placed at random: pairs of hubs are
     expected to be connected anyway, so their edge is less informative.
   * :math:`I(L; z)` is what a common neighbor :math:`z` tells about an edge between two of its
     neighbors: the mean, over the pairs :math:`m, n` of its neighbors, of
     :math:`I(L_{mn}) - I(L_{mn} | z)`, where :math:`I(L_{mn} | z) = -\\log_2 C_z`, with
     :math:`C_z` the local clustering coefficient of :math:`z`.

   A common neighbor whose neighbors are all unconnected (:math:`C_z = 0`) makes the score
   :math:`-\\infty`: the pair is ranked last. Pairs without common neighbors score
   :math:`-I(L_{uv})`, so they are ranked by degrees, like :func:`preferential_attachment_lp`.

   Neighborhoods are sets, as in :func:`common_neighbors_lp` (self loops and parallel edges ignored),
   and :math:`M` is the number of connected pairs. Clustering coefficients are computed on the
   device of the network; the information of each node (a sum over the pairs of its neighbors,
   grouped by degree) is compiled with Numba, on the CPU.

   :param net: An undirected :class:`~graphoskop.Network`, not multilayer.
   :param pairs: Pairs, as in :func:`preferential_attachment_lp`.
   :param name: Name of the score column.
   :return: Scores, as in :func:`preferential_attachment_lp`.
   :raises KeyError: If a node does not exist.
   :raises ValueError: If the network is directed or multilayer, or a pair does not have two
      nodes.

   Example:

   .. code-block:: python

      validation = gsk.mutual_information_index_lp(train, validation)
   """
   _check(net)
   if net.directed:
      raise ValueError("the mutual information index is defined for undirected networks")
   s, t = _endpoints(pairs)
   arcs = _arcs(net, None)
   device = arcs["key"].device
   u = torch.as_tensor(_positions(net, s), dtype = torch.int64, device = device)
   v = torch.as_tensor(_positions(net, t), dtype = torch.int64, device = device)
   N, src, dst = arcs["N"], arcs["src"], arcs["dst"]
   ones = torch.ones(N, dtype = torch.float64, device = device)
   mean = lambda a, b: (a + b) / 2
   k = arcs["out_ptr"][1:] - arcs["out_ptr"][:-1]
   M = int(src.shape[0]) // 2
   once = src < dst   # each edge once: its common neighbors close a triangle at both ends
   cn = _common(arcs, src[once], dst[once], mean, ones)
   triangles = torch.zeros(N, dtype = torch.float64, device = device).index_add_(0, src[once], cn).index_add_(0, dst[once], cn) / 2
   kd = k.double()
   clustering = torch.where(k > 1, triangles / (kd * (kd - 1) / 2).clamp(min = 1), torch.zeros_like(kd)).cpu().numpy()
   info = np.empty(N)
   _mi_node_information(arcs["out_ptr"].cpu().numpy(), dst.cpu().numpy(), k.cpu().numpy(), M, clustering, info)
   score = _common(arcs, u, v, mean, torch.as_tensor(np.where(np.isnan(info), 0., info), device = device)).cpu().numpy()
   kc, un, vn = k.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy()
   return _scored(pairs, score - _mi_pairs(M, kc[un], kc[vn]), name)


#: Largest number of pairs whose walk vectors are multiplied at once by :func:`local_random_walk_lp`.
_PAIR_CHUNK = 2 ** 16


def _transition(net, weight):
   """
   Random walk transition matrix of a network, on the CPU: :math:`P = D^{-1} A`, with :math:`A`
   the adjacency matrix of the distinct arcs, self loops left out (weights of parallel edges
   summed, all 1 without ``weight``), :math:`D` the out-strengths.

   :param net: Network.
   :param weight: Name of a numeric edge attribute, or ``None``.
   :return: ``P`` (:class:`scipy.sparse.csr_matrix`) and the out-strengths (NumPy array).
   """
   arcs = _arcs(net, weight)
   N = arcs["N"]
   s, t, w = (arcs[k].cpu().numpy() for k in ("src", "dst", "w"))
   k = np.bincount(s, w, minlength = N)
   P = sp.csr_matrix((w / k[s], (s, t)), shape = (N, N))
   return P, k


def local_random_walk_lp(net, pairs, steps = 3, superposed = False, weight = None, name = "score"):
   """
   Local random walk score of node pairs (Liu and Lü, 2010): how likely a random walker started
   at one node is to be at the other after ``steps`` steps, from both sides,
   :math:`s_{uv}(t) = q_u \\pi_{uv}(t) + q_v \\pi_{vu}(t)`, where :math:`\\pi_{uv}(t)` is the
   probability that a walk of :math:`t` steps from :math:`u` ends at :math:`v` and
   :math:`q_u = k_u / 2m` the stationary probability of :math:`u` (its share of the degrees).
   Few steps keep the walk local: :math:`t = 2` gives a degree-weighted count of the common
   neighbors, larger :math:`t` reaches pairs at distance up to :math:`t`. With
   ``superposed = True``, the superposed random walk, :math:`\\sum_{\\tau = 1}^{t} s_{uv}(\\tau)`,
   which favors nodes close to each other.

   The walk moves to a uniformly random neighbor (with ``weight``, to a neighbor with
   probability proportional to the weight of the edge); neighborhoods are sets, as in
   :func:`common_neighbors_lp` (self loops and parallel edges ignored, weights of parallel edges
   summed). In directed networks the walk follows the edges, and :math:`q_u` is the share of the
   out-degrees. Only the pairs are scored: :math:`\\pi_{uv}(t)` is the product of the
   distribution after :math:`\\lceil t / 2 \\rceil` steps from :math:`u` and the probabilities of
   reaching :math:`v` in the remaining steps, two sparse vectors with about :math:`k^{t / 2}`
   entries each (:math:`k` the average degree), computed with SciPy on the CPU.

   :param net: A :class:`~graphoskop.Network`, not multilayer.
   :param pairs: Pairs, as in :func:`preferential_attachment_lp`.
   :param steps: Number of steps :math:`t`, at least 1.
   :param superposed: Whether to sum the scores of 1 to ``steps`` steps.
   :param weight: Name of a numeric edge attribute (positive values), or ``None``.
   :param name: Name of the score column.
   :return: Scores, as in :func:`preferential_attachment_lp`.
   :raises KeyError: If a node or the edge attribute does not exist.
   :raises ValueError: If the network is multilayer, a pair does not have two nodes, or
      ``steps`` is smaller than 1.

   Example:

   .. code-block:: python

      validation = gsk.local_random_walk_lp(train, validation, steps = 3)
      validation = gsk.local_random_walk_lp(train, validation, steps = 3, superposed = True, name = "srw")
   """
   _check(net)
   if steps < 1:
      raise ValueError(f"steps must be at least 1, got {steps}")
   s, t = _endpoints(pairs)
   us, vs = _positions(net, s), _positions(net, t)
   P, k = _transition(net, weight)
   q = k / k.sum() if k.sum() > 0 else k
   nodes, inv = np.unique(np.r_[us, vs], return_inverse = True)
   iu, iv = inv[:len(us)], inv[len(us):]
   E = sp.csr_matrix((np.ones(len(nodes)), (np.arange(len(nodes)), nodes)), shape = (len(nodes), P.shape[0]))
   forward, backward = [E], [E]   # rows: distributions after a steps from each node; probabilities of reaching it in b steps
   PT = P.T.tocsr()
   for i in range((steps + 1) // 2):
      forward.append(forward[-1] @ P)
   for i in range(steps // 2):
      backward.append(backward[-1] @ PT)
   score = np.zeros(len(us))
   for tau in range(1 if superposed else steps, steps + 1):
      F, B = forward[(tau + 1) // 2], backward[tau // 2]
      for c in range(0, len(us), _PAIR_CHUNK):
         x, y = iu[c:c + _PAIR_CHUNK], iv[c:c + _PAIR_CHUNK]
         uv = np.asarray(F[x].multiply(B[y]).sum(1)).ravel()   # pi_uv(tau)
         vu = np.asarray(F[y].multiply(B[x]).sum(1)).ravel()   # pi_vu(tau)
         score[c:c + _PAIR_CHUNK] += q[us[c:c + _PAIR_CHUNK]] * uv + q[vs[c:c + _PAIR_CHUNK]] * vu
   return _scored(pairs, score, name)


@njit(cache = True)
def _splitmix(x):
   """A well mixed 64-bit seed from any integer (splitmix64), never 0."""
   x = (x + np.uint64(0x9E3779B97F4A7C15))
   x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
   x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
   x = x ^ (x >> np.uint64(31))
   return x if x != 0 else np.uint64(1)


@njit(cache = True, parallel = True)
def _simrank_walks(iptr, inn, us, vs, decay, walks, length, seed, out):
   """
   Monte Carlo SimRank: for each pair, ``walks`` pairs of simultaneous backward random walks (to a
   random in-neighbor at each step), and the mean of ``decay ** t`` over them, ``t`` the step at
   which the two walks first meet (0 if they never meet within ``length`` steps, or one reaches
   a node without in-neighbors). Pair ``p`` draws from its own generator, seeded from ``seed``
   and ``p``, so results do not depend on the number of threads.
   """
   for p in prange(us.shape[0]):
      if us[p] == vs[p]:
         out[p] = 1.
         continue
      state = np.empty(1, dtype = np.uint64)
      state[0] = _splitmix(seed ^ _splitmix(np.uint64(p)))
      total = 0.
      for r in range(walks):
         a, b, f = us[p], vs[p], 1.
         for step in range(length):
            ka, kb = iptr[a + 1] - iptr[a], iptr[b + 1] - iptr[b]
            if ka == 0 or kb == 0:
               break
            a = inn[iptr[a] + int(_random(state) * ka)]
            b = inn[iptr[b] + int(_random(state) * kb)]
            f *= decay
            if a == b:
               total += f
               break
      out[p] = total / walks


def simrank_lp(net, pairs, decay = 0.8, walks = 100, length = 10, seed = None, name = "score"):
   """
   SimRank similarity of node pairs (Jeh and Widom, 2002): two nodes are similar if they are
   pointed to by similar nodes,

   .. math::

      s(u, v) = \\frac{C}{|I(u)| |I(v)|} \\sum_{a \\in I(u)} \\sum_{b \\in I(v)} s(a, b),
      \\qquad s(u, u) = 1,

   with :math:`I(u)` the in-neighbors of :math:`u` (its neighbors in undirected networks) and
   :math:`C` the ``decay``; a node without in-neighbors is similar to no other node.

   The exact scores need all the pairs of nodes at once (:math:`n^2` values), so only the given
   pairs are estimated, from the random surfer-pairs model of Jeh and Widom:
   :math:`s(u, v) = E[C^t]`, where :math:`t` is the first step at which two random walkers, started
   at :math:`u` and :math:`v` and moving at the same time to a random in-neighbor, meet. Each pair
   averages ``walks`` such pairs of walks, of at most ``length`` steps, in parallel over the pairs
   (compiled with Numba, on the CPU). The standard error is at most :math:`0.5 / \\sqrt{walks}`,
   and stopping after ``length`` steps underestimates a score by less than
   :math:`C^{length + 1}`. Results depend on ``seed`` only, not on the number of threads.

   Neighborhoods are sets, as in :func:`common_neighbors_lp` (self loops and parallel edges ignored).

   :param net: A :class:`~graphoskop.Network`, not multilayer.
   :param pairs: Pairs, as in :func:`preferential_attachment_lp`.
   :param decay: The decay :math:`C`, between 0 and 1.
   :param walks: Number of pairs of walks per pair of nodes.
   :param length: Largest number of steps of a walk.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param name: Name of the score column.
   :return: Scores, as in :func:`preferential_attachment_lp`.
   :raises KeyError: If a node does not exist.
   :raises ValueError: If the network is multilayer, a pair does not have two nodes, or
      ``decay``, ``walks`` or ``length`` are not valid.

   Example:

   .. code-block:: python

      validation = gsk.simrank_lp(train, validation, walks = 1000, seed = 1)
   """
   _check(net)
   if not 0 < decay < 1 or walks < 1 or length < 1:
      raise ValueError(f"need 0 < decay < 1, walks >= 1 and length >= 1, got {decay}, {walks}, {length}")
   s, t = _endpoints(pairs)
   us, vs = _positions(net, s), _positions(net, t)
   arcs = _arcs(net, None)
   iptr = arcs["in_ptr"].cpu().numpy()
   inn = arcs["src"][arcs["in_order"]].cpu().numpy()   # in-neighbors, grouped by node
   out = np.zeros(len(us))
   seed = int(np.random.default_rng(seed).integers(1, 2 ** 63))
   _simrank_walks(iptr, inn, us.astype(np.int64), vs.astype(np.int64), float(decay), int(walks), int(length), np.uint64(seed), out)
   return _scored(pairs, out, name)


#: Largest number of entries of the blocks of right-hand sides solved at once by
#: :func:`vertex_similarity_lp`.
_BLOCK = 2 ** 24


def _block_cg(matvec, B, tol, max_iter):
   """
   Conjugate gradients on the columns of ``B`` at once, for a symmetric positive definite
   operator.

   :param matvec: Function applying the operator to a block of columns.
   :param B: Right-hand sides (float64 tensor, one column each).
   :param tol: Relative residual at which a column stops.
   :param max_iter: Largest number of iterations.
   :return: Solutions.
   :raises ValueError: If a column does not converge.
   """
   X = torch.zeros_like(B)
   R = B.clone()
   Pd = R.clone()
   rr = (R * R).sum(0)
   stop = tol ** 2 * rr
   for i in range(max_iter):
      if bool((rr <= stop).all()):
         return X
      AP = matvec(Pd)
      a = rr / (Pd * AP).sum(0).clamp(min = 1e-300)
      X += a * Pd
      R -= a * AP
      new = (R * R).sum(0)
      Pd = R + (new / rr.clamp(min = 1e-300)) * Pd
      rr = new
   raise ValueError(f"vertex similarity: conjugate gradients did not converge in {max_iter} iterations")


def vertex_similarity_lp(net, pairs, phi = 0.9, weight = None, name = "score", tol = 1e-10, max_iter = 10000):
   """
   Vertex similarity of node pairs (Leicht, Holme and Newman, 2006): two nodes are similar if
   their neighbors are similar, counting the paths of all lengths between them, each of length
   :math:`l` weighted by :math:`(\\phi / \\lambda_1)^l`, relative to the number expected in a
   random network with the same degrees,

   .. math::

      S = 2 m \\lambda_1 D^{-1} \\left(I - \\frac{\\phi}{\\lambda_1} A\\right)^{-1} D^{-1},

   with :math:`A` the adjacency matrix, :math:`D` the diagonal matrix of the degrees,
   :math:`\\lambda_1` the largest eigenvalue of :math:`A`, :math:`m` the number of edges and
   :math:`0 < \\phi < 1`: close to 1, long paths count almost as much as short ones. Unlike the
   local scores, it relates nodes at any distance (in the same component), like
   :func:`~graphoskop.katz_centrality`.

   Neighborhoods are sets, as in :func:`common_neighbors_lp` (self loops and parallel edges ignored;
   with ``weight``, :math:`A` holds the weights, summed over parallel edges, :math:`D` the
   strengths and :math:`2m` their sum). Only the pairs are scored: the linear systems
   :math:`(I - \\phi A / \\lambda_1) x = e_u` are solved by conjugate gradients for the
   distinct nodes on one side of the pairs, a block at a time, on the device of the network;
   each costs a few tens of products with :math:`A` (more as :math:`\\phi` approaches 1), so the
   time grows with the number of distinct nodes times the number of edges. Pairs with a node
   without neighbors score 0.

   :param net: An undirected :class:`~graphoskop.Network`, not multilayer.
   :param pairs: Pairs, as in :func:`preferential_attachment_lp`.
   :param phi: The parameter :math:`\\phi`, between 0 and 1 (excluded).
   :param weight: Name of a numeric edge attribute (positive values), or ``None``.
   :param name: Name of the score column.
   :param tol: Relative residual at which conjugate gradients stop.
   :param max_iter: Largest number of conjugate gradient iterations.
   :return: Scores, as in :func:`preferential_attachment_lp`.
   :raises KeyError: If a node or the edge attribute does not exist.
   :raises ValueError: If the network is directed or multilayer, a pair does not have two nodes,
      ``phi`` is not between 0 and 1, or conjugate gradients do not converge.

   Example:

   .. code-block:: python

      validation = gsk.vertex_similarity_lp(train, validation, phi = 0.9)
   """
   _check(net)
   if net.directed:
      raise ValueError("vertex similarity is defined for undirected networks")
   if not 0 < phi < 1:
      raise ValueError(f"phi must be between 0 and 1 (excluded), got {phi}")
   s, t = _endpoints(pairs)
   us, vs = _positions(net, s), _positions(net, t)
   arcs = _arcs(net, weight)
   N, src, dst, w = arcs["N"], arcs["src"], arcs["dst"], arcs["w"]
   device = src.device
   score = np.zeros(len(us))
   if src.shape[0] == 0:
      return _scored(pairs, score, name)
   k = torch.zeros(N, dtype = torch.float64, device = device).index_add_(0, src, w)
   cpu = sp.csr_matrix((w.cpu().numpy(), (src.cpu().numpy(), dst.cpu().numpy())), shape = (N, N))
   lam = float(eigsh(cpu, k = 1, which = "LA", return_eigenvectors = False)[0]) if N > 2 else float(np.abs(cpu.toarray()).max())
   with _quiet_sparse():
      A = torch.sparse_coo_tensor(torch.stack([src, dst]), w, (N, N)).coalesce().to_sparse_csr()
   alpha = phi / lam
   matvec = lambda X: X - alpha * (A @ X)
   if len(np.unique(us)) > len(np.unique(vs)):   # S is symmetric: solve for the side with fewer nodes
      us, vs = vs, us
   cols, inv = np.unique(us, return_inverse = True)
   size = max(1, _BLOCK // max(N, 1))
   rows = torch.as_tensor(vs, device = device)
   kk = k.cpu().numpy()
   for c in range(0, len(cols), size):
      block = cols[c:c + size]
      Bm = torch.zeros((N, len(block)), dtype = torch.float64, device = device)
      Bm[torch.as_tensor(block, device = device), torch.arange(len(block), device = device)] = 1.
      X = _block_cg(matvec, Bm, tol, max_iter)
      sel = np.flatnonzero((inv >= c) & (inv < c + size))
      score[sel] = X[rows[sel], torch.as_tensor(inv[sel] - c, device = device)].cpu().numpy()
   den = kk[us] * kk[vs]
   score = np.where(den > 0, k.sum().item() * lam * score / np.where(den > 0, den, 1), 0.)
   return _scored(pairs, score, name)


@njit(cache = True)
def _hrg_ll(e, pairs):
   """Log-likelihood of one internal node: :math:`e \\log p + (L R - e) \\log(1 - p)`, :math:`p = e / (L R)`."""
   if e == 0 or e == pairs:
      return 0.
   p = e / pairs
   return e * math.log(p) + (pairs - e) * math.log(1. - p)


@njit(cache = True)
def _hrg_lca(parent, size, i, j):
   """Lowest common ancestor of two nodes of the dendrogram: an ancestor has more leaves."""
   while i != j:
      if size[i] < size[j]:
         i = parent[i]
      elif size[j] < size[i]:
         j = parent[j]
      else:
         i, j = parent[i], parent[j]
   return i


@njit(cache = True)
def _hrg_between(ptr, nbr, size, lo, leafat, x, y):
   """
   Number of edges between the leaves of the disjoint subtrees ``x`` and ``y``. The leaves of a
   subtree ``z`` are at the positions ``lo[z]`` to ``lo[z] + size[z] - 1`` of the leaf order
   (``leafat`` is the leaf at each position), so the neighbors of the leaves of the smaller
   subtree are counted if their position is in the range of the other.
   """
   if size[x] > size[y]:
      x, y = y, x
   a, b = lo[y], lo[y] + size[y]
   count = 0
   for p in range(lo[x], lo[x] + size[x]):
      z = leafat[p]
      for k in range(ptr[z], ptr[z + 1]):
         q = lo[nbr[k]]
         count += (q >= a) & (q < b)
   return count


@njit(cache = True)
def _hrg_shift(z0, delta, left, right, lo, stack, leaves, nl):
   """Move the subtree ``z0`` by ``delta`` positions; its leaves are appended to ``leaves[nl:]``."""
   top = 1
   stack[0] = z0
   while top > 0:
      top -= 1
      z = stack[top]
      lo[z] += delta
      if left[z] < 0:
         leaves[nl] = z
         nl += 1
      else:
         stack[top], stack[top + 1] = left[z], right[z]
         top += 2
   return nl


@njit(cache = True)
def _hrg_sweep(n, ptr, nbr, parent, left, right, size, e, lo, leafat, stack, leaves, state, ll):
   """
   ``n`` moves of the chain: an internal node ``r`` with parent ``a``, children ``x``, ``y`` and
   sibling ``u`` becomes the parent of ``y`` and ``u``, ``a`` of ``r`` and ``x``, with the
   Metropolis acceptance rule. After an accepted move, the three subtrees are laid out again
   within the positions of ``a`` in the order, among the four valid ones, that moves the fewest
   leaves.

   :return: The log-likelihood after the moves.
   """
   for step in range(n if n > 2 else 0):
      r = n + int(_random(state) * (n - 1))
      a = parent[r]
      if a < 0:   # the root has no parent to rearrange with
         continue
      u = left[a] if right[a] == r else right[a]
      x, y = (left[r], right[r]) if _random(state) < 0.5 else (right[r], left[r])
      xu = _hrg_between(ptr, nbr, size, lo, leafat, x, u)
      er, ea = e[a] - xu, e[r] + xu
      new = _hrg_ll(er, size[y] * size[u]) + _hrg_ll(ea, (size[y] + size[u]) * size[x])
      old = _hrg_ll(e[r], size[x] * size[y]) + _hrg_ll(e[a], size[r] * size[u])
      if new >= old or _random(state) < math.exp(new - old):
         A, sx, sy, su = lo[a], size[x], size[y], size[u]
         best, order, px, py, pu = -1, 0, 0, 0, 0
         for c in range(4):   # block orders: x y u, x u y, y u x, u y x
            if c == 0:
               cx, cy, cu = A, A + sx, A + sx + sy
            elif c == 1:
               cx, cu, cy = A, A + sx, A + sx + su
            elif c == 2:
               cy, cu, cx = A, A + sy, A + sy + su
            else:
               cu, cy, cx = A, A + su, A + su + sy
            cost = (sx if cx != lo[x] else 0) + (sy if cy != lo[y] else 0) + (su if cu != lo[u] else 0)
            if best < 0 or cost < best:
               best, order, px, py, pu = cost, c, cx, cy, cu
         nl = 0
         if px != lo[x]:
            nl = _hrg_shift(x, px - lo[x], left, right, lo, stack, leaves, nl)
         if py != lo[y]:
            nl = _hrg_shift(y, py - lo[y], left, right, lo, stack, leaves, nl)
         if pu != lo[u]:
            nl = _hrg_shift(u, pu - lo[u], left, right, lo, stack, leaves, nl)
         for k in range(nl):
            leafat[lo[leaves[k]]] = leaves[k]
         left[r], right[r] = (y, u) if order == 0 or order == 2 else (u, y)
         left[a], right[a] = (x, r) if order <= 1 else (r, x)
         parent[u], parent[x] = r, a
         size[r] = sy + su
         lo[r] = min(py, pu)
         e[r], e[a] = er, ea
         ll += new - old
   return ll


@njit(cache = True)
def _hrg_init(n, src, dst, state, parent, left, right, size, e, lo, leafat, stack):
   """
   Random dendrogram, merging two random subtrees at a time: nodes ``0`` to ``n - 1`` are the
   leaves, ``n`` to ``2 n - 2`` the internal nodes, filled into the given arrays, with the leaf
   order (``lo``, ``leafat``) of the left-to-right traversal.

   :return: Its log-likelihood.
   """
   m = 2 * n - 1
   parent[:], left[:], right[:], size[:], e[:] = -1, -1, -1, 1, 0
   roots = np.arange(n)
   for r in range(n - 1):
      k = n - r
      a = int(_random(state) * k)
      b = int(_random(state) * (k - 1))
      b += b >= a
      z = n + r
      left[z], right[z], size[z] = roots[a], roots[b], size[roots[a]] + size[roots[b]]
      parent[roots[a]], parent[roots[b]] = z, z
      roots[a] = z
      roots[b] = roots[k - 1]
   for k in range(src.shape[0]):
      e[_hrg_lca(parent, size, src[k], dst[k])] += 1
   ll = 0.
   for z in range(n, m):
      ll += _hrg_ll(e[z], size[left[z]] * size[right[z]])
   lo[roots[0]] = 0   # the root
   top = 1
   stack[0] = roots[0]
   while top > 0:
      top -= 1
      z = stack[top]
      if left[z] < 0:
         leafat[lo[z]] = z
      else:
         lo[left[z]], lo[right[z]] = lo[z], lo[z] + size[left[z]]
         stack[top], stack[top + 1] = left[z], right[z]
         top += 2
   return ll


@njit(cache = True, parallel = True)
def _hrg_run(n, ptr, nbr, parent, left, right, size, e, lo, leafat, states, ll, sweeps, interval, samples, us, vs, out):
   """
   Advance independent chains in parallel, chain ``c`` on the dendrogram in row ``c`` of the tree
   arrays: ``sweeps`` sweeps, then ``samples[c]`` samples every ``interval`` sweeps, adding the
   probability of the lowest common ancestor of every pair to ``out[c]``.
   """
   for c in prange(states.shape[0]):
      state = states[c:c + 1]
      stack = np.empty(2 * n - 1, dtype = np.int64)
      leaves = np.empty(n, dtype = np.int64)
      x = ll[c]
      for i in range(sweeps):
         x = _hrg_sweep(n, ptr, nbr, parent[c], left[c], right[c], size[c], e[c], lo[c], leafat[c], stack, leaves, state, x)
      for sample in range(samples[c]):
         for i in range(interval):
            x = _hrg_sweep(n, ptr, nbr, parent[c], left[c], right[c], size[c], e[c], lo[c], leafat[c], stack, leaves, state, x)
         for p in range(us.shape[0]):
            if us[p] != vs[p]:
               z = _hrg_lca(parent[c], size[c], us[p], vs[p])
               out[c, p] += e[c, z] / (size[c, left[c, z]] * size[c, right[c, z]])
      ll[c] = x


def hierarchical_random_graph_lp(net, pairs, name = "score", samples = 100, burnin = 5000, interval = 1, chains = None, seed = None):
   """
   Hierarchical random graph score of node pairs (Clauset, Moore and Newman, 2008): the
   probability of an edge between the two nodes in hierarchical random graphs fitted to the
   network.

   A hierarchical random graph is a dendrogram, a binary tree whose leaves are the nodes of the
   network, with a probability :math:`p_r` at each internal node :math:`r`: two nodes are connected
   with the probability of their lowest common ancestor. Nodes in the same branch deep down the
   tree are therefore connected with a probability of their own, which can be high (communities)
   or low (e.g. the two types of a bipartite network), so that, unlike the common neighbor
   scores, the model also predicts edges in disassortative networks. Given a dendrogram, the
   likelihood is largest with :math:`p_r = E_r / (L_r R_r)`, the fraction of the pairs of leaves
   with :math:`r` as lowest common ancestor (one in the left subtree, :math:`L_r` leaves, and one
   in the right one, :math:`R_r` leaves) that are connected (:math:`E_r` edges), and its log is
   :math:`\\sum_r E_r \\log p_r + (L_r R_r - E_r) \\log(1 - p_r)`.

   Dendrograms are sampled in proportion to their likelihood by Markov chain Monte Carlo, from a
   random dendrogram: a move picks an internal node :math:`r` and its parent, whose three subtrees
   below can be arranged in three ways, and proposes one of the other two, accepted with the
   Metropolis rule. Several independent chains, from different random dendrograms, run in
   parallel (one per CPU thread by default) and share the samples. Each chain first needs to reach
   dendrograms of high likelihood: ``burnin`` sweeps of :math:`n` moves (:math:`n` nodes) are
   discarded. Then each chain samples a dendrogram every ``interval`` sweeps, and the score of a
   pair is its mean edge probability over the ``samples`` dendrograms.

   The likelihood climbs slowly, with long plateaus, and needs more sweeps on larger networks: it
   took about 2,000 sweeps to level off on the karate club (34 nodes), 3,000 on a network of 500
   nodes with communities, 5,000 on one of 1,000 nodes, still drifting up slowly after that. With
   the default of 5,000 sweeps, the AUC was 0.86 on 1,000 nodes (0.87-0.88 with 20,000 sweeps)
   and 0.75-0.77 on 2,000 nodes (0.84 with 20,000 sweeps): give a larger ``burnin`` on larger
   networks. The posterior distribution of the dendrograms is rugged, so chains can keep sampling
   at different likelihood levels for a long time, and the convergence tests that compare chains,
   such as the potential scale reduction factor, rarely pass even when the predictions are
   stable: rather, compare the scores, or their AUC, between two seeds.

   The model is undirected and unweighted: directions, weights, parallel edges and self loops are
   ignored. All the nodes of ``net`` are leaves of the dendrograms, isolated ones included. The
   leaves are kept in an order in which every subtree is a range of positions, so that the edges
   between two subtrees, needed by every move, are counted by listing the neighbors of the leaves
   of the smaller one and comparing their positions with the range of the other: a move costs
   the degrees of the leaves of a small subtree, and an accepted move moves the leaves of at most
   two subtrees. Fitting is still much slower than the other scores, since it needs thousands of
   sweeps (see the Performance section of :doc:`link_prediction`). The same dendrograms give
   hierarchical communities: :func:`~graphoskop.hierarchical_random_graph_communities`.

   :param net: A :class:`~graphoskop.Network`, not multilayer.
   :param pairs: Pairs, as in :func:`preferential_attachment_lp`.
   :param name: Name of the score column.
   :param samples: Number of sampled dendrograms.
   :param burnin: Number of sweeps of each chain before its first sample.
   :param interval: Number of sweeps between samples.
   :param chains: Number of chains, ``None`` for the number of Numba threads.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :return: Scores, as in :func:`preferential_attachment_lp`; 0 for a pair of a node with itself.
   :raises KeyError: If a node does not exist.
   :raises ValueError: If the network is multilayer, a pair does not have two nodes, or
      ``samples``, ``burnin`` or ``interval`` are not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      train, validation = gsk.link_prediction_split(net, seed = 1)
      validation = gsk.hierarchical_random_graph_lp(train, validation, samples = 200, seed = 1)
      gsk.link_prediction_evaluation(validation)
   """
   _check(net)
   if samples < 1 or burnin < 0 or interval < 1:
      raise ValueError(f"need samples >= 1, burnin >= 0 and interval >= 1, got {samples}, {burnin}, {interval}")
   s, t = _endpoints(pairs)
   us, vs = _positions(net, s), _positions(net, t)
   N, ptr, nbr, counts, states, trees, ll = _hrg_setup(net, samples, chains, seed)
   out = np.zeros((len(counts), len(us)))
   us, vs = us.astype(np.int64), vs.astype(np.int64)
   run = lambda sweeps, k: _hrg_run(N, ptr, nbr, *trees, states, ll, sweeps, interval, k, us, vs, out)
   if N > 2:   # otherwise there is a single dendrogram
      run(int(burnin), np.zeros(len(counts), dtype = np.int64))
   run(0, counts)
   return _scored(pairs, out.sum(0) / samples, name)


def _hrg_setup(net, samples, chains, seed):
   """
   Graph and random starting dendrograms of the chains of a hierarchical random graph: directions,
   weights, parallel edges and self loops ignored, all the nodes of the network as leaves.

   :return: ``(N, ptr, nbr, counts, states, trees, ll)``: number of nodes, CSR of the neighbors,
      number of samples of each chain, random states, tree arrays (``parent``, ``left``,
      ``right``, ``size``, ``e``, ``lo``, ``leafat``, one row per chain) and log-likelihoods.
   """
   N = net.number_of_nodes()
   ei = np.concatenate([d.edge_index.cpu().numpy() + net.node_offsets[c] for c, d in enumerate(net.components)], 1)
   a, b = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
   key = _unique((a * N + b)[a != b])
   src, dst = key // N, key % N
   both = np.concatenate([src, dst])
   order = np.argsort(both, kind = "stable")
   ptr = np.r_[0, np.cumsum(np.bincount(both, minlength = N))]
   nbr = np.concatenate([dst, src])[order]
   C = min(chains or numba.get_num_threads(), samples)
   counts = np.full(C, samples // C, dtype = np.int64)
   counts[:samples % C] += 1
   states = np.random.default_rng(seed).integers(1, 2 ** 63, size = C, dtype = np.uint64)
   m = max(2 * N - 1, 1)
   parent, left, right, size, e, lo = (np.empty((C, m), dtype = np.int64) for _ in range(6))
   leafat, stack = np.empty((C, N), dtype = np.int64), np.empty(m, dtype = np.int64)
   ll = np.array([_hrg_init(N, src, dst, states[c:c + 1], parent[c], left[c], right[c], size[c], e[c], lo[c], leafat[c], stack) for c in range(C)])
   return N, ptr, nbr, counts, states, (parent, left, right, size, e, lo, leafat), ll


@njit(cache = True, parallel = True)
def _hrg_trees(n, ptr, nbr, parent, left, right, size, e, lo, leafat, states, ll, interval, samples, keys, H, LO, SZ, P, LEAF, LL):
   """
   Sample dendrograms with chains already burnt in (as :func:`_hrg_run`): chain ``c`` writes its
   samples from row ``sum(samples[:c])`` on. For every sample, the leaf order (``LEAF``), its
   log-likelihood (``LL``) and, for every internal node, the first position (``LO``) and number
   (``SZ``) of its leaves, its edge probability (``P``) and a hash of its set of leaves (``H``, the
   sum of the random ``keys`` of the leaves, modulo :math:`2^{64}`).
   """
   start = np.zeros(states.shape[0] + 1, dtype = np.int64)
   for c in range(states.shape[0]):
      start[c + 1] = start[c] + samples[c]
   for c in prange(states.shape[0]):
      state = states[c:c + 1]
      stack = np.empty(2 * n - 1, dtype = np.int64)
      leaves = np.empty(n, dtype = np.int64)
      prefix = np.zeros(n + 1, dtype = np.uint64)
      x = ll[c]
      for k in range(start[c], start[c + 1]):
         for i in range(interval):
            x = _hrg_sweep(n, ptr, nbr, parent[c], left[c], right[c], size[c], e[c], lo[c], leafat[c], stack, leaves, state, x)
         for q in range(n):
            prefix[q + 1] = prefix[q] + keys[leafat[c, q]]
            LEAF[k, q] = leafat[c, q]
         for z in range(n, 2 * n - 1):
            a, b = lo[c, z], lo[c, z] + size[c, z]
            H[k, z - n], LO[k, z - n], SZ[k, z - n] = prefix[b] - prefix[a], a, size[c, z]
            P[k, z - n] = e[c, z] / (size[c, left[c, z]] * size[c, right[c, z]])
         LL[k] = x
      ll[c] = x


@njit(cache = True)
def _sbm_h(nodes, l, a, b):
   """Term of the pair of groups ``(a, b)`` in :math:`H`: :math:`\\ln(r + 1) + \\ln \\binom{r}{l}`."""
   r = nodes[a] * nodes[b] if a != b else nodes[a] * (nodes[a] - 1) // 2
   e = l[a, b]
   return math.log(r + 1) + math.lgamma(r + 1) - math.lgamma(e + 1) - math.lgamma(r - e + 1)


@njit(cache = True)
def _sbm_terms(nodes, l, active, G, a, b):
   """Sum of the terms of :math:`H` of all the pairs of groups involving ``a`` or ``b``."""
   h = 0.
   for k in range(G):
      g = active[k]
      if g != a and g != b:
         h += _sbm_h(nodes, l, a, g) + _sbm_h(nodes, l, b, g)
   return h + _sbm_h(nodes, l, a, a) + _sbm_h(nodes, l, b, b) + _sbm_h(nodes, l, a, b)


@njit(cache = True)
def _sbm_move(ptr, nbr, label, nodes, l, active, where, i, b, G):
   """
   Move node ``i`` to group ``b``, updating the group sizes, the links between groups and the
   list of the non-empty groups.

   :return: New number of non-empty groups.
   """
   a = label[i]
   for k in range(ptr[i], ptr[i + 1]):
      g = label[nbr[k]]
      l[a, g] -= 1
      if g != a:
         l[g, a] -= 1
   label[i] = b
   for k in range(ptr[i], ptr[i + 1]):
      g = label[nbr[k]]
      l[b, g] += 1
      if g != b:
         l[g, b] += 1
   nodes[a] -= 1
   nodes[b] += 1
   if nodes[a] == 0:   # a is empty now
      G -= 1
      last = active[G]
      active[where[a]], where[last] = last, where[a]
   if nodes[b] == 1:   # b was empty
      active[G], where[b] = b, G
      G += 1
   return G


@njit(cache = True)
def _sbm_sweep(n, K, ptr, nbr, label, nodes, l, active, where, G, state):
   """
   ``n`` Metropolis moves: a random node to a random group label, accepted with probability
   :math:`\\min(1, e^{-\\Delta H} m'/m)`, where :math:`m = (K - G)!` makes partitions, not labelings,
   equally likely a priori (a partition with :math:`G` groups has :math:`K! / (K - G)!` labelings).

   :return: New number of non-empty groups.
   """
   for step in range(n):
      i = int(_random(state) * n)
      a, b = label[i], int(_random(state) * K)
      if a == b:
         continue
      old = _sbm_terms(nodes, l, active, G, a, b) - math.lgamma(K - G + 1)
      G2 = _sbm_move(ptr, nbr, label, nodes, l, active, where, i, b, G)
      new = _sbm_terms(nodes, l, active, G2, a, b) - math.lgamma(K - G2 + 1)
      d = old - new   # log of the acceptance ratio: -(H' - H) + log m' - log m
      if d >= 0 or _random(state) < math.exp(d):
         G = G2
      else:
         G = _sbm_move(ptr, nbr, label, nodes, l, active, where, i, a, G2)
   return G


@njit(cache = True, parallel = True)
def _sbm_run(n, K, ptr, nbr, states, start, burnin, interval, samples, us, vs, out):
   """
   Independent chains in parallel, from random partitions into ``start`` groups: ``burnin``
   sweeps, then ``samples[c]`` samples every ``interval`` sweeps, adding the reliability
   :math:`(l + 1) / (r + 2)` of the groups of every pair to ``out[c]``.
   """
   for c in prange(states.shape[0]):
      state = states[c:c + 1]
      label = np.empty(n, dtype = np.int64)
      nodes = np.zeros(K, dtype = np.int64)
      l = np.zeros((K, K), dtype = np.int64)
      active, where = np.empty(K, dtype = np.int64), np.full(K, -1)
      for i in range(n):
         label[i] = int(_random(state) * start)
         nodes[label[i]] += 1
      for i in range(n):
         for k in range(ptr[i], ptr[i + 1]):
            l[label[i], label[nbr[k]]] += 1   # each link counted from both ends: l[a, b] and l[b, a]
      for g in range(K):
         l[g, g] //= 2
      G = 0
      for g in range(K):
         if nodes[g] > 0:
            active[G], where[g] = g, G
            G += 1
      for s in range(burnin):
         G = _sbm_sweep(n, K, ptr, nbr, label, nodes, l, active, where, G, state)
      for s in range(samples[c]):
         for t in range(interval):
            G = _sbm_sweep(n, K, ptr, nbr, label, nodes, l, active, where, G, state)
         for p in range(us.shape[0]):
            a, b = label[us[p]], label[vs[p]]
            r = nodes[a] * nodes[b] if a != b else nodes[a] * (nodes[a] - 1) // 2
            out[c, p] += (l[a, b] + 1) / (r + 2)


def block_model_reliability_lp(net, pairs, name = "score", samples = 100, burnin = 1000, interval = 10, groups = None, chains = None, seed = None):
   """
   Stochastic block model reliability of node pairs (Guimerà and Sales-Pardo, 2009): the
   probability that the two nodes are connected, averaged over all the partitions of the nodes
   into groups, each weighted by how well a stochastic block model with those groups explains the
   network. In a block model, two nodes of groups :math:`\\alpha` and :math:`\\beta` are
   connected with a probability that depends only on the groups; with :math:`l_{\\alpha\\beta}`
   links among the :math:`r_{\\alpha\\beta}` pairs of nodes of the two groups, the reliability of
   a pair is

   .. math::

      R_{uv} = \\frac{1}{Z} \\sum_P \\frac{l_{\\sigma_u \\sigma_v} + 1}{r_{\\sigma_u \\sigma_v} + 2}
      \\exp(-H(P)), \\qquad
      H(P) = \\sum_{\\alpha \\le \\beta} \\left[\\ln(r_{\\alpha\\beta} + 1)
      + \\ln \\binom{r_{\\alpha\\beta}}{l_{\\alpha\\beta}}\\right],

   with :math:`\\sigma_u` the group of :math:`u` in the partition :math:`P`. The partitions are
   sampled in proportion to :math:`\\exp(-H(P))` by Markov chain Monte Carlo: a move puts a
   random node in a random group (a new one, possibly), accepted with the Metropolis rule. Each
   chain starts from a random partition into :math:`\\lceil \\sqrt{n} \\rceil` groups, runs
   ``burnin`` sweeps of :math:`n` moves (:math:`n` nodes), then samples a partition every
   ``interval`` sweeps; the score of a pair is its mean reliability over the ``samples``
   partitions. Chains are independent, one per CPU thread by default, and share the samples. Like
   :func:`hierarchical_random_graph_lp`, the model learns any pattern of connection between groups,
   assortative or not.

   The model is undirected and unweighted: neighborhoods are sets, as in
   :func:`common_neighbors_lp` (self loops and parallel edges ignored). Partitions have at most
   ``groups`` groups (by default the number of nodes, at most 1,000): every chain keeps a
   ``groups`` by ``groups`` table of links between groups, and a move costs the degree of the node
   plus the number of non-empty groups. Compiled with Numba, on the CPU.

   :param net: An undirected :class:`~graphoskop.Network`, not multilayer.
   :param pairs: Pairs, as in :func:`preferential_attachment_lp`.
   :param name: Name of the score column.
   :param samples: Number of sampled partitions.
   :param burnin: Number of sweeps of each chain before its first sample.
   :param interval: Number of sweeps between samples.
   :param groups: Largest number of groups, ``None`` for the number of nodes, at most 1,000.
   :param chains: Number of chains, ``None`` for the number of Numba threads.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :return: Scores, as in :func:`preferential_attachment_lp`.
   :raises KeyError: If a node does not exist.
   :raises ValueError: If the network is directed or multilayer, a pair does not have two nodes,
      or ``samples``, ``burnin``, ``interval`` or ``groups`` are not valid.

   Example:

   .. code-block:: python

      validation = gsk.block_model_reliability_lp(train, validation, seed = 1)
   """
   _check(net)
   if net.directed:
      raise ValueError("the block model reliability is defined for undirected networks")
   if samples < 1 or burnin < 0 or interval < 1 or (groups is not None and groups < 1):
      raise ValueError(f"need samples >= 1, burnin >= 0, interval >= 1 and groups >= 1, got {samples}, {burnin}, {interval}, {groups}")
   s, t = _endpoints(pairs)
   us, vs = _positions(net, s), _positions(net, t)
   arcs = _arcs(net, None)
   N = arcs["N"]
   K = int(groups) if groups is not None else min(N, 1000)
   C = min(chains or numba.get_num_threads(), samples)
   counts = np.full(C, samples // C, dtype = np.int64)
   counts[:samples % C] += 1
   states = np.random.default_rng(seed).integers(1, 2 ** 63, size = C, dtype = np.uint64)
   out = np.zeros((C, len(us)))
   start = min(K, int(math.ceil(math.sqrt(N))))
   _sbm_run(N, K, arcs["out_ptr"].cpu().numpy(), arcs["dst"].cpu().numpy(), states, start, int(burnin), int(interval), counts, us.astype(np.int64), vs.astype(np.int64), out)
   return _scored(pairs, out.sum(0) / samples, name)


#: Scores that :func:`ensemble_link_prediction` accepts by name.
_SCORES = ("preferential_attachment_lp", "common_neighbors_lp", "common_neighbors_jaccard_lp", "adamic_adar_lp", "resource_allocation_lp", "car_index_lp", "mutual_information_index_lp", "local_random_walk_lp", "simrank_lp", "vertex_similarity_lp", "hierarchical_random_graph_lp", "block_model_reliability_lp")


@njit(cache = True)
def _local_kemeny(R, order):
   """
   Local Kemenization (Dwork et al., 2001): insert the pairs one at a time, in the given order,
   moving each one up past the pairs above it as long as a strict majority of the rankings puts it
   higher.

   :param R: Ranks of the pairs (one row per pair, one column per score, higher is better).
   :param order: Starting order, best first.
   :return: Final order, best first.
   """
   out = np.empty(order.shape[0], dtype = np.int64)
   m = R.shape[1]
   for k in range(order.shape[0]):
      x = order[k]
      j = k
      while j > 0:
         y = out[j - 1]
         votes = 0
         for c in range(m):
            if R[x, c] > R[y, c]:
               votes += 1
            elif R[x, c] < R[y, c]:
               votes -= 1
         if votes <= 0:
            break
         out[j] = y
         j -= 1
      out[j] = x
   return out


def ensemble_link_prediction(net, pairs, methods = None, aggregation = "borda", columns = None, name = "score"):
   """
   Ensemble of link prediction scores: several scores rank the pairs, and their rankings are
   aggregated into one.

   The scores are either computed, by giving ``methods``, a list of score functions of this
   module (e.g. :func:`common_neighbors_lp`), their names (e.g. ``"adamic_adar_lp"``), or pairs
   ``(function, keyword arguments)`` (e.g. ``(gsk.simrank_lp, {"seed": 1})``), each run on ``net``
   and stored in a column named after the function (with ``_2``, ``_3``... for repeated
   functions); or taken from ``pairs``, a :class:`pandas.DataFrame` with the endpoints in its
   first two columns (as ``source`` and ``target``) and scores in others: the columns named in
   ``columns``, or all the other numeric columns except ``label``. ``net`` is then not needed
   (``None``). Both can be combined: computed scores and the given ``columns``.

   Only the order of each score matters, so scores on different scales combine. Each score ranks
   the pairs from the lowest to the highest, ties sharing their average rank, NaN ranked lowest.

   * ``aggregation = "borda"``: the Borda count, the sum over the scores of the number of pairs
     ranked below the pair (ties counting one half).
   * ``aggregation = "kendall"``: the ranking that minimizes the total Kendall tau distance to the
     rankings of the scores (the number of pairs of pairs they order differently), the Kemeny
     ranking. Finding it is NP-hard, so it is approximated by local Kemenization (Dwork, Kumar,
     Naor and Sivakumar, 2001): from the Borda ranking, each pair moves up past the pairs above it
     as long as a strict majority of the scores ranks it higher. The result is locally Kemeny
     optimal (no two consecutive pairs are ordered against the majority), and every move brings it
     closer to the scores, so it is at least as close as the Borda order it starts from (Borda
     ties broken by the order of the pairs). Unlike the Borda count it leaves no ties, except
     between pairs that all the scores rank the same, so where the scores disagree evenly it can
     be farther from them than the Borda count with its ties. The score is the position from the
     bottom (the best pair scores the number of pairs), averaged over pairs that all the scores
     rank the same. Compiled with Numba; the time grows with the number of moves, small when the scores
     mostly agree.

   :param net: A :class:`~graphoskop.Network` to compute ``methods`` in, or ``None``.
   :param pairs: Pairs, as in :func:`preferential_attachment_lp`; a :class:`pandas.DataFrame` of
      scores without ``methods``.
   :param methods: Scores to compute, or ``None``.
   :param aggregation: ``"borda"`` or ``"kendall"``.
   :param columns: Names of score columns of ``pairs`` to aggregate as well, or ``None``.
   :param name: Name of the ensemble score column.
   :return: A float for one pair; otherwise a :class:`pandas.DataFrame`: the pairs (a copy of the
      table, or columns ``source`` and ``target``), the computed scores, and the column ``name``.
   :raises ValueError: If ``aggregation`` is not valid, if there are no scores to aggregate, if a
      method is not a score of this module, or if ``methods`` are given without ``net``.
   :raises KeyError: If a column does not exist.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      train, validation = gsk.link_prediction_split(net, seed = 1)
      methods = [gsk.common_neighbors_lp, "resource_allocation_lp", (gsk.local_random_walk_lp, {"steps": 3})]
      validation = gsk.ensemble_link_prediction(train, validation, methods, name = "borda")
      validation = gsk.ensemble_link_prediction(None, validation, aggregation = "kendall",
                                                columns = ["common_neighbors_lp", "resource_allocation_lp", "local_random_walk_lp"], name = "kemeny")
      gsk.link_prediction_evaluation(validation, score = "kemeny")
   """
   if aggregation not in ("borda", "kendall"):
      raise ValueError(f"unknown aggregation: {aggregation}")
   if methods is not None and net is None:
      raise ValueError("methods are computed on a network: give net")
   one = isinstance(pairs, tuple)
   if isinstance(pairs, pd.DataFrame):
      table = pairs.copy()
   else:
      s, t = _endpoints(pairs)
      table = pd.DataFrame({"source": s, "target": t})
   names = list(columns or [])
   for c in names:
      if c not in table.columns:
         raise KeyError(f"no score column {c!r}")
   module = globals()
   for m in methods or []:
      f, kw = m if isinstance(m, tuple) else (m, {})
      f = module.get(f) if isinstance(f, str) else f
      if getattr(f, "__name__", None) not in _SCORES or module[f.__name__] is not f:
         raise ValueError(f"{m!r} is not a link prediction score; scores: {list(_SCORES)}")
      col, k = f.__name__, 1
      while col in table.columns:
         k += 1
         col = f"{f.__name__}_{k}"
      table = f(net, table, name = col, **kw)
      names.append(col)
   if methods is None and columns is None:
      skip = set(table.columns[:2]) | {"label", name}
      names = [c for c in table.columns if c not in skip and pd.api.types.is_numeric_dtype(table[c])]
   if not names:
      raise ValueError("no scores to aggregate")
   X = table[names].to_numpy(dtype = np.float64)
   R = np.column_stack([rankdata(np.where(np.isnan(x), -np.inf, x)) for x in X.T])   # 1 for the lowest
   borda = (R - 1).sum(1)
   if aggregation == "borda":
      score = borda
   else:
      order = _local_kemeny(R, np.argsort(-borda, kind = "stable"))
      score = np.empty(len(order))
      score[order] = len(order) - np.arange(len(order), dtype = np.float64)
      Ro = R[order]
      start = np.r_[0, np.flatnonzero((Ro[1:] != Ro[:-1]).any(1)) + 1]   # runs of pairs ranked the same by every score
      group = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(order)]))
      score[order] = (np.bincount(group, score[order]) / np.bincount(group))[group]
   if one:
      return float(score[0])
   table[name] = score
   return table


def link_prediction_sign_evaluation(net, validation, sign = SIGN, predictions = False):
   """
   How well social balance and social status predict the signs of the hidden edges of a signed
   network (Leskovec, Huttenlocher and Kleinberg, 2010): only the positives of the validation set
   (pairs that are connected in the full network) are evaluated, against their sign, from the
   two-step paths :math:`u - z - v` through their common neighbors in the training network.

   * Social balance ("the friend of my enemy is my enemy"): a path predicts the product of the
     signs of its two edges, which would close a balanced triangle; directions are ignored. The
     vote of a pair is the sum of these products over all the paths, all the edges between
     :math:`u` and :math:`z` times all those between :math:`z` and :math:`v`.
   * Social status (directed networks): a positive edge :math:`a \\to b` means that :math:`a`
     regards :math:`b` as higher in status, a negative one as lower, so it says
     :math:`\\sigma_b - \\sigma_a = \\pm 1`. Each path gives :math:`\\sigma_v - \\sigma_u`, the sum
     of the differences along its two edges (followed in either direction), and the vote of a
     pair is the sum over the paths: the edge :math:`u \\to v` is predicted positive if
     :math:`v` ranks higher.

   A pair is predicted by the sign of its vote, and not predicted if the vote is 0 (no common
   neighbor, or a tie). Signs 0 or NaN in the training network count as absent edges.

   :param net: The training network, a signed :class:`~graphoskop.Network` (see :doc:`signed`),
      not multilayer.
   :param validation: Validation set from :func:`link_prediction_split` of the signed network:
      columns ``source``, ``target``, ``label`` and the sign of the positives.
   :param sign: Name of the sign edge attribute and of the sign column of ``validation``.
   :param predictions: Whether to also return the votes of every evaluated pair.
   :return: Dictionary with ``"edges"`` (number of evaluated edges), ``"balance_accuracy"`` and
      ``"balance_coverage"`` (fraction of correct signs among the predicted edges, and fraction
      of the edges predicted), ``"status_accuracy"`` and ``"status_coverage"`` (NaN in undirected
      networks), and ``"majority_accuracy"``, the accuracy of predicting every edge with the
      most common sign of the training network, as a baseline. With ``predictions``, also
      ``"predictions"``: the evaluated rows of ``validation`` with columns ``balance`` and
      ``status`` (the votes).
   :raises ValueError: If the network is multilayer, or not signed.
   :raises KeyError: If ``validation`` has no ``sign`` column, or a node does not exist.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("signed_edges.tsv", directed = True)   # edge attribute "sign"
      train, validation = gsk.link_prediction_split(net, seed = 1)
      gsk.link_prediction_sign_evaluation(train, validation)
   """
   _check(net)
   if sign not in net.edge_attributes:
      raise ValueError(f"not a signed network: no edge attribute {sign!r}")
   if sign not in validation.columns:
      raise KeyError(f"the validation set has no {sign!r} column: split the signed network with link_prediction_split")
   rows = validation[(validation["label"] == 1) & validation[sign].isin([1, -1])]
   N = net.number_of_nodes()
   a, b, x = [], [], []
   for c in range(len(net.components)):
      ei, X = net._input_edges(c)
      ei = ei + int(net.node_offsets[c])
      a.append(ei[0])
      b.append(ei[1])
      x.append(X[:, net.edge_attributes.index(sign)].double())
   a, b, x = torch.cat(a), torch.cat(b), torch.cat(x)
   ok = (a != b) & ((x == 1) | (x == -1))
   a, b, x = a[ok], b[ok], x[ok]
   one = torch.ones_like(x)
   # both ends of every edge: count, sum of signs, status difference sigma(other end) - sigma(this end)
   values = torch.cat([torch.stack([one, x, x], 1), torch.stack([one, x, -x], 1)])
   table = _arc_table(N, torch.cat([a, b]), torch.cat([b, a]), values)
   device = table["key"].device
   u = torch.as_tensor(_positions(net, rows["source"].tolist()), dtype = torch.int64, device = device)
   v = torch.as_tensor(_positions(net, rows["target"].tolist()), dtype = torch.int64, device = device)
   balance = torch.zeros(u.shape[0], dtype = torch.float64, device = device)
   status = torch.zeros(u.shape[0], dtype = torch.float64, device = device)
   V = table["values"]
   for p, z, _, _, i_uz, i_zv in _hits(table, u, v):
      balance.index_add_(0, p, V[i_uz, 1] * V[i_zv, 1])
      status.index_add_(0, p, V[i_zv, 0] * V[i_uz, 2] + V[i_uz, 0] * V[i_zv, 2])
   truth = rows[sign].to_numpy(dtype = np.float64)
   score = lambda vote: (float((np.sign(vote) == truth)[vote != 0].mean()) if (vote != 0).any() else np.nan, float((vote != 0).mean()) if len(vote) else np.nan)
   bal, sta = balance.cpu().numpy(), status.cpu().numpy()
   majority = 1. if bool((x > 0).sum() >= (x < 0).sum()) else -1.
   result = {"edges": len(rows)}
   result["balance_accuracy"], result["balance_coverage"] = score(bal)
   result["status_accuracy"], result["status_coverage"] = score(sta) if net.directed else (np.nan, np.nan)
   result["majority_accuracy"] = float((truth == majority).mean()) if len(rows) else np.nan
   if predictions:
      out = rows.copy()
      out["balance"] = bal
      out["status"] = sta if net.directed else np.nan
      result["predictions"] = out
   return result


def _thresholds(x, y):
   """
   True positives above each threshold, one threshold per distinct score, from the highest.

   :param x: Scores (float64 NumPy array, no NaN).
   :param y: Labels (boolean NumPy array).
   :return: ``(tp, n)``: number of positives and of pairs scored at or above each threshold.
   """
   order = np.argsort(-x, kind = "stable")
   xs, tp = x[order], np.cumsum(y[order])
   last = np.r_[np.flatnonzero(xs[1:] != xs[:-1]), len(xs) - 1]   # end of each group of equal scores (infinite ones included)
   return tp[last], last + 1


def _curves(tp, n, p, q, points):
   """
   ROC and precision-recall curves at evenly spaced points.

   :param tp: Positives at or above each threshold (see :func:`_thresholds`).
   :param n: Pairs at or above each threshold.
   :param p: Number of positives.
   :param q: Number of negatives.
   :param points: Number of points between 0 and 1.
   :return: Two :class:`pandas.DataFrame`: ``fpr`` and ``tpr``, ``recall`` and ``precision``.
   """
   fpr, tpr = np.r_[0, (n - tp) / q], np.r_[0, tp / p]
   top = np.r_[fpr[1:] != fpr[:-1], True]   # vertical segments: keep their top point
   grid = np.linspace(0, 1, points)
   roc = pd.DataFrame({"fpr": grid, "tpr": np.interp(grid, fpr[top], tpr[top])})
   first = np.searchsorted(tp / p, grid - 1e-12)   # first threshold reaching each recall
   pr = pd.DataFrame({"recall": grid, "precision": (tp / n)[first]})
   return roc, pr


def link_prediction_evaluation(validation, score = "score", label = "label", curves = False, points = 101):
   """
   How well a score separates the positives from the negatives of a validation set.

   * ``auc``: area under the ROC curve, the probability that a random positive has a higher score
     than a random negative (ties count one half); 0.5 for a random score, 1 for a perfect one.
   * ``average_precision``: area under the precision-recall curve, the precision averaged over the
     positives, each at the threshold of its score (pairs with equal scores enter together, as in
     scikit-learn); the fraction of positives for a random score, 1 for a perfect one.

   With ``curves = True``, also the two curves, at ``points`` evenly spaced values of their x
   between 0 and 1, as tables ready to plot:

   * ROC curve: columns ``fpr`` (false positive rate, the fraction of the negatives scored at or
     above a threshold) and ``tpr`` (true positive rate, the same fraction of the positives),
     linearly interpolated between the thresholds (ties give a diagonal segment), with the
     highest ``tpr`` where the curve rises vertically;
   * precision-recall curve: columns ``recall`` (the true positive rate) and ``precision`` (the
     fraction of positives among the pairs at or above a threshold), at each recall the precision
     of the first threshold reaching it: a step function, whose area is ``average_precision``
     (linear interpolation would overestimate precision between thresholds, see Davis and
     Goadrich, 2006).

   Pairs with a NaN score are left out.

   :param validation: :class:`pandas.DataFrame` with a score column and a label column (1 for the
      positives, 0 for the negatives), e.g. from :func:`preferential_attachment_lp`.
   :param score: Name of the score column.
   :param label: Name of the label column.
   :param curves: Whether to also return the ROC and precision-recall curves.
   :param points: Number of points of the curves, the first at 0 and the last at 1.
   :return: Dictionary with keys ``"auc"`` and ``"average_precision"`` (floats) and, with
      ``curves``, ``"roc_curve"`` and ``"precision_recall_curve"`` (:class:`pandas.DataFrame`).
   :raises ValueError: If there are no positives or no negatives, or fewer than 2 points.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      train, validation = gsk.link_prediction_split(net, seed = 1)
      gsk.link_prediction_evaluation(gsk.preferential_attachment_lp(train, validation))
      validation = gsk.adamic_adar_lp(train, validation, name = "aa")
      result = gsk.link_prediction_evaluation(validation, score = "aa", curves = True)
      result["auc"], result["average_precision"]
      result["roc_curve"].plot(x = "fpr", y = "tpr")
      result["precision_recall_curve"].plot(x = "recall", y = "precision")
   """
   x = validation[score].to_numpy(dtype = np.float64)
   y = validation[label].to_numpy()[~np.isnan(x)] == 1
   x = x[~np.isnan(x)]
   p = int(y.sum())
   if p == 0 or p == len(y):
      raise ValueError("the validation set needs both positives and negatives")
   if curves and points < 2:
      raise ValueError(f"curves need at least 2 points, got {points}")
   r = rankdata(x)
   auc = (r[y].sum() - p * (p + 1) / 2) / (p * (len(y) - p))
   tp, n = _thresholds(x, y)
   ap = float(np.sum(np.diff(np.r_[0, tp]) * tp / n) / p)
   result = {"auc": float(auc), "average_precision": ap}
   if curves:
      result["roc_curve"], result["precision_recall_curve"] = _curves(tp, n, p, len(y) - p, points)
   return result
