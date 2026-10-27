"""
Memory in sequences: networks whose nodes are a place together with some of the steps before it,
built from paths (ships between ports, passengers between airports, users between web pages, the
walks of :func:`~graphoskop.random_walk`). In an ordinary network, a walker at ``c`` moves on the
same way whatever it came from; here node ``c|b`` (at ``c``, coming from ``b``) has its own out
edges. A memory network (Rosvall et al., 2014) has a node for every past of a fixed length; a
higher-order dependency network (Xu, Wickramarathne and Chawla, 2016) only for the pasts that
the data show to change the next step, of any length up to a maximum. Both are plain directed
weighted networks: walks, PageRank, Laplacians and communities apply as they are, and
:func:`first_order_values` gathers their results back over the places.
"""

import os
import numpy as np
import pandas as pd
import torch
from ._policy import _higher_order
from .progress import _bar


def _sequences(sequences, sep, labeled):
   """
   Nodes of a set of sequences, concatenated.

   :param sequences: Iterable of sequences, long table, or path of a text file (see
      :func:`higher_order_dependency_network`).
   :param sep: Separator of the nodes in a file (``None``: whitespace).
   :param labeled: Whether every line of a file starts with the id of its sequence (skipped).
   :return: ``(codes, ids, sequence)``: node code at every position, node ids by code, and
      sequence number of every position (positions of a sequence are contiguous, in order).
   """
   if isinstance(sequences, (str, os.PathLike)):
      with open(sequences) as f:
         seqs = [line.split(sep)[int(labeled):] for line in f if line.strip()]
      values = pd.Series([v for s in seqs for v in s], dtype = object)
      number = pd.to_numeric(values, errors = "coerce")
      if len(values) and number.notna().all() and (number % 1 == 0).all():
         values = number.astype(np.int64)
      lengths = [len(s) for s in seqs]
   elif isinstance(sequences, pd.DataFrame):
      if "node" not in sequences:
         raise ValueError("a table of sequences needs a column 'node'")
      keys = [c for c in sequences.columns if c not in ("node", "step")]
      t = sequences.assign(_seq = sequences.groupby(keys, sort = False).ngroup() if keys else 0)
      t = t.sort_values(["_seq", "step"] if "step" in t else "_seq", kind = "stable")
      values = t["node"].reset_index(drop = True)
      lengths = np.bincount(t["_seq"].to_numpy()) if len(t) else []
   else:
      seqs = [list(s) for s in sequences]
      values = pd.Series([v for s in seqs for v in s], dtype = object).infer_objects()
      lengths = [len(s) for s in seqs]
   codes, ids = pd.factorize(values)
   return codes.astype(np.int64), np.asarray(ids, dtype = object), np.repeat(np.arange(len(lengths)), lengths)


def _unique(keys, device):
   """
   Sorted distinct values of an int64 array, the position of every value among them, and the
   number of times each occurs (sorted on ``device``).
   """
   u, inv, cnt = torch.unique(torch.from_numpy(keys).to(device), return_inverse = True, return_counts = True)
   return u.cpu().numpy(), inv.cpu().numpy(), cnt.cpu().numpy()


@_higher_order()
def higher_order_dependency_network(sequences, max_order = 5, min_support = 1, sep = None, device = "cpu", progress = None, labeled = False):
   """
   Higher-order network of a set of sequences (HON, Xu, Wickramarathne and Chawla, 2016), by the
   rule extraction and network wiring of the paper (Algorithms 1 and 2, as in the authors' code).

   1. **Observations.** Every subsequence of the data is counted as a source (the last ``k``
      steps, :math:`k \\leq` ``max_order``) followed by a target (the next node). Pairs seen fewer
      than ``min_support`` times are dropped, and the rest give the next-step distribution
      :math:`P(\\cdot | s)` of every source :math:`s`.
   2. **Rules.** Starting from every node, a source is extended by one earlier step at a time. An
      extension :math:`s'` is a new dependency if its distribution differs enough from that of the
      last valid source :math:`s` it extends:

      .. math:: D_{KL}(P(\\cdot | s') \\,\\|\\, P(\\cdot | s)) > \\frac{k'}{\\log_2(1 + n(s'))},

      with :math:`k'` the order of :math:`s'` and :math:`n(s')` the number of its observations: a
      higher order, or less data, needs a larger difference. Extensions that are not new
      dependencies are still followed, in case an even longer past matters.
   3. **Wiring.** Every valid source (and every source on its way, its prefixes) becomes a node,
      named after the reference code: the current node, ``|``, then its past, most recent step
      first, steps separated by ``.``: ``c|`` for node ``c``, ``c|b`` for ``c`` coming from ``b``,
      ``c|b.a`` for ``c`` after ``a`` then ``b`` (``.`` is a separator, not a decimal point). Each of its observed next steps ``t`` becomes an
      edge to the longest node matching the end of the path (``t|c.b`` if it exists, otherwise
      ``t|c``, otherwise ``t|``), so that a walker carries exactly the past that matters.

   Edges have the attributes ``weight`` (the number of observations of the step, as in the paper;
   with the defaults of the analysis functions, the walks follow it) and ``probability`` (the
   weight over the out weight of its node). Nodes have the attributes ``first_order_node`` (the
   place, with the type of the input ids) and ``order`` (the length of the past of the node, 1 for
   first-order nodes). Without higher-order dependencies (or with ``max_order = 1``), the result
   is the first-order network of the steps. Repeated consecutive nodes count as steps (self
   loops); remove them first if they are not moves.

   The paper uses ``max_order = 5`` and ``min_support`` between 5 and 10, to filter noise. The
   default here is ``min_support = 1``, because dropping the rare steps of a past rescales the
   others, so a thinly observed past (say 6 observations: 5 to one node, 1 to another) looks
   certain of its next step and passes the threshold. On paths with no memory at all (walks on
   three random networks of 60 nodes, 2.4 million steps each), ``max_order = 5`` gave 9,000 to
   18,000 spurious higher-order nodes with ``min_support = 5`` and 4 to 90 with 1 (none up to
   ``max_order = 4``); on a tenth of the authors' synthetic data, it gave 200 spurious nodes with
   5 and none with 1 (the planted ones were found either way). More
   generally, the threshold is not a significance test: pasts observed few times compared with
   the number of their next steps can pass it by chance. A check: build the network of as many
   walks of the first-order network (``max_order = 1``, walking with ``weight``) with the same
   settings, and compare the numbers of higher-order nodes.

   All counts are computed at once by sorting (on ``device``): 10 million steps take about 8 s on
   a CPU. The result is on ``device``.

   :param sequences: The paths: an iterable of sequences of node ids (e.g. lists); a
      :class:`pandas.DataFrame` with one row per visit, a column ``node``, an optional column
      ``step`` giving the order, and the other columns identifying the sequence (the output of
      :func:`~graphoskop.random_walk` works as it is); or the path of a text file with one sequence
      per line (ids that all look like integers are read as integers; see ``labeled``).
   :param max_order: Longest past considered (1 gives the first-order network).
   :param min_support: Fewest observations for a step to count (see above before raising it).
   :param sep: Separator of the nodes in a file (``None``: any whitespace).
   :param device: Device of the counts and of the result.
   :param progress: Whether to show a progress bar (reading, each order, wiring; ``None``: as set
      by :func:`~graphoskop.set_progress`; see :doc:`progress`).
   :param labeled: Whether every line of the file starts with the id of its sequence, as in the
      authors' data (``ship port port ...``): the id is skipped. Without it, the id would be read
      as the first node of the path, giving nodes such as ``35|34.99388`` (at 35, coming from 34,
      coming from "node" 99388, the trajectory id).
   :return: Directed :class:`~graphoskop.Network`.
   :raises ValueError: If ``max_order`` < 1, or no step is left.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      paths = [["a", "c", "d"], ["b", "c", "e"]] * 10   # after c, the next step depends on the past
      hon = gsk.higher_order_dependency_network(paths)
      sorted(hon.node_ids[0])        # ['a|', 'b|', 'c|', 'c|a', 'c|b', 'd|', 'e|']
      list(hon.edges(labels = True)) # a| -> c|a -> d|, b| -> c|b -> e|; c| (no in edges) keeps both
      pr = gsk.pagerank(hon, weight = "weight")
      gsk.first_order_values(hon, pr)                     # PageRank of a, b, c, d, e

      walks = gsk.random_walk(net, list(net.node_ids[0]), 2000, walks = 20)
      fon = gsk.higher_order_dependency_network(walks, max_order = 1)   # the first-order network
      hon = gsk.higher_order_dependency_network(walks, max_order = 3)   # no memory: (almost) no new node
   """
   if max_order < 1:
      raise ValueError("max_order must be at least 1")
   K = int(max_order)
   with _bar(K + 2, "higher_order_dependency_network", progress) as bar:
      x, ids, seq = _sequences(sequences, sep, labeled)
      bar.update()
      levels = _observations(x, seq, len(ids), K, max(min_support, 1), device, bar)
      src, tgt, w, nodes, path, level = _wiring(levels, len(ids), K)
      bar.update()
   if len(src) == 0:
      raise ValueError("higher_order_dependency_network: no step is left (sequences too short, or min_support too high)")
   return _network(ids, src, tgt, w, nodes, path, level, device)


def _network(ids, src, tgt, w, nodes, path, level, device, count = None):
   """
   Directed network of the states ``nodes`` (with their paths and orders) and the transitions
   ``src`` -> ``tgt`` observed ``w`` times, named and with attributes as described in
   :func:`higher_order_dependency_network`; ``count``, if given, becomes the node attribute
   ``count``.
   """
   text = [str(v) for v in ids]
   label = np.array([text[p[0]] + "|" + ".".join(text[c] for c in p[1:] if c >= 0) for p in path], dtype = object)
   where = np.searchsorted(nodes, np.concatenate([src, tgt]))
   out = np.bincount(where[:len(src)], weights = w, minlength = len(nodes))
   edges = pd.DataFrame({"source": label[where[:len(src)]], "target": label[where[len(src):]], "weight": w, "probability": w / out[where[:len(src)]]})
   table = pd.DataFrame({"node": label, "first_order_node": pd.Categorical(ids[path[:, 0]]), "order": level.astype(np.int64)})
   if count is not None:
      table["count"] = count
   from .network import load_network
   return load_network(edges, nodes = table, directed = True, device = device)


def _node_paths(L, level, local, K):
   """
   Path of every source (of length ``level``, index ``local`` among those of its length): its
   last node first, then the earlier ones, -1 beyond its length.
   """
   path = np.full((len(local), K), -1, dtype = np.int64)
   cur = local.copy()
   for d in range(K):
      alive = level - d >= 1
      for k in np.unique(level[alive] - d):
         m = alive & (level - d == k)
         path[m, d] = L["last"][k][cur[m]]
         cur[m] = L["prefix"][k][cur[m]] if k > 1 else -1
   return path


def _observations(x, seq, N, K, min_support, device, bar):
   """
   Sources of every length up to ``K`` and their observed next steps.

   :return: Dictionary of lists indexed by length ``k`` (entry 0 unused): ``skey`` (sorted keys
      ``prefix * N + last`` of the sources), ``last``, ``prefix`` and ``suffix`` (index of the
      source without its last or first node, among those of length ``k - 1``), and ``psrc``,
      ``ptgt``, ``pcnt``: the pairs (source, next node) seen at least ``min_support`` times.
   """
   P = len(x)
   shifted = lambda a, j: np.concatenate([a[j:], np.full(min(j, len(a)), -1, dtype = a.dtype)])
   pos_id = [None, x]               # id of the source of length k starting at every position (-1: none)
   L = {"skey": [None, np.arange(N)], "last": [None, np.arange(N)], "prefix": [None, np.full(N, -1)], "suffix": [None, np.full(N, -1)], "psrc": [None], "ptgt": [None], "pcnt": [None]}
   for k in range(1, K + 1):
      step = np.flatnonzero((shifted(seq, k) == seq) & (pos_id[k] >= 0))   # a source of length k, then a step
      u, inv, cnt = _unique(pos_id[k][step] * N + x[step + k], device)
      keep = cnt >= min_support
      L["psrc"].append(u[keep] // N)
      L["ptgt"].append(u[keep] % N)
      L["pcnt"].append(cnt[keep].astype(np.float64))
      if k < K:                     # the sources of length k + 1 are these pairs, before pruning
         ids_k = np.full(P, -1, dtype = np.int64)
         ids_k[step] = inv
         first = np.empty(len(u), dtype = np.int64)
         first[inv[::-1]] = step[::-1]
         pos_id.append(ids_k)
         L["skey"].append(u)
         L["prefix"].append(u // N)
         L["last"].append(u % N)
         L["suffix"].append(pos_id[k][first + 1])
      bar.update()
   return L


def _wiring(L, N, K):
   """
   Rules (Algorithm 1) and edges (Algorithm 2) from the observations of :func:`_observations`.

   :return: ``(src, tgt, w, nodes, path, level)``: edges between global source ids and their
      weights, the ids of the nodes, the path of every node (current node first, then the earlier
      ones, -1 beyond its length) and its order.
   """
   skey, last, prefix, suffix, psrc, ptgt, pcnt = (L[key] for key in ("skey", "last", "prefix", "suffix", "psrc", "ptgt", "pcnt"))
   n = [0] + [len(last[k]) for k in range(1, K + 1)]
   off = np.cumsum([0] + n[1:])                               # global source id = off[k - 1] + local
   support = [None] + [np.bincount(psrc[k], weights = pcnt[k], minlength = n[k]) for k in range(1, K + 1)]
   # all observed pairs, by global source, for lookups
   gkey = np.concatenate([(off[k - 1] + psrc[k]) * N + ptgt[k] for k in range(1, K + 1)])
   gcnt = np.concatenate(pcnt[1:])
   gsup = np.concatenate(support[1:])
   order = np.argsort(gkey)
   gkey, gcnt = gkey[order], gcnt[order]
   # Algorithm 1: last valid source of every source, extending one step back at a time
   valid = [None, off[0] + np.arange(n[1])]
   leaf = []
   for k in range(2, K + 1):
      s = psrc[k]
      v = valid[k - 1][suffix[k][s]]
      p = pcnt[k] / support[k][s]
      q = gcnt[np.searchsorted(gkey, v * N + ptgt[k])] / gsup[v]
      kld = np.bincount(s, weights = p * np.log2(p / q), minlength = n[k])
      dist = support[k] > 0
      with np.errstate(divide = "ignore"):
         new = dist & (kld > k / np.log2(1 + support[k]))
      valid.append(np.where(new, off[k - 1] + np.arange(n[k]), valid[k - 1][np.maximum(suffix[k], 0)]))
      extended = np.zeros(n[k - 1], dtype = bool)
      extended[suffix[k][dist]] = True
      leaf.append(valid[k - 1][(support[k - 1] > 0) & ~extended])
   leaf.append(valid[K][support[K] > 0])
   rule = np.zeros(off[-1], dtype = bool)
   rule[np.concatenate(leaf)] = True
   rule[off[0]:off[1]] |= support[1] > 0
   for k in range(K, 1, -1):                                  # prefixes of rules are rules
      r = np.flatnonzero(rule[off[k - 1]:off[k]])
      rule[off[k - 2] + prefix[k][r]] = True
   # Algorithm 2: every observed step of a rule source, to the longest rule matching the end of the path
   src, tgt, w = [], [], []
   for k in range(1, K + 1):
      keep = rule[off[k - 1] + psrc[k]]
      s, t = psrc[k][keep], ptgt[k][keep]
      to = off[0] + t                                         # first-order node t by default
      cur, lev, done = s.copy(), k, np.zeros(len(s), dtype = bool)
      while lev >= 1:
         if lev + 1 <= K and n[lev + 1]:
            j = np.searchsorted(skey[lev + 1], cur * N + t).clip(max = n[lev + 1] - 1)
            hit = ~done & (skey[lev + 1][j] == cur * N + t) & rule[off[lev] + j]
            to = np.where(hit, off[lev] + j, to)
            done |= hit
         if lev > 1:
            cur[~done] = suffix[lev][cur[~done]]
         lev -= 1
      src.append(off[k - 1] + s)
      tgt.append(to)
      w.append(pcnt[k][keep])
   src, tgt, w = np.concatenate(src), np.concatenate(tgt), np.concatenate(w)
   # paths of the nodes
   nodes = np.unique(np.concatenate([src, tgt]))
   level = np.searchsorted(off, nodes, side = "right")
   path = _node_paths(L, level, nodes - off[level - 1], K)
   return src, tgt, w, nodes, path, level


@_higher_order()
def memory_network(sequences, order = 2, sep = None, labeled = False, device = "cpu", progress = None):
   """
   Memory network of a set of sequences (Rosvall, Esquivel, Lancichinetti, West and Lambiotte,
   2014): the network of a Markov chain of fixed order :math:`m` on the paths. Its nodes (memory,
   or state, nodes) are the observed sequences of :math:`m` steps, named as in
   :func:`higher_order_dependency_network`: with ``order = 2``, ``k|j`` is being at ``k`` coming
   from ``j``, the link :math:`\\overrightarrow{jk}` of the first-order network. Every observed
   :math:`m + 1` steps (trigrams :math:`i \\to j \\to k` for ``order = 2``) add one to the weight
   of the edge from ``j|i`` to ``k|j``, so the walker moves with

   .. math:: p(\\overrightarrow{ij} \\to \\overrightarrow{jk}) = \\frac{W(\\overrightarrow{ij}
      \\to \\overrightarrow{jk})}{\\sum_l W(\\overrightarrow{ij} \\to \\overrightarrow{jl})}.

   Unlike :func:`higher_order_dependency_network`, every past of length :math:`m` is a node,
   whether it changes the next step or not: the model is simpler, and larger (one node per link
   of the first-order network for ``order = 2``). As in the paper, a memory node never left in
   the data (the last steps of paths) moves as its physical node does in the data of one order
   less (``k|j`` follows the observed steps out of ``k``, to the nodes ``l|k``), so that the
   walker does not get stuck. ``order = 1`` gives the first-order network.

   Edges have the attributes ``weight`` (observations) and ``probability``; nodes have
   ``first_order_node`` (the physical node), ``order`` and ``count`` (the observations of the
   :math:`m` steps, i.e. for ``order = 2`` the weight :math:`W(j \\to k)` of the link of the
   first-order network). The analyses of the paper:

   * **Flow and rankings.** PageRank of the memory network, with the teleportation of the paper
     (to memory nodes in proportion to their ``count``), summed over physical nodes:
     ``first_order_values(mem, pagerank(mem, weight = "weight", personalization = "count"))``.
   * **Overlapping modules.** :func:`~graphoskop.infomap_communities` with
     ``physical = "first_order_node"``: memory nodes of the same physical node in a module share
     its codeword, and a physical node is in every module of its memory nodes
     (``first_order_values(mem, "community", how = set)``).

   :param sequences: The paths, as in :func:`higher_order_dependency_network` (iterable of
      sequences, long table, or text file).
   :param order: Order :math:`m` of the Markov chain, the number of steps a memory node holds.
   :param sep: Separator of the nodes in a file (``None``: any whitespace).
   :param labeled: Whether every line of the file starts with the id of its sequence (skipped).
   :param device: Device of the counts and of the result.
   :param progress: Whether to show a progress bar (``None``: as set by
      :func:`~graphoskop.set_progress`; see :doc:`progress`).
   :return: Directed :class:`~graphoskop.Network`.
   :raises ValueError: If ``order`` < 1, or no path has ``order + 1`` steps.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      paths = [["a", "c", "d"], ["b", "c", "e"]] * 10
      mem = gsk.memory_network(paths)
      sorted(n for c in mem.node_ids for n in c)            # ['c|a', 'c|b', 'd|c', 'e|c']
      mem.number_of_components()            # 2: from a, c always leads to d; from b, to e

      mem = gsk.memory_network("trips.txt", labeled = True)   # lines "id place place ..."
      pr = gsk.pagerank(mem, weight = "weight", personalization = "count")
      gsk.first_order_values(mem, pr)                       # flow through c, d, e
      gsk.infomap_communities(mem, weight = "weight", physical = "first_order_node")
      gsk.first_order_values(mem, "community", how = set)   # modules of every physical node
   """
   if order < 1:
      raise ValueError("order must be at least 1")
   m = int(order)
   with _bar(m + 2, "memory_network", progress) as bar:
      x, ids, seq = _sequences(sequences, sep, labeled)
      bar.update()
      L = _observations(x, seq, len(ids), m, 1, device, bar)
      N, n = len(ids), len(L["last"][m])
      key = lambda c, t: np.searchsorted(L["skey"][m], c * N + t)   # the state (c, t), c of length m - 1
      s, t, w = L["psrc"][m], L["ptgt"][m], L["pcnt"][m]
      to = t if m == 1 else key(L["suffix"][m][s], t)
      count = np.bincount(x, minlength = N).astype(np.float64) if m == 1 else L["pcnt"][m - 1]
      if m > 1:   # states never left: the steps out of their physical node, one order less
         stuck = np.flatnonzero(np.bincount(s, minlength = n) == 0)
         c = L["suffix"][m][stuck]
         ptr = np.r_[0, np.cumsum(np.bincount(L["psrc"][m - 1], minlength = len(L["last"][m - 1])))]
         k = ptr[c + 1] - ptr[c]
         j = np.repeat(ptr[c], k) + np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
         s, to, w = np.r_[s, np.repeat(stuck, k)], np.r_[to, key(L["psrc"][m - 1][j], L["ptgt"][m - 1][j])], np.r_[w, L["pcnt"][m - 1][j]]
      bar.update()
   if len(s) == 0:
      raise ValueError(f"memory_network: no path has {m + 1} steps")
   nodes = np.arange(n)
   return _network(ids, s, to, w, nodes, _node_paths(L, np.full(n, m), nodes, m), np.full(n, m), device, count)


@_higher_order()
def first_order_values(net, values, component = 0, how = "sum"):
   """
   Values of the nodes of a network with memory (:func:`higher_order_dependency_network`,
   :func:`memory_network`) gathered over the places they stand for, the nodes of the first-order
   network: e.g. the PageRank of a place is the sum of the PageRank of its nodes (Xu et al., 2016;
   Rosvall et al., 2014), and its communities are those of any of its nodes (overlapping
   communities, with ``how = set``).

   :param net: Network from :func:`higher_order_dependency_network`.
   :param values: Vector over the nodes of the component (in their internal order, as returned
      by the analysis functions), a list of them when ``component`` is ``None``, or the name of a
      node attribute.
   :param component: Index of the component, or ``None`` for all components.
   :param how: How the values of the nodes of a place are combined: any aggregation of
      :meth:`pandas.core.groupby.SeriesGroupBy.agg` (``"sum"``, ``"mean"``, ``"max"``, ``set``,
      ...).
   :return: :class:`pandas.Series` indexed by the first-order nodes.
   :raises KeyError: If ``net`` has no node attribute ``first_order_node``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      hon = gsk.higher_order_dependency_network("trajectories.txt")
      gsk.first_order_values(hon, gsk.pagerank(hon, weight = "weight"))
      gsk.infomap_communities(hon, weight = "weight")
      gsk.first_order_values(hon, "community", how = set)  # every community of every place
   """
   comps = range(len(net.components)) if component is None else [component]
   if isinstance(values, str):
      values = [net.get_node_attribute(values, c) for c in comps]
   elif component is not None:
      values = [values]
   vec = lambda v: np.asarray(v.detach().cpu().numpy() if torch.is_tensor(v) else v)
   place = np.concatenate([net.get_node_attribute("first_order_node", c) for c in comps])
   return pd.Series(np.concatenate([vec(v) for v in values])).groupby(place, sort = False).agg(how).rename_axis("first_order_node")
