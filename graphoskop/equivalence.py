"""
Equivalence of nodes: how far two nodes have the same position in the network.

Two nodes are structurally equivalent (Lorrain and White, 1971) when they have exactly the same
neighbors; :func:`structural_equivalence` measures how close two nodes are to it. Two nodes are
automorphically equivalent when some relabeling of the nodes keeping all edges (an automorphism)
maps one onto the other: they cannot be told apart by the structure of the network, although their
neighbors can differ. Structurally equivalent nodes are automorphically equivalent;
:func:`automorphic_equivalence` measures how close two nodes are to it. Two nodes are regularly
equivalent when they are joined to equivalent nodes, not necessarily the same ones;
:func:`regular_equivalence` measures how close two nodes are to it with the classical REGE
algorithm, and :func:`path_regular_equivalence` with the path counts of Leicht, Holme and Newman
(2006), a different quantity that mostly reflects proximity.
Structural equivalence implies automorphic equivalence, which implies regular equivalence. The
``pairwise_`` versions give the scores of all pairs of nodes at once, as ``n`` by ``n`` matrices,
much faster than calling the single versions on every pair.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from ._policy import _higher_order


def _neighbor_entries(net, comps, mode, weight):
   """
   Neighbor relations of the nodes of some components, as (node, neighbor, weight) entries:
   successors, predecessors or both (tagged as different neighbors), parallel edges summed.

   :return: ``(node, neighbor key, weight, number of nodes)``, tensors on the device of the
      network; neighbor keys are network wide indices, times 2, plus 1 for predecessors with
      ``mode = "both"``; nodes are numbered over the components given.
   """
   offset = np.cumsum([0] + [net.components[c].num_nodes for c in comps])
   a, b, w = [], [], []
   for c, o in zip(comps, offset):
      s, t = net.components[c].edge_index
      x = torch.ones(s.shape[0], dtype = torch.float64, device = s.device) if weight is None else net._numeric_edge(weight, c).double()
      sides = [(s, t, 0)] if not net.directed or mode == "out" else [(t, s, 0)] if mode == "in" else [(s, t, 0), (t, s, 1)]
      for p, q, tag in sides:
         a.append(p + int(o))
         b.append((q + int(o)) * 2 + tag)
         w.append(x)
   a, b, w = torch.cat(a), torch.cat(b), torch.cat(w)
   if bool((w < 0).any()):
      raise ValueError("structural equivalence: weights must be non negative")
   n = int(offset[-1])
   key, inv = torch.unique(a * (2 * n) + b, return_inverse = True)   # sum parallel edges
   w = torch.zeros(len(key), dtype = torch.float64, device = w.device).index_add_(0, inv, w)
   if weight is None:
      w = torch.ones_like(w)
   return key // (2 * n), key % (2 * n), w, n


def _check_mode(mode):
   if mode not in ("both", "out", "in"):
      raise ValueError(f"unknown mode: {mode}")


@_higher_order()
def structural_equivalence(net, u, v, mode = "both", weight = None):
   """
   Degree of structural equivalence of two nodes: the Jaccard coefficient of their sets of
   neighbors, :math:`|N(u) \\cap N(v)| / |N(u) \\cup N(v)|`, 1 if they have the same neighbors and 0
   if they have none in common.

   In directed networks, ``mode`` chooses the neighbors: ``"out"`` (successors), ``"in"``
   (predecessors) or ``"both"``, where a node is equivalent only if it has the same successors and
   the same predecessors (the two sets are compared together: a successor of one and a predecessor
   of the other do not match). With ``weight``, the weighted Jaccard (Ruzicka) coefficient
   :math:`\\sum_x \\min(w_{ux}, w_{vx}) / \\sum_x \\max(w_{ux}, w_{vx})`, with the weights of the
   edges to each neighbor (summed over parallel edges); weights must be non negative.

   Neighbors are taken as they are: a self loop makes a node its own neighbor, and two adjacent
   nodes count each other as neighbors (so the two nodes of an isolated edge score 0, although they
   are interchangeable). Two nodes without neighbors score 0 (as NetworkX's
   ``jaccard_coefficient``). Nodes in different components share no neighbors. In multilayer
   networks neighbors are those of the supra-graph: copies in different layers are different
   neighbors, and the copies of a node in other layers, joined by couplings, are neighbors too. An
   entity stands for all its copies, with the union of their neighbors.

   :param net: A :class:`~graphoskop.Network`.
   :param u: A node (input id, internal id ``(component, index)``, or entity in multilayer
      networks).
   :param v: Another node.
   :param mode: ``"both"``, ``"out"`` or ``"in"``; ignored in undirected networks.
   :param weight: Name of the edge attribute used as weight, or ``None`` for the plain Jaccard
      coefficient.
   :return: The score, a float between 0 and 1.
   :raises ValueError: If ``mode`` is not valid or weights are negative.
   :raises KeyError: If there is no such node.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.structural_equivalence(net, "alice", "bob")
      gsk.structural_equivalence(net, "alice", "bob", weight = "weight")
   """
   _check_mode(mode)
   comps = sorted({c for x in (u, v) for c, _ in net._nodes(x)})
   node, nb, w, _ = _neighbor_entries(net, comps, mode, weight)
   offset = dict(zip(comps, np.cumsum([0] + [net.components[c].num_nodes for c in comps])))
   side = []
   for x in (u, v):
      mask = torch.zeros_like(node, dtype = torch.bool)
      for c, i in net._nodes(x):
         mask |= node == int(offset[c]) + i
      k, inv = torch.unique(nb[mask], return_inverse = True)
      side.append(dict(zip(k.tolist(), torch.zeros(len(k), dtype = torch.float64, device = w.device).index_add_(0, inv, w[mask]).tolist())))
   a, b = side
   keys = set(a) | set(b)
   den = sum(max(a.get(k, 0.), b.get(k, 0.)) for k in keys)
   return sum(min(a.get(k, 0.), b.get(k, 0.)) for k in keys) / den if den > 0 else 0.


@_higher_order()
def pairwise_structural_equivalence(net, component = 0, mode = "both", weight = None, batch_size = 2 ** 24):
   """
   Degree of structural equivalence (see :func:`structural_equivalence`) of all pairs of nodes.

   The overlaps :math:`\\sum_x \\min(w_{ux}, w_{vx})` of all pairs are accumulated over the common
   neighbors: every node :math:`x` contributes to each pair of its neighbors, so the work is
   :math:`\\sum_x d_x^2` (the number of 2-paths), done in blocks of about ``batch_size`` pairs on
   the device of the network; then
   :math:`J_{uv} = M_{uv} / (s_u + s_v - M_{uv})`, with :math:`s` the number (or total weight) of
   neighbors. Pairs of nodes in different components score 0, so a component is computed on its
   own; ``component = None`` gives the matrix of the whole network.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param mode: ``"both"``, ``"out"`` or ``"in"``; ignored in undirected networks.
   :param weight: Name of the edge attribute used as weight, or ``None`` for the plain Jaccard
      coefficient.
   :param batch_size: Largest number of neighbor pairs expanded at once.
   :return: Dense ``n`` by ``n`` float64 tensor on the device of the network, rows and columns
      ordered by internal index (component by component with ``None``).
   :raises ValueError: If ``mode`` is not valid or weights are negative.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      J = gsk.pairwise_structural_equivalence(net)
      c, i = net.node_map["alice"]
      c, j = net.node_map["bob"]
      J[i, j]                                 # if both are in component 0
   """
   _check_mode(mode)
   comps = list(range(len(net.components))) if component is None else [component]
   node, nb, w, n = _neighbor_entries(net, comps, mode, weight)
   dev = w.device
   order = torch.argsort(nb, stable = True)   # group the entries by neighbor
   node, nb, w = node[order], nb[order], w[order]
   key, size = torch.unique_consecutive(nb, return_counts = True)
   start = torch.cumsum(size, 0) - size
   grp = torch.repeat_interleave(torch.arange(len(key), device = dev), size)
   pairs = size[grp]   # pairs per entry: one with each entry of its group
   M = torch.zeros(n * n, dtype = torch.float64, device = dev)
   cum = torch.cumsum(pairs, 0)
   a = 0
   while a < len(node):   # blocks of entries with about batch_size pairs
      z = max(a + 1, int(torch.searchsorted(cum, cum[a] - pairs[a] + batch_size, right = True)))
      e = torch.arange(a, z, device = dev)
      i = torch.repeat_interleave(e, pairs[e])
      j = start[grp[i]] + torch.arange(len(i), device = dev) - torch.repeat_interleave(torch.cumsum(pairs[e], 0) - pairs[e], pairs[e])
      M.index_add_(0, node[i] * n + node[j], torch.minimum(w[i], w[j]))
      a = z
   M = M.view(n, n)
   s = torch.zeros(n, dtype = torch.float64, device = dev).index_add_(0, node, w)
   den = s[:, None] + s[None, :] - M
   return torch.where(den > 0, M / den.clamp(min = 1e-300), torch.zeros_like(M))


def _refine(col, s, t, gen):
   """
   Color refinement (1-dimensional Weisfeiler-Leman): a node's new color is its color with the
   multisets of the colors of its successors and of its predecessors, until no color class splits.
   Multisets are compared by sums of random 64-bit hashes of the colors (collisions are
   vanishingly unlikely, and only coarsen the colors).

   :param col: Initial colors (int64 tensor).
   :param s: Arc sources.
   :param t: Arc targets.
   :param gen: Random generator.
   :return: Stable colors, numbered from 0 in an order that depends only on the structure.
   """
   col = torch.unique(col, return_inverse = True)[1]
   while True:
      k = int(col.max()) + 1
      r = torch.randint(-2 ** 62, 2 ** 62, (2, k), generator = gen)
      zero = torch.zeros(len(col), dtype = torch.long)
      out = zero.index_add(0, s, r[0][col[t]])
      inc = zero.index_add(0, t, r[1][col[s]])
      new = torch.unique(torch.stack([col, out, inc], 1), dim = 0, return_inverse = True)[1]
      if int(new.max()) + 1 == k:
         return col
      col = new


def _automorphism(col, s, t, n, keys, gen):
   """
   Searches an automorphism by individualization and refinement: ``col`` colors two copies of the
   graph (nodes ``0`` to ``n - 1`` and ``n`` to ``2n - 1``); an isomorphism from the first copy to
   the second keeping the colors is an automorphism of the graph.

   :param col: Colors of the two copies.
   :param s: Arc sources of the two copies (those of the first copy first).
   :param t: Arc targets.
   :param n: Number of nodes of the graph.
   :param keys: Sorted arc keys ``source * n + target`` of the graph.
   :param gen: Random generator.
   :return: The automorphism (node ``i`` goes to ``perm[i]``), or ``None`` if none exists.
   """
   col = _refine(col, s, t, gen)
   a, b = col[:n], col[n:]
   ca, cb = torch.bincount(a, minlength = int(col.max()) + 1), torch.bincount(b, minlength = int(col.max()) + 1)
   if not torch.equal(ca, cb):
      return None
   if int(ca.max()) == 1:   # one node per color: the candidate map, checked on the arcs
      perm = torch.empty(n, dtype = torch.long)
      perm[torch.argsort(a)] = torch.argsort(b)
      g, h = perm[s[:len(keys)]], perm[t[:len(keys)]]
      return perm if torch.equal(torch.sort(g * n + h).values, keys) else None
   cls = int(torch.where(ca > 1, ca, n + 1).argmin())   # smallest class with several nodes
   x = int((a == cls).nonzero()[0])
   for y in (b == cls).nonzero()[:, 0].tolist():
      new = col.clone()
      new[x] = new[n + y] = int(col.max()) + 1
      perm = _automorphism(new, s, t, n, keys, gen)
      if perm is not None:
         return perm
   return None


def _twins(n, s, t, col):
   """
   Twin reduction: nodes with the same color and the same successors and predecessors (false
   twins), or the same ones counting themselves (true twins, joined to each other), can be
   swapped by an automorphism, so each class of twins becomes a single node, colored by its color,
   size and kind; automorphisms of the reduced graph keeping colors lift to the graph. Searching
   on it avoids individualizing twins one by one. Neighbor sets are compared by sums of random
   64-bit hashes.

   :return: ``(class of every node, number of classes, arc sources, arc targets, sorted arc keys,
      colors)`` of the reduced graph.
   """
   h = torch.randint(-2 ** 62, 2 ** 62, (2, n), generator = torch.Generator().manual_seed(1))
   zero = torch.zeros(n, dtype = torch.long)
   sig = zero.index_add(0, s, h[0][t]) + zero.index_add(0, t, h[1][s])   # open neighborhood
   group = lambda x: torch.unique(torch.stack([col, x], 1), dim = 0, return_inverse = True)[1]
   false = group(sig)
   true = group(sig + h[0] + h[1])   # closed neighborhood
   single = torch.bincount(false)[false] == 1
   cls = torch.unique(torch.stack([single.long(), torch.where(single, true, false)], 1), dim = 0, return_inverse = True)[1]
   k = int(cls.max()) + 1
   size = torch.bincount(cls, minlength = k)
   kind, color = torch.zeros(k, dtype = torch.long), torch.zeros(k, dtype = torch.long)
   kind[cls] = single.long()   # 0: false twins (no edges inside), 1: true twins (a clique) or a single node
   color[cls] = col
   key = torch.unique(cls[s] * k + cls[t])
   key = key[key // k != key % k]
   return cls, k, key // k, key % k, key, torch.unique(torch.stack([color, size, kind], 1), dim = 0, return_inverse = True)[1]


def _equivalent(red, x, y, gen):
   """
   Whether some automorphism maps class ``x`` onto class ``y`` of the reduced graph.

   :param red: Reduced graph, as returned by :func:`_twins`.
   :return: The automorphism of the reduced graph, or ``None``.
   """
   import sys
   _, k, s, t, keys, col = red
   init = torch.cat([col, col])
   init[x] = init[k + y] = int(col.max()) + 1
   limit = sys.getrecursionlimit()
   sys.setrecursionlimit(max(limit, 10 * k + 1000))
   try:
      return _automorphism(init, torch.cat([s, s + k]), torch.cat([t, t + k]), k, keys, gen)
   finally:
      sys.setrecursionlimit(limit)


def _simple(net, comps):
   """
   Simple directed graph (arcs both ways for undirected edges) of some components, without self
   loops, with the initial colors of the automorphisms (self loop, layer).

   :return: ``(n, arc sources, arc targets, sorted arc keys, colors, offsets)``, on the CPU.
   """
   offset = np.cumsum([0] + [net.components[c].num_nodes for c in comps])
   n = int(offset[-1])
   s = torch.cat([net.components[c].edge_index[0].cpu() + int(o) for c, o in zip(comps, offset)])
   t = torch.cat([net.components[c].edge_index[1].cpu() + int(o) for c, o in zip(comps, offset)])
   col = torch.zeros(n, dtype = torch.long)
   col[s[s == t]] = 1
   if net.layer_labels is not None:
      col += 2 * torch.cat([net.components[c].layer.cpu() for c in comps])
   keep = s != t
   keys = torch.unique(s[keep] * n + t[keep])
   return n, keys // n, keys % n, keys, col, offset


def _distances(n, s, t, directed, sources):
   """
   Distances from the sources to all nodes, and to them in directed networks; unreachable: -1.

   :return: int64 array of shape ``(len(sources), n, 1 or 2)``.
   """
   g = csr_matrix((np.ones(len(s)), (s.numpy(), t.numpy())), shape = (n, n))
   d = [shortest_path(g, unweighted = True, indices = sources)] + ([shortest_path(g.T.tocsr(), unweighted = True, indices = sources)] if directed else [])
   return np.stack([np.where(np.isfinite(x), x, -1).astype(np.int64).reshape(len(sources), n) for x in d], 2)


def _profiles(d, extra):
   """
   Distance profiles: for each source, the number of nodes at each distance (pair of distances
   from and to it in directed networks).

   :param d: Distances, as given by :func:`_distances`.
   :param extra: Further unreachable nodes counted for every source (nodes of other components).
   :return: Counts, ``len(sources)`` by number of distinct distance values.
   """
   B, n, k = d.shape
   base = int(d.max()) + 2
   code = sum((d[:, :, j] + 1) * base ** j for j in range(k))   # one integer per distance pair
   values, inv = np.unique(code, return_inverse = True)
   rows = np.stack([values // base ** j % base - 1 for j in range(k)], 1)
   P = np.bincount(np.repeat(np.arange(B), n) * len(rows) + inv.ravel(), minlength = B * len(rows)).reshape(B, len(rows))
   if extra:
      unreach = np.flatnonzero((rows == -1).all(1))
      if len(unreach):
         P[:, unreach[0]] += extra
      else:
         P = np.concatenate([P, np.full((B, 1), extra)], 1)
   return P


@_higher_order()
def automorphic_equivalence(net, u, v, seed = 0):
   """
   Degree of automorphic equivalence of two nodes (Borgatti and Everett, 1992): 1 if some
   relabeling of the nodes that keeps all edges (an automorphism) maps one onto the other, so that
   they have the same position in the structure of the network; less the more their positions
   differ.

   As in the role similarity of Everett and Borgatti (1988), nodes are compared by their profiles
   of geodesic distances: the number of nodes at each distance from the node (pairs of distances
   from and to it in directed networks; unreachable nodes, also in other components, count at
   distance infinity). An automorphism keeps distances, so automorphically equivalent nodes have
   the same profile. The score is the weighted Jaccard coefficient of the two profiles,
   :math:`\\sum_k \\min(P_{uk}, P_{vk}) / \\sum_k \\max(P_{uk}, P_{vk})` (Everett and Borgatti compare
   them by Euclidean distance; this puts the comparison between 0 and 1). When the profiles are
   equal, an automorphism mapping ``u`` onto ``v`` is searched exactly, by individualization and
   refinement (the strategy of nauty) on two copies of the network, one with ``u`` marked and one
   with ``v``: if it exists the score is 1, otherwise (equal distance profiles without
   equivalence) it is :math:`1 - 1 / n`, with :math:`n` the number of nodes of the network, so that
   1 always means automorphic equivalence.

   The automorphisms considered keep edge directions, self loops and, in multilayer networks, the
   layer of every copy; weights and edge multiplicities are ignored. The whole network is used
   (automorphisms can swap isomorphic components). The profiles take two breadth first searches;
   the search for an automorphism is usually immediate, but can take exponential time on rare,
   very symmetric networks. It runs on the CPU. A node is equivalent to itself (score 1).

   :param net: A :class:`~graphoskop.Network`.
   :param u: A node (input id or internal id ``(component, index)``; in multilayer networks a copy
      ``(entity, layer)``).
   :param v: Another node.
   :param seed: Seed of the random hashes of refinement.
   :return: The score, a float between 0 and 1.
   :raises ValueError: If a node stands for several copies.
   :raises KeyError: If there is no such node.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.automorphic_equivalence(net, "alice", "bob")
   """
   n, s, t, keys, col, offset = _simple(net, list(range(len(net.components))))
   ids = []
   for x in (u, v):
      copies = net._nodes(x)
      if len(copies) != 1:
         raise ValueError(f"automorphic equivalence: {x} stands for {len(copies)} copies; give one copy (entity, layer)")
      ids.append(int(offset[copies[0][0]]) + copies[0][1])
   iu, iv = ids
   if iu == iv:
      return 1.
   P = _profiles(_distances(n, s, t, net.directed, [iu, iv]), 0)
   score = np.minimum(P[0], P[1]).sum() / np.maximum(P[0], P[1]).sum()
   if score < 1:
      return float(score)
   red = _twins(n, s, t, col)
   x, y = int(red[0][iu]), int(red[0][iv])
   found = x == y or _equivalent(red, x, y, torch.Generator().manual_seed(seed)) is not None
   return 1. if found else 1 - 1 / n


@_higher_order()
def pairwise_automorphic_equivalence(net, component = 0, seed = 0, batch_size = 2 ** 24):
   """
   Degree of automorphic equivalence (see :func:`automorphic_equivalence`) of all pairs of nodes.

   The distance profiles of all nodes come from one breadth first search per node (on the CPU).
   Every profile counts all :math:`N` nodes of the network, so the weighted Jaccard coefficient of
   two profiles is :math:`M / (2N - M)`, with :math:`M` the sum of their minima, computed for all
   pairs in blocks of rows (about ``batch_size`` numbers at once) on the device of the network.
   Pairs with equal profiles are then split into automorphism orbits: twins (nodes with the same
   neighbors) are equivalent and merged first, the other nodes are separated by color refinement,
   which automorphisms keep, and each automorphism found by the exact search is applied to all
   nodes at once (it maps every node to an equivalent one), so most pairs are settled without a
   search of their own. Pairs in the same orbit score 1, the others with equal
   profiles :math:`1 - 1 / N`.

   With a component index, its nodes are compared (the nodes of other components count as
   unreachable in the profiles, as in :func:`automorphic_equivalence`); ``component = None``
   compares all nodes of the network, so that isomorphic components are found equivalent.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param seed: Seed of the random hashes of refinement.
   :param batch_size: Bound on the numbers held at once while comparing profiles.
   :return: Dense ``n`` by ``n`` float64 tensor on the device of the network, rows and columns
      ordered by internal index (component by component with ``None``).

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      A = gsk.pairwise_automorphic_equivalence(net)
      orbits = (A == 1)                          # automorphic equivalence, as a boolean matrix
   """
   comps = list(range(len(net.components))) if component is None else [component]
   N = sum(d.num_nodes for d in net.components)
   n, s, t, keys, col, _ = _simple(net, comps)
   dev = net.components[comps[0]].edge_index.device
   P = torch.as_tensor(_profiles(_distances(n, s, t, net.directed, np.arange(n)), N - n), dtype = torch.float64, device = dev)
   M = torch.empty((n, n), dtype = torch.float64, device = dev)
   step = max(1, batch_size // max(1, n * P.shape[1]))
   for i in range(0, n, step):
      M[i:i + step] = torch.minimum(P[i:i + step, None, :], P[None, :, :]).sum(2)
   score = M / (2 * N - M)
   # equal profiles: orbits, on the graph with twins reduced, by refinement and automorphisms
   red = _twins(n, s, t, col)
   gen = torch.Generator().manual_seed(seed)
   prof = torch.unique(P.cpu(), dim = 0, return_inverse = True)[1]
   k = red[1]
   qprof = torch.zeros(k, dtype = torch.long)
   qprof[red[0]] = prof   # twins have equal profiles
   cls = torch.unique(torch.stack([qprof, _refine(red[5], red[2], red[3], gen)], 1), dim = 0, return_inverse = True)[1]
   parent = list(range(k))

   def root(x):
      while parent[x] != x:
         parent[x] = parent[parent[x]]
         x = parent[x]
      return x

   for c in torch.unique(cls).tolist():
      members = (cls == c).nonzero()[:, 0].tolist()
      while len(members) > 1:   # orbit of the first member, then of the first one left out
         x = members[0]
         for y in members[1:]:
            if root(x) == root(y):
               continue
            perm = _equivalent(red, x, y, gen)
            if perm is not None:   # every class is equivalent to its image
               for p, q in enumerate(perm.tolist()):
                  rp, rq = root(p), root(q)
                  if rp != rq:
                     parent[rp] = rq
         members = [y for y in members if root(y) != root(x)]
   orbit = torch.as_tensor([root(int(x)) for x in red[0]], device = dev)
   equal = M == N
   score[equal] = 1 - 1 / N
   score[(orbit[:, None] == orbit[None, :]) & equal] = 1.
   return score


def _undirected(net, c, weight):
   """
   Edges of a component as an undirected weighted adjacency (both directions; directed edges
   become undirected), with the degrees (row sums) and the largest eigenvalue.

   :return: ``(sources, targets, weights, degrees, largest eigenvalue)``, on the device of the
      component.
   :raises ValueError: If weights are negative.
   """
   data = net.components[c]
   n = data.num_nodes
   s, t = data.edge_index
   w = torch.ones(s.shape[0], dtype = torch.float64, device = s.device) if weight is None else net._numeric_edge(weight, c).double()
   if bool((w < 0).any()):
      raise ValueError("regular equivalence: weights must be non negative")
   if net.directed:
      loop = s == t
      s, t, w = torch.cat([s, t[~loop]]), torch.cat([t, s[~loop]]), torch.cat([w, w[~loop]])
   k = torch.zeros(n, dtype = torch.float64, device = s.device).index_add_(0, s, w)
   x = torch.full((n,), n ** -0.5, dtype = torch.float64, device = s.device)
   lam = 0.
   for _ in range(100000):   # power iteration on A + I (no oscillation), Rayleigh quotient
      ax = torch.zeros_like(x).index_add_(0, t, w * x[s])
      new = float(x @ ax)
      y = ax + x
      x = y / y.norm() if float(y.norm()) > 0 else x
      if abs(new - lam) <= 1e-13 * max(1., abs(new)):
         break
      lam = new
   return s, t, w, k, new


@_higher_order()
def path_regular_equivalence(net, u, v, alpha = 0.97, weight = None, tol = 1e-12, max_iter = 100000):
   """
   Path-based regular equivalence of two nodes: the vertex similarity of Leicht, Holme and Newman
   (2006), where a node is similar to another if its neighbors are similar to it, counting the
   paths between them. See :func:`pairwise_path_regular_equivalence` for the definition and how
   it differs from :func:`regular_equivalence`; this computes a single pair by iterating
   :math:`x = \\frac{\\alpha}{\\lambda_1} A x + e_v` on the device of the network (no ``n`` by
   ``n`` matrix), until the vector changes by less than ``tol`` per node.

   :param net: A :class:`~graphoskop.Network`.
   :param u: A node (input id or internal id ``(component, index)``; in multilayer networks a copy
      ``(entity, layer)``).
   :param v: Another node.
   :param alpha: Weight of longer paths, between 0 and 1 (0.9 to 0.99 recommended).
   :param weight: Name of the edge attribute used as weight, or ``None`` for unit weights.
   :param tol: Tolerance per node.
   :param max_iter: Largest number of iterations.
   :return: The similarity, a non negative float (not bounded by 1).
   :raises ValueError: If ``alpha`` is not between 0 and 1, weights are negative, or a node stands
      for several copies.
   :raises KeyError: If there is no such node.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.path_regular_equivalence(net, "alice", "bob")
   """
   from .centrality import _katz_iter
   if not 0 < alpha < 1:
      raise ValueError("regular equivalence: alpha must be between 0 and 1")
   ids = []
   for x in (u, v):
      copies = net._nodes(x)
      if len(copies) != 1:
         raise ValueError(f"regular equivalence: {x} stands for {len(copies)} copies; give one copy (entity, layer)")
      ids.append(copies[0])
   (cu, iu), (cv, iv) = ids
   if cu != cv:
      return 0.
   s, t, w, k, lam = _undirected(net, cu, weight)
   if k[iu] == 0 or k[iv] == 0 or lam <= 0:
      return 0.
   e = torch.zeros(len(k), dtype = torch.float64, device = k.device)
   e[iv] = 1
   x = _katz_iter(s, t, w, e, alpha / lam, tol, max_iter)
   if x is None:
      raise ValueError("regular equivalence: the iteration did not converge")
   return float(k.sum() * lam * x[iu] / (k[iu] * k[iv]))


@_higher_order()
def pairwise_path_regular_equivalence(net, component = 0, alpha = 0.97, weight = None):
   """
   Path-based regular equivalence of all pairs of nodes: the vertex similarity of Leicht, Holme and
   Newman (2006).

   Two nodes are regularly equivalent (White and Reitz, 1983) when they are joined to equivalent
   nodes, not necessarily the same ones (as structural equivalence requires). Leicht, Holme and
   Newman make it graded and recursive in a different way from :func:`regular_equivalence`:
   :math:`i` is similar to :math:`j` if :math:`i` has a neighbor similar to :math:`j`,
   :math:`S = \\phi A S + I` (every node is similar to itself), so
   :math:`S = (I - \\phi A)^{-1} = \\sum_l \\phi^l A^l`, which counts the paths of every length
   :math:`l` between the two nodes. Each count is divided by its expected value in a random
   network with the same degrees, :math:`k_i k_j \\lambda_1^{l - 1} / 2m`, with :math:`\\lambda_1`
   the largest eigenvalue of :math:`A`, and longer paths are discounted by :math:`\\alpha^l`:

   .. math:: S = 2 m \\lambda_1 D^{-1} \\left( I - \\frac{\\alpha}{\\lambda_1} A \\right)^{-1} D^{-1},

   with :math:`D` the diagonal matrix of the degrees (strengths with ``weight``) and :math:`m` the
   number (total weight) of edges. :math:`\\alpha` must be between 0 and 1; values close to 1 (0.9
   to 0.99, as recommended by the authors) give weight to the whole structure of the network,
   small values reduce the measure to common neighbors over :math:`k_i k_j`, a structural
   equivalence (the :math:`l = 2` term). Similarities are ratios of observed to expected paths,
   non negative and not bounded by 1; they rank pairs within a network.

   Since it counts paths between the two nodes, this similarity rewards proximity: nodes with the
   same role far apart score low. E.g. in two identical families (a parent with four children)
   whose parents are joined by a path of six edges, the two parents (regularly, and even
   automorphically, equivalent) score 5.4 with :math:`\\alpha = 0.97`, while a parent and one of
   its own children score 68.7; over all pairs, similarities follow the shortest path distance
   closely (Spearman correlation 0.87 to 0.97 for :math:`\\alpha \\leq 0.5`, 0.25 to 0.75 for
   :math:`\\alpha = 0.97`, on Zachary's karate club, a Barabasi-Albert and a Watts-Strogatz
   network). For roles independent of distance see :func:`regular_equivalence` (the classical
   measure) and :func:`automorphic_equivalence`.

   Edges are taken as undirected (the expected path counts assume it) and weights as their
   multiplicities; a self loop adds its weight once to the diagonal of :math:`A`. Each component
   is computed on its own (pairs in different components have no paths); nodes without edges get
   0. The inverse is a dense ``n`` by ``n`` matrix, computed by Cholesky factorization
   (:math:`I - \\frac{\\alpha}{\\lambda_1} A` is positive definite) on the device of the network.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest).
   :param alpha: Weight of longer paths, between 0 and 1 (0.9 to 0.99 recommended).
   :param weight: Name of the edge attribute used as weight, or ``None`` for unit weights.
   :return: Dense ``n`` by ``n`` float64 tensor on the device of the network, rows and columns
      ordered by internal index.
   :raises ValueError: If ``alpha`` is not between 0 and 1 or weights are negative.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      S = gsk.pairwise_path_regular_equivalence(net, alpha = 0.95)
   """
   if not 0 < alpha < 1:
      raise ValueError("regular equivalence: alpha must be between 0 and 1")
   s, t, w, k, lam = _undirected(net, component, weight)
   n = len(k)
   if lam <= 0:
      return torch.zeros((n, n), dtype = torch.float64, device = k.device)
   M = torch.zeros((n, n), dtype = torch.float64, device = k.device)
   M.index_put_((s, t), -alpha / lam * w, accumulate = True)
   M.diagonal().add_(1)
   L = torch.linalg.cholesky(M)
   S = torch.cholesky_inverse(L)
   inv = torch.where(k > 0, 1 / k.clamp(min = 1e-300), torch.zeros_like(k))
   return S.mul_(k.sum() * lam).mul_(inv[:, None]).mul_(inv[None, :])


def _rege(net, comps, weight, iterations, tol, batch_size):
   """
   REGE (White and Reitz, 1983), in the form of Wasserman and Faust (1994, section 12.4), on some
   components taken together.

   :return: Dense ``n`` by ``n`` matrix on the device of the network, nodes numbered over the
      components given (component by component, internal order).
   :raises ValueError: If weights are negative.
   """
   offset = np.cumsum([0] + [net.components[c].num_nodes for c in comps])
   n = int(offset[-1])
   dev = net.components[comps[0]].edge_index.device
   a, b, w = [], [], []
   for c, o in zip(comps, offset):
      s, t = net.components[c].edge_index
      a.append(s + int(o))
      b.append(t + int(o))
      w.append(torch.ones(s.shape[0], dtype = torch.float64, device = dev) if weight is None else net._numeric_edge(weight, c).double())
   a, b, w = torch.cat(a), torch.cat(b), torch.cat(w)
   if bool((w < 0).any()):
      raise ValueError("regular equivalence: weights must be non negative")
   if not net.directed:   # both directions stored: keep one, values x_ik = x_ki
      keep = a <= b
      a, b, w = a[keep], b[keep], w[keep]
      a, b, w = torch.cat([a, b[a != b]]), torch.cat([b, a[a != b]]), torch.cat([w, w[a != b]])
   key, inv = torch.unique(a * n + b, return_inverse = True)   # x, parallel edges summed
   x = torch.zeros(len(key), dtype = torch.float64, device = dev).index_add_(0, inv, w)
   pair = torch.unique(torch.cat([key, (key % n) * n + key // n]))   # neighbor relations (i, k), sorted by i
   ei, ek = pair // n, pair % n

   def lookup(q):   # x of the arcs with keys q (0 if absent)
      j = torch.searchsorted(key, q).clamp(max = len(key) - 1)
      return torch.where(key[j] == q, x[j], torch.zeros_like(x[j]))

   xo, xi = lookup(ei * n + ek), lookup(ek * n + ei)   # x_ik, x_ki
   E = len(pair)
   M = torch.ones((n, n), dtype = torch.float64, device = dev)
   step = max(1, batch_size // (E + n))
   rounds = iterations if iterations is not None else 1000
   for _ in range(rounds):
      T = torch.zeros((n, n), dtype = torch.float64, device = dev)
      D = torch.zeros((n, n), dtype = torch.float64, device = dev)
      for z in range(0, E, step):   # relations (i, k) against all relations (j, m)
         i, k, o, r = ei[z:z + step], ek[z:z + step], xo[z:z + step, None], xi[z:z + step, None]
         num = M[k][:, ek] * (torch.minimum(o, xo[None]) + torch.minimum(r, xi[None]))
         den = torch.maximum(o, xo[None]) + torch.maximum(r, xi[None])
         idx = ei[None].expand(len(i), E)
         best = torch.full((len(i), n), -torch.inf, dtype = torch.float64, device = dev).scatter_reduce(1, idx, num, "amax")
         tie = num == best.gather(1, idx)   # best match m* of k among j's neighbors; ties: smallest denominator
         dmin = torch.full((len(i), n), torch.inf, dtype = torch.float64, device = dev).scatter_reduce(1, idx, torch.where(tie, den, torch.inf), "amin")
         alone = (o + r).expand(len(i), n)   # j without neighbors: no match
         T.index_add_(0, i, torch.where(torch.isfinite(best), best, torch.zeros_like(best)))
         D.index_add_(0, i, torch.where(torch.isfinite(dmin), dmin, alone))
      num, den = T + T.T, D + D.T
      new = torch.where(den > 0, num / den.clamp(min = 1e-300), torch.ones_like(num))   # two nodes without neighbors: 1
      change = float((new - M).abs().max())
      M = new
      if iterations is None and change < tol:
         break
   return M


@_higher_order()
def regular_equivalence(net, u, v, weight = None, iterations = 3, tol = 1e-10, batch_size = 2 ** 24):
   """
   Degree of regular equivalence of two nodes, by REGE (White and Reitz, 1983): two nodes are
   regularly equivalent when they are joined to equivalent nodes, not necessarily the same ones.
   See :func:`pairwise_regular_equivalence` for the algorithm. The scores of a pair depend on
   those of the pairs of their neighbors, and so on, so all pairs of nodes of the components of
   ``u`` and ``v`` are computed, and the pair is picked.

   :param net: A :class:`~graphoskop.Network`.
   :param u: A node (input id or internal id ``(component, index)``; in multilayer networks a copy
      ``(entity, layer)``).
   :param v: Another node.
   :param weight: Name of the edge attribute used as edge value, or ``None`` for unit values.
   :param iterations: Number of iterations (3 as UCINET), or ``None`` to iterate until the scores
      change by less than ``tol``.
   :param tol: Tolerance, with ``iterations = None``.
   :param batch_size: Bound on the numbers held at once.
   :return: The score, a float between 0 and 1.
   :raises ValueError: If weights are negative, or a node stands for several copies.
   :raises KeyError: If there is no such node.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", directed = True)
      gsk.regular_equivalence(net, "alice", "bob")
   """
   ids = []
   for x in (u, v):
      copies = net._nodes(x)
      if len(copies) != 1:
         raise ValueError(f"regular equivalence: {x} stands for {len(copies)} copies; give one copy (entity, layer)")
      ids.append(copies[0])
   comps = sorted({c for c, _ in ids})
   offset = dict(zip(comps, np.cumsum([0] + [net.components[c].num_nodes for c in comps])))
   M = _rege(net, comps, weight, iterations, tol, batch_size)
   return float(M[int(offset[ids[0][0]]) + ids[0][1], int(offset[ids[1][0]]) + ids[1][1]])


@_higher_order()
def pairwise_regular_equivalence(net, component = 0, weight = None, iterations = 3, tol = 1e-10, batch_size = 2 ** 24):
   """
   Degree of regular equivalence of all pairs of nodes, by REGE (White and Reitz, 1983), in the form
   given by Wasserman and Faust (1994, section 12.4).

   Two nodes are regularly equivalent when they are joined to equivalent nodes, not necessarily the
   same ones: two parents are equivalent because both have children, whoever the children are.
   REGE starts with all pairs equivalent (:math:`M^0_{ij} = 1`) and, at every iteration, matches
   each neighbor :math:`k` of :math:`i` to the neighbor :math:`m` of :math:`j` most equivalent to
   it, with ties in the same directions and of similar values:

   .. math::

      M^{t+1}_{ij} = \\frac{\\sum_k \\max_m M^t_{km} \\left( \\min(x_{ik}, x_{jm}) + \\min(x_{ki}, x_{mj}) \\right) + (i \\leftrightarrow j)}
      {\\sum_k \\left( \\max(x_{ik}, x_{jm^*}) + \\max(x_{ki}, x_{m^*j}) \\right) + (i \\leftrightarrow j)},

   with :math:`k` over the neighbors of :math:`i` (in either direction), :math:`m` over those of
   :math:`j`, :math:`m^*` the best match (ties broken by the smallest denominator; a node without
   neighbors offers no match), :math:`x` the edge values (``weight``, or 1; parallel edges summed),
   and :math:`(i \\leftrightarrow j)` the same sums matching the neighbors of :math:`j` to those of
   :math:`i`. Scores are between 0 and 1, 1 for regularly equivalent nodes, and do not increase
   from one iteration to the next. UCINET runs 3 iterations by default, as here, which gives graded
   scores; with ``iterations = None`` they are run until the scores change by less than ``tol``,
   which tends to separate the regularly equivalent pairs (1) from the others (towards 0). Two nodes
   without neighbors score 1 (they are regularly equivalent), a node with neighbors and one
   without score 0.

   Regular equivalence needs directions or values to tell nodes apart: in an undirected network
   with unit values, every node with neighbors is regularly equivalent to every other (each of
   its neighbors matches any neighbor of the other, all equivalent at the start), so all such
   pairs score 1. Use it on directed or valued networks; for undirected unweighted ones see
   :func:`automorphic_equivalence`. Unlike :func:`path_regular_equivalence`, it does not depend on
   the distance between the two nodes: nodes in different components can be equivalent, so
   ``component = None`` compares all nodes of the network. E.g. with two separate families, a
   parent with edges to four children and one with edges to three, the two parents score 1, and so
   do any two children; a parent and a child score 0. The parents are not automorphically
   equivalent (the families differ in size), and have no paths between them.

   Every iteration compares every neighbor relation :math:`(i, k)` with every relation
   :math:`(j, m)`, :math:`(2m)^2` comparisons for :math:`m` edges, in blocks of about
   ``batch_size`` numbers on the device of the network.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param weight: Name of the edge attribute used as edge value, or ``None`` for unit values.
   :param iterations: Number of iterations (3 as UCINET), or ``None`` to iterate until the scores
      change by less than ``tol``.
   :param tol: Tolerance, with ``iterations = None``.
   :param batch_size: Bound on the numbers held at once.
   :return: Dense ``n`` by ``n`` float64 tensor on the device of the network, rows and columns
      ordered by internal index (component by component with ``None``).
   :raises ValueError: If weights are negative.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", directed = True)
      R = gsk.pairwise_regular_equivalence(net, component = None)
      R = gsk.pairwise_regular_equivalence(net, weight = "weight", iterations = None)
   """
   comps = list(range(len(net.components))) if component is None else [component]
   return _rege(net, comps, weight, iterations, tol, batch_size)
