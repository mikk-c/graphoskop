"""
Centrality of nodes based on shortest paths: closeness, harmonic and betweenness centrality, and
betweenness centrality of edges; local reaching centrality of directed networks; and, not based on
shortest paths, random walk (current flow) betweenness (:func:`flow_centrality`), eigenvector
centrality (:func:`eigenvector_centrality`), Katz centrality (:func:`katz_centrality`), alpha
centrality (:func:`alpha_centrality`), PageRank (:func:`pagerank`), hubs and authorities
(:func:`hits`), k-core centrality (:func:`k_core_centrality`) and the SpringRank hierarchy of
directed networks (:func:`springrank`). Any of them can be turned into a
centralization of the whole network with :func:`centralization`.

Paths are those of :mod:`~graphoskop.shortest_paths`: lengths are the values of an edge attribute
(``weight``) or 1, with ``inverse = True`` for attributes measuring affinity; directed networks are
searched along edge directions; in multilayer networks paths run on the supra-graph, where
changing layer along a coupling is a step, and every copy has its own centrality.

``component`` selects what counts as the network: an index computes the centralities of that
component as if it were the whole network (normalizations use its number of nodes), ``None``
computes them for all components at once, as one network whose nodes in different components
are unreachable from each other (normalizations use the number of nodes of the whole network).
Results are float64 vectors over the nodes of a component, ordered by internal index, on its
device (a list of them, one per component, with ``component = None``), ready for
:meth:`~graphoskop.Network.set_node_attribute`.

The lengths of all shortest paths are the costly part. They can be computed once with
:func:`~graphoskop.all_pairs_shortest_path_lengths` (with the same ``component``, ``weight`` and
``inverse``) and passed as ``lengths``, to all these functions and to
:func:`~graphoskop.average_shortest_path_length`; nothing is then searched again. Without them,
lengths are found by Dijkstra on the CPU, in blocks of sources (in parallel processes for large
work, see :mod:`~graphoskop.shortest_paths`), and never stored all at once.
"""

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

from .network import _quiet_sparse
from .shortest_paths import _graph, _lengths, _dijkstra, _check_lengths, _source_stats


def _components(net, component, lengths):
   """
   Components to compute, their precomputed lengths, and the number of nodes of the network
   considered.

   :return: ``(components, lengths per component or Nones, N)``.
   """
   comps = list(range(len(net.components))) if component is None else [component]
   if lengths is None:
      lengths = [None] * len(comps)
   elif component is not None:
      lengths = [lengths]
   if len(lengths) != len(comps):
      raise ValueError("lengths must be one tensor per component, as given by all_pairs_shortest_path_lengths(net, None, ...)")
   N = sum(net.components[c].num_nodes for c in comps)
   return comps, lengths, N


def _directional(net, direction):
   """
   Whether lengths are taken to the node (``"in"``) rather than from it.

   :raises ValueError: If ``direction`` is not valid.
   """
   if direction not in ("out", "in"):
      raise ValueError(f"unknown direction: {direction}")
   return net.directed and direction == "in"


def closeness_centrality(net, component = 0, weight = None, inverse = False, direction = "out", lin = False, lengths = None, n_jobs = None):
   """
   Closeness centrality: how close a node is to the nodes it reaches.

   :math:`C(i) = r_i / \\sum_j d(i, j)`, with the sum over the :math:`r_i` nodes reachable from
   :math:`i` (in an undirected connected network, :math:`r_i = n - 1`); 0 if :math:`i` reaches no
   node. Nodes in small components are close to the few nodes they reach, so with
   ``lin = True`` (Lin, 1976; Wasserman and Faust, 1994) the value is scaled by the fraction of
   the network they reach, :math:`C(i) = \\frac{r_i}{N - 1} \\frac{r_i}{\\sum_j d(i, j)}`, with
   :math:`N` the number of nodes of the network considered: use it with ``component = None`` to
   compare nodes of different components (or of directed networks where not every node is
   reachable). See also :func:`harmonic_centrality`.

   In directed networks, ``direction = "out"`` uses the lengths from the node to the others,
   ``"in"`` those from the others to the node (NetworkX uses ``"in"``; its ``wf_improved`` is
   ``lin``).

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components at
      once (see the module description).
   :param weight: Name of the edge attribute used as edge length, or ``None`` for unit lengths.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :param direction: ``"out"`` or ``"in"``; ignored in undirected networks.
   :param lin: Scale by the fraction of the network reached.
   :param lengths: Precomputed lengths (see the module description), or ``None``.
   :param n_jobs: Number of processes for the searches, ``None`` for all CPUs.
   :return: Vector over the nodes of the component, or a list of them if ``component`` is ``None``.
   :raises ValueError: If ``direction`` is not valid, lengths are negative or NaN, or ``lengths``
      does not match.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      c = gsk.closeness_centrality(net)                              # largest component
      cs = gsk.closeness_centrality(net, component = None, lin = True)   # comparable across components
      net.set_node_attribute("closeness", c)
   """
   rev = _directional(net, direction)
   comps, lengths, N = _components(net, component, lengths)
   out = []
   for c, L in zip(comps, lengths):
      data = net.components[c]
      r, d, _ = _source_stats(net, c, np.arange(data.num_nodes), weight, inverse, n_jobs, rev, L)
      with np.errstate(invalid = "ignore", divide = "ignore"):
         v = np.where(d > 0, r / d, 0.)
      if lin:
         v = v * r / (N - 1) if N > 1 else v * 0
      out.append(torch.as_tensor(v, dtype = torch.float64, device = data.edge_index.device))
   return out if component is None else out[0]


def harmonic_centrality(net, component = 0, weight = None, inverse = False, direction = "out", normalized = True, lengths = None, n_jobs = None):
   """
   Harmonic centrality (Marchiori and Latora, 2000; Boldi and Vigna, 2014): the reciprocal of the
   harmonic mean of the lengths from a node to all the other nodes.

   :math:`H(i) = \\frac{1}{N - 1} \\sum_{j \\ne i} \\frac{1}{d(i, j)}`, with :math:`1 / \\infty = 0`
   for unreachable nodes, so nodes of different components (``component = None``) are compared
   directly, without corrections. :math:`N` is the number of nodes of the network considered;
   ``normalized = False`` gives the plain sum (as NetworkX). Nodes at length 0 (along edges of
   length 0) are left out of the sum.

   In directed networks, ``direction = "out"`` uses the lengths from the node to the others,
   ``"in"`` those from the others to the node (as NetworkX).

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components at
      once (see the module description).
   :param weight: Name of the edge attribute used as edge length, or ``None`` for unit lengths.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :param direction: ``"out"`` or ``"in"``; ignored in undirected networks.
   :param normalized: Divide by :math:`N - 1`.
   :param lengths: Precomputed lengths (see the module description), or ``None``.
   :param n_jobs: Number of processes for the searches, ``None`` for all CPUs.
   :return: Vector over the nodes of the component, or a list of them if ``component`` is ``None``.
   :raises ValueError: If ``direction`` is not valid, lengths are negative or NaN, or ``lengths``
      does not match.

   Example:

   .. code-block:: python

      h = gsk.harmonic_centrality(net, component = None)         # all components at once
   """
   rev = _directional(net, direction)
   comps, lengths, N = _components(net, component, lengths)
   out = []
   for c, L in zip(comps, lengths):
      data = net.components[c]
      _, _, h = _source_stats(net, c, np.arange(data.num_nodes), weight, inverse, n_jobs, rev, L)
      if normalized:
         h = h / (N - 1) if N > 1 else h * 0
      out.append(torch.as_tensor(h, dtype = torch.float64, device = data.edge_index.device))
   return out if component is None else out[0]


def _dag_betweenness(D, sources, u, v, w, edges = False):
   """
   Dependencies (Brandes, 2001) of a block of sources, from their shortest path lengths.

   An edge :math:`u \\to v` is on a shortest path from :math:`s` if :math:`d(s, u) + w = d(s, v)`
   (up to a relative tolerance of :math:`10^{-9}`, so that ties survive rounding). These edges form
   a directed acyclic graph per source; each is visited once, ordered by the depth of its target
   (the largest number of edges of a shortest path reaching it): forward to count the shortest paths :math:`\\sigma`, then backward to accumulate
   :math:`\\delta(u) = \\sum_v \\frac{\\sigma(u)}{\\sigma(v)} (1 + \\delta(v))`.

   :param D: Lengths from the sources, ``B`` by ``n``, float64.
   :param sources: Source indices (tensor).
   :param u: Edge sources, distinct pairs without self loops.
   :param v: Edge targets.
   :param w: Positive edge lengths, float64.
   :param edges: Also return the dependencies of the edges.
   :return: Sum over the sources of their dependencies, length ``n``, and, with ``edges``, of the
      dependencies of the edges, aligned with ``u``.
   """
   B, n = D.shape
   start = torch.arange(B, device = D.device) * n + sources   # flat index of each source
   Du, Dv = D[:, u], D[:, v]
   b, e = (torch.isfinite(Du) & ((Du + w - Dv).abs() <= 1e-9 * Dv.abs().clamp(min = 1))).nonzero(as_tuple = True)
   U, V = b * n + u[e], b * n + v[e]
   E = e
   level = torch.full((B * n,), -1, dtype = torch.long, device = D.device)   # longest path in the DAG
   level[start] = 0
   while True:
      new = level.scatter_reduce(0, V, torch.where(level[U] >= 0, level[U] + 1, -1), "amax")
      if torch.equal(new, level):
         break
      level = new
   lev = level[V]
   del Du, Dv, b
   order = torch.argsort(lev, stable = True)
   U, V, E, lev = U[order], V[order], E[order], lev[order]
   bounds = torch.cumsum(torch.bincount(lev), 0).tolist() if lev.shape[0] else [0]
   sigma = torch.zeros(B * n, dtype = torch.float64, device = D.device)
   sigma[start] = 1
   for k in range(1, len(bounds)):   # forward, depth by depth
      a, z = bounds[k - 1], bounds[k]
      sigma.index_add_(0, V[a:z], sigma[U[a:z]])
   delta = torch.zeros_like(sigma)
   eb = torch.zeros(u.shape[0], dtype = torch.float64, device = D.device)
   for k in range(len(bounds) - 1, 0, -1):   # backward
      a, z = bounds[k - 1], bounds[k]
      dep = sigma[U[a:z]] / sigma[V[a:z]] * (1 + delta[V[a:z]])
      delta.index_add_(0, U[a:z], dep)
      if edges:
         eb.index_add_(0, E[a:z], dep)
   delta[start] = 0
   bc = delta.view(B, n).sum(0)
   return (bc, eb) if edges else bc


def _bfs_betweenness(pu, pv, n, dev, batch_size, edges):
   """
   Betweenness with unit lengths, by breadth first searches from blocks of sources at once,
   written as products of the sparse adjacency matrix with dense ``n`` by ``B`` matrices (one
   column per source), level by level: forward, the number of shortest paths :math:`\\sigma`
   reaching each node at each level; backward, the dependencies
   :math:`\\delta(u) = \\sum_v \\frac{\\sigma(u)}{\\sigma(v)} (1 + \\delta(v))` over the
   edges to the next level. No lengths are searched or stored.

   :param pu: Edge sources (distinct pairs, no self loops), NumPy.
   :param pv: Edge targets.
   :param n: Number of nodes.
   :param dev: Device.
   :param batch_size: Bound on the numbers held at once (about 8 per node and source).
   :param edges: Also return the betweenness of the edges.
   :return: Node betweenness (sum over ordered pairs), and edge betweenness aligned with
      ``pu`` (or ``None``).
   """
   def csr(a, b):
      g = sp.csr_matrix((np.ones(len(a)), (a, b)), shape = (n, n))
      with _quiet_sparse():
         return torch.sparse_csr_tensor(torch.as_tensor(g.indptr), torch.as_tensor(g.indices), torch.as_tensor(g.data), (n, n), device = dev)

   A, AT = csr(pu, pv), csr(pv, pu)
   u, v = torch.as_tensor(pu, device = dev), torch.as_tensor(pv, device = dev)
   bc = torch.zeros(n, dtype = torch.float64, device = dev)
   eb = torch.zeros(len(pu), dtype = torch.float64, device = dev)
   B = max(1, min(n, batch_size // (8 * n)))
   for i in range(0, n, B):
      src = torch.arange(i, min(i + B, n), device = dev)
      cols = torch.arange(len(src), device = dev)
      sigma = torch.zeros((n, len(src)), dtype = torch.float64, device = dev)
      sigma[src, cols] = 1
      lev = torch.zeros((n, len(src)), dtype = torch.int32, device = dev)   # level + 1, 0 if not reached
      lev[src, cols] = 1
      front, d = sigma, 1
      while True:   # forward
         front = (AT @ front).masked_fill_(lev > 0, 0)
         new = front > 0
         if not new.any():
            break
         d += 1
         lev.masked_fill_(new, d)
         sigma += front
      inv = torch.where(lev > 0, 1 / sigma, 0.)
      delta = torch.zeros_like(sigma)
      for k in range(d, 1, -1):   # backward
         coef = (1 + delta).mul_(inv).masked_fill_(lev != k, 0)
         delta += (A @ coef).mul_(sigma).masked_fill_(lev != k - 1, 0)
      if edges:   # dependency of u -> v: sigma(u) (1 + delta(v)) / sigma(v), if v is one level down
         coef = (1 + delta).mul_(inv)
         step = max(1, batch_size // (4 * len(src)))
         for a in range(0, len(pu), step):
            x, y = u[a:a + step], v[a:a + step]
            eb[a:a + step] += (sigma[x] * coef[y]).masked_fill_(lev[y] != lev[x] + 1, 0).sum(1)
      delta[src, cols] = 0
      bc += delta.sum(1)
   return bc, (eb if edges else None)


def _dijkstra_betweenness(G, L, u, v, w, n, dev, batch_size, n_jobs, edges):
   """
   Betweenness with lengths: lengths from blocks of sources (by Dijkstra, or from ``L``), then the
   shortest path edges counted by :func:`_dag_betweenness`.

   :return: Node betweenness (sum over ordered pairs), and edge betweenness aligned with ``u``.
   """
   bc = torch.zeros(n, dtype = torch.float64, device = dev)
   eb = torch.zeros(len(w), dtype = torch.float64, device = dev)
   step = max(1, batch_size // (len(w) + n))
   for i in range(0, n, step):
      src = np.arange(i, min(i + step, n))
      if L is None:
         D = torch.as_tensor(_dijkstra(G, src, False, False, n_jobs), dtype = torch.float64, device = dev)
      else:
         D = L[i:i + len(src)].double()
      r = _dag_betweenness(D, torch.as_tensor(src, device = dev), u, v, w, edges)
      if edges:
         bc += r[0]
         eb += r[1]
      else:
         bc += r
   return bc, eb


def _accumulate(net, c, L, weight, inverse, batch_size, n_jobs, edges):
   """
   Betweenness of the nodes of a component, and of its edges (see :func:`betweenness_centrality`).

   :return: Node betweenness (sum over ordered pairs), and, with ``edges``, the betweenness of
      every stored edge, aligned with ``edge_index`` (``None`` otherwise).
   :raises ValueError: If some edge lengths are not positive, or ``L`` does not match.
   """
   data = net.components[c]
   n, dev = data.num_nodes, data.edge_index.device
   G = _graph(net, c, weight, inverse)
   g = G.tocoo()
   keep = (g.row != g.col) & np.isfinite(g.data)
   if (g.data[keep] <= 0).any():
      raise ValueError("betweenness: edge lengths must be positive")
   pu, pv = g.row[keep], g.col[keep]
   u, v = (torch.as_tensor(x, dtype = torch.long, device = dev) for x in (pu, pv))
   w = torch.as_tensor(g.data[keep], dtype = torch.float64, device = dev)
   if L is not None:
      L = _check_lengths(net, c, L).to(dev)
   if weight is None:
      bc, eb = _bfs_betweenness(pu, pv, n, dev, batch_size, edges)
   else:
      bc, eb = _dijkstra_betweenness(G, L, u, v, w, n, dev, batch_size, n_jobs, edges)
   if not edges:
      return bc, None
   # pair values to stored edges: the shortest of parallel edges (the first on ties) gets the value
   eb = eb.cpu().numpy()
   pk = pu.astype(np.int64) * n + pv   # sorted (CSR order)
   s, t = data.edge_index.cpu().numpy()
   wl = _lengths(net, c, weight, inverse)
   m = len(s) if net.directed else net.number_of_edges(c)
   a, b = (s[:m], t[:m]) if net.directed else (np.minimum(s[:m], t[:m]), np.maximum(s[:m], t[:m]))
   ok = a != b
   order = np.lexsort((np.arange(m), wl[:m], b, a))
   order = order[ok[order]]
   first = order[np.r_[True, (a[order][1:] != a[order][:-1]) | (b[order][1:] != b[order][:-1])][:len(order)]]

   def value(x, y):   # value of the pairs x -> y, 0 if not an edge of the CSR
      k = x.astype(np.int64) * n + y
      j = np.searchsorted(pk, k).clip(max = len(pk) - 1)
      return np.where(pk[j] == k, eb[j], 0.) if len(pk) else np.zeros(len(k))

   val = value(a[first], b[first]) + (0 if net.directed else value(b[first], a[first]))
   val = np.where(np.isfinite(wl[first]), val, 0.)
   out = np.zeros(len(s))
   out[first] = val
   if not net.directed:   # reverse copies, stored after the input edges in the same order (no self loops)
      rev = m + np.cumsum(ok)[ok] - 1
      out[rev] = out[:m][ok]
   return bc, torch.as_tensor(out, dtype = torch.float64, device = dev)


def betweenness_centrality(net, component = 0, weight = None, inverse = False, normalized = True, lengths = None, batch_size = 2 ** 24, n_jobs = None):
   """
   Betweenness centrality (Freeman, 1977): the fraction of shortest paths between other nodes that
   pass through a node, :math:`B(i) = \\sum_{s \\ne i \\ne t} \\sigma_{st}(i) / \\sigma_{st}`, with
   :math:`\\sigma_{st}` the number of shortest paths from :math:`s` to :math:`t` and
   :math:`\\sigma_{st}(i)` those through :math:`i`. All tied shortest paths count.

   With ``normalized = True`` the sum over ordered pairs is divided by :math:`(N - 1)(N - 2)`, the
   number of ordered pairs of other nodes, with :math:`N` the number of nodes of the network
   considered (as NetworkX, also for undirected networks); otherwise the plain sum, over
   unordered pairs in undirected networks.

   Paths are counted forward and then backward as in Brandes' algorithm (2001), on the device of
   the network, for blocks of sources at once (about ``batch_size`` numbers in memory). With unit
   lengths (``weight = None``) the breadth first searches of a block of sources run together,
   level by level, as products of the sparse adjacency matrix with a dense matrix of one column
   per source, which count the paths as they go: no lengths are searched or stored, and
   ``lengths`` is not needed. With lengths, only the lengths of the shortest paths are needed: an
   edge :math:`u \\to v` of length :math:`w` is on a shortest path from :math:`s` exactly when
   :math:`d(s, u) + w = d(s, v)`, and the paths are counted along these edges. Edge lengths must
   be positive (a cycle of length 0 would give infinitely many shortest paths). Parallel edges
   count once, with the shortest length (igraph counts each parallel edge as a separate path), and
   self loops are ignored. With ``lengths``, they must have been computed with the same ``weight``
   and ``inverse``, which are still needed to know the edge lengths.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components at
      once (see the module description).
   :param weight: Name of the edge attribute used as edge length, or ``None`` for unit lengths.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :param normalized: Divide by :math:`(N - 1)(N - 2)`.
   :param lengths: Precomputed lengths (see the module description), or ``None``.
   :param batch_size: Bound on the numbers held at once (about one per source and edge).
   :param n_jobs: Number of processes for the searches, ``None`` for all CPUs.
   :return: Vector over the nodes of the component, or a list of them if ``component`` is ``None``.
   :raises ValueError: If some edge lengths are not positive, or ``lengths`` does not match.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", device = "cuda")
      P = gsk.all_pairs_shortest_path_lengths(net, weight = "weight", inverse = True)
      b = gsk.betweenness_centrality(net, weight = "weight", inverse = True, lengths = P)
      c = gsk.closeness_centrality(net, lengths = P)                 # same lengths, no new search
   """
   comps, lengths, N = _components(net, component, lengths)
   out = []
   for c, L in zip(comps, lengths):
      bc = _accumulate(net, c, L, weight, inverse, batch_size, n_jobs, False)[0]
      if normalized:
         bc = bc / ((N - 1) * (N - 2)) if N > 2 else bc * 0
      elif not net.directed:
         bc = bc / 2
      out.append(bc)
   return out if component is None else out[0]


def edge_betweenness_centrality(net, component = 0, weight = None, inverse = False, normalized = True, lengths = None, batch_size = 2 ** 24, n_jobs = None):
   """
   Edge betweenness centrality (Girvan and Newman, 2002): the fraction of shortest paths between
   nodes that pass along an edge, :math:`B(e) = \\sum_{s \\ne t} \\sigma_{st}(e) / \\sigma_{st}`,
   with :math:`\\sigma_{st}` the number of shortest paths from :math:`s` to :math:`t` and
   :math:`\\sigma_{st}(e)` those using :math:`e`. All tied shortest paths count, and, unlike node
   betweenness, the pairs whose shortest paths start or end at the edge count too (an edge is on
   the shortest path between its own endpoints, unless a shorter path exists).

   With ``normalized = True`` the sum over ordered pairs is divided by :math:`N (N - 1)`, the
   number of ordered pairs, with :math:`N` the number of nodes of the network considered (as
   NetworkX, also for undirected networks); otherwise the plain sum, over unordered pairs in
   undirected networks.

   Computed in the same pass as :func:`betweenness_centrality`, with the same algorithm, devices,
   memory bound and conditions on the lengths: the dependency of each edge on a shortest path is
   one of the terms of Brandes' backward accumulation. In undirected networks an edge counts the
   paths along it in both directions. Among parallel edges, only the shortest (the first in input
   order on ties) carries the paths, the others get 0; self loops and edges of infinite length get
   0.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components at
      once (see the module description).
   :param weight: Name of the edge attribute used as edge length, or ``None`` for unit lengths.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :param normalized: Divide by :math:`N (N - 1)`.
   :param lengths: Precomputed lengths (see the module description), or ``None``.
   :param batch_size: Bound on the numbers held at once (about one per source and edge).
   :param n_jobs: Number of processes for the searches, ``None`` for all CPUs.
   :return: Vector over the stored edges of the component, aligned with ``edge_index`` (both
      directions of an undirected edge get its value), ready for
      :meth:`~graphoskop.Network.set_edge_attribute`; a list of them if ``component`` is ``None``.
   :raises ValueError: If some edge lengths are not positive, or ``lengths`` does not match.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      eb = gsk.edge_betweenness_centrality(net)
      net.set_edge_attribute("betweenness", eb)
      ebs = gsk.edge_betweenness_centrality(net, component = None)
      net.set_edge_attribute("betweenness", ebs, component = None)
   """
   comps, lengths, N = _components(net, component, lengths)
   out = []
   for c, L in zip(comps, lengths):
      eb = _accumulate(net, c, L, weight, inverse, batch_size, n_jobs, True)[1]
      if normalized:
         eb = eb / (N * (N - 1)) if N > 1 else eb * 0
      elif not net.directed:
         eb = eb / 2
      out.append(eb)
   return out if component is None else out[0]


def flow_centrality(net, component = 0, weight = None, normalized = True, endpoints = False, batch_size = 2 ** 24):
   """
   Random walk betweenness centrality (Newman, 2005), also known as current flow betweenness: as
   :func:`betweenness_centrality`, but with all random walks from :math:`s` to :math:`t` instead of
   only the shortest paths. A node counts the net number of times the walks pass through it,
   which equals the current through it when a unit current enters the network at :math:`s` and
   leaves at :math:`t`, edges being resistors of conductance equal to their weight.

   The current from a pair is computed from the pseudoinverse :math:`T` of the Laplacian: node
   :math:`i` gets :math:`I_i^{st} = \\frac{1}{2} \\sum_j A_{ij} |T_{is} - T_{it} - T_{js} + T_{jt}|`,
   and :math:`B(i) = \\sum_{s < t} I_i^{st}`, over the pairs of other nodes (``endpoints = False``)
   or of all nodes, with :math:`I_s^{st} = I_t^{st} = 1` (``endpoints = True``, Newman's
   definition). With ``normalized = True`` the sum is divided by the number of pairs,
   :math:`(N - 1)(N - 2) / 2` or :math:`N (N - 1) / 2`, with :math:`N` the number of nodes of the
   network considered (the first as NetworkX's ``current_flow_betweenness_centrality``).

   The current along an edge :math:`(i, j)` is :math:`A_{ij} |x_s - x_t|`, with
   :math:`x = T_{i \\cdot} - T_{j \\cdot}`, so its sum over all pairs is found by sorting
   :math:`x`: :math:`\\sum_{s < t} |x_s - x_t| = \\sum_k (2k - n + 1) x_{(k)}`. The cost is a dense
   pseudoinverse per component (as :func:`~graphoskop.laplacian_pinv`, :math:`n^2` numbers in
   memory) and a sort of :math:`n` numbers per edge, in blocks of edges (about ``batch_size``
   numbers at once), all on the device of the network: :math:`O(n^3 + m n \\log n)`, instead of
   :math:`O((m + n) n^2)` for Newman's algorithm.

   Edges are taken as undirected (in directed networks each edge is a resistor whatever its
   direction), parallel edges add their conductances, self loops are ignored. Weights must be
   positive. Pairs of nodes in different components carry no current, so with
   ``component = None`` they add nothing. In multilayer networks the current flows on the
   supra-graph, couplings included.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components at
      once (see the module description).
   :param weight: Name of the edge attribute used as edge conductance, or ``None`` for unit
      conductances.
   :param normalized: Divide by the number of pairs.
   :param endpoints: Count the pairs where the node is the source or the target, with current 1.
   :param batch_size: Bound on the numbers held at once while sorting (about one per edge and
      node of a block).
   :return: Vector over the nodes of the component, or a list of them if ``component`` is ``None``.
   :raises ValueError: If some weights are not positive.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", device = "cuda")
      f = gsk.flow_centrality(net)
      f = gsk.flow_centrality(net, weight = "weight", endpoints = True)   # Newman's normalization
      net.set_node_attribute("flow", f)
   """
   from .utils import _pinv_null
   comps, _, N = _components(net, component, None)
   out = []
   for c in comps:
      data = net.components[c]
      n, dev = data.num_nodes, data.edge_index.device
      ei, X = net._input_edges(c)
      u, v = ei
      w = torch.ones(u.shape[0], dtype = torch.float64, device = dev) if weight is None else X[:, net._column(net.edge_attributes, weight, "edge")].double()
      keep = u != v
      u, v, w = u[keep], v[keep], w[keep]
      if not bool((w > 0).all()):
         raise ValueError("flow centrality: weights must be positive")
      b = torch.zeros(n, dtype = torch.float64, device = dev)
      if n > 2:
         L = torch.zeros((n, n), dtype = torch.float64, device = dev)
         L.index_put_((u, v), -w, accumulate = True)
         L.index_put_((v, u), -w, accumulate = True)
         d = torch.zeros(n, dtype = torch.float64, device = dev).index_add_(0, u, w).index_add_(0, v, w)
         L.diagonal().add_(d)
         T = _pinv_null(L, torch.ones(n, dtype = torch.float64, device = dev))
         coef = torch.arange(n, dtype = torch.float64, device = dev).mul_(2).sub_(n - 1)
         step = max(1, batch_size // n)
         for a in range(0, len(w), step):   # current along each edge, summed over the pairs
            x = (T[u[a:a + step]] - T[v[a:a + step]]).sort(1).values
            f = (x @ coef) * w[a:a + step]
            b.index_add_(0, u[a:a + step], f).index_add_(0, v[a:a + step], f)
         del L, T
         b = b / 2 + (n - 1) / 2 * (1 if endpoints else -1)   # each node is an endpoint of n - 1 pairs, with current 1 (counted 1/2 above)
      else:
         b.fill_(n - 1 if endpoints else 0)
      if normalized:
         k = N * (N - 1) / 2 if endpoints else (N - 1) * (N - 2) / 2
         b = b / k if k > 0 else b * 0
      out.append(b)
   return out if component is None else out[0]


def reach_centrality(net, component = 0, direction = "out", lengths = None, n_jobs = None):
   """
   Local reaching centrality (Mones, Vicsek and Vicsek, 2012): the fraction of the other nodes
   that a node reaches along edge directions, :math:`C_R(i) = r_i / (N - 1)`, with :math:`r_i` the
   number of nodes reachable from :math:`i` and :math:`N` the number of nodes of the network
   considered (see the module description). With ``direction = "in"``, the fraction of nodes from
   which :math:`i` is reached. Defined for directed networks only: in an undirected network every
   node reaches its whole component. Edge weights do not matter (reaching does not depend on
   them; the weighted variant of the paper averages weights along one shortest path, which is not
   unique, and is not provided).

   All nodes of a strongly connected component reach the same nodes, so a single breadth first
   search per strongly connected component is needed (on the CPU, in parallel processes for
   large work, as :func:`closeness_centrality`), or none with precomputed ``lengths``.

   :param net: A directed :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components at
      once (see the module description).
   :param direction: ``"out"`` (nodes reached) or ``"in"`` (nodes reaching).
   :param lengths: Precomputed lengths (see the module description), or ``None``.
   :param n_jobs: Number of processes for the searches, ``None`` for all CPUs.
   :return: Vector over the nodes of the component, or a list of them if ``component`` is ``None``.
   :raises ValueError: If the network is undirected, ``direction`` is not valid, or ``lengths``
      does not match.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", directed = True)
      r = gsk.reach_centrality(net, component = None)
      net.set_node_attribute("reach", r, component = None)

   See :func:`~graphoskop.global_reach_centrality` for the hierarchy measure built on it.
   """
   if not net.directed:
      raise ValueError("reach centrality is defined for directed networks only")
   rev = _directional(net, direction)
   comps, lengths, N = _components(net, component, lengths)
   out = []
   for c, L in zip(comps, lengths):
      data = net.components[c]
      n = data.num_nodes
      if L is None:   # one source per strongly connected component
         scc = connected_components(_graph(net, c, None, False), connection = "strong")[1]
         rep = np.unique(scc, return_index = True)[1]
         r = _source_stats(net, c, rep, None, False, n_jobs, rev)[0][scc]
      else:
         r = _source_stats(net, c, np.arange(n), None, False, n_jobs, rev, L)[0]
      out.append(torch.as_tensor(r / (N - 1) if N > 1 else r * 0, dtype = torch.float64, device = data.edge_index.device))
   return out if component is None else out[0]



def eigenvector_centrality(net, component = 0, weight = None, direction = "in", tol = 1e-10, max_iter = 10000):
   """
   Eigenvector centrality (Bonacich, 1972): a node is central if its neighbors are,
   :math:`\\lambda x_i = \\sum_j A_{ji} x_j`, with :math:`x` the eigenvector of the largest
   eigenvalue :math:`\\lambda` of the adjacency matrix, non negative and of unit Euclidean norm.

   In directed networks, ``direction = "in"`` gives a node the centrality of the nodes with edges
   to it (as NetworkX), ``"out"`` that of the nodes it has edges to. If the network is not
   strongly connected, nodes not reached from a strongly connected part with the largest
   eigenvalue get 0 (e.g. all of them in a directed acyclic network, where the iteration does not
   settle and a warning is raised): see :func:`~graphoskop.stationary_distribution` for a measure
   that is positive everywhere. In multilayer networks the eigenvector is that of the
   supra-adjacency matrix, couplings included. Weights add up over parallel edges; a self loop
   counts once in undirected networks.

   Computed by power iteration on :math:`I + A^T` (same eigenvector, without the oscillation of
   bipartite networks), with one sparse product per step on the device of the network, starting
   from a constant vector (as NetworkX), until the vector changes by less than ``tol`` per node
   (1-norm of the change below :math:`n \\cdot` ``tol``). Each component is computed and normalized
   on its own.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components
      (each normalized on its own).
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param direction: ``"in"`` or ``"out"``; ignored in undirected networks.
   :param tol: Tolerance per node.
   :param max_iter: Largest number of iterations; a warning is raised if it is reached.
   :return: Vector over the nodes of the component, or a list of them if ``component`` is ``None``.
   :raises ValueError: If ``direction`` is not valid, or some weights are negative.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", device = "cuda")
      x = gsk.eigenvector_centrality(net, weight = "weight")
      net.set_node_attribute("eigenvector", x)
   """
   if direction not in ("in", "out"):
      raise ValueError(f"unknown direction: {direction}")
   out = []
   for c in range(len(net.components)) if component is None else [component]:
      data = net.components[c]
      n = data.num_nodes
      s, t = data.edge_index
      if net.directed and direction == "out":
         s, t = t, s
      w = torch.ones(s.shape[0], dtype = torch.float64, device = s.device) if weight is None else net._numeric_edge(weight, c).double()
      if bool((w < 0).any()):
         raise ValueError("eigenvector centrality: weights must be non negative")
      x = torch.full((n,), n ** -0.5, dtype = torch.float64, device = s.device)
      for it in range(max_iter):
         y = x.index_add(0, t, w * x[s])
         y = y / y.norm()
         done = float((y - x).abs().sum()) < n * tol
         x = y
         if done:
            break
      else:
         warnings.warn(f"eigenvector centrality: no convergence in {max_iter} iterations (component {c})")
      out.append(x)
   return out if component is None else out[0]


def pagerank(net, component = 0, weight = None, damping = 0.85, personalization = None, tol = 1e-12, max_iter = 10000):
   """
   PageRank (Brin and Page, 1998): the stationary distribution of a random walker who, at each
   step, follows an out edge with probability ``damping`` (choosing it proportionally to its
   weight), and otherwise jumps to a random node. Nodes without out edges (dangling) always jump.
   Jumps go to a uniformly random node, or, with ``personalization``, to a node drawn from those
   values (personalized PageRank, as NetworkX, where dangling nodes jump the same way).

   The values sum to 1 over the network considered: with ``component = None`` the walker jumps
   between components, so values are comparable across them (one walk on the whole network); with
   a component index, the walk stays in that component (as
   :func:`~graphoskop.stationary_distribution`, which this generalizes). Undirected edges are
   followed both ways; in multilayer networks the walker moves along couplings too. Computed by
   power iteration, one sparse product per step on the device of the network, until the 1-norm
   of the change is below ``tol``.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param damping: Probability of following an edge (``1 - alpha`` of
      :func:`~graphoskop.stationary_distribution`).
   :param personalization: Jump distribution: name of a numeric node attribute, or dictionary (or
      :class:`pandas.Series`) from input node id to value (missing nodes get 0), or ``None`` for
      uniform. Values must be non negative, not all 0; they are normalized to sum 1.
   :param tol: Convergence threshold (1-norm of the change between iterations).
   :param max_iter: Largest number of iterations; a warning is raised if it is reached.
   :return: Vector over the nodes of the component, or a list of them if ``component`` is ``None``.
   :raises ValueError: If weights or personalization values are negative, or the personalization
      is all 0.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", directed = True, device = "cuda")
      pr = gsk.pagerank(net, component = None)                       # whole network
      ppr = gsk.pagerank(net, component = None, personalization = {"alice": 1})
   """
   from .utils import _walk
   comps = list(range(len(net.components))) if component is None else [component]
   dev = net.components[comps[0]].edge_index.device
   sizes = [net.components[c].num_nodes for c in comps]
   off = np.cumsum([0] + sizes)
   ei = torch.cat([net.components[c].edge_index + int(o) for c, o in zip(comps, off)], 1)
   w = torch.cat([torch.ones(net.components[c].edge_index.shape[1], dtype = torch.float64, device = dev) if weight is None else net._numeric_edge(weight, c).double() for c in comps])
   if bool((w < 0).any()):
      raise ValueError("pagerank: weights must be non negative")
   v = None
   if personalization is not None:
      if isinstance(personalization, str):
         v = torch.cat([torch.as_tensor(net.get_node_attribute(personalization, c), device = dev).double() for c in comps])
      else:
         ids = np.concatenate([net.node_ids[c] for c in comps])
         v = torch.as_tensor(pd.Series(ids).map(personalization).fillna(0).to_numpy(dtype = np.float64), device = dev)
      v = v.nan_to_num(0.)
      if bool((v < 0).any()) or not bool((v > 0).any()):
         raise ValueError("pagerank: personalization values must be non negative, not all 0")
      v = v / v.sum()
   phi = _walk(ei, w, int(off[-1]), 1 - damping, tol, max_iter, v)[0]
   out = list(phi.split(sizes))
   return out if component is None else out[0]


def hits(net, component = 0, weight = None, tol = 1e-10, max_iter = 10000):
   """
   Hubs and authorities (HITS; Kleinberg, 1999): a good authority is pointed to by good hubs, a
   good hub points to good authorities, :math:`a \\propto A^T h` and :math:`h \\propto A a`, so
   :math:`a` is the leading eigenvector of :math:`A^T A` and :math:`h` that of :math:`A A^T`.
   Each sums to 1 over the nodes of a component (as NetworkX, which normalizes over the whole
   network); each component is computed on its own, since a single power iteration over several
   components keeps only the one with the largest eigenvalue. In undirected networks hubs and
   authorities are the same. Weights add up over parallel edges; in multilayer networks the scores
   are those of the supra-graph, couplings included.

   Computed by power iteration on :math:`A^T A` (positive semidefinite, so it does not oscillate),
   with two sparse products per step on the device of the network, starting from a constant
   vector, until the authorities change by less than ``tol`` per node (1-norm of the change below
   :math:`n \\cdot` ``tol``). A component without edges gets 0. If the largest eigenvalue of
   :math:`A^T A` is repeated (e.g. two disconnected parts once direction is taken into account),
   the scores are not unique; this iteration returns the ones reached from the constant vector.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param tol: Tolerance per node.
   :param max_iter: Largest number of iterations; a warning is raised if it is reached.
   :return: ``(hubs, authorities)``, two vectors over the nodes of the component, or two lists of
      them if ``component`` is ``None``.
   :raises ValueError: If some weights are negative.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("links.tsv", directed = True, device = "cuda")
      hubs, authorities = gsk.hits(net)
      net.set_node_attribute("authority", authorities)
   """
   hubs, auths = [], []
   for c in range(len(net.components)) if component is None else [component]:
      data = net.components[c]
      n = data.num_nodes
      s, t = data.edge_index
      dev = s.device
      w = torch.ones(s.shape[0], dtype = torch.float64, device = dev) if weight is None else net._numeric_edge(weight, c).double()
      if bool((w < 0).any()):
         raise ValueError("hits: weights must be non negative")
      zero = torch.zeros(n, dtype = torch.float64, device = dev)
      a = torch.full((n,), 1 / n, dtype = torch.float64, device = dev)
      for _ in range(max_iter):
         h = zero.index_add(0, s, w * a[t])
         new = zero.index_add(0, t, w * h[s])
         total = float(new.sum())
         if total == 0:   # no edges
            a = new
            break
         new = new / total
         change = float((new - a).abs().sum())
         a = new
         if change < n * tol:
            break
      else:
         warnings.warn(f"hits: no convergence in {max_iter} iterations (component {c})")
      h = zero.index_add(0, s, w * a[t])
      h = h / h.sum() if float(h.sum()) > 0 else h
      hubs.append(h)
      auths.append(a)
   return (hubs, auths) if component is None else (hubs[0], auths[0])


def _katz_iter(s, t, w, b, alpha, tol, max_iter):
   """
   Solves :math:`x = \\alpha A^T x + b` by iteration from 0 (the series of walks), with one sparse
   product per step.

   :param s: Edge sources (``A_{st} = w``).
   :param t: Edge targets.
   :param w: Edge weights, float64.
   :param b: Constant term, float64.
   :param alpha: Factor :math:`\\alpha`.
   :param tol: Tolerance per node (1-norm of the change below ``n * tol``).
   :param max_iter: Largest number of iterations.
   :return: The solution, or ``None`` if the iteration does not converge (it diverges when
      :math:`|\\alpha| \\lambda_{max} \\geq 1`).
   """
   n = b.shape[0]
   x = torch.zeros_like(b)
   first = None
   for _ in range(max_iter):
      y = b.index_add(0, t, alpha * w * x[s])
      change = float((y - x).abs().sum())
      x = y
      if change < n * tol:
         return x
      first = change if first is None else first
      if not np.isfinite(change) or change > 1e8 * max(first, 1e-300):   # diverging
         return None
   return None


def katz_centrality(net, component = 0, weight = None, alpha = 0.1, beta = 1., direction = "in", normalized = True, tol = 1e-10, max_iter = 10000):
   """
   Katz centrality (Katz, 1953): the walks reaching a node, those of :math:`k` edges discounted by
   :math:`\\alpha^k`, plus a baseline :math:`\\beta_i` for every node,
   :math:`x_i = \\alpha \\sum_j A_{ji} x_j + \\beta_i`, i.e.
   :math:`x = \\beta + \\alpha A^T \\beta + \\alpha^2 (A^T)^2 \\beta + \\ldots`. The series converges
   only if :math:`\\alpha < 1 / \\lambda_{max}`, the reciprocal of the largest eigenvalue of the
   adjacency matrix; as :math:`\\alpha` approaches it the ranking approaches that of
   :func:`eigenvector_centrality`. Unlike eigenvector centrality it is positive on every node,
   also in directed acyclic networks.

   In directed networks, ``direction = "in"`` counts the walks reaching a node (as NetworkX),
   ``"out"`` those leaving it. Weights add up over parallel edges; a self loop counts once in
   undirected networks. In multilayer networks walks run on the supra-graph, couplings included.

   Computed by iterating the equation from :math:`x = 0`, with one sparse product per step on the
   device of the network, until the vector changes by less than ``tol`` per node (1-norm of the
   change below :math:`n \\cdot` ``tol``). With ``normalized = True`` the result has unit Euclidean
   norm over the network considered: with ``component = None`` all components are normalized
   together, so values are comparable across them. If :math:`\\beta` is 0 everywhere, so is the
   result.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param alpha: Attenuation factor :math:`\\alpha`.
   :param beta: Baseline :math:`\\beta`: a number for all nodes, the name of a numeric node
      attribute, or a dictionary (or :class:`pandas.Series`) from input node id to value (missing
      nodes get 0).
   :param direction: ``"in"`` or ``"out"``; ignored in undirected networks.
   :param normalized: Scale to unit Euclidean norm.
   :param tol: Tolerance per node.
   :param max_iter: Largest number of iterations.
   :return: Vector over the nodes of the component, or a list of them if ``component`` is ``None``.
   :raises ValueError: If ``direction`` is not valid, weights are negative, or the iteration does
      not converge (``alpha`` too large).

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", directed = True, device = "cuda")
      x = gsk.katz_centrality(net, alpha = 0.05)
      xs = gsk.katz_centrality(net, component = None, beta = "prior")   # node attribute as baseline
   """
   if direction not in ("in", "out"):
      raise ValueError(f"unknown direction: {direction}")
   comps = list(range(len(net.components))) if component is None else [component]
   out = []
   for c in comps:
      data = net.components[c]
      n = data.num_nodes
      s, t = data.edge_index
      if net.directed and direction == "out":
         s, t = t, s
      dev = s.device
      w = torch.ones(s.shape[0], dtype = torch.float64, device = dev) if weight is None else net._numeric_edge(weight, c).double()
      if bool((w < 0).any()):
         raise ValueError("katz centrality: weights must be non negative")
      if isinstance(beta, str):
         b = torch.as_tensor(net.get_node_attribute(beta, c), device = dev).double().nan_to_num(0.)
      elif isinstance(beta, (dict, pd.Series)):
         b = torch.as_tensor(pd.Series(net.node_ids[c]).map(beta).fillna(0).to_numpy(dtype = np.float64), device = dev)
      else:
         b = torch.full((n,), float(beta), dtype = torch.float64, device = dev)
      x = _katz_iter(s, t, w, b, alpha, tol, max_iter)
      if x is None:
         raise ValueError(f"katz centrality: no convergence in {max_iter} iterations (component {c}); alpha must be below 1 / largest eigenvalue of the adjacency matrix (see alpha_centrality)")
      out.append(x)
   if normalized:
      norm = torch.cat(out).norm()
      out = [x / norm if norm > 0 else x for x in out]
   return out if component is None else out[0]


def alpha_centrality(net, component = 0, weight = None, alpha = 1., exo = 1., direction = "in", loops = False, tol = 1e-10, max_iter = 10000):
   """
   Alpha centrality (Bonacich and Lloyd, 2001): status received from the neighbors plus an
   exogenous source :math:`e_i` of status for every node,
   :math:`x_i = \\alpha \\sum_j A_{ji} x_j + e_i`, i.e. :math:`x = (I - \\alpha A^T)^{-1} e`.

   It is the equation of :func:`katz_centrality` with :math:`\\beta = e`, but defined by the
   matrix inverse rather than by the series of walks: :math:`\\alpha` can be any value for which
   :math:`I - \\alpha A^T` is invertible, also above :math:`1 / \\lambda_{max}` (where the series
   diverges and values can be negative) or negative. Results are not normalized. Defaults are
   those of R igraph's ``alpha_centrality``: :math:`\\alpha = 1`, :math:`e = 1`, self loops
   ignored.

   In directed networks, ``direction = "in"`` gives a node the status of the nodes with edges to
   it (as igraph), ``"out"`` that of the nodes it has edges to. Weights add up over parallel edges;
   a kept self loop counts once in undirected networks. In multilayer networks the status flows on
   the supra-graph, couplings included.

   The system is solved as :func:`katz_centrality` does, by iterating the equation on the device
   of the network, which converges if :math:`|\\alpha| \\lambda_{max} < 1`, until the vector
   changes by less than ``tol`` per node. If the iteration diverges (or does not converge in
   ``max_iter`` steps), the system is solved exactly by sparse LU factorization (SuperLU, through
   :mod:`scipy.sparse.linalg`) on the CPU. This is fast on small or sparse, clustered networks,
   but its cost can grow quickly (to the cube of the number of nodes) on large networks with
   random-like structure, e.g. minutes for 20,000 nodes and 100,000 random edges. The default
   :math:`\\alpha = 1` is above :math:`1 / \\lambda_{max}` for most networks, so it always takes
   this route.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param alpha: Relative importance :math:`\\alpha` of the endogenous status.
   :param exo: Exogenous status :math:`e`: a number for all nodes, the name of a numeric node
      attribute, or a dictionary (or :class:`pandas.Series`) from input node id to value (missing
      nodes get 0).
   :param direction: ``"in"`` or ``"out"``; ignored in undirected networks.
   :param loops: Keep self loops.
   :param tol: Tolerance per node of the iteration.
   :param max_iter: Largest number of iterations before the exact solve.
   :return: Vector over the nodes of the component, or a list of them if ``component`` is ``None``.
   :raises ValueError: If ``direction`` is not valid, or the iteration does not converge and
      :math:`I - \\alpha A^T` is singular up to rounding (:math:`1 / \\alpha` is an eigenvalue of
      the adjacency matrix: a pivot of the LU factorization is below :math:`10^{-12}` times the
      largest). If the matrix is singular but the iteration converges (e.g. :math:`e = 0`), its
      result, a solution of the equation, is returned.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", directed = True)
      x = gsk.alpha_centrality(net, alpha = 0.5, exo = "wealth")
   """
   from scipy.sparse.linalg import splu
   if direction not in ("in", "out"):
      raise ValueError(f"unknown direction: {direction}")
   out = []
   for c in range(len(net.components)) if component is None else [component]:
      data = net.components[c]
      n, dev = data.num_nodes, data.edge_index.device
      s, t = data.edge_index
      if net.directed and direction == "out":
         s, t = t, s
      w = torch.ones(s.shape[0], dtype = torch.float64, device = dev) if weight is None else net._numeric_edge(weight, c).double()
      if not loops:
         keep = s != t
         s, t, w = s[keep], t[keep], w[keep]
      if isinstance(exo, str):
         e = torch.as_tensor(net.get_node_attribute(exo, c), device = dev).double().nan_to_num(0.)
      elif isinstance(exo, (dict, pd.Series)):
         e = torch.as_tensor(pd.Series(net.node_ids[c]).map(exo).fillna(0).to_numpy(dtype = np.float64), device = dev)
      else:
         e = torch.full((n,), float(exo), dtype = torch.float64, device = dev)
      x = _katz_iter(s, t, w, e, alpha, tol, max_iter)
      if x is None:   # series diverges: exact solve
         s, t, w = s.cpu().numpy(), t.cpu().numpy(), w.cpu().numpy()
         M = sp.identity(n, format = "csc") - alpha * sp.csc_matrix((w, (t, s)), shape = (n, n))
         try:
            lu = splu(M)
            d = np.abs(lu.U.diagonal())
            x = lu.solve(e.cpu().numpy()) if d.min() > 1e-12 * d.max() else None   # else singular up to rounding
         except RuntimeError:   # exactly singular
            x = None
         if x is None or not np.isfinite(x).all():
            raise ValueError(f"alpha centrality: I - alpha A^T is singular (component {c}); 1 / alpha is an eigenvalue of the adjacency matrix")
      out.append(torch.as_tensor(x, dtype = torch.float64, device = dev))
   return out if component is None else out[0]


def springrank(net, component = 0, weight = None, alpha = 0., tol = 1e-10, max_iter = None):
   """
   SpringRank (De Bacco, Larremore and Moore, 2018): the ranks :math:`s` of the nodes of a
   directed network that minimize the energy of a system of springs, one per edge, each of rest
   length 1,

   .. math::

      H(s) = \\frac{1}{2} \\sum_{i, j} A_{ij} (s_i - s_j - 1)^2 + \\frac{\\alpha}{2} \\sum_i s_i^2,

   so that an edge from :math:`i` to :math:`j` pulls :math:`i` one unit above :math:`j` (e.g.
   :math:`i` beats :math:`j`, or :math:`i` places a graduate at :math:`j`; to rank the targets of
   the edges higher, reverse them with :func:`~graphoskop.flip_edge_direction`). The minimum
   solves the linear system

   .. math::

      \\left[D^{out} + D^{in} - (A + A^T) + \\alpha I\\right] s = d^{out} - d^{in},

   with :math:`d^{out}`, :math:`d^{in}` the (weighted) out- and in-degrees: the Laplacian of the
   network with directions ignored, so the system is solved by conjugate gradient (Jacobi
   preconditioned), one sparse product per iteration on the device of the network, in double
   precision. Without regularization (:math:`\\alpha = 0`) ranks are defined up to a shift in each
   component: they are centered to mean 0 in each component, as in the reference implementation,
   so ranks are comparable only within a component. With :math:`\\alpha > 0` they are pulled
   towards 0 and are unique.

   Ranks are in units of the rest length: an edge from :math:`i` to :math:`j` is relaxed when
   :math:`s_i = s_j + 1`. Parallel edges add their weights; self loops pull nothing. In
   multilayer networks the supra-graph is used.

   :param net: A directed :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param weight: Name of the edge attribute used as edge weight (number of interactions), or
      ``None`` for unit weights.
   :param alpha: Regularization :math:`\\alpha \\ge 0`.
   :param tol: Relative residual tolerance of the conjugate gradient.
   :param max_iter: Largest number of iterations, ``10 * n`` if ``None``; a warning is raised if
      it is reached.
   :return: float64 vector of ranks over the nodes of the component, on its device (a list of them,
      one per component, with ``component = None``), higher for higher ranked nodes.
   :raises ValueError: If the network is undirected (all ranks would be 0), weights are negative,
      or ``alpha`` is negative.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("dominance.tsv", directed = True)   # winner, loser, number of fights
      s = gsk.springrank(net, weight = "weight")
      net.set_node_attribute("springrank", s)
      s = gsk.springrank(net, component = None, alpha = 1)       # regularized, all components
   """
   if not net.directed:
      raise ValueError("springrank needs a directed network")
   if alpha < 0:
      raise ValueError("springrank: alpha must be non negative")
   comps = list(range(len(net.components))) if component is None else [component]
   dev = net.components[comps[0]].edge_index.device
   sizes = [net.components[c].num_nodes for c in comps]
   off = np.cumsum([0] + sizes)
   n = int(off[-1])
   s, t = torch.cat([net.components[c].edge_index + int(o) for c, o in zip(comps, off)], 1)
   w = torch.cat([torch.ones(net.components[c].edge_index.shape[1], dtype = torch.float64, device = dev) if weight is None else net._numeric_edge(weight, c).double() for c in comps])
   if bool((w < 0).any()):
      raise ValueError("springrank: weights must be non negative")
   keep = s != t
   s, t, w = s[keep], t[keep], w[keep]
   dout = torch.zeros(n, dtype = torch.float64, device = dev).index_add_(0, s, w)
   din = torch.zeros(n, dtype = torch.float64, device = dev).index_add_(0, t, w)
   diag = dout + din + alpha
   mv = lambda x: diag * x - torch.zeros_like(x).index_add_(0, s, w * x[t]).index_add_(0, t, w * x[s])
   b = dout - din
   pre = torch.where(diag > 0, 1 / diag, 1.)
   x = torch.zeros_like(b)
   r = b.clone()
   z = pre * r
   p = z.clone()
   rz = r @ z
   stop = tol * float(b.norm())
   max_iter = 10 * n if max_iter is None else max_iter
   for it in range(max_iter):
      if float(r.norm()) <= stop:
         break
      Ap = mv(p)
      step = rz / (p @ Ap)
      x += step * p
      r -= step * Ap
      z = pre * r
      rz, old = r @ z, rz
      p = z + (rz / old) * p
   else:
      if float(r.norm()) > stop:
         warnings.warn(f"springrank: conjugate gradient did not converge in {max_iter} iterations")
   out = list(torch.split(x, sizes))
   if alpha == 0:
      out = [v - v.mean() for v in out]
   return out if component is None else out[0]


def k_core_centrality(net, component = 0, mode = "all"):
   """
   k-core centrality, or core number (Seidman, 1983): the largest :math:`k` such that the node
   belongs to the :math:`k`-core, the largest subgraph in which every node has at least :math:`k`
   neighbors (obtained by repeatedly removing the nodes with fewer than :math:`k` neighbors).
   Nodes without neighbors have 0.

   Edges count once whatever their weight and multiplicity, and self loops are ignored (as
   NetworkX's ``core_number``). In directed networks, ``mode`` chooses the degree used: ``"all"``
   (in plus out, as NetworkX, where a pair of reciprocal edges counts twice), ``"in"`` or ``"out"``
   (as igraph's ``coreness``). In multilayer networks cores are those of the supra-graph, where
   couplings are edges like the others.

   Computed by peeling on the device of the network: at level :math:`k` (starting at the smallest
   degree), all nodes with degree at most :math:`k` are removed at once and the degrees of their
   neighbors updated, wave after wave, until none is left at that level; they get core number
   :math:`k`. Each wave only touches the edges of the nodes it removes, so the work is
   proportional to the number of edges, plus a small fixed cost per wave and a pass over the
   nodes per level. Waves are few on most networks; long chains of nodes (e.g. a path) are peeled
   one node from each end per wave.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param mode: ``"all"``, ``"in"`` or ``"out"``; ignored in undirected networks.
   :return: int64 vector over the nodes of the component, or a list of them if ``component`` is
      ``None``.
   :raises ValueError: If ``mode`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      core = gsk.k_core_centrality(net)
      net.set_node_attribute("core", core)
      main_core = net.node_ids[0][(core == core.max()).cpu().numpy()]
   """
   if mode not in ("all", "in", "out"):
      raise ValueError(f"unknown mode: {mode}")
   out = []
   for c in range(len(net.components)) if component is None else [component]:
      data = net.components[c]
      n = data.num_nodes
      s, t = data.edge_index
      dev = s.device
      keep = s != t
      key = torch.unique(s[keep] * n + t[keep])   # simple graph, sorted (undirected: both directions stored)
      s, t = key // n, key % n

      def csr(rows, cols):   # neighbor lists: pointers and columns
         order = torch.argsort(rows, stable = True)
         ptr = torch.zeros(n + 1, dtype = torch.long, device = dev)
         ptr[1:] = torch.bincount(rows, minlength = n).cumsum(0)
         return ptr, cols[order]

      # a node's degree counts its out edges (in edges for "in"); removing r takes one from the
      # nodes with an edge to r (from r for "in")
      lists = [csr(t, s)] if not net.directed or mode == "out" else [csr(s, t)] if mode == "in" else [csr(t, s), csr(s, t)]
      ends = [s] if not net.directed or mode == "out" else [t] if mode == "in" else [s, t]
      deg = sum(torch.bincount(x, minlength = n) for x in ends)
      core = torch.zeros(n, dtype = torch.long, device = dev)
      alive = torch.ones(n, dtype = torch.bool, device = dev)
      k = 0
      while bool(alive.any()):
         k = max(k, int(deg[alive].min()))
         rem = (alive & (deg <= k)).nonzero()[:, 0]
         while rem.shape[0]:   # waves at level k: only the neighbors of removed nodes can follow
            core[rem] = k
            alive[rem] = False
            nb = []
            for ptr, col in lists:
               start = ptr[rem]
               cnt = ptr[rem + 1] - start
               idx = torch.arange(int(cnt.sum()), device = dev) + torch.repeat_interleave(start - (cnt.cumsum(0) - cnt), cnt)
               nb.append(col[idx])
            nb = torch.cat(nb)
            deg.index_add_(0, nb, torch.full_like(nb, -1))
            nb = nb.unique()
            rem = nb[alive[nb] & (deg[nb] <= k)]
      out.append(core)
   return out if component is None else out[0]


def _values(x):
   """
   Centrality values as one float64 NumPy array (lists of vectors, one per component, are joined).
   """
   if isinstance(x, (list, tuple)):
      return np.concatenate([_values(v) for v in x]) if len(x) else np.zeros(0)
   if torch.is_tensor(x):
      return x.double().cpu().numpy().ravel()
   return np.asarray(x, dtype = np.float64).ravel()


def centralization(net, centrality = "degree", component = 0, maximum = None, **kwargs):
   """
   Centralization of a network for a centrality measure (Freeman, 1979): how much the most central
   node exceeds all the others, relative to the largest possible such excess in a network with the
   same number of nodes,

   .. math:: C = \\frac{\\sum_i (c_{max} - c_i)}{\\max_{G'} \\sum_i (c_{max}(G') - c_i(G'))}.

   It is 1 for a star (for the usual measures), 0 when all nodes are equally central. The maximum
   is computed by running the same centrality, with the same options, on a star with the same
   number of nodes (in directed networks the largest over the star with edges out of the center,
   the star with edges into it and the star with edges both ways), so it follows the normalization
   and the options of the centrality used. The star is the most centralized network for degree,
   closeness and betweenness (Freeman, 1979) and for eigenvector centrality, as used by igraph's
   centralization functions; for other measures it is the conventional reference but not
   necessarily the maximum (values can then exceed 1), and for some it is meaningless (a star has
   all core numbers equal to 1, so the result of :func:`k_core_centrality` is NaN): pass
   ``maximum`` then. See :func:`~graphoskop.degree_centralization` for a centralization of degrees
   based on entropy and random networks instead.

   Every edge of the star has, for each edge attribute of ``net``, its mean value over the network
   considered, so that ``weight`` options work and the result does not depend on the units of the
   weights. The star has no node attributes: options naming node attributes (e.g. ``beta`` of
   :func:`katz_centrality` given as an attribute name) cannot be used without ``maximum``. Measures
   that cannot be computed on the star raise an error asking for ``maximum``: e.g.
   :func:`katz_centrality` and :func:`alpha_centrality` with an :math:`\\alpha` above
   :math:`1 / \\sqrt{n - 1}`, the reciprocal of the largest eigenvalue of a star. The result also
   depends on how the centrality is normalized: e.g. :func:`eigenvector_centrality` has unit
   Euclidean norm, while igraph scales it to a largest value of 1, which gives a different
   centralization (pass a function rescaling the vector to reproduce it).

   :param net: A :class:`~graphoskop.Network`.
   :param centrality: A centrality function of this module (or any function taking the network,
      ``component`` and the options, and returning one value per node, or a list of vectors for
      ``component = None``), or ``"degree"`` for the degree (options ``direction``, ``"out"``,
      ``"in"`` or ``"total"``, and ``weight``, as :func:`~graphoskop.degree`).
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network
      (compared with a star of all its nodes).
   :param maximum: Largest sum of differences, or ``None`` to compute it on the star.
   :param kwargs: Options of the centrality, used for the network and the star.
   :return: The centralization, a float (NaN for fewer than 2 nodes or a maximum of 0).

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.centralization(net)                                       # degree centralization
      gsk.centralization(net, gsk.betweenness_centrality)
      gsk.centralization(net, gsk.closeness_centrality, weight = "weight")
      dnet = gsk.load_network("edges.tsv", directed = True)
      gsk.centralization(dnet, "degree", direction = "in")
   """
   if centrality == "degree":
      from .network_statistics import _degrees

      def centrality(net, component, direction = "out", weight = None):
         comps = range(len(net.components)) if component is None else [component]
         return [_degrees(net, c, direction, weight) for c in comps]
   c = _values(centrality(net, component = component, **kwargs))
   if len(c) < 2:
      return float("nan")
   total = float((c.max() - c).sum())
   if maximum is None:
      from .network import load_network
      n = len(c)
      a, b = np.zeros(n - 1, dtype = np.int64), np.arange(1, n)
      stars = [(a, b)] if not net.directed else [(a, b), (b, a), (np.r_[a, b], np.r_[b, a])]
      maximum = 0.
      comps = range(len(net.components)) if component is None else [component]
      means = [np.nanmean(np.concatenate([net.get_edge_attribute(name, c).double().cpu().numpy() for c in comps])) for name in net.edge_attributes]
      for s, t in stars:
         e = pd.DataFrame({"source": s, "target": t})
         for name, v in zip(net.edge_attributes, means):
            e[name] = v if np.isfinite(v) else 1.
         star = load_network(e, directed = net.directed, device = net.components[0].edge_index.device)
         try:
            x = _values(centrality(star, component = 0, **kwargs))
         except ValueError as err:
            raise ValueError(f"centralization: the centrality cannot be computed on the reference star of {n} nodes ({err}); pass maximum") from err
         maximum = max(maximum, float((x.max() - x).sum()))
   return total / maximum if maximum > 0 else float("nan")
