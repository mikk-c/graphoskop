"""
Hierarchy in directed networks: how far the nodes are organized in levels, with some nodes
reaching (or controlling) many others and being reached by few.

:func:`global_reach_centrality` measures it from the local reaching centrality of the nodes (see
:mod:`~graphoskop.centrality`); :func:`flow_hierarchy` from the edges (or nodes) in no directed
cycle, through which flows only go one way; :func:`arborescence_hierarchy` from the edges that
survive the reduction of the network to an arborescence; :func:`agony_hierarchy` from the agony of
the best division of the nodes into levels.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .centrality import reach_centrality
from .network_statistics import _strong_labels, connected_components_nodes
from .shortest_paths import _source_stats
from .transformations import _restrict, network_condensation, subgraph


def global_reach_centrality(net, component = 0, direction = "out", lengths = None, n_jobs = None):
   """
   Global reaching centrality (Mones, Vicsek and Vicsek, 2012), a measure of hierarchy: how much
   the most reaching node exceeds the others,
   :math:`GRC = \\sum_i (C_R^{max} - C_R(i)) / (N - 1)`, with :math:`C_R` the local reaching
   centrality of :func:`~graphoskop.reach_centrality` and :math:`C_R^{max}` its largest value. It
   is 1 for a star with edges out of its center and 0 when every node reaches all the others (e.g.
   a directed cycle). With ``component = None`` the sum and the maximum run over all components.

   :param net: A directed :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components at
      once.
   :param direction: ``"out"`` (nodes reached) or ``"in"`` (nodes reaching).
   :param lengths: Precomputed lengths, as for :func:`~graphoskop.reach_centrality`, or ``None``.
   :param n_jobs: Number of processes for the searches, ``None`` for all CPUs.
   :return: The global reaching centrality, a float.
   :raises ValueError: If the network is undirected, ``direction`` is not valid, or ``lengths``
      does not match.

   Example:

   .. code-block:: python

      gsk.global_reach_centrality(net)                    # largest component
   """
   r = reach_centrality(net, component, direction, lengths, n_jobs)
   r = torch.cat(r) if component is None else r
   N = r.shape[0]
   return float((r.max() - r).sum() / (N - 1)) if N > 1 else 0.


def flow_hierarchy(net, component = 0, weight = None, by = "edges"):
   """
   Flow hierarchy (Luo and Magee, 2011), a measure of hierarchy: the fraction of the edges that
   are in no directed cycle, i.e. that join two different strongly connected components (a self
   loop is a cycle). It is 1 for a directed acyclic graph, where flows only go down the levels,
   and 0 when every edge is in a cycle (e.g. a strongly connected network).

   With ``weight``, the fraction of the total weight; with ``by = "nodes"``, the fraction of the
   nodes in no cycle (alone in their strongly connected component, without self loop) instead of
   the edges. Strongly connected components are found on the CPU
   (:func:`scipy.sparse.csgraph.connected_components`); in multilayer networks on the
   supra-graph. With ``component = None`` all components count.

   :param net: A directed :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights
      (ignored with ``by = "nodes"``).
   :param by: ``"edges"`` (as in the paper) or ``"nodes"``.
   :return: The flow hierarchy, a float between 0 and 1 (NaN without edges, or without nodes).
   :raises ValueError: If the network is undirected, ``by`` is not valid, or weights are negative.

   Example:

   .. code-block:: python

      gsk.flow_hierarchy(net)                            # fraction of edges in no cycle
      gsk.flow_hierarchy(net, component = None, weight = "weight")
      gsk.flow_hierarchy(net, by = "nodes")              # fraction of nodes in no cycle
   """
   if not net.directed:
      raise ValueError("flow_hierarchy needs a directed network")
   if by not in ("edges", "nodes"):
      raise ValueError(f"unknown by: {by}")
   out, total = 0., 0.
   for c in range(len(net.components)) if component is None else [component]:
      _, lab = _strong_labels(net, c)
      s, t = net.components[c].edge_index.cpu().numpy()
      cyc = lab[s] == lab[t]   # in a cycle
      if by == "edges":
         w = np.ones(len(s)) if weight is None else net._numeric_edge(weight, c).double().cpu().numpy()
         if (w < 0).any():
            raise ValueError("flow_hierarchy: weights must be non negative")
         out, total = out + w[~cyc].sum(), total + w.sum()
      else:
         n = net.components[c].num_nodes
         size = np.bincount(lab)[lab]
         loop = np.zeros(n, dtype = bool)
         loop[s[s == t]] = True
         out, total = out + ((size == 1) & ~loop).sum(), total + n
   return float(out / total) if total > 0 else float("nan")


def arborescence_hierarchy(net, component = 0, arborescence = False):
   """
   Arborescence hierarchicalness (Coscia, 2018): the fraction of the edges of a directed network
   that survive its reduction to an arborescence (a forest of trees with edges pointing away from
   their roots), :math:`A_G = |E^*| / |E|`. It is 1 for a network that already is an
   arborescence, and close to 0 for a network that is one big cycle.

   The reduction has two steps. First, the condensation: every strongly connected component
   collapses into one node, its internal edges are lost, and the edges between two components
   become one (see :func:`~graphoskop.network_condensation`). Then, every node with several parents
   keeps only the edge from the parent with the lowest closeness, :math:`1 / \\sum_j d(i, j)` over
   the nodes :math:`j` it reaches (0 for nodes reaching none): the parent closest to the roots. As
   every node with a parent keeps exactly one edge, :math:`|E^*|` is the number of nodes of the
   condensation with at least one parent, so the score needs only the strongly connected
   components (found on the CPU, :math:`O(n + m)`); the closeness, a search from every node of the
   condensation (as in :func:`~graphoskop.closeness_centrality`), is computed only to return the
   arborescence, and is the costly part. Edges point from higher to lower levels: to read them the other way, reverse
   them with :func:`~graphoskop.flip_edge_direction`. :math:`|E|` counts the distinct edges of
   the network (parallel edges once, self loops included).

   :param net: A directed :class:`~graphoskop.Network`, not multilayer.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param arborescence: Whether to also return the arborescence.
   :return: The score, a float between 0 and 1 (NaN without edges); with ``arborescence``, the pair
      ``(score, forest)``, the forest a directed :class:`~graphoskop.Network` whose node ``i``
      collapses the ``i``-th strongly connected component of the network considered (largest
      first), with the qualitative node attribute ``members`` holding its nodes.
   :raises ValueError: If the network is undirected or multilayer.

   Example:

   .. code-block:: python

      gsk.arborescence_hierarchy(net)                        # largest component
      score, forest = gsk.arborescence_hierarchy(net, component = None, arborescence = True)
      forest.get_node_attribute("members", None)             # nodes of each condensed node
   """
   if not net.directed:
      raise ValueError("arborescence_hierarchy needs a directed network")
   if net.layer_labels is not None:
      raise ValueError("arborescence_hierarchy is not defined for multilayer networks")
   kept, total = 0, 0
   for c in range(len(net.components)) if component is None else [component]:
      k, lab = _strong_labels(net, c)
      s, t = net.components[c].edge_index.cpu().numpy()
      n = net.components[c].num_nodes
      total += len(np.unique(s.astype(np.int64) * n + t))
      between = lab[s] != lab[t]
      kept += len(np.unique(lab[t[between]]))   # condensed nodes with a parent
   score = kept / total if total else float("nan")
   if not arborescence:
      return score
   sub = net if component is None else subgraph(net, [(component, i) for i in range(net.components[component].num_nodes)])
   members = connected_components_nodes(sub, type = "strong")
   cond = network_condensation(sub)
   keep = []
   for c, data in enumerate(cond.components):
      s, t = data.edge_index.cpu().numpy()
      dist = _source_stats(cond, c, np.arange(data.num_nodes), None, False, None)[1]
      close = np.where(dist > 0, 1 / np.where(dist > 0, dist, 1), 0)
      best = np.lexsort((s, close[s], t))   # by child, then parent closeness, then parent index
      first = np.r_[True, t[best][1:] != t[best][:-1]][:len(s)]   # no edges: empty
      mask = np.zeros(len(s), dtype = bool)
      mask[best[first]] = True
      keep.append(mask)
   forest = _restrict(cond, [np.ones(d.num_nodes, dtype = bool) for d in cond.components], keep)
   forest.set_node_attribute(pd.DataFrame({"id": np.arange(len(members)), "members": [tuple(m) for m in members]}), node_column = "id")
   return score, forest


def agony_hierarchy(net, component = 0, weight = None, ranks = False):
   """
   Agony (Gupte et al., 2011; Tatti, 2015), a measure of hierarchy: the levels of the nodes, 0 at
   the top, are chosen so that edges point from higher to lower levels as much as possible. An
   edge from :math:`u` to :math:`v` costs nothing if :math:`v` is at least one level below
   :math:`u`, and otherwise its agony :math:`r(u) - r(v) + 1` (times its weight), so the agony of
   the network is

   .. math::

      A(G) = \\min_r \\sum_{(u, v) \\in E} w_{uv} \\max\\left(0, r(u) - r(v) + 1\\right),

   and its hierarchy :math:`h = 1 - A(G) / W`, with :math:`W` the total weight of the edges (their
   number without ``weight``): 1 for a directed acyclic graph, 0 when every edge is in a directed
   cycle that cannot be ordered (e.g. a directed cycle, whose agony is its length). Self loops
   always cost their weight; parallel edges add their weights.

   The minimum is a linear program whose dual is a maximum circulation: :math:`A(G)` is the
   largest total flow of a circulation in which each edge carries at most its weight (for unit
   weights, the largest Eulerian subgraph). It is solved exactly by HiGHS
   (:func:`scipy.optimize.linprog`: interior point with crossover, or dual simplex if needed; on the
   CPU), one component at a time, and the levels are the dual values of the flow conservation
   constraints, integers (the network matrix is totally unimodular), shifted so that the top level
   is 0. The levels found are one optimal ranking among possibly many. The time grows faster than
   the number of edges: about 4 s for 100,000 edges and 20 s for 300,000 on two CPU cores.

   :param net: A directed :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components
      (agonies and weights summed).
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param ranks: Whether to also return the levels.
   :return: The hierarchy :math:`h`, a float (NaN without edges); with ``ranks``, the pair
      ``(h, levels)``, the levels an int64 vector over the nodes of the component, on its device
      (a list of them, one per component, with ``component = None``). The agony is
      :math:`(1 - h) W`.
   :raises ValueError: If the network is undirected or weights are negative.
   :raises RuntimeError: If the linear program solver fails.

   Example:

   .. code-block:: python

      h = gsk.agony_hierarchy(net)
      h, levels = gsk.agony_hierarchy(net, weight = "weight", ranks = True)
      net.set_node_attribute("level", levels)
   """
   from scipy.optimize import linprog
   if not net.directed:
      raise ValueError("agony_hierarchy needs a directed network")
   agony, total, levels = 0., 0., []
   for c in range(len(net.components)) if component is None else [component]:
      data = net.components[c]
      n = data.num_nodes
      s, t = data.edge_index.cpu().numpy()
      w = np.ones(len(s)) if weight is None else net._numeric_edge(weight, c).double().cpu().numpy()
      if (w < 0).any():
         raise ValueError("agony_hierarchy: weights must be non negative")
      total += w.sum()
      r = np.zeros(n, dtype = np.int64)
      if len(s):
         A = sp.csr_matrix((np.r_[np.ones(len(s)), -np.ones(len(s))], (np.r_[s, t], np.r_[np.arange(len(s)), np.arange(len(s))])), shape = (n, len(s)))
         found = None
         for method in ("highs-ipm", "highs-ds"):   # interior point with crossover, faster; dual simplex if its duals are not integers
            res = linprog(-np.ones(len(s)), A_eq = A, b_eq = np.zeros(n), bounds = np.c_[np.zeros(len(s)), w], method = method)
            if res.status != 0:
               raise RuntimeError(f"agony_hierarchy: the linear program failed ({res.message})")
            best = -res.fun
            for sign in (1, -1):   # levels are the duals, up to their sign convention
               y = np.rint(sign * res.eqlin.marginals).astype(np.int64)
               if abs((w * np.maximum(0, y[s] - y[t] + 1)).sum() - best) <= 1e-6 * max(1, best):
                  found = y - y.min()
                  break
            if found is not None:
               break
         if found is None:
            raise RuntimeError("agony_hierarchy: could not recover integer levels from the solution")
         r = found
         agony += best
      levels.append(torch.as_tensor(r, device = data.edge_index.device))
   h = float(1 - agony / total) if total > 0 else float("nan")
   if not ranks:
      return h
   return h, (levels if component is None else levels[0])
