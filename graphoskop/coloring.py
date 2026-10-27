"""
Graph coloring and independent sets.

A coloring assigns a color to every node so that joined nodes have different colors: the nodes of
each color form an independent set (no two of them joined). Both the fewest colors needed (the
chromatic number) and the largest independent set (the independence number) are NP-hard to find,
so the functions here combine bounds, which often settle the answer at once on real networks, with
exact searches that stop after ``time_limit`` seconds; if a search is stopped, the best solution
found is returned with a warning saying it is not proven optimal.

Edges are taken as undirected and unweighted (in directed networks, two nodes are joined if there
is an edge in either direction); self loops are ignored (a node joined to itself could not be
colored, nor be in an independent set). In multilayer networks only intra-layer edges count, so
each layer is colored on its own (couplings would force the copies of an entity to have different
colors). Components are solved independently. The searches run on the CPU.
"""

import sys
import time
import warnings

import numpy as np
import torch

from ._policy import _higher_order
from .communities import _bron_kerbosch


class _Timeout(Exception):
   pass


def _adjacency(net, component):
   """
   Undirected adjacency of a component, without self loops, parallel edges or interlayer edges.

   :return: List of sets, one per node (internal index).
   """
   data = net.components[component]
   s, t = data.edge_index
   keep = s != t
   if net.layer_labels is not None:
      keep &= ~net.interlayer_mask(component)
   adj = [set() for _ in range(data.num_nodes)]
   for a, b in zip(s[keep].tolist(), t[keep].tolist()):
      adj[a].add(b)
      adj[b].add(a)
   return adj


def _dsatur(adj, nodes, k = None, deadline = None):
   """
   DSATUR (Brelaz, 1979): color next the node with the most distinct colors among its neighbors
   (ties: most uncolored neighbors). Without ``k``, the greedy heuristic; with ``k``, an exact
   backtracking search for a coloring with at most ``k`` colors.

   :param adj: Adjacency sets.
   :param nodes: Nodes to color (their neighbors outside are ignored).
   :param k: Number of colors allowed, or ``None`` for the heuristic.
   :param deadline: Time after which the search raises :class:`_Timeout`.
   :return: Dictionary from node to color, or ``None`` if no ``k``-coloring exists.
   """
   nodes = set(nodes)
   color = {}
   seen = {v: {} for v in nodes}   # colors among neighbors, with their counts

   def pick():
      return max((v for v in nodes if v not in color), key = lambda v: (len(seen[v]), len(adj[v] & nodes)))

   def assign(v, c, sign):
      for u in adj[v] & nodes:
         d = seen[u]
         d[c] = d.get(c, 0) + sign
         if d[c] == 0:
            del d[c]

   if k is None:
      while len(color) < len(nodes):
         v = pick()
         c = next(c for c in range(len(nodes)) if c not in seen[v])
         color[v] = c
         assign(v, c, 1)
      return color

   def search(used):
      if deadline is not None and time.time() > deadline:
         raise _Timeout
      if len(color) == len(nodes):
         return True
      v = pick()
      for c in range(min(used + 1, k)):   # an existing color, or one new color (symmetry breaking)
         if c in seen[v]:
            continue
         color[v] = c
         assign(v, c, 1)
         if search(max(used, c + 1)):
            return True
         assign(v, c, -1)
         del color[v]
      return False

   return dict(color) if search(0) else None


def _color_component(adj, deadline):
   """
   Coloring of one component with as few colors as can be proven within the deadline.

   :return: ``(color of each node, number of colors, lower bound, proven optimal)``.
   """
   n = len(adj)
   if not any(adj):
      return [0] * n, 1, 1, True
   best = _dsatur(adj, range(n))
   ub = max(best.values()) + 1
   tri = {v: {u for u in adj[v] if adj[u] & adj[v]} for v in range(n)}   # edges in triangles
   tri = {v: a for v, a in tri.items() if a}
   lb = max([len(c) for c in _bron_kerbosch(tri, 3)] + [2])   # largest clique
   proven = ub == lb
   k = ub - 1
   while not proven and k >= lb:   # is there a k-coloring?
      core, deg = set(range(n)), [len(a) for a in adj]
      low = [v for v in core if deg[v] < k]   # nodes with fewer than k neighbors: colored last
      peeled = []
      while low:
         v = low.pop()
         if v in core:
            core.discard(v)
            peeled.append(v)
            for u in adj[v] & core:
               deg[u] -= 1
               if deg[u] == k - 1:
                  low.append(u)
      try:
         found = _dsatur(adj, core, k, deadline)
      except _Timeout:
         break
      if found is None:
         proven = True
         break
      for v in reversed(peeled):   # always a free color: fewer than k neighbors
         found[v] = next(c for c in range(k) if c not in {found[u] for u in adj[v] if u in found})
      best, ub = found, k
      k -= 1
   proven = proven or ub == lb
   return [best[v] for v in range(n)], ub, lb, proven


@_higher_order()
def graph_coloring(net, component = None, time_limit = 60):
   """
   Coloring of the nodes with as few colors as possible, as a division of the nodes into
   independent sets (see the module description for which edges count).

   For each component, the number of colors is bracketed between a lower bound, the size of the
   largest clique (its nodes need different colors; found as in
   :func:`~graphoskop.maximal_cliques`), and the coloring found by DSATUR (Brelaz, 1979). If they
   differ, an exact DSATUR backtracking search tries each smaller number of colors :math:`k`, after
   setting aside the nodes with fewer than :math:`k` neighbors (repeatedly; they can always be
   colored afterwards). The coloring returned is then optimal, unless ``time_limit`` is reached:
   the best coloring found is returned, with a warning giving the bounds. Components are colored
   independently (smallest first, so that the time limit is spent on the hardest), with the same
   colors, so the number of colors is the largest over components.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param time_limit: Seconds after which the exact search stops (for the whole call).
   :return: List of arrays of input node ids, one per color (independent sets), largest first.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      colors = gsk.graph_coloring(net)
      len(colors)                       # number of colors used (the chromatic number if no warning)
   """
   deadline = time.time() + time_limit
   limit = sys.getrecursionlimit()
   sys.setrecursionlimit(max(limit, 10 * max(d.num_nodes for d in net.components) + 1000))
   classes, open_ = {}, []
   try:
      for c in range(len(net.components))[::-1] if component is None else [component]:   # smallest first, so that they finish
         col, ub, lb, proven = _color_component(_adjacency(net, c), deadline)
         if not proven:
            open_.append((c, lb, ub))
         ids = net.node_ids[c]
         col = np.asarray(col)
         for k in range(ub):
            classes.setdefault(k, []).append(ids[col == k])
   finally:
      sys.setrecursionlimit(limit)
   if open_:
      warnings.warn(f"time limit reached: coloring not proven optimal in {len(open_)} components (component, lower bound, colors used): {open_[:5]}")
   out = [np.concatenate(v) for v in classes.values()]
   return sorted(out, key = len, reverse = True)


def _mis(adj, deadline):
   """
   Maximum independent set by branch and reduce.

   Reductions: nodes without neighbors and simplicial nodes (neighbors all joined to each other,
   degree 1 included) are taken; a node whose closed neighborhood contains that of a neighbor is
   dominated and removed. Branching on a node of largest degree (out, then in), with the bound of a
   greedy clique cover (an independent set has at most one node per clique), and splitting into
   connected pieces.

   :param adj: Dictionary from node to set of neighbors (consumed).
   :param deadline: Time after which :class:`_Timeout` is raised.
   :return: A maximum independent set of the graph.
   """
   if time.time() > deadline:
      raise _Timeout
   taken = set()

   def remove(vs):
      for v in vs:
         for u in adj.pop(v, ()):
            if u in adj:
               adj[u].discard(v)

   changed = True
   while changed:
      changed = False
      for v in list(adj):
         if v not in adj:
            continue
         a = adj[v]
         if all(a - {u} <= adj[u] for u in a):   # simplicial: take it
            taken.add(v)
            remove([v] + list(a))
            changed = True
         else:
            for u in a:   # v dominated by a neighbor u: N[u] within N[v]
               if adj[u] - {v} <= a:
                  remove([v])
                  changed = True
                  break
   if not adj:
      return taken
   pieces, seen = [], set()   # connected pieces, solved independently
   for v in adj:
      if v not in seen:
         stack, piece = [v], set()
         seen.add(v)
         while stack:
            x = stack.pop()
            piece.add(x)
            for u in adj[x]:
               if u not in seen:
                  seen.add(u)
                  stack.append(u)
         pieces.append(piece)
   if len(pieces) > 1:
      for p in pieces:
         taken |= _mis({v: adj[v] & p for v in p}, deadline)
      return taken
   # bound: greedy clique cover
   cover, left = 0, set(adj)
   for v in sorted(adj, key = lambda v: len(adj[v])):
      if v in left:
         clique = {v}
         for u in sorted(adj[v] & left, key = lambda u: -len(adj[u])):
            if all(u in adj[w] for w in clique):
               clique.add(u)
         left -= clique
         cover += 1
   v = max(adj, key = lambda v: len(adj[v]))
   best = None
   for include in (False, True):
      if best is not None and len(best) >= cover:   # cannot do better here
         break
      sub = {u: set(a) for u, a in adj.items()}
      for u in [v] + (list(adj[v]) if include else []):
         for w in sub.pop(u, ()):
            if w in sub:
               sub[w].discard(u)
      got = _mis(sub, deadline) | ({v} if include else set())
      if best is None or len(got) > len(best):
         best = got
   return taken | best


def _greedy_mis(adj):
   """
   Independent set by repeatedly taking a node of smallest degree and removing its neighbors.
   """
   adj = {v: set(a) for v, a in adj.items()}
   out = set()
   while adj:
      v = min(adj, key = lambda v: len(adj[v]))
      out.add(v)
      for u in [v] + list(adj[v]):
         for w in adj.pop(u, ()):
            if w in adj:
               adj[w].discard(u)
   return out


@_higher_order()
def maximum_independent_set(net, component = None, time_limit = 60):
   """
   Largest independent set: the largest set of nodes no two of which are joined (see the module
   description for which edges count).

   Exact branch and reduce on each component: nodes that can always be taken (no neighbors, or
   neighbors all joined to each other) are taken, dominated nodes are removed, the remaining graph
   is split into connected pieces, and the search branches on a node of largest degree, pruned by a
   clique cover bound. Sparse real networks are often reduced to nothing without branching. If
   ``time_limit`` is reached, a greedy set (smallest degree first) is returned for the components
   not finished, with a warning that it is not proven maximum.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param time_limit: Seconds after which the search stops (for the whole call).
   :return: Array of input node ids.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      mis = gsk.maximum_independent_set(net)
      len(mis)                          # independence number if no warning
   """
   deadline = time.time() + time_limit
   limit = sys.getrecursionlimit()
   sys.setrecursionlimit(max(limit, 10 * max(d.num_nodes for d in net.components) + 1000))
   out, stopped = [], 0
   try:
      for c in range(len(net.components))[::-1] if component is None else [component]:   # smallest first, so that they finish
         adj = dict(enumerate(_adjacency(net, c)))
         try:
            got = _mis({v: set(a) for v, a in adj.items()}, deadline)
         except _Timeout:
            got = _greedy_mis(adj)
            stopped += 1
         out.append(net.node_ids[c][np.array(sorted(got), dtype = np.int64)])
   finally:
      sys.setrecursionlimit(limit)
   if stopped:
      warnings.warn(f"time limit reached: independent set not proven maximum in {stopped} components (greedy set returned there)")
   return np.concatenate(out) if out else np.array([])
