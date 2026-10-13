"""
Graph exploration: breadth first and depth first search from a node.

Both return the nodes reached from the source, in the order they are visited, with the node they
were reached from (their parent in the search tree) and their depth in that tree; in breadth first
search the depth is the number of edges of a shortest path from the source. Only the component of
the source is explored. Searches run on the CPU with :mod:`scipy.sparse.csgraph` (compiled code,
:math:`O(n + m)`), whatever the device of the network, as in :mod:`~graphoskop.shortest_paths`.
Neighbors are visited in order of internal index; parallel edges and self loops do not matter.

In directed networks ``direction`` chooses the edges followed: ``"out"`` (along their direction),
``"in"`` (against it) or ``"both"``. In multilayer networks the search runs on the supra-graph
(couplings are edges like the others), and an entity given as source stands for all its copies,
which are all searched from at once (as one source linked to each of them).
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, depth_first_order


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


def bfs(net, source, direction = "out", max_depth = None):
   """
   Breadth first search: the nodes reachable from ``source``, by increasing number of edges from
   it (see the module description).

   :param net: A :class:`~graphoskop.Network`.
   :param source: Starting node (input id, internal id ``(component, index)``, or entity in
      multilayer networks).
   :param direction: In directed networks, ``"out"``, ``"in"`` or ``"both"``; ignored otherwise.
   :param max_depth: Only nodes at most this many edges away, or ``None`` for all.
   :return: :class:`pandas.DataFrame` in visit order, with columns ``node`` (input id),
      ``depth`` (number of edges from the source, 0 for the source) and ``parent`` (input id of
      the node it was reached from, missing for the source).
   :raises ValueError: If ``direction`` is not valid.
   :raises KeyError: If there is no such node.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      tree = gsk.bfs(net, "alice")
      friends_of_friends = tree.node[tree.depth == 2]
      gsk.bfs(net, "alice", max_depth = 1)          # alice and her neighbors
   """
   out = _search(net, source, direction, False)
   return out if max_depth is None else out[out.depth <= max_depth].reset_index(drop = True)


def dfs(net, source, direction = "out"):
   """
   Depth first search: the nodes reachable from ``source``, each followed by all the nodes reached
   through it before backtracking (preorder; see the module description).

   :param net: A :class:`~graphoskop.Network`.
   :param source: Starting node (input id, internal id ``(component, index)``, or entity in
      multilayer networks).
   :param direction: In directed networks, ``"out"``, ``"in"`` or ``"both"``; ignored otherwise.
   :return: :class:`pandas.DataFrame` in visit order, with columns ``node`` (input id),
      ``depth`` (depth in the search tree, 0 for the source) and ``parent`` (input id of the node
      it was reached from, missing for the source).
   :raises ValueError: If ``direction`` is not valid.
   :raises KeyError: If there is no such node.

   Example:

   .. code-block:: python

      tree = gsk.dfs(net, "alice")
      tree.node.tolist()                            # visit order
   """
   return _search(net, source, direction, True)
