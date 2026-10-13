"""
Planarity: whether a graph can be drawn on the plane without edge crossings.

The test is the Left-Right planarity test (de Fraysseix and Rosenstiehl, 1982; in the formulation
of Brandes, 2009), linear in the size of the graph, written in plain Python without recursion, so
that it works on graphs of any depth. Edge directions, self loops and parallel edges do not change
planarity and are ignored. It runs on the CPU, whatever the device of the network.
"""

import numpy as np


def _lr_planar(n, edges):
   """
   Left-Right planarity test of a simple undirected graph.

   :param n: Number of nodes (indices ``0`` to ``n - 1``).
   :param edges: List of pairs of distinct nodes, each unordered pair at most once.
   :return: ``True`` if the graph is planar.
   """
   m = len(edges)
   if n > 2 and m > 3 * n - 6:
      return False
   adj = [[] for _ in range(n)]
   for e, (a, b) in enumerate(edges):
      adj[a].append((b, e))
      adj[b].append((a, e))
   height, parent = [-1] * n, [-1] * n
   src, dst = [-1] * m, [-1] * m
   low, low2, depth = [0] * m, [0] * m, [0] * m
   roots, pos = [], [0] * n
   for s in range(n):   # orientation: DFS tree edges down, back edges up
      if height[s] >= 0:
         continue
      height[s] = 0
      roots.append(s)
      st = [s]
      while st:
         v = st[-1]
         av = adj[v]
         if pos[v] < len(av):
            w, e = av[pos[v]]
            pos[v] += 1
            if src[e] >= 0:
               continue
            src[e], dst[e] = v, w
            hv = height[v]
            if height[w] < 0:
               low[e] = low2[e] = hv
               parent[w] = e
               height[w] = hv + 1
               st.append(w)
               continue
            low[e], low2[e] = height[w], hv
         else:
            st.pop()
            e = parent[v]
            if e < 0:
               continue
            v = src[e]
         le, l2 = low[e], low2[e]   # e is finished: nesting depth, lowpoints of the parent edge
         depth[e] = 2 * le + (l2 < height[v])
         p = parent[v]
         if p >= 0:
            lp = low[p]
            if le < lp:
               low2[p] = min(lp, l2)
               low[p] = le
            elif le > lp:
               if le < low2[p]:
                  low2[p] = le
            elif l2 < low2[p]:
               low2[p] = l2
   out = [[] for _ in range(n)]
   for e in sorted(range(m), key = depth.__getitem__):
      out[src[e]].append(e)
   ref, lowpt_edge, bottom = [None] * m, [None] * m, [None] * m
   S = []   # conflict pairs [left low, left high, right low, right high]

   def conflicting(h, b):
      return h is not None and low[h] > low[b]

   def lowest(P):
      if P[0] is None and P[1] is None:
         return low[P[2]]
      if P[2] is None and P[3] is None:
         return low[P[0]]
      return min(low[P[0]], low[P[2]])

   def add_constraints(ei, e):
      P = [None, None, None, None]
      while True:   # return edges of ei go right
         Q = S.pop()
         if Q[0] is not None or Q[1] is not None:
            Q[:] = Q[2:] + Q[:2]
         if Q[0] is not None or Q[1] is not None:
            return False
         if low[Q[2]] > low[e]:
            if P[2] is None and P[3] is None:
               P[3] = Q[3]
            else:
               ref[P[2]] = Q[3]
            P[2] = Q[2]
         else:
            ref[Q[2]] = lowpt_edge[e]
         if (S[-1] if S else None) is bottom[ei]:
            break
      while S and (conflicting(S[-1][1], ei) or conflicting(S[-1][3], ei)):   # conflicting edges of earlier siblings go left
         Q = S.pop()
         if conflicting(Q[3], ei):
            Q[:] = Q[2:] + Q[:2]
         if conflicting(Q[3], ei):
            return False
         if P[2] is not None:
            ref[P[2]] = Q[3]
         if Q[2] is not None:
            P[2] = Q[2]
         if P[0] is None and P[1] is None:
            P[1] = Q[1]
         else:
            ref[P[0]] = Q[1]
         P[0] = Q[0]
      if any(x is not None for x in P):
         S.append(P)
      return True

   def integrate(v, ei):
      e = parent[v]
      if low[ei] < height[v]:
         if ei == out[v][0]:
            lowpt_edge[e] = lowpt_edge[ei]
         elif not add_constraints(ei, e):
            return False
      return True

   def trim(u):   # drop back edges ending at u
      while S and lowest(S[-1]) == height[u]:
         S.pop()
      if S:
         P = S[-1]
         while P[1] is not None and dst[P[1]] == u:
            P[1] = ref[P[1]]
         if P[1] is None and P[0] is not None:
            ref[P[0]] = P[2]
            P[0] = None
         while P[3] is not None and dst[P[3]] == u:
            P[3] = ref[P[3]]
         if P[3] is None and P[2] is not None:
            ref[P[2]] = P[0]
            P[2] = None

   pos = [0] * n
   for s in roots:   # testing: constraints between return edges
      st = [s]
      while st:
         v = st[-1]
         if pos[v] < len(out[v]):
            ei = out[v][pos[v]]
            pos[v] += 1
            bottom[ei] = S[-1] if S else None
            if ei == parent[dst[ei]]:
               st.append(dst[ei])
               continue
            lowpt_edge[ei] = ei
            S.append([None, None, ei, ei])
            if not integrate(v, ei):
               return False
         else:
            st.pop()
            e = parent[v]
            if e >= 0:
               u = src[e]
               trim(u)
               if low[e] < height[u]:   # e has return edges: reference to the highest
                  hl, hr = S[-1][1], S[-1][3]
                  ref[e] = hl if hl is not None and (hr is None or low[hl] > low[hr]) else hr
               if not integrate(u, e):
                  return False
   return True


def _simple(a, b):
   """
   Undirected simple graph of an edge list: self loops and parallel edges dropped, nodes relabeled.

   :param a: First endpoint of each edge (int array).
   :param b: Second endpoint of each edge.
   :return: ``(number of nodes, list of pairs)``.
   """
   keep = a != b
   lo, hi = np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])
   nodes, idx = np.unique(np.concatenate([lo, hi]), return_inverse = True)
   pairs = np.unique(idx.reshape(2, -1).T, axis = 0)
   return len(nodes), pairs.tolist()


def _planar_edges(a, b):
   """
   Whether the graph of an edge list is planar (edge directions, self loops and parallel edges are
   ignored).

   :param a: First endpoint of each edge (array of integers).
   :param b: Second endpoint of each edge.
   :return: ``True`` if planar.
   """
   n, pairs = _simple(np.asarray(a), np.asarray(b))
   return _lr_planar(n, pairs)
