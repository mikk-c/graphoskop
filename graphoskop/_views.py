"""
Networks on which the Laplacian, random walk and node vector functions run, for hypergraphs and
simplicial complexes (see :doc:`higher_order`).
"""


def _walk_view(net, weight):
   """
   ``(network, weight)`` to analyze instead of ``(net, weight)``: the Zhou network of a hypergraph
   or the 1-skeleton of a simplicial complex; ``net`` itself otherwise (see
   :func:`graphoskop.higher_order._walk_view`).
   """
   if getattr(net, "higher_order", None) is None:
      return net, weight
   from .higher_order import _walk_view as view
   return view(net, weight)


#: How Laplacian solvers, stationary distributions and sparse Fourier bases of hypergraphs are
#: computed (see :func:`graphoskop.set_hypergraph_route`).
_ROUTE = {"route": "auto", "ratio": 1.}


def _incidence(net, weight):
   """
   Whether to compute on the incidence matrix of an undirected hypergraph instead of its walk
   network: always, never, or (``"auto"``) when the walk network would have more pairs of nodes
   than ``ratio`` times the memberships (see :func:`graphoskop.set_hypergraph_route`).
   """
   if getattr(net, "higher_order", None) != "hypergraph" or net.directed or _ROUTE["route"] == "walk_network":
      return False
   if _ROUTE["route"] == "incidence":
      return True
   pairs = memberships = 0
   for c, k in enumerate(net.bipartite_split):
      s, t = net._input_edges(c)[0]
      size = (s.maximum(t) - int(k)).bincount().double()
      pairs += float((size * (size - 1) / 2).sum())
      memberships += float(size.sum())
   return pairs > _ROUTE["ratio"] * memberships
