"""
Communities: quality of partitions of the nodes into groups, and maximal cliques (community
discovery algorithms will come here too).

A partition assigns every node to a community, given either as the name of a node attribute
(numeric or qualitative; its values are community labels) or as labels in a dictionary (or
:class:`pandas.Series`) indexed by input node id, or in a sequence ordered as the nodes of the
network (component by component, by internal index: ``np.concatenate(net.node_ids)``). Labels
can be of any type; only equality matters. In multilayer networks every copy ``(entity, layer)``
has its own label, so a community can span several layers. A node in several communities has a
list, tuple or set of labels (see :ref:`overlapping-communities`): the partition is then a cover,
for which :func:`modularity` is the overlapping modularity.
"""

import numpy as np
import pandas as pd
import torch

from .network import _explode, _memberships
from .network_statistics import _skeleton, _paths


def _labels(net, partition, comps):
   """
   Communities of the nodes of some components, as integer codes, one row per membership: a node
   in several communities (a list, tuple or set of labels) has several rows.

   :param net: Network.
   :param partition: Node attribute name, dictionary or Series by node id, or sequence in network
      wide node order.
   :param comps: Component indices.
   :return: Node (position among the nodes of ``comps``, in order) and community code of every
      membership, both int64 arrays sorted by node, and the number of nodes.
   :raises ValueError: If a node has no label, or a sequence has the wrong length.
   :raises KeyError: If there is no such attribute.
   """
   if isinstance(partition, str):
      parts = []
      for c in comps:
         v = net.get_node_attribute(partition, c)
         parts.append(v.cpu().numpy() if torch.is_tensor(v) else v)
      values = np.concatenate(parts)
   elif isinstance(partition, (dict, pd.Series)):
      ids = np.concatenate([net.node_ids[c] for c in comps])
      values = pd.Series(ids).map(partition).to_numpy()
   else:
      values = _memberships(partition)
      if len(values) != net.number_of_nodes():
         raise ValueError(f"the partition has {len(values)} labels, the network {net.number_of_nodes()} nodes")
      values = np.concatenate([values[net.node_offsets[c]:net.node_offsets[c + 1]] for c in comps])
   values = _memberships(values)
   cols, flat = _explode({"node": np.arange(len(values))}, values)
   codes = pd.factorize(pd.Series(flat, dtype = object))[0]
   if (codes < 0).any():
      raise ValueError(f"{len(np.unique(cols['node'][codes < 0]))} nodes have no community label")
   node = cols["node"].astype(np.int64)
   order = np.lexsort((codes, node))
   return node[order], codes[order].astype(np.int64), len(values)


def modularity(net, partition, weight = None, resolution = 1, component = None, overlap = "shen"):
   """
   Modularity of a partition of the nodes into communities: the weight of the edges within
   communities minus its expected value in a null model that keeps the degrees.

   The variant follows the kind of network:

   * undirected (Newman, 2006): :math:`Q = \\sum_c \\left[ \\frac{L_c}{m} - \\gamma \\left( \\frac{K_c}{2m} \\right)^2 \\right]`,
     with :math:`L_c` the weight of the edges within community :math:`c`, :math:`K_c` the sum of
     the degrees (strengths) of its nodes and :math:`m` the total edge weight;
   * directed (Leicht and Newman, 2008): :math:`Q = \\sum_c \\left[ \\frac{L_c}{m} - \\gamma \\frac{K^{out}_c K^{in}_c}{m^2} \\right]`;
   * bipartite (Barber, 2007): the null model only joins nodes of different types,
     :math:`Q = \\sum_c \\left[ \\frac{L_c}{m} - \\gamma \\frac{K^{(0)}_c K^{(1)}_c}{m^2} \\right]`,
     with :math:`K^{(0)}_c` and :math:`K^{(1)}_c` the degrees of its nodes of each type (directed
     bipartite networks use the directed variant);
   * multilayer (Mucha et al., 2010): each layer has its own null model (its own :math:`m_l` and
     degrees, with :math:`\\gamma` the same for all layers), and couplings and other interlayer
     edges add their weight when both ends are in the same community, without a null model term:
     :math:`Q = \\frac{1}{2\\mu} \\sum \\left[ (A_{ij} - \\gamma \\frac{k_i k_j}{2 m_l}) \\delta_{l_i l_j} + C_{ij} \\right] \\delta(g_i, g_j)`,
     with :math:`2\\mu` the total weight of intra-layer and interlayer edges (both directions);
     directed networks use the directed null model in each layer. With one layer and no
     interlayer edges this is the ordinary modularity.

   Communities can overlap: a node in several communities has a list, tuple or set of labels
   (see :doc:`network`). Overlapping modularity then weighs every pair of nodes by their
   belonging to each community they share, in the same variants (directed, bipartite,
   multilayer), following ``overlap``:

   * ``"shen"`` (Shen et al., 2009): a node in :math:`O_i` communities belongs to each with
     :math:`a_{ic} = 1 / O_i`, and :math:`Q = \\frac{1}{2m} \\sum_c \\sum_{ij} \\left[ A_{ij} -
     \\gamma \\frac{k_i k_j}{2m} \\right] a_{ic} a_{jc}`. Without overlaps this is the modularity
     above;
   * ``"nicosia"`` (Nicosia et al., 2009): belonging :math:`\\alpha_{ic} = 1 / O_i` (0 outside the
     community) enters through :math:`g(\\alpha) = 1 / (1 + e^{-(2 p \\alpha - p)})`, with
     :math:`p = 30` as in the paper, and :math:`Q = \\frac{1}{2m} \\sum_c \\sum_{ij} \\left[ g_{ic}
     g_{jc} A_{ij} - \\gamma G_c^2 \\frac{g_{ic} k_i g_{jc} k_j}{2m} \\right]`, with :math:`G_c` the
     mean of :math:`g_{jc}` over all nodes (of the layer). It is sharper than Shen's (a node in 3
     communities barely belongs to any) and, unlike it, it differs from the classic modularity
     even without overlaps (its null model shrinks by :math:`G_c^2`); it is used whenever asked.

   A self loop counts once in :math:`L_c` and twice in the degree of its node (in undirected
   networks), and parallel edges add up, as in NetworkX and igraph. ``resolution``
   (:math:`\\gamma`) above 1 favors smaller communities. Modularity is at most 1; it is 0 for a
   single community containing everything (with :math:`\\gamma = 1`). NaN if there are no edges.
   The computation is one pass over the edges, on the device of the network.

   :param net: A :class:`~graphoskop.Network`.
   :param partition: The communities: name of a node attribute, dictionary (or Series) from node
      id to label, or sequence of labels in network wide node order (see the module description).
   :param weight: Edge attribute used as weight, or ``None`` for unit weights.
   :param resolution: Resolution :math:`\\gamma`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param overlap: Overlapping modularity: ``"shen"`` (used only if some node is in several
      communities; otherwise it is the classic modularity) or ``"nicosia"``.
   :return: The modularity, a float.
   :raises ValueError: If a node has no label, a sequence has the wrong length, or ``overlap`` is
      not valid.
   :raises KeyError: If there is no such node or edge attribute.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")            # nodes.tsv has a "party" column
      gsk.modularity(net, "party")
      gsk.modularity(net, {"alice": 0, "bob": 0, "carol": 1, ...})
      gsk.modularity(net, labels, weight = "weight", resolution = 0.5)
      gsk.modularity(net, {"alice": [0, 1], "bob": 0, "carol": 1, ...})   # alice in both communities
      gsk.modularity(net, "community", overlap = "nicosia")
   """
   if overlap not in ("shen", "nicosia"):
      raise ValueError(f"unknown overlap: {overlap}")
   comps = list(range(len(net.components)) if component is None else [component])
   node, code, n = _labels(net, partition, comps)
   K = int(code.max()) + 1 if len(code) else 0
   O = np.bincount(node, minlength = n)
   if overlap == "shen":   # belonging 1 / O; 0 outside
      a, a0 = 1 / O[node], 0.
   else:
      g = lambda x: 1 / (1 + np.exp(-(60 * x - 30)))
      a, a0 = g(1 / O[node]), float(g(0.))
   A = np.bincount(node, a, minlength = n)   # belonging of each node summed over its communities
   ml = net.layer_labels is not None
   barber = net.bipartite_split is not None and not net.directed
   L = len(net.layer_labels) if ml else 1
   inside, total = 0., 0.
   m, nl = np.zeros(L), np.zeros(L)
   ka, kb, G = np.zeros((L, K)), np.zeros((L, K)), np.zeros((L, K))
   da, db = np.zeros((L, K)), np.zeros((L, K))   # degrees of the members (Nicosia: the others count with a0)
   off = np.cumsum([0] + [net.components[c].num_nodes for c in comps])
   for j, c in enumerate(comps):
      data = net.components[c]
      s, t = data.edge_index
      dev = s.device
      w = torch.ones(s.shape[0], dtype = torch.float64, device = dev) if weight is None else net.get_edge_attribute(weight, c).double()
      if not net.directed:   # other edges are stored in both directions, self loops once
         w = torch.where(s == t, 2 * w, w)
      mine = (node >= off[j]) & (node < off[j + 1])   # memberships of this component, by node
      mn, mc = torch.as_tensor(node[mine] - off[j], device = dev), torch.as_tensor(code[mine], device = dev)
      ma = torch.as_tensor(a[mine], device = dev)
      first = torch.as_tensor(np.searchsorted(node[mine] - off[j], np.arange(data.num_nodes)), device = dev)
      cnt = torch.as_tensor(O[off[j]:off[j + 1]], device = dev)
      # edge term: sum over the communities of s of a_sc a_tc (a0 for a non-member)
      e = torch.repeat_interleave(torch.arange(s.shape[0], device = dev), cnt[s])
      r = torch.arange(e.shape[0], device = dev) - torch.repeat_interleave(torch.cumsum(cnt[s], 0) - cnt[s], cnt[s])
      ms = first[s][e] + r   # memberships of the source of each edge
      keys = mn * K + mc   # sorted by node, then code
      want = t[e] * K + mc[ms]
      hit = torch.searchsorted(keys, want).clamp(max = max(keys.shape[0] - 1, 0))
      found = keys[hit] == want
      at = torch.where(found, ma[hit], a0)
      pair = torch.zeros(s.shape[0], dtype = torch.float64, device = dev).index_add_(0, e, ma[ms] * at)
      if a0:   # communities of t only, and of neither
         common = torch.zeros(s.shape[0], dtype = torch.float64, device = dev).index_add_(0, e, torch.where(found, ma[hit], 0.).double())
         shared = torch.zeros(s.shape[0], dtype = torch.float64, device = dev).index_add_(0, e, found.double())
         At = torch.as_tensor(A[off[j]:off[j + 1]], device = dev)
         pair += a0 * (At[t] - common) + a0 * a0 * (K - cnt[s] - cnt[t] + shared)
      inside += float((w * pair).sum())
      total += float(w.sum())
      lay = data.layer if ml else torch.zeros(data.num_nodes, dtype = torch.long, device = dev)
      intra = ~net.interlayer_mask(c)   # null model: intra-layer edges only
      if barber:   # only the direction from type 0 to type 1
         intra &= s < net.bipartite_split[c]
      si, ti, wi = s[intra], t[intra], w[intra]
      m += torch.bincount(lay[si], wi, minlength = L).cpu().numpy()
      kout = torch.bincount(si, wi, minlength = data.num_nodes)
      kin_ = torch.bincount(ti, wi, minlength = data.num_nodes)
      lm = lay[mn] * K + mc
      ka += torch.bincount(lm, ma * kout[mn], minlength = L * K).view(L, K).cpu().numpy()
      kb += torch.bincount(lm, ma * kin_[mn], minlength = L * K).view(L, K).cpu().numpy()
      if a0:   # non-members add a0 times their degrees, and count in the mean belonging
         da += torch.bincount(lm, kout[mn], minlength = L * K).view(L, K).cpu().numpy()
         db += torch.bincount(lm, kin_[mn], minlength = L * K).view(L, K).cpu().numpy()
         G += torch.bincount(lm, ma - a0, minlength = L * K).view(L, K).cpu().numpy()
         nl += torch.bincount(lay, minlength = L).cpu().numpy()
         tot_out = torch.bincount(lay, kout, minlength = L).cpu().numpy()
         tot_in = torch.bincount(lay, kin_, minlength = L).cpu().numpy()
         ka += a0 * tot_out[:, None]
         kb += a0 * tot_in[:, None]
   if total == 0:
      return float("nan")
   if a0:
      ka -= a0 * da
      kb -= a0 * db
      G = np.where(nl[:, None] > 0, G / np.maximum(nl[:, None], 1) + a0, 0)
   with np.errstate(invalid = "ignore", divide = "ignore"):
      null = np.where(m > 0, ((G ** 2 if a0 else 1) * ka * kb).sum(1) / m, 0).sum()
   if barber:   # one direction kept, so the null model counts each edge once
      null = 2 * null
   return (inside - resolution * null) / total


def _bron_kerbosch(adj, min_size):
   """
   Maximal cliques of a graph, by Bron-Kerbosch with pivoting (Tomita et al., 2006), iterative,
   with the vertices taken in order of increasing degree at the top level.

   :param adj: Dictionary from vertex to the set of its neighbors.
   :param min_size: Smallest clique returned.
   :return: Generator of cliques, as lists of vertices.
   """
   order = sorted(adj, key = lambda v: len(adj[v]))
   pos = {v: i for i, v in enumerate(order)}
   for v in order:
      later = {u for u in adj[v] if pos[u] > pos[v]}
      stack = [([v], later, adj[v] - later)]
      while stack:
         R, P, X = stack.pop()
         if not P:
            if not X and len(R) >= min_size:
               yield R
            continue
         if len(R) + len(P) < min_size:
            continue
         u = max(P | X, key = lambda w: len(adj[w] & P))   # pivot: fewest branches
         for w in list(P - adj[u]):
            stack.append((R + [w], P & adj[w], X & adj[w]))
            P = P - {w}
            X = X | {w}


def maximal_cliques(net, component = None, min_size = 3, batch_size = 2 ** 22):
   """
   Maximal cliques: sets of nodes all joined to each other, not contained in a larger such set.

   Only non-trivial cliques are returned, with at least ``min_size`` nodes (3 by default: single
   nodes and edges are left out). Edges are taken as undirected and unweighted (in directed
   networks, two nodes are joined if there is an edge in either direction), self loops and parallel
   edges are ignored. In multilayer networks only intra-layer edges count, so cliques are within a
   layer (couplings would make the copies of each entity a clique).

   Edges in no triangle cannot be in a clique of 3 or more nodes, so they are removed first, with
   the triangle enumeration of :func:`~graphoskop.global_clustering` (on the device of the
   network); maximal cliques are then listed by Bron-Kerbosch with pivoting (Tomita et al., 2006)
   on what remains, on the CPU. The number of maximal cliques can grow exponentially with the size
   of dense networks.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param min_size: Smallest clique returned (at least 3).
   :param batch_size: Number of 2-paths expanded at once in the triangle enumeration.
   :return: List of arrays of input node ids, one per clique, largest first.
   :raises ValueError: If ``min_size`` is below 3.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      cliques = gsk.maximal_cliques(net)
      len(cliques[0])                          # size of the largest clique
      big = gsk.maximal_cliques(net, min_size = 5)
   """
   if min_size < 3:
      raise ValueError("min_size must be at least 3 (smaller cliques are single nodes and edges)")
   out = []
   for c in range(len(net.components)) if component is None else [component]:
      data = net.components[c]
      n = data.num_nodes
      s, t = data.edge_index
      keep = s != t
      if net.layer_labels is not None:
         keep &= ~net.interlayer_mask(c)
      key = torch.unique(s[keep] * n + t[keep])
      if not key.shape[0]:
         continue
      src, dst, _, _ = _skeleton(key // n, key % n, n)
      tri = torch.zeros(src.shape[0], dtype = torch.bool, device = src.device)
      for ab, bc, ac in _paths(src, dst, n, batch_size):   # edges in at least one triangle
         tri[ab] = tri[bc] = tri[ac] = True
      a, b = src[tri].cpu().numpy(), dst[tri].cpu().numpy()
      adj = {}
      for x, y in zip(a.tolist(), b.tolist()):
         adj.setdefault(x, set()).add(y)
         adj.setdefault(y, set()).add(x)
      ids = net.node_ids[c]
      out += [ids[np.array(R)] for R in _bron_kerbosch(adj, min_size)]
   return sorted(out, key = len, reverse = True)
