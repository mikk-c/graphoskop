"""
Communities: discovery of communities, quality of partitions of the nodes into groups, and
maximal cliques.

:func:`block_model_communities` finds communities by fitting the stochastic blockmodels of the
generators (nested, degree-corrected, plain, planted partition, mixed membership) to a network;
:func:`infomap_communities` by compressing the description of a random walk on it (Infomap);
:func:`label_propagation_communities` by letting nodes adopt the labels of most of their neighbors;
:func:`leiden_communities` by maximizing modularity (or the constant Potts model) with the Leiden
algorithm; :func:`hierarchical_random_graph_communities` by the consensus of hierarchical random
graphs.
Infomap and label propagation can also find overlapping communities (``max_communities`` above
1), written as covers; :func:`bigclam_communities` finds overlapping communities by fitting the
community affiliation model of BigCLAM, :func:`demon_communities` by merging the communities of
the ego networks of all nodes, and :func:`oslom_communities` by keeping the statistically
significant clusters. Link communities group the links instead of the nodes,
so that nodes are in the communities of their links: :func:`hierarchical_link_communities` by
hierarchical clustering of the links, :func:`line_graph_communities` by partitioning the line
graph.
The quality of any partition is measured by :func:`modularity`, by :func:`codelength`, the map
equation of Infomap, and community by community by :func:`conductance`, :func:`internal_density` and
:func:`out_degree_fraction`. Two partitions are compared by :func:`normalized_mutual_information`,
:func:`adjusted_mutual_information` and :func:`adjusted_rand_index`.

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

from ._policy import _higher_order
from .network import _explode, _memberships
from .network_statistics import _skeleton, _paths


def _labels(net, partition, comps, uniques = False):
   """
   Communities of the nodes of some components, as integer codes, one row per membership: a node
   in several communities (a list, tuple or set of labels) has several rows.

   :param net: Network.
   :param partition: Node attribute name, dictionary or Series by node id, or sequence in network
      wide node order.
   :param comps: Component indices.
   :param uniques: Whether to also return the label of every code.
   :return: Node (position among the nodes of ``comps``, in order) and community code of every
      membership, both int64 arrays sorted by node, and the number of nodes (and the labels).
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
   codes, labels = pd.factorize(pd.Series(flat, dtype = object))
   if (codes < 0).any():
      raise ValueError(f"{len(np.unique(cols['node'][codes < 0]))} nodes have no community label")
   node = cols["node"].astype(np.int64)
   order = np.lexsort((codes, node))
   if uniques:
      return node[order], codes[order].astype(np.int64), len(values), np.asarray(labels)
   return node[order], codes[order].astype(np.int64), len(values)


def _write_cover(net, attribute, node, code):
   """
   Write a cover (memberships ``node``, ``code``, over network wide node positions) as a node
   attribute: communities numbered by decreasing size, a node in one community has its number, a
   node in several a tuple of numbers (qualitative attribute; numeric if no node is in several).

   :return: The number of communities and of nodes in several.
   """
   uniq, inv, size = np.unique(code, return_inverse = True, return_counts = True)
   rank = np.empty(len(uniq), dtype = np.int64)
   rank[np.lexsort((uniq, -size))] = np.arange(len(uniq))
   code = rank[inv]
   order = np.lexsort((code, node))
   node, code = node[order], code[order]
   N = int(net.node_offsets[-1])
   cuts = np.searchsorted(node, np.arange(N + 1))
   multi = int((np.diff(cuts) > 1).sum())
   offs = net.node_offsets
   if multi:
      cover = [int(code[cuts[i]]) if cuts[i + 1] - cuts[i] == 1 else tuple(int(x) for x in code[cuts[i]:cuts[i + 1]]) for i in range(N)]
      if attribute in net.node_attributes:   # a numeric attribute cannot hold tuples
         net._drop_node_attribute(attribute)
      net.set_node_attribute(attribute, [np.array(cover[offs[c]:offs[c + 1]] + [None], dtype = object)[:-1] for c in range(len(net.components))], None)
   else:   # a partition: numeric, as the other methods write it
      if attribute in net.qualitative_node_attributes:
         net._drop_node_attribute(attribute)
      lab = code[cuts[:-1]].astype(np.float64)
      net.set_node_attribute(attribute, [lab[offs[c]:offs[c + 1]] for c in range(len(net.components))], None)
   return int(len(uniq)), multi


def _drop_nested(node, code):
   """
   Remove the communities of a cover contained in another one (of two equal ones, the second).

   :return: The memberships left, ``(node, code)``.
   """
   size = np.bincount(code)
   m = pd.DataFrame({"node": node, "code": code})
   pairs = m.merge(m, on = "node")
   pairs = pairs[pairs.code_x != pairs.code_y].groupby(["code_x", "code_y"]).size().reset_index(name = "shared")
   a, b, k = pairs.code_x.to_numpy(), pairs.code_y.to_numpy(), pairs.shared.to_numpy()
   inside = (k == size[a]) & ((size[b] > size[a]) | (b < a))   # a within b (equal: drop the later)
   drop = np.isin(code, np.unique(a[inside]))
   return node[~drop], code[~drop]


@_higher_order()
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
      w = torch.ones(s.shape[0], dtype = torch.float64, device = dev) if weight is None else net._numeric_edge(weight, c).double()
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


@_higher_order()
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
      ids = net.node_ids[c]
      out += [ids[np.array(R)] for R in _cliques(net, c, min_size, batch_size)[0]]
   return sorted(out, key = len, reverse = True)


def _cliques(net, component, min_size, batch_size):
   """
   Maximal cliques of a component (see :func:`maximal_cliques`), and the edges in no triangle
   (the maximal cliques of two nodes).

   :param net: Network.
   :param component: Component index.
   :param min_size: Smallest clique returned (at least 3).
   :param batch_size: Number of 2-paths expanded at once in the triangle enumeration.
   :return: List of lists of node indices (one per clique), and the two endpoints (NumPy arrays of
      node indices, smaller first) of the edges in no triangle, each pair once.
   """
   data = net.components[component]
   n = data.num_nodes
   s, t = data.edge_index
   keep = s != t
   if net.layer_labels is not None:
      keep &= ~net.interlayer_mask(component)
   key = torch.unique(s[keep] * n + t[keep])
   if not key.shape[0]:
      return [], np.zeros(0, dtype = np.int64), np.zeros(0, dtype = np.int64)
   src, dst, _, _ = _skeleton(key // n, key % n, n)
   tri = torch.zeros(src.shape[0], dtype = torch.bool, device = src.device)
   for ab, bc, ac in _paths(src, dst, n, batch_size):   # edges in at least one triangle
      tri[ab] = tri[bc] = tri[ac] = True
   a, b = src[tri].cpu().numpy(), dst[tri].cpu().numpy()
   adj = {}
   for x, y in zip(a.tolist(), b.tolist()):
      adj.setdefault(x, set()).add(y)
      adj.setdefault(y, set()).add(x)
   u, v = src[~tri].cpu().numpy(), dst[~tri].cpu().numpy()
   return list(_bron_kerbosch(adj, min_size)), np.minimum(u, v), np.maximum(u, v)


def _mixed_membership(N, s, t, K, b, directed, max_iter, device):
   """
   Maximum likelihood memberships and block probabilities of the mixed membership blockmodel,
   with Poisson edge counts (the Bernoulli likelihood of sparse networks): the negative log
   likelihood :math:`\\sum_{(i, j)} p_{ij} - \\sum_{(i, j) \\in E} \\ln p_{ij}`, with
   :math:`p_{ij} = \\pi_i^T B \\pi_j`, computed in :math:`O((m + N) K^2)` from the sum of the
   memberships, minimized by L-BFGS over :math:`\\pi = \\mathrm{softmax}(Z)` and
   :math:`B = \\exp(W)`, from the partition ``b``.

   :return: ``(pi, B, log_likelihood)``, float64 tensors on ``device``.
   """
   s, t = torch.as_tensor(s, device = device), torch.as_tensor(t, device = device)
   b = torch.as_tensor(b, device = device)
   n = torch.bincount(b, minlength = K).double()
   e = torch.zeros(K, K, dtype = torch.float64, device = device).index_put_((b[s], b[t]), torch.ones(len(s), dtype = torch.float64, device = device), accumulate = True)
   pairs = torch.outer(n, n) - torch.diag(n)   # ordered pairs between blocks
   if not directed:
      e, pairs = e + e.T, pairs   # e counts both directions, pairs ordered: densities as for directed
   dens = ((e + 0.5) / (pairs + 1)).clamp(max = 1)
   Z = torch.log(torch.full((N, K), 0.1 / K, dtype = torch.float64, device = device).index_put_((torch.arange(N, device = device), b), torch.tensor(0.9, dtype = torch.float64, device = device))).requires_grad_()
   W = torch.log(dens).requires_grad_()

   def unpack():
      B = torch.exp(W if directed else (W + W.T) / 2)
      return torch.softmax(Z, 1), B

   def nll():
      pi, B = unpack()
      p = ((pi[s] @ B) * pi[t]).sum(1)
      tot = pi.sum(0)
      total = tot @ B @ tot - ((pi @ B) * pi).sum()   # ordered pairs of distinct nodes
      return (total if directed else total / 2) - torch.log(p.clamp(min = 1e-300)).sum()

   opt = torch.optim.LBFGS([Z, W], max_iter = max_iter, history_size = 20, tolerance_grad = 1e-9, tolerance_change = 1e-12, line_search_fn = "strong_wolfe")

   def closure():
      opt.zero_grad()
      loss = nll()
      loss.backward()
      return loss

   opt.step(closure)
   with torch.no_grad():
      pi, B = unpack()
      return pi, B, -float(nll())


@_higher_order()
def block_model_communities(net, model = "nested_block_model", groups = None, attribute = "community", threshold = None, max_blocks = 4096, sweeps = 10, max_iter = 500, seed = None):
   """
   Communities from the stochastic blockmodels of the generators (see
   :doc:`generators`): the partition, the hierarchy of partitions or the overlapping
   memberships under which the network is most likely, given the generator of the model as
   ``model`` (the function or its name).

   * :func:`~graphoskop.nested_block_model` (default), :func:`~graphoskop.degree_corrected_block_model`,
     :func:`~graphoskop.stochastic_block_model`, :func:`~graphoskop.planted_partition_graph`: the
     partition minimizing the description length of Peixoto's nonparametric microcanonical
     version of each model, as in :func:`~graphoskop.fit_nested_block_model` (which this
     generalizes; see it for the description length and the agglomerative heuristic). The
     description length counts the cost of the groups too, so the number of groups is chosen by
     the model, unless ``groups`` fixes it (for the nested model, the groups of the bottom level).
     The degree-corrected models expect groups of nodes with any degrees; the others, nodes of
     similar degrees in each group (with broad degrees they tend to group hubs together); the
     planted partition has one density inside all groups and one between them (higher inside for
     communities, lower for bipartite-like structures).
   * :func:`~graphoskop.mixed_membership_block_model`: overlapping communities. Every node has
     weights :math:`\\pi_i` over ``groups`` blocks (by default, the number of groups of the
     stochastic blockmodel fitted first, which tends to be larger than the number of blocks, as
     nodes mixing blocks in different proportions form groups of their own: give ``groups`` when
     it is known), and :math:`\\pi_i^T B \\pi_j` edges are expected between
     nodes :math:`i` and :math:`j`. The memberships and the block matrix :math:`B` maximize the
     likelihood of the edges (Poisson, the limit of the Bernoulli likelihood for sparse networks),
     computed in :math:`O((m + n) K^2)` without going through all pairs of nodes, by L-BFGS on the
     device of the network (up to ``max_iter`` iterations), starting from the partition of the
     stochastic blockmodel with ``groups`` groups. This is a local maximum: the likelihood is not
     concave.

   The groups are written as node attributes: for the partitions, the numeric node attribute
   ``attribute`` (groups numbered from 0), and for the nested model also ``attribute + "_1"``,
   ``attribute + "_2"``, ... for the levels above; for the mixed memberships, the qualitative node
   attribute ``attribute`` with the blocks of weight at least ``threshold`` of every node (a tuple
   of blocks, or the block for a node in one; see :ref:`overlapping-communities`), as in the
   generator. The partitions are fitted by compiled kernels on the CPU (see
   :doc:`error_correction`), on undirected networks (parallel edges count once, self loops are
   ignored); the mixed memberships also on directed ones.

   :param net: A :class:`~graphoskop.Network`, not multilayer, modified in place; undirected,
      except for the mixed membership model.
   :param model: Generator of the model, or its name.
   :param groups: Number of groups (of the bottom level for the nested model; of blocks for the
      mixed membership model), or ``None`` to let the description length choose it.
   :param attribute: Name of the node attribute written.
   :param threshold: Mixed membership model: smallest weight of a block for a node to be listed in
      it; ``None`` for ``1 / (groups + 1)``.
   :param max_blocks: Maximum number of groups of the bottom level.
   :param sweeps: Maximum number of greedy sweeps after each merge round.
   :param max_iter: Mixed membership model: maximum number of L-BFGS iterations.
   :param seed: Seed of the random numbers, or ``None``.
   :return: A dictionary with ``model`` (the generator name) and ``groups`` (the number of groups
      of every level, bottom first, for the partitions; the number of blocks for the mixed
      memberships); for the partitions, ``description_length``; for the mixed memberships,
      ``log_likelihood``, ``memberships`` (the :math:`n` by ``groups`` float64 tensor of the
      weights of every node, in the network wide node order, on the device of the network) and
      ``probs`` (the matrix :math:`B`, expected edges between nodes fully in each pair of blocks).
   :raises ValueError: If ``model`` is not one of these generators, the network is multilayer or,
      except for the mixed membership model, directed, or ``groups`` is not between 1 and the
      number of nodes (and ``max_blocks``).

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      fit = gsk.block_model_communities(net, seed = 0)             # nested: community, community_1, ...
      fit = gsk.block_model_communities(net, gsk.degree_corrected_block_model, seed = 0)
      fit = gsk.block_model_communities(net, "planted_partition_graph", groups = 4, seed = 0)
      gsk.modularity(net, "community")
      fit = gsk.block_model_communities(net, "mixed_membership_block_model", groups = 3, seed = 0)
      fit["memberships"]                                             # weights of every node
   """
   from .error_correction import _fit, _pairs, _write_levels, _MODELS
   name = getattr(model, "__name__", model)
   kinds = {k: v for k, v in _MODELS.items() if v[1] != "one"}
   if name not in kinds and name != "mixed_membership_block_model":
      raise ValueError(f"unknown model {model!r}: give one of the generators {', '.join(sorted(list(kinds) + ['mixed_membership_block_model']))}")
   if net.layer_labels is not None:
      raise ValueError("block_model_communities does not work on multilayer networks (see layer_network or aggregate_network)")
   mixed = name == "mixed_membership_block_model"
   N = int(net.node_offsets[-1])
   if groups is not None and not 1 <= groups <= min(N, max_blocks):
      raise ValueError(f"groups must be between 1 and {min(N, max_blocks)}")
   if mixed and net.directed:
      S, T = [], []
      for c in range(len(net.components)):
         ei = net._input_edges(c)[0]
         S.append(ei[0].cpu().numpy() + int(net.node_offsets[c]))
         T.append(ei[1].cpu().numpy() + int(net.node_offsets[c]))
      s, t = np.concatenate(S), np.concatenate(T)
      keep = s != t
      key = np.unique(s[keep] * N + t[keep])
      s, t = key // N, key % N
      sym = np.unique(np.minimum(s, t) * N + np.maximum(s, t))
      eu, ev = sym // N, sym % N   # for the starting partition
   else:
      N, s, t, _ = _pairs(net, "block_model_communities")
      key = np.unique(s * N + t)
      eu, ev = key // N, key % N
      s, t = eu, ev
   kind, shape = (2, "flat") if mixed else kinds[name]
   b0, par, Bs, S = _fit(N, eu.astype(np.int64), ev.astype(np.int64), max_blocks = max_blocks, sweeps = sweeps, seed = seed, kind = kind, flat = shape == "flat", fixed = groups)
   if shape == "flat":   # number the groups left from 0
      _, b0 = np.unique(b0, return_inverse = True)
      Bs = [int(b0.max()) + 1 if N else 0]
   if not mixed:
      _write_levels(net, b0, par, Bs, attribute)
      return {"model": name, "groups": [int(x) for x in Bs], "description_length": float(S)}
   K = Bs[0]
   dev = net.components[0].edge_index.device
   pi, B, ll = _mixed_membership(N, s, t, K, b0, net.directed, max_iter, dev)
   cut = 1 / (K + 1) if threshold is None else threshold
   P = pi.cpu().numpy()
   cover = [tuple(int(a) for a in np.flatnonzero(row >= cut)) or (int(row.argmax()),) for row in P]
   cover = [x[0] if len(x) == 1 else x for x in cover]
   offs = net.node_offsets
   net.set_node_attribute(attribute, [np.array(cover[offs[c]:offs[c + 1]] + [None], dtype = object)[:-1] for c in range(len(net.components))], None)
   return {"model": name, "groups": int(K), "log_likelihood": ll, "memberships": pi, "probs": B}


def _infomap_flows(net, weight, directed, teleportation, raw = False):
   """
   Flow of the random walk of Infomap on the whole network (all components).

   :return: ``(N, src, dst, link_flow, visit_rate)``, NumPy arrays over network wide node
      positions, links without self loops; with ``raw``, the link weights instead of the flows
      (links in both directions if not ``directed``).
   """
   N = int(net.node_offsets[-1])
   S, T, W = [], [], []
   for c in range(len(net.components)):
      ei, _ = net._input_edges(c)
      w = torch.ones(ei.shape[1], dtype = torch.float64) if weight is None else net._numeric_edge(weight, c)[:ei.shape[1]].double().cpu()
      S.append(ei[0].cpu().numpy() + int(net.node_offsets[c]))
      T.append(ei[1].cpu().numpy() + int(net.node_offsets[c]))
      W.append(w.numpy())
   s, t, w = np.concatenate(S), np.concatenate(T), np.concatenate(W)
   if (w < 0).any():
      raise ValueError("infomap_communities: weights must be non negative")
   keep = (s != t) & (w > 0)
   s, t, w = s[keep], t[keep], w[keep]
   if not net.directed:
      s, t, w = np.r_[s, t], np.r_[t, s], np.r_[w, w]
   key, inv = np.unique(s * N + t, return_inverse = True)   # parallel edges add
   s, t, w = key // N, key % N, np.bincount(inv, w)
   if raw:
      return N, s, t, w, None
   if not len(w):
      return N, s, t, w, np.zeros(N)
   if not directed:   # both directions: flow proportional to strength
      f = w / w.sum()
      return N, s, t, f, np.bincount(s, f, minlength = N)
   out = np.bincount(s, w, minlength = N)
   v = out / out.sum()   # teleportation to links: to the start of a random link
   p = np.full(N, 1 / N)
   step = (1 - teleportation) * w / out[s]
   for _ in range(10000):
      q = np.bincount(t, p[s] * step, minlength = N) + (teleportation * p[out > 0].sum() + p[out == 0].sum()) * v
      if np.abs(q - p).sum() < 1e-15:
         p = q
         break
      p = q
   f = p[s] * step
   f = f / f.sum()   # unrecorded teleportation: only the steps along links are coded
   return N, s, t, f, np.bincount(t, f, minlength = N)


@_higher_order()
def infomap_communities(net, attribute = "community", weight = None, directed = None, hierarchical = True, teleportation = 0.15, max_communities = 1, trials = 1, sweeps = 50, seed = None, physical = None):
   """
   Infomap (Rosvall and Bergstrom, 2008): the modules that compress the description of a random
   walk on the network the most, by the map equation. The walker's steps are described with one
   codeword per node, unique within its module, plus codewords to enter and exit modules: when
   the walker stays long in modules, short codewords reused across modules pay off. The map
   equation is the average length of the description of a step, in bits; for a two-level
   partition into modules :math:`i`,

   .. math::

      L = q \\, H(\\mathcal{Q}) + \\sum_i p_i^{\\circlearrowright} H(\\mathcal{P}^i),

   with :math:`q = \\sum_i q_i^{in}` the rate of entering modules, :math:`H(\\mathcal{Q})` the
   entropy of the module entry rates, :math:`p_i^{\\circlearrowright} = q_i^{out} + \\sum_{\\alpha \\in i}
   p_\\alpha` the rate of use of module :math:`i`'s codebook, and :math:`H(\\mathcal{P}^i)` the
   entropy of its exit rate and of the visit rates of its nodes.

   * Hierarchy (``hierarchical = True``; Rosvall and Bergstrom, 2011): modules can have
     submodules, each with its own codebook holding its exit and the entries of its submodules, at
     any depth, and depths can differ across branches. Every module is split recursively while a
     two-level partition of it, its index codebook also holding its exit, is shorter than its own
     codebook; then modules are grouped into super-modules while this shortens the top codebook.
   * Directions (``directed``, by default as the network): the walker follows edge directions,
     teleporting with probability ``teleportation`` (and always from nodes without out-edges) to
     the start of a random edge, so that the network needs no strong connectivity; as in the
     Infomap software, teleportation steps are not coded (unrecorded teleportation: Lambiotte and
     Rosvall, 2012), so only the flow along edges defines the modules. Undirected networks, or
     ``directed = False``, give a walker that steps along an edge with probability proportional to
     its weight, with visit rates proportional to the node strengths.
   * Layers (multilayer networks; De Domenico et al., 2015): the walker moves on the supra-graph,
     changing layer along couplings and interlayer edges in proportion to their weights, and every
     copy ``(entity, layer)`` has its own module, but the copies of an entity in the same module
     share one codeword (the physical node), so that entities active in several layers are not
     penalized for being in a module through several of them.
   * Memory (``physical``; Rosvall et al., 2014): the nodes are states of a walker with memory
     (:func:`~graphoskop.memory_network`, :func:`~graphoskop.higher_order_dependency_network`),
     and the node attribute ``physical`` gives the place of each (``"first_order_node"``). As for
     layers, the states of a place in the same module share its codeword, and a place belongs to
     every module of its states: overlapping modules of places, found without the overlapping
     search.

   The map equation is minimized as in the Infomap software: greedy moves of nodes between
   modules, in random order, then the modules found move as nodes and merge while it shortens the
   codelength (as in the Louvain method), then fine tuning (the nodes move again from the modules
   found), repeated while the codelength decreases. The moves run in compiled kernels (Numba) on
   the CPU; the result is a heuristic optimum, possibly different across seeds: ``trials`` runs
   keep the shortest. Self loops are ignored; parallel edges add their weights; nodes without edges
   are never visited and are modules of their own. Unlike the Infomap software, modules are not
   moved by their submodules (coarse tuning), so codelengths can be slightly longer on hard
   networks.

   Overlapping modules (``max_communities`` above 1; Esquivel and Rosvall, 2011): a node can be in
   several modules, with a codeword in each. The walker stays in its module when it steps to a node
   of the module; otherwise it exits it and enters one of the modules of the new node, chosen
   uniformly; the visit rates of the node are thus split among its modules, as the rates of being
   there in each. A node in two modules saves the exits and entries of the walker passing through
   it between them, at the cost of two codewords. From the two-level partition (``hierarchical``
   is not used), the change of codelength of every node joining one more module of a neighbor is
   computed (with the other visit rates fixed; exact while the neighbors are in one module), the
   best per node are sorted, and the best :math:`k` of them are applied together, :math:`k`
   chosen by the exact codelength (stationary visit rates of all the states, by power iteration)
   on a doubling grid refined around its best point (Esquivel and Rosvall fit parabolas instead),
   repeated while the codelength decreases. A node is in at most ``max_communities`` modules. Not
   for multilayer networks.

   Modules are numbered by decreasing flow (0 is the module most visited by the walker) and
   written as numeric node attributes: ``attribute`` for the top level (the modules of a two-level
   partition with ``hierarchical = False``) and ``attribute + "_1"``, ``attribute + "_2"``, ... for
   the levels below, each refining the one above (in branches that stop earlier, nodes keep their
   deepest module).

   :param net: A :class:`~graphoskop.Network`, modified in place.
   :param attribute: Name of the node attribute of the top level of modules.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param directed: Whether the walker follows edge directions; ``None`` for the directions of
      the network.
   :param hierarchical: Whether to search modules within modules (and super-modules).
   :param teleportation: Probability of teleporting at each step, in directed networks.
   :param max_communities: Maximum number of modules of a node; above 1 for overlapping modules,
      written as one attribute (a node in several modules has a tuple of numbers, by decreasing
      size).
   :param trials: Number of independent runs of the two-level search, keeping the shortest.
   :param sweeps: Maximum number of sweeps of node moves in each round.
   :param seed: Seed of the random numbers, or ``None``.
   :param physical: Name of the node attribute giving the physical node of every node (states
      of a network with memory), or ``None`` (in multilayer networks, the entities).
   :return: A dictionary with the ``codelength`` (bits per step of the walker) of the whole
      hierarchy, the same as :func:`codelength` of the list of all the levels
      (``[attribute, attribute + "_1", ...]``; a single level alone is scored as a two-level
      partition, with a longer codelength, since it lacks the codebooks of the other levels), the
      ``codelength_one_module`` (all nodes in one module: the entropy of the visit rates; the
      compression is the difference), and ``modules``, the number of modules of every level, top
      first; with overlaps, the codelength is that of the overlapping modules and the number of
      ``overlapping`` nodes is added.
   :raises ValueError: If weights are negative, ``directed`` is ``True`` for an undirected
      network, ``teleportation`` is not between 0 and 1, or ``max_communities`` is not a positive
      integer (or above 1 in a multilayer network or with ``physical``), or ``physical`` has
      missing values.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", directed = True)
      res = gsk.infomap_communities(net, seed = 0)               # community, community_1, ...
      res["modules"], res["codelength"]
      res = gsk.infomap_communities(net, hierarchical = False, directed = False, trials = 10)
      res = gsk.infomap_communities(net, max_communities = 2, attribute = "cover")   # overlapping
      mem = gsk.memory_network("paths.txt")                      # states "k|j": at k, from j
      gsk.infomap_communities(mem, weight = "weight", physical = "first_order_node")
   """
   from ._infomap import _Items, _two_level, _split, _coarsen, _tree_codelength, _leaves, _pl
   if directed and not net.directed:
      raise ValueError("an undirected network has no directions to follow")
   if not 0 <= teleportation < 1:
      raise ValueError("teleportation must be between 0 and 1")
   if int(max_communities) != max_communities or max_communities < 1 or (max_communities > 1 and (net.layer_labels is not None or physical is not None)):
      raise ValueError("max_communities must be a positive integer, 1 in multilayer networks or with physical nodes")
   directed = net.directed if directed is None else directed
   rng = np.random.default_rng(seed)
   N, s, t, f, p = _infomap_flows(net, weight, directed, teleportation)
   ml = net.layer_labels is not None or physical is not None
   ent = _physical(net, physical, N)
   nent = int(ent.max()) + 1 if N else 1
   items = _Items(p, s, t, f, np.zeros(N), np.zeros(N), np.arange(N), ent, p)
   assign, _ = _two_level(items, ml, nent, 0., trials, sweeps, rng) if N else (np.zeros(0, dtype = np.int64), 0.)
   if max_communities > 1 and N:
      from ._fuzzy import _fuzzy_search
      _, rs, rt, rw, _ = _infomap_flows(net, weight, directed, teleportation, raw = True)
      mem, nm, L, _ = _fuzzy_search(N, rs, rt, rw, directed, teleportation, assign, int(max_communities))
      K, multi = _write_cover(net, attribute, np.repeat(np.arange(N), nm), mem[np.arange(max_communities)[None, :] < nm[:, None]])
      one = float(-_pl(p).sum()) + 0.
      return {"codelength": float(L), "codelength_one_module": one, "modules": [K], "overlapping": multi}
   tree = [np.flatnonzero(assign == g) for g in range(int(assign.max()) + 1)] if N else []
   if hierarchical and len(tree) > 1:
      tree = [_split(items, nodes, ml, nent, trials, sweeps, rng) for nodes in tree]
      while len(tree) > 2:   # super-modules
         top = np.zeros(N, dtype = np.int64)
         for g, sub in enumerate(tree):
            for nodes, _ in _leaves(sub):
               top[nodes] = g
         sup = _coarsen(items, top, trials, sweeps, rng)
         if sup is None:
            break
         tree = [[tree[g] for g in np.flatnonzero(sup == h)] for h in range(int(sup.max()) + 1)]

   def flow(sub):
      return sum(p[nodes].sum() for nodes, _ in _leaves(sub))

   def ordered(sub):   # children by decreasing flow
      return sorted([ordered(c) for c in sub], key = flow, reverse = True) if isinstance(sub, list) else sub

   tree = ordered(tree)
   leaves = _leaves(tree)
   depth = max(len(path) for _, path in leaves) if leaves else 1
   counts = []
   for d in range(1, depth + 1):
      labels, lab = {}, np.zeros(N)
      for nodes, path in leaves:
         lab[nodes] = labels.setdefault(path[:d], len(labels))
      counts.append(len(labels))
      key = attribute if d == 1 else f"{attribute}_{d - 1}"
      if key in net.qualitative_node_attributes:   # a cover written before
         net._drop_node_attribute(key)
      net.set_node_attribute(key, [lab[net.node_offsets[c]:net.node_offsets[c + 1]] for c in range(len(net.components))], None)
   one = float(-_pl(np.bincount(ent, p, minlength = nent)).sum()) + 0. if N else 0.
   return {"codelength": _tree_codelength(items, tree) if N else 0., "codelength_one_module": one, "modules": counts}


def _physical(net, physical, N):
   """
   Physical node of every node, numbered from 0, network wide order: the values of the node
   attribute ``physical``, the entities of a multilayer network, or the nodes themselves.
   """
   if physical is not None:
      v = [net.get_node_attribute(physical, c) for c in range(len(net.components))]
      code = pd.factorize(np.concatenate([x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype = object) for x in v]) if v else [])[0]
      if (code < 0).any():
         raise ValueError(f"node attribute {physical!r} has missing values: every node needs a physical node")
      return code
   ent = np.concatenate([d.entity.cpu().numpy() for d in net.components]) if net.layer_labels is not None else np.arange(N)
   return np.unique(ent, return_inverse = True)[1]


def _hierarchy_tree(paths, nodes, d):
   """
   Tree of modules from the module of every node at every level (``paths``, one column per level,
   top first): a module whose nodes all have the same module at the next level is a leaf.
   """
   if d == paths.shape[1] or len(np.unique(paths[nodes, d])) == 1 and d > 0:
      return nodes
   return [_hierarchy_tree(paths, nodes[paths[nodes, d] == g], d + 1) for g in np.unique(paths[nodes, d])]


@_higher_order()
def codelength(net, partition, weight = None, directed = None, teleportation = 0.15, normalized = False, physical = None):
   """
   Codelength of a partition by the map equation (see :func:`infomap_communities`): the average
   number of bits per step needed to describe a random walk on the network with the modules of
   the partition. The shorter, the better the modules trap the walker, so it measures the quality
   of any partition (from any method), as :func:`modularity` does from a different point of view.

   The walk is that of :func:`infomap_communities`, with the same ``weight``, ``directed`` and
   ``teleportation``, and in multilayer networks the copies of an entity in the same module share
   a codeword (as do the nodes of the same ``physical`` node). A hierarchy of modules is given as a list of node attribute names, top level first,
   each level refining the one above (as written by :func:`infomap_communities`); its codelength is
   that of the hierarchical map equation. A single level is always scored as a two-level partition:
   to compare with the codelength returned by :func:`infomap_communities`, give all its levels.
   Overlapping communities (a node with several labels) are scored by the overlapping map equation
   of Esquivel and Rosvall (2011), as found by :func:`infomap_communities` with
   ``max_communities`` above 1 (not in multilayer networks, nor as a hierarchy).

   With ``normalized``, the codelength is divided by that of a single module (no communities: the
   entropy of the visit rates of the nodes, which grows with the number of nodes), so that
   networks of different sizes can be compared: below 1 when the partition compresses the walk,
   above 1 when it makes its description longer.

   :param net: A :class:`~graphoskop.Network`.
   :param partition: Node attribute name, dictionary or :class:`pandas.Series` by node id, or
      sequence in network wide node order, as for :func:`modularity` (one community per node);
      or a list of node attribute names, top level first, for a hierarchy.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param directed: Whether the walker follows edge directions; ``None`` for the directions of
      the network.
   :param teleportation: Probability of teleporting at each step, in directed networks.
   :param normalized: Whether to divide by the codelength of a single module.
   :param physical: Name of the node attribute giving the physical node of every node, whose
      nodes in the same module share a codeword (see :func:`infomap_communities`), or ``None``.
   :return: The codelength in bits (or its ratio to that of a single module); NaN normalized for a
      network without edges.
   :raises ValueError: If a node has no community, a hierarchy has overlapping communities, or as
      :func:`infomap_communities`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      gsk.codelength(net, "party")                                   # bits per step
      gsk.codelength(net, "party", normalized = True)                # comparable across networks
      gsk.infomap_communities(net, attribute = "module")
      gsk.codelength(net, ["module", "module_1"])                    # hierarchical map equation
   """
   from ._infomap import _Items, _tree_codelength, _pl
   if directed and not net.directed:
      raise ValueError("an undirected network has no directions to follow")
   if not 0 <= teleportation < 1:
      raise ValueError("teleportation must be between 0 and 1")
   directed = net.directed if directed is None else directed
   comps = list(range(len(net.components)))
   names = net.node_attributes + net.qualitative_node_attributes
   levels = partition if isinstance(partition, list) and partition and all(isinstance(x, str) and x in names for x in partition) else [partition]
   paths = []
   for level in levels:
      node, code, n = _labels(net, level, comps)
      if len(node) != n:
         if len(levels) > 1:
            raise ValueError("a hierarchy of overlapping communities is not supported")
         if physical is not None:
            raise ValueError("overlapping communities with physical nodes are not supported")
         return _overlapping_codelength(net, node, code, n, weight, directed, teleportation, normalized)
      paths.append(code)
   N, s, t, f, p = _infomap_flows(net, weight, directed, teleportation)
   ent = _physical(net, physical, N)
   items = _Items(p, s, t, f, np.zeros(N), np.zeros(N), np.arange(N), ent, p)
   L = _tree_codelength(items, _hierarchy_tree(np.stack(paths, 1), np.arange(N), 0)) + 0.
   if not normalized:
      return L
   one = float(-_pl(np.bincount(ent, p)).sum())
   return L / one if one > 0 else float("nan")


def _overlapping_codelength(net, node, code, n, weight, directed, teleportation, normalized):
   """
   Overlapping map equation (Esquivel and Rosvall, 2011) of a cover, memberships ``node``,
   ``code`` (see :func:`codelength`).
   """
   from ._fuzzy import _state_flows, _aggregates, _codelength
   from ._infomap import _pl
   if net.layer_labels is not None:
      raise ValueError("overlapping communities of multilayer networks are not supported")
   N, s, t, w, _ = _infomap_flows(net, weight, directed, teleportation, raw = True)
   cnt = np.bincount(node, minlength = N)
   mem = np.full((N, int(cnt.max())), -1, dtype = np.int64)
   mem[node, np.arange(len(node)) - np.repeat(np.cumsum(cnt) - cnt, cnt)] = code
   out = np.bincount(s, w, minlength = N)
   q = w / np.where(out[s] > 0, out[s], 1)
   optr = np.r_[0, np.cumsum(np.bincount(s, minlength = N))]
   v = out / out.sum() if directed and out.sum() > 0 else np.zeros(N)
   tau = teleportation if directed else 0.
   p = np.where(mem >= 0, 1 / N / cnt[:, None], 0.)
   _state_flows(N, optr, t, q, tau, v, out == 0, mem, cnt, p, 1e-15, 20000)
   L = float(_codelength(*_aggregates(N, optr, t, q, tau, out == 0, mem, cnt, p)[1:5])) + 0.
   if not normalized:
      return L
   one = float(-_pl(_infomap_flows(net, weight, directed, teleportation)[4]).sum())
   return L / one if one > 0 else float("nan")


@_higher_order()
def label_propagation_communities(net, attribute = "community", weight = None, strategy = "semisynchronous", max_communities = 1, max_iter = 100, seed = None):
   """
   Label propagation (Raghavan, Albert and Kumara, 2007): every node starts with its own label and
   repeatedly takes the label carried by most of its neighbors (by total edge weight), until every
   node has a label of largest weight among its neighbors; the groups of nodes sharing a label are
   the communities. A node keeps its label when it is one of the most frequent (ties among the
   others are broken at random), so that the process stops.

   How the labels are updated matters. If all nodes change at once (synchronous updates), the two
   sides of a bipartite network (or any two sets of nodes linked mostly between them) can swap their
   labels at every step, forever. Two strategies avoid it:

   * ``"semisynchronous"`` (Cordasco and Gargano, 2010): the nodes are colored (greedily, so that
     neighbors have different colors), and the nodes of one color update at once, color after color:
     neighbors never change together, the process always converges, and every step runs on the
     device of the network, for all the nodes of a color in parallel;
   * ``"asynchronous"`` (the original): the nodes update one at a time, in a new random order at
     every sweep, each seeing the labels already changed in the sweep; it runs in a compiled kernel
     (Numba) on the CPU.

   Directions are ignored, self loops do not vote and parallel edges add their weights. Nodes
   without edges are communities of their own. The result depends on the random order and ties:
   different seeds can give different communities.

   Overlapping communities (``max_communities`` above 1) follow COPRA (Gregory, 2010): every node holds up
   to ``max_communities`` labels, each with a belonging coefficient, the coefficients summing to 1 (at the
   start, its own label with coefficient 1). A node takes the labels of its neighbors with the
   mean of their coefficients, weighted by the edge weights, drops the labels below ``1 /
   max_communities`` (keeping the largest, ties at random, if all are below) and normalizes the
   coefficients of the others; the process stops when a sweep changes no set of labels or, as in
   COPRA, when the labels in use stay the same and none of them is held by fewer nodes than before
   (a few nodes can keep switching labels). Then, as in COPRA, a community contained in another one
   is removed. A node joins several communities only if they all carry at least ``1 / max_communities`` of
   its neighborhood: ``max_communities = 2`` needs an almost exact tie, so 3 or more is more useful. Both strategies apply (Gregory updates all nodes at once, which can oscillate as above);
   both run in a compiled kernel on the CPU. ``max_communities = 1`` is the classic label propagation.

   Communities are numbered by decreasing size and written as the numeric node attribute
   ``attribute``; with ``max_communities`` above 1, a node in several communities has a tuple of numbers.

   :param net: A :class:`~graphoskop.Network`, modified in place.
   :param attribute: Name of the node attribute written.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param strategy: ``"semisynchronous"`` or ``"asynchronous"``.
   :param max_communities: Maximum number of communities of a node (COPRA's :math:`v`).
   :param max_iter: Maximum number of sweeps over all the nodes; a warning is raised if they end
      before convergence.
   :param seed: Seed of the random numbers, or ``None``.
   :return: A dictionary with the number of ``communities``, the number of ``iterations`` (sweeps
      over all the nodes) and whether the labels ``converged``; with ``max_communities`` above 1, also the
      number of ``overlapping`` nodes (in several communities).
   :raises ValueError: If ``strategy`` or ``max_communities`` are not valid, or weights are negative.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", device = "cuda")
      gsk.label_propagation_communities(net, seed = 0)              # on the GPU, color by color
      gsk.label_propagation_communities(net, strategy = "asynchronous", attribute = "lpa")
      gsk.label_propagation_communities(net, max_communities = 3, attribute = "cover")   # up to 3 per node
   """
   import warnings
   from ._propagation import _async_sweeps, _greedy_coloring, _copra
   if strategy not in ("semisynchronous", "asynchronous"):
      raise ValueError(f"unknown strategy: {strategy}")
   if int(max_communities) != max_communities or max_communities < 1:
      raise ValueError("max_communities must be a positive integer")
   N = int(net.node_offsets[-1])
   dev = net.components[0].edge_index.device
   S, T, W = [], [], []
   for c in range(len(net.components)):
      ei = net.components[c].edge_index
      S.append(ei[0] + int(net.node_offsets[c]))
      T.append(ei[1] + int(net.node_offsets[c]))
      W.append(torch.ones(ei.shape[1], dtype = torch.float64, device = dev) if weight is None else net._numeric_edge(weight, c).double())
   s, t, w = torch.cat(S), torch.cat(T), torch.cat(W)
   if bool((w < 0).any()):
      raise ValueError("label_propagation_communities: weights must be non negative")
   keep = s != t
   s, t, w = s[keep], t[keep], w[keep]
   if net.directed:   # directions ignored: both ways
      s, t, w = torch.cat([s, t]), torch.cat([t, s]), torch.cat([w, w])
   key, inv = torch.unique(t * N + s, return_inverse = True)   # by receiving node, parallel edges merged
   t, s = key // N, key % N
   w = torch.zeros(len(key), dtype = torch.float64, device = dev).index_add_(0, inv, w)
   ptr = torch.cat([torch.zeros(1, dtype = torch.long, device = dev), torch.cumsum(torch.bincount(t, minlength = N), 0)])
   rng = np.random.default_rng(seed)
   if max_communities > 1:
      P, nb, ww = ptr.cpu().numpy(), s.cpu().numpy(), w.cpu().numpy()
      if strategy == "asynchronous":
         cptr, cls = np.arange(N + 1), np.arange(N)
      else:
         color = _greedy_coloring(N, P, nb)
         cls = np.argsort(color, kind = "stable")
         cptr = np.r_[0, np.cumsum(np.bincount(color))] if N else np.zeros(1, dtype = np.int64)
      lab, _, cnt, it, ok = _copra(N, P, nb, ww, int(max_communities), cptr, cls, strategy == "asynchronous", max_iter, int(rng.integers(2 ** 31)))
      if not ok:
         warnings.warn(f"label propagation did not converge in {max_iter} iterations")
      node = np.repeat(np.arange(N), cnt)
      code = lab[np.arange(max_communities)[None, :] < cnt[:, None]]
      K, multi = _write_cover(net, attribute, *_drop_nested(node, code))
      return {"communities": K, "iterations": int(it), "converged": bool(ok), "overlapping": multi}
   if strategy == "asynchronous":
      labels, it, ok = _async_sweeps(N, ptr.cpu().numpy(), s.cpu().numpy(), w.cpu().numpy(), np.arange(N), max_iter, int(rng.integers(2 ** 31)))
   else:
      color = torch.as_tensor(_greedy_coloring(N, ptr.cpu().numpy(), s.cpu().numpy()), device = dev)
      gen = torch.Generator(device = dev)
      gen.manual_seed(int(rng.integers(2 ** 62)))
      lab = torch.arange(N, device = dev)
      classes = [torch.nonzero(color[t] == k).flatten() for k in range(int(color.max()) + 1 if N else 0)]
      ok, it = False, 0
      while it < max_iter and not ok:
         it += 1
         ok = True
         for e in classes:   # edges into the nodes of one color
            if not len(e):
               continue
            v, l = t[e], lab[s[e]]
            k, kinv = torch.unique(v * N + l, return_inverse = True)
            sc = torch.zeros(len(k), dtype = torch.float64, device = dev).index_add_(0, kinv, w[e])
            node, cand = k // N, k % N
            top = torch.zeros(N, dtype = torch.float64, device = dev).scatter_reduce_(0, node, sc, "amax")
            best = sc >= top[node] * (1 - 1e-12)
            node, cand = node[best], cand[best]
            stay = torch.zeros(N, dtype = torch.bool, device = dev)
            stay[node[cand == lab[node]]] = True
            r = torch.rand(len(node), device = dev, generator = gen)
            pick = torch.full((N,), -1., device = dev).scatter_reduce_(0, node, r, "amax")
            chosen = (r == pick[node]) & ~stay[node]
            if bool(chosen.any()):
               ok = False
               lab[node[chosen]] = cand[chosen]
      labels = lab.cpu().numpy()
   if not ok:
      warnings.warn(f"label propagation did not converge in {max_iter} iterations")
   K, _ = _write_cover(net, attribute, np.arange(N), np.asarray(labels, dtype = np.int64))
   return {"communities": K, "iterations": int(it), "converged": bool(ok)}


@_higher_order()
def leiden_communities(net, attribute = "community", weight = None, quality = "modularity", resolution = 1, randomness = 0.01, initial = None, max_iter = 100, seed = None):
   """
   Leiden algorithm (Traag, Waltman and van Eck, 2019): communities of high quality, every one of
   them guaranteed to be connected. Like the Louvain method, nodes move greedily to the community
   that improves the quality the most and the communities found become the nodes of a smaller
   network, at every level; but Louvain can leave communities internally disconnected (a node that
   held them together moves away), which Leiden avoids with a refinement step. Every level has
   three phases:

   * fast local moving: nodes are taken from a queue (all of them at first, in random order) and
     moved to the best community; when a node moves, its neighbors outside its new community join
     the queue again, so that only the nodes affected by changes are visited;
   * refinement: every community is split into the singletons of its nodes, which then merge, in
     random order, only with clusters of the same community, only if both are well connected to
     the rest of the community, and choosing at random among the merges that do not decrease the
     quality, with probability proportional to :math:`\\exp(\\Delta H / \\theta)` (``randomness``
     :math:`\\theta`, with :math:`\\Delta H` in units of edge weight as in the paper; the smaller,
     the greedier), which explores the space of partitions better;
   * aggregation: the refined clusters become the nodes of the next level, starting in the
     communities found by the local moving (a community can thus be split at the next level).

   The levels end when every community is a single node; the whole algorithm then starts again from
   the partition found, until it does not improve (at most ``max_iter`` times).

   ``quality`` is the function maximized:

   * ``"modularity"``: the modularity of :func:`modularity`, in the variant of the network (directed,
     bipartite and multilayer), with resolution :math:`\\gamma`;
   * ``"cpm"``: the constant Potts model (Traag, Van Dooren and Nesterov, 2011),
     :math:`H = \\sum_c \\left[ E_c - \\gamma \\binom{n_c}{2} \\right]`, with :math:`E_c` the weight
     of the edges within community :math:`c` and :math:`n_c` its number of nodes (ordered pairs
     :math:`n_c (n_c - 1)` in directed networks; pairs of nodes of different types in bipartite
     networks; pairs in the same layer in multilayer networks): communities have internal density
     at least :math:`\\gamma` and density between them is below :math:`\\gamma`. Unlike modularity, it
     has no resolution limit: :math:`\\gamma` sets the density of the communities whatever the size
     of the network.

   Self loops count inside their node's community, parallel edges add their weights. Nodes without
   edges are communities of their own. The moves run in compiled kernels (Numba) on the CPU; the
   result is a heuristic optimum that can differ across seeds.

   Communities are numbered by decreasing size and written as the numeric node attribute
   ``attribute``.

   :param net: A :class:`~graphoskop.Network`, modified in place.
   :param attribute: Name of the node attribute written.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param quality: ``"modularity"`` or ``"cpm"``.
   :param resolution: Resolution :math:`\\gamma`: above 1 (modularity), or larger (CPM), gives
      smaller communities.
   :param randomness: Randomness :math:`\\theta` of the refinement, positive (0.01 as in the paper).
   :param initial: Partition to start from (node attribute name, dictionary or Series by node id,
      or sequence in network wide node order, as for :func:`modularity`), or ``None`` for
      singletons.
   :param max_iter: Maximum number of runs of the algorithm, each starting from the result of the
      previous one.
   :param seed: Seed of the random numbers, or ``None``.
   :return: A dictionary with the ``quality`` of the communities found (modularity, or :math:`H`),
      the number of ``communities`` and the number of ``iterations`` (runs of the algorithm).
   :raises ValueError: If ``quality`` is not valid, weights are negative, ``randomness`` is not
      positive, or a node has no community or several in ``initial``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.leiden_communities(net, seed = 0)                          # maximum modularity
      gsk.leiden_communities(net, resolution = 2, attribute = "small")
      gsk.leiden_communities(net, quality = "cpm", resolution = 0.05)  # density at least 0.05
   """
   from ._leiden import _local_moving, _refine, _aggregate
   if quality not in ("modularity", "cpm"):
      raise ValueError(f"unknown quality: {quality}")
   if randomness <= 0:
      raise ValueError("randomness must be positive")
   comps = list(range(len(net.components)))
   N = int(net.node_offsets[-1])
   ml = net.layer_labels is not None
   barber = net.bipartite_split is not None and not net.directed
   nl = len(net.layer_labels) if ml else 1
   S, T, W, lay, kout, kin, typ = [], [], [], [], [], [], []
   for c in comps:
      data = net.components[c]
      s, t = data.edge_index
      w = torch.ones(s.shape[0], dtype = torch.float64, device = s.device) if weight is None else net._numeric_edge(weight, c).double()
      if not net.directed:   # stored self loops count twice, as in modularity
         w = torch.where(s == t, 2 * w, w)
      intra = ~net.interlayer_mask(c)
      if barber:
         intra &= s < net.bipartite_split[c]
      kout.append(torch.bincount(s[intra], w[intra], minlength = data.num_nodes).cpu().numpy())
      kin.append(torch.bincount(t[intra], w[intra], minlength = data.num_nodes).cpu().numpy())
      lay.append(data.layer.cpu().numpy() if ml else np.zeros(data.num_nodes, dtype = np.int64))
      typ.append(np.arange(data.num_nodes) < net.bipartite_split[c] if barber else np.ones(data.num_nodes, dtype = bool))
      S.append(s.cpu().numpy() + int(net.node_offsets[c]))
      T.append(t.cpu().numpy() + int(net.node_offsets[c]))
      W.append(w.cpu().numpy())
   s, t, w = np.concatenate(S), np.concatenate(T), np.concatenate(W)
   lay, typ = np.concatenate(lay).astype(np.int64), np.concatenate(typ)
   if (w < 0).any():
      raise ValueError("leiden_communities: weights must be non negative")
   a, b = np.zeros((N, nl)), np.zeros((N, nl))
   if quality == "modularity":
      a[np.arange(N), lay], b[np.arange(N), lay] = np.concatenate(kout), np.concatenate(kin)
      m = np.bincount(lay, a.sum(1), minlength = nl)
      inv_m = np.where(m > 0, (2 if barber else 1) / np.where(m > 0, m, 1), 0.)
      div = w.sum()
   else:   # node counts, between types if bipartite
      a[np.arange(N), lay], b[np.arange(N), lay] = typ, ~typ if barber else typ
      inv_m = np.full(nl, 2. if barber else 1.)
      div = 1. if net.directed else 2.
   inside_w = w[s == t]   # self loops, inside whatever the partition
   keep = s != t   # symmetric: both directions summed, without self loops
   key, inv = np.unique(np.r_[s[keep] * N + t[keep], t[keep] * N + s[keep]], return_inverse = True)
   ws, src, nbr = np.bincount(inv, np.r_[w[keep], w[keep]]), key // N, key % N
   ptr = np.r_[0, np.cumsum(np.bincount(src, minlength = N))]
   scale = 0.5   # gains in units of edge weight (each edge once), as in the paper

   def score(comm):
      inside = inside_w.sum() + ws[comm[src] == comm[nbr]].sum() / 2
      A = np.stack([np.bincount(comm, a[:, l], minlength = N) for l in range(nl)], 1)
      B = np.stack([np.bincount(comm, b[:, l], minlength = N) for l in range(nl)], 1)
      null = (A * B * inv_m).sum() - (quality == "cpm") * (a * b * inv_m).sum()
      return (inside - resolution * null) / div if div > 0 else float("nan")

   rng = np.random.default_rng(seed)
   if initial is None:
      comm = np.arange(N)
   else:
      node, comm, n = _labels(net, initial, comps)
      if len(node) != n:
         raise ValueError("leiden_communities needs one initial community per node")
   best, it = score(comm), 0
   while it < max_iter:
      it += 1
      n, P, nb, wg, ag, bg, sg, mem, cg = N, ptr, nbr, ws, a, b, src, np.arange(N), comm.copy()
      while True:
         cg, _ = _local_moving(n, P, nb, wg, ag, bg, inv_m, resolution, cg, int(rng.integers(2 ** 31)))
         k = len(np.unique(cg))
         if k == n:
            break
         ref = _refine(n, P, nb, wg, ag, bg, inv_m, resolution, cg, randomness, scale, int(rng.integers(2 ** 31)))
         ref = np.unique(ref, return_inverse = True)[1]
         if ref.max() + 1 == n:   # nothing merged: aggregate the communities
            ref = np.unique(cg, return_inverse = True)[1]
         up = np.zeros(ref.max() + 1, dtype = np.int64)
         up[ref] = np.unique(cg, return_inverse = True)[1]
         mem = ref[mem]
         n, P, nb, wg, ag, bg, sg = _aggregate(ref, sg, nb, wg, ag, bg)
         cg = up
      new = cg[mem]
      q = score(new)
      if not q > best + 1e-12 * abs(best):
         break
      comm, best = new, q
   K, _ = _write_cover(net, attribute, np.arange(N), comm)
   return {"quality": float(best), "communities": K, "iterations": int(it)}


def _community_stats(net, partition, weight, component):
   """
   Edge weights of the communities of a partition (or cover), directions ignored and self loops
   dropped.

   :return: A dictionary with ``node`` and ``code`` of every membership, the ``degree`` of every
      node (strength), the weight ``inside`` from every membership to its own community, and for
      every community its ``labels``, ``size``, number ``size1`` of nodes of type 1 (bipartite),
      ``volume`` (sum of degrees) and ``internal`` (weight inside, each edge counted twice); and the
      ``total`` volume of the network.
   """
   comps = list(range(len(net.components)) if component is None else [component])
   node, code, n, labels = _labels(net, partition, comps, True)
   K = len(labels)
   deg, inside = np.zeros(n), np.zeros(len(node))
   typ = np.zeros(n)
   off = np.cumsum([0] + [net.components[c].num_nodes for c in comps])
   for j, c in enumerate(comps):
      data = net.components[c]
      s, t = data.edge_index
      dev = s.device
      w = torch.ones(s.shape[0], dtype = torch.float64, device = dev) if weight is None else net._numeric_edge(weight, c).double()
      keep = s != t
      s, t, w = s[keep], t[keep], w[keep]
      if net.directed:   # directions ignored
         s, t, w = torch.cat([s, t]), torch.cat([t, s]), torch.cat([w, w])
      deg[off[j]:off[j + 1]] = torch.bincount(s, w, minlength = data.num_nodes).cpu().numpy()
      if net.bipartite_split is not None:
         typ[off[j]:off[j + 1]] = np.arange(data.num_nodes) >= net.bipartite_split[c]
      rows = np.flatnonzero((node >= off[j]) & (node < off[j + 1]))
      mn, mc = torch.as_tensor(node[rows] - off[j], device = dev), torch.as_tensor(code[rows], device = dev)
      cnt = torch.bincount(mn, minlength = data.num_nodes)
      first = torch.cumsum(cnt, 0) - cnt
      e = torch.repeat_interleave(torch.arange(s.shape[0], device = dev), cnt[s])   # (edge, membership of its source)
      r = torch.arange(e.shape[0], device = dev) - torch.repeat_interleave(torch.cumsum(cnt[s], 0) - cnt[s], cnt[s])
      ms = first[s][e] + r
      keys = mn * K + mc
      want = t[e] * K + mc[ms]
      hit = torch.searchsorted(keys, want).clamp(max = max(keys.shape[0] - 1, 0))
      found = (keys[hit] == want).double() if keys.shape[0] else torch.zeros(0, dtype = torch.float64, device = dev)
      inside[rows] = torch.zeros(len(rows), dtype = torch.float64, device = dev).index_add_(0, ms, w[e] * found).cpu().numpy()
   return {"node": node, "code": code, "degree": deg, "inside": inside, "labels": labels,
           "size": np.bincount(code, minlength = K).astype(float), "size1": np.bincount(code, typ[node], minlength = K),
           "volume": np.bincount(code, deg[node], minlength = K), "internal": np.bincount(code, inside, minlength = K),
           "total": deg.sum()}


def _summarize(values, st, community, aggregate):
   """
   One value per community: that of ``community``, their (size weighted) mean, or all of them.
   """
   if community is not None:
      index = dict(zip(st["labels"], range(len(st["labels"]))))
      if community not in index:
         raise KeyError(f"no community {community!r}")
      return float(values[index[community]])
   if aggregate is None:
      return pd.Series(values, index = st["labels"])
   if aggregate not in ("mean", "size"):
      raise ValueError(f"unknown aggregate: {aggregate}")
   ok = ~np.isnan(values)
   if not ok.any():
      return float("nan")
   return float(np.average(values[ok], weights = st["size"][ok] if aggregate == "size" else None))


@_higher_order()
def conductance(net, partition, community = None, weight = None, aggregate = "mean", component = None):
   """
   Conductance of communities: the weight of the edges leaving a community (its cut) over the
   smaller of its volume (sum of the degrees of its nodes) and the volume of the rest of the network,

   .. math::

      \\phi(S) = \\frac{\\mathrm{cut}(S)}{\\min(\\mathrm{vol}(S), \\mathrm{vol}(V \\setminus S))}.

   It is between 0 (no edge leaves the community) and 1; the lower, the better separated the
   community. Directions are ignored and self loops dropped; parallel edges add their weights. In a
   cover (overlapping communities) every community is the set of its nodes. NaN for a community
   whose volume, or that of the rest, is 0.

   :param net: A :class:`~graphoskop.Network`.
   :param partition: The communities, as for :func:`modularity` (a cover is allowed).
   :param community: Label of a community, to get its conductance only.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param aggregate: Overall value, without ``community``: ``"mean"`` over the communities,
      ``"size"`` for their mean weighted by number of nodes, or ``None`` for a
      :class:`pandas.Series` with the value of every community (by label). NaN values are skipped.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :return: A float, or a :class:`pandas.Series`.
   :raises KeyError: If there is no such community or attribute.
   :raises ValueError: If ``aggregate`` is not valid, or as :func:`modularity`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      gsk.conductance(net, "party")                       # mean over the parties
      gsk.conductance(net, "party", community = "green")
      gsk.conductance(net, "party", aggregate = None)     # every party
   """
   st = _community_stats(net, partition, weight, component)
   cut = st["volume"] - st["internal"]
   den = np.minimum(st["volume"], st["total"] - st["volume"])
   with np.errstate(invalid = "ignore", divide = "ignore"):
      phi = np.where(den > 0, cut / np.where(den > 0, den, 1), np.nan)
   return _summarize(phi, st, community, aggregate)


@_higher_order()
def internal_density(net, partition, community = None, weight = None, aggregate = "mean", component = None):
   """
   Internal density of communities: the weight of the edges within a community over the number of
   pairs of its nodes that could be connected, :math:`\\frac{2 E_S}{n_S (n_S - 1)}` in undirected
   networks (:math:`\\frac{E_S}{n_S (n_S - 1)}` in directed ones, where :math:`E_S` counts the edges
   in both directions; :math:`\\frac{E_S}{n^{(0)}_S n^{(1)}_S}` in bipartite ones, pairs of nodes of
   different types). Multilayer networks are taken as their supra-graph: copies are nodes, and
   couplings and interlayer edges are edges. With unit weights it is between 0 and 1 (1 for a
   clique). Self loops are dropped; parallel edges add their weights. NaN for a community without
   pairs (a single node).

   :param net: A :class:`~graphoskop.Network`.
   :param partition: The communities, as for :func:`modularity` (a cover is allowed).
   :param community: Label of a community, to get its density only.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param aggregate: Overall value, without ``community``, as for :func:`conductance`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :return: A float, or a :class:`pandas.Series`.
   :raises KeyError: If there is no such community or attribute.
   :raises ValueError: If ``aggregate`` is not valid, or as :func:`modularity`.

   Example:

   .. code-block:: python

      gsk.internal_density(net, "party")                  # mean over the parties
      gsk.internal_density(net, "party", aggregate = "size")
   """
   st = _community_stats(net, partition, weight, component)
   n, n1 = st["size"], st["size1"]
   pairs = (n - n1) * n1 if net.bipartite_split is not None else n * (n - 1) / 2
   pairs = pairs * (2 if net.directed else 1)
   with np.errstate(invalid = "ignore", divide = "ignore"):
      d = np.where(pairs > 0, st["internal"] / (2 * np.where(pairs > 0, pairs, 1)), np.nan)
   return _summarize(d, st, community, aggregate)


@_higher_order()
def out_degree_fraction(net, partition, kind = "average", community = None, weight = None, aggregate = "mean", component = None):
   """
   Out-degree fraction of communities (Yang and Leskovec, 2012): how much of the degree of the
   nodes of a community points outside it. For a node :math:`u` of community :math:`S`, the
   fraction is :math:`f_u = \\frac{d_u - d_u^S}{d_u}`, with :math:`d_u` its degree (strength) and
   :math:`d_u^S` the part of it towards nodes of :math:`S`; ``kind`` gives, for the community:

   * ``"maximum"``: the largest :math:`f_u` among its nodes;
   * ``"average"``: the mean :math:`f_u` over its nodes;
   * ``"flake"`` (Flake et al., 2000): the fraction of its nodes with more weight outside the
     community than inside (:math:`d_u^S < d_u / 2`).

   All are between 0 and 1; the lower, the better. Nodes without edges count 0. Directions are
   ignored and self loops dropped; parallel edges add their weights. In a cover every community
   is the set of its nodes.

   :param net: A :class:`~graphoskop.Network`.
   :param partition: The communities, as for :func:`modularity` (a cover is allowed).
   :param kind: ``"maximum"``, ``"average"`` or ``"flake"``.
   :param community: Label of a community, to get its value only.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param aggregate: Overall value, without ``community``, as for :func:`conductance`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :return: A float, or a :class:`pandas.Series`.
   :raises KeyError: If there is no such community or attribute.
   :raises ValueError: If ``kind`` or ``aggregate`` are not valid, or as :func:`modularity`.

   Example:

   .. code-block:: python

      gsk.out_degree_fraction(net, "party")                       # average, mean over the parties
      gsk.out_degree_fraction(net, "party", kind = "flake", aggregate = None)
   """
   if kind not in ("maximum", "average", "flake"):
      raise ValueError(f"unknown kind: {kind}")
   st = _community_stats(net, partition, weight, component)
   d, code, K = st["degree"][st["node"]], st["code"], len(st["labels"])
   with np.errstate(invalid = "ignore", divide = "ignore"):
      f = np.where(d > 0, (d - st["inside"]) / np.where(d > 0, d, 1), 0.)
   if kind == "maximum":
      v = np.zeros(K)
      np.maximum.at(v, code, f)
   elif kind == "average":
      v = np.bincount(code, f, minlength = K) / st["size"]
   else:
      v = np.bincount(code, st["inside"] < d / 2, minlength = K) / st["size"]
   return _summarize(v, st, community, aggregate)


def _two_partitions(net, partition1, partition2):
   """
   Memberships of two partitions (or covers) of all the nodes.

   :return: ``(n, (node1, code1), (node2, code2), overlapping)``.
   """
   comps = list(range(len(net.components)))
   node1, code1, n = _labels(net, partition1, comps)
   node2, code2, _ = _labels(net, partition2, comps)
   return n, (node1, code1), (node2, code2), len(node1) != n or len(node2) != n


def _entropy(counts):
   p = counts[counts > 0] / counts.sum()
   return float(-(p * np.log(p)).sum())


def _contingency(code1, code2):
   """
   Sizes of the clusters of two partitions (codes of the same nodes) and their nonzero
   intersections.
   """
   a, b = np.bincount(code1).astype(float), np.bincount(code2).astype(float)
   key, nij = np.unique(code1 * len(b) + code2, return_counts = True)
   return a, b, key // len(b), key % len(b), nij.astype(float)


def _mean(x, y, average):
   """
   Mean of two entropies, by ``average``.
   """
   return {"arithmetic": (x + y) / 2, "geometric": np.sqrt(x * y), "max": max(x, y), "min": min(x, y)}[average]


@_higher_order()
def normalized_mutual_information(net, partition1, partition2, average = "arithmetic", overlap = "mcdaid"):
   """
   Normalized mutual information between two partitions of the nodes (or covers, with overlapping
   communities): 1 if they are the same (up to the labels), close to 0 if independent.

   For partitions, :math:`NMI = I(X; Y) / \\mathrm{mean}(H(X), H(Y))`, with :math:`I` the mutual
   information and :math:`H` the entropy of the community sizes; ``average`` is the mean in the
   denominator (``"arithmetic"``, as in scikit-learn, ``"geometric"``, ``"max"`` or ``"min"``).

   If some node is in several communities, of either partition, the overlapping variant is used:
   every community is a binary variable over the nodes (in it or not), each community of one cover
   is explained by the community of the other that leaves the least uncertainty on it, among those
   that agree with it more than they disagree (Lancichinetti, Fortunato and Kertész, 2009): a
   community much smaller than any community of the other cover containing it stays unexplained,
   so covers that merge many communities score low. With ``overlap``:

   * ``"mcdaid"`` (McDaid, Greene and Hurley, 2011): the conditional entropies of the communities
     are summed, :math:`I = \\frac{1}{2} [H(X) - H(X|Y) + H(Y) - H(Y|X)]` and
     :math:`NMI = I / \\max(H(X), H(Y))`;
   * ``"lfk"`` (Lancichinetti, Fortunato and Kertész, 2009): :math:`NMI = 1 - \\frac{1}{2}
     [H(X|Y)_{norm} + H(Y|X)_{norm}]`, with the conditional entropy of every community divided by
     its entropy and averaged over the communities of the cover.

   Only pairs of communities sharing nodes are compared. The overlapping variants differ from the
   classic one even without overlaps.

   :param net: A :class:`~graphoskop.Network`.
   :param partition1: The first partition: name of a node attribute, dictionary (or Series) from
      node id to label, or sequence of labels in network wide node order, as for
      :func:`modularity`; a node can have several labels.
   :param partition2: The second partition, in the same forms.
   :param average: Mean of the entropies in the denominator, for partitions.
   :param overlap: Variant for covers: ``"mcdaid"`` or ``"lfk"``.
   :return: The NMI, a float.
   :raises ValueError: If ``average`` or ``overlap`` are not valid, or a node has no label.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.lfr_benchmark_graph(n = 1000, mu = 0.3, seed = 0)        # planted "community"
      gsk.leiden_communities(net, attribute = "found", seed = 0)
      gsk.normalized_mutual_information(net, "community", "found")
      gsk.normalized_mutual_information(net, "cover", "found", overlap = "lfk")
   """
   if average not in ("arithmetic", "geometric", "max", "min"):
      raise ValueError(f"unknown average: {average}")
   if overlap not in ("mcdaid", "lfk"):
      raise ValueError(f"unknown overlap: {overlap}")
   n, (n1, c1), (n2, c2), cover = _two_partitions(net, partition1, partition2)
   if cover:
      return _overlapping_nmi(n, n1, c1, n2, c2, overlap)
   a, b, _, _, nij = _contingency(c1, c2)
   if len(a) == len(b) == 1:
      return 1.
   ha, hb = _entropy(a), _entropy(b)
   mi = ha + hb - _entropy(nij)
   den = _mean(ha, hb, average)
   return float(max(mi, 0.) / den) if den > 0 else 0.


def _overlapping_nmi(n, n1, c1, n2, c2, overlap):
   """
   NMI of two covers (Lancichinetti, Fortunato and Kertész, 2009; McDaid, Greene and Hurley, 2011).
   """
   h = lambda p: np.where(p > 0, -p * np.log2(np.where(p > 0, p, 1)), 0.)
   sx, sy = np.bincount(c1) / n, np.bincount(c2) / n
   Hx, Hy = h(sx) + h(1 - sx), h(sy) + h(1 - sy)
   pairs = pd.merge(pd.DataFrame({"node": n1, "x": c1}), pd.DataFrame({"node": n2, "y": c2}), on = "node")
   k = pairs.groupby(["x", "y"]).size()
   x, y = k.index.get_level_values(0).to_numpy(), k.index.get_level_values(1).to_numpy()
   p11 = k.to_numpy() / n
   p10, p01 = sx[x] - p11, sy[y] - p11
   p00 = 1 - p11 - p10 - p01
   ok = h(p11) + h(p00) >= h(p01) + h(p10)
   joint = h(p11) + h(p10) + h(p01) + h(p00)
   Hxy, Hyx = Hx.copy(), Hy.copy()   # H(X_k | Y), H(Y_l | X): the best community of the other cover
   np.minimum.at(Hxy, x[ok], (joint - Hy[y])[ok])
   np.minimum.at(Hyx, y[ok], (joint - Hx[x])[ok])
   if overlap == "lfk":
      with np.errstate(invalid = "ignore", divide = "ignore"):
         rx, ry = np.where(Hx > 0, Hxy / np.where(Hx > 0, Hx, 1), 0.), np.where(Hy > 0, Hyx / np.where(Hy > 0, Hy, 1), 0.)
      return float(1 - (rx.mean() + ry.mean()) / 2)
   den = max(Hx.sum(), Hy.sum())
   if den == 0:
      return 1.
   return float((Hx.sum() - Hxy.sum() + Hy.sum() - Hyx.sum()) / 2 / den)


def _expected_mutual_information(a, b, n):
   """
   Expected mutual information of two partitions with cluster sizes ``a`` and ``b`` under random
   permutations (hypergeometric model; Vinh, Epps and Bailey, 2010), in nats.
   """
   from scipy.special import gammaln
   emi = 0.
   B = b[:, None]
   const = gammaln(B + 1) + gammaln(n - B + 1) - gammaln(n + 1)
   sizes, mult = np.unique(a, return_counts = True)
   for ai, m in zip(sizes, mult):
      step = max(1, int(4e6 // len(b)))
      for lo in range(1, int(min(ai, b.max())) + 1, step):
         X = np.arange(lo, min(lo + step, int(min(ai, b.max())) + 1), dtype = float)[None, :]
         ok = (X <= B) & (X >= ai + B - n)
         Xs, Bs = np.where(ok, X, 1.), np.where(ok, B, 1.)
         logp = gammaln(ai + 1) + gammaln(n - ai + 1) + const - gammaln(Xs + 1) - gammaln(ai - Xs + 1) - gammaln(np.maximum(Bs - Xs, 0) + 1) - gammaln(np.maximum(n - ai - Bs + Xs, 0) + 1)
         term = Xs / n * np.log(n * Xs / (ai * Bs)) * np.exp(logp)
         emi += m * term[ok].sum()
   return emi


@_higher_order()
def adjusted_mutual_information(net, partition1, partition2, average = "arithmetic"):
   """
   Adjusted mutual information between two partitions of the nodes (Vinh, Epps and Bailey, 2010):
   the mutual information corrected for chance,

   .. math::

      AMI = \\frac{I(X; Y) - E[I]}{\\mathrm{mean}(H(X), H(Y)) - E[I]},

   with :math:`E[I]` its expected value between random partitions with the same community sizes
   (hypergeometric model). 1 if the partitions are the same, about 0 (possibly negative) if they are
   independent: unlike :func:`normalized_mutual_information`, it does not grow with the number of
   communities. ``average`` as for :func:`normalized_mutual_information`. As in scikit-learn.
   There is no adjusted variant for overlapping communities: covers raise an error.

   :param net: A :class:`~graphoskop.Network`.
   :param partition1: The first partition, as for :func:`normalized_mutual_information`.
   :param partition2: The second partition.
   :param average: Mean of the entropies in the denominator.
   :return: The AMI, a float.
   :raises ValueError: If ``average`` is not valid, a node has no label, or several.

   Example:

   .. code-block:: python

      gsk.adjusted_mutual_information(net, "community", "found")
   """
   if average not in ("arithmetic", "geometric", "max", "min"):
      raise ValueError(f"unknown average: {average}")
   n, (n1, c1), (n2, c2), cover = _two_partitions(net, partition1, partition2)
   if cover:
      raise ValueError("adjusted mutual information is not defined for overlapping communities")
   a, b, _, _, nij = _contingency(c1, c2)
   if len(a) == len(b) == 1 or len(a) == len(b) == n:
      return 1.
   ha, hb = _entropy(a), _entropy(b)
   mi = ha + hb - _entropy(nij)
   emi = _expected_mutual_information(a, b, n)
   den = _mean(ha, hb, average) - emi
   if abs(den) < np.finfo(float).eps:
      den = np.finfo(float).eps if den >= 0 else -np.finfo(float).eps
   return float((mi - emi) / den)


@_higher_order()
def adjusted_rand_index(net, partition1, partition2):
   """
   Adjusted Rand index between two partitions of the nodes (Hubert and Arabie, 1985): the fraction
   of pairs of nodes on which the partitions agree (together in both, or apart in both), corrected
   for chance,

   .. math::

      ARI = \\frac{\\sum_{ij} \\binom{n_{ij}}{2} - \\sum_i \\binom{a_i}{2} \\sum_j \\binom{b_j}{2} / \\binom{n}{2}}
      {\\frac{1}{2} [\\sum_i \\binom{a_i}{2} + \\sum_j \\binom{b_j}{2}] - \\sum_i \\binom{a_i}{2} \\sum_j \\binom{b_j}{2} / \\binom{n}{2}},

   with :math:`n_{ij}` the nodes in community :math:`i` of one and :math:`j` of the other, and
   :math:`a_i`, :math:`b_j` the community sizes. 1 for the same partition, about 0 (possibly
   negative) for independent ones.

   If some node is in several communities, the Omega index is used (Collins and Dent, 1988), its
   generalization to covers: two partitions agree on a pair of nodes when the pair shares the same
   number of communities in both, and Omega is the fraction of agreeing pairs corrected for chance
   (given how many pairs share 0, 1, 2, ... communities in each). Without overlaps it equals the
   adjusted Rand index.

   :param net: A :class:`~graphoskop.Network`.
   :param partition1: The first partition, as for :func:`normalized_mutual_information`; a node can
      have several labels.
   :param partition2: The second partition.
   :return: The adjusted Rand index (or Omega index), a float.
   :raises ValueError: If a node has no label.

   Example:

   .. code-block:: python

      gsk.adjusted_rand_index(net, "community", "found")
      gsk.adjusted_rand_index(net, "cover", "found")       # Omega index
   """
   n, (n1, c1), (n2, c2), cover = _two_partitions(net, partition1, partition2)
   if cover:
      return _omega(n, n1, c1, n2, c2)
   a, b, _, _, nij = _contingency(c1, c2)
   pairs = lambda x: (x * (x - 1) / 2).sum()
   index, sa, sb, total = pairs(nij), pairs(a), pairs(b), n * (n - 1) / 2
   expected = sa * sb / total if total > 0 else 0.
   den = (sa + sb) / 2 - expected
   return float((index - expected) / den) if den != 0 else 1.


def _omega(n, n1, c1, n2, c2):
   """
   Omega index of two covers (Collins and Dent, 1988). Nodes with the same communities in both
   covers are grouped in types, and only pairs of types sharing a community are listed.
   """
   sig1 = pd.Series(c1).groupby(n1).agg(tuple)
   sig2 = pd.Series(c2).groupby(n2).agg(tuple)
   typ, uniq = pd.factorize(pd.Series(list(zip(sig1.to_numpy(), sig2.to_numpy()))))
   size = np.bincount(typ).astype(float)
   t1 = np.array([len(u[0]) for u in uniq])
   t2 = np.array([len(u[1]) for u in uniq])
   counts = {}

   def add(j1, j2, w):
      for p, q, v in zip(j1, j2, w):
         counts[(p, q)] = counts.get((p, q), 0.) + v

   add(t1, t2, size * (size - 1) / 2)   # pairs within a type
   shared = []
   for nodes, codes in ((n1, c1), (n2, c2)):   # pairs of types sharing a community: how many
      rows = pd.DataFrame({"c": codes, "t": typ[nodes]}).drop_duplicates()
      m = rows.merge(rows, on = "c")
      m = m[m.t_x < m.t_y]
      shared.append(m.groupby(["t_x", "t_y"]).size())
   both = pd.concat(shared, axis = 1).fillna(0)
   x, y = both.index.get_level_values(0).to_numpy(), both.index.get_level_values(1).to_numpy()
   add(both[0].to_numpy().astype(int), both[1].to_numpy().astype(int), size[x] * size[y])
   M = n * (n - 1) / 2
   counts[(0, 0)] = counts.get((0, 0), 0.) + M - sum(counts.values())
   if M == 0:
      return 1.
   obs = sum(v for (p, q), v in counts.items() if p == q) / M
   m1, m2 = {}, {}
   for (p, q), v in counts.items():
      m1[p], m2[q] = m1.get(p, 0.) + v, m2.get(q, 0.) + v
   exp = sum(v * m2.get(p, 0.) for p, v in m1.items()) / M ** 2
   return float((obs - exp) / (1 - exp)) if exp < 1 else 1.


@_higher_order()
def hierarchical_random_graph_communities(net, attribute = "community", samples = 100, burnin = 20000, interval = 10, consensus = 0.5, chains = None, seed = None):
   """
   Hierarchical communities from hierarchical random graphs (Clauset, Moore and Newman, 2008): the
   consensus of the dendrograms sampled in proportion to their likelihood, as in
   :func:`~graphoskop.hierarchical_random_graph_lp`.

   A hierarchical random graph is a dendrogram whose leaves are the nodes, with a probability
   :math:`p_r` of connection between the leaves whose lowest common ancestor is the internal node
   :math:`r`. Every internal node is a group of nodes (its leaves). The dendrograms are sampled by
   Markov chain Monte Carlo (``burnin`` sweeps of :math:`n` moves per chain, then a sample every
   ``interval`` sweeps, ``samples`` in all, from chains running in parallel), and the groups found
   in more than a fraction ``consensus`` of the samples form the majority consensus tree: with
   ``consensus`` at least 0.5 the groups are nested or disjoint, so they form a tree, in which an
   internal node can have more than two children, and nodes whose place in the hierarchy is
   uncertain hang from the lowest group that is certain. Groups can be communities (dense inside)
   or sets of nodes linked mostly to the rest (as the two types of a bipartite network): the model
   learns both.

   The levels of the consensus tree are written as numeric node attributes, top first:
   ``attribute`` for the children of the root, ``attribute + "_1"``, ``attribute + "_2"``, ... for
   the levels below, each refining the one above (in branches that stop earlier, nodes keep their
   deepest group; a node hanging from the root is a group of its own). Groups are numbered by
   decreasing size at every level.

   The model is undirected and unweighted: directions, weights, parallel edges and self loops are
   ignored; isolated nodes are leaves too. The chains are those of
   :func:`~graphoskop.hierarchical_random_graph_lp`, but the consensus needs them to have converged
   more than the edge probabilities do: a group is kept only if the chains agree on it, so chains
   still climbing towards different dendrograms leave most nodes hanging from the root. Hence the
   larger default ``burnin``: on a benchmark of 1,000 nodes, 5,000 sweeps found 3 groups, 20,000
   sweeps 30 (normalized mutual information with the planted communities 0.58 and 0.88). Larger
   networks need more; if the top level has mostly single nodes, raise ``burnin``. The samples are
   kept in memory (about 30 bytes per node and sample).

   :param net: A :class:`~graphoskop.Network`, not multilayer, modified in place.
   :param attribute: Name of the node attribute of the top level.
   :param samples: Number of sampled dendrograms.
   :param burnin: Number of sweeps of each chain before its first sample.
   :param interval: Number of sweeps between samples.
   :param consensus: A group is kept if found in more than this fraction of the samples, between
      0.5 and 1.
   :param chains: Number of chains, ``None`` for the number of Numba threads.
   :param seed: Seed of the random numbers, or ``None``.
   :return: A dictionary with the mean ``log_likelihood`` of the sampled dendrograms, the number
      of ``groups`` of the consensus tree (its internal nodes, the root excluded) and
      ``communities``, the number of groups of every level, top first.
   :raises ValueError: If the network is multilayer, or ``samples``, ``burnin``, ``interval`` or
      ``consensus`` are not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      res = gsk.hierarchical_random_graph_communities(net, seed = 0)    # community, community_1, ...
      res["communities"]
      gsk.hierarchical_random_graph_communities(net, burnin = 20000, consensus = 0.9, attribute = "sure")
   """
   from .link_prediction import _hrg_setup, _hrg_run, _hrg_trees
   if net.layer_labels is not None:
      raise ValueError("hierarchical random graphs do not support multilayer networks")
   if samples < 1 or burnin < 0 or interval < 1:
      raise ValueError(f"need samples >= 1, burnin >= 0 and interval >= 1, got {samples}, {burnin}, {interval}")
   if not 0.5 <= consensus < 1:
      raise ValueError("consensus must be at least 0.5 and below 1")
   N, ptr, nbr, counts, states, trees, ll = _hrg_setup(net, samples, chains, seed)
   m = max(N - 1, 0)
   H, P = np.zeros((samples, m), dtype = np.uint64), np.zeros((samples, m))
   LO, SZ = np.zeros((samples, m), dtype = np.int32), np.zeros((samples, m), dtype = np.int32)
   LEAF, LL = np.zeros((samples, N), dtype = np.int32), np.zeros(samples)
   none = np.zeros(0, dtype = np.int64)
   if N > 2:
      _hrg_run(N, ptr, nbr, *trees, states, ll, int(burnin), 1, np.zeros(len(counts), dtype = np.int64), none, none, np.zeros((len(counts), 0)))
   keys = np.random.default_rng(seed).integers(0, 2 ** 63, size = N, dtype = np.uint64) * np.uint64(2) + np.uint64(1)
   if N > 1:
      _hrg_trees(N, ptr, nbr, *trees, states, ll, interval, counts, keys, H, LO, SZ, P, LEAF, LL)
   uniq, first, cnt = np.unique(H.ravel(), return_index = True, return_counts = True)
   size = SZ.ravel()[first]
   keep = np.flatnonzero((cnt > consensus * samples) & (size < N))
   keep = keep[np.argsort(-size[keep], kind = "stable")]   # parents before children
   owner = np.full(N, -1)   # deepest group of every node (-1: the root)
   parent, depth, members = np.zeros(len(keep), dtype = np.int64), np.zeros(len(keep), dtype = np.int64), []
   for g, j in enumerate(keep):
      k, z = divmod(int(first[j]), m)
      leaves = LEAF[k, LO[k, z]:LO[k, z] + SZ[k, z]]
      parent[g] = owner[leaves[0]]
      depth[g] = depth[parent[g]] + 1 if parent[g] >= 0 else 1
      owner[leaves] = g
   D = int(depth.max()) if len(keep) else 1
   at = np.arange(len(keep))   # ancestor at the current level (or the group itself, if shallower)
   counts_out, offs = [], net.node_offsets
   for d in range(1, D + 1):
      for g in range(len(keep)):
         at[g] = at[parent[g]] if depth[g] > d else g
      lab = len(keep) + np.arange(N)   # nodes hanging from the root: alone
      lab[owner >= 0] = at[owner[owner >= 0]]
      uq, inv, sz = np.unique(lab, return_inverse = True, return_counts = True)
      rank = np.empty(len(uq), dtype = np.int64)
      rank[np.lexsort((uq, -sz))] = np.arange(len(uq))   # by decreasing size
      lab = rank[inv].astype(np.float64)
      counts_out.append(int(len(uq)))
      key = attribute if d == 1 else f"{attribute}_{d - 1}"
      net.set_node_attribute(key, [lab[offs[c]:offs[c + 1]] for c in range(len(net.components))], None)
   return {"log_likelihood": float(LL.mean()) if N > 1 else 0., "groups": int(len(keep)), "communities": counts_out}


def _bigclam_fit(N, crow, col, F, hs, ht, eps, max_iter, tol = 1e-4):
   """
   :func:`_bigclam_run` without the warnings of torch on sparse CSR tensors.
   """
   from .network import _quiet_sparse
   with _quiet_sparse():
      return _bigclam_run(N, crow, col, F, hs, ht, eps, max_iter, tol)


def _bigclam_run(N, crow, col, F, hs, ht, eps, max_iter, tol = 1e-4):
   """
   Maximum likelihood affiliations of BigCLAM from ``F``, on its device: edges in a symmetric CSR
   pattern (``crow``, ``col``; both directions, no self loops), pairs ``(hs, ht)`` held out
   (neither edges nor non-edges), background probability ``eps``
   (:math:`P(u \\sim v) = 1 - (1 - \\epsilon) e^{-F_u \\cdot F_v}`). The products
   :math:`F_u \\cdot F_v` on the edges are sampled matrix products, the sums over neighbors sparse
   products. Accelerated projected gradient ascent (FISTA, momentum restarted whenever the
   likelihood would decrease), each step scaled by the inverse diagonal of the curvature of the
   edge terms plus 1 and shortened until it increases the likelihood enough (Armijo); stops when
   the likelihood grows by less than ``tol`` (relative) in 20 steps.

   :return: ``(F, log_likelihood, iterations)``.
   """
   la = float(np.log1p(-eps))
   m = col.shape[0] // 2
   pairs = N * (N - 1) / 2 - m - len(hs)   # non-edges
   ones = torch.ones(col.shape[0], dtype = F.dtype, device = F.device)
   pattern = torch.sparse_csr_tensor(crow, col, ones, (N, N))
   csr = lambda v: torch.sparse_csr_tensor(crow, col, v, (N, N))

   def ll(F):
      x = torch.sparse.sampled_addmm(pattern, F, F.T, beta = 0.).values()
      tot = F.sum(0)
      other = (tot @ tot - (F * F).sum()) / 2 - x.sum() / 2 - (F[hs] * F[ht]).sum()   # sum over the non-edges
      return float(torch.log1p(-torch.exp(la - x)).sum() / 2 + pairs * la - other)

   def grad(F):   # gradient and inverse diagonal curvature
      x = torch.sparse.sampled_addmm(pattern, F, F.T, beta = 0.).values()
      e = torch.exp(la - x)
      g = csr(1 / (1 - e)) @ F - (F.sum(0)[None, :] - F)
      g = g.index_add_(0, hs, F[ht]).index_add_(0, ht, F[hs])
      return g, 1 / (csr(e / (1 - e) ** 2) @ (F * F) + 1)

   Y, tk, step, cur = F.clone(), 1., 1., ll(F)
   hist = [cur]
   it = 0
   for it in range(1, max_iter + 1):
      ly = ll(Y)
      g, P = grad(Y)
      while True:
         Fn = (Y + step * P * g).clamp(min = 0)
         new = ll(Fn)
         if new >= ly + 1e-4 * float((g * (Fn - Y)).sum()) or step < 1e-10:
            break
         step *= 0.5
      if new < cur:   # restart the momentum
         Y, tk = F.clone(), 1.
         continue
      tn = (1 + (1 + 4 * tk * tk) ** 0.5) / 2
      Y = (Fn + (tk - 1) / tn * (Fn - F)).clamp(min = 0)
      F, cur, tk, step = Fn, new, tn, min(step * 1.5, 1.)
      hist.append(cur)
      if len(hist) > 20 and cur - hist[-21] <= tol * abs(cur):
         break
   return F, cur, it


def _bigclam_init(N, ptr, nbr, deg, tri, K, gen):
   """
   Starting factors of BigCLAM: the neighborhoods (node and neighbors) of the ``K`` nodes whose
   neighborhood has the smallest conductance among those not beaten by a neighbor's (locally
   minimal neighborhoods, ties broken by index; random nodes if there are fewer), plus small
   noise.
   """
   dev = deg.device
   nsum = torch.zeros(N, dtype = torch.float64, device = dev).index_add_(0, torch.repeat_interleave(torch.arange(N, device = dev), ptr[1:] - ptr[:-1]), deg[nbr])
   vol = deg + nsum
   cut = vol - 2 * (deg + tri)   # edges at the node, and among its neighbors, are inside
   total = deg.sum()
   phi = torch.where(torch.minimum(vol, total - vol) > 0, cut / torch.minimum(vol, total - vol).clamp(min = 1e-300), torch.full_like(vol, float("inf")))
   src = torch.repeat_interleave(torch.arange(N, device = dev), ptr[1:] - ptr[:-1])
   beaten = (phi[nbr] < phi[src]) | ((phi[nbr] == phi[src]) & (nbr < src))   # ties: lower index wins
   lost = torch.zeros(N, dtype = torch.bool, device = dev).index_put_((src[beaten],), torch.ones(1, dtype = torch.bool, device = dev))
   local = torch.nonzero(~lost & (deg > 0)).flatten()
   seeds = local[torch.argsort(phi[local], stable = True)][:K]
   if len(seeds) < K:
      rest = torch.ones(N, dtype = torch.bool, device = dev)
      rest[seeds] = False
      pool = torch.nonzero(rest).flatten()
      seeds = torch.cat([seeds, pool[torch.randperm(len(pool), generator = gen, device = "cpu").to(dev)[:K - len(seeds)]]])
   F = 0.01 * torch.rand((N, K), generator = gen, dtype = torch.float64, device = "cpu").to(dev)
   F[seeds, torch.arange(len(seeds), device = dev)] = 1.
   cnt = ptr[seeds + 1] - ptr[seeds]
   rows = nbr[torch.cat([torch.arange(ptr[u], ptr[u + 1], device = dev) for u in seeds.tolist()])] if len(seeds) else seeds
   F[rows, torch.repeat_interleave(torch.arange(len(seeds), device = dev), cnt)] = 1.
   return F


@_higher_order()
def bigclam_communities(net, attribute = "community", groups = None, max_groups = 100, max_iter = 2000, seed = None):
   """
   BigCLAM (Yang and Leskovec, 2013): overlapping communities from a community affiliation model,
   the nonnegative factorization of the network that makes it most likely. Every node :math:`u`
   has a nonnegative affiliation :math:`F_{uc}` to every community :math:`c`; each community
   independently links two nodes with probability :math:`1 - e^{-F_{uc} F_{vc}}`, so that

   .. math::

      P(u \\sim v) = 1 - (1 - \\epsilon) e^{-F_u \\cdot F_v},

   with a background probability :math:`\\epsilon = 2m / (n (n - 1))` (the density of the network)
   of linking any two nodes, as in the community affiliation graph model (Yang and Leskovec, 2012)
   that BigCLAM relaxes: the more communities two nodes share, the more likely they are linked, so
   overlaps of communities are denser than the communities themselves, as in real networks (unlike
   methods that assume overlaps are sparse). The log-likelihood

   .. math::

      \\ell(F) = \\sum_{(u, v) \\in E} \\log(1 - (1 - \\epsilon) e^{-F_u \\cdot F_v}) + \\sum_{(u, v) \\notin E} [\\log(1 - \\epsilon) - F_u \\cdot F_v]

   is maximized over :math:`F \\geq 0`; the sum over the non-edges is the sum over all pairs,
   :math:`\\frac{1}{2} (\\| \\sum_u F_u \\|^2 - \\sum_u \\|F_u\\|^2)`, minus that over the edges,
   so every step costs :math:`O((m + n) K)` (sparse products). Unlike the paper (one node at a time,
   on the CPU), all the affiliations move together, on the device of the network, by accelerated
   projected gradient ascent (FISTA, with restarts), each step scaled by the inverse curvature of
   the edge terms and shortened until it increases the likelihood (the node by node updates of the
   paper stopped at a lower likelihood in our tests), until the log-likelihood grows by less than
   0.01% in 20 steps. The affiliations start from locally minimal neighborhoods, as in the paper:
   the neighborhoods (a node and its neighbors) of lowest conductance not beaten by those of their
   neighbors.

   A node is in community :math:`c` if :math:`F_{uc} \\geq \\delta = \\sqrt{-\\log(1 - \\epsilon)}`,
   the affiliation at which a community links two nodes as likely as the background; a node in none is put in the community of its largest affiliation (a
   community of its own if all are 0). Communities left empty are dropped.

   The number of communities ``groups``, if not given, is chosen as in the paper, by the likelihood
   of held-out pairs of nodes: 20% of the edges, and as many non-adjacent pairs (standing for 20% of
   all the non-adjacent pairs, their likelihood weighted accordingly), are held out, the model is
   fitted to the rest for 10 numbers between 2 and ``max_groups`` (evenly spaced on a log scale),
   and the number giving the largest likelihood of the held-out pairs is fitted to the whole
   network. This takes 10 more fits: giving ``groups`` is much faster.

   Directions, weights, parallel edges and self loops are ignored. Communities are numbered by
   decreasing size and written as the node attribute ``attribute``, with a tuple of numbers for a
   node in several communities.

   :param net: A :class:`~graphoskop.Network`, not multilayer, modified in place.
   :param attribute: Name of the node attribute written.
   :param groups: Number of communities :math:`K` of the model, or ``None`` to choose it.
   :param max_groups: Largest number of communities tried when ``groups`` is ``None``.
   :param max_iter: Maximum number of gradient steps of each fit.
   :param seed: Seed of the random numbers, or ``None``.
   :return: A dictionary with the number of ``groups`` of the model, the number of
      ``communities`` found (non-empty, counting the nodes without any affiliation, each alone),
      the number of ``overlapping`` nodes, the
      ``log_likelihood`` and the ``memberships`` (affiliations :math:`F`, a tensor of
      ``n x groups`` in network wide node order, columns in the order of the model, not of the
      community numbers).
   :raises ValueError: If the network is multilayer, or ``groups`` or ``max_groups`` are not
      valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", device = "cuda")
      res = gsk.bigclam_communities(net, seed = 0)                 # number of communities chosen
      res["groups"], res["overlapping"]
      res = gsk.bigclam_communities(net, groups = 50, attribute = "cover")
   """
   from .network_statistics import _skeleton, _paths
   if net.layer_labels is not None:
      raise ValueError("bigclam_communities does not support multilayer networks")
   N = int(net.node_offsets[-1])
   if groups is not None and not 1 <= groups <= N or max_groups < 2:
      raise ValueError("groups must be between 1 and the number of nodes, max_groups at least 2")
   dev = net.components[0].edge_index.device
   ei = torch.cat([d.edge_index + int(net.node_offsets[c]) for c, d in enumerate(net.components)], 1)
   a, b = torch.minimum(ei[0], ei[1]), torch.maximum(ei[0], ei[1])
   key = torch.unique((a * N + b)[a != b])
   s, t = key // N, key % N   # every edge once
   gen = torch.Generator().manual_seed(int(np.random.default_rng(seed).integers(2 ** 62)))

   def graph(s, t):
      both = torch.cat([s, t])
      order = torch.argsort(both * N + torch.cat([t, s]))   # rows, columns sorted (CSR)
      ptr = torch.cat([torch.zeros(1, dtype = torch.long, device = dev), torch.cumsum(torch.bincount(both, minlength = N), 0)])
      nbr = torch.cat([t, s])[order]
      deg = torch.bincount(both, minlength = N).double()
      tri = torch.zeros(N, dtype = torch.float64, device = dev)
      sk, dk, _, _ = _skeleton(s, t, N)
      for ab, bc, ac in _paths(sk, dk, N, 2 ** 22):
         for x in (sk[ab], dk[ab], dk[bc]):
            tri.index_add_(0, x, torch.ones(len(x), dtype = torch.float64, device = dev))
      return ptr, nbr, deg, tri

   none = torch.zeros(0, dtype = torch.long, device = dev)
   eps = min(2 * len(s) / (N * (N - 1)), 0.5) if N > 1 else 0.
   if groups is None:
      perm = torch.randperm(len(s), generator = gen).to(dev)
      h = len(s) // 5
      hs, ht, ks, kt = s[perm[:h]], t[perm[:h]], s[perm[h:]], t[perm[h:]]
      u = torch.randint(0, N, (4 * h + 16,), generator = gen).to(dev)
      v = torch.randint(0, N, (4 * h + 16,), generator = gen).to(dev)
      x, y = torch.minimum(u, v), torch.maximum(u, v)
      ok = (x != y) & ~torch.isin(x * N + y, key)
      nk = torch.unique(x[ok] * N + y[ok])
      nk = nk[torch.randperm(len(nk), generator = gen).to(dev)[:h]]
      ns, nt = nk // N, nk % N
      scale = 0.2 * (N * (N - 1) / 2 - len(s)) / max(len(nk), 1)   # sampled non-edges stand for 20% of all
      la = float(np.log1p(-eps))
      ptr, nbr, deg, tri = graph(ks, kt)
      best = None
      for K in sorted(set(np.round(np.geomspace(2, max_groups, 10)).astype(int).tolist())):
         if K > N:
            break
         F, _, _ = _bigclam_fit(N, ptr, nbr, _bigclam_init(N, ptr, nbr, deg, tri, K, gen), torch.cat([hs, ns]), torch.cat([ht, nt]), eps, max_iter)
         with torch.no_grad():
            score = float(torch.log1p(-torch.exp(la - (F[hs] * F[ht]).sum(1))).sum() + scale * (len(nk) * la - (F[ns] * F[nt]).sum()))
         if best is None or score > best[0]:
            best = (score, K)
      groups = best[1]
   ptr, nbr, deg, tri = graph(s, t)
   F, ll, _ = _bigclam_fit(N, ptr, nbr, _bigclam_init(N, ptr, nbr, deg, tri, int(groups), gen), none, none, eps, max_iter)
   delta = np.sqrt(-np.log(1 - eps)) if eps > 0 else 0.
   Fc = F.cpu().numpy()
   inside = Fc >= delta if delta > 0 else Fc > 0
   none_in = ~inside.any(1)
   top = Fc.argmax(1)
   inside[none_in & (Fc.max(1) > 0), top[none_in & (Fc.max(1) > 0)]] = True
   node, code = np.nonzero(inside)
   alone = np.flatnonzero(~inside.any(1))   # all affiliations 0
   node, code = np.r_[node, alone], np.r_[code, groups + np.arange(len(alone))]
   K, multi = _write_cover(net, attribute, node, code)
   return {"groups": int(groups), "communities": K, "overlapping": multi, "log_likelihood": ll, "memberships": F}



def _links(net, weight):
   """
   Links of a network for link communities: the distinct pairs of adjacent nodes (directions
   ignored, parallel edges merged, weights summed, self loops dropped), over network wide node
   positions, and the pairs of links sharing a node.

   :return: ``(N, a, b, w, p, q, k)``: links ``(a, b)`` with ``a < b`` and weights ``w``, and every
      pair of links ``(p, q)`` sharing the node ``k``.
   """
   if net.layer_labels is not None:
      raise ValueError("link communities of multilayer networks are not supported")
   N = int(net.node_offsets[-1])
   S, T, W = [], [], []
   for c in range(len(net.components)):
      ei, _ = net._input_edges(c)
      S.append(ei[0].cpu().numpy() + int(net.node_offsets[c]))
      T.append(ei[1].cpu().numpy() + int(net.node_offsets[c]))
      W.append(np.ones(ei.shape[1]) if weight is None else net._numeric_edge(weight, c)[:ei.shape[1]].double().cpu().numpy())
   s, t, w = np.concatenate(S), np.concatenate(T), np.concatenate(W)
   if (w < 0).any():
      raise ValueError("weights must be non negative")
   keep = s != t
   x, y = np.minimum(s, t)[keep], np.maximum(s, t)[keep]
   key, inv = np.unique(x * N + y, return_inverse = True)
   a, b = key // N, key % N
   w = np.bincount(inv, w[keep]) if weight is not None else np.ones(len(key))
   M = len(a)
   ends = np.r_[a, b]   # incidences: node, link
   order = np.argsort(ends, kind = "stable")
   inc, nodes = np.r_[np.arange(M), np.arange(M)][order], ends[order]
   deg = np.bincount(ends, minlength = N)
   start = np.r_[0, np.cumsum(deg)]
   pos = np.arange(2 * M) - start[nodes]
   cnt = deg[nodes] - 1 - pos   # later incidences of the same node
   first = np.repeat(np.arange(2 * M), cnt)
   second = first + 1 + np.arange(len(first)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
   return N, a, b, w, inc[first], inc[second], nodes[first]


def _write_links(net, attribute, N, a, b, lab):
   """
   Write link communities ``lab`` of the links ``(a, b)`` as the numeric edge attribute
   ``attribute`` (every edge gets the community of its pair of nodes; self loops NaN) and the cover
   of the nodes they induce as the node attribute ``attribute`` (a node is in the communities of
   its links; a node without links is alone). Communities are numbered by decreasing number of
   nodes.

   :return: The number of link communities and of nodes in several.
   """
   node, code = np.r_[a, b], np.r_[lab, lab]
   key = np.unique(node * (lab.max() + 1 if len(lab) else 1) + code)
   node, code = key // (lab.max() + 1 if len(lab) else 1), key % (lab.max() + 1 if len(lab) else 1)
   uniq, inv, size = np.unique(code, return_inverse = True, return_counts = True)
   rank = np.empty(len(uniq), dtype = np.int64)
   rank[np.lexsort((uniq, -size))] = np.arange(len(uniq))
   code = rank[inv]
   K = len(uniq)
   lab = rank[np.searchsorted(uniq, lab)] if len(lab) else lab
   alone = np.setdiff1d(np.arange(N), node)
   _, multi = _write_cover(net, attribute, np.r_[node, alone], np.r_[code, K + np.arange(len(alone))])
   keys = a * N + b
   values = []
   for c in range(len(net.components)):
      s, t = net.components[c].edge_index.cpu().numpy() + int(net.node_offsets[c])
      k = np.minimum(s, t) * N + np.maximum(s, t)
      hit = np.clip(np.searchsorted(keys, k), 0, max(len(keys) - 1, 0))
      found = (keys[hit] == k) if len(keys) else np.zeros(len(k), dtype = bool)
      values.append(np.where(found & (s != t), lab[hit] if len(lab) else 0, np.nan).astype(np.float64))
   net.set_edge_attribute(attribute, values, None)
   return K, multi


@_higher_order()
def hierarchical_link_communities(net, attribute = "link_community", weight = None, threshold = None):
   """
   Link communities by hierarchical link clustering (Ahn, Bagrow and Lehmann, 2010): the links,
   rather than the nodes, are grouped, so that a node is in the communities of all its links and
   communities overlap naturally (a node can be in many), at every scale.

   Two links sharing a node :math:`k`, :math:`(i, k)` and :math:`(j, k)`, are as similar as the
   neighborhoods of :math:`i` and :math:`j`, nodes included: the Jaccard coefficient
   :math:`S = |N^+(i) \\cap N^+(j)| / |N^+(i) \\cup N^+(j)|`; with ``weight``, the Tanimoto
   coefficient :math:`S = a_i \\cdot a_j / (|a_i|^2 + |a_j|^2 - a_i \\cdot a_j)` of the vectors of
   weights :math:`a_{ij} = w_{ij}`, with :math:`a_{ii}` the mean weight of the links of
   :math:`i`. Links are merged by single linkage, by decreasing similarity, building a dendrogram
   of links, which is cut where the partition density

   .. math::

      D = \\frac{2}{M} \\sum_c m_c \\frac{m_c - (n_c - 1)}{(n_c - 2)(n_c - 1)}

   is largest (:math:`M` links in all, :math:`m_c` links and :math:`n_c` nodes in community
   :math:`c`; a community of one link counts 0): the density of every community beyond that of a
   tree on its nodes, weighted by its links. Or it is cut at a given similarity ``threshold``.

   The pairs of links sharing a node are :math:`\\sum_k \\binom{k_k}{2}`, so the time grows with
   the squares of the degrees; the clustering runs in a compiled kernel (Numba) on the CPU, keeping
   the nodes of every cluster to update the partition density at every merge.

   Directions are ignored, parallel edges are one link (weights summed), self loops are in no
   community. The link communities, numbered by decreasing number of nodes, are written as the
   numeric edge attribute ``attribute`` (both stored directions of an edge, and its parallel
   copies, get the community of the link; self loops NaN), and the communities of the nodes as the
   node attribute ``attribute``: a node in several has a tuple of numbers; a node without links is
   a community of its own.

   :param net: A :class:`~graphoskop.Network`, not multilayer, modified in place.
   :param attribute: Name of the edge and node attributes written.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param threshold: Similarity at which the dendrogram is cut, or ``None`` for the largest
      partition density.
   :return: A dictionary with the number of link ``communities``, the ``partition_density`` of the
      cut, the similarity ``threshold`` of the cut (infinite if no merge increases the partition
      density: every link alone), and the number of ``overlapping`` nodes (in several communities).
   :raises ValueError: If the network is multilayer or weights are negative.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      res = gsk.hierarchical_link_communities(net)                 # link_community, edges and nodes
      res["communities"], res["partition_density"]
      gsk.hierarchical_link_communities(net, threshold = 0.3, attribute = "links")
   """
   from ._links import _single_linkage, _cut
   N, a, b, w, p, q, k = _links(net, weight)
   M = len(a)
   i, j = np.where(a[p] == k, b[p], a[p]), np.where(a[q] == k, b[q], a[q])   # the other ends
   i, j = np.minimum(i, j), np.maximum(i, j)
   s = np.bincount(np.r_[a, b], np.r_[w, w], minlength = N)
   deg = np.bincount(np.r_[a, b], minlength = N)
   self_w = np.where(deg > 0, s / np.maximum(deg, 1), 0.)   # a_ii
   norm = np.bincount(np.r_[a, b], np.r_[w, w] ** 2, minlength = N) + self_w ** 2
   pk, pinv = np.unique(i * N + j, return_inverse = True)
   dot = np.bincount(pinv, w[p] * w[q])   # common neighbors k
   keys = a * N + b
   hit = np.clip(np.searchsorted(keys, pk), 0, max(M - 1, 0))
   adj = (keys[hit] == pk) if M else np.zeros(len(pk), dtype = bool)
   pi, pj = pk // N, pk % N
   dot = dot + np.where(adj, (self_w[pi] + self_w[pj]) * w[hit], 0.)   # i and j in each other's vectors
   sim = (dot / (norm[pi] + norm[pj] - dot))[pinv] if len(pk) else np.zeros(0)
   order = np.argsort(-sim, kind = "stable")
   p, q, sim = p[order], q[order], sim[order]
   if threshold is None:
      D, threshold = _single_linkage(N, a, b, p, q, sim) if M else (0., np.inf)
   lab = _cut(M, p, q, sim, float(threshold)) if M else np.zeros(0, dtype = np.int64)
   K, multi = _write_links(net, attribute, N, a, b, lab)
   _, comp = np.unique(lab, return_inverse = True)
   mc = np.bincount(comp)
   nc = np.bincount(np.unique(np.r_[comp * N + a, comp * N + b]) // N)
   D = float(2 / M * sum(x * (x - y + 1) / ((y - 2) * (y - 1)) for x, y in zip(mc, nc) if y > 2)) if M else 0.
   return {"communities": K, "partition_density": D, "threshold": float(threshold), "overlapping": multi}


@_higher_order()
def line_graph_communities(net, attribute = "link_community", weight = None, line_graph = "random_walk", resolution = 1, seed = None):
   """
   Link communities from the line graph (Evans and Lambiotte, 2009, 2010): the communities of the
   line graph, whose nodes are the links of the network, two of them adjacent if they share a
   node, are communities of links, so that a node is in the communities of all its links and
   communities overlap. The line graph is partitioned by maximizing its modularity with the Leiden
   algorithm (:func:`leiden_communities`, at ``resolution``). Its edge weights, for links
   :math:`\\alpha` and :math:`\\beta` sharing the node :math:`i`, follow ``line_graph``:

   * ``"plain"``: 1 (the line graph :math:`C`). A node of degree :math:`k` becomes a clique of
     :math:`\\binom{k}{2}` edges, so that high degree nodes dominate;
   * ``"weighted"``: :math:`1 / (k_i - 1)` (the line graph :math:`D`), so that the pairs of links
     of a node weigh :math:`k_i / 2` in all, in proportion to its degree as in the network;
   * ``"random_walk"``: :math:`w_\\alpha w_\\beta / s_i` with self loops
     :math:`\\sum_{i \\in \\alpha} w_\\alpha^2 / s_i` (the line graph :math:`E`, with the strength
     :math:`s_i` of node :math:`i`; :math:`1 / k_i` without weights): the flow of a random walker
     moving from link to link through their shared nodes, in proportion to the link weights (the
     only variant using ``weight``; Evans and Lambiotte, 2010). Its modularity is that of a
     partition of the links of the original network with a null model of walkers on links.

   The line graph has :math:`\\sum_i \\binom{k_i}{2}` edges, so memory grows with the squares of
   the degrees.

   Directions are ignored, parallel edges are one link (weights summed), self loops are in no
   community. Link communities are written as for :func:`hierarchical_link_communities`: the
   numeric edge attribute ``attribute`` and the node cover ``attribute``.

   :param net: A :class:`~graphoskop.Network`, not multilayer, modified in place.
   :param attribute: Name of the edge and node attributes written.
   :param weight: Name of the edge attribute used as edge weight (``"random_walk"`` only), or
      ``None`` for unit weights.
   :param line_graph: ``"random_walk"``, ``"weighted"`` or ``"plain"``.
   :param resolution: Resolution of the modularity of the line graph.
   :param seed: Seed of the random numbers, or ``None``.
   :return: A dictionary with the number of link ``communities``, the ``modularity`` of the
      partition of the line graph, and the number of ``overlapping`` nodes (in several
      communities).
   :raises ValueError: If the network is multilayer, ``line_graph`` is not valid, weights are
      negative, or ``weight`` is given with another line graph than ``"random_walk"``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      res = gsk.line_graph_communities(net, seed = 0)              # link_community, edges and nodes
      gsk.line_graph_communities(net, line_graph = "weighted", attribute = "links")
   """
   from .network import load_network
   if line_graph not in ("random_walk", "weighted", "plain"):
      raise ValueError(f"unknown line_graph: {line_graph}")
   if weight is not None and line_graph != "random_walk":
      raise ValueError("weights are used only by the random_walk line graph")
   N, a, b, w, p, q, k = _links(net, weight)
   M = len(a)
   deg = np.bincount(np.r_[a, b], minlength = N)
   s = np.bincount(np.r_[a, b], np.r_[w, w], minlength = N)
   if line_graph == "plain":
      lw = np.ones(len(p))
   elif line_graph == "weighted":
      lw = 1 / (deg[k] - 1.)
   else:
      lw = w[p] * w[q] / s[k]
   src, dst = p, q
   if line_graph == "random_walk":   # self loops: half of E_aa (they count twice in degrees)
      loop = (w ** 2 / s[a] + w ** 2 / s[b]) / 2
      src, dst, lw = np.r_[src, np.arange(M)], np.r_[dst, np.arange(M)], np.r_[lw, loop]
   key, inv = np.unique(src * max(M, 1) + dst, return_inverse = True)   # links sharing both ends: once
   lw = np.bincount(inv, lw)
   src, dst = key // max(M, 1), key % max(M, 1)
   L = load_network(pd.DataFrame({"source": src, "target": dst, "weight": lw}), pd.DataFrame({"node": np.arange(M)}), device = net.components[0].edge_index.device)
   r = leiden_communities(L, attribute = "community", weight = "weight", resolution = resolution, seed = seed) if M else {"quality": float("nan")}
   lab = np.zeros(M, dtype = np.int64)
   if M:
      ids = np.concatenate(L.node_ids).astype(np.int64)
      lab[ids] = np.concatenate([np.asarray(x.cpu() if torch.is_tensor(x) else x) for x in L.get_node_attribute("community", None)]).astype(np.int64)
   K, multi = _write_links(net, attribute, N, a, b, lab)
   return {"communities": K, "modularity": float(r["quality"]), "overlapping": multi}


@_higher_order()
def demon_communities(net, attribute = "community", epsilon = 0.25, min_size = 3, weight = None, max_iter = 100, seed = None):
   """
   DEMON (Coscia, Rossetti, Giannotti and Pedreschi, 2012): overlapping communities from the local
   point of view of every node. For every node, its ego network without it (its neighbors and the
   edges among them) is split into communities by label propagation (as in
   :func:`label_propagation_communities`, asynchronous); the node is added back to each of them,
   since it belongs to all the communities of its neighbors. Local communities with fewer than
   ``min_size`` nodes are dropped, and the others are merged into global ones: a local community
   merges into a global one if the nodes they share are at least a fraction ``epsilon`` of the
   smaller of the two.

   The ego networks are independent: their label propagations run in parallel, in a compiled
   kernel (Numba) on the CPU. Merging is sequential, as in the original: local communities are
   taken in order (by ego, then label), and each joins a global community or becomes a new one.
   Each local community is compared only with the global communities containing some of its nodes,
   found in an inverted index (the communities of every node), so its cost grows with the
   memberships of its nodes rather than with the number of communities. When several global
   communities are similar enough, it joins the most similar (the original joins the first found,
   in an order that changes as communities merge). Merging all similar pairs at once instead, by
   connected components, chains communities into a few giant ones at small ``epsilon``: the
   sequential merge limits chains, since a global community only grows when a new local
   community is close to it.

   Directions are ignored and self loops dropped; parallel edges add their weights. Communities
   are numbered by decreasing size and written as the node attribute ``attribute``, a tuple of
   numbers for a node in several; a node in none (all its communities smaller than ``min_size``)
   is a community of its own.

   :param net: A :class:`~graphoskop.Network`, modified in place.
   :param attribute: Name of the node attribute written.
   :param epsilon: Smallest shared fraction of the smaller community for two communities to merge,
      between 0 (excluded) and 1: the smaller, the more merges (fewer, larger communities).
   :param min_size: Smallest number of nodes of a community (the ego included).
   :param weight: Name of the edge attribute used as edge weight in the label propagations, or
      ``None`` for unit weights.
   :param max_iter: Maximum number of sweeps of each label propagation.
   :param seed: Seed of the random numbers, or ``None``.
   :return: A dictionary with the number of ``communities`` (of at least ``min_size`` nodes), the
      number of ``local`` communities of at least ``min_size`` nodes found in the ego networks
      (before merging), and the number of ``overlapping`` nodes.
   :raises ValueError: If ``epsilon`` is not in :math:`(0, 1]`, ``min_size`` is below 1, or
      weights are negative.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      res = gsk.demon_communities(net, seed = 0)
      res["communities"], res["overlapping"]
      gsk.demon_communities(net, epsilon = 0.5, min_size = 5, attribute = "demon")   # fewer merges
   """
   from ._demon import _ego_labels, _merge
   if not 0 < epsilon <= 1:
      raise ValueError("epsilon must be in (0, 1]")
   if min_size < 1:
      raise ValueError("min_size must be at least 1")
   N = int(net.node_offsets[-1])
   S, T, W = [], [], []
   for c in range(len(net.components)):
      ei = net.components[c].edge_index.cpu().numpy()
      S.append(ei[0] + int(net.node_offsets[c]))
      T.append(ei[1] + int(net.node_offsets[c]))
      W.append(np.ones(ei.shape[1]) if weight is None else net._numeric_edge(weight, c).double().cpu().numpy())
   s, t, w = np.concatenate(S), np.concatenate(T), np.concatenate(W)
   if (w < 0).any():
      raise ValueError("weights must be non negative")
   keep = s != t
   s, t, w = s[keep], t[keep], w[keep]
   if net.directed:   # both ways
      s, t, w = np.r_[s, t], np.r_[t, s], np.r_[w, w]
   key, inv = np.unique(s * N + t, return_inverse = True)   # parallel edges merged, CSR order
   w = np.bincount(inv, w) if len(key) else np.zeros(0)
   s, t = key // N, key % N
   ptr = np.r_[0, np.cumsum(np.bincount(s, minlength = N))]
   lab = _ego_labels(N, ptr, t, w, max_iter, int(np.random.default_rng(seed).integers(2 ** 31)))
   # local communities: (ego, label) -> its neighbors, plus the ego, in order of ego
   ckey, cid = np.unique(s * N + lab, return_inverse = True)
   rows, cols = np.r_[cid, np.arange(len(ckey))], np.r_[t, ckey // N]
   size = np.bincount(rows, minlength = len(ckey))
   ok = size >= min_size
   order = np.argsort(rows, kind = "stable")
   rows, cols = rows[order], cols[order]
   sel = ok[rows]
   new = np.cumsum(ok) - 1
   rows, cols = new[rows[sel]], cols[sel]
   local = int(ok.sum())
   cptr = np.r_[0, np.cumsum(np.bincount(rows, minlength = local))]
   node, code, K = _merge(N, cptr, cols.astype(np.int64), float(epsilon))
   alone = np.setdiff1d(np.arange(N), node)
   _, multi = _write_cover(net, attribute, np.r_[node, alone], np.r_[code, K + np.arange(len(alone))])
   return {"communities": int(K), "local": local, "overlapping": multi}


@_higher_order()
def oslom_communities(net, attribute = "community", tolerance = 0.1, iterations = 10, runs = 10, union = 0.7, seed = None):
   """
   OSLOM (Lancichinetti, Radicchi, Ramasco and Fortunato, 2011): overlapping communities that are
   statistically significant, by the order statistics of a local fitness score. Nodes not
   significantly linked to any community are left out of all (homeless).

   **Significance.** For a node :math:`i` outside a cluster :math:`C`, with :math:`k_i` links of
   which :math:`k_i^{in}` to :math:`C`, the probability of having at least :math:`k_i^{in}` links
   to :math:`C` in the configuration model (the hypergeometric approximation of the paper:
   :math:`k_i` stubs drawn among those not inside :math:`C`, of which :math:`m_C^{out}` leave
   :math:`C`) gives its r-score, drawn uniformly between the probabilities of more than and of at
   least :math:`k_i^{in}` links (a random tie break of the discrete distribution). Under the null
   model the r-scores of the :math:`N - n_C` external nodes are uniform, so the :math:`q`-th
   smallest has the distribution :math:`\\Omega_q(r) = P(\\mathrm{Bin}(N - n_C, r) \\geq q)`; the
   :math:`q` best nodes are significantly linked to :math:`C` if :math:`\\phi(\\Omega_q(r_q)) < P`
   (``tolerance``), with :math:`\\phi` the distribution of :math:`\\min_q \\Omega_q(r_q)` under the
   null model (the correction for looking at the best :math:`q`; simulated once, for :math:`q` up
   to 200).

   **Single cluster analysis.** External nodes are added while significant (for the first
   :math:`q`, from 1 up, with :math:`\\phi(\\Omega_q(r_q)) < P`: the best node alone, or else the
   best two together, and so on), then the internal node with the worst r-score (as an external
   node of the rest) is removed while not significant, alternately until nothing changes. Since
   the scores are random, the analysis is repeated ``runs`` times (the paper uses 100): the
   cluster is significant if more than half the outcomes are non-empty, and keeps the nodes in
   more than half of them.

   **Cover.** As in the OSLOM software, which explores the network with a Louvain-like method,
   the candidate clusters are the communities of the Leiden algorithm (:func:`leiden_communities`,
   a different seed in each of the ``iterations``), each going through the single cluster
   analysis: it can lose nodes, and gain nodes of other clusters (the overlaps). (Growing clusters
   from single random nodes, as also described in the paper, let clusters reaching nodes shared
   with another community absorb that community too.) A cluster must be minimal: its own
   subnetwork is explored the same way, and if two or more significant subclusters cover more than
   ``union`` of it (the paper's :math:`P_2 = 0.7`), they replace it. Then clusters sharing more than
   half of the smaller one are resolved, most overlapping first: they merge unless their union
   has significant subclusters covering more than ``union`` of it, in which case the larger is
   kept. Not implemented: the hierarchical levels (clusters of clusters) and weighted or directed
   networks (directions are ignored, weights not used).

   The single cluster analyses run in a compiled kernel (Numba) on the CPU; the result depends on
   the seed. Communities are numbered by decreasing size and written as the node attribute
   ``attribute``, a tuple of numbers for a node in several; a homeless node is a community of its
   own.

   :param net: A :class:`~graphoskop.Network`, modified in place.
   :param attribute: Name of the node attribute written.
   :param tolerance: Significance threshold :math:`P`: larger keeps more and larger clusters.
   :param iterations: Number of runs of the Leiden algorithm giving candidate clusters.
   :param runs: Number of repetitions of every single cluster analysis.
   :param union: Fraction of a cluster that significant subclusters must cover to replace it.
   :param seed: Seed of the random numbers, or ``None``.
   :return: A dictionary with the number of significant ``communities``, the number of
      ``homeless`` nodes and the number of ``overlapping`` nodes.
   :raises ValueError: If ``tolerance`` or ``union`` are not in :math:`(0, 1)`, or ``iterations``
      or ``runs`` are below 1.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      res = gsk.oslom_communities(net, seed = 0)
      res["communities"], res["homeless"], res["overlapping"]
      gsk.oslom_communities(net, tolerance = 0.5, runs = 100, attribute = "oslom")
   """
   from ._oslom import _clean, _phi_table
   from .network import load_network
   if not 0 < tolerance < 1 or not 0 < union < 1:
      raise ValueError("tolerance and union must be between 0 and 1")
   if iterations < 1 or runs < 1:
      raise ValueError("iterations and runs must be at least 1")
   N = int(net.node_offsets[-1])
   S, T = [], []
   for c in range(len(net.components)):
      ei = net.components[c].edge_index.cpu().numpy()
      S.append(ei[0] + int(net.node_offsets[c]))
      T.append(ei[1] + int(net.node_offsets[c]))
   s, t = np.concatenate(S), np.concatenate(T)
   keep = s != t
   x, y = np.minimum(s, t)[keep], np.maximum(s, t)[keep]
   key = np.unique(x * N + y)   # simple, undirected
   grid, samples = _phi_table()
   rng = np.random.default_rng(seed)
   from scipy.special import gammaln
   lf = gammaln(np.arange(2 * len(key) + 2) + 1.)   # log k!

   class Graph:   # CSR of a simple undirected graph, with work arrays
      def __init__(self, n, a, b):
         both = np.r_[a, b]
         order = np.argsort(both * n + np.r_[b, a], kind = "stable")
         self.n, self.a, self.b = n, a, b
         self.nbr = np.r_[b, a][order].astype(np.int64)
         self.deg = np.bincount(both, minlength = n).astype(np.int64)
         self.ptr = np.r_[0, np.cumsum(self.deg)].astype(np.int64)
         self.work = (np.zeros(n, dtype = np.bool_), np.zeros(n, dtype = np.int64), np.zeros(n, dtype = np.bool_))

      def clean(self, nodes):
         return _clean(self.ptr, self.nbr, self.deg, int(2 * len(self.a)), lf, np.asarray(nodes, dtype = np.int64), 0, float(tolerance), int(runs), grid, samples, int(rng.integers(2 ** 31)), *self.work)

      def modules(self):   # Leiden communities, as lists of nodes
         if not len(self.a):
            return []
         L = load_network(pd.DataFrame({"source": self.a, "target": self.b}), pd.DataFrame({"node": np.arange(self.n)}))
         leiden_communities(L, attribute = "m", seed = int(rng.integers(2 ** 31)))
         ids = np.concatenate(L.node_ids).astype(np.int64)
         lab = np.concatenate([np.asarray(v.cpu() if torch.is_tensor(v) else v) for v in L.get_node_attribute("m", None)]).astype(np.int64)
         return [ids[lab == g] for g in np.unique(lab)]

      def sub(self, nodes):   # induced subgraph, with the map back
         nodes = np.sort(np.asarray(nodes, dtype = np.int64))
         loc = np.full(self.n, -1)
         loc[nodes] = np.arange(len(nodes))
         ok = (loc[self.a] >= 0) & (loc[self.b] >= 0)
         return Graph(len(nodes), loc[self.a[ok]], loc[self.b[ok]]), nodes

      def significant(self, iters):   # cleaned candidate clusters, distinct
         out = {}
         for it in range(iters):
            for m in self.modules():
               C = self.clean(m)
               if len(C) > 1:
                  out.setdefault(tuple(np.sort(C).tolist()), None)
         return [np.array(c) for c in out]

   G = Graph(N, key // N, key % N)

   cache = {}

   def split(C):   # significant subclusters covering more than `union` of C, or None (cached)
      k = tuple(np.sort(C).tolist())
      if k not in cache:
         cache[k] = _split(C)
      return cache[k]

   def _split(C):
      H, nodes = G.sub(C)
      parts = [p for p in H.significant(1) if len(p) < len(C)]
      cover = np.unique(np.concatenate(parts)) if parts else np.zeros(0)
      return [nodes[p] for p in parts] if len(parts) > 1 and len(cover) > union * len(C) else None

   def minimal(C, depth = 0):
      parts = split(C) if depth < 5 else None
      return [C] if parts is None else [q for p in parts for q in minimal(p, depth + 1)]

   clusters = {}
   for C in G.significant(iterations):
      for q in minimal(C):
         clusters.setdefault(tuple(np.sort(q).tolist()), None)
   clusters = [set(c) for c in clusters]
   while True:   # resolve similar clusters, most overlapping first
      best = None
      member = {}
      for i, c in enumerate(clusters):
         for v in c:
            member.setdefault(v, []).append(i)
      shared = {}
      for v, cs in member.items():
         for p in range(len(cs)):
            for q in range(p + 1, len(cs)):
               shared[(cs[p], cs[q])] = shared.get((cs[p], cs[q]), 0) + 1
      for (i, j), k in shared.items():
         o = k / min(len(clusters[i]), len(clusters[j]))
         if o > 0.5 and (best is None or o > best[0]):
            best = (o, i, j)
      if best is None:
         break
      _, i, j = best
      U = clusters[i] | clusters[j]
      new = max(clusters[i], clusters[j], key = len) if split(np.array(sorted(U))) is not None else U
      clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
      if all(new != c for c in clusters):
         clusters.append(new)
   node = np.array([v for c in clusters for v in c], dtype = np.int64)
   code = np.array([k for k, c in enumerate(clusters) for _ in c], dtype = np.int64)
   alone = np.setdiff1d(np.arange(N), node)
   _, multi = _write_cover(net, attribute, np.r_[node, alone], np.r_[code, len(clusters) + np.arange(len(alone))])
   return {"communities": len(clusters), "homeless": int(len(alone)), "overlapping": multi}
