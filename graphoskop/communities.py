"""
Communities: discovery of communities, quality of partitions of the nodes into groups, and
maximal cliques.

:func:`block_model_communities` finds communities by fitting the stochastic blockmodels of the
generators (nested, degree-corrected, plain, planted partition, mixed membership) to a network.

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
