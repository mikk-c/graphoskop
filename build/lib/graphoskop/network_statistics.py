"""
Simple network statistics: connected components, degree of nodes, degree distributions, average
degree, power law fits of the degree distribution, degree centralization, density,
reciprocity, average shortest path length, clustering coefficients,
assortativity.

Nodes are given by their input id (a key of :attr:`~graphoskop.Network.node_map`) or by their
internal id ``(component, index)``; a tuple that is both raises :class:`ValueError`. Parallel
edges are counted separately. A self loop adds 1 to the out- and the in-degree of its node, so 2
to its degree in undirected networks.

In multilayer networks only intra-layer edges count: couplings and other interlayer edges never
do. The degree of a copy ``(entity, layer)`` is its degree within its layer; an entity can also be
given on its own, and its degree is then the sum over its copies (the overlapping degree, parallel
edges in different layers counting separately). The number of distinct neighbors of an entity
across layers is its degree in :func:`~graphoskop.aggregate_network`.

Clustering coefficients count the triangles around nodes. Edges are taken as unweighted, parallel
edges count once and self loops are ignored. For node :math:`i`, with :math:`A` the adjacency
matrix, :math:`d_i` its total (in plus out) degree and :math:`d^\\leftrightarrow_i` its number of
reciprocated edges, the local clustering coefficient is (Fagiolo, 2007)

.. math::

   C_i = \\frac{[(A + A^T)^3]_{ii}}{2 \\, [d_i (d_i - 1) - 2 d^\\leftrightarrow_i]},

the fraction of the possible directed triangles around :math:`i` that exist. In undirected
networks it is the usual :math:`C_i = 2 T_i / (k_i (k_i - 1))`, with :math:`T_i` the number of
triangles through :math:`i` and :math:`k_i` its number of neighbors. The global clustering
coefficient (transitivity) is the sum of the numerators over the sum of the denominators: in
undirected networks, three times the number of triangles over the number of connected triples.
The average clustering coefficient is the mean of the local ones.

Triangles are found without forming matrix powers: the pairs of adjacent nodes are oriented from
lower to higher degree, and each triangle is found once by expanding the 2-paths along the
orientation (the work grows with :math:`m^{3/2}` at most, even with hubs). Paths are expanded in
batches of ``batch_size``, so memory stays bounded (about 50 bytes per path), and all the work runs
on the device of the network.

In multilayer networks only intra-layer edges count, so triangles are within layers: the clustering
of a copy is its clustering in its layer, and global and average clustering are given per layer.
"""

import warnings

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from scipy import integrate, optimize, special, stats
from scipy.sparse.csgraph import connected_components

from .shortest_paths import _source_stats


def _strong_labels(net, component):
   """
   Strongly connected component of every node of a (weakly connected) component.

   :param net: Network.
   :param component: Component index.
   :return: ``(number, labels)``, as :func:`scipy.sparse.csgraph.connected_components`.
   """
   data = net.components[component]
   s, t = data.edge_index.cpu().numpy()
   n = data.num_nodes
   return connected_components(sp.csr_matrix((np.ones(len(s)), (s, t)), shape = (n, n)), directed = True, connection = "strong")


def _check_type(net, type):
   """
   Whether strongly connected components are asked for (in undirected networks they are the
   weakly connected ones).

   :raises ValueError: If ``type`` is not valid.
   """
   if type not in ("weak", "strong"):
      raise ValueError(f"unknown type of connected components: {type}")
   return type == "strong" and net.directed


def connected_components_count(net, type = "weak"):
   """
   Number of connected components.

   Weakly connected components (``type = "weak"``) are the components the network is split into
   (see :class:`~graphoskop.Network`), isolated nodes included (each is a component). Strongly
   connected components (``type = "strong"``, for directed networks) are the largest sets of nodes
   with a directed path from every node to every other; each weakly connected component splits into
   one or more of them. In undirected networks the two coincide. In multilayer networks the
   components are those of the supra-graph, interlayer edges included.

   :param net: A :class:`~graphoskop.Network`.
   :param type: ``"weak"`` or ``"strong"``.
   :return: Number of components.
   :raises ValueError: If ``type`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("follows.tsv", directed = True)
      gsk.connected_components_count(net)                    # weakly connected
      gsk.connected_components_count(net, type = "strong")
   """
   if not _check_type(net, type):
      return len(net.components)
   return sum(_strong_labels(net, c)[0] for c in range(len(net.components)))


def connected_components_nodes(net, type = "weak"):
   """
   Nodes of each connected component (see :func:`connected_components_count` for the types).

   :param net: A :class:`~graphoskop.Network`.
   :param type: ``"weak"`` or ``"strong"``.
   :return: List of arrays of input node ids, one per component, largest first (for weakly
      connected components, the same order as the components of the network, and the same as
      :attr:`~graphoskop.Network.node_ids`).
   :raises ValueError: If ``type`` is not valid.

   Example:

   .. code-block:: python

      sccs = gsk.connected_components_nodes(net, type = "strong")
      core = gsk.subgraph(net, sccs[0])            # the largest strongly connected component
   """
   if not _check_type(net, type):
      return [ids.copy() for ids in net.node_ids]
   out = []
   for c in range(len(net.components)):
      k, labels = _strong_labels(net, c)
      order = np.argsort(labels, kind = "stable")
      out += np.split(net.node_ids[c][order], np.cumsum(np.bincount(labels, minlength = k))[:-1])
   return sorted(out, key = len, reverse = True)


def _degrees(net, component, direction, weight = None):
   """
   Degree of every node of a component, counting intra-layer edges only in multilayer networks.

   :param net: Network.
   :param component: Component index.
   :param direction: ``"out"``, ``"in"`` or ``"total"`` (ignored in undirected networks).
   :param weight: Edge attribute summed instead of counting edges, or ``None``.
   :return: Vector of length ``n`` (integer, float with ``weight``), on the device of the component.
   :raises ValueError: If ``direction`` is not valid.
   :raises KeyError: If there is no such edge attribute.
   """
   if direction not in ("out", "in", "total"):
      raise ValueError(f"unknown direction: {direction}")
   data = net.components[component]
   s, t = data.edge_index
   w = None if weight is None else net.get_edge_attribute(weight, component).double()
   if net.layer_labels is not None:
      intra = ~net.interlayer_mask(component)
      s, t, w = s[intra], t[intra], None if w is None else w[intra]
   n = data.num_nodes
   count = lambda x: torch.bincount(x, w, minlength = n)
   if not net.directed:   # both directions stored, self loops once
      loop = s == t
      return count(s) + torch.bincount(s[loop], None if w is None else w[loop], minlength = n)
   if direction == "out":
      return count(s)
   if direction == "in":
      return count(t)
   return count(s) + count(t)


def _skeleton(s, t, n):
   """
   Undirected skeleton of a set of directed edges, oriented from lower to higher rank.

   The rank of a node is its number of neighbors (ties broken by index), so that orienting each
   pair of adjacent nodes towards the higher rank bounds the out-degrees, and the triangle
   enumeration of :func:`_paths` takes at most :math:`O(m^{3/2})` work.

   :param s: Sources, distinct ``(s, t)`` pairs without self loops.
   :param t: Targets.
   :param n: Number of nodes.
   :return: ``(src, dst, slot, along)``: the skeleton edges, sorted by ``src * n + dst``, the
      position of the skeleton edge of each input edge, and whether the input edge goes along it
      (from ``src`` to ``dst``).
   """
   pair, inv = torch.unique(torch.minimum(s, t) * n + torch.maximum(s, t), return_inverse = True)
   lo, hi = pair // n, pair % n
   rank = torch.bincount(torch.cat([lo, hi]), minlength = n) * n + torch.arange(n, device = s.device)
   up = rank[lo] < rank[hi]
   src, dst = torch.where(up, lo, hi), torch.where(up, hi, lo)
   order = torch.argsort(src * n + dst)
   pos = torch.empty_like(order)
   pos[order] = torch.arange(order.shape[0], device = s.device)
   return src[order], dst[order], pos[inv], rank[s] < rank[t]


def _paths(s, t, n, batch_size):
   """
   Triangles of an oriented skeleton, in batches.

   The skeleton has one edge per pair of adjacent nodes, oriented from lower to higher rank, so
   each triangle ``a -> b -> c`` (closed by ``a -> c``) is found once, by expanding the 2-paths
   along the orientation. Paths are expanded in batches of at most ``batch_size`` (unless a single
   edge needs more).

   :param s: Sources of the skeleton edges, sorted with the targets by ``s * n + t``.
   :param t: Targets.
   :param n: Number of nodes.
   :param batch_size: Paths expanded at once.
   :return: Generator of ``(ab, bc, ac)``, positions of the three skeleton edges of each triangle.
   """
   key = s * n + t
   m = key.shape[0]
   ptr = torch.zeros(n + 1, dtype = torch.long, device = s.device)
   ptr[1:] = torch.cumsum(torch.bincount(s, minlength = n), 0)
   work = ptr[t + 1] - ptr[t]   # out-edges of the middle node of the paths starting with each edge
   cum = torch.cumsum(work, 0)
   start, done = 0, 0
   while start < m:
      end = max(int(torch.searchsorted(cum, done + batch_size, right = True)), start + 1)
      first = torch.arange(start, end, device = s.device)
      k = work[first]
      first = torch.repeat_interleave(first, k)
      second = ptr[t[first]] + torch.arange(first.shape[0], device = s.device) - torch.repeat_interleave(torch.cumsum(k, 0) - k, k)
      q = s[first] * n + t[second]
      third = torch.searchsorted(key, q).clamp(max = m - 1)
      hit = key[third] == q
      yield first[hit], second[hit], third[hit]
      done = int(cum[end - 1])
      start = end


def _table(values, distribution):
   """
   Degree distribution of some values.

   :param values: Degrees.
   :param distribution: ``"count"``, ``"pdf"`` or ``"ccdf"``.
   :return: :class:`pandas.DataFrame` with columns ``degree`` and ``distribution``.
   """
   values, counts = np.unique(values, return_counts = True)
   if distribution == "count":
      return pd.DataFrame({"degree": values, "count": counts})
   p = counts / counts.sum()
   return pd.DataFrame({"degree": values, distribution: p if distribution == "pdf" else p[::-1].cumsum()[::-1]})


def degree(net, node = None, distribution = "count", direction = "out", component = None, by = "layer", weight = None):
   """
   Degree of nodes, or degree distribution.

   With ``weight``, the weighted degree (strength): each edge adds the value of that edge
   attribute instead of 1 (in undirected networks a self loop adds twice its value, as it adds 2
   to the degree); NaN values make the degree NaN. Values are then floats, and the distribution
   lists every distinct value.

   * ``node`` given: the degree of that node, as an ``int`` (a ``float`` with ``weight``).
   * ``node`` a list (or array) of nodes: a :class:`pandas.DataFrame` with columns ``node`` (as
     given) and ``degree``.
   * ``node = None``: the degree distribution, as a :class:`pandas.DataFrame` with column
     ``degree`` (the values that occur, increasing) and a column named as ``distribution``:
     ``"count"`` (number of nodes with that degree), ``"pdf"`` (fraction of nodes with that
     degree) or ``"ccdf"`` (fraction of nodes with that degree or higher).

   In multilayer networks (see the module description), ``by`` chooses the distribution:
   ``"layer"`` gives the distribution of each layer, over its copies, in one table with a first
   column ``layer`` (pdf and ccdf normalized within each layer); ``"entity"`` gives the
   distribution of the degrees of the entities (sums over their copies). Pooling the copies of all
   layers into one distribution is not offered: it would mix layers, and count each entity once per
   layer.

   :param net: A :class:`~graphoskop.Network`.
   :param node: Node (input id, internal id ``(component, index)`` or, in multilayer networks,
      entity), list of nodes, or ``None``.
   :param distribution: ``"count"``, ``"pdf"`` or ``"ccdf"``; used when ``node`` is ``None``.
   :param direction: In directed networks, ``"out"``, ``"in"`` or ``"total"`` (in plus out)
      degree; ignored in undirected networks.
   :param component: For the distribution, index of the component (0 is the largest), or
      ``None`` for the whole network; ignored when ``node`` is given. With ``by = "entity"``,
      only the copies in that component are summed.
   :param by: In multilayer networks, ``"layer"`` or ``"entity"``; ignored otherwise.
   :param weight: Name of a numeric edge attribute for the weighted degree, or ``None`` for the
      number of edges.
   :return: Degree, or :class:`pandas.DataFrame`.
   :raises KeyError: If a node or the edge attribute does not exist.
   :raises ValueError: If ``distribution``, ``direction`` or ``by`` is not valid, or if a node is
      both an input id and an internal id.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", directed = True)
      k = gsk.degree(net, "alice")                                # out-degree of a node
      k = gsk.degree(net, (0, 5), direction = "in")               # by internal id
      table = gsk.degree(net, ["alice", "bob"])                   # several nodes
      ccdf = gsk.degree(net, distribution = "ccdf", direction = "total")
      s = gsk.degree(net, "alice", weight = "weight")             # weighted out-degree (strength)

      net = gsk.load_network("multiplex.tsv", multilayer = True, coupling = "all")
      k = gsk.degree(net, ("alice", "work"))                      # degree in layer "work"
      k = gsk.degree(net, "alice")                                # summed over the layers
      pdf = gsk.degree(net, distribution = "pdf")                 # one distribution per layer
      pdf = gsk.degree(net, distribution = "pdf", by = "entity")
   """
   ml = net.layer_labels is not None
   if node is not None and not isinstance(node, (list, np.ndarray, pd.Series, pd.Index)):
      cast = int if weight is None else float
      return cast(sum(_degrees(net, c, direction, weight)[i].item() for c, i in net._nodes(node)))
   if node is not None:
      ids = [net._nodes(x) for x in node]
      deg = {c: _degrees(net, c, direction, weight).cpu().numpy() for c in set(c for x in ids for c, _ in x)}
      cast = int if weight is None else float
      return pd.DataFrame({"node": list(node), "degree": [cast(sum(deg[c][i] for c, i in x)) for x in ids]})
   if distribution not in ("count", "pdf", "ccdf"):
      raise ValueError(f"unknown distribution: {distribution}")
   if ml and by not in ("layer", "entity"):
      raise ValueError(f"unknown by: {by}")
   comps = range(len(net.components)) if component is None else [component]
   values = np.concatenate([_degrees(net, c, direction, weight).cpu().numpy() for c in comps])
   if not ml:
      return _table(values, distribution)
   codes = np.concatenate([getattr(net.components[c], "layer" if by == "layer" else "entity").cpu().numpy() for c in comps])
   if by == "entity":
      sums = np.bincount(codes, values)[np.bincount(codes) > 0]
      return _table(sums if weight is not None else sums.astype(np.int64), distribution)
   tables = []
   for l, label in enumerate(net.layer_labels):
      if (codes == l).any():
         tables.append(_table(values[codes == l], distribution))
         tables[-1].insert(0, "layer", label)
   return pd.concat(tables, ignore_index = True)


def average_degree(net, direction = "out", component = None, by = "layer"):
   """
   Average degree: the mean of the degrees of all nodes, as given by :func:`degree`.

   This is :math:`2m / n` in undirected networks (each edge adds 1 to the degree of both its
   nodes) and :math:`m / n` in directed networks for the out- or the in-degree (each edge adds 1 to
   one out-degree and one in-degree, so the two averages are equal), :math:`2m / n` for the total
   degree. Here :math:`m` counts edges as :meth:`~graphoskop.Network.number_of_edges` does
   (parallel edges and self loops included) and :math:`n` counts all nodes, isolated nodes
   included. NaN if there are no nodes.

   In multilayer networks only intra-layer edges count, and ``by`` chooses the average:
   ``"layer"`` gives the average degree of the copies of each layer (:math:`2 m_l / n_l` in
   undirected networks), ``"entity"`` the average of the degrees of the entities (sums over their
   copies, as in :func:`degree`).

   :param net: A :class:`~graphoskop.Network`.
   :param direction: In directed networks, ``"out"``, ``"in"`` or ``"total"``; ignored in
      undirected networks.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param by: In multilayer networks, ``"layer"`` or ``"entity"``; ignored otherwise.
   :return: A float; with ``by = "layer"`` in multilayer networks, a :class:`pandas.DataFrame`
      with columns ``layer`` and ``average_degree``, one row per layer with copies.
   :raises ValueError: If ``direction`` or ``by`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.average_degree(net)                          # 2 m / n
      net = gsk.load_network("edges.tsv", directed = True)
      gsk.average_degree(net)                          # m / n, out-degree (equal to in-degree)
      gsk.average_degree(net, direction = "total")     # 2 m / n
   """
   ml = net.layer_labels is not None
   if ml and by not in ("layer", "entity"):
      raise ValueError(f"unknown by: {by}")
   comps = range(len(net.components)) if component is None else [component]
   values = np.concatenate([_degrees(net, c, direction).cpu().numpy() for c in comps]).astype(np.float64)
   if not ml:
      return float(values.mean()) if len(values) else float("nan")
   codes = np.concatenate([getattr(net.components[c], by).cpu().numpy() for c in comps])
   if by == "entity":
      sums = np.bincount(codes, values)[np.bincount(codes) > 0]
      return float(sums.mean()) if len(sums) else float("nan")
   rows = [(label, float(values[codes == l].mean())) for l, label in enumerate(net.layer_labels) if (codes == l).any()]
   return pd.DataFrame(rows, columns = ["layer", "average_degree"])


def _pl_fit(k, kmin):
   """
   Discrete power law :math:`p(k) = k^{-\\alpha} / \\zeta(\\alpha, k_{min})` fitted by maximum likelihood.

   :param k: Degrees, all at least ``kmin``.
   :param kmin: Smallest degree of the tail.
   :return: ``(alpha, log-likelihood of each degree)``.
   """
   lk = np.log(k)
   a = optimize.minimize_scalar(lambda a: a * lk.sum() + len(k) * np.log(special.zeta(a, kmin)), bounds = (1 + 1e-6, 20), method = "bounded", options = {"xatol": 1e-10}).x
   return a, -a * lk - np.log(special.zeta(a, kmin))


def _ks(k, kmin, a):
   """
   Kolmogorov-Smirnov distance between the degrees of the tail and a fitted discrete power law.
   """
   x, c = np.unique(k, return_counts = True)
   empirical = np.cumsum(c) / len(k)
   fitted = 1 - special.zeta(a, x + 1) / special.zeta(a, kmin)
   return float(np.abs(empirical - fitted).max())


def _cutoff_norm(a, lam, kmin):
   """
   :math:`\\sum_{j \\ge k_{min}} j^{-\\alpha} e^{-\\lambda j}`: exact for the first 10,000 terms, the
   rest by integration (the terms are smooth there).
   """
   if lam <= 0:
      return special.zeta(a, kmin) if a > 1 else np.inf
   j = np.arange(kmin, kmin + 10000, dtype = np.float64)
   head = np.exp(-a * np.log(j) - lam * j).sum()
   with warnings.catch_warnings():   # slow convergence only where the likelihood is very poor anyway
      warnings.simplefilter("ignore", integrate.IntegrationWarning)
      tail = integrate.quad(lambda x: np.exp(-a * np.log(x) - lam * x), kmin + 9999.5, np.inf)[0]
   return head + tail


def _exponential(k, kmin):
   """
   Discrete exponential (geometric above ``kmin``), :math:`p(k) = (1 - e^{-\\lambda}) e^{-\\lambda (k - k_{min})}`,
   fitted by maximum likelihood (closed form).

   :return: ``(lambda, log-likelihood of each degree)``.
   """
   lam = np.log1p(1 / max(k.mean() - kmin, 1e-12))
   return lam, np.log1p(-np.exp(-lam)) - lam * (k - kmin)


def _alternatives(k, kmin, a_pl):
   """
   Per-degree log-likelihoods of the discrete exponential, lognormal and power law with exponential
   cutoff fitted by maximum likelihood to the tail, and the cutoff fit.

   :return: ``(exponential, lognormal, cutoff, (alpha, lambda))`` (log-likelihoods, then the
      parameters of the cutoff fit).
   """
   exponential = _exponential(k, kmin)[1]
   # lognormal, mass of [k - 1/2, k + 1/2) (rounding), normalized above kmin - 1/2
   def log_mass(z0, z1):   # log(Phi(z1) - Phi(z0)), from the side where it is not 1 - tiny
      with np.errstate(divide = "ignore", invalid = "ignore"):
         right = stats.norm.logsf(z0) + np.log1p(-np.exp(stats.norm.logsf(z1) - stats.norm.logsf(z0)))
         left = stats.norm.logcdf(z1) + np.log1p(-np.exp(stats.norm.logcdf(z0) - stats.norm.logcdf(z1)))
      return np.where(z0 > 0, right, left)
   def ln_ll(p):
      mu, sigma = p[0], np.exp(p[1])
      z = lambda x: (np.log(x) - mu) / sigma
      return log_mass(z(k - 0.5), z(k + 0.5)) - stats.norm.logsf(z(kmin - 0.5))
   lk = np.log(k)
   starts = ([lk.mean(), np.log(max(lk.std(), 0.1))], [np.log(kmin), 0.], [-2., 1.], [-10., 1.5])
   def nll(p):
      v = -ln_ll(p).sum()
      return v if np.isfinite(v) else np.inf
   p = min((optimize.minimize(nll, x0, method = "Nelder-Mead", options = {"xatol": 1e-8, "fatol": 1e-9, "maxiter": 2000}) for x0 in starts), key = lambda r: r.fun).x
   lognormal = ln_ll(p)
   # power law with exponential cutoff
   f = lambda p: p[0] * lk.sum() + p[1] * k.sum() + len(k) * np.log(_cutoff_norm(p[0], p[1], kmin))
   best = min((optimize.minimize(f, x0, method = "L-BFGS-B", bounds = [(0, 20), (0, 10)]) for x0 in ([a_pl, 1e-4], [max(a_pl - 1, 0.1), 1 / k.mean()])), key = lambda r: r.fun)
   a, lam = best.x
   cutoff = -a * lk - lam * k - np.log(_cutoff_norm(a, lam, kmin))
   return exponential, lognormal, cutoff, (a, lam)


def _vuong(l1, l2, nested = False):
   """
   Log-likelihood ratio test (Vuong, 1989; Clauset et al., 2009, appendix C).

   :return: ``(R, p)``: the log-likelihood ratio (positive favors the first model) and its p-value;
      for nested models, ``p`` follows Wilks' theorem with one degree of freedom.
   """
   d = l1 - l2
   R = float(d.sum())
   if nested:
      return R, float(special.erfc(np.sqrt(abs(R))))
   sd = d.std()
   return R, float(special.erfc(abs(R) / (np.sqrt(2 * len(d)) * sd))) if sd > 0 else 1.


def degree_power_law(net, direction = "out", component = None, layer = None, k_min = None, significance = 0.1, ccdf = False):
   """
   Fit of a power law to the degree distribution, with the tests needed to decide whether it is
   one (Clauset, Shalizi and Newman, 2009).

   Degrees are integers, so all distributions are discrete, fitted by maximum likelihood to the
   tail :math:`k \\ge k_{min}` (degree 0 is left out):

   1. **Regime.** :math:`k_{min}` (the shift) is the smallest degree from which the distribution is
      closest to a power law: for each candidate (each degree value leaving at least 10 nodes in the
      tail), the exponent is fitted and the Kolmogorov-Smirnov distance between the tail and the
      fit is computed; the minimum wins. ``k_min`` fixes it instead. A shift exists when
      :math:`k_{min}` is above the smallest degree. The tests below compare models on this tail.
   2. **Exponential.** Log-likelihood ratio test (Vuong) against a discrete exponential, over all
      positive degrees and over the tail. If the exponential is significantly better over all
      degrees and the tail does not significantly favor the power law, the distribution is not a
      power law: no exponent is given (NaN) and a warning is issued, and the best fit returned is
      the exponential over all positive degrees. Both tests are needed: on
      exponential data, the search of step 1 can settle on a short far tail where a steep power
      law looks acceptable and the test lacks power; while a power law tail above a different body
      (a shift) must not be refused because of its body. If only the tail test is inconclusive, a
      warning is issued and the fit goes on.
   3. **Lognormal.** Same test against a discrete lognormal. Unless the power law is significantly
      better, a warning is issued (a lognormal and a power law are hard to tell apart on finite
      data), and the fit goes on.
   4. **Cutoff.** Nested likelihood ratio test against a power law with exponential cutoff,
      :math:`p(k) \\propto k^{-\\alpha} e^{-\\lambda k}`. If the cutoff is significantly better, the
      exponent and :math:`\\lambda` of that fit are returned; otherwise the pure power law exponent,
      with :math:`\\lambda = 0`.

   A test is significant when its p-value is below ``significance`` (0.1 as in Clauset et al.).
   The goodness-of-fit p-value of Clauset et al. (by bootstrap) is not computed. Weighted degrees
   are not integers, so they are not supported.

   :param net: A :class:`~graphoskop.Network`.
   :param direction: In directed networks, ``"out"``, ``"in"`` or ``"total"`` degree.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param layer: In multilayer networks, a layer (its copies' degrees), or ``None`` for the degrees
      of the entities (sums over their copies, as :func:`degree` with ``by = "entity"``).
   :param k_min: Smallest degree of the tail, or ``None`` to choose it as above.
   :param significance: Threshold on the p-values of the tests.
   :param ccdf: Also return the fitted complementary cumulative distribution (see below).
   :return: Dictionary describing the best fit and the tests behind it:

      * ``verdict``: ``"power law"``, ``"power law with exponential cutoff"`` or ``"not a power law"``;
      * ``model``: the best fit, ``"power law"``, ``"power law with exponential cutoff"`` or
        ``"exponential"`` (when the verdict is ``"not a power law"``);
      * ``exponent``: :math:`\\alpha` (NaN for the exponential); ``cutoff``: :math:`\\lambda` of the
        power law with cutoff (0 otherwise); ``rate``: :math:`\\lambda` of the exponential (only then);
      * ``k_min``: smallest degree of the fitted range (for the exponential, the smallest positive
        degree);
      * ``normalization``: :math:`Z`, the sum of the unnormalized model over the fitted range
        (:math:`\\zeta(\\alpha, k_{min})` for the power law,
        :math:`\\sum_{j \\ge k_{min}} j^{-\\alpha} e^{-\\lambda j}` with cutoff,
        :math:`e^{-\\lambda k_{min}} / (1 - e^{-\\lambda})` for the exponential);
      * ``tail_fraction``: :math:`f`, the fraction of the nodes considered (degree 0 included) with
        degree at least ``k_min``;
      * ``formula``: the fitted distribution written out, :math:`P(k)` being the fraction of the
        nodes with degree :math:`k`, e.g. ``"P(k) = 0.2647 * k^(-2.368) / 0.025817, for k >= 12"``;
      * ``n`` (nodes with positive degree), ``n_tail`` (nodes in the fitted range), ``ks``
        (Kolmogorov-Smirnov distance of the power law on its tail, NaN for the exponential);
      * ``tests``: :class:`pandas.DataFrame`, one row per test: ``alternative``, ``degrees``
        (``"all"`` or ``"tail"``), ``loglikelihood_ratio`` (positive favors the power law), ``p``
        and ``favors``;
      * with ``ccdf = True``, ``ccdf``: :class:`pandas.DataFrame` with one row per degree value
        from ``k_min``: ``degree``, ``ccdf`` (fraction of nodes with that degree or higher, as
        :func:`degree` with ``distribution = "ccdf"``) and ``fit`` (the same from the best fit).
        Plotting both columns on log-log axes shows the fit over the empirical distribution.
   :raises ValueError: If fewer than 10 nodes are in the tail, or a parameter is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      fit = gsk.degree_power_law(net)
      fit["exponent"], fit["k_min"], fit["cutoff"]
      fit["formula"]       # e.g. "P(k) = 0.2647 * k^(-2.368) / 0.025817, for k >= 12"
      fit["tests"]
      fit = gsk.degree_power_law(net, ccdf = True)
      fit["ccdf"].plot(x = "degree", y = ["ccdf", "fit"], loglog = True)
   """
   comps = range(len(net.components)) if component is None else [component]
   values = np.concatenate([_degrees(net, c, direction).cpu().numpy() for c in comps])
   if net.layer_labels is not None:
      codes = np.concatenate([(net.components[c].entity if layer is None else net.components[c].layer).cpu().numpy() for c in comps])
      if layer is None:
         values = np.bincount(codes, values)[np.bincount(codes) > 0]
      else:
         if layer not in net.layer_labels:
            raise ValueError(f"no layer {layer!r}; layers: {net.layer_labels}")
         values = values[codes == list(net.layer_labels).index(layer)]
   k = np.sort(values[values > 0].astype(np.float64))
   if k_min is None:
      candidates = [x for x in np.unique(k) if (k >= x).sum() >= 10]
      if not candidates:
         raise ValueError("fewer than 10 nodes with positive degree: no fit")
      with np.errstate(all = "ignore"):
         k_min = min(candidates, key = lambda x: _ks(k[k >= x], x, _pl_fit(k[k >= x], x)[0]))
   tail = k[k >= k_min]
   if len(tail) < 10:
      raise ValueError(f"only {len(tail)} nodes with degree at least {k_min}: no fit")
   with np.errstate(all = "ignore"):
      a, pl = _pl_fit(tail, k_min)
      exponential, lognormal, cutoff, (a_c, lam) = _alternatives(tail, k_min, a)
      kmin0 = k.min()   # exponential test over all positive degrees, too
      a0, pl0 = _pl_fit(k, kmin0)
      lam0, exponential0 = _exponential(k, kmin0)
   rows = []
   for name, rng, l1, l2, nested in (("exponential", "all", pl0 if kmin0 < k_min else pl, exponential0, False), ("exponential", "tail", pl, exponential, False), ("lognormal", "tail", pl, lognormal, False), ("power law with exponential cutoff", "tail", pl, cutoff, True)):
      R, p = _vuong(l1, l2, nested)
      rows.append((name, rng, R, p, ("power law" if R > 0 else name) if p < significance else "inconclusive"))
   tests = pd.DataFrame(rows, columns = ["alternative", "degrees", "loglikelihood_ratio", "p", "favors"])
   out = {"exponent": float(a), "k_min": int(k_min), "cutoff": 0., "n": len(k), "n_tail": len(tail), "ks": _ks(tail, k_min, a), "verdict": "power law", "model": "power law", "tests": tests}

   def finish(out):   # normalization, formula and, with ccdf, the fitted curve of the best fit
      kmin = out["k_min"]
      x = np.unique(k[k >= kmin])
      f = float((values >= kmin).sum() / len(values))
      if out["model"] == "exponential":
         l = out["rate"]
         Z = float(np.exp(-l * kmin) / -np.expm1(-l))
         formula = f"P(k) = {f:.6g} * (1 - exp(-{l:.6g})) * exp(-{l:.6g} * (k - {kmin})), for k >= {kmin}"
         fit = lambda: f * np.exp(-l * (x - kmin))
      elif out["cutoff"] == 0:
         e = out["exponent"]
         Z = float(special.zeta(e, kmin))
         formula = f"P(k) = {f:.6g} * k^(-{e:.6g}) / {Z:.6g}, for k >= {kmin}"
         fit = lambda: f * special.zeta(e, x) / Z
      else:
         e, l = out["exponent"], out["cutoff"]
         Z = float(_cutoff_norm(e, l, kmin))
         formula = f"P(k) = {f:.6g} * k^(-{e:.6g}) * exp(-{l:.6g} * k) / {Z:.6g}, for k >= {kmin}"
         fit = lambda: f * np.array([_cutoff_norm(e, l, v) for v in x]) / Z
      out.update(normalization = Z, tail_fraction = f, formula = formula)
      if ccdf:
         emp = np.array([(values >= v).sum() for v in x]) / len(values) if len(values) * len(x) >= 2 ** 26 else (values[None, :] >= x[:, None]).sum(1) / len(values)
         out["ccdf"] = pd.DataFrame({"degree": x, "ccdf": emp, "fit": fit()})
      return out

   if tests.favors[0] == "exponential" and tests.favors[1] != "power law":
      warnings.warn("the degree distribution is not a power law: an exponential fits significantly better over all degrees, and the tail does not favor a power law; no exponent is given, the best fit is the exponential")
      out.update(exponent = float("nan"), verdict = "not a power law", model = "exponential", rate = float(lam0), k_min = int(kmin0), n_tail = len(k), ks = float("nan"))
      return finish(out)
   if tests.favors[1] != "power law":
      warnings.warn(f"an exponential tail cannot be ruled out (test on the tail favors: {tests.favors[1]})")
   if tests.favors[2] != "power law":
      warnings.warn(f"a lognormal fits the degree distribution as well as a power law or better (favors: {tests.favors[2]}); the exponent is given anyway")
   if tests.favors[3] == "power law with exponential cutoff":
      out.update(exponent = float(a_c), cutoff = float(lam), verdict = "power law with exponential cutoff", model = "power law with exponential cutoff")
   return finish(out)


def _share_entropy(k):
   """
   Entropy of the degree shares :math:`k_i / \\sum_j k_j`, in nats (0 for no degree).
   """
   k = k[k > 0].astype(np.float64)
   if not len(k):
      return np.nan
   p = k / k.sum()
   return float(-(p * np.log(p)).sum())


def _expected_entropy(n, m, pairs, per_node, total):
   """
   Expected degree share entropy of uniform random graphs with ``n`` nodes and ``m`` edges.

   The degrees sum to ``total`` in every such graph, so
   :math:`E[H] = \\log T - \\frac{n}{T} E[k \\log k]`, and a node's degree is hypergeometric: it
   takes part in ``per_node`` of the ``pairs`` possible edges, ``m`` of which are drawn.
   """
   mean = per_node * m / pairs
   top = min(per_node, int(mean + 50 * np.sqrt(mean) + 50))   # beyond, probabilities vanish
   k = np.arange(1, top + 1)
   pk = stats.hypergeom(pairs, per_node, m).pmf(k)
   return float(np.log(total) - n * (pk * k * np.log(k)).sum() / total)


def _minimum_entropy(n, m):
   """
   Smallest degree share entropy of a connected graph with ``n`` nodes and ``m`` edges, for
   :math:`n - 1 \\leq m \\leq 2n - 3` (Cambie and Mazzamurro, 2022, Theorem 8); NaN otherwise.
   """
   if n < 3 or not n - 1 <= m <= 2 * n - 3:
      return np.nan
   special_cases = {n - 1: [n - 1] + [1] * (n - 1), n + 2: [n - 1, 3, 3, 3] + [1] * (n - 4),
                    n + 4: [n - 1, 4, 4, 3, 3] + [1] * (n - 5), n + 5: [n - 1, 4, 4, 4, 4] + [1] * (n - 5)}
   d = special_cases.get(m, [n - 1, m - n + 2] + [2] * (m - n + 1) + [1] * (2 * n - m - 3))
   return _share_entropy(np.array(d))


def degree_centralization(net, component = None, mode = "all"):
   """
   Degree centralization from the entropy of the degree sequence, adjusted for the size and density
   of the network.

   The degree share entropy (the first degree-based entropy of Cambie and Mazzamurro, 2022),
   :math:`H = -\\sum_i \\frac{k_i}{T} \\log \\frac{k_i}{T}` with :math:`T = \\sum_i k_i`, is
   :math:`\\log n` when all nodes have the same degree and decreases as the edges concentrate on
   fewer nodes. It also grows with the number of nodes, so it is compared with its expected value
   :math:`E[H]` in uniform random graphs with the same numbers of nodes :math:`n` and edges
   :math:`m` (Erdos-Renyi :math:`G(n, m)`). There the degrees always sum to :math:`T`, so
   :math:`H = \\log T - \\frac{1}{T} \\sum_i k_i \\log k_i` and
   :math:`E[H] = \\log T - \\frac{n}{T} E[k \\log k]`, exactly, with the degree :math:`k` of a node
   hypergeometric: a node takes part in :math:`n - 1` of the :math:`n (n - 1) / 2` possible edges
   (:math:`n (n - 1)` in directed networks, :math:`2 (n - 1)` per node with ``mode = "all"``),
   :math:`m` of which are drawn. No sampling is needed.

   The centralization is :math:`C = 1 - e^{H - E[H]}`: :math:`e^H` is the effective number of nodes
   sharing the edges, so :math:`C` is the fraction of effective nodes lost with respect to a random
   network of the same size and density. It is 0 for degrees as even as at random, tends to 1 as
   the edges concentrate on a few hubs (a star has :math:`C = 0.74` with 101 nodes, 0.97 with
   10,001), and is negative for networks more even than random (e.g. a cycle). Networks of the same
   kind and different sizes have similar values.

   For connected undirected networks with :math:`n - 1 \\leq m \\leq 2n - 3` (sparse networks: trees
   up to average degree about 4), the smallest possible entropy :math:`H_{min}` is known (Cambie and
   Mazzamurro, 2022): a node joined to all others plus a second hub. The relative centralization
   :math:`C' = (E[H] - H) / (E[H] - H_{min})` then places the network between random (0) and the
   most centralized network with its numbers of nodes and edges (1). Since that range widens with
   :math:`\\log n` while networks of the same kind stay at a similar distance from random,
   :math:`C'` decreases with size and is meant to compare networks of similar size; outside that
   range (directed, disconnected or denser networks) it is NaN.

   Degrees are those of the simple graph: parallel edges count once, self loops are ignored, and
   weights are not used (the closed form holds for unweighted graphs only). Nodes without edges
   count in :math:`n`. In multilayer networks the supra-graph is used.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param mode: In directed networks, the degree used: ``"all"`` (in plus out), ``"in"`` or
      ``"out"``; ignored in undirected networks.
   :return: Dictionary with ``"centralization"`` (:math:`C`), ``"relative_centralization"``
      (:math:`C'`), ``"entropy"`` (:math:`H`), ``"expected_entropy"`` (:math:`E[H]`) and
      ``"minimum_entropy"`` (:math:`H_{min}`, NaN when unknown); all NaN without edges.
   :raises ValueError: If ``mode`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.degree_centralization(net)["centralization"]
      gsk.degree_centralization(net, component = 0)             # largest component, with C'
   """
   if mode not in ("all", "in", "out"):
      raise ValueError(f"unknown mode: {mode}")
   comps = list(range(len(net.components))) if component is None else [component]
   n = sum(net.components[c].num_nodes for c in comps)
   keys = []
   for c, o in zip(comps, np.cumsum([0] + [net.components[c].num_nodes for c in comps])):
      s, t = net.components[c].edge_index.cpu().numpy()
      keep = s != t
      s, t = s[keep] + o, t[keep] + o
      if not net.directed:
         s, t = np.minimum(s, t), np.maximum(s, t)
      keys.append(s.astype(np.int64) * n + t)
   key = np.unique(np.concatenate(keys)) if keys else np.zeros(0, dtype = np.int64)
   s, t = key // n, key % n
   m = len(key)
   if not net.directed:
      k, pairs, per_node, total = np.bincount(s, minlength = n) + np.bincount(t, minlength = n), n * (n - 1) // 2, n - 1, 2 * m
   elif mode == "all":
      k, pairs, per_node, total = np.bincount(s, minlength = n) + np.bincount(t, minlength = n), n * (n - 1), 2 * (n - 1), 2 * m
   else:
      k, pairs, per_node, total = np.bincount(s if mode == "out" else t, minlength = n), n * (n - 1), n - 1, m
   nan = float("nan")
   if m == 0:
      return {"centralization": nan, "relative_centralization": nan, "entropy": nan, "expected_entropy": nan, "minimum_entropy": nan}
   H = _share_entropy(k)
   E = _expected_entropy(n, m, pairs, per_node, total)
   hmin = _minimum_entropy(n, m) if not net.directed and len(comps) == 1 else nan
   rel = (E - H) / (E - hmin) if np.isfinite(hmin) and E > hmin else nan
   return {"centralization": float(1 - np.exp(H - E)), "relative_centralization": float(rel), "entropy": H, "expected_entropy": E, "minimum_entropy": float(hmin)}


def density(net, component = None, by = "layer"):
   """
   Density: the fraction of the possible edges that are present.

   * undirected networks: :math:`2m / (n (n - 1))`;
   * directed networks: :math:`m / (n (n - 1))` (both directions are possible for every pair);
   * bipartite networks: :math:`m / (n_0 n_1)`, or :math:`m / (2 n_0 n_1)` if directed, with
     :math:`n_0` and :math:`n_1` the numbers of nodes of the two types (edges are only possible
     between types).

   :math:`m` counts the distinct pairs of nodes joined by an edge (distinct ordered pairs in
   directed networks): parallel edges count once and self loops are left out, so the density is
   between 0 and 1 (NetworkX instead counts every edge, self loops included). :math:`n` counts all
   nodes, isolated nodes included. NaN if no edge is possible.

   In multilayer networks only intra-layer edges count: couplings and other interlayer edges are
   left out, and the possible edges are those within each layer. ``by = "layer"`` gives the
   density of each layer, over its copies; ``by = "network"`` pools the layers, as the fraction of
   all the possible intra-layer edges, :math:`\\sum_l m_l / \\sum_l \\binom{n_l}{2}` in undirected
   networks.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param by: In multilayer networks, ``"layer"`` or ``"network"``; ignored otherwise.
   :return: A float; with ``by = "layer"`` in multilayer networks, a :class:`pandas.DataFrame`
      with columns ``layer`` and ``density``, one row per layer with copies.
   :raises ValueError: If ``by`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      gsk.density(gsk.load_network("edges.tsv"))                         # 2 m / (n (n - 1))
      gsk.density(gsk.load_network("edges.tsv", directed = True))        # m / (n (n - 1))
      net = gsk.load_network("multiplex.tsv", multilayer = True, coupling = "all")
      gsk.density(net)                                                   # one density per layer
      gsk.density(net, by = "network")                                   # all layers pooled
   """
   ml = net.layer_labels is not None
   if ml and by not in ("layer", "network"):
      raise ValueError(f"unknown by: {by}")
   L = len(net.layer_labels) if ml else 1
   edges, nodes = np.zeros(L), np.zeros((L, 2))   # nodes: per layer, by type (type 0 only if not bipartite)
   for c in range(len(net.components)) if component is None else [component]:
      data = net.components[c]
      n = data.num_nodes
      s, t = data.edge_index
      keep = s != t
      if ml:
         keep &= ~net.interlayer_mask(c)
      s, t = s[keep], t[keep]
      key = s * n + t if net.directed else torch.minimum(s, t) * n + torch.maximum(s, t)
      key = torch.unique(key)
      lay = data.layer if ml else torch.zeros(n, dtype = torch.long, device = s.device)
      edges += torch.bincount(lay[key // n], minlength = L).cpu().numpy()
      kind = torch.zeros(n, dtype = torch.long, device = s.device)
      if net.bipartite_split is not None:
         kind[net.bipartite_split[c]:] = 1
      nodes += torch.bincount(lay * 2 + kind, minlength = 2 * L).view(L, 2).cpu().numpy()
   if net.bipartite_split is not None:
      possible = nodes[:, 0] * nodes[:, 1] * (2 if net.directed else 1)
   else:
      possible = nodes[:, 0] * (nodes[:, 0] - 1) / (1 if net.directed else 2)
   ratio = lambda m, p: float(m / p) if p > 0 else float("nan")
   if not ml or by == "network":
      return ratio(edges.sum(), possible.sum())
   rows = [(label, ratio(edges[l], possible[l])) for l, label in enumerate(net.layer_labels) if nodes[l].sum() > 0]
   return pd.DataFrame(rows, columns = ["layer", "density"])


def reciprocity(net, normalized = False, component = None, by = "layer"):
   """
   Reciprocity of a directed network: the fraction of edges whose reverse edge is also present,
   :math:`r = L^\\leftrightarrow / L`.

   :math:`L` counts the distinct ordered pairs joined by an edge (parallel edges count once, self
   loops are left out, as in :func:`density`), and :math:`L^\\leftrightarrow` those whose reverse
   pair is joined too. With ``normalized = True``, the measure of Garlaschelli and Loffredo (2004),
   :math:`\\rho = (r - a) / (1 - a)` with :math:`a` the density (:func:`density`): 0 if edges are
   reciprocated as often as by chance, positive if more often, negative if less, so that networks
   of different density can be compared. NaN if there are no edges (or, normalized, if the density
   is 1).

   In multilayer networks only intra-layer edges count; ``by = "layer"`` gives the reciprocity of
   each layer, ``by = "network"`` pools the edges of all layers (and their densities, as in
   :func:`density`).

   :param net: A directed :class:`~graphoskop.Network`.
   :param normalized: Return :math:`\\rho` instead of :math:`r`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param by: In multilayer networks, ``"layer"`` or ``"network"``; ignored otherwise.
   :return: A float; with ``by = "layer"`` in multilayer networks, a :class:`pandas.DataFrame`
      with columns ``layer`` and ``reciprocity``, one row per layer with edges.
   :raises ValueError: If the network is undirected (where every edge is reciprocated), or ``by``
      is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("follows.tsv", directed = True)
      gsk.reciprocity(net)                       # fraction of reciprocated edges
      gsk.reciprocity(net, normalized = True)    # compared with the density
   """
   if not net.directed:
      raise ValueError("reciprocity is for directed networks: in undirected networks every edge is reciprocated")
   ml = net.layer_labels is not None
   if ml and by not in ("layer", "network"):
      raise ValueError(f"unknown by: {by}")
   L = len(net.layer_labels) if ml else 1
   edges, mutual = np.zeros(L), np.zeros(L)
   for c in range(len(net.components)) if component is None else [component]:
      data = net.components[c]
      n = data.num_nodes
      s, t = data.edge_index
      keep = s != t
      if ml:
         keep &= ~net.interlayer_mask(c)
      key = torch.unique(s[keep] * n + t[keep])   # sorted
      if not key.shape[0]:
         continue
      rev = (key % n) * n + key // n
      found = key[torch.searchsorted(key, rev).clamp(max = key.shape[0] - 1)] == rev
      lay = (data.layer if ml else torch.zeros(n, dtype = torch.long, device = s.device))[key // n]
      edges += torch.bincount(lay, minlength = L).cpu().numpy()
      mutual += torch.bincount(lay[found], minlength = L).cpu().numpy()
   if ml and by == "network":
      edges, mutual = edges.sum(keepdims = True), mutual.sum(keepdims = True)
   r = np.where(edges > 0, mutual / np.maximum(edges, 1), np.nan)
   if normalized:
      a = density(net, component, by)
      if isinstance(a, pd.DataFrame):   # one row per layer with copies: align with all layers
         a = a.set_index("layer")["density"].reindex(list(net.layer_labels)).to_numpy()
      a = np.atleast_1d(np.asarray(a, dtype = float))
      with np.errstate(invalid = "ignore", divide = "ignore"):
         r = np.where(a < 1, (r - a) / (1 - a), np.nan)
   if not ml or by == "network":
      return float(r[0])
   return pd.DataFrame([(label, float(r[l])) for l, label in enumerate(net.layer_labels) if edges[l] > 0], columns = ["layer", "reciprocity"])


def average_shortest_path_length(net, component = 0, weight = None, inverse = False, samples = None, seed = 0, n_jobs = None, lengths = None):
   """
   Average shortest path length: the mean length of the shortest paths between all ordered pairs
   of distinct nodes connected by a path.

   Paths are those of :mod:`~graphoskop.shortest_paths` (same engine, same ``weight`` and
   ``inverse`` options, shortest of parallel edges, directions followed in directed networks, the
   supra-graph in multilayer networks, where changing layer along a coupling is a step and pairs
   of copies of the same entity count). Pairs without a path are left out: nodes in different
   components, and, in directed networks, pairs where the target cannot be reached from the
   source (NetworkX instead raises an error unless the network is strongly connected). In an
   undirected component every pair counts, so this is the usual
   :math:`\\frac{1}{n (n - 1)} \\sum_{s \\ne t} d(s, t)`. NaN if no pair is connected.

   Lengths from each batch of sources are summed as soon as they are found, so memory stays at a
   few rows of lengths, not :math:`n^2`; searches are split across processes as in
   :func:`~graphoskop.all_pairs_shortest_path_lengths` (see :mod:`~graphoskop.shortest_paths` for
   ``n_jobs`` and the ``if __name__ == "__main__":`` guard). On large networks, ``samples`` uses
   only that many random source nodes (drawn uniformly from the nodes considered, ``seed`` for
   reproducibility), which estimates the average with :math:`O(\\text{samples} \\cdot m)` work.

   With ``lengths``, the lengths precomputed by :func:`~graphoskop.all_pairs_shortest_path_lengths`
   (with the same ``component``, ``weight`` and ``inverse``) are used and no path is searched
   again; ``weight`` and ``inverse`` are then ignored. The sums run on their device.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network
      (pairs in different components left out).
   :param weight: Name of the edge attribute used as edge length, or ``None`` for unit lengths.
   :param inverse: Use the reciprocal of the edge attribute as length.
   :param samples: Number of random source nodes, or ``None`` for all nodes (exact).
   :param seed: Seed of the random sources.
   :param n_jobs: Number of processes for the searches, ``None`` for all CPUs.
   :param lengths: Precomputed lengths: a tensor for one component, or the list of tensors of
      ``all_pairs_shortest_path_lengths(net, None, ...)`` for ``component = None``.
   :return: The average length, a float.
   :raises ValueError: If some lengths are negative or NaN, or ``lengths`` does not match.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.average_shortest_path_length(net)                                   # largest component
      gsk.average_shortest_path_length(net, weight = "weight", inverse = True)
      gsk.average_shortest_path_length(net, samples = 1000)                   # estimate
      P = gsk.all_pairs_shortest_path_lengths(net)
      gsk.average_shortest_path_length(net, lengths = P)                      # no new search

      if __name__ == "__main__":   # needed for parallel searches on some platforms
         gsk.average_shortest_path_length(big_net, n_jobs = 8)
   """
   comps = list(range(len(net.components)) if component is None else [component])
   sizes = np.array([net.components[c].num_nodes for c in comps])
   chosen = [np.arange(k) for k in sizes]
   if samples is not None and samples < sizes.sum():   # uniform over all nodes considered
      pick = np.sort(np.random.default_rng(seed).choice(int(sizes.sum()), samples, replace = False))
      bounds = np.concatenate([[0], np.cumsum(sizes)])
      chosen = [pick[(pick >= a) & (pick < b)] - a for a, b in zip(bounds[:-1], bounds[1:])]
   if lengths is not None and component is not None:
      lengths = [lengths]
   total, count = 0., 0
   for k, (c, src, n) in enumerate(zip(comps, chosen, sizes)):
      if n < 2 or not len(src):
         continue
      r, d, _ = _source_stats(net, c, src, weight, inverse, n_jobs, lengths = None if lengths is None else lengths[k])
      total, count = total + d.sum(), count + r.sum()
   return total / count if count else float("nan")


def _clustering(net, component, batch_size):
   """
   Numerator and denominator of the local clustering coefficient of every node of a component.

   :param net: Network.
   :param component: Component index.
   :param batch_size: Paths expanded at once.
   :return: ``(num, den)``, int64 vectors of length ``n`` on the device of the component:
      :math:`[(A + A^T)^3]_{ii} / 2` and :math:`d_i (d_i - 1) - 2 d^\\leftrightarrow_i`.
   """
   data = net.components[component]
   n = data.num_nodes
   s, t = data.edge_index
   keep = s != t
   if net.layer_labels is not None:
      keep &= ~net.interlayer_mask(component)
   key = torch.unique(s[keep] * n + t[keep])
   src, dst, slot, along = _skeleton(key // n, key % n, n)
   w = torch.bincount(slot, minlength = src.shape[0])   # 1 or 2 directions per pair
   d = torch.bincount(src, w, minlength = n) + torch.bincount(dst, w, minlength = n)
   recip = torch.bincount(src[w == 2], minlength = n) + torch.bincount(dst[w == 2], minlength = n)
   num = torch.zeros(n, dtype = torch.long, device = s.device)
   for ab, bc, ac in _paths(src, dst, n, batch_size) if src.shape[0] else ():
      x = w[ab] * w[bc] * w[ac]
      for v in (src[ab], dst[ab], dst[bc]):
         num.index_add_(0, v, x)
   return num, (d * (d - 1) - 2 * recip).long()


def local_clustering(net, node = None, component = 0, batch_size = 2 ** 22):
   """
   Local clustering coefficient of nodes (see the module description).

   NaN for nodes with fewer than two neighbors, where it is undefined.

   :param net: A :class:`~graphoskop.Network`.
   :param node: Node (input id or internal id ``(component, index)``; in multilayer networks a
      copy ``(entity, layer)``), list of nodes, or ``None``.
   :param component: With ``node = None``, index of the component (0 is the largest), or ``None``
      for a list with one vector per component; ignored when ``node`` is given.
   :param batch_size: Number of 2-paths expanded at once; larger is faster, especially on GPU,
      and uses more memory.
   :return: With ``node``, a float; with a list of nodes, a :class:`pandas.DataFrame` with columns
      ``node`` (as given) and ``clustering``; with ``node = None``, a float64 vector over the nodes
      of the component, on its device, ordered by internal index (as node attributes).
   :raises KeyError: If a node does not exist.
   :raises ValueError: If a node is an entity of a multilayer network (give one of its copies),
      or is both an input id and an internal id.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.local_clustering(net, "alice")
      gsk.local_clustering(net, ["alice", "bob"])
      c = gsk.local_clustering(net)                  # vector over component 0
      net.set_node_attribute("clustering", c)
   """
   if node is None:
      if component is None:
         return [local_clustering(net, None, c, batch_size) for c in range(len(net.components))]
      num, den = _clustering(net, component, batch_size)
      return torch.where(den > 0, num.double() / den.clamp(min = 1), torch.nan)
   single = not isinstance(node, (list, np.ndarray, pd.Series, pd.Index))
   ids = []
   for x in [node] if single else node:
      found = net._nodes(x)
      if len(found) > 1:
         raise ValueError(f"{x!r} is an entity: give one of its copies (entity, layer)")
      ids.append(found[0])
   vec = {c: local_clustering(net, None, c, batch_size).cpu().numpy() for c in set(c for c, _ in ids)}
   values = [float(vec[c][i]) for c, i in ids]
   return values[0] if single else pd.DataFrame({"node": list(node), "clustering": values})


def _per_layer(net, component, batch_size, reduce):
   """
   A clustering statistic of the whole network or of a component, per layer in multilayer networks.

   :param net: Network.
   :param component: Component index, or ``None`` for all components.
   :param batch_size: Paths expanded at once.
   :param reduce: Function of the concatenated ``(num, den)`` (NumPy int64 arrays) of a set of
      nodes, returning the statistic.
   :return: The statistic, or a :class:`pandas.DataFrame` with columns ``layer`` and
      ``clustering`` in multilayer networks (layers with at least one copy).
   """
   comps = range(len(net.components)) if component is None else [component]
   parts = [_clustering(net, c, batch_size) for c in comps]
   num = np.concatenate([x.cpu().numpy() for x, _ in parts])
   den = np.concatenate([y.cpu().numpy() for _, y in parts])
   if net.layer_labels is None:
      return reduce(num, den)
   lay = np.concatenate([net.components[c].layer.cpu().numpy() for c in comps])
   return pd.DataFrame([(label, reduce(num[lay == l], den[lay == l])) for l, label in enumerate(net.layer_labels) if (lay == l).any()], columns = ["layer", "clustering"])


def global_clustering(net, component = None, batch_size = 2 ** 22):
   """
   Global clustering coefficient (transitivity, see the module description).

   :math:`\\sum_i [(A + A^T)^3]_{ii} / \\sum_i 2 [d_i (d_i - 1) - 2 d^\\leftrightarrow_i]`; in
   undirected networks, three times the number of triangles over the number of connected triples.
   NaN if there are no connected triples.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param batch_size: Number of 2-paths expanded at once; larger is faster, especially on GPU,
      and uses more memory.
   :return: A float; in multilayer networks, a :class:`pandas.DataFrame` with columns ``layer``
      and ``clustering``, one row per layer.

   Example:

   .. code-block:: python

      gsk.global_clustering(net)
   """
   return _per_layer(net, component, batch_size, lambda num, den: float(num.sum() / den.sum()) if den.sum() else float("nan"))


def average_clustering(net, component = None, skip_undefined = False, batch_size = 2 ** 22):
   """
   Average clustering coefficient: the mean of the local clustering coefficients
   (:func:`local_clustering`).

   Nodes with fewer than two neighbors, where the local coefficient is undefined, count as 0 (as
   in Watts and Strogatz, 1998, and NetworkX), or are left out with ``skip_undefined = True``. NaN
   if no node is left.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param skip_undefined: Leave out the nodes with fewer than two neighbors.
   :param batch_size: Number of 2-paths expanded at once; larger is faster, especially on GPU,
      and uses more memory.
   :return: A float; in multilayer networks, a :class:`pandas.DataFrame` with columns ``layer``
      and ``clustering``, one row per layer (the mean over its copies).

   Example:

   .. code-block:: python

      gsk.average_clustering(net)
      gsk.average_clustering(net, skip_undefined = True)
   """
   def reduce(num, den):
      c = num[den > 0] / den[den > 0]
      size = len(c) if skip_undefined else len(den)
      return float(c.sum() / size) if size else float("nan")
   return _per_layer(net, component, batch_size, reduce)


def assortativity(net, attribute = None, component = None, weight = None, direction = ("out", "in")):
   """
   Assortativity: the tendency of nodes to connect to nodes with similar attribute values
   (Newman, 2002 and 2003).

   The kind of assortativity follows the type of the attribute:

   * numeric attribute: the Pearson correlation between the values at the two ends of the edges;
   * qualitative attribute: :math:`r = (\\sum_i e_{ii} - \\sum_i a_i b_i) / (1 - \\sum_i a_i b_i)`,
     with :math:`e_{ij}` the fraction of edges from a node of category :math:`i` to one of
     category :math:`j`, :math:`a_i = \\sum_j e_{ij}` and :math:`b_j = \\sum_i e_{ij}`;
   * no attribute (``attribute = None``): degree assortativity, the Pearson correlation between
     the degrees at the two ends of the edges, computed from the edges without any node attribute.

   Both range from -1 (disassortative) to 1 (assortative). In undirected networks every edge counts
   in both directions, so the result is symmetric; in directed networks the value at the source is
   compared with the value at the target, and for degree assortativity ``direction`` chooses which
   degree is taken at each end (default: out-degree of the source, in-degree of the target). A self
   loop counts as much as any other edge (as in Newman's formula and igraph; NetworkX counts self
   loops of undirected networks half), and degrees count them as :func:`degree` does. Edges with a missing value (NaN)
   at either end are left out; in long node tables, nodes without a numeric value have 0, as in
   :meth:`~graphoskop.Network.get_node_attribute`. With ``weight``, edges count proportionally to
   that edge attribute. NaN if undefined (no edges, or no variance).

   In multilayer networks only intra-layer edges count, degrees are within layers (as in
   :func:`degree`), and the result is given per layer.

   All the work runs on the device of the network, in one pass over the edges.

   :param net: A :class:`~graphoskop.Network`.
   :param attribute: Node attribute name (numeric or qualitative), or ``None`` for degree
      assortativity.
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param weight: Edge attribute used as weight, or ``None`` for unit weights.
   :param direction: For degree assortativity in directed networks, the degree at the source and
      at the target: each ``"out"``, ``"in"`` or ``"total"``.
   :return: A float; in multilayer networks, a :class:`pandas.DataFrame` with columns ``layer``
      and ``assortativity``, one row per layer with edges.
   :raises KeyError: If there is no such node or edge attribute.
   :raises ValueError: If ``direction`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      gsk.assortativity(net)                       # degree
      gsk.assortativity(net, "age")                # numeric attribute: Pearson
      gsk.assortativity(net, "city")               # qualitative attribute: categorical
      gsk.assortativity(net, weight = "weight")
   """
   comps = range(len(net.components)) if component is None else [component]
   qual = isinstance(attribute, str) and attribute in net.qualitative_node_attributes
   if qual:   # category codes, consistent over components
      values = [net.get_node_attribute(attribute, c) for c in comps]
      codes = pd.factorize(pd.Series(np.concatenate(values)), use_na_sentinel = True)[0]
      K = int(codes.max()) + 1 if len(codes) and codes.max() >= 0 else 1
      codes = np.split(codes, np.cumsum([len(v) for v in values])[:-1])
   ml = net.layer_labels is not None
   L = len(net.layer_labels) if ml else 1
   stats, shift = None, None
   for k, c in enumerate(comps):
      data = net.components[c]
      s, t = data.edge_index
      w = torch.ones(s.shape[0], dtype = torch.float64, device = s.device) if weight is None else net.get_edge_attribute(weight, c).double()
      if attribute is None:
         if direction[0] not in ("out", "in", "total") or direction[1] not in ("out", "in", "total"):
            raise ValueError(f"unknown direction: {direction}")
         x, y = (_degrees(net, c, d).double() for d in direction)
      elif qual:
         x = y = torch.as_tensor(codes[k], device = s.device).double()
         x = y = torch.where(x < 0, torch.nan, x)
      else:
         x = y = net.get_node_attribute(attribute, c).double()
      keep = ~(torch.isnan(x[s]) | torch.isnan(y[t]) | torch.isnan(w))
      if ml:
         keep &= ~net.interlayer_mask(c)
      s, t, w = s[keep], t[keep], w[keep]
      if not net.directed:   # other edges are stored in both directions, self loops once
         w = torch.where(s == t, 2 * w, w)
      a, b = x[s], y[t]
      g = data.layer[s] if ml else torch.zeros_like(s)
      if qual:   # sum_i e_ii, then sum_i a_i b_i through per-(layer, category) totals
         a, b = a.long(), b.long()
         cols = [torch.bincount(g, w * (a == b), minlength = L), torch.bincount(g, w, minlength = L)]
         sa = torch.bincount(g * K + a, w, minlength = L * K).view(L, K)
         sb = torch.bincount(g * K + b, w, minlength = L * K).view(L, K)
         part = [cols[0], cols[1], sa, sb]
      else:
         if shift is None and a.shape[0]:   # a common shift, against cancellation in the moments
            shift = float(a.mean())
         a, b = a - (shift or 0.), b - (shift or 0.)
         part = [torch.bincount(g, z, minlength = L) for z in (w, w * a, w * b, w * a * a, w * b * b, w * a * b)]
      part = [p.cpu() for p in part]
      stats = part if stats is None else [u + v for u, v in zip(stats, part)]

   def coef(l):
      if stats is None:
         return float("nan")
      if qual:
         W = float(stats[1][l])
         if W == 0:
            return float("nan")
         ab = float((stats[2][l] * stats[3][l]).sum()) / W ** 2
         return (float(stats[0][l]) / W - ab) / (1 - ab) if ab < 1 else float("nan")
      W, sa, sb, saa, sbb, sab = (float(z[l]) for z in stats)
      if W == 0:
         return float("nan")
      cov = sab / W - sa * sb / W ** 2
      va, vb = saa / W - (sa / W) ** 2, sbb / W - (sb / W) ** 2
      return cov / np.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")

   if not ml:
      return coef(0)
   rows = [(label, coef(l)) for l, label in enumerate(net.layer_labels) if stats is not None and float(stats[1 if qual else 0][l]) > 0]
   return pd.DataFrame(rows, columns = ["layer", "assortativity"])
