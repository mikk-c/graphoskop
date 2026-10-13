"""
Network backboning: scoring edges against a null model, so that edges weaker than expected can be
filtered out.

Scoring functions store their scores as new edge attributes of the :class:`~graphoskop.Network`
(existing attributes with the same name are overwritten); computations run in double precision on
the device of each component. :func:`~graphoskop.extract_backbone` then keeps the edges whose score
passes a threshold, :func:`~graphoskop.test_bb_thresholds` summarizes the backbones given by a
range of thresholds, and :func:`~graphoskop.find_bb_thresholds` finds the thresholds at which the
backbone changes qualitatively.

Scoring methods:

* :func:`~graphoskop.noise_corrected`: score and standard deviation, thresholded as a z-score.
* :func:`~graphoskop.disparity_filter`: score only.
* :func:`~graphoskop.doubly_stochastic`: score only.
* :func:`~graphoskop.high_salience_skeleton`: score only.

:func:`~graphoskop.spanning_tree`, :func:`~graphoskop.maximally_filtered_graph` and
:func:`~graphoskop.triangulated_filtered_graph` instead filter the edges directly, keeping a
maximum or minimum spanning tree, or a planar filtered graph, of every component.

The functions above take the name of the score attribute, ``prefix + "_score"``; if the attribute
``prefix + "_sdev"`` exists, it is used as its standard deviation, otherwise the
score is thresholded directly.

In multilayer networks every method works layer by layer, as if each layer were a separate
network (the same as running it on :func:`~graphoskop.layer_network` of each layer): only
intra-layer edges are scored, with strengths, degrees, totals and paths of their own layer;
couplings and other interlayer edges get NaN scores. :func:`~graphoskop.extract_backbone`,
:func:`~graphoskop.spanning_tree`, :func:`~graphoskop.maximally_filtered_graph` and
:func:`~graphoskop.triangulated_filtered_graph` always keep the interlayer edges between the
copies they keep, and the statistics of :func:`~graphoskop.test_bb_thresholds` and
:func:`~graphoskop.find_bb_thresholds` are computed on the intra-layer edges.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree

from .network import _quiet_sparse, _subset
from .planarity import _lr_planar
from .utils import _weights


def _intra(net, c):
   """
   Mask of the intra-layer edges of a component, and the layer of each edge.

   :param net: Network.
   :param c: Component index.
   :return: Boolean tensor and int64 tensor aligned with ``edge_index``, or ``(None, None)`` if
      the network is not multilayer (all edges are intra-layer).
   """
   if net.layer_labels is None:
      return None, None
   return ~net.interlayer_mask(c), net.components[c].layer[net.components[c].edge_index[0]]


def _scatter(score, keep):
   """
   Scores of the intra-layer edges spread over all edges, NaN on interlayer edges.

   :param score: Score of each edge selected by ``keep``.
   :param keep: Mask of the edges with a score, or ``None`` for all edges.
   :return: Score of each edge.
   """
   if keep is None:
      return score
   out = torch.full(keep.shape, float("nan"), dtype = score.dtype, device = score.device)
   out[keep] = score
   return out


def _inter_input(net):
   """
   Mask of the interlayer edges among the input edges of all components (as concatenated by
   :func:`extract_backbone`).

   :param net: Network.
   :return: Boolean :class:`numpy.ndarray`, all ``False`` if the network is not multilayer.
   """
   return np.concatenate([net.interlayer_mask(c)[:net.number_of_edges(c)].cpu().numpy() for c in range(len(net.components))])


def _noise_corrected(ei, w, n, total):
   """
   Noise-corrected score of every edge of a component and its standard deviation.

   :param ei: Edge index.
   :param w: Edge weights.
   :param n: Number of nodes.
   :param total: Total edge weight of the network (of the layer of each edge, in multilayer
      networks), :math:`n_{..}`, as a number or a tensor with one value per edge.
   :return: ``(score, sdev)``, float64.
   """
   s, t = ei
   x = w.double()
   a = torch.zeros(n, dtype = torch.float64, device = x.device).index_add_(0, s, x)[s]   # n_i.
   b = torch.zeros(n, dtype = torch.float64, device = x.device).index_add_(0, t, x)[t]   # n_.j
   ab = a * b
   d = (ab - total * (a + b)) / ab ** 2
   p0 = ab.div_(total ** 2)                                                              # prior mean
   v = (total - a).mul_(total - b).mul_(p0).div_(total ** 2 * (total - 1))               # prior variance
   del a, b
   alpha = p0 ** 2 * (1 - p0) / v - p0
   beta = p0 * (1 - p0 ** 2) / v - (1 - p0)
   e = (alpha + x).div_(alpha.add_(beta).add_(total))                                    # posterior mean
   del v, alpha, beta
   kappa = p0.reciprocal_().div_(total)
   kx = kappa * x
   score = (kx - 1) / (kx + 1)
   sdev = e.mul_(1 - e).mul_(total).sqrt_().mul_(2 * (d.mul_(x).add_(kappa)).abs_()).div_(kx.add_(1) ** 2)
   return score, sdev


def noise_corrected(net, weight = "weight", prefix = "nc"):
   """
   Noise-corrected backbone scores of all edges (Coscia and Neffke, 2017).

   The weight :math:`n_{ij}` of each edge is compared with its expectation given the total weights
   of its source and target, :math:`n_{i.}` and :math:`n_{.j}`, and of the network,
   :math:`n_{..}` (summed over all components). The score lies in :math:`[-1, 1]`, positive when
   the edge is stronger than expected; its standard deviation comes from a Bayesian (beta-binomial)
   estimate of the variance of :math:`n_{ij}`. The backbone keeps the edges with
   ``score > delta * sdev``, e.g. with ``delta`` 1.28, 1.64 or 2.32 (one-sided p-values of 0.1,
   0.05 and 0.01).

   In undirected networks every edge is stored in both directions, so :math:`n_{i.} = n_{.i}` is
   the strength of the node and both directions get the same values. Self loops are scored too;
   they count once in the strength of their node. Edges whose source or target holds all the
   weight of the network get NaN.

   In multilayer networks, the strengths and :math:`n_{..}` are those of the layer of the edge
   (:math:`n_{i.l}`, :math:`n_{.jl}`, :math:`n_{..l}`): each layer is scored on its own.

   :param net: A :class:`~graphoskop.Network`, modified in place.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param prefix: Prefix of the names of the two new edge attributes, ``prefix + "_score"`` and
      ``prefix + "_sdev"``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", directed = True)
      gsk.noise_corrected(net)
      keep = net.get_edge_attribute("nc_score") > 1.64 * net.get_edge_attribute("nc_sdev")
   """
   L = 1 if net.layer_labels is None else len(net.layer_labels)
   total = torch.zeros(L, dtype = torch.float64, device = net.components[0].edge_index.device)
   for c in range(len(net.components)):
      keep, lay = _intra(net, c)
      w = _weights(net, c, weight).double()
      total.index_add_(0, torch.zeros_like(w, dtype = torch.long) if keep is None else lay[keep], w if keep is None else w[keep])
   for c, data in enumerate(net.components):
      keep, lay = _intra(net, c)
      if keep is None:
         score, sdev = _noise_corrected(data.edge_index, _weights(net, c, weight), data.num_nodes, total[0])
      else:
         score, sdev = _noise_corrected(data.edge_index[:, keep], _weights(net, c, weight)[keep], data.num_nodes, total[lay[keep]])
      net.set_edge_attribute(prefix + "_score", _scatter(score, keep), c)
      net.set_edge_attribute(prefix + "_sdev", _scatter(sdev, keep), c)


def _disparity(p, k):
   """
   Disparity filter score :math:`1 - (1 - p)^{k - 1}`, computed accurately for small ``p``; 0
   for nodes with a single edge.

   :param p: Share of the edge weight in the strength of the node.
   :param k: Number of edges of the node.
   :return: Score.
   """
   return torch.where(k > 1, -torch.expm1((k - 1) * torch.log1p(-p)), 0)


def disparity_filter(net, weight = "weight", policy = "max", prefix = "df"):
   """
   Disparity filter backbone scores of all edges (Serrano, Boguna and Vespignani, 2009).

   From the point of view of a node with ``k`` edges, the shares of its strength taken by its edges
   are compared with ``k`` uniformly random shares: an edge taking a share ``p`` gets the score
   :math:`1 - \\alpha`, with :math:`\\alpha = (1 - p)^{k - 1}` the probability of a share at least
   as large. Nodes with a single edge give the score 0. The backbone keeps the edges with
   ``score > threshold``, e.g. 0.95 for :math:`\\alpha < 0.05`:

   .. code-block:: python

      gsk.extract_backbone(net, 0.95, score = "df_score")

   Every edge has two scores: from its source (its share of the out-strength, among the out-edges)
   and from its target (its share of the in-strength, among the in-edges). In undirected networks
   these are the shares of the strengths of its two endpoints. ``policy`` combines them:

   * ``"max"``: the edge passes if it is significant for at least one endpoint (as in the
     original paper);
   * ``"min"``: the edge passes only if it is significant for both endpoints;
   * ``"out"`` or ``"in"`` (directed networks only): the score from the source or from the target.

   Parallel edges and self loops count as separate edges of their nodes (a self loop once in
   undirected networks). In multilayer networks, strengths and degrees count the intra-layer
   edges only, so they are those of each copy in its layer.

   :param net: A :class:`~graphoskop.Network`, modified in place.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param policy: ``"max"``, ``"min"``, ``"out"`` or ``"in"``.
   :param prefix: The score is stored in the edge attribute ``prefix + "_score"``.
   :raises ValueError: If ``policy`` is not valid for the network.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.disparity_filter(net)
      backbone = gsk.extract_backbone(net, 0.95, score = "df_score")
   """
   if policy not in (("max", "min", "out", "in") if net.directed else ("max", "min")):
      raise ValueError(f"unknown policy for a{' directed' if net.directed else 'n undirected'} network: {policy}")
   for c, data in enumerate(net.components):
      keep = _intra(net, c)[0]
      src, trg = data.edge_index if keep is None else data.edge_index[:, keep]
      n = data.num_nodes
      x = _weights(net, c, weight).double() if keep is None else _weights(net, c, weight).double()[keep]
      stats = lambda i: (torch.zeros(n, dtype = torch.float64, device = x.device).index_add_(0, i, x), torch.bincount(i, minlength = n).double())
      S_out, K_out = stats(src)
      S_in, K_in = (S_out, K_out) if not net.directed else stats(trg)   # undirected: both directions stored
      score = None
      if policy != "in":
         score = _disparity(x / S_out[src], K_out[src])
      if policy != "out":
         other = _disparity(x / S_in[trg], K_in[trg])
         score = other if score is None else (torch.maximum if policy == "max" else torch.minimum)(score, other)
      net.set_edge_attribute(prefix + "_score", _scatter(score, keep), c)


_SINKHORN_ITER = 100   # Sinkhorn-Knopp iterations before switching to the Newton method


def _products(r, k, a, cid, C, n, pseudocount):
   """
   Products with :math:`M = A + \\epsilon J` and its transpose, :math:`J` having a block of ones
   per component, without forming :math:`J`.

   :param r: Row of each non-zero cell of :math:`A`.
   :param k: Column of each non-zero cell.
   :param a: Value of each non-zero cell.
   :param cid: Component of each node.
   :param C: Number of components.
   :param n: Number of nodes.
   :param pseudocount: :math:`\\epsilon`.
   :return: Functions computing :math:`M d` and :math:`M^T d`.
   """
   def block(d):
      return pseudocount * torch.zeros(C, dtype = torch.float64, device = d.device).index_add_(0, cid, d)[cid]

   def rows_of(d):
      return torch.zeros(n, dtype = torch.float64, device = d.device).index_add_(0, r, a * d[k]) + block(d)

   def cols_of(d):
      return torch.zeros(n, dtype = torch.float64, device = d.device).index_add_(0, k, a * d[r]) + block(d)

   return rows_of, cols_of


def _balance(mv, n, tol, max_iter, x0 = None):
   """
   Scaling vector ``x`` making :math:`\\mathrm{diag}(x) B \\mathrm{diag}(x)` doubly stochastic, for a
   symmetric non-negative :math:`B` with total support, by the Newton method of Knight and Ruiz
   (2013, "A fast algorithm for matrix balancing"): inner conjugate gradient steps kept inside a
   safe region, converging in a few dozen products with :math:`B` where Sinkhorn-Knopp iteration
   can need thousands. Convergence is tested on the largest deviation, since the norm over all
   rows of a large matrix cannot go below the rounding errors of its sums.

   :param mv: Product with :math:`B`.
   :param n: Size of :math:`B`.
   :param tol: Tolerance on the largest entry of the residual :math:`1 - x \\circ B x`.
   :param max_iter: Maximum number of Newton iterations (and of inner iterations per Newton
      iteration).
   :param x0: Starting point (positive), or ``None`` for ones.
   :return: ``x`` (float64), and whether it converged.
   """
   g, etamax, delta, Delta = 0.9, 0.1, 0.1, 3.0
   e = torch.ones(n, dtype = torch.float64, device = mv.device)
   x = e.clone() if x0 is None else x0.clone()
   v = x * mv(x)
   rk = 1 - v
   rho = float(rk @ rk)
   rout = rold = rho
   eta = etamax
   rt = tol ** 2
   for _ in range(max_iter):
      if float(rk.abs().max()) <= tol:
         return x, True
      y = e.clone()
      innertol = max(eta ** 2 * rout, rt)
      for k in range(max_iter):
         if rho <= innertol:
            break
         if k == 0:
            Z = rk / v
            p = Z.clone()
            rho = float(rk @ Z)
         else:
            p = Z + (rho / rho_old) * p
         w = x * mv(x * p) + v * p
         alpha = rho / float(p @ w)
         ap = alpha * p
         ynew = y + ap
         if float(ynew.min()) <= delta:   # step to the boundary of the safe region
            neg = ap < 0
            y = y + float(((delta - y[neg]) / ap[neg]).min()) * ap
            break
         if float(ynew.max()) >= Delta:
            big = ynew > Delta
            y = y + float(((Delta - y[big]) / ap[big]).min()) * ap
            break
         y = ynew
         rk = rk - alpha * w
         rho_old = rho
         Z = rk / v
         rho = float(rk @ Z)
      x = x * y
      v = x * mv(x)
      rk = 1 - v
      rho = float(rk @ rk)
      rout = rho
      rat = rout / rold
      rold = rout
      eta_o, eta = eta, g * rat
      if g * eta_o ** 2 > 0.1:
         eta = max(eta, g * eta_o ** 2)
      eta = max(min(eta, etamax), 0.5 * tol / max(rout ** 0.5, 1e-300))
   return x, float(rk.abs().max()) <= tol


def doubly_stochastic(net, weight = "weight", pseudocount = 0.001, tol = 1e-12, max_iter = 1000, prefix = "ds"):
   """
   Doubly stochastic backbone scores of all edges (Slater, 2009).

   The weighted adjacency matrix of each component, plus ``pseudocount`` in every cell, is scaled
   into a doubly stochastic matrix :math:`D_1 (A + \\epsilon J) D_2` (every row and column summing
   to 1); the score of an edge is its entry. The scaling of a positive matrix is unique. It is
   computed by Sinkhorn-Knopp iteration, which converges in a few iterations on most components,
   and, for the components not converged after 100 iterations (typically small, nearly
   decomposable ones, e.g. directed components with a small pseudocount), by the Newton method
   of Knight and Ruiz (2013) started from there. Components are scaled independently, so the
   scores of each component sum to its number of nodes. The pseudocount makes the matrix positive,
   which guarantees that the scaling exists; it is applied implicitly, so memory is linear in the
   number of edges and nodes. Parallel edges are summed into one cell and each gets its score;
   self loops are diagonal cells.

   In multilayer networks, the matrices are those of the components of each layer (connected by
   intra-layer edges), as in :func:`~graphoskop.layer_network`.

   The backbone keeps the edges with ``score > threshold``. Scores scale with the inverse of the
   component size, so a threshold means different things in components of very different sizes.
   Slater's second stage, adding edges by decreasing score until the network is connected, keeps
   the edges with a score of at least the smallest score of a maximum spanning forest:
   :func:`~graphoskop.find_bb_thresholds` gives it as the smaller of ``first_isolate`` and
   ``first_component_break`` (keep the edges with a score above a threshold just below it).

   :param net: A :class:`~graphoskop.Network`, modified in place.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param pseudocount: Value added to every cell of each component's matrix, in units of the
      weights. With 0 the scaling exists only for matrices with total support, and never if a
      node has no out- or in-edges.
   :param tol: Convergence threshold: every row and column sum within ``tol`` of 1.
   :param max_iter: Maximum number of Newton iterations after the Sinkhorn-Knopp ones; if reached,
      a warning is issued and the current scores are stored.
   :param prefix: The score is stored in the edge attribute ``prefix + "_score"``.
   :raises ValueError: If ``pseudocount`` is 0 and a node has no out- or in-edges.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.doubly_stochastic(net)
      backbone = gsk.extract_backbone(net, 0.1, score = "ds_score")
   """
   dev = net.components[0].edge_index.device
   sizes = [d.num_nodes for d in net.components]
   offset = np.cumsum([0] + sizes)
   n, C = int(offset[-1]), len(sizes)
   s = torch.cat([d.edge_index[0] + int(o) for d, o in zip(net.components, offset)])
   t = torch.cat([d.edge_index[1] + int(o) for d, o in zip(net.components, offset)])
   x = torch.cat([_weights(net, c, weight).double() for c in range(C)])
   cid = torch.repeat_interleave(torch.arange(C, device = dev), torch.tensor(sizes, device = dev))
   intra = None
   if net.layer_labels is not None:   # intra-layer edges, one block per connected piece of a layer
      intra = torch.cat([_intra(net, c)[0] for c in range(C)])
      s, t, x = s[intra], t[intra], x[intra]
      piece = connected_components(sp.coo_matrix((np.ones(len(s)), (s.cpu().numpy(), t.cpu().numpy())), shape = (n, n)), connection = "weak")[1]
      cid, C = torch.as_tensor(piece, dtype = torch.long, device = dev), int(piece.max()) + 1
   if pseudocount == 0 and ((torch.bincount(s, minlength = n) == 0).any() or (torch.bincount(t, minlength = n) == 0).any()):
      raise ValueError("a node has no out- or in-edges: the matrix cannot be made doubly stochastic without a pseudocount")
   cell, inv = torch.unique(s * n + t, return_inverse = True)   # parallel edges summed into one cell
   a = torch.zeros(len(cell), dtype = torch.float64, device = dev).index_add_(0, inv, x)
   r, k = cell // n, cell % n
   del s, t, x, cell

   # Sinkhorn-Knopp first: it converges in a few iterations on most components
   rows_of, cols_of = _products(r, k, a, cid, C, n, pseudocount)
   d2 = torch.ones(n, dtype = torch.float64, device = dev)
   d1 = 1 / rows_of(d2)
   for _ in range(_SINKHORN_ITER):
      d2 = 1 / cols_of(d1)
      rows = rows_of(d2)
      err = torch.zeros(C, dtype = torch.float64, device = dev).scatter_reduce_(0, cid, (d1 * rows - 1).abs(), "amax")
      d1 = 1 / rows
      if not bool((err >= tol).any()):
         break
   d2 = 1 / cols_of(d1)
   err = torch.zeros(C, dtype = torch.float64, device = dev).scatter_reduce_(0, cid, (d1 * rows_of(d2) - 1).abs(), "amax")

   # Newton method on the components left (typically small, nearly decomposable ones)
   if bool((err >= tol).any()):
      m = (err >= tol)[cid]
      new = torch.cumsum(m, 0) - 1
      cm = m[r]
      sub_cid = torch.unique(cid[m], return_inverse = True)[1]
      sn = int(m.sum())
      sub_rows, sub_cols = _products(new[r[cm]], new[k[cm]], a[cm], sub_cid, int(sub_cid.max()) + 1, sn, pseudocount)
      mv = lambda z: torch.cat([sub_rows(z[sn:]), sub_cols(z[:sn])])
      mv.device = dev
      z, ok = _balance(mv, 2 * sn, tol, max_iter, torch.cat([d1[m], d2[m]]))
      if not ok:
         warnings.warn("doubly_stochastic: the matrix balancing did not converge")
      d1[m], d2[m] = z[:sn], z[sn:]
   score = _scatter((d1[r] * (a + pseudocount) * d2[k])[inv], intra)
   bounds = np.cumsum([0] + [d.edge_index.shape[1] for d in net.components])
   for c in range(len(net.components)):
      net.set_edge_attribute(prefix + "_score", score[bounds[c]:bounds[c + 1]], c)


_HSS_BLOCK = 2 ** 22   # entries of the roots by nodes and roots by edges matrices handled at once
_HSS_TIES = 1e-10      # relative tolerance for an edge to lie on a shortest path


def _hss_counts(graph, s, t, d, roots, device = "cpu", threads = None):
   """
   For every edge, the number of roots whose shortest-path tree (the union of all shortest paths
   from the root) contains it: the edges with ``dist(s) + d == dist(t)`` (up to ties), the others
   being unreachable (NaN distances compare as false).

   :param graph: Sparse distance matrix (shortest of parallel edges, no self loops).
   :param s: Source of each edge (tensor on ``device``).
   :param t: Target of each edge (tensor on ``device``).
   :param d: Distance of each edge (tensor on ``device``).
   :param roots: Roots (:class:`numpy.ndarray`).
   :param device: Device where edges are checked.
   :param threads: Number of torch threads, or ``None`` to leave it unchanged (worker processes).
   :return: Counts (tensor on ``device``).
   """
   if threads is not None:
      torch.set_num_threads(threads)
   n = graph.shape[0]
   count = torch.zeros(len(s), dtype = torch.int64, device = device)
   batch = max(1, _HSS_BLOCK // max(len(s), n, 1))
   for i in range(0, len(roots), batch):
      D = torch.from_numpy(dijkstra(graph, directed = True, indices = roots[i:i + batch])).to(device)
      D[torch.isinf(D)] = torch.nan
      count += ((D[:, s] + d) <= (D * (1 + _HSS_TIES))[:, t]).sum(0)
   return count


def high_salience_skeleton(net, weight = "weight", n_jobs = 1, prefix = "hss"):
   """
   Link salience of all edges, for the high salience skeleton (Grady, Thiemann and Brockmann,
   2012).

   With effective distances :math:`1 / w` (edges with a non-positive weight are not used), the
   shortest-path tree of a root is the union of all shortest paths from it to the other nodes
   (all tied paths included). The salience of an edge is the fraction of roots whose tree contains
   it, so it lies in :math:`[0, 1]`; the high salience skeleton is made of the edges with salience
   close to 1, and salience is typically bimodal, so a threshold such as 0.5 works:

   .. code-block:: python

      gsk.extract_backbone(net, 0.5, score = "hss_score")

   The paper defines salience on connected undirected networks, normalized by the number of nodes.
   Here each component is normalized by its own number of nodes (the roots that can reach the edge
   at all), which is the same on a connected network. In directed networks trees follow the edge
   directions; in undirected networks an edge is in a tree if it is in either direction. Among
   parallel edges only the shortest ones can be on a shortest path; self loops never are. Paths
   whose lengths differ by less than one part in :math:`10^{10}` count as tied, so that rounding
   in the sums of distances does not split ties.

   In multilayer networks, trees use the intra-layer edges only, and each edge is normalized by
   the number of nodes of its connected piece within its layer (its component in
   :func:`~graphoskop.layer_network`).

   This needs a shortest-path search from every node, :math:`O(N E \\log N)`, run by
   :func:`scipy.sparse.csgraph.dijkstra` on the CPU; the edges on each tree are then found on the
   device of the component. ``n_jobs`` processes split the roots of large components (checking
   edges on the CPU); on systems that start processes by spawning (Windows, macOS), the calling
   script must then be protected by ``if __name__ == "__main__":``.

   :param net: A :class:`~graphoskop.Network`, modified in place.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param n_jobs: Number of processes.
   :param prefix: The score is stored in the edge attribute ``prefix + "_score"``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.high_salience_skeleton(net)
      skeleton = gsk.extract_backbone(net, 0.5, score = "hss_score")
   """
   for c, data in enumerate(net.components):
      n = data.num_nodes
      s, t = data.edge_index.cpu().numpy()
      w = _weights(net, c, weight).double().cpu().numpy()
      keep = _intra(net, c)[0]
      intra = np.ones(len(s), dtype = bool) if keep is None else keep.cpu().numpy()
      ok = (w > 0) & (s != t) & intra
      d = 1 / w[ok]
      so, to = s[ok], t[ok]
      order = np.lexsort((d, to, so))   # shortest of parallel edges first
      first = np.ones(len(order), dtype = bool)
      first[1:] = (so[order][1:] != so[order][:-1]) | (to[order][1:] != to[order][:-1])
      graph = sp.csr_matrix((d[order][first], (so[order][first], to[order][first])), shape = (n, n))
      roots = np.arange(n)
      edges = [torch.from_numpy(x) for x in (so, to, d)]
      if n_jobs > 1 and n >= 1000:
         chunks = np.array_split(roots, 4 * n_jobs)
         with ProcessPoolExecutor(n_jobs) as pool:
            count = sum(pool.map(_hss_counts, *zip(*[(graph, *edges, r, "cpu", 1) for r in chunks])))
      else:
         dev = data.edge_index.device
         count = _hss_counts(graph, *[x.to(dev) for x in edges], roots, dev)
      full = np.zeros(len(s))
      full[ok] = count.cpu().numpy()
      if not net.directed:   # an edge is in a tree in either direction: add the two stored copies
         m = net.number_of_edges(c)
         loops = np.flatnonzero(s[:m] != t[:m])
         full[:m][loops] += full[m:]
         full[m:] = full[:m][loops]
      size = n
      if net.layer_labels is not None:   # nodes of the piece of each edge within its layer
         piece = connected_components(sp.coo_matrix((np.ones(int(intra.sum())), (s[intra], t[intra])), shape = (n, n)), connection = "weak")[1]
         size = np.bincount(piece)[piece[s]]
      full = full / size
      full[~intra] = np.nan
      net.set_edge_attribute(prefix + "_score", torch.as_tensor(full), c)


def _max_spanning_forest(a, b, sig, n):
   """
   Maximum significance spanning forest of a multigraph, by Kruskal: for every threshold, its edges
   with significance above the threshold connect the same nodes as all edges above it.

   :param a: First endpoint of each edge.
   :param b: Second endpoint of each edge (edge direction is ignored, self loops are skipped).
   :param sig: Significance of each edge (NaN is the least significant).
   :param n: Number of nodes.
   :return: Indices of the forest edges.
   """
   cand = np.flatnonzero(a != b)
   lo, hi = np.minimum(a[cand], b[cand]), np.maximum(a[cand], b[cand])
   order = np.lexsort((-np.nan_to_num(sig[cand], nan = -np.inf), hi, lo))   # best edge first within each pair of nodes
   cand, lo, hi = cand[order], lo[order], hi[order]
   first = np.ones(len(cand), dtype = bool)
   first[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
   cand, lo, hi = cand[first], lo[first], hi[first]
   order = np.argsort(-np.nan_to_num(sig[cand], nan = -np.inf), kind = "stable")
   rank = np.empty(len(cand))
   rank[order] = np.arange(1, len(cand) + 1)                                   # positive and unique
   tree = minimum_spanning_tree(sp.csr_matrix((rank, (lo, hi)), shape = (n, n))).tocoo()
   return cand[order[tree.data.astype(np.int64) - 1]]


def _spanning_edges(s, t, keep, sig, n):
   """
   Dropped edges joining the pieces left by the kept edges into a maximum significance spanning
   forest (Kruskal on the graph of the pieces): ``p - 1`` edges for an original component split
   into ``p`` pieces, the fewest possible.

   :param s: Source of each edge.
   :param t: Target of each edge.
   :param keep: Mask of the kept edges.
   :param sig: Significance of each edge (NaN is the least significant).
   :param n: Number of nodes.
   :return: Indices of the edges to add.
   """
   piece = connected_components(sp.coo_matrix((np.ones(int(keep.sum())), (s[keep], t[keep])), shape = (n, n)), connection = "weak")[1]
   cand = np.flatnonzero(~keep)
   return cand[_max_spanning_forest(piece[s[cand]], piece[t[cand]], sig[cand], piece.max() + 1)]


def _isolate_edges(s, t, keep, sig, n):
   """
   Dropped edges reattaching the nodes left without edges: going from the most significant, an
   edge is added if one of its endpoints is still isolated. Only edges that are the most
   significant of one of their isolated endpoints can be added, so this is at most one edge per
   isolated node.

   :param s: Source of each edge.
   :param t: Target of each edge.
   :param keep: Mask of the kept edges.
   :param sig: Significance of each edge (NaN is the least significant).
   :param n: Number of nodes.
   :return: Indices of the edges to add.
   """
   iso = (np.bincount(s[keep], minlength = n) + np.bincount(t[keep], minlength = n)) == 0
   cand = np.flatnonzero(~keep & (s != t) & (iso[s] | iso[t]))
   node = np.concatenate([s[cand], t[cand]])
   edge = np.concatenate([cand, cand])
   ok = iso[node]
   node, edge = node[ok], edge[ok]
   order = np.lexsort((-np.nan_to_num(sig[edge], nan = -np.inf), node))
   node, edge = node[order], edge[order]
   first = np.ones(len(node), dtype = bool)
   first[1:] = node[1:] != node[:-1]
   best = np.unique(edge[first])
   best = best[np.argsort(-np.nan_to_num(sig[best], nan = -np.inf), kind = "stable")]
   add = []
   for e in best:
      if iso[s[e]] or iso[t[e]]:
         add.append(e)
         iso[s[e]] = iso[t[e]] = False
   return np.array(add, dtype = np.int64)


def extract_backbone(net, threshold, score = "nc_score", keep_isolates = False, reconnect_components = False, reconnect_isolates = False):
   """
   Backbone of a network: the edges whose score is above a threshold.

   If the score has a standard deviation (an attribute ``prefix + "_sdev"`` next to
   ``prefix + "_score"``, as stored by :func:`~graphoskop.noise_corrected`), the edges with
   ``score > threshold * sdev`` are kept, ``threshold`` being the z-score (e.g. 1.28, 1.64 or
   2.32 for one-sided p-values of 0.1, 0.05 and 0.01). Otherwise the edges with
   ``score > threshold`` are kept, for scores without a standard deviation or to threshold any
   edge attribute directly (e.g. the weight). Edges with a NaN score or standard deviation are
   dropped.

   Dropped edges can be added back, ranked by the same criterion (``score / sdev``, or
   ``score``; NaN ranks last):

   * ``reconnect_components``: each original component split into ``p`` pieces (isolated nodes
     included) is joined again with ``p - 1`` edges, the fewest possible, chosen as a maximum
     spanning forest of the pieces (Kruskal: repeatedly the most significant edge joining two
     different pieces). The backbone then has the same components, with the same nodes, as
     ``net``, so this implies ``reconnect_isolates``.
   * ``reconnect_isolates``: each node left without edges gets back an edge, without rejoining
     larger pieces. Going from the most significant edge, an edge is added if one of its endpoints
     is still isolated, so at most one edge per isolated node and fewer when an edge joins two of
     them. Nodes without edges in ``net`` stay isolated.

   Removing edges can split components, so the backbone is a new :class:`~graphoskop.Network`,
   with components recomputed and sorted by size; ``net`` is not modified. All edge and node
   attributes (numeric and qualitative) are kept; a bipartite network gives a bipartite backbone,
   with the same node types. In undirected networks both directions of an edge are kept or dropped
   together, according to the values of the input direction.

   In multilayer networks only intra-layer edges are filtered, and reconnection works within each
   layer (its components and isolated copies). Interlayer edges (couplings included) are always
   kept between the copies that are kept; a copy left without intra-layer edges is dropped with
   its interlayer edges, unless ``keep_isolates``. The backbone is multilayer, with the same
   layers.

   :param net: A :class:`~graphoskop.Network`.
   :param threshold: Threshold.
   :param score: Name of the edge attribute with the score.
   :param keep_isolates: Keep nodes left without edges, as isolated components.
   :param reconnect_components: Add back the fewest most significant dropped edges keeping every
      original component connected.
   :param reconnect_isolates: Add back the most significant dropped edges of the nodes left
      without edges.
   :return: The backbone, a new :class:`~graphoskop.Network`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", directed = True)
      gsk.noise_corrected(net)
      backbone = gsk.extract_backbone(net, 1.64)                                  # z-score
      backbone = gsk.extract_backbone(net, 1.64, reconnect_components = True)     # same components
      strong = gsk.extract_backbone(net, 10, score = "weight")                    # weight above 10
   """
   sdev = _sdev(net, score)
   js = net._column(net.edge_attributes, score, "edge")
   jd = None if sdev is None else net._column(net.edge_attributes, sdev, "edge")
   names = np.concatenate(net.node_ids)
   offset = np.cumsum([0] + [d.num_nodes for d in net.components])
   s, t, keep, sig = [], [], [], []
   for c in range(len(net.components)):
      ei, X = net._input_edges(c)
      sc = X[:, js]
      keep.append((sc > (threshold if jd is None else threshold * X[:, jd])).cpu().numpy())
      sig.append((sc if jd is None else sc.double() / X[:, jd]).cpu().numpy())
      ei = ei.cpu().numpy() + offset[c]
      s.append(ei[0])
      t.append(ei[1])
   eoffset = np.cumsum([0] + [len(k) for k in keep])
   s, t, keep, sig = np.concatenate(s), np.concatenate(t), np.concatenate(keep), np.concatenate(sig).astype(np.float64)
   inter = _inter_input(net)
   keep &= ~inter
   e = np.flatnonzero(~inter)   # reconnect within layers
   if reconnect_components:
      keep[e[_spanning_edges(s[e], t[e], keep[e], sig[e], len(names))]] = True
   elif reconnect_isolates:
      keep[e[_isolate_edges(s[e], t[e], keep[e], sig[e], len(names))]] = True
   return _subnetwork(net, s, t, keep | inter, eoffset, keep_isolates, inter)


def _subnetwork(net, s, t, keep, eoffset, keep_isolates, inter = None):
   """
   New network with some of the input edges of a network, all attributes kept, components
   recomputed.

   :param net: Network.
   :param s: Global source id of each input edge (components concatenated).
   :param t: Global target id of each input edge.
   :param keep: Mask of the edges to keep.
   :param eoffset: Position of the first edge of each component, and the number of edges.
   :param keep_isolates: Keep the nodes left without edges.
   :param inter: Mask of the interlayer edges, or ``None``: nodes are then kept for their
      intra-layer edges only, and interlayer edges only between kept nodes.
   :return: The new :class:`~graphoskop.Network`, with all node attributes (numeric and
      qualitative) and, in bipartite networks, node types.
   """
   n = int(net.node_offsets[-1])
   rows = np.arange(n)
   if not keep_isolates:
      used = np.zeros(n, dtype = bool)
      k = keep if inter is None else keep & ~inter
      used[s[k]] = used[t[k]] = True
      if inter is not None:
         keep = keep & (~inter | (used[s] & used[t]))
   ea = np.concatenate([net._input_edges(c)[1][torch.as_tensor(keep[eoffset[c]:eoffset[c + 1]], device = net.components[c].edge_attr.device)].cpu().numpy() for c in range(len(net.components))])
   s, t = s[keep], t[keep]
   if not keep_isolates:
      rows = rows[used]
      new = np.cumsum(used) - 1
      s, t = new[s], new[t]
   return _subset(net, s, t, ea, rows)


def spanning_tree(net, weight = "weight", kind = "maximum"):
   """
   Maximum or minimum spanning tree of every component (a spanning forest), as a new network.

   Unlike the scoring functions, this filters the edges directly: each component keeps ``n - 1``
   of its edges, connecting all its nodes with the largest (or smallest) total weight, so the
   result has the same components, with the same nodes, as ``net``, which is not modified. All
   edge and node attributes (numeric and qualitative) are kept, and node types in bipartite
   networks. ``weight`` can be any edge attribute, e.g. a backbone score.

   Edge directions are ignored when choosing the edges (the tree spans each weakly connected
   component) and kept in the result. In multilayer networks the trees span the components of each
   layer, made of intra-layer edges only, and all interlayer edges are kept. Among parallel edges only the best one can be chosen; self
   loops never are. Edges with a NaN weight are used last, only if needed to connect the
   component. Ties are broken arbitrarily but deterministically.

   :param net: A :class:`~graphoskop.Network`.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights
      (any spanning tree).
   :param kind: ``"maximum"`` or ``"minimum"``.
   :return: The spanning forest, a new :class:`~graphoskop.Network`.
   :raises ValueError: If ``kind`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      tree = gsk.spanning_tree(net)                                  # maximum weight
      tree = gsk.spanning_tree(net, "distance", kind = "minimum")
      tree = gsk.spanning_tree(net, "nc_score")                      # most significant edges
   """
   if kind not in ("maximum", "minimum"):
      raise ValueError(f"unknown spanning tree kind: {kind}")
   offset = np.cumsum([0] + [d.num_nodes for d in net.components])
   s, t, w = [], [], []
   for c in range(len(net.components)):
      ei, X = net._input_edges(c)
      ei = ei.cpu().numpy() + offset[c]
      s.append(ei[0])
      t.append(ei[1])
      w.append(np.ones(ei.shape[1]) if weight is None else X[:, net._column(net.edge_attributes, weight, "edge")].double().cpu().numpy())
   eoffset = np.cumsum([0] + [len(x) for x in s])
   s, t, w = np.concatenate(s), np.concatenate(t), np.concatenate(w)
   keep = _inter_input(net)
   e = np.flatnonzero(~keep)
   keep[e[_max_spanning_forest(s[e], t[e], w[e] if kind == "maximum" else -w[e], int(offset[-1]))]] = True
   return _subnetwork(net, s, t, keep, eoffset, True)


def _planar_filter(a, b, w, n):
   """
   Greedy planar filtering (Tumminello et al., 2005): going from the largest weight, an edge is
   kept if the kept edges stay planar, by the Left-Right test of :meth:`~graphoskop.Network.is_planar`.

   The edge by edge test is shortened in three ways. Edges joining two pieces (connected
   components of the kept edges) are kept untested, and edges inside a piece already maximal
   planar (``3k - 6`` edges on ``k`` nodes) are dropped untested. Runs of edges are tested at once:
   a run is kept whole if planar (planarity is inherited by subgraphs), otherwise the first edge
   breaking planarity is found by bisection; run lengths double after a success and halve after a
   failure. A single edge is tested on the kept edges within 2, 3, 4... steps of its endpoints, up
   to its whole piece: if a part with the edge is not planar the whole is not, so most dropped
   edges are dropped after small tests.

   :param a: First endpoint of each edge.
   :param b: Second endpoint of each edge (edge direction is ignored, self loops are skipped).
   :param w: Weight of each edge (NaN is the smallest).
   :param n: Number of nodes.
   :return: Indices of the kept edges.
   """
   cand = np.flatnonzero(a != b)
   lo, hi = np.minimum(a[cand], b[cand]), np.maximum(a[cand], b[cand])
   order = np.lexsort((-np.nan_to_num(w[cand], nan = -np.inf), hi, lo))   # best edge of each pair of nodes
   cand, lo, hi = cand[order], lo[order], hi[order]
   first = np.ones(len(cand), dtype = bool)
   first[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
   cand = cand[first]
   cand = cand[np.argsort(-np.nan_to_num(w[cand], nan = -np.inf), kind = "stable")]
   pairs = list(zip(a[cand].tolist(), b[cand].tolist()))
   kept, g = [], []   # indices and endpoints of the kept edges
   adj = [set() for _ in range(n)]
   parent = list(range(n))
   size, edges = [1] * n, [0] * n   # nodes and kept edges of each piece (at its root)

   def root(x):
      while parent[x] != x:
         parent[x] = parent[parent[x]]
         x = parent[x]
      return x

   def accept(run):
      kept.extend(run)
      for j in run:
         u, v = pairs[j]
         g.append((u, v))
         adj[u].add(v)
         adj[v].add(u)
         x, y = root(u), root(v)
         if x != y:
            parent[y] = x
            size[x] += size[y]
            edges[x] += edges[y]
         edges[x] += 1

   def test(e):   # relabel the nodes, then test
      ids = {}
      e = [(ids.setdefault(u, len(ids)), ids.setdefault(v, len(ids))) for u, v in e]
      return _lr_planar(len(ids), e)

   def planar(run):
      if len(run) > 1:
         return test(g + [pairs[j] for j in run])
      u, v = pairs[run[0]]
      ball, front = {u, v}, {u, v}
      for r in range(1, n + 1):   # balls of growing radius, from 2, up to the whole piece
         front = {y for x in front for y in adj[x]} - ball
         ball |= front
         if front and r < 3:
            continue
         ok = test([(x, y) for x in ball for y in adj[x] if y in ball and x < y] + [(u, v)])
         if not ok or not front:
            return ok

   i, step = 0, 1
   while i < len(pairs):
      run = []
      while i < len(pairs) and len(run) < step:
         x, y = root(pairs[i][0]), root(pairs[i][1])
         if x != y and not run:   # joins two pieces: always planar
            accept([i])
         elif x != y or edges[x] < 3 * size[x] - 6:   # else: maximal planar piece, no edge fits
            run.append(i)
         i += 1
      if not run:
         continue
      if planar(run):
         accept(run)
         step *= 2
         continue
      lo_, hi_ = 0, len(run) - 1   # run[:lo_] planar, run[:hi_ + 1] not
      while lo_ < hi_:
         mid = (lo_ + hi_) // 2
         if planar(run[:mid + 1]):
            lo_ = mid + 1
         else:
            hi_ = mid
      accept(run[:lo_])
      i = run[lo_] + 1   # retry the edges after the dropped one
      step = max(1, step // 2)
   return cand[np.array(kept, dtype = np.int64)]


def maximally_filtered_graph(net, weight = "weight", kind = "maximum"):
   """
   Planar maximally filtered graph (PMFG; Tumminello et al., 2005) of a network, as a new network:
   a planar subgraph with a large total weight.

   Edges are taken from the largest weight to the smallest, and an edge is kept if the kept edges
   stay planar (they can be drawn on a plane without crossings), as Kruskal's algorithm keeps an
   edge if no cycle is formed. The result contains the maximum spanning tree (see
   :func:`~graphoskop.spanning_tree`) and keeps the same components, with the same nodes, as
   ``net``; a component of :math:`n \\geq 3` nodes keeps at most :math:`3(n - 2)` edges, exactly
   that many if it was a complete graph (it is then a triangulation, with only triangles and
   4-cliques as cliques). Finding the planar subgraph of largest total weight is NP-hard: this
   greedy construction is the one of the paper, and it is exact for planar networks, which are
   returned whole.

   As in :func:`~graphoskop.spanning_tree`, edge directions are ignored when choosing the edges and
   kept in the result, only the best of parallel edges can be chosen, self loops never are, edges
   with a NaN weight come last, and ties are broken deterministically. All edge and node attributes
   (numeric and qualitative) are kept, and node types in bipartite networks. In multilayer networks
   each layer is filtered on its own (intra-layer edges only), and all interlayer edges are kept.

   Runs on the CPU, with the planarity test of :meth:`~graphoskop.Network.is_planar`. In dense
   networks most edges are dropped, each after a planarity test; these are shortened by testing
   first only the kept edges near the dropped one (a part with it that is not planar is enough),
   and edges likely to be kept are tested in runs. For a complete graph of :math:`n` nodes this
   takes about 10 seconds for :math:`n = 200` and 2 minutes for :math:`n = 500` (the cost grows
   about as :math:`n^3`; :func:`~graphoskop.triangulated_filtered_graph` is a fast alternative).

   :param net: A :class:`~graphoskop.Network`.
   :param weight: Name of the edge attribute used as edge weight (e.g. a correlation or a backbone
      score), or ``None`` for unit weights (edges taken in input order).
   :param kind: ``"maximum"``, or ``"minimum"`` to favor the smallest weights (e.g. distances).
   :return: The PMFG, a new :class:`~graphoskop.Network`.
   :raises ValueError: If ``kind`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("correlations.tsv")                   # complete graph, weight column
      pmfg = gsk.maximally_filtered_graph(net)
      pmfg.number_of_edges()                                       # 3 * (n - 2)
      pmfg = gsk.maximally_filtered_graph(net, "distance", kind = "minimum")
   """
   if kind not in ("maximum", "minimum"):
      raise ValueError(f"unknown kind: {kind}")
   offset = np.cumsum([0] + [d.num_nodes for d in net.components])
   s, t, w = [], [], []
   for c in range(len(net.components)):
      ei, X = net._input_edges(c)
      ei = ei.cpu().numpy() + offset[c]
      s.append(ei[0])
      t.append(ei[1])
      w.append(np.ones(ei.shape[1]) if weight is None else X[:, net._column(net.edge_attributes, weight, "edge")].double().cpu().numpy())
   eoffset = np.cumsum([0] + [len(x) for x in s])
   s, t, w = np.concatenate(s), np.concatenate(t), np.concatenate(w)
   keep = _inter_input(net)
   e = np.flatnonzero(~keep)
   keep[e[_planar_filter(s[e], t[e], w[e] if kind == "maximum" else -w[e], int(offset[-1]))]] = True
   return _subnetwork(net, s, t, keep, eoffset, True)


def _sdev(net, score):
   """
   Standard deviation attribute of a score attribute: for a score named ``prefix + "_score"``,
   the attribute ``prefix + "_sdev"`` if it exists (as stored by
   :func:`~graphoskop.noise_corrected`), otherwise ``None``.

   :param net: Network.
   :param score: Name (or position) of the score attribute.
   :return: Name of the standard deviation attribute, or ``None``.
   """
   if isinstance(score, str) and score.endswith("_score") and score[:-6] + "_sdev" in net.edge_attributes:
      return score[:-6] + "_sdev"
   return None


def _edge_arrays(net, score, sdev):
   """
   Input edges of all components, with their scores, as :class:`numpy.ndarray`.

   :param net: Network.
   :param score: Name of the score attribute.
   :param sdev: Name of the standard deviation attribute, or ``None``.
   :return: ``(s, t, sc, sd, n)``: global source and target ids, scores and standard deviations
      (``None`` without ``sdev``) in the dtype of the edge attributes, and number of nodes.
      Interlayer edges are left out.
   """
   js = net._column(net.edge_attributes, score, "edge")
   jd = None if sdev is None else net._column(net.edge_attributes, sdev, "edge")
   offset = np.cumsum([0] + [d.num_nodes for d in net.components])
   s, t, sc, sd = [], [], [], []
   for c in range(len(net.components)):
      ei, X = net._input_edges(c)
      ei = ei.cpu().numpy() + offset[c]
      s.append(ei[0])
      t.append(ei[1])
      sc.append(X[:, js].cpu().numpy())
      if jd is not None:
         sd.append(X[:, jd].cpu().numpy())
   e = ~_inter_input(net)
   return np.concatenate(s)[e], np.concatenate(t)[e], np.concatenate(sc)[e], None if jd is None else np.concatenate(sd)[e], int(offset[-1])


def _keep(sc, sd, threshold):
   """
   Edges kept at a threshold, compared as in :func:`extract_backbone` (in the dtype of the scores).
   """
   threshold = float(threshold)
   return sc > (threshold if sd is None else threshold * sd)


def _bb_stats(s, t, keep, n, directed):
   """
   Non-isolated nodes, edges, average degree and components with more than one node of a backbone.

   :param s: Source of each edge.
   :param t: Target of each edge.
   :param keep: Mask of the kept edges.
   :param n: Number of nodes.
   :param directed: Whether the edges are directed.
   :return: ``(N, E, D, C)``.
   """
   u, v = s[keep], t[keep]
   link = u != v
   linked = np.zeros(n, dtype = bool)
   linked[u[link]] = linked[v[link]] = True
   N, E = int(linked.sum()), len(u)
   labels = connected_components(sp.coo_matrix((np.ones(int(link.sum())), (u[link], v[link])), shape = (n, n)), connection = "weak")[1]
   return N, E, (E if directed else 2 * E) / N if N else np.nan, int((np.bincount(labels) > 1).sum())


_COLUMNS = ["threshold", "nodes", "nodes_pct", "edges", "edges_pct", "avg_degree", "avg_degree_pct", "components", "components_pct"]


def _bb_rows(arrays, thresholds, directed):
   """
   Rows of :func:`test_bb_thresholds` for some thresholds.

   :param arrays: Output of :func:`_edge_arrays`.
   :param thresholds: Thresholds (NaN gives a row of NaN).
   :param directed: Whether the edges are directed.
   :return: List of rows.
   """
   s, t, sc, sd, n = arrays
   pct = lambda a, b: 100 * a / b if b else np.nan
   N0, E0, D0, C0 = _bb_stats(s, t, np.ones(len(s), dtype = bool), n, directed)
   rows = []
   for thr in thresholds:
      if np.isnan(thr):
         rows.append([np.nan] * len(_COLUMNS))
         continue
      N, E, D, C = _bb_stats(s, t, _keep(sc, sd, thr), n, directed)
      rows.append([thr, N, pct(N, N0), E, pct(E, E0), D, pct(D, D0), C, pct(C, C0)])
   return rows


def test_bb_thresholds(net, min_threshold, max_threshold, step, score = "nc_score"):
   """
   Size of the backbones given by a regular grid of thresholds, to help choosing one.

   The thresholds go from ``min_threshold`` to ``max_threshold`` (included if on the grid) in
   steps of ``step``, and edges are kept as in :func:`extract_backbone` (without reconnection).
   For each threshold the result reports, in absolute terms and as a percentage of the original
   network (the same statistic computed on ``net``):

   * ``nodes``: nodes that are not isolated, i.e. with at least one edge to another node;
   * ``edges``: edges (in undirected networks, each counted once; self loops included);
   * ``avg_degree``: average degree of the non-isolated nodes, :math:`2E / N` in undirected and
     :math:`E / N` (average out- or in-degree) in directed networks;
   * ``components``: weakly connected components with more than one node.

   Backbones are not built, so this is much faster than calling :func:`extract_backbone` for
   each threshold.

   :param net: A :class:`~graphoskop.Network`.
   :param min_threshold: First threshold.
   :param max_threshold: Last threshold.
   :param step: Distance between consecutive thresholds, positive.
   :param score: Name of the edge attribute with the score (its standard deviation, if any, is
      found as in :func:`extract_backbone`).
   :return: :class:`pandas.DataFrame` with one row per threshold and columns ``threshold``,
      ``nodes``, ``nodes_pct``, ``edges``, ``edges_pct``, ``avg_degree``, ``avg_degree_pct``,
      ``components``, ``components_pct``.
   :raises ValueError: If ``step`` is not positive or ``max_threshold < min_threshold``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", directed = True)
      gsk.noise_corrected(net)
      table = gsk.test_bb_thresholds(net, 1, 3, 0.25)   # z-scores 1, 1.25, ..., 3
   """
   if step <= 0 or max_threshold < min_threshold:
      raise ValueError("step must be positive and max_threshold at least min_threshold")
   k = int(np.floor((max_threshold - min_threshold) / step + 1e-9)) + 1
   thresholds = np.round(min_threshold + step * np.arange(k), 10).tolist()
   return pd.DataFrame(_bb_rows(_edge_arrays(net, score, _sdev(net, score)), thresholds, net.directed), columns = _COLUMNS)


def find_bb_thresholds(net, target_degree = None, score = "nc_score"):
   """
   Thresholds at which the backbone changes qualitatively.

   An edge is kept when its significance (``score / sdev`` if the score has a standard deviation,
   as in :func:`extract_backbone`, otherwise ``score``) is above the threshold, so every statistic of :func:`test_bb_thresholds` only changes at the
   significance of some edge. The thresholds below are found exactly among these values (no grid),
   then checked with the comparison of :func:`extract_backbone` and, if rounding of the scores
   makes it disagree, moved up by the smallest float32 steps until it agrees.

   * ``first_isolate``: smallest threshold at which a node that has edges to other nodes in
     ``net`` is left without them; every smaller threshold keeps all such nodes.
   * ``first_component_break``: smallest threshold at which the number of components with more
     than one node differs from ``net`` (a component splits into pieces with more than one node,
     or vanishes into isolated nodes). Nodes falling off a component without splitting it are
     reported by ``first_isolate``.
   * ``target_degree``: smallest threshold at which the average degree (as in
     :func:`test_bb_thresholds`) reaches ``target_degree``, from above if it is lower than the
     average degree of ``net`` and from below otherwise.

   A threshold is NaN if the change never happens, and ``-inf`` if it happens at every threshold
   (e.g. a node whose edges all have a NaN significance). Standard deviations are assumed
   positive.

   :param net: A :class:`~graphoskop.Network`.
   :param target_degree: Target average degree, or ``None`` to skip that row.
   :param score: Name of the edge attribute with the score (its standard deviation, if any, is
      found as in :func:`extract_backbone`).
   :return: :class:`pandas.DataFrame` with a ``criterion`` column and the columns of
      :func:`test_bb_thresholds`, one row per criterion.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", directed = True)
      gsk.noise_corrected(net)
      table = gsk.find_bb_thresholds(net, target_degree = 3)
      backbone = gsk.extract_backbone(net, table.set_index("criterion").threshold["first_isolate"] - 0.01)
   """
   arrays = _edge_arrays(net, score, _sdev(net, score))
   s, t, sc, sd, n = arrays
   z = sc.astype(np.float64) if sd is None else sc.astype(np.float64) / sd
   z[np.isnan(z)] = -np.inf                                           # never kept
   link = s != t
   N0, _, D0, C0 = _bb_stats(s, t, np.ones(len(s), dtype = bool), n, net.directed)

   # first isolate: a node is isolated from the largest significance of its edges to other nodes
   best = np.full(n, np.nan)
   np.fmax.at(best, s[link], z[link])
   np.fmax.at(best, t[link], z[link])
   best = best[~np.isnan(best)]
   isolate = best.min() if len(best) else np.nan

   # first component break: components above a threshold are those of the maximum spanning
   # forest edges above it; adding them by decreasing significance, an edge joining two nodes
   # seen for the first time adds a component, one joining two seen nodes removes one
   f = _max_spanning_forest(s, t, z, n)
   f = f[np.argsort(-z[f], kind = "stable")]
   ends = np.stack([s[f], t[f]], 1).ravel()
   new = np.zeros(len(ends), dtype = bool)
   new[np.unique(ends, return_index = True)[1]] = True
   count = np.cumsum(new[0::2].astype(np.int64) + new[1::2] - 1)       # components after each edge
   zf = z[f]
   last = np.flatnonzero(np.r_[zf[1:] != zf[:-1], True])                # last edge of each significance
   lower = np.r_[zf[0:1], zf[last[:-1] + 1]] if len(f) else np.array([])  # threshold range starts
   state = np.r_[0, count[last[:-1]]] if len(f) else np.array([])        # components there
   broken = lower[state != C0]
   component = broken.min() if len(broken) else np.nan

   # target degree: edges and non-isolated nodes above each candidate threshold
   degree = np.nan
   if target_degree is not None:
      u = np.unique(z)
      E = len(z) - np.searchsorted(np.sort(z), u, side = "right")
      N = len(best) - np.searchsorted(np.sort(best), u, side = "right")
      with np.errstate(divide = "ignore", invalid = "ignore"):
         D = np.where(N > 0, (E if net.directed else 2 * E) / N, np.nan)
      hit = u[(D <= target_degree) if D0 >= target_degree else (D >= target_degree)]
      degree = hit.min() if len(hit) else np.nan

   checks = [("first_isolate", isolate, lambda N, E, D, C: N < N0),
      ("first_component_break", component, lambda N, E, D, C: C != C0),
      ("target_degree", degree, lambda N, E, D, C: (D <= target_degree) if D0 >= target_degree else (D >= target_degree))]
   names, thresholds = [], []
   for name, thr, ok in checks[:2 if target_degree is None else 3]:
      if np.isfinite(thr) and sd is not None:   # rounding of score > threshold * sdev
         thr = float(np.float32(thr))
         for _ in range(64):
            if ok(*_bb_stats(s, t, _keep(sc, sd, thr), n, net.directed)):
               break
            thr = float(np.nextafter(np.float32(thr), np.float32(np.inf)))
      names.append(name)
      thresholds.append(thr)
   rows = _bb_rows(arrays, [x if np.isfinite(x) else np.nan for x in thresholds], net.directed)
   table = pd.DataFrame(rows, columns = _COLUMNS)
   table["threshold"] = thresholds
   table.insert(0, "criterion", names)
   return table


def _lexmax(cnt, g):
   """
   Position of the largest ``(cnt, g)`` pair, lexicographically, along the last dimension.

   :param cnt: Primary keys.
   :param g: Secondary keys (same shape).
   :return: Indices (first one on ties).
   """
   g = torch.where(cnt == cnt.max(-1, keepdim = True).values, g, -torch.inf)
   return g.argmax(-1)


def _tmfg(W, P):
   """
   Triangulated maximally filtered graph of a dense weight matrix (Massara et al., 2016).

   :param W: ``(k, k)`` weights, 0 for absent edges, ``k >= 4``.
   :param P: ``(k, k)`` int mask of the edges present (absent ones are used last).
   :return: ``(3k - 6, 2)`` array of the chosen pairs (local indices).
   """
   k = W.shape[0]
   dev = W.device
   mean = W[P > 0].mean() if (P > 0).any() else W.new_tensor(0.)
   start = ((P > 0) & (W > mean)).mul(W).sum(1).argsort(descending = True, stable = True)[:4]
   rem = torch.ones(k, dtype = torch.bool, device = dev)
   rem[start] = False
   faces = torch.empty((2 * k - 4, 3), dtype = torch.long, device = dev)
   faces[:4] = start[torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], device = dev)]
   nf = 4
   pairs = [(x, y) for i, x in enumerate(start.tolist()) for y in start.tolist()[i + 1:]]
   bv = torch.zeros(2 * k - 4, dtype = torch.long, device = dev)   # best vertex of each face, its keys
   bc = torch.full((2 * k - 4,), -1, dtype = P.dtype, device = dev)
   bg = torch.full((2 * k - 4,), -torch.inf, dtype = W.dtype, device = dev)

   def update(f):   # best remaining vertex of faces f
      a, b, c = faces[f].T
      cnt = torch.where(rem, P[a] + P[b] + P[c], -1)
      g = W[a] + W[b] + W[c]
      v = _lexmax(cnt, g)
      bv[f] = v
      bc[f] = cnt.gather(1, v[:, None])[:, 0]
      bg[f] = g.gather(1, v[:, None])[:, 0]

   update(torch.arange(4, device = dev))
   for _ in range(k - 4):
      f = int(_lexmax(bc[:nf], bg[:nf]))
      v = int(bv[f])
      a, b, c = faces[f].tolist()
      pairs += [(v, a), (v, b), (v, c)]
      rem[v] = False
      faces[f:f + 1] = faces.new_tensor([[a, b, v]])
      faces[nf:nf + 2] = faces.new_tensor([[a, c, v], [b, c, v]])
      nf += 2
      stale = torch.nonzero(bv[:nf] == v)[:, 0]   # includes f
      update(torch.cat([stale, torch.tensor([nf - 2, nf - 1], device = dev)]).unique())
   return np.array(pairs)


def triangulated_filtered_graph(net, weight = "weight", kind = "maximum"):
   """
   Triangulated maximally filtered graph (TMFG; Massara et al., 2016) of a network, as a new
   network: a planar subgraph with a large total weight, much faster to build than
   :func:`~graphoskop.maximally_filtered_graph`.

   Starting from the 4 nodes with the largest strength (sum of their weights above the mean
   weight), joined as a tetrahedron, the node joined with the largest total weight to the 3 nodes
   of a triangular face is repeatedly inserted in that face, which it splits into 3 new faces,
   until all nodes are in. The result is a maximal planar graph (a triangulation, :math:`3(n - 2)`
   edges for :math:`n \\geq 3` nodes) made of 4-cliques glued on triangles, so it is chordal; it
   is usually close to the PMFG in total weight. Each step needs an arg-max over the faces and an
   update of the faces that had the inserted node as best candidate, all as tensor operations on
   the device of the network, so a GPU helps for large networks. On a CPU, a complete graph of
   :math:`n` nodes takes about 0.4 seconds for :math:`n = 500` (the PMFG takes minutes) and 13
   seconds for :math:`n = 5000`. Ties are broken by node order.

   The method is designed for complete graphs (e.g. correlation matrices), and works on a dense
   :math:`n \\times n` weight matrix of every component (memory grows as :math:`n^2`). In sparse
   networks the pairs without an edge are used last (a node is inserted where it has the most
   edges to the face, then by weight); they are not in the result, which can then have fewer edges
   and more components than ``net``. Components of 3 or fewer nodes are kept whole.

   As in :func:`~graphoskop.spanning_tree`, edge directions are ignored when choosing the edges and
   kept in the result, only the best of parallel edges can be chosen and self loops never are;
   edges with a NaN weight are treated as absent. All edge and node attributes (numeric and
   qualitative) are kept, and node types in bipartite networks. In multilayer networks each layer
   is filtered on its own (intra-layer edges only), and all interlayer edges are kept.

   :param net: A :class:`~graphoskop.Network`.
   :param weight: Name of the edge attribute used as edge weight (e.g. a correlation or a backbone
      score), or ``None`` for unit weights.
   :param kind: ``"maximum"``, or ``"minimum"`` to favor the smallest weights (e.g. distances).
   :return: The TMFG, a new :class:`~graphoskop.Network`.
   :raises ValueError: If ``kind`` is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("correlations.tsv", device = "cuda")  # complete graph, weight column
      tmfg = gsk.triangulated_filtered_graph(net)
      tmfg.number_of_edges()                                       # 3 * (n - 2)
   """
   if kind not in ("maximum", "minimum"):
      raise ValueError(f"unknown kind: {kind}")
   offset = np.cumsum([0] + [d.num_nodes for d in net.components])
   s, t, w = [], [], []
   for c in range(len(net.components)):
      ei, X = net._input_edges(c)
      ei = ei.cpu().numpy() + offset[c]
      s.append(ei[0])
      t.append(ei[1])
      w.append(np.ones(ei.shape[1]) if weight is None else X[:, net._column(net.edge_attributes, weight, "edge")].double().cpu().numpy())
   eoffset = np.cumsum([0] + [len(x) for x in s])
   s, t, w = np.concatenate(s), np.concatenate(t), np.concatenate(w)
   w = w if kind == "maximum" else -w
   n = int(offset[-1])
   keep = _inter_input(net)
   cand = np.flatnonzero(~keep & (s != t) & ~np.isnan(w))
   lo, hi = np.minimum(s[cand], t[cand]), np.maximum(s[cand], t[cand])
   piece = connected_components(sp.coo_matrix((np.ones(len(cand)), (lo, hi)), shape = (n, n)), directed = False)[1]
   nord = np.argsort(piece, kind = "stable")
   loc = np.empty(n, dtype = np.int64)   # index of each node within its piece, in node order
   start = np.r_[0, np.flatnonzero(np.diff(piece[nord])) + 1]
   loc[nord] = np.arange(n) - np.repeat(start, np.diff(np.r_[start, n]))
   size = np.bincount(piece)
   pc = piece[lo]
   order = np.argsort(pc, kind = "stable") if pc.any() else np.arange(len(pc))
   dev = net.components[0].edge_index.device
   for e in np.split(order, np.flatnonzero(np.diff(pc[order])) + 1):   # edges of each piece
      if not len(e):
         continue
      k = int(size[pc[e[0]]])
      a, b = loc[lo[e]], loc[hi[e]]   # a < b
      key = torch.as_tensor(a * k + b, device = dev)
      W = torch.full((k * k,), -torch.inf, dtype = torch.float64, device = dev)
      W = W.scatter_reduce(0, key, torch.as_tensor(w[cand[e]], device = dev), "amax").view(k, k)   # best of parallel edges
      W = torch.maximum(W, W.T)
      P = torch.isfinite(W).to(torch.int8)
      if k < 4:
         ch = np.array([(x, y) for x in range(k) for y in range(x + 1, k)])
      else:
         ch = np.sort(_tmfg(W.nan_to_num(neginf = 0.), P), 1)
      best = W.cpu().numpy()[a, b]
      sel = np.flatnonzero(np.isin(a * k + b, ch[:, 0] * k + ch[:, 1]) & (w[cand[e]] == best))
      keep[cand[e[sel[np.unique((a * k + b)[sel], return_index = True)[1]]]]] = True   # one edge per pair
   return _subnetwork(net, s, t, keep, eoffset, True)
