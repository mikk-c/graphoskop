"""
Network layouts: positions of the nodes in the plane or in space, to draw a network.

Layout functions store the positions as numeric node attributes ``pos_x``, ``pos_y`` and, in three
dimensions, ``pos_z`` (the prefix ``pos`` can be changed), so that any plotting library can draw
the network from :meth:`~graphoskop.Network.get_node_attribute` or from the node table of
:meth:`~graphoskop.Network.to_pandas_dataframes`. They also return the positions as a
:class:`pandas.DataFrame`.
"""

import inspect
import itertools
import warnings

import numpy as np
import pandas as pd
import torch
from numba import njit

from ._policy import _higher_order
from .network import _quiet_sparse
from .progress import _bar

#: Components with at most this many nodes repel exactly, all pairs, and are laid out together.
_SMALL = 256
#: Most node pairs (squared sizes summed) of the small components laid out together.
_PAIRS = 2 ** 22
#: Number of pivots of the initial layout (pivot multidimensional scaling).
_PIVOTS = 50
#: Nodes with at most this many neighbors closer than this many mean edge lengths are pushed
#: apart.
_FEW, _SPACING = 3, 0.7
_TURN = np.pi / 4   # largest angle between a node and its children in radial layouts
#: Most stress majorization steps per round of overlap removal.
_STEPS = 2
#: Most entries of the temporary tensors of one step (chunks of nodes above it).
_CHUNK = 2 ** 22


def _far_offsets(dimensions):
   """
   Interaction lists of the grid approximation of the repulsion: for each parity pattern of a
   cell (whether its coordinates are odd), the offsets of the cells of the same level that are
   children of the neighbors of its parent but not neighbors of the cell itself.

   :return: Long tensor of shape ``(2 ** dimensions, 6 ** dimensions - 3 ** dimensions, dimensions)``.
   """
   table = []
   for bits in range(2 ** dimensions):
      b = [(bits >> a) & 1 for a in range(dimensions)]
      offs = itertools.product(*[range(-2 - b[a], 4 - b[a]) for a in range(dimensions)])
      table.append([o for o in offs if max(abs(x) for x in o) > 1])
   return torch.tensor(table)


_FAR = {2: _far_offsets(2), 3: _far_offsets(3)}
_NEAR = {D: torch.tensor(list(itertools.product((-1, 0, 1), repeat = D))) for D in (2, 3)}


def _groups(sizes):
   """
   Components laid out together: each large one alone, the small ones in batches whose node
   pairs and pivot matrices stay small.

   :param sizes: Number of nodes of every component.
   :return: List of lists of component indices.
   """
   groups, cur, pairs, P = [], [], 0, 0
   for c, m in enumerate(sizes):
      if m > _SMALL:
         groups.append([c])
         continue
      p = max(P, min(_PIVOTS, m))
      if cur and (pairs + m * m > _PAIRS or (len(cur) + 1) * p * p > _CHUNK):
         groups.append(cur)
         cur, pairs, p = [], 0, min(_PIVOTS, m)
      cur.append(c)
      pairs, P = pairs + m * m, p
   return groups + [cur] * bool(cur)


def _edges(net, comps, weight, device):
   """
   Simple undirected edges of a batch of components, without self loops: parallel and reciprocal
   edges are merged, summing their weights.

   :return: ``u``, ``v`` (batch node indices, ``u < v``), weights (float), node degrees.
   """
   off = np.cumsum([0] + [net.components[c].num_nodes for c in comps]).tolist()
   s = torch.cat([net.components[c].edge_index[0] + o for c, o in zip(comps, off)])
   t = torch.cat([net.components[c].edge_index[1] + o for c, o in zip(comps, off)])
   if weight is None:
      w = torch.ones(len(s), device = device)
   else:
      w = torch.cat([net._numeric_edge(weight, c) for c in comps]).float()
      if torch.isnan(w).any() or (w < 0).any():
         raise ValueError("force_directed_layout: weights must be non negative numbers")
   n, keep = off[-1], s != t
   key, inv = torch.unique(torch.minimum(s, t)[keep] * n + torch.maximum(s, t)[keep], return_inverse = True)
   w = torch.zeros(len(key), device = device).index_add_(0, inv, w[keep])
   w = w / w.mean() if len(w) and w.mean() > 0 else w   # step sizes independent of the weight scale
   u, v = key // n, key % n
   k = torch.bincount(u, minlength = n) + torch.bincount(v, minlength = n)
   return u, v, w, k.float()


def _bfs(n, u, v, sources):
   """
   Hop distances from sources, many searches at once (one per column) with sparse products.

   :param sources: Boolean tensor ``(n, P)``, the sources of each search.
   :return: Float tensor ``(n, P)``; nodes not reached are at distance 0.
   """
   D = torch.zeros(sources.shape, device = sources.device)
   if not len(u):
      return D
   with _quiet_sparse():
      A = torch.sparse_coo_tensor(torch.stack([torch.cat([u, v]), torch.cat([v, u])]), torch.ones(2 * len(u), device = u.device), (n, n)).coalesce().to_sparse_csr()
   seen, F, level = sources.clone(), sources.float(), 0
   while True:
      level += 1
      F = (A @ F > 0) & ~seen
      if not F.any():
         return D
      D[F], seen, F = level, seen | F, F.float()


def _pivot_mds(n, u, v, g, sizes, dimensions, rng):
   """
   Initial layout by pivot multidimensional scaling (Brandes and Pich, 2006): classical scaling of
   the hop distances from random pivots, every component on its own (batched), so that nodes are
   placed as their distances in the network say, with no preferred shape.

   :param g: Component of every node in the batch (components are contiguous).
   :param sizes: Number of nodes of each component of the batch, NumPy array.
   :return: Positions, float tensor ``(n, dimensions)``.
   """
   device, B = u.device, len(sizes)
   P = int(min(_PIVOTS, sizes.max()))
   starts = np.cumsum(sizes) - sizes
   piv = starts + (rng.random((P, B)) * sizes).astype(np.int64)   # random pivots of every component
   src = torch.zeros(n, P, dtype = torch.bool, device = device)
   src[torch.as_tensor(piv.ravel(), device = device), torch.arange(P, device = device).repeat_interleave(B)] = True
   D2 = _bfs(n, u, v, src) ** 2
   cnt = torch.as_tensor(sizes, dtype = torch.float, device = device)
   col = torch.zeros(B, P, device = device).index_add_(0, g, D2) / cnt[:, None]
   C = -0.5 * (D2 - D2.mean(1, keepdim = True) - col[g] + col.mean(1)[g][:, None])   # double centering
   if P < dimensions:
      C = torch.cat([C, torch.zeros(n, dimensions - P, device = device)], 1)
      P = dimensions
   M = torch.zeros(B, P, P, device = device)
   step = max(1, _CHUNK // (P * P))
   for a in range(0, n, step):
      M.index_add_(0, g[a:a + step], C[a:a + step, :, None] * C[a:a + step, None, :])
   V = torch.linalg.eigh(M)[1][:, :, -dimensions:].flip(-1)   # top eigenvectors of C^T C
   return torch.einsum("np,npd->nd", C, V[g])


def _attraction(X, u, v, w, a):
   """
   Forces of the edges: each pulls its ends together with magnitude ``w d^(a - 1)``.
   """
   D = X[v] - X[u]
   d = D.norm(dim = 1, keepdim = True).clamp_min(1e-9)
   f = (w[:, None] * d ** (a - 2)) * D
   return torch.zeros_like(X).index_add_(0, u, f).index_add_(0, v, -f)


def _pair_repulsion(X, i, j, q):
   """
   Exact repulsion over the given node pairs: magnitude ``q / d^2``.
   """
   F = torch.zeros_like(X)
   step = max(1, _CHUNK // X.shape[1])
   for a in range(0, len(i), step):
      ii, jj = i[a:a + step], j[a:a + step]
      D = X[ii] - X[jj]
      f = (q[a:a + step] * (D * D).sum(1).clamp_min(1e-18).rsqrt() ** 3)[:, None] * D
      F.index_add_(0, ii, f).index_add_(0, jj, -f)
   return F


def _grid_repulsion(X, q):
   """
   Repulsion of every node from all others, ``sum_v q_v / d^2`` along the direction away
   from ``v``, approximated on nested grids (a Barnes-Hut scheme on regular cells): nodes in the
   same or adjacent finest cells repel exactly, farther ones as the total mass of their cell at its
   center of mass, at the finest level where the two cells are not adjacent. Linear time in the
   number of nodes, for a fixed depth. Grids are padded with 3 empty cells on every side, so that
   the cells around a node are always at the same offsets of its linear index.

   :param q: Mass of every node.
   :return: Forces, without the mass of the node they act on.
   """
   n, D = X.shape
   dev = X.device
   lo = X.min(0).values
   side = (X.max(0).values - lo).max().clamp_min(1e-9) * (1 + 1e-5)
   top = 22 // D

   def linear(c, m):   # linear indices in the padded grid of m cells per side
      return ((c + 3) * (m + 6) ** torch.arange(D, device = dev)).sum(-1)

   L = max(2, min(top, int(np.ceil(np.log2(max(n, 2)) / D))))
   while True:   # deepen the grid until the exact near field is small
      cell = ((X - lo) / side * 2 ** L).long().clamp(0, 2 ** L - 1)
      lin = linear(cell, 2 ** L)
      cnt = torch.bincount(lin, minlength = (2 ** L + 6) ** D)
      nb = lin[:, None] + linear(_NEAR[D].to(dev), 2 ** L) - linear(torch.zeros(D, dtype = torch.long, device = dev), 2 ** L)
      cc = cnt[nb]
      if L >= top or cc.sum() <= 16 * n:
         break
      L += 1
   order = torch.argsort(lin)   # near field: all pairs of nodes in adjacent cells
   start = (torch.cumsum(cnt, 0) - cnt)[nb].reshape(-1)
   cc = cc.reshape(-1)
   slot = torch.repeat_interleave(torch.arange(len(cc), device = dev), cc)
   within = torch.arange(len(slot), device = dev) - torch.repeat_interleave(torch.cumsum(cc, 0) - cc, cc)
   i, j = slot // nb.shape[1], order[start[slot] + within]
   keep = i != j
   i, j = i[keep], j[keep]
   Dv = X[i] - X[j]
   F = torch.zeros_like(X).index_add_(0, i, (q[j] * (Dv * Dv).sum(1).clamp_min(1e-18).rsqrt() ** 3)[:, None] * Dv)
   K = _FAR[D].shape[1]
   step = max(1, _CHUNK // (K * (D + 1)))
   QX = torch.cat([q[:, None], q[:, None] * X], 1)
   for level in range(2, L + 1):   # far field: cell masses at their centers of mass
      cl, m = cell >> (L - level), 2 ** level
      li = linear(cl, m)
      T = torch.zeros((m + 6) ** D, D + 1, device = dev).index_add_(0, li, QX)
      T[:, 1:] /= T[:, :1].clamp_min(1e-30)
      off = linear(_FAR[D].to(dev), m) - linear(torch.zeros(D, dtype = torch.long, device = dev), m)
      pattern = ((cl & 1) << torch.arange(D, device = dev)).sum(1)
      for a in range(0, n, step):
         C = T[li[a:a + step, None] + off[pattern[a:a + step]]]
         Dv = X[a:a + step, None, :] - C[..., 1:]
         F[a:a + step] += ((C[..., 0] * (Dv * Dv).sum(2).clamp_min(1e-18).rsqrt() ** 3)[..., None] * Dv).sum(1)
   return F


def _close_pairs(X, nodes, g, spacing):
   """
   Pairs of the given nodes in the same component closer than its ``spacing``, found by hashing
   the nodes into cells of the spacing.

   :param nodes: Indices of the nodes.
   :param spacing: Smallest distance in each component.
   :return: The pairs ``(i, j)``, ``i < j``.
   """
   if len(nodes) < 2:
      return nodes[:0], nodes[:0]
   dev, D = X.device, X.shape[1]
   Y = X[nodes] / spacing[g[nodes]][:, None]   # cells of size 1
   c = Y.floor().long()
   c = c - c.min(0).values + 1
   m = int(c.max()) + 2
   key = g[nodes] * m ** D + (c * m ** torch.arange(D, device = dev)).sum(1)
   order = torch.argsort(key)
   sk = key[order]
   nb = key[:, None] + (_NEAR[D].to(dev) * m ** torch.arange(D, device = dev)).sum(1)
   lo, hi = torch.searchsorted(sk, nb), torch.searchsorted(sk, nb, right = True)
   cc = (hi - lo).reshape(-1)
   slot = torch.repeat_interleave(torch.arange(len(cc), device = dev), cc)
   within = torch.arange(len(slot), device = dev) - torch.repeat_interleave(torch.cumsum(cc, 0) - cc, cc)
   i, j = slot // nb.shape[1], order[lo.reshape(-1)[slot] + within]
   keep = (i < j) & ((Y[i] - Y[j]).norm(dim = 1) < 1)
   return nodes[i[keep]], nodes[j[keep]]


def _separate(X, i, j, spacing):
   """
   Push apart the node pairs ``(i, j)`` closer than ``spacing`` (one value per pair), so that
   nodes with few edges do not pile up (leaves of the same hub, above all): each node moves by
   the mean of a quarter of its overlaps.
   """
   Dv = X[i] - X[j]
   d = Dv.norm(dim = 1, keepdim = True).clamp_min(1e-9)
   over = (spacing[:, None] - d).clamp_min(0)
   push = (over / (4 * d)) * Dv
   P = torch.zeros_like(X).index_add_(0, i, push).index_add_(0, j, -push)
   hit = (over[:, 0] > 0).float()
   cnt = torch.zeros(len(X), device = X.device).index_add_(0, i, hit).index_add_(0, j, hit)
   return X + P / cnt.clamp_min(1)[:, None]


def _layout(n, u, v, w, k, g, sizes, dimensions, a, iterations, rng, bar):
   """
   Force-directed layout of a batch of components: pivot multidimensional scaling, then
   ``iterations`` steps along the forces of the edge-repulsion energy model, with the repulsion
   exact over all pairs (small components) or on nested grids (one large component). The
   repulsion constant of each component makes its initial scale (mean edge length 1) a
   stationary point of the energy under rescaling, so that steps are in edge lengths.

   :return: Positions, float tensor ``(n, dimensions)``.
   """
   dev, B = u.device, len(sizes)
   X = _pivot_mds(n, u, v, g, sizes, dimensions, rng)
   ge = g[u]
   ne = torch.bincount(ge, minlength = B).clamp_min(1)
   le = torch.zeros(B, device = dev).index_add_(0, ge, (X[u] - X[v]).norm(dim = 1)) / ne
   X = X / torch.where(le > 0, le, 1.)[g][:, None]   # mean edge length 1
   X = X + torch.as_tensor(rng.normal(0, 0.05, (n, dimensions)), dtype = torch.float, device = dev)
   q = k + 1   # repulsion masses: degree + 1
   eA = torch.zeros(B, device = dev).index_add_(0, ge, w * (X[u] - X[v]).norm(dim = 1) ** a)
   small = bool(sizes.max() <= _SMALL)
   if small:   # all pairs of each component
      end = torch.as_tensor(np.cumsum(sizes), device = dev)[g]
      many = end - torch.arange(n, device = dev) - 1
      i = torch.repeat_interleave(torch.arange(n, device = dev), many)
      j = i + 1 + torch.arange(len(i), device = dev) - torch.repeat_interleave(torch.cumsum(many, 0) - many, many)
      kk = q[i] * q[j]
      eR = torch.zeros(B, device = dev).index_add_(0, g[i], kk / (X[i] - X[j]).norm(dim = 1).clamp_min(1e-9))
      kk = kk * torch.where(eR > 0, eA / eR, 0.)[g[i]]   # repulsion strength: initial scale in equilibrium
      repulsion = lambda Y: _pair_repulsion(Y, i, j, kk)
      pair = (k[i] <= _FEW) & (k[j] <= _FEW)
      si, sj = i[pair], j[pair]
      close = lambda Y, sp: (si, sj)
   else:   # one large component, repulsion constant from sampled pairs
      i = torch.as_tensor(rng.integers(0, n, 2 ** 18), device = dev)
      j = torch.as_tensor(rng.integers(0, n, 2 ** 18), device = dev)
      eR = (q[i] * q[j] / (X[i] - X[j]).norm(dim = 1).clamp_min(1e-9))[i != j].mean() * n * (n - 1) / 2
      qq = q * eA[0] / eR
      repulsion = lambda Y: _grid_repulsion(Y, qq)
      few = torch.nonzero(k <= _FEW)[:, 0]
      close = lambda Y, sp: _close_pairs(Y, few, g, sp)
   for it in range(iterations):   # steps capped by a temperature cooling from 1 to 0.01 edge lengths
      temp = 0.01 ** (it / max(1, iterations - 1))
      le = torch.zeros(B, device = dev).index_add_(0, ge, (X[u] - X[v]).norm(dim = 1)) / ne
      spacing = _SPACING * torch.where(le > 0, le, 1.)
      R = repulsion(X)
      S = (_attraction(X, u, v, w, a) + (R if small else q[:, None] * R)) / q[:, None]
      X = X + S * (temp / S.norm(dim = 1, keepdim = True).clamp_min(temp))
      ci, cj = close(X, spacing)   # nodes with few neighbors kept apart
      X = _separate(X, ci, cj, spacing[g[ci]])
      bar.update(1)
   return X


def _orient(X, g, B):
   """
   Center every component and rotate it onto its principal axes, the widest first.
   """
   dev, D = X.device, X.shape[1]
   cnt = torch.bincount(g, minlength = B).float()[:, None]
   X = X - (torch.zeros(B, D, device = dev).index_add_(0, g, X) / cnt)[g]
   cov = torch.zeros(B, D, D, device = dev).index_add_(0, g, X[:, :, None] * X[:, None, :])
   V = torch.linalg.eigh(cov.double())[1].flip(-1).float()
   return torch.einsum("nd,nde->ne", X, V[g])


def _pack(X, sizes, gap):
   """
   Place the components side by side, in rows (shelves) of about the width of a square holding
   them all, in the given order, left to right and top to bottom.

   :param X: Positions of all nodes, NumPy array, components contiguous and centered.
   :param sizes: Number of nodes of every component.
   :return: The translated positions.
   """
   g = np.repeat(np.arange(len(sizes)), sizes)
   lo = np.full((len(sizes), 2), np.inf)
   hi = np.full((len(sizes), 2), -np.inf)
   np.minimum.at(lo, g, X[:, :2])
   np.maximum.at(hi, g, X[:, :2])
   wh = hi - lo + gap
   width = max(wh[:, 0].max(), np.sqrt((wh[:, 0] * wh[:, 1]).sum()))
   x, y, row, shift = 0., 0., 0., np.zeros((len(sizes), 2))
   for b, (cw, ch) in enumerate(wh.tolist()):
      if x > 0 and x + cw > width:
         x, y, row = 0., y - row, 0.
      shift[b] = x - lo[b, 0], y - hi[b, 1]
      x, row = x + cw, max(row, ch)
   X = X.copy()
   X[:, :2] += shift[g]
   return X


@_higher_order()
def force_directed_layout(net, dimensions = 2, weight = None, iterations = 200, clustering = 0.25, edge_length = 1., seed = None, attribute = "pos", progress = None):
   """
   Force-directed layout in the spirit of organic layouts (such as yFiles' Organic layout): edges
   pull their nodes together, all nodes push each other apart, and the nodes move until the
   forces balance. Groups of nodes sharing many edges end up together, apart from the rest of
   the network; the shape of the drawing follows the network, not a circle or a sphere; nodes
   with one or few edges stay close to their neighbors.

   The layout minimizes the energy of the edge-repulsion model of Noack (2007),

   .. math::

      U = \\sum_{(u, v) \\in E} w_{uv} \\frac{d_{uv}^a}{a} + C \\sum_{u < v}
      \\frac{(k_u + 1)(k_v + 1)}{d_{uv}}, \\qquad a = 2 - \\mathrm{clustering},

   where :math:`d_{uv}` is the distance between two nodes, :math:`w_{uv}` the weight of their
   edge (1 without ``weight``) and :math:`k_u` the number of neighbors of :math:`u`. Three
   choices set its look apart from classical spring embedders (Fruchterman and Reingold):

   - Repulsion is weighted by degree (plus one), so every node is pushed away in proportion to
     the edges pulling it back: a leaf is not shot away from its only neighbor by the whole
     network, and hubs, with all their neighbors, keep their distances.
   - The repulsion force decays as :math:`1 / d^2`, faster than :math:`1 / d`, and there is no
     gravity towards the center and no frame: far away parts of the network do not press the
     drawing into a disk, its shape comes from the edges.
   - The attraction exponent :math:`a` sets how strongly groups separate: with :math:`a = 2`
     (``clustering = 0``, springs as in ForceAtlas2) edges have similar lengths and groups
     touch; with :math:`a = 1` (``clustering = 1``, the LinLog attraction of Noack) densely
     connected groups shrink and move apart, joined by long edges, which shows communities best
     but makes trees spiky. Long edges are the edges between groups, stretched by design: on a
     network of 24,880 nodes with 22 communities, 84% of the edges longer than 5 times the median
     joined two communities. The default, 0.25, came from that network: compared with 0.5, it
     halves those long edges (from 3.4% to 1.9% of all edges) while the 10 nearest nodes of a
     node are still 71% in its community (75% with 0.5, 67% with 0). Neither the degree
     weighting of the repulsion nor weighting edges by their common neighbors gave a better
     balance, and the layout does not depend on the seed or on a hotter start.

   Each component is laid out on its own. It starts from pivot multidimensional scaling (Brandes
   and Pich, 2006: classical scaling of the hop distances from 50 random pivots), which already
   places nodes as their distances in the network say, then takes ``iterations`` steps along the
   forces divided by degree plus one, each capped by a temperature cooling from 1 to 0.01 mean
   edge lengths. :math:`C` makes the starting scale a stationary point of the energy. After each
   step, nodes with at most 3 neighbors closer than 0.7 mean edge lengths to another such node of
   their component are pushed apart, so that leaves around a hub do not pile up. Repulsion is
   exact, over all pairs, in components of at most 256 nodes (laid out together, in batches);
   in larger ones it is approximated on nested grids, Barnes-Hut style: nodes in the same or
   adjacent cells of the finest grid repel exactly, farther ones as the total degree of their
   cell at its center of mass (errors about 0.1% of the mean force on test layouts), in time
   linear in the number of nodes. Everything runs on the device of the network.

   Finally every component is rotated onto its principal axes (its widest extent along
   ``pos_x``), scaled to a mean edge length of ``edge_length``, and the components are packed
   in rows, largest first, two edge lengths apart; isolated nodes come last. Directed networks
   are laid out as undirected, self loops are ignored, parallel and reciprocal edges merged
   (weights summed). In multilayer networks every copy of a node is a node of its own, held
   near its other copies by the interlayer edges.

   Runtime, with the default 200 iterations on two CPU threads: about 2.5 s for 1,000 nodes,
   20 s for 10,000 and 110 s for 50,000 (LFR benchmark graphs, average degree 15). Time grows
   about linearly with nodes, edges and iterations; large networks are best laid out on a GPU.

   :param net: A :class:`~graphoskop.Network`.
   :param dimensions: 2 (``pos_x``, ``pos_y``) or 3 (also ``pos_z``).
   :param weight: Name of a numeric edge attribute: edges with larger weights pull harder (weights
      are rescaled to mean 1), or ``None``.
   :param iterations: Number of steps after the initial layout.
   :param clustering: Between 0 and 1: how strongly densely connected groups are kept together
      and apart from the rest (see above).
   :param edge_length: Mean length of the edges in the result.
   :param seed: Seed of the random pivots and of the small initial perturbation, for reproducible
      layouts.
   :param attribute: Prefix of the numeric node attributes ``<attribute>_x``, ``<attribute>_y``
      (and ``<attribute>_z``) where to store the positions, or ``None`` to only return them.
   :param progress: Whether to show a progress bar (see :doc:`progress`); ``None`` uses the
      global setting.
   :return: :class:`pandas.DataFrame` with one row per node, indexed by node id, in the network
      wide node order, and one column per dimension (named as the attributes, or ``x``, ``y``,
      ``z``).
   :raises ValueError: If ``dimensions`` is not 2 or 3, ``clustering`` is not between 0 and 1,
      ``iterations`` is negative, or weights are negative or missing.

   Example:

   .. code-block:: python

      import graphoskop as gsk
      import matplotlib.pyplot as plt

      net = gsk.load_network("edges.tsv")
      pos = gsk.force_directed_layout(net, seed = 0)    # also node attributes pos_x, pos_y
      edges = [(pos.loc[a], pos.loc[b]) for a, b in net.edges(None, labels = True)]
      for a, b in edges:
         plt.plot([a.pos_x, b.pos_x], [a.pos_y, b.pos_y], color = "gray", lw = 0.5, zorder = 1)
      plt.scatter(pos.pos_x, pos.pos_y, s = 10, zorder = 2)
      plt.gca().set_aspect("equal")

      gsk.force_directed_layout(net, clustering = 1)    # communities further apart
      gsk.force_directed_layout(net, 3, attribute = "pos3d")   # pos3d_x, pos3d_y, pos3d_z
   """
   if dimensions not in (2, 3):
      raise ValueError(f"dimensions must be 2 or 3, got {dimensions}")
   if not 0 <= clustering <= 1:
      raise ValueError(f"clustering must be between 0 and 1, got {clustering}")
   if iterations < 0:
      raise ValueError(f"iterations must not be negative, got {iterations}")
   dev = net.components[0].edge_index.device
   rng = np.random.default_rng(seed)
   sizes = np.diff(net.node_offsets)
   groups = _groups(sizes.tolist())
   out = []
   with _bar(iterations * len(groups), "force_directed_layout", progress) as bar:
      for comps in groups:
         sz = sizes[comps]
         n = int(sz.sum())
         u, v, w, k = _edges(net, comps, weight, dev)
         g = torch.repeat_interleave(torch.arange(len(comps), device = dev), torch.as_tensor(sz, device = dev))
         X = _layout(n, u, v, w, k, g, sz, dimensions, 2 - clustering, iterations, rng, bar)
         le = torch.zeros(len(comps), device = dev).index_add_(0, g[u], (X[u] - X[v]).norm(dim = 1))
         le = le / torch.bincount(g[u], minlength = len(comps)).clamp_min(1)
         out.append(_orient(X * (edge_length / torch.where(le > 0, le, 1.))[g][:, None], g, len(comps)))
   return _store(net, _pack(torch.cat(out).cpu().double().numpy(), sizes, 2 * edge_length), attribute)


def _store(net, X, attribute):
   """
   Positions as a :class:`pandas.DataFrame` indexed by node id, stored as the numeric node
   attributes ``<attribute>_x``, ``<attribute>_y`` (and ``<attribute>_z``) unless ``attribute`` is
   ``None``.

   :param X: NumPy array, one row per node in the network wide order.
   """
   axes = "xyz"[:X.shape[1]]
   cols = [f"{attribute}_{x}" for x in axes] if attribute is not None else list(axes)
   if attribute is not None:
      T = torch.as_tensor(X, dtype = torch.float, device = net.components[0].edge_index.device)
      for j, name in enumerate(cols):
         net.set_node_attribute(name, [T[o:o2, j] for o, o2 in zip(net.node_offsets[:-1], net.node_offsets[1:])], None)
   ids = [x for c in range(len(net.components)) for x in net.node_ids[c].tolist()]
   return pd.DataFrame(X, index = pd.Index(ids, tupleize_cols = False, name = "node"), columns = cols)


def _vectors(net, attributes, embedding, seed):
   """
   Node vectors to lay out: the given numeric node attributes (names, or the prefix of
   ``<prefix>_0``, ``<prefix>_1``, ...), or the output of an embedding function.

   :return: Float tensor, one row per node in the network wide order, on the device of the network.
   """
   dev = net.components[0].edge_index.device
   if embedding is not None:
      if isinstance(embedding, str):
         import graphoskop
         f = getattr(graphoskop, embedding)
         X = f(net, seed = seed) if "seed" in inspect.signature(f).parameters else f(net)
      else:
         X = embedding(net)
      X = torch.as_tensor(np.array(X, dtype = np.float32), device = dev)   # a writable copy (pandas tables give read-only arrays)
   else:
      if isinstance(attributes, str):
         names = [a for a in net.node_attributes if a == attributes] or [a for a in net.node_attributes if a.startswith(attributes + "_")]
         if not names:
            raise KeyError(f"no numeric node attribute {attributes!r} or {attributes + '_*'!r}")
      else:
         names = list(attributes)
      X = torch.stack([torch.cat(net.get_node_attribute(a, None)) for a in names], 1).float()
   if X.dim() != 2 or len(X) != net.number_of_nodes():
      raise ValueError(f"umap_layout: expected one vector per node ({net.number_of_nodes()}), got shape {tuple(X.shape)}")
   if torch.isnan(X).any():
      raise ValueError("umap_layout: node vectors have missing values (NaN)")
   return X


def _knn(X, k, metric):
   """
   Exact nearest neighbors of every row, in chunks of rows.

   :param metric: ``"euclidean"`` or ``"cosine"`` (one minus the cosine similarity).
   :return: Indices and distances of the ``k`` nearest other rows, nearest first.
   """
   if metric == "cosine":
      X = X / X.norm(dim = 1, keepdim = True).clamp_min(1e-12)
   elif metric != "euclidean":
      raise ValueError(f"metric must be 'euclidean' or 'cosine', got {metric!r}")
   n, out = len(X), []
   step = max(1, _CHUNK // n)
   for a in range(0, n, step):
      D = torch.cdist(X[a:a + step], X)
      D[torch.arange(len(D)), torch.arange(a, a + len(D))] = float("inf")
      d, i = D.topk(k, largest = False)
      out.append((i, d ** 2 / 2 if metric == "cosine" else d))
   return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])


def _fuzzy(idx, d):
   """
   Fuzzy neighbor graph of UMAP (McInnes et al., 2018): every node is joined to its neighbors with
   memberships ``exp(-(d - rho) / sigma)``, where ``rho`` is the distance to the nearest neighbor
   and ``sigma`` (by bisection) makes the memberships of each node sum to ``log2(k + 1)`` (the
   node counts as one of its own ``k + 1`` neighbors, as in ``umap-learn``); the two
   directions of a pair are merged by the fuzzy union ``1 - (1 - w_uv)(1 - w_vu)``.

   :return: ``u``, ``v`` (``u < v``) and memberships of the undirected edges.
   """
   n, k = d.shape
   dev = d.device
   rho = torch.where(d > 0, d, float("inf")).min(1).values
   rho = torch.where(torch.isinf(rho), 0., rho)[:, None]
   lo, hi = torch.zeros(n, 1, device = dev), torch.full((n, 1), float("inf"), device = dev)
   sigma = torch.ones(n, 1, device = dev)
   for _ in range(64):
      big = torch.exp(-(d - rho).clamp_min(0) / sigma).sum(1, keepdim = True) > np.log2(k + 1)
      hi = torch.where(big, sigma, hi)
      lo = torch.where(big, lo, sigma)
      sigma = torch.where(torch.isinf(hi), sigma * 2, (lo + hi) / 2)
   sigma = torch.maximum(sigma, 1e-3 * d.mean(1, keepdim = True))
   w = torch.exp(-(d - rho).clamp_min(0) / sigma.clamp_min(1e-12)).reshape(-1)
   s, t = torch.arange(n, device = dev).repeat_interleave(k), idx.reshape(-1)
   key, inv = torch.unique(torch.minimum(s, t) * n + torch.maximum(s, t), return_inverse = True)
   miss = torch.zeros(len(key), device = dev).index_add_(0, inv, torch.log1p(-w.clamp_max(1 - 1e-7)))
   return key // n, key % n, 1 - torch.exp(miss)


def _ab(min_dist, spread):
   """
   Parameters ``a``, ``b`` of the UMAP similarity ``1 / (1 + a d^(2b))`` in the layout, fit (least
   squares) to 1 up to ``min_dist`` and ``exp(-(d - min_dist) / spread)`` beyond.
   """
   from scipy.optimize import curve_fit
   x = np.linspace(0, 3 * spread, 300)
   y = np.where(x < min_dist, 1., np.exp(-(x - min_dist) / spread))
   return curve_fit(lambda x, a, b: 1 / (1 + a * x ** (2 * b)), x, y)[0].tolist()


def _graph_start(n, u, v, dimensions, rng, device):
   """
   Initial layout of a graph: pivot multidimensional scaling of each connected component,
   components packed side by side (as :func:`force_directed_layout` does), scaled to a width of 20.

   :return: Positions, float tensor ``(n, dimensions)``.
   """
   from scipy.sparse import coo_matrix
   from scipy.sparse.csgraph import connected_components
   lab = connected_components(coo_matrix((np.ones(len(u)), (u.cpu().numpy(), v.cpu().numpy())), shape = (n, n)), directed = False)[1]
   sizes = np.bincount(lab)
   rank = np.argsort(np.argsort(-sizes, kind = "stable"))   # components by decreasing size
   order = np.argsort(rank[lab], kind = "stable")
   sizes = np.sort(sizes)[::-1].copy()
   new = torch.as_tensor(np.argsort(order), device = device)   # old node -> sorted position
   su, sv = new[u], new[v]
   gl = torch.as_tensor(np.repeat(np.arange(len(sizes)), sizes), device = device)
   out, start = [], 0
   for comps in _groups(sizes.tolist()):
      m = int(sizes[comps].sum())
      keep = (su >= start) & (su < start + m)
      g = gl[start:start + m] - comps[0]
      X = _pivot_mds(m, su[keep] - start, sv[keep] - start, g, sizes[comps], dimensions, rng)
      ge = g[su[keep] - start]
      le = torch.zeros(len(comps), device = device).index_add_(0, ge, (X[su[keep] - start] - X[sv[keep] - start]).norm(dim = 1))
      le = le / torch.bincount(ge, minlength = len(comps)).clamp_min(1)
      out.append(_orient(X / torch.where(le > 0, le, 1.)[g][:, None], g, len(comps)))
      start += m
   X = _pack(torch.cat(out).cpu().double().numpy(), sizes, 2.)
   X = X - X.mean(0)
   X = 20 * X / max(np.abs(X).max() * 2, 1e-12)
   return torch.as_tensor(X[new.cpu().numpy()], dtype = torch.float, device = device)


def _umap(X, u, v, w, epochs, a, b, rng, bar, anchors = None, alignment = 0., negatives = 5):
   """
   UMAP optimization: in every epoch each edge is sampled with probability proportional to its
   membership and pulls its two ends together; each sampled edge also pushes its first end away
   from ``negatives`` random nodes. All the moves of an epoch are applied at once (as in GPU
   implementations), each clipped at 4 per coordinate, and so is their sum for every node: a hub
   gets thousands of moves per epoch, which, summed unchecked, could throw it far away, where
   UMAP's attraction (decaying with distance) cannot bring it back. The learning rate decreases
   from 1 to 0. Nodes with anchors are also pulled towards them, by ``alignment`` times the
   learning rate times their distance (as the regularization of aligned UMAP).

   :param X: Initial positions.
   :param anchors: Anchor of every node (NaN for none), or ``None``.
   :return: Final positions.
   """
   n, dev = len(X), X.device
   keep = w >= w.max() / epochs   # edges sampled less than once in all epochs are dropped
   s, t, p = torch.cat([u[keep], v[keep]]), torch.cat([v[keep], u[keep]]), (w[keep] / w.max()).repeat(2)
   gen = torch.Generator(device = dev).manual_seed(int(rng.integers(2 ** 62)))
   if anchors is not None:
      held = ~torch.isnan(anchors[:, 0])
      anchors = torch.nan_to_num(anchors)
   for e in range(epochs):
      lr = 1 - e / epochs
      on = torch.rand(len(p), generator = gen, device = dev) < p
      i, j = s[on], t[on]
      D = X[i] - X[j]
      d2 = (D * D).sum(1, keepdim = True)
      pull = (-2 * a * b * d2.clamp_min(1e-12) ** (b - 1) / (1 + a * d2 ** b) * D).clamp(-4, 4) * lr
      F = torch.zeros_like(X).index_add_(0, i, pull).index_add_(0, j, -pull)
      i = i.repeat(negatives)
      k = torch.randint(0, n, (len(i),), generator = gen, device = dev)
      D = X[i] - X[k]
      d2 = (D * D).sum(1, keepdim = True)
      push = (2 * b / ((0.001 + d2) * (1 + a * d2 ** b)) * D).clamp(-4, 4) * lr
      F.index_add_(0, i, push * (i != k)[:, None])
      X = X + F.clamp(-4 * lr, 4 * lr)   # a node moves at most as much as one clipped move
      if anchors is not None:
         X = X - (alignment * lr) * held[:, None] * (X - anchors)
      bar.update(1)
   return X


def _anchors(net, initial, dimensions):
   """
   Previous positions of the nodes: numeric node attributes ``<initial>_x``, ``<initial>_y`` (and
   ``<initial>_z``), or a table indexed by node id with one column per dimension.

   :return: Float tensor ``(n, dimensions)``, network wide node order, NaN where a node has none.
   """
   dev = net.components[0].edge_index.device
   if isinstance(initial, str):
      A = torch.stack([torch.cat(net.get_node_attribute(f"{initial}_{x}", None)) for x in "xyz"[:dimensions]], 1).float()
      return A.to(dev)
   T = pd.DataFrame(initial)
   if T.shape[1] != dimensions:
      raise ValueError(f"umap_layout: initial positions have {T.shape[1]} columns, expected {dimensions}")
   ids = pd.Index([x for c in range(len(net.components)) for x in net.node_ids[c].tolist()], tupleize_cols = False)
   if len(ids) and len(T) and isinstance(ids[0], str) and not isinstance(T.index[0], str):
      T.index = T.index.map(str)   # ids read from files are text (as in load_network)
   return torch.as_tensor(np.array(T.reindex(ids), dtype = np.float32), device = dev)


def _place(X0, A, u, v, rng):
   """
   Starting positions with previous positions: nodes that have one start there, the others at the
   mean of their already placed neighbors (spreading out from the placed nodes one hop per round,
   with a small random offset); nodes not connected to any placed node keep their positions in
   ``X0``, moved beside the placed ones.
   """
   placed = ~torch.isnan(A[:, 0])
   if not placed.any():
      return X0
   X = torch.nan_to_num(A)
   s, t = torch.cat([u, v]), torch.cat([v, u])
   while True:
      m = placed[t] & ~placed[s]
      if not m.any():
         break
      S = torch.zeros_like(X).index_add_(0, s[m], X[t[m]])
      c = torch.bincount(s[m], minlength = len(X))
      new = c > 0
      X[new] = S[new] / c[new][:, None] + torch.as_tensor(rng.normal(0, 0.05, (int(new.sum()), X.shape[1])), dtype = X.dtype, device = X.device)
      placed = placed | new
   if not placed.all():   # beside the placed nodes, along the first axis
      R = X0[~placed]
      shift = X[placed, 0].max() - R[:, 0].min() + 2
      X[~placed] = R + torch.cat([shift[None], (X[placed, 1:].mean(0) - R[:, 1:].mean(0))])
   return X


@_higher_order()
def umap_layout(net, attributes = None, embedding = None, dimensions = 2, n_neighbors = 15, min_dist = 0.1, metric = "euclidean", weight = None, epochs = None, seed = None, attribute = "pos", progress = None, initial = None, alignment = 0.1):
   """
   Layout by UMAP (uniform manifold approximation and projection, McInnes, Healy and Melville,
   2018): nodes with similar vectors end up close, and groups of similar nodes form separate
   clusters. The vectors can be numeric node attributes, node embeddings learned from the
   network, or the network itself:

   - ``attributes`` given: the vectors are those node attributes (for instance an embedding
     already stored with ``attribute = "emb"`` by :func:`~graphoskop.node2vec`, or any numeric
     node attributes);
   - ``embedding`` given: the vectors are computed first, by a node embedding function of
     graphoskop given by name (``"node2vec"``, ``"laplacian_eigenmaps"``, ``"gae"``, ...: called
     with its defaults and ``seed``) or by any function taking the network and returning one row
     per node (a :class:`pandas.DataFrame`, array or tensor), for instance
     ``lambda net: gsk.node2vec(net, dimensions = 32, q = 2)``;
   - neither: UMAP runs on the adjacency matrix itself, taking the edges of the network as its
     neighbor graph (memberships proportional to ``weight``, the largest being 1), which makes it a
     force-directed layout with UMAP's attraction and repulsion.

   With vectors, the neighbor graph links every node to its ``n_neighbors - 1`` nearest nodes
   (the node itself counts as a neighbor, as in ``umap-learn``; exact search, in chunks, on the
   device of the network: quadratic in the number of nodes),
   with memberships :math:`\\exp(-(d - \\rho) / \\sigma)`, where :math:`\\rho` is the distance to the
   nearest neighbor and :math:`\\sigma` makes the memberships of each node sum to
   :math:`\\log_2` ``n_neighbors``; the two directions of a pair are merged by the fuzzy union
   :math:`w_{uv} + w_{vu} - w_{uv} w_{vu}`.

   The layout starts from pivot multidimensional scaling of each connected component of the
   neighbor graph, the components packed side by side (as in :func:`force_directed_layout`),
   then follows UMAP's optimization: in every epoch each edge is sampled with probability equal to
   its membership and pulls its ends together with the gradient of
   :math:`\\log(1 + a d^{2b})`, and each sampled edge pushes its first end away from 5 random
   nodes; :math:`a` and :math:`b` are fit so that similarity in the layout is about 1 up to
   ``min_dist`` and decays exponentially beyond. All the moves of an epoch are applied at once,
   as GPU implementations do (the reference implementation moves nodes one edge at a time), so
   results agree with the ``umap-learn`` package in quality, not point by point. The total move
   of a node in an epoch is clipped as a single move is (4 per coordinate, times the learning
   rate): otherwise the thousands of moves of a hub, summed, could throw it far from the rest,
   where UMAP's attraction, which weakens with distance, cannot bring it back.

   **Aligned layouts** (``initial``): to follow a network that changes over time, give the
   positions of the previous snapshot, as in aligned UMAP. Nodes that have one start there and,
   at every epoch, are pulled back towards it by ``alignment`` times their distance from it (times
   the learning rate), so they move only as much as the new edges and nodes require; new nodes
   start at the mean of their neighbors that have a position (spreading one hop per round, with
   a small random offset), and parts of the network with no such node start beside the old
   layout. On a 2,000-node network with 10% of its edges rewired and 300 new nodes, old nodes moved
   by a median 33% of the width of the layout when laid out again from scratch, 16% with
   ``alignment = 0.01``, 5.5% with 0.1 and 2% with 1, groups staying as compact. Previous
   positions should be in the scale of UMAP layouts (as given by this function).

   Unlike :func:`force_directed_layout`, distances between clusters carry little meaning (groups
   with few edges to the rest can drift far from it), and positions are left as UMAP produces
   them (no rescaling or packing at the end).

   :param net: A :class:`~graphoskop.Network`.
   :param attributes: Names of numeric node attributes, or a prefix ``p`` selecting the attribute
      ``p`` or the attributes ``p_0``, ``p_1``, ... (as stored by the embedding functions); ``None``
      to use ``embedding`` or the adjacency matrix.
   :param embedding: Name of a node embedding function of graphoskop, or a function of the
      network returning one vector per node in the network wide node order; ignored if
      ``attributes`` is given.
   :param dimensions: 2 (``pos_x``, ``pos_y``) or 3 (also ``pos_z``).
   :param n_neighbors: Number of neighbors of every node in the neighbor graph of the vectors:
      smaller values keep finer local structure, larger ones more of the global structure.
   :param min_dist: Smallest distance between nodes in the layout, roughly: smaller values pack
      clusters tighter.
   :param metric: Distance between vectors, ``"euclidean"`` or ``"cosine"``.
   :param weight: Without vectors, name of a numeric edge attribute giving the strength of the
      edges, or ``None``.
   :param epochs: Number of epochs; ``None`` is 500 for networks up to 10,000 nodes, 200 above
      (as ``umap-learn``).
   :param seed: Seed of the pivots, of the edge sampling and of the negative samples (and of the
      embedding function).
   :param attribute: Prefix of the numeric node attributes where to store the positions, or
      ``None`` to only return them.
   :param progress: Whether to show a progress bar (see :doc:`progress`); ``None`` uses the
      global setting.
   :param initial: Previous positions: the prefix of numeric node attributes ``<initial>_x``,
      ``<initial>_y`` (and ``<initial>_z``; missing values for nodes without one), or a table
      indexed by node id with one column per dimension (such as the output of this function on the
      previous snapshot; ids not in the network are ignored); ``None`` for a new layout.
   :param alignment: With ``initial``, how strongly nodes are held at their previous positions:
      0 only starts them there, larger values move them less.
   :return: :class:`pandas.DataFrame` with one row per node, indexed by node id, in the network
      wide node order, and one column per dimension.
   :raises ValueError: If ``dimensions`` is not 2 or 3, the vectors do not have one row per node
      or have missing values, the metric is unknown, or ``initial`` does not have one column per
      dimension.
   :raises KeyError: If an attribute does not exist.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.umap_layout(net)                                   # UMAP of the adjacency matrix
      gsk.umap_layout(net, embedding = "node2vec", seed = 0) # UMAP of node2vec vectors
      gsk.node2vec(net, 64, attribute = "emb")
      gsk.umap_layout(net, "emb", metric = "cosine")         # stored vectors emb_0 ... emb_63
      gsk.umap_layout(net, ["age", "income"], attribute = "pos_attr")

      pos = gsk.umap_layout(gsk.load_network("edges_2024.tsv"), seed = 0)
      net = gsk.load_network("edges_2025.tsv")
      gsk.umap_layout(net, initial = pos, seed = 0)          # aligned with the 2024 layout
   """
   if dimensions not in (2, 3):
      raise ValueError(f"dimensions must be 2 or 3, got {dimensions}")
   dev = net.components[0].edge_index.device
   rng = np.random.default_rng(seed)
   n = net.number_of_nodes()
   if attributes is None and embedding is None:
      u, v, w, _ = _edges(net, list(range(len(net.components))), weight, dev)
   elif n < 2:
      u = v = torch.zeros(0, dtype = torch.long, device = dev)
      w = torch.zeros(0, device = dev)
   else:
      X = _vectors(net, attributes, embedding, seed)
      idx, d = _knn(X, max(1, min(n_neighbors - 1, n - 1)), metric)
      u, v, w = _fuzzy(idx, d)
   epochs = (500 if n <= 10000 else 200) if epochs is None else epochs
   a, b = _ab(min_dist, 1.)
   with _bar(epochs, "umap_layout", progress) as bar:
      X = _graph_start(n, u, v, dimensions, rng, dev)
      A = None if initial is None else _anchors(net, initial, dimensions)
      if A is not None:
         X = _place(X, A, u, v, rng)
      if len(w) and epochs > 0:
         X = _umap(X, u, v, w, epochs, a, b, rng, bar, A, alignment)
   return _store(net, X.cpu().double().numpy(), attribute)


def _codes(net, name):
   """
   Codes of the distinct values of a node attribute, numbered in sorted order of the values
   (numbers numerically, text alphabetically, mixed values as text), missing values last.

   :return: NumPy integer array, network wide node order.
   """
   vals = net.get_node_attribute(name, None)
   if name in net.qualitative_node_attributes:
      v = pd.Series(np.concatenate([np.asarray(x, dtype = object) for x in vals]), dtype = object)
   else:
      v = pd.Series(torch.cat(vals).cpu().numpy())
   try:
      codes, uniq = pd.factorize(v, sort = True)
   except TypeError:   # values that do not compare (text and numbers, several values per node)
      codes, uniq = pd.factorize(v.where(v.isna(), v.map(str)), sort = True)
   return np.where(codes < 0, len(uniq), codes)


def _all_edges(net, weight):
   """
   Edges of all components in network wide node indices, in both directions, with their weights
   (1 without ``weight``).

   :return: NumPy arrays ``s``, ``t``, ``w``.
   """
   off = net.node_offsets
   s = np.concatenate([d.edge_index[0].cpu().numpy() + o for d, o in zip(net.components, off)])
   t = np.concatenate([d.edge_index[1].cpu().numpy() + o for d, o in zip(net.components, off)])
   w = np.ones(len(s)) if weight is None else torch.cat(net._numeric_edge(weight, None)).double().cpu().numpy()
   return np.concatenate([s, t]), np.concatenate([t, s]), np.concatenate([w, w])


def _consecutive(keys):
   """
   Ids ``0, 1, ...`` of the distinct rows of ``keys``, in lexicographic order of the rows.
   """
   perm = np.lexsort(keys.T[::-1])
   change = np.concatenate([[True], (keys[perm][1:] != keys[perm][:-1]).any(1)])
   ids = np.empty(len(keys), dtype = np.int64)
   ids[perm] = np.cumsum(change) - 1
   return ids


def _arc_cost(orders, W, size, a, c, cyclic):
   """
   Cost of arrangements of groups along an arc: the weight of every pair of groups times the
   distance between the centers of their arcs (the shorter way around, on a full circle), plus
   the weight to the previous sector times the distance from the start and the weight to the next
   sector times the distance from the end.

   :param orders: Arrangements, integer array ``(P, k)`` (group at each place).
   :return: Cost of each arrangement.
   """
   sz = size[orders]
   pos = np.take_along_axis(np.cumsum(sz, 1) - sz / 2, np.argsort(orders, 1), 1)   # center of each group
   S = size.sum()
   d = np.abs(pos[:, :, None] - pos[:, None, :])
   if cyclic:
      d = np.minimum(d, S - d)
   return (W * d).sum((1, 2)) / 2 + (pos * a).sum(1) + ((S - pos) * c).sum(1)


def _arrange(W, size, a, c, cyclic):
   """
   Order of groups along an arc (a full circle if ``cyclic``) placing groups with many edges
   between them close, and groups with many edges to the previous (next) sector near its start
   (end): exhaustive for up to 7 groups, else spectral sequencing (Fiedler vector, or angle in the
   two-dimensional spectral embedding on a circle) improved by moving one group at a time to the
   place that lowers the cost most (up to 50 groups; swaps of neighboring groups up to 200).
   Without any edge, the given order.

   :param W: Weights between groups, symmetric ``(k, k)``.
   :param size: Number of nodes of each group.
   :param a: Weights of each group to the previous sector; ``c``: to the next one.
   :return: Integer array, the groups in their order.
   """
   k = len(size)
   if W.sum() + a.sum() + c.sum() == 0:
      return np.arange(k)
   if k <= 7:
      first = [0] if cyclic else []
      rest = [i for i in range(k) if i not in first]
      orders = np.array([first + list(p) for p in itertools.permutations(rest)])
      return orders[np.argmin(_arc_cost(orders, W, size, a, c, cyclic))]
   if cyclic:
      M = W
   else:   # the previous and next sectors as two more groups
      M = np.zeros((k + 2, k + 2))
      M[:k, :k], M[:k, k], M[k, :k], M[:k, k + 1], M[k + 1, :k] = W, a, a, c, c
   deg = M.sum(1)
   h = 1 / np.sqrt(np.where(deg > 0, deg, 1))
   vec = np.linalg.eigh(np.eye(len(M)) - h[:, None] * M * h[None, :])[1]
   if cyclic:
      order = np.argsort(np.arctan2(vec[:, 2], vec[:, 1]))
      order = np.roll(order, -int(np.argmax(order == 0)))
      if order[-1] < order[1]:
         order = np.concatenate([[0], order[1:][::-1]])
   else:
      f = vec[:, 1] * (1 if vec[k, 1] <= vec[k + 1, 1] else -1)   # previous sector first
      order = np.argsort(f[:k], kind = "stable")
   best = _arc_cost(order[None], W, size, a, c, cyclic)[0]
   for _ in range(20 if k <= 200 else 0):   # best move of one group to another place, while it helps
      if k <= 50:
         cands = np.array([np.insert(np.delete(order, i), j, order[i]) for i in range(k) for j in range(k) if i != j])
      else:   # swaps of neighbors only
         cands = np.array([np.concatenate([order[:i], order[i + 1:i + 2], order[i:i + 1], order[i + 2:]]) for i in range(k - 1)])
      cost = _arc_cost(cands, W, size, a, c, cyclic)
      if cost.min() >= best - 1e-12:
         break
      order, best = cands[np.argmin(cost)], cost.min()
   if cyclic:   # start from group 0, in the direction of the smaller neighbor
      order = np.roll(order, -int(np.argmax(order == 0)))
      if order[-1] < order[1]:
         order = np.concatenate([[0], order[1:][::-1]])
   return order


def _arrange_groups(G, comp, s, t, w, cyclic = True):
   """
   Arrangement of the groups of every level inside the sectors of the level above (around the
   whole circle at the first level), from the edges between them and to the adjacent sectors.

   :param G: Group codes, one column per level.
   :param cyclic: Whether the groups of a component go around a circle (else along a line: the
      first and last groups have no previous and next neighbors).
   :return: Codes of the same groups, renumbered so that their sorted order is the arrangement.
   """
   G = G.copy()
   for l in range(G.shape[1]):
      pid = _consecutive(np.column_stack([comp, G[:, :l]]))   # parent sector, numbered around the circles
      P = pid.max() + 1
      pc = np.zeros(P, dtype = np.int64)
      pc[pid] = comp
      lo = np.full(comp.max() + 1, P)
      np.minimum.at(lo, pc, np.arange(P))
      hi = np.full(comp.max() + 1, -1)
      np.maximum.at(hi, pc, np.arange(P))
      p = np.arange(P)
      prev = np.where(p > lo[pc], p - 1, hi[pc] if cyclic else -1)
      nxt = np.where(p < hi[pc], p + 1, lo[pc] if cyclic else -1)
      alone = lo[pc] == hi[pc]   # a single sector: no neighbors
      pair = _consecutive(np.column_stack([pid, G[:, l]]))   # child group, numbered parent by parent
      Np = pair.max() + 1
      pp = np.zeros(Np, dtype = np.int64)
      pp[pair] = pid
      code = np.zeros(Np, dtype = np.int64)
      code[pair] = G[:, l]
      size = np.bincount(pair, minlength = Np).astype(float)
      ps, pt = pair[s], pair[t]
      inner = (pid[s] == pid[t]) & (ps != pt)
      key, inv = np.unique(ps[inner] * Np + pt[inner], return_inverse = True)
      wi = np.bincount(inv, weights = w[inner])
      ki, kj = key // Np, key % Np
      side = ~alone[pid[s]]
      a = np.bincount(ps, weights = w * (side & (pid[t] == prev[pid[s]])), minlength = Np)
      c = np.bincount(ps, weights = w * (side & (pid[t] == nxt[pid[s]])), minlength = Np)
      start = np.searchsorted(pp, np.arange(P + 1))   # children of parent q: start[q] to start[q + 1]
      estart = np.searchsorted(pp[ki], np.arange(P + 1))
      rank = np.zeros(Np, dtype = np.int64)
      for q in np.flatnonzero(np.diff(start) > 1):
         b, e = start[q], start[q + 1]
         W = np.zeros((e - b, e - b))
         W[ki[estart[q]:estart[q + 1]] - b, kj[estart[q]:estart[q + 1]] - b] = wi[estart[q]:estart[q + 1]]
         order = _arrange(W, size[b:e], a[b:e], c[b:e], cyclic and l == 0)
         rank[b + order] = np.arange(e - b)
      G[:, l] = rank[pair]
   return G


def _neighbor_balance(net, fine, weight, cyclic = True):
   """
   Position of every node within its group, from its edges to the two adjacent groups around the
   circle: ``(c_next - c_prev) / (c_next + c_prev)``, where ``c_prev`` and ``c_next`` count (or
   sum the weights of) its edges to the previous and to the next group of its component (0 without
   such edges, or if the component has fewer than three groups).

   :param fine: Group of every node (network wide order), numbered consecutively around the
      circles, component after component.
   :param cyclic: Whether the groups go around a circle (else along a line: the first group has no
      previous one, the last no next one).
   :return: NumPy array.
   """
   n = len(fine)
   s, t, w = _all_edges(net, weight)
   comp = np.repeat(np.arange(len(net.components)), np.diff(net.node_offsets))
   g0 = np.full(len(net.components), n)
   g1 = np.full(len(net.components), -1)
   np.minimum.at(g0, comp, fine)   # first and last group of every component
   np.maximum.at(g1, comp, fine)
   g0, g1 = g0[comp], g1[comp]
   prev = np.where(fine > g0, fine - 1, g1 if cyclic else -1)   # around the circle (or along a line)
   nxt = np.where(fine < g1, fine + 1, g0 if cyclic else -1)
   ok = (g1 - g0 >= 2) if cyclic else (g1 > g0)
   cp = np.bincount(s, weights = w * (fine[t] == prev[s]), minlength = n)
   cn = np.bincount(s, weights = w * (fine[t] == nxt[s]), minlength = n)
   tot = cp + cn
   return np.where(ok & (tot > 0), (cn - cp) / np.where(tot > 0, tot, 1), 0.)


@_higher_order()
def circular_layout(net, groups = None, order = None, weight = None, spacing = 1., gap = 1., attribute = "pos", arrange = True):
   """
   Circular layout: the nodes of every component on a circle, in sectors by group (hierarchical
   groups nest, as sectors of sectors), and in each sector in the order given by a numeric
   attribute or by their edges.

   - ``groups``: one node attribute or a list of them, by decreasing importance: nodes with the
     same value of the first share a sector of the circle; inside it, nodes with the same value
     of the second share a smaller sector; and so on. Attributes can be qualitative or numeric:
     their distinct values are the groups (numeric codes of communities work as groups; nodes
     with missing values form one more group). Empty slots (``gap`` times the number of levels
     below the one that changes) separate the sectors: wider gaps between the sectors of the
     first attribute, narrower between finer ones.
   - Sectors are arranged by their edges (``arrange = True``), level by level from the first:
     inside the sector of their parent group (around the whole circle for the first attribute),
     groups with many edges between them are placed close, and groups with many edges to the
     previous (next) sector of the parent's level near its start (end). The arrangement minimizes
     the sum, over pairs of groups, of their edges (or weights) times the distance between the
     centers of their arcs, plus the edges to the adjacent sectors times the distance from
     them: exactly for up to 7 groups in a sector; above, from the order of the Fiedler vector
     (the angle in the spectral embedding, around the whole circle), improved by moving one group
     at a time where it lowers the sum most, up to 50 groups (by swapping neighboring groups up
     to 200; spectral order only above). Groups without edges keep the order of their values.
     With ``arrange = False``, sectors follow the sorted order of the values (numbers
     numerically, text alphabetically), clockwise from the top, missing values last; useful for
     ordered groups (age classes, years).
   - ``order``: a numeric node attribute sorting the nodes inside the finest sectors (or around
     the whole circle without ``groups``), clockwise by increasing value; missing values last.
   - Without ``order``, nodes inside a sector are sorted by their edges to the two adjacent sectors
     of the same level of detail: if sectors a, b, c follow each other, the nodes of b go from
     those with edges mostly to a (next to a) to those with edges mostly to c (next to c), by
     :math:`(e_c - e_a) / (e_c + e_a)`; nodes with neither stay in the middle, in the network
     order. This takes one pass over the edges (components with fewer than 3 sectors have no
     distinct neighbors on both sides, so their nodes keep the network order).
   - Without ``groups`` and ``order``, nodes are sorted by their angle in the Laplacian eigenmap of
     their component (:func:`~graphoskop.laplacian_eigenmaps`, 2 dimensions): neighbors end up
     close on the circle, which makes fewer and shorter chords than a random order.

   Each sector gets an arc proportional to its number of nodes. Every component has its own
   circle, with consecutive nodes ``spacing`` apart along it (its radius grows with the number of
   nodes); the circles are packed in rows, largest component first (as in
   :func:`force_directed_layout`). Directed networks are treated as undirected.

   :param net: A :class:`~graphoskop.Network`.
   :param groups: Name of a node attribute, or list of names in order of importance, or ``None``.
   :param order: Name of a numeric node attribute sorting the nodes in their sectors, or ``None``.
   :param weight: Name of a numeric edge attribute: without ``order``, nodes are sorted by the
      weights of their edges to the adjacent sectors instead of their number (and the Laplacian
      eigenmap, without ``groups``, is weighted); ``None`` counts edges.
   :param spacing: Distance between consecutive nodes along a circle.
   :param gap: Empty slots between sectors, per level of groups below the level that changes;
      0 for none.
   :param attribute: Prefix of the numeric node attributes ``<attribute>_x``, ``<attribute>_y``
      where to store the positions, or ``None`` to only return them.
   :param arrange: Whether to arrange the sectors by their edges (``True``) or in sorted order of
      the group values (``False``).
   :return: :class:`pandas.DataFrame` with one row per node, indexed by node id, in the network
      wide node order, and the columns ``<attribute>_x``, ``<attribute>_y`` (``x``, ``y`` with
      ``attribute = None``).
   :raises KeyError: If an attribute does not exist.
   :raises ValueError: If ``order`` is not a numeric node attribute.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      gsk.circular_layout(net)                                  # neighbors close on the circle
      gsk.circular_layout(net, order = "age")                   # clockwise by age
      gsk.circular_layout(net, ["country", "city"])             # sectors of sectors
      gsk.circular_layout(net, ["country", "city"], order = "age")
      gsk.circular_layout(net, "age_class", arrange = False)    # sectors in order of the values
   """
   sizes = np.diff(net.node_offsets)
   n = int(sizes.sum())
   comp = np.repeat(np.arange(len(sizes)), sizes)
   names = [groups] if isinstance(groups, str) else list(groups or [])
   G = np.stack([_codes(net, a) for a in names], 1) if names else np.zeros((n, 0), dtype = np.int64)
   if names and arrange:
      G = _arrange_groups(G, comp, *_all_edges(net, weight))
   fine = _consecutive(np.column_stack([comp, G]))   # finest group, numbered around the circles
   if order is not None:
      if order not in net.node_attributes:
         raise ValueError(f"order must be a numeric node attribute, got {order!r}")
      within = torch.cat(net.get_node_attribute(order, None)).double().cpu().numpy()
      within = np.where(np.isnan(within), np.inf, within)
   elif names:
      within = _neighbor_balance(net, fine, weight)
   else:
      from .gnn import _spectral
      E = _spectral(net, 2, weight, False)
      within = np.arctan2(E[:, 1], E[:, 0])
   perm = np.lexsort((np.arange(n), within, fine))   # around the circles
   L = G.shape[1]
   Gs, cs = G[perm], comp[perm]
   diff = Gs[1:] != Gs[:-1]
   level = np.where(diff.any(1), diff.argmax(1), L) if L else np.zeros(n - 1, dtype = np.int64)
   step = np.concatenate([[0.], np.where(cs[1:] == cs[:-1], 1 + gap * (L - level), 0.)])
   start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
   slot = np.cumsum(step)
   slot = slot - slot[start[cs]]   # position along the circle of its component
   firsts, lasts = G[perm][start], G[perm][start + sizes - 1]
   wrap = (firsts != lasts)
   wrap_level = np.where(wrap.any(1), wrap.argmax(1), L) if L else np.zeros(len(sizes), dtype = np.int64)
   total = slot[start + sizes - 1] + 1 + gap * (L - wrap_level)   # slots of every circle
   angle = np.pi / 2 - 2 * np.pi * slot / total[cs]   # clockwise from the top
   radius = np.where(sizes > 1, spacing * total / (2 * np.pi), 0.)[cs]
   X = np.empty((n, 2))
   X[perm] = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
   return _store(net, _pack(X, sizes, 2 * spacing), attribute)


def _space_levels(X, comp, branching = 8, rounds = 20):
   """
   Groups of a hierarchy of the space of a layout, from coarse to fine: level :math:`l` has
   ``branching ** l`` clusters of the positions (k-means: random nodes as first centers, fixed
   seed, then ``rounds`` rounds of Lloyd's algorithm), intersected with the groups of the level
   above, until groups have about 4 nodes; splits follow the empty space between clusters of
   nodes.

   :param X: Positions, one row per node.
   :param comp: Component of every node (the groups never span two components).
   :return: Integer array, one column per level.
   """
   from scipy.spatial import cKDTree
   n = len(X)
   rng = np.random.default_rng(0)
   cell, levels, K = comp.astype(np.int64), [], branching
   while K < n / 4:
      C = X[rng.choice(n, K, replace = False)]
      for _ in range(rounds):
         near = cKDTree(C).query(X)[1]
         count = np.bincount(near, minlength = K)
         C = np.where(count[:, None] > 0, np.column_stack([np.bincount(near, X[:, j], K) for j in range(X.shape[1])]) / np.maximum(count, 1)[:, None], C)
      cell = _consecutive(np.column_stack([cell, cKDTree(C).query(X)[1]]))
      levels.append(cell)
      K *= branching
   return np.column_stack(levels) if levels else np.zeros((n, 0), dtype = np.int64)


def _bezier_basis(k):
   """
   The clamped uniform B-spline of degree ``min(3, k - 1)`` with ``k`` control points as a chain
   of cubic Bézier curves, one per knot span: each piece goes from the curve at the start of its
   span to the curve at its end, with inner control points one third of the span along the
   derivatives (exact, since the pieces are polynomials of degree at most 3).

   :return: Array ``3 p + 1`` by ``k`` (``p`` pieces): the Bézier points are this matrix times the
      control points of the B-spline.
   """
   from scipy.interpolate import BSpline
   d = min(3, k - 1)
   spans = k - d
   u = np.linspace(0, 1, spans + 1)
   S = BSpline(np.concatenate([np.zeros(d), u, np.ones(d)]), np.eye(k), d)
   V, dV = S(u), S.derivative()(u) / (3 * spans)
   B = np.empty((3 * spans + 1, k))
   B[0::3] = V
   B[1::3] = V[:-1] + dV[:-1]
   B[2::3] = V[1:] - dV[1:]
   return B


@_higher_order()
def bundle_edges(net, groups = None, beta = 0.8, attribute = "pos"):
   """
   Hierarchical edge bundling (Holten, 2006), as in graph-tool: every edge is drawn as a curve
   that follows the hierarchy of groups of the nodes, so that edges between the same groups run
   together in bundles and the drawing shows the connections between groups rather than a
   tangle of lines.

   The hierarchy is a tree: its leaves are the nodes, above them the groups of the finest level,
   then the groups that contain them, up to one root per component. Every group sits at the
   center (mean position) of its nodes in the current layout. The curve of an edge between
   :math:`u` and :math:`v` is a cubic B-spline whose control points are the path in the tree from
   :math:`u` up to the smallest group containing both, and down to :math:`v`. As in graph-tool,
   the curve is returned as cubic Bézier curves, which drawing libraries draw exactly at any zoom. The strength
   ``beta`` straightens the path: every control point :math:`p_i` of a path of :math:`k` points
   moves to :math:`\\beta p_i + (1 - \\beta) (p_0 + \\frac{i}{k - 1} (p_{k - 1} - p_0))`, so that
   ``beta = 0`` gives straight edges and ``beta = 1`` curves that follow the hierarchy closely;
   the default 0.8 is graph-tool's. Edges within a finest group bend through its center only.

   - ``groups``: node attributes from the coarsest to the finest level, as the ``groups`` of
     :func:`circular_layout` (distinct values are groups, missing values one more group; a group
     of a level is split by the values of the next one). For the look of graph-tool's nested
     blockmodel drawings, fit the hierarchy with :func:`block_model_communities`, which writes the
     levels ``community`` (finest), ``community_1``, ..., and use them both for
     :func:`circular_layout` and here (see the example).
   - Without ``groups``, the hierarchy comes from the layout itself: clusters of the positions
     (k-means), 8 at the first level, 64 at the second, and so on, each split by the clusters of
     the next level, until groups have about 4 nodes. Edges then bundle between regions of the
     drawing, for any layout.

   The curve of every edge is a chain of cubic Bézier curves: points 0, 3, 6, ... are on the
   curve, the two points between each pair of them are the control points of that piece (piece
   :math:`i` is the Bézier curve of points :math:`3i` to :math:`3i + 3`). Edges have more pieces
   the further apart their nodes are in the hierarchy (one for edges within a finest group).
   Edges are taken as given in input, in the order of :meth:`~graphoskop.Network.edges` over all
   components (``component = None``); a self loop stays on its node. Computing the curves takes
   one pass over the edges per level of the hierarchy. The positions are not changed: run the
   function again after another layout.

   :param net: A :class:`~graphoskop.Network` with positions in the node attributes
      ``<attribute>_x``, ``<attribute>_y`` (and ``<attribute>_z``).
   :param groups: Name of a node attribute, or list of names from the coarsest level to the
      finest, or ``None`` for a hierarchy of the space of the layout.
   :param beta: Strength of the bundling, from 0 (straight edges) to 1.
   :param attribute: Prefix of the position attributes.
   :return: :class:`pandas.DataFrame` with one row per Bézier point, edge by edge: the position of
      the edge (``edge``), the position of the point along its curve (``point``, 0 at the source)
      and its coordinates ``x``, ``y`` (and ``z``).
   :raises KeyError: If the network has no positions with that prefix, or an attribute does not
      exist.
   :raises ValueError: If ``beta`` is not between 0 and 1.

   Example:

   .. code-block:: python

      import graphoskop as gsk
      import matplotlib.pyplot as plt
      from matplotlib.patches import PathPatch
      from matplotlib.path import Path

      net = gsk.load_network("edges.tsv")
      gsk.block_model_communities(net)                          # community, community_1, ...
      levels = ["community_2", "community_1", "community"]      # coarsest first
      gsk.circular_layout(net, groups = levels)
      bends = gsk.bundle_edges(net, groups = levels)
      ax = plt.gca()
      for e, curve in bends.groupby("edge"):
         xy = curve[["x", "y"]].to_numpy()
         path = Path(xy, [Path.MOVETO] + [Path.CURVE4] * (len(xy) - 1))
         ax.add_patch(PathPatch(path, fill = False, edgecolor = "gray", alpha = 0.3, lw = 0.5))
   """
   if not 0 <= beta <= 1:
      raise ValueError(f"beta must be between 0 and 1, got {beta}")
   X, _ = _positions(net, attribute)
   sizes = np.diff(net.node_offsets)
   n, D = X.shape
   comp = np.repeat(np.arange(len(sizes)), sizes)
   names = [groups] if isinstance(groups, str) else list(groups or [])
   if names:
      G = np.column_stack([comp] + [_consecutive(np.column_stack([comp] + [_codes(net, a) for a in names[:l + 1]])) for l in range(len(names))])
   else:
      G = np.column_stack([comp, _space_levels(X, comp)])
   L = G.shape[1]   # tree levels: the root of the component, then the groups
   T, P = [], []
   for l in range(L):   # tree nodes of every level, at the mean position of their nodes
      ids = G[:, l]
      count = np.bincount(ids)
      C = np.column_stack([np.bincount(ids, X[:, j]) for j in range(D)]) / np.maximum(count, 1)[:, None]
      T.append(ids + sum(len(p) for p in P))
      P.append(C)
   T, P = np.column_stack(T), np.concatenate(P)
   E = [net._input_edges(c)[0].cpu().numpy() + o for c, o in zip(range(len(net.components)), net.node_offsets)]
   s, t = (np.concatenate([e[i] for e in E]) for i in (0, 1)) if E else (np.zeros(0, dtype = np.int64),) * 2
   m = len(s)
   same = T[s] == T[t]
   c = same.sum(1) - 1   # level of the smallest common group (the root of the component is level 0)
   k = 3 + 2 * (L - 1 - c)                   # control points of every edge (u, groups, v)
   count = 3 * (k - np.minimum(3, k - 1)) + 1   # its Bezier points: 3 per piece, plus 1
   start = np.concatenate([[0], np.cumsum(count)])
   xyz = np.empty((start[-1], D))
   for lca in np.unique(c):
      e = np.flatnonzero(c == lca)
      up = T[s[e]][:, L - 1:lca:-1]          # groups of the source below the common one, finest first
      down = T[t[e]][:, lca + 1:]            # groups of the target below the common one, coarsest first
      path = np.concatenate([X[s[e]][:, None], P[up], P[T[s[e], lca]][:, None], P[down], X[t[e]][:, None]], 1)
      line = path[:, :1] + np.linspace(0, 1, path.shape[1])[None, :, None] * (path[:, -1:] - path[:, :1])
      curve = _bezier_basis(path.shape[1]) @ (beta * path + (1 - beta) * line)
      loops = s[e] == t[e]
      curve[loops] = X[s[e][loops]][:, None]
      xyz[(start[e][:, None] + np.arange(curve.shape[1])).ravel()] = curve.reshape(-1, D)
   edge = np.repeat(np.arange(m), count)
   table = pd.DataFrame({"edge": edge, "point": np.arange(start[-1]) - start[edge]})
   for j, a in enumerate("xyz"[:D]):
      table[a] = xyz[:, j]
   return table


@njit(cache = True)
def _spread(t, starts, ends, gap):
   """
   Positions closest to the targets ``t`` (least squares) that keep the order of every segment
   ``[starts[k], ends[k])`` with at least ``gap`` between consecutive nodes: isotonic regression
   (pool adjacent violators) of ``t[i] - i * gap`` in each segment.
   """
   x = np.empty_like(t)
   val = np.empty(len(t))
   wt = np.empty(len(t))
   cnt = np.empty(len(t), np.int64)
   for k in range(len(starts)):
      a, b, top = starts[k], ends[k], 0
      for i in range(b - a):
         v, w, c = t[a + i] - i * gap, 1., 1
         while top > 0 and val[top - 1] > v:   # merge with the previous block while out of order
            v = (val[top - 1] * wt[top - 1] + v * w) / (wt[top - 1] + w)
            w += wt[top - 1]
            c += cnt[top - 1]
            top -= 1
         val[top], wt[top], cnt[top] = v, w, c
         top += 1
      i = 0
      for j in range(top):
         for _ in range(cnt[j]):
            x[a + i] = val[j] + i * gap
            i += 1
   return x


@njit(cache = True)
def _finish_order(indptr, nbr, roots):
   """
   Finishing times of an iterative depth-first search from the given roots, in order (every node
   is reached: the roots should list all nodes).
   """
   n = len(indptr) - 1
   state = np.zeros(n, np.int8)
   fin = np.full(n, -1)
   stack = np.empty(n, np.int64)
   ptr = np.empty(n, np.int64)
   clock = 0
   for r in roots:
      if state[r]:
         continue
      top = 0
      stack[0], ptr[r], state[r] = r, indptr[r], 1
      while top >= 0:
         u = stack[top]
         if ptr[u] < indptr[u + 1]:
            v = nbr[ptr[u]]
            ptr[u] += 1
            if state[v] == 0:
               top += 1
               stack[top], ptr[v], state[v] = v, indptr[v], 1
         else:
            state[u], fin[u] = 2, clock
            clock += 1
            top -= 1
   return fin


def _expand(indptr, nodes):
   """
   Positions in CSR order of the edges leaving ``nodes``, and the source of each.
   """
   cnt = indptr[nodes + 1] - indptr[nodes]
   src = np.repeat(nodes, cnt)
   idx = np.repeat(indptr[nodes] - np.cumsum(cnt) + cnt, cnt) + np.arange(cnt.sum())
   return idx, src


def _layers(net, levels, root, s, t, comp):
   """
   Layer of every node (network wide order): the dense rank of a numeric attribute (missing values
   last); in directed networks, the longest path from a source, after reversing the edges that
   close a cycle in a depth-first search from the sources; in undirected networks, the distance
   from the root of the component (the given one, or the node with most neighbors).

   :param s: Sources of the edges (as stored, network wide); ``t``: their targets.
   """
   n = len(comp)
   if levels is not None:
      v = torch.cat(net.get_node_attribute(levels, None)).double().cpu().numpy()
      return np.unique(np.where(np.isnan(v), np.inf, v), return_inverse = True)[1].reshape(-1)
   if net.directed:
      keep = s != t
      s, t = s[keep], t[keep]
      o = np.argsort(s, kind = "stable")
      indeg = np.bincount(t, minlength = n)
      roots = np.lexsort((np.arange(n), indeg > 0))   # sources first
      fin = _finish_order(np.searchsorted(s[o], np.arange(n + 1)), t[o], roots)
      back = fin[s] < fin[t]   # edges closing a cycle in the search, reversed
      s, t = np.where(back, t, s), np.where(back, s, t)
      key = np.unique(s * n + t)
      cs, ct = key // n, key % n   # acyclic, sorted by source
      indptr = np.searchsorted(cs, np.arange(n + 1))
      indeg = np.bincount(ct, minlength = n)
      layer = np.zeros(n, dtype = np.int64)
      front = np.flatnonzero(indeg == 0)
      while len(front):   # longest path, sources first (Kahn's algorithm, a frontier at a time)
         idx, src = _expand(indptr, front)
         nxt = ct[idx]
         np.maximum.at(layer, nxt, layer[src] + 1)
         np.subtract.at(indeg, nxt, 1)
         front = np.unique(nxt[indeg[nxt] == 0])
      return layer
   u, v = np.concatenate([s, t]), np.concatenate([t, s])
   o = np.argsort(u, kind = "stable")
   u, v = u[o], v[o]
   indptr = np.searchsorted(u, np.arange(n + 1))
   deg = np.diff(indptr)
   roots = np.full(comp.max() + 1, -1)
   for r in ([] if root is None else [root] if not isinstance(root, (list, set, np.ndarray, pd.Index)) else root):
      c, i = net.node_map[r]
      roots[c] = net.node_offsets[c] + i
   best = np.lexsort((-deg, comp))   # node with most neighbors of every component
   first = np.searchsorted(comp[best], np.arange(len(roots)))
   roots = np.where(roots < 0, best[first], roots)
   dist = np.full(n, -1)
   dist[roots] = 0
   front, d = roots, 0
   while len(front):   # breadth-first search from all roots at once
      idx, _ = _expand(indptr, front)
      nb = np.unique(v[idx])
      front = nb[dist[nb] < 0]
      d += 1
      dist[front] = d
   return dist


def _segments(comp, layer, key, tie):
   """
   Order of the nodes by component, layer and key, the start and end of every (component, layer)
   segment in that order, and the centered rank of every node in its segment.
   """
   perm = np.lexsort((tie, key, layer, comp))
   cl = np.column_stack([comp[perm], layer[perm]])
   change = np.concatenate([[True], (cl[1:] != cl[:-1]).any(1), [True]])
   bounds = np.flatnonzero(change)
   starts, ends = bounds[:-1], bounds[1:]
   seg = np.repeat(np.arange(len(starts)), ends - starts)
   rank = np.empty(len(perm))
   rank[perm] = np.arange(len(perm)) - starts[seg] - (ends - starts - 1)[seg] / 2
   return perm, starts, ends, rank


@_higher_order()
def hierarchical_layout(net, levels = None, root = None, direction = "down", spacing = 1., layer_spacing = 2., attribute = "pos"):
   """
   Hierarchical (layered) layout, in the style of Sugiyama, Tagawa and Toda (1981): nodes in
   horizontal layers, edges going from one layer to the next, few edge crossings, and every node
   near the middle of its neighbors.

   1. Layers. With ``levels``, the distinct values of a numeric node attribute, in increasing
      order (for instance a year, or a rank from :func:`~graphoskop.agony_hierarchy`); missing
      values make the last layer. Otherwise, in directed networks, every node goes one layer
      below its lowest predecessor (longest path from the sources), so that all edges point
      down; in networks with cycles, the edges that close a cycle in a depth-first search from
      the sources point up instead (as in Graphviz's ``dot``). In undirected
      networks, the layer is the distance from a root: the node given in ``root`` for its
      component, or the node with most neighbors (trees drawn from their hub).
   2. Order in the layers, to reduce crossings: layer after layer, alternately downwards and
      upwards, the nodes of a layer are sorted by the mean position of their neighbors in the
      layers above (below) (barycenter heuristic, 25 passes, the last downwards). Edges spanning several layers count
      with the position of their other end, edges inside a layer are not used.
   3. Positions in the layers: every node is pulled to the mean position of its neighbors in the
      layers below (nodes without any, such as leaves, to those above), keeping the order and at
      least ``spacing`` between consecutive nodes (an isotonic regression per layer, 30 rounds),
      so that parents sit above the middle of their children, as in tidy tree drawings.
   4. With ``direction = "radial"``, layers become concentric circles, from the inside out, as the
      radial tree drawings of Eades (1992): in a spanning tree (the parent of a node is its
      neighbor in the closest layer above, the nearest in position), every node gets a wedge of
      the circle proportional to the number of leaves below it, its children splitting the wedge
      in their order in the layers (clockwise from the top), and sits in the middle of its wedge;
      so every leaf has its own angle, whatever its depth, and parents sit at the middle angle of
      their children. Children fan out at most 45 degrees from their parent (their wedges
      shrinking around it if needed), since edges spanning wider angles would bend back inside
      the circle of the parent and cross other edges. A first layer of one node (a root) is at
      the center; every circle is
      ``layer_spacing`` beyond the previous one, or farther where consecutive nodes on it would
      otherwise be closer than ``spacing`` (inner circles then grow in proportion, so that radii
      rise evenly). Radial layouts suit trees and hierarchies; edges outside the spanning tree,
      as in networks with many paths between layers, become chords.

   Each component is laid out on its own and the components are packed in rows, largest first
   (as in :func:`force_directed_layout`).

   :param net: A :class:`~graphoskop.Network`.
   :param levels: Name of a numeric node attribute giving the layers, or ``None``.
   :param root: Without ``levels``, in undirected networks: the node id (or a list of ids, one
      per component) at the top; ``None`` for the node with most neighbors of each component.
   :param direction: Where the layers go: ``"down"`` (first layer at the top), ``"up"``,
      ``"right"``, ``"left"``, or ``"radial"`` (first layer at the center, the others on
      concentric circles around it).
   :param spacing: Smallest distance between nodes of a layer.
   :param layer_spacing: Distance between consecutive layers.
   :param attribute: Prefix of the numeric node attributes ``<attribute>_x``, ``<attribute>_y``
      where to store the positions, or ``None`` to only return them.
   :return: :class:`pandas.DataFrame` with one row per node, indexed by node id, in the network
      wide node order, and the columns ``<attribute>_x``, ``<attribute>_y`` (``x``, ``y`` with
      ``attribute = None``).
   :raises KeyError: If ``levels`` or a root does not exist.
   :raises ValueError: If ``direction`` is unknown.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("citations.tsv", directed = True)
      gsk.hierarchical_layout(net)                        # edges point down
      gsk.hierarchical_layout(net, levels = "year", direction = "right")
      tree = gsk.load_network("tree.tsv")
      gsk.hierarchical_layout(tree, root = "ceo")
      gsk.hierarchical_layout(tree, root = "ceo", direction = "radial")   # root at the center
   """
   if direction not in ("down", "up", "right", "left", "radial"):
      raise ValueError(f"direction must be 'down', 'up', 'right', 'left' or 'radial', got {direction!r}")
   sizes = np.diff(net.node_offsets)
   n = int(sizes.sum())
   comp = np.repeat(np.arange(len(sizes)), sizes)
   off = net.node_offsets
   s = np.concatenate([d.edge_index[0].cpu().numpy() + o for d, o in zip(net.components, off)])
   t = np.concatenate([d.edge_index[1].cpu().numpy() + o for d, o in zip(net.components, off)])
   layer = _layers(net, levels, root, s, t, comp)
   u, v = np.concatenate([s, t]), np.concatenate([t, s])
   other = layer[u] != layer[v]
   u, v = u[other], v[other]
   tie = np.arange(n)
   x = _segments(comp, layer, tie, tie)[3]
   nodes = np.argsort(layer, kind = "stable")
   nb = np.searchsorted(layer[nodes], np.arange(layer.max() + 2))   # nodes of layer l: nodes[nb[l]:nb[l + 1]]
   down, up = layer[v] < layer[u], layer[v] > layer[u]
   ed, eu = np.flatnonzero(down), np.flatnonzero(up)
   ed, eu = ed[np.argsort(layer[u[ed]], kind = "stable")], eu[np.argsort(layer[u[eu]], kind = "stable")]
   bd = np.searchsorted(layer[u[ed]], np.arange(layer.max() + 2))
   bu = np.searchsorted(layer[u[eu]], np.arange(layer.max() + 2))
   L = layer.max() + 1
   for k in range(25):   # barycenters, a layer at a time, downwards then upwards (last down)
      E, b, seq = (ed, bd, range(1, L)) if k % 2 == 0 else (eu, bu, range(L - 2, -1, -1))
      for l in seq:
         e = E[b[l]:b[l + 1]]
         if not len(e):
            continue
         nl = nodes[nb[l]:nb[l + 1]]
         cnt = np.bincount(u[e], minlength = n)[nl]
         key = np.where(cnt > 0, np.bincount(u[e], weights = x[v[e]], minlength = n)[nl] / np.maximum(cnt, 1), x[nl])
         o = np.lexsort((x[nl], key, comp[nl]))
         first = np.searchsorted(comp[nl][o], comp[nl][o], side = "left")
         last = np.searchsorted(comp[nl][o], comp[nl][o], side = "right")
         x[nl[o]] = np.arange(len(nl)) - first - (last - first - 1) / 2
   perm, starts, ends, x = _segments(comp, layer, x, tie)
   x = x * spacing
   below = layer[v] > layer[u]
   cb, ca = np.bincount(u[below], minlength = n), np.bincount(u[~below], minlength = n)
   for _ in range(30):   # every node above the middle of its neighbors below (else of those above)
      mb = np.bincount(u[below], weights = x[v[below]], minlength = n) / np.maximum(cb, 1)
      ma = np.bincount(u[~below], weights = x[v[~below]], minlength = n) / np.maximum(ca, 1)
      target = np.where(cb > 0, mb, np.where(ca > 0, ma, x))
      x[perm] = _spread(target[perm], starts, ends, float(spacing))
   if direction == "radial":
      X = _radial(x, layer, comp, perm, starts, ends, u, v, spacing, layer_spacing)
   else:
      y = -layer_spacing * (layer - layer.min())
      X = np.column_stack({"down": (x, y), "up": (x, -y), "right": (-y, x), "left": (y, x)}[direction])
   return _store(net, _pack(X, sizes, 2 * spacing), attribute)


def _radial(x, layer, comp, perm, starts, ends, u, v, spacing, layer_spacing):
   """
   Layers as concentric circles, from the inside out, as the radial trees of Eades (1992): every
   node gets a wedge of the circle proportional to the number of leaves below it in a spanning
   tree (its parent: its neighbor in the closest layer above, the nearest in position), the
   children of a node splitting its wedge in their order in the layers, and sits in the middle of
   its wedge; nodes of the first layer of a component share the whole circle. The children of a
   node off the center fan out at most ``_TURN`` (45 degrees) from it, their wedges shrinking
   around it if needed: edges spanning wide angles would bend back inside the circle of their
   parent and cross the edges of neighboring wedges. The k-th layer of a
   component is its k-th circle (the first at the center if it has a single node): every circle
   is ``layer_spacing`` beyond the previous one, or farther if needed so that consecutive nodes
   on it are at least ``spacing`` apart, inner circles growing in proportion to such outer needs
   so that radii rise evenly.

   :param perm: Nodes sorted by component, layer and position; ``starts``, ``ends``: the
      (component, layer) segments of ``perm``.
   :param u: Sources of the edges between layers (both directions); ``v``: their targets.
   :return: Positions, NumPy array ``(n, 2)``.
   """
   n = len(x)
   parent = np.full(n, -1)
   m = layer[v] < layer[u]
   o = np.lexsort((np.abs(x[u[m]] - x[v[m]]), -layer[v[m]], u[m]))
   pu, pv = u[m][o], v[m][o]
   head = np.r_[True, pu[1:] != pu[:-1]]
   parent[pu[head]] = pv[head]
   leaves = np.zeros(n)
   for a, b in zip(starts[::-1].tolist(), ends[::-1].tolist()):   # deepest layers first: leaves below every node
      seg = perm[a:b]
      leaves[seg] = np.where(leaves[seg] == 0, 1, leaves[seg])
      kid = seg[parent[seg] >= 0]
      np.add.at(leaves, parent[kid], leaves[kid])
   roots = perm[parent[perm] < 0]   # sorted by component and position
   rc = comp[roots]
   total = np.bincount(rc, weights = leaves[roots])
   before = np.cumsum(leaves[roots]) - leaves[roots]
   first_root = np.searchsorted(rc, rc)
   seg_comp = comp[perm[starts]]
   first = np.r_[True, seg_comp[1:] != seg_comp[:-1]]
   k = np.arange(len(starts)) - np.maximum.accumulate(np.where(first, np.arange(len(starts)), 0))   # circle of every segment
   k = k + np.repeat(np.where(first & (ends - starts > 1), 1, 0)[first], np.diff(np.r_[np.flatnonzero(first), len(starts)]))   # a first layer of several nodes on circle 1
   ring = np.empty(n, dtype = int)
   ring[perm] = np.repeat(k, ends - starts)
   start, width = np.zeros(n), np.zeros(n)
   start[roots] = 2 * np.pi * (before - before[first_root]) / total[rc]
   width[roots] = 2 * np.pi * leaves[roots] / total[rc]
   for a, b in zip(starts.tolist(), ends.tolist()):   # wedges of the children, layer after layer
      seg = perm[a:b]
      kid = seg[parent[seg] >= 0]
      if not len(kid):
         continue
      kid = kid[np.lexsort((x[kid], parent[kid]))]
      p = parent[kid]
      cum = np.cumsum(leaves[kid])
      grp = np.searchsorted(p, p)
      done = cum - leaves[kid] - (cum[grp] - leaves[kid[grp]])
      side = 1 - np.minimum(leaves[kid[grp]], leaves[kid[np.searchsorted(p, p, side = "right") - 1]]) / leaves[p]
      span = np.where(ring[p] > 0, np.minimum(width[p], 2 * _TURN / np.maximum(side, 1e-12)), width[p])   # children at most _TURN from their parent
      start[kid] = start[p] + (width[p] - span) / 2 + span * done / leaves[p]
      width[kid] = span * leaves[kid] / leaves[p]
   theta = start + width / 2
   need = np.zeros(len(starts))   # smallest radius keeping consecutive nodes spacing apart
   for j, (a, b) in enumerate(zip(starts.tolist(), ends.tolist())):
      if b - a > 1:
         th = np.sort(theta[perm[a:b]])
         need[j] = spacing / (2 * np.sin(min(np.diff(np.r_[th, th[0] + 2 * np.pi]).min(), np.pi) / 2))
   ahead = np.zeros(len(starts))   # largest need per circle index of this or later circles: inner circles grow in proportion
   for j in range(len(starts) - 1, -1, -1):
      own = need[j] / k[j] if k[j] else 0.
      ahead[j] = own if j == len(starts) - 1 or first[j + 1] else max(own, ahead[j + 1])
   radius, r = np.zeros(n), 0.
   for j, (a, b) in enumerate(zip(starts.tolist(), ends.tolist())):   # circle by circle, from the inside
      r = max(need[j], k[j] * ahead[j], (0. if k[j] == 0 else layer_spacing) if first[j] else r + layer_spacing)
      radius[perm[a:b]] = r
   return np.column_stack([radius * np.sin(theta), radius * np.cos(theta)])


def _shelves(w, h, gap):
   """
   Blocks in rows (shelves) of about the width of a square holding them, in order, the rows
   snaking (left to right, then right to left) so that consecutive blocks stay side by side,
   ``gap`` cells apart.

   :param w: Widths of the blocks (cells); ``h``: heights.
   :return: Column and row offsets of the blocks, and the width and height of the whole.
   """
   width = max(w.max(), int(np.ceil(np.sqrt(((w + gap) * (h + gap)).sum()))))
   x, y = np.zeros(len(w), dtype = np.int64), np.zeros(len(w), dtype = np.int64)
   row, cx, cy, high = np.zeros(len(w), dtype = np.int64), 0, 0, 0
   r = 0
   for i, (bw, bh) in enumerate(zip(w.tolist(), h.tolist())):
      if cx > 0 and cx + bw > width:
         cx, cy, high, r = 0, cy + high + gap, 0, r + 1
      x[i], y[i], row[i] = cx, cy, r
      cx, high = cx + bw + gap, max(high, bh)
   W = int((x + w).max())
   odd = row % 2 == 1
   x[odd] = W - x[odd] - w[odd]   # right to left
   return x, y, W, int((y + h).max())


def _snap(block, tx, ty, height, tie):
   """
   Cells of the nodes of every block, near their targets: nodes sorted by target column fill the
   columns of their block (``height`` nodes each, left to right), then each column is sorted by
   target row.

   :return: Column and row of every node in its block.
   """
   o = np.lexsort((tie, tx, block))
   start = np.searchsorted(block[o], block[o])
   col = np.empty(len(o), dtype = np.int64)
   col[o] = (np.arange(len(o)) - start) // height[block[o]]
   o = np.lexsort((tie, ty, col, block))
   key = block[o] * (col.max() + 1) + col[o]
   row = np.empty(len(o), dtype = np.int64)
   row[o] = np.arange(len(o)) - np.searchsorted(key, key)
   return col, row


@_higher_order()
def grid_layout(net, groups = None, order = None, weight = None, spacing = 1., gap = 1., attribute = "pos", arrange = True):
   """
   Grid layout: the nodes of every component on the cells of a regular grid, in rectangular
   blocks by group (hierarchical groups nest, as blocks of blocks), the counterpart of
   :func:`circular_layout` on a grid.

   - ``groups``: one node attribute or a list of them, by decreasing importance; their distinct
     values are the groups (qualitative or numeric). Every group of the last attribute is a block
     of about square shape (``ceil(sqrt(n))`` columns); the blocks of a group of the attribute
     above are packed in rows of about the width of a square holding them, the rows snaking left
     to right and right to left so that consecutive blocks stay side by side; and so on up to the
     whole component. Empty cells (``gap`` times the number of levels below the one that
     changes) separate the blocks: wider gaps between the blocks of the first attribute.
   - The order of the blocks follows the arrangement of :func:`circular_layout` (``arrange =
     True``): groups with many edges between them are consecutive, and so side by side, and groups
     with many edges to the previous (next) block of their parent's level come first (last), along
     the line of the snaking rows instead of around a circle. With ``arrange = False``, blocks
     follow the sorted order of the values.
   - ``order``: a numeric node attribute filling every block row by row (reading order) by
     increasing value, missing values last.
   - Without ``order``, nodes are placed in their block by their edges to other blocks: each node
     is pulled from the center of its block towards the centers of the blocks of its neighbors in
     other groups (weighted by the edges), as far as the rank of its number of such edges in its
     block, then nodes are snapped to the cells (sorted into columns by their target column, then
     within columns by their target row). Nodes without edges to other groups fill the middle of
     the block, those with many fill its borders, on the side of the blocks they are linked to.
   - Without ``groups`` and ``order``, every component is one block and nodes are snapped from
     their position in the Laplacian eigenmap of the component
     (:func:`~graphoskop.laplacian_eigenmaps`, 2 dimensions), so that neighbors are close; the
     embedding is turned (in steps of 5 degrees) to the angle giving the shortest edges on the
     grid (a lattice network comes back as a lattice).

   Columns and rows are ``spacing`` apart; rows go downwards. Components get their own grids,
   packed in rows, largest first (as in :func:`force_directed_layout`). Directed networks are
   treated as undirected.

   :param net: A :class:`~graphoskop.Network`.
   :param groups: Name of a node attribute, or list of names in order of importance, or ``None``.
   :param order: Name of a numeric node attribute sorting the nodes in their blocks, or ``None``.
   :param weight: Name of a numeric edge attribute weighting the edges in the arrangement of the
      blocks and the placement of the nodes, or ``None``.
   :param spacing: Distance between consecutive columns and rows.
   :param gap: Empty cells between blocks, per level of groups below the level that changes; 0
      for none.
   :param attribute: Prefix of the numeric node attributes ``<attribute>_x``, ``<attribute>_y``
      where to store the positions, or ``None`` to only return them.
   :param arrange: Whether to arrange the blocks by their edges (``True``) or in sorted order of
      the group values (``False``).
   :return: :class:`pandas.DataFrame` with one row per node, indexed by node id, in the network
      wide node order, and the columns ``<attribute>_x``, ``<attribute>_y`` (``x``, ``y`` with
      ``attribute = None``).
   :raises KeyError: If an attribute does not exist.
   :raises ValueError: If ``order`` is not a numeric node attribute.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      gsk.grid_layout(net)                                  # neighbors close on the grid
      gsk.grid_layout(net, ["country", "city"])             # blocks of blocks
      gsk.grid_layout(net, "community", order = "age")      # blocks filled by age
   """
   sizes = np.diff(net.node_offsets)
   n = int(sizes.sum())
   comp = np.repeat(np.arange(len(sizes)), sizes)
   names = [groups] if isinstance(groups, str) else list(groups or [])
   G = np.stack([_codes(net, a) for a in names], 1) if names else np.zeros((n, 0), dtype = np.int64)
   s, t, w = _all_edges(net, weight)
   if names and arrange:
      G = _arrange_groups(G, comp, s, t, w, cyclic = False)
   L = G.shape[1]
   gid = [_consecutive(np.column_stack([comp, G[:, :l]])) for l in range(L + 1)]   # groups of every level, in order
   fine = gid[L]
   size = np.bincount(fine)
   bw = np.ceil(np.sqrt(size)).astype(np.int64)   # blocks of the finest groups
   bh = -(-size // bw)
   ox, oy = [None] * L + [np.zeros(len(size), dtype = np.int64)], [None] * L + [np.zeros(len(size), dtype = np.int64)]
   cw, ch = bw, bh
   for l in range(L - 1, -1, -1):   # pack the blocks of every group of level l + 1 into their parent
      parent = np.zeros(gid[l + 1].max() + 1, dtype = np.int64)
      parent[gid[l + 1]] = gid[l]
      P = gid[l].max() + 1
      pw, ph = np.zeros(P, dtype = np.int64), np.zeros(P, dtype = np.int64)
      x, y = np.zeros(len(parent), dtype = np.int64), np.zeros(len(parent), dtype = np.int64)
      start = np.searchsorted(parent, np.arange(P + 1))
      for q in range(P):
         b, e = start[q], start[q + 1]
         x[b:e], y[b:e], pw[q], ph[q] = _shelves(cw[b:e], ch[b:e], int(round(gap * (L - l))))
      ox[l + 1], oy[l + 1], cw, ch = x, y, pw, ph
   bx, by = np.zeros(len(size), dtype = np.int64), np.zeros(len(size), dtype = np.int64)   # block corners in their component
   first = np.zeros(len(size), dtype = np.int64)
   first[fine[::-1]] = np.arange(n)[::-1]   # a node of every finest group
   for l in range(1, L + 1):
      bx += ox[l][gid[l][first]]
      by += oy[l][gid[l][first]]
   tie = np.arange(n)
   if order is not None:
      if order not in net.node_attributes:
         raise ValueError(f"order must be a numeric node attribute, got {order!r}")
      v = torch.cat(net.get_node_attribute(order, None)).double().cpu().numpy()
      o = np.lexsort((tie, np.where(np.isnan(v), np.inf, v), fine))
      r = np.empty(n, dtype = np.int64)
      r[o] = np.arange(n) - np.searchsorted(fine[o], fine[o])
      col, row = r % bw[fine], r // bw[fine]
   else:
      if names:   # pulled towards the blocks of the neighbors in other groups
         cx, cy = bx + bw / 2, by + bh / 2
         out = fine[s] != fine[t]
         s2, t2, w2 = s[out], t[out], w[out]
         dx = np.bincount(s2, weights = w2 * (cx[fine[t2]] - cx[fine[s2]]), minlength = n)
         dy = np.bincount(s2, weights = w2 * (cy[fine[t2]] - cy[fine[s2]]), minlength = n)
         k = np.bincount(s2, weights = w2, minlength = n)
         o = np.lexsort((k, fine))
         first_pos = np.searchsorted(fine[o], fine[o])
         rho = np.empty(n)
         rho[o] = (np.arange(n) - first_pos) / np.maximum(size[fine[o]] - 1, 1)   # rank of the edges out of the group
         rho = np.where(k > 0, rho, 0.)
         norm = np.hypot(dx, dy)
         tx = rho * np.divide(dx, norm, out = np.zeros(n), where = norm > 0) * bw[fine]
         ty = rho * np.divide(dy, norm, out = np.zeros(n), where = norm > 0) * bh[fine]
      else:
         from .gnn import _spectral
         E = _spectral(net, 2, weight, False)
         best = None
         for a in np.radians(np.arange(0, 90, 5)):   # the turn of the embedding with the shortest edges on the grid
            tx, ty = np.cos(a) * E[:, 0] - np.sin(a) * E[:, 1], np.sin(a) * E[:, 0] + np.cos(a) * E[:, 1]
            col, row = _snap(fine, tx, ty, bh, tie)
            cost = (w * (np.abs(col[s] - col[t]) + np.abs(row[s] - row[t]))).sum()
            if best is None or cost < best[0]:
               best = (cost, col, row)
         col, row = best[1:]
      if names:
         col, row = _snap(fine, tx, ty, bh, tie)
   X = np.column_stack([(bx[fine] + col) * spacing, -(by[fine] + row) * spacing]).astype(float)
   return _store(net, _pack(X, sizes, 2 * spacing), attribute)


def _positions(layout, attribute):
   """
   Positions of a layout: a table (one column per axis, x first), or the numeric node attributes
   ``<attribute>_x``, ``<attribute>_y`` (and ``<attribute>_z``) of a network.

   :return: NumPy array, one row per node, and the table (``None`` for a network).
   """
   if isinstance(layout, pd.DataFrame):
      return np.array(layout, dtype = float), layout
   names = [f"{attribute}_{a}" for a in "xyz" if f"{attribute}_{a}" in layout.node_attributes]
   if len(names) < 2:
      raise KeyError(f"no node attributes {attribute}_x and {attribute}_y: compute a layout first, or give its table")
   return np.column_stack([torch.cat(layout.get_node_attribute(a, None)).double().cpu().numpy() for a in names]), None


def _transformed(layout, X, table, attribute):
   """
   The transformed positions: a new table like ``table``, or stored back in the network.
   """
   if table is not None:
      return pd.DataFrame(X, index = table.index, columns = table.columns)
   return _store(layout, X, attribute)


def _axes(axis, dimensions):
   """
   Columns of the axes ``"x"``, ``"y"``, ``"z"`` or ``"both"`` (all of them).
   """
   if axis == "both":
      return list(range(dimensions))
   if axis not in ("x", "y", "z")[:dimensions]:
      raise ValueError(f"axis must be one of {', '.join(map(repr, ('x', 'y', 'z')[:dimensions]))} or 'both', got {axis!r}")
   return ["xyz".index(axis)]


@_higher_order()
def mirror_layout(layout, axis = "x", attribute = "pos"):
   """
   Mirror a layout: flip its ``axis`` coordinates around the center of the layout (the middle of
   its bounding box), so that the drawing stays in place; ``"x"`` swaps left and right, ``"y"`` top
   and bottom.

   :param layout: A :class:`~graphoskop.Network` with positions in node attributes (changed in
      place), or a table of positions as returned by the layout functions (a new table is
      returned).
   :param axis: ``"x"``, ``"y"``, ``"z"`` (3-d layouts) or ``"both"`` (all axes: a rotation by 180
      degrees in two dimensions).
   :param attribute: With a network, the prefix of the position attributes.
   :return: :class:`pandas.DataFrame` of the new positions, as the layout functions return.
   :raises KeyError: If the network has no positions with that prefix.
   :raises ValueError: If the axis is unknown.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.hierarchical_layout(net)
      gsk.mirror_layout(net, "y")             # root at the bottom
      gsk.rotate_layout(net, 90)              # root on the left
      gsk.scale_layout(net, 2, axis = "x")    # twice as wide
   """
   X, table = _positions(layout, attribute)
   for j in _axes(axis, X.shape[1]):
      X[:, j] = np.nanmin(X[:, j]) + np.nanmax(X[:, j]) - X[:, j]
   return _transformed(layout, X, table, attribute)


@_higher_order()
def rotate_layout(layout, degrees, axis = "z", attribute = "pos"):
   """
   Rotate a layout counterclockwise by ``degrees`` around the center of the layout (the middle of
   its bounding box). In two dimensions the rotation is in the plane; in three, around the
   ``axis`` (``"z"``: in the x-y plane, as seen from above).

   :param layout: A :class:`~graphoskop.Network` with positions in node attributes (changed in
      place), or a table of positions as returned by the layout functions (a new table is
      returned).
   :param degrees: Angle of the rotation, counterclockwise; negative for clockwise.
   :param axis: Axis of the rotation in three dimensions: ``"x"``, ``"y"`` or ``"z"``.
   :param attribute: With a network, the prefix of the position attributes.
   :return: :class:`pandas.DataFrame` of the new positions, as the layout functions return.
   :raises KeyError: If the network has no positions with that prefix.
   :raises ValueError: If the axis is unknown.
   """
   X, table = _positions(layout, attribute)
   if X.shape[1] == 3 and axis not in ("x", "y", "z"):
      raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
   i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis] if X.shape[1] == 3 else (0, 1)
   c = (np.nanmin(X, 0) + np.nanmax(X, 0)) / 2
   a = np.radians(degrees)
   xi, xj = X[:, i] - c[i], X[:, j] - c[j]
   X[:, i], X[:, j] = c[i] + np.cos(a) * xi - np.sin(a) * xj, c[j] + np.sin(a) * xi + np.cos(a) * xj
   return _transformed(layout, X, table, attribute)


@_higher_order()
def scale_layout(layout, factor, axis = "both", attribute = "pos"):
   """
   Zoom a layout in (``factor`` above 1) or out (below 1) around its center (the middle of its
   bounding box), along all axes or only one, which stretches or squeezes the drawing.

   :param layout: A :class:`~graphoskop.Network` with positions in node attributes (changed in
      place), or a table of positions as returned by the layout functions (a new table is
      returned).
   :param factor: Positive scale factor.
   :param axis: ``"both"`` (all axes), ``"x"``, ``"y"`` or ``"z"`` (3-d layouts).
   :param attribute: With a network, the prefix of the position attributes.
   :return: :class:`pandas.DataFrame` of the new positions, as the layout functions return.
   :raises KeyError: If the network has no positions with that prefix.
   :raises ValueError: If the axis is unknown or the factor is not positive.
   """
   if not factor > 0:
      raise ValueError(f"factor must be positive, got {factor}")
   X, table = _positions(layout, attribute)
   for j in _axes(axis, X.shape[1]):
      c = (np.nanmin(X[:, j]) + np.nanmax(X[:, j])) / 2
      X[:, j] = c + factor * (X[:, j] - c)
   return _transformed(layout, X, table, attribute)


def _block_cg(L, B, X, tol = 1e-3, maxiter = 1000):
   """
   Solve ``L X = B`` for all columns at once by conjugate gradients with a Jacobi (diagonal)
   preconditioner, from the guess ``X``, until the residual falls ``tol`` times below its initial
   value (the guess is close: steps of stress majorization need no exact solution); ``L`` is
   sparse, symmetric and positive definite.
   """
   d = L.diagonal()[:, None]
   R = B - L @ X
   Z = R / d
   P = Z.copy()
   rz = (R * Z).sum(0)
   stop = tol * np.linalg.norm(R, axis = 0)
   for _ in range(maxiter):
      if (np.linalg.norm(R, axis = 0) <= stop).all():
         break
      Q = L @ P
      a = rz / np.maximum((P * Q).sum(0), 1e-300)
      X = X + a * P
      R = R - a * Q
      Z = R / d
      rz, old = (R * Z).sum(0), rz
      P = Z + (rz / np.maximum(old, 1e-300)) * P
   return X


def _overlapping(X, r, gap):
   """
   Pairs of nodes whose disks (radius ``r``, plus ``gap``) overlap, found by hashing the nodes
   into cells as large as the largest disk.

   :return: ``i``, ``j`` (``i < j``).
   """
   n, D = X.shape
   cell = 2 * r.max() + gap
   cell = max(cell, float(np.ptp(X, 0).max()) / (2 ** (62 // D) - 4))   # cell keys fit in int64
   c = np.floor(X / cell).astype(np.int64)
   c = c - c.min(0) + 1
   m = int(c.max()) + 2
   key = (c * m ** np.arange(D)).sum(1)
   order = np.argsort(key, kind = "stable")
   sk = key[order]
   near = np.array(list(itertools.product((-1, 0, 1), repeat = D)))
   nb = key[:, None] + (near * m ** np.arange(D)).sum(1)
   lo, hi = np.searchsorted(sk, nb), np.searchsorted(sk, nb, side = "right")
   cc = (hi - lo).reshape(-1)
   slot = np.repeat(np.arange(len(cc)), cc)
   i = slot // nb.shape[1]
   j = order[np.repeat(lo.reshape(-1), cc) + np.arange(len(slot)) - np.repeat(np.cumsum(cc) - cc, cc)]
   keep = (i < j) & (np.linalg.norm(X[i] - X[j], axis = 1) < r[i] + r[j] + gap)
   return i[keep], j[keep]


def _proximity(X):
   """
   Edges of the Delaunay triangulation of the positions (pairs ``i < j``): the neighbors whose
   relative positions the overlap removal keeps.
   """
   from scipy.spatial import Delaunay
   if len(X) <= X.shape[1] + 1:
      i, j = np.triu_indices(len(X), 1)
      return i, j
   S = Delaunay(X, qhull_options = "QJ").simplices   # joggled: collinear and repeated positions work
   pairs = np.concatenate([S[:, [a, b]] for a, b in itertools.combinations(range(S.shape[1]), 2)])
   pairs = np.sort(pairs, 1).astype(np.int64)   # qhull gives int32: i * n overflows above 46340 nodes
   key = np.unique(pairs[:, 0] * len(X) + pairs[:, 1])
   return key // len(X), key % len(X)


@_higher_order()
def remove_overlaps(layout, size = 1., margin = 0., attribute = "pos", max_iter = 1000):
   """
   Remove the overlaps between nodes drawn as disks (spheres in three dimensions), moving them as
   little as possible and keeping their relative positions, with the proximity stress model PRISM
   (Gansner and Hu, 2010).

   Pushing overlapping nodes apart creates new overlaps with their neighbors, and pushing those
   in turn spreads distortions; PRISM moves all nodes at once instead, keeping the layout's
   structure. The neighbors of every node are the edges of the Delaunay triangulation of the
   positions (the proximity graph): every proximity edge whose two disks overlap gets a desired
   length longer than its current one by the factor that would remove the overlap (at most 1.5
   at a time), every other edge its current length, and the layout is moved to the positions
   that best fit these lengths (stress majorization: least squares on the relative positions,
   weighted by the inverse squared lengths, solved by conjugate gradients). So nodes move apart
   where they overlap and their surroundings follow, with distances and orientations between
   neighbors kept as far as possible. The triangulation is recomputed and the steps repeated
   until no proximity edge overlaps; then pairs that still overlap without being neighbors in the
   triangulation are added to the proximity graph until no two disks overlap. Overlapping edges
   aim 5% beyond the overlap, which avoids a long tail of ever smaller pushes. The center of the
   layout does not move.

   The result is not the layout with the smallest total displacement, which would let overlapping
   groups squeeze through each other, but the one that removes overlaps with the least distortion
   of the arrangement (no node jumps across another). Nodes at the same position are first set
   apart by a tiny random offset (fixed seed).

   :param layout: A :class:`~graphoskop.Network` with positions in node attributes (changed in
      place), or a table of positions as returned by the layout functions (a new table is
      returned).
   :param size: Diameter of the nodes, in layout units: a number for all nodes, the name of a
      numeric node attribute (with a network), or one value per node (an array or a
      :class:`pandas.Series` indexed by node id).
   :param margin: Smallest empty space left between two nodes.
   :param attribute: With a network, the prefix of the position attributes.
   :param max_iter: Most rounds of triangulation and stress majorization; a warning says if
      overlaps remain after them.
   :return: :class:`pandas.DataFrame` of the new positions, as the layout functions return.
   :raises KeyError: If the network has no positions with that prefix, or no such size attribute.
   :raises ValueError: If sizes are negative or missing.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      gsk.force_directed_layout(net)
      gsk.remove_overlaps(net, 0.5)                    # disks of diameter 0.5 (edges ~1 long)
      gsk.remove_overlaps(net, "degree_size", margin = 0.05)
   """
   from scipy.sparse import coo_matrix, identity
   X, table = _positions(layout, attribute)
   n, D = X.shape
   if isinstance(size, str):
      r = torch.cat(layout.get_node_attribute(size, None)).double().cpu().numpy() / 2
   elif isinstance(size, pd.Series) and table is not None:
      r = size.reindex(table.index).to_numpy(dtype = float) / 2
   else:
      r = np.broadcast_to(np.asarray(size, dtype = float), (n,)) / 2
   if np.isnan(r).any() or (r < 0).any():
      raise ValueError("remove_overlaps: sizes must be non negative numbers")
   if n < 2 or r.max() == 0:
      return _transformed(layout, X, table, attribute)
   center = X.mean(0)
   key, inv = np.unique(np.round(X, 12), axis = 0, return_inverse = True)
   if len(key) < n:   # nodes at the same position
      X = X + np.random.default_rng(0).normal(0, 1e-6 * max(np.ptp(X), r.max()), X.shape)
   extra = np.zeros(0, dtype = np.int64)
   for it in range(max_iter):
      i, j = _proximity(X)
      d = np.linalg.norm(X[i] - X[j], axis = 1)
      if not (d < r[i] + r[j] + margin).any():   # no overlap along the triangulation
         oi, oj = _overlapping(X, r, margin)
         if not len(oi):
            break
         extra = np.union1d(extra, oi * n + oj)   # overlapping pairs that are not neighbors join the proximity graph
      if len(extra):
         key = np.union1d(i * n + j, extra)
         i, j = key // n, key % n
         d = np.linalg.norm(X[i] - X[j], axis = 1)
      need = r[i] + r[j] + margin
      target = d * np.where(d < need, np.clip(1.05 * need / np.maximum(d, 1e-12), 1, 1.5), 1.)   # 5% beyond the overlap: no slow tail
      w = 1 / target ** 2
      L = coo_matrix((np.concatenate([-w, -w]), (np.concatenate([i, j]), np.concatenate([j, i]))), shape = (n, n)).tocsr()
      L = L - coo_matrix((np.asarray(L.sum(1)).ravel(), (np.arange(n), np.arange(n))), shape = (n, n))
      L = (L + 1e-6 * w.mean() * identity(n)).tocsr()   # positive definite (the layout keeps its center)
      stress = np.inf
      for _ in range(_STEPS):   # stress majorization steps, while the stress decreases
         diff = X[i] - X[j]
         dist = np.maximum(np.linalg.norm(diff, axis = 1), 1e-12)
         now = (w * (dist - target) ** 2).sum()
         if now > stress * (1 - 1e-3):
            break
         stress = now
         f = (w * target / dist)[:, None] * diff
         B = np.column_stack([np.bincount(i, f[:, k], n) - np.bincount(j, f[:, k], n) for k in range(D)])
         X = _block_cg(L, B + 1e-6 * w.mean() * X, X)
   else:
      if len(_overlapping(X, r, margin)[0]):
         warnings.warn(f"remove_overlaps: overlaps remain after {max_iter} rounds")
   X = X - X.mean(0) + center
   return _transformed(layout, X, table, attribute)


def _sort_key(net, order, s, w, n, descending):
   """
   Values sorting the nodes: the sums of the weights of their edges (``order = "degree"``), or a
   numeric node attribute; negated if ``descending``, missing values last.
   """
   if order == "degree":
      v = np.bincount(s, weights = w, minlength = n)
   elif order in net.node_attributes:
      v = torch.cat(net.get_node_attribute(order, None)).double().cpu().numpy()
   else:
      raise ValueError(f"order must be 'degree' or a numeric node attribute, got {order!r}")
   v = -v if descending else v
   return np.where(np.isnan(v), np.inf, v)


def _sequence(perm, comp, G, gap):
   """
   Positions along the axis of the nodes sorted by ``perm``: 0, 1, 2, ... with ``gap`` times the
   number of levels below the one that changes empty places between groups (components count as
   one level above the first).
   """
   L = G.shape[1]
   keys = np.column_stack([comp, G])[perm]
   diff = keys[1:] != keys[:-1]
   level = np.where(diff.any(1), diff.argmax(1), L + 1) - 1   # -1: another component
   pos = np.empty(len(perm))
   pos[perm] = np.concatenate([[0.], np.cumsum(1 + gap * (L - level))])
   return pos


@_higher_order()
def matrix_layout(net, groups = None, order = None, column_groups = None, column_order = None, weight = None, descending = False, gap = 0., attribute = "matrix", arrange = True):
   """
   Order of the rows and columns of the adjacency matrix, to draw it (a matrix, or heat map,
   visualization): the counterpart of :func:`circular_layout` along a line, with the network's
   own structure, groups and numeric attributes placing similar rows and columns next to each
   other.

   **Square matrices** (one-mode networks): rows and columns follow the same order. With
   ``groups`` (one node attribute or a list, by decreasing importance), the nodes of every group
   are consecutive, inside the range of their parent group, and groups are arranged as the
   sectors of :func:`circular_layout`, along a line instead of a circle: groups with many edges
   between them are close (the sum of their edges times their distance is minimized), and groups
   with many edges to the previous (next) group of the level above come first (last); with
   ``arrange = False``, groups follow the sorted order of their values. Inside the finest groups,
   nodes are sorted by ``order``, or else by their edges to the previous and next group, from
   those with edges mostly to the previous one to those with edges mostly to the next one. Without
   ``groups`` and ``order``, nodes are sorted by the Fiedler vector of their component (the
   second eigenvector of the normalized Laplacian: spectral seriation), which puts densely
   connected nodes next to each other: communities become blocks on the diagonal, clearly when
   there are few of them (with many, give them as ``groups``, from any community detection).

   **Rectangular matrices** (bipartite networks): rows are the nodes of the first type, columns
   those of the second, and each axis has its own groups and order (``groups`` and ``order`` for
   the rows, ``column_groups`` and ``column_order`` for the columns, each a mix of qualitative
   and numeric attributes as above). Rows have no edges between them, so they are placed by
   the columns they are linked to, and columns by their rows: starting from the spectral
   co-clustering order (the Fiedler vector of the bipartite network, Dhillon 2001), each row
   moves to the mean position of its columns and each column to the mean position of its rows,
   alternately, 20 times (the barycenter method of two-layer drawings), groups moving as wholes
   (by the mean of their members, inside their parent group; with ``arrange = False`` in the
   order of their values) and nodes with an ``order`` sorted by it inside their finest group.
   Dense blocks gather near the diagonal.

   ``order = "degree"`` sorts by the sums of the rows (columns), the weighted degree with
   ``weight`` (in directed networks, in and out edges together); with ``descending = True``,
   the fullest rows come first, so ``matrix_layout(net, order = "degree", column_order =
   "degree", descending = True)`` draws a nested bipartite network as an upper triangular
   matrix (see :func:`~graphoskop.nestedness`).

   Components follow each other along the diagonal, largest first. The positions are stored as
   one numeric node attribute: the row of a node, which in square matrices is also its column; in
   bipartite networks the row of a node of the first type and the column of a node of the second.

   :param net: A :class:`~graphoskop.Network`.
   :param groups: Name of a node attribute, or list of names in order of importance, grouping
      the rows (and columns of square matrices), or ``None``.
   :param order: ``"degree"``, or the name of a numeric node attribute, sorting the rows (and
      columns of square matrices) inside their finest groups, or ``None``.
   :param column_groups: As ``groups``, for the columns of a bipartite network.
   :param column_order: As ``order``, for the columns of a bipartite network.
   :param weight: Name of a numeric edge attribute weighting the edges (in the arrangements, the
      barycenters and ``"degree"``), or ``None``.
   :param descending: Whether orders go from the largest value to the smallest.
   :param gap: Empty rows (columns) between groups, per level of groups below the level that
      changes (components count as one more level); 0 for none.
   :param attribute: Name of the numeric node attribute where to store the positions, or ``None``
      to only return them.
   :param arrange: Whether to arrange groups by their edges (``True``) or in sorted order of their
      values (``False``).
   :return: :class:`pandas.DataFrame` with one row per node, indexed by node id, in the network
      wide node order, with the column ``attribute`` (``index`` with ``attribute = None``), the
      position of the node, and, in bipartite networks, the column ``axis`` (``"row"`` or
      ``"column"``).
   :raises KeyError: If an attribute does not exist.
   :raises ValueError: If ``order`` is neither ``"degree"`` nor a numeric node attribute, or
      column criteria are given for a network that is not bipartite.

   Example:

   .. code-block:: python

      import graphoskop as gsk
      import matplotlib.pyplot as plt

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      pos = gsk.matrix_layout(net, "community")["matrix"]     # communities as diagonal blocks
      edges = [(pos[a], pos[b]) for a, b in net.edges(None, labels = True)]
      plt.scatter([b for a, b in edges] + [a for a, b in edges], [a for a, b in edges] + [b for a, b in edges], s = 1, marker = "s")
      plt.gca().invert_yaxis()

      gsk.matrix_layout(bip, order = "degree", column_order = "degree", descending = True)   # nested
      gsk.matrix_layout(bip, ["country", "city"], column_groups = "sector", column_order = "size")
   """
   sizes = np.diff(net.node_offsets)
   n = int(sizes.sum())
   comp = np.repeat(np.arange(len(sizes)), sizes)
   s, t, w = _all_edges(net, weight)
   tie = np.arange(n)

   def codes(g):
      g = [g] if isinstance(g, str) else list(g or [])
      return np.stack([_codes(net, a) for a in g], 1) if g else np.zeros((n, 0), dtype = np.int64)

   if net.bipartite_split is None:
      if column_groups is not None or column_order is not None:
         raise ValueError("column_groups and column_order need a bipartite network: square matrices have one order")
      G = codes(groups)
      if G.shape[1] and arrange:
         G = _arrange_groups(G, comp, s, t, w, cyclic = False)
      fine = _consecutive(np.column_stack([comp, G]))
      if order is not None:
         within = _sort_key(net, order, s, w, n, descending)
      elif G.shape[1]:
         within = _neighbor_balance(net, fine, weight, cyclic = False)
      else:
         from .gnn import _spectral
         within = _spectral(net, 1, weight, False)[:, 0]
      pos = _sequence(np.lexsort((tie, within, fine)), comp, G, gap)
      table = pd.DataFrame({attribute or "index": pos})
   else:
      local = tie - np.repeat(net.node_offsets[:-1], sizes)
      axis = (local >= np.repeat(net.bipartite_split, sizes)).astype(np.int64)   # 0 rows, 1 columns
      G = [codes(groups), codes(column_groups)]
      key = [None if o is None else _sort_key(net, o, s, w, n, descending) for o in (order, column_order)]
      from .gnn import _spectral
      p = _spectral(net, 1, weight, False)[:, 0]
      for k in range(21):   # barycenters: rows from their columns, columns from their rows
         a = k % 2
         A = np.flatnonzero(axis == a)
         if k:
            cnt = np.bincount(s, weights = w, minlength = n)
            bary = np.where(cnt > 0, np.bincount(s, weights = w * p[t], minlength = n) / np.where(cnt > 0, cnt, 1), p)
         else:
            bary = p
         g = G[a][A]
         cols = [comp[A]]
         for l in range(g.shape[1]):
            gid = _consecutive(np.column_stack([comp[A], g[:, :l + 1]]))
            mean = np.bincount(gid, weights = bary[A]) / np.bincount(gid)
            cols += [mean[gid] if arrange else g[:, l], g[:, l]]
         cols.append(bary[A] if key[a] is None else key[a][A])
         o = np.lexsort([tie[A]] + cols[::-1])
         p[A[o]] = np.arange(len(A))
      pos = np.empty(n)
      for a in (0, 1):
         A = np.flatnonzero(axis == a)
         o = np.argsort(p[A], kind = "stable")
         pos[A] = _sequence(o, comp[A], G[a][A], gap)
      table = pd.DataFrame({attribute or "index": pos, "axis": np.where(axis == 0, "row", "column")})
   if attribute is not None:
      T = torch.as_tensor(pos, dtype = torch.float, device = net.components[0].edge_index.device)
      net.set_node_attribute(attribute, [T[o:o2] for o, o2 in zip(net.node_offsets[:-1], net.node_offsets[1:])], None)
   ids = [x for c in range(len(net.components)) for x in net.node_ids[c].tolist()]
   table.index = pd.Index(ids, tupleize_cols = False, name = "node")
   return table
