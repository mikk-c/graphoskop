"""
Signed networks: construction, signed adjacency matrix and structural balance.

A signed network is an ordinary :class:`~graphoskop.Network` whose edges carry a sign, +1 (a
positive relationship: friendship, trust, alliance) or -1 (a negative one: enmity, distrust,
conflict), in a numeric edge attribute, by default :data:`SIGN`. It can be seen as a multilayer
network with two layers, positive and negative, flattened into one network of entities: an edge is
in exactly one of the two layers. :func:`signed_network` builds one in either of two ways:

* from a multilayer network, naming its positive and its negative layer. Two entities cannot be
  joined by an edge in both layers (in the same direction, if directed);
* from an ordinary network, thresholding an edge attribute: edges at or above the threshold are
  positive, edges below it negative.

The signed adjacency matrix :math:`A` (:func:`signed_adjacency`) has :math:`A_{ij}` equal to the
sign of the edge from :math:`i` to :math:`j`, and 0 without an edge; parallel edges count once (they
must have the same sign). Edges with sign 0 or NaN in the sign attribute are treated as absent.
"""

import numpy as np
import pandas as pd
import torch

from .network import LAYER, _quiet_sparse
from .multilayer import _tables, _flat_tables, _rebuild, _entity_nodes
from .network_statistics import _skeleton, _paths

#: Default name of the edge attribute holding the sign (+1 or -1) of each edge in signed networks.
SIGN = "sign"


def _conflicts(edges, signs, directed):
   """
   Check that no two entities are joined by both a positive and a negative edge.

   :param edges: Edge table with columns ``source`` and ``target``.
   :param signs: Sign of each edge (+1 or -1).
   :param directed: Whether edges are directed (if not, the order of the endpoints does not matter).
   :raises ValueError: If an edge is both positive and negative.
   """
   m = len(edges)
   codes, ids = pd.factorize(pd.concat([edges["source"], edges["target"]], ignore_index = True))
   a, b = codes[:m].astype(np.int64), codes[m:].astype(np.int64)
   if not directed:
      a, b = np.minimum(a, b), np.maximum(a, b)
   key = a * len(ids) + b
   both = np.intersect1d(key[signs > 0], key[signs < 0])
   if len(both):
      ids = np.asarray(ids)
      example = [(ids[k // len(ids)], ids[k % len(ids)]) for k in both[:5]]
      raise ValueError(f"{len(both)} edges are both positive and negative, e.g. {example}")


def signed_network(net, positive = None, negative = None, weight = None, threshold = 0, sign = SIGN):
   """
   Signed network, from two layers of a multilayer network or by thresholding an edge attribute.

   * ``positive`` and ``negative`` given: ``net`` must be multilayer. The result is a network of
     entities with the intra-layer edges of the two layers, signed +1 and -1. Couplings, other
     interlayer edges and the other layers are left out. An edge present in both layers (between
     the same two entities, in the same direction if directed) raises :class:`ValueError`: an
     edge cannot be positive and negative at the same time. Nodes are the entities with a copy in
     either layer; node attributes take the first value found in layer order, as in
     :func:`~graphoskop.aggregate_network`.
   * ``weight`` given: ``net`` must not be multilayer. Edges whose ``weight`` is greater than or
     equal to ``threshold`` are signed +1, the others -1; edges with a NaN ``weight`` are dropped.
     Parallel edges with opposite signs raise :class:`ValueError`. All nodes are kept, also those
     left without edges.

   All edge attributes are kept, and the sign is stored as the edge attribute ``sign`` (replaced
   if it exists). Components are recomputed, so they are the components of the signed network;
   node attributes and node types (bipartite networks) are kept, and so are the device and
   directedness.

   :param net: A :class:`~graphoskop.Network`.
   :param positive: Positive layer, one of ``net.layer_labels``.
   :param negative: Negative layer, one of ``net.layer_labels``.
   :param weight: Edge attribute (name or position in :attr:`~graphoskop.Network.edge_attributes`)
      to threshold.
   :param threshold: Threshold on ``weight``.
   :param sign: Name of the edge attribute storing the sign.
   :return: The signed network, a new :class:`~graphoskop.Network`.
   :raises ValueError: If both or neither of the two ways are given, if ``net`` is not multilayer
      (layers) or is multilayer (weight), if the two layers are the same, or if an edge is both
      positive and negative.
   :raises KeyError: If there is no such layer or edge attribute.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("relations.tsv", multilayer = True)       # layers "ally", "enemy"
      signed = gsk.signed_network(net, positive = "ally", negative = "enemy")

      net = gsk.load_network("votes.tsv")                               # column "score"
      signed = gsk.signed_network(net, weight = "score", threshold = 0.5)
      gsk.structural_balance(signed)
   """
   by_layer = positive is not None or negative is not None
   if by_layer == (weight is not None):
      raise ValueError("give either the positive and negative layers, or a weight and a threshold")
   if by_layer:
      if positive is None or negative is None:
         raise ValueError("give both the positive and the negative layer")
      if positive == negative:
         raise ValueError("the positive and the negative layer must be different")
      edges, _, nodes = _tables(net)
      for layer in (positive, negative):
         if layer not in net.layer_labels:
            raise KeyError(f"no layer {layer!r}; layers: {net.layer_labels}")
      pos, neg = (edges[LAYER] == positive).to_numpy(), (edges[LAYER] == negative).to_numpy()
      edges = edges[pos | neg].drop(columns = LAYER)
      signs = np.where(pos[pos | neg], 1, -1)
      nodes = _entity_nodes(net, nodes[nodes[LAYER].isin([positive, negative]).to_numpy()])
   else:
      if net.layer_labels is not None:
         raise ValueError("the network is multilayer: give the positive and negative layers, or threshold one layer (layer_network)")
      j = net._column(net.edge_attributes, weight, "edge")
      edges, nodes = _flat_tables(net)
      w = edges.iloc[:, 2 + j].to_numpy()
      keep = ~np.isnan(w)
      edges, signs = edges[keep], np.where(w[keep] >= threshold, 1, -1)
   _conflicts(edges, signs, net.directed)
   edges = edges.assign(**{sign: signs.astype(np.float32)})
   return _rebuild(net, edges, nodes)


def _signed_edges(net, component, sign):
   """
   Nonzero entries of the signed adjacency matrix of a component.

   :param net: Signed network.
   :param component: Component index.
   :param sign: Name or position of the sign edge attribute.
   :return: ``(source, target, value)``, sorted by ``source * n + target``; values are int8 (+1
      or -1), on the device of the component.
   :raises ValueError: If the network is multilayer, or parallel edges have opposite signs.
   """
   if net.layer_labels is not None:
      raise ValueError("the network is multilayer: build a signed network first (signed_network)")
   data = net.components[component]
   s, t = data.edge_index
   v = torch.sign(net._numeric_edge(sign, component)).nan_to_num(0)
   keep = v != 0
   n = data.num_nodes
   key, inv = torch.unique(s[keep] * n + t[keep], return_inverse = True)
   v = v[keep]
   lo = torch.full(key.shape, 2, dtype = v.dtype, device = v.device).scatter_reduce(0, inv, v, "amin")
   hi = torch.full(key.shape, -2, dtype = v.dtype, device = v.device).scatter_reduce(0, inv, v, "amax")
   if bool((lo != hi).any()):
      raise ValueError(f"parallel edges with opposite signs in component {component}")
   return key // n, key % n, hi.to(torch.int8)


def signed_adjacency(net, component = 0, sign = SIGN):
   """
   Signed adjacency matrix of a component of a signed network.

   :math:`A_{ij}` is the sign (+1 or -1) of the edge from :math:`i` to :math:`j`, 0 without an
   edge; the matrix is symmetric in undirected networks. Parallel edges count once, self loops are
   kept on the diagonal. Rows and columns are ordered by internal node index.

   :param net: A signed :class:`~graphoskop.Network` (see :func:`signed_network`).
   :param component: Index of the component (0 is the largest), or ``None`` for a list with one
      matrix per component.
   :param sign: Edge attribute (name or position) holding the sign.
   :return: Sparse COO tensor (coalesced), ``n`` by ``n``, in the dtype of the edge attributes and
      on the device of the component.
   :raises ValueError: If the network is multilayer, or parallel edges have opposite signs.
   :raises KeyError: If there is no such edge attribute.

   Example:

   .. code-block:: python

      A = gsk.signed_adjacency(signed)
      A.to_dense()
   """
   if component is None:
      return [signed_adjacency(net, c, sign) for c in range(len(net.components))]
   s, t, v = _signed_edges(net, component, sign)
   n = net.components[component].num_nodes
   dtype = net.components[component].edge_attr.dtype
   with _quiet_sparse():
      return torch.sparse_coo_tensor(torch.stack([s, t]), v.to(dtype), (n, n), is_coalesced = True)




# Directed triangles: the 8 ways to pick one direction per skeleton edge of a triangle a -> b -> c
# (closed by a -> c). Each gives a cycle (positions in cycle order) or a feedforward loop
# (positions source -> middle, middle -> sink, source -> sink), as (slot, against) per position.
_COMBOS = [
   ("ffl", ((0, 0), (1, 0), (2, 0))), ("cycle", ((0, 0), (1, 0), (2, 1))),
   ("ffl", ((2, 0), (1, 1), (0, 0))), ("ffl", ((2, 1), (0, 0), (1, 1))),
   ("ffl", ((0, 1), (2, 0), (1, 0))), ("ffl", ((1, 0), (2, 1), (0, 1))),
   ("cycle", ((0, 1), (1, 1), (2, 0))), ("ffl", ((1, 1), (0, 1), (2, 1))),
]


def _labels(directed):
   """
   Triangle types, in the order of the counts of :func:`_census`.

   :param directed: Whether the network is directed.
   :return: List of labels.
   """
   cls = ["+++", "++-", "+--", "---"]
   if not directed:
      return cls
   return ["cycle " + c for c in cls] + ["feedforward " + "".join("-" if k >> i & 1 else "+" for i in (2, 1, 0)) for k in range(8)]


def _census(net, component, sign, batch_size, null):
   """
   Signed triangle counts of a component, and the statistics needed for their null model.

   :param net: Signed network.
   :param component: Component index.
   :param sign: Name or position of the sign edge attribute.
   :param batch_size: Paths expanded at once.
   :param null: Also collect the statistics of the null model.
   :return: Dictionary with ``counts`` (one per label of :func:`_labels`), and with ``null``:
      ``edges`` and ``negative`` (signed edges, self loops excluded, and negative ones), ``share``
      (ordered pairs of triangles sharing an edge, by position types: 4 by 4, types cycle,
      source -> middle, middle -> sink, source -> sink; one type in undirected networks) and
      ``flips`` (ordered pairs of feedforward loops sharing two edges, flipping the first or the
      second position); all Python ints.
   """
   s, t, v = _signed_edges(net, component, sign)
   n = net.components[component].num_nodes
   keep = s != t
   s, t, v = s[keep], t[keep], v[keep]
   dev, directed = s.device, net.directed
   src, dst, slot, along = _skeleton(s, t, n)
   m = src.shape[0]
   fb = torch.zeros((m, 2), dtype = torch.long, device = dev).index_put_((slot, (~along).long()), v.long())
   f = fb[:, 0]   # sign along (column 0) and against (column 1) each skeleton edge, 0 if absent
   counts = torch.zeros(12 if directed else 4, dtype = torch.long, device = dev)
   uses = torch.zeros((2 * m, 4) if directed else (m, 1), dtype = torch.long, device = dev)
   flips = torch.zeros(2, dtype = torch.long, device = dev)
   for slots in _paths(src, dst, n, batch_size) if m else ():
      if not directed:
         neg = sum((f[e] < 0).long() for e in slots)
         counts += torch.bincount(neg, minlength = 4)
         if null:
            uses[:, 0] += sum(torch.bincount(e, minlength = m) for e in slots)
         continue
      for shape, pos in _COMBOS:
         sg = [fb[slots[k], d] for k, d in pos]
         ok = (sg[0] != 0) & (sg[1] != 0) & (sg[2] != 0)
         neg = [(x[ok] < 0).long() for x in sg]
         cls = neg[0] + neg[1] + neg[2] if shape == "cycle" else 4 + 4 * neg[0] + 2 * neg[1] + neg[2]
         counts += torch.bincount(cls, minlength = 12)
         if null:
            for j, (k, d) in enumerate(pos):
               uses[:, 0 if shape == "cycle" else j + 1] += torch.bincount(2 * slots[k][ok] + d, minlength = 2 * m)
            if shape == "ffl":   # reciprocated first or second position: a twin sharing two edges
               for j in (0, 1):
                  k, d = pos[j]
                  flips[j] += int((fb[slots[k][ok], 1 - d] != 0).sum())
   out = {"counts": counts.tolist()}
   if null:
      out["edges"] = int((fb != 0).sum()) if directed else m
      out["negative"] = int((fb < 0).sum()) if directed else int((f < 0).sum())
      k = uses.shape[1]
      out["share"] = [[int((uses[:, i] * uses[:, j]).sum()) - (i == j) * int(uses[:, i].sum()) for j in range(k)] for i in range(k)]
      out["flips"] = flips.tolist()
   return out


def _falling(x, k):
   """
   Falling factorial :math:`x (x - 1) \\cdots (x - k + 1)`.
   """
   r = 1
   for i in range(k):
      r *= x - i
   return r


def _null_moments(patterns, edges, negative, total, share, flips):
   """
   Exact mean and variance of the number of triangles of one type when the signs are shuffled.

   Signs are a uniformly random permutation of the observed ones over the signed edges, so a given
   assignment of ``j`` negative signs to ``d`` distinct edges has probability
   :math:`[N]_j [M - N]_{d - j} / [M]_d` (falling factorials, :math:`M` edges, :math:`N`
   negative). The variance sums the covariances of all ordered pairs of triangles, grouped by the
   edges they share: none, one (by its position in each triangle) or two (feedforward loops on the
   same three nodes). Exact rational arithmetic.

   :param patterns: Sign patterns (tuples of +1 / -1, one per position) of the type.
   :param edges: Number of signed edges.
   :param negative: Number of negative edges.
   :param total: Number of triangles of the shape of the type (all, cycles, or feedforward loops).
   :param share: ``{(i, j): pairs}``, ordered pairs of triangles of that shape sharing exactly the
      edge at position ``i`` of the first and ``j`` of the second.
   :param flips: Ordered pairs of feedforward loops sharing two edges, the first position flipped
      and the second position flipped (zeros for other shapes).
   :return: ``(mean, variance)`` as floats.
   """
   from fractions import Fraction

   def h(d, j):
      den = _falling(edges, d)
      return Fraction(_falling(negative, j) * _falling(edges - negative, d - j), den) if den else Fraction(0)

   def both(d, cond, shared):   # both triangles of the type; shared: negative signs counted twice
      return sum(h(d, neg(x) + neg(y) - shared(x)) for x in patterns for y in patterns if cond(x, y))

   neg = lambda x: sum(y < 0 for y in x)
   p = sum(h(3, neg(x)) for x in patterns)
   var, rest = total * p * (1 - p), total * (total - 1)
   for (i, j), c in share.items():
      var += c * (both(5, lambda x, y: x[i] == y[j], lambda x: x[i] < 0) - p * p)
      rest -= c
   pset = set(patterns)
   for c, twin in zip(flips, (lambda x, w: (w, x[2], x[1]), lambda x, w: (x[2], w, x[0]))):
      var += c * (sum(h(4, neg(x) + (w < 0)) for x in patterns for w in (1, -1) if twin(x, w) in pset) - p * p)
      rest -= c
   var += rest * (both(6, lambda x, y: True, lambda x: 0) - p * p)
   return float(total * p), float(var)


def signed_triangle_counts(net, component = None, sign = SIGN, batch_size = 2 ** 22):
   """
   Number of triangles of each sign pattern, compared with randomly shuffled signs.

   In undirected networks there are four types, by the signs of their edges: ``"+++"``, ``"++-"``,
   ``"+--"`` and ``"---"``. In directed networks a triangle is three directed edges joining three
   nodes, one per pair (a pair joined in both directions is in two triangles). It is either a
   cycle :math:`i \\to j \\to k \\to i`, with types ``"cycle +++"`` to ``"cycle ---"`` as above, or
   a feedforward loop :math:`i \\to j \\to k`, :math:`i \\to k`, with eight types such as
   ``"feedforward +-+"``, the signs given in the order source to middle, middle to sink, source
   to sink. Which types are balanced is left to the user; :func:`structural_balance` takes
   ``"+++"`` and ``"+--"`` (cycles, in directed networks).

   Each count is compared with its expected value when the signs are shuffled at random among the
   edges, keeping the number of positive and negative edges (self loops are left out). The mean
   and the variance under this null model are computed exactly, without sampling: the variance
   accounts for triangles sharing edges, from how many triangles use each edge in each position.
   ``ratio`` is the count over its expected value (above 1: overrepresented), ``z`` the count minus
   its expected value, over the standard deviation. Self loops are ignored and parallel edges count
   once. The triangles are found as in :func:`structural_balance`; the null model adds memory for
   4 integers per edge.

   :param net: A signed :class:`~graphoskop.Network` (see :func:`signed_network`).
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network
      (signs are then shuffled over all its edges).
   :param sign: Edge attribute (name or position) holding the sign.
   :param batch_size: Number of 2-paths expanded at once; larger is faster, especially on GPU,
      and uses more memory.
   :return: :class:`pandas.DataFrame` with columns ``triangle`` (the type), ``count``,
      ``expected``, ``ratio`` and ``z``; NaN where the expected value or the variance is 0.
   :raises ValueError: If the network is multilayer, or parallel edges have opposite signs.
   :raises KeyError: If there is no such edge attribute.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      signed = gsk.signed_network(gsk.load_network("ratings.tsv"), weight = "score", threshold = 3)
      gsk.signed_triangle_counts(signed)
      #   triangle  count  expected  ratio      z
      # 0      +++   8120    7311.4   1.11   12.0
      # ...
   """
   comps = range(len(net.components)) if component is None else [component]
   parts = [_census(net, c, sign, batch_size, True) for c in comps]
   directed = net.directed
   k = 4 if directed else 1
   counts = np.sum([x["counts"] for x in parts], 0) if parts else np.zeros(12 if directed else 4, dtype = np.int64)
   edges, negative = sum(x["edges"] for x in parts), sum(x["negative"] for x in parts)
   share = [[sum(x["share"][i][j] for x in parts) for j in range(k)] for i in range(k)]
   ab, bc = (sum(x["flips"][j] for x in parts) for j in (0, 1)) if directed else (0, 0)
   if directed:   # twins counted in share at both shared edges: keep pairs sharing exactly one
      share[2][3] -= ab
      share[3][2] -= ab
      share[3][1] -= bc
      share[1][3] -= bc
   rows = []
   for c, label in enumerate(_labels(directed)):
      kind, sg = label.split() if directed else ("", label)
      if kind == "feedforward":
         patterns = [tuple(1 if x == "+" else -1 for x in sg)]
         total = int(counts[4:].sum())
         sh = {(i, j): share[i + 1][j + 1] for i in range(3) for j in range(3)}
         fl = (ab, bc)
      else:
         patterns = [p for p in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)) if sum(x < 0 for x in p) == sg.count("-")]
         total = int(counts[:4].sum())
         sh, fl = {(0, 0): share[0][0]}, (0, 0)
      mean, var = _null_moments(patterns, edges, negative, total, sh, fl)
      rows.append((label, int(counts[c]), mean))
      rows[-1] += ((counts[c] / mean) if mean else float("nan"), (counts[c] - mean) / np.sqrt(var) if var > 0 else float("nan"))
   return pd.DataFrame(rows, columns = ["triangle", "count", "expected", "ratio", "z"])


def _expm_multiply(A, X, beta, steps, shift):
   """
   :math:`e^{-s} (e^{\\beta A} - I - \\beta A - \\beta^2 A^2 / 2 - \\beta^3 A^3 / 6) X`, column by
   column, with the Arnoldi method (full reorthogonalization): the walks of length 4 or more.

   :param A: Sparse ``n`` by ``n`` matrix, float64.
   :param X: Dense ``n`` by ``p`` matrix.
   :param beta: Factor :math:`\\beta`.
   :param steps: Krylov dimension.
   :param shift: Shift :math:`s` (avoids overflow), or ``None`` to take the largest real part of
      the Ritz values of :math:`\\beta A`.
   :return: ``(result, shift)``.
   """
   n, p = X.shape
   steps = min(steps, n)
   norm = X.norm(dim = 0)
   V = X.new_zeros((p, steps, n))   # Krylov basis of each column
   H = X.new_zeros((p, steps, steps))
   V[:, 0] = (X / norm.clamp(min = 1e-300)).T
   for j in range(steps):
      w = (A @ V[:, j].T).T.unsqueeze(2)
      for _ in range(2):   # classical Gram-Schmidt, twice
         h = torch.bmm(V[:, :j + 1], w)
         H[:, :j + 1, j] += h[:, :, 0]
         w = w - torch.bmm(V[:, :j + 1].transpose(1, 2), h)
      if j + 1 < steps:
         w = w[:, :, 0]
         nrm = w.norm(dim = 1)
         H[:, j + 1, j] = nrm
         ok = nrm > 1e-12 * H[:, :j + 2, j].abs().amax(1).clamp(min = 1e-300)
         V[:, j + 1] = torch.where(ok[:, None], w / nrm.clamp(min = 1e-300)[:, None], 0)
   if shift is None:
      shift = float(torch.linalg.eigvals(beta * H).real.max())
   B, I = beta * H, torch.eye(steps, dtype = H.dtype, device = H.device)
   B2 = B @ B
   E = torch.linalg.matrix_exp(B - shift * I) - np.exp(-shift) * (I + B + B2 / 2 + B2 @ B / 6)
   return norm * torch.bmm(V.transpose(1, 2), E[:, :, :1])[:, :, 0].T, shift


def _trace_exp(s, t, v, n, beta, probes, steps, batch_size, tr3):
   """
   :math:`\\mathrm{Tr}(e^{\\beta M})` for a matrix given by its nonzero entries (zero diagonal), as
   ``(shift, value)`` with the trace equal to ``value * exp(shift)``.

   Exact (from the eigenvalues) if ``probes`` is ``None``. Otherwise the walks of length up to 3 are
   counted exactly (:math:`n + \\beta^2 \\mathrm{Tr}(M^2) / 2 + \\beta^3 \\mathrm{Tr}(M^3) / 6`;
   :math:`\\mathrm{Tr}(M) = 0`), and the trace of the rest, the walks of length 4 or more, is estimated with
   Hutch++ (Meyer et al., 2021): exact on the span :math:`Q` of the rest applied to the first block
   of random vectors, and estimated with the second block projected out of :math:`Q`.

   :param probes: ``(S, G)``, two ``n`` by ``k`` blocks of random signs, or ``None``.
   :param tr3: :math:`\\mathrm{Tr}(M^3)`, from the triangle counts.
   :return: ``(shift, value)``, floats.
   """
   dev = s.device
   if probes is None:
      M = torch.zeros((n, n), dtype = torch.float64, device = dev)
      M[s, t] = v
      lam = beta * (torch.linalg.eigvals(M) if bool((M != M.T).any()) else torch.linalg.eigvalsh(M))
      shift = float(lam.real.max())
      return shift, float(torch.exp(lam - shift).real.sum())
   with _quiet_sparse():
      M = torch.sparse_coo_tensor(torch.stack([s, t]), v, (n, n)).coalesce().to_sparse_csr()
   chunk = max(1, batch_size // (steps * n))

   def apply(X, shift):
      out = []
      for i in range(0, X.shape[1], chunk):
         Y, shift = _expm_multiply(M, X[:, i:i + chunk], beta, steps, shift)
         out.append(Y)
      return torch.cat(out, 1), shift

   S, G = probes
   Y, shift = apply(S, None)
   Q = torch.linalg.qr(Y).Q
   G = G - Q @ (Q.T @ G)
   rest = float((Q * apply(Q, shift)[0]).sum() + (G * apply(G, shift)[0]).sum() / G.shape[1])
   key = s * n + t   # sorted
   rev = t * n + s
   i = torch.searchsorted(key, rev).clamp(max = max(key.shape[0] - 1, 0))
   tr2 = float((v * v[i] * (key[i] == rev)).sum()) if key.shape[0] else 0.
   return shift, (n + beta ** 2 * tr2 / 2 + beta ** 3 * tr3 / 6) * np.exp(-shift) + rest


def structural_balance(net, component = None, sign = SIGN, beta = None, exact = None, samples = 60, steps = 40, seed = 0, batch_size = 2 ** 22):
   """
   Structural balance of a signed network, over its triangles or over all its closed walks.

   With ``beta = None``, over triangles:

   .. math::

      \\frac{\\mathrm{Tr}(A^3)}{\\mathrm{Tr}(|A|^3)} = \\frac{B - U}{B + U},

   with :math:`A` the signed adjacency matrix (:func:`signed_adjacency`), :math:`B` the number of
   balanced triangles (the product of the signs of their edges is positive: ``"+++"`` and
   ``"+--"``) and :math:`U` the number of unbalanced ones (``"++-"`` and ``"---"``): 1 if all
   triangles are balanced, -1 if none is, NaN if there are no triangles. In directed networks the
   triangles are the directed 3-cycles :math:`i \\to j \\to k \\to i`. For the counts of each
   type, and their comparison with shuffled signs, see :func:`signed_triangle_counts`. The traces
   are computed without forming :math:`A^3`: the pairs of adjacent nodes are oriented from lower to
   higher degree, and each triangle is found once by expanding the 2-paths along the orientation
   (the work grows with :math:`m^{3/2}` at most, even with hubs). Paths are expanded in batches of
   ``batch_size`` (about 50 bytes per path). Counts are exact (integers).

   With ``beta`` given, over all closed walks (walk-based balance, Estrada and Benzi, 2014):

   .. math::

      \\frac{\\mathrm{Tr}(e^{\\beta A})}{\\mathrm{Tr}(e^{\\beta |A|})}
      = \\frac{\\sum_k \\frac{\\beta^k}{k!} \\mathrm{Tr}(A^k)}{\\sum_k \\frac{\\beta^k}{k!} \\mathrm{Tr}(|A|^k)},

   in :math:`(0, 1]` for undirected networks. Closed walks of length :math:`k` weigh
   :math:`\\beta^k / k!`, so longer cycles count less, and more so the smaller :math:`\\beta` (even
   with :math:`\\beta = 1` a walk of length :math:`k` weighs :math:`1 / k!`). The traces include
   the walks of length 0 (one per node, isolated nodes included) and 2 (always positive), so the
   ratio tends to 1 as :math:`\\beta \\to 0`; as :math:`\\beta` grows it is dominated by the
   largest eigenvalues. Traces are computed per component and summed:

   * exactly, from the eigenvalues of the dense matrix (:math:`O(n^3)`), for components with at most
     2048 nodes (``exact = None``), or for all components (``exact = True``);
   * otherwise by stochastic estimation (Hutch++, Meyer et al., 2021), without forming
     :math:`e^{\\beta A}`: :math:`e^{\\beta A}` is applied to ``samples`` random vectors with
     ``steps`` Arnoldi iterations each (sparse matrix products, on the device of the network). The
     walks of length up to 3 are counted exactly (length 3 from the triangles); for the longer ones, the dominant directions found
     with the first half of the vectors are traced exactly, and only the remainder is estimated with
     the second half. :math:`A` and :math:`|A|` use the same random vectors, so that their errors
     partly cancel in the ratio. Memory is about ``8 * steps * n`` bytes per vector, with vectors
     processed in groups of at most ``batch_size / (steps * n)``. More ``samples`` reduce the
     random error, more ``steps`` the truncation error (needed for large :math:`\\beta \\lambda_{max}`).
     Results depend on ``seed``.

   Self loops are ignored and parallel edges count once.

   :param net: A signed :class:`~graphoskop.Network` (see :func:`signed_network`).
   :param component: Index of the component (0 is the largest), or ``None`` for the whole network.
   :param sign: Edge attribute (name or position) holding the sign.
   :param beta: ``None`` for triangles, or a positive number for closed walks.
   :param exact: With ``beta``: ``None`` (exact for small components only), ``True`` or ``False``.
   :param samples: With ``beta``: number of random vectors of the stochastic estimate.
   :param steps: With ``beta``: Arnoldi iterations per vector.
   :param seed: With ``beta``: seed of the random vectors.
   :param batch_size: Number of 2-paths expanded at once (triangles), or bound on the number of
      entries of the Krylov bases held at once (walks); larger is faster, especially on GPU, and
      uses more memory.
   :return: The balance, a float.
   :raises ValueError: If the network is multilayer, parallel edges have opposite signs, or
      ``beta`` is not positive.
   :raises KeyError: If there is no such edge attribute.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("relations.tsv", multilayer = True)
      signed = gsk.signed_network(net, positive = "ally", negative = "enemy")
      gsk.structural_balance(signed)                  # triangles: 0.62
      gsk.structural_balance(signed, beta = 0.5)      # all closed walks
   """
   comps = range(len(net.components)) if component is None else [component]
   if beta is None:
      c = np.sum([_census(net, x, sign, batch_size, False)["counts"][:4] for x in comps], 0) if len(comps) else np.zeros(4)
      total = c.sum()
      return float((c[0] - c[1] + c[2] - c[3]) / total) if total else float("nan")
   if not beta > 0:
      raise ValueError("beta must be positive")
   gen = torch.Generator().manual_seed(seed)
   parts = []
   for x in comps:
      s, t, v = _signed_edges(net, x, sign)
      keep = s != t
      s, t, v = s[keep], t[keep], v[keep].double()
      n = net.components[x].num_nodes
      probes, tr3 = None, (0, 0)
      if not (n <= 2048 if exact is None else exact):   # the same random vectors for A and |A|
         k = max(1, samples // 2)
         probes = [(torch.randint(0, 2, (n, k), generator = gen, dtype = torch.float64) * 2 - 1).to(s.device) for _ in range(2)]
         c = _census(net, x, sign, batch_size, False)["counts"][:4]
         f = 3 if net.directed else 6   # closed walks per triangle (directed: per 3-cycle)
         tr3 = (f * (c[0] - c[1] + c[2] - c[3]), f * sum(c))
      parts.append([_trace_exp(s, t, w, n, beta, probes, steps, batch_size, x3) for w, x3 in zip((v, v.abs()), tr3)])
   if not parts:
      return float("nan")
   top = max(sh for p in parts for sh, _ in p)
   num = sum(val * np.exp(sh - top) for (sh, val), _ in parts)
   den = sum(val * np.exp(sh - top) for _, (sh, val) in parts)
   return float(num / den)
