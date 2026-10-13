"""
Analysis of vectors defined on the nodes of a network: generalized Euclidean distance, network
correlation, variance and entropy.

In multilayer networks, vectors are defined on the copies ``(entity, layer)`` of a component, and
distances use the supra-Laplacian (see :doc:`laplacians`): this is the multilayer generalized
Euclidean distance (Coscia, 2022). An attribute given for a whole entity has its value on every
copy; for the probability distributions of :func:`network_variance` and :func:`network_entropy`,
this gives more mass to the entities present in more layers.
"""

import torch

from .network import _quiet_sparse
from .utils import _vector, effective_resistance, laplacian_pinv, laplacian_solver


_BLOCK_ELEMENTS = 2 ** 23   # entries of each n by b (or k by b) block handled at once by the pairwise functions


class _Attributes:
   """
   Node attributes of a component, centered, and read one block of attributes at a time.

   The full ``n`` by ``k`` table is never densified or copied: sparse (long format) tables stay
   sparse, and centering is applied block by block (or, for products, as a rank one correction).

   :param X: ``node_attr`` of the component (dense or sparse CSR).
   :param center: ``"rows"`` to subtract from each node its mean over the attributes (missing
      values ignored), ``"columns"`` to subtract from each attribute its mean over the nodes.
   """

   def __init__(self, X, center):
      self.n, self.k = X.shape
      self.sparse = X.layout == torch.sparse_csr
      self.device = X.device
      self.center = center
      if self.sparse:
         with _quiet_sparse():
            X = X.to_sparse_coo()
         r, c = X.indices()
         order = torch.argsort(c, stable = True)
         self.r, self.c, self.v = r[order], c[order], X.values()[order].double()
         self.bounds = torch.searchsorted(self.c, torch.arange(self.k + 1, device = self.device)).tolist()
         if center == "rows":
            self.m = torch.zeros(self.n, dtype = torch.float64, device = self.device).index_add_(0, self.r, self.v) / self.k
         else:
            self.m = torch.zeros(self.k, dtype = torch.float64, device = self.device).index_add_(0, self.c, self.v) / self.n
      else:
         self.X = X
         if center == "rows":
            total = torch.zeros(self.n, dtype = torch.float64, device = self.device)
            count = torch.zeros(self.n, dtype = torch.float64, device = self.device)
            for J in self.blocks(_BLOCK_ELEMENTS // max(self.n, 1)):
               B = X[:, J].double()
               total += B.nan_to_num().sum(1)
               count += B.isfinite().sum(1)
            self.m = total / count
         else:
            self.m = torch.cat([X[:, J].double().mean(0) for J in self.blocks(_BLOCK_ELEMENTS // max(self.n, 1))])

   def blocks(self, b):
      """
      Slices of ``b`` attributes covering all attributes.
      """
      b = max(1, b)
      return [slice(i, min(i + b, self.k)) for i in range(0, self.k, b)]

   def block(self, J):
      """
      Dense centered ``n`` by ``|J|`` float64 block of attributes.
      """
      if self.sparse:
         a, e = self.bounds[J.start], self.bounds[J.stop]
         B = torch.zeros(self.n, J.stop - J.start, dtype = torch.float64, device = self.device)
         B.index_put_((self.r[a:e], self.c[a:e] - J.start), self.v[a:e])
      else:
         B = self.X[:, J].double()
      return B - (self.m[:, None] if self.center == "rows" else self.m[None, J])

   def xty(self, stop, Y, b):
      """
      :math:`\\hat{X}_{:, :stop}^T Y`, with :math:`\\hat{X}` the centered attributes.
      """
      if not self.sparse:
         return torch.cat([self.block(I).T.to(Y.dtype) @ Y for I in self.blocks(b) if I.start < stop])[:stop]
      e = self.bounds[stop]
      with _quiet_sparse():
         XT = torch.sparse_coo_tensor(torch.stack([self.c[:e], self.r[:e]]), self.v[:e].to(Y.dtype), (stop, self.n))
      P = XT @ Y
      if self.center == "rows":
         return P - (self.m.to(Y.dtype) @ Y)[None, :]
      return P - self.m[:stop, None].to(Y.dtype) * Y.sum(0)[None, :]


def _pairwise(attrs, apply, finish, output_device, output_dtype):
   """
   Build a symmetric ``k`` by ``k`` result one block of attributes at a time.

   For each block :math:`J` of attributes, ``apply`` gives :math:`S_J` (an ``n`` by ``|J|`` product
   with :math:`L^+` or :math:`W`), and the entries between :math:`J` and all previous attributes are
   finished from :math:`\\hat{X}^T S_J` and written, with their mirror images, into the result on
   ``output_device``. Only a few ``n`` by ``|J|`` and ``k`` by ``|J|`` blocks are on the compute
   device at a time.

   :param attrs: :class:`_Attributes`.
   :param apply: Function from a centered block to :math:`S_J`.
   :param finish: Function from (:math:`\\hat{X}_{:, :stop}^T S_J`, diagonal terms up to ``stop``,
      diagonal terms of :math:`J`) to the result entries.
   :param output_device: Device of the result.
   :param output_dtype: Dtype of the result.
   :return: Result.
   """
   k = attrs.k
   b = _BLOCK_ELEMENTS // max(attrs.n, k, 1)
   out = torch.empty(k, k, dtype = output_dtype, device = output_device)
   d = torch.empty(k, dtype = torch.float64, device = attrs.device)
   for J in attrs.blocks(b):
      XJ = attrs.block(J)
      S = apply(XJ)
      d[J] = (XJ.to(S.dtype) * S).sum(0).real
      R = finish(attrs.xty(J.stop, S, b).real, d[:J.stop], d[J]).to(output_dtype).to(output_device)
      out[:J.stop, J] = R
      out[J, :J.stop] = R.T
   return out, d


def generalized_euclidean(net, a, b = None, component = 0, lpinv = None, sparse = False, weight = None, kind = "out", q = 0.25, alpha = 0.15):
   """
   Generalized Euclidean distance :math:`\\sqrt{(a - b)^H L^+ (a - b)}` between two node vectors,
   or norm :math:`\\sqrt{a^H L^+ a}` of a single one, with :math:`L^+` the pseudoinverse of the
   Laplacian (see :func:`~graphoskop.laplacian` for the available Laplacians).

   :math:`L^+` is used as given in ``lpinv``, either as a dense tensor (from
   :func:`~graphoskop.laplacian_pinv`) or as a solver (from
   :func:`~graphoskop.laplacian_solver`). If ``lpinv`` is ``None`` it is computed,
   densely or, if ``sparse``, as a solver. Pass ``lpinv`` to reuse it across calls.

   For Hermitian Laplacians (undirected, ``"magnetic"``, ``"chung"``) the quadratic form is non
   negative. The ``"out"`` Laplacian is not symmetric, so the quadratic form can be negative, and
   the result is then NaN. Missing attribute values propagate as NaN.

   :param net: A :class:`~graphoskop.Network`.
   :param a: Node attribute (position in ``net.node_attributes`` or name), or vector of length
      ``n`` (:class:`numpy.ndarray` or :class:`torch.Tensor`).
   :param b: Second node vector, same modalities as ``a``, or ``None``.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
      With ``None``, vectors and ``lpinv`` are given as lists with one entry per component.
   :param lpinv: Cached pseudoinverse or solver, or ``None``.
   :param sparse: Compute a solver instead of the dense pseudoinverse when ``lpinv`` is ``None``.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param kind: Laplacian of directed networks: ``"out"``, ``"magnetic"`` or ``"chung"``.
   :param q: Charge of the magnetic Laplacian.
   :param alpha: Teleportation probability of the Chung Laplacian.
   :return: Distance, or list of distances if ``component`` is ``None``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      d = gsk.generalized_euclidean(net, "age", "height")       # attributes by name
      d = gsk.generalized_euclidean(net, 0, sparse = True)      # norm, attribute by position
      lp = gsk.laplacian_pinv(net)
      d = gsk.generalized_euclidean(net, x, y, lpinv = lp)      # vectors, cached pseudoinverse
      ds = gsk.generalized_euclidean(net, "age", component = None)
   """
   if component is None:
      pick = lambda x, c: x[c] if isinstance(x, list) else x
      return [generalized_euclidean(net, pick(a, c), pick(b, c), c, pick(lpinv, c), sparse, weight, kind, q, alpha) for c in range(len(net.components))]
   d = _vector(net, component, a)
   if b is not None:
      d = d - _vector(net, component, b)
   if lpinv is None:
      lpinv = (laplacian_solver if sparse else laplacian_pinv)(net, component, weight, kind, q, alpha)
   if callable(lpinv):
      x = lpinv(d)
      d = d.to(x.dtype)
   else:
      d = d.to(lpinv.dtype)
      x = lpinv @ d
   v = (d.conj() @ x).real
   if not net.directed or kind != "out":
      v = v.clamp(min = 0)
   return float(v.sqrt())


def pairwise_generalized_euclidean(net, component = 0, lpinv = None, sparse = None, weight = None, kind = "out", q = 0.25, alpha = 0.15, output_device = None, output_dtype = torch.float64):
   """
   Generalized Euclidean distances between all pairs of node attributes.

   Works like :func:`generalized_euclidean`, applied to every pair of columns of ``node_attr``.
   Instead of one product with :math:`L^+` per pair, it computes :math:`Y = L^+ X` once for the
   ``k`` attributes :math:`X` and then all squared distances from the ``k`` by ``k`` matrix
   :math:`G = X^H Y`, as :math:`G_{ii} + G_{jj} - G_{ij} - G_{ji}`. The attributes are first
   centered across attributes for each node, which leaves the distances unchanged and reduces
   rounding errors. The cost is ``k`` products with :math:`L^+` (or ``k`` solves, run together)
   plus an ``n`` by ``k`` by ``k`` product.

   :math:`L^+` is used as given in ``lpinv`` (dense tensor or solver). If ``lpinv`` is ``None``, a
   solver is built when ``sparse`` is ``True`` or ``None`` (the default, usually faster, since
   only ``k`` solves are needed), and the dense pseudoinverse when ``sparse`` is ``False``.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
      With ``None``, ``lpinv`` is given as a list with one entry per component.
   :param lpinv: Cached pseudoinverse or solver, or ``None``.
   :param sparse: Laplacian solver (``True`` or ``None``) or dense pseudoinverse (``False``) when
      ``lpinv`` is ``None``.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param kind: Laplacian of directed networks: ``"out"``, ``"magnetic"`` or ``"chung"``.
   :param q: Charge of the magnetic Laplacian.
   :param alpha: Teleportation probability of the Chung Laplacian.
   :param output_device: Device of the result, ``None`` for the device of the component. With
      ``"cpu"`` the computation still runs on the device of the component (e.g. the GPU), one
      block of rows at a time, and only the result is kept on the CPU: useful when the ``k`` by
      ``k`` result does not fit in GPU memory.
   :param output_dtype: Dtype of the result. ``torch.float32`` halves its memory (a ``k`` by ``k``
      float64 matrix needs :math:`8 k^2` bytes, 3.2 GB for 20,000 attributes); the computation
      itself stays in double precision.
   :return: ``k`` by ``k`` tensor, rows and columns ordered as ``net.node_attributes``, or
      a list of them if ``component`` is ``None``. NaN where the quadratic form is negative
      (possible only for the ``"out"`` Laplacian) or for attributes with missing values.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      D = gsk.pairwise_generalized_euclidean(net, weight = "weight")
      D = gsk.pairwise_generalized_euclidean(net, weight = "weight", output_device = "cpu")
   """
   if component is None:
      pick = lambda x, c: x[c] if isinstance(x, list) else x
      return [pairwise_generalized_euclidean(net, c, pick(lpinv, c), sparse, weight, kind, q, alpha, output_device, output_dtype) for c in range(len(net.components))]
   attrs = _Attributes(net.components[component].node_attr, "rows")
   if lpinv is None:
      lpinv = (laplacian_pinv if sparse is False else laplacian_solver)(net, component, weight, kind, q, alpha)
   hermitian = not net.directed or kind != "out"
   if callable(lpinv):
      if not hermitian and not hasattr(lpinv, "transpose"):
         raise ValueError("pairwise_generalized_euclidean: the \"out\" Laplacian needs a solver from laplacian_solver, which can also solve with the transpose")
      apply = (lambda B: 2 * lpinv(B)) if hermitian else (lambda B: lpinv(B) + lpinv.transpose(B))
   else:
      apply = (lambda B: 2 * (lpinv @ B.to(lpinv.dtype))) if hermitian else (lambda B: lpinv @ B.to(lpinv.dtype) + lpinv.mT @ B.to(lpinv.dtype))

   def finish(G, d_rows, d_cols):   # D^2_ij = d_i + d_j - x_i^T (L^+ + L^+^T) x_j, with d_i = x_i^T L^+ x_i
      D2 = G.neg_().add_(d_rows[:, None] / 2).add_(d_cols[None, :] / 2)
      if hermitian:
         D2.clamp_(min = 0)
      return D2.sqrt_()

   D = _pairwise(attrs, lambda B: apply(B).to(torch.complex128 if net.directed and kind == "magnetic" else torch.float64), finish, attrs.device if output_device is None else output_device, output_dtype)[0]
   D.fill_diagonal_(0)
   return D


def network_correlation(net, a, b, component = 0, W = None, k = 1, lpinv = None, weight = None, kind = "out", q = 0.25, alpha = 0.15):
   """
   Network Pearson correlation (Coscia, 2021; Coscia and Devriendt, 2024,
   https://arxiv.org/abs/2402.09489):

   .. math::

      \\rho_{a, b, G} = \\frac{\\hat{a}^T W \\hat{b}}{\\sqrt{(\\hat{a}^T W \\hat{a}) (\\hat{b}^T W \\hat{b})}},

   with :math:`\\hat{a} = a - \\bar{a}` the centered vectors. The numerator is
   :math:`\\sum W \\times (\\hat{a} \\otimes \\hat{b})`: every pair of entries is compared, weighted by
   the proximity of the two nodes. With :math:`W = I` this is the Pearson correlation.

   The default weights are :math:`W = e^{-k \\Omega}` (entrywise), with :math:`\\Omega` the
   effective resistance matrix (see :func:`~graphoskop.effective_resistance`). The
   correlation is real and in :math:`[-1, 1]` when :math:`W` is positive definite on vectors with
   zero sum, which holds for any :math:`k > 0` when :math:`\\Omega` is a metric of negative type:
   true for undirected, ``"chung"`` and ``"magnetic"`` Laplacians, not guaranteed for ``"out"``.
   ``k`` sets how fast proximity decays: as :math:`k \\to 0` all pairs weigh the same, as
   :math:`k \\to \\infty` the result tends to the Pearson correlation. Choose it relative to the
   scale of :math:`\\Omega`: the ``"chung"`` Laplacian is scaled by the stationary distribution,
   which sums to 1, so its resistances are about ``n`` times larger and need a ``k`` about ``n``
   times smaller.

   :math:`W` can be passed precomputed (an ``n`` by ``n`` tensor), to reuse it across calls; it is
   then used as given, and ``k``, ``lpinv`` and the Laplacian arguments are ignored. Otherwise it is
   computed from ``lpinv`` (cached dense pseudoinverse) or from scratch, in place, needing one
   ``n`` by ``n`` matrix besides ``lpinv``.

   :param net: A :class:`~graphoskop.Network`.
   :param a: Node attribute (position in ``net.node_attributes`` or name), or vector of length
      ``n`` (:class:`numpy.ndarray` or :class:`torch.Tensor`).
   :param b: Second node vector, same modalities as ``a``.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
      With ``None``, vectors, ``W`` and ``lpinv`` are given as lists with one entry per component.
   :param W: Precomputed weight matrix, or ``None``.
   :param k: Scale of the effective resistance in :math:`W = e^{-k \\Omega}`.
   :param lpinv: Cached dense pseudoinverse of the Laplacian, or ``None``.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param kind: Laplacian of directed networks: ``"out"``, ``"magnetic"`` or ``"chung"``.
   :param q: Charge of the magnetic Laplacian.
   :param alpha: Teleportation probability of the Chung Laplacian.
   :return: Correlation, or list of correlations if ``component`` is ``None``. NaN for constant
      vectors or missing values.

   Example:

   .. code-block:: python

      import torch
      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      r = gsk.network_correlation(net, "age", "height")
      W = torch.exp(-2 * gsk.effective_resistance(net))         # cached weights, k = 2
      r = gsk.network_correlation(net, "age", "height", W = W)
   """
   if component is None:
      pick = lambda x, c: x[c] if isinstance(x, list) else x
      return [network_correlation(net, pick(a, c), pick(b, c), c, pick(W, c), k, pick(lpinv, c), weight, kind, q, alpha) for c in range(len(net.components))]
   x = torch.stack([_vector(net, component, a), _vector(net, component, b)], 1).double()
   x = x - x.mean(0)
   if W is None:
      W = effective_resistance(net, component, lpinv, weight, kind, q, alpha).mul_(-k).exp_()
   G = x.T @ (W @ x.to(W.dtype)).double()
   return float(G[0, 1] / (G[0, 0] * G[1, 1]).sqrt())


def pairwise_network_correlation(net, component = 0, W = None, k = 1, lpinv = None, weight = None, kind = "out", q = 0.25, alpha = 0.15, output_device = None, output_dtype = torch.float64):
   """
   Network Pearson correlations between all pairs of node attributes.

   Works like :func:`network_correlation`, applied to every pair of columns of ``node_attr``. With
   :math:`\\hat{X}` the centered attributes, it computes :math:`G = \\hat{X}^T W \\hat{X}` with a single
   product of :math:`W` with the ``n`` by ``k`` matrix :math:`\\hat{X}`, and then all correlations as
   :math:`G_{ij} / \\sqrt{G_{ii} G_{jj}}`. :math:`W` is computed once, or reused if given.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
      With ``None``, ``W`` and ``lpinv`` are given as lists with one entry per component.
   :param W: Precomputed weight matrix, or ``None`` for :math:`e^{-k \\Omega}`.
   :param k: Scale of the effective resistance in :math:`W = e^{-k \\Omega}`.
   :param lpinv: Cached dense pseudoinverse of the Laplacian, or ``None``.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param kind: Laplacian of directed networks: ``"out"``, ``"magnetic"`` or ``"chung"``.
   :param q: Charge of the magnetic Laplacian.
   :param alpha: Teleportation probability of the Chung Laplacian.
   :param output_device: Device of the result, ``None`` for the device of the component. With
      ``"cpu"`` the computation still runs on the device of the component (e.g. the GPU), one
      block of rows at a time, and only the result is kept on the CPU: useful when the ``k`` by
      ``k`` result does not fit in GPU memory.
   :param output_dtype: Dtype of the result. ``torch.float32`` halves its memory (a ``k`` by ``k``
      float64 matrix needs :math:`8 k^2` bytes, 3.2 GB for 20,000 attributes); the computation
      itself stays in double precision.
   :return: ``k`` by ``k`` tensor, rows and columns ordered as ``net.node_attributes``, or
      a list of them if ``component`` is ``None``. NaN for constant attributes or missing values.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      C = gsk.pairwise_network_correlation(net, weight = "weight")
      C = gsk.pairwise_network_correlation(net, weight = "weight", output_device = "cpu")
   """
   if component is None:
      pick = lambda x, c: x[c] if isinstance(x, list) else x
      return [pairwise_network_correlation(net, c, pick(W, c), k, pick(lpinv, c), weight, kind, q, alpha, output_device, output_dtype) for c in range(len(net.components))]
   attrs = _Attributes(net.components[component].node_attr, "columns")
   if W is None:
      W = effective_resistance(net, component, lpinv, weight, kind, q, alpha).mul_(-k).exp_()
   finish = lambda G, d_rows, d_cols: G.div_(d_rows.sqrt()[:, None]).div_(d_cols.sqrt()[None, :])
   C, d = _pairwise(attrs, lambda B: (W @ B.to(W.dtype)).double(), finish, attrs.device if output_device is None else output_device, output_dtype)
   C.diagonal()[(d.isfinite() & (d > 0)).to(C.device)] = 1
   return C


def _distribution(net, component, a, name):
   """
   Node vector normalized to a probability distribution.

   :param net: Network.
   :param component: Component index.
   :param a: Node attribute or vector.
   :param name: Name of the calling function, for error messages.
   :return: Float64 distribution.
   :raises ValueError: If ``a`` has negative entries or sums to zero.
   """
   p = _vector(net, component, a).double()
   if (p < 0).any():
      raise ValueError(f"{name}: the vector has negative entries, so it is not a probability mass")
   if p.sum() == 0:
      raise ValueError(f"{name}: the vector sums to zero, so it is not a probability mass")
   return p / p.sum()


def network_variance(net, a, component = 0, R = None, lpinv = None, weight = None, kind = "out", q = 0.25, alpha = 0.15):
   """
   Network variance of a node vector interpreted as a probability distribution (Devriendt,
   Martin-Gutierrez and Lambiotte, 2022, https://arxiv.org/abs/2008.09155):

   .. math::

      \\mathrm{var}(p) = \\frac{1}{2} \\sum_{i, j} p_i p_j d^2(i, j) = \\frac{1}{2} p^T \\Omega p,

   with :math:`p = a / \\sum_i a_i`, :math:`\\Omega` the effective resistance matrix (see
   :func:`~graphoskop.effective_resistance`) and :math:`d = \\sqrt{\\Omega}` the square
   root effective resistance distance, as in the paper. On an undirected network, Kemeny's constant
   is :math:`\\kappa = 2 m \\, \\mathrm{var}(\\pi)`, with :math:`\\pi` the stationary distribution of
   the random walk and :math:`m` the total edge weight, and the Kirchhoff index is
   :math:`n^2 \\, \\mathrm{var}(u / n)`, with :math:`u / n` the uniform distribution.

   ``a`` must be a probability mass up to normalization: a :class:`ValueError` is raised if it has
   negative entries or sums to zero.

   :math:`\\Omega` can be passed precomputed as ``R``, to reuse it across calls; it is then used as
   given and left unchanged, and ``lpinv`` and the Laplacian arguments are ignored. Otherwise it is
   computed from ``lpinv`` (cached dense pseudoinverse) or from scratch.

   :param net: A :class:`~graphoskop.Network`.
   :param a: Node attribute (position in ``net.node_attributes`` or name), or vector of length
      ``n`` (:class:`numpy.ndarray` or :class:`torch.Tensor`), with non negative entries.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
      With ``None``, vectors, ``R`` and ``lpinv`` are given as lists with one entry per component.
   :param R: Precomputed effective resistance matrix, or ``None``.
   :param lpinv: Cached dense pseudoinverse of the Laplacian, or ``None``.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param kind: Laplacian of directed networks: ``"out"``, ``"magnetic"`` or ``"chung"``.
   :param q: Charge of the magnetic Laplacian.
   :param alpha: Teleportation probability of the Chung Laplacian.
   :return: Variance, or list of variances if ``component`` is ``None``. NaN for missing values.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      v = gsk.network_variance(net, "population")
      R = gsk.effective_resistance(net)
      v = gsk.network_variance(net, "income", R = R)       # cached resistances
   """
   if component is None:
      pick = lambda x, c: x[c] if isinstance(x, list) else x
      return [network_variance(net, pick(a, c), c, pick(R, c), pick(lpinv, c), weight, kind, q, alpha) for c in range(len(net.components))]
   p = _distribution(net, component, a, "network_variance")
   if R is None:
      R = effective_resistance(net, component, lpinv, weight, kind, q, alpha)
   return float(p @ (R @ p.to(R.dtype)).double() / 2)


def network_entropy(net, a, component = 0, W = None, k = 1, lpinv = None, weight = None, kind = "out", q = 0.25, alpha = 0.15):
   """
   Network entropy of a node vector interpreted as a probability distribution:

   .. math::

      H(p) = -\\sum_i p_i \\log (W p)_i,

   with :math:`p = a / \\sum_i a_i` and node similarities :math:`W = e^{-k \\Omega}` (entrywise),
   :math:`\\Omega` being the effective resistance matrix (see
   :func:`~graphoskop.effective_resistance`). :math:`(W p)_i` is the probability mass
   near node :math:`i`, so mass spread over distant nodes gives a high entropy and mass concentrated
   on nearby nodes a low one. With :math:`W = I` this is the Shannon entropy. It is the order 1
   similarity-sensitive diversity of Leinster and Cobbold (2012). Nodes with :math:`p_i = 0` add
   nothing to the sum.

   ``a`` must be a probability mass up to normalization: a :class:`ValueError` is raised if it has
   negative entries or sums to zero.

   :math:`W` can be passed precomputed (an ``n`` by ``n`` tensor), to reuse it across calls (for
   instance the one of :func:`network_correlation`); it is then used as given, and ``k``,
   ``lpinv`` and the Laplacian arguments are ignored. Otherwise it is computed from ``lpinv``
   (cached dense pseudoinverse) or from scratch. As for :func:`network_correlation`, choose ``k``
   relative to the scale of :math:`\\Omega`.

   :param net: A :class:`~graphoskop.Network`.
   :param a: Node attribute (position in ``net.node_attributes`` or name), or vector of length
      ``n`` (:class:`numpy.ndarray` or :class:`torch.Tensor`), with non negative entries.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
      With ``None``, vectors, ``W`` and ``lpinv`` are given as lists with one entry per component.
   :param W: Precomputed similarity matrix, or ``None`` for :math:`e^{-k \\Omega}`.
   :param k: Scale of the effective resistance in :math:`W = e^{-k \\Omega}`.
   :param lpinv: Cached dense pseudoinverse of the Laplacian, or ``None``.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param kind: Laplacian of directed networks: ``"out"``, ``"magnetic"`` or ``"chung"``.
   :param q: Charge of the magnetic Laplacian.
   :param alpha: Teleportation probability of the Chung Laplacian.
   :return: Entropy (in nats), or list of entropies if ``component`` is ``None``. NaN for missing
      values.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      h = gsk.network_entropy(net, "population")
   """
   if component is None:
      pick = lambda x, c: x[c] if isinstance(x, list) else x
      return [network_entropy(net, pick(a, c), c, pick(W, c), k, pick(lpinv, c), weight, kind, q, alpha) for c in range(len(net.components))]
   p = _distribution(net, component, a, "network_entropy")
   if W is None:
      W = effective_resistance(net, component, lpinv, weight, kind, q, alpha).mul_(-k).exp_()
   s = (W @ p.to(W.dtype)).double()
   m = p != 0
   return float(-(p[m] * s[m].log()).sum()) + 0.0   # + 0.0 turns -0.0 into 0.0
