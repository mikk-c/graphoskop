"""
Laplacians of network components, their pseudoinverses, and Laplacian solvers.

All functions take a :class:`~graphoskop.Network` and a component index (default 0,
the largest component; ``None`` applies the function to every component and returns a list).

For undirected networks the Laplacian is :math:`D - A`. For directed networks ``kind`` selects:

* ``"out"``: out-degree Laplacian :math:`D_{out} - A`.
* ``"magnetic"``: magnetic Laplacian :math:`D_s - A_s \\odot e^{i \\Theta}`, with
  :math:`A_s = (A + A^T) / 2`, :math:`D_s` its degree matrix and
  :math:`\\Theta = 2 \\pi q (A - A^T)`. Hermitian and complex valued.
* ``"chung"``: Chung's combinatorial Laplacian :math:`\\Phi - (\\Phi P + P^T \\Phi) / 2`, with
  :math:`P` the random walk transition matrix with teleportation probability ``alpha`` (dangling
  nodes teleport uniformly) and :math:`\\Phi` the diagonal matrix of its stationary distribution.
  Symmetric and real valued; dense if ``alpha > 0`` or there are dangling nodes. The stationary
  distribution is available on its own as :func:`~graphoskop.stationary_distribution`.

:math:`A` is the (weighted) adjacency matrix; parallel edges are summed and self loops do not
affect the Laplacian. Rows and columns are ordered by internal node index.

In multilayer networks :math:`A` is the supra-adjacency matrix: rows and columns are the copies
``(entity, layer)`` of the component, and couplings and other interlayer edges are entries like
intra-layer edges, so :math:`L` is the supra-Laplacian (as in Coscia, 2022, "Generalized Euclidean
measure to estimate distances on multilayer networks"). Coupling weights must be positive, so that
a component is connected through them; infinitely strong couplings give the Laplacian of
:func:`~graphoskop.aggregate_network`. Dense matrices have one row per copy, so they take about
:math:`L^2` times the memory of a single layer with :math:`L` layers: on large networks, prefer
:func:`~graphoskop.laplacian_solver`. The random walks of :func:`stationary_distribution` and of
the Chung Laplacian move along couplings like along any edge, with probability proportional to
their weight.
"""

import math
import warnings

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import breadth_first_order, connected_components

from .network import _quiet_sparse


def _weights(net, component, weight):
   """
   Edge weights of a component.

   :param net: Network.
   :param component: Component index.
   :param weight: Edge attribute name used as weight, or ``None`` for unit weights.
   :return: Weights, in the dtype of the edge attributes.
   """
   data = net.components[component]
   if weight is None:
      return torch.ones(data.edge_index.shape[1], dtype = data.edge_attr.dtype, device = data.edge_index.device)
   return net.get_edge_attribute(weight, component)


def _walk(ei, w, n, alpha, tol, max_iter, v = None):
   """
   Random walk with teleportation and its stationary distribution, by power iteration.

   :param ei: Edge index.
   :param w: Edge weights.
   :param n: Number of nodes.
   :param alpha: Teleportation probability.
   :param tol: Convergence threshold (L1).
   :param max_iter: Maximum number of iterations.
   :param v: Teleportation distribution (float64, summing to 1; also used from dangling nodes), or
      ``None`` for uniform.
   :return: Stationary distribution, transition probability of each edge (without
      teleportation) and mask of dangling nodes, in float64.
   """
   s, t = ei
   d = torch.zeros(n, dtype = torch.float64, device = w.device).index_add_(0, s, w.double())
   p = w.double() / d[s]
   dangling = d == 0
   phi = torch.full((n,), 1 / n, dtype = torch.float64, device = w.device)
   for _ in range(max_iter):
      jump = alpha + (1 - alpha) * phi[dangling].sum()
      new = (1 - alpha) * torch.zeros_like(phi).index_add_(0, t, phi[s] * p) + (jump / n if v is None else jump * v)
      done = (new - phi).abs().sum() < tol
      phi = new
      if done:
         return phi, p, dangling
   warnings.warn("stationary distribution: power iteration did not converge")
   return phi, p, dangling


def stationary_distribution(net, component = 0, weight = None, alpha = 0.15, tol = 1e-12, max_iter = 10000):
   """
   Stationary distribution of the random walk with teleportation on network components.

   At each step the walker follows an out edge with probability proportional to its weight, or,
   with probability ``alpha``, jumps to a uniformly random node. Dangling nodes (without out edges)
   always jump. With ``alpha = 0`` this is the plain random walk, whose stationary distribution is
   unique only on strongly connected aperiodic components. On undirected networks with
   ``alpha = 0`` it is proportional to the (weighted) degree.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param alpha: Teleportation probability.
   :param tol: Convergence threshold of the power iteration (L1 change between iterations).
   :param max_iter: Maximum number of power iterations.
   :return: Float64 vector of length ``n`` summing to 1, on the device of the component, or a list
      of them if ``component`` is ``None``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", directed = True)
      phi = gsk.stationary_distribution(net)                 # PageRank, damping 0.85
      phi = gsk.stationary_distribution(net, alpha = 0)      # plain random walk
   """
   if component is None:
      return [stationary_distribution(net, c, weight, alpha, tol, max_iter) for c in range(len(net.components))]
   data = net.components[component]
   return _walk(data.edge_index, _weights(net, component, weight), data.num_nodes, alpha, tol, max_iter)[0]


def _gauge(u, v, th, n):
   """
   Null space vector of a magnetic Laplacian, if its phases are balanced.

   Phases are propagated along a breadth first spanning tree (by pointer jumping) and then checked
   on every edge: the Laplacian is singular exactly when they agree.

   :param u: Row of each non-zero entry.
   :param v: Column of each non-zero entry.
   :param th: Phase :math:`\\Theta_{uv}` of each non-zero entry.
   :param n: Number of nodes.
   :return: Vector :math:`e^{i \\theta}` spanning the null space, or ``None`` if non-singular.
   """
   u, v, th = (x.cpu().numpy() for x in (u, v, th))
   par = breadth_first_order(sp.csr_matrix((np.ones(len(u)), (u, v)), shape = (n, n)), 0, directed = False, return_predecessors = True)[1]
   par[0] = 0
   ph = -np.asarray(sp.csr_matrix((th, (u, v)), shape = (n, n))[par, np.arange(n)]).ravel()
   while (par != 0).any():
      ph, par = ph + ph[par], par[par]
   if np.abs(np.exp(1j * (th - ph[u] + ph[v])) - 1).max(initial = 0) > 1e-6:
      return None
   return np.exp(1j * ph)


def _operator(net, component, weight, kind, q, alpha, dtype):
   """
   Laplacian of a component as a sparse part plus a low rank part, with its null space.

   The Laplacian is :math:`L = S - (g 1^T + 1 g^T) / 2`.

   :param net: Network.
   :param component: Component index.
   :param weight: Edge attribute name used as weight, or ``None``.
   :param kind: Laplacian kind of directed networks.
   :param q: Magnetic charge.
   :param alpha: Chung teleportation probability.
   :param dtype: Real floating point type (complex counterpart for the magnetic Laplacian).
   :return: ``(S, g, null, hermitian)``: coalesced sparse COO tensor, low rank vector or ``None``,
      null space vector (entries of modulus 1) or ``None`` if unknown or trivial, and whether
      :math:`L` is Hermitian.
   """
   data = net.components[component]
   n, ei = data.num_nodes, data.edge_index
   dev = ei.device
   w = _weights(net, component, weight).to(dtype)
   ones = torch.ones(n, dtype = dtype, device = dev)

   def coo(i, x):
      with _quiet_sparse():
         return torch.sparse_coo_tensor(i, x, (n, n)).coalesce()

   if not net.directed or kind == "out":
      S = coo(torch.cat([ei, ei[0].repeat(2, 1)], 1), torch.cat([-w, w]))
      return S, None, None if net.directed else ones, not net.directed
   if kind == "chung":
      s, t = ei
      phi, p, dangling = _walk(ei, w, n, alpha, 1e-12, 10000)
      x = ((1 - alpha) * phi[s] * p / 2).to(dtype)
      r = torch.arange(n, device = dev)
      S = coo(torch.cat([ei, ei.flip(0), torch.stack([r, r])], 1), torch.cat([-x, -x, phi.to(dtype)]))
      return S, (phi * (alpha + (1 - alpha) * dangling) / n).to(dtype), ones, True
   if kind == "magnetic":
      cw = w.to(torch.complex64 if dtype == torch.float32 else torch.complex128)
      z = coo(torch.cat([ei, ei.flip(0)], 1), torch.cat([cw, 1j * cw]))
      (u, v), a, b = z.indices(), z.values().real, z.values().imag
      th = 2 * math.pi * q * (a - b)
      S = coo(torch.cat([z.indices(), torch.stack([u, u])], 1), torch.cat([-(a + b) / 2 * torch.exp(1j * th), ((a + b) / 2).to(cw.dtype)]))
      g = _gauge(u, v, th.double(), n)
      return S, None, None if g is None else torch.tensor(g, dtype = cw.dtype, device = dev), True
   raise ValueError(f"unknown Laplacian kind: {kind}")


def _dense(S, g):
   """
   Dense Laplacian from its sparse and low rank parts, with a single ``n`` by ``n`` allocation.

   :param S: Sparse part.
   :param g: Low rank vector, or ``None``.
   :return: Dense Laplacian.
   """
   L = S.to_dense()
   if g is not None:
      ones = torch.ones_like(g)
      L.addr_(g, ones, alpha = -0.5).addr_(ones, g, alpha = -0.5)
   return L


def _pinv_null(L, g):
   """
   Pseudoinverse of a Hermitian positive semidefinite matrix whose null space is spanned by ``g``
   (with :math:`|g_i| = 1`), or which is positive definite if ``g`` is ``None``.

   Uses :math:`L^+ = (L + g g^H / n)^{-1} - g g^H / n`, computed in place on ``L`` with a
   Cholesky factorization. The factorization runs on the transposed view, which is column major
   as LAPACK expects, so no copy of ``L`` is made. The view holds :math:`\\overline{L}`, whose
   inverse is the conjugate of the inverse, and the transpose of that conjugate is again the
   inverse of the Hermitian matrix :math:`L`.

   :param L: Matrix, overwritten with the result.
   :param g: Null space vector, or ``None``.
   :return: ``L``, now holding the pseudoinverse.
   """
   n = L.shape[0]
   if g is not None:
      L.addr_(g, g.conj(), alpha = 1 / n)
   M = L.mT
   torch.linalg.cholesky(M, out = M)
   torch.cholesky_inverse(M, out = M)
   if g is not None:
      L.addr_(g, g.conj(), alpha = -1 / n)
   return L


def laplacian(net, component = 0, weight = None, kind = "out", q = 0.25, alpha = 0.15, sparse = False):
   """
   Laplacian of network components (see the module description for the available kinds).

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param kind: Laplacian of directed networks: ``"out"``, ``"magnetic"`` or ``"chung"``.
   :param q: Charge of the magnetic Laplacian.
   :param alpha: Teleportation probability of the Chung Laplacian.
   :param sparse: Return a sparse CSR tensor instead of a dense one. Not possible for a Chung
      Laplacian with teleportation or dangling nodes, which is dense.
   :return: ``n`` by ``n`` tensor on the device of the component, or a list of them if
      ``component`` is ``None``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", directed = True)
      L = gsk.laplacian(net)                                   # largest component, out-degree
      L = gsk.laplacian(net, kind = "magnetic", sparse = True)
   """
   if component is None:
      return [laplacian(net, c, weight, kind, q, alpha, sparse) for c in range(len(net.components))]
   S, g, _, _ = _operator(net, component, weight, kind, q, alpha, net.components[component].edge_attr.dtype)
   if not sparse:
      return _dense(S, g)
   if g is not None and bool(g.any()):
      raise ValueError("this Chung Laplacian is dense: use alpha = 0 on a component without dangling nodes, or sparse = False")
   with _quiet_sparse():
      return S.to_sparse_csr()


def laplacian_pinv(net, component = 0, weight = None, kind = "out", q = 0.25, alpha = 0.15):
   """
   Moore-Penrose pseudoinverse of the Laplacian of network components.

   Undirected, ``"magnetic"`` and ``"chung"`` Laplacians are Hermitian positive semidefinite with a
   known null space (constant vector, gauge vector of balanced magnetic phases, or none), so the
   pseudoinverse is computed exactly and in place with a Cholesky factorization: the only
   ``n`` by ``n`` matrix allocated is the result. ``"out"`` has no known null space and uses
   :func:`torch.linalg.pinv` (singular value decomposition, several ``n`` by ``n`` matrices, small
   singular values below its default threshold are treated as zero). For components too large for
   a dense ``n`` by ``n`` matrix, use :func:`laplacian_solver`.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param kind: Laplacian of directed networks: ``"out"``, ``"magnetic"`` or ``"chung"``.
   :param q: Charge of the magnetic Laplacian.
   :param alpha: Teleportation probability of the Chung Laplacian; use ``alpha > 0`` unless the
      component is strongly connected and aperiodic.
   :return: Dense ``n`` by ``n`` tensor on the device of the component, or a list of them if
      ``component`` is ``None``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      lp = gsk.laplacian_pinv(net)                            # largest component, unweighted
      lps = gsk.laplacian_pinv(net, None, weight = "weight")  # all components, weighted

      net = gsk.load_network("edges.tsv", directed = True)
      lp = gsk.laplacian_pinv(net, kind = "magnetic", q = 0.1)
      lp = gsk.laplacian_pinv(net, kind = "chung", alpha = 0.1)
   """
   if component is None:
      return [laplacian_pinv(net, c, weight, kind, q, alpha) for c in range(len(net.components))]
   S, g, null, hermitian = _operator(net, component, weight, kind, q, alpha, net.components[component].edge_attr.dtype)
   L = _dense(S, g)
   del S
   return _pinv_null(L, null) if hermitian else torch.linalg.pinv(L)


def effective_resistance(net, component = 0, lpinv = None, weight = None, kind = "out", q = 0.25, alpha = 0.15):
   """
   Effective resistance between all pairs of nodes,
   :math:`R_{uv} = (e_u - e_v)^H L^+ (e_u - e_v) = L^+_{uu} + L^+_{vv} - L^+_{uv} - L^+_{vu}`,
   with :math:`L^+` the pseudoinverse of the Laplacian (see
   :func:`~graphoskop.laplacian`). For complex Laplacians the real part is taken.

   :math:`L^+` is used as given in ``lpinv`` (a dense tensor from
   :func:`~graphoskop.laplacian_pinv`, left unchanged), or computed if ``None``. A
   computed real symmetric :math:`L^+` is turned into :math:`R` in place, so no further ``n`` by
   ``n`` matrix is allocated.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
      With ``None``, ``lpinv`` is given as a list with one entry per component.
   :param lpinv: Cached dense pseudoinverse, or ``None``.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param kind: Laplacian of directed networks: ``"out"``, ``"magnetic"`` or ``"chung"``.
   :param q: Charge of the magnetic Laplacian.
   :param alpha: Teleportation probability of the Chung Laplacian.
   :return: Real ``n`` by ``n`` tensor, rows and columns ordered by internal node index, or a list
      of them if ``component`` is ``None``.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      R = gsk.effective_resistance(net, weight = "weight")
      c, i = net.node_map["alice"]
      c, j = net.node_map["bob"]
      r = R[i, j]                                     # if both are in component 0
   """
   if component is None:
      pick = lambda x, c: x[c] if isinstance(x, list) else x
      return [effective_resistance(net, c, pick(lpinv, c), weight, kind, q, alpha) for c in range(len(net.components))]
   own = lpinv is None
   if own:
      lpinv = laplacian_pinv(net, component, weight, kind, q, alpha)
   d = lpinv.diagonal().real.clone()
   if lpinv.is_complex():
      R = lpinv.real.mul(-2)
   elif not net.directed or kind != "out":
      R = lpinv.mul_(-2) if own else lpinv.mul(-2)
   else:
      R = torch.add(lpinv, lpinv.mT).neg_()
   ones = torch.ones_like(d)
   R.addr_(d, ones).addr_(ones, d)
   R.fill_diagonal_(0)
   return R


def hitting_time(net, component = 0, lpinv = None, weight = None, alpha = 0):
   """
   Hitting times between all pairs of nodes: :math:`H_{ij}` is the expected number of steps a
   random walk starting at :math:`i` takes to reach :math:`j` for the first time
   (:math:`H_{ii} = 0`). The walk is the one of :func:`stationary_distribution`: it follows an out
   edge with probability proportional to its weight (self loops included: the walk may stay), or,
   with probability ``alpha``, jumps to a uniformly random node; dangling nodes always jump. Hitting
   times are not symmetric, even in undirected networks; :math:`H_{ij} + H_{ji}` is the commute
   time, :math:`\\mathrm{vol} \\cdot R_{ij}` in undirected networks (:func:`effective_resistance`).

   * Undirected networks with ``alpha = 0``: from the Laplacian pseudoinverse,
     :math:`H_{ij} = \\sum_k d_k (L^+_{ik} - L^+_{ij} - L^+_{jk} + L^+_{jj})`, with :math:`d` the
     (weighted) degrees. :math:`L^+` is used as given in ``lpinv`` (from
     :func:`laplacian_pinv` with the same ``weight``) or computed.
   * Otherwise (directed networks, or ``alpha > 0``), from the fundamental matrix of the walk
     (Kemeny and Snell): :math:`H_{ij} = (Z_{jj} - Z_{ij}) / \\pi_j`, with
     :math:`Z = (I - P + \\mathbf{1} \\pi^T)^{-1}`, :math:`P` the transition matrix and :math:`\\pi`
     its stationary distribution, found exactly (not by power iteration), so periodic walks are
     fine. ``lpinv`` is ignored. With ``alpha = 0`` the walk must be able to reach every node from
     every node (dangling nodes jump anywhere, so this fails when a set of nodes has edges only
     among themselves; restrict to a strongly connected component, see
     :func:`~graphoskop.to_strongly_connected`, or use ``alpha > 0``), otherwise some hitting
     times are infinite.

   Both are dense :math:`O(n^3)` computations in float64 on the device of the component, with a few
   ``n`` by ``n`` matrices. In multilayer networks the walk moves along couplings like along any
   edge.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
      With ``None``, ``lpinv`` is given as a list with one entry per component.
   :param lpinv: Cached dense pseudoinverse (undirected networks, ``alpha = 0``), or ``None``.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param alpha: Teleportation probability.
   :return: Float64 ``n`` by ``n`` tensor, rows (origins) and columns (targets) ordered by internal
      node index, or a list of them if ``component`` is ``None``.
   :raises ValueError: If ``alpha = 0`` and some node cannot reach some other.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      H = gsk.hitting_time(net)
      c, i = net.node_map["alice"]
      c, j = net.node_map["bob"]
      H[i, j], H[j, i]                         # alice to bob, bob to alice (if in component 0)

      net = gsk.load_network("follows.tsv", directed = True)
      H = gsk.hitting_time(net, alpha = 0.15)  # PageRank walk
   """
   if component is None:
      pick = lambda x, c: x[c] if isinstance(x, list) else x
      return [hitting_time(net, c, pick(lpinv, c), weight, alpha) for c in range(len(net.components))]
   data = net.components[component]
   n, (s, t) = data.num_nodes, data.edge_index
   w = _weights(net, component, weight).double()
   d = torch.zeros(n, dtype = torch.float64, device = w.device).index_add_(0, s, w)
   if not net.directed and alpha == 0:
      if lpinv is None:   # in float64, for accuracy
         S, g, null, _ = _operator(net, component, weight, "out", 0.25, 0.15, torch.float64)
         lpinv = _pinv_null(_dense(S, g), null)
      Lp = lpinv.double()
      u = Lp @ d
      vol = d.sum()
      H = (vol * Lp).neg_().add_(u[:, None]).sub_(u[None, :]).add_(vol * Lp.diagonal()[None, :])
      return H.fill_diagonal_(0)
   P = torch.zeros((n, n), dtype = torch.float64, device = w.device)
   P.index_put_((s, t), (1 - alpha) * w / d[s], accumulate = True)
   P += ((alpha + (1 - alpha) * (d == 0)) / n)[:, None]   # teleportation, dangling nodes
   if alpha == 0:
      if connected_components(sp.csr_matrix((P > 0).cpu().numpy()), directed = True, connection = "strong")[0] > 1:
         raise ValueError("some nodes cannot reach some others, so some hitting times are infinite: use alpha > 0, or restrict to a strongly connected component (to_strongly_connected)")
   I = torch.eye(n, dtype = torch.float64, device = w.device)
   pi = torch.linalg.solve((I - P + 1).T, torch.ones(n, dtype = torch.float64, device = w.device))   # pi^T (I - P + 1 1^T) = 1^T
   Z = torch.linalg.inv(I - P + pi[None, :])
   del P, I
   return (Z.diagonal()[None, :] - Z).div_(pi[None, :]).fill_diagonal_(0)


def _vector(net, component, x):
   """
   Node vector of a component, as a tensor on the component's device.

   :param net: Network.
   :param component: Component index.
   :param x: Node attribute (position in ``net.node_attributes`` or name), or a vector
      (:class:`numpy.ndarray` or :class:`torch.Tensor`).
   :return: Vector.
   :raises ValueError: If ``x`` names a qualitative node attribute.
   """
   if isinstance(x, str) and x in net.qualitative_node_attributes:
      raise ValueError(f"node attribute {x!r} is qualitative: only numeric attributes can be used here")
   if isinstance(x, (int, np.integer, str)):
      return net.get_node_attribute(x, component)
   return torch.as_tensor(x, device = net.components[component].edge_index.device)


def _div(a, b):
   """
   Elementwise ``a / b``, with 0 where ``b`` is 0.
   """
   return torch.where(b == 0, 0, a / torch.where(b == 0, 1, b))


def _cg(mv, b, dinv, null, tol, max_iter):
   """
   Jacobi preconditioned conjugate gradient for :math:`L^+ b`, with :math:`L` Hermitian positive
   semidefinite. ``b`` is projected onto the range of :math:`L`, making the system consistent, and
   the solution is projected out of the null space, giving the minimum norm solution.

   :param mv: Product with :math:`L`.
   :param b: Right hand sides, one per column.
   :param dinv: Inverse diagonal of :math:`L`.
   :param null: Null space vector (entries of modulus 1), or ``None``.
   :param tol: Relative residual tolerance.
   :param max_iter: Maximum number of iterations.
   :return: Solutions, one per column.
   """
   proj = (lambda y: y) if null is None else (lambda y: y - null[:, None] * (null.conj() @ y)[None] / len(null))
   r = proj(b)
   stop = tol * r.norm(dim = 0)
   x, z = torch.zeros_like(r), dinv * r
   p, rz = z.clone(), (r.conj() * z).sum(0).real
   for _ in range(max_iter):
      if not (r.norm(dim = 0) > stop).any():   # NaN right hand sides count as done
         return proj(x)
      Ap = mv(p)
      a = _div(rz, (p.conj() * Ap).sum(0).real)
      x += a * p
      r -= a * Ap
      z = dinv * r
      rz, old = (r.conj() * z).sum(0).real, rz
      p = z + _div(rz, old) * p
   warnings.warn("laplacian_solver: conjugate gradient did not converge")
   return proj(x)


def _cgls(mv, mvt, b, tol, max_iter):
   """
   Conjugate gradient on the normal equations (CGLS) for :math:`L^+ b`. Started from zero, the
   iterates stay in the range of :math:`L^T`, so they converge to the minimum norm least squares
   solution.

   :param mv: Product with :math:`L`.
   :param mvt: Product with :math:`L^T`.
   :param b: Right hand sides, one per column.
   :param tol: Relative tolerance on the normal equations residual :math:`L^T r`.
   :param max_iter: Maximum number of iterations.
   :return: Solutions, one per column.
   """
   x, r = torch.zeros_like(b), b.clone()
   s = mvt(r)
   p, gamma = s.clone(), (s * s).sum(0)
   stop = tol * s.norm(dim = 0)
   for _ in range(max_iter):
      if not (s.norm(dim = 0) > stop).any():   # NaN right hand sides count as done
         return x
      Ap = mv(p)
      a = _div(gamma, (Ap * Ap).sum(0))
      x += a * p
      r -= a * Ap
      s = mvt(r)
      gamma, old = (s * s).sum(0), gamma
      p = s + _div(gamma, old) * p
   warnings.warn("laplacian_solver: CGLS did not converge")
   return x


def laplacian_solver(net, component = 0, weight = None, kind = "out", q = 0.25, alpha = 0.15, tol = 1e-8, max_iter = None):
   """
   Solver computing :math:`L^+ b` without forming the pseudoinverse :math:`L^+`.

   The Laplacian is kept sparse (plus a rank two term for the Chung Laplacian), so memory is
   linear in the number of edges and nodes. Hermitian Laplacians (undirected, ``"magnetic"``,
   ``"chung"``) use Jacobi preconditioned conjugate gradient on the range of :math:`L`; the
   ``"out"`` Laplacian uses CGLS, which converges to the minimum norm least squares solution.
   Both give :math:`L^+ b`. Computations run in double precision on the device of the component.

   The returned function ``solve(b)`` accepts:

   * a vector of length ``n`` or an ``n`` by ``k`` matrix (one right hand side per column), as a
     :class:`numpy.ndarray` (the result is a :class:`numpy.ndarray`) or a :class:`torch.Tensor`;
   * a numeric node attribute, by position in ``net.node_attributes`` or by name (the result is
     a :class:`torch.Tensor`). Missing values must be filled beforehand, as they propagate as NaN.

   :param net: A :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components.
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param kind: Laplacian of directed networks: ``"out"``, ``"magnetic"`` or ``"chung"``.
   :param q: Charge of the magnetic Laplacian.
   :param alpha: Teleportation probability of the Chung Laplacian.
   :param tol: Relative residual tolerance.
   :param max_iter: Maximum number of iterations, ``10 * n`` if ``None``.
   :return: The function ``solve``, or a list of them if ``component`` is ``None``. Its attribute
      ``solve.transpose`` computes :math:`(L^+)^T b` in the same way.

   Example:

   .. code-block:: python

      import numpy as np
      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      solve = gsk.laplacian_solver(net, weight = "weight")
      x = solve("age")                                          # node attribute by name
      x = solve(0)                                              # node attribute by position
      x = solve(np.random.rand(net.components[0].num_nodes))
   """
   if component is None:
      return [laplacian_solver(net, c, weight, kind, q, alpha, tol, max_iter) for c in range(len(net.components))]
   data = net.components[component]
   n = data.num_nodes
   max_iter = 10 * n if max_iter is None else max_iter
   S, g, null, hermitian = _operator(net, component, weight, kind, q, alpha, torch.float64)
   with _quiet_sparse():
      A = S.to_sparse_csr()
      At = None if hermitian else S.t().coalesce().to_sparse_csr()
   if g is None:
      mv = lambda x: A @ x
   else:
      mv = lambda x: A @ x - (g[:, None] * x.sum(0) + (g @ x)[None]) / 2
   i = S.indices()
   d = torch.zeros(n, dtype = torch.float64, device = A.device).index_add_(0, i[0][i[0] == i[1]], S.values()[i[0] == i[1]].real)
   if g is not None:
      d -= g
   dinv = torch.where(d > 0, 1 / d, 1).to(S.dtype)[:, None]

   def solver(transpose):
      def solve(b):
         numpy = isinstance(b, np.ndarray)
         b = _vector(net, component, b)
         vector = b.dim() == 1
         b = b.reshape(n, -1).to(S.dtype)
         if hermitian:
            x = _cg(mv, b.conj(), dinv, null, tol, max_iter).conj() if transpose else _cg(mv, b, dinv, null, tol, max_iter)
         elif transpose:   # (L^+)^T b = (L^T)^+ b: CGLS with the roles of L and L^T swapped
            x = _cgls(lambda y: At @ y, mv, b, tol, max_iter)
         else:
            x = _cgls(mv, lambda y: At @ y, b, tol, max_iter)
         x = x[:, 0] if vector else x
         return x.cpu().numpy() if numpy else x
      return solve

   solve = solver(False)
   solve.transpose = solver(True)
   return solve
