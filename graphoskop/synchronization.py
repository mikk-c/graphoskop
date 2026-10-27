"""
Synchronization of coupled oscillators: the Kuramoto model on networks, and its higher-order
versions on simplicial complexes, where oscillators live on edges and triangles too.

Every oscillator has a phase that turns at its own natural frequency and is pulled towards the
phases it is coupled with, with strength ``sigma``. Weakly coupled, the phases drift
independently; strongly coupled, they lock together. The **order parameter**
:math:`R = |\\frac{1}{n} \\sum_j e^{i \\theta_j}|` measures this, from 0 (incoherent) to 1
(synchronized), and its curve against ``sigma`` shows the synchronization transition, continuous
or explosive (abrupt, with hysteresis).

* :func:`kuramoto_model`: phases on the nodes of a network (Kuramoto, 1975; on networks, Arenas,
  Díaz-Guilera, Kurths, Moreno and Zhou, 2008): :math:`\\dot\\theta_i = \\omega_i + \\sigma
  \\sum_j A_{ji} \\sin(\\theta_j - \\theta_i)`, with weights and directions.
* :func:`simplicial_kuramoto_model`: phases on the :math:`k`-simplices of a simplicial complex,
  coupled through the boundary matrices (Millán, Torres and Bianconi, 2020), with the explosive
  variant; :math:`k = 1` puts the oscillators on the edges of a complex or of a network.
* :func:`node_link_kuramoto_model`: phases on the nodes and on the edges at once, each coupling
  modulated by the order of the other (Ghorbanchian, Restrepo, Torres and Bianconi, 2021), whose
  explosive transition comes from the coupling of the two topological signals.

The dynamics are integrated with the classical Runge-Kutta method (fixed step ``dt``), in double
precision, on the device of the network, and order parameters are averaged over time after a
transient. A value of ``sigma`` or a sequence of them can be given, with independent ``runs``
(frequencies and initial phases drawn per run, shared by all ``sigma`` values). Without
``adiabatic``, all pairs of ``sigma`` and run are integrated at once, as the columns of one
matrix: a whole transition curve at the cost of a single simulation of many columns, which suits
a GPU. With ``adiabatic``, the values of ``sigma`` are followed in the given order, each starting
from the final phases of the previous one: a forward sweep followed by a backward one shows the
hysteresis loop of an explosive transition.

The coupling uses the boundary matrices of :func:`~graphoskop.boundary_matrix`: :math:`B_1` maps
edges to nodes, :math:`B_2` triangles to edges, and simplices have the reference orientation of
their nodes (see :meth:`~graphoskop.Network.set_simplex_attribute`). A plain network is a
complex without triangles: its edges, taken undirected, unweighted and without self-loops or
repetitions, in the order of their node pairs (internal indices), are its 1-simplices.
"""

import math

import numpy as np
import pandas as pd
import torch

from ._policy import _higher_order
from ._views import _walk_view
from .generators import _generator
from .higher_order import _operator, _simplices
from .network import _quiet_sparse
from .progress import _bar


def _edge_operator(net, component):
   """
   Boundary matrix :math:`B_1` of the edges of a plain network, and its transpose (sparse CSR,
   double), edges as distinct undirected node pairs ``u < v``, sorted; ``None`` without edges.
   """
   s, t = net.components[component].edge_index
   n = net.components[component].num_nodes
   keep = s != t
   key = torch.unique(torch.minimum(s, t)[keep] * n + torch.maximum(s, t)[keep])
   m = len(key)
   if not m:
      return None
   rows = torch.stack([key // n, key % n], 1).reshape(-1)
   cols = torch.arange(m, device = s.device).repeat_interleave(2)
   vals = torch.tensor([-1., 1.], dtype = torch.float64, device = s.device).repeat(m)
   with _quiet_sparse():
      B = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, m)).coalesce()
      return B.to_sparse_csr(), B.t().coalesce().to_sparse_csr()


def _coupling(net, component, k, what):
   """
   Boundary matrices around the :math:`k`-simplices: :math:`B_k` and :math:`B_{k + 1}` (each with
   its transpose, or ``None``), and the number of :math:`k`-simplices.

   :raises ValueError: If ``k`` is negative, or above 1 on a plain network.
   """
   if k < 0 or (net.higher_order is None and k > 1):
      raise ValueError(f"{what}: k must be at least 0{'' if net.higher_order else ', and at most 1 on a network (it has no triangles)'}, got {k}")
   if net.higher_order == "simplicial":
      return _operator(net, component, k), _operator(net, component, k + 1), len(_simplices(net, component, k))
   B1 = _edge_operator(net, component)
   n = [net.components[component].num_nodes, 0 if B1 is None else B1[0].shape[1]]
   return (B1 if k == 1 else None), (B1 if k == 0 else None), n[k]


def _values(x, n, runs, gen, device, uniform):
   """
   Natural frequencies or initial phases of ``n`` oscillators, one column per run: drawn (normal
   with ``x = (mean, sd)``, or uniform in :math:`[0, 2 \\pi)` if ``x`` is ``None`` and
   ``uniform``), or given (vector, shared by the runs, or ``n`` by ``runs`` matrix).
   """
   if x is None and uniform:
      return 2 * math.pi * torch.rand((n, runs), generator = gen, device = device, dtype = torch.float64)
   if isinstance(x, tuple):
      return x[0] + x[1] * torch.randn((n, runs), generator = gen, device = device, dtype = torch.float64)
   v = torch.as_tensor(x, dtype = torch.float64, device = device)
   if v.shape not in ((n,), (n, runs)):
      raise ValueError(f"expected a vector of length {n} or a matrix of {n} by {runs} (runs), got shape {tuple(v.shape)}")
   return (v[:, None] if v.ndim == 1 else v).expand(n, runs).clone()


def _order(y):
   """Order parameter of every column of phases (NaN without phases)."""
   if not len(y):
      return torch.full((y.shape[1],), float("nan"), dtype = y.dtype, device = y.device)
   return torch.hypot(torch.cos(y).mean(0), torch.sin(y).mean(0))


def _run(fields, rhs, observe, names, sigma, runs, adiabatic, time, dt, transient, seed, device, phases, progress, what):
   """
   Integrate oscillators for one or more coupling strengths and runs (see the module description).

   :param fields: One ``(n, omega, initial)`` per kind of oscillator (nodes, edges, ...).
   :param rhs: Function of ``(phases, frequencies, sigma)`` (tuples of ``n`` by ``b`` matrices,
      vector of length ``b``) giving the time derivatives.
   :param observe: Function of the phases giving the order parameters, ``len(names)`` by ``b``.
   :param names: Columns of the order parameters in the output.
   :return: Table, and the final phases if ``phases``.
   """
   if dt <= 0 or time < dt:
      raise ValueError(f"{what}: need 0 < dt <= time, got dt = {dt}, time = {time}")
   transient = time / 2 if transient is None else transient
   steps = int(round(time / dt))
   keep = steps - int(round(transient / dt))
   if not 0 < keep <= steps:
      raise ValueError(f"{what}: the transient must be at least 0 and shorter than the time, got {transient}")
   if runs < 1:
      raise ValueError(f"{what}: runs must be at least 1, got {runs}")
   sig = torch.as_tensor(np.atleast_1d(np.asarray(sigma, dtype = np.float64)), device = device)
   S = len(sig)
   gen = _generator(seed, device)
   omega = [_values(o, n, runs, gen, device, False) for n, o, _ in fields]
   state = [_values(x, n, runs, gen, device, True) for n, _, x in fields]
   blocks = [(i, i + 1) for i in range(S)] if adiabatic else [(0, S)]
   out, final = [], []
   with _bar(steps * len(blocks), what, progress) as bar:
      for a, b in blocks:
         w = tuple(o.repeat(1, b - a) for o in omega)
         s = sig[a:b].repeat_interleave(runs)
         x = tuple(state if adiabatic else (v.repeat(1, b - a) for v in state))
         f = lambda y: rhs(y, w, s)
         total = 0.
         for i in range(steps):
            k1 = f(x)
            k2 = f(tuple(v + dt / 2 * d for v, d in zip(x, k1)))
            k3 = f(tuple(v + dt / 2 * d for v, d in zip(x, k2)))
            k4 = f(tuple(v + dt * d for v, d in zip(x, k3)))
            x = tuple(torch.remainder(v + dt / 6 * (d1 + 2 * d2 + 2 * d3 + d4), 2 * math.pi) for v, d1, d2, d3, d4 in zip(x, k1, k2, k3, k4))
            if i >= steps - keep:
               total = total + observe(x)
            bar.update(1)
         out.append(total / keep)
         final.append(x)
         state = list(x)
   R = torch.cat(out, 1).cpu().numpy()
   table = pd.DataFrame({"sigma": np.repeat(sig.cpu().numpy(), runs), "run": np.tile(np.arange(runs), S), **{c: R[j] for j, c in enumerate(names)}})
   if not phases:
      return table
   return (table,) + tuple(torch.cat([x[j] for x in final], 1).reshape(-1, S, runs) for j in range(len(fields)))


@_higher_order(simplicial = True)
def kuramoto_model(net, sigma, omega = (0., 1.), theta = None, component = 0, weight = None, runs = 1, adiabatic = False, time = 100., dt = 0.01, transient = None, seed = None, phases = False, progress = None):
   """
   Kuramoto model on the nodes of a network: every node is an oscillator with phase
   :math:`\\theta_i` and natural frequency :math:`\\omega_i`, pulled by its neighbors,

   .. math:: \\dot\\theta_i = \\omega_i + \\sigma \\sum_j A_{ji} \\sin(\\theta_j - \\theta_i),

   with :math:`A_{ji}` the weight of the edge from :math:`j` to :math:`i` (directed networks: a
   node follows the nodes with edges towards it). For undirected networks this is
   :math:`\\dot\\theta = \\omega - \\sigma B_1 W \\sin(B_1^T \\theta)`. The coupling is not
   divided by the degree or the number of nodes: scale ``sigma`` for that. The order parameter
   ``order`` is :math:`R = |\\frac{1}{n} \\sum_j e^{i \\theta_j}|`, averaged over time after the
   transient. On a simplicial complex, the model runs on its 1-skeleton (for oscillators on its
   edges, see :func:`simplicial_kuramoto_model`). See the module description for ``sigma``,
   ``runs`` and ``adiabatic``.

   :param net: A :class:`~graphoskop.Network` or a simplicial complex.
   :param sigma: Coupling strength, or sequence of them.
   :param omega: Natural frequencies: ``(mean, sd)`` to draw them from a normal distribution for
      every run, or a vector over the nodes of the component (shared by the runs), or a matrix with
      one column per run.
   :param theta: Initial phases, as ``omega`` but without ``(mean, sd)``; ``None`` to draw them
      uniformly in :math:`[0, 2 \\pi)` for every run.
   :param component: Index of the component (0 is the largest).
   :param weight: Name of the edge attribute used as edge weight, or ``None`` for unit weights.
   :param runs: Number of independent runs (frequencies and initial phases).
   :param adiabatic: Whether to follow the values of ``sigma`` in order, each from the final phases
      of the previous one (hysteresis), instead of all at once from the initial phases.
   :param time: Integration time of each value of ``sigma``.
   :param dt: Time step.
   :param transient: Time discarded before averaging the order parameter, or ``None`` for half of
      ``time``.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param phases: Whether to also return the final phases.
   :param progress: Whether to show a progress bar over the time steps, or ``None`` for the global
      setting (see :doc:`progress`).
   :return: :class:`pandas.DataFrame` with one row per value of ``sigma`` (in the given order) and
      run: columns ``sigma``, ``run`` and ``order``; with ``phases``, also the final phases, a
      tensor ``[n, len(sigma), runs]``.
   :raises ValueError: If ``net`` is a hypergraph, a vector has the wrong shape, or the times are
      inconsistent.

   Example:

   .. code-block:: python

      import graphoskop as gsk
      import numpy as np

      net = gsk.load_network("edges.tsv")
      curve = gsk.kuramoto_model(net, np.linspace(0, 1, 21), runs = 10, seed = 0)
      curve.groupby("sigma")["order"].mean()                 # the synchronization transition
      up = np.linspace(0, 1, 21)
      loop = gsk.kuramoto_model(net, np.r_[up, up[::-1]], adiabatic = True, seed = 0)
   """
   net, weight = _walk_view(net, weight)
   data = net.components[component]
   s, t = data.edge_index
   w = torch.ones(len(s), dtype = torch.float64, device = s.device) if weight is None else net._numeric_edge(weight, component).double()
   n = data.num_nodes
   with _quiet_sparse():
      At = torch.sparse_coo_tensor(torch.stack([t, s]), w, (n, n)).coalesce().to_sparse_csr()   # A^T: row i holds the edges into i

   def rhs(x, omega, sig):
      c, z = torch.cos(x[0]), torch.sin(x[0])
      return (omega[0] + sig * (c * (At @ z) - z * (At @ c)),)   # sum_j A_ji sin(theta_j - theta_i), without a sine per edge

   return _run([(n, omega, theta)], rhs, lambda x: _order(x[0])[None], ["order"], sigma, runs, adiabatic, time, dt, transient, seed, s.device, phases, progress, "kuramoto_model")


@_higher_order(simplicial = True)
def simplicial_kuramoto_model(net, sigma, k = 1, explosive = False, omega = (0., 1.), theta = None, component = 0, runs = 1, adiabatic = False, time = 100., dt = 0.01, transient = None, seed = None, phases = False, progress = None):
   """
   Higher-order Kuramoto model of Millán, Torres and Bianconi (2020): oscillators on the
   :math:`k`-simplices of a simplicial complex (on its edges for :math:`k = 1`: a topological
   signal), each coupled with the simplices it shares a face or a coface with,

   .. math:: \\dot\\theta = \\omega - \\sigma B_{k + 1} \\sin(B_{k + 1}^T \\theta)
      - \\sigma B_k^T \\sin(B_k \\theta).

   The phases project on the :math:`(k - 1)`-simplices, :math:`\\theta^- = B_k \\theta` (a
   discrete divergence), and on the :math:`(k + 1)`-simplices, :math:`\\theta^+ = B_{k + 1}^T
   \\theta` (a discrete curl), whose dynamics decouple; their order parameters ``order_down`` and
   ``order_up`` (:math:`R^-` and :math:`R^+`) synchronize, while the phases themselves need not
   (``order``). With ``explosive``, each coupling is modulated by the order of the other
   projection, :math:`\\sigma R^-` on the first term and :math:`\\sigma R^+` on the second, which
   makes the transition discontinuous, with hysteresis (follow it with ``adiabatic``).

   For :math:`k = 0` this is the Kuramoto model of the 1-skeleton (:func:`kuramoto_model`). On a
   plain network, :math:`k` is 0 or 1, and edges have no triangles: the second term only. The
   order parameter of a projection with no simplex (``order_down`` for :math:`k = 0`, ``order_up``
   without :math:`(k + 1)`-simplices) is NaN. Phases are over :meth:`~graphoskop.Network.simplices`
   (or the sorted edges of a network), in reference orientation. See the module description for
   ``sigma``, ``runs`` and ``adiabatic``.

   :param net: A simplicial complex or a :class:`~graphoskop.Network`.
   :param sigma: Coupling strength, or sequence of them.
   :param k: Dimension of the simplices carrying the oscillators.
   :param explosive: Whether to use the explosive (adaptive) coupling.
   :param omega: Natural frequencies, ``(mean, sd)`` of a normal distribution or values (see
      :func:`kuramoto_model`), over the :math:`k`-simplices.
   :param theta: Initial phases, ``None`` for uniform ones (see :func:`kuramoto_model`).
   :param component: Index of the component (0 is the largest).
   :param runs: Number of independent runs.
   :param adiabatic: Whether to follow the values of ``sigma`` in order (see
      :func:`kuramoto_model`).
   :param time: Integration time of each value of ``sigma``.
   :param dt: Time step.
   :param transient: Time discarded before averaging, or ``None`` for half of ``time``.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param phases: Whether to also return the final phases.
   :param progress: Whether to show a progress bar over the time steps, or ``None`` for the global
      setting (see :doc:`progress`).
   :return: :class:`pandas.DataFrame` with one row per value of ``sigma`` and run: columns
      ``sigma``, ``run``, ``order``, ``order_down`` and ``order_up``; with ``phases``, also the
      final phases, a tensor ``[n_k, len(sigma), runs]``.
   :raises ValueError: If ``net`` is a hypergraph, ``k`` is out of range, ``explosive`` lacks the
      simplices of one projection, a vector has the wrong shape, or the times are inconsistent.

   Example:

   .. code-block:: python

      import graphoskop as gsk
      import numpy as np

      net = gsk.clique_promotion(gsk.load_network("edges.tsv"))       # the clique complex
      up = np.linspace(0, 3, 31)
      loop = gsk.simplicial_kuramoto_model(net, np.r_[up, up[::-1]], explosive = True, omega = (2., 1.), adiabatic = True, seed = 0)
   """
   down, upper, n = _coupling(net, component, k, "simplicial_kuramoto_model")
   if explosive and (down is None or upper is None):
      raise ValueError(f"the explosive model needs simplices of dimensions {k - 1} and {k + 1} around the {k}-simplices")
   device = net.components[component].edge_index.device

   def projections(th):
      return (None if down is None else down[0] @ th), (None if upper is None else upper[1] @ th)

   def rhs(x, omega, sig):
      th = x[0]
      lo, hi = projections(th)
      d = omega[0].clone()
      if hi is not None:
         d -= sig * (_order(lo) if explosive else 1.) * (upper[0] @ torch.sin(hi))
      if lo is not None:
         d -= sig * (_order(hi) if explosive else 1.) * (down[1] @ torch.sin(lo))
      return (d,)

   def observe(x):
      lo, hi = projections(x[0])
      nan = torch.full((x[0].shape[1],), float("nan"), dtype = torch.float64, device = device)
      return torch.stack([_order(x[0]), nan if lo is None else _order(lo), nan if hi is None else _order(hi)])

   return _run([(n, omega, theta)], rhs, observe, ["order", "order_down", "order_up"], sigma, runs, adiabatic, time, dt, transient, seed, device, phases, progress, "simplicial_kuramoto_model")


@_higher_order(simplicial = True)
def node_link_kuramoto_model(net, sigma, model = "nl", omega = (0., 1.), link_omega = (0., 1.), theta = None, phi = None, component = 0, runs = 1, adiabatic = False, time = 100., dt = 0.01, transient = None, seed = None, phases = False, progress = None):
   """
   Coupled Kuramoto dynamics of node and link signals (Ghorbanchian, Restrepo, Torres and Bianconi,
   2021): phases :math:`\\theta` on the nodes and :math:`\\phi` on the edges of a network or
   simplicial complex, each coupled through the boundary matrices and modulated by the order of
   the other,

   .. math::
      \\dot\\theta &= \\omega - \\sigma R_1^{down} B_1 \\sin(B_1^T \\theta), \\\\
      \\dot\\phi &= \\tilde\\omega - \\sigma R_0 (R_1^{up})^{m - 1} B_1^T \\sin(B_1 \\phi)
      - \\sigma (R_1^{down})^{m - 1} B_2 \\sin(B_2^T \\phi),

   with the order parameters :math:`R_0` of the node phases (``order_nodes``), :math:`R_1^{down}`
   of the projection of the link phases on the nodes, :math:`B_1 \\phi` (``order_down``), and
   :math:`R_1^{up}` of their projection on the triangles, :math:`B_2^T \\phi` (``order_up``). Model
   ``"nl"`` (:math:`m = 1`) couples nodes and links; model ``"nlt"`` (:math:`m = 2`) also couples
   the triangles, and needs them. The coupling of the two signals makes the transition
   explosive, with hysteresis (follow it with ``adiabatic``). On a plain network there are no
   triangles: the last term is absent and ``order_up`` is NaN. See the module description for
   ``sigma``, ``runs`` and ``adiabatic``.

   :param net: A :class:`~graphoskop.Network` or a simplicial complex.
   :param sigma: Coupling strength, or sequence of them.
   :param model: ``"nl"`` (nodes and links) or ``"nlt"`` (nodes, links and triangles).
   :param omega: Natural frequencies of the nodes, ``(mean, sd)`` of a normal distribution or
      values (see :func:`kuramoto_model`).
   :param link_omega: Natural frequencies of the edges, as ``omega``, over the edges in the order
      of :meth:`~graphoskop.Network.simplices` (or the sorted edges of a network).
   :param theta: Initial node phases, ``None`` for uniform ones.
   :param phi: Initial edge phases, ``None`` for uniform ones.
   :param component: Index of the component (0 is the largest).
   :param runs: Number of independent runs.
   :param adiabatic: Whether to follow the values of ``sigma`` in order (see
      :func:`kuramoto_model`).
   :param time: Integration time of each value of ``sigma``.
   :param dt: Time step.
   :param transient: Time discarded before averaging, or ``None`` for half of ``time``.
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :param phases: Whether to also return the final phases.
   :param progress: Whether to show a progress bar over the time steps, or ``None`` for the global
      setting (see :doc:`progress`).
   :return: :class:`pandas.DataFrame` with one row per value of ``sigma`` and run: columns
      ``sigma``, ``run``, ``order_nodes``, ``order_down`` and ``order_up``; with ``phases``, also
      the final node and edge phases, tensors ``[n, len(sigma), runs]`` and
      ``[edges, len(sigma), runs]``.
   :raises ValueError: If ``net`` is a hypergraph or has no edges, ``model`` is unknown or
      ``"nlt"`` without triangles, a vector has the wrong shape, or the times are inconsistent.

   Example:

   .. code-block:: python

      import graphoskop as gsk
      import numpy as np

      net = gsk.load_network("edges.tsv")
      up = np.linspace(0, 5, 26)
      loop = gsk.node_link_kuramoto_model(net, np.r_[up, up[::-1]], omega = (2., 1.), link_omega = (2., 1.), adiabatic = True, seed = 0)
   """
   if model not in ("nl", "nlt"):
      raise ValueError(f"unknown model {model!r}: use 'nl' or 'nlt'")
   B1, B2, n = _coupling(net, component, 1, "node_link_kuramoto_model")
   if B1 is None:
      raise ValueError("node_link_kuramoto_model needs edges")
   if model == "nlt" and B2 is None:
      raise ValueError("model 'nlt' needs triangles (a simplicial complex with filled triangles)")
   device = B1[0].device

   def orders(x):
      th, ph = x
      down, up = B1[0] @ ph, None if B2 is None else B2[1] @ ph
      return _order(th), _order(down), (None if up is None else _order(up)), down, up

   def rhs(x, omega, sig):
      th, ph = x
      r0, rd, ru, down, up = orders(x)
      dth = omega[0] - sig * rd * (B1[0] @ torch.sin(B1[1] @ th))
      dph = omega[1] - sig * r0 * (ru if model == "nlt" else 1.) * (B1[1] @ torch.sin(down))
      if up is not None:
         dph = dph - sig * (rd if model == "nlt" else 1.) * (B2[0] @ torch.sin(up))
      return dth, dph

   def observe(x):
      r0, rd, ru, _, _ = orders(x)
      return torch.stack([r0, rd, torch.full_like(r0, float("nan")) if ru is None else ru])

   return _run([(B1[0].shape[0], omega, theta), (n, link_omega, phi)], rhs, observe, ["order_nodes", "order_down", "order_up"], sigma, runs, adiabatic, time, dt, transient, seed, device, phases, progress, "node_link_kuramoto_model")
