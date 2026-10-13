"""
Hierarchy in directed networks: how far the nodes are organized in levels, with some nodes
reaching (or controlling) many others and being reached by few.

:func:`global_reach_centrality` measures it from the local reaching centrality of the nodes (see
:mod:`~graphoskop.centrality`).
"""

import torch

from .centrality import reach_centrality


def global_reach_centrality(net, component = 0, direction = "out", lengths = None, n_jobs = None):
   """
   Global reaching centrality (Mones, Vicsek and Vicsek, 2012), a measure of hierarchy: how much
   the most reaching node exceeds the others,
   :math:`GRC = \\sum_i (C_R^{max} - C_R(i)) / (N - 1)`, with :math:`C_R` the local reaching
   centrality of :func:`~graphoskop.reach_centrality` and :math:`C_R^{max}` its largest value. It
   is 1 for a star with edges out of its center and 0 when every node reaches all the others (e.g.
   a directed cycle). With ``component = None`` the sum and the maximum run over all components.

   :param net: A directed :class:`~graphoskop.Network`.
   :param component: Index of the component (0 is the largest), or ``None`` for all components at
      once.
   :param direction: ``"out"`` (nodes reached) or ``"in"`` (nodes reaching).
   :param lengths: Precomputed lengths, as for :func:`~graphoskop.reach_centrality`, or ``None``.
   :param n_jobs: Number of processes for the searches, ``None`` for all CPUs.
   :return: The global reaching centrality, a float.
   :raises ValueError: If the network is undirected, ``direction`` is not valid, or ``lengths``
      does not match.

   Example:

   .. code-block:: python

      gsk.global_reach_centrality(net)                    # largest component
   """
   r = reach_centrality(net, component, direction, lengths, n_jobs)
   r = torch.cat(r) if component is None else r
   N = r.shape[0]
   return float((r.max() - r).sum() / (N - 1)) if N > 1 else 0.
