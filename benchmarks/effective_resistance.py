"""
Runtime of the matrix of effective resistances between all pairs of nodes on LFR graphs (see
:mod:`harness`). Writes ``effective_resistance.csv`` and ``.png``.

NetworkX has its own function (:func:`networkx.resistance_distance`, all pairs as a dictionary of
dictionaries). igraph and graph-tool have none: their users would take the Laplacian from the
library and finish with NumPy, as NetworkX does internally (pseudoinverse of the dense Laplacian,
:math:`R_{uv} = L^+_{uu} + L^+_{vv} - 2 L^+_{uv}`). Results are compared through the sum of all
resistances.
"""

import numpy as np

import graphoskop as gsk
from harness import run


def resistance(L):
   """
   Effective resistances from a dense Laplacian, as NetworkX computes them.

   :param L: Dense Laplacian, NumPy array.
   :return: Dense matrix of effective resistances.
   """
   P = np.linalg.pinv(L, hermitian = True)
   d = np.diag(P)
   return d[:, None] + d[None, :] - 2 * P


def networkx_er(g):
   import networkx as nx
   return nx.resistance_distance(g)


def igraph_er(g):
   from scipy.sparse.csgraph import laplacian
   return resistance(laplacian(g.get_adjacency_sparse().astype(float)).toarray())


def graph_tool_er(g):
   import graph_tool.all as gt
   return resistance(gt.laplacian(g).toarray())


SIZES = [2000, 2500, 3000, 3500, 4000]   # networkx: about 5 minutes on the largest

FUNCTIONS = {
   "graphoskop (GPU)": gsk.effective_resistance,
   "graphoskop (CPU)": gsk.effective_resistance,
   "graph-tool": graph_tool_er,
   "igraph": igraph_er,
   "networkx": networkx_er,
}

SUMMARY = {
   "graphoskop (GPU)": lambda R: R.double().sum().item(),
   "graphoskop (CPU)": lambda R: R.double().sum().item(),
   "graph-tool": np.sum,
   "igraph": np.sum,
   "networkx": lambda d: sum(sum(row.values()) for row in d.values()),
}

if __name__ == "__main__":
   run("effective_resistance", "Effective resistance (all pairs)", FUNCTIONS, SIZES, SUMMARY)
