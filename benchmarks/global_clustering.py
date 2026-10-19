"""
Runtime of the global clustering coefficient (transitivity) on LFR graphs (see :mod:`harness`).
Writes ``global_clustering.csv`` and ``.png``.
"""

import graphoskop as gsk
from harness import run


def networkx_clustering(g):
   import networkx as nx
   return nx.transitivity(g)


def graph_tool_clustering(g):
   import graph_tool.all as gt
   return gt.global_clustering(g)[0]


SIZES = [80000, 160000, 320000, 640000, 1280000]

FUNCTIONS = {
   "graphoskop (GPU)": lambda g: gsk.global_clustering(g, 0),
   "graphoskop (CPU)": lambda g: gsk.global_clustering(g, 0),
   "graph-tool": graph_tool_clustering,
   "igraph": lambda g: g.transitivity_undirected(),
   "networkx": networkx_clustering,
}

if __name__ == "__main__":
   run("global_clustering", "Global clustering coefficient", FUNCTIONS, SIZES)
