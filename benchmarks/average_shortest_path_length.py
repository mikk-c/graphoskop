"""
Runtime of the average shortest path length (all pairs, exact) on LFR graphs (see
:mod:`harness`). Writes ``average_shortest_path_length.csv`` and ``.png``.
"""

import numpy as np

import graphoskop as gsk
from harness import run


def networkx_apl(g):
   import networkx as nx
   return nx.average_shortest_path_length(g)


def graph_tool_apl(g):
   import graph_tool.all as gt
   counts, bins = gt.distance_histogram(g)   # all pairs, unit bins starting at distance 0
   d = bins[:len(counts)]
   return (counts * d).sum() / counts[d > 0].sum()


SIZES = [4000, 6000, 8000, 10000, 12000]   # networkx: about 5 minutes on the largest

FUNCTIONS = {
   "graphoskop (GPU)": gsk.average_shortest_path_length,
   "graphoskop (CPU)": gsk.average_shortest_path_length,
   "graph-tool": graph_tool_apl,
   "igraph": lambda g: g.average_path_length(directed = False),
   "networkx": networkx_apl,
}

if __name__ == "__main__":   # needed by the parallel searches of graphoskop
   run("average_shortest_path_length", "Average shortest path length", FUNCTIONS, SIZES)
