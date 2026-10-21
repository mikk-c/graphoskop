"""
Runtime of community discovery with the hierarchical degree-corrected stochastic blockmodel (the
nested blockmodel of Peixoto, 2014) on LFR graphs (see :mod:`harness`). Writes
``hierarchical_block_model.csv`` and ``.png``.

Only Graphoskop (:func:`graphoskop.block_model_communities`, whose default model is the nested
degree-corrected blockmodel) and graph-tool (:func:`graph_tool.inference.minimize_nested_blockmodel_dl`,
degree-corrected by default) fit it: NetworkX and igraph have no stochastic blockmodels. Both
minimize the description length with their default settings; the fits are heuristic, so the
description lengths found (in nats) are saved and printed but not required to agree. Graphoskop
compiles its kernels on first use: the untimed warm-up runs of the harness compile them, so the
timed runs are those of the compiled code. Its fit runs on the CPU whatever the device, so the
two Graphoskop lines differ only by the transfer of the edges.
"""

import graphoskop as gsk
from harness import run


def graphoskop_sbm(g):
   return gsk.block_model_communities(g, seed = 0)


def graph_tool_sbm(g):
   import graph_tool.all as gt
   return gt.minimize_nested_blockmodel_dl(g)


SIZES = [2000, 4000, 8000, 12000, 16000]   # the slowest: about a minute on the largest

FUNCTIONS = {
   "graphoskop (GPU)": graphoskop_sbm,
   "graphoskop (CPU)": graphoskop_sbm,
   "graph-tool": graph_tool_sbm,
}

SUMMARY = {   # description length, in nats
   "graphoskop (GPU)": lambda r: r["description_length"],
   "graphoskop (CPU)": lambda r: r["description_length"],
   "graph-tool": lambda state: state.entropy(),
}

if __name__ == "__main__":
   run("hierarchical_block_model", "Communities: hierarchical degree-corrected blockmodel", FUNCTIONS, SIZES, SUMMARY, compare = False)
