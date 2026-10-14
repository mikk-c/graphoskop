"""
Graphoskop: analysis of networks and of vectors defined on their nodes.

All public functions and classes are available at the top level of the package.

Quick start, with an edge list ``edges.tsv``::

   source   target   weight
   alice    bob      1.5
   bob      carol    2.0

and a node table ``nodes.tsv``::

   id       age   height   city
   alice    34    170      Rome
   carol    29    165      Oslo

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv", "nodes.tsv")
   g = net.components[0]      # largest weakly connected component
   g.edge_index               # edges, internal node ids
   g.edge_attr                # weight, one row per edge
   g.node_attr                # age and height (numeric), one row per node
   net.qualitative_node_attr  # city (qualitative), always in CPU memory
   net.node_map["alice"]      # (component, index)

Multilayer networks are loaded with ``multilayer = True`` from an edge list with a ``layer``
column; nodes are then copies ``(entity, layer)`` (see :doc:`multilayer`). Signed networks carry
a +1 / -1 sign on their edges, built from a positive and a negative layer or by thresholding a
weight (see :doc:`signed`).

Synthetic networks come from generators, which can also randomize an existing network, keeping
its nodes and their attributes (see :doc:`generators`).
"""

from importlib.metadata import version, PackageNotFoundError

from .network import BIPARTITE, LAYER, Network, load_network, save_network, load_pickled_network
from .multilayer import layer_network, aggregate_network
from .transformations import to_undirected, remove_isolates, remove_self_loops, remove_parallel_edges, subgraph, to_strongly_connected
from .signed import SIGN, signed_network, signed_adjacency, structural_balance, signed_triangle_counts
from .utils import laplacian, laplacian_pinv, laplacian_solver, stationary_distribution, effective_resistance, hitting_time
from .shortest_paths import shortest_path, shortest_path_length, single_source_shortest_paths, single_source_shortest_path_lengths, all_pairs_shortest_paths, all_pairs_shortest_path_lengths, shortest_path_length_distribution, diameter
from .communities import modularity, maximal_cliques
from .coloring import graph_coloring, maximum_independent_set
from .exploration import bfs, dfs
from .centrality import closeness_centrality, harmonic_centrality, betweenness_centrality, edge_betweenness_centrality, flow_centrality, reach_centrality, eigenvector_centrality, pagerank, katz_centrality, alpha_centrality, hits, k_core_centrality, centralization
from .hierarchy import global_reach_centrality
from .equivalence import structural_equivalence, automorphic_equivalence, pairwise_structural_equivalence, pairwise_automorphic_equivalence, regular_equivalence, pairwise_regular_equivalence, path_regular_equivalence, pairwise_path_regular_equivalence
from .network_statistics import connected_components_count, connected_components_nodes, degree, average_degree, degree_power_law, degree_centralization, density, reciprocity, average_shortest_path_length, local_clustering, global_clustering, average_clustering, assortativity
from .backboning import noise_corrected, disparity_filter, doubly_stochastic, high_salience_skeleton, extract_backbone, test_bb_thresholds, find_bb_thresholds, spanning_tree, maximally_filtered_graph, triangulated_filtered_graph
from .bipartite_projections import simple_count_projection, hist_intersection_projection, hyperbolic_projection, resource_allocation_projection, random_walk_projection
from .generators import gnm_random_graph, gnp_random_graph, configuration_model, caveman_graph, watts_strogatz_graph, barabasi_albert_graph, link_selection_graph, copying_graph, holme_kim_graph, planted_partition_graph, stochastic_block_model
from .node_vector_analysis import generalized_euclidean, pairwise_generalized_euclidean, network_correlation, pairwise_network_correlation, network_variance, network_entropy

try:
   __version__ = version("graphoskop")
except PackageNotFoundError:
   __version__ = "unknown"

__all__ = ["BIPARTITE", "LAYER", "Network", "load_network", "save_network", "load_pickled_network", "layer_network", "aggregate_network", "to_undirected", "remove_isolates", "remove_self_loops", "remove_parallel_edges", "subgraph", "to_strongly_connected", "SIGN", "signed_network", "signed_adjacency", "structural_balance", "signed_triangle_counts", "laplacian", "laplacian_pinv", "laplacian_solver", "stationary_distribution", "generalized_euclidean", "pairwise_generalized_euclidean", "effective_resistance", "hitting_time", "network_correlation", "pairwise_network_correlation", "network_variance", "network_entropy", "shortest_path", "shortest_path_length", "single_source_shortest_paths", "single_source_shortest_path_lengths", "all_pairs_shortest_paths", "all_pairs_shortest_path_lengths", "shortest_path_length_distribution", "diameter", "noise_corrected", "disparity_filter", "doubly_stochastic", "high_salience_skeleton", "extract_backbone", "test_bb_thresholds", "find_bb_thresholds", "spanning_tree", "maximally_filtered_graph", "triangulated_filtered_graph", "connected_components_count", "connected_components_nodes", "degree", "average_degree", "degree_power_law", "degree_centralization", "density", "reciprocity", "average_shortest_path_length", "local_clustering", "global_clustering", "average_clustering", "assortativity", "modularity", "maximal_cliques", "graph_coloring", "maximum_independent_set", "bfs", "dfs", "closeness_centrality", "harmonic_centrality", "betweenness_centrality", "edge_betweenness_centrality", "flow_centrality", "reach_centrality", "eigenvector_centrality", "pagerank", "hits", "k_core_centrality", "centralization", "katz_centrality", "alpha_centrality", "global_reach_centrality", "structural_equivalence", "automorphic_equivalence", "pairwise_structural_equivalence", "pairwise_automorphic_equivalence", "regular_equivalence", "pairwise_regular_equivalence", "path_regular_equivalence", "pairwise_path_regular_equivalence", "simple_count_projection", "hist_intersection_projection", "hyperbolic_projection", "resource_allocation_projection", "random_walk_projection", "gnm_random_graph", "gnp_random_graph", "caveman_graph", "watts_strogatz_graph", "barabasi_albert_graph", "link_selection_graph", "copying_graph", "holme_kim_graph", "configuration_model", "planted_partition_graph", "stochastic_block_model"]
