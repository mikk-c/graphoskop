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

Exponential random graph models, with node attributes and dyadic covariates, are written and
fitted as in R's ``ergm`` package (see :doc:`ergm`).

Epidemics spread on networks with compartmental models (SI, SIS, SIR), by simple or complex
contagion, simulated in parallel runs (see :doc:`epidemics`). The robustness of a network to node
or edge failures and attacks, another percolation process, and cascades of failures driven by
loads and capacities are in :doc:`robustness`.

Core-periphery structures, a dense core of nodes and a sparse periphery attached to it, are fit
with the discrete and continuous models of Borgatti and Everett and the model of Rombach et al. (see
:doc:`core_periphery`).

Link prediction hides some edges of a network and scores the hidden pairs, and as many
unconnected ones, from the rest of the network (see :doc:`link_prediction`). Network error
correction does both jobs of cleaning observed networks, finding spurious and missing edges, with
the nested stochastic blockmodel (see :doc:`error_correction`).
"""

from importlib.metadata import version, PackageNotFoundError

from .network import BIPARTITE, LAYER, Network, load_network, save_network, load_pickled_network, load_matrix, save_matrix
from .multilayer import layer_network, aggregate_network
from .transformations import to_undirected, remove_isolates, remove_self_loops, remove_parallel_edges, subgraph, ego_network, to_strongly_connected, add_nodes, add_edges, remove_nodes, remove_edges, flip_edge_direction, network_union, network_difference, network_intersection, quotient_network, network_condensation
from .signed import SIGN, signed_network, signed_adjacency, structural_balance, signed_triangle_counts
from .utils import laplacian, laplacian_pinv, laplacian_solver, stationary_distribution, effective_resistance, hitting_time
from .shortest_paths import shortest_path, shortest_path_length, single_source_shortest_paths, single_source_shortest_path_lengths, all_pairs_shortest_paths, all_pairs_shortest_path_lengths, shortest_path_length_distribution, diameter
from .communities import modularity, maximal_cliques, block_model_communities
from .core_periphery import discrete_core_periphery, continuous_core_periphery, rombach_core_periphery, nestedness, nestedness_isocline
from .coloring import graph_coloring, maximum_independent_set
from .exploration import bfs, dfs, random_walk, metropolis_hastings_random_walk
from .centrality import closeness_centrality, harmonic_centrality, betweenness_centrality, edge_betweenness_centrality, flow_centrality, reach_centrality, eigenvector_centrality, pagerank, katz_centrality, alpha_centrality, hits, k_core_centrality, springrank, centralization
from .hierarchy import global_reach_centrality, flow_hierarchy, arborescence_hierarchy, agony_hierarchy
from .equivalence import structural_equivalence, automorphic_equivalence, pairwise_structural_equivalence, pairwise_automorphic_equivalence, regular_equivalence, pairwise_regular_equivalence, path_regular_equivalence, pairwise_path_regular_equivalence
from .network_statistics import connected_components_count, connected_components_nodes, degree, average_degree, degree_power_law, degree_centralization, density, reciprocity, average_shortest_path_length, local_clustering, global_clustering, average_clustering, assortativity
from .backboning import noise_corrected, convex_skeleton, disparity_filter, polya_filter, doubly_stochastic, high_salience_skeleton, extract_backbone, test_bb_thresholds, find_bb_thresholds, spanning_tree, maximally_filtered_graph, triangulated_filtered_graph
from .bipartite_projections import simple_count_projection, hist_intersection_projection, hyperbolic_projection, resource_allocation_projection, random_walk_projection, cosine_similarity_projection, pearson_correlation_projection, inverse_euclidean_projection
from .generators import chain_graph, gnm_random_graph, gnp_random_graph, configuration_model, edge_swap, kronecker_graph, geometric_graph, caveman_graph, watts_strogatz_graph, barabasi_albert_graph, link_selection_graph, copying_graph, holme_kim_graph, planted_partition_graph, stochastic_block_model, degree_corrected_block_model, mixed_membership_block_model, nested_block_model, girvan_newman_graph, lfr_benchmark_graph, randomization_test
from .ergms import ergm, ergm_statistics, ERGMFit
from .epidemics import si_model, sis_model, sir_model, complex_si_model, complex_sis_model, complex_sir_model
from .robustness import node_removal, edge_removal, cascade_failures
from .error_correction import fit_nested_block_model, measured_network_correction, uncertain_network_correction
from .link_prediction import link_prediction_split, preferential_attachment_lp, common_neighbors_lp, common_neighbors_jaccard_lp, adamic_adar_lp, resource_allocation_lp, car_index_lp, mutual_information_index_lp, local_random_walk_lp, simrank_lp, vertex_similarity_lp, hierarchical_random_graph_lp, block_model_reliability_lp, ensemble_link_prediction, link_prediction_evaluation, link_prediction_sign_evaluation
from .node_vector_analysis import generalized_euclidean, pairwise_generalized_euclidean, network_correlation, pairwise_network_correlation, network_variance, network_entropy

try:
   __version__ = version("graphoskop")
except PackageNotFoundError:
   __version__ = "unknown"

__all__ = ["BIPARTITE", "LAYER", "Network", "load_network", "save_network", "load_pickled_network",
           "layer_network", "aggregate_network", "to_undirected", "remove_isolates",
           "remove_self_loops", "remove_parallel_edges", "subgraph", "ego_network",
           "to_strongly_connected", "add_nodes", "add_edges", "remove_nodes", "remove_edges",
           "flip_edge_direction", "network_union", "network_difference", "network_intersection",
           "quotient_network", "network_condensation", "SIGN", "signed_network", "signed_adjacency",
           "structural_balance", "signed_triangle_counts", "laplacian", "laplacian_pinv",
           "laplacian_solver", "stationary_distribution", "effective_resistance", "hitting_time",
           "shortest_path", "shortest_path_length", "single_source_shortest_paths",
           "single_source_shortest_path_lengths", "all_pairs_shortest_paths",
           "all_pairs_shortest_path_lengths", "shortest_path_length_distribution", "diameter",
           "modularity", "maximal_cliques", "block_model_communities", "discrete_core_periphery",
           "continuous_core_periphery", "graph_coloring", "maximum_independent_set", "bfs", "dfs",
           "random_walk", "metropolis_hastings_random_walk", "closeness_centrality",
           "harmonic_centrality", "betweenness_centrality", "edge_betweenness_centrality",
           "flow_centrality", "reach_centrality", "eigenvector_centrality", "pagerank",
           "katz_centrality", "alpha_centrality", "hits", "k_core_centrality", "springrank",
           "centralization", "global_reach_centrality", "flow_hierarchy", "arborescence_hierarchy",
           "structural_equivalence", "automorphic_equivalence", "pairwise_structural_equivalence",
           "pairwise_automorphic_equivalence", "regular_equivalence",
           "pairwise_regular_equivalence", "path_regular_equivalence",
           "pairwise_path_regular_equivalence", "connected_components_count",
           "connected_components_nodes", "degree", "average_degree", "degree_power_law",
           "degree_centralization", "density", "reciprocity", "average_shortest_path_length",
           "local_clustering", "global_clustering", "average_clustering", "assortativity",
           "noise_corrected", "convex_skeleton", "disparity_filter", "polya_filter",
           "doubly_stochastic", "high_salience_skeleton", "extract_backbone", "test_bb_thresholds",
           "find_bb_thresholds", "spanning_tree", "maximally_filtered_graph",
           "triangulated_filtered_graph", "simple_count_projection", "hist_intersection_projection",
           "hyperbolic_projection", "resource_allocation_projection", "random_walk_projection",
           "cosine_similarity_projection", "pearson_correlation_projection",
           "inverse_euclidean_projection", "chain_graph", "gnm_random_graph", "gnp_random_graph",
           "configuration_model", "edge_swap", "kronecker_graph", "geometric_graph",
           "caveman_graph", "watts_strogatz_graph", "barabasi_albert_graph", "link_selection_graph",
           "copying_graph", "holme_kim_graph", "planted_partition_graph", "stochastic_block_model",
           "degree_corrected_block_model", "mixed_membership_block_model", "nested_block_model",
           "girvan_newman_graph", "lfr_benchmark_graph", "randomization_test", "ergm",
           "ergm_statistics", "ERGMFit", "si_model", "sis_model", "sir_model", "complex_si_model",
           "complex_sis_model", "complex_sir_model", "node_removal", "edge_removal",
           "cascade_failures", "link_prediction_split", "preferential_attachment_lp",
           "common_neighbors_lp", "common_neighbors_jaccard_lp", "adamic_adar_lp",
           "resource_allocation_lp", "car_index_lp", "mutual_information_index_lp",
           "local_random_walk_lp", "simrank_lp", "vertex_similarity_lp",
           "hierarchical_random_graph_lp", "block_model_reliability_lp", "ensemble_link_prediction",
           "link_prediction_evaluation", "link_prediction_sign_evaluation", "generalized_euclidean",
           "pairwise_generalized_euclidean", "network_correlation", "pairwise_network_correlation",
           "network_variance", "network_entropy", "fit_nested_block_model",
           "measured_network_correction", "uncertain_network_correction"]
