# Graphoskop

Analysis of networks and of vectors defined on their nodes.

## Install

    pip install graphoskop

## Usage

    import graphoskop as gsk
    net = gsk.load_network("edges.tsv", "nodes.tsv")
    net = gsk.load_network(edges_df, nodes_df)          # or from pandas DataFrames
    edges_df, nodes_df = net.to_pandas_dataframes()     # and back
    gsk.save_network(net, "out/net")                    # out/net_edges.tsv, out/net_nodes.tsv (and _interlayer.tsv)
    gsk.save_network(net, "out/net", pickle = True)     # out/net.pt, read with gsk.load_pickled_network

Directed networks can be symmetrized, combining the attributes of the two directions:

    und = gsk.to_undirected(net, combine = "sum")       # or "mean", "max"; sign: disagree = "zero", "positive", "negative"
    core = gsk.remove_isolates(net)                     # drop nodes without edges
    clean = gsk.remove_self_loops(net)                  # drop edges from a node to itself
    simple = gsk.remove_parallel_edges(net)             # merge parallel edges (same direction, same layer)
    ego = gsk.subgraph(net, ["alice", "bob"])           # induced subgraph
    scc = gsk.to_strongly_connected(net)                # directed: split into strongly connected components

Degree and clustering:

    gsk.connected_components_count(net, type = "strong"), gsk.connected_components_nodes(net)
    gsk.degree(net, "alice"), gsk.degree(net, distribution = "ccdf")
    gsk.degree(net, "alice", weight = "weight")         # weighted degree (strength)
    gsk.degree_power_law(net)                           # exponent, k_min and cutoff, after likelihood ratio tests (Clauset et al.)
    gsk.degree_power_law(net, ccdf = True)["ccdf"]     # empirical and fitted ccdf, to plot
    gsk.degree_power_law(net)["formula"]                # best fit written out, e.g. "P(k) = 0.27 * k^(-2.37) / 0.026, for k >= 12"
    gsk.degree_centralization(net)["centralization"]    # degree entropy vs random networks of the same size
    gsk.average_degree(net)                             # 2m / n (undirected), m / n (directed)
    gsk.density(net)                                    # also directed, bipartite, multilayer (per layer)
    gsk.reciprocity(net), gsk.reciprocity(net, normalized = True)   # directed networks
    gsk.average_shortest_path_length(net)               # same paths as the shortest_paths module
    gsk.shortest_path_length_distribution(net)          # number of node pairs at each distance (edges)
    gsk.diameter(net)                                   # longest shortest path
    net.is_planar()                                     # drawable without edge crossings
    gsk.maximally_filtered_graph(net, "weight")         # planar backbone (PMFG), a new network
    gsk.triangulated_filtered_graph(net, "weight")      # faster planar backbone (TMFG), on the GPU
    gsk.bfs(net, "alice", max_depth = 2), gsk.dfs(net, "alice")   # exploration: node, depth, parent in visit order

Random walks: `gsk.stationary_distribution(net)`, `gsk.effective_resistance(net)`, and hitting times
`H = gsk.hitting_time(net)` (`H[i, j]`: expected steps from `i` to reach `j`; `alpha` for teleportation).

Centralities (closeness, harmonic, betweenness of nodes and edges) can share precomputed shortest path lengths:

    P = gsk.all_pairs_shortest_path_lengths(net)
    b = gsk.betweenness_centrality(net, lengths = P)    # also closeness_centrality, harmonic_centrality
    eb = gsk.edge_betweenness_centrality(net, lengths = P)   # per stored edge: net.set_edge_attribute("eb", eb)
    c = gsk.closeness_centrality(net, component = None, lin = True)   # all components, comparable
    f = gsk.flow_centrality(net)                        # random walk betweenness (Newman, 2005)
    gsk.eigenvector_centrality(net, weight = "weight")   # power iteration on the device
    gsk.pagerank(net, component = None, personalization = {"alice": 1})   # personalized PageRank
    hubs, authorities = gsk.hits(net)                   # HITS
    gsk.k_core_centrality(net)                          # core number of every node
    gsk.centralization(net, gsk.betweenness_centrality)   # Freeman centralization of any centrality
    gsk.structural_equivalence(net, "alice", "bob")   # Jaccard coefficient of their neighbors
    gsk.automorphic_equivalence(net, "alice", "bob")  # 1 if some automorphism swaps them
    gsk.pairwise_structural_equivalence(net), gsk.pairwise_automorphic_equivalence(net)   # all pairs, n x n
    gsk.pairwise_regular_equivalence(dnet)              # REGE: directed or valued networks
    gsk.pairwise_path_regular_equivalence(net, alpha = 0.97)   # Leicht, Holme and Newman similarity
    gsk.katz_centrality(net, alpha = 0.05)              # alpha below 1 / largest eigenvalue
    gsk.alpha_centrality(net, alpha = 0.5, exo = "wealth")   # Bonacich alpha centrality, any alpha
    gsk.reach_centrality(dnet)                          # directed networks: fraction of nodes reached
    gsk.global_reach_centrality(dnet)                   # hierarchy of a directed network
    gsk.modularity(net, "party")                        # partition from a node attribute, a dict or a list
    gsk.maximal_cliques(net)                            # node sets of maximal cliques with 3+ nodes
    gsk.graph_coloring(net), gsk.maximum_independent_set(net)   # exact when proven within time_limit
    gsk.local_clustering(net, "alice")                  # or a vector over a component
    gsk.global_clustering(net), gsk.average_clustering(net)
    gsk.assortativity(net), gsk.assortativity(net, "age"), gsk.assortativity(net, "city")   # degree, numeric, qualitative

Numeric node attributes go to the chosen `device` (e.g. `"cuda"`); qualitative ones (strings,
categories) always stay in CPU memory.

Bipartite networks need a `bipartite_type` column in the node table, with two values:

    net = gsk.load_network("edges.tsv", "nodes.tsv", bipartite = True)
    users = gsk.simple_count_projection(net, "user")   # projection onto the "user" nodes
    users = gsk.hyperbolic_projection(net, "user")     # also hist_intersection_projection
    items = gsk.resource_allocation_projection(net, "item", lambd = 1)   # ProbS; 0 HeatS, 0.5 hybrid
    items = gsk.random_walk_projection(net, "item")    # YCN random walks

Multilayer networks need a `layer` column in the edge table; nodes become copies `(entity, layer)`:

    net = gsk.load_network("edges.tsv", "nodes.tsv", multilayer = True, coupling = "all")   # or "ordinal", "temporal"
    gsk.laplacian(net)                                  # supra-Laplacian
    gsk.degree(net, ("alice", "work")), gsk.degree(net, "alice")   # in one layer, summed over layers
    work = gsk.layer_network(net, "work")               # one layer as an ordinary network
    flat = gsk.aggregate_network(net)                   # all layers flattened

Signed networks carry a +1 / -1 `sign` edge attribute; build them from a positive and a negative
layer, or by thresholding an edge weight:

    signed = gsk.signed_network(net, positive = "ally", negative = "enemy")   # multilayer net
    signed = gsk.signed_network(net, weight = "score", threshold = 0)         # >= 0 positive
    A = gsk.signed_adjacency(signed)                    # sparse, per component
    gsk.structural_balance(signed)                      # Tr(A^3) / Tr(|A|^3)
    gsk.structural_balance(signed, beta = 0.5)          # Tr(e^(beta A)) / Tr(e^(beta |A|)), all closed walks
    gsk.signed_triangle_counts(signed)                  # per sign pattern, vs. shuffled signs (exact)

Synthetic networks: every generator takes its parameters, or an existing network to randomize
(same nodes and attributes, new random edges, edge attributes shuffled over them):

    g = gsk.gnm_random_graph(1000, 5000, seed = 1)       # random: 1000 nodes, 5000 random edges
    g = gsk.gnp_random_graph(1000, 0.01, seed = 1)       # random: each pair an edge with probability 0.01
    b = gsk.gnm_random_graph((100, 300), 1000)           # bipartite
    c = gsk.caveman_graph(10, 5, connected = True)       # clustered: 10 cliques of 5 in a ring; node attribute "cave"
    w = gsk.watts_strogatz_graph(1000, 6, 0.1)           # clustered: small world; rewire = "target", "both" or "add"
    ba = gsk.barabasi_albert_graph(10000, 6)             # broad degree: preferential attachment, average degree 6 (m = 3)
    ls = gsk.link_selection_graph(10000, 6)              # broad degree: new nodes link to an end of random links
    cp = gsk.copying_graph(10000, 6)                     # broad degree: new nodes copy a link of random nodes
    hk = gsk.holme_kim_graph(10000, 6, 0.8)              # broad degree with clustering: triangles closed with probability 0.8
    rnd = gsk.gnm_random_graph(net = net)                # randomization of net, same n and m
    rnd = gsk.gnp_random_graph(net = net)                # same n, p = density of net
    rnd = gsk.caveman_graph(net = net)                   # nodes in random caves, cave size from the average degree
    rnd = gsk.watts_strogatz_graph(net = net, p = 0.1)   # nodes on a ring, same n and m, then rewired
    rnd = gsk.barabasi_albert_graph(net = net)           # same n and m, nodes arrive in random order
    rnd = gsk.link_selection_graph(net = net)            # same n and m, nodes arrive in random order
    rnd = gsk.copying_graph(net = net)                   # same n and m, nodes arrive in random order
    rnd = gsk.holme_kim_graph(net = net)                 # same n, m and average clustering (p inferred)

