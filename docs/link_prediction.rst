Link prediction
===============

.. automodule:: graphoskop.link_prediction
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv", "nodes.tsv")
   train, validation = gsk.link_prediction_split(net, fraction = 0.1, negatives = 1, seed = 1)
   validation = gsk.preferential_attachment_lp(train, validation)            # adds a "score" column
   gsk.link_prediction_evaluation(validation)                 # {"auc": ..., "average_precision": ...}

   # several scores side by side, one column each
   for f in (gsk.common_neighbors_lp, gsk.common_neighbors_jaccard_lp, gsk.adamic_adar_lp, gsk.resource_allocation_lp,
             gsk.car_index_lp, gsk.mutual_information_index_lp, gsk.local_random_walk_lp, gsk.vertex_similarity_lp):
      validation = f(train, validation, name = f.__name__)
   validation = gsk.simrank_lp(train, validation, walks = 1000, name = "simrank_lp", seed = 1)
   validation = gsk.hierarchical_random_graph_lp(train, validation, name = "hrg", seed = 1)
   validation = gsk.block_model_reliability_lp(train, validation, name = "sbm", seed = 1)
   result = gsk.link_prediction_evaluation(validation, score = "hrg", curves = True)
   ax = result["roc_curve"].plot(x = "fpr", y = "tpr", label = "HRG")
   result["precision_recall_curve"].plot(x = "recall", y = "precision")

   # ensembles: aggregate the rankings of several scores
   validation = gsk.ensemble_link_prediction(train, validation, ["common_neighbors_lp", "local_random_walk_lp"], name = "borda")
   validation = gsk.ensemble_link_prediction(None, validation, aggregation = "kendall",
                                             columns = ["common_neighbors_lp", "local_random_walk_lp", "sbm"], name = "kemeny")

   # signed networks: can social balance and social status predict the signs of the hidden edges?
   signed = gsk.load_network("signed_edges.tsv", directed = True)   # edge attribute "sign"
   s_train, s_validation = gsk.link_prediction_split(signed, seed = 1)   # s_validation gets a "sign" column
   gsk.link_prediction_sign_evaluation(s_train, s_validation)         # balance_accuracy, status_accuracy, ...

   # weighted scores, one pair or a list of pairs
   validation = gsk.resource_allocation_lp(train, validation, weight = "weight", name = "ra_weighted")
   validation = gsk.local_random_walk_lp(train, validation, steps = 4, superposed = True, name = "srw")
   gsk.common_neighbors_lp(net, ("alice", "bob"))
   gsk.common_neighbors_jaccard_lp(net, [("alice", "bob"), ("bob", "carol")])

Ensembles
---------

Scores that capture different patterns can be combined by aggregating their rankings, with the
Borda count or a locally Kemeny optimal ranking (Kendall tau). On the benchmark networks below, an
ensemble of common neighbors, resource allocation, CAR, local random walks and preferential
attachment scores between its members (AUC 0.75 with Borda and 0.72 with Kendall on the karate
club, where local random walks alone reach 0.89; 0.90 on the communities, where the best member
reaches 0.92): aggregation helps when the scores are good and make different mistakes, and is
dragged down by weak ones.

.. autofunction:: graphoskop.ensemble_link_prediction

Evaluation
----------

A score is evaluated by how well it ranks the positives of the validation set above the
negatives. In signed networks, the signs of the positives can also be predicted from the signs
of the training network, by social balance or social status.

.. autofunction:: graphoskop.link_prediction_evaluation

.. autofunction:: graphoskop.link_prediction_sign_evaluation

Performance
-----------

On two CPU threads, splitting a random network with 1,000,000 nodes and 5,000,000 edges (10% of
the edges hidden, as many negatives) takes 2.8 s, most of it rebuilding the training network.
Scoring the 1,000,000 pairs of the validation set takes 0.6 s by preferential attachment, 5-6 s
by each of the common neighbor scores (2 s to sort the edges, 3 s to look up the 9,000,000
candidate common neighbors), 19 s by mutual information (most of it counting the triangles of
every node), 4.6 s by local random walks of 2 steps, 11.5 s of 3 steps, 17 s of 4 steps, and
66 s by SimRank (100 walks of at most 10 steps per pair, random accesses to memory). Evaluating
them takes 0.2 s, curves included. With 100,000 nodes and 500,000 edges, the split takes 0.2 s,
the common neighbor scores 0.3 s, mutual information 1.3 s, local random walks 0.3-1 s, SimRank
2.2 s, the evaluation 0.01 s.

Vertex similarity solves a linear system per distinct node on one side of the pairs, so its time
grows with the number of distinct nodes times the number of edges: on random networks with 5
edges per node and validation sets as above, 0.9 s with 2,000 nodes, 10 s with 5,000, 41 s with
10,000 (on the CPU; sparse products on a GPU are much faster).

Hierarchical random graphs and the block model reliability need thousands of sweeps of
:math:`n` moves of a sequential Markov chain, two chains in parallel on two threads (the first
call also compiles them, a few seconds). A hierarchical random graph move takes about 0.3
microseconds, almost independently of the size of the network. With the default 5,000 sweeps of
burn-in and 100 samples: under 0.1 s on the karate club (34 nodes), 0.1 s on 120 nodes and 541
edges, 0.9 s on 500 nodes and 4,158 edges, 1.8 s on 1,000 nodes and 9,323 edges, 3.9 s on 2,000
nodes and 18,489 edges; with 20,000 sweeps, which larger networks need (see
:func:`~graphoskop.hierarchical_random_graph_lp`), 3.5 s, 7 s and 16 s. A block model move costs the
number of groups, so with the default 2,000 sweeps the reliability takes 0.1 s on the karate
club, 0.5 s on 120 nodes, 5 s on 500 nodes and 13 s on 1,000.

For comparison, on the same networks and number of moves, the C++ implementation of
hierarchical random graphs in igraph (Clauset's original code, version 0.10.15, compiled with
``-O2``, one thread) takes 1.2 microseconds per move on the karate club, 4.6 on 500 nodes, 6.5 on
1,000 and 12.5 on 2,000, and PyHRG (pure Python on networkx) 120 microseconds on the karate club
and 520 on 500 nodes. igraph counts the edges between two subtrees by inserting the leaves of
both into balanced search trees, a cost that grows with the subtrees; here every subtree is a
range of positions in a leaf order, so the count only lists the neighbors of the smaller
subtree. Both chains make the same moves and reach the same likelihoods after the same number of
moves, with large differences between seeds in both.

How the scores compare (AUC, 10% of the edges hidden, as many negatives, default parameters,
seed 1). The two groups have 60 nodes each, with edges only between groups (the best possible
AUC is 0.78); the communities have 50 nodes each, with dense links within and few between (500
and 1,000 nodes).

.. list-table::
   :header-rows: 1

   * - Score
     - Karate club
     - Two groups
     - 10 communities
     - 20 communities
   * - :func:`~graphoskop.preferential_attachment_lp`
     - 0.63
     - 0.44
     - 0.50
     - 0.48
   * - :func:`~graphoskop.common_neighbors_lp`
     - 0.58
     - 0.32
     - 0.91
     - 0.87
   * - :func:`~graphoskop.common_neighbors_jaccard_lp`
     - 0.47
     - 0.32
     - 0.91
     - 0.87
   * - :func:`~graphoskop.adamic_adar_lp`
     - 0.60
     - 0.32
     - 0.91
     - 0.87
   * - :func:`~graphoskop.resource_allocation_lp`
     - 0.62
     - 0.32
     - 0.91
     - 0.87
   * - :func:`~graphoskop.car_index_lp`
     - 0.70
     - 0.50
     - 0.76
     - 0.74
   * - :func:`~graphoskop.mutual_information_index_lp`
     - 0.52
     - 0.65
     - 0.90
     - 0.86
   * - :func:`~graphoskop.local_random_walk_lp`
     - 0.89
     - 0.74
     - 0.92
     - 0.89
   * - :func:`~graphoskop.simrank_lp`
     - 0.52
     - 0.22
     - 0.93
     - 0.86
   * - :func:`~graphoskop.vertex_similarity_lp`
     - 0.39
     - 0.35
     - 0.93
     - 0.89
   * - :func:`~graphoskop.hierarchical_random_graph_lp`
     - 0.86
     - 0.77
     - 0.89
     - 0.86
   * - :func:`~graphoskop.block_model_reliability_lp`
     - 0.89
     - 0.78
     - 0.92
     - 0.89

Common neighbors only see triangles, so they fail where edges are between groups rather than
within them. Local random walks of 3 steps reach the nodes at distance 3, which, in the two
groups, are in the other group; hierarchical random graphs and block models learn either pattern.
Between seeds, the scores of the Markov chain models vary by a few hundredths.

Scores
------

Association rules
~~~~~~~~~~~~~~~~~

Frequent patterns that differ by one edge give association rules
(:func:`~graphoskop.graph_association_rules`): a pair is scored by the rules whose antecedent
occurs around it and whose added edge would join it, so the score learns any labeled local
structure that precedes an edge (feed-forward triads between roles, for instance), not only
triangles. Rules whose added edge brings a new node predict new neighbors that are not in the
network yet. In a directed network of 20,000 nodes with three roles (hubs, brokers and leaves
following a broker and two of its hubs, plus random edges), mining the training network
(:func:`~graphoskop.single_graph_mining`, MNI threshold 500, up to 3 edges: 206 patterns, 2.1 s)
gives 407 rules, which score the 14,774 validation pairs in 1 s with AUC 0.79, against 0.63 for
common neighbors and 0.76 for preferential attachment; predicting the new neighbors of all the
nodes too takes 5 s.

.. autofunction:: graphoskop.association_rules_lp

Common neighbors
~~~~~~~~~~~~~~~~

Two nodes with many neighbors in common are likely to become neighbors themselves (triadic
closure). The scores of this family count the common neighbors of the two nodes, each with a
weight, or as a fraction of all their neighbors, or with the links among them. Neighborhoods are
sets (self loops and parallel edges ignored); in directed networks, the score of ``(u, v)``
counts the paths :math:`u \to z \to v`. They work best in assortative networks with many
triangles, and give 0 to any pair at distance more than 2 (except
:func:`~graphoskop.mutual_information_index_lp`, which ranks those pairs by degree).

.. autofunction:: graphoskop.adamic_adar_lp

.. autofunction:: graphoskop.car_index_lp

.. autofunction:: graphoskop.common_neighbors_lp

.. autofunction:: graphoskop.common_neighbors_jaccard_lp

.. autofunction:: graphoskop.mutual_information_index_lp

.. autofunction:: graphoskop.resource_allocation_lp

Hierarchical random graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: graphoskop.hierarchical_random_graph_lp

Preferential attachment
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: graphoskop.preferential_attachment_lp

Random walks
~~~~~~~~~~~~

Two nodes are similar if random walkers move easily from one to the other, or, started at the
two, soon meet. Unlike common neighbors, walks of a few steps also relate nodes farther apart.

.. autofunction:: graphoskop.local_random_walk_lp

.. autofunction:: graphoskop.simrank_lp

Stochastic block models
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: graphoskop.block_model_reliability_lp

Vertex similarity
~~~~~~~~~~~~~~~~~

.. autofunction:: graphoskop.vertex_similarity_lp

Training and validation sets
----------------------------

.. autofunction:: graphoskop.link_prediction_split
