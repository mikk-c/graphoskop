Graph neural networks
=====================

.. automodule:: graphoskop.gnn
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv", "nodes.tsv")       # node attributes: age, income, city
   out = gsk.gcn(net, "city", [32, 16, 4], qualitative = True, split = (0.6, 0.2), seed = 1)
   out["scores"]                                          # accuracy and macro F1, training and validation
   out["predictions"].query("set == 'unlabeled'")         # predicted city of the nodes without one
   out["history"].plot(x = "epoch")                       # training and validation loss
   reg = gsk.gcn(net, "income", [16, 1], activations = ["tanh", None], epochs = 500, learning_rate = 0.005)

Autoencoders
------------

Graph autoencoders learn node vectors without a target: an encoder of GCN layers gives every node
a vector from the node attributes and the edges, trained so that the dot products of the vectors
reconstruct the edges. Unlike the shallow embeddings below, they use the node attributes, and the
trained encoder (``out["model"]``) can embed new nodes from their attributes.

.. code-block:: python

   train, validation = gsk.link_prediction_split(net, seed = 1)
   out = gsk.vgae(train, seed = 1, attribute = "z")          # vectors stored as z_0 ... z_15
   scored = gsk.embedding_lp(train, validation, [f"z_{j}" for j in range(16)])

.. autofunction:: graphoskop.gae

.. autofunction:: graphoskop.vgae

Graph embeddings
----------------

A network whose components are the graphs of a dataset (molecules, ego networks...) gets one
vector per graph with :func:`~graphoskop.diffpool`, trained to predict a graph attribute or, without
one, unsupervised.

.. code-block:: python

   molecules = gsk.load_network("bonds.tsv", "atoms.tsv")    # one component per molecule
   out = gsk.diffpool(molecules, "toxic", seed = 1)           # a node attribute, one value per molecule
   out["scores"], out["predictions"]                          # as for the node models below
   free = gsk.diffpool(molecules, seed = 1)["embeddings"]     # unsupervised: one row per molecule

.. autofunction:: graphoskop.diffpool

Higher-order networks
---------------------

Hypergraphs and simplicial complexes (see :doc:`higher_order`) have two models, with the
parameters and results of the models below. :func:`~graphoskop.hgnn` predicts a node attribute:
its layers (the hypergraph convolution of ``HypergraphConv`` in PyTorch Geometric) pass features
from the nodes to their hyperedges and back, so that a group of any size counts once, without the many pairs of the clique
expansion; on a simplicial complex, the facets are the hyperedges. :func:`~graphoskop.scnn`
predicts a simplex attribute of a simplicial complex, a value or a class of every edge (or
triangle...), from the other attributes of the simplices of that dimension: its layers mix every
simplex with those sharing a face with it and, separately, those sharing a coface with it,
through the lower and upper Hodge Laplacians (see :ref:`hodge-laplacians`). The other models
refuse higher-order networks: pass them :func:`~graphoskop.clique_expansion` for the network of
co-members.

.. code-block:: python

   import graphoskop as gsk

   hg = gsk.load_hypergraph("authorships.tsv", "authors.tsv")      # papers and their authors
   out = gsk.hgnn(hg, "field", [32, 5], seed = 1)                  # a node attribute
   out = gsk.hgnn(hg, "field", [32, 5], weight = "citations")      # a hyperedge attribute

   sc = gsk.load_simplicial_complex("cliques.tsv")
   sc.set_simplex_attribute(edge_table, dimension = 1)              # node_0, node_1, kind, weight
   out = gsk.scnn(sc, "kind", [32, 3], order = 2, seed = 1)         # a class of every edge
   out["predictions"]                                               # one row per edge, Simplex ids

On a planted hypergraph (400 nodes in 4 groups, 600 hyperedges of 3 to 6 nodes, 85% inside a
group, one noisy group feature), :func:`~graphoskop.hgnn` reaches a validation accuracy of 0.84
(0.45 from the features alone), and :func:`~graphoskop.gcn` on the clique expansion 0.95: the
hypergraph layer keeps more of every node's own (noisy) features, which here smooths too little;
on 100,000 nodes and 100,000 hyperedges, 50 epochs take 11 s on two CPU threads. With a constant
feature only, :func:`~graphoskop.scnn` tells the edges in a filled triangle from the others
(accuracy 1.0), which needs the upper neighbors; on 72,849 edges and 42,850 triangles, 50 epochs
take 10 s.

.. autofunction:: graphoskop.hgnn

.. autofunction:: graphoskop.scnn

Models
------

All models take the same parameters and return the same results; the attention models also take
the number of heads. The weights of ``weight`` average the neighbors in :func:`~graphoskop.gcn`
and :func:`~graphoskop.graphsage`, and enter the attention as an edge feature in
:func:`~graphoskop.gat` and :func:`~graphoskop.gatv2`.

.. autofunction:: graphoskop.gat

.. autofunction:: graphoskop.gatv2

.. autofunction:: graphoskop.gcn

.. autofunction:: graphoskop.graphsage

Shallow embeddings
------------------

Shallow embeddings learn one vector per node directly, from the structure alone: no node
attribute is used or predicted. The vectors can be stored as node attributes and used anywhere
numeric attributes are, e.g. as features of the models above, in node vector analysis or for
clustering. :func:`~graphoskop.node2vec` learns them from random walks (for DeepWalk, use
``p = q = 1``, the default), so that nearby nodes get similar vectors;
:func:`~graphoskop.struc2vec` from walks on a multilayer graph of structural similarity, so that
nodes with similar roles get similar vectors, wherever they are;
:func:`~graphoskop.metapath2vec` from walks that follow a chain of node types (and relations) in a
heterogeneous network; :func:`~graphoskop.laplacian_eigenmaps` and
:func:`~graphoskop.locally_linear_embedding` compute them as eigenvectors, with no training and
nothing random.

.. code-block:: python

   emb = gsk.node2vec(net, seed = 1)                       # DataFrame: one row per node id
   emb = gsk.node2vec(net, q = 0.5, attribute = "n2v")     # also node attributes n2v_0, n2v_1, ...
   le = gsk.laplacian_eigenmaps(net, 16)                   # eigenvectors: exact, no training
   lle = gsk.locally_linear_embedding(net, 16, attribute = "lle")
   roles = gsk.struc2vec(net, seed = 1)                    # structural roles, not proximity
   typed = gsk.metapath2vec(net, "kind", ["author", "paper", "venue", "paper", "author"])
   out = gsk.gcn(net, "city", [16, 4], features = [f"n2v_{j}" for j in range(128)])

Stored vectors also predict links, with :func:`~graphoskop.embedding_lp` (see :doc:`link_prediction`).

.. autofunction:: graphoskop.laplacian_eigenmaps

.. autofunction:: graphoskop.locally_linear_embedding

.. autofunction:: graphoskop.metapath2vec

.. autofunction:: graphoskop.node2vec

.. autofunction:: graphoskop.struc2vec

Performance
-----------

On two CPU threads, with planted partition networks whose nodes carry, for each group, a noisy
feature (shifted by half a standard deviation in the node's own group), predicting the group with
two layers (32 channels, then one per group) and 200 epochs, validation accuracy and time:

=============  ===================  ======================
Model          1,000 nodes,         100,000 nodes,
               5,000 edges,         540,000 edges,
               4 groups             10 groups
=============  ===================  ======================
``gcn``        0.985, 1.0 s         0.925, 80 s
``graphsage``  1.0, 0.9 s           0.949, 49 s
``gat``        0.975, 2.1 s         0.937, 251 s
``gatv2``      0.975, 2.7 s         0.939, 512 s
=============  ===================  ======================

The features alone (the same model on the network without edges) reach 0.365 on the small
network. GCN and GraphSAGE layers are products with a normalized adjacency matrix, computed once
as a sparse matrix (3 to 7 times faster than passing the edges to every layer); attention layers
score every edge at every epoch, and are 3 to 10 times slower on CPU. On a GPU, all the training
runs there.

Graph autoencoders, with the defaults (``[32, 16]``, 200 epochs), on two CPU threads. Link
prediction (10% of the edges hidden, as many negatives, dot product of the vectors), without node
attributes (one-hot ids): on 20 planted communities of 100 nodes, AUC 0.805 for
:func:`~graphoskop.gae` and 0.792 for :func:`~graphoskop.vgae` in 14 s (node2vec 0.871, common
neighbors 0.802); on a Holme-Kim network of 3,000 nodes, 0.723 and 0.711 in 5-6 s (node2vec
0.724, common neighbors 0.766). With node attributes that carry a weak signal (8 planted
communities of 250 nodes, one noisy feature per community), the communities are recovered from
the vectors with accuracy 0.760 (:func:`~graphoskop.gae`) and 0.753 (:func:`~graphoskop.vgae`),
against 0.220 from the attributes alone. On 20,000 nodes and 108,000 edges, 50 epochs take 20 s.

Many sparse attributes (e.g. a long node table with one attribute per label, genre and year):
on a network of 105,317 nodes, 1.09 million edges and 87,122 numeric attributes with 2.5 million
nonzero values, the feature matrix is built in 0.8 s and takes 30 MB as a sparse matrix (the dense
one would take 37 GB); 100 epochs of ``gcn`` with ``[64, 2]`` take 128 s on two CPU threads with
1.9 GB of memory in all (validation accuracy 0.93 on a planted signal), and ``graphsage`` 180 s
with 2.1 GB. With the 105,317 dummies of an attribute with one value per node (``qualitative =
True``), the features stay sparse too (192,439 columns).

:func:`~graphoskop.node2vec` with the defaults (128 dimensions, 10 walks of 20 nodes per node, one
epoch), on two CPU threads: 2,000 nodes in 8 planted communities, 1.6 s, communities recovered
from the vectors with accuracy 0.94 (logistic regression, 5-fold cross validation; 0.14 with
random vectors), and link prediction by dot product with AUC 0.87, against 0.80 for
:func:`~graphoskop.common_neighbors_lp` and :func:`~graphoskop.adamic_adar_lp`; 100,000 nodes and
543,000 edges in 134 s, accuracy 0.97. Initial vectors with standard deviation 1, as in PyTorch
Geometric, gave accuracy 0.14 and AUC 0.53 after one epoch (0.17 and 0.57 after three). The walks
were checked against the exact node2vec transition probabilities (weighted or not, directed or
not, several ``p`` and ``q``). Most of the time goes to training; on a GPU, walks and training
both run there.

The spectral embeddings, with 16 dimensions (communities recovered by a logistic regression on
standardized vectors): on 2,000 nodes in 8 communities, 1.1 s each, accuracy 0.988 for both
(node2vec, 128 dimensions: 0.93), link prediction AUC with :func:`~graphoskop.embedding_lp` 0.873
for both (node2vec 0.872; common neighbors 0.80); on 100,000 nodes, 543,000 edges and 10
communities, accuracy 1.000 for both (node2vec 0.965), in 28 s for
:func:`~graphoskop.laplacian_eigenmaps` (70 s with 64 dimensions) and 42 s for
:func:`~graphoskop.locally_linear_embedding` (167 s with 64). Their vectors satisfy their
eigenvalue equations to machine precision (checked against dense generalized eigensolvers).
Without standardization, the tiny entries of the vectors of a large network make a penalized
regression fail (accuracy at chance level on the 100,000 nodes).

:func:`~graphoskop.struc2vec`, 16 dimensions, on the mirrored karate club of the paper (two copies
joined by one edge): the mirror image of a node is its nearest node (standardized Euclidean
distance) for 41% of the nodes, and ranks 2.5 on average among the 67 others (node2vec: 1%, rank
38.9; many karate members are structurally equivalent to others, hence the ties). On a barbell
graph (two cliques of 10 joined by a path of 10), the mirror of every path node is its nearest
node with ``layers = 6`` (0.7 nodes closer with 3; node2vec 9.9). Rings and distances were
checked against breadth first searches and a direct dynamic time warping, and the walks against
the exact probabilities of the multilayer graph. With the defaults (128 dimensions, walks of 80
nodes), on Barabási-Albert networks: 10,000 nodes in 38 s (2.5 s for rings, distances and walks,
the rest training; 9 s with walks of 20 nodes), 100,000 nodes in 525 s, 2.3 GB.

:func:`~graphoskop.metapath2vec`, on a synthetic bibliography (4 research areas, 4,000 authors,
12,000 papers and 40 venues; each paper of an area, its venue of that area with probability 0.6
and each of its 3 authors with probability 0.5, at random otherwise), predicting the area of the
authors from their vectors (logistic regression on standardized vectors, 5-fold, mean of 2
seeds): with the metapath author, paper, venue, paper, author and the defaults, 0.617 in 3.5 s;
with 40 walks per author, as many walks as :func:`~graphoskop.node2vec` makes from all the nodes,
0.688 in 13.6 s (metapath2vec++ 0.677), against 0.648 in 13.3 s for node2vec; author, paper,
author alone 0.424 (coauthors mix areas). On 160,400 nodes and 479,982 edges: 53 s with the
defaults, 221 s with 40 walks per author. The walks were checked to follow the metapath (with and
without relations, directed or not), with uniform choices among the valid neighbors.

:func:`~graphoskop.diffpool`, on graphs of 20 to 40 nodes of two kinds with the same expected degree,
random graphs and graphs of two dense communities (no node attributes, so the feature is the
degree, identical on average in both kinds: only the structure tells them apart): on 410 graphs
(11,813 nodes), 150 epochs take 20 s on two CPU threads, with validation accuracy 0.854 (0.878
with two pooling levels, ``clusters = (0.25, 0.5)``), and the unsupervised embeddings separate the
kinds with accuracy 0.776 (logistic regression, 5-fold); on 4,077 graphs (118,190 nodes, 542,538
edges), 50 epochs take 71 s with 1.9 GB of memory, validation accuracy 0.834. Predicting the mean
of a node attribute (regression): validation :math:`R^2` 0.978. The first level is computed from
sparse products, checked against the dense :math:`\|A - S S^T\|_F^2`. On a single graph of 6,000
nodes and 15,000 edges, with the default 1,500 clusters, 20 epochs take 28 s and 1 GB (pooled with
dense products of the rows of each graph when the products of all the nodes at once would take
more than :math:`2^{26}` numbers).

With constant features (no node attributes but the target), every node starts the same and the
convolutions cannot tell groups apart: graph convolutions need informative features.
