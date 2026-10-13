Multilayer networks
===================

.. automodule:: graphoskop.multilayer
   :no-members:

Representation
--------------

A multilayer network is stored as its supra-graph. A node is a copy of an entity in one layer, with
input id ``(entity, layer)``, and internal id ``(component, index)`` like any other node. Entities
absent from a layer have no copy there. Edges come in three kinds, all stored as ordinary edges:

* intra-layer edges, between copies in the same layer;
* couplings, between copies of the same entity in different layers;
* other interlayer edges, between copies of different entities in different layers.

Components are the weakly connected components of the supra-graph, so a component can hold copies
of many layers, connected through couplings. Within a component, nodes are sorted by layer (by type
first, then by layer, in bipartite networks). Every component has two int64 tensors on its device:
``layer`` (position of the layer of each node in :attr:`~graphoskop.Network.layer_labels`) and
``entity`` (position of its entity in :attr:`~graphoskop.Network.entity_ids`). An edge ``(s, t)``
is interlayer if ``layer[s] != layer[t]`` (see :meth:`~graphoskop.Network.interlayer_mask`) and a
coupling if moreover ``entity[s] == entity[t]``. :attr:`~graphoskop.Network.entity_map` maps every
entity to its copies.

.. autodata:: graphoskop.LAYER
   :annotation: = "layer"

Loading
-------

:func:`~graphoskop.load_network` with ``multilayer = True`` reads an edge table with a ``layer``
column, an optional table of interlayer edges and an optional node table with the entity and the
layer in its first two columns. Couplings are generated with ``coupling``:

* ``"all"``: every pair of copies of each entity (a multiplex); a warning is issued if there are
  more couplings than intra-layer edges, which happens with many layers;
* ``"ordinal"``: copies in adjacent layers only, undirected (stored in both directions in directed
  networks). A copy is not coupled across a layer where its entity has no copy. Adjacency follows
  the layer order, so a layer without any node (e.g. a time step without contacts) must be
  declared in ``layers`` to separate its neighbors;
* ``"temporal"``: from each copy to the copy in the next layer, forward only, so the network must be
  directed (give undirected contacts in both directions). Paths then respect time;
* ``None``: no generated couplings (interlayer edges are then only those given explicitly).

Layers are ordered by their values, or by ``layers``. ``coupling_weight`` is a positive number or
an ``L`` by ``L`` matrix (one value per pair of layers, in layer order), stored in every edge
attribute of the generated couplings.

.. code-block:: python

   import graphoskop as gsk

   # edges.tsv:   source   target   layer     weight
   #              alice    bob      work      2
   #              alice    carol    friends   1
   net = gsk.load_network("edges.tsv", "nodes.tsv", multilayer = True, coupling = "all", coupling_weight = 0.5)
   net.layer_labels                     # ("friends", "work")
   net.node_map[("alice", "work")]      # (component, index)
   net.entity_map["alice"]              # internal ids of both copies
   net.number_of_edges(kind = "inter")  # couplings

   # snapshots of a contact network, couplings forward in time
   net = gsk.load_network("contacts.tsv", multilayer = True, layer = "day", coupling = "temporal", directed = True)

In the node table, a row with a missing layer applies to all the copies of its entity, and the rows
of a copy override it: entity-wide attributes are given once, layer-specific ones per copy.

Analyses
--------

* **Laplacians and node vectors** (:doc:`laplacians`, :doc:`node_vector_analysis`): the Laplacian of
  a component is the supra-Laplacian, so :func:`~graphoskop.generalized_euclidean` is the
  multilayer generalized Euclidean distance (Coscia, 2022), and resistances, solvers, correlations,
  variances and entropies are their multilayer versions. Couplings weigh as affinities. Dense
  matrices have one row per copy.
* **Degree** (:func:`~graphoskop.degree`): intra-layer edges only. A copy has its degree in its
  layer, an entity the sum over its copies; distributions are given per layer, or per entity with
  ``by = "entity"``.
* **Shortest paths** (:doc:`shortest_paths`): on the supra-graph, changing layer along a coupling
  is a step. Entities can be given as origins or destinations (shortest over their copies). Use
  ``inverse = True`` for lengths consistent with coupling affinities.
* **Backboning** (:doc:`backboning`): layer by layer, as on :func:`layer_network`; interlayer edges
  are never scored and are kept between kept copies.
* **Bipartite projections** (:doc:`bipartite_projections`): a bipartite multilayer network is
  bipartite in every layer (an entity has the same type in all its copies, and interlayer edges join
  nodes of the same type); projections are layer by layer, and keep the interlayer edges between
  the copies of the kept type.

Saving
------

:func:`~graphoskop.save_network` writes ``prefix + "_edges.tsv"`` (intra-layer edges, with a
``layer`` column), ``prefix + "_interlayer.tsv"`` (all interlayer edges, couplings included) and
``prefix + "_nodes.tsv"`` (one row per copy). Load them back with ``multilayer = True`` and
``interlayer_edges``; generated couplings already in the interlayer table are not added twice.

Single layers and aggregation
-----------------------------

.. autofunction:: graphoskop.layer_network

.. autofunction:: graphoskop.aggregate_network

Signed networks
---------------

A signed network is a two-layer network, positive and negative, flattened into one network of
entities whose edges carry a sign: build it from two layers with
``gsk.signed_network(net, positive = ..., negative = ...)`` (see :doc:`signed`).
