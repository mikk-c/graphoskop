"""
Multilayer networks: single layers and the aggregate network.

A multilayer network (loaded with ``multilayer = True``, see :func:`~graphoskop.load_network`) is
stored as its supra-graph. Each node is a copy of an entity in a layer, with input id
``(entity, layer)``; entities absent from a layer have no copy there. Intra-layer edges join copies
in the same layer; interlayer edges join copies in different layers: couplings when both are
copies of the same entity, otherwise interdependencies between different entities. All of them are
ordinary edges of the components, so the Laplacian of a component is the supra-Laplacian of the
multilayer network (e.g. as in the multilayer generalized Euclidean distance, Coscia 2022), and
every function working on Laplacians, node vectors or paths works on the supra-graph unchanged.
Couplings are explicit edges, with their weight in the edge attributes: the Laplacians read it as
an affinity (the larger, the stronger the coupling); with unit weights (``weight = None``) every
coupling weighs 1.

The functions here turn a multilayer network into ordinary networks: one of its layers, or all the
layers flattened into one network of entities.
"""

import numpy as np
import pandas as pd

from .network import LAYER, _distinct, _load, _release_memory


def _tables(net):
   """
   Tables of a multilayer network, checking that it is multilayer.

   :param net: Network.
   :return: ``(edges, interlayer, nodes)`` as given by :meth:`~graphoskop.Network.to_pandas_dataframes`.
   :raises ValueError: If the network is not multilayer.
   """
   if net.layer_labels is None:
      raise ValueError("the network is not multilayer: load it with multilayer = True")
   return net.to_pandas_dataframes()


def _flat_tables(net):
   """
   Edge and node tables of an ordinary network. The node table lists every node (declaration rows
   only, if there are no node attributes), so that isolated nodes are kept when loading back.

   :param net: Network, not multilayer.
   :return: ``(edges, nodes)``.
   """
   tables = net.to_pandas_dataframes()
   if isinstance(tables, tuple):
      return tables
   ids = np.concatenate(net.node_ids)
   if net.node_format == "wide":
      return tables, pd.DataFrame({"node": ids})
   return tables, pd.DataFrame({"node": ids, "attribute": np.nan, "value": np.nan})


def _rebuild(net, edges, nodes):
   """
   Ordinary network from an edge table (columns ``source``, ``target``, then attributes) and a
   node table of entities, with the settings of a multilayer network.

   :param net: Multilayer network.
   :param edges: Edge table.
   :param nodes: Node table (``node`` column first).
   :return: The :class:`~graphoskop.Network`.
   """
   new = _load(edges, nodes, source = "source", target = "target", directed = net.directed, node_format = net.node_format, bipartite = net.bipartite_labels is not None, device = net.components[0].edge_index.device)
   _release_memory()
   return new


def layer_network(net, layer):
   """
   One layer of a multilayer network, as an ordinary network of entities.

   The nodes are the copies in that layer, with the entity as id, and the edges are the intra-layer
   edges of the layer; couplings and other interlayer edges are left out. All node and edge
   attributes are kept (an attribute classified as qualitative is read back as numeric if all its
   values in the layer look like numbers, as when saving and loading), and so are node types in
   bipartite networks. Components are recomputed, so they are the components of the layer.

   :param net: A multilayer :class:`~graphoskop.Network`.
   :param layer: Layer, one of ``net.layer_labels``.
   :return: The layer, a new :class:`~graphoskop.Network`.
   :raises ValueError: If ``net`` is not multilayer, or the layer has no node.
   :raises KeyError: If there is no such layer.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("multiplex.tsv", multilayer = True, coupling = "all")
      work = gsk.layer_network(net, "work")
      gsk.degree(work, "alice") == gsk.degree(net, ("alice", "work"))   # True
   """
   edges, _, nodes = _tables(net)
   if layer not in net.layer_labels:
      raise KeyError(f"no layer {layer!r}; layers: {net.layer_labels}")
   nodes = nodes[(nodes[LAYER] == layer).to_numpy()].drop(columns = LAYER)
   if len(nodes) == 0:
      raise ValueError(f"layer {layer!r} has no node")
   return _rebuild(net, edges[(edges[LAYER] == layer).to_numpy()].drop(columns = LAYER), nodes)


def aggregate_network(net, merge = True):
   """
   Aggregate (flattened) network of a multilayer network: one node per entity, with the edges of
   all layers.

   This is the limit of infinitely strong couplings: the copies of each entity collapse into one
   node, couplings disappear and the other interlayer edges become ordinary edges between their
   entities (except in bipartite networks, where they would join two nodes of the same type, and
   are left out). With ``merge = True`` the edges between the same two entities (in the same
   direction, if directed), from any layer, are merged into one edge whose numeric attributes are
   the sums of theirs (NaN only if all are NaN) and whose qualitative attributes keep their
   distinct values (one value, or a tuple of them): the weight of an edge is then its total weight
   over the layers, and the degree of an entity its number of distinct neighbors. With
   ``merge = False`` they stay as parallel edges, which give the same Laplacians.

   Node attributes take, for each entity, the first value found in layer order (attributes given
   for the whole entity have the same value in every copy); node types are kept in bipartite
   networks.

   :param net: A multilayer :class:`~graphoskop.Network`.
   :param merge: Merge the edges between the same two entities.
   :return: The aggregate network, a new :class:`~graphoskop.Network`.
   :raises ValueError: If ``net`` is not multilayer.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("multiplex.tsv", multilayer = True, coupling = "all")
      flat = gsk.aggregate_network(net)                    # edge weights summed over the layers
      gsk.degree(flat, "alice")                            # distinct neighbors of alice
   """
   edges, inter, nodes = _tables(net)
   edges = edges.drop(columns = LAYER)
   if net.bipartite_labels is None:
      inter = inter[(inter["source"] != inter["target"]).to_numpy()].drop(columns = ["source_layer", "target_layer"])
      edges = pd.concat([edges, inter], ignore_index = True)
   if merge and len(edges):
      codes, ids = pd.factorize(pd.concat([edges["source"], edges["target"]], ignore_index = True))
      a, b = codes[:len(edges)], codes[len(edges):]
      if not net.directed:
         a, b = np.minimum(a, b), np.maximum(a, b)
      attrs = edges.drop(columns = ["source", "target"])
      g = attrs.groupby([a, b], sort = False)
      num = list(attrs.select_dtypes(["number", "bool"]).columns)
      merged = g[num].sum(min_count = 1)
      for c in attrs.columns:   # qualitative attributes: their distinct values
         if c not in num:
            merged[c] = g[c].agg(_distinct)
      merged = merged[list(attrs.columns)]
      ids = np.asarray(ids)
      merged.insert(0, "target", ids[merged.index.get_level_values(1)])
      merged.insert(0, "source", ids[merged.index.get_level_values(0)])
      edges = merged.reset_index(drop = True)
   return _rebuild(net, edges, _entity_nodes(net, nodes))


def _entity_nodes(net, nodes):
   """
   Node table of entities from a node table of copies: for each entity and attribute, the first
   value found in layer order.

   :param net: Multilayer network.
   :param nodes: Node table of copies, as given by :meth:`~graphoskop.Network.to_pandas_dataframes`.
   :return: Node table of entities (``node`` column first, no ``layer`` column).
   """
   rank = pd.Index(list(net.layer_labels)).get_indexer(nodes[LAYER])
   nodes = nodes.iloc[np.argsort(rank, kind = "stable")].drop(columns = LAYER)
   if net.node_format == "wide":
      return nodes.groupby("node", sort = False).first().reset_index()
   return nodes.drop_duplicates(["node", "attribute"], keep = "first")
