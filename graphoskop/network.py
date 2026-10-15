"""
Network container, loading of networks from edge list and node attribute tables, conversion
back to tables, and saving to disk.

Node attributes are either numeric or qualitative. Numeric attributes are stored in the
``node_attr`` tensor of each component, on the device of the network. Qualitative attributes
(strings, categories, any non-numeric value) are stored once for the whole network in
:attr:`Network.qualitative_node_attr`, a :class:`pandas.DataFrame` that always stays in CPU
memory, whatever the device.

A network loaded with ``bipartite = True`` has two node types, given by the mandatory node
attribute :data:`BIPARTITE`, and no edge between two nodes of the same type.

A network loaded with ``multilayer = True`` is stored as its supra-graph: every node is a copy of
an entity in one layer, with input id ``(entity, layer)``, and intra-layer edges, couplings
(interlayer edges between copies of the same entity) and other interlayer edges are all ordinary
edges (see :doc:`multilayer`).
"""

import contextlib
import ctypes
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Data

#: Name of the node attribute holding the node type in bipartite networks.
BIPARTITE = "bipartite_type"

#: Default name of the edge table column holding the layer of each edge in multilayer networks.
LAYER = "layer"


class Network:
   """
   A network split into its weakly connected components.

   Nodes are ordered network wide by component, then by index within the component: component
   ``c`` holds the positions ``node_offsets[c]`` to ``node_offsets[c + 1] - 1`` of this order. In
   bipartite networks, the nodes of type 0 come first in every component: node ``i`` of component
   ``c`` has type 0 if ``i < bipartite_split[c]``, type 1 otherwise. The biadjacency matrix of a
   component then has rows ``0`` to ``bipartite_split[c] - 1`` and columns
   ``bipartite_split[c]`` to ``n - 1``, and the attributes of each type are a slice of the node
   tables.

   :ivar components: One :class:`torch_geometric.data.Data` per weakly connected component,
      sorted by number of nodes in descending order (component 0 is the largest).
   :ivar node_map: Maps each input node id to its internal id ``(component, index)``.
   :ivar node_ids: For each component, the array of input node ids ordered by internal index.
   :ivar node_offsets: Position of the first node of each component in the network wide node
      order, followed by the number of nodes.
   :ivar node_attributes: Numeric node attribute names, in order of appearance in the node table;
      position ``j`` is column ``j`` of ``node_attr``.
   :ivar qualitative_node_attributes: Qualitative node attribute names, in order of appearance in
      the node table. A name is never in both lists.
   :ivar qualitative_node_attr: Qualitative node attributes, in CPU memory. In wide format, one
      categorical column per attribute and one row per node, in the network wide node order. In
      long format, columns ``node`` (position in the network wide node order), ``attribute`` and
      ``value`` (both categorical), one row per stored value.
   :ivar node_format: ``"wide"`` or ``"long"``, the format of the node table.
   :ivar edge_attributes: Numeric edge attribute names; position ``j`` is column ``j`` of ``edge_attr``.
   :ivar directed: Whether the edges are directed.
   :ivar bipartite_labels: In bipartite networks, the values of :data:`BIPARTITE` of type 0 and
      type 1, in this order (sorted); ``None`` otherwise.
   :ivar bipartite_split: In bipartite networks, the number of nodes of type 0 of each component;
      ``None`` otherwise.
   :ivar layer_labels: In multilayer networks, the layers in their order (sorted, or as given to
      :func:`load_network`); ``None`` otherwise. Each component then also has the int64 tensors
      ``layer`` (position of the layer of each node in ``layer_labels``) and ``entity`` (position
      of its entity in ``entity_ids``), on its device. Within each component, nodes are sorted by
      layer (by type first, then by layer, in bipartite networks).
   :ivar entity_ids: In multilayer networks, the input ids of the entities; ``None`` otherwise.
   :ivar entity_map: In multilayer networks, maps each entity to the list of the internal ids
      ``(component, index)`` of its copies, in layer order; ``None`` otherwise.

   Example, reading and setting attributes by name:

   .. code-block:: python

      age = net.get_node_attribute("age")             # vector over the nodes of component 0
      city = net.get_node_attribute("city")           # qualitative: NumPy array, on the CPU
      c, i = net.node_map["alice"]
      age_alice = net.get_node_attribute("age", c)[i]
      w = net.get_edge_attribute("weight")            # vector over the edges of component 0
      w_ab = net.get_edge_attribute("weight", edge = ("alice", "bob"))   # one edge, None if absent
      net.set_edge_attribute("log_weight", w.log())   # new attribute, NaN in other components
      net.set_node_attribute("age", [a + 1 for a in net.get_node_attribute("age", None)], None)
      net.set_node_attribute("group", ["a"] * net.number_of_nodes(0))   # new qualitative attribute
      net.number_of_nodes(), net.number_of_edges(), net.number_of_components()
   """

   def __init__(self, components, node_map, node_ids, node_attributes, edge_attributes, directed, qualitative_node_attr, qualitative_node_attributes, node_format = "wide", bipartite_labels = None, bipartite_split = None, layer_labels = None, entity_ids = None):
      self.components = components
      self.node_map = node_map
      self.node_ids = node_ids
      self.node_offsets = np.cumsum([0] + [d.num_nodes for d in components])
      self.node_attributes = node_attributes
      self.qualitative_node_attributes = qualitative_node_attributes
      self.qualitative_node_attr = qualitative_node_attr
      self.node_format = node_format
      self.edge_attributes = edge_attributes
      self.directed = directed
      self.bipartite_labels = bipartite_labels
      self.bipartite_split = bipartite_split
      self.layer_labels = layer_labels
      self.entity_ids = entity_ids
      self.entity_map = None
      if layer_labels is not None:
         sizes = np.diff(self.node_offsets)
         ent = np.concatenate([d.entity.cpu().numpy() for d in components])
         comp = np.repeat(np.arange(len(components)), sizes)
         local = np.arange(len(ent)) - np.repeat(self.node_offsets[:-1], sizes)
         lay = np.concatenate([d.layer.cpu().numpy() for d in components])
         order = np.lexsort((lay, ent))
         cut = np.flatnonzero(ent[order][1:] != ent[order][:-1]) + 1
         self.entity_map = {entity_ids[ent[g[0]]]: list(zip(comp[g].tolist(), local[g].tolist())) for g in np.split(order, cut) if len(g)}

   def _nodes(self, node):
      """
      Internal ids of a node given by its input id, by an entity (multilayer networks: all its
      copies) or by its internal id ``(component, index)``. Ids read from files are strings, so if
      nothing matches, the id is tried again as a string (the entity part only, for
      ``(entity, layer)`` pairs).

      :param node: Input id, entity, or internal id.
      :return: List of internal ids ``(component, index)``.
      :raises KeyError: If there is no such node.
      :raises ValueError: If ``node`` is both an input id (or an entity) and a valid internal id.
      """
      pair = isinstance(node, tuple) and len(node) == 2
      found = None
      for key in (node, (str(node[0]), node[1]) if pair else str(node)):
         if key in self.node_map:
            found = [self.node_map[key]]
         elif self.entity_map is not None and key in self.entity_map:
            found = self.entity_map[key]
         if found is not None:
            break
      if pair and all(isinstance(x, (int, np.integer)) for x in node) and 0 <= node[0] < len(self.components) and 0 <= node[1] < self.components[node[0]].num_nodes:
         if found is not None:
            raise ValueError(f"{node!r} is both a node id and an internal id (component, index): give the node by its id, node_ids[c][i], or the internal id by the id of that node")
         found = [(int(node[0]), int(node[1]))]
      if found is None:
         raise KeyError(f"no node {node!r}")
      return found

   def interlayer_mask(self, component = 0):
      """
      Which stored edges of a component are interlayer edges (their endpoints are in different
      layers): couplings and the other interlayer edges, as opposed to intra-layer edges. Couplings
      are the interlayer edges with ``entity[s] == entity[t]``.

      :param component: Index of the component (0 is the largest).
      :return: Boolean vector aligned with the columns of ``edge_index``, on the device of the
         component; all ``False`` if the network is not multilayer.
      """
      data = self.components[component]
      s, t = data.edge_index
      if self.layer_labels is None:
         return torch.zeros(s.shape[0], dtype = torch.bool, device = s.device)
      return data.layer[s] != data.layer[t]

   @staticmethod
   def _column(names, key, kind):
      """
      Position of an attribute given by name or position.

      :param names: Attribute names.
      :param key: Name or position.
      :param kind: ``"node"`` or ``"edge"``, for error messages.
      :return: Position.
      :raises KeyError: If there is no attribute with that name.
      """
      if not isinstance(key, str):
         return int(key)
      if key not in names:
         raise KeyError(f"no {kind} attribute named {key!r}; {kind} attributes: {names}")
      return names.index(key)

   @staticmethod
   def _values(values, rows, X):
      """
      Values to store in a component, as a vector in the dtype and on the device of its table.

      :param values: Vector (:class:`torch.Tensor`, :class:`numpy.ndarray` or list).
      :param rows: Expected length.
      :param X: ``node_attr`` or ``edge_attr`` of the component.
      :return: Vector.
      :raises ValueError: If the length is not ``rows``.
      """
      v = torch.as_tensor(values, dtype = X.dtype, device = X.device)
      if v.shape != (rows,):
         raise ValueError(f"expected a vector of length {rows}, got shape {tuple(v.shape)}")
      return v

   def _input_edges(self, component):
      """
      Edges of a component as given in input: in undirected networks, the reverse copies added by
      :func:`load_network` (stored after the input edges) are left out.

      :param component: Component index.
      :return: Edge index and edge attributes (views).
      """
      data = self.components[component]
      ei, X = data.edge_index, data.edge_attr
      if self.directed:
         return ei, X
      m = self.number_of_edges(component)
      return ei[:, :m], X[:m]

   def get_node_attribute(self, key, component = 0):
      """
      Node attribute as a vector over the nodes of a component.

      Numeric attributes work for both wide (dense) and long (sparse) node tables; in the latter,
      nodes without a value for the attribute get 0. The vector is a copy, ordered by internal node
      index, on the device of the component. Qualitative attributes are returned as a NumPy object
      array, with NaN for missing values.

      :param key: Attribute name, or position in :attr:`node_attributes`.
      :param component: Index of the component (0 is the largest), or ``None`` for all components.
      :return: Vector of length ``n``, or a list of them if ``component`` is ``None``.
      :raises KeyError: If there is no node attribute with that name.
      """
      if component is None:
         return [self.get_node_attribute(key, c) for c in range(len(self.components))]
      if isinstance(key, str) and key in self.qualitative_node_attributes:
         return self._get_qualitative(key, component)
      j = self._column(self.node_attributes, key, "node")
      data = self.components[component]
      X = data.node_attr
      if X.layout != torch.sparse_csr:
         return X[:, j].clone()
      with _quiet_sparse():
         X = X.to_sparse_coo()
      m = X.indices()[1] == j
      return torch.zeros(data.num_nodes, dtype = X.dtype, device = X.device).index_put_((X.indices()[0][m],), X.values()[m])

   def _get_qualitative(self, key, component):
      """
      Qualitative node attribute over the nodes of a component.

      :param key: Attribute name.
      :param component: Component index.
      :return: NumPy object array, NaN for missing values.
      """
      a, b = self.node_offsets[component], self.node_offsets[component + 1]
      Q = self.qualitative_node_attr
      if self.node_format == "wide":
         return np.asarray(Q[key].iloc[a:b], dtype = object)
      v = np.full(b - a, np.nan, dtype = object)
      r = Q[((Q["attribute"] == key) & (Q["node"] >= a) & (Q["node"] < b)).to_numpy()]
      v[r["node"].to_numpy() - a] = np.asarray(r["value"], dtype = object)
      return v

   def set_node_attribute(self, key, values, component = 0):
      """
      Set a node attribute on the nodes of a component, creating it if it does not exist.

      A new attribute is numeric if its values are (a :class:`torch.Tensor`, or an array or list of
      numbers or booleans), qualitative otherwise. A new numeric attribute is appended to
      :attr:`node_attributes` and to ``node_attr`` of every component; components not set get NaN
      (wide, dense tables) or no value, read as 0 (long, sparse tables). In sparse tables, zero
      values are not stored. Values are cast to the dtype of ``node_attr``. A new qualitative
      attribute is appended to :attr:`qualitative_node_attributes`; components not set get NaN
      (missing).
      A node can have several values of a qualitative attribute, given as a list, tuple or set
      (see :ref:`overlapping-communities`).

      :param key: Attribute name, or position in :attr:`node_attributes`.
      :param values: Vector of length ``n``, ordered by internal node index, or a list of them
         (one per component) if ``component`` is ``None``.
      :param component: Index of the component (0 is the largest), or ``None`` for all components.
      :raises KeyError: If ``key`` is a position and there is no node attribute there.
      :raises ValueError: If a vector does not have one value per node, if non-numeric values are
         given for a numeric attribute, if ``key`` is :data:`BIPARTITE` in a bipartite network
         (node types fix the node order and cannot change), or if ``key`` is ``"layer"`` in a
         multilayer network (the layer is part of the node id).
      """
      if self.bipartite_labels is not None and key == BIPARTITE:
         raise ValueError(f"{BIPARTITE!r} cannot be changed: node types fix the node order of a bipartite network")
      if self.layer_labels is not None and key == LAYER:
         raise ValueError(f"{LAYER!r} cannot be a node attribute of a multilayer network: the layer is part of the node id")
      first = values[0] if component is None and len(values) > 0 else values
      new = isinstance(key, str) and key not in self.node_attributes and key not in self.qualitative_node_attributes
      if new and not _is_numeric(first):
         self.qualitative_node_attributes.append(key)
         if self.node_format == "wide":
            self.qualitative_node_attr[key] = pd.Categorical(np.full(self.node_offsets[-1], np.nan, dtype = object))
      if isinstance(key, str) and key in self.qualitative_node_attributes:
         self._set_qualitative(key, values, component)
         return
      if not _is_numeric(first):
         raise ValueError(f"node attribute {key!r} is numeric, got non-numeric values")
      if new:
         self.node_attributes.append(key)
         for data in self.components:
            X = data.node_attr
            if X.layout != torch.sparse_csr:
               data.node_attr = torch.cat([X, X.new_full((X.shape[0], 1), float("nan"))], 1)
            else:
               with _quiet_sparse():
                  data.node_attr = torch.sparse_csr_tensor(X.crow_indices(), X.col_indices(), X.values(), size = (X.shape[0], X.shape[1] + 1))
      j = self._column(self.node_attributes, key, "node")
      if j >= len(self.node_attributes):
         raise KeyError(f"no node attribute at position {j}; node attributes: {self.node_attributes}")
      if component is None:
         for c, v in enumerate(values):
            self.set_node_attribute(j, v, c)
         return
      data = self.components[component]
      X = data.node_attr
      v = self._values(values, data.num_nodes, X)
      if X.layout != torch.sparse_csr:
         X[:, j] = v
         return
      with _quiet_sparse():
         C = X.to_sparse_coo()
         keep = C.indices()[1] != j
         nz = v.nonzero().squeeze(1)
         i = torch.cat([C.indices()[:, keep], torch.stack([nz, torch.full_like(nz, j)])], 1)
         data.node_attr = torch.sparse_coo_tensor(i, torch.cat([C.values()[keep], v[nz]]), X.shape).coalesce().to_sparse_csr()

   def _set_qualitative(self, key, values, component):
      """
      Set an existing qualitative node attribute on the nodes of a component.

      :param key: Attribute name.
      :param values: Vector of length ``n``, or a list of them if ``component`` is ``None``.
      :param component: Component index, or ``None`` for all components.
      :raises ValueError: If a vector does not have one value per node.
      """
      vectors = values if component is None else [values]
      comps = range(len(self.components)) if component is None else [component]
      if len(vectors) != len(comps):
         raise ValueError(f"expected one vector per component ({len(comps)}), got {len(vectors)}")
      parts = []
      for c, x in zip(comps, vectors):   # all components at once: one update of the table
         x = _memberships(x.cpu().numpy() if isinstance(x, torch.Tensor) else x)
         if x.shape != (self.components[c].num_nodes,):
            raise ValueError(f"expected a vector of length {self.components[c].num_nodes}, got shape {x.shape}")
         parts.append(x)
      a, b = self.node_offsets[comps[0]], self.node_offsets[comps[-1] + 1]
      v = np.concatenate(parts)
      Q = self.qualitative_node_attr
      if self.node_format == "wide":
         col = np.asarray(Q[key], dtype = object)
         col[a:b] = v
         Q[key] = pd.Categorical(col)
         return
      drop = ((Q["attribute"] == key) & (Q["node"] >= a) & (Q["node"] < b)).to_numpy()
      ok = ~pd.isna(v)
      new = pd.DataFrame({"node": np.arange(a, b)[ok], "attribute": key, "value": v[ok]})
      self.qualitative_node_attr = _categorize(pd.concat([Q[~drop], new], ignore_index = True))

   def get_edge_attribute(self, key, component = 0, edge = None):
      """
      Edge attribute as a vector over the edges of a component, or its value on one edge.

      The vector is a copy, aligned with the columns of ``edge_index``, on the device of the
      component. In undirected networks every edge is stored in both directions, so each value
      appears twice (self loops once), as in ``edge_attr``.

      With ``edge = (u, v)``, given by input node ids, the value on the edge between ``u`` and
      ``v`` is returned instead, as a Python number, and ``component`` is ignored. In directed
      networks the edge goes from ``u`` to ``v``; in undirected networks the order does not matter.
      If the edge is not in the network (including when a node is not), ``None`` is returned. If
      there are parallel edges, the list of their values is returned, in input order.

      :param key: Attribute name, or position in :attr:`edge_attributes`.
      :param component: Index of the component (0 is the largest), or ``None`` for all components.
      :param edge: Pair of input node ids ``(u, v)``, or ``None``.
      :return: Vector with one entry per stored edge, or a list of them if ``component`` is
         ``None``; with ``edge``, a number, a list of numbers or ``None``.
      :raises KeyError: If there is no edge attribute with that name.
      """
      j = self._column(self.edge_attributes, key, "edge")
      if edge is not None:
         u, v = (self.node_map.get(x) for x in edge)
         if u is None or v is None or u[0] != v[0]:
            return None
         ei, X = self._input_edges(u[0])
         m = (ei[0] == u[1]) & (ei[1] == v[1])
         if not self.directed:
            m |= (ei[0] == v[1]) & (ei[1] == u[1])
         w = X[m, j].tolist()
         return None if not w else w[0] if len(w) == 1 else w
      if component is None:
         return [self.get_edge_attribute(j, c) for c in range(len(self.components))]
      return self.components[component].edge_attr[:, j].clone()

   def set_edge_attribute(self, key, values, component = 0):
      """
      Set an edge attribute on the edges of a component, creating it if it does not exist.

      A new attribute is appended to :attr:`edge_attributes` and to ``edge_attr`` of every
      component; components not set get NaN. Values are aligned with the columns of
      ``edge_index``, as returned by :meth:`get_edge_attribute`: in undirected networks both
      directions of an edge are stored and should get the same value. Values are cast to the dtype
      of ``edge_attr``.

      :param key: Attribute name, or position in :attr:`edge_attributes`.
      :param values: Vector with one entry per stored edge, or a list of them (one per component)
         if ``component`` is ``None``.
      :param component: Index of the component (0 is the largest), or ``None`` for all components.
      :raises KeyError: If ``key`` is a position and there is no edge attribute there.
      :raises ValueError: If a vector does not have one value per stored edge.
      """
      if isinstance(key, str) and key not in self.edge_attributes:
         self.edge_attributes.append(key)
         for data in self.components:
            X = data.edge_attr
            data.edge_attr = torch.cat([X, X.new_full((X.shape[0], 1), float("nan"))], 1)
      j = self._column(self.edge_attributes, key, "edge")
      if j >= len(self.edge_attributes):
         raise KeyError(f"no edge attribute at position {j}; edge attributes: {self.edge_attributes}")
      if component is None:
         for c, v in enumerate(values):
            self.set_edge_attribute(j, v, c)
         return
      X = self.components[component].edge_attr
      X[:, j] = self._values(values, X.shape[0], X)

   def number_of_nodes(self, component = None):
      """
      Number of nodes, isolated nodes included.

      :param component: Index of the component (0 is the largest), or ``None`` for all components.
      :return: Number of nodes.
      """
      if component is None:
         return sum(d.num_nodes for d in self.components)
      return self.components[component].num_nodes

   def number_of_edges(self, component = None, kind = "all"):
      """
      Number of edges, as given in input: in undirected networks each edge counts once, although it
      is stored in both directions. Parallel edges and self loops count.

      :param component: Index of the component (0 is the largest), or ``None`` for all components.
      :param kind: In multilayer networks, ``"all"`` edges, ``"intra"`` (intra-layer edges only)
         or ``"inter"`` (interlayer edges only: couplings and the others).
      :return: Number of edges.
      :raises ValueError: If ``kind`` is not valid.
      """
      if kind not in ("all", "intra", "inter"):
         raise ValueError(f"unknown kind of edges: {kind}")
      if component is None:
         return sum(self.number_of_edges(c, kind) for c in range(len(self.components)))
      ei = self.components[component].edge_index
      m = ei.shape[1] if self.directed else (ei.shape[1] + int((ei[0] == ei[1]).sum())) // 2
      if kind == "all":
         return m
      inter = int(self.interlayer_mask(component).sum()) // (1 if self.directed else 2)
      return inter if kind == "inter" else m - inter

   def number_of_components(self):
      """
      Number of weakly connected components, isolated nodes included (each is a component).

      :return: Number of components.
      """
      return len(self.components)

   def is_planar(self, component = None):
      """
      Whether the network is planar: whether it can be drawn on the plane without edge crossings.

      Left-Right planarity test (de Fraysseix and Rosenstiehl, 1982; Brandes, 2009), linear in the
      size of each component, in plain Python on the CPU (about 10 seconds for 600,000 edges).
      Edge directions, self loops, parallel edges and weights do not matter. A network is planar
      if all its components are. In multilayer networks the test is on the supra-graph,
      interlayer edges included (a layer alone is tested on :func:`~graphoskop.layer_network`).

      :param component: Index of the component (0 is the largest), or ``None`` for the whole
         network.
      :return: ``True`` if planar.

      Example:

      .. code-block:: python

         import graphoskop as gsk

         net = gsk.load_network("edges.tsv")
         net.is_planar()
         net.is_planar(component = 0)   # largest component only
      """
      from .planarity import _planar_edges
      for c in range(len(self.components)) if component is None else [component]:
         s, t = self.components[c].edge_index.cpu().numpy()
         if not _planar_edges(s, t):
            return False
      return True

   def to_pandas_dataframes(self):
      """
      Edges and nodes of all components, with their attributes, as :class:`pandas.DataFrame`.

      The edge table has columns ``source`` and ``target`` (input node ids), then one column per
      edge attribute. Edges are grouped by component, in input order within each component. In
      undirected networks each edge appears once, as in the input (the reverse copies stored by
      :func:`load_network` are dropped).

      The node table is returned only if there are node attributes (always in bipartite networks,
      which have :data:`BIPARTITE`). Nodes are in the network wide node order. Wide node tables
      give a ``node`` column, then one column per numeric attribute, then one (categorical) column
      per qualitative attribute, with a row for every node and NaN for missing values. Long node
      tables give columns ``node``, ``attribute`` and ``value``, with a row for every stored value
      (numeric values first). Both tables can be passed back to :func:`load_network`, with the same
      ``node_format`` and ``bipartite``. Nodes without edges are kept only in the node table, so
      they are lost if there are no node attributes.

      Multilayer networks always give three tables, ``(edges, interlayer, nodes)``, which load back
      with ``multilayer = True`` and ``interlayer_edges = interlayer`` (and no ``coupling``: the
      couplings are in ``interlayer``):

      * ``edges``: intra-layer edges, with columns ``source``, ``target`` (entities), ``layer``,
        then the edge attributes;
      * ``interlayer``: interlayer edges (couplings included), with columns ``source``,
        ``source_layer``, ``target``, ``target_layer``, then the edge attributes;
      * ``nodes``: one node per copy, with columns ``node`` (entity) and ``layer``, then the
        attributes as above. In long format, a copy without any value gets a row with a missing
        attribute and value, so that copies without edges are not lost.

      :return: The edge table, or ``(edges, nodes)`` if there are node attributes, or
         ``(edges, interlayer, nodes)`` for multilayer networks.
      :raises ValueError: If an attribute is named like one of the id columns.

      Example:

      .. code-block:: python

         import graphoskop as gsk

         net = gsk.load_network("edges.tsv", "nodes.tsv")
         gsk.noise_corrected(net)
         edges, nodes = net.to_pandas_dataframes()
         edges.to_csv("scored_edges.tsv", sep = "\\t", index = False)
      """
      ml = self.layer_labels is not None
      src, tgt, attr, ls, lt = [], [], [], [], []
      for c, data in enumerate(self.components):
         ei, X = self._input_edges(c)
         ei = ei.cpu().numpy()
         ids = self.entity_ids[data.entity.cpu().numpy()] if ml else self.node_ids[c]
         src.append(ids[ei[0]])
         tgt.append(ids[ei[1]])
         attr.append(X.cpu().numpy())
         if ml:
            lay = data.layer.cpu().numpy()
            ls.append(lay[ei[0]])
            lt.append(lay[ei[1]])
      edges = pd.DataFrame(np.concatenate(attr), columns = self.edge_attributes)
      edges.insert(0, "target", np.concatenate(tgt))
      edges.insert(0, "source", np.concatenate(src))
      if ml:
         ls, lt = np.concatenate(ls), np.concatenate(lt)
         labels = _label_array(self.layer_labels)
         inter = ls != lt
         interlayer = edges[inter].reset_index(drop = True)
         interlayer.insert(1, "source_layer", labels[ls[inter]])
         interlayer.insert(3, "target_layer", labels[lt[inter]])
         edges = edges[~inter].reset_index(drop = True)
         edges.insert(2, LAYER, labels[ls[~inter]])
      elif not self.node_attributes and not self.qualitative_node_attributes:
         return edges
      idc = _id_columns(self)
      Q = self.qualitative_node_attr
      if self.node_format == "wide":
         nodes = pd.DataFrame(np.concatenate([d.node_attr.cpu().numpy() for d in self.components]), columns = self.node_attributes)
         for a in self.qualitative_node_attributes:
            nodes[a] = Q[a].to_numpy()
         for k in reversed(list(idc)):
            nodes.insert(0, k, idc[k])
         return (edges, interlayer, nodes) if ml else (edges, nodes)
      names = np.array(self.node_attributes, dtype = object)
      pos, attribute, value = [], [], []
      for data, off in zip(self.components, self.node_offsets):
         with _quiet_sparse():
            X = data.node_attr.to_sparse_coo()
         r, c = X.indices().cpu().numpy()
         pos.append(r + off)
         attribute.append(names[c])
         value.append(X.values().cpu().numpy().astype(object))
      cols, vals = _explode({"node": Q["node"].to_numpy(), "attribute": np.asarray(Q["attribute"], dtype = object)}, np.asarray(Q["value"], dtype = object))   # one row per community of a node
      pos.append(cols["node"])
      attribute.append(cols["attribute"])
      value.append(vals)
      if ml:   # copies without values, so that they are not lost
         bare = np.ones(int(self.node_offsets[-1]), dtype = bool)
         for p in pos:
            bare[p] = False
         bare = np.flatnonzero(bare)
         pos.append(bare)
         attribute.append(np.full(len(bare), np.nan, dtype = object))
         value.append(np.full(len(bare), np.nan, dtype = object))
      pos = np.concatenate(pos)
      nodes = pd.DataFrame({**{k: v[pos] for k, v in idc.items()}, "attribute": np.concatenate(attribute), "value": np.concatenate(value)})
      if not self.qualitative_node_attributes:
         nodes["value"] = nodes["value"].astype(np.float32)
      return (edges, interlayer, nodes) if ml else (edges, nodes)


@contextlib.contextmanager
def _quiet_sparse():
   """
   Context for creating sparse CSR tensors without warnings.

   Explicitly disables sparse invariant checks (the library only builds valid tensors, from
   canonical scipy CSR matrices or coalesced COO tensors) and silences the beta state warning.
   The global setting is used because some torch versions warn even when
   ``check_invariants = False`` is passed.
   """
   with warnings.catch_warnings(), torch.sparse.check_sparse_tensor_invariants(enable = False):
      warnings.filterwarnings("ignore", message = "Sparse CSR tensor support is in beta state")
      yield


def _is_numeric(values):
   """
   Whether values can be stored as a numeric attribute.

   :param values: Vector (:class:`torch.Tensor`, :class:`numpy.ndarray` or list).
   :return: ``True`` for tensors and for arrays of numbers or booleans.
   """
   if isinstance(values, torch.Tensor):
      return True
   try:
      a = np.asarray(values)
   except ValueError:   # ragged nested values, e.g. lists of communities
      return False
   return a.ndim == 1 and a.dtype.kind in "biuf"


def _membership(v):
   """
   A qualitative value, with a collection of values (list, tuple, set, array: e.g. the
   communities of a node in several) as a tuple of its distinct values, in order (sorted for
   sets), and a collection of one value as that value. Strings are single values.

   :param v: Value.
   :return: The value, a tuple of two or more values, or NaN for an empty collection.
   """
   if isinstance(v, (str, bytes)) or not isinstance(v, (list, tuple, set, frozenset, np.ndarray, pd.Series)):
      return v
   items = list(v)
   if isinstance(v, (set, frozenset)):
      try:
         items = sorted(items)
      except TypeError:
         pass
   items = list(dict.fromkeys(x.item() if isinstance(x, np.generic) else x for x in items if not (x is None or (isinstance(x, float) and np.isnan(x)))))
   return np.nan if not items else items[0] if len(items) == 1 else tuple(items)


def _memberships(values):
   """
   Qualitative values as a one dimensional object array, collections normalized by
   :func:`_membership` (only scanned when the values are not all strings or numbers).

   :param values: Sequence of values.
   :return: :class:`numpy.ndarray` of objects.
   """
   v = pd.Series(list(values), dtype = object)
   if pd.api.types.infer_dtype(v, skipna = True) in ("string", "integer", "floating", "mixed-integer-float", "boolean", "empty"):
      return v.to_numpy()
   try:   # each distinct value once
      code, uniq = pd.factorize(v)
   except TypeError:   # lists, sets: not hashable
      return v.map(_membership).to_numpy()
   out = np.empty(len(uniq) + 1, dtype = object)
   out[:-1] = [_membership(x) for x in uniq]
   out[-1] = np.nan
   return out[code]


def _explode(columns, value):
   """
   Rows of a long table with one row per value of every collection (tuple) of values.

   :param columns: Dictionary of the other columns, arrays aligned with ``value``.
   :param value: Object array of values, some of them tuples.
   :return: The columns and the values, repeated and flattened.
   """
   size = np.fromiter((len(x) if isinstance(x, tuple) else 1 for x in value), dtype = np.int64, count = len(value))
   if (size == 1).all():
      return columns, value
   flat = np.empty(int(size.sum()), dtype = object)
   flat[:] = [y for x in value for y in (x if isinstance(x, tuple) else (x,))]
   return {k: np.repeat(np.asarray(c), size) for k, c in columns.items()}, flat


def _merge_values(Q):
   """
   Long qualitative rows with the several values of a node and attribute merged into a tuple
   (a node in several communities).

   :param Q: Table with columns ``node``, ``attribute`` and ``value``.
   :return: Table with one row per node and attribute.
   """
   dup = Q.duplicated(["node", "attribute"], keep = False).to_numpy()
   if not dup.any():
      return Q
   merged = Q[dup].groupby(["node", "attribute"], sort = False)["value"].agg(lambda x: _membership(tuple(x))).reset_index()
   return pd.concat([Q[~dup], merged], ignore_index = True)


def _categorize(Q):
   """
   Long qualitative table with categorical ``attribute`` and ``value`` columns, sorted by node.

   :param Q: Table with columns ``node``, ``attribute`` and ``value``.
   :return: The table.
   """
   Q = Q.sort_values("node", kind = "stable", ignore_index = True)
   Q["attribute"] = pd.Categorical(np.asarray(Q["attribute"], dtype = object))
   Q["value"] = pd.Categorical(np.asarray(Q["value"], dtype = object))
   return Q


def _empty_qualitative(n, node_format):
   """
   Qualitative node table without attributes.

   :param n: Number of nodes.
   :param node_format: ``"wide"`` or ``"long"``.
   :return: The table.
   """
   if node_format == "wide":
      return pd.DataFrame(index = pd.RangeIndex(n))
   return _categorize(pd.DataFrame({"node": np.empty(0, dtype = np.int64), "attribute": [], "value": []}))


def _label_array(labels):
   """
   Layer labels as an array that can be indexed by layer positions.

   :param labels: Layer labels.
   :return: :class:`numpy.ndarray` (numeric if all labels are numbers, object otherwise).
   """
   return pd.Index(list(labels)).to_numpy()


def _node_codes(net):
   """
   Layer and entity of every node of a multilayer network.

   :param net: Network.
   :return: Two int64 :class:`numpy.ndarray` in the network wide node order (positions in
      ``net.layer_labels`` and ``net.entity_ids``), or ``None`` if the network is not multilayer.
   """
   if net.layer_labels is None:
      return None
   return tuple(np.concatenate([getattr(d, a).cpu().numpy() for d in net.components]) for a in ("layer", "entity"))


def _id_columns(net, rows = None):
   """
   Id columns of the node table: ``node`` and, in multilayer networks, ``layer`` (with ``node``
   the entity).

   :param net: Network.
   :param rows: Positions of the nodes in the network wide node order, or ``None`` for all.
   :return: Dictionary from column name to :class:`numpy.ndarray`.
   """
   if net.layer_labels is None:
      ids = np.concatenate(net.node_ids)
      return {"node": ids if rows is None else ids[rows]}
   lay, ent = _node_codes(net)
   if rows is not None:
      lay, ent = lay[rows], ent[rows]
   return {"node": net.entity_ids[ent], LAYER: _label_array(net.layer_labels)[lay]}


def _column(columns, col):
   """
   Resolve a column given either by name or by position.

   :param columns: Column names.
   :param col: Column name or numeric position.
   :return: Column name.
   """
   return col if col in columns else columns[col]


def _read(table, sep, header, text):
   """
   Read a table, parsing some columns as strings and the others as numbers where possible.

   Columns whose non-missing values are all numeric are parsed directly as numbers, which needs a
   few bytes per value instead of a Python string object per value. A
   :class:`pandas.DataFrame` is used as it is, without copying it.

   :param table: File path or :class:`pandas.DataFrame`.
   :param sep: Column separator.
   :param header: Header row (0) or ``None``.
   :param text: Names or positions of the columns to read as strings.
   :return: Table, and the names of the ``text`` columns.
   """
   if isinstance(table, pd.DataFrame):
      return table, [_column(table.columns, c) for c in text]
   columns = pd.read_csv(table, sep = sep, header = header, nrows = 0).columns
   names = [_column(columns, c) for c in text]
   return pd.read_csv(table, sep = sep, header = header, dtype = {c: str for c in names}), names


def _numeric(df):
   """
   Keep the columns of a table whose non-missing values are all numeric or boolean.

   :param df: Table.
   :return: Table of the numeric columns.
   """
   return df.select_dtypes(["number", "bool"])


def _positions(perm):
   """
   Inverse of a permutation.

   :param perm: Node order (integer ids in component order).
   :return: Position of each integer id in that order.
   """
   row = np.empty(len(perm), dtype = np.int64)
   row[perm] = np.arange(len(perm))
   return row


def _block(x, start, size, device):
   """
   Extract the rows of a component from a node attribute matrix.

   :param x: Dense :class:`numpy.ndarray` or canonical :class:`scipy.sparse.csr_matrix`.
   :param start: First row of the component.
   :param size: Number of rows of the component.
   :param device: Device of the returned tensor.
   :return: Dense tensor, or sparse CSR tensor if ``x`` is sparse.
   """
   b = x[start:start + size]
   if not sp.issparse(b):
      return torch.tensor(b, device = device)
   with _quiet_sparse():
      return torch.sparse_csr_tensor(torch.tensor(b.indptr, dtype = torch.long), torch.tensor(b.indices, dtype = torch.long), torch.tensor(b.data), size = b.shape, device = device)


def _release_memory():
   """
   Return memory freed after loading to the operating system (glibc only).

   Parsing large tables leaves freed memory in the allocator's heap, where later large allocations
   (such as a dense result matrix) cannot use it; trimming the heap makes it available again.
   """
   try:
      ctypes.CDLL("libc.so.6").malloc_trim(0)
   except (OSError, AttributeError):
      pass


def _build(s, t, names, ea, edge_attributes, node_table, node_attributes, directed, device, qualitative_table = None, qualitative_attributes = (), node_format = "wide", types = None, labels = None, ml = None):
   """
   Network from edges between integer node ids: split into weakly connected components, sorted
   by size, with nodes ordered by component (then by type in bipartite networks, then by layer in
   multilayer networks) and edges grouped by component in input order.

   :param s: Source of each edge, integer ids from 0 to ``len(names) - 1``.
   :param t: Target of each edge.
   :param names: Node ids, by integer id (:class:`numpy.ndarray`).
   :param ea: Edge attributes, ``m`` by ``k`` float32 :class:`numpy.ndarray`.
   :param edge_attributes: Edge attribute names.
   :param node_table: Function taking the node order (integer ids in network wide order) and
      returning the numeric node attribute matrix with rows in that order (dense
      :class:`numpy.ndarray` or canonical :class:`scipy.sparse.csr_matrix`), or ``None`` if there
      are no node attributes.
   :param node_attributes: Numeric node attribute names.
   :param directed: Whether the edges are directed; if not, reverse copies are added.
   :param device: Device of the tensors.
   :param qualitative_table: Function taking the node order and returning the qualitative node
      table (see :attr:`Network.qualitative_node_attr`) in that order, or ``None``.
   :param qualitative_attributes: Qualitative node attribute names.
   :param node_format: ``"wide"`` or ``"long"``.
   :param types: Node type (0 or 1) of each integer id in bipartite networks, ``None`` otherwise.
   :param labels: Values of :data:`BIPARTITE` of type 0 and type 1, in bipartite networks.
   :param ml: In multilayer networks, ``(layer, entity, layer_labels, entity_ids)``: layer and
      entity (positions in ``layer_labels`` and ``entity_ids``) of each integer id; ``None``
      otherwise.
   :return: The :class:`Network`.
   """
   n = len(names)

   # components sorted by size, nodes ordered by component (then by type)
   adj = sp.coo_matrix((np.ones(len(s)), (s, t)), shape = (n, n))
   labels_cc = connected_components(adj, directed = True, connection = "weak")[1]
   del adj
   sizes = np.bincount(labels_cc)
   rank = np.empty_like(sizes)
   rank[np.argsort(-sizes, kind = "stable")] = np.arange(len(sizes))
   comp = rank[labels_cc]
   counts = np.bincount(comp)
   starts = np.cumsum(counts) - counts
   perm = np.lexsort(tuple(k for k in (None if ml is None else ml[0], types, comp) if k is not None))
   local = np.empty(n, dtype = np.int64)
   local[perm] = np.arange(n) - starts[comp[perm]]
   x = np.empty((n, 0), dtype = np.float32) if node_table is None else node_table(perm)
   q = _empty_qualitative(n, node_format) if qualitative_table is None else qualitative_table(perm)

   # edges grouped by component, input order kept
   ecomp = comp[s]
   eperm = np.argsort(ecomp, kind = "stable")
   ecounts = np.bincount(ecomp, minlength = len(counts))
   del ecomp
   estarts = np.cumsum(ecounts) - ecounts

   components = []
   for c in range(len(counts)):
      e = eperm[estarts[c]:estarts[c] + ecounts[c]]
      src, dst = local[s[e]], local[t[e]]
      keep = np.zeros(0, dtype = np.int64) if directed else np.flatnonzero(src != dst)
      n = len(e)
      ei = np.empty((2, n + len(keep)), dtype = np.int64)   # input edges, then reverses (no self loops)
      ei[0, :n], ei[1, :n] = src, dst
      np.take(dst, keep, out = ei[0, n:])
      np.take(src, keep, out = ei[1, n:])
      del src, dst
      w = ea[e] if directed else np.concatenate([ea[e], ea[e[keep]]])
      data = Data(edge_index = torch.from_numpy(ei).to(device), edge_attr = torch.from_numpy(w).to(device), node_attr = _block(x, starts[c], counts[c], device), num_nodes = int(counts[c]))
      if ml is not None:
         block = perm[starts[c]:starts[c] + counts[c]]
         data.layer = torch.as_tensor(ml[0][block], dtype = torch.long, device = device)
         data.entity = torch.as_tensor(ml[1][block], dtype = torch.long, device = device)
      components.append(data)

   node_map = dict(zip(names.tolist(), zip(comp.tolist(), local.tolist())))
   node_ids = np.split(names[perm], np.cumsum(counts)[:-1])
   split = None if types is None else np.bincount(comp[types == 0], minlength = len(counts)).tolist()
   return Network(components, node_map, node_ids, list(node_attributes), list(edge_attributes), directed, q, list(qualitative_attributes), node_format, labels, split, *((None, None) if ml is None else ml[2:]))


def _node_table(net, rows):
   """
   Numeric node attribute matrix of all components, restricted to some nodes.

   :param net: Network.
   :param rows: Positions of the nodes to keep, in the network wide node order.
   :return: Dense :class:`numpy.ndarray`, or canonical :class:`scipy.sparse.csr_matrix` if the node
      table is sparse.
   """
   X = [d.node_attr for d in net.components]
   if X[0].layout != torch.sparse_csr:
      return np.concatenate([x.cpu().numpy() for x in X])[rows]
   with _quiet_sparse():
      blocks = [sp.csr_matrix((x.values().cpu().numpy(), x.col_indices().cpu().numpy(), x.crow_indices().cpu().numpy()), shape = x.shape) for x in X]
   x = sp.vstack(blocks, format = "csr")[rows]
   x.sum_duplicates()
   return x


def _qualitative_rows(net, rows):
   """
   Qualitative node table, restricted to some nodes and reordered.

   :param net: Network.
   :param rows: Positions of the nodes to keep, in the network wide node order, in their new order.
   :return: Table in the format of :attr:`Network.qualitative_node_attr`.
   """
   Q = net.qualitative_node_attr
   if net.node_format == "wide":
      return Q.iloc[rows].reset_index(drop = True)
   inv = np.full(net.node_offsets[-1], -1, dtype = np.int64)
   inv[rows] = np.arange(len(rows))
   new = inv[Q["node"].to_numpy()]
   Q = Q[new >= 0].copy()
   Q["node"] = new[new >= 0]
   return _categorize(Q)


def _node_types(net):
   """
   Node type of every node in bipartite networks.

   :param net: Network.
   :return: int8 :class:`numpy.ndarray` (0 or 1) in the network wide node order, or ``None`` if
      the network is not bipartite.
   """
   if net.bipartite_labels is None:
      return None
   sizes = np.diff(net.node_offsets)
   local = np.arange(net.node_offsets[-1]) - np.repeat(net.node_offsets[:-1], sizes)
   return (local >= np.repeat(net.bipartite_split, sizes)).astype(np.int8)


def _subset(net, s, t, ea, rows):
   """
   New network with some of the nodes of a network and edges between them, all node attributes,
   node types and layers kept, components recomputed.

   :param net: Network.
   :param s: Source of each edge, as a position in ``rows``.
   :param t: Target of each edge, as a position in ``rows``.
   :param ea: Edge attributes, ``m`` by ``k`` float32 :class:`numpy.ndarray`.
   :param rows: Positions of the nodes to keep, in the network wide node order.
   :return: The new :class:`Network`.
   """
   names = np.concatenate(net.node_ids)[rows]
   types = _node_types(net)
   codes = _node_codes(net)
   ml = None if codes is None else (codes[0][rows], codes[1][rows], net.layer_labels, net.entity_ids)
   device = net.components[0].edge_index.device
   return _build(s, t, names, ea, net.edge_attributes, lambda perm: _node_table(net, rows[perm]), net.node_attributes, net.directed, device, lambda perm: _qualitative_rows(net, rows[perm]), net.qualitative_node_attributes, net.node_format, None if types is None else types[rows], net.bipartite_labels, ml)


def _bipartite_types(key, val, names, s, t, inter = None, entity = None):
   """
   Node types of a bipartite network, checked against its edges.

   :param key: Integer id of each node table entry of :data:`BIPARTITE`.
   :param val: Value of each entry.
   :param names: Node ids, by integer id.
   :param s: Source of each edge, integer ids.
   :param t: Target of each edge, integer ids.
   :param inter: In multilayer networks, mask of the interlayer edges, which must join nodes of
      the same type (the other edges must join nodes of different types); ``None`` otherwise.
   :param entity: In multilayer networks, entity of each integer id: all the copies of an entity
      must have the same type.
   :return: Type (0 or 1) of each integer id, as int8 :class:`numpy.ndarray`, and the values of
      type 0 and type 1.
   :raises ValueError: If there are more than two values, if a node has no type or more than one,
      if an edge joins two nodes of the same type, or, in multilayer networks, if an interlayer
      edge joins nodes of different types or the copies of an entity have different types.
   """
   val = np.asarray(val, dtype = object)
   ok = ~pd.isna(val)
   key, val = key[ok], val[ok]
   try:
      labels = sorted(pd.unique(val))
   except TypeError:
      raise ValueError(f"the values of {BIPARTITE!r} cannot be ordered: {list(pd.unique(val))[:5]}") from None
   if len(labels) > 2:
      raise ValueError(f"{BIPARTITE!r} has {len(labels)} distinct values, at most 2 allowed: {labels[:5]}")
   code = pd.Index(labels).get_indexer(val).astype(np.int8)
   types = np.full(len(names), -1, dtype = np.int8)
   types[key] = code
   if (types[key] != code).any():
      raise ValueError(f"node {names[key[types[key] != code][0]]!r} has two different values of {BIPARTITE!r}: ids must be unique across node types")
   if (types < 0).any():
      raise ValueError(f"node {names[np.argmax(types < 0)]!r} has no value of {BIPARTITE!r}")
   same = types[s] == types[t]
   if inter is not None:
      if (inter & ~same).any():
         e = np.argmax(inter & ~same)
         raise ValueError(f"interlayer edge {names[s[e]]!r} - {names[t[e]]!r} joins two nodes with different {BIPARTITE!r}: interlayer edges of bipartite networks must join nodes of the same type")
      same &= ~inter
      lo = np.full(entity.max() + 1, 2, dtype = np.int8)
      hi = np.full(entity.max() + 1, -1, dtype = np.int8)
      np.minimum.at(lo, entity, types)
      np.maximum.at(hi, entity, types)
      if (lo[entity] != hi[entity]).any():
         raise ValueError(f"node {names[np.argmax(lo[entity] != hi[entity])]!r} has a different {BIPARTITE!r} than other copies of its entity: an entity has the same type in all layers")
   if same.any():
      e = np.argmax(same)
      raise ValueError(f"edge {names[s[e]]!r} - {names[t[e]]!r} joins two nodes with the same {BIPARTITE!r}: the network is not bipartite")
   return types, tuple(pd.Index(labels).tolist())


def _couplings(ent, lay, mode, directed):
   """
   Couplings between the copies of each entity.

   :param ent: Entity of each copy.
   :param lay: Layer (position in the layer order) of each copy.
   :param mode: ``"all"`` (every pair of copies), ``"ordinal"`` (copies in adjacent layers) or
      ``"temporal"`` (from each copy to the copy in the next layer, if any).
   :param directed: Whether the network is directed: then ``"all"`` and ``"ordinal"`` couplings
      are stored in both directions.
   :return: Source and target copies of each coupling.
   """
   order = np.lexsort((lay, ent))
   e, l = ent[order], lay[order]
   if mode == "all":
      start = np.flatnonzero(np.r_[True, e[1:] != e[:-1]]) if len(e) else np.zeros(0, dtype = np.int64)
      size = np.diff(np.r_[start, len(e)])
      later = np.repeat(size, size) - (np.arange(len(e)) - np.repeat(start, size)) - 1   # copies after each one
      a = np.repeat(np.arange(len(e)), later)
      b = a + 1 + np.arange(len(a)) - np.repeat(np.cumsum(later) - later, later)
   else:
      a = np.flatnonzero((e[1:] == e[:-1]) & (l[1:] == l[:-1] + 1))
      b = a + 1
   a, b = order[a], order[b]
   if directed and mode != "temporal":
      a, b = np.concatenate([a, b]), np.concatenate([b, a])
   return a, b


def _present(x):
   """
   Non-missing values of a layer column, with integral floats (from columns with missing values)
   turned back into integers.

   :param x: :class:`pandas.Series`.
   :return: :class:`pandas.Series`.
   """
   x = x[x.notna()]
   if x.dtype.kind == "f" and bool((x == np.floor(x)).all()):
      x = x.astype(np.int64)
   return x


def _layered(edges, src, tgt, lcol, eattr, inter, iattr, nodes, node_format, directed, coupling, coupling_weight, layers):
   """
   Copies of the entities of a multilayer network, with their edges and node table rows.

   :return: ``(names, s, t, ea, edge_attributes, key, table, ml, inter)``: ids ``(entity, layer)``
      of the copies by integer id; source and target of every edge (intra-layer edges, then
      interlayer edges, then generated couplings) and their attributes; integer id of every node
      table row, after expanding rows without a layer to all the copies of their entity and
      merging rows of the same copy (``None`` without node table); attribute columns of these
      rows (wide) or their attribute names and values (long); ``(layer, entity, layer_labels,
      entity_ids)`` as taken by :func:`_build`; mask of the interlayer edges.
   :raises ValueError: See :func:`load_network`.
   """
   m, mi = len(edges), 0 if inter is None else len(inter)
   ic = None if inter is None else list(inter.columns)
   ent = [edges[src], edges[tgt]] + ([] if inter is None else [inter[ic[0]], inter[ic[2]]]) + ([] if nodes is None else [nodes.iloc[:, 0]])
   lay = [edges[lcol], edges[lcol]] + ([] if inter is None else [inter[ic[1]], inter[ic[3]]]) + ([] if nodes is None else [nodes.iloc[:, 1]])
   ecode, entity_ids = pd.factorize(pd.concat(ent, ignore_index = True))
   entity_ids = entity_ids.to_numpy()
   if layers is None:
      try:
         labels = sorted(pd.unique(pd.concat([_present(x) for x in lay], ignore_index = True)))
      except TypeError:
         raise ValueError("the layers cannot be ordered: give their order with layers") from None
   else:
      labels = list(layers)
   labels = [x.item() if isinstance(x, np.generic) else x for x in labels]
   L = len(labels)
   index = pd.Index(labels)
   lcode = np.concatenate([index.get_indexer(x) for x in lay])
   missing = np.concatenate([x.isna().to_numpy() for x in lay])
   if ((lcode < 0) & ~missing).any():
      raise ValueError(f"layer {pd.concat(lay, ignore_index = True)[np.argmax((lcode < 0) & ~missing)]!r} is not in layers")
   if missing[:2 * (m + mi)].any():
      raise ValueError("every edge needs a layer")

   # copies: integer ids of the (entity, layer) pairs
   copy = np.full(len(lcode), -1, dtype = np.int64)
   valid = lcode >= 0
   copy[valid], keys = pd.factorize(ecode[valid] * L + lcode[valid])
   keys = np.asarray(keys)
   c_lay, c_ent = keys % L, keys // L
   names = np.fromiter(zip(entity_ids[c_ent].tolist(), _label_array(labels)[c_lay].tolist()), dtype = object, count = len(keys))
   s, t = copy[:m], copy[m:2 * m]
   si, ti = copy[2 * m:2 * m + mi], copy[2 * m + mi:2 * (m + mi)]
   if (c_lay[si] == c_lay[ti]).any():
      e = np.argmax(c_lay[si] == c_lay[ti])
      raise ValueError(f"interlayer edge {names[si[e]]!r} - {names[ti[e]]!r} joins two nodes of the same layer: give it in the edge table")

   # couplings, except those already given as interlayer edges
   cs, ct = _couplings(c_ent, c_lay, coupling, directed) if coupling else (np.zeros(0, dtype = np.int64),) * 2
   if mi and len(cs):
      given = si * len(keys) + ti if directed else np.concatenate([si * len(keys) + ti, ti * len(keys) + si])
      new = ~np.isin(cs * len(keys) + ct, given)
      cs, ct = cs[new], ct[new]
   if coupling == "all" and len(cs) > m:
      warnings.warn(f"coupling = 'all' adds {len(cs):,} couplings, more than the {m:,} intra-layer edges: consider coupling = 'ordinal'")
   w = np.asarray(coupling_weight, dtype = np.float64)
   if w.ndim and w.shape != (L, L):
      raise ValueError(f"coupling_weight must be a number or a {L} by {L} matrix (layers in order), got shape {w.shape}")
   cw = w if w.ndim == 0 else w[c_lay[cs], c_lay[ct]]
   if not np.all(np.isfinite(cw) & (cw > 0)):
      raise ValueError("coupling weights must be positive and finite (use aggregate_network for infinite couplings)")

   # edge attributes: intra-layer ones, then those only in the interlayer table
   cols = list(eattr.columns) + ([] if inter is None else [c for c in iattr.columns if c not in eattr.columns])
   ea = np.full((m + mi + len(cs), len(cols)), np.nan, dtype = np.float32)
   ea[:m, :eattr.shape[1]] = eattr.to_numpy(np.float32)
   for j, c in enumerate(cols):
      if inter is not None and c in iattr.columns:
         ea[m:m + mi, j] = iattr[c].to_numpy(np.float32)
   ea[m + mi:] = cw if w.ndim == 0 else cw[:, None]
   ml = (c_lay, c_ent, tuple(labels), entity_ids)
   inter_mask = np.r_[np.zeros(m, dtype = bool), np.ones(mi + len(cs), dtype = bool)]
   s, t = np.concatenate([s, si, cs]), np.concatenate([t, ti, ct])
   if nodes is None:
      return names, s, t, ea, cols, None, None, ml, inter_mask

   # node rows: rows without a layer apply to every copy of their entity, rows of a copy override
   off = 2 * (m + mi)
   ncopy, nent = copy[off:], ecode[off:]
   bc = np.flatnonzero(ncopy < 0)
   order = np.argsort(c_ent, kind = "stable")
   start = np.searchsorted(c_ent[order], nent[bc], "left")
   cnt = np.searchsorted(c_ent[order], nent[bc], "right") - start
   pos = np.repeat(start, cnt) + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
   own = np.flatnonzero(ncopy >= 0)
   rows = np.concatenate([np.repeat(bc, cnt), own])
   key = np.concatenate([order[pos], ncopy[own]])
   table = nodes.iloc[rows].reset_index(drop = True)
   if node_format == "wide":
      feats = table.iloc[:, 2:]
      if LAYER in feats.columns:
         raise ValueError(f"{LAYER!r} cannot be a node attribute of a multilayer network: the layer is the second column of the node table")
      if len(np.unique(key)) < len(key):   # last non-missing value of each attribute
         feats = feats.groupby(key, sort = False).last()
         key = feats.index.to_numpy()
         feats = feats.reset_index(drop = True)
      return names, s, t, ea, cols, key, feats, ml, inter_mask
   rows = pd.DataFrame({"key": key, "attribute": table.iloc[:, 2].to_numpy(), "value": table.iloc[:, 3].to_numpy(), "own": np.r_[np.zeros(len(key) - len(own), dtype = bool), np.ones(len(own), dtype = bool)]})
   top = rows.groupby(["key", "attribute"], sort = False, dropna = False)["own"].transform("max")
   rows = rows[(rows["own"] == top).to_numpy()].drop(columns = "own").reset_index(drop = True)   # several rows of a level: several values
   if (rows["attribute"] == LAYER).any():
      raise ValueError(f"{LAYER!r} cannot be a node attribute of a multilayer network: the layer is the second column of the node table")
   return names, s, t, ea, cols, rows["key"].to_numpy(), (rows["attribute"], rows["value"]), ml, inter_mask


def _load(edges, nodes = None, sep = "\t", header = True, source = 0, target = 1, edge_attributes = True, directed = False, node_format = "wide", bipartite = False, device = "cpu", multilayer = False, layer = LAYER, interlayer_edges = None, coupling = None, coupling_weight = 1, layers = None):
   """
   Body of :func:`load_network`, so that its temporary tables are freed when it returns.
   """
   if bipartite and nodes is None:
      raise ValueError(f"a bipartite network needs a node table with a {BIPARTITE!r} attribute")
   if not multilayer and (interlayer_edges is not None or coupling is not None or layers is not None):
      raise ValueError("interlayer_edges, coupling and layers are for multilayer networks: load with multilayer = True")
   if coupling not in (None, "all", "ordinal", "temporal"):
      raise ValueError(f"unknown coupling: {coupling}")
   if coupling == "temporal" and not directed:
      raise ValueError("temporal couplings go forward in time, so they need directed = True (give undirected contacts in both directions)")
   hdr = 0 if header else None
   edges, (src, tgt) = _read(edges, sep, hdr, [source, target])
   drop = [src, tgt]
   if multilayer:
      if isinstance(layer, str) and layer not in edges.columns:
         raise ValueError(f"a multilayer network needs a {layer!r} column in the edge table")
      lcol = _column(edges.columns, layer)
      drop.append(lcol)
   eattr = _numeric(edges.drop(columns = drop)) if edge_attributes else edges[[]]
   inter, iattr = None, None
   if multilayer and interlayer_edges is not None:
      inter = _read(interlayer_edges, sep, hdr, [0, 2])[0]
      iattr = _numeric(inter.drop(columns = inter.columns[:4])) if edge_attributes else inter[[]]
   if nodes is not None:
      nodes = _read(nodes, sep, hdr, [0] if node_format == "wide" else [0, 2] if multilayer else [0, 1])[0]
   m = len(edges)

   # integer ids of the nodes (copies, in multilayer networks), edges and node table rows
   ml, inter_mask, key, feats, name, v = None, None, None, None, None, None
   if multilayer:
      names, s, t, ea, enames, key, rows, ml, inter_mask = _layered(edges, src, tgt, lcol, eattr, inter, iattr, nodes, node_format, directed, coupling, coupling_weight, layers)
      if nodes is not None:
         feats, (name, v) = (rows, (None, None)) if node_format == "wide" else (None, rows)
   else:
      ids = [edges[src], edges[tgt]] + ([] if nodes is None else [nodes.iloc[:, 0]])
      codes, names = pd.factorize(pd.concat(ids, ignore_index = True))
      names = names.to_numpy()
      s, t = codes[:m], codes[m:2 * m]
      ea, enames = eattr.to_numpy(np.float32), eattr.columns
      if nodes is not None:
         key = codes[2 * m:]
         feats, name, v = (nodes.iloc[:, 1:], None, None) if node_format == "wide" else (None, nodes.iloc[:, 1], nodes.iloc[:, 2])
   del edges, eattr, inter, iattr
   n = len(names)

   # node attributes, rows in the order given by _build
   attrs, quals, table, qtable, tkey, tval = [], [], None, None, None, None
   if nodes is not None:
      if node_format == "wide":
         if bipartite and BIPARTITE not in feats.columns:
            raise ValueError(f"a bipartite network needs a {BIPARTITE!r} column in the node table")
         num = _numeric(feats.drop(columns = [BIPARTITE]) if bipartite else feats)
         attrs = list(num.columns)
         quals = [a for a in feats.columns if a not in attrs]
         if bipartite:
            tkey, tval = key, feats[BIPARTITE].to_numpy()

         def table(perm):
            row = _positions(perm)[key]
            x = np.full((n, len(attrs)), np.nan, dtype = np.float32)
            for j, a in enumerate(attrs):   # column by column: no extra copy of the whole table
               x[row, j] = num[a].to_numpy(np.float32)
            return x

         def qtable(perm):
            row = _positions(perm)[key]
            Q = {}
            for a in quals:   # as category codes: no Python object per value
               cat = pd.Categorical(_memberships(feats[a]))
               c = np.full(n, -1, dtype = cat.codes.dtype)
               c[row] = cat.codes
               Q[a] = pd.Categorical.from_codes(c, cat.categories)
            return pd.DataFrame(Q, index = pd.RangeIndex(n))
      else:
         val = pd.to_numeric(v, errors = "coerce")
         has = v.notna().to_numpy()
         text = name.isin(set(name[has & val.isna().to_numpy()])).to_numpy()   # attributes with a non-numeric value
         repeated = pd.DataFrame({"k": key, "a": name.to_numpy()})[has].duplicated(keep = False).to_numpy()
         multi = name.isin(set(name[has][repeated])).to_numpy()   # attributes with several values for a node
         isq = text | multi | (bipartite & (name == BIPARTITE).to_numpy())
         ok = has & ~isq
         attrs = list(name[ok].unique())
         col = pd.Index(attrs).get_indexer(name[ok])
         qrow = has & isq
         qval = np.where(text, np.asarray(v, dtype = object), val.to_numpy())[qrow]
         quals = list(name[qrow].unique())
         for a in set(name[multi & ~text]):   # numeric labels of several values: integers stay integers
            m = (name[qrow] == a).to_numpy()
            x = qval[m].astype(float)
            if (x == np.round(x)).all():
               qval[m] = [int(y) for y in x]
         if bipartite:
            if BIPARTITE not in quals:
               raise ValueError(f"a bipartite network needs {BIPARTITE!r} values in the node table")
            b = (name[qrow] == BIPARTITE).to_numpy()
            tkey, tval = key[qrow][b], qval[b]

         def table(perm):
            x = sp.csr_matrix((val[ok].to_numpy(np.float32), (key[ok], col)), shape = (n, len(attrs)))[perm]
            x.sum_duplicates()
            return x

         def qtable(perm):
            Q = pd.DataFrame({"node": _positions(perm)[key[qrow]], "attribute": name[qrow].to_numpy(), "value": qval})
            return _categorize(_merge_values(Q.drop_duplicates()))

   types, labels = _bipartite_types(tkey, tval, names, s, t, inter_mask, None if ml is None else ml[1]) if bipartite else (None, None)
   return _build(s, t, names, ea, enames, table, attrs, directed, device, qtable, quals, node_format, types, labels, ml)


def load_network(edges, nodes = None, sep = "\t", header = True, source = 0, target = 1, edge_attributes = True, directed = False, node_format = "wide", bipartite = False, device = "cpu", multilayer = False, layer = LAYER, interlayer_edges = None, coupling = None, coupling_weight = 1, layers = None):
   """
   Load a network from an edge list and a node attribute table, given as files or as
   :class:`pandas.DataFrame`.

   Node ids can be arbitrary strings. Ids read from files are strings; ids in a DataFrame are kept
   as they are, so the edge and node tables must use the same type (e.g. both ``int``). DataFrames
   are not modified, and their column names play the role of the header row. Nodes are split by
   weakly connected component and remapped to ``(component, index)``, with ``index`` from 0 to
   n - 1. Nodes appearing only in the node table become isolated components. In undirected
   networks every edge is stored in both directions (input edges first, then their reverses, self
   loops once), following the torch_geometric convention. Every component has an ``edge_attr``
   and a ``node_attr`` tensor, with zero columns when there are no such attributes. Only numeric
   edge attributes are stored.

   Node attributes whose values are all numbers (or booleans) are numeric, and are stored in
   ``node_attr`` on ``device``. The others are qualitative, and are stored in
   :attr:`Network.qualitative_node_attr`, always in CPU memory. In long format, this is decided
   for each attribute name: one non-numeric value makes the attribute qualitative.

   A node can have several values of a qualitative attribute, e.g. overlapping communities (see
   :ref:`overlapping-communities`): in long format, as several rows with the same node and
   attribute (which makes the attribute qualitative, even with numeric values; integer values stay
   integers); in a wide DataFrame, as a list, tuple or set in the cell. They are stored as a tuple
   of the distinct values.

   With ``bipartite = True``, the node table must give every node, isolated ones included, a value
   of the :data:`BIPARTITE` attribute (``"bipartite_type"``), with at most two distinct values
   overall. The smaller value (in sort order) is type 0, the other type 1, as recorded in
   :attr:`Network.bipartite_labels`. :data:`BIPARTITE` is always stored as a qualitative
   attribute, whatever its values. In every component, nodes of type 0 come first (see
   :class:`Network`). Edges can join nodes in any column order and, if directed, go from either
   type to the other.

   With ``multilayer = True`` (see :doc:`multilayer`), every node is a copy of an entity in one
   layer, with input id ``(entity, layer)``:

   * the edge table needs a ``layer`` column (name or position given by ``layer``): each edge joins
     the copies of its two entities in that layer;
   * ``interlayer_edges`` (file or DataFrame, same ``sep`` and ``header``) gives interlayer edges
     by position: source, source layer, target, target layer, then edge attributes (matched by
     name with those of the edge table; missing ones are NaN). An interlayer edge between the two
     layers of the same entity is a coupling;
   * ``coupling`` generates couplings between the copies of every entity: ``"all"`` (every pair of
     copies; a warning is issued if they outnumber the intra-layer edges), ``"ordinal"`` (copies in
     adjacent layers only, undirected) or ``"temporal"`` (from each copy to the copy in the next
     layer, if any: forward in time only, so it needs ``directed = True``). Layers are adjacent in
     the layer order: an entity without a copy in a layer is not coupled across it, but a layer
     with no node at all exists only if declared in ``layers`` (e.g. time steps 1, 2, 4 without
     ``layers`` make 2 and 4 adjacent). In directed networks
     ``"all"`` and ``"ordinal"`` couplings are stored in both directions. Couplings already given in
     ``interlayer_edges`` are not added again. Generated couplings get ``coupling_weight`` (a
     positive number, or a matrix with one value per pair of layers, in layer order) in every edge
     attribute;
   * the node table (if any) has the entity in the first column and the layer in the second, then
     the attributes (wide), or the attribute name and the value (long). Each row is a copy; copies
     appearing only there have no edges. A row with a missing layer applies to all the copies of
     its entity (those with edges or with their own rows), and the rows of a copy override it, so
     entity-wide attributes (e.g. :data:`BIPARTITE`) are given once. A long row with a missing
     attribute only declares the copy;
   * layers are ordered by their values (parsed as numbers when possible), or as given in
     ``layers``, which also lets layers without nodes exist. The order matters for ordinal and
     temporal couplings and is recorded in :attr:`Network.layer_labels`.

   A bipartite multilayer network must be bipartite in every layer; all the copies of an entity
   have the same type, and interlayer edges must join nodes of the same type.

   :param edges: Edge list, as a file path or a :class:`pandas.DataFrame`.
   :param nodes: Node attribute table, as a file path or a :class:`pandas.DataFrame`, or ``None``.
   :param sep: Column separator of both files (ignored for DataFrames).
   :param header: Whether both files have a header row (ignored for DataFrames).
   :param source: Name or position of the source node column.
   :param target: Name or position of the target node column.
   :param edge_attributes: Whether to load the remaining edge columns as edge attributes.
   :param directed: Whether the edges are directed.
   :param node_format: ``"wide"`` (node id, then one column per attribute; numeric attributes
      stored as a dense ``node_attr``) or ``"long"`` (node id, attribute name, value; numeric
      attributes stored as a sparse CSR ``node_attr``).
   :param bipartite: Whether the network is bipartite, with node types given by :data:`BIPARTITE`.
   :param device: Device where all tensors are created, e.g. ``"cuda"``. Qualitative attributes
      stay in CPU memory.
   :param multilayer: Whether the network is multilayer.
   :param layer: Name or position of the layer column of the edge table, in multilayer networks.
   :param interlayer_edges: Interlayer edges, as a file path or a :class:`pandas.DataFrame`, or
      ``None``.
   :param coupling: ``"all"``, ``"ordinal"``, ``"temporal"`` or ``None`` (no generated couplings).
   :param coupling_weight: Value of every edge attribute of the generated couplings: a positive
      number, or an ``L`` by ``L`` matrix indexed by layer positions.
   :param layers: Order of the layers, or ``None`` for their sort order.
   :return: The loaded :class:`Network`.
   :raises ValueError: In bipartite networks, if there is no node table or no :data:`BIPARTITE`
      attribute, if it has more than two distinct values, if a node has no type or two different
      ones (ids must be unique across node types), or if an edge joins two nodes of the same type.
      In multilayer networks, if there is no layer column, if an edge has no layer, if a layer is
      not in ``layers`` or layers cannot be sorted, if an interlayer edge joins two nodes of the
      same layer, if the coupling or its weight is not valid, if ``"temporal"`` coupling is asked
      for an undirected network, if a node attribute is named ``"layer"``, or, in bipartite
      multilayer networks, if an interlayer edge joins nodes of different types or the copies of an
      entity have different types. If ``interlayer_edges``, ``coupling`` or ``layers`` are given
      without ``multilayer = True``.

   Examples:

   .. code-block:: python

      import graphoskop as gsk

      # tab separated, headers, first two columns are source and target
      net = gsk.load_network("edges.tsv", "nodes.tsv")

      # comma separated, named endpoint columns, directed
      net = gsk.load_network("edges.csv", "nodes.csv", sep = ",", source = "from", target = "to", directed = True)

      # no headers, no edge attributes, node table in long format
      net = gsk.load_network("edges.tsv", "nodes.tsv", header = False, edge_attributes = False, node_format = "long")

      # everything on the GPU (except qualitative node attributes)
      net = gsk.load_network("edges.tsv", "nodes.tsv", device = "cuda")

      # from DataFrames, node table in long format (id, attribute name, value)
      import pandas as pd
      edges = pd.DataFrame({"from": ["alice", "bob"], "to": ["bob", "carol"], "weight": [1.5, 2.0]})
      nodes = pd.DataFrame({"id": ["alice", "carol"], "attribute": ["age", "city"], "value": [34, "Rome"]})
      net = gsk.load_network(edges, nodes, source = "from", target = "to", node_format = "long")

      # bipartite: users and the items they bought
      edges = pd.DataFrame({"user": ["u1", "u1", "u2"], "item": ["i1", "i2", "i1"]})
      nodes = pd.DataFrame({"id": ["u1", "u2", "i1", "i2", "i3"], "bipartite_type": ["user", "user", "item", "item", "item"]})
      net = gsk.load_network(edges, nodes, bipartite = True)
      net.bipartite_labels                 # ("item", "user"): items are type 0
      n0 = net.bipartite_split[0]          # items of component 0 are nodes 0 to n0 - 1

      # multiplex: friendship and work layers, all copies of each person coupled
      edges = pd.DataFrame({"source": ["a", "b", "a"], "target": ["b", "c", "c"], "layer": ["friends", "friends", "work"]})
      nodes = pd.DataFrame({"id": ["a", "d"], "layer": [None, "work"], "age": [30, 41]})
      net = gsk.load_network(edges, nodes, multilayer = True, coupling = "all", coupling_weight = 0.5)
      net.node_map[("a", "work")]          # (component, index) of the copy of "a" in layer "work"

      # temporal network: snapshots 1, 2, 3, couplings forward in time
      net = gsk.load_network("contacts.tsv", multilayer = True, layer = "time", coupling = "temporal", directed = True)
   """
   net = _load(edges, nodes, sep, header, source, target, edge_attributes, directed, node_format, bipartite, device, multilayer, layer, interlayer_edges, coupling, coupling_weight, layers)
   _release_memory()
   return net


def _wide_rows(net, rows):
   """
   Wide node table of some nodes, as written by :func:`save_network`.

   Numeric values absent from a long (sparse) table are 0, qualitative ones are NaN.

   :param net: Network.
   :param rows: Positions of the nodes, in the network wide node order.
   :return: :class:`pandas.DataFrame` with a ``node`` column (and a ``layer`` column in multilayer
      networks), then one column per numeric and per qualitative attribute.
   """
   x = _node_table(net, rows)
   nodes = pd.DataFrame(x.toarray() if sp.issparse(x) else x, columns = net.node_attributes)
   Q = _qualitative_rows(net, rows)
   for a in net.qualitative_node_attributes:
      if net.node_format == "wide":
         nodes[a] = Q[a].to_numpy()
      else:
         col = np.full(len(rows), np.nan, dtype = object)
         m = (Q["attribute"] == a).to_numpy()
         col[Q["node"].to_numpy()[m]] = np.asarray(Q["value"], dtype = object)[m]
         nodes[a] = col
   idc = _id_columns(net, rows)
   for k in reversed(list(idc)):
      nodes.insert(0, k, idc[k])
   return nodes


def _multivalued(net):
   """
   Qualitative node attributes with several values for some node (e.g. overlapping communities).

   :param net: Network.
   :return: List of attribute names.
   """
   Q = net.qualitative_node_attr
   if net.node_format == "wide":
      return [a for a in net.qualitative_node_attributes if any(isinstance(c, tuple) for c in Q[a].cat.categories)]
   multi = Q["value"].map(lambda x: isinstance(x, tuple)).to_numpy(bool)
   return list(pd.unique(np.asarray(Q["attribute"], dtype = object)[multi]))


def _to_long(nodes, ids = 1):
   """
   Long node table from a wide one, node by node; missing values are left out. With two id
   columns (multilayer networks), a node without any value gets a row with a missing attribute and
   value, so that it is not lost.

   :param nodes: Wide node table, with ``ids`` id columns first (``node``, and ``layer``).
   :param ids: Number of id columns.
   :return: :class:`pandas.DataFrame` with the id columns, ``attribute`` and ``value``.
   """
   nodes = nodes.reset_index(drop = True)
   idc = list(nodes.columns[:ids])
   long = nodes.drop(columns = idc).melt(var_name = "attribute", value_name = "value", ignore_index = False)
   long = long[long["value"].notna().to_numpy()]
   if ids > 1:
      bare = np.setdiff1d(np.arange(len(nodes)), long.index.to_numpy())
      long = pd.concat([long, pd.DataFrame({"attribute": np.nan, "value": np.nan}, index = bare)])
   long = long.sort_index(kind = "stable").explode("value")   # one row per community of a node
   for k in reversed(idc):
      long.insert(0, k, nodes[k].to_numpy()[long.index])
   return long.reset_index(drop = True)


def _table_sizes(net, nodes, sep, sample = 1000):
   """
   Expected size in bytes of the node table file in wide and in long format, from random samples.

   :param net: Network.
   :param nodes: Node table of the network, in its own format (from
      :meth:`Network.to_pandas_dataframes`).
   :param sep: Column separator.
   :param sample: Number of sampled rows.
   :return: Dictionary with the sizes of ``"wide"`` and ``"long"``.
   """
   rng = np.random.default_rng(0)
   pick = lambda n: np.sort(rng.choice(n, min(n, sample), replace = False))
   size = lambda df, rows: len(df.to_csv(sep = sep, index = False, header = False)) / max(len(df), 1) * rows
   n = int(net.node_offsets[-1])
   ids = 1 if net.layer_labels is None else 2
   if net.node_format == "wide":
      cells = int(nodes.iloc[:, ids:].notna().to_numpy().sum())
      wide = nodes.iloc[pick(n)]
      return {"wide": size(wide, n), "long": size(_to_long(wide, ids), cells)}
   return {"wide": size(_wide_rows(net, pick(n)), n), "long": size(nodes.iloc[pick(len(nodes))], len(nodes))}


def save_network(net, prefix, sep = "\t", node_format = None, pickle = False, warn_size = 1e8, max_size = 1e9):
   """
   Write a network to disk, as an edge table and a node table, or as a Python object.

   The edge table goes to ``prefix + "_edges.tsv"`` and, if there are node attributes, the node
   table to ``prefix + "_nodes.tsv"`` (``.csv`` instead of ``.tsv`` if ``sep`` is not a tab).
   Both have a header row, and are written as by :meth:`Network.to_pandas_dataframes`: edges with
   columns ``source``, ``target`` and their attributes, nodes in the network wide node order. Values
   containing the separator, a quote or a line break are quoted, following the CSV standard. All
   numeric and qualitative attributes are written, so the files load back with
   :func:`load_network`, passing the same ``sep``, ``node_format``, ``directed`` and ``bipartite``
   (a bipartite network writes :data:`BIPARTITE` for every node, and loads back as bipartite).
   Ids and qualitative values that look like numbers are read back as numbers.

   The node table can be written in either format, whatever the format of ``net``. In long format,
   a numeric value that is missing (NaN) in a wide table is left out, and so reads back as 0. In
   wide format, numeric values absent from a long table are written as 0.

   A node with several values of an attribute (see :ref:`overlapping-communities`) gets one row per
   value in a long table, which loads back as several values. A wide table cannot hold them: the
   long format is then the default, and asking for the wide one raises :class:`ValueError`.

   A multilayer network also writes its interlayer edges (couplings included) to
   ``prefix + "_interlayer.tsv"``, and always writes the node table (one row per copy, with the
   entity and the layer). The files load back with ``multilayer = True``,
   ``interlayer_edges = prefix + "_interlayer.tsv"`` and the same ``directed``, ``node_format``
   and ``bipartite``, without ``coupling``.

   The size of the node table file is estimated in both formats from a sample of nodes, before
   writing anything. If it is expected above ``warn_size`` bytes in the chosen format, a warning is
   issued. If moreover the chosen format is not the smaller one (wide with many attributes and few
   values per node, or long with dense attributes) and the file is expected above ``max_size``
   bytes, :class:`ValueError` is raised and nothing is written. A node table that is large in both
   formats is always written in the smaller one, with a warning.

   With ``pickle = True``, the :class:`Network` object is written as it is to ``prefix + ".pt"``
   with :func:`torch.save`, tensors on their current device, and is read back with
   :func:`load_pickled_network`.

   :param net: A :class:`Network`.
   :param prefix: Path and file name prefix, e.g. ``"data/flights"``.
   :param sep: Column separator, e.g. ``"\\t"``, ``","``, ``";"`` or ``" "``.
   :param node_format: ``"wide"`` or ``"long"`` node table, or ``None`` for the format of ``net``
      (long if some node has several values of an attribute).
   :param pickle: Write the Python object instead of tables.
   :param warn_size: Expected size in bytes of the node table above which a warning is issued.
   :param max_size: Expected size in bytes of the node table, in the larger of the two formats,
      above which nothing is written.
   :raises ValueError: If ``node_format`` is not valid, if it is ``"wide"`` and some node has
      several values of an attribute, or if the node table would be larger than ``max_size`` in the
      chosen format and smaller in the other one.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("purchases.tsv", "nodes.tsv", bipartite = True)
      gsk.save_network(net, "out/purchases")                  # out/purchases_edges.tsv, out/purchases_nodes.tsv
      net = gsk.load_network("out/purchases_edges.tsv", "out/purchases_nodes.tsv", bipartite = True)

      gsk.save_network(net, "out/purchases", sep = ",", node_format = "long")   # .csv files
      net = gsk.load_network("out/purchases_edges.csv", "out/purchases_nodes.csv", sep = ",", node_format = "long", bipartite = True)

      gsk.save_network(net, "out/purchases", pickle = True)   # out/purchases.pt
      net = gsk.load_pickled_network("out/purchases.pt")

      net = gsk.load_network("multiplex.tsv", multilayer = True, coupling = "all")
      gsk.save_network(net, "out/multiplex")                  # _edges.tsv, _interlayer.tsv, _nodes.tsv
      net = gsk.load_network("out/multiplex_edges.tsv", "out/multiplex_nodes.tsv", multilayer = True, interlayer_edges = "out/multiplex_interlayer.tsv")
   """
   if pickle:
      torch.save(net, prefix + ".pt")
      return
   multi = _multivalued(net)
   node_format = ("long" if multi else net.node_format) if node_format is None else node_format
   if node_format not in ("wide", "long"):
      raise ValueError(f"unknown node format: {node_format}")
   ext = ".tsv" if sep == "\t" else ".csv"
   if node_format == "wide" and multi:
      raise ValueError(f"node attributes {multi} have several values for some nodes, which a wide table cannot hold: use node_format = 'long' (one row per value) or pickle = True")
   ml = net.layer_labels is not None
   has_nodes = bool(net.node_attributes or net.qualitative_node_attributes) or ml
   out = net.to_pandas_dataframes()
   if ml:
      edges, inter, nodes = out
   else:
      edges, nodes = out if has_nodes else (out, None)
   if has_nodes:
      size = _table_sizes(net, nodes, sep)
      other = "long" if node_format == "wide" else "wide"
      if size[node_format] > warn_size:
         msg = f"the {node_format} node table is expected to take about {size[node_format] / 1e6:,.1f} MB, against {size[other] / 1e6:,.1f} MB in {other} format"
         if size[node_format] > size[other] and size[node_format] > max_size:
            raise ValueError(msg + f": use node_format = {other!r} or pickle = True, or raise max_size")
         warnings.warn(msg + (f": consider node_format = {other!r}" if size[node_format] > size[other] else ""))
   edges.to_csv(prefix + "_edges" + ext, sep = sep, index = False)
   if ml:
      inter.to_csv(prefix + "_interlayer" + ext, sep = sep, index = False)
   if not has_nodes:
      return
   if node_format != net.node_format:
      nodes = _to_long(nodes, 2 if ml else 1) if node_format == "long" else _wide_rows(net, np.arange(int(net.node_offsets[-1])))
   nodes.to_csv(prefix + "_nodes" + ext, sep = sep, index = False)


def load_pickled_network(path, device = None):
   """
   Read a network written by :func:`save_network` with ``pickle = True``.

   Unpickling can run arbitrary code: only read files from trusted sources.

   :param path: File path.
   :param device: Device where all tensors are moved, e.g. ``"cuda"`` or ``"cpu"``, or ``None``
      to keep the device they were saved from. Qualitative attributes stay in CPU memory.
   :return: The :class:`Network`.
   """
   return torch.load(path, map_location = device, weights_only = False)
