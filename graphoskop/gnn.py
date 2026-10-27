"""
Graph neural networks: node attributes predicted from the other node attributes and the network,
with the layers of PyTorch Geometric. Four models share the same parameters and output:
:func:`gcn` (graph convolutions), :func:`graphsage` (own features kept apart from the mean of the
neighbors'), :func:`gat` and :func:`gatv2` (attention weighted averages of the neighbors).
:func:`gae` and :func:`vgae`, graph autoencoders, learn node vectors without a target, from the
node attributes and the edges, by reconstructing the edges.
:func:`node2vec`, :func:`struc2vec`, :func:`metapath2vec`, :func:`laplacian_eigenmaps` and
:func:`locally_linear_embedding` are shallow embeddings instead: unsupervised, they give one vector
per node from the structure alone, with no target and no features, learned from random walks
(similar vectors for nearby nodes, for nodes with similar roles with :func:`struc2vec`, or along
chains of node types with :func:`metapath2vec`) or computed as eigenvectors.
:func:`diffpool` gives one vector per graph (per component), pooling the nodes of each graph into
clusters, with or without a graph attribute to predict. On hypergraphs and simplicial complexes,
:func:`hgnn` predicts node attributes through the hyperedges (or facets), and :func:`scnn` predicts
attributes of the edges, triangles... of a complex, with the same parameters and output.

A model is a stack of graph convolution layers, given by the number of output channels of each
layer: ``[32, 16, 2]`` is three layers, from the input features to 32, then 16, then 2 channels,
the last one being the output (one channel per class, or one for a number). An activation (or
none) follows each layer, and dropout precedes each layer. The node attribute to predict is the
target; the features are all the other numeric node attributes (missing values replaced by the
mean, then standardized) and, optionally, the qualitative ones as dummy variables (one 0 / 1
column per value; a node with several values gets several ones). A network without features
gets a constant feature, so that the model learns from the structure alone.

The feature matrix is built directly on the device of the network. When it would be large (more
than 67 million entries) and mostly zeros (fewer than 10% nonzero), as with the many attributes
of a long node table (one per label, genre, year...) or the dummies of an attribute with a value
per node, it is kept sparse: its memory is that of its nonzero values, and the layers multiply it
as a sparse matrix. Sparse numeric features are standardized by scaling only (unit variance,
mean not removed), so that zeros stay zeros.

The nodes with a known target are split at random into a training set, whose loss trains the
model, and a validation set, which only measures how well the model generalizes; the nodes
without a target are predicted too. The whole network is one graph: all components, on its device
(GPU if there).
"""

import contextlib
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ._policy import _higher_order
from .isomorphism import _node_values
from .network import BIPARTITE, _quiet_sparse
from .progress import _bar

#: Activations by name.
_ACTIVATIONS = {"relu": torch.relu, "elu": F.elu, "leaky_relu": F.leaky_relu, "gelu": F.gelu, "tanh": torch.tanh, "sigmoid": torch.sigmoid, "softmax": lambda x: torch.softmax(x, 1), "log_softmax": lambda x: torch.log_softmax(x, 1)}

#: Losses by name.
_LOSSES = {"cross_entropy": F.cross_entropy, "nll": F.nll_loss, "mse": lambda p, y: F.mse_loss(p.reshape(y.shape), y), "l1": lambda p, y: F.l1_loss(p.reshape(y.shape), y), "huber": lambda p, y: F.huber_loss(p.reshape(y.shape), y)}

#: Optimizers by name.
_OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}


#: A feature matrix with more entries than this, and fewer than 10% nonzero, is kept sparse.
_SPARSE = 2 ** 26


def _dummies(net, a):
   """
   Dummy variables of a qualitative node attribute: one column per value (sorted as text), with a
   one for every node with that value (several ones for a node with several values).

   :return: ``(rows, columns, names)``: network wide node positions and column positions of the
      ones (NumPy arrays), and the column names.
   """
   if net.node_format == "long":   # the table already has one row per (node, value)
      Q = net.qualitative_node_attr
      Q = Q[(Q["attribute"] == a).to_numpy()]
      node, val = Q["node"].to_numpy(np.int64), Q["value"].to_numpy(dtype = object)
   else:
      node, val = [], []
      for i, x in enumerate(_node_values(net, a)):
         for y in (x if isinstance(x, (tuple, list, set, frozenset)) else () if pd.isna(x) else (x,)):
            node.append(i)
            val.append(y)
      node, val = np.asarray(node, dtype = np.int64), np.asarray(val, dtype = object)
   values = sorted(set(val.tolist()), key = str)
   col = pd.Index(values).get_indexer(val) if len(val) else np.zeros(0, dtype = np.int64)
   pairs = np.unique(np.stack([node, col]), axis = 1) if len(val) else np.zeros((2, 0), dtype = np.int64)
   return pairs[0], pairs[1], [f"{a}={v}" for v in values]


@contextlib.contextmanager
def _deterministic(dev, seed):
   """
   Deterministic algorithms of PyTorch while training on the CPU with a seed, so that runs repeat
   exactly (parallel sparse products otherwise sum in varying orders); the caller's setting is
   restored after. On a GPU, where they can need extra configuration, results repeat up to
   rounding.
   """
   before, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
   if seed is not None and dev.type == "cpu":
      torch.use_deterministic_algorithms(True)
   try:
      yield
   finally:
      torch.use_deterministic_algorithms(before, warn_only = warn)


def _features(net, target, features, qualitative, standardize, rows = None):
   """
   Feature matrix of all the nodes, in the network wide order, built on the device of the network:
   dense, or sparse CSR when the dense matrix would have more than :data:`_SPARSE` entries and
   fewer than 10% nonzero (e.g. the many attributes of a long node table, or the dummies of an
   attribute with many values). Missing numeric values are replaced by the mean of their
   attribute. Standardized numeric attributes have zero mean and unit variance in a dense matrix;
   in a sparse one they are only scaled to unit variance, so that zeros stay zeros.

   :param rows: Network wide positions of the rows to keep (the nodes of a hypergraph or
      simplicial complex, without its hyperedges or facets), or ``None`` for all.
   :return: ``(x, names)``: float32 tensor and column names.
   """
   dev, n = net.components[0].edge_index.device, int(net.node_offsets[-1])
   pos = None
   if rows is not None:   # position of every stored row among the kept ones, -1 if dropped
      rows = torch.as_tensor(rows, device = dev)
      pos = torch.full((n,), -1, dtype = torch.long, device = dev)
      pos[rows] = torch.arange(len(rows), device = dev)
      n = len(rows)
   keep = [j for j, a in enumerate(net.node_attributes) if a != target and (features is None or a in features)]
   new = torch.full((len(net.node_attributes),), -1, dtype = torch.long, device = dev)
   new[keep] = torch.arange(len(keep), device = dev)
   k = len(keep)
   if net.node_format == "long":   # numeric attributes as entries (row, column, value)
      parts = []
      for d, o in zip(net.components, net.node_offsets):
         with _quiet_sparse():
            C = d.node_attr.to_sparse_coo().coalesce()
         c = new[C.indices()[1]]
         m = c >= 0
         parts.append((C.indices()[0][m] + int(o), c[m], C.values()[m].double()))
      r, c, v = (torch.cat([p[q] for p in parts]) for q in range(3)) if parts else (torch.zeros(0, dtype = torch.long, device = dev),) * 2 + (torch.zeros(0, dtype = torch.float64, device = dev),)
      if pos is not None:
         r = pos[r]
         r, c, v = r[r >= 0], c[r >= 0], v[r >= 0]
      bad = torch.isnan(v)
      tot = torch.zeros(k, dtype = torch.float64, device = dev).index_add_(0, c[~bad], v[~bad])
      mean = tot / (n - torch.bincount(c[bad], minlength = k)).clamp(min = 1)
      v = torch.where(bad, mean[c], v)
      dense = None
   else:
      dense = torch.cat([d.node_attr[:, keep] for d in net.components]).double() if k else torch.zeros((int(net.node_offsets[-1]), 0), dtype = torch.float64, device = dev)
      dense = dense if rows is None else dense[rows]
      some = ~torch.isnan(dense).all(0)   # attributes without any value are left out
      dense, keep = dense[:, some], [j for j, x in zip(keep, some.tolist()) if x]
      dense = torch.where(torch.isnan(dense), torch.nanmean(dense, 0), dense)
      k = len(keep)
   names = [net.node_attributes[j] for j in keep]
   qr, qc = [np.zeros(0, dtype = np.int64)] * 2
   if qualitative:
      rows, cols = [], []
      for a in net.qualitative_node_attributes:
         if a != target and (features is None or a in features) and not (rows is not None and a == BIPARTITE):   # nodes of a higher-order network: all of one type
            x, y, nm = _dummies(net, a)
            if pos is not None:   # values of the dropped rows only: no column
               x = pos.cpu().numpy()[x]
               x, y = x[x >= 0], y[x >= 0]
               used = np.unique(y)
               y, nm = np.searchsorted(used, y), [nm[j] for j in used]
            rows.append(x)
            cols.append(y + len(names))
            names += nm
      if rows:
         qr, qc = np.concatenate(rows), np.concatenate(cols)
   K = len(names)
   if not K:
      return torch.ones((n, 1), device = dev), ["constant"]
   nnz = (n * k if dense is not None else v.shape[0]) + len(qr)
   sparse = n * K > _SPARSE and nnz < 0.1 * n * K
   if dense is None:   # moments of the long attributes, zeros included
      tot = torch.zeros(k, dtype = torch.float64, device = dev).index_add_(0, c, v)
      sq = torch.zeros(k, dtype = torch.float64, device = dev).index_add_(0, c, v * v)
      mean, sd = tot / n, (sq / n - (tot / n) ** 2).clamp(min = 0).sqrt()
   else:
      mean, sd = dense.mean(0), dense.std(0, unbiased = False)
   sd = torch.where(sd > 0, sd, torch.ones_like(sd))
   if not standardize:
      mean, sd = torch.zeros_like(mean), torch.ones_like(sd)
   qr, qc = torch.as_tensor(qr, device = dev), torch.as_tensor(qc, device = dev)
   if sparse:
      if dense is not None:   # wide attributes are dense columns in a sparse matrix
         r, c = torch.arange(n, device = dev).repeat_interleave(k), torch.arange(k, device = dev).repeat(n)
         v = ((dense - mean) / sd).reshape(-1)
      else:
         v = v / sd[c]
      idx = torch.cat([torch.stack([r, c]), torch.stack([qr, qc])], 1)
      val = torch.cat([v, torch.ones(len(qr), dtype = torch.float64, device = dev)]).float()
      with _quiet_sparse():
         return torch.sparse_coo_tensor(idx, val, (n, K)).coalesce().to_sparse_csr(), names
   X = torch.zeros((n, K), device = dev)
   if dense is None:
      X[:, :k].index_put_((r, c), v.float(), accumulate = True)
      X[:, :k] = (X[:, :k] - mean.float()) / sd.float()
   else:
      X[:, :k] = ((dense - mean) / sd).float()
   X[qr, qc] = 1.
   return X, names


def _target(net, target, task, rows = None):
   """
   Target of all the nodes: class codes (classification) or values (regression), and which nodes
   have one.

   :param rows: Network wide positions of the nodes to keep (see :func:`_features`), or ``None``.
   :return: ``(y, known, classes, task)``.
   """
   if target in net.qualitative_node_attributes:
      vals = _node_values(net, target)
      if net.node_format == "long":   # a set of values per node: one value, or none
         vals = np.array([next(iter(x)) if len(x) == 1 else x if len(x) else np.nan for x in vals] + [None], dtype = object)[:-1]
      if any(isinstance(x, (tuple, list, set, frozenset)) for x in vals):
         raise ValueError(f"target {target!r} has several values for some nodes")
      task = task or "classification"
      if task != "classification":
         raise ValueError(f"target {target!r} is qualitative: only classification is possible")
   else:
      vals = np.concatenate([net._node_vector(target, c).double().cpu().numpy() for c in range(len(net.components))])
      task = task or "regression"
   if rows is not None:
      vals = vals[rows]
   return _labels(vals, task)


def _labels(vals, task):
   """
   Class codes (classification) or values (regression) of a target given by its values, and which
   items have one (see :func:`_target`).
   """
   known = ~pd.isna(pd.Series(vals, dtype = object)).to_numpy()
   if task == "regression":
      return vals.astype(np.float64), known, None, task
   if task != "classification":
      raise ValueError(f"unknown task: {task}")
   codes, classes = pd.factorize(pd.Series(vals[known], dtype = object), sort = True)
   y = np.full(len(vals), -1, dtype = np.int64)
   y[known] = codes
   return y, known, np.asarray(classes, dtype = object), task


def _scores(task, pred, y, classes):
   """
   Prediction scores: accuracy and macro F1 (classification), or mean squared error, mean absolute
   error and :math:`R^2` (regression).
   """
   if not len(y):
      return {}
   if task == "classification":
      f1 = []
      for c in range(len(classes)):
         tp = int(((pred == c) & (y == c)).sum())
         den = int((pred == c).sum()) + int((y == c).sum())
         if den:
            f1.append(2 * tp / den)
      return {"accuracy": float((pred == y).mean()), "f1_macro": float(np.mean(f1))}
   err = pred - y
   var = float(((y - y.mean()) ** 2).mean())
   return {"mse": float((err ** 2).mean()), "mae": float(np.abs(err).mean()), "r2": 1 - float((err ** 2).mean()) / var if var > 0 else float("nan")}


def _gcn_adjacency(edge_index, edge_weight, n):
   """
   Normalized adjacency matrix of GCN, :math:`D^{-1/2} (A + I) D^{-1/2}`, transposed (row = target
   of the messages), as a sparse CSR tensor: computed once, it makes every layer a sparse matrix
   product, several times faster than passing the edges.
   """
   from torch_geometric.nn.conv.gcn_conv import gcn_norm
   from .network import _quiet_sparse
   ei, w = gcn_norm(edge_index, edge_weight, n, add_self_loops = True)
   with _quiet_sparse():
      return (torch.sparse_coo_tensor(torch.stack([ei[1], ei[0]]), w, (n, n)).coalesce().to_sparse_csr(),)


def _mean_adjacency(edge_index, edge_weight, n):
   """
   Row normalized adjacency matrix, transposed (row = target of the messages), as a sparse CSR
   tensor: its product with the features is the (weighted) mean of the features of the
   neighbors, parallel edges counted with their multiplicity, as GraphSAGE's mean aggregation,
   several times faster than passing the edges.
   """
   from .network import _quiet_sparse
   w = torch.ones(edge_index.shape[1], device = edge_index.device) if edge_weight is None else edge_weight.float()
   tot = torch.zeros(n, device = w.device).index_add_(0, edge_index[1], w)
   with _quiet_sparse():
      return (torch.sparse_coo_tensor(torch.stack([edge_index[1], edge_index[0]]), w / tot[edge_index[1]], (n, n)).coalesce().to_sparse_csr(),)


def _edge_list(edge_index, edge_weight, n):
   """
   The edges themselves (attention layers score every edge), with the weights as a one-column
   edge attribute.
   """
   return (edge_index,) if edge_weight is None else (edge_index, edge_weight.float()[:, None])


class _SAGE(torch.nn.Module):
   """
   A ``SAGEConv`` layer (sum over the row normalized adjacency: the mean of the neighbors). With
   sparse features, the features of the neighbors are projected before they are averaged: the
   same model and parameters (both steps are linear), without the averaged sparse matrix, which
   has many more nonzeros than the features.
   """

   def __init__(self, conv):
      super().__init__()
      self.conv = conv

   def forward(self, x, adj):
      c = self.conv
      if x.layout != torch.sparse_csr:
         return c(x, adj)
      out = adj @ F.linear(x, c.lin_l.weight, c.lin_l.bias)
      return out + c.lin_r(x) if c.root_weight else out


class _Stack(torch.nn.Module):
   """
   Graph convolution layers, each followed by its activation and preceded by dropout.
   """

   def __init__(self, layer, sizes, activations, dropout):
      super().__init__()
      self.layers = torch.nn.ModuleList([layer(a, b) for a, b in zip(sizes[:-1], sizes[1:])])
      self.activations = activations
      self.dropout = dropout

   def forward(self, x, graph):
      for conv, act in zip(self.layers, self.activations):
         if x.layout == torch.sparse_csr:   # sparse features: dropout of the stored values
            if self.training and self.dropout > 0:
               x = torch.sparse_csr_tensor(x.crow_indices(), x.col_indices(), F.dropout(x.values(), self.dropout, True), x.shape)
         else:
            x = F.dropout(x, self.dropout, self.training)
         x = conv(x, *graph)
         if act is not None:
            x = act(x)
      return x


def _train(net, layer, prepare, target, channels, activations, features, qualitative, standardize, task, weight, split, epochs, optimizer, learning_rate, weight_decay, dropout, loss, seed, progress, name):
   """
   Body of the node prediction functions, for any layer type ``layer(in_channels,
   out_channels)`` called as ``conv(x, *graph)``, with the tuple ``graph = prepare(edge_index,
   edge_weight, number of nodes)`` computed once.
   """
   y, known, classes, task = _target(net, target, task)
   x, names = _features(net, target, features, qualitative, standardize)
   ei = torch.cat([d.edge_index + int(o) for d, o in zip(net.components, net.node_offsets)], 1)   # the whole network as one graph, nodes in the network wide order
   w = None if weight is None else torch.cat(net._numeric_edge(weight, None)).float()
   graph = prepare(ei, w, net.number_of_nodes())
   return _fit(layer, graph, x, names, y, known, classes, task, np.concatenate(net.node_ids), channels, activations, split, epochs, optimizer, learning_rate, weight_decay, dropout, loss, seed, progress, name)


def _fit(layer, graph, X, names, y, known, classes, task, ids, channels, activations, split, epochs, optimizer, learning_rate, weight_decay, dropout, loss, seed, progress, name, key = "node"):
   """
   Training and evaluation of a stack of layers ``layer(in_channels, out_channels)``, called as
   ``conv(x, *graph)``, predicting ``y`` (from :func:`_target`) from the features ``X``, for
   items (nodes, simplices) with ids ``ids``, named ``key`` in the predictions.
   """
   channels = [int(c) for c in channels]
   if not channels or min(channels) < 1:
      raise ValueError(f"channels must be a non-empty list of positive integers, got {channels}")
   if activations is None:
      activations = ["relu"] * (len(channels) - 1) + [None]
   if len(activations) != len(channels):
      raise ValueError(f"one activation (or None) per layer is needed: {len(channels)} layers, {len(activations)} activations")
   acts = []
   for a in activations:
      if isinstance(a, str) and a not in _ACTIVATIONS:
         raise ValueError(f"unknown activation {a!r}; known: {sorted(_ACTIVATIONS)}")
      acts.append(_ACTIVATIONS[a] if isinstance(a, str) else a)
   if not 0 <= dropout < 1:
      raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
   ftrain, fval = split
   if ftrain <= 0 or fval < 0 or ftrain + fval > 1 + 1e-12:
      raise ValueError(f"split must be (training, validation) fractions with a positive training fraction and a sum of at most 1, got {split}")
   out = len(classes) if task == "classification" else 1
   if channels[-1] != out:
      raise ValueError(f"the last layer must have {out} channels ({'one per class' if task == 'classification' else 'one number'}), got {channels[-1]}")
   if loss is None:
      loss = "cross_entropy" if task == "classification" else "mse"
   if isinstance(loss, str):
      if loss not in _LOSSES:
         raise ValueError(f"unknown loss {loss!r}; known: {sorted(_LOSSES)}")
      loss = _LOSSES[loss]
   if isinstance(optimizer, str):
      if optimizer not in _OPTIMIZERS:
         raise ValueError(f"unknown optimizer {optimizer!r}; known: {sorted(_OPTIMIZERS)}")
      optimizer = _OPTIMIZERS[optimizer]

   dev = X.device
   Y = torch.as_tensor(y, dtype = torch.long if task == "classification" else torch.float32, device = dev)

   # split of the nodes with a target
   rng = np.random.default_rng(seed)
   lab = rng.permutation(np.flatnonzero(known))
   ntr = int(round(ftrain * len(lab)))
   nva = min(int(round(fval * len(lab))), len(lab) - ntr)
   if ntr == 0:
      raise ValueError(f"no {key} to train on: the target is missing everywhere, or the training fraction is too small")
   tr, va = lab[:ntr], lab[ntr:ntr + nva]
   part = np.full(len(y), "unlabeled", dtype = object)
   part[known] = "unused"
   part[tr], part[va] = "training", "validation"
   trt, vat = torch.as_tensor(tr, device = dev), torch.as_tensor(va, device = dev)

   devices = [dev] if dev.type == "cuda" else []
   with torch.random.fork_rng(devices = devices), _deterministic(dev, seed):   # seeded weights and dropout, global state untouched
      if seed is not None:
         torch.manual_seed(seed)
      model = _Stack(layer, [X.shape[1]] + channels, acts, dropout).to(dev)
      opt = optimizer(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
      history = []
      with _bar(epochs, name, progress) as bar:
         for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            out_all = model(X, graph)
            ltr = loss(out_all[trt], Y[trt])
            ltr.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
               out_all = model(X, graph)
               lva = loss(out_all[vat], Y[vat]).item() if len(va) else float("nan")
            history.append((epoch, ltr.item(), lva))
            bar.update()
   model.eval()
   with torch.no_grad():
      out_all = model(X, graph).cpu()
   if task == "classification":
      pc = out_all.argmax(1).numpy()
      pred = classes[pc]
      score = {s: _scores(task, pc[i], y[i], classes) for s, i in (("training", tr), ("validation", va))}
   else:
      pred = out_all[:, 0].numpy().astype(np.float64)
      score = {s: _scores(task, pred[i], y[i], None) for s, i in (("training", tr), ("validation", va))}
   truth = classes[np.where(y >= 0, y, 0)] if task == "classification" else y
   predictions = pd.DataFrame({key: ids, "set": part, "target": np.where(known, truth, np.nan) if task == "regression" else np.where(known, truth.astype(object), np.nan), "prediction": pred})
   if task == "classification":
      prob = torch.softmax(out_all, 1).numpy()
      for j, c in enumerate(classes):
         predictions[f"p_{c}"] = prob[:, j]
   return {"model": model, "scores": pd.DataFrame(score).T, "predictions": predictions, "history": pd.DataFrame(history, columns = ["epoch", "training_loss", "validation_loss"]), "features": names, "task": task}


#: Parameters, return value and errors common to all the models.
_COMMON = """
   ``channels`` gives the output size of each layer, the last one being the output: as many
   channels as classes for classification, 1 for regression. ``activations`` has one entry per
   layer, applied after it: a name (``"relu"``, ``"elu"``, ``"leaky_relu"``, ``"gelu"``,
   ``"tanh"``, ``"sigmoid"``, ``"softmax"``, ``"log_softmax"``), a function of a tensor, or
   ``None`` for no activation; by default ReLU after every layer but the last. The default loss
   for classification, cross entropy, expects raw scores from the last layer: leave its
   activation ``None`` (or use ``"log_softmax"`` with ``loss = "nll"``).

   The task is classification for a qualitative target and regression for a numeric one, unless
   ``task`` says otherwise (``"classification"`` of a numeric target treats each value as a
   class). Features, split and training are described in the module description. The network
   is used as it is: in directed networks, features flow along the edges, from source to target
   (use :func:`~graphoskop.to_undirected` to let them flow both ways).

   :param net: A :class:`~graphoskop.Network`.
   :param target: Name of the node attribute to predict.
   :param channels: Output channels of each layer, e.g. ``[32, 16, 2]``.
   :param activations: One activation per layer (names, functions or ``None``), or ``None`` for
      the default.
   :param qualitative: Whether to include the qualitative node attributes as dummy variables.
   :param features: Names of the node attributes to use as features (``None``: all but the
      target).
   :param standardize: Whether to standardize the numeric features (zero mean, unit variance).
   :param task: ``"classification"``, ``"regression"`` or ``None`` (from the type of the target).
   :param weight: Name of a numeric edge attribute weighting the edges, or ``None`` (see above).
   :param split: Fractions of the nodes with a target in the training and validation sets (the
      rest, if any, is not used).
   :param epochs: Number of training epochs (full passes over the network).
   :param optimizer: ``"adam"``, ``"adamw"``, ``"sgd"``, ``"rmsprop"``, or a
      :class:`torch.optim.Optimizer` class.
   :param learning_rate: Learning rate.
   :param weight_decay: Weight decay (L2 penalty).
   :param dropout: Dropout probability before each layer.
   :param loss: ``"cross_entropy"``, ``"nll"``, ``"mse"``, ``"l1"``, ``"huber"``, a function
      ``loss(output, target)``, or ``None`` (cross entropy for classification, mean squared error
      for regression).
   :param seed: Seed of the split, the initial weights and dropout, or ``None``. With a seed, runs
      repeat exactly on the CPU (PyTorch's deterministic algorithms, during training), and up to
      rounding on a GPU.{extra}
   :param progress: Whether to show a progress bar over the epochs (``None``: as set by
      :func:`~graphoskop.set_progress`; see :doc:`progress`).
   :return: Dictionary with ``"scores"`` (:class:`pandas.DataFrame`, rows ``training`` and
      ``validation``: accuracy and macro F1, or mean squared error, mean absolute error and
      :math:`R^2`), ``"predictions"`` (one row per node: ``node`` id, ``set`` (``training``,
      ``validation``, ``unused`` or ``unlabeled``), ``target``, ``prediction`` and, for
      classification, the probability ``p_<class>`` of each class), ``"history"`` (loss on the
      training and validation sets at each epoch), ``"model"`` (the trained
      :class:`torch.nn.Module`), ``"features"`` (feature names) and ``"task"``.
   :raises ValueError: If the channels, activations, split, loss or optimizer are not valid, or the
      last layer does not match the target.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")       # node attributes: age, income, city
      out = gsk.{name}(net, "city", [32, 16, 4], qualitative = True, seed = 1)   # 4 cities
      out["scores"]                                          # accuracy, macro F1 on both sets
      out["predictions"].query("set == 'unlabeled'")         # cities of the nodes without one
      out["history"].plot(x = "epoch")                       # learning curves
      reg = gsk.{name}(net, "income", [16, 1], activations = ["tanh", None], epochs = 500)
"""

_HEADS = """
   :param heads: Number of attention heads of each layer; their outputs are averaged, so that
      each layer keeps the number of channels given in ``channels``."""


@_higher_order()
def gcn(net, target, channels, activations = None, qualitative = False, features = None, standardize = True, task = None, weight = None, split = (0.8, 0.2), epochs = 200, optimizer = "adam", learning_rate = 0.01, weight_decay = 5e-4, dropout = 0.5, loss = None, seed = None, progress = None):
   from torch_geometric.nn import GCNConv
   return _train(net, lambda a, b: GCNConv(a, b, normalize = False), _gcn_adjacency, target, channels, activations, features, qualitative, standardize, task, weight, split, epochs, optimizer, learning_rate, weight_decay, dropout, loss, seed, progress, "gcn")


@_higher_order()
def graphsage(net, target, channels, activations = None, qualitative = False, features = None, standardize = True, task = None, weight = None, split = (0.8, 0.2), epochs = 200, optimizer = "adam", learning_rate = 0.01, weight_decay = 5e-4, dropout = 0.5, loss = None, seed = None, progress = None):
   from torch_geometric.nn import SAGEConv
   return _train(net, lambda a, b: _SAGE(SAGEConv(a, b, aggr = "sum")), _mean_adjacency, target, channels, activations, features, qualitative, standardize, task, weight, split, epochs, optimizer, learning_rate, weight_decay, dropout, loss, seed, progress, "graphsage")


@_higher_order()
def gat(net, target, channels, activations = None, qualitative = False, features = None, standardize = True, task = None, weight = None, split = (0.8, 0.2), epochs = 200, optimizer = "adam", learning_rate = 0.01, weight_decay = 5e-4, dropout = 0.5, loss = None, seed = None, heads = 1, progress = None):
   from torch_geometric.nn import GATConv
   return _train(net, lambda a, b: GATConv(a, b, heads = heads, concat = False, edge_dim = None if weight is None else 1), _edge_list, target, channels, activations, features, qualitative, standardize, task, weight, split, epochs, optimizer, learning_rate, weight_decay, dropout, loss, seed, progress, "gat")


@_higher_order()
def gatv2(net, target, channels, activations = None, qualitative = False, features = None, standardize = True, task = None, weight = None, split = (0.8, 0.2), epochs = 200, optimizer = "adam", learning_rate = 0.01, weight_decay = 5e-4, dropout = 0.5, loss = None, seed = None, heads = 1, progress = None):
   from torch_geometric.nn import GATv2Conv
   return _train(net, lambda a, b: GATv2Conv(a, b, heads = heads, concat = False, edge_dim = None if weight is None else 1), _edge_list, target, channels, activations, features, qualitative, standardize, task, weight, split, epochs, optimizer, learning_rate, weight_decay, dropout, loss, seed, progress, "gatv2")


gcn.__doc__ = """
   Graph convolutional network (Kipf and Welling, 2017) predicting a node attribute, with the
   ``GCNConv`` layers of PyTorch Geometric: each layer averages the features of the neighbors of
   every node (and its own), normalized by the degrees, and transforms them linearly. With
   ``weight``, the edges are weighted in the average. Each layer is a product with the normalized
   adjacency matrix, computed once as a sparse matrix.
""" + _COMMON.format(extra = "", name = "gcn")

graphsage.__doc__ = """
   GraphSAGE (Hamilton, Ying and Leskovec, 2017) predicting a node attribute, with the ``SAGEConv``
   layers of PyTorch Geometric: each layer combines a linear transformation of the features of
   every node with one of the mean of the features of its neighbors, so that a node keeps its own
   information apart from its neighborhood's. With ``weight``, the mean is weighted. Each layer is
   a product with the row normalized adjacency matrix, computed once as a sparse matrix.
""" + _COMMON.format(extra = "", name = "graphsage")

gat.__doc__ = """
   Graph attention network (Veličković et al., 2018) predicting a node attribute, with the
   ``GATConv`` layers of PyTorch Geometric: each layer averages the transformed features of the
   neighbors of every node (and its own) with weights learned from the features of both ends of
   each edge (attention), with ``heads`` independent attentions averaged. With ``weight``, the
   edge attribute also enters the attention. Attention is computed edge by edge, so a layer is
   slower than in :func:`gcn` or :func:`graphsage`.
""" + _COMMON.format(extra = _HEADS, name = "gat")

gatv2.__doc__ = """
   Graph attention network v2 (Brody, Alon and Yahav, 2022) predicting a node attribute, with the
   ``GATv2Conv`` layers of PyTorch Geometric: as :func:`gat`, but the attention applies its
   nonlinearity before combining the two ends of an edge, so that how much a node attends to a
   neighbor can depend on the node (dynamic attention), which the attention of :func:`gat`
   cannot express.
""" + _COMMON.format(extra = _HEADS, name = "gatv2")


class _HyperConv(torch.nn.Module):
   """
   Hypergraph convolution, :math:`D^{-1} H W B^{-1} H^T X \\Theta + b`, as two products with
   sparse matrices computed once: the mean of the features of the nodes of every hyperedge, then
   the weighted mean of the hyperedges of every node.
   """

   def __init__(self, a, b):
      from torch_geometric.nn.dense.linear import Linear
      super().__init__()
      self.lin = Linear(a, b, bias = False, weight_initializer = "glorot")   # as HypergraphConv
      self.bias = torch.nn.Parameter(torch.zeros(b))

   def forward(self, x, to_edges, to_nodes):
      return to_nodes @ (to_edges @ self.lin(x)) + self.bias


def _hyperedges(net, weight):
   """
   The two matrices of :class:`_HyperConv` for a hypergraph or simplicial complex, nodes and
   hyperedges (or facets) numbered network wide: :math:`B^{-1} H^T` (hyperedges by nodes) and
   :math:`D^{-1} H W` (nodes by hyperedges), with :math:`D = H W 1`. A node in no hyperedge gets
   one holding it alone, so that it keeps its own features.

   :return: ``(to_edges, to_nodes)``, sparse CSR float32 tensors.
   :raises ValueError: If a weight is negative or missing.
   """
   dev = net.components[0].edge_index.device
   k = np.asarray(net.bipartite_split, dtype = np.int64)
   h = np.array([d.num_nodes for d in net.components], dtype = np.int64) - k
   n, H = int(k.sum()), int(h.sum())
   nodes, edges = [], []
   for c, (ko, ho) in enumerate(zip(np.r_[0, np.cumsum(k)[:-1]], np.r_[0, np.cumsum(h)[:-1]])):
      s, t = net._input_edges(c)[0]
      nodes.append(torch.minimum(s, t) + int(ko))
      edges.append(torch.maximum(s, t) - int(k[c]) + int(ho))
   node, edge = torch.cat(nodes), torch.cat(edges)
   alone = torch.nonzero(torch.bincount(node, minlength = n) == 0).reshape(-1)
   node, edge = torch.cat([node, alone]), torch.cat([edge, H + torch.arange(len(alone), device = dev)])
   E = H + len(alone)
   w = torch.ones(E, device = dev)
   if weight is not None:
      get = net.get_hyperedge_attribute if net.higher_order == "hypergraph" else net.get_facet_attribute
      x = torch.cat([v.float() for v in get(weight, None)])
      if not (torch.isfinite(x) & (x >= 0)).all():
         raise ValueError(f"the weights {weight!r} must be given and non negative on every hyperedge")
      w[:H] = x
   size = torch.bincount(edge, minlength = E).float()
   deg = torch.zeros(n, device = dev).index_add_(0, node, w[edge])
   with _quiet_sparse():
      to_edges = torch.sparse_coo_tensor(torch.stack([edge, node]), 1 / size[edge], (E, n)).coalesce().to_sparse_csr()
      to_nodes = torch.sparse_coo_tensor(torch.stack([node, edge]), torch.where(deg[node] > 0, w[edge] / deg[node].clamp(min = 1e-30), 0.), (n, E)).coalesce().to_sparse_csr()
   return to_edges, to_nodes


def _higher_order_net(net, what):
   """
   Check that a network is a hypergraph or a simplicial complex.

   :raises ValueError: If it is not.
   """
   if net.higher_order is None:
      raise ValueError(f"{what} is for hypergraphs and simplicial complexes (gsk.load_hypergraph, gsk.load_simplicial_complex); on a network, use gcn, graphsage, gat or gatv2")


@_higher_order(hypergraph = True, simplicial = True)
def hgnn(net, target, channels, activations = None, qualitative = False, features = None, standardize = True, task = None, weight = None, split = (0.8, 0.2), epochs = 200, optimizer = "adam", learning_rate = 0.01, weight_decay = 5e-4, dropout = 0.5, loss = None, seed = None, progress = None):
   """
   Hypergraph neural network predicting a node attribute of a hypergraph or a simplicial complex,
   with the hypergraph convolution of Feng et al. (2019) and Bai, Zhang and Torr (2021), the
   ``HypergraphConv`` layer of PyTorch Geometric. Each layer averages the features of the nodes of
   every hyperedge, then gives every node the weighted average of its hyperedges and transforms
   it linearly, :math:`D^{-1} H W B^{-1} H^T X \\Theta`, with :math:`H` the incidence matrix,
   :math:`W` the hyperedge weights, :math:`B` the hyperedge sizes and :math:`D = H W 1`. A group
   counts once whatever its size, where in the clique expansion a large group adds many edges;
   a node keeps a large share of its own features (through the means of its hyperedges), so the
   layer smooths less than :func:`gcn` on the clique expansion. On a simplicial complex the
   hyperedges are its facets: its faces add no group beyond them. A node in no hyperedge gets
   one holding it alone, so that it keeps its own features. Directions of directed hypergraphs
   are ignored.

   Each layer is two products with sparse matrices computed once, as in :func:`gcn`, rather than
   ``HypergraphConv`` itself, whose hyperedge weights enter the node degrees only, not the
   averages (same layer and initialization otherwise; without weights, the outputs are the
   same). Its attention variant, which scores nodes against hyperedge features, is left out.

   With ``weight``, a numeric hyperedge (or facet) attribute weights the hyperedges in the average
   of every node; the hyperedges added for nodes in none weigh 1. Features, target, split,
   training and the returned dictionary are those of :func:`gcn`: features and target are node
   attributes (of the nodes only, not of the hyperedges or facets).

   :param net: A hypergraph or a simplicial complex.
   :param target: Name of the node attribute to predict.
   :param channels: Output channels of each layer, e.g. ``[32, 16, 2]`` (see :func:`gcn`).
   :param activations: One activation per layer (names, functions or ``None``), or ``None`` for
      the default (see :func:`gcn`).
   :param qualitative: Whether to include the qualitative node attributes as dummy variables.
   :param features: Names of the node attributes to use as features (``None``: all but the
      target).
   :param standardize: Whether to standardize the numeric features (zero mean, unit variance).
   :param task: ``"classification"``, ``"regression"`` or ``None`` (from the type of the target).
   :param weight: Name of a numeric hyperedge (facet) attribute weighting the hyperedges, or
      ``None``.
   :param split: Fractions of the nodes with a target in the training and validation sets.
   :param epochs: Number of training epochs.
   :param optimizer: ``"adam"``, ``"adamw"``, ``"sgd"``, ``"rmsprop"``, or a
      :class:`torch.optim.Optimizer` class.
   :param learning_rate: Learning rate.
   :param weight_decay: Weight decay (L2 penalty).
   :param dropout: Dropout probability before each layer.
   :param loss: Loss name or function, or ``None`` (see :func:`gcn`).
   :param seed: Seed of the split, the initial weights and dropout, or ``None`` (see :func:`gcn`).
   :param progress: Whether to show a progress bar over the epochs (``None``: as set by
      :func:`~graphoskop.set_progress`; see :doc:`progress`).
   :return: Dictionary with ``"scores"``, ``"predictions"`` (one row per node), ``"history"``,
      ``"model"``, ``"features"`` and ``"task"``, as :func:`gcn`.
   :raises ValueError: If ``net`` is a plain network, a weight is negative or missing, and as
      :func:`gcn`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_hypergraph("authorships.tsv", "authors.tsv")       # papers and their authors
      out = gsk.hgnn(net, "field", [32, 5], seed = 1)                   # 5 fields
      out["scores"]
      out = gsk.hgnn(net, "field", [32, 5], weight = "citations", seed = 1)
   """
   _higher_order_net(net, "hgnn")
   from .higher_order import _node_rows
   rows = _node_rows(net)[0]
   y, known, classes, task = _target(net, target, task, rows)
   x, names = _features(net, target, features, qualitative, standardize, rows)
   ids = np.concatenate([i[:k] for i, k in zip(net.node_ids, net.bipartite_split)])
   return _fit(_HyperConv, _hyperedges(net, weight), x, names, y, known, classes, task, ids, channels, activations, split, epochs, optimizer, learning_rate, weight_decay, dropout, loss, seed, progress, "hgnn")


class _SimplicialConv(torch.nn.Module):
   """
   Simplicial convolution of order :math:`K` (Yang, Isufi and Leus, 2022):
   :math:`X W_0 + \\sum_{p = 1}^K L_d^p X W_{d, p} + \\sum_{p = 1}^K L_u^p X W_{u, p}`, with the lower
   and upper Laplacians :math:`L_d` and :math:`L_u` (``None``: absent).
   """

   def __init__(self, a, b, order):
      super().__init__()
      self.lin = torch.nn.Linear(a, b)
      self.down = torch.nn.ModuleList([torch.nn.Linear(a, b, bias = False) for _ in range(order)])
      self.up = torch.nn.ModuleList([torch.nn.Linear(a, b, bias = False) for _ in range(order)])

   def forward(self, x, down, up):
      out = self.lin(x)
      for L, lins in ((down, self.down), (up, self.up)):
         z = x
         for lin in (lins if L is not None else ()):
            z = L @ z
            out = out + lin(z)
      return out


def _laplacians(net, k, oriented):
   """
   Lower and upper Laplacians of the k-simplices of all the components (block diagonal),
   :math:`B_k^T B_k` and :math:`B_{k + 1} B_{k + 1}^T`, in absolute value unless ``oriented``,
   each divided by its largest absolute row sum (so that its powers stay bounded), as sparse CSR
   float32 tensors, ``None`` when absent.
   """
   from .higher_order import _boundary, _simplices
   dev = net.components[0].edge_index.device
   out = []
   for j, up in ((k, False), (k + 1, True)):
      rows, cols, vals, off_lo, off_hi = [], [], [], 0, 0
      for c in range(len(net.components)):
         r, s, v, shape = _boundary(net, c, j)
         rows.append(r + off_lo)
         cols.append(s + off_hi)
         vals.append(v)
         off_lo, off_hi = off_lo + shape[0], off_hi + shape[1]
      if not off_lo or not off_hi:
         out.append(None)
         continue
      with _quiet_sparse():
         B = torch.sparse_coo_tensor(torch.stack([torch.cat(rows), torch.cat(cols)]), torch.cat(vals).float(), (off_lo, off_hi)).coalesce()
         L = (B @ B.t() if up else B.t() @ B).coalesce()
         v = L.values() if oriented else L.values().abs()
         bound = torch.zeros(L.shape[0], device = dev).index_add_(0, L.indices()[0], v.abs()).max().clamp(min = 1)
         out.append(torch.sparse_coo_tensor(L.indices(), v / bound, L.shape).coalesce().to_sparse_csr())
   return tuple(out)


def _simplex_features(net, k, target, features, qualitative, standardize):
   """
   Feature matrix of all the k-simplices (components in order, simplices as in
   :meth:`~graphoskop.Network.simplices`): the other simplex attributes of dimension k, numeric
   ones with missing values replaced by the mean and standardized, qualitative ones as dummies.

   :return: ``(x, names)``.
   """
   dev = net.components[0].edge_index.device
   comps = range(len(net.components))
   cols, names = [], []
   for a, d in net.simplex_attributes.items():
      if d != k or a == target or (features is not None and a not in features):
         continue
      if net._simplex_attr[a]["qualitative"]:
         if qualitative:
            vals = pd.Series(np.concatenate([net.get_simplex_attribute(a, c) for c in comps]), dtype = object)
            for value in sorted(set(vals.dropna().tolist()), key = str):
               cols.append(torch.as_tensor((vals == value).to_numpy(), dtype = torch.float32, device = dev))
               names.append(f"{a}={value}")
         continue
      v = torch.cat([net.get_simplex_attribute(a, c) for c in comps]).double()
      if torch.isnan(v).all():
         continue
      v = torch.where(torch.isnan(v), torch.nanmean(v), v)
      sd = v.std(unbiased = False)
      cols.append(((v - v.mean()) / (sd if sd > 0 else 1.) if standardize else v).float())
      names.append(a)
   n = sum(len(net.simplices(k, c)) for c in comps)
   if not cols:
      return torch.ones((n, 1), device = dev), ["constant"]
   return torch.stack(cols, 1), names


@_higher_order(simplicial = True)
def scnn(net, target, channels, order = 1, oriented = False, activations = None, qualitative = False, features = None, standardize = True, task = None, split = (0.8, 0.2), epochs = 200, optimizer = "adam", learning_rate = 0.01, weight_decay = 5e-4, dropout = 0.5, loss = None, seed = None, progress = None):
   """
   Simplicial convolutional neural network (Yang, Isufi and Leus, 2022; after Ebli, Defferrard and
   Spreemann, 2020) predicting a simplex attribute of a simplicial complex, such as a value or a
   class of its edges or triangles (see :meth:`~graphoskop.Network.set_simplex_attribute`), from
   the other simplex attributes of the same dimension. Each layer combines, for every
   :math:`k`-simplex, its own features with those of the simplices sharing a face with it (lower
   neighbors, through the lower Laplacian :math:`L_d = B_k^T B_k`) and those sharing a coface
   with it (upper neighbors, through :math:`L_u = B_{k + 1} B_{k + 1}^T`), each with its own weights:
   :math:`X W_0 + \\sum_{p = 1}^K (L_d^p X W_{d, p} + L_u^p X W_{u, p})`. With ``order`` :math:`K`,
   it reaches up to :math:`K` steps away along faces and cofaces. Two edges are lower neighbors
   when they share a node, upper neighbors when they share a filled triangle: the layer tells the
   two apart, which a graph network on the line graph cannot. PyTorch Geometric has no such
   layer: it is implemented here, as sparse matrix products.

   With ``oriented``, the attributes are oriented signals (flows: a value on the edge from ``u`` to
   ``v`` is minus the value from ``v`` to ``u``, in the reference orientation of
   :meth:`~graphoskop.Network.set_simplex_attribute`), and the Laplacians keep the signs of the
   boundary matrices, so that the layers see how flows add up or cancel around nodes and
   triangles. Without it (the default), the attributes are values of the simplices whatever their
   orientation (a weight, a type), and the Laplacians are taken in absolute value, so that the
   arbitrary reference orientation does not matter. Each Laplacian is divided by its largest
   absolute row sum.

   Features are the other numeric simplex attributes of the dimension of the target (missing
   values replaced by the mean, then standardized) and, with ``qualitative``, the qualitative ones
   as dummies; without any, a constant feature. All components are used. Split, training and
   the returned dictionary are those of :func:`gcn`, with simplices instead of nodes: the
   predictions have a ``simplex`` column (:class:`~graphoskop.Simplex` ids) instead of ``node``.

   :param net: A simplicial complex.
   :param target: Name of the simplex attribute to predict (its dimension sets that of the model,
      at least 1).
   :param channels: Output channels of each layer, e.g. ``[32, 2]`` (see :func:`gcn`).
   :param order: Number of powers :math:`K` of each Laplacian in a layer.
   :param oriented: Whether the attributes are oriented signals (see above).
   :param activations: One activation per layer (names, functions or ``None``), or ``None`` for
      the default (see :func:`gcn`).
   :param qualitative: Whether to include the qualitative simplex attributes as dummies.
   :param features: Names of the simplex attributes to use as features (``None``: all those of the
      same dimension but the target).
   :param standardize: Whether to standardize the numeric features.
   :param task: ``"classification"``, ``"regression"`` or ``None`` (from the type of the target).
   :param split: Fractions of the simplices with a target in the training and validation sets.
   :param epochs: Number of training epochs.
   :param optimizer: ``"adam"``, ``"adamw"``, ``"sgd"``, ``"rmsprop"``, or a
      :class:`torch.optim.Optimizer` class.
   :param learning_rate: Learning rate.
   :param weight_decay: Weight decay (L2 penalty).
   :param dropout: Dropout probability before each layer.
   :param loss: Loss name or function, or ``None`` (see :func:`gcn`).
   :param seed: Seed of the split, the initial weights and dropout, or ``None`` (see :func:`gcn`).
   :param progress: Whether to show a progress bar over the epochs (``None``: as set by
      :func:`~graphoskop.set_progress`; see :doc:`progress`).
   :return: Dictionary with ``"scores"``, ``"predictions"`` (one row per simplex), ``"history"``,
      ``"model"``, ``"features"`` and ``"task"``, as :func:`gcn`.
   :raises ValueError: If ``net`` is not a simplicial complex, ``target`` is not a simplex
      attribute, ``order`` is below 1, and as :func:`gcn`.

   Example:

   .. code-block:: python

      import graphoskop as gsk
      import pandas as pd

      net = gsk.load_simplicial_complex("cliques.tsv")
      net.set_simplex_attribute(pd.read_csv("edges.tsv", sep = "\\t"), dimension = 1)   # node_0, node_1, kind, weight
      out = gsk.scnn(net, "kind", [32, 3], seed = 1)                  # edge types from edge weights
      out["predictions"].query("set == 'unlabeled'")                  # one row per edge
   """
   from .higher_order import _simplicial
   _simplicial(net, "scnn")
   if target not in net.simplex_attributes:
      raise ValueError(f"scnn predicts a simplex attribute; simplex attributes: {net.simplex_attributes} (node attributes: use hgnn)")
   if order < 1:
      raise ValueError(f"order must be at least 1, got {order}")
   k = net.simplex_attributes[target]
   comps = range(len(net.components))
   vals = np.concatenate([net.get_simplex_attribute(target, c) if net._simplex_attr[target]["qualitative"] else net.get_simplex_attribute(target, c).double().cpu().numpy() for c in comps])
   qual = net._simplex_attr[target]["qualitative"]
   task = task or ("classification" if qual else "regression")
   if qual and task != "classification":
      raise ValueError(f"target {target!r} is qualitative: only classification is possible")
   y, known, classes, task = _labels(vals, task)
   x, names = _simplex_features(net, k, target, features, qualitative, standardize)
   graph = _laplacians(net, k, oriented)
   ids = np.concatenate([net.simplices(k, c, labels = True) for c in comps])
   return _fit(lambda a, b: _SimplicialConv(a, b, int(order)), graph, x, names, y, known, classes, task, ids, channels, activations, split, epochs, optimizer, learning_rate, weight_decay, dropout, loss, seed, progress, "scnn", "simplex")


#: Standard deviation of the initial vectors, times the square root of their dimension.
_INIT = 0.1


class _Walker:
   """
   Second order random walks of node2vec (Grover and Leskovec, 2016) on the whole network, all
   advancing together on its device. From ``v``, reached from ``t``, the next node ``x`` is a
   neighbor of ``v`` drawn with probability proportional to the edge weight times ``1 / p`` if
   ``x`` is ``t``, 1 if ``x`` is a neighbor of ``t``, ``1 / q`` otherwise; exactly, by rejection:
   a neighbor drawn by weight is accepted with probability its factor over the largest factor.
   """

   def __init__(self, net, weight, p, q):
      n = net.number_of_nodes()
      s = torch.cat([d.edge_index[0] + int(o) for d, o in zip(net.components, net.node_offsets)])
      t = torch.cat([d.edge_index[1] + int(o) for d, o in zip(net.components, net.node_offsets)])
      w = None if weight is None else torch.cat(net._numeric_edge(weight, None)).double()
      if w is not None and (torch.isnan(w).any() or (w < 0).any()):
         raise ValueError("node2vec: weights must be non negative numbers")
      key, order = torch.sort(s * n + t)   # out edges by source, then target: lookups of node pairs
      self.n, self.key, self.t = n, key, t[order]
      self.deg = torch.bincount(s, minlength = n)
      self.first = torch.cumsum(self.deg, 0) - self.deg
      if w is None:
         self.cw = None
         self.k = self.deg.double()
      else:
         self.cw = torch.cumsum(w[order], 0)
         self.k = torch.zeros(n, dtype = torch.float64, device = s.device).index_add_(0, s, w)
         self.before = torch.where(self.first > 0, self.cw[(self.first - 1).clamp(min = 0)], torch.zeros_like(self.k))
      self.f = (1. / p, 1., 1. / q)
      self.top = max(self.f)
      self.plain = p == 1 and q == 1   # every factor 1: plain walks, no rejection

   def _draw(self, u, gen):
      """One neighbor of every node of ``u`` (with edges to follow), drawn by weight."""
      r = torch.rand(len(u), dtype = torch.float64, device = u.device, generator = gen)
      if self.cw is None:
         e = self.first[u] + (r * self.deg[u]).long()
      else:
         e = torch.searchsorted(self.cw, self.before[u] + r * self.k[u], right = True)
      return self.t[torch.minimum(torch.maximum(e, self.first[u]), self.first[u] + self.deg[u] - 1)]

   def walks(self, start, length, gen):
      """
      Walks of ``length`` nodes from ``start``: int64 tensor, one row per walk, -1 after a walk
      stopped at a node without edges to follow.
      """
      W = torch.full((len(start), length), -1, dtype = torch.long, device = start.device)
      W[:, 0] = start
      for j in range(1, length):
         cur, prev = W[:, j - 1], W[:, j - 2] if j > 1 else None
         todo = torch.nonzero((cur >= 0) & (self.k[cur.clamp(min = 0)] > 0)).squeeze(1)
         while len(todo):
            u = cur[todo]
            x = self._draw(u, gen)
            if prev is None or self.plain:
               ok = torch.ones(len(todo), dtype = torch.bool, device = u.device)
            else:
               b = prev[todo]
               pos = torch.searchsorted(self.key, b * self.n + x).clamp(max = len(self.key) - 1)
               f = torch.where(x == b, self.f[0], torch.where(self.key[pos] == b * self.n + x, self.f[1], self.f[2]))
               ok = torch.rand(len(todo), dtype = torch.float64, device = u.device, generator = gen) * self.top < f
            W[todo[ok], j] = x[ok]
            todo = todo[~ok]
      return W


def _skipgram(walker, n, dev, dimensions, walk_length, context_size, walks_per_node, negative_samples, epochs, batch_size, learning_rate, gen, progress, name, starts = None, sampler = None):
   """
   Node vectors trained by skip-gram with negative sampling on walks (see :func:`node2vec`):
   ``walker.walks(start, length, gen)`` gives the walks from the nodes of ``start``, one row per
   walk, -1 after a walk stopped. ``starts``: the nodes walks start from (``None``: all);
   ``sampler(rows, gen)``: the random nodes of the negative examples, ``rows`` by ``walk_length``
   (``None``: uniform over all the nodes).

   :return: Tensor of the vectors, one row per node, on ``dev``.
   """
   devices = [dev] if dev.type == "cuda" else []
   with torch.random.fork_rng(devices = devices):
      torch.manual_seed(int(torch.randint(2 ** 62, (1,), generator = gen, device = dev)))
      emb = torch.nn.Embedding(n, dimensions, sparse = True).to(dev)
      torch.nn.init.normal_(emb.weight, std = _INIT / math.sqrt(dimensions))
   opt = torch.optim.SparseAdam(list(emb.parameters()), lr = learning_rate)
   m = n if starts is None else len(starts)
   steps = -(-m // batch_size)
   I, J = (x.reshape(-1).to(dev) for x in torch.meshgrid(torch.arange(walk_length - context_size + 1), torch.arange(1, context_size), indexing = "ij"))
   J = I + J   # pairs (i, i + k) of the stretches of context_size nodes starting at i
   with _bar(epochs * steps, name, progress) as bar:
      for _ in range(epochs):
         order = torch.randperm(m, device = dev, generator = gen)
         if starts is not None:
            order = starts[order]
         for b in range(steps):
            W = walker.walks(order[b * batch_size:(b + 1) * batch_size].repeat(walks_per_node), walk_length, gen)
            ok = W >= 0
            H = emb(W.clamp(min = 0))   # each node of a walk once, all its dot products at once
            v = ok[:, I] & ok[:, J]
            loss = -F.logsigmoid(torch.bmm(H, H.transpose(1, 2))[:, I, J][v]).sum()
            if negative_samples:   # the same pairs, with the second node random
               rows = len(W) * negative_samples
               Hn = emb(torch.randint(n, (rows, walk_length), device = dev, generator = gen) if sampler is None else sampler(rows, gen))
               dots = torch.bmm(H.repeat(negative_samples, 1, 1), Hn.transpose(1, 2))[:, I, J]
               loss = loss - F.logsigmoid(-dots[ok[:, I].repeat(negative_samples, 1)]).sum() / negative_samples
            opt.zero_grad()
            (loss / v.sum().clamp(min = 1)).backward()   # no value read from the device
            opt.step()
            bar.update()
   return emb.weight.detach()


@_higher_order()
def node2vec(net, dimensions = 128, walk_length = 20, context_size = 10, walks_per_node = 10, p = 1., q = 1., negative_samples = 1, weight = None, epochs = 1, batch_size = 128, learning_rate = 0.025, seed = None, attribute = None, progress = None):
   """
   node2vec node embeddings (Grover and Leskovec, 2016): a vector of ``dimensions`` numbers per
   node, such that nodes that appear close to each other in random walks have similar vectors
   (large dot product). Unlike the models above it is unsupervised: no node attribute is used or
   predicted, only the edges.

   Every epoch, ``walks_per_node`` walks of ``walk_length`` nodes start from every node. Walks are
   of second order: from ``v``, reached from ``t``, the next node is a neighbor ``x`` of ``v``,
   with probability proportional to the weight of the edge times ``1 / p`` if ``x`` is ``t``
   (going back), 1 if ``x`` is also a neighbor of ``t``, ``1 / q`` otherwise (going away). A low
   ``q`` explores far from the start (as a depth first search: similar vectors for nodes of the
   same community); a high ``q`` stays around it (as a breadth first search: similar vectors for
   nodes with similar roles, e.g. hubs); a high ``p`` avoids going back. With ``p = q = 1`` walks
   are plain random walks: this is DeepWalk (Perozzi, Al-Rfou and Skiena, 2014). In directed
   networks walks follow the edges, and stop at a node without out edges.

   The vectors are trained on the device of the network, with the objective of PyTorch
   Geometric's ``Node2Vec`` (skip-gram with negative sampling): in every stretch of
   ``context_size`` consecutive nodes of a walk, the first node should have a large dot product
   with each of the others (positive pairs), and a small one with ``negative_samples`` random
   nodes in their place (negative pairs); the loss is the logistic loss of all pairs, minimized
   with sparse Adam over batches of ``batch_size`` starting nodes. The dot products of each walk
   are computed all at once, and walks are drawn on the device too (by rejection, so exactly with
   the probabilities above). The initial vectors are small (standard deviation
   :math:`0.1 / \\sqrt{dimensions}`, as in word2vec): with the larger ones of PyTorch Geometric
   (standard deviation 1), dot products start saturated and training needs tens of epochs, while
   one epoch is usually enough here. Isolated nodes keep their random initial vectors. In
   multilayer networks walks cross layers through the interlayer edges, and every copy of an
   entity gets its own vector.

   :param net: A :class:`~graphoskop.Network`.
   :param dimensions: Number of numbers of each vector.
   :param walk_length: Number of nodes of each walk, the start included.
   :param context_size: Number of consecutive nodes of a walk in each positive example (at most
      ``walk_length``).
   :param walks_per_node: Number of walks from every node, every epoch.
   :param p: Return parameter: going back is ``1 / p`` times as likely.
   :param q: In-out parameter: going away from the previous node is ``1 / q`` times as likely.
   :param negative_samples: Number of negative examples per positive one.
   :param weight: Name of a numeric edge attribute weighting the edges, or ``None``.
   :param epochs: Number of passes over all the nodes, each with new walks.
   :param batch_size: Number of starting nodes per training step.
   :param learning_rate: Learning rate.
   :param seed: Seed of the walks, the negative examples and the initial vectors, or ``None``.
   :param attribute: Prefix of numeric node attributes ``<attribute>_0``, ``<attribute>_1``, ...
      where to store the vectors (e.g. as features of :func:`gcn`), or ``None`` to only return them.
   :param progress: Whether to show a progress bar over the training steps (``None``: as set by
      :func:`~graphoskop.set_progress`; see :doc:`progress`).
   :return: :class:`pandas.DataFrame` with one row per node, indexed by node id (``(entity,
      layer)`` in multilayer networks), in the network wide node order, and one column per
      dimension.
   :raises ValueError: If a size or parameter is not positive, ``context_size`` is larger than
      ``walk_length``, or weights are negative or missing.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      emb = gsk.node2vec(net, seed = 1)                         # DeepWalk: p = q = 1
      communities = gsk.node2vec(net, q = 0.5, epochs = 5)      # explores away: communities
      roles = gsk.node2vec(net, p = 1, q = 2, attribute = "n2v") # stays close: roles, stored
      out = gsk.gcn(net, "city", [16, 4], features = [f"n2v_{j}" for j in range(128)])
      emb.loc["alice"] @ emb.loc["bob"]                         # similarity of two nodes
   """
   if min(dimensions, walk_length, context_size, walks_per_node, epochs, batch_size) < 1 or negative_samples < 0 or p <= 0 or q <= 0:
      raise ValueError("node2vec: sizes, epochs, p and q must be positive, negative_samples non negative")
   if context_size > walk_length:
      raise ValueError(f"context_size ({context_size}) cannot be larger than walk_length ({walk_length})")
   dev, n = net.components[0].edge_index.device, net.number_of_nodes()
   walker = _Walker(net, weight, p, q)
   gen = torch.Generator(device = dev)
   gen.manual_seed(int(np.random.default_rng(seed).integers(2 ** 62)))
   X = _skipgram(walker, n, dev, dimensions, walk_length, context_size, walks_per_node, negative_samples, epochs, batch_size, learning_rate, gen, progress, "node2vec")
   return _embedding(net, X, attribute)


def _embedding(net, X, attribute):
   """
   Node vectors as a :class:`pandas.DataFrame` indexed by node id, stored as numeric node
   attributes ``<attribute>_j`` if ``attribute`` is not ``None``.

   :param X: Tensor or array, one row per node in the network wide order.
   """
   X = torch.as_tensor(X, device = net.components[0].edge_index.device)
   if attribute is not None:
      for j in range(X.shape[1]):
         net.set_node_attribute(f"{attribute}_{j}", [X[o:o2, j] for o, o2 in zip(net.node_offsets[:-1], net.node_offsets[1:])], None)
   ids = [x for c in range(len(net.components)) for x in net.node_ids[c].tolist()]
   return pd.DataFrame(X.cpu().numpy(), index = pd.Index(ids, tupleize_cols = False, name = "node"))


#: Components with at most this many nodes are solved with dense eigendecompositions.
_DENSE = 2000


def _spectral(net, dimensions, weight, lle):
   """
   Bottom eigenvectors of every component, without the trivial one: of the normalized Laplacian
   (Laplacian eigenmaps, as generalized eigenvectors of :math:`L y = \\lambda D y`) or of
   :math:`(I - W)^T (I - W)` (locally linear embedding), on the undirected network without self
   loops. Components with fewer nodes than ``dimensions + 1`` get zeros in the dimensions they do
   not have; the sign of every vector makes its largest entry (in absolute value) positive.

   :return: NumPy array, one row per node in the network wide order.
   """
   import scipy.sparse as sp
   from scipy.sparse.linalg import LinearOperator, eigsh
   if dimensions < 1:
      raise ValueError(f"dimensions must be positive, got {dimensions}")
   X = np.zeros((net.number_of_nodes(), dimensions))
   for c, d in enumerate(net.components):
      m, o = d.num_nodes, int(net.node_offsets[c])
      if m < 2:
         continue
      s, t = d.edge_index.cpu().numpy()
      w = np.ones(len(s)) if weight is None else net._numeric_edge(weight, c).double().cpu().numpy()
      if np.isnan(w).any() or (w < 0).any():
         raise ValueError("spectral embeddings: weights must be non negative numbers")
      keep = s != t
      A = sp.csr_matrix((w[keep], (s[keep], t[keep])), shape = (m, m))
      if net.directed:
         A = (A + A.T).tocsr()
      deg = np.asarray(A.sum(1)).ravel()
      inv = np.divide(1., deg, out = np.zeros(m), where = deg > 0)
      k = min(dimensions, m - 1)
      if lle:
         IW = (sp.identity(m) - sp.diags(inv) @ A).tocsr()   # each node reconstructed from its neighbors
         if m <= _DENSE or k + 1 >= m - 1:
            B = IW.toarray()
            val, vec = np.linalg.eigh(B.T @ B)
         else:   # M = IW^T IW applied factor by factor: far fewer nonzeros than M itself
            IWT, v0 = IW.T.tocsr(), np.random.default_rng(0).normal(size = m)
            M = LinearOperator((m, m), matvec = lambda x: IWT @ (IW @ x), dtype = float)
            top = eigsh(M, k = 1, which = "LA", v0 = v0, return_eigenvectors = False)[0] * 1.001
            val, vec = eigsh(LinearOperator((m, m), matvec = lambda x: top * x - IWT @ (IW @ x), dtype = float), k = k + 1, which = "LA", v0 = v0)
            val = top - val
         order = np.argsort(val)
      else:
         h = np.sqrt(inv)
         S = (sp.diags(h) @ A @ sp.diags(h)).tocsr()   # normalized adjacency: largest = smallest of the normalized Laplacian
         if m <= _DENSE or k + 1 >= m - 1:
            val, vec = np.linalg.eigh(S.toarray())
         else:
            val, vec = eigsh(S, k = k + 1, which = "LA", v0 = np.random.default_rng(0).normal(size = m))
         order = np.argsort(-val)
         vec = vec * h[:, None]   # generalized eigenvectors: D^(-1/2) u
      vec = vec[:, order[1:k + 1]]   # without the trivial one
      sign = np.sign(vec[np.abs(vec).argmax(0), np.arange(vec.shape[1])])
      X[o:o + m, :vec.shape[1]] = vec * np.where(sign == 0, 1, sign)
   return X


@_higher_order()
def laplacian_eigenmaps(net, dimensions = 16, weight = None, attribute = None):
   """
   Laplacian eigenmaps (Belkin and Niyogi, 2003): node vectors that keep neighbors close, the
   ``dimensions`` smallest non trivial solutions of :math:`L y = \\lambda D y`, with :math:`L = D - A`
   the Laplacian and :math:`D` the degree matrix. They minimize :math:`\\sum_{uv} A_{uv} \\|y_u -
   y_v\\|^2` under :math:`y^T D y = 1` and orthogonality to the constant vector: nodes joined by
   many or heavy edges get similar vectors, and the first dimension is the Fiedler vector of the
   normalized Laplacian, which splits the network into its two most separated parts. Unlike
   :func:`node2vec`, there is no training and nothing random: the vectors are eigenvectors.

   Each component is embedded on its own (its trivial solution is constant on it), so vectors of
   different components are not comparable; a component with fewer than ``dimensions + 1``
   nodes gets zeros in the dimensions it does not have, isolated nodes get zeros. Directed
   networks are treated as undirected (:math:`A + A^T`), self loops are ignored, parallel edges
   are summed. Components of at most 2,000 nodes are solved exactly with a dense
   eigendecomposition, larger ones with the Lanczos method (ARPACK, through SciPy, on the CPU)
   on the normalized adjacency :math:`D^{-1/2} A D^{-1/2}`, whose largest eigenvalues are the
   smallest of the normalized Laplacian. Vectors are ordered by eigenvalue, the smoothest first,
   and the sign of each makes its largest entry positive.
   The vectors are normalized as the eigenvectors they are, so their entries shrink with the size
   of the network (about :math:`1 / \\sqrt{n}`): standardize them before methods sensitive to
   scale, such as penalized regressions (:func:`~graphoskop.embedding_lp` and the models above
   standardize their inputs already).

   :param net: A :class:`~graphoskop.Network`.
   :param dimensions: Number of numbers of each vector.
   :param weight: Name of a numeric edge attribute weighting the edges, or ``None``.
   :param attribute: Prefix of numeric node attributes ``<attribute>_0``, ``<attribute>_1``, ...
      where to store the vectors, or ``None`` to only return them.
   :return: :class:`pandas.DataFrame` with one row per node, indexed by node id, in the network
      wide node order, and one column per dimension.
   :raises ValueError: If ``dimensions`` is not positive or weights are negative or missing.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      emb = gsk.laplacian_eigenmaps(net, dimensions = 2)     # e.g. to draw the network
      side = emb[0] > 0                                      # spectral bisection
      gsk.laplacian_eigenmaps(net, 32, weight = "weight", attribute = "le")
   """
   return _embedding(net, _spectral(net, dimensions, weight, False), attribute)


@_higher_order()
def locally_linear_embedding(net, dimensions = 16, weight = None, attribute = None):
   """
   Locally linear embedding of a network (Roweis and Saul, 2000; in the graph form of Goyal and
   Ferrara, 2018): node vectors such that every node is as close as possible to the weighted mean
   of its neighbors, the ``dimensions`` smallest non trivial eigenvectors of
   :math:`M = (I - W)^T (I - W)`, where :math:`W = D^{-1} A` holds the reconstruction weights of
   each node from its neighbors (proportional to the edge weights). They minimize
   :math:`\\sum_u \\|y_u - \\sum_v W_{uv} y_v\\|^2` with orthonormal dimensions, orthogonal to the
   constant vector. Where Laplacian eigenmaps (:func:`laplacian_eigenmaps`) pull each node towards
   every neighbor, here a node only has to sit at the center of its neighbors, which preserves
   the local geometry of the network (a node between two groups sits between them). There is no
   training and nothing random.

   As in :func:`laplacian_eigenmaps`, each component is embedded on its own (zeros for the
   dimensions a small component does not have, and for isolated nodes), directed networks are
   treated as undirected, self loops are ignored and parallel edges summed. Components of at
   most 2,000 nodes are solved exactly with a dense eigendecomposition, larger ones with the
   Lanczos method (ARPACK, through SciPy, on the CPU) on :math:`c I - M`, with :math:`c` just above
   the largest eigenvalue of :math:`M`, whose largest eigenvalues are the smallest of
   :math:`M`. Vectors are ordered by eigenvalue, and the sign of each makes its largest entry
   positive.
   The vectors are normalized as the eigenvectors they are, so their entries shrink with the size
   of the network (about :math:`1 / \\sqrt{n}`): standardize them before methods sensitive to
   scale, such as penalized regressions (:func:`~graphoskop.embedding_lp` and the models above
   standardize their inputs already).

   :param net: A :class:`~graphoskop.Network`.
   :param dimensions: Number of numbers of each vector.
   :param weight: Name of a numeric edge attribute weighting the edges, or ``None``.
   :param attribute: Prefix of numeric node attributes ``<attribute>_0``, ``<attribute>_1``, ...
      where to store the vectors, or ``None`` to only return them.
   :return: :class:`pandas.DataFrame` with one row per node, indexed by node id, in the network
      wide node order, and one column per dimension.
   :raises ValueError: If ``dimensions`` is not positive or weights are negative or missing.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv")
      emb = gsk.locally_linear_embedding(net, dimensions = 8, attribute = "lle")
      scored = gsk.embedding_lp(net, [("alice", "bob")], [f"lle_{j}" for j in range(8)])
   """
   return _embedding(net, _spectral(net, dimensions, weight, True), attribute)


#: Largest number of entries of the dense pooled matrices of all the graphs (graphs times clusters
#: squared) that diffpool accepts: about 8 GB with the matrices derived from them and gradients.
_POOLED = 2 ** 28

#: Largest number of products (nodes times clusters squared) of the first pooling level computed
#: at once; above it, graph by graph, whose memory grows with nodes times clusters only.
_POOL = 2 ** 26


class _SAGELayer(torch.nn.Module):
   """
   GraphSAGE layer with mean aggregation, :math:`W_r x_u + \\text{mean}_{v \\to u} W_l x_v`, the
   features projected before they are averaged (cheap with sparse features); ``mean`` applies the
   row normalized adjacency.
   """

   def __init__(self, a, b):
      super().__init__()
      self.lin_l, self.lin_r = torch.nn.Linear(a, b), torch.nn.Linear(a, b, bias = False)

   def forward(self, x, mean):
      return self.lin_r(x) + mean(self.lin_l(x))


class _DiffPool(torch.nn.Module):
   """
   DiffPool network: at every level an embedding GNN and an assignment GNN (two GraphSAGE layers
   each), then pooling to the clusters; after the last level an embedding GNN whose node vectors,
   averaged, are the graph embedding; a linear head for the target, if any.
   """

   def __init__(self, inputs, channels, clusters, outputs):
      super().__init__()
      gnn = lambda a, b: torch.nn.ModuleList([_SAGELayer(a, channels), _SAGELayer(channels, b)])
      sizes = [inputs] + [channels] * len(clusters)
      self.embed = torch.nn.ModuleList([gnn(a, channels) for a in sizes])
      self.assign = torch.nn.ModuleList([gnn(a, k) for a, k in zip(sizes, clusters)])
      self.head = torch.nn.Linear(channels, outputs) if outputs else None

   @staticmethod
   def _run(layers, x, mean, last = True):
      x = torch.relu(layers[0](x, mean))
      x = layers[1](x, mean)
      return torch.relu(x) if last else x

   def forward(self, x, mean, adjacency, g, G, sizes, norm2):
      """
      :return: ``(graph embeddings, outputs or None, level 0 node embeddings before their last
         activation, level 0 assignments, link loss, entropy loss)``.
      """
      link, ent = 0., 0.
      Z0, S0 = None, None
      for level, (emb, asg) in enumerate(zip(self.embed, self.assign)):
         Zl = self._run(emb, x, mean, False)
         Z = torch.relu(Zl)
         S = torch.softmax(self._run(asg, x, mean, False), -1)
         K = S.shape[-1]
         if level == 0:   # the network itself, sparse: sums over the nodes of each graph
            Z0, S0 = Zl, S   # before the activation: dot products of either sign, for the reconstruction
            N, dev, AS = S.shape[0], S.device, adjacency(S)
            if N * K * K <= _POOL:   # small: sums of the products of every node
               x = torch.stack([torch.zeros(G, Z.shape[1], device = dev).index_add_(0, g, S[:, k:k + 1] * Z) for k in range(K)], 1)
               A = torch.zeros(G, K, K, device = dev).index_add_(0, g, S[:, :, None] * AS[:, None, :])
               StS = torch.zeros(G, K, K, device = dev).index_add_(0, g, S[:, :, None] * S[:, None, :])
            else:   # large: products of the rows of each graph (nodes are stored graph by graph)
               cut = [0] + torch.cumsum(torch.bincount(g, minlength = G), 0).tolist()
               block = lambda Y: torch.stack([S[a:b].T @ Y[a:b] for a, b in zip(cut[:-1], cut[1:])])
               x, A, StS = block(Z), block(AS), block(S)   # S_g^T Z_g, S_g^T A_g S_g, S_g^T S_g
            err = norm2 - 2 * A.diagonal(0, 1, 2).sum(1) + (StS * StS).sum((1, 2))   # ||A - S S^T||^2 of every graph
            link = link + (err.clamp(min = 0) / (sizes * sizes)).mean()
         else:   # pooled graphs: small and dense
            x = S.transpose(1, 2) @ Z
            SA = S.transpose(1, 2) @ A
            err = ((A - S @ S.transpose(1, 2)) ** 2).sum((1, 2))
            A = SA @ S
            link = link + (err / A.shape[1] ** 2).mean()
         ent = ent + (-(S * torch.log(S + 1e-15)).sum(-1)).mean()
         deg = A.sum(-1, keepdim = True).clamp(min = 1e-12)
         mean = lambda h, A = A, deg = deg: (A @ h) / deg
      h = self._run(self.embed[-1], x, mean).mean(1)
      return h, None if self.head is None else self.head(h), Z0, S0, link, ent


@_higher_order()
def diffpool(net, target = None, clusters = (0.25,), channels = 64, features = None, qualitative = False, standardize = True, task = None, weight = None, split = (0.8, 0.2), epochs = 100, optimizer = "adam", learning_rate = 0.01, weight_decay = 0., loss = None, seed = None, progress = None):
   """
   Graph embeddings by differentiable pooling (DiffPool; Ying, You, Morris, Ren, Hamilton and
   Leskovec, 2018): one vector per graph, where every component of ``net`` is a graph (a network
   whose components are the graphs of a dataset, as in
   :func:`~graphoskop.transactional_graph_mining`), learned from the node attributes and the
   edges.

   At every pooling level, two graph neural networks run on each graph: one gives every node a
   vector :math:`Z`, the other a soft assignment :math:`S` of every node to the clusters of the
   level (a softmax over them). The graph is then replaced by its clusters: their vectors are
   :math:`S^T Z` and their weighted adjacency :math:`S^T A S`, so the next level runs on a
   smaller, coarser graph. After the last level, a last network embeds the clusters, and the mean
   of their vectors is the graph embedding. Every network has two GraphSAGE layers (mean of the
   neighbors, as in the paper) with ReLU activations. ``clusters`` gives the number of clusters of
   each level: a fraction of the size of the level before (of the largest graph, for the first
   level), or a count; the default is one level with 25% of the nodes of the largest graph, as in
   the paper. Every graph gets the same number of clusters, whatever its size.

   The networks are trained together, by gradient descent over all the graphs at once, with two
   losses of the paper that make clusters meaningful: the link loss, :math:`\\|A - S S^T\\|_F^2`
   (nodes joined by an edge should share their clusters), divided by the number of entries of
   :math:`A` and averaged over the graphs and levels, and the entropy of the assignments, averaged
   over the nodes and levels (each node should belong mostly to one cluster). Then:

   * with a ``target``, a graph attribute to predict, the graph embedding goes through a linear
     layer to predict it, and its loss (cross entropy for classification, mean squared error for
     regression, or ``loss``) is added: the graphs with a known target are split into a training
     set, whose loss trains the networks, and a validation set, as in :func:`gcn`;
   * without a target, the cluster losses alone would only train the assignments, not the node
     vectors, so the node vectors of the first level (before their last activation, so that dot
     products can be negative) are also trained to reconstruct the edges, as in graph autoencoders
     (Kipf and Welling, 2016): the logistic loss of the dot products of the two ends of every edge,
     against those of as many random pairs of nodes of the same graph.

   The features are the node attributes, prepared as in :func:`gcn` (numeric ones, qualitative ones
   as dummies with ``qualitative``, sparse when large). Without any, the feature is the (log, standardized)
   in-degree: the mean of constant features over the neighbors would be constant, and the networks
   could not see the structure of the graphs. The first level runs on the sparse network
   (no dense matrix of the nodes of a graph is built): its memory grows with the number of nodes
   times the number of clusters, plus the number of graphs times the square of the number of
   clusters (the pooled graphs, dense).

   DiffPool is meant for datasets of many small graphs (molecules, proteins, ego networks: tens to
   hundreds of nodes each), to compare or classify graphs. It runs on a single large graph too
   (6,000 nodes in about 1.4 s per epoch, 1 GB), but gives a single vector, meaningful only next to
   the vectors of other graphs computed in the same call; to embed the nodes of one network, use
   :func:`node2vec`, :func:`laplacian_eigenmaps` or :func:`locally_linear_embedding`. Since the
   default number of clusters is a share of the largest graph, a dataset with one large component
   gives every graph that many clusters, each pooled into a dense matrix of clusters by clusters:
   with many graphs (isolated nodes are graphs too) this needs a lot of memory, so ``diffpool``
   refuses more than :math:`2^{28}` entries in all (graphs times clusters squared), with a
   :class:`ValueError` that suggests a count of clusters to give in ``clusters`` instead. A graph label given as the name of a node attribute must have one value per
   component (missing on some nodes allowed). With ``weight``, edges are weighted in the
   neighborhood means and in :math:`A`.

   :param net: A :class:`~graphoskop.Network`, one graph per component.
   :param target: The graph attribute to predict: the name of a node attribute with one value per
      component, or one value per component (list, array or :class:`pandas.Series` indexed by
      component), missing values allowed; or ``None`` for unsupervised embeddings.
   :param clusters: Number of clusters of each pooling level: fractions of the size of the level
      before, or counts.
   :param channels: Size of every node vector and of the graph embedding.
   :param features: Names of the node attributes to use as features (``None``: all but the target).
   :param qualitative: Whether to include the qualitative node attributes as dummy variables.
   :param standardize: Whether to standardize the numeric features.
   :param task: ``"classification"``, ``"regression"`` or ``None`` (from the type of the target).
   :param weight: Name of a numeric edge attribute weighting the edges, or ``None``.
   :param split: Fractions of the graphs with a target in the training and validation sets.
   :param epochs: Number of training epochs (full passes over all the graphs).
   :param optimizer: ``"adam"``, ``"adamw"``, ``"sgd"``, ``"rmsprop"``, or a
      :class:`torch.optim.Optimizer` class.
   :param learning_rate: Learning rate.
   :param weight_decay: Weight decay (L2 penalty).
   :param loss: Loss of the target: as in :func:`gcn`, or ``None`` for the default of the task.
   :param seed: Seed of the split, the initial weights and the random pairs, or ``None``.
   :param progress: Whether to show a progress bar over the epochs (``None``: as set by
      :func:`~graphoskop.set_progress`; see :doc:`progress`).
   :return: Dictionary with ``"embeddings"`` (:class:`pandas.DataFrame`, one row per graph,
      indexed by component, one column per channel), ``"assignments"`` (one row per node:
      ``node`` id, ``graph`` and its most likely first level ``cluster``), ``"history"`` (losses
      at each epoch), ``"model"`` (the trained :class:`torch.nn.Module`) and ``"features"``; with a
      target also ``"scores"``, ``"predictions"`` (one row per graph: ``graph``, ``set``,
      ``target``, ``prediction`` and the probability ``p_<class>`` of each class) and ``"task"``,
      as in :func:`gcn`.
   :raises ValueError: If a target given by name has several values in a component, the target
      does not have one value per component, or a parameter is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      molecules = gsk.load_network("bonds.tsv", "atoms.tsv")   # one component per molecule
      out = gsk.diffpool(molecules, "toxic", seed = 1)            # node attribute: one value per molecule
      out["scores"]                                               # accuracy and F1, training and validation
      out["embeddings"]                                           # one vector per molecule
      free = gsk.diffpool(molecules, clusters = (0.25, 0.25))     # unsupervised, two pooling levels
   """
   G = len(net.components)
   dev, n = net.components[0].edge_index.device, net.number_of_nodes()
   off = torch.as_tensor(np.asarray(net.node_offsets), device = dev)
   sizes_np = np.diff(np.asarray(net.node_offsets))
   g = torch.repeat_interleave(torch.arange(G, device = dev), torch.as_tensor(sizes_np, device = dev))
   ks, prev = [], int(sizes_np.max())
   for c in clusters:
      prev = max(1, int(math.ceil(c * prev)) if c < 1 else int(c))
      ks.append(prev)
   if not ks or channels < 1 or epochs < 0:
      raise ValueError("diffpool: clusters must have at least one level, channels and epochs must be positive")
   if G * ks[0] ** 2 > _POOLED:   # every graph is pooled into a dense matrix of its clusters
      raise ValueError(f"diffpool: {G:,} graphs (components) of {ks[0]:,} clusters each need dense pooled matrices of {G * ks[0] ** 2:,} numbers (about {G * ks[0] ** 2 * 32 / 2 ** 30:.0f} GB with their gradients); "
                       f"the default number of clusters is a share of the largest graph ({int(sizes_np.max()):,} nodes): give fewer clusters as a count, e.g. clusters = ({max(2, int(math.sqrt(_POOLED / G)) // 2)},), "
                       f"or keep only the components to embed (e.g. without isolated nodes, which are graphs too)")

   # graph targets
   y, known, classes = None, np.zeros(G, dtype = bool), None
   if target is not None:
      if isinstance(target, str):
         vals = _node_values(net, target)
         if net.node_format == "long" and target in net.qualitative_node_attributes:
            vals = np.array([next(iter(x)) if len(x) == 1 else x if len(x) else np.nan for x in vals] + [None], dtype = object)[:-1]
         per = []
         for c in range(G):
            v = pd.unique(pd.Series(vals[net.node_offsets[c]:net.node_offsets[c + 1]], dtype = object).dropna())
            if len(v) > 1:
               raise ValueError(f"target {target!r} has several values in component {c}: {list(v)[:3]}")
            per.append(v[0] if len(v) else np.nan)
         vals = np.array(per + [None], dtype = object)[:-1]
         numeric = target in net.node_attributes
      else:
         vals = np.array(list(target.reindex(range(G)) if isinstance(target, pd.Series) else target) + [None], dtype = object)[:-1]
         if len(vals) != G:
            raise ValueError(f"the target needs one value per component ({G}), got {len(vals)}")
         numeric = all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in vals if not pd.isna(v))
      known = ~pd.isna(pd.Series(vals, dtype = object)).to_numpy()
      task = task or ("regression" if numeric else "classification")
      if task == "classification":
         codes, classes = pd.factorize(pd.Series(vals[known], dtype = object), sort = True)
         classes = np.asarray(classes, dtype = object)
         y = np.full(G, -1, dtype = np.int64)
         y[known] = codes
      elif task == "regression":
         y = np.where(known, vals, np.nan).astype(np.float64)
      else:
         raise ValueError(f"unknown task: {task}")
      loss = _LOSSES[loss or ("cross_entropy" if task == "classification" else "mse")] if not callable(loss) else loss
   if isinstance(optimizer, str):
      if optimizer not in _OPTIMIZERS:
         raise ValueError(f"unknown optimizer {optimizer!r}; known: {sorted(_OPTIMIZERS)}")
      optimizer = _OPTIMIZERS[optimizer]

   # features, neighborhood means and adjacency of the whole network
   X, names = _features(net, target if isinstance(target, str) else None, features, qualitative, standardize)
   ei = torch.cat([d.edge_index + int(o) for d, o in zip(net.components, net.node_offsets)], 1)
   w = torch.ones(ei.shape[1], device = dev) if weight is None else torch.cat(net._numeric_edge(weight, None)).float()
   if names == ["constant"]:   # means of constant features are constant: the degree instead
      k = torch.log1p(torch.zeros(n, device = dev).index_add_(0, ei[1], torch.ones(ei.shape[1], device = dev)))
      X, names = ((k - k.mean()) / k.std().clamp(min = 1e-12))[:, None], ["degree"]
   M, = _mean_adjacency(ei, w, n)
   with _quiet_sparse():
      Acsr = torch.sparse_coo_tensor(torch.stack([ei[1], ei[0]]), w, (n, n)).coalesce().to_sparse_csr()
   rows = torch.repeat_interleave(torch.arange(n, device = dev), torch.diff(Acsr.crow_indices()))
   norm2 = torch.zeros(G, device = dev).index_add_(0, g[rows], Acsr.values() ** 2)   # ||A||^2 of every graph, parallel edges summed
   sizes = torch.as_tensor(sizes_np, dtype = torch.float32, device = dev)
   keep = ei[0] != ei[1]
   es, et = ei[0][keep], ei[1][keep]

   rng = np.random.default_rng(seed)
   lab = rng.permutation(np.flatnonzero(known))
   ntr = int(round(split[0] * len(lab)))
   tr, va = lab[:ntr], lab[ntr:ntr + min(int(round(split[1] * len(lab))), len(lab) - ntr)]
   if target is not None and ntr == 0:
      raise ValueError("no graph to train on: the target is missing everywhere, or the training fraction is too small")
   trt, vat = torch.as_tensor(tr, device = dev), torch.as_tensor(va, device = dev)
   Y = None if y is None else torch.as_tensor(y, dtype = torch.long if task == "classification" else torch.float32, device = dev)
   gen = torch.Generator(device = dev)
   gen.manual_seed(int(rng.integers(2 ** 62)))

   devices = [dev] if dev.type == "cuda" else []
   with torch.random.fork_rng(devices = devices):
      if seed is not None:
         torch.manual_seed(seed)
      model = _DiffPool(X.shape[1], channels, ks, 0 if target is None else (len(classes) if task == "classification" else 1)).to(dev)
   opt = optimizer(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
   run = lambda: model(X, lambda h: M @ h, lambda h: Acsr @ h, g, G, sizes, norm2)
   history = []
   with _bar(epochs, "diffpool", progress) as bar:
      for epoch in range(epochs):
         model.train()
         opt.zero_grad()
         h, out, Z, S, link, ent = run()
         if target is None:   # edges against random pairs of nodes of the same graph
            wn = off[g[es]] + (torch.rand(len(es), device = dev, generator = gen) * sizes[g[es]]).long()
            rec = -F.logsigmoid((Z[es] * Z[et]).sum(1)).mean() - F.logsigmoid(-(Z[es] * Z[wn]).sum(1)).mean() if len(es) else 0.
            total = rec + link + ent
         else:
            total = loss(out[trt], Y[trt]) + link + ent
         total.backward()
         opt.step()
         if target is None:
            history.append((epoch, float(total.item()), float(rec) if not torch.is_tensor(rec) else rec.item(), link.item(), ent.item()))
         else:
            model.eval()
            with torch.no_grad():
               lva = loss(run()[1][vat], Y[vat]).item() if len(va) else float("nan")
            history.append((epoch, total.item(), lva, link.item(), ent.item()))
         bar.update()
   model.eval()
   with torch.no_grad():
      h, out, Z, S, link, ent = run()
   result = {"model": model, "features": names,
             "embeddings": pd.DataFrame(h.cpu().numpy(), index = pd.RangeIndex(G, name = "graph")),
             "assignments": pd.DataFrame({"node": np.concatenate(net.node_ids), "graph": g.cpu().numpy(), "cluster": S.argmax(1).cpu().numpy()}),
             "history": pd.DataFrame(history, columns = ["epoch", "loss", "reconstruction" if target is None else "validation_loss", "link", "entropy"])}
   if target is None:
      return result
   part = np.full(G, "unlabeled", dtype = object)
   part[known] = "unused"
   part[tr], part[va] = "training", "validation"
   out = out.cpu()
   if task == "classification":
      pc = out.argmax(1).numpy()
      pred = classes[pc]
      score = {s: _scores(task, pc[i], y[i], classes) for s, i in (("training", tr), ("validation", va))}
      truth = np.where(known, classes[np.where(y >= 0, y, 0)].astype(object), np.nan)
   else:
      pred = out[:, 0].numpy().astype(np.float64)
      score = {s: _scores(task, pred[i], y[i], None) for s, i in (("training", tr), ("validation", va))}
      truth = y
   predictions = pd.DataFrame({"graph": np.arange(G), "set": part, "target": truth, "prediction": pred})
   if task == "classification":
      prob = torch.softmax(out, 1).numpy()
      for j, c in enumerate(classes):
         predictions[f"p_{c}"] = prob[:, j]
   result.update(scores = pd.DataFrame(score).T, predictions = predictions, task = task)
   return result


class _LayerWalker:
   """
   Walks on the multilayer graph of :func:`struc2vec` (compiled, on the CPU), in the interface of
   :func:`_skipgram`.
   """

   def __init__(self, L, lptr, lnbr, lcw, up, stay, dev):
      self.args, self.stay, self.dev = (L, lptr, lnbr, lcw, up), stay, dev

   def walks(self, start, length, gen):
      from ._struc2vec import _walks
      seeds = torch.randint(0, 2 ** 62, (len(start),), generator = gen, device = gen.device).cpu().numpy().astype(np.uint64)
      out = np.empty((len(start), length), dtype = np.int64)
      _walks(start.cpu().numpy(), length, *self.args, self.stay, seeds, out)
      return torch.as_tensor(out, device = self.dev)


@_higher_order()
def struc2vec(net, dimensions = 128, walk_length = 80, context_size = 10, walks_per_node = 10, layers = 3, stay = 0.3, candidates = None, negative_samples = 1, epochs = 1, batch_size = 128, learning_rate = 0.025, seed = None, attribute = None, progress = None):
   """
   struc2vec node embeddings (Ribeiro, Saverese and Figueiredo, 2017): vectors that are similar for
   nodes with similar structural roles (hubs, bridges, peripheral nodes...), wherever they are in
   the network, even in different components. :func:`node2vec` instead gives similar vectors to
   nodes that are close to each other.

   The structural distance of two nodes :math:`u` and :math:`v` compares their neighborhoods ring
   by ring: :math:`R_k(u)` is the set of nodes at distance exactly :math:`k` from :math:`u`, and
   :math:`f_k(u, v) = f_{k - 1}(u, v) + g(s(R_k(u)), s(R_k(v)))`, where :math:`s` is the sorted
   degree sequence of a ring and :math:`g` the dynamic time warping distance of two sequences,
   matching degrees :math:`a` and :math:`b` at cost :math:`(\\max(a, b) + 0.5) / (\\min(a, b) +
   0.5) - 1` (the 0.5 lets isolated nodes, of degree 0, be compared). Layer :math:`k` of a
   multilayer graph joins nodes with weight :math:`e^{-f_k(u, v)}`, for :math:`k = 0` (the degrees
   of the two nodes) to ``layers``; a node is joined to itself in the layers above and below, with
   weight :math:`\\log(\\Gamma_k(u) + e)` up, where :math:`\\Gamma_k(u)` is the number of its
   edges in layer :math:`k` heavier than the mean edge of the layer, and 1 down. Random walks
   start at layer 0 of every node; at every step a walk moves within its layer with probability
   ``stay``, to a neighbor drawn by edge weight, or else changes layer, up or down by these
   weights. The vectors are then trained on the walks as in :func:`node2vec` (skip-gram with
   negative sampling).

   As in the paper, three approximations make this feasible on large networks: degree sequences
   are compressed into distinct degrees with their counts (a match then costs the cost above
   times the larger count); only ``layers`` layers are used (rings up to that distance); and each
   node is compared only with the ``candidates`` nodes closest to it in the order of the degrees
   (half with lower or equal degree, half with higher or equal), :math:`2 \\lceil \\log_2 n \\rceil`
   by default, or with all the other nodes when the network has at most 2,000 nodes. Rings come
   from breadth first searches from every node, and distances from dynamic time warping of every
   pair of candidates, both compiled and in parallel on the CPU threads; walks are drawn on the
   CPU too, and the vectors are trained on the device of the network. Training takes most of the
   time, and grows with ``walk_length``: the default, 80 as in the paper, gives slightly better
   vectors than 20 (the default of :func:`node2vec`), which is about 4 times faster.

   The network is used as undirected and simple (directions, weights, self loops and parallel
   edges ignored): roles here are made of degrees. Rings of a node never leave its component, but
   nodes of different components are compared like any others.

   :param net: A :class:`~graphoskop.Network`.
   :param dimensions: Number of numbers of each vector.
   :param walk_length: Number of nodes of each walk, the start included.
   :param context_size: Number of consecutive nodes of a walk in each positive example.
   :param walks_per_node: Number of walks from every node, every epoch.
   :param layers: Largest ring distance compared (layers 0 to ``layers``).
   :param stay: Probability that a walk moves within its layer at each step.
   :param candidates: Number of nodes each node is compared with (the closest in degree), or
      ``None`` for :math:`2 \\lceil \\log_2 n \\rceil` (all nodes with at most 2,000 nodes).
   :param negative_samples: Number of negative examples per positive one.
   :param epochs: Number of passes over all the nodes, each with new walks.
   :param batch_size: Number of starting nodes per training step.
   :param learning_rate: Learning rate.
   :param seed: Seed of the walks, the negative examples and the initial vectors, or ``None``.
   :param attribute: Prefix of numeric node attributes ``<attribute>_0``, ``<attribute>_1``, ...
      where to store the vectors, or ``None`` to only return them.
   :param progress: Whether to show progress bars over the rings, the distances and the training
      steps (``None``: as set by :func:`~graphoskop.set_progress`; see :doc:`progress`).
   :return: :class:`pandas.DataFrame` with one row per node, indexed by node id, in the network
      wide node order, and one column per dimension.
   :raises ValueError: If a size or parameter is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("airports.tsv")
      roles = gsk.struc2vec(net, seed = 1)                            # hubs near hubs, wherever they are
      roles = gsk.struc2vec(net, layers = 5, candidates = 50, attribute = "s2v")
   """
   from ._struc2vec import _distances, _ring_fill, _ring_lengths
   if min(dimensions, walk_length, context_size, walks_per_node, epochs, batch_size) < 1 or negative_samples < 0 or layers < 0 or not 0 < stay <= 1:
      raise ValueError("struc2vec: sizes and epochs must be positive, layers and negative_samples non negative, stay in (0, 1]")
   if context_size > walk_length:
      raise ValueError(f"context_size ({context_size}) cannot be larger than walk_length ({walk_length})")
   dev, n, L = net.components[0].edge_index.device, net.number_of_nodes(), layers + 1

   # undirected simple graph, network wide
   ei = torch.cat([d.edge_index + int(o) for d, o in zip(net.components, net.node_offsets)], 1).cpu().numpy()
   a, b = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
   key = np.unique(a[a != b] * n + b[a != b])
   s, t = np.r_[key // n, key % n], np.r_[key % n, key // n]
   o = np.argsort(s, kind = "stable")
   nbr = t[o]
   ptr = np.zeros(n + 1, dtype = np.int64)
   ptr[1:] = np.cumsum(np.bincount(s, minlength = n))
   deg = np.diff(ptr)

   # compressed degree sequences of the rings, then distances of the candidate pairs
   chunks = lambda m: max(1, min(100, m // 1000))
   nodes = np.arange(n, dtype = np.int64)
   lens = np.zeros((n, L), dtype = np.int64)
   with _bar(2 * n, "struc2vec: rings", progress) as bar:
      for part in np.array_split(nodes, chunks(n)):   # contiguous ranges: slices are views
         _ring_lengths(ptr, nbr, deg, part, L, lens[part[0]:part[-1] + 1])
         bar.update(len(part))
      sptr = np.zeros(n * L + 1, dtype = np.int64)
      sptr[1:] = np.cumsum(lens.reshape(-1))
      sdeg, scnt = np.empty(sptr[-1], dtype = np.int64), np.empty(sptr[-1], dtype = np.int64)
      for part in np.array_split(nodes, chunks(n)):
         _ring_fill(ptr, nbr, deg, part, L, sptr[part[0] * L:(part[-1] + 1) * L + 1].copy() if len(part) else sptr[:1], sdeg, scnt)
         bar.update(len(part))
   order = np.lexsort((nodes, deg))   # by degree
   if candidates is None and n <= 2000:
      h = n - 1   # all pairs
   else:   # h before and h after in the degree order
      h = max(1, min(math.ceil((2 * math.ceil(math.log2(max(n, 2))) if candidates is None else int(candidates)) / 2), n - 1))
   us = np.concatenate([order[:n - j] for j in range(1, h + 1)]) if n > 1 else np.zeros(0, dtype = np.int64)
   vs = np.concatenate([order[j:] for j in range(1, h + 1)]) if n > 1 else np.zeros(0, dtype = np.int64)
   D = np.empty((len(us), L))
   with _bar(len(us), "struc2vec: distances", progress) as bar:
      for part in np.array_split(np.arange(len(us)), chunks(len(us))):
         if len(part):
            _distances(us[part], vs[part], L, sptr, sdeg, scnt, D[part[0]:part[-1] + 1])
         bar.update(len(part))

   # layer graphs: weights exp(-f), each node's row relative to its closest neighbor (no underflow)
   lptr = np.zeros((L, n + 1), dtype = np.int64)
   lnbr, lcw, up, base = [], [], np.zeros((L, n)), 0
   has = np.zeros((L, n), dtype = bool)
   for k in range(L):
      ok = np.isfinite(D[:, k])
      x, y, f = np.r_[us[ok], vs[ok]], np.r_[vs[ok], us[ok]], np.r_[D[ok, k], D[ok, k]]
      o = np.lexsort((y, x))
      x, y, f = x[o], y[o], f[o]
      cnt = np.bincount(x, minlength = n)
      lptr[k, 1:] = base + np.cumsum(cnt)
      lptr[k, 0] = base
      first = np.r_[0, np.cumsum(cnt)[:-1]][x]
      rowmin = np.full(n, np.inf)
      np.minimum.at(rowmin, x, f)
      w = np.exp(-(f - rowmin[x]))
      cw = np.cumsum(w)
      lcw.append(cw - np.r_[0., cw][first])   # cumulative within each row
      lnbr.append(y)
      wt = np.exp(-f)   # true weights, for the heavier-than-average counts
      if len(wt):
         up[k] = np.log(np.bincount(x, weights = wt > wt.mean(), minlength = n) + math.e)
      has[k] = cnt > 0
      base += len(y)
   up[:-1] *= has[1:]   # up only into a layer where the node has edges
   up[-1] = 0.
   lnbr = np.concatenate(lnbr) if lnbr else np.zeros(0, dtype = np.int64)
   lcw = np.concatenate(lcw) if lcw else np.zeros(0)

   walker = _LayerWalker(L, lptr, lnbr.astype(np.int64), lcw, up, float(stay), dev)
   gen = torch.Generator(device = dev)
   gen.manual_seed(int(np.random.default_rng(seed).integers(2 ** 62)))
   X = _skipgram(walker, n, dev, dimensions, walk_length, context_size, walks_per_node, negative_samples, epochs, batch_size, learning_rate, gen, progress, "struc2vec")
   return _embedding(net, X, attribute)


class _MetaWalker:
   """
   Walks that follow a metapath (see :func:`metapath2vec`), all advancing together on the device:
   out edges sorted by ``source * C + code``, where ``code`` identifies the type of the target and
   (with edge types) the relation, so the edges a step can follow from a node are a contiguous
   range, found by binary search; the next node is drawn uniformly in it.
   """

   def __init__(self, s, t, ntype, etype, steps, T, C):
      code = ntype[t] if etype is None else etype * T + ntype[t]   # T node types (the last: none)
      self.C = C   # number of codes, so that keys of different nodes never overlap
      self.key, order = torch.sort(s * self.C + code)
      self.t, self.steps = t[order], steps   # steps: code of every step of the metapath

   def walks(self, start, length, gen):
      W = torch.full((len(start), length), -1, dtype = torch.long, device = start.device)
      W[:, 0] = start
      for j in range(1, length):
         cur = W[:, j - 1]
         q = cur.clamp(min = 0) * self.C + int(self.steps[(j - 1) % len(self.steps)])
         lo, hi = torch.searchsorted(self.key, q), torch.searchsorted(self.key, q, right = True)
         ok = (cur >= 0) & (hi > lo)
         pick = lo + (torch.rand(len(cur), device = cur.device, generator = gen) * (hi - lo)).long()
         W[:, j] = torch.where(ok, self.t[pick.clamp(max = max(len(self.t) - 1, 0))], -1)
      return W


@_higher_order()
def metapath2vec(net, node_type, metapath, edge_type = None, dimensions = 128, walk_length = 20, context_size = 10, walks_per_node = 10, plus = False, negative_samples = 1, epochs = 1, batch_size = 128, learning_rate = 0.025, seed = None, attribute = None, progress = None):
   """
   metapath2vec node embeddings of heterogeneous networks (Dong, Chawla and Swami, 2017), whose
   nodes (and possibly edges) have types: authors, papers and venues, users and items... The
   walks follow a metapath, a sequence of node types such as author, paper, venue, paper, author:
   from a node of the type at one position, the walk moves to a neighbor (out-neighbor in directed
   networks) of the type at the next position, uniformly among them, so that it relates nodes
   through a meaningful chain of relations (two authors through a venue where both published).
   The vectors are then trained on the walks as in :func:`node2vec` (skip-gram with negative
   sampling), with all the types in a single space.

   The node types are a qualitative node attribute, ``node_type``, and the metapath a sequence of
   its values. With ``edge_type``, a qualitative edge attribute, the metapath alternates node types
   and edge types (relations), e.g. ``["author", "writes", "paper", "cites", "paper", "written_by",
   "author"]``, and a step follows only edges of its relation, for networks with several kinds of
   edges between the same types; without it, any edge to a node of the next type will do. The
   metapath must start and end with the same node type, so that walks repeat it (for
   ``walk_length`` nodes); walks start from every node of that type, ``walks_per_node`` times per
   epoch, and stop at a node without a neighbor of the next type (or relation). Nodes of types not
   in the metapath, and nodes no walk reaches, keep their random initial vectors.

   With ``plus = True``, this is metapath2vec++: the random nodes of every negative example are
   drawn among the nodes of the type expected at their position in the walk, instead of among all
   the nodes, so that a vector is contrasted with nodes of its own kind.

   :param net: A :class:`~graphoskop.Network`.
   :param node_type: Name of the qualitative node attribute with the type of every node (one per
      node).
   :param metapath: Sequence of node types (and, with ``edge_type``, edge types between them),
      starting and ending with the same node type.
   :param edge_type: Name of the qualitative edge attribute with the type of every edge, or
      ``None`` (the metapath has only node types).
   :param dimensions: Number of numbers of each vector.
   :param walk_length: Number of nodes of each walk, the start included.
   :param context_size: Number of consecutive nodes of a walk in each positive example.
   :param walks_per_node: Number of walks from every node of the first type, every epoch.
   :param plus: Whether to draw negative examples among the nodes of the expected type
      (metapath2vec++).
   :param negative_samples: Number of negative examples per positive one.
   :param epochs: Number of passes over the starting nodes, each with new walks.
   :param batch_size: Number of starting nodes per training step.
   :param learning_rate: Learning rate.
   :param seed: Seed of the walks, the negative examples and the initial vectors, or ``None``.
   :param attribute: Prefix of numeric node attributes ``<attribute>_0``, ``<attribute>_1``, ...
      where to store the vectors, or ``None`` to only return them.
   :param progress: Whether to show a progress bar over the training steps (``None``: as set by
      :func:`~graphoskop.set_progress`; see :doc:`progress`).
   :return: :class:`pandas.DataFrame` with one row per node, indexed by node id, in the network
      wide node order, and one column per dimension.
   :raises KeyError: If ``node_type`` or ``edge_type`` is not a qualitative attribute.
   :raises ValueError: If the metapath is too short, does not start and end with the same node type,
      does not alternate node and edge types with ``edge_type``, names a type that no node or edge
      has, if a node has several types, or a size is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("bibliography.tsv", "entities.tsv")   # node attribute "kind"
      emb = gsk.metapath2vec(net, "kind", ["author", "paper", "venue", "paper", "author"], seed = 1)
      emb = gsk.metapath2vec(net, "kind", ["author", "paper", "author"], plus = True)
      emb = gsk.metapath2vec(net, "kind", ["author", "writes", "paper", "written_by", "author"], edge_type = "relation")
   """
   if min(dimensions, walk_length, context_size, walks_per_node, epochs, batch_size) < 1 or negative_samples < 0:
      raise ValueError("metapath2vec: sizes and epochs must be positive, negative_samples non negative")
   if context_size > walk_length:
      raise ValueError(f"context_size ({context_size}) cannot be larger than walk_length ({walk_length})")
   if node_type not in net.qualitative_node_attributes:
      raise KeyError(f"no qualitative node attribute named {node_type!r}")
   if edge_type is not None and edge_type not in net.qualitative_edge_attributes:
      raise KeyError(f"no qualitative edge attribute named {edge_type!r}")
   path = list(metapath)
   types, rels = (path, None) if edge_type is None else (path[::2], path[1::2])
   if len(types) < 2 or (edge_type is not None and len(path) % 2 == 0):
      raise ValueError("the metapath needs at least two node types" + ("" if edge_type is None else ", alternating with edge types (an odd number of entries)"))
   if types[0] != types[-1]:
      raise ValueError(f"the metapath must start and end with the same node type, so that walks can repeat it: {types[0]!r} and {types[-1]!r}")
   dev, n = net.components[0].edge_index.device, net.number_of_nodes()

   # node types and edge types as codes
   vals = _node_values(net, node_type)
   if net.node_format == "long":
      if any(len(x) > 1 for x in vals):
         raise ValueError(f"node attribute {node_type!r} gives several types to some nodes")
      vals = np.array([next(iter(x)) if len(x) else np.nan for x in vals] + [None], dtype = object)[:-1]
   nt = pd.Index(pd.unique(pd.Series(vals, dtype = object).dropna()), dtype = object)
   for x in types:
      if x not in nt:
         raise ValueError(f"no node has type {x!r}; types: {list(nt)[:10]}")
   ntype = torch.as_tensor(nt.get_indexer(pd.Series(vals, dtype = object)), device = dev)   # -1: no type
   T = len(nt) + 1
   ntype = torch.where(ntype < 0, T - 1, ntype)
   s = torch.cat([d.edge_index[0] + int(o) for d, o in zip(net.components, net.node_offsets)])
   t = torch.cat([d.edge_index[1] + int(o) for d, o in zip(net.components, net.node_offsets)])
   etype = None
   if edge_type is not None:
      ev = pd.Series(np.concatenate(net.get_edge_attribute(edge_type, None)), dtype = object)
      et = pd.Index(pd.unique(ev.dropna()), dtype = object)
      for r in rels:
         if r not in et:
            raise ValueError(f"no edge has type {r!r}; types: {list(et)[:10]}")
      etype = torch.as_tensor(et.get_indexer(ev), device = dev)
      etype = torch.where(etype < 0, len(et), etype)
   steps = [int(nt.get_loc(x)) if edge_type is None else int(et.get_loc(r)) * T + int(nt.get_loc(x)) for x, r in zip(types[1:], rels or [None] * (len(types) - 1))]
   walker = _MetaWalker(s, t, ntype, etype, steps, T, T if edge_type is None else (len(et) + 1) * T)
   starts = torch.nonzero(ntype == int(nt.get_loc(types[0]))).squeeze(1)

   sampler = None
   if plus:   # negatives of the type of their position in the walk
      period = [int(nt.get_loc(x)) for x in types[:-1]]
      pools = {c: torch.nonzero(ntype == c).squeeze(1) for c in set(period)}

      def sampler(rows, gen):
         cols = [pools[period[j % len(period)]] for j in range(walk_length)]
         return torch.stack([p[torch.randint(len(p), (rows,), device = dev, generator = gen)] for p in cols], 1)

   gen = torch.Generator(device = dev)
   gen.manual_seed(int(np.random.default_rng(seed).integers(2 ** 62)))
   X = _skipgram(walker, n, dev, dimensions, walk_length, context_size, walks_per_node, negative_samples, epochs, batch_size, learning_rate, gen, progress, "metapath2vec", starts, sampler)
   return _embedding(net, X, attribute)


class _VariationalEncoder(torch.nn.Module):
   """
   Encoder of a variational graph autoencoder: GCN layers, then two GCN layers giving the mean and
   the log standard deviation of every node vector.
   """

   def __init__(self, trunk, mu, logstd):
      super().__init__()
      self.trunk, self.mu, self.logstd = trunk, mu, logstd

   def forward(self, x, graph):
      h = self.trunk(x, graph) if self.trunk is not None else x
      return self.mu(h, *graph), self.logstd(h, *graph)


def _autoencoder(net, channels, variational, features, qualitative, standardize, weight, epochs, optimizer, learning_rate, weight_decay, dropout, seed, attribute, progress):
   """
   Body of :func:`gae` and :func:`vgae`.
   """
   from torch_geometric.nn import GAE, GCNConv, VGAE
   channels = [int(c) for c in channels]
   if not channels or min(channels) < 1 or epochs < 0 or not 0 <= dropout < 1:
      raise ValueError("graph autoencoders: channels must be a non-empty list of positive integers, epochs non negative, dropout in [0, 1)")
   if isinstance(optimizer, str):
      if optimizer not in _OPTIMIZERS:
         raise ValueError(f"unknown optimizer {optimizer!r}; known: {sorted(_OPTIMIZERS)}")
      optimizer = _OPTIMIZERS[optimizer]
   dev, n = net.components[0].edge_index.device, net.number_of_nodes()
   X, names = _features(net, None, features, qualitative, standardize)
   if names == ["constant"]:   # no features: one-hot node ids, as in the paper (sparse)
      with _quiet_sparse():
         X = torch.sparse_coo_tensor(torch.arange(n, device = dev).repeat(2, 1), torch.ones(n, device = dev), (n, n)).to_sparse_csr()
      names = ["identity"]
   ei = torch.cat([d.edge_index + int(o) for d, o in zip(net.components, net.node_offsets)], 1)
   w = None if weight is None else torch.cat(net._numeric_edge(weight, None)).float()
   graph = _gcn_adjacency(ei, w, n)
   pos = ei[:, ei[0] != ei[1]]   # the edges to reconstruct
   layer = lambda a, b: GCNConv(a, b, normalize = False)
   sizes = [X.shape[1]] + channels
   devices = [dev] if dev.type == "cuda" else []
   history = []
   state = random.getstate()   # PyTorch Geometric draws the negative pairs with Python's generator
   with torch.random.fork_rng(devices = devices), _deterministic(dev, seed):   # seeded weights, dropout and negative pairs
      if seed is not None:
         torch.manual_seed(seed)
         random.seed(seed)
      if variational:
         trunk = _Stack(layer, sizes[:-1], [torch.relu] * (len(sizes) - 2), dropout) if len(sizes) > 2 else None
         model = VGAE(_VariationalEncoder(trunk, layer(sizes[-2], sizes[-1]), layer(sizes[-2], sizes[-1]))).to(dev)
      else:
         model = GAE(_Stack(layer, sizes, [torch.relu] * (len(sizes) - 2) + [None], dropout)).to(dev)
      opt = optimizer(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
      with _bar(epochs, "vgae" if variational else "gae", progress) as bar:
         for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            z = model.encode(X, graph)
            loss = model.recon_loss(z, pos)
            if variational:
               loss = loss + model.kl_loss() / n
            loss.backward()
            opt.step()
            history.append((epoch, loss.item()))
            bar.update()
   random.setstate(state)   # the caller's random numbers are untouched
   model.eval()
   with torch.no_grad():
      z = model.encode(X, graph)   # VGAE in evaluation: the means
   return {"embeddings": _embedding(net, z, attribute), "history": pd.DataFrame(history, columns = ["epoch", "loss"]), "model": model, "features": names}


#: Parameters, return value and errors common to the graph autoencoders.
_AUTOENCODER = """
   The encoder is a stack of GCN layers (as in :func:`gcn`, with the normalized adjacency
   computed once as a sparse matrix), with ``channels`` giving the output size of each layer, the
   last one being the size of the node vectors (``[32, 16]`` as in the paper), ReLU between them
   and dropout before each. The features are the node attributes, prepared as in :func:`gcn`;
   without any, the features are the one-hot node ids (the identity matrix, sparse), as in the
   paper. The decoder is the inner product: the probability of an edge between :math:`u` and
   :math:`v` is :math:`\\sigma(z_u^T z_v)`, and the loss is the binary cross entropy of all the
   edges (self loops excluded) against as many random pairs of nodes, drawn anew at every epoch,
   with PyTorch Geometric's ``{cls}``. Training is unsupervised and uses all the nodes and edges at
   every epoch: to test the vectors on link prediction, train on the training network of
   :func:`~graphoskop.link_prediction_split` and score the pairs with the dot product of their
   vectors, or with :func:`~graphoskop.embedding_lp`.

   :param net: A :class:`~graphoskop.Network`.
   :param channels: Output channels of each layer, the last one being the size of the node vectors.
   :param features: Names of the node attributes to use as features (``None``: all).
   :param qualitative: Whether to include the qualitative node attributes as dummy variables.
   :param standardize: Whether to standardize the numeric features.
   :param weight: Name of a numeric edge attribute weighting the edges in the GCN layers, or
      ``None``.
   :param epochs: Number of training epochs (full passes over the network).
   :param optimizer: ``"adam"``, ``"adamw"``, ``"sgd"``, ``"rmsprop"``, or a
      :class:`torch.optim.Optimizer` class.
   :param learning_rate: Learning rate.
   :param weight_decay: Weight decay (L2 penalty).
   :param dropout: Dropout probability before each layer.
   :param seed: Seed of the initial weights, dropout and random pairs, or ``None``. With a seed, runs
      repeat exactly on the CPU, and up to rounding on a GPU.
   :param attribute: Prefix of numeric node attributes ``<attribute>_0``, ``<attribute>_1``, ...
      where to store the vectors, or ``None`` to only return them.
   :param progress: Whether to show a progress bar over the epochs (``None``: as set by
      :func:`~graphoskop.set_progress`; see :doc:`progress`).
   :return: Dictionary with ``"embeddings"`` (:class:`pandas.DataFrame`, one row per node,
      indexed by node id, one column per dimension{vec}), ``"history"`` (loss at each epoch),
      ``"model"`` (the trained ``{cls}`` of PyTorch Geometric, whose ``decoder`` scores pairs) and
      ``"features"`` (feature names).
   :raises ValueError: If ``channels``, ``epochs``, ``dropout`` or ``optimizer`` are not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      train, validation = gsk.link_prediction_split(net, seed = 1)
      out = gsk.{name}(train, seed = 1, attribute = "z")
      z = out["embeddings"]
      validation["score"] = (z.loc[validation["source"]].to_numpy() * z.loc[validation["target"]].to_numpy()).sum(1)
      gsk.link_prediction_evaluation(validation)["auc"]
"""


@_higher_order()
def gae(net, channels = (32, 16), features = None, qualitative = False, standardize = True, weight = None, epochs = 200, optimizer = "adam", learning_rate = 0.01, weight_decay = 0., dropout = 0., seed = None, attribute = None, progress = None):
   return _autoencoder(net, channels, False, features, qualitative, standardize, weight, epochs, optimizer, learning_rate, weight_decay, dropout, seed, attribute, progress)


@_higher_order()
def vgae(net, channels = (32, 16), features = None, qualitative = False, standardize = True, weight = None, epochs = 200, optimizer = "adam", learning_rate = 0.01, weight_decay = 0., dropout = 0., seed = None, attribute = None, progress = None):
   return _autoencoder(net, channels, True, features, qualitative, standardize, weight, epochs, optimizer, learning_rate, weight_decay, dropout, seed, attribute, progress)


gae.__doc__ = """
   Graph autoencoder (Kipf and Welling, 2016): node vectors learned by a graph neural network
   (the encoder) so that the edges of the network can be reconstructed from them (the decoder),
   from the node attributes and the edges, without any target.
""" + _AUTOENCODER.format(cls = "GAE", name = "gae", vec = "")

vgae.__doc__ = """
   Variational graph autoencoder (Kipf and Welling, 2016): as :func:`gae`, but the encoder gives a
   distribution of every node vector, a Gaussian with a mean and a standard deviation per
   dimension (the last layer is two GCN layers, one for the means and one for the log standard
   deviations, after the shared ones), and the decoder reconstructs the edges from vectors drawn
   from them. The loss adds the Kullback-Leibler divergence of the distributions from the
   standard normal, divided by the number of nodes, which keeps the vectors in a smooth space
   around the origin.
""" + _AUTOENCODER.format(cls = "VGAE", name = "vgae", vec = ": the means of the distributions")
