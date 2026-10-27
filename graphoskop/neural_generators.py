"""
Neural generators: networks drawn from deep generative models trained on observed networks, which
learn their structure (degrees, clustering, communities, motifs) without a formula for it.

:func:`graphrnn` trains GraphRNN (You, Ying, Ren, Hamilton and Leskovec, 2018), an autoregressive
model that writes a graph node by node, and returns a :class:`GraphRNNFit`, whose
:meth:`~GraphRNNFit.simulate` draws new networks, as :meth:`~graphoskop.ERGMFit.simulate` does for
exponential random graph models (so a fit is also a null model for
:func:`~graphoskop.randomization_test`).
"""

import math

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from ._policy import _higher_order
from .network import load_network
from .progress import _bar

#: Most node steps (nodes of all the graphs of a batch) per training step.
_NODES = 2 ** 15

#: Most edge positions (rows times ``M``) in the loss of a training step.
_POSITIONS = 2 ** 19


class _GraphRNN(torch.nn.Module):
   """
   GraphRNN: a graph-level GRU reads the adjacency vector of every node (its edges to the ``M``
   nodes before it in the BFS order) and gives, for the next node, either the probabilities of its
   ``M`` edges directly (GraphRNN-S, ``simple``) or the initial state of an edge-level GRU that
   decides them one after the other.
   """

   def __init__(self, M, embedding, hidden, layers, edge_embedding, edge_hidden, edge_layers, simple):
      super().__init__()
      self.M, self.simple, self.edge_layers = M, simple, edge_layers
      self.inp = torch.nn.Linear(M, embedding)
      self.gru = torch.nn.GRU(embedding, hidden, layers, batch_first = True)
      mlp = lambda a, b: torch.nn.Sequential(torch.nn.Linear(a, embedding), torch.nn.ReLU(), torch.nn.Linear(embedding, b))
      if simple:
         self.out = mlp(hidden, M)
      else:
         self.to_edge = mlp(hidden, edge_hidden)
         self.edge_inp = torch.nn.Linear(1, edge_embedding)
         self.edge_gru = torch.nn.GRU(edge_embedding, edge_hidden, edge_layers, batch_first = True)
         self.edge_out = torch.nn.Sequential(torch.nn.Linear(edge_hidden, edge_embedding), torch.nn.ReLU(), torch.nn.Linear(edge_embedding, 1))

   def _edge_state(self, h):
      """Initial state of the edge-level GRU: from the graph-level output in its first layer, zeros in the others."""
      first = self.to_edge(h)[None]
      return torch.cat([first, torch.zeros((self.edge_layers - 1,) + first.shape[1:], device = h.device)])

   def states(self, B, T, b, t, c):
      """
      Graph-level outputs (batch, steps, hidden) of a batch with teacher forcing, from its edges in
      sparse form: step ``t + 1`` of sequence ``b`` reads a 1 at position ``c`` (step 0 reads all
      ones). The input layer sums the columns of its weights, so no dense input is built.
      """
      W = self.inp.weight.t()
      z = self.inp.bias.expand(B * T, -1)
      z = z.index_add(0, torch.arange(B, device = W.device) * T, W.sum(0).expand(B, -1))
      keep = t + 1 < T
      z = z.index_add(0, b[keep] * T + t[keep] + 1, W[c[keep]])
      return self.gru(torch.relu(z.view(B, T, -1)))[0]

   def logits(self, h, y):
      """
      Edge logits of the nodes with graph-level outputs ``h`` (rows, hidden) and target edges ``y``
      (rows, M), with teacher forcing.
      """
      if self.simple:
         return self.out(h)
      e = torch.cat([torch.ones_like(y[:, :1]), y[:, :-1]], 1)[:, :, None]   # previous edge, 1 to start
      return self.edge_out(self.edge_gru(self.edge_inp(e), self._edge_state(h))[0])[:, :, 0]

   @torch.no_grad()
   def sample(self, G, max_nodes, gen, bar):
      """
      ``G`` graphs, node by node, until a node without edges (the end) or ``max_nodes`` nodes.

      :return: ``(edges, sizes)``: rows ``(graph, i, j)`` of every edge, and the number of nodes of
         every graph.
      """
      dev = next(self.parameters()).device
      x, state = torch.ones((G, 1, self.M), device = dev), None
      alive, sizes, edges = torch.ones(G, dtype = torch.bool, device = dev), torch.ones(G, dtype = torch.long, device = dev), []
      for i in range(1, max_nodes):
         o, state = self.gru(torch.relu(self.inp(x)), state)
         w = min(i, self.M)   # node i can only link to the nodes before it
         S = torch.zeros((G, self.M), device = dev)
         if self.simple:
            S[:, :w] = torch.bernoulli(torch.sigmoid(self.out(o[:, 0]))[:, :w], generator = gen)
         else:
            es, e = self._edge_state(o[:, 0]), torch.ones((G, 1, 1), device = dev)
            for k in range(w):
               r, es = self.edge_gru(self.edge_inp(e), es)
               S[:, k] = torch.bernoulli(torch.sigmoid(self.edge_out(r)[:, 0, 0]), generator = gen)
               e = S[:, k, None, None]
         alive &= S.any(1)   # a node without edges ends its graph
         bar.update()
         if not alive.any():
            break
         sizes += alive.long()
         g, k = torch.nonzero(S * alive[:, None], as_tuple = True)
         edges.append(torch.stack([g, torch.full_like(g, i), i - 1 - k], 1))
         x = S[:, None]
      E = torch.cat(edges) if edges else torch.zeros((0, 3), dtype = torch.long, device = dev)
      return E.cpu().numpy(), sizes.cpu().numpy()


def _bfs_sequence(A, M, rng):
   """
   Adjacency vectors of a graph in a random BFS order (random start, neighbors in random order), in
   sparse form: row ``i - 1`` holds the edges of node ``i`` to nodes ``i - 1``, ``i - 2``, ...
   ``i - M``, and a last row of zeros (row ``n - 1``) ends the graph.

   :param A: Symmetric CSR adjacency of a connected graph.
   :return: ``(rows, columns)`` of the ones of the ``(n, M)`` matrix.
   """
   n = A.shape[0]
   perm = rng.permutation(n)
   B = A[perm][:, perm]   # relabeling at random randomizes the order of the neighbors
   order = breadth_first_order(B, int(rng.integers(n)), directed = False, return_predecessors = False)
   pos = np.empty(n, dtype = np.int64)
   pos[order] = np.arange(n)
   C = B.tocoo()
   a, b = pos[C.row], pos[C.col]
   keep = (a > b) & (a - b <= M)
   return a[keep] - 1, a[keep] - b[keep] - 1


def _bandwidth(A, rng, times):
   """Largest distance in a random BFS order between the two ends of an edge, over ``times`` orders."""
   n, best = A.shape[0], 0
   for _ in range(times):
      perm = rng.permutation(n)
      B = A[perm][:, perm]
      order = breadth_first_order(B, int(rng.integers(n)), directed = False, return_predecessors = False)
      pos = np.empty(n, dtype = np.int64)
      pos[order] = np.arange(n)
      C = B.tocoo()
      if C.nnz:
         best = max(best, int(np.abs(pos[C.row] - pos[C.col]).max()))
   return best


class GraphRNNFit:
   """
   A trained GraphRNN, as returned by :func:`graphrnn`.

   :ivar model: The trained :class:`torch.nn.Module`.
   :ivar max_previous: ``M``, the number of earlier nodes (in the BFS order) a new node can link to.
   :ivar max_nodes: Largest number of nodes of a generated graph (the largest training graph).
   :ivar dataset: Whether the training graphs were the components of the network (``True``) or its
      largest component alone (``False``).
   :ivar graphs: Number of graphs of a simulated network: as many as the training graphs (one in
      single network mode).
   :ivar history: :class:`pandas.DataFrame` of the loss at every training step.
   """

   def __init__(self, model, max_previous, max_nodes, dataset, graphs, history):
      self.model, self.max_previous, self.max_nodes, self.dataset, self.graphs, self.history = model, max_previous, max_nodes, dataset, graphs, history

   def simulate(self, count = 1, graphs = None, seed = None, progress = None):
      """
      Networks drawn from the trained model: every network has ``graphs`` graphs as its components
      (as many as the training graphs, by default), each written node by node until the model ends
      it (or ``max_nodes`` nodes). Generated networks are undirected, with integer node ids and no
      attributes.

      :param count: Number of networks.
      :param graphs: Number of graphs in every network, or ``None`` for :attr:`graphs`.
      :param seed: Seed of the random numbers, or ``None``.
      :param progress: Whether to show a progress bar over the nodes (``None``: as set by
         :func:`~graphoskop.set_progress`; see :doc:`progress`). Nodes are written one after the
         other, and the full model decides their edges one after the other too: a graph of ``n``
         nodes takes about ``n * M`` small steps (minutes or more for thousands of nodes).
      :return: List of :class:`~graphoskop.Network`.

      Example:

      .. code-block:: python

         fit = gsk.graphrnn(molecules, seed = 1)
         sims = fit.simulate(10, seed = 2)          # 10 networks, as many molecules as the data
         more = fit.simulate(graphs = 1000)[0]      # one network of 1,000 generated molecules
      """
      graphs = self.graphs if graphs is None else int(graphs)
      dev = next(self.model.parameters()).device
      gen = torch.Generator(device = dev)
      gen.manual_seed(int(np.random.default_rng(seed).integers(2 ** 62)))
      self.model.eval()
      out = []
      with _bar(count * (self.max_nodes - 1), "graphrnn simulate", progress) as bar:
         for _ in range(count):
            E, sizes = self.model.sample(graphs, self.max_nodes, gen, bar)
            off = np.r_[0, np.cumsum(sizes)]
            edges = pd.DataFrame({"source": off[E[:, 0]] + E[:, 1], "target": off[E[:, 0]] + E[:, 2]})
            out.append(load_network(edges, pd.DataFrame({"id": np.arange(off[-1])}), device = dev))
      return out


@_higher_order()
def graphrnn(net, dataset = None, simple = False, max_previous = None, steps = 2000, batch_size = 32, learning_rate = 0.003, hidden = 128, layers = 4, embedding = 64, edge_hidden = 16, edge_layers = 4, edge_embedding = 8, seed = None, progress = None):
   """
   Train GraphRNN (You, Ying, Ren, Hamilton and Leskovec, 2018), a deep autoregressive generator of
   graphs, on a network, and return the fitted model, whose :meth:`~GraphRNNFit.simulate` draws new
   networks like it.

   A graph is written as a sequence: its nodes in the order of a breadth first search (from a
   random node, neighbors in random order), and for every new node its edges to the ``M`` nodes
   before it in that order (in a BFS order, the neighbors of a node are always close before it). A
   graph-level recurrent network (a GRU) reads the edges of every node and gives a state for the
   next one; from it, an edge-level GRU decides the edges of the new node one after the other, each
   depending on the ones before (``simple = False``, the full GraphRNN), or a small network gives
   all their probabilities at once, independently (``simple = True``, GraphRNN-S, faster, but
   weaker where edges depend on each other, as in communities). A node without any edge ends the
   graph, so the model also learns the sizes of the graphs (with independent edges, GraphRNN-S
   draws nodes without edges more often than it should, and its graphs tend to be too small). ``M`` (``max_previous``) is by default
   the largest distance, in random BFS orders of the training graphs, between the two ends of an
   edge.

   The training graphs are the components of ``net`` (``dataset = True``: a network whose
   components are the graphs of a dataset, as in the paper), or its largest component alone,
   learned from many random BFS orders of it (``dataset = False``); with ``dataset = None``, the
   components if none holds more than half of the nodes, the largest component otherwise. At every
   training step, ``batch_size`` training graphs are drawn at random, each in a new random BFS
   order (fewer if the graphs are large: at most 32,768 nodes in all, e.g. 5 orders of a network of
   6,000 nodes), and the networks are trained on them with teacher forcing (binary cross entropy of the
   edges, Adam). The defaults are those of the paper (graph-level GRU of 4 layers of 128, edge-level
   of 4 layers of 16, learning rate 0.003). The network is used as undirected and unweighted
   (generated networks are undirected); attributes, node types and layers are ignored.

   Training and sampling run on the device of the network. A sequence has one step per node, and
   every step up to ``M`` edges, so GraphRNN suits graphs of up to a few thousand nodes (the paper:
   up to 2,000): a single large network makes long sequences, with a large ``M`` if it has no
   narrow BFS order (random graphs and expanders have none, nor networks with hubs). To bound the
   memory, the graph-level GRU reads the edges in sparse form, and when the nodes of a batch have
   more than 524,288 edge positions (nodes times ``M``), the loss of the step is computed on a
   random sample of its nodes (the graph-level GRU still reads all of them): an unbiased estimate,
   with the same expected gradient, which needs more steps. Training then takes about 3 GB whatever the size of the network
   (measured on networks of 6,000 nodes and 30,000 edges, with ``M`` from 300 to 4,500).

   :param net: A :class:`~graphoskop.Network`.
   :param dataset: Whether to train on all the components (``True``), on the largest component
      alone (``False``), or to choose (``None``, see above).
   :param simple: Whether to use GraphRNN-S instead of the full GraphRNN.
   :param max_previous: ``M``, or ``None`` to measure it on the training graphs.
   :param steps: Number of training steps.
   :param batch_size: Number of training graphs (BFS orders) per step.
   :param learning_rate: Learning rate.
   :param hidden: Size of the state of the graph-level GRU.
   :param layers: Number of layers of the graph-level GRU.
   :param embedding: Size of the inputs of the graph-level GRU and of its output network.
   :param edge_hidden: Size of the state of the edge-level GRU.
   :param edge_layers: Number of layers of the edge-level GRU.
   :param edge_embedding: Size of the inputs of the edge-level GRU and of its output network.
   :param seed: Seed of the orders, the initial weights and the training, or ``None``.
   :param progress: Whether to show a progress bar over the training steps (``None``: as set by
      :func:`~graphoskop.set_progress`; see :doc:`progress`).
   :return: :class:`GraphRNNFit`.
   :raises ValueError: If a size or parameter is not valid.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      molecules = gsk.load_network("bonds.tsv")                 # one component per molecule
      fit = gsk.graphrnn(molecules, seed = 1)
      fake = fit.simulate(seed = 2)[0]                          # as many generated molecules
      gsk.randomization_test(molecules, gsk.average_clustering, fit, n = 50)
      ego = gsk.graphrnn(one_network, dataset = False, steps = 5000)   # a single network
   """
   if min(steps, batch_size, hidden, layers, embedding, edge_hidden, edge_layers, edge_embedding) < 1 or learning_rate <= 0:
      raise ValueError("graphrnn: sizes, steps and the learning rate must be positive")
   dev = net.components[0].edge_index.device
   sizes = np.array([d.num_nodes for d in net.components])
   if dataset is None:
      dataset = bool(sizes.max() <= sizes.sum() / 2)
   comps = range(len(net.components)) if dataset else [0]
   graphs = []
   for c in comps:
      d = net.components[c]
      s, t = d.edge_index.cpu().numpy()
      A = csr_matrix((np.ones(len(s)), (s, t)), shape = (d.num_nodes,) * 2)
      A = ((A + A.T) > 0).astype(np.float32).tocsr()
      A.setdiag(0)
      A.eliminate_zeros()
      graphs.append(A)
   rng = np.random.default_rng(seed)
   if max_previous is None:   # widest BFS order, over a few orders of the graphs
      times = max(1, 100 // len(graphs))
      max_previous = max(1, max(_bandwidth(A, rng, times) for A in graphs))
   M, max_nodes = int(max_previous), int(sizes[list(comps)].max())
   B = min(batch_size, max(1, _NODES // max_nodes))

   devices = [dev] if dev.type == "cuda" else []
   history = []
   with torch.random.fork_rng(devices = devices):
      if seed is not None:
         torch.manual_seed(seed)
      model = _GraphRNN(M, embedding, hidden, layers, edge_embedding, edge_hidden, edge_layers, simple).to(dev)
      opt = torch.optim.Adam(model.parameters(), lr = learning_rate)
      with _bar(steps, "graphrnn", progress) as bar:
         for step in range(steps):
            ks = rng.integers(len(graphs), size = B)
            lens = np.array([graphs[k].shape[0] for k in ks])
            seqs = [_bfs_sequence(graphs[k], M, rng) for k in ks]
            T = int(lens.max())
            b = np.concatenate([np.full(len(r), i) for i, (r, _) in enumerate(seqs)])
            t, c = np.concatenate([r for r, _ in seqs]), np.concatenate([c for _, c in seqs])
            rows = np.concatenate([i * T + np.arange(n) for i, n in enumerate(lens)])   # every node (and end of graph)
            if len(rows) * M > _POSITIONS:   # the loss on a sample of the nodes
               rows = np.sort(rng.choice(rows, max(1, _POSITIONS // M), replace = False))
            look = np.full(B * T, -1)
            look[rows] = np.arange(len(rows))
            at = look[b * T + t]
            Y = np.zeros((len(rows), M), dtype = np.float32)
            Y[at[at >= 0], c[at >= 0]] = 1
            width = np.minimum(rows % T + 1, M)   # node i links to at most min(i, M) earlier nodes
            mask = (np.arange(M)[None] < width[:, None]).astype(np.float32)
            b, t, c, rows, Y, mask = (torch.as_tensor(a, device = dev) for a in (b, t, c, rows, Y, mask))
            h = model.states(B, T, b, t, c).reshape(B * T, -1)[rows]
            logit = model.logits(h, Y)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, Y, reduction = "none") * mask).sum() / mask.sum().clamp(min = 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            history.append((step, loss.item()))
            bar.update()
   return GraphRNNFit(model, M, max_nodes, dataset, len(graphs), pd.DataFrame(history, columns = ["step", "loss"]))
