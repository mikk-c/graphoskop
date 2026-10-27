"""
Exponential random graph models (ERGMs): the probability of a network :math:`y` is
:math:`P_\\theta(y) = e^{\\theta^T g(y)} / \\kappa(\\theta)`, where :math:`g(y)` is a vector of
network statistics (number of edges, of triangles, of edges between nodes with the same value of
an attribute, ...) and :math:`\\theta` their coefficients: a positive coefficient makes networks
with more of that statistic more likely, all else equal. Equivalently, the log-odds of a tie,
given the rest of the network, is :math:`\\theta^T \\delta_{ij}(y)`, where
:math:`\\delta_{ij}(y)` is the change in the statistics when the tie is added (Robins, Pattison,
Kalish and Lusher, 2007).

Models are written as in R's ``ergm`` package (Hunter, Handcock, Butts, Goodreau and Morris,
2008), with the same terms, statistics and coefficient names, so that results can be compared:

.. code-block:: python

   fit = gsk.ergm(net, "edges + nodematch('city') + nodecov('age') + gwesp(0.5, fixed = TRUE)")

Node attributes enter through terms such as ``nodecov`` (numeric), ``nodefactor`` (one dummy
variable per category of a qualitative attribute, the first category being the baseline),
``nodematch`` (homophily) and ``nodemix`` (mixing between categories); dyadic covariates, such as
the weights of another network, through ``edgecov``. See :func:`ergm` for the list of terms.

Markov chain Monte Carlo on networks is sequential (every step depends on the network left by the
previous one), so it does not benefit from a GPU: change statistics and the sampler are compiled
with Numba and run on the CPU, a few chains in parallel threads, on the dense adjacency matrix.
Models are meant for networks of up to a few thousand nodes, as ERGMs usually are. The first fit
of a session compiles the kernels (cached on disk afterwards). Multilayer and bipartite networks
are not supported yet.
"""

import ast
import math
import re
import warnings

import numba
import numpy as np
import pandas as pd
import torch
from numba import njit, prange

from ._policy import _higher_order
from .network import _memberships


# ------------------------------------------------------------------------------------------------
# Network data
# ------------------------------------------------------------------------------------------------


class _Data:
   """
   A network as a dense 0/1 adjacency matrix, with its node attributes and dyadic covariates.

   :param net: Network, neither multilayer nor bipartite.
   :param covariates: Dictionary of dyadic covariates for ``edgecov``, or ``None``.
   :raises ValueError: If ``net`` is multilayer or bipartite.
   """

   def __init__(self, net, covariates):
      if net.layer_labels is not None or net.bipartite_labels is not None:
         raise ValueError("ERGMs of multilayer or bipartite networks are not supported")
      self.net, self.covariates = net, covariates or {}
      self.directed = net.directed
      self.ids = np.concatenate(net.node_ids)
      self.n = n = len(self.ids)
      S, T = [], []
      for c in range(len(net.components)):
         ei = net._input_edges(c)[0].cpu().numpy()
         S.append(ei[0] + int(net.node_offsets[c]))
         T.append(ei[1] + int(net.node_offsets[c]))
      S, T = np.concatenate(S), np.concatenate(T)
      keep = S != T   # self loops are not part of the model; parallel edges count once
      self.A = np.zeros((n, n), dtype = np.uint8)
      self.A[S[keep], T[keep]] = 1
      if not self.directed:
         self.A |= self.A.T
      self.dyads = n * (n - 1) // (1 if self.directed else 2)

   def attribute(self, name):
      """
      Node attribute over all the nodes, in network wide order.

      :param name: Attribute name.
      :return: float64 :class:`numpy.ndarray` (numeric) or object array (qualitative).
      :raises KeyError: If there is no such attribute.
      :raises ValueError: If some node has several values.
      """
      if name in self.net.qualitative_node_attributes:
         v = _memberships(np.concatenate([np.asarray(x, dtype = object) for x in self.net.get_node_attribute(name, None)]))
         if any(isinstance(x, tuple) for x in v):
            raise ValueError(f"some nodes have several {name!r} values: ERGM terms need one value per node")
         return v
      if name in self.net.node_attributes:
         return torch.cat(self.net.get_node_attribute(name, None)).double().cpu().numpy()
      raise KeyError(f"no node attribute {name!r}")

   def categories(self, name):
      """
      Categories of a node attribute: the code of every node (-1 if missing) and the sorted
      levels (in order of appearance if they cannot be sorted).

      :param name: Attribute name.
      :return: ``(codes, levels)``: int64 array and list.
      """
      v = pd.Series(self.attribute(name), dtype = object)
      try:
         code, levels = pd.factorize(v, sort = True)
      except TypeError:
         code, levels = pd.factorize(v)
      return code.astype(np.int64), [x.item() if isinstance(x, np.generic) else x for x in levels]

   def dyad_matrix(self, name, attribute):
      """
      Dense ``n`` by ``n`` matrix of a dyadic covariate (see ``edgecov`` in :func:`ergm`).

      :param name: Key of the covariate in ``covariates``.
      :param attribute: Edge attribute, for a covariate given as a network, or ``None``.
      :return: float64 :class:`numpy.ndarray`.
      :raises KeyError: If there is no such covariate.
      :raises ValueError: If it has the wrong shape.
      """
      if name not in self.covariates:
         raise KeyError(f"no dyadic covariate {name!r}: give it in covariates = {{{name!r}: ...}}")
      x = self.covariates[name]
      if hasattr(x, "components"):   # another network on the same nodes
         rows = []
         for c in range(len(x.components)):
            ei, ea = x._input_edges(c)
            ids = x.node_ids[c]
            w = ea[:, x.edge_attributes.index(attribute)].double().cpu().numpy() if attribute else np.ones(ei.shape[1])
            rows.append(pd.DataFrame({"s": ids[ei[0].cpu().numpy()], "t": ids[ei[1].cpu().numpy()], "w": w}))
         x, symmetric = pd.concat(rows), not x.directed
      elif isinstance(x, pd.DataFrame):
         x, symmetric = x.iloc[:, :3].set_axis(["s", "t", "w"], axis = 1), False
      else:
         M = np.asarray(x.cpu().numpy() if torch.is_tensor(x) else x, dtype = np.float64)
         if M.shape != (self.n, self.n):
            raise ValueError(f"dyadic covariate {name!r} must be {self.n} by {self.n} (network wide node order), got shape {M.shape}")
         return M
      where = {v: i for i, v in enumerate(self.ids.tolist())}
      i, j = x["s"].map(where).to_numpy(), x["t"].map(where).to_numpy()
      ok = ~(pd.isna(i) | pd.isna(j))
      W = np.zeros((self.n, self.n))
      np.add.at(W, (i[ok].astype(np.int64), j[ok].astype(np.int64)), x["w"].to_numpy(np.float64)[ok])
      if symmetric or not self.directed:
         W = W + W.T - np.diag(np.diag(W))
      return W


# ------------------------------------------------------------------------------------------------
# Terms: specifications for the compiled change statistics
# ------------------------------------------------------------------------------------------------

# kinds of terms, as understood by _delta
(EDGES, NODECOV, NODEFACTOR, NODEMATCH, NODEMIX, ABSDIFF, EDGECOV, MUTUAL, ASYMMETRIC, TRIANGLE,
 TTRIPLE, CTRIPLE, KSTAR, ISTAR, OSTAR, DEGREE, IDEGREE, ODEGREE, CONCURRENT, ISOLATES, GWDEGREE,
 ESP, DSP, GWESP, GWDSP) = range(25)


class _Spec:
   """
   A model term for the compiled kernels: its kind, statistic names, integer and float
   parameters, a node vector, node categories or a dyad matrix, and its statistics on the empty
   graph.
   """

   def __init__(self, kind, names, ints = (), floats = (), vec = None, cat = None, mat = None, empty = None, independent = False, sp = False):
      self.kind, self.names = kind, names
      self.ints, self.floats = np.asarray(ints, dtype = np.int64), np.asarray(floats, dtype = np.float64)
      self.vec, self.cat, self.mat = vec, cat, mat
      self.empty = np.zeros(len(names)) if empty is None else np.asarray(empty, dtype = np.float64)
      self.independent, self.sp = independent, sp


def _num(x):
   """Level as text, as R prints it (8 rather than 8.0)."""
   return f"{x:g}" if isinstance(x, float) else str(x)


def _levels(levels, given, default):
   """
   Selected levels, as in R's ``ergm``: ``None`` for ``default``; ``True`` for all; a negative
   integer ``-k`` to drop the ``k``-th; a list of level values (or of negative integers to drop).

   :return: Positions of the selected levels.
   """
   x = default if given is None else given
   all_ = list(range(len(levels)))
   if x is True:
      return all_
   if isinstance(x, int) and not isinstance(x, bool):
      x = [x]
   x = list(x)
   if all(isinstance(v, int) and not isinstance(v, bool) and v < 0 for v in x):
      drop = {-v - 1 for v in x}
      return [p for p in all_ if p not in drop]
   pos = {v: p for p, v in enumerate(levels)}
   missing = [v for v in x if v not in pos]
   if missing:
      raise ValueError(f"levels not in the attribute: {missing}")
   return [pos[v] for v in x]


def _many(x):
   return list(x) if isinstance(x, (list, tuple)) else [x]


def _undirected_only(d, term):
   if d.directed:
      raise ValueError(f"{term} is for undirected networks (see the directed terms of ergm)")


def _directed_only(d, term):
   if not d.directed:
      raise ValueError(f"{term} is for directed networks")


def _decay(term, decay, fixed):
   if fixed is False:
      warnings.warn(f"{term}: the decay is fixed at {decay} (curved terms, which estimate it, are not supported); in R write fixed = TRUE for the same model")
   return float(decay)


def _gw_tables(d, alpha):
   """Lookup tables over counts ``0 .. n``: :math:`r^k` and :math:`e^\\alpha (1 - r^k)`, :math:`r = 1 - e^{-\\alpha}`."""
   r = (1 - math.exp(-alpha)) ** np.arange(d.n + 1)
   return np.concatenate([r, math.exp(alpha) * (1 - r)])


def _edges(d):
   return _Spec(EDGES, ["edges"], independent = True)


def _nodecov(d, attr, mode = 0):
   x = np.asarray(d.attribute(attr), dtype = np.float64)
   return _Spec(NODECOV, [f"node{['', 'i', 'o'][mode]}cov.{attr}"], [mode], vec = x, independent = True)


def _nodefactor(d, attr, mode = 0, levels = None, base = None):
   code, lev = d.categories(attr)
   keep = _levels(lev, levels if base is None else -base, -1)
   return _Spec(NODEFACTOR, [f"node{['', 'i', 'o'][mode]}factor.{attr}.{_num(lev[p])}" for p in keep], [mode] + keep, cat = code, independent = True)


def _nodematch(d, attr, diff = False, keep = None, levels = None):
   code, lev = d.categories(attr)
   sel = _levels(lev, levels if keep is None else [lev[k - 1] for k in keep], True)
   names = [f"nodematch.{attr}.{_num(lev[p])}" for p in sel] if diff else [f"nodematch.{attr}"]
   return _Spec(NODEMATCH, names, [int(bool(diff))] + sel, cat = code, independent = True)


def _nodemix(d, attr, levels = None, levels2 = None):
   code, lev = d.categories(attr)
   rows = _levels(lev, levels, True)
   if d.directed:   # ordered pairs, sender level first, column by column
      pairs = [(a, b) for b in rows for a in rows]
   else:   # unordered pairs, column by column of the upper triangle
      pairs = [(a, b) for q, b in enumerate(rows) for a in rows[:q + 1]]
   if isinstance(levels2, list) and levels2 and isinstance(levels2[0], (list, tuple)):   # pairs of level values
      pos = {v: p for p, v in enumerate(lev)}
      pairs = [(pos[a], pos[b]) for a, b in levels2]
   else:
      pairs = [pairs[p] for p in _levels(pairs, levels2, -1)]
   if not d.directed:
      pairs = [tuple(sorted(p)) for p in pairs]
   L = len(lev)
   table = np.full(L * L, -1, dtype = np.int64)   # statistic of every pair of codes
   for s, (a, b) in enumerate(pairs):
      table[a * L + b] = s
   return _Spec(NODEMIX, [f"mix.{attr}.{_num(lev[a])}.{_num(lev[b])}" for a, b in pairs], [L] + table.tolist(), cat = code, independent = True)


def _absdiff(d, attr, pow = 1):
   x = np.asarray(d.attribute(attr), dtype = np.float64)
   name = f"absdiff.{attr}" if pow == 1 else f"absdiff{_num(float(pow))}.{attr}"
   return _Spec(ABSDIFF, [name], floats = [pow], vec = x, independent = True)


def _edgecov(d, x, attrname = None):
   return _Spec(EDGECOV, [f"edgecov.{x}" if attrname is None else f"edgecov.{attrname}"], mat = d.dyad_matrix(x, attrname), independent = True)


def _mutual(d):
   _directed_only(d, "mutual")
   return _Spec(MUTUAL, ["mutual"])


def _asymmetric(d):
   _directed_only(d, "asymmetric")
   return _Spec(ASYMMETRIC, ["asymmetric"])


def _triangle(d):
   return _Spec(TRIANGLE, ["triangle"])


def _ttriple(d):
   _directed_only(d, "ttriple")
   return _Spec(TTRIPLE, ["ttriple"])


def _ctriple(d):
   _directed_only(d, "ctriple")
   return _Spec(CTRIPLE, ["ctriple"])


def _kstar(d, k, attr = None):
   _undirected_only(d, "kstar")
   if attr is not None:
      raise ValueError("kstar with attr is not supported")
   return _Spec(KSTAR, [f"kstar{k}" for k in _many(k)], _many(k))


def _star(d, k, kind):
   _directed_only(d, f"{kind}star")
   return _Spec(ISTAR if kind == "i" else OSTAR, [f"{kind}star{k}" for k in _many(k)], _many(k))


def _degree(d, dd, kind = ""):
   ds = _many(dd)
   (_undirected_only if kind == "" else _directed_only)(d, f"{kind}degree")
   return _Spec({"": DEGREE, "i": IDEGREE, "o": ODEGREE}[kind], [f"{kind}degree{v}" for v in ds], ds, empty = [d.n if v == 0 else 0 for v in ds])


def _concurrent(d):
   _undirected_only(d, "concurrent")
   return _Spec(CONCURRENT, ["concurrent"])


def _isolates(d):
   return _Spec(ISOLATES, ["isolates"], empty = [d.n])


def _gwdegree(d, decay, fixed = None, cutoff = None, kind = ""):
   alpha = _decay(f"gw{kind}degree", decay, fixed)
   (_undirected_only if kind == "" else _directed_only)(d, f"gw{kind}degree")
   return _Spec(GWDEGREE, [f"gw{kind}deg.fixed.{_num(alpha)}"], [{"": 0, "i": 1, "o": 2}[kind]], _gw_tables(d, alpha))


def _sp_check(type):
   if type not in (None, "OTP"):
      raise ValueError("only OTP shared partners are supported")


def _gwesp(d, decay, fixed = None, cutoff = None, type = None):
   _sp_check(type)
   alpha = _decay("gwesp", decay, fixed)
   return _Spec(GWESP, [f"gwesp.fixed.{_num(alpha)}"], floats = _gw_tables(d, alpha), sp = True)


def _gwdsp(d, decay, fixed = None, cutoff = None, type = None):
   _sp_check(type)
   alpha = _decay("gwdsp", decay, fixed)
   return _Spec(GWDSP, [f"gwdsp.fixed.{_num(alpha)}"], floats = _gw_tables(d, alpha), sp = True)


def _esp(d, dd, type = None):
   _sp_check(type)
   return _Spec(ESP, [f"esp{v}" for v in _many(dd)], _many(dd), sp = True)


def _dsp(d, dd, type = None):
   _sp_check(type)
   ds = _many(dd)
   return _Spec(DSP, [f"dsp{v}" for v in ds], ds, empty = [d.dyads if v == 0 else 0 for v in ds], sp = True)


_TERMS = {
   "edges": _edges, "nodecov": _nodecov, "nodemain": _nodecov, "nodefactor": _nodefactor,
   "nodematch": _nodematch, "nodemix": _nodemix, "absdiff": _absdiff, "edgecov": _edgecov,
   "nodeicov": lambda d, attr: _nodecov(d, attr, 1), "nodeocov": lambda d, attr: _nodecov(d, attr, 2),
   "nodeifactor": lambda d, attr, **k: _nodefactor(d, attr, 1, **k), "nodeofactor": lambda d, attr, **k: _nodefactor(d, attr, 2, **k),
   "mutual": _mutual, "asymmetric": _asymmetric, "triangle": _triangle, "triangles": _triangle,
   "ttriple": _ttriple, "ctriple": _ctriple, "kstar": _kstar,
   "istar": lambda d, k: _star(d, k, "i"), "ostar": lambda d, k: _star(d, k, "o"),
   "degree": _degree, "idegree": lambda d, dd: _degree(d, dd, "i"), "odegree": lambda d, dd: _degree(d, dd, "o"),
   "concurrent": _concurrent, "isolates": _isolates,
   "gwdegree": _gwdegree, "gwidegree": lambda d, decay, **k: _gwdegree(d, decay, kind = "i", **k), "gwodegree": lambda d, decay, **k: _gwdegree(d, decay, kind = "o", **k),
   "gwesp": _gwesp, "gwdsp": _gwdsp, "esp": _esp, "dsp": _dsp,
}


def _arguments(text):
   """
   Arguments of a term, written as in R: numbers, strings, ``TRUE`` / ``FALSE``, ``c(...)`` vectors
   and ``a:b`` ranges, positional or ``name = value``.

   :param text: Text between the parentheses.
   :return: ``(args, kwargs)``.
   :raises ValueError: If an argument is not a constant.
   """
   text = re.sub(r"\bTRUE\b|\bT\b", "True", re.sub(r"\bFALSE\b|\bF\b", "False", text))
   text = re.sub(r"(-?\d+)\s*:\s*(-?\d+)", lambda m: "[" + ", ".join(map(str, range(int(m.group(1)), int(m.group(2)) + 1))) + "]", text)
   call = ast.parse(f"f({text})", mode = "eval").body

   def value(node):
      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "c":
         return [value(a) for a in node.args]
      if isinstance(node, ast.Name) and node.id == "NULL":
         return None
      try:
         return ast.literal_eval(node)
      except ValueError:
         raise ValueError(f"term arguments must be constants, got {ast.unparse(node)!r}") from None

   return [value(a) for a in call.args], {k.arg: value(k.value) for k in call.keywords}


def _parse(formula):
   """
   Terms of a model formula, as in R: ``"edges + nodematch('city', diff = TRUE) + gwesp(0.5)"``,
   with or without a left hand side ``net ~``.

   :return: List of ``(name, args, kwargs)``.
   :raises ValueError: If the formula is malformed.
   """
   if "~" in formula:
      formula = formula.split("~", 1)[1]
   parts, depth, quote, start = [], 0, None, 0
   for p, ch in enumerate(formula + "+"):
      if quote:
         quote = None if ch == quote else quote
      elif ch in "'\"":
         quote = ch
      elif ch == "(":
         depth += 1
      elif ch == ")":
         depth -= 1
      elif ch == "+" and depth == 0:
         parts.append(formula[start:p].strip())
         start = p + 1
   out = []
   for t in parts:
      m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_.]*)\s*(?:\((.*)\))?", t, re.S)
      if not m:
         raise ValueError(f"cannot read the term {t!r}")
      args, kwargs = _arguments(m.group(2)) if m.group(2) is not None and m.group(2).strip() else ([], {})
      out.append((m.group(1), args, kwargs))
   if not out:
      raise ValueError("the formula has no terms")
   return out


class _Model:
   """
   The terms of a formula on a network, packed into flat arrays for the compiled kernels.

   :param d: :class:`_Data`.
   :param formula: Model formula.
   :raises ValueError: For unknown terms or arguments, or terms that do not apply.
   """

   def __init__(self, d, formula):
      self.d, specs = d, []
      for name, args, kwargs in _parse(formula):
         if name not in _TERMS:
            raise ValueError(f"unknown ERGM term {name!r}; available: {', '.join(sorted(_TERMS))}")
         try:
            specs.append(_TERMS[name](d, *args, **kwargs))
         except TypeError as e:
            raise ValueError(f"{name}: {e}") from None
      self.names = [x for s in specs for x in s.names]
      if len(set(self.names)) < len(self.names):
         raise ValueError(f"repeated statistics: {self.names}")
      self.independent = all(s.independent for s in specs)
      self.sp = any(s.sp for s in specs)
      self.empty = np.concatenate([s.empty for s in specs])
      cum = lambda sizes: np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
      n = d.n
      vecs = [s.vec for s in specs if s.vec is not None] or [np.zeros(n)]
      cats = [s.cat for s in specs if s.cat is not None] or [np.zeros(n, dtype = np.int64)]
      mats = [s.mat for s in specs if s.mat is not None] or [np.zeros((1, 1))]
      index = lambda field: np.cumsum([getattr(s, field) is not None for s in specs]) - 1
      self.packed = (np.array([s.kind for s in specs], dtype = np.int64), cum([len(s.names) for s in specs]),
                     np.concatenate([s.ints for s in specs] + [np.zeros(0, np.int64)]), cum([len(s.ints) for s in specs]),
                     np.concatenate([s.floats for s in specs] + [np.zeros(0)]), cum([len(s.floats) for s in specs]),
                     index("vec").astype(np.int64), np.stack(vecs), index("cat").astype(np.int64), np.stack(cats),
                     index("mat").astype(np.int64), np.stack(mats))

   def state(self, A):
      """
      Two-path counts (if needed) and degrees of a graph, as kept by the kernels.

      :return: ``(SP, out-degrees, in-degrees)``.
      """
      SP = (A.astype(np.int32) @ A.astype(np.int32)) if self.sp else np.zeros((1, 1), dtype = np.int32)
      return SP, A.sum(1).astype(np.int64), A.sum(0).astype(np.int64)

   def stat(self, A):
      """
      Statistics of a graph: their values on the empty graph plus the change statistics of adding
      its ties one by one.

      :param A: 0/1 ``n`` by ``n`` adjacency matrix (symmetric if undirected).
      :return: float64 array.
      """
      i, j = np.nonzero(np.triu(A, 1) if not self.d.directed else A)
      return _statistics(*self.packed, self.d.directed, self.sp, self.d.n, self.empty, i.astype(np.int64), j.astype(np.int64))


# ------------------------------------------------------------------------------------------------
# Compiled kernels
# ------------------------------------------------------------------------------------------------


@njit(cache = True, nogil = True)
def _choose(x, k):
   """Binomial coefficient of a non-negative integer ``x``."""
   out = 1.
   for t in range(k):
      out *= max(x - t, 0) / (t + 1)
   return out


@njit(cache = True, nogil = True)
def _delta(K, off, ip, ipo, fp, fpo, vi, V, ci, C, mi, M, directed, A, SP, dout, din, i, j, out):
   """
   Change statistics of adding the tie ``i -> j`` (``i < j`` if undirected) to the graph without
   it, written to ``out``. ``A`` is the current 0/1 adjacency matrix (the tie may be there),
   ``SP = A @ A`` its two-path counts (for shared partner terms), ``dout`` and ``din`` its degrees.
   """
   n = A.shape[0]
   a = A[i, j]
   for t in range(K.shape[0]):
      kind, o = K[t], off[t]
      q = off[t + 1] - o
      I, F = ip[ipo[t]:ipo[t + 1]], fp[fpo[t]:fpo[t + 1]]
      if kind == EDGES:
         out[o] = 1.
      elif kind == NODECOV:
         x = V[vi[t]]
         out[o] = x[i] + x[j] if I[0] == 0 else (x[j] if I[0] == 1 else x[i])
      elif kind == NODEFACTOR:
         c = C[ci[t]]
         for s in range(q):
            lev = I[1 + s]
            if I[0] == 0:
               out[o + s] = (c[i] == lev) + (c[j] == lev)
            elif I[0] == 1:
               out[o + s] = c[j] == lev
            else:
               out[o + s] = c[i] == lev
      elif kind == NODEMATCH:
         c = C[ci[t]]
         same = c[i] == c[j] and c[i] >= 0
         if I[0] == 1:
            for s in range(q):
               out[o + s] = same and c[i] == I[1 + s]
         else:
            ok = False
            for s in range(1, I.shape[0]):
               ok = ok or c[i] == I[s]
            out[o] = same and ok
      elif kind == NODEMIX:
         c = C[ci[t]]
         L = I[0]
         for s in range(q):
            out[o + s] = 0.
         x, y = c[i], c[j]
         if not directed and x > y:
            x, y = y, x
         if x >= 0 and y >= 0 and I[1 + x * L + y] >= 0:
            out[o + I[1 + x * L + y]] = 1.
      elif kind == ABSDIFF:
         x = V[vi[t]]
         out[o] = abs(x[i] - x[j]) ** F[0]
      elif kind == EDGECOV:
         out[o] = M[mi[t]][i, j]
      elif kind == MUTUAL:
         out[o] = A[j, i]
      elif kind == ASYMMETRIC:
         out[o] = 1. - 2. * A[j, i]
      elif kind == TRIANGLE or kind == TTRIPLE or kind == CTRIPLE:
         s = 0.
         for k in range(n):
            if not directed:
               s += A[i, k] * A[j, k]
            else:
               if kind != CTRIPLE:
                  s += A[i, k] * A[k, j] + A[i, k] * A[j, k] + A[k, i] * A[k, j]
               if kind != TTRIPLE:
                  s += A[j, k] * A[k, i]
         out[o] = s
      elif kind == KSTAR:
         for s in range(q):
            out[o + s] = _choose(dout[i] - a, I[s] - 1) + _choose(dout[j] - a, I[s] - 1)
      elif kind == ISTAR:
         for s in range(q):
            out[o + s] = _choose(din[j] - a, I[s] - 1)
      elif kind == OSTAR:
         for s in range(q):
            out[o + s] = _choose(dout[i] - a, I[s] - 1)
      elif kind == DEGREE or kind == IDEGREE or kind == ODEGREE:
         for s in range(q):
            v = I[s]
            if kind == DEGREE:
               x, y = dout[i] - a, dout[j] - a
               out[o + s] = (x + 1 == v) - (x == v) + (y + 1 == v) - (y == v)
            else:
               x = din[j] - a if kind == IDEGREE else dout[i] - a
               out[o + s] = (x + 1 == v) - (x == v)
      elif kind == CONCURRENT:
         out[o] = (dout[i] - a == 1) + (dout[j] - a == 1)
      elif kind == ISOLATES:
         if directed:
            out[o] = -1. * (dout[i] - a + din[i] == 0) - 1. * (dout[j] + din[j] - a == 0)
         else:
            out[o] = -1. * (dout[i] - a == 0) - 1. * (dout[j] - a == 0)
      elif kind == GWDEGREE:
         if I[0] == 0:
            out[o] = F[dout[i] - a] + F[dout[j] - a]
         elif I[0] == 1:
            out[o] = F[din[j] - a]
         else:
            out[o] = F[dout[i] - a]
      else:   # shared partners: edgewise (esp) or dyadwise (dsp), counts or geometrically weighted
         for s in range(q):
            out[o + s] = 0.
         gw = kind == GWESP or kind == GWDSP
         if kind == ESP or kind == GWESP:   # the tie itself
            sp = SP[i, j]
            if gw:
               out[o] += F[n + 1 + sp]
            else:
               for s in range(q):
                  out[o + s] += sp == I[s]
         for k in range(n):
            if k == i or k == j:
               continue
            for side in range(2):
               # ties or dyads gaining the shared partner, and their counts without the tie
               if not directed:
                  if kind == ESP or kind == GWESP:
                     hit = A[i, k] * A[j, k] > 0
                  else:
                     hit = (A[j, k] if side == 0 else A[i, k]) > 0
                  x = (SP[i, k] if side == 0 else SP[j, k]) - a
               else:
                  if kind == ESP or kind == GWESP:
                     hit = (A[i, k] * A[j, k] if side == 0 else A[k, i] * A[k, j]) > 0
                  else:
                     hit = (A[j, k] if side == 0 else A[k, i]) > 0
                  x = (SP[i, k] if side == 0 else SP[k, j]) - a
               if hit:
                  if gw:
                     out[o] += F[x]
                  else:
                     for s in range(q):
                        out[o + s] += (x + 1 == I[s]) - (x == I[s])


@njit(cache = True, nogil = True)
def _apply(A, SP, dout, din, i, j, directed, sp):
   """Toggle the tie ``i -> j`` (both directions if undirected), keeping two-paths and degrees."""
   n = A.shape[0]
   s = 1 - 2 * A[i, j]
   if sp:   # (A + sX)^2 = A^2 + s (A X + X A) + X^2, with A before the toggle
      if directed:
         for c in range(n):
            SP[i, c] += s * A[j, c]
            SP[c, j] += s * A[c, i]
      else:
         a = A[i, j]
         for c in range(n):
            ui, uj = s * A[j, c], s * A[i, c]
            SP[i, c] += ui
            SP[j, c] += uj
            if c != i:
               SP[c, i] += ui
            if c != j:
               SP[c, j] += uj
         SP[i, i] += s * a + 1
         SP[j, j] += s * a + 1
   A[i, j] = 1 - A[i, j]
   dout[i] += s
   if directed:
      din[j] += s
   else:
      A[j, i] = A[i, j]
      dout[j] += s
      din[i] += s
      din[j] += s


@njit(cache = True, nogil = True)
def _statistics(K, off, ip, ipo, fp, fpo, vi, V, ci, C, mi, M, directed, sp, n, empty, I, J):
   """Statistics of the graph with ties ``(I, J)``, adding them one by one to the empty graph."""
   A = np.zeros((n, n), dtype = np.uint8)
   SP = np.zeros((n, n) if sp else (1, 1), dtype = np.int32)
   dout, din = np.zeros(n, dtype = np.int64), np.zeros(n, dtype = np.int64)
   stats = empty.copy()
   d = np.zeros(stats.shape[0])
   for e in range(I.shape[0]):
      _delta(K, off, ip, ipo, fp, fpo, vi, V, ci, C, mi, M, directed, A, SP, dout, din, I[e], J[e], d)
      stats += d
      _apply(A, SP, dout, din, I[e], J[e], directed, sp)
   return stats


@njit(cache = True, parallel = True)
def _change_matrix(K, off, ip, ipo, fp, fpo, vi, V, ci, C, mi, M, directed, A, SP, dout, din, I, J, p):
   """Change statistics of all the dyads ``(I, J)`` of a graph, in parallel."""
   X = np.zeros((I.shape[0], p))
   for e in prange(I.shape[0]):
      _delta(K, off, ip, ipo, fp, fpo, vi, V, ci, C, mi, M, directed, A, SP, dout, din, I[e], J[e], X[e])
   return X


@njit(cache = True, nogil = True)
def _random(state):
   """Uniform number in [0, 1) from a xorshift64* generator (one per chain, for reproducibility)."""
   x = state[0]
   x ^= x >> np.uint64(12)
   x ^= x << np.uint64(25)
   x ^= x >> np.uint64(27)
   state[0] = x
   return float((x * np.uint64(2685821657736338717)) >> np.uint64(11)) * (1. / 9007199254740992.)


@njit(cache = True, parallel = True)
def _sample(K, off, ip, ipo, fp, fpo, vi, V, ci, C, mi, M, directed, sp, A, SP, dout, din, edges, pos, count, stats, rng, theta, burnin, interval, size, out):
   """
   Metropolis-Hastings chains with the tie / no tie proposal, one per thread: ``burnin`` steps,
   then ``size`` samples of the statistics every ``interval`` steps, written to ``out``.
   """
   B, n = A.shape[0], A.shape[1]
   D = n * (n - 1) if directed else n * (n - 1) // 2
   p = theta.shape[0]
   for b in prange(B):
      d = np.zeros(p)
      st = rng[b:b + 1]
      recorded = 0
      for step in range(burnin + interval * size):
         E = count[b]
         if E > 0 and _random(st) < 0.5:   # a random tie, proposed for removal
            e = min(int(_random(st) * E), E - 1)
            i, j = edges[b, e, 0], edges[b, e, 1]
         else:   # a random dyad
            i = min(int(_random(st) * n), n - 1)
            j = min(int(_random(st) * (n - 1)), n - 2)
            if j >= i:
               j += 1
            if not directed and i > j:
               i, j = j, i
         adding = A[b, i, j] == 0
         _delta(K, off, ip, ipo, fp, fpo, vi, V, ci, C, mi, M, directed, A[b], SP[b], dout[b], din[b], i, j, d)
         sgn = 1. if adding else -1.
         lr = 0.
         for s in range(p):
            lr += sgn * d[s] * theta[s]
         if adding:   # proposal ratio of TNT
            lr += math.log((0.5 / (E + 1) + 0.5 / D) * D / (0.5 if E > 0 else 1.))
         else:
            lr += math.log((0.5 if E > 1 else 1.) / D / (0.5 / E + 0.5 / D))
         if math.log(_random(st) + 1e-300) < lr:
            _apply(A[b], SP[b], dout[b], din[b], i, j, directed, sp)
            for s in range(p):
               stats[b, s] += sgn * d[s]
            if adding:   # tie lists: new ties at the end, removed ones replaced by the last
               edges[b, E, 0], edges[b, E, 1] = i, j
               pos[b, i, j] = E
               count[b] = E + 1
            else:
               at = pos[b, i, j]
               li, lj = edges[b, E - 1, 0], edges[b, E - 1, 1]
               edges[b, at, 0], edges[b, at, 1] = li, lj
               pos[b, li, lj] = at
               pos[b, i, j] = -1
               count[b] = E - 1
         if step + 1 > burnin and (step + 1 - burnin) % interval == 0:
            out[b, recorded] = stats[b]
            recorded += 1


# ------------------------------------------------------------------------------------------------
# Estimation
# ------------------------------------------------------------------------------------------------


def _dyad_list(d):
   """All the dyads: ``i < j`` if undirected, ``i != j`` if directed."""
   n = d.n
   if d.directed:
      i, j = np.nonzero(~np.eye(n, dtype = bool))
   else:
      i, j = np.triu_indices(n, 1)
   return i.astype(np.int64), j.astype(np.int64)


def _change_rows(model, d):
   """
   Change statistics of all the dyads of the observed network, compressed: the distinct rows of
   change statistics and tie values, with their counts (as R's ``ergm`` does for the MPLE).

   :return: ``(X, y, w)``: float64 arrays.
   """
   I, J = _dyad_list(d)
   SP, dout, din = model.state(d.A)
   X = _change_matrix(*model.packed, d.directed, d.A, SP, dout, din, I, J, len(model.names))
   u, w = np.unique(np.column_stack([X, d.A[I, J]]), axis = 0, return_counts = True)
   return u[:, :-1], u[:, -1], w.astype(np.float64)


def _logistic(X, y, w, ridge = 0., iterations = 100):
   """
   Weighted logistic regression without intercept, by Newton's method (iteratively reweighted
   least squares), with an optional ridge penalty :math:`\\lambda \\|\\theta\\|^2 / 2`.

   :return: ``(coefficients, covariance, log-likelihood, information)``.
   """
   theta = np.zeros(X.shape[1])
   eye = np.eye(X.shape[1])
   for _ in range(iterations):
      p = 1 / (1 + np.exp(-X @ theta))
      H = (X * (w * p * (1 - p))[:, None]).T @ X + ridge * eye
      step = np.linalg.lstsq(H, X.T @ (w * (y - p)) - ridge * theta, rcond = None)[0]
      theta = theta + step
      if np.abs(step).max() < 1e-10:
         break
   eta = X @ theta
   p = 1 / (1 + np.exp(-eta))
   H = (X * (w * p * (1 - p))[:, None]).T @ X
   ll = float((w * (y * eta - np.logaddexp(0, eta))).sum())
   return theta, np.linalg.pinv(H, hermitian = True), ll, H


class _Chains:
   """
   Parallel Markov chains on networks, each with its adjacency matrix, two-path counts, degrees,
   list of ties and current statistics, all started at the observed network.

   :param model: :class:`_Model`.
   :param count: Number of chains.
   :param seed: Seed of the random numbers, or ``None``.
   """

   def __init__(self, model, count, seed):
      d = model.d
      self.model, self.d, self.B = model, d, count
      n = d.n
      self.A = np.repeat(d.A[None], count, 0)
      SP, dout, din = model.state(d.A)
      self.SP = np.repeat(SP[None], count, 0)
      self.dout, self.din = np.repeat(dout[None], count, 0), np.repeat(din[None], count, 0)
      i, j = np.nonzero(np.triu(d.A, 1) if not d.directed else d.A)
      m = len(i)
      self.edges = np.zeros((count, d.dyads, 2), dtype = np.int64)
      self.edges[:, :m, 0], self.edges[:, :m, 1] = i, j
      self.pos = np.full((count, n, n), -1, dtype = np.int64)
      self.pos[:, i, j] = np.arange(m)
      self.count = np.full(count, m, dtype = np.int64)
      self.stats = np.repeat(model.stat(d.A)[None], count, 0)
      rs = np.random.default_rng(seed)
      self.rng = rs.integers(1, 2 ** 63, size = count, dtype = np.uint64)

   def sample(self, theta, burnin, interval, size):
      """
      Statistics sampled at ``theta``: after ``burnin`` steps, ``size`` per chain, every
      ``interval`` steps.

      :return: Samples (chain by chain) by statistics, and the chain of every sample.
      """
      out = np.zeros((self.B, size, len(theta)))
      _sample(*self.model.packed, self.d.directed, self.model.sp, self.A, self.SP, self.dout, self.din, self.edges, self.pos,
              self.count, self.stats, self.rng, np.asarray(theta, dtype = np.float64), burnin, interval, size, out)
      return out.reshape(-1, len(theta)), np.repeat(np.arange(self.B), size)


def _in_hull(S, x):
   """
   Whether a point is in the convex hull of sampled points (a linear feasibility problem).
   """
   from scipy.optimize import linprog
   N = S.shape[0]
   res = linprog(np.zeros(N), A_eq = np.vstack([S.T, np.ones(N)]), b_eq = np.append(x, 1), bounds = (0, None), method = "highs")
   return res.status == 0


def _pinv(C):
   """
   Pseudo-inverse of a covariance matrix, ignoring directions of (numerically) zero variance,
   e.g. statistics that do not vary in the sample.
   """
   w, U = np.linalg.eigh(C)
   keep = w > 1e-9 * max(1., float(np.abs(np.diag(C)).max()))
   return (U[:, keep] / w[keep]) @ U[:, keep].T


def _mcmle_step(S, target):
   """
   Monte Carlo maximum likelihood step (Geyer and Thompson, 1992): with statistics ``S`` sampled
   at ``theta0``, :math:`\\ell(\\theta) - \\ell(\\theta_0) \\approx (\\theta - \\theta_0)^T x -
   \\log \\frac{1}{N} \\sum_t e^{(\\theta - \\theta_0)^T s_t}`, maximized by Newton's method with the
   target statistics ``x``. The step is then halved until the importance weights of the samples at
   the new :math:`\\theta` keep an effective sample size of at least 5% of the sample.

   :param S: Samples by statistics.
   :param target: Target statistics.
   :return: The maximizing :math:`\\theta - \\theta_0`.
   """
   D = S - target
   f = lambda v: np.logaddexp.reduce(D @ v)
   v = np.zeros(S.shape[1])
   for _ in range(100):
      z = D @ v
      w = np.exp(z - z.max())
      w /= w.sum()
      mean = w @ D
      C = (D - mean).T @ ((D - mean) * w[:, None])
      step = -_pinv(C) @ mean
      t = 1.
      while f(v + t * step) > f(v) + 1e-12 and t > 1e-8:   # backtracking
         t /= 2
      v = v + t * step
      if np.abs(t * step).max() < 1e-9:
         break
   for _ in range(30):   # trust region: the approximation needs enough effective samples
      z = (S - S.mean(0)) @ v
      w = np.exp(z - z.max())
      w /= w.sum()
      if 1 / (w ** 2).sum() >= 0.05 * S.shape[0]:
         break
      v = v / 2
   return v


def _covariances(S, unit):
   """
   Covariance of sampled statistics and Monte Carlo covariance of their mean, from the sums of the
   centred samples over independent units (batches of consecutive samples of a chain).

   :return: ``(I, V)``.
   """
   X = S - S.mean(0)
   I = X.T @ X / S.shape[0]
   U = np.zeros((int(unit.max()) + 1, S.shape[1]))
   np.add.at(U, unit, X / S.shape[0])
   return I, U.T @ U


class ERGMFit:
   """
   A fitted exponential random graph model, as returned by :func:`ergm`.

   :ivar coef: :class:`pandas.Series` of estimated coefficients, by statistic.
   :ivar stderr: :class:`pandas.Series` of standard errors (including the Monte Carlo error).
   :ivar summary: :class:`pandas.DataFrame` with columns ``Estimate``, ``Std. Error``,
      ``MCMC %`` (share of the variance of the estimate due to the Monte Carlo sample), ``z value``
      and ``Pr(>|z|)``, as R's ``summary()``.
   :ivar observed: :class:`pandas.Series` of the statistics of the network.
   :ivar mple: :class:`pandas.Series` of the maximum pseudo-likelihood estimates.
   :ivar method: ``"MLE"`` (exact, dyad independent models), ``"MCMC MLE"`` or ``"MPLE"``.
   :ivar iterations: Number of Monte Carlo MLE iterations.
   :ivar converged: Whether the Monte Carlo MLE converged.
   :ivar samples: :class:`pandas.DataFrame` of the statistics sampled at the estimate (MCMC MLE),
      or ``None``.
   :ivar loglik: Log-likelihood (dyad independent models; pseudo-likelihood for the MPLE of
      dependent ones), or ``None``.
   :ivar aic: Akaike information criterion, from ``loglik``, or ``None``.
   :ivar bic: Bayesian information criterion, with the number of dyads as sample size, or ``None``.
   """

   def __init__(self, names, coef, cov, mcmc, observed, mple, method, iterations, converged, samples, loglik, dyads):
      from scipy.stats import norm
      self.coef = pd.Series(coef, index = names)
      total = cov + mcmc
      self.stderr = pd.Series(np.sqrt(np.maximum(np.diag(total), 0)), index = names)
      z = self.coef / self.stderr
      share = np.round(100 * np.diag(mcmc) / np.maximum(np.diag(total), 1e-300)).astype(int) if mcmc.any() else 0
      self.summary = pd.DataFrame({"Estimate": self.coef, "Std. Error": self.stderr, "MCMC %": share, "z value": z, "Pr(>|z|)": 2 * norm.sf(np.abs(z))})
      self.covariance = pd.DataFrame(total, index = names, columns = names)
      self.observed, self.mple, self.method = observed, mple, method
      self.iterations, self.converged, self.samples, self.loglik = iterations, converged, samples, loglik
      self.aic = None if loglik is None else -2 * loglik + 2 * len(names)
      self.bic = None if loglik is None else -2 * loglik + math.log(dyads) * len(names)

   def _keep(self, net, model, X, y, w):
      """Keep the network, the model and the change statistic rows, for simulation and fit measures."""
      self._net, self._model, self._rows = net, model, (X, y, w)
      return self

   def _draw(self, count, burnin, interval, seed):
      """
      Networks drawn from the model at the estimate: chains in parallel threads, started at the
      observed network, each giving a network after ``burnin`` steps and then every ``interval``
      steps.

      :return: List of ``(i, j)`` tie arrays (network wide node positions), and their statistics.
      """
      B = min(count, numba.get_num_threads())
      chain = _Chains(self._model, B, seed)
      ties, stats = [], []
      for r in range(math.ceil(count / B)):
         S, _ = chain.sample(self.coef.to_numpy(), burnin if r == 0 else 0, interval, 1)
         for b in range(B):
            e = chain.edges[b, :chain.count[b]]
            ties.append((e[:, 0].copy(), e[:, 1].copy()))
            stats.append(S[b])
      return ties[:count], np.array(stats[:count])

   def simulate(self, count = 1, burnin = 16384, interval = 1024, seed = None):
      """
      Networks drawn from the fitted model, as R's ``simulate()``: Markov chains at the estimated
      coefficients, started at the observed network, give a network after ``burnin`` steps and
      then one every ``interval`` steps (several chains run in parallel threads). Simulated
      networks keep the nodes of the observed one, with their ids and attributes (see
      :doc:`generators`); their edges take the attributes of observed edges, drawn at random.

      :param count: Number of networks.
      :param burnin: Steps of every chain before its first network.
      :param interval: Steps of every chain between networks.
      :param seed: Seed of the random numbers, or ``None`` for a random seed.
      :return: List of :class:`~graphoskop.Network`.

      Example:

      .. code-block:: python

         fit = gsk.ergm(net, "edges + nodematch('city') + gwesp(0.5, fixed = TRUE)", seed = 1)
         sims = fit.simulate(100, seed = 2)
         [gsk.average_clustering(g) for g in sims]
      """
      from .generators import _randomize
      ties, _ = self._draw(count, burnin, interval, seed)
      out = []
      for k, (i, j) in enumerate(ties):
         make = lambda l, nodes, n0, m, loops, gen, i = i, j = j: (torch.as_tensor(i, device = nodes.device), torch.as_tensor(j, device = nodes.device))
         out.append(_randomize(self._net, None if seed is None else seed + k, make))
      return out

   def gof(self, count = 100, burnin = 16384, interval = 1024, seed = None):
      """
      Goodness of fit, as R's ``gof()``: distributions of statistics of the observed network
      compared with those of ``count`` networks simulated from the fitted model (see
      :meth:`simulate`): the degree distribution (in- and out-degrees if directed), edgewise shared
      partners (outgoing two-paths if directed), geodesic distances (number of dyads at each
      distance, ``inf`` for unreachable ones) and the statistics of the model. For every value
      the table gives the observed count, the minimum, mean and maximum of the simulated ones and
      the Monte Carlo p-value, :math:`\\min(1, 2 \\min(P(sim \\leq obs), P(sim \\geq obs)))`: small
      values flag features of the network the model does not reproduce.

      :param count: Number of simulated networks.
      :param burnin: Steps of every chain before its first network.
      :param interval: Steps of every chain between networks.
      :param seed: Seed of the random numbers, or ``None``.
      :return: Dictionary from ``"model"``, ``"degree"`` (``"idegree"`` and ``"odegree"`` if
         directed), ``"espartners"`` and ``"distance"`` to :class:`pandas.DataFrame` with columns
         ``obs``, ``min``, ``mean``, ``max`` and ``MC p-value``.
      """
      from scipy.sparse.csgraph import shortest_path
      d = self._model.d
      n, directed = d.n, d.directed
      ties, stats = self._draw(count, burnin, interval, seed)

      def features(A):
         if directed:
            out = {"idegree": np.bincount(A.sum(0), minlength = n), "odegree": np.bincount(A.sum(1), minlength = n)}
         else:
            out = {"degree": np.bincount(A.sum(1), minlength = n)}
         SP = A.astype(np.int64) @ A.astype(np.int64)
         i, j = np.nonzero(np.triu(A, 1) if not directed else A)
         out["espartners"] = np.bincount(SP[i, j], minlength = n - 1)[:n - 1]
         G = shortest_path(A.astype(float), unweighted = True, directed = directed)
         G = G[np.triu_indices(n, 1)] if not directed else G[~np.eye(n, dtype = bool)]
         G = np.where(np.isinf(G), n, G).astype(np.int64)   # unreachable last
         out["distance"] = np.bincount(G, minlength = n + 1)[1:]
         return out

      obs = features(d.A)
      sims = []
      for i, j in ties:
         A = np.zeros((n, n), dtype = np.uint8)
         A[i, j] = 1
         if not directed:
            A |= A.T
         sims.append(features(A))

      def table(o, S, index, trim = True):
         S = np.asarray(S, dtype = float)
         p = np.minimum(1, 2 * np.minimum((S <= o).mean(0), (S >= o).mean(0)))
         t = pd.DataFrame({"obs": o, "min": S.min(0), "mean": S.mean(0), "max": S.max(0), "MC p-value": p}, index = index)
         seen = np.nonzero((t[["obs", "max"]] > 0).any(axis = 1).to_numpy())[0]
         return t.iloc[:seen.max() + 1 if len(seen) else 1] if trim else t   # up to the last value seen

      out = {"model": table(self.observed.to_numpy(), stats, self.coef.index, trim = False)}
      for key in obs:
         S = [x[key] for x in sims]
         if key == "distance":   # finite distances seen, and the unreachable dyads
            t = table(obs[key], S, list(range(1, n)) + [math.inf], trim = False)
            out[key] = pd.concat([table(obs[key][:-1], [x[:-1] for x in S], list(range(1, n))), t.iloc[[-1]]])
         else:
            out[key] = table(obs[key], S, list(range(len(obs[key]))))
      return out

   def pseudo_r2(self):
      """
      McFadden's pseudo-:math:`R^2` of the model, :math:`1 - \\ell / \\ell_0`, with :math:`\\ell`
      the log pseudo-likelihood of the model (logistic regression of the dyads on their change
      statistics; the log-likelihood for dyad independent models) and :math:`\\ell_0` that of the
      density only model (``edges``), and its decomposition among the statistics other than
      ``edges``:

      * ``pmvd``: proportional marginal variance decomposition (Feldman, 2005), as in R's
        ``relaimpo``: the average over all orders of the statistics of the increase of
        :math:`R^2` when each is added, each order weighted proportionally to
        :math:`\\prod_{i < p} 1 / (R^2 - R^2_{(i)})`, :math:`R^2_{(i)}` that of the first ``i``
        statistics of the order: a statistic that adds nothing given the others gets no share;
      * ``lmg``: the same with equal weights (Lindeman, Merenda and Gold, 1980; relaimpo's default).

      Both add up to the pseudo-:math:`R^2`. They need the model refitted on all :math:`2^p`
      subsets of the ``p`` statistics, computed once over subsets rather than over the :math:`p!`
      orders, so they are given for up to 16 statistics.

      :return: ``(r2, table)``: the pseudo-:math:`R^2` and a :class:`pandas.DataFrame` with columns
         ``pmvd`` and ``lmg``, by statistic.
      :raises ValueError: If there are more than 16 statistics besides ``edges``.

      Example:

      .. code-block:: python

         r2, parts = fit.pseudo_r2()
         parts["pmvd"] / r2                                    # shares of the explained pseudo-likelihood
      """
      X, y, w = self._rows
      names = list(self.coef.index)
      cols = [k for k, nm in enumerate(names) if nm != "edges"]
      p = len(cols)
      if p > 16:
         raise ValueError(f"the decomposition needs 2^{p} fits: at most 16 statistics besides edges")
      one = np.ones((X.shape[0], 1))
      ll0 = _logistic(one, y, w)[2]
      R = np.zeros(2 ** p)   # pseudo-R2 of every subset of statistics (bit k: statistic cols[k])
      for mask in range(1, 2 ** p):
         sel = [cols[k] for k in range(p) if mask >> k & 1]
         R[mask] = 1 - _logistic(np.column_stack([one, X[:, sel]]), y, w)[2] / ll0
      full = 2 ** p - 1
      out = pd.DataFrame(0., index = [names[c] for c in cols], columns = ["pmvd", "lmg"])
      for col, gap in (("pmvd", lambda U: 1 / max(R[full] - R[U], 1e-12 * max(R[full], 1e-300)) if U != full else 1.), ("lmg", lambda U: 1.)):
         g = np.array([gap(U) for U in range(2 ** p)])
         f = np.zeros(2 ** p)   # weight of the orders of every prefix set
         f[0] = 1.
         for U in range(1, 2 ** p):
            f[U] = g[U] * sum(f[U ^ (1 << k)] for k in range(p) if U >> k & 1)
         b = np.zeros(2 ** p)   # weight of the completions of every prefix set
         b[full] = 1.
         for U in range(full - 1, -1, -1):
            b[U] = sum(g[U | (1 << k)] * b[U | (1 << k)] for k in range(p) if not U >> k & 1)
         for k in range(p):
            total = sum(f[T] * g[T | (1 << k)] * b[T | (1 << k)] * (R[T | (1 << k)] - R[T]) for T in range(2 ** p) if not T >> k & 1)
            out.iloc[k, out.columns.get_loc(col)] = total / b[0]
      return float(R[full]), out

   def __repr__(self):
      text = f"{self.method} results:\n\n{self.summary.to_string(float_format = lambda x: f'{x:.5g}')}"
      if self.loglik is not None:
         text += f"\n\nlog-likelihood {self.loglik:.4f}, AIC {self.aic:.4f}, BIC {self.bic:.4f}"
      if self.method == "MCMC MLE":
         text += f"\n\n{self.iterations} iterations, " + ("converged" if self.converged else "NOT converged")
      return text


@_higher_order()
def ergm(net, formula, method = "mcmc", covariates = None, samplesize = 1024, burnin = 16384, interval = 1024, max_iterations = 20, tolerance = 0.1, chains = None, seed = None):
   """
   Fit an exponential random graph model (ERGM), as R's ``ergm()``.

   The model gives every network :math:`y` the probability
   :math:`P_\\theta(y) = e^{\\theta^T g(y)} / \\kappa(\\theta)`, with :math:`g` the statistics of
   the terms of ``formula`` (see below). Estimation follows R's ``ergm``:

   * the maximum pseudo-likelihood estimate (MPLE; Strauss and Ikeda, 1990): a logistic
     regression of every dyad on its change statistics, computed on the distinct rows of change
     statistics. For dyad independent models (only terms whose change statistics do not depend
     on the rest of the network, such as ``edges``, ``nodecov``, ``nodematch``, ``edgecov``) it is
     the maximum likelihood estimate, with exact standard errors and log-likelihood;
   * otherwise, the Monte Carlo maximum likelihood estimate (MCMC MLE; Geyer and Thompson, 1992),
     starting from the MPLE: networks are sampled at the current coefficients by Markov chain Monte
     Carlo, the log-likelihood ratio is approximated from their statistics and maximized, and the
     step repeats until the observed statistics are well within those sampled. When the observed
     statistics are outside the convex hull of the sample, the step aims at a point between them
     and the sample mean (Hummel, Hunter and Handcock, 2012); steps are also kept short enough for
     the sample to approximate the likelihood (effective sample size of the importance weights at
     least 5%). Standard errors come from the inverse information matrix (covariance of
     statistics sampled at the estimate) plus the Monte Carlo error (from batch means), whose share
     is reported as ``MCMC %``.

   Networks are sampled by Metropolis-Hastings with R's tie / no tie proposal, compiled, with
   ``chains`` chains in parallel threads, each started at the observed network and continued from
   one iteration to the next: in every iteration each chain makes ``burnin`` steps, then
   ``samplesize / chains`` samples, ``interval`` steps apart (R's defaults). Convergence: the
   coefficients are updated until the observed statistics are within ``tolerance`` standard
   deviations of the mean sampled ones (Mahalanobis distance, which is about the distance of the
   estimate from the MLE in standard errors; at least twice its Monte Carlo noise,
   :math:`2 \\sqrt{p / samplesize}`); a last sample at the estimate gives the standard errors.
   When the MPLE is infinite or very large (separation), the chains start from a ridge penalized
   MPLE; if the sample cannot move the estimate for two iterations, the coefficients go halfway
   back to a density only model. A model that only produces empty or complete networks
   (degenerate) triggers a warning.

   Terms (as in R; ``attr`` is a node attribute name, numeric or qualitative, missing values
   matching nothing):

   * ``edges``: number of ties;
   * ``nodecov(attr)``: sum over ties of the sum of the attribute of the two nodes (numeric);
     directed variants ``nodeicov``, ``nodeocov`` (receiver, sender);
   * ``nodefactor(attr, levels = -1)``: per category, number of tie ends at nodes of that category,
     the first (in sorted order) being the baseline (``levels = TRUE`` for all); directed
     ``nodeifactor``, ``nodeofactor``. This is how qualitative attributes enter as dummy variables;
   * ``nodematch(attr, diff = FALSE, levels = NULL)``: ties between nodes with the same value
     (homophily), one statistic, or one per category with ``diff = TRUE``;
   * ``nodemix(attr, levels2 = -1)``: ties per pair of categories (ordered if directed), the first
     pair being the baseline;
   * ``absdiff(attr, pow = 1)``: sum over ties of :math:`|x_i - x_j|^{pow}`;
   * ``edgecov(x, attrname = NULL)``: sum over ties of a dyadic covariate ``covariates[x]``: an
     ``n`` by ``n`` matrix (network wide node order, ``np.concatenate(net.node_ids)``), a table of
     ``source, target, value`` rows, or another :class:`~graphoskop.Network` on the same nodes (its
     ties, or their edge attribute ``attrname``). Edge attributes of the modelled network itself
     cannot be covariates: they exist only on ties, so they would predict ties perfectly; they are
     covariates when they come from another relation (e.g. a past wave, another layer);
   * ``mutual``, ``asymmetric``: reciprocated and unreciprocated dyads (directed);
   * ``triangle``: triangles (directed: transitive plus cyclic triples), ``ttriple``,
     ``ctriple`` (directed);
   * ``kstar(k)``: k-stars (undirected); ``istar(k)``, ``ostar(k)`` (directed); ``k`` can be a
     vector, e.g. ``kstar(2:3)``;
   * ``degree(d)``, ``idegree(d)``, ``odegree(d)``: nodes with degree ``d``; ``concurrent``: nodes
     with degree at least 2; ``isolates``;
   * ``gwdegree(decay)``, ``gwidegree``, ``gwodegree``: geometrically weighted degree
     distribution, :math:`e^\\alpha \\sum_k (1 - (1 - e^{-\\alpha})^k) D_k`;
   * ``gwesp(decay)``, ``gwdsp(decay)``: geometrically weighted edgewise and dyadwise shared
     partners (directed: outgoing two-paths, R's default ``OTP``); ``esp(d)``, ``dsp(d)``: ties and
     dyads with exactly ``d`` shared partners.

   The decay of geometrically weighted terms is fixed: in R, write ``fixed = TRUE`` for the same
   model (R's default ``fixed = FALSE`` estimates it as a curved ERGM, not supported here). Self
   loops are left out, parallel edges count once, and the network is the union of its components.

   :param net: A :class:`~graphoskop.Network`, neither multilayer nor bipartite.
   :param formula: Model formula, as in R, with or without ``net ~``:
      ``"edges + nodematch('city', diff = TRUE) + gwesp(0.5, fixed = TRUE)"``.
   :param method: ``"mcmc"`` (MCMC MLE, or exact MLE for dyad independent models) or ``"mple"``.
   :param covariates: Dictionary of dyadic covariates for ``edgecov`` terms, or ``None``.
   :param samplesize: Number of networks sampled per iteration.
   :param burnin: Steps of every chain before sampling, in every iteration.
   :param interval: Steps of every chain between samples.
   :param max_iterations: Maximum number of MCMC MLE iterations.
   :param tolerance: Convergence tolerance, in standard deviations of the statistics.
   :param chains: Number of chains, run in parallel threads; ``None`` for the number of Numba
      threads (by default, the number of CPU cores).
   :param seed: Seed of the random numbers, or ``None`` for a random seed.
   :return: :class:`ERGMFit`.
   :raises ValueError: For unknown terms or arguments, terms that do not apply to the network
      (e.g. ``mutual`` on an undirected one), attributes with several values for some node, or
      multilayer or bipartite networks.
   :raises KeyError: For missing node attributes or dyadic covariates.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")                    # node attributes age, city
      fit = gsk.ergm(net, "edges + nodecov('age') + nodematch('city')")    # dyad independent: exact MLE
      fit.summary
      fit = gsk.ergm(net, "edges + nodematch('city') + gwesp(0.5, fixed = TRUE)", seed = 1)
      fit.coef["gwesp.fixed.0.5"], fit.stderr["gwesp.fixed.0.5"]

      trade = gsk.load_network("trade.tsv")                                # another relation, same nodes
      fit = gsk.ergm(net, "edges + edgecov('trade', 'weight')", covariates = {"trade": trade})
   """
   if method not in ("mcmc", "mple"):
      raise ValueError(f"method must be 'mcmc' or 'mple', got {method!r}")
   d = _Data(net, covariates)
   model = _Model(d, formula)
   names, p = model.names, len(model.names)
   observed = model.stat(d.A)
   X, y, w = _change_rows(model, d)
   theta, cov, ll, H = _logistic(X, y, w)
   mple = pd.Series(theta, index = names)
   if np.linalg.matrix_rank(H, hermitian = True) < p:
      warnings.warn("some statistics are linearly dependent (e.g. nodefactor with nodemix of the same attribute): their coefficients are not identifiable")
   extreme = not np.isfinite(theta).all() or np.abs(theta).max() > 10
   if extreme:
      warnings.warn("the MPLE has very large coefficients: the observed statistics may be at their extreme (infinite estimates)")
   obs = pd.Series(observed, index = names)
   if method == "mple" or model.independent:
      return ERGMFit(names, theta, cov, np.zeros((p, p)), obs, mple, "MLE" if model.independent else "MPLE", 0, True, None, ll, d.dyads)._keep(net, model, X, y, w)

   # Monte Carlo maximum likelihood
   B = chains or numba.get_num_threads()
   size = math.ceil(samplesize / B)
   chain = _Chains(model, B, seed)
   unit = lambda: np.repeat(np.arange(B) * 16, size) + np.tile(np.arange(size) * min(16, size) // size, B)   # batch means: 16 batches per chain
   if extreme:   # start from a ridge penalized MPLE, which is finite
      scale = float(((X ** 2) * w[:, None]).sum(0).mean()) / 4   # typical curvature of the log-likelihood
      for ridge in (1e-3, 1e-2, 1e-1, 1., 10.):
         theta = _logistic(X, y, w, ridge * scale)[0]
         if np.abs(theta).max() <= 10:
            break
   null = np.zeros(p)   # density only
   if "edges" in names:
      m = observed[names.index("edges")]
      null[names.index("edges")] = math.log(max(m, 0.5) / max(d.dyads - m, 0.5))
   converged, it, stuck = False, 0, 0
   for it in range(1, max_iterations + 1):
      S, _ = chain.sample(theta, burnin, interval, size)
      mean = S.mean(0)
      C, _ = _covariances(S, unit())
      diff = mean - observed
      distance = math.sqrt(max(float(diff @ _pinv(C) @ diff), 0.))   # about the error of theta, in standard errors
      gamma = 1.
      if not _in_hull(S, mean + 1.05 * (observed - mean)):   # Hummel stepping
         lo, hi = 0., 1.
         for _ in range(12):
            g = (lo + hi) / 2
            lo, hi = (g, hi) if _in_hull(S, mean + 1.05 * g * (observed - mean)) else (lo, g)
         gamma = lo
      stuck = stuck + 1 if gamma < 0.01 else 0
      if stuck >= 2:   # the sample cannot move the estimate (e.g. frozen chains): halfway back to the null model
         theta, stuck = (theta + null) / 2, 0
         continue
      theta = theta + _mcmle_step(S, mean + gamma * (observed - mean))
      if gamma == 1 and distance < max(tolerance, 2 * math.sqrt(p / S.shape[0])):
         converged = True
         break
   if not converged:
      warnings.warn(f"MCMC MLE did not converge in {max_iterations} iterations: the estimates may be unreliable (raise max_iterations, burnin or samplesize)")
   S, _ = chain.sample(theta, burnin, interval, size)   # final sample, for the standard errors
   if ((chain.count == 0) | (chain.count == d.dyads)).mean() > 0.5:
      warnings.warn("the model looks degenerate: most sampled networks are empty or complete")
   I, V = _covariances(S, unit())
   Iinv = _pinv(I)
   return ERGMFit(names, theta, Iinv, Iinv @ V @ Iinv, obs, mple, "MCMC MLE", it, converged, pd.DataFrame(S, columns = names), None, d.dyads)._keep(net, model, X, y, w)


@_higher_order()
def ergm_statistics(net, formula, covariates = None):
   """
   Network statistics of an ERGM formula: the values of the statistics :math:`g(y)` of the
   network, as R's ``summary(net ~ ...)``.

   :param net: A :class:`~graphoskop.Network`, neither multilayer nor bipartite.
   :param formula: Model formula (see :func:`ergm`).
   :param covariates: Dyadic covariates for ``edgecov`` terms (see :func:`ergm`).
   :return: :class:`pandas.Series` of statistics, by name.
   :raises ValueError: See :func:`ergm`.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("edges.tsv", "nodes.tsv")
      gsk.ergm_statistics(net, "edges + triangle + nodematch('city')")
   """
   d = _Data(net, covariates)
   model = _Model(d, formula)
   return pd.Series(model.stat(d.A), index = model.names)
