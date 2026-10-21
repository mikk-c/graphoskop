"""
Infomap engine (see :func:`~graphoskop.infomap_communities`): compiled moves minimizing the map
equation, aggregation of modules, two-level search and hierarchy.

A problem is a set of items (nodes, or groups of nodes) with:

* ``rate``: the rate of the codeword of the item in the codebook of its module (the visit rate of
  a node, or the entry rate of a submodule);
* links between items (``optr``, ``ocol``, ``ofl`` out of every item, ``iptr``, ``icol``, ``ifl``
  into it), carrying flow, without self loops;
* ``ext_out``, ``ext_in``: flow leaving or entering the set of all items (the parent module);
* entries (``eptr``, ``eent``, ``efl``): the physical nodes of the item and their rates (several
  items of the same physical node share one codeword in a module, as copies of the same entity in
  multilayer networks).

The codelength of a partition into modules, with ``base`` the exit rate of the parent module (0
for the whole network), is

.. math::

   L = \\mathrm{plogp}(b + \\sum_i q_i^{in}) - \\mathrm{plogp}(b) - \\sum_i \\mathrm{plogp}(q_i^{in})
   - \\sum_i \\mathrm{plogp}(q_i^{out}) + \\sum_i \\mathrm{plogp}(q_i^{out} + p_i)
   - \\sum_i \\sum_e \\mathrm{plogp}(p_{ie}),

with :math:`\\mathrm{plogp}(x) = x \\log_2 x`, :math:`q_i^{in}`, :math:`q_i^{out}` the flow
entering and leaving module :math:`i`, :math:`p_i` the sum of the rates of its items and
:math:`p_{ie}` that of its physical node :math:`e`.
"""

import numpy as np
from numba import njit, types
from numba.typed import Dict

_TOL = 1e-10


@njit(cache = True, inline = "always")
def _plogp(x):
   return x * np.log2(x) if x > 0 else 0.


@njit(cache = True)
def _core(n, rate, optr, ocol, ofl, iptr, icol, ifl, ext_out, ext_in, phys, eptr, eent, efl, nent, base, mod, sweeps, seed):
   """
   Greedy moves of items between modules (and to empty modules), in random order, while some
   move shortens the codelength (at most ``sweeps`` sweeps).

   :return: The module of every item (modules numbered below ``n``).
   """
   np.random.seed(seed)
   mod = mod.copy()
   O, I = ext_out.copy(), ext_in.copy()
   for x in range(n):
      for a in range(optr[x], optr[x + 1]):
         O[x] += ofl[a]
      for a in range(iptr[x], iptr[x + 1]):
         I[x] += ifl[a]
   mflow, mexit, menter = np.zeros(n), np.zeros(n), np.zeros(n)
   msize = np.zeros(n, dtype = np.int64)
   for x in range(n):
      m = mod[x]
      mflow[m] += rate[x]
      msize[m] += 1
      mexit[m] += ext_out[x]
      menter[m] += ext_in[x]
      for a in range(optr[x], optr[x + 1]):
         if mod[ocol[a]] != m:
            mexit[m] += ofl[a]
      for a in range(iptr[x], iptr[x + 1]):
         if mod[icol[a]] != m:
            menter[m] += ifl[a]
   P = Dict.empty(key_type = types.int64, value_type = types.float64)
   if phys:
      for x in range(n):
         for a in range(eptr[x], eptr[x + 1]):
            k = mod[x] * nent + eent[a]
            P[k] = P.get(k, 0.) + efl[a]
   enter_sum = menter.sum()
   free = np.empty(n, dtype = np.int64)   # stack of empty modules
   nfree = 0
   for m in range(n - 1, -1, -1):
      if msize[m] == 0:
         free[nfree] = m
         nfree += 1
   out_to, in_from = np.zeros(n), np.zeros(n)
   seen = np.zeros(n, dtype = np.bool_)
   touched = np.empty(n + 1, dtype = np.int64)
   order = np.arange(n)
   for sweep in range(sweeps):
      np.random.shuffle(order)
      moved = 0
      for x in order:
         i = mod[x]
         nt = 0
         for a in range(optr[x], optr[x + 1]):
            m = mod[ocol[a]]
            out_to[m] += ofl[a]
            if not seen[m]:
               seen[m] = True
               touched[nt] = m
               nt += 1
         for a in range(iptr[x], iptr[x + 1]):
            m = mod[icol[a]]
            in_from[m] += ifl[a]
            if not seen[m]:
               seen[m] = True
               touched[nt] = m
               nt += 1
         if msize[i] > 1 and nfree > 0:   # an empty module as a candidate
            touched[nt] = free[nfree - 1]
            nt += 1
         p, oi, ii = rate[x], out_to[i], in_from[i]
         ei = mexit[i] - (O[x] - oi) + ii
         ni = menter[i] - (I[x] - ii) + oi
         best, bj = -_TOL, -1
         for t in range(nt):
            j = touched[t]
            if j == i:
               continue
            oj, ij = out_to[j], in_from[j]
            ej = mexit[j] + (O[x] - oj) - ij
            nj = menter[j] + (I[x] - ij) - oj
            es = enter_sum - menter[i] - menter[j] + ni + nj
            d = _plogp(base + es) - _plogp(base + enter_sum)
            d -= _plogp(ni) + _plogp(nj) - _plogp(menter[i]) - _plogp(menter[j])
            d -= _plogp(ei) + _plogp(ej) - _plogp(mexit[i]) - _plogp(mexit[j])
            d += _plogp(ei + mflow[i] - p) + _plogp(ej + mflow[j] + p) - _plogp(mexit[i] + mflow[i]) - _plogp(mexit[j] + mflow[j])
            if phys:
               for a in range(eptr[x], eptr[x + 1]):
                  f = efl[a]
                  pi = P.get(i * nent + eent[a], 0.)
                  pj = P.get(j * nent + eent[a], 0.)
                  d -= _plogp(pi - f) - _plogp(pi) + _plogp(pj + f) - _plogp(pj)
            if d < best:
               best, bj = d, j
         if bj >= 0:
            j = bj
            oj, ij = out_to[j], in_from[j]
            ej = mexit[j] + (O[x] - oj) - ij
            nj = menter[j] + (I[x] - ij) - oj
            enter_sum += ni + nj - menter[i] - menter[j]
            mexit[i], menter[i], mexit[j], menter[j] = ei, ni, ej, nj
            mflow[i] -= p
            mflow[j] += p
            if msize[j] == 0:
               nfree -= 1   # j was the empty module on top of the stack
            msize[i] -= 1
            msize[j] += 1
            if msize[i] == 0:
               free[nfree] = i
               nfree += 1
               mexit[i], menter[i], mflow[i] = 0., 0., 0.
            if phys:
               for a in range(eptr[x], eptr[x + 1]):
                  P[i * nent + eent[a]] = P.get(i * nent + eent[a], 0.) - efl[a]
                  P[j * nent + eent[a]] = P.get(j * nent + eent[a], 0.) + efl[a]
            mod[x] = j
            moved += 1
         for t in range(nt):
            m = touched[t]
            out_to[m], in_from[m], seen[m] = 0., 0., False
      if moved == 0:
         break
   return mod


class _Items:
   """Items of a problem (see the module description), as NumPy arrays."""

   def __init__(self, rate, src, dst, fl, ext_out, ext_in, ent_item, ent_id, ent_fl):
      n = len(rate)
      self.n, self.rate = n, rate.astype(np.float64)
      self.src, self.dst, self.fl = src.astype(np.int64), dst.astype(np.int64), fl.astype(np.float64)
      o = np.argsort(self.src, kind = "stable")
      self.optr = np.r_[0, np.cumsum(np.bincount(self.src, minlength = n))].astype(np.int64)
      self.ocol, self.ofl = self.dst[o], self.fl[o]
      o = np.argsort(self.dst, kind = "stable")
      self.iptr = np.r_[0, np.cumsum(np.bincount(self.dst, minlength = n))].astype(np.int64)
      self.icol, self.ifl = self.src[o], self.fl[o]
      self.ext_out, self.ext_in = ext_out.astype(np.float64), ext_in.astype(np.float64)
      o = np.argsort(ent_item, kind = "stable")
      self.ent_item, self.ent_id, self.ent_fl = ent_item[o].astype(np.int64), ent_id[o].astype(np.int64), ent_fl[o].astype(np.float64)
      self.eptr = np.r_[0, np.cumsum(np.bincount(self.ent_item, minlength = n))].astype(np.int64)

   def core(self, mod, phys, nent, base, sweeps, seed):
      return _core(self.n, self.rate, self.optr, self.ocol, self.ofl, self.iptr, self.icol, self.ifl, self.ext_out, self.ext_in, phys, self.eptr,
                   self.ent_id, self.ent_fl, nent, base, mod.astype(np.int64), sweeps, seed)

   def aggregate(self, mod):
      """Items made of the modules of ``mod`` (numbered from 0)."""
      k = int(mod.max()) + 1
      cut = mod[self.src] != mod[self.dst]
      key = np.unique(mod[self.src[cut]] * k + mod[self.dst[cut]], return_inverse = True)
      fl = np.bincount(key[1], self.fl[cut]) if cut.any() else np.zeros(0)
      ekey, inv = np.unique(mod[self.ent_item] * (self.ent_id.max() + 1) + self.ent_id, return_inverse = True)
      return _Items(np.bincount(mod, self.rate, minlength = k), key[0] // k, key[0] % k, fl, np.bincount(mod, self.ext_out, minlength = k),
                    np.bincount(mod, self.ext_in, minlength = k), ekey // (self.ent_id.max() + 1), ekey % (self.ent_id.max() + 1), np.bincount(inv, self.ent_fl))

   def codelength(self, mod, base):
      """Codelength of a partition (see the module description)."""
      mod = np.unique(mod, return_inverse = True)[1]
      k = int(mod.max()) + 1
      cut = mod[self.src] != mod[self.dst]
      ex = np.bincount(mod[self.src[cut]], self.fl[cut], minlength = k) + np.bincount(mod, self.ext_out, minlength = k)
      en = np.bincount(mod[self.dst[cut]], self.fl[cut], minlength = k) + np.bincount(mod, self.ext_in, minlength = k)
      fl = np.bincount(mod, self.rate, minlength = k)
      ekey, inv = np.unique(mod[self.ent_item] * (self.ent_id.max() + 1) + self.ent_id, return_inverse = True)
      pe = np.bincount(inv, self.ent_fl)
      return float(_pl(base + en.sum()) - _pl(base) - _pl(en).sum() - _pl(ex).sum() + _pl(ex + fl).sum() - _pl(pe).sum())


def _pl(x):
   """plogp of an array (or number), 0 where not positive."""
   x = np.asarray(x, dtype = np.float64)
   return np.where(x > 0, x * np.log2(np.where(x > 0, x, 1)), 0.)


def _relabel(mod):
   """Modules numbered from 0."""
   return np.unique(mod, return_inverse = True)[1]


def _two_level(items, phys, nent, base, trials, sweeps, rng):
   """
   Best two-level partition of the items over ``trials`` trials. Each trial moves the items
   greedily, then moves the modules found, as items, while they merge (as in the Louvain method);
   then fine-tunes: the original items move again from the partition found, followed by the same
   aggregation, while the codelength decreases.

   :return: ``(module of every item, numbered from 0, codelength)``.
   """
   best = (None, np.inf)
   for _ in range(trials):
      assign = np.arange(items.n)
      L = items.codelength(assign, base)
      for tune in range(20):
         a = _relabel(items.core(assign, phys, nent, base, sweeps, int(rng.integers(2 ** 31))))
         cur = items.aggregate(a)
         while cur.n > 1:
            m = _relabel(cur.core(np.arange(cur.n), phys, nent, base, sweeps, int(rng.integers(2 ** 31))))
            if m.max() + 1 == cur.n:
               break
            a = m[a]
            cur = cur.aggregate(m)
         new = items.codelength(a, base)
         if new >= L - _TOL:
            break
         assign, L = a, new
      if L < best[1]:
         best = (_relabel(assign), L)
   return best


def _module_problem(items, nodes):
   """
   The items ``nodes`` of ``items`` as a problem of their own (a module): their links to the
   other items become external flow.
   """
   pos = np.full(items.n, -1)
   pos[nodes] = np.arange(len(nodes))
   s, d = pos[items.src], pos[items.dst]
   inside = (s >= 0) & (d >= 0)
   leave, come = (s >= 0) & (d < 0), (d >= 0) & (s < 0)
   ext_out = items.ext_out[nodes] + np.bincount(s[leave], items.fl[leave], minlength = len(nodes))
   ext_in = items.ext_in[nodes] + np.bincount(d[come], items.fl[come], minlength = len(nodes))
   e = pos[items.ent_item] >= 0
   return _Items(items.rate[nodes], s[inside], d[inside], items.fl[inside], ext_out, ext_in, pos[items.ent_item[e]], items.ent_id[e], items.ent_fl[e])


def _split(items, nodes, phys, nent, trials, sweeps, rng):
   """
   Recursive search of submodules of the module made of ``nodes``: a two-level partition of the
   module, whose index codebook also holds the exit of the module, is kept if it is shorter than
   the module's own codebook; then the same for every submodule.

   :return: The tree of the module: a list of subtrees, or the array of its nodes (a leaf module).
   """
   if len(nodes) < 2:
      return nodes
   sub = _module_problem(items, nodes)
   base = float(sub.ext_out.sum())
   pe = np.bincount(_relabel(sub.ent_id), sub.ent_fl)
   one = float(_pl(base + sub.rate.sum()) - _pl(base) - _pl(pe).sum())
   mod, L = _two_level(sub, phys, nent, base, trials, sweeps, rng)
   k = int(mod.max()) + 1
   if k < 2 or L >= one - _TOL:
      return nodes
   return [_split(items, nodes[mod == g], phys, nent, trials, sweeps, rng) for g in range(k)]


def _flows(items, assign):
   """Flow entering and leaving every module of ``assign`` (numbered from 0)."""
   k = int(assign.max()) + 1
   cut = assign[items.src] != assign[items.dst]
   ex = np.bincount(assign[items.src[cut]], items.fl[cut], minlength = k) + np.bincount(assign, items.ext_out, minlength = k)
   en = np.bincount(assign[items.dst[cut]], items.fl[cut], minlength = k) + np.bincount(assign, items.ext_in, minlength = k)
   return en, ex


def _coarsen(items, assign, trials, sweeps, rng):
   """
   Super-modules of the modules of ``assign``: a two-level partition of the network of modules,
   in which the codeword of a module is its entry, kept if it shortens the index codebook.

   :return: The super-module of every module, or ``None``.
   """
   en, ex = _flows(items, assign)
   agg = items.aggregate(assign)
   k = agg.n
   mods = _Items(en, agg.src, agg.dst, agg.fl, np.zeros(k), np.zeros(k), np.arange(k), np.arange(k), en)
   old = float(_pl(en.sum()) - _pl(en).sum())
   mod, L = _two_level(mods, False, k, 0., trials, sweeps, rng)
   if 1 < mod.max() + 1 < k and L < old - _TOL:
      return mod
   return None


def _leaves(tree, path = ()):
   """Leaf modules of a tree, as pairs (nodes, path of child positions from the root)."""
   if not isinstance(tree, list):
      return [(tree, path)]
   return [x for g, c in enumerate(tree) for x in _leaves(c, path + (g,))]


def _tree_codelength(items, tree):
   """
   Codelength of a hierarchy of modules: for every module, the codebook of its exit and of the
   entries of its submodules (or of the visit rates of its physical nodes, in leaf modules); the
   root has no exit.
   """
   leaves = _leaves(tree)
   depth = max(len(p) for _, p in leaves)
   total, entry = 0., {}
   for d in range(depth, -1, -1):   # the modules at depth d, from the bottom
      labels, assign = {}, np.zeros(items.n, dtype = np.int64)
      for nodes, path in leaves:   # shallower branches as modules of their own, not counted
         assign[nodes] = labels.setdefault(path[:d] if len(path) >= d else ("x",) + path, len(labels))
      en, ex = _flows(items, assign)
      for key, g in labels.items():
         if key[:1] == ("x",):
            continue
         sym = np.array([v for c, v in entry.items() if len(c) == d + 1 and c[:d] == key])
         if not len(sym):   # leaf module: its physical nodes
            e = assign[items.ent_item] == g
            sym = np.bincount(_relabel(items.ent_id[e]), items.ent_fl[e]) if e.any() else np.zeros(0)
         exit_ = ex[g] if d > 0 else 0.
         total += float(_pl(exit_ + sym.sum()) - _pl(exit_) - _pl(sym).sum())
         entry[key] = en[g]
   return total
