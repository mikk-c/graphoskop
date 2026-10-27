"""
Which public functions accept hypergraphs and simplicial complexes (see :doc:`higher_order`).

Every public function is wrapped by :func:`_higher_order`, which refuses higher-order networks
unless the function declares that it handles them: run on the incidence graph, most analyses
would give wrong answers without any warning (hyperedges counted as nodes, path lengths doubled).
This module imports nothing from the package, so that every module can use it.
"""

import functools
import inspect

_NETS = ("net", "net1", "net2")   # parameter names holding networks
_KINDS = {"hypergraph": "hypergraphs", "simplicial": "simplicial complexes"}


def _note(hypergraph, simplicial):
   """
   Sentence added to the docstring of a function taking networks.

   :param hypergraph: Whether the function handles hypergraphs.
   :param simplicial: Whether it handles simplicial complexes.
   :return: The sentence, in reStructuredText.
   """
   ok = [_KINDS[k] for k, v in (("hypergraph", hypergraph), ("simplicial", simplicial)) if v]
   no = [_KINDS[k] for k, v in (("hypergraph", hypergraph), ("simplicial", simplicial)) if not v]
   if not no:
      return "Hypergraphs and simplicial complexes: supported (see :doc:`higher_order`)."
   if not ok:
      return "Hypergraphs and simplicial complexes: not supported (see :doc:`higher_order`)."
   return f"{ok[0].capitalize()}: supported; {no[0]}: not supported (see :doc:`higher_order`)."


def _with_note(doc, note):
   """
   Docstring with a sentence inserted before its field list (``:param``, ``:return``, ...), or at
   its end if it has none.

   :param doc: Docstring, or ``None``.
   :param note: Sentence to insert.
   :return: The new docstring.
   """
   lines = (doc or "").split("\n")
   body = [l for l in lines[1:] if l.strip()]
   pad = " " * min((len(l) - len(l.lstrip()) for l in body), default = 0)   # 3 spaces, or none (Python 3.13 dedents docstrings)
   at = next((i for i, l in enumerate(lines) if l.strip().startswith((":param", ":return", ":raises"))), None)
   if at is None:
      return (doc or "").rstrip() + "\n\n" + pad + note + "\n"
   return "\n".join(lines[:at] + [pad + note, ""] + lines[at:])


def _higher_order(hypergraph = False, simplicial = False):
   """
   Decorator of a public function: raise :class:`ValueError` when one of its arguments is a
   hypergraph or a simplicial complex that it does not handle, and say in its docstring which
   ones it handles.

   :param hypergraph: Whether the function handles hypergraphs.
   :param simplicial: Whether it handles simplicial complexes.
   :return: The decorator.
   """
   allowed = {"hypergraph": hypergraph, "simplicial": simplicial}

   def wrap(f):
      @functools.wraps(f)
      def guarded(*args, **kwargs):
         for x in (*args, *kwargs.values()):
            kind = x.higher_order if hasattr(type(x), "_is_network") else None   # Network only: never read attributes of other objects
            if kind is not None and not allowed.get(kind, False):
               what = "the 1-skeleton" if kind == "simplicial" else "the clique expansion"
               raise ValueError(f"{f.__name__} does not support {_KINDS[kind]}: analyze gsk.incidence_network(net) (the bipartite network of nodes and {'facets' if kind == 'simplicial' else 'hyperedges'}) or gsk.clique_expansion(net) ({what})")
         return f(*args, **kwargs)

      guarded._higher_order = dict(allowed)
      if any(p in _NETS for p in inspect.signature(f).parameters):
         guarded.__doc__ = _with_note(f.__doc__, _note(hypergraph, simplicial))
      return guarded

   return wrap
