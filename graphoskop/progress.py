"""
Progress bars of long computations, in the terminal (or as a widget in Jupyter, if ipywidgets is
installed), with the time elapsed and the time left.

A bar appears as soon as a computation starts (or, with a delay set by :func:`set_progress`, only
once it has run that long, so that shorter computations print nothing), and disappears when it
ends. The bar follows the units of work the function does anyway (epochs, runs, swaps,
removals), so nothing is estimated in advance and nothing is computed only for the bar: the time
left is the time elapsed per unit, times the units left, and it is accurate when the units take
about the same time (units of different sizes are weighted by their expected cost); it starts
rough and improves as the computation advances. Updating a bar costs about 0.2 microseconds,
against milliseconds or more for each unit of work. On a GPU, bars read no value from the device,
so they never make it wait.

A function with a bar has a ``progress`` parameter: ``None`` (default) follows
:func:`set_progress`, ``True`` or ``False`` turn its bar on or off. Inside a function with a bar
(e.g. :func:`~graphoskop.randomization_test` of :func:`~graphoskop.edge_swap`), the inner
functions show none: the outer bar covers them (and if it is off, they are off too).

Functions with a bar: :func:`~graphoskop.cascade_failures` (average path length), :func:`~graphoskop.diffpool`,
:func:`~graphoskop.edge_removal` and :func:`~graphoskop.node_removal` (average path length, or
``recompute``), :func:`~graphoskop.edge_swap`, :func:`~graphoskop.gat`, :func:`~graphoskop.gatv2`,
:func:`~graphoskop.gcn`, :func:`~graphoskop.graphsage`, :func:`~graphoskop.nestedness` (null
model), :func:`~graphoskop.node2vec`, :func:`~graphoskop.randomization_test`.
"""

import sys

#: Global settings: bars on or off, seconds before a bar appears.
_SETTINGS = {"enabled": True, "delay": 0.}

#: Number of bars currently open (inner functions show none).
_OPEN = [0]


def set_progress(enabled = True, delay = 0.):
   """
   Turns the progress bars of all functions on or off, and sets how long a computation runs
   before its bar appears.

   :param enabled: Whether to show progress bars.
   :param delay: Seconds before a bar appears: 0 to show it at once, or e.g. 5 to show bars only
      for computations that last longer than that.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      gsk.set_progress(False)              # no bars (e.g. in batch jobs)
      gsk.set_progress(True, delay = 5)    # bars only for computations longer than 5 seconds
   """
   _SETTINGS["enabled"], _SETTINGS["delay"] = bool(enabled), float(delay)


class _Silent:
   """
   A bar that shows nothing (still counted among the open bars: the inner functions of a
   function without a bar show none either).
   """

   def update(self, n = 1):
      pass

   def __enter__(self):
      _OPEN[0] += 1
      return self

   def __exit__(self, *args):
      _OPEN[0] -= 1


class _Bar:
   """
   A tqdm bar, counted among the open bars while it is used as a context manager.
   """

   def __init__(self, total, desc):
      self.total, self.desc = total, desc

   def __enter__(self):
      _OPEN[0] += 1
      tty = sys.stderr is not None and hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
      self.bar = _tqdm()(total = self.total, desc = self.desc, delay = _SETTINGS["delay"], leave = False, dynamic_ncols = True,
                         mininterval = 0.1 if tty else 30., bar_format = "{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}")
      self.update = self.bar.update
      return self

   def __exit__(self, *args):
      _OPEN[0] -= 1
      self.bar.close()


def _tqdm():
   """
   The tqdm class: the Jupyter widget if running in a notebook with ipywidgets, else the terminal
   bar.
   """
   try:
      from IPython import get_ipython
      if "IPKernelApp" in get_ipython().config:
         import ipywidgets   # noqa: F401
         from tqdm.notebook import tqdm
         return tqdm
   except Exception:
      pass
   from tqdm.std import tqdm
   return tqdm


def _bar(total, desc, progress = None):
   """
   Progress bar of ``total`` units of work, to use as a context manager, with ``update(units)``
   after each piece of work; a bar that shows nothing if bars are off (``progress``, or the
   global setting if ``None``) or another bar is open.
   """
   if not (_SETTINGS["enabled"] if progress is None else progress) or _OPEN[0] or total <= 0:
      return _Silent()
   return _Bar(total, desc)
