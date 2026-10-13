Coloring and independent sets
=============================

.. automodule:: graphoskop.coloring
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv")
   colors = gsk.graph_coloring(net)              # one array of node ids per color, fewest colors
   mis = gsk.maximum_independent_set(net)        # node ids of a largest independent set

.. autofunction:: graphoskop.graph_coloring

.. autofunction:: graphoskop.maximum_independent_set
