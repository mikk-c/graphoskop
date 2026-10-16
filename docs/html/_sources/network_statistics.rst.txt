Network statistics
==================

.. automodule:: graphoskop.network_statistics

Assortativity
-------------

.. code-block:: python

   gsk.assortativity(net)              # degree assortativity
   gsk.assortativity(net, "age")       # numeric attribute: Pearson correlation over the edges
   gsk.assortativity(net, "city")      # qualitative attribute: Newman's categorical coefficient

.. autofunction:: graphoskop.assortativity

Average shortest path length
----------------------------

.. autofunction:: graphoskop.average_shortest_path_length

Clustering
----------

.. autofunction:: graphoskop.average_clustering

.. autofunction:: graphoskop.global_clustering

.. autofunction:: graphoskop.local_clustering

Connected components
--------------------

.. autofunction:: graphoskop.connected_components_count

.. autofunction:: graphoskop.connected_components_nodes

Degree
------

.. autofunction:: graphoskop.average_degree

.. autofunction:: graphoskop.degree

.. autofunction:: graphoskop.degree_centralization

.. autofunction:: graphoskop.degree_power_law

Density
-------

.. autofunction:: graphoskop.density

Reciprocity
-----------

.. autofunction:: graphoskop.reciprocity
