Homophily and assortativity
===========================

*Atlas:* Chapters 30 (Homophily) and 31 (Quantitative Assortativity); ego networks, Section 30.1
(Ego Networks), are made by :func:`~graphoskop.ego_network` (:doc:`transformations`).

The assortativity of a qualitative node attribute measures homophily (Section 30.2, Quantifying
Homophily); the assortativity of the degree, degree correlations (Section 31.1, Degree
Correlations); the assortativity of a numeric node attribute, Section 31.3 (Distribution of
Quantitative Attributes).

.. code-block:: python

   gsk.assortativity(net)              # degree assortativity
   gsk.assortativity(net, "age")       # numeric attribute: Pearson correlation over the edges
   gsk.assortativity(net, "city")      # qualitative attribute: Newman's categorical coefficient
   hg = gsk.load_hypergraph("meetings.tsv")
   gsk.assortativity(hg, kind = "top-2")   # Chodrow: degrees of the two best connected members of every hyperedge

.. autofunction:: graphoskop.assortativity
