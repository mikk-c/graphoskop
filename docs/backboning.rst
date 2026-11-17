Backboning
==========

.. automodule:: graphoskop.backboning

*Atlas:* Chapter 27 (Network Backboning).

Extracting a backbone
---------------------

The functions below score the edges; these extract the backbone at a threshold of a score, and help
choose the threshold.

.. autofunction:: graphoskop.extract_backbone

.. autofunction:: graphoskop.find_bb_thresholds

.. autofunction:: graphoskop.test_bb_thresholds

Naive
-----

*Atlas:* Section 27.1 (Naive).

.. autofunction:: graphoskop.threshold_filter

Doubly stochastic
-----------------

*Atlas:* Section 27.2 (Doubly Stochastic).

.. autofunction:: graphoskop.doubly_stochastic

High-salience skeleton
----------------------

*Atlas:* Section 27.3 (High-Salience Skeleton).

.. autofunction:: graphoskop.high_salience_skeleton

Convex network reduction
------------------------

*Atlas:* Section 27.4 (Convex Network Reduction).

.. autofunction:: graphoskop.convex_skeleton

Disparity filter
----------------

*Atlas:* Section 27.5 (Disparity Filter).

.. autofunction:: graphoskop.disparity_filter

Noise-corrected
---------------

*Atlas:* Section 27.6 (Noise-Corrected).

.. autofunction:: graphoskop.noise_corrected

Pólya filter
------------

*Atlas:* the introduction of Chapter 27 (Network Backboning).

.. autofunction:: graphoskop.polya_filter

Spanning trees and filtered graphs
----------------------------------

*Atlas:* Section 13.4 (Spanning Trees & Other Filtered Graphs), and the introduction of Chapter 27.

.. autofunction:: graphoskop.spanning_tree

.. autofunction:: graphoskop.maximally_filtered_graph

.. autofunction:: graphoskop.triangulated_filtered_graph
