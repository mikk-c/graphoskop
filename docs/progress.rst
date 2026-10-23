Progress bars
=============

.. automodule:: graphoskop.progress
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv", "nodes.tsv")
   out = gsk.gcn(net, "city", [32, 4], epochs = 2000)       # a bar while it runs
   out = gsk.gcn(net, "city", [32, 4], progress = False)    # no bar for this call
   gsk.set_progress(True, delay = 5)                         # bars only after 5 seconds
   gsk.set_progress(False)                                   # no bars at all

.. autofunction:: graphoskop.set_progress

Performance
-----------

Every function was timed (best of 3, two CPU threads) before progress bars existed, with its bar
off, with its bar shown from the start (the default, output to a file), and with
``set_progress(True, delay = 5)`` (shown after 5 seconds). The differences are run-to-run noise (up
to about 10%, as often faster as slower), and the results are identical in all four cases:

==================================  =========  =========  =========  =========
Function                            Before     Bar off    Bar shown  Delay 5 s
==================================  =========  =========  =========  =========
``gcn``, 10,000 nodes, 200 epochs   5.12 s     5.03 s     4.61 s     4.58 s   
``graphsage``, same                 3.52 s     3.70 s     3.63 s     3.94 s   
``gat``, 100 epochs                 7.32 s     7.01 s     6.75 s     7.69 s   
``gatv2``, 100 epochs               10.90 s    10.84 s    11.14 s    11.11 s  
``node_removal``, adaptive          8.96 s     9.41 s     9.16 s     9.34 s   
``node_removal``, path lengths      5.95 s     6.32 s     6.34 s     6.19 s   
``edge_removal``, path lengths      8.51 s     8.37 s     8.44 s     8.66 s   
``cascade_failures``, path lengths  5.36 s     5.42 s     5.54 s     5.39 s   
``randomization_test``, 100         4.39 s     4.30 s     4.33 s     4.77 s   
``edge_swap``, 2M swaps             0.85 s     0.82 s     0.85 s     0.86 s   
``nestedness``, 40 nulls            15.53 s    15.18 s    15.25 s    14.84 s  
==================================  =========  =========  =========  =========

In a terminal, a shown bar redraws at most 10 times a second (once every 30 seconds when the
output is not a terminal, e.g. a log file), which cost nothing measurable either (a 7 s
:func:`~graphoskop.gcn` run redrawn 122 times took the same time as without a bar).

The time left is an estimate from the work done so far: it starts rough and becomes accurate as
the computation advances (e.g. an adaptive attack of 9 s first estimated 6 s at 2 s, then 8 s at
3 s). Where units of work differ in size, they are weighted by their expected cost: in adaptive
attacks each step counts the items still in the network; for path lengths, each point counts the
sources of the searches times the items still in the network.
