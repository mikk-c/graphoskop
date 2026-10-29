Isomorphism
===========

.. automodule:: graphoskop.isomorphism
   :no-members:

.. code-block:: python

   import pandas as pd
   import graphoskop as gsk

   ring = gsk.add_edges(gsk.chain_graph(6), (5, 0))              # a 6-cycle
   path = gsk.chain_graph(4)
   gsk.is_isomorphic(ring, ring)                                 # True
   gsk.is_isomorphic(ring, path, subgraph = True)                # True: the cycle contains a 4-path
   gsk.is_isomorphic(path, ring, subgraph = True)                # False
   gsk.is_isomorphic(ring, path, subgraph = True, exact = False) # True: passes the filter

   net = gsk.load_network("edges.tsv", "nodes.tsv")
   other = gsk.load_network("edges2.tsv", "nodes2.tsv")
   gsk.is_isomorphic(net, other, exact = False)                  # Weisfeiler-Leman test only
   gsk.is_isomorphic(net, other, node_attributes = "city")       # cities must match too

   triangle = gsk.load_network(pd.DataFrame({"source": ["a", "b", "c"], "target": ["b", "c", "a"]}))
   m = gsk.isomorphisms(net, triangle, subgraph = True)          # every embedding of a triangle
   m["match"].nunique() // 6                                     # number of triangles

.. autofunction:: graphoskop.is_isomorphic

.. autofunction:: graphoskop.isomorphisms

Performance
-----------

Measured on two CPU threads (no GPU), testing a network against a copy with its nodes shuffled
(isomorphic) and against a copy where one swap of edge endpoints keeps the degrees but breaks the
isomorphism:

* Barabási-Albert network, 100,000 nodes and 300,000 edges: exact test 0.3 s (isomorphic) and
  0.2 s (not), approximate test 0.3 s; igraph's BLISS takes 14.6 s. With 1,000,000 nodes: 7.9 s
  and 6.9 s exact, 4.5 s and 3.5 s approximate.
* Random 3-regular network, 100,000 nodes: exact test 0.3 s (isomorphic) and 5.5 s (not: color
  refinement sees no difference, so every candidate image of the first node is tried); the
  approximate test cannot tell the two apart and answers ``True`` in 0.1 s. igraph's BLISS takes
  308 s on the isomorphic pair.
* 2D lattice, 300 by 300: 0.3 s and 0.2 s exact (BLISS: 0.4 s and 0.2 s); 1000 by 1000: 5.9 s and
  7.8 s exact, 3.8 s and 4.3 s approximate.

igraph's VF2 did not finish within five minutes on the 100,000-node Barabási-Albert pair, on a
10,000-node 3-regular pair or on a 100 by 100 lattice.

Subgraphs of a Barabási-Albert network with 100,000 nodes and 500,000 edges: whether it contains a
pattern of 3 to 7 nodes (triangle, 4-cycle, 6-path, 5-star, 6-wheel, 5- and 7-cliques) takes 0.15
to 0.45 s exactly and 0.1 to 0.3 s approximately, mostly spent building and filtering the
candidates; igraph's VF2 answers in milliseconds when the pattern is there, but takes 4.5 s and
2.4 s to rule out the cliques. The approximate test accepts both cliques, which are absent:
neighborhood filtering does not see them. Listing every mapping with :func:`~graphoskop.isomorphisms`: 3,960
for the triangle in 0.7 s (igraph's VF2 counts them in 9.8 s), 60,184 for the 4-cycle in 8.6 s
(igraph: 232 s).

On a GPU, the approximate tests and the first rounds of color refinement of the exact
isomorphism test run there; the searches always run on the CPU.
