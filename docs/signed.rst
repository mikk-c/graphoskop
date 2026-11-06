Signed networks
===============

.. automodule:: graphoskop.signed
   :no-members:

.. autodata:: graphoskop.SIGN
   :annotation: = "sign"

Building a signed network
-------------------------

.. code-block:: python

   import graphoskop as gsk

   # (i) two layers of a multilayer network
   # relations.tsv:   source   target   layer
   #                  alice    bob      ally
   #                  bob      carol    enemy
   net = gsk.load_network("relations.tsv", multilayer = True)
   signed = gsk.signed_network(net, positive = "ally", negative = "enemy")

   # (ii) thresholding an edge attribute of an ordinary network
   net = gsk.load_network("ratings.tsv")                  # column "score"
   signed = gsk.signed_network(net, weight = "score", threshold = 3)   # score >= 3 positive

   signed.get_edge_attribute("sign")                       # +1 / -1, one entry per stored edge

The result is an ordinary network, so every other function works on it; the sign is one more edge
attribute, kept when saving and loading.

.. autofunction:: graphoskop.signed_adjacency

.. autofunction:: graphoskop.signed_network

Structural balance and signed triangles
---------------------------------------

.. code-block:: python

   gsk.structural_balance(signed)        # Tr(A^3) / Tr(|A|^3), a float
   gsk.structural_balance(signed, beta = 0.5)   # Tr(exp(beta A)) / Tr(exp(beta |A|)), all closed walks
   gsk.signed_triangle_counts(signed)    # count of each type, expected under shuffled signs, ratio, z

.. autofunction:: graphoskop.signed_triangle_counts

.. autofunction:: graphoskop.structural_balance

Signed clustering coefficient
-----------------------------

The local clustering coefficient of Costantini and Perugini (2014) counts the triangles around a
node with their sign: positive triangles raise it, negative ones lower it, from -1 to 1. Three
variants generalize the Watts-Strogatz coefficient (signs only), the one of Onnela et al. (signed
cube roots of normalized weights) and the one of Zhang and Horvath (signed weights over absolute
ones), for undirected networks such as signed correlation networks.

.. code-block:: python

   c = gsk.signed_clustering(signed)                                  # signs only
   gsk.signed_clustering(signed, kind = "onnela", weight = "correlation")
   gsk.signed_clustering(signed, ["a", "b"], kind = "zhang", weight = "correlation")

.. autofunction:: graphoskop.signed_clustering

.. autodata:: graphoskop.SIGNED_CLUSTERING
