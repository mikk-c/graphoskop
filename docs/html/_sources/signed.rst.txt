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

.. autofunction:: graphoskop.signed_network

.. autofunction:: graphoskop.signed_adjacency

Structural balance and signed triangles
---------------------------------------

.. code-block:: python

   gsk.structural_balance(signed)        # Tr(A^3) / Tr(|A|^3), a float
   gsk.structural_balance(signed, beta = 0.5)   # Tr(exp(beta A)) / Tr(exp(beta |A|)), all closed walks
   gsk.signed_triangle_counts(signed)    # count of each type, expected under shuffled signs, ratio, z

.. autofunction:: graphoskop.structural_balance

.. autofunction:: graphoskop.signed_triangle_counts
