Core-periphery structure
========================

.. automodule:: graphoskop.core_periphery
   :no-members:

.. code-block:: python

   import graphoskop as gsk

   net = gsk.load_network("edges.tsv")
   core, rho = gsk.discrete_core_periphery(net)                  # boolean vector, correlation
   net.set_node_attribute("core", core)
   core, rho = gsk.discrete_core_periphery(net, core_periphery_ties = "missing")
   coreness, rho = gsk.continuous_core_periphery(net, weight = "weight")   # largest value 1
   score, quality = gsk.rombach_core_periphery(net)               # core scores, R per (alpha, beta)
   score, quality = gsk.rombach_core_periphery(net, alpha = 1, beta = 0.8, p = 2)

References:

* S. P. Borgatti and M. G. Everett (2000), Models of core/periphery structures, *Social
  Networks* 21(4), 375-395, https://doi.org/10.1016/S0378-8733(99)00019-2.
* M. P. Rombach, M. A. Porter, J. H. Fowler and P. J. Mucha (2014), Core-periphery structure in
  networks, *SIAM Journal on Applied Mathematics* 74(1), 167-190,
  https://doi.org/10.1137/120881683.

.. autofunction:: graphoskop.continuous_core_periphery

.. autofunction:: graphoskop.discrete_core_periphery

.. autofunction:: graphoskop.rombach_core_periphery

.. _nestedness:

Nestedness
----------

Nestedness of bipartite networks: the core-periphery structure of a biadjacency matrix in which
the neighbors of each node are a subset of the neighbors of every node of the same type with more
neighbors. Sorting rows and columns by decreasing degree, the ones of a perfectly nested matrix
fill its top-left corner, up to a line, the isocline, beyond which there are only zeros.

Rows are the nodes of type 0 and columns the nodes of type 1 (see
:attr:`~graphoskop.Network.bipartite_labels`), of all components, isolated nodes included; edges
are unweighted (parallel edges count once) and their direction, if any, is ignored.

:func:`nestedness` measures NODF (Almeida-Neto et al., 2008): for every pair of rows with different
degrees, the percentage of the neighbors of the row with fewer neighbors that are also neighbors of
the other (0 for pairs with equal degrees), averaged over all pairs of rows and of columns. It is 100
for a perfectly nested matrix. Its significance is the comparison with the values of random
matrices from a null model:

* ``"equiprobable"``: the same number of ones, placed uniformly at random;
* ``"proportional"`` (Bascompte et al., 2003): each cell is one, independently, with the mean of the
  densities of its row and of its column, :math:`(k_i / C + k_j / R) / 2`;
* ``"fixed"``: the same degrees for all nodes, by edge swaps (see :func:`~graphoskop.edge_swap`).

The sum over pairs is computed without forming pairs: for every edge from row :math:`j` to a
column, the rows of that column with more neighbors than :math:`j` each add :math:`1 / k_j`, and
they are counted by sorting the edges, in :math:`O(m \log m)` on the device of the network; the
random matrices are all scored together.

:func:`nestedness_isocline` gives the isocline as the fill line of the nestedness temperature
(Atmar and Patterson, 1993, as made explicit by Rodríguez-Gironés and Santamaría, 2006): with
:math:`x` the position of a column and :math:`y` that of a row, both from 0 to 1 from the
top-left corner, the ones are expected where :math:`y < f(x)`,

.. math::

   f(x) = 1 - \left(1 - (1 - x)^p\right)^{1 / p}, \quad \text{i.e.} \quad (1 - x)^p + (1 - y)^p = 1,

a quarter of a superellipse, symmetric between rows and columns. The area below it is
:math:`1 - \Gamma(1 + 1 / p)^2 / \Gamma(1 + 2 / p)`: :math:`p = 1` is the anti-diagonal (half of the
matrix filled), larger :math:`p` a smaller corner.

.. code-block:: python

   net = gsk.load_network("pollination.tsv", "species.tsv", bipartite = True)
   gsk.nestedness(net)                                      # nodf, nodf_rows, nodf_columns
   res = gsk.nestedness(net, null_model = "fixed", samples = 1000, seed = 1)
   res["z"], res["p_value"]
   iso = gsk.nestedness_isocline(net)                       # p, fill, rows, columns, line
   iso = gsk.nestedness_isocline(net, method = "fit")

References:

* M. Almeida-Neto, P. Guimarães, P. R. Guimarães Jr, R. D. Loyola and W. Ulrich (2008), A
  consistent metric for nestedness analysis in ecological systems: reconciling concept and
  measurement, *Oikos* 117(8), 1227-1239, https://doi.org/10.1111/j.0030-1299.2008.16644.x.
* J. Bascompte, P. Jordano, C. J. Melián and J. M. Olesen (2003), The nested assembly of
  plant-animal mutualistic networks, *PNAS* 100(16), 9383-9387,
  https://doi.org/10.1073/pnas.1633576100.
* W. Atmar and B. D. Patterson (1993), The measure of order and disorder in the distribution of
  species in fragmented habitat, *Oecologia* 96(3), 373-382, https://doi.org/10.1007/BF00317508.
* M. A. Rodríguez-Gironés and L. Santamaría (2006), A new algorithm to calculate the nestedness
  temperature of presence-absence matrices, *Journal of Biogeography* 33(5), 924-935,
  https://doi.org/10.1111/j.1365-2699.2006.01444.x.

.. autofunction:: graphoskop.nestedness

.. autofunction:: graphoskop.nestedness_isocline
