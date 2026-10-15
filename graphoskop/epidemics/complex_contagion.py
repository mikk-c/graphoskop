"""
Complex contagion: compartmental models of epidemics where the number of infected neighbors of a
susceptible node matters for its infection (reinforcement), as for behaviors and ideas that people
adopt only after seeing several of their contacts adopt them. Recoveries are as in simple
contagion: in the SIS and SIR models every infected node recovers with probability ``mu`` per
step, whatever its neighbors.

Five kinds of reinforcement decide the infection of a susceptible node with :math:`c` infected
neighbors (in-neighbors in directed networks):

* ``"classical"``: every infected neighbor tries to infect the node independently, with
  probability ``beta``, so the node is infected with probability :math:`1 - (1 - \\beta)^c`
  (one coin flip per infected neighbor, instead of one in simple contagion, so a small ``beta``
  spreads much more where nodes have many infected neighbors);
* ``"threshold"``: the node can be infected only if :math:`c \\ge k`, an absolute threshold, as in
  the threshold model of Granovetter (1978) in the form used for complex contagion by Centola and
  Macy (2007): hubs, with many neighbors, reach the threshold easily;
* ``"cascade"``: the node can be infected only if :math:`c \\ge \\phi d`, a fraction of its
  in-degree :math:`d`, as in the cascade model of Watts (2002): hubs, with a large denominator,
  rarely reach the threshold;
* ``"independent"``: the independent cascade model of Kempe, Kleinberg and Tardos (2003). When a
  node :math:`u` is infected, it makes one attempt to infect each of its susceptible
  out-neighbors :math:`v`, at the next step, which succeeds with probability :math:`p_{uv}`; if it
  fails, :math:`u` never tries :math:`v` again (until :math:`u` is infected anew, in SIS). The
  probability depends on both the origin and the target of the attempt, through a table of
  probabilities per pair of nodes, given as ``beta``. Unlike ``"classical"``, where infected
  neighbors try again at every step, each edge carries one attempt per infection;
* ``"decreasing"``: the decreasing cascade model of Kempe, Kleinberg and Tardos (2005), with one
  attempt per edge as in the independent cascade, but every failed attempt on :math:`v` makes the
  next ones less likely: an attempt after :math:`f` failed ones succeeds with probability
  :math:`p_{uv} \\delta^f`, with ``decay`` :math:`\\delta \\in (0, 1]` (:math:`\\delta = 1` is the
  independent cascade). Simultaneous attempts on a node count as made one after the other (in an
  arbitrary order, as in the original model); the count of failures restarts when the node is
  infected.

With the thresholds met, the node is infected with probability ``beta`` (``beta = 1`` gives the
deterministic models of Granovetter and Watts). In the cascades, ``beta`` is the table of
probabilities :math:`p_{uv}` of the attempts from origin :math:`u` to target :math:`v`: one
probability for all edges, the name of a numeric edge attribute, a matrix (NumPy array, tensor or
SciPy sparse matrix, rows for origins and columns for targets, indexed in the node order of the
nodes considered, e.g. ``net.node_ids[0]`` for component 0), or a :class:`pandas.DataFrame` with
origin, target and probability in its first three columns (pairs not listed get 0). Only the
pairs joined by an edge matter, and in undirected networks the two directions of an edge can have
different probabilities. Independent cascades on SI are exactly the original model, in which
infected (active) nodes stay infected; with SIR the spread is the same, and the infected nodes
are removed at rate ``mu``.

In the cascades every infection has one origin, the node whose attempt succeeded, so with
``forest = True`` the models also return who infected whom in the last run, as a directed
:class:`~graphoskop.Network`: an edge from origin to target for every infection, with the step of
the infection as edge attribute ``step``, and the infected nodes (seeds included, even those that
infected nobody) with their node attributes. In SI and SIR every node is infected at most once,
so this is a forest: one tree per seed, rooted at it, and each tree is a weakly connected
component (``forest.components``, :func:`~graphoskop.connected_components_nodes`). In SIS a node
can be infected again after recovering, and every infection is an edge, so the network can have
parallel edges and cycles. When several attempts on a node succeed at the same step, its origin
is the first of them in the order of the attempts. Thresholds are one value for all nodes, or one per
node (a vector, a dictionary or :class:`pandas.Series` from node id to value, or the name of a
numeric node attribute), so that, e.g., heterogeneous thresholds can be drawn from a
distribution. Degrees count edges: parallel edges count several times, and a self loop counts in
the in-degree (but never as an infected neighbor of a susceptible node).

Everything else (time steps, seeds, runs, engines, stored states, resuming, the plateau of SIS) is
as in :mod:`~graphoskop.epidemics.simple_contagion`. A seed node of a threshold model is infected
whatever its neighbors; a node whose threshold is above its in-degree can only be infected as a
seed.
"""

from .simple_contagion import _simulate


def complex_si_model(net, beta, reinforcement = "classical", threshold = None, decay = 0.5, seeds = 1, ranking = None, selection = "top", runs = 100, max_steps = 1000, component = 0, seed = None, engine = None, batch = None, attribute = "state", resume = False, forest = False, immunize = None, strategy = "random"):
   """
   Simulate the SI model with complex contagion, and return the fraction of nodes in each state
   at every time step, averaged over independent runs.

   As :func:`~graphoskop.si_model`, but a susceptible node is infected according to
   ``reinforcement`` (see the module description). A run stops when no susceptible node can be
   infected any more (in threshold models, the cascade may stop well before reaching all the
   nodes), or after ``max_steps`` steps.

   :param net: A :class:`~graphoskop.Network`.
   :param beta: Infection probability, in (0, 1]: per infected neighbor (``"classical"``), once
      the threshold is met (``"threshold"``, ``"cascade"``), or per attempt (``"independent"``,
      ``"decreasing"``: one value, or a table of probabilities per pair of nodes, see the module
      description).
   :param reinforcement: ``"classical"``, ``"threshold"``, ``"cascade"``, ``"independent"`` or
      ``"decreasing"``.
   :param threshold: With ``"threshold"``, the number of infected neighbors needed (at least 1);
      with ``"cascade"``, the share of the in-neighbors, in (0, 1]; one value, or one per node.
      Ignored by the other kinds.
   :param decay: With ``"decreasing"``, the factor of the probability of an attempt for every
      earlier failed attempt on the same node, in (0, 1].
   :param seeds: Number of seed nodes, or a fraction of the nodes if below 1.
   :param ranking: Node values used to choose the seeds, or ``None`` for random seeds.
   :param selection: ``"top"`` or ``"weighted"``.
   :param runs: Number of independent runs.
   :param max_steps: Largest number of steps of a run.
   :param component: Index of the component, or ``None`` for the whole network.
   :param seed: Seed of the random numbers, or ``None``.
   :param engine: ``"numba"``, ``"torch"`` or ``None``.
   :param batch: With the torch engine, the number of runs simulated at once, or ``None``.
   :param attribute: Name of the qualitative node attribute holding the states (``"S"`` or
      ``"I"``), or ``None``.
   :param resume: Whether to start from the states in ``attribute`` instead of seeds.
   :param forest: With ``"independent"`` or ``"decreasing"``, whether to return also who infected
      whom in the last run (see the module description).
   :param immunize: Nodes immunized at step 0, never infected (see
      :mod:`~graphoskop.epidemics.simple_contagion`): a number of nodes (a fraction of them if
      below 1), chosen by ``strategy``, or the node ids themselves; ``None`` for none.
   :param strategy: With a number of nodes to immunize: ``"random"``, ``"friend_of_random"``, or
      node values (vector, dictionary or :class:`pandas.Series`, name of a numeric node
      attribute) whose largest ones are immunized. Ignored with node ids.
   :return: :class:`pandas.DataFrame` indexed by step, with columns ``S``, ``S_low``,
      ``S_high``, ``I``, ``I_low`` and ``I_high``; with ``forest``, a pair (table, forest
      :class:`~graphoskop.Network`).
   :raises ValueError: As :func:`~graphoskop.si_model`, if ``reinforcement`` is not valid, if
      thresholds are missing or out of range, if probabilities per pair are out of [0, 1] or given
      for another kind of reinforcement, if ``decay`` is not valid, or if ``forest`` is asked for
      another kind of reinforcement.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("contacts.tsv")
      si = gsk.complex_si_model(net, 0.02, "classical", seeds = 5)        # one coin flip per infected neighbor
      si = gsk.complex_si_model(net, 1, "threshold", 2, seeds = 0.05)     # Granovetter: 2 infected neighbors
      si = gsk.complex_si_model(net, 1, "cascade", 0.18, seeds = 1)       # Watts: 18% of the neighbors
      si["I"].iloc[-1]                                                     # final size of the cascade
      si = gsk.complex_si_model(net, 1, "cascade", "threshold", seeds = 5)   # thresholds from a node attribute

      si = gsk.complex_si_model(net, 0.1, "independent", seeds = 5)       # Kempe et al.: one attempt per edge
      si = gsk.complex_si_model(net, "p", "independent", seeds = 5)       # probabilities from edge attribute "p"
      si = gsk.complex_si_model(net, P, "decreasing", decay = 0.5)        # P: matrix, origins by targets

      si, forest = gsk.complex_si_model(net, 0.1, "independent", seeds = 3, forest = True)
      edges = forest.to_pandas_dataframes()                               # source infected target at step
      trees = gsk.connected_components_nodes(forest)                      # one tree per seed
   """
   return _simulate(net, "SI", beta, 0., seeds, ranking, selection, runs, max_steps, component, seed, engine, batch, attribute, resume, reinforcement = reinforcement, threshold = threshold, decay = decay, forest = forest, immunize = immunize, strategy = strategy)


def complex_sis_model(net, beta, mu, reinforcement = "classical", threshold = None, decay = 0.5, seeds = 1, ranking = None, selection = "top", runs = 100, max_steps = 1000, component = 0, seed = None, engine = None, batch = None, attribute = "state", resume = False, window = 100, tolerance = 0.01, forest = False, immunize = None, strategy = "random"):
   """
   Simulate the SIS model with complex contagion, and return the fraction of nodes in each state
   at every time step, averaged over independent runs.

   As :func:`~graphoskop.sis_model` (recoveries, plateau test), but a susceptible node is infected
   according to ``reinforcement`` (see the module description). A node can lose the infected
   neighbors that let it pass its threshold, and wait until it has enough again.

   :param net: A :class:`~graphoskop.Network`.
   :param beta: Infection probability, in (0, 1]: per infected neighbor (``"classical"``), once
      the threshold is met, or per attempt (cascades: one value, or a table per pair of nodes).
   :param mu: Recovery probability per step, in (0, 1].
   :param reinforcement: ``"classical"``, ``"threshold"``, ``"cascade"``, ``"independent"`` or
      ``"decreasing"``.
   :param threshold: Count (``"threshold"``) or share (``"cascade"``) of infected neighbors
      needed; one value, or one per node.
   :param decay: With ``"decreasing"``, the factor of the probability for every earlier failed
      attempt on the same node, in (0, 1].
   :param seeds: Number of seed nodes, or a fraction of the nodes if below 1.
   :param ranking: Node values used to choose the seeds, or ``None`` for random seeds.
   :param selection: ``"top"`` or ``"weighted"``.
   :param runs: Number of independent runs.
   :param max_steps: Largest number of steps of a run.
   :param component: Index of the component, or ``None`` for the whole network.
   :param seed: Seed of the random numbers, or ``None``.
   :param engine: ``"numba"``, ``"torch"`` or ``None``.
   :param batch: With the torch engine, the number of runs simulated at once, or ``None``.
   :param attribute: Name of the qualitative node attribute holding the states, or ``None``.
   :param resume: Whether to start from the states in ``attribute`` instead of seeds.
   :param forest: With ``"independent"`` or ``"decreasing"``, whether to return also who infected
      whom in the last run (see the module description).
   :param window: Steps of each window of the plateau test, or ``None`` not to stop at the
      plateau.
   :param tolerance: Largest relative difference between the two windows at the plateau.
   :param immunize: Nodes immunized at step 0, never infected (see
      :mod:`~graphoskop.epidemics.simple_contagion`): a number of nodes (a fraction of them if
      below 1), chosen by ``strategy``, or the node ids themselves; ``None`` for none.
   :param strategy: With a number of nodes to immunize: ``"random"``, ``"friend_of_random"``, or
      node values (vector, dictionary or :class:`pandas.Series`, name of a numeric node
      attribute) whose largest ones are immunized. Ignored with node ids.
   :return: :class:`pandas.DataFrame` indexed by step, with columns ``S``, ``S_low``,
      ``S_high``, ``I``, ``I_low`` and ``I_high``; with ``forest``, a pair (table, network of the
      infections).
   :raises ValueError: As :func:`~graphoskop.sis_model`, if ``reinforcement``, the thresholds,
      the probabilities per pair or ``decay`` are not valid, or if ``forest`` is asked for another
      kind of reinforcement.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("contacts.tsv")
      sis = gsk.complex_sis_model(net, 0.05, 0.2, "classical", seeds = 10)
      sis = gsk.complex_sis_model(net, 0.5, 0.1, "threshold", 2, seeds = 0.1)
   """
   return _simulate(net, "SIS", beta, mu, seeds, ranking, selection, runs, max_steps, component, seed, engine, batch, attribute, resume, window, tolerance, reinforcement, threshold, decay, forest, immunize = immunize, strategy = strategy)


def complex_sir_model(net, beta, mu, reinforcement = "classical", threshold = None, decay = 0.5, seeds = 1, ranking = None, selection = "top", runs = 100, max_steps = 1000, component = 0, seed = None, engine = None, batch = None, attribute = "state", resume = False, forest = False, immunize = None, strategy = "random"):
   """
   Simulate the SIR model with complex contagion, and return the fraction of nodes in each state
   at every time step, averaged over independent runs.

   As :func:`~graphoskop.sir_model`, but a susceptible node is infected according to
   ``reinforcement`` (see the module description). Removed nodes no longer count as infected
   neighbors, so in threshold models a node can miss its threshold when its infected neighbors
   recover before enough of them are infected at the same time.

   :param net: A :class:`~graphoskop.Network`.
   :param beta: Infection probability, in (0, 1]: per infected neighbor (``"classical"``), once
      the threshold is met, or per attempt (cascades: one value, or a table per pair of nodes).
   :param mu: Recovery probability per step, in (0, 1].
   :param reinforcement: ``"classical"``, ``"threshold"``, ``"cascade"``, ``"independent"`` or
      ``"decreasing"``.
   :param threshold: Count (``"threshold"``) or share (``"cascade"``) of infected neighbors
      needed; one value, or one per node.
   :param decay: With ``"decreasing"``, the factor of the probability for every earlier failed
      attempt on the same node, in (0, 1].
   :param seeds: Number of seed nodes, or a fraction of the nodes if below 1.
   :param ranking: Node values used to choose the seeds, or ``None`` for random seeds.
   :param selection: ``"top"`` or ``"weighted"``.
   :param runs: Number of independent runs.
   :param max_steps: Largest number of steps of a run.
   :param component: Index of the component, or ``None`` for the whole network.
   :param seed: Seed of the random numbers, or ``None``.
   :param engine: ``"numba"``, ``"torch"`` or ``None``.
   :param batch: With the torch engine, the number of runs simulated at once, or ``None``.
   :param attribute: Name of the qualitative node attribute holding the states (``"S"``,
      ``"I"`` or ``"R"``), or ``None``.
   :param resume: Whether to start from the states in ``attribute`` instead of seeds.
   :param forest: With ``"independent"`` or ``"decreasing"``, whether to return also who infected
      whom in the last run (see the module description).
   :param immunize: Nodes immunized at step 0, never infected (see
      :mod:`~graphoskop.epidemics.simple_contagion`): a number of nodes (a fraction of them if
      below 1), chosen by ``strategy``, or the node ids themselves; ``None`` for none.
   :param strategy: With a number of nodes to immunize: ``"random"``, ``"friend_of_random"``, or
      node values (vector, dictionary or :class:`pandas.Series`, name of a numeric node
      attribute) whose largest ones are immunized. Ignored with node ids.
   :return: :class:`pandas.DataFrame` indexed by step, with columns ``S``, ``I`` and ``R`` and
      their confidence intervals (``S_low``, ``S_high``, ...); with ``forest``, a pair (table,
      forest :class:`~graphoskop.Network`).
   :raises ValueError: As :func:`~graphoskop.sir_model`, if ``reinforcement``, the thresholds,
      the probabilities per pair or ``decay`` are not valid, or if ``forest`` is asked for another
      kind of reinforcement.

   Example:

   .. code-block:: python

      import graphoskop as gsk

      net = gsk.load_network("contacts.tsv")
      sir = gsk.complex_sir_model(net, 0.05, 0.1, "classical", seeds = 5)
      sir = gsk.complex_sir_model(net, 1, 0.2, "cascade", 0.1, seeds = 0.02)
      sir["R"].iloc[-1]                                                    # final size
      sir = gsk.complex_sir_model(net, 0.2, 0.1, "decreasing", decay = 0.7, seeds = 5)
   """
   return _simulate(net, "SIR", beta, mu, seeds, ranking, selection, runs, max_steps, component, seed, engine, batch, attribute, resume, reinforcement = reinforcement, threshold = threshold, decay = decay, forest = forest, immunize = immunize, strategy = strategy)
