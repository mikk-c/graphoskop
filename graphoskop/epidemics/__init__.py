"""
Epidemics: models of how something (a disease, a piece of information, a behavior) spreads from
node to node through the edges of a network.

Nodes are in compartments (states), and move between them by contagion from their neighbors or
spontaneously. In simple contagion (:mod:`graphoskop.epidemics.simple_contagion`), one infected
neighbor is enough to infect a node: SI, SIS and SIR models. In complex contagion
(:mod:`graphoskop.epidemics.complex_contagion`), the number of infected neighbors matters: one
chance of infection per infected neighbor, or a threshold on their count or share (which, in the
non progressive model, also makes infected nodes susceptible again when they fall below it). In
higher-order contagion (:mod:`graphoskop.epidemics.higher_order_contagion`), on simplicial
complexes and hypergraphs, groups of nodes infect: a triangle or a hyperedge infects its last
susceptible node once all its other nodes are infected (the simplicial contagion model).
"""

from .simple_contagion import si_model, sis_model, sir_model
from .complex_contagion import complex_si_model, complex_sis_model, complex_sir_model, complex_nonprogressive_model
from .higher_order_contagion import higher_order_si_model, higher_order_sis_model, higher_order_sir_model
