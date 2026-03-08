"""
pyregg: Rare-Event Simulation for Random Geometric Graphs
==========================================================

Estimates rare-event probabilities in Gilbert random geometric graphs
using three estimators: Naïve Monte Carlo, Conditional Monte Carlo,
and Importance Sampling.

Submodules
----------
ec      : Edge count does not exceed a threshold
md      : Maximum degree does not exceed a threshold
mcc     : Maximum connected component size does not exceed a threshold
ntg     : Number of triangles does not exceed a threshold
mcs     : Maximum clique size does not exceed a threshold
planar  : Graph is planar
forest  : Graph is a forest (acyclic)

Quick Start
-----------
>>> import pyregg.ec as ec
>>> Z, RV, n = ec.importance_sampling(wind_len=10, kappa=0.3, int_range=1.0, level=15)

>>> import pyregg.planar as planar
>>> Z, RV, n = planar.importance_sampling(wind_len=10, kappa=1.2, int_range=1.0)

>>> import pyregg.forest as forest
>>> Z, RV, n = forest.importance_sampling(wind_len=10, kappa=0.3, int_range=1.0)
"""

from pyregg import ec, md, mcc, ntg, mcs, planar, forest

__version__ = "0.2.12"
__author__  = "Sarat Moka"
__all__     = ["ec", "md", "mcc", "ntg", "mcs", "planar", "forest"]
