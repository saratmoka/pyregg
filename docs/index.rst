pyregg — Rare-Event Simulation for Random Geometric Graphs
===========================================================

**pyregg** estimates the probability of rare events in Gilbert random geometric
graphs using three Monte Carlo estimators:

- **Naïve Monte Carlo** (NMC) — independent realisations of the graph
- **Conditional Monte Carlo** (CMC) — sequential point addition with analytic conditioning
- **Importance Sampling** (IS) — sequential point addition with cell blocking and likelihood-ratio correction

IS achieves substantially lower variance than CMC, which in turn outperforms NMC,
enabling estimation of probabilities as small as 10⁻⁶ and beyond.

Installation
------------

.. code-block:: bash

   pip install pyregg

Modules
-------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Rare event
   * - :mod:`pyregg.ec`
     - P(edge count ≤ ℓ)
   * - :mod:`pyregg.md`
     - P(maximum degree ≤ ℓ)
   * - :mod:`pyregg.mcc`
     - P(maximum connected component size ≤ ℓ)
   * - :mod:`pyregg.ntg`
     - P(number of triangles ≤ ℓ)
   * - :mod:`pyregg.mcs`
     - P(maximum clique size ≤ ℓ)
   * - :mod:`pyregg.planar`
     - P(graph is planar)
   * - :mod:`pyregg.forest`
     - P(graph is a forest / acyclic)

Quick Start
-----------

Each module exposes three functions with a common interface returning
``(probability, rel_variance, n_samples)``.

.. code-block:: python

   import pyregg.ec as ec

   # P(EC(G(X)) ≤ 15) on [0,10]² with κ = 0.3, r = 1
   Z, RV, n = ec.importance_sampling(wind_len=10, kappa=0.3, int_range=1.0, level=15)
   print(f"P ≈ {Z:.4e}  (relative variance {RV:.2f},  {n} samples)")

.. code-block:: python

   import pyregg.planar as planar

   # P(G(X) is planar) on [0,10]² with κ = 1.2, r = 1
   Z, RV, n = planar.importance_sampling(wind_len=10, kappa=1.2, int_range=1.0)
   print(f"P ≈ {Z:.4e}  (relative variance {RV:.2f},  {n} samples)")

.. code-block:: python

   import pyregg.forest as forest

   # P(G(X) is a forest) on [0,10]² with κ = 0.3, r = 1
   Z, RV, n = forest.importance_sampling(wind_len=10, kappa=0.3, int_range=1.0)
   print(f"P ≈ {Z:.4e}  (relative variance {RV:.2f},  {n} samples)")

API Reference
-------------

.. toctree::
   :maxdepth: 2

   ec
   md
   mcc
   ntg
   mcs
   planar
   forest

References
----------

- S. Moka, C. Hirsch, V. Schmidt & D. P. Kroese (2025).
  *Efficient Rare-Event Simulation for Random Geometric Graphs via Importance Sampling.*
  `arXiv:2504.10530 <https://arxiv.org/abs/2504.10530>`_

- C. Hirsch, S. B. Moka, T. Taimre & D. P. Kroese (2022).
  *Rare Events in Random Geometric Graphs.*
  Methodology and Computing in Applied Probability, 24, 1367–1383.
  `doi:10.1007/s11009-021-09857-7 <https://link.springer.com/article/10.1007/s11009-021-09857-7>`_
