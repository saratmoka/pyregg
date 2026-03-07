# pyregg

**Rare-event simulation for random geometric graphs.**

`pyregg` estimates the probability of rare events in Gilbert random geometric
graphs using three estimators: Naïve Monte Carlo (NMC), Conditional Monte Carlo
(CMC), and Importance Sampling (IS).

## Installation

```bash
pip install pyregg
```

## Rare Events

| Module | Rare Event |
|--------|------------|
| `pyregg.ec` | Edge count ≤ ℓ |
| `pyregg.md` | Maximum degree ≤ ℓ |
| `pyregg.mcc` | Maximum connected component size ≤ ℓ |
| `pyregg.ntg` | Number of triangles ≤ ℓ |
| `pyregg.mcs` | Maximum clique size ≤ ℓ |
| `pyregg.planar` | Graph is planar |
| `pyregg.forest` | Graph is a forest (acyclic) |

## Quick Start

Each module is **directly callable** with an optional `method` argument
(`'ismc'` by default). All calls return `(probability, rel_variance, n_samples)`.

```python
import pyregg.ec as ec

# Estimate P(EC(G(X)) ≤ 15) on [0,10]² with κ=0.3, r=1
Z, RV, n = ec(wind_len=10, kappa=0.3, int_range=1.0, level=15)
print(f"P ≈ {Z:.4e}  (relative variance {RV:.2f},  {n} samples)")
```

```python
import pyregg.planar as planar

# Estimate P(G(X) is planar) on [0,10]² with κ=1.2, r=1
Z, RV, n = planar(wind_len=10, kappa=1.2, int_range=1.0)
print(f"P ≈ {Z:.4e}  (relative variance {RV:.2f},  {n} samples)")
```

```python
import pyregg.forest as forest

# Estimate P(G(X) is a forest) on [0,10]² with κ=0.3, r=1
Z, RV, n = forest(wind_len=10, kappa=0.3, int_range=1.0)
print(f"P ≈ {Z:.4e}  (relative variance {RV:.2f},  {n} samples)")
```

## API

### Calling a module

```python
module(wind_len, kappa, int_range, [level,] method='ismc', **kwargs)
```

`method` selects the estimator: `'ismc'` (Importance Sampling, default),
`'cmc'` (Conditional Monte Carlo), or `'nmc'` (Naïve Monte Carlo).
`**kwargs` are forwarded to the chosen estimator (e.g. `grid_res`, `tol`).

The three estimators are also available as named functions:
`module.naive_mc(...)`, `module.conditional_mc(...)`,
`module.importance_sampling(...)`.

### Common parameters

| Parameter | Description |
|-----------|-------------|
| `wind_len` | Side length of the square window [0, `wind_len`]² |
| `kappa` | Intensity of the Poisson point process (points per unit area) |
| `int_range` | Connection radius — two points are connected if their distance ≤ `int_range` |
| `level` | Threshold ℓ (not used for `planar` or `forest`) |
| `method` | Estimator: `'ismc'` (default), `'cmc'`, or `'nmc'` |
| `grid_res` | IS grid cells per interaction-range interval; total cells = `(wind_len / int_range × grid_res)²` (IS only, default 10) |
| `max_iter` | Maximum number of samples (default 10⁸) |
| `warm_up` | Minimum samples before checking convergence |
| `tol` | Stop when relative variance / n < `tol` (default 0.001) |

### Estimators

**`'ismc'`** (default) — Sequential point addition with cells that would violate the
rare event *blocked*; likelihood-ratio correction ensures unbiasedness.

**`'cmc'`** — Sequential point addition with analytic conditioning at each step.

**`'nmc'`** — Independent realisations; fraction satisfying the rare event.

## Dependencies

```
Python  >= 3.10
NumPy   >= 1.24
SciPy   >= 1.10
Numba   >= 0.57
NetworkX >= 3.0
```

## References

- S. Moka, C. Hirsch, V. Schmidt & D. P. Kroese (2025).
  *Efficient Rare-Event Simulation for Random Geometric Graphs via Importance Sampling.*
  arXiv:2504.10530. https://arxiv.org/abs/2504.10530

- C. Hirsch, S. B. Moka, T. Taimre & D. P. Kroese (2022).
  *Rare Events in Random Geometric Graphs.*
  Methodology and Computing in Applied Probability, 24, 1367–1383.
  https://link.springer.com/article/10.1007/s11009-021-09857-7
