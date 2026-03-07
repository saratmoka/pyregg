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

Each module exposes three functions: `naive_mc`, `conditional_mc`, and
`importance_sampling`. All return `(probability, rel_variance, n_samples)`.

```python
import pyregg.ec as ec

# Estimate P(EC(G(X)) ≤ 15) on [0,10]² with κ=0.3, r=1
Z, RV, n = ec.importance_sampling(wind_len=10, kappa=0.3, int_range=1.0, level=15)
print(f"P ≈ {Z:.4e}  (relative variance {RV:.2f},  {n} samples)")
```

```python
import pyregg.planar as planar

# Estimate P(G(X) is planar) on [0,10]² with κ=1.2, r=1
Z, RV, n = planar.importance_sampling(wind_len=10, kappa=1.2, int_range=1.0)
print(f"P ≈ {Z:.4e}  (relative variance {RV:.2f},  {n} samples)")
```

```python
import pyregg.forest as forest

# Estimate P(G(X) is a forest) on [0,10]² with κ=0.3, r=1
Z, RV, n = forest.importance_sampling(wind_len=10, kappa=0.3, int_range=1.0)
print(f"P ≈ {Z:.4e}  (relative variance {RV:.2f},  {n} samples)")
```

## API

### Common parameters

| Parameter | Description |
|-----------|-------------|
| `wind_len` | Side length of the square window [0, `wind_len`]² |
| `kappa` | Intensity of the Poisson point process (points per unit area) |
| `int_range` | Connection radius — two points are connected if their distance ≤ `int_range` |
| `level` | Threshold ℓ (not used for `planar` or `forest`) |
| `grid_res` | IS grid cells per interaction-range interval; total cells = `(wind_len / int_range × grid_res)²` (IS only, default 10) |
| `max_iter` | Maximum number of samples (default 10⁸) |
| `warm_up` | Minimum samples before checking convergence |
| `tol` | Stop when relative variance / n < `tol` (default 0.001) |

### Estimators

**`naive_mc(...)`** — Independent realisations; fraction satisfying the rare event.

**`conditional_mc(...)`** — Sequential point addition with analytic conditioning at each step.

**`importance_sampling(...)`** — Sequential point addition with cells that would violate the
rare event *blocked*; likelihood-ratio correction ensures unbiasedness.

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
