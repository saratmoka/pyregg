"""
Maximum Clique Size (MCS)
==========================
Rare-event estimation for the probability that the size of the largest
clique in a Gilbert random geometric graph G(X) does not exceed a
threshold ℓ:

    P(MCS(G(X)) ≤ ℓ)

Three estimators are provided: Naïve Monte Carlo, Conditional Monte Carlo,
and Importance Sampling Monte Carlo.

Note
----
The IS estimator currently supports level ∈ {1, 2} only (triangle-free
and K₄-free graphs, respectively).

References
----------
Hirsch, C., Moka, S. B., Taimre, T., & Kroese, D. P. (2022).
    Rare events in random geometric graphs.
    *Methodology and Computing in Applied Probability*, 24, 1367–1383.

Moka, S., Hirsch, C., Schmidt, V., & Kroese, D. P. (2025).
    Efficient rare-event simulation for random geometric graphs via
    importance sampling. arXiv:2504.10530.
"""

from pyregg._mcs import (
    naiveMC       as _naiveMC,
    conditionalMC as _conditionalMC,
    ISMC          as _ISMC,
)

__all__ = ["naive_mc", "conditional_mc", "importance_sampling"]


def _unwrap(r):
    mean = r["mean"]
    rv   = r["mse"] / mean**2 - 1 if mean > 0 else float("inf")
    return mean, rv, r["niter"]


def naive_mc(wind_len, kappa, int_range, level,
             max_iter=10**8, warm_up=100_000, tol=0.001, *, seed=None):
    """
    Estimate P(MCS(G(X)) ≤ level) using Naïve Monte Carlo.

    Parameters
    ----------
    wind_len : float
        Side length of the square observation window [0, wind_len]².
    kappa : float
        Intensity of the Poisson point process (expected points per unit area).
    int_range : float
        Interaction range (connection radius) of the Gilbert graph.
    level : int
        Threshold ℓ. The rare event is {MCS(G(X)) ≤ level}.
    max_iter : int, optional
        Maximum number of samples. Default is 10**8.
    warm_up : int, optional
        Minimum samples before checking convergence. Default is 100,000.
    tol : float, optional
        Stop when estimated RV / n < tol. Default is 0.001.
    seed : int, optional
        Integer seed for reproducibility (keyword-only). By default no seed
        is set and results are non-deterministic.

    Returns
    -------
    probability : float
        Estimated rare-event probability P(MCS(G(X)) ≤ level).
    rel_variance : float
        Estimated relative variance of the estimator.
    n_samples : int
        Number of samples used.

    Examples
    --------
    >>> import pyregg.mcs as mcs
    >>> Z, RV, n = mcs.naive_mc(wind_len=10, kappa=0.5, int_range=1.0, level=1)
    """
    return _unwrap(_naiveMC(wind_len, kappa, int_range, level + 1, max_iter, warm_up, tol, seed))


def conditional_mc(wind_len, kappa, int_range, level,
                   max_iter=10**8, warm_up=1_000, tol=0.001, *, seed=None):
    """
    Estimate P(MCS(G(X)) ≤ level) using Conditional Monte Carlo.

    Parameters
    ----------
    wind_len : float
        Side length of the square observation window [0, wind_len]².
    kappa : float
        Intensity of the Poisson point process (expected points per unit area).
    int_range : float
        Interaction range (connection radius) of the Gilbert graph.
    level : int
        Threshold ℓ. The rare event is {MCS(G(X)) ≤ level}.
    max_iter : int, optional
        Maximum number of samples. Default is 10**8.
    warm_up : int, optional
        Minimum samples before checking convergence. Default is 1,000.
    tol : float, optional
        Stop when estimated RV / n < tol. Default is 0.001.
    seed : int, optional
        Integer seed for reproducibility (keyword-only). By default no seed
        is set and results are non-deterministic.

    Returns
    -------
    probability : float
        Estimated rare-event probability P(MCS(G(X)) ≤ level).
    rel_variance : float
        Estimated relative variance of the estimator.
    n_samples : int
        Number of samples used.

    Examples
    --------
    >>> import pyregg.mcs as mcs
    >>> Z, RV, n = mcs.conditional_mc(wind_len=10, kappa=0.5, int_range=1.0, level=1)
    """
    return _unwrap(_conditionalMC(wind_len, kappa, int_range, level + 1, max_iter, warm_up, tol, seed))


def importance_sampling(wind_len, kappa, int_range, level, grid_res=10,
                        max_iter=10**8, warm_up=100, tol=0.001, *, seed=None):
    """
    Estimate P(MCS(G(X)) ≤ level) using Importance Sampling Monte Carlo.

    Cells where placing a new point would form a clique of size level + 1
    are blocked. A likelihood-ratio correction yields an unbiased estimate
    with substantially lower variance than CMC.

    Note: level must be 1 or 2 (triangle-free or K₄-free graphs).

    Parameters
    ----------
    wind_len : float
        Side length of the square observation window [0, wind_len]².
    kappa : float
        Intensity of the Poisson point process (expected points per unit area).
    int_range : float
        Interaction range (connection radius) of the Gilbert graph.
    level : int
        Threshold ℓ ∈ {1, 2}. The rare event is {MCS(G(X)) ≤ level}.
    grid_res : int, optional
        Number of grid cells per interaction-range interval. The window is
        divided into (wind_len / int_range × grid_res)² cells total.
        Default is 10.
    max_iter : int, optional
        Maximum number of IS samples. Default is 10**8.
    warm_up : int, optional
        Minimum samples before checking convergence. Default is 100.
    tol : float, optional
        Stop when estimated RV / n < tol. Default is 0.001.
    seed : int, optional
        Integer seed for reproducibility (keyword-only). By default no seed
        is set and results are non-deterministic.

    Returns
    -------
    probability : float
        Estimated rare-event probability P(MCS(G(X)) ≤ level).
    rel_variance : float
        Estimated relative variance of the IS estimator.
    n_samples : int
        Number of IS samples used.

    Examples
    --------
    >>> import pyregg.mcs as mcs
    >>> Z, RV, n = mcs.importance_sampling(wind_len=10, kappa=0.5, int_range=1.0,
    ...                                    level=1, grid_res=10)
    """
    return _unwrap(_ISMC(wind_len, grid_res, kappa, int_range, level + 1, max_iter, warm_up, tol, seed))
