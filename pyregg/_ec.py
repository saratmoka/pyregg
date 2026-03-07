#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba-accelerated version of ec.py.

naiveMC and conditionalMC are identical to ec.py.

ISMC replaces the numpy-based inner loop with a @njit-compiled kernel,
eliminating Python/numpy call overhead for the ~300-element neighbourhood
updates and enabling native-speed loops throughout.

Key changes versus ec.py ISMC:
  - BinnedPoints stored as a pre-allocated 4-D numpy array (nBins x nBins x
    max_pts_bin x 2) instead of a list-of-lists, so it can be passed to @njit.
  - The neighbourhood update loop (OrderMatrix + hist) is a plain Python for-
    loop inside @njit — compiled to native code, no numpy call overhead.
  - Squared-distance comparison replaces linalg.norm (avoids sqrt + function
    call overhead inside the edge-count loop).
  - Working arrays (OrderMatrix, BinnedPoints, BinCounts, hist, LHR) are pre-
    allocated once outside the outer loop and reset inside the kernel.
  - The first call to ISMC triggers JIT compilation (~2–5 s one-time cost,
    excluded from the reported timing).

Author: Dr Sarat Moka, UNSW Mathematics and Statistics.
"""

import time
import numpy as np
from numpy import linalg
from scipy.stats import poisson
from IPython.display import clear_output
from numba import njit


# ── Utility ──────────────────────────────────────────────────────────────────

def sci(x, digits=2):
    return f"{x:.{digits}e}".replace("e-0", "e-").replace("e+0", "e+")


# ── Helpers shared by naiveMC / conditionalMC (identical to ec.py) ───────────

def newEdges_bin_version(NewPoint, BinnedPoints, IntRange, BinEdg, nBins):
    count = 0
    bin_x = int(NewPoint[0] / BinEdg)
    bin_y = int(NewPoint[1] / BinEdg)
    for i in range(max(0, bin_x - 1), min(bin_x + 2, nBins)):
        for j in range(max(0, bin_y - 1), min(bin_y + 2, nBins)):
            for pt in BinnedPoints[i][j]:
                if linalg.norm(pt - NewPoint) < IntRange:
                    count += 1
    return count


def distBtwCells(xx, yy):
    dist = np.linalg.norm(xx - yy)
    dist = max(dist, np.linalg.norm(xx + [0, 1] - yy))
    dist = max(dist, np.linalg.norm(xx + [1, 1] - yy))
    dist = max(dist, np.linalg.norm(xx + [1, 0] - yy))
    dist = max(dist, np.linalg.norm(xx - yy - [0, 1]))
    dist = max(dist, np.linalg.norm(xx - yy - [1, 1]))
    dist = max(dist, np.linalg.norm(xx - yy - [1, 0]))
    return dist


def generateNeighborsMatrix(GridEdg, IntRange):
    spread = int(IntRange / GridEdg) - 1
    if spread < 0:
        print("Error: cell diagonal length is bigger than IntRange")
    arr_size = 2 * spread + 1
    Neighbors = np.zeros((arr_size, arr_size), dtype=np.int32)
    center = np.array((spread, spread))
    for x in range(arr_size):
        for y in range(arr_size):
            dist = distBtwCells(np.array((x, y)), center)
            if dist * GridEdg <= IntRange:
                Neighbors[x][y] = 1
    return Neighbors


# ── Numba kernels ─────────────────────────────────────────────────────────────

@njit
def _seed_numba(seed):
    """Seed Numba's internal RNG (separate from numpy's global RNG)."""
    np.random.seed(seed)


@njit
def _ismc_sample(OrderMatrix, BinnedPoints, BinCounts, hist, LHR,
                 Neighbors, spread, GridSize, nBins,
                 GridEdg, BinEdg, ir2, Level,
                 total_cells, hist_len, q):
    """
    Run one IS sample (the full inner loop) entirely in native code.

    All working arrays are reset in-place at the start so they can be
    pre-allocated once and reused across outer iterations.

    Returns
    -------
    Y_tilde : float64
        The IS likelihood-ratio estimator q · LHR for this sample.
    """
    # ── Reset working arrays in-place ────────────────────────────────────────
    OrderMatrix.fill(0)
    BinCounts.fill(0)
    hist.fill(0)
    LHR.fill(0.0)
    hist[0] = total_cells
    LHR[0] = 1.0

    NonBlockCount = total_cells
    threshold     = Level
    EdgeCount     = 0
    n_max         = len(LHR) - 1

    for n in range(n_max):

        if NonBlockCount == 0:
            break

        # ── Select a random non-blocked cell ─────────────────────────────────
        # Dense regime (> 5 % non-blocked): rejection sampling, O(1) expected.
        # Sparse regime: sequential scan, O(total_cells) but compiled to native.
        if NonBlockCount * 20 > total_cells:
            while True:
                flat = np.random.randint(0, total_cells)
                row  = flat // GridSize
                col  = flat %  GridSize
                if OrderMatrix[row, col] <= threshold:
                    break
        else:
            chosen = np.random.randint(0, NonBlockCount)
            cnt    = 0
            row    = 0
            col    = 0
            done   = False
            for ii in range(GridSize):
                for jj in range(GridSize):
                    if OrderMatrix[ii, jj] <= threshold:
                        if cnt == chosen:
                            row  = ii
                            col  = jj
                            done = True
                            break
                        cnt += 1
                if done:
                    break

        # LHR uses NonBlockCount from BEFORE this step's updates
        LHR[n + 1] = LHR[n] * NonBlockCount / total_cells

        # ── Generate point uniformly inside the selected cell ─────────────────
        px = (row + np.random.random()) * GridEdg
        py = (col + np.random.random()) * GridEdg

        # ── Update OrderMatrix and hist for the neighbourhood ─────────────────
        x_left  = row         if row         < spread     else spread
        x_right = GridSize - row if GridSize - row < spread + 1 else spread + 1
        y_left  = col         if col         < spread     else spread
        y_right = GridSize - col if GridSize - col < spread + 1 else spread + 1

        for di in range(-x_left, x_right):
            for dj in range(-y_left, y_right):
                if Neighbors[spread + di, spread + dj] == 1:
                    ii    = row + di
                    jj    = col + dj
                    old_v = OrderMatrix[ii, jj]
                    new_v = old_v + 1
                    OrderMatrix[ii, jj] = new_v
                    if old_v <= threshold:
                        hist[old_v] -= 1
                        if new_v <= threshold:
                            hist[new_v] += 1
                        else:
                            NonBlockCount -= 1

        # ── Count new edges (squared distance, no sqrt) ───────────────────────
        bin_x  = int(px / BinEdg)
        bin_y  = int(py / BinEdg)
        lo_i   = bin_x - 1 if bin_x > 0      else 0
        hi_i   = bin_x + 2 if bin_x + 2 <= nBins else nBins
        lo_j   = bin_y - 1 if bin_y > 0      else 0
        hi_j   = bin_y + 2 if bin_y + 2 <= nBins else nBins
        new_edges = 0
        for i in range(lo_i, hi_i):
            for j in range(lo_j, hi_j):
                for k in range(BinCounts[i, j]):
                    dx = BinnedPoints[i, j, k, 0] - px
                    dy = BinnedPoints[i, j, k, 1] - py
                    if dx * dx + dy * dy < ir2:
                        new_edges += 1
        EdgeCount += new_edges

        if EdgeCount > Level:
            LHR[n + 1] = 0.0
            break

        # ── Store new point ───────────────────────────────────────────────────
        k = BinCounts[bin_x, bin_y]
        BinnedPoints[bin_x, bin_y, k, 0] = px
        BinnedPoints[bin_x, bin_y, k, 1] = py
        BinCounts[bin_x, bin_y] = k + 1

        # ── Tighten threshold when EdgeCount grows ────────────────────────────
        new_threshold = Level - EdgeCount
        if new_threshold < threshold:
            for v in range(new_threshold + 1, threshold + 1):
                NonBlockCount -= hist[v]
                hist[v] = 0
            threshold = new_threshold

    # ── Y_tilde = q · LHR ────────────────────────────────────────────────────
    return np.dot(q, LHR)


# ── naiveMC (identical to ec.py) ─────────────────────────────────────────────

def naiveMC(WindLen, Kappa, IntRange, Level, MaxIter=10**8, WarmUp=100000, Tol=0.001, Seed=None):
    if Seed is not None: np.random.seed(Seed)
    ExpPoiCount = Kappa * (WindLen ** 2)
    BinSize     = int(WindLen / IntRange)
    BinEdg      = WindLen / BinSize
    MeanEst     = 0.0
    Time        = 0.0
    Patience    = 0
    l           = 0
    stop        = False
    print("Warming up ...... ")

    while not stop:
        tic = time.process_time()
        l  += 1
        BinnedPoints = [[[] for _ in range(BinSize)] for _ in range(BinSize)]
        EdgeCount    = 0
        N            = np.random.poisson(ExpPoiCount)
        Y            = 1
        n            = 1
        while Y == 1 and n <= N:
            NewPoint  = WindLen * np.random.random_sample(2)
            n        += 1
            EdgeCount += newEdges_bin_version(NewPoint, BinnedPoints, IntRange, BinEdg, BinSize)
            bin_x = int(NewPoint[0] / BinEdg)
            bin_y = int(NewPoint[1] / BinEdg)
            BinnedPoints[bin_x][bin_y].append(NewPoint)
            if EdgeCount > Level:
                Y = 0

        MeanEst = ((l - 1) * MeanEst + Y) / l
        RV      = (1 / MeanEst - 1) if MeanEst != 0 else np.inf
        toc     = time.process_time()
        Time   += toc - tic

        if l % WarmUp == 0:
            clear_output(wait=True)
            print('\n----- Iteration: ', l, '-----')
            print('\nMean estimate Z (NMC):', sci(MeanEst))
            print('Relative variance of Y (NMC):', sci(RV))
            print('Relative variance of Z (NMC):', sci(RV / l))

        if l >= WarmUp:
            Patience = Patience + 1 if RV / l < Tol else 0

        if Patience >= 100 or l >= MaxIter:
            stop = True

    return {"mean": MeanEst, "mse": MeanEst, "time": Time, "niter": l}


# ── conditionalMC (identical to ec.py) ───────────────────────────────────────

def conditionalMC(WindLen, Kappa, IntRange, Level, MaxIter=10**8, WarmUp=10000, Tol=0.001, Seed=None):
    if Seed is not None: np.random.seed(Seed)
    ExpPoissonCount = Kappa * (WindLen ** 2)
    nBins           = int(WindLen / IntRange)
    BinEdg          = WindLen / nBins
    MeanEst         = 0.0
    MeanSqrEst      = 0.0
    Time            = 0.0
    Patience        = 0
    l               = 0
    stop            = False
    print("Warming up ...... ")

    while not stop:
        tic = time.process_time()
        l  += 1
        BinnedPoints = [[[] for _ in range(nBins)] for _ in range(nBins)]
        EdgeCount    = 0
        n            = 0
        while EdgeCount <= Level:
            NewPoint   = WindLen * np.random.random_sample(2)
            n         += 1
            EdgeCount  += newEdges_bin_version(NewPoint, BinnedPoints, IntRange, BinEdg, nBins)
            bin_x = int(NewPoint[0] / BinEdg)
            bin_y = int(NewPoint[1] / BinEdg)
            BinnedPoints[bin_x][bin_y].append(NewPoint)

        Y_hat      = poisson.cdf(n - 1, ExpPoissonCount)
        MeanEst    = ((l - 1) * MeanEst    + Y_hat)          / l
        MeanSqrEst = ((l - 1) * MeanSqrEst + Y_hat * Y_hat)  / l
        RV         = MeanSqrEst / (MeanEst ** 2) - 1
        toc        = time.process_time()
        Time      += toc - tic

        if l % WarmUp == 0:
            clear_output(wait=True)
            print('\n----- Iteration: ', l, '-----')
            print('\nMean estimate Z (CMC):', sci(MeanEst))
            print('Relative variance of Y_hat (CMC):', sci(RV))
            print('Relative variance of Z (CMC):', sci(RV / l))

        if l >= WarmUp:
            Patience = Patience + 1 if RV / l < Tol else 0

        if Patience >= 100 or l >= MaxIter:
            stop = True

    return {"mean": MeanEst, "mse": MeanSqrEst, "time": Time, "niter": l}


# ── ISMC (Numba-accelerated) ──────────────────────────────────────────────────

def ISMC(WindLen, GridRes, Kappa, IntRange, Level, MaxIter=10**8, WarmUp=1000, Tol=0.001, Seed=None):
    """
    Numba-accelerated IS Monte Carlo for edge-count rare events.

    Interface identical to ec.ISMC.  The first call triggers JIT compilation
    (~2–5 s); that one-time cost is excluded from the reported timing.
    """
    if Seed is not None:
        np.random.seed(Seed)
        _seed_numba(Seed)

    ExpPoissonCount = Kappa * (WindLen ** 2)
    nBins           = int(WindLen / IntRange)
    GridSize        = int(nBins * GridRes)
    GridEdg         = WindLen / GridSize
    BinEdg          = WindLen / nBins
    npts            = int(2 * ExpPoissonCount)
    ir2             = IntRange * IntRange

    q         = np.array([poisson.pmf(k, ExpPoissonCount) for k in range(npts + 1)])
    Neighbors = generateNeighborsMatrix(GridEdg, IntRange)
    Neighbors = Neighbors.astype(np.int32)
    spread    = (Neighbors.shape[0] - 1) // 2
    total_cells = GridSize * GridSize
    hist_len    = Level + 2

    # Pre-allocate working arrays once; _ismc_sample resets them each call.
    # max_pts_bin: conservative upper bound on points per bin per IS trajectory.
    max_pts_bin  = npts + 1
    OrderMatrix  = np.zeros((GridSize, GridSize),             dtype=np.int32)
    BinnedPoints = np.zeros((nBins, nBins, max_pts_bin, 2),   dtype=np.float64)
    BinCounts    = np.zeros((nBins, nBins),                   dtype=np.int32)
    hist         = np.zeros(hist_len,                         dtype=np.int64)
    LHR          = np.zeros(npts + 1,                         dtype=np.float64)

    # ── Trigger JIT compilation with one dry run (result discarded) ───────────
    print("Compiling Numba kernel (one-time) ...")
    _ = _ismc_sample(OrderMatrix, BinnedPoints, BinCounts, hist, LHR,
                     Neighbors, spread, GridSize, nBins,
                     GridEdg, BinEdg, ir2, Level,
                     total_cells, hist_len, q)
    print("Warming up ...... ")

    MeanEst    = 0.0
    MeanSqrEst = 0.0
    Time       = 0.0
    Patience   = 0
    l          = 0
    stop       = False

    while not stop:
        tic = time.process_time()
        l  += 1

        Y_tilde = _ismc_sample(OrderMatrix, BinnedPoints, BinCounts, hist, LHR,
                               Neighbors, spread, GridSize, nBins,
                               GridEdg, BinEdg, ir2, Level,
                               total_cells, hist_len, q)

        MeanEst    = ((l - 1) * MeanEst    + Y_tilde)           / l
        MeanSqrEst = ((l - 1) * MeanSqrEst + Y_tilde * Y_tilde) / l
        RV         = MeanSqrEst / (MeanEst ** 2) - 1
        toc        = time.process_time()
        Time      += toc - tic

        if l % WarmUp == 0:
            clear_output(wait=True)
            print('\n----- Iteration: ', l, '-----')
            print('\nGrid size:', GridSize, 'x', GridSize)
            print('Mean estimate Z (IS):', sci(MeanEst))
            print('Relative variance of Y_tilde (IS):', sci(RV))
            print('Relative variance of Z (IS):', sci(RV / l))

        if l >= WarmUp:
            Patience = Patience + 1 if RV / l < Tol else 0

        if Patience >= 100 or l >= MaxIter:
            stop = True

    return {"mean": MeanEst, "mse": MeanSqrEst, "time": Time, "niter": l}
