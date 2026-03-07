#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba-accelerated version of md.py.

naiveMC and conditionalMC are identical to md.py.

ISMC replaces the Python/numpy inner loop with a @njit-compiled kernel.
The main conversions required versus md.py:

  - non_blocked / cell_pos Python lists  →  pre-allocated int32 numpy arrays
    with an explicit n_non_blocked counter; swap-and-pop becomes pure array ops.
  - BinnedPoints list-of-(point, id) tuples  →  separate BinnedPts / BinnedIds
    / BinCounts numpy arrays.
  - degrees / node_row / node_col / node_blocked Python lists  →  pre-allocated
    numpy arrays indexed by node_count.
  - Neighbors list of numpy arrays  →  2-D int32 array of shape (n_nb, 2).
  - linalg.norm  →  squared-distance comparison (no sqrt, no function call).

The first call to ISMC triggers JIT compilation (~2-5 s, excluded from timing).

Author: Dr Sarat Moka, UNSW Mathematics and Statistics.
"""

import numpy as np
from scipy.stats import poisson
from scipy.spatial import cKDTree
import time
from IPython.display import clear_output
from numba import njit


# ── Utility ───────────────────────────────────────────────────────────────────

def sci(x, digits=2):
    return f"{x:.{digits}e}".replace("e-0", "e-").replace("e+0", "e+")


# ── Helpers shared by naiveMC / conditionalMC (identical to md.py) ────────────

def maxDegree(NumPoints, WindLen, IntRange):
    if NumPoints == 0:
        return 0
    points = np.random.uniform(0, WindLen, (NumPoints, 2))
    tree   = cKDTree(points)
    counts = tree.query_ball_point(points, IntRange, return_length=True)
    return int(np.max(counts)) - 1


def generatePointsUntilMaxDegree(MaxNumPoints, WindLen, IntRange, Level):
    nBins        = max(1, int(WindLen / IntRange))
    BinEdg       = WindLen / nBins
    BinnedPoints = [[[] for _ in range(nBins)] for _ in range(nBins)]
    degrees      = []
    MaxDeg       = 0
    node_count   = 0

    for i in range(MaxNumPoints):
        new_point = np.random.uniform(0, WindLen, 2)
        bin_x = min(int(new_point[0] / BinEdg), nBins - 1)
        bin_y = min(int(new_point[1] / BinEdg), nBins - 1)

        neighbor_ids = []
        for bx in range(max(0, bin_x - 1), min(bin_x + 2, nBins)):
            for by in range(max(0, bin_y - 1), min(bin_y + 2, nBins)):
                for pt, nid in BinnedPoints[bx][by]:
                    if np.linalg.norm(pt - new_point) < IntRange:
                        neighbor_ids.append(nid)

        new_deg = len(neighbor_ids)
        MaxDeg  = max(MaxDeg, new_deg)
        for nid in neighbor_ids:
            degrees[nid] += 1
            if degrees[nid] > MaxDeg:
                MaxDeg = degrees[nid]

        degrees.append(new_deg)
        BinnedPoints[bin_x][bin_y].append((new_point, node_count))
        node_count += 1

        if MaxDeg > Level:
            return MaxDeg, i

    return MaxDeg, MaxNumPoints


def distBtwCells(xx, yy):
    dist = np.linalg.norm(xx - yy)
    dist = max(dist, np.linalg.norm(xx + [0, 1] - yy))
    dist = max(dist, np.linalg.norm(xx + [1, 1] - yy))
    dist = max(dist, np.linalg.norm(xx + [1, 0] - yy))
    dist = max(dist, np.linalg.norm(xx - yy - [0, 1]))
    dist = max(dist, np.linalg.norm(xx - yy - [1, 1]))
    dist = max(dist, np.linalg.norm(xx - yy - [1, 0]))
    return dist


def generateNeighbors(GridEdg, IntRange):
    spread   = int(IntRange / GridEdg) - 1
    if spread < 0:
        print("Error: cell diagonal length is bigger than IntRange")
    arr_size = 2 * spread + 1
    center   = np.array((spread, spread))
    offsets  = []
    for x in range(arr_size):
        for y in range(arr_size):
            cell = np.array((x, y))
            dist = distBtwCells(cell, center)
            if dist * GridEdg <= IntRange:
                offsets.append(cell - center)
    # Return as 2-D int32 array for Numba compatibility
    return np.array(offsets, dtype=np.int32)


# ── Numba kernel ──────────────────────────────────────────────────────────────

@njit
def _seed_numba(seed):
    """Seed Numba's internal RNG."""
    np.random.seed(seed)


@njit
def _md_ismc_sample(OrderMatrix, BlockMatrix,
                    non_blocked, cell_pos,
                    BinnedPts, BinnedIds, BinCounts,
                    degrees, node_row, node_col, node_blocked,
                    neighbor_buf, LHR,
                    Neighbors, n_nb,
                    GridSize, nBins, GridEdg, BinEdg,
                    ir2, Level, total_cells, q):
    """
    Run one IS sample (inner loop) entirely in native code.

    All working arrays are reset in-place and reused across outer iterations.
    Returns Y_tilde = q · LHR for this sample.
    """
    # ── Reset working arrays ──────────────────────────────────────────────────
    OrderMatrix.fill(0)
    BlockMatrix.fill(0)
    BinCounts.fill(0)
    degrees.fill(0)
    node_blocked.fill(0)
    LHR.fill(0.0)
    LHR[0] = 1.0

    # Restore non_blocked = [0, 1, ..., total_cells-1] and cell_pos = identity
    n_non_blocked = total_cells
    for i in range(total_cells):
        non_blocked[i] = i
        cell_pos[i]    = i

    node_count = 0
    n_max      = len(LHR) - 1

    for n in range(n_max):

        if n_non_blocked == 0:
            break

        # Capture pre-blocking count for LHR factor (matches md.py: NonBlockCount = len(non_blocked))
        n_non_blocked_before = n_non_blocked

        # ── Select a random non-blocked cell in O(1) ──────────────────────────
        idx      = np.random.randint(0, n_non_blocked)
        flat_idx = non_blocked[idx]
        grid_i   = flat_idx // GridSize
        grid_j   = flat_idx %  GridSize

        # ── Generate point uniformly inside the cell ──────────────────────────
        px = (grid_i + np.random.random()) * GridEdg
        py = (grid_j + np.random.random()) * GridEdg

        # ── Find existing neighbours within IntRange ──────────────────────────
        bin_x = int(px / BinEdg)
        bin_y = int(py / BinEdg)
        if bin_x >= nBins: bin_x = nBins - 1
        if bin_y >= nBins: bin_y = nBins - 1

        lo_bx = bin_x - 1 if bin_x > 0          else 0
        hi_bx = bin_x + 2 if bin_x + 2 <= nBins  else nBins
        lo_by = bin_y - 1 if bin_y > 0          else 0
        hi_by = bin_y + 2 if bin_y + 2 <= nBins  else nBins

        n_nbrs = 0
        for bx in range(lo_bx, hi_bx):
            for by in range(lo_by, hi_by):
                for k in range(BinCounts[bx, by]):
                    dx = BinnedPts[bx, by, k, 0] - px
                    dy = BinnedPts[bx, by, k, 1] - py
                    if dx * dx + dy * dy < ir2:
                        neighbor_buf[n_nbrs] = BinnedIds[bx, by, k]
                        n_nbrs += 1

        # ── Increment degrees of existing neighbours ──────────────────────────
        for k in range(n_nbrs):
            degrees[neighbor_buf[k]] += 1

        # ── Register new node ─────────────────────────────────────────────────
        new_node_id              = node_count
        degrees[new_node_id]     = n_nbrs
        node_row[new_node_id]    = grid_i
        node_col[new_node_id]    = grid_j
        node_blocked[new_node_id] = 0
        k0 = BinCounts[bin_x, bin_y]
        BinnedPts[bin_x, bin_y, k0, 0] = px
        BinnedPts[bin_x, bin_y, k0, 1] = py
        BinnedIds[bin_x, bin_y, k0]     = new_node_id
        BinCounts[bin_x, bin_y]          = k0 + 1
        node_count += 1

        # ── Update OrderMatrix; block cells where OrderMatrix > Level ──────────
        for nb_i in range(n_nb):
            ci = grid_i + Neighbors[nb_i, 0]
            cj = grid_j + Neighbors[nb_i, 1]
            if 0 <= ci < GridSize and 0 <= cj < GridSize:
                OrderMatrix[ci, cj] += 1
                if OrderMatrix[ci, cj] > Level and not BlockMatrix[ci, cj]:
                    flat          = ci * GridSize + cj
                    BlockMatrix[ci, cj] = 1
                    pos           = cell_pos[flat]
                    last          = non_blocked[n_non_blocked - 1]
                    non_blocked[pos] = last
                    cell_pos[last]   = pos
                    n_non_blocked   -= 1

        # ── Degree-Level blocking: apply for any node that just hit Level ─────
        stop_inner = False
        for k in range(n_nbrs + 1):
            nid = neighbor_buf[k] if k < n_nbrs else new_node_id
            if degrees[nid] >= Level and not node_blocked[nid]:
                node_blocked[nid] = 1
                nr = node_row[nid]
                nc = node_col[nid]
                for nb_i in range(n_nb):
                    ci = nr + Neighbors[nb_i, 0]
                    cj = nc + Neighbors[nb_i, 1]
                    if 0 <= ci < GridSize and 0 <= cj < GridSize:
                        if not BlockMatrix[ci, cj]:
                            flat          = ci * GridSize + cj
                            BlockMatrix[ci, cj] = 1
                            pos           = cell_pos[flat]
                            last          = non_blocked[n_non_blocked - 1]
                            non_blocked[pos] = last
                            cell_pos[last]   = pos
                            n_non_blocked   -= 1
            if degrees[nid] > Level:
                stop_inner = True

        if stop_inner:
            break

        LHR[n + 1] = LHR[n] * n_non_blocked_before / total_cells

    return np.dot(q, LHR)


# ── naiveMC (identical to md.py) ─────────────────────────────────────────────

def naiveMC(WindLen, Kappa, IntRange, Level, MaxIter=10**8, WarmUp=100000, Tol=0.001, Seed=None):
    if Seed is not None: np.random.seed(Seed)
    ExpPoiCount = Kappa * (WindLen ** 2)
    MeanEst  = 0.0
    Time     = 0.0
    Patience = 0
    l        = 0
    stop     = False
    print("Warming up ...... ")

    while not stop:
        tic = time.process_time()
        l  += 1
        N   = np.random.poisson(ExpPoiCount)
        md  = maxDegree(N, WindLen, IntRange)
        Y   = 0 if md > Level else 1
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


# ── conditionalMC (identical to md.py) ───────────────────────────────────────

def conditionalMC(WindLen, Kappa, IntRange, Level, MaxIter=10**8, WarmUp=1000, Tol=0.001, Seed=None):
    if Seed is not None: np.random.seed(Seed)
    ExpPoissonCount = Kappa * (WindLen ** 2)
    MeanEst    = 0.0
    MeanSqrEst = 0.0
    Time       = 0.0
    Patience   = 0
    l          = 0
    stop       = False
    print("Warming up ...... ")

    while not stop:
        tic = time.process_time()
        l  += 1
        MaxDeg, n = generatePointsUntilMaxDegree(int(3 * ExpPoissonCount), WindLen, IntRange, Level)
        Y_hat      = poisson.cdf(n, ExpPoissonCount)
        MeanEst    = ((l - 1) * MeanEst    + Y_hat)         / l
        MeanSqrEst = ((l - 1) * MeanSqrEst + Y_hat * Y_hat) / l
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

def ISMC(WindLen, GridRes, Kappa, IntRange, Level, MaxIter=10**8, WarmUp=100, Tol=0.001, Seed=None):
    """
    Numba-accelerated IS Monte Carlo for max-degree rare events.

    Interface identical to md.ISMC.  The first call triggers JIT compilation
    (~2-5 s); that one-time cost is excluded from the reported timing.
    """
    if Seed is not None:
        np.random.seed(Seed)
        _seed_numba(Seed)

    ExpPoissonCount = Kappa * (WindLen ** 2)
    GridSize        = int(int(WindLen / IntRange) * GridRes)
    GridEdg         = WindLen / GridSize
    nBins           = int(WindLen / IntRange)
    BinEdg          = WindLen / nBins
    npts            = int(3 * ExpPoissonCount)
    ir2             = IntRange * IntRange

    q         = np.array([poisson.pmf(k, ExpPoissonCount) for k in range(npts + 1)])
    Neighbors = generateNeighbors(GridEdg, IntRange)   # shape (n_nb, 2), int32
    n_nb      = len(Neighbors)
    total_cells = GridSize * GridSize

    # ── Pre-allocate all working arrays (reset inside kernel each call) ────────
    max_pts_bin  = npts + 1
    OrderMatrix  = np.zeros((GridSize, GridSize),            dtype=np.int32)
    BlockMatrix  = np.zeros((GridSize, GridSize),            dtype=np.int8)
    non_blocked  = np.arange(total_cells,                    dtype=np.int32)
    cell_pos     = np.arange(total_cells,                    dtype=np.int32)
    BinnedPts    = np.zeros((nBins, nBins, max_pts_bin, 2),  dtype=np.float64)
    BinnedIds    = np.zeros((nBins, nBins, max_pts_bin),     dtype=np.int32)
    BinCounts    = np.zeros((nBins, nBins),                  dtype=np.int32)
    degrees      = np.zeros(npts + 1,                        dtype=np.int32)
    node_row     = np.zeros(npts + 1,                        dtype=np.int32)
    node_col     = np.zeros(npts + 1,                        dtype=np.int32)
    node_blocked = np.zeros(npts + 1,                        dtype=np.int8)
    neighbor_buf = np.zeros(npts + 1,                        dtype=np.int32)
    LHR          = np.zeros(npts + 1,                        dtype=np.float64)

    # ── Trigger JIT compilation with one dry run (result discarded) ────────────
    print("Compiling Numba kernel (one-time) ...")
    _ = _md_ismc_sample(OrderMatrix, BlockMatrix,
                        non_blocked, cell_pos,
                        BinnedPts, BinnedIds, BinCounts,
                        degrees, node_row, node_col, node_blocked,
                        neighbor_buf, LHR,
                        Neighbors, n_nb,
                        GridSize, nBins, GridEdg, BinEdg,
                        ir2, Level, total_cells, q)
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

        Y_tilde = _md_ismc_sample(OrderMatrix, BlockMatrix,
                                  non_blocked, cell_pos,
                                  BinnedPts, BinnedIds, BinCounts,
                                  degrees, node_row, node_col, node_blocked,
                                  neighbor_buf, LHR,
                                  Neighbors, n_nb,
                                  GridSize, nBins, GridEdg, BinEdg,
                                  ir2, Level, total_cells, q)

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
