#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba-accelerated version of ntg.py.

naiveMC and conditionalMC are identical to ntg.py.

ISMC replaces the Python/numpy inner loop with a @njit-compiled kernel.
Key conversions versus ntg.py:

  - non_blocked / cell_pos Python lists  →  pre-allocated int32 numpy arrays
    with an explicit n_non_blocked counter; swap-and-pop in native code.
  - BinnedPoints list-of-(point, id) tuples  →  BinnedPts / BinnedIds /
    BinCounts numpy arrays.
  - adj Python sets  →  adj_nbrs (2-D int32 array) + adj_count (int32 array).
  - neighbor_set Python set  →  in_neighbor (int8 array, set/cleared per step).
  - NeighborSet Python set  →  NbMat (2-D int8 array, O(1) offset lookup).
  - linalg.norm  →  squared-distance comparison.
  - Phase flag (1/2) and NTG counter tracked as native int scalars.

The first call to ISMC triggers JIT compilation (~2-5 s, excluded from timing).
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


# ── Helpers shared by naiveMC / conditionalMC (identical to ntg.py) ───────────

def numTriangles(NumPoints, WindLen, IntRange):
    if NumPoints < 3:
        return 0
    points = np.random.uniform(0, WindLen, (NumPoints, 2))
    tree = cKDTree(points)
    adj = [set() for _ in range(NumPoints)]
    for i, j in tree.query_pairs(IntRange):
        adj[i].add(j)
        adj[j].add(i)
    count = 0
    for i in range(NumPoints):
        for j in adj[i]:
            if j > i:
                count += sum(1 for k in adj[i] if k > j and k in adj[j])
    return count


def generatePointsUntilNTG(WindLen, IntRange, Level):
    nBins = max(1, int(WindLen / IntRange))
    BinEdg = WindLen / nBins
    BinnedPoints = [[[] for _ in range(nBins)] for _ in range(nBins)]
    adj = []
    NTG = 0
    n = 0

    while True:
        new_point = np.random.uniform(0, WindLen, 2)
        bin_x = min(int(new_point[0] / BinEdg), nBins - 1)
        bin_y = min(int(new_point[1] / BinEdg), nBins - 1)

        neighbor_ids = []
        for bx in range(max(0, bin_x - 1), min(bin_x + 2, nBins)):
            for by in range(max(0, bin_y - 1), min(bin_y + 2, nBins)):
                for pt, nid in BinnedPoints[bx][by]:
                    if np.linalg.norm(pt - new_point) < IntRange:
                        neighbor_ids.append(nid)

        neighbor_set = set(neighbor_ids)
        new_triangles = sum(len(neighbor_set & adj[A]) for A in neighbor_ids) // 2
        NTG += new_triangles

        new_id = len(adj)
        adj.append(set())
        n += 1
        for A in neighbor_ids:
            adj[new_id].add(A)
            adj[A].add(new_id)
        BinnedPoints[bin_x][bin_y].append((new_point, new_id))

        if NTG > Level:
            return n


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
    spread = int(IntRange / GridEdg) - 1
    if spread < 0:
        print("Error: cell diagonal length is bigger than IntRange")
    arr_size = 2 * spread + 1
    center = np.array((spread, spread))
    offsets = []
    for x in range(arr_size):
        for y in range(arr_size):
            cell = np.array((x, y))
            dist = distBtwCells(cell, center)
            if dist * GridEdg <= IntRange:
                offsets.append(cell - center)
    return np.array(offsets, dtype=np.int32)


# ── Numba kernel ──────────────────────────────────────────────────────────────

@njit
def _seed_numba(seed):
    np.random.seed(seed)


@njit
def _ntg_ismc_sample(BlockMatrix, non_blocked, cell_pos,
                     BinnedPts, BinnedIds, BinCounts,
                     adj_nbrs, adj_count,
                     in_neighbor, node_row, node_col, neighbor_buf, LHR,
                     Neighbors, NbMat,
                     n_nb, spread, GridSize, nBins, GridEdg, BinEdg,
                     ir2, Level, total_cells, q):
    """
    Run one IS sample (inner loop) entirely in native code.
    Returns Y_tilde = q · LHR for this sample.
    """
    # ── Reset working arrays ──────────────────────────────────────────────────
    BlockMatrix.fill(0)
    BinCounts.fill(0)
    adj_count.fill(0)
    LHR.fill(0.0)
    LHR[0] = 1.0

    n_non_blocked = total_cells
    for i in range(total_cells):
        non_blocked[i] = i
        cell_pos[i] = i

    NTG       = np.int64(0)
    phase     = np.int8(1)   # 1 = no blocking;  2 = blocking active
    node_count = 0
    n_max = len(LHR) - 1

    for n in range(n_max):

        if n_non_blocked == 0:
            break

        n_non_blocked_before = n_non_blocked

        # ── Select random non-blocked cell ────────────────────────────────────
        idx    = np.random.randint(0, n_non_blocked)
        flat   = non_blocked[idx]
        grid_i = flat // GridSize
        grid_j = flat %  GridSize

        # ── Generate point uniformly inside the cell ──────────────────────────
        px = (grid_i + np.random.random()) * GridEdg
        py = (grid_j + np.random.random()) * GridEdg

        # ── Find existing neighbours within IntRange ──────────────────────────
        bin_x = int(px / BinEdg)
        bin_y = int(py / BinEdg)
        if bin_x >= nBins: bin_x = nBins - 1
        if bin_y >= nBins: bin_y = nBins - 1

        lo_bx = bin_x - 1 if bin_x > 0         else 0
        hi_bx = bin_x + 2 if bin_x + 2 <= nBins else nBins
        lo_by = bin_y - 1 if bin_y > 0         else 0
        hi_by = bin_y + 2 if bin_y + 2 <= nBins else nBins

        n_nbrs = 0
        for bx in range(lo_bx, hi_bx):
            for by in range(lo_by, hi_by):
                for k in range(BinCounts[bx, by]):
                    dx = BinnedPts[bx, by, k, 0] - px
                    dy = BinnedPts[bx, by, k, 1] - py
                    if dx * dx + dy * dy < ir2:
                        neighbor_buf[n_nbrs] = BinnedIds[bx, by, k]
                        n_nbrs += 1

        # ── Set in_neighbor flags ─────────────────────────────────────────────
        for k in range(n_nbrs):
            in_neighbor[neighbor_buf[k]] = 1

        # ── Count new triangles (before registering edges) ────────────────────
        new_triangles = np.int64(0)
        for k in range(n_nbrs):
            A = neighbor_buf[k]
            for m in range(adj_count[A]):
                if in_neighbor[adj_nbrs[A, m]]:
                    new_triangles += 1
        new_triangles //= 2

        # ── Clear in_neighbor ─────────────────────────────────────────────────
        for k in range(n_nbrs):
            in_neighbor[neighbor_buf[k]] = 0

        if NTG + new_triangles > Level:
            break   # LHR[n+1] stays 0

        # ── Register node and edges ───────────────────────────────────────────
        NTG += new_triangles
        new_id = node_count
        adj_count[new_id] = n_nbrs
        for k in range(n_nbrs):
            A = neighbor_buf[k]
            adj_nbrs[new_id, k] = A
            adj_nbrs[A, adj_count[A]] = new_id
            adj_count[A] += 1
        node_row[new_id] = grid_i
        node_col[new_id] = grid_j

        k0 = BinCounts[bin_x, bin_y]
        BinnedPts[bin_x, bin_y, k0, 0] = px
        BinnedPts[bin_x, bin_y, k0, 1] = py
        BinnedIds[bin_x, bin_y, k0]    = new_id
        BinCounts[bin_x, bin_y]        = k0 + 1
        node_count += 1

        # ── Blocking update ───────────────────────────────────────────────────
        if NTG == Level:
            if phase == 1:
                # Phase 1 → 2: scan ALL current edges
                phase = np.int8(2)
                for u in range(node_count):
                    u_r = node_row[u]
                    u_c = node_col[u]
                    for m in range(adj_count[u]):
                        v = adj_nbrs[u, m]
                        if v > u:
                            v_r = node_row[v]
                            v_c = node_col[v]
                            for nb_i in range(n_nb):
                                ci = u_r + Neighbors[nb_i, 0]
                                cj = u_c + Neighbors[nb_i, 1]
                                if 0 <= ci < GridSize and 0 <= cj < GridSize:
                                    ddi = ci - v_r
                                    ddj = cj - v_c
                                    if (-spread <= ddi <= spread and
                                            -spread <= ddj <= spread and
                                            NbMat[ddi + spread, ddj + spread] and
                                            not BlockMatrix[ci, cj]):
                                        flat_c = ci * GridSize + cj
                                        BlockMatrix[ci, cj] = 1
                                        pos  = cell_pos[flat_c]
                                        last = non_blocked[n_non_blocked - 1]
                                        non_blocked[pos] = last
                                        cell_pos[last]   = pos
                                        n_non_blocked   -= 1
            else:
                # Phase 2: only new edges (new_id, A) for A in neighbor_buf
                gu_r = grid_i
                gu_c = grid_j
                for k in range(n_nbrs):
                    A    = neighbor_buf[k]
                    gv_r = node_row[A]
                    gv_c = node_col[A]
                    for nb_i in range(n_nb):
                        ci = gu_r + Neighbors[nb_i, 0]
                        cj = gu_c + Neighbors[nb_i, 1]
                        if 0 <= ci < GridSize and 0 <= cj < GridSize:
                            ddi = ci - gv_r
                            ddj = cj - gv_c
                            if (-spread <= ddi <= spread and
                                    -spread <= ddj <= spread and
                                    NbMat[ddi + spread, ddj + spread] and
                                    not BlockMatrix[ci, cj]):
                                flat_c = ci * GridSize + cj
                                BlockMatrix[ci, cj] = 1
                                pos  = cell_pos[flat_c]
                                last = non_blocked[n_non_blocked - 1]
                                non_blocked[pos] = last
                                cell_pos[last]   = pos
                                n_non_blocked   -= 1

        LHR[n + 1] = LHR[n] * n_non_blocked_before / total_cells

    return np.dot(q, LHR)


# ── naiveMC (identical to ntg.py) ─────────────────────────────────────────────

def naiveMC(WindLen, Kappa, IntRange, Level, MaxIter=10**8, WarmUp=100000, Tol=0.001, Seed=None):
    if Seed is not None: np.random.seed(Seed)
    ExpPoiCount = Kappa * (WindLen ** 2)
    MeanEst = 0.0
    Time = 0.0
    Patience = 0
    l = 0
    stop = False
    print("Warming up ...... ")

    while not stop:
        tic = time.process_time()
        l += 1
        N   = np.random.poisson(ExpPoiCount)
        ntg = numTriangles(N, WindLen, IntRange)
        Y   = 0 if ntg > Level else 1
        MeanEst = ((l - 1) * MeanEst + Y) / l
        RV = 1 / MeanEst - 1 if MeanEst != 0 else np.inf
        toc = time.process_time()
        Time += toc - tic

        if l % WarmUp == 0:
            clear_output(wait=True)
            print('\n----- Iteration:', l, '-----')
            print('\nMean estimate Z (NMC):', sci(MeanEst))
            print('Relative variance of Y (NMC):', sci(RV))
            print('Relative variance of Z (NMC):', sci(RV / l))

        if l >= WarmUp:
            Patience = Patience + 1 if RV / l < Tol else 0
        if Patience >= 100 or l >= MaxIter:
            stop = True

    return {"mean": MeanEst, "mse": MeanEst, "time": Time, "niter": l}


# ── conditionalMC (identical to ntg.py) ───────────────────────────────────────

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
        l += 1
        n     = generatePointsUntilNTG(WindLen, IntRange, Level)
        Y_hat = poisson.cdf(n - 1, ExpPoissonCount)
        MeanEst    = ((l - 1) * MeanEst    + Y_hat)           / l
        MeanSqrEst = ((l - 1) * MeanSqrEst + Y_hat * Y_hat)   / l
        RV = MeanSqrEst / (MeanEst ** 2) - 1
        toc = time.process_time()
        Time += toc - tic

        if l % WarmUp == 0:
            clear_output(wait=True)
            print('\n----- Iteration:', l, '-----')
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
    Numba-accelerated IS Monte Carlo for NTG rare events.
    Interface identical to ntg.ISMC.
    """
    if Seed is not None:
        np.random.seed(Seed)
        _seed_numba(Seed)

    ExpPoissonCount = Kappa * (WindLen ** 2)
    GridSize  = int(int(WindLen / IntRange) * GridRes)
    GridEdg   = WindLen / GridSize
    nBins     = int(WindLen / IntRange)
    BinEdg    = WindLen / nBins
    npts      = int(3 * ExpPoissonCount)
    ir2       = IntRange * IntRange
    spread    = int(IntRange / GridEdg) - 1

    q         = np.array([poisson.pmf(k, ExpPoissonCount) for k in range(npts + 1)])
    Neighbors = generateNeighbors(GridEdg, IntRange)
    n_nb      = len(Neighbors)
    total_cells = GridSize * GridSize

    # Build NbMat: 2D binary array for NeighborSet membership O(1) lookup
    NbMat_size = 2 * spread + 1
    NbMat = np.zeros((NbMat_size, NbMat_size), dtype=np.int8)
    for nb in Neighbors:
        di, dj = int(nb[0]), int(nb[1])
        NbMat[di + spread, dj + spread] = 1

    # ── Pre-allocate working arrays ────────────────────────────────────────────
    max_pts_bin = npts + 1
    max_nbrs    = npts + 1
    BlockMatrix = np.zeros((GridSize, GridSize),            dtype=np.int8)
    non_blocked = np.arange(total_cells,                    dtype=np.int32)
    cell_pos    = np.arange(total_cells,                    dtype=np.int32)
    BinnedPts   = np.zeros((nBins, nBins, max_pts_bin, 2),  dtype=np.float64)
    BinnedIds   = np.zeros((nBins, nBins, max_pts_bin),     dtype=np.int32)
    BinCounts   = np.zeros((nBins, nBins),                  dtype=np.int32)
    adj_nbrs    = np.zeros((npts + 1, max_nbrs),            dtype=np.int32)
    adj_count   = np.zeros(npts + 1,                        dtype=np.int32)
    in_neighbor = np.zeros(npts + 1,                        dtype=np.int8)
    node_row    = np.zeros(npts + 1,                        dtype=np.int32)
    node_col    = np.zeros(npts + 1,                        dtype=np.int32)
    neighbor_buf = np.zeros(npts + 1,                       dtype=np.int32)
    LHR         = np.zeros(npts + 1,                        dtype=np.float64)

    # ── Trigger JIT compilation with a dry run ─────────────────────────────────
    print("Compiling Numba kernel (one-time) ...")
    _ = _ntg_ismc_sample(BlockMatrix, non_blocked, cell_pos,
                         BinnedPts, BinnedIds, BinCounts,
                         adj_nbrs, adj_count,
                         in_neighbor, node_row, node_col, neighbor_buf, LHR,
                         Neighbors, NbMat,
                         n_nb, spread, GridSize, nBins, GridEdg, BinEdg,
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

        Y_tilde = _ntg_ismc_sample(BlockMatrix, non_blocked, cell_pos,
                                   BinnedPts, BinnedIds, BinCounts,
                                   adj_nbrs, adj_count,
                                   in_neighbor, node_row, node_col, neighbor_buf, LHR,
                                   Neighbors, NbMat,
                                   n_nb, spread, GridSize, nBins, GridEdg, BinEdg,
                                   ir2, Level, total_cells, q)

        MeanEst    = ((l - 1) * MeanEst    + Y_tilde)           / l
        MeanSqrEst = ((l - 1) * MeanSqrEst + Y_tilde * Y_tilde) / l
        RV = MeanSqrEst / (MeanEst ** 2) - 1
        toc = time.process_time()
        Time += toc - tic

        if l % WarmUp == 0:
            clear_output(wait=True)
            print('\n----- Iteration:', l, '-----')
            print('\nGrid size:', GridSize, 'x', GridSize)
            print('Mean estimate Z (IS):', sci(MeanEst))
            print('Relative variance of Y_tilde (IS):', sci(RV))
            print('Relative variance of Z (IS):', sci(RV / l))

        if l >= WarmUp:
            Patience = Patience + 1 if RV / l < Tol else 0
        if Patience >= 100 or l >= MaxIter:
            stop = True

    return {"mean": MeanEst, "mse": MeanSqrEst, "time": Time, "niter": l}
