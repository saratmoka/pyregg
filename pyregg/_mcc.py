#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba-accelerated version of mcc.py.

naiveMC and conditionalMC are identical to mcc.py.

ISMC replaces the Python/numpy inner loop with a @njit-compiled kernel.
Key conversions versus mcc.py:

  - non_blocked / cell_pos Python lists  →  pre-allocated int32 numpy arrays
    with an explicit n_non_blocked counter; swap-and-pop in native code.
  - BinnedPoints list-of-(point, id) tuples  →  BinnedPts / BinnedIds /
    BinCounts numpy arrays.
  - uf_parent / uf_size / uf_members Python lists  →  pre-allocated numpy
    arrays; Union-Find path-halving and union-by-size in native code.
  - node_row / node_col  →  pre-allocated numpy arrays.
  - linalg.norm  →  squared-distance comparison (no sqrt, no function call).

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


# ── Helpers shared by naiveMC / conditionalMC (identical to mcc.py) ───────────

def maxConnectedComponent(NumPoints, WindLen, IntRange):
    if NumPoints == 0:
        return 0
    points = np.random.uniform(0, WindLen, (NumPoints, 2))
    tree = cKDTree(points)
    parent = list(range(NumPoints))
    size = [1] * NumPoints

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in tree.query_pairs(IntRange):
        ri, rj = find(i), find(j)
        if ri != rj:
            if size[ri] < size[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            size[ri] += size[rj]

    return max(size[find(i)] for i in range(NumPoints))


def _uf_find(uf_parent, x):
    while uf_parent[x] != x:
        uf_parent[x] = uf_parent[uf_parent[x]]
        x = uf_parent[x]
    return x


def generatePointsUntilMCC(WindLen, IntRange, Level):
    nBins = max(1, int(WindLen / IntRange))
    BinEdg = WindLen / nBins
    BinnedPoints = [[[] for _ in range(nBins)] for _ in range(nBins)]
    uf_parent = []
    uf_size = []
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

        new_id = len(uf_parent)
        uf_parent.append(new_id)
        uf_size.append(1)
        n += 1

        for nid in neighbor_ids:
            rx = _uf_find(uf_parent, new_id)
            ry = _uf_find(uf_parent, nid)
            if rx != ry:
                if uf_size[rx] < uf_size[ry]:
                    rx, ry = ry, rx
                uf_parent[ry] = rx
                uf_size[rx] += uf_size[ry]

        BinnedPoints[bin_x][bin_y].append((new_point, new_id))

        root = _uf_find(uf_parent, new_id)
        if uf_size[root] > Level:
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
def _mcc_ismc_sample(BlockMatrix, non_blocked, cell_pos,
                     BinnedPts, BinnedIds, BinCounts,
                     uf_parent, uf_size, uf_members, uf_mcount,
                     node_row, node_col, neighbor_buf, LHR,
                     Neighbors, n_nb,
                     GridSize, nBins, GridEdg, BinEdg,
                     ir2, Level, total_cells, max_mcount, q):
    """
    Run one IS sample (inner loop) entirely in native code.
    Returns Y_tilde = q · LHR for this sample.
    """
    # ── Reset working arrays ──────────────────────────────────────────────────
    BlockMatrix.fill(0)
    BinCounts.fill(0)
    LHR.fill(0.0)
    LHR[0] = 1.0

    n_non_blocked = total_cells
    for i in range(total_cells):
        non_blocked[i] = i
        cell_pos[i] = i

    node_count = 0
    n_max = len(LHR) - 1

    for n in range(n_max):

        if n_non_blocked == 0:
            break

        n_non_blocked_before = n_non_blocked

        # ── Select random non-blocked cell ────────────────────────────────────
        idx     = np.random.randint(0, n_non_blocked)
        flat    = non_blocked[idx]
        grid_i  = flat // GridSize
        grid_j  = flat %  GridSize

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

        # ── Register new node in Union-Find ───────────────────────────────────
        new_id = node_count
        uf_parent[new_id] = new_id
        uf_size[new_id]   = 1
        uf_mcount[new_id] = 1
        uf_members[new_id, 0] = new_id
        node_row[new_id] = grid_i
        node_col[new_id] = grid_j

        k0 = BinCounts[bin_x, bin_y]
        BinnedPts[bin_x, bin_y, k0, 0] = px
        BinnedPts[bin_x, bin_y, k0, 1] = py
        BinnedIds[bin_x, bin_y, k0]    = new_id
        BinCounts[bin_x, bin_y]        = k0 + 1
        node_count += 1

        # ── Union new node with each neighbour's component ────────────────────
        for k in range(n_nbrs):
            nid = neighbor_buf[k]

            rx = new_id
            while uf_parent[rx] != rx:
                uf_parent[rx] = uf_parent[uf_parent[rx]]
                rx = uf_parent[rx]

            ry = nid
            while uf_parent[ry] != ry:
                uf_parent[ry] = uf_parent[uf_parent[ry]]
                ry = uf_parent[ry]

            if rx != ry:
                if uf_size[rx] < uf_size[ry]:
                    rx, ry = ry, rx
                uf_parent[ry] = rx
                cnt_rx = uf_mcount[rx]
                for m in range(uf_mcount[ry]):
                    idx2 = cnt_rx + m
                    if idx2 < max_mcount:
                        uf_members[rx, idx2] = uf_members[ry, m]
                uf_mcount[rx] = cnt_rx + uf_mcount[ry]
                uf_size[rx]  += uf_size[ry]

        # ── Find root of new node (after all unions) ──────────────────────────
        root = new_id
        while uf_parent[root] != root:
            uf_parent[root] = uf_parent[uf_parent[root]]
            root = uf_parent[root]
        new_size = uf_size[root]

        if new_size > Level:
            break   # LHR[n+1] stays 0

        elif new_size == Level:
            # Block all cells within IntRange of every member node
            for m in range(uf_mcount[root]):
                mid = uf_members[root, m]
                mr  = node_row[mid]
                mc  = node_col[mid]
                for nb_i in range(n_nb):
                    ci = mr + Neighbors[nb_i, 0]
                    cj = mc + Neighbors[nb_i, 1]
                    if 0 <= ci < GridSize and 0 <= cj < GridSize:
                        if not BlockMatrix[ci, cj]:
                            flat_c = ci * GridSize + cj
                            BlockMatrix[ci, cj] = 1
                            pos  = cell_pos[flat_c]
                            last = non_blocked[n_non_blocked - 1]
                            non_blocked[pos] = last
                            cell_pos[last]   = pos
                            n_non_blocked   -= 1

        LHR[n + 1] = LHR[n] * n_non_blocked_before / total_cells

    return np.dot(q, LHR)


# ── naiveMC (identical to mcc.py) ─────────────────────────────────────────────

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
        mcc = maxConnectedComponent(N, WindLen, IntRange)
        Y   = 0 if mcc > Level else 1
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


# ── conditionalMC (identical to mcc.py) ───────────────────────────────────────

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
        n     = generatePointsUntilMCC(WindLen, IntRange, Level)
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
    Numba-accelerated IS Monte Carlo for MCC rare events.
    Interface identical to mcc.ISMC.
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
    max_mcount = Level + 10   # safe upper bound for member list

    q         = np.array([poisson.pmf(k, ExpPoissonCount) for k in range(npts + 1)])
    Neighbors = generateNeighbors(GridEdg, IntRange)
    n_nb      = len(Neighbors)
    total_cells = GridSize * GridSize

    # ── Pre-allocate working arrays ────────────────────────────────────────────
    max_pts_bin = npts + 1
    BlockMatrix = np.zeros((GridSize, GridSize),            dtype=np.int8)
    non_blocked = np.arange(total_cells,                    dtype=np.int32)
    cell_pos    = np.arange(total_cells,                    dtype=np.int32)
    BinnedPts   = np.zeros((nBins, nBins, max_pts_bin, 2),  dtype=np.float64)
    BinnedIds   = np.zeros((nBins, nBins, max_pts_bin),     dtype=np.int32)
    BinCounts   = np.zeros((nBins, nBins),                  dtype=np.int32)
    uf_parent   = np.zeros(npts + 1,                        dtype=np.int32)
    uf_size     = np.ones(npts + 1,                         dtype=np.int32)
    uf_members  = np.full((npts + 1, max_mcount), -1,       dtype=np.int32)
    uf_mcount   = np.zeros(npts + 1,                        dtype=np.int32)
    node_row    = np.zeros(npts + 1,                        dtype=np.int32)
    node_col    = np.zeros(npts + 1,                        dtype=np.int32)
    neighbor_buf = np.zeros(npts + 1,                       dtype=np.int32)
    LHR         = np.zeros(npts + 1,                        dtype=np.float64)

    # ── Trigger JIT compilation with a dry run ─────────────────────────────────
    print("Compiling Numba kernel (one-time) ...")
    _ = _mcc_ismc_sample(BlockMatrix, non_blocked, cell_pos,
                         BinnedPts, BinnedIds, BinCounts,
                         uf_parent, uf_size, uf_members, uf_mcount,
                         node_row, node_col, neighbor_buf, LHR,
                         Neighbors, n_nb,
                         GridSize, nBins, GridEdg, BinEdg,
                         ir2, Level, total_cells, max_mcount, q)
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

        Y_tilde = _mcc_ismc_sample(BlockMatrix, non_blocked, cell_pos,
                                   BinnedPts, BinnedIds, BinCounts,
                                   uf_parent, uf_size, uf_members, uf_mcount,
                                   node_row, node_col, neighbor_buf, LHR,
                                   Neighbors, n_nb,
                                   GridSize, nBins, GridEdg, BinEdg,
                                   ir2, Level, total_cells, max_mcount, q)

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
