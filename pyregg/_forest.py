#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forest (Acyclicity) — Numba-accelerated rare-event simulation.

Rare event : G(X) is a forest (contains no cycle).

Algorithm notes
---------------
NMC  : Generate PPP, build graph with union-find, return 1 if acyclic.
CMC  : Add points sequentially; stop at the first cycle (step n);
       contribute poisson.cdf(n-1, mu).
ISMC : Sequential point addition with grid-based blocking.
       A cell C is blocked when its grid-cell neighbourhood (the Neighbors
       offsets) contains two existing nodes in the same connected component.
       Stopping: exact — cycle detected via union-find on actual distances.
       LHR correction ensures the estimator is unbiased.
"""

import numpy as np
from scipy.stats import poisson
from scipy.spatial import cKDTree
import time
from IPython.display import clear_output
from numba import njit


# ── Utility ────────────────────────────────────────────────────────────────────

def sci(x, digits=2):
    return f"{x:.{digits}e}".replace("e-0", "e-").replace("e+0", "e+")


# ── NMC helper ─────────────────────────────────────────────────────────────────

def isForest(NumPoints, WindLen, IntRange):
    """Return True if a random Gilbert graph on NumPoints points is a forest."""
    if NumPoints == 0:
        return True
    points = np.random.uniform(0, WindLen, (NumPoints, 2))
    tree   = cKDTree(points)
    parent = list(range(NumPoints))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in tree.query_pairs(IntRange):
        ri, rj = find(i), find(j)
        if ri == rj:
            return False          # cycle detected
        parent[rj] = ri
    return True


# ── CMC helper ─────────────────────────────────────────────────────────────────

def _uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def generatePointsUntilCycle(WindLen, IntRange):
    """Add uniform random points until a cycle is created; return step count n."""
    nBins        = max(1, int(WindLen / IntRange))
    BinEdg       = WindLen / nBins
    BinnedPoints = [[[] for _ in range(nBins)] for _ in range(nBins)]
    parent       = []
    n            = 0

    while True:
        p  = np.random.uniform(0, WindLen, 2)
        bx = min(int(p[0] / BinEdg), nBins - 1)
        by = min(int(p[1] / BinEdg), nBins - 1)

        nbrs = []
        for dbx in range(-1, 2):
            for dby in range(-1, 2):
                nbx, nby = bx + dbx, by + dby
                if 0 <= nbx < nBins and 0 <= nby < nBins:
                    for pt, nid in BinnedPoints[nbx][nby]:
                        if np.linalg.norm(pt - p) <= IntRange:
                            nbrs.append(nid)

        new_id = len(parent)
        parent.append(new_id)
        n += 1

        for nid in nbrs:
            rx = _uf_find(parent, new_id)
            ry = _uf_find(parent, nid)
            if rx == ry:
                return n          # cycle at step n
            parent[ry] = rx

        BinnedPoints[bx][by].append((p, new_id))


# ── Neighbour offsets (conservative: all corners within IntRange) ───────────────

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
        print("Error: cell diagonal is larger than IntRange")
    arr_size = 2 * spread + 1
    center   = np.array((spread, spread))
    offsets  = []
    for x in range(arr_size):
        for y in range(arr_size):
            cell = np.array((x, y))
            if distBtwCells(cell, center) * GridEdg <= IntRange:
                offsets.append(cell - center)
    return np.array(offsets, dtype=np.int32)


# ── Numba kernel ───────────────────────────────────────────────────────────────

@njit
def _seed_numba(seed):
    np.random.seed(seed)


@njit
def _forest_ismc_sample(BlockMatrix, non_blocked, cell_pos,
                        BinnedPts, BinnedIds, BinCounts,
                        NodeGrid, NodeGridCount,
                        uf_parent, uf_rank,
                        node_row, node_col,
                        neighbor_buf, root_buf, LHR,
                        Neighbors, n_nb,
                        GridSize, nBins, GridEdg, BinEdg,
                        ir2, total_cells, q):
    """
    One IS sample for the forest rare event.

    Blocking rule  : cell C is blocked when its Neighbors-neighbourhood
                     contains two existing nodes sharing a component root.
    Stopping rule  : exact — cycle detected via union-find on actual point
                     positions.
    Returns Y_tilde = dot(q, LHR).
    """
    # ── Reset working arrays ───────────────────────────────────────────────────
    BlockMatrix.fill(0)
    BinCounts.fill(0)
    NodeGridCount.fill(0)
    LHR.fill(0.0)
    LHR[0] = 1.0

    n_non_blocked = total_cells
    for i in range(total_cells):
        non_blocked[i] = i
        cell_pos[i]    = i

    node_count = 0
    n_max      = len(LHR) - 1

    for n in range(n_max):

        if n_non_blocked == 0:
            break

        n_non_blocked_before = n_non_blocked

        # ── Select random non-blocked cell ─────────────────────────────────────
        idx    = np.random.randint(0, n_non_blocked)
        flat   = non_blocked[idx]
        grid_i = flat // GridSize
        grid_j = flat %  GridSize

        # ── Generate point uniformly inside the cell ───────────────────────────
        px = (grid_i + np.random.random()) * GridEdg
        py = (grid_j + np.random.random()) * GridEdg

        # ── Find existing neighbours within IntRange ───────────────────────────
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

        # ── Register new node ──────────────────────────────────────────────────
        new_id            = node_count
        uf_parent[new_id] = new_id
        uf_rank[new_id]   = np.int32(0)
        node_row[new_id]  = grid_i
        node_col[new_id]  = grid_j

        k0 = BinCounts[bin_x, bin_y]
        BinnedPts[bin_x, bin_y, k0, 0] = px
        BinnedPts[bin_x, bin_y, k0, 1] = py
        BinnedIds[bin_x, bin_y, k0]    = new_id
        BinCounts[bin_x, bin_y]        = k0 + 1

        k1 = NodeGridCount[grid_i, grid_j]
        NodeGrid[grid_i, grid_j, k1]   = new_id
        NodeGridCount[grid_i, grid_j]  = k1 + 1

        node_count += 1

        # ── Union new node with neighbours; detect cycle ───────────────────────
        cycle = False
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

            if rx == ry:
                cycle = True
                break

            if uf_rank[rx] < uf_rank[ry]:
                rx, ry = ry, rx
            uf_parent[ry] = rx
            if uf_rank[rx] == uf_rank[ry]:
                uf_rank[rx] += 1

        if cycle:
            break                 # LHR[n+1] stays 0

        # ── Update blocking ────────────────────────────────────────────────────
        # Check each cell in the Neighbors of the new node's grid cell.
        # A cell (ci, cj) should be blocked if its own Neighbors-neighbourhood
        # contains two nodes that now share a component root.
        for nb_i in range(n_nb):
            ci = grid_i + Neighbors[nb_i, 0]
            cj = grid_j + Neighbors[nb_i, 1]
            if ci < 0 or ci >= GridSize or cj < 0 or cj >= GridSize:
                continue
            if BlockMatrix[ci, cj]:
                continue

            # Scan (ci,cj)'s Neighbors-neighbourhood for shared-root pairs
            n_seen      = np.int32(0)
            should_blk  = False

            for nb_j in range(n_nb):
                ni = ci + Neighbors[nb_j, 0]
                nj = cj + Neighbors[nb_j, 1]
                if ni < 0 or ni >= GridSize or nj < 0 or nj >= GridSize:
                    continue
                for m in range(NodeGridCount[ni, nj]):
                    node_id = NodeGrid[ni, nj, m]
                    # Path-halving find
                    root = node_id
                    while uf_parent[root] != root:
                        uf_parent[root] = uf_parent[uf_parent[root]]
                        root = uf_parent[root]
                    for r in range(n_seen):
                        if root_buf[r] == root:
                            should_blk = True
                            break
                    if should_blk:
                        break
                    root_buf[n_seen] = root
                    n_seen += 1
                if should_blk:
                    break

            if should_blk:
                flat_c             = ci * GridSize + cj
                BlockMatrix[ci, cj] = np.int8(1)
                pos                = cell_pos[flat_c]
                last               = non_blocked[n_non_blocked - 1]
                non_blocked[pos]   = last
                cell_pos[last]     = pos
                n_non_blocked     -= 1

        LHR[n + 1] = LHR[n] * n_non_blocked_before / total_cells

    return np.dot(q, LHR)


# ── naiveMC ────────────────────────────────────────────────────────────────────

def naiveMC(WindLen, Kappa, IntRange, MaxIter=10**8, WarmUp=100000, Tol=0.001, Seed=None):
    if Seed is not None:
        np.random.seed(Seed)
    ExpPoiCount = Kappa * (WindLen ** 2)
    MeanEst     = 0.0
    Time        = 0.0
    Patience    = 0
    l           = 0
    stop        = False
    print("Warming up ...... ")

    while not stop:
        tic  = time.process_time()
        l   += 1
        N    = np.random.poisson(ExpPoiCount)
        Y    = 1 if isForest(N, WindLen, IntRange) else 0
        MeanEst = ((l - 1) * MeanEst + Y) / l
        RV   = 1 / MeanEst - 1 if MeanEst > 0 else np.inf
        toc  = time.process_time()
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


# ── conditionalMC ──────────────────────────────────────────────────────────────

def conditionalMC(WindLen, Kappa, IntRange, MaxIter=10**8, WarmUp=1000, Tol=0.001, Seed=None):
    if Seed is not None:
        np.random.seed(Seed)
    ExpPoissonCount = Kappa * (WindLen ** 2)
    MeanEst         = 0.0
    MeanSqrEst      = 0.0
    Time            = 0.0
    Patience        = 0
    l               = 0
    stop            = False
    print("Warming up ...... ")

    while not stop:
        tic    = time.process_time()
        l     += 1
        n      = generatePointsUntilCycle(WindLen, IntRange)
        Y_hat  = poisson.cdf(n - 1, ExpPoissonCount)
        MeanEst    = ((l - 1) * MeanEst    + Y_hat)          / l
        MeanSqrEst = ((l - 1) * MeanSqrEst + Y_hat * Y_hat)  / l
        RV     = MeanSqrEst / (MeanEst ** 2) - 1
        toc    = time.process_time()
        Time  += toc - tic

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


# ── ISMC (Numba-accelerated) ───────────────────────────────────────────────────

def ISMC(WindLen, GridRes, Kappa, IntRange, MaxIter=10**8, WarmUp=100, Tol=0.001, Seed=None):
    """
    Numba-accelerated IS Monte Carlo for the forest (acyclicity) rare event.

    Blocking : cell C is blocked when its conservative Neighbors-neighbourhood
               contains two existing nodes in the same connected component.
    Stopping : exact cycle detection via union-find on actual point positions.
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
    total_cells     = GridSize * GridSize

    q         = np.array([poisson.pmf(k, ExpPoissonCount) for k in range(npts + 1)])
    Neighbors = generateNeighbors(GridEdg, IntRange)
    n_nb      = len(Neighbors)

    # ── Pre-allocate working arrays ────────────────────────────────────────────
    max_pts_bin  = npts + 1
    max_pts_cell = max(1, int(10 * ExpPoissonCount / total_cells) + 5)

    BlockMatrix   = np.zeros((GridSize, GridSize),            dtype=np.int8)
    non_blocked   = np.arange(total_cells,                    dtype=np.int32)
    cell_pos      = np.arange(total_cells,                    dtype=np.int32)
    BinnedPts     = np.zeros((nBins, nBins, max_pts_bin, 2),  dtype=np.float64)
    BinnedIds     = np.zeros((nBins, nBins, max_pts_bin),     dtype=np.int32)
    BinCounts     = np.zeros((nBins, nBins),                  dtype=np.int32)
    NodeGrid      = np.zeros((GridSize, GridSize, max_pts_cell), dtype=np.int32)
    NodeGridCount = np.zeros((GridSize, GridSize),            dtype=np.int32)
    uf_parent     = np.zeros(npts + 1,                        dtype=np.int32)
    uf_rank       = np.zeros(npts + 1,                        dtype=np.int32)
    node_row      = np.zeros(npts + 1,                        dtype=np.int32)
    node_col      = np.zeros(npts + 1,                        dtype=np.int32)
    neighbor_buf  = np.zeros(npts + 1,                        dtype=np.int32)
    root_buf      = np.zeros(npts + 1,                        dtype=np.int32)
    LHR           = np.zeros(npts + 1,                        dtype=np.float64)

    # ── Trigger JIT compilation with a dry run ─────────────────────────────────
    print("Compiling Numba kernel (one-time) ...")
    _ = _forest_ismc_sample(BlockMatrix, non_blocked, cell_pos,
                            BinnedPts, BinnedIds, BinCounts,
                            NodeGrid, NodeGridCount,
                            uf_parent, uf_rank,
                            node_row, node_col,
                            neighbor_buf, root_buf, LHR,
                            Neighbors, n_nb,
                            GridSize, nBins, GridEdg, BinEdg,
                            ir2, total_cells, q)
    print("Warming up ...... ")

    MeanEst    = 0.0
    MeanSqrEst = 0.0
    Time       = 0.0
    Patience   = 0
    l          = 0
    stop       = False

    while not stop:
        tic  = time.process_time()
        l   += 1

        Y_tilde = _forest_ismc_sample(BlockMatrix, non_blocked, cell_pos,
                                      BinnedPts, BinnedIds, BinCounts,
                                      NodeGrid, NodeGridCount,
                                      uf_parent, uf_rank,
                                      node_row, node_col,
                                      neighbor_buf, root_buf, LHR,
                                      Neighbors, n_nb,
                                      GridSize, nBins, GridEdg, BinEdg,
                                      ir2, total_cells, q)

        MeanEst    = ((l - 1) * MeanEst    + Y_tilde)           / l
        MeanSqrEst = ((l - 1) * MeanSqrEst + Y_tilde * Y_tilde) / l
        RV         = MeanSqrEst / (MeanEst ** 2) - 1 if MeanEst > 0 else np.inf
        toc  = time.process_time()
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
