#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba-accelerated version of planar.py.

naiveMC and conditionalMC are identical to planar.py.

ISMC uses a hybrid approach:
  - nx.is_planar (exact stopping condition) cannot be Numba-compiled; it stays
    in Python and is called once per inner-loop step.
  - Everything else — cell selection, point generation, neighbour lookup, K5
    blocking (via K3/K4 detection) and K_{3,3} blocking (via K_{2,3} detection),
    swap-and-pop — is compiled in a @njit helper `_planar_step`.

Because nx.is_planar dominates the per-step cost (~26 ms), the Numba portion
accelerates the blocking overhead (K3 detection + distance checks) which would
otherwise add ~1–5 ms of Python overhead per step.

Key conversions vs planar.py:
  - non_blocked / cell_pos Python lists  →  pre-allocated int32 numpy arrays.
  - BinnedPoints list-of-tuples  →  BinnedPts / BinnedIds / BinCounts arrays.
  - adj Python sets  →  adj_nbrs (2-D int32) + adj_count (int32 array).
  - points list  →  points_arr (2-D float64 array).
  - K3 triple positions  →  k3_triple_buf (N×6 float64 array).
  - linalg.norm  →  squared-distance comparison.

The first call to ISMC triggers JIT compilation (~2-5 s, excluded from timing).
"""

import numpy as np
from scipy.stats import poisson
from scipy.spatial import cKDTree
import networkx as nx
import time
from IPython.display import clear_output
from numba import njit


# ── Utility ───────────────────────────────────────────────────────────────────

def sci(x, digits=2):
    return f"{x:.{digits}e}".replace("e-0", "e-").replace("e+0", "e+")


# ── Helpers shared by naiveMC / conditionalMC (identical to planar.py) ────────

def isPlanar(NumPoints, WindLen, IntRange):
    if NumPoints == 0:
        return True
    points = np.random.uniform(0, WindLen, (NumPoints, 2))
    tree = cKDTree(points)
    G = nx.Graph()
    G.add_nodes_from(range(NumPoints))
    for i, j in tree.query_pairs(IntRange):
        G.add_edge(i, j)
    return nx.is_planar(G)


def generatePointsUntilNonPlanar(WindLen, IntRange):
    G = nx.Graph()
    nBins = max(1, int(WindLen / IntRange))
    BinEdg = WindLen / nBins
    BinnedPoints = [[[] for _ in range(nBins)] for _ in range(nBins)]
    n = 0

    while True:
        p = np.random.uniform(0, WindLen, 2)
        bin_x = min(int(p[0] / BinEdg), nBins - 1)
        bin_y = min(int(p[1] / BinEdg), nBins - 1)
        new_id = n
        G.add_node(new_id)
        for bx in range(max(0, bin_x - 1), min(bin_x + 2, nBins)):
            for by in range(max(0, bin_y - 1), min(bin_y + 2, nBins)):
                for pt, nid in BinnedPoints[bx][by]:
                    if np.linalg.norm(p - pt) <= IntRange:
                        G.add_edge(new_id, nid)
        BinnedPoints[bin_x][bin_y].append((p, new_id))
        n += 1
        if not nx.is_planar(G):
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


# ── Numba helper ──────────────────────────────────────────────────────────────

@njit
def _seed_numba(seed):
    np.random.seed(seed)


@njit
def _planar_step(non_blocked, cell_pos, BlockMatrix,
                 BinnedPts, BinnedIds, BinCounts,
                 adj_nbrs, adj_count,
                 points_arr, k3_triple_buf, k23_buf,
                 in_neighbor, in_neighbor2, neighbor_buf, common_buf,
                 n_non_blocked, node_count,
                 GridSize, nBins, GridEdg, BinEdg, ir2,
                 Neighbors, n_nb):
    """
    Perform one inner step (excluding nx.is_planar):
      1. Pick a random non-blocked cell.
      2. Generate a point uniformly inside it.
      3. Find existing neighbours within IntRange.
      4. Register the new node (BinnedPts, adj, points_arr).
      5a. Detect K3 triples among existing neighbours of new_node (K5 blocking).
      5b. Detect K_{2,3} patterns involving new_node (K_{3,3} blocking):
            Case 1: new_node is u-node: ({new_node, u'}, {w1,w2,w3}) for each
                    u' in N(new_node) with common old-neighbours w1,w2,w3.
            Case 2: new_node is w-node: ({u1,u2}, {new_node,w',w''}) for each
                    pair (u1,u2) in N(new_node) with common old-neighbours w',w''.
      6a. Block grid cells that would create K5.
      6b. Block grid cells that would create K_{3,3}.

    Returns
    -------
    (px, py, grid_i, grid_j, n_nbrs, n_non_blocked_before,
     new_n_non_blocked, new_node_count, n_k3_triples, n_k23_triples)
    """
    n_non_blocked_before = n_non_blocked

    # ── Select random non-blocked cell ────────────────────────────────────────
    idx    = np.random.randint(0, n_non_blocked)
    flat   = non_blocked[idx]
    grid_i = flat // GridSize
    grid_j = flat %  GridSize

    # ── Generate point uniformly inside the cell ──────────────────────────────
    px = (grid_i + np.random.random()) * GridEdg
    py = (grid_j + np.random.random()) * GridEdg

    # ── Find existing neighbours within IntRange ──────────────────────────────
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

    # ── Register new node ─────────────────────────────────────────────────────
    new_id = node_count
    points_arr[new_id, 0] = px
    points_arr[new_id, 1] = py
    adj_count[new_id] = n_nbrs
    for k in range(n_nbrs):
        A = neighbor_buf[k]
        adj_nbrs[new_id, k] = A
        adj_nbrs[A, adj_count[A]] = new_id
        adj_count[A] += 1

    k0 = BinCounts[bin_x, bin_y]
    BinnedPts[bin_x, bin_y, k0, 0] = px
    BinnedPts[bin_x, bin_y, k0, 1] = py
    BinnedIds[bin_x, bin_y, k0]    = new_id
    BinCounts[bin_x, bin_y]        = k0 + 1
    node_count += 1

    # ── Set in_neighbor flags for N(new_node) ─────────────────────────────────
    for k in range(n_nbrs):
        in_neighbor[neighbor_buf[k]] = 1

    # ── K3 detection in neighbourhood of new_node (K5 blocking) ───────────────
    # Find K3 triangles AMONG existing neighbors of new_node (not involving
    # new_node itself).  If {A, B, C} form a K3 and a cell is within IntRange
    # of all three AND within IntRange of new_node (guaranteed by Neighbors
    # loop below), placing a point there creates K5 = {cell, new_node, A, B, C}.
    n_k3 = 0
    for ii in range(n_nbrs):
        A = neighbor_buf[ii]
        for jj in range(ii + 1, n_nbrs):
            B = neighbor_buf[jj]
            # Check A-B adjacent (scan A's OLD neighbors; new_id is last entry)
            A_adj_B = False
            for m in range(adj_count[A] - 1):
                if adj_nbrs[A, m] == B:
                    A_adj_B = True
                    break
            if not A_adj_B:
                continue
            for kk in range(jj + 1, n_nbrs):
                C = neighbor_buf[kk]
                A_adj_C = False
                for m in range(adj_count[A] - 1):
                    if adj_nbrs[A, m] == C:
                        A_adj_C = True
                        break
                if not A_adj_C:
                    continue
                B_adj_C = False
                for m in range(adj_count[B] - 1):
                    if adj_nbrs[B, m] == C:
                        B_adj_C = True
                        break
                if not B_adj_C:
                    continue
                # K3 triple: (A, B, C) all OLD neighbors of new_node, mutually adjacent
                k3_triple_buf[n_k3, 0] = points_arr[A, 0]
                k3_triple_buf[n_k3, 1] = points_arr[A, 1]
                k3_triple_buf[n_k3, 2] = points_arr[B, 0]
                k3_triple_buf[n_k3, 3] = points_arr[B, 1]
                k3_triple_buf[n_k3, 4] = points_arr[C, 0]
                k3_triple_buf[n_k3, 5] = points_arr[C, 1]
                n_k3 += 1

    # ── K_{2,3} detection for K_{3,3} blocking ────────────────────────────────
    # in_neighbor is still marking N(new_node).
    n_k23 = 0

    # Case 1: new_node is a u-node: ({new_node, u'}, {w1, w2, w3})
    # For each u' in N(new_node), find W = N(new_node) ∩ OLD_N(u').
    # Each triple {w1,w2,w3} ⊆ W gives a K_{2,3}; block cells whose centre
    # is within IntRange of all three w-nodes.
    for ii in range(n_nbrs):
        u_prime = neighbor_buf[ii]
        n_common = 0
        for m in range(adj_count[u_prime] - 1):   # OLD neighbors of u_prime
            w = adj_nbrs[u_prime, m]
            if in_neighbor[w] == 1:                # w ∈ N(new_node)
                common_buf[n_common] = w
                n_common += 1
        for a in range(n_common):
            for b in range(a + 1, n_common):
                for c in range(b + 1, n_common):
                    w1 = common_buf[a]; w2 = common_buf[b]; w3 = common_buf[c]
                    k23_buf[n_k23, 0] = points_arr[w1, 0]
                    k23_buf[n_k23, 1] = points_arr[w1, 1]
                    k23_buf[n_k23, 2] = points_arr[w2, 0]
                    k23_buf[n_k23, 3] = points_arr[w2, 1]
                    k23_buf[n_k23, 4] = points_arr[w3, 0]
                    k23_buf[n_k23, 5] = points_arr[w3, 1]
                    n_k23 += 1

    # Case 2: new_node is a w-node: ({u1, u2}, {new_node, w', w''})
    # For each pair (u1,u2) ⊆ N(new_node), find W' = OLD_N(u1) ∩ OLD_N(u2).
    # Each pair {w',w''} ⊆ W' gives a K_{2,3}; block cells whose centre is
    # within IntRange of new_node, w', w'' (scanned from new_node's grid cell).
    for ii in range(n_nbrs):
        u1 = neighbor_buf[ii]
        n_u1_old = adj_count[u1] - 1              # number of OLD neighbors of u1
        for m in range(n_u1_old):                  # mark OLD N(u1)
            in_neighbor2[adj_nbrs[u1, m]] = 1
        for jj in range(ii + 1, n_nbrs):
            u2 = neighbor_buf[jj]
            n_common = 0
            for m in range(adj_count[u2] - 1):     # OLD neighbors of u2
                w = adj_nbrs[u2, m]
                if in_neighbor2[w] == 1:            # w ∈ OLD_N(u1) ∩ OLD_N(u2)
                    common_buf[n_common] = w
                    n_common += 1
            for a in range(n_common):
                for b in range(a + 1, n_common):
                    w_p = common_buf[a]; w_pp = common_buf[b]
                    k23_buf[n_k23, 0] = px                       # new_node
                    k23_buf[n_k23, 1] = py
                    k23_buf[n_k23, 2] = points_arr[w_p,  0]
                    k23_buf[n_k23, 3] = points_arr[w_p,  1]
                    k23_buf[n_k23, 4] = points_arr[w_pp, 0]
                    k23_buf[n_k23, 5] = points_arr[w_pp, 1]
                    n_k23 += 1
        for m in range(n_u1_old):                  # clear in_neighbor2 for u1
            in_neighbor2[adj_nbrs[u1, m]] = 0

    # ── Clear in_neighbor ─────────────────────────────────────────────────────
    for k in range(n_nbrs):
        in_neighbor[neighbor_buf[k]] = 0

    # ── K5 blocking ───────────────────────────────────────────────────────────
    if n_k3 > 0:
        for nb_i in range(n_nb):
            ci = grid_i + Neighbors[nb_i, 0]
            cj = grid_j + Neighbors[nb_i, 1]
            if 0 <= ci < GridSize and 0 <= cj < GridSize and not BlockMatrix[ci, cj]:
                tcx = (ci + 0.5) * GridEdg
                tcy = (cj + 0.5) * GridEdg
                for t in range(n_k3):
                    pa0 = k3_triple_buf[t, 0]; pa1 = k3_triple_buf[t, 1]
                    pb0 = k3_triple_buf[t, 2]; pb1 = k3_triple_buf[t, 3]
                    pc0 = k3_triple_buf[t, 4]; pc1 = k3_triple_buf[t, 5]
                    da = (tcx - pa0)**2 + (tcy - pa1)**2
                    db = (tcx - pb0)**2 + (tcy - pb1)**2
                    dc = (tcx - pc0)**2 + (tcy - pc1)**2
                    if da <= ir2 and db <= ir2 and dc <= ir2:
                        flat_c = ci * GridSize + cj
                        BlockMatrix[ci, cj] = 1
                        pos  = cell_pos[flat_c]
                        last = non_blocked[n_non_blocked - 1]
                        non_blocked[pos] = last
                        cell_pos[last]   = pos
                        n_non_blocked   -= 1
                        break   # cell blocked; skip remaining triples

    # ── K_{3,3} blocking ──────────────────────────────────────────────────────
    # For Case 1 triples (w1,w2,w3): scan from w1's grid cell.
    # For Case 2 triples (new_node,w',w''): scan from new_node's grid cell.
    # In both cases the first entry in k23_buf[t] is the scan-centre point.
    if n_k23 > 0:
        for t in range(n_k23):
            sx  = k23_buf[t, 0];  sy  = k23_buf[t, 1]
            scan_gi = int(sx / GridEdg)
            scan_gj = int(sy / GridEdg)
            if scan_gi >= GridSize: scan_gi = GridSize - 1
            if scan_gj >= GridSize: scan_gj = GridSize - 1
            for nb_i in range(n_nb):
                ci = scan_gi + Neighbors[nb_i, 0]
                cj = scan_gj + Neighbors[nb_i, 1]
                if 0 <= ci < GridSize and 0 <= cj < GridSize and not BlockMatrix[ci, cj]:
                    tcx = (ci + 0.5) * GridEdg
                    tcy = (cj + 0.5) * GridEdg
                    d1 = (tcx - k23_buf[t, 0])**2 + (tcy - k23_buf[t, 1])**2
                    d2 = (tcx - k23_buf[t, 2])**2 + (tcy - k23_buf[t, 3])**2
                    d3 = (tcx - k23_buf[t, 4])**2 + (tcy - k23_buf[t, 5])**2
                    if d1 <= ir2 and d2 <= ir2 and d3 <= ir2:
                        flat_c = ci * GridSize + cj
                        BlockMatrix[ci, cj] = 1
                        pos  = cell_pos[flat_c]
                        last = non_blocked[n_non_blocked - 1]
                        non_blocked[pos] = last
                        cell_pos[last]   = pos
                        n_non_blocked   -= 1
                        break   # cell blocked; skip remaining triples

    return (px, py,
            np.int32(grid_i), np.int32(grid_j),
            np.int32(n_nbrs),
            np.int32(n_non_blocked_before),
            np.int32(n_non_blocked),
            np.int32(node_count),
            np.int32(n_k3),
            np.int32(n_k23))


@njit
def _planar_reset(non_blocked, cell_pos):
    for i in range(len(non_blocked)):
        non_blocked[i] = i
        cell_pos[i]    = i


# ── naiveMC (identical to planar.py) ──────────────────────────────────────────

def naiveMC(WindLen, Kappa, IntRange, MaxIter=10**8, WarmUp=100000, Tol=0.001, Seed=None):
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
        N = np.random.poisson(ExpPoiCount)
        Y = 1 if isPlanar(N, WindLen, IntRange) else 0
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


# ── conditionalMC (identical to planar.py) ────────────────────────────────────

def conditionalMC(WindLen, Kappa, IntRange, MaxIter=10**8, WarmUp=1000, Tol=0.001, Seed=None):
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
        n     = generatePointsUntilNonPlanar(WindLen, IntRange)
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


# ── ISMC (hybrid: Numba step + Python planarity check) ────────────────────────

def ISMC(WindLen, GridRes, Kappa, IntRange, MaxIter=10**8, WarmUp=100, Tol=0.001, Seed=None):
    """
    IS Monte Carlo for planarity: Numba-compiled blocking step + Python nx.is_planar.
    Interface identical to planar.ISMC.
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

    q         = np.array([poisson.pmf(k, ExpPoissonCount) for k in range(npts + 1)])
    Neighbors = generateNeighbors(GridEdg, IntRange)
    n_nb      = len(Neighbors)
    total_cells = GridSize * GridSize

    # ── Pre-allocate working arrays ────────────────────────────────────────────
    max_pts_bin  = npts + 1
    max_nbrs     = npts + 1
    max_k3       = max(1, (npts * (npts - 1) * (npts - 2)) // 6 + 1)
    # In practice n_k3 per step is tiny (rarely > 20); cap at a safe value
    max_k3       = min(max_k3, 10000)

    BlockMatrix   = np.zeros((GridSize, GridSize),            dtype=np.int8)
    non_blocked   = np.arange(total_cells,                    dtype=np.int32)
    cell_pos      = np.arange(total_cells,                    dtype=np.int32)
    BinnedPts     = np.zeros((nBins, nBins, max_pts_bin, 2),  dtype=np.float64)
    BinnedIds     = np.zeros((nBins, nBins, max_pts_bin),     dtype=np.int32)
    BinCounts     = np.zeros((nBins, nBins),                  dtype=np.int32)
    adj_nbrs      = np.zeros((npts + 1, max_nbrs),            dtype=np.int32)
    adj_count     = np.zeros(npts + 1,                        dtype=np.int32)
    points_arr    = np.zeros((npts + 1, 2),                   dtype=np.float64)
    k3_triple_buf = np.zeros((max_k3, 6),                     dtype=np.float64)
    k23_buf       = np.zeros((max_k3, 6),                     dtype=np.float64)
    in_neighbor   = np.zeros(npts + 1,                        dtype=np.int8)
    in_neighbor2  = np.zeros(npts + 1,                        dtype=np.int8)
    neighbor_buf  = np.zeros(npts + 1,                        dtype=np.int32)
    common_buf    = np.zeros(npts + 1,                        dtype=np.int32)
    LHR           = np.zeros(npts + 1,                        dtype=np.float64)

    # ── Trigger JIT compilation with a dry run ─────────────────────────────────
    print("Compiling Numba kernel (one-time) ...")
    _planar_reset(non_blocked, cell_pos)
    _ = _planar_step(non_blocked, cell_pos, BlockMatrix,
                     BinnedPts, BinnedIds, BinCounts,
                     adj_nbrs, adj_count,
                     points_arr, k3_triple_buf, k23_buf,
                     in_neighbor, in_neighbor2, neighbor_buf, common_buf,
                     total_cells, 0,
                     GridSize, nBins, GridEdg, BinEdg, ir2,
                     Neighbors, n_nb)
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

        # ── Reset per-iteration state ──────────────────────────────────────────
        BlockMatrix.fill(0)
        BinCounts.fill(0)
        adj_count.fill(0)
        LHR.fill(0.0)
        LHR[0] = 1.0
        _planar_reset(non_blocked, cell_pos)
        G = nx.Graph()

        n_non_blocked = total_cells
        node_count    = 0

        for n in range(npts):
            if n_non_blocked == 0:
                break

            (px, py, grid_i, grid_j, n_nbrs,
             lhr_factor, n_non_blocked, node_count, _, _) = \
                _planar_step(non_blocked, cell_pos, BlockMatrix,
                             BinnedPts, BinnedIds, BinCounts,
                             adj_nbrs, adj_count,
                             points_arr, k3_triple_buf, k23_buf,
                             in_neighbor, in_neighbor2, neighbor_buf, common_buf,
                             n_non_blocked, node_count,
                             GridSize, nBins, GridEdg, BinEdg, ir2,
                             Neighbors, n_nb)

            # ── Update nx.Graph and check planarity ────────────────────────────
            new_id = node_count - 1
            G.add_node(new_id)
            for k in range(int(n_nbrs)):
                G.add_edge(new_id, int(neighbor_buf[k]))

            if not nx.is_planar(G):
                break   # LHR[n+1] stays 0

            LHR[n + 1] = LHR[n] * int(lhr_factor) / total_cells

        Y_tilde    = q @ LHR
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
