"""GPU-accelerated SMARTA solver — replaces rarfast C extension + MPI.

This module reimplements the core computations from ``rarfast.c`` and
``rarfunc.py`` using vectorised NumPy/CuPy operations, achieving GPU
acceleration on Colab without needing MPI or a C compiler.

The key kernels ported:
  - ``compute_F``  — view factor matrix (the main bottleneck)
  - ``compute_M``  — momentum accommodation matrix
  - ``compute_E``  — emission vector
  - ``compute_F1`` — coefficient matrix for linear system
  - ``tot_flux``   — total flux through a surface group
  - ``solve``      — orchestrates the full linear solve

All operations use the ``backend.get_xp()`` array module so they run
transparently on GPU (CuPy) or CPU (NumPy).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from stl import mesh as stl_mesh_lib

from .backend import get_xp, to_device, to_host


# ═══════════════════════════════════════════════════════════════════════════
# Universe — mesh + physics container (replaces rarfunc.Universe + Comm)
# ═══════════════════════════════════════════════════════════════════════════

class Universe:
    """Holds triangle mesh data and gas properties for the solver."""

    def __init__(self) -> None:
        # Per-triangle geometry (built up by add(), finalised by init())
        self._v1: List[np.ndarray] = []
        self._v2: List[np.ndarray] = []
        self._v3: List[np.ndarray] = []
        self._normals: List[np.ndarray] = []
        self._centers: List[np.ndarray] = []
        self._areas: List[float] = []
        self._types: List[int] = []     # 0=wall/outlet, 1=inlet
        self._rhos: List[float] = []    # reflectivity
        self._IDs: List[int] = []       # surface group ID
        self.N: int = 0

        # Physics per surface-group ID (set by prop())
        self._max_ids: int = 64
        self.n_density = np.zeros(self._max_ids)
        self.T = np.zeros(self._max_ids)
        self.S = np.zeros(self._max_ids)
        self.uhat = np.zeros((self._max_ids, 3))
        self.m: float = 0.0

        # Finalised arrays (on device after init())
        self.v1_d = None
        self.v2_d = None
        self.v3_d = None
        self.normals_d = None
        self.centers_d = None
        self.areas_d = None
        self.types_arr: np.ndarray = np.array([], dtype=np.int32)
        self.rhos_arr: np.ndarray = np.array([], dtype=np.float64)
        self.IDs_arr: np.ndarray = np.array([], dtype=np.int32)

    def add(self, meshfile: str, ID: int, surftype: str, rho_val: float = 0.0) -> None:
        """Load an STL file and append its triangles."""
        surfmesh = stl_mesh_lib.Mesh.from_file(meshfile)
        n_tri = len(surfmesh.normals)

        self._v1.extend(surfmesh.v0)
        self._v2.extend(surfmesh.v1)
        self._v3.extend(surfmesh.v2)

        for i in range(n_tri):
            normal = surfmesh.normals[i]
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal = normal / norm_len
            self._normals.append(normal)
            self._centers.append((surfmesh.v0[i] + surfmesh.v1[i] + surfmesh.v2[i]) / 3.0)
            self._areas.append(
                0.5 * np.linalg.norm(np.cross(surfmesh.v1[i] - surfmesh.v0[i],
                                               surfmesh.v2[i] - surfmesh.v0[i]))
            )

            if surftype == "wall":
                self._types.append(0)
                self._rhos.append(1.0)
            elif surftype == "outlet" or surftype == "output":
                self._types.append(0)
                self._rhos.append(rho_val)
            elif surftype == "inlet":
                self._types.append(1)
                self._rhos.append(0.0)
            else:
                self._types.append(-1)
                self._rhos.append(rho_val)

            self._IDs.append(ID)

        self.N += n_tri
        print(f"  Added {n_tri} triangles from {meshfile}")

    def init(self) -> None:
        """Finalise: convert lists to device arrays."""
        self.v1_d = to_device(np.array(self._v1, dtype=np.float64))
        self.v2_d = to_device(np.array(self._v2, dtype=np.float64))
        self.v3_d = to_device(np.array(self._v3, dtype=np.float64))
        self.normals_d = to_device(np.array(self._normals, dtype=np.float64))
        self.centers_d = to_device(np.array(self._centers, dtype=np.float64))
        self.areas_d = to_device(np.array(self._areas, dtype=np.float64))
        self.types_arr = np.array(self._types, dtype=np.int32)
        self.rhos_arr = np.array(self._rhos, dtype=np.float64)
        self.IDs_arr = np.array(self._IDs, dtype=np.int32)
        print(f"  Universe initialised: {self.N} triangles")

    def prop(self, ID: int, n: float = 0.0, T: float = 0.0,
             S: float = 0.0, uhat: list = None, m: float = 0.0) -> None:
        self.n_density[ID] = n
        self.T[ID] = T
        self.S[ID] = S
        if uhat is not None:
            self.uhat[ID] = uhat
        if m > 0:
            self.m = m


# ═══════════════════════════════════════════════════════════════════════════
# View Factor Matrix — the main computational bottleneck
# ═══════════════════════════════════════════════════════════════════════════

def _compute_view_factors_batch(universe: Universe) -> np.ndarray:
    """Compute the full N x N view factor matrix.

    Strategy: vectorise over all (i,j) pairs using broadcasting.
    The far-field approximation is used when distance^2 > factor * area,
    otherwise we fall back to the edge-integration formula (also vectorised
    over the 20 integration points).

    Returns the symmetric view factor matrix F on the **host** (CPU).
    """
    xp = get_xp()
    N = universe.N
    FACTOR = 400.0
    N_INT = 20  # integration points per edge pair

    centers = universe.centers_d   # (N, 3)
    normals = universe.normals_d   # (N, 3)
    areas = universe.areas_d       # (N,)
    v1 = universe.v1_d             # (N, 3)
    v2 = universe.v2_d             # (N, 3)
    v3 = universe.v3_d             # (N, 3)

    # ── Pairwise direction vectors  sij = centers[j] - centers[i] ──
    # Shape: (N, N, 3)
    sij = centers[None, :, :] - centers[:, None, :]       # (N, N, 3)
    sij2 = xp.sum(sij ** 2, axis=2)                       # (N, N)
    sij_norm = xp.sqrt(sij2)                               # (N, N)
    sij_norm = xp.where(sij_norm < 1e-30, 1e-30, sij_norm)
    sij_hat = sij / sij_norm[:, :, None]                   # (N, N, 3)

    # ── Visibility: triangles must face each other ──
    # dot(n_i, sij) > 0 AND dot(n_j, sij) < 0
    eps = 1e-6
    doti = xp.sum(normals[:, None, :] * sij_hat, axis=2)   # (N, N)
    dotj = xp.sum(normals[None, :, :] * sij_hat, axis=2)   # (N, N)
    visible = (doti > eps) & (dotj < -eps)

    # Only compute lower triangle (i > j), then symmetrise
    lower = xp.tril(xp.ones((N, N), dtype=bool), k=-1)
    mask = visible & lower                                 # (N, N)

    # ── Far-field approximation ──
    far_i = sij2 > FACTOR * areas[:, None]
    far_j = sij2 > FACTOR * areas[None, :]
    far = far_i & far_j & mask

    F = xp.zeros((N, N), dtype=xp.float64)

    # Far-field: F_ij = -dot(n_i, sij_hat) * dot(n_j, sij_hat) / (pi * sij2)
    if xp.any(far):
        F_far = -(doti * dotj) / (xp.pi * sij2)
        F = xp.where(far, F_far, F)

    # ── Near-field: edge-edge line integration ──
    near = mask & (~far)
    near_indices = xp.nonzero(near)  # tuple of (row_indices, col_indices)

    if len(near_indices[0]) > 0:
        ni_host = to_host(near_indices[0])
        nj_host = to_host(near_indices[1])
        n_pairs = len(ni_host)

        # Prepare edge vertices for the near-field pairs
        # Each triangle has 3 edges: (v1,v2), (v2,v3), (v3,v1)
        v1_h = to_host(v1)
        v2_h = to_host(v2)
        v3_h = to_host(v3)
        areas_h = to_host(areas)

        # Process in batches to manage memory
        BATCH = 50000
        F_near_vals = np.zeros(n_pairs, dtype=np.float64)

        for b_start in range(0, n_pairs, BATCH):
            b_end = min(b_start + BATCH, n_pairs)
            bi = ni_host[b_start:b_end]
            bj = nj_host[b_start:b_end]
            nb = b_end - b_start

            # Triangle i edges
            edges_i = np.stack([
                np.stack([v1_h[bi], v2_h[bi]], axis=1),  # edge 0: v1→v2
                np.stack([v2_h[bi], v3_h[bi]], axis=1),  # edge 1: v2→v3
                np.stack([v3_h[bi], v1_h[bi]], axis=1),  # edge 2: v3→v1
            ], axis=1)  # (nb, 3_edges, 2_endpoints, 3_xyz)

            # Triangle j edges
            edges_j = np.stack([
                np.stack([v1_h[bj], v2_h[bj]], axis=1),
                np.stack([v2_h[bj], v3_h[bj]], axis=1),
                np.stack([v3_h[bj], v1_h[bj]], axis=1),
            ], axis=1)  # (nb, 3, 2, 3)

            pair_sum = np.zeros(nb, dtype=np.float64)

            for k in range(3):  # edges of triangle i
                K1 = edges_i[:, k, 0, :]  # (nb, 3)
                K2 = edges_i[:, k, 1, :]  # (nb, 3)
                K_vec = K2 - K1
                lenk = np.linalg.norm(K_vec, axis=1)  # (nb,)
                lenk = np.where(lenk < 1e-30, 1e-30, lenk)

                for l in range(3):  # edges of triangle j
                    L1 = edges_j[:, l, 0, :]
                    L2 = edges_j[:, l, 1, :]
                    L_vec = L2 - L1
                    lenl = np.linalg.norm(L_vec, axis=1)
                    lenl = np.where(lenl < 1e-30, 1e-30, lenl)

                    Dkl = np.sum(K_vec * L_vec, axis=1) / (lenk * lenl)
                    dk = lenk / N_INT

                    integral = np.zeros(nb, dtype=np.float64)

                    for nk in range(N_INT):
                        t = (nk + 0.5) / N_INT
                        K_pt = K1 + t * K_vec  # (nb, 3)

                        KL1 = L1 - K_pt
                        KL2 = L2 - K_pt
                        Lalpha = np.linalg.norm(KL1, axis=1)
                        Lbeta = np.linalg.norm(KL2, axis=1)
                        Lalpha = np.where(Lalpha < 1e-30, 1e-30, Lalpha)
                        Lbeta = np.where(Lbeta < 1e-30, 1e-30, Lbeta)

                        cos_gamma_arg = (Lalpha**2 + Lbeta**2 - lenl**2) / (2 * Lalpha * Lbeta)
                        cos_gamma_arg = np.clip(cos_gamma_arg, -1.0, 1.0)
                        gamma = np.arccos(cos_gamma_arg)

                        cos_alpha = -np.sum(L_vec * KL1, axis=1) / (lenl * Lalpha)
                        cos_beta = np.sum(L_vec * KL2, axis=1) / (lenl * Lbeta)

                        pcross = np.cross(KL1, KL2)
                        P = np.linalg.norm(pcross, axis=1) / lenl

                        log_Lalpha = np.log(np.where(Lalpha < 1e-30, 1e-30, Lalpha))
                        log_Lbeta = np.log(np.where(Lbeta < 1e-30, 1e-30, Lbeta))

                        dint = (Lalpha * log_Lalpha * cos_alpha
                                + Lbeta * log_Lbeta * cos_beta
                                + P * gamma - lenl) * dk

                        integral += dint

                    integral *= Dkl
                    pair_sum += integral

            pair_sum = np.maximum(pair_sum, 0.0)
            Ai = areas_h[bi]
            Aj = areas_h[bj]
            F_near_vals[b_start:b_end] = pair_sum / (np.pi * 2.0 * Ai * Aj)

        # Write near-field values into F
        F_near_dev = to_device(F_near_vals)
        F[near_indices[0], near_indices[1]] = F_near_dev

    # Symmetrise: F[j,i] = F[i,j]
    F = F + F.T

    # Apply area weighting: F = (F + F.T) * areas  (already symmetric)
    # In original code: F = (F + F.T) * universe.areas
    # But we already have F + F.T in the symmetrisation step above.
    # The original does: F = F[offsets] then F = (F + F.T) * areas
    # Since we compute the full matrix without MPI reordering, we just
    # need the area weighting.
    F = F * areas[None, :]

    return to_host(F)


# ═══════════════════════════════════════════════════════════════════════════
# M, F1, E matrices + linear solve
# ═══════════════════════════════════════════════════════════════════════════

def _compute_M(universe: Universe) -> np.ndarray:
    """Compute the M matrix (momentum accommodation).

    M[i,j] = Mij(S, cos_alpha)  when type[j]==1 (inlet), else 0.
    The original code iterates over inlet triangles (type==1) in the *row*
    dimension, then transposes.  We replicate that logic.
    """
    N = universe.N
    centers = to_host(universe.centers_d)
    types = universe.types_arr
    IDs = universe.IDs_arr

    M = np.zeros((N, N), dtype=np.float64)

    inlet_mask = types == 1
    inlet_indices = np.where(inlet_mask)[0]

    if len(inlet_indices) == 0:
        return M

    for i in inlet_indices:
        ID = IDs[i]
        S = universe.S[ID]
        uhat_i = universe.uhat[ID]

        # sij for all j
        sij = centers - centers[i]  # (N, 3)
        sij_norm = np.linalg.norm(sij, axis=1)
        sij_norm[sij_norm < 1e-30] = 1e-30
        sij_hat = sij / sij_norm[:, None]

        cos_alpha = sij_hat @ uhat_i  # (N,)
        sig = S * cos_alpha
        phi = (np.sqrt(np.pi) * sig
               * np.exp(S**2 * (cos_alpha**2 - 1.0))
               * (sig**2 + 1.5) * (1.0 + _erf_vec(sig))
               + np.exp(-S**2) * (sig**2 + 1.0))
        phi[i] = 0.0
        M[i, :] = phi

    M = M.T
    return M


def _erf_vec(x):
    """Vectorised error function (scipy or numpy fallback)."""
    try:
        from scipy.special import erf
        return erf(x)
    except ImportError:
        # Rough approximation for fallback
        from numpy import tanh
        return tanh(1.20278 * x + 0.04 * x**3)


def _compute_F1(universe: Universe, F: np.ndarray) -> np.ndarray:
    """F1 = I - F * rhos."""
    rhos = universe.rhos_arr
    return -F * rhos[None, :] + np.eye(universe.N)


def _compute_E(universe: Universe) -> np.ndarray:
    """Emission vector E[i] = flux_thermal for inlet surfaces."""
    k_B = 1.38064852e-23
    N = universe.N
    E = np.zeros(N, dtype=np.float64)
    types = universe.types_arr
    IDs = universe.IDs_arr

    for i in range(N):
        if types[i] == 1:
            ID = IDs[i]
            n = universe.n_density[ID]
            T = universe.T[ID]
            m = universe.m
            E[i] = n * np.sqrt(8 * k_B * T / (np.pi * m)) / 4.0

    return E


def _tot_flux(universe: Universe, B: np.ndarray, ID: int) -> float:
    """Total flux through surface group *ID*."""
    mask = universe.IDs_arr == ID
    areas = to_host(universe.areas_d)
    return float(np.sum(B[mask] * areas[mask]))


# ═══════════════════════════════════════════════════════════════════════════
# Top-level solve
# ═══════════════════════════════════════════════════════════════════════════

def solve(universe: Universe, verbose: bool = False) -> float:
    """Run the full SMARTA solve and return the output flux.

    This replaces the entire SMARTA.py + rarfunc.py pipeline with a
    single-process (optionally GPU-accelerated) implementation.
    """
    xp = get_xp()

    if verbose:
        print("Computing view factor matrix...")
    F = _compute_view_factors_batch(universe)

    if verbose:
        print(f"  F range: [{F.min():.6e}, {F.max():.6e}]")
        print("Computing F1 matrix...")
    F1 = _compute_F1(universe, F)

    if verbose:
        print("Computing M matrix...")
    M = _compute_M(universe)

    if verbose:
        print("Computing E vector...")
    E = _compute_E(universe)

    if verbose:
        print("Solving linear system...")

    # b = dot(M * F, E)
    F2 = F
    b = (M * F2) @ E

    # GPU-accelerated solve if available
    F1_d = to_device(F1)
    b_d = to_device(b)
    B_d = xp.linalg.solve(F1_d, b_d)
    B = to_host(B_d)

    if verbose:
        bcheck = F1 @ B
        print(f"  Solution check (allclose): {np.allclose(bcheck, b)}")

    fout = _tot_flux(universe, B, 0)
    if verbose:
        print(f"  Total flux through wall group (ID=0): {fout:.6e}")

    return fout
