"""Simulation runner: wraps mesh generation + SMARTA evaluation.

Each evaluation:
  1. Validate parameters
  2. Generate meshes (4 faces + outlet) into a working directory
  3. Run the solver (GPU-accelerated or MPI fallback)
  4. Return the total flux through the output surface

Two backends are supported:
  - **GPU** (default): ``solver_gpu.py`` — single-process, CuPy/NumPy,
    no MPI or C extension required.  Works on Colab.
  - **MPI** (legacy): ``SMARTA_functions/rarfunc.py`` — requires mpi4py
    and the compiled ``rarfast`` C extension.

Set the environment variable ``INTAKE_BACKEND=mpi`` to force the legacy
backend (useful on HPC clusters).
"""

from __future__ import annotations

import os
import shutil
import sys
import traceback
from typing import Optional, Tuple

import numpy as np

from . import config
from .mesh_generator import (
    compute_effective_L,
    generate_all_meshes,
    generate_fixed_meshes,
    validate_params,
)


class SimulationRunner:
    """Manages the full simulation pipeline for one parameter evaluation."""

    def __init__(self, base_dir: str, L: Optional[float] = None,
                 verbose: bool = False) -> None:
        self.base_dir = os.path.abspath(base_dir)
        self.L = L if L is not None else compute_effective_L()
        self.verbose = verbose
        self._fixed_meshes_ready = False
        self._backend = os.environ.get("INTAKE_BACKEND", "gpu").lower()

    def setup(self) -> None:
        """One-time setup: create output dirs and generate fixed meshes."""
        os.makedirs(self.base_dir, exist_ok=True)
        mesh_dir = os.path.join(self.base_dir, config.MESH_DIR)
        os.makedirs(mesh_dir, exist_ok=True)

        # Generate inlet (fixed geometry)
        generate_fixed_meshes(mesh_dir=mesh_dir)

        # Copy honeycomb STL to working directory
        hc_src = os.path.abspath(config.HC_STL_PATH)
        hc_dst_dir = os.path.join(self.base_dir, os.path.dirname(config.HC_STL_PATH))
        os.makedirs(hc_dst_dir, exist_ok=True)
        hc_dst = os.path.join(self.base_dir, config.HC_STL_PATH)
        if not os.path.isfile(hc_dst):
            shutil.copy2(hc_src, hc_dst)

        self._fixed_meshes_ready = True

    def evaluate(self, params: list[float]) -> Tuple[float, bool]:
        """Run the full pipeline for parameter vector *params* = [a, d, e].

        Returns (objective_value, feasible).
        If the simulation fails or constraints are violated, returns
        (PENALTY_VALUE, False).
        """
        if not self._fixed_meshes_ready:
            self.setup()

        a, d, e = params[0], params[1], params[2]

        # --- Constraint check ---
        if not validate_params(a, d, e, self.L):
            return config.PENALTY_VALUE, False

        mesh_dir = os.path.join(self.base_dir, config.MESH_DIR)

        try:
            # --- Generate parametric meshes ---
            generate_all_meshes(a, d, e, self.L, mesh_dir=mesh_dir)

            # --- Run solver ---
            if self._backend == "mpi":
                flux = self._run_mpi(mesh_dir)
            else:
                flux = self._run_gpu(mesh_dir)
            return flux, True

        except Exception as exc:
            print(f"[SimulationRunner] Evaluation failed for params={params}: {exc}")
            traceback.print_exc()
            return config.PENALTY_VALUE, False

    # ─── GPU backend (default, works on Colab) ─────────────────────────

    def _run_gpu(self, mesh_dir: str) -> float:
        """Run the GPU/NumPy solver (no MPI, no C extension)."""
        from .solver_gpu import Universe, solve

        universe = Universe()

        # Load meshes
        for i in range(4):
            universe.add(os.path.join(mesh_dir, f"face{i}.stl"), 0, "wall")

        universe.add(os.path.join(mesh_dir, config.INLET_NAME), 1, "inlet")
        universe.add(os.path.join(mesh_dir, config.OUTLET_NAME), 2, "output")

        # Honeycomb
        hc_path = os.path.join(self.base_dir, config.HC_STL_PATH)
        universe.add(hc_path, 0, "wall")

        universe.init()

        # Physical properties
        universe.prop(0, T=config.WALL_TEMPERATURE)
        universe.prop(
            1,
            S=config.INLET_SPEED_RATIO,
            n=config.INLET_NUMBER_DENSITY,
            T=config.INLET_TEMPERATURE,
            m=config.INLET_MASS,
            uhat=config.INLET_UHAT,
        )

        return solve(universe, verbose=self.verbose)

    # ─── MPI backend (legacy, for HPC clusters) ────────────────────────

    def _run_mpi(self, mesh_dir: str) -> float:
        """Run the original MPI-based SMARTA solver."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        smarta_funcs = os.path.join(project_root, "SMARTA_functions")
        if smarta_funcs not in sys.path:
            sys.path.insert(0, smarta_funcs)

        from SMARTA_functions.rarfunc import (
            Comm, Universe, view_factors,
            compute_F1, compute_M, compute_E, tot_flux,
        )

        mpicomm = Comm()
        universe = Universe()

        for i in range(4):
            universe.add(mpicomm, os.path.join(mesh_dir, f"face{i}.stl"), 0, "wall")

        universe.add(mpicomm, os.path.join(mesh_dir, config.INLET_NAME), 1, "inlet")
        universe.add(mpicomm, os.path.join(mesh_dir, config.OUTLET_NAME), 2, "output")

        hc_path = os.path.join(self.base_dir, config.HC_STL_PATH)
        universe.add(mpicomm, hc_path, 0, "wall")

        universe.init(mpicomm)
        mpicomm.comm.Barrier()

        universe.prop(0, T=config.WALL_TEMPERATURE)
        universe.prop(
            1,
            S=config.INLET_SPEED_RATIO,
            n=config.INLET_NUMBER_DENSITY,
            T=config.INLET_TEMPERATURE,
            m=config.INLET_MASS,
            uhat=config.INLET_UHAT,
        )
        mpicomm.comm.Barrier()

        F = view_factors(mpicomm, universe)
        F1 = compute_F1(mpicomm, universe, F)
        M = compute_M(mpicomm, universe)
        E = compute_E(mpicomm, universe)

        fout = 0.0
        if mpicomm.rank == 0:
            b = np.dot(M * F, E)
            B = np.linalg.solve(F1, b)
            fout = tot_flux(universe, B, 0)

        mpicomm.comm.Barrier()
        return float(fout)
