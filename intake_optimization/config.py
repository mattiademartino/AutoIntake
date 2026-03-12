"""Configuration constants, parameter bounds, and paths for the ABEP intake optimization."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Runtime environment detection
# ---------------------------------------------------------------------------
ON_COLAB: bool = "google.colab" in sys.modules

def _detect_repo_root() -> str:
    """Return the repository root directory, adapting to Colab or local."""
    if ON_COLAB:
        # Standard clone location on Colab
        candidate = "/content/AutoIntake"
        if os.path.isdir(candidate):
            return candidate
    # Fall back to parent of intake_optimization package
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT: str = _detect_repo_root()

# ---------------------------------------------------------------------------
# Geometric constants (all in mm unless noted)
# ---------------------------------------------------------------------------
R: float = 20.0          # Outlet radius [mm]
L_SIDE: float = 94.0     # Square side length at inlet [mm] (called 'l' in thesis)
L_INTAKE: float = 80.0   # Total axial length of the intake [mm]

CUBESAT_CROSS_SECTION: float = 100.0  # CubeSat face side [mm]

# ---------------------------------------------------------------------------
# Honeycomb paths (fixed geometry, do not modify)
# ---------------------------------------------------------------------------
HC_STL_PATH: str = "Honeycombs/HC1.STL"

# ---------------------------------------------------------------------------
# Mesh output
# ---------------------------------------------------------------------------
MESH_DIR: str = "mesh"
FACE_NAMES: List[str] = ["face0.stl", "face1.stl", "face2.stl", "face3.stl"]
INLET_NAME: str = "HC1_inlet.stl"
OUTLET_NAME: str = "outlet.stl"
HC_MESH_NAME: str = "HC.stl"

# ---------------------------------------------------------------------------
# SMARTA simulation settings
# ---------------------------------------------------------------------------
WALL_TEMPERATURE: float = 300.0   # Wall temperature [K]
INLET_SPEED_RATIO: float = 2.0    # S parameter for inlet
INLET_NUMBER_DENSITY: float = 1e16  # n [1/m^3]
INLET_TEMPERATURE: float = 800.0  # T_inlet [K]
INLET_MASS: float = 4e-26         # Particle mass [kg]  (approx. for atomic O)
INLET_UHAT: List[float] = [0.0, 0.0, 1.0]  # Flow direction

# ---------------------------------------------------------------------------
# Mesh generation parameters
# ---------------------------------------------------------------------------
N_Z_SECTIONS: int = 10   # Number of axial sections for the parametric surface
N_X_POINTS: int = 5      # Number of transverse points per half-section

# ---------------------------------------------------------------------------
# Optimization parameter bounds
# ---------------------------------------------------------------------------
# Parameters (a, b, c) control the generatrix curve rho(z; a, b, c).
# b is computed from the boundary condition rho(L) = R.
# Physical constraints: rho(z) > 0 for all z in [0, L], geometry must fit in CubeSat.

PARAM_BOUNDS: List[Tuple[float, float]] = [
    (-2.0, 2.0),   # a: quadratic coefficient
    (-2.0, 2.0),   # d: sqrt coefficient
    (-2.0, 2.0),   # e: exponential coefficient
]

PARAM_NAMES: List[str] = ["a", "d", "e"]
N_PARAMS: int = len(PARAM_NAMES)

# ---------------------------------------------------------------------------
# Bayesian Optimization settings
# ---------------------------------------------------------------------------
N_INITIAL_POINTS: int = 10    # Random exploration phase
N_BO_ITERATIONS: int = 50     # Bayesian optimization iterations
TOTAL_BUDGET: int = N_INITIAL_POINTS + N_BO_ITERATIONS

# Penalty value assigned when simulation fails or constraints are violated.
# Must be worse (lower) than any feasible flux so that skopt (which sees
# -flux) never prefers infeasible points.
PENALTY_VALUE: float = -1e30

# Noise estimate for the GP (SMARTA is deterministic for same mesh, but
# mesh discretisation introduces small numerical noise)
GP_NOISE: float = 1e-5

# ---------------------------------------------------------------------------
# Simulation runner settings
# ---------------------------------------------------------------------------
SIMULATION_TIMEOUT: int = 600   # Max seconds per SMARTA evaluation

# ---------------------------------------------------------------------------
# Output / logging
# ---------------------------------------------------------------------------
if ON_COLAB:
    # On Colab, save results to Google Drive for persistence across sessions
    RESULTS_BASE_DIR: str = "/content/drive/MyDrive/AutoIntake/results"
else:
    RESULTS_BASE_DIR: str = "results"


@dataclass
class RunConfig:
    """Runtime configuration assembled from CLI arguments + defaults."""
    max_evaluations: int = TOTAL_BUDGET
    n_initial_points: int = N_INITIAL_POINTS
    output_dir: str = ""
    resume_from: str = ""
    param_bounds: List[Tuple[float, float]] = field(default_factory=lambda: list(PARAM_BOUNDS))
    timeout: int = SIMULATION_TIMEOUT
    verbose: bool = False

    def __post_init__(self) -> None:
        if not self.output_dir:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(RESULTS_BASE_DIR, timestamp)
