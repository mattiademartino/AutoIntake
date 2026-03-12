"""Array backend abstraction: CuPy (GPU) or NumPy (CPU).

All heavy numerical code should use ``xp`` from this module instead of
importing numpy/cupy directly.  This allows transparent GPU acceleration
on Colab (or any CUDA machine) with automatic CPU fallback.
"""

from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# Detect GPU availability
# ---------------------------------------------------------------------------
_USE_GPU: bool = False
_BACKEND_NAME: str = "numpy"

try:
    import cupy as cp          # type: ignore[import-untyped]
    # Quick smoke-test: allocate a tiny array to verify the GPU actually works
    _ = cp.zeros(1)
    _USE_GPU = True
    _BACKEND_NAME = "cupy"
except Exception:
    pass

# Environment override: INTAKE_FORCE_CPU=1 disables GPU
if os.environ.get("INTAKE_FORCE_CPU", "0") == "1":
    _USE_GPU = False
    _BACKEND_NAME = "numpy"


def has_gpu() -> bool:
    return _USE_GPU


def backend_name() -> str:
    return _BACKEND_NAME


def get_xp():
    """Return the array module (cupy or numpy)."""
    if _USE_GPU:
        import cupy as cp  # type: ignore[import-untyped]
        return cp
    return np


def to_device(arr: np.ndarray):
    """Move a numpy array to the active device (GPU copy or no-op)."""
    xp = get_xp()
    if xp is np:
        return arr
    return xp.asarray(arr)


def to_host(arr) -> np.ndarray:
    """Move an array back to CPU (cupy.asnumpy or no-op)."""
    if isinstance(arr, np.ndarray):
        return arr
    import cupy as cp  # type: ignore[import-untyped]
    return cp.asnumpy(arr)
