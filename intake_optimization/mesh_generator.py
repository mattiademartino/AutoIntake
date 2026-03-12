"""Parametric mesh generation for the ABEP compression channel.

Refactored from the original ``create_faces.py``.  The core maths
(``rho``, ``delta``, ``disp``, ``f`` and the 4-face rotation scheme)
are preserved exactly.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from stl import mesh as stl_mesh_lib

from . import config


# ---------------------------------------------------------------------------
# Generatrix maths
# ---------------------------------------------------------------------------

def _b_coeff(a: float, d: float, e: float, L: float) -> float:
    """Compute the linear coefficient *b* from the boundary condition rho(L) = R."""
    R = config.R
    l = config.L_SIDE
    return (R - a * L**2 - d * np.sqrt(L) - e * np.exp(-L) - l / np.sqrt(2) + e) / L


def rho(a: float, d: float, e: float, z: float, L: float) -> float:
    """Cross-section half-diagonal at axial position *z*."""
    b = _b_coeff(a, d, e, L)
    l = config.L_SIDE
    r = a * z**2 + b * z + d * np.sqrt(z) + e * np.exp(-z) + l / np.sqrt(2) - e
    return max(r, 1e-6)


def delta(z: float, L: float) -> float:
    """Corner rounding displacement at position *z*."""
    R = config.R
    return (z / L) * R * (1 - 1 / np.sqrt(2))


def disp(a: float, d: float, e: float, z: float, x: float, L: float) -> float:
    """Displacement from the flat face due to square-to-circle morphing."""
    delt = delta(z, L)
    r = rho(a, d, e, z, L)
    limit = 1e-6
    if delt / r < limit:
        return delt * (1 - 2 * (x / r) ** 2)
    raggio = (2 * delt**2 + r**2) / (4 * delt)
    radice = raggio**2 - x**2
    if radice < 0:
        return 0.0
    return np.sqrt(radice) - (raggio - delt)


def f_surface(a: float, d: float, e: float, x: float, z: float, L: float) -> float:
    """Height of the parametric surface at (x, z)."""
    return disp(a, d, e, z, x, L) + rho(a, d, e, z, L) / np.sqrt(2)


# ---------------------------------------------------------------------------
# Structured point grid
# ---------------------------------------------------------------------------

def generate_structured_points(
    a: float, d: float, e: float, L: float,
    n_z: int = config.N_Z_SECTIONS,
    n_x: int = config.N_X_POINTS,
) -> List[Dict[str, float]]:
    """Generate the structured point grid for one face of the channel."""
    z_vals = np.linspace(0, L, n_z)
    points: List[Dict[str, float]] = []

    for z_idx, z_val in enumerate(z_vals):
        rho_val = rho(a, d, e, z_val, L)
        x_range = rho_val / np.sqrt(2)
        x_vals = np.linspace(-x_range, x_range, 2 * n_x + 1)

        for x_idx, x_val in enumerate(x_vals):
            f_val = f_surface(a, d, e, x_val, z_val, L)
            points.append({
                "x": x_val, "f": f_val, "z": z_val,
                "x_idx": x_idx, "z_idx": z_idx,
            })
    return points


# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------

def rotate_points(points: List[Dict], angle_degrees: float) -> List[Dict]:
    """Rotate points around the Z axis by *angle_degrees*."""
    angle_rad = np.deg2rad(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotated: List[Dict] = []
    for p in points:
        x_new = p["x"] * cos_a - p["f"] * sin_a
        f_new = p["x"] * sin_a + p["f"] * cos_a
        rotated.append({
            "x": x_new, "f": f_new, "z": p["z"],
            "x_idx": p["x_idx"], "z_idx": p["z_idx"],
        })
    return rotated


# ---------------------------------------------------------------------------
# STL writers
# ---------------------------------------------------------------------------

def _triangle_normal(v1: Tuple, v2: Tuple, v3: Tuple) -> Tuple[float, float, float]:
    a = np.array(v1); b = np.array(v2); c = np.array(v3)
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm == 0:
        return (0.0, 0.0, 1.0)
    n = n / norm
    return (float(n[0]), float(n[1]), float(n[2]))


def write_face_stl(points: List[Dict], filename: str) -> None:
    """Write one parametric face as an ASCII STL file."""
    point_map: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    for p in points:
        point_map[(p["x_idx"], p["z_idx"])] = (p["x"], p["f"], p["z"])

    max_x_idx = max(p["x_idx"] for p in points)
    max_z_idx = max(p["z_idx"] for p in points)

    triangles: List[Tuple] = []
    for x_idx in range(max_x_idx):
        for z_idx in range(max_z_idx):
            p00 = point_map.get((x_idx, z_idx))
            p01 = point_map.get((x_idx, z_idx + 1))
            p11 = point_map.get((x_idx + 1, z_idx + 1))
            p10 = point_map.get((x_idx + 1, z_idx))
            if p00 and p01 and p11:
                triangles.append((p00, p01, p11))
            if p00 and p10 and p11:
                triangles.append((p00, p10, p11))

    with open(filename, "w") as f:
        f.write("solid generated_mesh\n")
        for tri in triangles:
            v1, v2, v3 = tri
            nx, ny, nz = _triangle_normal(v1, v2, v3)
            f.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n")
            f.write("    outer loop\n")
            for v in (v1, v2, v3):
                f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid generated_mesh\n")


def write_outlet_stl(L: float, R: float = config.R, filename: str = "outlet.stl",
                     target_edge_size: float = 5.0) -> None:
    """Create a circular cap mesh at z = L."""
    s = target_edge_size / np.sqrt(2.0)
    xs = np.arange(-R, R + s / 2, s)
    ys = np.arange(-R, R + s / 2, s)

    triangles = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[j], ys[j + 1]
            v00, v10, v01, v11 = (x0, y0, L), (x1, y0, L), (x0, y1, L), (x1, y1, L)
            tri1 = (v00, v10, v11)
            tri2 = (v00, v11, v01)

            def _centroid_inside(tri: Tuple) -> bool:
                cx = (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0
                cy = (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0
                return (cx * cx + cy * cy) <= (R + 1e-12) ** 2

            if _centroid_inside(tri1):
                triangles.append(tri1)
            if _centroid_inside(tri2):
                triangles.append(tri2)

    with open(filename, "w") as f:
        f.write("solid circular_cap\n")
        for tri in triangles:
            v1, v2, v3 = tri
            nx, ny, nz = _triangle_normal(v1, v2, v3)
            f.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n")
            f.write("    outer loop\n")
            for v in (v1, v2, v3):
                f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid circular_cap\n")


def write_inlet_stl(hc_stl_path: str = config.HC_STL_PATH,
                    z_offset: float = 0.1,
                    output_path: str = "inlet.stl",
                    n: int = 10) -> str:
    """Create a rectangular inlet mesh below the honeycomb z_min."""
    m = stl_mesh_lib.Mesh.from_file(hc_stl_path)
    x_min, x_max = float(m.x.min()), float(m.x.max())
    y_min, y_max = float(m.y.min()), float(m.y.max())
    z_min = float(m.z.min())
    z_inlet = z_min - z_offset

    total_tris = n * n * 2
    inlet_mesh = stl_mesh_lib.Mesh(np.zeros(total_tris, dtype=stl_mesh_lib.Mesh.dtype))
    xs = np.linspace(x_min, x_max, n + 1)
    ys = np.linspace(y_min, y_max, n + 1)

    tri_index = 0
    for i in range(n):
        for j in range(n):
            v0 = [xs[i], ys[j], z_inlet]
            v1 = [xs[i + 1], ys[j], z_inlet]
            v2 = [xs[i], ys[j + 1], z_inlet]
            v3 = [xs[i + 1], ys[j + 1], z_inlet]
            inlet_mesh.vectors[tri_index] = np.array([v0, v1, v2])
            tri_index += 1
            inlet_mesh.vectors[tri_index] = np.array([v1, v3, v2])
            tri_index += 1

    inlet_mesh.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------

def validate_params(a: float, d: float, e: float, L: float,
                    n_check: int = 50) -> bool:
    """Return True if the parameter set produces a valid geometry."""
    z_vals = np.linspace(0, L, n_check)
    for z in z_vals:
        r = rho(a, d, e, z, L)
        if r <= 0:
            return False
        # Check geometry fits in CubeSat cross-section
        f_max = f_surface(a, d, e, 0.0, z, L)
        if f_max > config.CUBESAT_CROSS_SECTION / 2:
            return False
    return True


# ---------------------------------------------------------------------------
# High-level: generate all mesh files for one parameter set
# ---------------------------------------------------------------------------

def generate_all_meshes(a: float, d: float, e: float, L: float,
                        mesh_dir: str = config.MESH_DIR) -> bool:
    """Generate all STL files (4 faces + outlet) for a given parameter set.

    Returns True if all files were written successfully.
    """
    os.makedirs(mesh_dir, exist_ok=True)

    # Generate structured points for one face
    points = generate_structured_points(a, d, e, L)

    # Create 4 rotated faces (0, 90, 180, 270 degrees)
    angles = [0, 90, 180, 270]
    for angle in angles:
        rotated = rotate_points(points, angle)
        face_path = os.path.join(mesh_dir, f"face{angle // 90}.stl")
        write_face_stl(rotated, face_path)

    # Outlet
    outlet_path = os.path.join(mesh_dir, config.OUTLET_NAME)
    write_outlet_stl(L, filename=outlet_path)

    return True


def generate_fixed_meshes(mesh_dir: str = config.MESH_DIR) -> None:
    """Generate inlet mesh (depends only on the honeycomb, not on parameters)."""
    os.makedirs(mesh_dir, exist_ok=True)
    inlet_path = os.path.join(mesh_dir, config.INLET_NAME)
    write_inlet_stl(output_path=inlet_path)


def compute_effective_L() -> float:
    """Compute the effective channel length L = L_intake - HC_height."""
    m = stl_mesh_lib.Mesh.from_file(config.HC_STL_PATH)
    points_z = m.points.reshape(-1, 3)[:, 2]
    hc_height = float(np.max(points_z) - np.min(points_z))
    return config.L_INTAKE - hc_height
