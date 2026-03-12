#!/usr/bin/env python3
"""Post-hoc visualisation of optimisation results.

Usage
-----
    python -m intake_optimization.visualize RESULTS_DIR

Produces:
    - Convergence plot (objective vs iteration)
    - Parameter trajectories
    - Best geometry 3D mesh plot
    - GP surrogate slices (2D, fixing other params at best values)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np

from . import config
from .utils import HistoryLogger


def plot_convergence(logger: HistoryLogger, save_dir: str) -> None:
    """Objective value vs iteration number."""
    import matplotlib.pyplot as plt

    records = logger.records
    iters = [r.iteration for r in records]
    objs = [r.objective for r in records]
    feasible = [r.feasible for r in records]

    # Running best
    best_so_far = []
    current_best = -np.inf
    for obj, feas in zip(objs, feasible):
        if feas and obj > current_best:
            current_best = obj
        best_so_far.append(current_best if current_best > -np.inf else np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Scatter: feasible vs infeasible
    feas_iters = [i for i, f in zip(iters, feasible) if f]
    feas_objs = [o for o, f in zip(objs, feasible) if f]
    infeas_iters = [i for i, f in zip(iters, feasible) if not f]
    infeas_objs = [o for o, f in zip(objs, feasible) if not f]

    ax.scatter(feas_iters, feas_objs, c="tab:blue", s=30, alpha=0.7, label="Feasible")
    if infeas_iters:
        ax.scatter(infeas_iters, infeas_objs, c="tab:red", s=30, alpha=0.5,
                   marker="x", label="Infeasible")
    ax.plot(iters, best_so_far, "k--", lw=1.5, label="Best so far")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Transmission flux")
    ax.set_title("Convergence Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "convergence.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved convergence.png")


def plot_param_trajectories(logger: HistoryLogger, save_dir: str) -> None:
    """Parameter values over iterations."""
    import matplotlib.pyplot as plt

    records = logger.records
    iters = [r.iteration for r in records]
    params = np.array([r.params for r in records])

    fig, axes = plt.subplots(1, config.N_PARAMS, figsize=(5 * config.N_PARAMS, 4))
    if config.N_PARAMS == 1:
        axes = [axes]

    for idx, (ax, name) in enumerate(zip(axes, config.PARAM_NAMES)):
        ax.plot(iters, params[:, idx], "o-", ms=3, lw=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(name)
        ax.set_title(f"Parameter '{name}' trajectory")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "param_trajectories.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved param_trajectories.png")


def plot_best_geometry(logger: HistoryLogger, save_dir: str) -> None:
    """3D wireframe of the best geometry found."""
    import matplotlib.pyplot as plt

    best = logger.best
    if best is None:
        print("  No feasible evaluation — skipping geometry plot.")
        return

    a, d, e = best.params
    from .mesh_generator import compute_effective_L, generate_structured_points, rotate_points
    L = compute_effective_L()
    points = generate_structured_points(a, d, e, L)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for angle in [0, 90, 180, 270]:
        rp = rotate_points(points, angle)
        xs = [p["x"] for p in rp]
        ys = [p["f"] for p in rp]
        zs = [p["z"] for p in rp]
        ax.scatter(xs, ys, zs, s=1, alpha=0.4)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(
        f"Best geometry  (a={a:.4f}, d={d:.4f}, e={e:.4f})\n"
        f"flux = {best.objective:.6e}"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "best_geometry.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved best_geometry.png")


def plot_gp_slices(logger: HistoryLogger, save_dir: str) -> None:
    """2D slices of the GP surrogate, fixing other params at best values."""
    try:
        from skopt import gp_minimize
        from skopt.space import Real
        from skopt.learning import GaussianProcessRegressor
        from skopt.learning.gaussian_process.kernels import Matern
    except ImportError:
        print("  scikit-optimize not available — skipping GP slice plot.")
        return
    import matplotlib.pyplot as plt

    best = logger.best
    if best is None or logger.n_evaluations < 5:
        print("  Not enough data for GP surrogate — skipping.")
        return

    X = logger.params_array()
    Y = -logger.objectives_array()  # negated (skopt convention)

    # Fit a GP
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=config.GP_NOISE, normalize_y=True)
    gp.fit(X, Y)

    fig, axes = plt.subplots(1, config.N_PARAMS, figsize=(5 * config.N_PARAMS, 4))
    if config.N_PARAMS == 1:
        axes = [axes]

    for idx, (ax, name) in enumerate(zip(axes, config.PARAM_NAMES)):
        lo, hi = config.PARAM_BOUNDS[idx]
        xs = np.linspace(lo, hi, 200)
        X_pred = np.tile(best.params, (200, 1))
        X_pred[:, idx] = xs
        mu, std = gp.predict(X_pred, return_std=True)
        mu = -mu  # back to maximisation convention

        ax.plot(xs, mu, "b-", lw=1.5)
        ax.fill_between(xs, mu - 1.96 * std, mu + 1.96 * std, alpha=0.2, color="b")
        ax.axvline(best.params[idx], color="r", ls="--", lw=1, label="best")
        ax.set_xlabel(name)
        ax.set_ylabel("Predicted flux")
        ax.set_title(f"GP slice along '{name}'")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "gp_slices.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved gp_slices.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise optimisation results")
    parser.add_argument("results_dir", help="Path to the results directory")
    args = parser.parse_args()

    results_dir = args.results_dir
    csv_path = os.path.join(results_dir, "history.csv")

    if not os.path.isfile(csv_path):
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    logger = HistoryLogger(csv_path)
    print(f"Loaded {logger.n_evaluations} evaluations from {csv_path}")

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_convergence(logger, plots_dir)
    plot_param_trajectories(logger, plots_dir)
    plot_best_geometry(logger, plots_dir)
    plot_gp_slices(logger, plots_dir)

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
