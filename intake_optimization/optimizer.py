#!/usr/bin/env python3
"""Bayesian Optimization loop for the ABEP intake geometry.

Usage
-----
    python -m intake_optimization.optimizer [OPTIONS]

Options
-------
    --max-evals N       Total evaluation budget (default: 60)
    --n-initial N       Random exploration points (default: 10)
    --output-dir DIR    Results directory (default: results/<timestamp>)
    --resume DIR        Resume from a previous run directory
    --timeout SECS      Per-evaluation timeout (default: 600)

The optimizer uses scikit-optimize's Gaussian-Process-based minimiser.
Since we want to *maximise* the transmission flux, we negate the objective
for skopt (which minimises).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import numpy as np

from . import config
from .config import RunConfig
from .mesh_generator import compute_effective_L
from .simulation import SimulationRunner
from .utils import Checkpoint, EvalRecord, HistoryLogger


def build_objective(runner: SimulationRunner, logger: HistoryLogger,
                    iteration_counter: list[int]):
    """Return a callable suitable for ``skopt.gp_minimize``."""

    def objective(params: List[float]) -> float:
        it = iteration_counter[0]
        iteration_counter[0] += 1

        t0 = time.time()
        flux, feasible = runner.evaluate(params)
        elapsed = time.time() - t0

        record = EvalRecord(
            iteration=it,
            params=list(params),
            objective=flux,
            feasible=feasible,
            elapsed_s=elapsed,
        )
        logger.log(record)

        best = logger.best
        best_str = f"{best.objective:.6e}" if best else "N/A"
        status = "OK" if feasible else "INFEASIBLE"
        print(
            f"  [{it:3d}] params=({params[0]:+.4f}, {params[1]:+.4f}, {params[2]:+.4f})  "
            f"flux={flux:.6e}  [{status}]  best={best_str}  ({elapsed:.1f}s)"
        )

        # skopt minimises, so negate for maximisation
        return -flux if feasible else -config.PENALTY_VALUE

    return objective


def run_optimization(cfg: RunConfig) -> None:
    """Execute the full Bayesian Optimization loop."""
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        print("ERROR: scikit-optimize is required. Install with:")
        print("  pip install scikit-optimize")
        sys.exit(1)

    # --- Setup directories ---
    os.makedirs(cfg.output_dir, exist_ok=True)

    csv_path = os.path.join(cfg.output_dir, "history.csv")
    ckpt_path = os.path.join(cfg.output_dir, "checkpoint.json")

    logger = HistoryLogger(csv_path)
    checkpoint = Checkpoint(ckpt_path)

    # --- Determine effective channel length ---
    L = compute_effective_L()
    print(f"Effective channel length L = {L:.4f} mm")

    # --- Report backend ---
    from .backend import backend_name, has_gpu
    print(f"Compute backend: {backend_name()} (GPU={'yes' if has_gpu() else 'no'})")

    # --- Simulation runner (operates inside the output dir) ---
    runner = SimulationRunner(base_dir=cfg.output_dir, L=L, verbose=cfg.verbose)
    runner.setup()

    # --- Resume support ---
    x0: Optional[List[List[float]]] = None
    y0: Optional[List[float]] = None

    if cfg.resume_from and os.path.isfile(os.path.join(cfg.resume_from, "history.csv")):
        print(f"Resuming from {cfg.resume_from}")
        prev_logger = HistoryLogger(os.path.join(cfg.resume_from, "history.csv"))
        if prev_logger.n_evaluations > 0:
            x0 = prev_logger.params_array().tolist()
            y0 = (-prev_logger.objectives_array()).tolist()  # negated for skopt
            print(f"  Loaded {len(x0)} previous evaluations")
    elif logger.n_evaluations > 0:
        # Resuming in the same directory
        x0 = logger.params_array().tolist()
        y0 = (-logger.objectives_array()).tolist()
        print(f"  Resuming with {len(x0)} existing evaluations in {cfg.output_dir}")

    # --- Determine remaining budget ---
    n_done = len(x0) if x0 else 0
    n_remaining = max(0, cfg.max_evaluations - n_done)

    if n_remaining == 0:
        print("Budget exhausted. No more evaluations to run.")
        _print_summary(logger)
        return

    n_initial = max(0, cfg.n_initial_points - n_done)

    print(f"\nStarting Bayesian Optimization")
    print(f"  Budget: {n_remaining} remaining evaluations ({n_done} already done)")
    print(f"  Initial random points: {n_initial}")
    print(f"  Parameter bounds: {cfg.param_bounds}")
    print()

    # --- Search space ---
    dimensions = [Real(lo, hi, name=name)
                  for (lo, hi), name in zip(cfg.param_bounds, config.PARAM_NAMES)]

    # --- Counter for iteration numbering ---
    counter = [n_done]

    # --- Objective function ---
    obj_func = build_objective(runner, logger, counter)

    # --- Run BO ---
    result = gp_minimize(
        func=obj_func,
        dimensions=dimensions,
        n_calls=n_remaining,
        n_initial_points=n_initial,
        x0=x0,
        y0=y0,
        acq_func="EI",         # Expected Improvement
        noise=config.GP_NOISE,
        random_state=42,
        verbose=False,
    )

    # --- Save checkpoint ---
    checkpoint.save({
        "best_params": list(result.x),
        "best_objective": -float(result.fun),
        "n_evaluations": counter[0],
        "param_bounds": cfg.param_bounds,
        "L": L,
    })

    _print_summary(logger)


def _print_summary(logger: HistoryLogger) -> None:
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    best = logger.best
    if best:
        print(f"  Best objective:  {best.objective:.6e}")
        print(f"  Best params:     a={best.params[0]:+.6f}, "
              f"d={best.params[1]:+.6f}, e={best.params[2]:+.6f}")
        print(f"  Found at iter:   {best.iteration}")
    else:
        print("  No feasible evaluation found.")
    print(f"  Total evaluations: {logger.n_evaluations}")
    print(f"  History saved to:  {logger.csv_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization for ABEP intake geometry",
    )
    parser.add_argument("--max-evals", type=int, default=config.TOTAL_BUDGET,
                        help="Total evaluation budget")
    parser.add_argument("--n-initial", type=int, default=config.N_INITIAL_POINTS,
                        help="Number of initial random exploration points")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output directory (default: results/<timestamp>)")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from a previous run directory")
    parser.add_argument("--timeout", type=int, default=config.SIMULATION_TIMEOUT,
                        help="Per-evaluation timeout in seconds")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed solver output per evaluation")
    args = parser.parse_args()

    return RunConfig(
        max_evaluations=args.max_evals,
        n_initial_points=args.n_initial,
        output_dir=args.output_dir,
        resume_from=args.resume,
        timeout=args.timeout,
        verbose=args.verbose,
    )


def main() -> None:
    cfg = parse_args()
    print(f"Output directory: {cfg.output_dir}")
    run_optimization(cfg)


if __name__ == "__main__":
    main()
