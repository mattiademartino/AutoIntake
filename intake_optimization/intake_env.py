"""Gymnasium environment wrapping the ABEP intake simulation for RL experiments.

This is an *optional* component.  The Bayesian Optimisation loop in
``optimizer.py`` is the primary method; this environment allows
experimenting with RL approaches (PPO, SAC, etc.) via stable-baselines3.

Framing
-------
- **Observation**: current parameter vector [a, d, e] + current best flux
- **Action**: continuous perturbation delta_v (bounded)
- **Reward**: change in flux (W_new - W_old), with penalty for infeasible moves
- **Episode**: fixed number of steps (evaluation budget)

Usage
-----
    from intake_optimization.intake_env import IntakeEnv
    env = IntakeEnv(max_steps=50)
    obs, info = env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

from . import config
from .mesh_generator import compute_effective_L
from .simulation import SimulationRunner


if HAS_GYM:

    class IntakeEnv(gym.Env):
        """Gymnasium environment for RL-based intake optimisation."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            max_steps: int = 50,
            perturbation_scale: float = 0.2,
            work_dir: str = "rl_workspace",
        ) -> None:
            super().__init__()

            self.max_steps = max_steps
            self.perturbation_scale = perturbation_scale

            # Observation: [a, d, e, best_flux]
            obs_low = np.array([b[0] for b in config.PARAM_BOUNDS] + [0.0], dtype=np.float32)
            obs_high = np.array([b[1] for b in config.PARAM_BOUNDS] + [1e25], dtype=np.float32)
            self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

            # Action: perturbation for each parameter
            act_bound = np.array([perturbation_scale] * config.N_PARAMS, dtype=np.float32)
            self.action_space = spaces.Box(-act_bound, act_bound, dtype=np.float32)

            # Simulation
            self.L = compute_effective_L()
            self.runner = SimulationRunner(base_dir=work_dir, L=self.L)
            self.runner.setup()

            # State
            self._params = np.zeros(config.N_PARAMS, dtype=np.float64)
            self._best_flux = 0.0
            self._current_flux = 0.0
            self._step_count = 0

        def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
            super().reset(seed=seed)

            # Start from centre of parameter space
            self._params = np.array([
                (lo + hi) / 2 for lo, hi in config.PARAM_BOUNDS
            ], dtype=np.float64)

            flux, feasible = self.runner.evaluate(self._params.tolist())
            self._current_flux = flux if feasible else 0.0
            self._best_flux = self._current_flux
            self._step_count = 0

            return self._get_obs(), {"flux": self._current_flux, "feasible": feasible}

        def step(
            self, action: np.ndarray
        ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
            self._step_count += 1

            # Apply perturbation and clip to bounds
            new_params = self._params + action.astype(np.float64)
            for i, (lo, hi) in enumerate(config.PARAM_BOUNDS):
                new_params[i] = np.clip(new_params[i], lo, hi)

            flux, feasible = self.runner.evaluate(new_params.tolist())

            if feasible:
                reward = flux - self._current_flux
                self._params = new_params
                self._current_flux = flux
                if flux > self._best_flux:
                    self._best_flux = flux
            else:
                reward = -abs(self._current_flux) * 0.1  # penalty

            terminated = False
            truncated = self._step_count >= self.max_steps

            info = {
                "flux": self._current_flux,
                "best_flux": self._best_flux,
                "feasible": feasible,
                "params": self._params.tolist(),
                "step": self._step_count,
            }

            return self._get_obs(), float(reward), terminated, truncated, info

        def _get_obs(self) -> np.ndarray:
            return np.array(
                list(self._params) + [self._best_flux], dtype=np.float32
            )

else:
    # Stub so imports don't break when gymnasium is not installed
    class IntakeEnv:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "gymnasium is required for IntakeEnv. Install with: pip install gymnasium"
            )
