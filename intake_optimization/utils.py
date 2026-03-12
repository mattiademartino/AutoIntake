"""Utility helpers: logging, CSV I/O, checkpointing."""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# History entry
# ---------------------------------------------------------------------------
@dataclass
class EvalRecord:
    """One evaluation of the objective function."""
    iteration: int
    params: List[float]
    objective: float
    feasible: bool
    elapsed_s: float
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# CSV history logger
# ---------------------------------------------------------------------------
class HistoryLogger:
    """Append-only CSV logger for optimisation history."""

    HEADER = ["iteration", "a", "d", "e", "objective", "feasible", "elapsed_s", "timestamp"]

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self._records: List[EvalRecord] = []
        if os.path.isfile(csv_path):
            self._load_existing()
        else:
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)

    # -- public API ----------------------------------------------------------

    def log(self, record: EvalRecord) -> None:
        self._records.append(record)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                record.iteration,
                *record.params,
                record.objective,
                int(record.feasible),
                f"{record.elapsed_s:.2f}",
                record.timestamp,
            ])
            f.flush()
            os.fsync(f.fileno())

    @property
    def records(self) -> List[EvalRecord]:
        return list(self._records)

    @property
    def best(self) -> Optional[EvalRecord]:
        feasible = [r for r in self._records if r.feasible]
        if not feasible:
            return None
        return max(feasible, key=lambda r: r.objective)

    @property
    def n_evaluations(self) -> int:
        return len(self._records)

    def params_array(self) -> np.ndarray:
        return np.array([r.params for r in self._records])

    def objectives_array(self) -> np.ndarray:
        return np.array([r.objective for r in self._records])

    # -- internals -----------------------------------------------------------

    def _load_existing(self) -> None:
        with open(self.csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 8:
                    continue
                rec = EvalRecord(
                    iteration=int(row[0]),
                    params=[float(row[1]), float(row[2]), float(row[3])],
                    objective=float(row[4]),
                    feasible=bool(int(row[5])),
                    elapsed_s=float(row[6]),
                    timestamp=row[7],
                )
                self._records.append(rec)


# ---------------------------------------------------------------------------
# JSON checkpoint for optimizer state
# ---------------------------------------------------------------------------
class Checkpoint:
    """Save / restore optimizer state to JSON."""

    def __init__(self, path: str) -> None:
        self.path = path

    def save(self, state: Dict[str, Any]) -> None:
        serialisable = _make_serialisable(state)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(serialisable, f, indent=2)
        os.replace(tmp, self.path)

    def load(self) -> Dict[str, Any]:
        with open(self.path, "r") as f:
            return json.load(f)

    def exists(self) -> bool:
        return os.path.isfile(self.path)


def _make_serialisable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    return obj
