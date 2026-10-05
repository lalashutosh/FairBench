"""Post-processing. STUB (P1)."""
from __future__ import annotations

import numpy as np

from .constraints import ConstraintSet
from .data import Universe


def filter_feasible(X: np.ndarray, cs: ConstraintSet, u: Universe) -> np.ndarray:
    raise NotImplementedError


def repair(x: np.ndarray, cs: ConstraintSet, u: Universe, seed: int | None = None) -> tuple[np.ndarray, bool]:
    """Greedy swap to feasibility; returns (x, was_repaired). Repair biases sampling, so the flag is recorded."""
    raise NotImplementedError


def assign_weights(x: np.ndarray, scheme: str = "equal") -> np.ndarray:
    raise NotImplementedError
