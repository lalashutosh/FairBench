"""Benchmark metrics. STUB (W1d)."""
from __future__ import annotations

import numpy as np

from .constraints import ConstraintSet
from .data import Universe


def acceptance_rate(X: np.ndarray, cs: ConstraintSet, u: Universe) -> float:
    raise NotImplementedError


def tv_to_uniform(X_feasible: np.ndarray, feasible_set: np.ndarray) -> float:
    """TV distance between empirical distribution and uniform over feasible_set."""
    raise NotImplementedError


def coverage(X_feasible: np.ndarray, feasible_set: np.ndarray) -> float:
    """Fraction of feasible portfolios seen at least once."""
    raise NotImplementedError


def cost_per_feasible_sample(n_cost: int, n_feasible: int) -> float:
    """Shots or chain moves per accepted feasible sample."""
    raise NotImplementedError
