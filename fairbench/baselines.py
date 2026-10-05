"""Classical samplers. STUB (W1d). All return (shots, n) uint8."""
from __future__ import annotations

import numpy as np

from .constraints import ConstraintSet
from .data import Universe


def enumerate_feasible(u: Universe, cs: ConstraintSet) -> np.ndarray:
    """Exact feasible set via k-combinations (n <= ~25). Returns (m, n) uint8."""
    raise NotImplementedError


def random_k_subsets(n: int, k: int, shots: int, seed: int | None = None) -> np.ndarray:
    """Uniform random weight-k bitstrings (classical Dicke analogue)."""
    raise NotImplementedError


def rejection_sample(u: Universe, cs: ConstraintSet, shots: int, seed: int | None = None) -> np.ndarray:
    """Random k-subsets filtered by constraints."""
    raise NotImplementedError


def mcmc_swap_sample(u: Universe, cs: ConstraintSet, shots: int, burn_in: int, thin: int,
                     seed: int | None = None) -> np.ndarray:
    """Metropolis swap chain accepting only feasible moves; records number of moves."""
    raise NotImplementedError
