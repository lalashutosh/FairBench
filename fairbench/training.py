"""Variational training. STUB (P1)."""
from __future__ import annotations

from typing import Callable

import numpy as np
from qiskit import QuantumCircuit


def train(ansatz: QuantumCircuit, objective_fn: Callable[[np.ndarray], float], backend: str,
          optimizer: str = "COBYLA", maxiter: int = 100, seed: int | None = None):
    """Optimise ansatz parameters; objective_fn maps sampled bitstrings (shots, n) -> float."""
    raise NotImplementedError
