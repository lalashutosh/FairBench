"""Sampling backends. STUB (W1c)."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def sample(circuit: QuantumCircuit, params, shots: int, backend: str = "aer_statevector",
           seed: int | None = None, **kwargs) -> np.ndarray:
    """Sample bitstrings -> (shots, n) uint8 with column i = asset i (convert Qiskit little-endian).
    backend in {"aer_statevector","aer_mps","ibm","vtt"}; ibm/vtt raise NotImplementedError."""
    raise NotImplementedError
