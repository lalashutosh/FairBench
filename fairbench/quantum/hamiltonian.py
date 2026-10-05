"""Cost Hamiltonians. STUB (W1b)."""
from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp

from ..constraints import ConstraintSet
from ..data import Universe


def penalty_operator(cs: ConstraintSet, u: Universe, weights) -> SparsePauliOp:
    """Quadratic penalties for non-cardinality constraints (x_i=(1-Z_i)/2). Cardinality NOT penalised."""
    raise NotImplementedError


def objective_operator(u: Universe, risk_aversion: float) -> SparsePauliOp:
    """Mean-variance QUBO: -mu.x + risk_aversion * x^T Sigma x."""
    raise NotImplementedError
