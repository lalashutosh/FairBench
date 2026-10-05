"""Ansatz building blocks. STUB (W1a)."""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp


def xy_mixer(n: int, beta: Parameter, topology: str = "ring") -> QuantumCircuit:
    """exp(-i beta (XX+YY)/2) on edges of 'ring' or 'complete' topology; preserves Hamming weight."""
    raise NotImplementedError


def cost_layer(op: SparsePauliOp, gamma: Parameter) -> QuantumCircuit:
    """exp(-i gamma op)."""
    raise NotImplementedError


def build_ansatz(n: int, k: int, p: int, cost_op: SparsePauliOp, topology: str = "ring") -> QuantumCircuit:
    """Dicke prep then p x (cost layer, mixer); parameter vectors gamma and beta."""
    raise NotImplementedError
