"""Ansatz building blocks: Dicke prep + (diagonal cost layer, Hamming-weight-preserving XY mixer)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp

from .dicke import dicke_state


def _edges(n: int, topology: str) -> list[tuple[int, int]]:
    if topology == "ring":
        if n < 2:
            return []
        if n == 2:
            return [(0, 1)]
        return [(i, (i + 1) % n) for i in range(n)]
    if topology == "complete":
        return list(combinations(range(n), 2))
    raise ValueError(f"unknown topology {topology!r}; use 'ring' or 'complete'")


def xy_mixer(n: int, beta: Parameter | float, topology: str = "ring") -> QuantumCircuit:
    """Apply exp(-i beta (X_i X_j + Y_i Y_j)/2) on every edge (i,j) of the topology.

    Convention: Qiskit's XXPlusYYGate(theta, 0) = exp(-i theta/4 (XX+YY)), so we use theta = 2*beta.
    On the {|01>,|10>} subspace this is a rotation [[cos b, -i sin b], [-i sin b, cos b]];
    |00>,|11> are untouched, so Hamming weight is preserved. Edges are applied sequentially
    (Trotterised product, not the exact exp of the summed ring/complete Hamiltonian).
    Ring for n=2 is the single edge (0,1); ring for n>=3 includes the wrap edge (n-1,0).
    """
    qc = QuantumCircuit(n, name="XYMixer")
    for i, j in _edges(n, topology):
        qc.append(XXPlusYYGate(2 * beta, 0.0), [i, j])
    return qc


def cost_layer(op: SparsePauliOp, gamma: Parameter | float) -> QuantumCircuit:
    """exp(-i gamma op) for a diagonal (Z-only) operator with real coefficients.

    Each term c * Z_{S} becomes a CX-parity ladder + RZ(2 gamma c) (RZ / RZZ for |S| = 1 / 2).
    The identity term is dropped (global phase). Raises ValueError on any X/Y component.
    """
    n = op.num_qubits
    qc = QuantumCircuit(n, name="Cost")
    for pauli, coeff in zip(op.paulis, op.coeffs):
        if np.any(pauli.x):
            raise ValueError(f"cost_layer requires a diagonal (Z-only) operator; got term {pauli}")
        if abs(np.imag(coeff)) > 1e-12:
            raise ValueError(f"cost_layer requires real coefficients; got {coeff}")
        c = float(np.real(coeff))
        qubits = [int(i) for i in np.flatnonzero(pauli.z)]
        if not qubits or c == 0.0:
            continue
        angle = 2 * c * gamma
        if len(qubits) == 1:
            qc.rz(angle, qubits[0])
        elif len(qubits) == 2:
            qc.rzz(angle, qubits[0], qubits[1])
        else:
            for a, b in zip(qubits[:-1], qubits[1:]):
                qc.cx(a, b)
            qc.rz(angle, qubits[-1])
            for a, b in reversed(list(zip(qubits[:-1], qubits[1:]))):
                qc.cx(a, b)
    return qc


def build_ansatz(
    n: int,
    k: int,
    p: int,
    cost_op: SparsePauliOp | None,
    topology: str = "ring",
) -> QuantumCircuit:
    """Dicke(n,k) prep, then p x (cost_layer(gamma_j), xy_mixer(beta_j)).

    Parameters: ParameterVector("γ", p) and ParameterVector("β", p) (2p total when cost_op given,
    p when cost_op is None -> mixer-only layers). p=0 gives the pure Dicke state.
    No measurements are added.
    """
    if p < 0:
        raise ValueError("p must be >= 0")
    if cost_op is not None and cost_op.num_qubits != n:
        raise ValueError(f"cost_op acts on {cost_op.num_qubits} qubits, expected {n}")
    qc = QuantumCircuit(n, name=f"Ansatz(n={n},k={k},p={p})")
    qc.compose(dicke_state(n, k), inplace=True)
    if p == 0:
        return qc
    gammas = ParameterVector("γ", p)
    betas = ParameterVector("β", p)
    for j in range(p):
        if cost_op is not None:
            qc.compose(cost_layer(cost_op, gammas[j]), inplace=True)
        qc.compose(xy_mixer(n, betas[j], topology), inplace=True)
    return qc
