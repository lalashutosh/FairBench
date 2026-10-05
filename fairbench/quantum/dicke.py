"""Dicke state preparation (Baertschi & Eidenbenz 2019, arXiv:1904.07358).

Deterministic split-and-cyclic-shift (SCS) construction: O(nk) gates, depth O(n), no ancillas.

Indexing: the paper's qubit j (1-based) is Qiskit qubit j-1. The paper's input
|0^{n-k} 1^k> (ones on the "right") therefore puts X on Qiskit qubits n-k..n-1.
The output |D^n_k> is permutation-symmetric, so Qiskit qubit i == asset i holds
regardless of this convention.
"""
from __future__ import annotations

import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate


def _scs(qc: QuantumCircuit, m: int, l_max: int) -> None:
    """Append SCS_{m,l_max} acting on paper qubits m-l_max..m (1-based)."""
    q = lambda j: j - 1  # paper index -> Qiskit index
    # gadget (i): two-qubit block on (m-1, m)
    theta = 2 * math.acos(math.sqrt(1 / m))
    qc.cx(q(m - 1), q(m))
    qc.cry(theta, q(m), q(m - 1))
    qc.cx(q(m - 1), q(m))
    # gadgets (ii)_l: three-qubit blocks on (m-l, m-l+1, m)
    for l in range(2, l_max + 1):
        theta = 2 * math.acos(math.sqrt(l / m))
        qc.cx(q(m - l), q(m))
        qc.append(RYGate(theta).control(2, annotated=False), [q(m), q(m - l + 1), q(m - l)])
        qc.cx(q(m - l), q(m))


def dicke_state(n: int, k: int) -> QuantumCircuit:
    """Deterministic Dicke state |D^n_k>: uniform superposition of all weight-k n-bit strings.

    U_{n,k} = SCS_{n,k} ... SCS_{k+1,k} . SCS_{k,k-1} ... SCS_{2,1} applied to |0^{n-k}1^k>.
    Edge cases: k=0 -> empty circuit (|0..0>); k=n -> X on every qubit.
    """
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need n>=1 and 0<=k<=n, got n={n}, k={k}")
    qc = QuantumCircuit(n, name=f"Dicke({n},{k})")
    for j in range(n - k, n):
        qc.x(j)
    if k == 0 or k == n:
        return qc
    for m in range(n, 1, -1):
        _scs(qc, m, min(k, m - 1))
    return qc
