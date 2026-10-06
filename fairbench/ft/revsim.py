"""Fast classical simulator for reversible (X / CX / CCX / MCX) circuits on
computational-basis inputs, vectorised over many inputs with numpy.

Diagonal phase gates Z / CZ / CCZ / MCZ (``mcx``-style controlled-Z) are also
supported by ``simulate_with_phase``: on a basis state they only contribute a sign
(-1)^{AND of their qubits}, which is tracked per input. Barriers / identities are
ignored. Any other gate raises ValueError (the simulator is exact only for
permutation + diagonal-sign circuits).
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

_PERM = {"x", "cx", "ccx", "mcx", "mcx_gray", "mcx_recursive", "mcx_vchain"}
_PHASE = {"z", "cz", "ccz"}
_SKIP = {"barrier", "id", "delay"}


def _ops(circuit: QuantumCircuit):
    for inst in circuit.data:
        name = inst.operation.name
        qs = [circuit.find_bit(q).index for q in inst.qubits]
        if name in _SKIP:
            continue
        if name in _PERM:
            nc = getattr(inst.operation, "num_ctrl_qubits", len(qs) - 1)
            ctrl_state = getattr(inst.operation, "ctrl_state", (1 << nc) - 1)
            if name != "x" and ctrl_state != (1 << nc) - 1:
                raise ValueError("open controls not supported; use explicit X gates")
            yield "perm", qs[:-1], qs[-1]
        elif name in _PHASE:
            yield "phase", qs, None
        else:
            raise ValueError(f"revsim: unsupported gate {name!r}")


def simulate_with_phase(circuit: QuantumCircuit, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run ``circuit`` on each row of ``inputs`` ((m, n_qubits) of 0/1; column j = qubit j).
    Returns (outputs (m, n_qubits) uint8, signs (m,) int8 in {+1, -1})."""
    X = np.atleast_2d(np.asarray(inputs, dtype=bool))
    if X.shape[1] != circuit.num_qubits:
        raise ValueError(f"inputs have {X.shape[1]} columns, circuit has {circuit.num_qubits} qubits")
    S = np.ascontiguousarray(X.T)  # (n_qubits, m): one contiguous row per qubit
    neg = np.zeros(S.shape[1], dtype=bool)
    for kind, ctrls, tgt in _ops(circuit):
        if kind == "perm":
            if not ctrls:
                S[tgt] ^= True
            elif len(ctrls) == 1:
                S[tgt] ^= S[ctrls[0]]
            else:
                a = S[ctrls[0]] & S[ctrls[1]]
                for c in ctrls[2:]:
                    a &= S[c]
                S[tgt] ^= a
        else:
            a = S[ctrls[0]].copy()
            for c in ctrls[1:]:
                a &= S[c]
            neg ^= a
    return S.T.astype(np.uint8), np.where(neg, -1, 1).astype(np.int8)


def simulate(circuit: QuantumCircuit, inputs: np.ndarray) -> np.ndarray:
    """Outputs (m, n_qubits) uint8 of a reversible circuit on basis-state inputs
    (phase gates, if any, are ignored here; use ``simulate_with_phase``)."""
    return simulate_with_phase(circuit, inputs)[0]
