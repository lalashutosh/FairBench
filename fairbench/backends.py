"""Sampling backends.

Bit-order convention: Qiskit bitstrings are little-endian (rightmost char = qubit 0).
Output arrays have column i = qubit i = asset i.

Credentials for hardware backends come from environment variables only
(IBM: QISKIT_IBM_TOKEN; VTT/IQM: IQM_TOKEN).
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile


def bitstrings_to_array(bitstrings: list[str], n: int) -> np.ndarray:
    """Convert Qiskit little-endian bitstrings to (len, n) uint8, column i = qubit i."""
    out = np.zeros((len(bitstrings), n), dtype=np.uint8)
    for r, s in enumerate(bitstrings):
        s = s.replace(" ", "")
        if len(s) != n:
            raise ValueError(f"bitstring {s!r} has length {len(s)}, expected {n}")
        out[r] = [1 if c == "1" else 0 for c in reversed(s)]
    return out


def ansatz_param_order(circuit: QuantumCircuit) -> list[str]:
    """Parameter names in the order a sequence of values is bound (Qiskit's name-sorted order)."""
    return [p.name for p in circuit.parameters]


def _bind(circuit: QuantumCircuit, params) -> QuantumCircuit:
    bound = _bind_raw(circuit, params)
    if bound.parameters:
        names = sorted(p.name for p in bound.parameters)
        raise ValueError(f"circuit still has unbound parameters after binding: {names}")
    return bound


def _bind_raw(circuit: QuantumCircuit, params) -> QuantumCircuit:
    if params is None:
        if circuit.parameters:
            raise ValueError("circuit has unbound parameters but params=None")
        return circuit.copy()
    if isinstance(params, dict):
        return circuit.assign_parameters(params, inplace=False)
    vals = list(np.asarray(params, dtype=float).ravel())
    # Qiskit's circuit.parameters is sorted by name; sequence is bound in that order.
    if len(vals) != len(circuit.parameters):
        raise ValueError(f"got {len(vals)} params, circuit has {len(circuit.parameters)}")
    return circuit.assign_parameters(vals, inplace=False)


def sample(circuit: QuantumCircuit, params, shots: int, backend: str = "aer_statevector",
           seed: int | None = None, **kwargs) -> np.ndarray:
    """Sample bitstrings -> (shots, n) uint8 with column i = asset i.

    params: None, a sequence (bound in ``circuit.parameters`` order, which is Qiskit's
    name-sorted order), or a dict Parameter -> value.
    backend in {"aer_statevector","aer_mps","ibm","vtt"}.
    kwargs (aer_mps): bond_dim (max bond dimension), truncation_threshold.
    ibm/vtt raise NotImplementedError.
    """
    if backend == "ibm":
        raise NotImplementedError(
            "IBM backend not wired yet; credentials will be read from env var QISKIT_IBM_TOKEN.")
    if backend == "vtt":
        raise NotImplementedError(
            "VTT/IQM backend not wired yet; credentials will be read from env var IQM_TOKEN.")
    if backend not in ("aer_statevector", "aer_mps"):
        raise ValueError(f"unknown backend {backend!r}")

    from qiskit_aer import AerSimulator

    n = circuit.num_qubits
    qc = _bind(circuit, params)
    qc.measure_all()

    if backend == "aer_statevector":
        sim = AerSimulator(method="statevector", seed_simulator=seed)
    else:
        opts = {}
        if kwargs.get("bond_dim") is not None:
            opts["matrix_product_state_max_bond_dimension"] = int(kwargs["bond_dim"])
        if kwargs.get("truncation_threshold") is not None:
            opts["matrix_product_state_truncation_threshold"] = float(kwargs["truncation_threshold"])
        sim = AerSimulator(method="matrix_product_state", seed_simulator=seed, **opts)

    tqc = transpile(qc, sim, seed_transpiler=seed)
    res = sim.run(tqc, shots=int(shots), memory=True).result()
    return bitstrings_to_array(res.get_memory(), n)
