from itertools import combinations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from scipy.linalg import expm

from fairbench.quantum.ansatz import build_ansatz, cost_layer, xy_mixer


def _weights(n):
    return np.array([bin(i).count("1") for i in range(2**n)])


def _random_weight_k_state(n, k, rng):
    idx = np.flatnonzero(_weights(n) == k)
    amp = np.zeros(2**n, dtype=complex)
    amp[idx] = rng.normal(size=len(idx)) + 1j * rng.normal(size=len(idx))
    return Statevector(amp / np.linalg.norm(amp))


def _random_diag_op(n, rng):
    terms = [("Z", [i], rng.normal()) for i in range(n)]
    terms += [("ZZ", [i, j], rng.normal()) for i, j in combinations(range(n), 2)]
    terms += [("", [], 1.7)]  # identity term
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


@pytest.mark.parametrize("topology", ["ring", "complete"])
@pytest.mark.parametrize("n,k", [(2, 1), (3, 1), (5, 2), (6, 3)])
def test_xy_mixer_preserves_weight(topology, n, k):
    rng = np.random.default_rng(1234 + n * 10 + k)
    w = _weights(n)
    for _ in range(3):
        beta = rng.uniform(-np.pi, np.pi)
        out = _random_weight_k_state(n, k, rng).evolve(xy_mixer(n, beta, topology))
        assert np.allclose(np.abs(out.data[w != k]), 0.0, atol=1e-10)
        assert np.isclose(np.sum(np.abs(out.data) ** 2), 1.0)


def test_xy_mixer_convention_single_edge():
    beta = 0.37
    XX = SparsePauliOp("XX").to_matrix()
    YY = SparsePauliOp("YY").to_matrix()
    target = expm(-1j * beta * (XX + YY) / 2)
    assert Operator(xy_mixer(2, beta)).equiv(Operator(target))
    np.testing.assert_allclose(Operator(xy_mixer(2, beta)).data, target, atol=1e-10)


def test_xy_mixer_edge_counts():
    assert xy_mixer(2, 0.1, "ring").size() == 1
    assert xy_mixer(5, 0.1, "ring").size() == 5
    assert xy_mixer(5, 0.1, "complete").size() == 10
    with pytest.raises(ValueError):
        xy_mixer(3, 0.1, "star")


def test_cost_layer_matches_exact_exponential():
    rng = np.random.default_rng(7)
    n = 4
    op = _random_diag_op(n, rng) + SparsePauliOp.from_sparse_list([("ZZZ", [0, 2, 3], 0.4)], n)
    gamma = 0.53
    target = expm(-1j * gamma * op.to_matrix())
    # equal up to global phase (identity term dropped)
    assert Operator(cost_layer(op, gamma)).equiv(Operator(target))


def test_cost_layer_rejects_non_diagonal():
    with pytest.raises(ValueError):
        cost_layer(SparsePauliOp(["ZI", "XZ"], coeffs=[1.0, 0.5]), 0.1)
    with pytest.raises(ValueError):
        cost_layer(SparsePauliOp(["YI"]), 0.1)


@pytest.mark.parametrize("topology", ["ring", "complete"])
def test_build_ansatz_stays_in_weight_k(topology):
    rng = np.random.default_rng(42)
    n, k, p = 6, 3, 2
    qc = build_ansatz(n, k, p, _random_diag_op(n, rng), topology)
    assert qc.num_parameters == 2 * p
    assert {par.name[0] for par in qc.parameters} == {"γ", "β"}
    assert qc.num_clbits == 0
    bound = qc.assign_parameters(rng.uniform(-np.pi, np.pi, size=2 * p))
    probs = Statevector(bound).probabilities()
    assert np.allclose(probs[_weights(n) != k], 0.0, atol=1e-10)
    assert np.isclose(probs.sum(), 1.0)


def test_build_ansatz_p0_is_dicke():
    qc = build_ansatz(5, 2, 0, None)
    assert qc.num_parameters == 0
    np.testing.assert_allclose(Statevector(qc).probabilities()[_weights(5) == 2], 0.1, atol=1e-10)


def test_build_ansatz_mixer_only():
    qc = build_ansatz(4, 2, 3, None)
    assert qc.num_parameters == 3
