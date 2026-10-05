import itertools

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from fairbench.constraints import (Cardinality, CarbonCap, ConstraintSet,
                                   Exclusion, MinESG, SectorCap)
from fairbench.data import synthetic_universe
from fairbench.quantum.hamiltonian import (diag_values, objective_operator,
                                           penalty_exactness, penalty_operator)

N = 8
U = synthetic_universe(N, 2, seed=3)  # sectors S0 (even idx), S1 (odd idx); 4 each
ALL = ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1).astype(np.uint8)


def weight_k(k):
    return ALL.sum(axis=1) == k


def no_xy(op):
    return not np.any(op.paulis.x)


def test_diag_values_bit_order():
    # Z on qubit 2 only: x_2 = (1 - Z_2)/2
    op = SparsePauliOp.from_sparse_list([("", [], 0.5), ("Z", [2], -0.5)], 4)
    v = diag_values(op, 4)
    expect = ((np.arange(16) >> 2) & 1).astype(float)
    assert np.allclose(v, expect)


def test_diag_values_rejects_offdiag():
    with pytest.raises(ValueError):
        diag_values(SparsePauliOp.from_sparse_list([("X", [0], 1.0)], 2), 2)


@pytest.mark.parametrize("q", [0.0, 0.5, 3.0])
def test_objective_bruteforce(q):
    op = objective_operator(U, q)
    assert op.num_qubits == N and no_xy(op)
    X = ALL.astype(float)
    ref = -X @ U.mu + q * np.einsum("bi,ij,bj->b", X, U.cov, X)
    assert np.allclose(diag_values(op, N), ref, atol=1e-9)


def test_exclusion_exact_all_states():
    cs = ConstraintSet([Cardinality(3), Exclusion([1, 4, 6])])
    op = penalty_operator(cs, U)
    assert no_xy(op) and op.num_qubits == N
    assert np.allclose(diag_values(op, N), ALL[:, [1, 4, 6]].sum(axis=1), atol=1e-9)


def test_cardinality_not_penalised():
    op = penalty_operator(ConstraintSet([Cardinality(3)]), U)
    assert np.allclose(diag_values(op, N), 0.0)


@pytest.mark.parametrize("k", range(1, N + 1))
@pytest.mark.parametrize("cap", range(0, 5))
def test_sector_cap(k, cap):
    cs = ConstraintSet([Cardinality(k), SectorCap("S0", cap)])
    op = penalty_operator(cs, U)
    assert no_xy(op)
    v = diag_values(op, N)
    assert np.all(v >= -1e-9)  # nonneg everywhere
    wk = weight_k(k)
    feas = cs.check_batch(ALL, U)[wk]
    vk = v[wk]
    enc = penalty_exactness(cs, U)[1]["encoding"]
    # every violation is penalised, by >= 1 at first overshoot
    assert np.all(vk[~feas] >= 1 - 1e-9)
    if enc in ("exact", "trivial", "infeasible"):
        assert np.array_equal(np.isclose(vk, 0, atol=1e-9), feas)
    else:
        assert enc == "surrogate"
        s = ALL[wk][:, 0::2].sum(axis=1)  # S0 count
        top2 = (s == cap) | (s == cap - 1)
        assert np.allclose(vk[top2], 0, atol=1e-9)
        assert np.all(vk[feas & ~top2] > 0)
        # monotone in the S0 count beyond the cap
        for a in range(cap, 4):
            if not (np.any(s == a + 1) and np.any(s == a)):
                continue
            assert vk[s == a + 1].min() > vk[s == a].max() - 1e-9


def test_sector_exactness_small_caps():
    for cap in (0, 1):
        cs = ConstraintSet([Cardinality(4), SectorCap("S1", cap)])
        assert penalty_exactness(cs, U)[1]["encoding"] == "exact"
    cs = ConstraintSet([Cardinality(4), SectorCap("S1", 4)])
    assert penalty_exactness(cs, U)[1]["encoding"] == "trivial"


@pytest.mark.parametrize("k", [2, 3, 4])
def test_avg_surrogates(k):
    esg_t = float(np.median(U.esg_score))
    carb_t = float(np.median(U.carbon))
    for c in (MinESG(esg_t), CarbonCap(carb_t)):
        cs = ConstraintSet([Cardinality(k), c])
        assert penalty_exactness(cs, U)[1]["encoding"] == "surrogate"
        op = penalty_operator(cs, U)
        assert no_xy(op)
        v = diag_values(op, N)
        assert np.all(v >= -1e-12)
        wk = weight_k(k)
        X = ALL[wk].astype(float)
        vk = v[wk]
        feas = cs.check_batch(ALL, U)[wk]
        # zero => feasible (sound)
        assert np.all(feas[np.isclose(vk, 0, atol=1e-12)])
        # violated => positive
        assert np.all(vk[~feas] > 0)
        # upper bound on normalised hinge violation of the average
        if isinstance(c, MinESG):
            vals, viol = U.esg_score, c.min_avg_score - X @ U.esg_score / k
        else:
            vals, viol = U.carbon, X @ U.carbon / k - c.max_avg
        R = np.ptp(vals)
        assert np.all(vk * R >= np.maximum(0, viol) - 1e-9)


def test_weights_and_combination():
    cs = ConstraintSet([Cardinality(3), Exclusion([0]), SectorCap("S1", 1),
                        MinESG(50.0)])
    parts = [penalty_operator(ConstraintSet([Cardinality(3), c]), U)
             for c in cs.constraints[1:]]
    pv = [diag_values(p, N) for p in parts]
    v = diag_values(penalty_operator(cs, U, {1: 2.0, "SectorCap": 3.0,
                                             "MinESG": 0.0}), N)
    assert np.allclose(v, 2 * pv[0] + 3 * pv[1], atol=1e-9)
    v1 = diag_values(penalty_operator(cs, U), N)
    assert np.allclose(v1, sum(pv), atol=1e-9)


def test_empty_penalty_is_zero():
    op = penalty_operator(ConstraintSet([]), U)
    assert op.num_qubits == N
    assert np.allclose(diag_values(op, N), 0)
