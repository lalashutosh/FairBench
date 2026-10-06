import itertools

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm
from scipy.stats import chisquare

from fairbench.baselines import enumerate_feasible
from fairbench.instances import p0_instance
from fairbench.quantum.ansatz import _edges
from fairbench.quantum.hamiltonian import penalty_operator
from fairbench.quantum.proposal import (circuit_proposal, circuit_proposal_batch, codes_to_index,
                                        exact_proposal_matrix, proposal_from_Q, states_to_codes,
                                        subspace_basis, subspace_hamiltonian, trotter_circuit,
                                        trotter_proposal_matrix, trotter_unitary,
                                        subspace_diagonal, penalty_separation)

N, K = 6, 3
_rng = np.random.default_rng(0)
PEN = SparsePauliOp.from_sparse_list(
    [("Z", [i], float(c)) for i, c in enumerate(_rng.normal(size=N))] + [("ZZ", [0, 3], 0.7)], N)
H_FIELD = 0.3 * _rng.normal(size=N)
ALPHA = 0.8


def full_H(n, topology, alpha=0.0, pen=None, h=None):
    terms = []
    for i, j in _edges(n, topology):
        terms += [("XX", [i, j], -0.5), ("YY", [i, j], -0.5)]
    op = SparsePauliOp.from_sparse_list(terms, n)
    if pen is not None:
        op = op + alpha * pen
    if h is not None:
        op = op + SparsePauliOp.from_sparse_list([("Z", [i], h[i]) for i in range(n)], n)
    return op.to_matrix()


def test_basis_order_matches_enumerate_feasible():
    states, index = subspace_basis(N, K)
    assert states.shape == (20, N) and len(index) == 20
    assert np.all(states.sum(1) == K)
    for r, c in enumerate(itertools.combinations(range(N), K)):
        assert set(np.flatnonzero(states[r])) == set(c)
    assert np.array_equal(codes_to_index(states, index), np.arange(20))
    # feasible rows of a real instance map to increasing subspace indices
    u, cs = p0_instance()
    F = enumerate_feasible(u, cs)
    _, idx12 = subspace_basis(u.n, cs.cardinality)
    rows = codes_to_index(F, idx12)
    assert np.all(np.diff(rows) > 0)


@pytest.mark.parametrize("topology", ["ring", "complete"])
def test_hamiltonian_matches_full(topology):
    states, _ = subspace_basis(N, K)
    c = states_to_codes(states)
    Hf = full_H(N, topology, ALPHA, PEN, H_FIELD)
    H = subspace_hamiltonian(N, K, topology, ALPHA, PEN, H_FIELD)
    assert np.allclose(Hf[np.ix_(c, c)], H, atol=1e-12)
    # weight conservation: no full-H amplitude leaves the subspace
    other = np.setdiff1d(np.arange(2 ** N), c)
    assert np.allclose(Hf[np.ix_(other, c)], 0)
    Hs = subspace_hamiltonian(N, K, topology, ALPHA, PEN, H_FIELD, sparse=True)
    assert np.allclose(Hs.toarray(), H)


@pytest.mark.parametrize("t", [0.3, 1.7, 5.0])
def test_exact_Q_properties(t):
    H = subspace_hamiltonian(N, K, "ring", ALPHA, PEN, H_FIELD)
    Q = exact_proposal_matrix(t, H)
    assert np.all(Q >= -1e-15)
    assert np.allclose(Q, Q.T, atol=1e-12)
    assert np.allclose(Q.sum(1), 1) and np.allclose(Q.sum(0), 1)
    states, _ = subspace_basis(N, K)
    c = states_to_codes(states)
    Uf = expm(-1j * full_H(N, "ring", ALPHA, PEN, H_FIELD) * t)
    assert np.allclose(Q, np.abs(Uf[np.ix_(c, c)].T) ** 2, atol=1e-10)


def test_t0_identity():
    H = subspace_hamiltonian(N, K, "complete", ALPHA, PEN, H_FIELD)
    assert np.allclose(exact_proposal_matrix(0.0, H), np.eye(20))
    assert np.allclose(trotter_proposal_matrix(0.0, 3, N, K, "complete", ALPHA, PEN, H_FIELD),
                       np.eye(20))


def test_trotter_unitary_sign_and_symmetry():
    n, t = 4, 0.9
    pen = SparsePauliOp.from_sparse_list([("Z", [0], 1.0), ("ZZ", [1, 2], 0.5)], n)
    h = np.array([0.2, -0.1, 0.0, 0.4])
    E = expm(-1j * full_H(n, "ring", 1.0, pen, h) * t)
    U = trotter_unitary(n, t, 64, "ring", 1.0, pen, h)
    ph = np.vdot(U.ravel(), E.ravel())
    ph /= abs(ph)  # cost_layer drops identity terms -> global phase only
    assert np.abs(U * ph - E).max() < 1e-3
    for topo in ("ring", "complete"):
        for r in (1, 3):
            U = trotter_unitary(N, 1.1, r, topo, ALPHA, PEN, H_FIELD)
            assert np.allclose(U, U.T, atol=1e-12)


@pytest.mark.parametrize("topology", ["ring", "complete"])
def test_trotter_converges_to_exact(topology):
    t = 1.3
    Q = exact_proposal_matrix(t, subspace_hamiltonian(N, K, topology, ALPHA, PEN, H_FIELD))
    errs = []
    for r in (1, 2, 4, 8, 16):
        Qt = trotter_proposal_matrix(t, r, N, K, topology, ALPHA, PEN, H_FIELD)
        assert np.allclose(Qt.sum(1), 1)  # weight preserved: no leakage
        assert np.allclose(Qt, Qt.T, atol=1e-12)
        errs.append(np.abs(Qt - Q).max())
    assert all(b < a for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 5e-3
    assert errs[-2] / errs[-1] > 3  # second order: ~4x per doubling


def test_circuit_preserves_weight():
    states, _ = subspace_basis(N, K)
    X = states[[0, 7, 19] * 10]
    Y = circuit_proposal_batch(X, 1.0, 3, seed=5, topology="complete", alpha=ALPHA, pen_op=PEN,
                               h=H_FIELD)
    assert Y.shape == X.shape and Y.dtype == np.uint8
    assert np.all(Y.sum(1) == K)
    y = circuit_proposal(states[4], 1.0, 3, seed=1)
    assert y.shape == (N,) and y.sum() == K


def test_circuit_samples_match_exact_row():
    t, r, topo = 1.2, 12, "ring"
    states, index = subspace_basis(N, K)
    x_i = 3
    Q = exact_proposal_matrix(t, subspace_hamiltonian(N, K, topo, ALPHA, PEN, H_FIELD))
    Qt = trotter_proposal_matrix(t, r, N, K, topo, ALPHA, PEN, H_FIELD)
    shots = 4000
    Y = circuit_proposal(states[x_i], t, r, seed=11, topology=topo, alpha=ALPHA, pen_op=PEN,
                         h=H_FIELD, shots=shots)
    cnt = np.bincount(codes_to_index(Y, index), minlength=len(states))
    keep = Qt[x_i] * shots > 5
    # exact circuit distribution
    exp_t = Qt[x_i][keep] / Qt[x_i][keep].sum() * cnt[keep].sum()
    assert chisquare(cnt[keep], exp_t).pvalue > 1e-3
    # and the exact (un-Trotterised) row
    exp_e = Q[x_i][keep] / Q[x_i][keep].sum() * cnt[keep].sum()
    assert chisquare(cnt[keep], exp_e).pvalue > 1e-3


def test_batch_matches_row():
    t, r = 0.8, 6
    states, index = subspace_basis(N, K)
    Qt = trotter_proposal_matrix(t, r, N, K, "ring")
    Y = circuit_proposal_batch(np.repeat(states[[2]], 2000, axis=0), t, r, seed=2)
    cnt = np.bincount(codes_to_index(Y, index), minlength=20)
    keep = Qt[2] * 2000 > 5
    exp = Qt[2][keep] / Qt[2][keep].sum() * cnt[keep].sum()
    assert chisquare(cnt[keep], exp).pvalue > 1e-3


def test_proposal_from_Q():
    H = subspace_hamiltonian(N, K, "ring")
    Q = exact_proposal_matrix(1.0, H)
    prop = proposal_from_Q(Q, 0)
    ys = np.array([prop(5) for _ in range(5000)])
    cnt = np.bincount(ys, minlength=20)
    keep = Q[5] * 5000 > 5
    exp = Q[5][keep] / Q[5][keep].sum() * cnt[keep].sum()
    assert chisquare(cnt[keep], exp).pvalue > 1e-3


def test_real_penalty_operator():
    u, cs = p0_instance()
    pen = penalty_operator(cs, u)
    H = subspace_hamiltonian(u.n, cs.cardinality, "ring", 2.0, pen)
    Q = exact_proposal_matrix(1.5, H)
    assert Q.shape == (495, 495) and np.allclose(Q, Q.T)
    qc = trotter_circuit(u.n, None, 1.0, 2, "ring", 2.0, pen)
    assert qc.num_qubits == 12


def test_penalty_separation_toy():
    d = np.array([0.0, 1.0, 2.0, 3.0, 1.0])
    m = np.array([True, True, False, False, False])
    r = penalty_separation(d, m)
    # infeasible scores {2,3,1} vs feasible {0,1}: pairs >: 2+2+1, tie (1 vs 1): 0.5 -> 5.5/6
    assert r["auc"] == pytest.approx(5.5 / 6)
    assert r["frac_infeasible_le_max_feasible"] == pytest.approx(1 / 3)
    assert penalty_separation(m.astype(float) * -1 + 1, m)["auc"] == 1.0
    with pytest.raises(ValueError):
        penalty_separation(d, np.ones(5, bool))


def test_subspace_diagonal_kinds():
    u, cs = p0_instance()
    n, k = u.n, cs.cardinality
    states, _ = subspace_basis(n, k)
    feas = np.asarray(cs.check_batch(states, u), bool)
    ind = subspace_diagonal("indicator", n, k, u, cs)
    assert np.array_equal(ind, (~feas).astype(float))
    vio = subspace_diagonal("violation", n, k, u, cs)
    assert np.allclose(vio[feas], 0) and (vio[~feas] > 0).all()
    sur = subspace_diagonal("surrogate", n, k, u, cs)
    full = np.zeros(2 ** n)
    from fairbench.quantum.hamiltonian import diag_values
    raw = diag_values(penalty_operator(cs, u), n)[states_to_codes(states)]
    assert np.allclose(sur, raw / raw.std())
    assert penalty_separation(ind, feas)["auc"] == 1.0
    with pytest.raises(ValueError):
        subspace_diagonal("nope", n, k, u, cs)
    # subspace_hamiltonian accepts the (C,) vector directly == the 2^n form
    full[states_to_codes(states)] = sur
    H1 = subspace_hamiltonian(n, k, "ring", 0.7, sur)
    H2 = subspace_hamiltonian(n, k, "ring", 0.7, full)
    assert np.allclose(H1, H2)
    assert np.allclose(np.diag(H1), 0.7 * sur)
