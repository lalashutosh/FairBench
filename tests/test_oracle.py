import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from fairbench.baselines import enumerate_feasible
from fairbench.constraints import (CarbonCap, Cardinality, ConstraintSet, Exclusion, MinESG,
                                   SectorCap)
from fairbench.data import synthetic_universe
from fairbench.ft import DEFAULT_BITS, feasibility_oracle, quantise, simulate, simulate_with_phase
from fairbench.ft.oracle import _all_weight_k, toffoli_count
from fairbench.instances import island_family, island_instance, p0_instance

INSTANCES = {"p0": lambda: p0_instance(), "island": lambda: island_instance()}
for _n in (12, 14):
    for _s in (0, 1, 2):
        INSTANCES[f"fam{_n}_{_s}"] = (lambda n=_n, s=_s: island_family(n, seed=s))


def _small(excl=True, weighted=True, n=6, k=3):
    u = synthetic_universe(n=n, n_sectors=2, seed=3)
    cons = [Cardinality(k), SectorCap("S0", 1)]
    if excl:
        cons.append(Exclusion([n - 1]))
    if weighted:
        cons.append(MinESG(float(np.median(u.esg_score))))
    return u, ConstraintSet(cons)


def _run_bit(u, cs, **kw):
    qc, info = feasibility_oracle(u, cs, mode="bit", **kw)
    X = _all_weight_k(u.n, cs.cardinality)
    inp = np.zeros((len(X), qc.num_qubits), np.uint8)
    inp[:, :u.n] = X
    return qc, info, X, inp, simulate(qc, inp)


def _rows(A):
    return {tuple(r) for r in np.asarray(A, np.uint8)}


@pytest.mark.parametrize("name", list(INSTANCES))
def test_bit_oracle_marks_exactly_feasible(name):
    u, cs = INSTANCES[name]()
    qc, info, X, inp, out = _run_bit(u, cs)
    n = u.n
    assert info.quantisation.bits == DEFAULT_BITS
    assert info.quantisation.exact and info.quantisation.joint_mismatch == 0
    assert np.array_equal(out[:, :n], X)                        # data untouched
    assert not out[:, n + 1:].any()                             # ancillas clean
    marked = X[out[:, n] == 1]
    assert _rows(marked) == _rows(enumerate_feasible(u, cs))   # marks exactly F
    # phase mode: sign == (-1)^f, everything else unchanged and clean
    qcp, _ = feasibility_oracle(u, cs, mode="phase")
    inp2 = np.zeros((len(X), qcp.num_qubits), np.uint8)
    inp2[:, :n] = X
    out2, sign = simulate_with_phase(qcp, inp2)
    assert np.array_equal(out2, inp2)
    assert np.array_equal(sign == -1, out[:, n] == 1)


@pytest.mark.parametrize("name", list(INSTANCES))
def test_toffoli_count_consistent(name):
    u, cs = INSTANCES[name]()
    for mode in ("bit", "phase"):
        qc, info = feasibility_oracle(u, cs, mode=mode)
        ops = qc.count_ops()
        assert set(ops) <= {"x", "cx", "ccx", "z", "cz", "ccz", "mcx"}
        mcx = sum(2 * i.operation.num_ctrl_qubits - 3 for i in qc.data if i.operation.name == "mcx")
        assert info.toffoli_count == ops.get("ccx", 0) + ops.get("ccz", 0) + mcx == toffoli_count(qc)
        assert info.toffoli_count == sum(c.get("toffoli", 0) for c in info.components.values())
        assert info.n_qubits == qc.num_qubits == info.n_data + info.n_ancilla
        # compute and uncompute are mirror images
        comp = sum(info.components[c]["toffoli"] for c in ("sector", "esg", "carbon", "cardinality"))
        assert info.components["uncompute"]["toffoli"] == comp


def test_exclusion_and_cardinality_flag():
    u, cs = _small(n=8, k=3)
    cs = ConstraintSet(cs.constraints + [CarbonCap(float(np.median(u.carbon)) + 50)])
    qc, info, X, inp, out = _run_bit(u, cs, check_cardinality=True, bits=8)
    assert info.quantisation.joint_mismatch == 0
    assert np.array_equal(X[out[:, u.n] == 1].astype(bool), X[cs.check_batch(X, u)].astype(bool))
    assert info.components["exclusion"]["literals"] == 1
    # with the cardinality flag, every non-weight-k input is rejected and ancillas stay clean
    allx = ((np.arange(2 ** u.n)[:, None] >> np.arange(u.n)) & 1).astype(np.uint8)
    inp = np.zeros((len(allx), qc.num_qubits), np.uint8)
    inp[:, :u.n] = allx
    o = simulate(qc, inp)
    assert not o[:, u.n + 1:].any() and np.array_equal(o[:, :u.n], allx)
    q = info.quantisation
    expect = (allx.sum(1) == 3) & q.check_batch(allx, u, cs)
    assert np.array_equal(o[:, u.n].astype(bool), expect)


def test_quantise_mc_and_auto():
    u, cs = island_family(18, seed=0)
    q = quantise(u, cs, bits=DEFAULT_BITS, mc_samples=20_000)
    assert not q.exact and q.n_eval == 20_000 and q.superset and q.false_neg == 0
    assert q.false_pos / max(q.n_feasible_eval, 1) < 0.02     # relative to |F|, not C(n,k)
    for r in q.rules:
        assert r.weights.min() >= 0 and r.weights.max() <= 2 ** DEFAULT_BITS - 1
    u, cs = p0_instance()
    qa = quantise(u, cs, "auto")
    assert qa.joint_mismatch == 0 and qa.bits <= DEFAULT_BITS


def test_phase_oracle_operator_tiny():
    """Full unitary check: phase oracle == diag((-1)^f) on the clean-ancilla subspace."""
    u, cs = _small(weighted=False, n=4, k=2)
    qc, info = feasibility_oracle(u, cs, mode="phase")
    assert qc.num_qubits <= 12
    U = Operator(qc).data
    n, N = u.n, qc.num_qubits
    xs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.uint8)
    f = (xs.sum(1) == 2) & ConstraintSet(cs.non_cardinality()).check_batch(xs, u)
    # oracle doesn't check cardinality: compare on all x with exact non-cardinality rules
    f_all = ConstraintSet(cs.non_cardinality()).check_batch(xs, u)
    cols = np.arange(2 ** n)  # ancillas = higher qubits = 0
    sub = U[np.ix_(cols, cols)]
    assert np.allclose(sub, np.diag(np.where(f_all, -1.0, 1.0)))
    assert np.allclose(np.abs(U[:, cols]).sum(0), 1)  # nothing leaks out of clean subspace
    assert f.sum() > 0


def test_phase_oracle_statevector_n6():
    u, cs = _small(n=6, k=3)
    qc, info = feasibility_oracle(u, cs, mode="phase", bits=3)
    assert info.quantisation.joint_mismatch == 0
    X = _all_weight_k(6, 3)
    f = cs.check_batch(X, u)
    # evolve the (Dicke-like) equal superposition of all weight-k |x>|0>: one statevector run
    idx = (X.astype(np.int64) << np.arange(6)).sum(1)
    psi = np.zeros(2 ** qc.num_qubits, complex)
    psi[idx] = 1 / np.sqrt(len(idx))
    out = Statevector(psi).evolve(qc).data
    expect = np.zeros_like(psi)
    expect[idx] = np.where(f, -1.0, 1.0) / np.sqrt(len(idx))
    assert np.allclose(out, expect)


def test_revsim_matches_statevector():
    rng = np.random.default_rng(1)
    qc = QuantumCircuit(6)
    for _ in range(60):
        g = rng.integers(5)
        q = rng.permutation(6)
        if g == 0:
            qc.x(q[0])
        elif g == 1:
            qc.cx(q[0], q[1])
        elif g == 2:
            qc.ccx(q[0], q[1], q[2])
        elif g == 3:
            qc.mcx(list(q[:4]), q[4])
        else:
            qc.ccz(q[0], q[1], q[2])
    for idx in range(0, 64, 5):
        bits = ((idx >> np.arange(6)) & 1).astype(np.uint8)
        out, sgn = simulate_with_phase(qc, bits[None])
        sv = Statevector.from_int(idx, 64).evolve(qc)
        j = int((out[0].astype(int) << np.arange(6)).sum())
        assert np.isclose(sv.data[j], sgn[0])


@pytest.mark.parametrize("n", [50, 100])
def test_superset_quantisation_no_false_negatives(n):
    """At n > 16 thresholds are ceil(t0 - k/2): every float-feasible state passes (F subset F_q)."""
    from fairbench.instances import scaled_family
    u, cs = scaled_family(n, seed=0, n_mc=20_000)
    q = quantise(u, cs, mc_samples=20_000)
    assert q.superset and not q.exact
    assert q.false_neg == 0 and q.n_feasible_eval > 500
    assert q.false_pos / q.n_feasible_eval < 0.02            # post-check loses < 2%
    # near-threshold states (the ones rounding can flip) also pass when float-feasible
    rng = np.random.default_rng(1)
    X = np.zeros((50_000, n), np.uint8)
    np.put_along_axis(X, np.argpartition(rng.random((50_000, n)), cs.cardinality - 1,
                                         axis=1)[:, :cs.cardinality], 1, axis=1)
    f = cs.check_batch(X, u)
    assert not (f & ~q.check_batch(X, u, cs)).any()


def test_built_oracle_matches_quantised_rule_n50():
    """Gate-level oracle (classical reversible sim) == Quantisation.check_batch at n = 50."""
    from fairbench.instances import scaled_family
    u, cs = scaled_family(50, seed=0, n_mc=20_000)
    q = quantise(u, cs, mc_samples=20_000)
    qc, info = feasibility_oracle(u, cs, mode="bit", quant=q)
    rng = np.random.default_rng(2)
    k = cs.cardinality
    X = np.zeros((2000, u.n), np.uint8)
    np.put_along_axis(X, np.argpartition(rng.random((2000, u.n)), k - 1, axis=1)[:, :k], 1, axis=1)
    inp = np.zeros((len(X), qc.num_qubits), np.uint8)
    inp[:, :u.n] = X
    out = simulate(qc, inp)
    assert np.array_equal(out[:, :u.n], X) and not out[:, u.n + 1:].any()
    expect = q.check_batch(X, u, cs)
    assert 0 < expect.sum() < len(X)
    assert np.array_equal(out[:, u.n].astype(bool), expect)
