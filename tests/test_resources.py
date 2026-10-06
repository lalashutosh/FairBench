import math

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from fairbench.constraints import Cardinality, ConstraintSet, Exclusion, MinESG, SectorCap
from fairbench.data import synthetic_universe
from fairbench.ft import feasibility_oracle, quantise
from fairbench.ft.oracle import _all_weight_k
from fairbench.ft.resources import (bbht_schedule, dicke_cost, dicke_rotation_counts,
                                    fixed_point_schedule, fixed_point_success, grover_circuit,
                                    grover_subspace_state, grover_success, known_schedule,
                                    oracle_formula_counts, reflection_circuit, reflection_cost,
                                    resource_estimate, ylc_subspace_state)
from fairbench.instances import island_family, island_instance, p0_instance, scaled_family
from fairbench.quantum.dicke import dicke_state

INST = {"p0": p0_instance, "island": island_instance, "fam14": lambda: island_family(14, seed=0)}


def _mask(u, cs):
    q = quantise(u, cs)
    X = _all_weight_k(u.n, cs.cardinality)
    return q.check_batch(X, u, cs), cs.check_batch(X, u)


@pytest.mark.parametrize("n,k", [(2, 1), (5, 2), (7, 3), (12, 4), (16, 5), (20, 19), (30, 6), (9, 0), (9, 9)])
def test_dicke_counts_match_circuit(n, k):
    ops = dict(dicke_state(n, k).count_ops())
    c = dicke_rotation_counts(n, k)
    for g in ("cry", "ccry", "cx", "x"):
        assert c[g] == ops.get(g, 0)
    d = dicke_cost(n, k)
    assert d.rotations == 2 * (c["cry"] + c["ccry"]) and d.toffoli == 2 * c["ccry"]
    assert d.rotation_depth <= d.rotations


def test_reflection_circuit():
    from qiskit.quantum_info import Operator
    for n in (2, 3, 4, 5):
        rc = reflection_cost(n)
        qc = reflection_circuit(n)
        assert sum(qc.count_ops().get(g, 0) for g in ("ccx", "ccz")) == rc.toffoli
        U = Operator(qc).data
        # restricted to ancillas |0>: I - 2|0><0| on data
        idx = np.arange(2 ** n)          # ancillas are high qubits -> low indices
        sub = U[np.ix_(idx, idx)]
        ref = np.eye(2 ** n)
        ref[0, 0] = -1
        assert np.allclose(sub, ref)


@pytest.mark.parametrize("n,seed", [(16, 1), (30, 2), (50, 0)])
def test_oracle_formula_matches_built(n, seed):
    u, cs = scaled_family(n, seed=seed, n_mc=20_000)
    q = quantise(u, cs, mc_samples=20_000)
    _, info = feasibility_oracle(u, cs, quant=q)
    f = oracle_formula_counts(u, cs, quant=q)
    assert (f["toffoli"], f["cnot"], f["n_ancilla"]) == (info.toffoli_count, info.cnot_count, info.n_ancilla)
    fb = oracle_formula_counts(u, cs, quant=q, mode="bit")
    _, ib = feasibility_oracle(u, cs, quant=q, mode="bit")
    assert (fb["toffoli"], fb["cnot"], fb["n_ancilla"]) == (ib.toffoli_count, ib.cnot_count, ib.n_ancilla)
    a = oracle_formula_counts(u, cs, approx=True)
    assert info.toffoli_count <= a["toffoli"] <= 1.05 * info.toffoli_count


@pytest.mark.parametrize("name", list(INST))
def test_oracle_formula_benchmarks(name):
    u, cs = INST[name]()
    _, info = feasibility_oracle(u, cs)
    f = oracle_formula_counts(u, cs, quant=info.quantisation)
    assert f["toffoli"] == info.toffoli_count and f["n_ancilla"] == info.n_ancilla


@pytest.mark.parametrize("name", list(INST))
def test_subspace_grover_matches_formula_and_uniform(name):
    u, cs = INST[name]()
    mq, mf = _mask(u, cs)
    assert (mq == mf).all()                      # zero quantisation mismatch here
    p = mq.mean()
    for r in range(6):
        pr = np.abs(grover_subspace_state(mq, r)) ** 2
        assert abs(pr[mq].sum() - grover_success(p, r)) < 1e-10
        assert np.ptp(pr[mq]) < 1e-12            # exactly uniform over F
        assert abs(pr.sum() - 1) < 1e-10


def test_fixed_point_and_ylc_sim():
    u, cs = island_family(14, seed=0)
    m, _ = _mask(u, cs)
    p = m.mean()
    for d in (0.1, 0.01):
        s = fixed_point_schedule(p, d)
        assert s["success_prob"] >= 1 - d * d - 1e-12
        pr = np.abs(ylc_subspace_state(m, s["iterations"], d)) ** 2
        assert abs(pr[m].sum() - s["success_prob"]) < 1e-10
        assert np.ptp(pr[m]) < 1e-12
        # guarantee holds for any P_F above the design lower bound
        s2 = fixed_point_schedule(p, d, p_lower=p / 4)
        for pp in (p / 4, p / 2, p, min(1, 3 * p)):
            assert fixed_point_success(pp, s2["iterations"], d) >= 1 - d * d - 1e-12


def test_bbht_scaling():
    vals = []
    for p in (0.1, 0.01, 1e-3, 1e-4):
        b = bbht_schedule(p, 1e10)
        assert np.isfinite(b["expected_oracle_calls"]) and b["expected_oracle_calls"] <= b["bbht_bound"]
        vals.append(b["expected_oracle_calls"] * math.sqrt(p))
    assert 0.3 < min(vals) and max(vals) < 9 / 4
    # Monte Carlo of the BBHT procedure agrees with the exact expectation
    p, N = 0.02, 2000
    th = math.asin(math.sqrt(p))
    rng = np.random.default_rng(0)
    tot = 0
    for _ in range(20000):
        m = 1.0
        while True:
            j = rng.integers(0, max(1, math.ceil(m - 1e-12)))
            tot += j
            if rng.random() < math.sin((2 * j + 1) * th) ** 2:
                break
            m = min(1.2 * m, math.sqrt(N))
    assert abs(tot / 20000 - bbht_schedule(p, N)["expected_oracle_calls"]) < 0.15


def test_known_schedule():
    for p in (0.5, 0.2, 0.03, 0.001):
        s = known_schedule(p)
        assert s["success_prob"] >= 0.5
        assert s["expected_oracle_calls"] <= 1.3 * (math.pi / 4) / math.sqrt(p) + 1


def _tiny():
    u = synthetic_universe(n=6, n_sectors=2, seed=3)
    cs = ConstraintSet([Cardinality(3), SectorCap("S0", 1), Exclusion([5]),
                        MinESG(float(np.median(u.esg_score)))])
    return u, cs


def test_full_circuit_small():
    u, cs = _tiny()
    q = quantise(u, cs, 2)
    X = _all_weight_k(u.n, 3)
    m = q.check_batch(X, u, cs)
    p = m.mean()
    assert 0 < p < 1
    codes = X @ (1 << np.arange(u.n))
    for r in (0, 1, 2):
        qc, info = grover_circuit(u, cs, r, bits=2, quant=q)
        assert qc.num_qubits <= 22
        probs = Statevector(qc).probabilities()
        data = probs.reshape(-1, 2 ** u.n)        # rows: ancilla configs (high qubits)
        assert abs(data[0].sum() - 1) < 1e-9      # ancillas clean
        pr = data[0][codes]
        assert abs(pr.sum() - 1) < 1e-9           # stays in weight-k subspace
        assert abs(pr[m].sum() - grover_success(p, r)) < 1e-9
        assert np.ptp(pr[m]) < 1e-9


def test_resource_estimate_api():
    u, cs = p0_instance()
    reps = {s: resource_estimate(u, cs, schedule=s) for s in ("known", "bbht", "fixed_point")}
    for s, r in reps.items():
        assert r.logical_qubits >= u.n and r.t_count_per_sample > 0
        assert r.toffoli_per_iterate >= r.components["oracle"]["toffoli"]
        assert r.components["dominant"] in ("oracle", "dicke_x2", "reflection")
    k = reps["known"]
    assert k.iterations == known_schedule(k.p_f)["iterations"]
    f = resource_estimate(u, cs, oracle_path="formula")
    assert f.toffoli_per_iterate == k.toffoli_per_iterate
    # T scales with t_per_toffoli
    k7 = resource_estimate(u, cs, t_per_toffoli=7)
    assert k7.t_per_iterate > k.t_per_iterate


def test_bbht_closed_form_and_huge_m():
    """Regression: bbht_schedule once allocated np.arange(M) per round (OOM at P_F ~ 1e-12)."""
    import time
    for p in (0.3, 0.01, 1e-5):
        th = math.asin(math.sqrt(p))
        for M in (1, 3, 50, 4097):
            js = np.arange(M)
            ref = np.mean(np.sin((2 * js + 1) * th) ** 2)
            assert abs(ref - (0.5 - math.sin(4 * M * th) / (4 * M * math.sin(2 * th)))) < 1e-12
    t0 = time.perf_counter()
    b = bbht_schedule(1e-12, 1e14)
    assert time.perf_counter() - t0 < 1.0
    assert np.isfinite(b["expected_oracle_calls"]) and b["expected_oracle_calls"] <= b["bbht_bound"]


def test_resource_totals_by_hand():
    u, cs = p0_instance()
    r = resource_estimate(u, cs, schedule="known")
    c = r.components
    dk = dicke_cost(u.n, cs.cardinality)
    rf = reflection_cost(u.n)
    assert r.toffoli_per_iterate == c["oracle"]["toffoli"] + 2 * dk.toffoli + rf.toffoli
    t_iter = r.toffoli_per_iterate * 4 + 2 * dk.rotations * r.t_per_rotation
    assert math.isclose(r.t_per_iterate, t_iter)
    t_prep = dk.toffoli * 4 + dk.rotations * r.t_per_rotation
    assert math.isclose(r.t_count_per_sample,
                        r.expected_attempts * t_prep + r.expected_oracle_calls_per_sample * t_iter)
    assert math.isclose(r.t_per_iterate, c["oracle"]["t"] + c["dicke_x2"]["t"] + c["reflection"]["t"])
