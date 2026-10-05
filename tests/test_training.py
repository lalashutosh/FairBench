"""Tests for objectives.py and training.py (fast, seeded, no network)."""
import numpy as np
import pytest
from qiskit.circuit import Parameter

from fairbench.baselines import enumerate_feasible
from fairbench.constraints import Cardinality, ConstraintSet, Exclusion, SectorCap
from fairbench.data import synthetic_universe
from fairbench.objectives import (all_basis_states, chi2_from_counts, cvar_objective, encode,
                                  uniformity_objective)
from fairbench.quantum.ansatz import build_ansatz
from fairbench.quantum.hamiltonian import diag_values, penalty_operator
from fairbench.training import (TrainResult, default_x0, exact_probabilities, train,
                                train_multistart)


def _toy(n=5, k=2):
    u = synthetic_universe(n=n, n_sectors=2, seed=1)
    s0 = sorted(set(u.sector))[0]
    cs = ConstraintSet([Cardinality(k), SectorCap(s0, 1), Exclusion(np.array([n - 1]))])
    F = enumerate_feasible(u, cs)
    assert 0 < len(F) < 10
    return u, cs, F


def _probs_from(rows, weights, n):
    p = np.zeros(2 ** n)
    np.add.at(p, encode(rows), weights)
    return p / p.sum()


# ---------------------------------------------------------------- objectives
def test_uniform_over_F_has_zero_divergence():
    u, cs, F = _toy()
    obj = uniformity_objective(cs, u, F, lam=1.0)
    probs = _probs_from(F, np.ones(len(F)), u.n)
    s = obj.report_exact(probs)
    assert s["P_F"] == pytest.approx(1.0)
    assert s["chi2"] == pytest.approx(0.0, abs=1e-12)
    assert s["tv"] == pytest.approx(0.0, abs=1e-12)
    assert obj.exact(probs) == pytest.approx(-1.0)
    # sampled: each feasible state repeated equally -> U-statistic slightly below 0 (unbiased)
    X = np.repeat(F, 200, axis=0)
    r = obj.report(X)
    assert r["P_F"] == 1.0 and abs(r["chi2"]) < 0.02


def test_all_mass_off_F():
    u, cs, F = _toy()
    fc = set(encode(F))
    off = np.array([x for x in all_basis_states(u.n) if encode(x[None])[0] not in fc][:3])
    obj = uniformity_objective(cs, u, F, lam=0.3)
    probs = _probs_from(off, np.ones(len(off)), u.n)
    assert obj.report_exact(probs)["P_F"] == 0.0
    assert obj.report(np.repeat(off, 10, axis=0))["P_F"] == 0.0
    M = len(F)
    assert obj(np.repeat(off, 10, axis=0)) == pytest.approx(0.3 * (M - 1))  # worst-case fallback


def test_chi2_point_mass_and_estimator_unbiased():
    M = 6
    assert chi2_from_counts(np.array([50, 0, 0, 0, 0, 0]), M) == pytest.approx(M - 1)
    q = np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.1])
    true = M * (q ** 2).sum() - 1
    rng = np.random.default_rng(0)
    est = [chi2_from_counts(np.bincount(rng.choice(M, 20, p=q), minlength=M), M)
           for _ in range(4000)]
    assert np.mean(est) == pytest.approx(true, abs=0.03)


def test_cvar_exact_and_sampled():
    u, cs, F = _toy()
    op = penalty_operator(cs, u)
    diag = diag_values(op)
    n = u.n
    probs = np.full(2 ** n, 2.0 ** -n)
    for alpha in (1.0, 0.25):
        obj = cvar_objective(op, alpha)
        sorted_e = np.sort(diag)
        expect = sorted_e[: int(alpha * 2 ** n)].mean()
        assert obj.exact(probs) == pytest.approx(expect)
        assert obj(all_basis_states(n)) == pytest.approx(expect)
    # energy-function form agrees with operator form
    obj_fn = cvar_objective(lambda X: diag[encode(X)], 0.25)
    assert obj_fn.exact(probs) == pytest.approx(cvar_objective(op, 0.25).exact(probs))


# ---------------------------------------------------------------- training
@pytest.fixture(scope="module")
def tiny():
    u, cs, F = _toy()
    qc = build_ansatz(u.n, cs.cardinality, 1, cost_op=penalty_operator(cs, u), topology="ring")
    obj = uniformity_objective(cs, u, F, lam=0.1)
    return u, cs, F, qc, obj


def test_exact_and_sampled_objective_agree(tiny):
    from fairbench.backends import sample
    u, cs, F, qc, obj = tiny
    x = {p: 0.4 for p in qc.parameters}
    ex = obj.report_exact(exact_probabilities(qc, x))
    X = sample(qc, x, 20000, seed=3)
    sm = obj.report(X)
    assert sm["P_F"] == pytest.approx(ex["P_F"], abs=0.015)
    assert sm["chi2"] == pytest.approx(ex["chi2"], abs=0.05)
    assert obj(X) == pytest.approx(obj.exact(exact_probabilities(qc, x)), abs=0.02)


def test_binding_by_dict_not_order(tiny):
    u, cs, F, qc, obj = tiny
    names = [p.name for p in qc.parameters]
    assert names[0].startswith("β")  # Qiskit sorts β before γ
    x0 = default_x0(qc)
    g = [p for p in qc.parameters if p.name.startswith("γ")][0]
    b = [p for p in qc.parameters if p.name.startswith("β")][0]
    pa = exact_probabilities(qc, {g: 0.7, b: 0.1})
    pb = exact_probabilities(qc, {b: 0.1, g: 0.7})
    assert np.allclose(pa, pb)
    assert set(x0) == set(qc.parameters)


@pytest.mark.parametrize("opt", ["COBYLA", "SPSA"])
def test_train_exact_lowers_objective(tiny, opt):
    u, cs, F, qc, obj = tiny
    res = train(qc, obj, optimizer=opt, maxiter=40, exact=True, seed=1)
    assert isinstance(res, TrainResult)
    assert res.final_value < res.initial_value - 1e-3
    assert res.final_value == pytest.approx(obj.exact(exact_probabilities(qc, res.params)))
    assert set(res.params) == set(qc.parameters)
    assert all(isinstance(k, Parameter) for k in res.params)
    assert res.n_evals == len(res.history)
    assert res.n_shots_total == 0 and res.exact


@pytest.mark.parametrize("opt,maxiter", [("COBYLA", 20), ("SPSA", 4)])
def test_train_sampled_cost_accounting(tiny, opt, maxiter):
    u, cs, F, qc, obj = tiny
    shots = 512
    res = train(qc, obj, optimizer=opt, maxiter=maxiter, shots=shots, seed=2)
    assert res.n_evals == len(res.history)
    assert res.n_shots_total == res.n_evals * shots == res.shots_equiv
    assert res.wall_time > 0
    if opt == "SPSA":
        assert res.n_evals == 1 + 2 * 5 + 2 * maxiter + 1  # init + calib pairs + iters + final
    else:
        assert res.n_evals <= maxiter + 1
    # sampled training should not end far worse than where it began (judged exactly)
    e0 = obj.exact(exact_probabilities(qc, res.x0))
    e1 = obj.exact(exact_probabilities(qc, res.params))
    assert e1 < e0 + 0.05


def test_sampled_training_is_seeded(tiny):
    u, cs, F, qc, obj = tiny
    r1 = train(qc, obj, optimizer="COBYLA", maxiter=5, shots=128, seed=7)
    r2 = train(qc, obj, optimizer="COBYLA", maxiter=5, shots=128, seed=7)
    assert r1.history == r2.history


def test_multistart_sums_cost(tiny):
    u, cs, F, qc, obj = tiny
    res = train_multistart(qc, obj, n_starts=3, seed=0, optimizer="COBYLA", maxiter=15,
                           exact=True)
    assert res.n_starts == 3
    assert res.n_evals > len(res.history)
    single = train(qc, obj, optimizer="COBYLA", maxiter=15, exact=True)
    assert res.final_value <= single.final_value + 1e-12


def test_exact_requires_exact_objective(tiny):
    u, cs, F, qc, obj = tiny
    with pytest.raises(ValueError):
        train(qc, lambda X: 0.0, exact=True)


def test_spsa_is_seeded(tiny):
    u, cs, F, qc, obj = tiny
    r1 = train(qc, obj, optimizer="SPSA", maxiter=10, exact=True, seed=5)
    r2 = train(qc, obj, optimizer="SPSA", maxiter=10, exact=True, seed=5)
    assert r1.history == r2.history
