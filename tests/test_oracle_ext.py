"""Oracle extensions: CountBound, AvgBound, LinearThreshold (return comparator) and the
quantisation modes (superset / subset / minmis). Gate-level reversible simulation of the
built oracle is compared with the quantised rule and with the float rule."""
import math

import numpy as np
import pytest

from fairbench.constraints import (AvgBound, CarbonCap, Cardinality, ConstraintSet, CountBound,
                                   Exclusion, LinearThreshold, MinESG, MinGroups, SectorCap,
                                   TrackingErrorCap, VolatilityCap)
from fairbench.data import synthetic_returns, synthetic_universe
from fairbench.ft import feasibility_oracle, quantise, simulate, simulate_with_phase
from fairbench.ft.oracle import QUANT_MODES, _all_weight_k, _random_weight_k
from fairbench.ft.resources import oracle_formula_counts
from fairbench.instances import island_family, p0_instance, scaled_family


def _gross(n, seed):
    """Random gross returns g_i = prod_t (1 + r_it) stand-in (lognormal around 1.08)."""
    return np.exp(np.random.default_rng(seed).normal(0.08, 0.25, n))


def _q(vals, p):
    """Threshold strictly between attained values near quantile p (no exact ties)."""
    v = np.unique(vals)
    j = min(max(int(p * (len(v) - 1)), 0), len(v) - 2)
    return 0.5 * (v[j] + v[j + 1])


def _ret_rule(u, k, p, direction, seed=0, tie=False):
    g = _gross(u.n, seed)
    X = _all_weight_k(u.n, k)
    t = _q(X @ g, p)
    if tie:  # threshold exactly on an attained value: exercises strict vs non-strict
        t = float(np.sort(X @ g)[int(p * len(X))])
    return LinearThreshold(g, t, direction, label="ret")


def _with_attr(u, seed=5):
    u.attributes = {"women": np.random.default_rng(seed).uniform(10, 60, u.n)}
    return u


def _avg_q(u, k, attr, p):
    X = _all_weight_k(u.n, k)
    return _q(X @ u.numeric(attr) / k, p)


# ------------------------------------------------------------------- small instances (exact)
def _c_count_island():
    u, cs = island_family(12, seed=0)
    return u, ConstraintSet(cs.constraints + [CountBound([0, 2, 4, 6, 8, 10, 11], 2, None)])


def _c_count_p0():
    u, cs = p0_instance()
    k = cs.cardinality
    return u, ConstraintSet(cs.constraints + [CountBound(list(range(0, u.n, 2)), 1, k - 1, "even")])


def _c_count_only():
    u = synthetic_universe(10, 2, seed=1)
    return u, ConstraintSet([Cardinality(4), Exclusion([9]),
                             CountBound([0, 1, 2, 3, 9], 1, 2), CountBound([3, 3, 5, 7], 2, 3)])


def _c_avg_esg():
    u = synthetic_universe(12, 3, seed=2)
    return u, ConstraintSet([Cardinality(4), AvgBound("esg", lower=_avg_q(u, 4, "esg", 0.4))])


def _c_avg_attr_both():
    u = _with_attr(synthetic_universe(11, 3, seed=3))
    lo, hi = _avg_q(u, 4, "women", 0.25), _avg_q(u, 4, "women", 0.8)
    return u, ConstraintSet([Cardinality(4), SectorCap("S0", 2), AvgBound("women", lo, hi)])


def _c_ret_lt():
    u = synthetic_universe(12, 3, seed=4)
    return u, ConstraintSet([Cardinality(5), _ret_rule(u, 5, 0.3, "<")])


def _c_ret_ge():
    u = synthetic_universe(12, 3, seed=4)
    return u, ConstraintSet([Cardinality(5), _ret_rule(u, 5, 0.6, ">=", seed=1)])


def _c_ret_lt_tie():
    u = synthetic_universe(10, 2, seed=6)
    return u, ConstraintSet([Cardinality(4), _ret_rule(u, 4, 0.5, "<", seed=2, tie=True)])


def _c_ret_ge_tie():
    u = synthetic_universe(10, 2, seed=6)
    return u, ConstraintSet([Cardinality(4), _ret_rule(u, 4, 0.5, ">=", seed=2, tie=True)])


def _c_island_ret():
    u, cs = island_family(12, seed=1)
    return u, ConstraintSet(cs.constraints + [_ret_rule(u, cs.cardinality, 0.5, "<", seed=3)])


def _c_p0_ret_ge():
    u, cs = p0_instance()
    return u, ConstraintSet(cs.constraints + [_ret_rule(u, cs.cardinality, 0.3, ">=", seed=4)])


def _c_signed_linear():
    u = synthetic_universe(12, 3, seed=7)
    w = np.random.default_rng(8).normal(0, 1, 12)
    X = _all_weight_k(12, 4)
    return u, ConstraintSet([Cardinality(4), Exclusion([1]), LinearThreshold(w, _q(X @ w, 0.5), "<")])


def _c_everything():
    u, cs = island_family(12, seed=2)
    u = _with_attr(u)
    k = cs.cardinality
    return u, ConstraintSet(cs.constraints + [
        CountBound([0, 1, 2, 3, 4, 5], 1, 4), AvgBound("women", lower=_avg_q(u, k, "women", 0.2)),
        AvgBound("esg", upper=_avg_q(u, k, "esg", 0.9)), _ret_rule(u, k, 0.6, "<", seed=5)])


CASES = {f.__name__[3:]: f for f in (_c_count_island, _c_count_p0, _c_count_only, _c_avg_esg,
                                     _c_avg_attr_both, _c_ret_lt, _c_ret_ge, _c_ret_lt_tie,
                                     _c_ret_ge_tie, _c_island_ret, _c_p0_ret_ge, _c_signed_linear,
                                     _c_everything)}


def _sim(qc, X, n):
    inp = np.zeros((len(X), qc.num_qubits), np.uint8)
    inp[:, :n] = X
    return inp, simulate(qc, inp)


@pytest.mark.parametrize("name", list(CASES))
def test_bit_oracle_equals_quantised_equals_float(name):
    u, cs = CASES[name]()
    n, k = u.n, cs.cardinality
    assert n <= 12
    qc, info = feasibility_oracle(u, cs, mode="bit")
    q = info.quantisation
    assert q.exact and q.mode == "minmis" and q.joint_mismatch == 0
    X = _all_weight_k(n, k)
    inp, out = _sim(qc, X, n)
    assert np.array_equal(out[:, :n], X)              # data untouched
    assert not out[:, n + 1:].any()                   # ancillas clean
    flag = out[:, n].astype(bool)
    fq, ff = q.check_batch(X, u, cs), cs.check_batch(X, u)
    assert np.array_equal(flag, fq)
    assert np.array_equal(flag, ff)
    assert 0 < ff.sum() < len(X)                      # non-trivial instance
    # phase mode: sign == (-1)^f, register unchanged
    qcp, ip = feasibility_oracle(u, cs, mode="phase", quant=q)
    inp2 = np.zeros((len(X), qcp.num_qubits), np.uint8)
    inp2[:, :n] = X
    out2, sign = simulate_with_phase(qcp, inp2)
    assert np.array_equal(out2, inp2)
    assert np.array_equal(sign == -1, ff)
    # formula counts == built counts (both modes)
    for mode, inf in (("bit", info), ("phase", ip)):
        f = oracle_formula_counts(u, cs, quant=q, mode=mode)
        assert (f["toffoli"], f["cnot"], f["n_ancilla"], f["n_qubits"], f["n_literals"]) == \
            (inf.toffoli_count, inf.cnot_count, inf.n_ancilla, inf.n_qubits, inf.n_literals)
        for c, d in f["components"].items():
            if "toffoli" in d and c in inf.components:
                assert d["toffoli"] == inf.components[c]["toffoli"], c
    # bookkeeping: per-component Toffolis add up, compute == uncompute
    assert info.toffoli_count == sum(c.get("toffoli", 0) for c in info.components.values())
    comp = sum(info.components[c]["toffoli"] for c in ("sector", "count", "esg", "carbon", "avg",
                                                       "linear", "cardinality"))
    assert info.components["uncompute"]["toffoli"] == comp


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("mode", ["superset", "subset"])
def test_exact_modes_bracket_and_gate_level(name, mode):
    u, cs = CASES[name]()
    n, k = u.n, cs.cardinality
    q = quantise(u, cs, mode=mode)
    assert q.mode == mode and q.superset == (mode == "superset")
    assert (q.false_neg if mode == "superset" else q.false_pos) == 0
    X = _all_weight_k(n, k)
    qc, _ = feasibility_oracle(u, cs, mode="bit", quant=q)
    _, out = _sim(qc, X, n)
    assert not out[:, n + 1:].any() and np.array_equal(out[:, :n], X)
    flag, ff = out[:, n].astype(bool), cs.check_batch(X, u)
    assert np.array_equal(flag, q.check_batch(X, u, cs))
    if mode == "superset":
        assert not (ff & ~flag).any()
    else:
        assert not (flag & ~ff).any()


def test_bracket_counts_small():
    u, cs = _c_everything()
    X = _all_weight_k(u.n, cs.cardinality)
    lo = quantise(u, cs, mode="subset").check_batch(X, u, cs).sum()
    hi = quantise(u, cs, mode="superset").check_batch(X, u, cs).sum()
    assert lo <= cs.check_batch(X, u).sum() <= hi


# ------------------------------------------------------------------- n = 40 Monte Carlo
@pytest.fixture(scope="module")
def big():
    u, cs = scaled_family(40, seed=0, n_mc=20_000)
    k = cs.cardinality
    u = _with_attr(u, seed=11)
    rng = np.random.default_rng(3)
    Xs = _random_weight_k(u.n, k, 20_000, rng)
    g = _gross(u.n, 9)
    cons = cs.constraints + [CountBound(list(range(0, 40, 3)), 1, None),
                             AvgBound("women", lower=float(np.quantile(Xs @ u.numeric("women") / k, 0.2))),
                             LinearThreshold(g, float(np.quantile(Xs @ g, 0.7)), "<", "ret")]
    return u, ConstraintSet(cons)


@pytest.mark.parametrize("mode", ["superset", "subset"])
def test_bracketing_n40_monte_carlo(big, mode):
    u, cs = big
    n, k = u.n, cs.cardinality
    q = quantise(u, cs, mc_samples=20_000, mode=mode)
    assert not q.exact and q.mode == mode
    held = _random_weight_k(n, k, 50_000, np.random.default_rng(77))
    fq, ff = q.check_batch(held, u, cs), cs.check_batch(held, u)
    assert ff.sum() > 300
    if mode == "superset":
        assert q.false_neg == 0 and not (ff & ~fq).any()
        assert q.false_pos / q.n_feasible_eval < 0.1
    else:
        assert q.false_pos == 0 and not (fq & ~ff).any()
        assert q.false_neg / q.n_feasible_eval < 0.1
    # gate-level oracle == quantised rule on random inputs
    qc, info = feasibility_oracle(u, cs, mode="bit", quant=q)
    X = held[:1500]
    _, out = _sim(qc, X, n)
    assert np.array_equal(out[:, :n], X) and not out[:, n + 1:].any()
    assert np.array_equal(out[:, n].astype(bool), fq[:1500])
    f = oracle_formula_counts(u, cs, quant=q, mode="bit")
    assert (f["toffoli"], f["cnot"], f["n_ancilla"]) == (info.toffoli_count, info.cnot_count, info.n_ancilla)


def test_default_mode_large_is_superset(big):
    u, cs = big
    q = quantise(u, cs, mc_samples=5_000)
    assert q.mode == "superset" and q.superset and q.false_neg == 0


# ------------------------------------------------------------------- API
def test_mode_api_and_legacy_kwarg():
    u, cs = _c_ret_lt()
    assert set(QUANT_MODES) == {"auto", "superset", "subset", "minmis"}
    a = quantise(u, cs, superset=True)
    b = quantise(u, cs, mode="superset")
    assert a.mode == b.mode == "superset" and [r.threshold for r in a.rules] == [r.threshold for r in b.rules]
    assert quantise(u, cs, superset=False).mode == "minmis"
    assert quantise(u, cs).mode == "minmis"            # exact -> minmis
    with pytest.raises(ValueError):
        quantise(u, cs, superset=True, mode="subset")
    with pytest.raises(ValueError):
        quantise(u, cs, mode="bogus")
    r = quantise(u, cs).rules[0]
    assert r.kind == "linear" and r.strict and r.label.startswith("ret<")


def test_linear_threshold_class():
    rng = np.random.default_rng(0)
    w = rng.normal(size=7)
    X = rng.integers(0, 2, (300, 7))
    u = synthetic_universe(7, 2, seed=0)
    for d, op in ((">=", np.greater_equal), ("<", np.less)):
        c = LinearThreshold(w, 0.1, d)
        assert np.array_equal(c.check_batch(X, u), op(X @ w, 0.1))
        assert c.check(X[0], u) == bool(op(X[0] @ w, 0.1))
    assert LinearThreshold(w, 0.0, ">=").check(np.zeros(7), u)      # empty: sum 0
    assert not LinearThreshold(w, 0.0, "<").check(np.zeros(7), u)
    with pytest.raises(ValueError):
        LinearThreshold(w, 0.0, "<=")
    with pytest.raises(ValueError):
        LinearThreshold([1.0, np.nan], 0.0)
    with pytest.raises(ValueError):
        LinearThreshold(w, 0.0).check_batch(np.ones((2, 6)), u)


def test_return_below_is_buy_and_hold_total_return():
    """(1/k) sum x_i g_i - 1 < s  <=>  LinearThreshold(g, k (1 + s), "<"), g = prod(1 + r)."""
    from fairbench.apps.attribution import performance, portfolio_returns
    u = synthetic_universe(12, 3, seed=1)
    u.returns = synthetic_returns(u, periods=60, seed=2)
    R = u.returns.to_numpy()
    g = np.prod(1.0 + R, axis=0)
    k = 4
    X = _all_weight_k(12, k)
    tr = performance(portfolio_returns(X / k, R, rebalance=False), "total_return")
    assert np.allclose(tr, X @ g / k - 1, atol=1e-12)
    s = float(np.median(tr))
    c = LinearThreshold(g, k * (1 + s), "<")
    near = np.abs(tr - s) < 1e-9
    assert np.array_equal(c.check_batch(X, u)[~near], (tr < s)[~near])


@pytest.mark.parametrize("make", [lambda: MinGroups("sector", 2), lambda: VolatilityCap(0.5),
                                  lambda: TrackingErrorCap(0.5)])
def test_nonlinear_rules_still_refused(make):
    u = synthetic_universe(8, 2, seed=0)
    c = make()
    cs = ConstraintSet([Cardinality(3), c])
    for mode in ("bit", "phase"):
        with pytest.raises(TypeError, match=type(c).__name__):
            feasibility_oracle(u, cs, mode=mode)
    with pytest.raises(TypeError, match=type(c).__name__):
        oracle_formula_counts(u, cs, approx=True)


def test_avgbound_missing_attribute_and_bad_countbound():
    u = synthetic_universe(8, 2, seed=0)
    with pytest.raises(KeyError):
        quantise(u, ConstraintSet([Cardinality(3), AvgBound("nope", lower=1.0)]))
    with pytest.raises((ValueError, IndexError)):
        feasibility_oracle(u, ConstraintSet([Cardinality(3), CountBound([0, 9], 1)]))


def test_countbound_max_matches_sectorcap_cost():
    """SectorCap(s, c) == CountBound(members of s, max_count=c): same circuit cost."""
    u, cs = island_family(12, seed=0)
    caps = [c for c in cs.constraints if isinstance(c, SectorCap)]
    swapped = [CountBound([i for i in range(u.n) if u.sector[i] == c.sector], 0, c.max_count)
               if isinstance(c, SectorCap) else c for c in cs.constraints]
    assert caps
    _, a = feasibility_oracle(u, cs)
    _, b = feasibility_oracle(u, ConstraintSet(swapped), quant=a.quantisation)
    assert (a.toffoli_count, a.cnot_count, a.n_ancilla) == (b.toffoli_count, b.cnot_count, b.n_ancilla)
    assert a.components["sector"]["toffoli"] == b.components["count"]["toffoli"]


def test_approx_formula_for_new_rules():
    u, cs = scaled_family(30, seed=1, n_mc=5_000)
    g = _gross(30, 2)
    X = _random_weight_k(30, cs.cardinality, 5_000, np.random.default_rng(0))
    cs = ConstraintSet(cs.constraints + [LinearThreshold(g, float(np.median(X @ g)), "<")])
    q = quantise(u, cs, mc_samples=5_000)
    _, info = feasibility_oracle(u, cs, quant=q)
    a = oracle_formula_counts(u, cs, approx=True)
    assert 0.95 * info.toffoli_count <= a["toffoli"] <= 1.05 * info.toffoli_count
    assert info.components["linear"]["toffoli"] > 0
    assert math.isclose(info.components["linear"]["toffoli"], info.components["esg"]["toffoli"], rel_tol=0.25)
