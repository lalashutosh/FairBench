"""Weight rules, the weight grid, the three reference distributions and the exact validator."""
import math

import numpy as np
import pytest
from scipy import stats

from fairbench.constraints import (Cardinality, ConstraintSet, Exclusion, HoldingsRange, PositionBound,
                                   SectorCap, WeightedAverage, WeightedRisk, WeightRuleSet, WeightSum,
                                   active_risk)
from fairbench.data import synthetic_universe
from fairbench.portfolio.exact import (compare_sample, compare_to_exact, exact_benchmark_aware,
                                       exact_uniform_subsets, exact_uniform_weights, tv_noise_floor)
from fairbench.portfolio.reference import (benchmark_aware, reference_statistics, uniform_subsets,
                                           uniform_weights, weighted_percentile_of, weighted_quantile)
from fairbench.portfolio.weights import (WeightGrid, composition_counts, enumerate_grid, grid_counts,
                                         grid_size, sample_compositions, sample_grid, subset_weights)


@pytest.fixture
def u():
    return synthetic_universe(8, 3, seed=1)


# ------------------------------------------------------------- weight rules
def test_position_bound_ignores_unheld_and_checks_held(u):
    r = PositionBound(0.1, 0.4)
    W = np.array([[0.5, 0.5, 0, 0, 0, 0, 0, 0], [0.4, 0.4, 0.2, 0, 0, 0, 0, 0],
                  [0.05, 0.35, 0.3, 0.3, 0, 0, 0, 0]])
    assert r.check_weights(W, u).tolist() == [False, True, False]


def test_weight_sum_and_holdings_range(u):
    W = np.array([[0.5, 0.3, 0.2, 0, 0, 0, 0, 0], [0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0]])
    assert WeightSum([0, 1], upper=0.5).check_weights(W, u).tolist() == [False, True]
    assert WeightSum([0, 1], lower=0.5).check_weights(W, u).tolist() == [True, False]
    assert HoldingsRange(3, 4).check_weights(W, u).tolist() == [True, False]


def test_weighted_average_missing_is_never_zero(u):
    v = np.array([10.0, np.nan, 30.0, 0, 0, 0, 0, 0])
    W = np.array([[0.5, 0.5, 0, 0, 0, 0, 0, 0], [0.5, 0, 0.5, 0, 0, 0, 0, 0]])
    strict = WeightedAverage(v, upper=25.0)
    assert strict.check_weights(W, u).tolist() == [False, True]  # NaN held -> cannot be evaluated -> fails
    assert np.isnan(strict.average(W, u)[0]) and strict.average(W, u)[1] == pytest.approx(20.0)
    loose = WeightedAverage(v, upper=25.0, missing="renormalise")
    assert loose.average(W, u)[0] == pytest.approx(10.0)  # 0.5*10 / 0.5 covered, not 0.5*10 + 0.5*0
    assert loose.check_weights(W, u).tolist() == [True, True]


def test_weighted_average_by_field_name(u):
    W = np.full((1, 8), 1 / 8)
    assert WeightedAverage("esg", lower=0).average(W, u)[0] == pytest.approx(u.esg_score.mean())
    with pytest.raises(KeyError):
        WeightedAverage("nope", lower=0).check_weights(W, u)


def test_weighted_risk_is_quadratic_form(u):
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(8), size=5)
    b = np.full(8, 1 / 8)
    te = active_risk(W, u.cov, b)
    for w, t in zip(W, te):
        assert t == pytest.approx(math.sqrt((w - b) @ u.cov @ (w - b)))
    # doubling the active position doubles TE: quadratic, not linear, in the variance
    assert active_risk(b + 2 * (W[0] - b), u.cov, b)[0] == pytest.approx(2 * te[0])
    cap = WeightedRisk(float(np.median(te)), b)
    assert cap.check_weights(W, u).tolist() == (te <= np.median(te) + 1e-9).tolist()


def test_rule_set_combines_support_and_weight_rules(u):
    rules = WeightRuleSet([PositionBound(0, 0.5)], support_rules=ConstraintSet([Exclusion([0])]))
    W = np.array([[0.5, 0.5, 0, 0, 0, 0, 0, 0], [0, 0.5, 0.5, 0, 0, 0, 0, 0], [0, 0.6, 0.4, 0, 0, 0, 0, 0],
                  [0, 0.5, 0.4, 0, 0, 0, 0, 0]])
    assert rules.check_weights(W, u).tolist() == [False, True, False, False]  # last row does not sum to 1
    v = rules.violations(W[:3], u)
    assert v["support_rules"] == 1 and v["0:PositionBound"] == 1


# --------------------------------------------------------------- the grid
def test_composition_counts_small_cases():
    c = composition_counts(4, 2, 1, 3)
    assert c[2][4] == 3  # (1,3) (2,2) (3,1)
    assert composition_counts(10, 3, 1, 10)[3][10] == math.comb(9, 2)  # stars and bars
    assert composition_counts(5, 2, 3, 5)[2][5] == 0


def test_grid_size_matches_enumeration_and_brute_force():
    g = WeightGrid(n=5, units=6, min_units=1, max_units=4, k_min=2, k_max=3)
    Q = enumerate_grid(g)
    assert len(Q) == grid_size(g) == sum(grid_counts(g).values())
    assert g.contains(Q).all()
    assert len({tuple(q) for q in Q.tolist()}) == len(Q)
    brute = [q for q in np.ndindex(*(5 * [5])) if sum(q) == 6 and 2 <= sum(x > 0 for x in q) <= 3]
    assert len(brute) == len(Q)


def test_grid_validation_and_allowed():
    with pytest.raises(ValueError):
        WeightGrid(n=3, units=4, k_min=4)
    g = WeightGrid(n=5, units=4, k_min=2, k_max=2, allowed=(1, 3, 4))
    Q = enumerate_grid(g)
    assert (Q[:, [0, 2]] == 0).all() and len(Q) == 3 * 3
    assert not g.contains(np.array([[2, 2, 0, 0, 0]]))[0]
    with pytest.raises(ValueError):
        enumerate_grid(WeightGrid(n=30, units=40, k_min=10, k_max=10), limit=1000)


def test_sample_compositions_is_uniform():
    rng = np.random.default_rng(0)
    S = sample_compositions(7, 3, 1, 4, 60_000, rng)
    assert (S.sum(axis=1) == 7).all() and S.min() >= 1 and S.max() <= 4
    _, counts = np.unique(S, axis=0, return_counts=True)
    assert len(counts) == composition_counts(7, 3, 1, 4)[3][7]
    assert stats.chisquare(counts).pvalue > 1e-3


def test_sample_grid_is_uniform_across_supports_and_k():
    g = WeightGrid(n=4, units=5, min_units=1, max_units=3, k_min=2, k_max=3)
    Q = sample_grid(g, 80_000, seed=3)
    assert g.contains(Q).all()
    _, counts = np.unique(Q, axis=0, return_counts=True)
    assert len(counts) == grid_size(g)
    assert stats.chisquare(counts).pvalue > 1e-3


def test_subset_weights_policies():
    X = np.array([[1, 1, 0, 0], [0, 1, 1, 1]])
    assert subset_weights(X).tolist() == [[0.5, 0.5, 0, 0], [0, 1 / 3, 1 / 3, 1 / 3]]
    b = np.array([0.4, 0.1, 0.3, 0.2])
    assert subset_weights(X, "benchmark", b)[0] == pytest.approx([0.8, 0.2, 0, 0])
    with pytest.raises(ValueError):
        subset_weights(np.array([[1, 0, 0, 0]]), "benchmark", np.array([0, 1, 1, 1.0]))
    with pytest.raises(ValueError):
        subset_weights(X, "cap")


# ------------------------------------------- the memo's worked toy example
def test_weight_rule_changes_which_names_are_feasible():
    """3 assets, 2 holdings, weights in quarters. D1 has 3 portfolios and D2 has 9. With
    "asset 0 at most 1/4", equal weights leave only {1,2}; the grid keeps 5 portfolios
    and two of them hold asset 0."""
    u3 = synthetic_universe(3, 1, seed=0)
    cs = ConstraintSet([Cardinality(2)])
    grid = WeightGrid(n=3, units=4, min_units=1, max_units=3, k_min=2, k_max=2)
    assert exact_uniform_subsets(u3, cs).n_feasible == 3
    assert exact_uniform_weights(u3, grid, None).n_feasible == 9
    cap = WeightRuleSet([WeightSum([0], upper=0.25)])
    d1 = exact_uniform_subsets(u3, cs, weight_rules=cap)
    d2 = exact_uniform_weights(u3, grid, cap)
    assert d1.n_feasible == 1 and d1.inclusion()[0] == 0
    assert d2.n_feasible == 5 and d2.inclusion()[0] == pytest.approx(2 / 5)
    # same mean return without the cap, wider spread on the grid
    r = np.array([0.10, -0.05, 0.02])
    a, b = exact_uniform_subsets(u3, cs), exact_uniform_weights(u3, grid, None)
    assert a.prob @ a.returns(r) == pytest.approx(b.prob @ b.returns(r))
    assert np.var(b.returns(r)) > np.var(a.returns(r))


# ------------------------------------------------- samplers against exact
def _small_case():
    u = synthetic_universe(10, 3, seed=4)
    cs = ConstraintSet([Cardinality(4), SectorCap("S0", 2), Exclusion([9])])
    rng = np.random.default_rng(7)
    b = rng.dirichlet(np.ones(10) * 2)
    r = rng.normal(0.02, 0.08, size=10)
    return u, cs, b, r


@pytest.mark.parametrize("policy", ["equal", "benchmark"])
def test_uniform_subsets_matches_exact(policy):
    u, cs, b, r = _small_case()
    exact = exact_uniform_subsets(u, cs, policy, b)
    ref = uniform_subsets(u, cs, 20_000, policy=policy, benchmark=b, seed=1)
    assert ref.n_violations == 0 and ref.weights.sum(axis=1) == pytest.approx(1.0)
    m = compare_sample(exact, ref, r, realised_return=float(np.median(exact.returns(r))))
    assert m["tv"] < 2.5 * m["tv_noise_floor"]
    assert m["marginal_error_max"] < 0.02 and abs(m["percentile_error"]) < 2.0
    assert m["kl"] is not None and m["kl"] < 0.05
    # the estimated count brackets the true one
    lo, hi = ref.log10_n_feasible_ci
    assert lo <= math.log10(exact.n_feasible) <= hi


def test_uniform_subsets_with_weight_rule_matches_exact():
    u, cs, b, r = _small_case()
    cap = WeightRuleSet([PositionBound(0, 0.45)])
    exact = exact_uniform_subsets(u, cs, "benchmark", b, cap)
    assert exact.n_feasible < exact_uniform_subsets(u, cs, "benchmark", b).n_feasible
    ref = uniform_subsets(u, cs, 20_000, policy="benchmark", benchmark=b, weight_rules=cap, seed=2)
    m = compare_sample(exact, ref, r)
    assert ref.n_violations == 0 and m["tv"] < 2.5 * m["tv_noise_floor"]


def test_uniform_weights_matches_exact():
    u = synthetic_universe(6, 2, seed=2)
    grid = WeightGrid(n=6, units=8, min_units=1, max_units=4, k_min=2, k_max=3)
    rules = WeightRuleSet([WeightSum(np.flatnonzero(np.array(u.sector) == "S0"), upper=0.5),
                           WeightedAverage("esg", lower=float(np.median(u.esg_score)))],
                          support_rules=ConstraintSet([Exclusion([5])]))
    exact = exact_uniform_weights(u, grid, rules)
    assert 0 < exact.n_feasible < grid_size(grid)
    ref = uniform_weights(u, grid, rules, 30_000, seed=5)
    r = np.random.default_rng(0).normal(0, 0.1, 6)
    m = compare_sample(exact, ref, r, realised_return=0.0)
    assert ref.n_violations == 0
    assert m["tv"] < 2.5 * m["tv_noise_floor"] and m["marginal_error_max"] < 0.02
    lo, hi = ref.log10_n_feasible_ci
    assert lo <= math.log10(exact.n_feasible) <= hi


def test_benchmark_aware_matches_exact_and_limits():
    u, cs, b, r = _small_case()
    base_exact = exact_uniform_subsets(u, cs, "equal")
    base = uniform_subsets(u, cs, 40_000, seed=3)
    te = active_risk(base_exact.weights, u.cov, b)
    tau = float(np.median(te))
    exact = exact_benchmark_aware(base_exact, u, b, tau)
    ref = benchmark_aware(base, u, b, tau)
    assert ref.distribution == "benchmark_aware" and ref.info["tau"] == tau
    assert ref.ess < base.ess and ref.importance.sum() == pytest.approx(1.0)
    m = compare_sample(exact, ref, r)
    assert m["tv"] < 3 * m["tv_noise_floor"]
    # tau -> infinity recovers the base distribution; tau -> 0 concentrates on the minimum-TE portfolio
    assert exact_benchmark_aware(base_exact, u, b, 1e6).prob == pytest.approx(base_exact.prob)
    tight = exact_benchmark_aware(base_exact, u, b, tau / 50)
    assert tight.prob.argmax() == te.argmin() and tight.prob.max() > 0.9
    with pytest.raises(ValueError):
        benchmark_aware(base, u, b, 0.0)


def test_pseudo_funds_have_uniform_percentiles():
    """Calibration: a fund drawn from the feasible set itself has a uniform percentile."""
    u, cs, b, r = _small_case()
    ref = uniform_subsets(u, cs, 4000, seed=11)
    ret = ref.returns(r)
    pseudo = uniform_subsets(u, cs, 500, seed=12).returns(r)
    pct = np.array([weighted_percentile_of(ret, x) for x in pseudo]) / 100
    assert stats.kstest(pct, "uniform").pvalue > 0.01


def test_reference_statistics_and_labels():
    u, cs, b, r = _small_case()
    ref = uniform_subsets(u, cs, 5000, seed=0)
    s = reference_statistics(ref, r, realised_return=float(np.quantile(ref.returns(r), 0.7)))
    assert s["p05"] <= s["median"] <= s["p95"]
    assert 65 < s["percentile"] < 75 and 0 < s["percentile_se"] < 2
    assert s["within_mandate_return_difference"] == pytest.approx(s["realised_return"] - s["median"])
    assert not any("manager" in k or "skill" in k for k in s)
    man = ref.manifest()
    for key in ("distribution", "definition", "sampler", "seed", "acceptance_rate", "log10_n_feasible",
                "n_violations", "weighting_policy", "effective_sample_size"):
        assert key in man
    with pytest.raises(ValueError):
        ref.returns(np.r_[r[:-1], np.nan])


def test_weighted_quantile_and_percentile():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert weighted_quantile(v, 0.5) == 2.0
    assert weighted_quantile(v, [0.0, 1.0]).tolist() == [1.0, 4.0]
    w = np.array([0.7, 0.1, 0.1, 0.1])
    assert weighted_quantile(v, 0.5, w) == 1.0
    assert weighted_percentile_of(v, 2.0) == pytest.approx(37.5)
    assert weighted_percentile_of(v, 2.0, w) == pytest.approx(75.0)


def test_rejection_gives_up_on_infeasible_rules():
    u, cs, b, r = _small_case()
    impossible = WeightRuleSet([PositionBound(0, 0.1)])  # 4 equal weights of 25% can never fit
    with pytest.raises(RuntimeError, match="too tight"):
        uniform_subsets(u, cs, 10, weight_rules=impossible, batch=1000, max_cost=3000)


# ----------------------------------------------------- the metrics themselves
def test_compare_to_exact_detects_bias_violations_and_missing_states():
    u, cs, b, r = _small_case()
    exact = exact_uniform_subsets(u, cs)
    good = compare_to_exact(exact, exact.sample(20_000, seed=0), asset_returns=r, realised_return=0.02)
    assert good["feasibility_rate"] == 1.0 and good["tv"] < 2.5 * good["tv_noise_floor"]
    assert good["kl"] is not None and abs(good["percentile_error"]) < 2

    # a sampler stuck on a quarter of the feasible set
    stuck = exact.weights[np.random.default_rng(0).integers(exact.n_feasible // 4, size=20_000)]
    bad = compare_to_exact(exact, stuck, asset_returns=r)
    assert bad["tv"] > 0.7 and bad["kl"] is None and bad["kl_smoothed"] > good["kl_smoothed"]
    assert bad["coverage"] <= 0.26 and bad["marginal_error_max"] > good["marginal_error_max"]

    # infeasible draws are counted, not silently dropped
    junk = np.vstack([exact.sample(900, seed=1), np.full((100, u.n), 1 / u.n)])
    assert compare_to_exact(exact, junk)["feasibility_rate"] == pytest.approx(0.9)
    assert compare_to_exact(exact, np.full((5, u.n), 1 / u.n))["tv"] == 1.0
    # a pre-filtered sampler reports its raw draws
    assert compare_to_exact(exact, exact.sample(500, seed=2), n_raw=2000)["feasibility_rate"] == 0.25


def test_tv_noise_floor_shrinks_with_samples():
    p = np.full(50, 1 / 50)
    assert tv_noise_floor(p, 100) > tv_noise_floor(p, 10_000) > 0
