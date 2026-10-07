"""QA-3: QAE attribution statistics vs exact enumeration and attribute(rebalance=False)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from demo_attribution import build_case  # noqa: E402

from fairbench.apps.attribution import attribute  # noqa: E402
from fairbench.baselines import enumerate_feasible  # noqa: E402
from fairbench.qae.attribution_qae import (ExactInstance, RankTable, SwapChainBank,  # noqa: E402
                                           classical_percentile, exact_masks, exhaustive_cost,
                                           iid_shared_sample, linear_return_scores,
                                           portfolio_score, qae_attribute, qae_percentile,
                                           qae_quantile, swap_connectivity)


@pytest.fixture(scope="module")
def case():
    u, cs, fund = build_case(16, 5, 0, 3.0, 0.08, 0.70, n_excluded=2)
    return u, cs, fund, ExactInstance.build(u, cs)


def test_scores_match_attribute_buy_and_hold(case):
    u, cs, fund, inst = case
    res = attribute(u, cs, fund, n_samples=500, rebalance=False, seed=3)
    assert portfolio_score(fund, u) == pytest.approx(res.fund, abs=1e-12)
    g = linear_return_scores(u)
    X = enumerate_feasible(u, cs)[:200]
    from fairbench.apps.attribution import _stats
    from fairbench.apps.attribution import _returns_array
    ref = _stats(X, _returns_array(u), "equal", u, "total_return", False, 252)
    np.testing.assert_allclose([portfolio_score(x, scores=g) for x in X], ref, atol=1e-12)


def test_exact_masks_reproduce_enumeration(case):
    u, cs, fund, inst = case
    s = portfolio_score(fund, u)
    F = enumerate_feasible(u, cs)
    from fairbench.apps.attribution import _returns_array, _stats
    sc = _stats(F, _returns_array(u), "equal", u, "total_return", False, 252)
    feas, below, scores = exact_masks(u, cs, s, inst=inst)
    assert feas.sum() == len(F) and inst.C == len(scores)
    assert below.sum() / feas.sum() == pytest.approx(np.mean(sc < s), abs=1.5 / len(sc))  # fund's own tie
    assert inst.percentile(s, "half") == pytest.approx(np.mean(sc < s), abs=1.5 / len(sc))
    assert inst.quantile(0.5) == pytest.approx(np.sort(sc)[(len(sc) + 1) // 2 - 1])


@pytest.mark.parametrize("method,kw", [("iqae", dict(eps=0.05)), ("mlae_exp", dict(K=8, shots=60))])
def test_qae_percentile_ci_coverage_frequency(case, method, kw):
    """Coverage frequency of the 95% CI over 30 independent repetitions (nominal 0.95; the
    delta-method ratio CI is slightly tight, so require >= 0.80), and the error is controlled."""
    u, cs, fund, inst = case
    ex = inst.percentile(portfolio_score(fund, u), "strict")
    cov, err = 0, []
    for i in range(30):
        r = qae_percentile(u, cs, fund, method=method, rng=np.random.default_rng(100 + i), inst=inst, **kw)
        cov += r.ci[0] <= ex <= r.ci[1]
        err.append(abs(r.estimate - ex))
    assert cov / 30 >= 0.80, f"coverage {cov}/30"
    assert np.median(err) < 0.02 and max(err) < 0.10


def test_qae_attribute_close_to_exact_and_accounting(case):
    u, cs, fund, inst = case
    a = qae_attribute(u, cs, fund, eps_pct=0.02, eps_med=0.02, rng=np.random.default_rng(5))
    e = a.exact
    assert abs(a.percentile - e["percentile"]) < 3.0
    assert abs(a.constraint_effect - e["constraint_effect"]) < 0.02
    assert abs(a.manager_effect - e["manager_effect"]) < 0.02
    q = a.queries
    assert q["total"] == q["percentile"] + q["null_median"] + q["benchmark_median"]
    p = a.details["percentile"]
    assert q["percentile"] == p.details["a_F"].oracle_queries + p.details["a_G"].oracle_queries + p.details["pilot_queries"]
    # null median rank error small, and benchmark uses the cardinality-only (all-feasible) set
    assert a.details["null_median"].details["rank_error"] <= 0.02   # 3 granules of 1/|F| (|F|=150)
    assert a.details["benchmark_median"].details["cardinality_only"]
    assert a.details["benchmark_median"].details["a_F_est"] == 1.0


def test_classical_baseline_unbiased(case):
    u, cs, fund, inst = case
    s = portfolio_score(fund, u)
    rng = np.random.default_rng(0)
    est = [classical_percentile(inst, s, 20_000, rng) for _ in range(40)]
    assert abs(np.mean(est) - inst.percentile(s, "strict")) < 0.01


def test_quantile_mlae_runs(case):
    u, cs, fund, inst = case
    errs = [qae_quantile(u, cs, 0.5, method="mlae_exp", K=9, shots=60, n_bisect=10,
                         rng=np.random.default_rng(2 + i), inst=inst).details["rank_error"] for i in range(6)]
    assert np.median(errs) <= 0.05 and max(errs) <= 0.15   # one fixed-(K,shots) bisection step can slip


def test_null_median_rank_error_realistic(case):
    """IQAE bisection at eps=0.02: median rank error <= 0.01 and worst of 8 seeds <= 0.05 (observed 0-0.033: heavy tail, 5 granules of 1/|F|)."""
    u, cs, fund, inst = case
    errs = [qae_quantile(u, cs, 0.5, "iqae", eps=0.02, rng=np.random.default_rng(50 + i), inst=inst,
                         oracle_kind="ideal").details["rank_error"] for i in range(8)]
    assert max(errs) <= 0.05 and np.median(errs) <= 0.01


def test_shared_sample_classical_comparison(case):
    u, cs, fund, inst = case
    a = qae_attribute(u, cs, fund, eps_pct=0.05, eps_med=0.05, rng=np.random.default_rng(9))
    q, c = a.queries, a.classical["queries"]
    # one shared i.i.d. set sized to the larger of the two quantum budgets; benchmark separate
    assert c["shared_set"] == max(q["percentile"], q["null_median"])
    assert c["benchmark"] == q["benchmark_median"]
    assert c["total"] == c["shared_set"] + c["benchmark"] <= c["separate_sets_total"] == q["total"]
    assert 0 < a.classical["feasible_in_shared_set"] <= c["shared_set"]
    e = a.exact
    assert abs(a.classical["percentile"] - e["percentile"]) < 3.0
    assert abs(a.classical["null_median"] - e["null_median"]) < 0.03
    assert abs(a.classical["constraint_effect"] - e["constraint_effect"]) < 0.03


def test_iid_shared_sample_statistics(case):
    u, cs, fund, inst = case
    tab = RankTable(inst, portfolio_score(fund, u))
    assert tab.exact_pct == pytest.approx(inst.percentile(tab.s, "strict"))
    rng = np.random.default_rng(1)
    big = [iid_shared_sample(tab, 2_000_000, rng) for _ in range(5)]
    assert all(abs(p - tab.exact_pct) < 0.01 and m < 0.01 for p, m, _ in big)
    p, m, nF = iid_shared_sample(tab, 10, np.random.default_rng(0))
    assert nF <= 10
    # error of the shared estimate shrinks like 1/sqrt(N_feasible): ~10x more samples -> ~3x lower rms
    e1 = np.sqrt(np.mean([(iid_shared_sample(tab, 2000, rng)[0] - tab.exact_pct) ** 2 for _ in range(200)]))
    e2 = np.sqrt(np.mean([(iid_shared_sample(tab, 20000, rng)[0] - tab.exact_pct) ** 2 for _ in range(200)]))
    assert 2.0 < e1 / e2 < 4.5


def test_swap_mcmc_converges_and_connectivity(case):
    u, cs, fund, inst = case
    conn = swap_connectivity(inst)
    assert 1 <= conn["n_components"] and 0 < conn["largest"] <= 1 and conn["start_coverage"] <= conn["largest"] + 1e-12
    # brute-force check of component structure on the enumerated feasible set
    F = np.flatnonzero(inst.feasible)
    sets = [frozenset(inst.idx[i].tolist()) for i in F]
    pos = {s: j for j, s in enumerate(sets)}
    par = list(range(len(sets)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for j, s in enumerate(sets):
        for i in s:
            for a_ in inst.allowed.tolist():
                if a_ not in s:
                    nb = (s - {i}) | {a_}
                    if nb in pos:
                        par[find(j)] = find(pos[nb])
    sizes = np.bincount([find(j) for j in range(len(sets))])
    assert conn["n_components"] == int((sizes > 0).sum())
    assert conn["largest"] == pytest.approx(sizes.max() / len(sets))
    tab = RankTable(inst, portfolio_score(fund, u))
    o = SwapChainBank(tab).run(16, 2 ** 14, np.random.default_rng(3))
    assert o["pct_err"].shape == (16, len(o["T"])) and o["init_q"].min() >= 1
    # connected instance -> long chains converge (median over chains, last checkpoint)
    if conn["n_components"] == 1:
        assert np.median(o["pct_err"][:, -1]) < 0.05 and np.median(o["med_err"][:, -1]) < 0.05
    assert exhaustive_cost(inst) == inst.C
