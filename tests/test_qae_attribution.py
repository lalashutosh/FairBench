"""QA-3: QAE attribution statistics vs exact enumeration and attribute(rebalance=False)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from demo_attribution import build_case  # noqa: E402

from fairbench.apps.attribution import attribute  # noqa: E402
from fairbench.baselines import enumerate_feasible  # noqa: E402
from fairbench.qae.attribution_qae import (ExactInstance, classical_percentile,  # noqa: E402
                                           exact_masks, linear_return_scores, portfolio_score,
                                           qae_attribute, qae_percentile, qae_quantile)


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


@pytest.mark.parametrize("method,kw", [("iqae", dict(eps=0.02)), ("mlae_exp", dict(K=8, shots=60))])
def test_qae_percentile_within_ci(case, method, kw):
    u, cs, fund, inst = case
    r = qae_percentile(u, cs, fund, method=method, rng=np.random.default_rng(11), inst=inst, **kw)
    ex = inst.percentile(portfolio_score(fund, u), "strict")
    w = r.ci[1] - r.ci[0]
    assert r.ci[0] - w <= ex <= r.ci[1] + w   # delta-method CI is slightly tight; allow one width
    assert abs(r.estimate - ex) < 0.05


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
    assert a.details["null_median"].details["rank_error"] < 0.05
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
    r = qae_quantile(u, cs, 0.5, method="mlae_exp", K=9, shots=60, n_bisect=10,
                     rng=np.random.default_rng(2), inst=inst)
    assert r.details["rank_error"] < 0.2
    assert r.oracle_queries > 0
