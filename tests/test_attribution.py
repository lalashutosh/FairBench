import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from fairbench.apps.attribution import (attribute, dicke_sampler, performance,
                                        plot_attribution, portfolio_returns,
                                        rejection_sampler, weights_matrix)
from fairbench.baselines import enumerate_feasible, rejection_sample
from fairbench.constraints import Cardinality, ConstraintSet
from fairbench.data import load_returns, load_universe, synthetic_returns
from fairbench.instances import p0_instance


def _case(seed=1, mu_shift=None):
    u, cs = p0_instance()
    u.returns = synthetic_returns(u, periods=120, mu_shift=mu_shift, seed=seed)
    return u, cs, enumerate_feasible(u, cs)


def _exact(u, F):
    return performance(portfolio_returns(weights_matrix(F, "equal", u), u.returns.to_numpy()))


# ------------------------------------------------------------------- data
def test_synthetic_returns_shape_seed_and_shift():
    u, _ = p0_instance()
    a, b = synthetic_returns(u, 50, seed=3), synthetic_returns(u, 50, seed=3)
    assert a.shape == (50, u.n) and list(a.columns) == u.tickers and a.equals(b)
    shift = np.zeros(u.n)
    shift[0] = 2.52
    d = synthetic_returns(u, 50, mu_shift=shift, seed=3) - a
    assert np.allclose(d.iloc[:, 0], 0.01) and np.allclose(d.iloc[:, 1:], 0.0)


def test_load_returns_reorders_and_validates(tmp_path):
    u, _ = p0_instance()
    r = synthetic_returns(u, 10, seed=0)
    r[u.tickers[::-1]].assign(EXTRA=0.0).to_csv(tmp_path / "r.csv")
    got = load_returns(tmp_path / "r.csv", u.tickers)
    assert list(got.columns) == u.tickers and np.allclose(got.to_numpy(), r.to_numpy())
    with pytest.raises(ValueError):
        load_returns(tmp_path / "r.csv", u.tickers + ["NOPE"])
    pd.DataFrame(dict(ticker=u.tickers, mu=u.mu, sector=u.sector, esg_score=u.esg_score,
                      carbon=u.carbon)).to_csv(tmp_path / "u.csv", index=False)
    assert load_universe(tmp_path / "u.csv", returns_path=tmp_path / "r.csv").returns.shape == (10, u.n)


# ---------------------------------------------------------------- returns
def test_portfolio_returns_rebalance_vs_buy_and_hold():
    R = np.array([[0.10, 0.00], [0.00, 0.10], [-0.05, 0.20]])
    w = np.array([[0.5, 0.5], [1.0, 0.0]])
    assert np.allclose(portfolio_returns(w, R), w @ R.T)
    bh = portfolio_returns(w, R, rebalance=False)
    growth = np.prod(1 + R, axis=0)
    assert np.allclose(performance(bh), w @ growth - 1)
    assert np.allclose(bh[1], R[:, 0])  # single asset: both conventions agree


def test_performance_metrics():
    P = np.array([[0.1, -0.1, 0.2], [0.01, 0.01, 0.01]])
    assert np.allclose(performance(P), [1.1 * 0.9 * 1.2 - 1, 1.01**3 - 1])
    sh = performance(P, "sharpe", periods_per_year=4)
    assert np.isclose(sh[0], P[0].mean() / P[0].std(ddof=1) * 2) and sh[1] == 0.0
    with pytest.raises(ValueError):
        performance(P, "alpha")


# --------------------------------------------------------------- samplers
def test_rejection_sampler_exact_count_feasible_seeded():
    u, cs, _ = _case()
    a, b = rejection_sampler(u, cs, 700, seed=4), rejection_sampler(u, cs, 700, seed=4)
    assert a.samples.shape == (700, u.n) and a.samples.dtype == np.uint8
    assert cs.check_batch(a.samples, u).all() and np.array_equal(a.samples, b.samples)
    assert a.cost >= 700 and a.cost_unit == "proposals" and a.info["n_accepted"] >= 700


def test_sampler_gives_up_on_empty_feasible_set():
    u, _, _ = _case()
    impossible = ConstraintSet([Cardinality(4), type("Never", (), {
        "check": lambda self, x, u: False,
        "check_batch": lambda self, X, u: np.zeros(len(np.atleast_2d(X)), bool)})()])
    with pytest.raises(RuntimeError):
        rejection_sampler(u, impossible, 5, seed=0, max_cost=3000)


def test_dicke_sampler_feasible_and_weight_k():
    u, cs, _ = _case()
    r = dicke_sampler()(u, cs, 300, 2)
    assert r.samples.shape == (300, u.n) and cs.check_batch(r.samples, u).all()
    assert r.cost_unit == "shots" and (r.samples.sum(axis=1) == cs.cardinality).all()


# -------------------------------------------------------------- attribute
def test_decomposition_is_additive_and_consistent():
    u, cs, F = _case()
    r = attribute(u, cs, F[3], n_samples=1500, seed=0)
    assert np.isclose(r.active, r.constraint_effect + r.manager_effect)
    assert np.isclose(r.fund, _exact(u, F)[3])
    assert r.n_null == 1500 and r.fund_feasible and 0 <= r.percentile <= 100
    assert r.null_median_ci[0] <= r.null_median <= r.null_median_ci[1]
    assert r.manager_effect_ci[0] <= r.manager_effect <= r.manager_effect_ci[1]
    assert r.constraint_effect_ci[0] <= r.constraint_effect <= r.constraint_effect_ci[1]
    d = r.to_dict()
    assert "null" not in d and d["n_null"] == 1500 and "percentile" in r.summary()


def test_percentile_and_median_match_exact_enumeration():
    u, cs, F = _case()
    exact = _exact(u, F)
    for i in np.argsort(exact)[[5, len(F) // 2, -5]]:
        r = attribute(u, cs, F[i], n_samples=6000, seed=1)
        true_pct = 100 * (np.sum(exact < exact[i]) + 0.5) / len(F)
        assert abs(r.percentile - true_pct) < 5 * r.percentile_se + 100 / len(F)
        assert r.null_median_ci[0] - 0.01 <= np.median(exact) <= r.null_median_ci[1] + 0.01


def test_no_rules_means_no_constraint_effect():
    u, _, _ = _case()
    cs = ConstraintSet([Cardinality(4)])
    x = rejection_sampler(u, cs, 1, seed=0).samples[0]
    r = attribute(u, cs, x, n_samples=6000, seed=2)
    assert r.constraint_effect_ci[0] <= 0.0 <= r.constraint_effect_ci[1]


def test_planted_esg_drag_shows_up_as_constraint_effect():
    u, cs = p0_instance()
    z = (u.esg_score - u.esg_score.mean()) / u.esg_score.std()
    u.returns = synthetic_returns(u, 252, mu_shift=-0.25 * z, seed=5)
    F = enumerate_feasible(u, cs)
    r = attribute(u, cs, F[0], n_samples=3000, seed=0)
    assert r.constraint_effect_ci[1] < 0  # the MinESG floor costs return when ESG is penalised
    assert np.median(r.null) < np.median(r.unconstrained)


def test_dicke_and_rejection_samplers_agree():
    u, cs, F = _case()
    a = attribute(u, cs, F[10], n_samples=3000, seed=0)
    b = attribute(u, cs, F[10], n_samples=3000, seed=0, sampler=dicke_sampler())
    assert b.sampler == "dicke[aer_statevector]" and b.cost_unit == "shots"
    assert abs(a.percentile - b.percentile) < 5 * np.hypot(a.percentile_se, b.percentile_se) + 1
    assert abs(a.null_median - b.null_median) < (a.null_median_ci[1] - a.null_median_ci[0]) * 2


def test_baseline_sampler_plugs_in_with_fewer_samples():
    u, cs, F = _case()
    r = attribute(u, cs, F[0], n_samples=800, sampler=rejection_sample, seed=0)
    assert 0 < r.n_null < 800 and r.unconstrained.size == 800


def test_seed_reproducible():
    u, cs, F = _case()
    a, b = attribute(u, cs, F[1], n_samples=500, seed=7), attribute(u, cs, F[1], n_samples=500, seed=7)
    assert np.array_equal(a.null, b.null) and a.percentile == b.percentile
    assert not np.array_equal(a.null, attribute(u, cs, F[1], n_samples=500, seed=8).null)


def test_explicit_weights_fund_returns_and_benchmark():
    u, cs, F = _case()
    R = u.returns.to_numpy()
    w = F[2] / F[2].sum()
    a = attribute(u, cs, F[2], n_samples=400, seed=0)
    assert np.isclose(attribute(u, cs, w, n_samples=400, seed=0).fund, a.fund)
    realised = R @ w - 0.0005  # a fee drag the holdings cannot see
    c = attribute(u, cs, F[2], fund_returns=realised, n_samples=400, seed=0)
    assert c.fund < a.fund and np.array_equal(c.null, a.null)
    index = R.mean(axis=1)
    d = attribute(u, cs, F[2], benchmark_returns=index, n_samples=400, seed=0)
    assert np.isclose(d.benchmark, np.prod(1 + index) - 1) and d.benchmark_label == "supplied benchmark"
    assert np.isclose(d.active, d.constraint_effect + d.manager_effect)
    assert attribute(u, cs, F[2], metric="sharpe", n_samples=400, seed=0).metric == "sharpe"


def test_infeasible_fund_is_flagged_and_bad_inputs_raise():
    u, cs, F = _case()
    bad = np.zeros(u.n, np.uint8)
    bad[np.argsort(u.esg_score)[:4]] = 1  # lowest-ESG names: breaks the floor
    with pytest.warns(UserWarning):
        r = attribute(u, cs, bad, n_samples=300, seed=0)
    assert not r.fund_feasible and "WARNING" in r.summary()
    with pytest.raises(ValueError):
        attribute(u, cs, F[0][:-1], n_samples=10)
    with pytest.raises(ValueError):
        attribute(u, cs, F[0] * 0.3, n_samples=10)  # weights not summing to 1
    with pytest.raises(ValueError):
        attribute(u, cs, F[0], fund_returns=np.zeros(3), n_samples=10)
    with pytest.raises(ValueError):
        attribute(u, cs, F[0], n_samples=10, sampler=lambda u, cs, m, s: rejection_sample(
            u, ConstraintSet([Cardinality(4)]), 200, s))  # unfiltered sampler output
    u.returns = None
    with pytest.raises(ValueError):
        attribute(u, cs, F[0], n_samples=10)


def test_plot_writes_file(tmp_path):
    u, cs, F = _case()
    r = attribute(u, cs, F[0], n_samples=400, seed=0)
    import matplotlib.pyplot as plt
    for res in (r, attribute(u, cs, F[0], n_samples=400, seed=0, metric="sharpe")):
        fig = plot_attribution(res, tmp_path / f"{res.metric}.png", note="synthetic")
        plt.close(fig)
        assert (tmp_path / f"{res.metric}.png").stat().st_size > 10_000
