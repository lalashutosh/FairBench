import math
from itertools import combinations

import numpy as np
import pytest

from fairbench.qae import (
    AEResult, IdealOracle, NoisyOracle, SubspaceOracle, canonical_ae, classical_mc,
    exp_schedule, iqae, linear_schedule, mlae, power_schedule, ratio_estimate,
)

A_VALUES = [0.003, 0.05, 0.3, 0.7]


def _slope(run, budgets, a, reps, rng):
    Q, E = [], []
    for b in budgets:
        rs = [run(b, rng) for _ in range(reps)]
        Q.append(np.median([r.oracle_queries for r in rs]))
        E.append(np.median([abs(r.estimate - a) for r in rs]))
    assert Q[-1] / Q[0] >= 100, "budgets must span >= 2 decades"
    return np.polyfit(np.log(Q), np.log(E), 1)[0]


# ------------------------------------------------------------------ 1. subspace oracle
def test_subspace_oracle_matches_sin2():
    rng = np.random.default_rng(0)
    n, k = 8, 3
    C = len(list(combinations(range(n), k)))
    mask = rng.random(C) < 0.3
    o = SubspaceOracle(mask)
    th = math.asin(math.sqrt(mask.mean()))
    ms = np.arange(11)
    np.testing.assert_allclose(o.prob_good(ms), np.sin((2 * ms + 1) * th) ** 2, atol=1e-12)
    np.testing.assert_allclose(IdealOracle(mask.mean()).prob_good(ms), o.prob_good(ms), atol=1e-12)


def test_noisy_oracle_model():
    o = NoisyOracle(0.2, gamma=0.01, p_mixed=0.4)
    for m in (0, 3, 50):
        f = math.exp(-0.01 * (2 * m + 1))
        exp = f * math.sin((2 * m + 1) * math.asin(math.sqrt(0.2))) ** 2 + (1 - f) * 0.4
        assert o.prob_good(m) == pytest.approx(exp)


# ------------------------------------------------------------------ 2. accuracy / coverage
@pytest.mark.parametrize("a", A_VALUES)
def test_estimators_accurate(a):
    rng = np.random.default_rng(int(a * 1e4))
    o = IdealOracle(a)
    r = classical_mc(o, 200_000, rng)
    assert abs(r.estimate - a) < 4 * math.sqrt(a * (1 - a) / 200_000) + 1e-4
    r = canonical_ae(o, 10, 20, rng)
    assert abs(r.estimate - a) <= 2 * math.pi * math.sqrt(a * (1 - a)) / 1024 + math.pi ** 2 / 1024 ** 2
    for meth in ("beta", "chernoff"):
        r = iqae(o, 1e-3, 0.05, rng, ci_method=meth)
        assert abs(r.estimate - a) < 2e-3
        assert r.ci[1] - r.ci[0] <= 2e-3 + 1e-9
    r = mlae(o, exp_schedule(9), 100, rng)
    assert abs(r.estimate - a) < 1e-3
    r = mlae(o, power_schedule(40, 0.5), 100, rng)
    assert abs(r.estimate - a) < 3e-3


@pytest.mark.parametrize("a", [0.003, 0.05, 0.3, 0.7])
def test_ci_coverage(a):
    rng = np.random.default_rng(7 + int(a * 1000))
    o = IdealOracle(a)
    reps, slack = 100, 0.06
    runs = {
        "mc": lambda: classical_mc(o, 2000, rng),
        "iqae_beta": lambda: iqae(o, 1e-3, 0.05, rng),
        "iqae_chernoff": lambda: iqae(o, 1e-3, 0.05, rng, ci_method="chernoff"),
        "mlae_fisher": lambda: mlae(o, exp_schedule(7), 100, rng),
    }
    for name, f in runs.items():
        cov = np.mean([lo <= a <= hi for lo, hi in (f().ci for _ in range(reps))])
        assert cov >= 0.95 - slack, (name, a, cov)


# ------------------------------------------------------------------ 3. scaling
def test_error_scaling_slopes():
    a, reps = 0.07, 30
    rng = np.random.default_rng(11)
    o = IdealOracle(a)
    s_mc = _slope(lambda b, g: classical_mc(o, b, g), [100, 1000, 10_000, 100_000], a, reps, rng)
    s_iq = _slope(lambda b, g: iqae(o, b, 0.05, g), [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], a, reps, rng)
    s_ml = _slope(lambda b, g: mlae(o, exp_schedule(b), 100, g), range(4, 12), a, reps, rng)
    assert -0.6 <= s_mc <= -0.4, s_mc
    assert -1.15 <= s_iq <= -0.8, s_iq
    assert -1.15 <= s_ml <= -0.8, s_ml


# ------------------------------------------------------------------ 4. noise
def test_noise_aware_mlae():
    a, g, pm = 0.07, 1e-3, 0.5
    rng = np.random.default_rng(3)
    o = NoisyOracle(a, g, pm)
    sched = exp_schedule(9)                     # depth up to 2*128+1 -> f ~ 0.77
    err = {}
    for shots in (100, 10_000):
        aw = [abs(mlae(o, sched, shots, rng, gamma=g, p_mixed=pm).estimate - a) for _ in range(20)]
        un = [abs(mlae(o, sched, shots, rng).estimate - a) for _ in range(20)]
        err[shots] = (np.median(aw), np.median(un))
    assert err[10_000][0] < err[100][0] / 5          # consistent: error shrinks
    assert err[10_000][1] > 0.5 * err[100][1]         # unaware: bias floor, no shrinkage
    assert err[10_000][1] > 20 * err[10_000][0]       # unaware is biased
    # noise-aware Fisher CI still covers
    cov = np.mean([lo <= a <= hi for lo, hi in
                   (mlae(o, sched, 500, rng, gamma=g, p_mixed=pm).ci for _ in range(100))])
    assert cov >= 0.89


# ------------------------------------------------------------------ 5. query accounting
def test_query_accounting_exact():
    rng = np.random.default_rng(0)
    o = IdealOracle(0.2)
    r = mlae(o, [0, 1, 3, 3, 8], [10, 20, 30, 5, 7], rng)
    assert r.oracle_queries == 0 * 10 + 1 * 20 + 3 * 30 + 3 * 5 + 8 * 7
    assert r.state_preps == 1 * 10 + 3 * 20 + 7 * 30 + 7 * 5 + 17 * 7
    assert r.shots == 72
    assert [(m, n) for m, n, _ in r.schedule] == [(0, 10), (1, 20), (3, 30), (3, 5), (8, 7)]
    r = classical_mc(o, 1234, rng)
    assert (r.oracle_queries, r.state_preps, r.shots) == (1234, 1234, 1234)
    r = canonical_ae(o, 5, 3, rng)
    assert r.oracle_queries == 3 * 31 and r.state_preps == 3 * 63
    r = iqae(o, 1e-3, 0.05, rng)
    assert r.oracle_queries == sum(m * n for m, n, _ in r.schedule)
    assert r.state_preps == sum((2 * m + 1) * n for m, n, _ in r.schedule)
    assert r.shots == sum(n for _, n, _ in r.schedule)


def test_schedules():
    assert exp_schedule(5) == [0, 1, 2, 4, 8]
    assert linear_schedule(4) == [0, 1, 2, 3]
    assert power_schedule(5, 0.5) == [0, 1, 1, 1, 2]       # k^(1/2)
    assert power_schedule(4, 1 / 3) == [0, 1, 2, 3]         # k^1


# ------------------------------------------------------------------ ratio
def test_ratio_estimate_subspace():
    rng = np.random.default_rng(5)
    C = 70                                       # C(8, 4)
    score = rng.random(C)
    feas = rng.random(C) < 0.5
    good = feas & (score < 0.4)
    true = good.sum() / feas.sum()
    rn = mlae(SubspaceOracle(good), exp_schedule(8), 200, rng)
    rd = mlae(SubspaceOracle(feas), exp_schedule(8), 200, rng)
    r = ratio_estimate(rn, rd)
    assert isinstance(r, AEResult)
    assert r.ci[0] <= true <= r.ci[1]
    assert abs(r.estimate - true) < 0.01
    assert r.oracle_queries == rn.oracle_queries + rd.oracle_queries
    assert r.state_preps == rn.state_preps + rd.state_preps


# ------------------------------------------------------------------ QA-F2 additions
def test_mlae_lr_default_coverage_small_amplitude():
    """Default LR CI: coverage >= 0.90 (nominal 0.95) incl. a=0.0055 where Fisher fails."""
    rng = np.random.default_rng(2024)
    reps = 250
    for a in (0.0055, 0.05, 0.3):
        o = IdealOracle(a)
        hit = 0
        for _ in range(reps):
            r = mlae(o, exp_schedule(10), 30, rng)
            assert r.extra["ci_method"] == "lr"
            z = 1.959963984540054
            assert r.extra["std"] == pytest.approx((r.ci[1] - r.ci[0]) / (2 * z))
            hit += r.ci[0] <= a <= r.ci[1]
        assert hit / reps >= 0.90, (a, hit / reps)
    r = mlae(IdealOracle(0.1), exp_schedule(6), 30, rng, ci="fisher")
    assert r.extra["ci_method"] == "fisher"


class _FirstCallZero:
    """Ideal oracle whose very first sample_hits returns 0 hits (a 1-in-1000 CI miss),
    forcing a later CI that is disjoint from the carried one."""
    def __init__(self, a):
        self._o = IdealOracle(a)
        self.a_true = a
        self.n = 0

    def sample_hits(self, m, shots, rng):
        self.n += 1
        return 0 if self.n == 1 else self._o.sample_hits(m, shots, rng)


def test_iqae_disjoint_interval_does_not_collapse():
    rng = np.random.default_rng(3)
    a = 0.07
    r = iqae(_FirstCallZero(a), 0.02, 0.05, rng, n_shots=100, max_rounds=500)
    assert r.extra["rounds"] < 500, "must not loop forever on an inconsistent interval"
    assert r.ci[1] - r.ci[0] > 0, "zero-width collapse"
    assert r.ci[1] - r.ci[0] <= 2 * 0.02 * 1.01 or r.extra["rounds"] > 0
    assert abs(r.estimate - a) < 0.05
    assert r.extra["restarts"] >= 1


def test_iqae_amplitude_tol_accounting_and_coverage():
    from fairbench.qae.estimators import iqae_amplitude_tol
    rng = np.random.default_rng(11)
    for a, eps_a in ((0.005, 0.002), (0.07, 0.01), (0.4, 0.01)):
        o = IdealOracle(a)
        reps, ok = 150, 0
        for _ in range(reps):
            r = iqae_amplitude_tol(o, eps_a, 0.05, rng, n_shots=100)
            ok += abs(r.estimate - a) <= eps_a
        assert ok / reps >= 0.93, (a, ok / reps)
        pq = r.extra["pilot_queries"]
        assert pq > 0 and pq < r.oracle_queries
        main_q = sum(m * n for m, n, _ in r.schedule)
        # result schedule is main run only; totals include pilot
        assert r.oracle_queries == main_q + pq
        assert r.shots == sum(n for _, n, _ in r.schedule) + r.extra["pilot_shots"]
        assert r.state_preps == (sum((2 * m + 1) * n for m, n, _ in r.schedule)
                                 + r.extra["pilot_state_preps"])
    r0 = iqae_amplitude_tol(IdealOracle(0.1), 0.01, 0.05, rng, pilot=False)
    assert r0.extra["pilot_queries"] == 0
