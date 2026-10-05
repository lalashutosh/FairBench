import numpy as np
import pytest

from fairbench.constraints import Cardinality, ConstraintSet, MinESG, SectorCap
from fairbench.data import synthetic_universe
from fairbench.metrics import tv_expected_uniform, tv_to_uniform
from fairbench.postprocess import (assign_weights, filter_feasible, postprocess,
                                   repair, repair_batch, violation)
from fairbench.baselines import enumerate_feasible

import itertools

N, K = 12, 4


def _instance():
    u = synthetic_universe(12, 3, seed=0)
    subs = np.array(list(itertools.combinations(range(N), K)))
    avg = u.esg_score[subs].mean(axis=1)
    thr = np.percentile(avg, 70)
    cs = ConstraintSet([Cardinality(K)] + [SectorCap(s, 2) for s in sorted(set(u.sector))] + [MinESG(thr)])
    return u, cs


def _rand_k(m, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((m, N), dtype=np.uint8)
    for r in range(m):
        X[r, rng.choice(N, K, replace=False)] = 1
    return X


def test_filter():
    u, cs = _instance()
    X = _rand_k(300, 1)
    S, mask = filter_feasible(X, cs, u)
    assert np.array_equal(mask, cs.check_batch(X, u)) and cs.check_batch(S, u).all() and len(S) == mask.sum()


def test_repair_feasible_and_k():
    u, cs = _instance()
    for x in _rand_k(200, 2):
        y, was = repair(x, cs, u, seed=0)
        assert cs.check(y, u) and y.sum() == K
        assert was == (not cs.check(x, u))


def test_repair_wrong_weight():
    u, cs = _instance()
    rng = np.random.default_rng(3)
    for w in (2, 6, 7):
        x = np.zeros(N, dtype=np.uint8)
        x[rng.choice(N, w, replace=False)] = 1
        y, _ = repair(x, cs, u, seed=0)
        assert cs.check(y, u) and y.sum() == K


def test_already_feasible_unchanged():
    u, cs = _instance()
    F = enumerate_feasible(u, cs)
    y, was = repair(F[0], cs, u, seed=0)
    assert not was and np.array_equal(y, F[0]) and violation(F[0], cs, u) == 0.0


def test_repair_deterministic():
    u, cs = _instance()
    X = _rand_k(100, 4)
    a, _, _ = repair_batch(X, cs, u, seed=7)
    b, _, _ = repair_batch(X, cs, u, seed=7)
    assert np.array_equal(a, b)


def test_violation_zero_iff_feasible():
    u, cs = _instance()
    X = _rand_k(300, 5)
    from fairbench.postprocess import violation_batch
    assert np.array_equal(violation_batch(X, cs, u) == 0, cs.check_batch(X, u))


def test_weights():
    u, cs = _instance()
    x = np.zeros(N, dtype=np.uint8); x[[0, 3, 5, 8]] = 1
    for s in ("equal", "inverse_vol", "esg_tilt"):
        w = assign_weights(x, s, u)
        assert w.shape == (N,) and np.isclose(w.sum(), 1) and (w[x == 0] == 0).all() and (w[x == 1] > 0).all()
    u.cov = np.diag([1.0, 4.0] + [1.0] * (N - 2))
    x = np.zeros(N, dtype=np.uint8); x[[0, 1]] = 1
    assert np.allclose(assign_weights(x, "inverse_vol", u)[[0, 1]], [2 / 3, 1 / 3])
    with pytest.raises(ValueError):
        assign_weights(x, "inverse_vol")


def test_postprocess_counts():
    u, cs = _instance()
    X = _rand_k(500, 6)
    f = postprocess(X, cs, u, "filter", seed=0)
    r = postprocess(X, cs, u, "repair", seed=0)
    assert f["n_in"] == r["n_in"] == 500 and f["n_feasible_raw"] == r["n_feasible_raw"]
    assert len(f["samples"]) == f["n_feasible_raw"]
    assert len(r["samples"]) == r["n_feasible_raw"] + r["n_repaired"]
    assert r["n_feasible_raw"] + r["n_repaired"] + r["n_failed"] == 500
    assert cs.check_batch(r["samples"], u).all()


def test_repair_bias_tv():
    """Repair distorts uniformity; filtering does not. Measured on seed 0 (see printed values)."""
    u, cs = _instance()
    F = enumerate_feasible(u, cs)
    X = _rand_k(20000, 11)
    f = postprocess(X, cs, u, "filter")["samples"]
    r = postprocess(X, cs, u, "repair", seed=0)["samples"]
    tv_f, tv_r = tv_to_uniform(f, F), tv_to_uniform(r, F)
    base = tv_expected_uniform(len(F), len(f), seed=0)
    print(f"|F|={len(F)} TV filter={tv_f:.4f} (finite-sample floor {base:.4f}) TV repair={tv_r:.4f}")
    assert tv_f < 1.5 * base
    assert tv_r > tv_f + 0.05
