import numpy as np
import pytest

from fairbench.constraints import (Cardinality, CarbonCap, ConstraintSet,
                                   Exclusion, MinESG, SectorCap)
from fairbench.data import synthetic_universe, load_universe

U = synthetic_universe(12, 3, seed=1)
RNG = np.random.default_rng(42)
X = (RNG.random((400, 12)) < 0.4).astype(np.uint8)
X[0] = 0  # include empty selection


def ref_card(x, u, k): return x.sum() == k
def ref_sector(x, u, s, m): return sum(x[i] for i in range(len(x)) if u.sector[i] == s) <= m
def ref_excl(x, u, idx): return all(x[i] == 0 for i in idx)
def ref_esg(x, u, t):
    h = [i for i in range(len(x)) if x[i]]
    return bool(h) and np.mean([u.esg_score[i] for i in h]) >= t
def ref_carbon(x, u, t):
    h = [i for i in range(len(x)) if x[i]]
    return bool(h) and np.mean([u.carbon[i] for i in h]) <= t


CASES = [
    (Cardinality(4), lambda x: ref_card(x, U, 4)),
    (SectorCap("S1", 1), lambda x: ref_sector(x, U, "S1", 1)),
    (Exclusion([0, 5, 7]), lambda x: ref_excl(x, U, [0, 5, 7])),
    (MinESG(50.0), lambda x: ref_esg(x, U, 50.0)),
    (CarbonCap(250.0), lambda x: ref_carbon(x, U, 250.0)),
]


@pytest.mark.parametrize("c,ref", CASES)
def test_constraint_matches_bruteforce(c, ref):
    expected = np.array([ref(x) for x in X])
    assert expected.any() and not expected.all()
    np.testing.assert_array_equal(c.check_batch(X, U), expected)
    for x, e in zip(X[:60], expected[:60]):
        assert c.check(x, U) == e


def test_empty_selection_infeasible():
    z = np.zeros(12, dtype=np.uint8)
    assert not MinESG(0.0).check(z, U)
    assert not CarbonCap(1e9).check(z, U)


def test_constraint_set():
    cs = ConstraintSet([c for c, _ in CASES])
    expected = np.array([all(r(x) for _, r in CASES) for x in X])
    np.testing.assert_array_equal(cs.check_batch(X, U), expected)
    assert all(cs.check(x, U) == e for x, e in zip(X, expected))
    assert cs.cardinality == 4
    assert ConstraintSet([MinESG(1)]).cardinality is None


def test_synthetic_deterministic():
    a, b = synthetic_universe(10, 3, 5), synthetic_universe(10, 3, 5)
    np.testing.assert_array_equal(a.mu, b.mu)
    np.testing.assert_array_equal(a.cov, b.cov)
    assert a.sector == b.sector == [f"S{i % 3}" for i in range(10)]


def test_load_universe(tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("ticker,mu,sector,esg_score,carbon\nA,0.1,X,60,100\nB,0.05,Y,40,200\n")
    u = load_universe(p)
    assert u.tickers == ["A", "B"] and u.sector == ["X", "Y"]
    np.testing.assert_allclose(u.cov, np.eye(2) * 0.04)
