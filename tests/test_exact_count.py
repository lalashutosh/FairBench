"""Exact-counting DP (fairbench.qae.exact_count) vs brute-force enumeration."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from fairbench.constraints import ConstraintSet, CountBound, Exclusion, LinearThreshold
from fairbench.data import synthetic_returns
from fairbench.instances import island_family, p0_instance, scaled_family
from fairbench.qae.exact_count import count_dp, make_rules, percentile_dp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _combos(allowed, k):
    return np.array(list(itertools.combinations([int(a) for a in allowed], k)), dtype=np.int64)


def brute_quantised(u, cs, bins, mode, extra=None):
    """Enumerate allowed weight-k states; exact rules as-is, weighted rules on the SAME integer weights/T."""
    from fairbench.qae.exact_count import WEIGHTED
    k = cs.cardinality
    full = ConstraintSet(list(cs.constraints) + ([extra] if extra is not None else []))
    C = _combos(full.allowed_indices(u.n), k)
    X = np.zeros((len(C), u.n), np.int64)
    np.put_along_axis(X, C, 1, axis=1)
    ok = ConstraintSet([c for c in full.constraints if not isinstance(c, WEIGHTED)]).check_batch(X, u)
    rules = make_rules(u, cs, bins, mode, extra)
    for w, T in rules:
        ok &= X @ w >= T
    okF = ConstraintSet([c for c in full.constraints if not isinstance(c, WEIGHTED)]).check_batch(X, u)
    for w, T in rules[:-1] if extra is not None else rules:
        okF &= X @ w >= T
    return int(ok.sum()), int(okF.sum()), X, full


def _cases():
    u0, c0 = p0_instance()
    u1, c1 = island_family(14, seed=1)
    u2, c2 = island_family(16, seed=2)
    from demo_attribution import build_case
    u3, c3, _ = build_case(16, 4, 0, 1.0, 0.08, 0.7, 2)
    return [("p0", u0, c0), ("island14", u1, c1), ("island16", u2, c2), ("demo16", u3, c3)]


@pytest.mark.parametrize("name,u,cs", _cases())
@pytest.mark.parametrize("bins", [8, 24])
def test_dp_equals_bruteforce_with_linear(name, u, cs, bins):
    k = cs.cardinality
    r = synthetic_returns(u, 252, seed=3).to_numpy()
    g = np.prod(1 + r, axis=0)
    C = _combos(cs.allowed_indices(u.n), k)
    s = float(np.median(g[C].sum(1) / k - 1.0))
    lin = LinearThreshold(g, k * (1 + s), "<")
    for mode in ("subset", "mid", "superset"):
        res = count_dp(u, cs, lin, bins, mode)
        nG, nF, X, full = brute_quantised(u, cs, bins, mode, lin)
        assert res.count == nG
        assert res.count_wo_last == nF


@pytest.mark.parametrize("name,u,cs", _cases())
def test_bracket_contains_float_count(name, u, cs):
    k = cs.cardinality
    r = synthetic_returns(u, 252, seed=3).to_numpy()
    g = np.prod(1 + r, axis=0)
    C = _combos(cs.allowed_indices(u.n), k)
    s = float(np.quantile(g[C].sum(1) / k - 1.0, 0.7))
    lin = LinearThreshold(g, k * (1 + s), "<")
    X = np.zeros((len(C), u.n), np.int64)
    np.put_along_axis(X, C, 1, axis=1)
    okF = cs.check_batch(X, u)
    okG = okF & lin.check_batch(X, u)
    p_true = okG.sum() / okF.sum()
    for bins in (4, 8, 16, 32):
        (lo, hi, pt), rt, peak = percentile_dp(u, cs, lin, bins)
        assert lo - 1e-12 <= p_true <= hi + 1e-12
        assert lo - 1e-12 <= pt <= hi + 1e-12


def test_countbound_dimension():
    u, cs = island_family(14, seed=1)
    k = cs.cardinality
    cs2 = ConstraintSet(list(cs.constraints) + [CountBound([0, 1, 2, 3, 4, 5], 1, 3), CountBound([6, 7, 8, 9], 1, None),
                                                Exclusion([13])])
    res = count_dp(u, cs2, None, 16, "mid")
    nG, nF, X, full = brute_quantised(u, cs2, 16, "mid")
    assert res.count == nG


def test_fine_bins_two_rules():
    """p0 (one weighted rule + the fund rule): 1024 bins is affordable and still equals brute force."""
    u, cs = p0_instance()
    k = cs.cardinality
    g = np.prod(1 + synthetic_returns(u, 252, seed=3).to_numpy(), axis=0)
    C = _combos(cs.allowed_indices(u.n), k)
    lin = LinearThreshold(g, k * (1 + float(np.median(g[C].sum(1) / k - 1.0))), "<")
    for mode in ("subset", "mid", "superset"):
        res = count_dp(u, cs, lin, 1024, mode)
        nG, nF, _, _ = brute_quantised(u, cs, 1024, mode, lin)
        assert (res.count, res.count_wo_last) == (nG, nF)
    (lo, hi, pt), _, _ = percentile_dp(u, cs, lin, 1024)
    assert hi - lo < 0.05
