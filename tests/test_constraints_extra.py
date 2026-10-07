import itertools
import math

import numpy as np
import pandas as pd
import pytest

from fairbench.apps.attribution import rejection_sampler
from fairbench.baselines import enumerate_feasible
from fairbench.constraints import (AvgBound, CarbonCap, Cardinality, ConstraintSet, CountBound,
                                   Exclusion, MinESG, MinGroups, SectorCap, TrackingErrorCap,
                                   VolatilityCap)
from fairbench.data import load_universe, synthetic_universe
from fairbench.ft import feasibility_oracle
from fairbench.quantum.hamiltonian import penalty_operator


# ---------------------------------------------------------------- helpers
def tiny():
    """6 assets, hand-set fields. sectors X X Y Y Z Z; countries DE FR DE FR DE UK."""
    u = synthetic_universe(6, 3, seed=0)
    u.sector = ["X", "X", "Y", "Y", "Z", "Z"]
    u.esg_score = np.array([10, 20, 30, 40, 50, 60], dtype=float)
    u.carbon = np.array([60, 50, 40, 30, 20, 10], dtype=float)
    u.attributes = {"cap": np.array([1, 2, 3, 4, 5, 6], dtype=float)}
    u.categories = {"country": ["DE", "FR", "DE", "FR", "DE", "UK"]}
    return u


def all_subsets(n):
    """(2^n, n) uint8, every 0/1 vector, including the empty one (row 0)."""
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.uint8)


def ksubsets(n, k):
    combos = list(itertools.combinations(range(n), k))
    X = np.zeros((len(combos), n), dtype=np.uint8)
    np.put_along_axis(X, np.array(combos), 1, axis=1)
    return X


def rows_agree(c, u, X):
    """check_batch equals row-by-row check; returns the batch result."""
    got = c.check_batch(X, u)
    assert got.dtype == bool and got.shape == (len(X),)
    assert [c.check(x, u) for x in X] == got.tolist()
    return got


# --------------------------------------------------------------- CountBound
def test_countbound_min_only():
    u, X = tiny(), all_subsets(6)
    cnt = X[:, [0, 2, 4]].sum(axis=1)
    got = rows_agree(CountBound([0, 2, 4], min_count=2), u, X)
    assert np.array_equal(got, cnt >= 2) and 0 < got.sum() < len(X)


def test_countbound_max_only():
    u, X = tiny(), all_subsets(6)
    cnt = X[:, [1, 3]].sum(axis=1)
    got = rows_agree(CountBound([1, 3], max_count=1), u, X)
    assert np.array_equal(got, cnt <= 1) and 0 < got.sum() < len(X)
    assert rows_agree(CountBound([1, 3], 0, 0), u, X).tolist() == (cnt == 0).tolist()  # max 0 forbids the group


def test_countbound_min_and_max():
    u, X = tiny(), all_subsets(6)
    cnt = X[:, [0, 1, 2, 3]].sum(axis=1)
    c = CountBound([0, 1, 2, 3], 1, 2, label="front")
    got = rows_agree(c, u, X)
    assert np.array_equal(got, (cnt >= 1) & (cnt <= 2)) and c.label == "front"
    assert c.check(np.array([1, 1, 1, 0, 0, 0]), u) is False  # 3 held in the group > max
    assert c.check(np.array([0, 0, 0, 0, 1, 1]), u) is False  # 0 held < min
    assert c.check(np.array([1, 0, 0, 0, 1, 1]), u) is True   # 1 held


def test_countbound_boundaries_are_inclusive():
    u = tiny()
    c = CountBound([0, 1, 2], 1, 2)
    assert not c.check(np.array([0, 0, 0, 1, 1, 1]), u)  # 0 < min
    assert c.check(np.array([1, 0, 0, 1, 1, 1]), u)      # == min
    assert c.check(np.array([1, 1, 0, 0, 0, 0]), u)      # == max
    assert not c.check(np.array([1, 1, 1, 0, 0, 0]), u)  # > max


def test_countbound_empty_index_list():
    u, X = tiny(), all_subsets(6)
    assert rows_agree(CountBound([], min_count=0), u, X).all()
    assert rows_agree(CountBound([], 0, 3), u, X).all()
    assert not rows_agree(CountBound([], min_count=1), u, X).any()  # nothing can ever be held there
    assert CountBound([]).max_count is None and CountBound([]).min_count == 0


# ----------------------------------------------------------------- AvgBound
def avg_of(values, X):
    cnt = X.sum(axis=1)
    return np.where(cnt > 0, (X @ values) / np.maximum(cnt, 1), np.nan)


@pytest.mark.parametrize("attr", ["esg", "carbon", "cap"])
def test_avgbound_lower_only(attr):
    u, X = tiny(), all_subsets(6)
    vals = u.numeric(attr)
    lo = float(np.median(vals))
    got = rows_agree(AvgBound(attr, lower=lo), u, X)
    a = avg_of(vals, X)
    assert np.array_equal(got, (X.sum(axis=1) > 0) & (a >= lo)) and 0 < got.sum() < len(X) - 1


@pytest.mark.parametrize("attr", ["esg", "carbon", "cap"])
def test_avgbound_upper_only(attr):
    u, X = tiny(), all_subsets(6)
    vals = u.numeric(attr)
    hi = float(np.median(vals))
    got = rows_agree(AvgBound(attr, upper=hi), u, X)
    a = avg_of(vals, X)
    assert np.array_equal(got, (X.sum(axis=1) > 0) & (a <= hi)) and 0 < got.sum() < len(X) - 1


def test_avgbound_both_bounds_and_inclusive_edges():
    u, X = tiny(), all_subsets(6)
    got = rows_agree(AvgBound("cap", lower=2.0, upper=4.0), u, X)
    a = avg_of(u.numeric("cap"), X)
    assert np.array_equal(got, (X.sum(axis=1) > 0) & (a >= 2.0) & (a <= 4.0))
    assert AvgBound("cap", 2.0, 4.0).check(np.array([1, 0, 0, 0, 0, 0]), u) is False   # avg 1
    assert AvgBound("cap", 2.0, 4.0).check(np.array([0, 1, 0, 0, 0, 0]), u) is True    # avg 2 = lower
    assert AvgBound("cap", 2.0, 4.0).check(np.array([0, 0, 0, 1, 0, 0]), u) is True    # avg 4 = upper
    assert AvgBound("cap", 2.0, 4.0).check(np.array([0, 0, 0, 0, 1, 0]), u) is False   # avg 5
    assert AvgBound("cap", 2.0, 4.0).check(np.array([1, 0, 0, 0, 0, 1]), u) is True    # avg 3.5


def test_avgbound_matches_the_original_esg_and_carbon_classes():
    u, X = tiny(), all_subsets(6)
    assert np.array_equal(AvgBound("esg", lower=35.0).check_batch(X, u), MinESG(35.0).check_batch(X, u))
    assert np.array_equal(AvgBound("carbon", upper=35.0).check_batch(X, u), CarbonCap(35.0).check_batch(X, u))


def test_avgbound_zero_bound_is_a_bound():
    u = tiny()
    u.attributes = {"net": np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])}
    c = AvgBound("net", lower=0.0)  # 0.0 is not "no bound"
    assert c.lower == 0.0 and c.upper is None
    assert c.check(np.array([0, 0, 0, 0, 1, 1]), u) and not c.check(np.array([1, 1, 0, 0, 0, 0]), u)


def test_avgbound_empty_selection_is_infeasible():
    u = tiny()
    empty = np.zeros((1, 6), dtype=np.uint8)
    for c in (AvgBound("esg", lower=-1e9), AvgBound("esg", upper=1e9), AvgBound("cap", -1e9, 1e9)):
        assert not c.check(empty[0], u) and not c.check_batch(empty, u)[0]


def test_avgbound_needs_a_bound():
    with pytest.raises(ValueError):
        AvgBound("esg")
    with pytest.raises(ValueError):
        AvgBound("cap", None, None)


def test_avgbound_unknown_attribute_raises_keyerror_on_check():
    u = tiny()
    c = AvgBound("water", lower=1.0)  # construction is fine
    with pytest.raises(KeyError, match="water"):
        c.check(np.ones(6, dtype=np.uint8), u)
    with pytest.raises(KeyError):
        c.check_batch(all_subsets(6), u)
    u.attributes = None
    with pytest.raises(KeyError):
        AvgBound("cap", lower=1.0).check(np.ones(6, dtype=np.uint8), u)


# --------------------------------------------------------------- MinGroups
def brute_groups(labels, X):
    return np.array([len(set(np.asarray(labels)[x.astype(bool)])) for x in X])


@pytest.mark.parametrize("category,m", [("sector", 2), ("sector", 3), ("country", 2), ("country", 3)])
def test_mingroups_matches_brute_force_over_k_subsets(category, m):
    u = tiny()
    X = ksubsets(6, 3)
    labels = u.labels(category)
    got = rows_agree(MinGroups(category, m), u, X)
    assert np.array_equal(got, brute_groups(labels, X) >= m)
    if (category, m) != ("sector", 2):  # three sectors of two assets: any 3 names span at least 2
        assert 0 < got.sum() < len(X)


def test_mingroups_over_all_subsets_including_empty():
    u, X = tiny(), all_subsets(6)
    got = MinGroups("country", 1).check_batch(X, u)
    assert np.array_equal(got, brute_groups(u.labels("country"), X) >= 1)
    assert not got[0] and got[1:].all()  # the empty selection spans no group
    assert MinGroups("country", 4).check_batch(X, u).sum() == 0  # only 3 countries exist


def test_mingroups_hand_checked_cases():
    u = tiny()
    c = MinGroups("sector", 2)
    assert c.check(np.array([1, 1, 0, 0, 0, 0]), u) is False  # both X
    assert c.check(np.array([1, 0, 1, 0, 0, 0]), u) is True   # X and Y
    c = MinGroups("country", 3)
    assert c.check(np.array([1, 1, 0, 0, 0, 1]), u) is True   # DE FR UK
    assert c.check(np.array([1, 0, 1, 0, 1, 0]), u) is False  # DE only


def test_mingroups_unknown_category_raises_keyerror():
    u = tiny()
    with pytest.raises(KeyError, match="region"):
        MinGroups("region", 2).check(np.ones(6, dtype=np.uint8), u)
    with pytest.raises(KeyError):
        MinGroups("region", 2).check_batch(all_subsets(6), u)
    u.categories = None
    with pytest.raises(KeyError):
        MinGroups("country", 2).check(np.ones(6, dtype=np.uint8), u)


# ---------------------------------------------------- VolatilityCap / TE cap
def risk_universe():
    u = synthetic_universe(7, 2, seed=4)
    assert np.all(np.linalg.eigvalsh(u.cov) > 0)
    return u


def direct_vol(x, u):
    w = x / x.sum()
    return math.sqrt(w @ u.cov @ w)


def direct_te(x, u, b=None):
    d = x / x.sum() - (np.full(u.n, 1.0 / u.n) if b is None else np.asarray(b, dtype=float))
    return math.sqrt(d @ u.cov @ d)


def mid_cap(r):
    """A cap strictly between two consecutive distinct risk values, so no float edge cases."""
    s = np.unique(np.round(r, 12))
    return float((s[len(s) // 2 - 1] + s[len(s) // 2]) / 2)


def test_volatility_risk_equals_sqrt_w_cov_w():
    u, X = risk_universe(), ksubsets(7, 3)
    r = VolatilityCap(1.0).risk(X, u)
    assert r.shape == (len(X),)
    assert np.allclose(r, [direct_vol(x.astype(float), u) for x in X])
    assert np.isclose(VolatilityCap(1.0).risk(X[0], u)[0], r[0])  # a single 1-D selection works too
    x = np.zeros(7); x[2] = 1  # one asset: its own volatility
    assert np.isclose(VolatilityCap(1.0).risk(x, u)[0], math.sqrt(u.cov[2, 2]))


def test_volatility_cap_accepts_exactly_the_selections_at_or_below_the_cap():
    u, X = risk_universe(), ksubsets(7, 3)
    r = np.array([direct_vol(x.astype(float), u) for x in X])
    cap = mid_cap(r)
    got = rows_agree(VolatilityCap(cap), u, X)
    assert np.array_equal(got, r <= cap) and 0 < got.sum() < len(X)
    edge = VolatilityCap(1.0).risk(X, u)[5]  # a cap equal to one selection's risk accepts it
    c = VolatilityCap(edge)
    assert c.check(X[5], u) and not VolatilityCap(edge - 1e-9).check(X[5], u)


def test_tracking_error_risk_equals_sqrt_active_cov_active_default_benchmark():
    u, X = risk_universe(), ksubsets(7, 3)
    r = TrackingErrorCap(1.0).risk(X, u)
    assert np.allclose(r, [direct_te(x.astype(float), u) for x in X])
    assert np.all(r > 0)


def test_tracking_error_with_supplied_benchmark():
    u, X = risk_universe(), ksubsets(7, 3)
    b = np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05, 0.0])
    c = TrackingErrorCap(1.0, benchmark=b)
    r = c.risk(X, u)
    assert np.allclose(r, [direct_te(x.astype(float), u, b) for x in X])
    assert not np.allclose(r, TrackingErrorCap(1.0).risk(X, u))  # the benchmark matters
    cap = mid_cap(r)
    got = rows_agree(TrackingErrorCap(cap, benchmark=b), u, X)
    assert np.array_equal(got, r <= cap) and 0 < got.sum() < len(X)
    # a selection that IS the benchmark has no tracking error
    b2 = np.zeros(7); b2[[1, 3, 5]] = 1 / 3
    c2 = TrackingErrorCap(1e-9, benchmark=b2)
    assert np.isclose(c2.risk(X, u)[list(map(tuple, X.tolist())).index((0, 1, 0, 1, 0, 1, 0))], 0.0, atol=1e-12)
    assert c2.check_batch(X, u).sum() == 1


def test_tracking_error_cap_accepts_exactly_the_selections_at_or_below_the_cap():
    u, X = risk_universe(), ksubsets(7, 3)
    r = np.array([direct_te(x.astype(float), u) for x in X])
    cap = mid_cap(r)
    got = rows_agree(TrackingErrorCap(cap), u, X)
    assert np.array_equal(got, r <= cap) and 0 < got.sum() < len(X)


def test_full_universe_has_zero_tracking_error_against_the_default_benchmark():
    u = risk_universe()
    full = np.ones(7, dtype=np.uint8)
    c = TrackingErrorCap(1e-9)
    assert np.isclose(c.risk(full, u)[0], 0.0, atol=1e-12)
    assert c.check(full, u) and not c.check(np.array([1, 1, 1, 0, 0, 0, 0]), u)
    assert not VolatilityCap(1e-9).check(full, u)  # but its volatility is not zero
    assert np.isclose(VolatilityCap(1.0).risk(full, u)[0], direct_vol(np.ones(7), u))


def test_risk_caps_reject_the_empty_selection():
    u = risk_universe()
    empty = np.zeros((1, 7), dtype=np.uint8)
    for c in (VolatilityCap(1e9), TrackingErrorCap(1e9), TrackingErrorCap(1e9, benchmark=np.full(7, 1 / 7))):
        assert np.isnan(c.risk(empty, u)[0])
        assert not c.check(empty[0], u) and not c.check_batch(empty, u)[0]
    X = all_subsets(7)
    assert not VolatilityCap(1e9).check_batch(X, u)[0] and VolatilityCap(1e9).check_batch(X, u)[1:].all()


def test_risk_caps_use_equal_weights_on_the_held_assets():
    u = risk_universe()
    u.cov = np.diag(np.arange(1.0, 8.0))  # variances 1..7
    x = np.zeros(7, dtype=np.uint8); x[[0, 1]] = 1
    assert np.isclose(VolatilityCap(1.0).risk(x, u)[0], math.sqrt((1 + 2) / 4))  # w = 1/2 each
    full_te = TrackingErrorCap(1.0).risk(x, u)[0]
    d = np.full(7, -1 / 7); d[[0, 1]] += 0.5
    assert np.isclose(full_te, math.sqrt(d @ u.cov @ d))


# ------------------------------------------------ inside a ConstraintSet
def new_constraints():
    return [CountBound([0, 1, 2], 1, 2),
            AvgBound("cap", lower=2.5),
            MinGroups("country", 2),
            VolatilityCap(0.3),
            TrackingErrorCap(0.3)]


@pytest.mark.parametrize("idx", range(5))
def test_each_new_class_works_in_a_constraintset_with_cardinality(idx):
    u = tiny()
    if idx >= 3:  # pick a cap that splits the 3-subsets
        r = np.sort(VolatilityCap(1).risk(ksubsets(6, 3), u) if idx == 3 else TrackingErrorCap(1).risk(ksubsets(6, 3), u))
        cons = [VolatilityCap(float(r[10])) if idx == 3 else TrackingErrorCap(float(r[10]))]
    else:
        cons = [new_constraints()[idx]]
    cs = ConstraintSet([Cardinality(3), *cons])
    X = all_subsets(6)
    want = (X.sum(axis=1) == 3) & cons[0].check_batch(X, u)
    got = cs.check_batch(X, u)
    assert np.array_equal(got, want) and 0 < got.sum() < (X.sum(axis=1) == 3).sum()
    assert [cs.check(x, u) for x in X] == got.tolist()
    assert cs.cardinality == 3 and cs.non_cardinality() == cons


def test_constraintset_check_batch_is_the_and_of_the_parts():
    u = tiny()
    r = np.sort(VolatilityCap(1).risk(ksubsets(6, 3), u))
    parts = [Cardinality(3), CountBound([0, 1, 2], 1, 2), AvgBound("cap", lower=2.5, upper=5.0),
             MinGroups("country", 2), VolatilityCap(float(r[-4])), TrackingErrorCap(1.0),
             Exclusion([5]), SectorCap("X", 1), MinESG(20.0), CarbonCap(50.0)]
    X = all_subsets(6)
    cs = ConstraintSet(parts)
    want = np.ones(len(X), dtype=bool)
    for p in parts:
        want &= p.check_batch(X, u)
    got = cs.check_batch(X, u)
    assert np.array_equal(got, want) and got.sum() > 0
    assert [cs.check(x, u) for x in X] == got.tolist()
    # each part on its own is strictly weaker than the whole
    assert all(p.check_batch(X, u).sum() > got.sum() for p in parts)


def test_new_classes_do_not_exclude_assets_for_allowed_indices():
    cs = ConstraintSet([Cardinality(3), *new_constraints()])
    assert cs.allowed_indices(6).tolist() == [0, 1, 2, 3, 4, 5]
    cs = ConstraintSet([Cardinality(3), *new_constraints(), Exclusion([4])])
    assert cs.allowed_indices(6).tolist() == [0, 1, 2, 3, 5]


# --------------------------------------------------------------- Universe
def test_universe_numeric_and_labels_builtins():
    u = synthetic_universe(5, 2, seed=1)
    assert np.array_equal(u.numeric("esg"), u.esg_score) and u.numeric("esg").dtype == float
    assert np.array_equal(u.numeric("carbon"), u.carbon)
    assert u.labels("sector").tolist() == ["S0", "S1", "S0", "S1", "S0"]
    assert u.labels("sector").dtype.kind == "U"
    assert u.attributes is None and u.categories is None
    assert u.numeric_names == ["esg", "carbon"] and u.label_names == ["sector"]


def test_universe_unknown_names_give_none():
    u = synthetic_universe(5, 2, seed=1)
    for name in ("water", "country", "sector", "ticker", ""):
        assert u.numeric(name) is None
    for name in ("country", "esg", "carbon", "flag", ""):
        assert u.labels(name) is None
    u = tiny()
    assert u.numeric("sector") is None and u.numeric("country") is None
    assert u.labels("cap") is None and u.labels("water") is None


def test_universe_custom_attributes_and_categories():
    u = tiny()
    u.attributes = {"zeta": np.array([1, 2, 3, 4, 5, 6]), "alpha": [6, 5, 4, 3, 2, 1]}  # ints and lists are coerced
    u.categories = {"region": ["N", "N", "S", "S", "S", "S"], "country": ["a", "b", "c", "d", "e", "f"]}
    z = u.numeric("zeta")
    assert z.dtype == float and z.tolist() == [1, 2, 3, 4, 5, 6]
    assert u.numeric("alpha").tolist() == [6, 5, 4, 3, 2, 1]
    assert u.labels("region").tolist() == ["N", "N", "S", "S", "S", "S"]
    assert u.numeric_names == ["esg", "carbon", "alpha", "zeta"]  # built-ins first, then sorted
    assert u.label_names == ["sector", "country", "region"]
    u.attributes, u.categories = {}, {}
    assert u.numeric_names == ["esg", "carbon"] and u.label_names == ["sector"] and u.numeric("zeta") is None


# ------------------------------------------------------------ load_universe
def test_load_universe_reads_attr_and_cat_columns(tmp_path):
    df = pd.DataFrame(dict(ticker=["X", "Y", "Z"], mu=[0.1, 0.2, 0.3], sector=["a", "b", "a"],
                           esg_score=[50.0, 60.0, 70.0], carbon=[1.0, 2.0, 3.0],
                           attr_mcap=[10, 20, 30], attr_women=[31.5, 40.0, 25.0],
                           cat_country=["DE", "FR", "DE"], cat_tier=[1, 2, 1],
                           flag_coal=[0, 1, 0], attribute_x=[9, 9, 9], category=["p", "q", "r"]))
    df.to_csv(tmp_path / "u.csv", index=False)
    u = load_universe(tmp_path / "u.csv")
    assert set(u.attributes) == {"mcap", "women"}  # "attribute_x" and "category" do not match the prefixes
    assert set(u.categories) == {"country", "tier"}
    assert all(v.dtype == float and v.shape == (3,) for v in u.attributes.values())
    assert u.attributes["mcap"].tolist() == [10.0, 20.0, 30.0] and u.attributes["women"].tolist() == [31.5, 40.0, 25.0]
    assert u.categories["country"] == ["DE", "FR", "DE"]
    assert u.categories["tier"] == ["1", "2", "1"]  # labels are strings
    assert u.labels("tier").tolist() == ["1", "2", "1"] and u.numeric("women").tolist() == [31.5, 40.0, 25.0]
    assert set(u.flags) == {"coal"}
    assert u.numeric_names == ["esg", "carbon", "mcap", "women"] and u.label_names == ["sector", "country", "tier"]


def test_load_universe_without_attr_or_cat_columns_gives_none(tmp_path):
    pd.DataFrame(dict(ticker=["X", "Y"], mu=[0.1, 0.2], sector=["a", "b"], esg_score=[50.0, 60.0],
                      carbon=[1.0, 2.0], flag_coal=[0, 1])).to_csv(tmp_path / "u.csv", index=False)
    u = load_universe(tmp_path / "u.csv")
    assert u.attributes is None and u.categories is None and u.flags is not None
    only_attr = pd.read_csv(tmp_path / "u.csv").assign(attr_a=[1.0, 2.0])
    only_attr.to_csv(tmp_path / "v.csv", index=False)
    v = load_universe(tmp_path / "v.csv")
    assert v.categories is None and v.attributes["a"].tolist() == [1.0, 2.0]


# ---------------------------------- Hamiltonian / oracle refuse new classes
@pytest.mark.parametrize("make", [lambda: CountBound([0, 1], 0, 1), lambda: AvgBound("cap", lower=1.0),
                                  lambda: MinGroups("sector", 2), lambda: VolatilityCap(0.5),
                                  lambda: TrackingErrorCap(0.5)],
                         ids=["CountBound", "AvgBound", "MinGroups", "VolatilityCap", "TrackingErrorCap"])
def test_oracle_and_hamiltonian_raise_typeerror_for_new_classes(make):
    u = tiny()
    c = make()
    cs = ConstraintSet([Cardinality(3), SectorCap("X", 1), c])
    with pytest.raises(TypeError, match=type(c).__name__):
        feasibility_oracle(u, cs)
    with pytest.raises(TypeError, match=type(c).__name__):
        feasibility_oracle(u, cs, mode="bit")
    with pytest.raises(TypeError, match=type(c).__name__):
        penalty_operator(cs, u)
    # the original classes alone are still encodable
    ok = ConstraintSet([Cardinality(3), SectorCap("X", 1)])
    feasibility_oracle(u, ok)
    penalty_operator(ok, u)


# ------------------------------------------------------------- samplers
def test_rejection_sampler_with_countbound_and_avgbound_covers_the_feasible_set():
    u = synthetic_universe(10, 2, seed=2)
    u.attributes = {"cap": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)}
    cs = ConstraintSet([Cardinality(4), CountBound([0, 1, 2, 3, 4], 1, 2), AvgBound("cap", lower=4.0, upper=7.0)])
    F = enumerate_feasible(u, cs)
    assert 10 < len(F) < math.comb(10, 4)
    X = rejection_sampler(u, cs, 3000, seed=1).samples
    assert X.shape == (3000, 10) and X.dtype == np.uint8
    assert cs.check_batch(X, u).all() and (X.sum(axis=1) == 4).all()
    assert {tuple(x) for x in X} == {tuple(x) for x in F}  # every feasible selection shows up, none other
    assert np.array_equal(X, rejection_sampler(u, cs, 3000, seed=1).samples)


def test_rejection_sampler_with_mingroups_and_risk_caps():
    u = tiny()
    r = np.sort(TrackingErrorCap(1).risk(ksubsets(6, 3), u))
    cs = ConstraintSet([Cardinality(3), MinGroups("country", 3), TrackingErrorCap(float(r[-3])),
                        VolatilityCap(1.0), Exclusion([0])])
    F = enumerate_feasible(u, cs)
    assert len(F) > 0
    X = rejection_sampler(u, cs, 400, seed=0).samples
    assert cs.check_batch(X, u).all() and X[:, 0].sum() == 0
    assert {tuple(x) for x in X} == {tuple(x) for x in F}
