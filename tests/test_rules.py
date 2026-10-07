import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fairbench.apps.attribution import dicke_sampler, rejection_sampler
from fairbench.baselines import enumerate_feasible
from fairbench.constraints import (AvgBound, CarbonCap, Cardinality, ConstraintSet, CountBound,
                                   Exclusion, MinESG, MinGroups, SectorCap, TrackingErrorCap,
                                   VolatilityCap)
from fairbench.data import load_universe, synthetic_universe
from fairbench.instances import named_universe
from fairbench.mandate import unverified_sources
from fairbench.metrics import tv_expected_uniform, tv_to_uniform
from fairbench.rules import (RULE_FIELDS, SPEC_SCHEMA, CompiledRules, compile_spec,
                             feasible_fraction, load_spec, save_spec, validate_spec)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_SPEC = EXAMPLES / "mandate_example.rules.json"


# ---------------------------------------------------------------- helpers
def small():
    """12 assets A000..A011 in three blocks of four; values chosen to be checkable by eye.
    index: 0-3 Tech, 4-7 Energy, 8-11 Health. universe mean ESG 40, mean carbon 1225/12.
    country DE {0,1,4,8,11}, FR {2,5,6,10}, UK {3,7,9}; attribute cap (mean 6.5);
    flags tobacco {1,6}, weapons {9}, green {0,2,3,8,10}, unused {}."""
    u = synthetic_universe(12, 3, seed=0)
    u.sector = ["Tech"] * 4 + ["Energy"] * 4 + ["Health"] * 4
    u.esg_score = np.array([10, 20, 30, 40, 5, 15, 25, 35, 60, 70, 80, 90], dtype=float)
    u.carbon = np.array([20, 30, 40, 50, 400, 300, 200, 100, 10, 15, 25, 35], dtype=float)

    def mask(*idx):
        m = np.zeros(12, dtype=bool)
        m[list(idx)] = True
        return m

    u.flags = {"tobacco": mask(1, 6), "weapons": mask(9), "unused": mask(), "green": mask(0, 2, 3, 8, 10)}
    u.categories = {"country": ["DE", "DE", "FR", "UK", "DE", "FR", "FR", "UK", "DE", "UK", "FR", "DE"]}
    u.attributes = {"cap": np.array([5, 1, 8, 2, 9, 3, 7, 4, 6, 10, 12, 11], dtype=float)}
    return u


def R(kind, **fields):
    return {"kind": kind, "source": "", "note": "", **fields}


def S(*rules):
    return dict(fund_name="Test Fund", rules=list(rules))


HOLD4 = R("holdings", min=4, max=4)

VALID = {
    "holdings": R("holdings", min=4, max=6),
    "position_cap": R("position_cap", max_weight=0.3),
    "sector_cap": R("sector_cap", sector="Tech", limit=2, unit="names"),
    "exclude": R("exclude", by="flag", values=["tobacco"]),
    "require": R("require", by="country", values=["DE"]),
    "exclude_threshold": R("exclude_threshold", metric="esg", side="below", value=20.0,
                           basis="absolute", scope="universe"),
    "group_limit": R("group_limit", by="sector", values=["Tech"], bound="max", limit=0.3, unit="weight"),
    "threshold_share": R("threshold_share", metric="esg", side="below", value=30.0, basis="absolute",
                         scope="universe", bound="min", limit=0.5, unit="weight"),
    "portfolio_average": R("portfolio_average", metric="esg", bound="min", value=1.0,
                           basis="relative_to_mean"),
    "min_groups": R("min_groups", category="sector", min=2),
    "risk_limit": R("risk_limit", measure="volatility", max=0.2),
    "unmapped": R("unmapped", reason="needs data we lack"),
}


def one(rule, k=4, u=None):
    """Compile a single rule (plus an explicit k) and return (CompiledRule, CompiledRules)."""
    cr = compile_spec(S(rule), u or small(), k=k)
    return cr.rules[0], cr


def excluded_of(compiled_rule):
    (c,) = compiled_rule.constraints
    assert isinstance(c, Exclusion)
    return sorted(c.indices.tolist())


def n_cardinality(cs):
    return sum(isinstance(c, Cardinality) for c in cs.constraints)


# ------------------------------------------------------------------ schema
def test_schema_objects_are_closed_and_fully_required():
    assert SPEC_SCHEMA["additionalProperties"] is False
    assert set(SPEC_SCHEMA["required"]) == set(SPEC_SCHEMA["properties"]) == {"fund_name", "rules"}
    variants = SPEC_SCHEMA["properties"]["rules"]["items"]["anyOf"]
    assert len(variants) == len(RULE_FIELDS)
    kinds = [v["properties"]["kind"]["const"] for v in variants]
    assert kinds == list(RULE_FIELDS)
    for v in variants:
        kind = v["properties"]["kind"]["const"]
        assert v["type"] == "object" and v["additionalProperties"] is False
        assert set(v["required"]) == set(v["properties"])
        assert set(v["properties"]) == {"kind", "source", "note"} | set(RULE_FIELDS[kind])


def test_schema_field_types_follow_rule_fields():
    by_kind = {v["properties"]["kind"]["const"]: v["properties"]
               for v in SPEC_SCHEMA["properties"]["rules"]["items"]["anyOf"]}
    assert by_kind["holdings"]["min"] == {"type": "integer"}
    assert by_kind["position_cap"]["max_weight"] == {"type": "number"}
    assert by_kind["sector_cap"]["sector"] == {"type": "string"}
    assert by_kind["exclude"]["values"] == {"type": "array", "items": {"type": "string"}}
    for kind in ("exclude", "require", "group_limit"):  # groupings are resolved against the universe
        assert by_kind[kind]["by"] == {"type": "string"} and by_kind[kind]["values"]["type"] == "array"
    for kind in ("exclude_threshold", "threshold_share", "portfolio_average"):  # so are numeric fields
        assert by_kind[kind]["metric"] == {"type": "string"}
    assert by_kind["min_groups"]["category"] == {"type": "string"} and by_kind["min_groups"]["min"] == {"type": "integer"}
    assert by_kind["portfolio_average"]["basis"]["enum"] == ["absolute", "percentile", "relative_to_mean"]
    assert by_kind["portfolio_average"]["bound"]["enum"] == ["min", "max"]
    assert by_kind["risk_limit"]["measure"]["enum"] == ["volatility", "tracking_error"]
    assert by_kind["risk_limit"]["max"] == {"type": "number"}
    for kind in ("group_limit", "threshold_share"):
        assert by_kind[kind]["bound"]["enum"] == ["min", "max"] and by_kind[kind]["unit"]["enum"] == ["names", "weight"]
        assert by_kind[kind]["limit"] == {"type": "number"}
    assert by_kind["threshold_share"]["side"]["enum"] == ["below", "above"]
    assert by_kind["threshold_share"]["scope"]["enum"] == ["universe", "sector"]
    assert by_kind["unmapped"]["reason"] == {"type": "string"}


# ---------------------------------------------------------------- validate
def test_validate_accepts_example_and_every_kind():
    spec = load_spec(EXAMPLE_SPEC)
    assert validate_spec(spec) is spec
    assert {r["kind"] for r in spec["rules"]} == set(RULE_FIELDS) - {"threshold_share"}
    assert set(VALID) == set(RULE_FIELDS)  # one valid sample of every kind, threshold_share included
    for rule in VALID.values():
        validate_spec(S(rule))
    validate_spec(S(*VALID.values()))
    validate_spec(S(R("position_cap", max_weight=1)))  # an int is a valid number
    validate_spec({"rules": []})  # an empty rule list is a valid spec


@pytest.mark.parametrize("kind", list(VALID))
def test_validate_rejects_missing_field_and_names_the_rule(kind):
    for name in [f for f in VALID[kind] if f != "kind"]:
        rule = {k: v for k, v in VALID[kind].items() if k != name}
        with pytest.raises(ValueError, match=rf"rule 1 \({kind}\): missing '{name}'"):
            validate_spec(S(VALID["unmapped"], rule))


def test_validate_rejects_unknown_kind():
    with pytest.raises(ValueError, match=r"rule 1: unknown kind 'frobnicate'"):
        validate_spec(S(VALID["holdings"], R("frobnicate", x=1)))
    with pytest.raises(ValueError, match="unknown kind"):
        validate_spec(S("not a dict"))
    with pytest.raises(ValueError, match="unknown kind"):
        validate_spec(S({"source": "", "note": ""}))  # no kind at all


@pytest.mark.parametrize("kind", [["holdings"], {"a": 1}, 3, None])
def test_validate_reports_non_string_kind_as_value_error(kind):
    with pytest.raises(ValueError, match="unknown kind"):
        validate_spec(S({"kind": kind, "source": "", "note": ""}))


@pytest.mark.parametrize("kind,name,bad", [
    ("position_cap", "max_weight", "0.07"),       # string where a number is required
    ("position_cap", "max_weight", True),         # bool is not a number
    ("position_cap", "max_weight", float("nan")),
    ("position_cap", "max_weight", None),
    ("holdings", "min", True),                    # bool where an int is required
    ("holdings", "max", 18.5),                    # float where an int is required
    ("holdings", "min", "18"),
    ("sector_cap", "sector", 3),
    ("exclude", "values", "tobacco"),             # a string is not a list of strings
    ("exclude", "values", ["tobacco", 1]),
    ("unmapped", "reason", 5),
    ("holdings", "source", 5),
    ("holdings", "note", None),
])
def test_validate_rejects_wrong_types(kind, name, bad):
    rule = dict(VALID[kind], **{name: bad})
    with pytest.raises(ValueError, match=rf"rule 0 \({kind}\): bad value for '{name}'"):
        validate_spec(S(rule))


@pytest.mark.parametrize("kind,name,bad", [
    ("sector_cap", "unit", "pct"),
    ("group_limit", "bound", "exact"),
    ("group_limit", "unit", "pct"),
    ("threshold_share", "bound", "exact"),
    ("threshold_share", "unit", "pct"),
    ("threshold_share", "side", "left"),
    ("threshold_share", "scope", "region"),
    ("risk_limit", "measure", "beta"),
    ("exclude_threshold", "side", "left"),
    ("exclude_threshold", "basis", "zscore"),
    ("exclude_threshold", "scope", "region"),
    ("portfolio_average", "bound", "equal"),
    ("portfolio_average", "basis", "zscore"),
])
def test_validate_rejects_bad_enum(kind, name, bad):
    rule = dict(VALID[kind], **{name: bad})
    with pytest.raises(ValueError, match=rf"rule 0 \({kind}\): bad value for '{name}'"):
        validate_spec(S(rule))


@pytest.mark.parametrize("kind,name,value", [
    ("exclude", "by", "country"), ("exclude", "by", "anything at all"), ("require", "by", ""),
    ("group_limit", "by", "region"), ("exclude_threshold", "metric", "water"),
    ("threshold_share", "metric", "market_cap_eur_bn"), ("portfolio_average", "metric", "water"),
    ("min_groups", "category", "continent"),
])
def test_validate_accepts_any_string_for_by_metric_and_category(kind, name, value):
    # these are resolved against the universe at compile time, not validated against a fixed list
    validate_spec(S(dict(VALID[kind], **{name: value})))


def test_validate_rejects_extra_field_and_lists_every_problem():
    with pytest.raises(ValueError, match=r"rule 0 \(holdings\): unexpected fields \['extra'\]"):
        validate_spec(S(dict(VALID["holdings"], extra=1)))
    bad = S(dict(VALID["holdings"], min="x"), R("nope"), dict(VALID["position_cap"], oops=1))
    with pytest.raises(ValueError) as e:
        validate_spec(bad)
    msg = str(e.value)
    assert "rule 0 (holdings)" in msg and "rule 1: unknown kind" in msg and "rule 2 (position_cap)" in msg


@pytest.mark.parametrize("spec", [{}, {"fund_name": "x"}, {"rules": "holdings"}, {"rules": None},
                                  [], "spec", None])
def test_validate_rejects_spec_without_rules_list(spec):
    with pytest.raises(ValueError, match="'rules' list"):
        validate_spec(spec)


@pytest.mark.parametrize("rule, msg", [
    (dict(kind="holdings", min=0, max=5), "min <= max"),
    (dict(kind="position_cap", max_weight=7), "fraction"),            # 7 written for 7%
    (dict(kind="sector_cap", sector="*", limit=15, unit="weight"), "fraction"),
    (dict(kind="sector_cap", sector="*", limit=-1, unit="names"), ">= 0"),
    (dict(kind="exclude_threshold", metric="esg", side="below", value=120, basis="percentile",
          scope="universe"), "percentile"),
    (dict(kind="portfolio_average", metric="carbon", bound="max", value=-0.3,
          basis="relative_to_mean"), "positive"),
])
def test_validate_rejects_values_out_of_range(rule, msg):
    with pytest.raises(ValueError, match=msg):
        validate_spec({"rules": [dict(source="", note="", **rule)]})


@pytest.mark.parametrize("rule, msg", [
    (dict(kind="risk_limit", measure="volatility", max=6), "annual fraction"),         # 6 written for 6%
    (dict(kind="risk_limit", measure="tracking_error", max=6.0), "annual fraction"),
    (dict(kind="risk_limit", measure="volatility", max=0), "annual fraction"),
    (dict(kind="risk_limit", measure="volatility", max=-0.05), "annual fraction"),
    (dict(kind="risk_limit", measure="volatility", max=1.01), "annual fraction"),
    (dict(kind="min_groups", category="sector", min=0), "min must be >= 1"),
    (dict(kind="min_groups", category="sector", min=-3), "min must be >= 1"),
    (dict(kind="group_limit", by="sector", values=["Tech"], bound="max", limit=15, unit="weight"), "fraction"),
    (dict(kind="group_limit", by="sector", values=["Tech"], bound="min", limit=1.2, unit="weight"), "fraction"),
    (dict(kind="group_limit", by="sector", values=["Tech"], bound="max", limit=-0.1, unit="weight"), ">= 0"),
    (dict(kind="group_limit", by="sector", values=["Tech"], bound="max", limit=-1, unit="names"), ">= 0"),
    (dict(kind="threshold_share", metric="esg", side="below", value=30.0, basis="absolute", scope="universe",
          bound="min", limit=50, unit="weight"), "fraction"),
    (dict(kind="threshold_share", metric="esg", side="below", value=30.0, basis="absolute", scope="universe",
          bound="min", limit=-0.5, unit="weight"), ">= 0"),
    (dict(kind="threshold_share", metric="esg", side="below", value=30.0, basis="absolute", scope="universe",
          bound="max", limit=-2, unit="names"), ">= 0"),
    (dict(kind="threshold_share", metric="esg", side="below", value=120, basis="percentile", scope="universe",
          bound="min", limit=0.5, unit="weight"), "percentile"),
    (dict(kind="threshold_share", metric="esg", side="below", value=-5, basis="percentile", scope="sector",
          bound="min", limit=0.5, unit="weight"), "percentile"),
    (dict(kind="threshold_share", metric="esg", side="above", value=0, basis="relative_to_mean", scope="universe",
          bound="min", limit=0.5, unit="weight"), "positive"),
    (dict(kind="portfolio_average", metric="esg", bound="min", value=101, basis="percentile"), "percentile"),
])
def test_validate_rejects_values_out_of_range_for_the_new_kinds(rule, msg):
    with pytest.raises(ValueError, match=msg):
        validate_spec({"rules": [dict(source="", note="", **rule)]})


def test_validate_accepts_values_at_the_edge_of_the_ranges():
    shares = dict(by="sector", values=["Tech"], bound="max", unit="weight")
    for limit in (0, 0.0, 1, 1.0):
        validate_spec(S(R("group_limit", limit=limit, **shares)))
    validate_spec(S(R("group_limit", **dict(shares, unit="names"), limit=3)))  # a count may exceed 1
    for value in (0, 100):
        validate_spec(S(dict(VALID["threshold_share"], basis="percentile", value=value)))
    for mx in (0.0001, 1, 1.0):
        validate_spec(S(R("risk_limit", measure="tracking_error", max=mx)))
    validate_spec(S(R("min_groups", category="sector", min=1)))


def test_sector_names_tolerate_surrounding_whitespace():
    cr = compile_spec(S(R("sector_cap", sector=" tech ", limit=1, unit="names")), small(), k=4)
    assert cr.rules[0].status == "mapped" and cr.rules[0].constraints[0].sector == "Tech"


def test_validate_rejects_non_string_fund_name():
    with pytest.raises(ValueError, match="fund_name"):
        validate_spec({"fund_name": 3, "rules": []})


def test_save_load_round_trip(tmp_path):
    spec = load_spec(EXAMPLE_SPEC)
    p = tmp_path / "out.json"
    save_spec(spec, p)
    assert load_spec(p) == spec == json.loads(p.read_text())
    assert p.read_text().endswith("\n")
    save_spec(spec, str(tmp_path / "again.json"))  # str paths work too
    with pytest.raises(ValueError):
        save_spec({"rules": [R("nope")]}, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()
    (tmp_path / "broken.json").write_text(json.dumps({"rules": [R("nope")]}))
    with pytest.raises(ValueError):
        load_spec(tmp_path / "broken.json")


# ----------------------------------------------------------------- holdings
def test_holdings_min_equals_max_gives_k_without_warning():
    cr = compile_spec(S(R("holdings", min=5, max=5)), small())
    assert cr.k == 5 and cr.warnings == []
    assert cr.constraint_set.cardinality == 5
    assert cr.rules[0].status == "mapped" and isinstance(cr.rules[0].constraints[0], Cardinality)


def test_holdings_range_without_k_takes_midpoint_with_warning():
    cr = compile_spec(S(R("holdings", min=4, max=8)), small())
    assert cr.k == 6 and cr.constraint_set.cardinality == 6
    assert len(cr.warnings) == 1 and "4-8" in cr.warnings[0] and "k=6" in cr.warnings[0]
    assert compile_spec(S(R("holdings", min=3, max=6)), small()).k == 4  # floor of the midpoint


def test_holdings_explicit_k_inside_range_is_silent_and_outside_warns():
    cr = compile_spec(S(R("holdings", min=4, max=8)), small(), k=7)
    assert cr.k == 7 and cr.warnings == []
    for k in (3, 9):
        cr = compile_spec(S(R("holdings", min=4, max=8)), small(), k=k)
        assert cr.k == k and cr.constraint_set.cardinality == k
        assert len(cr.warnings) == 1 and f"k={k}" in cr.warnings[0] and "4-8" in cr.warnings[0]


def test_several_holdings_rules_intersect_and_contradiction_raises():
    cr = compile_spec(S(R("holdings", min=3, max=8), R("holdings", min=5, max=10)), small())
    assert cr.k == 6 and "5-8" in cr.warnings[0]
    assert n_cardinality(cr.constraint_set) == 1  # two holdings rules, still one Cardinality
    with pytest.raises(ValueError, match="contradict"):
        compile_spec(S(R("holdings", min=3, max=4), R("holdings", min=6, max=7)), small())
    with pytest.raises(ValueError, match="min <= max"):
        compile_spec(S(R("holdings", min=6, max=4)), small(), k=5)


def test_no_holdings_rule_needs_k():
    with pytest.raises(ValueError, match="pass k"):
        compile_spec(S(VALID["position_cap"]), small())
    cr = compile_spec(S(), small(), k=5)
    assert cr.k == 5 and cr.rules == [] and cr.warnings == []
    assert [type(c) for c in cr.constraint_set.constraints] == [Cardinality]
    assert n_cardinality(compile_spec(S(VALID["position_cap"], VALID["exclude"]), small(), k=5).constraint_set) == 1


@pytest.mark.parametrize("k", [0, 12, 13, -1])
def test_k_must_be_between_zero_and_n(k):
    with pytest.raises(ValueError, match="0 < k < n"):
        compile_spec(S(), small(), k=k)


# ------------------------------------------------------------ position cap
def test_position_cap_trivial_when_equal_weight_fits():
    for cap in (0.25, 0.3, 1.0):  # k=4: 1/k = 25%, boundary included
        r, cr = one(R("position_cap", max_weight=cap))
        assert r.status == "trivial" and r.constraints == [] and cr.warnings == []
        assert len(cr.constraint_set.constraints) == 1


def test_position_cap_unmapped_with_warning_when_too_tight():
    r, cr = one(R("position_cap", max_weight=0.2))
    assert r.status == "unmapped" and r.constraints == []
    assert len(cr.warnings) == 1 and "20.0%" in cr.warnings[0] and "k=4" in cr.warnings[0]
    assert cr.unmapped == [r]


# -------------------------------------------------------------- sector cap
def test_sector_cap_names():
    r, cr = one(R("sector_cap", sector="Tech", limit=2, unit="names"))
    assert r.status == "mapped" and len(r.constraints) == 1
    c = r.constraints[0]
    assert isinstance(c, SectorCap) and (c.sector, c.max_count) == ("Tech", 2)
    assert cr.constraint_set.constraints[1:] == r.constraints


def test_sector_cap_weight_is_floor_of_limit_times_k():
    for limit, k, cap in [(0.5, 5, 2), (0.5, 4, 2), (0.3, 7, 2), (0.34, 3, 1), (0.99, 10, 9), (1.0, 4, 4)]:
        r, _ = one(R("sector_cap", sector="Energy", limit=limit, unit="weight"), k=k)
        (c,) = r.constraints
        assert (c.sector, c.max_count) == ("Energy", cap) and str(cap) in r.detail


def test_sector_cap_star_gives_one_cap_per_sector():
    for star in ("*", "each", "ANY", "All", " "):
        r, cr = one(R("sector_cap", sector=star, limit=0.5, unit="weight"))
        assert r.status == "mapped"
        assert [(c.sector, c.max_count) for c in r.constraints] == [("Energy", 2), ("Health", 2), ("Tech", 2)]
        assert len(cr.constraint_set.constraints) == 4


def test_sector_cap_named_sector_is_case_insensitive():
    for name in ("tech", "TECH", "tEcH"):
        r, _ = one(R("sector_cap", sector=name, limit=1, unit="names"))
        assert [c.sector for c in r.constraints] == ["Tech"]


def test_sector_cap_unknown_sector_is_unmapped():
    r, cr = one(R("sector_cap", sector="Mining", limit=1, unit="names"))
    assert r.status == "unmapped" and r.constraints == [] and "Mining" in r.detail
    assert len(cr.constraint_set.constraints) == 1


# ----------------------------------------------------------------- exclude
def test_exclude_by_sector_ticker_and_flag():
    r, _ = one(R("exclude", by="sector", values=["energy"]))
    assert r.status == "mapped" and excluded_of(r) == [4, 5, 6, 7]
    r, _ = one(R("exclude", by="sector", values=["Tech", "Health"]))
    assert excluded_of(r) == [0, 1, 2, 3, 8, 9, 10, 11]
    r, _ = one(R("exclude", by="ticker", values=["a003", "A009"]))
    assert r.status == "mapped" and excluded_of(r) == [3, 9]
    r, cr = one(R("exclude", by="flag", values=["tobacco"]))
    assert excluded_of(r) == [1, 6] and cr.warnings == []
    r, _ = one(R("exclude", by="flag", values=["Tobacco", "WEAPONS"]))  # union of flags
    assert excluded_of(r) == [1, 6, 9]


def test_exclude_unknown_values_warn_but_known_ones_still_apply():
    for by, values, known, unknown in [("sector", ["Tech", "Mining"], [0, 1, 2, 3], "Mining"),
                                       ("ticker", ["A001", "ZZZ9"], [1], "ZZZ9"),
                                       ("flag", ["tobacco", "gambling"], [1, 6], "gambling")]:
        r, cr = one(R("exclude", by=by, values=values))
        assert r.status == "mapped" and excluded_of(r) == known
        assert len(cr.warnings) == 1 and unknown in cr.warnings[0] and by in cr.warnings[0]


def test_exclude_nothing_matched_is_unmapped():
    for by, values in [("sector", ["Mining"]), ("ticker", ["ZZZ"]), ("flag", ["gambling"]), ("flag", [])]:
        r, cr = one(R("exclude", by=by, values=values))
        assert r.status == "unmapped" and r.constraints == []
        assert len(cr.constraint_set.constraints) == 1
        assert len(cr.warnings) == (1 if values else 0)
    r, _ = one(R("exclude", by="flag", values=["gambling"]), u=synthetic_universe(12, 3, seed=0))  # flags is None
    assert r.status == "unmapped"


def test_exclude_flag_that_marks_no_asset_is_trivial():
    r, cr = one(R("exclude", by="flag", values=["unused"]))
    assert r.status == "trivial" and r.constraints == [] and cr.warnings == []
    assert len(cr.constraint_set.constraints) == 1


# ------------------------------------------------------ exclude_threshold
def _expected_excluded(u, metric, side, value, basis, scope):
    vals = u.numeric(metric)
    sec = np.array(u.sector)
    out = set()
    for s in (sorted(set(u.sector)) if scope == "sector" else [None]):
        idx = np.arange(u.n) if s is None else np.flatnonzero(sec == s)
        v = vals[idx]
        level = {"absolute": value, "percentile": np.percentile(v, value),
                 "relative_to_mean": value * v.mean()}[basis]
        out |= set(idx[(v < level) if side == "below" else (v > level)].tolist())
    return sorted(out)


def test_exclude_threshold_hand_checked_cases():
    def ex(**kw):
        r, _ = one(R("exclude_threshold", **kw))
        return r
    base = dict(metric="esg", side="below", scope="universe")
    assert excluded_of(ex(**base, value=25.0, basis="absolute")) == [0, 1, 4, 5]  # 25 itself stays
    assert excluded_of(ex(**base, value=25, basis="percentile")) == [0, 4, 5]  # 25th pct = 18.75
    assert excluded_of(ex(**base, value=0.5, basis="relative_to_mean")) == [0, 4, 5]  # 0.5 * 40 = 20
    assert excluded_of(ex(metric="carbon", side="above", scope="universe", value=350.0,
                          basis="absolute")) == [4]
    assert excluded_of(ex(metric="carbon", side="above", scope="universe", value=1.0,
                          basis="relative_to_mean")) == [4, 5, 6]  # mean 102.08
    assert excluded_of(ex(metric="esg", side="below", scope="sector", value=50.0,
                          basis="percentile")) == [0, 1, 4, 5, 8, 9]  # below each sector median
    assert excluded_of(ex(metric="carbon", side="above", scope="sector", value=1.0,
                          basis="relative_to_mean")) == [2, 3, 4, 5, 10, 11]  # above each sector mean


@pytest.mark.parametrize("metric,side,scope,basis,value", [
    (m, s, sc, b, v) for m, s, sc in itertools.product(["esg", "carbon"], ["below", "above"], ["universe", "sector"])
    for b, v in [("absolute", 30.0), ("percentile", 40.0), ("relative_to_mean", 0.9)]])
def test_exclude_threshold_matches_numpy(metric, side, scope, basis, value):
    u = small()
    r, cr = one(R("exclude_threshold", metric=metric, side=side, value=value, basis=basis, scope=scope), u=u)
    want = _expected_excluded(u, metric, side, value, basis, scope)
    assert want, "test case should exclude something"
    assert r.status == "mapped" and excluded_of(r) == want
    assert cr.constraint_set.allowed_indices(u.n).tolist() == sorted(set(range(12)) - set(want))


def test_exclude_threshold_that_matches_nothing_is_trivial():
    r, cr = one(R("exclude_threshold", metric="esg", side="below", value=0.0, basis="absolute",
                  scope="universe"))
    assert r.status == "trivial" and r.constraints == [] and cr.warnings == []


# ------------------------------------------------------- portfolio_average
def test_portfolio_average_esg_min_levels():
    u = small()
    for basis, value, level in [("absolute", 33.0, 33.0), ("percentile", 50.0, 32.5),
                                ("relative_to_mean", 1.05, 42.0)]:
        r, cr = one(R("portfolio_average", metric="esg", bound="min", value=value, basis=basis), u=u)
        (c,) = r.constraints
        assert r.status == "mapped" and isinstance(c, MinESG) and math.isclose(c.min_avg_score, level)
        assert cr.constraint_set.constraints[1:] == [c]


def test_portfolio_average_carbon_max_levels():
    u = small()
    mean = 1225 / 12
    for basis, value, level in [("absolute", 150.0, 150.0),
                                ("percentile", 50.0, float(np.percentile(u.carbon, 50))),
                                ("relative_to_mean", 0.7, 0.7 * mean)]:
        r, _ = one(R("portfolio_average", metric="carbon", bound="max", value=value, basis=basis), u=u)
        (c,) = r.constraints
        assert r.status == "mapped" and isinstance(c, CarbonCap) and math.isclose(c.max_avg, level)


@pytest.mark.parametrize("metric,bound", [("esg", "max"), ("carbon", "min")])
def test_portfolio_average_other_direction_is_an_avgbound(metric, bound):
    u = small()
    mean = u.numeric(metric).mean()
    r, cr = one(R("portfolio_average", metric=metric, bound=bound, value=1.2, basis="relative_to_mean"), u=u)
    (c,) = r.constraints
    assert r.status == "mapped" and type(c) is AvgBound and c.attribute == metric
    assert math.isclose(c.upper if bound == "max" else c.lower, 1.2 * mean)
    assert (c.lower if bound == "max" else c.upper) is None
    assert metric in r.detail and "AvgBound" in r.detail
    assert cr.constraint_set.constraints[1:] == [c]


# ------------------------------------------------------ exclude by a category
def test_exclude_by_custom_category():
    r, cr = one(R("exclude", by="country", values=["fr", "UK"]))
    assert r.status == "mapped" and excluded_of(r) == [2, 3, 5, 6, 7, 9, 10] and cr.warnings == []
    assert "country" in r.detail
    r, cr = one(R("exclude", by="COUNTRY", values=["DE"]))  # the grouping name is case-insensitive too
    assert excluded_of(r) == [0, 1, 4, 8, 11]
    r, cr = one(R("exclude", by="country", values=["DE", "Atlantis"]))
    assert excluded_of(r) == [0, 1, 4, 8, 11] and len(cr.warnings) == 1 and "Atlantis" in cr.warnings[0]
    r, _ = one(R("exclude", by="country", values=["Atlantis"]))
    assert r.status == "unmapped" and r.constraints == []


def test_exclude_unknown_grouping_is_unmapped_and_lists_what_is_available():
    r, cr = one(R("exclude", by="continent", values=["Europe"]))
    assert r.status == "unmapped" and r.constraints == [] and "continent" in r.detail
    assert "available: flag, ticker, sector, country" in r.detail
    assert len(cr.constraint_set.constraints) == 1 and cr.warnings == []
    r, _ = one(R("exclude", by="country", values=["DE"]), u=synthetic_universe(12, 3, seed=0))  # no categories at all
    assert r.status == "unmapped" and "available: flag, ticker, sector" in r.detail and "country" not in r.detail.split("available")[1]


# ----------------------------------------------------------------- require
def test_require_excludes_the_complement_of_the_group():
    everyone = set(range(12))
    for by, values, group in [("sector", ["Tech"], {0, 1, 2, 3}),
                              ("sector", ["tech", "HEALTH"], {0, 1, 2, 3, 8, 9, 10, 11}),
                              ("flag", ["tobacco"], {1, 6}),
                              ("flag", ["Tobacco", "weapons"], {1, 6, 9}),
                              ("country", ["DE"], {0, 1, 4, 8, 11}),
                              ("country", ["fr", "uk"], {2, 3, 5, 6, 7, 9, 10}),
                              ("ticker", ["A000", "a005"], {0, 5})]:
        r, cr = one(R("require", by=by, values=values), k=2)
        assert r.status == "mapped" and cr.warnings == [], (by, values)
        assert excluded_of(r) == sorted(everyone - group), (by, values)
        assert cr.constraint_set.allowed_indices(12).tolist() == sorted(group)
        assert f"{len(group)} assets eligible" in r.detail and r.detail.startswith("only ")


def test_require_unknown_values_warn_but_known_ones_still_apply():
    r, cr = one(R("require", by="sector", values=["Tech", "Mining"]))
    assert r.status == "mapped" and excluded_of(r) == [4, 5, 6, 7, 8, 9, 10, 11]
    assert len(cr.warnings) == 1 and "Mining" in cr.warnings[0] and "require" in cr.warnings[0]
    r, cr = one(R("require", by="country", values=["DE", "Atlantis"]))
    assert excluded_of(r) == [2, 3, 5, 6, 7, 9, 10] and "Atlantis" in cr.warnings[0]


def test_require_with_no_match_is_unmapped():
    for by, values in [("sector", ["Mining"]), ("flag", ["gambling"]), ("country", ["Atlantis"]),
                       ("ticker", ["ZZZ"]), ("flag", [])]:
        r, cr = one(R("require", by=by, values=values))
        assert r.status == "unmapped" and r.constraints == []
        assert len(cr.constraint_set.constraints) == 1 and cr.constraint_set.allowed_indices(12).size == 12


def test_require_unknown_grouping_is_unmapped_and_lists_what_is_available():
    r, cr = one(R("require", by="continent", values=["Europe"]))
    assert r.status == "unmapped" and r.constraints == [] and "continent" in r.detail
    assert "available: flag, ticker, sector, country" in r.detail
    assert len(cr.constraint_set.constraints) == 1


def test_require_covering_every_asset_is_trivial_and_requiring_an_empty_flag_is_infeasible():
    r, cr = one(R("require", by="sector", values=["Tech", "Energy", "Health"]))
    assert r.status == "trivial" and r.constraints == [] and len(cr.constraint_set.constraints) == 1
    r, cr = one(R("require", by="flag", values=["unused"]))  # a flag that marks nobody
    assert r.status == "mapped" and excluded_of(r) == list(range(12))
    assert any("infeasible" in w and "0 assets" in w for w in cr.warnings)


# ------------------------------------------------------------- group_limit
def cb_of(rule):
    (c,) = rule.constraints
    assert isinstance(c, CountBound)
    return sorted(c.indices.tolist()), c.min_count, c.max_count


def test_group_limit_unit_weight_floors_a_maximum_and_ceils_a_minimum():
    # 30% of 5 names = 1.5: at most floor = 1, at least ceil = 2
    r, _ = one(R("group_limit", by="sector", values=["Tech"], bound="max", limit=0.3, unit="weight"), k=5)
    assert r.status == "mapped" and cb_of(r) == ([0, 1, 2, 3], 0, 1) and "floor(30% x 5) = 1" in r.detail
    r, cr = one(R("group_limit", by="sector", values=["Tech"], bound="min", limit=0.3, unit="weight"), k=5)
    assert r.status == "mapped" and cb_of(r) == ([0, 1, 2, 3], 2, None) and "ceil(30% x 5) = 2" in r.detail
    assert cr.warnings == [] and cr.constraint_set.constraints[1:] == r.constraints
    # a whole number of names is the same either way
    for bound, lo, hi in [("max", 0, 2), ("min", 2, None)]:
        r, _ = one(R("group_limit", by="sector", values=["Tech"], bound=bound, limit=0.4, unit="weight"), k=5)
        assert cb_of(r) == ([0, 1, 2, 3], lo, hi)


def test_group_limit_unit_names_takes_the_count_as_given():
    for bound, lo, hi in [("max", 0, 3), ("min", 3, None)]:
        r, _ = one(R("group_limit", by="sector", values=["Health"], bound=bound, limit=3, unit="names"), k=5)
        assert r.status == "mapped" and cb_of(r) == ([8, 9, 10, 11], lo, hi) and "3 names" in r.detail


def test_group_limit_by_flag_sector_country_and_ticker():
    cases = [("flag", ["tobacco"], [1, 6]), ("flag", ["Tobacco", "weapons"], [1, 6, 9]),
             ("sector", ["energy"], [4, 5, 6, 7]), ("sector", ["Tech", "Health"], [0, 1, 2, 3, 8, 9, 10, 11]),
             ("country", ["DE"], [0, 1, 4, 8, 11]), ("country", ["fr", "UK"], [2, 3, 5, 6, 7, 9, 10]),
             ("ticker", ["A000", "a002", "A003"], [0, 2, 3])]
    for by, values, idx in cases:
        r, cr = one(R("group_limit", by=by, values=values, bound="max", limit=1, unit="names"), k=4)
        assert r.status == "mapped" and cb_of(r) == (idx, 0, 1) and cr.warnings == [], (by, values)
        r, _ = one(R("group_limit", by=by, values=values, bound="min", limit=1, unit="names"), k=4)
        assert r.status == "mapped" and cb_of(r) == (idx, 1, None), (by, values)


def test_group_limit_by_sector_is_a_countbound_unless_it_is_each():
    r, _ = one(R("group_limit", by="sector", values=["Tech"], bound="max", limit=2, unit="names"))
    assert type(r.constraints[0]) is CountBound  # SectorCap is only for sector_cap rules and the "*" form


def test_group_limit_each_gives_one_constraint_per_category_value():
    r, cr = one(R("group_limit", by="country", values=["*"], bound="max", limit=0.5, unit="weight"))
    assert r.status == "mapped" and len(r.constraints) == 3 and "each country (3 of them)" in r.detail
    assert [(type(c), c.indices.tolist(), c.min_count, c.max_count) for c in r.constraints] == [
        (CountBound, [0, 1, 4, 8, 11], 0, 2), (CountBound, [2, 5, 6, 10], 0, 2), (CountBound, [3, 7, 9], 0, 2)]
    assert cr.constraint_set.constraints[1:] == r.constraints
    r, _ = one(R("group_limit", by="country", values=["each"], bound="min", limit=0.25, unit="weight"))
    assert [(c.min_count, c.max_count) for c in r.constraints] == [(1, None)] * 3 and "at least" in r.detail
    # sector + max -> SectorCap, sorted by name
    r, _ = one(R("group_limit", by="sector", values=["*"], bound="max", limit=0.5, unit="weight"))
    assert [(type(c), c.sector, c.max_count) for c in r.constraints] == [
        (SectorCap, "Energy", 2), (SectorCap, "Health", 2), (SectorCap, "Tech", 2)]
    # sector + min -> CountBound (a SectorCap cannot express a floor)
    r, _ = one(R("group_limit", by="sector", values=["ALL"], bound="min", limit=0.25, unit="weight"))
    assert [(type(c), c.indices.tolist(), c.min_count, c.max_count) for c in r.constraints] == [
        (CountBound, [4, 5, 6, 7], 1, None), (CountBound, [8, 9, 10, 11], 1, None), (CountBound, [0, 1, 2, 3], 1, None)]
    assert [c.label for c in r.constraints] == ["sector=Energy", "sector=Health", "sector=Tech"]


def test_group_limit_each_behaves_like_the_per_value_bounds_on_all_subsets():
    u = small()
    cr = compile_spec(S(HOLD4, R("group_limit", by="country", values=["*"], bound="max", limit=0.5, unit="weight")), u)
    X = np.zeros((495, 12), dtype=np.uint8)
    np.put_along_axis(X, np.array(list(itertools.combinations(range(12), 4))), 1, axis=1)
    country = np.array(u.categories["country"])
    want = np.array([all(x[country == c].sum() <= 2 for c in set(country)) for x in X])
    assert np.array_equal(cr.constraint_set.check_batch(X, u), want) and 0 < want.sum() < 495


def test_group_limit_max_that_rounds_to_zero_names_becomes_an_exclusion():
    for kw, group in [(dict(by="flag", values=["tobacco"], limit=0.15, unit="weight"), [1, 6]),     # floor(0.75) = 0
                      (dict(by="sector", values=["Energy"], limit=0, unit="names"), [4, 5, 6, 7]),
                      (dict(by="country", values=["UK"], limit=0.0, unit="weight"), [3, 7, 9])]:
        r, cr = one(R("group_limit", bound="max", **kw), k=5)
        assert r.status == "mapped" and excluded_of(r) == group and "excludes" in r.detail
        assert cr.constraint_set.allowed_indices(12).tolist() == sorted(set(range(12)) - set(group))
        assert not any(isinstance(c, CountBound) for c in cr.constraint_set.constraints)


def test_group_limit_max_that_cannot_bind_is_trivial():
    for kw in [dict(by="sector", values=["Tech"], limit=4, unit="names"),        # group has only 4 assets
               dict(by="flag", values=["tobacco"], limit=2, unit="names"),       # 2 assets, cap 2
               dict(by="sector", values=["Tech"], limit=0.9, unit="weight"),     # floor(4.5) = 4 >= 4 assets
               dict(by="country", values=["DE"], limit=1.0, unit="weight"),      # cap k = 5
               dict(by="flag", values=["unused"], limit=1, unit="names")]:       # an empty group can't exceed anything
        r, cr = one(R("group_limit", bound="max", **kw), k=5)
        assert r.status == "trivial" and r.constraints == [] and "cannot bind" in r.detail, kw
        assert len(cr.constraint_set.constraints) == 1 and cr.warnings == []
    r, _ = one(R("group_limit", by="sector", values=["Tech"], bound="max", limit=3, unit="names"), k=5)
    assert r.status == "mapped"  # one below the bound-or-group-size does bind


def test_group_limit_min_of_zero_is_trivial():
    for kw in [dict(limit=0, unit="names"), dict(limit=0.0, unit="weight")]:
        r, cr = one(R("group_limit", by="sector", values=["Tech"], bound="min", **kw), k=5)
        assert r.status == "trivial" and r.constraints == [] and "holds for any portfolio" in r.detail
        assert len(cr.constraint_set.constraints) == 1 and cr.warnings == []


def test_group_limit_impossible_min_is_mapped_with_a_warning():
    # tobacco has 2 assets; ceil(60% x 5) = 3 of them cannot be held
    for kw in [dict(limit=0.6, unit="weight"), dict(limit=3, unit="names")]:
        r, cr = one(R("group_limit", by="flag", values=["tobacco"], bound="min", **kw), k=5)
        assert r.status == "mapped" and cb_of(r) == ([1, 6], 3, None)
        assert len(cr.warnings) == 1 and "impossible" in cr.warnings[0] and "2 such assets" in cr.warnings[0]
        assert len(enumerate_feasible(small(), cr.constraint_set)) == 0
    # more than k also warns
    r, cr = one(R("group_limit", by="sector", values=["Tech"], bound="min", limit=5, unit="names"), k=4)
    assert r.status == "mapped" and len(cr.warnings) == 1
    r, cr = one(R("group_limit", by="sector", values=["Tech"], bound="min", limit=4, unit="names"), k=4)
    assert r.status == "mapped" and cr.warnings == []  # all four held: possible


def test_group_limit_unknown_values_warn_no_match_and_unknown_grouping_are_unmapped():
    r, cr = one(R("group_limit", by="country", values=["DE", "Atlantis"], bound="max", limit=2, unit="names"))
    assert r.status == "mapped" and cb_of(r) == ([0, 1, 4, 8, 11], 0, 2)
    assert len(cr.warnings) == 1 and "Atlantis" in cr.warnings[0] and "group_limit" in cr.warnings[0]
    r, cr = one(R("group_limit", by="country", values=["Atlantis"], bound="max", limit=2, unit="names"))
    assert r.status == "unmapped" and r.constraints == [] and len(cr.warnings) == 1
    r, cr = one(R("group_limit", by="continent", values=["*"], bound="max", limit=0.2, unit="weight"))
    assert r.status == "unmapped" and "continent" in r.detail and "available: flag, ticker, sector, country" in r.detail
    assert len(cr.constraint_set.constraints) == 1


# ----------------------------------------------------------- threshold_share
@pytest.mark.parametrize("metric,side,scope,basis,value", [
    (m, s, sc, b, v) for m, s, sc in itertools.product(["esg", "carbon", "cap"], ["below", "above"], ["universe", "sector"])
    for b, v in [("absolute", 30.0), ("percentile", 40.0), ("relative_to_mean", 0.9)]
    if len(_expected_excluded(small(), m, s, v, b, sc)) >= 2])  # at least two assets, so "at most 1 name" binds
def test_threshold_share_countbound_indices_match_numpy(metric, side, scope, basis, value):
    u = small()
    want = _expected_excluded(u, metric, side, value, basis, scope)  # the assets below/above the level
    base = dict(metric=metric, side=side, value=value, basis=basis, scope=scope, limit=1, unit="names")
    assert len(want) >= 2
    r, cr = one(R("threshold_share", bound="max", **base), u=u)
    assert r.status == "mapped" and cb_of(r) == (want, 0, 1) and cr.warnings == []
    r, _ = one(R("threshold_share", bound="min", **base), u=u)
    assert r.status == "mapped" and cb_of(r) == (want, 1, None)


def test_threshold_share_hand_checked_cases():
    u = small()
    # absolute: esg below 30 -> {0,1,4,5,6}; half of 4 names -> at least 2
    r, _ = one(R("threshold_share", metric="esg", side="below", value=30.0, basis="absolute", scope="universe",
                 bound="min", limit=0.5, unit="weight"), u=u)
    assert cb_of(r) == ([0, 1, 4, 5, 6], 2, None)
    assert np.array_equal(sorted(np.flatnonzero(u.esg_score < 30)), [0, 1, 4, 5, 6])
    # absolute: carbon above 100 -> {4,5,6} (100 itself is not above); at most one name
    r, _ = one(R("threshold_share", metric="carbon", side="above", value=100.0, basis="absolute", scope="universe",
                 bound="max", limit=1, unit="names"), u=u)
    assert cb_of(r) == ([4, 5, 6], 0, 1)
    # percentile: esg below its 25th percentile (18.75) -> {0,4,5}; a quarter of 4 names -> at least 1
    r, _ = one(R("threshold_share", metric="esg", side="below", value=25, basis="percentile", scope="universe",
                 bound="min", limit=0.25, unit="weight"), u=u)
    assert cb_of(r) == ([0, 4, 5], 1, None)
    # relative to the mean: esg above 1.0 x 40 -> Health only; at least half of 4 names
    r, _ = one(R("threshold_share", metric="esg", side="above", value=1.0, basis="relative_to_mean",
                 scope="universe", bound="min", limit=0.5, unit="weight"), u=u)
    assert cb_of(r) == ([8, 9, 10, 11], 2, None)
    # per sector: esg below the sector median -> two of each block; at most a quarter of 4 names = 1
    r, _ = one(R("threshold_share", metric="esg", side="below", value=50.0, basis="percentile", scope="sector",
                 bound="max", limit=0.25, unit="weight"), u=u)
    assert cb_of(r) == ([0, 1, 4, 5, 8, 9], 0, 1)
    assert "by sector" in r.detail or "esg below" in r.detail


def test_threshold_share_scope_sector_differs_from_scope_universe():
    u = small()
    kw = dict(metric="esg", side="below", value=50.0, basis="percentile", bound="min", limit=1, unit="names")
    uni, _ = one(R("threshold_share", scope="universe", **kw), u=u)
    sec, _ = one(R("threshold_share", scope="sector", **kw), u=u)
    assert cb_of(uni)[0] == [0, 1, 2, 4, 5, 6] and cb_of(sec)[0] == [0, 1, 4, 5, 8, 9]  # median 32.5 vs per-sector medians


def test_threshold_share_cannot_bind_cases_are_trivial_and_impossible_min_warns():
    u = small()
    none = dict(metric="esg", side="below", value=0.0, basis="absolute", scope="universe")  # nobody scores below 0
    r, cr = one(R("threshold_share", bound="max", limit=1, unit="names", **none), u=u)
    assert r.status == "trivial" and r.constraints == [] and "cannot bind" in r.detail and cr.warnings == []
    r, cr = one(R("threshold_share", bound="min", limit=1, unit="names", **none), u=u)
    assert r.status == "mapped" and cb_of(r) == ([], 1, None) and "impossible" in cr.warnings[0]
    r, cr = one(R("threshold_share", bound="min", limit=0, unit="names", **none), u=u)
    assert r.status == "trivial" and cr.warnings == []
    everyone = dict(metric="esg", side="above", value=0.0, basis="absolute", scope="universe")  # all 12 qualify
    r, _ = one(R("threshold_share", bound="max", limit=4, unit="names", **everyone), u=u)  # cap equals k
    assert r.status == "trivial"
    r, _ = one(R("threshold_share", bound="max", limit=0.25, unit="weight", **everyone), u=u)
    assert r.status == "mapped" and cb_of(r) == (list(range(12)), 0, 1)
    # a max that rounds to zero excludes everyone matching the level
    r, cr = one(R("threshold_share", metric="carbon", side="above", value=100.0, basis="absolute",
                  scope="universe", bound="max", limit=0.2, unit="weight"), u=u)  # floor(0.8) = 0
    assert r.status == "mapped" and excluded_of(r) == [4, 5, 6]


def test_threshold_share_unknown_metric_is_unmapped_and_lists_what_is_available():
    r, cr = one(R("threshold_share", metric="water", side="below", value=1.0, basis="absolute", scope="universe",
                  bound="min", limit=0.5, unit="weight"))
    assert r.status == "unmapped" and r.constraints == [] and "water" in r.detail
    assert "available: esg, carbon, cap" in r.detail and len(cr.constraint_set.constraints) == 1


# --------------------------------------------- exclude_threshold, custom field
def test_exclude_threshold_on_a_custom_attribute():
    u = small()  # cap = [5, 1, 8, 2, 9, 3, 7, 4, 6, 10, 12, 11], mean 6.5
    def ex(**kw):
        r, cr = one(R("exclude_threshold", **kw), u=u)
        return r, cr
    r, cr = ex(metric="cap", side="below", value=3.0, basis="absolute", scope="universe")
    assert r.status == "mapped" and excluded_of(r) == [1, 3] and "cap below 3" in r.detail
    assert cr.warnings == []
    r, _ = ex(metric="cap", side="above", value=10.0, basis="absolute", scope="universe")
    assert excluded_of(r) == [10, 11]
    r, _ = ex(metric="cap", side="below", value=0.5, basis="relative_to_mean", scope="universe")  # < 3.25
    assert excluded_of(r) == [1, 3, 5]
    r, _ = ex(metric="CAP", side="above", value=75.0, basis="percentile", scope="universe")
    assert excluded_of(r) == np.flatnonzero(u.attributes["cap"] > np.percentile(u.attributes["cap"], 75)).tolist()
    r, _ = ex(metric="cap", side="below", value=50.0, basis="percentile", scope="sector")
    assert excluded_of(r) == _expected_excluded(u, "cap", "below", 50.0, "percentile", "sector")
    r, _ = ex(metric="cap", side="below", value=0.0, basis="absolute", scope="universe")
    assert r.status == "trivial" and r.constraints == []


def test_exclude_threshold_unknown_metric_is_unmapped_and_lists_what_is_available():
    r, cr = one(R("exclude_threshold", metric="water", side="below", value=1.0, basis="absolute", scope="universe"))
    assert r.status == "unmapped" and r.constraints == [] and "water" in r.detail
    assert "available: esg, carbon, cap" in r.detail and len(cr.constraint_set.constraints) == 1
    r, _ = one(R("exclude_threshold", metric="cap", side="below", value=1.0, basis="absolute", scope="universe"),
               u=synthetic_universe(12, 3, seed=0))  # universe without attributes
    assert r.status == "unmapped" and "available: esg, carbon" in r.detail and "cap" not in r.detail.split("available")[1]


# -------------------------------------------------- portfolio_average, new forms
def test_portfolio_average_esg_min_and_carbon_max_keep_their_dedicated_classes():
    u = small()
    r, _ = one(R("portfolio_average", metric="esg", bound="min", value=1.0, basis="relative_to_mean"), u=u)
    assert type(r.constraints[0]) is MinESG and math.isclose(r.constraints[0].min_avg_score, 40.0)
    r, _ = one(R("portfolio_average", metric="Carbon", bound="max", value=1.0, basis="relative_to_mean"), u=u)
    assert type(r.constraints[0]) is CarbonCap and math.isclose(r.constraints[0].max_avg, 1225 / 12)


def test_portfolio_average_esg_max_and_carbon_min_are_avgbounds_with_the_right_side():
    u = small()
    r, cr = one(R("portfolio_average", metric="esg", bound="max", value=60.0, basis="absolute"), u=u)
    (c,) = r.constraints
    assert type(c) is AvgBound and (c.attribute, c.lower, c.upper) == ("esg", None, 60.0)
    r, _ = one(R("portfolio_average", metric="carbon", bound="min", value=50.0, basis="percentile"), u=u)
    (c,) = r.constraints
    assert type(c) is AvgBound and c.attribute == "carbon" and c.upper is None
    assert math.isclose(c.lower, float(np.percentile(u.carbon, 50)))
    # and they check what they say
    X = np.zeros((2, 12), dtype=np.uint8)
    X[0, [8, 9, 10, 11]] = 1   # esg average 75
    X[1, [0, 1, 4, 5]] = 1     # esg average 12.5
    r, _ = one(R("portfolio_average", metric="esg", bound="max", value=60.0, basis="absolute"), u=u)
    assert r.constraints[0].check_batch(X, u).tolist() == [False, True]


def test_portfolio_average_custom_attribute_min_and_max():
    u = small()  # cap mean 6.5, 25th percentile of cap = 3.75
    for bound in ("min", "max"):
        for basis, value, level in [("absolute", 6.0, 6.0), ("relative_to_mean", 0.8, 5.2),
                                    ("percentile", 25.0, float(np.percentile(u.attributes["cap"], 25)))]:
            r, cr = one(R("portfolio_average", metric="cap", bound=bound, value=value, basis=basis), u=u)
            (c,) = r.constraints
            assert r.status == "mapped" and type(c) is AvgBound and c.attribute == "cap"
            got, other = (c.lower, c.upper) if bound == "min" else (c.upper, c.lower)
            assert math.isclose(got, level) and other is None
            assert cr.constraint_set.constraints[1:] == [c] and "AvgBound" in r.detail
    r, _ = one(R("portfolio_average", metric="CAP", bound="min", value=6.0, basis="absolute"), u=u)
    assert r.constraints[0].attribute == "cap"  # stored under the universe's own spelling
    X = np.zeros((2, 12), dtype=np.uint8)
    X[0, [9, 10, 11, 2]] = 1   # cap 10, 12, 11, 8 -> avg 10.25
    X[1, [1, 3, 5, 7]] = 1     # cap 1, 2, 3, 4 -> avg 2.5
    r, _ = one(R("portfolio_average", metric="cap", bound="min", value=6.0, basis="absolute"), u=u)
    assert r.constraints[0].check_batch(X, u).tolist() == [True, False]


def test_portfolio_average_unknown_metric_is_unmapped_and_lists_what_is_available():
    r, cr = one(R("portfolio_average", metric="water", bound="min", value=1.0, basis="absolute"))
    assert r.status == "unmapped" and r.constraints == [] and "water" in r.detail
    assert "available: esg, carbon, cap" in r.detail and len(cr.constraint_set.constraints) == 1


# --------------------------------------------------------------- min_groups
def test_min_groups_by_sector_and_by_custom_category():
    r, cr = one(R("min_groups", category="sector", min=2))
    (c,) = r.constraints
    assert r.status == "mapped" and type(c) is MinGroups and (c.category, c.min_groups) == ("sector", 2)
    assert "3 in the universe" in r.detail and cr.warnings == [] and cr.constraint_set.constraints[1:] == [c]
    r, cr = one(R("min_groups", category="Country", min=3))
    (c,) = r.constraints
    assert r.status == "mapped" and (c.category, c.min_groups) == ("country", 3) and cr.warnings == []
    x = np.zeros(12, dtype=np.uint8)
    x[[0, 2, 3, 4]] = 1  # DE FR UK DE
    assert c.check(x, small())
    x[:] = 0
    x[[0, 1, 4, 8]] = 1  # only DE
    assert not c.check(x, small())


def test_min_groups_of_one_is_trivial():
    r, cr = one(R("min_groups", category="sector", min=1))
    assert r.status == "trivial" and r.constraints == [] and cr.warnings == []
    assert len(cr.constraint_set.constraints) == 1


def test_min_groups_impossible_min_warns():
    r, cr = one(R("min_groups", category="country", min=4))  # only 3 countries
    assert r.status == "mapped" and len(cr.warnings) == 1
    assert "impossible" in cr.warnings[0] and "3 in the universe" in cr.warnings[0]
    r, cr = one(R("min_groups", category="sector", min=3), k=2)  # more groups than names
    assert r.status == "mapped" and len(cr.warnings) == 1 and "k=2" in cr.warnings[0]
    r, cr = one(R("min_groups", category="sector", min=3), k=3)
    assert cr.warnings == []
    cr = compile_spec(S(R("min_groups", category="country", min=3)), small(), k=4)
    assert len(enumerate_feasible(small(), cr.constraint_set)) == 495 - len(
        [1 for c in itertools.combinations(range(12), 4) if len({small().categories["country"][i] for i in c}) < 3])


def test_min_groups_unknown_category_is_unmapped_and_lists_what_is_available():
    r, cr = one(R("min_groups", category="continent", min=2))
    assert r.status == "unmapped" and r.constraints == [] and "continent" in r.detail
    assert "available: sector, country" in r.detail and len(cr.constraint_set.constraints) == 1


# --------------------------------------------------------------- risk_limit
def test_risk_limit_volatility_and_tracking_error():
    r, cr = one(R("risk_limit", measure="volatility", max=0.2))
    (c,) = r.constraints
    assert r.status == "mapped" and type(c) is VolatilityCap and c.max_risk == 0.2
    assert "VolatilityCap" in r.detail and "20.0%" in r.detail and cr.constraint_set.constraints[1:] == [c]
    r, cr = one(R("risk_limit", measure="tracking_error", max=0.06))
    (c,) = r.constraints
    assert r.status == "mapped" and type(c) is TrackingErrorCap and c.max_risk == 0.06 and c.benchmark is None
    assert "TrackingErrorCap" in r.detail and "6.0%" in r.detail and cr.warnings == []


def test_risk_limit_constraints_use_the_universe_covariance():
    u = small()
    X = np.zeros((495, 12), dtype=np.uint8)
    np.put_along_axis(X, np.array(list(itertools.combinations(range(12), 4))), 1, axis=1)
    vol = np.sqrt([(x / 4) @ u.cov @ (x / 4) for x in X.astype(float)])
    cap = float(np.median(vol))
    r, _ = one(R("risk_limit", measure="volatility", max=cap), u=u)
    assert np.array_equal(r.constraints[0].check_batch(X, u), vol <= cap)


# ----------------------------------------- everything together, brute force
def _brute_spec_and_rule(u):
    X = np.zeros((495, 12), dtype=np.uint8)
    combos = list(itertools.combinations(range(12), 4))
    np.put_along_axis(X, np.array(combos), 1, axis=1)
    Xf = X.astype(float)
    vol = np.sqrt([(x / 4) @ u.cov @ (x / 4) for x in Xf])
    te = np.sqrt([(x / 4 - 1 / 12) @ u.cov @ (x / 4 - 1 / 12) for x in Xf])
    vol_cap, te_cap = float(np.quantile(vol, 0.9)), float(np.quantile(te, 0.9))
    spec = S(HOLD4,
             R("position_cap", max_weight=0.25),
             R("require", by="country", values=["DE", "FR"]),                                       # UK {3,7,9} out
             R("exclude_threshold", metric="carbon", side="above", value=350.0, basis="absolute",
               scope="universe"),                                                                    # {4} out
             R("group_limit", by="flag", values=["tobacco"], bound="max", limit=1, unit="names"),
             R("group_limit", by="flag", values=["green"], bound="max", limit=2, unit="names"),
             R("group_limit", by="sector", values=["Health"], bound="min", limit=0.25, unit="weight"),
             R("group_limit", by="country", values=["*"], bound="max", limit=0.5, unit="weight"),
             R("threshold_share", metric="carbon", side="above", value=1.0, basis="relative_to_mean",
               scope="universe", bound="max", limit=1, unit="names"),
             R("min_groups", category="sector", min=2),
             R("portfolio_average", metric="cap", bound="max", value=1.2, basis="relative_to_mean"),
             R("portfolio_average", metric="esg", bound="min", value=0.8, basis="relative_to_mean"),
             R("portfolio_average", metric="carbon", bound="min", value=10.0, basis="absolute"),
             R("risk_limit", measure="volatility", max=vol_cap),
             R("risk_limit", measure="tracking_error", max=te_cap),
             R("unmapped", reason="n/a"))
    return spec, X, vol, te, vol_cap, te_cap, combos


def test_compiled_constraints_agree_with_an_independent_evaluation_on_all_subsets():
    u = small()
    spec, X, vol, te, vol_cap, te_cap, combos = _brute_spec_and_rule(u)
    cr = compile_spec(spec, u)
    assert cr.k == 4 and cr.warnings == []
    kinds = {r.rule["kind"] for r in cr.rules}
    assert len(kinds) >= 10 and {"group_limit", "threshold_share", "min_groups", "risk_limit", "require"} <= kinds
    assert [r.status for r in cr.rules].count("trivial") == 1
    got = cr.constraint_set.check_batch(X, u)

    country, sector = np.array(u.categories["country"]), np.array(u.sector)
    green, tob = u.flags["green"], u.flags["tobacco"]
    cap, esg, carbon = u.attributes["cap"], u.esg_score, u.carbon
    banned = {i for i in range(12) if country[i] == "UK" or carbon[i] > 350.0}

    def ok(j, c):
        c = list(c)
        return (not banned & set(c)
                and tob[c].sum() <= 1
                and green[c].sum() <= 2
                and (sector[c] == "Health").sum() >= 1
                and all((country[c] == v).sum() <= 2 for v in set(country))
                and (carbon[c] > carbon.mean()).sum() <= 1
                and len(set(sector[c])) >= 2
                and cap[c].mean() <= 1.2 * cap.mean()
                and esg[c].mean() >= 0.8 * esg.mean()
                and carbon[c].mean() >= 10.0
                and vol[j] <= vol_cap and te[j] <= te_cap)

    want = np.array([ok(j, c) for j, c in enumerate(combos)])
    assert 0 < want.sum() < len(combos)
    assert np.array_equal(got, want)
    assert np.array_equal(enumerate_feasible(u, cr.constraint_set), X[want])
    assert all(cr.constraint_set.check(x, u) == w for x, w in zip(X[:60], want[:60]))
    # check_fund reports the failing rule(s) for a selection that breaks only one of them
    j = next(j for j, c in enumerate(combos) if want[j])
    x = X[j].copy()
    assert all(ok_ for _, ok_ in cr.check_fund(x, u))
    bad = x.copy()
    bad[int(np.flatnonzero(x)[0])] = 0
    bad[3] = 1  # swap in a UK name: the only rule it can break first is `require`
    failing = [r.rule["kind"] for r, ok_ in cr.check_fund(bad, u) if not ok_]
    assert "require" in failing and "holdings" not in failing


# ---------------------------------------------------------------- unmapped
def test_unmapped_passes_reason_through():
    r, cr = one(R("unmapped", reason="Voting does not restrict holdings."))
    assert r.status == "unmapped" and r.detail == "Voting does not restrict holdings."
    assert r.constraints == [] and r.rule["kind"] == "unmapped" and cr.warnings == []
    assert len(cr.constraint_set.constraints) == 1


# -------------------------------------------- the ConstraintSet that results
def test_cardinality_is_first_and_unique_whatever_the_rule_order():
    spec = S(VALID["exclude"], R("sector_cap", sector="*", limit=0.5, unit="weight"), HOLD4,
             R("portfolio_average", metric="esg", bound="min", value=0.5, basis="relative_to_mean"),
             R("holdings", min=3, max=5))
    cr = compile_spec(spec, small())
    cons = cr.constraint_set.constraints
    assert isinstance(cons[0], Cardinality) and cons[0].k == 4 and n_cardinality(cr.constraint_set) == 1
    assert len(cons) == 1 + 1 + 3 + 1  # exclusion, three sector caps, MinESG
    assert cr.constraint_set.cardinality == 4


def test_constraint_set_agrees_with_brute_force_on_all_subsets():
    u = small()
    spec = S(HOLD4,
             R("position_cap", max_weight=0.25),
             R("sector_cap", sector="*", limit=0.5, unit="weight"),
             R("exclude", by="flag", values=["weapons"]),
             R("exclude", by="ticker", values=["A001"]),
             R("exclude_threshold", metric="carbon", side="above", value=350.0, basis="absolute",
               scope="universe"),
             R("portfolio_average", metric="esg", bound="min", value=1.0, basis="relative_to_mean"),
             R("portfolio_average", metric="carbon", bound="max", value=1.0, basis="relative_to_mean"),
             R("unmapped", reason="n/a"))
    cr = compile_spec(spec, u)
    combos = list(itertools.combinations(range(12), 4))
    X = np.zeros((len(combos), 12), dtype=np.uint8)
    np.put_along_axis(X, np.array(combos), 1, axis=1)
    got = cr.constraint_set.check_batch(X, u)
    banned = {9, 1, 4}
    sector_of = np.array(u.sector)

    def ok(c):
        c = list(c)
        return (not banned & set(c)
                and all(np.sum(sector_of[c] == s) <= 2 for s in set(u.sector))
                and u.esg_score[c].mean() >= u.esg_score.mean()
                and u.carbon[c].mean() <= u.carbon.mean())

    want = np.array([ok(c) for c in combos])
    assert np.array_equal(got, want)
    assert 0 < want.sum() < len(combos)
    assert np.array_equal(enumerate_feasible(u, cr.constraint_set), X[want])
    assert all(cr.constraint_set.check(x, u) == w for x, w in zip(X[:40], want[:40]))


def test_exclusions_leaving_fewer_than_k_assets_warn_infeasible():
    cr = compile_spec(S(R("exclude", by="sector", values=["Tech", "Energy"])), small(), k=5)
    assert any("infeasible" in w and "4 assets" in w and "k=5" in w for w in cr.warnings)
    assert cr.constraint_set.allowed_indices(12).tolist() == [8, 9, 10, 11]
    cr = compile_spec(S(R("exclude", by="sector", values=["Tech", "Energy"])), small(), k=4)  # exactly k left
    assert not any("infeasible" in w for w in cr.warnings)
    # exclusions from different rules overlap: counted once
    cr = compile_spec(S(R("exclude", by="sector", values=["Tech"]),
                        R("exclude", by="ticker", values=["A000", "A001"]),
                        R("exclude", by="sector", values=["Energy"])), small(), k=5)
    assert any("4 assets" in w for w in cr.warnings)


# ------------------------------------------------------------ CompiledRules
def _mixed():
    return compile_spec(S(
        HOLD4,
        R("position_cap", max_weight=0.5),
        R("sector_cap", sector="Mining", limit=1, unit="names", source="x"),
        R("sector_cap", sector="Tech", limit=2, unit="names", source="Tech:\n   at most   two names " + "y" * 200),
        R("exclude", by="flag", values=["tobacco", "gambling"], source="No tobacco."),
        R("portfolio_average", metric="esg", bound="min", value=30.0, basis="absolute"),
        R("unmapped", reason="Engagement is not a holding rule.", source="We engage.")), small())


def test_report_mentions_every_rule_status_and_warning():
    cr = _mixed()
    assert isinstance(cr, CompiledRules) and cr.warnings  # unknown flag "gambling"
    text = cr.report()
    first = text.splitlines()[0]
    assert first.startswith("7 rules: 4 mapped, 1 trivial, 2 unmapped") and first.endswith("k = 4")
    for r in cr.rules:
        assert r.rule["kind"] in text and r.status in text and r.detail in text
    for w in cr.warnings:
        assert f"WARNING: {w}" in text
    assert 'source: "No tobacco."' in text
    assert 'source: "Tech: at most two names ' in text and "..." in text  # whitespace collapsed, long quote cut
    assert "y" * 150 not in text
    assert text.count("source:") == 4  # rules with an empty source print no source line


def test_report_without_warnings_has_no_warning_line():
    assert "WARNING" not in compile_spec(S(HOLD4), small()).report()


def test_unmapped_property_lists_only_unmapped_rules():
    cr = _mixed()
    assert [r.rule["kind"] for r in cr.unmapped] == ["sector_cap", "unmapped"]
    assert all(r.status == "unmapped" for r in cr.unmapped)
    assert compile_spec(S(HOLD4), small()).unmapped == []


def test_check_fund_returns_one_entry_per_mapped_rule_and_flags_the_broken_one():
    u = small()
    cr = compile_spec(S(HOLD4, R("position_cap", max_weight=0.5),
                        R("sector_cap", sector="Tech", limit=2, unit="names"),
                        R("exclude", by="flag", values=["tobacco"]),
                        R("portfolio_average", metric="esg", bound="min", value=30.0, basis="absolute"),
                        R("unmapped", reason="n/a")), u)

    def verdict(idx):
        x = np.zeros(12, dtype=np.uint8)
        x[idx] = 1
        res = cr.check_fund(x, u)
        assert [r for r, _ in res] == [r for r in cr.rules if r.constraints]
        assert len(res) == 4  # trivial and unmapped rules are not listed
        return {r.rule["kind"]: ok for r, ok in res}

    assert verdict([2, 3, 8, 9]) == dict(holdings=True, sector_cap=True, exclude=True, portfolio_average=True)
    assert verdict([2, 3, 8, 9, 10]) == dict(holdings=False, sector_cap=True, exclude=True, portfolio_average=True)
    assert verdict([0, 2, 3, 8]) == dict(holdings=True, sector_cap=False, exclude=True, portfolio_average=True)
    assert verdict([2, 3, 8, 6]) == dict(holdings=True, sector_cap=True, exclude=False, portfolio_average=True)
    assert verdict([4, 5, 7, 8]) == dict(holdings=True, sector_cap=True, exclude=True, portfolio_average=False)


# -------------------------------------------------------- feasible_fraction
def _fraction_case():
    u = small()
    cr = compile_spec(S(HOLD4, R("exclude", by="flag", values=["weapons"]),
                        R("sector_cap", sector="*", limit=0.5, unit="weight"),
                        R("portfolio_average", metric="esg", bound="min", value=0.9,
                          basis="relative_to_mean")), u)
    return u, cr.constraint_set


def test_feasible_fraction_matches_exact_fraction_of_allowed_subsets():
    u, cs = _fraction_case()
    allowed = cs.allowed_indices(u.n)
    assert allowed.size == 11
    exact = len(enumerate_feasible(u, cs)) / math.comb(allowed.size, 4)
    assert 0.2 < exact < 0.8
    p, lo, hi = feasible_fraction(u, cs, draws=20000, seed=0)
    assert lo <= p <= hi and lo <= exact <= hi and hi - lo < 0.03
    assert abs(p - exact) < 0.02
    p2, lo2, hi2 = feasible_fraction(u, cs, draws=25000, seed=1, chunk=7000)  # uneven last chunk
    assert lo2 <= exact <= hi2


def test_feasible_fraction_without_other_rules_is_one():
    u = small()
    p, lo, hi = feasible_fraction(u, ConstraintSet([Cardinality(4)]), draws=2000, seed=0)
    assert p == 1.0 and hi == 1.0 and lo > 0.99


def test_feasible_fraction_zero_when_exclusions_leave_too_few_assets():
    u = small()
    cr = compile_spec(S(R("exclude", by="sector", values=["Tech", "Energy"])), u, k=5)
    assert feasible_fraction(u, cr.constraint_set, draws=500) == (0.0, 0.0, 0.0)
    assert feasible_fraction(u, ConstraintSet([MinESG(1.0)]), draws=500) == (0.0, 0.0, 0.0)  # no k


def test_feasible_fraction_zero_hits_gives_zero_estimate_and_small_upper_bound():
    u = small()
    cs = ConstraintSet([Cardinality(4), MinESG(1e6)])
    p, lo, hi = feasible_fraction(u, cs, draws=3000, seed=0)
    assert p == 0.0 and lo < 1e-12 and 0 < hi < 3.5 / 3000 * 2


def test_feasible_fraction_is_seed_reproducible():
    u, cs = _fraction_case()
    a = feasible_fraction(u, cs, draws=3000, seed=5)
    assert a == feasible_fraction(u, cs, draws=3000, seed=5)
    assert a != feasible_fraction(u, cs, draws=3000, seed=6)
    assert feasible_fraction(u, cs, draws=3000, seed=None)[0] > 0  # unseeded still works


# ---------------------------------------------------------- example, end to end
def test_example_compiles_on_named_universe():
    u = named_universe(150)
    assert u.n == 150 and set(u.flags) == {"tobacco", "controversial_weapons", "thermal_coal", "sbti_target"}
    assert [int(u.flags[f].sum()) for f in sorted(u.flags)] == [2, 80, 3, 2]
    assert set(u.categories) == {"country", "region"} and set(u.attributes) == {"market_cap_eur_bn", "board_women_pct"}
    spec = load_spec(EXAMPLE_SPEC)
    assert len(spec["rules"]) == 18
    cr = compile_spec(spec, u, k=20)
    statuses = [r.status for r in cr.rules]
    assert (statuses.count("mapped"), statuses.count("trivial"), statuses.count("unmapped")) == (14, 1, 3)
    assert cr.warnings == [] and cr.k == 20
    assert [r.rule["kind"] for r in cr.unmapped] == ["unmapped"] * 3
    trivial = next(r for r in cr.rules if r.status == "trivial")
    assert trivial.rule["kind"] == "position_cap"
    caps = [c for c in cr.constraint_set.constraints if isinstance(c, SectorCap)]
    assert len(caps) == 10 and {c.max_count for c in caps} == {3}  # floor(0.15 * 20)
    assert n_cardinality(cr.constraint_set) == 1 and cr.constraint_set.cardinality == 20
    kinds = [type(c) for c in cr.constraint_set.constraints]
    for cls, n in [(MinGroups, 1), (MinESG, 1), (CarbonCap, 1), (AvgBound, 1), (TrackingErrorCap, 1),
                   (VolatilityCap, 0)]:
        assert kinds.count(cls) == n
    assert kinds.count(CountBound) == 11  # one per country (ten) plus the science-based-target floor
    p, lo, hi = feasible_fraction(u, cr.constraint_set, draws=20000, seed=0)
    assert 0.001 < p < 0.02 and lo <= p <= hi


def test_example_spec_details_on_named_universe():
    u = named_universe(150)
    cr = compile_spec(load_spec(EXAMPLE_SPEC), u, k=20)
    by_kind = {}
    for r in cr.rules:
        by_kind.setdefault(r.rule["kind"], []).append(r)
    # require region: everything outside Nordic / Western Europe is out
    assert excluded_of(by_kind["require"][0]) == np.flatnonzero(np.array(u.categories["region"]) == "North America").tolist()
    # at most 25% per country = 5 names, one CountBound per country
    country = np.array(u.categories["country"])
    cb = by_kind["group_limit"][0].constraints
    assert [(c.label, c.min_count, c.max_count) for c in cb] == [(f"country={v}", 0, 5) for v in sorted(set(country))]
    assert all(np.array_equal(c.indices, np.flatnonzero(country == c.label[8:])) for c in cb)
    # at least half = 10 names with a science-based target
    (sb,) = by_kind["group_limit"][1].constraints
    assert (sb.min_count, sb.max_count) == (10, None) and np.array_equal(sb.indices, np.flatnonzero(u.flags["sbti_target"]))
    # market cap and women on boards are the universe's own attributes
    (ex_cap,) = by_kind["exclude_threshold"][0].constraints
    assert ex_cap.indices.tolist() == np.flatnonzero(u.attributes["market_cap_eur_bn"] < 1).tolist()
    avg = [r.constraints[0] for r in by_kind["portfolio_average"]]
    assert [type(c) for c in avg] == [MinESG, CarbonCap, AvgBound]
    assert (avg[2].attribute, avg[2].lower, avg[2].upper) == ("board_women_pct", 34.0, None)
    assert math.isclose(avg[0].min_avg_score, 1.05 * u.esg_score.mean()) and math.isclose(avg[1].max_avg, 0.75 * u.carbon.mean())
    (ml,) = by_kind["min_groups"][0].constraints
    assert (ml.category, ml.min_groups) == ("sector", 8)
    (te,) = by_kind["risk_limit"][0].constraints
    assert type(te) is TrackingErrorCap and te.max_risk == 0.06 and te.benchmark is None


def test_example_sources_are_all_found_in_the_mandate_text():
    spec = load_spec(EXAMPLE_SPEC)
    text = (EXAMPLES / "mandate_example.txt").read_text()
    assert unverified_sources(spec, text) == []
    assert all(r["source"].strip() for r in spec["rules"])


def test_example_without_k_uses_midpoint_and_warns():
    cr = compile_spec(load_spec(EXAMPLE_SPEC), named_universe(150))
    assert cr.k == 20 and any("18-22" in w for w in cr.warnings)


# ------------------------------------------- allowed_indices and the samplers
def _sampler_case():
    u = small()
    excl = [1, 6, 9]
    cs = ConstraintSet([Cardinality(4), Exclusion(excl), SectorCap("Tech", 2), MinESG(25.0)])
    return u, cs, excl, enumerate_feasible(u, cs)


def test_allowed_indices_is_union_of_exclusions_ascending():
    assert ConstraintSet([Cardinality(2)]).allowed_indices(5).tolist() == [0, 1, 2, 3, 4]
    cs = ConstraintSet([Exclusion([3, 1]), Cardinality(2), Exclusion([1, 4]), MinESG(1.0), Exclusion([])])
    assert cs.allowed_indices(6).tolist() == [0, 2, 5]
    assert ConstraintSet([Exclusion(range(5))]).allowed_indices(5).size == 0


def test_rejection_sampler_respects_exclusions_and_is_close_to_uniform():
    u, cs, excl, F = _sampler_case()
    M = len(F)
    assert 30 < M < 300 and F[:, excl].sum() == 0
    shots = 6000
    r = rejection_sampler(u, cs, shots, seed=3)
    X = r.samples
    assert X.shape == (shots, 12) and X.dtype == np.uint8 and cs.check_batch(X, u).all()
    assert X[:, excl].sum() == 0 and (X.sum(axis=1) == 4).all()
    assert {tuple(x) for x in F} == {tuple(x) for x in X}  # every feasible selection shows up
    tv = tv_to_uniform(X, F)
    assert tv < 2 * tv_expected_uniform(M, shots, seed=0) + 0.02
    assert np.array_equal(X, rejection_sampler(u, cs, shots, seed=3).samples)
    assert r.cost >= shots and r.cost_unit == "proposals"


def test_dicke_sampler_respects_exclusions_and_is_close_to_uniform():
    u, cs, excl, F = _sampler_case()
    M = len(F)
    shots = 4000
    r = dicke_sampler()(u, cs, shots, 5)  # 9 qubits: the 3 excluded assets cost none
    X = r.samples
    assert X.shape == (shots, 12) and cs.check_batch(X, u).all() and X[:, excl].sum() == 0
    assert (X.sum(axis=1) == 4).all() and r.cost_unit == "shots" and r.cost >= shots
    assert {tuple(x) for x in F} == {tuple(x) for x in X}
    assert tv_to_uniform(X, F) < 2 * tv_expected_uniform(M, shots, seed=0) + 0.02


def test_samplers_raise_when_exclusions_leave_fewer_than_k_assets():
    u = small()
    cs = compile_spec(S(R("exclude", by="sector", values=["Tech", "Energy"])), u, k=5).constraint_set
    with pytest.raises(RuntimeError, match="fewer than k=5"):
        rejection_sampler(u, cs, 10, seed=0)
    with pytest.raises(RuntimeError, match="fewer than k=5"):
        dicke_sampler()(u, cs, 10, 0)


def test_compiled_rules_drive_the_sampler_end_to_end():
    u = small()
    cr = compile_spec(S(HOLD4, R("exclude", by="flag", values=["tobacco", "weapons"]),
                        R("sector_cap", sector="*", limit=0.5, unit="weight"),
                        R("portfolio_average", metric="esg", bound="min", value=1.0,
                          basis="relative_to_mean")), u)
    X = rejection_sampler(u, cr.constraint_set, 500, seed=0).samples
    assert cr.constraint_set.check_batch(X, u).all() and X[:, [1, 6, 9]].sum() == 0
    assert all(ok for x in X[:20] for _, ok in cr.check_fund(x, u))


# --------------------------------------------------------------------- data
def test_load_universe_reads_flag_columns(tmp_path):
    df = pd.DataFrame(dict(ticker=["X", "Y", "Z"], mu=[0.1, 0.2, 0.3], sector=["a", "b", "a"],
                           esg_score=[50.0, 60.0, 70.0], carbon=[1.0, 2.0, 3.0],
                           flag_tobacco=[0, 1, 0], flag_coal=[1, 1, 0], flagship=[1, 1, 1]))
    df.to_csv(tmp_path / "u.csv", index=False)
    u = load_universe(tmp_path / "u.csv")
    assert set(u.flags) == {"tobacco", "coal"}  # "flagship" has no "flag_" prefix
    assert all(v.dtype == bool and v.shape == (3,) for v in u.flags.values())
    assert u.flags["tobacco"].tolist() == [False, True, False] and u.flags["coal"].tolist() == [True, True, False]
    cr = compile_spec(S(R("exclude", by="flag", values=["Coal"])), u, k=1)
    assert cr.rules[0].status == "mapped" and excluded_of(cr.rules[0]) == [0, 1]


def test_load_universe_without_flag_columns_has_flags_none(tmp_path):
    pd.DataFrame(dict(ticker=["X", "Y"], mu=[0.1, 0.2], sector=["a", "b"], esg_score=[50.0, 60.0],
                      carbon=[1.0, 2.0])).to_csv(tmp_path / "u.csv", index=False)
    assert load_universe(tmp_path / "u.csv").flags is None
    assert synthetic_universe(5, 2).flags is None


@pytest.mark.parametrize("rule", [
    dict(kind="sector_cap", sector="*", limit=2.7, unit="names"),
    dict(kind="group_limit", by="flag", values=["tobacco"], bound="min", limit=1.5, unit="names"),
])
def test_validate_rejects_fractional_name_counts(rule):
    with pytest.raises(ValueError, match="whole number"):
        validate_spec({"rules": [dict(source="", note="", **rule)]})

