"""Rule specs: a fund mandate as JSON, and its compilation into a ConstraintSet.

A spec is plain JSON (``SPEC_SCHEMA``) written in the mandate's own terms: "at most 15%
per sector", "average carbon 30% below the universe", "no tobacco". ``compile_spec`` turns
it into numeric constraints for one Universe and reports, rule by rule, what each became,
including the rules that could not be expressed. Nothing here calls a model:
``fairbench.mandate`` produces specs from mandate text, and a spec can also be written or
corrected by hand.

Rule kinds
    holdings            number of holdings (min, max)                   -> Cardinality
    position_cap        max weight of one holding                        -> checked against 1/k
    sector_cap          max names or weight per sector ("*" = each)     -> SectorCap
    exclude             assets in a group may not be held                -> Exclusion
    require             only assets in a group may be held               -> Exclusion of the rest
    exclude_threshold   assets below/above a level of a numeric field    -> Exclusion
    group_limit         min or max share of the portfolio in a group     -> CountBound / SectorCap
    threshold_share     min or max share held below/above a level        -> CountBound
    portfolio_average   floor or ceiling on a portfolio average          -> MinESG / CarbonCap / AvgBound
    min_groups          at least m distinct sectors, countries, ...      -> MinGroups
    risk_limit          cap on volatility or tracking error              -> VolatilityCap / TrackingErrorCap
    unmapped            a rule with no numeric form here (kept for the record)

Groups (``by``): "sector", "ticker", "flag", or the name of a category in the universe
(``Universe.categories``, e.g. "country"); ``values`` are the names within it.
Numeric fields (``metric``): "esg", "carbon" or a key of ``Universe.attributes``.
Levels (``basis``): "absolute" is on the data's own scale; "percentile" is a percentile of
the asset values (0-100); "relative_to_mean" multiplies the mean of the asset values.
Shares and averages assume equally weighted holdings, as the constraints do: a weight share
``limit`` becomes floor(limit x k) names for a maximum and ceil(limit x k) for a minimum.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .baselines import random_k_subsets
from .constraints import (AvgBound, CarbonCap, Cardinality, Constraint, ConstraintSet, CountBound,
                          Exclusion, MinESG, MinGroups, SectorCap, TrackingErrorCap, VolatilityCap)
from .data import Universe

_BASIS = ["absolute", "percentile", "relative_to_mean"]
_LEVEL = {"metric": str, "side": ["below", "above"], "value": float, "basis": _BASIS,
          "scope": ["universe", "sector"]}
_SHARE = {"bound": ["min", "max"], "limit": float, "unit": ["names", "weight"]}

# kind -> {field: type or list of allowed strings}. Every rule also has "source" and "note".
RULE_FIELDS: dict[str, dict] = {
    "holdings": {"min": int, "max": int},
    "position_cap": {"max_weight": float},
    "sector_cap": {"sector": str, "limit": float, "unit": ["names", "weight"]},
    "exclude": {"by": str, "values": [str]},
    "require": {"by": str, "values": [str]},
    "exclude_threshold": dict(_LEVEL),
    "group_limit": {"by": str, "values": [str], **_SHARE},
    "threshold_share": {**_LEVEL, **_SHARE},
    "portfolio_average": {"metric": str, "bound": ["min", "max"], "value": float, "basis": _BASIS},
    "min_groups": {"category": str, "min": int},
    "risk_limit": {"measure": ["volatility", "tracking_error"], "max": float},
    "unmapped": {"reason": str},
}


def _json_type(t) -> dict:
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is str:
        return {"type": "string"}
    if t == [str]:
        return {"type": "array", "items": {"type": "string"}}
    return {"type": "string", "enum": list(t)}


def _rule_schema(kind: str, fields: dict) -> dict:
    props = {"kind": {"type": "string", "const": kind},
             "source": {"type": "string"}, "note": {"type": "string"}}
    props.update({name: _json_type(t) for name, t in fields.items()})
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


# JSON Schema of a spec; also the structured-output schema the model is held to.
SPEC_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "fund_name": {"type": "string"},
        "rules": {"type": "array", "items": {"anyOf": [_rule_schema(k, f) for k, f in RULE_FIELDS.items()]}},
    },
    "required": ["fund_name", "rules"],
    "additionalProperties": False,
}


def _type_ok(v, t) -> bool:
    if t is int:
        return isinstance(v, int) and not isinstance(v, bool)
    if t is float:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if t is str:
        return isinstance(v, str)
    if t == [str]:
        return isinstance(v, list) and all(isinstance(x, str) for x in v)
    return v in t


def _range_errors(r: dict) -> list[str]:
    """Values that are well-typed but cannot be meant (15 for 15%, a percentile of 120)."""
    kind, out = r["kind"], []
    if kind == "holdings" and not 1 <= r["min"] <= r["max"]:
        out.append(f"need 1 <= min <= max; got {r['min']}, {r['max']}")
    if kind == "position_cap" and not 0 < r["max_weight"] <= 1:
        out.append(f"max_weight is a fraction in (0, 1]; got {r['max_weight']}")
    if "limit" in r and (r["limit"] < 0 or (r["unit"] == "weight" and r["limit"] > 1)):
        out.append(f"limit must be >= 0, and a fraction <= 1 for unit 'weight'; got {r['limit']}")
    if "limit" in r and r["unit"] == "names" and r["limit"] != int(r["limit"]):
        out.append(f"limit must be a whole number for unit 'names'; got {r['limit']}")
    if "basis" in r:
        if r["basis"] == "percentile" and not 0 <= r["value"] <= 100:
            out.append(f"a percentile is between 0 and 100; got {r['value']}")
        if r["basis"] == "relative_to_mean" and r["value"] <= 0:
            out.append(f"a multiple of the mean must be positive; got {r['value']}")
    if kind == "min_groups" and r["min"] < 1:
        out.append(f"min must be >= 1; got {r['min']}")
    if kind == "risk_limit" and not 0 < r["max"] <= 1:
        out.append(f"max is an annual fraction in (0, 1] (6% is 0.06); got {r['max']}")
    return out


def validate_spec(spec: dict) -> dict:
    """Check a spec against ``RULE_FIELDS`` and basic value ranges; raise ValueError listing
    every problem. ``fund_name`` may be omitted in hand-written specs."""
    errs: list[str] = []
    if not isinstance(spec, dict) or not isinstance(spec.get("rules"), list):
        raise ValueError("spec must be an object with a 'rules' list")
    if not isinstance(spec.get("fund_name", ""), str):
        errs.append("fund_name must be a string")
    for i, r in enumerate(spec["rules"]):
        kind = r.get("kind") if isinstance(r, dict) else None
        if not isinstance(kind, str) or kind not in RULE_FIELDS:
            errs.append(f"rule {i}: unknown kind {kind!r}")
            continue
        want = {"source": str, "note": str, **RULE_FIELDS[kind]}
        for name, t in want.items():
            if name not in r:
                errs.append(f"rule {i} ({kind}): missing '{name}'")
            elif not _type_ok(r[name], t):
                errs.append(f"rule {i} ({kind}): bad value for '{name}': {r[name]!r}")
        extra = set(r) - set(want) - {"kind"}
        if extra:
            errs.append(f"rule {i} ({kind}): unexpected fields {sorted(extra)}")
        if not extra and all(n in r and _type_ok(r[n], t) for n, t in want.items()):
            errs += [f"rule {i} ({kind}): {m}" for m in _range_errors(r)]
    if errs:
        raise ValueError("invalid rule spec:\n  " + "\n  ".join(errs))
    return spec


def load_spec(path: str | Path) -> dict:
    return validate_spec(json.loads(Path(path).read_text()))


def save_spec(spec: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(validate_spec(spec), indent=2) + "\n")


# ------------------------------------------------------------------ compile
@dataclass
class CompiledRule:
    """One spec rule and what it became. status: "mapped" (constraints added),
    "trivial" (holds by construction, nothing added) or "unmapped" (not enforced)."""
    rule: dict
    status: str
    detail: str
    constraints: list[Constraint] = field(default_factory=list)


@dataclass
class CompiledRules:
    constraint_set: ConstraintSet
    rules: list[CompiledRule]
    k: int
    warnings: list[str]

    @property
    def unmapped(self) -> list[CompiledRule]:
        return [r for r in self.rules if r.status == "unmapped"]

    def check_fund(self, x: np.ndarray, u: Universe) -> list[tuple[CompiledRule, bool]]:
        """(rule, complies) for every mapped rule, for one selection x."""
        return [(r, all(c.check(x, u) for c in r.constraints)) for r in self.rules if r.constraints]

    def report(self) -> str:
        lines = [f"{len(self.rules)} rules: {sum(r.status == 'mapped' for r in self.rules)} mapped, "
                 f"{sum(r.status == 'trivial' for r in self.rules)} trivial, "
                 f"{len(self.unmapped)} unmapped; k = {self.k}"]
        for r in self.rules:
            src = " ".join(r.rule["source"].split())
            lines.append(f"  [{r.status:<8}] {r.rule['kind']:<18} {r.detail}")
            if src:
                lines.append(f"  {'':<10} source: \"{src[:110]}{'...' if len(src) > 110 else ''}\"")
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def _level(values: np.ndarray, value: float, basis: str) -> float:
    if basis == "absolute":
        return float(value)
    if basis == "percentile":
        return float(np.percentile(values, value))
    return float(value * values.mean())


def _match(names: list[str], pool: list[str]) -> tuple[list[str], list[str]]:
    """Case-insensitive match of names against pool -> (matched pool entries, unknown names)."""
    by_fold = {p.strip().casefold(): p for p in pool}
    hit = [by_fold[n.strip().casefold()] for n in names if n.strip().casefold() in by_fold]
    return hit, [n for n in names if n.strip().casefold() not in by_fold]


def _tick(u: Universe, idx: np.ndarray, most: int = 6) -> str:
    names = [u.tickers[i] for i in idx[:most]]
    return ", ".join(names) + (f", ... ({idx.size} assets)" if idx.size > most else f" ({idx.size} assets)")


def _is_each(values: list[str]) -> bool:
    return len(values) == 1 and values[0].strip().casefold() in ("*", "each", "any", "all")


def _group(u: Universe, by: str, values: list[str]):
    """Resolve a group of assets. Returns (kind, name, labels, hit, unknown) where kind is
    "flag", "ticker" or "label"; labels is the per-asset label array for kind "label".
    Returns None when ``by`` names nothing in the universe."""
    hit_by, _ = _match([by], ["flag", "ticker"] + u.label_names)
    if not hit_by:
        return None
    name = hit_by[0]
    if name == "flag":
        pool, labels = sorted(u.flags or {}), None
    elif name == "ticker":
        pool, labels = list(u.tickers), np.asarray(u.tickers, dtype=str)
    else:
        labels = u.labels(name)
        pool = sorted(set(labels.tolist()))
    hit, unknown = _match(values, pool)
    return ("flag" if name == "flag" else "ticker" if name == "ticker" else "label"), name, labels, hit, unknown


def _group_mask(u: Universe, kind: str, labels, hit: list[str]) -> np.ndarray:
    mask = np.zeros(u.n, dtype=bool)
    if kind == "flag":
        for f in hit:
            mask |= np.asarray(u.flags[f], dtype=bool)
        return mask
    return np.isin(labels, hit)


def _threshold_mask(u: Universe, r: dict):
    """Assets below/above the rule's level -> (mask, description), or None if the metric is unknown."""
    hit, _ = _match([r["metric"]], u.numeric_names)
    if not hit:
        return None
    vals = u.numeric(hit[0])
    sector_arr = np.asarray(u.sector, dtype=str)
    groups = [sector_arr == s for s in sorted(set(u.sector))] if r["scope"] == "sector" else [np.ones(u.n, bool)]
    mask = np.zeros(u.n, dtype=bool)
    levels = []
    for g in groups:
        lv = _level(vals[g], r["value"], r["basis"])
        levels.append(lv)
        mask |= g & ((vals < lv) if r["side"] == "below" else (vals > lv))
    lv_txt = f"{levels[0]:.4g}" if len(levels) == 1 else f"{min(levels):.4g}..{max(levels):.4g} by sector"
    return mask, f"{hit[0]} {r['side']} {lv_txt} ({r['basis']} {r['value']:g})"


def _share_count(r: dict, k: int) -> tuple[int, str]:
    """Names allowed (max) or required (min) for a share rule, and how that was derived."""
    if r["unit"] == "names":
        return int(r["limit"]), f"{int(r['limit'])} names"
    if r["bound"] == "max":
        c = int(math.floor(r["limit"] * k + 1e-9))
        return c, f"floor({r['limit']:.0%} x {k}) = {c} names"
    c = int(math.ceil(r["limit"] * k - 1e-9))
    return c, f"ceil({r['limit']:.0%} x {k}) = {c} names"


def compile_spec(spec: dict, u: Universe, k: int | None = None) -> CompiledRules:
    """Turn a rule spec into a ConstraintSet for universe ``u``.

    k: number of holdings to benchmark at, normally the fund's actual count. If omitted it
    comes from the spec's ``holdings`` rule (the midpoint when that is a range, with a warning).
    """
    validate_spec(spec)
    warnings: list[str] = []
    sectors = sorted(set(u.sector))
    hold = [r for r in spec["rules"] if r["kind"] == "holdings"]
    if hold:
        lo, hi = max(r["min"] for r in hold), min(r["max"] for r in hold)
        if lo > hi:
            raise ValueError(f"holdings rules contradict each other (min {lo} > max {hi})")
        if k is None:
            k = (lo + hi) // 2
            if lo != hi:
                warnings.append(f"mandate allows {lo}-{hi} holdings; using k={k}. "
                                "Pass k= the fund's actual number of holdings.")
        elif not lo <= k <= hi:
            warnings.append(f"k={k} is outside the mandate's {lo}-{hi} holdings")
    elif k is None:
        raise ValueError("spec has no holdings rule; pass k")
    if not 0 < k < u.n:
        raise ValueError(f"need 0 < k < n; got k={k}, n={u.n}")

    out: list[CompiledRule] = []
    excluded = np.zeros(u.n, dtype=bool)

    def add(rule, status, detail, cons=()):
        out.append(CompiledRule(rule, status, detail, list(cons)))

    def exclude(rule, mask, detail):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            add(rule, "trivial", f"{detail}: no asset in this universe matches")
        else:
            excluded[idx] = True
            add(rule, "mapped", f"{detail}: excludes {_tick(u, idx)}", [Exclusion(idx)])

    def bound(rule, mask, what, label):
        """Min or max number of holdings inside ``mask``."""
        idx = np.flatnonzero(mask)
        cnt, how = _share_count(rule, k)
        if rule["bound"] == "max":
            if cnt == 0:
                return exclude(rule, mask, f"at most {how} in {what}")
            if cnt >= min(k, idx.size):
                return add(rule, "trivial", f"at most {how} in {what}: cannot bind "
                                            f"({idx.size} such assets, k={k})")
            return add(rule, "mapped", f"CountBound(at most {how} in {what}; {idx.size} such assets)",
                       [CountBound(idx, 0, cnt, label)])
        if cnt == 0:
            return add(rule, "trivial", f"at least {how} in {what}: holds for any portfolio")
        if cnt > min(k, idx.size):
            warnings.append(f"at least {cnt} names in {what} is impossible ({idx.size} such assets, k={k})")
        add(rule, "mapped", f"CountBound(at least {how} in {what}; {idx.size} such assets)",
            [CountBound(idx, cnt, None, label)])

    for r in spec["rules"]:
        kind = r["kind"]
        if kind == "holdings":
            add(r, "mapped", f"Cardinality(k={k})  [mandate: {r['min']}-{r['max']}]", [Cardinality(k)])
        elif kind == "position_cap":
            if 1.0 / k <= r["max_weight"] + 1e-12:
                add(r, "trivial", f"equal weights give 1/k = {1 / k:.1%} <= {r['max_weight']:.1%}")
            else:
                add(r, "unmapped", f"1/k = {1 / k:.1%} exceeds the cap {r['max_weight']:.1%}")
                warnings.append(f"position cap {r['max_weight']:.1%} cannot hold with k={k} equal weights")
        elif kind == "sector_cap":
            every = r["sector"].strip() == "" or _is_each([r["sector"]])
            hit, unknown = (sectors, []) if every else _match([r["sector"]], sectors)
            if unknown:
                add(r, "unmapped", f"sector {r['sector']!r} is not in the universe {sectors}")
                continue
            cap = int(r["limit"]) if r["unit"] == "names" else int(math.floor(r["limit"] * k + 1e-9))
            what = f"{cap} names" if r["unit"] == "names" else f"floor({r['limit']:.0%} x {k}) = {cap} names"
            add(r, "mapped", f"SectorCap({'each sector' if every else hit[0]}, {what})",
                [SectorCap(s, cap) for s in hit])
        elif kind in ("exclude", "require", "group_limit"):
            g = _group(u, r["by"], r["values"])
            if g is None:
                add(r, "unmapped", f"the universe has no grouping {r['by']!r} "
                                   f"(available: flag, ticker, {', '.join(u.label_names)})")
                continue
            gkind, name, labels, hit, unknown = g
            each = kind == "group_limit" and gkind == "label" and _is_each(r["values"])
            if each:
                hit, unknown = sorted(set(labels.tolist())), []
            if unknown:
                warnings.append(f"{kind} by {name}: not in the universe data: {unknown}")
            if not hit:
                add(r, "unmapped", f"no {name} named {r['values']} in the universe data")
            elif kind == "exclude":
                exclude(r, _group_mask(u, gkind, labels, hit), f"{name} in {hit}")
            elif kind == "require":
                keep = _group_mask(u, gkind, labels, hit)
                exclude(r, ~keep, f"only {name} in {hit} ({int(keep.sum())} assets eligible)")
            elif each:
                cnt, how = _share_count(r, k)
                side = "at most" if r["bound"] == "max" else "at least"
                if name == "sector" and r["bound"] == "max":
                    cons = [SectorCap(v, cnt) for v in hit]
                else:
                    cons = [CountBound(np.flatnonzero(labels == v), cnt if r["bound"] == "min" else 0,
                                       cnt if r["bound"] == "max" else None, f"{name}={v}") for v in hit]
                add(r, "mapped", f"{side} {how} in each {name} ({len(hit)} of them)", cons)
            else:
                what = ("assets flagged " if gkind == "flag" else f"{name} ") + " / ".join(hit)
                bound(r, _group_mask(u, gkind, labels, hit), what, what)
        elif kind in ("exclude_threshold", "threshold_share"):
            t = _threshold_mask(u, r)
            if t is None:
                add(r, "unmapped", f"the universe has no numeric field {r['metric']!r} "
                                   f"(available: {', '.join(u.numeric_names)})")
            elif kind == "exclude_threshold":
                exclude(r, *t)
            else:
                bound(r, t[0], f"assets with {t[1]}", t[1])
        elif kind == "portfolio_average":
            hit, _ = _match([r["metric"]], u.numeric_names)
            if not hit:
                add(r, "unmapped", f"the universe has no numeric field {r['metric']!r} "
                                   f"(available: {', '.join(u.numeric_names)})")
                continue
            vals = u.numeric(hit[0])
            lv = _level(vals, r["value"], r["basis"])
            how = f"{r['basis']} {r['value']:g}; universe mean {vals.mean():.4g}"
            if (hit[0], r["bound"]) == ("esg", "min"):
                add(r, "mapped", f"MinESG(average >= {lv:.4g})  [{how}]", [MinESG(lv)])
            elif (hit[0], r["bound"]) == ("carbon", "max"):
                add(r, "mapped", f"CarbonCap(average <= {lv:.4g})  [{how}]", [CarbonCap(lv)])
            elif r["bound"] == "min":
                add(r, "mapped", f"AvgBound(average {hit[0]} >= {lv:.4g})  [{how}]", [AvgBound(hit[0], lower=lv)])
            else:
                add(r, "mapped", f"AvgBound(average {hit[0]} <= {lv:.4g})  [{how}]", [AvgBound(hit[0], upper=lv)])
        elif kind == "min_groups":
            hit, _ = _match([r["category"]], u.label_names)
            if not hit:
                add(r, "unmapped", f"the universe has no category {r['category']!r} "
                                   f"(available: {', '.join(u.label_names)})")
                continue
            have = len(set(u.labels(hit[0]).tolist()))
            if r["min"] > min(have, k):
                warnings.append(f"at least {r['min']} distinct {hit[0]} values is impossible "
                                f"({have} in the universe, k={k})")
            if r["min"] <= 1:
                add(r, "trivial", f"at least {r['min']} {hit[0]}: holds for any portfolio")
            else:
                add(r, "mapped", f"MinGroups(at least {r['min']} distinct {hit[0]} values; "
                                 f"{have} in the universe)", [MinGroups(hit[0], r["min"])])
        elif kind == "risk_limit":
            if r["measure"] == "volatility":
                add(r, "mapped", f"VolatilityCap(annual volatility <= {r['max']:.1%}; equal weights, universe covariance)",
                    [VolatilityCap(r["max"])])
            else:
                add(r, "mapped", f"TrackingErrorCap(tracking error <= {r['max']:.1%} against the "
                                 "equal-weight universe)", [TrackingErrorCap(r["max"])])
        else:
            add(r, "unmapped", r["reason"])

    cons: list[Constraint] = [Cardinality(k)]
    cons += [c for r in out for c in r.constraints if not isinstance(c, Cardinality)]
    left = int(u.n - excluded.sum())
    if left < k:
        warnings.append(f"exclusions leave {left} assets, fewer than k={k}: the rule set is infeasible")
    return CompiledRules(ConstraintSet(cons), out, k, warnings)


def feasible_fraction(u: Universe, cs: ConstraintSet, draws: int = 50_000, seed: int | None = 0,
                      chunk: int = 10_000) -> tuple[float, float, float]:
    """Monte Carlo share of random k-subsets of the non-excluded assets that satisfy every
    rule, with a 95% Wilson interval: (estimate, low, high). 0 hits means "below ~3/draws"."""
    k = cs.cardinality
    allowed = cs.allowed_indices(u.n)
    if k is None or allowed.size < k:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, draws, chunk):
        m = min(chunk, draws - start)
        X = np.zeros((m, u.n), dtype=np.uint8)
        X[:, allowed] = random_k_subsets(allowed.size, k, m, int(rng.integers(2**31)))
        hits += int(cs.check_batch(X, u).sum())
    p, z = hits / draws, 1.96
    centre = (p + z * z / (2 * draws)) / (1 + z * z / draws)
    half = z * math.sqrt(p * (1 - p) / draws + z * z / (4 * draws * draws)) / (1 + z * z / draws)
    return p, (max(centre - half, 0.0) if hits else 0.0), min(centre + half, 1.0)
