"""Compile canonical constraints into what the samplers enforce.

    constraints --select(policy)--> enforced / not enforced (each with the reason)
    enforced --to_constraint_set--> ConstraintSet for Stage 1 (equal-weight subsets),
                                    through the existing ``rules.compile_spec``
    enforced --to_weight_rules--> WeightRuleSet for Stage 2 (actual weights)

The same mandate sentence compiles differently in the two stages. "No more than 25% in one
industry" is a count of names when every holding has weight 1/k and a sum of weights when
weights are free; "tracking error below 2%" is a quadratic form in both. Nothing is
dropped: a rule that is not enforced is returned with the reason.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constraints import (ConstraintSet, HoldingsRange, PositionBound, WeightedAverage, WeightedRisk,
                           WeightRuleSet, WeightSum)
from ..data import Universe
from ..rules import (CompiledRules, _group, _group_mask, _is_each, _match, _threshold_mask, compile_spec)
from .dsl import Constraint

# rule kinds that restrict WHICH assets may be held (names), independent of their weights
_SELECTION = ("exclude", "require", "exclude_threshold", "min_groups")


@dataclass
class SelectedMandate:
    """enforced: the constraints a run applies. not_enforced: (constraint, reason) for the rest."""
    enforced: list[Constraint]
    not_enforced: list[tuple[Constraint, str]]
    policy: dict

    def spec(self, fund_name: str = "") -> dict:
        """The ``fairbench.rules`` spec of the enforced constraints."""
        return {"fund_name": fund_name, "rules": [dict(c.compile_target) for c in self.enforced]}

    def report(self) -> str:
        lines = [f"{len(self.enforced)} constraints enforced, {len(self.not_enforced)} not enforced "
                 f"(policy: {self.policy})"]
        for c in self.enforced:
            lines.append(f"  [enforced    ] {c.constraint_type:<10} {c.metric:<16} {c.status:<9} "
                         f"{c.observability:<12} conf {c.confidence:.2f}  \"{c.evidence_text[:80]}\"")
        for c, why in self.not_enforced:
            lines.append(f"  [not enforced] {c.constraint_type:<10} {c.metric:<16} {why}  \"{c.evidence_text[:80]}\"")
        return "\n".join(lines)


def select(constraints: list[Constraint], *, promote_soft: bool = False, require_approved: bool = True,
           allow_proxy: bool = True) -> SelectedMandate:
    """Split constraints into enforced and not enforced under a stated policy. Running once
    with promote_soft=False and once with True is the hard-versus-soft sensitivity check."""
    policy = dict(promote_soft=promote_soft, require_approved=require_approved, allow_proxy=allow_proxy)
    enforced, rest = [], []
    for c in constraints:
        why = c.why_not_enforced(**policy)
        (enforced.append(c) if why is None else rest.append((c, why)))
    return SelectedMandate(enforced, rest, policy)


def to_constraint_set(selected: SelectedMandate, u: Universe, k: int | None = None) -> CompiledRules:
    """Stage 1: the enforced constraints as a ConstraintSet for equal-weight k-subsets, via
    the existing rule compiler (so its per-rule report still applies)."""
    return compile_spec(selected.spec(), u, k=k)


def _values(u: Universe, name: str) -> tuple[str, np.ndarray] | None:
    hit, _ = _match([name], u.numeric_names)
    return (hit[0], u.numeric(hit[0])) if hit else None


def _level(vals: np.ndarray, value: float, basis: str, benchmark: np.ndarray | None) -> tuple[float, str]:
    """Threshold on the data's scale; missing values are left out, never read as zero."""
    known = ~np.isnan(vals)
    if basis == "absolute":
        return float(value), "absolute"
    if basis == "percentile":
        return float(np.percentile(vals[known], value)), f"percentile {value:g} of {int(known.sum())} covered assets"
    if benchmark is not None and benchmark[known].sum() > 0:
        ref = float(benchmark[known] @ vals[known] / benchmark[known].sum())
        return float(value * ref), f"{value:g} x benchmark-weighted average {ref:.4g}"
    ref = float(vals[known].mean())
    return float(value * ref), f"{value:g} x equal-weight universe average {ref:.4g}"


def to_weight_rules(selected: SelectedMandate, u: Universe, *, benchmark: np.ndarray | None = None,
                    missing: str = "ineligible") -> tuple[WeightRuleSet, list[str]]:
    """Stage 2: the enforced constraints as rules on actual weights.

    benchmark: (n,) benchmark weights. Tracking error and "relative to the benchmark" levels
    use them; without them the equal-weight universe stands in and the note says so.
    missing: how a weighted average treats assets with no value (see ``WeightedAverage``).
    Returns (rules, notes); a note is written for every rule, including any that could not
    be built, so nothing is dropped silently.
    """
    rules: list = []
    notes: list[str] = []
    selection: list[dict] = []
    sector_arr = np.asarray(u.sector, dtype=str)
    b = None if benchmark is None else np.asarray(benchmark, dtype=float)

    for c in selected.enforced:
        r, tag = c.compile_target, f"{c.constraint_id} ({c.compile_target['kind']})"
        kind = r["kind"]
        if kind in _SELECTION or r.get("unit") == "names":
            selection.append(r)
            notes.append(f"{tag}: selection rule on the set of held assets")
        elif kind == "holdings":
            rules.append(HoldingsRange(r["min"], r["max"]))
            notes.append(f"{tag}: {r['min']}-{r['max']} holdings")
        elif kind == "position_cap":
            rules.append(PositionBound(0.0, r["max_weight"]))
            notes.append(f"{tag}: each position <= {r['max_weight']:.2%}")
        elif kind == "sector_cap":
            every = r["sector"].strip() == "" or _is_each([r["sector"]])
            hit, unknown = (sorted(set(u.sector)), []) if every else _match([r["sector"]], sorted(set(u.sector)))
            if unknown or not hit:
                notes.append(f"{tag}: NOT BUILT, sector {r['sector']!r} is not in the universe")
                continue
            rules += [WeightSum(np.flatnonzero(sector_arr == s), upper=r["limit"], label=f"sector={s}") for s in hit]
            notes.append(f"{tag}: weight of {'each sector' if every else hit[0]} <= {r['limit']:.2%}")
        elif kind == "group_limit":
            g = _group(u, r["by"], r["values"])
            if g is None or not (g[3] or _is_each(r["values"])):
                notes.append(f"{tag}: NOT BUILT, no grouping {r['by']!r} / {r['values']} in the universe")
                continue
            gkind, name, labels, hit, _ = g
            lo, hi = (r["limit"], None) if r["bound"] == "min" else (None, r["limit"])
            if gkind == "label" and _is_each(r["values"]):
                for v in sorted(set(labels.tolist())):
                    rules.append(WeightSum(np.flatnonzero(labels == v), lo, hi, label=f"{name}={v}"))
                notes.append(f"{tag}: weight of each {name} {'>=' if lo is not None else '<='} {r['limit']:.2%}")
            else:
                mask = _group_mask(u, gkind, labels, hit)
                rules.append(WeightSum(np.flatnonzero(mask), lo, hi, label=f"{name} in {hit}"))
                notes.append(f"{tag}: weight of {name} in {hit} {'>=' if lo is not None else '<='} {r['limit']:.2%} "
                             f"({int(mask.sum())} assets)")
        elif kind == "threshold_share":
            t = _threshold_mask(u, r)
            if t is None:
                notes.append(f"{tag}: NOT BUILT, no numeric field {r['metric']!r} in the universe")
                continue
            lo, hi = (r["limit"], None) if r["bound"] == "min" else (None, r["limit"])
            rules.append(WeightSum(np.flatnonzero(t[0]), lo, hi, label=t[1]))
            notes.append(f"{tag}: weight of assets with {t[1]} {'>=' if lo is not None else '<='} {r['limit']:.2%}")
        elif kind == "portfolio_average":
            v = _values(u, r["metric"])
            if v is None:
                notes.append(f"{tag}: NOT BUILT, no numeric field {r['metric']!r} in the universe")
                continue
            name, vals = v
            level, how = _level(vals, r["value"], r["basis"], b)
            lo, hi = (level, None) if r["bound"] == "min" else (None, level)
            rules.append(WeightedAverage(name, lo, hi, missing=missing, label=f"avg {name}"))
            n_missing = int(np.isnan(vals).sum())
            notes.append(f"{tag}: weighted average {name} {'>=' if lo is not None else '<='} {level:.4g} [{how}]"
                         + (f"; {n_missing} assets have no value ({missing})" if n_missing else ""))
        elif kind == "risk_limit":
            if r["measure"] == "volatility":
                rules.append(WeightedRisk(r["max"], None, label="volatility"))
                notes.append(f"{tag}: volatility <= {r['max']:.2%} (quadratic; universe covariance)")
            else:
                bench = np.full(u.n, 1.0 / u.n) if b is None else b
                rules.append(WeightedRisk(r["max"], bench, label="tracking_error"))
                notes.append(f"{tag}: tracking error <= {r['max']:.2%} (quadratic) against "
                             + ("the supplied benchmark weights" if b is not None else
                                "the equal-weight universe, because no benchmark weights were given"))
        else:
            notes.append(f"{tag}: NOT BUILT, rule kind {kind!r} has no weight form")

    support = None
    if selection:
        compiled = compile_spec({"fund_name": "", "rules": selection}, u, k=1)
        support = ConstraintSet(compiled.constraint_set.non_cardinality())
        notes += [f"selection: {cr.detail}" for cr in compiled.rules]
        notes += [f"selection warning: {w}" for w in compiled.warnings if "fewer than k" not in w]
    return WeightRuleSet(rules, support_rules=support), notes
