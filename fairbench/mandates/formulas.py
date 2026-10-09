"""The formula sheet: every rule of a mandate as mathematics, with its evidence and its fate.

    documents --extract--> constraints --regional check--> constraints + findings
              --formula_sheet--> one document that says, rule by rule:
                  the sentence it came from and where,
                  how binding the wording is and whether the data exists,
                  the formula on a selection x (which stocks) and on weights w (how much),
                  the numbers for THIS universe and portfolio size,
                  how the quantum oracle can encode it (exact count / quantised sum / not at all),
                  and whether it is enforced, with the reason when it is not.

It is the hand-over between the language layer and the numerical layers: a person can
review it, and the samplers and the oracle consume exactly the rules it marks as enforced.
The last section gives the share of random portfolios that satisfy the enforced rules,
which is the number that decides whether a quantum estimator has any tightness to exploit.
"""
from __future__ import annotations

from ..constraints import (AvgBound, CarbonCap, Cardinality, CountBound, Exclusion, MinESG, MinGroups, SectorCap,
                           TrackingErrorCap, VolatilityCap)
from ..data import Universe
from ..rules import compile_spec, feasible_fraction
from .compile import select
from .dsl import Constraint

# kind -> (formula on the 0/1 selection x with k holdings, formula on weights w)
FORMULAS: dict[str, tuple[str, str]] = {
    "holdings": ("Σ_i x_i = k", "k_min ≤ #{i : w_i > 0} ≤ k_max"),
    "position_cap": ("1/k ≤ c  (every holding has weight 1/k)", "w_i ≤ c for every holding"),
    "sector_cap": ("Σ_{i∈s} x_i ≤ ⌊c·k⌋ for each sector s", "Σ_{i∈s} w_i ≤ c for each sector s"),
    "exclude": ("x_i = 0 for every i in the excluded group", "w_i = 0 for every i in the excluded group"),
    "require": ("x_i = 0 for every i outside the required group", "w_i = 0 for every i outside the required group"),
    "exclude_threshold": ("x_i = 0 for every i beyond the level", "w_i = 0 for every i beyond the level"),
    "group_limit": ("Σ_{i∈G} x_i ≥ ⌈c·k⌉  (minimum)  or  ≤ ⌊c·k⌋  (maximum)", "Σ_{i∈G} w_i ≥ c  or  ≤ c"),
    "threshold_share": ("Σ_{i∈G(level)} x_i ≥ ⌈c·k⌉  or  ≤ ⌊c·k⌋", "Σ_{i∈G(level)} w_i ≥ c  or  ≤ c"),
    "portfolio_average": ("Σ_i v_i x_i ≥ k·τ  (floor)  or  ≤ k·τ  (ceiling)", "Σ_i w_i v_i ≥ τ  or  ≤ τ"),
    "min_groups": ("#{groups g : Σ_{i∈g} x_i ≥ 1} ≥ m", "#{groups g : Σ_{i∈g} w_i > 0} ≥ m"),
    "risk_limit": ("(x/k − b)ᵀ Σ (x/k − b) ≤ τ²   (quadratic)", "(w − b)ᵀ Σ (w − b) ≤ τ²   (quadratic)"),
}
_EXACT = (Cardinality, Exclusion, SectorCap, CountBound)
_QUANTISED = (MinESG, CarbonCap, AvgBound)
_NOT_ENCODABLE = (MinGroups, VolatilityCap, TrackingErrorCap)


def quantum_encoding(compiled_constraints) -> str:
    """How ``ft.oracle`` can hold the compiled form of one rule."""
    cons = list(compiled_constraints)
    if not cons:
        return "nothing to encode"
    if any(isinstance(c, _NOT_ENCODABLE) for c in cons):
        return "not encodable (not a threshold on a sum, or quadratic): enforced by a classical check"
    if any(isinstance(c, _QUANTISED) for c in cons):
        return "encodable as a weighted sum with integer-quantised coefficients"
    if all(isinstance(c, _EXACT) for c in cons):
        return "encodable exactly (a count)"
    return "not encodable: enforced by a classical check"


def formula_sheet(constraints: list[Constraint], u: Universe, k: int, *, fund_name: str = "",
                  regional_findings: list[dict] | None = None, promote_soft: bool = False,
                  require_approved: bool = True, seed: int | None = 0) -> str:
    """Markdown formula sheet for ``constraints`` on universe ``u`` at ``k`` holdings.

    Enforcement follows the same policy as ``compile.select`` (hard rules only unless
    ``promote_soft``; reviewed rules only unless ``require_approved`` is False), so the
    sheet and the samplers cannot disagree about what is applied."""
    sel = select(constraints, promote_soft=promote_soft, require_approved=require_approved)
    reason = {c.constraint_id: why for c, why in sel.not_enforced}
    lines = [f"# Formula sheet{': ' + fund_name if fund_name else ''}", "",
             f"Universe of {u.n} securities, portfolio size k = {k}. {len(sel.enforced)} of {len(constraints)} rules "
             f"are enforced (policy: {'hard and soft' if promote_soft else 'hard'} wording, "
             f"{'reviewed rules only' if require_approved else 'drafts accepted'}).", "",
             "Notation: x_i = 1 if security i is held, else 0. w_i = its weight. Σ = sum. k = number of holdings.", ""]
    for n, c in enumerate(constraints, 1):
        t = c.compile_target
        kind = t["kind"] if t else None
        lines += [f"## Rule {n}: {c.metric.replace('_', ' ')}" + (f" ({kind})" if kind else ""), "",
                  f"> {c.evidence_text}", "",
                  f"- **Source:** {c.source_locator or 'not located'}" + (f", {c.source_url}" if c.source_url else ""),
                  f"- **Wording:** {c.constraint_type}" + (f" ({', '.join(c.modality_terms)})" if c.modality_terms else "")
                  + (f"; statutory basis: {c.regulatory_basis}" if c.regulatory_basis else ""),
                  f"- **Value:** {c.status}; data: {c.observability}; confidence {c.confidence:.2f}; review: {c.review_state}"]
        if kind in FORMULAS:
            on_x, on_w = FORMULAS[kind]
            compiled = compile_spec({"fund_name": "", "rules": [dict(t)]}, u, k=k).rules[0]
            lines += [f"- **Formula on the selection:** {on_x}", f"- **Formula on weights:** {on_w}",
                      f"- **With this universe's numbers:** {compiled.detail}",
                      f"- **Quantum oracle:** {quantum_encoding(compiled.constraints)}"]
        else:
            lines.append("- **Formula:** none. This rule has no numeric form here.")
        why = reason.get(c.constraint_id)
        lines += [f"- **Enforced:** {'yes' if why is None else 'no: ' + why}"] + ([f"- **Note:** {c.note}"] if c.note else []) + [""]

    if regional_findings:
        lines += ["## Regional requirements", "",
                  "| Requirement | Against the fund's own rules | Detail | Text re-checked |", "|---|---|---|---|"]
        lines += [f"| {f['title']} | {f['verdict']} | {f['why']} | {'yes' if f['verified'] else 'no'} |" for f in regional_findings]
        lines.append("")

    lines += ["## What the enforced rules leave", ""]
    spec = sel.spec(fund_name)
    if not any(r["kind"] == "holdings" for r in spec["rules"]):
        spec["rules"].insert(0, {"kind": "holdings", "min": k, "max": k, "source": "", "note": "portfolio size used for this run"})
    compiled = compile_spec(spec, u, k=k)
    frac, lo, hi = feasible_fraction(u, compiled.constraint_set, seed=seed)
    lines += [f"- Of random {k}-security portfolios drawn from the securities no exclusion forbids, **{frac:.2%}** satisfy "
              f"every enforced rule (95% interval {lo:.2%} to {hi:.2%}).",
              "- That share is what a sampler pays for: classical rejection needs about 1/share draws per valid "
              "portfolio, a quantum estimator about 1/√share queries. Near 100% there is nothing to gain from rule "
              "tightness, only from precision.",
              f"- Rules the quantum oracle cannot hold: "
              + (", ".join(sorted({cr.rule['kind'] for cr in compiled.rules if cr.constraints
                                   and 'not encodable' in quantum_encoding(cr.constraints)})) or "none") + "."]
    lines += [f"- Warning: {w}" for w in compiled.warnings]
    return "\n".join(lines) + "\n"
