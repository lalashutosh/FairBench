"""Evidence-backed constraint extraction: rule spec + source document -> canonical constraints.

    document text --fairbench.mandate.extract_rules (the one model call)--> rule spec
    rule spec + parsed document --constraints_from_spec (deterministic)--> [Constraint]

The model proposes rules with a quoted source sentence; everything that decides whether a
rule can be trusted is checked here in code:

    quote check    the quoted sentence must be in the document. If it is not, the rule is
                   kept with status "unknown" and is never enforced.
    number check   every number in the rule must appear in its quoted sentence (as written,
                   as a percentage or as a simple fraction). A threshold that is not in the
                   quote is treated as invented: the value is removed and the status is
                   "unknown".
    wording        hard or soft comes from ``modality.classify`` on the document's own text.
    observability  "observable" if the universe has the data the rule needs, "proxy" if that
                   data is a stated stand-in, "unobservable" if it is not available. An
                   unobservable rule is recorded and FairBench cannot reproduce it.
    dates          an effective date is stored only when the caller supplies it together
                   with where it came from; none is ever inferred.

``confidence`` is a rule-based score from these checks (see ``_confidence``). It is not a
probability estimated from data, and no rule becomes "approved" here: that takes a named
reviewer (``dsl.approve``).
"""
from __future__ import annotations

import hashlib
import re

from ..data import Universe
from ..ingest.documents import ParsedDocument, count_quote, locate_quote, number_appears, parse_text
from ..rules import _match, validate_spec
from .dsl import Constraint
from .modality import Modality, classify


def _rule_numbers(r: dict) -> list[float]:
    """The numbers a rule asserts, each of which must be found in its quote."""
    kind = r["kind"]
    if kind == "holdings":
        return [r["min"], r["max"]]
    if kind == "position_cap":
        return [r["max_weight"]]
    if kind in ("sector_cap", "group_limit"):
        return [r["limit"]]
    if kind == "exclude_threshold":
        return [r["value"]]
    if kind == "threshold_share":
        return [r["value"], r["limit"]]
    if kind == "portfolio_average":
        return [r["value"]]
    if kind == "min_groups":
        return [r["min"]]
    if kind == "risk_limit":
        return [r["max"]]
    return []


# number words a quote may use for a share or a percentile ("at least half", "the worst-scoring fifth")
_WORD_FRACTIONS = {"half": 0.5, "third": 1 / 3, "quarter": 0.25, "fifth": 0.2, "tenth": 0.1}


def _word_fraction(value: float, text: str) -> bool:
    low = text.casefold()
    return any(re.search(rf"\b{w}\b", low) and (abs(value - f) < 1e-9 or abs(value - 100 * f) < 1e-9)
               for w, f in _WORD_FRACTIONS.items())


def _grounded(value: float, r: dict, text: str) -> bool:
    """Is this number in the quote? A level given relative to a benchmark is written as a
    difference ("30% below" for 0.7), and "better than the benchmark" (1.0) states no
    number at all."""
    if number_appears(value, text) or _word_fraction(value, text):
        return True
    if r.get("basis") == "relative_to_mean" and value == r.get("value"):
        return value == 1.0 or number_appears(abs(1.0 - value), text)
    return False


def _field_status(u: Universe, r: dict, proxies: frozenset[str]) -> tuple[str, str]:
    """(observability, reason) for the data a rule needs."""
    kind = r["kind"]
    needed: list[tuple[str, bool]] = []  # (name, present)
    if kind in ("exclude_threshold", "threshold_share", "portfolio_average"):
        hit, _ = _match([r["metric"]], u.numeric_names)
        needed.append((hit[0] if hit else r["metric"], bool(hit)))
    if kind in ("exclude", "require", "group_limit"):
        hit, _ = _match([r["by"]], ["flag", "ticker"] + u.label_names)
        present = bool(hit)
        names = [r["by"]]
        if present and hit[0] == "flag":
            found, missing = _match(r["values"], sorted(u.flags or {}))
            present, names = not missing and bool(found), [f"flag:{v}" for v in r["values"]]
        needed += [(n, present) for n in names]
    if kind == "sector_cap":
        needed.append(("sector", True))
    if kind == "min_groups":
        hit, _ = _match([r["category"]], u.label_names)
        needed.append((hit[0] if hit else r["category"], bool(hit)))
    if kind == "risk_limit":
        needed.append(("covariance", True))
    if kind == "unmapped":
        return "unobservable", r["reason"]
    absent = [n for n, ok in needed if not ok]
    if absent:
        return "unobservable", "no public data for: " + ", ".join(absent)
    proxy = [n for n, _ in needed if n in proxies or n.removeprefix("flag:") in proxies]
    if kind == "risk_limit" and "covariance" not in proxy:
        proxy.append("covariance")  # a risk model is always an estimate
    if proxy:
        return "proxy", "rests on stand-in data: " + ", ".join(proxy)
    return "observable", ""


def _confidence(found: bool, occurrences: int, grounded: bool, m: Modality) -> float:
    """Rule-based score: 0 without the quote; 0.3 if a number is not in the quote; else 0.95
    less 0.15 for mixed hard/soft wording, 0.10 for a quote that occurs more than once
    (ambiguous location) and 0.10 for wording with no commitment or hedge at all."""
    if not found:
        return 0.0
    if not grounded:
        return 0.3
    return round(max(0.95 - 0.15 * m.mixed - 0.10 * (occurrences > 1) - 0.10 * m.unmarked, 0.0), 2)


_SHAPE = {  # kind -> (metric, aggregation, scope, operator)
    "holdings": ("holding_count", "count", "portfolio", "between"),
    "position_cap": ("position_weight", "per_security", "security", "<="),
    "exclude": ("eligibility", "per_security", "security", "not_in"),
    "require": ("eligibility", "per_security", "security", "in"),
    "min_groups": ("distinct_groups", "count", "portfolio", ">="),
    "unmapped": ("other", "none", "portfolio", "n/a"),
}


def _describe(r: dict) -> dict:
    """The descriptive fields of a Constraint for one spec rule."""
    kind = r["kind"]
    bound_op = {"min": ">=", "max": "<="}
    if kind in _SHAPE:
        metric, agg, scope, op = _SHAPE[kind]
        d = dict(metric=metric, aggregation=agg, scope=scope, operator=op)
        if kind == "holdings":
            d.update(value=[r["min"], r["max"]], unit="names")
        elif kind == "position_cap":
            d.update(value=r["max_weight"], unit="fraction_of_portfolio")
        elif kind in ("exclude", "require"):
            d.update(group_by=r["by"], group_values=list(r["values"]))
        elif kind == "min_groups":
            d.update(value=r["min"], unit="groups", group_by=r["category"])
        return d
    if kind == "sector_cap":
        names = r["unit"] == "names"
        return dict(metric="group_count" if names else "group_weight", aggregation="count" if names else "weight_sum",
                    scope="group", operator="<=", value=r["limit"], unit="names" if names else "fraction_of_portfolio",
                    group_by="sector", group_values=[r["sector"] or "*"])
    if kind == "group_limit":
        names = r["unit"] == "names"
        return dict(metric="group_count" if names else "group_weight", aggregation="count" if names else "weight_sum",
                    scope="group", operator=bound_op[r["bound"]], value=r["limit"],
                    unit="names" if names else "fraction_of_portfolio", group_by=r["by"], group_values=list(r["values"]))
    ref = {"absolute": "absolute", "percentile": "percentile_of_universe",
           "relative_to_mean": "relative_to_benchmark"}
    special = {"esg": "esg_score", "carbon": "carbon_intensity"}
    if kind == "exclude_threshold":  # "below the level is excluded" = eligible only at or above it
        return dict(metric="eligibility", aggregation="per_security", scope="security",
                    operator=">=" if r["side"] == "below" else "<=", value=r["value"], reference=ref[r["basis"]],
                    field_name=r["metric"], group_by="sector" if r["scope"] == "sector" else None)
    if kind == "threshold_share":
        names = r["unit"] == "names"
        return dict(metric="group_count" if names else "group_weight", aggregation="count" if names else "weight_sum",
                    scope="group", operator=bound_op[r["bound"]], value=r["limit"],
                    unit="names" if names else "fraction_of_portfolio", field_name=r["metric"], reference=ref[r["basis"]],
                    group_by=f"{r['metric']} {r['side']} {r['value']:g} ({r['basis']})")
    if kind == "portfolio_average":
        return dict(metric=special.get(r["metric"].casefold(), "numeric_field"), aggregation="weighted_average",
                    scope="portfolio", operator=bound_op[r["bound"]], value=r["value"], reference=ref[r["basis"]],
                    field_name=r["metric"])
    if kind == "risk_limit":
        return dict(metric=r["measure"], aggregation="quadratic_form", scope="portfolio", operator="<=",
                    value=r["max"], unit="annual_fraction")
    raise ValueError(f"unknown rule kind {kind!r}")


def constraints_from_spec(spec: dict, doc: ParsedDocument, u: Universe, *, fund_id: str,
                          mandate_version_id: str | None = None, source_url: str | None = None,
                          retrieved_at: str | None = None, proxies: frozenset[str] | set[str] = frozenset(),
                          effective_from: str | None = None, effective_date_basis: str | None = None,
                          extraction_method: str = "llm+quote_check+number_check") -> list[Constraint]:
    """Turn a rule spec into canonical constraints, checking each rule against ``doc``.

    proxies: names of universe fields or flags that are stand-ins for the data the mandate
    really refers to (for example a SIC-code exclusion flag standing in for a revenue test).
    effective_from / effective_date_basis: only if a date is known; both or neither.
    """
    validate_spec(spec)
    if bool(effective_from) != bool(effective_date_basis):
        raise ValueError("give effective_from together with effective_date_basis, or neither")
    proxies = frozenset(proxies)
    out: list[Constraint] = []
    for i, r in enumerate(spec["rules"]):
        loc = locate_quote(doc, r["source"]) if r["source"].strip() else None
        text = loc.text if loc else r["source"]
        modality = classify(text)
        ungrounded = [v for v in _rule_numbers(r) if not _grounded(v, r, text)] if loc else []
        observability, why = _field_status(u, r, proxies)
        desc = _describe(r)
        notes = [r["note"]] if r.get("note") else []
        status, target = "disclosed", (None if r["kind"] == "unmapped" else dict(r))
        if loc is None:
            status, target = "unknown", None
            notes.append("quote not found in the source document")
        elif ungrounded:
            status, target = "unknown", None
            desc["value"] = None
            notes.append("value(s) not found in the quoted evidence, treated as not established: "
                         + ", ".join(f"{v:g}" for v in ungrounded))
        elif observability == "proxy":
            status = "derived"
        if why:
            notes.append(why)
        digest = hashlib.sha256(f"{fund_id}|{mandate_version_id}|{i}|{r['kind']}|{r['source']}".encode()).hexdigest()
        out.append(Constraint(
            constraint_id=f"c_{digest[:12]}", fund_id=fund_id, mandate_version_id=mandate_version_id,
            constraint_type=modality.constraint_type, observability=observability, status=status,
            evidence_text=text, modality_terms=modality.terms, regulatory_basis=modality.regulatory_basis,
            confidence=_confidence(loc is not None, count_quote(doc, r["source"]) if loc else 0,
                                   not ungrounded, modality),
            source_url=source_url, source_document=doc.sha256_of_source,
            source_locator=loc.as_string() if loc else None, retrieved_at=retrieved_at,
            extraction_method=extraction_method, effective_from=effective_from,
            effective_date_basis=effective_date_basis, compile_target=target, note=" ".join(notes), **desc))
    return out


def extract_constraints(doc: ParsedDocument | str, u: Universe, *, fund_id: str, client=None,
                        **kwargs) -> tuple[list[Constraint], dict]:
    """One model call (through ``fairbench.mandate.extract_rules``, the only place a model is
    called) and then ``constraints_from_spec``. Returns (constraints, spec)."""
    from ..mandate import extract_rules

    if isinstance(doc, str):
        doc = parse_text(doc)
    extraction = extract_rules(doc.text, u, client=client)
    return constraints_from_spec(extraction.spec, doc, u, fund_id=fund_id, **kwargs), extraction.spec
