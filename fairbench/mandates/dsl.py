"""The canonical constraint record: one mandate rule with its evidence.

A ``Constraint`` says what a rule restricts (metric, aggregation, operator, value, unit,
scope, group), how binding its wording is (``constraint_type``), whether the data it needs
exists (``observability``), where it came from (URL, document, locator, verbatim text,
dates, method, confidence) and how far the value can be trusted (``status``):

    disclosed   the value is printed in the cited source
    derived     computed from disclosed values by a stated method
    assumed     chosen by FairBench or a reviewer; the note says why
    unknown     not established; the rule is recorded and never enforced

``compile_target`` holds the executable form, a rule of the ``fairbench.rules`` spec, or
None when the rule has no numeric form. A constraint is ENFORCEABLE only when all of these
hold: it has a compile target, its status is not "unknown", its data is not "unobservable",
it is hard (or the caller explicitly promotes soft rules), and it has been approved (or the
caller explicitly accepts drafts). ``why_not_enforced`` names the first failing condition,
so a report can say exactly why each rule was left out.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

METRICS = ("holding_count", "position_weight", "group_weight", "group_count", "eligibility",
           "esg_score", "carbon_intensity", "numeric_field", "distinct_groups", "tracking_error",
           "volatility", "turnover", "other")
AGGREGATIONS = ("per_security", "count", "weight_sum", "weighted_average", "equal_average",
                "quadratic_form", "none")
OPERATORS = ("<=", ">=", "==", "between", "in", "not_in", "n/a")
SCOPES = ("security", "portfolio", "group")
REFERENCES = ("absolute", "relative_to_benchmark", "percentile_of_universe")
CONSTRAINT_TYPES = ("hard", "soft", "regulatory")
OBSERVABILITY = ("observable", "proxy", "unobservable")
STATUSES = ("disclosed", "derived", "assumed", "unknown")
REVIEW_STATES = ("draft", "reviewed", "approved", "rejected")

_ENUMS = {"metric": METRICS, "aggregation": AGGREGATIONS, "operator": OPERATORS, "scope": SCOPES,
          "reference": REFERENCES, "constraint_type": CONSTRAINT_TYPES, "observability": OBSERVABILITY,
          "status": STATUSES, "review_state": REVIEW_STATES}


@dataclass
class Constraint:
    constraint_id: str
    fund_id: str
    metric: str
    operator: str
    constraint_type: str
    observability: str
    status: str
    evidence_text: str
    value: float | list | None = None
    unit: str | None = None
    aggregation: str = "none"
    scope: str = "portfolio"
    reference: str = "absolute"
    field_name: str | None = None        # the universe field the metric reads, e.g. "esg", "carbon"
    group_by: str | None = None
    group_values: list[str] | None = None
    weight_basis: str | None = None      # "net_assets" | "equity_sleeve" | "names"
    benchmark_id: str | None = None
    mandate_version_id: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    effective_date_basis: str | None = None  # where effective_from came from; None if no date is known
    modality_terms: list[str] = field(default_factory=list)
    regulatory_basis: str | None = None
    confidence: float = 0.0
    source_url: str | None = None
    source_document: str | None = None   # document id or sha256
    source_locator: str | None = None    # page / paragraph / table row
    retrieved_at: str | None = None
    extraction_method: str | None = None
    review_state: str = "draft"
    compile_target: dict | None = None
    note: str = ""

    def __post_init__(self):
        errs = [f"{name}={getattr(self, name)!r} is not one of {allowed}"
                for name, allowed in _ENUMS.items() if getattr(self, name) not in allowed]
        if not 0.0 <= self.confidence <= 1.0:
            errs.append(f"confidence {self.confidence} is outside [0, 1]")
        if self.effective_from and not self.effective_date_basis:
            errs.append("effective_from is set without effective_date_basis: a date must say where it came from")
        if self.status == "disclosed" and not self.evidence_text.strip():
            errs.append('status "disclosed" needs evidence_text')
        if errs:
            raise ValueError(f"invalid constraint {self.constraint_id}: " + "; ".join(errs))

    def why_not_enforced(self, *, promote_soft: bool = False, require_approved: bool = True,
                         allow_proxy: bool = True) -> str | None:
        """None if the rule may be enforced under this policy, else the reason it may not."""
        if self.compile_target is None:
            return "no numeric form"
        if self.status == "unknown":
            return "value not established (status unknown)"
        if self.observability == "unobservable":
            return "the data it needs is not public (unobservable)"
        if self.observability == "proxy" and not allow_proxy:
            return "rests on proxy data, and proxies were not allowed"
        if self.review_state == "rejected":
            return "rejected in review"
        if self.constraint_type == "soft" and not promote_soft:
            return "soft wording (" + (", ".join(self.modality_terms) or "no commitment wording") + ")"
        if require_approved and self.review_state != "approved":
            return f"not approved (review state {self.review_state})"
        return None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "Constraint":
        return cls(**d)

    def to_row(self, document_id: int | None = None, extraction_run_id: int | None = None,
               evidence_id: int | None = None) -> dict:
        """Mapping for ``store.db.add_constraint`` (the columns of the ``constraints`` table).
        Fields the table has no column for are folded into ``note`` as JSON."""
        if self.mandate_version_id is None:
            raise ValueError("a constraint needs a mandate_version_id to be stored")
        extra = {k: v for k, v in (("field_name", self.field_name), ("regulatory_basis", self.regulatory_basis),
                                   ("effective_date_basis", self.effective_date_basis)) if v}
        note = self.note + (" " if self.note and extra else "") + (json.dumps(extra, sort_keys=True) if extra else "")
        return dict(
            constraint_id=self.constraint_id, mandate_version_id=self.mandate_version_id, metric=self.metric,
            aggregation=self.aggregation, operator=self.operator, value=self.value, unit=self.unit,
            reference=self.reference, benchmark_id=self.benchmark_id, scope=self.scope, group_by=self.group_by,
            group_values=self.group_values, weight_basis=self.weight_basis, constraint_type=self.constraint_type,
            modality_terms=self.modality_terms, observability=self.observability,
            effective_from=self.effective_from, effective_to=self.effective_to, review_state=self.review_state,
            compile_target=None if self.compile_target is None else json.dumps(self.compile_target, sort_keys=True),
            note=note, evidence_id=evidence_id, document_id=document_id, locator=self.source_locator,
            extraction_run_id=extraction_run_id, confidence=self.confidence, status=self.status)


def dump_constraints(constraints: list[Constraint]) -> str:
    return json.dumps([c.to_dict() for c in constraints], indent=2, sort_keys=True) + "\n"


def load_constraints(text: str) -> list[Constraint]:
    return [Constraint.from_dict(d) for d in json.loads(text)]


def approve(constraints: list[Constraint], reviewer: str, ids: list[str] | None = None,
            promote: list[str] | None = None) -> list[Constraint]:
    """Record a human review. ``ids`` (default: all that are not unknown/rejected) become
    approved; ``promote`` lists soft constraints the reviewer turns into hard ones (for
    example a statutory 80% policy). The reviewer's name goes into the note; nothing is
    approved without one."""
    if not reviewer.strip():
        raise ValueError("a review needs a named reviewer")
    promote = set(promote or [])
    for c in constraints:
        if ids is not None and c.constraint_id not in ids and c.constraint_id not in promote:
            continue
        if c.status == "unknown" or c.review_state == "rejected":
            continue
        if c.constraint_id in promote and c.constraint_type == "soft":
            c.constraint_type = "regulatory" if c.regulatory_basis else "hard"
            c.note = (c.note + " " if c.note else "") + f"promoted from soft by {reviewer}"
        c.review_state = "approved"
        c.note = (c.note + " " if c.note else "") + f"approved by {reviewer}"
    return constraints
