"""Canonical constraints: hard/soft wording, evidence checks, selection policy, compilation, storage."""
from pathlib import Path

import numpy as np
import pytest

from fairbench.constraints import HoldingsRange, PositionBound, WeightedAverage, WeightSum
from fairbench.data import synthetic_universe
from fairbench.ingest.documents import parse_html
from fairbench.mandates.compile import select, to_constraint_set, to_weight_rules
from fairbench.mandates.dsl import Constraint, approve, dump_constraints, load_constraints
from fairbench.mandates.extract import constraints_from_spec, extract_constraints
from fairbench.mandates.modality import classify
from fairbench.store import db

FIXTURE = Path(__file__).parent / "fixtures" / "prospectus_synthetic.htm"

Q_HOLD = "The Fund normally holds between 35 and 45 securities."
Q_TOBACCO = ("The Fund will not invest in companies that derive 10% or more of their revenue from the "
             "manufacture of tobacco products.")
Q_CARBON = "The Fund seeks to maintain a portfolio carbon intensity below that of its benchmark."
Q_INDUSTRY = "The Fund may not invest more than 25% of its total assets in any one industry."
Q_ISSUER = ("With respect to 75% of its total assets, the Fund may not invest more than 5% of its total assets "
            "in the securities of any one issuer.")
Q_ESG = "The Adviser considers environmental, social and governance factors as part of its investment process."
Q_NAMES = ("Under normal circumstances, the Fund invests at least 80% of its net assets in equity securities of "
           "large-capitalization companies.")


def _rule(kind, source, **fields):
    return {"kind": kind, "source": source, "note": "", **fields}


SPEC = {"fund_name": "Synthetic Example Equity Fund", "rules": [
    _rule("holdings", Q_HOLD, min=35, max=45),
    _rule("exclude", Q_TOBACCO, by="flag", values=["tobacco"]),
    _rule("portfolio_average", Q_CARBON, metric="carbon", bound="max", value=1.0, basis="relative_to_mean"),
    _rule("sector_cap", Q_INDUSTRY, sector="*", limit=0.25, unit="weight"),
    _rule("position_cap", Q_ISSUER, max_weight=0.05),
    _rule("unmapped", Q_ESG, reason="ESG integration has no numeric form"),
    _rule("group_limit", Q_NAMES, by="flag", values=["large_cap"], bound="min", limit=0.8, unit="weight"),
    _rule("portfolio_average", Q_ESG, metric="esg", bound="min", value=60.0, basis="absolute"),  # invented number
    _rule("exclude", "The Fund never buys tobacco stocks.", by="flag", values=["tobacco"]),      # paraphrase
    _rule("exclude", Q_TOBACCO, by="flag", values=["thermal_coal"]),                              # no such data
]}


@pytest.fixture(scope="module")
def doc():
    return parse_html(FIXTURE.read_bytes())


@pytest.fixture(scope="module")
def u():
    u = synthetic_universe(60, 6, seed=3)
    rng = np.random.default_rng(0)
    u.flags = {"tobacco": np.arange(60) % 15 == 0, "large_cap": rng.random(60) < 0.9}
    return u


@pytest.fixture
def cons(doc, u):
    return constraints_from_spec(SPEC, doc, u, fund_id="S000000001", mandate_version_id="mv1",
                                 source_url="https://www.sec.gov/Archives/edgar/data/1/x/prospectus.htm",
                                 retrieved_at="2026-10-09T00:00:00Z", proxies={"tobacco"})


# ------------------------------------------------------------------ wording
@pytest.mark.parametrize("text,kind", [
    (Q_TOBACCO, "hard"), (Q_INDUSTRY, "hard"), (Q_ISSUER, "hard"),
    ("The Fund must hold at least 20 securities.", "hard"),
    ("The Fund shall not exceed 10% in any issuer.", "hard"),
    (Q_HOLD, "soft"), (Q_CARBON, "soft"), (Q_ESG, "soft"),
    ("The Fund intends to keep tracking error below 2%.", "soft"),
    ("The Fund expects to hold 40 names where practicable.", "soft"),
    ("The Fund may invest up to 20% in foreign securities.", "soft"),
    ("The Fund will generally not invest more than 5% in any issuer.", "soft"),
    ("The Fund invests in large companies.", "soft"),
])
def test_wording_classification(text, kind):
    assert classify(text).constraint_type == kind


def test_mixed_wording_is_soft_and_keeps_the_regulatory_hint():
    m = classify(Q_NAMES)
    assert m.constraint_type == "soft" and m.mixed
    assert "at least" in m.hard_terms and "under normal circumstances" in m.soft_terms
    assert m.regulatory_basis == "rule_35d-1_names_rule"
    assert classify(Q_ISSUER).regulatory_basis == "1940_act_diversification"
    assert classify(Q_INDUSTRY).regulatory_basis == "1940_act_concentration"
    assert classify("The Fund invests in large companies.").unmarked
    assert "may" not in classify("The Fund may not borrow.").soft_terms


# ------------------------------------------------------------------ evidence
def test_every_constraint_carries_its_evidence(cons, doc):
    assert len(cons) == len(SPEC["rules"])
    for c in cons[:7]:
        assert c.status in ("disclosed", "derived") and c.source_locator and c.evidence_text in doc.text
        assert c.source_url.startswith("https://www.sec.gov/") and c.source_document == doc.sha256_of_source
        assert c.retrieved_at and c.extraction_method and 0 < c.confidence <= 0.95
        assert c.effective_from is None and c.effective_date_basis is None  # no date was given: none invented
    assert len({c.constraint_id for c in cons}) == len(cons)


def test_types_statuses_and_observability(cons):
    hold, tob, carbon, ind, issuer, esg, names, invented, paraphrase, nodata = cons
    assert [c.constraint_type for c in (hold, tob, carbon, ind, issuer, esg, names)] == \
        ["soft", "hard", "soft", "hard", "hard", "soft", "soft"]
    assert hold.metric == "holding_count" and hold.value == [35, 45]
    assert tob.observability == "proxy" and tob.status == "derived"          # SIC-style flag stands in for revenue
    assert carbon.metric == "carbon_intensity" and carbon.reference == "relative_to_benchmark"
    assert ind.metric == "group_weight" and ind.regulatory_basis == "1940_act_concentration"
    assert issuer.metric == "position_weight" and issuer.value == 0.05
    assert esg.metric == "other" and esg.observability == "unobservable" and esg.compile_target is None
    assert names.regulatory_basis == "rule_35d-1_names_rule" and names.value == 0.8


def test_invented_number_is_rejected(cons):
    invented = cons[7]
    assert invented.status == "unknown" and invented.value is None and invented.compile_target is None
    assert "60" in invented.note and invented.confidence == 0.3


def test_paraphrased_quote_is_rejected(cons):
    paraphrase = cons[8]
    assert paraphrase.status == "unknown" and paraphrase.confidence == 0.0 and paraphrase.source_locator is None
    assert "quote not found" in paraphrase.note and paraphrase.compile_target is None


def test_rule_without_public_data_is_unobservable(cons):
    nodata = cons[9]
    assert nodata.observability == "unobservable" and "thermal_coal" in nodata.note
    assert nodata.why_not_enforced(require_approved=False) == "the data it needs is not public (unobservable)"


def test_effective_date_needs_its_basis(doc, u):
    with pytest.raises(ValueError):
        constraints_from_spec(SPEC, doc, u, fund_id="F", effective_from="2026-05-01")
    c = constraints_from_spec(SPEC, doc, u, fund_id="F", effective_from="2026-05-01",
                              effective_date_basis="date printed on the prospectus cover")[0]
    assert c.effective_from == "2026-05-01"
    with pytest.raises(ValueError):
        Constraint(constraint_id="x", fund_id="F", metric="other", operator="n/a", constraint_type="soft",
                   observability="unobservable", status="unknown", evidence_text="", effective_from="2026-01-01")


def test_constraint_validation():
    base = dict(constraint_id="x", fund_id="F", metric="other", operator="n/a", constraint_type="soft",
                observability="unobservable", status="unknown", evidence_text="t")
    Constraint(**base)
    for bad in (dict(status="guessed"), dict(constraint_type="firm"), dict(confidence=1.2), dict(metric="alpha"),
                dict(status="disclosed", evidence_text=" ")):
        with pytest.raises(ValueError):
            Constraint(**{**base, **bad})


def test_json_round_trip(cons):
    again = load_constraints(dump_constraints(cons))
    assert [c.to_dict() for c in again] == [c.to_dict() for c in cons]


# ---------------------------------------------------------- selection policy
def test_nothing_is_enforced_before_review(cons):
    s = select(cons)
    assert not s.enforced and len(s.not_enforced) == len(cons)
    assert any("not approved" in why for _, why in s.not_enforced)


def test_review_then_hard_only_and_soft_promoted(cons):
    with pytest.raises(ValueError):
        approve(cons, reviewer=" ")
    approve(cons, reviewer="A. Reviewer")
    hard = select(cons)
    assert [c.metric for c in hard.enforced] == ["eligibility", "group_weight", "position_weight"]
    reasons = {c.constraint_id: why for c, why in hard.not_enforced}
    assert reasons[cons[0].constraint_id].startswith("soft wording (normally")
    assert reasons[cons[5].constraint_id] == "no numeric form"
    assert reasons[cons[7].constraint_id] == "no numeric form"            # the invented threshold has no target
    assert reasons[cons[8].constraint_id] == "no numeric form"
    assert reasons[cons[9].constraint_id] == "the data it needs is not public (unobservable)"
    assert all(c.status != "unknown" for c in hard.enforced)

    both = select(cons, promote_soft=True)
    assert len(both.enforced) == 6 and "soft" in {c.constraint_type for c in both.enforced}
    assert len(select(cons, allow_proxy=False).enforced) == 2            # the tobacco proxy drops out
    assert "not enforced" in hard.report() and "enforced" in hard.report()


def test_reviewer_can_promote_a_statutory_soft_rule(cons):
    names = cons[6]
    approve(cons, reviewer="A. Reviewer", promote=[names.constraint_id])
    assert names.constraint_type == "regulatory" and "promoted from soft by A. Reviewer" in names.note
    assert names in select(cons).enforced


# ---------------------------------------------------------------- compiling
def test_stage1_compiles_through_the_existing_rule_compiler(cons, u):
    approve(cons, reviewer="A. Reviewer")
    compiled = to_constraint_set(select(cons, promote_soft=True), u, k=40)
    kinds = {r.rule["kind"]: r.status for r in compiled.rules}
    assert kinds["holdings"] == "mapped" and kinds["exclude"] == "mapped" and kinds["sector_cap"] == "mapped"
    assert kinds["position_cap"] == "trivial"                              # 1/40 = 2.5% <= 5%
    assert compiled.constraint_set.cardinality == 40
    with pytest.raises(ValueError):
        to_constraint_set(select(cons), u)                                 # hard-only set has no holdings rule: k needed


def test_stage2_compiles_to_weight_rules(cons, u):
    approve(cons, reviewer="A. Reviewer")
    b = np.random.default_rng(1).dirichlet(np.ones(60) * 3)
    rules, notes = to_weight_rules(select(cons, promote_soft=True), u, benchmark=b)
    kinds = [type(r) for r in rules.rules]
    assert kinds.count(WeightSum) == 6 + 1 and PositionBound in kinds and HoldingsRange in kinds
    assert WeightedAverage in kinds and rules.support_rules is not None
    assert any("benchmark-weighted average" in n for n in notes) and not any("NOT BUILT" in n for n in notes)

    ok = np.flatnonzero(~u.flags["tobacco"] & u.flags["large_cap"])
    order = ok[np.argsort(u.carbon[ok])]                                    # low-carbon names first
    pick = np.concatenate([order[np.array(u.sector)[order] == s][:7] for s in sorted(set(u.sector))])[:40]
    w = np.zeros(60)
    w[pick] = 1 / len(pick)
    assert rules.check_weight(w, u)
    concentrated = w.copy()
    concentrated[pick[0]] += 0.1
    concentrated[pick[1:5]] -= 0.025
    assert not rules.check_weight(concentrated, u)                          # 12.5% position breaks the 5% cap
    with_tobacco = w.copy()
    bad = int(np.flatnonzero(u.flags["tobacco"])[0])
    with_tobacco[bad], with_tobacco[pick[0]] = w[pick[0]], 0.0
    assert not rules.check_weight(with_tobacco, u)
    assert rules.violations(with_tobacco[None, :], u)["support_rules"] == 1


def test_hard_only_weight_rules_are_looser(cons, u):
    approve(cons, reviewer="A. Reviewer")
    hard, _ = to_weight_rules(select(cons), u)
    soft, _ = to_weight_rules(select(cons, promote_soft=True), u)
    assert len(hard.rules) < len(soft.rules)


# ------------------------------------------------------------------ storage
def test_constraints_store_with_provenance(cons, doc):
    conn = db.open_db()
    db.add_fund(conn, fund_id="S000000001", name="Synthetic Example Equity Fund")
    filing = db.add_filing(conn, accession="0000000001-26-000001", cik="0000000001", form_type="485BPOS",
                           filing_date="2026-04-30")
    document = db.add_document(conn, filing_id=filing, url=cons[0].source_url, sha256=doc.sha256_of_source,
                               retrieved_at="2026-10-09T00:00:00Z")
    run = db.start_run(conn, method="llm", tool_version="test")
    db.add_mandate_version(conn, mandate_version_id="mv1", fund_id="S000000001")
    for c in cons:
        ev = db.add_evidence(conn, document_id=document, locator=c.source_locator or "not located",
                             evidence_text=c.evidence_text)
        db.add_constraint(conn, c.to_row(document_id=document, extraction_run_id=run, evidence_id=ev))
    stored = db.get_constraints(conn, "mv1")
    assert len(stored) == len(cons)
    by_id = {s["constraint_id"]: s for s in stored}
    assert by_id[cons[0].constraint_id]["value"] == [35, 45]
    prov = db.provenance(conn, "constraints", cons[1].constraint_id)
    assert prov["source_url"] == cons[1].source_url and prov["filing_date"] == "2026-04-30"
    assert prov["status"] == "derived" and prov["locator"] == cons[1].source_locator
    assert db.check_provenance(conn)["constraints"] == 0


# -------------------------------------------------- the model call stays in mandate.py
def test_extract_constraints_uses_the_single_model_entry_point(doc, u):
    import json
    from types import SimpleNamespace

    class FakeMessages:
        def create(self, **kw):
            self.kw = kw
            return SimpleNamespace(stop_reason="end_turn", model="fake", usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                                   content=[SimpleNamespace(type="text", text=json.dumps(SPEC))])

    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))
    cons, spec = extract_constraints(doc, u, fund_id="S000000001", client=client)
    assert spec == SPEC and len(cons) == len(SPEC["rules"])
    assert Q_TOBACCO in client.beta.messages.kw["messages"][0]["content"]
