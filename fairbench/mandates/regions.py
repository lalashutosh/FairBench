"""Regional rule packs: what the law of a region requires of a fund, beside what the fund promises.

A fund's own documents are one source of rules. The region it is sold in is another:
US-registered funds carry statutory limits whatever their prospectus says, and EU funds
that use sustainability words in their NAME must meet minimum thresholds and exclusions.
This module holds those requirements as data, turns the ones that apply into canonical
constraints (``constraint_type = "regulatory"``), and checks a fund's extracted mandate
against them: is each regional requirement matched, exceeded, weaker, or not mentioned?

Every rule carries its citation and a ``verified`` flag. The packs were written from the
regulations as summarised in ``FAIRBENCH_INDUSTRY_STANDARDS_RESEARCH.md`` and from memory
of the legal texts; a rule with ``verified = False`` has NOT been re-read against the
official text and is stored with status "assumed". Nothing here is legal advice, and a
"consistent" verdict is a statement about the documents, not a compliance certificate.

Most regional rules need data the public does not have (which holdings count as meeting a
fund's own ESG characteristics; revenue shares by activity). Such a rule is still
recorded, with ``observability = "unobservable"``, so the report shows that the benchmark
does not capture it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..data import Universe
from ..rules import _match
from .dsl import Constraint

SEC_NAMES_RULE = "https://www.sec.gov/files/rules/final/2023/33-11238.pdf"
ESMA_NAMES = "https://www.esma.europa.eu/document/guidelines-funds-names-using-esg-or-sustainability-related-terms"
EU_BENCHMARKS = "https://eur-lex.europa.eu/eli/reg_del/2020/1818/oj"


@dataclass(frozen=True)
class RegionalRule:
    """applies_if: "always", "diversified" (the fund calls itself diversified), "name_focus"
    (its name suggests an investment focus), "name_pab" / "name_ctb" (EU name-term
    categories, see ``name_categories``) or "claims_pab". target: the executable form as a
    ``fairbench.rules`` rule, or None when the rule has no such form. needs: universe data
    the rule depends on ("sector", "carbon", or "flag:<name>")."""
    rule_id: str
    region: str
    title: str
    applies_if: str
    requirement: str
    formula: str
    citation: str
    url: str
    verified: bool
    target: dict | None = None
    needs: tuple[str, ...] = ()
    note: str = ""


def _excl(rule_id: str, applies: str, activity: str, flag: str, threshold: str, letter: str) -> RegionalRule:
    return RegionalRule(
        rule_id, "EU", f"Exclusion: {activity}", applies,
        f"Companies {threshold} are excluded.", f"x_i = 0 for every company i flagged {flag}",
        f"Commission Delegated Regulation (EU) 2020/1818, Article 12(1)({letter}), applied to fund names by the ESMA guidelines",
        EU_BENCHMARKS, False, {"kind": "exclude", "by": "flag", "values": [flag]}, (f"flag:{flag}",))


_PAB_ONLY = [
    ("d", "hard coal and lignite", "coal_1pct", "deriving 1% or more of revenue from exploration, mining, extraction, distribution or refining of hard coal and lignite"),
    ("e", "oil fuels", "oil_10pct", "deriving 10% or more of revenue from exploration, extraction, distribution or refining of oil fuels"),
    ("f", "gaseous fuels", "gas_50pct", "deriving 50% or more of revenue from exploration, extraction, manufacturing or distribution of gaseous fuels"),
    ("g", "high-carbon electricity", "high_carbon_power_50pct", "deriving 50% or more of revenue from electricity generation with a greenhouse-gas intensity above 100 g CO2e/kWh"),
]
_SHARED = [
    ("a", "controversial weapons", "controversial_weapons", "involved in any activity related to controversial weapons"),
    ("b", "tobacco", "tobacco", "involved in the cultivation and production of tobacco"),
    ("c", "UN Global Compact / OECD violations", "ungc_violation", "found in violation of the UN Global Compact principles or the OECD Guidelines for Multinational Enterprises"),
]

REGIONAL_RULES: tuple[RegionalRule, ...] = (
    RegionalRule("us_names_rule_80", "US", "Names rule: 80% in the focus the name suggests", "name_focus",
                 "At least 80% of the value of the fund's assets is invested in accordance with the investment focus its name suggests.",
                 "sum of w_i over holdings in the focus >= 0.80",
                 "SEC Rule 35d-1 under the Investment Company Act, as amended 2023 (Release 33-11238)", SEC_NAMES_RULE, True,
                 {"kind": "group_limit", "by": "flag", "values": ["in_name_focus"], "bound": "min", "limit": 0.8, "unit": "weight"},
                 ("flag:in_name_focus",), "Which holdings are in the focus is defined by the fund; public data does not say."),
    RegionalRule("us_diversified_75_5", "US", "Diversified fund: 75% / 5% / 10% test", "diversified",
                 "For 75% of total assets, no more than 5% of total assets is in any one issuer and no more than 10% of that issuer's voting securities is held.",
                 "sum of w_i over positions with w_i > 0.05  <=  0.25", "Investment Company Act of 1940, Section 5(b)(1)",
                 "https://www.govinfo.gov/content/pkg/COMPS-1879/pdf/COMPS-1879.pdf", False, None, (),
                 "A condition on the larger positions together, not a cap on each; the voting-securities limb needs share counts."),
    RegionalRule("us_concentration_25", "US", "Industry concentration: 25%", "always",
                 "No more than 25% of total assets in any one industry, unless the fund has a stated policy to concentrate.",
                 "sum of w_i over each industry  <=  0.25", "Investment Company Act of 1940, Section 8(b)(1), and SEC staff position on concentration",
                 "https://www.sec.gov/about/forms/formn-1a.pdf", False,
                 {"kind": "sector_cap", "sector": "*", "limit": 0.25, "unit": "weight"}, ("sector",),
                 "Industry labels are not in public holdings data; any classification used is a stand-in."),
    RegionalRule("us_ric_issuer_25", "US", "Tax diversification: 25% per issuer", "always",
                 "At each quarter-end no more than 25% of assets is in the securities of any one issuer.",
                 "w_i <= 0.25 for every holding", "Internal Revenue Code, Section 851(b)(3)(B)",
                 "https://www.govinfo.gov/app/details/USCODE-2023-title26/USCODE-2023-title26-subtitleA-chap1-subchapM-partI-sec851",
                 False, {"kind": "position_cap", "max_weight": 0.25}),
    RegionalRule("eu_name_80", "EU", "Fund names: 80% meeting the promoted characteristics", "name_any",
                 "At least 80% of investments are used to meet the environmental or social characteristics or the sustainable investment objective, in line with the binding elements of the strategy.",
                 "sum of w_i over holdings meeting the binding elements >= 0.80",
                 "ESMA Guidelines on funds' names using ESG or sustainability-related terms (ESMA34-472-440)", ESMA_NAMES, True,
                 {"kind": "group_limit", "by": "flag", "values": ["meets_binding_elements"], "bound": "min", "limit": 0.8, "unit": "weight"},
                 ("flag:meets_binding_elements",), "Which holdings meet the binding elements is the fund's own classification."),
    *[_excl(f"eu_excl_{flag}", "name_any", act, flag, thr, letter) for letter, act, flag, thr in _SHARED],
    *[_excl(f"eu_excl_{flag}", "name_pab", act, flag, thr, letter) for letter, act, flag, thr in _PAB_ONLY],
    RegionalRule("eu_pab_intensity_50", "EU", "Paris-aligned benchmark: 50% lower carbon intensity", "claims_pab",
                 "Portfolio greenhouse-gas intensity is at least 50% below that of the investable universe.",
                 "sum of w_i * intensity_i  <=  0.5 * universe intensity",
                 "Commission Delegated Regulation (EU) 2020/1818, Article 11", EU_BENCHMARKS, False,
                 {"kind": "portfolio_average", "metric": "carbon", "bound": "max", "value": 0.5, "basis": "relative_to_mean"}, ("carbon",)),
    RegionalRule("eu_pab_decarbonisation_7", "EU", "Paris-aligned benchmark: 7% a year decarbonisation", "claims_pab",
                 "Portfolio greenhouse-gas intensity falls by at least 7% a year on average.",
                 "intensity(year t) <= 0.93 * intensity(year t-1)", "Commission Delegated Regulation (EU) 2020/1818, Article 7", EU_BENCHMARKS,
                 False, None, ("carbon",), "A target over time needs a carbon history; one snapshot cannot test it."),
)

# ESMA groups name terms; the first group brings the Paris-aligned exclusions, the second the lighter set.
_PAB_TERMS = r"\b(esg|sri|sustainab\w*|environment\w*|green|climate|impact\w*|paris|net[- ]zero|responsible)\b"
_CTB_TERMS = r"\b(transition\w*|social\w*|governance|improv\w*|progress\w*|evolution|transformation)\b"


def name_categories(fund_name: str) -> set[str]:
    """Which EU name-term groups a fund name falls in: "name_pab" (sustainability,
    environmental, ESG or impact words: Paris-aligned exclusions apply), "name_ctb"
    (transition, social or governance words: the three shared exclusions apply),
    and "name_any" if either. A keyword reading of the name, to be confirmed by a person."""
    n = fund_name.casefold()
    out = set()
    if re.search(_PAB_TERMS, n):
        out.add("name_pab")
    if re.search(_CTB_TERMS, n):
        out.add("name_ctb")
    return out | ({"name_any"} if out else set())


def applicable_rules(region: str, *, fund_name: str = "", diversified: bool = True, claims_pab: bool = False) -> list[RegionalRule]:
    """The rules of ``region`` ("US" or "EU") that apply to a fund with this profile."""
    if region not in ("US", "EU"):
        raise ValueError('region must be "US" or "EU"')
    on = {"always"} | ({"diversified"} if diversified else set()) | ({"claims_pab"} if claims_pab else set())
    if region == "EU":
        on |= name_categories(fund_name)
    elif fund_name.strip():
        on.add("name_focus")
    return [r for r in REGIONAL_RULES if r.region == region and r.applies_if in on]


def _has(u: Universe, need: str) -> bool:
    if need.startswith("flag:"):
        return need[5:] in (u.flags or {})
    if need == "sector":
        return len(set(u.sector)) > 1
    hit, _ = _match([need], u.numeric_names)
    return bool(hit)


def regional_constraints(rules: list[RegionalRule], u: Universe, *, fund_id: str,
                         mandate_version_id: str | None = None, proxies: frozenset[str] | set[str] = frozenset()) -> list[Constraint]:
    """Regional rules as canonical constraints for universe ``u``. A rule whose data ``u``
    lacks is "unobservable" and keeps no executable form; one that rests on a stand-in
    named in ``proxies`` is "proxy". Status is "disclosed" when the rule text was verified
    against its source and "assumed" otherwise."""
    out = []
    for r in rules:
        missing = [n for n in r.needs if not _has(u, n)]
        proxy = [n for n in r.needs if n in proxies or n.removeprefix("flag:") in proxies]
        obs = "unobservable" if missing or r.target is None and r.needs else "proxy" if proxy else "observable"
        note = " ".join(x for x in (r.note, "no public data for: " + ", ".join(missing) if missing else "",
                                    "" if r.verified else "regulatory text not re-read against the official source") if x)
        out.append(Constraint(
            constraint_id=f"reg_{r.rule_id}", fund_id=fund_id, mandate_version_id=mandate_version_id, metric="other",
            operator="n/a", constraint_type="regulatory", observability=obs, status="disclosed" if r.verified else "assumed",
            evidence_text=r.requirement, confidence=0.9 if r.verified else 0.6, source_url=r.url, source_document=r.citation,
            source_locator=r.citation, extraction_method="regional rule pack", regulatory_basis=r.rule_id,
            compile_target=None if missing or r.target is None else dict(r.target, source=r.requirement, note=r.citation),
            note=note))
    return out


def _mandate_rules(mandate: list[Constraint], kind: str) -> list[tuple[Constraint, dict]]:
    return [(c, c.compile_target) for c in mandate if c.compile_target and c.compile_target.get("kind") == kind
            and c.constraint_type != "regulatory"]


def check_mandate(mandate: list[Constraint], rules: list[RegionalRule]) -> list[dict]:
    """Compare a fund's own constraints with each regional rule. Verdicts:
        "at least as strict"  the mandate states the same kind of limit, as tight or tighter
        "weaker"              it states a looser limit than the region requires
        "silent"              it states nothing of that kind (the regional rule still applies)
        "not comparable"      the regional rule has no executable form to compare with
    The comparison is on the kind of rule and its number, so it can miss a requirement
    worded in an unusual way; "silent" means "not found", and a person should confirm."""
    out = []
    for r in rules:
        t, verdict, match, why = r.target, "silent", None, "no rule of this kind in the fund's documents"
        if t is None:
            verdict, why = "not comparable", "the regional rule has no single-number form"
        elif t["kind"] == "position_cap":
            for c, m in _mandate_rules(mandate, "position_cap"):
                ok = m["max_weight"] <= t["max_weight"] + 1e-12
                verdict, match = ("at least as strict" if ok else "weaker"), c
                why = f"fund caps a position at {m['max_weight']:.0%}; the region allows {t['max_weight']:.0%}"
                if ok:
                    break
        elif t["kind"] == "sector_cap":
            for c, m in _mandate_rules(mandate, "sector_cap"):
                if m["unit"] != "weight" or m["sector"].strip() not in ("", "*"):
                    continue
                ok = m["limit"] <= t["limit"] + 1e-12
                verdict, match = ("at least as strict" if ok else "weaker"), c
                why = f"fund caps an industry at {m['limit']:.0%}; the region allows {t['limit']:.0%}"
                if ok:
                    break
        elif t["kind"] == "exclude":
            want = {v.casefold() for v in t["values"]}
            for c, m in _mandate_rules(mandate, "exclude"):
                if want & {v.casefold() for v in m["values"]}:
                    verdict, match, why = "at least as strict", c, "the fund excludes the same group"
                    break
        elif t["kind"] == "group_limit":
            for c, m in _mandate_rules(mandate, "group_limit"):
                if m["bound"] != "min" or m["unit"] != "weight":
                    continue
                ok = m["limit"] >= t["limit"] - 1e-12
                verdict, match = ("at least as strict" if ok else "weaker"), c
                why = (f"fund commits {m['limit']:.0%} to a group it defines; the region requires {t['limit']:.0%} "
                       "(whether the two groups are the same is for a person to judge)")
                if ok:
                    break
        elif t["kind"] == "portfolio_average":
            for c, m in _mandate_rules(mandate, "portfolio_average"):
                if m["metric"].casefold() != t["metric"] or m["bound"] != t["bound"] or m["basis"] != t["basis"]:
                    continue
                ok = m["value"] <= t["value"] + 1e-12 if t["bound"] == "max" else m["value"] >= t["value"] - 1e-12
                verdict, match = ("at least as strict" if ok else "weaker"), c
                why = f"fund states {m['value']:g} x the reference; the region requires {t['value']:g} x"
                if ok:
                    break
        out.append(dict(rule_id=r.rule_id, title=r.title, verdict=verdict, why=why, verified=r.verified,
                        matched_constraint=None if match is None else match.constraint_id,
                        matched_wording=None if match is None else match.constraint_type))
    return out
