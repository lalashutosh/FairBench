import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fairbench.data import synthetic_universe
from fairbench.instances import named_universe
from fairbench.mandate import (MODEL, SYSTEM_PROMPT, Extraction, build_request, extract_rules,
                               mandate_to_constraints, unverified_sources, universe_summary)
from fairbench.rules import RULE_FIELDS, SPEC_SCHEMA, CompiledRules, load_spec

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
MANDATE = (EXAMPLES / "mandate_example.txt").read_text()
REFERENCE = load_spec(EXAMPLES / "mandate_example.rules.json")


# ---------------------------------------------------------------- helpers
def fake_client(text=None, stop_reason="end_turn", content=None, model="claude-opus-5-5",
                input_tokens=1200, output_tokens=345):
    """Stands in for anthropic.Anthropic(): records create() kwargs, returns a canned response."""
    calls = []
    if content is None:
        content = [SimpleNamespace(type="text", text=text)]

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stop_reason=stop_reason, content=content, model=model,
                               usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens))

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)), calls=calls)
    return client


def thinking(text="Let me work out which flags apply."):
    return SimpleNamespace(type="thinking", thinking=text, signature="sig")


def small():
    u = synthetic_universe(12, 3, seed=0)
    u.sector = ["Tech"] * 4 + ["Energy"] * 4 + ["Health"] * 4
    u.flags = {"weapons": np.arange(12) == 9, "tobacco": np.isin(np.arange(12), [1, 6])}
    return u


def rule(kind, source, **fields):
    return {"kind": kind, "source": source, "note": "", **fields}


# ------------------------------------------------------------ universe_summary
def test_universe_summary_lists_sectors_flags_and_scales():
    u = small()
    u.esg_score = np.array([10, 20, 30, 40, 5, 15, 25, 35, 60, 70, 80, 90], dtype=float)
    u.carbon = np.arange(12, dtype=float) + 1.0
    lines = universe_summary(u).splitlines()
    assert lines[0] == "Assets: 12"
    assert lines[1] == "Sectors: Energy (4), Health (4), Tech (4)"
    assert lines[2] == "Flags: tobacco (2 assets), weapons (1 assets)"
    assert lines[3] == "Categories: none (sector only)"
    assert lines[4] == "Numeric fields (per asset):"
    assert lines[5] == "  esg (ESG score, higher is better): min 5, mean 40, max 90"
    assert lines[6] == "  carbon (carbon figure, lower is better): min 1, mean 6.5, max 12"
    assert lines[7] == "Tickers: " + ", ".join(u.tickers) and len(lines) == 8


def rich():
    u = small()
    u.categories = {"region": ["EU"] * 7 + ["US"] * 5,
                    "country": ["DE", "DE", "FR", "FR", "FR", "SE", "SE", "US", "US", "US", "CA", "CA"]}
    u.attributes = {"mcap": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float),
                    "board_women_pct": np.array([30.0] * 6 + [40.0] * 6)}
    return u


def test_universe_summary_lists_each_category_with_value_counts():
    lines = universe_summary(rich()).splitlines()
    i = lines.index("Categories: country, region")  # names sorted
    assert lines[i + 1] == "  country: CA (2), DE (2), FR (3), SE (2), US (3)"
    assert lines[i + 2] == "  region: EU (7), US (5)"
    assert lines[:i] == universe_summary(small()).splitlines()[:3]  # sectors and flags come first


def test_universe_summary_lists_each_numeric_attribute_with_min_mean_max():
    u = rich()
    lines = universe_summary(u).splitlines()
    i = lines.index("Numeric fields (per asset):")
    assert lines[i + 1].startswith("  esg (") and lines[i + 2].startswith("  carbon (")
    assert lines[i + 3] == "  board_women_pct: min 30, mean 35, max 40"  # sorted by name
    assert lines[i + 4] == "  mcap: min 1, mean 6.5, max 12"
    assert lines[i + 5] == "Tickers: " + ", ".join(u.tickers) and len(lines) == i + 6
    u.attributes = {"empty_of_variation": np.full(12, 7.0)}
    assert "  empty_of_variation: min 7, mean 7, max 7" in universe_summary(u).splitlines()


def test_universe_summary_without_categories_or_attributes():
    u = small()
    for cats, attrs in [(None, None), ({}, {})]:
        u.categories, u.attributes = cats, attrs
        text = universe_summary(u)
        assert "Categories: none (sector only)" in text.splitlines()
        assert text.count("\n  ") == 2  # only the esg and carbon lines are indented
    u.categories = {"country": ["DE"] * 12}
    assert "Categories: country" in universe_summary(u).splitlines() and "  country: DE (12)" in universe_summary(u)


def test_universe_summary_truncates_a_category_with_many_values():
    u = small()
    u.categories = {"country": [f"C{i:02d}" for i in range(12)], "region": ["EU"] * 6 + ["US"] * 6}
    lines = universe_summary(u, max_values=5).splitlines()
    (row,) = [x for x in lines if x.startswith("  country:")]
    assert row == "  country: C00 (1), C01 (1), C02 (1), C03 (1), C04 (1), ... (12 values)"
    assert "C05" not in "\n".join(lines)
    assert "  region: EU (6), US (6)" in lines  # a short category is not truncated
    (row,) = [x for x in universe_summary(u, max_values=12).splitlines() if x.startswith("  country:")]
    assert row.endswith("C11 (1)") and "..." not in row  # exactly max_values values: complete
    assert "Sectors: Energy (4), Health (4), Tech (4)" in lines
    big = synthetic_universe(300, 3, seed=0)
    big.sector = [f"Sec{i}" for i in range(300)]
    sectors = universe_summary(big).splitlines()[1]  # sectors are truncated at the default 40 values too
    assert sectors.endswith("... (300 values)") and sectors.count("(1)") == 40


def test_universe_summary_counts_unequal_sectors():
    u = small()
    u.sector = ["Tech"] * 7 + ["Energy"] * 5
    assert "Sectors: Energy (5), Tech (7)" in universe_summary(u)


def test_universe_summary_flags_none_when_universe_has_no_flags():
    u = small()
    for empty in (None, {}):
        u.flags = empty
        assert "Flags: none" in universe_summary(u).splitlines()


def test_universe_summary_omits_tickers_above_max_tickers():
    u = small()
    assert "Tickers:" in universe_summary(u, max_tickers=12)
    s = universe_summary(u, max_tickers=11)
    assert "Tickers:" not in s and "A000" not in s and s.splitlines()[0] == "Assets: 12"
    assert "Tickers:" not in universe_summary(synthetic_universe(450, 5, seed=0))  # default limit is 400


# ---------------------------------------------------------------- build_request
def test_build_request_defaults_and_structured_output():
    u = small()
    req = build_request("The fund holds 20 names.", u)
    assert MODEL == "claude-opus-5-5" and req["model"] == "claude-opus-5-5"
    assert req["output_config"]["format"] == {"type": "json_schema", "schema": SPEC_SCHEMA}
    assert req["output_config"]["effort"] == "high"
    assert set(req["output_config"]) == {"format", "effort"}
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert req["system"] == SYSTEM_PROMPT and req["max_tokens"] == 16000
    assert build_request("x", u, model="claude-other", max_tokens=99)["model"] == "claude-other"
    assert build_request("x", u, max_tokens=99)["max_tokens"] == 99


def test_build_request_message_carries_mandate_and_universe_names():
    u = small()
    req = build_request("  \n  The Fund excludes tobacco.\n\nSecond paragraph.  \n", u)
    (msg,) = req["messages"]
    assert msg["role"] == "user" and isinstance(msg["content"], str)
    text = msg["content"]
    assert "<mandate>\nThe Fund excludes tobacco.\n\nSecond paragraph.\n</mandate>" in text  # stripped
    for name in ["Tech", "Energy", "Health", "tobacco", "weapons", *u.tickers]:
        assert name in text
    assert universe_summary(u) in text and text.index("<universe>") < text.index("<mandate>")


def test_build_request_user_message_contains_category_and_attribute_names():
    u = rich()
    text = build_request("The Fund invests only in the EU.", u)["messages"][0]["content"]
    universe = text[text.index("<universe>"):text.index("</universe>")]
    for name in ["country", "region", "mcap", "board_women_pct", "esg", "carbon", "DE", "EU", "SE"]:
        assert name in universe
    assert universe_summary(u) in universe and text.index("Categories: country, region") < text.index("<mandate>")
    plain = build_request("x", small())["messages"][0]["content"]
    assert "country" not in plain and "mcap" not in plain and "Categories: none (sector only)" in plain


def test_build_request_user_message_holds_no_per_asset_values():
    u = synthetic_universe(500, 10, seed=1)  # above max_tickers, so no tickers either
    u.flags = None
    text = build_request("Hold 20 names.", u)["messages"][0]["content"]
    assert len(text) < 800 and len(universe_summary(u)) < 600 and len(text.splitlines()) < 20  # does not grow with n
    shown = {f"{v:.4g}" for v in (u.esg_score.min(), u.esg_score.mean(), u.esg_score.max(),
                                  u.carbon.min(), u.carbon.mean(), u.carbon.max())}
    leaked = [f"{v:.4g}" for v in np.r_[u.esg_score[:60], u.carbon[:60]] if f"{v:.4g}" not in shown]
    assert leaked and not [s for s in leaked if s in text]
    assert "Tickers:" not in text and u.tickers[0] not in text
    small_text = build_request("Hold 20 names.", small())["messages"][0]["content"]
    assert len(small_text.splitlines()) == len(text.splitlines()) + 1  # only the tickers line differs


def test_build_request_has_none_of_the_rejected_sampling_parameters():
    req = build_request("Hold 20 names.", small())
    for key in ("thinking", "temperature", "top_p", "top_k", "tool_choice", "tools"):
        assert key not in req
    assert set(req) == {"model", "max_tokens", "system", "messages", "output_config", "betas", "fallbacks"}


@pytest.mark.parametrize("text", ["", " ", "\n\t  \n"])
def test_build_request_rejects_empty_mandate(text):
    with pytest.raises(ValueError, match="empty"):
        build_request(text, small())


def test_system_prompt_names_every_rule_kind_and_the_injection_guard():
    assert "unmapped" in RULE_FIELDS and len(RULE_FIELDS) == 12
    for kind in RULE_FIELDS:
        assert f"- {kind}:" in SYSTEM_PROMPT  # listed as a bullet of the kinds the compiler understands
    assert "not as an instruction" in SYSTEM_PROMPT


# ----------------------------------------------------------- unverified_sources
def test_unverified_sources_verbatim_quote_passes():
    spec = {"rules": [rule("exclude", "Companies that derive more than 5% of their revenue from thermal coal are excluded.")]}
    assert unverified_sources(spec, MANDATE) == []


def test_unverified_sources_ignores_whitespace_and_case():
    spec = {"rules": [rule("holdings", "the fund   HOLDS between 18\nand\t22 equally weighted positions"),
                      rule("position_cap", "  No single holding may exceed 7% of the portfolio  ")]}
    assert unverified_sources(spec, MANDATE) == []
    assert unverified_sources(spec, MANDATE.replace(" ", "\n")) == []  # the text can be wrapped differently too


def test_unverified_sources_flags_paraphrase_invention_and_empty():
    spec = {"rules": [
        rule("exclude", "The Fund never buys tobacco companies."),                 # 0 paraphrase
        rule("holdings", "The Fund holds between 18 and 22 equally weighted positions selected from the investment universe."),  # 1 ok
        rule("unmapped", ""),                                                        # 2 empty
        rule("unmapped", "   \n "),                                                  # 3 whitespace only
        rule("position_cap", "No single holding may exceed 10% of the portfolio"),  # 4 wrong number
        rule("unmapped", "The manager votes at all shareholder meetings"),           # 5 ok (prefix of a sentence)
    ]}
    assert unverified_sources(spec, MANDATE) == [0, 2, 3, 4]


def test_unverified_sources_in_a_spec_with_no_rules():
    assert unverified_sources({"rules": []}, MANDATE) == []


# ---------------------------------------------------------------- extract_rules
def test_extract_rules_returns_validated_spec_and_passes_metadata_through():
    u = named_universe()
    client = fake_client(json.dumps(REFERENCE), content=[thinking(), SimpleNamespace(type="text", text=json.dumps(REFERENCE))],
                         model="claude-fallback-9", input_tokens=4321, output_tokens=876)
    ex = extract_rules(MANDATE, u, client=client)
    assert isinstance(ex, Extraction) and ex.spec == REFERENCE
    assert ex.unverified == []
    assert (ex.model, ex.input_tokens, ex.output_tokens) == ("claude-fallback-9", 4321, 876)
    (kwargs,) = client.calls
    assert kwargs == build_request(MANDATE, u)
    assert kwargs["model"] == "claude-opus-5-5"


def test_extract_rules_passes_model_choice_to_the_request():
    client = fake_client(json.dumps(REFERENCE))
    extract_rules(MANDATE, named_universe(), client=client, model="claude-sonnet-x")
    assert client.calls[0]["model"] == "claude-sonnet-x"


def test_extract_rules_picks_the_text_block_among_other_blocks():
    spec = {"fund_name": "F", "rules": []}
    content = [thinking("hmm"), SimpleNamespace(type="text", text=json.dumps(spec)),
               SimpleNamespace(type="text", text="not json, and not read")]
    assert extract_rules("Some mandate.", small(), client=fake_client(content=content)).spec == spec


def test_extract_rules_reports_invented_quotes():
    spec = {"fund_name": "F", "rules": [rule("exclude", "The Fund never buys oil.", by="sector", values=["Energy"]),
                                        rule("unmapped", "The manager votes at all shareholder meetings", reason="r")]}
    ex = extract_rules(MANDATE, small(), client=fake_client(json.dumps(spec)))
    assert ex.unverified == [0] and ex.spec == spec


@pytest.mark.parametrize("stop,match", [("refusal", "declined"), ("max_tokens", "cut off")])
def test_extract_rules_raises_on_refusal_and_truncation(stop, match):
    client = fake_client(stop_reason=stop, content=[])  # a refusal may carry no content at all
    with pytest.raises(RuntimeError, match=match):
        extract_rules(MANDATE, small(), client=client)
    assert len(client.calls) == 1


def test_extract_rules_raises_on_truncation_even_with_partial_text():
    client = fake_client('{"fund_name": "F", "rul', stop_reason="max_tokens")
    with pytest.raises(RuntimeError, match="max_tokens"):
        extract_rules(MANDATE, small(), client=client)


def test_extract_rules_raises_when_model_json_fails_validation():
    bad = {"fund_name": "F", "rules": [rule("frobnicate", "Hold names.")]}
    with pytest.raises(ValueError, match="unknown kind"):
        extract_rules(MANDATE, small(), client=fake_client(json.dumps(bad)))
    bad = {"fund_name": "F", "rules": [rule("position_cap", "x", max_weight="seven percent")]}
    with pytest.raises(ValueError, match=r"rule 0 \(position_cap\): bad value for 'max_weight'"):
        extract_rules(MANDATE, small(), client=fake_client(json.dumps(bad)))
    with pytest.raises(ValueError):  # text that is not JSON at all
        extract_rules(MANDATE, small(), client=fake_client("Sorry, here is a summary instead."))


@pytest.mark.parametrize("content", [[], [thinking()], [SimpleNamespace(type="tool_use")]])
def test_extract_rules_raises_without_a_text_block(content):
    with pytest.raises(RuntimeError, match="no text block"):
        extract_rules(MANDATE, small(), client=fake_client(content=content))


def test_extract_rules_rejects_empty_mandate_before_calling_the_model():
    client = fake_client(json.dumps(REFERENCE))
    with pytest.raises(ValueError, match="empty"):
        extract_rules("  \n", small(), client=client)
    assert client.calls == []


# ------------------------------------------------------- mandate_to_constraints
def test_mandate_to_constraints_returns_compiled_rules_and_extraction():
    u = named_universe(150)
    client = fake_client(json.dumps(REFERENCE))
    compiled, ex = mandate_to_constraints(MANDATE, u, k=20, client=client)
    assert isinstance(compiled, CompiledRules) and isinstance(ex, Extraction)
    assert ex.spec == REFERENCE and ex.unverified == [] and len(client.calls) == 1
    assert compiled.k == 20 and compiled.warnings == []
    assert [r.status for r in compiled.rules].count("mapped") == 14 and len(compiled.unmapped) == 3
    assert [r.status for r in compiled.rules].count("trivial") == 1
    assert compiled.constraint_set.cardinality == 20


def test_mandate_to_constraints_without_k_uses_the_mandates_range():
    compiled, _ = mandate_to_constraints(MANDATE, named_universe(150), client=fake_client(json.dumps(REFERENCE)))
    assert compiled.k == 20 and any("18-22" in w for w in compiled.warnings)


def test_mandate_to_constraints_warns_about_an_invented_quote():
    text = "The Fund holds four names. It excludes tobacco."
    spec = {"fund_name": "F", "rules": [
        rule("holdings", "The Fund holds four names.", min=4, max=4),                                  # 0 verbatim
        rule("exclude", "The Fund never invests in oil.", by="flag", values=["tobacco"]),             # 1 invented
        rule("exclude", "it EXCLUDES   tobacco.", by="sector", values=["Energy"]),                    # 2 whitespace/case only
    ]}
    compiled, ex = mandate_to_constraints(text, small(), client=fake_client(json.dumps(spec)))
    assert ex.unverified == [1]
    quote_warnings = [w for w in compiled.warnings if "quote not found" in w]
    assert quote_warnings == ["rule 1 (exclude): quote not found in the mandate text"]
    assert "rule 1 (exclude)" in compiled.report() and compiled.k == 4
    assert not any(w.startswith(("rule 0", "rule 2")) for w in compiled.warnings)
    assert len(compiled.rules) == 3  # the rule is still compiled; the warning is for the reviewer


def test_mandate_to_constraints_propagates_extraction_errors():
    with pytest.raises(RuntimeError):
        mandate_to_constraints(MANDATE, small(), k=4, client=fake_client(stop_reason="refusal", content=[]))
