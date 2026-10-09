"""SIC classification and the exclusion proxy (offline)."""
import json

import pandas as pd

from fairbench.ingest.sic import (exclusion_reason, match_issuers, normalise_company_name, parse_assigned_sic,
                                  parse_cik_lookup, parse_company_tickers, resolve_ambiguous, sic_exclusions)

COMPANIES = json.dumps({
    "0": {"cik_str": 34088, "ticker": "XOM", "title": "EXXON MOBIL CORP"},
    "1": {"cik_str": 764180, "ticker": "MO", "title": "ALTRIA GROUP, INC."},
    "2": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
    "3": {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc."},
    "4": {"cik_str": 87347, "ticker": "SLB", "title": "SCHLUMBERGER LIMITED/NV"},
    "5": {"cik_str": 111, "ticker": "AAA", "title": "Acme Holdings Inc"},
    "6": {"cik_str": 222, "ticker": "AAB", "title": "Acme Group Corp"},
}).encode()


def test_names_are_normalised_for_linking_only():
    assert normalise_company_name("Kraft Heinz Co/The") == "KRAFT HEINZ"
    assert normalise_company_name("SCHLUMBERGER LIMITED/NV") == normalise_company_name("Schlumberger NV") == "SCHLUMBERGER"
    assert normalise_company_name("AT&T Inc.") == "AT AND T"
    assert normalise_company_name("Acme Holdings Inc", loose=True) == "ACME"


def test_only_unique_name_matches_are_accepted():
    companies = parse_company_tickers(COMPANIES)
    assert len(companies) == 6 and companies["cik"].str.len().eq(10).all()        # share classes collapse to one registrant
    m = match_issuers(["Exxon Mobil Corp", "Altria Group Inc", "Alphabet Inc", "Schlumberger NV", "Acme", "Nobody plc"],
                      companies).set_index("issuer_name")
    assert m.loc["Exxon Mobil Corp", "cik"] == "0000034088" and m.loc["Exxon Mobil Corp", "match"] == "exact"
    assert m.loc["Altria Group Inc", "match"] == "exact" and m.loc["Schlumberger NV", "cik"] == "0000087347"
    assert pd.isna(m.loc["Acme", "cik"]) and m.loc["Acme", "match"] == "ambiguous"   # two registrants: not guessed
    assert pd.isna(m.loc["Nobody plc", "cik"]) and m.loc["Nobody plc", "match"] == "unmatched"


def test_sic_ranges_and_feed_parsing():
    assert exclusion_reason("1311").startswith("fossil fuels") and exclusion_reason("2911").startswith("fossil fuels")
    assert exclusion_reason("2111") == "tobacco" and exclusion_reason("2085").startswith("alcohol")
    assert exclusion_reason("3760").startswith("weapons")
    assert exclusion_reason("2080") is None      # "beverages" is shared with soft drinks: not excluded on the code alone
    assert exclusion_reason("7011") is None and exclusion_reason(None) is None and exclusion_reason("n/a") is None
    feed = (b'<feed xmlns="http://www.w3.org/2005/Atom"><company-info><assigned-sic>2911</assigned-sic>'
            b"<assigned-sic-desc>PETROLEUM REFINING</assigned-sic-desc><cik>0000034088</cik></company-info></feed>")
    assert parse_assigned_sic(feed) == {"sic": "2911", "sic_description": "PETROLEUM REFINING"}
    assert parse_assigned_sic(b'<feed xmlns="http://www.w3.org/2005/Atom"><company-info/></feed>')["sic"] is None


def test_exclusion_table_is_labelled_assumed_and_keeps_unmatched_rows():
    companies = parse_company_tickers(COMPANIES)
    hold = pd.DataFrame({"security_key": ["cusip:1", "cusip:2", "cusip:3"],
                         "issuer_name": ["Exxon Mobil Corp", "Alphabet Inc", "Nobody plc"]})
    sic = pd.DataFrame({"cik": ["0000034088", "0001652044"], "sic": ["2911", "7370"],
                        "sic_description": ["PETROLEUM REFINING", "SERVICES-COMPUTER PROGRAMMING"]})
    t = sic_exclusions(hold, match_issuers(hold["issuer_name"], companies), sic).set_index("security_key")
    assert t.loc["cusip:1", "excluded"] and not t.loc["cusip:2", "excluded"] and not t.loc["cusip:3", "excluded"]
    assert (t["status"] == "assumed").all() and len(t) == 3
    assert pd.isna(t.loc["cusip:3", "sic"]) and t.loc["cusip:3", "match"] == "unmatched"


def test_looser_name_forms_and_the_historical_list():
    lookup = parse_cik_lookup(b"SCHWAB CHARLES CORP:0000316709:\nW.W. GRAINGER, INC.:0000277135:\n"
                              b"HORTON D R INC /DE/:0000882184:\nVERTEX PHARMACEUTICALS INC / MA:0000875320:\n"
                              b"APACHE CORP:0000006769:\nAPACHE CORP:0000006770:\nbad line\n")
    assert len(lookup) == 6
    m = match_issuers(["Charles Schwab Corp. (The)", "WW Grainger, Inc.", "DR Horton, Inc.",
                       "Vertex Pharmaceuticals, Inc.", "Apache Corp"], lookup).set_index("issuer_name")
    assert m.loc["Charles Schwab Corp. (The)", "match"] == "reordered"
    assert m.loc["WW Grainger, Inc.", "match"] == "compact" and m.loc["DR Horton, Inc.", "match"] == "reordered"
    assert m.loc["Vertex Pharmaceuticals, Inc.", "match"] == "exact"
    assert m.loc["Apache Corp", "match"] == "ambiguous" and m.loc["Apache Corp", "candidates"] == "0000006769|0000006770"


def test_ambiguous_names_resolve_only_when_the_candidates_agree():
    m = pd.DataFrame({"issuer_name": ["Apache Corp", "Mixed Co", "Blank Co"], "cik": [None, None, None],
                      "registrant_title": [None] * 3, "match": ["ambiguous"] * 3,
                      "candidates": ["0000000001|0000000002", "0000000003|0000000004", "0000000005|0000000006"]})
    sic_of = {"0000000001": "1311", "0000000002": None, "0000000003": "1311", "0000000004": "7370",
              "0000000005": None, "0000000006": None}
    r = resolve_ambiguous(m, sic_of).set_index("issuer_name")
    assert r.loc["Apache Corp", "cik"] == "0000000001" and "same classification" in r.loc["Apache Corp", "match"]
    assert pd.isna(r.loc["Mixed Co", "cik"]) and r.loc["Mixed Co", "match"] == "ambiguous"     # oil vs software: not guessed
    assert pd.isna(r.loc["Blank Co", "cik"])
