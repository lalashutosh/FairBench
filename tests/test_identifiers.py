"""Identifier normaliser: check digits, round trips, placeholders, key precedence, report, history."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from fairbench.ingest.identifiers import (
    add_security_keys, build_identifier_history, clean_identifier, cusip_check_digit,
    cusip_to_isin, identifier_report, is_valid_cusip, is_valid_isin, is_valid_lei,
    isin_check_digit, isin_to_cusip, normalise_text, security_key)

# name -> (CUSIP, ISIN); all published identifiers of well-known US listings
KNOWN = {
    "Apple": ("037833100", "US0378331005"),
    "Microsoft": ("594918104", "US5949181045"),
    "Alphabet A": ("02079K305", "US02079K3059"),
    "Cisco": ("17275R102", None),
    "Amazon": ("023135106", "US0231351067"),
    "Tesla": ("88160R101", "US88160R1014"),
    "Meta": ("30303M102", "US30303M1027"),
}
GOOD_CUSIPS = [c for c, _ in KNOWN.values()]
GOOD_ISINS = [i for _, i in KNOWN.values() if i] + ["GB0002634946", "DE000BAY0017"]
GOOD_LEIS = ["5493006MHB84DD0ZWV18", "HWUPKR0MPOU8FGXBT394", "INR2EJN1ERAN0W5ZP974"]

APPLE, MSFT, GOOG = "037833100", "594918104", "02079K305"


def _swap_char(s: str, pos: int, new: str) -> str:
    return s[:pos] + new + s[pos + 1:]


# ------------------------------------------------------------------- CUSIP
@pytest.mark.parametrize("cusip", GOOD_CUSIPS)
def test_cusip_check_digit_and_validity(cusip):
    assert cusip_check_digit(cusip[:8]) == cusip[8]
    assert is_valid_cusip(cusip)
    assert is_valid_cusip(f"  {cusip.lower()} ")  # strip + upper


@pytest.mark.parametrize("cusip", GOOD_CUSIPS)
def test_cusip_single_digit_corruptions_are_caught(cusip):
    # a digit replaced by a different digit always changes the weighted digit sum mod 10
    for pos, ch in enumerate(cusip):
        if not ch.isdigit():
            continue
        for new in "0123456789":
            if new != ch:
                assert not is_valid_cusip(_swap_char(cusip, pos, new)), (cusip, pos, new)


def test_cusip_special_characters_and_hand_example():
    # hand calc for 03783310: values 0,3,7,8,3,3,1,0; positions 2,4,6,8 doubled -> 0,6,7,16,3,6,1,0;
    # digit sums 0+6+7+(1+6)+3+6+1+0 = 30 -> check digit (10 - 30 % 10) % 10 = 0
    assert cusip_check_digit("03783310") == "0"
    # '*' = 36, '@' = 37, '#' = 38 are accepted as characters
    assert cusip_check_digit("0000000*") in "0123456789"
    assert cusip_check_digit("0000000@") != cusip_check_digit("0000000#")


@pytest.mark.parametrize("bad", ["", "N/A", None, np.nan, "03783310", "0378331000", "03783310O", "000000000",
                                 "037833100 0", 37833100])
def test_invalid_cusips(bad):
    assert not is_valid_cusip(bad)


def test_all_zero_cusip_is_rejected_although_its_check_digit_works():
    assert cusip_check_digit("00000000") == "0"  # checksum alone would pass "000000000"
    assert not is_valid_cusip("000000000")


def test_cusip_check_digit_rejects_bad_input():
    with pytest.raises(ValueError):
        cusip_check_digit("0378331")
    with pytest.raises(ValueError):
        cusip_check_digit("0378331!")


# -------------------------------------------------------------------- ISIN
@pytest.mark.parametrize("isin", GOOD_ISINS)
def test_isin_check_digit_and_validity(isin):
    assert isin_check_digit(isin[:11]) == isin[11]
    assert is_valid_isin(isin)
    assert is_valid_isin(isin.lower())


@pytest.mark.parametrize("isin", GOOD_ISINS)
def test_isin_single_digit_corruptions_are_caught(isin):
    # Luhn catches every single-digit substitution; letters are left alone (letter swaps are not
    # guaranteed to be caught by any check digit scheme that maps A-Z to two digits)
    for pos in range(2, len(isin)):
        if not isin[pos].isdigit():
            continue
        for new in "0123456789":
            if new != isin[pos]:
                assert not is_valid_isin(_swap_char(isin, pos, new)), (isin, pos, new)


@pytest.mark.parametrize("bad", ["", None, np.nan, "US037833100", "US03783310050", "0S0378331005",
                                 "US037833100O", "US0378331005 5"])
def test_invalid_isins(bad):
    assert not is_valid_isin(bad)


def test_cusip_isin_round_trip():
    for cusip, isin in KNOWN.values():
        derived = cusip_to_isin(cusip)
        assert is_valid_isin(derived) and derived[:2] == "US"
        if isin:
            assert derived == isin
        assert isin_to_cusip(derived) == cusip


def test_cusip_to_isin_canada_and_errors():
    isin = cusip_to_isin(APPLE, country="CA")
    assert isin.startswith("CA") and is_valid_isin(isin)
    assert isin_to_cusip(isin) == APPLE
    with pytest.raises(ValueError):
        cusip_to_isin("037833101")  # wrong check digit: no ISIN is made up from it
    with pytest.raises(ValueError):
        cusip_to_isin(APPLE, country="USA")


def test_isin_to_cusip_only_for_us_and_ca():
    assert isin_to_cusip("GB0002634946") is None
    assert isin_to_cusip("DE000BAY0017") is None
    assert isin_to_cusip("US0378331006") is None  # bad ISIN check digit
    assert isin_to_cusip(None) is None
    assert isin_to_cusip("US0378331005") == APPLE


# --------------------------------------------------------------------- LEI
@pytest.mark.parametrize("lei", GOOD_LEIS)
def test_lei_valid_and_corruptions_caught(lei):
    assert is_valid_lei(lei) and is_valid_lei(lei.lower())
    for pos, ch in enumerate(lei):
        # replace by a character that expands to the same number of decimal digits: the integer
        # then changes by (b - a) * 10**k with |b - a| < 97, which MOD 97-10 always detects
        pool = "0123456789" if ch.isdigit() else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for new in pool:
            if new != ch:
                assert not is_valid_lei(_swap_char(lei, pos, new)), (lei, pos, new)


@pytest.mark.parametrize("bad", ["", None, np.nan, "5493006MHB84DD0ZWV1", "5493006MHB84DD0ZWV180",
                                 "5493006MHB84DD0ZWV1!", "00000000000000000000"])
def test_invalid_leis(bad):
    assert not is_valid_lei(bad)


# ------------------------------------------------------------------ cleaning
@pytest.mark.parametrize("placeholder", [None, np.nan, pd.NA, "", "   ", "N/A", "n/a", "NA", "none", "None",
                                         "000000000", "0", "00000000000000000000"])
def test_clean_identifier_placeholders(placeholder):
    assert clean_identifier(placeholder) is None


def test_clean_identifier_keeps_real_values():
    assert clean_identifier("  us0378331005 ") == "US0378331005"
    assert clean_identifier("0378331") == "0378331"  # not a placeholder: validity is checked elsewhere
    assert clean_identifier("A00000000") == "A00000000"


def test_normalise_text():
    assert normalise_text("  Apple  Inc. ") == "APPLE INC"
    assert normalise_text("AT&T Inc.") == "AT T INC"
    assert normalise_text("Coca-Cola Co") == normalise_text("COCA COLA CO")
    assert normalise_text("McDonald's Corp") == "MCDONALDS CORP"
    assert normalise_text(None) == "" and normalise_text(np.nan) == ""


# ------------------------------------------------------------- security key
def test_security_key_precedence():
    isin_us = "US5949181045"  # Microsoft
    lei = "INR2EJN1ERAN0W5ZP974"
    # 1. a valid CUSIP wins over everything else
    assert security_key(cusip=APPLE, isin=isin_us, lei=lei, title="x", issuer_name="y") == ("cusip", APPLE)
    assert security_key(cusip=" 037833100 ") == ("cusip", APPLE)
    # 2. valid ISIN: US/CA reduce to the embedded CUSIP, other countries stay ISINs
    assert security_key(isin=isin_us, lei=lei, title="x") == ("cusip", MSFT)
    assert security_key(isin="GB0002634946", title="x") == ("isin", "GB0002634946")
    assert security_key(isin="de000bay0017") == ("isin", "DE000BAY0017")
    # 3. valid LEI needs a title
    assert security_key(lei=lei, title="Common  Stock.", issuer_name="Microsoft") == (
        "lei_title", f"{lei}|COMMON STOCK")
    # 4. name scheme
    assert security_key(issuer_name="Microsoft Corp.", title="Common Stock") == (
        "name", "MICROSOFT CORP|COMMON STOCK")
    assert security_key(lei=lei, issuer_name="Microsoft Corp") == ("name", "MICROSOFT CORP|")
    assert security_key(title="Treasury Bill") == ("name", "|TREASURY BILL")
    # 5. nothing usable
    assert security_key() == ("unknown", "")
    assert security_key(cusip="000000000", isin="N/A", lei="", title=" ", issuer_name=None) == ("unknown", "")


def test_invalid_cusip_is_not_silently_used():
    bad = "037833101"  # Apple with a wrong check digit
    assert security_key(cusip=bad) == ("unknown", "")
    assert security_key(cusip=bad, issuer_name="Apple Inc") == ("name", "APPLE INC|")
    # ... but a valid ISIN next to it is used, and it yields the real CUSIP
    assert security_key(cusip=bad, isin="US0378331005") == ("cusip", APPLE)
    # an invalid ISIN and an invalid LEI are skipped as well
    assert security_key(isin="US0378331006", lei="HWUPKR0MPOU8FGXBT395", title="t") == ("name", "|T")


def test_us_isin_with_invalid_embedded_cusip_stays_an_isin():
    # a valid ISIN whose national number is not a valid CUSIP (e.g. a CINS-like code) is not reduced
    body = "US" + "12345678Z"
    isin = body + isin_check_digit(body)
    assert is_valid_isin(isin) and isin_to_cusip(isin) is None
    assert security_key(isin=isin) == ("isin", isin)


# ------------------------------------------------------- add_security_keys
def _holdings():
    return pd.DataFrame({
        "row_index": range(1, 8),
        "issuer_name": ["Apple Inc", "Microsoft Corp", "Alphabet Inc", "Foo Corp", "Bar Ltd", None, "Apple Inc"],
        "issuer_lei": ["HWUPKR0MPOU8FGXBT394", "INR2EJN1ERAN0W5ZP974", None, None, None, None, None],
        "title": ["Common", "Common", "Class A", "Common", "Bond", None, "Common"],
        "cusip": [APPLE, "594918105", "000000000", "N/A", None, None, " 037833100"],  # row 2: bad check digit
        "isin": [None, "US5949181045", "US02079K3059", None, "GB0002634946", None, "US0378331005"],
        "balance": [100.0, 50, 10, 5, 1000, 3, 20],
        "value_usd": [1000.0, 500, 200, 50, 300, 10, 100],
    })


def test_add_security_keys_values():
    h = _holdings()
    out = add_security_keys(h)
    assert out["key_scheme"].tolist() == ["cusip", "cusip", "cusip", "name", "isin", "unknown", "cusip"]
    assert out["key_value"].tolist() == [APPLE, MSFT, GOOG, "FOO CORP|COMMON", "GB0002634946", "", APPLE]
    assert out["security_key"].tolist() == [f"cusip:{APPLE}", f"cusip:{MSFT}", f"cusip:{GOOG}",
                                            "name:FOO CORP|COMMON", "isin:GB0002634946", "unknown:", f"cusip:{APPLE}"]
    assert out["cusip_valid"].tolist() == [True, False, False, False, False, False, True]
    assert out["isin_valid"].tolist() == [False, True, True, False, True, False, True]
    # derived_isin: CUSIP valid and ISIN missing -> derived; else the (cleaned) filed ISIN; else None
    assert out["derived_isin"].iloc[0] == "US0378331005"
    assert out["derived_isin"].iloc[1] == "US5949181045"
    assert out["derived_isin"].iloc[2] == "US02079K3059"
    assert out["derived_isin"].iloc[3] is None and out["derived_isin"].iloc[5] is None
    assert out["derived_isin"].iloc[4] == "GB0002634946"
    assert out["derived_isin"].iloc[6] == "US0378331005"


def test_add_security_keys_leaves_filed_columns_untouched():
    h = _holdings()
    before = h.copy(deep=True)
    out = add_security_keys(h)
    pd.testing.assert_frame_equal(h, before)  # input not mutated
    pd.testing.assert_frame_equal(out[list(h.columns)], before)  # filed columns identical, placeholders kept
    assert set(out.columns) - set(h.columns) == {
        "key_scheme", "key_value", "security_key", "cusip_valid", "isin_valid", "derived_isin"}


def test_add_security_keys_missing_columns_and_empty_frame():
    out = add_security_keys(pd.DataFrame({"cusip": [APPLE, None]}))
    assert out["security_key"].tolist() == [f"cusip:{APPLE}", "unknown:"]
    empty = add_security_keys(pd.DataFrame({"cusip": [], "isin": []}))
    assert len(empty) == 0 and "security_key" in empty.columns


def test_derived_isin_country():
    h = pd.DataFrame({"cusip": [APPLE, APPLE, MSFT], "isin": [None, None, None]})
    ca = add_security_keys(h, country="CA")["derived_isin"].tolist()
    assert ca[0].startswith("CA") and is_valid_isin(ca[0])
    per_row = add_security_keys(h, country=pd.Series(["US", None, "ca"]))["derived_isin"].tolist()
    assert per_row[0] == "US0378331005"
    assert per_row[1] is None  # unknown country: no prefix is guessed
    assert per_row[2].startswith("CA")
    with pytest.raises(ValueError):
        add_security_keys(h, country="USA")


# -------------------------------------------------------- identifier_report
def test_identifier_report_counts():
    rep = identifier_report(_holdings())
    assert rep["n_rows"] == 7
    assert rep["n_by_scheme"] == {"cusip": 4, "isin": 1, "lei_title": 0, "name": 1, "unknown": 1}
    # filed CUSIPs: APPLE ok, "594918105" invalid, placeholders x3 (000000000, N/A, None, None), APPLE ok
    assert rep["n_invalid_cusip"] == 1
    assert rep["n_missing_cusip"] == 4
    # Apple appears on rows 1 and 7 (same security, two rows); the unknown row is not a duplicate
    assert rep["n_duplicate_keys"] == 1 and rep["n_duplicate_rows"] == 1
    # cusip-keyed value: 1000 + 500 + 200 + 100 = 1800 of 1000+500+200+50+300+10+100 = 2160
    assert rep["cusip_value_share"] == pytest.approx(1800 / 2160)


def test_identifier_report_value_share_edge_cases():
    no_value = identifier_report(pd.DataFrame({"cusip": [APPLE]}))
    assert np.isnan(no_value["cusip_value_share"])
    assert no_value["n_duplicate_keys"] == 0
    # negative (derivative) values count by magnitude, so the share stays in [0, 1]
    h = pd.DataFrame({"cusip": [APPLE, None], "issuer_name": ["A", "B"], "value_usd": [100.0, -300.0]})
    assert identifier_report(h)["cusip_value_share"] == pytest.approx(0.25)


# --------------------------------------------------------------- history
def test_build_identifier_history_first_last_seen():
    new_cusip = "88160R101"  # Tesla, standing in for the post-corporate-action CUSIP of "Old Co"
    q1, q2, q3 = date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30)
    snaps = {
        q3: pd.DataFrame({"cusip": [new_cusip, APPLE, None], "issuer_name": ["Old Co Holdings", "Apple Inc.", None]}),
        q1: pd.DataFrame({"cusip": [APPLE, MSFT, None, None],
                          "issuer_name": ["Apple Inc", "Microsoft", "Cash Sweep Fund", None]}),
        q2: pd.DataFrame({"cusip": [APPLE, MSFT, None], "issuer_name": ["Apple Inc", "Microsoft", "Cash Sweep Fund"]}),
    }
    hist = build_identifier_history(snaps).set_index("security_key")
    assert list(build_identifier_history(snaps).columns) == [
        "security_key", "scheme", "value", "first_seen", "last_seen", "n_snapshots", "names"]
    assert "unknown:" not in hist.index  # rows without any usable identifier are not a security

    a = hist.loc[f"cusip:{APPLE}"]
    assert (a.first_seen, a.last_seen, a.n_snapshots) == (q1, q3, 3)
    assert a.scheme == "cusip" and a.value == APPLE
    assert a.names == "Apple Inc | Apple Inc."  # distinct names, order of first appearance

    m = hist.loc[f"cusip:{MSFT}"]
    assert (m.first_seen, m.last_seen, m.n_snapshots) == (q1, q2, 2)

    t = hist.loc[f"cusip:{new_cusip}"]
    assert (t.first_seen, t.last_seen, t.n_snapshots, t.names) == (q3, q3, 1, "Old Co Holdings")

    s = hist.loc["name:CASH SWEEP FUND|"]
    assert (s.first_seen, s.last_seen, s.n_snapshots, s.scheme) == (q1, q2, 2, "name")

    # sorted by first_seen then key
    order = build_identifier_history(snaps)
    assert order["first_seen"].is_monotonic_increasing


def test_build_identifier_history_empty():
    h = build_identifier_history({})
    assert len(h) == 0 and "names" in h.columns
