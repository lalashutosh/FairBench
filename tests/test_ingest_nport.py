import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fairbench.ingest.nport import HOLDING_COLUMNS, NportReport, parse_nport
from fairbench.ingest.nport_synth import build_nport_xml

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "nport_minimal.xml"
NS = "http://www.sec.gov/edgar/nport"


@pytest.fixture(scope="module")
def raw() -> bytes:
    return FIXTURE.read_bytes()


@pytest.fixture(scope="module")
def report(raw) -> NportReport:
    return parse_nport(raw)


def nan_eq(a, b) -> bool:
    return (np.isnan(a) and np.isnan(b)) or a == b


def py(s: pd.Series) -> list:
    """Series as a list with every missing value None, whatever the dtype."""
    return [None if pd.isna(x) else x for x in s.astype(object)]


# ---------------------------------------------------------------- the fixture, field by field
def test_header_and_general_info(report):
    r = report
    assert r.submission_type == "NPORT-P"
    assert r.registrant_name == "Synthetic Example Trust (not a real fund)"
    assert r.registrant_cik == "0000000001" and len(r.registrant_cik) == 10
    assert r.registrant_lei == "SYNTHETICLEI00000001"
    assert r.file_number == "811-00000"
    assert r.series_name == "Synthetic Example Fund (not a real fund)"
    assert r.series_id == "S000000001"
    assert r.series_lei == "SYNTHETICLEI00000002"
    assert r.class_ids == ["C000000001", "C000000002"]
    assert r.fiscal_year_end == date(2026, 9, 30)
    assert r.report_date == date(2026, 3, 31)
    assert r.is_final_filing is False


def test_fund_totals(report):
    assert report.total_assets == 10_050_000.0
    assert report.total_liabilities == 50_000.0
    assert report.net_assets == 10_000_000.0
    assert report.misc_securities_assets == 25_000.0
    assert report.cash_not_reported == 15_000.0


def test_monthly_returns_are_percent_with_na_as_nan(report):
    m = report.monthly_returns
    assert list(m.columns) == ["class_id", "month", "month_end", "return_pct"]
    assert m["class_id"].tolist() == ["C000000001"] * 3 + ["C000000002"] * 3
    assert m["month"].tolist() == [1, 2, 3, 1, 2, 3]
    assert m["month_end"].tolist() == [date(2026, 1, 31), date(2026, 2, 28), date(2026, 3, 31)] * 2
    got = m["return_pct"].to_numpy()
    want = np.array([1.25, -0.75, 2.5, 1.20, np.nan, 2.45])
    np.testing.assert_array_equal(got, want)  # as filed: percent, NaN for "N/A"


def test_flows_missing_attribute_is_nan(report):
    f = report.flows
    assert list(f.columns) == ["month", "month_end", "sales", "reinvestment", "redemption"]
    assert f["month"].tolist() == [1, 2, 3]
    assert f["month_end"].tolist() == [date(2026, 1, 31), date(2026, 2, 28), date(2026, 3, 31)]
    np.testing.assert_array_equal(f["sales"], [100_000.0, 120_000.0, 90_000.0])
    np.testing.assert_array_equal(f["redemption"], [40_000.0, 30_000.0, 50_000.0])
    np.testing.assert_array_equal(f["reinvestment"], [5_000.0, np.nan, 4_000.0])


def test_other_gains(report):
    g = report.other_gains
    assert list(g.columns) == ["month", "net_realized_gain", "net_unrealized_appreciation"]
    assert g["month"].tolist() == [1, 2, 3]
    np.testing.assert_array_equal(g["net_realized_gain"], [1000.0, -500.0, 0.0])
    np.testing.assert_array_equal(g["net_unrealized_appreciation"], [2000.0, 1500.5, -250.25])


def test_holdings_columns_exact():
    assert HOLDING_COLUMNS == [
        "row_index", "issuer_name", "issuer_lei", "title", "cusip", "isin", "ticker", "other_id",
        "other_id_desc", "balance", "units", "currency", "exchange_rate", "value_usd",
        "pct_net_assets", "payoff_profile", "asset_category", "issuer_category", "country",
        "is_restricted", "fair_value_level", "is_loaned", "loan_value", "has_derivative_info"]


def test_holdings_every_row_and_field(report):
    h = report.holdings
    assert list(h.columns) == HOLDING_COLUMNS
    assert len(h) == 6  # none dropped
    assert h["row_index"].tolist() == list(range(6))
    assert py(h["issuer_name"]) == [
        "Alpha Widgets Inc (fictional)", "Beta Software Corp (fictional)",
        "Gamma Maschinenbau AG (fictional)", "Synthetic Cash Reserves Fund (fictional)",
        "Delta Retail Holdings (fictional)", "Epsilon Private Placement LLC (fictional)"]
    assert py(h["issuer_lei"]) == [f"SYNTHETICLEI000000{n}" for n in (11, 12, 13, 14, 15)] + [None]
    assert py(h["title"])[0] == "Alpha Widgets Inc Common Stock"
    assert py(h["cusip"]) == ["ZZ0000011", "ZZ0000028", "ZZ0000036", "ZZ0000044", "ZZ0000051", "000000000"]
    assert py(h["isin"]) == ["US0000000011", None, "DE0000000013", None, "US0000000051", None]
    assert py(h["ticker"]) == [None, "BETA", None, "SCRXX", "DLTA", None]  # first of two tickers
    assert py(h["other_id"]) == [None, "B000001", None, None, None, "EPS-PP-1"]
    assert py(h["other_id_desc"]) == [None, "SEDOL", None, None, None, "Internal fund code"]
    np.testing.assert_array_equal(h["balance"], [10_000.0, 8_000.0, 5_000.0, 1_000_000.0, 6_000.0, 1.0])
    assert py(h["units"]) == ["NS", "NS", "NS", "NS", "NS", "OU"]
    assert py(h["currency"]) == ["USD", "USD", "EUR", "USD", "USD", "USD"]
    np.testing.assert_array_equal(h["exchange_rate"], [np.nan, np.nan, 0.92, np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(h["value_usd"], [2.5e6, 2.0e6, 1.5e6, 1.0e6, 1.8e6, 7.0e5])
    np.testing.assert_array_equal(h["pct_net_assets"], [25.0, 20.0, 15.0, 10.0, 18.0, 7.0])  # percent
    assert py(h["payoff_profile"]) == ["Long"] * 6
    assert py(h["asset_category"]) == ["EC", "EC", "EC", "STIV", "EC", "OTHER"]
    assert py(h["issuer_category"]) == ["CORP", "CORP", "CORP", "RF", "CORP", "OTHER"]
    assert py(h["country"]) == ["US", "US", "DE", "US", "US", "KY"]
    assert h["is_restricted"].tolist() == [False] * 5 + [True]
    assert py(h["fair_value_level"]) == ["1", "1", "2", "1", "1", "3"]
    assert h["is_loaned"].tolist() == [False, False, False, False, True, False]
    np.testing.assert_array_equal(h["loan_value"], [np.nan] * 4 + [450_000.0, np.nan])
    assert h["has_derivative_info"].tolist() == [False] * 5 + [True]
    assert h["has_derivative_info"].dtype == bool


def test_na_and_placeholders(report):
    h = report.holdings
    assert h.loc[5, "issuer_lei"] is None  # "N/A" -> None
    assert h.loc[5, "cusip"] == "000000000"  # placeholder cusip is kept as filed
    assert np.isnan(report.monthly_returns.loc[4, "return_pct"])  # "N/A"
    for col in ("title", "cusip", "isin", "ticker"):  # nothing is guessed
        assert h[col].map(lambda v: v is None or isinstance(v, str)).all()


def test_equity_sleeve_weights_and_exclusions(report):
    s = report.equity_sleeve()
    assert s.index.tolist() == [0, 1, 2, 4]  # STIV (3) and OTHER / OU (5) are out
    assert s["asset_category"].eq("EC").all() and s["units"].eq("NS").all()
    assert s["sleeve_weight"].sum() == pytest.approx(1.0)
    np.testing.assert_allclose(s["sleeve_weight"], np.array([2.5, 2.0, 1.5, 1.8]) / 7.8)
    assert "sleeve_weight" not in report.holdings.columns  # the report itself is untouched
    assert (s["pct_net_assets"] == report.holdings.loc[s.index, "pct_net_assets"]).all()  # still percent


def test_implied_price(report):
    p = report.implied_price()
    assert p.index.tolist() == report.equity_sleeve().index.tolist()
    np.testing.assert_allclose(p.to_numpy(), [250.0, 250.0, 300.0, 300.0])


def test_equity_sleeve_filters_each_condition(report):
    def sleeve_after(row, col, value):
        h = report.holdings.copy()
        h.loc[row, col] = value
        r = NportReport(**{**report.__dict__, "holdings": h})
        return r.equity_sleeve().index.tolist()

    assert sleeve_after(0, "units", "PA") == [1, 2, 4]
    assert sleeve_after(0, "payoff_profile", "Short") == [1, 2, 4]
    assert sleeve_after(0, "balance", 0.0) == [1, 2, 4]
    assert sleeve_after(0, "value_usd", -1.0) == [1, 2, 4]
    assert sleeve_after(0, "balance", np.nan) == [1, 2, 4]
    assert sleeve_after(3, "asset_category", "EC") == [0, 1, 2, 3, 4]


def test_empty_sleeve_is_empty_not_error(report):
    h = report.holdings.copy()
    h["asset_category"] = "DBT"
    r = NportReport(**{**report.__dict__, "holdings": h})
    assert len(r.equity_sleeve()) == 0 and len(r.implied_price()) == 0


def test_locator(report):
    assert report.locator(0) == "invstOrSecs/invstOrSec[1]"
    assert report.locator(5) == "invstOrSecs/invstOrSec[6]"
    for bad in (-1, 6):
        with pytest.raises(IndexError):
            report.locator(bad)


def test_locator_points_at_the_source_element(raw, report):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(raw)
    block = root.find(f"{{{NS}}}formData/{{{NS}}}invstOrSecs")
    for i in range(len(report.holdings)):
        el = block.findall(f"{{{NS}}}invstOrSec")[i]  # the i+1-th element, as the locator says
        assert el.find(f"{{{NS}}}name").text == report.holdings.loc[i, "issuer_name"]


# ---------------------------------------------------------------- input variants
def test_str_input(raw):
    assert parse_nport(raw.decode("utf-8")).holdings.shape == (6, len(HOLDING_COLUMNS))


def test_prefixed_and_other_namespace_variants(raw, report):
    text = raw.decode("utf-8")
    prefixed = re.sub(r"<(/?)([A-Za-z])", r"<\1np:\2", text)  # every element gets a prefix
    prefixed = prefixed.replace(f'xmlns="{NS}"', f'xmlns:np="{NS}/v2"')  # and another URI
    assert "<np:edgarSubmission" in prefixed and f"{NS}/v2" in prefixed
    unqualified = text.replace(f'xmlns="{NS}"', "")
    for variant in (prefixed, unqualified):
        r = parse_nport(variant)
        pd.testing.assert_frame_equal(r.holdings, report.holdings)
        pd.testing.assert_frame_equal(r.monthly_returns, report.monthly_returns)
        assert r.report_date == report.report_date and r.class_ids == report.class_ids


def test_month_end_leap_year(raw):
    r = parse_nport(raw.replace(b"2026-03-31", b"2024-03-31"))
    assert r.flows["month_end"].tolist() == [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31)]


def test_month_end_crosses_year(raw):
    r = parse_nport(raw.replace(b"2026-03-31", b"2026-01-31"))
    assert r.flows["month_end"].tolist() == [date(2025, 11, 30), date(2025, 12, 31), date(2026, 1, 31)]


def test_missing_optional_blocks(raw):
    minimal = (b'<edgarSubmission xmlns="' + NS.encode() + b'"><formData><genInfo>'
               b"<repPdDate>2026-03-31</repPdDate></genInfo></formData></edgarSubmission>")
    r = parse_nport(minimal)
    assert r.report_date == date(2026, 3, 31)
    assert r.registrant_name is None and r.registrant_cik is None and r.fiscal_year_end is None
    assert r.is_final_filing is None and r.class_ids == [] and r.submission_type is None
    assert np.isnan(r.net_assets) and np.isnan(r.cash_not_reported)
    assert len(r.holdings) == 0 and list(r.holdings.columns) == HOLDING_COLUMNS
    assert len(r.monthly_returns) == 0 and list(r.monthly_returns.columns)[0] == "class_id"
    assert len(r.flows) == 3 and r.flows[["sales", "reinvestment", "redemption"]].isna().all().all()
    assert r.other_gains["net_realized_gain"].isna().all()
    assert len(r.equity_sleeve()) == 0


def test_holding_without_optional_children():
    xml = (b'<edgarSubmission><formData><genInfo><repPdDate>2026-03-31</repPdDate></genInfo>'
           b"<invstOrSecs><invstOrSec><name>Bare Co (fictional)</name></invstOrSec></invstOrSecs>"
           b"</formData></edgarSubmission>")
    h = parse_nport(xml).holdings
    assert len(h) == 1 and h.loc[0, "issuer_name"] == "Bare Co (fictional)"
    for col in ("issuer_lei", "title", "cusip", "isin", "ticker", "other_id", "units", "currency",
                "payoff_profile", "asset_category", "issuer_category", "country",
                "is_restricted", "fair_value_level", "is_loaned"):
        assert h.loc[0, col] is None, col
    for col in ("balance", "exchange_rate", "value_usd", "pct_net_assets", "loan_value"):
        assert np.isnan(h.loc[0, col]), col
    assert h.loc[0, "has_derivative_info"] == False  # noqa: E712


# ---------------------------------------------------------------- errors
@pytest.mark.parametrize("bad", [
    b"<html><body>not a filing</body></html>",
    b'<feed xmlns="http://www.w3.org/2005/Atom"/>',
    b"<edgarSubmission><headerData><submissionType>13F-HR</submissionType></headerData></edgarSubmission>",
])
def test_non_nport_root_raises(bad):
    with pytest.raises(ValueError, match="not an N-PORT submission"):
        parse_nport(bad)


def test_malformed_xml_raises():
    with pytest.raises(ValueError, match="well-formed"):
        parse_nport(b"<edgarSubmission><formData>")


def test_missing_report_date_raises(raw):
    with pytest.raises(ValueError, match="repPdDate"):
        parse_nport(raw.replace(b"repPdDate", b"repPdDateX"))
    with pytest.raises(ValueError, match="repPdDate"):  # N/A counts as missing
        parse_nport(raw.replace(b"<repPdDate>2026-03-31<", b"<repPdDate>N/A<"))


def test_present_but_malformed_values_raise(raw):
    with pytest.raises(ValueError, match="valUSD"):
        parse_nport(raw.replace(b"<valUSD>2500000.00<", b"<valUSD>lots<"))
    with pytest.raises(ValueError, match="isRestrictedSec"):
        parse_nport(raw.replace(b"<isRestrictedSec>Y<", b"<isRestrictedSec>maybe<"))
    with pytest.raises(ValueError, match="repPdEnd"):
        parse_nport(raw.replace(b"2026-09-30", b"Sept 30"))


# ---------------------------------------------------------------- build_nport_xml round trip
def random_holdings(n: int = 30, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cur = rng.choice(["USD", "EUR", "JPY"], size=n, p=[0.6, 0.2, 0.2])
    value = np.round(rng.lognormal(12, 1.5, size=n), 2)
    bal = rng.lognormal(8, 2, size=n) * rng.choice([1.0, 0.1], size=n)  # many decimals
    loaned = rng.random(n) < 0.2
    return pd.DataFrame({
        "issuer_name": [f"Issuer {i} & Co <fictional>" for i in range(n)],
        "issuer_lei": [None if rng.random() < 0.3 else f"SYNTHETICLEI{i:08d}" for i in range(n)],
        "title": [f"Issuer {i} Common Stock" for i in range(n)],
        "cusip": [("000000000" if rng.random() < 0.2 else f"{rng.integers(10**8, 10**9)}") for _ in range(n)],
        "isin": [f"US{i:010d}" if i % 3 == 0 else None for i in range(n)],
        "ticker": [f"TK{i}" if i % 2 == 0 else None for i in range(n)],
        "other_id": [f"OID{i}" if i % 5 == 0 else None for i in range(n)],
        "other_id_desc": [f"Desc {i}" if i % 5 == 0 else None for i in range(n)],
        "balance": bal,
        "units": rng.choice(["NS", "PA", "OU"], size=n, p=[0.8, 0.1, 0.1]),
        "currency": cur,
        "exchange_rate": np.where(cur == "USD", np.nan, rng.uniform(0.5, 150, size=n)),
        "value_usd": value,
        "pct_net_assets": rng.uniform(0, 5, size=n),
        "payoff_profile": rng.choice(["Long", "Short"], size=n, p=[0.9, 0.1]),
        "asset_category": rng.choice(["EC", "EP", "DBT", "STIV", "OTHER"], size=n, p=[0.6, 0.1, 0.1, 0.1, 0.1]),
        "issuer_category": rng.choice(["CORP", "RF", "UST", "OTHER"], size=n),
        "country": rng.choice(["US", "DE", "JP", None], size=n),
        "is_restricted": [bool(x) for x in rng.random(n) < 0.1],
        "fair_value_level": rng.choice(["1", "2", "3", None], size=n),
        "is_loaned": [bool(x) for x in loaned],
        "loan_value": np.where(loaned, np.round(value * 0.3, 2), np.nan),
        "has_derivative_info": [bool(x) for x in rng.random(n) < 0.1],
    })


def build(holdings: pd.DataFrame, **kw) -> bytes:
    args = dict(registrant_name="Synthetic Example Trust (not a real fund)", registrant_cik="0000000001",
                series_name="Synthetic Example Fund (not a real fund)", series_id="S000000001",
                class_ids=["C000000001", "C000000002"], fiscal_year_end=date(2026, 9, 30),
                report_date=date(2026, 3, 31), net_assets=1_234_567.89, holdings=holdings)
    return build_nport_xml(**{**args, **kw})


def test_round_trip_random_holdings():
    src = random_holdings()
    r = parse_nport(build(src))
    got = r.holdings
    assert list(got.columns) == HOLDING_COLUMNS and len(got) == 30
    assert got["row_index"].tolist() == list(range(30))
    for col in ("balance", "exchange_rate", "value_usd", "pct_net_assets", "loan_value"):
        np.testing.assert_array_equal(got[col].to_numpy(), src[col].to_numpy(), err_msg=col)
        assert np.allclose(got[col], src[col], equal_nan=True, rtol=1e-12, atol=0)
    for col in ("issuer_name", "issuer_lei", "title", "cusip", "isin", "ticker", "other_id",
                "other_id_desc", "units", "currency", "payoff_profile", "asset_category",
                "issuer_category", "country", "fair_value_level"):
        assert py(got[col]) == py(src[col]), col
    for col in ("is_restricted", "is_loaned", "has_derivative_info"):
        assert got[col].tolist() == src[col].tolist(), col


def test_round_trip_is_stable_on_the_fixture_holdings(report):
    again = parse_nport(build(report.holdings))
    pd.testing.assert_frame_equal(again.holdings, report.holdings)


def test_round_trip_header_fields():
    src = random_holdings(5)
    rets = {"C000000001": (1.5, -0.25, 3.125), "C000000002": (1.4, float("nan"), 3.0)}
    flows = [{"sales": 1.0, "reinvestment": 2.0, "redemption": 3.0}, {"sales": 10.5}, {}]
    r = parse_nport(build(src, monthly_returns=rets, flows=flows, total_liabilities=234.5,
                          submission_type="NPORT-P/A", file_number="811-12345"))
    assert (r.submission_type, r.file_number) == ("NPORT-P/A", "811-12345")
    assert r.registrant_name == "Synthetic Example Trust (not a real fund)"
    assert r.registrant_cik == "0000000001" and r.series_id == "S000000001"
    assert r.class_ids == ["C000000001", "C000000002"]
    assert (r.fiscal_year_end, r.report_date) == (date(2026, 9, 30), date(2026, 3, 31))
    assert r.net_assets == 1_234_567.89 and r.total_liabilities == 234.5
    assert r.total_assets == 1_234_567.89 + 234.5  # default
    assert np.isnan(r.misc_securities_assets) and np.isnan(r.cash_not_reported)
    np.testing.assert_array_equal(r.monthly_returns["return_pct"],
                                  [1.5, -0.25, 3.125, 1.4, np.nan, 3.0])
    np.testing.assert_array_equal(r.flows["sales"], [1.0, 10.5, np.nan])
    np.testing.assert_array_equal(r.flows["reinvestment"], [2.0, np.nan, np.nan])
    np.testing.assert_array_equal(r.flows["redemption"], [3.0, np.nan, np.nan])


def test_explicit_total_assets_and_no_returns():
    r = parse_nport(build(random_holdings(3), total_assets=2_000_000.0))
    assert r.total_assets == 2_000_000.0 and len(r.monthly_returns) == 0


def test_defaults_for_missing_columns_and_computed_pct():
    src = pd.DataFrame({"issuer_name": ["Fictional A", "Fictional B"], "value_usd": [100.0, 250.5],
                        "balance": [10.0, 5.0], "units": ["NS", "NS"]})
    h = parse_nport(build(src, net_assets=1000.0)).holdings
    assert h["currency"].tolist() == ["USD", "USD"]
    assert h["payoff_profile"].tolist() == ["Long", "Long"]
    assert h["asset_category"].tolist() == ["EC", "EC"]
    assert h["issuer_category"].tolist() == ["CORP", "CORP"]
    assert h["country"].tolist() == ["US", "US"]
    assert h["is_restricted"].tolist() == [False, False]
    assert h["fair_value_level"].tolist() == ["1", "1"]
    assert h["is_loaned"].tolist() == [False, False]
    assert h["has_derivative_info"].tolist() == [False, False]
    np.testing.assert_allclose(h["pct_net_assets"], [10.0, 25.05])  # 100 * value / net assets
    # columns without a documented default stay absent rather than being invented
    assert h["title"].isna().all() and h["cusip"].isna().all() and h["isin"].isna().all()
    assert np.isnan(h["exchange_rate"]).all()


def test_null_in_present_column_is_absent_not_defaulted():
    src = pd.DataFrame({"issuer_name": ["A"], "value_usd": [1.0], "country": [None],
                        "is_restricted": [None], "is_loaned": [None]}, dtype=object)
    src["value_usd"] = src["value_usd"].astype(float)
    h = parse_nport(build(src)).holdings
    assert h.loc[0, "country"] is None and h.loc[0, "is_restricted"] is None
    assert h.loc[0, "is_loaned"] is None


def test_synthetic_output_is_namespaced_n_port():
    xml = build(random_holdings(2))
    text = xml.decode("utf-8")
    assert text.startswith("<?xml")
    assert f'xmlns="{NS}"' in text and "xmlns:ncom=" in text
    assert text.index("<headerData>") < text.index("<genInfo>") < text.index("<fundInfo>") \
        < text.index("<invstOrSecs>")
    assert "<name>" in text and "<valUSD>" in text
    assert re.search(r"\d[eE][+-]?\d", text) is None  # no exponent notation


def test_builder_rejects_bad_holdings():
    with pytest.raises(ValueError, match="required"):
        build(pd.DataFrame({"issuer_name": ["A"]}))
    with pytest.raises(ValueError, match="outside HOLDING_COLUMNS"):
        build(pd.DataFrame({"issuer_name": ["A"], "value_usd": [1.0], "colour": ["red"]}))
    with pytest.raises(ValueError, match="at most 3"):
        build(random_holdings(1), flows=[{}, {}, {}, {}])
