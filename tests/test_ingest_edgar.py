import json
from datetime import date

import pytest

from fairbench.ingest.edgar import (FUND_TICKERS_URL, FilingMeta, filing_document_url, filing_index_url,
                                    normalise_accession, parse_fund_tickers, parse_series_atom,
                                    parse_submissions, select_latest, series_filings_url,
                                    submissions_url)

ACC = "0001003715-26-002722"
ACC18 = "000100371526002722"
REAL_INDEX = "https://www.sec.gov/Archives/edgar/data/866256/000100371526002722/0001003715-26-002722-index.htm"


# ---------------------------------------------------------------- URLs
def test_submissions_url():
    assert submissions_url(866256) == "https://data.sec.gov/submissions/CIK0000866256.json"
    assert submissions_url("0000866256") == submissions_url("866256")
    with pytest.raises(ValueError):
        submissions_url("not-a-cik")


def test_series_filings_url():
    assert series_filings_url("S000000856") == (
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=S000000856&type=NPORT-P"
        "&dateb=&owner=include&start=0&count=40&output=atom")
    assert series_filings_url("866256", form="NPORT-P", count=100, start=80).endswith(
        "&start=80&count=100&output=atom")
    assert "type=NPORT-P%2FA&" in series_filings_url("S000000856", form="NPORT-P/A")


@pytest.mark.parametrize("count", [101, 1000, 0, -1])
def test_series_filings_url_count_cap(count):
    with pytest.raises(ValueError, match="count"):
        series_filings_url("S000000856", count=count)


def test_series_filings_url_negative_start():
    with pytest.raises(ValueError, match="start"):
        series_filings_url("S000000856", start=-1)


def test_filing_urls_real_example():
    assert filing_index_url("0000866256", ACC) == REAL_INDEX
    assert filing_index_url(866256, ACC18) == REAL_INDEX
    assert filing_document_url(866256, ACC) == (
        "https://www.sec.gov/Archives/edgar/data/866256/000100371526002722/primary_doc.xml")
    assert filing_document_url("866256", ACC18, "other.xml").endswith("/000100371526002722/other.xml")
    with pytest.raises(ValueError):
        filing_index_url(866256, "12345")


def test_fund_tickers_url():
    assert FUND_TICKERS_URL == "https://www.sec.gov/files/company_tickers_mf.json"


def test_all_urls_are_sec_https():
    for u in (submissions_url(1), series_filings_url("S000000001"), filing_index_url(1, ACC),
              filing_document_url(1, ACC), FUND_TICKERS_URL):
        assert u.startswith(("https://www.sec.gov/", "https://data.sec.gov/"))


# ---------------------------------------------------------------- accession numbers
@pytest.mark.parametrize("raw", [ACC, ACC18, f"  {ACC}\n", f" {ACC18} "])
def test_normalise_accession(raw):
    assert normalise_accession(raw) == ACC


@pytest.mark.parametrize("bad", ["", "12345", "0001003715-26-00272", "0001003715-2600-2722",
                                 "000100371526-002722", "0001003715-26-002722-1", "00010037152600272x",
                                 "00010037152600272", None, 123])
def test_normalise_accession_rejects(bad):
    with pytest.raises(ValueError):
        normalise_accession(bad)


# ---------------------------------------------------------------- Atom feed of a series
def entry(acc: str, form: str, filed: str, cik_path: str = "866256") -> str:
    flat = acc.replace("-", "")
    href = f"https://www.sec.gov/Archives/edgar/data/{cik_path}/{flat}/{acc}-index.htm"
    return f"""
  <entry>
    <category label="form type" scheme="https://www.sec.gov/" term="{form}" />
    <content type="text/xml">
      <accession-number>{acc}</accession-number>
      <act></act>
      <file-number>811-05576</file-number>
      <file-number-href>https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&amp;filenum=811-05576</file-number-href>
      <filing-date>{filed}</filing-date>
      <filing-href>{href}</filing-href>
      <filing-type>{form}</filing-type>
      <film-number></film-number>
      <form-name>Monthly Portfolio Investments Report</form-name>
      <size>1 MB</size>
    </content>
    <id>urn:tag:sec.gov,2008:accession-number={acc}</id>
    <link href="{href}" rel="alternate" type="text/html" />
    <summary type="html">Filed {filed}</summary>
    <title>{form}  Monthly Portfolio Investments Report</title>
    <updated>{filed}T10:00:00-04:00</updated>
  </entry>"""


ATOM = f"""<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>EDGAR Filings for EXAMPLE FUND TRUST (fictional)</title>
  <updated>2026-10-09T08:00:00-04:00</updated>
  <company-info>
    <addresses><address type="mailing"><city>NOWHERE</city></address></addresses>
    <cik>0000866256</cik>
    <conformed-name>EXAMPLE FUND TRUST (fictional)</conformed-name>
    <fiscal-year-end>0731</fiscal-year-end>
    <state-of-incorporation>MA</state-of-incorporation>
  </company-info>
  {entry(ACC, "NPORT-P", "2026-07-20")}
  {entry("0001003715-26-001500", "NPORT-P/A", "2026-05-02")}
  {entry("0001003715-26-001100", "NPORT-P", "2026-04-28")}
</feed>""".encode("iso-8859-1")


def test_parse_series_atom():
    info, filings = parse_series_atom(ATOM)
    assert info == {"cik": "0000866256", "name": "EXAMPLE FUND TRUST (fictional)", "fiscal_year_end": "0731"}
    assert len(filings) == 3
    first = filings[0]
    assert first == FilingMeta(cik="0000866256", accession=ACC, form="NPORT-P",
                               filing_date=date(2026, 7, 20), report_date=None, primary_document=None,
                               index_url=REAL_INDEX, is_amendment=False)
    assert [f.accession for f in filings] == [ACC, "0001003715-26-001500", "0001003715-26-001100"]
    assert [f.is_amendment for f in filings] == [False, True, False]
    assert filings[1].form == "NPORT-P/A" and filings[1].filing_date == date(2026, 5, 2)
    assert all(f.report_date is None and f.primary_document is None for f in filings)


def test_parse_series_atom_empty_feed_and_index_fallback():
    empty = b"""<feed xmlns="http://www.w3.org/2005/Atom"><company-info><cik>866256</cik>
        <conformed-name>X (fictional)</conformed-name></company-info></feed>"""
    info, filings = parse_series_atom(empty)
    assert info == {"cik": "0000866256", "name": "X (fictional)", "fiscal_year_end": None}
    assert filings == []
    no_href = ATOM.replace(f"<filing-href>{REAL_INDEX}</filing-href>".encode(), b"")
    assert parse_series_atom(no_href)[1][0].index_url == REAL_INDEX


def test_parse_series_atom_errors():
    with pytest.raises(ValueError, match="Atom"):
        parse_series_atom(b"<html/>")
    with pytest.raises(ValueError, match="well-formed"):
        parse_series_atom(b"<feed")
    with pytest.raises(ValueError, match="company-info/cik"):
        parse_series_atom(b'<feed xmlns="http://www.w3.org/2005/Atom"/>')
    with pytest.raises(ValueError, match="entry 1"):  # a row is never skipped silently
        parse_series_atom(ATOM.replace(b"<accession-number>", b"<accession-numberX>")
                          .replace(b"</accession-number>", b"</accession-numberX>"))
    with pytest.raises(ValueError, match="filing-date"):
        parse_series_atom(ATOM.replace(b"2026-07-20</filing-date>", b"</filing-date>"))


# ---------------------------------------------------------------- submissions JSON
RECENT = {
    "accessionNumber": [ACC, "0001003715-26-001500", "0001003715-25-009000"],
    "filingDate": ["2026-07-20", "2026-05-02", "2025-12-01"],
    "reportDate": ["2026-06-30", "", "2025-10-31"],
    "form": ["NPORT-P", "NPORT-P/A", "N-CSR"],
    "primaryDocument": ["primary_doc.xml", "primary_doc.xml", ""],
    "size": [1, 2, 3],
}
SUBMISSIONS = json.dumps({
    "cik": "866256", "name": "EXAMPLE FUND TRUST (fictional)", "sic": "", "sicDescription": "",
    "fiscalYearEnd": "0731",
    "filings": {"recent": RECENT,
                "files": [{"name": "CIK0000866256-submissions-001.json", "filingCount": 10},
                          {"name": "CIK0000866256-submissions-002.json", "filingCount": 5}]},
}).encode()


def test_parse_submissions_full_document():
    info, filings, pages = parse_submissions(SUBMISSIONS)
    assert info == {"cik": "0000866256", "name": "EXAMPLE FUND TRUST (fictional)", "sic": None,
                    "sic_description": None, "fiscal_year_end": "0731"}
    assert pages == ["https://data.sec.gov/submissions/CIK0000866256-submissions-001.json",
                     "https://data.sec.gov/submissions/CIK0000866256-submissions-002.json"]
    assert filings[0] == FilingMeta(cik="0000866256", accession=ACC, form="NPORT-P",
                                    filing_date=date(2026, 7, 20), report_date=date(2026, 6, 30),
                                    primary_document="primary_doc.xml", index_url=REAL_INDEX,
                                    is_amendment=False)
    assert filings[1].report_date is None and filings[1].is_amendment  # "" -> None
    assert filings[2].primary_document is None and filings[2].form == "N-CSR"
    assert [f.report_date for f in filings] == [date(2026, 6, 30), None, date(2025, 10, 31)]


def test_parse_submissions_sic_filled():
    doc = json.loads(SUBMISSIONS)
    doc["sic"], doc["sicDescription"] = "6726", "Unit Investment Trusts, Face-Amount Certificate Offices"
    info, _, _ = parse_submissions(json.dumps(doc).encode())
    assert info["sic"] == "6726" and info["sic_description"].startswith("Unit Investment")


def test_parse_submissions_older_page_shape():
    page = json.dumps({k: v[:2] for k, v in RECENT.items()}).encode()
    info, filings, pages = parse_submissions(page)
    assert info == {} and pages == [] and len(filings) == 2
    assert filings[0].cik is None and filings[0].index_url is None  # the page carries no CIK
    assert filings[0].accession == ACC and filings[0].report_date == date(2026, 6, 30)
    _, with_cik, _ = parse_submissions(page, cik=866256)
    assert with_cik[0] == parse_submissions(SUBMISSIONS)[1][0]
    assert with_cik[1].index_url.startswith("https://www.sec.gov/Archives/edgar/data/866256/")


def test_parse_submissions_optional_arrays_absent():
    slim = {k: RECENT[k] for k in ("accessionNumber", "filingDate", "form")}
    _, filings, _ = parse_submissions(json.dumps({"cik": 866256, "name": "X", "filings": {"recent": slim}}).encode())
    assert [f.report_date for f in filings] == [None] * 3 and [f.primary_document for f in filings] == [None] * 3


def test_parse_submissions_errors():
    with pytest.raises(ValueError, match="not a submissions document"):
        parse_submissions(b'{"hello": 1}')
    bad = {k: v[:2] if k == "form" else v for k, v in RECENT.items()}
    with pytest.raises(ValueError, match="form"):
        parse_submissions(json.dumps(bad).encode())
    with pytest.raises(ValueError, match="filingDate"):
        parse_submissions(json.dumps({**RECENT, "filingDate": ["2026-07-20", "", "2025-12-01"]}).encode())


# ---------------------------------------------------------------- company_tickers_mf.json
def test_parse_fund_tickers_standard_and_shuffled_order():
    rows = [[52848, "S000004440", "C000012204", "VFTNX"], [1, "S000000001", "C000000001", "ABCDX"]]
    std = json.dumps({"fields": ["cik", "seriesId", "classId", "symbol"], "data": rows}).encode()
    df = parse_fund_tickers(std)
    assert list(df.columns) == ["cik", "series_id", "class_id", "symbol"]
    assert df["cik"].tolist() == ["0000052848", "0000000001"]
    assert df["series_id"].tolist() == ["S000004440", "S000000001"]
    assert df["class_id"].tolist() == ["C000012204", "C000000001"]
    assert df["symbol"].tolist() == ["VFTNX", "ABCDX"]

    order = ["symbol", "classId", "cik", "seriesId"]
    idx = {"cik": 0, "seriesId": 1, "classId": 2, "symbol": 3}
    shuffled = json.dumps({"fields": order, "data": [[r[idx[f]] for f in order] for r in rows]}).encode()
    pd_equal = parse_fund_tickers(shuffled)
    assert pd_equal.equals(df)


def test_parse_fund_tickers_extra_field_empty_and_errors():
    extra = json.dumps({"fields": ["cik", "seriesId", "classId", "symbol", "name"],
                        "data": [[5, "S1", "C1", "AAAAX", "ignored"]]}).encode()
    df = parse_fund_tickers(extra)
    assert list(df.columns) == ["cik", "series_id", "class_id", "symbol"] and df["cik"].tolist() == ["0000000005"]
    empty = parse_fund_tickers(json.dumps({"fields": ["cik", "seriesId", "classId", "symbol"], "data": []}).encode())
    assert len(empty) == 0 and list(empty.columns) == ["cik", "series_id", "class_id", "symbol"]
    with pytest.raises(ValueError, match="symbol"):
        parse_fund_tickers(json.dumps({"fields": ["cik", "seriesId", "classId"], "data": []}).encode())
    with pytest.raises(ValueError, match="row 0"):
        parse_fund_tickers(json.dumps({"fields": ["cik", "seriesId", "classId", "symbol"],
                                       "data": [[1, "S1", "C1"]]}).encode())


# ---------------------------------------------------------------- select_latest
def meta(acc: str, form: str, filed: str, reported: str | None, cik: str = "0000866256") -> FilingMeta:
    return FilingMeta(cik=cik, accession=acc, form=form, filing_date=date.fromisoformat(filed),
                      report_date=None if reported is None else date.fromisoformat(reported),
                      primary_document="primary_doc.xml", index_url=filing_index_url(cik, acc),
                      is_amendment=form.endswith("/A"))


def test_select_latest_prefers_amendment_and_sorts():
    q1 = meta("0000000001-26-000001", "NPORT-P", "2026-05-28", "2026-03-31")
    q1a = meta("0000000001-26-000002", "NPORT-P/A", "2026-06-10", "2026-03-31")
    q4 = meta("0000000001-25-000009", "NPORT-P", "2026-01-29", "2025-12-31")
    q2 = meta("0000000001-26-000003", "NPORT-P", "2026-08-27", "2026-06-30")
    got = select_latest([q1a, q2, q1, q4])
    assert got == [q4, q1a, q2]


def test_select_latest_same_day_amendment_beats_original():
    orig = meta("0000000001-26-000005", "NPORT-P", "2026-05-28", "2026-03-31")
    amend = meta("0000000001-26-000004", "NPORT-P/A", "2026-05-28", "2026-03-31")  # lower accession
    assert select_latest([orig, amend]) == [amend]
    assert select_latest([amend, orig]) == [amend]


def test_select_latest_without_report_dates_groups_by_filing_date():
    a = meta("0000000001-26-000001", "NPORT-P", "2026-05-28", None)
    b = meta("0000000001-26-000002", "NPORT-P/A", "2026-05-28", None)
    c = meta("0000000001-26-000003", "NPORT-P", "2026-08-27", None)
    assert select_latest([c, a, b]) == [b, c]


def test_select_latest_keeps_distinct_registrants_forms_and_periods():
    a = meta("0000000001-26-000001", "NPORT-P", "2026-05-28", "2026-03-31")
    other_cik = meta("0000000002-26-000001", "NPORT-P", "2026-05-28", "2026-03-31", cik="0000000002")
    other_form = meta("0000000001-26-000007", "N-CSR", "2026-05-28", "2026-03-31")
    known_vs_unknown = meta("0000000001-26-000008", "NPORT-P", "2026-03-31", None)  # filing date == a's period
    got = select_latest([a, other_cik, other_form, known_vs_unknown])
    assert len(got) == 4
    assert select_latest([]) == []
