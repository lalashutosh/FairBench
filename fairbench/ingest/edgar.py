"""EDGAR listings: URL builders and parsers for the filing lists SEC publishes. No I/O.

Contract
    Pure functions over URLs, bytes and dataclasses; fetching is the caller's job. Every URL
    points at www.sec.gov or data.sec.gov. A value a listing does not carry is None, never a
    guess: the Atom feed has no report date, so ``FilingMeta.report_date`` is None for it.

Sources
    submissions JSON    https://data.sec.gov/submissions/CIK##########.json  (``parse_submissions``)
    series Atom feed    browse-edgar ... output=atom; a series id (S000000856) in the CIK
                        parameter lists only that fund's filings  (``parse_series_atom``)
    fund tickers        https://www.sec.gov/files/company_tickers_mf.json  (``parse_fund_tickers``)
Accession numbers are always dashed (0001003715-26-002722) in this module's outputs.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from urllib.parse import quote

import pandas as pd

FUND_TICKERS_URL = "https://www.sec.gov/files/company_tickers_mf.json"
_MAX_COUNT = 100  # EDGAR's cap on entries per page of a company listing
_DASHED = re.compile(r"(\d{10})-(\d{2})-(\d{6})", re.ASCII)
_PLAIN = re.compile(r"(\d{10})(\d{2})(\d{6})", re.ASCII)


# ---------------------------------------------------------------- identifiers and URLs
def normalise_accession(s: str) -> str:
    """Dashed accession number from the dashed or the 18-digit form; ValueError otherwise."""
    t = s.strip() if isinstance(s, str) else ""
    m = _DASHED.fullmatch(t) or _PLAIN.fullmatch(t)
    if m is None:
        raise ValueError(f"not an accession number (0001003715-26-002722 or 18 digits): {s!r}")
    return "-".join(m.groups())


def _cik10(cik: int | str) -> str:
    return f"{int(cik):010d}"


def submissions_url(cik: int | str) -> str:
    return f"https://data.sec.gov/submissions/CIK{_cik10(cik)}.json"


def series_filings_url(series_or_cik: str, form: str = "NPORT-P", count: int = 40, start: int = 0) -> str:
    """Atom listing of one form type for a registrant, or, given a series id, for that fund only."""
    if not 1 <= count <= _MAX_COUNT:
        raise ValueError(f"count must be in 1..{_MAX_COUNT} (EDGAR's cap); got {count}")
    if start < 0:
        raise ValueError(f"start must be >= 0; got {start}")
    return ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={quote(str(series_or_cik), safe='')}&type={quote(form, safe='')}"
            f"&dateb=&owner=include&start={start}&count={count}&output=atom")


def _archive_dir(cik: int | str, accession: str) -> tuple[str, str]:
    acc = normalise_accession(accession)
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}", acc


def filing_index_url(cik: int | str, accession: str) -> str:
    base, acc = _archive_dir(cik, accession)
    return f"{base}/{acc}-index.htm"


def filing_document_url(cik: int | str, accession: str, name: str = "primary_doc.xml") -> str:
    base, _ = _archive_dir(cik, accession)
    return f"{base}/{name}"


# ---------------------------------------------------------------- filing records
@dataclass(frozen=True)
class FilingMeta:
    cik: str | None  # 10 digits; None only for an older submissions page parsed without cik=
    accession: str  # dashed
    form: str
    filing_date: date
    report_date: date | None
    primary_document: str | None
    index_url: str | None  # None exactly when cik is None
    is_amendment: bool  # form ends with "/A"


def _meta(cik: str | None, accession: str, form: str, filing_date: date, report_date: date | None,
          primary_document: str | None, index_url: str | None = None) -> FilingMeta:
    acc = normalise_accession(accession)
    if index_url is None and cik is not None:
        index_url = filing_index_url(cik, acc)
    return FilingMeta(cik, acc, form, filing_date, report_date, primary_document, index_url,
                      form.endswith("/A"))


def _date(s: str | None, what: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise ValueError(f"{what}: not a yyyy-mm-dd date: {s!r}") from None


def _required_date(s: str | None, what: str) -> date:
    d = _date(s, what)
    if d is None:
        raise ValueError(f"{what}: missing")
    return d


# ---------------------------------------------------------------- Atom feed of one series
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(el: ET.Element | None, name: str) -> str | None:
    node = None if el is None else next((c for c in el if _local(c.tag) == name), None)
    text = None if node is None else (node.text or "").strip()
    return text or None


def parse_series_atom(xml_bytes: bytes) -> tuple[dict, list[FilingMeta]]:
    """Parse the browse-edgar Atom feed.

    Returns ({"cik", "name", "fiscal_year_end" (MMDD)}, filings in feed order). ValueError if the
    root is not an Atom feed, company-info/cik is missing, or an entry lacks its accession number,
    filing date or filing type. ``index_url`` is the entry's filing-href, or built from the
    accession when that is absent.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        raise ValueError(f"not well-formed XML: {e}") from e
    if _local(root.tag) != "feed":
        raise ValueError(f"not an Atom feed: root element is <{_local(root.tag)}>")
    company = next((c for c in root if _local(c.tag) == "company-info"), None)
    raw_cik = _child_text(company, "cik")
    if raw_cik is None:
        raise ValueError("Atom feed has no company-info/cik")
    cik = _cik10(raw_cik)
    info = {"cik": cik, "name": _child_text(company, "conformed-name"),
            "fiscal_year_end": _child_text(company, "fiscal-year-end")}

    filings = []
    for n, entry in enumerate((e for e in root if _local(e.tag) == "entry"), start=1):
        content = next((c for c in entry if _local(c.tag) == "content"), None)
        acc, form = _child_text(content, "accession-number"), _child_text(content, "filing-type")
        if acc is None or form is None:
            raise ValueError(f"entry {n}: missing accession-number or filing-type")
        filings.append(_meta(cik, acc, form, _required_date(_child_text(content, "filing-date"),
                                                            f"entry {n} filing-date"),
                             None, None, _child_text(content, "filing-href")))
    return info, filings


# ---------------------------------------------------------------- data.sec.gov submissions JSON
def _column(table: dict, key: str, n: int, required: bool) -> list:
    col = table.get(key)
    if col is None:
        if required:
            raise ValueError(f"submissions JSON has no {key!r} array")
        return [None] * n
    if len(col) != n:
        raise ValueError(f"submissions array {key!r} has {len(col)} items, expected {n}")
    return col


def _blank_none(s: str | None) -> str | None:
    return s if s else None


def parse_submissions(json_bytes: bytes, *, cik: int | str | None = None
                      ) -> tuple[dict, list[FilingMeta], list[str]]:
    """Parse a data.sec.gov submissions document.

    Returns (info, filings, urls of the older pages). Accepts both shapes: the full document
    ({"cik", "name", "filings": {"recent": {...}, "files": [...]}}) and an older page, which is
    just the parallel-array object. For an older page ``info`` is {} and the page carries no
    CIK, so pass ``cik=`` to fill ``FilingMeta.cik`` and ``index_url``; without it both are None.
    """
    doc = json.loads(json_bytes)
    if "filings" in doc:
        recent, files = doc["filings"].get("recent"), doc["filings"].get("files") or []
        if "cik" not in doc or recent is None:
            raise ValueError("submissions document lacks 'cik' or 'filings.recent'")
        info = {"cik": _cik10(doc["cik"]), "name": doc.get("name"), "sic": _blank_none(doc.get("sic")),
                "sic_description": _blank_none(doc.get("sicDescription")),
                "fiscal_year_end": _blank_none(doc.get("fiscalYearEnd"))}
        cik10 = info["cik"]
    elif "accessionNumber" in doc:
        recent, files, info = doc, [], {}
        cik10 = None if cik is None else _cik10(cik)
    else:
        raise ValueError("not a submissions document: no 'filings' object and no 'accessionNumber' array")

    accs = recent.get("accessionNumber")
    if accs is None:
        raise ValueError("submissions JSON has no 'accessionNumber' array")
    n = len(accs)
    dates, forms = _column(recent, "filingDate", n, True), _column(recent, "form", n, True)
    reports = _column(recent, "reportDate", n, False)
    docs = _column(recent, "primaryDocument", n, False)
    filings = [_meta(cik10, a, f, _required_date(d, f"filingDate[{i}]"), _date(r, f"reportDate[{i}]"),
                     _blank_none(p))
               for i, (a, f, d, r, p) in enumerate(zip(accs, forms, dates, reports, docs))]
    return info, filings, [f"https://data.sec.gov/submissions/{f['name']}" for f in files]


# ---------------------------------------------------------------- company_tickers_mf.json
def parse_fund_tickers(json_bytes: bytes) -> pd.DataFrame:
    """Mutual-fund ticker table: columns cik (10 digits), series_id, class_id, symbol.

    The column order of the file is read from its "fields" list, not assumed.
    """
    doc = json.loads(json_bytes)
    fields = doc["fields"]
    names = {"cik": "cik", "seriesId": "series_id", "classId": "class_id", "symbol": "symbol"}
    missing = [f for f in names if f not in fields]
    if missing:
        raise ValueError(f"company_tickers_mf fields lack {missing}; has {fields}")
    pos = {out: fields.index(f) for f, out in names.items()}
    rows = []
    for i, r in enumerate(doc["data"]):
        if len(r) != len(fields):
            raise ValueError(f"data row {i} has {len(r)} values, expected {len(fields)}")
        rows.append({out: r[j] for out, j in pos.items()})
    df = pd.DataFrame(rows, columns=list(names.values()), dtype=object)
    df["cik"] = [_cik10(c) for c in df["cik"]]
    return df


# ---------------------------------------------------------------- choosing among filings
def select_latest(filings: list[FilingMeta]) -> list[FilingMeta]:
    """One filing per period: the most recently filed, an amendment winning over its original.

    A period is the report date when known, else the filing date; filings of different CIKs or
    base forms (NPORT-P and NPORT-P/A are one base form) are never merged. Ties on filing date go
    to the amendment, then to the later accession number. Sorted by period, ascending.
    """
    best: dict[tuple, FilingMeta] = {}
    for f in filings:
        period = f.report_date or f.filing_date
        key = (f.cik, f.form.removesuffix("/A"), f.report_date is not None, period)
        cur = best.get(key)
        if cur is None or _rank(f) > _rank(cur):
            best[key] = f
    return sorted(best.values(), key=lambda f: (f.report_date or f.filing_date, f.filing_date, f.accession))


def _rank(f: FilingMeta) -> tuple:
    return (f.filing_date, f.is_amendment, f.accession)
