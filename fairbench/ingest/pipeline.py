"""Ingestion pipeline: SEC listing -> raw archive -> parsed N-PORT -> database with provenance.

    fetch_series_nport   live: list one fund's NPORT-P filings on EDGAR, download each
                         primary_doc.xml through the polite client (which archives the raw
                         bytes first), then store it
    ingest_nport_document  offline core: parse one N-PORT document and store the fund, the
                         filing, the document, the holdings snapshot, the monthly returns
                         and the flows, each row pointing at the document it came from

Re-running is idempotent: a filing that is already stored is skipped, and filings on
EDGAR never change, so the client serves them from the archive without a new request.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone

from ..store import db
from .edgar import FilingMeta, filing_document_url, parse_series_atom, select_latest, series_filings_url
from .identifiers import add_security_keys
from .nport import NportReport, parse_nport

TOOL_VERSION = "fairbench.ingest.nport/1"
_STORE_COLUMNS = ["row_index", "issuer_name", "issuer_lei", "title", "cusip", "isin", "ticker", "balance", "units",
                  "currency", "value_usd", "pct_net_assets", "payoff_profile", "asset_category", "issuer_category",
                  "country", "is_restricted", "fair_value_level", "is_loaned", "security_key"]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ingest_nport_document(conn, raw: bytes, *, url: str, retrieved_at: str | None = None,
                          accession: str | None = None, filing_date: date | str | None = None,
                          http_status: int | None = None, etag: str | None = None, archive_path: str | None = None,
                          run_id: int | None = None, source_id: int | None = None) -> tuple[NportReport, int | None]:
    """Parse and store one N-PORT document. Returns (report, snapshot_id); snapshot_id is
    None when this filing's snapshot was already stored. ``accession`` defaults to a
    stand-in derived from the content hash, for documents that did not come from EDGAR
    (the offline example); real ingestion always passes the EDGAR accession number."""
    report = parse_nport(raw)
    sha = hashlib.sha256(raw).hexdigest()
    if report.series_id is None:
        raise ValueError("the document has no series id; cannot tell which fund it belongs to")
    fye = report.fiscal_year_end.strftime("%m%d") if report.fiscal_year_end else None
    db.add_fund(conn, fund_id=report.series_id, name=report.series_name, registrant_cik=report.registrant_cik,
                registrant_name=report.registrant_name, fiscal_year_end=fye, asset_class="equity", currency="USD")
    for cid in report.class_ids:
        db.add_share_class(conn, class_id=cid, fund_id=report.series_id)
    filing_id = db.add_filing(
        conn, accession=accession or f"local-{sha[:20]}", cik=report.registrant_cik or "", form_type=report.submission_type or "NPORT-P",
        filing_date=None if filing_date is None else str(filing_date), report_date=str(report.report_date),
        is_amendment=int((report.submission_type or "").endswith("/A")))
    document_id = db.add_document(conn, filing_id=filing_id, url=url, sha256=sha, retrieved_at=retrieved_at or _now(),
                                  http_status=http_status, etag=etag, media_type="application/xml",
                                  archive_path=archive_path, source_id=source_id)
    if conn.execute("SELECT 1 FROM holding_snapshots WHERE fund_id = ? AND report_date = ? AND filing_id = ?",
                    (report.series_id, str(report.report_date), filing_id)).fetchone():
        return report, None
    if run_id is None:
        run_id = db.start_run(conn, method="xml_parse", tool_version=TOOL_VERSION)
    keyed = add_security_keys(report.holdings)
    snapshot_id = db.add_snapshot(
        conn, fund_id=report.series_id, report_date=str(report.report_date), filing_id=filing_id,
        document_id=document_id, run_id=run_id, net_assets=report.net_assets, total_assets=report.total_assets,
        total_liabilities=report.total_liabilities, holdings=keyed[_STORE_COLUMNS])
    db.add_fund_returns(conn, report.series_id, report.monthly_returns[["class_id", "month_end", "return_pct"]],
                        document_id, run_id)
    db.add_fund_flows(conn, report.series_id, report.flows[["month_end", "sales", "reinvestment", "redemption"]],
                      document_id, run_id)
    conn.commit()
    return report, snapshot_id


def list_series_filings(client, series_id: str, form: str = "NPORT-P", page: int = 40,
                        max_pages: int = 10) -> list[FilingMeta]:
    """Every filing of one form type for one fund (EDGAR series id), oldest first, one
    document per report period (an amendment replaces the filing it corrects)."""
    found: list[FilingMeta] = []
    for p in range(max_pages):
        _, filings = parse_series_atom(client.read(series_filings_url(series_id, form, count=page, start=p * page)))
        found += [f for f in filings if f.form.startswith(form)]
        if len(filings) < page:
            break
    return select_latest(found)


def fetch_series_nport(client, conn, series_id: str, *, since: date | None = None,
                       source_id: int | None = None) -> list[tuple[NportReport, int | None]]:
    """Download and store every public N-PORT report of one fund. ``client`` is an
    ``ingest.http.SecClient``; it archives each raw response before this function parses
    it. ``since``: skip filings made before this date."""
    out = []
    for f in list_series_filings(client, series_id):
        if since is not None and f.filing_date < since:
            continue
        url = filing_document_url(f.cik, f.accession)
        rec = client.get(url)
        raw = client.archive.read(rec)
        out.append(ingest_nport_document(
            conn, raw, url=url, retrieved_at=rec.retrieved_at, accession=f.accession, filing_date=f.filing_date,
            http_status=rec.status, etag=rec.etag, archive_path=str(rec.path), source_id=source_id))
    return out
