"""Download a fund's public N-PORT filings from SEC EDGAR and store them with provenance.

    export FAIRBENCH_SEC_USER_AGENT="Your Project Name contact@your-domain.org"
    .venv/bin/python scripts/real_fund_ingest.py --series S000000856 --series S000004310

LIVE: this is the only script that talks to the SEC. It needs the environment variable
above (SEC asks automated clients to declare who they are; there is no default and the
client refuses to run without it). It makes at most two requests a second, keeps every raw
response under data/raw before parsing it, never re-downloads a filing it already has, and
only ever contacts www.sec.gov and data.sec.gov.

--series takes EDGAR series ids. S000000856 is Parnassus Core Equity and S000004310 is
iShares Core S&P 500 (the parent-universe proxy); docs/RESEARCH.md (Part B)
lists the others. Output: data/fairbench.sqlite, data/raw/ and one XML per report date
under data/nport/<series>/ for scripts/real_fund_attribution.py. data/ is git-ignored.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from any folder, installed or not

import argparse
from datetime import date

from fairbench.ingest.archive import RawArchive
from fairbench.ingest.http import SecClient
from fairbench.ingest.pipeline import fetch_series_nport
from fairbench.store import db


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", action="append", required=True, help="EDGAR series id (repeatable)")
    ap.add_argument("--data", default="data", help="output directory (default: data)")
    ap.add_argument("--since", default=None, help="skip filings made before this date (YYYY-MM-DD)")
    ap.add_argument("--max-rps", type=float, default=2.0)
    args = ap.parse_args()

    root = Path(args.data)
    root.mkdir(parents=True, exist_ok=True)
    client = SecClient(RawArchive(root / "raw"), max_rps=args.max_rps)
    conn = db.open_db(root / "fairbench.sqlite")
    source = db.add_source(conn, name="SEC EDGAR", licence="public filings; SEC disclaims accuracy",
                           may_redistribute=1, terms_url="https://www.sec.gov/os/accessing-edgar-data",
                           terms_checked_on=str(date.today()))
    since = date.fromisoformat(args.since) if args.since else None
    for series in args.series:
        done = fetch_series_nport(client, conn, series, since=since, source_id=source)
        out = root / "nport" / series
        out.mkdir(parents=True, exist_ok=True)
        for report, _ in done:
            path = conn.execute(
                "SELECT d.archive_path FROM documents d JOIN filings f ON f.filing_id = d.filing_id "
                "JOIN holding_snapshots s ON s.filing_id = f.filing_id WHERE s.fund_id = ? AND s.report_date = ? "
                "ORDER BY f.filing_date DESC", (series, str(report.report_date))).fetchone()[0]
            (out / f"{report.report_date}.xml").write_bytes(Path(path).read_bytes())
        name = done[0][0].series_name if done else "?"
        dates = sorted(str(r.report_date) for r, _ in done)
        print(f"{series}  {name}: {len(done)} reports" + (f", {dates[0]} to {dates[-1]}" if dates else "")
              + f", {sum(s is not None for _, s in done)} new")
    bad = {t: n for t, n in db.check_provenance(conn).items() if n}
    print(f"requests made: {client.requests_made}; rows without provenance: {bad or 'none'}")


if __name__ == "__main__":
    main()
