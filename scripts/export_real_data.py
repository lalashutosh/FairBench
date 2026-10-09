"""Export a clean, real-data-only panel from the SEC N-PORT archive.

This script never reads the synthetic example.  It reads only data/fairbench.sqlite,
which is populated by scripts/real_fund_ingest.py, and writes sorted CSVs under
results/real_data/.

Important labels:
  * holdings and fund returns are disclosed by SEC filings;
  * quarter-end prices and returns are derived as N-PORT market_value / shares;
  * holding changes are derived from two disclosed snapshots and are not claimed to be
    direct manager decisions;
  * missing values stay missing.  The exporter never fills a price or return.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fairbench.portfolio.changes import holding_changes


DB = ROOT / "data" / "fairbench.sqlite"
OUT = ROOT / "results" / "real_data"
PARENT_ID = "S000004310"  # iShares Core S&P 500 ETF; a filing-based S&P 500 proxy.


def _write(df: pd.DataFrame, name: str, sort_by: list[str]) -> None:
    out = df.sort_values(sort_by, kind="mergesort", na_position="last").reset_index(drop=True)
    out.to_csv(OUT / name, index=False, na_rep="")
    print(f"{name}: {len(out):,} rows")


def _sql(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, conn)


def _holdings(conn: sqlite3.Connection) -> pd.DataFrame:
    return _sql(conn, """
        SELECT hs.fund_id, f.name AS fund_name, hs.report_date,
               fl.filing_date, fl.accession, d.url AS source_url,
               h.holding_id, h.row_index, h.security_id,
               s.security_key, s.title AS security_title,
               h.issuer_name, i.cik AS issuer_cik, i.sic_code,
               h.shares, h.units, h.market_value AS market_value_usd,
               h.pct_net_assets, h.asset_category, h.payoff_profile,
               h.issuer_category, h.country, h.currency,
               MAX(CASE WHEN si.scheme = 'cusip' THEN si.value END) AS cusip,
               MAX(CASE WHEN si.scheme = 'isin' THEN si.value END) AS isin,
               MAX(CASE WHEN si.scheme = 'ticker' THEN si.value END) AS ticker,
               h.status AS holding_status, h.confidence
        FROM holdings h
        JOIN holding_snapshots hs ON hs.snapshot_id = h.snapshot_id
        JOIN funds f ON f.fund_id = hs.fund_id
        LEFT JOIN filings fl ON fl.filing_id = hs.filing_id
        LEFT JOIN documents d ON d.document_id = h.document_id
        LEFT JOIN securities s ON s.security_id = h.security_id
        LEFT JOIN issuers i ON i.issuer_id = s.issuer_id
        LEFT JOIN security_identifiers si ON si.security_id = h.security_id
        WHERE h.asset_category = 'EC'
          AND h.payoff_profile = 'Long'
          AND h.units = 'NS'
          AND h.shares > 0
          AND h.market_value > 0
        GROUP BY h.holding_id
    """)


def _funds(conn: sqlite3.Connection) -> pd.DataFrame:
    return _sql(conn, """
        SELECT f.*, CASE WHEN f.fund_id IN ('S000004310','S000004347','S000028709')
                         THEN 1 ELSE 0 END AS benchmark_proxy
        FROM funds f
        ORDER BY f.fund_id
    """)


def _source_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    return _sql(conn, """
        SELECT hs.fund_id, f.name AS fund_name,
               COUNT(DISTINCT hs.snapshot_id) AS snapshots,
               MIN(hs.report_date) AS first_report_date,
               MAX(hs.report_date) AS last_report_date,
               COUNT(DISTINCT h.holding_id) AS equity_holding_rows,
               COUNT(DISTINCT h.security_id) AS distinct_securities,
               COUNT(DISTINCT d.document_id) AS source_documents,
               SUM(CASE WHEN h.security_id IS NULL THEN 1 ELSE 0 END) AS unresolved_security_rows,
               SUM(CASE WHEN h.security_id IS NOT NULL AND si_ticker.value IS NULL THEN 1 ELSE 0 END)
                   AS rows_without_ticker,
               SUM(CASE WHEN h.shares > 0 AND h.market_value > 0 THEN 1 ELSE 0 END) AS priced_rows
        FROM holding_snapshots hs
        JOIN funds f ON f.fund_id = hs.fund_id
        LEFT JOIN holdings h ON h.snapshot_id = hs.snapshot_id AND h.asset_category = 'EC'
             AND h.payoff_profile = 'Long' AND h.units = 'NS' AND h.shares > 0 AND h.market_value > 0
        LEFT JOIN documents d ON d.document_id = h.document_id
        LEFT JOIN security_identifiers si_ticker ON si_ticker.security_id = h.security_id
             AND si_ticker.scheme = 'ticker'
        GROUP BY hs.fund_id, f.name
        ORDER BY hs.fund_id
    """)


def _returns(conn: sqlite3.Connection) -> pd.DataFrame:
    return _sql(conn, """
        SELECT fr.fund_id, f.name AS fund_name, fr.class_id, fr.period_end,
               fr.nav_return, fr.status, fr.confidence,
               d.url AS source_url, fl.accession, fl.filing_date, fl.report_date
        FROM fund_returns fr
        JOIN funds f ON f.fund_id = fr.fund_id
        LEFT JOIN documents d ON d.document_id = fr.document_id
        LEFT JOIN filings fl ON fl.filing_id = d.filing_id
        ORDER BY fr.fund_id, fr.class_id, fr.period_end
    """)


def _flows(conn: sqlite3.Connection) -> pd.DataFrame:
    return _sql(conn, """
        SELECT ff.fund_id, f.name AS fund_name, ff.month_end,
               ff.sales, ff.reinvestments, ff.redemptions,
               ff.status, ff.confidence,
               d.url AS source_url, fl.accession, fl.filing_date, fl.report_date
        FROM fund_flows ff
        JOIN funds f ON f.fund_id = ff.fund_id
        LEFT JOIN documents d ON d.document_id = ff.document_id
        LEFT JOIN filings fl ON fl.filing_id = d.filing_id
        ORDER BY ff.fund_id, ff.month_end
    """)


def _price_panel(hold: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate N-PORT rows to one security per fund/date and derive price."""
    h = hold.copy()
    group = ["fund_id", "fund_name", "report_date", "security_id", "security_key",
             "issuer_name", "issuer_cik", "sic_code", "cusip", "isin", "ticker",
             "currency"]
    p = (h.groupby(group, dropna=False, as_index=False)
           .agg(shares=("shares", "sum"), market_value_usd=("market_value_usd", "sum"),
                pct_net_assets=("pct_net_assets", "sum"),
                source_url=("source_url", "first"), accession=("accession", "first"),
                filing_date=("filing_date", "first")))
    p["price_usd"] = p["market_value_usd"] / p["shares"]
    p["price_status"] = "derived"
    p["price_method"] = "SEC N-PORT market_value_usd / shares"
    p["dividends_included"] = False
    p["source_kind"] = "SEC N-PORT filing"
    return p


def _returns_from_prices(prices: pd.DataFrame, fund_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = prices[prices["fund_id"] == fund_id].copy()
    p["report_date"] = pd.to_datetime(p["report_date"])
    dates = sorted(p["report_date"].dropna().unique())
    rows, quality = [], []
    for start, end in zip(dates, dates[1:]):
        a = p[p["report_date"] == start].set_index("security_key")
        b = p[p["report_date"] == end].set_index("security_key")
        keys = a.index.union(b.index)
        both = a.index.intersection(b.index)
        paired = b.loc[b.index.intersection(a.index), "price_usd"].notna() & a.loc[b.index.intersection(a.index), "price_usd"].notna()
        for key in both[paired.to_numpy()]:
            x, y = float(a.loc[key, "price_usd"]), float(b.loc[key, "price_usd"])
            if x > 0 and y > 0:
                row = b.loc[key]
                rows.append({
                    "fund_id": fund_id, "from_date": start.date().isoformat(), "to_date": end.date().isoformat(),
                    "security_key": key, "security_id": row["security_id"], "issuer_name": row["issuer_name"],
                    "ticker": row["ticker"], "price_start_usd": x, "price_end_usd": y,
                    "price_return": y / x - 1.0, "status": "derived",
                    "source_kind": "SEC N-PORT market_value_usd / shares",
                    "from_source_url": a.loc[key, "source_url"], "to_source_url": row["source_url"],
                })
        quality.append({
            "fund_id": fund_id, "from_date": start.date().isoformat(), "to_date": end.date().isoformat(),
            "start_securities": len(a), "end_securities": len(b), "union_securities": len(keys),
            "paired_price_rows": int(paired.sum()), "start_only": len(a.index.difference(b.index)),
            "end_only": len(b.index.difference(a.index)),
            "missing_price_in_pair": int(len(both) - paired.sum()),
        })
    return pd.DataFrame(rows), pd.DataFrame(quality)


def _observed_changes(hold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fund_id, group in hold.groupby("fund_id", sort=True):
        snapshots = []
        for d, g in group.groupby("report_date", sort=True):
            x = g[["security_key", "issuer_name", "shares", "market_value_usd"]].copy()
            x = x.rename(columns={"shares": "balance", "market_value_usd": "value_usd"})
            snapshots.append((d, g["source_url"].dropna().iloc[0] if g["source_url"].notna().any() else None, x))
        for (d0, url0, prev), (d1, url1, curr) in zip(snapshots, snapshots[1:]):
            ch = holding_changes(prev, curr, key="security_key")
            ch.insert(0, "fund_id", fund_id)
            ch.insert(1, "from_date", d0)
            ch.insert(2, "to_date", d1)
            ch["from_source_url"] = url0
            ch["to_source_url"] = url1
            ch["status"] = "derived"
            ch["interpretation"] = "difference between two disclosed snapshots; not a direct manager decision"
            rows.append(ch)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    if not DB.exists():
        raise SystemExit(f"missing {DB}; run scripts/real_fund_ingest.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as conn:
        funds = _funds(conn)
        hold = _holdings(conn)
        prices = _price_panel(hold)
        fund_returns = _returns(conn)
        fund_flows = _flows(conn)
        quality = _source_summary(conn)

    _write(funds, "real_funds.csv", ["fund_id"])
    _write(hold, "real_quarterly_holdings.csv", ["fund_id", "report_date", "security_key", "holding_id"])
    _write(prices, "real_quarterly_implied_prices.csv", ["fund_id", "report_date", "security_key"])
    _write(prices[prices["fund_id"] == PARENT_ID], "real_snp500_proxy_quarterly_prices.csv",
           ["report_date", "security_key"])

    parent_returns, parent_quality = _returns_from_prices(prices, PARENT_ID)
    _write(parent_returns, "real_snp500_proxy_quarterly_returns.csv", ["from_date", "security_key"])
    _write(parent_quality, "real_snp500_proxy_return_quality.csv", ["from_date"])

    # Wide form is convenient for the quantum input adapter; blank cells remain missing.
    wide = (prices[prices["fund_id"] == PARENT_ID]
            .pivot_table(index="report_date", columns="security_key", values="price_usd", aggfunc="first")
            .sort_index())
    wide.index.name = "report_date"
    wide.to_csv(OUT / "real_snp500_proxy_quarterly_prices_wide.csv", na_rep="")
    print(f"real_snp500_proxy_quarterly_prices_wide.csv: {wide.shape[0]:,} rows x {wide.shape[1]:,} securities")

    _write(fund_returns, "real_fund_nav_returns.csv", ["fund_id", "class_id", "period_end"])
    _write(fund_flows, "real_fund_flows.csv", ["fund_id", "month_end"])
    _write(quality, "real_data_quality_report.csv", ["fund_id"])

    changes = _observed_changes(hold)
    if not changes.empty:
        _write(changes, "real_observed_holding_changes.csv", ["fund_id", "from_date", "security_key"])

    first, last = prices[prices["fund_id"] == PARENT_ID]["report_date"].min(), prices[prices["fund_id"] == PARENT_ID]["report_date"].max()
    manifest = f"""# FairBench real-data export

Generated from `data/fairbench.sqlite` on the current run. This directory contains no synthetic example data.

## Coverage

- SEC series ingested: {len(funds)}
- S&P 500 proxy: iShares Core S&P 500 ETF (`{PARENT_ID}`)
- S&P 500 proxy quarter-end price coverage: `{first}` through `{last}`
- The latest public N-PORT snapshot is not necessarily the latest calendar quarter; N-PORT reports are disclosed with a lag.
- `real_snp500_proxy_quarterly_prices.csv` is a long panel; the `_wide.csv` file is its model-friendly pivot.

## Truth labels

- `real_quarterly_holdings.csv`, `real_fund_nav_returns.csv`, and `real_fund_flows.csv`: `disclosed` SEC N-PORT facts.
- `real_quarterly_implied_prices.csv` and the S&P 500 price/return files: `derived` as `market_value_usd / shares`; dividends are not included.
- `real_observed_holding_changes.csv`: `derived` from adjacent disclosed snapshots; it does not claim to reveal intent, trades inside the quarter, or manager rationale.
- Missing prices and returns are left blank. No zeros or synthetic values are inserted.

## Files

| File | Use |
|---|---|
| `real_funds.csv` | Fund and benchmark-proxy registry |
| `real_quarterly_holdings.csv` | Clean long holdings panel |
| `real_quarterly_implied_prices.csv` | Quarter-end prices for all scraped funds |
| `real_snp500_proxy_quarterly_prices.csv` | S&P 500 proxy long price panel |
| `real_snp500_proxy_quarterly_prices_wide.csv` | Same panel pivoted for modelling |
| `real_snp500_proxy_quarterly_returns.csv` | Pairwise quarter returns, missing names omitted rather than filled |
| `real_snp500_proxy_return_quality.csv` | Start/end/missing/left/joined coverage by quarter |
| `real_observed_holding_changes.csv` | Snapshot-derived position changes |
| `real_data_quality_report.csv` | Counts and identifier coverage per fund |
| `external_market_quarterly_closes.csv` | Independent Yahoo close-price cross-check; not model input |
| `real_snp500_price_crosscheck_review.csv` | Cross-check rows labelled for review; no SEC rows overwritten |
| `real_data_validation.json` | Latest provenance and integrity validation |
"""
    (OUT / "README.md").write_text(manifest, encoding="utf-8")
    print(f"README.md: {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
