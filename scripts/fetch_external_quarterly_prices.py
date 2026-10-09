"""Fetch an external historical close-price cross-check for the real SEC panel.

The SEC N-PORT panel is the authoritative source for the fund's reported holdings and
market values.  This adapter adds an independently downloaded exchange-close cross-check
for symbols that have a Yahoo Finance chart series.  It does not overwrite the SEC data,
does not fill missing names, and keeps raw responses under data/market_prices/.

Yahoo chart data is for local research only; check Yahoo's terms before redistributing it.
The model-facing SEC-derived files remain unchanged.
"""
from __future__ import annotations

import json
import os
import re
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "real_data" / "real_snp500_proxy_quarterly_prices.csv"
RAW = ROOT / "data" / "market_prices" / "yahoo_chart"
OUT = ROOT / "results" / "real_data"
PER_SECOND = 2.0


def yahoo_symbol(ticker: str) -> str:
    """Yahoo's common spelling for share-class punctuation."""
    return re.sub(r"[^A-Za-z0-9^_-]", "-", str(ticker).strip()).replace(".", "-").upper()


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats().get("x509_ca"):
        for bundle in ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt",
                       "/etc/pki/tls/certs/ca-bundle.crt"):
            if os.path.exists(bundle):
                ctx.load_verify_locations(cafile=bundle)
                break
    return ctx


def fetch(symbol: str, start: int, end: int, raw_path: Path) -> dict:
    if raw_path.exists():
        return json.loads(raw_path.read_text(encoding="utf-8"))
    query = urlencode({"period1": start, "period2": end, "interval": "1d",
                       "events": "history", "includeAdjustedClose": "true"})
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
    req = Request(url, headers={"User-Agent": "FairBench research"})
    with urlopen(req, timeout=30, context=_ssl_context()) as response:
        payload = response.read()
    raw_path.write_bytes(payload)
    time.sleep(1.0 / PER_SECOND)
    return json.loads(payload)


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"missing {INPUT}; run scripts/export_real_data.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(INPUT, dtype={"ticker": "string", "security_key": "string"})
    panel = panel[panel["ticker"].notna() & panel["ticker"].ne("")].copy()
    dates = sorted(pd.to_datetime(panel["report_date"]).dt.date.unique())
    start = int(pd.Timestamp(min(dates)).timestamp()) - 14 * 86400
    end = int(pd.Timestamp(max(dates)).timestamp()) + 14 * 86400
    tickers = sorted(panel["ticker"].dropna().astype(str).str.upper().unique())
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    errors = []
    for n, ticker in enumerate(tickers, start=1):
        symbol = yahoo_symbol(ticker)
        raw_path = RAW / f"{symbol}.json"
        try:
            doc = fetch(symbol, start, end, raw_path)
            chart = (doc.get("chart") or {})
            result = (chart.get("result") or [None])[0]
            if not result:
                raise ValueError((chart.get("error") or {}).get("description", "no chart result"))
            ts = result.get("timestamp") or []
            quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
            closes = quote.get("close") or []
            adj = (((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or [])
            history = pd.DataFrame({"market_date": pd.to_datetime(ts, unit="s", utc=True).date,
                                    "market_close_usd": closes,
                                    "market_adj_close_usd": adj if len(adj) == len(ts) else [None] * len(ts)})
            history = history.dropna(subset=["market_close_usd"])
            for report_date in dates:
                eligible = history[history["market_date"] <= report_date]
                if eligible.empty:
                    continue
                x = eligible.iloc[-1]
                rows.append({"ticker": ticker, "yahoo_symbol": symbol, "report_date": report_date.isoformat(),
                             "market_date": x["market_date"].isoformat(),
                             "market_close_usd": float(x["market_close_usd"]),
                             "market_adj_close_usd": (None if pd.isna(x["market_adj_close_usd"])
                                                       else float(x["market_adj_close_usd"])),
                             "days_before_report": (report_date - x["market_date"]).days,
                             "source_url": f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
                             "retrieved_at": retrieved_at, "status": "disclosed_external"})
        except Exception as exc:  # preserve the missing series; do not make up a value
            errors.append({"ticker": ticker, "yahoo_symbol": symbol, "error": str(exc),
                           "retrieved_at": retrieved_at})
        if n % 25 == 0 or n == len(tickers):
            print(f"{n}/{len(tickers)} symbols; rows={len(rows)}; errors={len(errors)}", flush=True)

    market = pd.DataFrame(rows)
    if market.empty:
        pd.DataFrame(errors).sort_values(["ticker"]).to_csv(OUT / "external_market_errors.csv", index=False)
        print(pd.DataFrame(errors).head(5).to_string(index=False), flush=True)
        raise SystemExit("no external market rows were returned")
    market = market.sort_values(["report_date", "ticker"], kind="mergesort")
    market.to_csv(OUT / "external_market_quarterly_closes.csv", index=False, na_rep="")
    pd.DataFrame(errors).sort_values(["ticker"]).to_csv(OUT / "external_market_errors.csv", index=False)

    # Join only by ticker/date for the visible cross-check. Historical identifier changes remain
    # in the SEC panel and are never silently joined to another company.
    sec = panel[["report_date", "ticker", "security_key", "issuer_name", "price_usd"]].copy()
    sec["ticker"] = sec["ticker"].astype(str).str.upper()
    merged = sec.merge(market, on=["report_date", "ticker"], how="left")
    merged["relative_price_difference"] = merged["market_close_usd"] / merged["price_usd"] - 1.0
    merged["abs_relative_price_difference"] = merged["relative_price_difference"].abs()
    merged = merged.sort_values(["report_date", "security_key"], kind="mergesort")
    merged.to_csv(OUT / "real_snp500_price_crosscheck.csv", index=False, na_rep="")

    quality = (merged.groupby("report_date", as_index=False)
               .agg(sec_rows=("security_key", "size"), external_close_rows=("market_close_usd", "count"),
                    median_abs_relative_difference=("abs_relative_price_difference", "median"),
                    p95_abs_relative_difference=("abs_relative_price_difference", lambda s: s.quantile(.95)),
                    max_days_before_report=("days_before_report", "max")))
    quality.to_csv(OUT / "real_snp500_price_crosscheck_quality.csv", index=False, na_rep="")
    print(f"wrote {len(market):,} external rows, {len(errors):,} ticker errors, {len(merged):,} cross-check rows")


if __name__ == "__main__":
    main()
