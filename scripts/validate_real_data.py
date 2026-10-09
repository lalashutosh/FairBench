"""Validate the canonical FairBench real-data export.

This is a quarantine/quality-control step, not a data-cleaning step that invents
values.  SEC-derived rows that fail a check are reported; they are not silently
deleted or filled.  External exchange-close comparisons are explicitly advisory:
corporate actions, share classes, delistings, and ticker reuse can create large
differences even when the SEC filing is genuine.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "real_data"


def _read(name: str) -> pd.DataFrame:
    path = OUT / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _check(name: str, passed: bool, detail: str, severity: str = "error") -> dict:
    return {"name": name, "passed": bool(passed), "severity": severity, "detail": detail}


def main() -> None:
    required = [
        "real_funds.csv",
        "real_quarterly_holdings.csv",
        "real_quarterly_implied_prices.csv",
        "real_snp500_proxy_quarterly_prices.csv",
        "real_snp500_proxy_quarterly_returns.csv",
        "real_snp500_proxy_return_quality.csv",
        "real_observed_holding_changes.csv",
        "real_data_quality_report.csv",
        "external_market_quarterly_closes.csv",
        "real_snp500_price_crosscheck.csv",
    ]
    missing = [name for name in required if not (OUT / name).exists()]
    if missing:
        raise SystemExit(f"missing exported files: {', '.join(missing)}")

    funds = _read("real_funds.csv")
    holdings = _read("real_quarterly_holdings.csv")
    prices = _read("real_quarterly_implied_prices.csv")
    parent_prices = _read("real_snp500_proxy_quarterly_prices.csv")
    parent_returns = _read("real_snp500_proxy_quarterly_returns.csv")
    return_quality = _read("real_snp500_proxy_return_quality.csv")
    changes = _read("real_observed_holding_changes.csv")
    crosscheck = _read("real_snp500_price_crosscheck.csv")

    checks: list[dict] = []
    source_columns = ["source_url", "from_source_url", "to_source_url"]
    source_values = []
    for frame in (holdings, prices, parent_prices, parent_returns, changes):
        for column in source_columns:
            if column in frame.columns:
                source_values.extend(frame[column].dropna().astype(str).tolist())
    sec_url_pattern = re.compile(r"^https://www\.sec\.gov/Archives/")
    bad_sources = [url for url in source_values if not sec_url_pattern.match(url)]
    checks.append(_check(
        "sec_source_provenance",
        not bad_sources,
        f"{len(source_values):,} SEC source references checked; {len(bad_sources):,} invalid",
    ))

    all_text = "\n".join(
        str(value)
        for frame in (funds, holdings, prices, parent_prices, parent_returns, changes)
        for column in frame.columns
        for value in frame[column].dropna().head(100000).tolist()
    ).lower()
    synthetic_terms = ["synthetic example", "synthetic data", "fake data"]
    found_terms = [term for term in synthetic_terms if term in all_text]
    checks.append(_check(
        "no_synthetic_rows_in_export",
        not found_terms,
        f"synthetic markers found: {found_terms or 'none'}",
    ))

    positive_prices = prices["price_usd"].notna() & (prices["price_usd"] > 0)
    checks.append(_check(
        "derived_prices_positive",
        bool(positive_prices.all()),
        f"{int(positive_prices.sum()):,}/{len(prices):,} positive derived prices",
    ))

    price_key = ["fund_id", "report_date", "security_key"]
    duplicate_prices = int(prices.duplicated(price_key).sum())
    checks.append(_check(
        "no_duplicate_price_keys",
        duplicate_prices == 0,
        f"{duplicate_prices:,} duplicate (fund, report date, security) rows",
    ))

    expected_return = parent_returns["price_end_usd"] / parent_returns["price_start_usd"] - 1.0
    return_error = (expected_return - parent_returns["price_return"]).abs()
    return_ok = return_error.fillna(0).lt(1e-10)
    checks.append(_check(
        "returns_reproduce_from_prices",
        bool(return_ok.all()),
        f"{int(return_ok.sum()):,}/{len(return_ok):,} returns reproduce from start/end prices",
    ))

    return_sources = parent_returns[["from_source_url", "to_source_url"]].notna().all(axis=1)
    checks.append(_check(
        "returns_have_both_filing_sources",
        bool(return_sources.all()),
        f"{int(return_sources.sum()):,}/{len(return_sources):,} returns have both filing URLs",
    ))

    no_zero_fill = not ((parent_returns["price_start_usd"] == 0) | (parent_returns["price_end_usd"] == 0)).any()
    checks.append(_check(
        "no_zero_price_fill",
        no_zero_fill,
        "zero prices are not used as missing-value placeholders",
    ))

    date_values = pd.to_datetime(parent_prices["report_date"], errors="coerce")
    checks.append(_check(
        "valid_report_dates",
        bool(date_values.notna().all()),
        f"{int(date_values.notna().sum()):,}/{len(date_values):,} report dates parse",
    ))

    unique_keys = parent_prices["security_key"].nunique(dropna=True)
    unique_tickers = parent_prices["ticker"].dropna().nunique()
    per_snapshot = parent_prices.groupby("report_date")["security_key"].nunique()
    checks.append(_check(
        "parent_panel_has_index_scale",
        bool(per_snapshot.between(450, 550).all()),
        f"{len(per_snapshot):,} snapshots; {int(per_snapshot.min()):,}-{int(per_snapshot.max()):,} securities per snapshot; "
        f"{unique_keys:,} historical security keys / {unique_tickers:,} ticker symbols",
    ))

    # The external price series is a review aid only.  Keep every row and mark
    # mismatches for inspection instead of deleting genuine SEC observations.
    crosscheck["review_label"] = "external_missing"
    available = crosscheck["market_close_usd"].notna()
    abs_diff = crosscheck["abs_relative_price_difference"]
    crosscheck.loc[available & abs_diff.le(0.01), "review_label"] = "ok_within_1pct"
    crosscheck.loc[available & abs_diff.gt(0.01) & abs_diff.le(0.10), "review_label"] = "review_1_to_10pct"
    crosscheck.loc[available & abs_diff.gt(0.10), "review_label"] = "review_outlier_gt_10pct"
    review = crosscheck.sort_values(["review_label", "report_date", "security_key"], kind="mergesort")
    review.to_csv(OUT / "real_snp500_price_crosscheck_review.csv", index=False, na_rep="")
    available_count = int(available.sum())
    outlier_count = int((available & abs_diff.gt(0.10)).sum())
    within_1pct = int((available & abs_diff.le(0.01)).sum())
    checks.append(_check(
        "external_crosscheck_is_advisory",
        True,
        f"{available_count:,} external matches; {within_1pct:,} within 1%; {outlier_count:,} >10% flagged for review",
        severity="warning",
    ))

    passed = all(item["passed"] for item in checks if item["severity"] == "error")
    summary = {
        "status": "PASS" if passed else "FAIL",
        "canonical_input": "results/real_data",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "counts": {
            "funds": len(funds),
            "holdings": len(holdings),
            "derived_prices": len(prices),
            "snp500_proxy_price_rows": len(parent_prices),
            "snp500_proxy_return_rows": len(parent_returns),
            "observed_change_rows": len(changes),
            "historical_security_keys": int(unique_keys),
            "ticker_symbols": int(unique_tickers),
            "external_crosscheck_rows": len(crosscheck),
            "external_crosscheck_outliers_gt_10pct": outlier_count,
        },
        "checks": checks,
        "interpretation": [
            "SEC holdings, market values, shares, filings, returns, and flows are disclosed observations.",
            "Quarter prices and returns are derived from SEC market_value_usd / shares and exclude dividends.",
            "Snapshot changes are observed differences, not direct manager decisions or trade intent.",
            "External close-price mismatches are retained and flagged; they are never used to overwrite SEC rows.",
        ],
    }
    (OUT / "real_data_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
