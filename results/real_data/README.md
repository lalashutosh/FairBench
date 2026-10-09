# FairBench real-data export

Generated from `data/fairbench.sqlite` on the current run. This directory contains no synthetic example data.

## Coverage

- SEC series ingested: 11
- S&P 500 proxy: iShares Core S&P 500 ETF (`S000004310`)
- S&P 500 proxy quarter-end price coverage: `2019-09-30` through `2026-06-30`
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
