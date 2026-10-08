# FairBench on real US funds: research memo and implementation plan

_Written 2026-10-08. Status: **proposal, awaiting approval. No code has been changed.**_
_Inputs read: `README.md`, `PITCH_PLAN.md`, `fairbench_handoff.md`, `FAIRBENCH_INDUSTRY_STANDARDS_RESEARCH.md` (see note in §1), and the code in `fairbench/` (constraints, rules, mandate, data, baselines, postprocess, attribution, quantum, ft, qae)._

How to read the evidence labels in this memo: **verified** means I checked it against the named source in this session; **to verify** means it is from memory or a secondary source and must be confirmed at ingestion. Nothing marked "to verify" should be quoted as fact.

---

## 0. Summary

**The question.** When an ESG-constrained fund lags, how much of the gap sits in the rules that shrink its investable universe, and how much in the choices made inside that universe? FairBench answers descriptively: it builds a reference distribution of portfolios that were feasible under the same rules and locates the realised portfolio in it. It does not measure skill and is not a causal decomposition.

**Five findings that shape the plan.**

1. **Holdings, fund returns and flows are available from the SEC, free and structured, back to Q4 2019.** Form N-PORT gives every position (CUSIP, ISIN, shares, value, % of net assets, country, asset type), the fund's monthly total return per share class and its monthly flows. Only the third month of each fiscal quarter is public, 60 days after quarter-end. Monthly public holdings are not coming soon: the 2024 rule that would have required them is deferred to November 2027 and a February 2026 proposal would drop it.
2. **Three things the brief asks for are not in any SEC filing:** security-level ESG scores, emissions, and sector classification. N-PORT has no sector field. Vendor ESG ratings and GICS sectors are proprietary and cannot be redistributed. Public substitutes exist (SIC codes, EPA facility emissions, public exclusion lists) but they are proxies and must be labelled as such.
3. **US prospectuses contain few numeric ESG constraints.** Most ESG language is soft ("seeks to", "considers"). The hard rules that do exist are statutory (80% names-rule basket, diversification, 25% industry concentration) and business-activity exclusions with revenue thresholds. Explicit ESG-score floors and carbon caps appear almost only in index methodologies. A faithful extractor will therefore return many `soft` and `unobservable` rows. That is the honest result, not a failure.
4. **The parent universe can be rebuilt from public filings.** A fund benchmarked to the S&P 500 can take its investable universe, and the benchmark weights, from the N-PORT filing of an S&P 500 index fund on the same date. This avoids licensing index constituents and is the main reason for the fund selection in §3.
5. **Equal-weight random portfolios are not a fair reference for a cap-weighted fund.** In years when the largest stocks lead, any cap-weighted fund ranks high against equal-weight draws for reasons unrelated to selection. The current code is equal-weight throughout. Weighted reference distributions (§7) are a requirement for real funds, not an extension.

**Decisions I need from you before building** (details in §14):

| # | Decision | My recommendation |
|---|---|---|
| 1 | Contact string for the SEC `User-Agent` header | A team name plus an email you choose. SEC rejected my one direct request without it (HTTP 403). I did not use your email without asking. |
| 2 | Source for security returns | Start with prices implied by N-PORT itself (SEC-only, no dividends, labelled). Add a licensed source if Hanken has WRDS/CRSP or you approve a Tiingo key. |
| 3 | Rename "manager effect" / "constraint effect" in code and deck | Yes, but after the pitch unless you want it now; it touches tests and three slides. |
| 4 | Fund list in §3 | Approve the eight first-wave funds, or edit. |
| 5 | A fast path before the pitch | One active fund (Parnassus Core Equity) end to end on SEC-only data, clearly labelled a proxy mandate. |
| 6 | New optional dependencies | `ortools` (CP-SAT) and `pypdf` as extras; everything else stays on the standard library. |
| 7 | The standards research file | Commit it to `main`; it currently exists only in Codex checkpoint refs. |

---

## 1. What the repository does today, and where it differs from the brief

`FAIRBENCH_INDUSTRY_STANDARDS_RESEARCH.md` is **not on `main` and not in the working tree**. It exists only in Codex checkpoint refs (`refs/codex/turn-diffs/checkpoints/...`, 445 lines). I read it from the latest of those. Its conclusions that bear on this plan: use one passive and one active public case and label the active one a proxy; add weighted feasibility; add point-in-time data; stop calling the within-rule difference a manager effect; calibrate with pseudo-funds drawn from the feasible set.

| Area | Current state (file) | Gap against the brief |
|---|---|---|
| Portfolio representation | Binary selection `x`, fixed `k`; weights assigned after sampling by `postprocess.assign_weights` (`equal`, `inverse_vol`, `esg_tilt`) | Stage 1 exists. Stage 2 (discretised weights) does not. |
| Constraints | Eleven classes in `constraints.py`, all checked on `x`. Shares and averages are equal-weight count conversions (`rules._share_count`). | No weighted feasibility. `attribute()` accepts explicit fund weights but tests feasibility on the 0/1 support only. |
| Rule spec | `rules.py`: 12 kinds, each rule has a verbatim `source` quote and a `note` | No URL, page, dates, confidence, hard/soft type, or disclosed/derived/assumed/unknown status. |
| Mandate extraction | `mandate.py`: one Claude call, structured output, quote checked against the text. Never run live. | No modality (hard/soft) handling, no number-grounding check, no document locator. |
| Data | `Universe`: one cross-section; `esg_score` and `carbon` are required arrays | No time dimension, no identifiers beyond ticker, no way to represent missing ESG data. |
| Samplers | `rejection_sampler` (uniform k-subsets, exact), `mcmc_swap_sample`, `enumerate_feasible` (n ≤ 25), `qae/exact_count.count_dp` (exact counting) | This is "uniform over feasible subsets" only. No weighted or benchmark-aware distribution, no MILP/CP-SAT, no simulated annealing. |
| Attribution | `apps/attribution.py`: percentile, null median with CI, "constraint effect", "manager effect"; default benchmark is the median unconstrained k-subset; needs at least two return rows | Labels conflict with the brief. No reporting-period handling, no frozen-holdings versus NAV distinction, no run manifest. |
| Quantum | Dicke state + XY mixer (fixed `k`), penalty Hamiltonian for five rule types, fault-tolerant oracle for linear rules, amplitude estimation, mean-variance QUBO in `objective_operator` | Stage 2 encoding absent. Risk caps correctly rejected by the oracle as non-linear. |
| Storage | CSV and JSON files | No database, no raw-document archive, no provenance. |

Two conventions in the handoff that the plan keeps: all constraint logic stays in `constraints.py`, and the only model call stays in `mandate.py`. Tests stay offline.

---

## 2. Deliverable 1: source and coverage matrix

Legend for coverage: **full** = the field is reported as filed; **partial** = present for some rows or needs derivation; **none** = not in this source.

### 2.1 Official SEC sources

| Source | What it gives | Format / access | History | Lag | Checked |
|---|---|---|---|---|---|
| EDGAR submissions API `data.sec.gov/submissions/CIK##########.json` | Filing list per registrant: form type, dates, accession numbers | JSON, no key | Full filing history | Under a second | verified (SEC API page) |
| EDGAR archives `sec.gov/Archives/edgar/data/{cik}/{accession}/` | Raw filings: N-PORT `primary_doc.xml`, prospectus HTML, exhibits | XML/HTML | 1990s on | Real time | path pattern to verify |
| `company_tickers_mf.json` | Ticker → CIK, series ID, class ID for funds and ETFs | JSON | Current only | Nightly | verified (used for §3) |
| **Form N-PORT data sets** | Holdings, fund totals, monthly returns per class, flows, securities lending; up to 30 tab-delimited tables per quarter | ZIP per quarter, 229 MB (2019 Q4) to 441 MB (2026 Q3) | Oct 2019 – Sep 2026 | Posted quarterly; underlying reports public 60 days after fiscal quarter-end | verified (SEC page, readme) |
| Form N-CEN data sets | Annual census: index-fund flag, tracking difference and error for index funds, service providers | Flat files | Sep 2018 on | Annual, 75 days after fiscal year-end (to verify) | verified that it exists |
| Risk/Return Summary data sets | XBRL-tagged prospectus summary: objective, strategy narrative, risks, fees, turnover, annual and average returns, benchmark names | ZIP per quarter (~48 MB) | 2010s on (to verify) | Quarterly | verified that it exists |
| Form N-CSR / N-CSRS | Shareholder reports, management's discussion of performance; tailored shareholder reports tagged since mid-2024 (to verify) | HTML / inline XBRL | Full | Semi-annual | to verify |
| Forms 485BPOS, 497, 497K | Prospectus, statement of additional information (SAI), supplements, summary prospectus. Supplements carry mandate changes with effective dates. | HTML | Full | As filed | verified for Parnassus, TIAA-CREF, Nushares (2026 filings seen) |
| Form N-PX | Proxy votes, annual to 30 June, due 31 August | XML since 2024 (to verify); no SEC bulk data set found | Full | Annual | search found no bulk set |
| Form 13F data sets | Quarterly long positions of institutional **managers** | ZIP per quarter | 2013 Q2 on | 45 days | verified |

**The URL in the brief for N-PORT data sets returns 404.** The live page is `https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets`.

### 2.2 What each brief item can come from

| Brief item | Best source | Coverage | Label the value will carry |
|---|---|---|---|
| Fund name, CIK, series, class, ticker | `company_tickers_mf.json`, N-PORT header | full | disclosed |
| Adviser, benchmark name | N-CEN, prospectus XBRL | full | disclosed |
| Fund ISIN / CUSIP | Not in SEC structured data; issuer website | partial | disclosed (issuer) or unknown |
| Mandate rule text | 485BPOS / 497K / SAI; index methodology PDFs for index funds | full text; few numeric rules | disclosed, with hard/soft type |
| Holdings: issuer, CUSIP, shares, value, weight, date, asset type, country, currency | N-PORT | full | disclosed |
| Holdings: ISIN, ticker | N-PORT identifiers table | partial (filer's choice) | disclosed |
| Holdings: **sector** | **Not in N-PORT.** SIC code via issuer CIK; Fama-French industry map | partial | derived (never called GICS) |
| Fund NAV total return | N-PORT monthly returns per class | full, monthly | disclosed |
| Fund market-price return (ETFs) | Not in N-PORT; vendor or issuer site | none from SEC | via price adapter |
| Fund flows | N-PORT sales, redemptions, reinvestments per month | full | disclosed |
| Security total return, dividends, splits | **No SEC source.** Quarter-end prices can be implied from N-PORT (value ÷ shares). | partial | derived, or via licensed adapter |
| Benchmark return and weights | Index levels are proprietary. Proxy: the N-PORT of an index fund tracking it. | partial | derived ("proxy: fund X") |
| Risk-free rate | US Treasury / FRED bill yields | full | disclosed (public) |
| ESG score, controversy flags | **No public source.** Vendor ratings are licensed. | none | unobservable, or via licensed adapter |
| Emissions, carbon intensity | EPA Greenhouse Gas Reporting Program (US facilities, Scope 1), Climate TRACE estimates; revenue from SEC XBRL | partial | derived proxy; missing is not zero |
| Taxonomy alignment, principal adverse impacts | Not applicable to US funds | none | not_applicable |
| Position changes | Difference of consecutive N-PORT snapshots | full at quarter-ends | observed_position_change |
| Public explanations | N-CSR discussion, manager commentaries | partial | direct, with source and date |

### 2.3 Sources I recommend not using for holdings

- **Form 13F.** It is filed by the adviser, not the fund, aggregates every account the adviser manages, covers only long positions in listed 13(f) securities, and arrives 45 days after quarter-end. It cannot identify one fund's portfolio. Its one use here is the SEC's official list of 13(f) securities as a CUSIP-to-name cross-check.
- **Form N-PX.** Proxy votes are stewardship records. They go in their own table and never feed portfolio construction.
- **Issuer websites for bulk holdings history.** ETF issuers publish daily holdings, but their terms of use generally restrict automated collection. Manual download for spot checks only, unless a site's terms allow more.

### 2.4 Access rules the ingestion adapter will follow

SEC's stated ceiling is 10 requests per second per user, with a declared `User-Agent` (organisation and contact email). The adapter will: send the approved `User-Agent`; stay at or below 2 requests per second; send `If-None-Match` / `If-Modified-Since` when the server returned validators and otherwise compare content hashes; back off exponentially with jitter on 403, 429 and 5xx and stop after a fixed number of tries; write every response body unmodified to a content-addressed archive before parsing; never touch a login, paywall or `robots`-restricted path.

**Volume.** For the tracked funds, fetch each fund's own N-PORT XML (a few hundred small files in total). The quarterly bulk ZIPs are about 10 GB for the full history and are only needed if we later want market-wide universes.

---

## 3. Deliverables 2 and 3: candidate funds and their filings

### 3.1 Selection rule

A fund qualifies if (a) it is a US-registered public-equity ESG fund with N-PORT history of at least three years, (b) its rules are written down somewhere public, and (c) **its parent universe can be taken from the N-PORT of a plain index fund on the same benchmark**. I mixed two kinds on purpose:

- **Active funds**, where a human chooses inside the rules. This is the question FairBench exists for.
- **Index funds**, where a published methodology chooses. Here the "choice inside the rules" is the index's selection rule, so the expected within-rule difference between two funds on the same index is zero. These are calibration cases. Two funds tracking one index are a free replication test.

### 3.2 First-wave list (eight funds)

Identifiers are **verified** against `company_tickers_mf.json`, read through a page-fetch tool rather than parsed directly, so the first ingestion run should re-confirm them from the file itself. Holdings counts are approximate and **to verify** from N-PORT. Rule summaries are from the sources in §15 and must be re-read from the current prospectus at ingestion.

| # | Fund (ticker) | Type | Registrant CIK | Series ID | Class ID | Parent universe proxy | Approx. holdings | What the public rules look like |
|---|---|---|---|---|---|---|---|---|
| 1 | Parnassus Core Equity (PRBLX) | Active | 866256 (Parnassus Funds II) | S000000856 | C000002471 | S&P 500 via IVV | ~40 | 80% equities; large-cap definition tied to Russell 1000 median; exclusions on alcohol, tobacco, gambling, weapons and fossil fuels at a 10% revenue test (wording from earlier filings; confirm in the 30 Apr 2026 prospectus) |
| 2 | Nuveen Large Cap Responsible Equity (TISCX) | Quant-active | 1084380 (TIAA-CREF Funds) | S000005371 | C000014633 | S&P 500 via IVV | several hundred | 80% in large caps meeting ESG criteria; excludes owners of fossil fuel reserves; "generally" avoids seven activities; ESG data from MSCI. Benchmark changed 1 Mar 2024, renamed 1 May 2024, adviser changed 1 Aug 2026 |
| 3 | American Century Sustainable Equity (AFDAX) | Active | 100334 | S000006203 | C000017096 | S&P 500 via IVV | ~100 | to extract; expected mostly soft ESG integration |
| 4 | Calvert Equity (CSIEX) | Active | 356682 (Calvert Social Investment Fund) | S000008719 | C000023762 | Russell 1000 via IWB | ~50 | Calvert Principles (qualitative); proprietary research system |
| 5 | SPDR S&P 500 ESG ETF (EFIV) | Index | 1064642 (SPDR Series Trust) | S000069051 | C000220766 | S&P 500 via IVV | ~300 | Tracks S&P 500 Scored & Screened: activity exclusions, UN Global Compact screen, lowest 25% of ESG scores per industry group out, select to ~75% of float cap per industry group |
| 6 | Xtrackers S&P 500 Scored & Screened ETF (SNPE) | Index | 1503123 (DBX ETF Trust) | S000065220 | C000211086 | S&P 500 via IVV | ~300 | Same index as EFIV. Renamed from "S&P 500 ESG ETF" in Feb 2025 |
| 7 | iShares ESG Aware MSCI USA ETF (ESGU) | Optimised index | 1100663 (iShares Trust) | S000055381 | C000174221 | MSCI USA via EUSA | ~290 | Optimiser maximises ESG score subject to a 0.50% tracking-error budget and activity exclusions; other numeric bounds are in the methodology appendix, not yet read |
| 8 | iShares Paris-Aligned Climate Optimized MSCI USA ETF (PABU) | Optimised index | 1100663 | S000074757 | C000232922 | MSCI USA via EUSA | ~100–200 (sources disagree) | The only candidate with a numeric carbon rule: EU Paris-Aligned Benchmark minimums of a 50% cut in greenhouse-gas intensity versus the parent and 7% a year self-decarbonisation, plus exclusions |

Parent-universe funds (**verified** identifiers): iShares Core S&P 500 (IVV, 1100663, S000004310), iShares Russell 1000 (IWB, 1100663, S000004347), iShares MSCI USA Equal Weighted (EUSA, CIK 930667, S000028709; same constituents as MSCI USA by design, different weights, to verify).

**On the 100–300 holdings target.** Four of the eight are inside it (3, 5, 6, 7), and PABU probably is. Parnassus and Calvert hold about 40–50 names: I kept them because concentrated active funds are where selection matters most and sampling is cheapest. TISCX is probably larger than 300.

### 3.3 Reserves (identifiers verified)

SUSA (S000004436), DSI (S000013931, 400 names), SUSL (S000065418) and USSG (S000064994) as a second twin pair, SPYX (S000051701, single fossil-reserve exclusion), ESGV (S000063075), VFTAX (S000004440), FITLX (S000057366), PXLIX (S000055844), DSEFX (S000003423), GCEQX (S000007715), BAFWX (S000037789), NULG (S000055811), NULV (S000055812).

### 3.4 Survivorship warning

US ESG funds are closing in large numbers. Seen this session: Putnam Sustainable Leaders and Sustainable Future ETFs (liquidation June 2026), SPDR MSCI USA Climate Paris Aligned ETF (May 2026), JPMorgan's JCTR (delisted, year to verify), and Morningstar's count of 91 US sustainable-fund closures in one year. A list built in October 2026 from funds that still exist is survivor-biased. The schema keeps `is_last_filing` and a fund end date, and any multi-fund summary must say that closed funds are missing unless we add them.

### 3.5 Forms per fund

Every fund above is a registered open-end fund or ETF, so each is **required** to file the forms below. I confirmed 2026 prospectus filings for funds 1 and 2 and identifiers for all, but I have **not enumerated each fund's filing index**, because direct API calls are blocked until the `User-Agent` decision (§14). The table states what the rules require; the first ingestion run replaces it with observed counts.

| Form | Content used | Funds 1–4 (active) | Funds 5–8 (index) | Frequency |
|---|---|---|---|---|
| NPORT-P | Holdings, returns, flows | required | required | Monthly report, third month of fiscal quarter public |
| 485BPOS | Prospectus and SAI | required | required | Annual |
| 497 / 497K | Supplements, summary prospectus | seen for 1 and 2 | required | As needed |
| N-CSR / N-CSRS | Shareholder report, performance discussion | required | required | Semi-annual |
| N-CEN | Census; index tracking data | required | required, with tracking fields | Annual |
| N-PX | Proxy votes | required | required | Annual |
| 13F-HR | Adviser-level only | filed by the adviser | filed by the adviser | Quarterly |
| Index methodology (not an SEC form) | Selection and weighting rules | not applicable | provider website (S&P DJI, MSCI) | On change |

**History.** N-PORT structured data starts in Q4 2019, giving about 27 public quarter-ends for funds that existed then. EFIV (launched 2020, to verify) has about 24 and PABU (launched 2022, to verify) about 18. All clear the three-year bar.

---

## 4. Deliverable 4: data-gap report

| # | Gap | Consequence | Proposed handling |
|---|---|---|---|
| G1 | No security-level ESG scores in public data | ESG-floor and best-in-class rules cannot be reproduced | Store the rule with `observability = unobservable`. Do not compile it. Report it. Optional licensed adapter. |
| G2 | No public sector classification matching the one a fund uses (GICS) | Sector limits written in GICS terms can only be approximated | Derive sectors from SIC codes with a published mapping; label `derived`; run sensitivity to the mapping |
| G3 | Emissions data are partial | A carbon cap can be stated (PABU) but not evaluated on the provider's data | EPA and Climate TRACE proxies with coverage reported; a missing value is never treated as zero |
| G4 | No SEC source for security total returns | Feasible-portfolio returns need prices | SEC-only panel from N-PORT-implied prices (no dividends, splits flagged), or a licensed adapter (§14) |
| G5 | Public snapshots are quarterly, and on each fund's **fiscal** quarter | A fund and its parent-universe fund can report on different months | Prefer a parent fund on the same cycle; otherwise use the nearest earlier parent snapshot and record the gap in days |
| G6 | 60-day publication lag | The portfolio date and the date the public could know it differ | Store both; analysis is as of the portfolio date; a separate "knowable on" date is kept |
| G7 | Holdings are snapshots | Trades inside the quarter are invisible | Compare frozen-holdings return with NAV return and report the difference (§8) |
| G8 | Soft language dominates active prospectuses | Few compilable rules; the feasible set is mostly shaped by exclusions and statutory limits | Report the rule count by type per fund; run the reference distribution with and without soft rules promoted |
| G9 | Revenue-threshold exclusions need segment revenue | "More than 10% of revenue from tobacco" is not in N-PORT | SIC-code proxy plus public exclusion lists; label `assumed`; list every name excluded and why |
| G10 | Index constituents and levels are proprietary | Cannot ship "the S&P 500" | Use the index fund's own filing as the proxy and name it as such |
| G11 | CUSIP changes and corporate actions | Naive joins across dates break | Identifier history table; flag share-count jumps near split ratios for review |
| G12 | Securities that delist inside a quarter | Dropping them biases returns upward | Keep them in the universe; mark the return `unknown` if no source; report how many |
| G13 | Non-equity lines in equity funds (cash, money-market sweeps, futures, lending collateral) | Weights do not sum to 100% of equities | Keep all lines; define the equity sleeve explicitly; report the residual |
| G14 | Mandate text changes over time | One rule set does not fit all dates | `mandate_versions` with effective dates taken from supplements; never infer a date that is not stated |
| G15 | Survivorship in the fund list | Results across funds overstate | See §3.4 |
| G16 | Index methodology appendices not yet read | Numeric bounds for ESGU and PABU are incomplete | Read at ingestion; until then those constraints are `unknown` |

**What FairBench cannot reproduce from public data, stated once:** any rule that depends on a vendor ESG rating, a vendor controversy flag, a provider's emissions data set, or a manager's internal research score. For those, the rule is recorded with its evidence and marked unobservable, and the result page says the reference distribution omits it.

---

## 5. Deliverable 5: database and schema

**Engine.** SQLite through the standard library, one file, plus a content-addressed directory for raw documents and for large arrays. At this scope (about a dozen funds, a few parents, 27 quarters) the largest table is a few hundred thousand rows. No new dependency. The schema is versioned by a `schema_migrations` table and numbered SQL files.

**Provenance rule.** Every fact row carries five columns: `document_id`, `locator`, `extraction_run_id`, `confidence`, `status`. `status` is one of `disclosed`, `derived`, `assumed`, `unknown`. Source URL, filing date, report date and retrieval time are reached through `documents` and `filings`; a view per table (`v_holdings_provenance`, and so on) flattens them so each value can be shown with its full citation. `locator` is a page, table name and row key, XPath, or paragraph index, depending on the document.

| Table | Purpose | Key columns beyond provenance |
|---|---|---|
| `schema_migrations` | Schema version | `version`, `applied_at` |
| `sources` | Registry of data sources and their licence terms | `name`, `licence`, `may_redistribute`, `terms_url`, `terms_checked_on` |
| `funds` | One row per fund series | `fund_id`, `name`, `registrant_cik`, `series_id`, `adviser`, `asset_class`, `currency`, `inception_date`, `end_date` |
| `share_classes` | Classes and tickers | `class_id`, `fund_id`, `ticker`, `isin`, `cusip` |
| `issuers` | Legal entities | `issuer_id`, `name`, `lei`, `cik`, `sic_code`, `country` |
| `securities` | Instruments | `security_id`, `issuer_id`, `asset_type`, `currency` |
| `security_identifiers` | Identifier history | `security_id`, `scheme` (cusip, isin, ticker, figi), `value`, `valid_from`, `valid_to` |
| `filings` | EDGAR filings | `accession`, `cik`, `form_type`, `filing_date`, `report_date`, `is_amendment`, `supersedes` |
| `documents` | Every retrieved file | `document_id`, `filing_id`, `url`, `sha256`, `retrieved_at`, `http_status`, `etag`, `media_type`, `archive_path` |
| `document_evidence` | Quoted passages | `evidence_id`, `document_id`, `locator`, `evidence_text`, `char_start`, `char_end` |
| `extraction_runs` | Who or what extracted | `run_id`, `method` (xml_parse, xbrl_tag, regex, llm, manual), `tool_version`, `model`, `prompt_hash`, `started_at` |
| `mandate_versions` | A rule set valid for a period | `mandate_version_id`, `fund_id`, `effective_from`, `effective_to`, `review_state` (draft, reviewed, approved), `reviewer` |
| `constraints` | Canonical rules (§6) | `constraint_id`, `mandate_version_id`, fields of the DSL |
| `holding_snapshots` | Header per fund and date | `snapshot_id`, `fund_id`, `report_date`, `filing_id`, `net_assets`, `total_assets` |
| `holdings` | Positions | `snapshot_id`, `security_id`, `shares`, `units`, `market_value`, `pct_net_assets`, `payoff_profile`, `asset_category`, `issuer_category`, `country`, `currency`, `fair_value_level` |
| `prices` | Security prices and returns | `security_id`, `date`, `close`, `total_return_index`, `source_id` |
| `corporate_actions` | Dividends, splits, mergers | `security_id`, `ex_date`, `type`, `ratio_or_amount` |
| `fund_returns` | NAV and market-price returns | `class_id`, `period_end`, `nav_return`, `market_return` |
| `fund_flows` | Monthly flows | `fund_id`, `month_end`, `sales`, `redemptions`, `reinvestments` |
| `esg_observations` | Any ESG datum | `subject_type`, `subject_id`, `metric`, `value`, `unit`, `provider`, `data_date`, `missing_reason` |
| `benchmarks` | Benchmark definitions | `benchmark_id`, `name`, `provider`, `proxy_fund_id`, `is_proxy` |
| `benchmark_constituents` | Weights by date | `benchmark_id`, `date`, `security_id`, `weight` |
| `observed_holding_changes` | Differences between snapshots (§8) | `fund_id`, `security_id`, `from_date`, `to_date`, `change_type`, `shares_delta`, `weight_delta`, `drift_component`, `active_component`, `decision_observability`, `explanation_evidence_id` |
| `reference_universes` | The investable set used for a run | `universe_id`, `fund_id`, `as_of`, `definition`, `n_securities`, `array_path` |
| `portfolio_samples` | A drawn reference set | `sample_id`, `universe_id`, `mandate_version_id`, `distribution`, `weighting_policy`, `sampler`, `seed`, `n_samples`, `n_feasible_estimate`, `acceptance_rate`, `array_path`, `array_sha256` |
| `attribution_results` | One result per fund, period and distribution | `sample_id`, `period_start`, `period_end`, `realised_frozen_return`, `realised_nav_return`, `percentile`, `median`, `p05`, `p95`, `labels_version` |
| `validation_runs` | Sampler checks (§9) | `sampler`, `instance`, `feasibility_rate`, `violation_rate`, `tv`, `kl`, `marginal_error`, `return_dist_error`, `percentile_error`, `runtime_s`, `qubits`, `depth`, `shots`, `seed` |

Sample arrays are stored as compressed files referenced by path and hash, not as rows.

---

## 6. Deliverable 6: constraint DSL

### 6.1 Canonical record

The brief's example, extended with the fields needed to compile and to stay honest:

```json
{
  "constraint_id": "c_0042",
  "fund_id": "S000074757",
  "mandate_version_id": "mv_0003",
  "effective_from": "2024-11-01",
  "effective_to": null,
  "metric": "carbon_intensity",
  "aggregation": "weighted_average",
  "operator": "<=",
  "value": 0.5,
  "unit": "ratio_to_benchmark",
  "reference": "relative_to_benchmark",
  "benchmark_id": "msci_usa_proxy_eusa",
  "scope": "portfolio",
  "group_by": null,
  "weight_basis": "net_assets",
  "constraint_type": "hard",
  "modality_terms": ["at least"],
  "observability": "unobservable",
  "status": "disclosed",
  "confidence": 0.92,
  "source_url": "https://...",
  "source_document": "doc_0191",
  "source_page": 12,
  "evidence_text": "...",
  "retrieved_at": "2026-10-09T08:14:00Z",
  "extraction_method": "llm+quote_check+number_check",
  "review_state": "draft",
  "compile_target": null,
  "note": "Provider emissions data not public; rule recorded, not enforced."
}
```

Vocabulary: `metric` (holding_count, position_weight, group_weight, group_count, esg_score, carbon_intensity, market_cap, tracking_error, volatility, turnover, eligibility); `aggregation` (per_security, count, weight_sum, weighted_average, equal_average, quadratic_form); `operator` (`<=`, `>=`, `==`, `in`, `not_in`); `scope` (security, portfolio, group); `reference` (absolute, relative_to_benchmark, percentile_of_universe); `constraint_type` (hard, soft, regulatory); `observability` (observable, proxy, unobservable); `status` (disclosed, derived, assumed, unknown).

### 6.2 Hard versus soft

Classification is done by code, from a fixed word list, not by the model.

- **Hard markers:** must, shall, will not, may not, does not invest, is prohibited, excluded, at least, no more than, not exceed.
- **Soft markers:** seeks to, aims, normally, generally, typically, intends, expects, considers, may, approximately, where practicable, under normal circumstances.
- If both appear in the evidence, the rule is **soft**. The standard names-rule sentence ("under normal circumstances, at least 80%") is therefore stored as soft with `regulatory_basis = rule_35d-1`, and a reviewer may promote it. Every result is run twice when promotions exist: approved hard rules only, and with soft rules promoted. Both are reported.

### 6.3 Guards against invention

1. **Quote check** (exists): the evidence text must appear in the document.
2. **Number check** (new): every numeric `value` must appear in the evidence text. A value that is not there is rejected and the rule is stored as `unknown`.
3. **Date check** (new): `effective_from` is filled only from a date printed in the document or from the filing's own effective date, and the field records which.
4. **No silent proxies:** a proxy data source sets `observability = proxy` and `status = derived` or `assumed`, and the constraint lists the proxy.
5. Nothing is `approved` without a named reviewer.

### 6.4 Compilation, reusing existing code

- **Stage 1 (equal-weight subsets):** a new adapter turns approved, observable hard constraints into the existing `rules.py` spec, so `compile_spec` and its report keep working unchanged. Soft and unobservable rules become `unmapped` entries with their reason, so the existing report already shows what was left out.
- **Stage 2 (weights):** the same constraints compile to weighted forms: bounds `l ≤ w ≤ u`, linear rows `A w ≤ b`, and quadratic forms `(w − b)ᵀ Σ (w − b) ≤ τ²`. These need new classes in `constraints.py` with a `check_weights` method, since that file is where rule logic lives.

---

## 7. Portfolio representation and the reference distribution

### 7.1 Stage 1: equal-weight subsets (exists)

`x_i ∈ {0,1}`, `Σ x_i = k`, `w_i = x_i / k`. Exclusion fixes `x_i = 0`. An ESG floor is `Σ ESG_i x_i ≥ k · ESG_min`; a carbon cap is `Σ carbon_i x_i ≤ k · carbon_max`; a sector cap is `Σ_{i∈s} x_i ≤ c_s`. All are linear in `x` and already implemented.

One addition and one caution:

- **Holdings range.** For `k_min ≤ Σ x_i ≤ k_max`, drawing uniformly over all feasible subsets puts almost all the mass on the largest allowed `k`, because the number of subsets grows steeply with `k` (for 500 names, each extra holding near `k = 50` multiplies the count by about nine). The default stays "condition on the fund's actual `k`", with `k` varied in sensitivity.
- **Scale.** Rejection sampling is exact and fine for 40 names out of 500 with exclusions and loose caps. For 300 out of 500 with tight per-group rules it will stall. The plan adds an exact sampler that draws backwards through the counting table in `qae/exact_count.py`, which gives uniform samples without rejection wherever that table is tractable.

### 7.2 Stage 2: discretised weights (new)

`w_i = lower_i + step · Σ_b 2^b q_{i,b}` with binary `q_{i,b}`. Choosing `step` so that `(1 − Σ lower_i) / step` is an integer `M` turns the budget into the integer equality `Σ_i Σ_b 2^b q_{i,b} = M`. An upper bound that is not of the form `2^B − 1` steps is met exactly by shrinking the coefficient of the top bit, so no inequality is needed for it. Cardinality needs an indicator `y_i` per asset with `w_i = y_i · min_i + step · Σ_b 2^b q_{i,b}` and the requirement that `q_{i,b} = 0` when `y_i = 0`. Sector, ESG and carbon rules become linear in the bits. Turnover against the previous snapshot is linear after splitting into buys and sells.

**Risk and tracking error are quadratic:** `risk = wᵀ Σ w`, and tracking error is `(w − b)ᵀ Σ (w − b)`. In the bits they are quadratic forms, which is exactly the shape of a QUBO energy, so they fit naturally as an objective or as a soft weight on the distribution. As a hard inequality they are not linear and cannot be handled by the existing linear oracle; they are enforced by a classical quadratic check, as `constraints._RiskCap` already does.

### 7.3 Three reference distributions and why they differ

| | Definition | What it assumes about a "typical" feasible portfolio |
|---|---|---|
| **D1** Uniform over feasible subsets | Every feasible set of names is equally likely. Weights are a fixed function of the set: equal, or proportional to benchmark weight. | Any admissible combination of names is as likely as any other. |
| **D2** Uniform over feasible weighted portfolios | Every feasible point of the Stage 2 weight grid is equally likely. | Any admissible weighting is as likely as any other, including very uneven ones. |
| **D3** Benchmark-aware | D2 reweighted by `exp(−TE(w)² / 2τ²)`, where TE is tracking error against the parent benchmark and `τ` is a stated scale. | Portfolios near the benchmark are more likely, as they are for real funds. |

They are different objects, not three estimates of one thing.

- **D1 against D2.** With three assets, two holdings and weights in quarters, D1 has three portfolios, each (½, ½). D2 has nine: each pair with (¼, ¾), (½, ½), (¾, ¼). The average return is the same but D2 is more spread out, so the same fund sits closer to the middle of D2. Now add "asset A at most ¼". Under D1 with equal weights every pair containing A is infeasible, leaving one portfolio. Under D2 the pairs containing A survive with A at ¼, giving five. **A weight rule changes which names are feasible at all**, so D1 with equal weights can exclude portfolios a real manager could hold.
- **D2 against D3.** Same feasible set, different probabilities. As `τ → ∞`, D3 becomes D2. As `τ → 0`, D3 collapses onto the feasible portfolio closest to the benchmark, which is roughly what an optimised index fund is. D3's centre moves toward the benchmark return and its spread shrinks, so a given active return looks more extreme under D3 than under D2. The percentile will be reported as a curve in `τ`, not a single number.
- **Weighting policy inside D1.** Equal weights tilt every draw toward smaller companies compared with a cap-weighted fund. Benchmark-proportional weights (`w_i ∝ b_i x_i`, with `b_i` taken from the parent index fund's filing) remove most of that tilt. Both are run; the gap between them is reported as the sensitivity to weighting policy.

**D2 has two readings** when the number of holdings is limited: uniform over all feasible grid points (sets of names with more feasible weightings get more mass), or uniform over sets of names and then over weightings. They coincide when all assets share the same bounds and no other weighted rule binds. The first is the default; the second goes in sensitivity.

**Recommended headline for real funds:** D1 with benchmark-proportional weights at the fund's own `k`, shown beside D2 and the D3 curve. This is a judgement, and it is the main research decision for you to confirm.

### 7.4 What every result will carry

Distribution and its exact definition; weighting policy; number of feasible portfolios (exact from enumeration or the counting table when possible, otherwise acceptance rate × number of proposals with a Wilson interval, reported as a power of ten; for D2 at scale the grid size and acceptance rate); sampling method; seed; number of constraint violations in the sample (must be zero, checked independently of the sampler); effective sample size for reweighted or chain samples; and the same statistic under each other weighting policy.

---

## 8. Attribution logic, labels, and what cannot be observed

### 8.1 Procedure for one reporting period

For a fund with public snapshots at dates `t` and `t+1` (consecutive fiscal quarter-ends):

1. **Universe at `t`:** the parent index fund's holdings at `t` (or the nearest earlier snapshot, with the gap recorded), with benchmark weights from the same filing.
2. **Rules at `t`:** the mandate version effective at `t`.
3. **Feasible set:** sample under each distribution in §7.
4. **Returns over `(t, t+1]`:** each feasible portfolio is held without trading. So is the fund's disclosed portfolio at `t`. This is the **frozen-holdings return**.
5. **Percentile** of the fund's frozen-holdings return in each distribution, with Monte Carlo error, the median, and the 5–95% interval.
6. **Reported separately:** the fund's actual NAV return over the same months (from N-PORT) and its difference from the frozen-holdings return. That difference contains trades inside the quarter, fees, cash, flows and pricing differences, and is known in the literature as the return gap. It is not compared with the random portfolios, which have no such components.

Only information dated `t` or earlier defines the universe, rules and weights; returns come strictly after `t`.

**For index funds** the rule set used is the eligibility and structure rules only (exclusions, coverage per industry group, holdings count). If the index's ranking or optimiser were included, the feasible set would be a single portfolio. The index portfolio's position in the reference distribution then describes what the selection rule did relative to arbitrary selections meeting the same structure.

### 8.2 Labels

| Current name in code | Proposed name | Meaning |
|---|---|---|
| null distribution | rule-conditioned reference distribution | Returns of feasible portfolios |
| null median, interval | rule-conditioned return range | Median and 5–95% interval |
| constraint effect | rule-conditioned shift | Reference median under the rules minus reference median without them, same weighting policy |
| manager effect | within-mandate return difference | Fund's frozen-holdings return minus the rule-conditioned median |
| percentile | realised portfolio percentile | Rank of the fund's frozen-holdings return |
| the whole output | descriptive rule-versus-choice comparison | Not causal; not a measure of skill |

One period's percentile says where the disclosed portfolio fell among portfolios its rules allowed in that period. It does not show skill. Across periods the percentiles are shown as a series, without a pooled "skill" statistic.

### 8.3 Observed changes, not decisions

Between snapshots, for each security: change in shares (adjusted for flagged splits), change in weight, and the split of the weight change into **drift** (what the weight would have become with no trading, from relative price moves) and the **remainder**. Change types: `observed_new_position`, `observed_exit`, `observed_holding_increase`, `observed_holding_decrease`, plus sector and exposure changes aggregated from these. Flags: consistent with pro-rata scaling for fund flows; corporate action suspected; inside an index reconstitution window.

`decision_observability` takes four values:

- `direct`: the manager published the action and its reason, with source and date.
- `observed_position_change`: seen in two filings, no stated reason.
- `inferred`: deduced, for example from drift arithmetic.
- `unavailable`.

Limits recorded with every such row: holdings are snapshots; the filing date differs from the portfolio date; some positions may be reported in aggregate as miscellaneous securities; changes mix price effects, flows, corporate actions and rebalancing; order timing and rationale are not public; 13F data is adviser-level and delayed; proxy votes are stewardship, not portfolio construction.

---

## 9. Deliverable 8: validation and the classical-to-quantum plan

### 9.1 Classical baselines first

| Baseline | Status | Role | Uniform? |
|---|---|---|---|
| Rejection sampling | exists | Default sampler for D1 | Yes, exactly |
| Exhaustive enumeration, n ≤ 25 | exists | Ground truth for subsets | Exact |
| Exact counting table | exists (count only) | Add backward sampling for scale | Yes, exactly |
| Weight-grid enumeration | new | Ground truth for D2 on tiny cases | Exact |
| MILP / CP-SAT | new | Feasibility proof, which rule makes a set infeasible, exact minimum and maximum feasible return, enumeration on tiny cases | **No.** A solver's solution order is not a uniform sample and will not be used as one. |
| Simulated annealing | new | Classical baseline on the same energy function as the quantum encoding; finds starting points for chains | No; its bias is measured against the exact distribution |
| Swap chain | exists | Tight rule sets; needs multiple chains and effective-sample-size diagnostics | Only if it mixes |
| Hit-and-run on the weight polytope | new | D2 at scale | In the limit; diagnostics required |

### 9.2 Metrics, each against an exact distribution on small cases

Feasibility rate; violation rate per constraint; total variation distance, compared with the sampling-noise floor that `metrics.tv_expected_uniform` already computes; KL divergence from the exact distribution to the sampler's, reported only when the sampler has put mass on every exact state (otherwise it is infinite, and a smoothed value is reported and marked as such); largest and root-mean-square error in per-asset inclusion frequency; return-distribution error as a Kolmogorov–Smirnov statistic and a Wasserstein distance; percentile error for a planted fund; runtime; qubit count; transpiled depth and two-qubit gate count; number of samples.

**Calibration test from the standards document:** funds drawn from the feasible set itself must have uniformly distributed percentiles. Failure means the sampler or the weighting is wrong.

**Small real cases.** Simulators stop near 20–25 qubits. Real-data quantum checks therefore use sub-universes: one sector of the parent universe on one date, with the fund's own count in that sector. That is a real question (selection within a sector, allocation held fixed) and it is small enough for exact enumeration.

### 9.3 Conversion to a quantum encoding

| Element | Stage 1 (exists) | Stage 2 (new) |
|---|---|---|
| Variables | `N` qubits, one per eligible asset | `N · B` qubits for weight bits, plus `N` indicators if holdings are limited |
| Budget | Dicke state and XY mixer keep exactly `k` ones | Weighted sum of bits equals `M`. This is not a fixed number of ones, so the Dicke construction does not carry over. Use a squared penalty, which is a QUBO term. |
| Exclusions | Dropped from the register | Dropped from the register |
| Linear rules | Fault-tolerant oracle (exact after quantisation) or penalty | Same oracle with larger integer coefficients, or penalty with slack bits |
| Risk, tracking error | QUBO energy `xᵀ Σ x` in `objective_operator`; rejected by the linear oracle | Quadratic in the bits: a QUBO energy. Natural for D3 as a soft weight. As a hard cap it stays a classical check. |
| Example size | 16 assets: 16 qubits plus oracle workspace | 5 assets × 2 bits + 5 indicators + ~3 slack ≈ 18 qubits |

The adapter will emit, for a given constraint set: the QUBO matrix and offset, the variable map, the penalty weights with a note on which encodings are exact and which are surrogates (as `penalty_exactness` does today), and a decoder back to weights. Its output is tested against the exact grid enumeration: zero energy if and only if feasible, for every state of a small instance.

**Claims.** The repository's existing findings stand: Dicke plus filtering matches classical rejection; amplitude estimation needs fewer oracle queries in noiseless simulation only. Nothing in this plan supports a quantum advantage claim, and none will be made without a reproducible comparison against the baselines above.

---

## 10. Deliverable 7: quarterly refresh schedule

| When | Action | Trigger for re-extraction |
|---|---|---|
| Weekly (light) | Poll the submissions index for each tracked registrant; record new filings | A new 485BPOS, 497 or 497K opens a draft mandate version for review |
| About day 60–65 after each fund's fiscal quarter-end | Fetch the new NPORT-P; parse; compute holding changes; run validation checks | — |
| First week of January, April, July, October | Note the new SEC N-PORT bulk set (not downloaded unless needed); refresh `company_tickers_mf.json`; re-check identifiers | — |
| After each fund's fiscal year-end | N-CEN (about 75 days), annual prospectus update (about 120 days), N-CSR | New prospectus → mandate review |
| September | N-PX (due 31 August) into the stewardship table | — |
| On index-provider methodology change | Re-read methodology; new mandate version for the tracking funds | Yes |
| Each refresh | Re-run attribution for the newly completed period; write a manifest | — |
| Quarterly | Re-read licence terms of each non-SEC source and record the date | — |

Each fund has its own fiscal calendar, so the schedule is driven by a per-fund table of quarter-ends, filled from the first ingestion.

---

## 11. Deliverable 9: legal and licensing risks

This is a risk list for the team, not legal advice.

| Item | Risk | Handling |
|---|---|---|
| SEC data | Low. Public filings; SEC asks for fair access and a declared `User-Agent`, and disclaims accuracy. | Follow §2.4. Carry the SEC's accuracy disclaimer. |
| Prospectus and report text | Low to moderate. Written by registrants. | Store raw files privately; publish short quoted evidence with citation, not whole documents. |
| CUSIP | Moderate. Licensed identifier. | Use CUSIPs as they appear in filings; do not publish a CUSIP master file; use FIGI or an internal key in any shared example data. |
| GICS sectors | High if copied. Proprietary. | Do not use. SIC-derived sectors, named as such. |
| Index constituents, levels, names | Moderate to high. Provider intellectual property and trademarks. | Use index funds' own filings as proxies and say so. Refer to indexes by name only to describe a fund's stated benchmark. |
| Index methodology documents | Moderate. Copyrighted. | Store URL, hash and short quotes. |
| ESG ratings and controversy data | High. Licensed; public look-up pages forbid collection. | Not collected. Optional adapter for a licensed copy the user supplies. |
| Price data | Varies by vendor; many free tiers forbid redistribution; unofficial scrapers breach terms. | SEC-implied prices by default. Vendor data stays local and is never committed. |
| Fund-manager websites | Moderate. Terms often bar automated collection. | Manual download only, unless a site's terms permit more; record the terms date. |
| Naming real funds | Reputational. A percentile can be read as a verdict on a manager. | Neutral labels (§8.2), proxy-mandate label, full method and assumptions with every result, no skill language. |
| Model use | Low. Only public documents are sent to the API. | Keep it that way; no private data in prompts. |
| Investment advice | Low. | State that outputs are research, not advice. |

**What may be committed to the repository:** code, schema, tests, and a small example data set built only from SEC filings with internal or FIGI identifiers. Raw archives and any vendor data stay out of git.

---

## 12. Deliverable 10: staged implementation plan

Effort is in working sessions of roughly half a day. Each stage ends with offline tests and a short written result. Stages A–C can run before stage D is designed in detail.

| Stage | Builds (brief's item numbers) | Reuses | Done when |
|---|---|---|---|
| **A. Ingestion** (2 sessions) | Polite SEC client and raw archive (1, 2); filing metadata parser (3); N-PORT XML parser (4); SQLite schema v1 | — | One fund's full N-PORT history is in the database; every holding row resolves to a URL, filing date and retrieval time; re-running makes no new requests |
| **B. Identity and returns** (2) | Identifier normaliser (7); price and NAV adapter interface with the SEC-implied price source (8); holdings-difference calculator (9) | — | Weights re-derived from shares × price match reported weights within tolerance; unresolved identifiers are listed, not dropped |
| **C. Mandates** (2) | Document parser with paragraph locators (5); DSL v2 with modality, quote, number and date checks (6); adapter to the existing rule spec | `mandate.py`, `rules.py` | Two reviewers' hand encoding of two prospectuses compared with the extractor; every rule has evidence; no value absent from its quote |
| **D. Reference distributions** (3) | Point-in-time universe builder; weighting policies; D1 at scale; Stage 2 grid, D2 and D3 samplers (10); weighted constraint classes | `constraints.py`, `baselines.py`, `exact_count.py` | Each sampler passes §9.2 on exact small cases; pseudo-fund percentiles are uniform |
| **E. Exact validation and solvers** (2) | Grid enumeration, validation metrics (11); CP-SAT feasibility and extremes; simulated annealing | `metrics.py`, `enumerate_feasible` | Validation table produced for three small instances |
| **F. Attribution on real periods** (2) | Period runner, frozen versus NAV returns, neutral labels, run manifest | `apps/attribution.py` | Twelve quarters for one active fund with all three distributions and sensitivity |
| **G. Quantum encoding** (2) | Stage 2 QUBO adapter (12); small-case comparison with exact distributions | `quantum/hamiltonian.py`, `ft/oracle.py`, `backends.py` | Encoding proven exact or labelled surrogate on enumerated instances; metrics reported; no advantage claim |
| **H. Packaging** (1) | Example data set, tests (13); reproducibility report (14); handoff update | — | A clean checkout reproduces the example end to end offline |

**Fast path before the pitch (one to two sessions, optional).** Parnassus Core Equity, universe from IVV, twelve quarters, D1 with equal and benchmark-proportional weights, rules limited to the holdings count, statutory limits and SIC-proxy exclusions, returns from N-PORT-implied prices. Every slide number would carry: "proxy mandate; price returns without dividends; equal-weight and benchmark-weight references; not a measure of skill". If the identifier or price step does not close cleanly in the first session, stop and keep the synthetic result, as `PITCH_PLAN.md` already says.

**What will not be replaced.** `constraints.py`, `rules.py`, `baselines.py`, the quantum and amplitude-estimation code and their 680 tests stay as they are. New behaviour is added beside them; the label change in `attribution.py` keeps the old field names as aliases.

---

## 13. Extending to European SFDR funds later

The design keeps Europe out of scope but does not block it:

- **Sources are adapters.** A European adapter would read fund documents from manager sites and national registers. There is no European equivalent of N-PORT, so holdings would come from issuer disclosures or a licensed feed, and coverage would be thinner.
- **Mandate documents differ.** SFDR pre-contractual annexes state "binding elements" explicitly, with minimum sustainable-investment and taxonomy percentages. Those map directly to hard `weight_sum` constraints, which makes European rule extraction easier than the US case even though holdings are harder.
- **The DSL already has the vocabulary:** taxonomy alignment and principal adverse impact indicators are metrics with `not_applicable` status for US funds.
- **Fund-name rules** (an 80% basket plus benchmark-style exclusions) are `weight_sum` and `eligibility` constraints.
- **Identifiers** are keyed on ISIN and LEI as well as CUSIP, and currency is a column throughout.

---

## 14. Open decisions

1. **SEC `User-Agent` contact.** Required. Format: organisation or project name and a monitored email.
2. **Returns source.** (a) SEC-implied quarter-end prices, no dividends: free, redistributable, approximate. (b) A licensed daily source: accurate, not redistributable. (c) Both, with (a) as the public example and (b) as the working data. I recommend (c) if you have access to one.
3. **Headline reference distribution.** §7.3 recommends D1 with benchmark-proportional weights at the fund's `k`.
4. **Label change timing.** Before or after the pitch.
5. **Fund list.** §3.2.
6. **Fast path.** §12.
7. **Dependencies.** `ortools`, `pypdf` as optional extras.
8. **Live model calls.** Mandate extraction on real prospectuses needs the API key that has not yet been used; a few calls per fund.
9. **Standards research file.** Commit to `main` or leave out.

---

## 15. Sources

SEC and official:
[EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) ·
[Form N-PORT data sets](https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets) ·
[N-PORT data set readme](https://sec.gov/files/nport_readme.pdf) ·
[Form 13F data sets](https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets) ·
[Accessing EDGAR data](https://www.sec.gov/os/accessing-edgar-data) ·
[Fund ticker and series file](https://www.sec.gov/files/company_tickers_mf.json) ·
[Risk/Return Summary data sets](https://www.sec.gov/data-research/sec-markets-data/mutual-fund-prospectus-riskreturn-summary-data-sets) ·
[Form N-CEN data sets](https://www.sec.gov/dera/data/form-ncen-data-sets) ·
[SEC fact sheet on the 2026 N-PORT proposal](https://www.sec.gov/files/ic-35962-fact-sheet.pdf)

N-PORT rule status:
[Sidley, March 2026](https://www.sidley.com/en/insights/newsupdates/2026/03/us-sec-proposes-to-scale-back-2024-form-n-port-amendments) ·
[K&L Gates, March 2026](https://www.klgates.com/Fast-Track-to-Fine-Tuned-How-the-SECs-New-Form-N-PORT-Proposed-Amendments-Refine-the-Rules-for-Fund-Reporting-3-2-2026) ·
[Vedder Price on the extension](https://vedderprice.com/sec-extends-compliance-dates-for-form-n-port-amendments)

Fund documents and methodologies:
[Parnassus Funds II 497K, 2026](https://www.sec.gov/Archives/edgar/data/0000866256/000119312526195559/d97941d497k.htm) ·
[TIAA-CREF Funds 497K, 2026](https://www.sec.gov/Archives/edgar/data/0001084380/000093041326002290/c116917_497k.htm) ·
[MSCI Extended ESG Focus methodology](https://www.msci.com/index/methodology/latest/ExtendedESGFocus) ·
[MSCI Climate Paris Aligned Select methodology](https://www.msci.com/index/methodology/latest/ClimatePABSel) ·
[MSCI on the EU Paris-Aligned Benchmark](https://www.msci.com/our-solutions/climate-investing/climate-indexes/eu-paris-aligned-benchmark) ·
[S&P 500 ESG index design note](https://spglobal.com/en/research-insights/articles/the-sp-500-esg-index-integrating-environmental-social-and-governance-values-into-the-core) ·
[Cboe factsheet on the S&P 500 Scored & Screened index](https://res-certification.cboe.com/resources/esg/fs-sp-500-esg-index.pdf) ·
[iShares PABU page](https://www.ishares.com/us/products/325725/ishares-paris-aligned-climate-msci-usa-etf/)

Fund closures:
[Bloomberg Government on ESG fund closures](https://news.bgov.com/securities-law/wave-of-esg-fund-closures-builds-in-us-with-few-new-launches-1) ·
[Franklin Templeton on Putnam ETF liquidations](https://www.franklintempleton.com/press-releases/news-room/2026/franklin-templeton-announces-plan-to-liquidate-putnam-etfs) ·
[State Street ETF line-up changes](https://www.streetinsider.com/Business+Wire/State+Street+Investment+Management+Announces+Changes+to+ETF+Lineup/25606539.html)

### Not verified in this session

Fiscal year-ends and therefore public snapshot months of each fund; the filing deadlines quoted in §10 for N-CEN and prospectus updates; exact holdings counts; fund inception dates; the current Parnassus exclusion wording; numeric bounds in the MSCI methodology appendices; whether EUSA's constituents equal MSCI USA's on every date; the N-PORT XML path pattern and whether EDGAR returns cache validators; the structured format of N-PX and tailored shareholder reports; licence terms of every non-SEC source named as a proxy (EPA, Climate TRACE, OpenFIGI, GLEIF, public exclusion lists, any price vendor); whether Hanken provides WRDS/CRSP.
