# FairBench on real US funds: sources, candidate funds, data gaps

_Reference notes, researched 2026-10-08. This file keeps only what is not written down anywhere else: where the data
comes from, which funds are candidates and their SEC identifiers, what public data cannot provide, the refresh
calendar, licensing risks, and how Europe would fit. Section numbers have gaps because the design sections of the
original memo were removed once the design was built._

**Where the rest went.** Results and checks: [`REAL_FUND_RESULTS.md`](REAL_FUND_RESULTS.md); the nine-fund table: `results/skill_table/`. Database schema:
`fairbench/store/schema.sql`. Constraint format and hard/soft wording: `fairbench/mandates/`. Reference
distributions and the weight grid: `fairbench/portfolio/reference.py`, `weights.py`. Period logic and labels:
`fairbench/apps/real_fund.py`. Quantum encoding: `fairbench/quantum/encoding.py`. Each module's docstring states its contract.

How to read the evidence labels: **verified** means it was checked against the named source on 2026-10-08;
**to verify** means it is from memory or a secondary source and must be confirmed before it is quoted as fact.

---

## 0. Summary

**The question.** When an ESG-constrained fund lags, how much of the gap sits in the rules that shrink its investable universe, and how much in the choices made inside that universe? FairBench answers descriptively: it builds a reference distribution of portfolios that were feasible under the same rules and locates the realised portfolio in it. It does not measure skill and is not a causal decomposition.

**Five findings that shape the plan.**

1. **Holdings, fund returns and flows are available from the SEC, free and structured, back to Q4 2019.** Form N-PORT gives every position (CUSIP, ISIN, shares, value, % of net assets, country, asset type), the fund's monthly total return per share class and its monthly flows. Only the third month of each fiscal quarter is public, 60 days after quarter-end. Monthly public holdings are not coming soon: the 2024 rule that would have required them is deferred to November 2027 and a February 2026 proposal would drop it.
2. **Three things the brief asks for are not in any SEC filing:** security-level ESG scores, emissions, and sector classification. N-PORT has no sector field. Vendor ESG ratings and GICS sectors are proprietary and cannot be redistributed. Public substitutes exist (SIC codes, EPA facility emissions, public exclusion lists) but they are proxies and must be labelled as such.
3. **US prospectuses contain few numeric ESG constraints.** Most ESG language is soft ("seeks to", "considers"). The hard rules that do exist are statutory (80% names-rule basket, diversification, 25% industry concentration) and business-activity exclusions with revenue thresholds. Explicit ESG-score floors and carbon caps appear almost only in index methodologies. A faithful extractor will therefore return many `soft` and `unobservable` rows. That is the honest result, not a failure.
4. **The parent universe can be rebuilt from public filings.** A fund benchmarked to the S&P 500 can take its investable universe, and the benchmark weights, from the N-PORT filing of an S&P 500 index fund on the same date. This avoids licensing index constituents and is the main reason for the fund selection in §3.
5. **Equal-weight random portfolios are not a fair reference for a cap-weighted fund.** In years when the largest stocks lead, any cap-weighted fund ranks high against equal-weight draws for reasons unrelated to selection. The current code is equal-weight throughout. Weighted reference distributions are a requirement for real funds, not an extension.

---

## 2. Source and coverage matrix

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

## 3. Candidate funds and their filings

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

## 4. Data-gap report

| # | Gap | Consequence | Proposed handling |
|---|---|---|---|
| G1 | No security-level ESG scores in public data | ESG-floor and best-in-class rules cannot be reproduced | Store the rule with `observability = unobservable`. Do not compile it. Report it. Optional licensed adapter. |
| G2 | No public sector classification matching the one a fund uses (GICS) | Sector limits written in GICS terms can only be approximated | Derive sectors from SIC codes with a published mapping; label `derived`; run sensitivity to the mapping |
| G3 | Emissions data are partial | A carbon cap can be stated (PABU) but not evaluated on the provider's data | EPA and Climate TRACE proxies with coverage reported; a missing value is never treated as zero |
| G4 | No SEC source for security total returns | Feasible-portfolio returns need prices | SEC-only panel from N-PORT-implied prices (no dividends, splits flagged), or a licensed adapter (§14) |
| G5 | Public snapshots are quarterly, and on each fund's **fiscal** quarter | A fund and its parent-universe fund can report on different months | Prefer a parent fund on the same cycle; otherwise use the nearest earlier parent snapshot and record the gap in days |
| G6 | 60-day publication lag | The portfolio date and the date the public could know it differ | Store both; analysis is as of the portfolio date; a separate "knowable on" date is kept |
| G7 | Holdings are snapshots | Trades inside the quarter are invisible | Compare frozen-holdings return with NAV return and report the difference (done in `fairbench/apps/real_fund.py`) |
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

## 10. Quarterly refresh schedule

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

## 11. Legal and licensing risks

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
| Naming real funds | Reputational. A percentile can be read as a verdict on a manager. | Neutral labels (see `REAL_FUND_RESULTS.md`), proxy-mandate label, full method and assumptions with every result, no skill language. |
| Model use | Low. Only public documents are sent to the API. | Keep it that way; no private data in prompts. |
| Investment advice | Low. | State that outputs are research, not advice. |

**What may be committed to the repository:** code, schema, tests, and a small example data set built only from SEC filings with internal or FIGI identifiers. Raw archives and any vendor data stay out of git.

---

## 13. Extending to European SFDR funds later

The design keeps Europe out of scope but does not block it:

- **Sources are adapters.** A European adapter would read fund documents from manager sites and national registers. There is no European equivalent of N-PORT, so holdings would come from issuer disclosures or a licensed feed, and coverage would be thinner.
- **Mandate documents differ.** SFDR pre-contractual annexes state "binding elements" explicitly, with minimum sustainable-investment and taxonomy percentages. Those map directly to hard `weight_sum` constraints, which makes European rule extraction easier than the US case even though holdings are harder.
- **The DSL already has the vocabulary:** taxonomy alignment and principal adverse impact indicators are metrics with `not_applicable` status for US funds.
- **Fund-name rules** (an 80% basket plus benchmark-style exclusions) are `weight_sum` and `eligibility` constraints.
- **Identifiers** are keyed on ISIN and LEI as well as CUSIP, and currency is a column throughout.

---

## 14. Decisions taken

**Taken on 2026-10-09** (the team said "you decide"; each can be reversed):

| # | Decision | Taken |
|---|---|---|
| 1 | SEC `User-Agent` contact | Read from the environment variable `FAIRBENCH_SEC_USER_AGENT`; no default, and the client refuses to run without it. On 2026-10-09 the user asked for their contact email to be used, and the first downloads were made with it. |
| 2 | Returns source | Price returns implied by N-PORT (SEC-only) by default; a CSV adapter takes total returns from a licensed source. |
| 3 | Headline reference distribution | Uniform subsets with benchmark-proportional weights **capped at the fund's own largest position**. The cap was added during the build: without it, 40 names drawn from a 500-name index give one very large company about half the portfolio. |
| 4 | Label change | The new real-fund code uses the neutral labels. `apps/attribution.py` and the deck are untouched before the pitch. |
| 5 | Fund list | The eight funds of §3.2 stand; the scripts take any series id. |
| 6 | Fast path | Run on 2026-10-09 for Parnassus Core Equity; results in `results/real_fund_parnassus_core_equity*`. |
| 7 | Dependencies | None added. The MILP baseline uses `scipy.optimize.milp`, already a dependency; PDF parsing imports `pypdf` only if called. |
| 8 | Live model calls | Not made. Extraction is tested with a fake client. |
| 9 | Standards research file | Restored from the Codex checkpoint and committed. |

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
