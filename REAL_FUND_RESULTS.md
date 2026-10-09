# FairBench on a real fund: results and checks

_Generated 2026-10-09 by `scripts/reproducibility_report.py` from commit `385bdec`. Do not edit by
hand; re-run the script. Python 3.11.9, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1,
qiskit 2.5.2._

## In plain words

**The question.** When an ESG fund lags, is it the rules or the manager?

**The data.** All of it is free and public, from the US regulator (the SEC), which makes every fund publish what it
owns four times a year. No finance API and no paid data were used.

- Parnassus Core Equity Fund: its actual holdings at 28 quarter-ends, 2019-09-30 to 2026-06-30 (about 36 stocks each).
- iShares Core S&P 500 ETF: every company in the index on the same dates. This is the pool the random portfolios are
  drawn from, and it stands in for "the market".
- Prices: worked out from the filings themselves (value divided by number of shares). No dividends.

**The method.** For each quarter, take the fund's real portfolio and build 5,000 random portfolios of
the same size from companies that pass the rules. See where the fund ranks among them: 0 is worst, 100 is best, 50 is
the typical portfolio that follows the same rules.

**What came out.**

| | Price growth, 2019-09 to 2026-06 |
|---|---|
| iShares Core S&P 500 ETF | +149% |
| Parnassus Core Equity Fund | +135% |
| Typical 36-stock portfolio under the same rules | +105% |

- **The rules cost about 2 points** over the whole period (+105% with the exclusions, +107% without).
- **Quarter by quarter the fund ranked 54 out of 100 on average** (56 if the random
  portfolios are equally weighted). Pure luck would average 50, give or take 6, so anything between about
  39 and 61 cannot be told apart from luck. It beat the typical portfolio in
  59% of quarters.
- **Each quarter's portfolio held unchanged until 2026-06: average rank 61**
  (67 equally weighted). Read this one with care: the windows overlap almost completely, so the
  early portfolios dominate and there is no "give or take" for it.
- **Why the fund trailed the index by 14 points:** mostly because any 36-stock portfolio tends to trail
  an index that a few very large companies drove. The fund's picks did 30 points better than the typical
  portfolio its rules allowed.

![Parnassus Core Equity Fund against the index and the typical rule-abiding portfolio](results/real_fund_parnassus_core_equity.png)

**What must be said with it.**

- **Proxy rules.** The rules applied are the number of holdings and an exclusion list of 27 companies
  built from the SEC's own industry codes (1 alcohol; 22 fossil fuels; 2 tobacco; 2 weapons). Most defence, casino and drinks companies are not caught by
  those codes. The fund's real ESG research is private and cannot be reproduced.
- **"The manager's decisions" are quarter-end snapshots.** Trades inside a quarter, and the reasons for any
  trade, are not public.
- **The latest date is 2026-06-30.** Holdings become public about two months after each quarter-end.
- **Price returns, no dividends.** One fund. A rank is a description, not proof of skill.

## Quarter by quarter

"Typical portfolio" is the middle one of the random rule-abiding portfolios (benchmark-proportional weights, capped
at the fund's own largest position). "Exclusions' effect" is how much the exclusion list moved that middle portfolio.

| quarter ending | fund | index fund | typical portfolio | rank that quarter | rank if held to 2026-06 | exclusions' effect |
|---|---|---|---|---|---|---|
| 2019-12-31 | +6.4% | +8.5% | +7.2% | 36 | 85 | -0.02% |
| 2020-03-31 | -16.4% | -20.0% | -21.8% | 97 | 83 | +0.99% |
| 2020-06-30 | +18.2% | +20.0% | +17.9% | 54 | 85 | -0.11% |
| 2020-09-30 | +12.0% | +8.5% | +7.8% | 94 | 91 | +0.79% |
| 2020-12-31 | +10.9% | +11.6% | +13.7% | 17 | 88 | -0.14% |
| 2021-03-31 | +7.2% | +5.8% | +7.5% | 45 | 90 | -0.59% |
| 2021-06-30 | +7.2% | +8.1% | +6.7% | 61 | 85 | -0.07% |
| 2021-09-30 | -0.2% | +0.1% | -0.7% | 62 | 79 | +0.14% |
| 2021-12-31 | +10.5% | +10.6% | +9.3% | 67 | 78 | +0.10% |
| 2022-03-31 | -5.7% | -5.1% | -6.1% | 56 | 72 | -1.37% |
| 2022-06-30 | -16.3% | -16.5% | -14.8% | 28 | 71 | -0.47% |
| 2022-09-30 | -6.8% | -5.3% | -5.8% | 32 | 49 | -0.35% |
| 2022-12-31 | +10.0% | +7.0% | +10.8% | 40 | 59 | -0.59% |
| 2023-03-31 | +7.5% | +7.0% | +3.6% | 83 | 72 | +0.59% |
| 2023-06-30 | +8.3% | +8.3% | +4.4% | 91 | 70 | +0.30% |
| 2023-09-30 | -3.9% | -3.6% | -4.9% | 69 | 73 | -0.80% |
| 2023-12-31 | +11.3% | +11.2% | +12.1% | 37 | 34 | +0.98% |
| 2024-03-31 | +9.4% | +10.2% | +8.3% | 66 | 44 | -0.17% |
| 2024-06-30 | +1.9% | +3.8% | -1.3% | 90 | 68 | +0.15% |
| 2024-09-30 | +5.1% | +5.5% | +8.0% | 13 | 49 | +0.30% |
| 2024-12-31 | +0.9% | +2.1% | -1.6% | 78 | 23 | +0.15% |
| 2025-03-31 | -2.9% | -4.6% | -0.6% | 19 | 25 | -0.56% |
| 2025-06-30 | +10.4% | +10.5% | +7.2% | 79 | 36 | +0.46% |
| 2025-09-30 | +2.8% | +7.8% | +4.2% | 31 | 29 | -0.03% |
| 2025-12-31 | +2.0% | +2.3% | +1.5% | 58 | 34 | +0.02% |
| 2026-03-31 | -7.2% | -4.7% | -2.2% | 7 | 29 | -1.37% |
| 2026-06-30 | +14.0% | +14.6% | +12.8% | 54 | 54 | +0.84% |

## How good is the data

- The fund's own reported return (which includes dividends and fees) differs from the return of its frozen holdings
  by +0.11 points a quarter on average (median -0.05). The two are built from different numbers, so
  this is an independent check that the prices derived from the filings are right.
- Between 0 and 9 index companies per quarter have no usable return (they left both index funds, or their
  share count changed in a way no split explains). They are left out for that quarter, never given a made-up return.
- 91% to 97% of the fund's stock holdings are in the index. The rest are outside the pool the random portfolios
  are drawn from, and the result says so each quarter.
- Every random portfolio is re-checked against the rules after it is drawn: 0 violations.

## The same quarters through the quantum estimator

The quantum core (`QUANTUM_CORE.md`) was given exactly the same inputs: the universe, the rules, the fund's return.
It answers the equal-weight version of the question, the one its circuit can express. Everything in this section is a
noiseless **simulation** and counts oracle queries, not time. One query is one check of one portfolio.

- **It gets the same answer.** On the 16 largest index stocks, where every portfolio can be listed and the
  exact rank is known, the simulated quantum estimate was off by 0.55 points on average over 27 quarters;
  classical sampling with the same number of queries was off by 0.79.
- **It needs fewer queries, and the gap widens with precision.** To pin the fund's rank to ±1 point on the full
  universe: a median of 1,900 quantum queries against 5,366 classical samples
  (2.4× fewer). For the latest quarter:

| Precision (points) | Quantum queries | Classical samples | Ratio |
|---|---|---|---|
| ±2 | 1,400 | 2,004 | 1.4× |
| ±1 | 2,800 | 8,014 | 2.9× |
| ±0.5 | 3,100 | 32,058 | 10.3× |
| ±0.2 | 10,200 | 200,359 | 19.6× |
| ±0.1 | 40,300 | 801,438 | 19.9× |

- **Under these rules that is the only gain.** Every random draw is already a valid portfolio, so there is no rarity to exploit. A mandate
  with caps and ESG or carbon averages leaves far fewer valid portfolios, and that is where the second gain appears
  (classical cost grows like 1/share, quantum like 1/√share).
- **It cannot run on a machine that exists.** The full-universe circuit needs about 949 error-corrected qubits
  and 5.0e+06 T gates per step. Each quantum query would also be far slower than a laptop's check, so
  this is fewer queries, not less time. What may and may not be claimed: `QUANTUM_CORE.md` section 5.

## Repeat it

The commands are in `README.md` under "Real funds from public filings". Another fund needs its SEC series id and
an index fund that reports on the same months (`FAIRBENCH_REAL_DATA_RESEARCH_MEMO.md` lists candidates).

## Checks against known answers

**A made-up example with planted answers** (`examples/real_data_example/`: a fictional 60-company
index fund and a fictional 15-stock ESG fund over 5 quarters, in the same file format as the
real filings). A 2-for-1 share split, a company leaving the index and an exclusion list were planted. The pipeline
recovered the planted returns to within 4e-17, adjusted the split
(1 adjustment), left the departed company without a return instead of
inventing one, and found 0 rule violations in its random portfolios.

**Samplers against exact answers.** On universes small enough to list every valid portfolio, each sampler's
output is compared with the exact distribution. "tv" is the distance from it; a value near "tv noise floor" is
sampling noise, a value well above is bias. "kl" is blank when some valid portfolio was never drawn.

| distribution | sampler | kind | feasibility rate | tv | tv noise floor | kl | marginal error max | percentile error | qubits | depth |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 uniform subsets, equal weights | rejection | classical | 0.8003 | 0.05196 | 0.05405 | 0.008793 | 0.004908 | 0.1639 |  |  |
| D1 uniform subsets, equal weights | swap chain (thin 5) | classical | 1 | 0.05629 | 0.05405 | 0.01001 | 0.008748 | 0.2089 |  |  |
| D1 uniform subsets, equal weights | Dicke circuit + filter (simulator) | quantum | 0.774 | 0.05483 | 0.05405 | 0.009109 | 0.008852 | 0.03641 | 11 | 695 |
| D3 benchmark-aware, tau = median TE | rejection + importance reweighting | classical | 0.8003 | 0.0517 | 0.05588 | 0.008751 | 0.005777 | 0.1274 |  |  |
| D1 uniform subsets, benchmark weights | rejection | classical | 0.8003 | 0.05196 | 0.05405 | 0.008793 | 0.004908 | -0.04359 |  |  |
| D2 uniform weight grid | rejection | classical | 0.6036 | 0.01523 | 0.0165 | 0.0007545 | 0.004757 | -0.5614 |  |  |
| D2 uniform weight grid | simulated annealing on the QUBO + filter | classical | 0.8942 | 0.03539 | 0.01743 | 0.003978 | 0.0142 | -0.6094 |  |  |
| D2 uniform weight grid | amplitude amplification on the QUBO + filter (simulator) | quantum | 0.9993 | 0.01879 | 0.01657 | 0.0009972 | 0.009029 | 0.7471 | 15 | 3,127,186 |

Rejection sampling, the Dicke circuit and amplitude amplification sit at the noise floor: they draw the stated
distribution. Simulated annealing does not (0.03539 against a floor of 0.01743). The quantum rows ran on a
simulator on tiny cases, and the amplification circuit's depth comes from an oracle built from the listed answer:
this table shows the outputs are correct, not that anything is faster.

## What is observed, derived, assumed, missing

| | |
|---|---|
| **Disclosed** (read from a filing) | Holdings: company, identifiers, shares, value, share of the fund. Fund size. Monthly returns and flows. Filing and report dates. |
| **Derived** (computed, method stated) | Prices and price returns (value / shares at two dates). Split adjustments. Benchmark weights (from the index fund's filing). The fund's frozen-holdings return. Position cap and weight-grid bounds (from the fund's own positions). The covariance used for the benchmark-aware reference. Observed holding changes. |
| **Assumed** (a choice, reported with the result) | Which reference distribution and weighting. The exclusion list from industry codes. Returns kept when only the share count changed. Dropping securities with no return. An index fund standing in for the index. |
| **Missing** | Dividends. Returns of securities that left both index funds. Sector labels. Trades inside a quarter. |
| **Not reproducible from public data** | Any rule that depends on a vendor ESG rating, a provider's emissions data or a manager's internal score. Stored with its evidence, marked unobservable, never enforced. |
