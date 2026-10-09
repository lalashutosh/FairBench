# Reproducibility report: real-data layer

_Generated 2026-10-09 by `scripts/reproducibility_report.py` from commit `7ed0ca5`
(branch `claude/fairbench-research-plan-368b64`). Do not edit by hand; re-run the script._

## What was and was not run

| | Status |
|---|---|
| Synthetic worked example, end to end (parse, store, universe, returns, references, ranking) | **Run**, results below |
| Samplers against exact distributions on small universes, classical and quantum (simulator) | **Run**, results below |
| Live download from SEC EDGAR | **Not run.** The client needs `FAIRBENCH_SEC_USER_AGENT`; nobody has set it. The code path is tested offline against a fake transport. |
| Any real fund | **Not run.** No real holdings have been ingested. Every number here is synthetic. |
| Live model call for mandate extraction | **Not run.** Tested with a fake client only. |
| Quantum hardware | **Not run.** Simulator only. |

The example data are fictional and labelled so in every file. They were generated with
known values planted, so the pipeline's output can be checked against the truth.

## Environment

Python 3.11.9 on Darwin arm64;
numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, qiskit 2.5.2,
qiskit-aer 0.17.2.

## Commands

```bash
.venv/bin/python scripts/build_example_dataset.py
```

```bash
.venv/bin/python scripts/real_fund_attribution.py --example
```

```bash
.venv/bin/python scripts/validate_reference.py
```

```bash
.venv/bin/python scripts/reproducibility_report.py
```

Seeds: example data 0; reference samples 0 (5,000 portfolios per
distribution per period); validation 0. Inputs: 12 synthetic N-PORT documents under
`examples/real_data_example/nport/`, SHA-256 of each in `meta.json` and checked by this script.

## Worked example (synthetic)

A fictional ESG fund with 15 holdings against a fictional 60-company
index fund, 5 quarterly periods, 6 companies on an exclusion list.
Returns are price returns derived from the index fund's own filings (value / shares).

| period end | universe | unknown returns | splits adjusted | frozen return (derived) | frozen return (planted) | pct D1 equal | pct D1 bench | pct D2 | pct D3 benchmark-aware 0.25 x fund TE | pct D3 benchmark-aware 0.5 x fund TE | pct D3 benchmark-aware 1 x fund TE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2024-06-30 | 60 | 0 | 0 | -3.43% | -3.43% | 99.3 | 99.2 | 98.3 |  |  |  |
| 2024-09-30 | 60 | 0 | 0 | +2.08% | +2.08% | 62.1 | 56.0 | 60.2 |  |  |  |
| 2024-12-31 | 60 | 0 | 1 | +1.40% | +1.40% | 82.0 | 72.2 | 77.9 |  |  |  |
| 2025-03-31 | 59 | 1 | 0 | -2.34% | -2.34% | 27.7 | 26.8 | 29.5 |  |  |  |
| 2025-06-30 | 59 | 0 | 0 | +4.06% | +4.06% | 80.0 | 86.3 | 75.8 | 88.3 | 86.2 | 86.2 |

- Largest gap between the derived and the planted frozen-holdings return: 3.82e-17. Both are computed from
  the same synthetic holdings, so this checks the arithmetic and the handling of the split and the index
  deletion (either would move the number if mishandled). It says nothing about the quality of real data.
- The planted 2-for-1 split was adjusted in the period it happened (1 adjustment in total);
  the company that left the index has no derived return afterwards and was dropped from the universe for that
  period (1 unknown return in total), not filled in.
- Rule violations found by the independent re-check across all reference samples: 0.
- "pct" columns are the realised portfolio percentile under each reference distribution. They differ because the
  distributions are different objects (see `fairbench/portfolio/reference.py`). The benchmark-aware reference (D3)
  appears only once four earlier periods exist to estimate a covariance from, so that nothing from the future is used.
- A percentile is not evidence of skill. Here the fund's holdings were chosen by a random preference vector.

Per-period definitions, samplers, seeds, acceptance rates and estimated numbers of feasible portfolios are in
`results/real_fund_example.json`.

## Samplers against exact distributions

Small universes where every feasible portfolio can be listed. "tv" is total variation distance to the exact
distribution; a value near "tv noise floor" is sampling noise, a value well above it is bias. "kl" is blank when
it is infinite (some feasible portfolio was never drawn).

| distribution | sampler | kind | n samples | feasibility rate | tv | tv noise floor | kl | kl smoothed | marginal error max | return dist ks | percentile error | runtime s | qubits | depth | shots |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 uniform subsets, equal weights | rejection | classical | 20,000 | 0.8003 | 0.05196 | 0.05405 | 0.008793 | 0.008628 | 0.004908 | 0.006064 | 0.1639 | 0.03 |  |  |  |
| D1 uniform subsets, equal weights | swap chain (thin 5) | classical | 20,000 | 1 | 0.05629 | 0.05405 | 0.01001 | 0.00982 | 0.008748 | 0.006466 | 0.2089 | 1.051 |  |  |  |
| D1 uniform subsets, equal weights | Dicke circuit + filter (simulator) | quantum | 20,000 | 0.774 | 0.05483 | 0.05405 | 0.009109 | 0.008941 | 0.008852 | 0.004296 | 0.03641 | 0.336 | 11 | 695 | 25,841 |
| D3 benchmark-aware, tau = median TE | rejection + importance reweighting | classical | 20,000 | 0.8003 | 0.0517 | 0.05588 | 0.008751 | 0.008574 | 0.005777 | 0.006421 | 0.1274 | 0.019 |  |  |  |
| D1 uniform subsets, benchmark weights | rejection | classical | 20,000 | 0.8003 | 0.05196 | 0.05405 | 0.008793 | 0.008628 | 0.004908 | 0.006381 | -0.04359 | 0.03 |  |  |  |
| D2 uniform weight grid | rejection | classical | 20,000 | 0.6036 | 0.01523 | 0.0165 | 0.0007545 | 0.0007532 | 0.004757 | 0.006443 | -0.5614 | 0.023 |  |  |  |
| D2 uniform weight grid | simulated annealing on the QUBO + filter | classical | 17,885 | 0.8942 | 0.03539 | 0.01743 | 0.003978 | 0.00397 | 0.0142 | 0.01264 | -0.6094 | 1.028 |  |  |  |
| D2 uniform weight grid | amplitude amplification on the QUBO + filter (simulator) | quantum | 19,986 | 0.9993 | 0.01879 | 0.01657 | 0.0009972 | 0.0009954 | 0.009029 | 0.009735 | 0.7471 | 18.98 | 15 | 3,127,186 | 20,000 |

Reading it:
- Rejection sampling, the Dicke circuit and amplitude amplification sit at the noise floor: they draw the stated
  distribution. Simulated annealing on the same encoding does not (0.03539 against a floor of 0.01743); it is a baseline for the energy function, not a uniform sampler.
- The circuit depth of the amplification row comes from synthesising an oracle from the enumerated answer. It
  shows the output distribution is right; it is not a cost estimate, and nothing here is a claim of quantum advantage.
- Runtimes are wall-clock on this machine for these tiny instances and are not a speed comparison.

## What is observed, derived, assumed, missing

| Label | In this layer |
|---|---|
| **Disclosed** (read from a filing as filed) | Holdings: issuer, CUSIP, ISIN, shares, value, percent of net assets, asset type, country, currency. Fund net assets. Monthly total returns per share class. Monthly flows. Filing and report dates. |
| **Derived** (computed, method stated) | Security keys from identifiers. Equity-sleeve weights. Benchmark weights (from the index fund's filing). Security price returns (value / shares at two dates). Split adjustments. The fund's frozen-holdings return. Position cap and weight-grid bounds (from the fund's own largest and smallest positions). The single-index covariance. Observed holding changes. |
| **Assumed** (a choice, reported with the result) | The reference distribution itself and its weighting policy. The tilt scale tau in the benchmark-aware reference. Dropping, or filling, securities with unknown returns. An index fund as proxy for the index. Any exclusion list not printed in the mandate. Promotion of a soft rule to hard by a reviewer. |
| **Missing** | Dividends (price returns only, unless a total-return source is supplied). Returns of securities that left the reference fund within a period. Share changes under 8% that are really stock dividends. Sector classification. Trades inside a quarter. |
| **Not reproducible from public data** | Any rule that depends on a vendor ESG rating, a vendor controversy flag, a provider's emissions data or a manager's internal score. Such rules are stored with their evidence, marked unobservable, and never enforced. |

Holdings changes between snapshots are recorded as observations (`observed_position_change`), never as decisions:
they mix trading, price moves, investor flows and corporate actions, and neither the timing nor the reason is public.
