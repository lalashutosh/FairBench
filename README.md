# FairBench

**Is it the manager, or the mandate?** FairBench is rules-adjusted performance benchmarking for
ESG funds. It ranks a real fund against thousands of random portfolios that obey the *same*
investment rules, so the cost of the rules and the choices made inside them are reported separately.
That rank is a fraction of a set of portfolios, which is exactly what quantum amplitude estimation
computes: one pipeline feeds a classical estimator that runs today and a quantum estimator, validated
in simulation and run on a 50-qubit quantum computer.

Team #22 · Hanken Quantum x Finance Hackathon · 8–10 October 2026 · Python 3.11, Qiskit

![Quarterly rank of nine real US funds among random portfolios drawn from each fund's own universe. The eight ESG funds never land in the extreme tails (0 of 198 fund-quarters). The control, an equal-weighted index fund that really is built differently from its universe, lands there in 13 of 26 quarters.](results/skill_table/skill_validation.png)

## At a glance

| | |
|---|---|
| **Problem** | An ESG fund trails its index. Was it the manager's picks, or the rules the fund must follow? Index benchmarks, peer groups and Brinson attribution cannot separate the two. |
| **Solution** | Sample the portfolios the fund's own rules allow, then report where the real fund ranks among them (0–100; 50 is typical) and what the rules cost against the market. |
| **Real data** | Nine real US funds, 224 fund-quarters, 2019–2026, built only from free SEC filings. |
| **Finding** | All eight ESG funds rank inside the range luck alone would give; the control fund is flagged in 13 of 26 quarters. For Parnassus Core Equity, the 14-point gap to the S&P 500 comes mostly from holding 36 stocks in a market led by a few giants; its exclusions cost about 2 points. |
| **Quantum, in theory** | Amplitude estimation over a constraint-preserving circuit needs quadratically fewer queries: 12× fewer than classical sampling on the tightest rules (noiseless simulation). |
| **Quantum, in practice** | The circuit ran on VTT's 50-qubit Q50 through LUMI: valid portfolios 54% of the time against 37.5% by chance. A digital twin fitted to that data says what hardware the advantage needs. |
| **Demo** | One command, offline, a few seconds: `scripts/demo_pipeline.py`. |
| **Engineering** | 1,184 offline tests. Every number here is written by a script in `scripts/` to a file in `results/`. |
| **Honest scope** | No quantum advantage on today's hardware, and none is claimed. A rank is a description, not proof of skill. |

## The problem

An ESG fund must follow rules: no tobacco, no fossil fuels, a carbon cap, a fixed number of holdings.
When it lags the market, its board, its investors and its regulator ask the same question, and
today's tools answer a different one:

- **An index benchmark** says the fund lost to the S&P 500. The fund was never allowed to hold the S&P 500.
- **A peer group** compares it with other funds, which follow other rules.
- **Brinson and factor attribution** explain the gap by sector, stock or factor. They do not price the rules.

So a manager can be blamed for a mandate, or hide behind one.

## The solution

Draw many portfolios the fund could have held under its own rules. Their returns show the range of
outcomes the rules allowed. Then place the real fund in that range.

```
fund − benchmark = (same-rules median − benchmark)   what the rules cost
                 + (fund − same-rules median)        the choices made inside the rules
```

```
1 READ       the fund's policy text ─▶ rules, each tied to the sentence it came from   (Claude + checks)
2 CHECK      add what the fund's region requires (US names rule, EU fund-name rules)
3 FORMULAS   every rule as mathematics on the holdings, reviewed by a person
4 DATA       SEC filings ─▶ universe, benchmark weights, prices, the fund's portfolio
5 ESTIMATE   "among portfolios the rules allow, where does the fund's return rank?"
             ├─ classical   random rule-abiding portfolios, counted          today's product
             └─ quantum     the same question as one amplitude               simulated; fault-tolerant era
6 REPORT     rank per quarter, its error bar, what the rules cost, the checks
```

## What is new

| Approach | Question it answers | What it leaves open |
|---|---|---|
| Index benchmark | Did the fund beat the market? | Whether the fund was allowed to hold what drove the market |
| Peer group | Did it beat similar funds? | Peers follow different rules |
| Brinson / factor attribution | Which sectors, stocks or factors explain the gap? | What the rules themselves cost |
| Quantum portfolio optimisation | Which single portfolio is best? | No ranking; no proven speedup |
| **FairBench** | Among all portfolios the rules allow, where does this fund rank, and what did the rules cost? | Causality and skill, which no single ranking can establish |

1. **The reference is the fund's own rule set, read from its own documents.** Every rule keeps the
   sentence it came from. Rules that cannot be expressed or observed are listed, never dropped.
2. **It runs on free public data.** Holdings, the universe, benchmark weights and prices all come from
   SEC N-PORT filings. No licensed index, price or ESG feed is needed to reproduce a result.
3. **Quantum is used for estimation, not optimisation.** The fund's rank is a fraction of a set, and for
   that amplitude estimation has a proven quadratic query advantage. The fund's return goes inside the
   oracle as a threshold, so the rank comes out directly.
4. **Every claim is checked against a known answer:** planted effects, exact enumeration, a control
   fund, a real quantum computer. The negative results are published with the positive ones.

The longer argument: [`docs/WHY_FAIRBENCH.md`](docs/WHY_FAIRBENCH.md).

## Results

### 1. Nine real funds, 2019–2026

Each quarter a fund's disclosed portfolio is ranked among 20,000 random portfolios of the same size
from its own universe. Luck alone gives an average rank of 50; the luck band is where 95% of such
averages fall. Extreme quarters are ranks at or below 2.5 or at or above 97.5.

| Fund | Group | Quarters | Average rank | Luck band | Extreme quarters |
|---|---|---|---|---|---|
| Parnassus Core Equity | active ESG | 27 | 50.4 | 39.1–60.9 | 0 |
| Calvert Equity | active ESG | 27 | 46.1 | 39.1–60.9 | 0 |
| Nuveen Large Cap Responsible | active ESG | 26 | 46.4 | 38.9–61.1 | 0 |
| American Century Large Cap Equity | active ESG | 26 | 50.6 | 38.9–61.1 | 0 |
| SPDR S&P 500 ESG ETF | passive ESG | 23 | 54.3 | 38.2–61.8 | 0 |
| Xtrackers S&P 500 Scored & Screened | passive ESG | 26 | 56.4 | 38.9–61.1 | 0 |
| iShares ESG Aware MSCI USA | passive ESG | 26 | 52.2 | 38.9–61.1 | 0 |
| iShares Paris-Aligned Climate MSCI USA | passive ESG | 17 | 47.8 | 36.3–63.7 | 0 |
| iShares MSCI USA Equal Weighted | control, not ESG | 26 | 38.7 | 38.9–61.1 | **13** |

- **The method detects a fund that really differs.** The control is an equal-weighted index fund
  measured against a value-weighted reference. It is flagged in 13 of 26 quarters.
- **The ESG funds do not differ from what their universe allowed:** 0 extreme quarters in 198, and all
  eight averages inside the band. There is neither a selection penalty nor selection skill to report.

The rules here are the universe and the number of holdings; price returns, holdings frozen for the
quarter. Source: `scripts/skill_table.py`, `results/skill_table/`.

### 2. One fund in depth: Parnassus Core Equity

![Parnassus Core Equity against the S&P 500 index fund and the typical rule-abiding portfolio, 2019 to 2026.](results/real_fund_parnassus_core_equity.png)

| | Price growth, 2019-09 to 2026-06 |
|---|---|
| iShares Core S&P 500 ETF (the market) | +149% |
| Parnassus Core Equity | +135% |
| Typical 36-stock portfolio under the same rules | +105% |

- **The exclusions cost about 2 points** (27 fossil-fuel, tobacco, alcohol and weapons companies).
- **The gap to the index is mostly structural:** any 36-stock portfolio tends to trail an index a few
  very large companies drove. The fund finished 30 points ahead of the typical portfolio its rules allowed.
- **Quarter by quarter it ranked 54 of 100 on average** (luck: 50, give or take 6). Not distinguishable from luck.

This run uses a proxy mandate with capped weights, which is why its average (54) differs from the
table above (50.4). Details and checks: [`docs/REAL_FUND_RESULTS.md`](docs/REAL_FUND_RESULTS.md).

### 3. The quantum estimator, in simulation

![Queries needed for the whole attribution as rules tighten. The hybrid classical and quantum estimator needs 12 times fewer queries than classical sampling at the tightest rules; classical sampling is 1.5 times better on loose rules.](results/qae_hybrid_pitch.png)

Amplitude estimation over a Dicke state (every portfolio with exactly k holdings, in superposition)
measures the rank with error falling as 1/queries, against 1/√queries for sampling. Noiseless
simulation; query counts, not time.

| | Quantum | Classical |
|---|---|---|
| Error against queries, fitted slope | −1.00 | −0.48 |
| Whole attribution, tightest rules (0.23% of portfolios allowed) | **12× fewer queries** | |
| Whole attribution, loose rules (36% allowed) | | classical wins, 1.5× |
| Against swap-MCMC | 2–9× fewer | chains get trapped on tight rules |
| Real fund's 27 quarters, rank to ±1 point | median 1,900 queries | median 5,366 samples (median ratio 2.4×) |

The gain grows with precision: at a loose ±2 points the two are about level, at ±0.1 point the
quantum estimator needs about 20× fewer queries. The full-universe circuit needs about 949
error-corrected qubits: a fault-tolerant workload.

### 4. On a real quantum computer: VTT Q50

![Left: share of shots that are valid portfolios on VTT Q50, measured against random bitstrings, for four circuit sizes. Right: the amplitude-estimation signal against Grover steps, which decays to the random-output level after one step.](results/hw/hw_q50.png)

| Measurement, 2026-10-09, 2,000 shots per circuit | Result |
|---|---|
| Bell pair / 4-qubit GHZ | 95.0% / 84.5% |
| Valid portfolios, 4 assets hold 2 (46 two-qubit gates) | **54%** against 37.5% by chance |
| Valid portfolios, 8 assets hold 3 (239 two-qubit gates) | 28% against 22% by chance |
| Toy amplitude estimation of a fund's rank (exact 33.3%) | 27%: the right direction, not accurate |
| Digital twin, one fitted parameter (two-qubit error 1.8%) | reproduces the hardware data |
| Gate error the toy needs / real portfolio sizes need | about 10⁻³ / about 10⁻⁷ |

The circuit runs and beats chance. Depth is the wall, and the measured noise turns "needs better
hardware" into a number. Quantum method, studies and the full hardware run:
[`docs/QUANTUM_CORE.md`](docs/QUANTUM_CORE.md).

### 5. What we tested and ruled out

Four ways of using a quantum computer to *generate* rule-abiding portfolios were benchmarked first. None helps.

| Route | Result |
|---|---|
| Dicke state + filter | The same distribution as classical rejection sampling |
| Trained circuit layers | Slightly more acceptance, paid for with biased samples |
| Quantum-enhanced MCMC | A classical walk given the same information does at least as well |
| Amplitude amplification | About 10⁴× slower than classical under fault-tolerant cost assumptions |

A generated sample is one random draw, so it can never improve precision. An estimate can. That is why
the quantum core estimates the rank instead of generating portfolios.

### 6. From policy text to rules

On a fictional 18-rule mandate, 14 rules are enforced, 1 holds trivially and 3 are reported as not
expressible, each with its source sentence. The model translates; deterministic code computes every
threshold. Tested offline and against a hand-written reference; no live API call has been made, and
the real-fund results above do not depend on it.

## Run it

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lalashutosh/FairBench.git && cd FairBench
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
.venv/bin/python scripts/demo_pipeline.py          # the whole pipeline, offline, a few seconds
OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q    # 1,184 tests, a few minutes
```

| More demos | Command |
|---|---|
| Attribution on synthetic data with a planted answer | `.venv/bin/python scripts/demo_attribution.py` |
| Policy text to constraints | `.venv/bin/python scripts/demo_mandate.py` |
| Quantum estimator, loose against tight rules | `.venv/bin/python scripts/demo_qae.py` |
| Rebuild the nine-fund table | `.venv/bin/python scripts/skill_table.py` |

### Real funds from public filings

```bash
export FAIRBENCH_SEC_USER_AGENT="Your Project Name contact@your-domain.org"   # the SEC requires a contact
.venv/bin/python scripts/real_fund_ingest.py --series S000000856 --series S000004310
.venv/bin/python scripts/real_fund_ingest.py --series S000004347
.venv/bin/python scripts/build_sic_exclusions.py --parent-dir data/nport/S000004310

ARGS="--fund-dir data/nport/S000000856 --parent-dir data/nport/S000004310 --price-dir data/nport/S000004347 --exclude-file data/exclusions_sic.csv --name parnassus_core_equity"
.venv/bin/python scripts/real_fund_attribution.py $ARGS      # classical: the rank per quarter
.venv/bin/python scripts/real_fund_quantum.py $ARGS          # the same quarters through the quantum estimator
```

`S000000856` is Parnassus Core Equity, `S000004310` iShares Core S&P 500 (its universe), `S000004347`
iShares Russell 1000 (prices for companies that left the index). Without the network, try
`scripts/build_example_dataset.py` then `scripts/real_fund_attribution.py --example`.

### On your own data

```python
from fairbench.apps.attribution import attribute
from fairbench.constraints import Cardinality, ConstraintSet, Exclusion, MinESG, SectorCap
from fairbench.data import load_universe

u = load_universe("universe.csv", returns_path="returns.csv")
rules = ConstraintSet([Cardinality(30), SectorCap("Energy", 2), MinESG(60.0), Exclusion([4, 17])])
print(attribute(u, rules, fund_holdings, n_samples=5000).summary())
```

## Limits, stated plainly

- **A rank is a description, not proof of skill.** Over 27 quarters only an average above about 61 or
  below about 39 could be told apart from luck.
- **Proxy rules on real funds.** A fund's own ESG research is private; rules that need a vendor's ESG
  or emissions data are recorded, marked `unobservable`, and never enforced.
- **Price returns from filings,** so no dividends; holdings are quarter-end snapshots.
- **Quantum.** The query advantage is a noiseless-simulation result. The hardware run shows feasibility
  and measures the gap; it shows no advantage. Fewer queries is not less time on any machine today.

## Who it is for, and what comes next

An asset owner, fund-of-funds or consultant that reviews external equity managers every quarter
against mandates with ESG, carbon or sector rules. Today that review is a spreadsheet. FairBench makes
it a repeatable report: what the rules cost, where the manager ranked inside them, which rules could
not be checked. It sits beside the benchmark and attribution systems an institution already has.

The classical pipeline is the product today. The quantum core is a drop-in estimator for the same
numbers, ready for fault-tolerant hardware, with its resource needs already counted.

**Next:** more funds and quarters; a licensed ESG source; real mandates read from prospectuses; a pilot
on one institution's own mandate and holdings.

## Hackathon journey and resources

| When | What was built |
|---|---|
| Before the event, 5–7 October | Sampling engine, ten constraint types, Dicke-state circuits, classical baselines, the quantum sampling studies that found no advantage, the attribution app, the AI mandate reader. |
| During the event, 8–10 October | Amplitude estimation of the rank and its cost studies; the SEC data pipeline; the first real fund, then nine; the run on VTT Q50 and its digital twin; the one-command demo. |

**Resources:** Qiskit and Qiskit Aer; VTT Q50 through LUMI and FiQCI; SEC EDGAR N-PORT filings; NumPy,
pandas, SciPy, matplotlib, pytest; the Claude API for the mandate reader and Claude Code as a coding
assistant.

## Repository

```
README.md       this page
docs/           the longer write-ups (below)
fairbench/      the package: constraints, samplers, quantum/, qae/, ft/, mandates/, ingest/, apps/
scripts/        demos, studies, the hardware run
results/        every CSV, JSON and figure the scripts wrote
examples/       a fictional mandate and a synthetic filing set
tests/          1,184 offline tests
```

| In `docs/` | What is in it |
|---|---|
| [`WHY_FAIRBENCH.md`](docs/WHY_FAIRBENCH.md) | The argument in plain words: the problem, why existing tools miss it, why quantum fits |
| [`REAL_FUND_RESULTS.md`](docs/REAL_FUND_RESULTS.md) | Parnassus Core Equity quarter by quarter, with data checks |
| [`QUANTUM_CORE.md`](docs/QUANTUM_CORE.md) | The quantum estimator, its studies, and the VTT Q50 hardware run |
| [`RESEARCH.md`](docs/RESEARCH.md) | Background: market and standards, SEC data sources, fund identifiers |
| [`handoff.md`](docs/handoff.md) | The team's working notes |
