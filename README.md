# FairBench

When an ESG fund underperforms, is it the manager or the rules? FairBench answers that with
random-portfolio benchmarking: it samples many portfolios that obey the *same* rules as the
fund, builds the distribution of their returns, and ranks the real fund in it.

Built by Team #22 for the Hanken Quantum x Finance Hackathon (8–10 October 2026).

![Attribution demo: the fund against random portfolios that follow its rules](results/attribution_demo.png)

## What it does

```
fund - benchmark = (same-rules median - benchmark)   constraint effect: what the rules cost
                 + (fund - same-rules median)        manager effect: selection within the rules
```

- **Rules** are a `ConstraintSet`: number of holdings, sector caps, exclusions, an ESG floor, a carbon cap.
- **The sampler is pluggable.** Classical rejection sampling is the default. A constraint-preserving
  quantum circuit (Dicke state + XY mixer) that only outputs portfolios with exactly k holdings
  drops in as an alternative backend.
- **The output** is the constraint/manager split, the fund's percentile among rule-abiding
  portfolios, Monte Carlo confidence intervals, and a plot.

## Status, stated plainly

- **All attribution results so far use synthetic data** with a planted ground truth. No real fund has been analysed.
- **No quantum speedup was found.** Three routes were benchmarked against strong classical baselines:

| Route | Result |
|---|---|
| Dicke state + filter | Same distribution as classical rejection sampling. A correctness baseline, not an advantage. |
| Trained layers | Acceptance 0.24 → 0.32 on a toy case, at the cost of non-uniform samples; negligible gain on the hard "island" case. |
| Quantum-enhanced MCMC | Exact spectral gaps at n ≤ 16: no evidence of advantage. A classical walk given the same feasibility oracle does at least as well. |
| Amplitude amplification | Fault-tolerant resource estimate: ~10⁴× slower than classical at n = 100–200. Break-even needs a feasible fraction below ~1e-9 to 1e-12; our synthetic rule family sits at 0.01–0.2. |

- Quantum circuits run on simulators only (`aer_statevector`, `aer_mps`). The hardware backends are stubs.

Numbers, protocols and review-safe wording are in [`fairbench_handoff.md`](fairbench_handoff.md)
and [`BUILD_PLAN.md`](BUILD_PLAN.md).

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lalashutosh/FairBench.git && cd FairBench
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
```

Run the attribution demo (about 40 s, 0.3 GB of memory):

```bash
.venv/bin/python scripts/demo_attribution.py
```

It prints the attribution for a synthetic 100-asset case, then repeats a 16-asset case with
classical rejection, the Dicke circuit and exact enumeration to show they agree. Outputs go to
`results/attribution_demo.{png,json}` and `results/attribution_samplers.csv`.

Run the tests (255, about 35 s):

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q
```

## Using it on your own data

```python
from fairbench.apps.attribution import attribute, dicke_sampler, plot_attribution
from fairbench.constraints import Cardinality, ConstraintSet, Exclusion, MinESG, SectorCap
from fairbench.data import load_universe

# universe.csv: ticker, mu, sector, esg_score, carbon      (one row per asset)
# returns.csv:  first column = date, then one column of simple returns per ticker
u = load_universe("universe.csv", returns_path="returns.csv")

rules = ConstraintSet([Cardinality(30), SectorCap("Energy", 2), MinESG(60.0), Exclusion([4, 17])])

res = attribute(u, rules, fund_holdings, n_samples=5000)           # classical rejection sampling
# res = attribute(u, rules, fund_holdings, sampler=dicke_sampler())  # Dicke circuit (small n, simulator)
print(res.summary())
plot_attribution(res, "attribution.png")
```

`fund_holdings` is a 0/1 vector over the universe or explicit weights that sum to 1. Optional
arguments: `fund_returns` (the fund's realised return series), `benchmark_returns` (an index
instead of the default benchmark), `metric="sharpe"`, `weight_scheme`.

Things to keep in mind when reading a result:

- The default benchmark is the median random portfolio with the same number of holdings and no other rule.
- Holdings are held fixed over the window; there is no turnover model.
- The confidence intervals cover Monte Carlo sampling error only.
- A percentile from one window is not evidence of skill.
- A fund whose holdings break the rule set is flagged, because the null is then not its null.

## Layout

```
fairbench/
  data.py, constraints.py, instances.py     universe, rule logic, benchmark instances
  baselines.py, chains.py, metrics.py       classical samplers, MCMC chains, sampler metrics
  backends.py, training.py, objectives.py   circuit sampling backends and ansatz training
  postprocess.py                            filtering, repair, portfolio weights
  quantum/                                  Dicke state, XY mixer ansatz, Hamiltonians, MCMC proposal
  ft/                                       fault-tolerant oracle and resource model (amplitude amplification)
  apps/attribution.py                       the attribution tool
scripts/                                    demo and the benchmark studies
results/                                    CSV, JSON and plots produced by the scripts
tests/                                      pytest suite
```

## Memory note

Some study scripts are heavy. Cap threads, and for the largest ones cap memory too:

```bash
ulimit -v 4000000; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 1500 .venv/bin/python -u scripts/aa_estimate.py
```

Do not use `ulimit -v` with the test suite: Qiskit Aer reserves virtual memory and fails.
