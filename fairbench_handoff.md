# FairBench — handoff

_Last updated 2026-10-07 (end of the P2 session). Repo: `https://github.com/lalashutosh/FairBench.git`. Detailed wave-by-wave log and numbers: [`BUILD_PLAN.md`](BUILD_PLAN.md)._

## Start here (next session)

**State of the repo.** The P2 work (attribution app, demo, this handoff) is on branch **`p2-attribution`**, pushed to GitHub, one commit ahead of `main`. It is not merged into `main` yet; merge it (fast-forward) before starting new work, or keep working on the branch.

**What the last session did (2026-10-07).**
- Built `fairbench/apps/attribution.py`, `tests/test_attribution.py` (17 tests), `scripts/demo_attribution.py`; added returns loading and synthetic returns to `fairbench/data.py`. 255 tests green.
- Ran the demo and saved `results/attribution_demo.{png,json}` and `results/attribution_samplers.csv`.
- Drafted the pitch deck (11 slides with speaker notes): https://claude.ai/artifact/Do4km8NeU9qw4oZvxEniJK. It lives only in that artifact, not in the repo, and is private until shared from its Share menu. Its layout has **not** been checked visually.

**Do next, in this order.**
1. Get real fund data (see "Next tasks"). This is the one thing the pitch is missing; everything shown so far is synthetic.
2. Review the deck slide by slide, fill `[team member names]` / `[contact]`, cut to the time limit.
3. Only then the optional items (optimizer pivot, hardware run).

**Decisions made in the last session that the team has not confirmed.**
- Benchmark = median random portfolio with the same k and no other rule (an index can be passed instead via `benchmark_returns`).
- Demo scenario = `scaled_family(100, k=20, esg_q=0.6, carbon_q=0.3)` plus an exclusion list of the 10 most carbon-intensive names, a planted +8%/yr rally per s.d. of carbon intensity, fund planted at rank 70. Chosen so the fund lags the benchmark while ranking well within its rules. Change via the script's flags.
- Fund holdings are static over the window (no turnover model).

**Gaps to know about.**
- There is no file format for a rule set: a real fund's mandate has to be written in Python as a `ConstraintSet`.
- If the demo is re-run with other settings, the numbers on deck slides 4, 5 and 9 and the plot on slide 4 must be updated by hand.
- Deck slides in order: cover, problem, method, result, hard, sampler, routes, breakeven, backend, today, next. Slides 4 and 8 use `results/attribution_demo.png` and `results/aa_breakeven.png`.

**Working agreements.** Commit and push only when asked. Keep every pitch claim to the review-safe wording below. Update this file at the end of each session.

---

## Context

Team #22 "FairBench" at the Hanken Quantum x Finance Hackathon (8–10 Oct 2026).

**Problem (may pivot):** When an ESG fund underperforms, is it the manager or the ESG constraints? The approach is random-portfolio benchmarking: sample many portfolios that satisfy the *same* rules as the fund, build a null distribution of returns, and see where the real fund ranks. Classical constrained sampling can degrade as rule sets grow (MCMC gets trapped in islands; rejection-sampling acceptance falls).

**Quantum contribution (as tested, see findings below):** Constraint-preserving circuits (Dicke state + XY mixer) that only produce portfolios with exactly k holdings, plus penalty or trained layers for the other constraints, plus a fault-tolerant amplitude-amplification (AA) roadmap. We benchmarked all three honestly against strong classical baselines; **none shows a speedup at the scales we could test**. The pitch must use the review-safe wording in this file, not "quadratic speedup".

**Pivot plan:** If problem validation fails, the same engine becomes a CVaR-VQA ESG-constrained portfolio **optimizer**. Only the application layer changes. **Keep everything below `apps/` problem-agnostic.**

---

## Where things stand

| Area | Status |
|---|---|
| Data, constraints, quantum core, backends, baselines, metrics (P0) | Done, tested |
| Training, post-processing, island instances, ablation, MPS sweep (P1) | Done, tested |
| Quantum-enhanced MCMC study (Q) | Done — no quantum gain |
| Amplitude-amplification fault-tolerant resource estimate (AA) | Done — no wall-clock advantage |
| Attribution app `apps/attribution.py` (P2) | Done, tested (synthetic data only) |
| End-to-end demo `scripts/demo_attribution.py` | Done (`results/attribution_demo.{png,json}`, `attribution_samplers.csv`) |
| Pitch deck | First draft: https://claude.ai/artifact/Do4km8NeU9qw4oZvxEniJK (private until shared; team names are placeholders) |
| **Real fund data** | **Not started: every attribution result so far is synthetic** |
| Optimizer pivot `apps/optimizer.py` | Not started (2-line stub) |
| Hardware runs (`ibm` / `vtt` backends) | Stubs that raise `NotImplementedError` |

Test suite: **255 tests, all green** (~35 s).

### Key findings (use this wording in the pitch)

1. **Dicke + filter ≡ classical rejection sampling** in distribution. It is a correctness baseline, not an advantage (P0 toy: acceptance 0.240 vs 0.244).
2. **Trained layers (P1):** on the P0 toy, acceptance rises 0.24 → 0.32, but the distribution moves away from uniform (TV above sampling floor 0.05 vs 0.00). On the island instance the gain is negligible (0.0123 → 0.0138). Training costs ≥ ~400k shot-equivalents, so it needs ~400k feasible samples to pay back. Swap-MCMC is trapped on islands (coverage 0.31 vs 0.85 for rejection).
3. **MPS simulability (P1):** bond dimension needed grows fast with depth (n=20, p=2: max bond 232, 44 s), so the circuit is not trivially classically simulable at depth, but this is not an advantage claim.
4. **Quantum-enhanced MCMC (Q):** exact spectral gaps, n ≤ 16, parameters frozen on seeds 0–4 and reported on held-out seeds. QeMCMC beats penalty-blind chains 9–19× per step **only when given an exact feasibility oracle**, and a classical tilted walk given the same oracle does at least as well (ratio 0.42–0.84). Pitch line: *"We found no evidence of quantum advantage at n ≤ 16 (small instances, a trend not a proof)."*
5. **Amplitude amplification (AA):** we built a verified reversible feasibility oracle (~120–130·n Toffoli) and a full logical cost model. Samples are exactly uniform over the feasible set. Algorithmic logical qubits ≈ 200–400 at n = 100–200 (excludes factories/routing). Quantum takes ~0.3–1 s per sample (optimistic hardware) vs ~30–120 µs classically, i.e. ~10⁴× slower. Break-even needs a feasible fraction below ~1e-9 to 1e-12, but our synthetic constraint family sits at 0.01–0.2. Full pitch-safe paragraph: `BUILD_PLAN.md` → "AA-3 results".

6. **Attribution (P2), synthetic data with planted ground truth:** n = 100, k = 20, three years of Gaussian returns with a planted rally in carbon-heavy names, fund planted at rank 70 of rule-abiding portfolios. Result: fund −4.2 pts vs benchmark = constraint effect −9.4 pts [95% MC CI −9.9, −8.8] + manager effect +5.2 pts; recovered percentile 68.5 ± 0.7. At n = 16 the Dicke circuit (`aer_statevector`), classical rejection and exact enumeration agree (percentile 68.7 / 69.8 / 69.0; KS p = 0.82). **No real fund has been analysed.**

**Honest story for the pitch:** a constraint-preserving quantum sampler plus a careful benchmarking harness. Every speedup route was tested against fair classical baselines, and we report where the break-even would be. The practical tool (attribution) runs on classical rejection sampling today, with the quantum sampler as a drop-in backend.

---

## Getting started

```bash
git clone git@github.com:lalashutosh/FairBench.git && cd FairBench
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q        # 255 passed
.venv/bin/python scripts/p0_demo.py                     # toy benchmark table
.venv/bin/python scripts/demo_attribution.py            # pitch demo: attribution + sampler swap (~40 s, 0.3 GB)
```

### ⚠️ Memory safety (read before running scripts)
The dev laptop has 14 GB RAM. A runaway script once got OOM-killed and took VS Code down with it. For heavy scripts:
```bash
ulimit -v 4000000; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 1500 .venv/bin/python -u scripts/aa_estimate.py
```
Don't use `ulimit -v` with the full test suite: Qiskit Aer reserves virtual memory and fails with `bad_alloc`. Cap threads instead.

---

## Repo map

```
fairbench/
  data.py           Universe dataclass, CSV loaders (universe, returns), synthetic_universe, synthetic_returns
  constraints.py    Cardinality, SectorCap, Exclusion, MinESG, CarbonCap, ConstraintSet (ALL rule logic lives here)
  instances.py      p0_instance, island_instance, island_family(n, seed), scaled_family(n, seed) (n up to 200)
  backends.py       sample(circuit, params, shots, backend="aer_statevector"|"aer_mps"|"ibm"|"vtt")
  baselines.py      enumerate_feasible, random_k_subsets, rejection_sample, mcmc_swap_sample
  metrics.py        acceptance_rate, tv_to_uniform, coverage, cost_per_feasible_sample
  objectives.py     training objectives as callables on samples (uniformity, CVaR, ...)
  training.py       train(ansatz, objective_fn, backend, optimizer="COBYLA"|"SPSA", ...)
  postprocess.py    filter_feasible, repair (flagged: biases sampling), assign_weights
  chains.py         generic MH chains, pluggable proposals, exact transition matrix / spectral gap
  quantum/
    dicke.py        dicke_state(n, k)  (Bärtschi–Eidenbenz)
    ansatz.py       xy_mixer, cost_layer, build_ansatz(n, k, p, cost_op, topology)
    hamiltonian.py  penalty_operator, objective_operator (mean-variance QUBO for the optimizer pivot)
    proposal.py     quantum-enhanced MCMC proposal in the weight-k subspace
  ft/
    oracle.py       reversible feasibility oracle + integer quantisation of ESG/carbon rules
    revsim.py       fast classical simulator for reversible circuits
    resources.py    logical FT cost model: Dicke + oracle + reflection, known/BBHT/fixed-point schedules
  apps/
    attribution.py  attribute() -> AttributionResult, rejection_sampler, dicke_sampler, plot_attribution
    optimizer.py    P2 — EMPTY STUB (pivot)
scripts/
  demo_attribution.py  pitch demo: planted synthetic case, attribution, rejection vs Dicke vs exact
  p0_demo.py        toy benchmark: Dicke vs rejection vs MCMC
  p1_ablation.py    sampler ablation, identical post-processing (use --no-repair for speed)
  p1_mps_sweep.py   MPS bond-dimension sweep
  q_gap_sweep2.py   QeMCMC vs classical chains, exact spectral gaps (pre-registered protocol)
  aa_pf_scaling.py  feasible-fraction scaling n = 16..200
  aa_estimate.py    AA wall-clock vs classical, break-even plot
results/            CSV/PNG/JSON outputs of all scripts (plots ready for slides: attribution_demo.png,
                    ablation.png, mps_sweep.png, q2_gap.png, aa_pf_scaling.png, aa_breakeven.png)
BUILD_PLAN.md       full execution log, per-wave results, review findings, pitch-safe wording
```

## Core conventions (don't break these)
- A portfolio *selection* is a uint8 vector `x` of length n (`x[i] = 1` = asset i held). Every sampler returns an `(shots, n)` uint8 array, so all metrics work on any sampler. Weights come later via `postprocess.assign_weights`.
- Qiskit bitstrings are little-endian; `backends.sample` converts them so that column i is asset i (tested).
- Constraint logic lives only in `constraints.py`. Cardinality is enforced by the circuit and never penalised.
- Every random function takes `seed`; tests fix seeds and make no network or hardware calls.
- Python 3.11, type hints, NumPy-vectorised checks.

---

## Attribution API (P2)

```python
from fairbench.apps.attribution import attribute, dicke_sampler, plot_attribution
res = attribute(u, cs, fund_holdings, n_samples=5000)             # classical rejection (default)
res = attribute(u, cs, fund_holdings, sampler=dicke_sampler())    # Dicke circuit on aer_statevector
print(res.summary()); plot_attribution(res, "results/x.png")
```
- `u.returns` must hold the window's (T, n) simple returns (`data.load_returns` / `load_universe(returns_path=...)` for real data, `data.synthetic_returns` otherwise).
- `fund_holdings` is a 0/1 selection or explicit weights; `fund_returns` (realised series) and `benchmark_returns` (an index) are optional overrides; `metric` is `"total_return"` or `"sharpe"`.
- Decomposition: `fund − benchmark = constraint_effect (null median − benchmark) + manager_effect (fund − null median)`. Default benchmark = median random portfolio with the same k and no other rule.
- A sampler is any `(u, cs, n_samples, seed) -> SampleResult` with feasible samples. Non-uniform samplers (trained layers, repair, trapped MCMC) bias the null.
- Caveats to say out loud: holdings are static over the window; the CIs are Monte Carlo error only; one window's percentile is not proof of skill; a fund that violates the rule set is flagged (`fund_feasible`).

---

## Next tasks (suggested order)

1. **Real fund data (highest priority).** One fund's holdings, its mandate written as a `ConstraintSet`, universe attributes (sector, ESG score, carbon) and returns for the window. Loaders exist; nothing else in the pipeline needs to change. Until then every slide must say "synthetic".
2. **Pitch deck polish.** Fill the `[team member names]` / `[contact]` placeholders, adjust to the time limit, rehearse with the speaker notes.
3. **Optional:**
   - `apps/optimizer.py` (pivot: CVaR-VQA using `objective_operator` + `training.train` with a CVaR objective).
   - A short hardware run via `ibm` or `vtt` in `backends.py`, framed as feasibility only. Credentials from environment variables only. `dicke_sampler(backend=...)` will pick it up once `backends.sample` supports it.
   - Open nits: the MPS sweep plot label and the slow default ablation runtime (P1 notes in `BUILD_PLAN.md`).
