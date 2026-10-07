# FairBench — handoff

_Last updated 2026-10-07. Repo: `git@github.com:lalashutosh/FairBench.git` (`main`). Detailed wave-by-wave log and numbers: [`BUILD_PLAN.md`](BUILD_PLAN.md)._

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
| **Application layer: `apps/attribution.py`, `apps/optimizer.py` (P2)** | **Not started (2-line stubs)** |
| End-to-end demo script for the pitch | Not started (`scripts/p0_demo.py` is the toy benchmark only) |
| Pitch deck | Not started |
| Hardware runs (`ibm` / `vtt` backends) | Stubs that raise `NotImplementedError` |

Test suite: **238 tests, all green** (~30 s).

### Key findings (use this wording in the pitch)

1. **Dicke + filter ≡ classical rejection sampling** in distribution. It is a correctness baseline, not an advantage (P0 toy: acceptance 0.240 vs 0.244).
2. **Trained layers (P1):** on the P0 toy, acceptance rises 0.24 → 0.32, but the distribution moves away from uniform (TV above sampling floor 0.05 vs 0.00). On the island instance the gain is negligible (0.0123 → 0.0138). Training costs ≥ ~400k shot-equivalents, so it needs ~400k feasible samples to pay back. Swap-MCMC is trapped on islands (coverage 0.31 vs 0.85 for rejection).
3. **MPS simulability (P1):** bond dimension needed grows fast with depth (n=20, p=2: max bond 232, 44 s), so the circuit is not trivially classically simulable at depth, but this is not an advantage claim.
4. **Quantum-enhanced MCMC (Q):** exact spectral gaps, n ≤ 16, parameters frozen on seeds 0–4 and reported on held-out seeds. QeMCMC beats penalty-blind chains 9–19× per step **only when given an exact feasibility oracle**, and a classical tilted walk given the same oracle does at least as well (ratio 0.42–0.84). Pitch line: *"We found no evidence of quantum advantage at n ≤ 16 (small instances, a trend not a proof)."*
5. **Amplitude amplification (AA):** we built a verified reversible feasibility oracle (~120–130·n Toffoli) and a full logical cost model. Samples are exactly uniform over the feasible set. Algorithmic logical qubits ≈ 200–400 at n = 100–200 (excludes factories/routing). Quantum takes ~0.3–1 s per sample (optimistic hardware) vs ~30–120 µs classically, i.e. ~10⁴× slower. Break-even needs a feasible fraction below ~1e-9 to 1e-12, but our synthetic constraint family sits at 0.01–0.2. Full pitch-safe paragraph: `BUILD_PLAN.md` → "AA-3 results".

**Honest story for the pitch:** a constraint-preserving quantum sampler plus a careful benchmarking harness. Every speedup route was tested against fair classical baselines, and we report where the break-even would be. The practical tool (attribution) runs on classical rejection sampling today, with the quantum sampler as a drop-in backend.

---

## Getting started

```bash
git clone git@github.com:lalashutosh/FairBench.git && cd FairBench
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q        # 238 passed
.venv/bin/python scripts/p0_demo.py                     # toy benchmark table
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
  data.py           Universe dataclass, CSV loader, synthetic_universe
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
    attribution.py  P2 — EMPTY STUB
    optimizer.py    P2 — EMPTY STUB (pivot)
scripts/
  p0_demo.py        toy benchmark: Dicke vs rejection vs MCMC
  p1_ablation.py    sampler ablation, identical post-processing (use --no-repair for speed)
  p1_mps_sweep.py   MPS bond-dimension sweep
  q_gap_sweep2.py   QeMCMC vs classical chains, exact spectral gaps (pre-registered protocol)
  aa_pf_scaling.py  feasible-fraction scaling n = 16..200
  aa_estimate.py    AA wall-clock vs classical, break-even plot
results/            CSV/PNG/JSON outputs of all scripts (plots ready for slides: ablation.png,
                    mps_sweep.png, q2_gap.png, aa_pf_scaling.png, aa_breakeven.png)
BUILD_PLAN.md       full execution log, per-wave results, review findings, pitch-safe wording
```

## Core conventions (don't break these)
- A portfolio *selection* is a uint8 vector `x` of length n (`x[i] = 1` = asset i held). Every sampler returns an `(shots, n)` uint8 array, so all metrics work on any sampler. Weights come later via `postprocess.assign_weights`.
- Qiskit bitstrings are little-endian; `backends.sample` converts them so that column i is asset i (tested).
- Constraint logic lives only in `constraints.py`. Cardinality is enforced by the circuit and never penalised.
- Every random function takes `seed`; tests fix seeds and make no network or hardware calls.
- Python 3.11, type hints, NumPy-vectorised checks.

---

## Next tasks (suggested order)

1. **`apps/attribution.py` (highest priority).** Inputs: a `Universe` with historical `returns`, the fund's rule set as a `ConstraintSet`, and the fund's actual holdings and returns. Steps: draw N feasible portfolios (default `baselines.rejection_sample`; the sampler is a parameter so the Dicke/quantum backend drops in), weight them with `postprocess.assign_weights`, compute their returns over the window, and report the fund's percentile rank in that null distribution plus a plot. The **constraint-only effect** (null median vs unconstrained benchmark) vs the **manager effect** (fund vs null median) is the headline output. Keep it a thin layer over existing modules.
   - Needs real or realistic fund data. `data.load_universe(csv_path)` exists. If no real data, generate returns synthetically and say so.
2. **Demo script** `scripts/demo_attribution.py`: one command, prints the ranking and saves a plot to `results/`. Run with rejection sampling and with Dicke on `aer_statevector` (small n) to show the quantum sampler slots in.
3. **Pitch deck** from the findings and wording above. Plots are already in `results/`.
4. **Optional:**
   - `apps/optimizer.py` (pivot: CVaR-VQA using `objective_operator` + `training.train` with a CVaR objective).
   - A short hardware run via `ibm` or `vtt` in `backends.py`, framed as feasibility only. Credentials from environment variables only.
   - Open nits: the MPS sweep plot label and the slow default ablation runtime (P1 notes in `BUILD_PLAN.md`).
