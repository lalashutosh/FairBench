# FairBench — handoff

_Last updated 2026-10-07 (after the AI mandate layer and the ten constraint classes). Repo: `https://github.com/lalashutosh/FairBench.git`. Detailed wave-by-wave log and numbers: [`BUILD_PLAN.md`](BUILD_PLAN.md)._

## Start here (next session)

**State of the repo.** Everything is on `main`. Run `git pull` before starting.

**What the last session did (2026-10-07).**
- P2: `fairbench/apps/attribution.py`, `scripts/demo_attribution.py`, `README.md`, pitch deck draft (11 slides): https://claude.ai/artifact/Do4km8NeU9qw4oZvxEniJK (lives only in that artifact; private until shared; layout not checked visually).
- **Ten constraint classes** instead of five (`constraints.py`): added `CountBound`, `AvgBound`, `MinGroups`, `VolatilityCap`, `TrackingErrorCap`. The universe can now carry numeric `attributes` and label `categories` besides `flags`.
- **AI mandate layer (new):** `fairbench/mandate.py` asks Claude to turn a fund's policy text into a rule spec (JSON); `fairbench/rules.py` compiles the spec into a `ConstraintSet` and reports what every sentence became. Demo: `scripts/demo_mandate.py`. Details in "AI mandate layer" below.
- Samplers now draw only from non-excluded assets, which made the attribution demo ~10x cheaper and changed its numbers slightly. Results, README, deck slides 4/5/9 and this file use the new numbers.

**Do next, in this order.**
1. **Run the AI layer against the real API.** It has never made a live call. Set `ANTHROPIC_API_KEY` in the shell (the user does this; never commit a key), then `.venv/bin/python scripts/demo_mandate.py --live`. It prints whether Claude's spec matches the hand-written reference and saves `results/mandate_rules_live.json`. Fix the prompt in `fairbench/mandate.py` if they differ.
2. **Try it on a real fund's policy text** (a prospectus or SFDR pre-contractual disclosure). Expect new rule types; add them to `RULE_FIELDS` and `compile_spec` in `fairbench/rules.py`, and to the kinds list in `SYSTEM_PROMPT`.
3. **Real fund data** for the attribution: holdings, universe data (sector, ESG score, carbon, plus `flag_<name>` / `attr_<name>` / `cat_<name>` columns for whatever the mandate refers to) and returns. Everything shown so far is synthetic.
4. Deck: add a slide for the AI layer, fill `[team member names]` / `[contact]`, cut to the time limit.
5. Optional: optimizer pivot (`apps/optimizer.py`), hardware run.

**Decisions made by the assistant that the team has not confirmed.**
- Benchmark = median random portfolio with the same k and no other rule (an index can be passed via `benchmark_returns`).
- The model never sees per-asset values, only names and scales; all numbers are computed in code.
- Shares of the portfolio and portfolio averages are read as equal-weight statements: a maximum share becomes floor(limit x k) names, a minimum share ceil(limit x k).
- A holdings range in a mandate is benchmarked at the fund's actual k (midpoint if not given).
- Example mandate and universe are fictional/synthetic and were tuned so the rule set is tight but feasible (0.54% of portfolios).
- Tracking error and volatility limits are computed from `u.cov` with equal weights, against the equal-weight universe as benchmark.

**Gaps to know about.**
- No live API validation yet (see step 1). Evidence so far: offline tests with a fake client, and a dry run where a Claude Sonnet agent answered the exact prompt and reproduced the reference rules.
- The five new constraint classes are checked by the samplers and by attribution only. The penalty Hamiltonian (`quantum/hamiltonian.py`) and the fault-tolerant oracle (`ft/oracle.py`) still encode the original five and raise `TypeError` for the others. `CountBound` and `AvgBound` have the same shape as `SectorCap` and `MinESG`, so extending the oracle to them is mechanical; the two risk caps are quadratic and would need new circuitry.
- Still `unmapped`: weighting schemes other than equal weight, targets that change over time, letter-rating scales, and anything the universe has no data for.
- If a demo is re-run with other settings, the deck numbers (slides 4, 5, 9) must be updated by hand.
- Deck slides in order: cover, problem, method, result, hard, sampler, routes, breakeven, backend, today, next.

**Working agreements.** Work directly on `main`. Push only when asked. Create a branch only when asked. Sonnet subagents may be used for well-specified work. Keep every pitch claim to the review-safe wording below. Update this file at the end of each session.

---

## Context

Team #22 "FairBench" at the Hanken Quantum x Finance Hackathon (8–10 Oct 2026).

**Problem (may pivot):** When an ESG fund underperforms, is it the manager or the ESG constraints? The approach is random-portfolio benchmarking: sample many portfolios that satisfy the *same* rules as the fund, build a null distribution of returns, and see where the real fund ranks. Classical constrained sampling can degrade as rule sets grow (MCMC gets trapped in islands; rejection-sampling acceptance falls).

**Quantum contribution (as tested, see findings below):** Constraint-preserving circuits (Dicke state + XY mixer) that only produce portfolios with exactly k holdings, plus penalty or trained layers for the other constraints, plus a fault-tolerant amplitude-amplification (AA) roadmap. We benchmarked all three honestly against strong classical baselines; **none shows a speedup at the scales we could test**. The pitch must use the review-safe wording in this file, not "quadratic speedup".

**AI layer:** a fund's policy text is translated by Claude into a structured rule spec, and deterministic code compiles that spec into the constraints the samplers use, reporting sentence by sentence what was enforced and what could not be.

**Pivot plan:** If problem validation fails, the same engine becomes a CVaR-VQA ESG-constrained portfolio **optimizer**. Only the application layer changes. **Keep everything below `apps/` problem-agnostic.**

---

## Where things stand

| Area | Status |
|---|---|
| Data, constraints, quantum core, backends, baselines, metrics (P0) | Done, tested |
| Training, post-processing, island instances, ablation, MPS sweep (P1) | Done, tested |
| Quantum-enhanced MCMC study (Q) | Done — no quantum gain |
| Amplitude-amplification fault-tolerant resource estimate (AA) | Done — no wall-clock advantage |
| Quantum amplitude estimation of attribution statistics (QA) | Done on local branch `quantum-qae` (not pushed): quadratic query advantage in noiseless simulation, robust on fragmented mandates; no NISQ or wall-clock advantage |
| Attribution app `apps/attribution.py` (P2) | Done, tested (synthetic data only) |
| AI mandate layer `mandate.py` + rule compiler `rules.py` (M) | Built, tested offline; **no live API call yet** |
| Demos `scripts/demo_attribution.py`, `scripts/demo_mandate.py` | Done (`results/attribution_demo.*`, `attribution_samplers.csv`, `mandate_attribution.png`, `mandate_demo.json`) |
| Pitch deck | First draft: https://claude.ai/artifact/Do4km8NeU9qw4oZvxEniJK (private until shared; team names are placeholders) |
| **Real fund data** | **Not started: every attribution result so far is synthetic** |
| Optimizer pivot `apps/optimizer.py` | Not started (2-line stub) |
| Hardware runs (`ibm` / `vtt` backends) | Stubs that raise `NotImplementedError` |

Test suite: **578 tests, all green** (~40 s), all offline.

### Key findings (use this wording in the pitch)

1. **Dicke + filter ≡ classical rejection sampling** in distribution. It is a correctness baseline, not an advantage (P0 toy: acceptance 0.240 vs 0.244).
2. **Trained layers (P1):** on the P0 toy, acceptance rises 0.24 → 0.32, but the distribution moves away from uniform (TV above sampling floor 0.05 vs 0.00). On the island instance the gain is negligible (0.0123 → 0.0138). Training costs ≥ ~400k shot-equivalents, so it needs ~400k feasible samples to pay back. Swap-MCMC is trapped on islands (coverage 0.31 vs 0.85 for rejection).
3. **MPS simulability (P1):** bond dimension needed grows fast with depth (n=20, p=2: max bond 232, 44 s), so the circuit is not trivially classically simulable at depth, but this is not an advantage claim.
4. **Quantum-enhanced MCMC (Q):** exact spectral gaps, n ≤ 16, parameters frozen on seeds 0–4 and reported on held-out seeds. QeMCMC beats penalty-blind chains 9–19× per step **only when given an exact feasibility oracle**, and a classical tilted walk given the same oracle does at least as well (ratio 0.42–0.84). Pitch line: *"We found no evidence of quantum advantage at n ≤ 16 (small instances, a trend not a proof)."*
5. **Amplitude amplification (AA):** we built a verified reversible feasibility oracle (~120–130·n Toffoli) and a full logical cost model. Samples are exactly uniform over the feasible set. Algorithmic logical qubits ≈ 200–400 at n = 100–200 (excludes factories/routing). Quantum takes ~0.3–1 s per sample (optimistic hardware) vs ~30–120 µs classically, i.e. ~10⁴× slower. Break-even needs a feasible fraction below ~1e-9 to 1e-12, but our synthetic constraint family sits at 0.01–0.2. Full pitch-safe paragraph: `BUILD_PLAN.md` → "AA-3 results".

6. **Attribution (P2), synthetic data with planted ground truth:** n = 100, k = 20, three years of Gaussian returns with a planted rally in carbon-heavy names, fund planted at rank 70 of rule-abiding portfolios. Result: fund −3.4 pts vs benchmark = constraint effect −8.7 pts [95% MC CI −9.3, −8.3] + manager effect +5.4 pts; recovered percentile 69.3 ± 0.7. At n = 16 the Dicke circuit (`aer_statevector`, 14 qubits after exclusions), classical rejection and exact enumeration agree (percentile 67.5 / 68.5 / 68.3; KS p = 0.24). **No real fund has been analysed.**
7. **AI mandate layer (M), one fictional mandate:** 18 rules = 14 enforced, 1 trivially satisfied, 3 reported as not expressible (yearly decarbonisation target, EU Taxonomy share, stewardship). The compiled rules exclude 58 of 150 assets and leave 0.54% of 20-name portfolios feasible. Attribution with them: −1.5 pts = constraint −6.3 + manager +4.8, percentile 70.4 ± 0.6 (planted 70). Pitch line: *"An AI layer reads the policy text; deterministic code turns it into constraints and shows what was enforced and what could not be. Tested on one fictional mandate, not on real fund documents, and not yet against the live API."*

8. **Quantum amplitude estimation of the attribution (QA, branch `quantum-qae`):** the fund percentile is an amplitude over the Dicke state. In noiseless simulation QAE reaches a given precision with quadratically fewer queries than classical sampling (error ∝ queries^−1 vs ^−½). Against i.i.d. rejection the gap grows as rules tighten (queries ∝ P_F^−½ vs P_F^−1); swap-MCMC matches that scaling only while the feasible set stays connected, and tight rule sets split it into 14–19 islands that trap the chains. Example mandate (P_F 0.55%): ~7–21× fewer queries than rejection, ~2–5× fewer than MCMC at ±1 pp. Whole attribution (percentile + both medians): quantum wins only for tight rules (P_F ≲ 0.7%), 1.8–4.9×; loose rules favour plain sampling. Noisy hardware: ~17k two-qubit gates per step at 16 assets, advantage needs error rates ≲ 2e-6. Fault-tolerant wall-clock: ≥ 1e4× slower at ±1 pp, no break-even at any ε ≥ 1e-5. Full numbers and the exact pitch sentence: `BUILD_PLAN.md` → "QA results". Pitch line: *"The attribution percentile is an amplitude; quantum amplitude estimation needs quadratically fewer queries and stays robust where tight mandates fragment the feasible set — a fault-tolerant-era result we can quantify today, not a speedup on current hardware."*

**Honest story for the pitch:** a constraint-preserving quantum sampler plus a careful benchmarking harness. Every speedup route was tested against fair classical baselines, and we report where the break-even would be. The practical tool (attribution) runs on classical rejection sampling today, with the quantum sampler as a drop-in backend.

---

## Getting started

```bash
git clone git@github.com:lalashutosh/FairBench.git && cd FairBench
uv venv --python 3.11 .venv
uv pip install -e ".[dev,ai]"                           # ai = Anthropic SDK, only needed for --live
OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q        # 578 passed
.venv/bin/python scripts/p0_demo.py                     # toy benchmark table
.venv/bin/python scripts/demo_attribution.py            # pitch demo: attribution + sampler swap (~6 s)
.venv/bin/python scripts/demo_mandate.py                # mandate text -> rules -> attribution (~15 s; --live calls Claude)
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
  constraints.py    Cardinality, SectorCap, Exclusion, MinESG, CarbonCap, CountBound, AvgBound, MinGroups,
                    VolatilityCap, TrackingErrorCap, ConstraintSet (ALL rule logic lives here)
  instances.py      p0_instance, island_instance, island_family(n, seed), scaled_family(n, seed) (n up to 200), named_universe
  backends.py       sample(circuit, params, shots, backend="aer_statevector"|"aer_mps"|"ibm"|"vtt")
  baselines.py      enumerate_feasible, random_k_subsets, rejection_sample, mcmc_swap_sample
  metrics.py        acceptance_rate, tv_to_uniform, coverage, cost_per_feasible_sample
  objectives.py     training objectives as callables on samples (uniformity, CVaR, ...)
  training.py       train(ansatz, objective_fn, backend, optimizer="COBYLA"|"SPSA", ...)
  postprocess.py    filter_feasible, repair (flagged: biases sampling), assign_weights
  rules.py          rule spec (JSON schema), validate/load/save, compile_spec -> ConstraintSet + report, feasible_fraction
  mandate.py        AI layer: mandate text -> rule spec via Claude (prompt, request, quote check)
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
examples/           mandate_example.txt (fictional fund) + mandate_example.rules.json (reference spec)
scripts/
  demo_attribution.py  pitch demo: planted synthetic case, attribution, rejection vs Dicke vs exact
  demo_mandate.py   mandate text -> rule spec -> compiled constraints -> attribution
  p0_demo.py        toy benchmark: Dicke vs rejection vs MCMC
  p1_ablation.py    sampler ablation, identical post-processing (use --no-repair for speed)
  p1_mps_sweep.py   MPS bond-dimension sweep
  q_gap_sweep2.py   QeMCMC vs classical chains, exact spectral gaps (pre-registered protocol)
  aa_pf_scaling.py  feasible-fraction scaling n = 16..200
  aa_estimate.py    AA wall-clock vs classical, break-even plot
results/            CSV/PNG/JSON outputs of all scripts (plots ready for slides: attribution_demo.png,
                    ablation.png, mps_sweep.png, q2_gap.png, aa_pf_scaling.png, aa_breakeven.png)
README.md           public-facing overview, quick start, usage example
BUILD_PLAN.md       full execution log, per-wave results, review findings, pitch-safe wording
```

## Core conventions (don't break these)
- A portfolio *selection* is a uint8 vector `x` of length n (`x[i] = 1` = asset i held). Every sampler returns an `(shots, n)` uint8 array, so all metrics work on any sampler. Weights come later via `postprocess.assign_weights`.
- Qiskit bitstrings are little-endian; `backends.sample` converts them so that column i is asset i (tested).
- Constraint logic lives only in `constraints.py`. Cardinality is enforced by the circuit and never penalised.
- The model call lives only in `mandate.py`. Everything numeric about a mandate is computed in `rules.py`; the model never sees per-asset data.
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

## AI mandate layer (M)

```
mandate text --(Claude, structured output)--> rule spec (JSON) --(compile_spec)--> ConstraintSet + report
```
```python
from fairbench.mandate import mandate_to_constraints      # one model call
from fairbench.rules import compile_spec, load_spec       # no model call
compiled, extraction = mandate_to_constraints(text, u, k=20)
compiled = compile_spec(load_spec("examples/mandate_example.rules.json"), u, k=20)
print(compiled.report()); cs = compiled.constraint_set
```
- **Setup:** `uv pip install -e ".[ai]"`, then `ANTHROPIC_API_KEY` in the environment. Model `claude-opus-5-5`, structured output held to `rules.SPEC_SCHEMA`, effort high, server-side refusal fallback enabled. One call per mandate.
- **Rule kinds (12):** `holdings`, `position_cap`, `sector_cap`, `exclude`, `require`, `exclude_threshold`, `group_limit`, `threshold_share`, `portfolio_average`, `min_groups`, `risk_limit`, `unmapped`. Groups are named by sector, ticker, flag or any category in the universe (country, region); numeric rules work on `esg`, `carbon` or any `attributes` field. Levels are `absolute`, `percentile` or `relative_to_mean`. The README has a table of example sentences.
- **Universe data:** `flag_<name>`, `attr_<name>` and `cat_<name>` CSV columns become `Universe.flags`, `.attributes`, `.categories`.
- **Guard rails:** the model gets names and scales only; each rule has a verbatim quote that code checks against the text; rules with no numeric form are returned as `unmapped`; `compiled.report()` lists every rule with what it became; `compiled.check_fund(x, u)` says which rules a fund's holdings break; `feasible_fraction` flags rule sets that are too tight or infeasible.
- **To add a rule type:** add it to `RULE_FIELDS` and a branch in `compile_spec` (`rules.py`), describe it in `SYSTEM_PROMPT` (`mandate.py`), add a test. If it needs a new constraint class, that goes in `constraints.py`.
- **Tests are offline** (fake client). `scripts/demo_mandate.py --live` is the only thing that spends API money.

---

## Next tasks

See "Start here" at the top.
