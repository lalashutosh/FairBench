# FairBench — handoff

_Last updated 2026-10-09 (after the real-fund data layer, RD; before that the quantum amplitude estimation wave, QA). Repo: `https://github.com/lalashutosh/FairBench.git`. Detailed wave-by-wave log and numbers: [`BUILD_PLAN.md`](BUILD_PLAN.md); pitch schedule: [`PITCH_PLAN.md`](PITCH_PLAN.md)._

## Start here (next session)

**State of the repo.** `main` holds everything up to the QA wave (680 tests). The real-fund data layer (RD, below) is committed on branch `claude/fairbench-research-plan-368b64`, which is `main` plus new commits and fast-forwards cleanly; it is **not on `main` yet**. With it: 1162 tests, all green, all offline. Run `git pull` before starting.

**What the last session did (2026-10-07 night → 2026-10-08 morning): a meaningful quantum angle (QA wave).**
- **Idea:** the attribution outputs are Monte Carlo averages over the rule-abiding portfolios. The fund's percentile is a ratio of two amplitudes over the Dicke state, so **quantum amplitude estimation (QAE)** is the textbook route to a quadratic query advantage on exactly the quantity the tool reports.
- **Built:** `fairbench/qae/` (amplitude-estimation library; percentile / median / whole-attribution estimators; hybrid classical-warm-start median; exact-counting DP baseline; demo bridge). `constraints.LinearThreshold` (the "portfolio return below the fund's" rule). Oracle encodings for `CountBound`, `AvgBound` and `LinearThreshold`, with superset/subset quantisation modes.
- **Studies:** query scaling (`qae_scaling.py`); noise and NISQ feasibility (`qae_noise.py`); fault-tolerant break-even (`qae_breakeven.py`); example mandate (`qae_mandate.py`); exact-counting DP (`qae_dp_baseline.py`); hybrid median (`qae_hybrid.py`).
- **Pitch assets:** figure `results/qae_pitch_figure.png` (one 16:9 slide) and demo `scripts/demo_qae.py` (~2 s).
- **Reviews:** three independent read-only reviews. All findings fixed, including fair classical baselines (swap-MCMC, shared samples, enumeration), charging every quantum shot, and no use of true amplitudes to set parameters.
- **Headline (finding #8 below):** quadratically fewer queries in noiseless simulation, robust where tight mandates fragment the feasible set. No advantage on today's hardware, and no wall-clock advantage even fault-tolerant.

**What the last session did (2026-10-08 night → 2026-10-09): the real-fund data layer (RD wave).**
- **Research and plan:** [`FAIRBENCH_REAL_DATA_RESEARCH_MEMO.md`](FAIRBENCH_REAL_DATA_RESEARCH_MEMO.md) (SEC sources, eight candidate funds with SEC identifiers, data gaps, schema, constraint format, three reference distributions, legal risks; §14 records the decisions taken). Market context: [`FAIRBENCH_INDUSTRY_STANDARDS_RESEARCH.md`](FAIRBENCH_INDUSTRY_STANDARDS_RESEARCH.md), written in a Codex session and restored from Codex checkpoint refs, where it was the only copy.
- **Built, all tested offline:**
  - `fairbench/ingest/`: polite SEC client (declared `User-Agent` from `FAIRBENCH_SEC_USER_AGENT`, ≤ 2 requests/s, conditional requests, backoff, SEC hosts only) with a raw archive; EDGAR listing and N-PORT parsers; identifier normaliser; prospectus document parser with quote locations; return sources; the ingestion pipeline.
  - `fairbench/store/`: SQLite schema v1; every fact row carries document, locator, extraction run, confidence and a disclosed / derived / assumed / unknown status.
  - `fairbench/mandates/`: canonical constraint record; hard/soft wording by a fixed word list; quote, number and date checks; a selection policy (nothing is enforced before a named review); compilers to Stage 1 (`rules.compile_spec`) and Stage 2 (weight rules).
  - `fairbench/portfolio/`: weight grid (Stage 2) with exact counting and sampling; reference distributions D1 (uniform subsets), D2 (uniform weight grid), D3 (benchmark-aware tilt); exact small-universe validator and metrics; MILP baseline; holdings-change calculator.
  - `fairbench/quantum/encoding.py`: the weight grid as a QUBO, proved exact and one-to-one by brute force on small cases; simulated annealing and an amplitude-amplification circuit as samplers on it.
  - `fairbench/constraints.py`: weight-level rules added (`PositionBound`, `WeightSum`, `WeightedAverage`, `WeightedRisk`, `HoldingsRange`, `TurnoverCap`, `WeightRuleSet`). Nothing existing was changed.
  - `fairbench/apps/real_fund.py` and `scripts/real_fund_{ingest,attribution}.py`, `validate_reference.py`, `build_example_dataset.py`, `reproducibility_report.py`.
- **First real run (2026-10-09), Parnassus Core Equity vs S&P 500 portfolios, 2019-09-30 to 2026-06-30, 27 quarters, all from SEC filings** (downloaded with the user's contact email, at their request; raw files under git-ignored `data/`):
  - Price growth of the fund's disclosed portfolios, held a quarter at a time: +135%. S&P 500 index fund: +149%. Typical 36-stock portfolio allowed by the same rules: +105% (+107% without the exclusions).
  - Average quarterly percentile among rule-abiding portfolios: 54 with capped benchmark weights, 56 with equal weights. Random picks would average 50 give or take 5.6, so this is not distinguishable from 50.
  - The team's own measure (each quarter's portfolio held unchanged until 2026-06-30): average percentile 61 / 67. These windows overlap almost entirely, so there is no error bar for that average and early quarters dominate it.
  - Checks that came out well: the fund's reported NAV return differs from the return of its frozen holdings by +0.11 points a quarter on average (median -0.05); 0-9 index companies a quarter have no derivable return; 91-97% of the fund's equity weight sits in the S&P 500.
  - Outputs: `results/real_fund_parnassus_core_equity.{json,csv,png}`, `_hold.csv`, `_summary.json`, `_exclusions.csv`. Figure for a slide: `results/real_fund_parnassus_core_equity.png`.
  - **What must be said with it:** proxy mandate (holdings count plus 27 companies excluded by SEC industry code: fossil fuels, tobacco, one brewer, two weapons makers; most defence, casino and drinks companies are not caught, and the fund's own ESG research is not reproducible); price returns without dividends; one fund; not evidence of skill.
- **Still not run:** any other fund, a real prospectus through the rule extractor (no live model call), quantum hardware.

**To repeat or extend the real run.** `export FAIRBENCH_SEC_USER_AGENT="Name contact@domain"`, then the four commands in the README section "Real funds from public filings". Another fund needs its series id (memo §3) and a parent index fund that reports on the same months.

**Do next, in this order.**
1. **Run the AI layer against the real API.** It has never made a live call. Set `ANTHROPIC_API_KEY` in the shell (the user does this; never commit a key), then `.venv/bin/python scripts/demo_mandate.py --live`. It prints whether Claude's spec matches the hand-written reference and saves `results/mandate_rules_live.json`. Fix the prompt in `fairbench/mandate.py` if they differ.
2. **Try it on a real fund's policy text** (a prospectus or SFDR pre-contractual disclosure). Expect new rule types: add them to `RULE_FIELDS` and `compile_spec` in `fairbench/rules.py`, and to the kinds list in `SYSTEM_PROMPT`.
3. **Real fund data** for the attribution: holdings, universe data (sector, ESG score, carbon, plus `flag_<name>` / `attr_<name>` / `cat_<name>` columns for whatever the mandate refers to) and returns. Everything shown so far is synthetic.
4. **Deck:**
   - Add a slide for the AI layer.
   - Replace the quantum slides with the QA story: use `results/qae_pitch_figure.png` and the reviewed paragraph in `BUILD_PLAN.md` → "QA results".
   - Fill `[team member names]` / `[contact]` and cut to the time limit.
5. **Optional:** optimizer pivot (`apps/optimizer.py`); hardware run (feasibility only); quantum-walk speedup of MCMC (not explored).

**Decisions made by the assistant that the team has not confirmed.**
- RD wave (2026-10-09, after "you decide"): the SEC contact is read from an environment variable (on 2026-10-09 the user asked for their email to be used, and it was); returns default to N-PORT-implied price returns; the headline reference is uniform subsets with benchmark-proportional weights capped at the fund's own largest position; the new code uses neutral labels ("within-mandate return difference", "realised portfolio percentile") while `apps/attribution.py` and the deck keep "manager effect" until after the pitch; the work was committed on the session branch, not pushed and not merged.
- Benchmark = median random portfolio with the same k and no other rule (an index can be passed via `benchmark_returns`).
- The model never sees per-asset values, only names and scales; all numbers are computed in code.
- Shares of the portfolio and portfolio averages are read as equal-weight statements: a maximum share becomes floor(limit x k) names, a minimum share ceil(limit x k).
- A holdings range in a mandate is benchmarked at the fund's actual k (midpoint if not given).
- Example mandate and universe are fictional/synthetic and were tuned so the rule set is tight but feasible (0.54% of portfolios).
- Tracking error and volatility limits are computed from `u.cov` with equal weights, against the equal-weight universe as benchmark.

**Gaps to know about.**
- No live API validation yet (see step 1). Evidence so far: offline tests with a fake client, and a dry run where a Claude Sonnet agent answered the exact prompt and reproduced the reference rules.
- Constraint coverage on the quantum side:
  - **Oracle** (`ft/oracle.py`): encodes `CountBound`, `AvgBound` and `LinearThreshold` since QA-2.
  - **Not encodable:** `MinGroups`, `VolatilityCap` and `TrackingErrorCap` (not linear) still raise `TypeError`. That is why the example mandate's quantum numbers assume a hypothetical oracle.
  - **Penalty Hamiltonian** (`quantum/hamiltonian.py`): still covers only the original five.
- QAE numbers are noiseless simulations on synthetic data and count oracle queries, not wall-clock time. Never present them as a speedup (see the "must not say" list in `BUILD_PLAN.md` → "QA results").
- Still `unmapped`: weighting schemes other than equal weight, targets that change over time, letter-rating scales, and anything the universe has no data for.
- If a demo is re-run with other settings, the deck numbers (slides 4, 5, 9) must be updated by hand.
- Deck slides in order: cover, problem, method, result, hard, sampler, routes, breakeven, backend, today, next.

**Working agreements.** Work directly on `main`. Push only when asked. Create a branch only when asked. Sonnet subagents may be used for well-specified work. Keep every pitch claim to the review-safe wording below. Update this file at the end of each session.

---

## Context

Team #22 "FairBench" at the Hanken Quantum x Finance Hackathon (8–10 Oct 2026).

**Problem (may pivot):** When an ESG fund underperforms, is it the manager or the ESG constraints? The approach is random-portfolio benchmarking: sample many portfolios that satisfy the *same* rules as the fund, build a null distribution of returns, and see where the real fund ranks. Classical constrained sampling can degrade as rule sets grow (MCMC gets trapped in islands; rejection-sampling acceptance falls).

**Quantum contribution (as tested, see findings below):** constraint-preserving circuits (Dicke state + XY mixer) that only produce portfolios with exactly k holdings, penalty or trained layers, quantum-enhanced MCMC, amplitude amplification, and **quantum amplitude estimation of the attribution statistics**. The first four show no advantage against strong classical baselines. Amplitude estimation gives a quadratic *query* advantage on the fund percentile in noiseless simulation, and it stays robust where tight mandates fragment the feasible set. It is a fault-tolerant-era result: no advantage on today's hardware and no wall-clock advantage. The pitch must use the reviewed wording (`BUILD_PLAN.md` → "QA results"), never "quantum speedup".

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
| Quantum amplitude estimation of attribution statistics (QA) | Done, on `main`: quadratic query advantage in noiseless simulation, robust on fragmented mandates; no NISQ or wall-clock advantage |
| Attribution app `apps/attribution.py` (P2) | Done, tested (synthetic data only) |
| AI mandate layer `mandate.py` + rule compiler `rules.py` (M) | Built, tested offline; **no live API call yet** |
| Demos `scripts/demo_attribution.py`, `scripts/demo_mandate.py` | Done (`results/attribution_demo.*`, `attribution_samplers.csv`, `mandate_attribution.png`, `mandate_demo.json`) |
| Pitch deck | First draft: https://claude.ai/artifact/Do4km8NeU9qw4oZvxEniJK (private until shared; team names are placeholders) |
| **Real fund data** | Pipeline built (RD wave) and **run on one real fund, Parnassus Core Equity, under a proxy mandate**; everything else is still synthetic |
| Optimizer pivot `apps/optimizer.py` | Not started (2-line stub) |
| Hardware runs (`ibm` / `vtt` backends) | Stubs that raise `NotImplementedError` |

Test suite: **1162 tests, all green** (about a minute), all offline (680 before the RD wave).

### Key findings (use this wording in the pitch)

1. **Dicke + filter ≡ classical rejection sampling** in distribution. It is a correctness baseline, not an advantage (P0 toy: acceptance 0.240 vs 0.244).
2. **Trained layers (P1):** on the P0 toy, acceptance rises 0.24 → 0.32, but the distribution moves away from uniform (TV above sampling floor 0.05 vs 0.00). On the island instance the gain is negligible (0.0123 → 0.0138). Training costs ≥ ~400k shot-equivalents, so it needs ~400k feasible samples to pay back. Swap-MCMC is trapped on islands (coverage 0.31 vs 0.85 for rejection).
3. **MPS simulability (P1):** bond dimension needed grows fast with depth (n=20, p=2: max bond 232, 44 s), so the circuit is not trivially classically simulable at depth, but this is not an advantage claim.
4. **Quantum-enhanced MCMC (Q):** exact spectral gaps, n ≤ 16, parameters frozen on seeds 0–4 and reported on held-out seeds. QeMCMC beats penalty-blind chains 9–19× per step **only when given an exact feasibility oracle**, and a classical tilted walk given the same oracle does at least as well (ratio 0.42–0.84). Pitch line: *"We found no evidence of quantum advantage at n ≤ 16 (small instances, a trend not a proof)."*
5. **Amplitude amplification (AA):** we built a verified reversible feasibility oracle (~120–130·n Toffoli) and a full logical cost model. Samples are exactly uniform over the feasible set. Algorithmic logical qubits ≈ 200–400 at n = 100–200 (excludes factories/routing). Quantum takes ~0.3–1 s per sample (optimistic hardware) vs ~30–120 µs classically, i.e. ~10⁴× slower. Break-even needs a feasible fraction below ~1e-9 to 1e-12, but our synthetic constraint family sits at 0.01–0.2. Full pitch-safe paragraph: `BUILD_PLAN.md` → "AA-3 results".

6. **Attribution (P2), synthetic data with planted ground truth:** n = 100, k = 20, three years of Gaussian returns with a planted rally in carbon-heavy names, fund planted at rank 70 of rule-abiding portfolios. Result: fund −3.4 pts vs benchmark = constraint effect −8.7 pts [95% MC CI −9.3, −8.3] + manager effect +5.4 pts; recovered percentile 69.3 ± 0.7. At n = 16 the Dicke circuit (`aer_statevector`, 14 qubits after exclusions), classical rejection and exact enumeration agree (percentile 67.5 / 68.5 / 68.3; KS p = 0.24). **No real fund has been analysed.**
7. **AI mandate layer (M), one fictional mandate:** 18 rules = 14 enforced, 1 trivially satisfied, 3 reported as not expressible (yearly decarbonisation target, EU Taxonomy share, stewardship). The compiled rules exclude 58 of 150 assets and leave 0.54% of 20-name portfolios feasible. Attribution with them: −1.5 pts = constraint −6.3 + manager +4.8, percentile 70.4 ± 0.6 (planted 70). Pitch line: *"An AI layer reads the policy text; deterministic code turns it into constraints and shows what was enforced and what could not be. Tested on one fictional mandate, not on real fund documents, and not yet against the live API."*

8. **Quantum amplitude estimation of the attribution (QA):** in noiseless simulation, amplitude estimation needs quadratically fewer queries for the attribution percentile (error ∝ queries^−1 vs ^−½). It does not depend on the feasible set being connected, which matters on fragmented tight mandates where swap-MCMC gets trapped. Percentile only: on the example mandate ~7–20× fewer queries than rejection at ±1 pp, assuming an oracle for all rules (two of them cannot be encoded). Whole attribution (percentile + both medians, classical warm start for the medians): ~2× fewer queries than rejection at P_F 3%, 4–16× on tighter rules, parity or a small loss on loose rules; 2–9× fewer than swap-MCMC. Noisy hardware: ~17k two-qubit gates per step at 16 assets, advantage needs error rates of order 1e-6. Fault-tolerant wall-clock: ≥ 8e3× slower at ±1 pp. Exact-counting DP is intractable with ≥ 3 weighted rules or overlapping group caps. Full numbers and the reviewed pitch paragraph: `BUILD_PLAN.md` → "QA results"; figure `results/qae_pitch_figure.png`; demo `scripts/demo_qae.py`. Pitch line: *"A fault-tolerant-era result we can quantify today, not a speedup on current or near-term hardware."*

**Honest story for the pitch:** a constraint-preserving quantum sampler plus a careful benchmarking harness. Every speedup route was tested against fair classical baselines, and we report where the break-even would be. The practical tool (attribution) runs on classical rejection sampling today, with the quantum sampler as a drop-in backend.

---

## Getting started

```bash
git clone git@github.com:lalashutosh/FairBench.git && cd FairBench
uv venv --python 3.11 .venv
uv pip install -e ".[dev,ai]"                           # ai = Anthropic SDK, only needed for --live
OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q        # 680 passed
.venv/bin/python scripts/p0_demo.py                     # toy benchmark table
.venv/bin/python scripts/demo_attribution.py            # pitch demo: attribution + sampler swap (~6 s)
.venv/bin/python scripts/demo_mandate.py                # mandate text -> rules -> attribution (~15 s; --live calls Claude)
.venv/bin/python scripts/demo_qae.py                    # quantum amplitude estimation vs classical, loose and tight rules (~2 s)
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
    oracle.py       reversible feasibility oracle; encodes caps, exclusions, CountBound, weighted rules (MinESG, CarbonCap,
                    AvgBound, LinearThreshold); quantise modes superset/subset/minmis/auto
    revsim.py       fast classical simulator for reversible circuits
    resources.py    logical FT cost model: Dicke + oracle + reflection, known/BBHT/fixed-point schedules
  qae/
    estimators.py   amplitude estimation: IQAE, MLAE (LR CI), canonical AE, noisy model, iqae_amplitude_tol; exact query accounting
    attribution_qae.py  percentile / quantile / whole-attribution estimators (hybrid median default), classical baselines
    exact_count.py  exact-counting DP baseline (sector DP with brackets)
    demo_bridge.py  attribute_qae: exact vs classical vs simulated quantum side by side
  apps/
    attribution.py  attribute() -> AttributionResult, rejection_sampler, dicke_sampler, plot_attribution
    real_fund.py    build_period_case / run_period / run_history: a real fund vs its feasible portfolios
  ingest/           http.py (SecClient), archive.py, edgar.py, nport.py, nport_synth.py (synthetic documents),
                    identifiers.py, documents.py, prices.py (return sources), pipeline.py
  store/            schema.sql, db.py (provenance on every fact row)
  mandates/         dsl.py (Constraint), modality.py (hard/soft), extract.py (evidence checks), compile.py
  portfolio/        weights.py (grid), reference.py (D1/D2/D3), exact.py (validator), milp.py, changes.py
  quantum/encoding.py  Stage 2 QUBO, simulated annealing, amplitude-amplification sampler
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
  demo_qae.py       QAE pitch demo (loose + tight rules, errors vs exact, multi-rep reference)
  qae_scaling.py    error vs queries, queries vs P_F, whole attribution, fair baselines
  qae_hybrid.py     hybrid median + whole attribution at e50/e90 (200 reps)
  qae_mandate.py    example mandate (full / encodable), rejection vs MCMC vs quantum
  qae_noise.py      transpiled gate counts, noise model, NISQ thresholds
  qae_breakeven.py  fault-tolerant wall-clock break-even
  qae_dp_baseline.py  exact-counting DP vs sampling
  qae_pitch_figure.py  slide figure from saved results
results/            CSV/PNG/JSON outputs of all scripts (plots ready for slides: attribution_demo.png,
                    ablation.png, mps_sweep.png, q2_gap.png, aa_pf_scaling.png, aa_breakeven.png,
                    qae_pitch_figure.png, qae_scaling.png, qae_hybrid.png, qae_noise.png, qae_breakeven.png)
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
