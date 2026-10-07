# FairBench P0 build plan

Source of truth for interfaces: `fairbench_handoff.md`. This file = execution plan only.
Main thread orchestrates; all code written by subagents. File ownership per wave is disjoint, so parallel agents never touch the same file.

## Environment
- `uv venv --python 3.11 fairbench/.venv`; `uv pip install -e "fairbench[dev]"`.
- Run everything via `fairbench/.venv/bin/python` / `.venv/bin/pytest`.
- `git init` in `fairbench/` so later waves + review can diff.

## Waves

| Wave | Agent | Model | Owns (files) | Done when |
|---|---|---|---|---|
| W0 | scaffold | sonnet | `pyproject.toml`, all `__init__.py`, `data.py`, `constraints.py`, `tests/test_constraints.py`, P1/P2 stubs (`training.py`, `postprocess.py`, `apps/*.py`), stub files for W1 modules (signatures + `NotImplementedError`) | venv works, `pytest tests/test_constraints.py` green, `import fairbench.quantum` ok |
| W1a | quantum-core | **opus** | `quantum/dicke.py`, `quantum/ansatz.py`, `tests/test_dicke.py`, `tests/test_ansatz.py` | Dicke(6,3) exact uniform over 20 states; XY mixer weight-preserving (ring + complete); `build_ansatz` param vectors γ, β |
| W1b | hamiltonian | **opus** | `quantum/hamiltonian.py`, `tests/test_hamiltonian.py` | penalty op diag == brute-force penalty on all 2^n for small n; zero on feasible weight-k states (modulo constant); objective QUBO diag matches `-mu·x + λ xᵀΣx` |
| W1c | backends | sonnet | `backends.py`, `tests/test_backends.py` | bit-order round trip test; `aer_statevector` + `aer_mps` (with `bond_dim` kw); `ibm`/`vtt` raise `NotImplementedError`; returns `(shots,n)` uint8 |
| W1d | baselines+metrics | sonnet | `baselines.py`, `metrics.py`, `tests/test_baselines.py` | enumerate == brute force; rejection/MCMC outputs feasible; MCMC records move count; metrics unit-tested on toy sets |
| W2 | integration | sonnet | `scripts/p0_demo.py` | full `pytest` green; demo prints table (acceptance, TV, coverage, cost/feasible) for Dicke p=0, rejection, MCMC |
| W3 | review | **opus** (read-only) | none | findings list: correctness of Dicke construction, bit order, penalty math, TV/coverage defs, seeding, honesty caveat (Dicke+filter ≡ rejection) |
| W4 | fix | sonnet / opus by severity | files named in findings | re-run pytest + demo; report table |

Model choice rule: opus where math/quantum correctness is subtle (Dicke split-and-cyclic-shift, XY mixer, Pauli penalty expansion, review). Sonnet for mechanical/well-specified code.

## Shared contracts (all agents)
- Selection `x`: uint8 vector length n, column i = asset i. Every sampler returns `(shots, n)` uint8.
- Every random fn takes `seed: int | None`; tests fix seeds; no network/hardware in tests.
- Python 3.11, type hints, NumPy-vectorised checks, small pure functions.
- Constraint logic lives only in `constraints.py`. Cardinality never penalised in Hamiltonian.
- Do not edit files owned by another agent. Need change outside ownership → report it, don't do it.

## Sanity expectation for demo
Dicke p=0 + filter and rejection sampling must show ~same acceptance and TV (same distribution). If not → bug. State this in demo output.

## After P0 (not this build)
P1: `training.py` (COBYLA/SPSA, objective = callable on samples), `postprocess.py` (filter, repair w/ flag, weights). Ablation + MPS bond-dim sweep. P2: apps.

---
# P1 plan (started 2026-10-05)

P0 done: commit a42bd46, 115 tests green.

| Wave | Agent | Model | Owns | Done when |
|---|---|---|---|---|
| P1-A1 | training | **opus** | `training.py`, `objectives.py`, `tests/test_training.py` | train() works with COBYLA/SPSA; objectives = callables on samples (CVaR, uniformity); exact-probability fast path; trained n=12 ansatz beats Dicke acceptance with TV reported |
| P1-A2 | postprocess | sonnet | `postprocess.py`, `tests/test_postprocess.py` | filter_feasible, repair (flagged), assign_weights |
| P1-A3 | islands | sonnet | `instances.py`, `tests/test_instances.py`, `scripts/p0_demo.py` (thin=T fix only) | seeded instance with ≥2 swap components; MCMC coverage shown < 1 |
| P1-A4 | mps sweep | sonnet | `scripts/p1_mps_sweep.py` | bond-dim sweep: TV vs exact + wall time, saves CSV + PNG |
| P1-B | ablation | sonnet | `scripts/p1_ablation.py` | same post-processing, sampler swapped: random bitstrings, product-state circuit, Dicke p=0, trained ansatz, rejection, MCMC — on default + island instance; commit |
| P1-C | review | **opus** read-only | — | findings list |
| P1-D | fixes | by severity | per findings | green + commit |

P1 done: commit 2e7e2f4, 144 tests green. Open nits: ablation runtime 7m20 (repair mode dominates; add --no-repair), mps_sweep.png has stale "chi_needed" axis label, `wall_at_need` column name.

---
# Q plan — quantum-enhanced MCMC (append to BUILD_PLAN.md)

Base: P1 commit 2e7e2f4. Goal: test whether a quantum proposal un-traps MCMC on the island instance, measured by **exact spectral gap**, against fair classical chains. Decision gate at the end decides how it enters the pitch.

## Why this wave
- P1 showed swap-MCMC trapped on islands (coverage 0.32, TV−floor 0.417), while rejection/Dicke are unbiased but cost about 80 proposals per valid portfolio (exact valid fraction 1.26%).
- QeMCMC sits between the two: a proposal that is non-local (escapes islands) but concentrated near low-penalty states (higher acceptance than uniform proposals).
- Rejection sampling is itself an MH chain with an independent uniform proposal, and its gap equals the valid fraction (≈0.0126). **QeMCMC must beat both swap-MCMC and this independent chain.**

## Physics contract (all Q agents)
- Proposal: from feasible x (weight k), evolve |x> under H = −Σ_edges (XX+YY)/2 + α·H_pen + optional h·Σ Z_i fields, for time t, then measure → y.
- H is real and conserves Hamming weight → work in the **weight-k subspace** (dimension C(n,k)), with the basis index map shared with `enumerate_feasible`.
- Exact path: Q = |expm(−iHt)|² in the subspace (no Trotter). Q must be symmetric: assert `allclose(Q, Q.T)`.
- Circuit path: Trotterised, in **palindromic** order (symmetric product formula) so that Uᵀ = U. One shot per chain step via `backends.sample`.
- Target: uniform over the feasible set. MH acceptance for a symmetric Q is simply: accept iff y is feasible.
- Approximate simulators (MPS with small χ) and hardware noise can break symmetry and bias the stationary distribution. Always measure ‖Q−Qᵀ‖ and the stationary-distribution TV when Q is not exact.

## Waves

| Wave | Agent | Model | Owns | Done when |
|---|---|---|---|---|
| Q-0 | nits | sonnet | `scripts/p1_ablation.py` (`--no-repair`), `scripts/p1_mps_sweep.py` (axis label, `wall_at_need` rename) | ablation < 5 min with `--no-repair`; plot and CSV labels fixed |
| Q-A1 | proposal | **opus** | `quantum/proposal.py`, `tests/test_proposal.py` | `subspace_basis(n,k)`, `subspace_hamiltonian(n,k,topology,alpha,pen_op,h)`, `exact_proposal_matrix(t, …)`, `circuit_proposal(x, t, n_trotter, backend, seed)`; tests: Q symmetric, rows sum to 1, weight preserved, Trotter Q → exact Q as n_trotter grows (n=6), circuit 1-shot samples match exact Q rows (χ² test) |
| Q-A2 | chains | sonnet | `chains.py`, `tests/test_chains.py`, `instances.py` (add `island_family(n, k, seed)` only) | generic MH `run_chain(proposal, x0, steps, seed)` with proposals: swap, multi-swap(m random swaps), independent-uniform, quantum-exact, quantum-circuit; `transition_matrix(Q, feasible_mask)` (rejected mass on the diagonal); `spectral_gap(P)` = 1 − max(\|λ₂\|, \|λ_min\|); `mixing_time_tv(P, eps)`; tests: stationary dist uniform; independent-chain gap == valid fraction (analytic); swap-chain gap == 0 when islands are disconnected |
| Q-B | gap sweep | sonnet | `scripts/q_gap_sweep.py` | on P0 + island: grid over t × α (× h); best QeMCMC gap vs swap, best multi-swap m, independent; size trend on `island_family` n = 8, 10, 12, 14, 16 (exact); CSV + PNG; also a fixed-budget sampled run reporting the P1 table columns (coverage, TV−floor, cost per valid sample) |
| Q-C | mps proposals | sonnet | `scripts/q_mps_proposal.py` | proposals via `aer_mps` at χ ∈ {2,4,8,16,32}: TV of Q rows vs exact, ‖Q−Qᵀ‖, resulting gap, stationary-distribution TV, wall time; CSV + PNG |
| Q-D | review | **opus** read-only | — | findings: subspace/basis correctness, symmetry, gap definition, fairness of classical baselines (multi-swap tuned as hard as QeMCMC), whether t/α tuning on one instance leaks into the reported result (tune on seed A, report on seeds B…), and wording of claims |
| Q-E | fixes | by severity | per findings | green + commit |
| Q-F (optional) | hardware | sonnet | `scripts/q_hardware.py` | a few short chains on IBM in Session mode on the island instance: accepted fraction, proposal weight-violation rate, coverage after N steps; framed as feasibility only |

## Decision gate (after Q-E)
- **Best QeMCMC gap > max(multi-swap gap, independent gap), and the ratio grows with n:** QeMCMC becomes the headline technical result. Pitch line: "exact spectral gap, small instances, trend not proof" (Orfi et al. rule out speedups only for unstructured problems).
- **Gain only vs swap-MCMC:** report it as "escapes islands like rejection sampling, and needs no global proposal". Secondary result.
- **No gain:** one honest slide. Pitch rests on amplitude amplification plus simulation cost, as after P1.
- **MPS (Q-C) keeps the gain at small χ:** add a "quantum-inspired sampler runnable today on LUMI" line (Christmann et al. 2025).

## After Q
1. Amplitude-amplification resource estimate (logical qubits, T-count for the island-style oracle at n ≈ 100–200).
2. P2 attribution app (blocked on real fund data and validation answers).
3. Pitch deck from review-safe claims.
# Q results (Q-E2, scripts/q_gap_sweep2.py, results/q2_*; full sweep 317 s)
Frozen on island_family(12) seeds 0-4 (plateau rule: cheapest config within 2% of max median gap): QeMCMC-indicator complete a=32 t=5.9; QeMCMC-violation complete a=32 t=8.9; QeMCMC-surrogate ring a=0.5 t=3.9; CTRW-tilted-indicator complete beta=4 t=8.9; CTRW-tilted-violation complete beta=32 t=20; CTRW-tilted-surrogate complete beta=2 t=0.77; CTRW untilted complete t=0.51; multi-swap m*=9. Penalty-diagonal AUC (feasible vs infeasible): indicator 1.0, violation 1.0, surrogate QUBO ~0.77 (0.67-0.90).
Median [IQR] per-step gap ratio, held-out (n=12 seeds 5-9; n=14, 16 seeds 0-9):
| n | QeMCMC-ind / CTRW-tilted-ind | QeMCMC-ind / best blind | QeMCMC-viol / CTRW-viol | QeMCMC-surr / CTRW-surr |
| 12 | 0.42 [0.42-0.68] | 8.9 | 0.14 | 0.88 |
| 14 | 0.69 [0.64-0.73] | 15.1 | 0.16 | 0.30 |
| 16 | 0.84 [0.82-0.92] | 19.4 | 0.15 | 0.40 |
Per-cost (cost-tuned params, indicator oracle assumed n gates): QeMCMC is far worse than classical for all diags (ratios 1e-3..1e-2; the indicator ratio vs CTRW is erratic, IQR spans 1e-3..1e2, because the cost-tuned CTRW config is unstable). Per-cost vs best blind <0.04.
Gate verdict: NO QUANTUM GAIN. QeMCMC never beats tilted CTRW with the same diagonal (per step median <1 at all n; per cost <<1). The only advantage is over penalty-blind chains (indicator/violation, 9-19x per step), and a classical tilted CTRW with the same oracle diagonal gets it better. Surrogate QUBO diagonal gives no gain over blind (ratio <1.5). Caveats: exact gaps, n<=16, Trotter-cost fit extrapolated from n=12 surrogate, oracle cost assumed.
Pitch-safe wording: "We tested a quantum-enhanced MCMC proposal by exact spectral gap against fair classical baselines, including a classical walk given the same feasibility information. It escapes swap-chain islands only when given an exact feasibility oracle, and a classical walk with that oracle does at least as well; we found no evidence of quantum advantage at n<=16 (small instances, a trend not a proof)."

---
# AA plan — amplitude-amplification resource estimate (started 2026-10-06)

Base: commit 7faa59b. Goal: concrete fault-tolerant cost of exact-uniform feasible sampling via amplitude amplification (AA) over the Dicke state, at n ≈ 50–200, plus an honest wall-clock break-even vs classical rejection.

Key facts to keep straight:
- AA from Dicke with a feasibility phase oracle yields the uniform superposition over F ⇒ **exactly uniform** samples (no bias, unlike penalty/trained layers). Iterations ≈ (π/4)/√P_F; unknown P_F ⇒ exponential search or fixed-point AA (Yoder–Low–Chuang).
- Grover operator = Oracle · (Dicke† · reflect-about-0 · Dicke). Dicke prep cost includes rotation synthesis (T gates).
- Oracle must be EXACT feasibility (integer-quantised ESG/carbon); quantisation mismatch must be measured.
- Quadratic speedups are often eaten by FT overheads (Babbush et al. 2021). Report break-even P_F in wall-clock, with stated assumptions (logical cycle time, T-factory throughput, classical check time).

| Wave | Agent | Model | Owns | Done when |
|---|---|---|---|---|
| AA-1 | oracle | **opus** | `fairbench/ft/__init__.py`, `ft/oracle.py`, `ft/revsim.py`, `tests/test_oracle.py` | reversible feasibility oracle (Cardinality assumed by Dicke; SectorCap popcount+compare; Exclusion; MinESG/CarbonCap weighted integer sums + comparator); classical bit-level simulator; oracle marks exactly F on P0, island, island_family(12..16) seeds; quantisation mismatch reported; gate counts (Toffoli, CNOT, qubits) per component |
| AA-P | P_F scaling | sonnet | `instances.py` (add `scaled_family(n,k,seed)` only), `scripts/aa_pf_scaling.py`, `results/aa_pf_*` | MC estimate (with CI) of P_F for scaled family n=16..200; upper bound when undetected; plot |
| AA-2 | resources | **opus** | `ft/resources.py`, `tests/test_resources.py` | analytic logical counts (qubits, Toffoli/T, T-depth) for oracle + Dicke (+rotation synthesis) + reflection, cross-checked vs AA-1 circuit counts at small n; AA iteration counts (standard, exponential search, fixed-point); verify amplification in exact subspace sim at n≤16 |
| AA-3 | report script | sonnet | `scripts/aa_estimate.py`, `results/aa_*` | table n=50..200: P_F, iterations, logical qubits, T-count per sample, wall-clock per sample under stated FT assumptions vs classical rejection; break-even P_F curve; PNG; optional `qsharp` resource-estimator cross-check |
| AA-4 | review | **opus** read-only (time budget 30 min, thread cap) | — | findings + pitch-safe claims |
| AA-5 | fixes | by severity | per findings | green + commit |

---
# AA-3 results — after AA-4 review + AA-5 fixes (scripts/aa_estimate.py, results/aa_estimate.csv, aa_breakeven.{csv,png}, aa_assumptions.json; 36 s, parent RSS 0.9 GB, pool capped at 4)

OOM root cause (killed the 2026-10-06 13:01 session): `ft/resources.bbht_schedule` allocated `np.arange(M)` per round (M ~ 1e8 at P_F ≲ 1e-10 → 8–10 GB). Now BBHT Lemma 2 closed form `½ − sin(4Mθ)/(4M sin 2θ)`, `E[j] = (M−1)/2`; regression test added. Second OOM risk (AA-4 #1): `measure_classical` ran an uncapped 16-process pool each rebuilding `scaled_family` (~12 GB) → now instance built once, pool = 4, loop-only timing.

AA-5 fixes applied (AA-4 finding #):
- #2 oracle exactness: n > 16 now uses **superset thresholds** `T = ceil(t0 − k/2)` (rounding error ≤ ½ per asset ⇒ F ⊆ F_q provably). Float post-check of each measured sample ⇒ **exactly uniform over F**. FN = 0 at n = 50–200; FP cost |F|/|F_q| = 0.986–0.997, folded into all per-sample costs. Quantisation now reports FP/FN separately vs |F|.
- #4/#5 classical baseline: two measured baselines — naive (dense argsort + ConstraintSet) and lean (argpartition draw + O(k) index check, asserted equal to reference). Lean is 1.5–3× faster. Measured 4-process parallel efficiency 0.86–0.91 (lean). 64 cores is labelled *assumed ideal scaling*. n=150 now measured, not interpolated.
- #6 quantum sensitivity: `qbest` constants (RUS/mixed synthesis rs_const 1.15, t_per_toffoli 2 via measurement uncompute); log-depth Dicke not modelled.
- #7 `t_depth_per_toffoli` default 1 → 2 (consistent with 4-T Gidney AND).
- #8–12 caveats written into aa_assumptions.json notes (serial oracle depth bound; algorithmic qubits only; fixed-point assumes known P_F; seed-0 counts with median P_F).
- #13–15 tests: superset FN=0 at n=50,100 (incl. 50k held-out); gate-level oracle == quantised rule at n=50 (2k inputs, ancillas clean); resource totals by hand; BBHT closed form + huge-M regression. 238 tests green.

Per-sample cost, scaled_family seed-median P_F, known schedule, default constants:

| n | P_F | iters | algorithmic logical qubits | T / sample | quantum s/sample (opt / mod / factory) | classical lean 1 core s/sample | quantum ÷ lean (optimistic) |
|---|---|---|---|---|---|---|---|
| 50 | 0.176 | 1 | 112 | 1.2e5 | 0.071 / 0.71 / 0.12 | 5.5e-6 | 1.3e4 |
| 100 | 0.073 | 2 | 197 | 8.3e5 | 0.27 / 2.7 / 0.83 | 2.7e-5 | 1.0e4 |
| 150 | 0.047 | 3 | 297 | 2.6e6 | 0.60 / 6.0 / 2.6 | 7.5e-5 | 8.0e3 |
| 200 | 0.032 | 4 | 397 | 6.1e6 | 1.1 / 11 / 6.1 | 1.2e-4 | 8.7e3 |

BBHT (unknown P_F) ~1.3× known; fixed-point ~3–4× (assuming P_F known for p_lower). Dicke prep ×2 dominates iterate T (~85–92%). Oracle ~120–130·n Toffoli at 11 bits.

Break-even P_F* (known schedule, n = 100 / 200):
- vs lean 1 core: optimistic 2.3e-10 / 2.2e-10; moderate 2.0e-12 / 1.9e-12. Best-known quantum constants: optimistic 1.0e-9 / 9.4e-10.
- vs naive 1 core: optimistic 1.1e-9 / 1.9e-9. Analytic (c/((π/4)C_q))² agrees within 1.4×.
- vs 64 lean cores (assumed ideal): < 1e-12 in every case.
- P* ∝ c_classical², so a compiled O(k) classical check (~10–30× faster than numpy) would lower P* by a further 10²–10³.

Pitch-safe claim (AA-4 wording, updated): *AA over the Dicke state, with integer-quantised superset thresholds and a classical float post-check, gives samples exactly uniform over the feasible set; the oracle is verified gate-level against exact enumeration for n ≤ 16 and against the quantised rule at n = 50, with zero false negatives by construction. It costs ~120–130·n Toffoli for the oracle, but Dicke-state preparation dominates the T-count (~85–90%), using ~200–400 algorithmic logical qubits at n = 100–200 (excluding factories/routing). Under stated FT assumptions (1–10 µs per T-layer) it beats single-core numpy rejection sampling only if the feasible fraction is below ~1e-9 to 1e-12; that bound is sensitive to constants on both sides, and an optimised classical check pushes it lower. In our synthetic constraint family at n ≤ 200 the feasible fraction is 1e-2 to 2e-1, so there is no wall-clock advantage. Larger universes, where the feasible fraction shrinks, were not evaluated.*

---
# P2 — attribution app (2026-10-07)

Base: commit 3ade8bb. Files: `fairbench/apps/attribution.py`, `fairbench/data.py` (`load_returns`, `synthetic_returns`, `load_universe(returns_path=)`), `tests/test_attribution.py` (17 tests), `scripts/demo_attribution.py`. 255 tests green. Demo: 39 s, peak RSS 0.3 GB.

Design: `attribute(u, cs, fund_holdings, ...)` draws N feasible portfolios with a pluggable sampler, weights them with `postprocess.assign_weights`, computes the statistic over `u.returns`, and returns `fund − benchmark = constraint effect (null median − benchmark) + manager effect (fund − null median)` plus the fund's mid-rank percentile. Default benchmark = median cardinality-only random portfolio (same k, same weighting). CIs are distribution-free order-statistic intervals for the medians (Monte Carlo error only). Sampler output is checked for feasibility; an infeasible fund is flagged, not rejected.

Demo (SYNTHETIC, planted ground truth; `scripts/demo_attribution.py`, seed 0):
- Headline: `scaled_family(100, k=20, esg_q=0.6, carbon_q=0.3)` + Exclusion of the 10 most carbon-intensive names (feasible fraction ~0.36%), 756 Gaussian periods, planted +8%/yr per s.d. of carbon intensity, fund planted at rank 70 in an independent pool of 2000 feasible portfolios.

| quantity | value |
|---|---|
| fund / benchmark / same-rules median | 19.42% / 23.64% / 14.27% |
| active | −4.22 pts |
| constraint effect | −9.38 pts, 95% MC CI [−9.90, −8.80] |
| manager effect | +5.16 pts, 95% MC CI [+4.71, +5.44] |
| percentile | 68.5 ± 0.7 (planted 70) |
| Sharpe check | percentile 85.0, constraint −0.513, manager +0.634 |
| cost | 280 proposals per feasible sample |

- Sampler swap, n = 16, k = 5 (150 feasible portfolios): exact enumeration percentile 69.0; rejection 69.8 ± 0.6 (29.7 proposals/sample); Dicke `aer_statevector` 68.7 ± 0.7 (30.0 shots/sample); two-sample KS between the two nulls D = 0.013, p = 0.82.

Pitch-safe wording: *On a synthetic case with a planted effect, the tool recovers both the cost of the rules and the fund's planted rank; swapping the classical sampler for the Dicke circuit gives the same attribution within Monte Carlo error. No real fund has been analysed, holdings are static over the window, and a single window's percentile is not evidence of skill.*

---
# M — AI mandate layer (2026-10-07)

Base: commit 8e9dbbd. Goal: turn a fund's ESG policy text into the numeric constraints the engine uses, with every number traceable.

Pipeline: `mandate text --(Claude, structured output)--> rule spec (JSON) --(compile_spec, deterministic)--> ConstraintSet + per-rule report`.

Files:
- `fairbench/rules.py` (problem-agnostic, no model call): `SPEC_SCHEMA`, `validate_spec`, `load_spec`/`save_spec`, `compile_spec(spec, u, k) -> CompiledRules` (`constraint_set`, per-rule `status`/`detail`, `warnings`, `report()`, `check_fund()`), `feasible_fraction`.
- `fairbench/mandate.py` (AI layer): `SYSTEM_PROMPT`, `universe_summary`, `build_request`, `extract_rules`, `unverified_sources`, `mandate_to_constraints`. Model `claude-opus-5-5`, `output_config.format` = `SPEC_SCHEMA`, effort high, server-side refusal fallback on. Optional dependency: `uv pip install -e ".[ai]"`.
- `fairbench/data.py`: `Universe.flags` (name -> bool array), read from `flag_<name>` CSV columns.
- `fairbench/constraints.py`: `ConstraintSet.allowed_indices(n)`.
- `fairbench/instances.py`: `named_universe(n, seed)` (named sectors, flags; carbon set mainly by sector).
- `examples/mandate_example.txt` (fictional fund) + `examples/mandate_example.rules.json` (hand-written reference spec).
- `scripts/demo_mandate.py` (`--live` calls the API; default uses the reference spec).

Design decisions:
- The model translates language into the spec vocabulary only. It sees names and scales (sectors, flags, tickers, min/mean/max), never per-asset values. Thresholds, counts and excluded assets are computed in `compile_spec`.
- Relative statements stay relative in the spec (`relative_to_mean 0.7`, `percentile 20`), so a spec can be reused on another universe or date.
- Every rule carries a verbatim `source` quote; `unverified_sources` checks the quote is in the text.
- Rules with no numeric form are kept as `unmapped` with a reason, never dropped.
- Sector weight caps and portfolio averages assume equally weighted holdings (as the constraints do): cap = floor(limit x k) names.

Sampler change (affects earlier numbers): `rejection_sampler` and `dicke_sampler` now draw k-subsets of the non-excluded assets only (still exactly uniform on the feasible set; exclusions cost no proposals and no qubits). `scripts/demo_attribution.py` re-run, seed 0:

| quantity | before | now |
|---|---|---|
| proposals per feasible sample (n=100 headline) | 280 | 27.3 |
| fund / benchmark / same-rules median | 19.42% / 23.64% / 14.27% | 20.27% / 23.64% / 14.91% |
| active = constraint + manager | -4.22 = -9.38 + 5.16 | -3.38 = -8.73 + 5.36 |
| percentile (planted 70) | 68.5 +/- 0.7 | 69.3 +/- 0.7 |
| n=16 swap: exact / rejection / Dicke | 69.0 / 69.8 / 68.7, KS p=0.82 | 68.3 / 68.5 / 67.5, KS p=0.24; Dicke on 14 qubits |

(The planted fund differs between the two runs because the pool it is drawn from uses the sampler. Deck slides 4, 5 and 9 were updated to the new numbers.)

Mandate demo (SYNTHETIC; `scripts/demo_mandate.py`, reference spec, seed 0, 8 s): 10 rules = 7 mapped, 1 trivial (7% position cap holds at 1/k = 5%), 2 unmapped (EU Taxonomy share, stewardship). 23 of 100 assets excluded; feasible fraction 0.57% [0.50%, 0.64%] of 20-name portfolios from the remaining 77. Attribution: fund 16.74%, benchmark 23.29%, same-rules median 11.44%; active -6.55 = constraint -11.85 [-12.39, -11.31] + manager +5.30; percentile 71.5 +/- 0.6 (planted 70).

Validation status of the AI layer:
- Offline tests with a fake client (request shape, parsing, refusal/max_tokens/invalid JSON handling): `tests/test_rules.py` and `tests/test_mandate.py` (156 tests together), written by a Sonnet agent against the spec; they found one bug (`validate_spec` raised TypeError on a non-string `kind`), now fixed. `validate_spec` also rejects out-of-range values (15 written for 15%, a percentile above 100). 411 tests green.
- Prompt dry run: a Claude Sonnet agent was given the exact system prompt, user message and schema, with no access to the repo. Its spec (`results/mandate_rules_dryrun_sonnet.json`) validated, had no unverified quotes, matched the reference on every enforceable rule, and compiled to an identical constraint set (checked on 20k random portfolios). It split the stewardship sentence into two unmapped rules. One prompt line on equal weighting was added afterwards.
- **Not yet done: a live API call** (`--live`). No API key was available in the session. The dry run used a different model and no structured-output enforcement, so it is evidence about the prompt, not about the production path.

Pitch-safe wording: *An AI layer reads a fund's policy text and writes a structured rule spec; deterministic code turns the spec into numeric constraints and shows, sentence by sentence, what was enforced and what could not be. On a fictional mandate the extracted rules matched a hand-written reference. It has been tested on one fictional mandate and not on real fund documents.*

---
# M2 — more constraint types (2026-10-07)

Request: the five constraint classes were too few to express real mandates. Added five classes and five rule kinds; the AI layer and the compiler use them.

New constraint classes (`fairbench/constraints.py`):

| Class | Meaning | Typical mandate sentence |
|---|---|---|
| `CountBound(indices, min_count, max_count)` | number of holdings inside any asset group | "at least half in companies with science-based targets", "max 25% per country" |
| `AvgBound(attribute, lower, upper)` | portfolio average of any numeric field | "boards at least 34% women on average" |
| `MinGroups(category, min_groups)` | distinct sectors/countries held | "spread across at least eight sectors" |
| `VolatilityCap(max_vol)` | sqrt(w' cov w), equal weights | "volatility below 15%" |
| `TrackingErrorCap(max_te, benchmark)` | sqrt((w-b)' cov (w-b)), b = equal-weight universe by default | "tracking error will not exceed 6%" |

Data model: `Universe.attributes` (name -> floats), `Universe.categories` (name -> labels), `Universe.numeric(name)` / `.labels(name)`; CSV columns `attr_<name>`, `cat_<name>`.

New rule kinds (`fairbench/rules.py`): `require`, `group_limit` (values `["*"]` = every value of a category), `threshold_share`, `min_groups`, `risk_limit`. `by` and `metric` are now names resolved against the universe (sector / ticker / flag / any category; esg / carbon / any attribute); unknown names compile to `unmapped`. `portfolio_average` accepts any field and both bounds. A weight share becomes floor(limit x k) names for a maximum and ceil(limit x k) for a minimum. `validate_spec` rejects out-of-range values.

`ConstraintSet.check_batch` now runs cheap constraints first and gives later ones only the rows still feasible (same result; the mandate demo went from 76 s to 15 s).

Scope limits, stated plainly:
- The new classes are enforced by the samplers and attribution. `quantum/hamiltonian.penalty_operator` and `ft/oracle.feasibility_oracle` encode only the original five and raise `TypeError` otherwise (the oracle guard is new; before, unknown classes would have been skipped silently).
- Risk limits use `u.cov`, equal weights and the equal-weight universe as benchmark.
- Still unmapped: non-equal weighting schemes, time-varying targets, letter ratings, anything without data.

Example mandate (fictional, rewritten to exercise the new kinds) on `named_universe(150)`, `scripts/demo_mandate.py`, reference spec, seed 0, 15 s: 18 rules = 14 mapped, 1 trivial, 3 unmapped. 58 of 150 assets excluded; feasible fraction 0.54% [0.48%, 0.61%]. Attribution: fund 20.26%, benchmark 21.73%, same-rules median 15.45%; active -1.48 = constraint -6.29 [-6.80, -5.83] + manager +4.81; percentile 70.4 +/- 0.6 (planted 70). These replace the M numbers above (10 rules, n = 100).

Prompt dry run repeated with the new prompt (Claude Sonnet agent, no repo access, `results/mandate_rules_dryrun_sonnet.json`): valid spec, no unverified quotes, identical on all 15 enforceable rules including every new kind, identical compiled constraint set on 60k portfolios; it split the stewardship sentence into two unmapped rules. Still no live API call.

Tests after M2: 578 green (~40 s). `tests/test_constraints_extra.py` (new classes, data fields, oracle/Hamiltonian guards, samplers) and the extended `tests/test_rules.py` / `tests/test_mandate.py` were written by a Sonnet agent, including a brute-force check of a 16-rule spec against an independent evaluation over all 495 subsets of a 12-asset universe. Follow-up from its report: name counts must be whole numbers (`validate_spec`).

---
# QA plan — quantum amplitude estimation for attribution statistics (started 2026-10-08, branch `quantum-qae`, local only)

Base: 1a42733 (main after PR #3). Goal: a meaningful, honest quantum angle *on the finance quantity itself*. The attribution outputs are Monte Carlo expectations over the uniform feasible set F:
- percentile p = P_{x~U(F)}[S(x) < S_fund]  (= |G|/|F|, G = F ∩ {S < S_fund})
- null median / benchmark median (quantiles → bisection on a threshold, each step a threshold-probability estimate)
- feasible fraction P_F = |F|/C(n,k) (rule tightness)
Classical MC needs ~1/(ε² P_F) proposals; amplitude estimation (QAE) over the Dicke state needs ~1/(ε √P_F)-type Grover queries. Claim to test: the quadratic *query* advantage on the real attribution instance (exact simulation), then honest NISQ (noise) and FT (wall-clock) break-even.

Linearity contract: QAE path uses S(x) = buy-and-hold equal-weight total return = (1/k) Σ x_i g_i − 1, g_i = Π_t(1+r_it) (attribute(..., rebalance=False)). Linear in x ⇒ same weighted-sum comparator as MinESG. Rebalanced returns / Sharpe are NOT linear: out of scope, stated.

Query conventions: classical query = 1 proposal + feasibility/statistic evaluation. Quantum query = 1 application of Q = A S_0 A† S_χ (1 oracle call, 2 Dicke preps); a shot after m iterates costs m queries + 1 state prep.

| Wave | Agent | Model | Owns | Done when |
|---|---|---|---|---|
| QA-1 | AE estimators | **opus** | `fairbench/qae/__init__.py`, `qae/estimators.py`, `tests/test_qae_estimators.py` | IdealOracle / NoisyOracle / SubspaceOracle; classical MC, canonical (QPE) AE, IQAE (Grinko 2021), MLAE (Suzuki 2020; exp/linear/power-law schedules, noise-aware likelihood); CIs; query accounting; subspace oracle matches grover_subspace_state; error-vs-queries slope ≈ −1 (quantum) vs −½ (MC) |
| QA-2 | oracle ext | **opus** | `constraints.py` (add `LinearThreshold` only), `ft/oracle.py`, `ft/resources.py` (formula counts for new rules only), `tests/test_oracle_ext.py` | oracle encodes CountBound, AvgBound, LinearThreshold (return comparator); gate-level revsim == float rule (exact n ≤ 16) with superset/subset quantisation option; formula counts == built |
| QA-3 | QAE attribution + scaling | sonnet | `fairbench/qae/attribution_qae.py`, `scripts/qae_scaling.py`, `tests/test_qae_attribution.py`, `results/qae_scaling*` | `qae_attribute()` → percentile, null/benchmark medians, effects, with query counts; matches classical `attribute(rebalance=False)`; log-log error vs queries on the n=16 demo instance, fitted slopes, multi-seed |
| QA-4 | noise / NISQ | sonnet | `scripts/qae_noise.py`, `results/qae_noise*` | MLAE/IQAE under depolarising decay sweep; real circuit at tiny n on Aer noise model; transpiled CX/depth at n = 8..16 → honest NISQ feasibility statement |
| QA-5 | FT break-even | sonnet | `scripts/qae_breakeven.py`, `results/qae_breakeven*` | per-query FT cost with return comparator; classical lean cost incl. return eval; ε* (percentile precision) at which QAE wins, n = 50..200 |
| QA-6 | sweeps | haiku | results only | seed/hyper-parameter sweeps of QA-3/4 scripts as directed |
| QA-R | review | **opus** read-only | — | findings + pitch-safe claims; fair classical baselines (incl. exact counting DP / stratified MC / quasi-MC) |
| QA-F | fixes | by severity | per findings | green + commit |

Safety: every heavy run `ulimit -v 4000000`, BLAS threads ≤ 2, `timeout`; no uncapped multiprocessing.

---
# QA results (branch `quantum-qae`, commits b532dcd..27eb518; all numbers from results/qae_*; synthetic data)

**One line:** amplitude estimation (QAE) over the Dicke state gives the textbook quadratic *query* advantage on the attribution percentile, and it is the only method that stays robust on tight, fragmented mandates, but it is a fault-tolerant-era claim. Noisy hardware gives no advantage, and fault-tolerant wall-clock gives none at any precision an analyst needs.

What was built: `fairbench/qae/` (IQAE, MLAE with LR CI, canonical AE, noisy model, charged-pilot amplitude-tolerance IQAE; percentile / quantile / whole-attribution estimators), `constraints.LinearThreshold` + oracle encodings for CountBound / AvgBound / LinearThreshold (the "return < fund" comparator, ~1.45× the feasibility oracle), quantise modes superset/subset/minmis. 660 tests green.

Measured (noiseless simulation, exact amplitudes, query = one Grover iterate vs one classical proposal):
1. **Precision scaling (QA-3):** percentile error ∝ queries^−0.91…−1.02 (IQAE/MLAE) vs ^−0.48…−0.50 (classical i.i.d. rejection), n=16 and n=24.
2. **Rule tightness (QA-F1, percentile only, tol 0.01):** queries ∝ P_F^−0.41…−0.56 (quantum) vs P_F^−1.17 (rejection) vs P_F^−0.54 (swap-MCMC, when it works). Exhaustive enumeration beats all samplers at these small n (C = 7.5e4).
3. **Whole attribution (percentile + null median + benchmark median, one shared classical sample set, tol 0.01):** loose rules (P_F ≥ 0.03) — rejection wins or ties (rejection/quantum 0.5–0.6×); tight rules (P_F ≤ 0.007) — quantum 1.75–4.9× fewer queries than rejection, ~0.8–1.8× vs MCMC. The medians gain little from QAE (bisection ×14).
4. **Fragmentation:** at P_F ≤ 0.0023 the swap graph of F splits into 14–19 components; MCMC chains are trapped (q90 percentile error 0.26–0.44 at P_F 0.0023–0.0011). QAE has no connectivity requirement.
5. **Example mandate (QA-F3, n=150, k=20, P_F 0.55%):** at ±1 pp, rejection/quantum 7–21× (encodable sub-mandate 7–26×), MCMC/quantum 2–5× (encodable 3–10×) (range = nominal-target vs matched q95 error; MCMC partly extrapolated in chain length). Full mandate has NO oracle (TrackingErrorCap quadratic, MinGroups); encodable sub-mandate (P_F 0.68%): oracle 2.6e4 Toffoli, 391 logical qubits.
6. **NISQ (QA-4):** Grover iterate of the percentile oracle at n=16: ~1.7e4 CX (4.4e4 on heavy-hex), ~74 qubits. No crossover with classical at p2 ∈ 1e-2…1e-5; any advantage needs p2 ≲ 2e-6 (n=16), ≲ 1e-8 for 10×, tightening ~1/n². Aer density-matrix check at n=4: decay model is conservative (~2×); leakage out of the weight-k subspace is large.
7. **Fault-tolerant wall-clock (QA-F3):** no break-even at any ε ≥ 1e-5 (measured range) in 96 configurations; at ε=0.01 quantum is ≥ 1e4× slower than single-core numpy. Superset-oracle quantisation bias floor on the percentile: 0.8e-3…4e-3 (n=50–200). Unmodelled classical threat: sector-grouped exact-counting DP.

Pitch-safe wording:
> *"The attribution percentile is an amplitude. In noiseless simulation, quantum amplitude estimation over our constraint-preserving circuit reaches a given precision with quadratically fewer queries than classical sampling (error ∝ 1/queries vs 1/√queries), and the gap widens on tight mandates, where rule sets split the feasible portfolios into disconnected islands that trap classical MCMC. On the example mandate that is roughly 7–21× fewer queries than rejection sampling at ±1 percentile point. It is a fault-tolerant-era result: on today's noisy devices the circuit (~17k two-qubit gates per step at 16 assets) gives no advantage, and even fault-tolerant wall-clock time does not beat a laptop at analyst precision."*

Must not say: "quantum speedup for attribution", "advantage grows as mandates tighten" without "vs rejection sampling / where MCMC is trapped", any ε* break-even number, "7×" as a point value. Must qualify: noiseless simulation, synthetic data, percentile is the statistic that benefits (medians little), full mandate oracle hypothetical, quantisation bias floor ~0.1–0.4 pp.

8. **Exact-counting DP (QA-7, `qae/exact_count.py`, results/qae_dp*):** sector DP over (count, CountBound counts, binned weighted sums). A guaranteed ±0.01 bracket needs B≈1000 bins → ~1e12 cells at n=100 with 3 weighted rules; reachable B (12–24) gives unguaranteed point estimates within 0.5 pp at n=100 in ~100 s (rejection: 0.16 s for ±0.01). With ≤2 weighted rules the DP is cheap and a real classical competitor. Encodable mandate (4 weighted rules + 10 overlapping country caps): ≥2.6e16 cells even at B=4 — intractable. Wording: *"exact counting is unguaranteed or intractable at realistic precision once three or more weighted rules combine, and fails on mandates with overlapping group caps, average bounds, min-groups or risk caps."*

Demo: `scripts/demo_qae.py` (2 s) — loose vs tight instance, errors vs exact, multi-rep reference. Pitch figure: `results/qae_pitch_figure.png`.

Open: second review of the fixes not run; quantum-walk (Szegedy/Montanaro) speedup of MCMC not explored; bits=1 oracle bug fixed (ee0538e).
