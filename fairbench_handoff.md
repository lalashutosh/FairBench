# FairBench — handoff

_Last updated 2026-10-10. Repo: `https://github.com/lalashutosh/FairBench.git`. This is the short entry point: read it, then open only the file the task needs (table below)._

## Start here

**State.** 1,184 tests, all green, all offline: `OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q` (one to five minutes; run it in the background, a foreground call can time out).

**What exists.** `README.md` is the full overview, with every headline number and the file it comes from.
- **Core** (synthetic data): sampling engine, selection rules, attribution app, AI mandate layer, the quantum studies.
- **Real-fund layer:** SEC download, holdings database with provenance, universe and prices from filings, three reference distributions, regional rule check, formula sheet, weight-grid QUBO.
- **Real results:** Parnassus Core Equity in depth (`REAL_FUND_RESULTS.md`, generated) and a nine-fund table (`scripts/skill_table.py`, `results/skill_table/`).
- **Quantum:** amplitude estimation of the rank in simulation (`QUANTUM_CORE.md`) and a run on VTT Q50 with a digital twin (`Q50_RESULTS.md`).

**Documentation pass, 2026-10-10.** `README.md` was rewritten around the problem, the solution, what is new, the results and the limits. `WHY_FAIRBENCH.md` is new: the argument in plain words. `BUILD_PLAN.md` and `PITCH_PLAN.md` were deleted; the architecture is now in the README, and the study results, reviewed wording and the list of claims we do not make are in `QUANTUM_CORE.md` sections 11 and 12. The wave-by-wave build log is in git history (`git show cd90c94:BUILD_PLAN.md`).

**Two Parnassus numbers, both correct.** `REAL_FUND_RESULTS.md` gives an average rank of 54 (capped benchmark-proportional weights, exclusion list, 5,000 draws). The nine-fund table gives 50.4 (uncapped index weights, no exclusions, 20,000 draws, split-adjusted universe prices). Both are inside the luck band. Say which one a number comes from.

**Do next.**
1. **Real rules by hand:** the team writes a real fund's rule spec; then wire it into `scripts/real_fund_attribution.py` in place of the proxy exclusion list (`run_period` already accepts `selection_rules` and `weight_rules`). How to hand one over: `README.md` → "Real funds from public filings", last paragraph.
2. **One pipeline for both real-fund results:** `scripts/skill_table.py` reads the CSV export in `results/real_data/` and has its own split adjustment; `apps/real_fund.py` reads the database. Merge them.
3. **After the event:** more funds and quarters; a licensed ESG source; verified regional texts (most thresholds in `mandates/regions.py` are marked unverified); amplitude estimation over weights.

**Where to look.**

| File | Open it when |
|---|---|
| `README.md` | the overview, headline results, setting up, running demos, using the API, finding a module |
| `WHY_FAIRBENCH.md` | you need the argument: the problem, why existing tools miss it, why quantum fits |
| `REAL_FUND_RESULTS.md` | you need the Parnassus numbers, their checks, or what is disclosed / derived / assumed |
| `QUANTUM_CORE.md` | anything about the quantum estimator: how it works, what it can encode, where it does not help, the earlier studies (§11), what may be claimed (§12) |
| `Q50_RESULTS.md` | the hardware run: results, digital twin, job ids, how to reproduce |
| `FAIRBENCH_REAL_DATA_RESEARCH_MEMO.md` | adding a fund or a data source: SEC sources, fund identifiers, data gaps, licensing |
| `FAIRBENCH_INDUSTRY_STANDARDS_RESEARCH.md` | questions about the market, competitors, GIPS / SFDR / SEC rules |

**Decisions taken by the assistant that the team has not confirmed.**
- Real-fund layer: the SEC contact comes from the environment variable `FAIRBENCH_SEC_USER_AGENT` (the user's email was used at their request and is in no committed file); returns are price returns derived from filings; the headline reference is random same-size portfolios with benchmark-proportional weights capped at the fund's largest position; the exclusion list is a stand-in from SEC industry codes; new code says "within-mandate return difference" and "realised portfolio percentile", while `apps/attribution.py` still says "manager effect".
- Core: the default benchmark is the median random portfolio of the same size; the model never sees per-asset values; shares and averages in a mandate are read as equal-weight statements; a holdings range is benchmarked at the fund's actual count; risk limits use the universe covariance with equal weights.
- Documentation pass: the README lists Claude Code under "Resources used"; remove the line if the team prefers.

**Gaps.**
- No live model call has ever been made. The mandate layer is tested with a fake client.
- Real funds: proxy rules only; a fund's own ESG screen cannot be reproduced from public data; no dividends; the latest date is two months behind.
- Quantum: no advantage on hardware. The oracle encodes linear rules; minimum-groups and the two risk caps are not encodable; the penalty Hamiltonian covers the original five rules. `fairbench.backends.sample` still runs on simulators only; the hardware call lives in `scripts/hw_q50.py`.
- `REAL_FUND_RESULTS.md` is generated: change `scripts/reproducibility_report.py`, never the file.

**Working agreements.** Work directly on `main`. Push only when asked. Create a branch only when asked. Sonnet subagents may be used for well-specified work. Keep every claim inside `QUANTUM_CORE.md` §5 and §12. Update this file at the end of each session and keep it short: detail belongs in the files above.

## Findings, one line each

1. Dicke state + filter draws the same distribution as classical rejection sampling: a correctness baseline, not an advantage.
2. Trained layers raise acceptance slightly but leave the uniform distribution; training cost is not paid back.
3. The circuit is not trivially simulable at depth (MPS bond dimension grows fast); not an advantage claim.
4. Quantum-enhanced MCMC: no evidence of advantage at n ≤ 16; a classical walk with the same oracle does as well.
5. Amplitude amplification: about 10⁴ times slower than classical at n = 100–200; break-even needs a feasible fraction near 1e-9 to 1e-12.
6. Attribution on synthetic data recovers a planted rank (69.3 ± 0.7 against 70).
7. AI mandate layer on one fictional mandate: 14 of 18 rules enforced, 3 reported as not expressible; never run live.
8. Amplitude estimation of the percentile: quadratically fewer queries in noiseless simulation, 12× on the tightest rules, robust on fragmented mandates; no advantage on today's hardware and none in wall-clock time.
9. Parnassus Core Equity: rank 54 of 100, rules cost about 2 points. The quantum estimator, simulated on the same quarters, gives the same rank with a median of 1,900 queries against 5,366 classical samples for ±1 point; the circuit (about 949 error-corrected qubits) is beyond any current machine.
10. Nine real funds: all eight ESG funds inside the luck band, 0 extreme quarters in 198; the equal-weighted control is flagged in 13 of 26.
11. VTT Q50: valid portfolios 54% against 37.5% by chance at 4 assets; signal mostly gone after one Grover step; a one-parameter twin (two-qubit error 1.8%) reproduces the data.

## Core conventions (don't break these)
- A portfolio *selection* is a uint8 vector `x` of length n (`x[i] = 1` = asset i held). Every sampler returns an `(shots, n)` uint8 array, so all metrics work on any sampler. Weights come later via `postprocess.assign_weights`.
- Qiskit bitstrings are little-endian; `backends.sample` converts them so that column i is asset i (tested).
- Constraint logic lives only in `constraints.py`. Cardinality is enforced by the circuit and never penalised.
- The model call lives only in `mandate.py`. Everything numeric about a mandate is computed in `rules.py`; the model never sees per-asset data.
- Every random function takes `seed`; tests fix seeds and make no network or hardware calls.
- Python 3.11, type hints, NumPy-vectorised checks.
- New in the real-fund layer: weight rules also live in `constraints.py`; raw downloads and any licensed data go under the git-ignored `data/`; a value that is not known is stored as unknown, never filled in.

## Memory safety

The dev laptop has 14 GB. Cap threads for heavy scripts (`OMP_NUM_THREADS=1`), and for the old sweep scripts also `ulimit -v 4000000`; never use `ulimit -v` with the test suite (Qiskit Aer reserves virtual memory and fails).
