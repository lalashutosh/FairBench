# FairBench — handoff

_Last updated 2026-10-09. Repo: `https://github.com/lalashutosh/FairBench.git`. This is the short entry point: read it, then open only the file the task needs (table below)._

## Start here

**State.** Everything is on `main`. 1,162 tests, all green, all offline: `OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q` (about a minute; run it in the background, a foreground call can time out).

**What exists.**
- **Core** (synthetic data): sampling engine, selection rules, attribution app, AI mandate layer, five quantum studies. Setup, usage and layout are in `README.md`.
- **Real-fund layer** (2026-10-09): SEC download, holdings database with provenance, universe and prices from filings, three reference distributions, evidence-checked constraint format, weight-grid QUBO. How to run: `README.md` → "Real funds from public filings".
- **One real result:** Parnassus Core Equity against S&P 500 portfolios, 2019-09 to 2026-06, 27 quarters, from SEC filings only. Plain-language write-up, quarter table and checks: `REAL_FUND_RESULTS.md`. Slide figure: `results/real_fund_parnassus_core_equity.png`.

**The real result.** Price growth: fund +135%, S&P 500 index fund +149%, typical 36-stock portfolio under the same rules +105%. Average quarterly rank among rule-abiding portfolios: 54 of 100 (luck alone: 50 give or take 6). The exclusions cost about 2 points. It must always carry: *proxy rules (holdings count plus 27 companies excluded by SEC industry code), price returns without dividends, one fund, not a measure of skill.*

**Do next.**
1. **Deck:** a slide for the real result (the figure plus the caveat line above), a slide for the AI layer, the quantum slides as in `PITCH_PLAN.md`, team names and contact, cut to the time limit.
2. **Run the AI layer against the live API** (never done): the user sets `ANTHROPIC_API_KEY`, then `.venv/bin/python scripts/demo_mandate.py --live`.
3. **Optional:** a second fund (series ids in the memo §3; it needs an index fund that reports on the same months); a real prospectus through `fairbench/mandates/extract.py`; a hardware feasibility run.

**Where to look.**

| File | Open it when |
|---|---|
| `README.md` | setting up, running demos, using the API, finding a module |
| `REAL_FUND_RESULTS.md` | you need the real-fund numbers, their checks, or what is disclosed / derived / assumed |
| `PITCH_PLAN.md` | working on the pitch: schedule, judge questions, and the exact review-safe wording for every finding |
| `BUILD_PLAN.md` | you need the detail or the "must not say" list behind a quantum number |
| `FAIRBENCH_REAL_DATA_RESEARCH_MEMO.md` | adding a fund or a data source: SEC sources, fund identifiers, data gaps, licensing |
| `FAIRBENCH_INDUSTRY_STANDARDS_RESEARCH.md` | questions about the market, competitors, GIPS / SFDR / SEC rules |

**Decisions taken by the assistant that the team has not confirmed.**
- Real-fund layer: the SEC contact comes from the environment variable `FAIRBENCH_SEC_USER_AGENT` (the user's email was used at their request and is in no committed file); returns are price returns derived from filings; the headline reference is random same-size portfolios with benchmark-proportional weights capped at the fund's largest position; the exclusion list is a stand-in from SEC industry codes; new code says "within-mandate return difference" and "realised portfolio percentile", while `apps/attribution.py` and the deck still say "manager effect".
- Core: the default benchmark is the median random portfolio of the same size; the model never sees per-asset values; shares and averages in a mandate are read as equal-weight statements; a holdings range is benchmarked at the fund's actual count; risk limits use the universe covariance with equal weights.

**Gaps.**
- No live model call has ever been made; the mandate layer is tested with a fake client.
- Real funds: one fund only; the fund's own ESG screen cannot be reproduced from public data; no dividends; the latest date is two months behind.
- Quantum: simulator only. The oracle encodes linear rules; minimum-groups and the two risk caps are not encodable; the penalty Hamiltonian covers the original five rules. Amplitude-estimation numbers are noiseless query counts on synthetic data, never a speedup.
- Deck numbers (slides 4, 5, 9) must be updated by hand if a demo is re-run with other settings.

**Working agreements.** Work directly on `main`. Push only when asked. Create a branch only when asked. Sonnet subagents may be used for well-specified work. Keep every pitch claim to the review-safe wording in `PITCH_PLAN.md`. Update this file at the end of each session and keep it short: detail belongs in the files above.

## Findings, one line each

1. Dicke state + filter draws the same distribution as classical rejection sampling: a correctness baseline, not an advantage.
2. Trained layers raise acceptance slightly but leave the uniform distribution; training cost is not paid back.
3. The circuit is not trivially simulable at depth (MPS bond dimension grows fast); not an advantage claim.
4. Quantum-enhanced MCMC: no evidence of advantage at n ≤ 16; a classical walk with the same oracle does as well.
5. Amplitude amplification: about 10⁴ times slower than classical at n = 100–200; break-even needs a feasible fraction near 1e-9 to 1e-12.
6. Attribution on synthetic data recovers a planted rank (69.3 ± 0.7 against 70).
7. AI mandate layer on one fictional mandate: 14 of 18 rules enforced, 3 reported as not expressible; never run live.
8. Amplitude estimation of the percentile: quadratically fewer queries in noiseless simulation, robust on fragmented tight mandates; no advantage on today's hardware and none in wall-clock time.
9. Real fund (above): rank 54 of 100, rules cost about 2 points; under those rules every random draw is valid, so there is nothing for a quantum sampler to speed up.

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
