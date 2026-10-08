# FairBench: plan from now to the pitch

_Written 2026-10-07 from the repo state (`fairbench_handoff.md`, `BUILD_PLAN.md`, `README.md`). The hackathon runs 8–10 Oct, so this assumes the pitch is on the last day and about two and a half working days remain. I did not have access to the team chat, so anything decided there (pitch slot length, judging criteria, data the organisers provide, who owns what) is marked **[confirm]**. Fix those first._

_Updated 2026-10-08 morning: the quantum story changed overnight (QA wave, now on `main`). Items marked ✅ are done; the quantum framing, priorities and Q&A below reflect the new result._

## Where we are

Built and tested (680 offline tests): the sampling engine, ten constraint types, the attribution app, the AI mandate layer, and five honest quantum studies. The deck has a first draft.

**Done overnight 7→8 Oct (QA wave):** quantum amplitude estimation of the attribution statistics.
- The fund's percentile is a ratio of two amplitudes over the Dicke state.
- **Precision:** in noiseless simulation, error falls as 1/queries (quantum) vs 1/√queries (classical).
- **Example mandate:** 7–20× fewer queries than rejection sampling at ±1 percentile point for the percentile.
- **Whole attribution** (with a classical warm start for the medians): about 2× fewer queries at a 3% feasible fraction and 4–16× on tighter rules. Loose rules are at parity or slightly worse. It is 2–9× fewer than swap-MCMC, which gets trapped on fragmented tight mandates.
- **Limits:** no advantage on today's noisy hardware (about 17k two-qubit gates per step at 16 assets), and no wall-clock advantage even fault-tolerant.
- **Pitch assets:** slide figure `results/qae_pitch_figure.png` and offline demo `scripts/demo_qae.py` (~2 s).
- **Reviews:** three read-only reviews, all findings fixed. Numbers and the reviewed pitch paragraph are in `BUILD_PLAN.md` → "QA results".

What is still missing, in order of how much a judge would care:

1. **No real fund or real data.** Every number is synthetic with a planted answer.
2. **The AI layer has never made a live API call.**
3. ✅ **Quantum story upgraded.** It is no longer just "we tested it and found no advantage". It is now "quadratically fewer queries for the quantity the tool reports, robust where classical MCMC fails; a fault-tolerant-era result we can quantify today". It still has to be told precisely: query counts in simulation, not a speedup.
4. Deck gaps: no AI-layer slide, team names and contact are placeholders, not cut to the time limit, layout never checked visually.
5. ✅ The oracle now covers `CountBound`, `AvgBound` and a linear return rule. Still not encodable: min-groups and the two quadratic risk caps, so the example mandate's quantum numbers assume a hypothetical oracle for two of its rules. The penalty Hamiltonian still covers only the original five.

## The pitch in one paragraph (the spine for everything below)

> When an ESG fund underperforms, nobody can say whether it is the manager or the rules. FairBench reads the fund's policy text with an AI layer, turns it into exact constraints, samples thousands of portfolios that obey the same rules, and splits the fund's result into *what the rules cost* and *what the manager added*. The sampler is pluggable. The quantity the tool reports, the fund's percentile among rule-abiding portfolios, is exactly what quantum amplitude estimation is built for. In simulation it needs quadratically fewer queries than classical sampling, and it keeps working on tight mandates where classical MCMC gets trapped. We say plainly that this is a fault-tolerant-era result, not a speedup on today's machines, and we show where the break-even sits.

Every claim stays inside the review-safe wording in `fairbench_handoff.md` ("Key findings") and `BUILD_PLAN.md` → "QA results" (which has a "must not say" list). Never say "quantum speedup".

## Priorities

| Tier | Item | Why |
|---|---|---|
| Must | Live API run of the mandate layer | The headline AI claim is untested without it |
| Must | One real-data attribution, or an explicitly labelled fallback | Biggest credibility gap |
| Must | Deck finished and timed; two full rehearsals | The pitch is the deliverable |
| Must | Demo that cannot fail on stage (offline fallback, saved outputs) | Live demos break |
| Should | Q&A sheet with honest answers on quantum | Judges will probe this first |
| Should | Hardware or vendor-backend feasibility run, if the organisers give access **[confirm]** | Shows the circuit runs on real devices |
| ✅ Done | Extend oracle to `CountBound` and `AvgBound` (+ linear return rule) | Done in QA-2; Hamiltonian not extended (not needed for the pitch) |
| ✅ Done | Quantum angle on the finance quantity itself (QAE) | The headline quantum slide; see "Where we are" |
| Cut | Optimizer pivot (`apps/optimizer.py`) unless validation fails | Only a fallback |
| Cut | New quantum experiments | Five routes tested; QAE is the story. Only a hardware feasibility run if access is offered |

## Schedule

### Day 1 — Wed 7 Oct (today): unblock and decide

- [ ] **[confirm]** Pitch slot length, Q&A length, judging criteria, whether a live demo is allowed, and what data or hardware the organisers provide. Cut the plan to fit.
- [ ] **[confirm]** Answer the open problem-validation question: does the manager-vs-rules framing hold up with anyone who works with ESG funds (a mentor, a judge, a hackathon partner)? One short conversation. If it clearly fails, switch to the optimizer pivot now, not on day 3.
- [ ] Assign owners: AI layer and live run; real data; deck; demo and rehearsal; Q&A sheet.
- [ ] `git pull`, run the test suite, run `scripts/demo_attribution.py` and `scripts/demo_mandate.py` once so everyone knows the baseline works on their machine.
- [ ] **Live API run.** Teammate sets `ANTHROPIC_API_KEY` in their shell (never committed), then `scripts/demo_mandate.py --live`. Compare Claude's spec to `examples/mandate_example.rules.json`. If it differs, fix the prompt in `fairbench/mandate.py`, re-run, and save `results/mandate_rules_live.json`. Update the validation-status lines in the handoff and README.
- [ ] Start the real-data search (see Day 2). Timebox the hunt to two hours.

### Day 2 — Thu 8 Oct: real data and the story

**Real data (one owner, hard timebox: noon Day 2 to decide go or fallback).**

- [ ] Pick one real ESG fund with a public policy document (prospectus or SFDR pre-contractual disclosure) and public holdings. Prefer a European equity fund with a modest number of holdings (a focused fund of 20–60 names keeps the rules binding and the sampling cheap).
- [ ] Build `universe.csv` (ticker, mu, sector, esg_score, carbon, plus `flag_*`, `attr_*`, `cat_*` for what the mandate mentions) and `returns.csv` (prices to simple returns over a clear window). Free sources for prices; ESG and carbon scores are the hard part. Check what the organisers provide first **[confirm]**.
- [ ] Run the live mandate extraction on the real policy text. Expect new rule types: add them to `RULE_FIELDS`, `compile_spec` and `SYSTEM_PROMPT`, with a test each. Every unmapped sentence is shown on a slide as a feature, not hidden.
- [ ] Run `attribute(...)` on the real fund. Record: percentile, constraint effect, manager effect, feasible fraction, and what the fund's holdings break (`check_fund`).
- [ ] **Fallback if real data is not usable by noon:** keep the synthetic planted-truth case as the main result and say so on the slide ("validated on planted ground truth; real-fund run is the next step"). Do not present synthetic numbers as a real fund. Do not burn Day 3 on data.
- [ ] Whatever the outcome, state the caveats on the slide: static holdings, Monte Carlo CI only, one window's percentile is not proof of skill, equal-weight assumption.

**Story and deck (second owner, in parallel).**

- [ ] Add the AI-layer slide: mandate sentence → rule → compiled constraint, with the `unmapped` rules shown.
- [ ] Fill team names and contact. Cut to the time limit: one idea per slide, the headline result early, quantum findings as one clear "what we tested" slide plus the break-even slide.
- [ ] Check layout visually (never done). Update any number that changes after re-runs (deck slides 4, 5, 9).
- [x] Decide the quantum framing. Recommended, now with QAE: *"The fund's percentile is an amplitude. Quantum amplitude estimation over our constraint-preserving circuit needs quadratically fewer queries than classical sampling, and on tight mandates, where the valid portfolios split into islands that trap MCMC, it is the only method that stays reliable. That is the fault-tolerant-era case; on today's hardware there is no advantage, and we show exactly why."* Use the reviewed paragraph in `BUILD_PLAN.md` → "QA results" verbatim for anything written.
- [ ] Replace the quantum slides with one slide built on `results/qae_pitch_figure.png`, plus one "what we tested and where it does not help" slide (Dicke ≡ rejection, trained layers, QeMCMC, AA break-even, NISQ thresholds).

### Day 3 — Fri 9 Oct: freeze, rehearse, harden

- [ ] **Code freeze at a fixed time (suggest early afternoon).** After it, no code changes, only deck and rehearsal.
- [ ] Final run of every script that feeds a slide; regenerate `results/` and confirm the deck numbers match the files.
- [ ] **Demo hardening.** Primary: `scripts/demo_mandate.py` with the reference spec (offline, ~15 s). Quantum demo: `scripts/demo_qae.py` (offline, ~2 s; loose vs tight rules, errors vs exact). Optional live Claude call only if the network and key are confirmed on the stage machine. Backup: pre-rendered PNGs and a screen recording. Test on the actual presentation laptop.
- [ ] Two full timed rehearsals out loud, one with a teammate playing a skeptical judge. Cut anything that runs over.
- [ ] Write the Q&A sheet (below) and have each question assigned to one answerer.
- [ ] Export the deck to PDF as a backup. Final `git status` clean; push when asked.
- [ ] Sleep.

### Day 4 — Sat 10 Oct: pitch

- [ ] Morning: one last rehearsal, laptop and adapter check, demo smoke test, backups on a USB stick and in the cloud.
- [ ] Present. Keep the first 30 seconds on the problem, not the quantum circuit.

## Optional work, only if a Must slips nowhere

- **Hardware feasibility run** (`ibm` / `vtt` backends are stubs that raise `NotImplementedError`). Only worth it if the organisers provide access **[confirm]**. Frame it as feasibility: a small Dicke circuit runs, report the fraction of shots that stay in the weight-k subspace. Not an advantage claim.
- ✅ **Oracle for `CountBound` and `AvgBound`.** Done (QA-2). The two risk caps and min-groups remain stated limits.
- **Quantum-walk speedup of MCMC** (Szegedy / Montanaro). Not explored, and only worth a sentence in Q&A.

## Likely judge questions and honest answers

| Question | Answer |
|---|---|
| Where is the quantum advantage? | In query count, on the quantity the tool reports. Amplitude estimation of the fund percentile needs quadratically fewer oracle queries than classical sampling (noiseless simulation, fitted slopes −1.0 vs −0.5). On the example mandate that is 7–20× fewer queries for the percentile; for the whole attribution it is 2–16× on tight rules and parity on loose ones. It is not a wall-clock speedup today: on noisy hardware there is no advantage, and even fault-tolerant it is ≥ 8e3× slower at ±1 point. |
| Then why quantum at all? | Tight mandates are where classical tools struggle. Rejection sampling pays 1/P_F, and MCMC gets trapped when the valid portfolios split into islands (we measured 14–19 islands and 26–43% errors). Amplitude estimation pays √(1/P_F) and needs no connectivity. The application runs classically today, with the quantum estimator as a drop-in once fault-tolerant hardware exists. |
| Isn't that just textbook amplitude estimation? | Yes, deliberately. Our work is (1) showing the attribution statistics are amplitudes over a constraint-preserving state, (2) a verified oracle for the fund's rules and return threshold, and (3) honest baselines: rejection, swap-MCMC, exact-counting DP, noise and fault-tolerant cost models. |
| Why not a smarter classical method? | We tried. Swap-MCMC matches the √ scaling only while the feasible set is connected. An exact-counting DP is intractable once three or more weighted rules combine (the example mandate needs > 10¹⁶ states). |
| What did you NOT show? | Real hardware runs; real fund data; rules with no oracle (min-groups, risk caps); quantum-walk alternatives. |
| Is the result on a real fund? | State the truth: real fund if Day 2 succeeds, otherwise synthetic with planted ground truth that the tool recovers (percentile 69.3 ± 0.7 against a planted 70). |
| Can you trust the AI layer? | The model only translates language into a fixed rule vocabulary; it never sees per-asset data. Each rule carries a verbatim quote that code checks; thresholds are computed by deterministic code; rules it cannot express are reported, not dropped. Say how many real mandates were tried. |
| What does a percentile prove? | Not skill. One window, static holdings, Monte Carlo error only. It tells you whether the fund sits in the body or the tail of what its own rules allow. |
| Who would use this? | Allocators, fund-of-funds, regulators and the funds themselves, to separate "ESG cost" from "selection skill". **[confirm]** against anything learned from validation conversations. |
| What breaks it? | Mandates with weighting schemes other than equal weight, time-varying targets, letter ratings, or sentences with no data behind them. These come back as `unmapped`. |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| No usable real ESG data | Fallback decided by noon Day 2; synthetic case labelled as such |
| Live API call fails or the network is bad on stage | Reference spec path is offline; recorded backup |
| Prompt change moves the numbers | Re-run all demo scripts after the last change and update slides 4, 5, 9 |
| Over-claiming quantum | Reviewed wording only (`BUILD_PLAN.md` → "QA results", with its "must not say" list). Q&A sheet. One person checks the final deck against it: always say "queries", "noiseless simulation" and "synthetic data" |
| Memory crash on the dev laptop (happened before) | Use the capped commands in the handoff; do not run `aa_estimate.py` or the sweeps again unless a slide needs it |
| Team burnout on the last night | Code freeze on Day 3; no new features after it |

## Definition of done for the pitch

- Live mandate run recorded, or the deck says plainly that it has not been run live.
- The headline attribution is labelled correctly as real or synthetic.
- Deck fits the time limit, has names and contact, and every number matches a file in `results/`.
- Two timed rehearsals done; Q&A answers assigned.
- A backup of the demo works on the presentation laptop.
