# FairBench: plan from now to the pitch

_Written 2026-10-07 from the repo state (`fairbench_handoff.md`, `BUILD_PLAN.md`, `README.md`). The hackathon runs 8–10 Oct, so this assumes the pitch is on the last day and about two and a half working days remain. I did not have access to the team chat, so anything decided there (pitch slot length, judging criteria, data the organisers provide, who owns what) is marked **[confirm]**. Fix those first._

## Where we are

Built and tested (578 offline tests): the sampling engine, ten constraint types, the attribution app, the AI mandate layer, and four honest quantum benchmarks (none shows a speedup). The deck has a first draft.

What is still missing, in order of how much a judge would care:

1. **No real fund or real data.** Every number is synthetic with a planted answer.
2. **The AI layer has never made a live API call.**
3. **The quantum story is "we tested it and found no advantage".** That is honest and defensible, but it has to be framed as a result, not an apology.
4. Deck gaps: no AI-layer slide, team names and contact are placeholders, not cut to the time limit, layout never checked visually.
5. The quantum penalty Hamiltonian and oracle only cover the original five constraint types.

## The pitch in one paragraph (the spine for everything below)

> When an ESG fund underperforms, nobody can say whether it is the manager or the rules. FairBench reads the fund's policy text with an AI layer, turns it into exact constraints, samples thousands of portfolios that obey the same rules, and splits the fund's result into *what the rules cost* and *what the manager added*. The sampler is pluggable: we built and benchmarked a constraint-preserving quantum sampler against strong classical baselines, we report where it does not help, and we give the break-even point where a fault-tolerant machine would.

Every claim stays inside the review-safe wording in `fairbench_handoff.md` ("Key findings"). Never say "quantum speedup".

## Priorities

| Tier | Item | Why |
|---|---|---|
| Must | Live API run of the mandate layer | The headline AI claim is untested without it |
| Must | One real-data attribution, or an explicitly labelled fallback | Biggest credibility gap |
| Must | Deck finished and timed; two full rehearsals | The pitch is the deliverable |
| Must | Demo that cannot fail on stage (offline fallback, saved outputs) | Live demos break |
| Should | Q&A sheet with honest answers on quantum | Judges will probe this first |
| Should | Hardware or vendor-backend feasibility run, if the organisers give access **[confirm]** | Shows the circuit runs on real devices |
| Could | Extend oracle/Hamiltonian to `CountBound` and `AvgBound` | Mechanical, closes a stated gap |
| Cut | Optimizer pivot (`apps/optimizer.py`) unless validation fails | Only a fallback |
| Cut | New quantum experiments | Four routes already tested; more adds no pitch value |

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
- [ ] Decide the quantum framing (recommended): *"A constraint-preserving circuit that provably samples only valid portfolios, tested against fair classical baselines. No advantage at today's sizes, and here is exactly where the break-even is."* This is a result a technical judge can trust.

### Day 3 — Fri 9 Oct: freeze, rehearse, harden

- [ ] **Code freeze at a fixed time (suggest early afternoon).** After it, no code changes, only deck and rehearsal.
- [ ] Final run of every script that feeds a slide; regenerate `results/` and confirm the deck numbers match the files.
- [ ] **Demo hardening.** Primary: `scripts/demo_mandate.py` with the reference spec (offline, ~15 s). Optional live Claude call only if the network and key are confirmed on the stage machine. Backup: pre-rendered PNGs and a screen recording. Test on the actual presentation laptop.
- [ ] Two full timed rehearsals out loud, one with a teammate playing a skeptical judge. Cut anything that runs over.
- [ ] Write the Q&A sheet (below) and have each question assigned to one answerer.
- [ ] Export the deck to PDF as a backup. Final `git status` clean; push when asked.
- [ ] Sleep.

### Day 4 — Sat 10 Oct: pitch

- [ ] Morning: one last rehearsal, laptop and adapter check, demo smoke test, backups on a USB stick and in the cloud.
- [ ] Present. Keep the first 30 seconds on the problem, not the quantum circuit.

## Optional work, only if a Must slips nowhere

- **Hardware feasibility run** (`ibm` / `vtt` backends are stubs that raise `NotImplementedError`). Only worth it if the organisers provide access **[confirm]**. Frame it as feasibility: a small Dicke circuit runs, report the fraction of shots that stay in the weight-k subspace. Not an advantage claim.
- **Oracle and Hamiltonian for `CountBound` and `AvgBound`.** Same shape as `SectorCap` and `MinESG`; a few hours with tests. Closes the "quantum side only covers five constraint types" gap. The two risk caps are quadratic and would need new circuitry, so leave them as stated limits.

## Likely judge questions and honest answers

| Question | Answer |
|---|---|
| Where is the quantum advantage? | We did not find one, and we say so. Dicke + filter equals rejection sampling; trained layers and quantum MCMC gave no gain over fair classical baselines at n ≤ 16; amplitude amplification is ~10⁴× slower at n = 100–200 and breaks even only when the feasible fraction is below ~1e-9 to 1e-12. Our contribution is the rigorous test, the verified oracle and the break-even map. |
| Then why quantum at all? | The sampler is a drop-in backend; the application works today classically. The break-even analysis says which rule sets (very tight, large universes) would favour a fault-tolerant machine. We did not evaluate those sizes. |
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
| Over-claiming quantum | Review-safe wording only; Q&A sheet; one person checks the final deck against `fairbench_handoff.md` |
| Memory crash on the dev laptop (happened before) | Use the capped commands in the handoff; do not run `aa_estimate.py` or the sweeps again unless a slide needs it |
| Team burnout on the last night | Code freeze on Day 3; no new features after it |

## Definition of done for the pitch

- Live mandate run recorded, or the deck says plainly that it has not been run live.
- The headline attribution is labelled correctly as real or synthetic.
- Deck fits the time limit, has names and contact, and every number matches a file in `results/`.
- Two timed rehearsals done; Q&A answers assigned.
- A backup of the demo works on the presentation laptop.
