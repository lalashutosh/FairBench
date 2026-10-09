# Handoff: Quantum x Finance Hackathon (Hanken) — End Goal

**Date written:** Fri 2026-10-09
**Hard deadline:** Saturday 2026-10-10, **09:30** (Helsinki time). "No exceptions." Submit via the organisers' submission form.

## 1. End goal

Deliver ONE `.pptx` file that contains everything we present (slides, embedded videos, demos), and rehearse a **3-minute pitch** from it, followed by **3 minutes of Q&A**. Nothing outside the pptx can be shown.

Definition of done:
- [ ] `pitch.pptx` is complete, opens cleanly, and every video/demo plays from inside the file
- [ ] Pre-recorded demo (≤ ~45 s) embedded in the deck, works with sound off (sound is optional)
- [ ] Speaker script that reads in ≤ 2:50 aloud (leaves buffer; a timer is visible and a sound plays at 00:00)
- [ ] Q&A cheat sheet (see section 6)
- [ ] File submitted through the form before 09:30 Saturday, submission confirmation saved

## 2. What the jury scores (from the organisers' slides)

**The Story**
1. The problem
2. The solution (fit to the challenge)
3. Innovativeness and value — "the secret sauce"
4. Progress — pre-event and during the event
5. Resources used

**Methodological capabilities**
- Has the team presented a demo or other simulation of the solution?
- Was quantum utilised in the solution, in practice or in theory?

**Pitch guidance (verbatim intent)**
- Problem: say what you are solving, and make it evident the problem fits the challenge.
- Solution: how you solve it, what you offer, the value to users/customers (WHY would they use it), and why it is innovative vs. existing solutions.
- Maturity and possible business model: what is the plan, how could this be taken forward.
- Hackathon journey: technical progress pre-event and during the event; resources and tools used; **avoid too much detail**.
- Demo: what is the end product right now? Pre-recorded demos are highly recommended. Sound or silent is our choice.

**Presentation rules**
- Own structure is fine, but use the criteria as the guide. We decide how many people present.
- Don't switch slides too fast. Few slides, little text per slide. No small text or small diagrams.
- Slides are visible to the audience; use the microphone.
- Timer always visible. Sound at 00:00 if not finished (alternative: clap at 1 min remaining and at 00:00).
- Q&A is 3 min; keep answers concise and don't repeat pitch content. Organisers keep the Q&A time.
- All teams must be present; teams pitch in number order.

## 3. Project context (what we know)

- Team entry in the Hanken quantum x finance hackathon; we started from the quantum side (NISQ and future algorithms that may offer an advantage in finance).
- Chosen pain point: **is an ESG fund's underperformance caused by the manager or by the ESG constraints?** This needs sampling compliant (ESG-constrained) portfolios and comparing the fund against that distribution. Classical baseline: **MCMC** constrained sampling.
- An earlier idea (certified quantum randomness) was considered and set aside.

**Claude Code: before building anything, read the repo and the notebooks/scripts to establish what actually exists** and fill in section 7. Do not describe results in the deck that the code does not produce.

## 4. Proposed 3-minute structure (~180 s)

| # | Slide | Time | Purpose / criterion |
|---|-------|------|---------------------|
| 1 | Title + one-line hook | 0:10 | Team, project name, one sentence |
| 2 | The problem | 0:25 | Manager skill vs ESG constraint is indistinguishable today; why it matters (fund selection, fees, regulation). Make fit to the challenge explicit |
| 3 | The solution | 0:30 | Sample the space of ESG-compliant portfolios, place the fund in that distribution. Value: who uses it and why |
| 4 | Secret sauce (quantum) | 0:30 | Where quantum enters (practice or theory) and why it differs from plain MCMC. Be honest about NISQ vs future advantage |
| 5 | Demo | 0:45 | Pre-recorded, large visuals, one clear result (e.g. fund return vs constrained-portfolio distribution) |
| 6 | Journey + resources | 0:20 | Pre-event vs during-event progress; tools/hardware used. High level only |
| 7 | Maturity + next steps / business model | 0:20 | Plan, who pays, how it could be taken forward |

Rules for the deck: max ~20 words per slide, large fonts, one visual per slide, 16:9, high contrast (the projector washes out light grey).

## 5. Tasks for Claude Code (in order)

1. **Audit the repo** → fill section 7 (what runs, what results exist, what hardware/simulators were used).
2. **Harden the demo**: make one script/notebook run end to end in a single command; fix anything flaky. Produce the headline figure(s) at presentation size.
3. **Record the demo**: capture a screen recording (or generate an animated figure/GIF → MP4), ≤ 45 s, silent-friendly with captions.
4. **Build `pitch.pptx`** (use the `pptx` skill) following section 4. Embed the video in the file itself, not as a link. Check that fonts are standard or embedded.
5. **Write `script.md`**: spoken text per slide with timestamps, total ≤ 2:50. Mark who speaks if more than one presenter.
6. **Write `qa.md`**: likely jury questions + 2–3 sentence answers (section 6).
7. **QA the pptx**: open/convert to PDF, check every slide renders, text is large enough, video plays, file size reasonable for the form upload.
8. **Submission checklist**: file name, size, form fields needed; leave submission itself to the human.

## 6. Likely Q&A topics to prepare

- Is there a real quantum advantage here, or is this a classical result with a quantum angle? (Answer honestly: what was run, on what, and what is theoretical.)
- How does this beat or complement MCMC / existing attribution methods (e.g. factor-based performance attribution)?
- How are "ESG-compliant" portfolios defined, and what data was used?
- How does it scale to a realistic universe size?
- Who is the customer and what is the business model?
- What would the next three months look like?

## 7. Facts to fill in from the repo (do not guess)

- Problem formulation / encoding used: ___
- Quantum method (algorithm, qubit count, simulator vs hardware, provider): ___
- Classical baseline and comparison result: ___
- Data used: ___
- Headline result for the demo: ___
- Pre-event work vs work done during the event: ___
- Team members and who presents: ___
- Tools/resources (frameworks, compute, APIs): ___

## 8. Constraints and guardrails

- Everything shown must be inside the single pptx. No live internet dependencies, no live coding.
- No claims of quantum advantage that the work does not support; a clear, honest "quantum in theory / small-scale in practice" framing scores better than overreach.
- Keep the deck readable from the back of a hall: nothing below ~24 pt.
- Time is tight: today is Friday afternoon, deadline Saturday 09:30. Prioritise in this order: working demo → clear story → polish.