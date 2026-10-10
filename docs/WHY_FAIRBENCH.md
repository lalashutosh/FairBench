# Why FairBench: judging a fund against the rules it was given

_The argument behind the project, in plain words. Numbers come from the files named beside them; the
short version is in [`README.md`](../README.md)._

## The one-minute version

A fund with ESG rules trails its index. Everyone asks whether the manager or the rules are to blame,
and the standard tools cannot say, because each compares the fund with something it was not allowed to be.

FairBench compares the fund with what it *was* allowed to be. It draws thousands of random portfolios
that follow the fund's own rules, and reports two numbers: what the rules cost against the market, and
where the real fund ranks among the portfolios the rules permit.

We ran it on nine real US funds from public SEC filings. We then showed that this rank is the kind of
quantity a quantum computer estimates with quadratically fewer queries, ran the circuit on a 50-qubit
machine, and measured how far today's hardware is from making that useful.

## A question the standard tools do not answer

Take Parnassus Core Equity from September 2019 to June 2026. In price terms it grew 135%. The S&P 500
index fund grew 149%. The fund is 14 points behind.

- **The index benchmark** stops there: the fund lost.
- **A peer group** would rank it among other large-cap funds. They follow other rules, and the funds
  that closed are missing from the list.
- **Brinson attribution** would split the 14 points into sector allocation and stock selection, against
  the S&P 500. It takes the benchmark as given. If the fund's rules forbid the sector that led, that
  shows up as a bad allocation decision.
- **A factor model** would explain the gap by style tilts. It does not ask which tilts the rules forced.
- **An ESG index** is one portfolio that follows one set of rules. It is a single path, not the range of
  outcomes those rules allow.

None of these is wrong. Each answers a different question. The missing question is the one a board
asks: *given the rules we imposed, how did this manager do?*

FairBench's answer for the same fund ([`REAL_FUND_RESULTS.md`](REAL_FUND_RESULTS.md)):

| | Price growth, 2019-09 to 2026-06 |
|---|---|
| S&P 500 index fund | +149% |
| Parnassus Core Equity | +135% |
| Typical 36-stock portfolio under the same rules | +105% |

The typical portfolio the rules allowed grew 105%. The exclusions account for about 2 points of that.
Most of the gap to the index comes from something simpler: any 36-stock portfolio tends to trail an
index that a few very large companies drove. Against what it was allowed to hold, the fund finished 30
points ahead, and quarter by quarter it ranked 54 of 100 on average, which luck alone could produce.

So the reading changes from "the manager lost 14 points" to "the mandate's shape cost about 44, the
exclusions about 2 of them, and the fund's result inside the mandate is not distinguishable from
luck". That is a different conversation for a board to have.

## The idea, and what FairBench adds to it

Using random portfolios as a yardstick is not new. Ron Surz proposed portfolio opportunity
distributions in 1994, and Patrick Burns developed performance measurement with random portfolios in
the 2000s. A commercial product, FactSet's Cabot Reveal Plus, documents counterfactual portfolios and
Monte Carlo return distributions for skill analysis, and is the closest existing neighbour.

What has been missing is the part before and after the sampling. FairBench adds:

1. **Where the rules come from.** A fund's rules live in prose: a prospectus, a policy, a mandate
   letter. FairBench reads the text into a structured rule spec and compiles that into constraints.
   Every rule keeps the sentence it came from. The code checks the quote is really in the document and
   the numbers are really in the quote. Wording is classed as hard ("will not") or soft ("seeks to"),
   and only hard, reviewed rules are enforced. Rules with no numeric form, or that need data nobody
   outside the fund has, are listed as such. Nothing is dropped silently.
2. **What the fund's region adds.** The US names rule, EU fund-name guidelines and Paris-aligned
   benchmark exclusions are checked against the mandate, and every rule is written out as a formula a
   reviewer can sign (`results/mandate_formula_sheet.md`).
3. **Data anyone can get.** Holdings, the investable universe, benchmark weights and prices all come
   from the SEC's free Form N-PORT filings. The universe and its weights are taken from an index
   fund's own filing on the same date; prices are value divided by shares. A result can be reproduced
   without a licence for an index, a price feed or an ESG rating.
4. **Proof that the yardstick works.** A method like this is only worth something if it flags a fund
   that truly differs and stays quiet otherwise. On real data, an equal-weighted index fund (which is
   built differently from its value-weighted universe by design) is flagged in 13 of 26 quarters. Eight
   ESG funds are flagged in 0 of 198 (`results/skill_table/`). On synthetic data with a planted answer,
   the planted rank of 70 comes back as 69.3 ± 0.7. On small universes every sampler is scored against
   the exact answer.
5. **A quantum estimator for the same number.** This is the next section.

## Why this is the right problem for a quantum computer

Many quantum-finance projects pick portfolio *optimisation*: find the single best portfolio with QAOA,
VQE or an annealer. Those are heuristics. No speedup is proven for them, and a good classical solver
is hard to beat.

FairBench needs something else. The fund's rank is a fraction: of all the portfolios the rules allow,
what share did worse than the fund? Estimating a fraction of a set is what **quantum amplitude
estimation** does, with a proven quadratic advantage in the number of queries (Brassard, Høyer, Mosca
and Tapp 2002; Montanaro 2015 for Monte Carlo in general).

Three design choices make it fit:

- **The rules are built into the circuit's starting state.** A Dicke state is an equal superposition of
  every portfolio with exactly k holdings. The circuit cannot output a portfolio of the wrong size.
- **The fund's return goes inside the oracle as a threshold.** The comparison "did this portfolio do
  worse than the fund?" happens in superposition, and the rank comes out as one amplitude. No list of
  portfolios is ever produced.
- **Tight rules help instead of hurting.** Classical rejection sampling wastes every draw that breaks a
  rule, so its cost grows as 1/P_F, where P_F is the share of portfolios the rules allow. The quantum
  cost grows as 1/√P_F. Classical MCMC avoids the waste by walking between valid portfolios, but tight
  rules split the valid set into islands (14 to 19 of them in our tightest cases) and the walk gets
  trapped. Amplitude estimation does not need the islands to be connected.

What we measured ([`QUANTUM_CORE.md`](QUANTUM_CORE.md), noiseless simulation, query counts):

| | Result |
|---|---|
| Error against queries | falls as 1/queries (fitted slope −1.00), against 1/√queries classically (−0.48) |
| Whole attribution, tightest rules (0.23% of portfolios allowed) | 12× fewer queries than classical sampling |
| Whole attribution, loose rules (36% allowed) | classical sampling is 1.5× better |
| Against swap-MCMC | 2–9× fewer queries |
| Real fund's 27 quarters, rank to ±1 point | median 1,900 queries against 5,366 classical samples |
| Circuit for the full universe (476 stocks) | about 949 error-corrected qubits, 5.0 million T gates per step |

## What a real quantum computer said

We ran the constraint-preserving circuit on VTT's 50-qubit Q50 through LUMI
([`QUANTUM_CORE.md`](QUANTUM_CORE.md), section 13).

- It works at small depth: with 4 assets it output valid portfolios 54% of the time, against 37.5% for
  random bits.
- It stops working at depth: one amplitude-estimation step is about 150 two-qubit gates even for four
  assets, and after it most of the signal is gone.
- A digital twin with one fitted parameter (a two-qubit error of 1.8%) reproduces what the machine did.
  Turning that dial down shows the toy problem working at about 0.1% error, and real portfolio sizes
  needing about 0.00001%. That is error-corrected hardware.

So the hardware run does not show an advantage. It turns "this needs a better machine" into a measured
requirement.

## What we ruled out

The first plan was the obvious one: use a quantum circuit to *generate* rule-abiding portfolios faster
than a classical sampler. We tested four versions against strong classical baselines.

| Route | Verdict |
|---|---|
| Dicke state, then filter | The same distribution as classical rejection sampling, at the same cost |
| Trained circuit layers | A little more acceptance, paid for with biased samples and training cost |
| Quantum-enhanced MCMC | A classical walk given the same information does at least as well |
| Amplitude amplification | About 10⁴× slower than a laptop under fault-tolerant cost assumptions |

These results are why the project estimates instead of samples. A generated sample is one random draw
however it was produced, so it cannot improve precision. An estimate can. Finding the place where the
quantum method has a real edge meant first showing where it has none.

## What FairBench does not claim

- **Not a measure of skill.** A rank describes one period. Over 27 quarters only an average above about
  61 or below about 39 stands out from luck.
- **Not a causal split.** "The rules cost X" compares two typical portfolios; it is a description.
- **Not the fund's exact mandate.** On real funds the rules are a proxy built from public data. A
  fund's own ESG research cannot be reproduced from outside.
- **Not a quantum speedup today.** The query advantage is a noiseless-simulation result for
  fault-tolerant machines, and fewer queries is not less time: one quantum query costs far more than
  one classical check.
- **Not a replacement** for an index benchmark, a peer group or Brinson attribution. It sits beside them.

## Who would use it

An asset owner, fund-of-funds or consultant that reviews external equity managers every quarter
against mandates with ESG, carbon, sector or tracking-error rules. That review is a spreadsheet today.
With the institution's own mandate documents and holdings, the proxy rules above become the real ones,
and the report says what the mandate cost, where the manager ranked inside it, and which rules could
not be checked. Buyers, vendors and regulations are covered, with sources, in
[`RESEARCH.md`](RESEARCH.md), Part A.

## References

- R. Surz, "Portfolio Opportunity Distributions: An Innovation in Performance Evaluation", *Journal of Investing*, 1994.
- P. Burns, "Performance Measurement via Random Portfolios", Burns Statistics working paper, 2004.
- G. Brassard, P. Høyer, M. Mosca, A. Tapp, "Quantum Amplitude Amplification and Estimation", 2002.
- A. Montanaro, "Quantum speedup of Monte Carlo methods", *Proceedings of the Royal Society A*, 2015.
- D. Grinko, J. Gacon, C. Zoufal, S. Woerner, "Iterative quantum amplitude estimation", *npj Quantum Information*, 2021.
- A. Bärtschi, S. Eidenbenz, "Deterministic Preparation of Dicke States", 2019.
- R. Babbush et al., "Focus beyond Quadratic Speedups for Error-Corrected Quantum Advantage", *PRX Quantum*, 2021.
