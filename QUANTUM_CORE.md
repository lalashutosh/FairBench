# FairBench quantum core: how it works, where the advantage is, how to plug into it

_Written 2026-10-09. Audience: teammates building or connecting the classical layers (data, mandates, real-fund pipeline) and anyone answering quantum questions in the pitch. Numbers come from `results/qae_*`. Full study log: `BUILD_PLAN.md` → "QA results". Reviewed pitch wording and the "must not say" list are there too._

---

## 1. One-paragraph summary

FairBench ranks a fund among all the portfolios its own rules allowed. That rank (the percentile) and the two medians behind the constraint and manager effects are **fractions of a huge set of portfolios**. A fraction of a set is exactly what **quantum amplitude estimation (QAE)** measures.

In noiseless simulation, QAE reaches a given precision with **quadratically fewer oracle queries** than classical sampling:
- error ∝ 1/queries, against 1/√queries classically;
- queries grow as 1/√P_F, against 1/P_F for rejection sampling, where P_F is the share of portfolios the rules allow;
- it does **not** need the allowed portfolios to be connected, which is where classical MCMC breaks on tight mandates.

It is a **fault-tolerant-era capability**:
- on today's noisy hardware there is no advantage;
- even on fault-tolerant hardware, one quantum query is about 10⁴× more expensive than a laptop check, so there is no wall-clock speedup at analyst precision.

The classical pipeline stays the product today. The quantum core is a drop-in estimator for the few decisive numbers.

---

## 2. What the quantum core computes

```
classical inputs (universe, rules, fund, returns)
        │
        ▼
1. STATE PREP   Dicke state |D_k>: equal superposition of every portfolio with exactly k holdings
                (one qubit per asset; qubit i = 1 means "asset i held")
        │
        ▼
2. ORACLE       reversible circuit that flips the sign of a portfolio if it is
                  F: valid under the rules                          (run 1)
                  G: valid AND its return is below the fund's return (run 2)
        │
        ▼
3. GROVER STEP  Q = A·S0·A†·S_chi : a rotation by 2θ, where sin²θ = fraction marked
        │
        ▼
4. AMPLITUDE ESTIMATION (IQAE)   run Q^m for growing m, measure, narrow θ
        │
        ▼
5. CLASSICAL POST-PROCESSING
        a_F = |F| / C(n,k)        rule tightness
        a_G = |G| / C(n,k)
        percentile = a_G / a_F    fund's rank among rule-abiding portfolios
        null median, benchmark median: classical warm start (≈10 samples) + AE refinement
        constraint effect = null median − benchmark median
        manager effect    = fund − null median
```

**We never generate a list of portfolios.** The fund's return goes *into the oracle* as a threshold, so the comparison happens inside the superposition and the rank comes out directly. Histograms and example portfolios on screen still come from classical sampling (section 5).

---

## 3. Where the advantage comes from

**Not from "trying all portfolios at once".** Measuring the Dicke state straight away gives one random portfolio, exactly like a classical draw. Dicke plus filtering is provably the same distribution as classical rejection sampling.

**From coherent amplification.**
- The Grover step rotates the state by an angle 2θ that encodes the fraction we want.
- Repeating it m times *before* measuring multiplies the angle: P(good) = sin²((2m+1)θ).
- A shot after m steps costs m+1 queries but carries information that grows like (2m+1)². Information per query rises with circuit depth.
- Over N total queries the error falls like **1/N instead of 1/√N**. It's the same reason measuring a stack of 1,000 sheets beats measuring one sheet.

**Two sources of gain, one mechanism:**

| Gain | Classical (rejection) | Quantum (QAE) |
|---|---|---|
| Precision ε | ∝ 1/ε² | ∝ 1/ε |
| Rule tightness P_F | ∝ 1/P_F (illegal draws are wasted) | ∝ 1/√P_F (legal fraction is an amplitude) |

**Measured** (noiseless simulation, synthetic data; one query = one oracle call = one classical sample):

| Result | Value | Source |
|---|---|---|
| Error versus queries, fitted slope | −1.00 (IQAE) vs −0.48 (classical) | `qae_scaling.json` |
| Queries versus rule tightness, exponent | ≈ −0.5 (quantum) vs ≈ −1.2 (rejection) | `qae_scaling.json` |
| Example mandate (n = 150, P_F = 0.55%), percentile at ±1 point | 7–20× fewer queries than rejection | `qae_mandate.csv` |
| Whole attribution (percentile + both medians), typical run | 0.65× at P_F 36%, 1.1× at 10%, 2.4× at 3.4%, 4.1× at 0.7%, 12× at 0.23% | `qae_hybrid.json` |
| Same, 9 in 10 runs | 1.0×, 1.7×, 3.1×, 6.9×, 15.6× | `qae_hybrid.json` |
| Versus swap-MCMC (typical run) | 2–9× fewer queries | `qae_hybrid.json` |

---

## 4. Where the classical methods fall short (and where they don't)

| Classical method | Where it struggles | Where it is still better |
|---|---|---|
| **i.i.d. rejection sampling** (the attribution app's default) | Pays 1/P_F: on the example mandate, ±1 point takes about 1.4M draws. Pays 1/ε² for precision. | Loose rules; the full distribution from one sample set; wall-clock today (about 2 µs per draw). |
| **Swap-MCMC** (walk between legal portfolios) | Tight rules split the legal set into islands (14–19 at P_F ≤ 0.23%). Chains get trapped, with 90%-of-runs errors of 0.26–0.43. Correlated samples cost extra. | Connected legal sets: it matches QAE's 1/√P_F scaling there, and only the precision gain remains. |
| **Exact counting** (dynamic programming over sectors and binned sums) | The state space explodes with three or more weighted rules (ESG, carbon, return): about 1.5e12 cells for a guaranteed ±1 point at n = 100. The example mandate needs over 10¹⁶. | One or two weighted rules: cheap and exact. |
| **Exhaustive enumeration** | C(n,k) is astronomical (C(500,36) ≈ 10⁵⁵). | Tiny universes only. |
| **Optimisers** (MILP, QUBO, annealing) | Answer a different question: they find one portfolio, not a ranking (see `PITCH_PLAN.md` Q&A). | — |

Not tested: importance sampling and quasi-Monte Carlo. A reviewer may raise them; neither removes the 1/ε² precision cost of sampling.

---

## 5. Where it does NOT help (say this before a judge does)

- **Generating portfolios.** Quantum sampling (Dicke filtering, amplitude amplification, trained circuits, quantum MCMC) gave no practical advantage. Amplitude amplification only breaks even when fewer than about 1e-9 of portfolios are legal. Each sample is still one random draw, so it never improves precision.
- **The full distribution.** One classical sample set gives every quantile at once; each quantum number is a separate run.
- **Loose rules.** For the whole attribution at P_F ≥ 10%, classical is at parity or better.
- **Hardware today.**
  - One Grover step at n = 16 is about 1.7e4 two-qubit gates on 74 qubits.
  - An advantage needs two-qubit error rates around 1e-6; today's devices are around 1e-3.
  - The signal dies before one step.
- **Fault-tolerant wall-clock.** Each quantum query takes about 0.1–1 s against about 2 µs classically. At ±1 point that is at least 8,000× slower, and there is no break-even at any useful precision.
- **Non-linear statistics and rules** (section 6.4).

---

## 6. Input contract: what the classical layers must hand over

### 6.1 The objects

| Input | Type | Used for | Notes |
|---|---|---|---|
| Universe `u` | `fairbench.data.Universe` | Asset order, rule values | Asset order defines qubit order and must be identical in `u`, `cs` and the fund vector. |
| `u.returns` | `DataFrame (T, n)`, **simple** period returns | Return comparator | The core uses g_i = Π_t(1 + r_it) over the evaluation window. One window per call. |
| `u.sector`, `u.esg_score`, `u.carbon` | per-asset arrays | SectorCap, MinESG, CarbonCap | |
| `u.attributes[name]` | `(n,) float` | AvgBound | Read via `u.numeric(name)`. |
| `u.flags`, `u.categories` | bool or labels | Must be **compiled to index sets** first | Becomes `Exclusion(indices)` or `CountBound(indices, min, max)`. The quantum core never reads flags directly. |
| Rule set `cs` | `ConstraintSet` | Oracle | **Must contain `Cardinality(k)`**, because the Dicke state fixes k. A holdings range must be pinned to the fund's actual k. |
| Fund | `(n,)` 0/1 holdings vector, **or** a float score s | Threshold in the G oracle | With a vector, the core computes the fund's equal-weight buy-and-hold return. Pass s directly to use another, linear statistic (6.3). |

### 6.2 Rules the oracle can encode

Defined in `ft/oracle.py` as `ENCODED_RULES`:
- **Counting rules (exact):** `Cardinality`, `Exclusion`, `SectorCap`, `CountBound`.
- **Weighted-sum rules (integer-quantised, 11 bits by default):** `MinESG`, `CarbonCap`, `AvgBound`, `LinearThreshold`.

Cannot be encoded (the oracle raises `TypeError`):
- **`MinGroups`:** not a threshold on a sum.
- **`VolatilityCap`, `TrackingErrorCap`:** quadratic in the selection.
- **Weight-level rules** (`PositionBound`, `WeightSum`, `HoldingsRange` from the Stage 2 weight grid): the oracle acts on selections, not weights.

**What to do with a rule it can't encode:**
- (a) Leave it out of the quantum rule set and enforce it by classical filtering of decoded results, stating the change in P_F; or
- (b) report the quantum numbers as "assuming an oracle for these rules". The example mandate does this.

**Quantisation.**
- Weighted rules are rounded to integers. `quantise(..., mode="superset")` never wrongly rejects a legal portfolio, and a float post-check removes the extras. `mode="subset"` never wrongly accepts.
- Running both brackets the true value at twice the cost.
- The bias floor on the percentile is about 0.1–0.4 points.

### 6.3 Statistics: what can be compared against the fund

The G oracle needs "portfolio statistic < fund statistic" to be a **linear inequality in the 0/1 selection x**.

| Statistic | Linear? | How to express it |
|---|---|---|
| Equal-weight, buy-and-hold total return | ✅ | Built in: `LinearThreshold(g, k*(1+s), "<")` |
| **Value- or cap-weighted** buy-and-hold total return, with weights c_i ∝ start-of-window value | ✅ | Σ x_i c_i g_i / Σ x_i c_i − 1 < s ⇔ Σ x_i c_i (g_i − 1 − s) < 0. Use `LinearThreshold(c*(g-1-s), 0.0, "<")`. Verified exactly on 20k random portfolios. |
| Average ESG, carbon or any per-asset attribute | ✅ | `AvgBound` / `LinearThreshold` |
| Rebalanced return (weights reset each period) | ❌ | Product over periods of sums: not linear. |
| Sharpe ratio, volatility, tracking error | ❌ | Quadratic or ratio of non-linear terms. |
| Weight-grid portfolios (Stage 2) | Partly | The return is linear in the weight *bits*, but uniform state preparation over the grid needs amplitude amplification on the QUBO encoding first. Not built. |

**Implication for the real-fund pipeline.**
- The Parnassus run reports value-weighted, quarterly-held results. The **buy-and-hold, cap-weighted percentile within a quarter is quantum-expressible today** via the linearisation above.
- Multi-quarter rebalanced paths are not, so run one QAE call per quarter window.

### 6.4 Outputs

`qae_attribute(...)` returns a `QAEAttribution` with:
- `percentile` (0–100, like `AttributionResult`);
- `null_median`, `benchmark_median`, `constraint_effect`, `manager_effect` and `fund` (returns as fractions);
- `queries` (per statistic and total);
- `exact` (when enumerable);
- `classical` (same-budget classical estimates).

`demo_bridge.attribute_qae(u, cs, fund, eps)` puts exact, classical and quantum side by side.

### 6.5 Calling it

```python
import numpy as np
from fairbench.constraints import ConstraintSet, Cardinality, Exclusion, SectorCap
from fairbench.qae.attribution_qae import qae_attribute

# u: Universe with u.returns over ONE evaluation window (simple returns, T x n)
cs = ConstraintSet([Cardinality(k), Exclusion(excluded_idx), SectorCap("S3", 5)])
res = qae_attribute(u, cs, fund_holdings, eps_pct=0.01, eps_med=0.01,
                    rng=np.random.default_rng(0))
print(res.summary())
```

**Scale limits of the current simulator.**
- `qae_attribute` builds exact amplitudes by **enumerating** the k-subsets, so it is for small universes (C(n,k) up to a few million).
- For large n, estimate a_F and a_G classically (Monte Carlo) and feed them to `IdealOracle` from `fairbench.qae.estimators`, as `scripts/qae_mandate.py` does.
- For the gate-level oracle and cost model at large n, use `ft.oracle.feasibility_oracle` / `ft.resources.oracle_formula_counts` / `resource_estimate` with the same `cs` plus the `LinearThreshold`.

---

## 7. The Parnassus case in quantum terms

Proxy mandate: hold 36 stocks, 27 companies excluded, S&P 500 universe.

- **Encoding.** Prepare the Dicke state over the **allowed** assets only. Then every basis state is legal, so a_F = 1 and only one estimation run is needed: the percentile is a_G directly.
- **What that means for the advantage.** With no tightness to exploit, only the **precision gain** (1/ε against 1/ε²) applies. The "tight mandate" advantage appears when real mandates add caps and ESG or carbon averages that make P_F small.
- **Resources** (n = 500, k = 36, 11-bit weights, from `ft/resources.py`):

| Item | Value |
|---|---|
| Logical qubits | about 1,000 (500 data qubits plus ancillas) |
| Return comparator | about 3e4 Toffolis |
| Dicke preparation | about 3.5e4 rotations, the dominant cost |
| T gates per Grover step | about 5.6e6 |

  Squarely a fault-tolerant workload.
- **Quarterly scoring** is one estimate per quarter. The quantum economics are per estimate, so 28 quarters means 28 runs.

---

## 8. Validation status

| What | Status |
|---|---|
| Oracle circuits | Gate-level verified against the float rules on every portfolio for n ≤ 16, and at n = 50 on 2k random inputs |
| Amplitude estimation at small n | Exact state-vector Grover in the weight-k subspace |
| Amplitude estimation at large n | Simulated from exact or Monte Carlo amplitudes (binomial shots). Exact for an ideal noiseless device; no circuits simulated |
| Noise | Gate counts transpiled at n = 6–16; decay model checked against an Aer density-matrix simulation at n = 4 (the model is about 2× pessimistic) |
| Hardware | **Not run yet.** VTT Q50 / Aalto Q20 via LUMI and 10 minutes of IBM are available. Plan: Dicke-state fidelity and Grover-signal decay for a "measured hardware gap" slide |
| Tests | `tests/test_qae_*.py`, `tests/test_oracle*.py`, `tests/test_exact_count.py` (part of the full suite, offline) |

---

## 9. File map

| Path | Role |
|---|---|
| `fairbench/quantum/dicke.py` | Dicke state preparation |
| `fairbench/ft/oracle.py` | Reversible oracle and quantisation modes |
| `fairbench/ft/revsim.py` | Fast classical simulator for the oracle |
| `fairbench/ft/resources.py` | Logical cost model (qubits, Toffoli, T, depth) |
| `fairbench/qae/estimators.py` | IQAE, MLAE, canonical AE, noise model, exact query accounting |
| `fairbench/qae/attribution_qae.py` | Percentile, quantile (hybrid median), whole attribution, classical baselines |
| `fairbench/qae/exact_count.py` | Exact-counting DP baseline |
| `fairbench/qae/demo_bridge.py` | Exact / classical / quantum side by side |
| `fairbench/quantum/encoding.py` | Stage 2 weight grid as a QUBO, with annealing and amplitude amplification (sampling; separate from QAE) |
| `scripts/demo_qae.py` | 2-second demo: loose vs tight rules |
| `scripts/qae_*.py` | Studies (scaling, hybrid, mandate, noise, break-even, DP, pitch figure) |
| `results/qae_pitch_figure.png` | The slide |

---

## 10. Open items

1. Hardware characterisation on VTT Q50, Aalto Q20 and IBM (Dicke-state fidelity, Grover-signal decay).
2. Helper for the value-weighted threshold. The maths works today via `LinearThreshold`; a small wrapper plus a test would make it one call for `apps/real_fund.py`.
3. QAE over the Stage 2 weight grid: amplitude estimation on top of amplitude amplification on the QUBO encoding. Feasible in principle, not built.
4. Oracles for `MinGroups` and the quadratic risk caps: new circuitry; currently enforced classically.
5. Untested classical baselines: importance sampling, quasi-Monte Carlo, quantum-walk-accelerated MCMC.
