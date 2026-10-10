# FairBench on a real quantum computer: the VTT Q50 run

_Run 2026-10-09 on VTT Q50 through LUMI (event module `fiqci-vtt-qiskit-QxF`, project `project_462001763`). Every number below comes from `results/hw/`. Scripts: `scripts/hw_q50.py` (runs), `scripts/hw_q50_analyze.py` (fit and figure), `scripts/hw_twin.py` (digital twin)._

---

## 1. Summary

- FairBench's constraint-preserving circuit ran on VTT's 50-qubit Q50 through LUMI.
- It produced valid portfolios 54% of the time, against 37.5% by chance.
- Every two-qubit gate costs signal. One amplitude-estimation step (about 150 gates) erases most of it.
- A digital twin calibrated on the hardware data shows the toy problem working at about 0.1% gate error; real portfolio sizes need about 0.00001%. That is fault-tolerant hardware.

**In one sentence:** the circuit runs on real hardware and beats chance, and the measured noise says what hardware the quantum advantage needs.

**Figures:**

| Figure | Use |
|---|---|
| `results/hw/hw_q50.png` | Hardware: valid-portfolio share (left) and amplitude-estimation signal (right) |
| `results/hw/hw_twin_lumi.png` | Twin reproduces Q50 (left), and what lower error rates would give (right) |
| `results/qae_hybrid_pitch.png` | For comparison, the simulated result: 12× fewer queries on tight rules |

---

## 2. What we ran

| Experiment | What it tests | Circuits |
|---|---|---|
| **Diagnostics** | Is the device healthy? Readout (prepare 0000 / 1111 / 0011), Bell pair, 4-qubit GHZ | 5 tiny circuits |
| **A. Constraint-preserving state** (Dicke state) | Does the circuit output only valid portfolios (exactly k assets held)? | n = 4, 6, 8 assets |
| **B. Toy amplitude estimation** | Can the device estimate a fund's rank? 4 assets, hold 2: 6 portfolios, 2 of them worse than the fund, so the exact rank is **33.3%**. Grover steps m = 0–3. | 8 circuits |

Each circuit had 2,000 shots. Runs B and A were pinned to physical qubits 0–3 after the first run landed on poor qubits. B was repeated with FiQCI-EMS readout error mitigation.

---

## 3. Results

### Diagnostics: the device is healthy at low depth

| Test | Result |
|---|---|
| Readout correct (0000 / 1111 / 0011) | 92.7% / 86.3% / 89.1% (per-qubit flips 0.6–5.8%) |
| Bell pair (1 two-qubit gate) | **95.0%** |
| GHZ, 4 qubits (3 two-qubit gates) | **84.5%** |

### A. Constraint-preserving circuit (share of shots that are valid portfolios)

| Assets n, held k | Two-qubit gates | Measured | Random chance | Ideal |
|---|---|---|---|---|
| 4, 2 | 46 | **54%** (58–62% in the run-B circuits) | 37.5% | 100% |
| 6, 2 | 92 | 34% | 23% | 100% |
| 6, 3 | 144 | 32% | 31% | 100% |
| 8, 3 | 239 | 28% | 22% | 100% |

**Reading:** above chance in every case, but the margin over chance shrinks as the gate count grows.

### B. Toy amplitude estimation of the fund's rank (exact 33.3%)

| Grover steps m | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Two-qubit gates (after routing) | 46 | 154 | 273 | 386 |
| Ideal P(portfolio worse than fund) | 0.33 | 0.93 | 0.00 | 0.85 |
| **Measured, raw** | 0.19 | 0.19 | 0.13 | 0.13 |
| Measured with readout mitigation | 0.18 | 0.20 | 0.15 | 0.11 |
| Random-output level | 0.125 | 0.125 | 0.125 | 0.125 |

| | Raw | With readout mitigation |
|---|---|---|
| Noise-aware rank estimate (exact 33.3%) | 27% | 23% |
| Signal surviving one Grover step | ~18% | ~19% |

**Reading:** after one Grover step the output is close to random. The noise-aware fit moves the estimate toward the truth (plain reading 19%, fit 27%), but it is not accurate. Its confidence interval assumes a simple noise model and does **not** cover the true value, so don't quote the interval.

### Digital twin (Qiskit Aer, measured readout errors, one fitted two-qubit error)

| | LUMI run (CPU) | Laptop run |
|---|---|---|
| Fitted two-qubit error | **1.8%** | 2.1% |
| Reproduces the hardware? | Yes: Grover curve and Dicke valid share within a few points | same |

Sweep (LUMI run): signal surviving per Grover step, and the rank estimate.

| Two-qubit error | Today (1.8%) | 0.3% | 0.1% | 0.03% | 0.01% | 0.001% |
|---|---|---|---|---|---|---|
| Signal surviving per step | 13% | 67% | 86% | 94% | 96% | 98% |
| Rank estimate (exact 33.3%) | 35% (model-consistent only) | 33.1% | 33.4% | 33.4% | 33.5% | 33.4% |

**Scaling to the real problem.** One Grover step at 16 assets is about 17k two-qubit gates. Keeping half the signal over 100 steps needs a two-qubit error of about 4×10⁻⁷, or about 2×10⁻⁷ with realistic routing. Today's devices are around 10⁻². That gap is the case for fault tolerance.

---

## 4. Conclusions

1. **It runs on real hardware.** The constraint-preserving circuit produces valid portfolios well above chance on VTT Q50 (54% vs 37.5% at 4 assets).
2. **Depth is the wall.** Advantage over random shrinks with every two-qubit gate (46 gates: clear signal; about 150: mostly gone; about 240: near random). A single amplitude-estimation step costs about 150 gates even on the 4-asset toy.
3. **The noise is understood.** A one-parameter twin (two-qubit error about 2%, consistent with the Bell/GHZ tests) reproduces the hardware data on LUMI.
4. **The roadmap is quantified.** The toy problem works at about 10⁻³ two-qubit error. Real portfolio sizes need about 10⁻⁷, i.e. error-corrected (fault-tolerant) machines.
5. **No quantum advantage on today's hardware**, and we don't claim one. The advantage (12× fewer queries on tight rules) is a noiseless-simulation result for future fault-tolerant hardware.

---

## 5. What this run does not show

- **A quantum advantage on Q50.** There is none; the run is a feasibility and characterisation result.
- **That Q50 estimated the fund's rank correctly.** The fit gave 23–27% against an exact 33%, with an unreliable interval.
- **That the 12× (or any) query advantage was measured on hardware.** It comes from noiseless simulation.

## 6. Questions and answers

- **"Why so few qubits?"** Each extra two-qubit gate costs about 2% of the signal, so bigger circuits only return noise. The 4-asset toy is the largest that still shows signal.
- **"What would it take?"** About a 10× better gate error to run the toy amplitude estimation properly, and about 10⁵× better (error correction) for real portfolio sizes.
- **"Did error mitigation help?"** Readout mitigation (FiQCI-EMS level 1) slightly raised the valid-portfolio share (58% → 60–62%) but cannot restore lost coherence.
- **"First run looked random?"** The transpiler placed it on weaker qubits. Pinning to qubits 0–3 (which passed the diagnostics) gave the results above.

---

## 7. Provenance and how to reproduce

| Run | File (`results/hw/`) | Q50 job ID |
|---|---|---|
| A + B, transpiler-chosen qubits (superseded) | `hw_q50_q50_20261009T133933Z.json` | 8236191e-057a-4993-b797-ed9ee34e11cd |
| Diagnostics | `hw_q50_q50_20261009T135044Z.json` | c1ad5fb9-0215-4968-9cf7-5edffb427c80 |
| B pinned, raw | `hw_q50_q50_20261009T135154Z.json` | 619c40b9-fbdc-4c31-b97d-e43fba195019 |
| B pinned, readout mitigation | `hw_q50_q50_ems1_20261009T135459Z.json` | 48c0b02c-daf7-4921-b2ac-c9d33908017f |
| A pinned | `hw_q50_q50_20261009T141216Z.json` | 5365cb3b-2392-413d-8a82-53c103d7d3b0 |
| Twin on LUMI / laptop | `hw_twin_lumi.json` / `hw_twin.json` | (simulation) |

```bash
# on LUMI (Jupyter launched with the event module lines)
python scripts/hw_q50.py --backend q50 --exp diag
python scripts/hw_q50.py --backend q50 --exp B --layout 0,1,2,3 [--ems 1]
python scripts/hw_q50.py --backend q50 --exp A --layout 0,1,2,3,4,5,6,7
python scripts/hw_twin.py
# locally
python scripts/hw_q50_analyze.py --dicke results/hw/hw_q50_q50_20261009T141216Z.json \
    --qae results/hw/hw_q50_q50_20261009T135154Z.json --qae2 results/hw/hw_q50_q50_ems1_20261009T135459Z.json
```
