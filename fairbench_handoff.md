# FairBench — Claude Code handoff

## Context

Team #22 "FairBench" at the Hanken Quantum x Finance Hackathon (8–10 Oct 2026).

**Problem (may pivot):** When an ESG fund underperforms, is it the manager or the ESG constraints? The approach is random-portfolio benchmarking: sample many portfolios that satisfy the *same* rules as the fund, build a null distribution of returns, and see where the real fund ranks. Classical constrained sampling degrades as rule sets grow (MCMC gets trapped in islands, rejection sampling acceptance collapses).

**Quantum contribution:** Constraint-preserving circuits (Dicke state + XY mixer) that only ever produce portfolios with exactly k holdings, plus penalty or trained layers for the other constraints. The fault-tolerant roadmap adds amplitude amplification, giving a quadratic speedup over rejection sampling.

**Pivot plan:** If problem validation fails, the same engine becomes a CVaR-VQA ESG-constrained portfolio **optimizer**. Only the application layer changes. **Keep everything below the application layer problem-agnostic.**

## Architecture (build in this order)

| Priority | Layer | Modules |
|---|---|---|
| P0 | Data and constraints | `data.py`, `constraints.py` |
| P0 | Quantum core | `quantum/dicke.py`, `quantum/ansatz.py`, `quantum/hamiltonian.py`, `backends.py` |
| P1 | Training and post-processing | `training.py`, `postprocess.py` |
| P1 | Benchmark harness | `baselines.py`, `metrics.py` |
| P2 | Application (swap on pivot) | `apps/attribution.py`, `apps/optimizer.py` |

## Repo layout

```
fairbench/
  pyproject.toml
  fairbench/
    __init__.py
    data.py
    constraints.py
    backends.py
    training.py
    postprocess.py
    baselines.py
    metrics.py
    quantum/
      __init__.py
      dicke.py
      ansatz.py
      hamiltonian.py
    apps/
      __init__.py
      attribution.py      # P2 stub only for now
      optimizer.py        # P2 stub only for now
  scripts/
    p0_demo.py
  tests/
    test_constraints.py
    test_dicke.py
    test_ansatz.py
    test_baselines.py
```

## Interfaces

Representation: a portfolio *selection* is a binary vector `x` of length n (`x[i] = 1` means asset i is held). Weights are assigned later in post-processing (equal-weight by default).

### data.py
- `@dataclass Universe`: `tickers: list[str]`, `mu: np.ndarray (n,)`, `cov: np.ndarray (n,n)`, `sector: list[str]`, `esg_score: np.ndarray (n,)`, `carbon: np.ndarray (n,)`, `returns: pd.DataFrame | None` (historical, for attribution later).
- `load_universe(csv_path) -> Universe`
- `synthetic_universe(n, n_sectors, seed) -> Universe` for development and tests.

### constraints.py
- `Constraint` protocol: `check(x: np.ndarray, u: Universe) -> bool` and `check_batch(X: np.ndarray, u) -> np.ndarray[bool]` (vectorised over rows).
- Implementations: `Cardinality(k)`, `SectorCap(sector, max_count)`, `Exclusion(indices)`, `MinESG(min_avg_score)`, `CarbonCap(max_avg)`.
- `ConstraintSet(constraints)` with `check`, `check_batch`, and `cardinality` (returns k, used by the circuit).
- Constraints are plain data plus functions. Never hard-code ESG logic elsewhere.

### quantum/dicke.py
- `dicke_state(n, k) -> QuantumCircuit`: deterministic Dicke state preparation (Bärtschi & Eidenbenz 2019, split-and-cyclic-shift construction), giving an equal superposition of all weight-k bitstrings.

### quantum/ansatz.py
- `xy_mixer(n, beta: Parameter, topology="ring"|"complete") -> QuantumCircuit`: exp(-iβ(XX+YY)/2) on edges. Must preserve Hamming weight.
- `cost_layer(op: SparsePauliOp, gamma: Parameter) -> QuantumCircuit`
- `build_ansatz(n, k, p, cost_op, topology) -> QuantumCircuit` = Dicke prep followed by p × (cost layer, mixer), with parameter vectors γ and β.

### quantum/hamiltonian.py
- `penalty_operator(cs: ConstraintSet, u: Universe, weights) -> SparsePauliOp`: encodes non-cardinality constraints as quadratic penalties (x_i = (1 - Z_i)/2). Cardinality is **not** penalised, because the circuit enforces it.
- `objective_operator(u, risk_aversion) -> SparsePauliOp`: mean–variance QUBO, used by the optimizer pivot.

### backends.py
- `sample(circuit, params, shots, backend="aer_statevector"|"aer_mps"|"ibm"|"vtt", seed=None) -> np.ndarray[shots, n]` (uint8).
- **Bit-order convention:** Qiskit bitstrings are little-endian. Convert so that column i is asset i. Write a test for this.
- `ibm` and `vtt` are stubs for now and should raise `NotImplementedError` with a clear message. Credentials come from environment variables only.

### baselines.py
- `enumerate_feasible(u, cs) -> np.ndarray` (exact; iterate over k-combinations; only for n ≤ ~25).
- `random_k_subsets(n, k, shots, seed)`: classical analogue of the Dicke state.
- `rejection_sample(u, cs, shots, seed)`: random k-subsets filtered by the constraints.
- `mcmc_swap_sample(u, cs, shots, burn_in, thin, seed)`: Metropolis chain whose move swaps one held asset for one non-held asset and is accepted only if the result is feasible. Records the number of moves for cost accounting.

### metrics.py
- `acceptance_rate(X, cs, u)`
- `tv_to_uniform(X_feasible, feasible_set)`: total variation distance between the empirical distribution and uniform over the exact feasible set.
- `coverage(X_feasible, feasible_set)`: fraction of feasible portfolios seen at least once.
- `cost_per_feasible_sample(...)`: shots or chain moves per accepted feasible sample.

### training.py (P1)
- `train(ansatz, objective_fn, backend, optimizer="COBYLA"|"SPSA", maxiter, seed)`. The objective is a callable on sampled bitstrings, so CVaR (optimizer pivot) and a uniformity objective (benchmarking) are both just functions.

### postprocess.py (P1)
- `filter_feasible`, `repair` (greedy swap to feasibility, but **record that it was used**, because it biases sampling), `assign_weights(x, scheme="equal")`.

## P0 milestone — `scripts/p0_demo.py`

1. Synthetic universe: n = 12, k = 4, 3 sectors, SectorCap(max 2 per sector) and MinESG.
2. Exact feasible set via enumeration.
3. Sample with: (a) `build_ansatz` with p = 0 (pure Dicke) on Aer, then filtering; (b) `rejection_sample`; (c) `mcmc_swap_sample`.
4. Print a table of acceptance rate, TV distance to uniform, coverage, and cost per feasible sample.

## Tests (must pass before moving on)
- Constraints agree with brute force on random vectors.
- `dicke_state(6, 3)`: every sample has weight 3 and the distribution is uniform over all C(6,3) = 20 states (statevector check, exact).
- `xy_mixer` preserves Hamming weight for random β (statevector check).
- Bit-order round trip: preparing a known basis state returns the expected `x`.
- No network or hardware calls in tests. Fix all seeds.

## Important caveat (keep this honest in the pitch)
Pure Dicke + filtering produces **exactly the same distribution** as classical random k-subsets + rejection. It is a correctness baseline, **not** a quantum advantage. The quantum value has to come from:
- penalty or trained layers that push probability mass onto the feasible set *while staying close to uniform* (measure with TV distance, not just acceptance rate);
- amplitude amplification (fault-tolerant roadmap; resource estimate only);
- the ablation: same post-processing, swap only the sampler (random bitstrings, shallow/unentangled circuit, our circuit on a simulator, our circuit on hardware);
- the MPS simulability check (`aer_mps` with a bond-dimension sweep) showing where classical simulation gets expensive.

## Conventions
- Python 3.11. Dependencies: `qiskit>=1.0`, `qiskit-aer`, `numpy`, `pandas`, `scipy`, `pytest`, `matplotlib`. Add `qiskit-ibm-runtime` and `qiskit-iqm` only when wiring hardware.
- Type hints, small pure functions, NumPy-vectorised checks.
- Every sampler returns an `(shots, n)` uint8 array, so all metrics work on any sampler.
- Use a uniform seed argument everywhere for reproducibility.

## First task for Claude Code
Scaffold the repo above, implement all P0 modules and tests, then run `pytest` and `scripts/p0_demo.py`. Stub the P1 and P2 modules with docstrings and `NotImplementedError`. Report the demo table when done.