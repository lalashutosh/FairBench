"""Score every sampler against exact reference distributions on small universes.

    .venv/bin/python scripts/validate_reference.py

Classical samplers come first (rejection, swap chain, importance reweighting, simulated
annealing), then the quantum ones on the same instances (the Dicke circuit for Stage 1, an
amplitude-amplification circuit on the Stage 2 encoding), all on a simulator. For each:
feasibility and violation rate, total variation distance with its sampling-noise floor, KL
divergence (blank when it is infinite, with the smoothed value beside it), inclusion-
frequency error, return-distribution error, percentile error, runtime, and for circuits
the qubit count, transpiled depth, two-qubit gates and shots.

This table is a correctness check. It is not a speed comparison and supports no claim of
quantum advantage: the instances are tiny, the simulator is noiseless, and the
amplification circuit uses an oracle built from the enumerated answer.
Output: results/reference_validation.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from any folder, installed or not

import time

import numpy as np
import pandas as pd

from fairbench.baselines import mcmc_swap_sample
from fairbench.constraints import Cardinality, ConstraintSet, Exclusion, SectorCap, WeightRuleSet, WeightSum, active_risk
from fairbench.data import synthetic_universe
from fairbench.portfolio.exact import (compare_sample, compare_to_exact, exact_benchmark_aware,
                                       exact_uniform_subsets, exact_uniform_weights)
from fairbench.portfolio.milp import feasible_extremes
from fairbench.portfolio.reference import benchmark_aware, uniform_subsets, uniform_weights
from fairbench.portfolio.weights import WeightGrid, subset_weights
from fairbench.quantum.encoding import build_qubo, check_encoding, grover_sample, simulated_annealing

ROOT = Path(__file__).resolve().parents[1]
SEED, N = 0, 20_000
COLUMNS = ["instance", "distribution", "sampler", "kind", "n_samples", "n_feasible_exact", "feasibility_rate",
           "violation_rate", "tv", "tv_noise_floor", "kl", "kl_smoothed", "marginal_error_max",
           "marginal_error_rmse", "return_dist_ks", "return_dist_w1", "percentile_error", "runtime_s", "qubits",
           "depth", "two_qubit_gates", "shots", "seed"]


def row(instance, distribution, sampler, kind, metrics, runtime, **circuit) -> dict:
    return {"instance": instance, "distribution": distribution, "sampler": sampler, "kind": kind,
            "runtime_s": round(runtime, 3), "seed": SEED, **{k: metrics.get(k) for k in COLUMNS if k in metrics},
            **circuit}


def stage1(rows: list[dict]) -> None:
    u = synthetic_universe(12, 3, seed=4)
    cs = ConstraintSet([Cardinality(5), SectorCap("S0", 2), Exclusion([11])])
    rng = np.random.default_rng(7)
    b, r = rng.dirichlet(np.ones(12) * 2), rng.normal(0.02, 0.08, 12)
    name = "stage 1: 5 of 12, sector cap, 1 exclusion"
    for policy in ("equal", "benchmark"):
        exact = exact_uniform_subsets(u, cs, policy, b)
        fund = float(np.quantile(exact.returns(r), 0.7))
        dist = f"D1 uniform subsets, {policy} weights"
        t = time.perf_counter()
        ref = uniform_subsets(u, cs, N, policy=policy, benchmark=b, seed=SEED)
        rows.append(row(name, dist, "rejection", "classical", compare_sample(exact, ref, r, fund), time.perf_counter() - t))
        if policy == "equal":
            t = time.perf_counter()
            res = mcmc_swap_sample(u, cs, N, burn_in=2000, thin=5, seed=SEED)
            m = compare_to_exact(exact, subset_weights(res.samples), asset_returns=r, realised_return=fund, seed=SEED)
            rows.append(row(name, dist, "swap chain (thin 5)", "classical", m, time.perf_counter() - t))

            from qiskit import transpile

            from fairbench.apps.attribution import dicke_sampler
            from fairbench.quantum.ansatz import build_ansatz
            t = time.perf_counter()
            res = dicke_sampler()(u, cs, N, SEED)
            m = compare_to_exact(exact, subset_weights(res.samples), n_raw=res.cost, asset_returns=r,
                                 realised_return=fund, seed=SEED)
            qc = transpile(build_ansatz(11, 5, 0, None), basis_gates=["cx", "rz", "sx", "x"], optimization_level=1,
                           seed_transpiler=0)
            rows.append(row(name, dist, "Dicke circuit + filter (simulator)", "quantum", m, time.perf_counter() - t,
                            qubits=qc.num_qubits, depth=qc.depth(), two_qubit_gates=qc.count_ops().get("cx", 0),
                            shots=res.cost))
            te = active_risk(exact.weights, u.cov, b)
            tau = float(np.median(te))
            t = time.perf_counter()
            d3 = benchmark_aware(ref, u, b, tau)
            rows.append(row(name, "D3 benchmark-aware, tau = median TE", "rejection + importance reweighting",
                            "classical", compare_sample(exact_benchmark_aware(exact, u, b, tau), d3, r, fund),
                            time.perf_counter() - t))


def stage2(rows: list[dict]) -> dict:
    u = synthetic_universe(4, 2, seed=5)
    grid = WeightGrid(n=4, units=6, min_units=1, max_units=4, k_min=2, k_max=3)
    rules = WeightRuleSet([WeightSum([0, 1], upper=0.5, label="group cap 50%")])
    r = np.random.default_rng(3).normal(0.02, 0.08, 4)
    exact = exact_uniform_weights(u, grid, rules)
    fund = float(np.quantile(exact.returns(r), 0.7))
    name, dist = "stage 2: 4 assets, weights in sixths, group cap", "D2 uniform weight grid"

    t = time.perf_counter()
    ref = uniform_weights(u, grid, rules, N, seed=SEED)
    rows.append(row(name, dist, "rejection", "classical", compare_sample(exact, ref, r, fund), time.perf_counter() - t))

    model = build_qubo(grid, rules)
    proof = check_encoding(model, np.rint(exact.weights * grid.units))
    t = time.perf_counter()
    Z = simulated_annealing(model, N, sweeps=100, seed=SEED)
    ok = np.abs(model.penalty(Z)) < 1e-9
    m = compare_to_exact(exact, model.decode(Z[ok]), n_raw=len(Z), asset_returns=r, realised_return=fund, seed=SEED)
    rows.append(row(name, dist, "simulated annealing on the QUBO + filter", "classical", m, time.perf_counter() - t))

    t = time.perf_counter()
    Z, info = grover_sample(model, N, seed=SEED)
    ok = np.abs(model.penalty(Z)) < 1e-9
    m = compare_to_exact(exact, model.decode(Z[ok]), n_raw=len(Z), asset_returns=r, realised_return=fund, seed=SEED)
    rows.append(row(name, dist, "amplitude amplification on the QUBO + filter (simulator)", "quantum", m,
                    time.perf_counter() - t, qubits=info["n_qubits"], depth=info["depth"],
                    two_qubit_gates=info["two_qubit_gates"], shots=info["shots"]))
    ext = feasible_extremes(u, grid, rules, r)
    ret = exact.returns(r)
    return {"encoding": proof, "grover": info,
            "milp": {"min_return": ext["min_return"], "max_return": ext["max_return"],
                     "enumerated_min": float(ret.min()), "enumerated_max": float(ret.max())}}


def main() -> None:
    rows: list[dict] = []
    stage1(rows)
    extra = stage2(rows)
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    out = ROOT / "results" / "reference_validation.csv"
    df.to_csv(out, index=False)
    show = df[["distribution", "sampler", "feasibility_rate", "tv", "tv_noise_floor", "kl", "marginal_error_max",
               "return_dist_ks", "percentile_error", "runtime_s", "qubits", "depth"]]
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4g}", na_rep=""))
    p = extra["encoding"]
    print(f"\nStage 2 encoding: {p['n_qubits']} qubits, {p['n_zero_penalty']} zero-penalty bitstrings for "
          f"{p['n_feasible']} feasible portfolios, one-to-one: {p['decoded_equals_feasible'] and p['encode_roundtrip']}, "
          f"smallest penalty of an infeasible bitstring: {p['min_violation_penalty']:g}")
    g, ml = extra["grover"], extra["milp"]
    print(f"Amplitude amplification: {g['iterations']} iterations chosen from the EXACT feasible fraction "
          f"{g['feasible_fraction']:.4f}; oracle = {g['oracle']}")
    print(f"MILP extremes {ml['min_return']:.6f} / {ml['max_return']:.6f}; enumeration "
          f"{ml['enumerated_min']:.6f} / {ml['enumerated_max']:.6f}")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
