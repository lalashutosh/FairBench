"""P1-B: sampler ablation. Identical post-processing, only the sampler changes.

Samplers: random_bitstrings, product_state, dicke_p0, trained (penalty ansatz p=2, uniformity
objective, exact training), rejection, mcmc_swap. Instances: p0 (n=12) and island (n=16).
Primary post-processing = filter; repair-mode reported as secondary columns.
"""
from __future__ import annotations

import argparse, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qiskit import QuantumCircuit
from fairbench.backends import sample
from fairbench.baselines import enumerate_feasible, rejection_sample, mcmc_swap_sample
from fairbench.instances import p0_instance, island_instance, describe_instance
from fairbench.metrics import tv_to_uniform, tv_expected_uniform, coverage, cost_per_feasible_sample
from fairbench.objectives import uniformity_objective, uniformity_stats_exact
from fairbench.postprocess import postprocess
from fairbench.quantum.ansatz import build_ansatz
from fairbench.quantum.hamiltonian import penalty_operator
from fairbench.training import train_multistart, exact_probabilities


def product_circuit(n: int, k: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    theta = 2 * np.arcsin(np.sqrt(k / n))  # P(1) = sin^2(theta/2) = k/n
    for i in range(n):
        qc.ry(theta, i)
    return qc


def evaluate(name, X, cost, unit, cs, u, F, seed, instance):
    M = len(F)
    acc = float(cs.check_batch(X, u).mean())
    pf = postprocess(X, cs, u, mode="filter", seed=seed)
    S = pf["samples"]
    nf = len(S)
    pr = postprocess(X, cs, u, mode="repair", seed=seed)
    Sr = pr["samples"]
    return dict(
        instance=instance, sampler=name, n_raw=len(X), acceptance=acc, n_feasible=nf,
        TV_to_unif=tv_to_uniform(S, F) if nf else float("nan"),
        TV_floor=tv_expected_uniform(M, nf, seed=seed) if nf else float("nan"),
        coverage=coverage(S, F) if nf else 0.0,
        cost_per_feasible=cost_per_feasible_sample(cost, nf), cost_unit=unit,
        n_repaired=pr["n_repaired"], n_feasible_repair=len(Sr),
        TV_after_repair=tv_to_uniform(Sr, F) if len(Sr) else float("nan"),
        TV_floor_repair=tv_expected_uniform(M, len(Sr), seed=seed) if len(Sr) else float("nan"),
        coverage_repair=coverage(Sr, F) if len(Sr) else 0.0)


def run_instance(label, u, cs, shots, seed, maxiter, n_starts, lam):
    n, k = u.n, cs.cardinality
    F = enumerate_feasible(u, cs)
    M = len(F)
    info = describe_instance(u, cs)
    print(f"\n[{label}] n={n} k={k} M={M} frac={info['feasible_fraction']:.4f} "
          f"components={info['n_components']} sizes={info['component_sizes']}", flush=True)
    rows, train_rows = [], []
    rng = np.random.default_rng(seed)

    # 1 random bitstrings
    X = rng.integers(0, 2, size=(shots, n), dtype=np.uint8)
    rows.append(evaluate("random_bitstrings", X, shots, "shots", cs, u, F, seed, label))
    # 2 product state
    X = sample(product_circuit(n, k), None, shots, "aer_statevector", seed=seed)
    rows.append(evaluate("product_state", X, shots, "shots", cs, u, F, seed, label))
    # 3 dicke p=0
    X = sample(build_ansatz(n, k, 0, None), None, shots, "aer_statevector", seed=seed)
    rows.append(evaluate("dicke_p0", X, shots, "shots", cs, u, F, seed, label))
    # 4 trained
    op = penalty_operator(cs, u)
    ans = build_ansatz(n, k, 2, op, "ring")
    obj = uniformity_objective(cs, u, feasible_set=F, lam=lam)
    t0 = time.perf_counter()
    res = train_multistart(ans, obj, n_starts=n_starts, seed=seed, exact=True,
                           optimizer="COBYLA", maxiter=maxiter)
    wall = time.perf_counter() - t0
    probs = exact_probabilities(ans, res.params)
    st = uniformity_stats_exact(probs, obj.meta["fcodes"])
    X = sample(ans, res.params, shots, "aer_statevector", seed=seed)
    r = evaluate("trained", X, shots, "shots", cs, u, F, seed, label)
    r.update(exact_PF=st["P_F"], exact_TV=st["tv"],
             cost_uniform_corrected=(1 / (st["P_F"] * st["min_ratio"])
                                     if st["P_F"] * st["min_ratio"] > 0 else float("inf")))
    rows.append(r)
    train_rows.append(dict(instance=label, maxiter=maxiter, n_starts=n_starts, lam=lam,
                           n_evals=res.n_evals, shots_equiv=res.shots_equiv,
                           train_wall_s=wall, final_value=res.final_value))
    # dicke exact stats for reference (uniform-corrected cost of Dicke/rejection)
    pd0 = exact_probabilities(build_ansatz(n, k, 0, None), {})
    s0 = uniformity_stats_exact(pd0, obj.meta["fcodes"])
    for rr in rows:
        if rr["sampler"] == "dicke_p0":
            rr.update(exact_PF=s0["P_F"], exact_TV=s0["tv"],
                      cost_uniform_corrected=1 / (s0["P_F"] * s0["min_ratio"]))
    # 5 rejection
    rj = rejection_sample(u, cs, shots, seed=seed)
    r = evaluate("rejection", rj.raw, rj.cost, rj.cost_unit, cs, u, F, seed, label)
    rows.append(r)
    n_rej = len(rj.samples)
    # 6 mcmc: same number of feasible samples as rejection yields (thin=5, burn_in=200)
    mc = mcmc_swap_sample(u, cs, max(n_rej, 1), burn_in=200, thin=5, seed=seed, x0=F[0])
    rows.append(evaluate("mcmc_swap", mc.samples, mc.cost, mc.cost_unit, cs, u, F, seed, label))
    return rows, train_rows, info


def make_plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    insts = list(df.instance.unique())
    fig, axs = plt.subplots(1, len(insts), figsize=(6 * len(insts), 4.6), squeeze=False)
    for ax, inst in zip(axs[0], insts):
        d = df[df.instance == inst]
        for _, r in d.iterrows():
            ax.scatter(r.acceptance, r.TV_to_unif, s=60, zorder=3)
            ax.annotate(r.sampler, (r.acceptance, r.TV_to_unif), textcoords="offset points",
                        xytext=(5, 4), fontsize=8)
            ax.hlines(r.TV_floor, r.acceptance - 0.01, r.acceptance + 0.01, colors="gray",
                      lw=1, zorder=2)
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlabel("acceptance (raw feasible fraction)")
        ax.set_ylabel("TV(filtered samples, uniform on F)")
        ax.set_title(f"{inst}: acceptance vs TV (gray tick = finite-sample floor)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--maxiter-island", type=int, default=None)
    ap.add_argument("--n-starts", type=int, default=2)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, trows = [], []
    for label, (u, cs), mi in (("p0", p0_instance(), a.maxiter),
                               ("island", island_instance(), a.maxiter_island or a.maxiter)):
        r, t, info = run_instance(label, u, cs, a.shots, a.seed, mi, a.n_starts, a.lam)
        rows += r; trows += t
    df = pd.DataFrame(rows)
    tdf = pd.DataFrame(trows)
    df.to_csv(os.path.join(a.out, "ablation.csv"), index=False)
    tdf.to_csv(os.path.join(a.out, "ablation_training_cost.csv"), index=False)
    make_plot(df, os.path.join(a.out, "ablation.png"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    main_cols = ["instance", "sampler", "acceptance", "n_feasible", "TV_to_unif", "TV_floor",
                 "coverage", "cost_per_feasible", "cost_unit", "exact_PF", "exact_TV",
                 "cost_uniform_corrected"]
    print("\n=== Filter mode (primary) ===")
    print(df[main_cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== Repair mode (secondary) ===")
    print(df[["instance", "sampler", "n_repaired", "n_feasible_repair", "TV_after_repair",
              "TV_floor_repair", "coverage_repair"]].to_string(index=False,
                                                              float_format=lambda x: f"{x:.4g}"))
    print("\n=== Training cost (separate; trained sampler only) ===")
    print(tdf.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    g = lambda i, s, c: float(df[(df.instance == i) & (df.sampler == s)][c].iloc[0])
    print("\nNotes (shots=%d, seed=%d; MCMC draws as many samples as rejection kept, burn_in=200, "
          "thin=5, start=feasible_set[0]; cost units differ: shots / proposals / moves):" %
          (a.shots, a.seed))
    print(f"- dicke_p0 == rejection in distribution (both uniform over C(n,k) then filtered); "
          f"acceptance p0: {g('p0','dicke_p0','acceptance'):.3f} vs {g('p0','rejection','acceptance'):.3f}; "
          f"any difference is shot noise. Dicke+filter is NOT a quantum advantage.")
    print(f"- MCMC trapped on island instance: coverage {g('island','mcmc_swap','coverage'):.3f} "
          f"(TV {g('island','mcmc_swap','TV_to_unif'):.3f} vs floor {g('island','mcmc_swap','TV_floor'):.3f}) "
          f"vs rejection coverage {g('island','rejection','coverage'):.3f}.")
    for i in ("p0", "island"):
        print(f"- trained layers on {i}: exact P_F {g(i,'trained','exact_PF'):.3f} vs Dicke "
              f"{g(i,'dicke_p0','exact_PF'):.3f}; exact TV {g(i,'trained','exact_TV'):.3f} vs Dicke "
              f"{g(i,'dicke_p0','exact_TV'):.3f}; uniform-corrected cost "
              f"{g(i,'trained','cost_uniform_corrected'):.2f} vs Dicke "
              f"{g(i,'dicke_p0','cost_uniform_corrected'):.2f} (lower is better; trained wins only "
              f"if both P_F up and TV not worse; training cost excluded above).")
    print("- Repair mode biases the distribution (see TV_after_repair); filter is primary.")
    print("- No quantum advantage claimed at this scale; see MPS sweep.")


if __name__ == "__main__":
    main()
