"""P1-B: sampler ablation. Identical post-processing, only the sampler changes.

Samplers: random_bitstrings, product_state, dicke_p0, trained (penalty ansatz p=2, uniformity
objective, exact training), rejection, mcmc_swap. Instances: p0 (n=12) and island (n=16).
Primary post-processing = filter; repair-mode reported as secondary columns.
Multi-seed: the trained sampler is trained ONCE per instance (seed 0); each of --n-seeds sampling
seeds then re-runs every sampler. ablation.csv = mean/sd over seeds; ablation_per_seed.csv = raw.
exact_PF / exact_TV / cost_uniform_corrected are deterministic (statevector), not seed-averaged.
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


def run_instance(label, u, cs, shots, n_seeds, maxiter, n_starts, lam):
    n, k = u.n, cs.cardinality
    F = enumerate_feasible(u, cs)
    M = len(F)
    info = describe_instance(u, cs)
    print(f"\n[{label}] n={n} k={k} M={M} frac={info['feasible_fraction']:.4f} "
          f"components={info['n_components']} sizes={info['component_sizes']}", flush=True)
    rows, train_rows = [], []
    seed = 0  # training seed
    # trained sampler: train once
    op = penalty_operator(cs, u)
    ans = build_ansatz(n, k, 2, op, "ring")
    obj = uniformity_objective(cs, u, feasible_set=F, lam=lam)
    t0 = time.perf_counter()
    res = train_multistart(ans, obj, n_starts=n_starts, seed=seed, exact=True,
                           optimizer="COBYLA", maxiter=maxiter)
    wall = time.perf_counter() - t0
    st = uniformity_stats_exact(exact_probabilities(ans, res.params), obj.meta["fcodes"])
    s0 = uniformity_stats_exact(exact_probabilities(build_ansatz(n, k, 0, None), {}),
                                obj.meta["fcodes"])
    exact = {
        "trained": dict(exact_PF=st["P_F"], exact_TV=st["tv"],
                        cost_uniform_corrected=(1 / (st["P_F"] * st["min_ratio"])
                                                if st["P_F"] * st["min_ratio"] > 0 else float("inf"))),
        "dicke_p0": dict(exact_PF=s0["P_F"], exact_TV=s0["tv"],
                         cost_uniform_corrected=1 / (s0["P_F"] * s0["min_ratio"]))}
    train_rows.append(dict(instance=label, maxiter=maxiter, n_starts=n_starts, lam=lam,
                           n_evals=res.n_evals,
                           exact_evals_x_shots_lower_bound=res.shots_equiv,
                           train_wall_s=wall, final_value=res.final_value))

    for sd in range(n_seeds):
        rng = np.random.default_rng(sd)
        r = []
        X = rng.integers(0, 2, size=(shots, n), dtype=np.uint8)
        r.append(evaluate("random_bitstrings", X, shots, "shots", cs, u, F, sd, label))
        X = sample(product_circuit(n, k), None, shots, "aer_statevector", seed=sd)
        r.append(evaluate("product_state", X, shots, "shots", cs, u, F, sd, label))
        X = sample(build_ansatz(n, k, 0, None), None, shots, "aer_statevector", seed=sd)
        r.append(evaluate("dicke_p0", X, shots, "shots", cs, u, F, sd, label))
        X = sample(ans, res.params, shots, "aer_statevector", seed=sd)
        r.append(evaluate("trained", X, shots, "shots", cs, u, F, sd, label))
        rj = rejection_sample(u, cs, shots, seed=sd)
        r.append(evaluate("rejection", rj.raw, rj.cost, rj.cost_unit, cs, u, F, sd, label))
        # mcmc: same number of feasible samples as rejection yields (thin=5, burn_in=200)
        mc = mcmc_swap_sample(u, cs, max(len(rj.samples), 1), burn_in=200, thin=5, seed=sd, x0=F[0])
        r.append(evaluate("mcmc_swap", mc.samples, mc.cost, mc.cost_unit, cs, u, F, sd, label))
        for rr in r:
            rr["sample_seed"] = sd
            rr.update(exact.get(rr["sampler"], {}))
        rows += r
        print(f"  [{label}] seed {sd} done", flush=True)
    return rows, train_rows, info


METRICS = ["acceptance", "n_feasible", "TV_to_unif", "TV_minus_floor", "coverage", "cost_per_feasible"]
ORDER = ["random_bitstrings", "product_state", "dicke_p0", "trained", "rejection", "mcmc_swap"]


def aggregate(df):
    """mean/sd over sampling seeds per (instance, sampler); exact_* are deterministic."""
    df = df.copy()
    df["TV_minus_floor"] = df.TV_to_unif - df.TV_floor
    out = []
    for (inst, smp), g in df.groupby(["instance", "sampler"], sort=False):
        row = dict(instance=inst, sampler=smp, n_seeds=len(g), cost_unit=g.cost_unit.iloc[0])
        for c in METRICS:
            row[c + "_mean"], row[c + "_sd"] = g[c].mean(), g[c].std(ddof=1)
        for c in ("exact_PF", "exact_TV", "cost_uniform_corrected"):
            row[c] = g[c].iloc[0] if c in g and g[c].notna().any() else float("nan")
        out.append(row)
    return df, pd.DataFrame(out)


def make_plot(agg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    insts = list(agg.instance.unique())
    fig, axs = plt.subplots(1, len(insts), figsize=(6 * len(insts), 4.6), squeeze=False)
    for ax, inst in zip(axs[0], insts):
        for _, r in agg[agg.instance == inst].iterrows():
            ax.errorbar(r.acceptance_mean, r.TV_to_unif_mean, xerr=r.acceptance_sd,
                        yerr=r.TV_to_unif_sd, fmt="o", ms=6, capsize=3, zorder=3)
            ax.annotate(r.sampler, (r.acceptance_mean, r.TV_to_unif_mean), textcoords="offset points",
                        xytext=(5, 4), fontsize=8)
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlabel("sampled acceptance (mean +/- sd over seeds)")
        ax.set_ylabel("TV(filtered samples, uniform on F)")
        ax.set_title(f"{inst}: acceptance vs TV (error bars = sd over seeds)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--n-seeds", type=int, default=10)
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
        r, t, info = run_instance(label, u, cs, a.shots, a.n_seeds, mi, a.n_starts, a.lam)
        rows += r; trows += t
    df, agg = aggregate(pd.DataFrame(rows))
    tdf = pd.DataFrame(trows)
    agg["_o"] = agg.sampler.map(ORDER.index)
    agg = agg.sort_values(["instance", "_o"], kind="stable").drop(columns="_o")
    df.to_csv(os.path.join(a.out, "ablation_per_seed.csv"), index=False)
    agg.to_csv(os.path.join(a.out, "ablation.csv"), index=False)
    tdf.to_csv(os.path.join(a.out, "ablation_training_cost.csv"), index=False)
    make_plot(agg, os.path.join(a.out, "ablation.png"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    pm = lambda r, c, f="{:.4g}": (f + "+-" + f).format(r[c + "_mean"], r[c + "_sd"])
    tab = pd.DataFrame([dict(instance=r.instance, sampler=r.sampler, acceptance=pm(r, "acceptance", "{:.3f}"),
                             n_feasible=pm(r, "n_feasible", "{:.0f}"), TV=pm(r, "TV_to_unif", "{:.3f}"),
                             TV_minus_floor=pm(r, "TV_minus_floor", "{:.3f}"),
                             coverage=pm(r, "coverage", "{:.3f}"),
                             cost_per_feas=pm(r, "cost_per_feasible", "{:.2f}") + " " + r.cost_unit,
                             exact_PF=r.exact_PF, exact_TV=r.exact_TV,
                             cost_unif_corr=r.cost_uniform_corrected) for r in agg.itertuples()
                        for r in [pd.Series(r._asdict())]])
    print(f"\n=== Filter mode (primary): mean+-sd over {a.n_seeds} sampling seeds, shots={a.shots} ===")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== Repair mode (secondary; mean over seeds) ===")
    print(df.groupby(["instance", "sampler"], sort=False)[
        ["n_repaired", "n_feasible_repair", "TV_after_repair", "TV_floor_repair", "coverage_repair"]
    ].mean().to_string(float_format=lambda x: f"{x:.4g}"))
    print("\n=== Training cost (separate; trained sampler only; trained once, seed 0) ===")
    print(tdf.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    g = lambda i, s, c: float(agg[(agg.instance == i) & (agg.sampler == s)][c].iloc[0])
    print(f"\nNotes (MCMC draws as many samples as rejection kept, burn_in=200, thin=5, "
          "start=feasible_set[0]):")
    print("- dicke_p0 == rejection in distribution (both uniform over C(n,k) then filtered); "
          f"acceptance p0: {g('p0','dicke_p0','acceptance_mean'):.3f} vs {g('p0','rejection','acceptance_mean'):.3f}. "
          "Dicke+filter is NOT a quantum advantage.")
    print("- Island acceptance claims cite EXACT P_F (statevector), not the sampled acceptance "
          "(too few feasible hits at P_F ~ 1e-2..1e-3 for stable sampled estimates).")
    print(f"- MCMC row is a coverage/trap result, not a cost comparison: island coverage "
          f"{g('island','mcmc_swap','coverage_mean'):.3f} vs rejection {g('island','rejection','coverage_mean'):.3f}; "
          "its cost is in chain moves, not comparable to shots.")
    for i in ("p0", "island"):
        print(f"- trained layers on {i}: exact P_F {g(i,'trained','exact_PF'):.3f} vs Dicke "
              f"{g(i,'dicke_p0','exact_PF'):.3f}; exact TV {g(i,'trained','exact_TV'):.3f} vs Dicke "
              f"{g(i,'dicke_p0','exact_TV'):.3f}; uniform-corrected cost "
              f"{g(i,'trained','cost_uniform_corrected'):.2f} vs Dicke "
              f"{g(i,'dicke_p0','cost_uniform_corrected'):.2f} (lower is better).")
    pf_t, pf_d = g("p0", "trained", "exact_PF"), g("p0", "dicke_p0", "exact_PF")
    tc = float(tdf[tdf.instance == "p0"].exact_evals_x_shots_lower_bound.iloc[0])
    save = 1 / pf_d - 1 / pf_t
    if save > 0:
        print(f"- BREAK-EVEN (p0): training = {tc:.0f} shot-equivalents (lower bound); trained saves "
              f"1/P_F(Dicke) - 1/P_F(trained) = {save:.3f} shots per feasible sample, so it needs "
              f"~{tc/save:.0f} feasible samples to pay back, even ignoring uniformity loss.")
    else:
        print("- BREAK-EVEN (p0): trained P_F does not beat Dicke; no break-even.")
    print("- Repair mode biases the distribution (see TV_after_repair); filter is primary.")
    print("- No quantum advantage claimed at this scale; see MPS sweep.")


if __name__ == "__main__":
    main()
