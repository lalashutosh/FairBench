"""AA-3: wall-clock estimate of exact-uniform sampling by amplitude amplification vs
classical rejection sampling, and the break-even P_F.  Outputs results/aa_*.{csv,json,png}.

Oracle path used for circuit counts: "formula" (exact quantised counts, serial-depth upper bound).
Per-sample time = expected T-depth per sample (attempts * prep depth + oracle calls * iterate depth)
x layer time, and for factory-limited also T-count / factory throughput.
"""
from __future__ import annotations
import csv, json, math, multiprocessing as mp, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fairbench.instances import scaled_family
from fairbench.baselines import random_k_subsets
from fairbench.ft.resources import resource_estimate, dicke_cost
from fairbench.ft.oracle import quantise, DEFAULT_BITS
from fairbench.ft.resources import oracle_formula_counts

RES = ROOT / "results"
NS = [50, 100, 150, 200]
SCHEDS = ["known", "bbht", "fixed_point"]

# ---- wall-clock assumption sets (ASSUMPTIONS, not measurements). Values paraphrased from:
#  * Babbush et al. 2021 PRX Quantum 2, 010103 "Focus beyond quadratic speedups for error-corrected
#    quantum advantage": logical clock ~ 1 MHz-ish at best (surface code, ~1 us per logical op layer),
#    classical per-iterate overheads make quadratic speedups hard to realise.
#  * Gidney & Ekera 2021 Quantum 5, 433: surface-code cycle 1 us, reaction time ~10 us,
#    CCZ/T factories dominate footprint; time ~ Toffoli count x reaction time when factory-limited.
#  * Beverland et al. 2022 (arXiv:2211.07629, Azure Quantum resource estimator): presets from
#    ~100 ns (Majorana) to ~1 us (superconducting) physical ops; logical cycle ~ d x op time,
#    i.e. tens of us per logical layer at d~15-25 for gate-based qubits.
ASSUMPTIONS = {
    "optimistic": dict(t_layer_us=1.0, factory_T_per_us=None,
        note="1 us per T-depth layer (fast logical clock/reaction), unlimited T factories"),
    "moderate": dict(t_layer_us=10.0, factory_T_per_us=None,
        note="10 us per T-depth layer (reaction-limited, Gidney-Ekera-like), ample factories"),
    "factory_limited": dict(t_layer_us=1.0, factory_T_per_us=1.0,
        note="1 us/layer but total T throughput 1 T/us (modest factory bank); time=max(depth*1us, T/1 per us)"),
}


def quantum_time_s(rep, dk) -> dict:
    A = rep.expected_attempts
    Oc = rep.expected_oracle_calls_per_sample
    td_prep = dk.t_depth(rep.t_per_rotation, rep.assumptions["t_depth_per_toffoli"])
    depth = A * td_prep + Oc * rep.t_depth_per_iterate
    out = {"t_depth_per_sample": depth}
    for name, a in ASSUMPTIONS.items():
        t = depth * a["t_layer_us"] * 1e-6
        if a["factory_T_per_us"]:
            t = max(t, rep.t_count_per_sample / a["factory_T_per_us"] * 1e-6)
        out[name] = t
    return out


# ---- classical measurement
# Two classical baselines, both numpy, both timed per proposal (generate + check):
#  * "naive": random_k_subsets (dense double argsort) + ConstraintSet.check_batch (O(n * sectors))
#  * "lean":  argpartition index draw + O(k) index-based check (bincount sector counts, gathered
#             ESG/carbon sums). Verified equal to cs.check_batch. Compiled code would be faster
#             still; break-even P* scales as c_classical^2, so these are upper bounds on P*.
POOL = 4                      # capped: an uncapped pool of scaled_family builds OOM'd the box


def lean_check(idx, u, cs, si, ns):
    from fairbench.constraints import SectorCap, MinESG, CarbonCap
    B, k = idx.shape
    ok = np.ones(B, bool)
    caps = {c.sector: c.max_count for c in cs.constraints if isinstance(c, SectorCap)}
    if caps:
        cnt = np.zeros((B, ns), np.int32)
        np.add.at(cnt, (np.repeat(np.arange(B), k), si[idx].ravel()), 1)
        capv = np.array([caps.get(f"S{i}", k) for i in range(ns)])
        ok &= (cnt <= capv).all(1)
    for c in cs.constraints:
        if isinstance(c, MinESG):
            ok &= u.esg_score[idx].sum(1) / k >= c.min_avg_score      # same form as reference
        elif isinstance(c, CarbonCap):
            ok &= u.carbon[idx].sum(1) / k <= c.max_avg
    return ok


def _work(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    mode, u, cs, batch, reps, seed = args
    n, k = u.n, cs.cardinality
    si = np.array([int(s[1:]) for s in u.sector]); ns = int(si.max()) + 1
    rng = np.random.default_rng(seed)
    def one(r):
        if mode == "naive":
            return int(cs.check_batch(random_k_subsets(n, k, batch, r), u).sum())
        idx = np.argpartition(rng.random((batch, n)), k - 1, axis=1)[:, :k]
        return int(lean_check(idx, u, cs, si, ns).sum())
    one(0)                                            # warm-up
    t0 = time.perf_counter(); hits = 0
    for r in range(reps):
        hits += one(r + 1)
    return (time.perf_counter() - t0) / (batch * reps), hits / (batch * reps)


def measure_classical(n, batch=20000, reps=5):
    u, cs = scaled_family(n, seed=0)
    # lean check must agree exactly with the reference checker
    si = np.array([int(s[1:]) for s in u.sector]); ns = int(si.max()) + 1
    X = random_k_subsets(n, cs.cardinality, 5000, 123)
    idx = np.nonzero(X)[1].reshape(len(X), cs.cardinality)    # row-major: k indices per row
    assert (lean_check(idx, u, cs, si, ns) == cs.check_batch(X, u)).all(), "lean check != reference"
    out = {}
    for mode in ("naive", "lean"):
        t1, hit = _work((mode, u, cs, batch, reps, 0))
        with mp.get_context("fork").Pool(POOL) as p:
            per = p.map(_work, [(mode, u, cs, batch, reps, s) for s in range(POOL)])
        t_par = float(np.mean([t for t, _ in per]))  # per-proposal time inside each worker loop
        eff = t1 / t_par                              # parallel efficiency (1 = perfect)
        out[mode] = dict(t_check_ns=t1 * 1e9, pool=POOL, parallel_efficiency=eff,
                         allcore_t_check_ns=t1 / (POOL * eff) * 1e9, hit_rate=hit)
    return out


# Quantum-side sensitivity: best-known constants we do NOT model by default (AA-4 finding 6).
# rs_const 1.15: repeat-until-success / mixed-fallback synthesis (~3x fewer T per rotation);
# t_per_toffoli 2: measurement-based uncompute makes ~half the Toffolis T-free (Gidney 2018).
# Log-depth Dicke preparation is NOT modelled (would cut depth further).
QBEST = dict(rs_const=1.15, t_per_toffoli=2)
IDEAL_CORES = 64              # assumed perfectly parallel classical scenario (not measured)


def pstar_crossing(sub, qkey, ckey):
    d = [math.log(r[qkey] / r[ckey]) for r in sub]                  # grid is decreasing P
    for i in range(1, len(d)):
        if d[i - 1] > 0 >= d[i]:
            x0, x1 = math.log(sub[i - 1]["P_F"]), math.log(sub[i]["P_F"])
            return math.exp(x0 + (x1 - x0) * d[i - 1] / (d[i - 1] - d[i]))
    if d[0] <= 0:
        return ">1e-1 (quantum wins everywhere)"
    return "<1e-12"


def main():
    cl = {n: measure_classical(n) for n in NS}
    print("classical:", json.dumps(cl, indent=1))

    pf = list(csv.DictReader(open(RES / "aa_pf_scaling.csv")))
    def pfs(n, q="0.5", sweep="scaling"):
        return {int(r["seed"]): float(r["P_F"]) for r in pf if int(r["n"]) == n and r["esg_q"] == q and r["sweep"] == sweep}

    rows, info, dks, qinfo = [], {}, {}, {}
    for n in NS:
        u, cs = scaled_family(n, seed=0)
        t0 = time.time()
        quant = quantise(u, cs, DEFAULT_BITS)          # superset thresholds at n > 16: F subset of F_q
        oi = oracle_formula_counts(u, cs, DEFAULT_BITS, quant=quant)
        # post-check success factor |F| / |F_q| (float rules reject the oracle's extra states)
        post = quant.n_feasible_eval / max(quant.n_feasible_eval + quant.false_pos, 1)
        info[n] = (u, cs, oi, quant); dks[n] = dicke_cost(n, cs.cardinality)
        qinfo[n] = dict(false_pos=quant.false_pos, false_neg=quant.false_neg,
                        n_feasible_eval=quant.n_feasible_eval, n_eval=quant.n_eval,
                        superset=quant.superset, post_check_factor=post, oracle_toffoli=oi["toffoli"])
        print(f"n={n} oracle ready {time.time()-t0:.1f}s toffoli={oi['toffoli']} "
              f"FP={quant.false_pos} FN={quant.false_neg} post={post:.4f}")
        P = pfs(n)
        # NOTE: oracle / Dicke counts use the seed-0 instance; P_F basis is seed 0 or median of seeds 0-4
        for label, p in (("seed0", P[0]), ("median_seeds0-4", float(np.median(list(P.values()))))):
            pq = p / post                                # AA targets F_q; P_F(q) >= P_F
            for sch in SCHEDS:
                for variant, kw in (("default", {}), ("qbest", QBEST)):
                    rep = resource_estimate(u, cs, schedule=sch, p_f=pq, oracle_info=oi, quant=quant, **kw)
                    qt = quantum_time_s(rep, dks[n])
                    row = dict(n=n, k=cs.cardinality, pf_basis=label, P_F=p, P_Fq=pq, schedule=sch,
                               quantum_constants=variant, iterations=rep.iterations,
                               oracle_calls_per_sample=rep.expected_oracle_calls_per_sample / post,
                               logical_qubits_algorithmic=rep.logical_qubits,
                               T_per_sample=rep.t_count_per_sample / post,
                               T_depth_per_sample=qt["t_depth_per_sample"] / post,
                               dominant=rep.components["dominant"])
                    for base in ("naive", "lean"):
                        row[f"t_classical_{base}_1core_s"] = cl[n][base]["t_check_ns"] * 1e-9 / p
                        row[f"t_classical_{base}_pool{POOL}_s"] = cl[n][base]["allcore_t_check_ns"] * 1e-9 / p
                    for a in ASSUMPTIONS:
                        row[f"t_quantum_{a}_s"] = qt[a] / post
                        row[f"ratio_{a}_q_over_lean1"] = row[f"t_quantum_{a}_s"] / row["t_classical_lean_1core_s"]
                    rows.append(row)
    with open(RES / "aa_estimate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- break-even sweep (P_F hypothetical; post-check factor taken from the n's instance)
    grid = np.logspace(-1, -12, 111)
    be_rows, pstar = [], {}
    for n in (100, 200):
        u, cs, oi, quant = info[n]
        post = qinfo[n]["post_check_factor"]
        for sch in SCHEDS:
            for variant, kw in (("default", {}), ("qbest", QBEST)):
                sub = []
                for p in grid:
                    rep = resource_estimate(u, cs, schedule=sch, p_f=float(p) / post, oracle_info=oi,
                                            quant=quant, **kw)
                    qt = quantum_time_s(rep, dks[n])
                    r = dict(n=n, schedule=sch, quantum_constants=variant, P_F=float(p),
                             iterations=rep.iterations, T_per_sample=rep.t_count_per_sample / post)
                    for base in ("naive", "lean"):
                        c = cl[n][base]["t_check_ns"] * 1e-9 / p
                        r[f"t_classical_{base}_1core_s"] = c
                        r[f"t_classical_{base}_ideal{IDEAL_CORES}_s"] = c / IDEAL_CORES
                    for a in ASSUMPTIONS:
                        r[f"t_quantum_{a}_s"] = qt[a] / post
                    sub.append(r)
                be_rows += sub
                for a in ASSUMPTIONS:
                    for base in ("naive", "lean"):
                        for lab in ("1core", f"ideal{IDEAL_CORES}"):
                            pstar[f"n{n}|{sch}|{variant}|{a}|{base}_{lab}"] = pstar_crossing(
                                sub, f"t_quantum_{a}_s", f"t_classical_{base}_{lab}_s")
        # analytic check for 'known', default constants: sqrt(P*) = c / ((pi/4) C_q)
        rep = resource_estimate(u, cs, schedule="known", p_f=1e-6, oracle_info=oi, quant=quant)
        for a, aa in ASSUMPTIONS.items():
            Cq = rep.t_depth_per_iterate * aa["t_layer_us"] * 1e-6
            if aa["factory_T_per_us"]: Cq = max(Cq, rep.t_per_iterate / aa["factory_T_per_us"] * 1e-6)
            for base in ("naive", "lean"):
                c = cl[n][base]["t_check_ns"] * 1e-9
                pstar[f"n{n}|analytic_known|default|{a}|{base}_1core"] = (c / (math.pi / 4 * Cq)) ** 2
    with open(RES / "aa_breakeven.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(be_rows[0])); w.writeheader(); w.writerows(be_rows)
    json.dump(dict(assumptions=ASSUMPTIONS, quantum_best_constants=QBEST, classical_measured=cl,
                   quantisation=qinfo, breakeven_P_F_star=pstar, oracle_path="formula (serial Toffoli-depth upper bound)",
                   notes=["classical naive = random_k_subsets + ConstraintSet.check_batch; lean = argpartition draw + "
                          "O(k) index check; both numpy, timed per proposal inside the loop; pool = measured "
                          f"{POOL}-process efficiency; ideal{IDEAL_CORES} = ASSUMED perfect scaling, not measured",
                          "P_F from synthetic scaled_family (thresholds at quantiles of attained averages), n<=200 only",
                          "logical qubits are algorithmic only (data + ancilla), excluding magic-state factories and routing",
                          "fixed_point schedule uses p_lower = true P_F (assumes P_F known); a looser bound costs more",
                          "oracle/Dicke counts from seed-0 instance; P_F basis seed 0 or median seeds 0-4",
                          "times/T per sample divided by the post-check factor |F|/|F_q| (superset oracle)"],
                   ideal_cores=IDEAL_CORES, pool=POOL, bits=DEFAULT_BITS, eps_total=1e-3),
              open(RES / "aa_assumptions.json", "w"), indent=1, default=str)

    # ---- plot (default quantum constants; qbest shown as the green dotted band edge)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    cols = {"optimistic": "#1b9e77", "moderate": "#d95f02", "factory_limited": "#7570b3"}
    tight = {q: float(np.median(list(pfs(100, q, "tightness").values()))) for q in ("0.5", "0.7", "0.85", "0.95")}
    for ax, n in zip(axs, (100, 200)):
        for sch, ls in (("known", "-"), ("bbht", "--")):
            sub = [r for r in be_rows if r["n"] == n and r["schedule"] == sch and r["quantum_constants"] == "default"]
            p = [r["P_F"] for r in sub]
            for a, c in cols.items():
                ax.loglog(p, [r[f"t_quantum_{a}_s"] for r in sub], ls, color=c, label=f"quantum {a} ({sch})")
        sub = [r for r in be_rows if r["n"] == n and r["schedule"] == "known" and r["quantum_constants"] == "qbest"]
        ax.loglog(p, [r["t_quantum_optimistic_s"] for r in sub], ":", color=cols["optimistic"], lw=2,
                  label="quantum optimistic, best-known constants (known)")
        ax.loglog(p, [r["t_classical_naive_1core_s"] for r in sub], color="0.45", lw=1.5, label="classical naive numpy, 1 core")
        ax.loglog(p, [r["t_classical_lean_1core_s"] for r in sub], "k-", lw=2, label="classical lean numpy, 1 core")
        ax.loglog(p, [r[f"t_classical_lean_ideal{IDEAL_CORES}_s"] for r in sub], "k:", lw=2,
                  label=f"classical lean, {IDEAL_CORES} cores (assumed ideal scaling)")
        ax.axvspan(0.01, 0.2, color="gray", alpha=0.2, label="synthetic scaled_family P_F (n<=200)")
        if n == 100:
            for q, pv in tight.items():
                y = cl[100]["lean"]["t_check_ns"] * 1e-9 / pv
                ax.plot(pv, y, "k*", ms=9)
            ax.annotate("stars: tightness q=0.5..0.95", (tight["0.95"], cl[100]["lean"]["t_check_ns"] * 1e-9 / tight["0.95"]),
                        fontsize=7, xytext=(6, -14), textcoords="offset points")
        ax.invert_xaxis(); ax.set_xlabel("P_F (feasible fraction)"); ax.set_title(f"scaled_family n={n}")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("time per feasible sample (s)"); axs[0].legend(fontsize=6.3, loc="upper left")
    fig.suptitle("AA vs rejection sampling: wall-clock per feasible sample (synthetic constraint family)")
    fig.tight_layout(); fig.savefig(RES / "aa_breakeven.png", dpi=140)

    # ---- summary
    print("\nSUMMARY (median seeds, default constants)")
    for r in rows:
        if r["pf_basis"] == "median_seeds0-4" and r["quantum_constants"] == "default":
            print(f"n={r['n']:3d} {r['schedule']:11s} P={r['P_F']:.4f} it={r['iterations']:.1f} Q={r['logical_qubits_algorithmic']} "
                  f"T={r['T_per_sample']:.2e} opt={r['t_quantum_optimistic_s']:.2e} mod={r['t_quantum_moderate_s']:.2e} "
                  f"fac={r['t_quantum_factory_limited_s']:.2e} naive={r['t_classical_naive_1core_s']:.2e} "
                  f"lean={r['t_classical_lean_1core_s']:.2e} q/lean(opt)={r['ratio_optimistic_q_over_lean1']:.1e}")
    print("\nBREAK-EVEN P_F*:")
    for k_, v in pstar.items(): print(k_, v if isinstance(v, str) else f"{v:.3e}")


if __name__ == "__main__":
    main()
