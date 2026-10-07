"""QA-10: cheaper null/benchmark-median estimator -- HYBRID classical warm start + quantum rank
correction (``qae_quantile_hybrid``) vs the old sequential-test bisection (``qae_quantile``) vs the
classical sample median. Same P_F-sweep instances as scripts/qae_scaling.py parts (b)/(d)
(instance builder imported from it). Every query is charged: classical proposals of the warm start
(negative-binomial draw of the proposals needed for m0 feasible ones), the a_F AE, every
rank-correction AE (no free pilots). The a_F run is counted ONCE in the whole attribution (the
percentile already estimates it; ``aF_run`` of qae_quantile_hybrid reuses it).

Measured over `--reps` independent repetitions per knob point (classical: 10x more, cheap):
queries needed for the null-median RANK error (|rank(estimate) - 1/2|) to have median (e50) / 90th
percentile (e90) <= 0.01, by log-log interpolation of the median-query curve over the knob grid.
Whole attribution (tol 0.01, median-error criterion) = percentile + null median + benchmark median:
  rejection  = max(shared-set percentile, shared-set median) + cardinality-only classical median
  MCMC       = read from results/qae_scaling.json part d (not re-run)
  old quantum= q_pct (qae_scaling.json) + old bisection null median + old bisection bench median
  hybrid     = q_pct + hybrid null median (a_F not re-charged) + hybrid bench median
m0 (feasible warm-start samples) is swept; best-m0 per instance and a fixed rule are reported.
Usage: python scripts/qae_hybrid.py [--reps 24] [--quick] [--out qae_hybrid]"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qae_scaling as S  # noqa: E402  (instances, rngs, summarize, interp_q, expo)

from fairbench.qae.attribution_qae import (ExactInstance, RankTable, classical_quantile,  # noqa: E402
                                           exhaustive_cost, iid_shared_sample, qae_quantile,
                                           qae_quantile_hybrid)

RES = Path(__file__).resolve().parent.parent / "results"
TOL = 0.01
M0S = (5, 10, 20, 40, 80, 160)
EPS_H = [0.15, 0.1, 0.07, 0.05, 0.035, 0.025, 0.015, 0.01]
EPS_OLD = [0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.0125]
FIX = 10        # fixed warm-start rule (feasible samples) used for the headline hybrid numbers
CF = 2.0       # a_F relative precision = CF * eps


def stable(*a):
    return S.stable("".join(map(str, a)))


def pt(q, e, knob, qnof=None):
    d = S.summarize(q, e); d["knob"] = float(knob)
    if qnof is not None:
        d["qnof"] = float(np.median(qnof))
    return d


def hybrid_curve(inst, m0, reps, tag, ae="iqae"):
    pts = []
    for eps in EPS_H:
        r = S.rngs(stable(tag, ae), m0, int(eps * 1e6))
        q, e, qn = [], [], []
        for _ in range(reps):
            res = qae_quantile_hybrid(None, None, 0.5, eps, m0, rng=r, inst=inst, oracle_kind="ideal",
                                      rel_F=CF * eps, ae=ae)
            q.append(res.oracle_queries); qn.append(res.oracle_queries - res.details.get("aF_queries", 0))
            e.append(res.details.get("rank_error", 0.5))
        pts.append(pt(q, e, eps, qn))
    return pts


def old_curve(inst, reps, tag):
    pts = []
    for eps in EPS_OLD:
        r = S.rngs(stable(tag, "old"), int(eps * 1e6))
        q, e, qn = [], [], []
        for _ in range(reps):
            res = qae_quantile(None, None, 0.5, "iqae", eps=eps, rng=r, inst=inst, oracle_kind="ideal")
            q.append(res.oracle_queries); e.append(res.details.get("rank_error", 0.5))
        pts.append(pt(q, e, eps))
    return pts


def classical_curve(inst, reps, tag, ngrid):
    pts = []
    for N in ngrid:
        r = S.rngs(stable(tag, "cl"), int(N))
        e = []
        for _ in range(reps):
            est = classical_quantile(inst, 0.5, int(N), r)
            e.append(abs(inst.percentile(est, "strict") - 0.5) if np.isfinite(est) else 0.5)
        pts.append(pt([N] * reps, e, N))
    return pts


def cost_at(pts, tol, key="e50", qkey="q"):
    """interpolated queries (qkey in 'q' | 'qnof') at which error[key] crosses tol."""
    if qkey == "q":
        return S.interp_q(pts, tol, key)
    p2 = [dict(d, q=d["qnof"]) for d in pts]
    return S.interp_q(p2, tol, key)


def best_m0(curves, tol, key, qkey="q"):
    best = (None, None)
    for m0, pts in curves.items():
        c = cost_at(pts, tol, key, qkey)
        if c is not None and (best[0] is None or c < best[0]):
            best = (c, m0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="qae_hybrid")
    a = ap.parse_args()
    reps = 6 if a.quick else a.reps
    reps_cl = 20 if a.quick else 10 * reps
    t00 = time.time()
    RES.mkdir(exist_ok=True)
    old_json = json.load(open(RES / "qae_scaling.json"))

    # ------------------------------------------------ instances (copy of qae_scaling.main selection)
    targets = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    cand = {}
    for eq, cq in [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.4), (0.6, 0.3), (0.65, 0.3), (0.7, 0.3),
                   (0.75, 0.25), (0.78, 0.22), (0.8, 0.2), (0.8, 0.25), (0.82, 0.22), (0.78, 0.3), (0.85, 0.2),
                   (0.8, 0.17), (0.83, 0.19), (0.84, 0.22), (0.86, 0.25), (0.88, 0.3)]:
        try:
            c = S.tight_case(eq, cq)
        except ValueError:
            continue
        if c[2].feasible.sum() >= 60:
            cand[(eq, cq)] = c
    chosen = []
    for tg in targets:
        key = min(cand, key=lambda kq: abs(np.log(cand[kq][2].a_F / tg)))
        if all(abs(np.log(cand[key][2].a_F / cand[c][2].a_F)) > 0.3 for c in chosen):
            chosen.append(key)
    print(f"instances: {[(k, round(cand[k][2].a_F, 5)) for k in chosen]}  ({time.time()-t00:.0f}s)", flush=True)

    rows, summ = [], dict(reps=reps, reps_classical=reps_cl, tol=TOL, rel_F_factor=CF, m0_grid=list(M0S),
                          eps_hybrid=EPS_H, eps_old=EPS_OLD, cases=[])

    def add(part, case, PF, method, pts, m0=""):
        for d in pts:
            rows.append(dict(part=part, case=case, P_F=PF, method=method, m0=m0, **d))

    # ------------------------------------------------ benchmark median (cardinality only; same for all)
    u0, cs0, _, _ = cand[chosen[0]]
    inst_b = ExactInstance.build(u0, cs0, cardinality_only=True)
    ngb = np.unique(np.round(np.logspace(2, 5, 14)).astype(int))
    b_cl = classical_curve(inst_b, reps_cl, "bench", ngb)
    b_old = old_curve(inst_b, reps, "bench")
    b_hy = {m0: hybrid_curve(inst_b, m0, reps, "bench") for m0 in M0S}
    bench = dict(classical=cost_at(b_cl, TOL), old=cost_at(b_old, TOL),
                 hybrid_best=best_m0(b_hy, TOL, "e50"), hybrid_by_m0={m: cost_at(p, TOL) for m, p in b_hy.items()},
                 hybrid_best_e90=best_m0(b_hy, TOL, "e90"), old_e90=cost_at(b_old, TOL, "e90"),
                 classical_e90=cost_at(b_cl, TOL, "e90"))
    summ["benchmark"] = bench
    add("bench", "bench", 1.0, "classical", b_cl); add("bench", "bench", 1.0, "old", b_old)
    for m0, p in b_hy.items():
        add("bench", "bench", 1.0, "hybrid", p, m0)
    print(f"benchmark median: {bench}  ({time.time()-t00:.0f}s)", flush=True)

    # ------------------------------------------------ per P_F instance
    for key in chosen:
        u, cs, inst, s = cand[key]
        tag = f"esg{key[0]}_carb{key[1]}"
        PF, nF = inst.a_F, int(inst.feasible.sum())
        tab = RankTable(inst, s)
        ng = np.unique(np.round(np.logspace(2, 8.3, 22)).astype(int))
        iid_p, iid_m = S.iid_joint_curves(tab, reps_cl, "hyb_iid" + tag, ng)
        old = old_curve(inst, reps, tag)
        hy = {m0: hybrid_curve(inst, m0, reps, tag) for m0 in M0S}
        ml = hybrid_curve(inst, 20, reps, tag, ae="mlae")
        d_old = min(old_json["d"]["cases"], key=lambda c: abs(np.log(c["P_F"] / PF)))
        parts = d_old["tol"][str(TOL)]["parts"]
        q_pct = parts["q_pct"]
        n_pct, n_med = S.qcross(iid_p, TOL), S.qcross(iid_m, TOL)
        med = dict(classical_shared=n_med, classical_shared_e90=S.qcross(iid_m, TOL, "e90"),
                   old=cost_at(old, TOL), old_e90=cost_at(old, TOL, "e90"),
                   hybrid=(cost_at(hy[FIX], TOL), FIX), hybrid_e90=(cost_at(hy[FIX], TOL, "e90"), FIX),
                   hybrid_noF=(cost_at(hy[FIX], TOL, "e50", "qnof"), FIX),
                   hybrid_best=best_m0(hy, TOL, "e50"), hybrid_best_e90=best_m0(hy, TOL, "e90"),
                   hybrid_best_noF=best_m0(hy, TOL, "e50", "qnof"),
                   hybrid_by_m0={m: cost_at(p, TOL) for m, p in hy.items()},
                   hybrid_by_m0_e90={m: cost_at(p, TOL, "e90") for m, p in hy.items()},
                   mlae_m0_20=cost_at(ml, TOL), mlae_m0_20_e90=cost_at(ml, TOL, "e90"),
                   iqae_m0_20=cost_at(hy[20], TOL))
        # whole attribution at TOL
        def tot(*v):
            return None if any(x is None for x in v) else float(sum(x for x in v))
        bn = bench["hybrid_by_m0"][FIX]; bn_best = bench["hybrid_best"][0]
        whole = dict(
            rejection=tot(max(n_pct, n_med) if n_pct and n_med else None, bench["classical"]),
            mcmc=d_old["tol"][str(TOL)]["mcmc"],
            quantum_old=tot(q_pct, med["old"], bench["old"]),
            quantum_old_json=d_old["tol"][str(TOL)]["quantum"],
            quantum_hybrid=tot(q_pct, med["hybrid_noF"][0], bn),
            quantum_hybrid_aF_recharged=tot(q_pct, med["hybrid"][0], bn),
            quantum_hybrid_best_m0=tot(q_pct, med["hybrid_best_noF"][0], bn_best),
            exhaustive=exhaustive_cost(inst), parts=dict(q_pct=q_pct, rej_pct=n_pct, rej_med=n_med,
                                                       hyb_med_noF=med["hybrid_noF"][0], hyb_bench=bn,
                                                       old_med=med["old"], old_bench=bench["old"]))
        c = dict(esg_q=key[0], carbon_q=key[1], P_F=PF, n_feasible=nF, floor=bool(TOL < 1.0 / nF),
                 median=med, whole=whole)
        summ["cases"].append(c)
        add("med", tag, PF, "classical_shared", iid_m);
        add("med", tag, PF, "old", old); add("med", tag, PF, "hybrid_mlae_m0_20", ml)
        for m0, p in hy.items():
            add("med", tag, PF, "hybrid", p, m0)
        print(f"{tag}: P_F={PF:.4g} |F|={nF}\n   median: " +
              json.dumps({k: (round(v) if isinstance(v, float) else v) for k, v in med.items()
                          if k in ("classical_shared", "old", "hybrid", "hybrid_e90", "old_e90", "mlae_m0_20")}, default=str) +
              f"\n   whole: rej={whole['rejection']} mcmc={whole['mcmc']} old={whole['quantum_old']} "
              f"hyb={whole['quantum_hybrid']}  ({time.time()-t00:.0f}s)", flush=True)

    # ------------------------------------------------ exponents (non-floor instances)
    ok = [c for c in summ["cases"] if not c["floor"]]
    P = [c["P_F"] for c in ok]
    fit = {}
    for name, g in (("median:classical_shared", lambda c: c["median"]["classical_shared"]),
                    ("median:old", lambda c: c["median"]["old"]),
                    ("median:hybrid", lambda c: c["median"]["hybrid"][0]),
                    ("median:hybrid_e90", lambda c: c["median"]["hybrid_e90"][0]),
                    ("median:hybrid_best_m0", lambda c: c["median"]["hybrid_best"][0]),
                    ("whole:rejection", lambda c: c["whole"]["rejection"]),
                    ("whole:quantum_old", lambda c: c["whole"]["quantum_old"]),
                    ("whole:quantum_hybrid", lambda c: c["whole"]["quantum_hybrid"])):
        e = S.expo(P, [g(c) for c in ok])
        if e is not None:
            fit[name] = e
    summ["exponent_of_P_F"] = fit
    print("exponents:", {k: round(v, 2) for k, v in fit.items()})
    # best-m0 rule: cost of each fixed m0 relative to per-instance best; and m0*P_F
    rule = {}
    for m0 in M0S:
        rel = [c["median"]["hybrid_by_m0"][m0] / c["median"]["hybrid_best"][0] for c in summ["cases"]
               if c["median"]["hybrid_by_m0"][m0] and c["median"]["hybrid_best"][0]]
        rule[m0] = dict(mean_rel_cost=float(np.mean(rel)), max_rel_cost=float(np.max(rel)))
    summ["m0_rule"] = rule
    summ["best_m0_per_case"] = [(c["P_F"], c["median"]["hybrid_best"][1], c["median"]["hybrid_best_e90"][1]) for c in summ["cases"]]
    summ["fixed_m0"] = FIX
    summ["runtime_s"] = time.time() - t00

    # ------------------------------------------------ outputs
    json.dump(summ, open(RES / f"{a.out}.json", "w"), indent=1, default=str)
    flds = ["part", "case", "P_F", "method", "m0", "knob", "q", "qnof", "e50", "e25", "e75", "e90", "e95", "rms", "reps"]
    with open(RES / f"{a.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    plot(summ, a.out)
    print_table(summ)
    print(f"done {summ['runtime_s']:.0f}s")


def plot(summ, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = [c for c in summ["cases"]]
    P = np.array([c["P_F"] for c in cs])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    col = dict(rej="#1f77b4", mcmc="#7f7f7f", old="#d95f02", hyb="#1b9e77", ex="#999999")

    def line(a, ys, label, color, ls="-", mk="o"):
        x = [p for p, y in zip(P, ys) if y]; y = [v for v in ys if v]
        a.plot(x, y, ls, color=color, marker=mk, label=label, lw=1.8, ms=5)

    A = ax[0]
    line(A, [c["median"]["classical_shared"] for c in cs], "classical shared-sample median", col["rej"])
    line(A, [c["median"]["old"] for c in cs], "quantum bisection (old)", col["old"])
    line(A, [c["median"]["hybrid"][0] for c in cs], f"hybrid (m0={FIX}), e50<=0.01", col["hyb"])
    line(A, [c["median"]["hybrid_e90"][0] for c in cs], f"hybrid (m0={FIX}), e90<=0.01", col["hyb"], "--", "s")
    A.set_title("Null-median cost (rank error 0.01)")
    B = ax[1]
    line(B, [c["whole"]["rejection"] for c in cs], "rejection", col["rej"])
    line(B, [c["whole"]["mcmc"] for c in cs], "swap-MCMC (from qae_scaling)", col["mcmc"], "-", "^")
    line(B, [c["whole"]["quantum_old"] for c in cs], "quantum, old bisection", col["old"])
    line(B, [c["whole"]["quantum_hybrid"] for c in cs], "quantum, hybrid median", col["hyb"])
    line(B, [c["whole"]["exhaustive"] for c in cs], "exhaustive enumeration", col["ex"], ":", "")
    B.set_title("Whole attribution (percentile + 2 medians, tol 0.01)")
    for a_ in ax:
        a_.set_xscale("log"); a_.set_yscale("log"); a_.set_xlabel("feasible fraction P_F")
        a_.set_ylabel("queries"); a_.grid(alpha=.25, which="both"); a_.legend(fontsize=8)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / f"{out}.png", dpi=140)


def fmt(v):
    return "   -   " if v is None else f"{v:9.0f}"


def print_table(summ):
    print("\nNULL-MEDIAN queries for rank error <= 0.01 (e50 | e90)")
    print(f"{'P_F':>8}{'|F|':>7} {'classical':>9}{'old':>10}{'hybrid':>10} m0 | {'cl e90':>9}{'old e90':>10}{'hyb e90':>10} m0")
    for c in summ["cases"]:
        m = c["median"]
        print(f"{c['P_F']:8.4f}{c['n_feasible']:7d}{fmt(m['classical_shared'])}{fmt(m['old'])}{fmt(m['hybrid'][0])}{m['hybrid'][1]!s:>4} |"
              f"{fmt(m['classical_shared_e90'])}{fmt(m['old_e90'])}{fmt(m['hybrid_e90'][0])}{m['hybrid_e90'][1]!s:>4}")
    print("\nWHOLE ATTRIBUTION at tol 0.01")
    print(f"{'P_F':>8}{'rejection':>10}{'MCMC':>10}{'q old':>10}{'q hybrid':>10}{'rej/hyb':>9}{'mcmc/hyb':>9}{'old/hyb':>9}")
    for c in summ["cases"]:
        w = c["whole"]
        r = lambda a, b: f"{a / b:9.2f}" if a and b else "      -  "
        print(f"{c['P_F']:8.4f}{fmt(w['rejection'])}{fmt(w['mcmc'])}{fmt(w['quantum_old'])}{fmt(w['quantum_hybrid'])}"
              f"{r(w['rejection'], w['quantum_hybrid'])}{r(w['mcmc'], w['quantum_hybrid'])}{r(w['quantum_old'], w['quantum_hybrid'])}")


if __name__ == "__main__":
    main()
