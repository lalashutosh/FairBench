"""QA-3: error-vs-queries scaling of amplitude estimation vs classical rejection MC on the
attribution percentile and null median (EXACT oracle statistics, SIMULATED AE; synthetic data).

(a) headline: percentile abs. error vs total oracle queries (n=16 demo case; n=24,k=6 case)
(b) P_F dependence: queries for percentile error 0.01 vs feasible fraction
(c) null-median cost: queries for rank error <= tol (bisection QAE vs sample-median MC)

Query model (estimators.py): classical query = 1 proposal (random k-subset + feasibility +
score); quantum query = 1 application of Q. Totals include pilot runs and both amplitudes of the
ratio. Percentile = a_G/a_F. Oracle: IdealOracle(mask.mean()) (== SubspaceOracle law, verified
below on a small case). Usage: python scripts/qae_scaling.py [--reps 30] [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_attribution import build_case  # noqa: E402

from fairbench.constraints import ConstraintSet, Exclusion  # noqa: E402
from fairbench.data import synthetic_returns  # noqa: E402
from fairbench.instances import scaled_family  # noqa: E402
from fairbench.qae.attribution_qae import (ExactInstance, classical_percentile,  # noqa: E402
                                           classical_quantile, make_oracle, portfolio_score,
                                           qae_percentile, qae_quantile)
from fairbench.qae.estimators import SubspaceOracle, IdealOracle  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
PREMIUM = 0.08          # demo default
MN = 30                 # MLAE shots per schedule entry


def rngs(*key):
    return np.random.default_rng(np.random.SeedSequence([abs(hash(k)) % (2**31) if isinstance(k, str) else int(k) for k in key]))


def stable(s):
    return sum((i + 1) * ord(c) for i, c in enumerate(s))


# --------------------------------------------------------------------------- instances
def demo_case(n, k, seed=0):
    u, cs, fund = build_case(n, k, seed, 3.0, PREMIUM, 0.70, n_excluded=2 if n == 16 else n // 10)
    inst = ExactInstance.build(u, cs)
    return u, cs, inst, portfolio_score(fund, u, inst.g1)


def tight_case(esg_q, carbon_q, n=24, k=6, seed=0):
    """build_case recipe with adjustable rule tightness; fund = feasible portfolio at rank 0.70
    of the exact feasible scores (pool = the whole feasible set; no rejection pool needed)."""
    u, cs = scaled_family(n, k=k, seed=seed, esg_q=esg_q, carbon_q=carbon_q, n_mc=50_000)
    cs = ConstraintSet(cs.constraints + [Exclusion(np.argsort(u.carbon)[-2:])])
    z = (u.carbon - u.carbon.mean()) / u.carbon.std()
    u.returns = synthetic_returns(u, periods=756, mu_shift=PREMIUM * z, seed=seed + 1)
    inst = ExactInstance.build(u, cs)
    return u, cs, inst, inst.quantile(0.70)


def summarize(q, e):
    q, e = np.asarray(q, float), np.asarray(e, float)
    return dict(q=float(np.median(q)), e50=float(np.median(e)), e25=float(np.quantile(e, .25)),
                e75=float(np.quantile(e, .75)), e90=float(np.quantile(e, .9)), reps=len(e))


# --------------------------------------------------------------------------- percentile curves
def percentile_curves(inst, s, reps, tag, grids=None):
    """-> {method: [point dict]}; error = |p_hat - exact percentile (strict)|."""
    p = inst.percentile(s, "strict")
    g = dict(classical=np.unique(np.round(np.logspace(2, 7, 16)).astype(int)),
             iqae=[0.2, 0.1, 0.05, 0.025, 0.0125, 0.006, 0.003, 0.0015, 0.0008],
             iqae_s20=[0.2, 0.1, 0.05, 0.025, 0.0125, 0.006, 0.003, 0.0015, 0.0008],
             mlae_exp=list(range(3, 14)),
             mlae_power=[10, 20, 40, 80, 160, 320, 640, 1280])
    g.update(grids or {})
    out = {}
    for method, knobs in g.items():
        pts = []
        for knob in knobs:
            r = rngs(stable(tag), stable(method), int(knob * 1e6))
            q, e = [], []
            for _ in range(reps):
                if method == "classical":
                    est = classical_percentile(inst, s, int(knob), r)
                    q.append(int(knob)); e.append(abs(est - p))
                else:
                    kw = (dict(eps=knob) if method.startswith("iqae") else dict(K=int(knob), shots=MN))
                    if method == "iqae_s20":
                        kw["n_shots"] = 20
                    res = qae_percentile(None, None, s, method=method.split("_s")[0], rng=r, inst=inst,
                                         oracle_kind="ideal", **kw)
                    q.append(res.oracle_queries)
                    e.append(abs(res.estimate - p) if np.isfinite(res.estimate) else 0.5)
            d = summarize(q, e); d["knob"] = float(knob)
            pts.append(d)
        out[method] = pts
    return out, p


def loglog_fit(pts, frac=0.6, floor=1e-6):
    """slope of log median error vs log median queries over the last ``frac`` of the points
    (drops the pre-asymptotic start and points with error at the numerical floor)."""
    x = np.array([d["q"] for d in pts]); y = np.array([d["e50"] for d in pts])
    ok = y > floor
    x, y = x[ok], y[ok]
    i0 = int(len(x) * (1 - frac))
    if len(x) - i0 < 3:
        i0 = max(0, len(x) - 3)
    sl, ic = np.polyfit(np.log(x[i0:]), np.log(y[i0:]), 1)
    return float(sl), float(ic)


def interp_q(pts, target, key="e50"):
    """queries at which the (monotonised) error curve crosses ``target`` (log-log interp);
    None if the curve never gets below it, 'below' if already below at the first point."""
    x = np.array([d["q"] for d in pts]); y = np.array([d[key] for d in pts])
    o = np.argsort(x); x, y = x[o], y[o]
    y = np.maximum(np.minimum.accumulate(y), 1e-12)
    if y[-1] > target:
        return None
    if y[0] <= target:
        return float(x[0])
    j = int(np.argmax(y <= target))
    lx = np.interp(np.log(target), [np.log(y[j]), np.log(y[j - 1])], [np.log(x[j]), np.log(x[j - 1])])
    return float(np.exp(lx))


def crossover(q_pts, c_pts):
    """smallest budget at which median quantum error < median classical error and stays below
    for every larger measured budget. Classical curve log-log interpolated."""
    cx = np.log([d["q"] for d in c_pts]); cy = np.log([d["e50"] for d in c_pts])
    qx = np.array([d["q"] for d in q_pts]); qy = np.array([d["e50"] for d in q_pts])
    o = np.argsort(qx); qx, qy = qx[o], qy[o]
    inr = (np.log(qx) >= cx.min()) & (np.log(qx) <= cx.max())
    win = qy < np.exp(np.interp(np.log(qx), cx, cy))
    win = win & inr
    if not win.any():
        return None
    # last index where quantum does not win (within range) -> crossover is the next one
    bad = np.flatnonzero(inr & ~win)
    first = (bad.max() + 1) if bad.size else 0
    return float(qx[first]) if first < len(qx) and win[first] else None


def extrap_cross(fit_q, fit_c):
    """budget where fitted lines err = exp(ic) q^sl intersect."""
    (sq, iq), (sc, ic) = fit_q, fit_c
    if sq >= sc:
        return None
    return float(np.exp((ic - iq) / (sq - sc)))


# --------------------------------------------------------------------------- quantile curves
def quantile_curves(inst, reps, tag, eps_grid, n_grid):
    out = {}
    r = rngs(stable(tag), 1)
    pts = []
    for N in n_grid:
        e = []
        for _ in range(reps):
            est = classical_quantile(inst, 0.5, int(N), r)
            e.append(abs(inst.percentile(est, "strict") - 0.5) if np.isfinite(est) else 0.5)
        d = summarize([N] * reps, e); d["knob"] = float(N); pts.append(d)
    out["classical"] = pts
    pts = []
    for eps in eps_grid:
        r = rngs(stable(tag), 2, int(eps * 1e6))
        q, e = [], []
        for _ in range(reps):
            res = qae_quantile(None, None, 0.5, "iqae", eps=eps, rng=r, inst=inst, oracle_kind="ideal")
            q.append(res.oracle_queries); e.append(res.details["rank_error"] if "rank_error" in res.details else 0.5)
        d = summarize(q, e); d["knob"] = float(eps); pts.append(d)
    out["iqae"] = pts
    pts = []
    for eps in eps_grid:
        r = rngs(stable(tag), 3, int(eps * 1e6))
        q, e = [], []
        for _ in range(reps):
            res = qae_quantile(None, None, 0.5, "iqae", eps=eps, rng=r, inst=inst, oracle_kind="ideal", n_shots=20)
            q.append(res.oracle_queries); e.append(res.details["rank_error"] if "rank_error" in res.details else 0.5)
        d = summarize(q, e); d["knob"] = float(eps); pts.append(d)
    out["iqae_s20"] = pts
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--reps-c", type=int, default=20, help="repetitions for part (c) (slower)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    reps, reps_c = (6, 5) if a.quick else (a.reps, a.reps_c)
    RES.mkdir(exist_ok=True)
    t00 = time.time()
    rows, summ = [], dict(reps=reps, reps_c=reps_c, MLAE_shots_per_m=MN, premium=PREMIUM)

    # sanity: SubspaceOracle (exact Grover) == IdealOracle(mask.mean()) law on the n=16 case
    u16, cs16, inst16, s16 = demo_case(16, 5)
    mk = inst16.mask_below(s16)
    so, io = SubspaceOracle(mk), IdealOracle(float(mk.mean()))
    dev = max(abs(so.prob_good(m) - io.prob_good(m)) for m in (0, 1, 2, 5, 20, 60))
    summ["subspace_vs_ideal_max_dev"] = float(dev)
    print(f"SubspaceOracle vs IdealOracle max |dP| over m<=60: {dev:.2e}")

    # ---------------- (a)
    summ["a"] = {}
    curves_a = {}
    cases = {"n16_k5": (u16, cs16, inst16, s16), "n24_k6": demo_case(24, 6)}
    for name, (u, cs, inst, s) in cases.items():
        t0 = time.time()
        out, p = percentile_curves(inst, s, reps, "a" + name)
        curves_a[name] = out
        fits = {m: loglog_fit(pts) for m, pts in out.items()}
        cross = {m: crossover(out[m], out["classical"]) for m in out if m != "classical"}
        xex = {m: extrap_cross(fits[m], fits["classical"]) for m in out if m != "classical"}
        summ["a"][name] = dict(C=inst.C, P_F=inst.a_F, n_feasible=int(inst.feasible.sum()), percentile=p,
                               slopes={m: f[0] for m, f in fits.items()},
                               crossover_measured=cross, crossover_extrapolated=xex)
        for m, pts in out.items():
            for d in pts:
                rows.append(dict(part="a", case=name, P_F=inst.a_F, method=m, **d))
        print(f"(a) {name}: C={inst.C} P_F={inst.a_F:.4f} exact percentile={p:.4f} ({time.time()-t0:.0f}s)")

    # ---------------- (b)
    t0 = time.time()
    targets = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    cand = {}
    for eq, cq in [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.4), (0.6, 0.3), (0.65, 0.3), (0.7, 0.3),
                   (0.75, 0.25), (0.78, 0.22), (0.8, 0.2), (0.8, 0.25), (0.82, 0.22), (0.78, 0.3), (0.85, 0.2), (0.8, 0.17), (0.83, 0.19), (0.84, 0.22), (0.86, 0.25), (0.88, 0.3)]:
        try:
            c = tight_case(eq, cq)
        except ValueError:
            continue
        if c[2].feasible.sum() >= 60:
            cand[(eq, cq)] = c
    chosen = []
    for tg in targets:
        key = min(cand, key=lambda kq: abs(np.log(cand[kq][2].a_F / tg)))
        if all(abs(np.log(cand[key][2].a_F / cand[c][2].a_F)) > 0.3 for c in chosen):
            chosen.append(key)
    summ["b"] = dict(target_error=0.01, cases=[])
    bres = []
    for key in chosen:
        u, cs, inst, s = cand[key]
        out, p = percentile_curves(inst, s, reps, f"b{key}", grids=dict(
            classical=np.unique(np.round(np.logspace(2, 8.3, 20)).astype(int)),
            iqae=[0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.006, 0.003, 0.0015],
            iqae_s20=[0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.006, 0.003, 0.0015],
            mlae_exp=list(range(3, 14)), mlae_power=[]))
        out.pop("mlae_power")
        d = dict(esg_q=key[0], carbon_q=key[1], P_F=inst.a_F, n_feasible=int(inst.feasible.sum()),
                 q50={m: interp_q(pts, 0.01) for m, pts in out.items()},
                 q90={m: interp_q(pts, 0.01, "e90") for m, pts in out.items()})
        summ["b"]["cases"].append(d); bres.append(d)
        for m, pts in out.items():
            for pt in pts:
                rows.append(dict(part="b", case=f"esg{key[0]}_carb{key[1]}", P_F=inst.a_F, method=m, **pt))
        print(f"(b) esg_q={key[0]} carbon_q={key[1]}: P_F={inst.a_F:.4g} q@0.01 median-err: {d['q50']}")
    fitb = {}
    for crit in ("q50", "q90"):
        for m in ("classical", "iqae", "iqae_s20", "mlae_exp"):
            xs = [(b["P_F"], b[crit][m]) for b in bres if b[crit][m]]
            if len(xs) >= 3:
                fitb[f"{crit}:{m}"] = float(np.polyfit(np.log([x for x, _ in xs]), np.log([y for _, y in xs]), 1)[0])
    summ["b"]["exponent_of_P_F"] = fitb
    print(f"(b) fitted exponents (queries ~ P_F^x): {fitb}  ({time.time()-t0:.0f}s)")

    # ---------------- (c)
    t0 = time.time()
    summ["c"] = {}
    ccases = dict(cases)
    for b in bres:
        if b["P_F"] < 0.05 and b["P_F"] > 0.002:
            ccases[f"PF{b['P_F']:.4f}"] = cand[(b["esg_q"], b["carbon_q"])]
    curves_c = {}
    for name, (u, cs, inst, s) in ccases.items():
        out = quantile_curves(inst, reps_c, "c" + name, [0.25, 0.1, 0.05, 0.025, 0.0125, 0.006, 0.003],
                              np.unique(np.round(np.logspace(2, 6.5, 14)).astype(int)))
        curves_c[name] = out
        d = dict(P_F=inst.a_F, tol_cost={})
        for tol in (0.02, 0.01):
            qc, qq, q2 = (interp_q(out[m], tol) for m in ("classical", "iqae", "iqae_s20"))
            d["tol_cost"][str(tol)] = dict(classical=qc, iqae=qq, iqae_s20=q2, ratio_q_over_c=(qq / qc if qc and qq else None), ratio_s20_over_c=(q2 / qc if qc and q2 else None))
        summ["c"][name] = d
        for m, pts in out.items():
            for pt in pts:
                rows.append(dict(part="c", case=name, P_F=inst.a_F, method=m, **pt))
        print(f"(c) {name}: P_F={inst.a_F:.4g} {d['tol_cost']}")
    print(f"(c) done ({time.time()-t0:.0f}s)")

    # ---------------- outputs
    import csv
    with open(RES / "qae_scaling.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["part", "case", "P_F", "method", "knob", "q", "e50", "e25", "e75", "e90", "reps"])
        w.writeheader(); w.writerows(rows)
    summ["runtime_s"] = time.time() - t00
    (RES / "qae_scaling.json").write_text(json.dumps(summ, indent=2, default=lambda o: None))
    plot(curves_a, summ, bres, curves_c)
    print_table(summ)


COL = dict(classical="#444444", iqae="#0072B2", iqae_s20="#56B4E9", mlae_exp="#D55E00", mlae_power="#009E73")
LAB = dict(classical="classical rejection MC", iqae="IQAE (100 shots/round)", iqae_s20="IQAE (20 shots/round)", mlae_exp="MLAE (exp)", mlae_power="MLAE (power b=0.5)")


def plot(curves_a, summ, bres, curves_c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    for j, name in enumerate(curves_a):
        A = ax[0, j]
        for m, pts in curves_a[name].items():
            x = [d["q"] for d in pts]
            A.fill_between(x, [d["e25"] for d in pts], [d["e75"] for d in pts], color=COL[m], alpha=.15, lw=0)
            sl = summ["a"][name]["slopes"][m]
            A.plot(x, [d["e50"] for d in pts], "o-", ms=3, color=COL[m], label=f"{LAB[m]} (slope {sl:.2f})")
        for m, c in summ["a"][name]["crossover_measured"].items():
            if c:
                A.axvline(c, color=COL[m], ls=":", lw=1)
        s = summ["a"][name]
        A.set(xscale="log", yscale="log", xlabel="oracle queries (total)", ylabel="|percentile error|",
              title=f"{name}: C={s['C']}, P_F={s['P_F']:.3f}, exact pct {s['percentile']:.2f}")
        A.legend(fontsize=7.5); A.grid(alpha=.25, which="both")
    A = ax[0, 2]
    P = [b["P_F"] for b in bres]
    for m in ("classical", "iqae", "iqae_s20", "mlae_exp"):
        y = [b["q50"][m] for b in bres]
        e = summ["b"]["exponent_of_P_F"].get(f"q50:{m}")
        A.plot(P, [v if v else np.nan for v in y], "o-", color=COL[m], label=f"{LAB[m]}" + (f" ~P_F^{e:.2f}" if e is not None else ""))
    A.set(xscale="log", yscale="log", xlabel="feasible fraction P_F", ylabel="queries for median |err| = 0.01",
          title="(b) cost vs rule tightness"); A.legend(fontsize=8); A.grid(alpha=.25, which="both")
    for j, name in enumerate(list(curves_c)[:2]):
        A = ax[1, j]
        for m, pts in curves_c[name].items():
            x = [d["q"] for d in pts]
            A.fill_between(x, [d["e25"] for d in pts], [d["e75"] for d in pts], color=COL[m], alpha=.15, lw=0)
            A.plot(x, [d["e50"] for d in pts], "o-", ms=3, color=COL[m], label={"classical": "sample median (MC)", "iqae": "IQAE bisection (100 shots)", "iqae_s20": "IQAE bisection (20 shots)"}[m])
        A.set(xscale="log", yscale="log", xlabel="oracle queries (total)", ylabel="rank error of null median",
              title=f"(c) null median, {name} (P_F={summ['c'][name]['P_F']:.3f})")
        A.legend(fontsize=8); A.grid(alpha=.25, which="both")
    A = ax[1, 2]
    names = list(curves_c)
    for tol, mk in (("0.02", "o"), ("0.01", "s")):
        A.plot([summ["c"][n]["P_F"] for n in names], [summ["c"][n]["tol_cost"][tol]["classical"] or np.nan for n in names], mk + "-", color=COL["classical"], label=f"classical, rank tol {tol}")
        A.plot([summ["c"][n]["P_F"] for n in names], [summ["c"][n]["tol_cost"][tol]["iqae"] or np.nan for n in names], mk + "-", color=COL["iqae"], label=f"IQAE, rank tol {tol}")
    A.set(xscale="log", yscale="log", xlabel="P_F", ylabel="queries", title="(c) null-median cost vs P_F")
    A.legend(fontsize=8); A.grid(alpha=.25, which="both")
    fig.suptitle("QAE vs classical rejection MC for attribution statistics (exact amplitudes, simulated AE; synthetic data)")
    fig.tight_layout()
    fig.savefig(RES / "qae_scaling.png", dpi=130)


def fmt(v):
    return "n/a" if v is None else f"{v:.3g}"


def print_table(summ):
    print("\n==== SUMMARY ====")
    print(f"reps={summ['reps']} (c: {summ['reps_c']}), runtime {summ['runtime_s']:.0f}s")
    for name, s in summ["a"].items():
        print(f"(a) {name}  P_F={s['P_F']:.4f}  slopes: " + ", ".join(f"{m} {v:.2f}" for m, v in s["slopes"].items()))
        print("    crossover (measured budget / extrapolated from fits): " + ", ".join(
            f"{m} {fmt(s['crossover_measured'][m])} / {fmt(s['crossover_extrapolated'][m])}" for m in s["crossover_measured"]))
    print("(b) P_F      " + "  ".join(f"{m:>12}" for m in ("classical", "iqae", "iqae_s20", "mlae_exp")))
    for b in summ["b"]["cases"]:
        print(f"    {b['P_F']:<9.4g}" + "  ".join(f"{fmt(b['q50'][m]):>12}" for m in ("classical", "iqae", "iqae_s20", "mlae_exp")))
    print("    exponents:", {k: round(v, 2) for k, v in summ["b"]["exponent_of_P_F"].items()})
    for name, s in summ["c"].items():
        for tol, d in s["tol_cost"].items():
            print(f"(c) {name} P_F={s['P_F']:.4f} rank tol {tol}: classical {fmt(d['classical'])}  IQAE {fmt(d['iqae'])} (x{fmt(d['ratio_q_over_c'])})  IQAE-20shot {fmt(d['iqae_s20'])} (x{fmt(d['ratio_s20_over_c'])})")


if __name__ == "__main__":
    main()
