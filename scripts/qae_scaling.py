"""QA-3: error-vs-queries scaling of amplitude estimation vs classical rejection MC on the
attribution percentile and null median (EXACT oracle statistics, SIMULATED AE; synthetic data).

Classical baselines (all charged in the same query unit, one proposal / proposed move = 1 query):
  * "classical i.i.d. rejection (same uniform proposal)" -- N uniform k-subsets, keep feasible;
  * swap-move Metropolis MCMC on F (uniform target), R independent chains each started from a
    state found by rejection (cost charged), burn-in = 1/4 of the chain length (charged); error
    measured EMPIRICALLY over the R chains (can be trapped on island instances);
  * exhaustive enumeration, C(n_allowed, k) queries, exact.

(a) headline: percentile abs. error vs total oracle queries (n=16 demo case; n=24,k=6 case)
(b) P_F dependence: queries for percentile error tol in {0.01, 0.005} vs feasible fraction,
    for rejection / MCMC / IQAE / MLAE and the exhaustive line
(c) null-median cost: queries for rank error <= tol (bisection QAE vs sample-median MC)
(d) WHOLE attribution (percentile + null median + benchmark median) at tol in {0.01, 0.005}:
    quantum = sum of its three runs; rejection = ONE shared i.i.d. sample set for percentile and
    null median (+ separate cardinality-only i.i.d. set for the benchmark median); MCMC = one
    chain set for both (+ same cardinality-only i.i.d. benchmark set).
Points with tol < 1/|F| are flagged (granularity floor; enumeration is then trivially cheaper).

Query model (estimators.py): classical query = 1 proposal (random k-subset + feasibility +
score); quantum query = 1 application of Q. Totals include pilot runs and both amplitudes of the
ratio. Percentile = a_G/a_F. Oracle: IdealOracle(mask.mean()) (== SubspaceOracle law, verified
below on a small case). Usage: python scripts/qae_scaling.py [--reps 30] [--quick] [--out qae_scaling]
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
from fairbench.qae.attribution_qae import (ExactInstance, RankTable, SwapChainBank,  # noqa: E402
                                           classical_percentile, classical_quantile,
                                           exhaustive_cost, iid_shared_sample, make_oracle,
                                           portfolio_score, qae_percentile, qae_quantile,
                                           swap_connectivity)
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
    ok = np.isfinite(e)
    e = np.where(ok, e, 0.5)          # failed run (no feasible sample etc.) counts as a 0.5 error
    return dict(q=float(np.median(q)), e50=float(np.median(e)), e25=float(np.quantile(e, .25)),
                e75=float(np.quantile(e, .75)), e90=float(np.quantile(e, .9)),
                e95=float(np.quantile(e, .95)), rms=float(np.sqrt(np.mean(e ** 2))), reps=len(e))


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
def quantile_curves(inst, reps, tag, eps_grid, n_grid, s20=True):
    """null-median (or benchmark-median) rank error: classical sample median (separate set),
    IQAE bisection (100 shots/round [and 20 shots/round])."""
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
    for name, ns, key in (("iqae", 100, 2), ("iqae_s20", 20, 3)):
        if name == "iqae_s20" and not s20:
            continue
        pts = []
        for eps in eps_grid:
            r = rngs(stable(tag), key, int(eps * 1e6))
            q, e = [], []
            for _ in range(reps):
                res = qae_quantile(None, None, 0.5, "iqae", eps=eps, rng=r, inst=inst,
                                   oracle_kind="ideal", n_shots=ns)
                q.append(res.oracle_queries)
                e.append(res.details["rank_error"] if "rank_error" in res.details else 0.5)
            d = summarize(q, e); d["knob"] = float(eps); pts.append(d)
        out[name] = pts
    return out


# --------------------------------------------------------------------------- shared-set curves
def iid_joint_curves(tab, reps, tag, n_grid):
    """ONE shared i.i.d. rejection set per (rep, N): percentile abs err and null-median rank err
    from the same proposals. -> (pct_pts, med_pts)"""
    pp, mp = [], []
    for N in n_grid:
        r = rngs(stable(tag), int(N))
        e1, e2 = [], []
        for _ in range(reps):
            p, m, _ = iid_shared_sample(tab, int(N), r)
            e1.append(abs(p - tab.exact_pct) if np.isfinite(p) else 0.5)
            e2.append(m if np.isfinite(m) else 0.5)
        for pts, e in ((pp, e1), (mp, e2)):
            d = summarize([N] * reps, e); d["knob"] = float(N); pts.append(d)
    return pp, mp


def mcmc_curves(tab, R, t_max, tag):
    """swap-MCMC (R independent chains, charged init + 1/4 burn-in): (pct_pts, med_pts).
    Query of a rep at checkpoint T = T + its initial-rejection cost."""
    o = SwapChainBank(tab).run(R, t_max, rngs(stable(tag), 77))
    pp, mp = [], []
    for i, T in enumerate(o["T"]):
        q = T + o["init_q"]
        for pts, e in ((pp, o["pct_err"][:, i]), (mp, o["med_err"][:, i])):
            d = summarize(q, e); d["knob"] = float(T); pts.append(d)
    return pp, mp


def tmax_for(P_F):
    return 2 ** (16 if P_F > 0.05 else 18 if P_F > 0.005 else 20)


TOLS = (0.01, 0.005)
BIG = None


def qcross(pts, tol, key="e50"):
    return interp_q(pts, tol, key) if pts else None


def tot(*vals):
    return None if any(v is None for v in vals) else float(sum(vals))


def vmax(*vals):
    return None if any(v is None for v in vals) else float(max(vals))


def expo(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if y]
    if len(pts) < 3:
        return None
    return float(np.polyfit(np.log([a for a, _ in pts]), np.log([b for _, b in pts]), 1)[0])


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--reps-c", type=int, default=20, help="repetitions for quantile (bisection) curves (slower)")
    ap.add_argument("--chains", type=int, default=48, help="independent swap-MCMC chains per instance")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="qae_scaling")
    a = ap.parse_args()
    reps, reps_c, R = (6, 5, 12) if a.quick else (a.reps, a.reps_c, a.chains)
    RES.mkdir(exist_ok=True)
    t00 = time.time()
    rows, summ = [], dict(reps=reps, reps_c=reps_c, chains=R, MLAE_shots_per_m=MN, premium=PREMIUM,
                          mcmc_burn_in="T/4 moves per checkpoint T (charged); start by rejection (charged)")

    def add_rows(part, case, inst, curves):
        for m, pts in curves.items():
            for d in pts:
                rows.append(dict(part=part, case=case, P_F=inst.a_F, method=m, **d))

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
                               crossover_measured=cross, crossover_extrapolated=xex,
                               exhaustive=exhaustive_cost(inst), granularity=1.0 / int(inst.feasible.sum()))
        add_rows("a", name, inst, out)
        print(f"(a) {name}: C={inst.C} P_F={inst.a_F:.4f} exact percentile={p:.4f} ({time.time()-t0:.0f}s)")

    # ---------------- instances for (b)/(c)/(d)
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

    # benchmark-median curves (cardinality-only; depends only on the return data) -- cached by data
    bench_cache = {}

    def bench_curves(u, cs):
        inst_b = ExactInstance.build(u, cs, cardinality_only=True)
        key = (round(float(inst_b.g1.sum()), 9), inst_b.C)
        if key not in bench_cache:
            bench_cache[key] = quantile_curves(inst_b, reps_c, "bench" + str(key), [0.25, 0.1, 0.05, 0.025, 0.0125, 0.006, 0.003],
                                               np.unique(np.round(np.logspace(2, 6.5, 14)).astype(int)), s20=False)
        return bench_cache[key]

    summ["b"] = dict(tols=list(TOLS), cases=[])
    summ["d"] = dict(tols=list(TOLS), cases=[])
    summ["trap"] = []
    bres, curves_b, qcurves = [], {}, {}
    for key in chosen:
        u, cs, inst, s = cand[key]
        tag = f"esg{key[0]}_carb{key[1]}"
        nF = int(inst.feasible.sum())
        tab = RankTable(inst, s)
        out, p = percentile_curves(inst, s, reps, f"b{key}", grids=dict(
            classical=np.unique(np.round(np.logspace(2, 8.3, 20)).astype(int)),
            iqae=[0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.006, 0.003, 0.0015],
            iqae_s20=[0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.006, 0.003, 0.0015],
            mlae_exp=list(range(3, 14)), mlae_power=[]))
        out.pop("mlae_power")
        # shared i.i.d. set + MCMC (pct and median curves from the same samples)
        ng = np.unique(np.round(np.logspace(2, 8.3, 20)).astype(int))
        iid_p, iid_m = iid_joint_curves(tab, reps, "iid" + tag, ng)
        mc_p, mc_m = mcmc_curves(tab, R, tmax_for(inst.a_F), "mc" + tag)
        conn = swap_connectivity(inst)
        out["mcmc"] = mc_p
        curves_b[tag] = out
        # null-median quantum (+classical sample-median) curves and benchmark curves
        qc = quantile_curves(inst, reps_c, "c" + tag, [0.25, 0.1, 0.05, 0.025, 0.0125, 0.006, 0.003],
                             np.unique(np.round(np.logspace(2, 6.5, 14)).astype(int)), s20=False)
        qcurves[tag] = qc
        bc = bench_curves(u, cs)
        exh = exhaustive_cost(inst)
        b = dict(esg_q=key[0], carbon_q=key[1], P_F=inst.a_F, n_feasible=nF, exhaustive=exh,
                 granularity=1.0 / nF, connectivity=conn, q50={}, q95={}, rms_at_tol={}, whole={}, floor={})
        d = dict(P_F=inst.a_F, n_feasible=nF, exhaustive=exh, tol={})
        for tol in TOLS:
            k = str(tol)
            b["floor"][k] = bool(tol < 1.0 / nF)
            b["q50"][k] = {m: qcross(pts, tol) for m, pts in out.items()}
            b["q50"][k]["classical"] = qcross(out["classical"], tol)
            b["q95"][k] = {m: qcross(out[m], tol, "e95") for m in ("iqae", "iqae_s20", "mlae_exp")}
            # RMS / q95 of the quantum error at the budget where its MEDIAN error crosses tol
            for m in ("iqae", "iqae_s20"):
                qq = b["q50"][k][m]
                if qq:
                    j = int(np.argmin([abs(np.log(x["q"] / qq)) for x in out[m]]))
                    b["rms_at_tol"].setdefault(k, {})[m] = dict(q=out[m][j]["q"], e50=out[m][j]["e50"],
                                                                 rms=out[m][j]["rms"], e95=out[m][j]["e95"])
            # whole attribution
            qb_c, qb_q = qcross(bc["classical"], tol), qcross(bc["iqae"], tol)
            n_pct, n_med = qcross(iid_p, tol), qcross(iid_m, tol)
            m_pct, m_med = qcross(mc_p, tol), qcross(mc_m, tol)
            q_pct, q_med = b["q50"][k]["iqae"], qcross(qc["iqae"], tol)
            w = dict(rejection=tot(vmax(n_pct, n_med), qb_c), mcmc=tot(vmax(m_pct, m_med), qb_c),
                     quantum=tot(q_pct, q_med, qb_q), exhaustive=exh,
                     parts=dict(rej_pct=n_pct, rej_med=n_med, mcmc_pct=m_pct, mcmc_med=m_med,
                                bench_classical=qb_c, q_pct=q_pct, q_med=q_med, q_bench=qb_q))
            d["tol"][k] = w
        b["floor_flag"] = b["floor"]
        # trapping: asymptotic empirical MCMC error
        last_p, last_m = mc_p[-1], mc_m[-1]
        b["mcmc_final"] = dict(T=last_p["knob"], pct_e50=last_p["e50"], pct_e90=last_p["e90"], med_e50=last_m["e50"])
        summ["b"]["cases"].append(b); bres.append(b); summ["d"]["cases"].append(d)
        add_rows("b", tag, inst, out)
        add_rows("d_shared_iid", tag, inst, {"iid_pct": iid_p, "iid_med": iid_m, "mcmc_pct": mc_p, "mcmc_med": mc_m})
        add_rows("c", tag, inst, qc)
        print(f"(b) {tag}: P_F={inst.a_F:.4g} |F|={nF} C={exh} comps={conn['n_components']} "
              f"start_cov={conn['start_coverage']:.3f}\n    q@0.01: {b['q50']['0.01']}\n    whole@0.01: {d['tol']['0.01']}"
              f"\n    mcmc final T={last_p['knob']:.0f} pct e50={last_p['e50']:.4f} e90={last_p['e90']:.4f} med e50={last_m['e50']:.4f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    fitb = {}
    P = [b["P_F"] for b in bres]
    for tol in TOLS:
        k = str(tol)
        ok = [i for i, b in enumerate(bres) if not b["floor"][k]]
        for crit, src in (("q50", lambda b, m: b["q50"][k].get(m)), ("q95", lambda b, m: b["q95"][k].get(m))):
            for m in ("classical", "mcmc", "iqae", "iqae_s20", "mlae_exp"):
                if crit == "q95" and m in ("classical", "mcmc"):
                    continue
                e = expo([P[i] for i in ok], [src(bres[i], m) for i in ok])
                if e is not None:
                    fitb[f"tol{k}:{crit}:{m}"] = e
        for m in ("rejection", "mcmc", "quantum"):
            e = expo([P[i] for i in ok], [summ["d"]["cases"][i]["tol"][k][m] for i in ok])
            if e is not None:
                fitb[f"tol{k}:whole:{m}"] = e
    summ["b"]["exponent_of_P_F"] = fitb
    print("exponents (queries ~ P_F^x, floor-flagged excluded):", {k: round(v, 2) for k, v in fitb.items()})

    # ---------------- (c) null median cost on demo cases + b cases (curves reused)
    summ["c"] = {}
    ccases = {}
    for nm_, (u, cs, inst, s) in cases.items():
        qc = quantile_curves(inst, reps_c, "c" + nm_, [0.25, 0.1, 0.05, 0.025, 0.0125, 0.006, 0.003],
                             np.unique(np.round(np.logspace(2, 6.5, 14)).astype(int)))
        ccases[nm_] = (inst, qc)
    curves_c = dict(ccases)
    for b, key in zip(bres, chosen):
        inst = cand[key][2]
        curves_c[f"esg{key[0]}_carb{key[1]}"] = (inst, qcurves[f"esg{key[0]}_carb{key[1]}"])
    seen = []
    for name, (inst, out) in curves_c.items():
        dup = any(abs(np.log(inst.a_F / pf)) < 0.02 for pf in seen)
        seen.append(inst.a_F)
        d = dict(P_F=inst.a_F, duplicate_of_earlier=bool(dup), tol_cost={})
        for tol in (0.02, 0.01, 0.005):
            qc_, qq = qcross(out["classical"], tol), qcross(out["iqae"], tol)
            q2 = qcross(out.get("iqae_s20"), tol)
            d["tol_cost"][str(tol)] = dict(classical=qc_, iqae=qq, iqae_s20=q2,
                                           ratio_q_over_c=(qq / qc_ if qc_ and qq else None))
        summ["c"][name] = d
        if name in cases:
            add_rows("c", name, inst, out)
    # ---------------- outputs
    import csv
    flds = ["part", "case", "P_F", "method", "knob", "q", "e50", "e25", "e75", "e90", "e95", "rms", "reps"]
    with open(RES / f"{a.out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flds)
        w.writeheader(); w.writerows(rows)
    summ["runtime_s"] = time.time() - t00
    (RES / f"{a.out}.json").write_text(json.dumps(summ, indent=2, default=lambda o: None))
    plot(curves_a, summ, bres, curves_c, a.out)
    print_table(summ)


COL = dict(classical="#444444", mcmc="#CC79A7", iqae="#0072B2", iqae_s20="#56B4E9", mlae_exp="#D55E00",
           mlae_power="#009E73", rejection="#444444", quantum="#0072B2", exhaustive="#000000")
LAB = dict(classical="classical i.i.d. rejection (same uniform proposal)", mcmc="swap-MCMC (empirical err)",
           iqae="IQAE (100 shots/round)", iqae_s20="IQAE (20 shots/round)", mlae_exp="MLAE (exp)",
           mlae_power="MLAE (power b=0.5)", rejection="i.i.d. rejection (one shared set)",
           quantum="QAE (IQAE, sum of 3 runs)", exhaustive="exhaustive C(n',k)")


def plot(curves_a, summ, bres, curves_c, out_name="qae_scaling"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    fig, ax = plt.subplots(3, 3, figsize=(18, 14))
    for j, name in enumerate(curves_a):
        A = ax[0, j]
        s = summ["a"][name]
        for m, pts in curves_a[name].items():
            x = [d["q"] for d in pts]
            A.fill_between(x, [d["e25"] for d in pts], [d["e75"] for d in pts], color=COL[m], alpha=.15, lw=0)
            A.plot(x, [d["e50"] for d in pts], "o-", ms=3, color=COL[m], label=f"{LAB[m]} (slope {s['slopes'][m]:.2f})")
        A.axhline(s["granularity"], color="gray", ls="--", lw=1, label="1/|F| granularity floor")
        A.axvline(s["exhaustive"], color="k", ls="--", lw=1, label=f"exhaustive C={s['exhaustive']}")
        A.set(xscale="log", yscale="log", xlabel="oracle queries (total)", ylabel="|percentile error| (median, IQR band)",
              title=f"(a) {name}: C={s['C']}, P_F={s['P_F']:.3f}, pct {s['percentile']:.2f}")
        A.legend(fontsize=7); A.grid(alpha=.25, which="both")
    P = [b["P_F"] for b in bres]

    def cost_panel(A, getter, tol, methods, title):
        k = str(tol)
        for m in methods:
            ys = [getter(b, m, k) if not b["floor"][k] else None for b in bres]
            A.plot(P, [v if v else np.nan for v in ys], "o-", color=COL[m], ls="--" if m == "exhaustive" else "-",
                   label=LAB[m] + (f" ~P_F^{summ['b']['exponent_of_P_F'][f'tol{k}:{title[1]}:{m}']:.2f}"
                                   if f'tol{k}:{title[1]}:{m}' in summ['b']['exponent_of_P_F'] else ""))
        fl = [p for p, b in zip(P, bres) if b["floor"][k]]
        if fl:
            A.axvspan(min(fl) * 0.8, max(fl) * 1.25, color="gray", alpha=.12, label=f"tol < 1/|F| (floor): dropped")
        A.set(xscale="log", yscale="log", xlabel="feasible fraction P_F", ylabel=f"queries for median err <= {tol}",
              title=title[0] + f", tol {tol}")
        A.xaxis.set_minor_formatter(NullFormatter())
        A.set_xticks(P); A.set_xticklabels([f"{p:.3g}" for p in P], rotation=45, fontsize=8)
        A.legend(fontsize=6.5); A.grid(alpha=.25, which="both")

    cost_panel(ax[0, 2], lambda b, m, k: b["exhaustive"] if m == "exhaustive" else b["q50"][k].get(m), 0.01,
               ["classical", "mcmc", "iqae", "iqae_s20", "mlae_exp", "exhaustive"], ("(b) percentile only", "q50"))
    dd = summ["d"]["cases"]
    for jj, tol in enumerate(TOLS):
        cost_panel(ax[1, jj], lambda b, m, k, dd=dd, bres=bres: (b["exhaustive"] if m == "exhaustive" else dd[bres.index(b)]["tol"][k][m]),
                   tol, ["rejection", "mcmc", "quantum", "exhaustive"], ("(d) WHOLE attribution", "whole"))
    # trapping panel
    A = ax[1, 2]
    A.plot(P, [b["connectivity"]["start_coverage"] for b in bres], "o-", color=COL["mcmc"], label="expected swap-reachable fraction of F")
    A.plot(P, [b["connectivity"]["largest"] for b in bres], "s--", color="gray", label="largest swap component")
    A.set(xscale="log", xlabel="feasible fraction P_F", ylabel="fraction of F", ylim=(0, 1.05),
          title="swap-MCMC trapping (QAE has no such restriction)")
    A.xaxis.set_minor_formatter(NullFormatter()); A.set_xticks(P); A.set_xticklabels([f"{p:.3g}" for p in P], rotation=45, fontsize=8)
    A.legend(fontsize=8); A.grid(alpha=.25, which="both")
    names = list(curves_c)
    for j, name in enumerate(list(curves_c)[:2]):
        A = ax[2, j]
        inst, out = curves_c[name]
        for m, pts in out.items():
            x = [d["q"] for d in pts]
            A.fill_between(x, [d["e25"] for d in pts], [d["e75"] for d in pts], color=COL[m], alpha=.15, lw=0)
            A.plot(x, [d["e50"] for d in pts], "o-", ms=3, color=COL[m], label={"classical": "sample median (i.i.d.)", "iqae": "IQAE bisection (100 shots)", "iqae_s20": "IQAE bisection (20 shots)"}[m])
        A.axhline(1.0 / int(inst.feasible.sum()), color="gray", ls="--", lw=1)
        A.set(xscale="log", yscale="log", xlabel="oracle queries (total)", ylabel="rank error of null median",
              title=f"(c) null median, {name} (P_F={summ['c'][name]['P_F']:.3f})")
        A.legend(fontsize=8); A.grid(alpha=.25, which="both")
    A = ax[2, 2]
    names = [n for n in names if not summ["c"][n]["duplicate_of_earlier"]]
    PP = [summ["c"][n]["P_F"] for n in names]
    o = np.argsort(PP)
    for tol, mk in (("0.02", "o"), ("0.01", "s")):
        for m in ("classical", "iqae"):
            A.plot([PP[i] for i in o], [summ["c"][names[i]]["tol_cost"][tol][m] or np.nan for i in o], mk + "-",
                   color=COL[m], label=f"{'i.i.d. sample median' if m == 'classical' else 'IQAE'}, rank tol {tol}")
    A.set(xscale="log", yscale="log", xlabel="P_F", ylabel="queries", title="(c) null-median cost vs P_F (duplicate instances removed)")
    A.xaxis.set_minor_formatter(NullFormatter())
    A.legend(fontsize=8); A.grid(alpha=.25, which="both")
    fig.suptitle("QAE vs classical baselines (i.i.d. rejection, swap-MCMC, enumeration) for attribution statistics\n"
                 "exact amplitudes, simulated AE; synthetic data; MCMC error measured empirically", fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / f"{out_name}.png", dpi=120)


def fmt(v):
    return "n/a" if v is None else f"{v:.3g}"


def print_table(summ):
    print("\n==== SUMMARY ====")
    print(f"reps={summ['reps']} (quantile curves: {summ['reps_c']}), MCMC chains={summ['chains']}, runtime {summ['runtime_s']:.0f}s")
    for name, s in summ["a"].items():
        print(f"(a) {name}  P_F={s['P_F']:.4f} |F|={s['n_feasible']} exhaustive={s['exhaustive']} slopes: " + ", ".join(f"{m} {v:.2f}" for m, v in s["slopes"].items()))
        print("    crossover (measured budget / extrapolated from fits): " + ", ".join(
            f"{m} {fmt(s['crossover_measured'][m])} / {fmt(s['crossover_extrapolated'][m])}" for m in s["crossover_measured"]))
    for tol in summ["b"]["tols"]:
        k = str(tol)
        print(f"\n(b) PERCENTILE ONLY, queries for median |err| <= {tol}   [* = tol < 1/|F| floor]")
        ms = ("classical", "mcmc", "iqae", "iqae_s20", "mlae_exp")
        print(f"    {'P_F':<9}{'|F|':>7}{'C(n,k)':>9}" + "".join(f"{m:>11}" for m in ms) + f"{'iqae q95':>11}")
        for b in summ["b"]["cases"]:
            print(f"    {b['P_F']:<9.4g}{b['n_feasible']:>6}{'*' if b['floor'][k] else ' '}{b['exhaustive']:>9}"
                  + "".join(f"{fmt(b['q50'][k][m]):>11}" for m in ms) + f"{fmt(b['q95'][k]['iqae']):>11}")
        print(f"\n(d) WHOLE ATTRIBUTION (pct + null median + benchmark median), tol {tol}")
        print(f"    {'P_F':<9}{'exhaustive':>11}{'rejection':>11}{'MCMC':>11}{'quantum':>11}{'rej/Q':>8}{'mcmc/Q':>8}")
        for b, d in zip(summ["b"]["cases"], summ["d"]["cases"]):
            w = d["tol"][k]
            r1 = w["rejection"] / w["quantum"] if w["rejection"] and w["quantum"] else None
            r2 = w["mcmc"] / w["quantum"] if w["mcmc"] and w["quantum"] else None
            print(f"    {b['P_F']:<9.4g}{w['exhaustive']:>11}{fmt(w['rejection']):>11}{fmt(w['mcmc']):>11}{fmt(w['quantum']):>11}{fmt(r1):>8}{fmt(r2):>8}"
                  + ("  [floor]" if b["floor"][k] else ""))
    print("\nexponents (queries ~ P_F^x):")
    for k, v in summ["b"]["exponent_of_P_F"].items():
        print(f"    {k:<28}{v:+.2f}")
    print("\nIQAE heavy tails at the median-crossing budget (e50 / rms / q95 of |err|):")
    for b in summ["b"]["cases"]:
        for k, dd in b["rms_at_tol"].items():
            if "iqae" in dd:
                x = dd["iqae"]
                print(f"    P_F={b['P_F']:.4g} tol {k}: q={x['q']:.0f} e50={x['e50']:.4f} rms={x['rms']:.4f} q95={x['e95']:.4f}")
    print("\nMCMC trapping (swap components of F):")
    for b in summ["b"]["cases"]:
        c, f = b["connectivity"], b["mcmc_final"]
        print(f"    P_F={b['P_F']:.4g}: components={c['n_components']} (isolated {c['n_isolated']}) largest={c['largest']:.3f} "
              f"start_coverage={c['start_coverage']:.3f}; MCMC T={f['T']:.0f}: pct err e50={f['pct_e50']:.4f} e90={f['pct_e90']:.4f}, median-rank err e50={f['med_e50']:.4f}")
    for name, s in summ["c"].items():
        for tol, d in s["tol_cost"].items():
            print(f"(c) {name} P_F={s['P_F']:.4f}{' (dup)' if s['duplicate_of_earlier'] else ''} rank tol {tol}: classical {fmt(d['classical'])}  IQAE {fmt(d['iqae'])} (x{fmt(d['ratio_q_over_c'])})")


if __name__ == "__main__":
    main()
