"""QA-10 / QA-11: null/benchmark-median cost of the HYBRID classical-warm-start + quantum
monotone-rank-root estimator (``qae_quantile_hybrid``) vs the old sequential-test bisection
(``qae_quantile``) vs classical rejection and swap-MCMC, and the WHOLE attribution
(percentile + null median + benchmark median) at tol 0.01, by e50 AND e90.

Accounting (QA-11, estimators.py): every quantum shot costs m + 1 oracle queries (m Grover calls
+ 1 evaluation of the measured string = one classical sample); the pre-QA-11 Grover-only count
(sum shots * m) is also recorded ("grover" columns) to show how much the convention moved.
Every query is charged: warm-start proposals (negative-binomial draw for m0 = 10 feasible),
every AE incl. pilots and the hybrid's bisection fallback. a_F is charged ONCE in the whole
attribution: the hybrid null-median curve is measured with ``aF_run`` = a separate IQAE run
(rel. precision 2 eps) whose cost is NOT counted (it is the percentile's a_F run).

Measurement:
* >= 200 reps per knob point for every quantum method (``--reps``); classical rejection costs
  are ANALYTIC (normal approximation; N_F feasible samples needed for e_x <= tol with
  z_50 = 0.6745, z_90 = 1.645: median rank  N_F = (z 0.5 / tol)^2, percentile
  N_F = (z sqrt(p(1-p)) / tol)^2; proposals = N_F / P_F; shared set = max of the two) and
  cross-checked by 1000-rep simulation of the shared i.i.d. set at the analytic budgets.
* crossing budget = log-log regression of mean queries on error over the knob points with
  error in [tol/3, 3 tol] (nearest 4 points if fewer than 3), evaluated at tol -- not the first
  dip below tol. Costs are MEAN queries (expected budget).
* MCMC is classical (unchanged by the accounting): e50 read from results/qae_scaling.json
  part d; e90 fitted from the same run's results/qae_scaling.csv (48 chains, median-T queries).
* floor flags: 1/|F| > tol (rank granule exceeds the tolerance) and 2/|F| > tol.
Usage: python scripts/qae_hybrid.py [--reps 200] [--quick] [--out qae_hybrid]"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qae_scaling as S  # noqa: E402  (instances, rngs, expo)

import fairbench.qae.attribution_qae as A  # noqa: E402
from fairbench.qae.attribution_qae import (ExactInstance, RankTable, exhaustive_cost,  # noqa: E402
                                           iid_shared_sample, qae_percentile, qae_quantile,
                                           qae_quantile_hybrid)

RES = Path(__file__).resolve().parent.parent / "results"
TOL = 0.01
Z50, Z90 = 0.6744897501960817, 1.6448536269514722
M0 = 10            # warm-start feasible samples (fixed rule)
CF = 2.0           # a_F relative precision = CF * eps for the reused a_F run
EPS_PCT = [0.1, 0.07, 0.05, 0.035, 0.025, 0.018, 0.012, 0.008, 0.005]
EPS_H = [0.07, 0.05, 0.035, 0.025, 0.018, 0.012, 0.008, 0.0055]
EPS_OLD = [0.12, 0.07, 0.04, 0.025, 0.015, 0.009]

# ---------------------------------------------------------------- Grover-call counter (old convention)
_G = [0]
_iqae0, _mlae0 = A.iqae, A.mlae


def _iqae_c(*a, **k):
    r = _iqae0(*a, **k); _G[0] += r.extra["grover_calls"]; return r


def _mlae_c(*a, **k):
    r = _mlae0(*a, **k); _G[0] += r.extra["grover_calls"]; return r


A.iqae, A.mlae = _iqae_c, _mlae_c


def fitcross(q, e, tol=TOL):
    """budget at which error e(q) reaches tol: log-log regression of q on e over points with e in
    [tol/3, 3 tol] (nearest 4 in log distance if < 3 such points). None if never <= tol."""
    q, e = np.asarray(q, float), np.asarray(e, float)
    ok = e > 0
    if e.min() > tol:
        return None
    if ok.sum() < 2:
        return float(q[np.argmax(e <= tol)])
    q, e = q[ok], e[ok]
    m = (e > tol / 3) & (e < 3 * tol)
    if m.sum() < 3:
        j = np.argsort(np.abs(np.log(e / tol)))[:4]
        m = np.zeros(len(e), bool); m[j] = True
    if np.ptp(np.log(e[m])) < 1e-9:
        return float(np.exp(np.mean(np.log(q[m]))))
    b, a = np.polyfit(np.log(e[m]), np.log(q[m]), 1)
    return float(np.exp(a + b * np.log(tol)))


def point(knob, q, g, e, extra=None):
    e = np.where(np.isfinite(e), e, 0.5)
    d = dict(knob=float(knob), q=float(np.mean(q)), q_med=float(np.median(q)), q_grover=float(np.mean(g)),
             e50=float(np.median(e)), e90=float(np.quantile(e, .9)), e99=float(np.quantile(e, .99)),
             reps=len(e))
    d.update(extra or {})
    return d


def cross(pts, key, qkey="q"):
    return fitcross([d[qkey] for d in pts], [d[key] for d in pts])


def pct_curve(inst, s, reps, tag):
    ex = inst.percentile(s, "strict")
    pts = []
    for eps in EPS_PCT:
        r = S.rngs(S.stable(tag + "pct"), int(eps * 1e6))
        q, g, e = [], [], []
        for _ in range(reps):
            _G[0] = 0
            res = qae_percentile(None, None, s, "iqae", eps=eps, rng=r, inst=inst, oracle_kind="ideal")
            q.append(res.oracle_queries); g.append(_G[0]); e.append(abs(res.estimate - ex))
        pts.append(point(eps, q, g, np.array(e)))
    return pts


def hybrid_curve(inst, reps, tag):
    pts = []
    card = bool(np.all(inst.feasible))
    oF = A.make_oracle(inst.feasible, "ideal")
    for eps in EPS_H:
        r = S.rngs(S.stable(tag + "hyb"), int(eps * 1e6))
        q, g, e, conv, fb = [], [], [], [], []
        for _ in range(reps):
            aF_run = None if card else A._adaptive_iqae(oF, r, 0.05, rel_eps=min(0.5, CF * eps))[-1]
            _G[0] = 0
            res = qae_quantile_hybrid(None, None, 0.5, eps, M0, rng=r, inst=inst, oracle_kind="ideal",
                                      aF_run=aF_run)
            d = res.details
            q.append(res.oracle_queries); g.append(d.get("classical_queries", 0) + _G[0])
            e.append(d.get("rank_error", 0.5)); conv.append(d.get("converged", False))
            fb.append(d.get("n_fallback", 0) > 0)
        pts.append(point(eps, q, g, np.array(e), dict(converged=float(np.mean(conv)),
                                                       fallback_used=float(np.mean(fb)))))
    return pts


def old_curve(inst, reps, tag):
    pts = []
    for eps in EPS_OLD:
        r = S.rngs(S.stable(tag + "old"), int(eps * 1e6))
        q, g, e = [], [], []
        for _ in range(reps):
            _G[0] = 0
            res = qae_quantile(None, None, 0.5, "iqae", eps=eps, rng=r, inst=inst, oracle_kind="ideal")
            q.append(res.oracle_queries); g.append(_G[0]); e.append(res.details.get("rank_error", 0.5))
        pts.append(point(eps, q, g, np.array(e)))
    return pts


def rejection_analytic(p, PF):
    """proposals for e50 / e90 <= TOL: (percentile, median, shared = max)."""
    out = {}
    for k, z in (("e50", Z50), ("e90", Z90)):
        npct = (z * math.sqrt(p * (1 - p)) / TOL) ** 2 / PF
        nmed = (z * 0.5 / TOL) ** 2 / PF
        out[k] = dict(pct=npct, med=nmed, shared=max(npct, nmed))
    return out


def rejection_check(tab, N, reps, seed):
    r = np.random.default_rng(seed)
    ep, em = [], []
    for _ in range(reps):
        p, m, _ = iid_shared_sample(tab, int(round(N)), r)
        ep.append(abs(p - tab.exact_pct) if np.isfinite(p) else 0.5); em.append(m if np.isfinite(m) else 0.5)
    return dict(N=float(N), pct_e50=float(np.median(ep)), pct_e90=float(np.quantile(ep, .9)),
                med_e50=float(np.median(em)), med_e90=float(np.quantile(em, .9)))


def mcmc_from_scaling(tag):
    """(e50 json, e90 fitted from csv) MCMC shared-chain cost (max(pct, med)) without bench."""
    rows = [r for r in csv.DictReader(open(RES / "qae_scaling.csv"))
            if r["part"] == "d_shared_iid" and r["case"] == tag and r["method"] in ("mcmc_pct", "mcmc_med")]
    out = {}
    for key in ("e50", "e90"):
        c = []
        for meth in ("mcmc_pct", "mcmc_med"):
            rr = [r for r in rows if r["method"] == meth]
            c.append(fitcross([float(r["q"]) for r in rr], [float(r[key]) for r in rr]) if rr else None)
        out[key] = None if any(v is None for v in c) else max(c)
        out[key + "_parts"] = c
    return out


def tot(*v):
    return None if any(x is None for x in v) else float(sum(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="qae_hybrid")
    a = ap.parse_args()
    reps = 20 if a.quick else a.reps
    reps_chk = 200 if a.quick else 1000
    t00 = time.time()
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

    rows = []
    summ = dict(reps=reps, reps_rejection_check=reps_chk, tol=TOL, m0=M0, rel_F_factor=CF,
                eps_pct=EPS_PCT, eps_hybrid=EPS_H, eps_old=EPS_OLD,
                accounting="oracle_queries = sum shots (m+1); *_grover = sum shots m (pre-QA-11)",
                crossing="log-log regression of mean queries on error near tol", cases=[])

    def add(part, case, PF, method, pts):
        for d in pts:
            rows.append(dict(part=part, case=case, P_F=PF, method=method, **d))

    # ------------------------------------------------ benchmark median (cardinality only)
    u0, cs0, _, _ = cand[chosen[0]]
    inst_b = ExactInstance.build(u0, cs0, cardinality_only=True)
    b_old, b_hy = old_curve(inst_b, reps, "bench"), hybrid_curve(inst_b, reps, "bench")
    bench = dict(n_feasible=int(inst_b.C))
    for k in ("e50", "e90"):
        z = Z50 if k == "e50" else Z90
        bench[k] = dict(classical=(z * 0.5 / TOL) ** 2, old=cross(b_old, k), hybrid=cross(b_hy, k),
                        old_grover=cross(b_old, k, "q_grover"), hybrid_grover=cross(b_hy, k, "q_grover"))
    # classical cross-check (cardinality-only sample median, 1000 reps at the analytic budgets)
    chk = {}
    for k in ("e50", "e90"):
        r = np.random.default_rng(99)
        N = int(round(bench[k]["classical"]))
        e = [abs(inst_b.percentile(A.classical_quantile(inst_b, 0.5, N, r), "strict") - 0.5) for _ in range(reps_chk)]
        chk[k] = dict(N=N, e50=float(np.median(e)), e90=float(np.quantile(e, .9)))
    bench["classical_check"] = chk
    bench["hybrid_convergence"] = [(d["knob"], d["converged"], d["fallback_used"]) for d in b_hy]
    summ["benchmark"] = bench
    add("bench", "bench", 1.0, "old", b_old); add("bench", "bench", 1.0, "hybrid", b_hy)
    print(f"benchmark: {json.dumps({k: bench[k] for k in ('e50', 'e90', 'classical_check')}, default=round)}"
          f"  ({time.time()-t00:.0f}s)", flush=True)

    # ------------------------------------------------ per P_F instance
    for key in chosen:
        u, cs, inst, s = cand[key]
        tag = f"esg{key[0]}_carb{key[1]}"
        PF, nF = inst.a_F, int(inst.feasible.sum())
        tab = RankTable(inst, s)
        p_ex = tab.exact_pct
        pc = pct_curve(inst, s, reps, tag)
        old = old_curve(inst, reps, tag)
        hy = hybrid_curve(inst, reps, tag)
        rej = rejection_analytic(p_ex, PF)
        chk = {k: rejection_check(tab, rej[k]["shared"], reps_chk, 1000 + i) for i, k in enumerate(("e50", "e90"))}
        d_old = min(old_json["d"]["cases"], key=lambda c: abs(np.log(c["P_F"] / PF)))
        w_old = d_old["tol"][str(TOL)]
        mc = mcmc_from_scaling(tag)
        c = dict(esg_q=key[0], carbon_q=key[1], P_F=PF, n_feasible=nF, percentile_exact=p_ex,
                 floor_1F=bool(1.0 / nF > TOL), floor_2F=bool(2.0 / nF > TOL), exhaustive=exhaustive_cost(inst),
                 rejection_analytic=rej, rejection_check=chk,
                 hybrid_convergence=[(d["knob"], d["converged"], d["fallback_used"]) for d in hy],
                 old_accounting_json=dict(q_pct=w_old["parts"]["q_pct"], quantum_whole=w_old["quantum"],
                                          mcmc=w_old["mcmc"], rejection=w_old["rejection"]))
        for k in ("e50", "e90"):
            qp, qpg = cross(pc, k), cross(pc, k, "q_grover")
            om, omg = cross(old, k), cross(old, k, "q_grover")
            hm, hmg = cross(hy, k), cross(hy, k, "q_grover")
            B = bench[k]
            c[k] = dict(
                percentile=dict(quantum=qp, quantum_grover=qpg, rejection=rej[k]["pct"]),
                null_median=dict(old=om, old_grover=omg, hybrid=hm, hybrid_grover=hmg, rejection=rej[k]["med"]),
                whole=dict(rejection=rej[k]["shared"] + B["classical"],
                           mcmc=(w_old["mcmc"] if k == "e50" else tot(mc["e90"], B["classical"])),
                           mcmc_e90_parts=mc["e90_parts"] if k == "e90" else None,
                           quantum_old=tot(qp, om, B["old"]), quantum_hybrid=tot(qp, hm, B["hybrid"]),
                           quantum_old_grover=tot(qpg, omg, B["old_grover"]),
                           quantum_hybrid_grover=tot(qpg, hmg, B["hybrid_grover"])))
        summ["cases"].append(c)
        add("pct", tag, PF, "iqae_percentile", pc); add("med", tag, PF, "old_bisection", old)
        add("med", tag, PF, "hybrid", hy)
        w5, w9 = c["e50"]["whole"], c["e90"]["whole"]
        print(f"{tag}: P_F={PF:.4g} |F|={nF} floor(1/|F|>tol)={c['floor_1F']} (2/|F|>tol)={c['floor_2F']}\n"
              f"   e50 whole: rej={w5['rejection']:.0f} mcmc={w5['mcmc']} old={w5['quantum_old']} hyb={w5['quantum_hybrid']}\n"
              f"   e90 whole: rej={w9['rejection']:.0f} mcmc={w9['mcmc']} old={w9['quantum_old']} hyb={w9['quantum_hybrid']}\n"
              f"   rej check: {chk}\n   hybrid conv: {c['hybrid_convergence']}  ({time.time()-t00:.0f}s)", flush=True)

    # ------------------------------------------------ exponents (non-floor instances)
    ok = [c for c in summ["cases"] if not c["floor_2F"]]
    P = [c["P_F"] for c in ok]
    fit = {}
    for k in ("e50", "e90"):
        for m in ("rejection", "mcmc", "quantum_old", "quantum_hybrid"):
            e = S.expo(P, [c[k]["whole"][m] for c in ok])
            if e is not None:
                fit[f"{k}:whole:{m}"] = e
        for m in ("old", "hybrid"):
            e = S.expo(P, [c[k]["null_median"][m] for c in ok])
            if e is not None:
                fit[f"{k}:null_median:{m}"] = e
    summ["exponent_of_P_F"] = fit
    summ["runtime_s"] = time.time() - t00

    json.dump(summ, open(RES / f"{a.out}.json", "w"), indent=1, default=lambda o: None)
    flds = ["part", "case", "P_F", "method", "knob", "q", "q_med", "q_grover", "e50", "e90", "e99", "reps",
            "converged", "fallback_used"]
    with open(RES / f"{a.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    plot(summ, a.out)
    print_table(summ)
    print(f"done {summ['runtime_s']:.0f}s")


def plot(summ, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = summ["cases"]
    P = np.array([c["P_F"] for c in cs])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    col = dict(rejection="#1f77b4", mcmc="#7f7f7f", quantum_old="#d95f02", quantum_hybrid="#1b9e77")
    lab = dict(rejection="i.i.d. rejection (analytic)", mcmc="swap-MCMC (qae_scaling)",
               quantum_old="quantum, old bisection", quantum_hybrid="quantum, hybrid median")
    for A_, k in zip(ax, ("e50", "e90")):
        for m in col:
            ys = [c[k]["whole"][m] for c in cs]
            x = [p for p, y in zip(P, ys) if y]; y = [v for v in ys if v]
            A_.plot(x, y, "-o", color=col[m], label=lab[m], lw=1.8, ms=5)
        A_.plot(P, [c["exhaustive"] for c in cs], ":", color="#999999", label="exhaustive C(n',k)")
        fl = [c["P_F"] for c in cs if c["floor_2F"]]
        for f in fl:
            A_.axvspan(f / 1.15, f * 1.15, color="#eeeeee", zorder=0)
        A_.set_title(f"Whole attribution, {k} <= {summ['tol']} (queries = sum shots (m+1))")
        A_.set_xscale("log"); A_.set_yscale("log"); A_.set_xlabel("feasible fraction P_F")
        A_.grid(alpha=.25, which="both")
        for sp in ("top", "right"):
            A_.spines[sp].set_visible(False)
    ax[0].set_ylabel("expected queries"); ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / f"{out}.png", dpi=140)


def fmt(v):
    return "    never" if v is None else f"{v:9.0f}"


def print_table(summ):
    for k in ("e50", "e90"):
        print(f"\nWHOLE ATTRIBUTION, {k} <= {summ['tol']}  (expected queries; ratios = method / hybrid; "
              "* = 2/|F| > tol resolution floor)")
        print(f"{'P_F':>9}{'|F|':>7}{'rejection':>10}{'MCMC':>10}{'q old':>10}{'q hybrid':>10}"
              f"{'rej/hyb':>9}{'mcmc/hyb':>9}{'old/hyb':>9}")
        for c in summ["cases"]:
            w = c[k]["whole"]
            r = lambda a, b: f"{a / b:9.2f}" if a and b else "      -  "
            fl = "*" if c["floor_2F"] else " "
            print(f"{c['P_F']:8.4f}{fl}{c['n_feasible']:7d}{fmt(w['rejection'])}{fmt(w['mcmc'])}{fmt(w['quantum_old'])}"
                  f"{fmt(w['quantum_hybrid'])}{r(w['rejection'], w['quantum_hybrid'])}{r(w['mcmc'], w['quantum_hybrid'])}"
                  f"{r(w['quantum_old'], w['quantum_hybrid'])}")
    print("\nACCOUNTING CHANGE (e50): queries sum(m+1) vs Grover-only sum(m) [vs qae_scaling.json, old convention]")
    print(f"{'P_F':>9}{'pct new':>10}{'pct grov':>10}{'pct json':>10}{'hyb new':>10}{'hyb grov':>10}{'old new':>10}{'old grov':>10}{'old json':>10}")
    for c in summ["cases"]:
        e, w = c["e50"], c["e50"]["whole"]
        print(f"{c['P_F']:9.4f}{fmt(e['percentile']['quantum'])}{fmt(e['percentile']['quantum_grover'])}"
              f"{fmt(c['old_accounting_json']['q_pct'])}{fmt(w['quantum_hybrid'])}{fmt(w['quantum_hybrid_grover'])}"
              f"{fmt(w['quantum_old'])}{fmt(w['quantum_old_grover'])}{fmt(c['old_accounting_json']['quantum_whole'])}")


if __name__ == "__main__":
    main()
