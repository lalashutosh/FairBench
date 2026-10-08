"""One-command demo: quantum amplitude estimation plugged into the attribution tool.

Small instance (n=16, k=5; --n 24 for the larger one) where exact enumeration is available.
Prints exact / classical rejection / quantum AE side by side and writes results/demo_qae.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_attribution import RES, build_case  # noqa: E402
from fairbench.qae.demo_bridge import attribute_qae  # noqa: E402
import numpy as np  # noqa: E402
from fairbench.constraints import ConstraintSet, Exclusion  # noqa: E402
from fairbench.data import synthetic_returns  # noqa: E402
from fairbench.instances import scaled_family  # noqa: E402
from fairbench.qae.attribution_qae import ExactInstance  # noqa: E402

# (esg_q, carbon_q) candidates, as in scripts/qae_scaling.py part (b); n=24, k=6 gives P_F ~ 0.002-0.007
TIGHT_GRID = [(0.8, 0.2), (0.78, 0.22), (0.8, 0.25), (0.82, 0.22), (0.78, 0.3), (0.85, 0.2),
              (0.83, 0.19), (0.84, 0.22), (0.86, 0.25), (0.88, 0.3), (0.8, 0.17)]


def tight_case(seed=0, n=24, k=6, lo=0.002, hi=0.007, premium=0.08, rank=0.70):
    """qae_scaling.tight_case recipe; first grid point with P_F in [lo, hi]. Fund = feasible
    portfolio at rank ``rank`` of the exact feasible scores."""
    best = None
    for eq, cq in TIGHT_GRID:
        try:
            u, cs = scaled_family(n, k=k, seed=seed, esg_q=eq, carbon_q=cq, n_mc=50_000)
        except ValueError:
            continue
        cs = ConstraintSet(cs.constraints + [Exclusion(np.argsort(u.carbon)[-2:])])
        z = (u.carbon - u.carbon.mean()) / u.carbon.std()
        u.returns = synthetic_returns(u, periods=756, mu_shift=premium * z, seed=seed + 1)
        inst = ExactInstance.build(u, cs)
        pf = inst.a_F
        if inst.feasible.sum() >= 60 and lo <= pf <= hi:
            best = (u, cs, inst)
            break
    if best is None:
        raise RuntimeError("no tight instance in P_F range")
    u, cs, inst = best
    sf = np.flatnonzero(inst.feasible)
    order = sf[np.argsort(inst.scores[sf], kind="stable")]
    row = inst.idx[order[int(rank * (len(order) - 1))]]
    fund = np.zeros(u.n, dtype=int)
    fund[row.astype(int)] = 1
    return u, cs, fund, inst.a_F


def run(n=16, k=5, seed=0, years=3.0, premium=0.08, rank=0.70, eps=0.01, method="iqae", tight=False):
    if tight:
        u, cs, fund, pf = tight_case(seed, n=n if n > 16 else 24, k=k if k != 5 else 6,
                                     premium=premium, rank=rank)
        n, k = u.n, cs.cardinality
    else:
        u, cs, fund = build_case(n, k, seed, years, premium, rank, n_excluded=max(2, n // 8))
        pf = ExactInstance.build(u, cs).a_F
    res, det = attribute_qae(u, cs, fund, eps=eps, method=method, seed=seed)
    det["P_F"] = float(pf)
    p = res["exact"]["percentile"] / 100
    # percentile only: classical i.i.d. rejection needs nF = z^2 p(1-p)/eps^2 feasible hits (95% half-width eps)
    det["percentile_only"] = dict(
        quantum_queries=int(det["quantum_queries"]["percentile"]),
        classical_queries=int(np.ceil(1.96 ** 2 * p * (1 - p) / eps ** 2 / pf)),
        classical_basis="analytic: z^2 p(1-p)/(eps^2 P_F) rejection proposals")
    det["regime"] = "tight" if tight else "loose"
    return dict(synthetic=True, noiseless_simulation=True, n=n, k=k, seed=seed, years=years,
                premium=premium, planted_rank=rank, results=res, details=det)


def scaling_ref(pf):
    """whole-attribution required queries at tol 0.01 (multi-rep, 200 reps) from
    results/qae_hybrid.json (hybrid median, the qae_attribute default), closest P_F:
    (P_F, {criterion: (rejection, quantum_hybrid, mcmc)}) for e50 (typical run) and e90."""
    try:
        cases = json.loads((RES / "qae_hybrid.json").read_text())["cases"]
        c = min(cases, key=lambda c: abs(np.log(c["P_F"] / pf)))
        return c["P_F"], {k: (c[k]["whole"]["rejection"], c[k]["whole"]["quantum_hybrid"],
                              c[k]["whole"].get("mcmc")) for k in ("e50", "e90")}
    except Exception:
        return None


def show(out, method):
    r, d = out["results"], out["details"]
    print(f"regime: {d['regime']} (P_F={d['P_F']:.4f})   n={out['n']} k={out['k']} (synthetic), "
          f"fund return {d['fund_return']:+.2%}")
    print(f"{'method':<24}{'percentile':>11}{'constraint':>12}{'manager':>10}{'queries':>10}"
          f"{'|err| pct':>11}{'|err| CE':>10}{'|err| ME':>10}")
    e = r["exact"]
    errs = {}
    for key, name in (("exact", "exact enumeration"), ("classical", "classical rejection (5000)"),
                      ("classical_matched", "classical @ same budget"),
                      ("quantum", f"quantum AE ({method}, sim)")):
        x = r[key]
        qs = "-" if x["queries"] is None else str(x["queries"])
        er = [abs(x[f] - e[f]) for f in ("percentile", "constraint_effect", "manager_effect")]
        errs[key] = er
        ers = ["-"] * 3 if key == "exact" else [f"{er[0]:.2f}", f"{100 * er[1]:.2f}%", f"{100 * er[2]:.2f}%"]
        print(f"{name:<24}{x['percentile']:>11.2f}{x['constraint_effect']:>+12.2%}"
              f"{x['manager_effect']:>+10.2%}{qs:>10}{ers[0]:>11}{ers[1]:>10}{ers[2]:>10}")
    qq, cm, cr = r["quantum"]["queries"], r["classical_matched"]["queries"], r["classical"]["queries"]
    print(f"  budgets: quantum {qq} vs classical same-budget set {cm} ({100 * (qq / cm - 1):+.0f}%); "
          f"plain 5000-sample rejection uses {cr} ({100 * (cr / qq - 1):+.0f}% vs quantum) "
          f"with errors above -- compare rows, single run")
    ref = scaling_ref(d["P_F"])
    if ref:
        pf, t = ref
        for crit, lab in (("e50", "typical run"), ("e90", "9 in 10 runs")):
            rj, qh, mc = t[crit]
            if not (rj and qh):
                continue
            mcs = f", swap-MCMC {mc:,.0f}" if mc else ""
            print(f"  multi-rep, whole attribution at ±1 pt ({lab}, closest P_F={pf:.4f}): rejection {rj:,.0f}"
                  f"{mcs} vs quantum {qh:,.0f} queries ({rj / qh:.2f}x vs rejection)")
    else:
        print("  single run; see results/qae_hybrid.json for the multi-rep required-query comparison")
    po = d["percentile_only"]
    print(f"  percentile alone: quantum {po['quantum_queries']} vs classical {po['classical_queries']} "
          f"queries ({po['classical_queries'] / po['quantum_queries']:.2f}x)")
    print("  median: classical warm start (~10 feasible samples, charged) + amplitude-estimation "
          "refinement; point estimate, 90%-tail errors 3-10x the median\n")


def save(out, path, t0):
    out["runtime_s"] = round(time.time() - t0, 2)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--method", default="iqae")
    ap.add_argument("--tight", action="store_true", help="only the tight-rules preset (P_F ~ 0.002-0.007)")
    ap.add_argument("--loose", action="store_true", help="only the loose-rules case (--n/--k)")
    a = ap.parse_args()
    t0 = time.time()
    if not a.tight:
        out = run(a.n, a.k, a.seed, eps=a.eps, method=a.method)
        show(out, a.method)
        save(out, RES / ("demo_qae.json" if a.n == 16 else f"demo_qae_n{a.n}.json"), t0)
    if not a.loose:
        out = run(24, 6, a.seed, eps=a.eps, method=a.method, tight=True)
        show(out, a.method)
        save(out, RES / "demo_qae_tight.json", t0)
    print("Caveats: noiseless simulation of ideal amplitude estimation, not hardware;")
    print("  synthetic data with planted truth; query counts are oracle calls, not wall-clock")
    print("  (see BUILD_PLAN QA results for where the quantum count does and does not win).")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
