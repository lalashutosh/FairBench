"""QA-7: classical exact-counting DP baseline for the attribution percentile (threat to QAE).

Usage:  qae_dp_baseline.py --n 50|100|150|200   (one scaled_family size per process; writes results/qae_dp_part_n<N>.json)
        qae_dp_baseline.py --mandate            (encodable mandate variant: state-size / feasibility)
        qae_dp_baseline.py --rscale             (state & time vs number of weighted rules)
        qae_dp_baseline.py --finalize           (merge parts -> results/qae_dp.csv, results/qae_dp.json)

DP = fairbench.qae.exact_count (sector DP, dense saturating state, B quantisation levels per weighted rule).
Reference p: exact enumeration (n=50, C(50,5)=2.1M) else the 2M-sample Monte Carlo of qae_breakeven.build
(Wilson CI on the feasible count).  Guaranteed bracket [p_lo, p_hi] = subset/superset thresholds; point = plain
rounding.  SYNTHETIC data.
"""
from __future__ import annotations
import argparse, csv, glob, itertools, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
RES = ROOT / "results"
from fairbench.qae.exact_count import count_dp, percentile_dp, StateTooLarge, MAX_STATES   # noqa: E402
from fairbench.constraints import ConstraintSet, LinearThreshold, MinESG, CarbonCap          # noqa: E402

BS = [4, 8, 12, 16, 24, 32, 48, 64, 128, 256, 1024, 2048]          # B=64/256/1024/2048 are the requested grid
Z = 1.959964
TIME_BUDGET = 640.0


def wilson(h, n, z=Z):
    p = h / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, (max(c - half, 0.0), min(c + half, 1.0))


def exact_p(d):
    """Float-rule exact percentile by enumeration (only for tiny C(n,k))."""
    u, cs, g, k, s = d["u"], d["cs"], d["g"], d["k"], d["s_fund"]
    allowed = cs.allowed_indices(u.n)
    nF = nG = 0
    comb = itertools.combinations(allowed.tolist(), k)
    while True:
        chunk = list(itertools.islice(comb, 400_000))
        if not chunk:
            break
        C = np.array(chunk); X = np.zeros((len(C), u.n), np.int64); np.put_along_axis(X, C, 1, axis=1)
        ok = cs.check_batch(X, u)
        nF += int(ok.sum()); nG += int((ok & (g[C].sum(1) / k - 1.0 < s)).sum())
    return nG / nF, nF


def classical_times(n):
    out = {}
    for r in csv.DictReader(open(RES / "qae_breakeven.csv")):
        if int(r["n"]) == n and r["quantum_constants"] == "default" and r["assumption"] == "optimistic":
            out[r["baseline"]] = {k: float(r[k]) for k in ("t_classical_eps0.01_s", "t_classical_eps0.001_s",
                                                         "classical_proposals_eps0.01", "classical_proposals_eps0.001",
                                                         "queries_eps0.01", "queries_eps0.001", "t_proposal_s")}
    return out


def run_n(n):
    from qae_breakeven import build
    t00 = time.time()
    d = build(n); u, cs, k = d["u"], d["cs"], d["k"]
    lin = d["csG"].constraints[-1]
    nF = d["nF"]; hits = round(d["p"] * nF)
    p_mc, ci = wilson(hits, nF)
    ref = dict(kind="MC_2M", p=p_mc, ci=list(ci), n_feasible_samples=nF)
    if math.comb(n, k) <= 3_000_000:
        pe, nFe = exact_p(d)
        ref = dict(kind="exact_enumeration", p=pe, ci=[pe, pe], n_feasible=nFe, p_mc=p_mc, ci_mc=list(ci))
    print(f"n={n} k={k} aF={d['aF']:.4f} p_mc={p_mc:.5f} ci=({ci[0]:.5f},{ci[1]:.5f}) ref={ref['kind']} p={ref['p']:.5f}", flush=True)
    rows = []
    for B in BS:
        row = dict(n=n, k=k, B=B, ref_kind=ref["kind"], p_ref=ref["p"], ref_ci_lo=ref["ci"][0], ref_ci_hi=ref["ci"][1],
                   a_F=d["aF"])
        if time.time() - t00 > TIME_BUDGET:
            row.update(status="skipped_time_budget"); rows.append(row); print(row, flush=True); continue
        try:
            t0 = time.time()
            (lo, hi, pt), rt, peak = percentile_dp(u, cs, lin, B)
            row.update(status="ok", p_lo=lo, p_hi=hi, p_point=pt, width=hi - lo, runtime_s=rt, peak_states=peak,
                       table_GB=2 * 8 * peak / 1e9, point_err=pt - ref["p"],
                       point_in_ref_ci=bool(ref["ci"][0] <= pt <= ref["ci"][1]),
                       bracket_contains_ref=bool(lo - 1e-12 <= ref["p"] <= hi + 1e-12 or
                                                 (lo - 1e-12 <= ref["ci"][1] and hi + 1e-12 >= ref["ci"][0])))
        except StateTooLarge as e:
            row.update(status="infeasible", reason=str(e), peak_states=e.states, est_ops=e.ops,
                       table_GB=2 * 8 * (e.states or 0) / 1e9)
        rows.append(row)
        print({a: (round(b, 5) if isinstance(b, float) else b) for a, b in row.items() if a not in ("ref_kind",)}, flush=True)
    part = dict(n=n, k=k, ref=ref, classical=classical_times(n), rows=rows, total_s=time.time() - t00)
    (RES / f"qae_dp_part_n{n}.json").write_text(json.dumps(part, indent=2, default=str) + "\n")


def run_rscale():
    from qae_breakeven import build
    out = []
    for n in (50, 100):
        d = build(n); u, cs, k = d["u"], d["cs"], d["k"]; lin = d["csG"].constraints[-1]
        base = [c for c in cs.constraints if not isinstance(c, (MinESG, CarbonCap))]
        esg = [c for c in cs.constraints if isinstance(c, MinESG)]
        car = [c for c in cs.constraints if isinstance(c, CarbonCap)]
        for Rw, cons in ((1, base), (2, base + esg), (3, base + esg + car)):   # weighted dims INCLUDING the fund rule
            for B in (8, 16):
                row = dict(n=n, k=k, weighted_dims=Rw, B=B)
                try:
                    r = count_dp(u, ConstraintSet(cons), lin, B, "mid")
                    row.update(status="ok", peak_states=r.peak_states, runtime_s=r.runtime, shape=list(r.state_shape))
                except StateTooLarge as e:
                    row.update(status="infeasible", peak_states=e.states, reason=str(e))
                out.append(row); print(row, flush=True)
    (RES / "qae_dp_part_rscale.json").write_text(json.dumps(out, indent=2) + "\n")


def run_mandate():
    from fairbench.instances import named_universe
    from fairbench.rules import compile_spec, load_spec
    from fairbench.data import synthetic_returns
    from fairbench.qae.attribution_qae import linear_return_scores
    from fairbench.constraints import TrackingErrorCap, VolatilityCap, MinGroups, CountBound
    u = named_universe(150, seed=0)
    comp = compile_spec(load_spec(ROOT / "examples" / "mandate_example.rules.json"), u, k=20)
    z = (u.carbon - u.carbon.mean()) / u.carbon.std()
    u.returns = synthetic_returns(u, periods=756, mu_shift=0.08 * z, seed=1)
    cs = ConstraintSet([c for c in comp.constraint_set.constraints
                        if not isinstance(c, (TrackingErrorCap, VolatilityCap, MinGroups))])
    k = comp.k
    J = json.load(open(RES / "qae_mandate.json"))
    s_fund = J["s_fund_buy_hold"]; ref = J["variants"]["encodable"]
    g = linear_return_scores(u) + 1.0
    lin = LinearThreshold(g, k * (1 + s_fund), "<")
    kinds = {}
    for c in cs.constraints:
        kinds[type(c).__name__] = kinds.get(type(c).__name__, 0) + 1
    out = dict(rule_kinds=kinds, n_countbound_axes=sum(1 for c in cs.constraints if isinstance(c, CountBound)
                                                       and (c.max_count is not None and c.max_count < k or c.min_count > 0)),
               p_ref=ref["p"], p_ref_ci=ref["p_ci"], rows=[])
    for B in (4, 8, 16, 64, 256, 1024, 2048):
        row = dict(B=B)
        try:
            r = count_dp(u, cs, lin, B, "mid", max_states=MAX_STATES)
            row.update(status="ok", peak_states=r.peak_states, runtime_s=r.runtime, shape=list(r.state_shape),
                       p_point=r.count / r.count_wo_last)
        except StateTooLarge as e:
            row.update(status="infeasible", peak_states=float(e.states), shape=str(e).split("=")[0], table_GB=2 * 8 * e.states / 1e9)
        out["rows"].append(row); print(row, flush=True)
    (RES / "qae_dp_part_mandate.json").write_text(json.dumps(out, indent=2, default=str) + "\n")


def finalize():
    parts = [json.load(open(f)) for f in sorted(glob.glob(str(RES / "qae_dp_part_n*.json")))]
    rows = [r for p in parts for r in p["rows"]]
    fields = ["n", "k", "B", "status", "p_lo", "p_hi", "width", "p_point", "p_ref", "ref_kind", "ref_ci_lo", "ref_ci_hi",
              "point_err", "point_in_ref_ci", "bracket_contains_ref", "runtime_s", "peak_states", "table_GB", "reason"]
    with open(RES / "qae_dp.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    summary = []
    for p in parts:
        ok = [r for r in p["rows"] if r["status"] == "ok"]
        s = dict(n=p["n"], k=p["k"], ref=p["ref"], classical=p["classical"], largest_B_feasible=max((r["B"] for r in ok), default=None))
        # fit width ~ c/B on the largest-B points with width<0.9
        fit = [(r["B"], r["width"]) for r in ok if r["width"] < 0.9]
        if fit:
            c = float(np.median([b * w for b, w in fit])); s["bracket_width_c_over_B"] = c
            for eps in (0.01, 0.001):
                Bn = c / (2 * eps)                      # width <= 2 eps  <=> +-eps
                kk = p["k"]; ns_dims = 3
                s[f"B_for_bracket_pm{eps}"] = Bn
                s[f"states_for_bracket_pm{eps}"] = (kk + 1) * (kk * Bn / 2) ** ns_dims
        for eps in (0.01, 0.001):
            good = [r["B"] for r in ok if abs(r["point_err"]) <= eps]
            s[f"min_B_point_err_le_{eps}"] = min(good) if good else None
            gw = [r for r in ok if r["B"] == min(good)] if good else []
            s[f"runtime_at_that_B_{eps}"] = gw[0]["runtime_s"] if gw else None
            bw = [r for r in ok if r["width"] <= 2 * eps]
            s[f"min_B_bracket_pm{eps}_measured"] = min((r["B"] for r in bw), default=None)
        summary.append(s)
    extra = {}
    for nm in ("rscale", "mandate"):
        fp = RES / f"qae_dp_part_{nm}.json"
        if fp.exists():
            extra[nm] = json.load(open(fp))
    (RES / "qae_dp.json").write_text(json.dumps(dict(synthetic=True, summary=summary, parts=parts, **extra), indent=2, default=str) + "\n")
    print("saved results/qae_dp.csv, qae_dp.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int); ap.add_argument("--mandate", action="store_true")
    ap.add_argument("--rscale", action="store_true"); ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    if a.n: run_n(a.n)
    if a.mandate: run_mandate()
    if a.rscale: run_rscale()
    if a.finalize: finalize()
