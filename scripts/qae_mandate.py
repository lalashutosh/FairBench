"""QA-6 (revised, QA-F3): classical rejection vs swap-MCMC vs quantum amplitude estimation (QAE) on the
REAL-TOOL mandate instance.

Instance = scripts/demo_mandate.py defaults: named_universe(n=150, seed=0), compiled rules of
examples/mandate_example.rules.json with k=20, carbon-premium returns (seed 1), planted fund at
rank 0.70 of 1000 rejection samples (seed 2). Amplitudes are Monte Carlo estimates from >= 2M
uniform random k-subsets of the allowed assets:
  a_F = P(feasible), a_G = P(feasible AND S(x) < S_fund), p = a_G / a_F  (percentile of the fund),
  S(x) = equal-weight buy-and-hold total return.

TWO INSTANCE VARIANTS
  full       : the compiled mandate as-is. It contains TrackingErrorCap (a QUADRATIC rule) and
               MinGroups, which the phase oracle (fairbench/ft/oracle.py) CANNOT encode, so no oracle
               resource count exists for it (the classical comparison and the abstract query counts still do).
  encodable  : the same mandate with the non-encodable rules (TrackingErrorCap, VolatilityCap,
               MinGroups) DROPPED. This sub-mandate is a different (larger) feasible set with its own
               P_F and p; its oracle counts (Toffoli, qubits) come from oracle_formula_counts on the
               F set and on G = F + LinearThreshold(g, k(1+s_fund), '<').

QUANTUM (no true-amplitude parameter setting): each repetition first runs a CHARGED coarse IQAE pilot
on the F and G oracles (angle half-width 0.05 rad), uses the pilot estimates only to split the percentile
error between the F and G runs, and then calls fairbench.qae.estimators.iqae_amplitude_tol (which runs
its own charged pilot and sets the angle tolerance from the pilot CI). All pilot queries are in the totals.
MATCHED ACHIEVED ERROR: for a grid of nominal targets, >= 60 repetitions give the q95 of |p_hat - p| and
the mean total queries; queries needed for q95|err| <= eps are derived by log-log interpolation and by
the fit queries = kappa/q95err (both reported; plus the nominal-target number, which is what you must
budget to GUARANTEE the 95% CI half-width eps). IQAE's staircase makes achieved error << nominal at
most targets, so single nominal-target ratios flatter the quantum side or the classical side depending on
the grid point; ranges are reported.

CLASSICAL: (i) rejection MC, proposals = 1.96^2 p(1-p)/(eps^2 a_F) (validated by simulation);
(ii) swap-MCMC, uniform target on F: one query = one proposed swap (random held asset out, random
unheld allowed asset in) + feasibility check; chains start from a rejection-found feasible state (cost
1/a_F proposals charged), burn-in BURN charged, estimate = fraction of post-burn-in states with S<S_fund;
queries for q95|err| <= eps measured over independent chains (extrapolated by c/sqrt(L) beyond the longest
run, flagged).

SYNTHETIC DATA (planted ground truth); no live API.
Outputs: results/qae_mandate.csv, results/qae_mandate.json, results/qae_mandate_curves.csv.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"

from fairbench.baselines import random_k_subsets                                       # noqa: E402
from fairbench.apps.attribution import rejection_sampler, weights_matrix, performance, portfolio_returns  # noqa: E402
from fairbench.data import synthetic_returns                                            # noqa: E402
from fairbench.instances import named_universe                                          # noqa: E402
from fairbench.rules import compile_spec, load_spec                                     # noqa: E402
from fairbench.qae.attribution_qae import linear_return_scores, portfolio_score         # noqa: E402
from fairbench.qae.estimators import AEResult, IdealOracle, iqae, iqae_amplitude_tol, ratio_estimate  # noqa: E402
from fairbench.constraints import ConstraintSet, LinearThreshold, TrackingErrorCap, MinGroups  # noqa: E402
from fairbench.constraints import VolatilityCap                                         # noqa: E402
from fairbench.ft.oracle import quantise, DEFAULT_BITS                                  # noqa: E402
from fairbench.ft.resources import oracle_formula_counts                                # noqa: E402

N, K, SEED, PERIODS, PREMIUM, RANK = 150, 20, 0, 756, 0.08, 0.70
N_TOTAL, BATCH = 2_000_000, 50_000
Z = 1.96
EPS_REPORT = [0.02, 0.01, 0.005]
TARGETS = [0.1, 0.07, 0.05, 0.035, 0.025, 0.018, 0.013, 0.009, 0.0065, 0.0045]   # nominal IQAE targets
REPS = 200
N_CHAINS, L_MAX, BURN, N_POOL = 150, int(sys.argv[1]) if len(sys.argv) > 1 else 200_000, 1000, 400
NONENC = (TrackingErrorCap, VolatilityCap, MinGroups)


def wilson(h, n, z=Z):
    p = h / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, (max(c - half, 0.0), min(c + half, 1.0))


def alloc(aF, aG, eps_p):
    """Split absolute percentile error eps_p between F and G runs (amplitude units). Inputs may be
    PILOT estimates (never the truth)."""
    p = aG / aF
    best = None
    for phi in np.linspace(0.01, math.pi / 2 - 0.01, 1001):
        eF = aF * (eps_p / p) * math.cos(phi)
        eG = aG * (eps_p / p) * math.sin(phi)
        c = math.sqrt(aF * (1 - aF)) / eF + math.sqrt(aG * (1 - aG)) / eG
        if best is None or c < best[0]:
            best = (c, eF, eG)
    return best[1], best[2]


def quantum_rep(aF, aG, target, rng):
    """One percentile estimate: charged pilots -> allocation -> two iqae_amplitude_tol runs (each with
    its own charged pilot). The oracles hold the true amplitudes only as the simulated physical device."""
    oF, oG = IdealOracle(aF), IdealOracle(aG)
    pF = iqae(oF, 0.05, 0.05, rng)
    pG = iqae(oG, 0.05, 0.05, rng)
    aFh = min(max(pF.estimate, 1e-4), 0.9999)
    aGh = min(max(pG.estimate, 1e-4), aFh * 0.9999)
    eF, eG = alloc(aFh, aGh, target)
    rF = iqae_amplitude_tol(oF, eF, 0.05, rng)
    rG = iqae_amplitude_tol(oG, eG, 0.05, rng)
    r = ratio_estimate(rG, rF)
    return r.estimate, pF.oracle_queries + pG.oracle_queries + rF.oracle_queries + rG.oracle_queries


def loglog_solve(x, y, ytarget):
    """x ascending queries, y q95 error (decreasing). Interpolate x at y=ytarget in log-log; None if outside."""
    ly, lx = np.log(y), np.log(x)
    for a in range(len(x) - 1):
        if (ly[a] - math.log(ytarget)) * (ly[a + 1] - math.log(ytarget)) <= 0 and ly[a] != ly[a + 1]:
            t = (math.log(ytarget) - ly[a]) / (ly[a + 1] - ly[a])
            return float(math.exp(lx[a] + t * (lx[a + 1] - lx[a])))
    return None


def mcmc_swap(cs, u, starts, allowed, scores, s_fund, k, L, rng, checkpoints):
    """C parallel independent swap chains, uniform target on F. Returns {L_checkpoint: p_hat (C,)}.
    One query = one proposed swap + feasibility check (per chain)."""
    C, n = len(starts), u.n
    X = np.zeros((C, n), np.uint8)
    sel = np.array(starts)
    np.put_along_axis(X, sel, 1, axis=1)
    ssum = scores[sel].sum(1)
    thr = k * s_fund
    cum = np.zeros(C)
    nsteps = 0
    out = {}
    rows = np.arange(C)
    cps = set(checkpoints)
    for step in range(BURN + L):
        pos = rng.integers(0, k, C)
        i_out = sel[rows, pos]
        j = allowed[rng.integers(0, allowed.size, C)]
        bad = X[rows, j] == 1
        while bad.any():
            j[bad] = allowed[rng.integers(0, allowed.size, int(bad.sum()))]
            bad = X[rows, j] == 1
        Xn = X.copy()
        Xn[rows, i_out] = 0
        Xn[rows, j] = 1
        ok = cs.check_batch(Xn, u).astype(bool)
        a = np.flatnonzero(ok)
        X[a, i_out[a]] = 0
        X[a, j[a]] = 1
        sel[a, pos[a]] = j[a]
        ssum[a] += scores[j[a]] - scores[i_out[a]]
        if step >= BURN:
            cum += ssum < thr
            nsteps += 1
            if nsteps in cps:
                out[nsteps] = cum / nsteps
    return out


def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    u = named_universe(N, seed=SEED)
    spec = load_spec(ROOT / "examples" / "mandate_example.rules.json")
    compiled = compile_spec(spec, u, k=K)
    cs_full = compiled.constraint_set
    z = (u.carbon - u.carbon.mean()) / u.carbon.std()
    u.returns = synthetic_returns(u, periods=PERIODS, mu_shift=PREMIUM * z, seed=SEED + 1)
    cs_enc = ConstraintSet([c for c in cs_full.constraints if not isinstance(c, NONENC)])
    dropped = sorted({type(c).__name__ for c in cs_full.constraints if isinstance(c, NONENC)})
    allowed = cs_full.allowed_indices(u.n)
    k = compiled.k
    pool = rejection_sampler(u, cs_full, 1000, seed=SEED + 2).samples
    perf = performance(portfolio_returns(weights_matrix(pool, "equal", u), u.returns.to_numpy()))
    fund = pool[np.argsort(perf)[int(RANK * (len(pool) - 1))]]
    assert all(ok for _, ok in compiled.check_fund(fund, u))
    scores = linear_return_scores(u)
    s_fund = portfolio_score(fund, u, scores=scores)
    g = scores + 1.0
    print(f"INSTANCE n={u.n} k={k} allowed={allowed.size} rules={len(compiled.rules)} dropped_for_encodable={dropped} "
          f"S_fund={s_fund:.5f}  [{time.time() - t0:.1f}s]", flush=True)

    # ---- Monte Carlo, both variants on the same draws
    V = {"full": cs_full, "encodable": cs_enc}
    hF = {v: 0 for v in V}; hG = {v: 0 for v in V}; pools = {v: [] for v in V}; total = 0
    seeds = np.random.default_rng(12345)
    for _ in range(N_TOTAL // BATCH):
        X = np.zeros((BATCH, u.n), dtype=np.uint8)
        X[:, allowed] = random_k_subsets(allowed.size, k, BATCH, int(seeds.integers(2**31)))
        S = X.astype(float) @ scores / k
        for v, c in V.items():
            feas = c.check_batch(X, u).astype(bool)
            hF[v] += int(feas.sum()); hG[v] += int((feas & (S < s_fund)).sum())
            if len(pools[v]) < N_POOL:
                idx = np.flatnonzero(feas)[: N_POOL - len(pools[v])]
                pools[v].extend(np.flatnonzero(X[i]) for i in idx)
        total += BATCH
    print(f"MC done {total} draws [{time.time() - t0:.1f}s]", flush=True)

    # ---- oracle counts for the encodable variant (full is not encodable)
    oracle = {}
    qF = quantise(u, cs_enc, DEFAULT_BITS)
    csG_enc = ConstraintSet(list(cs_enc.constraints) + [LinearThreshold(g, k * (1 + s_fund), "<", "return<s_fund")])
    qG = quantise(u, csG_enc, DEFAULT_BITS)
    oiF = oracle_formula_counts(u, cs_enc, DEFAULT_BITS, quant=qF)
    oiG = oracle_formula_counts(u, csG_enc, DEFAULT_BITS, quant=qG)
    oracle = dict(
        F=dict(toffoli=oiF["toffoli"], toffoli_depth=oiF["toffoli_depth"], qubits=oiF["n_qubits"]),
        G=dict(toffoli=oiG["toffoli"], toffoli_depth=oiG["toffoli_depth"], qubits=oiG["n_qubits"]),
        quant_F=dict(false_pos=qF.false_pos, false_neg=qF.false_neg, n_eval=qF.n_eval,
                     n_feasible_eval=qF.n_feasible_eval, mode=qF.mode),
        quant_G=dict(false_pos=qG.false_pos, false_neg=qG.false_neg, n_eval=qG.n_eval,
                     n_feasible_eval=qG.n_feasible_eval, mode=qG.mode))
    print("ORACLE (encodable variant):", oracle, f"[{time.time() - t0:.1f}s]", flush=True)

    out_vars, rows, curves = {}, [], []
    for v, c in V.items():
        aF, ciF = wilson(hF[v], total); aG, ciG = wilson(hG[v], total); p = aG / aF
        sdF = (ciF[1] - ciF[0]) / (2 * Z); sdG = (ciG[1] - ciG[0]) / (2 * Z)
        ciP = ratio_estimate(AEResult(aG, ciG, 0, 0, total, "mc", [], dict(std=sdG)),
                             AEResult(aF, ciF, 0, 0, total, "mc", [], dict(std=sdF))).ci
        print(f"[{v}] a_F={aF:.6f} a_G={aG:.6f} p={p:.5f} CI [{ciP[0]:.5f},{ciP[1]:.5f}] hitsF={hF[v]}", flush=True)
        D = dict(a_F=aF, a_F_ci=list(ciF), a_G=aG, a_G_ci=list(ciG), p=p, p_ci=list(ciP), hits_F=hF[v], hits_G=hG[v])

        # ---- rejection (formula + simulation check)
        rng = np.random.default_rng(3)
        rej = {}
        for eps in EPS_REPORT:
            M = int(math.ceil(Z * Z * p * (1 - p) / (eps * eps * aF)))
            cat = np.array([aG, aF - aG, 1 - aF])
            cnt = rng.multinomial(M, cat, size=4000)
            ph = cnt[:, 0] / np.maximum(cnt[:, 0] + cnt[:, 1], 1)
            err = np.abs(ph - p)
            rej[eps] = dict(M=M, q95_abs_err=float(np.quantile(err, 0.95)), frac_within=float((err <= eps).mean()))
        D["rejection"] = rej

        # ---- quantum: matched-error sweep
        qrng = np.random.default_rng(7)
        sweep = []
        for tg in TARGETS:
            errs, qs = [], []
            for _ in range(REPS):
                est, q = quantum_rep(aF, aG, tg, qrng)
                errs.append(est - p); qs.append(q)
            ae = np.abs(errs)
            sweep.append(dict(target=tg, q95_abs_err=float(np.quantile(ae, 0.95)), q_mean=float(np.mean(qs)),
                              q_median=float(np.median(qs)), frac_within_target=float((ae <= tg).mean()), reps=REPS))
            curves.append(dict(variant=v, method="quantum_iqae", **sweep[-1]))
            print(f"  [{v}] Q target={tg:<7} q95err={sweep[-1]['q95_abs_err']:.4f} q_mean={sweep[-1]['q_mean']:.4g} "
                  f"within={sweep[-1]['frac_within_target']:.2f} [{time.time() - t0:.0f}s]", flush=True)
        order = np.argsort([s["q_mean"] for s in sweep])
        qx = np.array([sweep[i]["q_mean"] for i in order]); qy = np.array([sweep[i]["q95_abs_err"] for i in order])
        # envelope: best achieved error at <= queries (make monotone for interpolation)
        qy_mono = np.minimum.accumulate(qy)
        kappa = float(np.median(qx * qy))
        qtab = {}
        for eps in EPS_REPORT:
            nominal = float(np.interp(eps, [s["target"] for s in sorted(sweep, key=lambda s: s["target"])],
                                      [s["q_mean"] for s in sorted(sweep, key=lambda s: s["target"])]))
            nominal_extrap = eps < min(s["target"] for s in sweep)
            interp = loglog_solve(qx, qy_mono, eps)
            fit = kappa / eps
            qtab[eps] = dict(q_nominal_target=nominal, q_matched_interp=interp, q_matched_fit_kappa=fit)
        D["quantum"] = dict(sweep=sweep, kappa_q95=kappa, table=qtab)

        # ---- MCMC
        mrng = np.random.default_rng(99)
        starts = pools[v][:N_CHAINS]
        n_ch = len(starts)
        cps = sorted({int(x) for x in np.unique(np.round(np.logspace(2, math.log10(L_MAX), 14)))})
        tm = time.time()
        res = mcmc_swap(c, u, starts, allowed, scores, s_fund, k, L_MAX, mrng, cps)
        print(f"  [{v}] MCMC {n_ch} chains x {L_MAX} steps took {time.time() - tm:.0f}s", flush=True)
        mc_rows = []
        for L in cps:
            err = np.abs(res[L] - p)
            mc_rows.append(dict(L=L, q95_abs_err=float(np.quantile(err, 0.95)), rms=float(np.sqrt(np.mean(err**2)))))
            curves.append(dict(variant=v, method="mcmc_swap", target=None, q95_abs_err=mc_rows[-1]["q95_abs_err"],
                               q_mean=L + BURN + 1 / aF, q_median=None, frac_within_target=None, reps=n_ch))
        Ls = np.array([r["L"] for r in mc_rows], float); q95 = np.array([r["q95_abs_err"] for r in mc_rows])
        rms = np.array([r["rms"] for r in mc_rows])
        sel_ = Ls >= 2000
        c_fit = float(np.median(q95[sel_] * np.sqrt(Ls[sel_])))
        tau = float(np.median((rms[sel_] ** 2) * Ls[sel_] / (2 * p * (1 - p))))   # var = 2 tau p(1-p)/L
        mtab = {}
        for eps in EPS_REPORT:
            meas = loglog_solve(Ls, np.minimum.accumulate(q95), eps)
            L_need = meas if meas is not None else (c_fit / eps) ** 2
            mtab[eps] = dict(L_steps=L_need, measured=meas is not None,
                             queries_total=L_need + BURN + 1 / aF, extrapolated=meas is None)
        D["mcmc"] = dict(chains=n_ch, L_max=L_MAX, burn_charged=BURN, start_cost_each=1 / aF,
                         c_fit_q95_sqrtL=c_fit, tau_est=tau, curve=mc_rows, table=mtab)
        print(f"  [{v}] MCMC tau~{tau:.1f} c={c_fit:.3f}", {e: (round(m["queries_total"]), m["measured"]) for e, m in mtab.items()}, flush=True)

        for eps in EPS_REPORT:
            qt = qtab[eps]; qs_ = [x for x in (qt["q_nominal_target"], qt["q_matched_interp"], qt["q_matched_fit_kappa"]) if x]
            r_ = rej[eps]["M"]
            rows.append(dict(variant=v, eps=eps, rejection_proposals=r_, mcmc_queries=mtab[eps]["queries_total"],
                             mcmc_extrapolated=mtab[eps]["extrapolated"],
                             quantum_nominal=qt["q_nominal_target"], quantum_matched_interp=qt["q_matched_interp"],
                             quantum_matched_fit=qt["q_matched_fit_kappa"],
                             ratio_rej_over_q_min=r_ / max(qs_), ratio_rej_over_q_max=r_ / min(qs_),
                             ratio_mcmc_over_q_min=mtab[eps]["queries_total"] / max(qs_),
                             ratio_mcmc_over_q_max=mtab[eps]["queries_total"] / min(qs_),
                             rejection_frac_within_eps=rej[eps]["frac_within"], a_F=aF, p=p))
            print(f"  [{v}] eps={eps}: rej={r_:.4g} mcmc={mtab[eps]['queries_total']:.4g} "
                  f"Q(nominal/interp/fit)={qt['q_nominal_target']:.4g}/{qt['q_matched_interp']}/{qt['q_matched_fit_kappa']:.4g}", flush=True)
        out_vars[v] = D

    with open(RES / "qae_mandate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(RES / "qae_mandate_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    info = dict(
        synthetic=True, description="QA-6/QA-F3 real-tool mandate instance (demo_mandate defaults), MC amplitudes; two variants",
        n=N, k=K, allowed_assets=int(allowed.size), n_rules=len(compiled.rules), n_unmapped=len(compiled.unmapped),
        s_fund_buy_hold=s_fund, planted_rank=RANK, planted_premium=PREMIUM, mc_draws=total,
        non_encodable_rules_dropped_for_encodable_variant=dropped,
        oracle_encodable_variant=oracle, variants=out_vars, table=rows, reps_quantum=REPS,
        notes=[
            "The FULL compiled mandate contains TrackingErrorCap (quadratic) and MinGroups, which the phase oracle in fairbench/ft/oracle.py cannot encode: there is NO oracle resource count for it. Quantum query counts for 'full' are abstract (IdealOracle at the MC amplitudes) and describe a hypothetical oracle.",
            "The 'encodable' variant drops TrackingErrorCap/VolatilityCap/MinGroups; it is a DIFFERENT feasible set (own P_F and p); its oracle counts are from oracle_formula_counts on F and on G = F + LinearThreshold(return<s_fund) with default quantisation.",
            "Quantum parameters are set from CHARGED pilot runs (iqae pilot + iqae_amplitude_tol pilot); IdealOracle holds the MC amplitude only as the simulated device. Pilot queries are included in all totals.",
            "Matched achieved error: q95 of |p_hat-p| over >=200 repetitions vs mean total queries; queries for q95<=eps by log-log interpolation (q_matched_interp) and by kappa/eps with kappa=median(q*q95) (q_matched_fit). q_nominal_target is the cost of the nominal 95%-CI target (guaranteed). Ratios are reported as a [min,max] range over these.",
            "p reference is itself an MC estimate (2M draws); its error (see p_ci) limits meaningful comparison at eps near that width.",
            "Rejection = 1.96^2 p(1-p)/(eps^2 a_F) proposals (simulation of binomial/multinomial draws confirms ~95% coverage). One quantum 'query' is one Grover iterate (oracle + diffusion incl. Dicke preparation), one classical 'query' is one proposal; FT wall-clock per query differs enormously (see qae_breakeven).",
            "MCMC: uniform target on F, one query = one proposed swap + feasibility check; chains start from rejection-found feasible states (these are exact stationary draws, so burn-in is not needed for correctness; BURN=1000 is charged anyway as practitioners cannot know this); error is measured over independent chains. 'extrapolated' entries use q95 = c/sqrt(L) fitted at L>=2000. MCMC accuracy depends on mixing (tau estimate reported) and needs a connected feasible set under swaps.",
            "Not considered: classical exact-counting DP, stratified/importance sampling, which could change the classical side.",
        ],
        runtime_s=time.time() - t0)
    (RES / "qae_mandate.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    print(f"saved results/qae_mandate*  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
