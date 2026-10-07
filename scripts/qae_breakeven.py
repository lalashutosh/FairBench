"""QA-5: fault-tolerant wall-clock break-even of quantum amplitude estimation (QAE) vs classical
rejection Monte Carlo for the attribution PERCENTILE  p = |G|/|F|,
G = F AND buy-and-hold return < s_fund.  Outputs results/qae_breakeven{.csv,.json,.png,_queries.csv,_curves.csv}.

Quantum: p = a_G / a_F (two AE runs on the same Dicke prep, F oracle and G oracle).  Query counts
are MEASURED with fairbench.qae iqae / mlae on IdealOracle and fitted  queries = kappa*sqrt(a(1-a))/eps;
time per query = one Grover iterate (G or F oracle + 2 Dicke + reflection) from the AA T-depth model;
every shot adds one extra Dicke prep.  Classical: rejection MC, proposals = z^2 p(1-p)/(eps^2 a_F),
time/proposal MEASURED (numpy lean draw + check + gather-score).
"""
from __future__ import annotations
import csv, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from aa_estimate import ASSUMPTIONS, QBEST, lean_check, IDEAL_CORES                       # noqa: E402
from fairbench.instances import scaled_family                                              # noqa: E402
from fairbench.data import synthetic_returns                                               # noqa: E402
from fairbench.constraints import LinearThreshold, ConstraintSet                           # noqa: E402
from fairbench.ft.oracle import quantise, DEFAULT_BITS                                     # noqa: E402
from fairbench.ft.resources import oracle_formula_counts, dicke_cost, reflection_cost, resource_estimate  # noqa: E402
from fairbench.qae.estimators import IdealOracle, iqae, mlae, exp_schedule, ratio_estimate  # noqa: E402

RES = ROOT / "results"
NS = [50, 100, 150, 200]
Z = 1.959964
P_TARGET = 0.7
PERIODS = 756
EPS_TOTAL = 1e-3          # rotation-synthesis error budget (as aa_estimate)
# ASSUMED compiled-code lower bound for the classical proposal: 5 ns per selected asset (partial
# Fisher-Yates draw + sector count + ESG/carbon/return gathers, all O(k)) + 20 ns fixed.
COMPILED_NS_PER_K, COMPILED_NS_FIXED = 5.0, 20.0


# ----------------------------------------------------------------------------- instance
def build(n):
    u, cs = scaled_family(n, seed=0)
    r = synthetic_returns(u, PERIODS, seed=0).to_numpy()
    g = np.prod(1.0 + r, axis=0)                              # gross buy-and-hold factor per asset
    k = cs.cardinality
    si = np.array([int(s[1:]) for s in u.sector]); ns = int(si.max()) + 1
    rng = np.random.default_rng(1)
    sc_f = []
    nF = nP = 0
    for _ in range(100):                                       # 2M proposals
        idx = np.argpartition(rng.random((20000, n)), k - 1, axis=1)[:, :k]
        ok = lean_check(idx, u, cs, si, ns)
        nP += len(idx); nF += int(ok.sum())
        sc_f.append(g[idx[ok]].sum(1) / k - 1.0)
    sc = np.concatenate(sc_f)
    s_fund = float(np.quantile(sc, P_TARGET))
    aF = nF / nP
    csG = ConstraintSet(list(cs.constraints) + [LinearThreshold(g, k * (1 + s_fund), "<", "return<s_fund")])
    # sanity: lean+score G agrees with ConstraintSet G on a held-out batch
    idx = np.argpartition(rng.random((5000, n)), k - 1, axis=1)[:, :k]
    X = np.zeros((5000, n), np.int64); np.put_along_axis(X, idx, 1, axis=1)
    okF = lean_check(idx, u, cs, si, ns); okG = okF & (g[idx].sum(1) / k - 1.0 < s_fund)
    assert (okF == cs.check_batch(X, u)).all() and (okG == csG.check_batch(X, u)).all()
    pG = float((sc < s_fund).mean())
    return dict(u=u, cs=cs, csG=csG, g=g, s_fund=s_fund, k=k, si=si, ns=ns, aF=aF, aG=aF * pG, p=pG,
                nF=nF, nP=nP)


# ----------------------------------------------------------------------------- classical
def time_classical(d, batch=20000, reps=6):
    u, cs, g, k, si, ns, s = d["u"], d["cs"], d["g"], d["k"], d["si"], d["ns"], d["s_fund"]
    n = u.n; rng = np.random.default_rng(5)
    def one():
        idx = np.argpartition(rng.random((batch, n)), k - 1, axis=1)[:, :k]
        ok = lean_check(idx, u, cs, si, ns)
        sel = idx[ok]                                         # score only the feasible rows
        return int(((g[sel].sum(1) / k - 1.0) < s).sum())
    one()
    t0 = time.perf_counter()
    for _ in range(reps): one()
    t_num = (time.perf_counter() - t0) / (batch * reps)
    t_cmp = (COMPILED_NS_PER_K * k + COMPILED_NS_FIXED) * 1e-9
    return t_num, t_cmp


def verify_proposals(d, eps=0.02, reps=300):
    """Draw M = z^2 p(1-p)/(eps^2 aF) proposals, p_hat = G/F; SD of p_hat over reps should be eps/z."""
    u, cs, g, k, si, ns, s = d["u"], d["cs"], d["g"], d["k"], d["si"], d["ns"], d["s_fund"]
    n = u.n; p, aF = d["p"], d["aF"]
    M = int(Z ** 2 * p * (1 - p) / (eps ** 2 * aF))
    rng = np.random.default_rng(11); est = []
    for _ in range(reps):
        h = f = 0; left = M
        while left > 0:
            b = min(left, 20000); left -= b
            idx = np.argpartition(rng.random((b, n)), k - 1, axis=1)[:, :k]
            ok = lean_check(idx, u, cs, si, ns); sel = idx[ok]
            f += len(sel); h += int(((g[sel].sum(1) / k - 1.0) < s).sum())
        est.append(h / f)
    est = np.array(est)
    return dict(M=M, eps=eps, sd_phat=float(est.std()), predicted_sd=eps / Z, mean_phat=float(est.mean()),
                p_true=p, frac_within_eps=float((np.abs(est - p) <= eps).mean()), reps=reps)


# ----------------------------------------------------------------------------- empirical queries
def run_iqae(a, eps, reps, seed):
    rng = np.random.default_rng(seed); q, sh = [], []
    for _ in range(reps):
        r = iqae(IdealOracle(a), eps, 0.05, rng)
        assert (r.ci[1] - r.ci[0]) / 2 <= eps * (1 + 1e-9)
        q.append(r.oracle_queries); sh.append(r.shots)
    return float(np.median(q)), float(np.median(sh)), float(np.mean(q))


def run_mlae(a, K, N, reps, seed):
    rng = np.random.default_rng(seed); q, hw, sh = [], [], []
    for _ in range(reps):
        r = mlae(IdealOracle(a), exp_schedule(K), N, rng)
        q.append(r.oracle_queries); sh.append(r.shots); hw.append((r.ci[1] - r.ci[0]) / 2)
    return float(np.mean(q)), float(np.mean(sh)), float(np.median(hw))


def empirical(amps, eps_grid, reps_iqae=15):
    rows = []
    for a in amps:
        for e in eps_grid:
            qmed, smed, qmean = run_iqae(a, e, reps_iqae, 7)
            rows.append(dict(method="iqae", a=a, eps=e, queries=qmed, queries_mean=qmean, shots=smed,
                             kappa=2 * qmed * e))   # iqae eps is an ANGLE tolerance: a-halfwidth ~ 2 sqrt(a(1-a)) eps
        for K in range(3, 15):
            q, sh, hw = run_mlae(a, K, 50, 6, 9)
            rows.append(dict(method="mlae_exp", a=a, eps=hw, queries=q, queries_mean=q, shots=sh,
                             kappa=q * hw / math.sqrt(a * (1 - a))))
    return rows


# ----------------------------------------------------------------------------- quantum cost model
class QModel:
    def __init__(self, n, k, oiF, oiG, kappa, s0, s1):
        self.n, self.k = n, k
        self.oi = {"F": oiF, "G": oiG}
        self.dk = dicke_cost(n, k); self.rf = reflection_cost(n)
        self.kappa, self.s0, self.s1 = kappa, s0, s1

    def queries(self, a, eps):
        return self.kappa * math.sqrt(a * (1 - a)) / eps

    def shots(self, a, eps_a):
        eps_t = min(eps_a / (2 * math.sqrt(a * (1 - a))), 0.49)    # IQAE angle tolerance
        return max(self.s0 + self.s1 * math.log2(1 / eps_t), 1.0)

    def iterate(self, which, t_ry, tpt, tdpt):
        oi, dk, rf = self.oi[which], self.dk, self.rf
        tof = oi["toffoli"] + 2 * dk.toffoli + rf.toffoli
        rot = 2 * dk.rotations
        return dict(T=tof * tpt + rot * t_ry,
                    depth=oi["toffoli_depth"] * tdpt + 2 * dk.t_depth(t_ry, tdpt) + rf.toffoli_depth * tdpt,
                    prepT=dk.t_count(t_ry, tpt), prepD=dk.t_depth(t_ry, tdpt))

    def alloc(self, aF, aG, eps_p, w_depth=(1.0, 1.0)):
        """Split the percentile error eps_p (95% half-width, quadrature) between the F and G runs
        to minimise depth-weighted queries.  Returns (eps_F, eps_G)."""
        p = aG / aF; best = None
        for phi in np.linspace(0.01, math.pi / 2 - 0.01, 3001):
            eF = aF * (eps_p / p) * math.cos(phi); eG = aG * (eps_p / p) * math.sin(phi)
            c = w_depth[0] * math.sqrt(aF * (1 - aF)) / eF + w_depth[1] * math.sqrt(aG * (1 - aG)) / eG
            if best is None or c < best[0]: best = (c, eF, eG)
        return best[1], best[2]

    def time(self, aF, aG, eps_p, assump, variant):
        kw = QBEST if variant == "qbest" else dict(rs_const=3.0, t_per_toffoli=4)
        tdpt = 2
        itF = self.iterate("F", 0, 4, tdpt); itG = self.iterate("G", 0, 4, tdpt)
        eF, eG = self.alloc(aF, aG, eps_p, (itF["depth"], itG["depth"]))
        qF, qG = self.queries(aF, eF), self.queries(aG, eG)
        sF, sG = self.shots(aF, eF), self.shots(aG, eG)
        rot = self.dk.rotations
        n_rot = (2 * (qF + qG) + sF + sG) * rot
        t_ry = kw["rs_const"] * math.log2(max(n_rot, 1) / EPS_TOTAL)
        tpt = kw["t_per_toffoli"]
        itF = self.iterate("F", t_ry, tpt, tdpt); itG = self.iterate("G", t_ry, tpt, tdpt)
        depth = qF * itF["depth"] + qG * itG["depth"] + (sF + sG) * itF["prepD"]
        T = qF * itF["T"] + qG * itG["T"] + (sF + sG) * itF["prepT"]
        a = ASSUMPTIONS[assump]
        t = depth * a["t_layer_us"] * 1e-6
        if a["factory_T_per_us"]:
            t = max(t, T / a["factory_T_per_us"] * 1e-6)
        return dict(t=t, queries=qF + qG, shots=sF + sG, eps_F=eF, eps_G=eG, t_ry=t_ry, T=T, depth=depth,
                    iterF=itF["depth"], iterG=itG["depth"])


def classical_time(aF, p, eps, t_prop, cores=1):
    return Z ** 2 * p * (1 - p) / (eps ** 2 * aF) * t_prop / cores


def breakeven(fq, fc, lo=1e-9, hi=0.5):
    """largest eps in [lo, hi] at which quantum <= classical (quantum wins for eps below it); log-bisect.
    Returns (value, flag)."""
    r = lambda e: math.log(fq(e) / fc(e))
    if r(hi) <= 0: return hi, "quantum_wins_everywhere(<=0.5)"
    if r(lo) > 0: return lo, "classical_wins_even_at_1e-9"
    a, b = math.log(lo), math.log(hi)
    for _ in range(80):
        m = 0.5 * (a + b)
        if r(math.exp(m)) <= 0: a = m
        else: b = m
    return math.exp(a), "ok"


# ----------------------------------------------------------------------------- main
def main():
    t00 = time.time()
    info, cl, dat, ver = {}, {}, {}, {}
    for n in NS:
        t0 = time.time(); d = build(n); dat[n] = d
        qF = quantise(d["u"], d["cs"], DEFAULT_BITS); qG = quantise(d["u"], d["csG"], DEFAULT_BITS)
        oiF = oracle_formula_counts(d["u"], d["cs"], DEFAULT_BITS, quant=qF)
        oiG = oracle_formula_counts(d["u"], d["csG"], DEFAULT_BITS, quant=qG)
        info[n] = dict(oiF=oiF, oiG=oiG, qF=qF, qG=qG)
        cl[n] = time_classical(d)
        print(f"n={n} k={d['k']} aF={d['aF']:.4f} aG={d['aG']:.4f} p={d['p']:.3f} s_fund={d['s_fund']:.4f} "
              f"TofF={oiF['toffoli']} TofG={oiG['toffoli']} qubitsF/G={oiF['n_qubits']}/{oiG['n_qubits']} "
              f"t_np={cl[n][0]*1e9:.0f}ns t_cmp={cl[n][1]*1e9:.0f}ns FPG={qG.false_pos} FNG={qG.false_neg} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        # consistency of my iterate model against resource_estimate (same t_ry)
    ver = verify_proposals(dat[100]); print("proposal-formula check n=100:", ver, flush=True)

    # ---- empirical query counts
    amps = sorted({dat[n][x] for n in NS for x in ("aF", "aG")} | {0.5, 0.1, 0.01, 0.001})
    eps_grid = list(np.logspace(math.log10(0.05), -5, 14))
    emp = empirical(amps, eps_grid)
    print(f"empirical done [{time.time()-t00:.0f}s]", flush=True)
    with open(RES / "qae_breakeven_queries.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(emp[0])); w.writeheader(); w.writerows(emp)

    def fit(method, emax=0.01):
        R = [r for r in emp if r["method"] == method and r["eps"] <= min(emax, 0.1 * r["a"])]  # eps << a regime
        kap = np.array([r["kappa"] for r in R]); per_a = {}
        for a in amps:
            k_ = [r["kappa"] for r in R if r["a"] == a]
            per_a[a] = (float(np.median(k_)), float(np.std(k_) / np.mean(k_)))
        # shots ~ s0 + s1 log2(1/eps)
        A_ = np.array([[1.0, math.log2(1 / r["eps"])] for r in R]); y = np.array([r["shots"] for r in R])
        s0, s1 = np.linalg.lstsq(A_, y, rcond=None)[0]
        return dict(kappa=float(np.median(kap)), kappa_rel_spread=float(np.std(kap) / np.mean(kap)),
                    kappa_per_a=per_a, s0=float(s0), s1=float(s1))
    fi, fm = fit("iqae"), fit("mlae_exp", emax=0.02)
    print("IQAE fit:", {k: v for k, v in fi.items() if k != "kappa_per_a"})
    print("MLAE fit:", {k: v for k, v in fm.items() if k != "kappa_per_a"})
    # slope check log queries vs log eps (IQAE, n=100 aF)
    a100 = dat[100]["aF"]
    R = [r for r in emp if r["method"] == "iqae" and r["a"] == a100 and r["eps"] <= 0.01]
    slope = float(np.polyfit([math.log(r["eps"]) for r in R], [math.log(r["queries"]) for r in R], 1)[0])
    print(f"IQAE slope log q vs log eps (n=100 aF, eps<=0.01): {slope:.3f}")

    # ---- model + end-to-end validation (ratio_estimate with iqae at allocated eps)
    models = {n: QModel(n, dat[n]["k"], info[n]["oiF"], info[n]["oiG"], fi["kappa"], fi["s0"], fi["s1"]) for n in NS}
    val = []
    m = models[100]; aF, aG = dat[100]["aF"], dat[100]["aG"]; ptrue = aG / aF
    for ep in (0.05, 0.02, 0.01, 0.005):
        eF, eG = m.alloc(aF, aG, ep); rng = np.random.default_rng(3); err, hw, qs = [], [], []
        for _ in range(100):
            th = lambda a_, e_: min(e_ / (2 * math.sqrt(a_ * (1 - a_))), 0.49)
            rF = iqae(IdealOracle(aF), th(aF, eF), 0.05, rng); rG = iqae(IdealOracle(aG), th(aG, eG), 0.05, rng)
            rr = ratio_estimate(rG, rF)
            err.append(rr.estimate - ptrue); hw.append((rr.ci[1] - rr.ci[0]) / 2); qs.append(rr.oracle_queries)
        val.append(dict(eps_p=ep, mean_halfwidth=float(np.mean(hw)), rms_err=float(np.sqrt(np.mean(np.square(err)))),
                        max_abs_err=float(np.max(np.abs(err))),
                        frac_err_within_target=float((np.abs(err) <= ep).mean()),
                        median_abs_err=float(np.median(np.abs(err))), reps=100, mean_queries=float(np.mean(qs)),
                        model_queries=m.queries(aF, eF) + m.queries(aG, eG)))
        print("validate", val[-1], flush=True)

    # ---- time curves + break-even
    eps_curve = np.logspace(math.log10(0.05), -9, 90)
    base_defs = {   # name -> (t_prop selector, cores, label)
        "numpy_lean_1core": (0, 1), "compiled_1core_ASSUMED": (1, 1),
        f"numpy_lean_{IDEAL_CORES}core_ideal_ASSUMED": (0, IDEAL_CORES),
        f"compiled_{IDEAL_CORES}core_ideal_ASSUMED": (1, IDEAL_CORES)}
    rows, curves, estar = [], [], {}
    for n in NS:
        d, m = dat[n], models[n]; aF, aG, p = d["aF"], d["aG"], d["aG"] / d["aF"]
        for variant in ("default", "qbest"):
            for an in ASSUMPTIONS:
                fq = lambda e: m.time(aF, aG, e, an, variant)["t"]
                for bn, (sel, cores) in base_defs.items():
                    tp = cl[n][sel]
                    fc = lambda e: classical_time(aF, p, e, tp, cores)
                    es, flag = breakeven(fq, fc)
                    estar[(n, variant, an, bn)] = (es, flag)
                    row = dict(n=n, k=d["k"], a_F=aF, a_G=aG, p=p, quantum_constants=variant, assumption=an,
                               baseline=bn, eps_star=es, flag=flag, t_proposal_s=tp)
                    for e in (0.01, 0.001):
                        tq, tc = fq(e), fc(e)
                        row[f"t_quantum_eps{e}_s"] = tq; row[f"t_classical_eps{e}_s"] = tc
                        row[f"ratio_q_over_c_eps{e}"] = tq / tc
                    row["queries_eps0.01"] = m.time(aF, aG, 0.01, an, variant)["queries"]
                    row["queries_eps0.001"] = m.time(aF, aG, 0.001, an, variant)["queries"]
                    row["classical_proposals_eps0.01"] = Z**2*p*(1-p)/(1e-4*aF)
                    row["classical_proposals_eps0.001"] = Z**2*p*(1-p)/(1e-6*aF)
                    row["oracle_toffoli_F"] = info[n]["oiF"]["toffoli"]; row["oracle_toffoli_G"] = info[n]["oiG"]["toffoli"]
                    rows.append(row)
                for e in eps_curve:
                    r = m.time(aF, aG, e, an, variant)
                    curves.append(dict(n=n, quantum_constants=variant, assumption=an, eps=e, t_quantum_s=r["t"],
                                       queries=r["queries"], shots=r["shots"], t_ry=r["t_ry"],
                                       **{f"t_classical_{bn}_s": classical_time(aF, p, e, cl[n][s_], c_)
                                          for bn, (s_, c_) in base_defs.items()}))
    for fn, rs in (("qae_breakeven.csv", rows), ("qae_breakeven_curves.csv", curves)):
        with open(RES / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)

    oracles = {n: dict(toffoli_F=info[n]["oiF"]["toffoli"], toffoli_G=info[n]["oiG"]["toffoli"],
                       toffoli_depth_F=info[n]["oiF"]["toffoli_depth"], toffoli_depth_G=info[n]["oiG"]["toffoli_depth"],
                       qubits_F=info[n]["oiF"]["n_qubits"], qubits_G=info[n]["oiG"]["n_qubits"],
                       quant_G=dict(false_pos=info[n]["qG"].false_pos, false_neg=info[n]["qG"].false_neg,
                                    n_eval=info[n]["qG"].n_eval, mode=info[n]["qG"].mode),
                       quant_F=dict(false_pos=info[n]["qF"].false_pos, false_neg=info[n]["qF"].false_neg))
               for n in NS}
    json.dump(dict(
        assumptions=ASSUMPTIONS, quantum_best_constants=QBEST, ideal_cores=IDEAL_CORES,
        compiled_classical_assumption=f"{COMPILED_NS_PER_K} ns per selected asset + {COMPILED_NS_FIXED} ns (ASSUMED lower bound, not measured)",
        instance={n: dict(k=dat[n]["k"], a_F=dat[n]["aF"], a_G=dat[n]["aG"], p=dat[n]["p"], s_fund=dat[n]["s_fund"],
                          mc_proposals=dat[n]["nP"]) for n in NS},
        oracles=oracles, classical_t_proposal_s={n: dict(numpy_lean=cl[n][0], compiled_assumed=cl[n][1]) for n in NS},
        proposal_formula_check=ver, iqae_fit=fi, mlae_exp_fit=fm, iqae_slope_logq_logeps=slope,
        end_to_end_validation_n100=val,
        eps_star={f"n{n}|{v}|{a}|{b}": dict(eps_star=e, flag=f) for (n, v, a, b), (e, f) in estar.items()},
        notes=["percentile p=a_G/a_F via two IQAE runs (F oracle, G oracle); eps allocated between runs to minimise "
               "depth-weighted queries; CI halfwidths combine in quadrature (delta method); validated end-to-end.",
               "query counts are measured on IdealOracle (noiseless) at the instance amplitudes; model queries = kappa*sqrt(a(1-a))/eps_a with IQAE run at angle tolerance eps_a/(2 sqrt(a(1-a))) (IQAE's native eps is an angle tolerance; running it at eps=eps_a over-delivers for small a), shots = s0+s1*log2(1/eps); extrapolated below eps_amp=1e-5.",
               "time = queries x iterate T-depth + shots x one extra Dicke-prep depth; rotation T cost recomputed from total rotation count (eps_total=1e-3), t_depth_per_toffoli=2, serial oracle depth bound.",
               "oracle counts use superset-quantised thresholds (formula path); a_F, a_G in the AE are the TRUE float-rule amplitudes. A real superset oracle would estimate |G_q|/|F_q| (bias <= FP rate ~1%); a subset/superset pair brackets p at the same cost.",
               "final good/bad readout is a classical check of the measured bitstring (phase oracle), as in the AA study.",
               "shots are run serially on one QPU; parallel QPUs/shots not assumed. Algorithmic qubits only; factories, routing, QEC overhead of idle qubits not counted.",
               "classical baselines: numpy lean (measured, 1 core) incl. gather-score on feasible rows; compiled and 64-core rows are ASSUMED.",
               "classical alternatives NOT considered here (exact DP counting, stratified/quasi-MC, importance sampling) would lower the classical cost; eps* is therefore an upper bound on where QAE could win.",
               "quantum wins for eps < eps_star (it scales 1/eps vs 1/eps^2)."],
        runtime_s=time.time() - t00), open(RES / "qae_breakeven.json", "w"), indent=1, default=str)

    # ---- plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    cols = {"optimistic": "#1b9e77", "moderate": "#d95f02", "factory_limited": "#7570b3"}
    for ax, n in zip(axs, (100, 200)):
        C = [r for r in curves if r["n"] == n and r["quantum_constants"] == "default"]
        for an, c in cols.items():
            sub = [r for r in C if r["assumption"] == an]
            ax.loglog([r["eps"] for r in sub], [r["t_quantum_s"] for r in sub], color=c, label=f"QAE {an}")
            es, fl = estar[(n, "default", an, "numpy_lean_1core")]
            if fl == "ok":
                ax.plot(es, [r["t_quantum_s"] for r in sub if abs(r["eps"] - sub[np.argmin([abs(math.log(x["eps"] / es)) for x in sub])]["eps"]) < 1e-30][0], "*", color=c, ms=12)
        sub = [r for r in C if r["assumption"] == "optimistic"]
        qb = [r for r in curves if r["n"] == n and r["quantum_constants"] == "qbest" and r["assumption"] == "optimistic"]
        ax.loglog([r["eps"] for r in qb], [r["t_quantum_s"] for r in qb], ":", color=cols["optimistic"], lw=2,
                  label="QAE optimistic, best-known constants")
        for bn, st, lw in (("numpy_lean_1core", "k-", 2), ("compiled_1core_ASSUMED", "k--", 1.5),
                           (f"numpy_lean_{IDEAL_CORES}core_ideal_ASSUMED", "-", 1.2),
                           (f"compiled_{IDEAL_CORES}core_ideal_ASSUMED", ":", 1.5)):
            kw = dict(color="0.35") if st == "-" else {}
            ax.loglog([r["eps"] for r in sub], [r[f"t_classical_{bn}_s"] for r in sub], st, lw=lw,
                      label="classical " + bn.replace("_", " "), **kw)
        ax.axvline(0.01, color="gray", ls="-.", lw=0.8); ax.text(0.0105, ax.get_ylim()[0] * 3, "eps=0.01", fontsize=7, rotation=90)
        ax.axvspan(1e-5, 0.05, color="gray", alpha=0.07)
        ax.set_xlabel("percentile precision eps (95% CI half-width)"); ax.set_title(f"n={n}, p={dat[n]['aG']/dat[n]['aF']:.2f}, P_F={dat[n]['aF']:.3f}")
        ax.invert_xaxis(); ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("wall-clock time to reach eps (s)"); axs[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("QAE vs rejection MC for the attribution percentile (stars: eps* vs numpy-lean 1 core; shaded: measured query range, left of 1e-5 extrapolated)", fontsize=9)
    fig.tight_layout(); fig.savefig(RES / "qae_breakeven.png", dpi=140)

    # ---- summary
    print("\nORACLES (Toffoli)"); [print(n, oracles[n]) for n in NS]
    print("\nTIMES (default constants)")
    for r in rows:
        if r["quantum_constants"] == "default" and r["baseline"] in ("numpy_lean_1core", "compiled_1core_ASSUMED"):
            print(f"n={r['n']} {r['assumption']:15s} {r['baseline']:24s} eps*={r['eps_star']:.2e} "
                  f"tq(.01)={r['t_quantum_eps0.01_s']:.2e} tc(.01)={r['t_classical_eps0.01_s']:.2e} "
                  f"q/c(.01)={r['ratio_q_over_c_eps0.01']:.1e} tq(.001)={r['t_quantum_eps0.001_s']:.2e} "
                  f"tc(.001)={r['t_classical_eps0.001_s']:.2e} q/c(.001)={r['ratio_q_over_c_eps0.001']:.1e}")
    print("\nEPS* all:")
    for (n, v, a, b), (e, f) in estar.items():
        if n in (100, 200): print(n, v, a, b, f"{e:.2e}", f)
    print(f"total {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
