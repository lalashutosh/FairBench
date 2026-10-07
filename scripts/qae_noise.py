"""NISQ feasibility of quantum amplitude estimation (QAE) for attribution percentiles (QA-4).

Percentile = |G|/|F| over the Dicke state; G = F AND buy-and-hold return < s (s = fund's own
return). Amplitudes a_G = |G|/C(n,k), a_F = |F|/C(n,k) are each estimated by MLAE.

Parts
  1. transpile (basis cx,rz,sx,x; level 2; all-to-all and heavy-hex) one Grover iterate of the
     G and F oracles for n = 6..16, bits 4..11; quantisation mismatch per bits; growth fit.
  2. depolarising-decay model gamma(n, p2) from the transpiled counts; best achievable
     percentile error vs total cost (Fisher/Cramer-Rao optimum over MLAE schedules, plus
     simulated noise-aware / noise-unaware MLAE) vs classical Monte Carlo; crossover budget;
     required p2 for any advantage.
  3. ground-truth density-matrix Aer simulation of a tiny instance (n=4/5, k=2) vs the
     NoisyOracle formula; leakage out of the weight-k subspace.

Cost axis = state preparations (library field ``state_preps`` = sum N(2m+1); a classical
sample = 1). Oracle calls are ~half of that for deep schedules.
Run: ulimit -v 6000000; OMP_NUM_THREADS=2 python -u scripts/qae_noise.py
"""
import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fairbench.apps.attribution import portfolio_returns, weights_matrix
from fairbench.constraints import Cardinality, ConstraintSet, LinearThreshold
from fairbench.data import synthetic_returns
from fairbench.ft.oracle import quantise
from fairbench.ft.resources import grover_circuit
from fairbench.instances import scaled_family
from fairbench.qae.estimators import (NoisyOracle, _fisher, classical_mc, mlae,
                                      ratio_estimate)

RES = Path(__file__).resolve().parent.parent / "results"
BASIS = ["cx", "rz", "sx", "x"]
CASES = [(6, 3), (8, 3), (10, 4), (12, 4), (16, 5)]
BITS = list(range(4, 12))
P2_LIST = [1e-2, 1e-3, 1e-4, 1e-5]
P1_RATIO = 0.1          # p1 = p2 / 10 on sx, x (rz is virtual, error free)


# ============================================================================ instances
def case(n, k):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from demo_attribution import build_case
    u, cs, fund = build_case(n, k, 0, 3.0, 0.08, 0.7, max(1, n // 10))
    g = np.prod(1 + u.returns.to_numpy(), axis=0)
    X = np.array([[1 if i in c else 0 for i in range(n)] for c in itertools.combinations(range(n), k)])
    s = float(fund @ g / k - 1)                      # fund's buy-and-hold return
    csG = ConstraintSet(cs.constraints + [LinearThreshold(g, k * (1 + s), "<", "ret")])
    return u, cs, csG, X, s


def gate_counts(tc):
    ops = tc.count_ops()
    return dict(cx=int(ops.get("cx", 0)), n1q=int(ops.get("sx", 0) + ops.get("x", 0)),
                rz=int(ops.get("rz", 0)))


def tr(qc, cm=None):
    from qiskit import transpile
    return transpile(qc, basis_gates=BASIS, coupling_map=cm, optimization_level=2, seed_transpiler=7)


# ============================================================================ part 1
def part1():
    from qiskit.transpiler import CouplingMap
    cm = CouplingMap.from_heavy_hex(7)           # 115-qubit heavy-hex (ibm Brisbane-like family)
    rows, amps = [], {}
    for n, k in CASES:
        u, cs, csG, X, s = case(n, k)
        F, G = cs.check_batch(X, u), csG.check_batch(X, u)
        aF, aG = F.mean(), G.mean()
        for label, c in (("F", cs), ("G", csG)):
            true = F if label == "F" else G
            for b in BITS:
                q = quantise(u, c, b)
                mask_q = q.check_batch(X, u, c)
                row = dict(n=n, k=k, oracle=label, bits=b, mismatch=int(q.joint_mismatch),
                           mismatch_rate=float(q.joint_mismatch / len(X)), a_true=float(true.mean()),
                           a_quant=float(mask_q.mean()), F_true=int(F.sum()), n_states=len(X))
                cnt = {}
                for r in (1, 2):
                    qc, info = grover_circuit(u, c, r, quant=q)
                    t = tr(qc)
                    cnt[r] = (gate_counts(t), t.depth(), qc.num_qubits)
                    if r == 1:
                        th = tr(qc, cm)
                        row.update(hh_cx_r1=gate_counts(th)["cx"], hh_depth_r1=th.depth())
                    else:
                        th = tr(qc, cm)
                        row.update(hh_cx_r2=gate_counts(th)["cx"], hh_depth_r2=th.depth())
                row.update(qubits=cnt[1][2], cx_r1=cnt[1][0]["cx"], n1q_r1=cnt[1][0]["n1q"], depth_r1=cnt[1][1],
                           cx_iter=cnt[2][0]["cx"] - cnt[1][0]["cx"],
                           n1q_iter=cnt[2][0]["n1q"] - cnt[1][0]["n1q"],
                           depth_iter=cnt[2][1] - cnt[1][1],
                           hh_cx_iter=row["hh_cx_r2"] - row["hh_cx_r1"],
                           hh_depth_iter=row["hh_depth_r2"] - row["hh_depth_r1"])
                rows.append(row)
            print(f"  transpiled n={n} {label}: bits {BITS[0]}..{BITS[-1]} done", flush=True)
        amps[n] = dict(k=k, s=s, aF=float(aF), aG=float(aG), percentile=float(aG / aF),
                       F=int(F.sum()), G=int(G.sum()), n_states=len(X))
    return pd.DataFrame(rows), amps


def choose_bits(df, n):
    """Smallest bits with zero joint mismatch for both oracles (else max bits)."""
    d = df[df.n == n]
    ok = [b for b in BITS if (d[d.bits == b].mismatch == 0).all()]
    return ok[0] if ok else BITS[-1]


def fit_power(ns, ys):
    p = np.polyfit(np.log(ns), np.log(ys), 1)
    return float(p[0]), float(math.exp(p[1]))


# ============================================================================ part 2
def gamma_of(cx, n1q, p2):
    p1 = P1_RATIO * p2
    f_iter = (1 - p2) ** cx * (1 - p1) ** n1q
    return -math.log(f_iter) / 2.0, f_iter           # per state-prep unit (2 per iterate)


def exp_levels(M):
    return [0] + [2 ** j for j in range(int(math.log2(M)) + 1)] if M >= 1 else [0]


def info_per_cost(theta, gamma, pm, levels):
    """Fisher information on theta per unit cost (cost = sum_m (2m+1), N shots at each level)."""
    ms = np.array(levels)
    cost = float(np.sum(2 * ms + 1))
    return _fisher(theta, ms, np.ones(len(ms)), gamma, pm if gamma else 0.0) / cost


def best_quantum_rmse(aG, aF, gG, gF, pmG, pmF, Qs, nmin=10):
    """CRB-optimal RMSE of r = aG/aF at total cost Q (state preps) over exp-schedule depth caps
    M in {0, 2^0..2^20} (same cap for both runs) and cost split w between the G and F runs;
    needs >= nmin shots per level. Returns [(rmse, (MG, MF, w))]."""
    thG, thF = math.asin(math.sqrt(aG)), math.asin(math.sqrt(aF))
    r = aG / aF
    Ms = [0] + [2 ** j for j in range(0, 21)]
    ws = np.linspace(0.05, 0.95, 19)
    pre = []
    for M in Ms:
        lv = exp_levels(M)
        c = float(np.sum(2 * np.array(lv) + 1))
        pre.append((M, c, info_per_cost(thG, gG, pmG, lv), info_per_cost(thF, gF, pmF, lv)))
    out = []
    Qs = np.asarray(Qs, float)
    for Q in Qs:
        best = (np.inf, None)
        for M, c, iG, iF in pre:
            ok = (ws * Q / c >= nmin) & ((1 - ws) * Q / c >= nmin)
            if not ok.any():
                continue
            vG = math.sin(2 * thG) ** 2 / (iG * ws * Q)
            vF = math.sin(2 * thF) ** 2 / (iF * (1 - ws) * Q)
            v = np.where(ok, r * r * (vG / aG ** 2 + vF / aF ** 2), np.inf)
            j = int(np.argmin(v))
            if v[j] < best[0]:
                best = (float(v[j]), (M, M, float(ws[j])))
        out.append((math.sqrt(best[0]) if best[1] else np.inf, best[1]))
    return out


def classical_rmse(aG, aF, Q):
    r = aG / aF
    return math.sqrt(r * (1 - r) / (Q * aF))           # rejection from F: Q proposals, Q*aF in F


def sweep_cost_budgets():
    return np.logspace(2, 12, 41)


def part2(df, amps):
    Qs = sweep_cost_budgets()
    rows, summary = [], {}
    for n, k in CASES:
        b = choose_bits(df, n)
        dd = df[(df.n == n) & (df.bits == b)].set_index("oracle")
        aG, aF = float(dd.loc["G", "a_quant"]), float(dd.loc["F", "a_quant"])
        C = amps[n]["n_states"]
        pmG, pmF = aG * C / 2 ** n, aF * C / 2 ** n     # fully mixed n-qubit register, no post-selection
        for p2 in P2_LIST:
            gG, fG = gamma_of(dd.loc["G", "cx_iter"], dd.loc["G", "n1q_iter"], p2)
            gF, fF = gamma_of(dd.loc["F", "cx_iter"], dd.loc["F", "n1q_iter"], p2)
            res = best_quantum_rmse(aG, aF, gG, gF, pmG, pmF, Qs)
            cross, best_adv, best_adv_Q = None, 0.0, None
            for Q, (rq, cfg) in zip(Qs, res):
                rc = classical_rmse(aG, aF, Q)
                adv = rc / rq if np.isfinite(rq) else 0.0
                if adv > best_adv:
                    best_adv, best_adv_Q = adv, Q
                if cross is None and adv > 1:
                    cross = Q
                rows.append(dict(n=n, bits=b, p2=p2, gamma_G=gG, gamma_F=gF, f_iter_G=fG, f_iter_F=fF,
                                 Q=Q, rmse_quantum_crb=rq, rmse_classical=rc, advantage=adv,
                                 cfg=str(cfg)))
            summary[f"n{n}_p2{p2:g}"] = dict(n=n, bits=b, p2=p2, gamma_G=gG, f_iter_G=fG, gamma_F=gF,
                                             crossover_Q=cross, best_advantage=best_adv,
                                             best_advantage_Q=best_adv_Q)
            print(f"  n={n} p2={p2:g}: f_iter_G={fG:.3g} gamma_G={gG:.3g} crossover={cross} "
                  f"best advantage {best_adv:.2f}x at Q={best_adv_Q}", flush=True)
        # ideal (gamma = 0) reference
        res = best_quantum_rmse(aG, aF, 0.0, 0.0, pmG, pmF, Qs)
        cross = next((Q for Q, (rq, _) in zip(Qs, res) if np.isfinite(rq) and classical_rmse(aG, aF, Q) / rq > 1), None)
        for Q, (rq, cfg) in zip(Qs, res):
            rows.append(dict(n=n, bits=b, p2=0.0, gamma_G=0.0, gamma_F=0.0, f_iter_G=1.0, f_iter_F=1.0,
                             Q=Q, rmse_quantum_crb=rq, rmse_classical=classical_rmse(aG, aF, Q),
                             advantage=classical_rmse(aG, aF, Q) / rq if np.isfinite(rq) else 0.0, cfg=str(cfg)))
        summary[f"n{n}_ideal"] = dict(n=n, crossover_Q=cross)
    return pd.DataFrame(rows), summary


def threshold_p2(df, amps, n, Qmax=1e12, target=1.0):
    """Largest p2 with CRB advantage >= target at some Q <= Qmax (bisection in log p2)."""
    b = choose_bits(df, n)
    dd = df[(df.n == n) & (df.bits == b)].set_index("oracle")
    aG, aF = float(dd.loc["G", "a_quant"]), float(dd.loc["F", "a_quant"])
    C = amps[n]["n_states"]
    pmG, pmF = aG * C / 2 ** n, aF * C / 2 ** n
    Qs = np.logspace(2, math.log10(Qmax), 21)

    def adv(p2):
        gG, _ = gamma_of(dd.loc["G", "cx_iter"], dd.loc["G", "n1q_iter"], p2)
        gF, _ = gamma_of(dd.loc["F", "cx_iter"], dd.loc["F", "n1q_iter"], p2)
        res = best_quantum_rmse(aG, aF, gG, gF, pmG, pmF, Qs)
        return max((classical_rmse(aG, aF, Q) / rq if np.isfinite(rq) else 0.0) for Q, (rq, _) in zip(Qs, res)), gG

    lo, hi = -9.0, -1.0                      # log10 p2
    if adv(10 ** lo)[0] < target:
        return None, None
    for _ in range(9):
        mid = 0.5 * (lo + hi)
        if adv(10 ** mid)[0] >= target:
            lo = mid
        else:
            hi = mid
    p = 10 ** lo
    return p, adv(p)[1]


def simulate_curves(df, amps, n=16, n_seeds=30):
    """Simulated MLAE (noise-aware / unaware), classical MC: RMSE of percentile vs state preps."""
    b = choose_bits(df, n)
    dd = df[(df.n == n) & (df.bits == b)].set_index("oracle")
    aG, aF = float(dd.loc["G", "a_quant"]), float(dd.loc["F", "a_quant"])
    r = aG / aF
    C = amps[n]["n_states"]
    pmG, pmF = aG * C / 2 ** n, aF * C / 2 ** n
    Qs = np.logspace(2.5, 8, 12)
    thG = math.asin(math.sqrt(aG))
    rows = []
    for p2 in P2_LIST:
        if p2 > 0:
            gG, _ = gamma_of(dd.loc["G", "cx_iter"], dd.loc["G", "n1q_iter"], p2)
            gF, _ = gamma_of(dd.loc["F", "cx_iter"], dd.loc["F", "n1q_iter"], p2)
        else:
            gG = gF = 0.0
        orG, orF = NoisyOracle(aG, gG, pmG), NoisyOracle(aF, gF, pmF)
        crb = best_quantum_rmse(aG, aF, gG, gF, pmG, pmF, Qs)
        for Q, (rq, cfg) in zip(Qs, crb):
            if cfg is None:
                continue
            MG, MF, w = cfg
            lg, lf = exp_levels(MG), exp_levels(MF)
            NG = max(10, int(w * Q / np.sum(2 * np.array(lg) + 1)))
            NF = max(10, int((1 - w) * Q / np.sum(2 * np.array(lf) + 1)))
            for aware in (True, False):
                errs = []
                for seed in range(n_seeds):
                    rng = np.random.default_rng(1000 * seed + 17)
                    kw_G = dict(gamma=gG, p_mixed=pmG) if aware and gG else {}
                    kw_F = dict(gamma=gF, p_mixed=pmF) if aware and gF else {}
                    rG = mlae(orG, lg, NG, rng, **kw_G)
                    rF = mlae(orF, lf, NF, rng, **kw_F)
                    errs.append(rG.estimate / max(rF.estimate, 1e-9) - r)
                rows.append(dict(n=n, p2=p2, Q=Q, method="mlae_aware" if aware else "mlae_unaware",
                                 rmse=float(np.sqrt(np.mean(np.square(errs)))), bias=float(np.mean(errs)),
                                 MG=MG, MF=MF, NG=NG, NF=NF, crb=rq,
                                 Q_actual=int(NG * np.sum(2 * np.array(lg) + 1) + NF * np.sum(2 * np.array(lf) + 1))))
        print(f"  simulated curves p2={p2:g} done", flush=True)
    errs_c = []
    for Q in Qs:
        e = []
        for seed in range(n_seeds * 3):
            rng = np.random.default_rng(seed + 5)
            nq = int(Q)
            hF = rng.binomial(nq, aF)                       # proposals landing in F
            hG = rng.binomial(hF, r) if hF > 0 else 0       # of those, in G
            e.append((hG / hF if hF else 0.0) - r)
        rows.append(dict(n=n, p2=-1, Q=Q, method="classical_mc", rmse=float(np.sqrt(np.mean(np.square(e)))),
                         bias=float(np.mean(e)), Q_actual=int(Q)))
    return pd.DataFrame(rows)


# ============================================================================ part 3
def tiny_instance(n, k, qt, bits):
    u, _ = scaled_family(n, k=k, seed=0, n_mc=1000)
    u.returns = synthetic_returns(u, periods=252, mu_shift=0.0, seed=1)
    g = np.prod(1 + u.returns.to_numpy(), axis=0)
    X = np.array([[1 if i in c else 0 for i in range(n)] for c in itertools.combinations(range(n), k)])
    s = np.quantile(X @ g / k - 1, qt)
    csG = ConstraintSet([Cardinality(k), LinearThreshold(g, k * (1 + s), "<", "ret")])
    q = quantise(u, csG, bits)
    return u, csG, q, X


def part3(ms=(0, 1, 2, 3, 4), p2s=(1e-3, 3e-3, 1e-2)):
    from math import comb
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    out = []
    for (n, k, qt, bits) in [(4, 2, 0.35, 2)]:
        u, csG, q, X = tiny_instance(n, k, qt, bits)
        full = np.array([[(x >> i) & 1 for i in range(n)] for x in range(2 ** n)])
        goodmask = q.check_batch(full, u, csG)                # quantised G on the full 2^n space
        goodmask &= (full.sum(1) == k)
        a = goodmask.sum() / comb(n, k)
        pm = goodmask.sum() / 2 ** n
        theta = math.asin(math.sqrt(a))
        circs = {}
        for m in ms:
            qc, info = grover_circuit(u, csG, m, quant=q)
            circs[m] = tr(qc)
        nq = circs[ms[0]].num_qubits
        print(f"  tiny n={n},k={k},bits={bits}: qubits={nq}, a_G={a:.4f}, |G|={goodmask.sum()}, "
              f"pm=|G|/2^n={pm:.4f}", flush=True)
        # per-iterate counts (circuits are transpiled separately; difference m=1 - m=0)
        c0, c1 = gate_counts(circs[0]), gate_counts(circs[1])
        cx_it, n1_it = c1["cx"] - c0["cx"], c1["n1q"] - c0["n1q"]
        for p2 in p2s:
            p1 = P1_RATIO * p2
            nm = NoiseModel(basis_gates=BASIS)
            nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
            nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
            sim = AerSimulator(method="density_matrix", noise_model=nm)
            g_pred, f_it = gamma_of(cx_it, n1_it, p2)
            P, leak = [], []
            for m in ms:
                c = circs[m].copy()
                c.save_probabilities(list(range(n)), label="p")
                pr = sim.run(c, shots=1).result().data()["p"]
                pr = np.asarray(pr)
                P.append(float(pr[goodmask].sum()))
                leak.append(float(pr[full.sum(1) != k].sum()))
            P, leak = np.array(P), np.array(leak)
            kk = 2 * np.array(ms) + 1

            def model(gm, pmix):
                f = np.exp(-gm * kk)
                return f * np.sin(kk * theta) ** 2 + (1 - f) * pmix

            from scipy.optimize import minimize_scalar
            fit = minimize_scalar(lambda gm: np.sum((model(gm, pm) - P) ** 2), bounds=(0, 3), method="bounded")
            g_fit = float(fit.x)
            ideal = np.sin(kk * theta) ** 2
            leak_pred = (1 - np.exp(-g_pred * kk)) * (1 - comb(n, k) / 2 ** n)
            out.append(dict(n=n, k=k, bits=bits, qubits=nq, a_G=float(a), pm=float(pm), p2=p2, p1=p1,
                            cx_iter=cx_it, n1q_iter=n1_it, f_iter_pred=f_it, gamma_pred=g_pred, gamma_fit=g_fit,
                            f_iter_fit=math.exp(-2 * g_fit), P_sim=P.tolist(), P_ideal=ideal.tolist(),
                            P_model_pred=model(g_pred, pm).tolist(), P_model_fit=model(g_fit, pm).tolist(),
                            max_abs_err_pred=float(np.max(np.abs(model(g_pred, pm) - P))),
                            max_abs_err_fit=float(np.max(np.abs(model(g_fit, pm) - P))),
                            max_abs_err_ideal=float(np.max(np.abs(ideal - P))),
                            leak_sim=leak.tolist(), leak_pred=leak_pred.tolist()))
            print(f"    p2={p2:g}: f_iter pred {f_it:.3f} fit {math.exp(-2 * g_fit):.3f}; max|err| pred "
                  f"{out[-1]['max_abs_err_pred']:.4f} fit {out[-1]['max_abs_err_fit']:.4f} ideal "
                  f"{out[-1]['max_abs_err_ideal']:.4f}; leak(m=4) sim {leak[-1]:.3f} pred {leak_pred[-1]:.3f}",
                  flush=True)
    return out


# ============================================================================ plots
def make_plot(sweep, sim, tiny, df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    cols = {1e-2: "#d62728", 1e-3: "#ff7f0e", 1e-4: "#2ca02c", 1e-5: "#1f77b4", 0.0: "k"}
    s16 = sweep[sweep.n == 16]
    for p2, g in s16.groupby("p2"):
        ax[0].plot(g.Q, g.rmse_quantum_crb, color=cols.get(p2, "gray"), lw=1.8,
                   label="ideal QAE (CRB)" if p2 == 0 else f"p2={p2:g}")
    g0 = s16[s16.p2 == 0.0]
    ax[0].plot(g0.Q, g0.rmse_classical, "k--", lw=2, label="classical MC")
    for meth, mk in (("mlae_aware", "o"), ("mlae_unaware", "x")):
        d = sim[(sim.method == meth) & (sim.p2 > 0)]
        for p2, g in d.groupby("p2"):
            ax[0].plot(g.Q_actual, g.rmse, mk, color=cols[p2], ms=5, alpha=.7)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("cost: state preparations"); ax[0].set_ylabel("RMSE of percentile |G|/|F|")
    ax[0].set_title("n=16 demo (a_G, a_F from instance); lines CRB-optimal,\no = sim. noise-aware MLAE, x = unaware", fontsize=9)
    ax[0].legend(fontsize=8); ax[0].set_ylim(1e-4, 1)
    t = df[(df.oracle == "G") & (df.bits == 8)]
    ax[1].loglog(t.n, t.cx_iter, "o-", label="all-to-all CX / iterate")
    ax[1].loglog(t.n, t.hh_cx_iter, "s-", label="heavy-hex CX / iterate")
    ax[1].set_xlabel("n (assets)"); ax[1].set_ylabel("CX per Grover iterate (G oracle, 8 bits)")
    ax[1].set_title("transpiled gate cost, opt. level 2", fontsize=9); ax[1].legend(fontsize=8)
    tt = [r for r in tiny if r["n"] == 4]
    for r in tt:
        ms = np.arange(len(r["P_sim"]))
        ax[2].plot(ms, r["P_sim"], "o", color=cols.get(r["p2"], "gray"))
        ax[2].plot(ms, r["P_model_pred"], "-", color=cols.get(r["p2"], "gray"), label=f"p2={r['p2']:g} (model, gamma from CX count)")
    ax[2].plot(ms, tt[0]["P_ideal"], "k--", label="ideal")
    ax[2].set_xlabel("Grover iterates m"); ax[2].set_ylabel("P(good)")
    ax[2].set_title(f"Aer density-matrix check, n=4 k=2 ({tt[0]['qubits']} qubits); dots = sim", fontsize=9)
    ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--skip-sim", action="store_true")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    t0 = time.time()
    print("PART 1: transpile", flush=True)
    df, amps = part1()
    df.to_csv(RES / "qae_noise_transpile.csv", index=False)
    print(f"  ({time.time() - t0:.0f}s)", flush=True)

    fits = {}
    for key in ("cx_iter", "hh_cx_iter", "depth_iter", "hh_depth_iter", "qubits"):
        for lab, bsel in (("bits8", lambda n: 8), ("bits_min_exact", lambda n: choose_bits(df, n))):
            xs = [n for n, _ in CASES]
            ys = [float(df[(df.n == n) & (df.oracle == "G") & (df.bits == bsel(n))][key].iloc[0]) for n in xs]
            al, c = fit_power(xs, ys)
            fits[f"G_{key}_{lab}"] = dict(alpha=al, c=c, n=xs, y=ys)
    print("PART 2: noise sweep", flush=True)
    sweep, ssum = part2(df, amps)
    sweep.to_csv(RES / "qae_noise.csv", index=False)
    thr = {}
    for n, _ in CASES:
        for tgt in (1.0, 10.0):
            p, gm = threshold_p2(df, amps, n, target=tgt)
            thr[f"n{n}_adv{tgt:g}x"] = dict(p2_star=p, gamma_star=gm)
            print(f"  n={n}: p2* for >= {tgt:g}x advantage (Q<=1e12): {p}  (gamma {gm})", flush=True)
    # extrapolation of CX/iterate and p2* ~ gamma*/(CX_eff) to larger n
    f = fits["G_cx_iter_bits_min_exact"]
    ext = {}
    g_star = {t: float(np.mean([thr[f"n{n}_adv{t:g}x"]["gamma_star"] for n, _ in CASES
                                if thr[f"n{n}_adv{t:g}x"]["gamma_star"]])) for t in (1.0, 10.0)}
    for nn in (16, 32, 50, 100, 200):
        cx = f["c"] * nn ** f["alpha"]
        ext[str(nn)] = dict(cx_iter_extrap=cx, p2_star_any=2 * g_star[1.0] / (cx * (1 + P1_RATIO * 0.3)),
                            p2_star_10x=2 * g_star[10.0] / (cx * (1 + P1_RATIO * 0.3)))
    if not a.skip_sim:
        print("PART 2b: simulated MLAE curves (n=16)", flush=True)
        sim = simulate_curves(df, amps, 16, a.seeds)
        sim.to_csv(RES / "qae_noise_sim.csv", index=False)
    else:
        sim = pd.read_csv(RES / "qae_noise_sim.csv")
    print("PART 3: Aer density-matrix check", flush=True)
    tiny = part3()
    make_plot(sweep, sim, tiny, df, RES / "qae_noise.png")
    summ = dict(amplitudes=amps, bits_chosen={n: choose_bits(df, n) for n, _ in CASES}, power_fits=fits,
                p2=P2_LIST, p1_ratio=P1_RATIO, sweep_summary=ssum, p2_threshold=thr,
                gamma_star_mean=g_star, extrapolation=ext, tiny_noisy_sim=tiny,
                cost_axis="state preps = sum N(2m+1); classical sample = 1",
                runtime_s=time.time() - t0)
    (RES / "qae_noise.json").write_text(json.dumps(summ, indent=1, default=float))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
