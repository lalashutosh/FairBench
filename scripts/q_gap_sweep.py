"""Q-B: exact spectral-gap sweep of QeMCMC vs classical chains on the weight-k subspace.

Design (anti-overfitting):
 * TUNE (t, alpha, topology) and best multi-swap m on island_family(12) ONLY.
 * REPORT frozen params on held-out instances (family 14, family 16 = island_instance, P0)
   plus an "oracle (optimistic)" column tuned per instance (selection bias, upper bound).
 * Penalty scale: the penalty diagonal is divided by its std over the weight-k subspace of the
   instance, so alpha is in units of "one std of penalty energy" vs hopping amplitude 1.
   (Without it alpha means different things on different instances.) h = None (a uniform field is
   constant in a fixed-weight subspace; asset-dependent fields not explored).
 * One eigh per (instance, alpha, topology), reused for all t. Gap = 1 - max(|l2|,|lmin|) of the
   MH kernel on F (rejected mass on the diagonal), via fairbench.chains.spectral_gap.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fairbench.baselines import enumerate_feasible
from fairbench.chains import (feasible_mask, spectral_gap, mixing_time_tv, transition_matrix,
                              proposal_matrix_swap, proposal_matrix_independent, run_chain,
                              swap_proposal, multi_swap_proposal, independent_uniform_proposal,
                              matrix_proposal)
from fairbench.instances import p0_instance, island_instance, island_family_legacy as island_family, describe_instance
from fairbench.metrics import coverage, tv_to_uniform, tv_expected_uniform
from fairbench.quantum.hamiltonian import penalty_operator, diag_values
from fairbench.quantum.proposal import subspace_basis, subspace_hamiltonian, exact_proposal_matrix

TS = np.geomspace(0.1, 20.0, 12)
ALPHAS = [0, 0.25, 0.5, 1, 2, 4, 8]
TOPOS = ["ring", "complete"]
MS = list(range(1, 11))
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def P_from_rows(QF, F):
    """MH kernel on F from Q rows restricted to F (M, C); diagonal absorbs rejected mass."""
    P = QF[:, F].copy()
    np.fill_diagonal(P, 0.0)
    P[np.diag_indices_from(P)] = 1.0 - P.sum(1)
    return P


def gap_rows(QF, F):
    return spectral_gap(P_from_rows(QF, F))


class Inst:
    def __init__(self, name, u, cs):
        self.name, self.u, self.cs = name, u, cs
        self.n, self.k = u.n, cs.cardinality
        self.F = np.flatnonzero(feasible_mask(u, cs))
        self.M, self.C = len(self.F), math.comb(self.n, self.k)
        self.comps = describe_instance(u, cs)["n_components"]
        self.states, _ = subspace_basis(self.n, self.k)
        pen = diag_values(penalty_operator(cs, u), self.n)
        from fairbench.quantum.proposal import states_to_codes
        sub = pen[states_to_codes(self.states)]
        self.pen_scale = float(sub.std()) or 1.0
        self.pen_diag = pen / self.pen_scale  # normalised 2^n diagonal

    # classical
    def classical(self):
        S = proposal_matrix_swap(self.n, self.k)
        lam, V = np.linalg.eigh(S)
        out = {"swap": gap_rows(S[self.F], self.F)}
        ms = {}
        for m in MS:
            QF = (V[self.F] * lam ** m) @ V.T
            ms[m] = gap_rows(QF, self.F)
        out["multi"] = ms
        out["indep"] = gap_rows(proposal_matrix_independent(self.n, self.k)[self.F], self.F)
        return out

    def quantum_grid(self):
        rows = []
        for topo in TOPOS:
            for a in ALPHAS:
                H = subspace_hamiltonian(self.n, self.k, topo, float(a), self.pen_diag)
                w, V = np.linalg.eigh(H)
                VF = V[self.F]
                for t in TS:
                    UF = (VF * np.exp(-1j * w * t)) @ V.T
                    QF = np.abs(UF) ** 2
                    rows.append(dict(instance=self.name, n=self.n, k=self.k, M=self.M, C=self.C,
                                     topology=topo, alpha=a, t=float(t), gap=gap_rows(QF, self.F)))
        return rows

    def Q_exact(self, t, a, topo):
        H = subspace_hamiltonian(self.n, self.k, topo, float(a), self.pen_diag)
        return exact_proposal_matrix(t, H)


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=8192); args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    insts = {}
    def get(name):
        if name not in insts:
            if name == "p0": insts[name] = Inst("p0", *p0_instance())
            else:
                n = int(name[1:]); insts[name] = Inst(name, *island_family(n))
        return insts[name]
    names = ["f8", "f10", "f12", "f14", "f16", "p0"]
    grid, cls = [], {}
    for nm in names:
        I = get(nm); ts = time.time()
        cls[nm] = I.classical(); g = I.quantum_grid(); grid += g
        print(f"{nm}: n={I.n} k={I.k} M={I.M} C={I.C} comps={I.comps} ({time.time()-ts:.0f}s)", flush=True)
    G = pd.DataFrame(grid); G["instance"] = G.instance.map(lambda s: {"f16": "island16"}.get(s, s))
    G.to_csv(f"{OUT}/q_gap_grid.csv", index=False)
    G["instance"] = G.instance.replace({"island16": "f16"})

    # ---- TUNE on f12 only
    T = G[G.instance == "f12"]
    best = T.loc[T.gap.idxmax()]
    tp = dict(t=float(best.t), alpha=float(best.alpha), topology=best.topology)
    m_star = max(cls["f12"]["multi"], key=cls["f12"]["multi"].get)
    print("FROZEN", tp, "m*", m_star, "tune gap", best.gap)

    def frozen_gap(nm):
        d = G[(G.instance == nm) & (G.topology == tp["topology"]) & (G.alpha == tp["alpha"]) &
              np.isclose(G.t, tp["t"])]
        return float(d.gap.iloc[0])

    def row(nm):
        c, I = cls[nm], insts[nm]
        d = G[G.instance == nm]; o = d.loc[d.gap.idxmax()]
        mo = max(c["multi"], key=c["multi"].get)
        best_cls = max(c["swap"], c["multi"][m_star], c["indep"], c["multi"][mo])
        qf = frozen_gap(nm)
        return dict(instance=nm, n=I.n, k=I.k, M=I.M, C=I.C, swap_components=I.comps,
                    weak_singleton_island=bool(I.comps >= 2 and describe_instance(I.u, I.cs)["component_sizes"][-1] <= 1),
                    role="tune" if nm == "f12" else "held-out",
                    swap=c["swap"], multi_frozen_m=m_star, multi_frozen=c["multi"][m_star],
                    multi_oracle_m=mo, multi_oracle=c["multi"][mo], indep=c["indep"],
                    qemcmc_frozen=qf, qemcmc_oracle=float(o.gap),
                    oracle_t=float(o.t), oracle_alpha=float(o.alpha), oracle_topology=o.topology,
                    ratio_frozen=qf / max(c["swap"], c["multi"][m_star], c["indep"]),
                    ratio_oracle=float(o.gap) / best_cls)
    S = pd.DataFrame([row(nm) for nm in names]); S.to_csv(f"{OUT}/q_gap_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S.round(4).to_string())

    # ---- mixing times (headline configs)
    mix = {}
    for nm in ["f12", "f14", "f16", "p0"]:
        I = insts[nm]
        w, V = np.linalg.eigh(subspace_hamiltonian(I.n, I.k, tp["topology"], tp["alpha"], I.pen_diag))
        QF = np.abs((V[I.F] * np.exp(-1j * w * tp["t"])) @ V.T) ** 2
        Pq = P_from_rows(QF, I.F)
        S1 = proposal_matrix_swap(I.n, I.k)
        lam, V = np.linalg.eigh(S1)
        Pm = P_from_rows((V[I.F] * lam ** m_star) @ V.T, I.F)
        Pi = P_from_rows(proposal_matrix_independent(I.n, I.k)[I.F], I.F)
        mix[nm] = dict(qemcmc=mixing_time_tv(Pq, max_steps=20000), multi=mixing_time_tv(Pm, max_steps=20000),
                       indep=mixing_time_tv(Pi, max_steps=20000), swap=mixing_time_tv(P_from_rows(S1[I.F], I.F), max_steps=20000))
        print("mix", nm, mix[nm], flush=True)

    # sanity: P_from_rows == transition_matrix on f12
    I = insts["f12"]; Qx = I.Q_exact(tp["t"], tp["alpha"], tp["topology"])
    mask = np.zeros(I.C, bool); mask[I.F] = True
    assert np.allclose(transition_matrix(Qx, mask), P_from_rows(Qx[I.F], I.F), atol=1e-8)

    # ---- sampled run on island16
    I = insts["f16"]; u, cs = I.u, I.cs
    Fset = enumerate_feasible(u, cs); x0 = Fset[0]
    Qx = I.Q_exact(tp["t"], tp["alpha"], tp["topology"])
    props = {"qemcmc_frozen": matrix_proposal(Qx, I.states), f"multi_swap_m{m_star}": multi_swap_proposal(m_star),
             "independent": independent_uniform_proposal(I.n, I.k), "swap": swap_proposal()}
    floor = tv_expected_uniform(I.M, args.budget)
    samp = []
    for nme, pr in props.items():
        res = []
        for s in range(args.seeds):
            r = run_chain(pr, x0, args.budget, cs, u, seed=s)
            res.append(dict(coverage=coverage(r.states, Fset), tv=tv_to_uniform(r.states, Fset),
                            acc=r.n_accepted / r.n_proposals))
        R = pd.DataFrame(res)
        samp.append(dict(sampler=nme, coverage_mean=R.coverage.mean(), coverage_sd=R.coverage.std(),
                         tv_minus_floor_mean=(R.tv - floor).mean(), tv_minus_floor_sd=R.tv.std(),
                         tv_mean=R.tv.mean(), accept_mean=R.acc.mean(), accept_sd=R.acc.std(), floor=floor,
                         budget=args.budget, seeds=args.seeds,
                         note="states recorded every step (autocorrelated); floor assumes iid draws"))
        print(samp[-1], flush=True)
    pd.DataFrame(samp).to_csv(f"{OUT}/q_gap_sampled.csv", index=False)

    json.dump(dict(frozen=dict(**tp, best_m=int(m_star)), tuned_on="island_family(12)",
                   tune_gap=float(best.gap), pen_normalisation="penalty diagonal / std over weight-k subspace",
                   h=None, mixing_time_tv=mix, grid=dict(t=list(map(float, TS)), alpha=ALPHAS, topology=TOPOS)),
              open(f"{OUT}/q_gap_best.json", "w"), indent=1)

    # ---- plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    d = G[(G.instance == "f16") & (G.topology == tp["topology"])]
    for a, g in d.groupby("alpha"):
        ax[0].plot(g.t, g.gap, marker="o", ms=3, label=f"alpha={a}")
    c = cls["f16"]
    for lab, v, ls in [("swap", c["swap"], ":"), (f"multi-swap m={m_star}", c["multi"][m_star], "--"),
                       ("best multi-swap (oracle)", max(c["multi"].values()), "-."), ("independent (M/C)", c["indep"], "-")]:
        ax[0].axhline(v, color="k", ls=ls, lw=1, label=lab)
    ax[0].set_xscale("log"); ax[0].set_yscale("symlog", linthresh=1e-4)
    ax[0].set_xlabel("evolution time t"); ax[0].set_ylabel("spectral gap")
    ax[0].set_title(f"island instance (n=16), topology={tp['topology']}"); ax[0].legend(fontsize=6)
    ns = [8, 10, 12, 14, 16]
    X = S.set_index("instance")
    for lab, col in [("QeMCMC frozen (tuned n=12)", "qemcmc_frozen"), ("QeMCMC oracle (optimistic)", "qemcmc_oracle"),
                     ("multi-swap frozen m", "multi_frozen"), ("independent", "indep"), ("swap", "swap")]:
        ax[1].plot(ns, [X.loc[f"f{n}", col] for n in ns], marker="o", label=lab)
    ax[1].set_yscale("symlog", linthresh=1e-4); ax[1].set_xlabel("n (island_family)"); ax[1].set_ylabel("spectral gap")
    ax[1].set_title("size trend (n=8,10 weak: singleton island)"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/q_gap.png", dpi=140)
    print(f"runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
