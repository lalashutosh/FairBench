"""P1-A4: MPS simulability sweep. Where does classical MPS simulation of the ansatz get expensive?

Per (n, p): sample 2048 shots on aer_mps with bond_dim in CHIS; compare against a reference
(exact Statevector probs for n<=20, else unbounded aer_mps). Also records the true max bond dim
of the unbounded MPS (save_matrix_product_state).
"""
from __future__ import annotations

import argparse, math, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from fairbench.backends import sample
from fairbench.constraints import ConstraintSet, Cardinality, SectorCap, MinESG
from fairbench.data import synthetic_universe
from fairbench.quantum.ansatz import build_ansatz
from fairbench.quantum.hamiltonian import penalty_operator

SHOTS = 2048
CHIS = [2, 4, 8, 16, 32, 64, None]


def instance(n):
    u = synthetic_universe(n, 3, seed=0)
    k = n // 3
    cs = ConstraintSet([Cardinality(k)] + [SectorCap(f"S{i}", math.ceil(k / 2)) for i in range(3)]
                       + [MinESG(float(np.percentile(u.esg_score, 60)))])
    return u, k, cs, penalty_operator(cs, u)


def energies(op, X):
    z = op.paulis.z.astype(np.int64)
    c = (op.coeffs * (-1j) ** op.paulis.phase).real
    par = (X.astype(np.int64) @ z.T) & 1
    return (1 - 2 * par) @ c


def codes(X):
    return X.astype(np.int64) @ (1 << np.arange(X.shape[1], dtype=np.int64))


def tv_emp(A, B):
    ca, cb = np.unique(codes(A), return_counts=True), np.unique(codes(B), return_counts=True)
    da, db = dict(zip(*ca)), dict(zip(*cb))
    keys = set(da) | set(db)
    return 0.5 * sum(abs(da.get(x, 0) - db.get(x, 0)) for x in keys) / len(A)


def true_bond(circ, params):
    qc = circ.assign_parameters(params, inplace=False)
    qc.save_matrix_product_state()
    sim = AerSimulator(method="matrix_product_state")
    t = time.time()
    res = sim.run(transpile(qc, sim), shots=1).result()
    mps = res.data()["matrix_product_state"]
    return max([int((np.asarray(l) > 1e-10).sum()) for l in mps[1]] + [1]), time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[12, 16, 20, 24, 28])
    ap.add_argument("--ps", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--append", action="store_true", help="merge with existing mps_sweep.csv")
    ap.add_argument("--exact-max-n", type=int, default=20)
    ap.add_argument("--budget", type=float, default=270.0, help="total seconds")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    t_start = time.time()
    rows = []
    for n in a.ns:
        u, k, cs, op = instance(n)
        for p in a.ps:
            if time.time() - t_start > a.budget:
                print(f"[budget] skipping n={n} p={p}"); continue
            circ = build_ansatz(n, k, p, op, "ring")
            rng = np.random.default_rng(1000 * n + p)
            params = {}
            for prm in circ.parameters:
                params[prm] = float(rng.uniform(0, np.pi / 2))
            maxbond, _ = true_bond(circ, params)
            # reference
            if n <= a.exact_max_n:
                sv = Statevector(circ.assign_parameters(params))
                pr = sv.probabilities()
                rr = np.random.default_rng(7)
                idx = rr.choice(len(pr), size=SHOTS, p=pr / pr.sum())
                ref = ((idx[:, None] >> np.arange(n)) & 1).astype(np.uint8)
                idx2 = rr.choice(len(pr), size=SHOTS, p=pr / pr.sum())
                ref2 = ((idx2[:, None] >> np.arange(n)) & 1).astype(np.uint8)
                refname = "exact"
            else:
                ref = sample(circ, params, SHOTS, "aer_mps", seed=1)
                ref2 = sample(circ, params, SHOTS, "aer_mps", seed=2)
                refname = "mps_unbounded"
            tv_rr = tv_emp(ref, ref2)
            e_ref = energies(op, ref).mean()
            for chi in CHIS:
                if time.time() - t_start > a.budget:
                    print(f"[budget] stop at n={n} p={p} chi={chi}"); break
                t = time.time()
                X = sample(circ, params, SHOTS, "aer_mps", seed=3, bond_dim=chi)
                wall = time.time() - t
                rows.append(dict(n=n, p=p, k=k, chi=chi if chi else 0, chi_label=chi or "None",
                                 wall_s=wall, wk_rate=float((X.sum(1) == k).mean()),
                                 tv=tv_emp(X, ref), tv_refref=tv_rr,
                                 e_err=abs(energies(op, X).mean() - e_ref), ref=refname,
                                 true_maxbond=maxbond))
            print(f"n={n} p={p} done, true max bond={maxbond}, t={time.time()-t_start:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    old = os.path.join(a.out, "mps_sweep.csv")
    if a.append and os.path.exists(old):
        df = pd.concat([pd.read_csv(old), df]).drop_duplicates(["n", "p", "chi"], keep="last")
        df["chi_label"] = df.chi.map(lambda c: c if c else "None")
    df.to_csv(os.path.join(a.out, "mps_sweep.csv"), index=False)

    # chi needed
    need = []
    for (n, p), g in df.groupby(["n", "p"]):
        g = g.sort_values("chi")
        gb = g[g.chi > 0]
        # chi needed: weight-k >= 0.999 for this chi and every larger chi; TV within 1.5x ref-ref
        # floor only where that floor is informative (tv_refref < 0.5; at larger n sampling noise saturates TV).
        good = (gb.wk_rate >= 0.999) & ((gb.tv_refref >= 0.5) | (gb.tv <= 1.5 * gb.tv_refref))
        good = good[::-1].cummin()[::-1].astype(bool)
        ok = gb[good]
        chi_need = int(ok.chi.min()) if len(ok) else None  # None -> > 64
        unb = g[g.chi == 0].iloc[0]
        need.append(dict(n=n, p=p, true_maxbond=int(g.true_maxbond.iloc[0]), chi_needed=chi_need if chi_need else ">64",
                         wall_unbounded=unb.wall_s,
                         wall_at_need=float(ok.iloc[0].wall_s) if len(ok) else float(unb.wall_s)))
    nd = pd.DataFrame(need)
    print(nd.to_string(index=False))
    print(df[["n", "p", "chi_label", "wall_s", "wk_rate", "tv", "tv_refref", "e_err"]].round(4).to_string(index=False))
    nd.to_csv(os.path.join(a.out, "mps_chi_needed.csv"), index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n in sorted(df.n.unique()):
        g = df[(df.n == n) & (df.p == max(a.ps))].sort_values("chi")
        g = g[g.chi > 0]
        ax[0].plot(g.chi, g.wk_rate, "o-", label=f"n={n}")
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("max bond dim chi"); ax[0].set_ylabel("weight-k rate")
    ax[0].set_title(f"Hamming-weight preservation (p={max(a.ps)})"); ax[0].legend()
    for p in sorted(nd.p.unique()):
        g = nd[nd.p == p]
        ax[1].plot(g.n, g.wall_at_need, "o-", label=f"p={p}")
    ax[1].set_yscale("log"); ax[1].set_xlabel("n qubits"); ax[1].set_ylabel("wall time of 2048 shots at needed chi (s)")
    ax[1].set_title("MPS sampling time (unbounded if chi>64)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "mps_sweep.png"), dpi=130)
    print(f"total {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
