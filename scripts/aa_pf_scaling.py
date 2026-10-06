"""P_F scaling of scaled_family: MC feasible fraction with Wilson CI + rule decomposition."""
from __future__ import annotations

import math
import sys
import time
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from fairbench.constraints import CarbonCap, MinESG, SectorCap
from fairbench.instances import scaled_family


def wilson(h, N, z=1.96):
    if N == 0:
        return 0.0, 1.0
    p = h / N
    d = 1 + z * z / N
    c = (p + z * z / (2 * N)) / d
    w = z * math.sqrt(p * (1 - p) / N + z * z / (4 * N * N)) / d
    return max(0.0, c - w), min(1.0, c + w)


def estimate(u, cs, N_max=10_000_000, chunk=200_000, target_hits=3000, seed=12345):
    k = cs.cardinality
    n = u.n
    si = np.array([int(s[1:]) for s in u.sector])
    ns = si.max() + 1
    cap = next(c.max_count for c in cs.constraints if isinstance(c, SectorCap))
    mn = next(c.min_avg_score for c in cs.constraints if isinstance(c, MinESG))
    cm = next(c.max_avg for c in cs.constraints if isinstance(c, CarbonCap))
    rng = np.random.default_rng(seed)
    N = nc = ne = nf = 0
    while N < N_max and nf < target_hits:
        idx = np.argpartition(rng.random((chunk, n)), k - 1, axis=1)[:, :k]
        cnt = np.zeros((chunk, ns), np.int64)
        np.add.at(cnt, (np.repeat(np.arange(chunk), k), si[idx].ravel()), 1)
        a = (cnt <= cap).all(1)
        b = a & (u.esg_score[idx].mean(1) >= mn)
        c = b & (u.carbon[idx].mean(1) <= cm)
        N += chunk; nc += a.sum(); ne += b.sum(); nf += c.sum()
    return dict(N=N, hits=int(nf), n_caps=int(nc), n_caps_esg=int(ne),
                P_caps=nc / N, P_esg_given_caps=ne / max(nc, 1),
                P_carbon_given_caps_esg=nf / max(ne, 1))


def run(n, seed, q, k=None):
    u, cs = scaled_family(n, k=k, seed=seed, esg_q=q, carbon_q=q)
    e = estimate(u, cs)
    lo, hi = wilson(e["hits"], e["N"])
    p = e["hits"] / e["N"]
    ub = 3.0 / e["N"] if e["hits"] == 0 else float("nan")
    return dict(n=n, k=cs.cardinality, seed=seed, esg_q=q, carbon_q=q, P_F=p, ci_lo=lo, ci_hi=hi,
                n_samples=e["N"], hits=e["hits"], upper_bound_3overN=ub,
                P_caps=e["P_caps"], P_esg_given_caps=e["P_esg_given_caps"],
                P_carbon_given_caps_esg=e["P_carbon_given_caps_esg"])


if __name__ == "__main__":
    t0 = time.time()
    rows = []
    for n in (16, 25, 50, 100, 150, 200):
        for s in range(5):
            rows.append(dict(run(n, s, 0.5), sweep="scaling")); print(rows[-1], flush=True)
    for q in (0.7, 0.85, 0.95):  # 0.5 already in scaling grid at n=100
        for s in range(5):
            rows.append(dict(run(100, s, q), sweep="tightness")); print(rows[-1], flush=True)
    for s in range(5):
        r = [x for x in rows if x["n"] == 100 and x["seed"] == s and x["sweep"] == "scaling"][0]
        rows.append(dict(r, sweep="tightness"))
    df = pd.DataFrame(rows)
    df.to_csv("results/aa_pf_scaling.csv", index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sc = df[df.sweep == "scaling"]
    for s, g in sc.groupby("seed"):
        g = g.sort_values("n")
        y = np.where(g.hits > 0, g.P_F, g.upper_bound_3overN)
        lo = np.where(g.hits > 0, g.ci_lo, y); hi = np.where(g.hits > 0, g.ci_hi, y)
        ax[0].errorbar(g.n * (1 + 0.01 * s), y, yerr=[y - lo, hi - y], marker="o", ms=4, lw=1, capsize=2, label=f"seed {s}")
        for xx, yy, h in zip(g.n * (1 + 0.01 * s), y, g.hits):
            if h == 0: ax[0].plot(xx, yy, "v", color="k")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("n (k=max(5,n/10))"); ax[0].set_ylabel("P_F")
    ax[0].set_title("Feasible fraction (q=0.5; v = 3/N upper bound)"); ax[0].legend(fontsize=7)
    tg = df[df.sweep == "tightness"]
    for s, g in tg.groupby("seed"):
        g = g.sort_values("esg_q")
        y = np.where(g.hits > 0, g.P_F, g.upper_bound_3overN)
        ax[1].plot(g.esg_q, y, marker="o", ms=4, lw=1, label=f"seed {s}")
        for xx, yy, h in zip(g.esg_q, y, g.hits):
            if h == 0: ax[1].plot(xx, yy, "v", color="k")
    ax[1].set_yscale("log"); ax[1].set_xlabel("ESG & carbon quantile (n=100)"); ax[1].set_ylabel("P_F")
    ax[1].set_title("Rule tightness sweep"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("results/aa_pf_scaling.png", dpi=130)

    def tab(g):
        pf = g.P_F.where(g.hits > 0, g.upper_bound_3overN)
        med = pf.median()
        return pd.Series(dict(k=g.k.iloc[0], med=med, lo=pf.min(), hi=pf.max(), inv_sqrt=1 / math.sqrt(med),
                              zero_hit=int((g.hits == 0).sum()), P_caps=g.P_caps.median(),
                              P_esg=g.P_esg_given_caps.median(), P_carb=g.P_carbon_given_caps_esg.median()))
    print("\nSCALING (q=0.5)"); print(sc.groupby("n").apply(tab).to_string(float_format=lambda x: f"{x:.4g}"))
    print("\nTIGHTNESS n=100"); print(tg.groupby("esg_q").apply(tab).to_string(float_format=lambda x: f"{x:.4g}"))
    print(f"runtime {time.time()-t0:.0f}s")
