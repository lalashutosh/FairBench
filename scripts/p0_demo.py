"""P0 demo: Dicke(p=0)+filter vs rejection vs MCMC vs classical, on a toy universe."""
import argparse
import itertools
import math
import time

import numpy as np

from fairbench.backends import sample
from fairbench.baselines import (enumerate_feasible, mcmc_swap_sample,
                                 random_k_subsets, rejection_sample)
from fairbench.constraints import Cardinality, ConstraintSet, MinESG, SectorCap
from fairbench.data import synthetic_universe
from fairbench.metrics import (acceptance_rate, cost_per_feasible_sample, coverage,
                               tv_expected_uniform, tv_to_uniform)
from fairbench.quantum.ansatz import build_ansatz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--burn-in", type=int, default=200)
    ap.add_argument("--thin", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()

    n, k = 12, 4
    u = synthetic_universe(n=n, n_sectors=3, seed=0)
    # MinESG threshold: 70th percentile of avg ESG over all k-subsets (deterministic).
    avgs = np.array([u.esg_score[list(c)].mean() for c in itertools.combinations(range(n), k)])
    m = float(np.quantile(avgs, 0.70))
    cons = [Cardinality(k)] + [SectorCap(s, 2) for s in sorted(set(u.sector))] + [MinESG(m)]
    cs = ConstraintSet(cons)
    print(f"n={n} k={k} sectors={sorted(set(u.sector))} shots={a.shots} seed={a.seed}")
    print(f"MinESG threshold m = {m:.4f} (70th percentile of avg ESG over all k-subsets)")

    F = enumerate_feasible(u, cs)
    M = len(F)
    total = math.comb(n, k)
    print(f"Feasible set: M = {M}, M/C({n},{k}) = {M}/{total} = {M/total:.3f}\n")

    rows = []

    def add(name, feas, acc, cost, unit):
        nf = len(feas)
        rows.append((name, acc, nf, tv_to_uniform(feas, F) if nf else float("nan"),
                     tv_expected_uniform(M, nf, seed=a.seed) if nf else float("nan"),
                     coverage(feas, F), cost_per_feasible_sample(cost, nf), unit))

    # (a) Dicke p=0 on aer_statevector + filter; cost = shots
    qc = build_ansatz(n, k, 0, None)
    X = sample(qc, None, a.shots, backend="aer_statevector", seed=a.seed)
    ok = cs.check_batch(X, u)
    acc_d = acceptance_rate(X, cs, u)
    add("dicke_p0+filter", X[ok], acc_d, a.shots, "shots")

    # (b) rejection
    rj = rejection_sample(u, cs, a.shots, seed=a.seed)
    acc_r = rj.info["n_accepted"] / a.shots
    add("rejection", rj.samples, acc_r, rj.cost, rj.cost_unit)

    # (c) MCMC: requests as many states as rejection kept feasible, so n_feasible matches.
    n_target = max(len(rj.samples), 1)
    mc = mcmc_swap_sample(u, cs, n_target, a.burn_in, a.thin, seed=a.seed)
    add("mcmc_swap", mc.samples, 1.0, mc.cost, mc.cost_unit)

    # (d) classical random k-subsets + filter (sanity)
    Xc = random_k_subsets(n, k, a.shots, seed=a.seed + 1)
    okc = cs.check_batch(Xc, u)
    add("classical_k+filter", Xc[okc], float(okc.mean()), a.shots, "shots")

    hdr = ["sampler", "acceptance", "n_feasible", "TV_to_unif", "TV_floor", "coverage", "cost/feasible"]
    lines = [hdr]
    for r in rows:
        lines.append([r[0], f"{r[1]:.3f}", str(r[2]), f"{r[3]:.4f}", f"{r[4]:.4f}",
                      f"{r[5]:.3f}", f"{r[6]:.2f} {r[7]}/feasible"])
    w = [max(len(l[i]) for l in lines) for i in range(len(hdr))]
    for i, l in enumerate(lines):
        print("  ".join(c.ljust(w[j]) for j, c in enumerate(l)))
        if i == 0:
            print("  ".join("-" * x for x in w))

    p = (acc_d + acc_r) / 2
    se = math.sqrt(max(p * (1 - p), 1e-12) * 2 / a.shots)
    z = abs(acc_d - acc_r) / se
    print(f"\nacceptance Dicke={acc_d:.4f} rejection={acc_r:.4f} diff={z:.2f} standard errors")
    if z > 4:
        print("WARNING: Dicke and rejection acceptance differ by > 4 standard errors!")
    print("Dicke p=0 + filter == random k-subsets + rejection (same distribution) "
          "-- correctness baseline, not quantum advantage.")
    print(f"wall time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
