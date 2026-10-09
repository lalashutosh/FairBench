"""Attribution demo: is it the manager or the rules?

SYNTHETIC DATA with a planted ground truth (no real fund is analysed):
  * universe and rule set from ``scaled_family`` (sector caps, ESG floor, carbon cap)
    plus an exclusion list of the most carbon-intensive names;
  * returns are Gaussian with a planted "brown rally": carbon-intensive names earn
    ``--premium`` extra annual return per standard deviation of carbon intensity;
  * the fund is a rule-abiding portfolio planted at a known rank (``--rank``) among
    an independent pool of rule-abiding portfolios.
The demo checks that attribution recovers both: a negative constraint effect and a
fund percentile close to the planted rank.

Part 1 runs the headline case with classical rejection sampling. Part 2 repeats a
small case with rejection sampling and with the Dicke circuit on ``aer_statevector``
and compares both with exact enumeration: the quantum sampler is a drop-in backend
that draws the same distribution (a correctness check, not an advantage).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from any folder, installed or not

import argparse
import json
import time

import numpy as np

from fairbench.apps.attribution import (attribute, dicke_sampler, performance, plot_attribution,
                                        portfolio_returns, rejection_sampler, weights_matrix)
from fairbench.baselines import enumerate_feasible
from fairbench.constraints import ConstraintSet, Exclusion
from fairbench.data import synthetic_returns
from fairbench.instances import scaled_family

RES = Path(__file__).resolve().parent.parent / "results"


def build_case(n: int, k: int, seed: int, years: float, premium: float, rank: float, n_excluded: int):
    """Universe with synthetic returns, rule set, planted fund, and the fund's rank in its pool."""
    u, cs = scaled_family(n, k=k, seed=seed, esg_q=0.6, carbon_q=0.3, n_mc=50_000)
    cs = ConstraintSet(cs.constraints + [Exclusion(np.argsort(u.carbon)[-n_excluded:])])
    z = (u.carbon - u.carbon.mean()) / u.carbon.std()
    u.returns = synthetic_returns(u, periods=int(round(252 * years)), mu_shift=premium * z, seed=seed + 1)
    pool = rejection_sampler(u, cs, 2000, seed=seed + 2).samples  # independent of the null's seed
    perf = performance(portfolio_returns(weights_matrix(pool, "equal", u), u.returns.to_numpy()))
    fund = pool[np.argsort(perf)[int(rank * (len(pool) - 1))]]
    return u, cs, fund


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="universe size (headline case)")
    ap.add_argument("--k", type=int, default=20, help="holdings (headline case)")
    ap.add_argument("--samples", type=int, default=5000, help="random portfolios in the null")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--years", type=float, default=3.0, help="window length (252 periods per year)")
    ap.add_argument("--premium", type=float, default=0.08,
                    help="planted annual return per s.d. of carbon intensity")
    ap.add_argument("--rank", type=float, default=0.70, help="planted fund rank in [0, 1]")
    ap.add_argument("--no-quantum", action="store_true", help="skip part 2 (Dicke circuit on Aer)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    RES.mkdir(exist_ok=True)
    note = (f"Synthetic data, planted ground truth: carbon-heavy names earn +{a.premium:.0%}/yr per s.d.; "
            f"fund planted at rank {a.rank:.0%} of rule-abiding portfolios.")

    # ---- part 1: headline case, classical rejection sampling
    t0 = time.time()
    u, cs, fund = build_case(a.n, a.k, a.seed, a.years, a.premium, a.rank, n_excluded=a.n // 10)
    rules = ", ".join(sorted({type(c).__name__ for c in cs.constraints}))
    print(f"PART 1  n={u.n} assets, k={cs.cardinality} holdings, {len(u.returns)} periods; rules: {rules}")
    print(note)
    res = attribute(u, cs, fund, n_samples=a.samples, seed=a.seed)
    print(res.summary())
    sharpe = attribute(u, cs, fund, n_samples=a.samples, seed=a.seed, metric="sharpe")
    print(f"  risk-adjusted check (Sharpe): percentile {sharpe.percentile:.1f}, "
          f"constraint effect {sharpe.constraint_effect:+.3f}, manager effect {sharpe.manager_effect:+.3f}")
    verdict = ("the rules" if abs(res.constraint_effect) > abs(res.manager_effect) else "the manager")
    print(f"  => active return {res.active:+.1%}: mostly {verdict} "
          f"(planted rank {100 * a.rank:.0f}, recovered percentile {res.percentile:.0f})")
    import matplotlib.pyplot as plt
    plt.close(plot_attribution(res, RES / "attribution_demo.png", note=note))
    out = dict(synthetic=True, n=u.n, k=cs.cardinality, periods=len(u.returns), rules=rules,
               planted_premium=a.premium, planted_rank=a.rank, seed=a.seed,
               total_return=res.to_dict(), sharpe=sharpe.to_dict())
    (RES / "attribution_demo.json").write_text(json.dumps(out, indent=2))
    print(f"  saved {RES / 'attribution_demo.png'} and attribution_demo.json  ({time.time() - t0:.1f}s)\n")
    if a.no_quantum:
        return

    # ---- part 2: same pipeline, sampler swapped (small n so the circuit and the exact set are cheap)
    t0 = time.time()
    u, cs, fund = build_case(16, 5, a.seed, a.years, a.premium, a.rank, n_excluded=2)
    F = enumerate_feasible(u, cs)
    exact = performance(portfolio_returns(weights_matrix(F, "equal", u), u.returns.to_numpy()))
    f = float(performance(portfolio_returns(weights_matrix(fund, "equal", u), u.returns.to_numpy()))[0])
    print(f"PART 2  n={u.n}, k={cs.cardinality}: {len(F)} rule-abiding portfolios (exact enumeration)")
    rows = [("exact enumeration", 100 * (np.sum(exact < f) + 0.5 * np.sum(exact == f)) / len(F),
             float("nan"), float(np.median(exact)), float("nan"), "")]
    runs = {}
    for name, sampler in (("rejection (classical)", rejection_sampler),
                          ("Dicke circuit, aer_statevector", dicke_sampler("aer_statevector"))):
        r = runs[name] = attribute(u, cs, fund, n_samples=a.samples, sampler=sampler, seed=a.seed)
        rows.append((name, r.percentile, r.percentile_se, r.null_median, r.constraint_effect,
                     f"{r.cost / r.n_null:.1f} {r.cost_unit}"))
    print(f"  {'sampler':<32}{'percentile':>12}{'null median':>13}{'constraint':>12}  cost per sample")
    for name, pct, se, med, ce, cost in rows:
        pm = "" if np.isnan(se) else f" +/-{se:.1f}"
        ces = "" if np.isnan(ce) else f"{ce:+.2%}"
        print(f"  {name:<32}{pct:>6.1f}{pm:<8}{med:>11.2%}{ces:>12}  {cost}")
    from scipy.stats import ks_2samp
    ks = ks_2samp(*(r.null for r in runs.values()))
    print(f"  two-sample KS test, rejection vs Dicke null: D={ks.statistic:.3f}, p={ks.pvalue:.2f} "
          "(same distribution expected)")
    print("  Dicke + filter is classical rejection sampling in distribution: a drop-in backend, "
          "not a speedup.")
    with open(RES / "attribution_samplers.csv", "w") as fh:
        fh.write("sampler,percentile,percentile_se,null_median,constraint_effect,cost_per_sample\n")
        for name, pct, se, med, ce, cost in rows:
            fh.write(f"{name},{pct:.3f},{se:.3f},{med:.6f},{ce:.6f},{cost}\n")
    print(f"  saved {RES / 'attribution_samplers.csv'}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
