"""QA-6: classical rejection vs quantum amplitude estimation (QAE) on the REAL-TOOL mandate instance.

Instance = scripts/demo_mandate.py defaults: named_universe(n=150, seed=0), compiled rules of
examples/mandate_example.rules.json with k=20, carbon-premium returns (seed 1), planted fund at
rank 0.70 of 1000 rejection samples (seed 2). The instance is too large to enumerate, so the
amplitudes are Monte Carlo estimates from >= 2M uniform random k-subsets of the allowed assets
(same convention as fairbench.apps.attribution.rejection_sampler / classical_percentile).

  a_F = P(feasible)
  a_G = P(feasible AND S(x) < S_fund),  S(x) = equal-weight buy-and-hold total return
        (= mean_i g_i - 1, as portfolio_score in fairbench.qae.attribution_qae)
  p   = a_G / a_F  (percentile of the fund)

Classical proposals for target error eps: 1.96^2 p(1-p) / (eps^2 a_F).
Quantum: two IQAE runs (F and G) on IdealOracle at the true (MC) amplitudes, angle tolerance
eps_a / (2 sqrt(a(1-a))) as in scripts/qae_breakeven.py, error split between F and G by the
same allocation, combined with ratio_estimate. Reports median total oracle queries and the
fraction of repetitions with |p_hat - p| <= eps.

SYNTHETIC DATA (planted ground truth); no live API.
Outputs: results/qae_mandate.csv, results/qae_mandate.json.
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
from fairbench.qae.estimators import AEResult, IdealOracle, iqae, ratio_estimate        # noqa: E402

N, K, SEED, PERIODS, PREMIUM, RANK = 150, 20, 0, 756, 0.08, 0.70
N_TOTAL, BATCH = 2_000_000, 50_000
Z = 1.96
EPS_LIST = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
REPS = 30


def wilson(h: int, n: int, z: float = Z):
    p = h / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, (max(c - half, 0.0), min(c + half, 1.0))


def alloc(aF, aG, eps_p):
    """Split absolute percentile error eps_p between the F and G runs (copy of QModel.alloc,
    unit depth weights), returns (eps_F, eps_G) in amplitude units."""
    p = aG / aF
    best = None
    for phi in np.linspace(0.01, math.pi / 2 - 0.01, 3001):
        eF = aF * (eps_p / p) * math.cos(phi)
        eG = aG * (eps_p / p) * math.sin(phi)
        c = math.sqrt(aF * (1 - aF)) / eF + math.sqrt(aG * (1 - aG)) / eG
        if best is None or c < best[0]:
            best = (c, eF, eG)
    return best[1], best[2]


def angle_tol(a, eps_a):
    return min(eps_a / (2 * math.sqrt(a * (1 - a))), 0.49)


def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)

    # ---- instance (same construction as scripts/demo_mandate.py defaults)
    u = named_universe(N, seed=SEED)
    spec = load_spec(ROOT / "examples" / "mandate_example.rules.json")
    compiled = compile_spec(spec, u, k=K)
    cs = compiled.constraint_set
    z = (u.carbon - u.carbon.mean()) / u.carbon.std()
    u.returns = synthetic_returns(u, periods=PERIODS, mu_shift=PREMIUM * z, seed=SEED + 1)
    allowed = cs.allowed_indices(u.n)
    k = compiled.k
    pool = rejection_sampler(u, cs, 1000, seed=SEED + 2).samples
    perf = performance(portfolio_returns(weights_matrix(pool, "equal", u), u.returns.to_numpy()))
    fund = pool[np.argsort(perf)[int(RANK * (len(pool) - 1))]]
    assert all(ok for _, ok in compiled.check_fund(fund, u))
    scores = linear_return_scores(u)
    s_fund = portfolio_score(fund, u, scores=scores)
    print(f"INSTANCE n={u.n} k={k} allowed={allowed.size} rules={len(compiled.rules)} "
          f"S_fund(buy-and-hold EW)={s_fund:.5f}  [{time.time() - t0:.1f}s]", flush=True)

    # ---- Monte Carlo over uniform random k-subsets of the allowed assets
    hF = hG = total = 0
    seeds = np.random.default_rng(12345)
    for _ in range(N_TOTAL // BATCH):
        X = np.zeros((BATCH, u.n), dtype=np.uint8)
        X[:, allowed] = random_k_subsets(allowed.size, k, BATCH, int(seeds.integers(2**31)))
        feas = cs.check_batch(X, u).astype(bool)
        S = X.astype(float) @ scores / k          # equal weight over the k held names
        hF += int(feas.sum())
        hG += int((feas & (S < s_fund)).sum())
        total += BATCH
    aF, ciF = wilson(hF, total)
    aG, ciG = wilson(hG, total)
    p = aG / aF
    # ratio CI by delta method (same as ratio_estimate), via AEResult with the Wilson sd
    sdF = (ciF[1] - ciF[0]) / (2 * Z)
    sdG = (ciG[1] - ciG[0]) / (2 * Z)
    rr = ratio_estimate(AEResult(aG, ciG, 0, 0, total, "mc", [], dict(std=sdG)),
                        AEResult(aF, ciF, 0, 0, total, "mc", [], dict(std=sdF)))
    ciP = rr.ci
    print(f"MC  draws={total}  hits_F={hF} hits_G={hG}", flush=True)
    print(f"  a_F = {aF:.6f}  95% CI [{ciF[0]:.6f}, {ciF[1]:.6f}]", flush=True)
    print(f"  a_G = {aG:.6f}  95% CI [{ciG[0]:.6f}, {ciG[1]:.6f}]", flush=True)
    print(f"  p   = {p:.5f}  95% CI [{ciP[0]:.5f}, {ciP[1]:.5f}]  [{time.time() - t0:.1f}s]", flush=True)

    # ---- classical proposals vs IQAE queries per eps
    rows = []
    for eps in EPS_LIST:
        cls = Z * Z * p * (1 - p) / (eps * eps * aF)
        eF, eG = alloc(aF, aG, eps)
        q, hit, eF_t = [], 0, angle_tol(aF, eF)
        eG_t = angle_tol(aG, eG)
        rng = np.random.default_rng(7)
        for _ in range(REPS):
            r_F = iqae(IdealOracle(aF), angle_tol(aF, eF), 0.05, rng)
            r_G = iqae(IdealOracle(aG), angle_tol(aG, eG), 0.05, rng)
            r = ratio_estimate(r_G, r_F)
            q.append(r.oracle_queries)
            hit += abs(r.estimate - p) <= eps
        q_med = float(np.median(q))
        row = dict(eps=eps, classical_proposals=cls, quantum_queries_median=q_med,
                   quantum_queries_mean=float(np.mean(q)), frac_within_eps=hit / REPS,
                   ratio_classical_over_quantum=cls / q_med, eps_F_amp=eF, eps_G_amp=eG,
                   iqae_angle_tol_F=eF_t, iqae_angle_tol_G=eG_t, reps=REPS)
        rows.append(row)
        print(f"eps={eps:<6} classical={cls:14.4g}  quantum_median={q_med:12.4g}  "
              f"within_eps={hit / REPS:.2f}  ratio={cls / q_med:9.3g}  "
              f"[{time.time() - t0:.1f}s]", flush=True)

    with open(RES / "qae_mandate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    info = dict(
        synthetic=True, description="QA-6 real-tool mandate instance (demo_mandate defaults), MC amplitudes",
        n=N, k=K, allowed_assets=int(allowed.size), excluded_assets=int(N - allowed.size),
        n_rules=len(compiled.rules), n_unmapped=len(compiled.unmapped),
        s_fund_buy_hold=s_fund, planted_rank=RANK, planted_premium=PREMIUM,
        mc_draws=total, mc_batch=BATCH, hits_F=hF, hits_G=hG,
        a_F=dict(estimate=aF, ci=list(ciF)), a_G=dict(estimate=aG, ci=list(ciG)),
        p=dict(estimate=p, ci=list(ciP)), table=rows,
        notes="Classical proposals = 1.96^2 p(1-p)/(eps^2 a_F). Quantum = median total IQAE oracle "
              "queries of two IQAE runs (F, G) at amplitudes a_F, a_G on IdealOracle, combined with "
              "ratio_estimate; allocation as scripts/qae_breakeven.py alloc; angle tolerance "
              "eps_a/(2 sqrt(a(1-a))) capped at 0.49.",
    )
    (RES / "qae_mandate.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"saved results/qae_mandate.csv and results/qae_mandate.json  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
