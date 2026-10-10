"""The quantum estimator on a real-fund period: same inputs and same question as ``real_fund``.

``real_fund.build_period_case`` turns two filings into a ``PeriodCase`` (universe, returns,
the fund's portfolio). The classical path draws portfolios and counts how many the fund
beat. This module hands the SAME case to the quantum core (``docs/QUANTUM_CORE.md``): the fund's
return goes into the oracle as a threshold and the rank is one amplitude,

    a_G = share of valid k-stock portfolios whose return is below the fund's.

Three things are produced per period, each labelled for what it is:

``quantum_percentile``  the estimate an IDEAL (noiseless) quantum computer would return for
    the full universe, simulated from the amplitude, with its oracle-query count and the
    number of classical samples that give the same precision. No circuit is simulated at
    n = 500; the amplitude itself is measured by a large classical sample, which is the
    documented large-n method of the quantum core.
``exact_sleeve``  a small real sub-universe (the largest index names) where every
    portfolio can be listed: exact answer, state-vector Grover simulation and classical
    sampling side by side. This is the end-to-end correctness check on real data.
``oracle_resources``  the logical size of the circuit for the full universe (qubits,
    Toffoli and T gates per step). It is a fault-tolerant workload, far beyond today's machines.

What is quantum-expressible. The oracle needs "portfolio return < fund return" to be linear
in the 0/1 selection. Equal weights and benchmark-proportional weights are; the CAPPED
benchmark weighting that ``real_fund`` uses as its headline, the weight grid and the
benchmark-aware tilt are not. So the quantum numbers here answer the equal-weight (or
uncapped benchmark-weight) version of the question, and say so.

Exclusions are handled by preparing the Dicke state over the allowed stocks only, so every
basis state is a valid portfolio (a_F = 1) and one estimation run is enough. With no
rarity to exploit, the only quantum gain on such a rule set is in precision (queries
proportional to 1/eps instead of 1/eps^2). Nothing here is a wall-clock or hardware claim.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..baselines import random_k_subsets
from ..constraints import Cardinality, ConstraintSet, LinearThreshold
from ..data import Universe
from ..qae.attribution_qae import ExactInstance, classical_percentile, qae_percentile
from ..qae.estimators import IdealOracle, iqae_amplitude_tol
from .real_fund import PeriodCase

WEIGHTS = ("equal", "benchmark")


def _universe(keys: Sequence[str], r: np.ndarray) -> Universe:
    """A Universe over ``keys`` whose return window compounds to ``r`` (a zero second row
    pads the window: the quantum core reads g_i = prod_t (1 + r_it) and wants two rows)."""
    n = len(keys)
    returns = pd.DataFrame([r, np.zeros(n)], columns=list(keys), index=["period", "pad"])
    return Universe(tickers=list(keys), mu=np.full(n, np.nan), cov=np.full((n, n), np.nan), sector=["unknown"] * n,
                    esg_score=np.full(n, np.nan), carbon=np.full(n, np.nan), returns=returns)


def quantum_inputs(case: PeriodCase, weights: str = "equal", excluded_keys: Sequence[str] = ()) -> dict:
    """The quantum core's input contract for one period: universe over the ALLOWED stocks
    (qubit i = asset i), ``Cardinality(k)``, and the return comparison as a linear rule.

        equal weights       sum_i x_i g_i            <  k (1 + s)
        benchmark weights   sum_i x_i b_i (g_i-1-s)  <  0          (b = benchmark weight)

    with g_i = 1 + asset return and s the fund's frozen-holdings return."""
    if weights not in WEIGHTS:
        raise ValueError(f"weights must be one of {WEIGHTS}: the capped benchmark weighting is not linear in the "
                         "selection, so the oracle cannot express it")
    banned = set(excluded_keys)
    keep = np.array([k not in banned for k in case.keys])
    keys = [k for k, ok in zip(case.keys, keep) if ok]
    r, b = case.asset_returns[keep], case.benchmark[keep]
    k, s = case.k, case.fund_frozen_return
    if weights == "equal":
        rule = LinearThreshold(1.0 + r, k * (1.0 + s), "<", label="equal-weight return below the fund's")
    else:
        rule = LinearThreshold(b * (r - s), 0.0, "<", label="benchmark-weight return below the fund's")
    return dict(u=_universe(keys, r), cs=ConstraintSet([Cardinality(k)]), below_fund=rule, k=k, n=len(keys), s=s,
                weights=weights, n_excluded=int((~keep).sum()))


def _amplitude(inp: dict, n_mc: int, seed: int | None, batch: int = 20_000) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    hits = done = 0
    while done < n_mc:
        m = min(batch, n_mc - done)
        X = random_k_subsets(inp["n"], inp["k"], m, int(rng.integers(2**31)))
        hits += int(inp["below_fund"].check_batch(X, inp["u"]).sum())
        done += m
    return hits, done


def quantum_percentile(case: PeriodCase, *, weights: str = "equal", excluded_keys: Sequence[str] = (),
                       eps: float = 0.01, alpha: float = 0.05, n_mc: int = 200_000, reps: int = 25,
                       seed: int | None = 0) -> dict:
    """Percentile of the fund on the full universe as an ideal quantum computer would
    estimate it, to +-``eps`` (as a fraction; 0.01 = one percentile point) at confidence
    1 - ``alpha``, with what it costs in oracle queries.

    The amplitude a_G is first measured by ``n_mc`` classical draws; ``IdealOracle`` then
    plays a noiseless device with that amplitude and iterative amplitude estimation is run
    on it ``reps`` times (pilot queries are charged). ``classical_samples_same_precision``
    is z^2 a (1 - a) / eps^2, the i.i.d. samples a classical estimate needs for the same
    half-width. One query = one oracle call = one classical sample checked."""
    inp = quantum_inputs(case, weights, excluded_keys)
    hits, n = _amplitude(inp, n_mc, seed)
    a = hits / n
    z = float(norm.ppf(1 - alpha / 2))
    half = z * math.sqrt(max(a * (1 - a), 1e-12) / n)
    oracle = IdealOracle(min(max(a, 0.0), 1.0))
    rng = np.random.default_rng(seed)
    runs = [iqae_amplitude_tol(oracle, eps, alpha, rng) for _ in range(reps)]
    queries = np.array([r.oracle_queries for r in runs], dtype=float)
    est = np.array([r.estimate for r in runs])
    classical = z * z * a * (1 - a) / (eps * eps)
    return dict(
        weights=weights, n=inp["n"], k=inp["k"], n_excluded=inp["n_excluded"], fund_return=inp["s"], eps=eps, alpha=alpha,
        percentile=100 * a, percentile_mc_half_width=100 * half, n_mc=n,
        quantum_estimate=100 * float(np.median(est)), quantum_worst_error=100 * float(np.abs(est - a).max()),
        quantum_queries_median=float(np.median(queries)), quantum_queries_p90=float(np.quantile(queries, 0.9)),
        classical_samples_same_precision=float(classical),
        query_ratio=float(classical / np.median(queries)) if classical > 0 else float("nan"),
        device="ideal noiseless device simulated from the amplitude; no circuit simulated at this size",
        valid_fraction=1.0)


def exact_sleeve(case: PeriodCase, n_top: int = 16, eps: float = 0.02, max_states: int = 200_000,
                 seed: int | None = 0) -> dict | None:
    """Exact answer, state-vector Grover simulation and classical sampling on a small REAL
    sub-universe: the ``n_top`` largest index stocks, choosing as many as the fund holds
    among them, equal weights, over the same period. None if the fund holds none or all of
    them. ``n_top`` is reduced until every portfolio can be listed."""
    order = np.argsort(-case.benchmark)
    while n_top >= 6:
        idx = np.sort(order[:n_top])
        held = case.fund_weights[idx] > 0
        k = int(held.sum())
        if 0 < k < n_top and math.comb(n_top, k) <= max_states:
            break
        n_top -= 2
    else:
        return None
    if not 0 < k < n_top:
        return None
    u = _universe([case.keys[i] for i in idx], case.asset_returns[idx])
    cs = ConstraintSet([Cardinality(k)])
    inst = ExactInstance.build(u, cs)
    rng = np.random.default_rng(seed)
    q = qae_percentile(u, cs, held.astype(np.uint8), method="iqae", eps=eps, rng=rng, inst=inst,
                       oracle_kind="subspace")
    s = q.details["s"]
    same_budget = classical_percentile(inst, s, int(q.oracle_queries), np.random.default_rng(seed))
    return dict(n=n_top, k=k, n_portfolios=inst.C, fund_return=float(s), exact=100 * float(q.details["exact"]),
                quantum=100 * float(q.estimate), quantum_ci=(100 * float(q.ci[0]), 100 * float(q.ci[1])),
                quantum_queries=int(q.oracle_queries), classical_same_queries=100 * float(same_budget), eps=eps,
                device="exact state-vector Grover simulation in the weight-k subspace (noiseless)")


def oracle_resources(case: PeriodCase, weights: str = "equal", excluded_keys: Sequence[str] = (),
                     bits: int | None = None) -> dict:
    """Logical circuit size for the full-universe estimate (``ft.resources``): qubits, and
    Toffoli and T gates in one Grover step (state preparation twice, the oracle, a
    reflection). Formula counts without the quantisation search; a fault-tolerant cost
    model, not something a current machine can run."""
    from ..ft.resources import resource_estimate

    inp = quantum_inputs(case, weights, excluded_keys)
    cs = ConstraintSet([Cardinality(inp["k"]), inp["below_fund"]])
    rep = resource_estimate(inp["u"], cs, bits=bits, oracle_path="approx", p_f=0.5)
    return dict(n=inp["n"], k=inp["k"], logical_qubits=int(rep.logical_qubits),
                toffoli_per_step=float(rep.toffoli_per_iterate), t_gates_per_step=float(rep.t_per_iterate),
                rotations_per_step=int(rep.n_rotations_per_iterate),
                note="logical counts for one Grover step on an error-corrected machine; excludes magic-state "
                     "factories and routing")
