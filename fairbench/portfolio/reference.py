"""Reference distributions: which feasible portfolios a fund is compared with, and how likely each is.

Three distributions over portfolios that obey the same rules. They are different objects,
not three estimates of one thing, and every result names the one it used.

D1  ``uniform_subsets``   Every feasible SET OF NAMES is equally likely. The weights are a
                          fixed function of the set (equal, or proportional to benchmark
                          weight). This is what ``baselines.rejection_sample`` draws.
D2  ``uniform_weights``   Every feasible point of the weight grid (``weights.WeightGrid``)
                          is equally likely, so uneven weightings count as much as even
                          ones, and a set of names with more feasible weightings gets more
                          mass.
D3  ``benchmark_aware``   D1 or D2 reweighted by exp(-TE(w)^2 / (2 tau^2)), TE = tracking
                          error against the benchmark. tau -> infinity returns the base
                          distribution; tau -> 0 concentrates on the feasible portfolio
                          closest to the benchmark. tau is an ASSUMPTION and is reported.

All samplers here are exact for their stated distribution (rejection from an exactly
uniform proposal); D3 is carried as importance weights with its effective sample size.
A ``ReferenceSample`` records what is needed to report a result: the definition, the
sampler, the seed, the acceptance rate, the estimated number of feasible portfolios and an
independent re-check for rule violations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..baselines import random_k_subsets
from ..constraints import ConstraintSet, WeightRuleSet, active_risk
from ..data import Universe
from .weights import WeightGrid, grid_size, sample_grid, subset_weights

DISTRIBUTIONS = ("uniform_subsets", "uniform_weights", "benchmark_aware")


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(centre - half, 0.0), min(centre + half, 1.0)


@dataclass
class ReferenceSample:
    """Feasible portfolios drawn from one reference distribution.

    weights: (m, n), rows sum to 1.  importance: (m,) non-negative, sums to 1 (uniform
    unless the distribution is ``benchmark_aware``).  log10_n_feasible: log10 of the number
    of feasible portfolios, exact when ``n_feasible_exact`` else acceptance rate x proposal
    space, with ``log10_n_feasible_ci`` from the Wilson interval of the acceptance rate.
    n_violations: rows that fail an independent re-check of the rules (must be 0)."""
    weights: np.ndarray
    importance: np.ndarray
    distribution: str
    definition: str
    weighting_policy: str
    sampler: str
    seed: int | None
    n_proposed: int
    acceptance_rate: float
    log10_n_feasible: float | None
    log10_n_feasible_ci: tuple[float, float] | None
    n_feasible_exact: bool
    n_violations: int
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.weights.shape[0])

    @property
    def ess(self) -> float:
        """Effective sample size (sum w)^2 / sum w^2 of the importance weights."""
        return float(1.0 / np.sum(self.importance ** 2))

    def returns(self, asset_returns: np.ndarray) -> np.ndarray:
        """(m,) buy-and-hold return of each portfolio over ONE period, given the (n,)
        return of each asset over that period. NaN asset returns raise: the caller
        decides what to do with unknown returns, this function never fills them."""
        r = np.asarray(asset_returns, dtype=float)
        if r.shape != (self.weights.shape[1],):
            raise ValueError(f"asset_returns must have shape ({self.weights.shape[1]},)")
        if np.isnan(r).any():
            raise ValueError("asset_returns contains NaN; resolve unknown returns first")
        return self.weights @ r

    def manifest(self) -> dict:
        """The reportable facts about this sample (JSON-friendly)."""
        return dict(distribution=self.distribution, definition=self.definition,
                    weighting_policy=self.weighting_policy, sampler=self.sampler, seed=self.seed,
                    n_samples=self.n, n_proposed=self.n_proposed, acceptance_rate=self.acceptance_rate,
                    log10_n_feasible=self.log10_n_feasible, log10_n_feasible_ci=self.log10_n_feasible_ci,
                    n_feasible_exact=self.n_feasible_exact, n_violations=self.n_violations,
                    effective_sample_size=self.ess, **self.info)


def _log10_count(count: int, lo: float, acc: float, hi: float) -> tuple[float | None, tuple[float, float] | None]:
    if acc <= 0 or count <= 0:
        return None, None
    base = math.log10(count)
    return base + math.log10(acc), (base + math.log10(max(lo, 1e-300)), base + math.log10(hi))


def uniform_subsets(u: Universe, cs: ConstraintSet, n_samples: int, *, policy: str = "equal",
                    benchmark: np.ndarray | None = None, cap: float | None = None,
                    weight_rules: WeightRuleSet | None = None,
                    seed: int | None = 0, batch: int = 20_000, max_cost: int = 20_000_000) -> ReferenceSample:
    """D1. Uniform over the k-subsets that satisfy ``cs`` and whose policy weights satisfy
    ``weight_rules`` (if given). Rejection from uniform k-subsets of the non-excluded
    assets, so the kept draws are exactly uniform on the feasible subsets."""
    k = cs.cardinality
    if k is None:
        raise ValueError("ConstraintSet has no Cardinality constraint")
    allowed = cs.allowed_indices(u.n)
    if allowed.size < k:
        raise RuntimeError(f"exclusions leave {allowed.size} assets, fewer than k={k}")
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    have = cost = 0
    while have < n_samples:
        if cost >= max_cost:
            raise RuntimeError(f"only {have}/{n_samples} feasible portfolios after {cost} proposals "
                               f"(acceptance {have / cost:.2e}); the rule set is too tight for rejection")
        X = np.zeros((batch, u.n), dtype=np.uint8)
        X[:, allowed] = random_k_subsets(allowed.size, k, batch, int(rng.integers(2**31)))
        X = X[cs.check_batch(X, u)]
        W = subset_weights(X, policy, benchmark, cap) if len(X) else np.zeros((0, u.n))
        if weight_rules is not None and len(W):
            W = W[weight_rules.check_weights(W, u)]
        kept.append(W)
        have += len(W)
        cost += batch
    W_all = np.concatenate(kept)
    acc = have / cost
    lo, hi = wilson(have, cost)
    log10_n, ci = _log10_count(math.comb(int(allowed.size), k), lo, acc, hi)
    W = W_all[:n_samples]
    ok = cs.check_batch((W > 0).astype(np.uint8), u)
    if weight_rules is not None:
        ok &= weight_rules.check_weights(W, u)
    return ReferenceSample(
        weights=W, importance=np.full(len(W), 1.0 / len(W)), distribution="uniform_subsets",
        definition=(f"uniform over the {k}-name subsets of {allowed.size} eligible assets that satisfy the "
                    f"rules; weights " + {"equal": "equal", "benchmark": "proportional to benchmark weight",
                                           "benchmark_capped": f"proportional to benchmark weight, capped at {cap}"}[policy]),
        weighting_policy=policy, sampler="rejection from uniform k-subsets", seed=seed, n_proposed=cost,
        acceptance_rate=acc, log10_n_feasible=log10_n, log10_n_feasible_ci=ci, n_feasible_exact=False,
        n_violations=int((~ok).sum()), info={"k": k, "n_eligible": int(allowed.size), "cap": cap})


def uniform_weights(u: Universe, grid: WeightGrid, rules: WeightRuleSet | None, n_samples: int, *,
                    seed: int | None = 0, batch: int = 20_000, max_cost: int = 20_000_000) -> ReferenceSample:
    """D2. Uniform over the grid points that satisfy ``rules``. Rejection from exactly
    uniform grid points (``weights.sample_grid``), so the kept draws are exactly uniform
    on the feasible grid points."""
    if grid.n != u.n:
        raise ValueError(f"grid has n={grid.n}, universe has n={u.n}")
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    have = cost = 0
    while have < n_samples:
        if cost >= max_cost:
            raise RuntimeError(f"only {have}/{n_samples} feasible portfolios after {cost} proposals "
                               f"(acceptance {have / max(cost, 1):.2e}); the rule set is too tight for rejection")
        W = grid.weights(sample_grid(grid, batch, int(rng.integers(2**31))))
        if rules is not None:
            W = W[rules.check_weights(W, u)]
        kept.append(W)
        have += len(W)
        cost += batch
    acc = have / cost
    lo, hi = wilson(have, cost)
    log10_n, ci = _log10_count(grid_size(grid), lo, acc, hi)
    W = np.concatenate(kept)[:n_samples]
    ok = grid.contains(np.rint(W * grid.units))
    if rules is not None:
        ok &= rules.check_weights(W, u)
    return ReferenceSample(
        weights=W, importance=np.full(len(W), 1.0 / len(W)), distribution="uniform_weights",
        definition=(f"uniform over feasible weight-grid points: weights in steps of 1/{grid.units}, "
                    f"{grid.k_min}-{grid.k_max} holdings, each between {grid.min_units}/{grid.units} "
                    f"and {grid.max_units}/{grid.units}"),
        weighting_policy="grid", sampler="rejection from uniform grid points", seed=seed, n_proposed=cost,
        acceptance_rate=acc, log10_n_feasible=log10_n, log10_n_feasible_ci=ci, n_feasible_exact=False,
        n_violations=int((~ok).sum()),
        info={"units": grid.units, "k_min": grid.k_min, "k_max": grid.k_max,
              "n_eligible": grid.n_allowed})


def tilt_log_weights(W: np.ndarray, cov: np.ndarray, benchmark: np.ndarray, tau: float) -> np.ndarray:
    """log of the D3 tilt, -TE(w)^2 / (2 tau^2). Quadratic in w."""
    if not tau > 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return -active_risk(W, cov, benchmark) ** 2 / (2.0 * tau * tau)


def benchmark_aware(base: ReferenceSample, u: Universe, benchmark: np.ndarray, tau: float,
                    cov: np.ndarray | None = None) -> ReferenceSample:
    """D3. Reweight a D1 or D2 sample by exp(-TE^2 / (2 tau^2)) with ``tau`` in the units
    of sqrt(cov) (annual, if cov is annual). Returns the same portfolios with importance
    weights. Check ``ess``: a small tau leaves few effective samples, and the result
    then rests on a handful of portfolios."""
    logw = tilt_log_weights(base.weights, u.cov if cov is None else cov, benchmark, tau)
    logw += np.log(base.importance)
    w = np.exp(logw - logw.max())
    return ReferenceSample(
        weights=base.weights, importance=w / w.sum(), distribution="benchmark_aware",
        definition=f"{base.definition}; reweighted by exp(-TE^2 / (2 tau^2)) with tau = {tau:g}",
        weighting_policy=base.weighting_policy, sampler=f"{base.sampler} + importance reweighting",
        seed=base.seed, n_proposed=base.n_proposed, acceptance_rate=base.acceptance_rate,
        log10_n_feasible=base.log10_n_feasible, log10_n_feasible_ci=base.log10_n_feasible_ci,
        n_feasible_exact=base.n_feasible_exact, n_violations=base.n_violations,
        info={**base.info, "tau": float(tau), "base_distribution": base.distribution})


# --------------------------------------------------------------- statistics
def weighted_quantile(values: np.ndarray, q, importance: np.ndarray | None = None) -> np.ndarray:
    """Quantile(s) of a weighted sample: the smallest value whose cumulative weight
    reaches q (left-continuous inverse of the weighted distribution function)."""
    v = np.asarray(values, dtype=float)
    w = np.full(v.size, 1.0 / v.size) if importance is None else np.asarray(importance, dtype=float)
    order = np.argsort(v, kind="stable")
    cum = np.cumsum(w[order])
    idx = np.searchsorted(cum, np.asarray(q, dtype=float) * cum[-1] - 1e-12, side="left")
    return v[order][np.minimum(idx, v.size - 1)]


def weighted_percentile_of(values: np.ndarray, x: float, importance: np.ndarray | None = None) -> float:
    """Mid-rank of x in a weighted sample, 0..100: weight below x plus half the weight
    equal to x."""
    v = np.asarray(values, dtype=float)
    w = np.full(v.size, 1.0 / v.size) if importance is None else np.asarray(importance, dtype=float)
    return float(100.0 * (w[v < x].sum() + 0.5 * w[v == x].sum()) / w.sum())


def reference_statistics(ref: ReferenceSample, asset_returns: np.ndarray,
                         realised_return: float | None = None) -> dict:
    """The rule-conditioned return range of a reference sample over one period and, if
    ``realised_return`` is given, the realised portfolio percentile in it.

    Descriptive only. The percentile says where the realised return fell among the
    portfolios this distribution treats as available; it is not evidence of skill and
    not a causal split. Standard errors cover sampling error only (they use the
    effective sample size)."""
    r = ref.returns(asset_returns)
    p05, median, p95 = (float(x) for x in weighted_quantile(r, [0.05, 0.5, 0.95], ref.importance))
    out = dict(distribution=ref.distribution, n_samples=ref.n, effective_sample_size=ref.ess,
               median=median, p05=p05, p95=p95, mean=float(np.sum(ref.importance * r)))
    if realised_return is not None:
        p = weighted_percentile_of(r, realised_return, ref.importance) / 100.0
        out.update(realised_return=float(realised_return), percentile=100.0 * p,
                   percentile_se=100.0 * math.sqrt(max(p * (1 - p), 0.0) / ref.ess),
                   within_mandate_return_difference=float(realised_return) - median)
    return out
