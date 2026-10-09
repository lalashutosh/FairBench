"""Exact reference distributions on small universes, and the metrics that compare a sampler with them.

For a universe small enough to enumerate, the three reference distributions of
``reference`` are computed exactly as (portfolios, probabilities). ``compare_to_exact``
then scores any sampler's output against that truth: classical or quantum, uniform or not.

Metrics (all on the level of whole portfolios unless stated):
    feasibility_rate      share of the sampler's raw draws that are feasible portfolios
    violation_rate        1 - feasibility_rate
    tv                    total variation distance between the feasible draws and the exact
                          distribution; tv_noise_floor is its expected value for an exact
                          sampler with the same number of draws (a TV at the floor is
                          sampling noise, not bias)
    kl                    KL(exact || sample). Finite only when every exact portfolio was
                          drawn at least once; otherwise ``kl`` is None and ``kl_smoothed``
                          (add-half smoothing, labelled as such) is reported instead
    marginal_error_*      error of each asset's inclusion frequency P(w_i > 0)
    return_dist_ks / _w1  Kolmogorov-Smirnov and Wasserstein-1 distance between the two
                          return distributions over one period
    percentile_error      error, in percentile points, of a given return's rank
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..baselines import enumerate_feasible
from ..constraints import ConstraintSet, WeightRuleSet
from ..data import Universe
from .reference import ReferenceSample, tilt_log_weights, weighted_percentile_of
from .weights import WeightGrid, enumerate_grid, subset_weights

_ROUND = 10**9  # weights are matched after rounding to 1e-9


@dataclass
class ExactReference:
    """Every feasible portfolio and its probability. weights: (S, n); prob: (S,), sums to 1."""
    weights: np.ndarray
    prob: np.ndarray
    distribution: str
    definition: str

    @property
    def n_feasible(self) -> int:
        return int(self.weights.shape[0])

    def returns(self, asset_returns: np.ndarray) -> np.ndarray:
        return self.weights @ np.asarray(asset_returns, dtype=float)

    def inclusion(self) -> np.ndarray:
        """(n,) probability that each asset is held."""
        return self.prob @ (self.weights > 0)

    def percentile_of(self, x: float, asset_returns: np.ndarray) -> float:
        return weighted_percentile_of(self.returns(asset_returns), x, self.prob)

    def sample(self, m: int, seed: int | None = None) -> np.ndarray:
        """(m, n) i.i.d. draws from the exact distribution (the ideal sampler)."""
        rng = np.random.default_rng(seed)
        return self.weights[rng.choice(self.n_feasible, size=m, p=self.prob)]


def _uniform(W: np.ndarray, distribution: str, definition: str) -> ExactReference:
    if len(W) == 0:
        raise ValueError("the feasible set is empty")
    return ExactReference(W, np.full(len(W), 1.0 / len(W)), distribution, definition)


def exact_uniform_subsets(u: Universe, cs: ConstraintSet, policy: str = "equal",
                          benchmark: np.ndarray | None = None,
                          weight_rules: WeightRuleSet | None = None, cap: float | None = None) -> ExactReference:
    """D1 exactly: enumerate the feasible k-subsets (n <= 25) and weight them by ``policy``."""
    X = enumerate_feasible(u, cs)
    W = subset_weights(X, policy, benchmark, cap) if len(X) else np.zeros((0, u.n))
    if weight_rules is not None and len(W):
        W = W[weight_rules.check_weights(W, u)]
    return _uniform(W, "uniform_subsets", f"exact: uniform over feasible subsets, {policy} weights")


def exact_uniform_weights(u: Universe, grid: WeightGrid, rules: WeightRuleSet | None,
                          limit: int = 2_000_000) -> ExactReference:
    """D2 exactly: enumerate the weight grid and keep the points that satisfy ``rules``."""
    W = grid.weights(enumerate_grid(grid, limit))
    if rules is not None:
        W = W[rules.check_weights(W, u)]
    return _uniform(W, "uniform_weights", f"exact: uniform over feasible grid points, step 1/{grid.units}")


def exact_benchmark_aware(base: ExactReference, u: Universe, benchmark: np.ndarray, tau: float,
                          cov: np.ndarray | None = None) -> ExactReference:
    """D3 exactly: the base distribution times exp(-TE^2 / (2 tau^2)), renormalised."""
    logp = np.log(base.prob) + tilt_log_weights(base.weights, u.cov if cov is None else cov, benchmark, tau)
    p = np.exp(logp - logp.max())
    return ExactReference(base.weights, p / p.sum(), "benchmark_aware",
                          f"{base.definition}; tilted by exp(-TE^2 / (2 tau^2)), tau = {tau:g}")


# ------------------------------------------------------------------ metrics
def _keys(W: np.ndarray) -> np.ndarray:
    """One hashable key per row (weights rounded to 1e-9)."""
    Q = np.ascontiguousarray(np.rint(np.atleast_2d(W) * _ROUND).astype(np.int64))
    return Q.view(np.dtype((np.void, Q.dtype.itemsize * Q.shape[1]))).ravel()


def _cdf_distances(a: np.ndarray, pa: np.ndarray, b: np.ndarray, pb: np.ndarray) -> tuple[float, float]:
    """(KS, Wasserstein-1) between two weighted samples on the line."""
    grid = np.unique(np.concatenate([a, b]))

    def cdf(v, p):  # weighted step CDF evaluated on the grid
        order = np.argsort(v)
        cum = np.concatenate([[0.0], np.cumsum(p[order])])
        return cum[np.searchsorted(v[order], grid, side="right")]

    diff = np.abs(cdf(a, pa) - cdf(b, pb))
    return float(diff.max()), float(np.sum(diff[:-1] * np.diff(grid)))


def tv_noise_floor(prob: np.ndarray, n: int, reps: int = 200, seed: int | None = 0) -> float:
    """Expected TV between the exact distribution and ``n`` i.i.d. draws from it."""
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, prob, size=reps)
    return float(np.mean(0.5 * np.abs(counts / n - prob[None, :]).sum(axis=1)))


def compare_to_exact(exact: ExactReference, weights: np.ndarray, *, n_raw: int | None = None,
                     importance: np.ndarray | None = None, asset_returns: np.ndarray | None = None,
                     realised_return: float | None = None, seed: int | None = 0) -> dict:
    """Score a sampler against the exact distribution.

    weights: (m, n) portfolios the sampler produced. Rows that are not one of the exact
    feasible portfolios count as violations. n_raw: number of raw draws the sampler made
    to get them, if it filtered before returning (default m). importance: optional (m,)
    weights for a reweighted sample.
    """
    W = np.atleast_2d(np.asarray(weights, dtype=float))
    m = len(W)
    imp = np.full(m, 1.0 / max(m, 1)) if importance is None else np.asarray(importance, dtype=float) / np.sum(importance)
    index = {k.tobytes(): i for i, k in enumerate(_keys(exact.weights))}
    state = np.array([index.get(k.tobytes(), -1) for k in _keys(W)], dtype=np.int64) if m else np.zeros(0, np.int64)
    feasible = state >= 0
    n_raw = m if n_raw is None else int(n_raw)
    out: dict = dict(n_samples=m, n_raw=n_raw, n_feasible_exact=exact.n_feasible,
                     feasibility_rate=float(feasible.sum() / n_raw) if n_raw else float("nan"))
    out["violation_rate"] = 1.0 - out["feasibility_rate"] if n_raw else float("nan")
    if not feasible.any():
        return {**out, "tv": 1.0, "kl": None, "kl_smoothed": None, "note": "no feasible draws"}

    w = imp[feasible] / imp[feasible].sum()
    p_hat = np.bincount(state[feasible], weights=w, minlength=exact.n_feasible)
    n_eff = float(1.0 / np.sum(w ** 2))
    out["effective_sample_size"] = n_eff
    out["tv"] = float(0.5 * np.abs(p_hat - exact.prob).sum())
    out["tv_noise_floor"] = tv_noise_floor(exact.prob, max(int(round(n_eff)), 1), seed=seed)
    out["coverage"] = float((p_hat > 0).sum() / exact.n_feasible)
    if (p_hat > 0).all():
        out["kl"] = float(np.sum(exact.prob * np.log(exact.prob / p_hat)))
    else:
        out["kl"] = None  # infinite: some feasible portfolios were never drawn
    smooth = (p_hat * n_eff + 0.5) / (n_eff + 0.5 * exact.n_feasible)
    out["kl_smoothed"] = float(np.sum(exact.prob * np.log(exact.prob / smooth)))

    err = p_hat @ (exact.weights > 0) - exact.inclusion()
    out["marginal_error_max"] = float(np.abs(err).max())
    out["marginal_error_rmse"] = float(np.sqrt(np.mean(err ** 2)))
    if asset_returns is not None:
        r = exact.returns(asset_returns)
        seen = p_hat > 0
        out["return_dist_ks"], out["return_dist_w1"] = _cdf_distances(r[seen], p_hat[seen], r, exact.prob)
        if realised_return is not None:
            truth = exact.percentile_of(realised_return, asset_returns)
            est = weighted_percentile_of(r, realised_return, p_hat)
            out.update(percentile_exact=truth, percentile_estimate=est, percentile_error=est - truth)
    return out


def compare_sample(exact: ExactReference, ref: ReferenceSample, asset_returns: np.ndarray | None = None,
                   realised_return: float | None = None, seed: int | None = 0) -> dict:
    """``compare_to_exact`` for a ``ReferenceSample`` (uses its importance weights; the
    feasibility rate is the sampler's acceptance rate)."""
    out = compare_to_exact(exact, ref.weights, importance=ref.importance, asset_returns=asset_returns,
                           realised_return=realised_return, seed=seed)
    out["feasibility_rate"], out["violation_rate"] = ref.acceptance_rate, 1.0 - ref.acceptance_rate
    out["n_raw"] = ref.n_proposed
    out["n_violations_in_sample"] = ref.n_violations
    return out
