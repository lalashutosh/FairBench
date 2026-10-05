"""Benchmark metrics."""
from __future__ import annotations

import numpy as np

from .constraints import ConstraintSet
from .data import Universe


def _encode(X: np.ndarray) -> np.ndarray:
    """Rows -> Python-int-safe codes (bit i -> 2^i). Uses uint64 for n<=63, else bytes view."""
    X = np.atleast_2d(np.asarray(X)).astype(np.uint8)
    n = X.shape[1]
    if n <= 63:
        return (X.astype(np.uint64) << np.arange(n, dtype=np.uint64)).sum(axis=1, dtype=np.uint64)
    packed = np.packbits(X, axis=1, bitorder="little")
    return np.array([r.tobytes() for r in packed], dtype=object)


def acceptance_rate(X: np.ndarray, cs: ConstraintSet, u: Universe) -> float:
    """Fraction of rows of X satisfying all constraints (0.0 if X is empty)."""
    X = np.atleast_2d(X)
    if X.shape[0] == 0:
        return 0.0
    return float(cs.check_batch(X, u).mean())


def tv_to_uniform(X_feasible: np.ndarray, feasible_set: np.ndarray) -> float:
    """0.5 * sum |p_hat - 1/M| over the exact feasible set (M = len).

    Samples NOT in feasible_set count as mass off-support: they are included in the
    normalisation (p_hat = count/len(X)) and contribute fully to TV (0.5 * off-mass).
    Caveat: finite-sample bias, a perfect uniform sampler still has TV ~ sqrt(M/shots)
    scale; compare against tv_expected_uniform(M, shots)."""
    M = feasible_set.shape[0]
    X = np.atleast_2d(X_feasible)
    N = X.shape[0]
    if N == 0:
        return 1.0
    codes, counts = np.unique(_encode(X), return_counts=True)
    fcodes = _encode(feasible_set)
    in_set = np.isin(codes, fcodes)
    on_counts = counts[in_set]
    off_mass = counts[~in_set].sum() / N
    seen = on_counts.size
    # unseen feasible states contribute 1/M each
    tv = np.abs(on_counts / N - 1.0 / M).sum() + (M - seen) / M + off_mass
    return float(0.5 * tv)


def tv_expected_uniform(M: int, shots: int, seed: int | None = 0, reps: int = 20) -> float:
    """Mean TV of `shots` exact-uniform draws on M states vs uniform (finite-sample baseline)."""
    rng = np.random.default_rng(seed)
    tvs = []
    for _ in range(reps):
        c = np.bincount(rng.integers(0, M, size=shots), minlength=M)
        tvs.append(0.5 * np.abs(c / shots - 1.0 / M).sum())
    return float(np.mean(tvs))


def coverage(X_feasible: np.ndarray, feasible_set: np.ndarray) -> float:
    """Fraction of feasible_set seen at least once in X_feasible."""
    M = feasible_set.shape[0]
    X = np.atleast_2d(X_feasible)
    if M == 0:
        return 0.0
    if X.shape[0] == 0:
        return 0.0
    seen = np.intersect1d(_encode(X), _encode(feasible_set))
    return float(seen.size / M)


def cost_per_feasible_sample(n_cost: int, n_feasible: int) -> float:
    """Shots or chain moves per feasible sample (inf if none)."""
    return float("inf") if n_feasible == 0 else float(n_cost) / float(n_feasible)
