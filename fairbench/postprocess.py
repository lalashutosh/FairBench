"""Post-processing: filtering, greedy repair, weight assignment.

NOTE ON BIAS: `filter_feasible` keeps the sampler's conditional distribution on
the feasible set (a uniform sampler stays uniform). `repair` does NOT: infeasible
samples get moved to a nearby feasible point by a greedy rule, so mass piles up on
"attractor" states and the output is no longer uniform. Always report the repaired
fraction (`postprocess(...)["n_repaired"]`) next to any repair-based metric.
"""
from __future__ import annotations

import numpy as np

from .constraints import (Cardinality, CarbonCap, ConstraintSet, Exclusion,
                          MinESG, SectorCap)
from .data import Universe


class RepairFailed(RuntimeError):
    """Greedy repair could not reach feasibility (stuck in a local minimum / max_steps)."""


# ---------------------------------------------------------------- violation
def _avg_violation(X: np.ndarray, values: np.ndarray, bound: float, lower: bool) -> np.ndarray:
    cnt = X.sum(axis=1)
    avg = (X @ values) / np.maximum(cnt, 1)
    scale = max(float(np.ptp(values)), 1e-12)
    gap = (bound - avg) if lower else (avg - bound)
    return np.where(cnt > 0, np.maximum(gap, 0.0) / scale, 1.0)  # empty -> 1


def violation_batch(X: np.ndarray, cs: ConstraintSet, u: Universe) -> np.ndarray:
    """(m,) total normalised violation; 0 iff feasible (exactly, via check_batch fallback)."""
    X = np.atleast_2d(np.asarray(X)).astype(np.int64, copy=False)
    tot = np.zeros(X.shape[0])
    for c in cs.constraints:
        if isinstance(c, Cardinality):
            tot += np.abs(X.sum(axis=1) - c.k) / max(c.k, 1)
        elif isinstance(c, SectorCap):
            m = np.array([s == c.sector for s in u.sector], dtype=np.int64)
            tot += np.maximum(X @ m - c.max_count, 0) / max(c.max_count, 1)
        elif isinstance(c, Exclusion):
            if c.indices.size:
                tot += X[:, c.indices].sum(axis=1)
        elif isinstance(c, MinESG):
            tot += _avg_violation(X, u.esg_score, c.min_avg_score, lower=True)
        elif isinstance(c, CarbonCap):
            tot += _avg_violation(X, u.carbon, c.max_avg, lower=False)
        else:  # unknown constraint: 0/1
            tot += (~c.check_batch(X, u)).astype(float)
    # guard: numerical edge (e.g. avg slightly below bound by float error) must agree with check
    bad = ~cs.check_batch(X, u)
    return np.where(bad, np.maximum(tot, 1e-9), 0.0)


def violation(x: np.ndarray, cs: ConstraintSet, u: Universe) -> float:
    return float(violation_batch(np.asarray(x)[None, :], cs, u)[0])


# ------------------------------------------------------------------- filter
def filter_feasible(X: np.ndarray, cs: ConstraintSet, u: Universe) -> tuple[np.ndarray, np.ndarray]:
    """Return (X[mask], mask). Preserves the sampler's distribution conditioned on feasibility."""
    X = np.atleast_2d(X)
    mask = cs.check_batch(X, u)
    return X[mask], mask


# ------------------------------------------------------------------- repair
def _pick(vals: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(vals <= vals.min() + 1e-12)
    return int(rng.choice(best))


def repair(x: np.ndarray, cs: ConstraintSet, u: Universe, seed: int | None = None,
           max_steps: int = 100) -> tuple[np.ndarray, bool]:
    """Greedy repair to feasibility, preserving cardinality k (if the set has one).

    1. If weight != k: greedily drop (or add) the asset that most reduces `violation`.
    2. Then greedy single swaps (one held out, one non-held in) choosing the min-violation
       result; ties broken with rng(seed). Stops at violation 0.

    Returns (x_out, was_repaired): (x, False) if x was already feasible (unchanged copy),
    (x_fixed, True) on success. FAILURE POLICY: raises RepairFailed if no strictly
    improving move exists or max_steps is hit (callers wanting no exception use repair_batch).
    Repair biases the output distribution (see module docstring)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x).astype(np.uint8).copy()
    if cs.check(x, u):
        return x, False
    k = cs.cardinality
    cur = violation(x, cs, u)
    steps = 0
    if k is not None:
        while int(x.sum()) != k and steps < max_steps:
            add = int(x.sum()) < k
            idx = np.flatnonzero(x == (0 if add else 1))
            if idx.size == 0:
                break
            C = np.repeat(x[None, :], idx.size, axis=0)
            C[np.arange(idx.size), idx] = 1 if add else 0
            vals = violation_batch(C, cs, u)
            j = _pick(vals, rng)
            x, cur = C[j], float(vals[j])
            steps += 1
    while cur > 0 and steps < max_steps:
        ins, outs = np.flatnonzero(x == 1), np.flatnonzero(x == 0)
        cand: list[tuple[int, int]] = [(i, j) for i in ins for j in outs]
        if k is None:  # no cardinality: allow single flips too
            cand += [(i, -1) for i in ins] + [(-1, j) for j in outs]
        if not cand:
            break
        C = np.repeat(x[None, :], len(cand), axis=0)
        for r, (i, j) in enumerate(cand):
            if i >= 0:
                C[r, i] = 0
            if j >= 0:
                C[r, j] = 1
        vals = violation_batch(C, cs, u)
        r = _pick(vals, rng)
        if vals[r] >= cur - 1e-12:
            raise RepairFailed(f"no improving swap (violation {cur:.4g})")
        x, cur = C[r], float(vals[r])
        steps += 1
    if cur > 0:
        raise RepairFailed(f"max_steps={max_steps} reached (violation {cur:.4g})")
    return x.astype(np.uint8), True


def repair_batch(X: np.ndarray, cs: ConstraintSet, u: Universe, seed: int | None = None,
                 max_steps: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repair every infeasible row. Returns (X_out, repaired_mask, failed_mask).

    Failed rows are left unchanged in X_out (still infeasible) and flagged in failed_mask.
    WARNING: repair biases the sample distribution toward attractor states near the
    infeasible region; report repaired_mask.mean() whenever using repaired samples."""
    X = np.atleast_2d(X)
    X_out = X.astype(np.uint8).copy()
    repaired = np.zeros(X.shape[0], dtype=bool)
    failed = np.zeros(X.shape[0], dtype=bool)
    ss = np.random.SeedSequence(seed)
    seeds = ss.generate_state(max(X.shape[0], 1))
    ok = cs.check_batch(X, u)
    for r in np.flatnonzero(~ok):
        try:
            X_out[r], repaired[r] = repair(X[r], cs, u, seed=int(seeds[r]), max_steps=max_steps)
        except RepairFailed:
            failed[r] = True
    return X_out, repaired, failed


# ------------------------------------------------------------------ weights
def assign_weights(x: np.ndarray, scheme: str = "equal", u: Universe | None = None, **kw) -> np.ndarray:
    """(n,) float weights summing to 1 over held assets, 0 elsewhere.

    schemes: "equal"; "inverse_vol" (1/sqrt(diag cov), needs u); "esg_tilt" (∝ esg_score, needs u;
    non-positive scores are clipped at 0, falling back to equal if all zero)."""
    x = np.asarray(x).astype(bool)
    if not x.any():
        raise ValueError("empty selection")
    if scheme == "equal":
        raw = np.ones(x.size)
    elif scheme == "inverse_vol":
        if u is None:
            raise ValueError("inverse_vol needs u")
        raw = 1.0 / np.sqrt(np.maximum(np.diag(u.cov), 1e-12))
    elif scheme == "esg_tilt":
        if u is None:
            raise ValueError("esg_tilt needs u")
        raw = np.maximum(np.asarray(u.esg_score, dtype=float), 0.0)
        if raw[x].sum() <= 0:
            raw = np.ones(x.size)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    w = np.where(x, raw, 0.0)
    return w / w.sum()


# ----------------------------------------------------------------- pipeline
def postprocess(X: np.ndarray, cs: ConstraintSet, u: Universe, mode: str = "filter",
                seed: int | None = 0, max_steps: int = 100) -> dict:
    """Single pipeline applied identically to every sampler.

    mode "filter": keep feasible rows. mode "repair": keep feasible rows, repair the rest
    (failures dropped). Returns dict: samples (feasible only), n_in, n_feasible_raw
    (feasible before any repair), n_repaired, n_failed. Invariant:
    len(samples) == n_feasible_raw + n_repaired; n_in == n_feasible_raw + n_repaired + n_failed
    + (infeasible dropped, filter mode)."""
    X = np.atleast_2d(X)
    if mode == "filter":
        S, mask = filter_feasible(X, cs, u)
        return dict(samples=S, n_in=X.shape[0], n_feasible_raw=int(mask.sum()),
                    n_repaired=0, n_failed=0)
    if mode == "repair":
        raw_ok = cs.check_batch(X, u)
        Xo, rep, fail = repair_batch(X, cs, u, seed=seed, max_steps=max_steps)
        keep = ~fail
        return dict(samples=Xo[keep], n_in=X.shape[0], n_feasible_raw=int(raw_ok.sum()),
                    n_repaired=int(rep.sum()), n_failed=int(fail.sum()))
    raise ValueError(f"unknown mode {mode!r}")
