"""Portfolio weights: weighting policies for a selection, and the Stage 2 weight grid.

Stage 1 is a selection x (0/1 per asset) with weights that are a fixed function of it:
    "equal"             w_i = x_i / k
    "benchmark"         w_i = b_i x_i / sum_j b_j x_j     (b = benchmark weights)
    "benchmark_capped"  the same, with no weight above ``cap``: the excess of a capped name
                        is spread over the others in proportion to their benchmark weights.
                        Needed for concentrated portfolios: 40 names drawn from a 500-name
                        index hold about 8% of it, so one very large company would otherwise
                        get half the portfolio.

Stage 2 is a weight grid. Weights are whole numbers of units of size 1/``units``:
    w_i = q_i / units,   q_i = 0 (not held)  or  min_units <= q_i <= max_units,
    sum_i q_i = units,   k_min <= number of held assets <= k_max.
In the brief's notation w_i = lower_i + step * sum_b 2^b q_{i,b}: step = 1/units and the
binary expansion of q_i is the quantum encoding's business (``fairbench.quantum.encoding``).

Everything here is exact: ``grid_size`` counts grid points with integer arithmetic,
``sample_grid`` draws them uniformly without rejection, ``enumerate_grid`` lists them.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..baselines import random_k_subsets

POLICIES = ("equal", "benchmark", "benchmark_capped")


def _cap_and_redistribute(W: np.ndarray, cap: float) -> np.ndarray:
    """Rows of proportional weights -> the same rows with no entry above ``cap``, excess
    spread proportionally over the uncapped held entries (water-filling)."""
    W = W.copy()
    held = W > 0
    if ((held.sum(axis=1) * cap) < 1.0 - 1e-12).any():
        raise ValueError(f"a cap of {cap:g} cannot hold with so few holdings")
    capped = np.zeros_like(held)
    for _ in range(W.shape[1]):
        over = (W > cap + 1e-12) & ~capped
        if not over.any():
            break
        capped |= over
        free = held & ~capped
        room = 1.0 - cap * capped.sum(axis=1)
        base = np.where(free, W, 0.0).sum(axis=1)
        scale = np.divide(room, base, out=np.zeros_like(room), where=base > 0)
        W = np.where(capped, cap, np.where(free, W * scale[:, None], 0.0))
    return W


def subset_weights(X: np.ndarray, policy: str = "equal", benchmark: np.ndarray | None = None,
                   cap: float | None = None) -> np.ndarray:
    """(m, n) selections -> (m, n) weights under a weighting policy. A selection with no
    holdings, or with zero total benchmark weight under a benchmark policy, raises ValueError."""
    X = np.atleast_2d(np.asarray(X)).astype(float)
    if policy == "equal":
        raw = X
    elif policy in ("benchmark", "benchmark_capped"):
        if benchmark is None:
            raise ValueError(f'policy "{policy}" needs benchmark weights')
        b = np.asarray(benchmark, dtype=float)
        if b.shape != (X.shape[1],) or (b < 0).any():
            raise ValueError("benchmark must be a non-negative (n,) vector")
        raw = X * b[None, :]
    else:
        raise ValueError(f"unknown weighting policy {policy!r}; choose from {POLICIES}")
    tot = raw.sum(axis=1, keepdims=True)
    if (tot <= 0).any():
        raise ValueError("a selection has no weight under this policy")
    W = raw / tot
    if policy == "benchmark_capped":
        if cap is None or not 0 < cap <= 1:
            raise ValueError('policy "benchmark_capped" needs a cap in (0, 1]')
        W = _cap_and_redistribute(W, float(cap))
    return W


@dataclass(frozen=True)
class WeightGrid:
    """The Stage 2 feasible lattice before any mandate rule (see module docstring).
    ``allowed``: indices that may be held (None = all); the others are fixed at 0."""
    n: int
    units: int
    min_units: int = 1
    max_units: int | None = None
    k_min: int = 1
    k_max: int | None = None
    allowed: tuple[int, ...] | None = None

    def __post_init__(self):
        hi = self.units if self.max_units is None else self.max_units
        km = self.n_allowed if self.k_max is None else self.k_max
        object.__setattr__(self, "max_units", int(hi))
        object.__setattr__(self, "k_max", int(km))
        if self.allowed is not None:
            object.__setattr__(self, "allowed", tuple(int(i) for i in self.allowed))
        if not (self.units >= 1 and 1 <= self.min_units <= self.max_units):
            raise ValueError(f"need units >= 1 and 1 <= min_units <= max_units; got {self}")
        if not 1 <= self.k_min <= self.k_max <= self.n_allowed:
            raise ValueError(f"need 1 <= k_min <= k_max <= number of allowed assets; got {self}")

    @property
    def n_allowed(self) -> int:
        return self.n if self.allowed is None else len(self.allowed)

    @property
    def allowed_indices(self) -> np.ndarray:
        return np.arange(self.n) if self.allowed is None else np.asarray(self.allowed, dtype=int)

    @property
    def step(self) -> float:
        return 1.0 / self.units

    def weights(self, Q: np.ndarray) -> np.ndarray:
        return np.asarray(Q, dtype=float) / self.units

    def contains(self, Q: np.ndarray) -> np.ndarray:
        """(m,) True where the integer rows are grid points."""
        Q = np.atleast_2d(np.asarray(Q)).astype(np.int64)
        held = Q > 0
        cnt = held.sum(axis=1)
        ok = (Q.sum(axis=1) == self.units) & (cnt >= self.k_min) & (cnt <= self.k_max)
        ok &= np.all(~held | ((Q >= self.min_units) & (Q <= self.max_units)), axis=1)
        if self.allowed is not None:
            banned = np.ones(self.n, dtype=bool)
            banned[self.allowed_indices] = False
            ok &= ~held[:, banned].any(axis=1)
        return ok


def composition_counts(total: int, parts: int, lo: int, hi: int) -> list[list[int]]:
    """c[j][s] = number of ways to write s as an ORDERED sum of j integers in [lo, hi],
    for j = 0..parts and s = 0..total. Exact integers."""
    c = [[0] * (total + 1) for _ in range(parts + 1)]
    c[0][0] = 1
    for j in range(1, parts + 1):
        prev, row, window = c[j - 1], c[j], 0
        for s in range(total + 1):  # window = sum of prev[s - hi .. s - lo]
            if s - lo >= 0:
                window += prev[s - lo]
            if s - hi - 1 >= 0:
                window -= prev[s - hi - 1]
            row[s] = window
    return c


def grid_counts(grid: WeightGrid) -> dict[int, int]:
    """k -> number of grid points with exactly k holdings (exact)."""
    c = composition_counts(grid.units, grid.k_max, grid.min_units, grid.max_units)
    return {k: math.comb(grid.n_allowed, k) * c[k][grid.units] for k in range(grid.k_min, grid.k_max + 1)}


def grid_size(grid: WeightGrid) -> int:
    """Number of grid points (exact integer)."""
    return sum(grid_counts(grid).values())


def _row_counts_float(total: int, parts: int, lo: int, hi: int) -> list[np.ndarray]:
    """composition_counts with each row scaled to max 1 (floats). Sampling only needs the
    ratios inside one row, so the scale is irrelevant and nothing overflows."""
    rows = [np.zeros(total + 1)]
    rows[0][0] = 1.0
    for _ in range(parts):
        cs = np.concatenate([[0.0], np.cumsum(rows[-1])])
        s = np.arange(total + 1)
        row = cs[np.clip(s - lo + 1, 0, total + 1)] - cs[np.clip(s - hi, 0, total + 1)]
        row = np.maximum(row, 0.0)
        top = row.max()
        rows.append(row / top if top > 0 else row)
    return rows


def sample_compositions(total: int, parts: int, lo: int, hi: int, m: int,
                        rng: np.random.Generator) -> np.ndarray:
    """(m, parts) ordered compositions of ``total`` with parts in [lo, hi], uniform over
    all of them. Sequential: the first part is a with probability c[j-1][s-a] / c[j][s]."""
    rows = _row_counts_float(total, parts, lo, hi)
    if rows[parts][total] <= 0:
        raise ValueError(f"no composition of {total} into {parts} parts in [{lo}, {hi}]")
    out = np.empty((m, parts), dtype=np.int64)
    left = np.full(m, total, dtype=np.int64)
    choices = np.arange(lo, hi + 1)
    for j in range(parts, 0, -1):
        idx = left[:, None] - choices[None, :]
        p = np.where((idx >= 0) & (idx <= total), rows[j - 1][np.clip(idx, 0, total)], 0.0)
        cum = np.cumsum(p, axis=1)
        pick = (cum < (rng.random(m) * cum[:, -1])[:, None]).sum(axis=1)
        out[:, parts - j] = choices[np.minimum(pick, choices.size - 1)]
        left -= out[:, parts - j]
    assert not left.any()
    return out


def sample_grid(grid: WeightGrid, m: int, seed: int | None = None) -> np.ndarray:
    """(m, n) integer unit vectors, uniform over all grid points. Exact, no rejection:
    the number of holdings k is drawn with probability proportional to the number of grid
    points that have k holdings, then a uniform k-subset of the allowed assets, then a
    uniform composition of the units over it."""
    rng = np.random.default_rng(seed)
    counts = grid_counts(grid)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("the weight grid is empty")
    ks = np.asarray(sorted(counts))
    pk = np.asarray([float(Fraction(counts[int(k)], total)) for k in ks])
    drawn = rng.choice(ks, size=m, p=pk / pk.sum())
    Q = np.zeros((m, grid.n), dtype=np.int64)
    allowed = grid.allowed_indices
    for k in np.unique(drawn):
        rows = np.flatnonzero(drawn == k)
        X = random_k_subsets(allowed.size, int(k), rows.size, int(rng.integers(2**31)))
        parts = sample_compositions(grid.units, int(k), grid.min_units, grid.max_units, rows.size, rng)
        block = np.zeros((rows.size, allowed.size), dtype=np.int64)
        block[X.astype(bool)] = parts.ravel()  # row-major: parts go to held assets in index order
        Q[np.ix_(rows, allowed)] = block
    return Q


def _compositions(total: int, parts: int, lo: int, hi: int):
    if parts == 0:
        if total == 0:
            yield ()
        return
    for a in range(max(lo, total - hi * (parts - 1)), min(hi, total - lo * (parts - 1)) + 1):
        for rest in _compositions(total - a, parts - 1, lo, hi):
            yield (a,) + rest


def enumerate_grid(grid: WeightGrid, limit: int = 2_000_000) -> np.ndarray:
    """All grid points as (S, n) integer unit vectors. For small instances only: raises
    ValueError when the grid has more than ``limit`` points."""
    size = grid_size(grid)
    if size > limit:
        raise ValueError(f"the grid has {size} points, more than limit={limit}")
    Q = np.zeros((size, grid.n), dtype=np.int64)
    allowed, r = grid.allowed_indices, 0
    for k in range(grid.k_min, grid.k_max + 1):
        comps = np.asarray(list(_compositions(grid.units, k, grid.min_units, grid.max_units)), dtype=np.int64)
        if comps.size == 0:
            continue
        for support in itertools.combinations(allowed.tolist(), k):
            Q[r:r + len(comps), list(support)] = comps
            r += len(comps)
    assert r == size
    return Q
