"""Portfolio selection constraints. All ESG/rule logic lives here.

A selection is a binary vector x (length n); X is a (m, n) batch.
Average-based constraints (MinESG, CarbonCap) use the equal-weight average
over held assets. An EMPTY selection (no assets held) is INFEASIBLE for them.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .data import Universe


@runtime_checkable
class Constraint(Protocol):
    def check(self, x: np.ndarray, u: Universe) -> bool: ...
    def check_batch(self, X: np.ndarray, u: Universe) -> np.ndarray: ...


def _batch(X: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.asarray(X)).astype(np.int64, copy=False)


class _Base:
    def check(self, x: np.ndarray, u: Universe) -> bool:
        return bool(self.check_batch(np.asarray(x)[None, :], u)[0])

    def check_batch(self, X: np.ndarray, u: Universe) -> np.ndarray:
        raise NotImplementedError


class Cardinality(_Base):
    def __init__(self, k: int):
        self.k = int(k)

    def check_batch(self, X, u):
        return _batch(X).sum(axis=1) == self.k


class SectorCap(_Base):
    def __init__(self, sector: str, max_count: int):
        self.sector = sector
        self.max_count = int(max_count)

    def check_batch(self, X, u):
        mask = np.array([s == self.sector for s in u.sector], dtype=np.int64)
        return _batch(X) @ mask <= self.max_count


class Exclusion(_Base):
    def __init__(self, indices: Sequence[int]):
        self.indices = np.asarray(list(indices), dtype=int)

    def check_batch(self, X, u):
        X = _batch(X)
        if self.indices.size == 0:
            return np.ones(X.shape[0], dtype=bool)
        return X[:, self.indices].sum(axis=1) == 0


class _AvgConstraint(_Base):
    def _avg_ok(self, X, values, cmp):
        X = _batch(X)
        cnt = X.sum(axis=1)
        tot = X @ values
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        return (cnt > 0) & cmp(avg)


class MinESG(_AvgConstraint):
    """Average ESG score of held assets >= min_avg_score. Empty -> infeasible."""

    def __init__(self, min_avg_score: float):
        self.min_avg_score = float(min_avg_score)

    def check_batch(self, X, u):
        return self._avg_ok(X, u.esg_score, lambda a: a >= self.min_avg_score)


class CarbonCap(_AvgConstraint):
    """Average carbon of held assets <= max_avg. Empty -> infeasible."""

    def __init__(self, max_avg: float):
        self.max_avg = float(max_avg)

    def check_batch(self, X, u):
        return self._avg_ok(X, u.carbon, lambda a: a <= self.max_avg)


class ConstraintSet:
    def __init__(self, constraints: Sequence[Constraint]):
        self.constraints = list(constraints)

    def check(self, x: np.ndarray, u: Universe) -> bool:
        return all(c.check(x, u) for c in self.constraints)

    def check_batch(self, X: np.ndarray, u: Universe) -> np.ndarray:
        X = np.atleast_2d(X)
        ok = np.ones(X.shape[0], dtype=bool)
        for c in self.constraints:
            ok &= c.check_batch(X, u)
        return ok

    @property
    def cardinality(self) -> int | None:
        """k of the (first) Cardinality constraint, or None."""
        for c in self.constraints:
            if isinstance(c, Cardinality):
                return c.k
        return None

    def non_cardinality(self) -> list[Constraint]:
        return [c for c in self.constraints if not isinstance(c, Cardinality)]
