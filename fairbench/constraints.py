"""Portfolio selection constraints. All ESG/rule logic lives here.

Classes: Cardinality, SectorCap, Exclusion, MinESG, CarbonCap (the original five, which the
penalty Hamiltonian and the fault-tolerant oracle also encode) and CountBound, AvgBound,
MinGroups, VolatilityCap, TrackingErrorCap (checked by every sampler and by attribution;
no Hamiltonian or oracle encoding yet).

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


class CountBound(_Base):
    """min_count <= number of held assets among ``indices`` <= max_count (None = no cap).
    Covers minimum shares ("at least 60% in companies with science-based targets") and
    caps on any group of assets (a country, a flag, a theme)."""

    def __init__(self, indices: Sequence[int], min_count: int = 0, max_count: int | None = None,
                 label: str = ""):
        self.indices = np.asarray(list(indices), dtype=int)
        self.min_count = int(min_count)
        self.max_count = None if max_count is None else int(max_count)
        self.label = label

    def check_batch(self, X, u):
        cnt = _batch(X)[:, self.indices].sum(axis=1)
        ok = cnt >= self.min_count
        return ok if self.max_count is None else ok & (cnt <= self.max_count)


class AvgBound(_AvgConstraint):
    """lower <= average of a numeric attribute over held assets <= upper (None = open).
    ``attribute`` is "esg", "carbon" or a key of ``u.attributes``. Empty -> infeasible."""

    def __init__(self, attribute: str, lower: float | None = None, upper: float | None = None):
        if lower is None and upper is None:
            raise ValueError("AvgBound needs a lower or an upper bound")
        self.attribute = attribute
        self.lower = None if lower is None else float(lower)
        self.upper = None if upper is None else float(upper)

    def check_batch(self, X, u):
        values = u.numeric(self.attribute)
        if values is None:
            raise KeyError(f"universe has no numeric attribute {self.attribute!r}")
        return self._avg_ok(X, values, lambda a: ((a >= self.lower) if self.lower is not None else True)
                            & ((a <= self.upper) if self.upper is not None else True))


class MinGroups(_Base):
    """Held assets span at least ``min_groups`` distinct labels of ``category``
    ("sector" or a key of ``u.categories``), e.g. "invested across at least 8 sectors"."""

    cost = 1  # evaluated after the plain counting rules (see ConstraintSet.check_batch)

    def __init__(self, category: str, min_groups: int):
        self.category = category
        self.min_groups = int(min_groups)

    def check_batch(self, X, u):
        lab = u.labels(self.category)
        if lab is None:
            raise KeyError(f"universe has no category {self.category!r}")
        X = _batch(X)
        onehot = (lab[:, None] == np.unique(lab)[None, :]).astype(np.int64)  # (n, groups)
        return ((X @ onehot) > 0).sum(axis=1) >= self.min_groups


class _RiskCap(_Base):
    """Ex-ante annualised risk of the equally weighted selection from ``u.cov``."""

    cost = 2  # quadratic in n: evaluated last, on the rows every other rule accepts

    def _active(self, W: np.ndarray, u: Universe) -> np.ndarray:
        return W

    def risk(self, X, u) -> np.ndarray:
        """(m,) risk of each selection; nan for an empty one."""
        X = _batch(X).astype(float)
        cnt = X.sum(axis=1, keepdims=True)
        D = self._active(X / np.maximum(cnt, 1.0), u)
        var = np.einsum("ij,jk,ik->i", D, np.asarray(u.cov, dtype=float), D)
        return np.where(cnt[:, 0] > 0, np.sqrt(np.maximum(var, 0.0)), np.nan)

    def check_batch(self, X, u):
        r = self.risk(X, u)
        return ~np.isnan(r) & (r <= self.max_risk)


class VolatilityCap(_RiskCap):
    """sqrt(w' cov w) <= max_vol for equal weights w. Empty -> infeasible."""

    def __init__(self, max_vol: float):
        self.max_risk = float(max_vol)


class TrackingErrorCap(_RiskCap):
    """sqrt((w - b)' cov (w - b)) <= max_te, with b the benchmark weights (default: equal
    weight over the whole universe). Empty -> infeasible."""

    def __init__(self, max_te: float, benchmark: Sequence[float] | None = None):
        self.max_risk = float(max_te)
        self.benchmark = None if benchmark is None else np.asarray(benchmark, dtype=float)

    def _active(self, W, u):
        b = np.full(u.n, 1.0 / u.n) if self.benchmark is None else self.benchmark
        return W - b[None, :]


class ConstraintSet:
    def __init__(self, constraints: Sequence[Constraint]):
        self.constraints = list(constraints)

    def check(self, x: np.ndarray, u: Universe) -> bool:
        return all(c.check(x, u) for c in self.constraints)

    def check_batch(self, X: np.ndarray, u: Universe) -> np.ndarray:
        """AND of all constraints. Cheap ones run first and each later one only sees the
        rows still feasible, so the costly risk checks touch few rows."""
        X = np.atleast_2d(X)
        ok = np.ones(X.shape[0], dtype=bool)
        for c in sorted(self.constraints, key=lambda c: getattr(c, "cost", 0)):
            idx = np.flatnonzero(ok)
            if idx.size == 0:
                break
            ok[idx] = c.check_batch(X[idx], u)
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

    def allowed_indices(self, n: int) -> np.ndarray:
        """Assets no Exclusion constraint forbids, ascending. Every feasible selection
        lives on these, so samplers can draw from them instead of rejecting."""
        ok = np.ones(n, dtype=bool)
        for c in self.constraints:
            if isinstance(c, Exclusion):
                ok[c.indices] = False
        return np.flatnonzero(ok)
