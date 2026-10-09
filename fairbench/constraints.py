"""Portfolio selection constraints. All ESG/rule logic lives here.

Classes: Cardinality, SectorCap, Exclusion, MinESG, CarbonCap (the original five, which the
penalty Hamiltonian and the fault-tolerant oracle also encode) and CountBound, AvgBound,
LinearThreshold, MinGroups, VolatilityCap, TrackingErrorCap (checked by every sampler and by
attribution; no Hamiltonian encoding; ``ft.oracle`` also encodes CountBound, AvgBound and
LinearThreshold).

A selection is a binary vector x (length n); X is a (m, n) batch.
Average-based constraints (MinESG, CarbonCap) use the equal-weight average
over held assets. An EMPTY selection (no assets held) is INFEASIBLE for them.

Weight rules (HoldingsRange, PositionBound, WeightSum, WeightedAverage, WeightedRisk,
TurnoverCap, collected in a WeightRuleSet) are checked on a weight vector w instead; see the end of
this file. They have no quantum encoding here (``quantum.encoding`` builds one).
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


class LinearThreshold(_Base):
    """sum_i weights_i x_i  (direction)  threshold, direction ">=" or "<" (strict); float
    weights of any sign. Any linear rule, e.g. "equal-weight buy-and-hold total return below
    s": (1/k) sum_i x_i g_i - 1 < s  <=>  LinearThreshold(g, k * (1 + s), "<"), with
    g_i = prod_t (1 + r_it) the asset's gross return over the window. An empty selection
    has sum 0 (no special case)."""

    DIRECTIONS = (">=", "<")

    def __init__(self, weights: Sequence[float], threshold: float, direction: str = ">=",
                 label: str = ""):
        if direction not in self.DIRECTIONS:
            raise ValueError(f"direction must be one of {self.DIRECTIONS}, got {direction!r}")
        self.weights = np.asarray(weights, dtype=float).ravel()
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights must be finite")
        self.threshold = float(threshold)
        self.direction = direction
        self.label = label

    def values(self, X) -> np.ndarray:
        """(m,) sum_i weights_i x_i of each row."""
        X = _batch(X)
        if X.shape[1] != self.weights.size:
            raise ValueError(f"LinearThreshold has {self.weights.size} weights, X has {X.shape[1]} columns")
        return X @ self.weights

    def check_batch(self, X, u):
        v = self.values(X)
        return v >= self.threshold if self.direction == ">=" else v < self.threshold


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


# ------------------------------------------------------------------ weights
# Rules on a weight vector w (length n, w >= 0, sum 1); W is a (m, n) batch. An asset is
# held iff w_i > WEIGHT_TOL. These are the forms a mandate states ("no position above 5%",
# "at most 25% in one industry", "tracking error below 2%"); the classes above are their
# equal-weight count forms. Risk rules are QUADRATIC in w and are never linearised.

WEIGHT_TOL = 1e-9


def _wbatch(W: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.asarray(W, dtype=float))


class _WBase:
    def check_weight(self, w: np.ndarray, u: Universe) -> bool:
        return bool(self.check_weights(np.asarray(w, dtype=float)[None, :], u)[0])

    def check_weights(self, W: np.ndarray, u: Universe) -> np.ndarray:
        raise NotImplementedError


def _between(v: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
    ok = np.ones(v.shape, dtype=bool)
    if lo is not None:
        ok &= v >= lo - WEIGHT_TOL
    if hi is not None:
        ok &= v <= hi + WEIGHT_TOL
    return ok


class HoldingsRange(_WBase):
    """k_min <= number of held assets <= k_max."""

    def __init__(self, k_min: int, k_max: int | None = None):
        self.k_min = int(k_min)
        self.k_max = self.k_min if k_max is None else int(k_max)
        if not 0 <= self.k_min <= self.k_max:
            raise ValueError(f"need 0 <= k_min <= k_max; got {self.k_min}, {self.k_max}")

    def check_weights(self, W, u):
        cnt = (_wbatch(W) > WEIGHT_TOL).sum(axis=1)
        return (cnt >= self.k_min) & (cnt <= self.k_max)


class PositionBound(_WBase):
    """lower <= w_i <= upper for every HELD asset. ``lower`` is a minimum position size:
    an asset that is not held (w_i = 0) does not breach it."""

    def __init__(self, lower: float = 0.0, upper: float = 1.0):
        if not 0.0 <= lower <= upper <= 1.0:
            raise ValueError(f"need 0 <= lower <= upper <= 1; got {lower}, {upper}")
        self.lower, self.upper = float(lower), float(upper)

    def check_weights(self, W, u):
        W = _wbatch(W)
        held = W > WEIGHT_TOL
        return np.all(~held | ((W >= self.lower - WEIGHT_TOL) & (W <= self.upper + WEIGHT_TOL)), axis=1)


class WeightSum(_WBase):
    """lower <= total weight of the assets in ``indices`` <= upper (None = open): a sector,
    country or flagged-group weight limit."""

    def __init__(self, indices: Sequence[int], lower: float | None = None, upper: float | None = None,
                 label: str = ""):
        if lower is None and upper is None:
            raise ValueError("WeightSum needs a lower or an upper bound")
        self.indices = np.asarray(list(indices), dtype=int)
        self.lower = None if lower is None else float(lower)
        self.upper = None if upper is None else float(upper)
        self.label = label

    def check_weights(self, W, u):
        return _between(_wbatch(W)[:, self.indices].sum(axis=1), self.lower, self.upper)


class WeightedAverage(_WBase):
    """lower <= sum_i w_i v_i <= upper, e.g. a weighted ESG floor or carbon-intensity cap.
    ``values`` is an (n,) array or the name of a numeric field of the universe.
    A missing value (NaN) is never read as zero: with ``missing="ineligible"`` (default) a
    portfolio holding an asset with no value fails; with ``missing="renormalise"`` the
    average is taken over the covered weight, and a portfolio with no covered weight fails."""

    MISSING = ("ineligible", "renormalise")

    def __init__(self, values, lower: float | None = None, upper: float | None = None,
                 missing: str = "ineligible", label: str = ""):
        if lower is None and upper is None:
            raise ValueError("WeightedAverage needs a lower or an upper bound")
        if missing not in self.MISSING:
            raise ValueError(f"missing must be one of {self.MISSING}, got {missing!r}")
        self.values = values
        self.lower = None if lower is None else float(lower)
        self.upper = None if upper is None else float(upper)
        self.missing, self.label = missing, label

    def _values(self, u: Universe) -> np.ndarray:
        if isinstance(self.values, str):
            v = u.numeric(self.values)
            if v is None:
                raise KeyError(f"universe has no numeric attribute {self.values!r}")
            return v
        return np.asarray(self.values, dtype=float)

    def average(self, W, u) -> np.ndarray:
        """(m,) weighted average; NaN where it cannot be evaluated under ``missing``."""
        W, v = _wbatch(W), self._values(u)
        known = ~np.isnan(v)
        tot = W[:, known] @ v[known]
        uncovered = W[:, ~known].sum(axis=1)
        if self.missing == "ineligible":
            return np.where(uncovered > WEIGHT_TOL, np.nan, tot)
        covered = 1.0 - uncovered
        return np.where(covered > WEIGHT_TOL, tot / np.where(covered > WEIGHT_TOL, covered, 1.0), np.nan)

    def check_weights(self, W, u):
        a = self.average(W, u)
        ok = ~np.isnan(a)
        ok[ok] = _between(a[ok], self.lower, self.upper)
        return ok


def active_risk(W: np.ndarray, cov: np.ndarray, benchmark: np.ndarray | None = None) -> np.ndarray:
    """(m,) sqrt((w - b)' cov (w - b)); b = None gives absolute volatility sqrt(w' cov w)."""
    D = _wbatch(W) if benchmark is None else _wbatch(W) - np.asarray(benchmark, dtype=float)[None, :]
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, np.asarray(cov, dtype=float), D), 0.0))


class WeightedRisk(_WBase):
    """sqrt((w - b)' cov (w - b)) <= max_risk for the actual weights; ``benchmark`` None
    means absolute volatility. Quadratic in w."""

    cost = 2

    def __init__(self, max_risk: float, benchmark: Sequence[float] | None = None, label: str = ""):
        self.max_risk = float(max_risk)
        self.benchmark = None if benchmark is None else np.asarray(benchmark, dtype=float)
        self.label = label

    def risk(self, W, u) -> np.ndarray:
        return active_risk(W, u.cov, self.benchmark)

    def check_weights(self, W, u):
        return self.risk(W, u) <= self.max_risk + WEIGHT_TOL


class TurnoverCap(_WBase):
    """One-way turnover 0.5 * sum_i |w_i - previous_i| <= max_turnover against the previous
    weights (linear after splitting each change into a buy and a sell)."""

    def __init__(self, previous: Sequence[float], max_turnover: float, label: str = ""):
        self.previous = np.asarray(previous, dtype=float)
        self.max_turnover = float(max_turnover)
        self.label = label

    def turnover(self, W, u=None) -> np.ndarray:
        return 0.5 * np.abs(_wbatch(W) - self.previous[None, :]).sum(axis=1)

    def check_weights(self, W, u):
        return self.turnover(W) <= self.max_turnover + WEIGHT_TOL


class WeightRuleSet:
    """AND of weight rules, plus optional selection rules (a ``ConstraintSet`` such as
    exclusions or minimum sector counts) applied to the set of held assets."""

    def __init__(self, rules: Sequence[_WBase] = (), support_rules: ConstraintSet | None = None):
        self.rules = list(rules)
        self.support_rules = support_rules

    def check_weight(self, w: np.ndarray, u: Universe) -> bool:
        return bool(self.check_weights(np.asarray(w, dtype=float)[None, :], u)[0])

    def check_weights(self, W: np.ndarray, u: Universe) -> np.ndarray:
        W = _wbatch(W)
        ok = np.all(W >= -WEIGHT_TOL, axis=1) & (np.abs(W.sum(axis=1) - 1.0) <= 1e-6)
        if self.support_rules is not None and ok.any():
            idx = np.flatnonzero(ok)
            ok[idx] = self.support_rules.check_batch((W[idx] > WEIGHT_TOL).astype(np.uint8), u)
        for r in sorted(self.rules, key=lambda r: getattr(r, "cost", 0)):
            idx = np.flatnonzero(ok)
            if idx.size == 0:
                break
            ok[idx] = r.check_weights(W[idx], u)
        return ok

    def violations(self, W: np.ndarray, u: Universe) -> dict[str, int]:
        """Rows breaking each rule, counted independently (label or class name -> count)."""
        W = _wbatch(W)
        out: dict[str, int] = {}
        if self.support_rules is not None:
            out["support_rules"] = int((~self.support_rules.check_batch((W > WEIGHT_TOL).astype(np.uint8), u)).sum())
        for i, r in enumerate(self.rules):
            name = f"{i}:{getattr(r, 'label', '') or type(r).__name__}"
            out[name] = int((~r.check_weights(W, u)).sum())
        return out
