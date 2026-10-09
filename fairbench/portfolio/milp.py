"""Mixed-integer baseline for the weight grid: feasibility, exact extremes, and which rule binds.

A solver answers three questions that sampling cannot answer exactly:
    is the rule set feasible at all            ``feasible_extremes(...)["feasible"]``
    what are the lowest and highest returns    the exact end points of the rule-conditioned
    any feasible portfolio could have had      return range over one period
    which rule makes an infeasible set so      ``diagnose_infeasibility``

It is NOT a sampler. The order in which a solver finds solutions is not uniform over the
feasible set, so its solutions are never used as a reference distribution.

Model (``WeightGrid`` units): integer q_i in [0, max_units], binary y_i = 1 iff held,
min_units * y_i <= q_i <= max_units * y_i, sum q = units. Linear rules only: position
bounds, holdings range, group weights, weighted averages, and the selection rules that are
linear in y (exclusions, sector and group counts, equal-weight averages). Risk caps are
quadratic and raise TypeError; they belong to a quadratic solver or to the samplers.
Uses ``scipy.optimize.milp`` (HiGHS), which the project already depends on.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ..constraints import (AvgBound, CarbonCap, Cardinality, ConstraintSet, CountBound, Exclusion, HoldingsRange,
                           MinESG, PositionBound, SectorCap, WeightedAverage, WeightedRisk, WeightRuleSet,
                           WeightSum)
from ..data import Universe
from .weights import WeightGrid


class _Model:
    def __init__(self, grid: WeightGrid):
        self.g, self.n = grid, grid.n
        self.rows: list[tuple[np.ndarray, float, float, str]] = []
        self.q_hi = np.zeros(self.n)
        self.q_hi[grid.allowed_indices] = grid.max_units
        self.y_hi = np.zeros(self.n)
        self.y_hi[grid.allowed_indices] = 1.0

    def add(self, q=None, y=None, lo=-np.inf, hi=np.inf, name=""):
        row = np.zeros(2 * self.n)
        if q is not None:
            row[:self.n] = q
        if y is not None:
            row[self.n:] = y
        self.rows.append((row, lo, hi, name))


def _mask(n: int, idx) -> np.ndarray:
    m = np.zeros(n)
    m[np.asarray(idx, dtype=int)] = 1.0
    return m


def _build(u: Universe, grid: WeightGrid, rules: WeightRuleSet | None, skip: str | None = None) -> tuple[_Model, list[str]]:
    m, n, M = _Model(grid), grid.n, grid.units
    names: list[str] = []
    eye = np.eye(n)
    m.add(q=np.ones(n), lo=M, hi=M, name="budget")
    for i in grid.allowed_indices:
        m.add(q=eye[i], y=-grid.max_units * eye[i], hi=0, name="link")
        m.add(q=eye[i], y=-grid.min_units * eye[i], lo=0, name="link")
    m.add(y=np.ones(n), lo=grid.k_min, hi=grid.k_max, name="holdings")

    def rule(name: str) -> bool:
        names.append(name)
        return name != skip

    for i, r in enumerate([] if rules is None else rules.rules):
        name = f"{i}:{getattr(r, 'label', '') or type(r).__name__}"
        if isinstance(r, WeightedRisk):
            raise TypeError(f"{name} is quadratic in the weights; not a linear rule")
        if not rule(name):
            continue
        if isinstance(r, PositionBound):
            up, low = math.floor(r.upper * M + 1e-9), math.ceil(r.lower * M - 1e-9)
            for j in grid.allowed_indices:
                m.add(q=eye[j], y=-up * eye[j], hi=0, name=name)
                m.add(q=eye[j], y=-low * eye[j], lo=0, name=name)
        elif isinstance(r, HoldingsRange):
            m.add(y=np.ones(n), lo=r.k_min, hi=r.k_max, name=name)
        elif isinstance(r, WeightSum):
            m.add(q=_mask(n, r.indices), lo=-np.inf if r.lower is None else r.lower * M - 1e-9,
                  hi=np.inf if r.upper is None else r.upper * M + 1e-9, name=name)
        elif isinstance(r, WeightedAverage):
            v = r._values(u)
            known = ~np.isnan(v)
            vk = np.where(known, v, 0.0)
            if r.missing == "ineligible":
                m.y_hi[~known] = 0.0
                m.q_hi[~known] = 0.0
                m.add(q=vk, lo=-np.inf if r.lower is None else r.lower * M - 1e-9,
                      hi=np.inf if r.upper is None else r.upper * M + 1e-9, name=name)
            else:  # average over covered weight: sum_known (v - bound) q <=/>= 0, some covered weight
                m.add(q=known.astype(float), lo=1, name=name)
                if r.upper is not None:
                    m.add(q=np.where(known, v - r.upper, 0.0), hi=1e-9, name=name)
                if r.lower is not None:
                    m.add(q=np.where(known, v - r.lower, 0.0), lo=-1e-9, name=name)
        else:
            raise TypeError(f"no linear form for {name}")

    for i, c in enumerate([] if rules is None or rules.support_rules is None else rules.support_rules.constraints):
        name = f"support {i}:{type(c).__name__}"
        if not rule(name):
            continue
        if isinstance(c, Cardinality):
            m.add(y=np.ones(n), lo=c.k, hi=c.k, name=name)
        elif isinstance(c, Exclusion):
            m.y_hi[c.indices] = 0.0
            m.q_hi[c.indices] = 0.0
        elif isinstance(c, SectorCap):
            m.add(y=np.array([s == c.sector for s in u.sector], dtype=float), hi=c.max_count, name=name)
        elif isinstance(c, CountBound):
            m.add(y=_mask(n, c.indices), lo=c.min_count, hi=np.inf if c.max_count is None else c.max_count, name=name)
        elif isinstance(c, MinESG):  # equal-weight average over held names: sum (v - t) y >= 0
            m.add(y=np.asarray(u.esg_score, dtype=float) - c.min_avg_score, lo=0, name=name)
        elif isinstance(c, CarbonCap):
            m.add(y=np.asarray(u.carbon, dtype=float) - c.max_avg, hi=0, name=name)
        elif isinstance(c, AvgBound):
            v = u.numeric(c.attribute)
            if c.lower is not None:
                m.add(y=v - c.lower, lo=0, name=name)
            if c.upper is not None:
                m.add(y=v - c.upper, hi=0, name=name)
        else:
            raise TypeError(f"no linear form for {name}")
    return m, names


def _solve(m: _Model, objective: np.ndarray):
    A = np.vstack([r[0] for r in m.rows])
    cons = LinearConstraint(A, np.array([r[1] for r in m.rows]), np.array([r[2] for r in m.rows]))
    bounds = Bounds(np.zeros(2 * m.n), np.concatenate([m.q_hi, m.y_hi]))
    return milp(c=np.concatenate([objective, np.zeros(m.n)]), constraints=cons, integrality=np.ones(2 * m.n),
                bounds=bounds)


def feasible_extremes(u: Universe, grid: WeightGrid, rules: WeightRuleSet | None,
                      asset_returns: np.ndarray | None = None) -> dict:
    """Is the rule set feasible, and what are the exact lowest and highest one-period returns
    over all feasible grid portfolios? Returns {"feasible", "min_return", "max_return",
    "argmin", "argmax"} (weights; None when infeasible or when no returns are given)."""
    m, _ = _build(u, grid, rules)
    r = np.zeros(grid.n) if asset_returns is None else np.asarray(asset_returns, dtype=float)
    lo = _solve(m, r / grid.units)
    if not lo.success:
        return dict(feasible=False, min_return=None, max_return=None, argmin=None, argmax=None, message=lo.message)
    out = dict(feasible=True, min_return=None, max_return=None, argmin=None, argmax=None, message=lo.message)
    if asset_returns is not None:
        hi = _solve(m, -r / grid.units)
        w_lo, w_hi = np.rint(lo.x[:grid.n]) / grid.units, np.rint(hi.x[:grid.n]) / grid.units
        out.update(min_return=float(w_lo @ r), max_return=float(w_hi @ r), argmin=w_lo, argmax=w_hi)
    return out


def diagnose_infeasibility(u: Universe, grid: WeightGrid, rules: WeightRuleSet) -> list[str]:
    """For an infeasible rule set: the rules whose removal, one at a time, makes it feasible.
    Empty if the set is feasible, or if no single rule is responsible."""
    m, names = _build(u, grid, rules)
    if _solve(m, np.zeros(grid.n)).success:
        return []
    return [name for name in names if _solve(_build(u, grid, rules, skip=name)[0], np.zeros(grid.n)).success]
