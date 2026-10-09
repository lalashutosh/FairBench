"""MILP baseline: exact extremes and feasibility against brute-force enumeration."""
import numpy as np
import pytest

from fairbench.constraints import (ConstraintSet, CountBound, Exclusion, MinESG, PositionBound, SectorCap,
                                   WeightedAverage, WeightedRisk, WeightRuleSet, WeightSum)
from fairbench.data import synthetic_universe
from fairbench.portfolio.exact import exact_uniform_weights
from fairbench.portfolio.milp import diagnose_infeasibility, feasible_extremes
from fairbench.portfolio.weights import WeightGrid


@pytest.fixture
def case():
    u = synthetic_universe(7, 3, seed=6)
    grid = WeightGrid(n=7, units=8, min_units=1, max_units=4, k_min=2, k_max=4)
    r = np.random.default_rng(1).normal(0.01, 0.1, 7)
    return u, grid, r


@pytest.mark.parametrize("which", ["none", "weights", "support", "missing"])
def test_extremes_equal_brute_force(case, which):
    u, grid, r = case
    sector0 = np.flatnonzero(np.array(u.sector) == "S0")
    carbon = u.carbon.copy()
    rules = {
        "none": None,
        "weights": WeightRuleSet([WeightSum(sector0, upper=0.5), PositionBound(0.125, 0.375),
                                  WeightedAverage("esg", lower=float(np.median(u.esg_score)))]),
        "support": WeightRuleSet([WeightSum([1, 2], lower=0.25)], support_rules=ConstraintSet(
            [Exclusion([6]), SectorCap("S1", 1), CountBound([0, 1, 2], 1, 2), MinESG(float(np.percentile(u.esg_score, 30)))])),
        "missing": WeightRuleSet([WeightedAverage(np.where(np.arange(7) == 3, np.nan, carbon),
                                                  upper=float(np.median(carbon)), missing="renormalise")]),
    }[which]
    exact = exact_uniform_weights(u, grid, rules)
    ret = exact.returns(r)
    got = feasible_extremes(u, grid, rules, r)
    assert got["feasible"]
    assert got["min_return"] == pytest.approx(ret.min(), abs=1e-9)
    assert got["max_return"] == pytest.approx(ret.max(), abs=1e-9)
    for w in (got["argmin"], got["argmax"]):
        assert w.sum() == pytest.approx(1.0) and (rules is None or rules.check_weight(w, u))


def test_infeasible_set_is_diagnosed(case):
    u, grid, r = case
    rules = WeightRuleSet([WeightSum([0, 1], lower=0.75, label="needs_01"), WeightSum([0, 1], upper=0.25, label="caps_01"),
                           PositionBound(0, 0.5)])
    assert not feasible_extremes(u, grid, rules, r)["feasible"]
    assert diagnose_infeasibility(u, grid, rules) == ["0:needs_01", "1:caps_01"]
    assert diagnose_infeasibility(u, grid, WeightRuleSet([PositionBound(0, 0.5)])) == []


def test_feasibility_without_returns_and_quadratic_rule_is_refused(case):
    u, grid, _ = case
    out = feasible_extremes(u, grid, None)
    assert out["feasible"] and out["min_return"] is None
    with pytest.raises(TypeError, match="quadratic"):
        feasible_extremes(u, grid, WeightRuleSet([WeightedRisk(0.1)]))
