"""Stage 2 QUBO encoding: exactness, bijection, the quadratic objective, and the two samplers."""
import numpy as np
import pytest

from fairbench.constraints import (HoldingsRange, PositionBound, WeightedAverage, WeightRuleSet, WeightSum,
                                   active_risk)
from fairbench.data import synthetic_universe
from fairbench.portfolio.exact import compare_to_exact, exact_benchmark_aware, exact_uniform_weights
from fairbench.portfolio.weights import WeightGrid, enumerate_grid
from fairbench.quantum.encoding import (all_bitstrings, build_qubo, check_encoding, feasible_mask,
                                        grover_iterations, grover_sample, simulated_annealing)
from fairbench.quantum.hamiltonian import diag_values


def _assert_exact_bijection(model, feasible_units):
    rep = check_encoding(model, feasible_units)
    assert rep["n_zero_penalty"] == rep["n_feasible"] == rep["n_distinct_decoded"], rep
    assert rep["decoded_equals_feasible"] and rep["encode_roundtrip"], rep
    assert rep["min_penalty"] > -1e-9 and rep["min_violation_penalty"] >= 1 - 1e-9, rep
    return rep


@pytest.mark.parametrize("grid", [
    WeightGrid(n=4, units=6, min_units=1, max_units=4, k_min=2, k_max=3),   # R = 3 = 2^2 - 1: no slack
    WeightGrid(n=4, units=6, min_units=1, max_units=3, k_min=2, k_max=4),   # R = 2: position-cap slack
    WeightGrid(n=5, units=5, min_units=1, max_units=1, k_min=5, k_max=5),   # pure selection, B = 0
    WeightGrid(n=4, units=4, min_units=1, max_units=2, k_min=2, k_max=4),   # R = 1
    WeightGrid(n=5, units=6, min_units=2, max_units=5, k_min=2, k_max=2, allowed=(0, 2, 3, 4)),
])
def test_grid_encoding_is_exact_and_one_to_one(grid):
    model = build_qubo(grid)
    assert not model.unencoded
    _assert_exact_bijection(model, enumerate_grid(grid))


def test_no_slack_when_range_is_power_of_two_minus_one():
    free = build_qubo(WeightGrid(n=4, units=6, min_units=1, max_units=4, k_min=2, k_max=4))
    assert free.n_qubits == free.n_data_qubits == 4 * (1 + 2)
    slacked = build_qubo(WeightGrid(n=4, units=6, min_units=1, max_units=3, k_min=2, k_max=4))
    cap = next(t for t in slacked.terms if t["name"] == "position_cap")
    assert cap["n_slack"] > 0 and slacked.n_qubits == slacked.n_data_qubits + cap["n_slack"]


def test_holdings_term_only_when_it_binds():
    names = lambda g: [t["name"] for t in build_qubo(g).terms]
    assert "holdings" not in names(WeightGrid(n=4, units=6, min_units=1, max_units=4, k_min=2, k_max=4))
    assert "holdings" in names(WeightGrid(n=4, units=6, min_units=1, max_units=4, k_min=3, k_max=3))


def test_group_weight_rules_are_encoded_exactly():
    u = synthetic_universe(4, 2, seed=0)
    grid = WeightGrid(n=4, units=6, min_units=1, max_units=4, k_min=2, k_max=3)
    rules = WeightRuleSet([WeightSum([0, 1], upper=0.5, label="groupA"), WeightSum([2], lower=1 / 6, label="must2")])
    model = build_qubo(grid, rules)
    Q = enumerate_grid(grid)
    feas = Q[rules.check_weights(grid.weights(Q), u)]
    assert 0 < len(feas) < len(Q)
    _assert_exact_bijection(model, feas)
    assert [t["name"] for t in model.terms if "group" in t["name"] or "must" in t["name"]] == ["0:groupA", "1:must2"]


def test_unencodable_rules_are_listed_not_dropped():
    grid = WeightGrid(n=4, units=6, min_units=1, max_units=4, k_min=2, k_max=3)
    rules = WeightRuleSet([WeightedAverage("esg", lower=50.0, label="esg_floor"), PositionBound(0, 1.0),
                           HoldingsRange(1, 4)])
    model = build_qubo(grid, rules)
    assert model.unencoded == ["0:esg_floor"]
    with pytest.raises(ValueError, match="tighter than the grid"):
        build_qubo(grid, WeightRuleSet([PositionBound(0, 0.5)]))
    with pytest.raises(ValueError, match="tighter than the grid"):
        build_qubo(grid, WeightRuleSet([HoldingsRange(2, 2)]))


def test_objective_is_the_quadratic_tracking_error():
    u = synthetic_universe(4, 2, seed=3)
    grid = WeightGrid(n=4, units=6, min_units=1, max_units=4, k_min=2, k_max=3)
    b = np.array([0.4, 0.3, 0.2, 0.1])
    model = build_qubo(grid, cov=u.cov, benchmark=b, objective_scale=2.5)
    Z = all_bitstrings(model.n_qubits)
    te2 = active_risk(model.decode(Z), u.cov, b) ** 2
    assert model.objective(Z) == pytest.approx(2.5 * te2)          # every bitstring, feasible or not
    assert model.energy(Z) == pytest.approx(model.penalty(Z) + 2.5 * te2)
    with pytest.raises(ValueError):
        build_qubo(grid, objective_scale=1.0)


def test_operator_diagonal_matches_energy():
    u = synthetic_universe(3, 1, seed=1)
    grid = WeightGrid(n=3, units=4, min_units=1, max_units=2, k_min=2, k_max=3)
    model = build_qubo(grid, cov=u.cov, benchmark=np.full(3, 1 / 3), objective_scale=1.0, penalty_weight=3.0)
    assert diag_values(model.to_operator()) == pytest.approx(model.energy(all_bitstrings(model.n_qubits)))


def test_boltzmann_weights_of_the_energy_are_reference_distribution_d3():
    u = synthetic_universe(4, 2, seed=5)
    grid = WeightGrid(n=4, units=6, min_units=1, max_units=4, k_min=2, k_max=3)
    b, tau = np.array([0.4, 0.3, 0.2, 0.1]), 0.05
    model = build_qubo(grid, cov=u.cov, benchmark=b, objective_scale=1 / (2 * tau * tau))
    exact = exact_benchmark_aware(exact_uniform_weights(u, grid, None), u, b, tau)
    Z = model.encode(np.rint(exact.weights * grid.units))
    p = np.exp(-model.energy(Z))
    assert p / p.sum() == pytest.approx(exact.prob)


def _small_model():
    u = synthetic_universe(3, 1, seed=2)
    grid = WeightGrid(n=3, units=4, min_units=1, max_units=2, k_min=2, k_max=3)
    return u, grid, build_qubo(grid)


def test_grover_sampler_is_uniform_over_feasible_grid_points():
    u, grid, model = _small_model()
    exact = exact_uniform_weights(u, grid, None)
    Z, info = grover_sample(model, 4000, seed=1)
    assert info["n_qubits"] == model.n_qubits and info["depth"] > 0 and info["shots"] == 4000
    ok = np.abs(model.penalty(Z)) < 1e-9
    assert ok.mean() == pytest.approx(info["success_probability"], abs=0.03) and ok.mean() > 0.8
    r = np.array([0.05, -0.02, 0.01])
    m = compare_to_exact(exact, model.decode(Z[ok]), n_raw=len(Z), asset_returns=r, realised_return=0.0)
    assert m["tv"] < 2.5 * m["tv_noise_floor"] and m["kl"] is not None
    assert m["feasibility_rate"] == pytest.approx(ok.mean())


def test_grover_iterations_formula():
    assert grover_iterations(0.25) == 1 and grover_iterations(1.0) == 0
    assert grover_iterations(1 / 1024) == 25
    with pytest.raises(ValueError):
        grover_iterations(0.0)


def test_simulated_annealing_finds_feasible_states_and_is_scored_against_exact():
    u, grid, model = _small_model()
    Z = simulated_annealing(model, 3000, sweeps=60, seed=0)
    ok = np.abs(model.penalty(Z)) < 1e-9
    assert ok.mean() > 0.5
    exact = exact_uniform_weights(u, grid, None)
    m = compare_to_exact(exact, model.decode(Z[ok]), n_raw=len(Z))
    assert 0 <= m["tv"] <= 1 and m["feasibility_rate"] == pytest.approx(ok.mean())
    # with the tracking-error objective switched on, annealing concentrates on low-TE portfolios
    b = np.array([0.5, 0.25, 0.25])
    tilted = build_qubo(grid, cov=u.cov, benchmark=b, objective_scale=4000.0, penalty_weight=50.0)
    Zt = simulated_annealing(tilted, 500, sweeps=150, seed=1)
    okt = np.abs(tilted.penalty(Zt)) < 1e-9
    te = active_risk(tilted.decode(Zt[okt]), u.cov, b)
    assert okt.mean() > 0.5 and te.mean() < active_risk(exact.weights, u.cov, b).mean()


def test_feasible_mask_counts_grid_points():
    _, grid, model = _small_model()
    assert feasible_mask(model).sum() == len(enumerate_grid(grid))
