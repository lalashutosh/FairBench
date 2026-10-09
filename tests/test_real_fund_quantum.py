"""The quantum estimator on real-fund periods (synthetic example data, noiseless simulation)."""
import numpy as np
import pytest

from fairbench.apps.real_fund import build_period_case
from fairbench.apps.real_fund_quantum import exact_sleeve, oracle_resources, quantum_inputs, quantum_percentile
from fairbench.baselines import random_k_subsets
from fairbench.constraints import Cardinality, ConstraintSet, Exclusion
from fairbench.ingest.nport import parse_nport
from fairbench.ingest.nport_synth import example_history
from fairbench.portfolio.reference import uniform_subsets, weighted_percentile_of
from fairbench.portfolio.weights import subset_weights


@pytest.fixture(scope="module")
def ex():
    h = example_history(seed=0)
    d = sorted(h["fund"])
    P, F = {x: parse_nport(h["parent"][x]) for x in d}, {x: parse_nport(h["fund"][x]) for x in d}
    return dict(case=build_period_case(P[d[1]], P[d[2]], F[d[1]], F[d[2]]), excluded=h["meta"]["excluded_keys"])


@pytest.mark.parametrize("weights", ["equal", "benchmark"])
def test_linear_rule_is_exactly_the_return_comparison(ex, weights):
    """The oracle's rule "sum_i c_i x_i < t" must agree with "portfolio return < fund return"
    on every portfolio, for both weightings the quantum core can express."""
    case = ex["case"]
    inp = quantum_inputs(case, weights)
    X = random_k_subsets(inp["n"], inp["k"], 20_000, seed=1)
    W = subset_weights(X, "equal" if weights == "equal" else "benchmark", case.benchmark)
    direct = W @ case.asset_returns < case.fund_frozen_return
    assert np.array_equal(inp["below_fund"].check_batch(X, inp["u"]), direct)
    assert inp["cs"].cardinality == case.k and inp["n"] == case.universe.n and inp["s"] == case.fund_frozen_return


def test_exclusions_shrink_the_register_not_the_valid_fraction(ex):
    case = ex["case"]
    inp = quantum_inputs(case, "equal", ex["excluded"])
    assert inp["n"] == case.universe.n - 6 and inp["n_excluded"] == 6
    assert not set(inp["u"].tickers) & set(ex["excluded"])       # the Dicke state is over allowed stocks only


def test_capped_weighting_is_refused_because_it_is_not_linear(ex):
    with pytest.raises(ValueError, match="not linear"):
        quantum_inputs(ex["case"], "benchmark_capped")


def test_quantum_percentile_matches_the_classical_pipeline(ex):
    case = ex["case"]
    q = quantum_percentile(case, excluded_keys=ex["excluded"], eps=0.01, n_mc=60_000, reps=12, seed=0)
    pos = {k: i for i, k in enumerate(case.keys)}
    cs = ConstraintSet([Cardinality(case.k), Exclusion(sorted(pos[x] for x in ex["excluded"]))])
    ref = uniform_subsets(case.universe, cs, 8000, policy="equal", seed=3)
    classical = weighted_percentile_of(ref.returns(case.asset_returns), case.fund_frozen_return)
    assert abs(q["percentile"] - classical) < 2.5                  # same question, two estimators
    assert abs(q["quantum_estimate"] - q["percentile"]) < 1.0 and q["quantum_worst_error"] < 2.0
    assert q["valid_fraction"] == 1.0 and q["weights"] == "equal" and "noiseless" in q["device"]
    assert q["quantum_queries_p90"] >= q["quantum_queries_median"] > 0


def test_query_ratio_grows_as_the_precision_tightens(ex):
    """1/eps against 1/eps^2: asking for ten times the precision widens the gap."""
    case = ex["case"]
    loose = quantum_percentile(case, eps=0.02, n_mc=40_000, reps=10, seed=1)
    tight = quantum_percentile(case, eps=0.002, n_mc=40_000, reps=10, seed=1)
    assert tight["query_ratio"] > 3 * loose["query_ratio"]
    assert tight["classical_samples_same_precision"] == pytest.approx(100 * loose["classical_samples_same_precision"], rel=1e-6)


def test_exact_sleeve_agrees_three_ways(ex):
    s = exact_sleeve(ex["case"], n_top=14, eps=0.02, seed=0)
    assert s is not None and 0 < s["k"] < s["n"] <= 14 and s["n_portfolios"] > 0
    assert abs(s["quantum"] - s["exact"]) < 4.0 and abs(s["classical_same_queries"] - s["exact"]) < 6.0
    assert s["quantum_ci"][0] - 1.0 <= s["exact"] <= s["quantum_ci"][1] + 1.0
    assert "state-vector" in s["device"]


def test_oracle_resources_are_a_fault_tolerant_workload(ex):
    r = oracle_resources(ex["case"], excluded_keys=ex["excluded"])
    assert r["n"] == ex["case"].universe.n - 6 and r["logical_qubits"] > r["n"]      # data qubits plus workspace
    assert r["t_gates_per_step"] > r["toffoli_per_step"] > 0 and r["rotations_per_step"] > 0
