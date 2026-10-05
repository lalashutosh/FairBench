import itertools

import numpy as np
import pytest
from scipy.stats import chisquare

from fairbench.baselines import (enumerate_feasible, mcmc_swap_sample, random_k_subsets,
                                 rejection_sample)
from fairbench.constraints import Cardinality, ConstraintSet, MinESG, SectorCap
from fairbench.data import synthetic_universe
from fairbench.metrics import (acceptance_rate, cost_per_feasible_sample, coverage,
                               tv_expected_uniform, tv_to_uniform)

N, K = 10, 4


@pytest.fixture(scope="module")
def setup():
    u = synthetic_universe(N, 3, seed=1)
    cs = ConstraintSet([Cardinality(K), SectorCap("S0", 2), MinESG(np.median(u.esg_score))])
    F = enumerate_feasible(u, cs)
    return u, cs, F


def _codes(X):
    return (np.asarray(X, dtype=np.int64) << np.arange(X.shape[1])).sum(axis=1)


def test_enumerate_matches_bruteforce(setup):
    u, cs, F = setup
    allx = np.array(list(itertools.product([0, 1], repeat=N)), dtype=np.uint8)
    ref = allx[cs.check_batch(allx, u)]
    assert 5 < len(F) < 210
    assert F.dtype == np.uint8
    assert set(_codes(F)) == set(_codes(ref))
    assert len(F) == len(ref)


def test_enumerate_guards(setup):
    u, cs, _ = setup
    with pytest.raises(ValueError):
        enumerate_feasible(synthetic_universe(26, 3), cs)
    with pytest.raises(ValueError):
        enumerate_feasible(u, ConstraintSet([SectorCap("S0", 2)]))


def test_random_k_subsets_uniform():
    X = random_k_subsets(6, 3, 20000, seed=0)
    assert X.shape == (20000, 6) and X.dtype == np.uint8
    assert (X.sum(1) == 3).all()
    _, c = np.unique(_codes(X), return_counts=True)
    assert len(c) == 20
    assert chisquare(c).pvalue > 0.001


def test_rejection(setup):
    u, cs, F = setup
    r = rejection_sample(u, cs, 5000, seed=3)
    assert r.cost == 5000 and r.cost_unit == "proposals"
    assert r.raw.shape == (5000, N)
    assert cs.check_batch(r.samples, u).all()
    assert len(r.samples) == r.info["n_accepted"] > 0
    assert acceptance_rate(r.raw, cs, u) == pytest.approx(len(r.samples) / 5000)
    assert rejection_sample(u, cs, 100, seed=0, max_draws=10).cost == 10


def test_mcmc(setup):
    u, cs, F = setup
    burn, thin, shots = 200, 5, 6000
    r = mcmc_swap_sample(u, cs, shots, burn, thin, seed=4)
    assert r.samples.shape == (shots, N) and r.samples.dtype == np.uint8
    assert cs.check_batch(r.samples, u).all()
    assert r.cost == burn + shots * thin and r.cost_unit == "moves"
    assert 0 < r.info["accepted_moves"] <= r.cost
    assert coverage(r.samples, F) > 0.95
    tv = tv_to_uniform(r.samples, F)
    assert tv < 3 * tv_expected_uniform(len(F), shots, seed=0) + 0.05


def test_mcmc_errors(setup):
    u, cs, _ = setup
    with pytest.raises(ValueError):
        mcmc_swap_sample(u, ConstraintSet([SectorCap("S0", 2)]), 5, 1, 1)
    with pytest.raises(ValueError):
        mcmc_swap_sample(u, cs, 5, 1, 1, x0=np.ones(N, dtype=np.uint8))


def test_metrics_toy():
    F = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    X = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]] * 4, dtype=np.uint8)
    assert tv_to_uniform(X, F) == pytest.approx(0.0)
    assert coverage(X, F) == 1.0
    X2 = np.array([[1, 1, 0]] * 3 + [[1, 0, 1]], dtype=np.uint8)
    # p=(.75,.25,0) vs 1/3: 0.5*(.4167+.0833+.3333)
    assert tv_to_uniform(X2, F) == pytest.approx(0.5 * (5 / 12 + 1 / 12 + 1 / 3))
    assert coverage(X2, F) == pytest.approx(2 / 3)
    # off-support mass
    X3 = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.uint8)
    assert tv_to_uniform(X3, F) == pytest.approx(0.5 * (1 / 3 + 1 / 6 + 1 / 3 + 1 / 2))
    assert cost_per_feasible_sample(10, 4) == 2.5
    assert cost_per_feasible_sample(10, 0) == float("inf")
    assert 0 < tv_expected_uniform(20, 1000, seed=0) < 0.2
