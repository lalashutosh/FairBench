import numpy as np

from fairbench.baselines import enumerate_feasible, mcmc_swap_sample, rejection_sample
from fairbench.constraints import MinESG
from fairbench.instances import describe_instance, island_instance, p0_instance
from fairbench.metrics import coverage, swap_components


def test_p0_instance_matches_demo():
    import itertools
    from fairbench.constraints import Cardinality, ConstraintSet, SectorCap
    from fairbench.data import synthetic_universe
    u, cs = p0_instance()
    # independent rebuild of scripts/p0_demo.py's instance
    u2 = synthetic_universe(n=12, n_sectors=3, seed=0)
    avgs = np.array([u2.esg_score[list(c)].mean() for c in itertools.combinations(range(12), 4)])
    m = float(np.quantile(avgs, 0.70)) - 1e-9
    cs2 = ConstraintSet([Cardinality(4)] + [SectorCap(s, 2) for s in sorted(set(u2.sector))] + [MinESG(m)])
    assert np.array_equal(enumerate_feasible(u, cs), enumerate_feasible(u2, cs2))
    d = describe_instance(u, cs)
    assert d["n_components"] == 1 and d["M"] == 120 and d["total"] == 495


def test_island_instance_has_islands():
    u, cs = island_instance()
    d = describe_instance(u, cs)
    assert d["n"] <= 16 and d["n_components"] >= 2
    assert d["largest_frac"] <= 0.8 and sum(d["component_sizes"]) == d["M"]
    assert swap_components(enumerate_feasible(u, cs)) == d["n_components"]


def test_mcmc_trapped_rejection_covers():
    u, cs = island_instance()
    F = enumerate_feasible(u, cs)
    d = describe_instance(u, cs)
    x0 = F[0]
    mc = mcmc_swap_sample(u, cs, 3000, 200, 5, seed=0, x0=x0)
    cov_mc = coverage(mc.samples, F)
    assert cov_mc <= d["largest_frac"] + 1e-9
    rj = rejection_sample(u, cs, 300_000, seed=0)
    assert coverage(rj.samples, F) > 0.95
