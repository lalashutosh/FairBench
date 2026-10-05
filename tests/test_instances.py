import numpy as np

from fairbench.baselines import enumerate_feasible, mcmc_swap_sample, rejection_sample
from fairbench.constraints import MinESG
from fairbench.instances import describe_instance, island_instance, p0_instance
from fairbench.metrics import coverage, swap_components


def test_p0_instance_matches_demo():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "p0_demo", pathlib.Path(__file__).parents[1] / "scripts" / "p0_demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    u, cs = p0_instance()
    assert demo.p0_instance is p0_instance  # demo uses the shared instance, no copy
    from fairbench.training import p0_instance as tp0
    assert tp0 is p0_instance
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
    assert cov_mc >= 0.3  # chain does move within its island
    rj = rejection_sample(u, cs, 300_000, seed=0)
    assert coverage(rj.samples, F) > 0.95


def test_island_seed_must_be_zero():
    import pytest
    with pytest.raises(ValueError):
        island_instance(seed=1)


def test_dropping_any_rule_type_reconnects():
    from fairbench.constraints import CarbonCap, ConstraintSet, SectorCap
    from fairbench.metrics import swap_components
    u, cs = island_instance()
    assert swap_components(enumerate_feasible(u, cs)) == 2
    for typ in (SectorCap, MinESG, CarbonCap):
        cs2 = ConstraintSet([c for c in cs.constraints if not isinstance(c, typ)])
        assert swap_components(enumerate_feasible(u, cs2)) == 1, typ.__name__
