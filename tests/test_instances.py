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


def test_island_family_fixed_params_deterministic():
    from fairbench.instances import FAMILY_PARAMS, island_family
    a, b = island_family(14, seed=3), island_family(14, seed=3)
    assert np.array_equal(a[0].esg_score, b[0].esg_score)
    assert [vars(c) for c in a[1].constraints].__repr__() == [vars(c) for c in b[1].constraints].__repr__()
    assert np.array_equal(enumerate_feasible(*a), enumerate_feasible(*b))
    c = island_family(14, seed=4)
    assert not np.array_equal(a[0].esg_score, c[0].esg_score)  # seed varies the universe
    for n in (12, 16):  # same construction (k, sectors, cap) at every n
        u, cs = island_family(n, seed=0)
        assert cs.cardinality == FAMILY_PARAMS["k"] and len(set(u.sector)) == FAMILY_PARAMS["n_sectors"]


def test_island_family_islands_typical():
    from fairbench.instances import island_family
    good = 0
    for s in range(5):
        d = describe_instance(*island_family(14, seed=s))
        good += d["n_components"] >= 2 and d["component_sizes"][1] >= 3
    assert good >= 4


def test_island_family_legacy_kept():
    from fairbench.instances import island_family_legacy
    u, cs = island_family_legacy(16, 5, 0)
    u0, cs0 = island_instance()
    assert np.array_equal(u.esg_score, u0.esg_score)


def test_scaled_family_deterministic():
    from fairbench.instances import scaled_family
    u1, c1 = scaled_family(30, seed=2, n_mc=20_000)
    u2, c2 = scaled_family(30, seed=2, n_mc=20_000)
    assert np.array_equal(u1.esg_score, u2.esg_score) and np.array_equal(u1.carbon, u2.carbon)
    assert [vars(c) for c in c1.constraints] == [vars(c) for c in c2.constraints]
    assert c1.cardinality == 5


def test_scaled_family_mc_matches_exact_n16():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))
    from aa_pf_scaling import estimate, wilson
    from fairbench.instances import scaled_family
    u, cs = scaled_family(16, seed=0, n_mc=50_000)
    exact = len(enumerate_feasible(u, cs)) / 4368
    e = estimate(u, cs, N_max=400_000, chunk=100_000, target_hits=10**9)
    lo, hi = wilson(e["hits"], e["N"], z=3.0)  # 99.7% (fixed seed; 95% is a coin-flip-prone test)
    assert lo <= exact <= hi
