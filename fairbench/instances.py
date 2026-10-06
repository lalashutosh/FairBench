"""Benchmark instances: the P0 toy instance and an "island" instance.

Island instance
---------------
islands = which sector holds the singleton (sector mixes (2,1,2) [29] and (1,2,2) [26]).
Every single cross-island swap breaks the ESG floor, the carbon cap, or a sector cap;
removing any one rule type (SectorCap, MinESG, CarbonCap) reconnects the set. ESG and
carbon are positively correlated (0.81). A Metropolis swap chain (reject infeasible)
never crosses; rejection sampling and the Dicke circuit (all weight-k states) reach
both islands.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .baselines import enumerate_feasible
from .constraints import CarbonCap, Cardinality, ConstraintSet, MinESG, SectorCap
from .data import Universe, synthetic_universe


def p0_instance(seed: int = 0) -> tuple[Universe, ConstraintSet]:
    """Exact P0 demo instance: n=12, k=4, 3 sectors capped at 2, MinESG at the
    70th percentile of avg ESG over all k-subsets (minus 1e-9)."""
    n, k = 12, 4
    u = synthetic_universe(n=n, n_sectors=3, seed=seed)
    avgs = np.array([u.esg_score[list(c)].mean() for c in itertools.combinations(range(n), k)])
    m = float(np.quantile(avgs, 0.70)) - 1e-9
    cons = [Cardinality(k)] + [SectorCap(s, 2) for s in sorted(set(u.sector))] + [MinESG(m)]
    return u, ConstraintSet(cons)


def _component_sizes(F: np.ndarray) -> list[int]:
    """Sizes (descending) of swap-connected components of feasible set F."""
    F = np.unique(np.atleast_2d(F).astype(np.int32), axis=0)
    M = F.shape[0]
    parent = np.arange(M)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(M - 1):
        d = np.abs(F[i + 1:] - F[i]).sum(axis=1)
        for j in np.flatnonzero(d == 2) + i + 1:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    roots = np.array([find(i) for i in range(M)])
    return sorted(np.unique(roots, return_counts=True)[1].tolist(), reverse=True)


def _build(n: int, k: int, n_sectors: int, seed: int, cap: int, esg_q: float,
           carbon_q: float, noise: float = 8.0):
    """Sector-structured universe with POSITIVELY correlated ESG and carbon
    (clean-ESG names tend to be carbon-heavy industrials/utilities with strong
    disclosure; low-carbon names tend to score lower on ESG): per-sector ESG and
    carbon levels plus asset noise, carbon pulled towards 60 + 4.5*ESG.
    Rules: SectorCap(cap) per sector, MinESG at quantile ``esg_q`` and CarbonCap
    at quantile ``carbon_q`` of the average over sector-cap-feasible k-subsets."""
    u = synthetic_universe(n=n, n_sectors=n_sectors, seed=seed)
    r = np.random.default_rng(seed + 7)
    si = np.array([int(s[1:]) for s in u.sector])
    e = r.uniform(20, 85, n_sectors)
    c = r.uniform(60, 450, n_sectors)
    u.esg_score = np.clip(e[si] + r.normal(0, noise, n), 0, 100)
    u.carbon = np.clip(0.5 * c[si] + 0.5 * (60 + 4.5 * u.esg_score) + r.normal(0, noise * 5, n), 10, 500)
    base = [Cardinality(k)] + [SectorCap(f"S{i}", cap) for i in range(n_sectors)]
    comb = np.array(list(itertools.combinations(range(n), k)))
    X = np.zeros((len(comb), n), np.uint8)
    np.put_along_axis(X, comb, 1, 1)
    cb = comb[ConstraintSet(base).check_batch(X, u)]
    m = float(np.quantile(u.esg_score[cb].mean(1), esg_q)) - 1e-9
    cc = float(np.quantile(u.carbon[cb].mean(1), carbon_q)) + 1e-9
    return u, ConstraintSet(base + [MinESG(m), CarbonCap(cc)])


def search_island_instances(n: int = 16, k: int = 5, seeds=range(40), max_largest_frac: float = 0.7,
                            min_M: int = 40, min_second: int = 10,
                            n_sectors_opts=(3, 4), caps=(2, 3), esg_qs=(0.5, 0.7, 0.85),
                            carbon_qs=(0.2, 0.35, 0.5, 0.65), noise: float = 8.0) -> list[dict]:
    """Seeded deterministic grid search over plausible rule combos. Returns dicts
    (params + stats) with >=2 swap components, M >= min_M, second component >=
    min_second and largest <= max_largest_frac * M; biggest M first.
    The default call (~1 min) found ISLAND_PARAMS."""
    out = []
    for n_sec, cap, eq, cq in itertools.product(n_sectors_opts, caps, esg_qs, carbon_qs):
        for seed in seeds:
            u, cs = _build(n, k, n_sec, seed, cap, eq, cq, noise)
            F = enumerate_feasible(u, cs)
            if len(F) < min_M:
                continue
            sizes = _component_sizes(F)
            if len(sizes) >= 2 and sizes[1] >= min_second and sizes[0] <= max_largest_frac * len(F):
                out.append(dict(n=n, k=k, seed=seed, n_sectors=n_sec, cap=cap, esg_q=eq,
                                carbon_q=cq, noise=noise, M=len(F), sizes=sizes,
                                largest_frac=sizes[0] / len(F)))
    out.sort(key=lambda d: (-d["M"], d["largest_frac"]))
    return out


# Chosen from search_island_instances(): M=55 of C(16,5)=4368, components [29, 26].
ISLAND_PARAMS: dict = dict(n=16, k=5, seed=29, n_sectors=3, cap=2, esg_q=0.85, carbon_q=0.65, noise=8.0)


def island_instance(seed: int = 0, n: int | None = None, k: int | None = None) -> tuple[Universe, ConstraintSet]:
    """Deterministic island instance (2 swap components) from ISLAND_PARAMS. ``seed`` is kept
    for API stability but must be 0 (ValueError otherwise). n/k override only for
    experimentation (islands not guaranteed)."""
    if seed != 0:
        raise ValueError("island_instance is fixed (ISLAND_PARAMS); seed must be 0")
    p = dict(ISLAND_PARAMS)
    if n is not None:
        p["n"] = n
    if k is not None:
        p["k"] = k
    return _build(p["n"], p["k"], p["n_sectors"], p["seed"], p["cap"], p["esg_q"],
                  p["carbon_q"], p["noise"])


def describe_instance(u: Universe, cs: ConstraintSet) -> dict:
    """M, C(n,k), feasible fraction, swap components, component sizes."""
    F = enumerate_feasible(u, cs)
    k = cs.cardinality
    total = math.comb(u.n, k)
    M = len(F)
    sizes = _component_sizes(F) if M else []
    return dict(n=u.n, k=k, M=M, total=total, feasible_fraction=M / total,
                n_components=len(sizes), component_sizes=sizes,
                largest_frac=(sizes[0] / M) if M else float("nan"))


_FAMILY_CACHE: dict = {}


def island_family_legacy(n: int, k: int | None = None, seed: int = 0, n_seeds: int = 40,
                  n_sectors_opts=(3, 4), caps=(2, 3), esg_qs=(0.5, 0.7, 0.85),
                  carbon_qs=(0.2, 0.35, 0.5, 0.65)) -> tuple[Universe, ConstraintSet]:
    """LEGACY (Q-B results): per-n grid search over construction params, so instances at
    different n come from DIFFERENT constructions and a size trend over them is not
    meaningful. Use ``island_family`` for new work.

    Island-style instance at size n (default k = n//3): same construction as
    ``island_instance`` (``_build``: sector caps + opposing MinESG/CarbonCap tight
    quantiles on correlated ESG/carbon). Deterministic grid search over
    (sectors, cap, esg_q, carbon_q) x universe seeds ``seed..seed+n_seeds-1``.
    Preference (tiered): (1) >=2 swap components, M >= 20, second component >= 5 and
    largest <= 0.7 M, biggest M first; (2) any >=2 components, biggest M first;
    (3) no islands found: the largest-M non-trivial instance (check
    ``describe_instance`` -> n_components; nothing is faked).
    (n, k, seed) == (16, 5, 0) returns ``island_instance()``."""
    k = n // 3 if k is None else k
    if (n, k, seed) == (16, 5, 0):
        return island_instance()
    key = (n, k, seed, n_seeds, tuple(n_sectors_opts), tuple(caps), tuple(esg_qs), tuple(carbon_qs))
    if key not in _FAMILY_CACHE:
        best = {1: None, 2: None, 3: None}
        for n_sec, cap, eq, cq in itertools.product(n_sectors_opts, caps, esg_qs, carbon_qs):
            if n_sec * cap < k:
                continue
            for s in range(seed, seed + n_seeds):
                u, cs = _build(n, k, n_sec, s, cap, eq, cq)
                F = enumerate_feasible(u, cs)
                M = len(F)
                if M < 2:
                    continue
                sizes = _component_sizes(F)
                if len(sizes) >= 2 and M >= 20 and sizes[1] >= 5 and sizes[0] <= 0.7 * M:
                    tier = 1
                elif len(sizes) >= 2:
                    tier = 2
                else:
                    tier = 3
                if best[tier] is None or M > best[tier][0]:
                    best[tier] = (M, (n_sec, s, cap, eq, cq))
        pick = next(best[t] for t in (1, 2, 3) if best[t] is not None)
        _FAMILY_CACHE[key] = pick[1]
    n_sec, s, cap, eq, cq = _FAMILY_CACHE[key]
    return _build(n, k, n_sec, s, cap, eq, cq)


# Fixed construction for island_family (same for every n and seed). Chosen by a scan over
# sector levels / residual scale / quantiles (k=5, 3 sectors, cap 2) maximising the fraction
# of universe seeds with >= 2 swap components (second >= 3) at n = 12, 14, 16.
FAMILY_PARAMS: dict = dict(k=5, n_sectors=3, cap=2, esg_levels=(70.0, 40.0, 50.0),
                           carbon_levels=(250.0, 80.0, 200.0), sigma=10.0, gamma=6.0,
                           esg_q=0.5, carbon_q=0.5)


def _build_family(n: int, k: int, seed: int, n_sectors: int, cap: int, esg_levels, carbon_levels,
                  sigma: float, gamma: float, esg_q: float, carbon_q: float):
    """Designed island construction. Sector levels are FIXED (sector 0: clean ESG / high
    carbon, sector 1: low ESG / low carbon, sector 2: in between); the seed only draws the
    base universe (sector labels, returns) and asset residuals z_i ~ N(0,1):
        esg_i    = E[s_i] + sigma * z_i
        carbon_i = C[s_i] + gamma * sigma * z_i + N(0, sigma)
    (strong positive within-sector ESG/carbon correlation). Rules: SectorCap(cap) per
    sector, MinESG and CarbonCap at quantiles esg_q / carbon_q of the averages over
    sector-cap-feasible k-subsets. With k = 2*n_sectors - 1 every portfolio is "all sectors
    at cap but one"; ESG-tight and carbon-tight sector mixes need opposite residuals, so a
    single cross-mix swap breaks one of the two average rules -> swap islands."""
    if len(esg_levels) != n_sectors or len(carbon_levels) != n_sectors:
        raise ValueError("esg_levels / carbon_levels must have n_sectors entries")
    u = synthetic_universe(n=n, n_sectors=n_sectors, seed=seed)
    r = np.random.default_rng(seed + 7)
    si = np.array([int(s[1:]) for s in u.sector])
    z = r.normal(0.0, 1.0, n)
    u.esg_score = np.clip(np.asarray(esg_levels, float)[si] + sigma * z, 0, 100)
    u.carbon = np.clip(np.asarray(carbon_levels, float)[si] + gamma * sigma * z
                       + r.normal(0.0, sigma, n), 10, 500)
    base = [Cardinality(k)] + [SectorCap(f"S{i}", cap) for i in range(n_sectors)]
    comb = np.array(list(itertools.combinations(range(n), k)))
    X = np.zeros((len(comb), n), np.uint8)
    np.put_along_axis(X, comb, 1, 1)
    cb = comb[ConstraintSet(base).check_batch(X, u)]
    m = float(np.quantile(u.esg_score[cb].mean(1), esg_q)) - 1e-9
    cc = float(np.quantile(u.carbon[cb].mean(1), carbon_q)) + 1e-9
    return u, ConstraintSet(base + [MinESG(m), CarbonCap(cc)])


def island_family(n: int, k: int | None = None, seed: int = 0) -> tuple[Universe, ConstraintSet]:
    """Island-family instance of size n with FIXED construction params (``FAMILY_PARAMS``)
    for every n and seed; ``seed`` only varies the random universe (``_build_family``).
    Default k = FAMILY_PARAMS["k"] = 5 for all n (other k: islands not by design).

    Islands are typical but NOT guaranteed (nothing is filtered or faked): check
    ``describe_instance(u, cs)["n_components"]``. Seeds 0..9: >= 2 components with second
    component >= 3 for 8/10 (n=12), 9/10 (n=14), 9/10 (n=16); seed 6 is connected at
    n=14,16. Components are often several (fragmented), not exactly two. Intended for
    n >= 12: at n = 8, 10 (k=5 fixed) M <= 11 and the instances are degenerate.
    For a size trend, report all
    seeds (or the seeds that are islanded at every n) rather than cherry-picking.
    For the pre-fix per-n grid-searched instances (Q-B), see ``island_family_legacy``."""
    p = dict(FAMILY_PARAMS)
    if k is not None:
        p["k"] = int(k)
    if n < p["k"] + 1:
        raise ValueError(f"need n > k; got n={n}, k={p['k']}")
    return _build_family(n, p["k"], seed, p["n_sectors"], p["cap"], p["esg_levels"],
                         p["carbon_levels"], p["sigma"], p["gamma"], p["esg_q"], p["carbon_q"])
