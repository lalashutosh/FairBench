"""Benchmark instances: the P0 toy instance and an "island" instance.

Island instance (why swap chains get trapped)
---------------------------------------------
A swap replaces one held asset i by a non-held j; the sums of ESG and carbon
shift by (e_j - e_i, c_j - c_i). MinESG (sum ESG >= a) and CarbonCap (sum carbon
<= b) are two opposing half-planes; when ESG and carbon are positively
correlated across assets, the feasible region in (ESG, carbon) space is a thin
wedge near the corner (a, b). Moving between two feasible portfolios that
differ in two names needs two swaps; of the two orders, one first adds ESG
(and carbon) -> breaks the CarbonCap, the other first removes carbon (and ESG)
-> breaks MinESG. Both intermediates are infeasible, and per-sector caps
remove further paths. The feasible set splits into swap-disconnected islands;
a Metropolis swap chain (reject infeasible) never crosses, while rejection
sampling and the Dicke circuit (all weight-k states) reach every island.
Average-based rules are linear in the sums, so the two intermediates always sum
to the endpoints: the midpoint is feasible, the single steps are not.
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
    """Deterministic island instance (2 swap components). The hard-coded
    ISLAND_PARAMS are used; ``seed`` is accepted for interface symmetry but
    ignored. n/k override only for experimentation (islands not guaranteed)."""
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
