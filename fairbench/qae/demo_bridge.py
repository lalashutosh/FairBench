"""Bridge: run the classical attribution tool and the quantum-AE estimator side by side.

Units follow ``fairbench.apps.attribution``: percentile in 0-100, medians and effects in
total-return fractions (buy-and-hold, equal weight, rebalance=False). Quantum AE is a
noiseless simulation; query counts are oracle calls, not wall-clock time.
"""
from __future__ import annotations

import numpy as np

from fairbench.apps.attribution import attribute
from fairbench.qae.attribution_qae import qae_attribute

FIELDS = ("percentile", "null_median", "benchmark_median", "constraint_effect", "manager_effect")


def attribute_qae(u, cs, fund, eps: float = 0.01, method: str = "iqae", seed: int = 0,
                  n_samples: int = 5000):
    """Return (result, details). result[method] = dict of FIELDS (+ queries) for
    'exact', 'classical' (attribute() rejection sampling, n_samples) and 'quantum'."""
    rng = np.random.default_rng(seed)
    q = qae_attribute(u, cs, fund, method=method, eps_pct=eps, eps_med=eps, rng=rng)
    c = attribute(u, cs, fund, n_samples=n_samples, rebalance=False, seed=seed)
    exact = {k: float(q.exact[k]) for k in FIELDS}
    exact["queries"] = None
    classical = {"percentile": float(c.percentile), "null_median": float(c.null_median),
                 "benchmark_median": float(c.benchmark), "constraint_effect": float(c.constraint_effect),
                 "manager_effect": float(c.manager_effect), "queries": int(c.cost) + int(n_samples)}
    quantum = {k: float(getattr(q, k)) for k in FIELDS}
    quantum["queries"] = int(q.queries["total"])
    matched = dict(q.classical)
    matched_q = int(matched["queries"]["total"])
    matched = {k: float(matched[k]) for k in FIELDS}
    matched["queries"] = matched_q
    result = {"exact": exact, "classical": classical, "quantum": quantum,
              "classical_matched": matched}
    details = {"method": method, "eps": eps, "seed": seed, "fund_return": float(q.fund),
               "n_samples": n_samples, "quantum_queries": q.queries,
               "classical_matched_queries": q.classical["queries"],
               "classical_cost_unit": c.cost_unit, "classical_cost_raw": int(c.cost),
               "units": "percentile 0-100; medians/effects as total-return fractions"}
    return result, details
