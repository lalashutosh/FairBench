"""Training objectives: callables on samples (lower = better), with optional exact variants.

Every objective ``f`` is an :class:`Objective`:

* ``f(X)`` with ``X`` a ``(shots, n)`` uint8 array (column i = asset i) -> float. Works with any
  sampler (Aer, MPS, hardware, classical baselines).
* ``f.exact(probs)`` with ``probs`` a length-``2**n`` probability vector over basis states
  (entry b has x_i = bit i of b, i.e. Qiskit's ``Statevector.probabilities()`` order and
  ``hamiltonian.diag_values`` order) -> float. This is a *simulation shortcut* (needs the
  full statevector, n <= ~16-20); it does not exist on hardware.
* ``f.report(...)`` / ``f.report_exact(...)`` -> dict of the components (for logging).

Uniformity divergence: chi-square / collision probability
---------------------------------------------------------
Let F be the feasible set (|F| = M), P_F the feasible mass and q = p|F the conditional
distribution on F. We use

    D(q) = chi2(q || U_F) = M * sum_{x in F} q_x^2 - 1   ( = M * collision_prob - 1 )

Why this and not the alternatives:

* TV is non-smooth (|.|) and its plug-in estimate is strongly biased upward
  (~sqrt(M/N) even for a perfect sampler) - bad as a training signal.
* KL(U || q) = -log M - (1/M) sum log q_x is infinite whenever any feasible state has
  q_x = 0, and its plug-in estimate is infinite until every one of the M states is seen.
  Unusable from shots. KL(q || U) is finite but its plug-in estimate is biased by
  ~ (M-1)/(2N) and needs bias correction tricks.
* chi2 is smooth (quadratic in amplitudes' moduli), zero iff q is uniform on F, and the
  collision probability sum q_x^2 has an *unbiased* sample estimator: the U-statistic
  sum_x c_x (c_x - 1) / (N_F (N_F - 1)) over the N_F feasible shots (c_x = counts).
  It also controls TV:  TV(q, U_F) <= 0.5 * sqrt(chi2).

Finite-sample notes (sampled mode):

* P_F_hat = N_F / N is unbiased.
* chi2_hat uses the collision U-statistic, which is unbiased for chi2 *conditional on
  N_F >= 2* (the naive plug-in M * sum p_hat^2 - 1 would carry a +(M - 1)/N_F-ish upward bias).
  The product structure loss = -P_F + lam * chi2 is therefore unbiased up to the
  conditioning on N_F >= 2. If N_F < 2 we cannot estimate chi2 and return the worst-case
  value chi2 = M - 1 (all mass on one state) - a deliberately pessimistic fallback.
* The variance of chi2_hat is large when N_F is small relative to sqrt(M); use enough
  shots (N_F >> sqrt(M)) or the exact path for training and treat sampled numbers as noisy.
* Selecting the "best" parameters by a noisy sampled objective is optimistically biased
  (winner's curse); re-evaluate on fresh shots or exactly before reporting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .constraints import ConstraintSet
from .data import Universe


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def encode(X: np.ndarray) -> np.ndarray:
    """Rows of a (m, n) 0/1 array -> int64 basis index b = sum_i x_i 2^i (n <= 62)."""
    X = np.atleast_2d(np.asarray(X))
    n = X.shape[1]
    if n > 62:
        raise ValueError("encode supports n <= 62")
    return (X.astype(np.int64) << np.arange(n, dtype=np.int64)).sum(axis=1)


def all_basis_states(n: int) -> np.ndarray:
    """(2^n, n) uint8, row b has x_i = bit i of b."""
    b = np.arange(2 ** n, dtype=np.int64)
    return ((b[:, None] >> np.arange(n)[None, :]) & 1).astype(np.uint8)


@dataclass
class Objective:
    """Callable objective. ``fn(X)`` sampled; ``exact_fn(probs)`` optional exact path."""

    fn: Callable[[np.ndarray], float]
    exact_fn: Callable[[np.ndarray], float] | None = None
    name: str = "objective"
    report_fn: Callable[[np.ndarray], dict] | None = None
    report_exact_fn: Callable[[np.ndarray], dict] | None = None
    meta: dict = field(default_factory=dict)

    def __call__(self, X: np.ndarray) -> float:
        return float(self.fn(X))

    @property
    def has_exact(self) -> bool:
        return self.exact_fn is not None

    def exact(self, probs: np.ndarray) -> float:
        if self.exact_fn is None:
            raise NotImplementedError(f"{self.name} has no exact (probability-vector) variant")
        return float(self.exact_fn(np.asarray(probs, dtype=float)))

    def report(self, X: np.ndarray) -> dict:
        return self.report_fn(X) if self.report_fn else {"value": self(X)}

    def report_exact(self, probs: np.ndarray) -> dict:
        return self.report_exact_fn(probs) if self.report_exact_fn else {"value": self.exact(probs)}


# --------------------------------------------------------------------------
# CVaR (optimizer pivot)
# --------------------------------------------------------------------------
def _cvar_sorted(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """CVaR_alpha = mean of the lowest alpha-fraction of the (weighted) distribution."""
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    total = w.sum()
    if total <= 0:
        return float("nan")
    w = w / total
    cum = np.cumsum(w)
    # take full weight of states whose cumulative mass <= alpha, then a fractional part
    take = np.minimum(w, np.maximum(0.0, alpha - (cum - w)))
    return float((take * v).sum() / alpha)


def cvar_objective(op_or_energy_fn, alpha: float = 0.1, n: int | None = None) -> Objective:
    """CVaR_alpha of the energy (lower = better), Barkoutsos et al. 2020.

    ``op_or_energy_fn``: a diagonal ``SparsePauliOp`` (energies via
    ``hamiltonian.diag_values``) or a function ``E(X) -> (m,)`` energies on bitstrings.
    alpha = 1 gives the plain expectation. Sampled: lowest ceil-free alpha fraction of shots
    (fractional weight on the boundary shot). Exact: same on the full distribution.
    For an energy function the exact path evaluates E on all 2^n states (n <= ~20).
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if hasattr(op_or_energy_fn, "paulis"):
        from .quantum.hamiltonian import diag_values
        diag = diag_values(op_or_energy_fn)

        def energy(X):
            return diag[encode(X)]

        def exact_energies(nq):
            return diag
    else:
        energy = op_or_energy_fn
        cache: dict[int, np.ndarray] = {}

        def exact_energies(nq):
            if nq not in cache:
                cache[nq] = np.asarray(energy(all_basis_states(nq)), dtype=float)
            return cache[nq]

    def fn(X):
        X = np.atleast_2d(X)
        e = np.asarray(energy(X), dtype=float)
        return _cvar_sorted(e, np.ones_like(e), alpha)

    def exact_fn(probs):
        nq = int(round(np.log2(probs.size)))
        return _cvar_sorted(exact_energies(nq), probs, alpha)

    return Objective(fn, exact_fn, name=f"cvar(alpha={alpha})", meta={"alpha": alpha})


# --------------------------------------------------------------------------
# Uniformity over the feasible set
# --------------------------------------------------------------------------
def chi2_from_counts(counts: np.ndarray, M: int) -> float:
    """Unbiased (U-statistic) estimate of chi2(q || U_M) from counts over feasible states.

    Returns M - 1 (worst case) if fewer than 2 feasible samples.
    """
    c = np.asarray(counts, dtype=float)
    N = c.sum()
    if N < 2:
        return float(M - 1)
    coll = (c * (c - 1)).sum() / (N * (N - 1))
    return float(M * coll - 1.0)


def uniformity_stats_exact(probs: np.ndarray, fcodes: np.ndarray) -> dict:
    """Exact P_F, chi2, TV of p|F vs uniform, min-ratio from a basis-state probability vector."""
    probs = np.asarray(probs, dtype=float)
    M = fcodes.size
    pf_vec = probs[fcodes]
    P_F = float(pf_vec.sum())
    if P_F <= 0:
        return {"P_F": 0.0, "chi2": float(M - 1), "tv": 1.0, "min_ratio": 0.0, "M": M}
    q = pf_vec / P_F
    return {
        "P_F": P_F,
        "chi2": float(M * (q ** 2).sum() - 1.0),
        "tv": float(0.5 * np.abs(q - 1.0 / M).sum()),
        # M * min q: acceptance of exact rejection-correction q -> uniform (needs q known)
        "min_ratio": float(M * q.min()),
        "M": M,
    }


def uniformity_objective(cs: ConstraintSet | None, u: Universe | None,
                         feasible_set: np.ndarray | None = None, lam: float = 1.0) -> Objective:
    """loss = -P_F + lam * chi2(p|F || uniform on F).  Lower is better.

    ``feasible_set``: exact feasible set (m, n) (defaults to ``enumerate_feasible(u, cs)``).
    Sampled mode classifies shots by membership in ``feasible_set`` (equivalent to
    ``cs.check_batch`` when the set is exact). See module doc for the divergence choice and
    finite-sample bias. lam = 0 -> pure acceptance maximisation (ignores uniformity).
    """
    if feasible_set is None:
        if cs is None or u is None:
            raise ValueError("need feasible_set or (cs, u)")
        from .baselines import enumerate_feasible
        feasible_set = enumerate_feasible(u, cs)
    fcodes = np.unique(encode(feasible_set))
    M = int(fcodes.size)
    if M == 0:
        raise ValueError("feasible set is empty")
    lam = float(lam)

    def report(X):
        codes = encode(X)
        N = codes.size
        feas = codes[np.isin(codes, fcodes)]
        _, counts = np.unique(feas, return_counts=True)
        P_F = feas.size / N if N else 0.0
        chi2 = chi2_from_counts(counts, M)
        return {"P_F": P_F, "chi2": chi2, "n_feasible": int(feas.size), "N": int(N),
                "loss": -P_F + lam * chi2}

    def fn(X):
        return report(X)["loss"]

    def report_exact(probs):
        s = uniformity_stats_exact(probs, fcodes)
        s["loss"] = -s["P_F"] + lam * s["chi2"]
        return s

    def exact_fn(probs):
        return report_exact(probs)["loss"]

    return Objective(fn, exact_fn, name=f"uniformity(lam={lam})", report_fn=report,
                     report_exact_fn=report_exact,
                     meta={"lam": lam, "M": M, "fcodes": fcodes, "divergence": "chi2"})
