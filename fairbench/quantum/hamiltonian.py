"""Diagonal cost Hamiltonians on n qubits (no ancillas).

Conventions
-----------
* Binary variable x_i = (1 - Z_i) / 2, qubit i <-> asset i.
* Qiskit little-endian: built with ``SparsePauliOp.from_sparse_list`` so qubit
  indices are explicit (label char for qubit i sits at position n-1-i).
* ``diag_values(op, n)[b]`` is the energy of basis state b, where bit i of the
  integer b is x_i.

Penalty encodings (``penalty_operator``)
----------------------------------------
Cardinality is NOT penalised: the circuit (Dicke + XY mixer) fixes the Hamming
weight k = ``cs.cardinality``. "Exact" below means: on every weight-k state,
penalty == 0  <=>  constraint satisfied (penalty > 0 otherwise). Every term is
>= 0 on every basis state. All penalties are normalised so that one "unit" of
violation costs ~1 before the user weight lambda is applied.

* Exclusion(S):  P = sum_{i in S} x_i.  Exact (on all 2^n states). Counts held
  excluded assets.

* SectorCap(sector, c): let s = sum_{i in sector} x_i (m = sector size). On
  weight-k states s is confined to [lo, hi] = [max(0, k-(n-m)), min(k, m)]
  (k unknown -> [0, m]). Feasible reachable counts are lo..c' with
  c' = min(c, hi).
    - c' >= hi: constraint can never be violated -> zero operator (exact).
    - c  <  lo: never satisfiable -> P = (s - c)^2 > 0 everywhere reachable (exact).
    - otherwise P = (s - a)(s - b) / ((c'+1-a)(c'+1-b)), b = c', a = max(lo, c'-1).
      This quadratic in s (hence in x) is 0 at s in {a, b}, equals 1 at the
      first violation s = c'+1, and grows quadratically with overshoot.
      Exact iff c' - lo <= 1 (covers c in {0,1} and tight caps).
      If c' - lo >= 2 it is a SURROGATE: every violation is still strictly
      penalised, but feasible counts s <= c'-2 receive a spurious penalty
      (c'-s)(c'-s-1)/2. (A quadratic has at most two roots, so no ancilla-free
      quadratic can vanish on >= 3 feasible count values.)

* MinESG(m):  P = (1/(k*R)) * sum_i max(0, m - esg_i) x_i,  R = ptp(esg).
  CarbonCap(c): P = (1/(k*R)) * sum_i max(0, carbon_i - c) x_i, R = ptp(carbon).
  (k = cs.cardinality, or 1 if None.)  SURROGATE ("per-asset hinge"), linear:
    - P >= 0 everywhere; P == 0 => every held asset individually meets the
      threshold => the average constraint holds (sound: zero implies feasible
      on any non-empty state);
    - on weight-k states P*R >= max(0, m - avg_esg) (resp. avg_carbon - c),
      i.e. P upper-bounds the true normalised hinge violation;
    - NOT exact: feasible portfolios that mix in some below-threshold assets
      are still penalised. An exact one-sided average constraint needs slack
      qubits, which we deliberately avoid.

Weights: ``weights`` maps a constraint's index in ``cs.constraints`` (int) or
its class name (str, e.g. "SectorCap") to lambda; int keys take precedence;
default 1.0. ``penalty_exactness(cs, u)`` reports which encoding each
constraint received.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from ..constraints import (CarbonCap, Cardinality, ConstraintSet, Exclusion,
                           MinESG, SectorCap)
from ..data import Universe


# --------------------------------------------------------------------------
# QUBO -> Pauli
# --------------------------------------------------------------------------
def qubo_to_operator(n: int, const: float = 0.0, h=None, Q=None) -> SparsePauliOp:
    """Operator whose diagonal is const + h.x + x^T Q x (x_i = (1-Z_i)/2).

    Q may be any (n, n) matrix; its diagonal folds into the linear part
    (x_i^2 = x_i) and off-diagonal pairs are symmetrised.
    """
    h = np.zeros(n) if h is None else np.asarray(h, dtype=float).copy()
    if Q is not None:
        Q = np.asarray(Q, dtype=float)
        h = h + np.diag(Q)
        J = np.triu(Q, 1) + np.tril(Q, -1).T  # coefficient of x_i x_j, i<j
    else:
        J = np.zeros((n, n))
    c0 = float(const)
    terms: list[tuple[str, list[int], float]] = []
    zlin = np.zeros(n)
    # x_i = 1/2 - Z_i/2
    c0 += 0.5 * h.sum()
    zlin -= 0.5 * h
    # x_i x_j = (1 - Z_i - Z_j + Z_i Z_j)/4
    iu, ju = np.nonzero(np.triu(J, 1))
    for i, j in zip(iu, ju):
        w = J[i, j]
        c0 += 0.25 * w
        zlin[i] -= 0.25 * w
        zlin[j] -= 0.25 * w
        terms.append(("ZZ", [int(i), int(j)], 0.25 * w))
    for i in range(n):
        if zlin[i] != 0.0:
            terms.append(("Z", [i], zlin[i]))
    terms.append(("", [], c0))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()


def diag_values(op: SparsePauliOp, n: int | None = None) -> np.ndarray:
    """Energies of all 2^n basis states; entry b has x_i = bit i of b.

    Raises ValueError if op has any X/Y component (not diagonal).
    """
    n = op.num_qubits if n is None else int(n)
    if op.num_qubits != n:
        raise ValueError(f"op acts on {op.num_qubits} qubits, expected {n}")
    if np.any(op.paulis.x):
        raise ValueError("operator is not diagonal (has X/Y terms)")
    z = op.paulis.z.astype(np.int64)  # (T, n), column i = qubit i
    # phase from Pauli group (e.g. -I) folded into coefficients
    coeffs = op.coeffs * (-1j) ** op.paulis.phase
    b = np.arange(2 ** n)
    bits = ((b[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int64)
    parity = (bits @ z.T) & 1  # (2^n, T)
    vals = (1 - 2 * parity) @ coeffs
    return np.real_if_close(vals).real.astype(float)


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------
def objective_operator(u: Universe, risk_aversion: float) -> SparsePauliOp:
    """Mean-variance QUBO: diag = -mu.x + risk_aversion * x^T Sigma x."""
    return qubo_to_operator(u.n, 0.0, -np.asarray(u.mu, float),
                            float(risk_aversion) * np.asarray(u.cov, float))


# --------------------------------------------------------------------------
# Penalties
# --------------------------------------------------------------------------
def _sector_plan(c: SectorCap, u: Universe, k: int | None):
    """Return (mask, a, b, norm, kind) for a SectorCap; kind in
    {'trivial','infeasible','exact','surrogate'}."""
    mask = np.array([s == c.sector for s in u.sector], dtype=float)
    m = int(mask.sum())
    if k is None:
        lo, hi = 0, m
    else:
        lo, hi = max(0, k - (u.n - m)), min(k, m)
    cap = c.max_count
    if cap >= hi:
        return mask, None, None, None, "trivial"
    if cap < lo:
        return mask, float(cap), float(cap), 1.0, "infeasible"
    b = cap
    a = max(lo, cap - 1)
    norm = float((cap + 1 - a) * (cap + 1 - b))
    kind = "exact" if cap - lo <= 1 else "surrogate"
    return mask, float(a), float(b), norm, kind


def _avg_hinge(c, u: Universe, k: int | None):
    if isinstance(c, MinESG):
        v = np.asarray(u.esg_score, float)
        excess = np.maximum(0.0, c.min_avg_score - v)
    else:
        v = np.asarray(u.carbon, float)
        excess = np.maximum(0.0, v - c.max_avg)
    R = float(np.ptp(v)) or 1.0
    return excess / (R * (k if k else 1))


def _weight(weights: Mapping | None, idx: int, c) -> float:
    if not weights:
        return 1.0
    if idx in weights:
        return float(weights[idx])
    return float(weights.get(type(c).__name__, 1.0))


def penalty_operator(cs: ConstraintSet, u: Universe,
                     weights: Mapping | None = None) -> SparsePauliOp:
    """Sum of lambda_j * P_j over non-cardinality constraints (see module doc).

    Diagonal, n qubits, >= 0 on every basis state. Cardinality NOT penalised.
    """
    n, k = u.n, cs.cardinality
    const, h, Q = 0.0, np.zeros(n), np.zeros((n, n))
    for idx, c in enumerate(cs.constraints):
        if isinstance(c, Cardinality):
            continue
        lam = _weight(weights, idx, c)
        if lam == 0.0:
            continue
        if isinstance(c, Exclusion):
            if c.indices.size:
                np.add.at(h, c.indices, lam)
        elif isinstance(c, SectorCap):
            mask, a, b, norm, kind = _sector_plan(c, u, k)
            if kind == "trivial":
                continue
            # (s-a)(s-b) = s^2 - (a+b) s + ab, s^2 = x^T (mask mask^T) x
            w = lam / norm
            Q += w * np.outer(mask, mask)
            h += -w * (a + b) * mask
            const += w * a * b
        elif isinstance(c, (MinESG, CarbonCap)):
            h += lam * _avg_hinge(c, u, k)
        else:
            raise TypeError(f"no Hamiltonian encoding for {type(c).__name__}")
    return qubo_to_operator(n, const, h, Q)


def penalty_exactness(cs: ConstraintSet, u: Universe) -> dict[int, dict]:
    """Per-constraint encoding report keyed by index in ``cs.constraints``.

    Each value: {"type": class name, "encoding": one of
    'exact' | 'surrogate' | 'trivial' (never violable, zero op) |
    'infeasible' (never satisfiable on weight-k, all states penalised) |
    'not_penalised' (Cardinality, enforced by circuit), "note": str}.
    """
    k = cs.cardinality
    out: dict[int, dict] = {}
    for idx, c in enumerate(cs.constraints):
        name = type(c).__name__
        if isinstance(c, Cardinality):
            enc, note = "not_penalised", "enforced by the Dicke/XY circuit"
        elif isinstance(c, Exclusion):
            enc, note = "exact", "sum of held excluded assets"
        elif isinstance(c, SectorCap):
            *_, enc = _sector_plan(c, u, k)
            note = {"trivial": "cap >= max reachable count; zero operator",
                    "infeasible": "cap below min reachable count",
                    "exact": "two-root quadratic covers all feasible counts",
                    "surrogate": "violations penalised; feasible counts <= cap-2 "
                                 "also penalised"}[enc]
        elif isinstance(c, (MinESG, CarbonCap)):
            enc, note = "surrogate", ("per-asset hinge; zero => feasible, "
                                      "upper-bounds avg violation; not exact")
        else:
            enc, note = "unsupported", ""
        out[idx] = {"type": name, "encoding": enc, "note": note}
    return out
