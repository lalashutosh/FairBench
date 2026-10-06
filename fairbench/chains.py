"""MCMC chains with pluggable proposals + exact spectral analysis.

Target: uniform over the feasible set F (subset of weight-k states). A proposal
``propose(x, rng) -> y`` must be SYMMETRIC (Q[x,y] == Q[y,x]); then MH acceptance
is simply "y feasible". Proposals returning weight != k are rejected and counted
separately (``n_proposal_weight_violations``).

Asymmetric proposals (e.g. the tilted CTRW) need the MH ratio: use
``transition_matrix(Q, mask, mode="mh_min")``.

Exact analysis lives on the weight-k subspace in itertools.combinations order
(same as ``enumerate_feasible`` / quantum.proposal.subspace_basis).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .constraints import ConstraintSet
from .data import Universe

Proposal = Callable[[np.ndarray, np.random.Generator], np.ndarray]


# ---------------------------------------------------------------- proposals
def swap_proposal() -> Proposal:
    """Swap a uniformly random held asset with a uniformly random non-held one."""
    def propose(x, rng):
        y = x.copy()
        held, free = np.flatnonzero(x == 1), np.flatnonzero(x == 0)
        if held.size and free.size:
            y[held[rng.integers(held.size)]] = 0
            y[free[rng.integers(free.size)]] = 1
        return y
    return propose


def multi_swap_proposal(m: int) -> Proposal:
    """m sequential random swaps (each uniform held/non-held, intermediate states not
    checked). Q = S^m with S the swap matrix; m=1 is the swap proposal. Symmetric."""
    if m < 1:
        raise ValueError("m >= 1")
    one = swap_proposal()

    def propose(x, rng):
        y = x
        for _ in range(m):
            y = one(y, rng)
        return y
    return propose


def independent_uniform_proposal(n: int, k: int) -> Proposal:
    """Uniform random weight-k bitstring, independent of x (rejection sampling as MH)."""
    def propose(x, rng):
        y = np.zeros(n, np.uint8)
        y[rng.choice(n, k, replace=False)] = 1
        return y
    return propose


def matrix_proposal(Q: np.ndarray, basis_states: np.ndarray) -> Proposal:
    """Sample y ~ Q[idx(x), :]; basis_states (C, n) in subspace order (rows of Q)."""
    Q = np.asarray(Q, float)
    B = np.asarray(basis_states, np.uint8)
    if Q.shape != (len(B), len(B)):
        raise ValueError("Q must be (C, C) matching basis_states")
    cdf = np.cumsum(Q, axis=1)
    cdf /= cdf[:, -1:]
    index = {B[i].tobytes(): i for i in range(len(B))}

    def propose(x, rng):
        i = index[np.asarray(x, np.uint8).tobytes()]
        j = min(int(np.searchsorted(cdf[i], rng.random(), side="right")), len(B) - 1)
        return B[j].copy()
    return propose


def callable_proposal(fn: Callable) -> Proposal:
    """Passthrough for e.g. quantum-circuit proposals: fn(x, rng) -> y."""
    return fn


# -------------------------------------------------------------------- chain
@dataclass
class ChainResult:
    states: np.ndarray            # (m, n) uint8, recorded every `record_every` steps
    n_proposals: int
    n_accepted: int               # accepted moves incl. y == x (feasible proposals)
    n_proposal_weight_violations: int


def run_chain(propose: Proposal, x0: np.ndarray, steps: int, cs: ConstraintSet, u: Universe,
              seed: int | None = None, record_every: int = 1) -> ChainResult:
    """MH with uniform target on the feasible set, symmetric proposal: accept iff y has
    weight k and is feasible; otherwise stay."""
    if record_every < 1:
        raise ValueError("record_every >= 1")
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, np.uint8).copy()
    if not cs.check(x, u):
        raise ValueError("x0 is not feasible")
    k = int(x.sum())
    rec, acc, viol = [], 0, 0
    for t in range(1, steps + 1):
        y = np.asarray(propose(x, rng), np.uint8)
        if int(y.sum()) != k:
            viol += 1
        elif cs.check(y, u):
            x = y.copy()
            acc += 1
        if t % record_every == 0:
            rec.append(x.copy())
    states = np.array(rec, np.uint8).reshape(len(rec), x.size)
    return ChainResult(states, steps, acc, viol)


# ----------------------------------------------------------- exact analysis
def _basis(n: int, k: int) -> np.ndarray:
    comb = np.array(list(itertools.combinations(range(n), k)), dtype=np.int64).reshape(-1, k)
    B = np.zeros((len(comb), n), np.uint8)
    np.put_along_axis(B, comb, 1, 1)
    return B


def proposal_matrix_swap(n: int, k: int) -> np.ndarray:
    """(C,C): Q[x,y] = 1/(k(n-k)) for Hamming distance 2 (single swap), else 0."""
    B = _basis(n, k).astype(np.int32)
    D = B @ (1 - B).T                       # |x \ y|
    Q = (D == 1).astype(float)
    return Q / (k * (n - k))


def proposal_matrix_multi_swap(n: int, k: int, m: int) -> np.ndarray:
    """S^m (m sequential uniform swaps, self-returns included)."""
    return np.linalg.matrix_power(proposal_matrix_swap(n, k), m)


def proposal_matrix_independent(n: int, k: int) -> np.ndarray:
    C = math.comb(n, k)
    return np.full((C, C), 1.0 / C)


def proposal_matrix_ctrw(n: int, k: int, t: float, diag=None, beta: float = 1.0,
                         rate: str = "metropolis", topology: str = "complete") -> np.ndarray:
    """Classical continuous-time random walk proposal on the swap graph, Q = expm(t L).

    This is the classical analogue of the QeMCMC proposal: same graph (``topology``
    "complete" = all single swaps, i.e. the swap graph; "ring" = hops along ring edges, the
    QeMCMC ring mixer graph), same time parameter t, and -- when ``diag`` is given -- the
    SAME energy information (e.g. ``quantum.proposal.subspace_diagonal``) used as a tilt.
    Generator: L[x,y] = A[x,y] * r(beta (d_y - d_x)) for y != x (A = 0/1 adjacency, rate 1
    per edge untilted), L[x,x] = -sum_y L[x,y], with
        rate="metropolis": r(D) = min(1, exp(-D));  rate="barker": r(D) = 1/(1+exp(D)).
    diag=None (or beta=0) -> plain untilted CTRW: Q symmetric, doubly stochastic.
    Tilted: L is reversible w.r.t. pi ~ exp(-beta d), so Q is NOT symmetric (Q[x,y]/Q[y,x] =
    pi_y/pi_x); use ``transition_matrix(Q, mask, mode="mh_min")`` for the MH kernel with
    the uniform target. Computed exactly via the symmetrised generator
    D^1/2 L D^-1/2 (eigh). Basis = combinations order (``_basis``)."""
    B = _basis(n, k)
    C = len(B)
    if topology == "complete":
        Bi = B.astype(np.int32)
        A = (Bi @ (1 - Bi).T == 1).astype(float)
    elif topology == "ring":
        from .quantum.proposal import subspace_hamiltonian
        A = -subspace_hamiltonian(n, k, "ring")
    else:
        raise ValueError(f"unknown topology {topology!r}")
    if diag is None or beta == 0.0:
        d = np.zeros(C)
    else:
        d = np.asarray(diag, float)
        if d.shape != (C,):
            raise ValueError(f"diag must have shape ({C},)")
    D = beta * (d[None, :] - d[:, None])          # beta (d_y - d_x)
    if rate == "metropolis":
        R = np.exp(-np.maximum(D, 0.0))
    elif rate == "barker":
        R = 0.5 * (1.0 - np.tanh(D / 2.0))         # 1/(1+e^D), overflow-safe
    else:
        raise ValueError(f"unknown rate {rate!r}")
    L = A * R
    np.fill_diagonal(L, 0.0)
    out = L.sum(1)
    # S = D^1/2 L D^-1/2, pi ~ exp(-beta d): S[x,y] = L[x,y] exp(beta (d_y - d_x) / 2),
    # symmetric by detailed balance (pi_x L[x,y] = pi_y L[y,x])
    S = L * np.exp(np.clip(D / 2.0, -700, 700))
    S[A == 0] = 0.0
    if not np.allclose(S, S.T, atol=1e-9):
        raise AssertionError("symmetrised generator not symmetric (detailed balance broken)")
    S = 0.5 * (S + S.T) - np.diag(out)
    w, V = np.linalg.eigh(S)
    E = (V * np.exp(np.minimum(w, 0.0) * t)) @ V.T
    Q = E * np.exp(np.clip(-D / 2.0, -700, 700))  # Q[x,y] = E[x,y] exp(-beta(d_y-d_x)/2)
    Q = np.clip(Q, 0.0, None)
    Q /= Q.sum(1, keepdims=True)
    if diag is None or beta == 0.0:
        Q = 0.5 * (Q + Q.T)
    return Q


def transition_matrix(Q: np.ndarray, feasible_mask: np.ndarray,
                      mode: str = "assert_symmetric", sym_atol: float = 1e-9) -> np.ndarray:
    """MH kernel on F with the UNIFORM target; rejected mass on the diagonal.

    mode="assert_symmetric" (default): requires Q == Q.T (within sym_atol), else ValueError
        -- an asymmetric Q with "accept iff feasible" would NOT have a uniform stationary
        distribution. P[x,y] = Q[x,y] (y != x feasible).
    mode="mh_min": any row-stochastic Q; P[x,y] = min(Q[x,y], Q[y,x]) for y != x, i.e.
        acceptance min(1, Q[y,x]/Q[x,y]) -- a valid MH kernel for the uniform target
        (identical to the default for symmetric Q).
    Diagonal = 1 - off-diagonal row sum (self-proposals + rejections)."""
    Q = np.asarray(Q, float)
    mask = np.asarray(feasible_mask, bool)
    if Q.shape[0] != Q.shape[1] or mask.shape != (Q.shape[0],):
        raise ValueError("Q (C,C) and feasible_mask (C,) shape mismatch")
    if not np.allclose(Q.sum(1), 1.0, atol=1e-9):
        raise ValueError("Q rows must sum to 1")
    Qf = Q[np.ix_(mask, mask)]
    if mode == "assert_symmetric":
        asym = float(np.abs(Q - Q.T).max()) if Q.size else 0.0
        if asym > sym_atol:
            raise ValueError(f"Q is not symmetric (max |Q-Q^T| = {asym:.3g}); use mode='mh_min'")
        P = Qf.copy()
    elif mode == "mh_min":
        P = np.minimum(Qf, Qf.T)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    np.fill_diagonal(P, 0.0)
    P[np.diag_indices_from(P)] = 1.0 - P.sum(1)
    assert np.allclose(P.sum(1), 1.0)
    return P


def swap_component_labels(n: int, k: int, feasible_mask: np.ndarray) -> np.ndarray:
    """(M,) swap-connected component label of each feasible state (mask order)."""
    from scipy.sparse.csgraph import connected_components
    mask = np.asarray(feasible_mask, bool)
    A = proposal_matrix_swap(n, k)[np.ix_(mask, mask)] > 0
    return connected_components(A, directed=False)[1]


def move_stats(Q: np.ndarray, feasible_mask: np.ndarray, components=None) -> dict:
    """Per-step move statistics of the MH chain (mode "mh_min", uniform target), averaged
    over x ~ uniform on F:
      self_proposal      E[Q(x,x)]
      feasible_proposal  E[sum_{y in F} Q(x,y)]   (incl. y = x)
      move_rate          E[sum_{y in F, y != x} P(x,y)]   (accepted real moves)
      between_island     E[sum_{y in F, comp(y) != comp(x)} P(x,y)] (needs components:
                         (M,) labels over feasible states, e.g. swap_component_labels)."""
    Q = np.asarray(Q, float)
    mask = np.asarray(feasible_mask, bool)
    Qf = Q[np.ix_(mask, mask)]
    P = transition_matrix(Q, mask, mode="mh_min")
    off = P - np.diag(np.diag(P))
    out = dict(self_proposal=float(np.diag(Qf).mean()),
               feasible_proposal=float(Qf.sum(1).mean()),
               move_rate=float(off.sum(1).mean()))
    if components is not None:
        lab = np.asarray(components)
        if lab.shape != (int(mask.sum()),):
            raise ValueError("components must be (M,) labels over feasible states")
        out["between_island"] = float((off * (lab[:, None] != lab[None, :])).sum(1).mean())
    return out


def feasible_mask(u: Universe, cs: ConstraintSet) -> np.ndarray:
    """(C,) bool over the weight-k subspace in combinations order."""
    k = cs.cardinality
    return np.asarray(cs.check_batch(_basis(u.n, k), u), bool)


def _stationary(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of P for the eigenvalue closest to 1, normalised to sum 1."""
    w, V = np.linalg.eig(P.T)
    v = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    return v / v.sum()


def spectral_gap(P: np.ndarray, tol: float = 1e-10) -> float:
    """Absolute spectral gap 1 - max{|lambda| : lambda != the stationary eigenvalue 1}
    (= 1 - max(|lambda_2|, |lambda_min|) for reversible P); 0 if disconnected (eigenvalue 1
    repeated) or periodic (|lambda| = 1).

    Symmetric P: eigvalsh. Otherwise the stationary pi is computed; if P is reversible w.r.t.
    a strictly positive pi, the spectrum is taken from the symmetric similarity
    D^1/2 P D^-1/2 (D = diag pi; real, accurate); else from the complex eigenvalues, sorted
    by MODULUS, removing the one closest to 1."""
    P = np.asarray(P, float)
    if len(P) < 2:
        return 1.0
    if np.allclose(P, P.T, atol=1e-12):
        ev = np.linalg.eigvalsh((P + P.T) / 2)
    else:
        pi = _stationary(P)
        ev = None
        if pi.min() > 1e-12 * pi.max():
            F = pi[:, None] * P
            if np.allclose(F, F.T, atol=1e-12):
                s = np.sqrt(pi)
                Sm = (s[:, None] * P) / s[None, :]
                ev = np.linalg.eigvalsh((Sm + Sm.T) / 2)
        if ev is None:
            ev = np.linalg.eigvals(P)
    ev = np.asarray(ev)
    i1 = int(np.argmin(np.abs(ev - 1.0)))
    rest = np.delete(ev, i1)
    gap = 1.0 - float(np.abs(rest).max())
    return 0.0 if gap < tol else gap


def relaxation_time(P: np.ndarray) -> float:
    g = spectral_gap(P)
    return math.inf if g <= 0 else 1.0 / g


def mixing_time_tv(P: np.ndarray, eps: float = 0.25, x0: int | None = None,
                   max_steps: int = 100_000, pi: np.ndarray | None = None) -> float:
    """Steps until TV(P^t[x0,.], pi) <= eps (worst start if x0 is None); inf if cap hit.
    pi defaults to uniform (correct for every kernel built by ``transition_matrix``).
    Distribution rows are propagated by repeated multiplication."""
    M = len(P)
    D = np.eye(M) if x0 is None else np.eye(M)[[x0]]
    target = np.full(M, 1.0 / M) if pi is None else np.asarray(pi, float)
    for t in range(max_steps + 1):
        if 0.5 * np.abs(D - target).sum(1).max() <= eps:
            return float(t)
        D = D @ P
    return math.inf


# ------------------------------------------------------------- cost model
def min_trotter_steps(t: float, eps: float, n: int, k: int, topology: str = "ring",
                      alpha: float = 0.0, pen_op=None, h=None, r_max: int = 64) -> int:
    """Empirical: smallest palindromic-Trotter step count r with
    max_x TV(Q_trotter[x,.], Q_exact[x,.]) <= eps (full 2^n unitary -> n <= ~12). pen_op
    as in ``trotter_proposal_matrix`` (SparsePauliOp). Returns r_max + 1 if not reached."""
    from .quantum.proposal import (exact_proposal_matrix, subspace_hamiltonian,
                                   trotter_proposal_matrix)
    Qx = exact_proposal_matrix(t, subspace_hamiltonian(n, k, topology, alpha, pen_op, h))
    for r in range(1, r_max + 1):
        Qt = trotter_proposal_matrix(t, r, n, k, topology, alpha, pen_op, h)
        if 0.5 * np.abs(Qt - Qx).sum(1).max() <= eps:
            return r
    return r_max + 1


def proposal_cost(kind: str, **p) -> float:
    """Cost of ONE proposal in swap-equivalents (one elementary local move = one random
    swap classically = one two-qubit XY (partial-swap) gate on the quantum side). This is a
    bookkeeping CONVENTION, not a wall-clock model; report gaps per step AND per cost.

      "per_step"          1 (every proposal counts the same).
      "swap"              1.
      "multi_swap"        m                                (p: m)
      "independent"       k   (k random index draws)       (p: k)
      "ctrw"              t * degree, degree default k(n-k) (p: t, n, k[, degree])
                          = expected number of jumps of the UNTILTED walk (rate 1 per swap
                          edge); Metropolis tilting only thins jumps, so this is an upper
                          bound for the tilted walk. Pass degree=<number of ring edges
                          with differing bits, <= n> for topology "ring".
      "qemcmc"            n_trotter * gates_per_step (p: n_edges, and n_trotter or
                          (t, eps, H_norm)); gates_per_step = 2*n_edges - 1 two-qubit XY
                          gates (palindromic order: edges + reversed edges, middle pair
                          merged) + p.get("diag_2q_per_step", 0) (e.g. ZZ count of a QUBO
                          surrogate; an "indicator" diagonal needs a feasibility oracle,
                          whose cost must be passed here explicitly).
                          If n_trotter is not given: second-order bound
                          r = ceil((H_norm * t)^{3/2} / sqrt(eps)) (from error <= (||H|| t)^3
                          / r^2 up to constants; pessimistic). For a tighter value use
                          ``min_trotter_steps`` at small n and pass n_trotter.
    """
    if kind in ("per_step", "swap"):
        return 1.0
    if kind == "multi_swap":
        return float(p["m"])
    if kind == "independent":
        return float(p["k"])
    if kind == "ctrw":
        deg = p.get("degree")
        if deg is None:
            deg = p["k"] * (p["n"] - p["k"])
        return float(p["t"]) * float(deg)
    if kind == "qemcmc":
        r = p.get("n_trotter")
        if r is None:
            r = max(1, math.ceil((float(p["H_norm"]) * float(p["t"])) ** 1.5 / math.sqrt(float(p["eps"]))))
        per = max(2 * int(p["n_edges"]) - 1, 0) + float(p.get("diag_2q_per_step", 0))
        return float(r) * per
    raise ValueError(f"unknown kind {kind!r}")
