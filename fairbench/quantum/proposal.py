"""Quantum-enhanced MCMC proposal (Layden et al., Nature 2023 style), weight-k subspace.

Physics contract
----------------
From a feasible x (Hamming weight k) evolve |x> under

    H = -sum_{(i,j) in edges} (X_i X_j + Y_i Y_j)/2  +  alpha * diag(pen)  +  sum_i h_i Z_i

for time t and measure -> y.  (XX+YY)/2 on edge (i,j) is the hopping term
|..1_i..0_j..><..0_i..1_j..| + h.c. with amplitude 1, so in the computational basis
the off-diagonal entries of H are -1 between states differing by one swap along an edge.
H is real and conserves Hamming weight, so everything lives in the weight-k subspace of
dimension C(n, k).

Basis order
-----------
``subspace_basis(n, k)`` uses ``itertools.combinations(range(n), k)`` order, i.e. the same
order in which ``baselines.enumerate_feasible`` iterates (it then filters by feasibility,
so feasible rows map into subspace indices via ``codes_to_index``). Integer code of a
bitstring: bit i = x_i (Qiskit little-endian, same as ``hamiltonian.diag_values``).

Exact path:  Q[x, y] = |<y| exp(-iHt) |x>|^2 (doubly stochastic, symmetric since H real).
Circuit path: palindromic second-order Trotter step
    S(dt) = D(dt/2) . E_1(dt/2) ... E_m(dt/2) . E_m(dt/2) ... E_1(dt/2) . D(dt/2)
with every factor a symmetric matrix, so S^T = S and U = S^r has U^T = U exactly,
hence Q_trotter is exactly symmetric too. Edge factor E(tau) = exp(+i tau (XX+YY)/2)
= XXPlusYYGate(-2 tau, 0) (Qiskit: XXPlusYYGate(theta, 0) = exp(-i theta/4 (XX+YY))).
Diagonal factor D(tau) = exp(-i tau (alpha*pen + sum h_i Z_i)) via ``ansatz.cost_layer``.
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Operator, SparsePauliOp

from .ansatz import _edges, cost_layer
from .hamiltonian import diag_values

__all__ = [
    "subspace_basis", "codes_to_index", "states_to_codes", "subspace_hamiltonian",
    "exact_proposal_matrix", "proposal_from_Q", "diag_operator", "trotter_circuit",
    "trotter_unitary", "trotter_proposal_matrix", "circuit_proposal", "circuit_proposal_batch",
    "subspace_diagonal", "penalty_separation",
]


# --------------------------------------------------------------------------
# Basis
# --------------------------------------------------------------------------
def subspace_basis(n: int, k: int) -> tuple[np.ndarray, dict[int, int]]:
    """Weight-k basis in itertools.combinations(range(n), k) order.

    Returns (states (C, n) uint8 with column i = x_i, index: {integer code -> row}),
    where code = sum_i x_i 2^i.
    """
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need n>=1, 0<=k<=n; got n={n}, k={k}")
    combos = list(itertools.combinations(range(n), k))
    C = len(combos)
    states = np.zeros((C, n), dtype=np.uint8)
    if k:
        idx = np.asarray(combos, dtype=np.int64).reshape(C, k)
        np.put_along_axis(states, idx, 1, axis=1)
    codes = states_to_codes(states)
    return states, {int(c): r for r, c in enumerate(codes)}


def states_to_codes(X) -> np.ndarray:
    """(m, n) or (n,) 0/1 array -> int64 codes (bit i = x_i)."""
    X = np.atleast_2d(np.asarray(X)).astype(np.int64)
    return X @ (np.int64(1) << np.arange(X.shape[1], dtype=np.int64))


def codes_to_index(X, index: dict[int, int]) -> np.ndarray:
    """Map bitstrings (m, n) (or (n,)) to subspace rows. KeyError if a row is not weight k."""
    return np.array([index[int(c)] for c in states_to_codes(X)], dtype=np.int64)


# --------------------------------------------------------------------------
# Hamiltonian / exact Q
# --------------------------------------------------------------------------
def subspace_diagonal(kind: str, n: int, k: int, u, cs, pen_op=None) -> np.ndarray:
    """(C,) diagonal energy on the weight-k subspace (subspace_basis order), for use as
    ``subspace_hamiltonian(..., pen_op=<this vector>)`` (alpha multiplies it) or as the
    tilt of ``chains.proposal_matrix_ctrw``.

    kind:
      "surrogate"  P1 QUBO penalty_operator diagonal (pen_op if given, else
                   ``penalty_operator(cs, u)``), divided by its std over the subspace.
                   Realisable as a 2-local diagonal (ZZ/Z) Hamiltonian; it separates
                   feasible from infeasible states poorly (see penalty_separation).
      "indicator"  1[x infeasible] (exact). NOT a QUBO: implementing exp(-i a t 1[infeasible])
                   needs a reversible feasibility oracle (compute the feasibility bit,
                   phase it, uncompute) -- the same oracle amplitude amplification needs.
                   Its circuit cost is that oracle's, not a few ZZ rotations.
      "violation"  postprocess._raw_violation_batch (normalised total violation, 0 on F),
                   divided by its std. Also oracle-type (arithmetic on averages), not a QUBO.
    """
    states, _ = subspace_basis(n, k)
    if kind == "surrogate":
        if pen_op is None:
            from .hamiltonian import penalty_operator
            pen_op = penalty_operator(cs, u)
        full = diag_values(pen_op, n) if isinstance(pen_op, SparsePauliOp) else np.asarray(pen_op, float)
        if full.shape != (2 ** n,):
            raise ValueError("pen_op must be a SparsePauliOp or a length-2^n diagonal")
        d = full[states_to_codes(states)]
    elif kind == "indicator":
        d = (~np.asarray(cs.check_batch(states, u), bool)).astype(float)
    elif kind == "violation":
        from ..postprocess import _raw_violation_batch
        d = _raw_violation_batch(states, cs, u)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    d = np.asarray(d, float)
    if kind in ("surrogate", "violation"):
        sd = d.std()
        if sd > 0:
            d = d / sd
    return d


def penalty_separation(diag, feasible_mask) -> dict:
    """How well a diagonal ranks infeasible above feasible states.
    auc = P(d[infeasible] > d[feasible]) (+0.5 * ties), over all pairs;
    frac_infeasible_le_max_feasible = fraction of infeasible states with d <= max_F d
    (states the diagonal cannot push above every feasible one)."""
    d = np.asarray(diag, float)
    m = np.asarray(feasible_mask, bool)
    if d.shape != m.shape:
        raise ValueError("diag and feasible_mask shape mismatch")
    f, g = d[m], d[~m]
    if f.size == 0 or g.size == 0:
        raise ValueError("need both feasible and infeasible states")
    # rank-based AUC (Mann-Whitney), O(C log C)
    fs = np.sort(f)
    lt = np.searchsorted(fs, g, side="left")
    le = np.searchsorted(fs, g, side="right")
    auc = float((lt + 0.5 * (le - lt)).sum() / (f.size * g.size))
    return dict(auc=auc, frac_infeasible_le_max_feasible=float(np.mean(g <= f.max())),
                n_feasible=int(f.size), n_infeasible=int(g.size))


def _diag_terms(n: int, states: np.ndarray, alpha: float, pen_op, h) -> np.ndarray:
    d = np.zeros(states.shape[0])
    if pen_op is not None and alpha != 0.0:
        pen_arr = None if isinstance(pen_op, SparsePauliOp) else np.asarray(pen_op, dtype=float)
        if pen_arr is not None and pen_arr.shape == (states.shape[0],):
            d += alpha * pen_arr  # subspace diagonal (e.g. from subspace_diagonal)
        elif isinstance(pen_op, SparsePauliOp):
            full = diag_values(pen_op, n)
            d += alpha * full[states_to_codes(states)]
        else:  # precomputed 2^n diagonal
            full = np.asarray(pen_op, dtype=float)
            if full.shape != (2 ** n,):
                raise ValueError("pen_op must be a SparsePauliOp, a length-2^n diagonal or a "
                                 "length-C(n,k) subspace diagonal")
            d += alpha * full[states_to_codes(states)]
    if h is not None:
        h = np.asarray(h, dtype=float)
        if h.shape != (n,):
            raise ValueError(f"h must have shape ({n},)")
        d += (1.0 - 2.0 * states.astype(float)) @ h  # Z_i = 1 - 2 x_i
    return d


def subspace_hamiltonian(n: int, k: int, topology: str = "ring", alpha: float = 0.0,
                         pen_op=None, h=None, sparse: bool = False):
    """H restricted to the weight-k subspace, (C, C) real symmetric, basis = subspace_basis.

    pen_op: diagonal SparsePauliOp on n qubits, a length-2^n array of diagonal values, or a
    length-C(n,k) subspace diagonal in subspace_basis order (e.g. ``subspace_diagonal``).
    h: optional (n,) longitudinal fields (coefficient of Z_i).
    sparse=True returns a scipy.sparse.csr_matrix.
    """
    states, index = subspace_basis(n, k)
    C = states.shape[0]
    codes = states_to_codes(states)
    rows, cols = [], []
    for i, j in _edges(n, topology):
        flip = (1 << i) | (1 << j)
        mask = states[:, i] != states[:, j]
        for r in np.flatnonzero(mask):
            rows.append(r)
            cols.append(index[int(codes[r] ^ flip)])
    d = _diag_terms(n, states, alpha, pen_op, h)
    import scipy.sparse as sp
    Hs = sp.coo_matrix((-np.ones(len(rows)), (rows, cols)), shape=(C, C)).tocsr()
    Hs = Hs + sp.diags(d)
    if sparse:
        return Hs.tocsr()
    H = Hs.toarray()
    assert np.isrealobj(H) and np.allclose(H, H.T), "H must be real symmetric"
    return H


def exact_proposal_matrix(t: float, H: np.ndarray) -> np.ndarray:
    """Q[x, y] = |<y| exp(-iHt) |x>|^2 via eigh (H real symmetric)."""
    H = np.asarray(H.toarray() if hasattr(H, "toarray") else H, dtype=float)
    assert np.allclose(H, H.T), "H must be symmetric"
    w, V = np.linalg.eigh(H)
    U = (V * np.exp(-1j * w * t)) @ V.T  # U[y, x], symmetric
    Q = np.abs(U.T) ** 2
    Q = 0.5 * (Q + Q.T)  # remove round-off asymmetry (exact Q is symmetric)
    assert np.allclose(Q.sum(axis=1), 1.0, atol=1e-9), "rows must sum to 1"
    return Q


def proposal_from_Q(Q: np.ndarray, rng=None) -> Callable[[int], int]:
    """Sampler y_index ~ Q[x_index, :]. rng: Generator or seed."""
    rng = np.random.default_rng(rng)
    cdf = np.cumsum(np.asarray(Q, dtype=float), axis=1)
    cdf[:, -1] = 1.0

    def propose(x_index: int) -> int:
        return int(np.searchsorted(cdf[int(x_index)], rng.random(), side="right"))

    return propose


# --------------------------------------------------------------------------
# Trotter circuit
# --------------------------------------------------------------------------
def diag_operator(n: int, alpha: float = 0.0, pen_op: SparsePauliOp | None = None,
                  h=None) -> SparsePauliOp | None:
    """alpha*pen_op + sum_i h_i Z_i as a SparsePauliOp (None if zero)."""
    parts = []
    if pen_op is not None and alpha != 0.0:
        if not isinstance(pen_op, SparsePauliOp):
            raise TypeError("circuit path needs pen_op as a SparsePauliOp")
        parts.append(float(alpha) * pen_op)
    if h is not None and np.any(np.asarray(h) != 0):
        parts.append(SparsePauliOp.from_sparse_list(
            [("Z", [i], float(hi)) for i, hi in enumerate(np.asarray(h, float)) if hi != 0], n))
    if not parts:
        return None
    op = parts[0]
    for p in parts[1:]:
        op = op + p
    return op.simplify()


def _evolution(n: int, t: float, n_trotter: int, topology: str, alpha: float,
               pen_op, h) -> QuantumCircuit:
    if n_trotter < 1:
        raise ValueError("n_trotter must be >= 1")
    dt = float(t) / n_trotter
    edges = _edges(n, topology)
    dop = diag_operator(n, alpha, pen_op, h)
    half_d = cost_layer(dop, dt / 2) if dop is not None else None
    qc = QuantumCircuit(n, name="QeMCMCEvo")
    for _ in range(n_trotter):
        if half_d is not None:
            qc.compose(half_d, inplace=True)
        for e in edges + edges[::-1]:
            # exp(+i (dt/2) (XX+YY)/2) = XXPlusYY(theta) with -theta/4 = dt/4
            qc.append(XXPlusYYGate(-dt, 0.0), list(e))
        if half_d is not None:
            qc.compose(half_d, inplace=True)
    return qc


def trotter_circuit(n: int, x, t: float, n_trotter: int, topology: str = "ring",
                    alpha: float = 0.0, pen_op=None, h=None) -> QuantumCircuit:
    """Prepare |x> (X gates; x=None skips prep), then the palindromic Trotter evolution.
    No measurements."""
    qc = QuantumCircuit(n, name="QeMCMCProposal")
    if x is not None:
        x = np.asarray(x).ravel()
        if x.shape != (n,):
            raise ValueError(f"x must have length {n}")
        for i in np.flatnonzero(x):
            qc.x(int(i))
    qc.compose(_evolution(n, t, n_trotter, topology, alpha, pen_op, h), inplace=True)
    return qc


def trotter_unitary(n: int, t: float, n_trotter: int, topology: str = "ring",
                    alpha: float = 0.0, pen_op=None, h=None) -> np.ndarray:
    """Full 2^n unitary of the evolution part (U[out_code, in_code])."""
    return Operator(_evolution(n, t, n_trotter, topology, alpha, pen_op, h)).data


def trotter_proposal_matrix(t: float, n_trotter: int, n: int, k: int, topology: str = "ring",
                            alpha: float = 0.0, pen_op=None, h=None) -> np.ndarray:
    """Exact Q of the Trotter circuit restricted to the weight-k subspace (n <= ~12)."""
    states, _ = subspace_basis(n, k)
    codes = states_to_codes(states)
    U = trotter_unitary(n, t, n_trotter, topology, alpha, pen_op, h)
    Usub = U[np.ix_(codes, codes)]  # [y, x]
    Q = np.abs(Usub.T) ** 2
    return Q


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def circuit_proposal(x, t: float, n_trotter: int, backend: str = "aer_statevector",
                     seed: int | None = None, topology: str = "ring", alpha: float = 0.0,
                     pen_op=None, h=None, shots: int = 1, **kw) -> np.ndarray:
    """Run the proposal circuit from x; returns y (n,) uint8 (or (shots, n) if shots > 1).
    kw are passed to backends.sample (e.g. bond_dim for aer_mps)."""
    from ..backends import sample
    x = np.asarray(x, dtype=np.uint8).ravel()
    qc = trotter_circuit(x.size, x, t, n_trotter, topology, alpha, pen_op, h)
    Y = sample(qc, None, shots, backend=backend, seed=seed, **kw)
    return Y[0] if shots == 1 else Y


def circuit_proposal_batch(X, t: float, n_trotter: int, backend: str = "aer_statevector",
                           seed: int | None = None, topology: str = "ring", alpha: float = 0.0,
                           pen_op=None, h=None, **kw) -> np.ndarray:
    """One 1-shot proposal per row of X (m, n) -> Y (m, n) uint8.

    Builds one template circuit with RX(pi * x_i) prep parameters, transpiles it once,
    then binds each x and runs all experiments in a single simulator call.
    Simulator options mirror backends.sample.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    from ..backends import bitstrings_to_array

    X = np.atleast_2d(np.asarray(X, dtype=np.uint8))
    m, n = X.shape
    if backend not in ("aer_statevector", "aer_mps"):
        raise NotImplementedError(f"batch proposals only for aer backends, got {backend!r}")
    th = ParameterVector("prep", n)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.rx(th[i], i)  # RX(pi)|0> = -i|1>; global phase irrelevant
    qc.compose(_evolution(n, t, n_trotter, topology, alpha, pen_op, h), inplace=True)
    qc.measure_all()
    if backend == "aer_statevector":
        sim = AerSimulator(method="statevector", seed_simulator=seed)
    else:
        opts = {}
        if kw.get("bond_dim") is not None:
            opts["matrix_product_state_max_bond_dimension"] = int(kw["bond_dim"])
        if kw.get("truncation_threshold") is not None:
            opts["matrix_product_state_truncation_threshold"] = float(kw["truncation_threshold"])
        sim = AerSimulator(method="matrix_product_state", seed_simulator=seed, **opts)
    tqc = transpile(qc, sim, seed_transpiler=seed)
    params = list(th)
    circs = [tqc.assign_parameters(dict(zip(params, math.pi * row.astype(float))))
             for row in X]
    res = sim.run(circs, shots=1, memory=True).result()
    return np.concatenate([bitstrings_to_array(res.get_memory(i), n) for i in range(m)], axis=0)
