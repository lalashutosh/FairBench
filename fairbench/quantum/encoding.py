"""Stage 2 quantum encoding: the weight grid as binary variables and a QUBO.

Variables, per eligible asset i (``WeightGrid`` notation, units of 1/``units``):
    y_i        1 if the asset is held
    q_{i,b}    b = 0..B-1, plain binary:  a_i = sum_b 2^b q_{i,b}
    weight     w_i = (min_units * y_i + a_i) / units
plus slack variables where an inequality needs them. Energy of a bitstring z:

    E(z) = penalty_weight * P(z) + objective_scale * TE(w(z))^2

P(z) >= 0 is a sum of squared or product terms, all QUADRATIC in z:
    budget       (sum_i (min_units y_i + a_i) - units)^2
    link         sum_{i,b} q_{i,b} (1 - y_i)            no weight bits on an unheld asset
    position cap (a_i + t_i - R)^2                      only if R = max_units - min_units is
                                                        not of the form 2^B - 1
    holdings     (sum_i y_i - k)^2, or with a slack variable for a range
    group weight (sum_{i in G} units_i + t - cap)^2     for ``WeightSum`` rules
Tracking error and volatility are quadratic forms in w, hence in z: they enter as the
objective term, which is exactly a QUBO energy. They are NOT linear and are never encoded
as a linear rule; a hard risk cap stays a classical check on the decoded weights.

BIJECTION. Sampling needs more than optimisation does: uniform over zero-penalty bitstrings
is uniform over portfolios only if each feasible portfolio has exactly ONE zero-penalty
bitstring. Every slack here therefore has a unique representation (plain binary when its
range is 2^T - 1, otherwise one-hot with an at-most-one penalty), and ``QuboModel.terms``
says for each term whether it is exact. ``check_encoding`` verifies both properties by
brute force on small instances. Rules that are not encoded are listed in
``QuboModel.unencoded`` and must be enforced by filtering the decoded samples.

Samplers on the encoding, for comparison with exact distributions on small cases:
``simulated_annealing`` (classical baseline on the same energy) and ``grover_circuit`` /
``grover_sample`` (amplitude amplification of the zero-penalty states, which is exactly
uniform over them). Neither is a claim of advantage; see ``portfolio.exact`` for the metrics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..constraints import HoldingsRange, PositionBound, WeightRuleSet, WeightSum
from ..portfolio.weights import WeightGrid


def _is_pow2_minus_1(d: int) -> bool:
    return d >= 0 and (d + 1) & d == 0


@dataclass
class QuboModel:
    """E(z) = offset + lin.z + sum_{i<j} quad[i, j] z_i z_j, split into a penalty part and
    an objective part. ``names[v]`` describes variable v; ``terms`` reports every term."""
    grid: WeightGrid
    names: list[str]
    bits: int
    pen_lin: np.ndarray
    pen_quad: np.ndarray
    pen_offset: float
    obj_lin: np.ndarray
    obj_quad: np.ndarray
    obj_offset: float
    penalty_weight: float
    terms: list[dict] = field(default_factory=list)
    unencoded: list[str] = field(default_factory=list)
    slack: list[dict] = field(default_factory=list)  # {"vars": [...], "coefs": [...], "kind", "of"}

    @property
    def n_qubits(self) -> int:
        return len(self.names)

    @property
    def n_data_qubits(self) -> int:
        return self.grid.n_allowed * (1 + self.bits)

    @staticmethod
    def _eval(Z, lin, quad, off):
        Z = np.atleast_2d(np.asarray(Z)).astype(float)
        return off + Z @ lin + np.einsum("mi,ij,mj->m", Z, quad, Z)

    def penalty(self, Z: np.ndarray) -> np.ndarray:
        """(m,) unweighted penalty P(z); 0 iff every encoded rule holds."""
        return self._eval(Z, self.pen_lin, self.pen_quad, self.pen_offset)

    def objective(self, Z: np.ndarray) -> np.ndarray:
        return self._eval(Z, self.obj_lin, self.obj_quad, self.obj_offset)

    def energy(self, Z: np.ndarray) -> np.ndarray:
        return self.penalty_weight * self.penalty(Z) + self.objective(Z)

    def matrices(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(lin, quad strictly upper, offset) of the full energy."""
        return (self.penalty_weight * self.pen_lin + self.obj_lin,
                self.penalty_weight * self.pen_quad + self.obj_quad,
                self.penalty_weight * self.pen_offset + self.obj_offset)

    def units(self, Z: np.ndarray) -> np.ndarray:
        """(m, n) integer units min_units*y_i + a_i read off the data bits (whether or not
        the bitstring is feasible)."""
        Z = np.atleast_2d(np.asarray(Z)).astype(np.int64)
        na, B, g = self.grid.n_allowed, self.bits, self.grid
        y = Z[:, :na]
        a = np.zeros_like(y)
        for b in range(B):
            a += Z[:, na + b * na: na + (b + 1) * na] << b
        Q = np.zeros((Z.shape[0], g.n), dtype=np.int64)
        Q[:, g.allowed_indices] = g.min_units * y + a
        return Q

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """(m, n) weights units / grid.units. Rows only sum to 1 where the budget holds."""
        return self.units(Z) / self.grid.units

    def encode(self, Q: np.ndarray) -> np.ndarray:
        """(m, n_qubits) the unique zero-penalty bitstring of each feasible grid point
        (slack variables set to their forced values)."""
        Q = np.atleast_2d(np.asarray(Q)).astype(np.int64)
        g, na, B = self.grid, self.grid.n_allowed, self.bits
        q = Q[:, g.allowed_indices]
        Z = np.zeros((Q.shape[0], self.n_qubits), dtype=np.uint8)
        held = q > 0
        Z[:, :na] = held
        a = np.where(held, q - g.min_units, 0)
        for b in range(B):
            Z[:, na + b * na: na + (b + 1) * na] = (a >> b) & 1
        for s in self.slack:
            val = s["value"](Q)
            if s["kind"] == "binary":
                for j, v in enumerate(s["vars"]):
                    Z[:, v] = (val >> j) & 1
            else:  # one-hot over the values 1..D
                for j, v in enumerate(s["vars"]):
                    Z[:, v] = val == j + 1
        return Z

    def to_operator(self):
        """Diagonal SparsePauliOp of the full energy (x = (1 - Z)/2 per qubit)."""
        from .hamiltonian import qubo_to_operator
        lin, quad, off = self.matrices()
        return qubo_to_operator(self.n_qubits, off, lin, quad)


class _Builder:
    def __init__(self, n: int):
        self.lin, self.quad, self.off = np.zeros(n), np.zeros((n, n)), 0.0

    def grow(self, extra: int):
        n = self.lin.size + extra
        lin, quad = np.zeros(n), np.zeros((n, n))
        lin[:self.lin.size], quad[:self.lin.size, :self.lin.size] = self.lin, self.quad
        self.lin, self.quad = lin, quad

    def product(self, i: int, j: int, w: float):
        if i == j:
            self.lin[i] += w
        else:
            self.quad[min(i, j), max(i, j)] += w

    def squared(self, coefs: dict[int, float], const: float, w: float = 1.0):
        """Add w * (sum_v coefs[v] z_v + const)^2, using z^2 = z."""
        items = list(coefs.items())
        for a, (i, ci) in enumerate(items):
            self.lin[i] += w * (ci * ci + 2 * const * ci)
            for j, cj in items[a + 1:]:
                self.product(i, j, 2 * w * ci * cj)
        self.off += w * const * const


def build_qubo(grid: WeightGrid, rules: WeightRuleSet | None = None, *, cov: np.ndarray | None = None,
               benchmark: np.ndarray | None = None, objective_scale: float = 0.0,
               penalty_weight: float = 1.0) -> QuboModel:
    """QUBO for the weight grid.

    Encoded exactly: the grid itself (budget, link, position cap, holdings count) and any
    ``WeightSum`` rule whose bounds are whole units. ``PositionBound`` / ``HoldingsRange``
    rules are part of the grid and must agree with it (ValueError otherwise). Every other
    rule (weighted averages, risk caps, selection rules) is listed in ``unencoded``.

    objective_scale > 0 adds objective_scale * (w - b)' cov (w - b); with
    objective_scale = 1 / (2 tau^2) the energy is the exponent of reference distribution D3.
    """
    na, M, m_units = grid.n_allowed, grid.units, grid.min_units
    R = grid.max_units - m_units
    B = 0 if R == 0 else int(R).bit_length()
    names = [f"y[{i}]" for i in grid.allowed_indices] + \
            [f"q[{i},{b}]" for b in range(B) for i in grid.allowed_indices]
    pen = _Builder(len(names))
    terms: list[dict] = []
    slack: list[dict] = []
    unencoded: list[str] = []

    def y(i: int) -> int:
        return i

    def q(i: int, b: int) -> int:
        return na + b * na + i

    def unit_coefs(members) -> dict[int, float]:
        c = {y(i): float(m_units) for i in members}
        c.update({q(i, b): float(1 << b) for i in members for b in range(B)})
        return c

    def new_slack(D: int, of: str, value) -> dict[int, float]:
        """Variables whose weighted sum takes each value 0..D in exactly one way."""
        if D == 0:
            return {}
        start = len(names)
        if _is_pow2_minus_1(D):
            T = (D + 1).bit_length() - 1
            names.extend(f"s[{of},{j}]" for j in range(T))
            pen.grow(T)
            coefs = {start + j: float(1 << j) for j in range(T)}
            kind = "binary"
        else:
            names.extend(f"s[{of}={j + 1}]" for j in range(D))
            pen.grow(D)
            coefs = {start + j: float(j + 1) for j in range(D)}
            for a in range(D):  # at most one of the one-hot variables
                for b in range(a + 1, D):
                    pen.product(start + a, start + b, 1.0)
            kind = "one_hot"
        slack.append({"vars": list(coefs), "kind": kind, "of": of, "value": value})
        return coefs

    pen.squared(unit_coefs(range(na)), -float(M))
    terms.append({"name": "budget", "kind": "penalty", "exact": True, "n_slack": 0,
                  "note": "weights sum to 1"})
    for i in range(na):
        for b in range(B):
            pen.lin[q(i, b)] += 1.0
            pen.product(q(i, b), y(i), -1.0)
    if B:
        terms.append({"name": "link", "kind": "penalty", "exact": True, "n_slack": 0,
                      "note": "no weight bits on an unheld asset"})
    if B and not _is_pow2_minus_1(R):
        before = len(names)
        allowed = grid.allowed_indices
        for i in range(na):
            col = int(allowed[i])
            sl = new_slack(R, f"cap{col}", lambda Q, col=col: np.where(Q[:, col] > 0, grid.max_units - Q[:, col], R))
            pen.squared({**{q(i, b): float(1 << b) for b in range(B)}, **sl}, -float(R))
        terms.append({"name": "position_cap", "kind": "penalty", "exact": True, "n_slack": len(names) - before,
                      "note": f"max_units - min_units = {R} is not 2^B - 1; choose it so to save these qubits"})

    k_lo_free = math.ceil(M / grid.max_units)
    k_hi_free = min(na, M // m_units)
    if grid.k_min > k_lo_free or grid.k_max < k_hi_free:
        before = len(names)
        lo, hi = max(grid.k_min, k_lo_free), min(grid.k_max, k_hi_free)
        allowed = grid.allowed_indices
        sl = new_slack(hi - lo, "holdings", lambda Q: hi - (Q[:, allowed] > 0).sum(axis=1))
        pen.squared({**{y(i): 1.0 for i in range(na)}, **sl}, -float(hi))
        terms.append({"name": "holdings", "kind": "penalty", "exact": True, "n_slack": len(names) - before,
                      "note": f"{lo} <= number of holdings <= {hi}"})

    pos = {int(a): i for i, a in enumerate(grid.allowed_indices)}
    for r_i, rule in enumerate([] if rules is None else rules.rules):
        label = f"{r_i}:{getattr(rule, 'label', '') or type(rule).__name__}"
        if isinstance(rule, PositionBound):
            if round(rule.lower * M) > m_units or round(rule.upper * M) < grid.max_units:
                raise ValueError(f"{label} is tighter than the grid's position bounds; put it in the grid")
            continue
        if isinstance(rule, HoldingsRange):
            if rule.k_min > grid.k_min or rule.k_max < grid.k_max:
                raise ValueError(f"{label} is tighter than the grid's holdings range; put it in the grid")
            continue
        if not isinstance(rule, WeightSum):
            unencoded.append(label)
            continue
        members = [pos[int(i)] for i in rule.indices if int(i) in pos]
        cols = [int(i) for i in rule.indices if int(i) in pos]
        top = min(M, len(members) * grid.max_units)
        before = len(names)
        if rule.upper is not None:
            cap = int(math.floor(rule.upper * M + 1e-9))
            if cap < top:
                sl = new_slack(cap, f"{label}<=", lambda Q, cols=cols, cap=cap: cap - Q[:, cols].sum(axis=1))
                pen.squared({**unit_coefs(members), **sl}, -float(cap))
        if rule.lower is not None:
            floor_ = int(math.ceil(rule.lower * M - 1e-9))
            if floor_ > 0:
                sl = new_slack(top - floor_, f"{label}>=",
                               lambda Q, cols=cols, floor_=floor_: Q[:, cols].sum(axis=1) - floor_)
                pen.squared({**unit_coefs(members), **{v: -c for v, c in sl.items()}}, -float(floor_))
        terms.append({"name": label, "kind": "penalty", "exact": True, "n_slack": len(names) - before,
                      "note": "group weight bound in whole units"})
    if rules is not None and rules.support_rules is not None:
        unencoded.append("support_rules")

    nq = len(names)
    obj = _Builder(nq)
    if objective_scale:
        if cov is None:
            raise ValueError("objective_scale needs cov")
        A = np.zeros((grid.n, nq))  # w = A z
        for i, col in enumerate(grid.allowed_indices):
            A[col, y(i)] = m_units / M
            for b in range(B):
                A[col, q(i, b)] = (1 << b) / M
        bvec = np.zeros(grid.n) if benchmark is None else np.asarray(benchmark, dtype=float)
        S = np.asarray(cov, dtype=float)
        G = A.T @ S @ A  # z' G z - 2 b' S A z + b' S b
        obj.lin = objective_scale * (np.diag(G) - 2.0 * (bvec @ S @ A))
        obj.quad = objective_scale * 2.0 * np.triu(G, 1)
        obj.off = objective_scale * float(bvec @ S @ bvec)
        terms.append({"name": "tracking_error_squared" if benchmark is not None else "variance",
                      "kind": "objective", "exact": True, "n_slack": 0,
                      "note": "quadratic form in the weights; a QUBO energy, not a linear rule"})
    return QuboModel(grid=grid, names=names, bits=B, pen_lin=pen.lin, pen_quad=pen.quad, pen_offset=pen.off,
                     obj_lin=obj.lin, obj_quad=obj.quad, obj_offset=obj.off, penalty_weight=float(penalty_weight),
                     terms=terms, unencoded=unencoded, slack=slack)


def all_bitstrings(n: int) -> np.ndarray:
    """(2^n, n) uint8; row b has z_i = bit i of b (qubit i <-> variable i, as in ``backends``)."""
    if n > 24:
        raise ValueError(f"2^{n} bitstrings is too many to enumerate")
    b = np.arange(1 << n, dtype=np.int64)
    return ((b[:, None] >> np.arange(n)[None, :]) & 1).astype(np.uint8)


def feasible_mask(model: QuboModel, tol: float = 1e-9) -> np.ndarray:
    """(2^n_qubits,) True where the penalty is zero."""
    return np.abs(model.penalty(all_bitstrings(model.n_qubits))) < tol


def check_encoding(model: QuboModel, feasible_units: np.ndarray, tol: float = 1e-9) -> dict:
    """Brute-force proof for a small instance. ``feasible_units``: (S, n) grid points that
    satisfy every ENCODED rule. Returns counts; the encoding is exact and a bijection iff
    n_zero_penalty == n_feasible == n_distinct_decoded, min_violation_penalty >= 1 - tol
    and encode_roundtrip is True."""
    Z = all_bitstrings(model.n_qubits)
    P = model.penalty(Z)
    zero = np.abs(P) < tol
    decoded = {tuple(r) for r in model.units(Z[zero]).tolist()}
    target = {tuple(r) for r in np.atleast_2d(feasible_units).astype(np.int64).tolist()}
    enc = model.encode(feasible_units)
    return dict(n_qubits=model.n_qubits, n_zero_penalty=int(zero.sum()), n_feasible=len(target),
                n_distinct_decoded=len(decoded), decoded_equals_feasible=decoded == target,
                min_penalty=float(P.min()),
                min_violation_penalty=float(P[~zero].min()) if (~zero).any() else float("inf"),
                encode_roundtrip=bool(np.all(np.abs(model.penalty(enc)) < tol)
                                      and np.array_equal(model.units(enc), np.atleast_2d(feasible_units))))


# ------------------------------------------------------------------ samplers
def simulated_annealing(model: QuboModel, n_reads: int, sweeps: int = 200, beta0: float = 0.1,
                        beta1: float = 10.0, seed: int | None = 0) -> np.ndarray:
    """Classical baseline on the same energy: single-bit-flip Metropolis with a geometric
    inverse-temperature schedule, ``n_reads`` independent runs. Returns (n_reads, n_qubits)
    final states. NOT a uniform sampler of the feasible set; its bias is what
    ``portfolio.exact.compare_to_exact`` measures."""
    rng = np.random.default_rng(seed)
    lin, quad, _ = model.matrices()
    sym = quad + quad.T
    n = model.n_qubits
    Z = rng.integers(0, 2, size=(n_reads, n)).astype(float)
    field_ = lin[None, :] + Z @ sym
    for beta in np.geomspace(beta0, beta1, sweeps):
        for i in rng.permutation(n):
            dE = (1.0 - 2.0 * Z[:, i]) * field_[:, i]
            flip = (dE <= 0) | (rng.random(n_reads) < np.exp(-beta * np.clip(dE, 0, None)))
            delta = np.where(flip, 1.0 - 2.0 * Z[:, i], 0.0)
            Z[:, i] += delta
            field_ += delta[:, None] * sym[i][None, :]
    return Z.astype(np.uint8)


def grover_iterations(fraction: float) -> int:
    """Iterations that maximise the success probability for a KNOWN feasible fraction."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return max(int(math.floor(math.pi / (4 * math.asin(math.sqrt(fraction))))), 0)


def grover_circuit(model: QuboModel, iterations: int):
    """Amplitude amplification of the zero-penalty bitstrings from the uniform
    superposition: H on every qubit, then ``iterations`` times [phase oracle, diffusion].
    The oracle is a diagonal +-1 gate built from the enumerated penalty, which is honest
    for a small-case distribution check and NOT a cost model: a real oracle would compute
    the penalty reversibly (compare ``ft.oracle``). Samples that pass are exactly uniform
    over the zero-penalty bitstrings."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DiagonalGate

    n = model.n_qubits
    mask = feasible_mask(model)
    oracle = DiagonalGate(np.where(mask, -1.0, 1.0).astype(complex).tolist())
    zero = np.ones(1 << n, dtype=complex)
    zero[0] = -1.0
    reflect = DiagonalGate(zero.tolist())
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.append(oracle, range(n))
        qc.h(range(n))
        qc.append(reflect, range(n))
        qc.h(range(n))
    return qc


def grover_sample(model: QuboModel, shots: int, iterations: int | None = None, seed: int | None = 0,
                  backend: str = "aer_statevector") -> tuple[np.ndarray, dict]:
    """Run ``grover_circuit`` and return ((shots, n_qubits) bitstrings, info). With
    ``iterations`` None the count that is optimal for the exact feasible fraction is used;
    that fraction is known here only because the instance is small enough to enumerate."""
    from qiskit import transpile

    from ..backends import sample

    mask = feasible_mask(model)
    frac = float(mask.mean())
    r = grover_iterations(frac) if iterations is None else int(iterations)
    qc = grover_circuit(model, r)
    Z = sample(qc, None, shots, backend=backend, seed=seed)
    basis = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)
    theta = math.asin(math.sqrt(frac))
    info = dict(n_qubits=model.n_qubits, iterations=r, feasible_fraction=frac,
                success_probability=math.sin((2 * r + 1) * theta) ** 2, depth=int(basis.depth()),
                two_qubit_gates=int(basis.count_ops().get("cx", 0)), shots=int(shots),
                oracle="diagonal gate from the enumerated penalty (not a reversible arithmetic circuit)")
    return Z, info
