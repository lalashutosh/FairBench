"""Logical-level fault-tolerant (FT) resource model for EXACT-UNIFORM sampling over the
feasible set F by amplitude amplification (AA) from the Dicke state |D^n_k>.

Algorithm
---------
Start state |D> = D|0^n>, D = Baertschi-Eidenbenz Dicke preparation (``quantum.dicke``).
Grover iterate  G = D R_0 D^dagger O_f  (= (2|D><D| - I) O_f up to a global phase), with
O_f = diag((-1)^f) the phase-mode feasibility oracle (``ft.oracle``) and
R_0 = reflection about |0^n> on the n data qubits. Oracle / reflection / Dicke ancillas
are clean (|0>) between components, so they are REUSED (peak qubits = n + max ancillas).

Uniformity (exact, any number of iterations r): |D> is uniform over the C = C(n,k)
weight-k strings and both reflections act on span{|F>, |F_perp>} (|F> = uniform over F,
|F_perp> = uniform over the infeasible weight-k strings), so after r iterates the state is
sin((2r+1)theta)|F> + cos((2r+1)theta)|F_perp>, sin^2 theta = P_F = |F|/C. Conditioned on a
classically verified feasible outcome, the sample is EXACTLY uniform over F (also for
BBHT / fixed-point AA, which only change the coefficients). "Feasible" means the
QUANTISED rule the oracle implements (integer weights, ``ft.oracle.quantise``); the
mismatch vs the float rule is ``OracleInfo.quantisation.joint_mismatch`` (0 on the
benchmark instances at DEFAULT_BITS; Monte Carlo estimate at large n).

Cost model (all defaults are parameters; see ``ResourceReport.assumptions``)
---------------------------------------------------------------------------
* Dicke prep: (n-1) CRY + [k(k-1)/2 + (n-k-1)(k-1)] CCRY + 2 CX per CRY/CCRY block + k X
  (0 < k < n; closed form == circuit, tested).
    CRY(t)  = RY(t/2) CX RY(-t/2) CX              -> 2 RY + 2 CX
    CCRY(t) = Toffoli(c1,c2 -> a) CRY_a(t) Toffoli -> 2 Toffoli + 2 RY + 2 CX, 1 clean ancilla
  Every RY is treated as an arbitrary angle (a few are Clifford; ignored = conservative).
* Rotation synthesis (Ross & Selinger, arXiv:1403.2975): T(RY) = rs_const*log2(1/eps_rot)
  + rs_offset, default 3.0 * log2(1/eps) + 0. Alternatives: repeat-until-success
  (Bocharov-Roetteler-Svore, arXiv:1404.5320) ~1.15 log2(1/eps)+9.2; mixed / catalysed
  synthesis (Hastings 1612.01011; Campbell 1612.02689) roughly halves log2(1/eps) -> lower
  rs_const. Error budget: eps_rot = eps_total / N_rot_run, N_rot_run = all arbitrary
  rotations in one coherent run (state prep -> measurement; longest likely run for BBHT),
  so the summed synthesis error per sample is <= eps_total (subadditivity of the
  operator-norm error).
* Reflection R_0 (n >= 4): X^n, CCX chain into n-3 clean ancillas ending in one CCZ,
  mirror, X^n: 2n-5 Toffolis (n=3: 1 CCZ; n<=2: Clifford). Same MCZ construction as the
  oracle's AND (equivalently MCX_m -> 2m-3 Toffolis with m-2 ancillas).
* Oracle: exact Toffoli / CNOT / qubit counts from the built circuit
  (``feasibility_oracle``) or from ``oracle_formula_counts`` (same construction, no gates;
  exact when given the same quantisation, ``approx=True`` needs no quantisation search).
* Toffoli -> T: ``t_per_toffoli`` (default 4: Jones arXiv:1212.5069 / Gidney logical-AND
  arXiv:1709.06648, 4 T per Toffoli; 7 = textbook Clifford+T with no ancilla). NOTE: with
  Gidney's measurement-based uncompute, every mirrored (uncompute) Toffoli costs 0 T, so
  ~half the oracle / reflection / CCRY Toffolis are T-free: default 4 on ALL Toffolis is
  conservative by ~2x on those parts.
* T-depth: ASAP layering; each Toffoli adds ``t_depth_per_toffoli`` (default 2, matching the
  4-T Gidney AND / Jones Toffoli; 1 = Selinger
  T-depth-1 Toffoli with 7 T and 4 ancillas) and each RY adds its full
  synthesised T-count (single-qubit sequences are serial). Dicke CCRY ancillas are drawn
  from the (idle) reflection-ancilla pool, so CCRYs on disjoint qubits run in parallel.
  Oracle: Toffoli-depth of the built circuit (adders share one temp register -> nearly
  serial); formula path uses Toffoli count (upper bound). Reflection: serial chain 2n-5.

Schedules (per sample; the measured string is checked classically, at no oracle cost)
------------------------------------------------------------------------------------
* "known": r = round(pi/(4 theta) - 1/2), success sin^2((2r+1)theta), repeat until
  feasible: E[attempts] = 1/p_succ, E[oracle calls] = r/p_succ, E[Dicke preps] = (1+2r)/p_succ.
* "bbht": Boyer-Brassard-Hoyer-Tapp (quant-ph/9605034) exponential search, lambda = 6/5,
  m <- min(lambda m, sqrt(C)), j ~ U{0..ceil(m)-1}; EXACT expected cost from the round
  distribution (geometric tail once m is capped). Paper bound: (9/4)/sqrt(P_F).
* "fixed_point": Yoder-Low-Chuang (arXiv:1409.3305) with L = 2l+1 >= ln(2/delta)/sqrt(w),
  w = lower bound on P_F (default P_F); success 1 - delta^2 T_L(T_{1/L}(1/delta) sqrt(1-P_F))^2
  >= 1 - delta^2. Uses generalised (arbitrary-phase) reflections: oracle phase-mode
  Toffolis + 2 (AND into an ancilla + phase rotation) and reflection 2(n-1) Toffolis, each
  + 1 arbitrary rotation per iterate.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ..constraints import ConstraintSet
from ..data import Universe
from .oracle import (DEFAULT_BITS, ENCODED_RULES, OracleInfo, Quantisation, _acc_bits, _mode_threshold, _plan,
                     _weighted_specs, feasibility_oracle, quantise)

SCHEDULES = ("known", "bbht", "fixed_point")


# =========================================================================== Dicke
def dicke_rotation_counts(n: int, k: int) -> dict:
    """Closed-form gate counts of ``quantum.dicke.dicke_state(n, k)`` (undecomposed)."""
    if k == 0 or k == n:
        return dict(cry=0, ccry=0, cx=0, x=k)
    cry = n - 1
    ccry = k * (k - 1) // 2 + (n - k - 1) * (k - 1)
    return dict(cry=cry, ccry=ccry, cx=2 * (cry + ccry), x=k)


def _dicke_blocks(n: int, k: int):
    """Abstract block list of dicke_state: ('cry', (c, t)) / ('ccry', (c1, c2, t)) (CX
    sandwiches touch the same qubits and carry no T, so they are folded into the block)."""
    if k == 0 or k == n:
        return
    for m in range(n, 1, -1):
        lmax = min(k, m - 1)
        yield "cry", (m - 1, m - 2)               # paper (m-1, m) -> qiskit (m-2, m-1)
        for l in range(2, lmax + 1):
            yield "ccry", (m - 1, m - l, m - l - 1)


@dataclass
class DickeCost:
    n: int
    k: int
    n_cry: int
    n_ccry: int
    rotations: int        # arbitrary RY after decomposition
    toffoli: int
    cnot: int
    ancilla: int          # clean ancillas assumed available (borrowed from reflection pool)
    rotation_depth: int   # RY layers on the critical path
    toffoli_depth: int    # Toffoli layers on the critical path (with the rotation layers)

    def t_count(self, t_ry: float, t_per_toffoli: float) -> float:
        return self.rotations * t_ry + self.toffoli * t_per_toffoli

    def t_depth(self, t_ry: float, t_depth_per_toffoli: float) -> float:
        return self.rotation_depth * t_ry + self.toffoli_depth * t_depth_per_toffoli


def dicke_cost(n: int, k: int) -> DickeCost:
    """Decomposed Dicke cost (CRY -> 2 RY + 2 CX; CCRY -> 2 Toffoli + 2 RY + 2 CX + ancilla)
    and critical-path (rotation layers, Toffoli layers) via ASAP scheduling. The critical
    path is taken w.r.t. rotation layers first (dominant), ties broken by Toffoli layers."""
    c = dicke_rotation_counts(n, k)
    rot = 2 * (c["cry"] + c["ccry"])
    tof = 2 * c["ccry"]
    cx = c["cx"] + 2 * (c["cry"] + c["ccry"])
    t_rot = np.zeros(n, np.int64)  # (rotation layers, toffoli layers) finish time per qubit
    t_tof = np.zeros(n, np.int64)
    W = 10 ** 6                    # lexicographic weight: rotation layers dominate
    for kind, qs in _dicke_blocks(n, k):
        qs = list(qs)
        start = max(int(t_rot[q]) * W + int(t_tof[q]) for q in qs)
        dr, dt = (2, 0) if kind == "cry" else (2, 2)
        end = start + dr * W + dt
        for q in qs:
            t_rot[q], t_tof[q] = divmod(end, W)
    fin = max((int(a) * W + int(b) for a, b in zip(t_rot, t_tof)), default=0)
    rd, td = divmod(fin, W)
    anc = 0 if c["ccry"] == 0 else max(1, n // 3)
    return DickeCost(n, k, c["cry"], c["ccry"], rot, tof, cx, anc, int(rd), int(td))


# =========================================================================== reflection
@dataclass
class ReflectionCost:
    n: int
    toffoli: int
    ancilla: int
    cnot: int
    toffoli_depth: int


def reflection_cost(n: int, generalized: bool = False) -> ReflectionCost:
    """R_0 = I - 2|0^n><0^n| (X^n . MCZ . X^n). MCZ_n: n=1 Z, n=2 CZ, n=3 CCZ (1 Toffoli),
    n>=4 CCX chain into n-3 ancillas + CCZ + mirror = 2n-5 Toffolis.
    ``generalized``: phase e^{i a} on |0^n> (fixed-point AA): chain of n-1 ancillas
    (AND into last ancilla), one phase rotation, mirror -> 2(n-1) Toffolis."""
    if generalized:
        return ReflectionCost(n, 2 * max(n - 1, 0), max(n - 1, 0), 0, 2 * max(n - 1, 0))
    if n <= 2:
        return ReflectionCost(n, 0, 0, 1 if n == 2 else 0, 0)
    if n == 3:
        return ReflectionCost(n, 1, 0, 0, 1)
    return ReflectionCost(n, 2 * n - 5, n - 3, 0, 2 * n - 5)


def reflection_circuit(n: int, n_qubits: int | None = None, ancillas=None):
    """Explicit R_0 circuit (I - 2|0^n><0^n| on data qubits 0..n-1, using clean ancillas
    ``ancillas`` (default n..2n-4)). Gates: X, CZ, CCX, CCZ."""
    from qiskit import QuantumCircuit
    rc = reflection_cost(n)
    anc = list(ancillas) if ancillas is not None else list(range(n, n + rc.ancilla))
    nq = n_qubits if n_qubits is not None else n + rc.ancilla
    if len(anc) < rc.ancilla:
        raise ValueError("not enough ancillas")
    qc = QuantumCircuit(nq, name="R0")
    d = list(range(n))
    qc.x(d)
    if n == 1:
        qc.z(0)
    elif n == 2:
        qc.cz(0, 1)
    elif n == 3:
        qc.ccz(0, 1, 2)
    else:
        a = anc[:n - 3]
        chain = [(d[0], d[1], a[0])] + [(a[i - 1], d[i + 1], a[i]) for i in range(1, n - 3)]
        for g in chain:
            qc.ccx(*g)
        qc.ccz(a[-1], d[n - 2], d[n - 1])
        for g in reversed(chain):
            qc.ccx(*g)
    qc.x(d)
    return qc


# =========================================================================== oracle formula
def _plans(u: Universe, cs: ConstraintSet, rules):
    """Mirror of feasibility_oracle's planning (the same ``oracle._plan``):
    (excluded, [(comp, members, weights, T, smax, rr)])."""
    excluded, _, plans = _plan(u, cs, rules, cs.cardinality)
    return excluded, [(c, m, w, T, s, _acc_bits(T, s)) for c, _nm, m, w, T, s, _p in plans]


def _approx_rules(u: Universe, cs: ConstraintSet, bits: int):
    """Quantised rules without the scale / threshold search: s = s_max, T = rounding of the
    ideal threshold (ceil; strict rules floor + 1)."""
    k = cs.cardinality
    out = []
    for sp in _weighted_specs(u, cs, k):
        lo, hi = float(sp.v.min()), float(sp.v.max())
        s = (2 ** bits - 1) / (hi - lo) if hi > lo else 1.0
        out.append((sp.kind, np.rint(s * (sp.v - lo)).astype(np.int64),
                    _mode_threshold("minmis", sp.t0(s, lo, k), k, sp.strict)))
    return out


def oracle_formula_counts(u: Universe, cs: ConstraintSet, bits: int | None = None, *,
                          quant: Quantisation | None = None, approx: bool = False,
                          mode: str = "phase", check_cardinality: bool = False) -> dict:
    """Toffoli / CNOT / qubit counts of ``feasibility_oracle`` WITHOUT emitting gates.
    With ``quant`` (or approx=False: runs ``quantise``) the counts are EXACTLY those of the
    built circuit; ``approx=True`` skips the quantisation search (s = s_max, T = ceil of the
    ideal threshold), typically within ~1% of the built circuit."""
    bits = DEFAULT_BITS if bits is None else int(bits)
    n, k = u.n, cs.cardinality
    unknown = sorted({type(c).__name__ for c in cs.constraints if not isinstance(c, ENCODED_RULES)})
    if unknown:
        raise TypeError(f"no oracle encoding for {unknown}")
    if approx and quant is None:
        rules = _approx_rules(u, cs, bits)
    else:
        qz = quant if quant is not None else quantise(u, cs, bits)
        rules = [(r.kind, r.weights, r.threshold) for r in qz.rules]
    excluded, plans = _plans(u, cs, rules)
    comps: dict[str, dict] = {}
    tof = cx = 0
    Rs = []
    for comp, mem, w, T, smax, rr in plans:
        R = rr + 1
        Rs.append(R)
        t_c = len(mem) * (2 * (R - 1) if R > 1 else 0)
        x_c = sum(2 * bin(wi & ((1 << R) - 1)).count("1") + (4 * (R - 1) + 2 if R > 1 else 1)
                  for wi in w)
        d = comps.setdefault(comp, {"toffoli": 0, "cx": 0, "qubits": 0})
        d["toffoli"] += t_c
        d["cx"] += x_c
        d["qubits"] += R
        tof += t_c
        cx += x_c
    n_lit = len(excluded) + len(plans)
    if check_cardinality:
        Rc = max(1, math.ceil(math.log2(n + 1)))
        Rs.append(Rc)
        t_c = n * (2 * (Rc - 1) if Rc > 1 else 0)
        x_c = n * (2 + (4 * (Rc - 1) + 2 if Rc > 1 else 1))
        comps["cardinality"] = {"toffoli": t_c, "cx": x_c, "qubits": Rc}
        tof += t_c
        cx += x_c
        n_lit += Rc
    Rmax = max(Rs + [0])
    shared = (Rmax + 1) if Rmax > 1 else 0
    m = n_lit
    if mode == "bit":
        n_chain = max(m - 2, 0)
        t_and = 0 if m <= 1 else (1 if m == 2 else 2 * m - 3)
        cx_and = 1 if m == 1 else 0
    else:
        n_chain = max(m - 3, 0)
        t_and = 0 if m <= 2 else (1 if m == 3 else 2 * m - 5)
        cx_and = 1 if m == 2 else 0
    n_anc = sum(Rs) + shared + n_chain + (1 if mode == "bit" else 0)
    comps["and_flags"] = {"toffoli": t_and, "cx": cx_and, "qubits": n_chain}
    comps["uncompute"] = {"toffoli": tof, "cx": cx, "qubits": 0}
    comps["shared_adder"] = {"qubits": shared}
    return dict(mode=mode, n_data=n, n_ancilla=n_anc, n_qubits=n + n_anc,
                toffoli=2 * tof + t_and, cnot=2 * cx + cx_and, n_literals=m,
                toffoli_depth=2 * tof + t_and, components=comps, approx=bool(approx and quant is None))


def oracle_toffoli_depth(qc) -> int:
    """Toffoli-depth (CCX / CCZ layers) of a built oracle circuit."""
    return int(qc.depth(filter_function=lambda inst: inst.operation.name in ("ccx", "ccz")))


# =========================================================================== schedules
def _theta(p: float) -> float:
    if not 0 < p <= 1:
        raise ValueError(f"P_F must be in (0, 1], got {p}")
    return math.asin(math.sqrt(p))


def grover_success(p: float, r: int) -> float:
    return math.sin((2 * r + 1) * _theta(p)) ** 2


def known_schedule(p: float) -> dict:
    th = _theta(p)
    r = max(0, int(round(math.pi / (4 * th) - 0.5)))
    ps = math.sin((2 * r + 1) * th) ** 2
    return dict(schedule="known", iterations=r, success_prob=ps, expected_attempts=1 / ps,
                expected_oracle_calls=r / ps, expected_dicke_preps=(1 + 2 * r) / ps,
                max_iterations_per_run=r)


def bbht_schedule(p: float, N: float, lam: float = 6 / 5, tail_tol: float = 1e-3,
                  max_rounds: int = 100_000) -> dict:
    """Exact expected cost of BBHT exponential search (unknown P_F) from the round
    distribution: round with parameter m draws j ~ U{0..ceil(m)-1}, success
    E_j sin^2((2j+1) theta); m <- min(lam m, sqrt(N)). Once m is capped all later rounds are
    i.i.d. (geometric tail, summed in closed form). ``max_iterations_per_run``: largest j
    reachable before the survival probability drops below ``tail_tol`` (for the
    synthesis-error budget)."""
    th = _theta(p)
    cap = math.sqrt(N)
    m, S = 1.0, 1.0
    E_calls = E_preps = E_att = 0.0
    j_max = 0
    for _ in range(max_rounds):
        M = max(1, math.ceil(m - 1e-12))
        # closed form of mean_{j<M} sin^2((2j+1)th) (BBHT Lemma 2); O(1) memory for huge M
        pr = 0.5 - math.sin(4 * M * th) / (4 * M * math.sin(2 * th))
        cj = (M - 1) / 2
        if S >= tail_tol:
            j_max = M - 1
        capped = m >= cap - 1e-12
        if capped:
            E_calls += S * cj / pr
            E_preps += S * (1 + 2 * cj) / pr
            E_att += S / pr
            break
        E_calls += S * cj
        E_preps += S * (1 + 2 * cj)
        E_att += S
        S *= 1 - pr
        if S < 1e-18:
            break
        m = min(lam * m, cap)
    return dict(schedule="bbht", iterations=float(E_calls / E_att), success_prob=1 / E_att,
                expected_attempts=E_att, expected_oracle_calls=E_calls,
                expected_dicke_preps=E_preps, max_iterations_per_run=int(j_max),
                bbht_bound=(9 / 4) / math.sqrt(p), lam=lam)


def _cheb(L: int, x: float) -> float:
    """Chebyshev T_L(x) for x >= 0."""
    return math.cos(L * math.acos(x)) if x <= 1 else math.cosh(L * math.acosh(x))


def fixed_point_success(p: float, l: int, delta: float) -> float:
    """YLC success 1 - delta^2 T_L(T_{1/L}(1/delta) sqrt(1-p))^2, L = 2l+1."""
    L = 2 * l + 1
    g_inv = math.cosh(math.acosh(1 / delta) / L)
    return 1 - delta ** 2 * _cheb(L, g_inv * math.sqrt(1 - p)) ** 2


def fixed_point_schedule(p: float, delta: float = 0.01, p_lower: float | None = None) -> dict:
    w = p if p_lower is None else p_lower
    l = max(0, math.ceil((math.log(2 / delta) / math.sqrt(w) - 1) / 2))
    ps = fixed_point_success(p, l, delta)
    return dict(schedule="fixed_point", iterations=l, success_prob=ps, expected_attempts=1 / ps,
                expected_oracle_calls=l / ps, expected_dicke_preps=(1 + 2 * l) / ps,
                max_iterations_per_run=l, L=2 * l + 1, delta=delta, p_lower=w)


def ylc_phases(l: int, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """YLC phases alpha_j = -beta_{l-j+1} = 2 arccot(tan(2 pi j / L) sqrt(1 - gamma^2))."""
    L = 2 * l + 1
    gamma = 1 / math.cosh(math.acosh(1 / delta) / L)
    j = np.arange(1, l + 1)
    alpha = 2 * np.arctan2(1.0, np.tan(2 * np.pi * j / L) * math.sqrt(1 - gamma ** 2))
    beta = -alpha[::-1]
    return alpha, beta


# =========================================================================== subspace sim
def grover_subspace_state(mask: np.ndarray, r: int) -> np.ndarray:
    """Exact state after r ideal iterates G = (2|D><D| - I) O_f, in the weight-k subspace
    (``mask``: feasibility of each basis state). Returns complex amplitudes (C,)."""
    mask = np.asarray(mask, bool)
    C = len(mask)
    psi = np.full(C, 1 / math.sqrt(C), complex)
    sgn = np.where(mask, -1.0, 1.0)
    for _ in range(r):
        psi = psi * sgn
        psi = 2 * psi.mean() - psi          # 2|D><D|psi> - psi, <D|psi> sqrt(C) = sum psi/sqrt(C)
    return psi


def ylc_subspace_state(mask: np.ndarray, l: int, delta: float) -> np.ndarray:
    """Exact YLC fixed-point sequence in the weight-k subspace:
    G(a, b) = -S_s(a) S_t(b),  S_s(a) = I - (1 - e^{-ia})|s><s|,  S_t(b) = I - (1 - e^{ib}) Pi_F."""
    mask = np.asarray(mask, bool)
    C = len(mask)
    s = np.full(C, 1 / math.sqrt(C), complex)
    psi = s.copy()
    alpha, beta = ylc_phases(l, delta)
    for a, b in zip(alpha, beta):
        psi = np.where(mask, psi * np.exp(1j * b), psi)
        psi = psi - (1 - np.exp(-1j * a)) * s * np.vdot(s, psi)
        psi = -psi
    return psi


# =========================================================================== full circuit
def grover_circuit(u: Universe, cs: ConstraintSet, r: int, bits: int | None = None, quant=None):
    """Full circuit: Dicke, then r x [O_f (phase) ; Dicke^dagger ; R_0 ; Dicke]. Oracle,
    reflection share one clean-ancilla register (reuse). Returns (qc, OracleInfo)."""
    from qiskit import QuantumCircuit
    from ..quantum.dicke import dicke_state
    n, k = u.n, cs.cardinality
    orc, info = feasibility_oracle(u, cs, bits, mode="phase", quant=quant)
    rc = reflection_cost(n)
    nq = n + max(info.n_ancilla, rc.ancilla)
    qc = QuantumCircuit(nq)
    D = dicke_state(n, k)
    Dd = D.inverse()
    R0 = reflection_circuit(n, nq, ancillas=range(n, n + rc.ancilla))
    data = list(range(n))
    qc.compose(D, data, inplace=True)
    for _ in range(r):
        qc.compose(orc, list(range(orc.num_qubits)), inplace=True)
        qc.compose(Dd, data, inplace=True)
        qc.compose(R0, list(range(nq)), inplace=True)
        qc.compose(D, data, inplace=True)
    return qc, info


# =========================================================================== report
@dataclass
class ResourceReport:
    schedule: str
    n: int
    k: int
    p_f: float
    logical_qubits: int               # peak, with ancilla reuse
    logical_qubits_no_reuse: int
    toffoli_per_iterate: float
    t_per_iterate: float              # incl. rotation synthesis
    t_depth_per_iterate: float
    n_rotations_per_iterate: int
    iterations: float                 # r (known), l (fixed point), E[j | round] (bbht)
    expected_attempts: float
    expected_oracle_calls_per_sample: float
    expected_dicke_preps_per_sample: float
    toffoli_per_sample: float
    t_count_per_sample: float
    t_depth_per_run: float            # longest likely run (prep + iterates)
    eps_rot: float
    t_per_rotation: float
    success_prob: float
    components: dict = field(default_factory=dict)
    assumptions: dict = field(default_factory=dict)
    schedule_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _exact_pf(u: Universe, cs: ConstraintSet, quant: Quantisation | None, limit: int = 3_000_000):
    from .oracle import _all_weight_k
    n, k = u.n, cs.cardinality
    if math.comb(n, k) > limit:
        return None
    X = _all_weight_k(n, k)
    ok = quant.check_batch(X, u, cs) if quant is not None else cs.check_batch(X, u)
    return float(ok.mean())


def resource_estimate(u: Universe, cs: ConstraintSet, bits: int | None = None,
                      schedule: str = "known", eps_total: float = 1e-3,
                      t_per_toffoli: float = 4, rs_const: float = 3.0, rs_offset: float = 0.0,
                      t_depth_per_toffoli: float = 2, p_f: float | None = None,
                      oracle_info: OracleInfo | dict | None = None,
                      quant: Quantisation | None = None, oracle_path: str = "build",
                      delta: float = 0.01, p_lower: float | None = None,
                      bbht_lambda: float = 6 / 5) -> ResourceReport:
    """Logical FT cost of one exact-uniform sample from F by AA over the Dicke state.

    oracle_path: "build" (emit the circuit; exact counts and Toffoli-depth), "formula"
    (``oracle_formula_counts`` with quantisation; exact counts, serial depth bound) or
    "approx" (formula without quantisation search). ``oracle_info`` (OracleInfo or the
    dict of ``oracle_formula_counts``) overrides. ``p_f``: feasible fraction among weight-k
    strings (default: exact enumeration of the quantised rule if C(n,k) <= 3e6)."""
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    bits = DEFAULT_BITS if bits is None else int(bits)
    n, k = u.n, cs.cardinality
    C = math.comb(n, k)
    # ---- oracle
    o_depth = None
    path_used = "given" if oracle_info is not None else oracle_path
    if oracle_info is None:
        if oracle_path == "build":
            qz = quant if quant is not None else quantise(u, cs, bits)
            qc, oracle_info = feasibility_oracle(u, cs, bits, mode="phase", quant=qz)
            o_depth = oracle_toffoli_depth(qc)
            quant = qz
        elif oracle_path in ("formula", "approx"):
            if oracle_path == "formula" and quant is None:
                quant = quantise(u, cs, bits)
            oracle_info = oracle_formula_counts(u, cs, bits, quant=quant,
                                                approx=oracle_path == "approx")
        else:
            raise ValueError("oracle_path must be build|formula|approx")
    if isinstance(oracle_info, OracleInfo):
        o_tof, o_cx, o_anc = oracle_info.toffoli_count, oracle_info.cnot_count, oracle_info.n_ancilla
        o_mismatch = oracle_info.quantisation.joint_mismatch
        o_neval = oracle_info.quantisation.n_eval
        if quant is None:
            quant = oracle_info.quantisation
    else:
        o_tof, o_cx, o_anc = oracle_info["toffoli"], oracle_info["cnot"], oracle_info["n_ancilla"]
        o_depth = oracle_info.get("toffoli_depth")
        o_mismatch = quant.joint_mismatch if quant is not None else None
        o_neval = quant.n_eval if quant is not None else None
    if o_depth is None:
        o_depth = o_tof
    if p_f is None:
        p_f = _exact_pf(u, cs, quant)
        if p_f is None:
            raise ValueError("p_f required: C(n,k) too large to enumerate")
    # ---- schedule
    if schedule == "known":
        sch = known_schedule(p_f)
    elif schedule == "bbht":
        sch = bbht_schedule(p_f, C, bbht_lambda)
    else:
        sch = fixed_point_schedule(p_f, delta, p_lower)
    fp = schedule == "fixed_point"
    # ---- components
    dk = dicke_cost(n, k)
    rf = reflection_cost(n, generalized=fp)
    o_tof_it = o_tof + (2 if fp else 0)       # generalised oracle: AND into ancilla + phase
    extra_rot = 2 if fp else 0                # oracle phase + reflection phase rotations
    rot_iter = 2 * dk.rotations + extra_rot
    r_run = sch["max_iterations_per_run"]
    n_rot_run = dk.rotations + r_run * rot_iter
    eps_rot = eps_total / max(n_rot_run, 1)
    t_ry = rs_const * math.log2(1 / eps_rot) + rs_offset
    tof_iter = o_tof_it + 2 * dk.toffoli + rf.toffoli
    t_iter = tof_iter * t_per_toffoli + rot_iter * t_ry
    t_prep = dk.t_count(t_ry, t_per_toffoli)
    td_dicke = dk.t_depth(t_ry, t_depth_per_toffoli)
    td_iter = ((o_depth + (2 if fp else 0)) * t_depth_per_toffoli + 2 * td_dicke
               + rf.toffoli_depth * t_depth_per_toffoli + (2 * t_ry if fp else 0))
    A, Oc, Dp = sch["expected_attempts"], sch["expected_oracle_calls"], sch["expected_dicke_preps"]
    # each oracle call comes with one Dicke^dagger . R_0 . Dicke; each attempt with one prep
    tof_sample = A * dk.toffoli + Oc * tof_iter
    t_sample = A * t_prep + Oc * t_iter
    anc_o = o_anc + (1 if fp else 0)
    peak = n + max(anc_o, rf.ancilla, dk.ancilla)
    comps = {
        "oracle": dict(toffoli=o_tof_it, cnot=o_cx, ancilla=anc_o, toffoli_depth=o_depth,
                       t=o_tof_it * t_per_toffoli, rotations=1 if fp else 0,
                       quant_joint_mismatch=o_mismatch, quant_n_eval=o_neval),
        "dicke_x2": dict(toffoli=2 * dk.toffoli, rotations=2 * dk.rotations, cnot=2 * dk.cnot,
                         t=2 * t_prep, n_cry=dk.n_cry, n_ccry=dk.n_ccry,
                         rotation_depth=dk.rotation_depth, ancilla=dk.ancilla),
        "reflection": dict(toffoli=rf.toffoli, ancilla=rf.ancilla, rotations=1 if fp else 0,
                           t=rf.toffoli * t_per_toffoli + (t_ry if fp else 0)),
        "initial_prep": dict(toffoli=dk.toffoli, rotations=dk.rotations, t=t_prep),
    }
    tot = sum(comps[c]["t"] for c in ("oracle", "dicke_x2", "reflection"))
    for c in ("oracle", "dicke_x2", "reflection"):
        comps[c]["t_frac_of_iterate"] = comps[c]["t"] / tot if tot else 0.0
    comps["dominant"] = max(("oracle", "dicke_x2", "reflection"), key=lambda c: comps[c]["t"])
    assumptions = dict(
        bits=bits, eps_total=eps_total, rotation_error_budget="eps_rot = eps_total / N_rot_run",
        n_rot_run=n_rot_run, t_per_toffoli=t_per_toffoli,
        t_per_toffoli_note="4 = Jones/Gidney AND; 0 T if uncompute measurement-based (not used)",
        rotation_synthesis=f"T(RY) = {rs_const} log2(1/eps_rot) + {rs_offset} (Ross-Selinger)",
        t_depth_per_toffoli=t_depth_per_toffoli,
        cry="2 RY + 2 CX", ccry="2 Toffoli (AND to ancilla) + CRY",
        reflection="MCZ via CCX chain: 2n-5 Toffolis, n-3 ancillas" if not fp
        else "generalised: 2(n-1) Toffolis, n-1 ancillas, 1 phase rotation",
        oracle_path=path_used, classical_check="free (measured string verified classically)",
        ancilla_reuse=True, delta=delta if fp else None, bbht_lambda=bbht_lambda,
        feasibility="quantised rule (ft.oracle.quantise)")
    return ResourceReport(
        schedule=schedule, n=n, k=k, p_f=p_f, logical_qubits=peak,
        logical_qubits_no_reuse=n + anc_o + rf.ancilla + dk.ancilla,
        toffoli_per_iterate=tof_iter, t_per_iterate=t_iter, t_depth_per_iterate=td_iter,
        n_rotations_per_iterate=rot_iter, iterations=sch["iterations"], expected_attempts=A,
        expected_oracle_calls_per_sample=Oc, expected_dicke_preps_per_sample=Dp,
        toffoli_per_sample=tof_sample, t_count_per_sample=t_sample,
        t_depth_per_run=td_dicke + r_run * td_iter, eps_rot=eps_rot, t_per_rotation=t_ry,
        success_prob=sch["success_prob"], components=comps, assumptions=assumptions,
        schedule_info=sch)
