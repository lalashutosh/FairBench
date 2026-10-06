"""Reversible feasibility oracle for amplitude amplification over the Dicke state.

Scope / semantics
-----------------
* Inputs are weight-k basis states |x> (k = Cardinality, guaranteed by Dicke-state
  preparation), so Cardinality is NOT checked by default (``check_cardinality=True``
  adds an exact popcount == k test). Off the weight-k subspace the oracle is still
  a clean diagonal +-1 (compute / uncompute), but its flags are only meaningful on
  weight-k inputs unless ``check_cardinality=True``.
* On weight-k inputs the average rules are linear:
      MinESG(m)     <=>  sum_i esg_i x_i      >=  k m
      CarbonCap(c)  <=>  sum_i carbon_i x_i   <=  k c   <=>  sum_i (-carbon_i) x_i >= -k c
  Each is written as  sum_i v_i x_i >= k*theta  and QUANTISED (``quantise``):
      w_i = round(s (v_i - min v)),   s in [s_max/2, s_max],  s_max = (2^b - 1) / (max v - min v)
  i.e. ``b`` bits per weight (w_i in [0, 2^b - 1]); s is picked from ``n_scales`` grid
  values to minimise mismatch (largest s on ties). Shifting by min v is free on weight-k inputs (it moves the
  threshold by k*min v) and keeps all weights non-negative. The integer threshold T
  (test sum w_i x_i >= T) is chosen to minimise disagreement with the float rule on the
  weight-k states that pass the exact (sector / exclusion) rules - exactly enumerated
  for n <= 16, Monte Carlo (fit / held-out halves) above; ties -> nearest to s*k*(theta - min v).
  ``Quantisation.joint_mismatch`` counts weight-k states whose full quantised feasibility
  differs from ``ConstraintSet.check_batch``.
  AT SCALE QUANTISATION IS PART OF THE PROBLEM DEFINITION: the oracle decides feasibility
  exactly for the quantised rule (integer weights + integer threshold); with float scores
  there is no finite-precision circuit that is exact in general, and ESG/carbon data are
  themselves only meaningful to a few significant digits.
* Excluded assets are dropped from every sum / count: if one is held the final AND fails
  anyway, so their contribution never matters.

Circuit (gates X, CX, CCX; phase mode additionally one Z / CZ / CCZ; no MCX emitted)
-----------------------------------------------------------------------------------
* "Compare by overflow": a rule  S >= T  with  0 <= S <= Smax  uses an (r+1)-bit
  accumulator initialised (X gates) to 2^r - T with 2^r >= T and 2^r > Smax - T; after
  adding S, bit r == [S >= T]. The comparator is free (no extra gates).
* SectorCap(c) over sector members S' (non-excluded): popcount into such an accumulator
  with T = c + 1 (bit r = violation). Vacuous caps (min(|S'|, k) <= c) are skipped.
* Weighted sums: for each asset with w_i != 0, controlled-constant add: load w_i into a
  shared zero temp register with CX(x_i -> temp_j) for set bits j, Cuccaro ripple-carry
  add temp into the accumulator (R-bit modular, 2(R-1) Toffolis, one shared carry
  ancilla), unload with the same CXs. A popcount increment is the w = 1 case.
* Exclusion: no gates; each excluded data bit enters the final AND as a negative literal.
* AND of all m literals (negative literals flipped with X before/after): explicit CCX
  chain into m-2 clean ancillas, 2m-3 Toffolis (bit mode, target qubit) - i.e. an MCX_m
  decomposed with m-2 clean ancillas. Phase mode (no target): chain of m-3 ancillas
  ending in one CCZ (counted as a Toffoli): 2m-5 Toffolis (m=1,2,3: Z, CZ, CCZ).
* Then the full mirror (uncompute) of every accumulator: all ancillas return to |0>.
Toffoli count = #CCX + #CCZ + sum over MCX_m of (2m - 3) (none emitted here).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit

from ..constraints import CarbonCap, Cardinality, ConstraintSet, Exclusion, MinESG, SectorCap
from ..data import Universe

# Smallest b with zero joint quantisation mismatch on ALL of p0_instance, island_instance
# and island_family(n in {12, 14, 16}, seeds 0..4) (exact enumeration; tests/test_oracle.py).
# Per instance 5..8 bits usually suffice (``bits="auto"``); mismatch is not monotone in b.
# Residual counts at b = 8 / 9 / 10: 10 / 1 / 1 states over all 17 instances (thresholds sit
# at quantiles of attained averages, so near-ties are by construction).
DEFAULT_BITS = 11

MCX_TOFFOLI_RULE = "MCX with m controls -> 2m-3 Toffolis using m-2 clean ancillas (none emitted)"


# --------------------------------------------------------------------------- quantisation
@dataclass
class QuantRule:
    kind: str                 # "esg" | "carbon"
    float_threshold: float    # rule's own threshold (min_avg_score / max_avg)
    weights: np.ndarray       # (n,) int64 in [0, 2^bits - 1]
    threshold: int            # quantised test: sum w_i x_i >= threshold
    scale: float
    shift: float              # min of v (v = esg or -carbon)
    ideal_threshold: float    # s * k * (theta - shift)
    mismatch: int             # disagreements with the float rule on the fit/eval domain
    n_domain: int


@dataclass
class Quantisation:
    bits: int
    k: int
    rules: list[QuantRule]
    joint_mismatch: int       # weight-k states where quantised feasibility != float feasibility
    n_eval: int
    exact: bool               # True: all C(n,k) states enumerated; False: Monte Carlo
    false_pos: int = 0        # quantised-feasible but float-infeasible (removed by float post-check)
    false_neg: int = 0        # float-feasible but rejected by the oracle (biases the sample)
    n_feasible_eval: int = 0  # |F| among the evaluated states
    superset: bool = False    # thresholds chosen so F subset of F_q by construction
    mismatch_rate: float = field(init=False)

    def __post_init__(self):
        self.mismatch_rate = self.joint_mismatch / max(self.n_eval, 1)

    def check_batch(self, X: np.ndarray, u: Universe, cs: ConstraintSet) -> np.ndarray:
        """Quantised feasibility (exact rules as-is, weighted rules quantised)."""
        X = np.atleast_2d(X).astype(np.int64)
        ok = np.ones(len(X), dtype=bool)
        for c in cs.constraints:
            if not isinstance(c, (MinESG, CarbonCap)):
                ok &= c.check_batch(X, u)
        for r in self.rules:
            ok &= X @ r.weights >= r.threshold
        return ok


def _k_of(cs: ConstraintSet) -> int:
    k = cs.cardinality
    if k is None:
        raise ValueError("ConstraintSet needs a Cardinality constraint (Dicke weight k)")
    return k


def _all_weight_k(n: int, k: int) -> np.ndarray:
    comb = np.array(list(itertools.combinations(range(n), k)), dtype=np.int64).reshape(-1, k)
    X = np.zeros((len(comb), n), np.uint8)
    np.put_along_axis(X, comb, 1, axis=1)
    return X


def _random_weight_k(n: int, k: int, m: int, rng) -> np.ndarray:
    ranks = rng.random((m, n)).argsort(axis=1).argsort(axis=1)
    return (ranks < k).astype(np.uint8)


def _best_threshold(q: np.ndarray, y: np.ndarray, t0: float) -> tuple[int, int]:
    """Integer T minimising #(y & q<T) + #(~y & q>=T); ties -> nearest to t0."""
    qp, qn = np.sort(q[y]), np.sort(q[~y])
    cand = np.unique(np.concatenate([q, q + 1, [math.ceil(t0), math.floor(t0)]]).astype(np.int64))
    mis = np.searchsorted(qp, cand, side="left") + (len(qn) - np.searchsorted(qn, cand, side="left"))
    best = mis.min()
    c = cand[mis == best]
    T = int(c[np.argmin(np.abs(c - t0))])
    return T, int(best)


def quantise(u: Universe, cs: ConstraintSet, bits: int | str | None = None, *,
             exact_max_n: int = 16, mc_samples: int = 200_000, seed: int = 0,
             max_bits: int = 20, n_scales: int = 64,
             superset: bool | None = None) -> Quantisation:
    """Quantise MinESG / CarbonCap rules to integer weights (``bits`` per weight) and
    integer thresholds; report mismatch vs the float rules on weight-k states (exact for
    n <= ``exact_max_n``, else Monte Carlo with ``mc_samples`` fit + ``mc_samples`` held-out).
    ``bits=None`` -> DEFAULT_BITS; ``bits="auto"`` -> smallest b <= max_bits with zero
    joint mismatch (on the evaluated states).
    ``superset`` (default: True when not ``exact``): threshold ``T = ceil(t0 - k/2)``. Each
    weight is rounded by at most 1/2, so ``sum w_i x_i >= t0 - k/2`` for every float-feasible
    x, i.e. F is a subset of F_q with zero false negatives *provably* (not just on the sample).
    A classical float post-check of the measured sample then gives exactly uniform samples
    over F; the cost is a success factor |F|/|F_q| (``1 - false_pos/|F_q|``)."""
    if bits == "auto":
        for b in range(1, max_bits + 1):
            q = quantise(u, cs, b, exact_max_n=exact_max_n, mc_samples=mc_samples, seed=seed,
                         n_scales=n_scales, superset=superset)
            if q.joint_mismatch == 0:
                return q
        return q
    bits = DEFAULT_BITS if bits is None else int(bits)
    if bits < 1:
        raise ValueError("bits must be >= 1")
    n, k = u.n, _k_of(cs)
    exact = n <= exact_max_n
    if superset is None:
        superset = not exact
    if exact:
        X_fit = X_eval = _all_weight_k(n, k)
    else:
        rng = np.random.default_rng(seed)
        X_fit = _random_weight_k(n, k, mc_samples, rng)
        X_eval = _random_weight_k(n, k, mc_samples, rng)
    exact_rules = [c for c in cs.constraints if not isinstance(c, (MinESG, CarbonCap))]
    dom = ConstraintSet(exact_rules).check_batch(X_fit, u)
    Xd = X_fit[dom].astype(np.int64)
    rules = []
    for c in cs.constraints:
        if isinstance(c, MinESG):
            kind, v, theta = "esg", np.asarray(u.esg_score, float), c.min_avg_score
        elif isinstance(c, CarbonCap):
            kind, v, theta = "carbon", -np.asarray(u.carbon, float), -c.max_avg
        else:
            continue
        lo, hi = float(v.min()), float(v.max())
        s_max = (2 ** bits - 1) / (hi - lo) if hi > lo else 1.0
        y = c.check_batch(Xd, u) if len(Xd) else None
        best = None
        # scale search: s in [s_max/2, s_max] (weights stay < 2^bits); fewest mismatches,
        # ties -> largest scale (finest resolution)
        for s in s_max * np.linspace(1.0, 0.5, max(int(n_scales), 1)):
            w = np.rint(s * (v - lo)).astype(np.int64)
            t0 = s * k * (theta - lo)
            if superset:
                T = math.ceil(t0 - k / 2 - 1e-9)
                mis = int(((Xd @ w >= T) != y).sum()) if y is not None else 0
            elif y is not None:
                T, mis = _best_threshold(Xd @ w, y, t0)
            else:
                T, mis = math.ceil(t0), 0
            if best is None or mis < best[3]:
                best = (s, w, T, mis, t0)
            if mis == 0:
                break
        s, w, T, mis, t0 = best
        float_thr = c.min_avg_score if kind == "esg" else c.max_avg
        rules.append(QuantRule(kind, float_thr, w, T, s, lo, t0, mis, int(len(Xd))))
    qz = Quantisation(bits, k, rules, 0, len(X_eval), exact)
    fq, ff = qz.check_batch(X_eval, u, cs), cs.check_batch(X_eval, u)
    fp, fn = int((fq & ~ff).sum()), int((~fq & ff).sum())
    return Quantisation(bits, k, rules, fp + fn, len(X_eval), exact, fp, fn, int(ff.sum()),
                        bool(superset))


# --------------------------------------------------------------------------- circuit build
@dataclass
class OracleInfo:
    mode: str
    n_data: int
    n_ancilla: int            # every non-data qubit (incl. bit-mode target)
    n_qubits: int
    gate_counts: dict         # gate name -> count (whole circuit)
    toffoli_count: int        # CCX + CCZ + decomposed MCX (2m-3 each)
    cnot_count: int
    components: dict          # component -> {"toffoli", "cx", "x", "qubits"}
    layout: dict              # register name -> list of qubit indices
    quantisation: Quantisation
    n_literals: int           # inputs to the final AND
    skipped: list             # vacuous rules skipped (with reason)
    mcx_rule: str = MCX_TOFFOLI_RULE

    def summary(self) -> dict:
        return dict(mode=self.mode, n_data=self.n_data, n_ancilla=self.n_ancilla,
                    n_qubits=self.n_qubits, toffoli=self.toffoli_count, cnot=self.cnot_count,
                    bits=self.quantisation.bits, joint_mismatch=self.quantisation.joint_mismatch,
                    components={c: dict(v) for c, v in self.components.items()})


class _Builder:
    def __init__(self, n: int):
        self.nq = n
        self.gates: list[tuple[str, tuple, str]] = []
        self.layout: dict[str, list[int]] = {"data": list(range(n))}

    def alloc(self, m: int, name: str) -> list[int]:
        q = list(range(self.nq, self.nq + m))
        self.nq += m
        self.layout[name] = q
        return q

    def g(self, name, qs, comp):
        self.gates.append((name, tuple(qs), comp))

    # Cuccaro ripple-carry: b += a (mod 2^R); a restored, carry ancilla c restored.
    def add(self, a, b, c, comp):
        R = len(b)
        if R == 1:
            self.g("cx", (a[0], b[0]), comp)
            return
        prev = [c] + list(a[:R - 1])
        for j in range(R - 1):          # MAJ(prev, b_j, a_j)
            self.g("cx", (a[j], b[j]), comp)
            self.g("cx", (a[j], prev[j]), comp)
            self.g("ccx", (prev[j], b[j], a[j]), comp)
        self.g("cx", (a[R - 1], b[R - 1]), comp)
        self.g("cx", (a[R - 2], b[R - 1]), comp)
        for j in reversed(range(R - 1)):  # UMA(prev, b_j, a_j)
            self.g("ccx", (prev[j], b[j], a[j]), comp)
            self.g("cx", (a[j], prev[j]), comp)
            self.g("cx", (prev[j], b[j]), comp)

    def add_const_ctrl(self, ctrl, w, tmp, acc, c, comp):
        bits = [j for j in range(len(acc)) if (w >> j) & 1]
        for j in bits:
            self.g("cx", (ctrl, tmp[j]), comp)
        self.add(tmp[:len(acc)], acc, c, comp)
        for j in bits:
            self.g("cx", (ctrl, tmp[j]), comp)


def _acc_bits(T: int, smax: int) -> int:
    """r with 2^r >= T and 2^r > smax - T (accumulator has r+1 bits)."""
    r = 0
    while (1 << r) < T or (1 << r) <= smax - T:
        r += 1
    return r


def toffoli_count(circuit: QuantumCircuit) -> int:
    """CCX + CCZ + sum over MCX_m of (2m - 3) (see MCX_TOFFOLI_RULE)."""
    t = 0
    for inst in circuit.data:
        nm = inst.operation.name
        if nm in ("ccx", "ccz"):
            t += 1
        elif nm.startswith("mcx"):
            m = inst.operation.num_ctrl_qubits
            t += 1 if m == 2 else (0 if m < 2 else 2 * m - 3)
    return t


def feasibility_oracle(u: Universe, cs: ConstraintSet, bits: int | str | None = None,
                       mode: str = "phase", check_cardinality: bool = False,
                       quant: Quantisation | None = None, **quant_kw) -> tuple[QuantumCircuit, OracleInfo]:
    """Reversible feasibility oracle on n data qubits (qubit i = asset i) + ancillas.

    mode="bit":   |x>|0>_t|0>_anc -> |x>|f(x)>_t|0>_anc  (target = qubit n)
    mode="phase": |x>|0>_anc -> (-1)^{f(x)} |x>|0>_anc
    f = quantised feasibility (see module doc); Cardinality is assumed (Dicke) unless
    ``check_cardinality``. Returns (circuit, OracleInfo)."""
    if mode not in ("phase", "bit"):
        raise ValueError("mode must be 'phase' or 'bit'")
    n, k = u.n, _k_of(cs)
    qz = quant if quant is not None else quantise(u, cs, bits, **quant_kw)
    B = _Builder(n)
    target = B.alloc(1, "target")[0] if mode == "bit" else None

    excluded: set[int] = set()
    for c in cs.constraints:
        if isinstance(c, Exclusion):
            excluded |= {int(i) for i in c.indices}
    live = [i for i in range(n) if i not in excluded]
    literals: list[tuple[int, int, str]] = [(e, 0, "exclusion") for e in sorted(excluded)]
    skipped: list[str] = []

    # plan accumulators: (comp, name, members, weights, T, smax, flag_polarity)
    plans = []
    for c in cs.constraints:
        if isinstance(c, SectorCap):
            mem = [i for i in live if u.sector[i] == c.sector]
            if min(len(mem), k) <= c.max_count:
                skipped.append(f"SectorCap({c.sector},{c.max_count}): vacuous (|S'|={len(mem)}, k={k})")
                continue
            plans.append(("sector", f"sector_{c.sector}", mem, [1] * len(mem), c.max_count + 1,
                          min(len(mem), k), 0))
    for r in qz.rules:
        mem = [i for i in live if r.weights[i] != 0]
        smax = int(np.sort(r.weights[live])[::-1][:k].sum()) if live else 0
        if r.threshold <= 0:
            skipped.append(f"{r.kind}: quantised threshold {r.threshold} <= 0 (always satisfied)")
            continue
        plans.append((r.kind, r.kind, mem, [int(r.weights[i]) for i in mem], r.threshold, smax, 1))

    accs = []
    for comp, name, mem, w, T, smax, pol in plans:
        rr = _acc_bits(T, smax)
        accs.append((comp, name, mem, w, T, rr, pol, B.alloc(rr + 1, name)))
    card_acc = None
    if check_cardinality:
        Rc = max(1, math.ceil(math.log2(n + 1)))
        card_acc = B.alloc(Rc, "cardinality")
    Rmax = max([len(a[7]) for a in accs] + ([len(card_acc)] if card_acc else []) + [0])
    tmp = B.alloc(Rmax, "temp") if Rmax > 1 else []
    carry = B.alloc(1, "carry")[0] if Rmax > 1 else None

    # ---- compute
    for comp, name, mem, w, T, rr, pol, acc in accs:
        init = (1 << rr) - T
        for j in range(rr + 1):
            if (init >> j) & 1:
                B.g("x", (acc[j],), comp)
        for i, wi in zip(mem, w):
            B.add_const_ctrl(i, wi, tmp, acc, carry, comp)
        literals.append((acc[rr], pol, comp))
    if card_acc:
        for i in range(n):
            B.add_const_ctrl(i, 1, tmp, card_acc, carry, "cardinality")
        for j, q in enumerate(card_acc):
            literals.append((q, (k >> j) & 1, "cardinality"))
    n_compute = len(B.gates)

    # ---- AND of literals
    m = len(literals)
    lq = [q for q, _, _ in literals]
    # bit: CCX chain of m-2 ancillas, last CCX hits the target (2m-3 Toffolis);
    # phase: chain of m-3 ancillas, last step is a CCZ on 3 qubits (2m-5 Toffolis).
    n_chain = max(m - 2, 0) if mode == "bit" else max(m - 3, 0)
    chain = B.alloc(n_chain, "and_chain") if n_chain else []
    comp = "and_flags"
    for q, pol, _ in literals:
        if pol == 0:
            B.g("x", (q,), comp)
    global_phase = 0.0
    if m == 0:
        if mode == "bit":
            B.g("x", (target,), comp)
        else:
            global_phase = math.pi  # f == 1 everywhere
    elif mode == "bit" and m <= 2:
        B.g("cx", (lq[0], target), comp) if m == 1 else B.g("ccx", (lq[0], lq[1], target), comp)
    elif mode == "phase" and m <= 3:
        B.g({1: "z", 2: "cz", 3: "ccz"}[m], tuple(lq), comp)
    else:
        cg = [("ccx", (lq[0], lq[1], chain[0]))]
        for i in range(1, n_chain):
            cg.append(("ccx", (chain[i - 1], lq[i + 1], chain[i])))
        for nm, qs in cg:
            B.g(nm, qs, comp)
        if mode == "bit":
            B.g("ccx", (chain[-1], lq[m - 1], target), comp)
        else:
            B.g("ccz", (chain[-1], lq[m - 2], lq[m - 1]), comp)
        for nm, qs in reversed(cg):
            B.g(nm, qs, comp)
    for q, pol, _ in literals:
        if pol == 0:
            B.g("x", (q,), comp)

    # ---- uncompute (mirror; all gates self-inverse)
    for nm, qs, _ in reversed(B.gates[:n_compute]):
        B.g(nm, qs, "uncompute")

    qc = QuantumCircuit(B.nq, name=f"feas_oracle_{mode}")
    qc.global_phase = global_phase
    for nm, qs, _ in B.gates:
        getattr(qc, nm)(*qs)

    comps: dict[str, dict] = {c: {"toffoli": 0, "cx": 0, "x": 0, "qubits": 0}
                              for c in ("sector", "exclusion", "esg", "carbon", "cardinality",
                                        "and_flags", "uncompute")}
    for nm, qs, c in B.gates:
        d = comps[c]
        if nm in ("ccx", "ccz"):
            d["toffoli"] += 1
        elif nm in ("cx", "cz"):
            d["cx"] += 1
        elif nm in ("x", "z"):
            d["x"] += 1
    for comp_, name, *_r, acc in accs:
        comps[comp_]["qubits"] += len(acc)
    if card_acc:
        comps["cardinality"]["qubits"] += len(card_acc)
    comps["and_flags"]["qubits"] = len(chain) + (1 if mode == "bit" else 0)
    comps["exclusion"]["literals"] = len(excluded)
    comps["shared_adder"] = {"qubits": len(tmp) + (carry is not None)}

    counts = dict(qc.count_ops())
    info = OracleInfo(mode=mode, n_data=n, n_ancilla=B.nq - n, n_qubits=B.nq,
                      gate_counts={k_: int(v) for k_, v in counts.items()},
                      toffoli_count=toffoli_count(qc),
                      cnot_count=int(counts.get("cx", 0) + counts.get("cz", 0)),
                      components=comps, layout=B.layout, quantisation=qz, n_literals=m,
                      skipped=skipped)
    return qc, info
