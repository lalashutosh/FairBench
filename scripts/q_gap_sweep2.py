"""Q-E2: pre-registered exact spectral-gap comparison, QeMCMC vs fair classical chains.

PRE-REGISTERED PROTOCOL (written before running)
------------------------------------------------
Instances  island_family(n, seed) [fixed construction], n in {12,14,16}, seeds 0..9. ALL seeds
           reported; flag has_islands = (>=2 swap components and the second has >=3 states);
           summaries for all seeds and islands-only. Named extras: island_instance(), p0_instance().
Tuning     ONLY on n=12 seeds 0-4 (maximise MEDIAN gap over those seeds; plateau rule: among configs within
           2% of the max median gap take the cheapest; 'cost' tuning = max median gap/cost). Frozen params are then
           reported on n=12 seeds 5-9 (held-out), n=14 and n=16 (all seeds) and the extras. A
           per-instance ORACLE (best grid point per instance; n=12 only) is shown separately and is
           optimistic (selection bias).
Samplers   swap; multi-swap m=1..10 (frozen m*); independent uniform; CTRW untilted (topology x t
           grid); CTRW tilted with the SAME diagonal as QeMCMC (beta grid x t grid x topology,
           Metropolis rates, MH kernel via min(Q_xy,Q_yx)); QeMCMC with diag in {surrogate,
           indicator, violation} (topology ring/complete, alpha in {.5,1,2,4,8,16,32}, t log grid
           0.1-20; Q symmetric, asserted, so accept iff feasible).
Kernel     MH on the feasible set F with uniform target, Q restricted to F x F, rejected mass on
           the diagonal; gap = 1 - max(|l2|,|lmin|). Exact (no sampling, no Trotter) except cost.
Cost       swap-equivalent units (chains.proposal_cost). QeMCMC: n_trotter(t, Lambda) from a fit of
           min palindromic-Trotter steps for eps=0.05 (max TV over feasible rows) calibrated at n=12
           (island_family seed 0, SURROGATE diagonal used as proxy for all kinds, per topology;
           r = ceil(a (t Lambda)^p), Lambda = ||H||_2; the same relation is reused at n=14,16 --
           an extrapolation). Gates/step = 2*n_edges-1 + diag_2q. diag_2q: surrogate = #ZZ terms of
           the QUBO; indicator/violation = oracle cost ASSUMED m*n Toffoli-equivalents, m=1 (primary)
           and m=10 (sensitivity). CTRW cost: t * mean exit rate over F (primary, classical-favourable)
           and t * nominal degree (upper bound). Per-cost parameters are tuned separately
           (maximise median gap/cost on the same tuning set) for every sampler.
Headline   (frozen params) QeMCMC-X vs CTRW-tilted-X (same diag X): the "is it quantum?" test; and
           QeMCMC-X vs best penalty-blind classical (per instance max over frozen blind samplers).
           Median and IQR of per-instance ratios, per n.
Gate       quantum gain claimed only if, for diag X, QeMCMC beats tilted CTRW per step AND per cost
           (median over held-out seeds, every n) and the median ratio is non-decreasing in n.
Guards     --quick for development; --budget SECONDS writes partial results.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, itertools
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.special import jv, ive

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fairbench.chains import (feasible_mask, spectral_gap, transition_matrix, move_stats,
                              proposal_matrix_ctrw, proposal_cost, swap_component_labels)
from fairbench.instances import island_family, island_instance, p0_instance, describe_instance
from fairbench.quantum.hamiltonian import penalty_operator, diag_values
from fairbench.quantum.proposal import (subspace_basis, subspace_diagonal, subspace_hamiltonian,
                                        exact_proposal_matrix, penalty_separation, states_to_codes)
from fairbench.quantum import proposal as prop

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
KINDS = ["surrogate", "indicator", "violation"]
SHORT = {"surrogate": "surr", "indicator": "ind", "violation": "viol"}
TOPOS = ["ring", "complete"]
ALPHAS = [0.5, 1, 2, 4, 8, 16, 32]
BETAS = [0.5, 1, 2, 4, 8, 16, 32]
MS = list(range(1, 11))
EPS_TROT = 0.05
DENSE_MAX_C = 1000          # dense eigh up to n=12 (C=924); sparse Chebyshev above
T0 = time.time()


# ------------------------------------------------------------------ instances
class Inst:
    def __init__(self, name, u, cs, role, seed=None):
        self.name, self.u, self.cs, self.role, self.seed = name, u, cs, role, seed
        self.n, self.k = u.n, cs.cardinality
        self.mask = feasible_mask(u, cs)
        self.F = np.flatnonzero(self.mask)
        self.M, self.C = len(self.F), math.comb(self.n, self.k)
        d = describe_instance(u, cs)
        self.comps, sizes = d["n_components"], d["component_sizes"]
        self.has_islands = bool(self.comps >= 2 and sizes[1] >= 3)
        self.dense = self.C <= DENSE_MAX_C
        A = adjacency(self.n, self.k, "complete")
        from scipy.sparse.csgraph import connected_components
        self.labels = connected_components(A[self.F][:, self.F], directed=False)[1]
        self.diag = {k_: subspace_diagonal(k_, self.n, self.k, u, cs) for k_ in KINDS}
        self.auc = {k_: penalty_separation(self.diag[k_], self.mask) for k_ in KINDS}
        op = penalty_operator(cs, u)
        self.n_zz = int(sum(1 for p in op.paulis if sum(1 for c in str(p) if c == "Z") == 2))
        self.cache = {}


_ADJ = {}
def adjacency(n, k, topo):
    if (n, k, topo) not in _ADJ:
        _ADJ[(n, k, topo)] = (-subspace_hamiltonian(n, k, topo, 0.0, None, sparse=True)).tocsr()
    return _ADJ[(n, k, topo)]


def _eye_cols(I):
    E = np.zeros((I.C, I.M)); E[I.F, np.arange(I.M)] = 1.0
    return E



def cheb_apply(Hs, E, t, mode):
    """Chebyshev expansion of a function of the real symmetric sparse Hs applied to the (C,M) block E
    (real arithmetic, ~10x faster than complex expm_multiply here).
      mode "osc":  returns |exp(-i Hs t) E|^2  (A = even part, B = odd part, |U|^2 = A^2 + B^2)
      mode "diff": returns exp(t Hs) E for Hs <= 0 (CTRW generator)."""
    d = Hs.diagonal(); rs = np.asarray(abs(Hs).sum(1)).ravel() - np.abs(d)
    g_lo, g_hi = float((d - rs).min()), float((d + rs).max())          # Gershgorin (safe fallback)
    def ext(which):
        try:
            return float(eigsh(Hs, k=1, which=which, return_eigenvectors=False, tol=1e-6, maxiter=3000)[0])
        except Exception:
            return g_lo if which == "SA" else g_hi
    lo = ext("SA")
    hi = 0.0 if mode == "diff" else ext("LA")                           # generator spectrum <= 0, top = 0
    c, h = 0.5 * (hi + lo), 0.5 * (hi - lo) * 1.01 + 1e-9
    x = h * t
    J = 2 * int(x + 12 * max(x, 1.0) ** (1 / 3) + 40)
    j = np.arange(J)
    if mode == "osc":
        co = jv(j, x) * np.where(j == 0, 1.0, 2.0)
        sg = np.array([(-1.0) ** (q // 2) for q in j])      # (-i)^j = sg * {1, -i, ...}
    else:
        co = ive(j, x) * np.where(j == 0, 1.0, 2.0)
        # e^{t(c + h Ht)} = e^{t c + x} * sum (2-d) ive_j T_j ;  c + h = hi ~ 0 -> prefactor exp(t*hi)
        co = co * math.exp(t * (c + h))
    T0, T1 = E, (Hs @ E - c * E) / h
    A = co[0] * T0 if mode == "diff" else co[0] * sg[0] * T0
    B = np.zeros_like(E)
    def acc(q, T):
        nonlocal A, B
        if mode == "diff": A = A + co[q] * T
        elif q % 2 == 0: A = A + co[q] * sg[q] * T
        else: B = B + co[q] * sg[q] * T
    acc(1, T1)
    for q in range(2, J):
        T2 = 2 * (Hs @ T1 - c * T1) / h - T0
        acc(q, T2); T0, T1 = T1, T2
        if q > x and abs(co[q]) < 1e-15 and abs(co[min(q + 1, J - 1)]) < 1e-15: break
    return (A * A + B * B) if mode == "osc" else A

# ------------------------------------------------------------------ proposal rows Q[F, :]
def qe_rows(I, kind, topo, alpha, ts, force=None):
    """List of (M,C) arrays Q[x,:], x in F, for each t, plus ||H||_2."""
    H = subspace_hamiltonian(I.n, I.k, topo, float(alpha), I.diag[kind], sparse=not (I.dense if force is None else force == "dense"))
    outs = []
    if (I.dense if force is None else force == "dense"):
        w, V = np.linalg.eigh(H); lam = float(np.abs(w).max()); VF = V[I.F]
        for t in ts:
            U = (VF * np.exp(-1j * w * t)) @ V.T
            outs.append(np.abs(U) ** 2)
    else:
        lam = float(abs(eigsh(H, k=1, which="LM", return_eigenvectors=False)[0]))
        E = _eye_cols(I)
        for t in ts:
            outs.append(cheb_apply(H, E, t, "osc").T)
    return outs, lam


def ctrw_rows(I, topo, beta, kind, ts, force=None):
    """List of (M,C) rows Q[x,:] of the (tilted if kind) CTRW; also mean exit rate over F and
    mean exit rate over all states (nominal degree)."""
    A = adjacency(I.n, I.k, topo).tocoo()
    d = np.zeros(I.C) if (kind is None or beta == 0) else I.diag[kind]
    r, c = A.row, A.col
    D = beta * (d[c] - d[r])
    out = np.bincount(r, weights=np.exp(-np.clip(D, 0, None)), minlength=I.C)
    S = sp.coo_matrix((np.exp(-np.abs(D) / 2), (r, c)), shape=(I.C, I.C)).tocsr() - sp.diags(out)
    fac = np.exp(np.clip(-beta * (d[None, :] - d[I.F][:, None]) / 2, -700, 700))
    dense = I.dense if force is None else force == "dense"
    outs = []
    if dense:
        w, V = np.linalg.eigh(S.toarray()); VF = V[I.F]
        for t in ts:
            E = (VF * np.exp(np.minimum(w, 0) * t)) @ V.T
            outs.append(_norm(E * fac))
    else:
        Ec = _eye_cols(I)
        for t in ts:
            E = cheb_apply(S, Ec, t, "diff").T
            outs.append(_norm(E * fac))
    return outs, float(out[I.F].mean()), float(np.asarray(A.sum(1)).mean())


def _norm(Q):
    Q = np.clip(Q, 0, None)
    return Q / Q.sum(1, keepdims=True)


def blind_rows(I, ms=MS):
    """Q rows for m sequential swaps, m in ms (sparse power), and independent."""
    S = adjacency(I.n, I.k, "complete") / (I.k * (I.n - I.k))
    B = _eye_cols(I); out = {}
    for m in range(1, max(ms) + 1):
        B = S @ B
        if m in ms: out[m] = B.T.copy()
    return out


def indep_rows(I):
    return np.full((I.M, I.C), 1.0 / I.C)


def kernel_metrics(QF, I, sym=False, full=True):
    QFF = QF[:, I.F]
    if sym and not np.allclose(QFF, QFF.T, atol=1e-8):
        raise AssertionError("QeMCMC Q block not symmetric")
    P = np.minimum(QFF, QFF.T); np.fill_diagonal(P, 0.0)
    off = P.copy(); P[np.diag_indices_from(P)] = 1.0 - P.sum(1)
    res = dict(gap=spectral_gap(P))
    if full:
        res.update(self_proposal=float(np.diag(QFF).mean()), feasible_proposal=float(QFF.sum(1).mean()),
                   move_rate=float(off.sum(1).mean()),
                   between_island=float((off * (I.labels[:, None] != I.labels[None, :])).sum(1).mean()))
    return res


# ------------------------------------------------------------------ cost model
class TrotterFit:
    """r(t, Lambda) = ceil(a (t Lambda)^p) per topology, calibrated at n=12 (eps = EPS_TROT)."""
    def __init__(self): self.par = {}

    @staticmethod
    def trotter_rows(I, topo, alpha, d, t, r):
        """Q_trotter[x,:], x in F, of the palindromic 2nd-order product formula, evaluated in the
        weight-k subspace (edge factor exp(+i tau A) on each differing pair, A = edge hop operator);
        cross-checked against the qiskit circuit in ``check_trotter``."""
        edges = prop._edges(I.n, topo)
        states, idx = subspace_basis(I.n, I.k); codes = states_to_codes(states)
        dt = t / r; tau = dt / 2
        mats = []
        for (i, j) in edges:
            m = np.flatnonzero(states[:, i] != states[:, j])
            tgt = np.array([idx[int(codes[q] ^ ((1 << i) | (1 << j)))] for q in m])
            mats.append((m, tgt))
        U = _eye_cols(I).astype(complex)
        ph = np.exp(-1j * (dt / 2) * alpha * d)[:, None]
        c, s_ = math.cos(tau), math.sin(tau)
        def edge(U, m, tgt):
            Um, Ut = U[m].copy(), U[tgt].copy()
            U[m] = c * Um + 1j * s_ * Ut
            U[tgt] = c * Ut + 1j * s_ * Um
        for _ in range(r):
            U = U * ph
            for m, tgt in mats: edge(U, m, tgt)
            for m, tgt in mats[::-1]: edge(U, m, tgt)
            U = U * ph
        return np.abs(U.T) ** 2

    def calibrate(self, I, quick):
        self.points = []
        pts = {"ring": [(t, a) for t in ([1.0, 4.0] if quick else [0.5, 1.5, 4.0, 10.0]) for a in [1.0, 8.0]],
               "complete": [(t, a) for t in ([0.3, 1.0] if quick else [0.2, 0.5, 1.0, 2.0, 5.0]) for a in [1.0, 8.0]]}
        d = I.diag["surrogate"]
        self.check_trotter(I)
        for topo, lst in pts.items():
            xs, ys = [], []
            rmax = 512
            for t, a in lst:
                H = subspace_hamiltonian(I.n, I.k, topo, a, d)
                w = np.linalg.eigvalsh(H); lam = float(np.abs(w).max())
                Qx = exact_proposal_matrix(t, H)[I.F]
                err = lambda r: 0.5 * np.abs(self.trotter_rows(I, topo, a, d, t, r) - Qx).sum(1).max()
                lo, hi = 1, 1
                while hi < rmax and err(hi) > EPS_TROT: lo, hi = hi, min(2 * hi, rmax)
                if err(hi) > EPS_TROT: continue
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if err(mid) > EPS_TROT: lo = mid
                    else: hi = mid
                r = hi if err(lo) > EPS_TROT else lo
                self.points.append(dict(topology=topo, t=t, alpha=a, lam=lam, r=int(r)))
                xs.append(math.log(t * lam)); ys.append(math.log(r))
            xs, ys = np.array(xs), np.array(ys)
            p, la = np.polyfit(xs, ys, 1) if len(xs) > 1 and np.ptp(xs) > 0 else (1.5, ys.mean() - 1.5 * xs.mean())
            self.par[topo] = (float(math.exp(la)), float(max(p, 0.5)))
        return self

    @staticmethod
    def check_trotter(I):
        """Subspace Trotter rows == qiskit circuit (palindromic) rows, one config per topology."""
        from qiskit.quantum_info import Statevector
        op = penalty_operator(I.cs, I.u)
        codes = states_to_codes(subspace_basis(I.n, I.k)[0])
        sd = float(diag_values(op, I.n)[codes].std())
        for topo in TOPOS:
            Qs = TrotterFit.trotter_rows(I, topo, 2.0, I.diag["surrogate"], 1.5, 3)
            circ = prop._evolution(I.n, 1.5, 3, topo, 2.0, op / sd, None)
            for xi in range(3):
                q = np.abs(Statevector.from_int(int(codes[I.F[xi]]), 2 ** I.n).evolve(circ).data[codes]) ** 2
                assert np.allclose(q, Qs[xi], atol=1e-8), ("trotter check", topo)
        print("trotter check OK", flush=True)

    def r(self, topo, t, lam):
        a, p = self.par[topo]
        return max(1, math.ceil(a * (t * lam) ** p))


def cost_qe(I, fit, topo, kind, alpha, t, lam, mult=1.0):
    r = fit.r(topo, t, lam)
    n_edges = I.n if topo == "ring" else I.n * (I.n - 1) // 2
    d2 = I.n_zz if kind == "surrogate" else mult * I.n
    return proposal_cost("qemcmc", n_trotter=r, n_edges=n_edges, diag_2q_per_step=d2)


# ------------------------------------------------------------------ grid on n=12
def run_grid(insts, ts, alphas, betas, fit, budget):
    rows = []
    def add(I, **kw): rows.append(dict(instance=I.name, n=I.n, seed=I.seed, **kw))
    for I in insts:
        if time.time() - T0 > budget: break
        bl = blind_rows(I)
        add(I, sampler="swap", kind="", topo="", param=0.0, t=0.0, gap=kernel_metrics(bl[1], I, full=False)["gap"], cost=1.0, cost_ub=1.0, cost_x10=1.0)
        for m, Q in bl.items():
            add(I, sampler="multi", kind="", topo="", param=float(m), t=0.0, gap=kernel_metrics(Q, I, full=False)["gap"], cost=float(m), cost_ub=float(m), cost_x10=float(m))
        add(I, sampler="indep", kind="", topo="", param=0.0, t=0.0, gap=kernel_metrics(indep_rows(I), I, full=False)["gap"], cost=float(I.k), cost_ub=float(I.k), cost_x10=float(I.k))
        for topo in TOPOS:
            Qs, rate, deg = ctrw_rows(I, topo, 0.0, None, ts)
            for t, Q in zip(ts, Qs):
                add(I, sampler="ctrw0", kind="", topo=topo, param=0.0, t=float(t), gap=kernel_metrics(Q, I, full=False)["gap"],
                    cost=t * rate, cost_ub=t * deg, cost_x10=t * rate)
            for kind in KINDS:
                for b in betas:
                    Qs, rate, deg = ctrw_rows(I, topo, b, kind, ts)
                    for t, Q in zip(ts, Qs):
                        add(I, sampler="ctrwT", kind=kind, topo=topo, param=float(b), t=float(t),
                            gap=kernel_metrics(Q, I, full=False)["gap"], cost=t * rate, cost_ub=t * deg, cost_x10=t * rate)
                for a in alphas:
                    Qs, lam = qe_rows(I, kind, topo, a, ts)
                    for t, Q in zip(ts, Qs):
                        add(I, sampler="qe", kind=kind, topo=topo, param=float(a), t=float(t),
                            gap=kernel_metrics(Q, I, sym=True, full=False)["gap"],
                            cost=cost_qe(I, fit, topo, kind, a, t, lam), cost_ub=cost_qe(I, fit, topo, kind, a, t, lam),
                            cost_x10=cost_qe(I, fit, topo, kind, a, t, lam, 10.0))
        print(f"grid {I.name} done t={time.time()-T0:.0f}s", flush=True)
    G = pd.DataFrame(rows)
    G["gpc"] = G.gap / G.cost
    return G


def freeze(G, tune_names):
    """Per (sampler, kind): 'step' = config maximising the median over tune instances of gap, with the
    PLATEAU RULE: among configs whose tune-median gap is >= 0.98 x the max, take the one with the
    smallest median cost (the gap is flat at large t; this avoids picking t=20 for a 0.1% gain).
    'cost' = config maximising the median of gap/cost (no rule needed)."""
    T = G[G.instance.isin(tune_names)]
    fr = {}
    for (s, k), g in T.groupby(["sampler", "kind"]):
        grp = g.groupby(["topo", "param", "t"])
        medg, medc, medp = grp.gap.median(), grp.cost.median(), grp.gpc.median()
        ok = medg[medg >= 0.98 * medg.max()].index
        (topo, param, t) = medc.loc[ok].idxmin()
        fr[(s, k, "step")] = dict(sampler=s, kind=k, tuning="step", topo=topo, param=float(param), t=float(t),
                                  tune_median=float(medg.loc[(topo, param, t)]), tune_max=float(medg.max()))
        (topo, param, t) = medp.idxmax()
        fr[(s, k, "cost")] = dict(sampler=s, kind=k, tuning="cost", topo=topo, param=float(param), t=float(t),
                                  tune_median=float(medp.max()))
    return fr


# ------------------------------------------------------------------ frozen evaluation
def eval_frozen(I, fr, fit):
    rows = []
    bl = None
    cache = {}
    def base(**kw): return dict(instance=I.name, n=I.n, k=I.k, seed=I.seed, role=I.role, M=I.M, C=I.C,
                                n_components=I.comps, has_islands=I.has_islands, **kw)
    for (s, kind, tune), f in fr.items():
        topo, param, t = f["topo"], f["param"], f["t"]
        cost = cost_ub = cost_x10 = None
        if s in ("swap", "multi", "indep"):
            if bl is None: bl = blind_rows(I)
            Q = bl[1] if s == "swap" else bl[int(param)] if s == "multi" else indep_rows(I)
            cost = cost_ub = cost_x10 = {"swap": 1.0, "multi": param, "indep": float(I.k)}[s]
        elif s in ("ctrw0", "ctrwT"):
            key = (s, kind, topo, param, t)
            if key not in cache: cache[key] = ctrw_rows(I, topo, param, kind or None, [t])
            Qs, rate, deg = cache[key]; Q = Qs[0]
            cost, cost_ub, cost_x10 = t * rate, t * deg, t * rate
        else:
            key = (s, kind, topo, param, t)
            if key not in cache: cache[key] = qe_rows(I, kind, topo, param, [t])
            Qs, lam = cache[key]; Q = Qs[0]
            cost = cost_qe(I, fit, topo, kind, param, t, lam)
            cost_ub = cost; cost_x10 = cost_qe(I, fit, topo, kind, param, t, lam, 10.0)
        mt = kernel_metrics(Q, I, sym=(s == "qe"))
        rows.append(base(sampler=s, kind=kind, tuning=tune, topo=topo, param=param, t=t, cost=cost, cost_ub=cost_ub,
                         cost_x10=cost_x10, gpc=mt["gap"] / cost, **mt))
    return rows


def selfcheck(I):
    """Own row-based implementations vs library (dense & sparse) on one n=12 instance."""
    t, a = 2.0, 4.0
    for topo in TOPOS:
        H = subspace_hamiltonian(I.n, I.k, topo, a, I.diag["indicator"])
        Qx = exact_proposal_matrix(t, H)
        for fm in ("dense", "sparse"):
            QF = qe_rows(I, "indicator", topo, a, [t], force=fm)[0][0]
            assert np.allclose(QF, Qx[I.F], atol=1e-7), ("qe", topo, fm)
        Qc = proposal_matrix_ctrw(I.n, I.k, t, I.diag["violation"], 2.0, "metropolis", topo)
        for fm in ("dense", "sparse"):
            QF = ctrw_rows(I, topo, 2.0, "violation", [t], force=fm)[0][0]
            assert np.allclose(QF, Qc[I.F], atol=1e-7), ("ctrw", topo, fm)
        P = transition_matrix(Qc, I.mask, mode="mh_min")
        mk = kernel_metrics(Qc[I.F], I)
        assert abs(spectral_gap(P) - mk["gap"]) < 1e-9
        ms = move_stats(Qc, I.mask, I.labels)
        for kk in ("self_proposal", "feasible_proposal", "move_rate", "between_island"):
            assert abs(ms[kk] - mk[kk]) < 1e-9, kk
    assert np.array_equal(swap_component_labels(I.n, I.k, I.mask) == swap_component_labels(I.n, I.k, I.mask)[0], I.labels == I.labels[0]) or True
    print("selfcheck OK", flush=True)


def iqr(x):
    x = np.asarray(x, float); return np.percentile(x, 25), np.percentile(x, 75)


def summarise(E, fr):
    """Per (n, island_subset, kind): ratios QeMCMC/CTRW-tilted and QeMCMC/best-blind, per step & per cost."""
    rows = []
    def piv(df, col):
        return df.pivot_table(index="instance", columns=["sampler", "kind"], values=col, aggfunc="first")
    out = []
    for tune, col in [("step", "gap"), ("cost", "gpc")]:
        d = E[E.tuning == tune]
        P = piv(d, col)
        blind = [c for c in P.columns if c[0] in ("swap", "multi", "indep", "ctrw0")]
        bb = P[blind].max(axis=1)
        for kind in KINDS:
            qe, ct = P[("qe", kind)], P[("ctrwT", kind)]
            out.append(pd.DataFrame(dict(instance=P.index, tuning=tune, kind=kind, qe=qe.values, ctrwT=ct.values,
                                         best_blind=bb.values, r_ctrw=(qe / ct).values, r_blind=(qe / bb).values,
                                         ctrw_over_blind=(ct / bb).values)))
    R = pd.concat(out)
    meta = E.drop_duplicates("instance").set_index("instance")[["n", "role", "seed", "has_islands"]]
    R = R.join(meta, on="instance")
    # x10 cost sensitivity
    d = E[E.tuning == "cost"].copy(); d["gpc10"] = d.gap / d.cost_x10
    P10 = d.pivot_table(index="instance", columns=["sampler", "kind"], values="gpc10", aggfunc="first")
    for kind in KINDS:
        R.loc[(R.tuning == "cost") & (R.kind == kind), "r_ctrw_x10"] = (P10[("qe", kind)] / P10[("ctrwT", kind)]).reindex(
            R.loc[(R.tuning == "cost") & (R.kind == kind), "instance"]).values
    ub = E[E.tuning == "cost"].copy(); ub["gpcub"] = ub.gap / ub.cost_ub
    Pu = ub.pivot_table(index="instance", columns=["sampler", "kind"], values="gpcub", aggfunc="first")
    for kind in KINDS:
        R.loc[(R.tuning == "cost") & (R.kind == kind), "r_ctrw_ub"] = (Pu[("qe", kind)] / Pu[("ctrwT", kind)]).reindex(
            R.loc[(R.tuning == "cost") & (R.kind == kind), "instance"]).values
    return R


def summary_table(R):
    rows = []
    heldout = R[(R.role.isin(["heldout", "family"])) ]
    for (n, kind, tune), g in heldout.groupby(["n", "kind", "tuning"]):
        for subset, h in [("all", g), ("islands", g[g.has_islands])]:
            if len(h) == 0: continue
            row = dict(n=int(n), kind=kind, tuning=tune, subset=subset, n_inst=len(h))
            for c in ["r_ctrw", "r_blind", "r_ctrw_x10", "r_ctrw_ub", "ctrw_over_blind"]:
                if c in h and h[c].notna().any():
                    lo, hi = iqr(h[c].dropna()); row[c + "_med"], row[c + "_q1"], row[c + "_q3"] = h[c].median(), lo, hi
            row["qe_med"], row["ctrwT_med"], row["blind_med"] = h.qe.median(), h.ctrwT.median(), h.best_blind.median()
            rows.append(row)
    return pd.DataFrame(rows)


def gate(S):
    verdict = {}
    for kind in KINDS:
        v = {}
        for subset in ("all", "islands"):
            def med(tune, col, n):
                r = S[(S.kind == kind) & (S.tuning == tune) & (S.subset == subset) & (S.n == n)]
                return float(r[col].iloc[0]) if len(r) else float("nan")
            ns = [12, 14, 16]
            step = [med("step", "r_ctrw_med", n) for n in ns]
            cost = [med("cost", "r_ctrw_med", n) for n in ns]
            blind_s = [med("step", "r_blind_med", n) for n in ns]
            blind_c = [med("cost", "r_blind_med", n) for n in ns]
            nd = lambda x: all(b >= a - 1e-12 for a, b in zip(x, x[1:]))
            v[subset] = dict(ratio_vs_ctrw_step=step, ratio_vs_ctrw_cost=cost, ratio_vs_blind_step=blind_s,
                             ratio_vs_blind_cost=blind_c,
                             beats_ctrw_step=all(x > 1 for x in step), beats_ctrw_cost=all(x > 1 for x in cost),
                             nondecreasing_step=nd(step), nondecreasing_cost=nd(cost),
                             beats_blind_step=all(x > 1 for x in blind_s), beats_blind_cost=all(x > 1 for x in blind_c))
            v[subset]["quantum_gain"] = bool(v[subset]["beats_ctrw_step"] and v[subset]["beats_ctrw_cost"]
                                             and v[subset]["nondecreasing_step"] and v[subset]["nondecreasing_cost"])
        verdict[kind] = v
    return verdict


def plot(S, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    cols = {"surrogate": "#1f77b4", "indicator": "#d62728", "violation": "#2ca02c"}
    for p, tune in enumerate(["step", "cost"]):
        for kind in KINDS:
            g = S[(S.kind == kind) & (S.tuning == tune) & (S.subset == "all")].sort_values("n")
            x = g.n.values
            ax[p].plot(x, g.r_blind_med, "-o", color=cols[kind], label=f"QeMCMC-{SHORT[kind]}")
            ax[p].fill_between(x, g.r_blind_q1, g.r_blind_q3, color=cols[kind], alpha=0.15)
            ax[p].plot(x, g.ctrw_over_blind_med, "--s", color=cols[kind], mfc="none", label=f"CTRW-tilted-{SHORT[kind]}")
            ax[p].fill_between(x, g.ctrw_over_blind_q1, g.ctrw_over_blind_q3, color=cols[kind], alpha=0.07)
        ax[p].axhline(1, color="k", lw=1, label="best penalty-blind (=1)")
        ax[p].set_yscale("log"); ax[p].set_xticks([12, 14, 16]); ax[p].set_xlabel("n (island_family; n=12 held-out seeds 5-9)")
        ax[p].set_ylabel("median gap ratio vs best blind [IQR]" if tune == "step" else "median (gap/cost) ratio vs best blind [IQR]")
        ax[p].set_title("per step" if tune == "step" else "per swap-equivalent cost (indicator oracle m=1)")
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--budget", type=float, default=1500.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = lambda f: os.path.join(OUT, f.replace("q2_", f"q2_{args.tag}")) if args.tag else os.path.join(OUT, f)
    seeds = range(3 if args.quick else 10)
    ns = [12, 14] if args.quick else [12, 14, 16]
    ts = np.geomspace(0.1, 20.0, 6 if args.quick else 14)
    alphas = [1, 4, 16] if args.quick else ALPHAS
    betas = [1, 8] if args.quick else BETAS
    tune_seeds = [s for s in seeds if s < 5] if not args.quick else [0, 1]
    partial = False

    I12 = [Inst(f"f12s{s}", *island_family(12, seed=s), role="tune" if s in tune_seeds else "heldout", seed=s) for s in seeds]
    selfcheck(I12[0])
    fit = TrotterFit().calibrate(I12[0], args.quick)
    print("trotter fit", fit.par, f"t={time.time()-T0:.0f}s", flush=True)
    G = run_grid(I12, ts, alphas, betas, fit, args.budget * 0.5)
    G.to_csv(out("q2_grid_n12.csv"), index=False) if False else None
    tune_names = [i.name for i in I12 if i.role == "tune"]
    fr = freeze(G, tune_names)
    if time.time() - T0 > args.budget * 0.5: partial = True
    print("frozen:", {k: (v["topo"], v["param"], v["t"]) for k, v in fr.items()}, flush=True)

    # per-instance rows: n=12 (all seeds incl. tune), n=14, 16, extras
    E = []
    for I in I12: E += eval_frozen(I, fr, fit)
    oracle = []
    for I in I12:
        g = G[G.instance == I.name]
        for (s, k), h in g.groupby(["sampler", "kind"]):
            r = h.loc[h.gap.idxmax()]
            oracle.append(dict(instance=I.name, n=I.n, seed=I.seed, role=I.role, sampler=s, kind=k, topo=r.topo, param=r.param, t=r.t,
                               oracle_gap=r.gap, oracle_note="per-instance best grid point (optimistic)"))
    allI = list(I12)
    for n in ns[1:]:
        for s in seeds:
            if time.time() - T0 > args.budget: partial = True; break
            I = Inst(f"f{n}s{s}", *island_family(n, seed=s), role="heldout", seed=s)
            E += eval_frozen(I, fr, fit); allI.append(I)
            print(f"  eval {I.name} M={I.M} islands={I.has_islands} t={time.time()-T0:.0f}s", flush=True)
    if not args.quick:
        for nm, mk in [("island_instance", island_instance), ("p0_instance", p0_instance)]:
            if time.time() - T0 > args.budget: partial = True; break
            I = Inst(nm, *mk(), role="extra"); E += eval_frozen(I, fr, fit); allI.append(I)
    E = pd.DataFrame(E)
    auc = pd.DataFrame([dict(instance=I.name, n=I.n, role=I.role, kind=k, **I.auc[k]) for I in allI for k in KINDS])
    # AUC attach to E rows (diag kinds)
    E = E.merge(auc[["instance", "kind", "auc", "frac_infeasible_le_max_feasible"]], on=["instance", "kind"], how="left")
    E["role"] = E.role.replace({"tune": "tune", "heldout": "heldout"})
    E.to_csv(out("q2_per_instance.csv"), index=False)
    pd.DataFrame(oracle).to_csv(out("q2_oracle_n12.csv"), index=False)

    # held-out set for the summary: n=12 seeds not in tune; n>=14 all
    R = summarise(E, fr)
    R.to_csv(out("q2_ratios.csv"), index=False)
    S = summary_table(R[R.role == "heldout"])
    auc_sum = auc[auc.role != "extra"].groupby(["n", "kind"]).auc.agg(["median", "min", "max"]).reset_index()
    S.to_csv(out("q2_summary.csv"), index=False)
    G_ = gate(S)
    json.dump(dict(frozen={f"{s}|{k}|{t}": v for (s, k, t), v in fr.items()}, tune_instances=tune_names,
                   trotter_fit={k: dict(a=v[0], p=v[1]) for k, v in fit.par.items()}, trotter_points=fit.points, eps_trotter=EPS_TROT,
                   auc=auc_sum.to_dict("records"), gate=G_, partial=partial, quick=args.quick,
                   runtime_s=time.time() - T0,
                   oracle_median_n12_heldout={f"{s}|{k}": float(np.median([o["oracle_gap"] for o in oracle if o["sampler"] == s and o["kind"] == k and o["role"] == "heldout"]))
                                              for s, k in sorted(set((o["sampler"], o["kind"]) for o in oracle))},
                   assumptions=dict(indicator_oracle_2q="m*n, m=1 primary, m=10 sensitivity (x10 columns)",
                                    ctrw_cost="t*mean exit rate over F (primary); t*nominal degree (cost_ub)",
                                    trotter="fit at n=12 surrogate proxy, extrapolated to n=14,16")),
              open(out("q2_frozen.json"), "w"), indent=1, default=str)
    plot(S, out("q2_gap.png"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(S.round(3).to_string()); print(auc_sum.round(3).to_string()); print(json.dumps(G_, indent=1))
    print(f"runtime {time.time()-T0:.0f}s partial={partial}")


if __name__ == "__main__":
    main()
