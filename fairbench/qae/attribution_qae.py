"""Quantum amplitude estimation of attribution statistics (exact, simulated).

Statistic (linearity contract): S(x) = buy-and-hold equal-weight total return
= (1/k) sum_i x_i g_i - 1,  g_i = prod_t (1 + r_it)   (== attribute(..., rebalance=False)).
Over the Dicke state (uniform on all weight-k selections of the non-excluded assets,
C = C(n', k) states):

    a_F    = |F| / C                       feasible fraction
    a_G(s) = |F and {S < s}| / C
    percentile(s) = a_G(s) / a_F           (strict '<'; attribute() adds 0.5 * ties, a 0.5/|F| effect)
    null median   = smallest s with a_G(s)/a_F >= 1/2     (bisection on s)
    benchmark med = same with cardinality only (all n assets, a_F = 1)

AE runs are simulated (``fairbench.qae.estimators``): the oracle is ``SubspaceOracle`` (exact
Grover in the weight-k subspace, C <= ~5000) or ``IdealOracle(mask.mean())`` (provably the same
outcome law, sin^2((2m+1) theta)). Query conventions are those of estimators.py: one quantum
query = one application of Q (1 oracle call + 2 Dicke preps); one classical query = one
proposal (uniform random k-subset of the allowed assets) with its feasibility/score evaluation.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from ..apps.attribution import _returns_array
from ..constraints import Cardinality, ConstraintSet
from ..data import Universe
from .estimators import (AEResult, IdealOracle, SubspaceOracle, canonical_ae, classical_mc,
                         exp_schedule, iqae, linear_schedule, mlae, power_schedule,
                         ratio_estimate, _z)

__all__ = ["linear_return_scores", "portfolio_score", "ExactInstance", "exact_masks",
           "make_oracle", "run_ae", "qae_percentile", "qae_quantile", "qae_attribute",
           "classical_percentile", "classical_quantile", "QAEResult", "QAEAttribution",
           "RankTable", "iid_shared_sample", "SwapChainBank", "swap_connectivity", "exhaustive_cost"]


# ============================================================================ scores
def linear_return_scores(u: Universe) -> np.ndarray:
    """Per-asset buy-and-hold total return g_i - 1, g_i = prod_t (1 + r_it)."""
    return np.prod(1.0 + _returns_array(u), axis=0) - 1.0


def portfolio_score(x: np.ndarray, u: Universe | None = None, scores: np.ndarray | None = None) -> float:
    """S(x) = mean over held assets of g_i, minus 1 (equal-weight buy-and-hold total return).
    ``x``: 0/1 selection (or any positive holdings vector; support is used)."""
    if scores is None:
        scores = linear_return_scores(u)
    sel = np.asarray(x) > 0
    return float(np.mean(scores[sel] + 1.0) - 1.0)


# ============================================================================ exact instance
@dataclass
class ExactInstance:
    """Enumerated weight-k selections of the allowed assets with scores and feasibility."""
    n: int
    k: int
    allowed: np.ndarray           # asset indices
    idx: np.ndarray               # (C, k) positions into ``allowed`` ... stored as asset indices
    scores: np.ndarray            # (C,)  S(x)
    feasible: np.ndarray          # (C,)  bool
    g1: np.ndarray                # per-asset return (g_i - 1)
    _sf: np.ndarray = field(default=None, repr=False)

    @classmethod
    def build(cls, u: Universe, cs: ConstraintSet, cardinality_only: bool = False,
              max_states: int = 3_000_000) -> "ExactInstance":
        k = cs.cardinality
        if k is None:
            raise ValueError("constraint set needs a Cardinality rule")
        allowed = np.arange(u.n) if cardinality_only else cs.allowed_indices(u.n)
        C = math.comb(len(allowed), k)
        if C > max_states:
            raise ValueError(f"C(n'={len(allowed)}, k={k}) = {C} too large to enumerate")
        g1 = linear_return_scores(u)
        comb = np.fromiter(itertools.chain.from_iterable(itertools.combinations(allowed.tolist(), k)),
                           dtype=np.int16, count=C * k).reshape(C, k)
        scores = (g1[comb] + 1.0).mean(axis=1) - 1.0
        if cardinality_only:
            feas = np.ones(C, bool)
        else:
            feas = np.zeros(C, bool)
            ch = 50_000
            for i in range(0, C, ch):
                c = comb[i:i + ch]
                X = np.zeros((len(c), u.n), np.uint8)
                np.put_along_axis(X, c.astype(np.int64), 1, axis=1)
                feas[i:i + ch] = cs.check_batch(X, u)
        return cls(u.n, k, allowed, comb, scores, feas, g1, np.sort(scores[feas]))

    # ---- exact statistics
    @property
    def C(self) -> int:
        return len(self.scores)

    @property
    def a_F(self) -> float:
        return float(self.feasible.mean())

    def mask_below(self, s: float) -> np.ndarray:
        return self.feasible & (self.scores < s)

    def a_G(self, s: float) -> float:
        return float(np.searchsorted(self._sf, s, side="left")) / self.C

    def percentile(self, s: float, ties: str = "strict") -> float:
        """fraction of feasible with S < s ('strict'), or attribute()'s '<' + 0.5 '==' ('half')."""
        lo = np.searchsorted(self._sf, s, side="left")
        if ties == "strict":
            return float(lo / len(self._sf))
        hi = np.searchsorted(self._sf, s, side="right")
        return float((lo + 0.5 * (hi - lo)) / len(self._sf))

    def quantile(self, q: float = 0.5) -> float:
        """smallest s with fraction(S < s) >= q  == sorted[ceil(q m) - 1] (lower q-quantile)."""
        m = len(self._sf)
        return float(self._sf[max(int(math.ceil(q * m)) - 1, 0)])

    def score_range(self) -> tuple[float, float]:
        """data-only bounds on S over weight-k selections of the allowed assets."""
        v = np.sort(self.g1[self.allowed])
        return float(v[:self.k].mean()), float(v[-self.k:].mean())


def exact_masks(u: Universe, cs: ConstraintSet, s: float, cardinality_only: bool = False,
                inst: ExactInstance | None = None):
    """(feasible mask, below-threshold mask = feasible & S < s, scores) over the C states."""
    inst = inst or ExactInstance.build(u, cs, cardinality_only)
    return inst.feasible, inst.mask_below(s), inst.scores


# ============================================================================ oracle / AE dispatch
def make_oracle(mask: np.ndarray, kind: str = "auto", subspace_max: int = 5000):
    """kind: 'subspace' (exact state-vector Grover) | 'ideal' (IdealOracle(mask.mean()), same law)
    | 'auto' (subspace if C <= subspace_max else ideal)."""
    if kind == "subspace" or (kind == "auto" and len(mask) <= subspace_max):
        return SubspaceOracle(mask)
    return IdealOracle(float(np.mean(mask)))


def run_ae(oracle, method: str, rng: np.random.Generator, eps: float | None = None,
           alpha: float = 0.05, K: int | None = None, shots: int | None = None,
           beta: float = 0.5, n_eval: int | None = None, n_shots: int = 100) -> AEResult:
    """One AE run. method: classical | iqae(eps) | mlae/mlae_exp/mlae_linear/mlae_power(K, shots) |
    canonical(n_eval, shots)."""
    if method == "classical":
        return classical_mc(oracle, int(shots), rng)
    if method == "iqae":
        return iqae(oracle, eps, alpha, rng, n_shots=n_shots)
    if method in ("mlae", "mlae_exp"):
        return mlae(oracle, exp_schedule(K), shots, rng)
    if method == "mlae_linear":
        return mlae(oracle, linear_schedule(K), shots, rng)
    if method == "mlae_power":
        return mlae(oracle, power_schedule(K, beta), shots, rng)
    if method == "canonical":
        return canonical_ae(oracle, n_eval, shots, rng)
    raise ValueError(method)


def _adaptive_iqae(oracle, rng, alpha: float = 0.05, abs_eps: float = 0.0, rel_eps: float = 0.0,
                   eps0: float = 0.1, max_rounds: int = 14, n_shots: int = 100):
    """IQAE with an amplitude-adaptive stopping rule: repeat until the a-CI half-width is
    <= max(abs_eps, rel_eps * a_hat). ``iqae``'s ``eps`` is a bound on the *angle* theta
    (width 2 eps rad), and delta_a ~ 2 sqrt(a(1-a)) delta_theta, so for small a the angle
    target is eps_theta = target_hw / (2 sqrt(a_hat)): this is what makes the cost scale as
    1/(rel sqrt(a)) instead of 1/(rel a). Pilot runs are charged (all runs returned)."""
    runs, eps = [], eps0
    for _ in range(max_rounds):
        r = iqae(oracle, min(eps, 0.45), alpha, rng, n_shots=n_shots)
        runs.append(r)
        hw = 0.5 * (r.ci[1] - r.ci[0])
        tgt = max(abs_eps, rel_eps * r.estimate, 1e-12)
        if hw <= tgt:
            break
        a_ref = max(r.estimate, r.ci[0], 1e-12)
        eps = min(eps / 2, 0.9 * tgt / (2 * math.sqrt(a_ref * (1 - a_ref) + 1e-12)))
    return runs


# ============================================================================ percentile
@dataclass
class QAEResult:
    estimate: float
    ci: tuple
    oracle_queries: int
    details: dict = field(default_factory=dict)


def _fund_score(fund, u, scores=None) -> float:
    if np.ndim(fund) == 0:
        return float(fund)
    return portfolio_score(fund, u, scores)


def qae_percentile(u: Universe, cs: ConstraintSet, fund, method: str = "iqae",
                   eps: float = 0.01, alpha: float = 0.05, K: int = 8, shots: int = 50,
                   beta: float = 0.5, n_eval: int = 6, rng: np.random.Generator | None = None,
                   inst: ExactInstance | None = None, oracle_kind: str = "auto",
                   conf: float = 0.95, n_shots: int = 100) -> QAEResult:
    """Percentile (fraction in [0,1]) of ``fund`` (holdings vector or score) = a_G / a_F, two AE
    runs on the Dicke state. iqae: ``eps`` = target absolute percentile error; a_F is first
    estimated to relative precision eps/2 (pilot runs charged), a_G to absolute eps*a_F/2.
    Other methods: the same (K, shots | n_eval) is used for both runs.
    The ratio CI is the delta-method CI of ``ratio_estimate`` (independent runs assumed)."""
    rng = rng or np.random.default_rng(0)
    inst = inst or ExactInstance.build(u, cs)
    s = _fund_score(fund, u, inst.g1)
    o_F = make_oracle(inst.feasible, oracle_kind)
    o_G = make_oracle(inst.mask_below(s), oracle_kind)
    if method == "iqae":
        runs_F = _adaptive_iqae(o_F, rng, alpha, rel_eps=min(0.5, eps / 2), n_shots=n_shots)
        rF = runs_F[-1]
        runs_G = _adaptive_iqae(o_G, rng, alpha, abs_eps=eps * rF.estimate / 2, n_shots=n_shots)
        rG = runs_G[-1]
        extra_q = sum(r.oracle_queries for r in runs_F[:-1] + runs_G[:-1])
        extra_sp = sum(r.state_preps for r in runs_F[:-1] + runs_G[:-1])
    else:
        rF = run_ae(o_F, method, rng, K=K, shots=shots, beta=beta, n_eval=n_eval)
        rG = run_ae(o_G, method, rng, K=K, shots=shots, beta=beta, n_eval=n_eval)
        runs_F, extra_q, extra_sp = [rF], 0, 0
    if rF.estimate <= 0:
        return QAEResult(float("nan"), (0.0, 1.0), rF.oracle_queries + rG.oracle_queries + extra_q,
                         dict(a_F=rF, a_G=rG, failed="a_F estimate is 0"))
    r = ratio_estimate(rG, rF, conf=conf)
    q = r.oracle_queries + extra_q
    return QAEResult(r.estimate, r.ci, q,
                     dict(a_F=rF, a_G=rG, ratio=r, pilot_queries=extra_q,
                          state_preps=r.state_preps + extra_sp, s=s,
                          exact=inst.percentile(s, "strict"), method=method))


def classical_percentile(inst: ExactInstance, s: float, N: int, rng: np.random.Generator) -> float:
    """Rejection sampling with N proposals (queries): uniform random weight-k subsets of the
    allowed assets; keep feasible; empirical fraction below s. Returns 0.5 if none feasible.
    Sampled exactly from the multinomial (below, feasible-not-below, infeasible)."""
    pG, pF = inst.a_G(s), inst.a_F
    cnt = rng.multinomial(int(N), [pG, max(pF - pG, 0.0), max(1.0 - pF, 0.0)])
    nF = cnt[0] + cnt[1]
    return float(cnt[0] / nF) if nF else 0.5


# ============================================================================ quantile
def qae_quantile(u: Universe, cs: ConstraintSet, q: float = 0.5, method: str = "iqae",
                 eps: float = 0.02, alpha: float = 0.05, n_bisect: int = 14, K: int = 8,
                 shots: int = 50, rng: np.random.Generator | None = None,
                 cardinality_only: bool = False, inst: ExactInstance | None = None,
                 oracle_kind: str = "auto", rel_F: float | None = None,
                 n_shots: int = 100) -> QAEResult:
    """q-quantile of S over the feasible set (or over all weight-k selections if
    ``cardinality_only``; then a_F = 1 and no feasible-fraction AE is run) by bisection on s.

    iqae: a_F estimated once (relative precision ``rel_F`` = eps); every bisection step then
    decides 'a_G(s)/a_F >= q' with a sequential test: IQAE at rank precision 0.25, 0.125, ...
    down to ``eps``, stopping as soon as the CI excludes q (far-from-q steps are cheap);
    at the floor the point estimate decides. Other methods: fixed (K, shots) per step.
    Returns the bracket midpoint after ``n_bisect`` steps; details has total queries and the
    exact quantile / rank error."""
    rng = rng or np.random.default_rng(0)
    inst = inst or ExactInstance.build(u, cs, cardinality_only)
    cardinality_only = bool(np.all(inst.feasible))
    q_total = 0
    if cardinality_only:
        aF = 1.0
    else:
        o_F = make_oracle(inst.feasible, oracle_kind)
        if method == "iqae":
            runs = _adaptive_iqae(o_F, rng, alpha, rel_eps=min(0.5, rel_F or eps), n_shots=n_shots)
            rF = runs[-1]
            q_total += sum(r.oracle_queries for r in runs)
        else:
            rF = run_ae(o_F, method, rng, K=K, shots=shots)
            q_total += rF.oracle_queries
        aF = rF.estimate
        if aF <= 0:
            return QAEResult(float("nan"), (float("nan"),) * 2, q_total, dict(failed="a_F=0"))
    lo, hi = inst.score_range()
    pad = 1e-9 + 1e-6 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    steps = []
    for _ in range(n_bisect):
        s = 0.5 * (lo + hi)
        o_G = make_oracle(inst.mask_below(s), oracle_kind)
        if method == "iqae":
            e, above = 0.25, None
            while True:
                runs_s = _adaptive_iqae(o_G, rng, alpha, abs_eps=e * aF, n_shots=n_shots)
                r = runs_s[-1]
                q_total += sum(x.oracle_queries for x in runs_s)
                if r.ci[1] / aF < q:
                    above = False
                elif r.ci[0] / aF >= q:
                    above = True
                elif e <= eps:
                    above = r.estimate / aF >= q
                if above is not None:
                    break
                e /= 2
        else:
            r = run_ae(o_G, method, rng, K=K, shots=shots)
            q_total += r.oracle_queries
            above = r.estimate / aF >= q
        steps.append((s, above))
        if above:
            hi = s
        else:
            lo = s
    est = 0.5 * (lo + hi)
    ex = inst.quantile(q)
    rank = inst.percentile(est, "strict")
    return QAEResult(est, (lo, hi), q_total,
                     dict(exact=ex, rank_error=abs(rank - q), rank=rank, a_F_est=aF,
                          steps=steps, method=method, cardinality_only=cardinality_only,
                          resolution=hi - lo))


def classical_quantile(inst: ExactInstance, q: float, N: int, rng: np.random.Generator) -> float:
    """N proposals (uniform random k-subsets of the allowed assets), keep feasible, empirical
    lower q-quantile of S. NaN if none feasible."""
    i = rng.integers(0, inst.C, size=int(N))
    sc = inst.scores[i[inst.feasible[i]]]
    if sc.size == 0:
        return float("nan")
    return float(np.sort(sc)[max(int(math.ceil(q * sc.size)) - 1, 0)])


# ============================================================================ fair classical baselines
def exhaustive_cost(inst: ExactInstance) -> int:
    """Classical enumeration: C(n_allowed, k) feasibility+score evaluations give the EXACT
    percentile and medians (zero error). Beats every sampler when this is small."""
    return math.comb(len(inst.allowed), inst.k)


class RankTable:
    """Per-state lookup for the feasible set F, ordered by score: ``bins`` equal-count rank bins
    (bin of state = floor(rank * B / |F|), -1 if infeasible) and the 'below the fund' flag.
    Median/percentile estimators from bin-count histograms: the median-rank resolution is
    1/(2B) <= 2.5e-4 for B=2000 (<< the 5e-3 tolerances used)."""

    def __init__(self, inst: ExactInstance, s: float, bins: int = 2000):
        self.inst, self.s = inst, s
        F = np.flatnonzero(inst.feasible)
        self.nF = len(F)
        self.B = B = max(1, min(bins, self.nF))
        order = F[np.argsort(inst.scores[F], kind="stable")]
        self.bin = np.full(inst.C, -1, np.int32)
        self.bin[order] = (np.arange(self.nF) * B // self.nF).astype(np.int32)
        self.below = inst.feasible & (inst.scores < s)
        self.exact_pct = float(self.below.sum() / self.nF)

    def pmf(self) -> np.ndarray:
        """probabilities of the 2B+1 categories (bin*2+below ; last = infeasible) under one
        uniform proposal over all C weight-k selections."""
        cat = np.where(self.bin >= 0, self.bin * 2 + self.below, 2 * self.B)
        return np.bincount(cat, minlength=2 * self.B + 1) / self.inst.C

    def median_rank_error(self, hist: np.ndarray) -> np.ndarray:
        """hist (..., B) feasible counts per rank bin -> |rank of sample median - 1/2|
        (NaN if empty)."""
        c = np.cumsum(hist, axis=-1)
        n = c[..., -1]
        m = (c >= 0.5 * n[..., None]).argmax(axis=-1)
        err = np.abs((m + 0.5) / self.B - 0.5)
        return np.where(n > 0, err, np.nan)


def iid_shared_sample(tab: RankTable, N: int, rng: np.random.Generator):
    """ONE i.i.d. rejection sample set of N uniform proposals (exact multinomial draw), shared by
    both statistics. Returns (percentile estimate, median-rank error, #feasible); estimates are
    NaN if no proposal was feasible."""
    cnt = rng.multinomial(int(N), tab.pmf())
    nF = int(cnt[:-1].sum())
    if nF == 0:
        return float("nan"), float("nan"), 0
    pair = cnt[:-1].reshape(-1, 2)
    return (float(pair[:, 1].sum() / nF),
            float(tab.median_rank_error(pair.sum(axis=1))), nF)


def swap_connectivity(inst: ExactInstance) -> dict:
    """Connected components of F under single swaps (one held asset <-> one free allowed asset).
    A swap Metropolis chain started in a component never leaves it. Returns n_components,
    largest component fraction of |F|, and ``start_coverage`` = E[fraction of F reachable from
    a start drawn uniformly from F] = sum_c (|c|/|F|)^2."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    allowed = inst.allowed.astype(np.int64)
    nA, k = len(allowed), inst.k
    masks = (np.int64(1) << inst.idx.astype(np.int64)).sum(axis=1)
    order = np.argsort(masks)
    sm = masks[order]
    feas = inst.feasible[order]
    fpos = np.flatnonzero(feas)
    comp_id = np.full(len(sm), -1, np.int64)
    comp_id[fpos] = np.arange(len(fpos))
    idx = inst.idx[order][fpos].astype(np.int64)             # (nF, k)
    rows, cols = [], []
    # held -> mask of free assets per state
    m_held = (np.int64(1) << idx).sum(axis=1)
    allmask = (np.int64(1) << allowed).sum()
    m_free = allmask & ~m_held
    free_assets = [a for a in allowed.tolist()]
    for ai in range(k):
        i = idx[:, ai]
        for j in free_assets:
            has = (m_free >> j) & 1
            sel = np.flatnonzero(has)
            nm = m_held[sel] ^ (np.int64(1) << i[sel]) ^ (np.int64(1) << j)
            pos = np.searchsorted(sm, nm)
            ok = feas[pos]
            rows.append(sel[ok]); cols.append(comp_id[pos[ok]])
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    nF = len(fpos)
    g = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(nF, nF))
    nc, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab) / nF
    return dict(n_components=int(nc), largest=float(sizes.max()), start_coverage=float((sizes ** 2).sum()),
                n_isolated=int((np.bincount(lab) == 1).sum()))


class SwapChainBank:
    """R independent swap-move Metropolis chains on F (uniform target; propose swapping a random
    held asset with a random free allowed asset, accept iff feasible, otherwise stay).
    One classical query = one proposed move incl. its feasibility check. Each chain starts at a
    state drawn uniformly from F, whose rejection-sampling cost (Geometric(P_F) proposals,
    drawn per chain) is CHARGED. Estimator at move-checkpoint T: uses the states visited during
    moves (T/4, T], i.e. a burn-in of T/4 moves, also charged. All chains are advanced in
    lockstep with vectorised numpy (identical law to ``baselines.mcmc_swap_sample``, but the
    feasibility check is a lookup in the enumerated table)."""

    def __init__(self, tab: RankTable):
        self.tab = tab
        inst = tab.inst
        masks = (np.int64(1) << inst.idx.astype(np.int64)).sum(axis=1)
        self.order = np.argsort(masks)
        self.sm = masks[self.order]
        self.bin_s = tab.bin[self.order]
        self.below_s = tab.below[self.order]
        self.idx_s = inst.idx[self.order].astype(np.int64)
        self.fpos = np.flatnonzero(self.bin_s >= 0)

    def run(self, R: int, t_max: int, rng: np.random.Generator, j0: int = 8, back: int = 4):
        """-> dict(T=checkpoints, pct_err (R,len), med_err (R,len), init_q (R,)).
        Checkpoints T_j = round(2^(j/2)); burn-in T_{j-back} = T_j/4 (back=4)."""
        tab, inst = self.tab, self.tab.inst
        k, B = inst.k, tab.B
        allowed = inst.allowed.astype(np.int64)
        nf = len(allowed) - k
        pos = rng.choice(self.fpos, size=R)
        init_q = rng.geometric(tab.nF / inst.C, size=R)
        held = self.idx_s[pos].copy()
        free = np.empty((R, nf), np.int64)
        for r in range(R):
            free[r] = np.setdiff1d(allowed, held[r])
        mask = self.sm[pos].copy()
        ar = np.arange(R)
        hist = np.zeros((R, B), np.int32)
        below = np.zeros(R, np.int64)
        Ts = [int(round(2 ** (j / 2))) for j in range(0, 200) if int(round(2 ** (j / 2))) <= t_max]
        Ts = sorted(set(Ts))
        cp = {T: i for i, T in enumerate(Ts)}
        snaps = {}
        one = np.int64(1)
        blk = 2048
        for t in range(1, t_max + 1):
            if (t - 1) % blk == 0:
                A_blk = rng.integers(k, size=(blk, R)); B_blk = rng.integers(nf, size=(blk, R))
            a, b = A_blk[(t - 1) % blk], B_blk[(t - 1) % blk]
            i = held[ar, a]; j = free[ar, b]
            nm = mask ^ (one << i) ^ (one << j)
            p = np.searchsorted(self.sm, nm)
            ok = self.bin_s[p] >= 0
            sel = np.flatnonzero(ok)
            if sel.size:
                held[sel, a[sel]] = j[sel]
                free[sel, b[sel]] = i[sel]
                mask[sel] = nm[sel]
                pos[sel] = p[sel]
            hist[ar, self.bin_s[pos]] += 1
            below += self.below_s[pos]
            if t in cp:
                snaps[t] = (hist.copy(), below.copy())
        T_out, pe, me = [], [], []
        for idx_j, T in enumerate(Ts):
            if idx_j - back < 0 or Ts[idx_j - back] < 1:
                continue
            T0 = Ts[idx_j - back]
            h1, b1 = snaps[T]; h0, b0 = snaps[T0]
            n = T - T0
            pe.append(np.abs((b1 - b0) / n - tab.exact_pct))
            me.append(tab.median_rank_error(h1 - h0))
            T_out.append(T)
        return dict(T=np.array(T_out), pct_err=np.array(pe).T, med_err=np.array(me).T, init_q=init_q)


# ============================================================================ top level
@dataclass
class QAEAttribution:
    percentile: float                 # percent, like AttributionResult
    null_median: float
    benchmark_median: float
    constraint_effect: float
    manager_effect: float
    fund: float
    queries: dict                     # percentile / null_median / benchmark_median / total
    exact: dict
    classical: dict                   # same-budget classical estimates
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        e, c = self.exact, self.classical
        return "\n".join([
            f"  fund buy-and-hold return {self.fund:+.2%}   quantum queries {self.queries}",
            f"  {'':<20}{'QAE':>10}{'exact':>10}{'classical (shared set)':>24}",
            f"  {'percentile (%)':<20}{self.percentile:>10.2f}{e['percentile']:>10.2f}{c['percentile']:>24.2f}",
            f"  {'null median':<20}{self.null_median:>10.2%}{e['null_median']:>10.2%}{c['null_median']:>24.2%}",
            f"  {'benchmark median':<20}{self.benchmark_median:>10.2%}{e['benchmark_median']:>10.2%}"
            f"{c['benchmark_median']:>24.2%}",
            f"  {'constraint effect':<20}{self.constraint_effect:>+10.2%}{e['constraint_effect']:>+10.2%}"
            f"{c['constraint_effect']:>+24.2%}",
            f"  {'manager effect':<20}{self.manager_effect:>+10.2%}{e['manager_effect']:>+10.2%}"
            f"{c['manager_effect']:>+24.2%}"])


def qae_attribute(u: Universe, cs: ConstraintSet, fund, method: str = "iqae",
                  eps_pct: float = 0.02, eps_med: float = 0.02, alpha: float = 0.05,
                  n_bisect: int = 14, K: int = 8, shots: int = 50,
                  rng: np.random.Generator | None = None, oracle_kind: str = "auto") -> QAEAttribution:
    """Percentile, null/benchmark median, constraint and manager effects by AE, with the exact
    values and the classical i.i.d. rejection estimates: one shared sample set (percentile + null
    median) of max(quantum percentile, quantum null-median) proposals, plus a cardinality-only
    set for the benchmark median. Total-return, equal-weight, buy-and-hold (rebalance=False) statistic only."""
    rng = rng or np.random.default_rng(0)
    inst = ExactInstance.build(u, cs)
    inst_b = ExactInstance.build(u, cs, cardinality_only=True)
    s = _fund_score(fund, u, inst.g1)
    p = qae_percentile(u, cs, s, method, eps_pct, alpha, K, shots, rng=rng, inst=inst,
                       oracle_kind=oracle_kind)
    nm = qae_quantile(u, cs, 0.5, method, eps_med, alpha, n_bisect, K, shots, rng, False, inst,
                      oracle_kind)
    bm = qae_quantile(u, cs, 0.5, method, eps_med, alpha, n_bisect, K, shots, rng, True, inst_b,
                      oracle_kind)
    ce, me = nm.estimate - bm.estimate, s - nm.estimate
    ex_nm, ex_bm = inst.quantile(0.5), inst_b.quantile(0.5)
    exact = dict(percentile=100 * inst.percentile(s, "strict"),
                 percentile_half_ties=100 * inst.percentile(s, "half"),
                 null_median=ex_nm, benchmark_median=ex_bm, constraint_effect=ex_nm - ex_bm,
                 manager_effect=s - ex_nm, P_F=inst.a_F, C=inst.C)
    # Fair classical comparison: ONE shared i.i.d. rejection set (percentile AND null median come
    # from the same proposals), sized to the larger of the two quantum budgets; the benchmark
    # median is a separate cardinality-only i.i.d. set (no feasibility test) with the quantum
    # benchmark budget. Also reported: the (unfair-to-classical) separate-sets accounting.
    N_shared = max(p.oracle_queries, nm.oracle_queries)
    tab = RankTable(inst, s)
    # percentile and null median both come from this one multinomial draw (the shared set)
    cnt = rng.multinomial(int(N_shared), tab.pmf())
    pair = cnt[:-1].reshape(-1, 2)
    nF = int(pair.sum())
    if nF:
        cp = float(pair[:, 1].sum() / nF)
        hist = pair.sum(axis=1)
        m = int(np.searchsorted(np.cumsum(hist), 0.5 * nF))
        sc_sorted = np.sort(inst.scores[inst.feasible])
        cnm = float(sc_sorted[min(int((m + 0.5) / tab.B * len(sc_sorted)), len(sc_sorted) - 1)])
    else:
        cp, cnm = float("nan"), float("nan")
    cbm = classical_quantile(inst_b, 0.5, bm.oracle_queries, rng)
    classical = dict(percentile=100 * cp, null_median=cnm, benchmark_median=cbm,
                     constraint_effect=cnm - cbm, manager_effect=s - cnm,
                     queries=dict(shared_set=N_shared, benchmark=bm.oracle_queries,
                                  total=N_shared + bm.oracle_queries,
                                  separate_sets_total=p.oracle_queries + nm.oracle_queries + bm.oracle_queries),
                     feasible_in_shared_set=nF)
    tot = p.oracle_queries + nm.oracle_queries + bm.oracle_queries
    return QAEAttribution(100 * p.estimate, nm.estimate, bm.estimate, ce, me, s,
                          dict(percentile=p.oracle_queries, null_median=nm.oracle_queries,
                               benchmark_median=bm.oracle_queries, total=tot),
                          exact, classical,
                          dict(percentile=p, null_median=nm, benchmark_median=bm))
