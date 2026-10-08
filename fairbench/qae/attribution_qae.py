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
           "make_oracle", "run_ae", "qae_percentile", "qae_quantile", "qae_quantile_hybrid", "qae_attribute",
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


def _ae_to_halfwidth(oracle, eps_a: float, a_guess: float, rng, alpha: float = 0.05,
                     ae: str = "iqae", n_shots: int = 100, mlae_shots: int = 30) -> AEResult:
    """One AE run aiming at an absolute amplitude CI half-width ``eps_a`` for an amplitude near
    ``a_guess`` (no pilot; the guess comes from the classical warm start). iqae: eps_theta =
    eps_a / (2 sqrt(a (1-a))). mlae: smallest exponential schedule (m = 0,1,2,..,2^(K-2), shots
    per m) whose Fisher-information half-width z sqrt(a(1-a)) / sqrt(sum shots (2m+1)^2) <= eps_a."""
    a_g = min(max(a_guess, eps_a / 2), 1 - eps_a / 2)
    if ae == "iqae":
        return iqae(oracle, min(0.45, eps_a / (2 * math.sqrt(a_g * (1 - a_g)))), alpha, rng, n_shots=n_shots)
    z = _z(1 - alpha)
    for K in range(2, 40):
        sched = exp_schedule(K)
        info = sum(mlae_shots * (2 * m + 1) ** 2 for m in sched)
        if z * math.sqrt(a_g * (1 - a_g) / info) <= eps_a:
            break
    return mlae(oracle, sched, mlae_shots, rng, conf=1 - alpha)


def _isotonic(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) least-squares fit of y on sorted x (pool adjacent
    violators). Returns fitted values in the order of x."""
    o = np.argsort(x, kind="stable")
    vals, wts, cnt = [], [], []
    for yi, wi in zip(np.asarray(y, float)[o], np.asarray(w, float)[o]):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / (wts[-1] + w2)
            wts[-1] += w2; cnt[-1] += c2
    fit_sorted = np.repeat(vals, cnt)
    out = np.empty(len(fit_sorted)); out[o] = fit_sorted
    return out


def qae_quantile_hybrid(u: Universe, cs: ConstraintSet, q: float = 0.5, eps: float = 0.01,
                        m0: int = 40, alpha: float = 0.05, shrink: float = 0.5, max_iter: int = 12,
                        rng: np.random.Generator | None = None, cardinality_only: bool = False,
                        inst: ExactInstance | None = None, oracle_kind: str = "auto",
                        rel_F: float | None = None, n_shots: int = 100,
                        ae: str = "iqae", mlae_shots: int = 30, aF_run: AEResult | None = None,
                        max_fallback: int | None = None) -> QAEResult:
    """Hybrid classical/quantum q-quantile (default q = 1/2): warm-started monotone rank-curve
    root finding, with a charged bisection fallback.

    1. CLASSICAL WARM START (charged): propose uniform weight-k subsets until ``m0`` feasible ones
       are seen (N0 ~ m0 / P_F proposals; ``cardinality_only``: N0 = m0). Their sorted scores give a
       piecewise-linear empirical CDF F0 and quantile function QF0; thresholds are parametrised by
       their empirical rank t = F0(s) (s = QF0(t)).
    2. a_F by IQAE to relative precision ``rel_F`` (default eps), unless ``aF_run`` (a finished
       a_F AE run, e.g. the percentile's) is supplied -- then it is reused and NOT charged again.
    3. Root-finding loop (at most ``max_iter`` AE runs): at t_1 = q, then at the current root,
       ONE AE of a_G(QF0(t)) to rank half-width h = max(eps, shrink * gap) (gap_0 = 1/sqrt(m0);
       gap = |q - r| + h/2 after each run; amplitude precision warm-started from the classical
       a_G guess, no AE pilot). Measured ranks r_j = a_G/a_F at t_j are fitted by a MONOTONE
       (non-decreasing) rank curve: weighted isotonic regression of r_j on t_j with weights 1/h_j^2
       plus the exact anchors r(0) = 0, r(1) = 1, linearly interpolated. The next t is the root of
       fit(t) = q, clipped to the open bracket of measured thresholds whose rank CI excludes q
       (largest t with CI above-below q, smallest with CI wholly above). Coarse early points
       therefore cannot drag the root (the old offset interpolation through all points was
       non-monotone and stalled ~10% of runs at rank error 0.03-0.07).
       CONVERGED when the last run had h <= eps and |q - r_last| <= eps.
    4. FALLBACK (only if not converged after ``max_iter`` runs): charged bisection in t between the
       CI bracket, one AE at h = eps per step, until converged or ``max_fallback`` steps
       (default 2 ceil(log2(1/eps)) + 4) or the bracket collapses below score resolution.
    Estimate: converged -> QF0(root of the monotone fit) (a final shift of <= ~eps rank using the
    measured precise points), clipped to the CI bracket; otherwise the measured threshold whose
    rank is closest to q among the precise runs.
    ``ci``: a genuine confidence interval for the quantile: (s_lo, s_hi) = the bracketing measured
    thresholds whose rank CI (a_G CI over the a_F CI) lies wholly below / above q (data-only score
    bounds if none). Coverage >= 1 - 3 alpha (two a_G CIs and the a_F CI, union bound; a_F CI
    ignored when cardinality_only). ``details['rank_band']`` is the old (QF0(t-eps), QF0(t+eps))
    band (NOT a confidence interval).
    Cost = N0 + a_F + sum of all AE runs (main loop + fallback); all charged."""
    rng = rng or np.random.default_rng(0)
    inst = inst or ExactInstance.build(u, cs, cardinality_only)
    cardinality_only = bool(np.all(inst.feasible))
    sf = inst._sf
    q_total, aF_q = 0, 0
    if cardinality_only:
        N0, aF, aF_lo, aF_hi = int(m0), 1.0, 1.0, 1.0
    else:
        N0 = int(m0 + rng.negative_binomial(m0, inst.a_F))
        if aF_run is not None:
            rF = aF_run
        else:
            if ae == "iqae":
                runs = _adaptive_iqae(make_oracle(inst.feasible, oracle_kind), rng, alpha,
                                      rel_eps=min(0.5, rel_F or eps), n_shots=n_shots)
                rF = runs[-1]
                aF_q = sum(r.oracle_queries for r in runs)
            else:      # MLAE: guess a_F from the classical warm start (m0 / N0)
                aF0 = m0 / N0
                rF = _ae_to_halfwidth(make_oracle(inst.feasible, oracle_kind),
                                      min(0.5, rel_F or eps) * aF0, aF0, rng, alpha, ae, n_shots, mlae_shots)
                aF_q = rF.oracle_queries
        aF = rF.estimate
        if aF <= 0:
            return QAEResult(float("nan"), (float("nan"),) * 2, N0 + aF_q,
                             dict(failed="a_F=0", converged=False))
        aF_lo, aF_hi = max(rF.ci[0], 1e-15), max(rF.ci[1], aF)
    q_total = N0 + aF_q
    x = np.sort(sf[rng.integers(len(sf), size=int(m0))])
    lo, hi = inst.score_range()
    pad = 1e-9 + 1e-6 * (hi - lo)
    s_min, s_max = min(lo - pad, x[0] - pad), max(hi + pad, x[-1] + pad)
    xk = np.concatenate([[s_min], x, [s_max]])
    pk = np.concatenate([[0.0], (np.arange(len(x)) + 0.5) / len(x), [1.0]])
    F0 = lambda s: float(np.interp(s, xk, pk))
    QF0 = lambda p: float(np.interp(min(max(p, 0.0), 1.0), pk, xk))
    steps, runs_t = [], []       # runs_t: (t, r, h, r_ci_lo, r_ci_hi)
    br = [0.0, 1.0]              # t-bracket from rank CIs excluding q

    def measure(t, h):
        nonlocal q_total
        s = QF0(t)
        eps_a = h * aF
        a_g = max(min(0.5, aF * (q + gap)), eps_a / 2)
        r = _ae_to_halfwidth(make_oracle(inst.mask_below(s), oracle_kind), eps_a, a_g, rng, alpha, ae,
                             n_shots, mlae_shots)
        q_total += r.oracle_queries
        rank = min(r.estimate / aF, 1.0)
        c_lo, c_hi = r.ci[0] / aF_hi, min(r.ci[1] / aF_lo, 1.0)
        if c_hi < q:
            br[0] = max(br[0], t)
        elif c_lo > q:
            br[1] = min(br[1], t)
        runs_t.append((t, rank, h, c_lo, c_hi))
        steps.append((s, rank, h, r.oracle_queries))
        return rank

    def root():
        tt = np.array([0.0, 1.0] + [v[0] for v in runs_t])
        rr = np.array([0.0, 1.0] + [v[1] for v in runs_t])
        ww = np.array([1e12, 1e12] + [1.0 / v[2] ** 2 for v in runs_t])
        o = np.argsort(tt, kind="stable")
        tt, fit = tt[o], _isotonic(tt[o], rr[o], ww[o])
        if fit[0] >= q:
            t = 0.0
        else:
            j = int(np.argmax(fit >= q))          # first fitted point at/above q (fit[-1] = 1)
            k = len(fit) - 1 - int(np.argmax(fit[::-1] < q))   # last fitted point below q
            if fit[j] == q:                       # flat at q: middle of the flat run
                jj = j
                while jj + 1 < len(fit) and fit[jj + 1] == q:
                    jj += 1
                t = 0.5 * (tt[j] + tt[jj])
            else:
                t = tt[k] + (q - fit[k]) * (tt[j] - tt[k]) / (fit[j] - fit[k])
        a_, b_ = br
        if a_ < b_:
            t = min(max(t, a_), b_)
            if t in (a_, b_) and runs_t:        # do not re-measure a bracket end
                t = 0.5 * (a_ + b_)
        return float(t)

    def is_conv(v):
        return v[2] <= eps * (1 + 1e-12) and abs(q - v[1]) <= eps

    t, gap, converged = q, 1.0 / math.sqrt(m0), False
    for _ in range(max_iter):
        h = max(eps, shrink * gap)
        rank = measure(t, h)
        gap = abs(q - rank) + 0.5 * h
        if is_conv(runs_t[-1]):
            converged = True
            break
        t = root()
    n_main = len(steps)
    n_fb = 0
    if not converged:
        cap = max_fallback if max_fallback is not None else 2 * int(math.ceil(math.log2(1.0 / eps))) + 4
        a_, b_ = br
        for _ in range(cap):
            if QF0(b_) - QF0(a_) <= 1e-12 or b_ - a_ <= 1e-12:
                break
            tm = 0.5 * (a_ + b_)
            gap = 0.0
            rank = measure(tm, eps)
            n_fb += 1
            if is_conv(runs_t[-1]):
                converged = True
                break
            if rank < q:
                a_ = tm
            else:
                b_ = tm
            br[0], br[1] = max(br[0], a_), min(br[1], b_)
    if converged:
        t_est = root()
    else:
        prec = [v for v in runs_t if v[2] <= eps * (1 + 1e-12)] or runs_t
        t_est = min(prec, key=lambda v: abs(v[1] - q))[0]
    est = QF0(t_est)
    ex = inst.quantile(q)
    rk = inst.percentile(est, "strict")
    last = runs_t[-1]
    ci = (QF0(br[0]) if br[0] > 0 else s_min, QF0(br[1]) if br[1] < 1 else s_max)
    return QAEResult(est, ci, q_total,
                     dict(exact=ex, rank_error=abs(rk - q), rank=rk, a_F_est=aF, steps=steps,
                          method="hybrid", cardinality_only=cardinality_only, classical_queries=N0,
                          aF_queries=aF_q, m0=int(m0), n_iter=len(steps), n_main=n_main,
                          n_fallback=n_fb, converged=bool(converged), last_h=float(last[2]),
                          last_rank_measured=float(last[1]),
                          ci_note="CI: bracketing thresholds whose AE rank CI excludes q; >= 1-3 alpha",
                          rank_band=(QF0(t_est - eps), QF0(t_est + eps))))


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
                  rng: np.random.Generator | None = None, oracle_kind: str = "auto",
                  median_method: str = "hybrid", m0: int = 10) -> QAEAttribution:
    """Percentile, null/benchmark median, constraint and manager effects by AE, with the exact
    values and the classical i.i.d. rejection estimates: one shared sample set (percentile + null
    median) of max(quantum percentile, quantum null-median) proposals, plus a cardinality-only
    set for the benchmark median. Total-return, equal-weight, buy-and-hold (rebalance=False) statistic only.

    ``median_method``: "hybrid" (default; ``qae_quantile_hybrid``: ``m0`` classical feasible samples
    charged as proposals + one rank-correction AE chain; ``eps_med`` is the rank CI half-width of the
    last AE; the null median reuses the percentile's a_F run, charged once) or "bisect" (old
    ``qae_quantile`` sequential-test bisection, ``n_bisect`` steps)."""
    rng = rng or np.random.default_rng(0)
    inst = ExactInstance.build(u, cs)
    inst_b = ExactInstance.build(u, cs, cardinality_only=True)
    s = _fund_score(fund, u, inst.g1)
    p = qae_percentile(u, cs, s, method, eps_pct, alpha, K, shots, rng=rng, inst=inst,
                       oracle_kind=oracle_kind)
    if median_method == "hybrid":
        ae = "iqae" if method == "iqae" else "mlae"
        aF_run = p.details.get("a_F") if np.isfinite(p.estimate) else None
        nm = qae_quantile_hybrid(u, cs, 0.5, eps_med, m0, alpha, rng=rng, inst=inst,
                                 oracle_kind=oracle_kind, ae=ae, aF_run=aF_run)
        bm = qae_quantile_hybrid(u, cs, 0.5, eps_med, m0, alpha, rng=rng, cardinality_only=True,
                                 inst=inst_b, oracle_kind=oracle_kind, ae=ae)
    elif median_method == "bisect":
        nm = qae_quantile(u, cs, 0.5, method, eps_med, alpha, n_bisect, K, shots, rng, False, inst,
                          oracle_kind)
        bm = qae_quantile(u, cs, 0.5, method, eps_med, alpha, n_bisect, K, shots, rng, True, inst_b,
                          oracle_kind)
    else:
        raise ValueError(median_method)
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
