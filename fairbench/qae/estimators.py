"""Amplitude-estimation (AE) algorithms with exact query accounting.

Model
-----
A state preparation A with A|0> = sqrt(1-a)|bad> + sqrt(a)|good>, a = sin^2(theta),
theta in [0, pi/2]. Grover iterate Q = A S_0 A^dagger S_chi. After m applications of Q,
measuring gives "good" with probability sin^2((2m+1) theta). For attribution statistics A is
the Dicke preparation over the C(n,k) weight-k portfolios and "good" is a Boolean predicate
(feasible, or feasible AND S(x) < S_fund); a percentile is the ratio of two amplitudes
(``ratio_estimate``).

Cost conventions (all estimators)
---------------------------------
* quantum query      = one application of Q = 1 oracle call (S_chi) + 2 state preps (A, A^dagger).
* a shot after m iterates costs m queries and 2m+1 state preps (the extra 1 is the initial A).
* ``AEResult.oracle_queries`` = sum over shots of m  (classical MC: number of samples).
* ``AEResult.state_preps``    = sum over shots of 2m+1 (classical MC: number of samples).
* classical query    = one i.i.d. sample from the m=0 distribution (one proposal + one
  predicate evaluation). ``classical_mc`` is literally m=0 sampling of the same oracle.
* canonical (QPE) AE: one shot uses M-1 controlled-Q applications (2^j for j < n_eval),
  i.e. M-1 queries and 2(M-1)+1 state preps; the QFT is not counted.

Nothing here builds circuits: oracles return Binomial hit counts from the exact outcome
probabilities (``IdealOracle``: closed form; ``SubspaceOracle``: exact state-vector Grover
in the weight-k subspace; ``NoisyOracle``: depolarising-decay model). Randomness only via
``np.random.Generator``.

References
----------
* Brassard, Hoyer, Mosca, Tapp, "Quantum amplitude amplification and estimation" (2002).
* Grinko, Gacon, Zoufal, Woerner, "Iterative quantum amplitude estimation", npj QI 7, 52 (2021).
* Suzuki et al., "Amplitude estimation without phase estimation", QIP 19, 75 (2020).
* Giurgica-Tiron et al., "Low depth algorithms for quantum amplitude estimation",
  Quantum 6, 745 (2022)  (power-law schedules).
* Brown et al., "Quantum amplitude estimation in the presence of noise" (2020);
  Tanaka et al., "Amplitude estimation via maximum likelihood on noisy quantum computer",
  QIP 20, 293 (2021)  (depolarising-decay likelihood used in ``NoisyOracle`` / ``mlae``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import optimize, stats

__all__ = [
    "AEResult", "IdealOracle", "NoisyOracle", "SubspaceOracle",
    "classical_mc", "canonical_ae", "iqae", "iqae_amplitude_tol", "mlae", "ratio_estimate",
    "exp_schedule", "linear_schedule", "power_schedule",
]

_PEPS = 1e-12


# ============================================================================ result
@dataclass
class AEResult:
    estimate: float                      # estimate of the amplitude a = sin^2(theta)
    ci: tuple                            # (lo, hi) confidence interval on a
    oracle_queries: int                  # sum over shots of m (classical: #samples)
    state_preps: int                     # sum over shots of 2m+1 (classical: #samples)
    shots: int                           # total number of shots / samples
    method: str
    schedule: list = field(default_factory=list)   # [(m, shots, hits), ...]
    extra: dict = field(default_factory=dict)      # method-specific (std, theta, ...)


def _cost(schedule):
    q = sum(int(m) * int(n) for m, n, _ in schedule)
    p = sum((2 * int(m) + 1) * int(n) for m, n, _ in schedule)
    s = sum(int(n) for _, n, _ in schedule)
    return q, p, s


def _z(conf: float) -> float:
    return float(stats.norm.ppf(0.5 + conf / 2))


# ============================================================================ oracles
class _OracleBase:
    """Protocol: ``prob_good(m)`` (vectorised over m), ``sample_hits(m, shots, rng)``,
    attributes ``a_true`` (may be None) and ``theta`` (may be None)."""
    a_true: float | None = None

    @property
    def theta(self):
        return None if self.a_true is None else math.asin(math.sqrt(self.a_true))

    def prob_good(self, m):
        raise NotImplementedError

    def sample_hits(self, m: int, shots: int, rng: np.random.Generator) -> int:
        p = float(np.clip(self.prob_good(int(m)), 0.0, 1.0))
        return int(rng.binomial(int(shots), p))


class IdealOracle(_OracleBase):
    """Noiseless: P(good | m) = sin^2((2m+1) theta)."""

    def __init__(self, a: float):
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"a must be in [0, 1], got {a}")
        self.a_true = float(a)

    def prob_good(self, m):
        return np.sin((2 * np.asarray(m) + 1) * self.theta) ** 2


class NoisyOracle(_OracleBase):
    """Depolarising-decay model (Brown et al. 2020; Tanaka et al. 2021 noisy MLAE):

        P(good | m) = f * sin^2((2m+1) theta) + (1 - f) * p_mixed,   f = exp(-gamma (2m+1)).

    The circuit after m iterates contains 2m+1 state preps; each multiplies the coherent
    part by exp(-gamma) and the remainder is replaced by the fully mixed state, whose
    "good" probability is ``p_mixed`` (e.g. 0.5 for a single flag qubit, or the fraction of
    good basis states of the register measured). Note m=0 is also (slightly) noisy."""

    def __init__(self, a: float, gamma: float, p_mixed: float = 0.5):
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"a must be in [0, 1], got {a}")
        self.a_true, self.gamma, self.p_mixed = float(a), float(gamma), float(p_mixed)

    def prob_good(self, m):
        k = 2 * np.asarray(m) + 1
        f = np.exp(-self.gamma * k)
        return f * np.sin(k * self.theta) ** 2 + (1 - f) * self.p_mixed


class SubspaceOracle(_OracleBase):
    """Exact Grover in the weight-k subspace: A = Dicke (uniform over the C(n,k) basis
    states), good = ``mask_good``. P(good | m) = sum_{x in good} |<x|G^m|D>|^2 computed with
    ``fairbench.ft.resources.grover_subspace_state`` (equals sin^2((2m+1) theta), tested)."""

    def __init__(self, mask_good):
        self.mask = np.asarray(mask_good, bool)
        if self.mask.ndim != 1 or self.mask.size == 0:
            raise ValueError("mask_good must be a non-empty 1-D Boolean array")
        self.a_true = float(self.mask.mean())
        self._cache: dict[int, float] = {}

    def _p(self, m: int) -> float:
        if m not in self._cache:
            from fairbench.ft.resources import grover_subspace_state
            psi = grover_subspace_state(self.mask, m)
            self._cache[m] = float(np.sum(np.abs(psi[self.mask]) ** 2))
        return self._cache[m]

    def prob_good(self, m):
        ms = np.asarray(m)
        if ms.ndim == 0:
            return self._p(int(ms))
        return np.array([self._p(int(x)) for x in ms.ravel()]).reshape(ms.shape)


# ============================================================================ classical MC
def _binom_ci(h: int, n: int, conf: float, method: str = "wilson"):
    if n == 0:
        return 0.0, 1.0
    if method == "wilson":
        z = _z(conf)
        p = h / n
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return max(0.0, c - half), min(1.0, c + half)
    if method in ("cp", "beta", "clopper-pearson"):
        al = 1 - conf
        lo = 0.0 if h == 0 else float(stats.beta.ppf(al / 2, h, n - h + 1))
        hi = 1.0 if h == n else float(stats.beta.ppf(1 - al / 2, h + 1, n - h))
        return lo, hi
    raise ValueError(method)


def classical_mc(oracle, n_queries: int, rng: np.random.Generator, conf: float = 0.95,
                 ci_method: str = "wilson") -> AEResult:
    """Classical Monte Carlo: n_queries i.i.d. m=0 samples. CI: Wilson (default) or
    Clopper-Pearson ("cp")."""
    n = int(n_queries)
    h = oracle.sample_hits(0, n, rng)
    p = h / n
    sched = [(0, n, h)]
    # classical query = one sample (proposal + predicate evaluation)
    return AEResult(p, _binom_ci(h, n, conf, ci_method), n, n, n, "classical_mc", sched,
                    dict(std=math.sqrt(max(p * (1 - p), 1.0 / n) / n), conf=conf))


# ============================================================================ canonical AE
def _qpe_distribution(theta: float, M: int) -> np.ndarray:
    """Exact QPE outcome distribution for A|0> = mixture (equal weight, orthogonal
    eigenvectors) of the Q eigenphases +-theta/pi (in units of 2 pi): Fejer kernels."""
    y = np.arange(M)
    out = np.zeros(M)
    for w in (theta / math.pi, -theta / math.pi):
        d = w - y / M
        s = np.sin(math.pi * d)
        num = np.sin(M * math.pi * d)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(np.abs(s) < 1e-14, 1.0, (num / (M * np.where(s == 0, 1, s))) ** 2)
        out += 0.5 * f
    return out / out.sum()


def canonical_ae(oracle, n_eval_qubits: int, shots: int, rng: np.random.Generator) -> AEResult:
    """Brassard-Hoyer-Mosca-Tapp QPE-based AE with M = 2^n_eval. Each shot samples y from
    the exact outcome distribution; a_y = sin^2(pi y / M). Estimate = most frequent a_y
    over shots (ties: smallest). CI: BHMT bound |a - a_est| <= 2 pi sqrt(a(1-a))/M + pi^2/M^2
    (per-shot success >= 8/pi^2; with a majority vote over many shots much higher),
    evaluated at a_est. For ``NoisyOracle`` each shot is with prob f = exp(-gamma (2(M-1)+1))
    the ideal outcome, else uniform y (same depolarising model)."""
    M = 2 ** int(n_eval_qubits)
    th = oracle.theta
    if th is None:
        raise ValueError("canonical_ae needs an oracle with known a_true (analytic simulation)")
    P = _qpe_distribution(th, M)
    gamma = getattr(oracle, "gamma", None)
    if gamma:
        f = math.exp(-gamma * (2 * (M - 1) + 1))
        P = f * P + (1 - f) / M
    ys = rng.choice(M, size=int(shots), p=P)
    a_y = np.round(np.sin(math.pi * ys / M) ** 2, 14)
    vals, counts = np.unique(a_y, return_counts=True)
    est = float(vals[np.argmax(counts)])
    half = 2 * math.pi * math.sqrt(est * (1 - est)) / M + math.pi ** 2 / M ** 2
    m = M - 1
    sched = [(m, int(shots), int(np.sum(a_y == est)))]
    q, sp, s = _cost(sched)
    return AEResult(est, (max(0.0, est - half), min(1.0, est + half)), q, sp, s,
                    "canonical_ae", sched, dict(M=M, y=ys, std=half / 2))


# ============================================================================ IQAE
def _find_next_k(k: int, up: bool, th_l: float, th_u: float, min_ratio: float = 2.0):
    """FindNextK (Grinko et al. 2021, Alg. 2), angles in units of 2 pi (theta/(2 pi) in
    [0, 1/4]); returns (k, upper_half_circle)."""
    K_old = 4 * k + 2
    K = int(1 / (2 * (th_u - th_l)))
    K = K - (K - 2) % 4                     # largest 4k+2 <= K_max
    while K >= min_ratio * K_old:
        lo = K * th_l - math.floor(K * th_l)
        hi = K * th_u - math.floor(K * th_u)
        if lo <= hi <= 0.5:
            return (K - 2) // 4, True
        if 0.5 <= lo <= hi:
            return (K - 2) // 4, False
        K -= 4
    return k, up


def iqae(oracle, eps: float, alpha: float, rng: np.random.Generator, n_shots: int = 100,
         ci_method: str = "beta", min_ratio: float = 2.0, max_rounds: int = 100_000,
         max_disjoint: int = 3) -> AEResult:
    """Iterative AE (Grinko, Gacon, Zoufal, Woerner, npj QI 2021, Algorithm 1).

    Maintains a confidence interval [theta_l, theta_u] (angle units 2 pi); each round picks
    the largest k (FindNextK) such that (4k+2)[theta_l, theta_u] lies in one half circle,
    measures N shots after k iterates, combines with the previous rounds if k is unchanged,
    turns the hit-rate CI (Chernoff-Hoeffding with alpha/T split, or Clopper-Pearson
    ("beta") at level alpha/T) into an angle CI and intersects. ``eps`` is an ANGLE
    tolerance: stops when the theta half-width is <= eps radians, which implies an
    a-interval half width <= sin(2 theta) eps <= eps (so it over-delivers for small a; use
    ``iqae_amplitude_tol`` for an amplitude target). T = ceil(log2(pi/(8 eps))).
    Assumes a noiseless oracle: under noise (e.g. NoisyOracle) it can report a confident
    wrong interval; use noise-aware ``mlae`` there.
    Chernoff mode also applies the paper's "no overshooting" shot reduction
    N = ceil(N_shots L_max / (eps K 10)) when K > ceil(L_max/eps),
    L_max = arcsin((2/N_shots ln(2T/alpha))^(1/4)). Coverage >= 1 - alpha."""
    if not 0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    if ci_method not in ("beta", "chernoff"):
        raise ValueError(ci_method)
    T = max(1, math.ceil(math.log2(math.pi / (8 * eps))))
    L_max = None
    if ci_method == "chernoff":
        v = (2.0 / n_shots * math.log(2 * T / alpha)) ** 0.25
        L_max = math.asin(min(1.0, v))
    k, up = 0, True
    th_l, th_u = 0.0, 0.25
    sched = []                                  # per round (k, N, hits)
    rk_hits = rk_shots = 0                      # accumulated for consecutive equal k
    it = 0
    n_bad = n_restarts = 0
    while th_u - th_l > eps / math.pi:
        it += 1
        if it > max_rounds:
            break
        k_new, up = _find_next_k(k, up, th_l, th_u, min_ratio)
        K = 4 * k_new + 2
        N = n_shots
        if L_max is not None and K > math.ceil(L_max / eps):
            N = max(1, math.ceil(n_shots * L_max / (eps * K * 10)))
        h = oracle.sample_hits(k_new, N, rng)
        sched.append((k_new, N, h))
        if k_new == k and it > 1:
            rk_hits += h
            rk_shots += N
        else:
            rk_hits, rk_shots = h, N
        k = k_new
        p = rk_hits / rk_shots
        if ci_method == "chernoff":
            e = math.sqrt(math.log(2 * T / alpha) / (2 * rk_shots))
            a_min, a_max = max(0.0, p - e), min(1.0, p + e)
        else:
            a_min, a_max = _binom_ci(rk_hits, rk_shots, 1 - alpha / T, "cp")
        if up:
            t_min = math.acos(1 - 2 * a_min) / (2 * math.pi)
            t_max = math.acos(1 - 2 * a_max) / (2 * math.pi)
        else:
            t_min = 1 - math.acos(1 - 2 * a_max) / (2 * math.pi)
            t_max = 1 - math.acos(1 - 2 * a_min) / (2 * math.pi)
        new_u = (math.floor(K * th_u) + t_max) / K
        new_l = (math.floor(K * th_l) + t_min) / K
        # never widen; if the new CI is disjoint from the current one (a miss of one of the
        # two intervals, prob <= alpha) do NOT intersect (that collapses to zero width and
        # stops with a confident wrong answer): keep the previous interval and continue.
        c_l, c_u = max(th_l, new_l), min(th_u, new_u)
        if c_l <= c_u:
            th_l, th_u = c_l, c_u
            n_bad = 0
        else:
            n_bad += 1
            if n_bad > max_disjoint:
                # the carried interval is persistently inconsistent with fresh data (it must
                # have missed the truth): discard it and restart from the full range
                # (all queries so far stay charged).
                th_l, th_u, k, up = 0.0, 0.25, 0, True
                rk_hits = rk_shots = 0
                n_bad = 0
                n_restarts += 1
    a_l = math.sin(2 * math.pi * th_l) ** 2
    a_u = math.sin(2 * math.pi * th_u) ** 2
    est = 0.5 * (a_l + a_u)
    q, sp, s = _cost(sched)
    return AEResult(est, (a_l, a_u), q, sp, s, f"iqae_{ci_method}", sched,
                    dict(std=(a_u - a_l) / (2 * _z(1 - alpha)), rounds=len(sched), T=T, restarts=n_restarts,
                         theta=math.pi * (th_l + th_u), conf=1 - alpha))


def iqae_amplitude_tol(oracle, eps_a: float, alpha: float, rng: np.random.Generator,
                       n_shots: int = 100, pilot: bool = True, pilot_eps: float = 0.05,
                       pilot_shots: int = 100, a_floor: float | None = None, **kw) -> AEResult:
    """IQAE targeting an ABSOLUTE amplitude half-width ``eps_a`` (never reads ``a_true``).

    Pilot (charged): coarse IQAE (theta half-width ``pilot_eps`` rad, level alpha/2) gives
    a_hat and a CI. Since da = sin(2 theta) dtheta = 2 sqrt(a(1-a)) dtheta, the main run uses
    eps_theta = eps_a / (2 sqrt(a_g (1 - a_g))), where a_g is the point of the pilot CI
    closest to 1/2 (so the choice is conservative w.r.t. pilot error), guarded to
    [a_floor, 1 - a_floor] (default a_floor = eps_a / 2). Main run: IQAE at level alpha/2
    (union bound => overall coverage >= 1 - alpha). All pilot queries / state preps / shots
    are added to the returned result and recorded in ``extra['pilot_queries']`` (plus
    ``pilot_state_preps``, ``pilot_shots``, ``pilot_a_hat``, ``eps_theta``).
    ``pilot=False``: a_g = 1/2 (worst case, eps_theta = eps_a/1), nothing charged, full alpha."""
    if eps_a <= 0:
        raise ValueError("eps_a must be > 0")
    pq = ps = psh = 0
    a_hat = None
    if pilot:
        pr = iqae(oracle, pilot_eps, alpha / 2, rng, n_shots=pilot_shots, **kw)
        pq, ps, psh = pr.oracle_queries, pr.state_preps, pr.shots
        a_hat = pr.estimate
        lo, hi = pr.ci
        a_g = 0.5 if lo <= 0.5 <= hi else (lo if lo > 0.5 else hi)
        alpha_main = alpha / 2
    else:
        a_g, alpha_main = 0.5, alpha
    floor = eps_a / 2 if a_floor is None else a_floor
    a_g = min(max(a_g, floor), 1 - floor)
    eps_th = min(0.45, eps_a / (2 * math.sqrt(a_g * (1 - a_g))))
    r = iqae(oracle, eps_th, alpha_main, rng, n_shots=n_shots, **kw)
    r.oracle_queries += pq
    r.state_preps += ps
    r.shots += psh
    r.extra.update(pilot_queries=pq, pilot_state_preps=ps, pilot_shots=psh,
                   pilot_a_hat=a_hat, eps_theta=eps_th, eps_a=eps_a, conf=1 - alpha)
    r.method = r.method + "_atol"
    return r


# ============================================================================ MLAE
def exp_schedule(K: int) -> list[int]:
    """m = 0, 1, 2, 4, ..., 2^(K-2)  (K entries; Suzuki et al. 2020)."""
    return [0] + [2 ** j for j in range(K - 1)]


def linear_schedule(K: int) -> list[int]:
    """m = 0, 1, ..., K-1."""
    return list(range(K))


def power_schedule(K: int, eps_exponent: float = 0.5) -> list[int]:
    """Power-law schedule of Giurgica-Tiron et al. (2022): m_k = floor(k^((1-beta)/(2beta))),
    k = 0..K-1, beta = ``eps_exponent`` in (0, 1]. Trades depth for queries: total queries
    ~ eps^-(1+beta), max depth ~ eps^-(1-beta) (beta -> 0: Heisenberg-like, beta = 1:
    classical depth-1). Error-vs-queries slope ~ -1/(1+beta)."""
    b = float(eps_exponent)
    if not 0 < b <= 1:
        raise ValueError("eps_exponent (beta) must be in (0, 1]")
    p = (1 - b) / (2 * b)
    return [0] + [int(math.floor(k ** p + 1e-9)) for k in range(1, K)]


def _model_p(theta, ms, gamma, p_mixed):
    """P(good | m, theta): rows = ms, cols = theta."""
    k = (2 * np.asarray(ms, float) + 1)[:, None]
    s = np.sin(k * np.asarray(theta)[None, :]) ** 2
    if gamma:
        f = np.exp(-gamma * k)
        s = f * s + (1 - f) * p_mixed
    return s


def _loglik(theta, ms, N, H, gamma, p_mixed):
    p = np.clip(_model_p(theta, ms, gamma, p_mixed), _PEPS, 1 - _PEPS)
    return (H[:, None] * np.log(p) + (N - H)[:, None] * np.log1p(-p)).sum(axis=0)


def _fisher(theta, ms, N, gamma, p_mixed):
    k = 2 * np.asarray(ms, float) + 1
    f = np.exp(-gamma * k) if gamma else np.ones_like(k)
    s = np.sin(k * theta) ** 2
    p = np.clip(f * s + (1 - f) * (p_mixed if gamma else 0.0), _PEPS, 1 - _PEPS)
    dp = f * k * np.sin(2 * k * theta)
    return float(np.sum(N * dp ** 2 / (p * (1 - p))))


def mlae(oracle, schedule: Sequence[int], shots, rng: np.random.Generator,
         gamma: float | None = None, p_mixed: float | None = None, conf: float = 0.95,
         ci: str = "lr", grid_factor: int = 20, chunk: int = 200_000) -> AEResult:
    """Maximum-likelihood AE (Suzuki et al. 2020). For each m in ``schedule`` take ``shots``
    (int or per-m sequence) shots after m iterates; MLE of theta in [0, pi/2] by a dense
    grid (spacing << period of the deepest term, grid_factor points per 1/(2m_max+1)) then
    bounded scalar refinement around the best grid point (avoids local optima).

    ``gamma`` given: noise-aware likelihood P = f sin^2 + (1-f) p_mixed, f = exp(-gamma(2m+1))
    (Tanaka et al. 2021); p_mixed defaults to 0.5. CI on a: ``ci="lr"`` (default) is the likelihood-ratio interval (2 dlogL <= chi2_1(conf)) profiled on the dense grid; ``extra['std']`` = CI half-width / z. ``ci="fisher"`` (theta_hat +-
    z/sqrt(I(theta_hat)) mapped through sin^2, clipped to [0, pi/2]; undercovers at small a)."""
    ms = np.asarray(list(schedule), int)
    if ms.ndim != 1 or ms.size == 0 or np.any(ms < 0):
        raise ValueError("schedule must be a non-empty list of m >= 0")
    N = np.broadcast_to(np.asarray(shots, int), ms.shape).astype(int)
    H = np.array([oracle.sample_hits(int(m), int(n), rng) for m, n in zip(ms, N)], int)
    if gamma and p_mixed is None:
        p_mixed = 0.5
    g = float(gamma) if gamma else 0.0
    pm = float(p_mixed) if gamma else 0.0

    # --- dense grid
    kmax = 2 * int(ms.max()) + 1
    n_grid = max(2001, grid_factor * kmax + 1)
    grid = np.linspace(0.0, math.pi / 2, n_grid)
    best_ll, best_t = -np.inf, 0.0
    ll_all = np.empty(n_grid)
    for i in range(0, n_grid, chunk):
        tg = grid[i:i + chunk]
        ll = _loglik(tg, ms, N, H, g, pm)
        ll_all[i:i + chunk] = ll
        j = int(np.argmax(ll))
        if ll[j] > best_ll:
            best_ll, best_t = float(ll[j]), float(tg[j])
    step = grid[1] - grid[0]
    lo_b, hi_b = max(0.0, best_t - step), min(math.pi / 2, best_t + step)
    res = optimize.minimize_scalar(lambda t: -_loglik(np.array([t]), ms, N, H, g, pm)[0],
                                   bounds=(lo_b, hi_b), method="bounded",
                                   options=dict(xatol=1e-12))
    th = float(res.x) if -res.fun >= best_ll else best_t
    ll_hat = max(best_ll, float(-res.fun))
    est = math.sin(th) ** 2

    I = _fisher(th, ms, N, g, pm)
    sd_th = 1.0 / math.sqrt(I) if I > 0 else math.pi / 2
    z = _z(conf)
    if ci == "fisher":
        t_lo, t_hi = max(0.0, th - z * sd_th), min(math.pi / 2, th + z * sd_th)
    elif ci == "lr":
        # likelihood-ratio interval profiled over the dense grid: hull of ALL grid points
        # with 2 (logL_hat - logL) <= chi2_1(conf) (the LR set is multimodal for deep
        # schedules; a contiguous window around the MLE undercovers when the MLE sits on a
        # wrong alias), edges extended by one grid step.
        thr = 0.5 * float(stats.chi2.ppf(conf, 1))
        idx = np.flatnonzero((ll_hat - ll_all) <= thr)
        if idx.size == 0:
            idx = np.array([int(np.argmin(np.abs(grid - th)))])
        t_lo = float(grid[max(int(idx[0]) - 1, 0)])
        t_hi = float(grid[min(int(idx[-1]) + 1, n_grid - 1)])
        t_lo, t_hi = min(t_lo, th), max(t_hi, th)
    else:
        raise ValueError(ci)
    sched = [(int(m), int(n), int(h)) for m, n, h in zip(ms, N, H)]
    q, sp, s = _cost(sched)
    name = "mlae" + ("_noise_aware" if gamma else "")
    a_lo, a_hi = math.sin(t_lo) ** 2, math.sin(t_hi) ** 2
    # std consistent with the reported CI (LR: half-width / z; Fisher: delta method)
    std = (a_hi - a_lo) / (2 * z) if ci == "lr" else abs(math.sin(2 * th)) * sd_th
    return AEResult(est, (a_lo, a_hi), q, sp, s, name, sched,
                    dict(theta=th, theta_std=sd_th, fisher=I, ci_method=ci,
                         std=std, loglik=ll_hat, conf=conf,
                         gamma=gamma, p_mixed=p_mixed if gamma else None))


# ============================================================================ ratio
def ratio_estimate(res_num: AEResult, res_den: AEResult, conf: float = 0.95,
                   clip_unit: bool = True) -> AEResult:
    """r = a_num / a_den (e.g. percentile |G|/|F| from two AE runs over the same Dicke prep,
    each relative to C(n,k)). Delta-method CI assuming independent runs:
    sd(r) = r sqrt((sd_num/a_num)^2 + (sd_den/a_den)^2), with sd taken from
    ``extra['std']`` (else CI half-width / z). Costs are summed. ``clip_unit`` clips the
    CI to [0, 1] (valid when good_num is a subset of good_den)."""
    z = _z(conf)

    def sd(r):
        if "std" in r.extra:
            return float(r.extra["std"])
        return (r.ci[1] - r.ci[0]) / (2 * z)

    an, ad = res_num.estimate, res_den.estimate
    if ad <= 0:
        raise ValueError("denominator estimate is 0; ratio undefined")
    r = an / ad
    sn, sdd = sd(res_num), sd(res_den)
    s = math.sqrt((sn / ad) ** 2 + (an * sdd / ad ** 2) ** 2)
    lo, hi = r - z * s, r + z * s
    lo = max(0.0, lo)
    if clip_unit:
        r_c, hi = min(r, 1.0), min(hi, 1.0)
    else:
        r_c = r
    return AEResult(r_c, (lo, hi), res_num.oracle_queries + res_den.oracle_queries,
                    res_num.state_preps + res_den.state_preps,
                    res_num.shots + res_den.shots,
                    f"ratio[{res_num.method}/{res_den.method}]",
                    list(res_num.schedule) + list(res_den.schedule),
                    dict(std=s, num=res_num, den=res_den, raw_ratio=r))
