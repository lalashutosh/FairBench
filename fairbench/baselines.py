"""Classical samplers / exact enumeration. All samples are (rows, n) uint8."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .constraints import ConstraintSet
from .data import Universe


@dataclass
class SampleResult:
    """samples: (m, n) uint8 feasible samples. cost: proposals (rejection) or
    chain moves (MCMC). cost_unit: 'proposals' | 'moves'. raw: all proposals
    (rejection only). info: extra counters (e.g. accepted moves)."""
    samples: np.ndarray
    cost: int
    cost_unit: str
    raw: np.ndarray | None = None
    info: dict = field(default_factory=dict)


def _k(cs: ConstraintSet) -> int:
    k = cs.cardinality
    if k is None:
        raise ValueError("ConstraintSet has no Cardinality constraint")
    return k


def enumerate_feasible(u: Universe, cs: ConstraintSet, chunk: int = 100_000) -> np.ndarray:
    """Exact feasible set via k-combinations (n <= 25). (m, n) uint8, lexicographic."""
    n, k = u.n, _k(cs)
    if n > 25:
        raise ValueError(f"enumerate_feasible: n={n} > 25 is too large")
    out: list[np.ndarray] = []
    it = itertools.combinations(range(n), k)
    while True:
        combos = list(itertools.islice(it, chunk))
        if not combos:
            break
        idx = np.asarray(combos, dtype=np.int64).reshape(len(combos), k)
        X = np.zeros((len(combos), n), dtype=np.uint8)
        np.put_along_axis(X, idx, 1, axis=1)
        out.append(X[cs.check_batch(X, u)])
    if not out:
        return np.zeros((0, n), dtype=np.uint8)
    return np.concatenate(out, axis=0)


def random_k_subsets(n: int, k: int, shots: int, seed: int | None = None) -> np.ndarray:
    """Uniform random weight-k bitstrings (classical Dicke analogue). (shots, n) uint8."""
    rng = np.random.default_rng(seed)
    ranks = rng.random((shots, n)).argsort(axis=1).argsort(axis=1)
    return (ranks < k).astype(np.uint8)


def rejection_sample(u: Universe, cs: ConstraintSet, shots: int, seed: int | None = None,
                     max_draws: int | None = None) -> SampleResult:
    """Draw `shots` random k-subset proposals; keep feasible ones.
    samples = feasible proposals (<= shots), raw = all proposals, cost = shots.
    max_draws (if given) caps the proposals drawn (cost = min(shots, max_draws))."""
    n, k = u.n, _k(cs)
    m = shots if max_draws is None else min(shots, max_draws)
    raw = random_k_subsets(n, k, m, seed)
    ok = cs.check_batch(raw, u)
    return SampleResult(samples=raw[ok], cost=m, cost_unit="proposals", raw=raw,
                        info={"n_accepted": int(ok.sum())})


def mcmc_swap_sample(u: Universe, cs: ConstraintSet, shots: int, burn_in: int, thin: int,
                     seed: int | None = None, x0: np.ndarray | None = None,
                     max_init_tries: int = 100_000) -> SampleResult:
    """Metropolis swap chain: swap a random held i with a random non-held j,
    accept iff feasible (symmetric proposal -> uniform stationary law on the
    feasible set). Returns `shots` states after burn_in, every `thin` moves.
    cost = burn_in + shots*thin moves; info['accepted_moves'] = accepted swaps."""
    if shots < 0 or thin < 1 or burn_in < 0:
        raise ValueError(f"need shots>=0, thin>=1, burn_in>=0; got shots={shots}, "
                         f"thin={thin}, burn_in={burn_in}")
    n, k = u.n, _k(cs)
    rng = np.random.default_rng(seed)
    if x0 is None:
        cand = random_k_subsets(n, k, max_init_tries, int(rng.integers(2**31)))
        ok = np.flatnonzero(cs.check_batch(cand, u))
        if ok.size == 0:
            raise RuntimeError(f"no feasible start found in {max_init_tries} tries")
        x = cand[ok[0]].copy()
    else:
        x = np.asarray(x0, dtype=np.uint8).copy()
        if not cs.check(x, u):
            raise ValueError("x0 is not feasible")
    total = burn_in + shots * thin
    out = np.empty((shots, n), dtype=np.uint8)
    accepted = 0
    held = list(np.flatnonzero(x == 1))
    free = list(np.flatnonzero(x == 0))
    s = 0
    if 0 < k < n:
        for t in range(1, total + 1):
            a = int(rng.integers(len(held)))
            b = int(rng.integers(len(free)))
            i, j = held[a], free[b]
            x[i], x[j] = 0, 1
            if cs.check(x, u):
                held[a], free[b] = j, i
                accepted += 1
            else:
                x[i], x[j] = 1, 0
            if t > burn_in and (t - burn_in) % thin == 0:
                out[s] = x
                s += 1
    else:
        out[:] = x
        s = shots
    return SampleResult(samples=out, cost=total, cost_unit="moves",
                        info={"accepted_moves": accepted})
