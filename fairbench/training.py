"""Variational training of parameterised samplers.

``train(ansatz, objective_fn, ...)`` minimises an objective that is a callable on sampled
bitstrings (see ``fairbench.objectives``), so CVaR (optimizer pivot) and the uniformity
objective (benchmarking) are interchangeable.

Modes
-----
* sampled (default): each evaluation draws ``shots`` bitstrings via ``backends.sample``.
  This is what hardware would do. Cost is counted in shots.
* ``exact=True``: evaluates ``objective_fn.exact(probs)`` on ``Statevector`` probabilities.
  SIMULATION SHORTCUT (n <= ~16): noise-free, zero shots, not available on hardware. Its
  ``n_shots_total`` is 0; ``shots_equiv`` reports what the same number of evaluations would
  have cost at ``shots`` per evaluation.

Parameters are always bound by dict ``{Parameter: value}``; the internal vector order is
``list(ansatz.parameters)`` (Qiskit's name-sorted order: beta before gamma), but nothing
here relies on that order beyond being consistent within one call.

Optimizers: COBYLA (scipy; ``maxiter`` = max objective evaluations) and SPSA (own seeded
implementation, Spall 1998 gains with step calibration; ``maxiter`` = iterations, 2 evals each
plus 2*n_calib=10 calibration evals).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

from .backends import sample


@dataclass
class TrainResult:
    params: dict[Parameter, float]
    history: list[float]          # objective value at every evaluation, in order
    n_evals: int
    n_shots_total: int            # shots actually drawn (0 in exact mode)
    wall_time: float
    x0: dict[Parameter, float] = field(default_factory=dict)
    initial_value: float = float("nan")
    final_value: float = float("nan")   # objective at returned params (exact if exact mode)
    optimizer: str = ""
    exact: bool = False
    shots_equiv: int = 0          # n_evals * shots (hardware cost of same run)
    n_starts: int = 1


# --------------------------------------------------------------------------
# initial points
# --------------------------------------------------------------------------
def _is_gamma(p: Parameter) -> bool:
    return p.name.startswith("γ") or p.name.lower().startswith("gamma")


def _layer_index(p: Parameter) -> int:
    idx = getattr(p, "index", None)
    return int(idx) if idx is not None else 0


def default_x0(ansatz: QuantumCircuit, scale: float = 0.2) -> dict[Parameter, float]:
    """Small annealing-like ramp: gamma_j = scale*(j+.5)/p rising, beta_j = scale*(1-(j+.5)/p).

    Near the Dicke state (all angles small), so training starts from the baseline.
    Unknown parameters get scale/2.
    """
    params = list(ansatz.parameters)
    gam = [p for p in params if _is_gamma(p)]
    bet = [p for p in params if p.name.startswith("β") or p.name.lower().startswith("beta")]
    pdepth = max(len(gam), len(bet), 1)
    x0 = {}
    for p in params:
        j = _layer_index(p)
        f = (j + 0.5) / pdepth
        if p in gam:
            x0[p] = scale * f
        elif p in bet:
            x0[p] = scale * (1 - f)
        else:
            x0[p] = scale / 2
    return x0


def random_x0(ansatz: QuantumCircuit, rng: np.random.Generator,
              gamma_range=(0.0, 1.0), beta_range=(-0.6, 0.6)) -> dict[Parameter, float]:
    return {p: float(rng.uniform(*(gamma_range if _is_gamma(p) else beta_range)))
            for p in ansatz.parameters}


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def exact_probabilities(ansatz: QuantumCircuit, params: dict) -> np.ndarray:
    """Basis-state probabilities (index b: x_i = bit i of b). Simulation-only shortcut."""
    bound = ansatz.assign_parameters(params, inplace=False) if ansatz.parameters else ansatz
    return Statevector(bound).probabilities()


def _spsa(f: Callable[[np.ndarray], float], x0: np.ndarray, maxiter: int,
          rng: np.random.Generator, c: float = 0.1, target_step: float = 0.15,
          alpha: float = 0.602, gamma: float = 0.101, n_calib: int = 5) -> np.ndarray:
    """Plain SPSA with Rademacher perturbations and gain calibration (Spall 1998)."""
    x = x0.astype(float).copy()
    d = x.size
    A = 0.1 * maxiter
    # calibrate a so the first step has magnitude ~target_step
    mags = []
    for _ in range(n_calib):
        delta = rng.choice([-1.0, 1.0], size=d)
        mags.append(abs(f(x + c * delta) - f(x - c * delta)) / (2 * c))
    g0 = float(np.mean(mags)) or 1e-8
    a = target_step * (A + 1) ** alpha / g0
    for k in range(maxiter):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=d)
        yp, ym = f(x + ck * delta), f(x - ck * delta)
        x = x - ak * (yp - ym) / (2 * ck) * delta
    return x


def train(ansatz: QuantumCircuit, objective_fn, backend: str = "aer_statevector",
          optimizer: str = "COBYLA", maxiter: int = 100, shots: int = 1024,
          seed: int | None = None, x0: dict | None = None, exact: bool = False,
          rhobeg: float = 0.2, **backend_kwargs) -> TrainResult:
    """Minimise ``objective_fn`` over the ansatz parameters. See module doc.

    objective_fn: callable on (shots, n) uint8 samples; with ``exact=True`` it must expose
    ``.exact(probs)`` (an ``objectives.Objective``).
    seed drives SPSA perturbations and per-evaluation sampler seeds (fully reproducible).
    """
    params = list(ansatz.parameters)
    if not params:
        raise ValueError("ansatz has no parameters to train")
    if exact and not getattr(objective_fn, "has_exact", False):
        raise ValueError("exact=True needs an objective with an .exact(probs) variant")
    if x0 is None:
        x0 = default_x0(ansatz)
    missing = [p.name for p in params if p not in x0]
    if missing:
        raise ValueError(f"x0 missing parameters: {missing}")
    v0 = np.array([float(x0[p]) for p in params])

    rng = np.random.default_rng(seed)
    seed_stream = np.random.default_rng(rng.integers(2 ** 32))
    history: list[float] = []
    shots_used = [0]

    def to_dict(v):
        return {p: float(val) for p, val in zip(params, v)}

    def evaluate(v) -> float:
        d = to_dict(v)
        if exact:
            val = objective_fn.exact(exact_probabilities(ansatz, d))
        else:
            X = sample(ansatz, d, shots, backend=backend,
                       seed=int(seed_stream.integers(2 ** 31)), **backend_kwargs)
            shots_used[0] += shots
            val = float(objective_fn(X))
        history.append(val)
        return val

    t0 = time.perf_counter()
    initial_value = evaluate(v0)
    opt = optimizer.upper()
    if opt == "COBYLA":
        from scipy.optimize import minimize
        res = minimize(evaluate, v0, method="COBYLA",
                       options={"maxiter": max(int(maxiter) - 1, 1), "rhobeg": rhobeg})
        vbest = np.asarray(res.x, float)
    elif opt == "SPSA":
        vbest = _spsa(evaluate, v0, int(maxiter), rng)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}; use 'COBYLA' or 'SPSA'")
    final_value = evaluate(vbest)  # fresh evaluation at returned params (counted in cost)
    wall = time.perf_counter() - t0
    return TrainResult(params=to_dict(vbest), history=history, n_evals=len(history),
                       n_shots_total=shots_used[0], wall_time=wall, x0=to_dict(v0),
                       initial_value=initial_value, final_value=final_value,
                       optimizer=opt, exact=exact, shots_equiv=len(history) * int(shots))


def train_multistart(ansatz: QuantumCircuit, objective_fn, n_starts: int = 4,
                     seed: int | None = None, **kw) -> TrainResult:
    """Seeded multi-start: start 0 = ``default_x0``, others random (gamma in [0,1], beta in
    [-0.6,0.6]). Returns the start with the lowest ``final_value``; cost fields (n_evals,
    shots, wall_time) are summed over ALL starts. In sampled mode picking the min of noisy
    final values is optimistically biased - re-evaluate before reporting."""
    rng = np.random.default_rng(seed)
    results = []
    for s in range(n_starts):
        x0 = default_x0(ansatz) if s == 0 else random_x0(ansatz, rng)
        results.append(train(ansatz, objective_fn, x0=x0,
                             seed=int(rng.integers(2 ** 31)), **kw))
    best = min(results, key=lambda r: r.final_value)
    return TrainResult(params=best.params, history=best.history,
                       n_evals=sum(r.n_evals for r in results),
                       n_shots_total=sum(r.n_shots_total for r in results),
                       wall_time=sum(r.wall_time for r in results), x0=best.x0,
                       initial_value=best.initial_value, final_value=best.final_value,
                       optimizer=best.optimizer, exact=best.exact,
                       shots_equiv=sum(r.shots_equiv for r in results), n_starts=n_starts)


# --------------------------------------------------------------------------
# P1 experiment
# --------------------------------------------------------------------------
def p0_instance():
    """The P0 instance: synthetic_universe(12,3,seed=0), k=4, SectorCap 2/sector,
    MinESG at the 70th percentile of k-subset average ESG minus 1e-9 (as in p0_demo)."""
    from .constraints import Cardinality, ConstraintSet, MinESG, SectorCap
    from .data import synthetic_universe
    n, k = 12, 4
    u = synthetic_universe(n=n, n_sectors=3, seed=0)
    avgs = np.array([u.esg_score[list(c)].mean() for c in itertools.combinations(range(n), k)])
    m = float(np.quantile(avgs, 0.70)) - 1e-9
    cs = ConstraintSet([Cardinality(k)] + [SectorCap(s, 2) for s in sorted(set(u.sector))]
                       + [MinESG(m)])
    return u, cs


def p1_training_experiment(seed: int = 0, ps=(1, 2, 3), lams=(0.0, 0.05, 0.5),
                           optimizer: str = "COBYLA", maxiter: int = 300, n_starts: int = 3,
                           topology: str = "ring", verbose: bool = True) -> list[dict]:
    """Train the penalty-QAOA-style ansatz with the uniformity objective (exact mode) on the
    P0 instance and report P_F, TV(p|F, U_F), 1/P_F, and the honest
    "exact-uniform" cost 1/(P_F * M * min_x q_x) (rejection-correcting q to uniform), vs
    the Dicke p=0 baseline. Exact (statevector) numbers: no shot noise.
    """
    from .baselines import enumerate_feasible
    from .objectives import uniformity_objective
    from .quantum.ansatz import build_ansatz
    from .quantum.hamiltonian import penalty_operator

    u, cs = p0_instance()
    n, k = u.n, cs.cardinality
    F = enumerate_feasible(u, cs)
    op = penalty_operator(cs, u)
    rows = []

    def row(p, lam, stats, res=None):
        r = {"p": p, "lam": lam, "P_F": stats["P_F"], "TV": stats["tv"], "chi2": stats["chi2"],
             "inv_PF": 1 / stats["P_F"] if stats["P_F"] > 0 else float("inf"),
             "min_ratio": stats["min_ratio"],
             "cost_uniform": (1 / (stats["P_F"] * stats["min_ratio"])
                              if stats["P_F"] * stats["min_ratio"] > 0 else float("inf")),
             "evals": res.n_evals if res else 0, "wall": res.wall_time if res else 0.0}
        rows.append(r)
        if verbose:
            print(f"p={p} lam={lam:<5} P_F={r['P_F']:.3f} TV={r['TV']:.3f} chi2={r['chi2']:.3f} "
                  f"1/P_F={r['inv_PF']:.2f} M*minq={r['min_ratio']:.3f} "
                  f"cost_unif={r['cost_uniform']:.2f} evals={r['evals']} wall={r['wall']:.1f}s")

    base = uniformity_objective(cs, u, F, lam=0.0)
    row(0, float("nan"), base.report_exact(exact_probabilities(build_ansatz(n, k, 0, None), {})))
    for p in ps:
        qc = build_ansatz(n, k, p, cost_op=op, topology=topology)
        for lam in lams:
            obj = uniformity_objective(cs, u, F, lam=lam)
            res = train_multistart(qc, obj, n_starts=n_starts, seed=seed, optimizer=optimizer,
                                   maxiter=maxiter, exact=True)
            row(p, lam, obj.report_exact(exact_probabilities(qc, res.params)), res)
    return rows
