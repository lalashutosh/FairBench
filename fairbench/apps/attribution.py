"""Fund attribution: rank a fund against random portfolios that obey the same rules.

Draw N portfolios uniformly from the fund's feasible set (the *constrained null*),
compute the same performance statistic for each over the same window, and split the
fund's active return against an unconstrained benchmark into

    fund - benchmark = (null median - benchmark)   constraint effect: what the rules cost
                     + (fund - null median)        manager effect: selection within the rules

The benchmark defaults to the median random portfolio with the same number of holdings
and the same weighting but no other rule; pass ``benchmark_returns`` to use an index.
The fund's percentile in the constrained null says how many rule-abiding random picks
it beat in THIS window; one window is not by itself proof of skill. Holdings are held
fixed over the window (no turnover model).

The sampler is a parameter: any ``(u, cs, n_samples, seed) -> SampleResult`` whose
``samples`` are feasible (m, n) uint8 selections works. ``rejection_sampler`` is the
default; ``dicke_sampler()`` draws the same distribution from the Dicke circuit.
A sampler that is not uniform on the feasible set (trained layers, repair, a trapped
MCMC chain) biases the null and therefore the attribution.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..backends import sample
from ..baselines import SampleResult, _k, random_k_subsets
from ..constraints import ConstraintSet
from ..data import Universe
from ..postprocess import assign_weights, filter_feasible
from ..quantum.ansatz import build_ansatz

Sampler = Callable[[Universe, ConstraintSet, int, "int | None"], SampleResult]

METRICS = ("total_return", "sharpe")


# ----------------------------------------------------------------- samplers
def _sample_until(draw: Callable[[int, int], np.ndarray], u: Universe, cs: ConstraintSet,
                  n_samples: int, seed: int | None, unit: str, max_batch: int,
                  max_cost: int) -> SampleResult:
    """Call ``draw(m, seed) -> (m, n)`` raw proposals and filter until ``n_samples``
    feasible rows are collected. i.i.d. proposals, so truncating keeps the law."""
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    have = cost = 0
    batch = int(np.clip(2 * n_samples, 1024, max_batch))
    while have < n_samples:
        if cost >= max_cost:
            raise RuntimeError(f"only {have}/{n_samples} feasible samples after {cost} {unit} "
                               f"(acceptance {have / cost:.2e}); raise max_cost or use another sampler")
        raw = draw(batch, int(rng.integers(2**31)))
        S, _ = filter_feasible(raw, cs, u)
        kept.append(S)
        have += len(S)
        cost += len(raw)
        need = (n_samples - have) * cost / have if have else 2 * batch
        batch = int(np.clip(1.2 * need, 1024, max_batch))
    return SampleResult(samples=np.concatenate(kept)[:n_samples], cost=cost, cost_unit=unit,
                        info={"n_accepted": have})


def rejection_sampler(u: Universe, cs: ConstraintSet, n_samples: int, seed: int | None = None,
                      max_batch: int = 20_000, max_cost: int = 20_000_000) -> SampleResult:
    """Exactly ``n_samples`` uniform feasible selections by classical rejection
    (uniform random k-subsets, keep feasible). cost = proposals drawn."""
    n, k = u.n, _k(cs)
    return _sample_until(lambda m, s: random_k_subsets(n, k, m, s), u, cs, n_samples, seed,
                         "proposals", max_batch, max_cost)


def dicke_sampler(backend: str = "aer_statevector", circuit=None, params=None,
                  max_batch: int = 20_000, max_cost: int = 20_000_000, **backend_kwargs) -> Sampler:
    """Sampler factory for the quantum backend: measure the Dicke(n, k) state (uniform
    over k-subsets), keep feasible shots. Same distribution as ``rejection_sampler``.
    Pass ``circuit``/``params`` to use another ansatz; a trained ansatz is NOT uniform
    on the feasible set and biases the null. cost = shots."""
    def sampler(u: Universe, cs: ConstraintSet, n_samples: int, seed: int | None = None) -> SampleResult:
        qc = build_ansatz(u.n, _k(cs), 0, None) if circuit is None else circuit
        return _sample_until(lambda m, s: sample(qc, params, m, backend=backend, seed=s, **backend_kwargs),
                             u, cs, n_samples, seed, "shots", max_batch, max_cost)

    sampler.__name__ = f"dicke[{backend}]"
    return sampler


# ------------------------------------------------------------------ returns
def weights_matrix(X: np.ndarray, scheme: str = "equal", u: Universe | None = None) -> np.ndarray:
    """(m, n) selections -> (m, n) weights, one ``assign_weights`` call per row."""
    return np.stack([assign_weights(x, scheme, u) for x in np.atleast_2d(X)])


def portfolio_returns(W: np.ndarray, R: np.ndarray, rebalance: bool = True) -> np.ndarray:
    """(m, n) weights and (T, n) simple asset returns -> (m, T) portfolio period returns.
    rebalance=True resets to the target weights every period; False is buy-and-hold."""
    W, R = np.atleast_2d(W), np.asarray(R, dtype=float)
    if rebalance:
        return W @ R.T
    V = np.concatenate([np.ones((W.shape[0], 1)), W @ np.cumprod(1.0 + R, axis=0).T], axis=1)
    return V[:, 1:] / V[:, :-1] - 1.0


def performance(P: np.ndarray, metric: str = "total_return", periods_per_year: int = 252) -> np.ndarray:
    """(m, T) period returns -> (m,) statistic. "total_return": compounded over the
    window; "sharpe": annualised mean/std (zero risk-free rate)."""
    P = np.atleast_2d(P)
    if metric == "total_return":
        return np.prod(1.0 + P, axis=1) - 1.0
    if metric == "sharpe":
        sd = P.std(axis=1, ddof=1)
        return np.where(sd > 0, P.mean(axis=1) / np.where(sd > 0, sd, 1.0), 0.0) * np.sqrt(periods_per_year)
    raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")


def _stats(X: np.ndarray, R: np.ndarray, scheme: str, u: Universe, metric: str,
           rebalance: bool, periods_per_year: int, chunk: int = 2048) -> np.ndarray:
    """Statistic of every selection in X, chunked so the (m, T) block stays small."""
    out = [performance(portfolio_returns(weights_matrix(X[i:i + chunk], scheme, u), R, rebalance),
                       metric, periods_per_year) for i in range(0, len(X), chunk)]
    return np.concatenate(out) if out else np.zeros(0)


def _returns_array(u: Universe) -> np.ndarray:
    if u.returns is None:
        raise ValueError("universe has no historical returns (u.returns is None)")
    r = u.returns
    if isinstance(r, pd.DataFrame):
        cols = [str(c) for c in r.columns]
        if cols != list(u.tickers):
            if set(cols) != set(u.tickers):
                raise ValueError("u.returns columns do not match u.tickers")
            r = r.set_axis(cols, axis=1)[list(u.tickers)]
        r = r.to_numpy(dtype=float)
    r = np.asarray(r, dtype=float)
    if r.ndim != 2 or r.shape[1] != u.n or r.shape[0] < 2:
        raise ValueError(f"u.returns must be (T >= 2, n={u.n}); got {r.shape}")
    if not np.isfinite(r).all():
        raise ValueError("u.returns contains NaN/inf")
    return r


def _fund_weights(holdings: np.ndarray, scheme: str, u: Universe) -> tuple[np.ndarray, np.ndarray]:
    """Holdings as a 0/1 selection (weighted by ``scheme``) or as explicit weights
    (non-negative, summing to 1). Returns (selection uint8, weights)."""
    h = np.asarray(holdings, dtype=float).ravel()
    if h.size != u.n:
        raise ValueError(f"fund_holdings has length {h.size}, expected n={u.n}")
    if (h < 0).any() or not (h > 0).any():
        raise ValueError("fund_holdings must be non-negative and non-empty")
    x = (h > 0).astype(np.uint8)
    if np.isin(h, (0.0, 1.0)).all():
        return x, assign_weights(x, scheme, u)
    if abs(h.sum() - 1.0) > 1e-6:
        raise ValueError(f"fund weights sum to {h.sum():.6f}, expected 1")
    return x, h


def _median_ci(v: np.ndarray) -> tuple[float, float]:
    """Distribution-free 95% CI for the median from order statistics."""
    s = np.sort(v)
    half = 0.98 * np.sqrt(s.size)
    lo = max(int(np.floor(s.size / 2 - half)), 0)
    hi = min(int(np.ceil(s.size / 2 + half)), s.size - 1)
    return float(s[lo]), float(s[hi])


# ------------------------------------------------------------------- result
@dataclass
class AttributionResult:
    """All statistics are in units of ``metric``. ``null``: (N,) constrained random
    portfolios; ``unconstrained``: (N0,) cardinality-only random portfolios.
    ``percentile`` is the fund's mid-rank in ``null`` (0..100). The CIs and
    ``percentile_se`` cover Monte Carlo sampling error only, not market noise."""
    metric: str
    fund: float
    benchmark: float
    benchmark_label: str
    null: np.ndarray
    unconstrained: np.ndarray
    percentile: float
    percentile_se: float
    null_median: float
    null_median_ci: tuple[float, float]
    constraint_effect: float
    constraint_effect_ci: tuple[float, float]
    manager_effect: float
    manager_effect_ci: tuple[float, float]
    fund_feasible: bool
    cost: int
    cost_unit: str
    sampler: str
    weight_scheme: str

    @property
    def active(self) -> float:
        """fund - benchmark == constraint_effect + manager_effect."""
        return self.fund - self.benchmark

    @property
    def n_null(self) -> int:
        return int(self.null.size)

    def to_dict(self) -> dict:
        """Scalar fields only (JSON-friendly)."""
        d = {k: v for k, v in self.__dict__.items() if k not in ("null", "unconstrained")}
        d.update(active=self.active, n_null=self.n_null, n_unconstrained=int(self.unconstrained.size))
        return d

    def _fmt(self, v: float, signed: bool = True, digits: int = 2) -> str:
        s = "+" if signed else ""
        return f"{100 * v:{s}.{digits}f}%" if self.metric == "total_return" else f"{v:{s}.{digits + 1}f}"

    def summary(self) -> str:
        f, ci = self._fmt, lambda c: f"[{self._fmt(c[0])}, {self._fmt(c[1])}]"
        lines = [
            f"Attribution on {self.metric}: {self.n_null} rule-abiding random portfolios "
            f"(sampler {self.sampler}, {self.cost / self.n_null:.1f} {self.cost_unit} per sample, "
            f"{self.weight_scheme} weights)",
            f"  fund                         {f(self.fund, False):>9}",
            f"  benchmark                    {f(self.benchmark, False):>9}   {self.benchmark_label}",
            f"  constrained-null median      {f(self.null_median, False):>9}   95% CI {ci(self.null_median_ci)}",
            f"  active (fund - benchmark)    {f(self.active):>9}",
            f"    constraint effect          {f(self.constraint_effect):>9}   95% CI {ci(self.constraint_effect_ci)}"
            "   (null median - benchmark)",
            f"    manager effect             {f(self.manager_effect):>9}   95% CI {ci(self.manager_effect_ci)}"
            "   (fund - null median)",
            f"  fund percentile among rule-abiding portfolios: {self.percentile:.1f} "
            f"(+/- {self.percentile_se:.1f} Monte Carlo s.e.)",
        ]
        if not self.fund_feasible:
            lines.append("  WARNING: the fund's holdings violate the rule set; this null is not its null.")
        return "\n".join(lines)


# ---------------------------------------------------------------- attribute
def attribute(u: Universe, cs: ConstraintSet, fund_holdings: np.ndarray,
              fund_returns: np.ndarray | None = None, n_samples: int = 5000,
              sampler: Sampler = rejection_sampler, weight_scheme: str = "equal",
              metric: str = "total_return", benchmark_returns: np.ndarray | None = None,
              rebalance: bool = True, periods_per_year: int = 252,
              seed: int | None = 0) -> AttributionResult:
    """Rank a fund in the null distribution of portfolios obeying its rule set ``cs``.

    u.returns: (T, n) simple asset returns over the evaluation window.
    fund_holdings: (n,) 0/1 selection (weighted by ``weight_scheme``) or explicit weights.
    fund_returns: optional (T,) realised fund period returns; if omitted they are
        computed from the holdings (so fees and trading are ignored).
    benchmark_returns: optional (T,) period returns of an external benchmark; default
        benchmark is the median cardinality-only random portfolio.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")
    R = _returns_array(u)
    T = R.shape[0]
    ss = np.random.SeedSequence(seed)
    s_null, s_free = (int(s.generate_state(1)[0]) for s in ss.spawn(2))

    x_fund, w_fund = _fund_weights(fund_holdings, weight_scheme, u)
    fund_feasible = bool(cs.check(x_fund, u))
    if not fund_feasible:
        warnings.warn("fund holdings violate the constraint set; the null is not the fund's null",
                      stacklevel=2)
    if fund_returns is None:
        p_fund = portfolio_returns(w_fund, R, rebalance)
    else:
        p_fund = np.asarray(fund_returns, dtype=float).reshape(1, -1)
        if p_fund.shape[1] != T:
            raise ValueError(f"fund_returns has {p_fund.shape[1]} periods, u.returns has {T}")
    fund = float(performance(p_fund, metric, periods_per_year)[0])

    res = sampler(u, cs, n_samples, s_null)
    X = np.atleast_2d(res.samples)
    if X.shape[0] == 0:
        raise RuntimeError("sampler returned no feasible portfolios")
    if not cs.check_batch(X, u).all():
        raise ValueError("sampler returned infeasible portfolios; filter them before attribution")
    null = _stats(X, R, weight_scheme, u, metric, rebalance, periods_per_year)
    free = _stats(random_k_subsets(u.n, _k(cs), n_samples, s_free), R, weight_scheme, u,
                  metric, rebalance, periods_per_year)

    null_median = float(np.median(null))
    lo, hi = _median_ci(null)
    if benchmark_returns is None:
        benchmark = float(np.median(free))
        blo, bhi = _median_ci(free)
        label = f"median random portfolio, k={_k(cs)}, no other rule"
    else:
        b = np.asarray(benchmark_returns, dtype=float).reshape(1, -1)
        if b.shape[1] != T:
            raise ValueError(f"benchmark_returns has {b.shape[1]} periods, u.returns has {T}")
        benchmark = blo = bhi = float(performance(b, metric, periods_per_year)[0])
        label = "supplied benchmark"
    ce = null_median - benchmark
    # independent Monte Carlo errors of the two medians, combined in quadrature
    ce_ci = (ce - float(np.hypot(null_median - lo, bhi - benchmark)),
             ce + float(np.hypot(hi - null_median, benchmark - blo)))
    p = (np.sum(null < fund) + 0.5 * np.sum(null == fund)) / null.size
    return AttributionResult(
        metric=metric, fund=fund, benchmark=benchmark, benchmark_label=label,
        null=null, unconstrained=free, percentile=float(100 * p),
        percentile_se=float(100 * np.sqrt(p * (1 - p) / null.size)),
        null_median=null_median, null_median_ci=(lo, hi),
        constraint_effect=ce, constraint_effect_ci=ce_ci,
        manager_effect=fund - null_median, manager_effect_ci=(fund - hi, fund - lo),
        fund_feasible=fund_feasible, cost=int(res.cost), cost_unit=res.cost_unit,
        sampler=getattr(sampler, "__name__", type(sampler).__name__), weight_scheme=weight_scheme)


# --------------------------------------------------------------------- plot
_SURFACE, _INK, _INK2, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
_BLUE, _NEUTRAL = "#2a78d6", "#a9a8a2"


def plot_attribution(res: AttributionResult, path: str | Path | None = None,
                     title: str | None = None, note: str | None = None):
    """Two null distributions (unconstrained vs same-rules) with the fund marked and the
    constraint / manager effects drawn as spans. Returns the Figure; saves if ``path``."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    pct = res.metric == "total_return"
    fmt = res._fmt
    both = np.concatenate([res.null, res.unconstrained, [res.fund, res.benchmark]])
    bins = np.linspace(both.min(), both.max(), 61)
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    h0, *_ = ax.hist(res.unconstrained, bins=bins, density=True, histtype="stepfilled",
                     color=_NEUTRAL, alpha=0.45, lw=0)
    h1, *_ = ax.hist(res.null, bins=bins, density=True, histtype="stepfilled",
                     color=_BLUE, alpha=0.55, lw=0)
    top = max(h0.max(), h1.max())
    ax.axvline(res.benchmark, color=_INK2, lw=1.5, ls=(0, (4, 3)))
    ax.axvline(res.null_median, color=_BLUE, lw=1.5, ls=(0, (4, 3)))
    ax.axvline(res.fund, color=_INK, lw=2)

    marks = (res.benchmark, res.null_median, res.fund)

    def span(a: float, b: float, y: float, text: str):
        # label outside the span, on the side with no marker line to cross
        ax.annotate("", xy=(b, y), xytext=(a, y),
                    arrowprops=dict(arrowstyle="-|>", color=_INK, lw=1.4, shrinkA=0, shrinkB=0))
        left = min(a, b) <= min(marks)
        ax.annotate(text, xy=(min(a, b) if left else max(a, b), y), xytext=(-8 if left else 8, 0),
                    textcoords="offset points", ha="right" if left else "left", va="center",
                    fontsize=10, color=_INK).set_in_layout(False)

    span(res.benchmark, res.null_median, 1.08 * top, f"constraint effect {fmt(res.constraint_effect, digits=1)}")
    span(res.null_median, res.fund, 1.22 * top, f"manager effect {fmt(res.manager_effect, digits=1)}")
    ax.set_ylim(0, 1.32 * top)
    ax.set_xlim(bins[0] - 0.2 * np.ptp(bins), bins[-1] + 0.2 * np.ptp(bins))  # room for the labels
    ax.set_yticks([])
    ax.set_ylabel("share of random portfolios", color=_INK2)
    ax.set_xlabel("total return over the window" if pct else "annualised Sharpe ratio", color=_INK2)
    if pct:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.tick_params(colors=_INK2, length=0)
    ax.grid(axis="x", color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    handles = [
        Patch(color=_NEUTRAL, alpha=0.45, label=f"random portfolios, k only (n={res.unconstrained.size})"),
        Patch(color=_BLUE, alpha=0.55, label=f"random portfolios, same rules (n={res.n_null})"),
        Line2D([], [], color=_INK2, lw=1.5, ls=(0, (4, 3)), label=f"benchmark {fmt(res.benchmark, False, 1)}"),
        Line2D([], [], color=_BLUE, lw=1.5, ls=(0, (4, 3)), label=f"same-rules median {fmt(res.null_median, False, 1)}"),
        Line2D([], [], color=_INK, lw=2, label=f"fund {fmt(res.fund, False, 1)}"),
    ]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
                    frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(_INK)
    ax.set_title(title or f"Fund sits at percentile {res.percentile:.0f} of portfolios that follow its rules",
                 loc="left", fontsize=13, color=_INK, pad=12)
    if note:
        fig.text(0.012, 0.012, note, fontsize=8.5, color=_INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    if path is not None:
        fig.savefig(path, dpi=160, facecolor=_SURFACE)
    return fig
