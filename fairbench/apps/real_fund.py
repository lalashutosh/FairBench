"""A real fund against the portfolios its rules allowed, one reporting period at a time.

Inputs are public N-PORT snapshots: the fund's and a parent index fund's, at two consecutive
report dates t and t1. From them:

    universe        the parent fund's long common-stock holdings at t, with their weights as
                    the benchmark weights (a PROXY for the index, and named as such)
    returns         each security's return over (t, t1] from a ``ReturnSource``; by default
                    the price return implied by the parent's two filings (no dividends)
    realised        the fund's disclosed portfolio at t, held without trading to t1: the
                    FROZEN-HOLDINGS return. It is what gets ranked.
    reference       portfolios drawn from the universe under the same rules, held the same
                    way: equal-weight subsets, benchmark-weighted subsets, the weight grid,
                    and the benchmark-aware tilt (see ``portfolio.reference``)

Only data dated t or earlier defines the universe, the rules and the weights; returns come
after t. The fund's reported NAV return over the same months is shown beside the frozen
return and never ranked against the reference portfolios: the difference between the two
contains trading inside the period, fees, cash, flows and (with a price-only source)
dividends, none of which the reference portfolios have.

What the output is and is not. It is a descriptive rule-versus-choice comparison: where the
disclosed portfolio's return fell among portfolios that were feasible under the stated
rules, under an explicit reference distribution. It is not a causal decomposition and a
percentile is not evidence of skill. Labels used: rule-conditioned return range,
rule-conditioned shift, within-mandate return difference, realised portfolio percentile.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..constraints import (Cardinality, ConstraintSet, Exclusion, WeightRuleSet, active_risk)
from ..data import Universe
from ..ingest.identifiers import add_security_keys
from ..ingest.prices import NportImpliedReturns, ReturnSource, nav_period_return, net_flow_ratio
from ..portfolio.reference import (ReferenceSample, benchmark_aware, reference_statistics, uniform_subsets,
                                   uniform_weights)
from ..portfolio.weights import WeightGrid

LABELS_VERSION = "2026-10-09"
CAVEATS = (
    "Descriptive comparison, not a causal decomposition; a percentile is not evidence of skill.",
    "Holdings are a snapshot at the start date, held without trading to the end date.",
    "The filing became public about 60 days after the portfolio date.",
    "The universe and benchmark weights are a proxy taken from an index fund's own filing.",
    "Intervals and standard errors cover sampling error only.",
)


def keyed_sleeve(report) -> pd.DataFrame:
    """A report's long common-stock positions with security keys, one row per security
    (rows sharing a key are summed). Columns: security_key, issuer_name, balance, value_usd,
    weight. Positions with no usable identifier are left out and counted in ``.attrs``."""
    s = add_security_keys(report.equity_sleeve())
    known = ~s["key_scheme"].isin(["unknown"])
    g = s[known].groupby("security_key", sort=False).agg(
        issuer_name=("issuer_name", "first"), balance=("balance", "sum"), value_usd=("value_usd", "sum")).reset_index()
    g["weight"] = g["value_usd"] / g["value_usd"].sum()
    g.attrs = {"n_rows": int(len(s)), "n_unkeyed": int((~known).sum()),
               "unkeyed_value_share": float(s.loc[~known, "value_usd"].sum() / s["value_usd"].sum()) if len(s) else 0.0,
               "equity_share_of_net_assets": float(s["value_usd"].sum() / report.net_assets)
               if report.net_assets and report.net_assets > 0 else float("nan")}
    return g


@dataclass
class PeriodCase:
    """Everything one period's comparison needs. Arrays are aligned to ``keys``."""
    start: date
    end: date
    fund_id: str
    fund_name: str
    parent_id: str
    parent_name: str
    universe: Universe
    keys: list[str]
    benchmark: np.ndarray
    asset_returns: np.ndarray
    fund_weights: np.ndarray           # the fund's weights on universe names, renormalised to 1
    k: int
    fund_frozen_return: float          # all of the fund's sleeve with a known return, renormalised
    fund_in_universe_return: float
    benchmark_proxy_return: float
    fund_nav: dict
    coverage: dict
    return_kind: str
    return_source: str


def build_period_case(parent_start, parent_end, fund_start, fund_end=None, *, source: ReturnSource | None = None,
                      unknown_returns: str = "drop", sectors: Mapping[str, str] | None = None,
                      class_id: str | None = None) -> PeriodCase:
    """Assemble one period from N-PORT reports (``ingest.nport.NportReport``).

    unknown_returns: what to do with universe securities whose return over the period is
    not known (typically names that left the index fund before the end date):
        "drop"            leave them out of the universe (default). No number is made up,
                          but the choice uses knowledge from after t, so the count and the
                          benchmark weight affected are reported.
        "benchmark_fill"  give them the benchmark-weighted return of the known names; the
                          filled values are an assumption and are reported as such.
    sectors: optional {security_key: label}; a derived classification (for example from SIC
    codes). Without it every sector is "unknown" and sector rules cannot be applied.
    """
    if unknown_returns not in ("drop", "benchmark_fill"):
        raise ValueError('unknown_returns must be "drop" or "benchmark_fill"')
    start, end = parent_start.report_date, parent_end.report_date
    if not (fund_start.report_date == start and start < end):
        raise ValueError(f"report dates do not line up: parent {start} -> {end}, fund {fund_start.report_date}")
    ps, pe, fs = keyed_sleeve(parent_start), keyed_sleeve(parent_end), keyed_sleeve(fund_start)
    if source is None:
        source = NportImpliedReturns({start: ps.copy(), end: pe.copy()})
        if fund_end is not None and fund_end.report_date == end:  # prices for names only the fund holds
            source.add({start: fs.copy(), end: keyed_sleeve(fund_end)})
    all_keys = list(dict.fromkeys(ps["security_key"].tolist() + fs["security_key"].tolist()))
    rets = source.period_returns(all_keys, start, end)

    uni = ps.set_index("security_key")
    r_uni = rets.loc[uni.index, "ret"].to_numpy(dtype=float)
    unknown = np.isnan(r_uni)
    cov_note = {"n_universe_before": int(len(uni)), "n_unknown_return": int(unknown.sum()),
                "benchmark_weight_unknown_return": float(uni["weight"].to_numpy()[unknown].sum()),
                "unknown_returns_policy": unknown_returns,
                "n_split_adjusted": int(rets.loc[uni.index, "split_adjusted"].sum())}
    if unknown_returns == "drop":
        uni, r_uni = uni[~unknown], r_uni[~unknown]
    elif unknown.any():
        w = uni["weight"].to_numpy()
        r_uni = np.where(unknown, float(w[~unknown] @ r_uni[~unknown] / w[~unknown].sum()), r_uni)
    keys = uni.index.tolist()
    b = uni["weight"].to_numpy(dtype=float)
    b = b / b.sum()
    n = len(keys)

    f = fs.set_index("security_key")
    in_uni = f.index.isin(keys)
    w_in = f.loc[in_uni, "weight"]
    fund_w = np.zeros(n)
    pos = {k: i for i, k in enumerate(keys)}
    fund_w[[pos[k] for k in w_in.index]] = w_in.to_numpy()
    in_weight = float(fund_w.sum())
    if in_weight <= 0:
        raise ValueError("the fund holds nothing in the parent universe")
    fund_w /= in_weight
    r_fund = np.array(rets.loc[f.index, "ret"], dtype=float)  # a writable copy
    r_fund[in_uni] = r_uni[[pos[k] for k in f.index[in_uni]]]      # same numbers the references use
    known_f = ~np.isnan(r_fund)
    wf = f["weight"].to_numpy(dtype=float)
    frozen = float(wf[known_f] @ r_fund[known_f] / wf[known_f].sum())

    sector = [(sectors or {}).get(k, "unknown") for k in keys]
    u = Universe(tickers=keys, mu=np.full(n, np.nan), cov=np.full((n, n), np.nan), sector=sector,
                 esg_score=np.full(n, np.nan), carbon=np.full(n, np.nan))
    kinds = rets.loc[uni.index, "kind"].dropna().unique().tolist()
    coverage = {
        **cov_note, "n_universe": n, "n_fund_holdings": int(len(f)), "n_fund_in_universe": int(in_uni.sum()),
        "fund_weight_in_universe": in_weight, "fund_weight_off_universe": 1.0 - in_weight,
        "fund_weight_unknown_return": float(wf[~known_f].sum()),
        "fund_unkeyed_positions": fs.attrs["n_unkeyed"], "parent_unkeyed_positions": ps.attrs["n_unkeyed"],
        "fund_equity_share_of_net_assets": fs.attrs["equity_share_of_net_assets"],
        "parent_equity_share_of_net_assets": ps.attrs["equity_share_of_net_assets"],
        "sectors_known": sectors is not None,
        "max_price_disagreement": float(getattr(source, "max_price_disagreement", float("nan"))),
    }
    nav = nav_period_return(fund_end, class_id) if fund_end is not None and fund_end.report_date == end else \
        dict(class_id=class_id, nav_return=float("nan"), status="unknown", note="no fund report at the end date")
    if fund_end is not None:
        nav["net_flow_ratio"] = net_flow_ratio(fund_end)
    return PeriodCase(
        start=start, end=end, fund_id=fund_start.series_id or "", fund_name=fund_start.series_name or "",
        parent_id=parent_start.series_id or "", parent_name=parent_start.series_name or "", universe=u, keys=keys,
        benchmark=b, asset_returns=r_uni, fund_weights=fund_w, k=int((fund_w > 0).sum()),
        fund_frozen_return=frozen, fund_in_universe_return=float(fund_w @ r_uni),
        benchmark_proxy_return=float(b @ r_uni), fund_nav=nav, coverage=coverage,
        return_kind=kinds[0] if len(kinds) == 1 else "mixed", return_source=getattr(source, "name", "unknown"))


@dataclass
class PeriodResult:
    case: PeriodCase
    rows: list[dict]                    # one per reference distribution
    realised: dict
    rules: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        c = self.case
        return dict(fund_id=c.fund_id, fund_name=c.fund_name, parent_id=c.parent_id, parent_name=c.parent_name,
                    period_start=str(c.start), period_end=str(c.end), return_kind=c.return_kind,
                    return_source=c.return_source, k=c.k, coverage=c.coverage, realised=self.realised,
                    rules=self.rules, reference=self.rows, notes=self.notes, labels_version=LABELS_VERSION,
                    caveats=list(CAVEATS))

    def summary(self) -> str:
        c, r = self.case, self.realised
        pct = lambda v: "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:+.2f}%"
        lines = [
            f"{c.fund_name} ({c.fund_id})  {c.start} -> {c.end}   [{c.return_kind} returns from {c.return_source}]",
            f"  universe: {c.coverage['n_universe']} securities from {c.parent_name} (benchmark proxy), "
            f"{c.coverage['n_unknown_return']} with unknown return ({c.coverage['unknown_returns_policy']})",
            f"  fund: {c.coverage['n_fund_holdings']} holdings, {c.coverage['n_fund_in_universe']} in the universe "
            f"({100 * c.coverage['fund_weight_in_universe']:.1f}% of its equity weight)",
            f"  realised frozen-holdings return {pct(r['frozen_return'])}   benchmark proxy {pct(r['benchmark_proxy_return'])}"
            f"   reported NAV return {pct(r['nav_return'])}   NAV minus frozen {pct(r['nav_minus_frozen'])}",
            f"  rules applied: " + ("; ".join(self.rules) or "none beyond the number of holdings"),
            "  reference distribution                         median     5%      95%   percentile   ESS   log10(#feasible)",
        ]
        for row in self.rows:
            n = row.get("log10_n_feasible")
            lines.append(f"  {row['label']:<44} {pct(row['median']):>8} {pct(row['p05']):>8} {pct(row['p95']):>8} "
                         f"{row['percentile']:>7.1f} ±{row['percentile_se']:<4.1f} {row['effective_sample_size']:>6.0f}   "
                         + ("n/a" if n is None else f"{n:.1f}"))
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _row(label: str, ref: ReferenceSample, case: PeriodCase) -> dict:
    stats = reference_statistics(ref, case.asset_returns, case.fund_frozen_return)
    return {"label": label, **stats, **ref.manifest()}


def run_period(case: PeriodCase, *, excluded_keys: Sequence[str] = (), selection_rules: ConstraintSet | None = None,
               weight_rules: WeightRuleSet | None = None, rule_notes: Sequence[str] = (), n_samples: int = 5000,
               seed: int | None = 0, grid_step: float = 0.0025, cov: np.ndarray | None = None,
               tau_multiples: Sequence[float] = (0.25, 0.5, 1.0), cap: float | None = None) -> PeriodResult:
    """Rank the fund's frozen-holdings return in each reference distribution.

    The number of holdings is the fund's own count in the universe. ``excluded_keys`` and
    ``selection_rules`` restrict which names may be held; ``weight_rules`` apply to actual
    weights (weight grid and capped-benchmark references). ``cap`` is the largest position
    weight for the benchmark-weighted and grid references (default: the fund's own largest
    weight in the universe, a value DERIVED from the fund's holdings and reported).
    ``cov``: (n, n) covariance aligned to ``case.keys`` for the benchmark-aware reference;
    without it that reference is skipped and the result says so.
    """
    u, b, k, n = case.universe, case.benchmark, case.k, case.universe.n
    pos = {key: i for i, key in enumerate(case.keys)}
    excl = sorted({pos[x] for x in excluded_keys if x in pos})
    held_excluded = [case.keys[i] for i in excl if case.fund_weights[i] > 0]
    cons = [Cardinality(k)] + ([Exclusion(excl)] if excl else [])
    if selection_rules is not None:
        cons += [c for c in selection_rules.constraints if not isinstance(c, Cardinality)]
    cs = ConstraintSet(cons)
    rules = [f"{k} holdings"] + ([f"{len(excl)} securities excluded"] if excl else []) + list(rule_notes)
    notes: list[str] = []
    if held_excluded:
        notes.append(f"the fund itself holds {len(held_excluded)} excluded securities; the reference set "
                     "does not contain its portfolio")
    if case.coverage["fund_weight_off_universe"] > 0.02:
        notes.append(f"{100 * case.coverage['fund_weight_off_universe']:.1f}% of the fund's equity weight is outside "
                     "the parent universe; no reference portfolio can hold those names")
    cap_used = float(max(case.fund_weights.max(), 1.0 / k)) if cap is None else float(cap)
    cap_note = (f"position cap {100 * cap_used:.2f}% " +
                ("(the fund's own largest weight; derived)" if cap is None else "(supplied)"))
    seeds = np.random.SeedSequence(seed).generate_state(4)
    batch = int(min(20_000, max(2 * n_samples, 2_000)))  # (batch, n) arrays: keep memory modest at n ~ 500

    rows = [_row("D1 uniform subsets, equal weights",
                 uniform_subsets(u, cs, n_samples, policy="equal", seed=int(seeds[0]), batch=batch), case)]
    d1b = uniform_subsets(u, cs, n_samples, policy="benchmark_capped", benchmark=b, cap=cap_used,
                          weight_rules=weight_rules, seed=int(seeds[1]), batch=batch)
    rows.append(_row("D1 uniform subsets, capped benchmark weights", d1b, case))

    units = int(round(1.0 / grid_step))
    w_pos = case.fund_weights[case.fund_weights > 0]
    lo_u = max(1, min(int(math.floor(w_pos.min() * units + 1e-9)), units // k))
    hi_u = max(int(math.ceil(cap_used * units - 1e-9)), -(-units // k))
    allowed = None if not excl else tuple(i for i in range(n) if i not in set(excl))
    grid = WeightGrid(n=n, units=units, min_units=lo_u, max_units=hi_u, k_min=k, k_max=k, allowed=allowed)
    sel = ConstraintSet(cs.non_cardinality())
    grid_rules = WeightRuleSet([] if weight_rules is None else weight_rules.rules,
                               support_rules=sel if sel.constraints else None)
    d2 = uniform_weights(u, grid, grid_rules, n_samples, seed=int(seeds[2]), batch=batch)
    rows.append(_row(f"D2 uniform weight grid, step {100 * grid_step:g}%", d2, case))
    notes.append(f"{cap_note}; weight grid {lo_u}/{units} to {hi_u}/{units} per holding "
                 "(smallest position from the fund's own smallest weight; derived)")

    realised = dict(frozen_return=case.fund_frozen_return, in_universe_return=case.fund_in_universe_return,
                    benchmark_proxy_return=case.benchmark_proxy_return, nav_return=case.fund_nav["nav_return"],
                    nav_class_id=case.fund_nav.get("class_id"), nav_status=case.fund_nav["status"],
                    nav_minus_frozen=case.fund_nav["nav_return"] - case.fund_frozen_return,
                    net_flow_ratio=case.fund_nav.get("net_flow_ratio"))
    if cov is None:
        notes.append("benchmark-aware reference (D3) skipped: no covariance estimate was supplied")
    else:
        cov = np.asarray(cov, dtype=float)
        te = float(active_risk(case.fund_weights, cov, b)[0])
        realised["fund_tracking_error_same_units_as_cov"] = te
        for mult in tau_multiples:
            d3 = benchmark_aware(d1b, u, b, mult * te, cov=cov)
            rows.append(_row(f"D3 benchmark-aware, tau = {mult:g} x fund TE", d3, case))
            if d3.ess < 50:
                notes.append(f"D3 with tau = {mult:g} x fund TE rests on {d3.ess:.0f} effective portfolios; unreliable")
    if any(r["n_violations"] for r in rows):
        raise RuntimeError("a reference sample contains portfolios that break the rules")
    return PeriodResult(case=case, rows=rows, realised=realised, rules=rules, notes=notes)


def single_index_cov(history: pd.DataFrame, keys: Sequence[str], benchmark: np.ndarray,
                     min_periods: int = 4, shrink: float = 0.5) -> tuple[np.ndarray, dict] | None:
    """A covariance for the benchmark-aware reference from past period returns only.

    history: rows = past periods (all ending on or before the start date), columns =
    security keys, NaN where unknown. Model: r_i = beta_i * m + e_i with m the
    benchmark-weighted return of the securities known in that period. beta is estimated per
    security (needs 3 or more known periods) and shrunk toward 1 by ``shrink``; a security
    with too little history gets beta 1 and the median residual variance, and is counted.
    Returns (cov in per-period units, info) or None if there are fewer than ``min_periods``.
    This is a DERIVED, deliberately simple risk model; every result that uses it says so.
    """
    R = history.reindex(columns=list(keys)).to_numpy(dtype=float)
    if R.shape[0] < min_periods:
        return None
    b = np.asarray(benchmark, dtype=float)
    known = ~np.isnan(R)
    wsum = (known * b[None, :]).sum(axis=1)
    m = np.where(wsum > 0, (np.where(known, R, 0.0) * b[None, :]).sum(axis=1) / np.where(wsum > 0, wsum, 1.0), np.nan)
    ok_t = ~np.isnan(m)
    var_m = float(np.var(m[ok_t], ddof=1))
    n = R.shape[1]
    beta, resid, thin = np.ones(n), np.full(n, np.nan), 0
    for i in range(n):
        t = known[:, i] & ok_t
        if t.sum() < 3 or var_m <= 0:
            thin += 1
            continue
        raw = float(np.cov(R[t, i], m[t], ddof=1)[0, 1] / np.var(m[t], ddof=1)) if np.var(m[t], ddof=1) > 0 else 1.0
        beta[i] = shrink * 1.0 + (1.0 - shrink) * raw
        resid[i] = float(np.var(R[t, i] - beta[i] * m[t], ddof=1))
    resid = np.where(np.isnan(resid), np.nanmedian(resid) if not np.isnan(resid).all() else var_m, resid)
    cov = var_m * np.outer(beta, beta) + np.diag(np.maximum(resid, 1e-12))
    return cov, dict(model="single index, betas shrunk toward 1", n_periods=int(R.shape[0]),
                     n_thin_history=int(thin), shrink=shrink, units="variance per reporting period")


def run_history(parent_reports: Mapping[date, object], fund_reports: Mapping[date, object], *,
                min_cov_periods: int = 4, **run_kwargs) -> list[PeriodResult]:
    """``run_period`` for every pair of consecutive dates on which both funds reported.

    The covariance for a period uses only returns from periods that ended on or before its
    start date, so nothing from the future enters. Until ``min_cov_periods`` periods have
    accumulated, the benchmark-aware reference is skipped."""
    dates = sorted(set(parent_reports) & set(fund_reports))
    build_kw = {k: run_kwargs.pop(k) for k in ("source", "unknown_returns", "sectors", "class_id") if k in run_kwargs}
    out: list[PeriodResult] = []
    past: list[pd.Series] = []
    for t, t1 in zip(dates[:-1], dates[1:]):
        case = build_period_case(parent_reports[t], parent_reports[t1], fund_reports[t], fund_reports[t1], **build_kw)
        est = single_index_cov(pd.DataFrame(past), case.keys, case.benchmark, min_cov_periods) if past else None
        res = run_period(case, cov=None if est is None else est[0], **run_kwargs)
        if est is not None:
            res.notes.append(f"covariance for D3: {est[1]['model']}, {est[1]['n_periods']} past periods, "
                             f"{est[1]['n_thin_history']} securities with too little history (beta 1 assumed)")
        out.append(res)
        past.append(pd.Series(case.asset_returns, index=case.keys, name=str(t1)))
    return out


def history_table(results: Sequence[PeriodResult], label_contains: str = "capped benchmark") -> pd.DataFrame:
    """One row per period for one reference distribution: the realised frozen return, the
    rule-conditioned return range and the realised portfolio percentile."""
    rows = []
    for res in results:
        row = next((r for r in res.rows if label_contains in r["label"]), None)
        if row is None:
            continue
        rows.append(dict(period_start=str(res.case.start), period_end=str(res.case.end), k=res.case.k,
                         n_universe=res.case.universe.n, realised_frozen_return=res.realised["frozen_return"],
                         benchmark_proxy_return=res.realised["benchmark_proxy_return"],
                         nav_return=res.realised["nav_return"], reference=row["label"], median=row["median"],
                         p05=row["p05"], p95=row["p95"],
                         within_mandate_return_difference=row["within_mandate_return_difference"],
                         realised_portfolio_percentile=row["percentile"], percentile_se=row["percentile_se"],
                         fund_weight_in_universe=res.case.coverage["fund_weight_in_universe"]))
    return pd.DataFrame(rows)
