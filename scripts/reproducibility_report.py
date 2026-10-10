"""Write docs/REAL_FUND_RESULTS.md: what the real-fund run found, in plain words, with its checks.

    .venv/bin/python scripts/real_fund_attribution.py --example
    .venv/bin/python scripts/validate_reference.py
    .venv/bin/python scripts/reproducibility_report.py [--name parnassus_core_equity]

Every number in the file is read from results/ or examples/, never typed in by hand. The
real-fund sections need results/real_fund_<name>*; the checks need the synthetic example
and the validation table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RULED = "k names obeying the exclusions"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}

    def cell(c, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return fmt[c](v) if c in fmt else str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(c, v) for c, v in zip(cols, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _grow(returns) -> float:
    return float(100 * (np.prod(1 + np.asarray(returns, dtype=float)) - 1))


def real_sections(name: str) -> str:
    res = ROOT / "results"
    run = json.loads((res / f"real_fund_{name}.json").read_text())
    summ = json.loads((res / f"real_fund_{name}_summary.json").read_text())
    hold = pd.read_csv(res / f"real_fund_{name}_hold.csv")
    excl_file = res / f"real_fund_{name}_exclusions.csv"
    excl = pd.read_csv(excl_file) if excl_file.exists() else None
    periods = run["periods"]
    h = hold[(hold["reference"] == RULED) & (hold["weights"] == "benchmark_capped")].reset_index(drop=True)
    free = hold[(hold["reference"] == "any k names") & (hold["weights"] == "benchmark_capped")].reset_index(drop=True)
    by = {(r["reference"], r["weights"]): r for r in summ["by_reference"]}
    cap, eq = by[(RULED, "benchmark_capped")], by[(RULED, "equal")]
    fund = _grow([p["realised"]["frozen_return"] for p in periods])
    index = _grow([p["realised"]["benchmark_proxy_return"] for p in periods])
    typ, typ_free = _grow(h["median_1q"]), _grow(free["median_1q"])
    k = int(round(h["k"].median()))
    se = cap["se_if_random_1q"]
    gap = np.array([p["realised"]["nav_minus_frozen"] for p in periods]) * 100
    unk = [p["coverage"]["n_unknown_return"] for p in periods]
    inu = [100 * p["coverage"]["fund_weight_in_universe"] for p in periods]
    acc = min(r["acceptance_rate"] for p in periods for r in p["reference"] if r["label"].startswith("D1"))
    valid = ("every random draw is already a valid portfolio," if acc > 0.995
             else f"{100 * acc:.0f}% or more of random draws are already valid portfolios,")
    violations = sum(r["n_violations"] for p in periods for r in p["reference"]) + int(hold["n_violations"].sum())
    start, end = periods[0]["period_start"], periods[-1]["period_end"]
    reasons = "" if excl is None else "; ".join(f"{n} {why.split(' (')[0]}" for why, n in excl.groupby("reason").size().items())

    qfile = res / f"real_fund_{name}_quantum.json"
    if qfile.exists():
        q = json.loads(qfile.read_text())
        sl, rs = q["sleeve"], q["resources_last_period"]
        scale = "\n".join(f"| ±{r['eps_points']:g} | {r['quantum_queries']:,.0f} | {r['classical_samples']:,.0f} | {r['ratio']:.1f}× |"
                          for r in q["scaling_last_period"])
        quantum = f"""## The same quarters through the quantum estimator

The quantum core (`QUANTUM_CORE.md`) was given exactly the same inputs: the universe, the rules, the fund's return.
It answers the equal-weight version of the question, the one its circuit can express. Everything in this section is a
noiseless **simulation** and counts oracle queries, not time. One query is one check of one portfolio.

- **It gets the same answer.** On the {int(np.median([r['n'] for r in sl['rows']]))} largest index stocks, where every portfolio can be listed and the
  exact rank is known, the simulated quantum estimate was off by {sl['mean_abs_error_quantum']:.2f} points on average over {sl['n_periods']} quarters;
  classical sampling with the same number of queries was off by {sl['mean_abs_error_classical_same_queries']:.2f}.
- **It needs fewer queries, and the gap widens with precision.** To pin the fund's rank to ±{q['eps_points']:g} point on the full
  universe: a median of {q['median_quantum_queries']:,.0f} quantum queries against {q['median_classical_samples']:,.0f} classical samples
  ({q['median_query_ratio']:.1f}× fewer). For the latest quarter:

| Precision (points) | Quantum queries | Classical samples | Ratio |
|---|---|---|---|
{scale}

- **Under these rules that is the only gain.** {valid.capitalize()} so there is no rarity to exploit. A mandate
  with caps and ESG or carbon averages leaves far fewer valid portfolios, and that is where the second gain appears
  (classical cost grows like 1/share, quantum like 1/√share).
- **It cannot run on a machine that exists.** The full-universe circuit needs about {rs['logical_qubits']:,} error-corrected qubits
  and {rs['t_gates_per_step']:.1e} T gates per step. Each quantum query would also be far slower than a laptop's check, so
  this is fewer queries, not less time. What may and may not be claimed: `QUANTUM_CORE.md` section 5.
"""
    else:
        quantum = f"""## What this run says about quantum

Under these rules {valid} so there is nothing for a quantum sampler to speed up through rarity. Run
`scripts/real_fund_quantum.py` to put the quantum estimator on the same quarters. What may and may not be claimed:
`QUANTUM_CORE.md` section 5.
"""

    pct = lambda v: f"{100 * v:+.1f}%"
    table = pd.DataFrame({
        "quarter ending": [p["period_end"] for p in periods],
        "fund": [pct(v) for v in h["fund_return_1q"]],
        "index fund": [pct(p["realised"]["benchmark_proxy_return"]) for p in periods],
        "typical portfolio": [pct(v) for v in h["median_1q"]],
        "rank that quarter": [f"{v:.0f}" for v in h["percentile_1q"]],
        f"rank if held to {end[:7]}": [f"{v:.0f}" for v in h["percentile_to_end"]],
        "exclusions' effect": [f"{100 * v:+.2f}%" for v in h["rule_conditioned_shift_1q"]],
    })
    return f"""## In plain words

**The question.** When an ESG fund lags, is it the rules or the manager?

**The data.** All of it is free and public, from the US regulator (the SEC), which makes every fund publish what it
owns four times a year. No finance API and no paid data were used.

- {summ['fund']}: its actual holdings at {len(periods) + 1} quarter-ends, {start} to {end} (about {k} stocks each).
- {summ['universe_proxy']}: every company in the index on the same dates. This is the pool the random portfolios are
  drawn from, and it stands in for "the market".
- Prices: worked out from the filings themselves (value divided by number of shares). No dividends.

**The method.** For each quarter, take the fund's real portfolio and build {run['n_samples']:,} random portfolios of
the same size from companies that pass the rules. See where the fund ranks among them: 0 is worst, 100 is best, 50 is
the typical portfolio that follows the same rules.

**What came out.**

| | Price growth, {start[:7]} to {end[:7]} |
|---|---|
| {summ['universe_proxy']} | {index:+.0f}% |
| {summ['fund']} | {fund:+.0f}% |
| Typical {k}-stock portfolio under the same rules | {typ:+.0f}% |

- **The rules cost about {abs(typ - typ_free):.0f} points** over the whole period ({typ:+.0f}% with the exclusions, {typ_free:+.0f}% without).
- **Quarter by quarter the fund ranked {cap['mean_percentile_1q']:.0f} out of 100 on average** ({eq['mean_percentile_1q']:.0f} if the random
  portfolios are equally weighted). Pure luck would average 50, give or take {se:.0f}, so anything between about
  {50 - 2 * se:.0f} and {50 + 2 * se:.0f} cannot be told apart from luck. It beat the typical portfolio in
  {100 * cap['share_of_quarters_above_median']:.0f}% of quarters.
- **Each quarter's portfolio held unchanged until {end[:7]}: average rank {cap['mean_percentile_to_end']:.0f}**
  ({eq['mean_percentile_to_end']:.0f} equally weighted). Read this one with care: the windows overlap almost completely, so the
  early portfolios dominate and there is no "give or take" for it.
- **Why the fund trailed the index by {index - fund:.0f} points:** mostly because any {k}-stock portfolio tends to trail
  an index that a few very large companies drove. The fund's picks did {fund - typ:.0f} points better than the typical
  portfolio its rules allowed.

![{summ['fund']} against the index and the typical rule-abiding portfolio](../results/real_fund_{name}.png)

**What must be said with it.**

- **Proxy rules.** The rules applied are the number of holdings and an exclusion list of {summ['n_excluded_keys']} companies
  built from the SEC's own industry codes ({reasons}). Most defence, casino and drinks companies are not caught by
  those codes. The fund's real ESG research is private and cannot be reproduced.
- **"The manager's decisions" are quarter-end snapshots.** Trades inside a quarter, and the reasons for any
  trade, are not public.
- **The latest date is {end}.** Holdings become public about two months after each quarter-end.
- **Price returns, no dividends.** One fund. A rank is a description, not proof of skill.

## Quarter by quarter

"Typical portfolio" is the middle one of the random rule-abiding portfolios (benchmark-proportional weights, capped
at the fund's own largest position). "Exclusions' effect" is how much the exclusion list moved that middle portfolio.

{_md(table)}

## How good is the data

- The fund's own reported return (which includes dividends and fees) differs from the return of its frozen holdings
  by {gap.mean():+.2f} points a quarter on average (median {float(np.median(gap)):+.2f}). The two are built from different numbers, so
  this is an independent check that the prices derived from the filings are right.
- Between {min(unk)} and {max(unk)} index companies per quarter have no usable return (they left both index funds, or their
  share count changed in a way no split explains). They are left out for that quarter, never given a made-up return.
- {min(inu):.0f}% to {max(inu):.0f}% of the fund's stock holdings are in the index. The rest are outside the pool the random portfolios
  are drawn from, and the result says so each quarter.
- Every random portfolio is re-checked against the rules after it is drawn: {violations} violations.

{quantum}
## Repeat it

The commands are in `README.md` under "Real funds from public filings". Another fund needs its SEC series id and
an index fund that reports on the same months (`RESEARCH.md`, Part B, lists candidates).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="parnassus_core_equity")
    args = ap.parse_args()
    ex = ROOT / "examples" / "real_data_example"
    meta = json.loads((ex / "meta.json").read_text())
    res = json.loads((ROOT / "results" / "real_fund_example.json").read_text())
    val = pd.read_csv(ROOT / "results" / "reference_validation.csv")
    for rel, digest in meta["sha256"].items():  # the checks are only valid for these exact inputs
        if hashlib.sha256((ex / rel).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{rel} does not match meta.json; rebuild the example first")

    errs = [abs(p["realised"]["frozen_return"] - t["fund_frozen_price_return"])
            for p, t in zip(res["periods"], meta["periods"])]
    f4 = lambda v: f"{v:.4g}"
    vt = val[["distribution", "sampler", "kind", "feasibility_rate", "tv", "tv_noise_floor", "kl", "marginal_error_max",
              "percentile_error", "qubits", "depth"]].rename(columns=lambda c: c.replace("_", " "))
    vt_fmt = {c: f4 for c in vt.columns if vt[c].dtype.kind == "f"}
    for c in ("qubits", "depth"):
        vt_fmt[c] = lambda v: f"{int(v):,}"
    sa = val.loc[val["sampler"].str.contains("annealing")].iloc[0]
    have_real = (ROOT / "results" / f"real_fund_{args.name}_summary.json").exists()
    real = real_sections(args.name) if have_real else "_No real-fund results in results/ yet; see README for how to produce them._\n"

    text = f"""# FairBench on a real fund: results and checks

_Generated {date.today()} by `scripts/reproducibility_report.py` from commit `{_git('rev-parse', '--short', 'HEAD')}`. Do not edit by
hand; re-run the script. Python {platform.python_version()}, numpy {version('numpy')}, pandas {version('pandas')}, scipy {version('scipy')},
qiskit {version('qiskit')}._

{real}
## Checks against known answers

**A made-up example with planted answers** (`examples/real_data_example/`: a fictional {res['periods'][0]['coverage']['n_universe_before']}-company
index fund and a fictional {res['periods'][0]['k']}-stock ESG fund over {len(res['periods'])} quarters, in the same file format as the
real filings). A 2-for-1 share split, a company leaving the index and an exclusion list were planted. The pipeline
recovered the planted returns to within {max(errs):.0e}, adjusted the split
({sum(p['coverage']['n_split_adjusted'] for p in res['periods'])} adjustment), left the departed company without a return instead of
inventing one, and found {sum(r['n_violations'] for p in res['periods'] for r in p['reference'])} rule violations in its random portfolios.

**Samplers against exact answers.** On universes small enough to list every valid portfolio, each sampler's
output is compared with the exact distribution. "tv" is the distance from it; a value near "tv noise floor" is
sampling noise, a value well above is bias. "kl" is blank when some valid portfolio was never drawn.

{_md(vt, vt_fmt)}

Rejection sampling, the Dicke circuit and amplitude amplification sit at the noise floor: they draw the stated
distribution. Simulated annealing does not ({f4(float(sa['tv']))} against a floor of {f4(float(sa['tv_noise_floor']))}). The quantum rows ran on a
simulator on tiny cases, and the amplification circuit's depth comes from an oracle built from the listed answer:
this table shows the outputs are correct, not that anything is faster.

## What is observed, derived, assumed, missing

| | |
|---|---|
| **Disclosed** (read from a filing) | Holdings: company, identifiers, shares, value, share of the fund. Fund size. Monthly returns and flows. Filing and report dates. |
| **Derived** (computed, method stated) | Prices and price returns (value / shares at two dates). Split adjustments. Benchmark weights (from the index fund's filing). The fund's frozen-holdings return. Position cap and weight-grid bounds (from the fund's own positions). The covariance used for the benchmark-aware reference. Observed holding changes. |
| **Assumed** (a choice, reported with the result) | Which reference distribution and weighting. The exclusion list from industry codes. Returns kept when only the share count changed. Dropping securities with no return. An index fund standing in for the index. |
| **Missing** | Dividends. Returns of securities that left both index funds. Sector labels. Trades inside a quarter. |
| **Not reproducible from public data** | Any rule that depends on a vendor ESG rating, a provider's emissions data or a manager's internal score. Stored with its evidence, marked unobservable, never enforced. |
"""
    (ROOT / "docs" / "REAL_FUND_RESULTS.md").write_text(text)
    print(f"wrote docs/REAL_FUND_RESULTS.md ({len(text.splitlines())} lines, {len(text) // 1024} KB)")


if __name__ == "__main__":
    main()
