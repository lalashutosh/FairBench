"""Write REPRODUCIBILITY.md for the real-data layer from the files the other scripts produce.

    .venv/bin/python scripts/build_example_dataset.py
    .venv/bin/python scripts/real_fund_attribution.py --example
    .venv/bin/python scripts/validate_reference.py
    .venv/bin/python scripts/reproducibility_report.py

Every number in the report is read from results/ or examples/, never typed in by hand.
"""
from __future__ import annotations

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


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    def cell(c, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return fmt[c](v) if c in fmt else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(c, v) for c, v in zip(cols, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    ex = ROOT / "examples" / "real_data_example"
    meta = json.loads((ex / "meta.json").read_text())
    res = json.loads((ROOT / "results" / "real_fund_example.json").read_text())
    val = pd.read_csv(ROOT / "results" / "reference_validation.csv")

    for rel, digest in meta["sha256"].items():  # the report is only valid for these exact inputs
        if hashlib.sha256((ex / rel).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{rel} does not match meta.json; rebuild the example first")

    pct = lambda v: f"{100 * v:+.2f}%"
    rows, errs = [], []
    for period, truth in zip(res["periods"], meta["periods"]):
        derived = period["realised"]["frozen_return"]
        errs.append(abs(derived - truth["fund_frozen_price_return"]))
        row = {"period end": period["period_end"], "universe": period["coverage"]["n_universe"],
               "unknown returns": period["coverage"]["n_unknown_return"],
               "splits adjusted": period["coverage"]["n_split_adjusted"],
               "frozen return (derived)": pct(derived), "frozen return (planted)": pct(truth["fund_frozen_price_return"])}
        for r in period["reference"]:
            short = r["label"].split(",")[0].replace(" uniform subsets", "").replace(" uniform weight grid", "") \
                + (" equal" if "equal" in r["label"] else " bench" if "capped" in r["label"] else
                   " " + r["label"].split("tau = ")[1] if "tau" in r["label"] else "")
            row[f"pct {short}"] = f"{r['percentile']:.1f}"
        rows.append(row)
    periods = pd.DataFrame(rows).fillna("")
    violations = sum(r["n_violations"] for p in res["periods"] for r in p["reference"])

    f4 = lambda v: f"{v:.4g}"
    vt = val[["distribution", "sampler", "kind", "n_samples", "feasibility_rate", "tv", "tv_noise_floor", "kl",
              "kl_smoothed", "marginal_error_max", "return_dist_ks", "percentile_error", "runtime_s", "qubits",
              "depth", "shots"]].rename(columns=lambda c: c.replace("_", " "))
    vt_fmt = {c: f4 for c in vt.columns if vt[c].dtype.kind == "f"}
    for c in ("n samples", "qubits", "depth", "shots"):
        vt_fmt[c] = lambda v: f"{int(v):,}"

    text = f"""# Reproducibility report: real-data layer

_Generated {date.today()} by `scripts/reproducibility_report.py` from commit `{_git('rev-parse', '--short', 'HEAD')}`
(branch `{_git('branch', '--show-current')}`). Do not edit by hand; re-run the script._

## What was and was not run

| | Status |
|---|---|
| Synthetic worked example, end to end (parse, store, universe, returns, references, ranking) | **Run**, results below |
| Samplers against exact distributions on small universes, classical and quantum (simulator) | **Run**, results below |
| Live download from SEC EDGAR | **Not run.** The client needs `FAIRBENCH_SEC_USER_AGENT`; nobody has set it. The code path is tested offline against a fake transport. |
| Any real fund | **Not run.** No real holdings have been ingested. Every number here is synthetic. |
| Live model call for mandate extraction | **Not run.** Tested with a fake client only. |
| Quantum hardware | **Not run.** Simulator only. |

The example data are fictional and labelled so in every file. They were generated with
known values planted, so the pipeline's output can be checked against the truth.

## Environment

Python {platform.python_version()} on {platform.system()} {platform.machine()};
numpy {version('numpy')}, pandas {version('pandas')}, scipy {version('scipy')}, qiskit {version('qiskit')},
qiskit-aer {version('qiskit-aer')}.

## Commands

```bash
.venv/bin/python scripts/build_example_dataset.py
```

```bash
.venv/bin/python scripts/real_fund_attribution.py --example
```

```bash
.venv/bin/python scripts/validate_reference.py
```

```bash
.venv/bin/python scripts/reproducibility_report.py
```

Seeds: example data {meta['seed']}; reference samples {res['seed']} ({res['n_samples']:,} portfolios per
distribution per period); validation 0. Inputs: {len(meta['sha256'])} synthetic N-PORT documents under
`examples/real_data_example/nport/`, SHA-256 of each in `meta.json` and checked by this script.

## Worked example (synthetic)

A fictional ESG fund with {res['periods'][0]['k']} holdings against a fictional {res['periods'][0]['coverage']['n_universe_before']}-company
index fund, {len(res['periods'])} quarterly periods, {len(res['excluded_keys'])} companies on an exclusion list.
Returns are price returns derived from the index fund's own filings (value / shares).

{_md(periods)}

- Largest gap between the derived and the planted frozen-holdings return: {max(errs):.2e}. Both are computed from
  the same synthetic holdings, so this checks the arithmetic and the handling of the split and the index
  deletion (either would move the number if mishandled). It says nothing about the quality of real data.
- The planted 2-for-1 split was adjusted in the period it happened ({sum(p['coverage']['n_split_adjusted'] for p in res['periods'])} adjustment in total);
  the company that left the index has no derived return afterwards and was dropped from the universe for that
  period ({sum(p['coverage']['n_unknown_return'] for p in res['periods'])} unknown return in total), not filled in.
- Rule violations found by the independent re-check across all reference samples: {violations}.
- "pct" columns are the realised portfolio percentile under each reference distribution. They differ because the
  distributions are different objects (see `fairbench/portfolio/reference.py`). The benchmark-aware reference (D3)
  appears only once four earlier periods exist to estimate a covariance from, so that nothing from the future is used.
- A percentile is not evidence of skill. Here the fund's holdings were chosen by a random preference vector.

Per-period definitions, samplers, seeds, acceptance rates and estimated numbers of feasible portfolios are in
`results/real_fund_example.json`.

## Samplers against exact distributions

Small universes where every feasible portfolio can be listed. "tv" is total variation distance to the exact
distribution; a value near "tv noise floor" is sampling noise, a value well above it is bias. "kl" is blank when
it is infinite (some feasible portfolio was never drawn).

{_md(vt, vt_fmt)}

Reading it:
- Rejection sampling, the Dicke circuit and amplitude amplification sit at the noise floor: they draw the stated
  distribution. Simulated annealing on the same encoding does not ({f4(float(val.loc[val['sampler'].str.contains('annealing'), 'tv'].iloc[0]))} against a floor of {f4(float(val.loc[val['sampler'].str.contains('annealing'), 'tv_noise_floor'].iloc[0]))}); it is a baseline for the energy function, not a uniform sampler.
- The circuit depth of the amplification row comes from synthesising an oracle from the enumerated answer. It
  shows the output distribution is right; it is not a cost estimate, and nothing here is a claim of quantum advantage.
- Runtimes are wall-clock on this machine for these tiny instances and are not a speed comparison.

## What is observed, derived, assumed, missing

| Label | In this layer |
|---|---|
| **Disclosed** (read from a filing as filed) | Holdings: issuer, CUSIP, ISIN, shares, value, percent of net assets, asset type, country, currency. Fund net assets. Monthly total returns per share class. Monthly flows. Filing and report dates. |
| **Derived** (computed, method stated) | Security keys from identifiers. Equity-sleeve weights. Benchmark weights (from the index fund's filing). Security price returns (value / shares at two dates). Split adjustments. The fund's frozen-holdings return. Position cap and weight-grid bounds (from the fund's own largest and smallest positions). The single-index covariance. Observed holding changes. |
| **Assumed** (a choice, reported with the result) | The reference distribution itself and its weighting policy. The tilt scale tau in the benchmark-aware reference. Dropping, or filling, securities with unknown returns. An index fund as proxy for the index. Any exclusion list not printed in the mandate. Promotion of a soft rule to hard by a reviewer. |
| **Missing** | Dividends (price returns only, unless a total-return source is supplied). Returns of securities that left the reference fund within a period. Share changes under 8% that are really stock dividends. Sector classification. Trades inside a quarter. |
| **Not reproducible from public data** | Any rule that depends on a vendor ESG rating, a vendor controversy flag, a provider's emissions data or a manager's internal score. Such rules are stored with their evidence, marked unobservable, and never enforced. |

Holdings changes between snapshots are recorded as observations (`observed_position_change`), never as decisions:
they mix trading, price moves, investor flows and corporate actions, and neither the timing nor the reason is public.
"""
    (ROOT / "REPRODUCIBILITY.md").write_text(text)
    print(f"wrote REPRODUCIBILITY.md ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
